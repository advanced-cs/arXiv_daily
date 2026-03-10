# 机器人 cs.RO

- **最新发布 162 篇**

- **更新 89 篇**

## 最新发布

#### [new 001] Adaptive Capacity Allocation for Vision Language Action Fine-tuning
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对视觉语言动作模型的适应性微调问题，提出LoRA-SP方法，通过动态调整参数容量提升多任务性能。**

- **链接: [https://arxiv.org/pdf/2603.07404](https://arxiv.org/pdf/2603.07404)**

> **作者:** Donghoon Kim; Minji Bae; Unghui Nam; Gyeonghun Kim; Suyun Lee; Kyuhong Shim; Byonghyo Shim
>
> **备注:** ICRA 2026 (Official Code: this https URL)
>
> **摘要:** Vision language action models (VLAs) are increasingly used for Physical AI, but deploying a pre-trained VLA model to unseen environments, embodiments, or tasks still requires adaptation. Parameter-efficient fine-tuning (PEFT), especially LoRA, is common for VLA policies, yet the exposed capacity knob, the rank, does not transfer uniformly: robotics transfer exhibits a higher and task-varying intrinsic rank than language fine-tuning. Small ranks suffice for LLMs (e.g., $r \in \{4, 8\}$), while spectral analyses indicate VLAs may require much larger ranks (e.g., $r \approx 128$) or near-full rank, a mismatch that worsens in multi-task settings. We present LoRA-SP (Select-Prune), a rank-adaptive fine-tuning method that replaces fixed-rank updates with input- and layer-wise capacity. LoRA-SP uses an SVD-style parameterization with a small router whose nonnegative scores act as singular values over a shared vector bank. The active set is chosen by an energy target on the cumulative squared scores $E(k) \ge \eta$, providing a direct link to approximation error via our spectral analysis. During training, $\eta$ concentrates energy on a few directions and teaches the router to rely on fewer vectors while preserving accuracy. This yields compact adapters that reduce cross-task interference and improve generalization. On four real-robot manipulation tasks collected on an unseen AgileX PiPER arm, across two VLA backbones ($\pi_0$ and SmolVLA), LoRA-SP matches or exceeds full fine-tuning with far fewer trainable parameters, and improves multi-task success by up to 31.6% over standard LoRA while remaining robust to rank choice.
>
---
#### [new 002] FlowTouch: View-Invariant Visuo-Tactile Prediction
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉触觉预测任务，旨在解决触觉信息无法在规划阶段获取的问题。通过引入FlowTouch模型，利用3D网格实现视图不变的触觉预测。**

- **链接: [https://arxiv.org/pdf/2603.08255](https://arxiv.org/pdf/2603.08255)**

> **作者:** Seongjin Bien; Carlo Kneissl; Tobias Jülg; Frank Fundel; Thomas Ressler-Antal; Florian Walter; Björn Ommer; Gitta Kutyniok; Wolfram Burgard
>
> **摘要:** Tactile sensation is essential for contact-rich manipulation tasks. It provides direct feedback on object geometry, surface properties, and interaction forces, enhancing perception and enabling fine-grained control. An inherent limitation of tactile sensors is that readings are available only when an object is touched. This precludes their use during planning and the initial execution phase of a task. Predicting tactile information from visual information can bridge this gap. A common approach is to learn a direct mapping from camera images to the output of vision-based tactile sensors. However, the resulting model will depend strongly on the specific setup and on how well the camera can capture the area where an object is touched. In this work, we introduce FlowTouch, a novel model for view-invariant visuo-tactile prediction. Our key idea is to use an object's local 3D mesh to encode rich information for predicting tactile patterns while abstracting away from scene-dependent details. FlowTouch integrates scene reconstruction and Flow Matching-based models for image generation. Our results show that FlowTouch is able to bridge the sim-to-real gap and generalize to new sensor instances. We further show that the resulting tactile images can be used for downstream grasp stability prediction. Our code, datasets and videos are available at this https URL
>
---
#### [new 003] Learning From Failures: Efficient Reinforcement Learning Control with Episodic Memory
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决机器人训练中因失败导致的样本效率低问题。通过引入FEMA机制，存储并利用失败经验，提升学习效果。**

- **链接: [https://arxiv.org/pdf/2603.07110](https://arxiv.org/pdf/2603.07110)**

> **作者:** Chenyang Miao
>
> **摘要:** Reinforcement learning has achieved remarkable success in robot learning. However, under challenging exploration and contact-rich dynamics, early-stage training is frequently dominated by premature terminations such as collisions and falls. As a result, learning is overwhelmed by short-horizon, low-return trajectories, which hinder convergence and limit long-horizon exploration. To alleviate this issue, we propose a technique called Failure Episodic Memory Alert (FEMA). FEMA explicitly stores short-horizon failure experiences through an episodic memory module. During interactions, it retrieves similar failure experiences and prevents the robot from recurrently relapsing into unstable states, guiding the policy toward long-horizon trajectories with greater long-term value. FEMA can be combined easily with model-free reinforcement learning algorithms, and yields a substantial sample-efficiency improvement of 33.11% on MuJoCo tasks across several classical RL algorithms. Furthermore, integrating FEMA into a parallelized PPO training pipeline demonstrates its effectiveness on a real-world bipedal robot task.
>
---
#### [new 004] Reasoning Knowledge-Gap in Drone Planning via LLM-based Active Elicitation
- **分类: cs.RO**

- **简介: 该论文属于无人机协同规划任务，旨在解决非专家操作员在环境不确定性下的低效控制问题。通过主动信息获取机制，减少人机交互频率，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.07824](https://arxiv.org/pdf/2603.07824)**

> **作者:** Zeyu Fang; Beomyeol Yu; Cheng Liu; Zeyuan Yang; Rongqian Chen; Yuxin Lin; Mahdi Imani; Tian Lan
>
> **摘要:** Human-AI joint planning in Unmanned Aerial Vehicles (UAVs) typically relies on control handover when facing environmental uncertainties, which is often inefficient and cognitively demanding for non-expert operators. To address this, we propose a novel framework that shifts the collaboration paradigm from control takeover to active information elicitation. We introduce the Minimal Information Neuro-Symbolic Tree (MINT), a reasoning mechanism that explicitly structures knowledge gaps regarding obstacles and goals into a queryable format. By leveraging large language models, our system formulates optimal binary queries to resolve specific ambiguities with minimal human interaction. We demonstrate the efficacy of this approach through a comprehensive workflow integrating a vision-language model for perception, voice interfaces, and a low-level UAV control module in both high-fidelity NVIDIA Isaac simulations and real-world deployments. Experimental results show that our method achieves a significant improvement in the success rate for complex search-and-rescue tasks while significantly reducing the frequency of human interaction compared to exhaustive querying baselines.
>
---
#### [new 005] Dual-Horizon Hybrid Internal Model for Low-Gravity Quadrupedal Jumping with Hardware-in-the-Loop Validation
- **分类: cs.RO**

- **简介: 该论文属于机器人跳跃运动控制任务，旨在解决低重力环境下四足机器人连续稳定跳跃的问题。通过双时域模型和硬件在环验证平台实现连续跳跃。**

- **链接: [https://arxiv.org/pdf/2603.07999](https://arxiv.org/pdf/2603.07999)**

> **作者:** Haozhe Xu; Yifei Zhao; Wenhao Feng; Zhipeng Wang; Hongrui Sang; Cheng Cheng; Xiuxian Li; Zhen Yin; Bin He
>
> **摘要:** Locomotion under reduced gravity is commonly realized through jumping, yet continuous pronking in lunar gravity remains challenging due to prolonged flight phases and sparse ground contact. The extended aerial duration increases landing impact sensitivity and makes stable attitude regulation over rough planetary terrain difficult. Existing approaches primarily address single jumps on flat surfaces and lack both continuous-terrain solutions and realistic hardware validation. This work presents a Dual-Horizon Hybrid Internal Model for continuous quadrupedal jumping under lunar gravity using proprioceptive sensing only. Two temporal encoders capture complementary time scales: a short-horizon branch models rapid vertical dynamics with explicit vertical velocity estimation, while a long-horizon branch models horizontal motion trends and center-of-mass height evolution across the jump cycle. The fused representation enables stable and continuous jumping under extended aerial phases characteristic of lunar gravity. To provide hardware-in-the-loop validation, we develop the MATRIX (Mixed-reality Adaptive Testbed for Robotic Integrated eXploration) platform, a digital-twin-driven system that offloads gravity through a pulley-counterweight mechanism and maps Unreal Engine lunar terrain to a motion platform and treadmill in real time. Using MATRIX, we demonstrate continuous jumping of a quadruped robot under lunar-gravity emulation across cratered lunar-like terrain.
>
---
#### [new 006] Seed2Scale: A Self-Evolving Data Engine for Embodied AI via Small to Large Model Synergy and Multimodal Evaluation
- **分类: cs.RO**

- **简介: 该论文提出Seed2Scale，解决Embodied AI数据生成中的探索限制与模型崩溃问题。通过小模型收集、大模型评估和目标模型学习的协同，提升数据质量与模型性能。**

- **链接: [https://arxiv.org/pdf/2603.08260](https://arxiv.org/pdf/2603.08260)**

> **作者:** Cong Tai; Zhaoyu Zheng; Haixu Long; Hansheng Wu; Zhengbin Long; Haodong Xiang; Rong Shi; Zhuo Cui; Shizhuang Zhang; Gang Qiu; He Wang; Ruifeng Li; Biao Liu; Zhenzhe Sun; Tao Shen
>
> **摘要:** Existing data generation methods suffer from exploration limits, embodiment gaps, and low signal-to-noise ratios, leading to performance degradation during self-iteration. To address these challenges, we propose Seed2Scale, a self-evolving data engine that overcomes the data bottleneck through a heterogeneous synergy of "small-model collection, large-model evaluation, and target-model learning". Starting with as few as four seed demonstrations, the engine employs the lightweight Vision-Language-Action model, SuperTiny, as a dedicated collector, leveraging its strong inductive bias for robust exploration in parallel environments. Concurrently, a pre-trained Vision-Language Model is integrated as a Verifer to autonomously perform success/failure judgment and quality scoring for the massive generated trajectories. Seed2Scale effectively mitigates model collapse, ensuring the stability of the self-evolution process. Experimental results demonstrate that Seed2Scale exhibits signifcant scaling potential: as iterations progress, the success rate of the target model shows a robust upward trend, achieving a performance improvement of 131.2%. Furthermore, Seed2Scale signifcantly outperforms existing data augmentation methods, providing a scalable and cost-effective pathway for the large-scale development of Generalist Embodied AI. Project page: this https URL
>
---
#### [new 007] The Talking Robot: Distortion-Robust Acoustic Models for Robot-Robot Communication
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出Artoo系统，用于机器人间鲁棒的声学通信。解决机器人语音传输中的失真问题，通过端到端神经网络优化，提升解码准确率。**

- **链接: [https://arxiv.org/pdf/2603.07072](https://arxiv.org/pdf/2603.07072)**

> **作者:** Hanlong Li; Karishma Kamalahasan; Jiahui Li; Kazuhiro Nakadai; Shreyas Kousik
>
> **摘要:** We present Artoo, a learned acoustic communication system for robots that replaces hand-designed signal processing with end-to-end co-trained neural networks. Our system pairs a lightweight text-to-speech (TTS) transmitter (1.18M parameters) with a conformer-based automatic speech recognition (ASR) receiver (938K parameters), jointly optimized through a differentiable channel. Unlike human speech, robot-to-robot communication is paralinguistics-free: the system need not preserve timbre, prosody, or naturalness, only maximize decoding accuracy under channel distortion. Through a three-phase co-training curriculum, the TTS transmitter learns to produce distortion-robust acoustic encodings that surpass the baseline under noise, achieving 8.3% CER at 0 dB SNR. The entire system requires only 2.1M parameters (8.4 MB) and runs in under 13 ms end-to-end on a CPU, making it suitable for deployment on resource-constrained robotic platforms.
>
---
#### [new 008] Aero-Promptness: Drag-Aware Aerodynamic Manipulability for Propeller-driven Vehicles
- **分类: cs.RO; cs.AI; eess.SY; math.OC**

- **简介: 该论文属于多旋翼控制分配任务，解决冗余系统中考虑气动阻力的操控性问题。通过构建几何框架DAAM，优化推力分配以提升控制性能。**

- **链接: [https://arxiv.org/pdf/2603.07998](https://arxiv.org/pdf/2603.07998)**

> **作者:** Antonio Franchi
>
> **摘要:** This work introduces the Drag-Aware Aerodynamic Manipulability (DAAM), a geometric framework for control allocation in redundant multirotors. By equipping the propeller spin-rate space with a Riemannian metric based on the remaining symmetric acceleration capacity of each motor, the formulation explicitly accounts for motor torque limits and aerodynamic drag. Mapping this metric through the nonlinear thrust law to the generalized force space yields a state-dependent manipulability volume. The log-determinant of this volume acts as a natural barrier function, strictly penalizing drag-induced saturation and low-spin thrust loss. Optimizing this volume along the allocation fibers provides a redundancy resolution strategy inherently invariant to arbitrary coordinate scaling in the generalized-force space. Analytically, we prove that the resulting optimal allocations locally form smooth embedded manifolds, and we geometrically characterize the global jump discontinuities that inevitably arise from physical actuator limits and spin-rate sign transitions.
>
---
#### [new 009] Adaptive Entropy-Driven Sensor Selection in a Camera-LiDAR Particle Filter for Single-Vessel Tracking
- **分类: cs.RO; cs.LG; eess.SP; eess.SY; physics.data-an**

- **简介: 该论文属于目标跟踪任务，解决海岸平台单船跟踪问题。针对相机和LiDAR各自的性能缺陷，提出自适应传感器选择方法，提升跟踪精度与连续性。**

- **链接: [https://arxiv.org/pdf/2603.08457](https://arxiv.org/pdf/2603.08457)**

> **作者:** Andrei Starodubov; Yaqub Aris Prabowo; Andreas Hadjipieris; Ioannis Kyriakides; Roberto Galeazzi
>
> **备注:** 8 pages, 5 figures, submitted to FUSION 2026 conference proceedings
>
> **摘要:** Robust single-vessel tracking from fixed coastal platforms is hindered by modality-specific degradations: cameras suffer from illumination and visual clutter, while LiDAR performance drops with range and intermittent returns. We present a heterogeneous multi-sensor fusion particle-filter tracker that incorporates an information-gain (entropy-reduction) adaptive sensing policy to select the most informative configuration at each fusion time bin. The approach is validated in a real maritime deployment at the CMMI Smart Marina Testbed (Ayia Napa Marina, Cyprus), using a shore-mounted 3D LiDAR and an elevated fixed camera to track a rigid inflatable boat with onboard GNSS ground truth. We compare LiDAR-only, camera-only, all-sensors, and adaptive configurations. Results show LiDAR dominates near-field accuracy, the camera sustains longer-range coverage when LiDAR becomes unavailable, and the adaptive policy achieves a favorable accuracy-continuity trade-off by switching modalities based on information gain. By avoiding continuous multi-stream processing, the adaptive configuration provides a practical baseline for resilient and resource-aware maritime surveillance.
>
---
#### [new 010] SAIL: Test-Time Scaling for In-Context Imitation Learning with VLM
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人模仿学习任务，解决环境变化下轨迹生成脆弱的问题。提出SAIL框架，通过测试时计算提升轨迹生成成功率。**

- **链接: [https://arxiv.org/pdf/2603.08269](https://arxiv.org/pdf/2603.08269)**

> **作者:** Makoto Sato; Yusuke Iwasawa; Yujin Tang; So Kuroki
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** In-context imitation learning allows robots to acquire skills from demonstrations, yet one-shot trajectory generation remains fragile under environmental variation. We propose SAIL, a framework that reframes robot imitation as an iterative refinement problem capable of scaling with test-time compute. SAIL utilizes Monte Carlo Tree Search, where each node is a complete trajectory and edges correspond to trajectory refinements. The process is guided by three core components: an automated archive of successful trajectories for contextually relevant retrieval, a vision language model-based scoring mechanism for trajectory evaluation, and a step-level feedback that provides trajectory-aligned scores for iterative refinement. Experiments across six diverse manipulation tasks in simulation and real-world validation clearly demonstrate that increasing test-time compute consistently improves success rates, achieving up to 95% on complex tasks. Our results suggest that trajectory-level test-time scaling is a robust path toward more generalizable robotic agents.
>
---
#### [new 011] Unified Structural-Hydrodynamic Modeling of Underwater Underactuated Mechanisms and Soft Robots
- **分类: cs.RO; physics.flu-dyn**

- **简介: 该论文属于水下机器人建模任务，解决 underwater underactuated 和 soft 机器人高维参数识别问题，通过优化框架实现结构与流体动力学参数的统一建模。**

- **链接: [https://arxiv.org/pdf/2603.07939](https://arxiv.org/pdf/2603.07939)**

> **作者:** Chenrui Zhang; Yiyuan Zhang; Yunfei Ye; Junkai Chen; Haozhe Wang; Cecilia Laschi
>
> **备注:** The first two listed authors contributed equally. Yiyuan Zhang is the corresponding author
>
> **摘要:** Underwater robots are widely deployed for ocean exploration and manipulation. Underactuated mechanisms are particularly advantageous in aquatic environments, as reducing actuator count lowers the risk of motor leakage while introducing inherent mechanical compliance. However, accurate modeling of underwater underactuated and soft robotic systems remains challenging because it requires identifying a high-dimensional set of internal structural and external hydrodynamic parameters. In this work, we propose a trajectory-driven global optimization framework for unified structural-hydrodynamic modeling of underwater multibody systems. Inspired by the Covariance Matrix Adaptation Evolution Strategy (CMA-ES), the proposed approach simultaneously identifies coupled internal elastic, damping, and distributed hydrodynamic parameters through trajectory-level matching between simulation and experimental motion. This enables high-fidelity reproduction of both underactuated mechanisms and compliant soft robotic systems in underwater environments. We first validate the framework on a link-by-link underactuated multibody mechanism, demonstrating accurate identification of distributed hydrodynamic coefficients, with a normalized end effector position error below 5% across multiple trajectories, varying initial conditions, and both active-passive and fully passive configurations. The identified modeling strategy is then transferred to a single octopus-inspired soft arm, showing strong real-to-sim consistency without manual retuning. Finally, eight identified arms are assembled into a swimming octopus robot, where the unified parameter set enables realistic whole body behavior without additional parameter calibration. These results demonstrate the scalability and transferability of the proposed structural-hydrodynamic modeling framework across underwater underactuated and soft robotic systems.
>
---
#### [new 012] Receding-Horizon Nullspace Optimization for Actuation-Aware Control Allocation in Omnidirectional UAVs
- **分类: cs.RO**

- **简介: 该论文针对多旋翼无人机的控制分配问题，提出一种考虑执行器动态的滚动时域优化方法，以减少电机指令振荡并提升轨迹跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.06832](https://arxiv.org/pdf/2603.06832)**

> **作者:** Riccardo Pretto; Mahmoud Hamandi; Abdullah Mohamed Ali; Gokhan Alcan; Anthony Tzes; Fares Abu-Dakka
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Fully actuated omnidirectional UAVs enable independent control of forces and torques along all six degrees of freedom, broadening the operational envelope for agile flight and aerial interaction tasks. However, conventional control allocation methods neglect the asymmetric dynamics of the onboard actuators, which can induce oscillatory motor commands and degrade trajectory tracking during dynamic maneuvers. This work proposes a receding-horizon, actuation-aware allocation strategy that explicitly incorporates asymmetric motor dynamics and exploits the redundancy of over-actuated platforms through nullspace optimization. By forward-simulating the closed-loop system over a prediction horizon, the method anticipates actuator-induced oscillations and suppresses them through smooth redistribution of motor commands, while preserving the desired body wrench exactly. The approach is formulated as a constrained optimal control problem solved online via Constrained iterative LQR. Simulation results on the OmniOcta platform demonstrate that the proposed method significantly reduces motor command oscillations compared to a conventional single-step quadratic programming allocator, yielding improved trajectory tracking in both position and orientation.
>
---
#### [new 013] Uncertainty Mitigation and Intent Inference: A Dual-Mode Human-Machine Joint Planning System
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，解决开放环境中不确定性与意图推断问题。提出双模式系统，通过对话减少语义模糊和对象不确定，实时感知人类意图，提升协作效率。**

- **链接: [https://arxiv.org/pdf/2603.07822](https://arxiv.org/pdf/2603.07822)**

> **作者:** Zeyu Fang; Yuxin Lin; Cheng Liu; Beomyeol Yu; Zeyuan Yang; Rongqian Chen; Taeyoung Lee; Mahdi Imani; Tian Lan
>
> **摘要:** Effective human-robot collaboration in open-world environments requires joint planning under uncertain conditions. However, existing approaches often treat humans as passive supervisors, preventing autonomous agents from becoming human-like teammates that can actively model teammate behaviors, reason about knowledge gaps, query, and elicit responses through communication to resolve uncertainties. To address these limitations, we propose a unified human-robot joint planning system designed to tackle dual sources of uncertainty: task-relevant knowledge gaps and latent human intent. Our system operates in two complementary modes. First, an uncertainty-mitigation joint planning module enables two-way conversations to resolve semantic ambiguity and object uncertainty. It utilizes an LLM-assisted active elicitation mechanism and a hypothesis-augmented A^* search, subsequently computing an optimal querying policy via dynamic programming to minimize interaction and verification costs. Second, a real-time intent-aware collaboration module maintains a probabilistic belief over the human's latent task intent via spatial and directional cues, enabling dynamic, coordination-aware task selection for agents without explicit communication. We validate the proposed system in both Gazebo simulations and real-world UAV deployments integrated with a Vision-Language Model (VLM)-based 3D semantic perception pipeline. Experimental results demonstrate that the system significantly cuts the interaction cost by 51.9% in uncertainty-mitigation planning and reduces the task execution time by 25.4% in intent-aware cooperation compared to the baselines.
>
---
#### [new 014] TempoFit: Plug-and-Play Layer-Wise Temporal KV Memory for Long-Horizon Vision-Language-Action Manipulation
- **分类: cs.RO**

- **简介: 该论文提出TempoFit，解决长时域视觉-语言-动作任务中的记忆缺失问题。通过层间时间键值记忆，提升模型长期决策能力，无需额外训练。**

- **链接: [https://arxiv.org/pdf/2603.07647](https://arxiv.org/pdf/2603.07647)**

> **作者:** Jun Sun; Boyu Yang; Jiahao Zhang; Ning Ma; Chencheng Wu; Siqing Zhang; Yiou Huang; Qiufeng Wang; Shan Liang; Yaran Chen
>
> **摘要:** Pretrained Vision-Language-Action (VLA) policies have achieved strong single-step manipulation, but their inference remains largely memoryless, which is brittle in non-Markovian long-horizon settings with occlusion, state aliasing, and subtle post-action changes. Prior approaches inject history either by stacking frames, which scales visual tokens and latency while adding near-duplicate pixels, or by learning additional temporal interfaces that require (re-)training and may break the original single-frame inference graph. We present TempoFit, a training-free temporal retrofit that upgrades frozen VLAs through state-level memory. Our key insight is that prefix attention K/V already form a model-native, content-addressable runtime state; reusing them across timesteps introduces history without new tokens or trainable modules. TempoFit stores layer-wise FIFO prefix K/V at selected intermediate layers, performs parameter-free K-to-K retrieval with Frame-Gap Temporal Bias (FGTB), a fixed recency bias inspired by positional biases in NLP, to keep decisions present-dominant, and injects the retrieved context via pre-attention residual loading with norm-preserving rescaling to avoid distribution shift under frozen weights. On LIBERO-LONG, TempoFit improves strong pretrained backbones by up to +4.0% average success rate while maintaining near-real-time latency, and it transfers consistently to CALVIN and real-robot long-horizon tasks.
>
---
#### [new 015] Tactile Recognition of Both Shapes and Materials with Automatic Feature Optimization-Enabled Meta Learning
- **分类: cs.RO**

- **简介: 该论文属于触觉识别任务，解决数据稀缺和学习效率低的问题。提出AFOP-ML框架，实现快速适应新类别的元学习，提升形状和材质识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.08423](https://arxiv.org/pdf/2603.08423)**

> **作者:** Hongliang Zhao; Wenhui Yang; Yang Chen; Zhuorui Wang; Baiheng Liu; Longhui Qin
>
> **备注:** 7 pages, 7 figures, conference paper accepted by ICRA 2026
>
> **摘要:** Tactile perception is indispensable for robots to implement various manipulations dexterously, especially in contact-rich scenarios. However, alongside the development of deep learning techniques, it meanwhile suffers from training data scarcity and a time-consuming learning process in practical applications since the collection of a large amount of tactile data is costly and sometimes even impossible. Hence, we propose an automatic feature optimization-enabled prototypical network to realize meta-learning, i.e., AFOP-ML framework. As a ``learn to learn" network, it not only adapts to new unseen classes rapidly with few-shot, but also learns how to determine the optimal feature space automatically. Based on the four-channel signals acquired from a tactile finger, both shapes and materials are recognized. On a 36-category benchmark, it outperforms several existing approaches by attaining an accuracy of 96.08% in 5-way-1-shot scenario, where only 1 example is available for training. It still remains 88.7% in the extreme 36-way-1-shot case. The generalization ability is further validated through three groups of experiment involving unseen shapes, materials and force/speed perturbations. More insights are additionally provided by this work for the interpretation of recognition tasks and improved design of tactile sensors.
>
---
#### [new 016] Relating Reinforcement Learning to Dynamic Programming-Based Planning
- **分类: cs.RO**

- **简介: 该论文属于强化学习与动态规划的对比研究，旨在揭示两者在最优规划中的联系。通过分析成本最小化与奖励最大化等条件，探讨算法等价性及参数影响，提出优化真实成本而非依赖人为参数的方法。**

- **链接: [https://arxiv.org/pdf/2603.07844](https://arxiv.org/pdf/2603.07844)**

> **作者:** Filip V. Georgiev; Kalle G. Timperi; Başak Sakçak; Steven M. LaValle
>
> **备注:** 43 pages, 8 figures
>
> **摘要:** This paper bridges some of the gap between optimal planning and reinforcement learning (RL), both of which share roots in dynamic programming applied to sequential decision making or optimal control. Whereas planning typically favors deterministic models, goal termination, and cost minimization, RL tends to favor stochastic models, infinite-horizon discounting, and reward maximization in addition to learning-related parameters such as the learning rate and greediness factor. A derandomized version of RL is developed, analyzed, and implemented to yield performance comparisons with value iteration and Dijkstra's algorithm using simple planning models. Next, mathematical analysis shows: 1) conditions under which cost minimization and reward maximization are equivalent, 2) conditions for equivalence of single-shot goal termination and infinite-horizon episodic learning, and 3) conditions under which discounting causes goal achievement to fail. The paper then advocates for defining and optimizing truecost, rather than inserting arbitrary parameters to guide operations. Performance studies are then extended to the stochastic case, using planning-oriented criteria and comparing value iteration to RL with learning rates and greediness factors.
>
---
#### [new 017] LITHE: Bridging Best-Effort Python and Real-Time C++ for Hot-Swapping Robotic Control Laws on Commodity Linux
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决高阶Python与实时C++控制逻辑无法动态更新的问题。通过LITHE架构实现安全的实时控制与动态演化。**

- **链接: [https://arxiv.org/pdf/2603.07442](https://arxiv.org/pdf/2603.07442)**

> **作者:** He Kai Lim; Tyler R. Clites
>
> **备注:** 8 pages, 5 figures. Submitted to IEEE/RSJ International Conference on Intelligent Robots & Systems (IROS) 2026
>
> **摘要:** Modern robotic systems rely on hierarchical control, where a high-level "Brain" (Python) directs a lower-level "Spine" (C++ real-time controller). Despite its necessity, this hierarchy makes it difficult for the Brain to completely rewrite the Spine's immutable control logic, consequently inhibiting fundamental adaptation for different tasks and environments. Conventional approaches require complex middleware, proprietary hardware, or sacrifice real-time performance. We present LITHE (Linux Isolated Threading for Hierarchical Execution), a lightweight software architecture that collapses the robot control hierarchy onto a commodity single-board computer (Raspberry Pi 4B with pi3hat), while maintaining safe frequency decoupling between the Brain and Spine. LITHE integrates strict CPU isolation (isolcpus), lock-free inter-process communication (IPC), and pipelined execution to meet high-frequency deadlines with minimal jitter. By adding multi-threaded dynamic linking, LITHE enables a Python-based Brain to dynamically evolve the logic of a 1kHz C++ Spine without interruption. We validate "functional real-time" system performance with worst-case execution time (WCET) < 100 $\mu$s and maximum release jitter (MRJ) < 4 $\mu$s under heavy load. We demonstrate a novel application where a large language model (LLM) supervisor performs online system identification to evolve a real-time controller on-the-fly, without interrupting the 1 kHz control loop. In essence, LITHE eliminates the "immutable compiled code" bottleneck for best-effort Brains to synthesize and inject completely new control laws into the real-time Spine. This bridges a critical gap between high-level AI and low-level real-time control to unlock continuous real-time evolution of embodied intelligence in safe, human-in-the-loop systems.
>
---
#### [new 018] GeoLoco: Leveraging 3D Geometric Priors from Visual Foundation Model for Robust RGB-Only Humanoid Locomotion
- **分类: cs.RO**

- **简介: 该论文提出GeoLoco，解决RGB-only人形机器人运动控制问题。利用视觉基础模型的3D几何先验，提升模拟到现实的迁移能力。**

- **链接: [https://arxiv.org/pdf/2603.07624](https://arxiv.org/pdf/2603.07624)**

> **作者:** Yufei Liu; Xieyuanli Chen; Hainan Pan; Chenghao Shi; Yanjie Chen; Kaihong Huang; Zhiwen Zeng; Huimin Lu
>
> **备注:** 8 pages, 6 figures, conference
>
> **摘要:** The prevailing paradigm of perceptive humanoid locomotion relies heavily on active depth sensors. However, this depth-centric approach fundamentally discards the rich semantic and dense appearance cues of the visual world, severing low-level control from the high-level reasoning essential for general embodied intelligence. While monocular RGB offers a ubiquitous, information-dense alternative, end-to-end reinforcement learning from raw 2D pixels suffers from extreme sample inefficiency and catastrophic sim-to-real collapse due to the inherent loss of geometric scale. To break this deadlock, we propose GeoLoco, a purely RGB-driven locomotion framework that conceptualizes monocular images as high-dimensional 3D latent representations by harnessing the powerful geometric priors of a frozen, scale-aware Visual Foundation Model (VFM). Rather than naive feature concatenation, we design a proprioceptive-query multi-head cross-attention mechanism that dynamically attends to task-critical topological features conditioned on the robot's real-time gait phase. Crucially, to prevent the policy from overfitting to superficial textures, we introduce a dual-head auxiliary learning scheme. This explicit regularization forces the high-dimensional latent space to strictly align with the physical terrain geometry, ensuring robust zero-shot sim-to-real transfer. Trained exclusively in simulation, GeoLoco achieves robust zero-shot transfer to the Unitree G1 humanoid and successfully negotiates challenging terrains.
>
---
#### [new 019] RoTri-Diff: A Spatial Robot-Object Triadic Interaction-Guided Diffusion Model for Bimanual Manipulation
- **分类: cs.RO**

- **简介: 该论文属于双臂操作任务，解决双臂与物体间动态几何关系建模不足的问题。提出RoTri-Diff框架，通过空间三元交互约束提升操作稳定性与协调性。**

- **链接: [https://arxiv.org/pdf/2603.07165](https://arxiv.org/pdf/2603.07165)**

> **作者:** Zixuan Chen; Nga Teng Chan; Yiwen Hou; Chenrui Tie; Zixuan Liu; Haonan Chen; Junting Chen; Jieqi Shi; Yang Gao; Jing Huo; Lin Shao
>
> **备注:** ICRA 2026
>
> **摘要:** Bimanual manipulation is a fundamental robotic skill that requires continuous and precise coordination between two arms. While imitation learning (IL) is the dominant paradigm for acquiring this capability, existing approaches, whether robot-centric or object-centric, often overlook the dynamic geometric relationship among the two arms and the manipulated object. This limitation frequently leads to inter-arm collisions, unstable grasps, and degraded performance in complex tasks. To address this, in this paper we explicitly models the Robot-Object Triadic Interaction (RoTri) representation in bimanual systems, by encoding the relative 6D poses between the two arms and the object to capture their spatial triadic relationship and establish continuous triangular geometric constraints. Building on this, we further introduce RoTri-Diff, a diffusion-based imitation learning framework that combines RoTri constraints with robot keyposes and object motion in a hierarchical diffusion process. This enables the generation of stable, coordinated trajectories and robust execution across different modes of bimanual manipulation. Extensive experiments show that our approach outperforms state-of-the-art baselines by 10.2% on 11 representative RLBench2 tasks and achieves stable performance on 4 challenging real-world bimanual tasks. Project website: this https URL.
>
---
#### [new 020] Robotic Foundation Models for Industrial Control: A Comprehensive Survey and Readiness Assessment Framework
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制领域，旨在评估机器人基础模型的工业适用性。通过分析需求与标准，提出评估框架并测试大量模型，发现工业成熟度有限。**

- **链接: [https://arxiv.org/pdf/2603.06749](https://arxiv.org/pdf/2603.06749)**

> **作者:** David Kube; Simon Hadwiger; Tobias Meisen
>
> **摘要:** Robotic foundation models (RFMs) are emerging as a promising route towards flexible, instruction- and demonstration-driven robot control, however, a critical investigation of their industrial applicability is still lacking. This survey gives an extensive overview over the RFM-landscape and analyses, driven by concrete implications, how industrial domains and use cases shape the requirements of RFMs, with particular focus on collaborative robot platforms, heterogeneous sensing and actuation, edge-computing constraints, and safety-critical operation. We synthesise industrial deployment perspectives into eleven interdependent implications and operationalise them into an assessment framework comprising a catalogue of 149 concrete criteria, spanning both model capabilities and ecosystem requirements. Using this framework, we evaluate 324 manipulation-capable RFMs via 48,276 criterion-level decisions obtained via a conservative LLM-assisted evaluation pipeline, validated against expert judgements. The results indicate that industrial maturity is limited and uneven: even the highest-rated models satisfy only a fraction of criteria and typically exhibit narrow implication-specific peaks rather than integrated coverage. We conclude that progress towards industry-grade RFMs depends less on isolated benchmark successes than on systematic incorporation of safety, real-time feasibility, robust perception, interaction, and cost-effective system integration into auditable deployment stacks.
>
---
#### [new 021] Towards Scalable Probabilistic Human Motion Prediction with Gaussian Processes for Safe Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文属于人体运动预测任务，旨在提升人机协作的安全性。通过结构化多任务变分高斯过程框架，实现准确且可解释的不确定性估计，解决实时运动预测与安全避障问题。**

- **链接: [https://arxiv.org/pdf/2603.07096](https://arxiv.org/pdf/2603.07096)**

> **作者:** Jinger Chong; Xiaotong Zhang; Kamal Youcef-Toumi
>
> **备注:** Submitted to IROS 2026
>
> **摘要:** Accurate human motion prediction with well-calibrated uncertainty is critical for safe human-robot collaboration (HRC), where robots must anticipate and react to human movements in real time. We propose a structured multitask variational Gaussian Process (GP) framework for full-body human motion prediction that captures temporal correlations and leverages joint-dimension-level factorization for scalability, while using a continuous 6D rotation representation to preserve kinematic consistency. Evaluated on Human3.6M (H3.6M), our model achieves up to 50 lower kernel density estimate negative log-likelihood (KDE NLL) than strong baselines, a mean continuous ranked probability score (CRPS) of 0.021 m, and deterministic mean angle error (MAE) that is 3-18% higher than competitive deep learning methods. Empirical coverage analysis shows that the fraction of ground-truth outcomes contained within predicted confidence intervals gradually decreases with horizon, remaining conservative for lower-confidence intervals and near-nominal for higher-confidence intervals, with only modest calibration drift at longer horizons. Despite its probabilistic formulation, our model requires only 0.24-0.35 M parameters, roughly eight times fewer than comparable approaches, and exhibits modest inference times, indicating suitability for real-time deployment. Extensive ablation studies further validated the choice of 6D rotation representation and Matern 3/2 + Linear kernel, and guided the selection of the number of inducing points and latent dimensionality. These results demonstrate that scalable GP-based models can deliver competitive accuracy together with reliable and interpretable uncertainty estimates for downstream robotics tasks such as motion planning and collision avoidance.
>
---
#### [new 022] LIPP: Load-Aware Informative Path Planning with Physical Sampling
- **分类: cs.RO**

- **简介: 该论文提出LIPP，解决信息路径规划中负载影响能耗的问题，通过建模负载与能耗的耦合，优化路径和采样策略。**

- **链接: [https://arxiv.org/pdf/2603.06924](https://arxiv.org/pdf/2603.06924)**

> **作者:** Hojune Kim; Guangyao Shi; Gaurav S. Sukhatme
>
> **摘要:** In classical Informative Path Planning (C-IPP), robots are typically modeled as mobile sensors that acquire digital measurements such as images or radiation levels. In this model - since making a measurement leaves the robot's physical state unchanged - traversal costs are determined solely by the path taken. This is a natural assumption for many missions, but does not extend to settings involving physical sample collection, where each collected sample adds mass and increases the energy cost of all subsequent motion. As a result, IPP formulations that ignore this coupling between information gain and load-dependent traversal cost can produce plans that are distance-efficient but energy-suboptimal, collecting fewer samples and less data than the energy budget would permit. In this paper, we introduce Load-aware Informative Path Planning (LIPP ), a generalization of C-IPP that explicitly models this coupling and the resulting order-dependent traversal costs. We formulate LIPP as a Mixed-Integer Quadratic Program (MIQP) that jointly optimizes routing, visitation order, and per-location sampling count under an energy budget. We show that LIPP strictly generalizes C-IPP: as sample unit mass $\lambda \to 0$, the load-dependent energy model reduces exactly to the classical distance budget constraint, recovering C-IPP as a special case. We further derive theoretical bounds on the path-length increase of LIPP relative to C-IPP, characterizing the trade-off for improved energy efficiency. Finally, through extensive simulations across 2000 diverse mission scenarios, we demonstrate that LIPP matches the behavior of C-IPP at zero sample mass and progressively achieves higher uncertainty reduction per unit energy as sample mass increases.
>
---
#### [new 023] Model-Based and Neural-Aided Approaches for Dog Dead Reckoning
- **分类: cs.RO**

- **简介: 该论文属于定位任务，解决惯性传感器累积漂移导致的定位误差问题。提出三种仅使用惯性传感器的定位算法，通过实验验证神经辅助方法优于模型方法。**

- **链接: [https://arxiv.org/pdf/2603.07582](https://arxiv.org/pdf/2603.07582)**

> **作者:** Gal Versano. Itai Savin; Itzik Klein
>
> **摘要:** Modern canine applications span medical and service roles, while robotic legged dogs serve as autonomous platforms for high-risk industrial inspection, disaster response, and search and rescue operations. For both, accurate positioning remains a significant challenge due to the cumulative drift inherent in inertial sensing. To bridge this gap, we propose three algorithms for accurate positioning using only inertial sensors, collectively referred to as dog dead reckoning (DDR). To evaluate our approaches, we designed DogMotion, a wearable unit for canine data recording. Using DogMotion, we recorded a dataset of 13 minutes. Additionally, we utilized a robotic legged dog dataset with a duration of 116 minutes. Across the two distinct datasets we demonstrate that our neural-aided methods consistently outperform model-based approaches, achieving an absolute distance error of less than 10\%. Consequently, we provide a lightweight and low-cost positioning solution for both biological and legged robotic dogs. To support reproducibility, our codebase and associated datasets have been made publicly available.
>
---
#### [new 024] Towards Human-Like Manipulation through RL-Augmented Teleoperation and Mixture-of-Dexterous-Experts VLA
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决高自由度、多技能及多模态感知的复杂操作问题。提出IMCopilot和MoDE-VLA框架，提升灵巧操作性能。**

- **链接: [https://arxiv.org/pdf/2603.08122](https://arxiv.org/pdf/2603.08122)**

> **作者:** Tutian Tang; Xingyu Ji; Wanli Xing; Ce Hao; Wenqiang Xu; Lin Shao; Cewu Lu; Qiaojun Yu; Jiangmiao Pang; Kaifeng Zhang
>
> **备注:** Project Homepage: this https URL
>
> **摘要:** While Vision-Language-Action (VLA) models have demonstrated remarkable success in robotic manipulation, their application has largely been confined to low-degree-of-freedom end-effectors performing simple, vision-guided pick-and-place tasks. Extending these models to human-like, bimanual dexterous manipulation-specifically contact-rich in-hand operations-introduces critical challenges in high-fidelity data acquisition, multi-skill learning, and multimodal sensory fusion. In this paper, we propose an integrated framework to address these bottlenecks, built upon two components. First, we introduce IMCopilot (In-hand Manipulation Copilot), a suite of reinforcement learning-trained atomic skills that plays a dual role: it acts as a shared-autonomy assistant to simplify teleoperation data collection, and it serves as a callable low-level execution primitive for the VLA. Second, we present MoDE-VLA (Mixture-of-Dexterous-Experts VLA), an architecture that seamlessly integrates heterogeneous force and tactile modalities into a pretrained VLA backbone. By utilizing a residual injection mechanism, MoDE-VLA enables contact-aware refinement without degrading the model's pretrained knowledge. We validate our approach on four tasks of escalating complexity, demonstrating doubled success rate improvement over the baseline in dexterous contact-rich tasks.
>
---
#### [new 025] Human-Aware Robot Behaviour in Self-Driving Labs
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于人机协作任务，旨在解决共享实验室中机器人与人类协同效率低的问题。通过AI感知方法实现主动交互，提升自动化实验室效率。**

- **链接: [https://arxiv.org/pdf/2603.08420](https://arxiv.org/pdf/2603.08420)**

> **作者:** Satheeshkumar Veeramani; Anna Kisil; Abigail Bentley; Hatem Fakhruldeen; Gabriella Pizzuto; Andrew I. Cooper
>
> **摘要:** Self-driving laboratories (SDLs) are rapidly transforming research in chemistry and materials science to accelerate new discoveries. Mobile robot chemists (MRCs) play a pivotal role by autonomously navigating the lab to transport samples, effectively connecting synthesis, analysis, and characterisation equipment. The instruments within an SDL are typically designed or retrofitted to be accessed by both human and robotic chemists, ensuring operational flexibility and integration between manual and automated workflows. In many scenarios, human and robotic chemists may need to use the same equipment simultaneously. Currently, MRCs rely on simple LiDAR-based obstruction detection, which forces the robot to passively wait if a human is present. This lack of situational awareness leads to unnecessary delays and inefficient coordination in time-critical automated workflows in human-robot shared labs. To address this, we present an initial study of an embodied, AI-driven perception method that facilitates proactive human-robot interaction in shared-access scenarios. Our method features a hierarchical human intention prediction model that allows the robot to distinguish between preparatory actions (waiting) and transient interactions (accessing the instrument). Our results demonstrate that the proposed approach enhances efficiency by enabling proactive human-robot interaction, streamlining coordination, and potentially increasing the efficiency of autonomous scientific labs.
>
---
#### [new 026] VORL-EXPLORE: A Hybrid Learning Planning Approach to Multi-Robot Exploration in Dynamic Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多机器人探索任务，解决动态环境中路径规划与任务分配的问题。提出VORL-EXPLORE框架，结合学习与规划，提升探索效率与安全性。**

- **链接: [https://arxiv.org/pdf/2603.07973](https://arxiv.org/pdf/2603.07973)**

> **作者:** Ning Liu; Sen Shen; Zheng Li; Sheng Liu; Dongkun Han; Shangke Lyu; Thomas Braunl
>
> **摘要:** Hierarchical multi-robot exploration commonly decouples frontier allocation from local navigation, which can make the system brittle in dense and dynamic environments. Because the allocator lacks direct awareness of execution difficulty, robots may cluster at bottlenecks, trigger oscillatory replanning, and generate redundant coverage. We propose VORL-EXPLORE, a hybrid learning and planning framework that addresses this limitation through execution fidelity, a shared estimate of local navigability that couples task allocation with motion execution. This fidelity signal is incorporated into a fidelity-coupled Voronoi objective with inter-robot repulsion to reduce contention before it emerges. It also drives a risk-aware adaptive arbitration mechanism between global A* guidance and a reactive reinforcement learning policy, balancing long-range efficiency with safe interaction in confined spaces. The framework further supports online self-supervised recalibration of the fidelity model using pseudo-labels derived from recent progress and safety outcomes, enabling adaptation to non-stationary obstacles without manual risk tuning. We evaluate this capability separately in a dedicated severe-traffic ablation. Extensive experiments in randomized grids and a Gazebo factory scenario show high success rates, shorter path length, lower overlap, and robust collision avoidance. The source code will be made publicly available upon acceptance.
>
---
#### [new 027] A Pivot-Based Kirigami Utensil for Hand-Held and Robot-Assisted Feeding
- **分类: cs.RO**

- **简介: 该论文属于辅助进食工具设计任务，旨在解决行动不便者使用传统餐具困难的问题。研究提出一种可手动或机器人使用的剪刀式折纸餐具，提升食物抓取与防洒能力。**

- **链接: [https://arxiv.org/pdf/2603.06716](https://arxiv.org/pdf/2603.06716)**

> **作者:** Keone Leao; Grace Brotherson; Iain Mischel; Sagar Parekh; Dylan P. Losey
>
> **摘要:** Eating is a daily challenge for over 60 million adults with essential tremors and other mobility limitations. For these users, traditional utensils like forks or spoons are difficult to manipulate -- resulting in accidental spills and restricting the types of food that can be consumed. Prior work has developed rigid, hand-held utensils that often fail to secure food, as well as soft, shape-changing utensils made strictly for robot-assisted feeding. To assist a broader range of users, we introduce a re-designed kiri-spoon that can be leveraged as either a hand-held utensil or a robot-mounted attachment. Our key idea -- developed in collaboration with stakeholders -- is a pivot-based design. With this design the kiri-spoon behaves like a pair of pliers: users squeeze the handles to change the shape of the utensil and enclose food morsels. In practice, users can apply this kiri-spoon as either a spoon (that scoops food) or as a fork (that pinches food); when the handles are closed, the utensil wraps around the morsel and prevents it from accidentally falling. We characterize the amount of force required to open or close the kiri-spoon, and show how designers can modify this force through kinematic or material changes. A highlight of our design is its accessibility: the hand-held version consists of just four 3D printed parts that snap together. By adding a servo motor, we can extend this same kinematic structure to robot-assisted feeding. Across our user studies, adults with disabilities and elderly adults with Parkinson's reported that the kiri-spoon better met their needs and provided a more effective means of spill prevention than existing alternatives. See a video of our kiri-spoon here: this https URL
>
---
#### [new 028] Viewpoint-Agnostic Grasp Pipeline using VLM and Partial Observations
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于机器人抓取任务，解决 cluttered 环境中因遮挡导致的抓取难题。通过语言引导和点云补全，提升抓取鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07866](https://arxiv.org/pdf/2603.07866)**

> **作者:** Dilermando Almeida; Juliano Negri; Guilherme Lazzarini; Thiago H. Segreto; Ranulfo Bezerra; Ricardo V. Godoy; Marcelo Becker
>
> **摘要:** Robust grasping in cluttered, unstructured environments remains challenging for mobile legged manipulators due to occlusions that lead to partial observations, unreliable depth estimates, and the need for collision-free, execution-feasible approaches. In this paper we present an end-to-end pipeline for language-guided grasping that bridges open-vocabulary target selection to safe grasp execution on a real robot. Given a natural-language command, the system grounds the target in RGB using open-vocabulary detection and promptable instance segmentation, extracts an object-centric point cloud from RGB-D, and improves geometric reliability under occlusion via back-projected depth compensation and two-stage point cloud completion. We then generate and collision-filter 6-DoF grasp candidates and select an executable grasp using safety-oriented heuristics that account for reachability, approach feasibility, and clearance. We evaluate the method on a quadruped robot with an arm in two cluttered tabletop scenarios, using paired trials against a view-dependent baseline. The proposed approach achieves a 90% overall success rate (9/10) against 30% (3/10) for the baseline, demonstrating substantially improved robustness to occlusions and partial observations in clutter.
>
---
#### [new 029] EquiBim: Learning Symmetry-Equivariant Policy for Bimanual Manipulation
- **分类: cs.RO**

- **简介: 该论文属于双臂机器人操作任务，旨在解决对称性不足导致的行为不一致问题。提出EquiBim框架，强制策略在对称变换下保持一致性，提升性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.08541](https://arxiv.org/pdf/2603.08541)**

> **作者:** Zhiyuan Zhang; Aditya Mohan; Seungho Han; Wan Shou; Dongyi Wang; Yu She
>
> **备注:** Submitted to IROS 2026. 8 pages, 6 figures
>
> **摘要:** Robotic imitation learning has achieved impressive success in learning complex manipulation behaviors from demonstrations. However, many existing robot learning methods do not explicitly account for the physical symmetries of robotic systems, often resulting in asymmetric or inconsistent behaviors under symmetric observations. This limitation is particularly pronounced in dual-arm manipulation, where bilateral symmetry is inherent to both the robot morphology and the structure of many tasks. In this paper, we introduce EquiBim, a symmetry-equivariant policy learning framework for bimanual manipulation that enforces bilateral equivariance between observations and actions during training. Our approach formulates physical symmetry as a group action on both observation and action spaces, and imposes an equivariance constraint on policy predictions under symmetric transformations. The framework is model-agnostic and can be seamlessly integrated into a wide range of imitation learning pipelines with diverse observation modalities and action representations, including point cloud-based and image-based policies, as well as both end-effector-space and joint-space parameterizations. We evaluate EquiBim on RoboTwin, a dual-arm robotic platform with symmetric kinematics, and evaluate it across diverse observation and action configurations in simulation. We further validate the approach on a real-world dual-arm system. Across both simulation and physical experiments, our method consistently improves performance and robustness under distribution shifts. These results suggest that explicitly enforcing physical symmetry provides a simple yet effective inductive bias for bimanual robot learning.
>
---
#### [new 030] VSL-Skin: Individually Addressable Phase-Change Voxel Skin for Variable-Stiffness and Virtual Joints Bridging Soft and Rigid Robots
- **分类: cs.RO**

- **简介: 该论文属于软体机器人领域，解决传统机器人刚柔不足的问题。提出VSL-Skin系统，实现像素级刚度调控与自修复，提升机器人适应性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.06979](https://arxiv.org/pdf/2603.06979)**

> **作者:** Zihan Oliver Zeng; Jiajun An; Preston Luk; Upinder Kaur
>
> **备注:** ICRA 2026
>
> **摘要:** Soft robots are compliant but often cannot support loads or hold their shape, while rigid robots provide structural strength but are less adaptable. Existing variable-stiffness systems usually operate at the scale of whole segments or patches, which limits precise control over stiffness distribution and virtual joint placement. This paper presents the Variable Stiffness Lattice Skin (VSL-Skin), the first system to enable individually addressable voxel-level morphological control with centimeter-scale precision. The system provides three main capabilities: nearly two orders of magnitude stiffness modulation across axial (15-1200 N/mm), shear (45-850 N/mm), bending (8*10^2 - 3*10^4 N/deg), and torsional modes with centimeter-scale spatial control; the first demonstrated 30% axial compression in phase-change systems while maintaining structural integrity; and autonomous component-level self-repair through thermal cycling, which eliminates fatigue accumulation and enables programmable sacrificial joints for predictable failure management. Selective voxel activation creates six canonical virtual joint types with programmable compliance while preserving structural integrity in non-activated regions. The platform incorporates closed-form design models and finite element analysis for predictive synthesis of stiffness patterns and joint placement. Experimental validation demonstrates 30% axial contraction, thermal switching in 75-second cycles, and cut-to-fit integration that preserves addressability after trimming. The row-column architecture enables platform-agnostic deployment across diverse robotic systems without specialized infrastructure. This framework establishes morphological intelligence as an engineerable system property and advances autonomous reconfigurable robotics.
>
---
#### [new 031] PhaForce: Phase-Scheduled Visual-Force Policy Learning with Slow Planning and Fast Correction for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文提出PhaForce，用于高接触操作任务，解决视觉与力反馈融合不足的问题，通过分阶段调度实现高效控制。**

- **链接: [https://arxiv.org/pdf/2603.08342](https://arxiv.org/pdf/2603.08342)**

> **作者:** Mingxin Wang; Zhirun Yue; Renhao Lu; Yizhe Li; Zihan Wang; Guoping Pan; Kangkang Dong; Jun Cheng; Yi Cheng; Houde Liu
>
> **摘要:** Contact-rich manipulation requires not only vision-dominant task semantics but also closed-loop reactions to force/torque (F/T) transients. Yet, generative visuomotor policies are typically constrained to low-frequency updates due to inference latency and action chunking, underutilizing F/T for control-rate feedback. Furthermore, existing force-aware methods often inject force continuously and indiscriminately, lacking an explicit mechanism to schedule when / how much / where to apply force across different task phases. We propose PhaForce, a phase-scheduled visual--force policy that coordinates low-rate chunk-level planning and high-rate residual correction via a unified contact/phase schedule. PhaForce comprises (i) a contact-aware phase predictor (CAP) that estimates contact probability and phase belief, (ii) a Slow diffusion planner that performs dual-gated visual--force fusion with orthogonal residual injection to preserve vision semantics while conditioning on force, and (iii) a Fast corrector that applies control-rate phase-routed residuals in interpretable corrective subspaces for within-chunk micro-adjustments. Across multiple real-robot contact-rich tasks, PhaForce achieves an average success rate of 86% (+40 pp over baselines), while also substantially improving contact quality by regulating interaction forces and exhibiting robust adaptability to OOD geometric shifts.
>
---
#### [new 032] DexKnot: Generalizable Visuomotor Policy Learning for Dexterous Bag-Knotting Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决机器人在未知塑料袋情况下打结的泛化问题。通过结合关键点感知与扩散策略，实现有效抓握与打结。**

- **链接: [https://arxiv.org/pdf/2603.07136](https://arxiv.org/pdf/2603.07136)**

> **作者:** Jiayuan Zhang; Ruihai Wu; Haojun Chen; Yuran Wang; Yifan Zhong; Ceyao Zhang; Yaodong Yang; Yuanpei Chen
>
> **摘要:** Knotting plastic bags is a common task in daily life, yet it is challenging for robots due to the bags' infinite degrees of freedom and complex physical dynamics. Existing methods often struggle in generalization to unseen bag instances or deformations. To address this, we present DexKnot, a framework that combines keypoint affordance with diffusion policy to learn a generalizable bag-knotting policy. Our approach learns a shape-agnostic representation of bags from keypoint correspondence data collected through real-world manual deformation. For an unseen bag configuration, the keypoints can be identified by matching the representation to a reference. These keypoints are then provided to a diffusion transformer, which generates robot action based on a small number of human demonstrations. DexKnot enables effective policy generalization by reducing the dimensionality of observation space into a sparse set of keypoints. Experiments show that DexKnot achieves reliable and consistent knotting performance across a variety of previously unseen instances and deformations.
>
---
#### [new 033] Identifying Influential Actions in Human-Robot Interactions
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在识别影响人类行为的机器人动作。通过引入转移熵方法，分析机器人动作对人类响应的影响，提升系统设计与适应性。**

- **链接: [https://arxiv.org/pdf/2603.07885](https://arxiv.org/pdf/2603.07885)**

> **作者:** Haoyang Jiang; Chenfei Xu; Yuya Okadome; Yukata Nakamura
>
> **备注:** Presented at the 30th International Symposium on Artificial Life and Robotics (AROB 30th). Beppu, Japan, January 2025
>
> **摘要:** Human-robot interaction combines robotics, cognitive science, and human factors to study collaborative systems. This paper introduces a method for identifying influential robot actions using transfer entropy, a statistic that measures directed information transfer between time series. TE is effective for capturing complex, nonlinear interactions. We apply this method to analyze how robot actions affect human behavior during a conversation with a remotely controlled robot avatar. By focusing on the impact of proximity, our approach demonstrates TE's capability to identify key actions influencing human responses, highlighting its potential to improve the design and adaptability of robotic systems.
>
---
#### [new 034] SysNav: Multi-Level Systematic Cooperation Enables Real-World, Cross-Embodiment Object Navigation
- **分类: cs.RO**

- **简介: 该论文属于对象导航任务，解决真实环境中物体导航的复杂问题。通过SysNav系统，实现多层级协作，提升导航成功率和效率。**

- **链接: [https://arxiv.org/pdf/2603.06914](https://arxiv.org/pdf/2603.06914)**

> **作者:** Haokun Zhu; Zongtai Li; Zihan Liu; Kevin Guo; Zhengzhi Lin; Yuxin Cai; Guofei Chen; Chen Lv; Wenshan Wang; Jean Oh; Ji Zhang
>
> **摘要:** Object navigation (ObjectNav) in real-world environments is a complex problem that requires simultaneously addressing multiple challenges, including complex spatial structure, long-horizon planning and semantic understanding. Recent advances in Vision-Language Models (VLMs) offer promising capabilities for semantic understanding, yet effectively integrating them into real-world navigation systems remains a non-trivial challenge. In this work, we formulate real-world ObjectNav as a system-level problem and introduce SysNav, a three-level ObjectNav system designed for real-world crossembodiment deployment. SysNav decouples semantic reasoning, navigation planning and motion control to ensure robustness and generalizability. At the high-level, we summarize the environment into a structured scene representation and leverage VLMs to provide semantic-grounded navigation guidance. At the mid-level, we introduce a hierarchical room-based navigation strategy that reserves VLM guidance for room-level decisions, which effectively utilizes its reasoning ability while ensuring system efficiency. At the low-level, planned waypoints are executed through different embodiment-specific motion control modules. We deploy our system on three embodiments, a custom-built wheeled robot, the Unitree Go2 quadruped and the Unitree G1 humanoid, and conduct 190 real-world experiments. Our system achieves substantial improvements in both success rate and navigation efficiency. To the best of our knowledge, SysNav is the first system capable of reliably and efficiently completing building-scale long-range object navigation in complex real-world environments. Furthermore, extensive experiments on four simulation benchmarks demonstrate state-of-the-art performance. Project page is available at: this https URL.
>
---
#### [new 035] Learning-Based Robust Control: Unifying Exploration and Distributional Robustness for Reliable Robotics via Free Energy
- **分类: cs.RO; math.OC**

- **简介: 该论文属于机器人控制任务，旨在解决可靠控制中学习策略与保证鲁棒性的矛盾。通过结合自由能原理，提出一种联合学习环境与奖励的鲁棒控制方法。**

- **链接: [https://arxiv.org/pdf/2603.06831](https://arxiv.org/pdf/2603.06831)**

> **作者:** Hozefa Jesawada; Giovanni Russo; Abdalla Swikir; Fares Abu-Dakka
>
> **摘要:** A key challenge towards reliable robotic control is devising computational models that can both learn policies and guarantee robustness when deployed in the field. Inspired by the free energy principle in computational neuroscience, to address these challenges, we propose a model for policy computation that jointly learns environment dynamics and rewards, while ensuring robustness to epistemic uncertainties. Expounding a distributionally robust free energy principle, we propose a modification to the maximum diffusion learning framework. After explicitly characterizing robustness of our policies to epistemic uncertainties in both environment and reward, we validate their effectiveness on continuous-control benchmarks, via both simulations and real-world experiments involving manipulation with a Franka Research~3 arm. Across simulation and zero-shot deployment, our approach narrows the sim-to-real gap, and enables repeatable tabletop manipulation without task-specific fine-tuning.
>
---
#### [new 036] Choose What to Observe: Task-Aware Semantic-Geometric Representations for Visuomotor Policy
- **分类: cs.RO**

- **简介: 该论文属于视觉-运动策略任务，解决视觉干扰导致策略脆弱的问题。通过构建语义-几何表征，提升策略在外观变化下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07875](https://arxiv.org/pdf/2603.07875)**

> **作者:** Haoran Ding; Liang Ma; Yaxun Yang; Wen Yang; Tianyu Liu; Anqing Duan; Xiaodan Liang; Dezhen Song; Ivan Laptev; Yoshihiko Nakamura
>
> **摘要:** Visuomotor policies learned from demonstrations often overfit to nuisance visual factors in raw RGB observations, resulting in brittle behavior under appearance shifts such as background changes and object recoloring. We propose a task-aware observation interface that canonicalizes visual input into a shared representation, improving robustness to out-of-distribution (OOD) appearance changes without modifying or fine-tuning the policy. Given an RGB image and an open-vocabulary specification of task-relevant entities, we use SAM3 to segment the target object and robot/gripper. We construct an L0 observation by repainting segmented entities with predefined semantic colors on a constant background. For tasks requiring stronger geometric cues, we further inject monocular depth from Depth Anything 3 into the segmented regions via depth-guided overwrite, yielding a unified semantic--geometric observation (L1) that remains a standard 3-channel, image-like input. We evaluate on RoboMimic (Lift), ManiSkill YCB grasping under clutter, four RLBench tasks under controlled appearance shifts, and two real-world Franka tasks (ReachX and CloseCabinet). Across benchmarks and policy backbones (Flow Matching Policy and SmolVLA), our interface preserves in-distribution performance while substantially improving robustness under OOD visual shifts.
>
---
#### [new 037] CAR: Cross-Vehicle Kinodynamics Adaptation via Mobility Representation
- **分类: cs.RO**

- **简介: 该论文属于自主移动任务，解决异构车辆动力学适应问题。提出CAR框架，通过共享潜在空间实现快速动力学迁移，减少数据收集和计算开销。**

- **链接: [https://arxiv.org/pdf/2603.06866](https://arxiv.org/pdf/2603.06866)**

> **作者:** Tong Xu; Chenhui Pan; Xuesu Xiao
>
> **摘要:** Developing autonomous off-road mobility typically requires either extensive, platform-specific data collection or relies on simplified abstractions, such as unicycle or bicycle models, that fail to capture the complex kinodynamics of diverse platforms, ranging from wheeled to tracked vehicles. This limitation hinders scalability across evolving heterogeneous autonomous robot fleets. To address this challenge, we propose Cross-vehicle kinodynamics Adaptation via mobility Representation (CAR), a novel framework that enables rapid mobility transfer to new vehicles. CAR employs a Transformer encoder with Adaptive Layer Normalization to embed vehicle trajectory transitions and physical configurations into a shared mobility latent space. By identifying and extracting commonality from nearest neighbors within this latent space, our approach enables rapid kinodynamics adaptation to novel platforms with minimal data collection and computational overhead. We evaluate CAR using the Verti-Bench simulator, built on the Chrono multi-physics engine, and validate its performance on four distinct physical configurations of the Verti-4-Wheeler platform. With only one minute of new trajectory data, CAR achieves up to 67.2% reduction in prediction error compared to direct neighbor transfer across diverse unseen vehicle configurations, demonstrating the effectiveness of cross-vehicle mobility knowledge transfer in both simulated and real-world environments.
>
---
#### [new 038] NaviDriveVLM: Decoupling High-Level Reasoning and Motion Planning for Autonomous Driving
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自动驾驶任务，解决VLM中高阶推理与运动规划的权衡问题。提出NaviDriveVLM框架，分离推理与控制，提升性能并降低训练成本。**

- **链接: [https://arxiv.org/pdf/2603.07901](https://arxiv.org/pdf/2603.07901)**

> **作者:** Ximeng Tao; Pardis Taghavi; Dimitar Filev; Reza Langari; Gaurav Pandey
>
> **摘要:** Vision-language models (VLMs) have emerged as a promising direction for end-to-end autonomous driving (AD) by jointly modeling visual observations, driving context, and language-based reasoning. However, existing VLM-based systems face a trade-off between high-level reasoning and motion planning: large models offer strong semantic understanding but are costly to adapt for precise control, whereas small VLM models can be fine-tuned efficiently but often exhibit weaker reasoning. We propose NaviDriveVLM, a decoupled framework that separates reasoning from action generation using a large-scale Navigator and a lightweight trainable Driver. This design preserves reasoning ability, reduces training cost, and provides an explicit interpretable intermediate representation for downstream planning. Experiments on the nuScenes benchmark show that NaviDriveVLM outperforms large VLM baselines in end-to-end motion planning.
>
---
#### [new 039] C$^2$-Explorer: Contiguity-Driven Task Allocation with Connectivity-Aware Task Representation for Decentralized Multi-UAV Exploration
- **分类: cs.RO**

- **简介: 该论文属于多无人机探索任务，解决有限通信下的任务表示与分配问题。提出C²-Explorer框架，通过连通性图和邻域惩罚实现更连续的任务分配，提升探索效率。**

- **链接: [https://arxiv.org/pdf/2603.07699](https://arxiv.org/pdf/2603.07699)**

> **作者:** Xinlu Yan; Mingjie Zhang; Yuhao Fang; Yanke Sun; Jun Ma; Youmin Gong; Boyu Zhou; Jie Mei
>
> **摘要:** Efficient multi-UAV exploration under limited communication is severely bottlenecked by inadequate task representation and allocation. Previous task representations either impose heavy communication requirements for coordination or lack the flexibility to handle complex environments, often leading to inefficient traversal. Furthermore, short-horizon allocation strategies neglect spatiotemporal contiguity, causing non-contiguous assignments and frequent cross-region detours. To address this, we propose C$^2$-Explorer, a decentralized framework that constructs a connectivity graph to decompose disconnected unknown components into independent task units. We then introduce a contiguity-driven allocation formulation with a graph-based neighborhood penalty to discourage non-adjacent assignments, promoting more contiguous task sequences over time. Extensive simulation experiments show that C$^2$-Explorer consistently outperforms state-of-the-art (SOTA) baselines, reducing average exploration time by 43.1\% and path length by 33.3\%. Real-world flights further demonstrate the system's feasibility. The code will be released at this https URL
>
---
#### [new 040] MoMaStage: Skill-State Graph Guided Planning and Closed-Loop Execution for Long-Horizon Indoor Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文属于长期室内移动操作任务，解决长时序执行中的逻辑不一致和环境适应性问题。提出MoMaStage框架，结合视觉语言模型与技能状态图，实现结构化规划与闭环执行。**

- **链接: [https://arxiv.org/pdf/2603.08383](https://arxiv.org/pdf/2603.08383)**

> **作者:** Chenxu Li; Zixuan Chen; Yetao Li; Jiapeng Xu; Hongyu Ding; Jieqi Shi; Jing Huo; Yang Gao
>
> **备注:** 8 pages
>
> **摘要:** Indoor mobile manipulation (MoMA) enables robots to translate natural language instructions into physical actions, yet long-horizon execution remains challenging due to cascading errors and limited generalization across diverse environments. Learning-based approaches often fail to maintain logical consistency over extended horizons, while methods relying on explicit scene representations impose rigid structural assumptions that reduce adaptability in dynamic settings. To address these limitations, we propose MoMaStage, a structured vision-language framework for long-horizon MoMA that eliminates the need for explicit scene mapping. MoMaStage grounds a Vision-Language Model (VLM) within a Hierarchical Skill Library and a topology-aware Skill-State Graph, constraining task decomposition and skill composition within a feasible transition space. This structured grounding ensures that generated plans remain logically consistent and topologically valid with respect to the agent's evolving physical state. To enhance robustness, MoMaStage incorporates a closed-loop execution mechanism that monitors proprioceptive feedback and triggers graph-constrained semantic replanning when deviations are detected, maintaining alignment between planned skills and physical outcomes. Extensive experiments in physics-rich simulations and real-world environments demonstrate that MoMaStage outperforms state-of-the-art baselines, achieving substantially higher planning success, reducing token overhead, and significantly improving overall task success rates in long-horizon mobile manipulation. Video demonstrations are available on the project website: this https URL.
>
---
#### [new 041] Robodimm: A Physics-Grounded Framework for Automated Actuator Sizing in Scalable Modular Robots
- **分类: cs.RO**

- **简介: 该论文提出Robodimm框架，用于解决模块化机器人中执行器尺寸自动选择的问题，通过动力学分析和优化方法提升设计效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.06864](https://arxiv.org/pdf/2603.06864)**

> **作者:** J. L. Torres; M. Munoz; J. D. Alvarez; J. L. Blanco; A. Gimenez
>
> **备注:** 8 pages, 3 figures. Preprint version submitted to arXiv
>
> **摘要:** Selecting an appropriate motor-gearbox combination is a critical design task in robotics because it directly affects cost, mass, and dynamic performance. This process is especially challenging in modular robots with closed kinematic chains, where joint torques are coupled and actuator inertia propagates through the mechanism. We present Robodimm, a software framework for automated actuator sizing in scalable robot architectures. By leveraging Pinocchio for dynamics and Pink for inverse kinematics, Robodimm uses a Karush-Kuhn-Tucker (KKT) formulation for constrained inverse dynamics. The platform supports parametric scaling, interactive trajectory programming through jog modes, and a two-round validation workflow that addresses actuator self-weight effects.
>
---
#### [new 042] Material Driven HRI Design: Aesthetics as Explainability
- **分类: cs.RO**

- **简介: 论文探讨如何通过材料设计提升人机交互的可解释性，解决机器人外观与用户期望不匹配的问题。通过分析6个机器人，提出材料作为交互信号的框架。**

- **链接: [https://arxiv.org/pdf/2603.06879](https://arxiv.org/pdf/2603.06879)**

> **作者:** Natalie Friedman; Kevin Weatherwax; Chengchao Zhu
>
> **备注:** 4 pages, 1 table, 2026 ACM/IEEE Human-Robot Interaction Conference Workshop on Articulating the Value of Design Research for HRI
>
> **摘要:** Aesthetics - often treated as secondary to function-guides how people interpret robots' roles. A great deal of robot designs - both real and fictitious - use sleek industrial aesthetics. These feature hard glossy plastics, hiding as much of the underlying mechanical and electrical components as possible, resembling something akin to a nude humanoid figure. This leaves robots as something of a blank slate to which end-users apply coverings to, often based on media of fiction and non-fiction alike. We argue that designers can take cues from fashion to design interaction and set appropriate expectations. Rather than viewing appearance as decoration, we propose that color, texture, and material choices function as interaction signals. These signals can invite or discourage touch, clarify a robot's role, and help align user expectations with a robot's actual capabilities. When done thoughtfully, such cues can create familiarity and legibility; when done poorly, they can lead to wrong expectations. This preliminary paper proposes a framework describing how materials can create explainability by signaling expectations for interaction, task, and environment. We use this framework to do a content analysis of 6 robots.
>
---
#### [new 043] T2Nav Algebraic Topology Aware Temporal Graph Memory and Loop Detection for ZeroShot Visual Navigation
- **分类: cs.RO**

- **简介: 该论文属于零样本视觉导航任务，旨在解决自主代理在未知环境中高效导航的问题。通过整合多源数据和图推理，实现可靠路径规划与环路检测。**

- **链接: [https://arxiv.org/pdf/2603.06918](https://arxiv.org/pdf/2603.06918)**

> **作者:** Quang-Anh N. D.; Duc Pham; Minh-Anh Nguyen; Tung Doan; Tuan Dang
>
> **摘要:** Deploying autonomous agents in real world environments is challenging, particularly for navigation, where systems must adapt to situations they have not encountered before. Traditional learning approaches require substantial amounts of data, constant tuning, and, sometimes, starting over for each new task. That makes them hard to scale and not very flexible. Recent breakthroughs in foundation models, such as large language models and vision language models, enable systems to attempt new navigation tasks without requiring additional training. However, many of these methods only work with specific input types, employ relatively basic reasoning, and fail to fully exploit the details they observe or the structure of the spaces. Here, we introduce T2Nav, a zeroshot navigation system that integrates heterogeneous data and employs graph-based reasoning. By directly incorporating visual information into the graph and matching it to the environment, our approach enables the system to strike a good balance between exploration and goal attainment. This strategy allows robust obstacle avoidance, reliable loop closure detection, and efficient path planning while eliminating redundant exploration patterns. The system demonstrates flexibility by handling goals specified using reference images of target object instances, making it particularly suitable for scenarios in which agents must navigate to visually similar yet spatially distinct instances. Experiments demonstrate that our approach is efficient and adapts well to unknown environments, moving toward practical zero-shot instance-image navigation capabilities.
>
---
#### [new 044] Stability-Guided Exploration for Diverse Motion Generation
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决数据收集瓶颈问题。通过结合RRT与采样MPC，提出一种新方法，在黑盒仿真中生成多样化长时序操作策略。**

- **链接: [https://arxiv.org/pdf/2603.06773](https://arxiv.org/pdf/2603.06773)**

> **作者:** Eckart Cobo-Briesewitz; Tilman Burghoff; Denis Shcherba; Armand Jordana; Marc Toussaint
>
> **摘要:** Scaling up datasets is highly effective in improving the performance of deep learning models, including in the field of robot learning. However, data collection still proves to be a bottleneck. Approaches relying on collecting human demonstrations are labor-intensive and inherently limited: they tend to be narrow, task-specific, and fail to adequately explore the full space of feasible states. Synthetic data generation could remedy this, but current techniques mostly rely on local trajectory optimization and fail to find diverse solutions. In this work, we propose a novel method capable of finding diverse long-horizon manipulations through black-box simulation. We achieve this by combining an RRT-style search with sampling-based MPC, together with a novel sampling scheme that guides the exploration toward stable configurations. Specifically, we sample from a manifold of stable states while growing a search tree directly through simulation, without restricting the planner to purely stable motions. We demonstrate the method's ability to discover diverse manipulation strategies, including pushing, grasping, pivoting, throwing, and tool use, across different robot morphologies, without task-specific guidance.
>
---
#### [new 045] Exp-Force: Experience-Conditioned Pre-Grasp Force Selection with Vision-Language Models
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决预接触抓取力选择问题。通过视觉-语言模型，利用历史经验预测最小可行抓取力，提升抓取安全性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08668](https://arxiv.org/pdf/2603.08668)**

> **作者:** Siqi Shang; Minchao Huang; Bill Fan; Lillian Chin
>
> **摘要:** Accurate pre-contact grasp force selection is critical for safe and reliable robotic manipulation. Adaptive controllers regulate force after contact but still require a reasonable initial estimate. Starting a grasp with too little force requires reactive adjustment, while starting a grasp with too high a force risks damaging fragile objects. This trade-off is particularly challenging for compliant grippers, whose contact mechanics are difficult to model analytically. We propose Exp-Force, an experience-conditioned framework that predicts the minimum feasible grasping force from a single RGB image. The method retrieves a small set of relevant prior grasping experiences and conditions a vision-language model on these examples for in-context inference, without analytic contact models or manually designed heuristics. On 129 object instances, ExpForce achieves a best-case MAE of 0.43 N, reducing error by 72% over zero-shot inference. In real-world tests on 30 unseen objects, it improves appropriate force selection rate from 63% to 87%. These results demonstrate that Exp-Force enables reliable and generalizable pre-grasp force selection by leveraging prior interaction experiences. this http URL
>
---
#### [new 046] Vision-Guided MPPI for Agile Drone Racing: Navigating Arbitrary Gate Poses via Neural Signed Distance Fields
- **分类: cs.RO**

- **简介: 该论文属于自主无人机竞速任务，解决复杂门位下的快速导航问题。提出Gate-SDF与MPPI结合的框架，实现无需参考轨迹的实时避障飞行。**

- **链接: [https://arxiv.org/pdf/2603.07199](https://arxiv.org/pdf/2603.07199)**

> **作者:** Fangguo Zhao; Hanbing Zhang; Zhouheng Li; Xin Guan; Shuo Li
>
> **摘要:** Autonomous drone racing requires the tight coupling of perception, planning, and control under extreme agility. However, recent approaches typically rely on precomputed spatial reference trajectories or explicit 6-DoF gate pose estimation, rendering them brittle to spatial perturbations, unmodeled track changes, and sensor noise. Conversely, end-to-end learning policies frequently overfit to specific track layouts and struggle with zero-shot generalization. To address these fundamental limitations, we propose a fully onboard, vision guided optimal control framework that enables reference-free agile flight through arbitrarily placed and oriented gates. Central to our approach is Gate-SDF, a novel, implicitly learned neural signed distance field. Gate-SDF directly processes raw, noisy depth images to predict a continuous spatial field that provides both collision repulsion and active geometric guidance toward the valid traversal area. We seamlessly integrate this representation into a sampling-based Model Predictive Path Integral (MPPI) controller. By fully exploiting GPU parallelism, the framework evaluates these continuous spatial constraints across thousands of simulated trajectory rollouts simultaneously in real time. Furthermore, our formulation inherently maintains spatial consistency, ensuring robust navigation even under severe visual occlusion during aggressive maneuvers. Extensive simulations and real-world experiments demonstrate that the proposed system achieves high-speed agile flight and successfully navigates unseen tracks subject to severe unmodeled gate displacements and orientation perturbations. Videos are available at this https URL
>
---
#### [new 047] ICLR: In-Context Imitation Learning with Visual Reasoning
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决传统方法缺乏任务意图表示的问题。提出ICLR框架，结合视觉推理提升机器人在复杂任务中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.07530](https://arxiv.org/pdf/2603.07530)**

> **作者:** Toan Nguyen; Weiduo Yuan; Songlin Wei; Hui Li; Daniel Seita; Yue Wang
>
> **备注:** Project website: this https URL
>
> **摘要:** In-context imitation learning enables robots to adapt to new tasks from a small number of demonstrations without additional training. However, existing approaches typically condition only on state-action trajectories and lack explicit representations of task intent. This limitation hinders performance in complex and ambiguous task settings where the same actions may be consistent with different objectives. To address this, we present In-Context Imitation Learning with Visual Reasoning (ICLR), a novel framework that augments demonstration prompts with structured visual reasoning traces representing anticipated future robot trajectories in image space. ICLR also jointly learns to generate reasoning traces and low-level actions within a unified autoregressive transformer, enabling the model to mimic not only action prediction but also the reasoning process that leads to those actions. We extensively evaluate ICLR in both simulation and real-world manipulation tasks and demonstrate consistent improvements in success rates and generalization to unseen tasks and novel object configurations compared to other in-context imitation learning methods. These results suggest that incorporating embodied visual reasoning represents a promising direction for enhancing the robustness and generalization of robotic in-context learning systems.
>
---
#### [new 048] Two-Stage Path Following for Mobile Manipulators via Dimensionality-Reduced Graph Search and Numerical Optimization
- **分类: cs.RO**

- **简介: 该论文属于移动机械臂路径规划任务，解决高维配置空间和运动学约束下的路径跟随问题。通过两阶段方法，先进行离散搜索，再优化连续轨迹，提升精度与平滑度。**

- **链接: [https://arxiv.org/pdf/2603.07003](https://arxiv.org/pdf/2603.07003)**

> **作者:** Fuyu Guo; Yuting Mei; Yuyao Zhang; Qian Tang
>
> **摘要:** Efficient path following for mobile manipulators is often hindered by high-dimensional configuration spaces and kinematic constraints. This paper presents a robust two-stage configuration planning framework that decouples the 8-DoF planning problem into a tractable 2-DoF base optimization under a yaw-fixed base planning assumption. In the first stage, the proposed approach utilizes IRM to discretize the task-space path into a multi-layer graph, where an initial feasible path is extracted via a Dijkstra-based dynamic programming approach to ensure computational efficiency and global optimality within the discretized graph. In the second stage, to overcome discrete search quantization, feasible base regions are transformed into convex hulls, enabling subsequent continuous refinement via the L-BFGS algorithm to maximize trajectory smoothness while strictly enforcing reachability constraints. Simulation results demonstrate the theoretical precision of the proposed method by achieving sub-millimeter kinematic accuracy in simulation, and physical experiments on an omnidirectional mobile manipulator further validate the framework's robustness and practical applicability.
>
---
#### [new 049] PanoDP: Learning Collision-Free Navigation with Panoramic Depth and Differentiable Physics
- **分类: cs.RO**

- **简介: 该论文提出PanoDP，用于解决复杂环境中无碰撞导航问题。结合全景深度感知与可微物理训练，提升导航安全性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.07644](https://arxiv.org/pdf/2603.07644)**

> **作者:** Hao Zhong; Pei Chi; Jiang Zhao; Shenghai Yuan; Xuyang Gao; Thien-Minh Nguyen; Lihua Xie
>
> **摘要:** Autonomous collision-free navigation in cluttered environments requires safe decision-making under partial observability with both static structure and dynamic obstacles. We present \textbf{PanoDP}, a communication-free learning framework that combines four-view panoramic depth perception with differentiable-physics-based training signals. PanoDP encodes panoramic depth using a lightweight CNN and optimizes policies with dense differentiable collision and motion-feasibility terms, improving training stability beyond sparse terminal collisions. We evaluate PanoDP on a controlled ring-to-center benchmark with systematic sweeps over agent count, obstacle density/layout, and dynamic behaviors, and further test out-of-distribution generalization in an external simulator (e.g., AirSim). Across settings, PanoDP increases collision-free and completion rates over single-view and non-physics-guided baselines under matched training budgets, and ablations (view masking, rotation augmentation) confirm the policy leverages 360-degree information. Code will be open source upon acceptance.
>
---
#### [new 050] SurgSync: Time-Synchronized Multi-Modal Data Collection Framework and Dataset for Surgical Robotics
- **分类: cs.RO**

- **简介: 该论文提出SurgSync框架，解决手术机器人缺乏高质量训练数据的问题。通过多模态数据采集与同步，支持AI训练与实时推理，提升手术自动化水平。**

- **链接: [https://arxiv.org/pdf/2603.06919](https://arxiv.org/pdf/2603.06919)**

> **作者:** Haoying Zhou; Chang Liu; Yimeng Wu; Junlin Wu; Zijian Wu; Yu Chung Lee; Sara Martuscelli; Spetimiu E. Salcudean; Gregory S. Fischer; Peter Kazanzides
>
> **备注:** Accepted By International Conference on Robotics and Automation (ICRA), IEEE, 2026. More details can be found at this https URL
>
> **摘要:** Most existing robotic surgery systems adopt a human-in-the-loop paradigm, often with the surgeon directly teleoperating the robotic system. Adding intelligence to these robots would enable higher-level control, such as supervised autonomy or even full autonomy. However, artificial intelligence (AI) requires large amounts of training data, which is currently lacking. This work proposes SurgSync, a multi-modal data collection framework with offline and online synchronization to support training and real-time inference, respectively. The framework is implemented on a da Vinci Research Kit (dVRK) and introduces (1) dual-mode (online/offline-matching) synchronized recorders, (2) a modern stereo endoscope to achieve image quality on par with clinical systems, and (3) additional sensors such as a side-view camera and a novel capacitive contact sensor to provide ground truth contact data. The framework also incorporates a post-processing toolbox for tasks such as depth estimation, optical flow, and a practical kinematic reprojection method using Gaussian heatmap. User studies with participants of varying skill levels are performed with ex-vivo tissue to provide clinically realistic data, and a network for surgical skill assessment is employed to demonstrate utilization of the collected data. Through the user study experiments, we obtained a dataset of 214 validated instances across multiple canonical training tasks. All software and data are available at this http URL.
>
---
#### [new 051] EndoSERV: A Vision-based Endoluminal Robot Navigation System
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人导航任务，旨在解决内窥镜机器人在复杂腔体中定位困难的问题。提出EndoSERV方法，通过图像特征映射和分段定位提高导航精度。**

- **链接: [https://arxiv.org/pdf/2603.08324](https://arxiv.org/pdf/2603.08324)**

> **作者:** Junyang Wu; Fangfang Xie; Minghui Zhang; Hanxiao Zhang; Jiayuan Sun; Yun Gu; Guang-Zhong Yang
>
> **摘要:** Robot-assisted endoluminal procedures are increasingly used for early cancer intervention. However, the intricate, narrow and tortuous pathways within the luminal anatomy pose substantial difficulties for robot navigation. Vision-based navigation offers a promising solution, but existing localization approaches are error-prone due to tissue deformation, in vivo artifacts and a lack of distinctive landmarks for consistent localization. This paper presents a novel EndoSERV localization method to address these challenges. It includes two main parts, \textit{i.e.}, \textbf{SE}gment-to-structure and \textbf{R}eal-to-\textbf{V}irtual mapping, and hence the name. For long-range and complex luminal structures, we divide them into smaller sub-segments and estimate the odometry independently. To cater for label insufficiency, an efficient transfer technique maps real image features to the virtual domain to use virtual pose ground truth. The training phases of EndoSERV include an offline pretraining to extract texture-agnostic features, and an online phase that adapts to real-world conditions. Extensive experiments based on both public and clinical datasets have been performed to demonstrate the effectiveness of the method even without any real pose labels.
>
---
#### [new 052] AtomicVLA: Unlocking the Potential of Atomic Skill Learning in Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出AtomicVLA，解决机器人长序列任务和持续学习问题。通过原子技能抽象与动态专家组合，提升任务规划与执行效果。**

- **链接: [https://arxiv.org/pdf/2603.07648](https://arxiv.org/pdf/2603.07648)**

> **作者:** Likui Zhang; Tao Tang; Zhihao Zhan; Xiuwei Chen; Zisheng Chen; Jianhua Han; Jiangtong Zhu; Pei Xu; Hang Xu; Hefeng Wu; Liang Lin; Xiaodan Liang
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Recent advances in Visual-Language-Action (VLA) models have shown promising potential for robotic manipulation tasks. However, real-world robotic tasks often involve long-horizon, multi-step problem-solving and require generalization for continual skill acquisition, extending beyond single actions or skills. These challenges present significant barriers for existing VLA models, which use monolithic action decoders trained on aggregated data, resulting in poor scalability. To address these challenges, we propose AtomicVLA, a unified planning-and-execution framework that jointly generates task-level plans, atomic skill abstractions, and fine-grained actions. AtomicVLA constructs a scalable atomic skill library through a Skill-Guided Mixture-of-Experts (SG-MoE), where each expert specializes in mastering generic yet precise atomic skills. Furthermore, we introduce a flexible routing encoder that automatically assigns dedicated atomic experts to new skills, enabling continual learning. We validate our approach through extensive experiments. In simulation, AtomicVLA outperforms $\pi_{0}$ by 2.4\% on LIBERO, 10\% on LIBERO-LONG, and outperforms $\pi_{0}$ and $\pi_{0.5}$ by 0.22 and 0.25 in average task length on CALVIN. Additionally, our AtomicVLA consistently surpasses baselines by 18.3\% and 21\% in real-world long-horizon tasks and continual learning. These results highlight the effectiveness of atomic skill abstraction and dynamic expert composition for long-horizon and lifelong robotic tasks. The project page is \href{this https URL}{here}.
>
---
#### [new 053] ACCURATE: Arbitrary-shaped Continuum Reconstruction Under Robust Adaptive Two-view Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决柔性连续体（如导管）的高精度重建问题。通过结合神经网络与几何约束算法，提升重建准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07533](https://arxiv.org/pdf/2603.07533)**

> **作者:** Yaozhi Zhang; Shun Yu; Yugang Zhang; Yang Liu
>
> **摘要:** Accurate reconstruction of arbitrary-shaped long slender continuum bodies, such as guidewires, catheters and other soft continuum manipulators, is essential for accurate mechanical simulation. However, existing image-based reconstruction approaches often suffer from limited accuracy because they often underutilize camera geometry, or lack generality as they rely on rigid geometric assumptions that may fail for continuum robots with complex and highly deformable shapes. To address these limitations, we propose ACCURATE, a 3D reconstruction framework integrating an image segmentation neural network with a geometry-constrained topology traversal and dynamic programming algorithm that enforces global biplanar geometric consistency, minimizes the cumulative point-to-epipolar-line distance, and remains robust to occlusions and epipolar ambiguities cases caused by noise and discretization. Our method achieves high reconstruction accuracy on both simulated and real phantom datasets acquired using a clinical X-ray C-arm system, with mean absolute errors below 1.0 mm.
>
---
#### [new 054] Embedding Classical Balance Control Principles in Reinforcement Learning for Humanoid Recovery
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决人形机器人跌倒后恢复的问题。通过将经典平衡指标嵌入强化学习，提升恢复能力与泛化性。**

- **链接: [https://arxiv.org/pdf/2603.08619](https://arxiv.org/pdf/2603.08619)**

> **作者:** Nehar Poddar; Stephen McCrory; Luigi Penco; Geoffrey Clark; Hakki Erhan Svil; Robert Griffin
>
> **摘要:** Humanoid robots remain vulnerable to falls and unrecoverable failure states, limiting their practical utility in unstructured environments. While reinforcement learning has demonstrated stand-up behaviors, existing approaches treat recovery as a pure task-reward problem without an explicit representation of the balance state. We present a unified RL policy that addresses this limitation by embedding classical balance metrics: capture point, center-of-mass state, and centroidal momentum, as privileged critic inputs and shaping rewards directly around these quantities during training, while the actor relies solely on proprioception for zero-shot hardware transfer. Without reference trajectories or scripted contacts, a single policy spans the full recovery spectrum: ankle and hip strategies for small disturbances, corrective stepping under large pushes, and compliant falling with multi-contact stand-up using the hands, elbows, and knees. Trained on the Unitree H1-2 in Isaac Lab, the policy achieves a 93.4% recovery rate across randomized initial poses and unscripted fall configurations. An ablation study shows that removing the balance-informed structure causes stand-up learning to fail entirely, confirming that these metrics provide a meaningful learning signal rather than incidental structure. Sim-to-sim transfer to MuJoCo and preliminary hardware experiments further demonstrate cross-environment generalization. These results show that embedding interpretable balance structure into the learning framework substantially reduces time spent in failure states and broadens the envelope of autonomous recovery.
>
---
#### [new 055] Diff-Muscle: Efficient Learning for Musculoskeletal Robotic Table Tennis
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决肌肉骨骼机器人在复杂任务中的高效学习问题。通过引入Diff-Muscle算法，将动作空间降维，并结合分层强化学习框架，提升机器人在乒乓球任务中的表现。**

- **链接: [https://arxiv.org/pdf/2603.08617](https://arxiv.org/pdf/2603.08617)**

> **作者:** Wentao Zhao; Jun Guo; Kangyao Huang; Xin Liu; Huaping Liu
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Musculoskeletal robots provide superior advantages in flexibility and dexterity, positioning them as a promising frontier towards embodied intelligence. However, current research is largely confined to relative simple tasks, restricting the exploration of their full potential in multi-segment coordination. Furthermore, efficient learning remains a challenge, primarily due to the high-dimensional action space and inherent overactuated structures. To address these challenges, we propose Diff-Muscle, a musculoskeletal robot control algorithm that leverages differential flatness to reformulate policy learning from the redundant muscle-activation space into a significantly lower-dimensional joint space. Furthermore, we utilize the highly dynamic robotic table tennis task to evaluate our algorithm. Specifically, we propose a hierarchical reinforcement learning framework that integrates a Kinematics-based Muscle Actuation Controller (K-MAC) with high-level trajectory planning, enabling a musculoskeletal robot to perform dexterous and precise rallies. Experimental results demonstrate that Diff-Muscle significantly outperforms state-of-the-art baselines in success rates while maintaining minimal muscle activation. Notably, the proposed framework successfully enables the musculoskeletal robots to achieve continuous rallies in a challenging dual-robot setting.
>
---
#### [new 056] RoboCritics: Enabling Reliable End-to-End LLM Robot Programming through Expert-Informed Critics
- **分类: cs.RO**

- **简介: 该论文属于机器人编程任务，旨在解决LLM生成代码不透明带来的安全与可靠性问题。通过引入专家指导的运动级批评机制，提升程序验证与调试能力。**

- **链接: [https://arxiv.org/pdf/2603.06842](https://arxiv.org/pdf/2603.06842)**

> **作者:** Callie Y. Kim; Nathan Thomas White; Evan He; Frederic Sala; Bilge Mutlu
>
> **备注:** 10 pages, 5 figures, Proceedings of the 21st ACM/IEEE International Conference on Human Robot Interaction (HRI 2026)
>
> **摘要:** End-user robot programming grants users the flexibility to re-task robots in situ, yet it remains challenging for novices due to the need for specialized robotics knowledge. Large Language Models (LLMs) hold the potential to lower the barrier to robot programming by enabling task specification through natural language. However, current LLM-based approaches generate opaque, "black-box" code that is difficult to verify or debug, creating tangible safety and reliability risks in physical systems. We present RoboCritics, an approach that augments LLM-based robot programming with expert-informed motion-level critics. These critics encode robotics expertise to analyze motion-level execution traces for issues such as joint speed violations, collisions, and unsafe end-effector poses. When violations are detected, critics surface transparent feedback and offer one-click fixes that forward structured messages back to the LLM, enabling iterative refinement while keeping users in the loop. We instantiated RoboCritics in a web-based interface connected to a UR3e robot and evaluated it in a between-subjects user study (n=18). Compared to a baseline LLM interface, RoboCritics reduced safety violations, improved execution quality, and shaped how participants verified and refined their programs. Our findings demonstrate that RoboCritics enables more reliable and user-centered end-to-end robot programming with LLMs.
>
---
#### [new 057] POIROT: Investigating Direct Tangible vs. Digitally Mediated Interaction and Attitude Moderation in Multi-party Murder Mystery Games
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，探讨机器人作为游戏主持人时，实体互动与数字媒介对用户体验的影响。研究解决的是不同用户态度下互动方式效果差异的问题，通过实验验证了用户对机器人的负面态度会调节互动效果。**

- **链接: [https://arxiv.org/pdf/2603.08136](https://arxiv.org/pdf/2603.08136)**

> **作者:** Wen Chen; Rongxi Chen; Shankai Chen; Huiyang Gong; Minghui Guo; Yingri Xu; Xintong Wu; Xinyi Fu
>
> **备注:** 16 pages, 7 figures. Accepted to the 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI 2026)
>
> **摘要:** As social robots take on increasingly complex roles like game masters (GMs) in multi-party games, the expectation that physicality universally enhances user experience remains debated. This study challenges the "one-size-fits-all" view of tangible interaction by identifying a critical boundary condition: users' Negative Attitudes towards Robots (NARS). In a between-subjects experiment (N = 67), a custom-built robot GM facilitated a multi-party murder mystery game (MMG) by delivering clues either through direct tangible interaction or a digitally mediated interface. Baseline multivariate analysis (MANOVA) showed no significant main effect of delivery modality, confirming that tangibility alone does not guarantee superior engagement. However, primary analysis using multilevel linear models (MLM) revealed a reliable moderation: participants high in NARS experienced markedly lower narrative immersion under tangible delivery, whereas those with low NARS scores showed no such decrement. Qualitative findings further illuminate this divergence: tangibility provides novelty and engagement for some but imposes excessive proxemic friction for anxious users, for whom the digital interface acts as a protective social buffer. These results advance a conditional model of HRI and emphasize the necessity for adaptive systems that can tailor interaction modalities to user predispositions.
>
---
#### [new 058] The Neural Compass: Probabilistic Relative Feature Fields for Robotic Search
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人目标搜索任务，旨在解决如何从无标签数据中学习对象间的相对特征关系。提出ProReFF模型和基于学习的对齐策略，提升搜索效率。**

- **链接: [https://arxiv.org/pdf/2603.08544](https://arxiv.org/pdf/2603.08544)**

> **作者:** Gabriele Somaschini; Adrian Röfer; Abhinav Valada
>
> **备注:** 9 pages, 7 figures, 2 tables, submitted to IROS 2026
>
> **摘要:** Object co-occurrences provide a key cue for finding objects successfully and efficiently in unfamiliar environments. Typically, one looks for cups in kitchens and views fridges as evidence of being in a kitchen. Such priors have also been exploited in artificial agents, but they are typically learned from explicitly labeled data or queried from language models. It is still unclear whether these relations can be learned implicitly from unlabeled observations alone. In this work, we address this problem and propose ProReFF, a feature field model trained to predict relative distributions of features obtained from pre-trained vision language models. In addition, we introduce a learning-based strategy that enables training from unlabeled and potentially contradictory data by aligning inconsistent observations into a coherent relative distribution. For the downstream object search task, we propose an agent that leverages predicted feature distributions as a semantic prior to guide exploration toward regions with a high likelihood of containing the object. We present extensive evaluations demonstrating that ProReFF captures meaningful relative feature distributions in natural scenes and provides insight into the impact of our proposed alignment step. We further evaluate the performance of our search agent in 100 challenges in the Matterport3D simulator, comparing with feature-based baselines and human participants. The proposed agent is 20% more efficient than the strongest baseline and achieves up to 80% of human performance.
>
---
#### [new 059] Morphology-Independent Facial Expression Imitation for Human-Face Robots
- **分类: cs.RO**

- **简介: 该论文属于人脸表情模仿任务，旨在解决面部形态对表情模仿的影响问题。通过解耦表情与形态，实现更真实的机器人表情模仿。**

- **链接: [https://arxiv.org/pdf/2603.07068](https://arxiv.org/pdf/2603.07068)**

> **作者:** Xu Chen; Rui Gao; Che Sun; Zhehang Liu; Yuwei Wu; Shuo Yang; Yunde Jia
>
> **摘要:** Accurate facial expression imitation on human-face robots is crucial for achieving natural human-robot interaction. Most existing methods have achieved photorealistic expression imitation through mapping 2D facial landmarks to a robot's actuator commands. Their imitation of landmark trajectories is susceptible to interference from facial morphology, which would lead to a performance drop. In this paper, we propose a morphology-independent expression imitation method that decouples expressions from facial morphology to eliminate morphological influence and produce more realistic expressions for human-face robots. Specifically, we construct an expression decoupling module to learn expression semantics by disentangling the expression representation from the morphology representation in a self-supervised manner. We devise an expression transfer module to map the representations to the robot's actuator commands through a learning objective of perceiving expression errors, producing accurate facial expressions based on the learned expression semantics. To support experimental validation, a custom-designed and highly expressive human-face robot, namely Pengrui, is developed to serve as an experimental platform for realistic expression imitation. Extensive experiments demonstrate that our method enables the human-face robot to reproduce a wide range of human-like expressions effectively. All code and implementation details of the robot will be released.
>
---
#### [new 060] Collaborative Planning with Concurrent Synchronization for Operationally Constrained UAV-UGV Teams
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同规划任务，解决UAV与UGV在能量和地形约束下的同步协作问题。提出CoPCS方法，实现高效协同规划与实时充电。**

- **链接: [https://arxiv.org/pdf/2603.06898](https://arxiv.org/pdf/2603.06898)**

> **作者:** Zihao Deng; Qianhuang Li; Peng Gao; Maggie Wigness; John Rogers; Donghyun Kim; Hao Zhang
>
> **摘要:** Collaborative planning under operational constraints is an essential capability for heterogeneous robot teams tackling complex large-scale real-world tasks. Unmanned Aerial Vehicles (UAVs) offer rapid environmental coverage, but flight time is often limited by energy constraints, whereas Unmanned Ground Vehicles (UGVs) have greater energy capacity to support long-duration missions, but movement is constrained by traversable terrain. Individually, neither can complete tasks such as environmental monitoring. Effective UAV-UGV collaboration therefore requires energy-constrained multi-UAV task planning, traversability-constrained multi-UGV path planning, and crucially, synchronized concurrent co-planning to ensure timely in-mission recharging. To enable these capabilities, we propose Collaborative Planning with Concurrent Synchronization (CoPCS), a learning-based approach that integrates a heterogeneous graph transformer for operationally constrained task encoding with a transformer decoder for joint, synchronized co-planning that enables UAVs and UGVs to act concurrently in a coordinated manner. CoPCS is trained end-to-end under a unified imitation learning paradigm. We conducted extensive experiments to evaluate CoPCS in both robotic simulations and physical robot teams. Experimental results demonstrate that our method provides the novel multi-robot capability of synchronized concurrent co-planning and substantially improves team performance. More details of this work are available on the project website: this https URL.
>
---
#### [new 061] Energy-Efficient Collaborative Transport of Tether-Suspended Payloads via Rotating Equilibrium
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于协同空中运输任务，旨在解决传统静态平衡导致的能耗高问题。通过旋转平衡机制，使四旋翼机产生纯垂直推力，降低整体能耗。**

- **链接: [https://arxiv.org/pdf/2603.06955](https://arxiv.org/pdf/2603.06955)**

> **作者:** Eric Foss; Andrew Tai; Carlo Bosio; Mark W. Mueller
>
> **备注:** 7 pages, 8 figures
>
> **摘要:** Collaborative aerial transportation of tethered payloads is fundamentally limited by space, power, and weight constraints. Conventional approaches rely on static equilibrium conditions, where each vehicle tilts to generate the forces that ensure they maintain a formation geometry that avoids aerodynamic interactions and collision. This horizontal thrust component represents a significant energy penalty compared to the ideal case in which each vehicle produces purely vertical thrust to lift the payload. Operating in tighter tether configurations can minimize this effect, but at the cost of either having to fly the vehicles in closer proximity, which risks collision, or significantly increasing the length of the tether, which increases complexity and reduces potential use-cases. We propose operating the tether-suspended flying system at a rotating equilibrium. By maintaining steady circular motion, centrifugal forces provide the necessary horizontal tether tension, allowing each quadrotor to generate purely vertical thrust and thus reducing the total force (and power) required compared to an equilibrium where the thrusts are not vertical. It also allows for a wider range of tether configurations to be used without sacrificing efficiency. Results demonstrate that rotating equilibria can reduce power consumption relative to static lifting by up to 20%, making collaborative aerial solutions more practically relevant.
>
---
#### [new 062] A Comprehensive Analysis of the Effects of Network Quality of Service on Robotic Telesurgery
- **分类: cs.RO**

- **简介: 该论文属于网络质量对远程手术影响的研究，旨在分析网络退化对任务执行的影响。通过工具NetFI和用户实验，评估不同网络条件下的手术表现与安全。**

- **链接: [https://arxiv.org/pdf/2603.06824](https://arxiv.org/pdf/2603.06824)**

> **作者:** Zhaomeng Zhang; Seyed Hamid Reza Roodabeh; Homa Alemzadeh
>
> **备注:** Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** The viability of long-distance telesurgery hinges on reliable network Quality of Service (QoS), yet the impact of realistic network degradations on task performance is not sufficiently understood. This paper presents a comprehensive analysis of how packet loss, delay, and communication loss affect telesurgical task execution. We introduce NetFI, a novel fault injection tool that emulates different network conditions using stochastic QoS models informed by real-world network data. By integrating NetFI with a surgical simulation platform, we conduct a user study involving 15 participants at three proficiency levels, performing a standardized Peg Transfer task under varying levels of packet loss, delay, and communication loss. We analyze the effect of network QoS on overall task performance and the fine-grained motion primitives (MPs) using objective performance and safety metrics and subjective operator's perception of workload. We identify specific MPs vulnerable to network degradation and find strong correlations between proficiency, objective performance, and subjective workload. These findings offer quantitative insights into the operational boundaries of telesurgery. Our open-source tools and annotated dataset provide a foundation for developing robust and network-aware control and mitigation strategies.
>
---
#### [new 063] Toward Global Intent Inference for Human Motion by Inverse Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于人类运动意图推断任务，旨在用统一成本函数解释和预测人类伸手动作。通过逆强化学习方法，验证了时间变化成本函数的有效性。**

- **链接: [https://arxiv.org/pdf/2603.07797](https://arxiv.org/pdf/2603.07797)**

> **作者:** Sarmad Mehrdad; Maxime Sabbah; Vincent Bonnet; Ludovic Righetti
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** This paper investigates whether a single, unified cost function can explain and predict human reaching movements, in contrast with existing approaches that rely on subject- or posture-specific optimization criteria. Using the Minimal Observation Inverse Reinforcement Learning (MO-IRL) algorithm, together with a seven-dimensional set of candidate cost terms, we efficiently estimate time-varying cost weights for a standard planar reaching task. MO-IRL provides orders-of-magnitude faster convergence than bilevel formulations, while using only a fraction of the available data, enabling the practical exploration of time-varying cost structures. Three levels of generality are evaluated: Subject-Dependent Posture-Dependent, Subject-Dependent Posture-Independent, and Subject-Independent Posture-Independent. Across all cases, time-varying weights substantially improve trajectory reconstruction, yielding an average 27% reduction in RMSE compared to the baseline. The inferred costs consistently highlight a dominant role for joint-acceleration regulation, complemented by smaller contributions from torque-change smoothness. Overall, a single subject- and posture-agnostic time-varying cost function is shown to predict human reaching trajectories with high accuracy, supporting the existence of a unified optimality principle governing this class of movements.
>
---
#### [new 064] A Robust Antenna Provides Tactile Feedback in a Multi-legged Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决多足机器人在复杂环境中导航困难的问题。通过设计具有触觉反馈的仿生天线，提升机器人自主避障能力。**

- **链接: [https://arxiv.org/pdf/2603.07795](https://arxiv.org/pdf/2603.07795)**

> **作者:** Zhaochen J. Xu; Juntao He; Delfin Aydan; Malaika Taylor; Tianyu Wang; Jianfeng Lin; Wesley Dyer; Daniel I. Goldman
>
> **摘要:** Multi-legged elongate robots hold promise for maneuvering through complex environments. Prior work has demonstrated that reliable locomotion can be achieved using open-loop body undulation and foot placement on rugose terrain. However, robust navigation through confined spaces remains challenging when body-environment contact is extensive and terrain rheology varies rapidly. To address this challenge, we develop a pair of tactile antennae for multi-legged robots that enable real-time sensing of surrounding geometry, modeling the morphology and function of biological centipede antennae. Each antenna features gradient compliance, with a stiff base and soft tip, allowing repeated deformation and elastic recovery. Robophysical experiments reveal a relationship between continuous antenna curvature and contact force, leading to a simplified mapping from antenna deformation to inferred discrete collision states. We incorporate this mapping into a controller that selects among a set of locomotor maneuvers based on the inferred collision state. Experiments in obstacle-rich and confined environments demonstrate that tactile feedback enables reliable steering and allows the robot to recover from near-stuck conditions without requiring global environmental information or real-time vision. These results highlight how mechanically tuned tactile appendages can simplify sensing and enhance autonomy in elongate multi-legged robots operating in constrained spaces.
>
---
#### [new 065] A General Lie-Group Framework for Continuum Soft Robot Modeling
- **分类: cs.RO**

- **简介: 该论文属于软体机器人建模任务，旨在解决传统方法的局限性。提出一种基于李群的通用框架，实现几何局部控制和高效仿真。**

- **链接: [https://arxiv.org/pdf/2603.08232](https://arxiv.org/pdf/2603.08232)**

> **作者:** Lingxiao Xun; Benoît Rosa; Jérôme Szewczyk; Brahim Tamadazte
>
> **摘要:** This paper introduces a general Lie group framework for modeling continuum soft robots, employing Cosserat rod theory combined with cumulative parameterization on the Lie group SE(3). This novel approach addresses limitations present in current strain-based and configuration-based methods by providing geometric local control and eliminating unit quaternion constraints. The paper derives unified analytical expressions for kinematics, statics, and dynamics, including recursive Jacobian computations and an energy-conserving integrator suitable for real-time simulation and control. Additionally, the framework is extended to handle complex robotic structures, including segmented, branched, nested, and rigid-soft composite configurations, facilitating a modular and unified modeling strategy. The effectiveness, generality, and computational efficiency of the proposed methodology are demonstrated through various scenarios, including large-deformation rods, concentric tube robots, parallel robots, cable-driven robots, and articulated fingers. This work enhances modeling flexibility and numerical performance, providing an improved toolset for designing, simulating, and controlling soft robotic systems.
>
---
#### [new 066] Efficient Trajectory Optimization for Autonomous Racing via Formula-1 Data-Driven Initialization
- **分类: cs.RO**

- **简介: 该论文属于自主赛车轨迹优化任务，旨在解决传统初始化方法收敛慢、效果差的问题。通过学习Formula 1数据，提出一种基于轨迹偏移的初始化策略，提升优化效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.07126](https://arxiv.org/pdf/2603.07126)**

> **作者:** Samir Shehadeh; Lukas Kutsch; Nils Dengler; Sicong Pan; Maren Bennewitz
>
> **摘要:** Trajectory optimization is a central component of fast and efficient autonomous racing. However practical optimization pipelines remain highly sensitive to initialization and may converge slowly or to suboptimal local solutions when seeded with heuristic trajectories such as the centerline or minimum-curvature paths. To address this limitation, we leverage expert driving behavior as a initialization prior and propose a learning-informed initialization strategy based on real-world Formula 1 telemetry. To this end, we first construct a multi-track Formula~1 trajectory dataset by reconstructing and aligning noisy GPS telemetry to a standardized reference-line representation across 17 tracks. Building on this, we present a neural network that predicts an expert-like raceline offset directly from local track geometry, without explicitly modeling vehicle dynamics or forces. The predicted raceline is then used as an informed seed for a minimum-time optimal control solver. Experiments on all 17 tracks demonstrate that the learned initialization accelerates solver convergence and significantly reduces runtime compared to traditional geometric baselines, while preserving the final optimized lap time.
>
---
#### [new 067] Soft Rigid Hybrid Gripper with Inflatable Silicone Pockets for Tunable Frictional Grasping
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决传统夹爪对易损物体的损伤问题。通过设计软硬结合的气囊夹爪，调节摩擦力实现稳定抓取。**

- **链接: [https://arxiv.org/pdf/2603.07308](https://arxiv.org/pdf/2603.07308)**

> **作者:** Hoang Hiep Ly; Cong-Nhat Nguyen; Doan-Quang Tran; Quoc-Khanh Dang; Ngoc Duy Tran; Thi Thoa Mac; Anh Nguyen; Xuan-Thuan Nguyen; Tung D. Ta
>
> **摘要:** Grasping objects with diverse mechanical properties, such as heavy, slippery, or fragile items, remains a significant challenge in robotics. Conventional rigid grippers typically rely on increasing the normal forces to secure an object, however, this can cause damage to fragile objects due to excessive force. To address this limitation, we propose a soft rigid hybrid gripper finger that combines rigid structural shells with soft, inflatable silicone pockets, which could be integrated into a conventional gripper. The hybrid gripper can actively modulate its surface friction by varying the internal air pressure of the silicone pockets, enabling the gripper to securely grasp objects without increasing the gripping force. This is demonstrated by fundamental experimental results, in which an increase in internal pressure leads to a proportional increase in the effective coefficient of friction. The gripping experiments also show that the integrated gripper can stably lift heavy and slippery objects or fragile, deformable objects, such as eggs, tofu, fruits, and paper cups, with minimal damage by increasing friction rather than applying high force.
>
---
#### [new 068] Perceptive Variable-Timing Footstep Planning for Humanoid Locomotion on Disconnected Footholds
- **分类: cs.RO**

- **简介: 该论文属于双足机器人步态规划任务，解决在不连续支撑点上安全行走的问题。通过混合整数预测控制，联合规划脚位与步长时间，提升动态一致性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07400](https://arxiv.org/pdf/2603.07400)**

> **作者:** Zhaoyang Xiang; Upama Pant; Ayonga Hereid
>
> **备注:** 8 pages, 5 figures, 1 table, 3 algorithms. Supplemental video at: this https URL
>
> **摘要:** Many real-world walking scenarios contain obstacles and unsafe ground patches (e.g., slippery or cluttered areas), leaving a disconnected set of admissible footholds that can be modeled as stepping-stone-like regions. We propose an onboard, perceptive mixed-integer model predictive control framework that jointly plans foot placement and step duration using step-to-step Divergent Component of Motion (DCM) dynamics. Ego-centric depth images are fused into a probabilistic local heightmap, from which we extract a union of convex steppable regions. Region membership is enforced with binary variables in a mixed-integer quadratic program (MIQP). To keep the optimization tractable while certifying safety, we embed capturability bounds in the DCM space: a lateral one-step condition (preventing leg crossing) and a sagittal infinite-step bound that limits unstable growth. We further re-plan within the step by back-propagating the measured instantaneous DCM to update the initial DCM, improving robustness to model mismatch and external disturbances. We evaluate the approach in simulation on Digit on randomized stepping-stone fields, including external pushes. The planner generates terrain-aware, dynamically consistent footstep sequences with adaptive timing and millisecond-level solve times.
>
---
#### [new 069] Long-Short Term Agents for Pure-Vision Bronchoscopy Robotic Autonomy
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主支气管镜导航任务，解决传统导航依赖外部设备的问题。提出一种纯视觉框架，结合长短时代理实现自主导航，验证其在多种模型中的可行性。**

- **链接: [https://arxiv.org/pdf/2603.07909](https://arxiv.org/pdf/2603.07909)**

> **作者:** Junyang Wu; Mingyi Luo; Fangfang Xie; Minghui Zhang; Hanxiao Zhang; Chunxi Zhang; Junhao Wang; Jiayuan Sun; Yun Gu; Guang-Zhong Yang
>
> **摘要:** Accurate intraoperative navigation is essential for robot-assisted endoluminal intervention, but remains difficult because of limited endoscopic field of view and dynamic artifacts. Existing navigation platforms often rely on external localization technologies, such as electromagnetic tracking or shape sensing, which increase hardware complexity and remain vulnerable to intraoperative anatomical mismatch. We present a vision-only autonomy framework that performs long-horizon bronchoscopic navigation using preoperative CT-derived virtual targets and live endoscopic video, without external tracking during navigation. The framework uses hierarchical long-short agents: a short-term reactive agent for continuous low-latency motion control, and a long-term strategic agent for decision support at anatomically ambiguous points. When their recommendations conflict, a world-model critic predicts future visual states for candidate actions and selects the action whose predicted state best matches the target view. We evaluated the system in a high-fidelity airway phantom, three ex vivo porcine lungs, and a live porcine model. The system reached all planned segmental targets in the phantom, maintained 80\% success to the eighth generation ex vivo, and achieved in vivo navigation performance comparable to the expert bronchoscopist. These results support the preclinical feasibility of sensor-free autonomous bronchoscopic navigation.
>
---
#### [new 070] StructBiHOI: Structured Articulation Modeling for Long--Horizon Bimanual Hand--Object Interaction Generation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多手物体交互生成任务，解决长时序、精细关节协调和跨手协作问题。提出StructBiHOI框架，通过结构化关节规划与帧级优化实现稳定、真实的双臂操作生成。**

- **链接: [https://arxiv.org/pdf/2603.08390](https://arxiv.org/pdf/2603.08390)**

> **作者:** Zhi Wang; Liu Liu; Ruonan Liu; Dan Guo; Meng Wang
>
> **摘要:** Recent progress in 3D hand--object interaction (HOI) generation has primarily focused on single--hand grasp synthesis, while bimanual manipulation remains significantly more challenging. Long--horizon planning instability, fine--grained joint articulation, and complex cross--hand coordination make coherent bimanual generation difficult, especially under multimodal conditions. Existing approaches often struggle to simultaneously ensure temporal consistency, physical plausibility, and semantic alignment over extended sequences. We propose StructBiHOI, a Structured articulation modeling framework for long-horizon Bimanual HOI generation. Our key insight is to structurally disentangle temporal joint planning from frame--level manipulation refinement. Specifically, a jointVAE models long-term joint evolution conditioned on object geometry and task semantics, while a maniVAE refines fine-grained hand poses at the single--frame level. To enable stable and efficient long--sequence generation, we incorporate a state--space--inspired diffusion denoiser based on Mamba, which models long--range dependencies with linear complexity. This hierarchical design facilitates coherent dual-hand coordination and articulated object interaction. Extensive experiments on bimanual manipulation and single-hand grasping benchmarks demonstrate that our method achieves superior long--horizon stability, motion realism, and computational efficiency compared to strong baselines.
>
---
#### [new 071] Is Your Safe Controller Actually Safe? A Critical Review of CBF Tautologies and Hidden Assumptions
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全控制任务，旨在解决CBF理论与实际应用间的差距。通过分析CBF的假设与限制，指出其在输入约束系统中的局限性，并提供构建有效安全论证的指导。**

- **链接: [https://arxiv.org/pdf/2603.06954](https://arxiv.org/pdf/2603.06954)**

> **作者:** Taekyung Kim
>
> **备注:** Interactive web demo: this https URL
>
> **摘要:** This tutorial provides a critical review of the practical application of Control Barrier Functions (CBFs) in robotic safety. While the theoretical foundations of CBFs are well-established, I identify a recurring gap between the mathematical assumption of a safe controller's existence and its constructive realization in systems with input constraints. I highlight the distinction between candidate and valid CBFs by analyzing the interplay of system dynamics, actuation limits, and class-K functions. I further show that some purported demonstrations of safe robot policies or controllers are limited to passively safe systems, such as single integrators or kinematic manipulators, where safety is already inherited from the underlying physics and even naive geometric hard constraints suffice to prevent collisions. By revisiting simple low-dimensional examples, I show when CBF formulations provide valid safety guarantees and when they fail due to common misuses. I then provide practical guidelines for constructing realizable safety arguments for systems without such passive safety. The goal of this tutorial is to bridge the gap between theoretical guarantees and actual implementation, supported by an open-source interactive web demonstration that visualizes these concepts intuitively.
>
---
#### [new 072] STRIDE: Structured Lagrangian and Stochastic Residual Dynamics via Flow Matching
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出STRIDE框架，用于机器人动力学建模。任务是提升不确定环境下的预测准确性。通过分离刚体力学与非保守交互，结合LNN和CFM，减少预测误差。**

- **链接: [https://arxiv.org/pdf/2603.08478](https://arxiv.org/pdf/2603.08478)**

> **作者:** Prakrut Kotecha; Ganga Nair B; Shishir Kolathaya
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Robotic systems operating in unstructured environments must operate under significant uncertainty arising from intermittent contacts, frictional variability, and unmodeled compliance. While recent model-free approaches have demonstrated impressive performance, many deployment settings still require predictive models that support planning, constraint handling, and online adaptation. Analytical rigid-body models provide strong physical structure but often fail to capture complex interaction effects, whereas purely data-driven models may violate physical consistency, exhibit data bias, and accumulate long-horizon drift. In this work, we propose STRIDE, a dynamics learning framework that explicitly separates conservative rigid-body mechanics from uncertain, effectively stochastic non-conservative interaction effects. The structured component is modeled using a Lagrangian Neural Network (LNN) to preserve energy-consistent inertial dynamics, while residual interaction forces are represented using Conditional Flow Matching (CFM) to capture multi-modal interaction phenomena. The two components are trained jointly end-to-end, enabling the model to retain physical structure while representing complex stochastic behavior. We evaluate STRIDE on systems of increasing complexity, including a pendulum, the Unitree Go1 quadruped, and the Unitree G1 humanoid. Results show 20% reduction in long-horizon prediction error and 30% reduction in contact force prediction error compared to deterministic residual baselines, supporting more reliable model-based control in uncertain robotic environments.
>
---
#### [new 073] UniUncer: Unified Dynamic Static Uncertainty for End to End Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决E2E驾驶中的不确定性问题。提出UniUncer框架，统一建模静态与动态场景的不确定性，提升规划可靠性。**

- **链接: [https://arxiv.org/pdf/2603.07686](https://arxiv.org/pdf/2603.07686)**

> **作者:** Yu Gao; Jijun Wang; Zongzheng Zhang; Anqing Jiang; Yiru Wang; Yuwen Heng; Shuo Wang; Hao Sun; Zhangfeng Hu; Hao Zhao
>
> **备注:** ICRA 2026
>
> **摘要:** End-to-end (E2E) driving has become a cornerstone of both industry deployment and academic research, offering a single learnable pipeline that maps multi-sensor inputs to actions while avoiding hand-engineered modules. However, the reliability of such pipelines strongly depends on how well they handle uncertainty: sensors are noisy, semantics can be ambiguous, and interaction with other road users is inherently stochastic. Uncertainty also appears in multiple forms: classification vs. localization, and, crucially, in both static map elements and dynamic agents. Existing E2E approaches model only static-map uncertainty, leaving planning vulnerable to overconfident and unreliable inputs. We present UniUncer, the first lightweight, unified uncertainty framework that jointly estimates and uses uncertainty for both static and dynamic scene elements inside an E2E planner. Concretely: (1) we convert deterministic heads to probabilistic Laplace regressors that output per-vertex location and scale for vectorized static and dynamic entities; (2) we introduce an uncertainty-fusion module that encodes these parameters and injects them into object/map queries to form uncertainty-aware queries; and (3) we design an uncertainty-aware gate that adaptively modulates reliance on historical inputs (ego status or temporal perception queries) based on current uncertainty levels. The design adds minimal overhead and drops throughput by only $\sim$0.5 FPS while remaining plug-and-play for common E2E backbones. On nuScenes (open-loop), UniUncer reduces average L2 trajectory error by 7\%. On NavsimV2 (pseudo closed-loop), it improves overall EPDMS by 10.8\% and notable stage two gains in challenging, interaction-heavy scenes. Ablations confirm that dynamic-agent uncertainty and the uncertainty-aware gate are both necessary.
>
---
#### [new 074] Rethinking the semantic classification of indoor places by mobile robots
- **分类: cs.RO**

- **简介: 该论文属于室内场景语义分类任务，旨在解决服务机器人对环境理解不足的问题。通过放松房间内区域的标签区分，提高机器人搜索物体的效率。**

- **链接: [https://arxiv.org/pdf/2603.08512](https://arxiv.org/pdf/2603.08512)**

> **作者:** Oscar Martinez Mozos; Alejandra C. Hernandez; Clara Gomez; Ramon Barber
>
> **备注:** Presented at the Workshop on Semantic Scene Understanding for Human Robot Interaction, in the ACM/IEEE International Conference on Human-Robot Interaction (HRI), Stockholm, Sweden, 2023
>
> **摘要:** A significant challenge in service robots is the semantic understanding of their surrounding areas. Traditional approaches addressed this problem by segmenting the floor plan into regions corresponding to full rooms that are assigned labels consistent with human perception, e.g. office or kitchen. However, different areas inside the same room can be used in different ways: Could the table and the chair in my kitchen become my office? What is the category of that area now? office or kitchen? To adapt to these circumstances we propose a new paradigm where we intentionally relax the resulting labeling of semantic classifiers by allowing confusions inside rooms. Our hypothesis is that those confusions can be beneficial to a service robot. We present a proof of concept in the task of searching for objects.
>
---
#### [new 075] Nonlinear Performance Degradation of Vision-Based Teleoperation under Network Latency
- **分类: cs.RO**

- **简介: 该论文研究视觉遥操作在网络延迟下的非线性性能退化问题，通过实验平台LAVT分析延迟对闭环稳定性的影响，旨在为未来延迟补偿提供基准。**

- **链接: [https://arxiv.org/pdf/2603.06850](https://arxiv.org/pdf/2603.06850)**

> **作者:** Aws Khalil; Jaerock Kwon
>
> **摘要:** Teleoperation is increasingly being adopted as a critical fallback for autonomous vehicles. However, the impact of network latency on vision-based, perception-driven control remains insufficiently studied. The present work investigates the nonlinear degradation of closed-loop stability in camera-based lane keeping under varying network delays. To conduct this study, we developed the Latency-Aware Vision Teleoperation testbed (LAVT), a research-oriented ROS 2 framework that enables precise, distributed one-way latency measurement and reproducible delay injection. Using LAVT, we performed 180 closed-loop experiments in simulation across diverse road geometries. Our findings reveal a sharp collapse in stability between 150 ms and 225 ms of one-way perception latency, where route completion rates drop from 100% to below 50% as oscillatory instability and phase-lag effects emerge. We further demonstrate that additional control-channel delay compounds these effects, significantly accelerating system failure even under constant visual latency. By combining this systematic empirical characterization with the LAVT testbed, this work provides quantitative insights into perception-driven instability and establishes a reproducible baseline for future latency-compensation and predictive control strategies. Project page, supplementary video, and code are available at this https URL
>
---
#### [new 076] RoboPCA: Pose-centered Affordance Learning from Human Demonstrations for Robot Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboPCA，解决机器人操作中空间可操作性预测问题，通过联合预测接触区域和姿态，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.07691](https://arxiv.org/pdf/2603.07691)**

> **作者:** Zhanqi Xiao; Ruiping Wang; Xilin Chen
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Understanding spatial affordances -- comprising the contact regions of object interaction and the corresponding contact poses -- is essential for robots to effectively manipulate objects and accomplish diverse tasks. However, existing spatial affordance prediction methods mainly focus on locating the contact regions while delegating the pose to independent pose estimation approaches, which can lead to task failures due to inconsistencies between predicted contact regions and candidate poses. In this work, we propose RoboPCA, a pose-centered affordance prediction framework that jointly predicts task-appropriate contact regions and poses conditioned on instructions. To enable scalable data collection for pose-centered affordance learning, we devise Human2Afford, a data curation pipeline that automatically recovers scene-level 3D information and infers pose-centered affordance annotations from human demonstrations. With Human2Afford, scene depth and the interaction object's mask are extracted to provide 3D context and object localization, while pose-centered affordance annotations are obtained by tracking object points within the contact region and analyzing hand-object interaction patterns to establish a mapping from the 3D hand mesh to the robot end-effector orientation. By integrating geometry-appearance cues through an RGB-D encoder and incorporating mask-enhanced features to emphasize task-relevant object regions into the diffusion-based framework, RoboPCA outperforms baseline methods on image datasets, simulation, and real robots, and exhibits strong generalization across tasks and categories.
>
---
#### [new 077] HybridMimic: Hybrid RL-Centroidal Control for Humanoid Motion Mimicking
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动模仿任务，解决RL在动态环境中的物理可行性问题。提出HybridMimic框架，结合模型预测与强化学习，提升控制精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.06775](https://arxiv.org/pdf/2603.06775)**

> **作者:** Ludwig Chee-Ying Tay; I-Chia Chang; Yan Gu
>
> **摘要:** Motion mimicking, i.e., encouraging the control policy to mimic human motion, facilitates the learning of complex tasks via reinforcement learning (RL) for humanoid robots. Although standard RL frameworks demonstrate impressive locomotion agility, they often bypass explicit reasoning about robot dynamics during deployment, which is a design choice that can lead to physically infeasible commands when the robot encounters out-of-distribution environments. By integrating model-based principles, hybrid approaches can improve performance; however, existing methods typically rely on predefined contact timing, limiting their versatility. This paper introduces HybridMimic, a framework in which a learned policy dynamically modulates a centroidal-model-based controller by predicting continuous contact states and desired centroidal velocities. This architecture exploits the physical grounding of centroidal dynamics to generate feedforward torques that remain feasible even under domain shift. Using physics-informed rewards, the policy is trained to efficiently utilize the centroidal controller's optimization by outputting precise control targets and reference torques. Through hardware experiments on the Booster T1 humanoid, HybridMimic reduces the average base position tracking error by 13\% compared to a state-of-the-art RL baseline, demonstrating the robustness of dynamics-aware deployment.
>
---
#### [new 078] Adaptive Vision-Based Control of Redundant Robots with Null-Space Interaction for Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文属于人机协作控制任务，旨在解决冗余机器人在未知环境中的安全有效协作问题。提出一种基于视觉的自适应控制与空域交互方法，提升协作性能。**

- **链接: [https://arxiv.org/pdf/2603.08089](https://arxiv.org/pdf/2603.08089)**

> **作者:** Xiangjie Yan; Chen Chen; Xiang Li
>
> **摘要:** Human-robot collaboration aims to extend human ability through cooperation with robots. This technology is currently helping people with physical disabilities, has transformed the manufacturing process of companies, improved surgical performance, and will likely revolutionize the daily lives of everyone in the future. Being able to enhance the performance of both sides, such that human-robot collaboration outperforms a single robot/human, remains an open issue. For safer and more effective collaboration, a new control scheme has been proposed for redundant robots in this paper, consisting of an adaptive vision-based control term in task space and an interactive control term in null space. Such a formulation allows the robot to autonomously carry out tasks in an unknown environment without prior calibration while also interacting with humans to deal with unforeseen changes (e.g., potential collision, temporary needs) under the redundant configuration. The decoupling between task space and null space helps to explore the collaboration safely and effectively without affecting the main task of the robot end-effector. The stability of the closed-loop system has been rigorously proved with Lyapunov methods, and both the convergence of the position error in task space and that of the damping model in null space are guaranteed. The experimental results of a robot manipulator guided with the technology of augmented reality (AR) are presented to illustrate the performance of the control scheme.
>
---
#### [new 079] VLN-Cache: Enabling Token Caching for VLN Models with Visual/Semantic Dynamics Awareness
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉语言导航任务，解决大模型推理成本高的问题。提出VLN-Cache框架，通过视觉和语义动态感知实现高效token缓存，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2603.07080](https://arxiv.org/pdf/2603.07080)**

> **作者:** Zihao Zheng; Zhihao Mao; Xingyue Zhou; Jiayu Chen; Maoliang Li; Xinhao Sun; Hailong Zou; Zhaobo Zhang; Xuanzhe Liu; Donggang Cao; Hong Mei; Xiang Chen
>
> **摘要:** Vision-and-Language Navigation (VLN) increasingly relies on large vision-language models, but their inference cost conflicts with real-time deployment. Token caching is a promising training-free strategy that avoids redundant computation by reusing stable visual tokens across frames. However, existing methods assume a static camera and fixed semantic focus, assumptions that VLN fundamentally violates. We identify two failure modes: (1) visual dynamics, where viewpoint shift displaces token positions across frames, causing position-wise matching to pair misaligned content; (2) semantic dynamics, where token relevance shifts across task stages as navigation progresses, making cached states stale. We propose VLN-Cache, a visual-dynamic-aware and semantic-dynamic-aware caching framework that introduces view-aligned remapping to recover geometric correspondences and a task-relevance saliency filter to veto reuse at semantic transitions. A layer-adaptive entropy policy further balances the per-layer reuse budget. Experiments on the R2R-CE simulation benchmark show up to 1.52x speedup while maintaining competitive navigation success rates.
>
---
#### [new 080] CRED: Counterfactual Reasoning and Environment Design for Active Preference Learning
- **分类: cs.RO**

- **简介: 该论文属于主动偏好学习任务，旨在解决人工编码反馈不可行的问题。提出CRED方法，通过环境设计和反事实推理生成轨迹对，提升奖励函数推断的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2603.08531](https://arxiv.org/pdf/2603.08531)**

> **作者:** Yi-Shiuan Tung; Gyanig Kumar; Wei Jiang; Bradley Hayes; Alessandro Roncone
>
> **备注:** IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** As a robot's operational environment and tasks to perform within it grow in complexity, the explicit specification and balancing of optimization objectives to achieve a preferred behavior profile moves increasingly farther out of reach. These systems benefit strongly by being able to align their behavior to reflect human preferences and respond to corrections, but manually encoding this feedback is infeasible. Active preference learning (APL) learns human reward functions by presenting trajectories for ranking. However, existing methods sample from fixed trajectory sets or replay buffers that limit query diversity and often fail to identify informative comparisons. We propose CRED, a novel trajectory generation method for APL that improves reward inference by jointly optimizing environment design and trajectory selection to efficiently query and extract preferences from users. CRED "imagines" new scenarios through environment design and leverages counterfactual reasoning -- by sampling possible rewards from its current belief and asking "What if this were the true preference?" -- to generate trajectory pairs that expose differences between competing reward functions. Comprehensive experiments and a user study show that CRED significantly outperforms state-of-the-art methods in reward accuracy and sample efficiency and receives higher user ratings.
>
---
#### [new 081] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手势生成任务，旨在解决3D物体与文本指令间模态差异大、抓取不物理有效的问题。提出AffordGrasp框架，结合语义与空间约束，生成精确且合理的抓取姿态。**

- **链接: [https://arxiv.org/pdf/2603.08021](https://arxiv.org/pdf/2603.08021)**

> **作者:** Xiaofei Wu; Yi Zhang; Yumeng Liu; Yuexin Ma; Yujiao Shi; Xuming He
>
> **摘要:** Generating human grasping poses that accurately reflect both object geometry and user-specified interaction semantics is essential for natural hand-object interactions in AR/VR and embodied AI. However, existing semantic grasping approaches struggle with the large modality gap between 3D object representations and textual instructions, and often lack explicit spatial or semantic constraints, leading to physically invalid or semantically inconsistent grasps. In this work, we present AffordGrasp, a diffusion-based framework that produces physically stable and semantically faithful human grasps with high precision. We first introduce a scalable annotation pipeline that automatically enriches hand-object interaction datasets with fine-grained structured language labels capturing interaction intent. Building upon these annotations, AffordGrasp integrates an affordance-aware latent representation of hand poses with a dual-conditioning diffusion process, enabling the model to jointly reason over object geometry, spatial affordances, and instruction semantics. A distribution adjustment module further enforces physical contact consistency and semantic alignment. We evaluate AffordGrasp across four instruction-augmented benchmarks derived from HO-3D, OakInk, GRAB, and AffordPose, and observe substantial improvements over state-of-the-art methods in grasp quality, semantic accuracy, and diversity.
>
---
#### [new 082] An Open-Source Robotics Research Platform for Autonomous Laparoscopic Surgery
- **分类: cs.RO**

- **简介: 该论文属于自主腹腔镜手术领域，解决传统平台机械限制问题，提出一种开源RCM控制器，实现高精度手术操作。**

- **链接: [https://arxiv.org/pdf/2603.08490](https://arxiv.org/pdf/2603.08490)**

> **作者:** Ariel Rodriguez; Lorenzo Mazza; Martin Lelis; Rayan Younis; Sebastian Bodenstedt; Martin Wagner; Stefanie Speidel
>
> **备注:** Submitted to iROS 2026
>
> **摘要:** Autonomous robot-assisted surgery demands reliable, high-precision platforms that strictly adhere to the safety and kinematic constraints of minimally invasive procedures. Existing research platforms, primarily based on the da Vinci Research Kit, suffer from cable-driven mechanical limitations that degrade state-space consistency and hinder the downstream training of reliable autonomous policies. We present an open-source, robot-agnostic Remote Center of Motion (RCM) controller based on a closed-form analytical velocity solver that enforces the trocar constraint deterministically without iterative optimization. The controller operates in Cartesian space, enabling any industrial manipulator to function as a surgical robot. We provide implementations for the UR5e and Franka Emika Panda manipulators, and integrate stereoscopic 3D perception. We integrate the robot control into a full-stack ROS-based surgical robotics platform supporting teleoperation, demonstration recording, and deployment of learned policies via a decoupled server-client architecture. We validate the system on a bowel grasping and retraction task across phantom, ex vivo, and in vivo porcine laparoscopic procedures. RCM deviations remain sub-millimeter across all conditions, and trajectory smoothness metrics (SPARC, LDLJ) are comparable to expert demonstrations from the JIGSAWS benchmark recorded on the da Vinci system. These results demonstrate that the platform provides the precision and robustness required for teleoperation, data collection and autonomous policy deployment in realistic surgical scenarios.
>
---
#### [new 083] R2F: Repurposing Ray Frontiers for LLM-free Object Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于室内零样本物体导航任务，解决大模型推理带来的延迟问题。提出R2F框架，利用射线前沿实现无需大模型的实时导航。**

- **链接: [https://arxiv.org/pdf/2603.08475](https://arxiv.org/pdf/2603.08475)**

> **作者:** Francesco Argenziano; John Mark Alexis Marcelo; Michele Brienza; Abdel Hakim Drid; Emanuele Musumeci; Daniele Nardi; Domenico D. Bloisi; Vincenzo Suriani
>
> **摘要:** Zero-shot open-vocabulary object navigation has progressed rapidly with the emergence of large Vision-Language Models (VLMs) and Large Language Models (LLMs), now widely used as high-level decision-makers instead of end-to-end policies. Although effective, such systems often rely on iterative large-model queries at inference time, introducing latency and computational overhead that limit real-time deployment. To address this problem, we repurpose ray frontiers (R2F), a recently proposed frontier-based exploration paradigm, to develop an LLM-free framework for indoor open-vocabulary object navigation. While ray frontiers were originally used to bias exploration using semantic cues carried along rays, we reinterpret frontier regions as explicit, direction-conditioned semantic hypotheses that serve as navigation goals. Language-aligned features accumulated along out-of-range rays are stored sparsely at frontiers, where each region maintains multiple directional embeddings encoding plausible unseen content. In this way, navigation then reduces to embedding-based frontier scoring and goal tracking within a classical mapping and planning pipeline, eliminating iterative large-model reasoning. We further introduce R2F-VLN, a lightweight extension for free-form language instructions using syntactic parsing and relational verification without additional VLM or LLM components. Experiments in Habitat-sim and on a real robotic platform demonstrate competitive state-of-the-art zero-shot performance with real-time execution, achieving up to 6 times faster runtime than VLM-based alternatives.
>
---
#### [new 084] 3PoinTr: 3D Point Tracks for Robot Manipulation Pretraining from Casual Videos
- **分类: cs.RO**

- **简介: 该论文提出3PoinTr，用于从人类视频中预训练机器人操作策略，解决数据效率和身体差异问题。通过3D点轨迹实现跨身体表征，提升操作任务的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.08485](https://arxiv.org/pdf/2603.08485)**

> **作者:** Adam Hung; Bardienus Pieter Duisterhof; Jeffrey Ichnowski
>
> **摘要:** Data-efficient training of robust robot policies is the key to unlocking automation in a wide array of novel tasks. Current systems require large volumes of demonstrations to achieve robustness, which is impractical in many applications. Learning policies directly from human videos is a promising alternative that removes teleoperation costs, but it shifts the challenge toward overcoming the embodiment gap (differences in kinematics and strategies between robots and humans), often requiring restrictive and carefully choreographed human motions. We propose 3PoinTr, a method for pretraining robot policies from casual and unconstrained human videos, enabling learning from motions natural for humans. 3PoinTr uses a transformer architecture to predict 3D point tracks as an intermediate embodiment-agnostic representation. 3D point tracks encode goal specifications, scene geometry, and spatiotemporal relationships. We use a Perceiver IO architecture to extract a compact representation for sample-efficient behavior cloning, even when point tracks violate downstream embodiment-specific constraints. We conduct thorough evaluation on simulated and real-world tasks, and find that 3PoinTr achieves robust spatial generalization on diverse categories of manipulation tasks with only 20 action-labeled robot demonstrations. 3PoinTr outperforms the baselines, including behavior cloning methods, as well as prior methods for pretraining from human videos. We also provide evaluations of 3PoinTr's 3D point track predictions compared to an existing point track prediction baseline. We find that 3PoinTr produces more accurate and higher quality point tracks due to a lightweight yet expressive architecture built on a single transformer, in addition to a training formulation that preserves supervision of partially occluded points. Project page: this https URL.
>
---
#### [new 085] A Multi-Layer Sim-to-Real Framework for Gaze-Driven Assistive Neck Exoskeletons
- **分类: cs.RO**

- **简介: 该论文属于辅助机器人控制任务，旨在解决因神经疾病导致的头部下垂问题。通过VR收集眼动与头动数据，训练基于眼动预测头部运动的模型，并提出多层控制器选择框架，提升辅助颈外骨骼的个性化控制性能。**

- **链接: [https://arxiv.org/pdf/2603.06779](https://arxiv.org/pdf/2603.06779)**

> **作者:** Colin Rubow; Eric Brewer; Ian Bales; Haohan Zhang; Daniel S. Brown
>
> **备注:** IEEE International Conference on Robotics & Automation (ICRA), 2026. Equal Contribution from the first two authors
>
> **摘要:** Dropped head syndrome, caused by neck muscle weakness from neurological diseases, severely impairs an individual's ability to support and move their head, causing pain and making everyday tasks challenging. Our long-term goal is to develop an assistive powered neck exoskeleton that restores natural movement. However, predicting a user's intended head movement remains a key challenge. We leverage virtual reality (VR) to collect coupled eye and head movement data from healthy individuals to train models capable of predicting head movement based solely on eye gaze. We also propose a novel multi-layer controller selection framework, where head control strategies are evaluated across decreasing levels of abstraction -- from simulation and VR to a physical neck exoskeleton. This pipeline effectively rejects poor-performing controllers early, identifying two novel gaze-driven models that achieve strong performance when deployed on the physical exoskeleton. Our results reveal that no single controller is universally preferred, highlighting the necessity for personalization in gaze-driven assistive control. Our work demonstrates the utility of VR-based evaluation for accelerating the development of intuitive, safe, and personalized assistive robots.
>
---
#### [new 086] ACLM: ADMM-Based Distributed Model Predictive Control for Collaborative Loco-Manipulation
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出一种基于ADMM的分布式模型预测控制框架，用于多足机器人协同搬运任务，解决动态耦合与实时性问题，通过分解优化问题实现高效协作。**

- **链接: [https://arxiv.org/pdf/2603.07095](https://arxiv.org/pdf/2603.07095)**

> **作者:** Ziyi Zhou; Pengyuan Shu; Ruize Cao; Yuntian Zhao; Ye Zhao
>
> **摘要:** Collaborative transportation of heavy payloads via loco-manipulation is a challenging yet essential capability for legged robots operating in complex, unstructured environments. Centralized planning methods, e.g., holistic trajectory optimization, capture dynamic coupling among robots and payloads but scale poorly with system size, limiting real-time applicability. In contrast, hierarchical and fully decentralized approaches often neglect force and dynamic interactions, leading to conservative behavior. This study proposes an Alternating Direction Method of Multipliers (ADMM)-based distributed model predictive control framework for collaborative loco-manipulation with a team of quadruped robots with manipulators. By exploiting the payload-induced coupling structure, the global optimal control problem is decomposed into parallel individual-robot-level subproblems with consensus constraints. The distributed planner operates in a receding-horizon fashion and achieves fast convergence, requiring only a few ADMM iterations per planning cycle. A wrench-aware whole-body controller executes the planned trajectories, tracking both motion and interaction wrenches. Extensive simulations with up to four robots demonstrate scalability, real-time performance, and robustness to model uncertainty.
>
---
#### [new 087] GuideTWSI: A Diverse Tactile Walking Surface Indicator Dataset from Synthetic and Real-World Images for Blind and Low-Vision Navigation
- **分类: cs.RO; cs.HC; eess.SY**

- **简介: 该论文属于视觉导航任务，旨在解决盲人和低视力人群在城市环境中准确识别触觉步行指示器（TWSIs）的问题。研究构建了一个包含合成与真实图像的多样化TWSI数据集，以提升模型对不同类型TWSI的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.07060](https://arxiv.org/pdf/2603.07060)**

> **作者:** Hochul Hwang; Soowan Yang; Anh N. H. Nguyen; Parth Goel; Krisha Adhikari; Sunghoon I. Lee; Joydeep Biswas; Nicholas A. Giudice; Donghyun Kim
>
> **摘要:** Tactile Walking Surface Indicators (TWSIs) are safety-critical landmarks that blind and low-vision (BLV) pedestrians use to locate crossings and hazard zones. From our observation sessions with BLV guide dog handlers, trainers, and an O&M specialist, we confirmed the critical importance of reliable and accurate TWSI segmentation for navigation assistance of BLV individuals. Achieving such reliability requires large-scale annotated data. However, TWSIs are severely underrepresented in existing urban perception datasets, and even existing dedicated paving datasets are limited: they lack robot-relevant viewpoints (e.g., egocentric or top-down) and are geographically biased toward East Asian directional bars - raised parallel strips used for continuous guidance along sidewalks. This narrow focus overlooks truncated domes - rows of round bumps used primarily in North America and Europe as detectable warnings at curbs, crossings, and platform edges. As a result, models trained only on bar-centric data struggle to generalize to dome-based warnings, leading to missed detections and false stops in safety-critical environments.
>
---
#### [new 088] Model-based thermal drift compensation for high-precision hexapod robot actuators
- **分类: cs.RO**

- **简介: 该论文属于高精度机器人控制任务，旨在解决热膨胀导致的定位误差问题。通过建立温度与膨胀关系模型，实现对六足机器人执行器热漂移的补偿。**

- **链接: [https://arxiv.org/pdf/2603.07141](https://arxiv.org/pdf/2603.07141)**

> **作者:** Clément Robert; Alain Vissiere; Olivier Company; Pierre Noire; Thierry Roux; Sébastien Krut
>
> **摘要:** Thermal expansion is a significant source of positioning error in high-precision hexapod robots (Gough-Stewart platforms). Any variation in the temperature of the hexapod's parts induces expansion, which alters their kinematic model and reduces the robot's accuracy and repeatability. These variations may arise from internal heat sources (such as motors, encoders, and electronics) or from environmental changes. In this study, a method is proposed to anticipate and therefore correct the thermal drift of one of the hexapod precision electro-mechanical actuators. This method is based on determining a model that links the expansion state of the actuator at any given moment to the temperature of some well-chosen points on its surface. This model was initially developed theoretically. Its coefficients were then adjusted experimentally on a specific test-bench, based on a rigorous measurement campaign of actuator expansion using a high-precision interferometric measurement system. Experimental validation demonstrates a reduction of thermally induced expansion by more than 80%. This paves the way for thermal drift correction across the entire robot or similar robotics parts.
>
---
#### [new 089] Gradient-based Nested Co-Design of Aerodynamic Shape and Control for Winged Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于飞行器设计任务，解决气动外形与控制协同优化问题。提出一种基于梯度的嵌套协同设计框架，联合优化气动形状和运动规划，提升飞行性能。**

- **链接: [https://arxiv.org/pdf/2603.06760](https://arxiv.org/pdf/2603.06760)**

> **作者:** Daniele Affinita; Mingda Xu; Benoît Valentin Gherardi; Pascal Fua
>
> **摘要:** Designing aerial robots for specialized tasks, from perching to payload delivery, requires tailoring their aerodynamic shape to specific mission requirements. For tasks involving wide flight envelopes, the usual sequential process of first determining the shape and then the motion planner is likely to be suboptimal due to the inherent nonlinear interactions between them. This limitation has been motivating co-design research, which involves jointly optimizing the aerodynamic shape and the motion planner. In this paper, we present a general-purpose, gradient-based, nested co-design framework where the motion planner solves an optimal control problem and the aerodynamic forces used in the dynamics model are determined by a neural surrogate model. This enables us to model complex subsonic flow conditions encountered in aerial robotics and to overcome the limited applicability of existing co-design methods. These limitations stem from the simplifying assumptions they require for computational tractability to either the planner or the aerodynamics. We validate our method on two complex dynamic tasks for fixed-wing gliders: perching and a short landing. Our optimized designs improve task performance compared to an evolutionary baseline in a fraction of the computation time.
>
---
#### [new 090] FoMo: A Multi-Season Dataset for Robot Navigation in Forêt Montmorency
- **分类: cs.RO**

- **简介: 该论文提出FoMo数据集，用于机器人在森林环境中的导航任务。旨在解决季节变化对定位与建图方法的影响问题，通过多季节数据评估算法鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.08433](https://arxiv.org/pdf/2603.08433)**

> **作者:** Matěj Boxan; Gabriel Jeanson; Alexander Krawciw; Effie Daum; Xinyuan Qiao; Sven Lilge; Timothy D. Barfoot; François Pomerleau
>
> **摘要:** The Forêt Montmorency (FoMo) dataset is a comprehensive multi-season data collection, recorded over the span of one year in a boreal forest. Featuring a unique combination of on- and off-pavement environments with significant environmental changes, the dataset challenges established odometry and SLAM pipelines. Some highlights of the data include the accumulation of snow exceeding 1 m, significant vegetation growth in front of sensors, and operations at the traction limits of the platform. In total, the FoMo dataset includes over 64 km of six diverse trajectories, repeated during 12 deployments throughout the year. The dataset features data from one rotating and one hybrid solid-state lidar, a Frequency Modulated Continuous Wave (FMCW) radar, full-HD images from a stereo camera and a wide lens monocular camera, as well as data from two IMUs. Ground Truth is calculated by post-processing three GNSS receivers mounted on the Uncrewed Ground Vehicle (UGV) and a static GNSS base station. Additional metadata, such as one measurement per minute from an on-site weather station, camera calibration intrinsics, and vehicle power consumption, is available for all sequences. To highlight the relevance of the dataset, we performed a preliminary evaluation of the robustness of a lidar-inertial, radar-gyro, and a visual-inertial localization and mapping techniques to seasonal changes. We show that seasonal changes have serious effects on the re-localization capabilities of the state-of-the-art methods. The dataset and development kit are available at this https URL.
>
---
#### [new 091] AeroPlace-Flow: Language-Grounded Object Placement for Aerial Manipulators via Visual Foresight and Object Flow
- **分类: cs.RO**

- **简介: 该论文提出AeroPlace-Flow，解决空中机械臂语言引导的物体放置问题。通过视觉预判和几何推理，实现无需预设姿态的精准放置。**

- **链接: [https://arxiv.org/pdf/2603.07744](https://arxiv.org/pdf/2603.07744)**

> **作者:** Sarthak Mishra; Rishabh Dev Yadav; Naveen Nair; Wei Pan; Spandan Roy
>
> **摘要:** Precise object placement remains underexplored in aerial manipulation, where most systems rely on predefined target coordinates and focus primarily on grasping and control. Specifying exact placement poses, however, is cumbersome in real-world settings, where users naturally communicate goals through language. In this work, we present AeroPlace-Flow, a training-free framework for language-grounded aerial object placement that unifies visual foresight with explicit 3D geometric reasoning and object flow. Given RGB-D observations of the object and the placement scene, along with a natural language instruction, AeroPlace-Flow first synthesizes a task-complete goal image using image editing models. The imagined configuration is then grounded into metric 3D space through depth alignment and object-centric reasoning, enabling the inference of a collision-aware object flow that transports the grasped object to a language and contact-consistent placement configuration. The resulting motion is executed via standard trajectory tracking for an aerial manipulator. AeroPlace-Flow produces executable placement targets without requiring predefined poses or task-specific training. We validate our approach through extensive simulation and real-world experiments, demonstrating reliable language-conditioned placement across diverse aerial scenarios with an average success rate of 75% on hardware.
>
---
#### [new 092] Physics-infused Learning for Aerial Manipulator in Winds and Near-Wall Environments
- **分类: cs.RO**

- **简介: 该论文属于空中机械臂控制任务，解决风场和近墙环境下的扰动问题。融合物理模型与学习方法，提升轨迹跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.07826](https://arxiv.org/pdf/2603.07826)**

> **作者:** Yiming Zhang; Junyi Geng
>
> **摘要:** Aerial manipulation (AM) expands UAV capabilities beyond passive observation to contact-based operations at high altitudes and in otherwise inaccessible environments. Although recent advances show promise, most AM systems are developed in controlled settings that overlook key aerodynamic effects. Simplified thrust models are often insufficient to capture the nonlinear wind disturbances and proximity-induced flow variations present in real-world environments near infrastructure, while high-fidelity CFD methods remain impractical for real-time use. Learning-based models are computationally efficient at inference, but often struggle to generalize to unseen condition. This paper combines both approaches by integrating a physics-based blade-element model with a learning-based residual force estimator, along with a rotor-speed allocation strategy for disturbance compensation, resulting in a unified control framework. The blade-element model computes per-rotor aerodynamic forces under wind and provides a refined feedforward disturbance estimate. A learning-based estimator then predicts the residual forces not captured by the model, enabling compensation for unmodeled aerodynamic effects. An online adaptation mechanism further updates the residual-force prediction and rotor-speed allocation jointly to reduce the mismatch between desired and realized thrust. We evaluate this framework in both free-flight and wall-contact tracking tasks in a simulated near-wall wind environment. Results demonstrate improved disturbance estimation and trajectory-tracking accuracy over conventional approaches, enabling robust wall-contact execution under challenging aerodynamic conditions.
>
---
#### [new 093] Underwater Embodied Intelligence for Autonomous Robots: A Constraint-Coupled Perspective on Planning, Control, and Deployment
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于水下机器人自主控制任务，旨在解决真实海洋环境中自主性受限的问题。通过约束耦合视角，融合强化学习与多机器人协作，提出提升系统鲁棒性的方法。**

- **链接: [https://arxiv.org/pdf/2603.07393](https://arxiv.org/pdf/2603.07393)**

> **作者:** Jingzehua Xu; Guanwen Xie; Jiwei Tang; Shuai Zhang; Xiaofan Li
>
> **备注:** This article is currently under review
>
> **摘要:** Autonomous underwater robots are increasingly deployed for environmental monitoring, infrastructure inspection, subsea resource exploration, and long-horizon exploration. Yet, despite rapid advances in learning-based planning and control, reliable autonomy in real ocean environments remains fundamentally constrained by tightly coupled physical limits. Hydrodynamic uncertainty, partial observability, bandwidth-limited communication, and energy scarcity are not independent challenges; they interact within the closed perception-planning-control loop and often amplify one another over time. This Review develops a constraint-coupled perspective on underwater embodied intelligence, arguing that planning and control must be understood within tightly coupled sensing, communication, coordination, and resource constraints in real ocean environments. We synthesize recent progress in reinforcement learning, belief-aware planning, hybrid control, multi-robot coordination, and foundation-model integration through this embodied perspective. Across representative application domains, we show how environmental monitoring, inspection, exploration, and cooperative missions expose distinct stress profiles of cross-layer coupling. To unify these observations, we introduce a cross-layer failure taxonomy spanning epistemic, dynamic, and coordination breakdowns, and analyze how errors cascade across autonomy layers under uncertainty. Building on this structure, we outline research directions toward physics-grounded world models, certifiable learning-enabled control, communication-aware coordination, and deployment-aware system design. By internalizing constraint coupling rather than treating it as an external disturbance, underwater embodied intelligence may evolve from performance-driven adaptation toward resilient, scalable, and verifiable autonomy under real ocean conditions.
>
---
#### [new 094] GSAT: Geometric Traversability Estimation using Self-supervised Learning with Anomaly Detection for Diverse Terrains
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，旨在解决环境可行驶性估计问题。针对传统方法依赖人工阈值和自监督学习的正样本问题，提出GSAT方法，通过异常检测实现无需额外标注的可行驶区域分类。**

- **链接: [https://arxiv.org/pdf/2603.07480](https://arxiv.org/pdf/2603.07480)**

> **作者:** Dongjin Cho; Miryeong Park; Juhui Lee; Geonmo Yang; Younggun Cho
>
> **备注:** 8 pages, 8 figures, accepted to ICRA 2026
>
> **摘要:** Safe autonomous navigation requires reliable estimation of environmental traversability. Traditional methods have relied on semantic or geometry-based approaches with human-defined thresholds, but these methods often yield unreliable predictions due to the inherent subjectivity of human supervision. While self-supervised approaches enable robots to learn from their own experience, they still face a fundamental challenge: the positive-only learning problem. To address these limitations, recent studies have employed Positive-Unlabeled (PU) learning, where the core challenge is identifying positive samples without explicit negative supervision. In this work, we propose GSAT, which addresses these limitations by constructing a positive hypersphere in latent space to classify traversable regions through anomaly detection without requiring additional prototypes (e.g., unlabeled or negative). Furthermore, our approach employs joint learning of anomaly classification and traversability prediction to more efficiently utilize robot experience. We comprehensively evaluate the proposed framework through ablation studies, validation on heterogeneous real-world robotic platforms, and autonomous navigation demonstrations in simulation environments.
>
---
#### [new 095] Cable-driven Continuum Robotics: Proprioception via Proximal-integrated Force Sensing
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决微尺度连续机器人三维力感知难题。通过集成缆绳张力与六轴力传感器，提出新型本体感觉方法。**

- **链接: [https://arxiv.org/pdf/2603.07426](https://arxiv.org/pdf/2603.07426)**

> **作者:** Gang Zhang; Junyan Yan; Jibiao Chen; Shing Shin Cheng
>
> **摘要:** Micro-scale continuum robots face significant limitations in achieving three-dimensional contact force perception, primarily due to structural miniaturization, nonlinear mechanical, and sensor integration. To overcome these limitations, this paper introduces a novel proprioception method for cable-driven continuum robots based on proximal-integrated force sensing (i.e., cable tension and six-axis force/torque (F/T) sensor), inspired by the tendon-joint collaborative sensing mechanism of the finger. By integrating biomechanically inspired design principles with nonlinear modeling, the proposed method addresses the challenge of force perception (including the three-dimensional contact force and the location of the contact point) and shape estimation in micro-scale continuum robots. First, a quasi-bionic mapping between human tissues/organs and robot components is established, enabling the transfer of the integrated sensing strategy of tendons, joints, and neural feedback to the robotic system. Second, a multimodal perception strategy is developed based on the structural constraints inherent to continuum robots. The complex relationships among mechanical and material nonlinearities, robot motion states, and contact forces are formulated as an optimization problem to reduce the perception complexity. Finally, experimental validation demonstrates the effectiveness of the proposed method. This work lays the foundation for developing safer and smarter continuum robots, enabling broader clinical adoption in complex environments.
>
---
#### [new 096] InterReal: A Unified Physics-Based Imitation Framework for Learning Human-Object Interaction Skills
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机交互任务，解决机器人在现实环境中学习精细操作技能的问题。提出InterReal框架，通过物理仿真的模仿学习提升机器人对物体的交互能力。**

- **链接: [https://arxiv.org/pdf/2603.07516](https://arxiv.org/pdf/2603.07516)**

> **作者:** Dayang Liang; Yuhang Lin; Xinzhe Liu; Jiyuan Shi; Yunlong Liu; Chenjia Bai
>
> **摘要:** Interaction is one of the core abilities of humanoid robots. However, most existing frameworks focus on non-interactive whole-body control, which limits their practical applicability. In this work, we develop InterReal, a unified physics-based imitation learning framework for Real-world human-object Interaction (HOI) control. InterReal enables humanoid robots to track HOI reference motions, facilitating the learning of fine-grained interactive skills and their deployment in real-world settings. Within this framework, we first introduce a HOI motion data augmentation scheme with hand-object contact constraints, and utilize the augmented motions to improve policy stability under object perturbations. Second, we propose an automatic reward learner to address the challenge of large-scale reward shaping. A meta-policy guided by critical tracking error metrics explores and allocates reward signals to the low-level reinforcement learning objective, which enables more effective learning of interactive policies. Experiments on HOI tasks of box-picking and box-pushing demonstrate that InterReal achieves the best tracking accuracy and the highest task success rate compared to recent baselines. Furthermore, we validate the framework on the real-world robot Unitree G1, which demonstrates its practical effectiveness and robustness beyond simulation.
>
---
#### [new 097] VertiAdaptor: Online Kinodynamics Adaptation for Vertically Challenging Terrain
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决复杂地形下车辆动力学模型适应问题。提出VertiAdaptor框架，通过融合高程与语义信息实现快速在线适应。**

- **链接: [https://arxiv.org/pdf/2603.06887](https://arxiv.org/pdf/2603.06887)**

> **作者:** Tong Xu; Chenhui Pan; Aniket Datar; Xuesu Xiao
>
> **摘要:** Autonomous driving in off-road environments presents significant challenges due to the dynamic and unpredictable nature of unstructured terrain. Traditional kinodynamic models often struggle to generalize across diverse geometric and semantic terrain types, underscoring the need for real-time adaptation to ensure safe and reliable navigation. We propose VertiAdaptor (VA), a novel online adaptation framework that efficiently integrates elevation with semantic embeddings to enable terrain-aware kinodynamic modeling and planning via function encoders. VA learns a kinodynamic space spanned by a set of neural ordinary differential equation basis functions, capturing complex vehicle-terrain interactions across varied environments. After offline training, the proposed approach can rapidly adapt to new, unseen environments by identifying kinodynamics in the learned space through a computationally efficient least-squares calculation. We evaluate VA within the Verti-Bench simulator, built on the Chrono multi-physics engine, and validate its performance both in simulation and on a physical Verti-4-Wheeler platform. Our results demonstrate that VA improves prediction accuracy by up to 23.9% and achieves a 5X faster adaptation time, advancing the robustness and reliability of autonomous robots in complex and evolving off-road environments.
>
---
#### [new 098] TacDexGrasp: Compliant and Robust Dexterous Grasping with Tactile Feedback
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决多指手在未知物体抓取中的力分布与防滑问题。通过触觉反馈和SOCP控制器，有效抑制平移和旋转滑动，提升抓取稳定性。**

- **链接: [https://arxiv.org/pdf/2603.07040](https://arxiv.org/pdf/2603.07040)**

> **作者:** Yubin Ke; Jiayi Chen; Hang Lv; Xiao Zhou; He Wang
>
> **备注:** 8pages, 7 figures
>
> **摘要:** Multi-fingered hands offer great potential for compliant and robust grasping of unknown objects, yet their high-dimensional force control presents a significant challenge. This work addresses two key problems: (1) distributing forces across multiple contacts to counteract an object's weight, and (2) preventing rotational slip caused by gravitational torque when a grasp is distant from the object's center of mass. We address these challenges via tactile feedback and a Second-Order Cone Programming (SOCP)-based controller, without explicit torque modeling or slip detection. Our key insights are (1) rotational slip inevitably induces translational slip at some contact points for a multi-fingered grasp, and (2) the ratio of tangential to normal force at each contact is an effective early stability indicator. By actively constraining this ratio for each finger below the estimated friction coefficient, our controller maintains grasp stability against both translational and rotational slip. Real-world experiments on 12 diverse objects demonstrate the robustness and compliance of our approach.
>
---
#### [new 099] SSP: Safety-guaranteed Surgical Policy via Joint Optimization of Behavioral and Spatial Constraints
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决手术中数据驱动策略的安全性问题。通过联合优化行为与空间约束，提出SSP框架确保手术政策的安全性。**

- **链接: [https://arxiv.org/pdf/2603.07032](https://arxiv.org/pdf/2603.07032)**

> **作者:** Jianshu Hu; ZhiYuan Guan; Lei Song; Kantaphat Leelakunwet; Hesheng Wang; Wei Xiao; Qi Dou; Yutong Ban
>
> **摘要:** The paradigm of robot-assisted surgery is shifting toward data-driven autonomy, where policies learned via Reinforcement Learning (RL) or Imitation Learning (IL) enable the execution of complex tasks. However, these ``black-box" policies often lack formal safety guarantees, a critical requirement for clinical deployment. In this paper, we propose the Safety-guaranteed Surgical Policy (SSP) framework to bridge the gap between data-driven generality and formal safety. We utilize Neural Ordinary Differential Equations (Neural ODEs) to learn an uncertainty-aware dynamics model from demonstration data. This learned model underpins a robust Control Barrier Function (CBF) safety controller, which minimally alters the actions of a surgical policy to ensure strict safety under uncertainty. Our controller enforces two constraint categories: behavioral constraints (restricting the task space of the agent) and spatial constraints (defining surgical no-go zones). We instantiate the SSP framework with surgical policies derived from RL, IL and Control Lyapunov Functions (CLF). Validation on in both the SurRoL simulation and da Vinci Research Kit (dVRK) demonstrates that our method achieves a near-zero constraint violation rate while maintaining high task success rates compared to unconstrained baselines.
>
---
#### [new 100] See and Switch: Vision-Based Branching for Interactive Robot-Skill Programming
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人技能编程任务，解决现实环境下的分支选择与异常检测问题。提出See & Switch框架，利用视觉实现交互式条件分支，提升机器人执行灵活性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.08057](https://arxiv.org/pdf/2603.08057)**

> **作者:** Petr Vanc; Jan Kristof Behrens; Václav Hlaváč; Karla Stepanova
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** Programming robots by demonstration (PbD) is an intuitive concept, but scaling it to real-world variability remains a challenge for most current teaching frameworks. Conditional task graphs are very expressive and can be defined incrementally, which fits very well with the PbD idea. However, acting using conditional task graphs requires reliable perception-grounded online branch selection. In this paper, we present See & Switch, an interactive teaching-and-execution framework that represents tasks as user-extendable graphs of skill parts connected via decision states (DS), enabling conditional branching during replay. Unlike prior approaches that rely on manual branching or low-dimensional signals (e.g., proprioception), our vision-based Switcher uses eye-in-hand images (high-dimensional) to select among competing successor skill parts and to detect out-of-distribution contexts that require new demonstrations. We integrate kinesthetic teaching, joystick control, and hand gestures via an input-modality-abstraction layer and demonstrate that our proposed method is teaching modality-independent, enabling efficient in-situ recovery demonstrations. The system is validated in experiments on three challenging dexterous manipulation tasks. We evaluate our method under diverse conditions and furthermore conduct user studies with 8 participants. We show that the proposed method reliably performs branch selection and anomaly detection for novice users, achieving 90.7 % and 87.9 % accuracy, respectively, across 576 real-robot rollouts. We provide all code and data required to reproduce our experiments at this http URL.
>
---
#### [new 101] AtomVLA: Scalable Post-Training for Robotic Manipulation via Predictive Latent World Models
- **分类: cs.RO**

- **简介: 该论文提出AtomVLA框架，解决VLA模型在长任务中指令接地不足导致的误差累积问题，通过子任务分解和预测世界模型提升机器人操作的鲁棒性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2603.08519](https://arxiv.org/pdf/2603.08519)**

> **作者:** Xiaoquan Sun; Zetian Xu; Chen Cao; Zonghe Liu; Yihan Sun; Jingrui Pang; Ruijian Zhang; Zhen Yang; Kang Pang; Dingxin He; Mingqi Yuan; Jiayu Chen
>
> **摘要:** Vision-Language-Action (VLA) models demonstrate remarkable potential for generalizable robotic manipulation. The execution of complex multi-step behaviors in VLA models can be improved by robust instruction grounding, a critical component for effective control. However, current paradigms predominantly rely on coarse, high-level task instructions during supervised fine-tuning. This instruction grounding gap leaves models without explicit intermediate guidance, leading to severe compounding errors in long-horizon tasks. Therefore, bridging this instruction gap and providing scalable post-training for VLA models is urgent. To tackle this problem, we propose \method, the first subtask-aware VLA framework integrated with a scalable offline post-training pipeline. Our framework leverages a large language model to decompose high-level demonstrations into fine-grained atomic subtasks. This approach utilizes a pretrained predictive world model to score candidate action chunks against subtask goals in the latent space, mitigating error accumulation while significantly improving long-horizon robustness. Furthermore, this approach enables highly efficient Group Relative Policy Optimization without the prohibitive expenses associated with online rollouts on physical robots. Extensive simulations validate that our AtomVLA maintains strong robustness under perturbations. When evaluated against fundamental baseline models, it achieves an average success rate of 97.0\% on the LIBERO benchmark and 48.0\% on the LIBERO-PRO benchmark. Finally, experiments conducted in the real world using the Galaxea R1 Lite platform confirm its broad applicability across diverse tasks, especially long-horizon tasks. All datasets, checkpoints, and code will be released to the public domain following the acceptance of this work for future research.
>
---
#### [new 102] A Contrastive Fewshot RGBD Traversability Segmentation Framework for Indoor Robotic Navigation
- **分类: cs.RO**

- **简介: 该论文属于室内导航中的可通行性分割任务，旨在提升机器人对细小障碍物的检测能力。通过多模态数据和负样本对比学习，提高分割精度。**

- **链接: [https://arxiv.org/pdf/2603.06927](https://arxiv.org/pdf/2603.06927)**

> **作者:** Qiyuan An; Tuan Dang; Fillia Makedon
>
> **摘要:** Indoor traversability segmentation aims to identify safe, navigable free space for autonomous agents, which is critical for robotic navigation. Pure vision-based models often fail to detect thin obstacles, such as chair legs, which can pose serious safety risks. We propose a multi-modal segmentation framework that leverages RGB images and sparse 1D laser depth information to capture geometric interactions and improve the detection of challenging obstacles. To reduce the reliance on large labeled datasets, we adopt the few-shot segmentation (FSS) paradigm, enabling the model to generalize from limited annotated examples. Traditional FSS methods focus solely on positive prototypes, often leading to overfitting to the support set and poor generalization. To address this, we introduce a negative contrastive learning (NCL) branch that leverages negative prototypes (obstacles) to refine free-space predictions. Additionally, we design a two-stage attention depth module to align 1D depth vectors with RGB images both horizontally and vertically. Extensive experiments on our custom-collected indoor RGB-D traversability dataset demonstrate that our method outperforms state-of-the-art FSS and RGB-D segmentation baselines, achieving up to 9\% higher mIoU under both 1-shot and 5-shot settings. These results highlight the effectiveness of leveraging negative prototypes and sparse depth for robust and efficient traversability segmentation.
>
---
#### [new 103] LAR-MoE: Latent-Aligned Routing for Mixture of Experts in Robotic Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，解决多任务动态下策略部署难题。通过LAR-MoE框架，实现无监督技能分解与专家路由，提升模型性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.08476](https://arxiv.org/pdf/2603.08476)**

> **作者:** Ariel Rodriguez; Chenpan Li; Lorenzo Mazza; Rayan Younis; Ortrun Hellig; Sebastian Bodenstedt; Martin Wagner; Stefanie Speidel
>
> **备注:** Submitted to iROS 2026
>
> **摘要:** Imitation learning enables robots to acquire manipulation skills from demonstrations, yet deploying a policy across tasks with heterogeneous dynamics remains challenging, as models tend to average over distinct behavioral modes present in the demonstrations. Mixture-of-Experts (MoE) architectures address this by activating specialized subnetworks, but requires meaningful skill decompositions for expert routing. We introduce Latent-Aligned Routing for Mixture of Experts (LAR-MoE), a two-stage framework that decouples unsupervised skill discovery from policy learning. In pre-training, we learn a joint latent representation between observations and future actions through student-teacher co-training. In a post-training stage, the expert routing is regularized to follow the structure of the learned latent space, preventing expert collapse while maintaining parameter efficiency. We evaluate LAR-MoE in simulation and on hardware. On the LIBERO benchmark, our method achieves a 95.2% average success rate with 150M parameters. On a surgical bowel grasping and retraction task, LAR-MoE matches a supervised MoE baseline without requiring any phase annotations, and transfers zero-shot to ex vivo porcine tissue. Our findings suggest that latent-aligned routing provides a principled alternative to supervised skill decomposition, enabling structured expert specialization from unlabeled demonstrations.
>
---
#### [new 104] Perception-Aware Communication-Free Multi-UAV Coordination in the Wild
- **分类: cs.RO**

- **简介: 该论文属于多无人机协同任务，解决复杂环境中无通信的导航问题。通过3D LiDAR实现自主避障与定位，提出感知-aware的导航框架，提升协作安全性与有效性。**

- **链接: [https://arxiv.org/pdf/2603.08379](https://arxiv.org/pdf/2603.08379)**

> **作者:** Manuel Boldrer; Michal Kamler; Afzal Ahmad; Martin Saska
>
> **摘要:** We present a communication-free method for safe multi-robot coordination in complex environments such as forests with dense canopy cover, where GNSS is unavailable. Our approach relies on an onboard anisotropic 3D LiDAR sensor used for SLAM as well as for detecting obstacles and neighboring robots. We develop a novel perception-aware 3D navigation framework that enables robots to safely and effectively progress toward a goal region despite limited sensor field-of-view. The approach is evaluated through extensive simulations across diverse scenarios and validated in real-world field experiments, demonstrating its scalability, robustness, and reliability.
>
---
#### [new 105] Hierarchical Multi-Modal Planning for Fixed-Altitude Sparse Target Search and Sampling
- **分类: cs.RO**

- **简介: 该论文属于水下自主导航任务，旨在解决稀疏珊瑚搜索与采样的效率问题。提出HIMoS框架，通过分层规划提高任务效率。**

- **链接: [https://arxiv.org/pdf/2603.08336](https://arxiv.org/pdf/2603.08336)**

> **作者:** Lingpeng Chen; Yuchen Zheng; Apple Pui-Yi Chui; Junfeng Wu; Ziyang Hong
>
> **备注:** 8 pages, 9 figures, conference
>
> **摘要:** Efficient monitoring of sparse benthic phenomena, such as coral colonies, presents a great challenge for Autonomous Underwater Vehicles. Traditional exhaustive coverage strategies are energy-inefficient, while recent adaptive sampling approaches rely on costly vertical maneuvers. To address these limitations, we propose HIMoS (Hierarchical Informative Multi-Modal Search), a fixed-altitude framework for sparse coral search-and-sample missions. The system integrates a heterogeneous sensor suite within a two-layer planning architecture. At the strategic level, a Global Planner optimizes topological routes to maximize potential discovery. At the tactical level, a receding-horizon Local Planner leverages differentiable belief propagation to generate kinematically feasible trajectories that balance acoustic substrate exploration, visual coral search, and close-range sampling. Validated in high-fidelity simulations derived from real-world coral reef benthic surveys, our approach demonstrates superior mission efficiency compared to state-of-the-art baselines.
>
---
#### [new 106] DeReCo: Decoupling Representation and Coordination Learning for Object-Adaptive Decentralized Multi-Robot Cooperative Transport
- **分类: cs.RO**

- **简介: 该论文属于多机器人协作运输任务，旨在解决对象适应性与协调学习的耦合问题。提出DeReCo框架，通过解耦表示与协调学习，提升训练效率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.08111](https://arxiv.org/pdf/2603.08111)**

> **作者:** Kazuki Shibata; Ryosuke Sota; Shandil Dhiresh Bosch; Yuki Kadokawa; Tsurumine Yoshihisa; Takamitsu Matsubara
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Generalizing decentralized multi-robot cooperative transport across objects with diverse shapes and physical properties remains a fundamental challenge. Under decentralized execution, two key challenges arise: object-dependent representation learning under partial observability and coordination learning in multi-agent reinforcement learning (MARL) under non-stationarity. A typical approach jointly optimizes object-dependent representations and coordinated policies in an end-to-end manner while randomizing object shapes and physical properties during training. However, this joint optimization tightly couples representation and coordination learning, introducing bidirectional interference: inaccurate representations under partial observability destabilize coordination learning, while non-stationarity in MARL further degrades representation learning, resulting in sample-inefficient training. To address this structural coupling, we propose DeReCo, a novel MARL framework that decouples representation and coordination learning for object-adaptive multi-robot cooperative transport, improving sample efficiency and generalization across objects and transport scenarios. DeReCo adopts a three-stage training strategy: (1) centralized coordination learning with privileged object information, (2) reconstruction of object-dependent representations from local observations, and (3) progressive removal of privileged information for decentralized execution. This decoupling mitigates interference between representation and coordination learning and enables stable and sample-efficient training. Experimental results show that DeReCo outperforms baselines in simulation on three training objects, generalizes to six unseen objects with varying masses and friction coefficients, and achieves superior performance on two unseen objects in real-robot experiments.
>
---
#### [new 107] Omnidirectional Humanoid Locomotion on Stairs via Unsafe Stepping Penalty and Sparse LiDAR Elevation Mapping
- **分类: cs.RO**

- **简介: 该论文属于机器人步态规划任务，解决人形机器人在楼梯上安全移动的问题。通过引入密集不安全踏步惩罚和稀疏LiDAR高程映射，提升地形感知与足点选择的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.07928](https://arxiv.org/pdf/2603.07928)**

> **作者:** Yuzhi Jiang; Yujun Liang; Junhao Li; Han Ding; Lijun Zhu
>
> **摘要:** Humanoid robots, characterized by numerous degrees of freedom and a high center of gravity, are inherently unstable. Safe omnidirectional locomotion on stairs requires both omnidirectional terrain perception and reliable foothold selection. Existing methods often rely on forward-facing depth cameras, which create blind zones that restrict omnidirectional mobility. Furthermore, sparse post-contact unsafe stepping penalties lead to low learning efficiency and suboptimal strategies. To realize safe stair-traversal gaits, this paper introduces a single-stage training framework incorporating a dense unsafe stepping penalty that provides continuous feedback as the foot approaches a hazardous placement. To obtain stable and reliable elevation maps, we build a rolling point-cloud mapping system with spatiotemporal confidence decay and a self-protection zone mechanism, producing temporally consistent local maps. These maps are further refined by an Edge-Guided Asymmetric U-Net (EGAU), which mitigates reconstruction distortion caused by sparse LiDAR returns on stair risers. Simulation and real-robot experiments show that the proposed method achieves a near-100\% safe stepping rate on stair terrains in simulation, while maintaining a remarkably high safe stepping rate in real-world deployments. Furthermore, it completes a continuous long-distance walking test on complex outdoor terrains, demonstrating reliable sim-to-real transfer and long-term stability.
>
---
#### [new 108] A Distributed Gaussian Process Model for Multi-Robot Mapping
- **分类: cs.RO; cs.LG; stat.ML**

- **简介: 该论文属于多机器人协同建图任务，解决分布式学习中的全局函数建模问题。提出DistGP方法，利用稀疏高斯过程模型实现多机器人协同训练，提升通信稀疏环境下的学习性能。**

- **链接: [https://arxiv.org/pdf/2603.07351](https://arxiv.org/pdf/2603.07351)**

> **作者:** Seth Nabarro; Mark van der Wilk; Andrew J. Davison
>
> **备注:** ICRA 2026, 8 pages
>
> **摘要:** We propose DistGP: a multi-robot learning method for collaborative learning of a global function using only local experience and computation. We utilise a sparse Gaussian process (GP) model with a factorisation that mirrors the multi-robot structure of the task, and admits distributed training via Gaussian belief propagation (GBP). Our loopy model outperforms Tree-Structured GPs \cite{bui2014tree} and can be trained online and in settings with dynamic connectivity. We show that such distributed, asynchronous training can reach the same performance as a centralised, batch-trained model, albeit with slower convergence. Last, we compare to DiNNO \cite{yu2022dinno}, a distributed neural network (NN) optimiser, and find DistGP achieves superior accuracy, is more robust to sparse communication and is better able to learn continually.
>
---
#### [new 109] HSC-VLA: Hierarchical Scene-Clearing for Robust Bimanual Manipulation in Dense Clutter
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决高密度杂乱环境中指令执行失败的问题。提出HSC-VLA框架，通过分层结构分离视觉语义与运动控制，提升复杂场景下的操作性能。**

- **链接: [https://arxiv.org/pdf/2603.07484](https://arxiv.org/pdf/2603.07484)**

> **作者:** Zhen Liu; Xinyu Ning; Zhe Hu; XinXin Xie; Yitong Liu; Zhongzhu Pu
>
> **摘要:** Modern Vision--Language--Action models often suffer from critical instruction-following failures in high-density manipulation environments, where task-irrelevant visual clutter dilutes attention, corrupts grounding, and substantially degrades performance in complex long-horizon scenarios. To overcome the representation bottleneck of monolithic end-to-end architectures, we propose HSC-VLA, a hierarchical framework that decouples high-level visual-semantic reasoning from low-level, high-frequency sensorimotor execution through an explicit scene-clearing abstraction. HSC-VLA employs a high-level Brain to decompose long-horizon tasks and to generate task-specific scene masks that preserve task-relevant geometry while suppressing distractors. The filtered observations are then passed to a low-level Cerebellum, a diffusion-based policy that performs bimanual manipulation using only mask-filtered vision and proprioception. Extensive experiments in densely cluttered supermarket shelves demonstrate that HSC-VLA achieves 86.7\% aggregate success under high-density clutter, surpassing the best monolithic baseline ($\pi_0$-Full FT at 34.3\%) by 52.4\%. HSC-VLA also exhibits strong long-horizon performance, reaching 72\% on clutter sorting and 66\% on restocking, demonstrating strong robustness and effective failure recovery in complex cluttered manipulation.
>
---
#### [new 110] Residual Control for Fast Recovery from Dynamics Shifts
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决动态变化下的快速恢复问题。通过残差控制架构，在不更新策略的情况下实现对未观测动态变化的快速补偿。**

- **链接: [https://arxiv.org/pdf/2603.07775](https://arxiv.org/pdf/2603.07775)**

> **作者:** Nethmi Jayasinghe; Diana Gontero; Francesco Migliarba; Spencer T. Brown; Vinod K. Sangwan; Mark C. Hersam; Amit Ranjan Trivedi
>
> **摘要:** Robotic systems operating in real-world environments inevitably encounter unobserved dynamics shifts during continuous execution, including changes in actuation, mass distribution, or contact conditions. When such shifts occur mid-episode, even locally stabilizing learned policies can experience substantial transient performance degradation. While input-to-state stability guarantees bounded state deviation, it does not ensure rapid restoration of task-level performance. We address inference-time recovery under frozen policy parameters by casting adaptation as constrained disturbance shaping around a nominal stabilizing controller. We propose a stability-aligned residual control architecture in which a reinforcement learning policy trained under nominal dynamics remains fixed at deployment, and adaptation occurs exclusively through a bounded additive residual channel. A Stability Alignment Gate (SAG) regulates corrective authority through magnitude constraints, directional coherence with the nominal action, performance-conditioned activation, and adaptive gain modulation. These mechanisms preserve the nominal closed-loop structure while enabling rapid compensation for unobserved dynamics shifts without retraining or privileged disturbance information. Across mid-episode perturbations including actuator degradation, mass variation, and contact changes, the proposed method consistently reduces recovery time relative to frozen and online-adaptation baselines while maintaining near-nominal steady-state performance. Recovery time is reduced by \textbf{87\%} on the Go1 quadruped, \textbf{48\%} on the Cassie biped, \textbf{30\%} on the H1 humanoid, and \textbf{20\%} on the Scout wheeled platform on average across evaluated conditions relative to a frozen SAC policy.
>
---
#### [new 111] Vector Field Augmented Differentiable Policy Learning for Vision-Based Drone Racing
- **分类: cs.RO**

- **简介: 该论文属于无人机竞速任务，解决复杂环境下的高速飞行与避障问题。提出DiffRacing框架，结合向量场与梯度学习，提升训练效率与飞行性能。**

- **链接: [https://arxiv.org/pdf/2603.08019](https://arxiv.org/pdf/2603.08019)**

> **作者:** Yang Su; Feng Yu; Yu Hu; Xinze Niu; Linzuo Zhang; Fangyu Sun; Danping Zou
>
> **备注:** 8 pages, 7 figures, RAL 2026 March
>
> **摘要:** Autonomous drone racing in complex environments requires agile, high-speed flight while maintaining reliable obstacle avoidance. Differentiable-physics-based policy learning has recently demonstrated high sample efficiency and remarkable performance across various tasks, including agile drone flight and quadruped locomotion. However, applying such methods to drone racing remains difficult, as key objective like gate traversal are inherently hard to express as smooth, differentiable losses. To address these challenges, we propose DiffRacing, a novel vector field-augmented differentiable policy learning framework. DiffRacing integrates differentiable losses and vector fields into the training process to provide continuous and stable gradient signals, balancing obstacle avoidance and high-speed gate traversal. In addition, a differentiable Delta Action Model compensates for dynamics mismatch, enabling efficient sim-to-real transfer without explicit system identification. Extensive simulation and real-world experiments demonstrate that DiffRacing achieves superior sample efficiency, faster convergence, and robust flight performance, thereby demonstrating that vector fields can augment traditional gradient-based policy learning with a task-specific geometric prior.
>
---
#### [new 112] FeasibleCap: Real-Time Embodiment Constraint Guidance for In-the-Wild Robot Demonstration Collection
- **分类: cs.RO**

- **简介: 该论文提出FeasibleCap系统，解决机器人演示数据收集中的执行可行性问题。通过实时反馈引导，提升数据质量，无需依赖模型或硬件。属于机器人数据收集任务。**

- **链接: [https://arxiv.org/pdf/2603.07580](https://arxiv.org/pdf/2603.07580)**

> **作者:** Zi Yin; Fanhong Li; Yun Gui; Jia Liu
>
> **摘要:** Gripper-in-hand data collection decouples demonstration acquisition from robot hardware, but whether a trajectory is executable on the target robot remains unknown until a separate replay-and-validate stage. Failed demonstrations therefore inflate the effective cost per usable trajectory through repeated collection, diagnosis, and validation. Existing collection-time feedback systems mitigate this issue but rely on head-worn AR/VR displays, robot-in-the-loop hardware, or learned dynamics models; real-time executability feedback has not yet been integrated into the gripper-in-hand data collection paradigm. We present \textbf{FeasibleCap}, a gripper-in-hand data collection system that brings real-time executability guidance into robot-free capture. At each frame, FeasibleCap checks reachability, joint-rate limits, and collisions against a target robot model and closes the loop through on-device visual overlays and haptic cues, allowing demonstrators to correct motions during collection without learned models, headsets, or robot hardware. On pick-and-place and tossing tasks, FeasibleCap improves replay success and reduces the fraction of infeasible frames, with the largest gains on tossing. Simulation experiments further indicate that enforcing executability constraints during collection does not sacrifice cross-embodiment transfer across robot platforms. Hardware designs and software are available at this https URL.
>
---
#### [new 113] UniGround: Universal 3D Visual Grounding via Training-Free Scene Parsing
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UniGround，解决3D视觉定位任务中的泛化与鲁棒性问题，通过无需训练的视觉和几何推理实现开放世界3D定位。**

- **链接: [https://arxiv.org/pdf/2603.08131](https://arxiv.org/pdf/2603.08131)**

> **作者:** Jiaxi Zhang; Yunheng Wang; Wei Lu; Taowen Wang; Weisheng Xu; Shuning Zhang; Yixiao Feng; Yuetong Fang; Renjing Xu
>
> **备注:** 14 pages,6 figures,3 tables
>
> **摘要:** Understanding and localizing objects in complex 3D environments from natural language descriptions, known as 3D Visual Grounding (3DVG), is a foundational challenge in embodied AI, with broad implications for robotics, augmented reality, and human-machine interaction. Large-scale pre-trained foundation models have driven significant progress on this front, enabling open-vocabulary 3DVG that allows systems to locate arbitrary objects in a given scene. However, their reliance on pre-trained models constrains 3D perception and reasoning within the inherited knowledge boundaries, resulting in limited generalization to unseen spatial relationships and poor robustness to out-of-distribution scenes. In this paper, we replace this constrained perception with training-free visual and geometric reasoning, thereby unlocking open-world 3DVG that enables the localization of any object in any scene beyond the training data. Specifically, the proposed UniGround operates in two stages: a Global Candidate Filtering stage that constructs scene candidates through training-free 3D topology and multi-view semantic encoding, and a Local Precision Grounding stage that leverages multi-scale visual prompting and structured reasoning to precisely identify the target object. Experiments on ScanRefer and EmbodiedScan show that UniGround achieves 46.1\%/34.1\% Acc@0.25/0.5 on ScanRefer and 28.7\% Acc@0.25 on EmbodiedScan, establishing a new state-of-the-art among zero-shot methods on EmbodiedScan without any 3D supervision. We further evaluate UniGround in real-world environments under uncontrolled reconstruction conditions and substantial domain shift, showing training-free reasoning generalizes robustly beyond curated benchmarks.
>
---
#### [new 114] MetaWorld-X: Hierarchical World Modeling via VLM-Orchestrated Experts for Humanoid Loco-Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人控制任务，旨在解决同时进行运动与操作（loco-manipulation）时的稳定性与泛化问题。提出MetaWorld-X框架，通过分层专家策略和语义引导组合实现更自然、稳定的控制。**

- **链接: [https://arxiv.org/pdf/2603.08572](https://arxiv.org/pdf/2603.08572)**

> **作者:** Yutong Shen; Hangxu Liu; Penghui Liu; Jiashuo Luo; Yongkang Zhang; Rex Morvley; Chen Jiang; Jianwei Zhang; Lei Zhang
>
> **备注:** 8 figures, this https URL
>
> **摘要:** Learning natural, stable, and compositionally generalizable whole-body control policies for humanoid robots performing simultaneous locomotion and manipulation (loco-manipulation) remains a fundamental challenge in robotics. Existing reinforcement learning approaches typically rely on a single monolithic policy to acquire multiple skills, which often leads to cross-skill gradient interference and motion pattern conflicts in high-degree-of-freedom systems. As a result, generated behaviors frequently exhibit unnatural movements, limited stability, and poor generalization to complex task compositions. To address these limitations, we propose MetaWorld-X, a hierarchical world model framework for humanoid control. Guided by a divide-and-conquer principle, our method decomposes complex control problems into a set of specialized expert policies (Specialized Expert Policies, SEP). Each expert is trained under human motion priors through imitation-constrained reinforcement learning, introducing biomechanically consistent inductive biases that ensure natural and physically plausible motion generation. Building upon this foundation, we further develop an Intelligent Routing Mechanism (IRM) supervised by a Vision-Language Model (VLM), enabling semantic-driven expert composition. The VLM-guided router dynamically integrates expert policies according to high-level task semantics, facilitating compositional generalization and adaptive execution in multi-stage loco-manipulation tasks.
>
---
#### [new 115] Multifingered force-aware control for humanoid robots
- **分类: cs.RO**

- **简介: 该论文属于多指机器人力感知控制任务，解决物体抓取中的力分布与稳定问题。通过设计控制器优化力分布，提升抓取稳定性。**

- **链接: [https://arxiv.org/pdf/2603.08142](https://arxiv.org/pdf/2603.08142)**

> **作者:** Pasquale Marra; Gabriele M. Caddeo; Ugo Pattacini; Lorenzo Natale
>
> **备注:** This work has been accepted for publication in ICRA 2026
>
> **摘要:** In this paper, we address force-aware control and force distribution in robotic platforms with multi-fingered hands. Given a target goal and force estimates from tactile sensors, we design a controller that adapts the motion of the torso, arm, wrist, and fingers, redistributing forces to maintain stable contact with objects of varying mass distribution or unstable contacts. To estimate forces, we collect a dataset of tactile signals and ground-truth force measurements using five Xela magnetic sensors interacting with indenters, and train force estimators. We then introduce a model-based control scheme that minimizes the distance between the Center of Pressure (CoP) and the centroid of the fingertips contact polygon. Since our method relies on estimated forces rather than raw tactile signals, it has the potential to be applied to any sensor capable of force estimation. We validate our framework on a balancing task with five objects, achieving a $82.7\%$ success rate, and further evaluate it in multi-object scenarios, achieving $80\%$ accuracy. Code and data can be found here this https URL.
>
---
#### [new 116] Feasibility Restoration under Conflicting STL Specifications with Pareto-Optimal Refinement
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决冲突STL规范下的可行性问题。通过两阶段方法先恢复可行性，再进行多目标优化，实现安全决策。**

- **链接: [https://arxiv.org/pdf/2603.06947](https://arxiv.org/pdf/2603.06947)**

> **作者:** Tianhao Wu; Yiwei Lyu
>
> **摘要:** Signal Temporal Logic (STL) is expressive formal language that specifies spatio-temporal requirements in robotics. Its quantitative robustness semantics can be easily integrated with optimization-based control frameworks. However, STL specifications may become conflicting in real-world applications, where safety rules, traffic regulations, and task objectives can be cannot be satisfied together. In these situations, traditional STL-constrained Model Predictive Control (MPC) becomes infeasible and default to conservative behaviors such as freezing, which can largely increase risks in safety-critical scenarios. In this paper, we proposes a unified two-stage framework that first restores feasibility via minimal relaxation, then refine the feasible solution by formulating it as a value-aware multi-objective optimization problem. Using $\varepsilon$-constraint method, we approximate the Pareto front of the multi-objective optimization, which allows analysis of tradeoffs among competing objectives and counterfactual analysis of alternative actions. We demonstrate that the proposed approach avoids deadlock under conflicting STL specifications and enables interpretable decision-making in safety-critical applications by conducting a case study in autonomous driving.
>
---
#### [new 117] CN-CBF: Composite Neural Control Barrier Function for Safe Robot Navigation in Dynamic Environments
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于机器人安全导航任务，解决动态环境中路径规划的安全性问题。提出一种复合神经控制屏障函数方法，提升导航成功率并保证安全性。**

- **链接: [https://arxiv.org/pdf/2603.06921](https://arxiv.org/pdf/2603.06921)**

> **作者:** Bojan Derajić; Sebastian Bernhard; Wolfgang Hönig
>
> **摘要:** Safe navigation of autonomous robots remains one of the core challenges in the field, especially in dynamic and uncertain environments. One of the prevalent approaches is safety filtering based on control barrier functions (CBFs), which are easy to deploy but difficult to design. Motivated by the shortcomings of existing learning- and model-based methods, we propose a simple yet effective neural CBF design method for safe robot navigation in dynamic environments. We employ the idea of a composite CBF, where multiple neural CBFs are combined into a single CBF. The individual CBFs are trained via the Hamilton-Jacobi reachability framework to approximate the optimal safe set for single moving obstacles. Additionally, we use the residual neural architecture, which guarantees that the estimated safe set does not intersect with the corresponding failure set. The method is extensively evaluated in simulation experiments for a ground robot and a quadrotor, comparing it against several baseline methods. The results show improved success rates of up to 18\% compared to the best baseline, without increasing the conservativeness of the motion. Also, the method is demonstrated in hardware experiments for both types of robots.
>
---
#### [new 118] Inverse Resistive Force Theory (I-RFT): Learning granular properties through robot-terrain physical interactions
- **分类: cs.RO**

- **简介: 该论文属于机器人地形感知任务，旨在解决软颗粒地形属性估计问题。通过引入I-RFT框架，结合物理模型与机器学习，实现从接触力中推断地形特性。**

- **链接: [https://arxiv.org/pdf/2603.07796](https://arxiv.org/pdf/2603.07796)**

> **作者:** Shipeng Liu; Feng Xue; Yifeng Zhang; Tarunika Ponnusamy; Feifei Qian
>
> **摘要:** For robots to navigate safely and efficiently on soft, granular terrains, it is crucial to gather information about the terrain's mechanical properties, which directly affect locomotion performance. Recent research has developed robotic legs that can accurately sense ground reaction forces during locomotion. However, existing tests of granular property estimation often rely on specific foot trajectories, such as vertical penetration or horizontal shear, limiting their applicability during natural locomotion. To address this limitation, we introduce a physics-informed machine learning framework, Inverse Resistive Force Theory (I-RFT), which integrates the Granular Resistive Force Theory model with Gaussian Processes to infer terrain properties from proprioceptively measured contact forces under arbitrary gait trajectories. By embedding the granular force model within the learning process, I-RFT preserves physical consistency while enabling generalization across diverse motion primitives. Experimental results demonstrate that I-RFT accurately estimates terrain properties across multiple gait trajectories and toe shapes. Moreover, we show that the quantified uncertainty over the terrain resistance stress map could enable robots to optimize foot design and gait trajectories for efficient information gathering. This approach establishes a new foundation for data-efficient characterization of complex granular environments and opens new avenues for locomotion strategies that actively adapt gait for autonomous terrain exploration.
>
---
#### [new 119] Approximate Imitation Learning for Event-based Quadrotor Flight in Cluttered Environments
- **分类: cs.RO**

- **简介: 该论文研究基于事件相机的四旋翼高速飞行任务，解决传统相机在高速场景下的性能问题。通过端到端神经网络和近似模仿学习方法，实现高效控制策略训练。**

- **链接: [https://arxiv.org/pdf/2603.07578](https://arxiv.org/pdf/2603.07578)**

> **作者:** Nico Messikommer; Jiaxu Xing; Leonard Bauersfeld; Marco Cannici; Elie Aljalbout; Davide Scaramuzza
>
> **摘要:** Event cameras offer high temporal resolution and low latency, making them ideal sensors for high-speed robotic applications where conventional cameras suffer from image degradations such as motion blur. In addition, their low power consumption can enhance endurance, which is critical for resource-constrained platforms. Motivated by these properties, we present a novel approach that enables a quadrotor to fly through cluttered environments at high speed by perceiving the environment with a single event camera. Our proposed method employs an end-to-end neural network trained to map event data directly to control commands, eliminating the reliance on standard cameras. To enable efficient training in simulation, where rendering synthetic event data is computationally expensive, we propose Approximate Imitation Learning, a novel imitation learning framework. Our approach leverages a large-scale offline dataset to learn a task-specific representation space. Subsequently, the policy is trained through online interactions that rely solely on lightweight, simulated state information, eliminating the need to render events during training. This enables the efficient training of event-based control policies for fast quadrotor flight, highlighting the potential of our framework for other modalities where data simulation is costly or impractical. Our approach outperforms standard imitation learning baselines in simulation and demonstrates robust performance in real-world flight tests, achieving speeds up to 9.8 ms-1 in cluttered environments.
>
---
#### [new 120] CONTACT: CONtact-aware TACTile Learning for Robotic Disassembly
- **分类: cs.RO**

- **简介: 该论文属于机器人拆解任务，解决接触依赖操作的可靠性问题。通过对比视觉、触觉RGB和力场传感配置，发现力场传感在接触和变形场景中效果最佳。**

- **链接: [https://arxiv.org/pdf/2603.08560](https://arxiv.org/pdf/2603.08560)**

> **作者:** Yosuke Saka; Jyun-Chi Hu; Adeesh Desai; Zhiyuan Zhang; Bihao Zhang; Quan Khanh Luu; Md Rakibul Islam Prince; Minghui Zheng; Yu She
>
> **备注:** Submitted to IROS 2026, 8 pages, 6 figures
>
> **摘要:** Robotic disassembly involves contact-rich interactions in which successful manipulation depends not only on geometric alignment but also on force-dependent state transitions. While vision-based policies perform well in structured settings, their reliability often degrades in tight-tolerance, contact-dominated, or deformable scenarios. In this work, we systematically investigate the role of tactile sensing in robotic disassembly through both simulation and real-world experiments. We construct five rigid-body disassembly tasks in simulation with increasing geometric constraints and extraction difficulty. We further design five real-world tasks, including three rigid and two deformable scenarios, to evaluate contact-dependent manipulation. Within a unified learning framework, we compare three sensing configurations: Vision Only, Vision + tactile RGB (TacRGB), and Vision + tactile force field (TacFF). Across both simulation and real-world experiments, TacFF-based policies consistently achieve the highest success rates, with particularly notable gains in contact-dependent and deformable settings. Notably, naive fusion of TacRGB and TacFF underperforms either modality alone, indicating that simple concatenation can dilute task-relevant force information. Our results show that tactile sensing plays a critical, task-dependent role in robotic disassembly, with structured force-field representations being particularly effective in contact-dominated scenarios.
>
---
#### [new 121] Kinematics-Aware Latent World Models for Data-Efficient Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主驾驶任务，旨在解决数据效率问题。通过引入运动学信息和几何监督，构建更精确的潜在世界模型，提升策略优化效果。**

- **链接: [https://arxiv.org/pdf/2603.07264](https://arxiv.org/pdf/2603.07264)**

> **作者:** Jiazhuo Li; Linjiang Cao; Qi Liu; Xi Xiong
>
> **备注:** 6 pages, 5 figures. Under review at IEEE ITSC
>
> **摘要:** Data-efficient learning remains a central challenge in autonomous driving due to the high cost and safety risks of large-scale real-world interaction. Although world-model-based reinforcement learning enables policy optimization through latent imagination, existing approaches often lack explicit mechanisms to encode spatial and kinematic structure essential for driving tasks. In this work, we build upon the Recurrent State-Space Model (RSSM) and propose a kinematics-aware latent world model framework for autonomous driving. Vehicle kinematic information is incorporated into the observation encoder to ground latent transitions in physically meaningful motion dynamics, while geometry-aware supervision regularizes the RSSM latent state to capture task-relevant spatial structure beyond pixel reconstruction. The resulting structured latent dynamics improve long-horizon imagination fidelity and stabilize policy optimization. Experiments in a driving simulation benchmark demonstrate consistent gains over both model-free and pixel-based world-model baselines in terms of sample efficiency and driving performance. Ablation studies further verify that the proposed design enhances spatial representation quality within the latent space. These results suggest that integrating kinematic grounding into RSSM-based world models provides a scalable and physically grounded paradigm for autonomous driving policy learning.
>
---
#### [new 122] TRIAGE: Type-Routed Interventions via Aleatoric-Epistemic Gated Estimation in Robotic Manipulation and Adaptive Perception -- Don't Treat All Uncertainty the Same
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制与感知任务，解决不确定性处理不精准的问题。通过分解不确定性和针对性响应，提升控制与跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.08128](https://arxiv.org/pdf/2603.08128)**

> **作者:** Divake Kumar; Sina Tayebati; Devashri Naik; Patrick Poggi; Amanda Sofie Rios; Nilesh Ahuja; Amit Ranjan Trivedi
>
> **摘要:** Most uncertainty-aware robotic systems collapse prediction uncertainty into a single scalar score and use it to trigger uniform corrective responses. This aggregation obscures whether uncertainty arises from corrupted observations or from mismatch between the learned model and the true system dynamics. As a result, corrective actions may be applied to the wrong component of the closed loop, degrading performance relative to leaving the policy unchanged. We introduce a lightweight post hoc framework that decomposes uncertainty into aleatoric and epistemic components and uses these signals to regulate system responses at inference time. Aleatoric uncertainty is estimated from deviations in the observation distribution using a Mahalanobis density model, while epistemic uncertainty is detected using a noise robust forward dynamics ensemble that isolates model mismatch from measurement corruption. The two signals remain empirically near orthogonal during closed loop execution and enable type specific responses. High aleatoric uncertainty triggers observation recovery, while high epistemic uncertainty moderates control actions. The same signals also regulate adaptive perception by guiding model capacity selection during tracking inference. Experiments demonstrate consistent improvements across both control and perception tasks. In robotic manipulation, the decomposed controller improves task success from 59.4% to 80.4% under compound perturbations and outperforms a combined uncertainty baseline by up to 21.0%. In adaptive tracking inference on MOT17, uncertainty-guided model selection reduces average compute by 58.2% relative to a fixed high capacity detector while preserving detection quality within 0.4%. Code and demo videos are available at this https URL.
>
---
#### [new 123] Bilevel Planning with Learned Symbolic Abstractions from Interaction Data
- **分类: cs.RO**

- **简介: 该论文属于智能体规划任务，解决复杂环境中连续与离散表示结合的问题。通过双层框架，利用学习到的符号规则生成计划，并用连续模型验证，提升规划效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08599](https://arxiv.org/pdf/2603.08599)**

> **作者:** Fatih Dogangun; Burcu Kilic; Serdar Bahar; Emre Ugur
>
> **摘要:** Intelligent agents must reason over both continuous dynamics and discrete representations to generate effective plans in complex environments. Previous studies have shown that symbolic abstractions can emerge from neural effect predictors trained with a robot's unsupervised exploration. However, these methods rely on deterministic symbolic domains, lack mechanisms to verify the generated symbolic plans, and operate only at the abstract level, often failing to capture the continuous dynamics of the environment. To overcome these limitations, we propose a bilevel neuro-symbolic framework in which learned probabilistic symbolic rules generate candidate plans rapidly at the high level, and learned continuous effect models verify these plans and perform forward search when necessary at the low level. Our experiments on multi-object manipulation tasks demonstrate that the proposed bilevel method outperforms symbolic-only approaches, reliably identifying failing plans through verification, and achieves planning performance statistically comparable to continuous forward search while resolving most problems via efficient symbolic reasoning.
>
---
#### [new 124] Less is More: Robust Zero-Communication 3D Pursuit-Evasion via Representational Parsimony
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究3D追逐逃避任务，解决通信延迟与部分可观测性问题。通过简化观测接口和引入CGCA机制，提升无通信协作的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.08273](https://arxiv.org/pdf/2603.08273)**

> **作者:** Jialin Ying; Zhihao Li; Zicheng Dong; Guohua Wu; Yihuan Liao
>
> **备注:** 7 pages, 10 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Asymmetric 3D pursuit-evasion in cluttered voxel environments is difficult under communication latency, partial observability, and nonholonomic maneuver limits. While many MARL methods rely on richer inter-agent coupling or centralized signals, these dependencies can become fragility sources when communication is delayed or noisy. Building on an inherited path-guided decentralized pursuit scaffold, we study a robustness-oriented question: can representational parsimony improve communication-free coordination? We instantiate this principle with (i) a parsimonious actor observation interface that removes team-coupled channels (83-D to 50-D), and (ii) Contribution-Gated Credit Assignment (CGCA), a locality-aware credit structure for communication-denied cooperation. In Stage-5 evaluation (4 pursuers vs. 1 evader), our configuration reaches 0.753 +/- 0.091 success and 0.223 +/- 0.066 collision, outperforming the 83-D FULL OBS counterpart (0.721 +/- 0.071, 0.253 +/- 0.089). It further shows graceful degradation under speed/yaw/noise/delay stress tests and resilient zero-shot transfer on urban-canyon maps (about 61% success at density 0.24). These results support a practical paradigm shift: explicitly severing redundant cross-agent channels can suppress compounding error cascades and improve robustness in latency-prone deployment.
>
---
#### [new 125] Unifying Sidewinding and Rolling: A Wave-Based Framework for Self-Righting in Elongated Limbless and Multi-Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人自正问题研究，旨在解决长体多足机器人在复杂地形中可靠自正的难题。通过对比生物实例和实验分析，提出基于波形的自正框架，并揭示形态与策略的耦合关系。**

- **链接: [https://arxiv.org/pdf/2603.07417](https://arxiv.org/pdf/2603.07417)**

> **作者:** Hangjun Liu; Jiarui Geng; Jinxuan Ding; Gengzhi He; Xiyuan Wang; Melisa Arukgoda; Joe DiGennaro; George Ubertalli; Grigoriy Blekherman; Baxi Chong
>
> **摘要:** Centipede-like robots offer unique locomotion advantages due to their small cross-sectional area for accessing confined spaces, and their redundant legs enhance robustness in cluttered environments such as search-and-rescue and pipe inspection. However, elongated robots are particularly vulnerable to tipping over when climbing large obstacles, making reliable self-righting essential for field deployment. Self-righting strategies for elongate, multi-legged systems remain poorly understood. In this study, we conduct a comparative biomechanics and robophysical investigation to address three key questions: (1) What self-righting strategies are effective for elongate, many-legged systems? (2) How should these strategies depend on morphological parameters such as leg length and leg number? (3) Is there a morphological limit beyond which reliable self-righting becomes infeasible? We compare two biological exemplars: Scolopendra subspinipes (short legs) and Scutigera coleoptrata (house centipedes with long legs). Scolopendra subspinipes reliably self-rights both during aerial phases and through ground-assisted self-righting, whereas house centipedes rely predominantly on aerial reorientation and struggle to generate effective self-righting torques during ground contact. Motivated by these observations, we construct a parameterized space of bio-inspired self-righting strategies and develop an elongate robot with adjustable leg lengths. Systematic experiments reveal that increasing leg length necessitates a shift in control strategy to prevent torque over-concentration in mid-body actuators, and we identify a critical limb-length threshold above which robust self-righting becomes challenging. These results establish morphology-strategy coupling principles for self-righting in elongate robots and provide design guidelines for centipede-like systems operating in uncertain terrain.
>
---
#### [new 126] Low-Cost Teleoperation Extension for Mobile Manipulators
- **分类: cs.RO**

- **简介: 该论文属于机器人 teleoperation 任务，旨在解决移动双臂机械臂的高成本控制问题。通过开源框架和常用硬件实现直观的全身控制，提升操作效率与用户体验。**

- **链接: [https://arxiv.org/pdf/2603.07672](https://arxiv.org/pdf/2603.07672)**

> **作者:** Danil Belov; Artem Erkhov; Yaroslav Savotin; Tatiana Podladchikova; Pavel Osinenko
>
> **摘要:** Teleoperation of mobile bimanual manipulators requires simultaneous control of high-dimensional systems, often necessitating expensive specialized equipment. We present an open-source teleoperation framework that enables intuitive whole body control using readily available commodity hardware. Our system combines smartphone-based head tracking for camera control, leader arms for bilateral manipulation, and foot pedals for hands-free base navigation. Using a standard smartphone with IMU and display, we eliminate the need for costly VR helmets while maintaining immersive visual feedback. The modular architecture integrates seamlessly with the XLeRobot framework, but can be easily adapted to other types of mobile manipulators. We validate our approach through user studies that demonstrate improved task performance and reduced cognitive load compared to keyboard-based control.
>
---
#### [new 127] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决现有方法感知效率低、推理不聚焦的问题。提出ProFocus框架，结合大语言模型和视觉语言模型，实现主动感知和聚焦推理。**

- **链接: [https://arxiv.org/pdf/2603.05530](https://arxiv.org/pdf/2603.05530)**

> **作者:** Wei Xue; Mingcheng Li; Xuecheng Wu; Jingqun Tang; Dingkang Yang; Lihua Zhang
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to accurately perceive complex visual environments and reason over navigation instructions and histories. However, existing methods passively process redundant visual inputs and treat all historical contexts indiscriminately, resulting in inefficient perception and unfocused reasoning. To address these challenges, we propose \textbf{ProFocus}, a training-free progressive framework that unifies \underline{Pro}active Perception and \underline{Focus}ed Reasoning through collaboration between large language models (LLMs) and vision-language models (VLMs). For proactive perception, ProFocus transforms panoramic observations into structured ego-centric semantic maps, enabling the orchestration agent to identify missing visual information needed for reliable decision-making, and to generate targeted visual queries with corresponding focus regions that guide the perception agent to acquire the required observations. For focused reasoning, we propose Branch-Diverse Monte Carlo Tree Search (BD-MCTS) to identify top-$k$ high-value waypoints from extensive historical candidates. The decision agent focuses reasoning on the historical contexts associated with these waypoints, rather than considering all historical waypoints equally. Extensive experiments validate the effectiveness of ProFocus, achieving state-of-the-art performance among zero-shot methods on R2R and REVERIE benchmarks.
>
---
#### [new 128] DAISS: Phase-Aware Imitation Learning for Dual-Arm Robotic Ultrasound-Guided Interventions
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决超声引导下双臂操作的自动化问题。通过模仿学习构建相位感知策略，实现精准双臂协调控制。**

- **链接: [https://arxiv.org/pdf/2603.07663](https://arxiv.org/pdf/2603.07663)**

> **作者:** Feng Li; Pei Liu; Shiting Wang; Ning Wang; Zhongliang Jiang; Nassir Navab; Yuan Bi
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Imitation learning has shown strong potential for automating complex robotic manipulation. In medical robotics, ultrasound-guided needle insertion demands precise bimanual coordination, as clinicians must simultaneously manipulate an ultrasound probe to maintain an optimal acoustic view while steering an interventional needle. Automating this asymmetric workflow -- and reliably transferring expert strategies to robots -- remains highly challenging. In this paper, we present the Dual-Arm Interventional Surgical System (DAISS), a teleoperated platform that collects high-fidelity dual-arm demonstrations and learns a phase-aware imitation policy for ultrasound-guided interventions. To avoid constraining the operator's natural behavior, DAISS uses a flexible NDI-based leader interface for teleoperating two coordinated follower arms. To support robust execution under real-time ultrasound feedback, we develop a lightweight, data-efficient imitation policy. Specifically, the policy incorporates a phase-aware architecture and a dynamic mask loss tailored to asymmetric bimanual control. Conditioned on a planned trajectory, the network fuses real-time ultrasound with external visual observations to generate smooth, coordinated dual-arm motions. Experimental results show that DAISS can learn personalized expert strategies from limited demonstrations. Overall, these findings highlight the promise of phase-aware imitation-learning-driven dual-arm robots for improving precision and reducing cognitive workload in image-guided interventions.
>
---
#### [new 129] Tutorial on Aided Inertial Navigation Systems: A Modern Treatment Using Lie-Group Theoretical Methods
- **分类: cs.RO; eess.SY**

- **简介: 本文探讨基于李群理论的辅助惯性导航系统，解决导航信息融合问题，提出几何框架以提升系统性能。**

- **链接: [https://arxiv.org/pdf/2603.07143](https://arxiv.org/pdf/2603.07143)**

> **作者:** Soulaimane Berkane
>
> **摘要:** This tutorial presents a control-oriented introduction to aided inertial navigation systems using a Lie-group formulation centered on the extended Special Euclidean group SE_2(3). The focus is on developing a clear and implementation-oriented geometric framework for fusing inertial measurements with aiding information, while making the role of invariance and symmetry explicit. Recent extensions, including higher-order state representations, synchronous observer designs, and equivariant filtering methods, are discussed as natural continuations of the same underlying principles. The goal is to provide readers with a coherent system-theoretic perspective that supports both understanding and practical use of modern aided inertial navigation methods.
>
---
#### [new 130] Preference-Conditioned Reinforcement Learning for Space-Time Efficient Online 3D Bin Packing
- **分类: cs.RO**

- **简介: 该论文属于机器人装箱任务，解决空间与时间效率的平衡问题。通过强化学习方法，提升装箱密度同时减少操作时间。**

- **链接: [https://arxiv.org/pdf/2603.07800](https://arxiv.org/pdf/2603.07800)**

> **作者:** Nikita Sarawgi; Omey M. Manyar; Fan Wang; Thinh H. Nguyen; Daniel Seita; Satyandra K. Gupta
>
> **备注:** 8 pages, 5 figures. Accepted to IEEE International Conference on Robotics and Automation 2026. Project Website: this https URL
>
> **摘要:** Robotic bin packing is widely deployed in warehouse automation, with current systems achieving robust performance through heuristic and learning-based strategies. These systems must balance compact placement with rapid execution, where selecting alternative items or reorienting them can improve space utilization but introduce additional time. We propose a selection-based formulation that explicitly reasons over this trade-off: at each step, the robot evaluates multiple candidate actions, weighing expected packing benefit against estimated operational time. This enables time-aware strategies that selectively accept increased operational time when it yields meaningful spatial improvements. Our method, STEP (Space-Time Efficient Packing), uses a preference-conditioned, Transformer-based reinforcement learning policy, and allows generalization across candidate set sizes and integration with standard placement modules. It achieves a 44% reduction in operational time without compromising packing density. Additional material is available at this https URL.
>
---
#### [new 131] SaiVLA-0: Cerebrum--Pons--Cerebellum Tripartite Architecture for Compute-Aware Vision-Language-Action
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出SaiVLA-0架构，解决视觉-语言-动作协同问题，通过脑区类比设计模块化系统，提升控制效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.08124](https://arxiv.org/pdf/2603.08124)**

> **作者:** Xiang Shi; Wenlong Huang; Menglin Zou; Xinhai Sun
>
> **备注:** 14 pages, 3 figures
>
> **摘要:** We revisit Vision-Language-Action through a neuroscience-inspired triad. Biologically, the Cerebrum provides stable high-level multimodal priors and remains frozen; the Pons Adapter integrates these cortical features with real-time proprioceptive inputs and compiles intent into execution-ready tokens; and the Cerebellum (ParaCAT) performs fast, parallel categorical decoding for online control, with hysteresis/EMA/temperature/entropy for stability. A fixed-ratio schedule and two-stage feature caching make the system compute-aware and reproducible. Inspired by active, foveated vision, our wrist ROIs are geometrically tied to the end-effector via calibrated projection, providing a movement-stabilized, high-resolution view that is sensitive to fine-grained pose changes and complements the global context of the main view. The design is modular: upgrading the Cerebrum only retrains the Pons; changing robots only trains the Cerebellum; cerebellum-only RL can further refine control without touching high-level semantics. As a concept-and-protocol paper with preliminary evidence, we outline a timing protocol under matched conditions (GPU, resolution, batch) to verify anticipated efficiency gains. We also report preliminary LIBERO evidence showing that split feature caching reduces training time (7.5h to 4.5h) and improves average success (86.5% to 92.5%) under official N1.5 head-only training, and that SaiVLA0 reaches 99.0% mean success.
>
---
#### [new 132] Foundational World Models Accurately Detect Bimanual Manipulator Failures
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人故障检测任务，解决bimanual manipulator在复杂环境中的异常检测问题。通过训练概率世界模型，结合不确定性估计实现高效故障检测。**

- **链接: [https://arxiv.org/pdf/2603.06987](https://arxiv.org/pdf/2603.06987)**

> **作者:** Isaac R. Ward; Michelle Ho; Houjun Liu; Aaron Feldman; Joseph Vincent; Liam Kruse; Sean Cheong; Duncan Eddy; Mykel J. Kochenderfer; Mac Schwager
>
> **备注:** 8 pages, 5 figures, accepted at the 2026 IEEE International Conference on Robotics and Automation
>
> **摘要:** Deploying visuomotor robots at scale is challenging due to the potential for anomalous failures to degrade performance, cause damage, or endanger human life. Bimanual manipulators are no exception; these robots have vast state spaces comprised of high-dimensional images and proprioceptive signals. Explicitly defining failure modes within such state spaces is infeasible. In this work, we overcome these challenges by training a probabilistic, history informed, world model within the compressed latent space of a pretrained vision foundation model (NVIDIA's Cosmos Tokenizer). The model outputs uncertainty estimates alongside its predictions that serve as non-conformity scores within a conformal prediction framework. We use these scores to develop a runtime monitor, correlating periods of high uncertainty with anomalous failures. To test these methods, we use the simulated Push-T environment and the Bimanual Cable Manipulation dataset, the latter of which we introduce in this work. This new dataset features trajectories with multiple synchronized camera views, proprioceptive signals, and annotated failures from a challenging data center maintenance task. We benchmark our methods against baselines from the anomaly detection and out-of-distribution detection literature, and show that our approach considerably outperforms statistical techniques. Furthermore, we show that our approach requires approximately one twentieth of the trainable parameters as the next-best learning-based approach, yet outperforms it by 3.8% in terms of failure detection rate, paving the way toward safely deploying manipulator robots in real-world environments where reliability is non-negotiable.
>
---
#### [new 133] Dynamic Targeting of Satellite Observations Using Supplemental Geostationary Satellite Data and Hierarchical Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于卫星观测任务，旨在解决动态目标定位问题。通过引入静止轨道卫星数据和分层规划方法，提升观测效率，有效应对目标分布稀疏的场景。**

- **链接: [https://arxiv.org/pdf/2603.06719](https://arxiv.org/pdf/2603.06719)**

> **作者:** Akseli Kangaslahti; Itai Zilberstein; Alberto Candela; Steve Chien
>
> **备注:** Appears in the proceedings of the 2026 IEEE International Conference on Robotics and Automation
>
> **摘要:** The Dynamic Targeting (DT) mission concept is an approach to satellite observation in which a lookahead sensor gathers information about the upcoming environment and uses this information to intelligently plan observations. Previous work has shown that DT has the potential to increase the science return across applications. However, DT mission concepts must address challenges, such as the limited spatial extent of onboard lookahead data and instrument mobility, data throughput, and onboard computation constraints. In this work, we show how the performance of DT systems can be improved by using supplementary data streamed from geostationary satellites that provide lookahead information up to 35 minutes ahead of time rather than the 1 minute latency from an onboard lookahead sensor. While there is a greater volume of geostationary data, the search space for observation planning explodes exponentially with the size of the horizon. To address this, we introduce a hierarchical planning approach in which the geostationary data is used to plan a long-term observation blueprint in polynomial time, then the onboard lookahead data is leveraged to refine that plan over short-term horizons. We compare the performance of our approach to that of traditional DT planners relying on onboard lookahead data across four different problem instances: three cloud avoidance variations and a storm hunting scenario. We show that our hierarchical planner outperforms the traditional DT planners by up to 41% and examine the features of the scenarios that affect the performance of our approach. We demonstrate that incorporating geostationary satellite data is most effective for dynamic problem instances in which the targets of interest are sparsely distributed throughout the overflight.
>
---
#### [new 134] Interactive World Simulator for Robot Policy Training and Evaluation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出Interactive World Simulator，用于机器人策略训练与评估。解决传统世界模型速度慢、物理一致性差的问题，通过一致性模型实现快速稳定模拟，支持长时序交互。**

- **链接: [https://arxiv.org/pdf/2603.08546](https://arxiv.org/pdf/2603.08546)**

> **作者:** Yixuan Wang; Rhythm Syed; Fangyu Wu; Mengchao Zhang; Aykut Onol; Jose Barreiros; Hooshang Nayyeri; Tony Dear; Huan Zhang; Yunzhu Li
>
> **备注:** Project Page: this https URL
>
> **摘要:** Action-conditioned video prediction models (often referred to as world models) have shown strong potential for robotics applications, but existing approaches are often slow and struggle to capture physically consistent interactions over long horizons, limiting their usefulness for scalable robot policy training and evaluation. We present Interactive World Simulator, a framework for building interactive world models from a moderate-sized robot interaction dataset. Our approach leverages consistency models for both image decoding and latent-space dynamics prediction, enabling fast and stable simulation of physical interactions. In our experiments, the learned world models produce interaction-consistent pixel-level predictions and support stable long-horizon interactions for more than 10 minutes at 15 FPS on a single RTX 4090 GPU. Our framework enables scalable demonstration collection solely within the world models to train state-of-the-art imitation policies. Through extensive real-world evaluation across diverse tasks involving rigid objects, deformable objects, object piles, and their interactions, we find that policies trained on world-model-generated data perform comparably to those trained on the same amount of real-world data. Additionally, we evaluate policies both within the world models and in the real world across diverse tasks, and observe a strong correlation between simulated and real-world performance. Together, these results establish the Interactive World Simulator as a stable and physically consistent surrogate for scalable robotic data generation and faithful, reproducible policy evaluation.
>
---
#### [new 135] Exoskeleton Control through Learning to Reduce Biological Joint Moments in Simulations
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于外骨骼控制任务，旨在通过学习减少生物关节力矩。工作包括构建RL框架和验证管道，使用仿真训练控制器并验证其效果。**

- **链接: [https://arxiv.org/pdf/2603.07629](https://arxiv.org/pdf/2603.07629)**

> **作者:** Zihang You; Xianlian Zhou
>
> **摘要:** Data-driven joint-moment predictors offer a scalable alternative to laboratory-based inverse-dynamics pipelines for biomechanics estimation and exoskeleton control. Meanwhile, physics-based reinforcement learning (RL) enables simulation-trained controllers to learn dynamics-aware assistance strategies without extensive human experimentation. However, quantitative verification of simulation-trained exoskeleton torque predictors, and their impact on human joint power injection, remains limited. This paper presents (1) an RL framework to learn exoskeleton assistance policies that reduce biological joint moments, and (2) a validation pipeline that verifies the trained control networks using an open-source gait dataset through inference and comparison with biological joint moments. Simulation-trained multilayer perceptron (MLP) controllers are developed for level-ground and ramp walking, mapping short-horizon histories of bilateral hip and knee kinematics to normalized assistance torques. Results show that predicted assistance preserves task-intensity trends across speeds and inclines. Agreement is particularly strong at the hip, with cross-correlation coefficients reaching 0.94 at 1.8 m/s and 0.98 during 5° decline walking, demonstrating near-matched temporal structure. Discrepancies increase at higher speeds and steeper inclines, especially at the knee, and are more pronounced in joint power comparisons. Delay tuning biases assistance toward greater positive power injection; modest timing shifts increase positive power and improve agreement in specific gait intervals. Together, these results establish a quantitative validation framework for simulation-trained exoskeleton controllers, demonstrate strong sim-to-data consistency at the torque level, and highlight both the promise and the remaining challenges for sim-to-real transfer.
>
---
#### [new 136] Failure Mechanisms and Risk Estimation for Legged Robot Locomotion on Granular Slopes
- **分类: cs.RO**

- **简介: 该论文研究腿式机器人在颗粒斜坡上的运动失效机制与风险评估。针对颗粒地形中运动性能下降的问题，通过实验和建模分析，提出预测模型并构建失效相图，以提升机器人在复杂地形中的安全性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.06928](https://arxiv.org/pdf/2603.06928)**

> **作者:** Xingjue Liao; Feifei Qian
>
> **摘要:** Locomotion on granular slopes such as sand dunes remains a fundamental challenge for legged robots due to reduced shear strength and gravity-induced anisotropic yielding of granular media. Using a hexapedal robot on a tiltable granular bed, we systematically measure locomotion speed together with slope-dependent normal and shear granular resistive forces. While normal penetration resistance remains nearly unchanged with inclination, shear resistance decreases substantially as slope angle increases. Guided by these measurements, we develop a simple robot-terrain interaction model that predicts anchoring timing, step length, and resulting robot speed, as functions of terrain strength and slope angle. The model reveals that slope-induced performance loss is primarily governed by delayed anchoring and increased backward slip rather than excessive sinkage. By extending the model to generalized terrain conditions, we construct failure phase diagrams that identify sinkage- and slippage-induced failure regimes, enabling quantitative risk estimation for locomotion on granular slopes. This physics-informed framework provides predictive insight into terrain-dependent failure mechanisms and offers guidance for safer and more robust robot operation on deformable inclines.
>
---
#### [new 137] RoboRouter: Training-Free Policy Routing for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出RoboRouter，解决机器人操作中策略泛化不足的问题。通过智能路由不同策略，提升任务成功率，无需训练即可集成新策略。**

- **链接: [https://arxiv.org/pdf/2603.07892](https://arxiv.org/pdf/2603.07892)**

> **作者:** Yiteng Chen; Zhe Cao; Hongjia Ren; Chenjie Yang; Wenbo Li; Shiyi Wang; Yemin Wang; Li Zhang; Yanming Shao; Zhenjun Zhao; Huiping Zhuang; Qingyao Wu
>
> **摘要:** Research on robotic manipulation has developed a diverse set of policy paradigms, including vision-language-action (VLA) models, vision-action (VA) policies, and code-based compositional approaches. Concrete policies typically attain high success rates on specific task distributions but lim-ited generalization beyond it. Rather than proposing an other monolithic policy, we propose to leverage the complementary strengths of existing approaches through intelligent policy routing. We introduce RoboRouter, a training-free framework that maintains a pool of heterogeneous policies and learns to select the best-performing policy for each task through accumulated execution experience. Given a new task, RoboRouter constructs a semantic task representation, retrieves historical records of similar tasks, predicts the optimal policy choice without requiring trial-and-error, and incorporates structured feedback to refine subsequent routing decisions. Integrating a new policy into the system requires only lightweight evaluation and incurs no training overhead. Across simulation benchmark and real-world evaluations, RoboRouter consistently outperforms than in-dividual policies, improving average success rate by more than 3% in simulation and over 13% in real-world settings, while preserving execution efficiency. Our results demonstrate that intelligent routing across heterogeneous, off-the-shelf policies provides a practical and scalable pathway toward building more capable robotic systems.
>
---
#### [new 138] Multi-Agent Off-World Exploration for Sparse Evidence Discovery via Gaussian Belief Mapping and Dual-Domain Coverage
- **分类: cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决外星探索中目标稀疏、感知有限的问题。提出基于高斯信念映射和双域覆盖的框架，提升探索效率与安全性。**

- **链接: [https://arxiv.org/pdf/2603.07650](https://arxiv.org/pdf/2603.07650)**

> **作者:** Zhuoran Qiao; Tianxin Hu; Thien-Minh Nguyen; Shenghai Yuan
>
> **摘要:** Off-world multi-robot exploration is challenged by sparse targets, limited sensing, hazardous terrain, and restricted communication. Many scientifically valuable clues are visually ambiguous and often require close-range observations, making efficient and safe informative path planning essential. Existing methods often rely on predefined areas of interest (AOIs), which may be incomplete or biased, and typically handle terrain risk only through soft penalties, which are insufficient for avoiding non-recoverable regions. To address these issues, we propose a multi-agent informative path planning framework for sparse evidence discovery based on Gaussian belief mapping and dual-domain coverage. The method maintains Gaussian-process-based interest and risk beliefs and combines them with trajectory-intent representations to support coordinated sequential decision-making among multiple agents. It further prioritizes search inside the AOI while preserving limited exploration outside it, thereby improving robustness to AOI bias. In addition, the risk-aware design helps agents balance information gain and operational safety in hazardous environments. Experimental results in simulated lunar environments show that the proposed method consistently outperforms sampling-based and greedy baselines under different budgets and communication ranges. In particular, it achieves lower final uncertainty in risk-aware settings and remains robust under limited communication, demonstrating its effectiveness for cooperative off-world robotic exploration.
>
---
#### [new 139] SMAT: Staged Multi-Agent Training for Co-Adaptive Exoskeleton Control
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决外骨骼与用户协同适应问题。通过分阶段多智能体训练方法，提升外骨骼辅助效果，实现稳定、高效的机械功率输出。**

- **链接: [https://arxiv.org/pdf/2603.07618](https://arxiv.org/pdf/2603.07618)**

> **作者:** Yifei Yuan; Ghaith Androwis; Xianlian Zhou
>
> **摘要:** Effective exoskeleton assistance requires co-adaptation: as the device alters joint dynamics, the user reorganizes neuromuscular coordination, creating a non-stationary learning problem. Most learning-based approaches do not explicitly account for the sequential nature of human motor adaptation, leading to training instability and poorly timed assistance. We propose Staged Multi-Agent Training (SMAT), a four-stage curriculum designed to mirror how users naturally acclimate to a wearable device. In SMAT, a musculoskeletal human actor and a bilateral hip exoskeleton actor are trained progressively: the human first learns unassisted gait, then adapts to the added device mass; the exoskeleton subsequently learns a positive assistance pattern against a stabilized human policy, and finally both agents co-adapt with full torque capacity and bidirectional feedback. We implement SMAT in the MyoAssist simulation environment using a 26-muscle lower-limb model and an attached hip exoskeleton. Our musculoskeletal simulations demonstrate that the learned exoskeleton control policy produces an average 10.1% reduction in hip muscle activation relative to the no-assist condition. We validated the learned controller in an offline setting using open-source gait data, then deployed it to a physical hip exoskeleton for treadmill experiments with five subjects. The resulting policy delivers consistent assistance and predominantly positive mechanical power without the need for any explicitly imposed timing shift (mean positive power: 13.6 W at 6 Nm RMS torque to 23.8 W at 9.3 Nm RMS torque, with minimal negative power) consistently across all subjects without subject-specific retraining.
>
---
#### [new 140] One-Shot Badminton Shuttle Detection for Mobile Robots
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，解决移动机器人视角下羽毛球的实时检测问题。构建了首个相关数据集，提出优化的YOLOv8模型，在不同环境中实现高效检测。**

- **链接: [https://arxiv.org/pdf/2603.06691](https://arxiv.org/pdf/2603.06691)**

> **作者:** Florentin Dipner; William Talbot; Turcan Tuna; Andrei Cramariuc; Marco Hutter
>
> **备注:** Under review for IEEE R-AP
>
> **摘要:** This paper presents a robust one-shot badminton shuttlecock detection framework for non-stationary robots. To address the lack of egocentric shuttlecock detection datasets, we introduce a dataset of 20,510 semi-automatically annotated frames captured across 11 distinct backgrounds in diverse indoor and outdoor environments, and categorize each frame into one of three difficulty levels. For labeling, we present a novel semi-automatic annotation pipeline, that enables efficient labeling from stationary camera footage. We propose a metric suited to our downstream use case and fine-tune a YOLOv8 network optimized for real-time shuttlecock detection, achieving an F1-score of 0.86 under our metric in test environments similar to training, and 0.70 in entirely unseen environments. Our analysis reveals that detection performance is critically dependent on shuttlecock size and background texture complexity. Qualitative experiments confirm their applicability to robots with moving cameras. Unlike prior work with stationary camera setups, our detector is specifically designed for the egocentric, dynamic viewpoints of mobile robots, providing a foundational building block for downstream tasks, including tracking, trajectory estimation, and system (re)-initialization.
>
---
#### [new 141] A Recipe for Stable Offline Multi-agent Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于多智能体强化学习任务，解决离线MARL稳定性问题。针对非线性价值分解的不稳定性，提出SVN技术，提升训练稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2603.08399](https://arxiv.org/pdf/2603.08399)**

> **作者:** Dongsu Lee; Daehee Lee; Amy Zhang
>
> **备注:** Preprint
>
> **摘要:** Despite remarkable achievements in single-agent offline reinforcement learning (RL), multi-agent RL (MARL) has struggled to adopt this paradigm, largely persisting with on-policy training and self-play from scratch. One reason for this gap comes from the instability of non-linear value decomposition, leading prior works to avoid complex mixing networks in favor of linear value decomposition (e.g., VDN) with value regularization used in single-agent setups. In this work, we analyze the source of instability in non-linear value decomposition within the offline MARL setting. Our observations confirm that they induce value-scale amplification and unstable optimization. To alleviate this, we propose a simple technique, scale-invariant value normalization (SVN), that stabilizes actor-critic training without altering the Bellman fixed point. Empirically, we examine the interaction among key components of offline MARL (e.g., value decomposition, value learning, and policy extraction) and derive a practical recipe that unlocks its full potential.
>
---
#### [new 142] Faster-HEAL: An Efficient and Privacy-Preserving Collaborative Perception Framework for Heterogeneous Autonomous Vehicles
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的协同感知任务，旨在解决异构车辆间特征域差异导致的检测性能下降问题。提出Faster-HEAL框架，通过低秩视觉提示和金字塔融合实现高效隐私保护的特征对齐。**

- **链接: [https://arxiv.org/pdf/2603.07314](https://arxiv.org/pdf/2603.07314)**

> **作者:** Armin Maleki; Hayder Radha
>
> **备注:** Accepted to appear in the 2026 IEEE Intelligent Vehicles Symposium (IV 2026), Detroit, MI, USA, June 22-25, 2026. 6 pages, 1 figure, 4 tables
>
> **摘要:** Collaborative perception (CP) is a promising paradigm for improving situational awareness in autonomous vehicles by overcoming the limitations of single-agent perception. However, most existing approaches assume homogeneous agents, which restricts their applicability in real-world scenarios where vehicles use diverse sensors and perception models. This heterogeneity introduces a feature domain gap that degrades detection performance. Prior works address this issue by retraining entire models/major components, or using feature interpreters for each new agent type, which is computationally expensive, compromises privacy, and may reduce single-agent accuracy. We propose Faster-HEAL, a lightweight and privacy-preserving CP framework that fine-tunes a low-rank visual prompt to align heterogeneous features with a unified feature space while leveraging pyramid fusion for robust feature aggregation. This approach reduces the trainable parameters by 94%, enabling efficient adaptation to new agents without retraining large models. Experiments on the OPV2V-H dataset show that Faster-HEAL improves detection performance by 2% over state-of-the-art methods with significantly lower computational overhead, offering a practical solution for scalable heterogeneous CP.
>
---
#### [new 143] ReconDrive: Fast Feed-Forward 4D Gaussian Splatting for Autonomous Driving Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶场景重建任务，旨在解决4D高斯溅射的效率与质量问题。提出ReconDrive框架，通过改进模型结构实现快速高质量重建。**

- **链接: [https://arxiv.org/pdf/2603.07552](https://arxiv.org/pdf/2603.07552)**

> **作者:** Haibao Yu; Kuntao Xiao; Jiahang Wang; Ruiyang Hao; Yuxin Huang; Guoran Hu; Haifang Qin; Bowen Jing; Yuntian Bo; Ping Luo
>
> **摘要:** High-fidelity visual reconstruction and novel-view synthesis are essential for realistic closed-loop evaluation in autonomous driving. While 4D Gaussian Splatting (4DGS) offers a promising balance of accuracy and efficiency, existing per-scene optimization methods require costly iterative refinement, rendering them unscalable for extensive urban environments. Conversely, current feed-forward approaches often suffer from degraded photometric quality. To address these limitations, we propose ReconDrive, a feed-forward framework that leverages and extends the 3D foundation model VGGT for rapid, high-fidelity 4DGS generation. Our architecture introduces two core adaptations to tailor the foundation model to dynamic driving scenes: (1) Hybrid Gaussian Prediction Heads, which decouple the regression of spatial coordinates and appearance attributes to overcome the photometric deficiencies inherent in generalized foundation features; and (2) a Static-Dynamic 4D Composition strategy that explicitly captures temporal motion via velocity modeling to represent complex dynamic environments. Benchmarked on nuScenes, ReconDrive significantly outperforms existing feed-forward baselines in reconstruction, novel-view synthesis, and 3D perception. It achieves performance competitive with per-scene optimization while being orders of magnitude faster, providing a scalable and practical solution for realistic driving simulation.
>
---
#### [new 144] A Lightweight Digital-Twin-Based Framework for Edge-Assisted Vehicle Tracking and Collision Prediction
- **分类: cs.CV; cs.NI; cs.RO; eess.SP**

- **简介: 该论文属于智能交通系统中的车辆跟踪与碰撞预测任务，旨在解决边缘设备计算资源有限的问题。通过轻量级数字孪生框架，利用目标检测实现高效跟踪与碰撞预测。**

- **链接: [https://arxiv.org/pdf/2603.07338](https://arxiv.org/pdf/2603.07338)**

> **作者:** Murat Arda Onsu; Poonam Lohan; Burak Kantarci; Aisha Syed; Matthew Andrews; Sean Kennedy
>
> **备注:** 6 pages, 2 figures, IEEE ICC 2026 Workshops (under submission)
>
> **摘要:** Vehicle tracking, motion estimation, and collision prediction are fundamental components of traffic safety and management in Intelligent Transportation Systems (ITS). Many recent approaches rely on computationally intensive prediction models, which limits their practical deployment on resource-constrained edge devices. This paper presents a lightweight digital-twin-based framework for vehicle tracking and spatiotemporal collision prediction that relies solely on object detection, without requiring complex trajectory prediction networks. The framework is implemented and evaluated in Quanser Interactive Labs (QLabs), a high-fidelity digital twin of an urban traffic environment that enables controlled and repeatable scenario generation. A YOLO-based detector is deployed on simulated edge cameras to localize vehicles and extract frame-level centroid trajectories. Offline path maps are constructed from multiple traversals and indexed using K-D trees to support efficient online association between detected vehicles and road segments. During runtime, consistent vehicle identifiers are maintained, vehicle speed and direction are estimated from the temporal evolution of path indices, and future positions are predicted accordingly. Potential collisions are identified by analyzing both spatial proximity and temporal overlap of predicted future trajectories. Our experimental results across diverse simulated urban scenarios show that the proposed framework predicts approximately 88% of collision events prior to occurrence while maintaining low computational overhead suitable for edge deployment. Rather than introducing a computationally intensive prediction model, this work introduces a lightweight digital-twin-based solution for vehicle tracking and collision prediction, tailored for real-time edge deployment in ITS.
>
---
#### [new 145] SLNet: A Super-Lightweight Geometry-Adaptive Network for 3D Point Cloud Recognition
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出SLNet，用于3D点云识别任务，旨在以轻量模型实现高精度。通过NAPE和GMU结构，在保持性能的同时显著降低计算成本。**

- **链接: [https://arxiv.org/pdf/2603.07454](https://arxiv.org/pdf/2603.07454)**

> **作者:** Mohammad Saeid; Amir Salarpour; Pedram MohajerAnsari; Mert D. Pesé
>
> **备注:** Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** We present SLNet, a lightweight backbone for 3D point cloud recognition designed to achieve strong performance without the computational cost of many recent attention, graph, and deep MLP based models. The model is built on two simple ideas: NAPE (Nonparametric Adaptive Point Embedding), which captures spatial structure using a combination of Gaussian RBF and cosine bases with input adaptive bandwidth and blending, and GMU (Geometric Modulation Unit), a per channel affine modulator that adds only 2D learnable parameters. These components are used within a four stage hierarchical encoder with FPS+kNN grouping, nonparametric normalization, and shared residual MLPs. In experiments, SLNet shows that a very small model can still remain highly competitive across several 3D recognition tasks. On ModelNet40, SLNet-S with 0.14M parameters and 0.31 GFLOPs achieves 93.64% overall accuracy, outperforming PointMLP-elite with 5x fewer parameters, while SLNet-M with 0.55M parameters and 1.22 GFLOPs reaches 93.92%, exceeding PointMLP with 24x fewer parameters. On ScanObjectNN, SLNet-M achieves 84.25% overall accuracy within 1.2 percentage points of PointMLP while using 28x fewer parameters. For large scale scene segmentation, SLNet-T extends the backbone with local Point Transformer attention and reaches 58.2% mIoU on S3DIS Area 5 with only 2.5M parameters, more than 17x fewer than Point Transformer V3. We also introduce NetScore+, which extends NetScore by incorporating latency and peak memory so that efficiency can be evaluated in a more deployment oriented way. Across multiple benchmarks and hardware settings, SLNet delivers a strong overall balance between accuracy and efficiency. Code is available at: this https URL.
>
---
#### [new 146] TeamHOI: Learning a Unified Policy for Cooperative Human-Object Interactions with Any Team Size
- **分类: cs.CV; cs.GR; cs.MA; cs.RO**

- **简介: 该论文属于多智能体协作任务，旨在解决多人类-物体交互的协同控制问题。通过统一策略实现不同规模团队的协作，提升运动真实性和任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.07988](https://arxiv.org/pdf/2603.07988)**

> **作者:** Stefan Lionar; Gim Hee Lee
>
> **备注:** CVPR 2026. Project page: this https URL Code: this https URL
>
> **摘要:** Physics-based humanoid control has achieved remarkable progress in enabling realistic and high-performing single-agent behaviors, yet extending these capabilities to cooperative human-object interaction (HOI) remains challenging. We present TeamHOI, a framework that enables a single decentralized policy to handle cooperative HOIs across any number of cooperating agents. Each agent operates using local observations while attending to other teammates through a Transformer-based policy network with teammate tokens, allowing scalable coordination across variable team sizes. To enforce motion realism while addressing the scarcity of cooperative HOI data, we further introduce a masked Adversarial Motion Prior (AMP) strategy that uses single-human reference motions while masking object-interacting body parts during training. The masked regions are then guided through task rewards to produce diverse and physically plausible cooperative behaviors. We evaluate TeamHOI on a challenging cooperative carrying task involving two to eight humanoid agents and varied object geometries. Finally, to promote stable carrying, we design a team-size- and shape-agnostic formation reward. TeamHOI achieves high success rates and demonstrates coherent cooperation across diverse configurations with a single policy.
>
---
#### [new 147] Improved Constrained Generation by Bridging Pretrained Generative Models
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于约束生成任务，旨在解决在复杂约束下生成真实样本的问题。工作是通过微调预训练生成模型，在保持生成质量的同时满足约束条件。**

- **链接: [https://arxiv.org/pdf/2603.06742](https://arxiv.org/pdf/2603.06742)**

> **作者:** Xiaoxuan Liang; Saeid Naderiparizi; Yunpeng Liu; Berend Zwartsenberg; Frank Wood
>
> **摘要:** Constrained generative modeling is fundamental to applications such as robotic control and autonomous driving, where models must respect physical laws and safety-critical constraints. In real-world settings, these constraints rarely take the form of simple linear inequalities, but instead complex feasible regions that resemble road maps or other structured spatial domains. We propose a constrained generation framework that generates samples directly within such feasible regions while preserving realism. Our method fine-tunes a pretrained generative model to enforce constraints while maintaining generative fidelity. Experimentally, our method exhibits characteristics distinct from existing fine-tuning and training-free constrained baselines, revealing a new compromise between constraint satisfaction and sampling quality.
>
---
#### [new 148] Spherical-GOF: Geometry-Aware Panoramic Gaussian Opacity Fields for 3D Scene Reconstruction
- **分类: cs.CV; cs.GR; cs.RO; eess.IV**

- **简介: 该论文属于3D场景重建任务，旨在解决全景图像中3D高斯点云渲染的几何不一致问题。提出Spherical-GOF框架，实现更精确的全景渲染与深度估计。**

- **链接: [https://arxiv.org/pdf/2603.08503](https://arxiv.org/pdf/2603.08503)**

> **作者:** Zhe Yang; Guoqiang Zhao; Sheng Wu; Kai Luo; Kailun Yang
>
> **备注:** The source code and dataset will be released at this https URL
>
> **摘要:** Omnidirectional images are increasingly used in robotics and vision due to their wide field of view. However, extending 3D Gaussian Splatting (3DGS) to panoramic camera models remains challenging, as existing formulations are designed for perspective projections and naive adaptations often introduce distortion and geometric inconsistencies. We present Spherical-GOF, an omnidirectional Gaussian rendering framework built upon Gaussian Opacity Fields (GOF). Unlike projection-based rasterization, Spherical-GOF performs GOF ray sampling directly on the unit sphere in spherical ray space, enabling consistent ray-Gaussian interactions for panoramic rendering. To make the spherical ray casting efficient and robust, we derive a conservative spherical bounding rule for fast ray-Gaussian culling and introduce a spherical filtering scheme that adapts Gaussian footprints to distortion-varying panoramic pixel sampling. Extensive experiments on standard panoramic benchmarks (OmniBlender and OmniPhotos) demonstrate competitive photometric quality and substantially improved geometric consistency. Compared with the strongest baseline, Spherical-GOF reduces depth reprojection error by 57% and improves cycle inlier ratio by 21%. Qualitative results show cleaner depth and more coherent normal maps, with strong robustness to global panorama rotations. We further validate generalization on OmniRob, a real-world robotic omnidirectional dataset introduced in this work, featuring UAV and quadruped platforms. The source code and the OmniRob dataset will be released at this https URL.
>
---
#### [new 149] OccTrack360: 4D Panoptic Occupancy Tracking from Surround-View Fisheye Cameras
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出OccTrack360基准和FoSOcc框架，解决环绕鱼眼相机的4D全景占用跟踪问题，提升动态3D环境理解的准确性与连续性。**

- **链接: [https://arxiv.org/pdf/2603.08521](https://arxiv.org/pdf/2603.08521)**

> **作者:** Yongzhi Lin; Kai Luo; Yuanfan Zheng; Hao Shi; Mengfei Duan; Yang Liu; Kailun Yang
>
> **备注:** The benchmark and source code will be made publicly available at this https URL
>
> **摘要:** Understanding dynamic 3D environments in a spatially continuous and temporally consistent manner is fundamental for robotics and autonomous driving. While recent advances in occupancy prediction provide a unified representation of scene geometry and semantics, progress in 4D panoptic occupancy tracking remains limited by the lack of benchmarks that support surround-view fisheye sensing, long temporal sequences, and instance-level voxel tracking. To address this gap, we present OccTrack360, a new benchmark for 4D panoptic occupancy tracking from surround-view fisheye cameras. OccTrack360 provides substantially longer and more diverse sequences (174~2234 frames) than prior benchmarks, together with principled voxel visibility annotations, including an all-direction occlusion mask and an MEI-based fisheye field-of-view mask. To establish a strong fisheye-oriented baseline, we further propose Focus on Sphere Occ (FoSOcc), a framework that addresses two core challenges in fisheye occupancy tracking: distorted spherical projection and inaccurate voxel-space localization. FoSOcc includes a Center Focusing Module (CFM) to enhance instance-aware spatial localization through supervised focus guidance, and a Spherical Lift Module (SLM) that extends perspective lifting to fisheye imaging under the Unified Projection Model. Extensive experiments on Occ3D-Waymo and OccTrack360 show that our method improves occupancy tracking quality with notable gains on geometrically regular categories, and establishes a strong baseline for future research on surround-view fisheye 4D occupancy tracking. The benchmark and source code will be made publicly available at this https URL.
>
---
#### [new 150] RAPID: Redundancy-Aware and Compatibility-Optimal Edge-Cloud Partitioned Inference for Diverse VLA models
- **分类: cs.DC; cs.RO**

- **简介: 该论文属于边缘云协同推理任务，旨在解决VLA模型高推理成本及冗余问题。提出RAPID框架，优化分区策略，提升效率并保持物理连续性。**

- **链接: [https://arxiv.org/pdf/2603.07949](https://arxiv.org/pdf/2603.07949)**

> **作者:** Zihao Zheng; Sicheng Tian; Hangyu Cao; Chenyue Li; Jiayu Chen; Maoliang Li; Xinhao Sun; Hailong Zou; Guojie Luo; Xiang Chen
>
> **摘要:** Vision Language Action (VLA) models are mainstream in embodied intelligence but face high inference costs. Edge-Cloud Collaborative (ECC) inference offers an effective fix by easing edge-device computing pressure to meet real-time needs. However, existing ECC frameworks are suboptimal for VLA models due to two challenges: (1) Mainstream environment-oriented edge-cloud partitioning methods are susceptible to interference from visual noise; (2) Existing edge-cloud partitioning methods overlook the step-wise redundancy unique to embodied tasks, thereby disrupting the physical continuity of motion. To address these issues, we propose a novel ECC inference framework, termed RAPID. Specifically, we developed an implementation tailored to the proposed framework. Experiments demonstrate this achieves a speedup of up to 1.73x with only 5%~7% overhead.
>
---
#### [new 151] Edged USLAM: Edge-Aware Event-Based SLAM with Learning-Based Depth Priors
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位与建图任务，解决传统SLAM在极端条件下的失效问题。通过融合事件相机与IMU，提出Edged USLAM系统，提升稳定性与精度。**

- **链接: [https://arxiv.org/pdf/2603.08150](https://arxiv.org/pdf/2603.08150)**

> **作者:** Şebnem Sarıözkan; Hürkan Şahin; Olaya Álvarez-Tuñón; Erdal Kayacan
>
> **备注:** 8 pages, 7 figures, 3 tables. Accepted to ICRA 2026. Project code and datasets available at this https URL
>
> **摘要:** Conventional visual simultaneous localization and mapping (SLAM) algorithms often fail under rapid motion, low illumination, or abrupt lighting transitions due to motion blur and limited dynamic range. Event cameras mitigate these issues with high temporal resolution and high dynamic range (HDR), but their sparse, asynchronous outputs complicate feature extraction and integration with other sensors; e.g. inertial measurement units (IMUs) and standard cameras. We present Edged USLAM, a hybrid visual-inertial system that extends Ultimate SLAM (USLAM) with an edge-aware front-end and a lightweight depth module. The frontend enhances event frames for robust feature tracking and nonlinear motion compensation, while the depth module provides coarse, region-of-interest (ROI)-based scene depth to improve motion compensation and scale consistency. Evaluations across public benchmarks and real-world unmanned air vehicle (UAV) flights demonstrate that performance varies significantly by scenario. For instance, event-only methods like point-line event-based visual-inertial odometry (PL-EVIO) or learning-based pipelines such as deep event-based visual odometry (DEVO) excel in highly aggressive or extreme HDR conditions. In contrast, Edged USLAM provides superior stability and minimal drift in slow or structured trajectories, ensuring consistently accurate localization on real flights under challenging illumination. These findings highlight the complementary strengths of event-only, learning-based, and hybrid approaches, while positioning Edged USLAM as a robust solution for diverse aerial navigation tasks.
>
---
#### [new 152] Trajectory Tracking Control Design for Autonomous Helicopters with Guaranteed Error Bounds
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于自主直升机轨迹跟踪控制任务，解决如何计算并保证轨迹跟踪误差边界的问题。通过构建闭环误差动态模型，利用RPI集方法得出位置误差上限，用于轨迹规划的缓冲区设计。**

- **链接: [https://arxiv.org/pdf/2603.08045](https://arxiv.org/pdf/2603.08045)**

> **作者:** Philipp Schitz; Johann C. Dauer; Paolo Mercorelli
>
> **备注:** Submitted to the 2026 International Conference on Unmanned Aircraft Systems (ICUAS)
>
> **摘要:** This paper presents a systematic framework for computing formally guaranteed trajectory tracking error bounds for autonomous helicopters based on Robust Positive Invariant (RPI) sets. The approach focuses on establishing a closed-loop translational error dynamics which is cast into polytopic linear parameter-varying form with bounded additive and state-dependent disturbances. Ellipsoidal RPI sets are computed, yielding explicit position error bounds suitable as certified buffer zones in upper-level trajectory planning. Three controller architectures are compared with respect to the conservatism of their error bounds and tracking performance. Simulation results on a nonlinear helicopter model demonstrate that all architectures respect the derived bounds, while highlighting trade-offs between dynamical fidelity and conservatism in invariant set computation.
>
---
#### [new 153] TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings
- **分类: cs.CV; cs.CL; cs.ET; cs.MM; cs.RO**

- **简介: 该论文属于视觉-语言模型的地理时间理解任务，旨在解决模型在真实场景中对时空属性推理能力不足的问题。作者构建了TimeSpot基准数据集，用于评估和提升模型的geo-temporal推理能力。**

- **链接: [https://arxiv.org/pdf/2603.06687](https://arxiv.org/pdf/2603.06687)**

> **作者:** Azmine Toushik Wasi; Shahriyar Zaman Ridoy; Koushik Ahamed Tonmoy; Kinga Tshering; S. M. Muhtasimul Hasan; Wahid Faisal; Tasnim Mohiuddin; Md Rizwan Parvez
>
> **备注:** 66 Pages. In Review
>
> **摘要:** Geo-temporal understanding, the ability to infer location, time, and contextual properties from visual input alone, underpins applications such as disaster management, traffic planning, embodied navigation, world modeling, and geography education. Although recent vision-language models (VLMs) have advanced image geo-localization using cues like landmarks and road signs, their ability to reason about temporal signals and physically grounded spatial cues remains limited. To address this gap, we introduce TimeSpot, a benchmark for evaluating real-world geo-temporal reasoning in VLMs. TimeSpot comprises 1,455 ground-level images from 80 countries and requires structured prediction of temporal attributes (season, month, time of day, daylight phase) and geographic attributes (continent, country, climate zone, environment type, latitude-longitude) directly from visual evidence. It also includes spatial-temporal reasoning tasks that test physical plausibility under real-world uncertainty. Evaluations of state-of-the-art open- and closed-source VLMs show low performance, particularly for temporal inference. While supervised fine-tuning yields improvements, results remain insufficient, highlighting the need for new methods to achieve robust, physically grounded geo-temporal understanding. TimeSpot is available at: this https URL.
>
---
#### [new 154] Inverse-dynamics observer design for a linear single-track vehicle model with distributed tire dynamics
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于车辆状态估计任务，旨在准确估算侧滑角和轮胎力。通过结合线性单轨模型与分布轮胎动力学，利用动态逆方法从传感器数据中重构车辆状态。**

- **链接: [https://arxiv.org/pdf/2603.07499](https://arxiv.org/pdf/2603.07499)**

> **作者:** Luigi Romano; Ole Morten Aamo; Jan Åslund; Erik Frisk
>
> **备注:** 6 pages, 5 figures. Accepted at ECC 2026
>
> **摘要:** Accurate estimation of the vehicle's sideslip angle and tire forces is essential for enhancing safety and handling performances in unknown driving scenarios. To this end, the present paper proposes an innovative observer that combines a linear single-track model with a distributed representation of the tires and information collected from standard sensors. In particular, by adopting a comprehensive representation of the tires in terms of hyperbolic partial differential equations (PDEs), the proposed estimation strategy exploits dynamical inversion to reconstruct the lumped and distributed vehicle states solely from yaw rate and lateral acceleration measurements. Simulation results demonstrate the effectiveness of the observer in estimating the sideslip angle and tire forces even in the presence of noise and model uncertainties.
>
---
#### [new 155] Directing the Robot: Scaffolding Creative Human-AI-Robot Interaction
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决传统交互中人类被定位为监督者而非合作者的问题。通过提出“支架”概念，强调人类在创意与协作中的主导作用，促进AI与机器人的协同工作。**

- **链接: [https://arxiv.org/pdf/2603.07748](https://arxiv.org/pdf/2603.07748)**

> **作者:** Jordan Aiko Deja; Isidro Butaslac; Nicko Reginio Caluya; Maheshya Weerasinghe
>
> **备注:** 4 pages, 1 figure
>
> **摘要:** Robots are moving beyond industrial settings into creative, educational, and public environments where interaction is open-ended and improvisational. Yet much of human-AI-robot interaction remains framed around performance and efficiency, positioning humans as supervisors rather than collaborators. We propose a re-framing of AI interaction with robots as scaffolding: infrastructure that enables humans to shape robotic behaviour over time while remaining meaningfully in control. Through scenarios from creative practice, learning-by-teaching, and embodied interaction, we illustrate how humans can act as executive directors, defining intent and steering revisions, while AI mediates between human expression and robotic execution. We outline design and evaluation implications that foreground creativity, agency, and flow. Finally, we discuss open challenges in social, scalable, and mission-critical contexts. We invite the community to rethink interacting with Robots and AI not as autonomy, but as sustained support for human creativity.
>
---
#### [new 156] MRDrive: An Open Source Mixed Reality Driving Simulator for Automotive User Research
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出MRDrive，一种混合现实驾驶模拟器，用于汽车用户研究。解决传统模拟器在真实交互与灵活性间的平衡问题，支持人机交互、注意力等研究。**

- **链接: [https://arxiv.org/pdf/2603.08080](https://arxiv.org/pdf/2603.08080)**

> **作者:** Patrick Ebel; Michał Patryk Miazga; Martin Lorenz; Timur Getselev; Pavlo Bazilinskyy; Celine Conzen
>
> **备注:** This version has been accepted at CHI 2026
>
> **摘要:** Designing and evaluating in-vehicle interfaces requires experimental platforms that combine ecological validity with experimental control. Driving simulators are widely used for this purpose. However, they face a fundamental trade-off: high-fidelity physical simulators are costly and difficult to adapt, while virtual reality simulators provide flexibility at the expense of physical interaction with the vehicle. In this work, we present MRDrive, an open mixed-reality driving simulator designed to support HCI research on in-vehicle interaction, attention, and explainability in manual and automated driving contexts. MRDrive enables drivers and passengers to interact with a real vehicle cabin while being fully immersed in a virtual driving environment. We demonstrate the capabilities of MRDrive through a small pilot study that illustrates how the simulator can be used to collect and analyze eye-tracking and touch interaction data in an automated driving scenario. MRDRive is available at: this https URL
>
---
#### [new 157] MWM: Mobile World Models for Action-Conditioned Consistent Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决世界模型在多步预测中的动作一致性问题。提出MWM框架，结合结构预训练和动作一致性训练，提升预测一致性与推理效率。**

- **链接: [https://arxiv.org/pdf/2603.07799](https://arxiv.org/pdf/2603.07799)**

> **作者:** Han Yan; Zishang Xiang; Zeyu Zhang; Hao Tang
>
> **摘要:** World models enable planning in imagined future predicted space, offering a promising framework for embodied navigation. However, existing navigation world models often lack action-conditioned consistency, so visually plausible predictions can still drift under multi-step rollout and degrade planning. Moreover, efficient deployment requires few-step diffusion inference, but existing distillation methods do not explicitly preserve rollout consistency, creating a training-inference mismatch. To address these challenges, we propose MWM, a mobile world model for planning-based image-goal navigation. Specifically, we introduce a two-stage training framework that combines structure pretraining with Action-Conditioned Consistency (ACC) post-training to improve action-conditioned rollout consistency. We further introduce Inference-Consistent State Distillation (ICSD) for few-step diffusion distillation with improved rollout consistency. Our experiments on benchmark and real-world tasks demonstrate consistent gains in visual fidelity, trajectory accuracy, planning success, and inference efficiency. Code: this https URL. Website: this https URL.
>
---
#### [new 158] DyQ-VLA: Temporal-Dynamic-Aware Quantization for Embodied Vision-Language-Action Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型优化任务，解决静态量化在时序动态和实时性上的不足。提出DyQ-VLA动态量化框架，提升模型效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.07904](https://arxiv.org/pdf/2603.07904)**

> **作者:** Zihao Zheng; Hangyu Cao; Sicheng Tian; Jiayu Chen; Maoliang Li; Xinhao Sun; Hailong Zou; Zhaobo Zhang; Xuanzhe Liu; Donggang Cao; Hong Mei; Xiang Chen
>
> **摘要:** Vision-Language-Action (VLA) models are dominant in embodied intelligence but are constrained by inference overheads. While model quantization alleviates these bottlenecks for edge deployment, static quantization approaches remain suboptimal for VLAs due to two critical challenges: (1) Temporal-dynamic sensitivity, where fixed precision wastes resources by ignoring stage-varying error tolerances; and (2) Real-time allocation, where identifying real-time sensitivity to guide bit allocation remains unsolved. To address these challenges, we propose DyQ-VLA, a dynamic quantization framework for VLAs. Specifically, a sensitivity-aware switching strategy leverages real-time kinematic proxies to trigger the bit-width switch, while a kinematic-guided module dynamically allocates the optimal bit-width. Experiments show that DyQ-VLA requires only 30.9% of the original memory footprint while maintaining 99.5% of its original performance, achieving 1.49x simulation and up to 1.43x real-world speedups.
>
---
#### [new 159] Don't Freeze, Don't Crash: Extending the Safe Operating Range of Neural Navigation in Dense Crowds
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于机器人导航任务，解决密集人群中的安全导航问题。通过强化学习方法，提升模型在不同密度人群中的泛化能力，减少碰撞和冻结现象。**

- **链接: [https://arxiv.org/pdf/2603.06729](https://arxiv.org/pdf/2603.06729)**

> **作者:** Jiefu Zhang; Yang Xu; Vaneet Aggarwal
>
> **摘要:** Navigating safely through dense crowds requires collision avoidance that generalizes beyond the densities seen during training. Learning-based crowd navigation can break under out-of-distribution crowd sizes due to density-sensitive observation normalization and social-cost scaling, while analytical solvers often remain safe but freeze in tight interactions. We propose a reinforcement learning approach for dense, variable-density navigation that attains zero-shot density generalization using a density-invariant observation encoding with density-randomized training and physics-informed proxemic reward shaping with density-adaptive scaling. The encoding represents the distance-sorted $K$ nearest pedestrians plus bounded crowd summaries, keeping input statistics stable as crowd size grows. Trained with $N\!\in\![11,16]$ pedestrians in a $3\mathrm{m}\times3\mathrm{m}$ arena and evaluated up to $N\!=\!21$ pedestrians ($1.3\times$ denser), our policy reaches the goal in $>99\%$ of episodes and achieves $86\%$ collision-free success in random crowds, with markedly less freezing than analytical methods and a $>\!60$-point collision-free margin over learning-based benchmark methods. Codes are available at \href{this https URL}{this https URL}.
>
---
#### [new 160] MotionBits: Video Segmentation through Motion-Level Analysis of Rigid Bodies
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视频分割任务，旨在解决运动刚体的准确分割问题。提出MotionBit概念及无需学习的图方法，在基准上提升性能37.3%。**

- **链接: [https://arxiv.org/pdf/2603.06846](https://arxiv.org/pdf/2603.06846)**

> **作者:** Howard H. Qian; Kejia Ren; Yu Xiang; Vicente Ordonez; Kaiyu Hang
>
> **备注:** 23 pages, 18 figures
>
> **摘要:** Rigid bodies constitute the smallest manipulable elements in the real world, and understanding how they physically interact is fundamental to embodied reasoning and robotic manipulation. Thus, accurate detection, segmentation, and tracking of moving rigid bodies is essential for enabling reasoning modules to interpret and act in diverse environments. However, current segmentation models trained on semantic grouping are limited in their ability to provide meaningful interaction-level cues for completing embodied tasks. To address this gap, we introduce MotionBit, a novel concept that, unlike prior formulations, defines the smallest unit in motion-based segmentation through kinematic spatial twist equivalence, independent of semantics. In this paper, we contribute (1) the MotionBit concept and definition, (2) a hand-labeled benchmark, called MoRiBo, for evaluating moving rigid-body segmentation across robotic manipulation and human-in-the-wild videos, and (3) a learning-free graph-based MotionBits segmentation method that outperforms state-of-the-art embodied perception methods by 37.3\% in macro-averaged mIoU on the MoRiBo benchmark. Finally, we demonstrate the effectiveness of MotionBits segmentation for downstream embodied reasoning and manipulation tasks, highlighting its importance as a fundamental primitive for understanding physical interactions.
>
---
#### [new 161] FOMO-3D: Using Vision Foundation Models for Long-Tailed 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D目标检测任务，旨在解决长尾分布数据下的检测难题。通过融合视觉基础模型的语义和深度先验，提出FOMO-3D方法提升检测性能。**

- **链接: [https://arxiv.org/pdf/2603.08611](https://arxiv.org/pdf/2603.08611)**

> **作者:** Anqi Joyce Yang; James Tu; Nikita Dvornik; Enxu Li; Raquel Urtasun
>
> **备注:** Published at 9th Annual Conference on Robot Learning (CoRL 2025)
>
> **摘要:** In order to navigate complex traffic environments, self-driving vehicles must recognize many semantic classes pertaining to vulnerable road users or traffic control devices. However, many safety-critical objects (e.g., construction worker) appear infrequently in nominal traffic conditions, leading to a severe shortage of training examples from driving data alone. Recent vision foundation models, which are trained on a large corpus of data, can serve as a good source of external prior knowledge to improve generalization. We propose FOMO-3D, the first multi-modal 3D detector to leverage vision foundation models for long-tailed 3D detection. Specifically, FOMO-3D exploits rich semantic and depth priors from OWLv2 and Metric3Dv2 within a two-stage detection paradigm that first generates proposals with a LiDAR-based branch and a novel camera-based branch, and refines them with attention especially to image features from OWL. Evaluations on real-world driving data show that using rich priors from vision foundation models with careful multi-modal fusion designs leads to large gains for long-tailed 3D detection. Project website is at this https URL.
>
---
#### [new 162] Fusion-Poly: A Polyhedral Framework Based on Spatial-Temporal Fusion for 3D Multi-Object Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D多目标跟踪任务，解决异步传感器数据融合问题。提出Fusion-Poly框架，实现时空联合跟踪，提升轨迹一致性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.08199](https://arxiv.org/pdf/2603.08199)**

> **作者:** Xian Wu; Yitao Wu; Xiaoyu Li; Zijia Li; Lijun Zhao; Lining Sun
>
> **摘要:** LiDAR-camera 3D multi-object tracking (MOT) combines rich visual semantics with accurate depth cues to improve trajectory consistency and tracking reliability. In practice, however, LiDAR and cameras operate at different sampling rates. To maintain temporal alignment, existing data pipelines usually synchronize heterogeneous sensor streams and annotate them at a reduced shared frequency, forcing most prior methods to perform spatial fusion only at synchronized timestamps through projection-based or learnable cross-sensor association. As a result, abundant asynchronous observations remain underexploited, despite their potential to support more frequent association and more robust trajectory estimation over short temporal intervals. To address this limitation, we propose Fusion-Poly, a spatial-temporal fusion framework for 3D MOT that integrates asynchronous LiDAR and camera data. Fusion-Poly associates trajectories with multi-modal observations at synchronized timestamps and with single-modal observations at asynchronous timestamps, enabling higher-frequency updates of motion and existence states. The framework contains three key components: a frequency-aware cascade matching module that adapts to synchronized and asynchronous frames according to available detection modalities; a frequency-aware trajectory estimation module that maintains trajectories through high-frequency motion prediction, differential updates, and confidence-calibrated lifecycle management; and a full-state observation alignment module that improves cross-modal consistency at synchronized timestamps by optimizing image-projection errors. On the nuScenes test set, Fusion-Poly achieves 76.5% AMOTA, establishing a new state of the art among tracking-by-detection 3D MOT methods. Extensive ablation studies further validate the effectiveness of each component. Code will be released.
>
---
## 更新

#### [replaced 001] ViTaPEs: Visuotactile Position Encodings for Cross-Modal Alignment in Multimodal Transformers
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出ViTaPEs，解决多模态对齐问题，通过双阶段位置编码提升视觉与触觉信息融合，增强模型泛化与迁移能力。**

- **链接: [https://arxiv.org/pdf/2505.20032](https://arxiv.org/pdf/2505.20032)**

> **作者:** Fotios Lygerakis; Ozan Özdenizci; Elmar Rückert
>
> **摘要:** Tactile sensing provides local essential information that is complementary to visual perception, such as texture, compliance, and force. Despite recent advances in visuotactile representation learning, challenges remain in fusing these modalities and generalizing across tasks and environments without heavy reliance on pre-trained vision-language models. Moreover, existing methods do not study positional encodings, thereby overlooking the multi-stage spatial reasoning needed to capture fine-grained visuotactile correlations. We introduce ViTaPEs, a transformer-based architecture for learning task-agnostic visuotactile representations from paired vision and tactile inputs. Our key idea is a two-stage positional injection: local (modality-specific) positional encodings are added within each stream, and a global positional encoding is added on the joint token sequence immediately before attention, providing a shared positional vocabulary at the stage where cross-modal interaction occurs. We make the positional injection points explicit and conduct controlled ablations that isolate their effect before a token-wise nonlinearity versus immediately before self-attention. Experiments on multiple large-scale real-world datasets show that ViTaPEs not only surpasses state-of-the-art baselines across various recognition tasks but also demonstrates zero-shot generalization to unseen, out-of-domain scenarios. We further demonstrate the transfer-learning strength of ViTaPEs in a robotic grasping task, where it outperforms state-of-the-art baselines in predicting grasp success. Project page: this https URL
>
---
#### [replaced 002] OVerSeeC: Open-Vocabulary Costmap Generation from Satellite Images and Natural Language
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出OVerSeeC，解决从卫星图像和自然语言生成全局成本地图的任务。针对未知实体和动态任务需求，通过分解为解析、定位、合成模块，实现灵活、可扩展的路径规划。**

- **链接: [https://arxiv.org/pdf/2602.18606](https://arxiv.org/pdf/2602.18606)**

> **作者:** Rwik Rana; Jesse Quattrociocchi; Dongmyeong Lee; Christian Ellis; Amanda Adkins; Adam Uccello; Garrett Warnell; Joydeep Biswas
>
> **备注:** Website : this https URL
>
> **摘要:** Aerial imagery provides essential global context for autonomous navigation, enabling route planning at scales inaccessible to onboard sensing. We address the problem of generating global costmaps for long-range planning directly from satellite imagery when entities and mission-specific traversal rules are expressed in natural language at test time. This setting is challenging since mission requirements vary, terrain entities may be unknown at deployment, and user prompts often encode compositional traversal logic. Existing approaches relying on fixed ontologies and static cost mappings cannot accommodate such flexibility. While foundation models excel at language interpretation and open-vocabulary perception, no single model can simultaneously parse nuanced mission directives, locate arbitrary entities in large-scale imagery, and synthesize them into an executable cost function for planners. We therefore propose OVerSeeC, a zero-shot modular framework that decomposes the problem into Interpret-Locate-Synthesize: (i) an LLM extracts entities and ranked preferences, (ii) an open-vocabulary segmentation pipeline identifies these entities from high-resolution imagery, and (iii) the LLM uses the user's natural language preferences and masks to synthesize executable costmap code. Empirically, OVerSeeC handles novel entities, respects ranked and compositional preferences, and produces routes consistent with human-drawn trajectories across diverse regions, demonstrating robustness to distribution shifts. This shows that modular composition of foundation models enables open-vocabulary, preference-aligned costmap generation for scalable, mission-adaptive global planning.
>
---
#### [replaced 003] From Pixels to Predicates: Learning Symbolic World Models via Pretrained Vision-Language Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人决策任务，旨在解决复杂环境中的长期规划问题。通过预训练视觉-语言模型学习符号化世界模型，实现零样本泛化与新目标的达成。**

- **链接: [https://arxiv.org/pdf/2501.00296](https://arxiv.org/pdf/2501.00296)**

> **作者:** Ashay Athalye; Nishanth Kumar; Tom Silver; Yichao Liang; Jiuguang Wang; Tomás Lozano-Pérez; Leslie Pack Kaelbling
>
> **备注:** A version of this paper appears in the official proceedings of RA-L, Volume 11, Issue 4
>
> **摘要:** Our aim is to learn to solve long-horizon decision-making problems in complex robotics domains given low-level skills and a handful of short-horizon demonstrations containing sequences of images. To this end, we focus on learning abstract symbolic world models that facilitate zero-shot generalization to novel goals via planning. A critical component of such models is the set of symbolic predicates that define properties of and relationships between objects. In this work, we leverage pretrained vision-language models (VLMs) to propose a large set of visual predicates potentially relevant for decision-making, and to evaluate those predicates directly from camera images. At training time, we pass the proposed predicates and demonstrations into an optimization-based model-learning algorithm to obtain an abstract symbolic world model that is defined in terms of a compact subset of the proposed predicates. At test time, given a novel goal in a novel setting, we use the VLM to construct a symbolic description of the current world state, and then use a search-based planning algorithm to find a sequence of low-level skills that achieves the goal. We demonstrate empirically across experiments in both simulation and the real world that our method can generalize aggressively, applying its learned world model to solve problems with a wide variety of object types, arrangements, numbers of objects, and visual backgrounds, as well as novel goals and much longer horizons than those seen at training time.
>
---
#### [replaced 004] ActivePose: Active 6D Object Pose Estimation and Tracking for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于6D物体位姿估计与跟踪任务，旨在解决视角模糊和物体移动导致的定位问题。通过结合视觉语言模型和“机器人想象”，动态调整相机视角以提高精度。**

- **链接: [https://arxiv.org/pdf/2509.11364](https://arxiv.org/pdf/2509.11364)**

> **作者:** Sheng Liu; Zhe Li; Weiheng Wang; Han Sun; Heng Zhang; Hongpeng Chen; Yusen Qin; Arash Ajoudani; Yizhao Wang
>
> **备注:** 6D Pose, Diffusion Policy, Robot Learning
>
> **摘要:** Accurate 6-DoF object pose estimation and tracking are critical for reliable robotic manipulation. However, zero-shot methods often fail under viewpoint-induced ambiguities and fixed-camera setups struggle when objects move or become self-occluded. To address these challenges, we propose an active pose estimation pipeline that combines a Vision-Language Model (VLM) with "robotic imagination" to dynamically detect and resolve ambiguities in real time. In an offline stage, we render a dense set of views of the CAD model, compute the FoundationPose entropy for each view, and construct a geometric-aware prompt that includes low-entropy (unambiguous) and high-entropy (ambiguous) examples. At runtime, the system: (1) queries the VLM on the live image for an ambiguity score; (2) if ambiguity is detected, imagines a discrete set of candidate camera poses by rendering virtual views, scores each based on a weighted combination of VLM ambiguity probability and FoundationPose entropy, and then moves the camera to the Next-Best-View (NBV) to obtain a disambiguated pose estimation. Furthermore, since moving objects may leave the camera's field of view, we introduce an active pose tracking module: a diffusion-policy trained via imitation learning, which generates camera trajectories that preserve object visibility and minimize pose ambiguity. Experiments in simulation and real-world show that our approach significantly outperforms classical baselines.
>
---
#### [replaced 005] Agile in the Face of Delay: Asynchronous End-to-End Learning for Real-World Aerial Navigation
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决高频率控制与低频率感知不匹配问题。通过异步强化学习框架，实现高效、敏捷的空中导航。**

- **链接: [https://arxiv.org/pdf/2509.13816](https://arxiv.org/pdf/2509.13816)**

> **作者:** Yude Li; Zhexuan Zhou; Huizhe Li; Youmin Gong; Jie Mei
>
> **摘要:** Robust autonomous navigation for Autonomous Aerial Vehicles (AAVs) in complex environments is a critical capability. However, modern end-to-end navigation faces a key challenge: the high-frequency control loop needed for agile flight conflicts with low-frequency perception streams, which are limited by sensor update rates and significant computational cost. This mismatch forces conventional synchronous models into undesirably low control rates. To resolve this, we propose an asynchronous reinforcement learning framework that decouples perception and control, enabling a high-frequency policy to act on the latest IMU state for immediate reactivity, while incorporating perception features asynchronously. To manage the resulting data staleness, we introduce a theoretically-grounded Temporal Encoding Module (TEM) that explicitly conditions the policy on perception delays, a strategy complemented by a two-stage curriculum to ensure stable and efficient training. Validated in extensive simulations, our method was successfully deployed in zero-shot sim-to-real transfer on an onboard NUC, where it sustains a 100~Hz control rate and demonstrates robust, agile navigation in cluttered real-world environments. Our source code will be released for community reference.
>
---
#### [replaced 006] Event-Based Visual Teach-and-Repeat via Fast Fourier-Domain Cross-Correlation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉教-重复导航任务，解决机器人实时路径重现已知轨迹的问题。通过事件相机与傅里叶域互相关方法，实现高效低延迟的视觉匹配。**

- **链接: [https://arxiv.org/pdf/2509.17287](https://arxiv.org/pdf/2509.17287)**

> **作者:** Gokul B. Nair; Alejandro Fontan; Michael Milford; Tobias Fischer
>
> **备注:** 8 Pages, 5 Figures, Under Review
>
> **摘要:** Visual teach-and-repeat (VT&R) navigation enables robots to autonomously traverse previously demonstrated paths using visual feedback. We present a novel event-camera-based VT\&R system. Our system formulates event-stream matching as frequency-domain cross-correlation, transforming spatial convolutions into efficient Fourier-space multiplications. By exploiting the binary structure of event frames and applying image compression techniques, we achieve a processing latency of just 2.88 ms, about 3.5 times faster than conventional camera-based baselines that are optimised for runtime efficiency. Experiments using a Prophesee EVK4 HD event camera mounted on an AgileX Scout Mini robot demonstrate successful autonomous navigation across 3000+ meters of indoor and outdoor trajectories in daytime and nighttime conditions. Our system maintains Cross-Track Errors (XTE) below 15 cm, demonstrating the practical viability of event-based perception for real-time VT\&R navigation.
>
---
#### [replaced 007] EB-MBD: Emerging-Barrier Model-Based Diffusion for Safe Trajectory Optimization in Highly Constrained Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于安全轨迹优化任务，解决高约束环境下模型扩散算法性能下降的问题。提出EB-MBD方法，通过引入渐进障碍函数提升求解质量与效率。**

- **链接: [https://arxiv.org/pdf/2510.07700](https://arxiv.org/pdf/2510.07700)**

> **作者:** Raghav Mishra; Ian R. Manchester
>
> **备注:** Accepted to ICRA 2026. Code available at this https URL
>
> **摘要:** We propose enforcing constraints on Model-Based Diffusion by introducing emerging barrier functions inspired by interior point methods. We demonstrate that the standard Model-Based Diffusion algorithm can lead to catastrophic performance degradation in highly constrained environments, even on simple 2D systems due to sample inefficiency in the Monte Carlo approximation of the score function. We introduce Emerging-Barrier Model-Based Diffusion (EB-MBD) which uses progressively introduced barrier constraints to avoid these problems, significantly improving solution quality, without expensive projection operations such as projections. We analyze the sampling liveliness of samples at each iteration to inform barrier parameter scheduling choice. We demonstrate results for 2D collision avoidance and a 3D underwater manipulator system and show that our method achieves lower cost solutions than Model-Based Diffusion, and requires orders of magnitude less computation time than projection based methods.
>
---
#### [replaced 008] Task-Oriented Robot-Human Handovers on Legged Manipulators
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究任务导向的机器人-人类交接（TOH），解决物体交接通用性不足的问题。提出AFT-Handover框架，结合大模型与纹理传递，实现零样本泛化交接。**

- **链接: [https://arxiv.org/pdf/2602.05760](https://arxiv.org/pdf/2602.05760)**

> **作者:** Andreea Tulbure; Carmen Scheidemann; Elias Steiner; Marco Hutter
>
> **备注:** Accepted to 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI) 2026
>
> **摘要:** Task-oriented handovers (TOH) are fundamental to effective human-robot collaboration, requiring robots to present objects in a way that supports the human's intended post-handover use. Existing approaches are typically based on object- or task-specific affordances, but their ability to generalize to novel scenarios is limited. To address this gap, we present AFT-Handover, a framework that integrates large language model (LLM)-driven affordance reasoning with efficient texture-based affordance transfer to achieve zero-shot, generalizable TOH. Given a novel object-task pair, the method retrieves a proxy exemplar from a database, establishes part-level correspondences via LLM reasoning, and texturizes affordances for feature-based point cloud transfer. We evaluate AFT-Handover across diverse task-object pairs, showing improved handover success rates and stronger generalization compared to baselines. In a comparative user study, our framework is significantly preferred over the current state-of-the-art, effectively reducing human regrasping before tool use. Finally, we demonstrate TOH on legged manipulators, highlighting the potential of our framework for real-world robot-human handovers.
>
---
#### [replaced 009] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DrivingGen，用于评估自动驾驶中的生成式世界模型。解决缺乏全面基准的问题，整合多样数据与新指标，评估视觉真实、轨迹合理性等。**

- **链接: [https://arxiv.org/pdf/2601.01528](https://arxiv.org/pdf/2601.01528)**

> **作者:** Yang Zhou; Hao Shao; Letian Wang; Zhuofan Zong; Hongsheng Li; Steven L. Waslander
>
> **备注:** ICLR 2026 Poster; Project Website: this https URL
>
> **摘要:** Video generation models, as one form of world models, have emerged as one of the most exciting frontiers in AI, promising agents the ability to imagine the future by modeling the temporal evolution of complex scenes. In autonomous driving, this vision gives rise to driving world models: generative simulators that imagine ego and agent futures, enabling scalable simulation, safe testing of corner cases, and rich synthetic data generation. Yet, despite fast-growing research activity, the field lacks a rigorous benchmark to measure progress and guide priorities. Existing evaluations remain limited: generic video metrics overlook safety-critical imaging factors; trajectory plausibility is rarely quantified; temporal and agent-level consistency is neglected; and controllability with respect to ego conditioning is ignored. Moreover, current datasets fail to cover the diversity of conditions required for real-world deployment. To address these gaps, we present DrivingGen, the first comprehensive benchmark for generative driving world models. DrivingGen combines a diverse evaluation dataset curated from both driving datasets and internet-scale video sources, spanning varied weather, time of day, geographic regions, and complex maneuvers, with a suite of new metrics that jointly assess visual realism, trajectory plausibility, temporal coherence, and controllability. Benchmarking 14 state-of-the-art models reveals clear trade-offs: general models look better but break physics, while driving-specific ones capture motion realistically but lag in visual quality. DrivingGen offers a unified evaluation framework to foster reliable, controllable, and deployable driving world models, enabling scalable simulation, planning, and data-driven decision-making.
>
---
#### [replaced 010] Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于神经网络控制任务，旨在利用果蝇全脑连接组实现全身运动控制。通过构建FlyGM模型，解决传统方法需任务定制的问题，提升控制效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.17997](https://arxiv.org/pdf/2602.17997)**

> **作者:** Zehao Jin; Yaoye Zhu; Chen Zhang; Yanan Sui
>
> **摘要:** Whole-brain biological neural networks naturally support the learning and control of whole-body movements. However, the use of brain connectomes as neural network controllers in embodied reinforcement learning remains unexplored. We investigate using the exact neural architecture of an adult fruit fly's brain for the control of its body movement. We develop Fly-connectomic Graph Model (FlyGM), whose static structure is identical to the complete connectome of an adult Drosophila for whole-body locomotion control. To perform dynamical control, FlyGM represents the static connectome as a directed message-passing graph to impose a biologically grounded information flow from sensory inputs to motor outputs. Integrated with a biomechanical fruit fly model, our method achieves stable control across diverse locomotion tasks without task-specific architectural tuning. To verify the structural advantages of the connectome-based model, we compare it against a degree-preserving rewired graph, a random graph, and multilayer perceptrons, showing that FlyGM yields higher sample efficiency and superior performance. This work demonstrates that static brain connectomes can be transformed to instantiate effective neural policy for embodied learning of movement control.
>
---
#### [replaced 011] Safe Navigation of Bipedal Robots via Koopman Operator-Based Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决 bipedal 机器人在复杂环境中的安全导航问题。通过 Koopman 理论构建模型预测控制器，提升轨迹预测精度与环境适应能力。**

- **链接: [https://arxiv.org/pdf/2409.14736](https://arxiv.org/pdf/2409.14736)**

> **作者:** Jeonghwan Kim; Yunhai Han; Harish Ravichandar; Sehoon Ha
>
> **备注:** 9 pages
>
> **摘要:** Nonlinearity in dynamics has long been a major challenge in robotics, often causing significant performance degradation in existing control algorithms. For example, the navigation of bipedal robots can exhibit nonlinear behaviors even under simple velocity commands, as their actual dynamics are governed by complex whole-body movements and discrete contacts. In this work, we propose a safe navigation framework inspired by Koopman operator theory. We first train a low-level locomotion policy using deep reinforcement learning, and then capture its low-frequency, base-level dynamics by learning linearized dynamics in a high-dimensional lifted space. Then, our model-predictive controller (MPC) efficiently optimizes control signals via a standard quadratic objective and the linear dynamics constraint in the lifted space. We demonstrate that the Koopman model more accurately predicts bipedal robot trajectories than baseline approaches. We also show that the proposed navigation framework achieves improved safety with better success rates in dense environments with narrow passages.
>
---
#### [replaced 012] SeedPolicy: Horizon Scaling via Self-Evolving Diffusion Policy for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出SeedPolicy，解决长视界机器人操作中的模仿学习问题。通过自进化注意力模块，提升时间建模能力，实现高效、可扩展的策略学习。**

- **链接: [https://arxiv.org/pdf/2603.05117](https://arxiv.org/pdf/2603.05117)**

> **作者:** Youqiang Gui; Yuxuan Zhou; Shen Cheng; Xinyang Yuan; Haoqiang Fan; Peng Cheng; Shuaicheng Liu
>
> **备注:** 16 pages, 13 figures
>
> **摘要:** Imitation Learning (IL) enables robots to acquire manipulation skills from expert demonstrations. Diffusion Policy (DP) models multi-modal expert behaviors but suffers performance degradation as observation horizons increase, limiting long-horizon manipulation. We propose Self-Evolving Gated Attention (SEGA), a temporal module that maintains a time-evolving latent state via gated attention, enabling efficient recurrent updates that compress long-horizon observations into a fixed-size representation while filtering irrelevant temporal information. Integrating SEGA into DP yields Self-Evolving Diffusion Policy (SeedPolicy), which resolves the temporal modeling bottleneck and enables scalable horizon extension with moderate overhead. On the RoboTwin 2.0 benchmark with 50 manipulation tasks, SeedPolicy outperforms DP and other IL baselines. Averaged across both CNN and Transformer backbones, SeedPolicy achieves 36.8% relative improvement in clean settings and 169% relative improvement in randomized challenging settings over the DP. Compared to vision-language-action models such as RDT with 1.2B parameters, SeedPolicy achieves competitive performance with one to two orders of magnitude fewer parameters, demonstrating strong efficiency and scalability. These results establish SeedPolicy as a state-of-the-art imitation learning method for long-horizon robotic manipulation. Code is available at: this https URL.
>
---
#### [replaced 013] They See Me Rolling: High-Speed Event Vision-Based Tactile Roller Sensor for Large Surface Inspection
- **分类: cs.RO**

- **简介: 该论文属于工业表面检测任务，解决传统传感器速度慢、精度低的问题。提出一种基于事件视觉的滚动触觉传感器，实现高速高精度3D扫描。**

- **链接: [https://arxiv.org/pdf/2507.19914](https://arxiv.org/pdf/2507.19914)**

> **作者:** Akram Khairi; Hussain Sajwani; Abdallah Mohammad Alkilany; Laith AbuAssi; Mohamad Halwani; Islam Mohamed Zaid; Ahmed Awadalla; Dewald Swart; Abdulla Ayyad; Yahya Zweiri
>
> **备注:** Accepted to IEEE T-RO - Project Page: this https URL
>
> **摘要:** Inspecting large-scale industrial surfaces like aircraft fuselages for quality control requires capturing their precise 3D surface geometry at high resolution. Vision-based tactile sensors (VBTSs) offer high local resolution but require slow 'press-and-lift' measurements stitched for large areas. Approaches with sliding or roller/belt VBTS designs provide measurements continuity. However, they face significant challenges respectively: sliding struggles with friction/wear and both approaches are speed-limited by conventional camera frame rates and motion blur, making large-area scanning time consuming. Thus, a rapid, continuous, high-resolution method is needed. We introduce a novel tactile sensor integrating a neuromorphic camera in a rolling mechanism to achieve this. Leveraging its high temporal resolution and robustness to motion blur, our system uses a modified event-based multi-view stereo approach for 3D reconstruction. We demonstrate state-of-the-art scanning speeds up to 0.5 m/s, achieving Mean Absolute Error below 100 microns -- 11 times faster than prior continuous tactile sensing methods. A multi-reference Bayesian fusion strategy enhances accuracy (reducing MAE by 25.2\% compared to EMVS) and mitigates curvature errors. We also validate high-speed feature recognition via Braille reading 2.6 times faster than previous approaches.
>
---
#### [replaced 014] FreeTacMan: Robot-free Visuo-Tactile Data Collection System for Contact-rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决接触丰富操作中的数据收集问题。提出FreeTacMan系统，实现高效、准确的视觉触觉数据采集。**

- **链接: [https://arxiv.org/pdf/2506.01941](https://arxiv.org/pdf/2506.01941)**

> **作者:** Longyan Wu; Checheng Yu; Jieji Ren; Li Chen; Yufei Jiang; Ran Huang; Guoying Gu; Hongyang Li
>
> **摘要:** Enabling robots with contact-rich manipulation remains a pivotal challenge in robot learning, which is substantially hindered by the data collection gap, including its inefficiency and limited sensor setup. While prior work has explored handheld paradigms, their rod-based mechanical structures remain rigid and unintuitive, providing limited tactile feedback and posing challenges for operators. Motivated by the dexterity and force feedback of human motion, we propose FreeTacMan, a human-centric and robot-free data collection system for accurate and efficient robot manipulation. Concretely, we design a wearable gripper with visuo-tactile sensors for data collection, which can be worn by human fingers for intuitive control. A high-precision optical tracking system is introduced to capture end-effector poses while synchronizing visual and tactile feedback simultaneously. We leverage FreeTacMan to collect a large-scale multimodal dataset, comprising over 3000k paired visuo-tactile images with end-effector poses, 10k demonstration trajectories across 50 diverse contact-rich manipulation tasks. FreeTacMan achieves multiple improvements in data collection performance over prior works and enables effective policy learning from self-collected datasets. By open-sourcing the hardware and the dataset, we aim to facilitate reproducibility and support research in visuo-tactile manipulation.
>
---
#### [replaced 015] ViLAM: Distilling Vision-Language Reasoning into Attention Maps for Social Robot Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ViLAM方法，将视觉-语言推理知识提炼为注意力图，用于社交机器人导航，解决传统方法依赖专家示范的问题。**

- **链接: [https://arxiv.org/pdf/2503.09820](https://arxiv.org/pdf/2503.09820)**

> **作者:** Mohamed Elnoor; Kasun Weerakoon; Gershom Seneviratne; Jing Liang; Vignesh Rajagopal; Dinesh Manocha
>
> **摘要:** We introduce ViLAM, a novel method for distilling vision-language reasoning from large Vision-Language Models (VLMs) into spatial attention maps for socially compliant robot navigation. Unlike traditional methods that rely on expert demonstrations or human-annotated datasets, ViLAM performs knowledge distillation and fine-tuning at the intermediate layer representation (attention) level by aligning attention maps from a pretrained vision-action model with socially guided attention maps derived from a large VLM. These distilled attention maps highlight key navigational regions in a scene and serve as socially informed spatial cost maps for motion planning. To achieve this, we introduce a novel attention-level distillation loss that fuses knowledge from both sources, generating augmented attention maps with enhanced social awareness. These refined attention maps are then used as a traversability costmap within a socially aware local planner for navigation. We validate our approach through real-world experiments on a Husky wheeled robot, and demonstrate 14.2% - 50% improvements in success rate over existing methods.
>
---
#### [replaced 016] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO**

- **简介: 该论文属于AI验证任务，旨在解决MLLM在评估行为时的同意偏差问题。通过引入SGV方法，提升验证准确性与人类对齐度。**

- **链接: [https://arxiv.org/pdf/2507.11662](https://arxiv.org/pdf/2507.11662)**

> **作者:** Moises Andrade; Joonhyuk Cha; Brandon Ho; Vriksha Srihari; Karmesh Yadav; Zsolt Kira
>
> **备注:** ICLR 2026. Code, models, and data publicly available at this https URL
>
> **摘要:** Verifiers--functions assigning rewards to agent behavior--have been key to AI progress in math, code, and games. However, extending gains to domains without clear-cut success criteria remains a challenge: while humans can recognize desired outcomes, translating this intuition into scalable rules is nontrivial. Multimodal LLMs (MLLMs) offer a promising solution, given their world knowledge, human-preference alignment, and reasoning capabilities. We evaluate MLLM verifiers across web navigation, computer use, and robotics, spanning 13+ models, 28+ designs, and thousands of trajectories from diverse agents. We identify a critical limitation: a strong tendency for MLLMs to over-validate agent behavior--a phenomenon we term agreement bias. This bias is pervasive, resilient to test-time scaling, and can harm applications relying on MLLM judgments/rewards (e.g., self-improvement, steering, online supervision). We discuss several considerations for evaluating and designing MLLM verifiers, and introduce SGV, a lightweight method that better leverages their capabilities by modulating (un)conditional generation. First, an MLLM is elicited to generate broad priors about desired behavior, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. Our methods yield more human-aligned verifiers, improving failure detection by 25pp and accuracy by 14pp. In self-improvement and online supervision, they boost task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena--surpassing the previous state of the art by 20pp. As a byproduct, we release an update of VisualWebArena featuring strong agent baselines, more human-aligned oracles, container parallelism with high fidelity and proper resets, >10x speedups, and VWA-Lite, a 1/3 subset with comparable evaluation fidelity.
>
---
#### [replaced 017] Preference-Conditioned Multi-Objective RL for Integrated Command Tracking and Force Compliance in Humanoid Locomotion
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动控制任务，解决导航追踪与外部力适应的平衡问题。提出一种偏好条件下的多目标强化学习框架，实现灵活的运动策略调整。**

- **链接: [https://arxiv.org/pdf/2510.10851](https://arxiv.org/pdf/2510.10851)**

> **作者:** Tingxuan Leng; Yushi Wang; Tinglong Zheng; Changsheng Luo; Mingguo Zhao
>
> **摘要:** Humanoid locomotion requires not only accurate command tracking for navigation but also compliant responses to external forces during human interaction. Despite significant progress, existing RL approaches mainly emphasize robustness, yielding policies that resist external forces but lack compliance particularly challenging for inherently unstable humanoids. In this work, we address this by formulating humanoid locomotion as a multi-objective optimization problem that balances command tracking and external force compliance. We introduce a preference-conditioned multi-objective RL (MORL) framework that enables a single omnidirectional locomotion policy to trade off between command following and force compliance via a user-specified preference input. External forces are modeled via velocity-resistance factor for consistent reward design, and training leverages an encoder-decoder structure that infers task-relevant privileged features from deployable observations. We validate our approach in both simulation and real-world experiments on a humanoid robot. Experimental results in simulation and on hardware show that the framework trains stably and enables deployable preference-conditioned humanoid locomotion.
>
---
#### [replaced 018] GeoAware-VLA: Implicit Geometry Aware Vision-Language-Action Model
- **分类: cs.RO**

- **简介: 该论文提出GeoAware-VLA模型，解决VLA模型在未见视角下的泛化问题。通过引入几何先验增强视觉骨干，提升模型在不同视角下的表现。**

- **链接: [https://arxiv.org/pdf/2509.14117](https://arxiv.org/pdf/2509.14117)**

> **作者:** Ali Abouzeid; Malak Mansour; Qinbo Sun; Zezhou Sun; Dezhen Song
>
> **备注:** Under Review, Project Page this https URL
>
> **摘要:** Vision-Language-Action (VLA) models often fail to generalize to unseen camera viewpoints, a limitation stemming from their difficulty in inferring robust 3D geometry from 2D images. We introduce GeoAware-VLA, a simple yet effective approach that enhances viewpoint invariance by integrating strong geometric priors into the vision backbone. Instead of training a visual encoder or relying on explicit 3D data, we leverage a frozen, pretrained geometric vision model as a feature extractor. A lightweight, trainable projection layer then adapts these geometrically-rich features for the policy decoder, relieving it of the burden of learning 3D consistency from scratch. Through extensive evaluations on the LIBERO and CALVIN benchmarks, we show that GeoAware-VLA preserves and even improves in-distribution performance while achieving substantial gains in zero-shot generalization to unseen camera poses, improving unseen-view success rates by an average of 35 percentage points on LIBERO and over 11 percentage points on CALVIN compared to their respective baselines. Crucially, these gains transfer to the physical world, where our model shows significant improvement on a real robotic platform. Our approach proves effective across both continuous and discrete action spaces, highlighting that robust geometric grounding is a key ingredient for building more generalizable robotic agents.
>
---
#### [replaced 019] NaviTrace: Evaluating Embodied Navigation of Vision-Language Models
- **分类: cs.RO**

- **简介: 该论文提出NaviTrace，一个用于评估视觉-语言模型在机器人导航中表现的基准。解决真实环境测试成本高、模拟过于简化的问题，通过高质量数据集和新评分指标系统评估模型导航能力。**

- **链接: [https://arxiv.org/pdf/2510.26909](https://arxiv.org/pdf/2510.26909)**

> **作者:** Tim Windecker; Manthan Patel; Moritz Reuss; Richard Schwarzkopf; Cesar Cadena; Rudolf Lioutikov; Marco Hutter; Jonas Frey
>
> **备注:** 11 pages, 6 figures, with appendix, accepted to ICRA 2026
>
> **摘要:** Vision-language models demonstrate unprecedented performance and generalization across a wide range of tasks and scenarios. Integrating these foundation models into robotic navigation systems opens pathways toward building general-purpose robots. Yet, evaluating these models' navigation capabilities remains constrained by costly real-world trials, overly simplified simulations, and limited benchmarks. We introduce NaviTrace, a high-quality Visual Question Answering benchmark where a model receives an instruction and embodiment type (human, legged robot, wheeled robot, bicycle) and must output a 2D navigation trace in image space. Across 1000 scenarios and more than 3000 expert traces, we systematically evaluate eight state-of-the-art VLMs using a newly introduced semantic-aware trace score. This metric combines Dynamic Time Warping distance, goal endpoint error, and embodiment-conditioned penalties derived from per-pixel semantics and correlates with human preferences. Our evaluation reveals consistent gap to human performance caused by poor spatial grounding and goal localization. NaviTrace establishes a scalable and reproducible benchmark for real-world robotic navigation. The benchmark and leaderboard can be found at this https URL.
>
---
#### [replaced 020] Assigning Multi-Robot Tasks to Multitasking Robots
- **分类: cs.RO**

- **简介: 论文研究多机器人任务分配问题，解决单任务机器人效率低下的问题。提出一种考虑物理约束的多任务分配框架，并通过MAX-SAT和启发式方法求解，验证了 multitasking 的优势。**

- **链接: [https://arxiv.org/pdf/2506.15032](https://arxiv.org/pdf/2506.15032)**

> **作者:** Winston Smith; Yu Zhang
>
> **摘要:** One simplifying assumption in existing and well-performing task allocation methods is that the robots are single-tasking: each robot operates on a single task at any given time. While this assumption is harmless to make in some situations, it can be inefficient or even infeasible in others. In this paper, we consider assigning multi-robot tasks to multitasking robots. The key contribution is a novel task allocation framework that incorporates the consideration of physical constraints introduced by multitasking. This is in contrast to the existing work where such constraints are largely ignored. After formulating the problem, we propose a compilation to weighted MAX-SAT, which allows us to leverage existing solvers for a solution. A more efficient greedy heuristic is then introduced. For evaluation, we first compare our methods with a modern baseline that is efficient for single-tasking robots to validate the benefits of multitasking in synthetic domains. Then, using a site-clearing scenario in simulation, we further illustrate the complex task interaction considered by the multitasking robots in our approach to demonstrate its performance. Finally, we demonstrate a physical experiment to show how multitasking enabled by our approach can benefit task efficiency in a realistic setting.
>
---
#### [replaced 021] Vectorized Online POMDP Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人决策任务，解决部分可观测环境下的规划问题。提出VOPP方法，通过向量化计算实现高效并行POMDP求解，克服传统方法的同步瓶颈。**

- **链接: [https://arxiv.org/pdf/2510.27191](https://arxiv.org/pdf/2510.27191)**

> **作者:** Marcus Hoerger; Muhammad Sudrajat; Hanna Kurniawati
>
> **备注:** 8 pages, 3 figures. Accepted at ICRA 2026
>
> **摘要:** Planning under partial observability is an essential capability of autonomous robots. The Partially Observable Markov Decision Process (POMDP) provides a powerful framework for planning under partial observability problems, capturing the stochastic effects of actions and the limited information available through noisy observations. POMDP solving could benefit tremendously from massive parallelization on today's hardware, but parallelizing POMDP solvers has been challenging. Most solvers rely on interleaving numerical optimization over actions with the estimation of their values, which creates dependencies and synchronization bottlenecks between parallel processes that can offset the benefits of parallelization. In this paper, we propose Vectorized Online POMDP Planner (VOPP), a novel parallel online solver that leverages a recent POMDP formulation which analytically solves part of the optimization component, leaving numerical computations to consist of only estimation of expectations. VOPP represents all data structures related to planning as a collection of tensors, and implements all planning steps as fully vectorized computations over this representation. The result is a massively parallel online solver with no dependencies or synchronization bottlenecks between concurrent processes. Experimental results indicate that VOPP is at least $20\times$ more efficient in computing near-optimal solutions compared to an existing state-of-the-art parallel online solver. Moreover, VOPP outperforms state-of-the-art sequential online solvers, while using a planning budget that is $1000\times$ smaller.
>
---
#### [replaced 022] GeoNav: Empowering MLLMs with dual-scale geospatial reasoning for language-goal aerial navigation
- **分类: cs.RO**

- **简介: 该论文属于语言引导的无人机导航任务，解决复杂户外场景中目标定位问题。提出GeoNav，通过双尺度空间推理实现高效导航与精确定位。**

- **链接: [https://arxiv.org/pdf/2504.09587](https://arxiv.org/pdf/2504.09587)**

> **作者:** Haotian Xu; Yue Hu; Chen Gao; Zhengqiu Zhu; Yong Zhao; Yong Li; Quanjun Yin
>
> **备注:** Published in Pattern Recognition (2026)
>
> **摘要:** Language-goal aerial navigation requires UAVs to localize targets in the complex outdoors, such as urban blocks based on textual instructions. The indoor methods are often hard to scale to urban scenes due to ambiguous objects, limited visual field, and spatial reasoning. In this work, we propose GeoNav, a multi-modal agent for long-range aerial navigation with geospatial awareness. GeoNav operates in three phases-landmark navigation, target search, and precise localization-mimicking human coarse-to-fine spatial reasoning patterns. To support such reasoning, it dynamically builds dual-scale spatial representations. The first is a global but schematic cognitive map, which fuses prior geographic knowledge and embodied visual cues into a top-down and explicit annotated form. It enables fast navigation to the landmark region via intuitive map-based reasoning. The second is a local but delicate scene graph representing hierarchical spatial relationships between landmarks and objects, utilized for accurate target localization. On top of the structured memory, GeoNav employs a spatial chain-of-thought mechanism to enable MLLMs with efficient and interpretable action-making across stages. On the CityNav benchmark, GeoNav surpasses the current SOTA up to 18.4% in success rate and significantly eliminates navigation error. The ablation studies highlight the importance of each module, positioning structured spatial perception as the key to advanced UAV navigation. Published in Pattern Recognition, 2026.
>
---
#### [replaced 023] AgenticLab: A Real-World Robot Agent Platform that Can See, Think, and Act
- **分类: cs.RO**

- **简介: 该论文提出AgenticLab平台，解决真实机器人在开放环境中的操作问题。通过构建闭环系统，提升机器人感知、推理与执行能力。**

- **链接: [https://arxiv.org/pdf/2602.01662](https://arxiv.org/pdf/2602.01662)**

> **作者:** Pengyuan Guo; Zhonghao Mai; Zhengtong Xu; Kaidi Zhang; Heng Zhang; Zichen Miao; Arash Ajoudani; Zachary Kingston; Qiang Qiu; Yu She
>
> **备注:** Added appendix
>
> **摘要:** Recent advances in large vision-language models (VLMs) have demonstrated generalizable open-vocabulary perception and reasoning, yet their real-robot manipulation capability remains unclear for long-horizon, closed-loop execution in unstructured, in-the-wild environments. Prior VLM-based manipulation pipelines are difficult to compare across different research groups' setups, and many evaluations rely on simulation, privileged state, or specially designed setups. We present AgenticLab, a model-agnostic robot agent platform and benchmark for open-world manipulation. AgenticLab provides a closed-loop agent pipeline for perception, task decomposition, online verification, and replanning. Using AgenticLab, we benchmark state-of-the-art VLM-based agents on real-robot tasks in unstructured environments. Our benchmark reveals several failure modes that offline vision-language tests (e.g., VQA and static image understanding) fail to capture, including breakdowns in multi-step grounding consistency, object grounding under occlusion and scene changes, and insufficient spatial reasoning for reliable manipulation. We will release the full hardware and software stack to support reproducible evaluation and accelerate research on general-purpose robot agents.
>
---
#### [replaced 024] Contact-Grounded Policy: Dexterous Visuotactile Policy with Generative Contact Grounding
- **分类: cs.RO**

- **简介: 该论文提出一种触觉引导的机械臂控制策略，解决多指灵巧操作中接触状态建模难题，通过预测机器人状态与触觉反馈的耦合轨迹实现精准操控。**

- **链接: [https://arxiv.org/pdf/2603.05687](https://arxiv.org/pdf/2603.05687)**

> **作者:** Zhengtong Xu; Yeping Wang; Ben Abbatematteo; Jom Preechayasomboon; Sonny Chan; Nick Colonnese; Amirhossein H. Memar
>
> **摘要:** Contact-rich dexterous manipulation with multi-finger hands remains an open challenge in robotics because task success depends on multi-point contacts that continuously evolve and are highly sensitive to object geometry, frictional transitions, and slip. Recently, tactile-informed manipulation policies have shown promise. However, most use tactile signals as additional observations rather than modeling contact state or how their action outputs interact with low-level controller dynamics. We present Contact-Grounded Policy (CGP), a visuotactile policy that grounds multi-point contacts by predicting coupled trajectories of actual robot state and tactile feedback, and using a learned contact-consistency mapping to convert these predictions into executable target robot states for a compliance controller. CGP consists of two components: (i) a conditional diffusion model that forecasts future robot state and tactile feedback in a compressed latent space, and (ii) a learned contact-consistency mapping that converts the predicted robot state-tactile pair into executable targets for a compliance controller, enabling it to realize the intended contacts. We evaluate CGP using a physical four-finger Allegro V5 hand with Digit360 fingertip tactile sensors, and a simulated five-finger Tesollo DG-5F hand with dense whole-hand tactile arrays. Across a range of dexterous tasks including in-hand manipulation, delicate grasping, and tool use, CGP outperforms visuomotor and visuotactile diffusion-policy baselines.
>
---
#### [replaced 025] Scalable Aerial GNSS Localization for Marine Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于水下机器人定位任务，旨在解决传统GNSS在水面反射和高成本问题。通过使用搭载GNSS的无人机辅助定位，实现高效、可扩展的海洋机器人定位。**

- **链接: [https://arxiv.org/pdf/2505.04095](https://arxiv.org/pdf/2505.04095)**

> **作者:** Shuo Wen; Edwin Meriaux; Mariana Sosa Guzmán; Charlotte Morissette; Chloe Si; Bobak Baghi; Gregory Dudek
>
> **备注:** International Conference on Robotics and Automation 2025 Workshop Robots in the Wild
>
> **摘要:** Accurate localization is crucial for water robotics, yet traditional onboard Global Navigation Satellite System (GNSS) approaches are difficult or ineffective due to signal reflection on the water's surface and its high cost of aquatic GNSS receivers. Existing approaches, such as inertial navigation, Doppler Velocity Loggers (DVL), SLAM, and acoustic-based methods, face challenges like error accumulation and high computational complexity. Therefore, a more efficient and scalable solution remains necessary. This paper proposes an alternative approach that leverages an aerial drone equipped with GNSS localization to track and localize a marine robot once it is near the surface of the water. Our results show that this novel adaptation enables accurate single and multi-robot marine robot localization.
>
---
#### [replaced 026] RoboLayout: Differentiable 3D Scene Generation for Embodied Agents
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出RoboLayout，解决3D场景生成中满足实体代理交互需求的问题。通过引入可达性约束和局部优化，提升场景可操作性与生成效率。**

- **链接: [https://arxiv.org/pdf/2603.05522](https://arxiv.org/pdf/2603.05522)**

> **作者:** Ali Shamsaddinlou
>
> **摘要:** Recent advances in vision language models (VLMs) have shown strong potential for spatial reasoning and 3D scene layout generation from open-ended language instructions. However, generating layouts that are not only semantically coherent but also feasible for interaction by embodied agents remains challenging, particularly in physically constrained indoor environments. In this paper, RoboLayout is introduced as an extension of LayoutVLM that augments the original framework with agent-aware reasoning and improved optimization stability. RoboLayout integrates explicit reachability constraints into a differentiable layout optimization process, enabling the generation of layouts that are navigable and actionable by embodied agents. Importantly, the agent abstraction is not limited to a specific robot platform and can represent diverse entities with distinct physical capabilities, such as service robots, warehouse robots, humans of different age groups, or animals, allowing environment design to be tailored to the intended agent. In addition, a local refinement stage is proposed that selectively reoptimizes problematic object placements while keeping the remainder of the scene fixed, improving convergence efficiency without increasing global optimization iterations. Overall, RoboLayout preserves the strong semantic alignment and physical plausibility of LayoutVLM while enhancing applicability to agent-centric indoor scene generation, as demonstrated by experimental results across diverse scene configurations.
>
---
#### [replaced 027] VL-Nav: A Neuro-Symbolic Approach for Reasoning-based Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决机器人在复杂指令下自主导航的问题。提出VL-Nav系统，结合神经符号方法提升任务分解与探索效率。**

- **链接: [https://arxiv.org/pdf/2502.00931](https://arxiv.org/pdf/2502.00931)**

> **作者:** Yi Du; Taimeng Fu; Zhipeng Zhao; Shaoshu Su; Zitong Zhan; Zhuoqun Chen; Bowen Li; Chen Wang
>
> **摘要:** Navigating unseen, large-scale environments based on complex and abstract human instructions remains a formidable challenge for autonomous mobile robots. Addressing this requires robots to infer implicit semantics and efficiently explore large-scale task spaces. However, existing methods, ranging from end-to-end learning to foundation model-based modular architectures, often lack the capability to decompose complex tasks or employ efficient exploration strategies, leading to robot aimless wandering or target recognition failures. To address these limitations, we propose VL-Nav, a neuro-symbolic (NeSy) vision-language navigation system. The proposed system intertwines neural reasoning with symbolic guidance through two core components: (1) a NeSy task planner that leverages a symbolic 3D scene graph and image memory system to enhance the vision language models' (VLMs) neural reasoning capabilities for task decomposition and replanning; and (2) a NeSy exploration system that couples neural semantic cues with the symbolic heuristic function to efficiently gather the task-related information while minimizing unnecessary repeat travel during exploration. Validated on the DARPA TIAMAT Challenge navigation tasks, our system achieved an 83.4% success rate (SR) in indoor environments and 75% in outdoor scenarios. VL-Nav achieved an 86.3% SR in real-world experiments, including a challenging 483-meter run. Finally, we validate the system with complex instructions in a 3D multi-floor scenario.
>
---
#### [replaced 028] Strengthening Generative Robot Policies through Predictive World Modeling
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出生成式预测控制（GPC），用于强化机器人策略。解决机器人控制中泛化能力不足的问题，通过生成式策略克隆、预测世界模型训练和在线规划优化，提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2502.00622](https://arxiv.org/pdf/2502.00622)**

> **作者:** Han Qi; Haocheng Yin; Aris Zhu; Yilun Du; Heng Yang
>
> **备注:** Acceptance to RAL. Website: this https URL
>
> **摘要:** We present generative predictive control (GPC), a learning control framework that (i) clones a generative diffusion-based policy from expert demonstrations, (ii) trains a predictive action-conditioned world model from both expert demonstrations and random explorations, and (iii) synthesizes an online planner that ranks and optimizes the action proposals from (i) by looking ahead into the future using the world model from (ii). Across a variety of robotic manipulation tasks, we demonstrate that GPC consistently outperforms behavior cloning in both state-based and vision-based settings, in simulation and in the real world.
>
---
#### [replaced 029] Pretraining in Actor-Critic Reinforcement Learning for Robot Locomotion
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人运动强化学习任务，旨在解决技能学习从零开始效率低的问题。通过预训练神经网络模型提升样本效率和任务性能。**

- **链接: [https://arxiv.org/pdf/2510.12363](https://arxiv.org/pdf/2510.12363)**

> **作者:** Jiale Fan; Andrei Cramariuc; Tifanny Portela; Marco Hutter
>
> **摘要:** The pretraining-finetuning paradigm has facilitated numerous transformative advancements in artificial intelligence research in recent years. However, in the domain of reinforcement learning (RL) for robot locomotion, individual skills are often learned from scratch despite the high likelihood that some generalizable knowledge is shared across all task-specific policies belonging to the same robot embodiment. This work aims to define a paradigm for pretraining neural network models that encapsulate such knowledge and can subsequently serve as a basis for warm-starting the RL process in classic actor-critic algorithms, such as Proximal Policy Optimization (PPO). We begin with a task-agnostic exploration-based data collection algorithm to gather diverse, dynamic transition data, which is then used to train a Proprioceptive Inverse Dynamics Model (PIDM) through supervised learning. The pretrained weights are then loaded into both the actor and critic networks to warm-start the policy optimization of actual tasks. We systematically validated our proposed method with 9 distinct robot locomotion RL environments comprising 3 different robot embodiments, showing significant benefits of this initialization strategy. Our proposed approach on average improves sample efficiency by 36.9% and task performance by 7.3% compared to random initialization. We further present key ablation studies and empirical analyses that shed light on the mechanisms behind the effectiveness of this method.
>
---
#### [replaced 030] ELHPlan: Efficient Long-Horizon Task Planning for Multi-Agent Collaboration
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于多智能体协作任务，解决LLM在长周期规划中的适应性与效率矛盾。提出ELHPlan框架，通过动作链实现高效可靠规划。**

- **链接: [https://arxiv.org/pdf/2509.24230](https://arxiv.org/pdf/2509.24230)**

> **作者:** Shaobin Ling; Yun Wang; Chenyou Fan; Tin Lun Lam; Junjie Hu
>
> **摘要:** Large Language Models (LLMs) enable intelligent multi-robot collaboration but face fundamental trade-offs: open-loop methods that compile tasks into formal representations for external executors produce sound plans but lack adaptability in partially observable environments, while iterative methods incur prohibitive computational costs that scale poorly with team size and task complexity. In this paper, we propose Efficient Long-Horizon Planning (ELHPlan), a novel framework that introduces Action Chains, sequences of actions explicitly bound to sub-goal intentions, as the fundamental planning primitive. ELHPlan operates via a cyclical process: 1) constructing intention-bound action sequences, 2) proactively validating for conflicts and feasibility, 3) refining issues through targeted mechanisms, and 4) executing validated actions. This design balances adaptability and efficiency by providing intention-bound action sequences with longer lookahead while avoiding expensive full re-planning. We further advocate comprehensive efficiency metrics, including token consumption and planning time, to more holistically evaluate multi-agent collaboration. Our experiments on benchmarks TDW-MAT and C-WAH demonstrate that ELHPlan achieves comparable task success rates while consuming only 30-40% of the tokens required by state-of-the-art methods. Our research establishes a new efficiency-effectiveness frontier for LLM-based multi-agent planning systems.
>
---
#### [replaced 031] M3CAD: Towards Generic Cooperative Autonomous Driving Benchmark
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出M³CAD基准，用于通用协同自动驾驶研究，解决多车辆协同感知与通信效率问题，设计多模态数据集并提出多级融合方法。**

- **链接: [https://arxiv.org/pdf/2505.06746](https://arxiv.org/pdf/2505.06746)**

> **作者:** Morui Zhu; Yongqi Zhu; Yihao Zhu; Qi Chen; Deyuan Qu; Song Fu; Qing Yang
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** We introduce M$^3$CAD, a comprehensive benchmark designed to advance research in generic cooperative autonomous driving. M$^3$CAD comprises 204 sequences with 30,000 frames. Each sequence includes data from multiple vehicles and different types of sensors, e.g., LiDAR point clouds, RGB images, and GPS/IMU, supporting a variety of autonomous driving tasks, including object detection and tracking, mapping, motion forecasting, occupancy prediction, and path planning. This rich multimodal setup enables M$^3$CAD to support both single-vehicle and multi-vehicle cooperative autonomous driving research. To the best of our knowledge, M$^3$CAD is the most complete benchmark specifically designed for cooperative, multi-task autonomous driving research. To test its effectiveness, we use M$^3$CAD to evaluate both state-of-the-art single-vehicle and cooperative driving solutions, setting baseline performance results. Since most existing cooperative perception methods focus on merging features but often ignore network bandwidth requirements, we propose a new multi-level fusion approach which adaptively balances communication efficiency and perception accuracy based on the current network conditions. We release M$^3$CAD, along with the baseline models and evaluation results, to support the development of robust cooperative autonomous driving systems. All resources will be made publicly available on this https URL
>
---
#### [replaced 032] PEPA: a Persistently Autonomous Embodied Agent with Personalities
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PEPA，一种具有个性的自主机器人系统，解决长期动态环境中依赖外部指令的问题。通过三层架构实现自主目标生成与行为演化。**

- **链接: [https://arxiv.org/pdf/2603.00117](https://arxiv.org/pdf/2603.00117)**

> **作者:** Kaige Liu; Yang Li; Lijun Zhu; Weinan Zhang
>
> **摘要:** Living organisms exhibit persistent autonomy through internally generated goals and self-sustaining behavioral organization, yet current embodied agents remain driven by externally scripted objectives. This dependence on predefined task specifications limits their capacity for long-term deployment in dynamic, unstructured environments where continuous human intervention is impractical. We propose that personality traits provide an intrinsic organizational principle for achieving persistent autonomy. Analogous to genotypic biases shaping biological behavioral tendencies, personalities enable agents to autonomously generate goals and sustain behavioral evolution without external supervision. To realize this, we develop PEPA, a three-layer cognitive architecture that operates through three interacting systems: Sys3 autonomously synthesizes personality-aligned goals and refines them via episodic memory and daily self-reflection; Sys2 performs deliberative reasoning to translate goals into executable action plans; Sys1 grounds the agent in sensorimotor interaction, executing actions and recording experiences. We validate the framework through real-world deployment on a quadruped robot in a multi-floor office building. Operating without reliance on fixed task specifications, the robot autonomously arbitrates between user requests and personality-driven motivations, navigating elevators and exploring environments accordingly. Quantitative analysis across five distinct personality prototypes demonstrates stable, trait-aligned behaviors. The results confirm that personality-driven cognitive architectures enable sustained autonomous operation characteristic of persistent embodied systems. Code and demo videos are available at this https URL.
>
---
#### [replaced 033] xTED: Cross-Domain Adaptation via Diffusion-Based Trajectory Editing
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于跨域强化学习任务，旨在解决目标域数据不足时的策略迁移问题。通过设计扩散模型进行轨迹编辑，直接对齐源域与目标域数据，提升数据真实性和动态可靠性。**

- **链接: [https://arxiv.org/pdf/2409.08687](https://arxiv.org/pdf/2409.08687)**

> **作者:** Haoyi Niu; Qimao Chen; Tenglong Liu; Jianxiong Li; Guyue Zhou; Yi Zhang; Jianming Hu; Xianyuan Zhan
>
> **备注:** xTED offers a novel, generic, flexible, simple and effective paradigm that casts cross-domain policy adaptation as a data pre-processing problem
>
> **摘要:** Reusing pre-collected data from different domains is an appealing solution for decision-making tasks, especially when data in the target domain are limited. Existing cross-domain policy transfer methods mostly aim at learning domain correspondences or corrections to facilitate policy learning, such as learning task/domain-specific discriminators, representations, or policies. This design philosophy often results in heavy model architectures or task/domain-specific modeling, lacking flexibility. This reality makes us wonder: can we directly bridge the domain gaps universally at the data level, instead of relying on complex downstream cross-domain policy transfer procedures? In this study, we propose the Cross-Domain Trajectory EDiting (xTED) framework that employs a specially designed diffusion model for cross-domain trajectory adaptation. Our proposed model architecture effectively captures the intricate dependencies among states, actions, and rewards, as well as the dynamics patterns within target data. Edited by adding noises and denoising with the pre-trained diffusion model, source domain trajectories can be transformed to align with target domain properties while preserving original semantic information. This process effectively corrects underlying domain gaps, enhancing state realism and dynamics reliability in source data, and allowing flexible integration with various single-domain and cross-domain downstream policy learning methods. Despite its simplicity, xTED demonstrates superior performance in extensive simulation and real-robot experiments.
>
---
#### [replaced 034] A Robust Placeability Metric for Model-Free Unified Pick-and-Place Reasoning
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取与放置任务，解决在不完整观测下如何可靠地进行抓放规划的问题。提出一种概率性放置度量，结合稳定性、可抓性和间隙评估，提升抓放成功率。**

- **链接: [https://arxiv.org/pdf/2510.14584](https://arxiv.org/pdf/2510.14584)**

> **作者:** Benno Wingender; Nils Dengler; Rohit Menon; Sicong Pan; Maren Bennewitz
>
> **摘要:** Reliable manipulation of previously unseen objects remains a fundamental challenge for autonomous robotic systems operating in unstructured environments. In particular, robust pick-and-place planning directly from noisy and only partial real-world observations, where object surfaces are inherently incomplete due to occlusions (e.g., bottom faces on a tabletop), is difficult. As a result, many existing methods rely on strong object priors (e.g., CAD models) or to assume placement on continuous, flat support surfaces such as planar tabletops, without explicitly accounting for edge proximity or inclined supports. In this work, we introduce a robust probabilistic placeability metric that evaluates 6D object placement poses from partial observations by jointly scoring object stability, graspability, and clearance from raw point cloud geometry. Using this metric, we generate diverse multi-orientation placement candidates and condition grasp scoring on these placements, enabling model-free unified pick-and-place reasoning. Simulation and real-robot experiments on unseen objects and challenging support geometries confirm that our metric yields accurate stability predictions and consistently improves end-to-end pick-and-place success by producing stable, collision-free grasp-place pairs directly from partial point clouds.
>
---
#### [replaced 035] FoldNet: Learning Generalizable Closed-Loop Policy for Garment Folding via Keypoint-Driven Asset and Demonstration Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对服装折叠任务，解决数据生成困难问题，提出合成数据集和关键点驱动的策略，提升机器人折叠成功率。**

- **链接: [https://arxiv.org/pdf/2505.09109](https://arxiv.org/pdf/2505.09109)**

> **作者:** Yuxing Chen; Bowen Xiao; He Wang
>
> **备注:** Project: this https URL
>
> **摘要:** Due to the deformability of garments, generating a large amount of high-quality data for robotic garment manipulation tasks is highly challenging. In this paper, we present a synthetic garment dataset that can be used for robotic garment folding. We begin by constructing geometric garment templates based on keypoints and applying generative models to generate realistic texture patterns. Leveraging these keypoint annotations, we generate folding demonstrations in simulation and train folding policies via closed-loop imitation learning. To improve robustness, we propose KG-DAgger, which uses a keypoint-based strategy to generate demonstration data for recovering from failures. KG-DAgger significantly improves the model performance, boosting the real-world success rate by 25\%. After training with 15K trajectories (about 2M image-action pairs), the model achieves a 75\% success rate in the real world. Experiments in both simulation and real-world settings validate the effectiveness of our proposed framework.
>
---
#### [replaced 036] Diffusion Stabilizer Policy for Automated Surgical Robot Manipulations
- **分类: cs.RO**

- **简介: 该论文属于手术机器人自动化任务，旨在解决手术操作中轨迹不完美问题。提出基于扩散模型的策略学习框架DSP，提升机器人在扰动数据下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2503.01252](https://arxiv.org/pdf/2503.01252)**

> **作者:** Chonlam Ho; Jianshu Hu; Lei Song; Hesheng Wang; Qi Dou; Yutong Ban
>
> **备注:** ICRA 2026
>
> **摘要:** Intelligent surgical robots have the potential to revolutionize clinical practice by enabling more precise and automated surgical procedures. However, the automation of such robot for surgical tasks remains under-explored compared to recent advancements in solving household manipulation tasks. These successes have been largely driven by (1) advanced models, such as transformers and diffusion models, and (2) large-scale data utilization. Aiming to extend these successes to the domain of surgical robotics, we propose a diffusion-based policy learning framework, called Diffusion Stabilizer Policy (DSP), which enables training with imperfect or even failed trajectories. Our approach consists of two stages: first, we train the diffusion stabilizer policy using only clean data. Then, the policy is continuously updated using a mixture of clean and perturbed data, with filtering based on the prediction error on actions. Comprehensive experiments conducted in various surgical environments demonstrate the superior performance of our method in perturbation-free settings and its robustness when handling perturbed demonstrations.
>
---
#### [replaced 037] PAD-TRO: Projection-Augmented Diffusion for Direct Trajectory Optimization
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于轨迹优化任务，解决动态可行性问题。提出PAD-TRO方法，通过模型扩散直接生成状态序列，并引入无梯度投影机制，提升无人机导航的成功率和准确性。**

- **链接: [https://arxiv.org/pdf/2510.04436](https://arxiv.org/pdf/2510.04436)**

> **作者:** Jushan Chen; Santiago Paternain
>
> **备注:** Final manuscript. Accepted for publication at the 2026 American Control Conference
>
> **摘要:** Recently, diffusion models have gained popularity and attention in trajectory optimization due to their capability of modeling multi-modal probability distributions. However, addressing nonlinear equality constraints, i.e, dynamic feasibility, remains a great challenge in diffusion-based trajectory optimization. Recent diffusion-based trajectory optimization frameworks rely on a single-shooting style approach where the denoised control sequence is applied to forward propagate the dynamical system, which cannot explicitly enforce constraints on the states and frequently leads to sub-optimal solutions. In this work, we propose a novel direct trajectory optimization approach via model-based diffusion, which directly generates a sequence of states. To ensure dynamic feasibility, we propose a gradient-free projection mechanism that is incorporated into the reverse diffusion process. Our results show that, compared to a recent state-of-the-art baseline, our approach leads to zero dynamic feasibility error and approximately 4x higher success rate in a quadrotor waypoint navigation scenario involving dense static obstacles.
>
---
#### [replaced 038] BEV-Patch-PF: Particle Filtering with BEV-Aerial Feature Matching for Off-Road Geo-Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于定位任务，解决无GPS的越野地理定位问题。通过融合BEV特征与航拍图匹配，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.15111](https://arxiv.org/pdf/2512.15111)**

> **作者:** Dongmyeong Lee; Jesse Quattrociocchi; Christian Ellis; Rwik Rana; Amanda Adkins; Adam Uccello; Garrett Warnell; Joydeep Biswas
>
> **摘要:** We propose BEV-Patch-PF, a GPS-free sequential geo-localization system that integrates a particle filter with learned bird's-eye-view (BEV) and aerial feature maps. From onboard RGB and depth images, we construct a BEV feature map. For each 3-DoF particle pose hypothesis, we crop the corresponding patch from an aerial feature map computed from a local aerial image queried around the approximate location. BEV-Patch-PF computes a per-particle log-likelihood by matching the BEV feature to the aerial patch feature. On two real-world off-road datasets, our method achieves 9.7x lower absolute trajectory error (ATE) on seen routes and 6.6x lower ATE on unseen routes than a retrieval-based baseline, while maintaining accuracy under dense canopy and shadow. The system runs in real time at 10 Hz on an NVIDIA Tesla T4, enabling practical robot deployment.
>
---
#### [replaced 039] Vision-Guided Targeted Grasping and Vibration for Robotic Pollination in Controlled Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人精准授粉任务，解决无风环境下的自动授粉问题。通过视觉引导和振动建模，实现茎秆抓取与花粉释放，提升农业自动化水平。**

- **链接: [https://arxiv.org/pdf/2510.06146](https://arxiv.org/pdf/2510.06146)**

> **作者:** Jaehwan Jeong; Tuan-Anh Vu; Radha Lahoti; Jiawen Wang; Vivek Alumootil; Sangpil Kim; M. Khalid Jawed
>
> **备注:** YouTube: this https URL GitHub: this https URL
>
> **摘要:** Robotic pollination offers a promising alternative to manual labor and bumblebee-assisted methods in controlled agriculture, where wind-driven pollination is absent and regulatory restrictions limit the use of commercial pollinators. In this work, we present and validate a vision-guided robotic framework that uses data from an end-effector mounted RGB-D sensor and combines 3D plant reconstruction, targeted grasp planning, and physics-based vibration modeling to enable precise pollination. First, the plant is reconstructed in 3D and registered to the robot coordinate frame to identify obstacle-free grasp poses along the main stem. Second, a discrete elastic rod model predicts the relationship between actuation parameters and flower dynamics, guiding the selection of optimal pollination strategies. Finally, a manipulator with soft grippers grasps the stem and applies controlled vibrations to induce pollen release. End-to-end experiments demonstrate a 92.5\% main-stem grasping success rate, and simulation-guided optimization of vibration parameters further validates the feasibility of our approach, ensuring that the robot can safely and effectively perform pollination without damaging the flower. To our knowledge, this is the first robotic system to jointly integrate vision-based grasping and vibration modeling for automated precision pollination.
>
---
#### [replaced 040] OIPP: Object-Adaptive Impact Point Predictor for Catching Diverse In-Flight Objects
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在准确预测飞行物体的落点。针对数据不足和轨迹相似性问题，构建了8000条轨迹的数据集，并提出OIPP模型提升预测精度。**

- **链接: [https://arxiv.org/pdf/2509.15254](https://arxiv.org/pdf/2509.15254)**

> **作者:** Ngoc Huy Nguyen; Kazuki Shibata; Takamitsu Matsubara
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** In this study, we address the problem of in-flight object catching using a quadruped robot with a basket. Our objective is to accurately predict the impact point, defined as the object's landing position. This task poses two key challenges: the absence of public datasets capturing diverse objects under unsteady aerodynamics, which are essential for training reliable predictors; and the difficulty of accurate early-stage impact point prediction when trajectories appear similar across objects. To overcome these issues, we construct a real-world dataset of 8,000 trajectories from 20 objects, providing a foundation for advancing in-flight object catching under complex aerodynamics. We then propose the Object-Adaptive Impact Point Predictor (OIPP), consisting of two modules: (i) an Object-Adaptive Encoder (OAE) that extracts object-dependent representations from motion histories, and (ii) an Impact Point Predictor (IPP) that estimates the impact point from these representations. Two IPP variants are implemented: a Neural Acceleration Estimator (NAE)-based method that predicts trajectories and derives the impact point, and a Direct Point Estimator (DPE)-based method that directly outputs it. Experimental results show that our dataset is more diverse and complex than existing datasets, and that our method outperforms baselines on both 15 seen and 5 unseen objects. Furthermore, we show that improved early-stage prediction enhances catching success in simulation and demonstrate the effectiveness of our approach through real-robot experiments. The demonstration is available at this https URL.
>
---
#### [replaced 041] Ego-Vision World Model for Humanoid Contact Planning
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于人形机器人接触规划任务，旨在提升其在非结构化环境中的自主性。解决传统方法在接触复杂性和样本效率上的不足，提出结合学习世界模型与MPC的框架，实现高效、多任务的接触规划。**

- **链接: [https://arxiv.org/pdf/2510.11682](https://arxiv.org/pdf/2510.11682)**

> **作者:** Hang Liu; Yuman Gao; Sangli Teng; Yufeng Chi; Yakun Sophia Shao; Zhongyu Li; Maani Ghaffari; Koushil Sreenath
>
> **摘要:** Enabling humanoid robots to exploit physical contact, rather than simply avoid collisions, is crucial for autonomy in unstructured environments. Traditional optimization-based planners struggle with contact complexity, while on-policy reinforcement learning (RL) is sample-inefficient and has limited multi-task ability. We propose a framework combining a learned world model with sampling-based Model Predictive Control (MPC), trained on a demonstration-free offline dataset to predict future outcomes in a compressed latent space. To address sparse contact rewards and sensor noise, the MPC uses a learned surrogate value function for dense, robust planning. Our single, scalable model supports contact-aware tasks, including wall support after perturbation, blocking incoming objects, and traversing height-limited arches, with improved sample efficiency and multi-task capability over on-policy RL. Deployed on a physical humanoid, our system achieves robust, real-time contact planning from proprioception and ego-centric depth images. Code and dataset are available at our website: this https URL
>
---
#### [replaced 042] ClearDepth: Enhanced Stereo Perception of Transparent Objects for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人感知任务，旨在解决透明物体深度感知难题。通过视觉Transformer和特征后融合模块提升深度恢复精度，并利用Sim2Real仿真生成数据，提高实际应用效果。**

- **链接: [https://arxiv.org/pdf/2409.08926](https://arxiv.org/pdf/2409.08926)**

> **作者:** Kaixin Bai; Huajian Zeng; Lei Zhang; Yiwen Liu; Hongli Xu; Zhaopeng Chen; Jianwei Zhang
>
> **备注:** 9 pages
>
> **摘要:** Transparent object depth perception poses a challenge in everyday life and logistics, primarily due to the inability of standard 3D sensors to accurately capture depth on transparent or reflective surfaces. This limitation significantly affects depth map and point cloud-reliant applications, especially in robotic manipulation. We developed a vision transformer-based algorithm for stereo depth recovery of transparent objects. This approach is complemented by an innovative feature post-fusion module, which enhances the accuracy of depth recovery by structural features in images. To address the high costs associated with dataset collection for stereo camera-based perception of transparent objects, our method incorporates a parameter-aligned, domain-adaptive, and physically realistic Sim2Real simulation for efficient data generation, accelerated by AI algorithm. Our experimental results demonstrate the model's exceptional Sim2Real generalizability in real-world scenarios, enabling precise depth mapping of transparent objects to assist in robotic manipulation. Project details are available at this https URL .
>
---
#### [replaced 043] LagMemo: Language 3D Gaussian Splatting Memory for Multi-modal Open-vocabulary Multi-goal Visual Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出LagMemo，用于多模态、开放式词汇的多目标视觉导航任务。通过构建3D语言高斯点云记忆，实现高效目标定位与验证。**

- **链接: [https://arxiv.org/pdf/2510.24118](https://arxiv.org/pdf/2510.24118)**

> **作者:** Haotian Zhou; Xiaole Wang; He Li; Zhuo Qi; Jinrun Yin; Haiyu Kong; Jianghuan Xu; Huijing Zhao
>
> **摘要:** Navigating to a designated goal using visual information is a fundamental capability for intelligent robots. To address the practical demands of multi-modal, open-vocabulary goal queries and multi-goal visual navigation, we propose LagMemo, a navigation system that leverages a language 3D Gaussian Splatting memory. During a one-time exploration, LagMemo constructs a unified 3D language memory with robust spatial-semantic correlations. With incoming task goals, the system efficiently queries the memory, predicts candidate goal locations, and integrates a local perception-based verification mechanism to dynamically match and validate goals. For fair and rigorous evaluation, we curate GOAT-Core, a high-quality core split distilled from GOAT-Bench. Experimental results show that LagMemo's memory module enables effective multi-modal open-vocabulary localization, and significantly outperforms state-of-the-art methods in multi-goal visual navigation. Project page: this https URL
>
---
#### [replaced 044] Accelerating Robotic Reinforcement Learning with Agent Guidance
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人强化学习任务，旨在解决样本效率低的问题。通过引入多模态代理替代人类监督，提升训练效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2602.11978](https://arxiv.org/pdf/2602.11978)**

> **作者:** Haojun Chen; Zili Zou; Chengdong Ma; Yaoxiang Pu; Haotong Zhang; Yuanpei Chen; Yaodong Yang
>
> **摘要:** Reinforcement Learning (RL) offers a powerful paradigm for autonomous robots to master generalist manipulation skills through trial-and-error. However, its real-world application is stifled by low sample efficiency. Recent Human-in-the-Loop (HIL) methods accelerate training by using human corrections, yet this approach faces a scalability barrier. Reliance on human supervisors imposes a 1:1 supervision ratio that limits scalability, suffers from operator fatigue over extended sessions, and introduces high variance due to inconsistent human proficiency. We present Agent-guided Policy Search (AGPS), a framework that automates the training pipeline by replacing human supervisors with a multimodal agent. Our key insight is that the agent can be viewed as a semantic world model, injecting intrinsic value priors to structure physical exploration. By using tools, the agent provides precise guidance via corrective waypoints and spatial constraints for exploration pruning. We validate our approach on three tasks, ranging from precision insertion to deformable object manipulation. Results demonstrate that AGPS outperforms HIL methods in sample efficiency. This automates the supervision pipeline, unlocking the path to labor-free and scalable robot learning. Project website: this https URL.
>
---
#### [replaced 045] IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人运动规划任务，解决在杂乱环境中如何安全进行接触式运动的问题。通过引入视觉-语言模型，识别可接触区域，生成安全路径。**

- **链接: [https://arxiv.org/pdf/2503.10110](https://arxiv.org/pdf/2503.10110)**

> **作者:** Yiyang Ling; Karan Owalekar; Oluwatobiloba Adesanya; Erdem Bıyık; Daniel Seita
>
> **摘要:** Motion planning involves determining a sequence of robot configurations to reach a desired pose, subject to movement and safety constraints. Traditional motion planning finds collision-free paths, but this is overly restrictive in clutter, where it may not be possible for a robot to accomplish a task without contact. In addition, contacts range from relatively benign (e.g. brushing a soft pillow) to more dangerous (e.g. toppling a glass vase), making it difficult to characterize which may be acceptable. In this paper, we propose IMPACT, a novel motion planning framework that uses Vision-Language Models (VLMs) to infer environment semantics, identifying which parts of the environment can best tolerate contact based on object properties and locations. Our approach generates an anisotropic cost map that encodes directional push safety. We pair this map with a contact-aware A* planner to find stable contact-rich paths. We perform experiments using 20 simulation and 10 real-world scenes and assess using task success rate, object displacements, and feedback from human evaluators. Our results over 3200 simulation and 200 real-world trials suggest that IMPACT enables efficient contact-rich motion planning in cluttered settings while outperforming alternative methods and ablations. Our project website is available at this https URL.
>
---
#### [replaced 046] Hybrid Diffusion Policies with Projective Geometric Algebra for Efficient Robot Manipulation Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人操作学习任务，旨在提升扩散策略的训练效率。通过引入投影几何代数，构建混合扩散策略hPGA-DP，有效提升空间推理能力与收敛速度。**

- **链接: [https://arxiv.org/pdf/2507.05695](https://arxiv.org/pdf/2507.05695)**

> **作者:** Xiatao Sun; Yuxuan Wang; Shuo Yang; Yinxing Chen; Daniel Rakita
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Diffusion policies are a powerful paradigm for robot learning, but their training is often inefficient. A key reason is that networks must relearn fundamental spatial concepts, such as translations and rotations, from scratch for every new task. To alleviate this redundancy, we propose embedding geometric inductive biases directly into the network architecture using Projective Geometric Algebra (PGA). PGA provides a unified algebraic framework for representing geometric primitives and transformations, allowing neural networks to reason about spatial structure more effectively. In this paper, we introduce hPGA-DP, a novel hybrid diffusion policy that capitalizes on these benefits. Our architecture leverages the Projective Geometric Algebra Transformer (P-GATr) as a state encoder and action decoder, while employing established U-Net or Transformer-based modules for the core denoising process. Through extensive experiments and ablation studies in both simulated and real-world environments, we demonstrate that hPGA-DP significantly improves task performance and training efficiency. Notably, our hybrid approach achieves substantially faster convergence compared to both standard diffusion policies and architectures that rely solely on P-GATr. The project website is available at: this https URL.
>
---
#### [replaced 047] ActivePusher: Active Learning and Planning with Residual Physics for Nonprehensile Manipulation
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
#### [replaced 048] ReViP: Mitigating False Completion in Vision-Language-Action Models with Vision-Proprioception Rebalance
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决VLA模型中的错误完成问题。通过引入ReViP框架和False-Completion基准，提升模型在扰动下的视觉感知与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.16667](https://arxiv.org/pdf/2601.16667)**

> **作者:** Zhuohao Li; Yinghao Li; Jian-Jian Jiang; Lang Zhou; Tianyu Zhang; Jiadong Yin; Mu Lin; Yi-Kin Wei; Wei-Shi Zheng
>
> **摘要:** Vision-Language-Action (VLA) models have advanced robotic manipulation by combining vision, language, and proprioception to predict actions. However, previous methods fuse proprioceptive signals directly with vision-language features, resulting in state-dominant bias and \textbf{false completions} despite visible execution failures. We systematically analyze this failure mode, attributing it to modality imbalance, where policies overly rely on internal state progression and underuse visual evidence. To address this, we introduce the first \textbf{False-Completion Benchmark Suite}, featuring eight tasks with three controlled perturbations (\emph{Object Drop}, \emph{Distractor Swap}, \emph{Relayout}) to comprehensively evaluate false completion. Moreover, we propose \textbf{ReViP}, a novel VLA framework with \textbf{Vi}sion-\textbf{P}roprioception \textbf{Re}balance to enhance visual grounding and robustness under perturbations. The key insight is to introduce auxiliary \emph{progress-aware visual cues} to adaptively modulate the coupling between semantic perception and proprioceptive dynamics. Specifically, progress-aware visual cues are extracted by an external Task-Stage Observer, which performs task-relevant reasoning on real-time observations to drive task-stage feature-wise linear modulation, enhancing environmental awareness and mitigating state-driven errors. Extensive experiments show that ReViP effectively mitigates false completion and improves success rates over strong VLA baselines, achieving a \textbf{26\%} gain over $\pi_0$ model on our suite, with gains extending to LIBERO, RoboTwin 2.0, and real-world evaluations.
>
---
#### [replaced 049] Bio-inspired tail oscillation enables robot fast crawling on deformable granular terrains
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决机器人在松散颗粒地形上移动困难的问题。通过仿生尾部振荡设计，提升机器人在沙土等可变形地形上的爬行速度与效率。**

- **链接: [https://arxiv.org/pdf/2509.12468](https://arxiv.org/pdf/2509.12468)**

> **作者:** Shipeng Liu; Meghana Sagare; Shubham Patil; Feifei Qian
>
> **摘要:** Deformable substrates such as sand and mud present significant challenges for terrestrial robots due to complex robot-terrain interactions. Inspired by mudskippers, amphibious animals that naturally adjust their tail morphology and movement jointly to navigate such environments, we investigate how tail design and control can jointly enhance flipper-driven locomotion on granular media. Using a bio-inspired robot modeled after the mudskipper, we experimentally compared locomotion performance between idle and actively oscillating tail configurations. Tail oscillation increased robot speed by 67% and reduced body drag by 46%. Shear force measurements revealed that this improvement was enabled by tail oscillation fluidizing the substrate, thereby reducing resistance. Additionally, tail morphology strongly influenced the oscillation strategy: designs with larger horizontal surface areas leveraged the oscillation-reduced shear resistance more effectively by limiting insertion depth. Based on these findings, we present a design principle to inform tail action selection based on substrate strength and tail morphology. Our results offer new insights into tail design and control for improving robot locomotion on deformable substrates, with implications for agricultural robotics, search and rescue, and environmental exploration.
>
---
#### [replaced 050] Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于隐式表面建模任务，解决从单张图像高效构建隐式距离表示的问题。提出FINS框架，利用单图快速生成高保真表面和SDF场。**

- **链接: [https://arxiv.org/pdf/2509.20681](https://arxiv.org/pdf/2509.20681)**

> **作者:** Wei-Teng Chu; Tianyi Zhang; Matthew Johnson-Roberson; Weiming Zhi
>
> **摘要:** Implicit representations have been widely applied in robotics for obstacle avoidance and path planning. In this paper, we explore the problem of constructing an implicit distance representation from a single image. Past methods for implicit surface reconstruction, such as NeuS and its variants generally require a large set of multi-view images as input, and require long training times. In this work, we propose Fast Image-to-Neural Surface (FINS), a lightweight framework that can reconstruct high-fidelity surfaces and SDF fields based on a single or a small set of images. FINS integrates a multi-resolution hash grid encoder with lightweight geometry and color heads, making the training via an approximate second-order optimizer highly efficient and capable of converging within a few seconds. Additionally, we achieve the construction of a neural surface requiring only a single RGB image, by leveraging pre-trained foundation models to estimate the geometry inherent in the image. Our experiments demonstrate that under the same conditions, our method outperforms state-of-the-art baselines in both convergence speed and accuracy on surface reconstruction and SDF field estimation. Moreover, we demonstrate the applicability of FINS for robot surface following tasks and show its scalability to a variety of benchmark datasets.
>
---
#### [replaced 051] $π$-StepNFT: Wider Space Needs Finer Steps in Online RL for Flow-based VLAs
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出$\pi$-StepNFT，解决流式视觉-语言-动作模型在在线强化学习中的多步采样问题，通过细粒度步骤引导提升性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.02083](https://arxiv.org/pdf/2603.02083)**

> **作者:** Siting Wang; Xiaofeng Wang; Zheng Zhu; Minnan Pei; Xinyu Cui; Cheng Deng; Jian Zhao; Guan Huang; Haifeng Zhang; Jun Wang
>
> **摘要:** Flow-based vision-language-action (VLA) models excel in embodied control but suffer from intractable likelihoods during multi-step sampling, hindering online reinforcement learning. We propose \textbf{\textit{$\boldsymbol{\pi}$-StepNFT}} (Step-wise Negative-aware Fine-Tuning), a critic-and-likelihood-free framework that requires only a single forward pass per optimization step and eliminates auxiliary value networks. We identify that wider exploration spaces necessitate finer-grained, step-wise guidance for alignment. Empirically, $\pi$-StepNFT unlocks latent potential on LIBERO with competitive few-shot robustness. Moreover, it achieves superior generalization on ManiSkill, outperforming value-based baselines in OOD scenarios by preventing overfitting to multimodal features. This property offers a scalable solution promising for complex real-world applications.
>
---
#### [replaced 052] Beyond Collision Cones: Dynamic Obstacle Avoidance for Nonholonomic Robots via Dynamic Parabolic Control Barrier Functions
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人路径规划任务，解决非完整机器人在动态障碍物环境中的安全避障问题。提出动态抛物线控制屏障函数，提升导航成功率和可行性。**

- **链接: [https://arxiv.org/pdf/2510.01402](https://arxiv.org/pdf/2510.01402)**

> **作者:** Hun Kuk Park; Taekyung Kim; Dimitra Panagou
>
> **备注:** The first two authors contributed equally to this work. 2026 IEEE International Conference on Robotics and Automation (ICRA). Project page: this https URL
>
> **摘要:** Control Barrier Functions (CBFs) are a powerful tool for ensuring the safety of autonomous systems, yet applying them to nonholonomic robots in cluttered, dynamic environments remains an open challenge. State-of-the-art methods often rely on collision-cone or velocity-obstacle constraints which, by only considering the angle of the relative velocity, are inherently conservative and can render the CBF-based quadratic program infeasible, particularly in dense scenarios. To address this issue, we propose a Dynamic Parabolic Control Barrier Function (DPCBF) that defines the safe set using a parabolic boundary. The parabola's vertex and curvature dynamically adapt based on both the distance to an obstacle and the magnitude of the relative velocity, creating a less restrictive safety constraint. We prove that the proposed DPCBF is valid for a kinematic bicycle model subject to input constraints. Extensive comparative simulations demonstrate that our DPCBF-based controller significantly enhances navigation success rates and QP feasibility compared to baseline methods. Our approach successfully navigates through dense environments with up to 100 dynamic obstacles, scenarios where collision cone-based methods fail due to infeasibility.
>
---
#### [replaced 053] MobiDock: Design and Control of A Modular Self Reconfigurable Bimanual Mobile Manipulator via Robotic Docking
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决多机器人协作的稳定性与协调问题。通过设计MobiDock系统，实现机器人物理连接与重构，提升整体性能。**

- **链接: [https://arxiv.org/pdf/2510.27178](https://arxiv.org/pdf/2510.27178)**

> **作者:** Xuan-Thuan Nguyen; Khac Nam Nguyen; Ngoc Duy Tran; Thi Thoa Mac; Anh Nguyen; Hoang Hiep Ly; Tung D. Ta
>
> **备注:** IROS2026 submited
>
> **摘要:** Multi-robot systems, particularly mobile manipulators, face challenges in control coordination and dynamic stability when working together. To address this issue, this study proposes MobiDock, a modular self-reconfigurable mobile manipulator system that allows two independent robots to physically connect and form a unified mobile bimanual platform. This process helps transform a complex multi-robot control problem into the management of a simpler, single system. The system utilizes an autonomous docking strategy based on computer vision with AprilTag markers and a new threaded screw-lock mechanism. Experimental results show that the docked configuration demonstrates better performance in dynamic stability and operational efficiency compared to two independently cooperating robots. Specifically, the unified system has lower Root Mean Square (RMS) Acceleration and Jerk values, higher angular precision, and completes tasks significantly faster. These findings confirm that physical reconfiguration is a powerful design principle that simplifies cooperative control, improving stability and performance for complex tasks in real-world environments.
>
---
#### [replaced 054] Iterative Closed-Loop Motion Synthesis for Scaling the Capabilities of Humanoid Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，旨在解决人体模型控制数据不足与难度限制问题。通过闭环生成高质量运动数据并迭代优化，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2602.21599](https://arxiv.org/pdf/2602.21599)**

> **作者:** Weisheng Xu; Qiwei Wu; Jiaxi Zhang; Tan Jing; Yangfan Li; Yuetong Fang; Jiaqi Xiong; Kai Wu; Rong Ou; Renjing Xu
>
> **摘要:** Physics-based humanoid control relies on training with motion datasets that have diverse data distributions. However, the fixed difficulty distribution of datasets limits the performance ceiling of the trained control policies. Additionally, the method of acquiring high-quality data through professional motion capture systems is constrained by costs, making it difficult to achieve large-scale scalability. To address these issues, we propose a closed-loop automated motion data generation and iterative framework. It can generate high-quality motion data with rich action semantics, including martial arts, dance, combat, sports, gymnastics, and more. Furthermore, our framework enables difficulty iteration of policies and data through physical metrics and objective evaluations, allowing the trained tracker to break through its original difficulty limits. On the PHC single-primitive tracker, using only approximately 1/10 of the AMASS dataset size, the average failure rate on the test set (2201 clips) is reduced by 45% compared to the baseline. Finally, we conduct comprehensive ablation and comparative experiments to highlight the rationality and advantages of our framework.
>
---
#### [replaced 055] Task Parameter Extrapolation via Learning Inverse Tasks from Forward Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于机器人学习领域，旨在解决技能策略在新条件下的泛化问题。通过逆任务学习，构建正反任务的共同表示，实现高效知识迁移。**

- **链接: [https://arxiv.org/pdf/2603.05576](https://arxiv.org/pdf/2603.05576)**

> **作者:** Serdar Bahar; Fatih Dogangun; Matteo Saveriano; Yukie Nagai; Emre Ugur
>
> **备注:** Corrected author affiliation
>
> **摘要:** Generalizing skill policies to novel conditions remains a key challenge in robot learning. Imitation learning methods, while data-efficient, are largely confined to the training region and consistently fail on input data outside it, leading to unpredictable policy failures. Alternatively, transfer learning approaches offer methods for trajectory generation robust to both changes in environment or tasks, but they remain data-hungry and lack accuracy in zero-shot generalization. We address these challenges by framing the problem in the context of task inversion learning and proposing a novel joint learning approach to achieve accurate and efficient knowledge transfer. Our method constructs a common representation of the forward and inverse tasks, and leverages auxiliary forward demonstrations from novel configurations to successfully execute the corresponding inverse tasks, without any direct supervision. We show the extrapolation capabilities of our framework via ablation studies and experiments in simulated and real-world environments that require complex manipulation skills with a diverse set of objects and tools, where we outperform diffusion-based alternatives.
>
---
#### [replaced 056] EasyInsert: A Data-Efficient and Generalizable Insertion Policy
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人插入任务，解决复杂环境中物体插入的泛化与效率问题。提出EasyInsert方法，通过视觉策略实现高效、通用的插入，无需CAD模型，仅需少量数据即可达成高成功率。**

- **链接: [https://arxiv.org/pdf/2505.16187](https://arxiv.org/pdf/2505.16187)**

> **作者:** Guanghe Li; Junming Zhao; Shengjie Wang; Yang Gao
>
> **摘要:** Robotic insertion is a highly challenging task that requires exceptional precision in cluttered environments. Existing methods often have poor generalization capabilities. They typically function in restricted and structured environments, and frequently fail when the plug and socket are far apart, when the scene is densely cluttered, or when handling novel objects. They also rely on strong assumptions such as access to CAD models or a digital twin in simulation. To address these limitations, we propose EasyInsert. Inspired by human intuition, it formulates insertion as a delta-pose regression problem, which unlocks an efficient, highly scalable data collection pipeline with minimal human labor to train an end-to-end visual policy. During execution, the visual policy predicts the relative pose between plug and socket to drive a multi-phase, coarse-to-fine insertion process. EasyInsert demonstrates strong zero-shot generalization capability for unseen objects in cluttered environments, robustly handling cases with significant initial pose deviations. In real-world experiments, by leveraging just 1 hour of human teleoperation data to bootstrap a large-scale automated data collection process, EasyInsert achieves an over 90% success rate in zero-shot insertion for 13 out of 15 unseen novel objects, including challenging objects like Type-C cables, HDMI cables, and Ethernet cables. Furthermore, requiring only a single manual reset, EasyInsert allows for fast adaptation to novel test objects through automated data collection and fine-tuning, achieving an over 90% success rate across all 15 objects.
>
---
#### [replaced 057] SToRM: Supervised Token Reduction for Multi-modal LLMs toward efficient end-to-end autonomous driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决多模态大语言模型计算成本高的问题。提出SToRM框架，在减少视觉token的同时保持性能，提升效率。**

- **链接: [https://arxiv.org/pdf/2602.11656](https://arxiv.org/pdf/2602.11656)**

> **作者:** Seo Hyun Kim; Jin Bok Park; Do Yeon Koo; Hogun Park; Il Yong Chun
>
> **摘要:** In autonomous driving, end-to-end (E2E) driving systems that predict control commands directly from sensor data have achieved significant advancements. For safe driving in unexpected scenarios, these systems may additionally rely on human interventions such as natural language instructions. Using a multi-modal large language model (MLLM) facilitates human-vehicle interaction and can improve performance in such scenarios. However, this approach requires substantial computational resources due to its reliance on an LLM and numerous visual tokens from sensor inputs, which are limited in autonomous vehicles. Many MLLM studies have explored reducing visual tokens, but often suffer end-task performance degradation compared to using all tokens. To enable efficient E2E driving while maintaining performance comparable to using all tokens, this paper proposes the first Supervised Token Reduction framework for multi-modal LLMs (SToRM). The proposed framework consists of three key elements. First, a lightweight importance predictor with short-term sliding windows estimates token importance scores. Second, a supervised training approach uses an auxiliary path to obtain pseudo-supervision signals from an all-token LLM pass. Third, an anchor-context merging module partitions tokens into anchors and context tokens, and merges context tokens into relevant anchors to reduce redundancy while minimizing information loss. Experiments on the LangAuto benchmark show that SToRM outperforms state-of-the-art E2E driving MLLMs under the same reduced-token budget, maintaining all-token performance while reducing computational cost by up to 30x.
>
---
#### [replaced 058] Smart placement, faster robots-a comparison of algorithms for robot base-pose optimization
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决工业机器人基座姿态优化问题。通过比较BO、ES、GAs和SGD算法，提升机器人部署效率与成功率。**

- **链接: [https://arxiv.org/pdf/2504.19577](https://arxiv.org/pdf/2504.19577)**

> **作者:** Matthias Mayer; Matthias Althoff
>
> **备注:** 10 pages, 3 Figures, 1 Table. Find visualizations and source code at this https URL. Supplementary Tables can be found at this https URL
>
> **摘要:** Robotic automation is a key technology that increases the efficiency and flexibility of manufacturing processes. However, one of the challenges in deploying robots in novel environments is finding the optimal base pose for the robot, which affects its reachability and deployment cost. Yet, existing research on automatically optimizing the base pose of robots has not been compared. We address this problem by optimizing the base pose of industrial robots with Bayesian optimization (BO), exhaustive search (ES), genetic algorithms (GAs), and stochastic gradient descent (SGD), and we find that all algorithms can reduce the cycle time for various evaluated tasks in synthetic and real-world environments. Stochastic gradient descent shows superior performance with regard to the success rate, solving more than 90% of our real-world tasks, while genetic algorithms show the lowest final costs. All benchmarks and implemented methods are available as baselines against which novel approaches can be compared.
>
---
#### [replaced 059] Holistic Optimization of Modular Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人优化任务，解决模块化机器人在特定任务中组合、定位和轨迹的协同优化问题。通过联合优化，减少循环时间并提高解决方案可行性。**

- **链接: [https://arxiv.org/pdf/2505.00400](https://arxiv.org/pdf/2505.00400)**

> **作者:** Matthias Mayer; Matthias Althoff
>
> **备注:** 14 Pages, 6 figures, 8 tables. Please find and reference the open-access published version at this https URL
>
> **摘要:** Modular robots have the potential to revolutionize automation, as one can optimize their composition for any given task. However, finding optimal compositions is non-trivial. In addition, different compositions require different base positions and trajectories to fully use the potential of modular robots. We address this problem holistically for the first time by jointly optimizing the composition, base placement, and trajectory to minimize the cycle time of a given task. Our approach is evaluated on over 300 industrial benchmarks requiring point-to-point movements. Overall, we reduce cycle time by up to 25 % and find feasible solutions in twice as many benchmarks compared to optimizing the module composition alone. In the first real-world validation of modular robots optimized for point-to-point movement, we find that the optimized robot is successfully deployed in nine out of ten cases in less than an hour.
>
---
#### [replaced 060] HumanHalo - Safe and Efficient 3D Navigation Among Humans via Minimally Conservative MPC
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决3D环境下安全高效避障问题。提出HumanMPC框架，结合模型预测控制与人类运动预测，确保安全且不过度保守。**

- **链接: [https://arxiv.org/pdf/2510.17525](https://arxiv.org/pdf/2510.17525)**

> **作者:** Simon Schaefer; Helen Oleynikova; Sandra Hirche; Stefan Leutenegger
>
> **摘要:** Safe and efficient robotic navigation among humans is essential for integrating robots into everyday environments. Most existing approaches focus on simplified 2D crowd navigation and fail to account for the full complexity of human body dynamics beyond root motion. We present HumanMPC, a Model Predictive Control (MPC) framework for 3D Micro Air Vehicle (MAV) navigation among humans that combines theoretical safety guarantees with data-driven models for realistic human motion forecasting. Our approach introduces a novel twist to reachability-based safety formulation that constrains only the initial control input for safety while modeling its effects over the entire planning horizon, enabling safe yet efficient navigation. We validate HumanMPC in both simulated experiments using real human trajectories and in the real-world, demonstrating its effectiveness across tasks ranging from goal-directed navigation to visual servoing for human tracking. While we apply our method to MAVs in this work, it is generic and can be adapted by other platforms. Our results show that the method ensures safety without excessive conservatism and outperforms baseline approaches in both efficiency and reliability.
>
---
#### [replaced 061] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出EgoDex数据集，解决仿生操作数据稀缺问题。通过收集大量第一视角视频及手部姿态数据，支持模仿学习，推动机器人、视觉和模型发展。**

- **链接: [https://arxiv.org/pdf/2505.11709](https://arxiv.org/pdf/2505.11709)**

> **作者:** Ryan Hoque; Peide Huang; David J. Yoon; Mouli Sivapurapu; Jian Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** Imitation learning for manipulation has a well-known data scarcity problem. Unlike natural language and 2D computer vision, there is no Internet-scale corpus of data for dexterous manipulation. One appealing option is egocentric human video, a passively scalable data source. However, existing large-scale datasets such as Ego4D do not have native hand pose annotations and do not focus on object manipulation. To this end, we use Apple Vision Pro to collect EgoDex: the largest and most diverse dataset of dexterous human manipulation to date. EgoDex has 829 hours of egocentric video with paired 3D hand and finger tracking data collected at the time of recording, where multiple calibrated cameras and on-device SLAM can be used to precisely track the pose of every joint of each hand. The dataset covers a wide range of diverse manipulation behaviors with everyday household objects in 194 different tabletop tasks ranging from tying shoelaces to folding laundry. Furthermore, we train and systematically evaluate imitation learning policies for hand trajectory prediction on the dataset, introducing metrics and benchmarks for measuring progress in this increasingly important area. By releasing this large-scale dataset, we hope to push the frontier of robotics, computer vision, and foundation models. EgoDex is publicly available for download at this https URL.
>
---
#### [replaced 062] MEM: Multi-Scale Embodied Memory for Vision Language Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人学习任务，旨在解决长时序任务中多粒度记忆问题。提出MEM架构，结合视频与文本记忆，提升机器人执行复杂任务的能力。**

- **链接: [https://arxiv.org/pdf/2603.03596](https://arxiv.org/pdf/2603.03596)**

> **作者:** Marcel Torne; Karl Pertsch; Homer Walke; Kyle Vedder; Suraj Nair; Brian Ichter; Allen Z. Ren; Haohuan Wang; Jiaming Tang; Kyle Stachowicz; Karan Dhabalia; Michael Equi; Quan Vuong; Jost Tobias Springenberg; Sergey Levine; Chelsea Finn; Danny Driess
>
> **备注:** Website: this https URL
>
> **摘要:** Conventionally, memory in end-to-end robotic learning involves inputting a sequence of past observations into the learned policy. However, in complex multi-stage real-world tasks, the robot's memory must represent past events at multiple levels of granularity: from long-term memory that captures abstracted semantic concepts (e.g., a robot cooking dinner should remember which stages of the recipe are already done) to short-term memory that captures recent events and compensates for occlusions (e.g., a robot remembering the object it wants to pick up once its arm occludes it). In this work, our main insight is that an effective memory architecture for long-horizon robotic control should combine multiple modalities to capture these different levels of abstraction. We introduce Multi-Scale Embodied Memory (MEM), an approach for mixed-modal long-horizon memory in robot policies. MEM combines video-based short-horizon memory, compressed via a video encoder, with text-based long-horizon memory. Together, they enable robot policies to perform tasks that span up to fifteen minutes, like cleaning up a kitchen, or preparing a grilled cheese sandwich. Additionally, we find that memory enables MEM policies to intelligently adapt manipulation strategies in-context.
>
---
#### [replaced 063] RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboPARA框架，解决双臂机器人任务并行规划问题，通过两阶段方法提升任务协同效率。**

- **链接: [https://arxiv.org/pdf/2506.06683](https://arxiv.org/pdf/2506.06683)**

> **作者:** Shiying Duan; Pei Ren; Nanxiang Jiang; Zhengping Che; Jian Tang; Zhaoxin Fan; Yifan Sun; Wenjun Wu
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking scenarios. While existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm collaboration. To address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism planning. RoboPARA employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task coherence. In addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty levels. Extensive experiments demonstrate that RoboPARA significantly outperforms existing planning methods, achieving higher efficiency and reliability, particularly in complex task combinations. Our code is publicly available at this https URL.
>
---
#### [replaced 064] M4Diffuser: Multi-View Diffusion Policy with Manipulability-Aware Control for Robust Mobile Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于移动操作任务，解决复杂环境中操作效率与鲁棒性问题。提出M4Diffuser框架，结合多视角扩散策略与新型QP控制器，提升任务成功率和安全性。**

- **链接: [https://arxiv.org/pdf/2509.14980](https://arxiv.org/pdf/2509.14980)**

> **作者:** Ju Dong; Lei Zhang; Liding Zhang; Yao Ling; Yu Fu; Kaixin Bai; Zoltán-Csaba Márton; Zhenshan Bing; Zhaopeng Chen; Alois Christian Knoll; Jianwei Zhang
>
> **备注:** Project page: this https URL, 10 pages, 9 figures
>
> **摘要:** Mobile manipulation requires the coordinated control of a mobile base and a robotic arm while simultaneously perceiving both global scene context and fine-grained object details. Existing single-view approaches often fail in unstructured environments due to limited fields of view, exploration, and generalization abilities. Moreover, classical controllers, although stable, struggle with efficiency and manipulability near singularities. To address these challenges, we propose M4Diffuser, a hybrid framework that integrates a Multi-View Diffusion Policy with a novel Reduced and Manipulability-aware QP (ReM-QP) controller for mobile manipulation. The diffusion policy leverages proprioceptive states and complementary camera perspectives with both close-range object details and global scene context to generate task-relevant end-effector goals in the world frame. These high-level goals are then executed by the ReM-QP controller, which eliminates slack variables for computational efficiency and incorporates manipulability-aware preferences for robustness near singularities. Comprehensive experiments in simulation and real-world environments show that M4Diffuser achieves 7 to 56 percent higher success rates and reduces collisions by 3 to 31 percent over baselines. Our approach demonstrates robust performance for smooth whole-body coordination, and strong generalization to unseen tasks, paving the way for reliable mobile manipulation in unstructured environments. Details of the demo and supplemental material are available on our project website this https URL.
>
---
#### [replaced 065] Compose by Focus: Scene Graph-based Atomic Skills
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人技能学习任务，旨在解决长周期任务中技能组合的鲁棒性问题。通过引入基于场景图的表示和框架，提升技能执行与组合的稳定性。**

- **链接: [https://arxiv.org/pdf/2509.16053](https://arxiv.org/pdf/2509.16053)**

> **作者:** Han Qi; Changhe Chen; Heng Yang
>
> **备注:** Acceptance to ICRA 2026. Website: this https URL
>
> **摘要:** A key requirement for generalist robots is compositional generalization - the ability to combine atomic skills to solve complex, long-horizon tasks. While prior work has primarily focused on synthesizing a planner that sequences pre-learned skills, robust execution of the individual skills themselves remains challenging, as visuomotor policies often fail under distribution shifts induced by scene composition. To address this, we introduce a scene graph-based representation that focuses on task-relevant objects and relations, thereby mitigating sensitivity to irrelevant variation. Building on this idea, we develop a scene-graph skill learning framework that integrates graph neural networks with diffusion-based imitation learning, and further combine "focused" scene-graph skills with a vision-language model (VLM) based task planner. Experiments in both simulation and real-world manipulation tasks demonstrate substantially higher success rates than state-of-the-art baselines, highlighting improved robustness and compositional generalization in long-horizon tasks.
>
---
#### [replaced 066] CDE: Concept-Driven Exploration for Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于视觉强化学习任务，解决视觉控制中探索效率低的问题。通过引入概念驱动的探索方法，利用预训练视觉语言模型生成概念，提升探索效果。**

- **链接: [https://arxiv.org/pdf/2510.08851](https://arxiv.org/pdf/2510.08851)**

> **作者:** Le Mao; Andrew H. Liu; Renos Zabounidis; Yanan Niu; Zachary Kingston; Joseph Campbell
>
> **备注:** Preprint
>
> **摘要:** Intelligent exploration remains a critical challenge in reinforcement learning (RL), especially in visual control tasks. Unlike low-dimensional state-based RL, visual RL must extract task-relevant structure from raw pixels, making exploration inefficient. We propose Concept-Driven Exploration (CDE), which leverages a pre-trained vision-language model (VLM) to generate object-centric visual concepts from textual task descriptions as weak, potentially noisy supervisory signals. Rather than directly conditioning on these noisy signals, CDE trains a policy to reconstruct the concepts via an auxiliary objective, learning general representations of the concepts and using reconstruction accuracy as an intrinsic reward to guide exploration toward task-relevant objects. Across five challenging simulated visual manipulation tasks, CDE achieves efficient, targeted exploration and remains robust to both synthetic errors and noisy VLM predictions. Finally, we demonstrate real-world transfer by deploying CDE on a Franka arm, attaining an 80\% success rate in a real-world manipulation task.
>
---
#### [replaced 067] Graph Neural Model Predictive Control for High-Dimensional Systems
- **分类: cs.RO**

- **简介: 该论文属于控制任务，解决高维系统实时控制问题。通过结合图神经网络与模型预测控制，提升计算效率与控制精度。**

- **链接: [https://arxiv.org/pdf/2602.17601](https://arxiv.org/pdf/2602.17601)**

> **作者:** Patrick Benito Eberhard; Luis Pabon; Daniele Gammelli; Hugo Buurmeijer; Amon Lahr; Mark Leone; Andrea Carron; Marco Pavone
>
> **摘要:** The control of high-dimensional systems, such as soft robots, requires models that faithfully capture complex dynamics while remaining computationally tractable. This work presents a framework that integrates Graph Neural Network (GNN)-based dynamics models with structure-exploiting Model Predictive Control to enable real-time control of high-dimensional systems. By representing the system as a graph with localized interactions, the GNN preserves sparsity, while a tailored condensing algorithm eliminates state variables from the control problem, ensuring efficient computation. The complexity of our condensing algorithm scales linearly with the number of system nodes, and leverages Graphics Processing Unit (GPU) parallelization to achieve real-time performance. The proposed approach is validated in simulation and experimentally on a physical soft robotic trunk. Results show that our method scales to systems with up to 1,000 nodes at 100 Hz in closed-loop, and demonstrates real-time reference tracking on hardware with sub-centimeter accuracy, outperforming baselines by 63.6%. Finally, we show the capability of our method to achieve effective full-body obstacle avoidance.
>
---
#### [replaced 068] ORN-CBF: Learning Observation-conditioned Residual Neural Control Barrier Functions via Hypernetworks
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于安全控制领域，旨在解决自主系统在部分可观测环境中的安全问题。提出基于观测的神经CBF方法，提升安全集的准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.16614](https://arxiv.org/pdf/2509.16614)**

> **作者:** Bojan Derajić; Sebastian Bernhard; Wolfgang Hönig
>
> **摘要:** Control barrier functions (CBFs) have been demonstrated as an effective method for safety-critical control of autonomous systems. Although CBFs are simple to deploy, their design remains challenging, motivating the development of learning-based approaches. Yet, issues such as suboptimal safe sets, applicability in partially observable environments, and lack of rigorous safety guarantees persist. In this work, we propose observation-conditioned neural CBFs based on Hamilton-Jacobi (HJ) reachability analysis, which approximately recover the maximal safe sets. We exploit certain mathematical properties of the HJ value function, ensuring that the predicted safe set never intersects with the observed failure set. Moreover, we leverage a hypernetwork-based architecture that is particularly suitable for the design of observation-conditioned safety filters. The proposed method is examined both in simulation and hardware experiments for a ground robot and a quadcopter. The results show improved success rates and generalization to out-of-domain environments compared to the baselines.
>
---
#### [replaced 069] SAC-Loco: Safe and Adjustable Compliant Quadrupedal Locomotion
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于四足机器人运动控制任务，旨在解决外部力扰动下的安全与可调柔顺性问题。工作包括设计柔性控制框架、安全恢复策略及实时安全评估机制。**

- **链接: [https://arxiv.org/pdf/2509.23223](https://arxiv.org/pdf/2509.23223)**

> **作者:** Aoqian Zhang; Zixuan Zhuang; Chunzheng Wang; Shuzhi Sam Ge; Fan Shi; Cheng Xiang
>
> **摘要:** Quadruped robots are designed to achieve agile and robust locomotion by drawing inspiration from legged animals. However, most existing control methods for quadruped robots lack a key capacity observed in animals: the ability to exhibit diverse compliance behaviors while ensuring stability when experiencing external forces. In particular, achieving adjustable compliance while maintaining robust safety under force disturbances remains a significant challenge. In this work, we propose a safety aware compliant locomotion framework that integrates adjustable disturbance compliance with robust failure prevention. We first train a force compliant policy with adjustable compliance levels using a teacher student reinforcement learning framework, allowing deployment without explicit force sensing. To handle disturbances beyond the limits of compliant control, we develop a safety oriented policy for rapid recovery and stabilization. Finally, we introduce a learned safety critic that monitors the robot's safety in real time and coordinates between compliant locomotion and recovery behaviors. Together, this framework enables quadruped robots to achieve smooth force compliance and robust safety under a wide range of external force disturbances.
>
---
#### [replaced 070] MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MCGS-SLAM，用于高保真建图的多相机SLAM系统，解决单目系统鲁棒性差、覆盖范围小的问题，通过RGB输入和高斯点云优化实现精准定位与重建。**

- **链接: [https://arxiv.org/pdf/2509.14191](https://arxiv.org/pdf/2509.14191)**

> **作者:** Zhihao Cao; Hanyu Wu; Li Wa Tang; Zizhou Luo; Wei Zhang; Marc Pollefeys; Zihan Zhu; Martin R. Oswald
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving.
>
---
#### [replaced 071] Influence-Based Reward Modulation for Implicit Communication in Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在提升隐式沟通能力。通过调节影响因子，增强协作与独立性，无需预设意图。**

- **链接: [https://arxiv.org/pdf/2406.12253](https://arxiv.org/pdf/2406.12253)**

> **作者:** Haoyang Jiang; Elizabeth A. Croft; Michael G. Burke
>
> **备注:** Preprint. 26 pages, 15 figures. Submitted to IEEE Transactions on Human-Robot Interaction (THRI). Accepted manuscript version
>
> **摘要:** Communication is essential for successful interaction. In human-robot interaction, implicit communication holds the potential to enhance robots' understanding of human needs, emotions, and intentions. This paper introduces a method to foster implicit communication in HRI without explicitly modelling human intentions or relying on pre-existing knowledge. Leveraging Transfer Entropy, we modulate influence between agents in social interactions in scenarios involving either collaboration or competition. By integrating influence into agents' rewards within a partially observable Markov decision process, we demonstrate that boosting influence enhances collaboration and interaction, while resisting influence promotes social independence and diminishes performance in certain scenarios. Our findings are validated through simulations and real-world experiments with human participants in social navigation and autonomous driving settings.
>
---
#### [replaced 072] Automated Pest Counting in Water Traps through Active Robotic Stirring for Occlusion Handling
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动化 pest 计数任务，解决 occlusion 导致的计数不准问题。通过主动机械搅拌和自适应控制，提升计数准确性和效率。**

- **链接: [https://arxiv.org/pdf/2510.21732](https://arxiv.org/pdf/2510.21732)**

> **作者:** Xumin Gao; Mark Stevens; Grzegorz Cielniak
>
> **摘要:** Existing image-based pest counting methods rely on single static images and often produce inaccurate results under occlusion. To address this issue, this paper proposes an automated pest counting method in water traps through active robotic stirring. First, an automated robotic arm-based stirring system is developed to redistribute pests and reveal occluded individuals for counting. Then, the effects of different stirring patterns on pest counting performance are investigated. Six stirring patterns are designed and evaluated across different pest density scenarios to identify the optimal one. Finally, a heuristic counting confidence-driven closed-loop control system is proposed for adaptive-speed robotic stirring, adjusting the stirring speed based on the average change rate of counting confidence between consecutive frames. Experimental results show that the four circles is the optimal stirring pattern, achieving the lowest overall mean absolute counting error of 4.384 and the highest overall mean counting confidence of 0.721. Compared with constant-speed stirring, adaptive-speed stirring reduces task execution time by up to 44.7% and achieves more stable performance across different pest density scenarios. Moreover, the proposed pest counting method reduces the mean absolute counting error by up to 3.428 compared to the single static image counting method under high-density scenarios where occlusion is severe.
>
---
#### [replaced 073] DemoDiffusion: One-Shot Human Imitation using pre-trained Diffusion Policy
- **分类: cs.RO; cs.LG**

- **简介: 论文提出DemoDiffusion，用于机器人通过单次人类示范完成操作任务。解决无任务特定训练和配对数据下的模仿学习问题，通过运动轨迹转换与扩散策略优化实现高效适应。**

- **链接: [https://arxiv.org/pdf/2506.20668](https://arxiv.org/pdf/2506.20668)**

> **作者:** Sungjae Park; Homanga Bharadhwaj; Shubham Tulsiani
>
> **备注:** 11 pages. Published at ICRA 2026
>
> **摘要:** We propose DemoDiffusion, a simple method for enabling robots to perform manipulation tasks by imitating a single human demonstration, without requiring task-specific training or paired human-robot data. Our approach is based on two insights. First, the hand motion in a human demonstration provides a useful prior for the robot's end-effector trajectory, which we can convert into a rough open-loop robot motion trajectory via kinematic retargeting. Second, while this retargeted motion captures the overall structure of the task, it may not align well with plausible robot actions in-context. To address this, we leverage a pre-trained generalist diffusion policy to modify the trajectory, ensuring it both follows the human motion and remains within the distribution of plausible robot actions. Unlike approaches based on online reinforcement learning or paired human-robot data, our method enables robust adaptation to new tasks and scenes with minimal effort. In real-world experiments across 8 diverse manipulation tasks, DemoDiffusion achieves 83.8\% average success rate, compared to 13.8\% for the pre-trained policy and 52.5\% for kinematic retargeting, succeeding even on tasks where the pre-trained generalist policy fails entirely. Project page: this https URL
>
---
#### [replaced 074] RetoVLA: Reusing Register Tokens for Spatial Reasoning in Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出RetoVLA，解决轻量级视觉-语言-动作模型中空间推理能力不足的问题，通过重用注册令牌提升场景理解，提升机器人任务成功率。**

- **链接: [https://arxiv.org/pdf/2509.21243](https://arxiv.org/pdf/2509.21243)**

> **作者:** Jiyeon Koo; Taewan Cho; Hyunjoon Kang; Eunseom Pyo; Tae Gyun Oh; Taeryang Kim; Andrew Jaeyong Choi
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated robust performance across diverse robotic tasks. However, their high memory and computational demands often limit real-time deployment. While existing model compression techniques reduce the parameter footprint, they often drop in 3D spatial reasoning and scene layout understanding. This work introduces RetoVLA, an architecture designed to maintain spatial awareness in lightweight models by repurposing Register Tokens-learnable parameters originally introduced to mitigate attention artifacts in Vision Transformers. While these tokens are generally discarded once used, we repurpose them for their dense representation of global spatial context. RetoVLA integrates these recycled tokens directly into the action-planning module through a dedicated spatial context injection path. Our proposed design enables the recovery of global context without increasing the total parameter count. Real-world experiments using a 7-DOF manipulator show a 17.1%p improvement in average success rates over the baseline. Our results demonstrate that leveraging internal register tokens provides a highly effective mechanism for developing efficient, spatially-aware robotic agents. A video demonstration is available at: this https URL
>
---
#### [replaced 075] Autonomous UAV-Quadruped Docking in Complex Terrains via Active Posture Alignment and Constraint-Aware Control
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决UAV与四足机器人在复杂地形中的自主对接问题。通过姿态对齐和约束控制方法，实现稳定 docking。**

- **链接: [https://arxiv.org/pdf/2509.21571](https://arxiv.org/pdf/2509.21571)**

> **作者:** Haozhe Xu; Cheng Cheng; Hongrui Sang; Zhipeng Wang; Qiyong He; Xiuxian Li; Bin He
>
> **摘要:** Autonomous docking between Unmanned Aerial Vehicles (UAVs) and ground robots is essential for heterogeneous systems, yet most existing approaches target wheeled platforms whose limited mobility constrains exploration in complex terrains. Quadruped robots offer superior adaptability but undergo frequent posture variations, making it difficult to provide a stable landing surface for UAVs. To address these challenges, we propose an autonomous UAV-quadruped docking framework for GPS-denied environments. On the quadruped side, a Hybrid Internal Model with Horizontal Alignment (HIM-HA), learned via deep reinforcement learning, actively stabilizes the torso to provide a level platform. On the UAV side, a three-phase strategy is adopted, consisting of long-range acquisition with a median-filtered YOLOv8 detector, close-range tracking with a constraint-aware controller that integrates a Nonsingular Fast Terminal Sliding Mode Controller (NFTSMC) and a logarithmic Barrier Function (BF) to guarantee finite-time error convergence under field-of-view (FOV) constraints, and terminal descent guided by a Safety Period (SP) mechanism that jointly verifies tracking accuracy and platform stability. The proposed framework is validated in both simulation and real-world scenarios, successfully achieving docking on outdoor staircases higher than 17 cm and rough slopes steeper than 30 degrees. Supplementary materials and videos are available at: this https URL.
>
---
#### [replaced 076] Radio-based Multi-Robot Odometry and Relative Localization
- **分类: cs.RO**

- **简介: 该论文属于多机器人相对定位任务，旨在解决在复杂环境中精确估计无人机与地面机器人相对位置的问题。通过融合UWB、雷达和惯性传感器数据，提出一种优化框架，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.26558](https://arxiv.org/pdf/2509.26558)**

> **作者:** Andrés Martínez-Silva; David Alejo; Luis Merino; Fernando Caballero
>
> **摘要:** Radio-based methods such as Ultra-Wideband (UWB) and RAdio Detection And Ranging (radar), which have traditionally seen limited adoption in robotics, are experiencing a boost in popularity thanks to their robustness to harsh environmental conditions and cluttered environments. This work proposes a multi-robot UGV-UAV localization system that leverages the two technologies with inexpensive and readily-available sensors, such as Inertial Measurement Units (IMUs) and wheel encoders, to estimate the relative position of an aerial robot with respect to a ground robot. The first stage of the system pipeline includes a nonlinear optimization framework to trilaterate the location of the aerial platform based on UWB range data, and a radar pre-processing module with loosely coupled ego-motion estimation which has been adapted for a multi-robot scenario. Then, the pre-processed radar data as well as the relative transformation are fed to a pose-graph optimization framework with odometry and inter-robot constraints. The system, implemented for the Robotic Operating System (ROS 2) with the Ceres optimizer, has been validated in Software-in-the-Loop (SITL) simulations and in a real-world dataset. The proposed relative localization module outperforms state-of-the-art closed-form methods which are less robust to noise. Our SITL environment includes a custom Gazebo plugin for generating realistic UWB measurements modeled after real data. Conveniently, the proposed factor graph formulation makes the system readily extensible to full Simultaneous Localization And Mapping (SLAM). Finally, all the code and experimental data is publicly available to support reproducibility and to serve as a common open dataset for benchmarking.
>
---
#### [replaced 077] Stable Multi-Drone GNSS Tracking System for Marine Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于海洋机器人跟踪任务，解决水下GNSS信号失效问题。通过多无人机协同，结合视觉与EKF算法，实现稳定实时定位与多机一致性跟踪。**

- **链接: [https://arxiv.org/pdf/2511.18694](https://arxiv.org/pdf/2511.18694)**

> **作者:** Shuo Wen; Edwin Meriaux; Mariana Sosa Guzmán; Zhizun Wang; Junming Shi; Gregory Dudek
>
> **摘要:** Stable and accurate tracking is essential for marine robotics, yet Global Navigation Satellite System (GNSS) signals vanish immediately below the sea surface. Traditional alternatives suffer from error accumulation, high computational demands, or infrastructure dependence. In this work, we present a multi-drone GNSS-based tracking system for surface and near-surface marine robots. Our approach combines efficient visual detection, lightweight multi-object tracking, GNSS-based triangulation, and a confidence-weighted Extended Kalman Filter (EKF) to provide stable GNSS estimation in real time. We further introduce a cross-drone tracking ID alignment algorithm that enforces global consistency across views, enabling robust multi-robot tracking with cooperative aerial coverage. We validate our system in diversified complex settings to show the accuracy and robustness of the proposed algorithm.
>
---
#### [replaced 078] Diffusion-SAFE: Diffusion-Native Human-to-Robot Driving Handover for Shared Autonomy
- **分类: cs.RO**

- **简介: 该论文提出Diffusion-SAFE，用于共享自主驾驶中的安全人机控制交接。任务是实现安全、平滑的控制转移，解决风险预测与控制权切换问题。通过扩散模型实现意图预测与安全引导的计划生成。**

- **链接: [https://arxiv.org/pdf/2505.09889](https://arxiv.org/pdf/2505.09889)**

> **作者:** Yunxin Fan; Monroe Kennedy III
>
> **摘要:** Shared autonomy in driving requires anticipating human behavior, flagging risk before it becomes unavoidable, and transferring control safely and smoothly. We propose Diffusion-SAFE, a closed-loop framework built on two diffusion models: an evaluator that predicts multimodal human-intent action sequences for probabilistic risk detection, and a safety-guided copilot that steers its denoising process toward safe regions using the gradient of a map-based safety certificate. When risk is detected, control is transferred through partial diffusion: the human plan is forward-noised to an intermediate level and denoised by the safety-guided copilot. The forward-diffusion ratio $\rho$ acts as a continuous takeover knob-small $\rho$ keeps the output close to human intent, while increasing $\rho$ shifts authority toward the copilot, avoiding the mixed-unsafe pitfall of action-level blending. Unlike methods relying on hand-crafted score functions, our diffusion formulation supports both safety evaluation and plan generation directly from demonstrations. We evaluate Diffusion-SAFE in simulation and on a real ROS-based race car, achieving 93.0%/87.0% (sim/real) handover success rates with smooth transitions.
>
---
#### [replaced 079] IPPO Learns the Game, Not the Team: A Study on Generalization in Heterogeneous Agent Teams
- **分类: cs.RO**

- **简介: 该论文属于多智能体强化学习任务，研究在异构团队中策略的泛化能力。解决的问题是自对弈PPO代理是否能学习到通用协作策略而非仅适应训练伙伴。工作包括引入RPT方法并验证IPPO在新队友上的泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.08877](https://arxiv.org/pdf/2512.08877)**

> **作者:** Ryan LeRoy; Jack Kolb
>
> **备注:** 4 pages, 3 figures, appendix
>
> **摘要:** Multi-Agent Reinforcement Learning (MARL) is commonly deployed in settings where agents are trained via self-play with homogeneous teammates, often using parameter sharing and a single policy architecture. This opens the question: to what extent do self-play PPO agents learn general coordination strategies grounded in the underlying game, compared to overfitting to their training partners' behaviors? This paper investigates the question using the Heterogeneous Multi-Agent Challenge (HeMAC) environment, which features distinct Observer and Drone agents with complementary capabilities. We introduce Rotating Policy Training (RPT), an approach that rotates heterogeneous teammate policies of different learning algorithms during training, to expose the agent to a broader range of partner strategies. When playing alongside a withheld teammate policy (DDQN), we find that RPT achieves similar performance to a standard self-play baseline, IPPO, where all agents were trained sharing a single PPO policy. This result indicates that in this heterogeneous multi-agent setting, the IPPO baseline generalizes to novel teammate algorithms despite not experiencing teammate diversity during training. This shows that a simple IPPO baseline may possess the level of generalization to novel teammates that a diverse training regimen was designed to achieve.
>
---
#### [replaced 080] Input-to-State Stable Coupled Oscillator Networks for Closed-form Model-based Control in Latent Space
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于控制理论任务，旨在解决物理系统在潜在空间中的高效控制问题。通过提出耦合振子网络模型，解决现有方法在物理结构、稳定性及输入映射上的不足，实现稳定且有效的潜在空间控制。**

- **链接: [https://arxiv.org/pdf/2409.08439](https://arxiv.org/pdf/2409.08439)**

> **作者:** Maximilian Stölzle; Cosimo Della Santina
>
> **备注:** 38th Conference on Neural Information Processing Systems (NeurIPS 2024) spotlight, 50 pages
>
> **摘要:** Even though a variety of methods have been proposed in the literature, efficient and effective latent-space control (i.e., control in a learned low-dimensional space) of physical systems remains an open challenge. We argue that a promising avenue is to leverage powerful and well-understood closed-form strategies from control theory literature in combination with learned dynamics, such as potential-energy shaping. We identify three fundamental shortcomings in existing latent-space models that have so far prevented this powerful combination: (i) they lack the mathematical structure of a physical system, (ii) they do not inherently conserve the stability properties of the real systems, (iii) these methods do not have an invertible mapping between input and latent-space forcing. This work proposes a novel Coupled Oscillator Network (CON) model that simultaneously tackles all these issues. More specifically, (i) we show analytically that CON is a Lagrangian system - i.e., it possesses well-defined potential and kinetic energy terms. Then, (ii) we provide formal proof of global Input-to-State stability using Lyapunov arguments. Moving to the experimental side, we demonstrate that CON reaches SoA performance when learning complex nonlinear dynamics of mechanical systems directly from images. An additional methodological innovation contributing to achieving this third goal is an approximated closed-form solution for efficient integration of network dynamics, which eases efficient training. We tackle (iii) by approximating the forcing-to-input mapping with a decoder that is trained to reconstruct the input based on the encoded latent space force. Finally, we show how these properties enable latent-space control. We use an integral-saturated PID with potential force compensation and demonstrate high-quality performance on a soft robot using raw pixels as the only feedback information.
>
---
#### [replaced 081] Synchronized Online Friction Estimation and Adaptive Grasp Control for Robust Gentle Grasp
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决柔性抓取中的摩擦估计与力控制问题。通过同步实时摩擦估计与自适应控制，提升抓取的稳定性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.02026](https://arxiv.org/pdf/2602.02026)**

> **作者:** Zhenwei Niu; Xiaoyi Chen; Jiayu Hu; Zhaoyang Liu; Tang Jian; Xiaozu Ju
>
> **摘要:** We introduce a unified framework for gentle robotic grasping that synergistically couples real-time friction estimation with adaptive grasp control. We propose a new particle filter-based method for real-time estimation of the friction coefficient using vision-based tactile sensors. This estimate is seamlessly integrated into a reactive controller that dynamically modulates grasp force to maintain a stable grip. The two processes operate synchronously in a closed-loop: the controller uses the current best estimate to adjust the force, while new tactile feedback from this action continuously refines the estimation. This creates a highly responsive and robust sensorimotor cycle. The reliability and efficiency of the complete framework are validated through extensive robotic experiments.
>
---
#### [replaced 082] CroSTAta: Cross-State Transition Attention Transformer for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决执行过程中遇到未训练变化时策略鲁棒性不足的问题。提出Cross-State Transition Attention Transformer，通过状态转移注意力机制提升策略对历史执行的适应能力。**

- **链接: [https://arxiv.org/pdf/2510.00726](https://arxiv.org/pdf/2510.00726)**

> **作者:** Giovanni Minelli; Giulio Turrisi; Victor Barasuol; Claudio Semini
>
> **备注:** Code and data available at this https URL
>
> **摘要:** Learning robotic manipulation policies through supervised learning from demonstrations remains challenging when policies encounter execution variations not explicitly covered during training. While incorporating historical context through attention mechanisms can improve robustness, standard approaches process all past states in a sequence without explicitly modeling the temporal structure that demonstrations may include, such as failure and recovery patterns. We propose a Cross-State Transition Attention Transformer that employs a novel State Transition Attention (STA) mechanism to modulate standard attention weights based on learned state evolution patterns, enabling policies to better adapt their behavior based on execution history. Our approach combines this structured attention with temporal masking during training, where visual information is randomly removed from recent timesteps to encourage temporal reasoning from historical context. Evaluation in simulation shows that STA consistently outperforms standard attention approach and temporal modeling methods like TCN and LSTM networks, achieving more than 2x improvement over cross-attention on precision-critical tasks. The source code and data can be accessed at this https URL
>
---
#### [replaced 083] Green-VLA: Staged Vision-Language-Action Model for Generalist Robots
- **分类: cs.RO**

- **简介: 该论文提出Green-VLA框架，解决机器人通用控制问题。通过多阶段训练和强化学习，提升机器人在不同形态下的任务执行能力与安全性。**

- **链接: [https://arxiv.org/pdf/2602.00919](https://arxiv.org/pdf/2602.00919)**

> **作者:** I. Apanasevich; M. Artemyev; R. Babakyan; P. Fedotova; D. Grankin; E. Kupryashin; A. Misailidi; D. Nerus; A. Nutalapati; G. Sidorov; I. Efremov; M. Gerasyov; D. Pikurov; Y. Senchenko; S. Davidenko; D. Kulikov; M. Sultankin; K. Askarbek; O. Shamanin; D. Statovoy; E. Zalyaev; I. Zorin; A. Letkin; E. Rusakov; A. Silchenko; V. Vorobyov; S. Sobolnikov; A. Postnikov
>
> **备注:** 22 pages, 14 figures
>
> **摘要:** We introduce Green-VLA, a staged Vision-Language-Action (VLA) framework for real-world deployment on the Green humanoid robot while maintaining generalization across diverse embodiments. Green-VLA follows a five stage curriculum: (L0) foundational VLMs, (L1) multimodal grounding, (R0) multi-embodiment pretraining, (R1) embodiment-specific adaptation, and (R2) reinforcement-learning (RL) policy alignment. We couple a scalable data-processing pipeline (3,000 hours of demonstrations) with temporal alignment and quality filtering, and use a unified, embodiment-aware action interface enabling a single policy to control humanoids, mobile manipulators, and fixed-base arms. At inference, the VLA controller is enhanced with episode-progress prediction, out-of-distribution detection, and joint-prediction-based guidance to improve safety and precise target selection. Experiments on Simpler BRIDGE WidowX and CALVIN ABC-D, as well as real-robot evaluations, show strong generalization and performance gains from RL alignment in success rate, robustness, and long-horizon efficiency.
>
---
#### [replaced 084] DropVLA: An Action-Level Backdoor Attack on Vision-Language-Action Models
- **分类: cs.CR; cs.AI; cs.RO**

- **简介: 该论文属于安全领域，研究VLA模型的后门攻击问题。提出DropVLA攻击方法，在少量数据污染下实现对特定动作的控制，保持任务性能不变。**

- **链接: [https://arxiv.org/pdf/2510.10932](https://arxiv.org/pdf/2510.10932)**

> **作者:** Zonghuan Xu; Jiayu Li; Yunhan Zhao; Xiang Zheng; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** 8 pages, 6 tables, 3 figures. Under review
>
> **摘要:** Vision-Language-Action (VLA) models map multimodal perception and language instructions to executable robot actions, making them particularly vulnerable to behavioral backdoor manipulation: a hidden trigger introduced during training can induce unintended physical actions while nominal task performance remains intact. Prior work on VLA backdoors primarily studies untargeted attacks or task-level hijacking, leaving fine-grained control over individual actions largely unexplored. In this work, we present DropVLA, an action-level backdoor attack that forces a reusable action primitive (e.g., open_gripper) to execute at attacker-chosen decision points under a realistic pipeline-black-box setting with limited data-poisoning access, using a window-consistent relabeling scheme for chunked fine-tuning. On OpenVLA-7B evaluated with LIBERO, vision-only poisoning achieves 98.67%-99.83% attack success rate (ASR) with only 0.31% poisoned episodes while preserving 98.50%-99.17% clean-task retention, and successfully triggers the targeted action within 25 control steps at 500 Hz (0.05 s). Text-only triggers are unstable at low poisoning budgets, and combining text with vision provides no consistent ASR improvement over vision-only attacks. The backdoor remains robust to moderate trigger variations and transfers across evaluation suites (96.27%, 99.09%), whereas text-only largely fails (0.72%). We further validate physical-world feasibility on a 7-DoF Franka arm with pi0-fast, demonstrating non-trivial attack efficacy under camera-relative motion that induces image-plane trigger drift. These results reveal that VLA models can be covertly steered at the granularity of safety-critical actions with minimal poisoning and without observable degradation of nominal performance.
>
---
#### [replaced 085] Unsupervised Discovery of Failure Taxonomies from Deployment Logs
- **分类: cs.RO**

- **简介: 该论文属于故障分类任务，旨在从部署日志中无监督发现故障分类体系。解决手动分析大规模故障数据不现实的问题，通过视觉-语言推理和聚类方法，自动识别可操作的故障模式。**

- **链接: [https://arxiv.org/pdf/2506.06570](https://arxiv.org/pdf/2506.06570)**

> **作者:** Aryaman Gupta; Yusuf Umut Ciftci; Somil Bansal
>
> **摘要:** As robotic systems become increasingly integrated into real-world environments, ranging from autonomous vehicles to household assistants, they inevitably encounter diverse and unstructured scenarios that lead to failures. While such failures pose safety and reliability challenges, they also provide rich perceptual data for improving system robustness. However, manually analyzing large-scale failure datasets is impractical and does not scale. In this work, we introduce the problem of unsupervised discovery of failure taxonomies from large volumes of raw failure logs, aiming to obtain semantically coherent and actionable failure modes directly from perceptual trajectories. Our approach first infers structured failure explanations from multimodal inputs using vision-language reasoning, and then performs clustering in the resulting semantic reasoning space, enabling the discovery of recurring failure modes rather than isolated episode-level descriptions. We evaluate our method across robotic manipulation, indoor navigation, and autonomous driving domains, and demonstrate that the discovered taxonomies are consistent, interpretable, and practically useful. In particular, we show that structured failure taxonomies guide targeted data collection for offline policy refinement and enhance runtime failure monitoring systems. Website: this https URL
>
---
#### [replaced 086] Utility Theory based Cognitive Modeling in the Application of Robotics: A Survey
- **分类: cs.RO; cs.AI; cs.MA; cs.NE; eess.SY**

- **简介: 该论文属于机器人认知建模领域，探讨如何利用效用理论构建认知模型，解决多智能体系统中的决策与协作问题，综述现有研究并提出未来方向。**

- **链接: [https://arxiv.org/pdf/2306.09445](https://arxiv.org/pdf/2306.09445)**

> **作者:** Qin Yang
>
> **摘要:** Cognitive modeling, which explores the essence of cognition, including motivation, emotion, and perception, has been widely applied in the artificial intelligence (AI) agent domains, such as robotics. From the computational perspective, various cognitive functionalities have been developed through utility theory to provide a detailed and process-based understanding for specifying corresponding computational models of representations, mechanisms, and processes. Especially for decision-making and learning in multi-agent/robot systems (MAS/MRS), a suitable cognitive model can guide agents in choosing reasonable strategies to achieve their current needs and learning to cooperate and organize their behaviors, optimizing the system's utility, building stable and reliable relationships, and guaranteeing each group member's sustainable development, similar to the human society. This survey examines existing robotic systems for developmental cognitive models in the context of utility theory. We discuss the evolution of cognitive modeling in robotics from behavior-based robotics (BBR) and cognitive architectures to the properties of value systems in robots, such as the studies on motivations as artificial value systems, and the utility theory based cognitive modeling for generating and updating strategies in robotic interactions. Then, we examine the extent to which existing value systems support the application of robotics from an AI agent cognitive modeling perspective, including single-agent and multi-agent systems, trust among agents, and human-robot interaction. Finally, we survey the existing literature of current value systems in relevant fields and propose several promising research directions, along with some open problems that we deem necessary for further investigation.
>
---
#### [replaced 087] LIVE-GS: Online LiDAR-Inertial-Visual State Estimation and Globally Consistent Mapping with 3D Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，解决3DGS与LiDAR融合中的精度和效率问题，提出LIVE-GS框架实现高精度全局一致映射。**

- **链接: [https://arxiv.org/pdf/2507.23273](https://arxiv.org/pdf/2507.23273)**

> **作者:** Jaeseok Park; Chanoh Park; Minsu Kim; Minkyoung Kim; Soohwan Kim
>
> **摘要:** While 3D Gaussian Splatting (3DGS) enabled photorealistic mapping, its integration into SLAM has largely followed traditional camera-centric pipelines. As a result, they inherit well-known weaknesses such as high computational load, failure in texture-poor or illumination-varying environments, and limited operational range, particularly for RGB-D setups. On the other hand, LiDAR emerges as a robust alternative, but its integration with 3DGS introduces new challenges, such as the need for tighter global alignment for photorealistic quality and prolonged optimization times caused by sparse data. To address these challenges, we propose LIVE-GS, an online LiDAR-Inertial Visual SLAM framework that tightly couples 3D Gaussian Splatting with LiDAR-based surfels to ensure high-precision map consistency through global geometric optimization. Particularly, to handle sparse data, our system employs a depth-invariant Gaussian initialization strategy for efficient representation and a bounded sigmoid constraint to prevent uncontrolled Gaussian growth. Experiments on public and our datasets demonstrate competitive performance in rendering quality and map-building efficiency compared with representative 3DGS SLAM baselines.
>
---
#### [replaced 088] MetricNet: Recovering Metric Scale in Generative Navigation Policies
- **分类: cs.RO; cs.CV**

- **简介: 论文提出MetricNet，解决生成式导航中路径无度量尺度和短视问题，通过预测路标间距离提升导航安全性与效果。**

- **链接: [https://arxiv.org/pdf/2509.13965](https://arxiv.org/pdf/2509.13965)**

> **作者:** Abhijeet Nayak; Débora Oliveira Makowski; Samiran Gode; Cordelia Schmid; Wolfram Burgard
>
> **备注:** Accepted to ICRA'26
>
> **摘要:** Generative navigation policies have made rapid progress in improving end-to-end learned navigation. Despite their promising results, this paradigm has two structural problems. First, the sampled trajectories exist in an abstract, unscaled space without metric grounding. Second, the control strategy discards the full path, instead moving directly towards a single waypoint. This leads to short-sighted and unsafe actions, moving the robot towards obstacles that a complete and correctly scaled path would circumvent. To address these issues, we propose MetricNet, an effective add-on for generative navigation that predicts the metric distance between waypoints, grounding policy outputs in metric coordinates. We evaluate our method in simulation with a new benchmarking framework and show that executing MetricNet-scaled waypoints significantly improves both navigation and exploration performance. Beyond simulation, we further validate our approach in real-world experiments. Finally, we propose MetricNav, which integrates MetricNet into a navigation policy to guide the robot away from obstacles while still moving towards the goal.
>
---
#### [replaced 089] Context Matters! Relaxing Goals with LLMs for Feasible 3D Scene Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于3D场景规划任务，解决传统规划与LLM在复杂环境中的可行性问题。通过融合LLM与经典规划，提出ContextMatters框架，实现目标松弛与上下文适应。**

- **链接: [https://arxiv.org/pdf/2506.15828](https://arxiv.org/pdf/2506.15828)**

> **作者:** Emanuele Musumeci; Michele Brienza; Francesco Argenziano; Abdel Hakim Drid; Vincenzo Suriani; Daniele Nardi; Domenico D. Bloisi
>
> **摘要:** Embodied agents need to plan and act reliably in real and complex 3D environments. Classical planning (e.g., PDDL) offers structure and guarantees, but in practice it fails under noisy perception and incorrect predicate grounding. On the other hand, Large Language Models (LLMs)-based planners leverage commonsense reasoning, yet frequently propose actions that are unfeasible or unsafe. Following recent works that combine the two approaches, we introduce ContextMatters, a framework that fuses LLMs and classical planning to perform hierarchical goal relaxation: the LLM helps ground symbols to the scene and, when the target is unreachable, it proposes functionally equivalent goals that progressively relax constraints, adapting the goal to the context of the agent's environment. Operating on 3D Scene Graphs, this mechanism turns many nominally unfeasible tasks into tractable plans and enables context-aware partial achievement when full completion is not achievable. Our experimental results show a +52.45% Success Rate improvement over state-of-the-art LLMs+PDDL baseline, demonstrating the effectiveness of our approach. Moreover, we validate the execution of ContextMatter in a real world scenario by deploying it on a TIAGo robot. Code, dataset, and supplementary materials are available to the community at this https URL.
>
---
