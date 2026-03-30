# 机器人 cs.RO

- **最新发布 29 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的微调任务，旨在解决SFT的分布偏移和RL的样本效率低问题。提出VLA-OPD框架，通过在线策略蒸馏提升模型性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.26666](https://arxiv.org/pdf/2603.26666)**

> **作者:** Zhide Zhong; Haodong Yan; Junfeng Li; Junjie He; Tianran Zhang; Haoang Li
>
> **摘要:** Although pre-trained Vision-Language-Action (VLA) models exhibit impressive generalization in robotic manipulation, post-training remains crucial to ensure reliable performance during deployment. However, standard offline Supervised Fine-Tuning (SFT) suffers from distribution shifts and catastrophic forgetting of pre-trained capabilities, while online Reinforcement Learning (RL) struggles with sparse rewards and poor sample efficiency. In this paper, we propose On-Policy VLA Distillation (VLA-OPD), a framework bridging the efficiency of SFT with the robustness of RL. Instead of relying on sparse environmental rewards, VLA-OPD leverages an expert teacher to provide dense, token-level supervision on the student's self-generated trajectories. This enables active error correction on policy-induced states while preserving pre-trained general capabilities through gentle alignment. Crucially, we formulate VLA-OPD via a Reverse-KL objective. Unlike standard Forward-KL that induces mode-covering entropy explosion, or Hard-CE that causes premature entropy collapse, our bounded mode-seeking objective ensures stable policy learning by filtering out the teacher's epistemic uncertainty while maintaining action diversity. Experiments on LIBERO and RoboTwin2.0 benchmarks demonstrate that VLA-OPD significantly improves sample efficiency over RL and robustness over SFT, while effectively mitigating catastrophic forgetting during post-training.
>
---
#### [new 002] Addressing Ambiguity in Imitation Learning through Product of Experts based Negative Feedback
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决模仿学习中因示范不明确导致的性能问题。通过引入基于专家产品负反馈机制，提升机器人从失败中学习的能力，显著提高任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.26467](https://arxiv.org/pdf/2603.26467)**

> **作者:** John Bateman; Andy M. Tyrrell; Jihong Zhu
>
> **摘要:** Programming robots to perform complex tasks is often difficult and time consuming, requiring expert knowledge and skills in robot software and sometimes hardware. Imitation learning is a method for training robots to perform tasks by leveraging human expertise through demonstrations. Typically, the assumption is that those demonstrations are performed by a single, highly competent expert. However, in many real-world applications that use user demonstrations for tasks or incorporate both user data and pretrained data, such as home robotics including assistive robots, this is unlikely to be the case. This paper presents research towards a system which can leverage suboptimal demonstrations to solve ambiguous tasks; and particularly learn from its own failures. This is a negative-feedback system which achieves significant improvement over purely positive imitation learning for ambiguous tasks, achieving a 90% improvement in success rate against a system that does not utilise negative feedback, compared to a 50% improvement in success rate when utilised on a real robot, as well as demonstrating higher efficacy, memory efficiency and time efficiency than a comparable negative feedback scheme. The novel scheme presented in this paper is validated through simulated and real-robot experiments.
>
---
#### [new 003] DTP-Attack: A decision-based black-box adversarial attack on trajectory prediction
- **分类: cs.RO**

- **简介: 该论文属于轨迹预测任务，旨在解决对抗攻击问题。提出DTP-Attack方法，在无需模型信息的情况下实现轨迹预测系统的决策攻击。**

- **链接: [https://arxiv.org/pdf/2603.26462](https://arxiv.org/pdf/2603.26462)**

> **作者:** Jiaxiang Li; Jun Yan; Daniel Watzenig; Huilin Yin
>
> **备注:** ICRA 2026
>
> **摘要:** Trajectory prediction systems are critical for autonomous vehicle safety, yet remain vulnerable to adversarial attacks that can cause catastrophic traffic behavior misinterpretations. Existing attack methods require white-box access with gradient information and rely on rigid physical constraints, limiting real-world applicability. We propose DTP-Attack, a decision-based black-box adversarial attack framework tailored for trajectory prediction systems. Our method operates exclusively on binary decision outputs without requiring model internals or gradients, making it practical for real-world scenarios. DTP-Attack employs a novel boundary walking algorithm that navigates adversarial regions without fixed constraints, naturally maintaining trajectory realism through proximity preservation. Unlike existing approaches, our method supports both intention misclassification attacks and prediction accuracy degradation. Extensive evaluation on nuScenes and Apolloscape datasets across state-of-the-art models including Trajectron++ and Grip++ demonstrates superior performance. DTP-Attack achieves 41 - 81% attack success rates for intention misclassification attacks that manipulate perceived driving maneuvers with perturbations below 0.45 m, and increases prediction errors by 1.9 - 4.2 for accuracy degradation. Our method consistently outperforms existing black-box approaches while maintaining high controllability and reliability across diverse scenarios. These results reveal fundamental vulnerabilities in current trajectory prediction systems, highlighting urgent needs for robust defenses in safety-critical autonomous driving applications.
>
---
#### [new 004] 120 Minutes and a Laptop: Minimalist Image-goal Navigation via Unsupervised Exploration and Offline RL
- **分类: cs.RO**

- **简介: 该论文属于图像目标导航任务，解决现实环境中高效学习问题。提出MINav方法，通过无监督探索和离线强化学习，在消费级设备上快速训练并部署导航策略。**

- **链接: [https://arxiv.org/pdf/2603.26441](https://arxiv.org/pdf/2603.26441)**

> **作者:** Xiaoming Liu; Borong Zhang; Qingbiao Li; Steven Morad
>
> **备注:** 8 pages, 8 figures, submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** The prevailing paradigm for image-goal visual navigation often assumes access to large-scale datasets, substantial pretraining, and significant computational resources. In this work, we challenge this assumption. We show that we can collect a dataset, train an in-domain policy, and deploy it to the real world (1) in less than 120 minutes, (2) on a consumer laptop, (3) without any human intervention. Our method, MINav, formulates image-goal navigation as an offline goal-conditioned reinforcement learning problem, combining unsupervised data collection with hindsight goal relabeling and offline policy learning. Experiments in simulation and the real world show that MINav improves exploration efficiency, outperforms zero-shot navigation baselines in target environments, and scales favorably with dataset size. These results suggest that effective real-world robotic learning can be achieved with high computational efficiency, lowering the barrier to rapid policy prototyping and deployment.
>
---
#### [new 005] Can Vision Foundation Models Navigate? Zero-Shot Real-World Evaluation and Lessons Learned
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉导航任务，旨在评估视觉导航模型的性能与局限性。通过真实世界测试，分析模型在路径质量、碰撞行为及环境变化下的表现，揭示其几何理解、位置区分和泛化能力的问题。**

- **链接: [https://arxiv.org/pdf/2603.25937](https://arxiv.org/pdf/2603.25937)**

> **作者:** Maeva Guerrier; Karthik Soma; Jana Pavlasek; Giovanni Beltrame
>
> **摘要:** Visual Navigation Models (VNMs) promise generalizable, robot navigation by learning from large-scale visual demonstrations. Despite growing real-world deployment, existing evaluations rely almost exclusively on success rate, whether the robot reaches its goal, which conceals trajectory quality, collision behavior, and robustness to environmental change. We present a real-world evaluation of five state-of-the-art VNMs (GNM, ViNT, NoMaD, NaviBridger, and CrossFormer) across two robot platforms and five environments spanning indoor and outdoor settings. Beyond success rate, we combine path-based metrics with vision-based goal-recognition scores and assess robustness through controlled image perturbations (motion blur, sunflare). Our analysis uncovers three systematic limitations: (a) even architecturally sophisticated diffusion and transformer-based models exhibit frequent collisions, indicating limited geometric understanding; (b) models fail to discriminate between different locations that are perceptually similar, however some semantics differences are present, causing goal prediction errors in repetitive environments; and (c) performance degrades under distribution shift. We will publicly release our evaluation codebase and dataset to facilitate reproducible benchmarking of VNMs.
>
---
#### [new 006] The Multi-AMR Buffer Storage, Retrieval, and Reshuffling Problem: Exact and Heuristic Approaches
- **分类: cs.RO; cs.AI; cs.MA; math.OC**

- **简介: 该论文研究多AMR协同的缓冲区存储与调度问题，旨在解决高密度生产环境中的自动化管理难题。通过构建数学模型和启发式算法提升效率。**

- **链接: [https://arxiv.org/pdf/2603.26542](https://arxiv.org/pdf/2603.26542)**

> **作者:** Max Disselnmeyer; Thomas Bömer; Laura Dörr; Bastian Amberg; Anne Meyer
>
> **备注:** 52 pages, 15 figures and tables
>
> **摘要:** Buffer zones are essential in production systems to decouple sequential processes. In dense floor storage environments, such as space-constrained brownfield facilities, manual operation is increasingly challenged by severe labor shortages and rising operational costs. Automating these zones requires solving the Buffer Storage, Retrieval, and Reshuffling Problem (BSRRP). While previous work has addressed scenarios where the focus is limited to reshuffling and retrieving a fixed set of items, real-world manufacturing necessitates an adaptive approach that also incorporates arriving unit loads. This paper introduces the Multi-AMR BSRRP, coordinating a robot fleet to manage concurrent reshuffling, alongside time-windowed storage and retrieval tasks, within a shared floor area. We formulate a Binary Integer Programming (IP) model to obtain exact solutions for benchmarking purposes. As the problem is NP-hard, rendering exact methods computationally intractable for industrial scales, we propose a hierarchical heuristic. This approach decomposes the problem into an A* search for task-level sequence planning of unit load placements, and a Constraint Programming (CP) approach for multi-robot coordination and scheduling. Experiments demonstrate orders-of-magnitude computation time reductions compared to the exact formulation. These results confirm the heuristic's viability as responsive control logic for high-density production environments.
>
---
#### [new 007] Realtime-VLA V2: Learning to Run VLAs Fast, Smooth, and Accurate
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在提升VLA模型在真实机器人上的执行速度与精度。通过校准、规划与学习方法，实现高效、平滑的机器人操作。**

- **链接: [https://arxiv.org/pdf/2603.26360](https://arxiv.org/pdf/2603.26360)**

> **作者:** Chen Yang; Yucheng Hu; Yunchao Ma; Yunhuan Yang; Jing Tan; Haoqiang Fan
>
> **摘要:** In deployment of the VLA models to real-world robotic tasks, execution speed matters. In previous work arXiv:2510.26742 we analyze how to make neural computation of VLAs on GPU fast. However, we leave the question of how to actually deploy the VLA system on the real robots open. In this report we describe a set of practical techniques to achieve the end-to-end result of running a VLA-driven robot at an impressive speed in real world tasks that require both accuracy and dexterity. The stack of technology ranges across calibration, planning & control, and learning based method to identify optimal execution speed. In the tasks we show, the robot even executes in a speed on par with casual human operation and approaching the hardware limit of our lightweight arm. The unaccelerated videos and inference traces are provided in this https URL.
>
---
#### [new 008] Generalizable task-oriented object grasping through LLM-guided ontology and similarity-based planning
- **分类: cs.RO**

- **简介: 该论文属于任务导向抓取（TOG）领域，解决对象部分识别与抓取规划的泛化问题。通过构建语义本体、几何分析和相似匹配方法，提升抓取的准确性和适应性。**

- **链接: [https://arxiv.org/pdf/2603.26412](https://arxiv.org/pdf/2603.26412)**

> **作者:** Hao Chen; Takuya Kiyokawa; Weiwei Wan; Kensuke Harada
>
> **备注:** Accepted by Robotics and Autonomous Systems
>
> **摘要:** Task-oriented grasping (TOG) is more challenging than simple object grasping because it requires precise identification of object parts and careful selection of grasping areas to ensure effective and robust manipulation. While recent approaches have trained large-scale vision-language models to integrate part-level object segmentation with task-aware grasp planning, their instability in part recognition and grasp inference limits their ability to generalize across diverse objects and tasks. To address this issue, we introduce a novel, geometry-centric strategy for more generalizable TOG that does not rely on semantic features from visual recognition, effectively overcoming the viewpoint sensitivity of model-based approaches. Our main proposals include: 1) an object-part-task ontology for functional part selection based on intuitive human commands, constructed using a Large Language Model (LLM); 2) a sampling-based geometric analysis method for identifying the selected object part from observed point clouds, incorporating multiple point distribution and distance metrics; and 3) a similarity matching framework for imitative grasp planning, utilizing similar known objects with pre-existing segmentation and grasping knowledge as references to guide the planning for unknown targets. We validate the high accuracy of our approach in functional part selection, identification, and grasp generation through real-world experiments. Additionally, we demonstrate the method's generalization capabilities to novel-category objects by extending existing ontological knowledge, showcasing its adaptability to a broad range of objects and tasks.
>
---
#### [new 009] DFM-VLA: Iterative Action Refinement for Robot Manipulation via Discrete Flow Matching
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DFM-VLA，解决机器人操作中动作序列生成的误差修正问题，通过离散流匹配实现迭代优化，提升操作性能与成功率。**

- **链接: [https://arxiv.org/pdf/2603.26320](https://arxiv.org/pdf/2603.26320)**

> **作者:** Jiayi Chen; Wenxuan Song; Shuai Chen; Jingbo Wang; Zhijun Li; Haoang Li
>
> **摘要:** Vision--Language--Action (VLA) models that encode actions using a discrete tokenization scheme are increasingly adopted for robotic manipulation, but existing decoding paradigms remain fundamentally limited. Whether actions are decoded sequentially by autoregressive VLAs or in parallel by discrete diffusion VLAs, once a token is generated, it is typically fixed and cannot be revised in subsequent iterations, so early token errors cannot be effectively corrected later. We propose DFM-VLA, a discrete flow matching VLA for iterative refinement of action tokens. DFM-VLA~models a token-level probability velocity field that dynamically updates the full action sequence across refinement iterations. We investigate two ways to construct the velocity field: an auxiliary velocity-head formulation and an action-embedding-guided formulation. Our framework further adopts a two-stage decoding strategy with an iterative refinement stage followed by deterministic validation for stable convergence. Extensive experiments on CALVIN, LIBERO, and real-world manipulation tasks show that DFM-VLA consistently outperforms strong autoregressive, discrete diffusion, and continuous diffusion baselines in manipulation performance while retaining high inference efficiency. In particular, DFM-VLA achieves an average success length of 4.44 on CALVIN and an average success rate of 95.7\% on LIBERO, highlighting the value of action refinement via discrete flow matching for robotic manipulation. Our project is available \url{this https URL}
>
---
#### [new 010] Meta-Adaptive Beam Search Planning for Transformer-Based Reinforcement Learning Control of UAVs with Overhead Manipulators under Flight Disturbances
- **分类: cs.RO**

- **简介: 该论文属于无人机控制任务，解决无人机与机械臂协同运动中的跟踪误差问题。通过引入基于Transformer的强化学习框架和自适应束搜索规划器，提升控制稳定性与精度。**

- **链接: [https://arxiv.org/pdf/2603.26612](https://arxiv.org/pdf/2603.26612)**

> **作者:** Hazim Alzorgan; Sayed Pedram Haeri Boroujeni; Abolfazl Razi
>
> **摘要:** Drones equipped with overhead manipulators offer unique capabilities for inspection, maintenance, and contact-based interaction. However, the motion of the drone and its manipulator is tightly linked, and even small attitude changes caused by wind or control imperfections shift the end-effector away from its intended path. This coupling makes reliable tracking difficult and also limits the direct use of learning-based arm controllers that were originally designed for fixed-base robots. These effects appear consistently in our tests whenever the UAV body experiences drift or rapid attitude corrections. To address this behavior, we develop a reinforcement-learning (RL) framework with a transformer-based double deep Q learning (DDQN), with the core idea of using an adaptive beam-search planner that applies a short-horizon beam search over candidate control sequences using the learned critic as the forward estimator. This allows the controller to anticipate the end-effector's motion through simulated rollouts rather than executing those actions directly on the actual model, realizing a software-in-the-loop (SITL) approach. The lookahead relies on value estimates from a Transformer critic that processes short sequences of states, while a DDQN backbone provides the one-step targets needed to keep the learning process stable. Evaluated on a 3-DoF aerial manipulator under identical training conditions, the proposed meta-adaptive planner shows the strongest overall performance with a 10.2% reward increase, a substantial reduction in mean tracking error (from about 6% to 3%), and a 29.6% improvement in the combined reward-error metric relative to the DDQN baseline. Our method exhibits elevated stability in tracking target tip trajectory (by maintaining 5 cm tracking error) when the drone base exhibits drifts due to external disturbances, as opposed to the fixed-beam and Transformer-only variants.
>
---
#### [new 011] DiffusionAnything: End-to-End In-context Diffusion Learning for Unified Navigation and Pre-Grasp Motion
- **分类: cs.RO**

- **简介: 该论文提出DiffusionAnything，解决机器人导航与抓取任务中的运动规划问题，通过统一的图像空间扩散策略，实现零样本泛化和低资源部署。**

- **链接: [https://arxiv.org/pdf/2603.26322](https://arxiv.org/pdf/2603.26322)**

> **作者:** Iana Zhura; Yara Mahmoud; Jeffrin Sam; Hung Khang Nguyen; Didar Seyidov; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **摘要:** Efficiently predicting motion plans directly from vision remains a fundamental challenge in robotics, where planning typically requires explicit goal specification and task-specific design. Recent vision-language-action (VLA) models infer actions directly from visual input but demand massive computational resources, extensive training data, and fail zero-shot in novel scenes. We present a unified image-space diffusion policy handling both meter-scale navigation and centimeter-scale manipulation via multi-scale feature modulation, with only 5 minutes of self-supervised data per task. Three key innovations drive the framework: (1) Multi-scale FiLM conditioning on task mode, depth scale, and spatial attention enables task-appropriate behavior in a single model; (2) trajectory-aligned depth prediction focuses metric 3D reasoning along generated waypoints; (3) self-supervised attention from AnyTraverse enables goal-directed inference without vision-language models and depth sensors. Operating purely from RGB input (2.0 GB memory, 10 Hz), the model achieves robust zero-shot generalization to novel scenes while remaining suitable for onboard deployment.
>
---
#### [new 012] Adapt as You Say: Online Interactive Bimanual Skill Adaptation via Human Language Feedback
- **分类: cs.RO**

- **简介: 该论文属于机器人技能适应任务，解决如何通过语言反馈在线适应双臂操作技能的问题。提出BiSAIL框架，实现零样本在线调整，提升任务泛化与跨平台适应能力。**

- **链接: [https://arxiv.org/pdf/2603.26466](https://arxiv.org/pdf/2603.26466)**

> **作者:** Zhuo Li; Dianxi Li; Tao Teng; Quentin Rouxel; Zhipeng Dong; Dennis Hong; Darwin Caldwell; Fei Chen
>
> **备注:** 11 pages, 15 figures, submitted to IEEE TMECH
>
> **摘要:** Developing general-purpose robots capable of autonomously operating in human living environments requires the ability to adapt to continuously evolving task conditions. However, adapting high-dimensional coordinated bimanual skills to novel task variations at deployment remains a fundamental challenge. In this work, we present BiSAIL (Bimanual Skill Adaptation via Interactive Language), a novel framework that enables zero-shot online adaptation of offline-learned bimanual skills through interactive language feedback. The key idea of BiSAIL is to adopt a hierarchical reason-then-modulate paradigm, which first infers generalized adaptation objectives from multimodal task variations, and then adapts bimanual motions via diffusion modulation to achieve the inferred objectives. Extensive real-robot experiments across six bimanual tasks and two dual-arm platforms demonstrate that BiSAIL significantly outperforms existing methods in human-in-the-loop adaptability, task generalization and cross-embodiment scalability. This work enables the development of adaptive bimanual assistants that can be flexibly customized by non-expert users via intuitive verbal corrections. Experimental videos and code are available at this https URL.
>
---
#### [new 013] Chasing Autonomy: Dynamic Retargeting and Control Guided RL for Performant and Controllable Humanoid Running
- **分类: cs.RO**

- **简介: 该论文研究人形机器人动态跑步控制问题，通过强化学习实现高效、可控的运动。提出优化管道生成周期参考库，提升速度与续航能力，并验证其在真实环境中的表现。**

- **链接: [https://arxiv.org/pdf/2603.25902](https://arxiv.org/pdf/2603.25902)**

> **作者:** Zachary Olkin; William D. Compton; Ryan M. Bena; Aaron D. Ames
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Humanoid robots have the promise of locomoting like humans, including fast and dynamic running. Recently, reinforcement learning (RL) controllers that can mimic human motions have become popular as they can generate very dynamic behaviors, but they are often restricted to single motion play-back which hinders their deployment in long duration and autonomous locomotion. In this paper, we present a pipeline to dynamically retarget human motions through an optimization routine with hard constraints to generate improved periodic reference libraries from a single human demonstration. We then study the effect of both the reference motion and the reward structure on the reference and commanded velocity tracking, concluding that a goal-conditioned and control-guided reward which tracks dynamically optimized human data results in the best performance. We deploy the policy on hardware, demonstrating its speed and endurance by achieving running speeds of up to 3.3 m/s on a Unitree G1 robot and traversing hundreds of meters in real-world environments. Additionally, to demonstrate the controllability of the locomotion, we use the controller in a full perception and planning autonomy stack for obstacle avoidance while running outdoors.
>
---
#### [new 014] T-800: An 800 Hz Data Glove for Precise Hand Gesture Tracking
- **分类: cs.RO**

- **简介: 该论文提出T-800数据手套，解决高频率手部动作捕捉问题，通过800Hz采样实现精准手势追踪。**

- **链接: [https://arxiv.org/pdf/2603.26403](https://arxiv.org/pdf/2603.26403)**

> **作者:** Haoyang Luo; Zihang Zhao; Leiyao Cui; Saiyao Zhang; Liu Yang; Zhi Han; Xiyuan Tang; Yixin Zhu
>
> **摘要:** Human dexterity relies on rapid, sub-second motor adjustments, yet capturing these high-frequency dynamics remains an enduring challenge in biomechanics and robotics. Existing motion capture paradigms are compromised by a trade-off between temporal resolution and visual occlusion, failing to record the fine-grained hand motion of fast, contact-rich manipulation. Here we introduce T-800, a high-bandwidth data glove system that achieves synchronized, full-hand motion tracking at 800 Hz. By integrating a novel broadcast-based synchronization mechanism with a mechanical stress isolation architecture, our system maintains sub-frame temporal alignment across 18 distributed inertial measurement units (IMUs) during extended, vigorous movements. We demonstrate that T-800 recovers fine-grained manipulation details previously lost to temporal undersampling. Our analysis reveals that human dexterity exhibits significantly high-frequency motion energy (>100 Hz) that was fundamentally inaccessible due to the Nyquist sampling limit imposed by previous hardware constraints. To validate the system's utility for robotic manipulation, we implement a kinematic retargeting algorithm that maps T-800's high-fidelity human gestures onto dexterous robotic hand models. This demonstrates that the high-frequency motion data can be accurately translated while respecting the kinematic constraints of robotic hands, providing the rich behavioral data necessary for training robust control policies in the future.
>
---
#### [new 015] Optimal Prioritized Dissipation and Closed-Form Damping Limitation under Actuator Constraints for Haptic Interfaces
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于力反馈系统稳定性研究，解决haptic接口在执行器约束下的稳定与透明度问题。提出一种基于优先耗散的阻尼限制方法，优化能耗分配，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2603.26347](https://arxiv.org/pdf/2603.26347)**

> **作者:** Camilla Celli; Andrea Bini; Valerio Novelli; Alessandro Filippeschi; Francesco Porcini; Antonio Frisoli
>
> **摘要:** In haptics, guaranteeing stability is essential to ensure safe interaction with remote or virtual environments. One of the most relevant methods at the state-of-the-art is the Time Domain Passivity Approach (TDPA). However, its high conservatism leads to a significant degradation of transparency. Moreover, the stabilizing action may conflict with the device's physical limitations. State-of-the-art solutions have attempted to address these actuator limits, but they still fail to account simultaneously for the power limits of each actuator while maximizing transparency. This work proposes a new damping limitation method based on prioritized dissipation actions. It prioritizes an optimal dissipation direction that minimizes actuator load, while any excess dissipation is allocated to the orthogonal hyperplane. The solution provides a closed-form formulation and is robust in multi-DoF scenarios, even in the presence of actuator and motion anisotropies. The method is experimentally validated using a parallel haptic interface interacting with a virtual environment and tested under different operating conditions.
>
---
#### [new 016] Policy-Guided World Model Planning for Language-Conditioned Visual Navigation
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于视觉导航任务，解决语言引导下的长期路径规划问题。通过结合预训练策略与世界模型，提升导航准确性和指令遵循能力。**

- **链接: [https://arxiv.org/pdf/2603.25981](https://arxiv.org/pdf/2603.25981)**

> **作者:** Amirhosein Chahe; Lifeng Zhou
>
> **摘要:** Navigating to a visually specified goal given natural language instructions remains a fundamental challenge in embodied AI. Existing approaches either rely on reactive policies that struggle with long-horizon planning, or employ world models that suffer from poor action initialization in high-dimensional spaces. We present PiJEPA, a two-stage framework that combines the strengths of learned navigation policies with latent world model planning for instruction-conditioned visual navigation. In the first stage, we finetune an Octo-based generalist policy, augmented with a frozen pretrained vision encoder (DINOv2 or V-JEPA-2), on the CAST navigation dataset to produce an informed action distribution conditioned on the current observation and language instruction. In the second stage, we use this policy-derived distribution to warm-start Model Predictive Path Integral (MPPI) planning over a separately trained JEPA world model, which predicts future latent states in the embedding space of the same frozen encoder. By initializing the MPPI sampling distribution from the policy prior rather than from an uninformed Gaussian, our planner converges faster to high-quality action sequences that reach the goal. We systematically study the effect of the vision encoder backbone, comparing DINOv2 and V-JEPA-2, across both the policy and world model components. Experiments on real-world navigation tasks demonstrate that PiJEPA significantly outperforms both standalone policy execution and uninformed world model planning, achieving improved goal-reaching accuracy and instruction-following fidelity.
>
---
#### [new 017] ETA-VLA: Efficient Token Adaptation via Temporal Fusion and Intra-LLM Sparsification for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决多帧图像处理带来的计算负担问题。提出ETA-VLA框架，通过稀疏化和时间融合提升效率。**

- **链接: [https://arxiv.org/pdf/2603.25766](https://arxiv.org/pdf/2603.25766)**

> **作者:** Yiru Wang; Anqing Jiang; Shuo Wang; Yuwen Heng; Zichong Gu; Hao Sun
>
> **摘要:** The integration of Vision-Language-Action (VLA) models into autonomous driving systems offers a unified framework for interpreting complex scenes and executing control commands. However, the necessity to incorporate historical multi-view frames for accurate temporal reasoning imposes a severe computational burden, primarily driven by the quadratic complexity of self-attention mechanisms in Large Language Models (LLMs). To alleviate this bottleneck, we propose ETA-VLA, an Efficient Token Adaptation framework for VLA models. ETA-VLA processes the past $n$ frames of multi-view images and introduces a novel Intra-LLM Sparse Aggregator (ILSA). Drawing inspiration from human driver attention allocation, ILSA dynamically identifies and prunes redundant visual tokens guided by textual queries and temporal consistency. Specifically, we utilize a text-guided scoring mechanism alongside a diversity-preserving sparsification strategy to select a sparse subset of critical tokens, ensuring comprehensive awareness of the driving scene. Extensive experiments on the NAVSIM v2 demonstrate that ETA-VLA achieves driving performance comparable to state-of-the-art baselines while reducing computational FLOPs by approximately 32\%. Notably, our method prunes 85% of visual tokens and reduces inference FLOPs by 61\%, but still retaining 94% of the original accuracy on the NAVSIM v2 benchmark.
>
---
#### [new 018] Massive Parallel Deep Reinforcement Learning for Active SLAM
- **分类: cs.RO**

- **简介: 该论文属于Active SLAM任务，旨在解决DRL训练效率低的问题。提出一种可扩展的并行DRL框架，显著缩短训练时间，支持连续动作和更真实场景。**

- **链接: [https://arxiv.org/pdf/2603.25834](https://arxiv.org/pdf/2603.25834)**

> **作者:** Martín Arce Llobera; Julio A. Placed; Mariano De Paula; Pablo De Cristóforis
>
> **摘要:** Recent advances in parallel computing and GPU acceleration have created new opportunities for computation-intensive learning problems such as Active SLAM -- where actions are selected to reduce uncertainty and improve joint mapping and localization. However, existing DRL-based approaches remain constrained by the lack of scalable parallel training. In this work, we address this challenge by proposing a scalable end-to-end DRL framework for Active SLAM that enables massively parallel training. Compared with the state of the art, our method significantly reduces training time, supports continuous action spaces and facilitates the exploration of more realistic scenarios. It is released as an open-source framework to promote reproducibility and community adoption.
>
---
#### [new 019] SwarmCoDe: A Scalable Co-Design Framework for Heterogeneous Robot Swarms via Dynamic Speciation
- **分类: cs.RO; cs.MA; cs.NE**

- **简介: 该论文提出SwarmCoDe框架，解决大规模异构机器人集群的协同设计问题。通过动态物种划分和进化机制，实现高效优化与规模扩展。**

- **链接: [https://arxiv.org/pdf/2603.26240](https://arxiv.org/pdf/2603.26240)**

> **作者:** Andrew Wilhelm; Josie Hughes
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Robot swarms offer inherent robustness and the capacity to execute complex, collaborative tasks surpassing the capabilities of single-agent systems. Co-designing these systems is critical, as marginal improvements in individual performance or unit cost compound significantly at scale. However, under traditional frameworks, this scale renders co-design intractable due to exponentially large, non-intuitive design spaces. To address this, we propose SwarmCoDe, a novel Collaborative Co-Evolutionary Algorithm (CCEA) that utilizes dynamic speciation to automatically scale swarm heterogeneity to match task complexity. Inspired by biological signaling mechanisms for inter-species cooperation, the algorithm uses evolved genetic tags and a selectivity gene to facilitate the emergent identification of symbiotically beneficial partners without predefined species boundaries. Additionally, an evolved dominance gene dictates the relative swarm composition, decoupling the physical swarm size from the evolutionary population. We apply SwarmCoDe to simultaneously optimize task planning and hardware morphology under fabrication budgets, successfully evolving specialized swarms of up to 200 agents -- four times the size of the evolutionary population. This framework provides a scalable, computationally viable pathway for the holistic co-design of large-scale, heterogeneous robot swarms.
>
---
#### [new 020] Ruka-v2: Tendon Driven Open-Source Dexterous Hand with Wrist and Abduction for Robot Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文介绍Ruka-v2，一款开源、肌腱驱动的人类手部机器人，解决缺乏灵活硬件的问题。新增腕部和手指外展功能，提升操作能力，适用于机器人学习任务。**

- **链接: [https://arxiv.org/pdf/2603.26660](https://arxiv.org/pdf/2603.26660)**

> **作者:** Xinqi; Ruoxi Hu; Alejandro Ojeda Olarte; Zhuoran Chen; Kenny Ma; Charles Cheng Ji; Lerrel Pinto; Raunaq Bhirangi; Irmak Guzey
>
> **摘要:** Lack of accessible and dexterous robot hardware has been a significant bottleneck to achieving human-level dexterity in robots. Last year, we released Ruka, a fully open-sourced, tendon-driven humanoid hand with 11 degrees of freedom - 2 per finger and 3 at the thumb - buildable for under $1,300. It was one of the first fully open-sourced humanoid hands, and introduced a novel data-driven approach to finger control that captures tendon dynamics within the control system. Despite these contributions, Ruka lacked two degrees of freedom essential for closely imitating human behavior: wrist mobility and finger adduction/abduction. In this paper, we introduce Ruka-v2: a fully open-sourced, tendon-driven humanoid hand featuring a decoupled 2-DOF parallel wrist and abduction/adduction at the fingers. The parallel wrist adds smooth, independent flexion/extension and radial/ulnar deviation, enabling manipulation in confined environments such as cabinets. Abduction enables motions such as grasping thin objects, in-hand rotation, and calligraphy. We present the design of Ruka-v2 and evaluate it against Ruka through user studies on teleoperated tasks, finding a 51.3% reduction in completion time and a 21.2% increase in success rate. We further demonstrate its full range of applications for robot learning: bimanual and single-arm teleoperation across 13 dexterous tasks, and autonomous policy learning on 3 tasks. All 3D print files, assembly instructions, controller software, and videos are available at this https URL .
>
---
#### [new 021] Partial Motion Imitation for Learning Cart Pushing with Legged Manipulators
- **分类: cs.RO**

- **简介: 该论文属于腿足机器人操作任务，旨在解决运动与操作协同控制难题。通过部分模仿学习，将运动策略迁移至推车任务，提升操作稳定性与精度。**

- **链接: [https://arxiv.org/pdf/2603.26659](https://arxiv.org/pdf/2603.26659)**

> **作者:** Mili Das; Morgan Byrd; Donghoon Baek; Sehoon Ha
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Loco-manipulation is a key capability for legged robots to perform practical mobile manipulation tasks, such as transporting and pushing objects, in real-world environments. However, learning robust loco-manipulation skills remains challenging due to the difficulty of maintaining stable locomotion while simultaneously performing precise manipulation behaviors. This work proposes a partial imitation learning approach that transfers the locomotion style learned from a locomotion task to cart loco-manipulation. A robust locomotion policy is first trained with extensive domain and terrain randomization, and a loco-manipulation policy is then learned by imitating only lower-body motions using a partial adversarial motion prior. We conduct experiments demonstrating that the learned policy successfully pushes a cart along diverse trajectories in IsaacLab and transfers effectively to MuJoCo. We also compare our method to several baselines and show that the proposed approach achieves more stable and accurate loco-manipulation behaviors.
>
---
#### [new 022] Emergent Neural Automaton Policies: Learning Symbolic Structure from Visuomotor Trajectories
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决长周期任务中策略缺乏结构先验的问题。提出ENAP框架，通过从视觉运动轨迹中自适应生成符号化策略，提升样本效率与可解释性。**

- **链接: [https://arxiv.org/pdf/2603.25903](https://arxiv.org/pdf/2603.25903)**

> **作者:** Yiyuan Pan; Xusheng Luo; Hanjiang Hu; Peiqi Yu; Changliu Liu
>
> **摘要:** Scaling robot learning to long-horizon tasks remains a formidable challenge. While end-to-end policies often lack the structural priors needed for effective long-term reasoning, traditional neuro-symbolic methods rely heavily on hand-crafted symbolic priors. To address the issue, we introduce ENAP (Emergent Neural Automaton Policy), a framework that allows a bi-level neuro-symbolic policy adaptively emerge from visuomotor demonstrations. Specifically, we first employ adaptive clustering and an extension of the L* algorithm to infer a Mealy state machine from visuomotor data, which serves as an interpretable high-level planner capturing latent task modes. Then, this discrete structure guides a low-level reactive residual network to learn precise continuous control via behavior cloning (BC). By explicitly modeling the task structure with discrete transitions and continuous residuals, ENAP achieves high sample efficiency and interpretability without requiring task-specific labels. Extensive experiments on complex manipulation and long-horizon tasks demonstrate that ENAP outperforms state-of-the-art (SoTA) end-to-end VLA policies by up to 27% in low-data regimes, while offering a structured representation of robotic intent (Fig. 1).
>
---
#### [new 023] Line-of-Sight-Constrained Multi-Robot Mapless Navigation via Polygonal Visible Regions
- **分类: cs.RO**

- **简介: 该论文属于多机器人导航任务，解决未知障碍物下的视线连通性问题。通过构建局部可见区域并共享信息，实现无需地图的多机器人连通导航。**

- **链接: [https://arxiv.org/pdf/2603.26314](https://arxiv.org/pdf/2603.26314)**

> **作者:** Ruofei Bai; Shenghai Yuan; Xinhang Xu; Xingyu Ji; Xiaowei Li; Hongliang Guo; Wei-Yun Yau; Lihua Xie
>
> **备注:** 10 pages, 7 figures. See videos and code: this https URL
>
> **摘要:** Multi-robot systems rely on underlying connectivity to ensure reliable communication and timely coordination. This paper studies the line-of-sight (LoS) connectivity maintenance problem in multi-robot navigation with unknown obstacles. Prior works typically assume known environment maps to formulate LoS constraints between robots, which hinders their practical deployment. To overcome this limitation, we propose an inherently distributed approach where each robot only constructs an egocentric visible region based on its real-time LiDAR scans, instead of endeavoring to build a global map online. The individual visible regions are shared through distributed communication to establish inter-robot LoS constraints, which are then incorporated into a multi-robot navigation framework to ensure LoS-connectivity. Moreover, we enhance the robustness of connectivity maintenance by proposing a more accurate LoS-distance metric, which further enables flexible topology optimization that eliminates redundant and effort-demanding connections. The proposed framework is evaluated through extensive multi-robot navigation and exploration tasks in both simulation and real-world experiments. Results show that it reliably maintains LoS-connectivity between robots in challenging environments cluttered with obstacles, even under large visible ranges and fragile minimal topologies, where existing methods consistently fail. Ablation studies also reveal that topology optimization boosts navigation efficiency by around $20\%$, demonstrating the framework's potential for efficient navigation under connectivity constraints.
>
---
#### [new 024] GeoReFormer: Geometry-Aware Refinement for Lane Segment Detection and Topology Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦于自动驾驶中的车道线检测与拓扑推理任务，解决传统方法未能有效编码几何与关系结构的问题。提出GeoReFormer模型，通过几何感知的查询初始化和拓扑传播提升检测精度与一致性。**

- **链接: [https://arxiv.org/pdf/2603.26018](https://arxiv.org/pdf/2603.26018)**

> **作者:** Danny Abraham; Nikhil Kamalkumar Advani; Arun Das; Nikil Dutt
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Accurate 3D lane segment detection and topology reasoning are critical for structured online map construction in autonomous driving. Recent transformer-based approaches formulate this task as query-based set prediction, yet largely inherit decoder designs originally developed for compact object detection. However, lane segments are continuous polylines embedded in directed graphs, and generic query initialization and unconstrained refinement do not explicitly encode this geometric and relational structure. We propose GeoReFormer (Geometry-aware Refinement Transformer), a unified query-based architecture that embeds geometry- and topology-aware inductive biases directly within the transformer decoder. GeoReFormer introduces data-driven geometric priors for structured query initialization, bounded coordinate-space refinement for stable polyline deformation, and per-query gated topology propagation to selectively integrate relational context. On the OpenLane-V2 benchmark, GeoReFormer achieves state-of-the-art performance with 34.5% mAP while improving topology consistency over strong transformer baselines, demonstrating the utility of explicit geometric and relational structure encoding.
>
---
#### [new 025] Drive-Through 3D Vehicle Exterior Reconstruction via Dynamic-Scene SfM and Distortion-Aware Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决复杂场景下车辆外表面高保真重建问题，通过动态场景SfM和畸变感知高斯喷射方法实现高质量模型生成。**

- **链接: [https://arxiv.org/pdf/2603.26638](https://arxiv.org/pdf/2603.26638)**

> **作者:** Nitin Kulkarni; Akhil Devarashetti; Charlie Cluss; Livio Forte; Philip Schneider; Chunming Qiao; Alina Vereshchaka
>
> **备注:** 8 pages, 7 figures, Submitted to IEEE IROS 2026 (under review)
>
> **摘要:** High-fidelity 3D reconstruction of vehicle exteriors improves buyer confidence in online automotive marketplaces, but generating these models in cluttered dealership drive-throughs presents severe technical challenges. Unlike static-scene photogrammetry, this setting features a dynamic vehicle moving against heavily cluttered, static backgrounds. This problem is further compounded by wide-angle lens distortion, specular automotive paint, and non-rigid wheel rotations that violate classical epipolar constraints. We propose an end-to-end pipeline utilizing a two-pillar camera rig. First, we resolve dynamic-scene ambiguities by coupling SAM 3 for instance segmentation with motion-gating to cleanly isolate the moving vehicle, explicitly masking out non-rigid wheels to enforce strict epipolar geometry. Second, we extract robust correspondences directly on raw, distorted 4K imagery using the RoMa v2 learned matcher guided by semantic confidence masks. Third, these matches are integrated into a rig-aware SfM optimization that utilizes CAD-derived relative pose priors to eliminate scale drift. Finally, we use a distortion-aware 3D Gaussian Splatting framework (3DGUT) coupled with a stochastic Markov Chain Monte Carlo (MCMC) densification strategy to render reflective surfaces. Evaluations on 25 real-world vehicles across 10 dealerships demonstrate that our full pipeline achieves a PSNR of 28.66 dB, an SSIM of 0.89, and an LPIPS of 0.21 on held-out views, representing a 3.85 dB improvement over standard 3D-GS, delivering inspection-grade interactive 3D models without controlled studio infrastructure.
>
---
#### [new 026] 4DRaL: Bridging 4D Radar with LiDAR for Place Recognition using Knowledge Distillation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于 place recognition 任务，旨在提升4D雷达在恶劣天气下的定位性能。通过知识蒸馏，将LiDAR模型的知识迁移至4D雷达模型，解决其数据稀疏和噪声问题。**

- **链接: [https://arxiv.org/pdf/2603.26206](https://arxiv.org/pdf/2603.26206)**

> **作者:** Ningyuan Huang; Zhiheng Li; Zheng Fang
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Place recognition is crucial for loop closure detection and global localization in robotics. Although mainstream algorithms typically rely on cameras and LiDAR, these sensors are susceptible to adverse weather conditions. Fortunately, the recently developed 4D millimeter-wave radar (4D radar) offers a promising solution for all-weather place recognition. However, the inherent noise and sparsity in 4D radar data significantly limit its performance. Thus, in this paper, we propose a novel framework called 4DRaL that leverages knowledge distillation (KD) to enhance the place recognition performance of 4D radar. Its core is to adopt a high-performance LiDAR-to-LiDAR (L2L) place recognition model as a teacher to guide the training of a 4D radar-to-4D radar (R2R) place recognition model. 4DRaL comprises three key KD modules: a local image enhancement module to handle the sparsity of raw 4D radar points, a feature distribution distillation module that ensures the student model generates more discriminative features, and a response distillation module to maintain consistency in feature space between the teacher and student models. More importantly, 4DRaL can also be trained for 4D radar-to-LiDAR (R2L) place recognition through different module configurations. Experimental results prove that 4DRaL achieves state-of-the-art performance in both R2R and R2L tasks regardless of normal or adverse weather.
>
---
#### [new 027] User Involvement in Robotic Wheelchair Development: A Decade of Limited Progress
- **分类: cs.HC; cs.RO**

- **简介: 论文属于文献综述任务，探讨机器人轮椅研发中用户参与不足的问题。研究分析了过去十年用户参与情况，发现参与度低且多集中在后期评估，提出需加强全程用户参与以提升产品 usability 和采纳率。**

- **链接: [https://arxiv.org/pdf/2603.26543](https://arxiv.org/pdf/2603.26543)**

> **作者:** Mario Andres Chavarria; Santiago Price Torrendell; Aude Billard; Samia Hurst; Sébastien Kessler; Michael Stein; Kenji Suzuki; Sophie Weerts; Diego Paez-Granados; Minerva Rivas Velarde
>
> **摘要:** Robotic wheelchairs (RWs) offer significant potential to enhance autonomy and participation for people with mobility impairments, yet many systems have failed to achieve sustained real-world adoption. This narrative literature review examined the extent and quality of end-user involvement in RW design, development, and evaluation over the past decade (2015--2025), assessed against core principles shared by major user-involvement approaches (e.g., user-/human-centered design, participatory/co-design, and inclusive design). The findings indicate that user involvement remains limited and is predominantly concentrated in late-stage evaluation rather than in early requirements definition or iterative co-design. Of the 399 records screened, only 23 studies (about 6%) met the inclusion criteria of verifiable end-user involvement, and many relied on small samples, often around ten participants, with limited justification for sample size selection, proxy users, laboratory-based validation, and non-standardized feedback methods. Research teams were largely engineering-dominated (about 89%) and geographically concentrated in high-income countries. Despite strong evidence that sustained user engagement improves usability and adoption in assistive technology, its systematic implementation in RW research remains rare. Advancing the field requires embedding participatory methodologies throughout the design lifecycle and addressing systemic barriers that constrain meaningful user involvement.
>
---
#### [new 028] Curvature-aware Expected Free Energy as an Acquisition Function for Bayesian Optimization
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于贝叶斯优化任务，旨在解决同时优化和学习函数的问题。提出基于期望自由能的获取函数，并证明其收敛性及有效性。**

- **链接: [https://arxiv.org/pdf/2603.26339](https://arxiv.org/pdf/2603.26339)**

> **作者:** Ajith Anil Meera; Wouter Kouw
>
> **备注:** under review
>
> **摘要:** We propose an Expected Free Energy-based acquisition function for Bayesian optimization to solve the joint learning and optimization problem, i.e., optimize and learn the underlying function simultaneously. We show that, under specific assumptions, Expected Free Energy reduces to Upper Confidence Bound, Lower Confidence Bound, and Expected Information Gain. We prove that Expected Free Energy has unbiased convergence guarantees for concave functions. Using the results from these derivations, we introduce a curvature-aware update law for Expected Free Energy and show its proof of concept using a system identification problem on a Van der Pol oscillator. Through rigorous simulation experiments, we show that our adaptive Expected Free Energy-based acquisition function outperforms state-of-the-art acquisition functions with the least final simple regret and error in learning the Gaussian process.
>
---
#### [new 029] DRUM: Diffusion-based Raydrop-aware Unpaired Mapping for Sim2Real LiDAR Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于Sim2Real任务，解决合成LiDAR数据到真实数据的域适应问题。提出DRUM框架，利用扩散模型生成真实特征，提升分割性能。**

- **链接: [https://arxiv.org/pdf/2603.26263](https://arxiv.org/pdf/2603.26263)**

> **作者:** Tomoya Miyawaki; Kazuto Nakashima; Yumi Iwashita; Ryo Kurazume
>
> **备注:** ICRA 2026
>
> **摘要:** LiDAR-based semantic segmentation is a key component for autonomous mobile robots, yet large-scale annotation of LiDAR point clouds is prohibitively expensive and time-consuming. Although simulators can provide labeled synthetic data, models trained on synthetic data often underperform on real-world data due to a data-level domain gap. To address this issue, we propose DRUM, a novel Sim2Real translation framework. We leverage a diffusion model pre-trained on unlabeled real-world data as a generative prior and translate synthetic data by reproducing two key measurement characteristics: reflectance intensity and raydrop noise. To improve sample fidelity, we introduce a raydrop-aware masked guidance mechanism that selectively enforces consistency with the input synthetic data while preserving realistic raydrop noise induced by the diffusion prior. Experimental results demonstrate that DRUM consistently improves Sim2Real performance across multiple representations of LiDAR data. The project page is available at this https URL.
>
---
## 更新

#### [replaced 001] CACTO-SL: Using Sobolev Learning to improve Continuous Actor-Critic with Trajectory Optimization
- **分类: cs.RO; cs.LG; math.OC**

- **简介: 该论文提出CACTO-SL，结合轨迹优化与强化学习，解决最优控制问题。通过引入Sobolev学习提升训练效率，减少计算时间并改善结果一致性。**

- **链接: [https://arxiv.org/pdf/2312.10666](https://arxiv.org/pdf/2312.10666)**

> **作者:** Elisa Alboni; Gianluigi Grandesso; Gastone Pietro Rosati Papini; Justin Carpentier; Andrea Del Prete
>
> **摘要:** Trajectory Optimization (TO) and Reinforcement Learning (RL) are powerful and complementary tools to solve optimal control problems. On the one hand, TO can efficiently compute locally-optimal solutions, but it tends to get stuck in local minima if the problem is not convex. On the other hand, RL is typically less sensitive to non-convexity, but it requires a much higher computational effort. Recently, we have proposed CACTO (Continuous Actor-Critic with Trajectory Optimization), an algorithm that uses TO to guide the exploration of an actor-critic RL algorithm. In turns, the policy encoded by the actor is used to warm-start TO, closing the loop between TO and RL. In this work, we present an extension of CACTO exploiting the idea of Sobolev learning. To make the training of the critic network faster and more data efficient, we enrich it with the gradient of the Value function, computed via a backward pass of the differential dynamic programming algorithm. Our results show that the new algorithm is more efficient than the original CACTO, reducing the number of TO episodes by a factor ranging from 3 to 10, and consequently the computation time. Moreover, we show that CACTO-SL helps TO to find better minima and to produce more consistent results.
>
---
#### [replaced 002] Before We Trust Them: Decision-Making Failures in Navigation of Foundation Models
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于导航决策任务，研究基础模型在导航中的决策失误。工作包括设计六个诊断任务，评估模型在不同信息条件下的表现，发现模型存在结构崩溃、幻觉推理等缺陷。**

- **链接: [https://arxiv.org/pdf/2601.05529](https://arxiv.org/pdf/2601.05529)**

> **作者:** Jua Han; Jaeyoon Seo; Jungbin Min; Sieun Choi; Huichan Seo; Jihie Kim; Jean Oh
>
> **备注:** Corrected author order in metadata; manuscript changed
>
> **摘要:** High success rates on navigation-related tasks do not necessarily translate into reliable decision making by foundation models. To examine this gap, we evaluate current models on six diagnostic tasks spanning three settings: reasoning under complete spatial information, reasoning under incomplete spatial information, and reasoning under safety-relevant information. Our results show that important decision-making failures can persist even when overall performance is strong, underscoring the need for failure-focused analysis to understand model limitations and guide future progress. In a path-planning setting with unknown cells, GPT-5 achieved a high success rate of 93%, yet the remaining cases still included invalid paths. We also find that newer models are not always more reliable than their predecessors. In reasoning under safety-relevant information, Gemini-2.5 Flash achieved only 67% on the challenging emergency-evacuation task, underperforming Gemini-2.0 Flash, which reached 100% under the same condition. Across all evaluations, models exhibited structural collapse, hallucinated reasoning, constraint violations, and unsafe decisions. These findings show that foundation models still exhibit substantial failures in navigation-related decision making and require fine-grained evaluation before they can be trusted. Project page: this https URL
>
---
#### [replaced 003] Ground Reaction Inertial Poser: Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GRIP方法，用于从稀疏IMU和足底压力传感器数据中重建物理合理的运动，解决人体运动捕捉任务中的动态与地面交互问题。**

- **链接: [https://arxiv.org/pdf/2603.16233](https://arxiv.org/pdf/2603.16233)**

> **作者:** Ryosuke Hori; Jyun-Ting Song; Zhengyi Luo; Jinkun Cao; Soyong Shin; Hideo Saito; Kris Kitani
>
> **摘要:** We propose Ground Reaction Inertial Poser (GRIP), a method that reconstructs physically plausible human motion using four wearable devices. Unlike conventional IMU-only approaches, GRIP combines IMU signals with foot pressure data to capture both body dynamics and ground interactions. Furthermore, rather than relying solely on kinematic estimation, GRIP uses a digital twin of a person, in the form of a synthetic humanoid in a physics simulator, to reconstruct realistic and physically plausible motion. At its core, GRIP consists of two modules: KinematicsNet, which estimates body poses and velocities from sensor data, and DynamicsNet, which controls the humanoid in the simulator using the residual between the KinematicsNet prediction and the simulated humanoid state. To enable robust training and fair evaluation, we introduce a large-scale dataset, Pressure and Inertial Sensing for Human Motion and Interaction (PRISM), that captures diverse human motions with synchronized IMUs and insole pressure sensors. Experimental results show that GRIP outperforms existing IMU-only and IMU-pressure fusion methods across all evaluated datasets, achieving higher global pose accuracy and improved physical consistency.
>
---
#### [replaced 004] ABot-PhysWorld: Interactive World Foundation Model for Robotic Manipulation with Physics Alignment
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视频生成中物理不真实的问题。提出ABot-PhysWorld模型，提升物理合理性与动作控制能力，并引入EZSbench评估基准。**

- **链接: [https://arxiv.org/pdf/2603.23376](https://arxiv.org/pdf/2603.23376)**

> **作者:** Yuzhi Chen; Ronghan Chen; Dongjie Huo; Yandan Yang; Dekang Qi; Haoyun Liu; Tong Lin; Shuang Zeng; Junjin Xiao; Xinyuan Chang; Feng Xiong; Xing Wei; Zhiheng Ma; Mu Xu
>
> **备注:** Code: this https URL
>
> **摘要:** Video-based world models offer a powerful paradigm for embodied simulation and planning, yet state-of-the-art models often generate physically implausible manipulations - such as object penetration and anti-gravity motion - due to training on generic visual data and likelihood-based objectives that ignore physical laws. We present ABot-PhysWorld, a 14B Diffusion Transformer model that generates visually realistic, physically plausible, and action-controllable videos. Built on a curated dataset of three million manipulation clips with physics-aware annotation, it uses a novel DPO-based post-training framework with decoupled discriminators to suppress unphysical behaviors while preserving visual quality. A parallel context block enables precise spatial action injection for cross-embodiment control. To better evaluate generalization, we introduce EZSbench, the first training-independent embodied zero-shot benchmark combining real and synthetic unseen robot-task-scene combinations. It employs a decoupled protocol to separately assess physical realism and action alignment. ABot-PhysWorld achieves new state-of-the-art performance on PBench and EZSbench, surpassing Veo 3.1 and Sora v2 Pro in physical plausibility and trajectory consistency. We will release EZSbench to promote standardized evaluation in embodied video generation.
>
---
#### [replaced 005] MMaDA-VLA: Large Diffusion Vision-Language-Action Model with Unified Multi-Modal Instruction and Generation
- **分类: cs.RO**

- **简介: 该论文提出MMaDA-VLA，解决机器人操作中视觉-语言-动作对齐问题，通过统一多模态生成框架提升长序列一致性与任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.25406](https://arxiv.org/pdf/2603.25406)**

> **作者:** Yang Liu; Pengxiang Ding; Tengyue Jiang; Xudong Wang; Wenxuan Song; Minghui Lin; Han Zhao; Hongyin Zhang; Zifeng Zhuang; Wei Zhao; Siteng Huang; Jinkui Shi; Donglin Wang
>
> **摘要:** Vision-Language-Action (VLA) models aim to control robots for manipulation from visual observations and natural-language instructions. However, existing hierarchical and autoregressive paradigms often introduce architectural overhead, suffer from temporal inconsistency and long-horizon error accumulation, and lack a mechanism to capture environment dynamics without extra modules. To this end, we present MMaDA-VLA, a fully native pre-trained large diffusion VLA model that unifies multi-modal understanding and generation in a single framework. Our key idea is a native discrete diffusion formulation that embeds language, images, and continuous robot controls into one discrete token space and trains a single backbone with masked token denoising to jointly generate a future goal observation and an action chunk in parallel. Iterative denoising enables global, order-free refinement, improving long-horizon consistency while grounding actions in predicted future visual outcomes without auxiliary world models. Experiments across simulation benchmarks and real-world tasks show state-of-the-art performance, achieving 98.0% average success on LIBERO and 4.78 average length on CALVIN.
>
---
#### [replaced 006] Can a Robot Walk the Robotic Dog: Triple-Zero Collaborative Navigation for Heterogeneous Multi-Agent Systems
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人协同导航任务，解决异构机器人在无训练、无先验知识、无仿真环境下的路径规划问题。提出TZPP框架，实现高效、适应性强的协作导航。**

- **链接: [https://arxiv.org/pdf/2603.21723](https://arxiv.org/pdf/2603.21723)**

> **作者:** Yaxuan Wang; Yifan Xiang; Ke Li; Xun Zhang; BoWen Ye; Zhuochen Fan; Fei Wei; Tong Yang
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** We present Triple Zero Path Planning (TZPP), a collaborative framework for heterogeneous multi-robot systems that requires zero training, zero prior knowledge, and zero simulation. TZPP employs a coordinator--explorer architecture: a humanoid robot handles task coordination, while a quadruped robot explores and identifies feasible paths using guidance from a multimodal large language model. We implement TZPP on Unitree G1 and Go2 robots and evaluate it across diverse indoor and outdoor environments, including obstacle-rich and landmark-sparse settings. Experiments show that TZPP achieves robust, human-comparable efficiency and strong adaptability to unseen scenarios. By eliminating reliance on training and simulation, TZPP offers a practical path toward real-world deployment of heterogeneous robot cooperation. Our code and video are provided at: this https URL
>
---
#### [replaced 007] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CoMo方法，用于从互联网视频中学习连续潜在运动，解决机器人学习中的信息丢失和动态捕捉不足问题。**

- **链接: [https://arxiv.org/pdf/2505.17006](https://arxiv.org/pdf/2505.17006)**

> **作者:** Jiange Yang; Yansong Shi; Haoyi Zhu; Mingyu Liu; Kaijing Ma; Yating Wang; Gangshan Wu; Tong He; Limin Wang
>
> **备注:** CVPR 2026
>
> **摘要:** Unsupervised learning of latent motion from Internet videos is crucial for robot learning. Existing discrete methods generally mitigate the shortcut learning caused by extracting excessive static backgrounds through vector quantization with a small codebook size. However, they suffer from information loss and struggle to capture more complex and fine-grained dynamics. Moreover, there is an inherent gap between the distribution of discrete latent motion and continuous robot action, which hinders the joint learning of a unified policy. We propose CoMo, which aims to learn more precise continuous latent motion from internet-scale videos. CoMo employs an early temporal difference (Td) mechanism to increase the shortcut learning difficulty and explicitly enhance motion cues. Additionally, to ensure latent motion better captures meaningful foregrounds, we further propose a temporal contrastive learning (Tcl) scheme. Specifically, positive pairs are constructed with a small future frame temporal offset, while negative pairs are formed by directly reversing the temporal direction. The proposed Td and Tcl work synergistically and effectively ensure that the latent motion focuses better on the foreground and reinforces motion cues. Critically, CoMo exhibits strong zeroshot generalization, enabling it to generate effective pseudo action labels for unseen videos. Extensive simulated and real-world experiments show that policies co-trained with CoMo pseudo action labels achieve superior performance with both diffusion and auto-regressive architectures.
>
---
#### [replaced 008] Towards Automated Chicken Deboning via Learning-based Dynamically-Adaptive 6-DoF Multi-Material Cutting
- **分类: cs.RO**

- **简介: 该论文属于机器人切割任务，旨在解决鸡肩部自动去骨问题。通过动态适应的6-DoF切割策略和强化学习，提升切割精度与安全性。**

- **链接: [https://arxiv.org/pdf/2510.15376](https://arxiv.org/pdf/2510.15376)**

> **作者:** Zhaodong Yang; Ai-Ping Hu; Harish Ravichandar
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Automating chicken shoulder deboning requires precise 6-DoF cutting through a partially occluded, deformable, multi-material joint, since contact with the bones presents serious health and safety risks. Our work makes both systems-level and algorithmic contributions to train and deploy a reactive force-feedback cutting policy that dynamically adapts a nominal trajectory and enables full 6-DoF knife control to traverse the narrow joint gap while avoiding contact with the bones. First, we introduce an open-source custom-built simulator for multi-material cutting that models coupling, fracture, and cutting forces, and supports reinforcement learning, enabling efficient training and rapid prototyping. Second, we design a reusable physical testbed to emulate the chicken shoulder: two rigid "bone" spheres with controllable pose embedded in a softer block, enabling rigorous and repeatable evaluation while preserving essential multi-material characteristics of the target problem. Third, we train and deploy a residual RL policy, with discretized force observations and domain randomization, enabling robust zero-shot sim-to-real transfer and the first demonstration of a learned policy that debones a real chicken shoulder. Our experiments in our simulator, on our physical testbed, and on real chicken shoulders show that our learned policy reliably navigates the joint gap and reduces undesired bone/cartilage contact, resulting in up to a 4x improvement over existing open-loop cutting baselines in terms of success rate and bone avoidance. Our results also illustrate the necessity of force feedback for safe and effective multi-material cutting. The project website is at this https URL.
>
---
#### [replaced 009] Control of a commercially available vehicle by a tetraplegic human using a brain-computer interface
- **分类: eess.SY; cs.NE; cs.RO**

- **简介: 论文属于脑机接口应用任务，旨在解决瘫痪患者自主驾驶问题。通过BCI系统实现车辆控制，验证了其在真实环境中的可行性和安全性。**

- **链接: [https://arxiv.org/pdf/2508.11805](https://arxiv.org/pdf/2508.11805)**

> **作者:** Xinyun Zou; Jorge Gamez; Meghna Menon; Phillip Ring; Chadwick Boulay; Likhith Chitneni; Jackson Brennecke; Shana R. Melby; Gracy Kureel; Kelsie Pejsa; Emily R. Rosario; Ausaf A. Bari; Aniruddh Ravindran; Tyson Aflalo; Spencer S. Kellis; Dimitar Filev; Florian Solzbacher; Richard A. Andersen
>
> **备注:** 50 pages, 7 figures, 1 table. 27 supplementary pages, 9 supplementary figures, 13 supplementary tables, 9 supplementary movies available as ancillary files
>
> **摘要:** Brain-computer interfaces (BCIs) read neural signals directly from the brain to infer motor planning and execution. However, the implementation of this technology has been largely limited to laboratory settings, with few real-world applications. We developed a BCI system to drive a vehicle in both simulated and real-world environments. We demonstrate that an individual with tetraplegia, implanted with intracortical BCI electrodes in the posterior parietal cortex (PPC) and the hand knob region of the motor cortex (MC), reacts at least as fast and precisely as motor intact participants. This BCI participant, living in California, could also remotely drive a Ford Mustang Mach-E vehicle in Michigan. Our teledriving tasks relied on cursor movement control for speed and steering in a closed urban test facility and through a predefined obstacle course. These two tasks serve as a proof-of-concept that takes into account the safety and feasibility of BCI-controlled driving. The final BCI system added click control for full-stop braking and thus enabled bimanual cursor-and-click control for simulated town driving with the same proficiency level as the motor intact control group through a virtual town with traffic. This first-of-its-kind implantable BCI application not only highlights the versatility and innovative potentials of BCIs but also illuminates the promising future for the development of life-changing solutions to improve independent mobility for those who suffer catastrophic neurological injury.
>
---
#### [replaced 010] HELIOS: Hierarchical Exploration for Language-Grounded Interaction in Open Scenes
- **分类: cs.RO**

- **简介: 该论文提出HELIOS，解决开放场景中语言引导的移动操作任务。通过构建层次化场景表示，融合多视角信息，提升环境感知与目标识别能力。**

- **链接: [https://arxiv.org/pdf/2509.22498](https://arxiv.org/pdf/2509.22498)**

> **作者:** Katrina Ashton; Chahyon Ku; Shrey Shah; Saumit Vedula; Tingrui Zhang; Wen Jiang; Kostas Daniilidis; Bernadette Bucher
>
> **摘要:** Language-specified mobile manipulation tasks in novel environments simultaneously face challenges interacting with a scene which is only partially observed, grounding semantic information from language instructions to the partially observed scene, and actively updating knowledge of the scene with new observations. To address these challenges, we propose HELIOS, a hierarchical scene representation and associated search objective. We construct 2D maps containing the relevant semantic and occupancy information for navigation while simultaneously actively constructing 3D Gaussian representations of task-relevant objects. We fuse observations across this multi-layered representation while explicitly modeling the multi-view consistency of the detections of each object using the Dirichlet distribution. Planning is formulated as a search problem over our hierarchical representation. We formulate an objective that jointly considers (i) exploration of unobserved or uncertain regions of the environment and (ii) information gathering from additional observations of candidate objects. This objective integrates frontier-based exploration with the expected information gain associated with improving semantic consistency of object detections. We evaluate HELIOS on the OVMM benchmark in the Habitat simulator, a pick and place benchmark in which perception is challenging due to large and complex scenes with comparatively small target objects. HELIOS achieves state-of-the-art results on OVMM. We demonstrate HELIOS performing language specified pick and place in a real world office environment on a Spot robot. Our method leverages pretrained VLMs to achieve these results in simulation and the real world without any task specific training.
>
---
#### [replaced 011] A Narwhal-Inspired Sensing-to-Control Framework for Small Fixed-Wing Aircraft
- **分类: cs.RO**

- **简介: 该论文属于飞行控制任务，旨在提升小型固定翼无人机的低速机动性。通过仿生传感器和数据驱动模型，提高气流感知与控制精度。**

- **链接: [https://arxiv.org/pdf/2510.07160](https://arxiv.org/pdf/2510.07160)**

> **作者:** Fengze Xie; Xiaozhou Fan; Jacob Schuster; Yisong Yue; Morteza Gharib
>
> **摘要:** Fixed-wing unmanned aerial vehicles (UAVs) offer endurance and efficiency but lack low-speed agility due to highly coupled dynamics. We present an end-to-end sensing-to-control pipeline that combines bio-inspired hardware, physics-informed dynamics learning, and convex control allocation. Measuring airflow on a small airframe is difficult because near-body aerodynamics, propeller slipstream, control-surface actuation, and ambient gusts distort pressure signals. Inspired by the narwhal's protruding tusk, we mount in-house multi-hole probes far upstream and complement them with sparse, carefully placed wing pressure sensors for local flow measurement. A data-driven calibration maps probe pressures to airspeed and flow angles. We then learn a control-affine dynamics model using the estimated airspeed/angles and sparse sensors. A soft left/right symmetry regularizer improves identifiability under partial observability and limits confounding between wing pressures and flaperon inputs. Desired wrenches (forces and moments) are realized by a regularized least-squares allocator that yields smooth, trimmed actuation. Wind-tunnel studies across a wide operating range show that adding wing pressures reduces force-estimation error by 25-30%, the proposed model degrades less under distribution shift (about 12% versus 44% for an unstructured baseline), and force tracking improves with smoother inputs, including a 27% reduction in normal-force RMSE versus a plain affine model and 34% versus an unstructured baseline.
>
---
#### [replaced 012] Fast-dVLA: Accelerating Discrete Diffusion VLA to Real-Time Performance
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决预训练VLA模型在微调中性能提升有限且成本高的问题。通过解耦辅助任务目标，提升模型能力并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2603.25661](https://arxiv.org/pdf/2603.25661)**

> **作者:** Wenxuan Song; Jiayi Chen; Shuai Chen; Jingbo Wang; Pengxiang Ding; Han Zhao; Yikai Qin; Xinhu Zheng; Donglin Wang; Yan Wang; Haoang Li
>
> **摘要:** This paper proposes a novel approach to address the challenge that pretrained VLA models often fail to effectively improve performance and reduce adaptation costs during standard supervised finetuning (SFT). Some advanced finetuning methods with auxiliary training objectives can improve performance and reduce the number of convergence steps. However, they typically incur significant computational overhead due to the additional losses from auxiliary tasks. To simultaneously achieve the enhanced capabilities of auxiliary training with the simplicity of standard SFT, we decouple the two objectives of auxiliary task training within the parameter space, namely, enhancing general capabilities and fitting task-specific action distributions. To deliver this goal, we only need to train the model to converge on a small-scale task set using two distinct training strategies. The difference between the resulting model parameters can then be interpreted as capability vectors provided by auxiliary tasks. These vectors are then merged with pretrained parameters to form a capability-enhanced meta model. Moreover, when standard SFT is augmented with a lightweight orthogonal regularization loss, the merged model attains performance comparable to auxiliary finetuned baselines with reduced computational overhead. Experimental results demonstrate that this approach is highly effective across diverse robot tasks. Project page: this https URL
>
---
#### [replaced 013] Robust Route Planning for Sidewalk Delivery Robots
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决 sidewalk 机器人在动态环境中的可靠路径问题。通过引入鲁棒优化和仿真，比较不同不确定性集方法，提升机器人在复杂环境下的效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2507.12067](https://arxiv.org/pdf/2507.12067)**

> **作者:** Xing Tong; Michele D. Simoni
>
> **摘要:** Sidewalk delivery robots are a promising solution for last-mile freight distribution. Yet, they operate in dynamic environments characterized by pedestrian flows and potential obstacles, which make travel times highly uncertain and can significantly affect their efficiency. This study addresses the robust route planning problem for sidewalk robots by explicitly accounting for travel time uncertainty generated through simulated interactions between robots, pedestrians, and obstacles. Robust optimization is integrated with simulation to reproduce the effect of obstacles and pedestrian flows and generate realistic travel times. Three different approaches to derive uncertainty sets are investigated, including budgeted, ellipsoidal, and support vector clustering (SVC)-based methods, together with a distributionally robust shortest path (DRSP) method based on ambiguity sets that model uncertainty in travel-time distributions. A realistic case study reproducing pedestrian patterns in Stockholm's city center is used to evaluate the efficiency of robust routing across various robot designs and environmental conditions. Results show that, when compared to a conventional shortest path (SP) method, robust routing significantly enhances operational reliability under variable sidewalk conditions. The ellipsoidal and DRSP approaches outperform the other methods in terms of average and worst-case delay. Sensitivity analyses reveal that robust approaches are higher for sidewalk delivery robots that are wider, slower, and more conservative in their navigation behaviors, especially in adverse weather and high pedestrian congestion scenarios.
>
---
#### [replaced 014] An Efficient Closed-Form Solution to Full Visual-Inertial State Initialization
- **分类: cs.RO**

- **简介: 该论文属于视觉-惯性状态初始化任务，旨在无需非线性优化即可快速准确恢复状态。通过解析解和分阶段方案，提升初始化效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.18910](https://arxiv.org/pdf/2511.18910)**

> **作者:** Samuel Cerezo; Seong Hun Lee; Javier Civera
>
> **备注:** 8 pages, 3 figures, 6 tables. Accepted to RA-L
>
> **摘要:** In this letter, we present a closed-form initialization method that recovers the full visual-inertial state without nonlinear optimization. Unlike previous approaches that rely on iterative solvers, our formulation yields analytical, easy-to-implement, and numerically stable solutions for reliable start-up. Our method builds on small-rotation and constant-velocity approximations, which keep the formulation compact while preserving the essential coupling between motion and inertial measurements. We further propose an observability-driven, two-stage initialization scheme that balances accuracy with initialization latency. Extensive experiments on the EuRoC dataset validate our assumptions: our method achieves 10-20% lower initialization error than optimization-based approaches, while using 4x shorter initialization windows and reducing computational cost by 5x.
>
---
#### [replaced 015] SOMA: Strategic Orchestration and Memory-Augmented System for Vision-Language-Action Model Robustness via In-Context Adaptation
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的鲁棒性提升任务，旨在解决VLA模型在分布外任务中的感知噪声和环境变化问题。通过SOMA系统实现无需参数微调的在线适应。**

- **链接: [https://arxiv.org/pdf/2603.24060](https://arxiv.org/pdf/2603.24060)**

> **作者:** Zhuoran Li; Zhiyang Li; Kaijun Zhou; Jinyu Gu
>
> **备注:** 9 pages, 16 figures, 3 table
>
> **摘要:** Despite the promise of Vision-Language-Action (VLA) models as generalist robotic controllers, their robustness against perceptual noise and environmental variations in out-of-distribution (OOD) tasks remains fundamentally limited by the absence of long-term memory, causal failure attribution, and dynamic intervention capability. To address this, we propose SOMA, a Strategic Orchestration and Memory-Augmented System that upgrades frozen VLA policies for robust in-context adaptation without parameter fine-tuning. Specifically, SOMA operates through an online pipeline of contrastive Dual-Memory Retrieval-Augmented Generation (RAG), an Attribution-Driven Large-Language-Model (LLM) Orchestrator, and extensible Model Context Protocol (MCP) interventions, while an offline Memory Consolidation module continuously distills the execution traces into reliable priors. Experimental evaluations across three backbone models (pi0, pi0.5, and SmolVLA) on LIBERO-PRO and our proposed LIBERO-SOMA benchmarks demonstrate that SOMA achieves an average absolute success rate gain of 56.6%. This includes a significant absolute improvement of 89.1% in long-horizon task chaining. Project page and source code are available at: this https URL.
>
---
#### [replaced 016] VG-Mapping: Variation-aware Density Control for Online 3D Gaussian Mapping in Semi-static Scenes
- **分类: cs.RO**

- **简介: 该论文属于在线3D地图重建任务，旨在解决半静态场景中地图更新效率与准确性问题。提出VG-Mapping方法，通过变化感知的密度控制策略提升地图质量与更新效率。**

- **链接: [https://arxiv.org/pdf/2510.09962](https://arxiv.org/pdf/2510.09962)**

> **作者:** Yicheng He; Jingwen Yu; Guangcheng Chen; Hong Zhang
>
> **摘要:** Maintaining an up-to-date map that accurately reflects recent changes in the environment is crucial, especially for robots that repeatedly traverse the same space. Failing to promptly update the changed regions can degrade map quality, resulting in poor localization, inefficient operations, and even lost robots. 3D Gaussian Splatting (3DGS) has recently seen widespread adoption in online map reconstruction due to its dense, differentiable, and photorealistic properties, yet accurately and efficiently updating the regions of change remains a challenge. In this paper, we propose VG-Mapping, a novel online 3DGS-based mapping system tailored for such semi-static scenes. Our approach introduces a variation-aware density control strategy that decouples Gaussian density regulation from optimization. Specifically, we identify regions with variation to guide initialization and pruning, which avoids the use of stale information in defining the starting point for the subsequent optimization. Furthermore, to address the absence of public benchmarks for this task, we construct a RGB-D dataset comprising both synthetic and real-world semi-static environments. Experimental results demonstrate that our method substantially improves the rendering quality and map update efficiency in semi-static scenes. The code and dataset are available at this https URL.
>
---
#### [replaced 017] Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多智能体驾驶仿真任务，旨在提升行为模型的效率与鲁棒性。通过优化场景表示和交互建模，实现更高效的训练与推理。**

- **链接: [https://arxiv.org/pdf/2512.05812](https://arxiv.org/pdf/2512.05812)**

> **作者:** Fabian Konstantinidis; Moritz Sackmann; Ulrich Hofmann; Christoph Stiller
>
> **备注:** This is the author's accepted version of a paper to appear in the IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.
>
---
#### [replaced 018] Out-of-Sight Embodied Agents: Multimodal Tracking, Sensor Fusion, and Trajectory Forecasting
- **分类: cs.CV; cs.LG; cs.MA; cs.MM; cs.RO**

- **简介: 该论文属于轨迹预测任务，解决出视线目标的轨迹预测与去噪问题。通过视觉-定位对齐模块，提升自动驾驶等场景下的感知鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.15219](https://arxiv.org/pdf/2509.15219)**

> **作者:** Haichao Zhang; Yi Xu; Yun Fu
>
> **备注:** Published in IEEE Transactions on Pattern Analysis and Machine Intelligence (Early Access), pp. 1-14, March 23, 2026
>
> **摘要:** Trajectory prediction is a fundamental problem in computer vision, vision-language-action models, world models, and autonomous systems, with broad impact on autonomous driving, robotics, and surveillance. However, most existing methods assume complete and clean observations, and therefore do not adequately handle out-of-sight agents or noisy sensing signals caused by limited camera coverage, occlusions, and the absence of ground-truth denoised trajectories. These challenges raise safety concerns and reduce robustness in real-world deployment. In this extended study, we introduce major improvements to Out-of-Sight Trajectory (OST), a task for predicting noise-free visual trajectories of out-of-sight objects from noisy sensor observations. Building on our prior work, we expand Out-of-Sight Trajectory Prediction (OOSTraj) from pedestrians to both pedestrians and vehicles, increasing its relevance to autonomous driving, robotics, and surveillance. Our improved Vision-Positioning Denoising Module exploits camera calibration to establish vision-position correspondence, mitigating the lack of direct visual cues and enabling effective unsupervised denoising of noisy sensor signals. Extensive experiments on the Vi-Fi and JRDB datasets show that our method achieves state-of-the-art results for both trajectory denoising and trajectory prediction, with clear gains over prior baselines. We also compare with classical denoising methods, including Kalman filtering, and adapt recent trajectory prediction models to this setting, establishing a stronger benchmark. To the best of our knowledge, this is the first work to use vision-positioning projection to denoise noisy sensor trajectories of out-of-sight agents, opening new directions for future research.
>
---
#### [replaced 019] Task Tokens: A Flexible Approach to Adapting Behavior Foundation Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决BFM在特定任务中需复杂提示工程的问题。通过引入“Task Tokens”，利用强化学习训练任务编码器，提升模型性能并保持其泛化能力。**

- **链接: [https://arxiv.org/pdf/2503.22886](https://arxiv.org/pdf/2503.22886)**

> **作者:** Ron Vainshtein; Zohar Rimon; Shie Mannor; Chen Tessler
>
> **摘要:** Recent advancements in imitation learning have led to transformer-based behavior foundation models (BFMs) that enable multi-modal, human-like control for humanoid agents. While excelling at zero-shot generation of robust behaviors, BFMs often require meticulous prompt engineering for specific tasks, potentially yielding suboptimal results. We introduce "Task Tokens", a method to effectively tailor BFMs to specific tasks while preserving their flexibility. Our approach leverages the transformer architecture of BFMs to learn a new task-specific encoder through reinforcement learning, keeping the original BFM frozen. This allows incorporation of user-defined priors, balancing reward design and prompt engineering. By training a task encoder to map observations to tokens, used as additional BFM inputs, we guide performance improvement while maintaining the model's diverse control characteristics. We demonstrate Task Tokens' efficacy across various tasks, including out-of-distribution scenarios, and show their compatibility with other prompting modalities. Our results suggest that Task Tokens offer a promising approach for adapting BFMs to specific control tasks while retaining their generalization capabilities.
>
---
#### [replaced 020] Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Wanderland框架，解决开放世界具身AI的可重复评估问题。通过高保真模拟和几何精准重建，提升导航与视图合成性能，推动相关研究。**

- **链接: [https://arxiv.org/pdf/2511.20620](https://arxiv.org/pdf/2511.20620)**

> **作者:** Xinhao Liu; Jiaqi Li; Youming Deng; Ruxin Chen; Yingjia Zhang; Yifei Ma; Li Guo; Yiming Li; Jing Zhang; Chen Feng
>
> **备注:** CVPR 2026
>
> **摘要:** Reproducible closed-loop evaluation remains a major bottleneck in Embodied AI such as visual navigation. A promising path forward is high-fidelity simulation that combines photorealistic sensor rendering with geometrically grounded interaction in complex, open-world urban environments. Although recent video-3DGS methods ease open-world scene capturing, they are still unsuitable for benchmarking due to large visual and geometric sim-to-real gaps. To address these challenges, we introduce Wanderland, a real-to-sim framework that features multi-sensor capture, reliable reconstruction, accurate geometry, and robust view synthesis. Using this pipeline, we curate a diverse dataset of indoor-outdoor urban scenes and systematically demonstrate how image-only pipelines scale poorly, how geometry quality impacts novel view synthesis, and how all of these adversely affect navigation policy learning and evaluation reliability. Beyond serving as a trusted testbed for embodied navigation, Wanderland's rich raw sensor data further allows benchmarking of 3D reconstruction and novel view synthesis models. Our work establishes a new foundation for reproducible research in open-world embodied AI. Project website is at this https URL.
>
---
#### [replaced 021] The Competence Shadow: Theory and Bounds of AI Assistance in Safety Engineering
- **分类: cs.AI; cs.ET; cs.HC; cs.RO; cs.SE**

- **简介: 该论文属于安全工程与AI协作任务，旨在解决AI辅助可能引入系统性盲点的问题。提出“能力阴影”概念，分析协作结构对安全分析的影响，强调协作设计的重要性。**

- **链接: [https://arxiv.org/pdf/2603.25197](https://arxiv.org/pdf/2603.25197)**

> **作者:** Umair Siddique
>
> **备注:** 8 Pages, 3 Figures, 2 table
>
> **摘要:** As AI assistants become integrated into safety engineering workflows for Physical AI systems, a critical question emerges: does AI assistance improve safety analysis quality, or introduce systematic blind spots that surface only through post-deployment incidents? This paper develops a formal framework for AI assistance in safety analysis. We first establish why safety engineering resists benchmark-driven evaluation: safety competence is irreducibly multidimensional, constrained by context-dependent correctness, inherent incompleteness, and legitimate expert disagreement. We formalize this through a five-dimensional competence framework capturing domain knowledge, standards expertise, operational experience, contextual understanding, and judgment. We introduce the competence shadow: the systematic narrowing of human reasoning induced by AI-generated safety analysis. The shadow is not what the AI presents, but what it prevents from being considered. We formalize four canonical human-AI collaboration structures and derive closed-form performance bounds, demonstrating that the competence shadow compounds multiplicatively to produce degradation far exceeding naive additive estimates. The central finding is that AI assistance in safety engineering is a collaboration design problem, not a software procurement decision. The same tool degrades or improves analysis quality depending entirely on how it is used. We derive non-degradation conditions for shadow-resistant workflows and call for a shift from tool qualification toward workflow qualification for trustworthy Physical AI.
>
---
#### [replaced 022] IRIS-SLAM: Unified Geo-Instance Representations for Robust Semantic Localization and Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出IRIS-SLAM，解决语义定位与建图中的几何-实例统一表示问题，提升地图一致性与回环检测可靠性。**

- **链接: [https://arxiv.org/pdf/2602.18709](https://arxiv.org/pdf/2602.18709)**

> **作者:** Tingyang Xiao; Liu Liu; Wei Feng; Zhengyu Zou; Xiaolin Zhou; Wei Sui; Hao Li; Dingwen Zhang; Zhizhong Su
>
> **摘要:** Geometry foundation models have significantly advanced dense geometric SLAM, yet existing systems often lack deep semantic understanding and robust loop closure capabilities. Meanwhile, contemporary semantic mapping approaches are frequently hindered by decoupled architectures and fragile data association. We propose IRIS-SLAM, a novel RGB semantic SLAM system that leverages unified geometric-instance representations derived from an instance-extended foundation model. By extending a geometry foundation model to concurrently predict dense geometry and cross-view consistent instance embeddings, we enable a semantic-synergized association mechanism and instance-guided loop closure detection. Our approach effectively utilizes viewpoint-agnostic semantic anchors to bridge the gap between geometric reconstruction and open-vocabulary mapping. Experimental results demonstrate that IRIS-SLAM significantly outperforms state-of-the-art methods, particularly in map consistency and wide-baseline loop closure reliability.
>
---
