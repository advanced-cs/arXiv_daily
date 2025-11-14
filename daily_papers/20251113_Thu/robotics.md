# 机器人 cs.RO

- **最新发布 34 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] CoRL-MPPI: Enhancing MPPI With Learnable Behaviours For Efficient And Provably-Safe Multi-Robot Collision Avoidance
- **分类: cs.RO; cs.MA**

- **简介: 论文提出CoRL-MPPI，将强化学习策略嵌入MPPI框架，以提升多机器人分布式避障的采样效率与安全性，兼顾理论保障与导航性能，解决传统MPPI随机采样低效的问题。**

- **链接: [https://arxiv.org/pdf/2511.09331v1](https://arxiv.org/pdf/2511.09331v1)**

> **作者:** Stepan Dergachev; Artem Pshenitsyn; Aleksandr Panov; Alexey Skrynnik; Konstantin Yakovlev
>
> **备注:** The manuscript includes 9 pages, 4 figures, and 1 table
>
> **摘要:** Decentralized collision avoidance remains a core challenge for scalable multi-robot systems. One of the promising approaches to tackle this problem is Model Predictive Path Integral (MPPI) -- a framework that is naturally suited to handle any robot motion model and provides strong theoretical guarantees. Still, in practice MPPI-based controller may provide suboptimal trajectories as its performance relies heavily on uninformed random sampling. In this work, we introduce CoRL-MPPI, a novel fusion of Cooperative Reinforcement Learning and MPPI to address this limitation. We train an action policy (approximated as deep neural network) in simulation that learns local cooperative collision avoidance behaviors. This learned policy is then embedded into the MPPI framework to guide its sampling distribution, biasing it towards more intelligent and cooperative actions. Notably, CoRL-MPPI preserves all the theoretical guarantees of regular MPPI. We evaluate our approach in dense, dynamic simulation environments against state-of-the-art baselines, including ORCA, BVC, and a multi-agent MPPI implementation. Our results demonstrate that CoRL-MPPI significantly improves navigation efficiency (measured by success rate and makespan) and safety, enabling agile and robust multi-robot navigation.
>
---
#### [new 002] UMIGen: A Unified Framework for Egocentric Point Cloud Generation and Cross-Embodiment Robotic Imitation Learning
- **分类: cs.RO**

- **简介: 论文提出UMIGen，解决机器人模仿学习中数据采集成本高、缺乏3D几何信息的问题。通过手持设备Cloud-UMI采集点云动作对，并结合可见性优化生成符合视角的3D观测，实现跨机器人形态的高效数据生成与迁移。**

- **链接: [https://arxiv.org/pdf/2511.09302v1](https://arxiv.org/pdf/2511.09302v1)**

> **作者:** Yan Huang; Shoujie Li; Xingting Li; Wenbo Ding
>
> **摘要:** Data-driven robotic learning faces an obvious dilemma: robust policies demand large-scale, high-quality demonstration data, yet collecting such data remains a major challenge owing to high operational costs, dependence on specialized hardware, and the limited spatial generalization capability of current methods. The Universal Manipulation Interface (UMI) relaxes the strict hardware requirements for data collection, but it is restricted to capturing only RGB images of a scene and omits the 3D geometric information on which many tasks rely. Inspired by DemoGen, we propose UMIGen, a unified framework that consists of two key components: (1) Cloud-UMI, a handheld data collection device that requires no visual SLAM and simultaneously records point cloud observation-action pairs; and (2) a visibility-aware optimization mechanism that extends the DemoGen pipeline to egocentric 3D observations by generating only points within the camera's field of view. These two components enable efficient data generation that aligns with real egocentric observations and can be directly transferred across different robot embodiments without any post-processing. Experiments in both simulated and real-world settings demonstrate that UMIGen supports strong cross-embodiment generalization and accelerates data collection in diverse manipulation tasks.
>
---
#### [new 003] APEX: Action Priors Enable Efficient Exploration for Robust Motion Tracking on Legged Robots
- **分类: cs.RO**

- **简介: APEX提出一种无需参考数据的强化学习框架，通过衰减动作先验引导机器人探索，实现高效、鲁棒的多地形运动跟踪，提升样本效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.09091v1](https://arxiv.org/pdf/2511.09091v1)**

> **作者:** Shivam Sood; Laukik Nakhwa; Sun Ge; Yuhong Cao; Jin Cheng; Fatemah Zargarbashi; Taerim Yoon; Sungjoon Choi; Stelian Coros; Guillaume Sartoretti
>
> **摘要:** Learning natural, animal-like locomotion from demonstrations has become a core paradigm in legged robotics. Despite the recent advancements in motion tracking, most existing methods demand extensive tuning and rely on reference data during deployment, limiting adaptability. We present APEX (Action Priors enable Efficient Exploration), a plug-and-play extension to state-of-the-art motion tracking algorithms that eliminates any dependence on reference data during deployment, improves sample efficiency, and reduces parameter tuning effort. APEX integrates expert demonstrations directly into reinforcement learning (RL) by incorporating decaying action priors, which initially bias exploration toward expert demonstrations but gradually allow the policy to explore independently. This is combined with a multi-critic framework that balances task performance with motion style. Moreover, APEX enables a single policy to learn diverse motions and transfer reference-like styles across different terrains and velocities, while remaining robust to variations in reward design. We validate the effectiveness of our method through extensive experiments in both simulation and on a Unitree Go2 robot. By leveraging demonstrations to guide exploration during RL training, without imposing explicit bias toward them, APEX enables legged robots to learn with greater stability, efficiency, and generalization. We believe this approach paves the way for guidance-driven RL to boost natural skill acquisition in a wide array of robotic tasks, from locomotion to manipulation. Website and code: https://marmotlab.github.io/APEX/.
>
---
#### [new 004] SMF-VO: Direct Ego-Motion Estimation via Sparse Motion Fields
- **分类: cs.RO; cs.CV**

- **简介: 论文提出SMF-VO，一种轻量级视觉里程计方法，直接从稀疏光流估计相机瞬时速度，规避传统位姿估计与地图优化，提升实时性。适用于资源受限设备，实现超100 FPS的CPU推理。**

- **链接: [https://arxiv.org/pdf/2511.09072v1](https://arxiv.org/pdf/2511.09072v1)**

> **作者:** Sangheon Yang; Yeongin Yoon; Hong Mo Jung; Jongwoo Lim
>
> **摘要:** Traditional Visual Odometry (VO) and Visual Inertial Odometry (VIO) methods rely on a 'pose-centric' paradigm, which computes absolute camera poses from the local map thus requires large-scale landmark maintenance and continuous map optimization. This approach is computationally expensive, limiting their real-time performance on resource-constrained devices. To overcome these limitations, we introduce Sparse Motion Field Visual Odometry (SMF-VO), a lightweight, 'motion-centric' framework. Our approach directly estimates instantaneous linear and angular velocity from sparse optical flow, bypassing the need for explicit pose estimation or expensive landmark tracking. We also employed a generalized 3D ray-based motion field formulation that works accurately with various camera models, including wide-field-of-view lenses. SMF-VO demonstrates superior efficiency and competitive accuracy on benchmark datasets, achieving over 100 FPS on a Raspberry Pi 5 using only a CPU. Our work establishes a scalable and efficient alternative to conventional methods, making it highly suitable for mobile robotics and wearable devices.
>
---
#### [new 005] SPIDER: Scalable Physics-Informed Dexterous Retargeting
- **分类: cs.RO; cs.CV**

- **简介: SPIDER提出一种物理驱动的灵巧重定向框架，将人类运动数据转化为机器人可执行的动态可行轨迹，解决人机本体差异与数据稀缺问题，实现跨9种机器人、6类数据的高效规模化生成，显著提升策略学习效率。**

- **链接: [https://arxiv.org/pdf/2511.09484v1](https://arxiv.org/pdf/2511.09484v1)**

> **作者:** Chaoyi Pan; Changhao Wang; Haozhi Qi; Zixi Liu; Homanga Bharadhwaj; Akash Sharma; Tingfan Wu; Guanya Shi; Jitendra Malik; Francois Hogan
>
> **备注:** Project website: https://jc-bao.github.io/spider-project/
>
> **摘要:** Learning dexterous and agile policy for humanoid and dexterous hand control requires large-scale demonstrations, but collecting robot-specific data is prohibitively expensive. In contrast, abundant human motion data is readily available from motion capture, videos, and virtual reality, which could help address the data scarcity problem. However, due to the embodiment gap and missing dynamic information like force and torque, these demonstrations cannot be directly executed on robots. To bridge this gap, we propose Scalable Physics-Informed DExterous Retargeting (SPIDER), a physics-based retargeting framework to transform and augment kinematic-only human demonstrations to dynamically feasible robot trajectories at scale. Our key insight is that human demonstrations should provide global task structure and objective, while large-scale physics-based sampling with curriculum-style virtual contact guidance should refine trajectories to ensure dynamical feasibility and correct contact sequences. SPIDER scales across diverse 9 humanoid/dexterous hand embodiments and 6 datasets, improving success rates by 18% compared to standard sampling, while being 10X faster than reinforcement learning (RL) baselines, and enabling the generation of a 2.4M frames dynamic-feasible robot dataset for policy learning. As a universal physics-based retargeting method, SPIDER can work with diverse quality data and generate diverse and high-quality data to enable efficient policy learning with methods like RL.
>
---
#### [new 006] Data Assessment for Embodied Intelligence
- **分类: cs.RO**

- **简介: 该论文面向具身智能，解决数据集多样性与可学习性评估难题，提出统一多模态表征与无训练的可学习性度量方法，实现对数据集信息量与学习效率的高效、可解释评估。**

- **链接: [https://arxiv.org/pdf/2511.09119v1](https://arxiv.org/pdf/2511.09119v1)**

> **作者:** Jiahao Xiao; Bowen Yan; Jianbo Zhang; Jia Wang; Chunyi Li; Zhengxue Cheng; Guangtao Zhai
>
> **摘要:** In embodied intelligence, datasets play a pivotal role, serving as both a knowledge repository and a conduit for information transfer. The two most critical attributes of a dataset are the amount of information it provides and how easily this information can be learned by models. However, the multimodal nature of embodied data makes evaluating these properties particularly challenging. Prior work has largely focused on diversity, typically counting tasks and scenes or evaluating isolated modalities, which fails to provide a comprehensive picture of dataset diversity. On the other hand, the learnability of datasets has received little attention and is usually assessed post-hoc through model training, an expensive, time-consuming process that also lacks interpretability, offering little guidance on how to improve a dataset. In this work, we address both challenges by introducing two principled, data-driven tools. First, we construct a unified multimodal representation for each data sample and, based on it, propose diversity entropy, a continuous measure that characterizes the amount of information contained in a dataset. Second, we introduce the first interpretable, data-driven algorithm to efficiently quantify dataset learnability without training, enabling researchers to assess a dataset's learnability immediately upon its release. We validate our algorithm on both simulated and real-world embodied datasets, demonstrating that it yields faithful, actionable insights that enable researchers to jointly improve diversity and learnability. We hope this work provides a foundation for designing higher-quality datasets that advance the development of embodied intelligence.
>
---
#### [new 007] WMPO: World Model-based Policy Optimization for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 论文提出WMPO，一种基于世界模型的视觉-语言-动作策略优化方法，解决VLA模型依赖专家数据、样本效率低的问题，通过像素级仿真预测实现无真实交互的在线强化学习，提升性能与自修正能力。**

- **链接: [https://arxiv.org/pdf/2511.09515v1](https://arxiv.org/pdf/2511.09515v1)**

> **作者:** Fangqi Zhu; Zhengyang Yan; Zicong Hong; Quanxin Shou; Xiao Ma; Song Guo
>
> **备注:** project website: https://wm-po.github.io
>
> **摘要:** Vision-Language-Action (VLA) models have shown strong potential for general-purpose robotic manipulation, but their reliance on expert demonstrations limits their ability to learn from failures and perform self-corrections. Reinforcement learning (RL) addresses these through self-improving interactions with the physical environment, but suffers from high sample complexity on real robots. We introduce World-Model-based Policy Optimization (WMPO), a principled framework for on-policy VLA RL without interacting with the real environment. In contrast to widely used latent world models, WMPO focuses on pixel-based predictions that align the "imagined" trajectories with the VLA features pretrained with web-scale images. Crucially, WMPO enables the policy to perform on-policy GRPO that provides stronger performance than the often-used off-policy methods. Extensive experiments in both simulation and real-robot settings demonstrate that WMPO (i) substantially improves sample efficiency, (ii) achieves stronger overall performance, (iii) exhibits emergent behaviors such as self-correction, and (iv) demonstrates robust generalization and lifelong learning capabilities.
>
---
#### [new 008] UniMM-V2X: MoE-Enhanced Multi-Level Fusion for End-to-End Cooperative Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: UniMM-V2X提出一种端到端协同自动驾驶框架，通过多级融合与MoE架构，实现感知、预测与规划的协同优化，解决单体智能感知受限与任务脱节问题，在DAIR-V2X上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.09013v1](https://arxiv.org/pdf/2511.09013v1)**

> **作者:** Ziyi Song; Chen Xia; Chenbing Wang; Haibao Yu; Sheng Zhou; Zhisheng Niu
>
> **摘要:** Autonomous driving holds transformative potential but remains fundamentally constrained by the limited perception and isolated decision-making with standalone intelligence. While recent multi-agent approaches introduce cooperation, they often focus merely on perception-level tasks, overlooking the alignment with downstream planning and control, or fall short in leveraging the full capacity of the recent emerging end-to-end autonomous driving. In this paper, we present UniMM-V2X, a novel end-to-end multi-agent framework that enables hierarchical cooperation across perception, prediction, and planning. At the core of our framework is a multi-level fusion strategy that unifies perception and prediction cooperation, allowing agents to share queries and reason cooperatively for consistent and safe decision-making. To adapt to diverse downstream tasks and further enhance the quality of multi-level fusion, we incorporate a Mixture-of-Experts (MoE) architecture to dynamically enhance the BEV representations. We further extend MoE into the decoder to better capture diverse motion patterns. Extensive experiments on the DAIR-V2X dataset demonstrate our approach achieves state-of-the-art (SOTA) performance with a 39.7% improvement in perception accuracy, a 7.2% reduction in prediction error, and a 33.2% improvement in planning performance compared with UniV2X, showcasing the strength of our MoE-enhanced multi-level cooperative paradigm.
>
---
#### [new 009] MAP-VLA: Memory-Augmented Prompting for Vision-Language-Action Model in Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向机器人长时程操作任务，解决预训练VLA模型缺乏记忆导致的性能不足问题，提出MAP-VLA框架，通过可学习的演示记忆提示实现即插即用的动态增强，显著提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.09516v1](https://arxiv.org/pdf/2511.09516v1)**

> **作者:** Runhao Li; Wenkai Guo; Zhenyu Wu; Changyuan Wang; Haoyuan Deng; Zhenyu Weng; Yap-Peng Tan; Ziwei Wang
>
> **摘要:** Pre-trained Vision-Language-Action (VLA) models have achieved remarkable success in improving robustness and generalization for end-to-end robotic manipulation. However, these models struggle with long-horizon tasks due to their lack of memory and reliance solely on immediate sensory inputs. To address this limitation, we propose Memory-Augmented Prompting for Vision-Language-Action model (MAP-VLA), a novel framework that empowers pre-trained VLA models with demonstration-derived memory prompts to augment action generation for long-horizon robotic manipulation tasks. To achieve this, MAP-VLA first constructs a memory library from historical demonstrations, where each memory unit captures information about a specific stage of a task. These memory units are implemented as learnable soft prompts optimized through prompt tuning. Then, during real-time task execution, MAP-VLA retrieves relevant memory through trajectory similarity matching and dynamically integrates it into the VLA model for augmented action generation. Importantly, this prompt tuning and retrieval augmentation approach operates as a plug-and-play module for a frozen VLA model, offering a lightweight and flexible solution to improve task performance. Experimental results show that MAP-VLA delivers up to 7.0% absolute performance gains in the simulation benchmark and 25.0% on real robot evaluations for long-horizon tasks, surpassing the current state-of-the-art methods.
>
---
#### [new 010] IFG: Internet-Scale Guidance for Functional Grasping Generation
- **分类: cs.RO; cs.AI; cs.CV; cs.GR; cs.LG**

- **简介: 该论文提出IFG框架，结合互联网规模视觉模型的语义理解与仿真驱动的力闭合抓取生成，实现无需人工标注的实时高精度3D抓取规划，解决机器人抓取中语义与几何精度脱节的问题。**

- **链接: [https://arxiv.org/pdf/2511.09558v1](https://arxiv.org/pdf/2511.09558v1)**

> **作者:** Ray Muxin Liu; Mingxuan Li; Kenneth Shaw; Deepak Pathak
>
> **备注:** Website at https://ifgrasping.github.io/
>
> **摘要:** Large Vision Models trained on internet-scale data have demonstrated strong capabilities in segmenting and semantically understanding object parts, even in cluttered, crowded scenes. However, while these models can direct a robot toward the general region of an object, they lack the geometric understanding required to precisely control dexterous robotic hands for 3D grasping. To overcome this, our key insight is to leverage simulation with a force-closure grasping generation pipeline that understands local geometries of the hand and object in the scene. Because this pipeline is slow and requires ground-truth observations, the resulting data is distilled into a diffusion model that operates in real-time on camera point clouds. By combining the global semantic understanding of internet-scale models with the geometric precision of a simulation-based locally-aware force-closure, \our achieves high-performance semantic grasping without any manually collected training data. For visualizations of this please visit our website at https://ifgrasping.github.io/
>
---
#### [new 011] RGMP: Recurrent Geometric-prior Multimodal Policy for Generalizable Humanoid Robot Manipulation
- **分类: cs.RO**

- **简介: 论文提出RGMP框架，面向通用人形机器人操作任务，解决数据效率低与几何推理缺失问题。通过几何先验技能选择与自适应高斯递归网络，实现稀疏演示下的高效跨域泛化，成功率87%，数据效率提升5倍。**

- **链接: [https://arxiv.org/pdf/2511.09141v1](https://arxiv.org/pdf/2511.09141v1)**

> **作者:** Xuetao Li; Wenke Huang; Nengyuan Pan; Kaiyan Zhao; Songhua Yang; Yiming Wang; Mengde Li; Mang Ye; Jifeng Xuan; Miao Li
>
> **摘要:** Humanoid robots exhibit significant potential in executing diverse human-level skills. However, current research predominantly relies on data-driven approaches that necessitate extensive training datasets to achieve robust multimodal decision-making capabilities and generalizable visuomotor control. These methods raise concerns due to the neglect of geometric reasoning in unseen scenarios and the inefficient modeling of robot-target relationships within the training data, resulting in significant waste of training resources. To address these limitations, we present the Recurrent Geometric-prior Multimodal Policy (RGMP), an end-to-end framework that unifies geometric-semantic skill reasoning with data-efficient visuomotor control. For perception capabilities, we propose the Geometric-prior Skill Selector, which infuses geometric inductive biases into a vision language model, producing adaptive skill sequences for unseen scenes with minimal spatial common sense tuning. To achieve data-efficient robotic motion synthesis, we introduce the Adaptive Recursive Gaussian Network, which parameterizes robot-object interactions as a compact hierarchy of Gaussian processes that recursively encode multi-scale spatial relationships, yielding dexterous, data-efficient motion synthesis even from sparse demonstrations. Evaluated on both our humanoid robot and desktop dual-arm robot, the RGMP framework achieves 87% task success in generalization tests and exhibits 5x greater data efficiency than the state-of-the-art model. This performance underscores its superior cross-domain generalization, enabled by geometric-semantic reasoning and recursive-Gaussion adaptation.
>
---
#### [new 012] MirrorLimb: Implementing hand pose acquisition and robot teleoperation based on RealMirror
- **分类: cs.RO; cs.HC**

- **简介: 论文提出MirrorLimb系统，基于PICO与RealMirror实现低成本、实时手部位姿获取与机器人遥操作，解决传统视觉追踪成本高、精度不足问题，支持多类型末端执行器控制，助力VLA数据集构建与上肢机器人研究。**

- **链接: [https://arxiv.org/pdf/2511.08865v1](https://arxiv.org/pdf/2511.08865v1)**

> **作者:** Cong Tai; Hansheng Wu; Haixu Long; Zhengbin Long; Zhaoyu Zheng; Haodong Xiang; Tao Shen
>
> **摘要:** In this work, we present a PICO-based robot remote operating framework that enables low-cost, real-time acquisition of hand motion and pose data, outperforming mainstream visual tracking and motion capture solutions in terms of cost-effectiveness. The framework is natively compatible with the RealMirror ecosystem, offering ready-to-use functionality for stable and precise robotic trajectory recording within the Isaac simulation environment, thereby facilitating the construction of Vision-Language-Action (VLA) datasets. Additionally, the system supports real-time teleoperation of a variety of end-effector-equipped robots, including dexterous hands and robotic grippers. This work aims to lower the technical barriers in the study of upper-limb robotic manipulation, thereby accelerating advancements in VLA-related research.
>
---
#### [new 013] A Shared Control Framework for Mobile Robots with Planning-Level Intention Prediction
- **分类: cs.RO**

- **简介: 该论文提出一种基于规划层意图预测的移动机器人共享控制框架，通过深度强化学习联合建模意图域预测与路径重规划，利用Voronoi算法生成仿真训练数据，无需真实人机数据，显著降低操作负荷并提升安全性。**

- **链接: [https://arxiv.org/pdf/2511.08912v1](https://arxiv.org/pdf/2511.08912v1)**

> **作者:** Jinyu Zhang; Lijun Han; Feng Jian; Lingxi Zhang; Hesheng Wang
>
> **摘要:** In mobile robot shared control, effectively understanding human motion intention is critical for seamless human-robot collaboration. This paper presents a novel shared control framework featuring planning-level intention prediction. A path replanning algorithm is designed to adjust the robot's desired trajectory according to inferred human intentions. To represent future motion intentions, we introduce the concept of an intention domain, which serves as a constraint for path replanning. The intention-domain prediction and path replanning problems are jointly formulated as a Markov Decision Process and solved through deep reinforcement learning. In addition, a Voronoi-based human trajectory generation algorithm is developed, allowing the model to be trained entirely in simulation without human participation or demonstration data. Extensive simulations and real-world user studies demonstrate that the proposed method significantly reduces operator workload and enhances safety, without compromising task efficiency compared with existing assistive teleoperation approaches.
>
---
#### [new 014] SpatialActor: Exploring Disentangled Spatial Representations for Robust Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 论文提出SpatialActor，面向机器人操作任务，解决深度噪声下语义与几何纠缠导致的定位不准问题，通过解耦语义与几何表示，融合多源几何信息与低层空间线索，提升操作鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.09555v1](https://arxiv.org/pdf/2511.09555v1)**

> **作者:** Hao Shi; Bin Xie; Yingfei Liu; Yang Yue; Tiancai Wang; Haoqiang Fan; Xiangyu Zhang; Gao Huang
>
> **备注:** AAAI 2026 Oral | Project Page: https://shihao1895.github.io/SpatialActor
>
> **摘要:** Robotic manipulation requires precise spatial understanding to interact with objects in the real world. Point-based methods suffer from sparse sampling, leading to the loss of fine-grained semantics. Image-based methods typically feed RGB and depth into 2D backbones pre-trained on 3D auxiliary tasks, but their entangled semantics and geometry are sensitive to inherent depth noise in real-world that disrupts semantic understanding. Moreover, these methods focus on high-level geometry while overlooking low-level spatial cues essential for precise interaction. We propose SpatialActor, a disentangled framework for robust robotic manipulation that explicitly decouples semantics and geometry. The Semantic-guided Geometric Module adaptively fuses two complementary geometry from noisy depth and semantic-guided expert priors. Also, a Spatial Transformer leverages low-level spatial cues for accurate 2D-3D mapping and enables interaction among spatial features. We evaluate SpatialActor on multiple simulation and real-world scenarios across 50+ tasks. It achieves state-of-the-art performance with 87.4% on RLBench and improves by 13.9% to 19.4% under varying noisy conditions, showing strong robustness. Moreover, it significantly enhances few-shot generalization to new tasks and maintains robustness under various spatial perturbations. Project Page: https://shihao1895.github.io/SpatialActor
>
---
#### [new 015] Decoupling Torque and Stiffness: A Unified Modeling and Control Framework for Antagonistic Artificial Muscles
- **分类: cs.RO**

- **简介: 该论文提出一种统一建模与控制框架，实现人工肌肉的扭矩与刚度独立调控，解决动态接触中解耦失效问题。通过解析逆动力学与生物启发坐标，显著提升响应速度与交互稳定性，适用于人机安全交互。**

- **链接: [https://arxiv.org/pdf/2511.09104v1](https://arxiv.org/pdf/2511.09104v1)**

> **作者:** Amirhossein Kazemipour; Robert K. Katzschmann
>
> **摘要:** Antagonistic soft actuators built from artificial muscles (PAMs, HASELs, DEAs) promise plant-level torque-stiffness decoupling, yet existing controllers for soft muscles struggle to maintain independent control through dynamic contact transients. We present a unified framework enabling independent torque and stiffness commands in real-time for diverse soft actuator types. Our unified force law captures diverse soft muscle physics in a single model with sub-ms computation, while our cascaded controller with analytical inverse dynamics maintains decoupling despite model errors and disturbances. Using co-contraction/bias coordinates, the controller independently modulates torque via bias and stiffness via co-contraction-replicating biological impedance strategies. Simulation-based validation through contact experiments demonstrates maintained independence: 200x faster settling on soft surfaces, 81% force reduction on rigid surfaces, and stable interaction vs 22-54% stability for fixed policies. This framework provides a foundation for enabling musculoskeletal antagonistic systems to execute adaptive impedance control for safe human-robot interaction.
>
---
#### [new 016] XPRESS: X-Band Radar Place Recognition via Elliptical Scan Shaping
- **分类: cs.RO**

- **简介: 该论文提出XPRESS算法，解决X波段雷达分辨率低、信息少导致的自主导航难题，通过椭圆扫描整形与密度规则优化，实现仅依赖X波段雷达的地点识别，显著提升检索鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.08863v1](https://arxiv.org/pdf/2511.08863v1)**

> **作者:** Hyesu Jang; Wooseong Yang; Ayoung Kim; Dongje Lee; Hanguen Kim
>
> **备注:** 9 pages, 9 figures, Published in IEEE RA-L
>
> **摘要:** X-band radar serves as the primary sensor on maritime vessels, however, its application in autonomous navigation has been limited due to low sensor resolution and insufficient information content. To enable X-band radar-only autonomous navigation in maritime environments, this paper proposes a place recognition algorithm specifically tailored for X-band radar, incorporating an object density-based rule for efficient candidate selection and intentional degradation of radar detections to achieve robust retrieval performance. The proposed algorithm was evaluated on both public maritime radar datasets and our own collected dataset, and its performance was compared against state-of-the-art radar place recognition methods. An ablation study was conducted to assess the algorithm's performance sensitivity with respect to key parameters.
>
---
#### [new 017] LODESTAR: Degeneracy-Aware LiDAR-Inertial Odometry with Adaptive Schmidt-Kalman Filter and Data Exploitation
- **分类: cs.RO**

- **简介: LODESTAR提出一种抗退化LiDAR-惯性里程计方法，针对走廊、高空等场景中测量稀疏与不平衡问题，设计退化感知的自适应Schmidt-Kalman滤波与数据利用模块，提升状态估计的精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.09142v1](https://arxiv.org/pdf/2511.09142v1)**

> **作者:** Eungchang Mason Lee; Kevin Christiansen Marsim; Hyun Myung
>
> **备注:** 8 pages, 5 figures, 6 tables, accepted for the publication in IEEE Robotics and Automation Letters
>
> **摘要:** LiDAR-inertial odometry (LIO) has been widely used in robotics due to its high accuracy. However, its performance degrades in degenerate environments, such as long corridors and high-altitude flights, where LiDAR measurements are imbalanced or sparse, leading to ill-posed state estimation. In this letter, we present LODESTAR, a novel LIO method that addresses these degeneracies through two key modules: degeneracy-aware adaptive Schmidt-Kalman filter (DA-ASKF) and degeneracy-aware data exploitation (DA-DE). DA-ASKF employs a sliding window to utilize past states and measurements as additional constraints. Specifically, it introduces degeneracy-aware sliding modes that adaptively classify states as active or fixed based on their degeneracy level. Using Schmidt-Kalman update, it partially optimizes active states while preserving fixed states. These fixed states influence the update of active states via their covariances, serving as reference anchors--akin to a lodestar. Additionally, DA-DE prunes less-informative measurements from active states and selectively exploits measurements from fixed states, based on their localizability contribution and the condition number of the Jacobian matrix. Consequently, DA-ASKF enables degeneracy-aware constrained optimization and mitigates measurement sparsity, while DA-DE addresses measurement imbalance. Experimental results show that LODESTAR outperforms existing LiDAR-based odometry methods and degeneracy-aware modules in terms of accuracy and robustness under various degenerate conditions.
>
---
#### [new 018] A Quantum Tunneling and Bio-Phototactic Driven Enhanced Dwarf Mongoose Optimizer for UAV Trajectory Planning and Engineering Problem
- **分类: cs.RO; math.OC**

- **简介: 该论文提出EDMO算法，用于解决无人机三维轨迹规划与工程优化问题，通过量子隧穿、生物趋光性和正交学习策略提升收敛性与多样性，显著优于14种主流算法。**

- **链接: [https://arxiv.org/pdf/2511.09020v1](https://arxiv.org/pdf/2511.09020v1)**

> **作者:** Mingyang Yu; Haorui Yang; Kangning An; Xinjian Wei; Xiaoxuan Xu; Jing Xu
>
> **摘要:** With the widespread adoption of unmanned aerial vehicles (UAV), effective path planning has become increasingly important. Although traditional search methods have been extensively applied, metaheuristic algorithms have gained popularity due to their efficiency and problem-specific heuristics. However, challenges such as premature convergence and lack of solution diversity still hinder their performance in complex scenarios. To address these issues, this paper proposes an Enhanced Multi-Strategy Dwarf Mongoose Optimization (EDMO) algorithm, tailored for three-dimensional UAV trajectory planning in dynamic and obstacle-rich environments. EDMO integrates three novel strategies: (1) a Dynamic Quantum Tunneling Optimization Strategy (DQTOS) to enable particles to probabilistically escape local optima; (2) a Bio-phototactic Dynamic Focusing Search Strategy (BDFSS) inspired by microbial phototaxis for adaptive local refinement; and (3) an Orthogonal Lens Opposition-Based Learning (OLOBL) strategy to enhance global exploration through structured dimensional recombination. EDMO is benchmarked on 39 standard test functions from CEC2017 and CEC2020, outperforming 14 advanced algorithms in convergence speed, robustness, and optimization accuracy. Furthermore, real-world validations on UAV three-dimensional path planning and three engineering design tasks confirm its practical applicability and effectiveness in field robotics missions requiring intelligent, adaptive, and time-efficient planning.
>
---
#### [new 019] Intuitive Programming, Adaptive Task Planning, and Dynamic Role Allocation in Human-Robot Collaboration
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文综述人机协作中实现直观交互的关键技术，解决人类被动、机器人缺乏适应性的问题，构建从多模态指令理解、自适应任务规划到动态角色分配与反馈闭环的完整交互框架。**

- **链接: [https://arxiv.org/pdf/2511.08732v1](https://arxiv.org/pdf/2511.08732v1)**

> **作者:** Marta Lagomarsino; Elena Merlo; Andrea Pupa; Timo Birr; Franziska Krebs; Cristian Secchi; Tamim Asfour; Arash Ajoudani
>
> **备注:** Published in the Annual Review of Control, Robotics, and Autonomous Systems, Volume 9; copyright 2026 the author(s), CC BY 4.0, https://www.annualreviews.org
>
> **摘要:** Remarkable capabilities have been achieved by robotics and AI, mastering complex tasks and environments. Yet, humans often remain passive observers, fascinated but uncertain how to engage. Robots, in turn, cannot reach their full potential in human-populated environments without effectively modeling human states and intentions and adapting their behavior. To achieve a synergistic human-robot collaboration (HRC), a continuous information flow should be established: humans must intuitively communicate instructions, share expertise, and express needs. In parallel, robots must clearly convey their internal state and forthcoming actions to keep users informed, comfortable, and in control. This review identifies and connects key components enabling intuitive information exchange and skill transfer between humans and robots. We examine the full interaction pipeline: from the human-to-robot communication bridge translating multimodal inputs into robot-understandable representations, through adaptive planning and role allocation, to the control layer and feedback mechanisms to close the loop. Finally, we highlight trends and promising directions toward more adaptive, accessible HRC.
>
---
#### [new 020] CENIC: Convex Error-controlled Numerical Integration for Contact
- **分类: cs.RO; cs.CE; physics.comp-ph**

- **简介: CENIC是一种面向接触动力学的连续时间数值积分器，融合凸时步与误差控制技术，解决传统离散模拟器在精度与效率间的权衡难题，实现媲美MuJoCo等工具的实时速度与收敛保证。**

- **链接: [https://arxiv.org/pdf/2511.08771v1](https://arxiv.org/pdf/2511.08771v1)**

> **作者:** Vince Kurtz; Alejandro Castro
>
> **备注:** 18 pages with 19 figures. Submitted to IEEE Transactions on Robotics (T-RO). The supplemental video is available publicly at https://www.youtube.com/watch?v=9ZZ15MfCgtI
>
> **摘要:** State-of-the-art robotics simulators operate in discrete time. This requires users to choose a time step, which is both critical and challenging: large steps can produce non-physical artifacts, while small steps force the simulation to run slowly. Continuous-time error-controlled integration avoids such issues by automatically adjusting the time step to achieve a desired accuracy. But existing error-controlled integrators struggle with the stiff dynamics of contact, and cannot meet the speed and scalability requirements of modern robotics workflows. We introduce CENIC, a new continuous-time integrator that brings together recent advances in convex time-stepping and error-controlled integration, inheriting benefits from both continuous integration and discrete time-stepping. CENIC runs at fast real-time rates comparable to discrete-time robotics simulators like MuJoCo, Drake and Isaac Sim, while also providing guarantees on accuracy and convergence.
>
---
#### [new 021] Dual-Arm Whole-Body Motion Planning: Leveraging Overlapping Kinematic Chains
- **分类: cs.RO**

- **简介: 该论文针对双臂机器人高维构型空间中的实时运动规划难题，提出利用共用关节约束构建动态路网，高效联合搜索左右臂-躯干链，显著降低维度诅咒，在19自由度机器人上实现0.4秒平均规划时间与99.9%成功率。**

- **链接: [https://arxiv.org/pdf/2511.08778v1](https://arxiv.org/pdf/2511.08778v1)**

> **作者:** Richard Cheng; Peter Werner; Carolyn Matl
>
> **备注:** Published in Humanoids 2025
>
> **摘要:** High degree-of-freedom dual-arm robots are becoming increasingly common due to their morphology enabling them to operate effectively in human environments. However, motion planning in real-time within unknown, changing environments remains a challenge for such robots due to the high dimensionality of the configuration space and the complex collision-avoidance constraints that must be obeyed. In this work, we propose a novel way to alleviate the curse of dimensionality by leveraging the structure imposed by shared joints (e.g. torso joints) in a dual-arm robot. First, we build two dynamic roadmaps (DRM) for each kinematic chain (i.e. left arm + torso, right arm + torso) with specific structure induced by the shared joints. Then, we show that we can leverage this structure to efficiently search through the composition of the two roadmaps and largely sidestep the curse of dimensionality. Finally, we run several experiments in a real-world grocery store with this motion planner on a 19 DoF mobile manipulation robot executing a grocery fulfillment task, achieving 0.4s average planning times with 99.9% success rate across more than 2000 motion plans.
>
---
#### [new 022] Unveiling the Impact of Data and Model Scaling on High-Level Control for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文面向人形机器人高层控制，解决数据稀缺与表示学习难题，提出Humanoid-Union数据集与SCHUR框架，利用人类视频生成大规模机器人运动数据，实现数据与模型扩展下的高质量运动生成与文本-动作对齐。**

- **链接: [https://arxiv.org/pdf/2511.09241v1](https://arxiv.org/pdf/2511.09241v1)**

> **作者:** Yuxi Wei; Zirui Wang; Kangning Yin; Yue Hu; Jingbo Wang; Siheng Chen
>
> **摘要:** Data scaling has long remained a critical bottleneck in robot learning. For humanoid robots, human videos and motion data are abundant and widely available, offering a free and large-scale data source. Besides, the semantics related to the motions enable modality alignment and high-level robot control learning. However, how to effectively mine raw video, extract robot-learnable representations, and leverage them for scalable learning remains an open problem. To address this, we introduce Humanoid-Union, a large-scale dataset generated through an autonomous pipeline, comprising over 260 hours of diverse, high-quality humanoid robot motion data with semantic annotations derived from human motion videos. The dataset can be further expanded via the same pipeline. Building on this data resource, we propose SCHUR, a scalable learning framework designed to explore the impact of large-scale data on high-level control in humanoid robots. Experimental results demonstrate that SCHUR achieves high robot motion generation quality and strong text-motion alignment under data and model scaling, with 37\% reconstruction improvement under MPJPE and 25\% alignment improvement under FID comparing with previous methods. Its effectiveness is further validated through deployment in real-world humanoid robot.
>
---
#### [new 023] Think, Remember, Navigate: Zero-Shot Object-Goal Navigation with VLM-Powered Reasoning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出一种零样本目标导航方法，利用VLM作为主动规划者，通过结构化推理、动作历史反馈和地图融合，提升机器人在未知环境中的导航效率与逻辑性。**

- **链接: [https://arxiv.org/pdf/2511.08942v1](https://arxiv.org/pdf/2511.08942v1)**

> **作者:** Mobin Habibpour; Fatemeh Afghah
>
> **摘要:** While Vision-Language Models (VLMs) are set to transform robotic navigation, existing methods often underutilize their reasoning capabilities. To unlock the full potential of VLMs in robotics, we shift their role from passive observers to active strategists in the navigation process. Our framework outsources high-level planning to a VLM, which leverages its contextual understanding to guide a frontier-based exploration agent. This intelligent guidance is achieved through a trio of techniques: structured chain-of-thought prompting that elicits logical, step-by-step reasoning; dynamic inclusion of the agent's recent action history to prevent getting stuck in loops; and a novel capability that enables the VLM to interpret top-down obstacle maps alongside first-person views, thereby enhancing spatial awareness. When tested on challenging benchmarks like HM3D, Gibson, and MP3D, this method produces exceptionally direct and logical trajectories, marking a substantial improvement in navigation efficiency over existing approaches and charting a path toward more capable embodied agents.
>
---
#### [new 024] ATOM-CBF: Adaptive Safe Perception-Based Control under Out-of-Distribution Measurements
- **分类: cs.RO; eess.SY**

- **简介: 论文提出ATOM-CBF，面向高维感知输入的安全控制任务，解决OoD测量导致的感知不确定性威胁系统安全问题，通过自适应误差边界与安全滤波器实现实时自适应安全控制，无需真值或分布先验。**

- **链接: [https://arxiv.org/pdf/2511.08741v1](https://arxiv.org/pdf/2511.08741v1)**

> **作者:** Kai S. Yun; Navid Azizan
>
> **摘要:** Ensuring the safety of real-world systems is challenging, especially when they rely on learned perception modules to infer the system state from high-dimensional sensor data. These perception modules are vulnerable to epistemic uncertainty, often failing when encountering out-of-distribution (OoD) measurements not seen during training. To address this gap, we introduce ATOM-CBF (Adaptive-To-OoD-Measurement Control Barrier Function), a novel safe control framework that explicitly computes and adapts to the epistemic uncertainty from OoD measurements, without the need for ground-truth labels or information on distribution shifts. Our approach features two key components: (1) an OoD-aware adaptive perception error margin and (2) a safety filter that integrates this adaptive error margin, enabling the filter to adjust its conservatism in real-time. We provide empirical validation in simulations, demonstrating that ATOM-CBF maintains safety for an F1Tenth vehicle with LiDAR scans and a quadruped robot with RGB images.
>
---
#### [new 025] Practical and Performant Enhancements for Maximization of Algebraic Connectivity
- **分类: cs.RO; cs.LG**

- **简介: 该论文面向图结构在线状态估计，解决MAC算法计算慢、需人工设边的问题。提出高效求解器、优化步长策略与自动连通保障机制，显著提升算法速度与实用性，支持实时应用。**

- **链接: [https://arxiv.org/pdf/2511.08694v1](https://arxiv.org/pdf/2511.08694v1)**

> **作者:** Leonard Jung; Alan Papalia; Kevin Doherty; Michael Everett
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** Long-term state estimation over graphs remains challenging as current graph estimation methods scale poorly on large, long-term graphs. To address this, our work advances a current state-of-the-art graph sparsification algorithm, maximizing algebraic connectivity (MAC). MAC is a sparsification method that preserves estimation performance by maximizing the algebraic connectivity, a spectral graph property that is directly connected to the estimation error. Unfortunately, MAC remains computationally prohibitive for online use and requires users to manually pre-specify a connectivity-preserving edge set. Our contributions close these gaps along three complementary fronts: we develop a specialized solver for algebraic connectivity that yields an average 2x runtime speedup; we investigate advanced step size strategies for MAC's optimization procedure to enhance both convergence speed and solution quality; and we propose automatic schemes that guarantee graph connectivity without requiring manual specification of edges. Together, these contributions make MAC more scalable, reliable, and suitable for real-time estimation applications.
>
---
#### [new 026] Expand Your SCOPE: Semantic Cognition over Potential-Based Exploration for Embodied Visual Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对具身视觉导航任务，提出SCOPE框架，利用视觉-语言模型挖掘前沿信息构建时空势能图，结合自反思机制提升决策可靠性，实现零样本下更精准的长程导航。**

- **链接: [https://arxiv.org/pdf/2511.08935v1](https://arxiv.org/pdf/2511.08935v1)**

> **作者:** Ningnan Wang; Weihuang Chen; Liming Chen; Haoxuan Ji; Zhongyu Guo; Xuchong Zhang; Hongbin Sun
>
> **摘要:** Embodied visual navigation remains a challenging task, as agents must explore unknown environments with limited knowledge. Existing zero-shot studies have shown that incorporating memory mechanisms to support goal-directed behavior can improve long-horizon planning performance. However, they overlook visual frontier boundaries, which fundamentally dictate future trajectories and observations, and fall short of inferring the relationship between partial visual observations and navigation goals. In this paper, we propose Semantic Cognition Over Potential-based Exploration (SCOPE), a zero-shot framework that explicitly leverages frontier information to drive potential-based exploration, enabling more informed and goal-relevant decisions. SCOPE estimates exploration potential with a Vision-Language Model and organizes it into a spatio-temporal potential graph, capturing boundary dynamics to support long-horizon planning. In addition, SCOPE incorporates a self-reconsideration mechanism that revisits and refines prior decisions, enhancing reliability and reducing overconfident errors. Experimental results on two diverse embodied navigation tasks show that SCOPE outperforms state-of-the-art baselines by 4.6\% in accuracy. Further analysis demonstrates that its core components lead to improved calibration, stronger generalization, and higher decision quality.
>
---
#### [new 027] Low-cost Multi-agent Fleet for Acoustic Cooperative Localization Research
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出低成本可配置水下多智能体平台CoUGARs，解决水下协同定位研究成本高、难部署问题，基于商用与3D打印部件构建系统，集成声学定位与通信软件，并在仿真与实湖试验中验证。**

- **链接: [https://arxiv.org/pdf/2511.08822v1](https://arxiv.org/pdf/2511.08822v1)**

> **作者:** Nelson Durrant; Braden Meyers; Matthew McMurray; Clayton Smith; Brighton Anderson; Tristan Hodgins; Kalliyan Velasco; Joshua G. Mangelson
>
> **摘要:** Real-world underwater testing for multi-agent autonomy presents substantial financial and engineering challenges. In this work, we introduce the Configurable Underwater Group of Autonomous Robots (CoUGARs) as a low-cost, configurable autonomous-underwater-vehicle (AUV) platform for multi-agent autonomy research. The base design costs less than $3,000 USD (as of May 2025) and is based on commercially-available and 3D-printed parts, enabling quick customization for various sensor payloads and configurations. Our current expanded model is equipped with a doppler velocity log (DVL) and ultra-short-baseline (USBL) acoustic array/transducer to support research on acoustic-based cooperative localization. State estimation, navigation, and acoustic communications software has been developed and deployed using a containerized software stack and is tightly integrated with the HoloOcean simulator. The system was tested both in simulation and via in-situ field trials in Utah lakes and reservoirs.
>
---
#### [new 028] D-AWSIM: Distributed Autonomous Driving Simulator for Dynamic Map Generation Framework
- **分类: cs.RO**

- **简介: 论文提出D-AWSIM分布式自动驾驶仿真平台，解决大规模交通与传感器仿真成本高、单机容量不足问题，通过分布式架构支持动态地图生成，实现高效仿真与算法验证。**

- **链接: [https://arxiv.org/pdf/2511.09080v1](https://arxiv.org/pdf/2511.09080v1)**

> **作者:** Shunsuke Ito; Chaoran Zhao; Ryo Okamura; Takuya Azumi
>
> **备注:** 9 pages. This version includes minor lstlisting configuration adjustments for successful compilation. No changes to content or layout. Originally published at Euromicro DSD 2025
>
> **摘要:** Autonomous driving systems have achieved significant advances, and full autonomy within defined operational design domains near practical deployment. Expanding these domains requires addressing safety assurance under diverse conditions. Information sharing through vehicle-to-vehicle and vehicle-to-infrastructure communication, enabled by a Dynamic Map platform built from vehicle and roadside sensor data, offers a promising solution. Real-world experiments with numerous infrastructure sensors incur high costs and regulatory challenges. Conventional single-host simulators lack the capacity for large-scale urban traffic scenarios. This paper proposes D-AWSIM, a distributed simulator that partitions its workload across multiple machines to support the simulation of extensive sensor deployment and dense traffic environments. A Dynamic Map generation framework on D-AWSIM enables researchers to explore information-sharing strategies without relying on physical testbeds. The evaluation shows that D-AWSIM increases throughput for vehicle count and LiDAR sensor processing substantially compared to a single-machine setup. Integration with Autoware demonstrates applicability for autonomous driving research.
>
---
#### [new 029] A Multi-Drone Multi-View Dataset and Deep Learning Framework for Pedestrian Detection and Tracking
- **分类: cs.CV; cs.IT; cs.LG; cs.RO; eess.IV**

- **简介: 该论文提出MATRIX数据集与深度学习框架，解决动态多无人机视角下行人检测与跟踪难题，通过BEV特征融合与实时校准，在复杂遮挡环境中实现约90%精度，显著优于静态相机方法。**

- **链接: [https://arxiv.org/pdf/2511.08615v1](https://arxiv.org/pdf/2511.08615v1)**

> **作者:** Kosta Dakic; Kanchana Thilakarathna; Rodrigo N. Calheiros; Teng Joon Lim
>
> **备注:** Introduction of the MATRIX Dataset, featuring synchronized footage from eight drones in an urban environment with comprehensive annotations for detection and tracking, available at https://github.com/KostaDakic/MATRIX/tree/main
>
> **摘要:** Multi-drone surveillance systems offer enhanced coverage and robustness for pedestrian tracking, yet existing approaches struggle with dynamic camera positions and complex occlusions. This paper introduces MATRIX (Multi-Aerial TRacking In compleX environments), a comprehensive dataset featuring synchronized footage from eight drones with continuously changing positions, and a novel deep learning framework for multi-view detection and tracking. Unlike existing datasets that rely on static cameras or limited drone coverage, MATRIX provides a challenging scenario with 40 pedestrians and a significant architectural obstruction in an urban environment. Our framework addresses the unique challenges of dynamic drone-based surveillance through real-time camera calibration, feature-based image registration, and multi-view feature fusion in bird's-eye-view (BEV) representation. Experimental results demonstrate that while static camera methods maintain over 90\% detection and tracking precision and accuracy metrics in a simplified MATRIX environment without an obstruction, 10 pedestrians and a much smaller observational area, their performance significantly degrades in the complex environment. Our proposed approach maintains robust performance with $\sim$90\% detection and tracking accuracy, as well as successfully tracks $\sim$80\% of trajectories under challenging conditions. Transfer learning experiments reveal strong generalization capabilities, with the pretrained model achieving much higher detection and tracking accuracy performance compared to training the model from scratch. Additionally, systematic camera dropout experiments reveal graceful performance degradation, demonstrating practical robustness for real-world deployments where camera failures may occur. The MATRIX dataset and framework provide essential benchmarks for advancing dynamic multi-view surveillance systems.
>
---
#### [new 030] HOTFLoc++: End-to-End Hierarchical LiDAR Place Recognition, Re-Ranking, and 6-DoF Metric Localisation in Forests
- **分类: cs.CV; cs.RO**

- **简介: HOTFLoc++提出一种端到端LiDAR定位框架，用于森林等复杂环境中的场景识别、重排序与6自由度精确定位。通过八叉树变换器提取多尺度特征，结合可学习几何验证模块，显著提升鲁棒性与效率，Recall@1达90.7%以上。**

- **链接: [https://arxiv.org/pdf/2511.09170v1](https://arxiv.org/pdf/2511.09170v1)**

> **作者:** Ethan Griffiths; Maryam Haghighat; Simon Denman; Clinton Fookes; Milad Ramezani
>
> **备注:** 9 pages, 2 figures. Submitted to RA-L
>
> **摘要:** This article presents HOTFLoc++, an end-to-end framework for LiDAR place recognition, re-ranking, and 6-DoF metric localisation in forests. Leveraging an octree-based transformer, our approach extracts hierarchical local descriptors at multiple granularities to increase robustness to clutter, self-similarity, and viewpoint changes in challenging scenarios, including ground-to-ground and ground-to-aerial in forest and urban environments. We propose a learnable multi-scale geometric verification module to reduce re-ranking failures in the presence of degraded single-scale correspondences. Our coarse-to-fine registration approach achieves comparable or lower localisation errors to baselines, with runtime improvements of two orders of magnitude over RANSAC for dense point clouds. Experimental results on public datasets show the superiority of our approach compared to state-of-the-art methods, achieving an average Recall@1 of 90.7% on CS-Wild-Places: an improvement of 29.6 percentage points over baselines, while maintaining high performance on single-source benchmarks with an average Recall@1 of 91.7% and 96.0% on Wild-Places and MulRan, respectively. Our method achieves under 2 m and 5 degrees error for 97.2% of 6-DoF registration attempts, with our multi-scale re-ranking module reducing localisation errors by ~2$\times$ on average. The code will be available upon acceptance.
>
---
#### [new 031] Information-Driven Fault Detection and Identification for Multi-Agent Spacecraft Systems: Collaborative On-Orbit Inspection Mission
- **分类: eess.SY; cs.AI; cs.MA; cs.RO**

- **简介: 该论文针对多航天器协同在轨巡检任务，提出一种信息驱动的故障检测与识别框架，通过统一成本函数联动任务分配与局部决策，实现传感器、执行器与状态估计器的故障定位与分类，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.08752v1](https://arxiv.org/pdf/2511.08752v1)**

> **作者:** Akshita Gupta; Arna Bhardwaj; Yashwanth Kumar Nakka; Changrak Choi; Amir Rahmani
>
> **备注:** AIAA Book Chapter (accepted)
>
> **摘要:** This work presents a global-to-local, task-aware fault detection and identification (FDI) framework for multi-spacecraft systems conducting collaborative inspection missions in low Earth orbit. The inspection task is represented by a global information-driven cost functional that integrates the sensor model, spacecraft poses, and mission-level information-gain objectives. This formulation links guidance, control, and FDI by using the same cost function to drive both global task allocation and local sensing or motion decisions. Fault detection is achieved through comparisons between expected and observed task metrics, while higher-order cost-gradient measures enable the identification of faults among sensors, actuators, and state estimators. An adaptive thresholding mechanism captures the time-varying inspection geometry and dynamic mission conditions. Simulation results for representative multi-spacecraft inspection scenarios demonstrate the reliability of fault localization and classification under uncertainty, providing a unified, information-driven foundation for resilient autonomous inspection architectures.
>
---
#### [new 032] Diffusion Policies with Value-Conditional Optimization for Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文面向离线强化学习，解决因OOD动作导致的价值高估问题。提出DIVO方法，利用优势值加权扩散模型，精准引导高回报动作生成，在保持保守性的同时提升探索效率，显著提升D4RL基准性能。**

- **链接: [https://arxiv.org/pdf/2511.08922v1](https://arxiv.org/pdf/2511.08922v1)**

> **作者:** Yunchang Ma; Tenglong Liu; Yixing Lan; Xin Yin; Changxin Zhang; Xinglong Zhang; Xin Xu
>
> **备注:** IROS 2025
>
> **摘要:** In offline reinforcement learning, value overestimation caused by out-of-distribution (OOD) actions significantly limits policy performance. Recently, diffusion models have been leveraged for their strong distribution-matching capabilities, enforcing conservatism through behavior policy constraints. However, existing methods often apply indiscriminate regularization to redundant actions in low-quality datasets, resulting in excessive conservatism and an imbalance between the expressiveness and efficiency of diffusion modeling. To address these issues, we propose DIffusion policies with Value-conditional Optimization (DIVO), a novel approach that leverages diffusion models to generate high-quality, broadly covered in-distribution state-action samples while facilitating efficient policy improvement. Specifically, DIVO introduces a binary-weighted mechanism that utilizes the advantage values of actions in the offline dataset to guide diffusion model training. This enables a more precise alignment with the dataset's distribution while selectively expanding the boundaries of high-advantage actions. During policy improvement, DIVO dynamically filters high-return-potential actions from the diffusion model, effectively guiding the learned policy toward better performance. This approach achieves a critical balance between conservatism and explorability in offline RL. We evaluate DIVO on the D4RL benchmark and compare it against state-of-the-art baselines. Empirical results demonstrate that DIVO achieves superior performance, delivering significant improvements in average returns across locomotion tasks and outperforming existing methods in the challenging AntMaze domain, where sparse rewards pose a major difficulty.
>
---
#### [new 033] Argus: Resilience-Oriented Safety Assurance Framework for End-to-End ADSs
- **分类: cs.AI; cs.RO; cs.SE**

- **简介: 论文提出Argus框架，面向端到端自动驾驶系统（ADS），解决其在复杂场景中安全性不足的问题。通过实时监控轨迹并动态接管控制，显著提升系统韧性，减少64.38%违规，提升驾驶评分达150.30%。**

- **链接: [https://arxiv.org/pdf/2511.09032v1](https://arxiv.org/pdf/2511.09032v1)**

> **作者:** Dingji Wang; You Lu; Bihuan Chen; Shuo Hao; Haowen Jiang; Yifan Tian; Xin Peng
>
> **备注:** The paper has been accepted by the 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025
>
> **摘要:** End-to-end autonomous driving systems (ADSs), with their strong capabilities in environmental perception and generalizable driving decisions, are attracting growing attention from both academia and industry. However, once deployed on public roads, ADSs are inevitably exposed to diverse driving hazards that may compromise safety and degrade system performance. This raises a strong demand for resilience of ADSs, particularly the capability to continuously monitor driving hazards and adaptively respond to potential safety violations, which is crucial for maintaining robust driving behaviors in complex driving scenarios. To bridge this gap, we propose a runtime resilience-oriented framework, Argus, to mitigate the driving hazards, thus preventing potential safety violations and improving the driving performance of an ADS. Argus continuously monitors the trajectories generated by the ADS for potential hazards and, whenever the EGO vehicle is deemed unsafe, seamlessly takes control through a hazard mitigator. We integrate Argus with three state-of-the-art end-to-end ADSs, i.e., TCP, UniAD and VAD. Our evaluation has demonstrated that Argus effectively and efficiently enhances the resilience of ADSs, improving the driving score of the ADS by up to 150.30% on average, and preventing up to 64.38% of the violations, with little additional time overhead.
>
---
#### [new 034] Evader-Agnostic Team-Based Pursuit Strategies in Partially-Observable Environments
- **分类: cs.MA; cs.RO**

- **简介: 该论文研究部分可观测城市环境中双无人机协同追捕未知行为逃逸无人机的问题，提出一种基于有界理性的神经符号算法，离线训练多级策略，在线分类对手并响应，显著提升追捕成功率。**

- **链接: [https://arxiv.org/pdf/2511.05812v1](https://arxiv.org/pdf/2511.05812v1)**

> **作者:** Addison Kalanther; Daniel Bostwick; Chinmay Maheshwari; Shankar Sastry
>
> **摘要:** We consider a scenario where a team of two unmanned aerial vehicles (UAVs) pursue an evader UAV within an urban environment. Each agent has a limited view of their environment where buildings can occlude their field-of-view. Additionally, the pursuer team is agnostic about the evader in terms of its initial and final location, and the behavior of the evader. Consequently, the team needs to gather information by searching the environment and then track it to eventually intercept. To solve this multi-player, partially-observable, pursuit-evasion game, we develop a two-phase neuro-symbolic algorithm centered around the principle of bounded rationality. First, we devise an offline approach using deep reinforcement learning to progressively train adversarial policies for the pursuer team against fictitious evaders. This creates $k$-levels of rationality for each agent in preparation for the online phase. Then, we employ an online classification algorithm to determine a "best guess" of our current opponent from the set of iteratively-trained strategic agents and apply the best player response. Using this schema, we improved average performance when facing a random evader in our environment.
>
---
## 更新

#### [replaced 001] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2509.25822v4](https://arxiv.org/pdf/2509.25822v4)**

> **作者:** Jing Wang; Weiting Peng; Jing Tang; Zeyu Gong; Xihua Wang; Bo Tao; Li Cheng
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Existing imitation learning methods decouple perception and action, which overlooks the causal reciprocity between sensory representations and action execution that humans naturally leverage for adaptive behaviors. To bridge this gap, we introduce Action-Guided Diffusion Policy (DP-AG), a unified representation learning that explicitly models a dynamic interplay between perception and action through probabilistic latent dynamics. DP-AG encodes latent observations into a Gaussian posterior via variational inference and evolves them using an action-guided SDE, where the Vector-Jacobian Product (VJP) of the diffusion policy's noise predictions serves as a structured stochastic force driving latent updates. To promote bidirectional learning between perception and action, we introduce a cycle-consistent contrastive loss that organizes the gradient flow of the noise predictor into a coherent perception-action loop, enforcing mutually consistent transitions in both latent updates and action refinements. Theoretically, we derive a variational lower bound for the action-guided SDE, and prove that the contrastive objective enhances continuity in both latent and action trajectories. Empirically, DP-AG significantly outperforms state-of-the-art methods across simulation benchmarks and real-world UR5 manipulation tasks. As a result, our DP-AG offers a promising step toward bridging biological adaptability and artificial policy learning.
>
---
#### [replaced 002] Evolutionary Policy Optimization
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2503.19037v3](https://arxiv.org/pdf/2503.19037v3)**

> **作者:** Jianren Wang; Yifan Su; Abhinav Gupta; Deepak Pathak
>
> **备注:** Website at https://yifansu1301.github.io/EPO/
>
> **摘要:** On-policy reinforcement learning (RL) algorithms are widely used for their strong asymptotic performance and training stability, but they struggle to scale with larger batch sizes, as additional parallel environments yield redundant data due to limited policy-induced diversity. In contrast, Evolutionary Algorithms (EAs) scale naturally and encourage exploration via randomized population-based search, but are often sample-inefficient. We propose Evolutionary Policy Optimization (EPO), a hybrid algorithm that combines the scalability and diversity of EAs with the performance and stability of policy gradients. EPO maintains a population of agents conditioned on latent variables, shares actor-critic network parameters for coherence and memory efficiency, and aggregates diverse experiences into a master agent. Across tasks in dexterous manipulation, legged locomotion, and classic control, EPO outperforms state-of-the-art baselines in sample efficiency, asymptotic performance, and scalability.
>
---
#### [replaced 003] SafeFlow: Safe Robot Motion Planning with Flow Matching via Control Barrier Functions
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2504.08661v3](https://arxiv.org/pdf/2504.08661v3)**

> **作者:** Xiaobing Dai; Zewen Yang; Dian Yu; Fangzhou Liu; Hamid Sadeghian; Sami Haddadin; Sandra Hirche
>
> **摘要:** Recent advances in generative modeling have led to promising results in robot motion planning, particularly through diffusion and flow matching (FM)-based models that capture complex, multimodal trajectory distributions. However, these methods are typically trained offline and remain limited when faced with new environments with constraints, often lacking explicit mechanisms to ensure safety during deployment. In this work, safe flow matching (SafeFlow), a motion planning framework, is proposed for trajectory generation that integrates flow matching with safety guarantees. SafeFlow leverages our proposed flow matching barrier functions (FMBF) to ensure the planned trajectories remain within safe regions across the entire planning horizon. Crucially, our approach enables training-free, real-time safety enforcement at test time, eliminating the need for retraining. We evaluate SafeFlow on a diverse set of tasks, including planar robot navigation and 7-DoF manipulation, demonstrating superior safety and planning performance compared to state-of-the-art generative planners. Comprehensive resources are available on the project website: https://safeflowmatching.github.io.
>
---
#### [replaced 004] Mixed-Density Diffuser: Efficient Planning with Non-Uniform Temporal Resolution
- **分类: cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2510.23026v3](https://arxiv.org/pdf/2510.23026v3)**

> **作者:** Crimson Stambaugh; Rajesh P. N. Rao
>
> **备注:** European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN) (under review)
>
> **摘要:** Recent studies demonstrate that diffusion planners benefit from sparse-step planning over single-step planning. Training models to skip steps in their trajectories helps capture long-term dependencies without additional or memory computational cost. However, predicting excessively sparse plans degrades performance. We hypothesize this temporal density threshold is non-uniform across a temporal horizon and that certain parts of a planned trajectory should be more densely planned. We propose Mixed-Density Diffuser (MDD), a diffusion planner where the densities throughout the horizon are tunable hyperparameters. We show that MDD achieves a new SOTA across the Maze2D, Franka Kitchen, and Antmaze D4RL task domains.
>
---
#### [replaced 005] Target Tracking via LiDAR-RADAR Sensor Fusion for Autonomous Racing
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2505.20043v3](https://arxiv.org/pdf/2505.20043v3)**

> **作者:** Marcello Cellina; Matteo Corno; Sergio Matteo Savaresi
>
> **备注:** IEEE Conference, 6 pages
>
> **摘要:** High Speed multi-vehicle Autonomous Racing will increase the safety and performance of road-going Autonomous Vehicles. Precise vehicle detection and dynamics estimation from a moving platform is a key requirement for planning and executing complex autonomous overtaking maneuvers. To address this requirement, we have developed a Latency-Aware EKF-based Multi Target Tracking algorithm fusing LiDAR and RADAR measurements. The algorithm explots the different sensor characteristics by explicitly integrating the Range Rate in the EKF Measurement Function, as well as a-priori knowledge of the racetrack during state prediction. It can handle Out-Of-Sequence Measurements via Reprocessing using a double State and Measurement Buffer, ensuring sensor delay compensation with no information loss. This algorithm has been implemented on Team PoliMOVE's autonomous racecar, and was proved experimentally by completing a number of fully autonomous overtaking maneuvers at speeds up to 275 km/h.
>
---
#### [replaced 006] Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2506.07823v2](https://arxiv.org/pdf/2506.07823v2)**

> **作者:** Lorenzo Amatucci; João Sousa-Pinto; Giulio Turrisi; Dominique Orban; Victor Barasuol; Claudio Semini
>
> **摘要:** This paper introduces a novel Model Predictive Control (MPC) implementation for legged robot locomotion that leverages GPU parallelization. Our approach enables both temporal and state-space parallelization by incorporating a parallel associative scan to solve the primal-dual Karush-Kuhn-Tucker (KKT) system. In this way, the optimal control problem is solved in $\mathcal{O}(n\log{N} + m)$ complexity, instead of $\mathcal{O}(N(n + m)^3)$, where $n$, $m$, and $N$ are the dimension of the system state, control vector, and the length of the prediction horizon. We demonstrate the advantages of this implementation over two state-of-the-art solvers (acados and crocoddyl), achieving up to a 60\% improvement in runtime for Whole Body Dynamics (WB)-MPC and a 700\% improvement for Single Rigid Body Dynamics (SRBD)-MPC when varying the prediction horizon length. The presented formulation scales efficiently with the problem state dimensions as well, enabling the definition of a centralized controller for up to 16 legged robots that can be computed in less than 25 ms. Furthermore, thanks to the JAX implementation, the solver supports large-scale parallelization across multiple environments, allowing the possibility of performing learning with the MPC in the loop directly in GPU.
>
---
#### [replaced 007] vS-Graphs: Tightly Coupling Visual SLAM and 3D Scene Graphs Exploiting Hierarchical Scene Understanding
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.01783v2](https://arxiv.org/pdf/2503.01783v2)**

> **作者:** Ali Tourani; Saad Ejaz; Hriday Bavle; Miguel Fernandez-Cortizas; David Morilla-Cabello; Jose Luis Sanchez-Lopez; Holger Voos
>
> **备注:** 19 pages, 10 figures, 5 tables
>
> **摘要:** Current Visual Simultaneous Localization and Mapping (VSLAM) systems often struggle to create maps that are both semantically rich and easily interpretable. While incorporating semantic scene knowledge aids in building richer maps with contextual associations among mapped objects, representing them in structured formats, such as scene graphs, has not been widely addressed, resulting in complex map comprehension and limited scalability. This paper introduces vS-Graphs, a novel real-time VSLAM framework that integrates vision-based scene understanding with map reconstruction and comprehensible graph-based representation. The framework infers structural elements (i.e., rooms and floors) from detected building components (i.e., walls and ground surfaces) and incorporates them into optimizable 3D scene graphs. This solution enhances the reconstructed map's semantic richness, comprehensibility, and localization accuracy. Extensive experiments on standard benchmarks and real-world datasets demonstrate that vS-Graphs achieves an average of 15.22% accuracy gain across all tested datasets compared to state-of-the-art VSLAM methods. Furthermore, the proposed framework achieves environment-driven semantic entity detection accuracy comparable to that of precise LiDAR-based frameworks, using only visual features. The code is publicly available at https://github.com/snt-arg/visual_sgraphs and is actively being improved. Moreover, a web page containing more media and evaluation outcomes is available on https://snt-arg.github.io/vsgraphs-results/.
>
---
#### [replaced 008] SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2504.13713v4](https://arxiv.org/pdf/2504.13713v4)**

> **作者:** Samuel Cerezo; Gaetano Meli; Tomás Berriel Martins; Kirill Safronov; Javier Civera
>
> **备注:** 9 pages, 8 figures, submitted to The International Journal of Robotics Research (IJRR)
>
> **摘要:** Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
>
---
#### [replaced 009] 4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2412.13639v4](https://arxiv.org/pdf/2412.13639v4)**

> **作者:** Fernando Amodeo; Luis Merino; Fernando Caballero
>
> **备注:** Our code and results can be publicly accessed at: https://github.com/robotics-upo/gaussian-rio-cpp
>
> **摘要:** 4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly used for odometry and SLAM (Simultaneous Location and Mapping). However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing registration algorithms, especially those originally intended for more accurate sensors such as LiDAR. Following the success of 3D Gaussian Splatting for vision, in this paper we propose a summarized representation for radar scenes based on global simultaneous optimization of 3D Gaussians as opposed to voxel-based approaches, and leveraging its inherent Probability Density Function (PDF) for registration. Moreover, we propose tackling the problem of radar noise entirely within the scan matching process by optimizing multiple registration hypotheses for better protection against local optima of the PDF. Finally, following existing practice we implement an Extended Kalman Filter-based Radar-Inertial Odometry pipeline in order to evaluate the effectiveness of our system. Experiments using publicly available 4D radar datasets show that our Gaussian approach is comparable to existing registration algorithms, outperforming them in several sequences.
>
---
#### [replaced 010] KoopMotion: Learning Almost Divergence Free Koopman Flow Fields for Motion Planning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2509.09074v2](https://arxiv.org/pdf/2509.09074v2)**

> **作者:** Alice Kate Li; Thales C Silva; Victoria Edwards; Vijay Kumar; M. Ani Hsieh
>
> **备注:** Revised with link to code. Accepted to CoRL 2025 (Conference on Robot Learning). 15 pages 11 figures
>
> **摘要:** In this work, we propose a novel flow field-based motion planning method that drives a robot from any initial state to a desired reference trajectory such that it converges to the trajectory's end point. Despite demonstrated efficacy in using Koopman operator theory for modeling dynamical systems, Koopman does not inherently enforce convergence to desired trajectories nor to specified goals - a requirement when learning from demonstrations (LfD). We present KoopMotion which represents motion flow fields as dynamical systems, parameterized by Koopman Operators to mimic desired trajectories, and leverages the divergence properties of the learnt flow fields to obtain smooth motion fields that converge to a desired reference trajectory when a robot is placed away from the desired trajectory, and tracks the trajectory until the end point. To demonstrate the effectiveness of our approach, we show evaluations of KoopMotion on the LASA human handwriting dataset and a 3D manipulator end-effector trajectory dataset, including spectral analysis. We also perform experiments on a physical robot, verifying KoopMotion on a miniature autonomous surface vehicle operating in a non-static fluid flow environment. Our approach is highly sample efficient in both space and time, requiring only 3\% of the LASA dataset to generate dense motion plans. Additionally, KoopMotion provides a significant improvement over baselines when comparing metrics that measure spatial and temporal dynamics modeling efficacy. Code at: \href{https://alicekl.github.io/koop-motion/}{\color{blue}{https://alicekl.github.io/koop-motion}}.
>
---
#### [replaced 011] Certified Training with Branch-and-Bound for Lyapunov-stable Neural Control
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2411.18235v2](https://arxiv.org/pdf/2411.18235v2)**

> **作者:** Zhouxing Shi; Haoyu Li; Cho-Jui Hsieh; Huan Zhang
>
> **备注:** Preprint
>
> **摘要:** We study the problem of learning verifiably Lyapunov-stable neural controllers that provably satisfy the Lyapunov asymptotic stability condition within a region-of-attraction (ROA). Unlike previous works that adopted counterexample-guided training without considering the computation of verification in training, we introduce Certified Training with Branch-and-Bound (CT-BaB), a new certified training framework that optimizes certified bounds, thereby reducing the discrepancy between training and test-time verification that also computes certified bounds. To achieve a relatively global guarantee on an entire input region-of-interest, we propose a training-time BaB technique that maintains a dynamic training dataset and adaptively splits hard input subregions into smaller ones, to tighten certified bounds and ease the training. Meanwhile, subregions created by the training-time BaB also inform test-time verification, for a more efficient training-aware verification. We demonstrate that CT-BaB yields verification-friendly models that can be more efficiently verified at test time while achieving stronger verifiable guarantees with larger ROA. On the largest output-feedback 2D Quadrotor system experimented, CT-BaB reduces verification time by over 11X relative to the previous state-of-the-art baseline while achieving 164X larger ROA.
>
---
#### [replaced 012] Robust Bayesian Scene Reconstruction with Retrieval-Augmented Priors for Precise Grasping and Planning
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2411.19461v3](https://arxiv.org/pdf/2411.19461v3)**

> **作者:** Herbert Wright; Weiming Zhi; Martin Matak; Matthew Johnson-Roberson; Tucker Hermans
>
> **摘要:** Constructing 3D representations of object geometry is critical for many robotics tasks, particularly manipulation problems. These representations must be built from potentially noisy partial observations. In this work, we focus on the problem of reconstructing a multi-object scene from a single RGBD image using a fixed camera. Traditional scene representation methods generally cannot infer the geometry of unobserved regions of the objects in the image. Attempts have been made to leverage deep learning to train on a dataset of known objects and representations, and then generalize to new observations. However, this can be brittle to noisy real-world observations and objects not contained in the dataset, and do not provide well-calibrated reconstruction confidences. We propose BRRP, a reconstruction method that leverages preexisting mesh datasets to build an informative prior during robust probabilistic reconstruction. We introduce the concept of a retrieval-augmented prior, where we retrieve relevant components of our prior distribution from a database of objects during inference. The resulting prior enables estimation of the geometry of occluded portions of the in-scene objects. Our method produces a distribution over object shape that can be used for reconstruction and measuring uncertainty. We evaluate our method in both simulated scenes and in the real world. We demonstrate the robustness of our method against deep learning-only approaches while being more accurate than a method without an informative prior. Through real-world experiments, we particularly highlight the capability of BRRP to enable successful dexterous manipulation in clutter.
>
---
#### [replaced 013] Gaussian-Process-based Adaptive Tracking Control with Dynamic Active Learning for Autonomous Ground Vehicles
- **分类: eess.SY; cs.RO**

- **链接: [https://arxiv.org/pdf/2501.14672v3](https://arxiv.org/pdf/2501.14672v3)**

> **作者:** Kristóf Floch; Tamás Péni; Roland Tóth
>
> **备注:** Submitted to IEEE Transactions on Control Systems Technology (revised)
>
> **摘要:** This article proposes an active-learning-based adaptive trajectory tracking control method for autonomous ground vehicles to compensate for modeling errors and unmodeled dynamics. The nominal vehicle model is decoupled into lateral and longitudinal subsystems, which are augmented with online Gaussian Processes (GPs), using measurement data. The estimated mean functions of the GPs are used to construct a feedback compensator, which, together with an LPV state feedback controller designed for the nominal system, gives the adaptive control structure. To assist exploration of the dynamics, the paper proposes a new, dynamic active learning method to collect the most informative samples to accelerate the training process. To analyze the performance of the overall learning tool-chain provided controller, a novel iterative, counterexample-based algorithm is proposed for calculating the induced L2 gain between the reference trajectory and the tracking error. The analysis can be executed for a set of possible realizations of the to-be-controlled system, giving robust performance certificate of the learning method under variation of the vehicle dynamics. The efficiency of the proposed control approach is shown on a high-fidelity physics simulator and in real experiments using a 1/10 scale F1TENTH electric car.
>
---
#### [replaced 014] Touch in the Wild: Learning Fine-Grained Manipulation with a Portable Visuo-Tactile Gripper
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.15062v2](https://arxiv.org/pdf/2507.15062v2)**

> **作者:** Xinyue Zhu; Binghao Huang; Yunzhu Li
>
> **备注:** More videos can be found on our website:https://binghao-huang.github.io/touch_in_the_wild/
>
> **摘要:** Handheld grippers are increasingly used to collect human demonstrations due to their ease of deployment and versatility. However, most existing designs lack tactile sensing, despite the critical role of tactile feedback in precise manipulation. We present a portable, lightweight gripper with integrated tactile sensors that enables synchronized collection of visual and tactile data in diverse, real-world, and in-the-wild settings. Building on this hardware, we propose a cross-modal representation learning framework that integrates visual and tactile signals while preserving their distinct characteristics. The learning procedure allows the emergence of interpretable representations that consistently focus on contacting regions relevant for physical interactions. When used for downstream manipulation tasks, these representations enable more efficient and effective policy learning, supporting precise robotic manipulation based on multimodal feedback. We validate our approach on fine-grained tasks such as test tube insertion and pipette-based fluid transfer, demonstrating improved accuracy and robustness under external disturbances. Our project page is available at https://binghao-huang.github.io/touch_in_the_wild/ .
>
---
#### [replaced 015] Real Garment Benchmark (RGBench): A Comprehensive Benchmark for Robotic Garment Manipulation featuring a High-Fidelity Scalable Simulator
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.06434v2](https://arxiv.org/pdf/2511.06434v2)**

> **作者:** Wenkang Hu; Xincheng Tang; Yanzhi E; Yitong Li; Zhengjie Shu; Wei Li; Huamin Wang; Ruigang Yang
>
> **备注:** 2026 AAAI Accept
>
> **摘要:** While there has been significant progress to use simulated data to learn robotic manipulation of rigid objects, applying its success to deformable objects has been hindered by the lack of both deformable object models and realistic non-rigid body simulators. In this paper, we present Real Garment Benchmark (RGBench), a comprehensive benchmark for robotic manipulation of garments. It features a diverse set of over 6000 garment mesh models, a new high-performance simulator, and a comprehensive protocol to evaluate garment simulation quality with carefully measured real garment dynamics. Our experiments demonstrate that our simulator outperforms currently available cloth simulators by a large margin, reducing simulation error by 20% while maintaining a speed of 3 times faster. We will publicly release RGBench to accelerate future research in robotic garment manipulation. Website: https://rgbench.github.io/
>
---
#### [replaced 016] Survey of Vision-Language-Action Models for Embodied Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2508.15201v2](https://arxiv.org/pdf/2508.15201v2)**

> **作者:** Haoran Li; Yuhui Chen; Wenbo Cui; Weiheng Liu; Kai Liu; Mingcai Zhou; Zhengtao Zhang; Dongbin Zhao
>
> **备注:** in Chinese language
>
> **摘要:** Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions.
>
---
#### [replaced 017] ViSA-Flow: Accelerating Robot Skill Learning via Large-Scale Video Semantic Action Flow
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2505.01288v3](https://arxiv.org/pdf/2505.01288v3)**

> **作者:** Changhe Chen; Quantao Yang; Xiaohao Xu; Nima Fazeli; Olov Andersson
>
> **摘要:** One of the central challenges preventing robots from acquiring complex manipulation skills is the prohibitive cost of collecting large-scale robot demonstrations. In contrast, humans are able to learn efficiently by watching others interact with their environment. To bridge this gap, we introduce semantic action flow as a core intermediate representation capturing the essential spatio-temporal manipulator-object interactions, invariant to superficial visual differences. We present ViSA-Flow, a framework that learns this representation self-supervised from unlabeled large-scale video data. First, a generative model is pre-trained on semantic action flows automatically extracted from large-scale human-object interaction video data, learning a robust prior over manipulation structure. Second, this prior is efficiently adapted to a target robot by fine-tuning on a small set of robot demonstrations processed through the same semantic abstraction pipeline. We demonstrate through extensive experiments on the CALVIN benchmark and real-world tasks that ViSA-Flow achieves state-of-the-art performance, particularly in low-data regimes, outperforming prior methods by effectively transferring knowledge from human video observation to robotic execution. Videos are available at https://visaflow-web.github.io/ViSAFLOW.
>
---
#### [replaced 018] MLM: Learning Multi-task Loco-Manipulation Whole-Body Control for Quadruped Robot with Arm
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2508.10538v2](https://arxiv.org/pdf/2508.10538v2)**

> **作者:** Xin Liu; Bida Ma; Chenkun Qi; Yan Ding; Nuo Xu; Zhaxizhuoma; Guorong Zhang; Pengan Chen; Kehui Liu; Zhongjie Jia; Chuyue Guan; Yule Mo; Jiaqi Liu; Feng Gao; Jiangwei Zhong; Bin Zhao; Xuelong Li
>
> **摘要:** Whole-body loco-manipulation for quadruped robots with arms remains a challenging problem, particularly in achieving multi-task control. To address this, we propose MLM, a reinforcement learning framework driven by both real-world and simulation data. It enables a six-DoF robotic arm-equipped quadruped robot to perform whole-body loco-manipulation for multiple tasks autonomously or under human teleoperation. To address the problem of balancing multiple tasks during the learning of loco-manipulation, we introduce a trajectory library with an adaptive, curriculum-based sampling mechanism. This approach allows the policy to efficiently leverage real-world collected trajectories for learning multi-task loco-manipulation. To address deployment scenarios with only historical observations and to enhance the performance of policy execution across tasks with different spatial ranges, we propose a Trajectory-Velocity Prediction policy network. It predicts unobservable future trajectories and velocities. By leveraging extensive simulation data and curriculum-based rewards, our controller achieves whole-body behaviors in simulation and zero-shot transfer to real-world deployment. Ablation studies in simulation verify the necessity and effectiveness of our approach, while real-world experiments on a Go2 robot with an Airbot robotic arm demonstrate the policy's good performance in multi-task execution.
>
---
#### [replaced 019] MOSAIC: A Skill-Centric Algorithmic Framework for Long-Horizon Manipulation Planning
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2504.16738v2](https://arxiv.org/pdf/2504.16738v2)**

> **作者:** Itamar Mishani; Yorai Shaoul; Maxim Likhachev
>
> **备注:** Under review. Project page: https://skill-mosaic.github.io
>
> **摘要:** Planning long-horizon manipulation motions using a set of predefined skills is a central challenge in robotics; solving it efficiently could enable general-purpose robots to tackle novel tasks by flexibly composing generic skills. Solutions to this problem lie in an infinitely vast space of parameterized skill sequences -- a space where common incremental methods struggle to find sequences that have non-obvious intermediate steps. Some approaches reason over lower-dimensional, symbolic spaces, which are more tractable to explore but may be brittle and are laborious to construct. In this work, we introduce MOSAIC, a skill-centric, multi-directional planning approach that targets these challenges by reasoning about which skills to employ and where they are most likely to succeed, by utilizing physics simulation to estimate skill execution outcomes. Specifically, MOSAIC employs two complementary skill families: Generators, which identify ``islands of competence'' where skills are demonstrably effective, and Connectors, which link these skill-trajectories by solving boundary value problems. By focusing planning efforts on regions of high competence, MOSAIC efficiently discovers physically-grounded solutions. We demonstrate its efficacy on complex long-horizon problems in both simulation and the real world, using a diverse set of skills including generative diffusion models, motion planning algorithms, and manipulation-specific models. Visit skill-mosaic.github.io for demonstrations and examples.
>
---
#### [replaced 020] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications
- **分类: cs.CY; cs.AI; cs.HC; cs.RO**

- **链接: [https://arxiv.org/pdf/2505.18371v2](https://arxiv.org/pdf/2505.18371v2)**

> **作者:** Riley Simmons-Edler; Jean Dong; Paul Lushenko; Kanaka Rajan; Ryan P. Badman
>
> **备注:** Published at NeurIPS 2025, 10 pages, 2 tables, 1 figure
>
> **摘要:** Military weapon systems and command-and-control infrastructure augmented by artificial intelligence (AI) have seen rapid development and deployment in recent years. However, the sociotechnical impacts of AI on combat systems, military decision-making, and the norms of warfare have been understudied. We focus on a specific subset of lethal autonomous weapon systems (LAWS) that use AI for targeting or battlefield decisions. We refer to this subset as AI-powered lethal autonomous weapon systems (AI-LAWS) and argue that they introduce novel risks- including unanticipated escalation, poor reliability in unfamiliar environments, and erosion of human oversight- all of which threaten both military effectiveness and the openness of AI research. These risks cannot be addressed by high-level policy alone; effective regulation must be grounded in the technical behavior of AI models. We argue that AI researchers must be involved throughout the regulatory lifecycle. Thus, we propose a clear, behavior-based definition of AI-LAWS- systems that introduce unique risks through their use of modern AI- as a foundation for technically grounded regulation, given that existing frameworks do not distinguish them from conventional LAWS. Using this definition, we propose several technically-informed policy directions and invite greater participation from the AI research community in military AI policy discussions.
>
---
#### [replaced 021] LLM4AD: Large Language Models for Autonomous Driving - Concept, Review, Benchmark, Experiments, and Future Trends
- **分类: cs.RO; cs.AI; cs.CL; cs.HC**

- **链接: [https://arxiv.org/pdf/2410.15281v4](https://arxiv.org/pdf/2410.15281v4)**

> **作者:** Can Cui; Yunsheng Ma; Sung-Yeon Park; Zichong Yang; Yupeng Zhou; Juanwu Lu; Juntong Peng; Jiaru Zhang; Ruqi Zhang; Lingxi Li; Yaobin Chen; Jitesh H. Panchal; Amr Abdelraouf; Rohit Gupta; Kyungtae Han; Ziran Wang
>
> **摘要:** With the broader adoption and highly successful development of Large Language Models (LLMs), there has been growing interest and demand for applying LLMs to autonomous driving technology. Driven by their natural language understanding and reasoning capabilities, LLMs have the potential to enhance various aspects of autonomous driving systems, from perception and scene understanding to interactive decision-making. In this paper, we first introduce the novel concept of designing Large Language Models for Autonomous Driving (LLM4AD), followed by a review of existing LLM4AD studies. Then, we propose a comprehensive benchmark for evaluating the instruction-following and reasoning abilities of LLM4AD systems, which includes LaMPilot-Bench, CARLA Leaderboard 1.0 Benchmark in simulation and NuPlanQA for multi-view visual question answering. Furthermore, we conduct extensive real-world experiments on autonomous vehicle platforms, examining both on-cloud and on-edge LLM deployment for personalized decision-making and motion control. Next, we explore the future trends of integrating language diffusion models into autonomous driving, exemplified by the proposed ViLaD (Vision-Language Diffusion) framework. Finally, we discuss the main challenges of LLM4AD, including latency, deployment, security and privacy, safety, trust and transparency, and personalization.
>
---
#### [replaced 022] Trends in Motion Prediction Toward Deployable and Generalizable Autonomy: A Revisit and Perspectives
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2505.09074v4](https://arxiv.org/pdf/2505.09074v4)**

> **作者:** Letian Wang; Marc-Antoine Lavoie; Sandro Papais; Barza Nisar; Yuxiao Chen; Wenhao Ding; Boris Ivanovic; Hao Shao; Abulikemu Abuduweili; Evan Cook; Yang Zhou; Peter Karkus; Jiachen Li; Changliu Liu; Marco Pavone; Steven Waslander
>
> **备注:** (Book) To Appear in Foundation and Trends in Robotics. 163 pages, 40 figures, 13 tables
>
> **摘要:** Motion prediction, recently popularized as world models, refers to the anticipation of future agent states or scene evolution, which is rooted in human cognition, bridging perception and decision-making. It enables intelligent systems, such as robots and self-driving cars, to act safely in dynamic, human-involved environments, and informs broader time-series reasoning challenges. With advances in methods, representations, and datasets, the field has seen rapid progress, reflected in quickly evolving benchmark results. Yet, when state-of-the-art methods are deployed in the real world, they often struggle to generalize to open-world conditions and fall short of deployment standards. This reveals a gap between research benchmarks, which are often idealized or ill-posed, and real-world complexity. To address this gap, this survey revisits the generalization and deployability of motion prediction models, with an emphasis on applications of robotics, autonomous driving, and human motion. We first offer a comprehensive taxonomy of motion prediction methods, covering representations, modeling strategies, application domains, and evaluation protocols. We then study two key challenges: (1) how to push motion prediction models to be deployable to realistic deployment standards, where motion prediction does not act in a vacuum, but functions as one module of closed-loop autonomy stacks - it takes input localization and perception, and informs downstream planning and control. 2) How to generalize motion prediction models from limited seen scenarios/datasets to the open-world settings. Throughout the paper, we highlight critical open challenges to guide future work, aiming to recalibrate the community's efforts, fostering progress that is not only measurable but also meaningful for real-world applications. The project webpage can be found here https://trends-in-motion-prediction-2025.github.io/.
>
---
