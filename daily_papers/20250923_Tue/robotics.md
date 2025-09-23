# 机器人 cs.RO

- **最新发布 93 篇**

- **更新 49 篇**

## 最新发布

#### [new 001] Robot Conga: A Leader-Follower Walking Approach to Sequential Path Following in Multi-Agent Systems
- **分类: cs.RO; cs.SY; eess.SY; math.DS; physics.app-ph; 49; I.2.9**

- **简介: 该论文研究多智能体系统的顺序路径跟踪任务，旨在解决传统方法同步困难和行为僵硬的问题。提出Robot Conga算法，基于领导者位移而非时间更新跟随者状态，实现稳定间距与快速收敛的轨迹跟踪。**

- **链接: [http://arxiv.org/pdf/2509.16482v1](http://arxiv.org/pdf/2509.16482v1)**

> **作者:** Pranav Tiwari; Soumyodipta Nath
>
> **备注:** 6 Pages, 8 Figures. First two authors have contributed equally
>
> **摘要:** Coordinated path following in multi-agent systems is a key challenge in robotics, with applications in automated logistics, surveillance, and collaborative exploration. Traditional formation control techniques often rely on time-parameterized trajectories and path integrals, which can result in synchronization issues and rigid behavior. In this work, we address the problem of sequential path following, where agents maintain fixed spatial separation along a common trajectory, guided by a leader under centralized control. We introduce Robot Conga, a leader-follower control strategy that updates each agent's desired state based on the leader's spatial displacement rather than time, assuming access to a global position reference, an assumption valid in indoor environments equipped with motion capture, vision-based tracking, or UWB localization systems. The algorithm was validated in simulation using both TurtleBot3 and quadruped (Laikago) robots. Results demonstrate accurate trajectory tracking, stable inter-agent spacing, and fast convergence, with all agents aligning within 250 time steps (approx. 0.25 seconds) in the quadruped case, and almost instantaneously in the TurtleBot3 implementation.
>
---
#### [new 002] Neural Network and ANFIS based auto-adaptive MPC for path tracking in autonomous vehicles
- **分类: cs.RO; math.OC**

- **简介: 该论文针对自动驾驶车辆的路径跟踪任务，研究如何解决环境变化和不确定性对横向控制的影响。提出了基于神经网络和ANFIS的自适应MPC控制器，并通过改进的粒子群算法进行调参，实验显示其在复杂场景下优于传统MPC。**

- **链接: [http://arxiv.org/pdf/2509.17213v1](http://arxiv.org/pdf/2509.17213v1)**

> **作者:** Yassine Kebbati; Naima Ait-Oufroukh; Vincent Vigneron; Dalil Ichala
>
> **摘要:** Self-driving cars operate in constantly changing environments and are exposed to a variety of uncertainties and disturbances. These factors render classical controllers ineffective, especially for lateral control. Therefore, an adaptive MPC controller is designed in this paper for the path tracking task, tuned by an improved particle swarm optimization algorithm. Online parameter adaptation is performed using Neural Networks and ANFIS. The designed controller showed promising results compared to standard MPC in triple lane change and trajectory tracking scenarios. Code can be found here: https://github.com/yassinekebbati/NN_MPC-vs-ANFIS_MPC
>
---
#### [new 003] Factorizing Diffusion Policies for Observation Modality Prioritization
- **分类: cs.RO**

- **简介: 该论文提出一种“因子化扩散策略”（FDP），用于机器人技能学习。针对不同观测模态在任务中的影响不同，FDP通过因子化观测条件，使策略能优先考虑特定模态，从而提升低数据和分布偏移下的性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.16830v1](http://arxiv.org/pdf/2509.16830v1)**

> **作者:** Omkar Patil; Prabin Rath; Kartikay Pangaonkar; Eric Rosen; Nakul Gopalan
>
> **备注:** 14 pages; website: https://fdp-policy.github.io/fdp-policy/
>
> **摘要:** Diffusion models have been extensively leveraged for learning robot skills from demonstrations. These policies are conditioned on several observational modalities such as proprioception, vision and tactile. However, observational modalities have varying levels of influence for different tasks that diffusion polices fail to capture. In this work, we propose 'Factorized Diffusion Policies' abbreviated as FDP, a novel policy formulation that enables observational modalities to have differing influence on the action diffusion process by design. This results in learning policies where certain observations modalities can be prioritized over the others such as $\texttt{vision>tactile}$ or $\texttt{proprioception>vision}$. FDP achieves modality prioritization by factorizing the observational conditioning for diffusion process, resulting in more performant and robust policies. Our factored approach shows strong performance improvements in low-data regimes with $15\%$ absolute improvement in success rate on several simulated benchmarks when compared to a standard diffusion policy that jointly conditions on all input modalities. Moreover, our benchmark and real-world experiments show that factored policies are naturally more robust with $40\%$ higher absolute success rate across several visuomotor tasks under distribution shifts such as visual distractors or camera occlusions, where existing diffusion policies fail catastrophically. FDP thus offers a safer and more robust alternative to standard diffusion policies for real-world deployment. Videos are available at https://fdp-policy.github.io/fdp-policy/ .
>
---
#### [new 004] V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts
- **分类: cs.RO**

- **简介: 该论文提出V2V-GoT框架，结合多模态大语言模型与图式推理，解决车辆遮挡导致的自动驾驶安全问题。通过协同感知与规划，提升车对车协作的自主驾驶性能。**

- **链接: [http://arxiv.org/pdf/2509.18053v1](http://arxiv.org/pdf/2509.18053v1)**

> **作者:** Hsu-kuang Chiu; Ryo Hachiuma; Chien-Yi Wang; Yu-Chiang Frank Wang; Min-Hung Chen; Stephen F. Smith
>
> **摘要:** Current state-of-the-art autonomous vehicles could face safety-critical situations when their local sensors are occluded by large nearby objects on the road. Vehicle-to-vehicle (V2V) cooperative autonomous driving has been proposed as a means of addressing this problem, and one recently introduced framework for cooperative autonomous driving has further adopted an approach that incorporates a Multimodal Large Language Model (MLLM) to integrate cooperative perception and planning processes. However, despite the potential benefit of applying graph-of-thoughts reasoning to the MLLM, this idea has not been considered by previous cooperative autonomous driving research. In this paper, we propose a novel graph-of-thoughts framework specifically designed for MLLM-based cooperative autonomous driving. Our graph-of-thoughts includes our proposed novel ideas of occlusion-aware perception and planning-aware prediction. We curate the V2V-GoT-QA dataset and develop the V2V-GoT model for training and testing the cooperative driving graph-of-thoughts. Our experimental results show that our method outperforms other baselines in cooperative perception, prediction, and planning tasks.
>
---
#### [new 005] DyDexHandover: Human-like Bimanual Dynamic Dexterous Handover using RGB-only Perception
- **分类: cs.RO**

- **简介: 该论文研究双臂机器人空中交接任务，旨在解决动态交接中感知与协调的问题。提出DyDexHandover框架，基于RGB图像，利用多智能体强化学习训练端到端策略，并通过人类策略正则化提升自然性与泛化能力，在仿真中取得良好效果。**

- **链接: [http://arxiv.org/pdf/2509.17350v1](http://arxiv.org/pdf/2509.17350v1)**

> **作者:** Haoran Zhou; Yangwei You; Shuaijun Wang
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Dynamic in air handover is a fundamental challenge for dual-arm robots, requiring accurate perception, precise coordination, and natural motion. Prior methods often rely on dynamics models, strong priors, or depth sensing, limiting generalization and naturalness. We present DyDexHandover, a novel framework that employs multi-agent reinforcement learning to train an end to end RGB based policy for bimanual object throwing and catching. To achieve more human-like behavior, the throwing policy is guided by a human policy regularization scheme, encouraging fluid and natural motion, and enhancing the generalization capability of the policy. A dual arm simulation environment was built in Isaac Sim for experimental evaluation. DyDexHandover achieves nearly 99 percent success on training objects and 75 percent on unseen objects, while generating human-like throwing and catching behaviors. To our knowledge, it is the first method to realize dual-arm in-air handover using only raw RGB perception.
>
---
#### [new 006] RadarSFD: Single-Frame Diffusion with Pretrained Priors for Radar Point Clouds
- **分类: cs.RO; eess.SP**

- **简介: 该论文提出RadarSFD，一种基于预训练先验的单帧扩散模型，用于从毫米波雷达点云重建高密度LiDAR-like点云。解决了小尺寸机器人系统中多帧依赖的问题，实现了无运动、无SAR的高精度点云感知。**

- **链接: [http://arxiv.org/pdf/2509.18068v1](http://arxiv.org/pdf/2509.18068v1)**

> **作者:** Bin Zhao; Nakul Garg
>
> **摘要:** Millimeter-wave radar provides perception robust to fog, smoke, dust, and low light, making it attractive for size, weight, and power constrained robotic platforms. Current radar imaging methods, however, rely on synthetic aperture or multi-frame aggregation to improve resolution, which is impractical for small aerial, inspection, or wearable systems. We present RadarSFD, a conditional latent diffusion framework that reconstructs dense LiDAR-like point clouds from a single radar frame without motion or SAR. Our approach transfers geometric priors from a pretrained monocular depth estimator into the diffusion backbone, anchors them to radar inputs via channel-wise latent concatenation, and regularizes outputs with a dual-space objective combining latent and pixel-space losses. On the RadarHD benchmark, RadarSFD achieves 35 cm Chamfer Distance and 28 cm Modified Hausdorff Distance, improving over the single-frame RadarHD baseline (56 cm, 45 cm) and remaining competitive with multi-frame methods using 5-41 frames. Qualitative results show recovery of fine walls and narrow gaps, and experiments across new environments confirm strong generalization. Ablation studies highlight the importance of pretrained initialization, radar BEV conditioning, and the dual-space loss. Together, these results establish the first practical single-frame, no-SAR mmWave radar pipeline for dense point cloud perception in compact robotic systems.
>
---
#### [new 007] Scalable Multi Agent Diffusion Policies for Coverage Control
- **分类: cs.RO**

- **简介: 该论文提出MADP，一种基于扩散模型的多智能体协作方法，用于解决去中心化机器人集群的覆盖控制问题。通过模仿学习训练策略，利用空间变换器实现去中心化推理，在不同环境和密度下表现出良好的泛化性和性能。**

- **链接: [http://arxiv.org/pdf/2509.17244v1](http://arxiv.org/pdf/2509.17244v1)**

> **作者:** Frederic Vatnsdal; Romina Garcia Camargo; Saurav Agarwal; Alejandro Ribeiro
>
> **摘要:** We propose MADP, a novel diffusion-model-based approach for collaboration in decentralized robot swarms. MADP leverages diffusion models to generate samples from complex and high-dimensional action distributions that capture the interdependencies between agents' actions. Each robot conditions policy sampling on a fused representation of its own observations and perceptual embeddings received from peers. To evaluate this approach, we task a team of holonomic robots piloted by MADP to address coverage control-a canonical multi agent navigation problem. The policy is trained via imitation learning from a clairvoyant expert on the coverage control problem, with the diffusion process parameterized by a spatial transformer architecture to enable decentralized inference. We evaluate the system under varying numbers, locations, and variances of importance density functions, capturing the robustness demands of real-world coverage tasks. Experiments demonstrate that our model inherits valuable properties from diffusion models, generalizing across agent densities and environments, and consistently outperforming state-of-the-art baselines.
>
---
#### [new 008] Video-to-BT: Generating Reactive Behavior Trees from Human Demonstration Videos for Robotic Assembly
- **分类: cs.RO**

- **简介: 该论文提出Video-to-BT框架，利用视觉-语言模型从人类演示视频中生成行为树（BT），用于机器人装配任务。旨在解决传统编程方法缺乏灵活性和鲁棒性的问题，实现高适应性的自主装配系统。**

- **链接: [http://arxiv.org/pdf/2509.16611v1](http://arxiv.org/pdf/2509.16611v1)**

> **作者:** Xiwei Zhao; Yiwei Wang; Yansong Wu; Fan Wu; Teng Sun; Zhonghua Miao; Sami Haddadin; Alois Knoll
>
> **摘要:** Modern manufacturing demands robotic assembly systems with enhanced flexibility and reliability. However, traditional approaches often rely on programming tailored to each product by experts for fixed settings, which are inherently inflexible to product changes and lack the robustness to handle variations. As Behavior Trees (BTs) are increasingly used in robotics for their modularity and reactivity, we propose a novel hierarchical framework, Video-to-BT, that seamlessly integrates high-level cognitive planning with low-level reactive control, with BTs serving both as the structured output of planning and as the governing structure for execution. Our approach leverages a Vision-Language Model (VLM) to decompose human demonstration videos into subtasks, from which Behavior Trees are generated. During the execution, the planned BTs combined with real-time scene interpretation enable the system to operate reactively in the dynamic environment, while VLM-driven replanning is triggered upon execution failure. This closed-loop architecture ensures stability and adaptivity. We validate our framework on real-world assembly tasks through a series of experiments, demonstrating high planning reliability, robust performance in long-horizon assembly tasks, and strong generalization across diverse and perturbed conditions. Project website: https://video2bt.github.io/video2bt_page/
>
---
#### [new 009] M3ET: Efficient Vision-Language Learning for Robotics based on Multimodal Mamba-Enhanced Transformer
- **分类: cs.RO**

- **简介: 该论文提出M3ET模型，用于机器人领域的高效视觉-语言学习。针对现有方法计算量大、语义提取受限的问题，M3ET结合Mamba模块和自适应注意力机制，优化多模态融合与对齐，实现轻量化设计，在保持较高VQA任务精度的同时显著降低参数量和资源消耗。**

- **链接: [http://arxiv.org/pdf/2509.18005v1](http://arxiv.org/pdf/2509.18005v1)**

> **作者:** Yanxin Zhang; Liang He; Zeyi Kang; Zuheng Ming; Kaixing Zhao
>
> **备注:** 8 pages
>
> **摘要:** In recent years, multimodal learning has become essential in robotic vision and information fusion, especially for understanding human behavior in complex environments. However, current methods struggle to fully leverage the textual modality, relying on supervised pretrained models, which limits semantic extraction in unsupervised robotic environments, particularly with significant modality loss. These methods also tend to be computationally intensive, leading to high resource consumption in real-world applications. To address these challenges, we propose the Multi Modal Mamba Enhanced Transformer (M3ET), a lightweight model designed for efficient multimodal learning, particularly on mobile platforms. By incorporating the Mamba module and a semantic-based adaptive attention mechanism, M3ET optimizes feature fusion, alignment, and modality reconstruction. Our experiments show that M3ET improves cross-task performance, with a 2.3 times increase in pretraining inference speed. In particular, the core VQA task accuracy of M3ET remains at 0.74, while the model's parameter count is reduced by 0.67. Although performance on the EQA task is limited, M3ET's lightweight design makes it well suited for deployment on resource-constrained robotic platforms.
>
---
#### [new 010] Subteaming and Adaptive Formation Control for Coordinated Multi-Robot Navigation
- **分类: cs.RO**

- **简介: 该论文研究多机器人协同导航任务，旨在解决复杂环境中机器人团队保持队形的难题。提出STAF方法，通过分层学习框架实现动态分组与自适应队形控制，提升狭窄等场景下的导航能力。**

- **链接: [http://arxiv.org/pdf/2509.16412v1](http://arxiv.org/pdf/2509.16412v1)**

> **作者:** Zihao Deng; Peng Gao; Williard Joshua Jose; Maggie Wigness; John Rogers; Brian Reily; Christopher Reardon; Hao Zhang
>
> **摘要:** Coordinated multi-robot navigation is essential for robots to operate as a team in diverse environments. During navigation, robot teams usually need to maintain specific formations, such as circular formations to protect human teammates at the center. However, in complex scenarios such as narrow corridors, rigidly preserving predefined formations can become infeasible. Therefore, robot teams must be capable of dynamically splitting into smaller subteams and adaptively controlling the subteams to navigate through such scenarios while preserving formations. To enable this capability, we introduce a novel method for SubTeaming and Adaptive Formation (STAF), which is built upon a unified hierarchical learning framework: (1) high-level deep graph cut for team splitting, (2) intermediate-level graph learning for facilitating coordinated navigation among subteams, and (3) low-level policy learning for controlling individual mobile robots to reach their goal positions while avoiding collisions. To evaluate STAF, we conducted extensive experiments in both indoor and outdoor environments using robotics simulations and physical robot teams. Experimental results show that STAF enables the novel capability for subteaming and adaptive formation control, and achieves promising performance in coordinated multi-robot navigation through challenging scenarios. More details are available on the project website: https://hcrlab.gitlab.io/project/STAF.
>
---
#### [new 011] SMART-3D: Three-Dimensional Self-Morphing Adaptive Replanning Tree
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文提出SMART-3D，一种用于动态三维环境的自适应重规划算法。它通过树结构的自变形实现快速避障路径规划，解决了三维环境中移动障碍物导致的路径阻塞问题，提升了实时性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.16812v1](http://arxiv.org/pdf/2509.16812v1)**

> **作者:** Priyanshu Agrawal; Shalabh Gupta; Zongyuan Shen
>
> **摘要:** This paper presents SMART-3D, an extension of the SMART algorithm to 3D environments. SMART-3D is a tree-based adaptive replanning algorithm for dynamic environments with fast moving obstacles. SMART-3D morphs the underlying tree to find a new path in real-time whenever the current path is blocked by obstacles. SMART-3D removed the grid decomposition requirement of the SMART algorithm by replacing the concept of hot-spots with that of hot-nodes, thus making it computationally efficient and scalable to 3D environments. The hot-nodes are nodes which allow for efficient reconnections to morph the existing tree to find a new safe and reliable path. The performance of SMART-3D is evaluated by extensive simulations in 2D and 3D environments populated with randomly moving dynamic obstacles. The results show that SMART-3D achieves high success rates and low replanning times, thus highlighting its suitability for real-time onboard applications.
>
---
#### [new 012] Learning and Optimization with 3D Orientations
- **分类: cs.RO; cs.LG; math.OC**

- **简介: 该论文研究3D姿态表示与优化问题，旨在统一梳理各种表示方法并进行实验对比。针对机器人领域常见任务（如优化、学习等），提供选择建议与参考实现。**

- **链接: [http://arxiv.org/pdf/2509.17274v1](http://arxiv.org/pdf/2509.17274v1)**

> **作者:** Alexandros Ntagkas; Constantinos Tsakonas; Chairi Kiourt; Konstantinos Chatzilygeroudis
>
> **备注:** 9 pages, 11 figures
>
> **摘要:** There exist numerous ways of representing 3D orientations. Each representation has both limitations and unique features. Choosing the best representation for one task is often a difficult chore, and there exist conflicting opinions on which representation is better suited for a set of family of tasks. Even worse, when dealing with scenarios where we need to learn or optimize functions with orientations as inputs and/or outputs, the set of possibilities (representations, loss functions, etc.) is even larger and it is not easy to decide what is best for each scenario. In this paper, we attempt to a) present clearly, concisely and with unified notation all available representations, and "tricks" related to 3D orientations (including Lie Group algebra), and b) benchmark them in representative scenarios. The first part feels like it is missing from the robotics literature as one has to read many different textbooks and papers in order have a concise and clear understanding of all possibilities, while the benchmark is necessary in order to come up with recommendations based on empirical evidence. More precisely, we experiment with the following settings that attempt to cover most widely used scenarios in robotics: 1) direct optimization, 2) imitation/supervised learning with a neural network controller, 3) reinforcement learning, and 4) trajectory optimization using differential dynamic programming. We finally provide guidelines depending on the scenario, and make available a reference implementation of all the orientation math described.
>
---
#### [new 013] Certifiably Optimal Doppler Positioning using Opportunistic LEO Satellites
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种基于凸优化的LEO卫星多普勒定位方法，用于GNSS备份。针对非凸定位问题易陷入局部最优的问题，采用GWA算法和SDP松弛实现无需初始值的全局最优定位，并通过实验证明其有效性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.17198v1](http://arxiv.org/pdf/2509.17198v1)**

> **作者:** Baoshan Song; Weisong Wen; Qi Zhang; Bing Xu; Li-Ta Hsu
>
> **备注:** This manuscript has been submitted to IEEE Transactions on Aerospace and Electronic Systems (TAES). The current version is uploaded to arXiv for open access and reference purposes only
>
> **摘要:** To provide backup and augmentation to global navigation satellite system (GNSS), Doppler shift from Low Earth Orbit (LEO) satellites can be employed as signals of opportunity (SOP) for position, navigation and timing (PNT). Since the Doppler positioning problem is non-convex, local searching methods may produce two types of estimates: a global optimum without notice or a local optimum given an inexact initial estimate. As exact initialization is unavailable in some unknown environments, a guaranteed global optimization method in no need of initialization becomes necessary. To achieve this goal, we propose a certifiably optimal LEO Doppler positioning method by utilizing convex optimization. In this paper, the certifiable positioning method is implemented through a graduated weight approximation (GWA) algorithm and semidefinite programming (SDP) relaxation. To guarantee the optimality, we derive the necessary conditions for optimality in ideal noiseless cases and sufficient noise bounds conditions in noisy cases. Simulation and real tests are conducted to evaluate the effectiveness and robustness of the proposed method. Specially, the real test using Iridium-NEXT satellites shows that the proposed method estimates an certifiably optimal solution with an 3D positioning error of 140 m without initial estimates while Gauss-Newton and Dog-Leg are trapped in local optima when the initial point is equal or larger than 1000 km away from the ground truth. Moreover, the certifiable estimation can also be used as initialization in local searching methods to lower down the 3D positioning error to 130 m.
>
---
#### [new 014] Dynamic Objects Relocalization in Changing Environments with Flow Matching
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人任务与运动规划领域，旨在解决动态环境中因长期变化导致的对象重定位问题。作者提出FlowMaps模型，基于流匹配技术，用于推断时空中的多模态对象位置，从而提升机器人在变化环境中的任务执行能力。**

- **链接: [http://arxiv.org/pdf/2509.16398v1](http://arxiv.org/pdf/2509.16398v1)**

> **作者:** Francesco Argenziano; Miguel Saavedra-Ruiz; Sacha Morin; Daniele Nardi; Liam Paull
>
> **摘要:** Task and motion planning are long-standing challenges in robotics, especially when robots have to deal with dynamic environments exhibiting long-term dynamics, such as households or warehouses. In these environments, long-term dynamics mostly stem from human activities, since previously detected objects can be moved or removed from the scene. This adds the necessity to find such objects again before completing the designed task, increasing the risk of failure due to missed relocalizations. However, in these settings, the nature of such human-object interactions is often overlooked, despite being governed by common habits and repetitive patterns. Our conjecture is that these cues can be exploited to recover the most likely objects' positions in the scene, helping to address the problem of unknown relocalization in changing environments. To this end we propose FlowMaps, a model based on Flow Matching that is able to infer multimodal object locations over space and time. Our results present statistical evidence to support our hypotheses, opening the way to more complex applications of our approach. The code is publically available at https://github.com/Fra-Tsuna/flowmaps
>
---
#### [new 015] MotionTrans: Human VR Data Enable Motion-Level Learning for Robotic Manipulation Policies
- **分类: cs.RO**

- **简介: 该论文提出MotionTrans框架，通过多任务人机协同训练，探索人类VR数据在机器人操作策略中的运动级学习潜力。旨在解决机器人模仿学习中运动知识获取困难的问题，实现零样本任务迁移与性能提升。**

- **链接: [http://arxiv.org/pdf/2509.17759v1](http://arxiv.org/pdf/2509.17759v1)**

> **作者:** Chengbo Yuan; Rui Zhou; Mengzhen Liu; Yingdong Hu; Shengjie Wang; Li Yi; Chuan Wen; Shanghang Zhang; Yang Gao
>
> **摘要:** Scaling real robot data is a key bottleneck in imitation learning, leading to the use of auxiliary data for policy training. While other aspects of robotic manipulation such as image or language understanding may be learned from internet-based datasets, acquiring motion knowledge remains challenging. Human data, with its rich diversity of manipulation behaviors, offers a valuable resource for this purpose. While previous works show that using human data can bring benefits, such as improving robustness and training efficiency, it remains unclear whether it can realize its greatest advantage: enabling robot policies to directly learn new motions for task completion. In this paper, we systematically explore this potential through multi-task human-robot cotraining. We introduce MotionTrans, a framework that includes a data collection system, a human data transformation pipeline, and a weighted cotraining strategy. By cotraining 30 human-robot tasks simultaneously, we direcly transfer motions of 13 tasks from human data to deployable end-to-end robot policies. Notably, 9 tasks achieve non-trivial success rates in zero-shot manner. MotionTrans also significantly enhances pretraining-finetuning performance (+40% success rate). Through ablation study, we also identify key factors for successful motion learning: cotraining with robot data and broad task-related motion coverage. These findings unlock the potential of motion-level learning from human data, offering insights into its effective use for training robotic manipulation policies. All data, code, and model weights are open-sourced https://motiontrans.github.io/.
>
---
#### [new 016] No Need for Real 3D: Fusing 2D Vision with Pseudo 3D Representations for Robotic Manipulation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对视觉机械臂操作任务，旨在解决3D点云获取成本高的问题。提出NoReal3D框架，利用单目图像生成伪3D点云特征，并与2D特征融合，实现无需真实3D数据的高效操作学习。**

- **链接: [http://arxiv.org/pdf/2509.16532v1](http://arxiv.org/pdf/2509.16532v1)**

> **作者:** Run Yu; Yangdi Liu; Wen-Da Wei; Chen Li
>
> **摘要:** Recently,vision-based robotic manipulation has garnered significant attention and witnessed substantial advancements. 2D image-based and 3D point cloud-based policy learning represent two predominant paradigms in the field, with recent studies showing that the latter consistently outperforms the former in terms of both policy performance and generalization, thereby underscoring the value and significance of 3D information. However, 3D point cloud-based approaches face the significant challenge of high data acquisition costs, limiting their scalability and real-world deployment. To address this issue, we propose a novel framework NoReal3D: which introduces the 3DStructureFormer, a learnable 3D perception module capable of transforming monocular images into geometrically meaningful pseudo-point cloud features, effectively fused with the 2D encoder output features. Specially, the generated pseudo-point clouds retain geometric and topological structures so we design a pseudo-point cloud encoder to preserve these properties, making it well-suited for our framework. We also investigate the effectiveness of different feature fusion strategies.Our framework enhances the robot's understanding of 3D spatial structures while completely eliminating the substantial costs associated with 3D point cloud acquisition.Extensive experiments across various tasks validate that our framework can achieve performance comparable to 3D point cloud-based methods, without the actual point cloud data.
>
---
#### [new 017] AERO-MPPI: Anchor-Guided Ensemble Trajectory Optimization for Agile Mapless Drone Navigation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出AERO-MPPI，一种用于无人机敏捷无图导航的GPU加速框架。通过锚点引导的MPPI集成优化，实现感知与规划融合，在复杂三维环境中提升避障与路径规划能力，实现实时、鲁棒的高速飞行。**

- **链接: [http://arxiv.org/pdf/2509.17340v1](http://arxiv.org/pdf/2509.17340v1)**

> **作者:** Xin Chen; Rui Huang; Longbin Tang; Lin Zhao
>
> **摘要:** Agile mapless navigation in cluttered 3D environments poses significant challenges for autonomous drones. Conventional mapping-planning-control pipelines incur high computational cost and propagate estimation errors. We present AERO-MPPI, a fully GPU-accelerated framework that unifies perception and planning through an anchor-guided ensemble of Model Predictive Path Integral (MPPI) optimizers. Specifically, we design a multi-resolution LiDAR point-cloud representation that rapidly extracts spatially distributed "anchors" as look-ahead intermediate endpoints, from which we construct polynomial trajectory guides to explore distinct homotopy path classes. At each planning step, we run multiple MPPI instances in parallel and evaluate them with a two-stage multi-objective cost that balances collision avoidance and goal reaching. Implemented entirely with NVIDIA Warp GPU kernels, AERO-MPPI achieves real-time onboard operation and mitigates the local-minima failures of single-MPPI approaches. Extensive simulations in forests, verticals, and inclines demonstrate sustained reliable flight above 7 m/s, with success rates above 80% and smoother trajectories compared to state-of-the-art baselines. Real-world experiments on a LiDAR-equipped quadrotor with NVIDIA Jetson Orin NX 16G confirm that AERO-MPPI runs in real time onboard and consistently achieves safe, agile, and robust flight in complex cluttered environments. The code will be open-sourced upon acceptance of the paper.
>
---
#### [new 018] EigenSafe: A Spectral Framework for Learning-Based Stochastic Safety Filtering
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **简介: 该论文提出EigenSafe，一种基于谱理论的框架，用于学习随机系统的安全控制。针对传统方法难以全面衡量安全性的挑战，通过学习主导特征对和备份策略，构建安全过滤器以提升系统安全性，并在模拟任务中验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.17750v1](http://arxiv.org/pdf/2509.17750v1)**

> **作者:** Inkyu Jang; Jonghae Park; Chams E. Mballo; Sihyun Cho; Claire J. Tomlin; H. Jin Kim
>
> **备注:** Workshop on Safe and Robust Robot Learning for Operation in the Real World (SAFE-ROL) at CoRL 2025
>
> **摘要:** We present EigenSafe, an operator-theoretic framework for learning-enabled safety-critical control for stochastic systems. In many robotic systems where dynamics are best modeled as stochastic systems due to factors such as sensing noise and environmental disturbances, it is challenging for conventional methods such as Hamilton-Jacobi reachability and control barrier functions to provide a holistic measure of safety. We derive a linear operator governing the dynamic programming principle for safety probability, and find that its dominant eigenpair provides information about safety for both individual states and the overall closed-loop system. The proposed learning framework, called EigenSafe, jointly learns this dominant eigenpair and a safe backup policy in an offline manner. The learned eigenfunction is then used to construct a safety filter that detects potentially unsafe situations and falls back to the backup policy. The framework is validated in three simulated stochastic safety-critical control tasks.
>
---
#### [new 019] Guided Multi-Fidelity Bayesian Optimization for Data-driven Controller Tuning with Digital Twins
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种引导式多保真贝叶斯优化框架，用于数据驱动的控制器调参。通过结合修正的数字孪生模拟和真实测量，解决模型误差问题，提升调参效率。方法包括多保真代理模型、自适应采集函数和动态权重调整。**

- **链接: [http://arxiv.org/pdf/2509.17952v1](http://arxiv.org/pdf/2509.17952v1)**

> **作者:** Mahdi Nobar; Jürg Keller; Alessandro Forino; John Lygeros; Alisa Rupenyan
>
> **备注:** This preprint is intended for submission to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** We propose a \textit{guided multi-fidelity Bayesian optimization} framework for data-efficient controller tuning that integrates corrected digital twin (DT) simulations with real-world measurements. The method targets closed-loop systems with limited-fidelity simulations or inexpensive approximations. To address model mismatch, we build a multi-fidelity surrogate with a learned correction model that refines DT estimates from real data. An adaptive cost-aware acquisition function balances expected improvement, fidelity, and sampling cost. Our method ensures adaptability as new measurements arrive. The accuracy of DTs is re-estimated, dynamically adapting both cross-source correlations and the acquisition function. This ensures that accurate DTs are used more frequently, while inaccurate DTs are appropriately downweighted. Experiments on robotic drive hardware and supporting numerical studies demonstrate that our method enhances tuning efficiency compared to standard Bayesian optimization (BO) and multi-fidelity methods.
>
---
#### [new 020] Combining Performance and Passivity in Linear Control of Series Elastic Actuators
- **分类: cs.RO**

- **简介: 该论文研究了串联弹性执行器的线性控制问题，旨在平衡机器人与人交互时的安全性与性能。通过分析不同控制策略，提出在执行器侧使用PD控制器并结合阻尼器，可在保证被动安全性的同时提升运动精度和响应能力。**

- **链接: [http://arxiv.org/pdf/2509.17210v1](http://arxiv.org/pdf/2509.17210v1)**

> **作者:** Shaunak A. Mehta; Dylan P. Losey
>
> **摘要:** When humans physically interact with robots, we need the robots to be both safe and performant. Series elastic actuators (SEAs) fundamentally advance safety by introducing compliant actuation. On the one hand, adding a spring mitigates the impact of accidental collisions between human and robot; but on the other hand, this spring introduces oscillations and fundamentally decreases the robot's ability to perform precise, accurate motions. So how should we trade off between physical safety and performance? In this paper, we enumerate the different linear control and mechanical configurations for series elastic actuators, and explore how each choice affects the rendered compliance, passivity, and tracking performance. While prior works focus on load side control, we find that actuator side control has significant benefits. Indeed, simple PD controllers on the actuator side allow for a much wider range of control gains that maintain safety, and combining these with a damper in the elastic transmission yields high performance. Our simulations and real world experiments suggest that, by designing a system with low physical stiffness and high controller gains, this solution enables accurate performance while also ensuring user safety during collisions.
>
---
#### [new 021] Orchestrate, Generate, Reflect: A VLM-Based Multi-Agent Collaboration Framework for Automated Driving Policy Learning
- **分类: cs.RO**

- **简介: 该论文提出OGR框架，基于视觉语言模型（VLM）的多智能体协作，解决自动驾驶策略学习中奖励函数和训练课程手动设计的问题。通过自动化生成与优化，实现策略在线进化，提升驾驶技能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.17042v1](http://arxiv.org/pdf/2509.17042v1)**

> **作者:** Zengqi Peng; Yusen Xie; Yubin Wang; Rui Yang; Qifeng Chen; Jun Ma
>
> **摘要:** The advancement of foundation models fosters new initiatives for policy learning in achieving safe and efficient autonomous driving. However, a critical bottleneck lies in the manual engineering of reward functions and training curricula for complex and dynamic driving tasks, which is a labor-intensive and time-consuming process. To address this problem, we propose OGR (Orchestrate, Generate, Reflect), a novel automated driving policy learning framework that leverages vision-language model (VLM)-based multi-agent collaboration. Our framework capitalizes on advanced reasoning and multimodal understanding capabilities of VLMs to construct a hierarchical agent system. Specifically, a centralized orchestrator plans high-level training objectives, while a generation module employs a two-step analyze-then-generate process for efficient generation of reward-curriculum pairs. A reflection module then facilitates iterative optimization based on the online evaluation. Furthermore, a dedicated memory module endows the VLM agents with the capabilities of long-term memory. To enhance robustness and diversity of the generation process, we introduce a parallel generation scheme and a human-in-the-loop technique for augmentation of the reward observation space. Through efficient multi-agent cooperation and leveraging rich multimodal information, OGR enables the online evolution of reinforcement learning policies to acquire interaction-aware driving skills. Extensive experiments in the CARLA simulator demonstrate the superior performance, robust generalizability across distinct urban scenarios, and strong compatibility with various RL algorithms. Further real-world experiments highlight the practical viability and effectiveness of our framework. The source code will be available upon acceptance of the paper.
>
---
#### [new 022] RoboSeek: You Need to Interact with Your Objects
- **分类: cs.RO**

- **简介: 该论文提出RoboSeek框架，针对长时序机器人操作任务中交互驱动学习的不足。通过仿真到现实的迁移学习方法，结合视觉先验和强化学习训练策略，实现了多平台通用的高成功率操作执行，验证了框架在复杂环境下的鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.17783v1](http://arxiv.org/pdf/2509.17783v1)**

> **作者:** Yibo Peng; Jiahao Yang; Shenhao Yan; Ziyu Huang; Shuang Li; Shuguang Cui; Yiming Zhao; Yatong Han
>
> **摘要:** Optimizing and refining action execution through exploration and interaction is a promising way for robotic manipulation. However, practical approaches to interaction driven robotic learning are still underexplored, particularly for long-horizon tasks where sequential decision-making, physical constraints, and perceptual uncertainties pose significant chal lenges. Motivated by embodied cognition theory, we propose RoboSeek, a framework for embodied action execution that leverages interactive experience to accomplish manipulation tasks. RoboSeek optimizes prior knowledge from high-level perception models through closed-loop training in simulation and achieves robust real-world execution via a real2sim2real transfer pipeline. Specifically, we first replicate real-world environments in simulation using 3D reconstruction to provide visually and physically consistent environments., then we train policies in simulation using reinforcement learning and the cross-entropy method leveraging visual priors. The learned policies are subsequently deployed on real robotic platforms for execution. RoboSeek is hardware-agnostic and is evaluated on multiple robotic platforms across eight long-horizon ma nipulation tasks involving sequential interactions, tool use, and object handling. Our approach achieves an average success rate of 79%, significantly outperforming baselines whose success rates remain below 50%, highlighting its generalization and robustness across tasks and platforms. Experimental results validate the effectiveness of our training framework in complex, dynamic real-world settings and demonstrate the stability of the proposed real2sim2real transfer mechanism, paving the way for more generalizable embodied robotic learning. Project Page: https://russderrick.github.io/Roboseek/
>
---
#### [new 023] ComposableNav: Instruction-Following Navigation in Dynamic Environments via Composable Diffusion
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出ComposableNav，用于解决机器人在动态环境中遵循复杂指令导航的问题。针对指令组合爆炸的挑战，通过可组合扩散模型学习并组合不同运动基元，在未见过的指令组合下生成有效轨迹，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.17941v1](http://arxiv.org/pdf/2509.17941v1)**

> **作者:** Zichao Hu; Chen Tang; Michael J. Munje; Yifeng Zhu; Alex Liu; Shuijing Liu; Garrett Warnell; Peter Stone; Joydeep Biswas
>
> **备注:** Conference on Robot Learning (CoRL) 2025 Project site: https://amrl.cs.utexas.edu/ComposableNav/
>
> **摘要:** This paper considers the problem of enabling robots to navigate dynamic environments while following instructions. The challenge lies in the combinatorial nature of instruction specifications: each instruction can include multiple specifications, and the number of possible specification combinations grows exponentially as the robot's skill set expands. For example, "overtake the pedestrian while staying on the right side of the road" consists of two specifications: "overtake the pedestrian" and "walk on the right side of the road." To tackle this challenge, we propose ComposableNav, based on the intuition that following an instruction involves independently satisfying its constituent specifications, each corresponding to a distinct motion primitive. Using diffusion models, ComposableNav learns each primitive separately, then composes them in parallel at deployment time to satisfy novel combinations of specifications unseen in training. Additionally, to avoid the onerous need for demonstrations of individual motion primitives, we propose a two-stage training procedure: (1) supervised pre-training to learn a base diffusion model for dynamic navigation, and (2) reinforcement learning fine-tuning that molds the base model into different motion primitives. Through simulation and real-world experiments, we show that ComposableNav enables robots to follow instructions by generating trajectories that satisfy diverse and unseen combinations of specifications, significantly outperforming both non-compositional VLM-based policies and costmap composing baselines. Videos and additional materials can be found on the project page: https://amrl.cs.utexas.edu/ComposableNav/
>
---
#### [new 024] Imagine2Act: Leveraging Object-Action Motion Consistency from Imagined Goals for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出Imagine2Act，针对关系型物体重排任务（ROR），解决现有方法在语义几何推理和动作预测耦合上的不足。通过生成目标图像与点云，并引入对象-动作一致性策略，实现高精度的机器人操作。**

- **链接: [http://arxiv.org/pdf/2509.17125v1](http://arxiv.org/pdf/2509.17125v1)**

> **作者:** Liang Heng; Jiadong Xu; Yiwen Wang; Xiaoqi Li; Muhe Cai; Yan Shen; Juan Zhu; Guanghui Ren; Hao Dong
>
> **摘要:** Relational object rearrangement (ROR) tasks (e.g., insert flower to vase) require a robot to manipulate objects with precise semantic and geometric reasoning. Existing approaches either rely on pre-collected demonstrations that struggle to capture complex geometric constraints or generate goal-state observations to capture semantic and geometric knowledge, but fail to explicitly couple object transformation with action prediction, resulting in errors due to generative noise. To address these limitations, we propose Imagine2Act, a 3D imitation-learning framework that incorporates semantic and geometric constraints of objects into policy learning to tackle high-precision manipulation tasks. We first generate imagined goal images conditioned on language instructions and reconstruct corresponding 3D point clouds to provide robust semantic and geometric priors. These imagined goal point clouds serve as additional inputs to the policy model, while an object-action consistency strategy with soft pose supervision explicitly aligns predicted end-effector motion with generated object transformation. This design enables Imagine2Act to reason about semantic and geometric relationships between objects and predict accurate actions across diverse tasks. Experiments in both simulation and the real world demonstrate that Imagine2Act outperforms previous state-of-the-art policies. More visualizations can be found at https://sites.google.com/view/imagine2act.
>
---
#### [new 025] Pose Estimation of a Cable-Driven Serpentine Manipulator Utilizing Intrinsic Dynamics via Physical Reservoir Computing
- **分类: cs.RO**

- **简介: 该论文属于机器人姿态估计任务，旨在解决柔性电缆驱动蛇形机械臂因结构柔导致的位姿预测误差问题。工作包括设计轻量化9自由度机械臂，并提出基于物理储层计算的方法，利用其内在动力学提升位姿估计精度。**

- **链接: [http://arxiv.org/pdf/2509.17308v1](http://arxiv.org/pdf/2509.17308v1)**

> **作者:** Kazutoshi Tanaka; Tomoya Takahashi; Masashi Hamaya
>
> **备注:** 9 pages, 7 figures. Accepted at IROS 2025. This is the preprint version
>
> **摘要:** Cable-driven serpentine manipulators hold great potential in unstructured environments, offering obstacle avoidance, multi-directional force application, and a lightweight design. By placing all motors and sensors at the base and employing plastic links, we can further reduce the arm's weight. To demonstrate this concept, we developed a 9-degree-of-freedom cable-driven serpentine manipulator with an arm length of 545 mm and a total mass of only 308 g. However, this design introduces flexibility-induced variations, such as cable slack, elongation, and link deformation. These variations result in discrepancies between analytical predictions and actual link positions, making pose estimation more challenging. To address this challenge, we propose a physical reservoir computing based pose estimation method that exploits the manipulator's intrinsic nonlinear dynamics as a high-dimensional reservoir. Experimental results show a mean pose error of 4.3 mm using our method, compared to 4.4 mm with a baseline long short-term memory network and 39.5 mm with an analytical approach. This work provides a new direction for control and perception strategies in lightweight cable-driven serpentine manipulators leveraging their intrinsic dynamics.
>
---
#### [new 026] CoPlanner: An Interactive Motion Planner with Contingency-Aware Diffusion for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出CoPlanner，针对自动驾驶中的轨迹预测与运动规划问题，旨在解决现有方法在多模态不确定性下缺乏备选策略和交互一致性不足的问题。通过引入条件扩散机制和多场景评分策略，实现联合建模的轨迹生成与鲁棒运动规划。**

- **链接: [http://arxiv.org/pdf/2509.17080v1](http://arxiv.org/pdf/2509.17080v1)**

> **作者:** Ruiguo Zhong; Ruoyu Yao; Pei Liu; Xiaolong Chen; Rui Yang; Jun Ma
>
> **摘要:** Accurate trajectory prediction and motion planning are crucial for autonomous driving systems to navigate safely in complex, interactive environments characterized by multimodal uncertainties. However, current generation-then-evaluation frameworks typically construct multiple plausible trajectory hypotheses but ultimately adopt a single most likely outcome, leading to overconfident decisions and a lack of fallback strategies that are vital for safety in rare but critical scenarios. Moreover, the usual decoupling of prediction and planning modules could result in socially inconsistent or unrealistic joint trajectories, especially in highly interactive traffic. To address these challenges, we propose a contingency-aware diffusion planner (CoPlanner), a unified framework that jointly models multi-agent interactive trajectory generation and contingency-aware motion planning. Specifically, the pivot-conditioned diffusion mechanism anchors trajectory sampling on a validated, shared short-term segment to preserve temporal consistency, while stochastically generating diverse long-horizon branches that capture multimodal motion evolutions. In parallel, we design a contingency-aware multi-scenario scoring strategy that evaluates candidate ego trajectories across multiple plausible long-horizon evolution scenarios, balancing safety, progress, and comfort. This integrated design preserves feasible fallback options and enhances robustness under uncertainty, leading to more realistic interaction-aware planning. Extensive closed-loop experiments on the nuPlan benchmark demonstrate that CoPlanner consistently surpasses state-of-the-art methods on both Val14 and Test14 datasets, achieving significant improvements in safety and comfort under both reactive and non-reactive settings. Code and model will be made publicly available upon acceptance.
>
---
#### [new 027] LLM-Guided Task- and Affordance-Level Exploration in Reinforcement Learning
- **分类: cs.RO; 68T40; I.2.9; I.2.6; I.2.7**

- **简介: 该论文研究强化学习中的探索问题，提出LLM-TALE框架，利用大语言模型在任务和功能层面引导智能体探索，提升样本效率与成功率，并实现零样本仿真到现实的迁移。**

- **链接: [http://arxiv.org/pdf/2509.16615v1](http://arxiv.org/pdf/2509.16615v1)**

> **作者:** Jelle Luijkx; Runyu Ma; Zlatan Ajanović; Jens Kober
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Reinforcement learning (RL) is a promising approach for robotic manipulation, but it can suffer from low sample efficiency and requires extensive exploration of large state-action spaces. Recent methods leverage the commonsense knowledge and reasoning abilities of large language models (LLMs) to guide exploration toward more meaningful states. However, LLMs can produce plans that are semantically plausible yet physically infeasible, yielding unreliable behavior. We introduce LLM-TALE, a framework that uses LLMs' planning to directly steer RL exploration. LLM-TALE integrates planning at both the task level and the affordance level, improving learning efficiency by directing agents toward semantically meaningful actions. Unlike prior approaches that assume optimal LLM-generated plans or rewards, LLM-TALE corrects suboptimality online and explores multimodal affordance-level plans without human supervision. We evaluate LLM-TALE on pick-and-place tasks in standard RL benchmarks, observing improvements in both sample efficiency and success rates over strong baselines. Real-robot experiments indicate promising zero-shot sim-to-real transfer. Code and supplementary material are available at https://llm-tale.github.io.
>
---
#### [new 028] FILIC: Dual-Loop Force-Guided Imitation Learning with Impedance Torque Control for Contact-Rich Manipulation Tasks
- **分类: cs.RO; 68T40, 93C85; I.2.9**

- **简介: 该论文提出FILIC，针对接触密集型操作任务（如插入、装配），解决传统模仿学习缺乏力感知的问题。通过双环结构结合Transformer策略与阻抗控制，并设计低成本力估计与反馈框架，提升操作的顺应性与安全性。**

- **链接: [http://arxiv.org/pdf/2509.17053v1](http://arxiv.org/pdf/2509.17053v1)**

> **作者:** Haizhou Ge; Yufei Jia; Zheng Li; Yue Li; Zhixing Chen; Ruqi Huang; Guyue Zhou
>
> **摘要:** Contact-rich manipulation is crucial for robots to perform tasks requiring precise force control, such as insertion, assembly, and in-hand manipulation. However, most imitation learning (IL) policies remain position-centric and lack explicit force awareness, and adding force/torque sensors to collaborative robot arms is often costly and requires additional hardware design. To overcome these issues, we propose FILIC, a Force-guided Imitation Learning framework with impedance torque control. FILIC integrates a Transformer-based IL policy with an impedance controller in a dual-loop structure, enabling compliant force-informed, force-executed manipulation. For robots without force/torque sensors, we introduce a cost-effective end-effector force estimator using joint torque measurements through analytical Jacobian-based inversion while compensating with model-predicted torques from a digital twin. We also design complementary force feedback frameworks via handheld haptics and VR visualization to improve demonstration quality. Experiments show that FILIC significantly outperforms vision-only and joint-torque-based methods, achieving safer, more compliant, and adaptable contact-rich manipulation. Our code can be found in https://github.com/TATP-233/FILIC.
>
---
#### [new 029] A Reliable Robot Motion Planner in Complex Real-world Environments via Action Imagination
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种基于动作想象的机器人运动规划框架I-MP，旨在复杂真实环境中提高机器人动作可靠性。通过构建感知-动作循环和计算能量梯度，实现对未知环境的鲁棒运动规划。**

- **链接: [http://arxiv.org/pdf/2509.16963v1](http://arxiv.org/pdf/2509.16963v1)**

> **作者:** Chengjin Wang; Yanmin Zhou; Zhipeng Wang; Zheng Yan; Feng Luan; Shuo Jiang; Runjie Shen; Hongrui Sang; Bin He
>
> **摘要:** Humans and animals can make real-time adjustments to movements by imagining their action outcomes to prevent unanticipated or even catastrophic motion failures in unknown unstructured environments. Action imagination, as a refined sensorimotor strategy, leverages perception-action loops to handle physical interaction-induced uncertainties in perception and system modeling within complex systems. Inspired by the action-awareness capability of animal intelligence, this study proposes an imagination-inspired motion planner (I-MP) framework that specifically enhances robots' action reliability by imagining plausible spatial states for approaching. After topologizing the workspace, I-MP build perception-action loop enabling robots autonomously build contact models. Leveraging fixed-point theory and Hausdorff distance, the planner computes convergent spatial states under interaction characteristics and mission constraints. By homogenously representing multi-dimensional environmental characteristics through work, the robot can approach the imagined spatial states via real-time computation of energy gradients. Consequently, experimental results demonstrate the practicality and robustness of I-MP in complex cluttered environments.
>
---
#### [new 030] Morphologies of a sagging elastica with intrinsic sensing and actuation
- **分类: cs.RO; physics.app-ph**

- **简介: 该论文研究软体机器人形态控制问题，针对传感与驱动能力受限下的形状调整任务。通过建立弹性体模型，分析比例反馈策略对形态稳定性的影响，揭示了传感器分布与驱动参数之间的权衡关系，并提出了降低形态误差的设计方法。**

- **链接: [http://arxiv.org/pdf/2509.17572v1](http://arxiv.org/pdf/2509.17572v1)**

> **作者:** Vishnu Deo Mishra; S Ganga Prasath
>
> **摘要:** The morphology of a slender soft-robot can be modified by sensing its shape via sensors and exerting moments via actuators embedded along its body. The actuating moments required to morph these soft-robots to a desired shape are often difficult to compute due to the geometric non-linearity associated with the structure, the errors in modeling the experimental system, and the limitations in sensing and feedback/actuation capabilities. In this article, we explore the effect of a simple feedback strategy (actuation being proportional to the sensed curvature) on the shape of a soft-robot, modeled as an elastica. The finite number of sensors and actuators, often seen in experiments, is captured in the model via filters of specified widths. Using proportional feedback, we study the simple task of straightening the device by compensating for the sagging introduced by its self-weight. The device undergoes a hierarchy of morphological instabilities defined in the phase-space given by the gravito-bending number, non-dimensional sensing/feedback gain, and the scaled width of the filter. For complex shape-morphing tasks, given a perfect model of the device with limited sensing and actuating capabilities, we find that a trade-off arises (set by the sensor spacing & actuator size) between capturing the long and short wavelength features. We show that the error in shape-morphing is minimal for a fixed filter width when we choose an appropriate actuating gain (whose magnitude goes as a square of the filter width). Our model provides a quantitative lens to study and design slender soft devices with limited sensing and actuating capabilities for complex maneuvering applications.
>
---
#### [new 031] Prepare Before You Act: Learning From Humans to Rearrange Initial States
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于机器人模仿学习任务，旨在解决初始状态分布外时策略泛化性差的问题。提出ReSET算法，通过自主调整环境状态使其接近训练数据，从而提升任务执行的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.18043v1](http://arxiv.org/pdf/2509.18043v1)**

> **作者:** Yinlong Dai; Andre Keyser; Dylan P. Losey
>
> **摘要:** Imitation learning (IL) has proven effective across a wide range of manipulation tasks. However, IL policies often struggle when faced with out-of-distribution observations; for instance, when the target object is in a previously unseen position or occluded by other objects. In these cases, extensive demonstrations are needed for current IL methods to reach robust and generalizable behaviors. But when humans are faced with these sorts of atypical initial states, we often rearrange the environment for more favorable task execution. For example, a person might rotate a coffee cup so that it is easier to grasp the handle, or push a box out of the way so they can directly grasp their target object. In this work we seek to equip robot learners with the same capability: enabling robots to prepare the environment before executing their given policy. We propose ReSET, an algorithm that takes initial states -- which are outside the policy's distribution -- and autonomously modifies object poses so that the restructured scene is similar to training data. Theoretically, we show that this two step process (rearranging the environment before rolling out the given policy) reduces the generalization gap. Practically, our ReSET algorithm combines action-agnostic human videos with task-agnostic teleoperation data to i) decide when to modify the scene, ii) predict what simplifying actions a human would take, and iii) map those predictions into robot action primitives. Comparisons with diffusion policies, VLAs, and other baselines show that using ReSET to prepare the environment enables more robust task execution with equal amounts of total training data. See videos at our project website: https://reset2025paper.github.io/
>
---
#### [new 032] ORN-CBF: Learning Observation-conditioned Residual Neural Control Barrier Functions via Hypernetworks
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出ORN-CBF方法，通过超网络学习观测条件下的残差神经控制屏障函数，旨在解决自主系统在部分可观环境中安全控制的问题。利用HJ可达性分析提升安全性，并验证了方法在地面机器人和四旋翼上的有效性。**

- **链接: [http://arxiv.org/pdf/2509.16614v1](http://arxiv.org/pdf/2509.16614v1)**

> **作者:** Bojan Derajić; Sebastian Bernhard; Wolfgang Hönig
>
> **摘要:** Control barrier functions (CBFs) have been demonstrated as an effective method for safety-critical control of autonomous systems. Although CBFs are simple to deploy, their design remains challenging, motivating the development of learning-based approaches. Yet, issues such as suboptimal safe sets, applicability in partially observable environments, and lack of rigorous safety guarantees persist. In this work, we propose observation-conditioned neural CBFs based on Hamilton-Jacobi (HJ) reachability analysis, which approximately recover the maximal safe sets. We exploit certain mathematical properties of the HJ value function, ensuring that the predicted safe set never intersects with the observed failure set. Moreover, we leverage a hypernetwork-based architecture that is particularly suitable for the design of observation-conditioned safety filters. The proposed method is examined both in simulation and hardware experiments for a ground robot and a quadcopter. The results show improved success rates and generalization to out-of-domain environments compared to the baselines.
>
---
#### [new 033] Learning Dexterous Manipulation with Quantized Hand State
- **分类: cs.RO**

- **简介: 该论文针对灵巧机械手操作中臂-手动作耦合控制难的问题，提出DQ-RISE方法。通过量化手部状态简化动作空间，并结合连续松弛实现协调控制，提升了学习效率与平衡性，推动了结构化、泛化的灵巧操作研究。**

- **链接: [http://arxiv.org/pdf/2509.17450v1](http://arxiv.org/pdf/2509.17450v1)**

> **作者:** Ying Feng; Hongjie Fang; Yinong He; Jingjing Chen; Chenxi Wang; Zihao He; Ruonan Liu; Cewu Lu
>
> **摘要:** Dexterous robotic hands enable robots to perform complex manipulations that require fine-grained control and adaptability. Achieving such manipulation is challenging because the high degrees of freedom tightly couple hand and arm motions, making learning and control difficult. Successful dexterous manipulation relies not only on precise hand motions, but also on accurate spatial positioning of the arm and coordinated arm-hand dynamics. However, most existing visuomotor policies represent arm and hand actions in a single combined space, which often causes high-dimensional hand actions to dominate the coupled action space and compromise arm control. To address this, we propose DQ-RISE, which quantizes hand states to simplify hand motion prediction while preserving essential patterns, and applies a continuous relaxation that allows arm actions to diffuse jointly with these compact hand states. This design enables the policy to learn arm-hand coordination from data while preventing hand actions from overwhelming the action space. Experiments show that DQ-RISE achieves more balanced and efficient learning, paving the way toward structured and generalizable dexterous manipulation. Project website: http://rise-policy.github.io/DQ-RISE/
>
---
#### [new 034] GPS Denied IBVS-Based Navigation and Collision Avoidance of UAV Using a Low-Cost RGB Camera
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种基于RGB相机的IBVS框架，用于无人机在无GPS环境下的导航与避障。针对多视觉目标和避障难题，采用单目深度估计实现自主避障，并在Jetson平台实现全机载运行。**

- **链接: [http://arxiv.org/pdf/2509.17435v1](http://arxiv.org/pdf/2509.17435v1)**

> **作者:** Xiaoyu Wang; Yan Rui Tan; William Leong; Sunan Huang; Rodney Teo; Cheng Xiang
>
> **摘要:** This paper proposes an image-based visual servoing (IBVS) framework for UAV navigation and collision avoidance using only an RGB camera. While UAV navigation has been extensively studied, it remains challenging to apply IBVS in missions involving multiple visual targets and collision avoidance. The proposed method achieves navigation without explicit path planning, and collision avoidance is realized through AI-based monocular depth estimation from RGB images. Unlike approaches that rely on stereo cameras or external workstations, our framework runs fully onboard a Jetson platform, ensuring a self-contained and deployable system. Experimental results validate that the UAV can navigate across multiple AprilTags and avoid obstacles effectively in GPS-denied environments.
>
---
#### [new 035] TranTac: Leveraging Transient Tactile Signals for Contact-Rich Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文提出TranTac，一种低成本、数据高效的触觉感知与控制框架，用于解决机器人精细插入任务中视觉不足的问题。通过集成6轴IMU传感器和基于Transformer的策略，实现对微小位姿变化的感知与动态调整，提升操作成功率。**

- **链接: [http://arxiv.org/pdf/2509.16550v1](http://arxiv.org/pdf/2509.16550v1)**

> **作者:** Yinghao Wu; Shuhong Hou; Haowen Zheng; Yichen Li; Weiyi Lu; Xun Zhou; Yitian Shao
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Robotic manipulation tasks such as inserting a key into a lock or plugging a USB device into a port can fail when visual perception is insufficient to detect misalignment. In these situations, touch sensing is crucial for the robot to monitor the task's states and make precise, timely adjustments. Current touch sensing solutions are either insensitive to detect subtle changes or demand excessive sensor data. Here, we introduce TranTac, a data-efficient and low-cost tactile sensing and control framework that integrates a single contact-sensitive 6-axis inertial measurement unit within the elastomeric tips of a robotic gripper for completing fine insertion tasks. Our customized sensing system can detect dynamic translational and torsional deformations at the micrometer scale, enabling the tracking of visually imperceptible pose changes of the grasped object. By leveraging transformer-based encoders and diffusion policy, TranTac can imitate human insertion behaviors using transient tactile cues detected at the gripper's tip during insertion processes. These cues enable the robot to dynamically control and correct the 6-DoF pose of the grasped object. When combined with vision, TranTac achieves an average success rate of 79% on object grasping and insertion tasks, outperforming both vision-only policy and the one augmented with end-effector 6D force/torque sensing. Contact localization performance is also validated through tactile-only misaligned insertion tasks, achieving an average success rate of 88%. We assess the generalizability by training TranTac on a single prism-slot pair and testing it on unseen data, including a USB plug and a metal key, and find that the insertion tasks can still be completed with an average success rate of nearly 70%. The proposed framework may inspire new robotic tactile sensing systems for delicate manipulation tasks.
>
---
#### [new 036] Improve bounding box in Carla Simulator
- **分类: cs.RO; cs.GR**

- **简介: 该论文针对CARLA仿真器中目标检测任务，旨在解决因遮挡导致的“幽灵框”问题。通过改进边界框生成方法，过滤无效检测，提升检测准确性。**

- **链接: [http://arxiv.org/pdf/2509.16773v1](http://arxiv.org/pdf/2509.16773v1)**

> **作者:** Mohamad Mofeed Chaar; Jamal Raiyn; Galia Weidl
>
> **备注:** 9 pages, 12 figures,VEHITS Conference 2024
>
> **摘要:** The CARLA simulator (Car Learning to Act) serves as a robust platform for testing algorithms and generating datasets in the field of Autonomous Driving (AD). It provides control over various environmental parameters, enabling thorough evaluation. Development bounding boxes are commonly utilized tools in deep learning and play a crucial role in AD applications. The predominant method for data generation in the CARLA Simulator involves identifying and delineating objects of interest, such as vehicles, using bounding boxes. The operation in CARLA entails capturing the coordinates of all objects on the map, which are subsequently aligned with the sensor's coordinate system at the ego vehicle and then enclosed within bounding boxes relative to the ego vehicle's perspective. However, this primary approach encounters challenges associated with object detection and bounding box annotation, such as ghost boxes. Although these procedures are generally effective at detecting vehicles and other objects within their direct line of sight, they may also produce false positives by identifying objects that are obscured by obstructions. We have enhanced the primary approach with the objective of filtering out unwanted boxes. Performance analysis indicates that the improved approach has achieved high accuracy.
>
---
#### [new 037] Generalized Momenta-Based Koopman Formalism for Robust Control of Euler-Lagrangian Systems
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种基于广义动量的Koopman方法，用于欧拉-拉格朗日系统的鲁棒控制。通过解耦执行器与状态相关动力学，减少模型复杂度并提升数据效率，结合GESO增强鲁棒性，并通过神经网络架构实现高效建模。**

- **链接: [http://arxiv.org/pdf/2509.17010v1](http://arxiv.org/pdf/2509.17010v1)**

> **作者:** Rajpal Singh; Aditya Singh; Chidre Shravista Kashyap; Jishnu Keshavan
>
> **摘要:** This paper presents a novel Koopman operator formulation for Euler Lagrangian dynamics that employs an implicit generalized momentum-based state space representation, which decouples a known linear actuation channel from state dependent dynamics and makes the system more amenable to linear Koopman modeling. By leveraging this structural separation, the proposed formulation only requires to learn the unactuated dynamics rather than the complete actuation dependent system, thereby significantly reducing the number of learnable parameters, improving data efficiency, and lowering overall model complexity. In contrast, conventional explicit formulations inherently couple inputs with the state dependent terms in a nonlinear manner, making them more suitable for bilinear Koopman models, which are more computationally expensive to train and deploy. Notably, the proposed scheme enables the formulation of linear models that achieve superior prediction performance compared to conventional bilinear models while remaining substantially more efficient. To realize this framework, we present two neural network architectures that construct Koopman embeddings from actuated or unactuated data, enabling flexible and efficient modeling across different tasks. Robustness is ensured through the integration of a linear Generalized Extended State Observer (GESO), which explicitly estimates disturbances and compensates for them in real time. The combined momentum-based Koopman and GESO framework is validated through comprehensive trajectory tracking simulations and experiments on robotic manipulators, demonstrating superior accuracy, robustness, and learning efficiency relative to state of the art alternatives.
>
---
#### [new 038] Automated Coral Spawn Monitoring for Reef Restoration: The Coral Spawn and Larvae Imaging Camera System (CSLICS)
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出CSLICS系统，用于自动化监测珊瑚产卵和幼虫，解决人工计数劳动强度大、效率低的问题。通过低成本摄像头和目标检测技术实现自动计数，提升了珊瑚养殖效率与生态修复能力。**

- **链接: [http://arxiv.org/pdf/2509.17299v1](http://arxiv.org/pdf/2509.17299v1)**

> **作者:** Dorian Tsai; Christopher A. Brunner; Riki Lamont; F. Mikaela Nordborg; Andrea Severati; Java Terry; Karen Jackel; Matthew Dunbabin; Tobias Fischer; Scarlett Raine
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Coral aquaculture for reef restoration requires accurate and continuous spawn counting for resource distribution and larval health monitoring, but current methods are labor-intensive and represent a critical bottleneck in the coral production pipeline. We propose the Coral Spawn and Larvae Imaging Camera System (CSLICS), which uses low cost modular cameras and object detectors trained using human-in-the-loop labeling approaches for automated spawn counting in larval rearing tanks. This paper details the system engineering, dataset collection, and computer vision techniques to detect, classify and count coral spawn. Experimental results from mass spawning events demonstrate an F1 score of 82.4\% for surface spawn detection at different embryogenesis stages, 65.3\% F1 score for sub-surface spawn detection, and a saving of 5,720 hours of labor per spawning event compared to manual sampling methods at the same frequency. Comparison of manual counts with CSLICS monitoring during a mass coral spawning event on the Great Barrier Reef demonstrates CSLICS' accurate measurement of fertilization success and sub-surface spawn counts. These findings enhance the coral aquaculture process and enable upscaling of coral reef restoration efforts to address climate change threats facing ecosystems like the Great Barrier Reef.
>
---
#### [new 039] The Surprising Effectiveness of Linear Models for Whole-Body Model-Predictive Control
- **分类: cs.RO**

- **简介: 该论文研究了全身模型预测控制任务，探讨是否需要考虑非线性。提出使用线性时不变模型实现四足和双足机器人基本运动控制，无需在线计算非线性动力学，解决了复杂机器人运动规划问题。**

- **链接: [http://arxiv.org/pdf/2509.17884v1](http://arxiv.org/pdf/2509.17884v1)**

> **作者:** Arun L. Bishop; Juan Alvarez-Padilla; Sam Schoedel; Ibrahima Sory Sow; Juee Chandrachud; Sheitej Sharma; Will Kraus; Beomyeong Park; Robert J. Griffin; John M. Dolan; Zachary Manchester
>
> **备注:** Accepted to IEEE Humanoids 2025. For videos and code visit https://linearwalking.github.io/
>
> **摘要:** When do locomotion controllers require reasoning about nonlinearities? In this work, we show that a whole-body model-predictive controller using a simple linear time-invariant approximation of the whole-body dynamics is able to execute basic locomotion tasks on complex legged robots. The formulation requires no online nonlinear dynamics evaluations or matrix inversions. We demonstrate walking, disturbance rejection, and even navigation to a goal position without a separate footstep planner on a quadrupedal robot. In addition, we demonstrate dynamic walking on a hydraulic humanoid, a robot with significant limb inertia, complex actuator dynamics, and large sim-to-real gap.
>
---
#### [new 040] RaFD: Flow-Guided Radar Detection for Robust Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出RaFD，一种基于雷达的物体检测框架，旨在解决雷达图像噪声和伪影导致的检测难题。通过估计帧间BEV流并引导特征传播，提升检测精度，在RADIATE数据集上取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.16261v1](http://arxiv.org/pdf/2509.16261v1)**

> **作者:** Shuocheng Yang; Zikun Xu; Jiahao Wang; Shahid Nawaz; Jianqiang Wang; Shaobing Xu
>
> **摘要:** Radar has shown strong potential for robust perception in autonomous driving; however, raw radar images are frequently degraded by noise and "ghost" artifacts, making object detection based solely on semantic features highly challenging. To address this limitation, we introduce RaFD, a radar-based object detection framework that estimates inter-frame bird's-eye-view (BEV) flow and leverages the resulting geometric cues to enhance detection accuracy. Specifically, we design a supervised flow estimation auxiliary task that is jointly trained with the detection network. The estimated flow is further utilized to guide feature propagation from the previous frame to the current one. Our flow-guided, radar-only detector achieves achieves state-of-the-art performance on the RADIATE dataset, underscoring the importance of incorporating geometric information to effectively interpret radar signals, which are inherently ambiguous in semantics.
>
---
#### [new 041] ByteWrist: A Parallel Robotic Wrist Enabling Flexible and Anthropomorphic Motion for Confined Spaces
- **分类: cs.RO**

- **简介: 该论文提出ByteWrist，一种用于狭窄空间操作的高柔性并联机械腕。针对现有机械腕在狭小环境中的局限性，设计了三阶段并联驱动结构和弧形末端连杆，提升了灵活性与刚性，并实现了精准的姿态控制。**

- **链接: [http://arxiv.org/pdf/2509.18084v1](http://arxiv.org/pdf/2509.18084v1)**

> **作者:** Jiawen Tian; Liqun Huang; Zhongren Cui; Jingchao Qiao; Jiafeng Xu; Xiao Ma; Zeyu Ren
>
> **备注:** Tech Report.13 pages, 9 figures. Project page: https://bytewrist.github.io/
>
> **摘要:** This paper introduces ByteWrist, a novel highly-flexible and anthropomorphic parallel wrist for robotic manipulation. ByteWrist addresses the critical limitations of existing serial and parallel wrists in narrow-space operations through a compact three-stage parallel drive mechanism integrated with arc-shaped end linkages. The design achieves precise RPY (Roll-Pitch-Yaw) motion while maintaining exceptional compactness, making it particularly suitable for complex unstructured environments such as home services, medical assistance, and precision assembly. The key innovations include: (1) a nested three-stage motor-driven linkages that minimize volume while enabling independent multi-DOF control, (2) arc-shaped end linkages that optimize force transmission and expand motion range, and (3) a central supporting ball functioning as a spherical joint that enhances structural stiffness without compromising flexibility. Meanwhile, we present comprehensive kinematic modeling including forward / inverse kinematics and a numerical Jacobian solution for precise control. Empirically, we observe ByteWrist demonstrates strong performance in narrow-space maneuverability and dual-arm cooperative manipulation tasks, outperforming Kinova-based systems. Results indicate significant improvements in compactness, efficiency, and stiffness compared to traditional designs, establishing ByteWrist as a promising solution for next-generation robotic manipulation in constrained environments.
>
---
#### [new 042] RoboManipBaselines: A Unified Framework for Imitation Learning in Robotic Manipulation across Real and Simulated Environments
- **分类: cs.RO**

- **简介: 该论文提出RoboManipBaselines，一个统一的机器人模仿学习框架，用于在仿真与真实环境中进行数据收集、训练和评估。旨在解决任务多样性、机器人适配及政策泛化问题，强调集成性、通用性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.17057v1](http://arxiv.org/pdf/2509.17057v1)**

> **作者:** Masaki Murooka; Tomohiro Motoda; Ryoichi Nakajo; Hanbit Oh; Koshi Makihara; Keisuke Shirai; Yukiyasu Domae
>
> **摘要:** RoboManipBaselines is an open framework for robot imitation learning that unifies data collection, training, and evaluation across simulation and real robots. We introduce it as a platform enabling systematic benchmarking of diverse tasks, robots, and multimodal policies with emphasis on integration, generality, extensibility, and reproducibility.
>
---
#### [new 043] SocialTraj: Two-Stage Socially-Aware Trajectory Prediction for Autonomous Driving via Conditional Diffusion Model
- **分类: cs.RO**

- **简介: 该论文提出SocialTraj，用于自动驾驶中的轨迹预测任务。针对现有方法难以捕捉多模态驾驶行为的问题，结合社会价值导向和社会心理学原理，通过条件扩散模型生成更符合实际且具有一致性的轨迹预测。**

- **链接: [http://arxiv.org/pdf/2509.17850v1](http://arxiv.org/pdf/2509.17850v1)**

> **作者:** Xiao Zhou; Zengqi Peng; Jun Ma
>
> **摘要:** Accurate trajectory prediction of surrounding vehicles (SVs) is crucial for autonomous driving systems to avoid misguided decisions and potential accidents. However, achieving reliable predictions in highly dynamic and complex traffic scenarios remains a significant challenge. One of the key impediments lies in the limited effectiveness of current approaches to capture the multi-modal behaviors of drivers, which leads to predicted trajectories that deviate from actual future motions. To address this issue, we propose SocialTraj, a novel trajectory prediction framework integrating social psychology principles through social value orientation (SVO). By utilizing Bayesian inverse reinforcement learning (IRL) to estimate the SVO of SVs, we obtain the critical social context to infer the future interaction trend. To ensure modal consistency in predicted behaviors, the estimated SVOs of SVs are embedded into a conditional denoising diffusion model that aligns generated trajectories with historical driving styles. Additionally, the planned future trajectory of the ego vehicle (EV) is explicitly incorporated to enhance interaction modeling. Extensive experiments on NGSIM and HighD datasets demonstrate that SocialTraj is capable of adapting to highly dynamic and interactive scenarios while generating socially compliant and behaviorally consistent trajectory predictions, outperforming existing baselines. Ablation studies demonstrate that dynamic SVO estimation and explicit ego-planning components notably improve prediction accuracy and substantially reduce inference time.
>
---
#### [new 044] HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos
- **分类: cs.RO**

- **简介: 该论文提出HDMI框架，旨在解决人形机器人与物体交互任务中运动数据稀缺的问题。通过从单目视频中提取轨迹并训练强化学习策略，实现零样本部署，完成多种操作任务。属于模仿学习与机器人控制任务。**

- **链接: [http://arxiv.org/pdf/2509.16757v1](http://arxiv.org/pdf/2509.16757v1)**

> **作者:** Haoyang Weng; Yitang Li; Nikhil Sobanbabu; Zihan Wang; Zhengyi Luo; Tairan He; Deva Ramanan; Guanya Shi
>
> **备注:** website: hdmi-humanoid.github.io
>
> **摘要:** Enabling robust whole-body humanoid-object interaction (HOI) remains challenging due to motion data scarcity and the contact-rich nature. We present HDMI (HumanoiD iMitation for Interaction), a simple and general framework that learns whole-body humanoid-object interaction skills directly from monocular RGB videos. Our pipeline (i) extracts and retargets human and object trajectories from unconstrained videos to build structured motion datasets, (ii) trains a reinforcement learning (RL) policy to co-track robot and object states with three key designs: a unified object representation, a residual action space, and a general interaction reward, and (iii) zero-shot deploys the RL policies on real humanoid robots. Extensive sim-to-real experiments on a Unitree G1 humanoid demonstrate the robustness and generality of our approach: HDMI achieves 67 consecutive door traversals and successfully performs 6 distinct loco-manipulation tasks in the real world and 14 tasks in simulation. Our results establish HDMI as a simple and general framework for acquiring interactive humanoid skills from human videos.
>
---
#### [new 045] KungfuBot2: Learning Versatile Motion Skills for Humanoid Whole-Body Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出VMS，一种统一的人形机器人全身控制器，旨在学习多样化的运动技能。通过混合跟踪目标和OMoE架构，解决长序列稳定性与技能泛化问题，实现动态动作的稳定模仿与推广。**

- **链接: [http://arxiv.org/pdf/2509.16638v1](http://arxiv.org/pdf/2509.16638v1)**

> **作者:** Jinrui Han; Weiji Xie; Jiakun Zheng; Jiyuan Shi; Weinan Zhang; Ting Xiao; Chenjia Bai
>
> **摘要:** Learning versatile whole-body skills by tracking various human motions is a fundamental step toward general-purpose humanoid robots. This task is particularly challenging because a single policy must master a broad repertoire of motion skills while ensuring stability over long-horizon sequences. To this end, we present VMS, a unified whole-body controller that enables humanoid robots to learn diverse and dynamic behaviors within a single policy. Our framework integrates a hybrid tracking objective that balances local motion fidelity with global trajectory consistency, and an Orthogonal Mixture-of-Experts (OMoE) architecture that encourages skill specialization while enhancing generalization across motions. A segment-level tracking reward is further introduced to relax rigid step-wise matching, enhancing robustness when handling global displacements and transient inaccuracies. We validate VMS extensively in both simulation and real-world experiments, demonstrating accurate imitation of dynamic skills, stable performance over minute-long sequences, and strong generalization to unseen motions. These results highlight the potential of VMS as a scalable foundation for versatile humanoid whole-body control. The project page is available at https://kungfubot2-humanoid.github.io.
>
---
#### [new 046] Sight Over Site: Perception-Aware Reinforcement Learning for Efficient Robotic Inspection
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人自主巡检任务，旨在解决传统导航方式忽视视觉感知导致的低效问题。提出一种端到端的强化学习框架，以目标可见性为核心目标，使机器人找到最短可见路径。方法在仿真中训练，实现实验验证。**

- **链接: [http://arxiv.org/pdf/2509.17877v1](http://arxiv.org/pdf/2509.17877v1)**

> **作者:** Richard Kuhlmann; Jakob Wolfram; Boyang Sun; Jiaxu Xing; Davide Scaramuzza; Marc Pollefeys; Cesar Cadena
>
> **摘要:** Autonomous inspection is a central problem in robotics, with applications ranging from industrial monitoring to search-and-rescue. Traditionally, inspection has often been reduced to navigation tasks, where the objective is to reach a predefined location while avoiding obstacles. However, this formulation captures only part of the real inspection problem. In real-world environments, the inspection targets may become visible well before their exact coordinates are reached, making further movement both redundant and inefficient. What matters more for inspection is not simply arriving at the target's position, but positioning the robot at a viewpoint from which the target becomes observable. In this work, we revisit inspection from a perception-aware perspective. We propose an end-to-end reinforcement learning framework that explicitly incorporates target visibility as the primary objective, enabling the robot to find the shortest trajectory that guarantees visual contact with the target without relying on a map. The learned policy leverages both perceptual and proprioceptive sensing and is trained entirely in simulation, before being deployed to a real-world robot. We further develop an algorithm to compute ground-truth shortest inspection paths, which provides a reference for evaluation. Through extensive experiments, we show that our method outperforms existing classical and learning-based navigation approaches, yielding more efficient inspection trajectories in both simulated and real-world settings. The project is avialable at https://sight-over-site.github.io/
>
---
#### [new 047] Ratatouille: Imitation Learning Ingredients for Real-world Social Robot Navigation
- **分类: cs.RO**

- **简介: 该论文研究社交机器人导航任务，旨在解决强化学习数据需求大且不安全的问题。提出Ratatouille方法，通过改进模仿学习的架构和训练策略，在不增加数据的情况下显著提升导航的安全性和成功率。**

- **链接: [http://arxiv.org/pdf/2509.17204v1](http://arxiv.org/pdf/2509.17204v1)**

> **作者:** James R. Han; Mithun Vanniasinghe; Hshmat Sahak; Nicholas Rhinehart; Timothy D. Barfoot
>
> **备注:** 8 pages. Under review at ICRA 2026
>
> **摘要:** Scaling Reinforcement Learning to in-the-wild social robot navigation is both data-intensive and unsafe, since policies must learn through direct interaction and inevitably encounter collisions. Offline Imitation learning (IL) avoids these risks by collecting expert demonstrations safely, training entirely offline, and deploying policies zero-shot. However, we find that naively applying Behaviour Cloning (BC) to social navigation is insufficient; achieving strong performance requires careful architectural and training choices. We present Ratatouille, a pipeline and model architecture that, without changing the data, reduces collisions per meter by 6 times and improves success rate by 3 times compared to naive BC. We validate our approach in both simulation and the real world, where we collected over 11 hours of data on a dense university campus. We further demonstrate qualitative results in a public food court. Our findings highlight that thoughtful IL design, rather than additional data, can substantially improve safety and reliability in real-world social navigation. Video: https://youtu.be/tOdLTXsaYLQ. Code will be released after acceptance.
>
---
#### [new 048] OpenGVL - Benchmarking Visual Temporal Progress for Data Curation
- **分类: cs.RO; cs.CL**

- **简介: 该论文提出OpenGVL，用于评估视觉时序任务进度预测模型在机器人和人类操作任务中的表现。针对数据稀缺问题，研究对比开源与闭源基础模型性能，并展示其在自动化数据筛选中的应用。**

- **链接: [http://arxiv.org/pdf/2509.17321v1](http://arxiv.org/pdf/2509.17321v1)**

> **作者:** Paweł Budzianowski; Emilia Wiśnios; Gracjan Góral; Igor Kulakov; Viktor Petrenko; Krzysztof Walas
>
> **摘要:** Data scarcity remains one of the most limiting factors in driving progress in robotics. However, the amount of available robotics data in the wild is growing exponentially, creating new opportunities for large-scale data utilization. Reliable temporal task completion prediction could help automatically annotate and curate this data at scale. The Generative Value Learning (GVL) approach was recently proposed, leveraging the knowledge embedded in vision-language models (VLMs) to predict task progress from visual observations. Building upon GVL, we propose OpenGVL, a comprehensive benchmark for estimating task progress across diverse challenging manipulation tasks involving both robotic and human embodiments. We evaluate the capabilities of publicly available open-source foundation models, showing that open-source model families significantly underperform closed-source counterparts, achieving only approximately $70\%$ of their performance on temporal progress prediction tasks. Furthermore, we demonstrate how OpenGVL can serve as a practical tool for automated data curation and filtering, enabling efficient quality assessment of large-scale robotics datasets. We release the benchmark along with the complete codebase at \href{github.com/budzianowski/opengvl}{OpenGVL}.
>
---
#### [new 049] Robot Learning with Sparsity and Scarcity
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文聚焦机器人学习中的数据稀疏与稀缺问题，分别在触觉感知和康复机器人领域开展研究。针对触觉数据稀疏性，提出无视觉的强化学习策略；针对康复数据稀缺性，开发少样本意图推断算法，提升机器人在有限数据下的性能。**

- **链接: [http://arxiv.org/pdf/2509.16834v1](http://arxiv.org/pdf/2509.16834v1)**

> **作者:** Jingxi Xu
>
> **摘要:** Unlike in language or vision, one of the fundamental challenges in robot learning is the lack of access to vast data resources. We can further break down the problem into (1) data sparsity from the angle of data representation and (2) data scarcity from the angle of data quantity. In this thesis, I will discuss selected works on two domains: (1) tactile sensing and (2) rehabilitation robots, which are exemplars of data sparsity and scarcity, respectively. Tactile sensing is an essential modality for robotics, but tactile data are often sparse, and for each interaction with the physical world, tactile sensors can only obtain information about the local area of contact. I will discuss my work on learning vision-free tactile-only exploration and manipulation policies through model-free reinforcement learning to make efficient use of sparse tactile information. On the other hand, rehabilitation robots are an example of data scarcity to the extreme due to the significant challenge of collecting biosignals from disabled-bodied subjects at scale for training. I will discuss my work in collaboration with the medical school and clinicians on intent inferral for stroke survivors, where a hand orthosis developed in our lab collects a set of biosignals from the patient and uses them to infer the activity that the patient intends to perform, so the orthosis can provide the right type of physical assistance at the right moment. My work develops machine learning algorithms that enable intent inferral with minimal data, including semi-supervised, meta-learning, and generative AI methods.
>
---
#### [new 050] Substrate-Timing-Independence for Meta-State Stability of Distributed Robotic Swarms
- **分类: cs.RO**

- **简介: 该论文研究分布式机器人集群的元状态稳定性问题，提出一种与底层实现时序无关的形式化方法，利用并发进程演算自动识别并修正设计缺陷，确保在不同硬件变化下系统行为一致正确。**

- **链接: [http://arxiv.org/pdf/2509.16492v1](http://arxiv.org/pdf/2509.16492v1)**

> **作者:** Tinapat Limsila; Mehul Sharma; Paulo Garcia
>
> **摘要:** Emergent properties in distributed systems arise due to timing unpredictability; asynchronous state evolution within each sub-system may lead the macro-system to faulty meta-states. Empirical validation of correctness is often prohibitively expensive, as the size of the state-space is too large to be tractable. In robotic swarms this problem is exacerbated, when compared to software systems, by the variability of the implementation substrate across the design, or even the deployment, process. We present an approach for formally reasoning about the correctness of robotic swarm design in a substrate-timing-independent way. By leveraging concurrent process calculi (namely, Communicating Sequential Processes), we introduce a methodology that can automatically identify possible causes of faulty meta-states and correct such designs such that meta-states are consistently stable, even in the presence of timing variability due to substrate changes. We evaluate this approach on a robotic swarm with a clearly identified fault, realized in both simulation and reality. Results support the research hypothesis, showing that the swarm reaches an illegal meta-state before the correction is applied, but behaves consistently correctly after the correction. Our techniques are transferable across different design methodologies, contributing to the toolbox of formal methods for roboticists.
>
---
#### [new 051] IDfRA: Self-Verification for Iterative Design in Robotic Assembly
- **分类: cs.RO**

- **简介: 论文提出IDfRA框架，用于自动化机器人装配设计（DfRA）。传统方法依赖人工规划和物理仿真，效率低且不适用于复杂场景。IDfRA通过迭代规划、执行与自验证，结合语义理解与物理可行性，在真实环境中提升设计质量，实验证明其在语义识别和装配成功率上优于基线方法。**

- **链接: [http://arxiv.org/pdf/2509.16998v1](http://arxiv.org/pdf/2509.16998v1)**

> **作者:** Nishka Khendry; Christos Margadji; Sebastian W. Pattinson
>
> **摘要:** As robots proliferate in manufacturing, Design for Robotic Assembly (DfRA), which is designing products for efficient automated assembly, is increasingly important. Traditional approaches to DfRA rely on manual planning, which is time-consuming, expensive and potentially impractical for complex objects. Large language models (LLM) have exhibited proficiency in semantic interpretation and robotic task planning, stimulating interest in their application to the automation of DfRA. But existing methodologies typically rely on heuristic strategies and rigid, hard-coded physics simulators that may not translate into real-world assembly contexts. In this work, we present Iterative Design for Robotic Assembly (IDfRA), a framework using iterative cycles of planning, execution, verification, and re-planning, each informed by self-assessment, to progressively enhance design quality within a fixed yet initially under-specified environment, thereby eliminating the physics simulation with the real world itself. The framework accepts as input a target structure together with a partial environmental representation. Through successive refinement, it converges toward solutions that reconcile semantic fidelity with physical feasibility. Empirical evaluation demonstrates that IDfRA attains 73.3\% top-1 accuracy in semantic recognisability, surpassing the baseline on this metric. Moreover, the resulting assembly plans exhibit robust physical feasibility, achieving an overall 86.9\% construction success rate, with design quality improving across iterations, albeit not always monotonically. Pairwise human evaluation further corroborates the advantages of IDfRA relative to alternative approaches. By integrating self-verification with context-aware adaptation, the framework evidences strong potential for deployment in unstructured manufacturing scenarios.
>
---
#### [new 052] DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DriveDPO，用于端到端自动驾驶策略学习。针对模仿学习方法安全性不足的问题，通过安全评分与人类模仿的联合优化，实现轨迹级偏好对齐，提升驾驶安全性与可靠性，在NAVSIM基准上取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.17940v1](http://arxiv.org/pdf/2509.17940v1)**

> **作者:** Shuyao Shang; Yuntao Chen; Yuqi Wang; Yingyan Li; Zhaoxiang Zhang
>
> **备注:** NeurIPS 2025
>
> **摘要:** End-to-end autonomous driving has substantially progressed by directly predicting future trajectories from raw perception inputs, which bypasses traditional modular pipelines. However, mainstream methods trained via imitation learning suffer from critical safety limitations, as they fail to distinguish between trajectories that appear human-like but are potentially unsafe. Some recent approaches attempt to address this by regressing multiple rule-driven scores but decoupling supervision from policy optimization, resulting in suboptimal performance. To tackle these challenges, we propose DriveDPO, a Safety Direct Preference Optimization Policy Learning framework. First, we distill a unified policy distribution from human imitation similarity and rule-based safety scores for direct policy optimization. Further, we introduce an iterative Direct Preference Optimization stage formulated as trajectory-level preference alignment. Extensive experiments on the NAVSIM benchmark demonstrate that DriveDPO achieves a new state-of-the-art PDMS of 90.0. Furthermore, qualitative results across diverse challenging scenarios highlight DriveDPO's ability to produce safer and more reliable driving behaviors.
>
---
#### [new 053] Tactile-Based Human Intent Recognition for Robot Assistive Navigation
- **分类: cs.RO**

- **简介: 该论文研究机器人辅助导航任务，旨在解决现有系统界面不够直观的问题。提出Tac-Nav系统，结合圆柱形触觉皮肤与CK-SVM算法，提升对用户触觉意图的识别效果，并通过实验验证其优越性。**

- **链接: [http://arxiv.org/pdf/2509.16353v1](http://arxiv.org/pdf/2509.16353v1)**

> **作者:** Shaoting Peng; Dakarai Crowder; Wenzhen Yuan; Katherine Driggs-Campbell
>
> **摘要:** Robot assistive navigation (RAN) is critical for enhancing the mobility and independence of the growing population of mobility-impaired individuals. However, existing systems often rely on interfaces that fail to replicate the intuitive and efficient physical communication observed between a person and a human caregiver, limiting their effectiveness. In this paper, we introduce Tac-Nav, a RAN system that leverages a cylindrical tactile skin mounted on a Stretch 3 mobile manipulator to provide a more natural and efficient interface for human navigational intent recognition. To robustly classify the tactile data, we developed the Cylindrical Kernel Support Vector Machine (CK-SVM), an algorithm that explicitly models the sensor's cylindrical geometry and is consequently robust to the natural rotational shifts present in a user's grasp. Comprehensive experiments were conducted to demonstrate the effectiveness of our classification algorithm and the overall system. Results show that CK-SVM achieved superior classification accuracy on both simulated (97.1%) and real-world (90.8%) datasets compared to four baseline models. Furthermore, a pilot study confirmed that users more preferred the Tac-Nav tactile interface over conventional joystick and voice-based controls.
>
---
#### [new 054] End-to-end RL Improves Dexterous Grasping Policies
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究基于视觉的灵巧抓取策略的端到端强化学习。针对视觉RL训练效率低的问题，提出将模拟器与RL计算分离至不同GPU的方法，提升环境数量和训练效果，并验证了深度蒸馏在现实中的优越性。**

- **链接: [http://arxiv.org/pdf/2509.16434v1](http://arxiv.org/pdf/2509.16434v1)**

> **作者:** Ritvik Singh; Karl Van Wyk; Pieter Abbeel; Jitendra Malik; Nathan Ratliff; Ankur Handa
>
> **备注:** See our blog post: https://e2e4robotics.com/
>
> **摘要:** This work explores techniques to scale up image-based end-to-end learning for dexterous grasping with an arm + hand system. Unlike state-based RL, vision-based RL is much more memory inefficient, resulting in relatively low batch sizes, which is not amenable for algorithms like PPO. Nevertheless, it is still an attractive method as unlike the more commonly used techniques which distill state-based policies into vision networks, end-to-end RL can allow for emergent active vision behaviors. We identify a key bottleneck in training these policies is the way most existing simulators scale to multiple GPUs using traditional data parallelism techniques. We propose a new method where we disaggregate the simulator and RL (both training and experience buffers) onto separate GPUs. On a node with four GPUs, we have the simulator running on three of them, and PPO running on the fourth. We are able to show that with the same number of GPUs, we can double the number of existing environments compared to the previous baseline of standard data parallelism. This allows us to train vision-based environments, end-to-end with depth, which were previously performing far worse with the baseline. We train and distill both depth and state-based policies into stereo RGB networks and show that depth distillation leads to better results, both in simulation and reality. This improvement is likely due to the observability gap between state and vision policies which does not exist when distilling depth policies into stereo RGB. We further show that the increased batch size brought about by disaggregated simulation also improves real world performance. When deploying in the real world, we improve upon the previous state-of-the-art vision-based results using our end-to-end policies.
>
---
#### [new 055] End2Race: Efficient End-to-End Imitation Learning for Real-Time F1Tenth Racing
- **分类: cs.RO**

- **简介: 该论文针对F1Tenth自主赛车任务，提出End2Race算法，旨在解决高速实时决策与模型效率问题。采用GRU架构和LiDAR归一化方法，实现高效模仿学习，在超越场景中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.16894v1](http://arxiv.org/pdf/2509.16894v1)**

> **作者:** Zhijie Qiao; Haowei Li; Zhong Cao; Henry X. Liu
>
> **摘要:** F1Tenth is a widely adopted reduced-scale platform for developing and testing autonomous racing algorithms, hosting annual competitions worldwide. With high operating speeds, dynamic environments, and head-to-head interactions, autonomous racing requires algorithms that diverge from those in classical autonomous driving. Training such algorithms is particularly challenging: the need for rapid decision-making at high speeds severely limits model capacity. To address this, we propose End2Race, a novel end-to-end imitation learning algorithm designed for head-to-head autonomous racing. End2Race leverages a Gated Recurrent Unit (GRU) architecture to capture continuous temporal dependencies, enabling both short-term responsiveness and long-term strategic planning. We also adopt a sigmoid-based normalization function that transforms raw LiDAR scans into spatial pressure tokens, facilitating effective model training and convergence. The algorithm is extremely efficient, achieving an inference time of less than 0.5 milliseconds on a consumer-class GPU. Experiments in the F1Tenth simulator demonstrate that End2Race achieves a 94.2% safety rate across 2,400 overtaking scenarios, each with an 8-second time limit, and successfully completes overtakes in 59.2% of cases. This surpasses previous methods and establishes ours as a leading solution for the F1Tenth racing testbed. Code is available at https://github.com/michigan-traffic-lab/End2Race.
>
---
#### [new 056] Fast Trajectory Planner with a Reinforcement Learning-based Controller for Robotic Manipulators
- **分类: cs.RO**

- **简介: 该论文提出一种结合视觉路径规划与强化学习避障的快速机械臂轨迹规划系统，旨在解决非结构化环境中无碰撞轨迹生成的问题。工作包括：任务空间视觉轨迹规划和改进PPO算法以提升关节空间避障精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.17381v1](http://arxiv.org/pdf/2509.17381v1)**

> **作者:** Yongliang Wang; Hamidreza Kasaei
>
> **备注:** Project page available at: https://sites.google.com/view/ftp4rm/home
>
> **摘要:** Generating obstacle-free trajectories for robotic manipulators in unstructured and cluttered environments remains a significant challenge. Existing motion planning methods often require additional computational effort to generate the final trajectory by solving kinematic or dynamic equations. This paper highlights the strong potential of model-free reinforcement learning methods over model-based approaches for obstacle-free trajectory planning in joint space. We propose a fast trajectory planning system for manipulators that combines vision-based path planning in task space with reinforcement learning-based obstacle avoidance in joint space. We divide the framework into two key components. The first introduces an innovative vision-based trajectory planner in task space, leveraging the large-scale fast segment anything (FSA) model in conjunction with basis spline (B-spline)-optimized kinodynamic path searching. The second component enhances the proximal policy optimization (PPO) algorithm by integrating action ensembles (AE) and policy feedback (PF), which greatly improve precision and stability in goal-reaching and obstacle avoidance within the joint space. These PPO enhancements increase the algorithm's adaptability across diverse robotic tasks, ensuring consistent execution of commands from the first component by the manipulator, while also enhancing both obstacle avoidance efficiency and reaching accuracy. The experimental results demonstrate the effectiveness of PPO enhancements, as well as simulation-to-simulation (Sim-to-Sim) and simulation-to-reality (Sim-to-Real) transfer, in improving model robustness and planner efficiency in complex scenarios. These enhancements allow the robot to perform obstacle avoidance and real-time trajectory planning in obstructed environments. Project page available at: https://sites.google.com/view/ftp4rm/home
>
---
#### [new 057] Event-Based Visual Teach-and-Repeat via Fast Fourier-Domain Cross-Correlation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出首个基于事件相机的视觉教-复现导航系统，解决传统帧相机因固定帧率导致的响应延迟问题。通过频域互相关方法实现高速（>300Hz）事件流匹配，提升机器人实时定位与导航性能。**

- **链接: [http://arxiv.org/pdf/2509.17287v1](http://arxiv.org/pdf/2509.17287v1)**

> **作者:** Gokul B. Nair; Alejandro Fontan; Michael Milford; Tobias Fischer
>
> **备注:** 8 Pages, 4 Figures, Under Review
>
> **摘要:** Visual teach-and-repeat navigation enables robots to autonomously traverse previously demonstrated paths by comparing current sensory input with recorded trajectories. However, conventional frame-based cameras fundamentally limit system responsiveness: their fixed frame rates (typically 30-60 Hz) create inherent latency between environmental changes and control responses. Here we present the first event-camera-based visual teach-and-repeat system. To achieve this, we develop a frequency-domain cross-correlation framework that transforms the event stream matching problem into computationally efficient Fourier space multiplications, capable of exceeding 300Hz processing rates, an order of magnitude faster than frame-based approaches. By exploiting the binary nature of event frames and applying image compression techniques, we further enhance the computational speed of the cross-correlation process without sacrificing localization accuracy. Extensive experiments using a Prophesee EVK4 HD event camera mounted on an AgileX Scout Mini robot demonstrate successful autonomous navigation across 4000+ meters of indoor and outdoor trajectories. Our system achieves ATEs below 24 cm while maintaining consistent high-frequency control updates. Our evaluations show that our approach achieves substantially higher update rates compared to conventional frame-based systems, underscoring the practical viability of event-based perception for real-time robotic navigation.
>
---
#### [new 058] Towards Learning Boulder Excavation with Hydraulic Excavators
- **分类: cs.RO**

- **简介: 该论文研究液压挖掘机自动挖掘不规则大石块的任务，旨在解决户外复杂环境下使用标准铲斗高效提取岩石的问题。通过强化学习训练策略，在模拟中结合土壤模型和稀疏LiDAR数据，实现在真实场景中70%的成功率，接近人类操作水平。**

- **链接: [http://arxiv.org/pdf/2509.17683v1](http://arxiv.org/pdf/2509.17683v1)**

> **作者:** Jonas Gruetter; Lorenzo Terenzi; Pascal Egli; Marco Hutter
>
> **摘要:** Construction sites frequently require removing large rocks before excavation or grading can proceed. Human operators typically extract these boulders using only standard digging buckets, avoiding time-consuming tool changes to specialized grippers. This task demands manipulating irregular objects with unknown geometries in harsh outdoor environments where dust, variable lighting, and occlusions hinder perception. The excavator must adapt to varying soil resistance--dragging along hard-packed surfaces or penetrating soft ground--while coordinating multiple hydraulic joints to secure rocks using a shovel. Current autonomous excavation focuses on continuous media (soil, gravel) or uses specialized grippers with detailed geometric planning for discrete objects. These approaches either cannot handle large irregular rocks or require impractical tool changes that interrupt workflow. We train a reinforcement learning policy in simulation using rigid-body dynamics and analytical soil models. The policy processes sparse LiDAR points (just 20 per rock) from vision-based segmentation and proprioceptive feedback to control standard excavator buckets. The learned agent discovers different strategies based on soil resistance: dragging along the surface in hard soil and penetrating directly in soft conditions. Field tests on a 12-ton excavator achieved 70% success across varied rocks (0.4-0.7m) and soil types, compared to 83% for human operators. This demonstrates that standard construction equipment can learn complex manipulation despite sparse perception and challenging outdoor conditions.
>
---
#### [new 059] Tac2Motion: Contact-Aware Reinforcement Learning with Tactile Feedback for Robotic Hand Manipulation
- **分类: cs.RO**

- **简介: 该论文提出Tac2Motion，一种结合触觉反馈的强化学习框架，用于解决机械手精细操作任务（如开盖）。通过触觉感知奖励设计与嵌入，提升抓取稳定性和操作效率，并在仿真和真实机械手上验证了方法的有效性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.17812v1](http://arxiv.org/pdf/2509.17812v1)**

> **作者:** Yitaek Kim; Casper Hewson Rask; Christoffer Sloth
>
> **备注:** This paper has submitted to Dexterous Humanoid Manipulation Workshop, Humanoid 2025
>
> **摘要:** This paper proposes Tac2Motion, a contact-aware reinforcement learning framework to facilitate the learning of contact-rich in-hand manipulation tasks, such as removing a lid. To this end, we propose tactile sensing-based reward shaping and incorporate the sensing into the observation space through embedding. The designed rewards encourage an agent to ensure firm grasping and smooth finger gaiting at the same time, leading to higher data efficiency and robust performance compared to the baseline. We verify the proposed framework on the opening a lid scenario, showing generalization of the trained policy into a couple of object types and various dynamics such as torsional friction. Lastly, the learned policy is demonstrated on the multi-fingered robot, Shadow Robot, showing that the control policy can be transferred to the real world. The video is available: https://youtu.be/poeJBPR7urQ.
>
---
#### [new 060] Enhancing the NAO: Extending Capabilities of Legacy Robots for Long-Term Research
- **分类: cs.RO; cs.HC; eess.AS**

- **简介: 该论文提出Enhanced NAO，通过升级传感器和计算资源，提升旧版NAO机器人的感知与交互能力。任务是延长机器人研究寿命，解决老旧平台不支持现代技术的问题。工作包括硬件升级、多模态感知与对话系统优化，验证了其交互性能的显著提升。**

- **链接: [http://arxiv.org/pdf/2509.17760v1](http://arxiv.org/pdf/2509.17760v1)**

> **作者:** Austin Wilson; Sahar Kapasi; Zane Greene; Alexis E. Block
>
> **摘要:** Many research groups face challenges when legacy (unsupported) robotic platforms lose manufacturer support and cannot accommodate modern sensing, speech, and interaction capabilities. We present the Enhanced NAO, a revitalized version of Aldebaran's NAO robot that uses upgraded microphones, RGB-D and thermal cameras, and additional compute resources in a fully self-contained package. This system combines cloud and local models for perception and dialogue, while preserving the NAO's expressive body and behaviors. In a pilot validation study, the Enhanced NAO delivered significantly higher conversational quality and stronger user preference compared to the NAO AI Edition, without increasing response latency. Key upgrades, such as beamforming microphones and low-latency audio processing, reduced artifacts like self-hearing and improved multi-party separation. Expanded visual and thermal sensing established a foundation for future interaction capabilities. Beyond the NAO, our framework provides a platform-agnostic strategy for extending the lifespan and research utility of legacy robots, ensuring they remain valuable tools for human-robot interaction.
>
---
#### [new 061] FiLM-Nav: Efficient and Generalizable Navigation via VLM Fine-tuning
- **分类: cs.RO**

- **简介: 该论文提出FiLM-Nav，通过微调预训练视觉语言模型（VLM）实现导航任务。旨在解决机器人在复杂环境中根据自然语言描述定位物体的问题。方法直接利用视觉轨迹和目标信息选择探索方向，结合多任务数据提升泛化能力，在多个导航基准上取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.16445v1](http://arxiv.org/pdf/2509.16445v1)**

> **作者:** Naoki Yokoyama; Sehoon Ha
>
> **摘要:** Enabling robotic assistants to navigate complex environments and locate objects described in free-form language is a critical capability for real-world deployment. While foundation models, particularly Vision-Language Models (VLMs), offer powerful semantic understanding, effectively adapting their web-scale knowledge for embodied decision-making remains a key challenge. We present FiLM-Nav (Fine-tuned Language Model for Navigation), an approach that directly fine-tunes pre-trained VLM as the navigation policy. In contrast to methods that use foundation models primarily in a zero-shot manner or for map annotation, FiLM-Nav learns to select the next best exploration frontier by conditioning directly on raw visual trajectory history and the navigation goal. Leveraging targeted simulated embodied experience allows the VLM to ground its powerful pre-trained representations in the specific dynamics and visual patterns relevant to goal-driven navigation. Critically, fine-tuning on a diverse data mixture combining ObjectNav, OVON, ImageNav, and an auxiliary spatial reasoning task proves essential for achieving robustness and broad generalization. FiLM-Nav sets a new state-of-the-art in both SPL and success rate on HM3D ObjectNav among open-vocabulary methods, and sets a state-of-the-art SPL on the challenging HM3D-OVON benchmark, demonstrating strong generalization to unseen object categories. Our work validates that directly fine-tuning VLMs on diverse simulated embodied data is a highly effective pathway towards generalizable and efficient semantic navigation capabilities.
>
---
#### [new 062] HOGraspFlow: Exploring Vision-based Generative Grasp Synthesis with Hand-Object Priors and Taxonomy Awareness
- **分类: cs.RO**

- **简介: 该论文提出HOGraspFlow，用于基于单张RGB图像生成多模态抓取姿态。任务是视觉驱动的抓取合成，无需目标几何先验。通过结合手-物交互重建和类别感知先验，实现了高精度、高成功率的无监督抓取生成。**

- **链接: [http://arxiv.org/pdf/2509.16871v1](http://arxiv.org/pdf/2509.16871v1)**

> **作者:** Yitian Shi; Zicheng Guo; Rosa Wolf; Edgar Welte; Rania Rayyes
>
> **备注:** under review
>
> **摘要:** We propose Hand-Object\emph{(HO)GraspFlow}, an affordance-centric approach that retargets a single RGB with hand-object interaction (HOI) into multi-modal executable parallel jaw grasps without explicit geometric priors on target objects. Building on foundation models for hand reconstruction and vision, we synthesize $SE(3)$ grasp poses with denoising flow matching (FM), conditioned on the following three complementary cues: RGB foundation features as visual semantics, HOI contact reconstruction, and taxonomy-aware prior on grasp types. Our approach demonstrates high fidelity in grasp synthesis without explicit HOI contact input or object geometry, while maintaining strong contact and taxonomy recognition. Another controlled comparison shows that \emph{HOGraspFlow} consistently outperforms diffusion-based variants (\emph{HOGraspDiff}), achieving high distributional fidelity and more stable optimization in $SE(3)$. We demonstrate a reliable, object-agnostic grasp synthesis from human demonstrations in real-world experiments, where an average success rate of over $83\%$ is achieved.
>
---
#### [new 063] High-Precision and High-Efficiency Trajectory Tracking for Excavators Based on Closed-Loop Dynamics
- **分类: cs.RO**

- **简介: 该论文针对挖掘机轨迹跟踪任务，旨在解决液压系统非线性动态带来的高精度跟踪难题。提出EfficientTrack方法，结合模型学习与闭环动力学，提升跟踪精度与效率，并通过仿真与实验证明其优越性。**

- **链接: [http://arxiv.org/pdf/2509.17387v1](http://arxiv.org/pdf/2509.17387v1)**

> **作者:** Ziqing Zou; Cong Wang; Yue Hu; Xiao Liu; Bowen Xu; Rong Xiong; Changjie Fan; Yingfeng Chen; Yue Wang
>
> **摘要:** The complex nonlinear dynamics of hydraulic excavators, such as time delays and control coupling, pose significant challenges to achieving high-precision trajectory tracking. Traditional control methods often fall short in such applications due to their inability to effectively handle these nonlinearities, while commonly used learning-based methods require extensive interactions with the environment, leading to inefficiency. To address these issues, we introduce EfficientTrack, a trajectory tracking method that integrates model-based learning to manage nonlinear dynamics and leverages closed-loop dynamics to improve learning efficiency, ultimately minimizing tracking errors. We validate our method through comprehensive experiments both in simulation and on a real-world excavator. Comparative experiments in simulation demonstrate that our method outperforms existing learning-based approaches, achieving the highest tracking precision and smoothness with the fewest interactions. Real-world experiments further show that our method remains effective under load conditions and possesses the ability for continual learning, highlighting its practical applicability. For implementation details and source code, please refer to https://github.com/ZiqingZou/EfficientTrack.
>
---
#### [new 064] SwarmChat: An LLM-Based, Context-Aware Multimodal Interaction System for Robotic Swarms
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出SwarmChat，一个基于大语言模型的多模态人机交互系统，用于机器人集群控制。旨在解决传统方法界面不直观、响应慢的问题，通过自然语言和多模态输入实现灵活、实时的群机器人交互与任务规划。**

- **链接: [http://arxiv.org/pdf/2509.16920v1](http://arxiv.org/pdf/2509.16920v1)**

> **作者:** Ettilla Mohiuddin Eumi; Hussein Abbass; Nadine Marcus
>
> **备注:** This paper has been accepted and presented at the 16th International Conference on Swarm Intelligence (ICSI 2025), held on July 11-15, 2025, in Yokohama, Japan
>
> **摘要:** Traditional Human-Swarm Interaction (HSI) methods often lack intuitive real-time adaptive interfaces, making decision making slower and increasing cognitive load while limiting command flexibility. To solve this, we present SwarmChat, a context-aware, multimodal interaction system powered by Large Language Models (LLMs). SwarmChat enables users to issue natural language commands to robotic swarms using multiple modalities, such as text, voice, or teleoperation. The system integrates four LLM-based modules: Context Generator, Intent Recognition, Task Planner, and Modality Selector. These modules collaboratively generate context from keywords, detect user intent, adapt commands based on real-time robot state, and suggest optimal communication modalities. Its three-layer architecture offers a dynamic interface with both fixed and customizable command options, supporting flexible control while optimizing cognitive effort. The preliminary evaluation also shows that the SwarmChat's LLM modules provide accurate context interpretation, relevant intent recognition, and effective command delivery, achieving high user satisfaction.
>
---
#### [new 065] A Framework for Optimal Ankle Design of Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文提出一种人形机器人踝部设计的优化框架，旨在解决并联机构配置选择问题。通过多目标优化方法，比较了SPU和RSU两种架构，验证了优化后的RSU性能优于传统设计。**

- **链接: [http://arxiv.org/pdf/2509.16469v1](http://arxiv.org/pdf/2509.16469v1)**

> **作者:** Guglielmo Cervettini; Roberto Mauceri; Alex Coppola; Fabio Bergonti; Luca Fiorio; Marco Maggiali; Daniele Pucci
>
> **备注:** This paper has been accepted for publication at the 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids), Seoul, 2025
>
> **摘要:** The design of the humanoid ankle is critical for safe and efficient ground interaction. Key factors such as mechanical compliance and motor mass distribution have driven the adoption of parallel mechanism architectures. However, selecting the optimal configuration depends on both actuator availability and task requirements. We propose a unified methodology for the design and evaluation of parallel ankle mechanisms. A multi-objective optimization synthesizes the mechanism geometry, the resulting solutions are evaluated using a scalar cost function that aggregates key performance metrics for cross-architecture comparison. We focus on two representative architectures: the Spherical-Prismatic-Universal (SPU) and the Revolute-Spherical-Universal (RSU). For both, we resolve the kinematics, and for the RSU, introduce a parameterization that ensures workspace feasibility and accelerates optimization. We validate our approach by redesigning the ankle of an existing humanoid robot. The optimized RSU consistently outperforms both the original serial design and a conventionally engineered RSU, reducing the cost function by up to 41% and 14%, respectively.
>
---
#### [new 066] Benchmarking Offline Reinforcement Learning for Emotion-Adaptive Social Robotics
- **分类: cs.RO**

- **简介: 该论文研究了用于情感自适应社交机器人的离线强化学习方法，旨在解决在线数据收集成本高和行为不安全的问题。作者构建了一个基准测试系统，对比了多种算法性能，发现BCQ和CQL在数据稀疏情况下表现更优。**

- **链接: [http://arxiv.org/pdf/2509.16858v1](http://arxiv.org/pdf/2509.16858v1)**

> **作者:** Soon Jynn Chu; Raju Gottumukkala; Alan Barhorst
>
> **备注:** Submitted to conference
>
> **摘要:** The ability of social robots to respond to human emotions is crucial for building trust and acceptance in human-robot collaborative environments. However, developing such capabilities through online reinforcement learning is sometimes impractical due to the prohibitive cost of data collection and the risk of generating unsafe behaviors. In this paper, we study the use of offline reinforcement learning as a practical and efficient alternative. This technique uses pre-collected data to enable emotion-adaptive social robots. We present a system architecture that integrates multimodal sensing and recognition, decision-making, and adaptive responses. Using a limited dataset from a human-robot game-playing scenario, we establish a benchmark for comparing offline reinforcement learning algorithms that do not require an online environment. Our results show that BCQ and CQL are more robust to data sparsity, achieving higher state-action values compared to NFQ, DQN, and DDQN. This work establishes a foundation for benchmarking offline RL in emotion-adaptive robotics and informs future deployment in real-world HRI. Our findings provide empirical insight into the performance of offline reinforcement learning algorithms in data-constrained HRI. This work establishes a foundation for benchmarking offline RL in emotion-adaptive robotics and informs its future deployment in real-world HRI, such as in conversational agents, educational partners, and personal assistants, require reliable emotional responsiveness.
>
---
#### [new 067] 3D Printable Soft Liquid Metal Sensors for Delicate Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文提出了一种可3D打印的软液态金属传感器，用于精细操作任务。针对易碎生态样本（如珊瑚）的无损操作与数据采集问题，开发了高保真“传感珊瑚”物理孪生体，并展示了其在自动识别和水下抓取中的应用。**

- **链接: [http://arxiv.org/pdf/2509.17389v1](http://arxiv.org/pdf/2509.17389v1)**

> **作者:** Lois Liow; Jonty Milford; Emre Uygun; Andre Farinha; Vinoth Viswanathan; Josh Pinskier; David Howard
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Robotics and automation are key enablers to increase throughput in ongoing conservation efforts across various threatened ecosystems. Cataloguing, digitisation, husbandry, and similar activities require the ability to interact with delicate, fragile samples without damaging them. Additionally, learning-based solutions to these tasks require the ability to safely acquire data to train manipulation policies through, e.g., reinforcement learning. To address these twin needs, we introduce a novel method to print free-form, highly sensorised soft 'physical twins'. We present an automated design workflow to create complex and customisable 3D soft sensing structures on demand from 3D scans or models. Compared to the state of the art, our soft liquid metal sensors faithfully recreate complex natural geometries and display excellent sensing properties suitable for validating performance in delicate manipulation tasks. We demonstrate the application of our physical twins as 'sensing corals': high-fidelity, 3D printed replicas of scanned corals that eliminate the need for live coral experimentation, whilst increasing data quality, offering an ethical and scalable pathway for advancing autonomous coral handling and soft manipulation broadly. Through extensive bench-top manipulation and underwater grasping experiments, we show that our sensing coral is able to detect grasps under 0.5 N, effectively capturing the delicate interactions and light contact forces required for coral handling. Finally, we showcase the value of our physical twins across two demonstrations: (i) automated coral labelling for lab identification and (ii) robotic coral aquaculture. Sensing physical twins such as ours can provide richer grasping feedback than conventional sensors providing experimental validation of prior to deployment in handling fragile and delicate items.
>
---
#### [new 068] History-Aware Visuomotor Policy Learning via Point Tracking
- **分类: cs.RO**

- **简介: 该论文针对视觉运动策略在长期依赖和重复状态下的不足，提出一种基于点跟踪的对象中心历史表示方法。通过结构化压缩历史信息，有效提升策略的记忆能力与任务性能。**

- **链接: [http://arxiv.org/pdf/2509.17141v1](http://arxiv.org/pdf/2509.17141v1)**

> **作者:** Jingjing Chen; Hongjie Fang; Chenxi Wang; Shiquan Wang; Cewu Lu
>
> **摘要:** Many manipulation tasks require memory beyond the current observation, yet most visuomotor policies rely on the Markov assumption and thus struggle with repeated states or long-horizon dependencies. Existing methods attempt to extend observation horizons but remain insufficient for diverse memory requirements. To this end, we propose an object-centric history representation based on point tracking, which abstracts past observations into a compact and structured form that retains only essential task-relevant information. Tracked points are encoded and aggregated at the object level, yielding a compact history representation that can be seamlessly integrated into various visuomotor policies. Our design provides full history-awareness with high computational efficiency, leading to improved overall task performance and decision accuracy. Through extensive evaluations on diverse manipulation tasks, we show that our method addresses multiple facets of memory requirements - such as task stage identification, spatial memorization, and action counting, as well as longer-term demands like continuous and pre-loaded memory - and consistently outperforms both Markovian baselines and prior history-based approaches. Project website: http://tonyfang.net/history
>
---
#### [new 069] MAST: Multi-Agent Spatial Transformer for Learning to Collaborate
- **分类: cs.RO**

- **简介: 该论文提出MAST，一种用于多机器人协作的多智能体空间变换器。针对去中心化协作系统中观测局部、通信受限的问题，设计了具有位置编码和窗口注意力机制的通信策略，提升了大规模团队任务执行效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.17195v1](http://arxiv.org/pdf/2509.17195v1)**

> **作者:** Damian Owerko; Frederic Vatnsdal; Saurav Agarwal; Vijay Kumar; Alejandro Ribeiro
>
> **摘要:** This article presents a novel multi-agent spatial transformer (MAST) for learning communication policies in large-scale decentralized and collaborative multi-robot systems (DC-MRS). Challenges in collaboration in DC-MRS arise from: (i) partial observable states as robots make only localized perception, (ii) limited communication range with no central server, and (iii) independent execution of actions. The robots need to optimize a common task-specific objective, which, under the restricted setting, must be done using a communication policy that exhibits the desired collaborative behavior. The proposed MAST is a decentralized transformer architecture that learns communication policies to compute abstract information to be shared with other agents and processes the received information with the robot's own observations. The MAST extends the standard transformer with new positional encoding strategies and attention operations that employ windowing to limit the receptive field for MRS. These are designed for local computation, shift-equivariance, and permutation equivariance, making it a promising approach for DC-MRS. We demonstrate the efficacy of MAST on decentralized assignment and navigation (DAN) and decentralized coverage control. Efficiently trained using imitation learning in a centralized setting, the decentralized MAST policy is robust to communication delays, scales to large teams, and performs better than the baselines and other learning-based approaches.
>
---
#### [new 070] FGGS-LiDAR: Ultra-Fast, GPU-Accelerated Simulation from General 3DGS Models to LiDAR
- **分类: cs.RO; 68T40, 68U05; I.6.8**

- **简介: 该论文提出FGGS-LiDAR，旨在将3D Gaussian Splatting模型高效转换为高精度LiDAR仿真数据。任务是构建通用、无需监督的几何转换框架，解决3DGS与LiDAR模拟不兼容问题，实现超快速GPU加速仿真。**

- **链接: [http://arxiv.org/pdf/2509.17390v1](http://arxiv.org/pdf/2509.17390v1)**

> **作者:** Junzhe Wu; Yufei Jia; Yiyi Yan; Zhixing Chen; Tiao Tan; Zifan Wang; Guangyu Wang
>
> **摘要:** While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic rendering, its vast ecosystem of assets remains incompatible with high-performance LiDAR simulation, a critical tool for robotics and autonomous driving. We present \textbf{FGGS-LiDAR}, a framework that bridges this gap with a truly plug-and-play approach. Our method converts \textit{any} pretrained 3DGS model into a high-fidelity, watertight mesh without requiring LiDAR-specific supervision or architectural alterations. This conversion is achieved through a general pipeline of volumetric discretization and Truncated Signed Distance Field (TSDF) extraction. We pair this with a highly optimized, GPU-accelerated ray-casting module that simulates LiDAR returns at over 500 FPS. We validate our approach on indoor and outdoor scenes, demonstrating exceptional geometric fidelity; By enabling the direct reuse of 3DGS assets for geometrically accurate depth sensing, our framework extends their utility beyond visualization and unlocks new capabilities for scalable, multimodal simulation. Our open-source implementation is available at https://github.com/TATP-233/FGGS-LiDAR.
>
---
#### [new 071] Robust and Resilient Soft Robotic Object Insertion with Compliance-Enabled Contact Formation and Failure Recovery
- **分类: cs.RO**

- **简介: 该论文研究机械臂物体插入任务，旨在解决因位姿误差和环境变化导致的失败问题。提出使用被动柔顺腕部实现接触自适应，并结合视觉语言模型进行故障检测与恢复，提升了系统的鲁棒性与恢复能力。**

- **链接: [http://arxiv.org/pdf/2509.17666v1](http://arxiv.org/pdf/2509.17666v1)**

> **作者:** Mimo Shirasaka; Cristian C. Beltran-Hernandez; Masashi Hamaya; Yoshitaka Ushiku
>
> **摘要:** Object insertion tasks are prone to failures under pose uncertainties and environmental variations, traditionally requiring manual finetuning or controller retraining. We present a novel approach for robust and resilient object insertion using a passively compliant soft wrist that enables safe contact absorption through large deformations, without high-frequency control or force sensing. Our method structures insertion as compliance-enabled contact formations, sequential contact states that progressively constrain degrees of freedom, and integrates automated failure recovery strategies. Our key insight is that wrist compliance permits safe, repeated recovery attempts; hence, we refer to it as compliance-enabled failure recovery. We employ a pre-trained vision-language model (VLM) that assesses each skill execution from terminal poses and images, identifies failure modes, and proposes recovery actions by selecting skills and updating goals. In simulation, our method achieved an 83% success rate, recovering from failures induced by randomized conditions--including grasp misalignments up to 5 degrees, hole-pose errors up to 20mm, fivefold increases in friction, and previously unseen square/rectangular pegs--and we further validate the approach on a real robot.
>
---
#### [new 072] HuMam: Humanoid Motion Control via End-to-End Deep Reinforcement Learning with Mamba
- **分类: cs.RO; cs.AI; cs.ET; cs.SY; eess.SP; eess.SY**

- **简介: 该论文提出HuMam，一种基于端到端深度强化学习的人形机器人运动控制框架。针对训练不稳定、特征融合效率低和能耗高的问题，采用Mamba编码器进行状态融合，并设计六项奖励函数优化控制策略，提升了学习效率与稳定性，降低了能耗。**

- **链接: [http://arxiv.org/pdf/2509.18046v1](http://arxiv.org/pdf/2509.18046v1)**

> **作者:** Yinuo Wang; Yuanyang Qi; Jinzhao Zhou; Gavin Tao
>
> **备注:** 10 pages
>
> **摘要:** End-to-end reinforcement learning (RL) for humanoid locomotion is appealing for its compact perception-action mapping, yet practical policies often suffer from training instability, inefficient feature fusion, and high actuation cost. We present HuMam, a state-centric end-to-end RL framework that employs a single-layer Mamba encoder to fuse robot-centric states with oriented footstep targets and a continuous phase clock. The policy outputs joint position targets tracked by a low-level PD loop and is optimized with PPO. A concise six-term reward balances contact quality, swing smoothness, foot placement, posture, and body stability while implicitly promoting energy saving. On the JVRC-1 humanoid in mc-mujoco, HuMam consistently improves learning efficiency, training stability, and overall task performance over a strong feedforward baseline, while reducing power consumption and torque peaks. To our knowledge, this is the first end-to-end humanoid RL controller that adopts Mamba as the fusion backbone, demonstrating tangible gains in efficiency, stability, and control economy.
>
---
#### [new 073] Geometric Interpolation of Rigid Body Motions
- **分类: cs.RO; cs.NA; math.DG; math.GR; math.NA; math.OC**

- **简介: 该论文研究刚体运动插值问题，提出满足初值和边界条件的高阶轨迹插值方法，解决了从初始到终端姿态的平滑路径规划问题，并给出了数值实例验证。**

- **链接: [http://arxiv.org/pdf/2509.16966v1](http://arxiv.org/pdf/2509.16966v1)**

> **作者:** Andreas Mueller
>
> **摘要:** The problem of interpolating a rigid body motion is to find a spatial trajectory between a prescribed initial and terminal pose. Two variants of this interpolation problem are addressed. The first is to find a solution that satisfies initial conditions on the k-1 derivatives of the rigid body twist. This is called the kth-order initial value trajectory interpolation problem (k-IV-TIP). The second is to find a solution that satisfies conditions on the rigid body twist and its k-1 derivatives at the initial and terminal pose. This is called the kth-order boundary value trajectory interpolation problem (k-BV-TIP). Solutions to the k-IV-TIP for k=1,...,4, i.e. the initial twist and up to the 4th time derivative are prescribed. Further, a solution to the 1-IV-TBP is presented, i.e. the initial and terminal twist are prescribed. The latter is a novel cubic interpolation between two spatial configurations with given initial and terminal twist. This interpolation is automatically identical to the minimum acceleration curve when the twists are set to zero. The general approach to derive higher-order solutions is presented. Numerical results are shown for two examples.
>
---
#### [new 074] GeCCo - a Generalist Contact-Conditioned Policy for Loco-Manipulation Skills on Legged Robots
- **分类: cs.RO**

- **简介: 该论文提出GeCCo，一种基于深度强化学习的通用接触条件策略，用于四足机器人的运动与操作任务。旨在解决传统方法需为每个新任务重新训练控制器的问题，通过预训练低级策略并结合高级接触规划器，实现高效、模块化的多任务控制。**

- **链接: [http://arxiv.org/pdf/2509.17582v1](http://arxiv.org/pdf/2509.17582v1)**

> **作者:** Vassil Atanassov; Wanming Yu; Siddhant Gangapurwala; James Wilson; Ioannis Havoutis
>
> **备注:** You can find an associated video here: https://youtu.be/o8Dd44MkG2E
>
> **摘要:** Most modern approaches to quadruped locomotion focus on using Deep Reinforcement Learning (DRL) to learn policies from scratch, in an end-to-end manner. Such methods often fail to scale, as every new problem or application requires time-consuming and iterative reward definition and tuning. We present Generalist Contact-Conditioned Policy (GeCCo) -- a low-level policy trained with Deep Reinforcement Learning that is capable of tracking arbitrary contact points on a quadruped robot. The strength of our approach is that it provides a general and modular low-level controller that can be reused for a wider range of high-level tasks, without the need to re-train new controllers from scratch. We demonstrate the scalability and robustness of our method by evaluating on a wide range of locomotion and manipulation tasks in a common framework and under a single generalist policy. These include a variety of gaits, traversing complex terrains (eg. stairs and slopes) as well as previously unseen stepping-stones and narrow beams, and interacting with objects (eg. pushing buttons, tracking trajectories). Our framework acquires new behaviors more efficiently, simply by combining a task-specific high-level contact planner and the pre-trained generalist policy. A supplementary video can be found at https://youtu.be/o8Dd44MkG2E.
>
---
#### [new 075] VideoArtGS: Building Digital Twins of Articulated Objects from Monocular Video
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VideoArtGS，旨在从单目视频中重建可动对象的高精度数字孪生。任务包括几何重建、部件分割与运动参数估计。工作重点在于设计运动先验引导和混合中心-网格模块，以提升重建精度并减少误差。**

- **链接: [http://arxiv.org/pdf/2509.17647v1](http://arxiv.org/pdf/2509.17647v1)**

> **作者:** Yu Liu; Baoxiong Jia; Ruijie Lu; Chuyue Gan; Huayu Chen; Junfeng Ni; Song-Chun Zhu; Siyuan Huang
>
> **摘要:** Building digital twins of articulated objects from monocular video presents an essential challenge in computer vision, which requires simultaneous reconstruction of object geometry, part segmentation, and articulation parameters from limited viewpoint inputs. Monocular video offers an attractive input format due to its simplicity and scalability; however, it's challenging to disentangle the object geometry and part dynamics with visual supervision alone, as the joint movement of the camera and parts leads to ill-posed estimation. While motion priors from pre-trained tracking models can alleviate the issue, how to effectively integrate them for articulation learning remains largely unexplored. To address this problem, we introduce VideoArtGS, a novel approach that reconstructs high-fidelity digital twins of articulated objects from monocular video. We propose a motion prior guidance pipeline that analyzes 3D tracks, filters noise, and provides reliable initialization of articulation parameters. We also design a hybrid center-grid part assignment module for articulation-based deformation fields that captures accurate part motion. VideoArtGS demonstrates state-of-the-art performance in articulation and mesh reconstruction, reducing the reconstruction error by about two orders of magnitude compared to existing methods. VideoArtGS enables practical digital twin creation from monocular video, establishing a new benchmark for video-based articulated object reconstruction. Our work is made publicly available at: https://videoartgs.github.io.
>
---
#### [new 076] SLAM-Former: Putting SLAM into One Transformer
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SLAM-Former，将完整的SLAM功能集成到一个Transformer中，实现单目图像的实时建图与跟踪，并通过前后端协同优化提升几何一致性。**

- **链接: [http://arxiv.org/pdf/2509.16909v1](http://arxiv.org/pdf/2509.16909v1)**

> **作者:** Yijun Yuan; Zhuoguang Chen; Kenan Li; Weibang Wang; Hang Zhao
>
> **备注:** Project Page:https://tsinghua-mars-lab.github.io/SLAM-Former
>
> **摘要:** We present SLAM-Former, a novel neural approach that integrates full SLAM capabilities into a single transformer. Similar to traditional SLAM systems, SLAM-Former comprises both a frontend and a backend that operate in tandem. The frontend processes sequential monocular images in real-time for incremental mapping and tracking, while the backend performs global refinement to ensure a geometrically consistent result. This alternating execution allows the frontend and backend to mutually promote one another, enhancing overall system performance. Comprehensive experimental results demonstrate that SLAM-Former achieves superior or highly competitive performance compared to state-of-the-art dense SLAM methods.
>
---
#### [new 077] DINOv3-Diffusion Policy: Self-Supervised Large Visual Model for Visuomotor Diffusion Policy Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究了DINOv3自监督视觉模型在机器人操作任务中的扩散策略学习效果。通过对比自监督与监督预训练模型，验证了DINOv3在多种训练模式下的性能优势，证明其可作为高效、通用的感知前端，提升策略成功率。**

- **链接: [http://arxiv.org/pdf/2509.17684v1](http://arxiv.org/pdf/2509.17684v1)**

> **作者:** ThankGod Egbe; Peng Wang; Zhihao Guo; Zidong Chen
>
> **摘要:** This paper evaluates DINOv3, a recent large-scale self-supervised vision backbone, for visuomotor diffusion policy learning in robotic manipulation. We investigate whether a purely self-supervised encoder can match or surpass conventional supervised ImageNet-pretrained backbones (e.g., ResNet-18) under three regimes: training from scratch, frozen, and finetuned. Across four benchmark tasks (Push-T, Lift, Can, Square) using a unified FiLM-conditioned diffusion policy, we find that (i) finetuned DINOv3 matches or exceeds ResNet-18 on several tasks, (ii) frozen DINOv3 remains competitive, indicating strong transferable priors, and (iii) self-supervised features improve sample efficiency and robustness. These results support self-supervised large visual models as effective, generalizable perceptual front-ends for action diffusion policies, motivating further exploration of scalable label-free pretraining in robotic manipulation. Compared to using ResNet18 as a backbone, our approach with DINOv3 achieves up to a 10% absolute increase in test-time success rates on challenging tasks such as Can, and on-the-par performance in tasks like Lift, PushT, and Square.
>
---
#### [new 078] Delay compensation of multi-input distinct delay nonlinear systems via neural operators
- **分类: eess.SY; cs.LG; cs.RO; cs.SY; math.DS**

- **简介: 该论文研究多输入非线性系统中不同执行延迟的补偿问题，提出通过神经算子实现近似预测器的稳定性分析，证明了满足统一误差界时系统的半全局实用稳定性，并在移动机器人实验中验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2509.17131v1](http://arxiv.org/pdf/2509.17131v1)**

> **作者:** Filip Bajraktari; Luke Bhan; Miroslav Krstic; Yuanyuan Shi
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** In this work, we present the first stability results for approximate predictors in multi-input non-linear systems with distinct actuation delays. We show that if the predictor approximation satisfies a uniform (in time) error bound, semi-global practical stability is correspondingly achieved. For such approximators, the required uniform error bound depends on the desired region of attraction and the number of control inputs in the system. The result is achieved through transforming the delay into a transport PDE and conducting analysis on the coupled ODE-PDE cascade. To highlight the viability of such error bounds, we demonstrate our results on a class of approximators - neural operators - showcasing sufficiency for satisfying such a universal bound both theoretically and in simulation on a mobile robot experiment.
>
---
#### [new 079] Text-Scene: A Scene-to-Language Parsing Framework for 3D Scene Understanding
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Text-Scene框架，用于3D场景理解任务。针对3D场景语言解析数据不足和复杂性问题，设计自动将3D场景转为文本描述的方法，并构建InPlan3D基准测试3D任务规划能力。**

- **链接: [http://arxiv.org/pdf/2509.16721v1](http://arxiv.org/pdf/2509.16721v1)**

> **作者:** Haoyuan Li; Rui Liu; Hehe Fan; Yi Yang
>
> **备注:** 19 pages, 12 figures, 6 tables
>
> **摘要:** Enabling agents to understand and interact with complex 3D scenes is a fundamental challenge for embodied artificial intelligence systems. While Multimodal Large Language Models (MLLMs) have achieved significant progress in 2D image understanding, extending such capabilities to 3D scenes remains difficult: 1) 3D environment involves richer concepts such as spatial relationships, affordances, physics, layout, and so on, 2) the absence of large-scale 3D vision-language datasets has posed a significant obstacle. In this paper, we introduce Text-Scene, a framework that automatically parses 3D scenes into textual descriptions for scene understanding. Given a 3D scene, our model identifies object attributes and spatial relationships, and then generates a coherent summary of the whole scene, bridging the gap between 3D observation and language without requiring human-in-the-loop intervention. By leveraging both geometric analysis and MLLMs, Text-Scene produces descriptions that are accurate, detailed, and human-interpretable, capturing object-level details and global-level context. Experimental results on benchmarks demonstrate that our textual parses can faithfully represent 3D scenes and benefit downstream tasks. To evaluate the reasoning capability of MLLMs, we present InPlan3D, a comprehensive benchmark for 3D task planning, consisting of 3174 long-term planning tasks across 636 indoor scenes. We emphasize clarity and accessibility in our approach, aiming to make 3D scene content understandable through language. Code and datasets will be released.
>
---
#### [new 080] StereoAdapter: Adapting Stereo Depth Estimation to Underwater Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对水下立体深度估计任务，旨在解决领域适应与多模态融合问题。提出StereoAdapter框架，通过参数高效的LoRA方法和递归立体优化模块，在无大量标注数据情况下提升水下3D重建精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.16415v1](http://arxiv.org/pdf/2509.16415v1)**

> **作者:** Zhengri Wu; Yiran Wang; Yu Wen; Zeyu Zhang; Biao Wu; Hao Tang
>
> **摘要:** Underwater stereo depth estimation provides accurate 3D geometry for robotics tasks such as navigation, inspection, and mapping, offering metric depth from low-cost passive cameras while avoiding the scale ambiguity of monocular methods. However, existing approaches face two critical challenges: (i) parameter-efficiently adapting large vision foundation encoders to the underwater domain without extensive labeled data, and (ii) tightly fusing globally coherent but scale-ambiguous monocular priors with locally metric yet photometrically fragile stereo correspondences. To address these challenges, we propose StereoAdapter, a parameter-efficient self-supervised framework that integrates a LoRA-adapted monocular foundation encoder with a recurrent stereo refinement module. We further introduce dynamic LoRA adaptation for efficient rank selection and pre-training on the synthetic UW-StereoDepth-40K dataset to enhance robustness under diverse underwater conditions. Comprehensive evaluations on both simulated and real-world benchmarks show improvements of 6.11% on TartanAir and 5.12% on SQUID compared to state-of-the-art methods, while real-world deployment with the BlueROV2 robot further demonstrates the consistent robustness of our approach. Code: https://github.com/AIGeeksGroup/StereoAdapter. Website: https://aigeeksgroup.github.io/StereoAdapter.
>
---
#### [new 081] A Regularized Riccati Recursion for Interior-Point Optimal Control
- **分类: math.OC; cs.MS; cs.RO; cs.SY; eess.SY; 49M37, 90C51, 93B45; G.1.6**

- **简介: 该论文提出一种正则化Riccati递归方法，用于求解带约束的非凸离散时间最优控制问题。通过正则化内点法，确保每一步均为增广障碍-拉格朗日目标函数的下降方向，并提供C++和JAX实现。**

- **链接: [http://arxiv.org/pdf/2509.16370v1](http://arxiv.org/pdf/2509.16370v1)**

> **作者:** João Sousa-Pinto; Dominique Orban
>
> **摘要:** We derive a closed-form extension of Riccati's recursion for solving regularized LQR problems. We also show how this can be used to solve general constrained, non-convex, discrete-time optimal control problems via a regularized interior point method, while guaranteeing that each step is a descent direction of an Augmented Barrier-Lagrangian merit function. We also provide MIT-licensed implementations of our method in C++ and JAX.
>
---
#### [new 082] DepTR-MOT: Unveiling the Potential of Depth-Informed Trajectory Refinement for Multi-Object Tracking
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文针对多目标跟踪（MOT）任务，旨在解决2D方法在遮挡和密集场景下的跟踪不稳定性问题。提出DepTR-MOT，通过引入实例级深度信息，提升轨迹鲁棒性，并在机器人跟踪数据集上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.17323v1](http://arxiv.org/pdf/2509.17323v1)**

> **作者:** Buyin Deng; Lingxin Huang; Kai Luo; Fei Teng; Kailun Yang
>
> **备注:** The source code will be made publicly available at https://github.com/warriordby/DepTR-MOT
>
> **摘要:** Visual Multi-Object Tracking (MOT) is a crucial component of robotic perception, yet existing Tracking-By-Detection (TBD) methods often rely on 2D cues, such as bounding boxes and motion modeling, which struggle under occlusions and close-proximity interactions. Trackers relying on these 2D cues are particularly unreliable in robotic environments, where dense targets and frequent occlusions are common. While depth information has the potential to alleviate these issues, most existing MOT datasets lack depth annotations, leading to its underexploited role in the domain. To unveil the potential of depth-informed trajectory refinement, we introduce DepTR-MOT, a DETR-based detector enhanced with instance-level depth information. Specifically, we propose two key innovations: (i) foundation model-based instance-level soft depth label supervision, which refines depth prediction, and (ii) the distillation of dense depth maps to maintain global depth consistency. These strategies enable DepTR-MOT to output instance-level depth during inference, without requiring foundation models and without additional computational cost. By incorporating depth cues, our method enhances the robustness of the TBD paradigm, effectively resolving occlusion and close-proximity challenges. Experiments on both the QuadTrack and DanceTrack datasets demonstrate the effectiveness of our approach, achieving HOTA scores of 27.59 and 44.47, respectively. In particular, results on QuadTrack, a robotic platform MOT dataset, highlight the advantages of our method in handling occlusion and close-proximity challenges in robotic tracking. The source code will be made publicly available at https://github.com/warriordby/DepTR-MOT.
>
---
#### [new 083] CoBEVMoE: Heterogeneity-aware Feature Fusion with Dynamic Mixture-of-Experts for Collaborative Perception
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出CoBEVMoE，用于多智能体协作感知任务。针对异构特征融合问题，设计了动态混合专家架构，在BEV空间中建模特征相似性与异质性，并引入DEML损失提升表现。**

- **链接: [http://arxiv.org/pdf/2509.17107v1](http://arxiv.org/pdf/2509.17107v1)**

> **作者:** Lingzhao Kong; Jiacheng Lin; Siyu Li; Kai Luo; Zhiyong Li; Kailun Yang
>
> **备注:** The source code will be made publicly available at https://github.com/godk0509/CoBEVMoE
>
> **摘要:** Collaborative perception aims to extend sensing coverage and improve perception accuracy by sharing information among multiple agents. However, due to differences in viewpoints and spatial positions, agents often acquire heterogeneous observations. Existing intermediate fusion methods primarily focus on aligning similar features, often overlooking the perceptual diversity among agents. To address this limitation, we propose CoBEVMoE, a novel collaborative perception framework that operates in the Bird's Eye View (BEV) space and incorporates a Dynamic Mixture-of-Experts (DMoE) architecture. In DMoE, each expert is dynamically generated based on the input features of a specific agent, enabling it to extract distinctive and reliable cues while attending to shared semantics. This design allows the fusion process to explicitly model both feature similarity and heterogeneity across agents. Furthermore, we introduce a Dynamic Expert Metric Loss (DEML) to enhance inter-expert diversity and improve the discriminability of the fused representation. Extensive experiments on the OPV2V and DAIR-V2X-C datasets demonstrate that CoBEVMoE achieves state-of-the-art performance. Specifically, it improves the IoU for Camera-based BEV segmentation by +1.5% on OPV2V and the AP@50 for LiDAR-based 3D object detection by +3.0% on DAIR-V2X-C, verifying the effectiveness of expert-based heterogeneous feature modeling in multi-agent collaborative perception. The source code will be made publicly available at https://github.com/godk0509/CoBEVMoE.
>
---
#### [new 084] L2M-Reg: Building-level Uncertainty-aware Registration of Outdoor LiDAR Point Clouds and Semantic 3D City Models
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于LiDAR点云与语义3D城市模型的配准任务，旨在解决LoD2模型不确定性导致的建筑级配准难题。提出了L2M-Reg方法，通过平面对应、约束模型和自适应估计提高配准精度与效率。**

- **链接: [http://arxiv.org/pdf/2509.16832v1](http://arxiv.org/pdf/2509.16832v1)**

> **作者:** Ziyang Xu; Benedikt Schwab; Yihui Yang; Thomas H. Kolbe; Christoph Holst
>
> **备注:** submit to ISPRS Journal of Photogrammetry and Remote Sensing
>
> **摘要:** Accurate registration between LiDAR (Light Detection and Ranging) point clouds and semantic 3D city models is a fundamental topic in urban digital twinning and a prerequisite for downstream tasks, such as digital construction, change detection and model refinement. However, achieving accurate LiDAR-to-Model registration at individual building level remains challenging, particularly due to the generalization uncertainty in semantic 3D city models at the Level of Detail 2 (LoD2). This paper addresses this gap by proposing L2M-Reg, a plane-based fine registration method that explicitly accounts for model uncertainty. L2M-Reg consists of three key steps: establishing reliable plane correspondence, building a pseudo-plane-constrained Gauss-Helmert model, and adaptively estimating vertical translation. Experiments on three real-world datasets demonstrate that L2M-Reg is both more accurate and computationally efficient than existing ICP-based and plane-based methods. Overall, L2M-Reg provides a novel building-level solution regarding LiDAR-to-Model registration when model uncertainty is present.
>
---
#### [new 085] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SQS，一种用于自动驾驶的查询驱动稀疏感知模型预训练方法。通过基于查询的高斯表示和自监督投影学习，提升占用预测和3D目标检测性能，实验表明其在多个任务上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.16588v1](http://arxiv.org/pdf/2509.16588v1)**

> **作者:** Haiming Zhang; Yiyao Zhu; Wending Zhou; Xu Yan; Yingjie Cai; Bingbing Liu; Shuguang Cui; Zhen Li
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** Sparse Perception Models (SPMs) adopt a query-driven paradigm that forgoes explicit dense BEV or volumetric construction, enabling highly efficient computation and accelerated inference. In this paper, we introduce SQS, a novel query-based splatting pre-training specifically designed to advance SPMs in autonomous driving. SQS introduces a plug-in module that predicts 3D Gaussian representations from sparse queries during pre-training, leveraging self-supervised splatting to learn fine-grained contextual features through the reconstruction of multi-view images and depth maps. During fine-tuning, the pre-trained Gaussian queries are seamlessly integrated into downstream networks via query interaction mechanisms that explicitly connect pre-trained queries with task-specific queries, effectively accommodating the diverse requirements of occupancy prediction and 3D object detection. Extensive experiments on autonomous driving benchmarks demonstrate that SQS delivers considerable performance gains across multiple query-based 3D perception tasks, notably in occupancy prediction and 3D object detection, outperforming prior state-of-the-art pre-training approaches by a significant margin (i.e., +1.3 mIoU on occupancy prediction and +1.0 NDS on 3D detection).
>
---
#### [new 086] Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs
- **分类: cs.CE; cs.HC; cs.RO**

- **简介: 该论文提出了ENGDESIGN基准，用于评估大语言模型（LLMs）在九个工程领域的设计能力。不同于传统问答任务，它强调LLMs在复杂约束下综合知识、生成可行设计方案的能力，并通过仿真验证设计效果，填补了工程设计评估的空白。**

- **链接: [http://arxiv.org/pdf/2509.16204v1](http://arxiv.org/pdf/2509.16204v1)**

> **作者:** Xingang Guo; Yaxin Li; Xiangyi Kong; Yilan Jiang; Xiayu Zhao; Zhihua Gong; Yufan Zhang; Daixuan Li; Tianle Sang; Beixiao Zhu; Gregory Jun; Yingbing Huang; Yiqi Liu; Yuqi Xue; Rahul Dev Kundu; Qi Jian Lim; Yizhou Zhao; Luke Alexander Granger; Mohamed Badr Younis; Darioush Keivan; Nippun Sabharwal; Shreyanka Sinha; Prakhar Agarwal; Kojo Vandyck; Hanlin Mai; Zichen Wang; Aditya Venkatesh; Ayush Barik; Jiankun Yang; Chongying Yue; Jingjie He; Libin Wang; Licheng Xu; Hao Chen; Jinwen Wang; Liujun Xu; Rushabh Shetty; Ziheng Guo; Dahui Song; Manvi Jha; Weijie Liang; Weiman Yan; Bryan Zhang; Sahil Bhandary Karnoor; Jialiang Zhang; Rutva Pandya; Xinyi Gong; Mithesh Ballae Ganesh; Feize Shi; Ruiling Xu; Yifan Zhang; Yanfeng Ouyang; Lianhui Qin; Elyse Rosenbaum; Corey Snyder; Peter Seiler; Geir Dullerud; Xiaojia Shelly Zhang; Zuofu Cheng; Pavan Kumar Hanumolu; Jian Huang; Mayank Kulkarni; Mahdi Namazifar; Huan Zhang; Bin Hu
>
> **摘要:** Today, industry pioneers dream of developing general-purpose AI engineers capable of designing and building humanity's most ambitious projects--from starships that will carry us to distant worlds to Dyson spheres that harness stellar energy. Yet engineering design represents a fundamentally different challenge for large language models (LLMs) compared to traditional textbook-style problem solving or factual question answering. Real-world engineering design demands the synthesis of domain knowledge, navigation of complex trade-offs, and management of the tedious processes that consume much of practicing engineers' time. Despite these shared challenges across engineering disciplines, no benchmark currently captures the unique demands of engineering design work. In this work, we introduce ENGDESIGN, an Engineering Design benchmark that evaluates LLMs' abilities to perform practical design tasks across nine engineering domains: Operating System Design, Computer Architecture Design, Control System Design, Mechanical Systems, Structural Design, Digital Hardware Design, Analog Integrated Circuit Design, Robotics, and Signal Processing. Unlike existing benchmarks that focus on factual recall or question answering, ENGDESIGN uniquely emphasizes LLMs' ability to synthesize domain knowledge, reason under constraints, and generate functional, objective-oriented designs. Each task in ENGDESIGN represents a real-world engineering design problem, accompanied by a detailed task description specifying design goals, constraints, and performance requirements. We pioneer a simulation-based evaluation paradigm where LLM-generated designs undergo rigorous testing through executable, domain-specific simulations-from circuit SPICE simulations to structural finite element analysis, from control system validation to robotic motion planning.
>
---
#### [new 087] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EmbodiedSplat，旨在解决Embodied AI中Sim-to-Real导航的挑战。通过使用iPhone采集场景并结合3D Gaussian Splatting与Habitat-Sim，实现个性化策略训练，提升真实世界导航成功率。**

- **链接: [http://arxiv.org/pdf/2509.17430v1](http://arxiv.org/pdf/2509.17430v1)**

> **作者:** Gunjan Chhablani; Xiaomeng Ye; Muhammad Zubair Irshad; Zsolt Kira
>
> **备注:** 16 pages, 18 figures, paper accepted at ICCV, 2025
>
> **摘要:** The field of Embodied AI predominantly relies on simulation for training and evaluation, often using either fully synthetic environments that lack photorealism or high-fidelity real-world reconstructions captured with expensive hardware. As a result, sim-to-real transfer remains a major challenge. In this paper, we introduce EmbodiedSplat, a novel approach that personalizes policy training by efficiently capturing the deployment environment and fine-tuning policies within the reconstructed scenes. Our method leverages 3D Gaussian Splatting (GS) and the Habitat-Sim simulator to bridge the gap between realistic scene capture and effective training environments. Using iPhone-captured deployment scenes, we reconstruct meshes via GS, enabling training in settings that closely approximate real-world conditions. We conduct a comprehensive analysis of training strategies, pre-training datasets, and mesh reconstruction techniques, evaluating their impact on sim-to-real predictivity in real-world scenarios. Experimental results demonstrate that agents fine-tuned with EmbodiedSplat outperform both zero-shot baselines pre-trained on large-scale real-world datasets (HM3D) and synthetically generated datasets (HSSD), achieving absolute success rate improvements of 20\% and 40\% on real-world Image Navigation task. Moreover, our approach yields a high sim-vs-real correlation (0.87--0.97) for the reconstructed meshes, underscoring its effectiveness in adapting policies to diverse environments with minimal effort. Project page: https://gchhablani.github.io/embodied-splat
>
---
#### [new 088] Trajectory Encryption Cooperative Salvo Guidance
- **分类: eess.SY; cs.MA; cs.RO; cs.SY; math.OC**

- **简介: 该论文研究协同同时拦截任务，解决多无人系统在异构条件下轨迹可预测性与鲁棒性问题。通过引入轨迹加密思想，利用异构制导策略生成多样轨迹，提升抗干扰能力并隐藏群体意图，实现灵活且隐蔽的同时目标拦截。**

- **链接: [http://arxiv.org/pdf/2509.17341v1](http://arxiv.org/pdf/2509.17341v1)**

> **作者:** Lohitvel Gopikannan; Shashi Ranjan Kumar; Abhinav Sinha
>
> **摘要:** This paper introduces the concept of trajectory encryption in cooperative simultaneous target interception, wherein heterogeneity in guidance principles across a team of unmanned autonomous systems is leveraged as a strategic design feature. By employing a mix of heterogeneous time-to-go formulations leading to a cooperative guidance strategy, the swarm of vehicles is able to generate diverse trajectory families. This diversity expands the feasible solution space for simultaneous target interception, enhances robustness under disturbances, and enables flexible time-to-go adjustments without predictable detouring. From an adversarial perspective, heterogeneity obscures the collective interception intent by preventing straightforward prediction of swarm dynamics, effectively acting as an encryption layer in the trajectory domain. Simulations demonstrate that the swarm of heterogeneous vehicles is able to intercept a moving target simultaneously from a diverse set of initial engagement configurations.
>
---
#### [new 089] Safe Guaranteed Dynamics Exploration with Probabilistic Models
- **分类: eess.SY; cs.LG; cs.RO; cs.SY; math.DS; math.OC**

- **简介: 该论文研究安全强化学习任务，旨在解决未知动力学系统中同时保证最优性和安全性的挑战。提出一种悲观安全框架，在线学习动力学模型并乐观探索，确保高概率安全操作，应用于自动驾驶和无人机导航等关键场景。**

- **链接: [http://arxiv.org/pdf/2509.16650v1](http://arxiv.org/pdf/2509.16650v1)**

> **作者:** Manish Prajapat; Johannes Köhler; Melanie N. Zeilinger; Andreas Krause
>
> **摘要:** Ensuring both optimality and safety is critical for the real-world deployment of agents, but becomes particularly challenging when the system dynamics are unknown. To address this problem, we introduce a notion of maximum safe dynamics learning via sufficient exploration in the space of safe policies. We propose a $\textit{pessimistically}$ safe framework that $\textit{optimistically}$ explores informative states and, despite not reaching them due to model uncertainty, ensures continuous online learning of dynamics. The framework achieves first-of-its-kind results: learning the dynamics model sufficiently $-$ up to an arbitrary small tolerance (subject to noise) $-$ in a finite time, while ensuring provably safe operation throughout with high probability and without requiring resets. Building on this, we propose an algorithm to maximize rewards while learning the dynamics $\textit{only to the extent needed}$ to achieve close-to-optimal performance. Unlike typical reinforcement learning (RL) methods, our approach operates online in a non-episodic setting and ensures safety throughout the learning process. We demonstrate the effectiveness of our approach in challenging domains such as autonomous car racing and drone navigation under aerodynamic effects $-$ scenarios where safety is critical and accurate modeling is difficult.
>
---
#### [new 090] Underground Multi-robot Systems at Work: a revolution in mining
- **分类: eess.SY; cs.RO; cs.SY; eess.SY (Primary), cs.RO (Secondary)**

- **简介: 该论文研究地下采矿中的多机器人系统任务，旨在解决传统设备难以在狭小、危险环境中作业的问题。提出基于HFSM的模块化多机器人协作架构，实现自主矿物提取与系统协调操作。**

- **链接: [http://arxiv.org/pdf/2509.16267v1](http://arxiv.org/pdf/2509.16267v1)**

> **作者:** Victor V. Puche; Kashish Verma; Matteo Fumagalli
>
> **备注:** 6 pages, 6 figures, submitted to IEEE SII 2026
>
> **摘要:** The growing global demand for critical raw materials (CRMs) has highlighted the need to access difficult and hazardous environments such as abandoned underground mines. These sites pose significant challenges for conventional machinery and human operators due to confined spaces, structural instability, and lack of infrastructure. To address this, we propose a modular multi-robot system designed for autonomous operation in such environments, enabling sequential mineral extraction tasks. Unlike existing work that focuses primarily on mapping and inspection through global behavior or central control, our approach incorporates physical interaction capabilities using specialized robots coordinated through local high-level behavior control. Our proposed system utilizes Hierarchical Finite State Machine (HFSM) behaviors to structure complex task execution across heterogeneous robotic platforms. Each robot has its own HFSM behavior to perform sequential autonomy while maintaining overall system coordination, achieved by triggering behavior execution through inter-robot communication. This architecture effectively integrates software and hardware components to support collaborative, task-driven multi-robot operation in confined underground environments.
>
---
#### [new 091] ST-GS: Vision-Based 3D Semantic Occupancy Prediction with Spatial-Temporal Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ST-GS框架，用于基于视觉的3D语义占用预测任务。针对现有方法在多视角空间交互和时序一致性上的不足，设计了空间聚合策略与几何感知时序融合方案，提升了建模效果和时序稳定性。**

- **链接: [http://arxiv.org/pdf/2509.16552v1](http://arxiv.org/pdf/2509.16552v1)**

> **作者:** Xiaoyang Yan; Muleilan Pei; Shaojie Shen
>
> **摘要:** 3D occupancy prediction is critical for comprehensive scene understanding in vision-centric autonomous driving. Recent advances have explored utilizing 3D semantic Gaussians to model occupancy while reducing computational overhead, but they remain constrained by insufficient multi-view spatial interaction and limited multi-frame temporal consistency. To overcome these issues, in this paper, we propose a novel Spatial-Temporal Gaussian Splatting (ST-GS) framework to enhance both spatial and temporal modeling in existing Gaussian-based pipelines. Specifically, we develop a guidance-informed spatial aggregation strategy within a dual-mode attention mechanism to strengthen spatial interaction in Gaussian representations. Furthermore, we introduce a geometry-aware temporal fusion scheme that effectively leverages historical context to improve temporal continuity in scene completion. Extensive experiments on the large-scale nuScenes occupancy prediction benchmark showcase that our proposed approach not only achieves state-of-the-art performance but also delivers markedly better temporal consistency compared to existing Gaussian-based methods.
>
---
#### [new 092] Servos for Local Map Exploration Onboard Nonholonomic Vehicles for Extremum Seeking
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文研究非完整车辆在极值搜索任务中的局部地图探索问题。针对传统扰动方法的局限，提出使用有界周期或拟周期信号估计多变量映射的任意阶导数，并设计伺服驱动传感器的源搜索控制器，实验验证了收敛速度的提升。**

- **链接: [http://arxiv.org/pdf/2509.16365v1](http://arxiv.org/pdf/2509.16365v1)**

> **作者:** Dylan James-Kavanaugh; Patrick McNamee; Qixu Wang; Zahra Nili Ahmadabadi
>
> **备注:** 12 pages, 8 figures, IEEE Transactions on Control Systems Technology Submission
>
> **摘要:** Extremum seeking control (ESC) often employs perturbation-based estimates of derivatives for some sensor field or cost function. These estimates are generally obtained by simply multiplying the output of a single-unit sensor by some time-varying function. Previous work has focused on sinusoidal perturbations to generate derivative estimates with results for arbitrary order derivatives of scalar maps or higher up to third-order derivatives of multivariable maps. This work extends the perturbations from sinusoidal to bounded periodic or almost periodic functions and considers multivariable maps. A necessary and sufficient condition is given for determining if time-varying functions exist for estimating arbitrary order derivatives of multivariable maps for any given bounded periodic or almost periodic dither signal. These results are then used in a source seeking controller for a nonholonomic vehicle with a sensor actuated by servo. The conducted simulation and real-world experiments demonstrate that by distributing the local map exploration to a servo, the nonholonomic vehicle was able to achieve a faster convergence to the source.
>
---
#### [new 093] Segment-to-Act: Label-Noise-Robust Action-Prompted Video Segmentation Towards Embodied Intelligence
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文研究动作提示视频分割任务，旨在解决标签噪声问题。工作包括引入两类噪声、构建首个带噪声基准ActiSeg-NL、适配六种噪声学习策略，并提出PMHM机制提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.16677v1](http://arxiv.org/pdf/2509.16677v1)**

> **作者:** Wenxin Li; Kunyu Peng; Di Wen; Ruiping Liu; Mengfei Duan; Kai Luo; Kailun Yang
>
> **备注:** The established benchmark and source code will be made publicly available at https://github.com/mylwx/ActiSeg-NL
>
> **摘要:** Embodied intelligence relies on accurately segmenting objects actively involved in interactions. Action-based video object segmentation addresses this by linking segmentation with action semantics, but it depends on large-scale annotations and prompts that are costly, inconsistent, and prone to multimodal noise such as imprecise masks and referential ambiguity. To date, this challenge remains unexplored. In this work, we take the first step by studying action-based video object segmentation under label noise, focusing on two sources: textual prompt noise (category flips and within-category noun substitutions) and mask annotation noise (perturbed object boundaries to mimic imprecise supervision). Our contributions are threefold. First, we introduce two types of label noises for the action-based video object segmentation task. Second, we build up the first action-based video object segmentation under a label noise benchmark ActiSeg-NL and adapt six label-noise learning strategies to this setting, and establish protocols for evaluating them under textual, boundary, and mixed noise. Third, we provide a comprehensive analysis linking noise types to failure modes and robustness gains, and we introduce a Parallel Mask Head Mechanism (PMHM) to address mask annotation noise. Qualitative evaluations further reveal characteristic failure modes, including boundary leakage and mislocalization under boundary perturbations, as well as occasional identity substitutions under textual flips. Our comparative analysis reveals that different learning strategies exhibit distinct robustness profiles, governed by a foreground-background trade-off where some achieve balanced performance while others prioritize foreground accuracy at the cost of background precision. The established benchmark and source code will be made publicly available at https://github.com/mylwx/ActiSeg-NL.
>
---
## 更新

#### [replaced 001] How Good are Foundation Models in Step-by-Step Embodied Reasoning?
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.15293v2](http://arxiv.org/pdf/2509.15293v2)**

> **作者:** Dinura Dissanayake; Ahmed Heakl; Omkar Thawakar; Noor Ahsan; Ritesh Thawkar; Ketan More; Jean Lahoud; Rao Anwer; Hisham Cholakkal; Ivan Laptev; Fahad Shahbaz Khan; Salman Khan
>
> **备注:** Project page: https://mbzuai-oryx.github.io/FoMER-Bench/
>
> **摘要:** Embodied agents operating in the physical world must make decisions that are not only effective but also safe, spatially coherent, and grounded in context. While recent advances in large multimodal models (LMMs) have shown promising capabilities in visual understanding and language generation, their ability to perform structured reasoning for real-world embodied tasks remains underexplored. In this work, we aim to understand how well foundation models can perform step-by-step reasoning in embodied environments. To this end, we propose the Foundation Model Embodied Reasoning (FoMER) benchmark, designed to evaluate the reasoning capabilities of LMMs in complex embodied decision-making scenarios. Our benchmark spans a diverse set of tasks that require agents to interpret multimodal observations, reason about physical constraints and safety, and generate valid next actions in natural language. We present (i) a large-scale, curated suite of embodied reasoning tasks, (ii) a novel evaluation framework that disentangles perceptual grounding from action reasoning, and (iii) empirical analysis of several leading LMMs under this setting. Our benchmark includes over 1.1k samples with detailed step-by-step reasoning across 10 tasks and 8 embodiments, covering three different robot types. Our results highlight both the potential and current limitations of LMMs in embodied reasoning, pointing towards key challenges and opportunities for future research in robot intelligence. Our data and code will be made publicly available.
>
---
#### [replaced 002] VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.09372v2](http://arxiv.org/pdf/2509.09372v2)**

> **作者:** Yihao Wang; Pengxiang Ding; Lingxiao Li; Can Cui; Zirui Ge; Xinyang Tong; Wenxuan Song; Han Zhao; Wei Zhao; Pengxu Hou; Siteng Huang; Yifan Tang; Wenhui Wang; Ru Zhang; Jianyi Liu; Donglin Wang
>
> **备注:** 28 pages; Project page: https://vla-adapter.github.io/; Github: https://github.com/OpenHelix-Team/VLA-Adapter; HuggingFace: https://huggingface.co/VLA-Adapter
>
> **摘要:** Vision-Language-Action (VLA) models typically bridge the gap between perceptual and action spaces by pre-training a large-scale Vision-Language Model (VLM) on robotic data. While this approach greatly enhances performance, it also incurs significant training costs. In this paper, we investigate how to effectively bridge vision-language (VL) representations to action (A). We introduce VLA-Adapter, a novel paradigm designed to reduce the reliance of VLA models on large-scale VLMs and extensive pre-training. To this end, we first systematically analyze the effectiveness of various VL conditions and present key findings on which conditions are essential for bridging perception and action spaces. Based on these insights, we propose a lightweight Policy module with Bridge Attention, which autonomously injects the optimal condition into the action space. In this way, our method achieves high performance using only a 0.5B-parameter backbone, without any robotic data pre-training. Extensive experiments on both simulated and real-world robotic benchmarks demonstrate that VLA-Adapter not only achieves state-of-the-art level performance, but also offers the fast inference speed reported to date. Furthermore, thanks to the proposed advanced bridging paradigm, VLA-Adapter enables the training of a powerful VLA model in just 8 hours on a single consumer-grade GPU, greatly lowering the barrier to deploying the VLA model. Project page: https://vla-adapter.github.io/.
>
---
#### [replaced 003] RGBSQGrasp: Inferring Local Superquadric Primitives from Single RGB Image for Graspability-Aware Bin Picking
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.02387v2](http://arxiv.org/pdf/2503.02387v2)**

> **作者:** Yifeng Xu; Fan Zhu; Ye Li; Sebastian Ren; Xiaonan Huang; Yuhao Chen
>
> **备注:** 2 pages, 2 figures, IROS2025 RGMCW Best Extended Abstract
>
> **摘要:** Bin picking is a challenging robotic task due to occlusions and physical constraints that limit visual information for object recognition and grasping. Existing approaches often rely on known CAD models or prior object geometries, restricting generalization to novel or unknown objects. Other methods directly regress grasp poses from RGB-D data without object priors, but the inherent noise in depth sensing and the lack of object understanding make grasp synthesis and evaluation more difficult. Superquadrics (SQ) offer a compact, interpretable shape representation that captures the physical and graspability understanding of objects. However, recovering them from limited viewpoints is challenging, as existing methods rely on multiple perspectives for near-complete point cloud reconstruction, limiting their effectiveness in bin-picking. To address these challenges, we propose \textbf{RGBSQGrasp}, a grasping framework that leverages superquadric shape primitives and foundation metric depth estimation models to infer grasp poses from a monocular RGB camera -- eliminating the need for depth sensors. Our framework integrates a universal, cross-platform dataset generation pipeline, a foundation model-based object point cloud estimation module, a global-local superquadric fitting network, and an SQ-guided grasp pose sampling module. By integrating these components, RGBSQGrasp reliably infers grasp poses through geometric reasoning, enhancing grasp stability and adaptability to unseen objects. Real-world robotic experiments demonstrate a 92\% grasp success rate, highlighting the effectiveness of RGBSQGrasp in packed bin-picking environments.
>
---
#### [replaced 004] Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.07639v2](http://arxiv.org/pdf/2506.07639v2)**

> **作者:** Zhekai Duan; Yuan Zhang; Shikai Geng; Gaowen Liu; Joschka Boedecker; Chris Xiaoxuan Lu
>
> **摘要:** Embodied Chain-of-Thought (ECoT) reasoning enhances vision-language-action (VLA) models by improving performance and interpretability through intermediate reasoning steps. However, its sequential autoregressive token generation introduces significant inference latency, limiting real-time deployment. We propose Fast ECoT, an inference-time acceleration method that exploits the structured and repetitive nature of ECoT to (1) cache and reuse high-level reasoning across timesteps and (2) parallelise the generation of modular reasoning steps. Additionally, we introduce an asynchronous scheduler that decouples reasoning from action decoding, further boosting responsiveness. Fast ECoT requires no model changes or additional training and integrates easily into existing VLA pipelines. Experiments in both simulation (LIBERO) and real-world robot tasks show up to a 7.5% reduction in latency with comparable or improved task success rate and reasoning faithfulness, bringing ECoT policies closer to practical real-time deployment.
>
---
#### [replaced 005] Look, Focus, Act: Efficient and Robust Robot Learning via Human Gaze and Foveated Vision Transformers
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15833v2](http://arxiv.org/pdf/2507.15833v2)**

> **作者:** Ian Chuang; Jinyu Zou; Andrew Lee; Dechen Gao; Iman Soltani
>
> **备注:** Project page: https://ian-chuang.github.io/gaze-av-aloha/
>
> **摘要:** Human vision is a highly active process driven by gaze, which directs attention to task-relevant regions through foveation, dramatically reducing visual processing. In contrast, robot learning systems typically rely on passive, uniform processing of raw camera images. In this work, we explore how incorporating human-like active gaze into robotic policies can enhance efficiency and robustness. We develop GIAVA (Gaze Integrated Active-Vision ALOHA), a robot vision system that emulates human head and neck movement, and gaze adjustment for foveated processing. Extending the AV-ALOHA robot platform, we introduce a framework for simultaneously collecting eye-tracking, perspective control, and robot manipulation demonstration data from a human operator. We also open-source a simulation benchmark and dataset for training robot policies that incorporate human gaze. Inspired by recent work in foveated image segmentation and given the widespread use of Vision Transformers (ViTs) in robot learning, we integrate gaze information into ViTs using a foveated patch tokenization scheme. Compared to uniform patch tokenization, this significantly reduces the number of tokens, and thus computation. Our results show that our method for foveated robot vision drastically reduces computational overhead, and enhances robustness to background distractors. Notably, on certain high-precision tasks, foveated vision also improves performance, as reflected in higher success rates. Together, these findings suggest that human-inspired foveated visual processing offers untapped potential and should be further considered as a useful inductive bias in robotic vision systems. https://ian-chuang.github.io/gaze-av-aloha/
>
---
#### [replaced 006] Evo-0: Vision-Language-Action Model with Implicit Spatial Understanding
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.00416v2](http://arxiv.org/pdf/2507.00416v2)**

> **作者:** Tao Lin; Gen Li; Yilei Zhong; Yanwen Zou; Yuxin Du; Jiting Liu; Encheng Gu; Bo Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising framework for enabling generalist robots capable of perceiving, reasoning, and acting in the real world. These models usually build upon pretrained Vision-Language Models (VLMs), which excel at semantic understanding due to large-scale image and text pretraining. However, existing VLMs typically lack precise spatial understanding capabilities, as they are primarily tuned on 2D image-text pairs without 3D supervision. To address this limitation, recent approaches have incorporated explicit 3D inputs such as point clouds or depth maps, but this necessitates additional depth sensors or pre-trained depth estimation models, which may yield defective results. In contrast, our work introduces a plug-and-play module that implicitly incorporates 3D geometry features into VLA models by leveraging an off-the-shelf visual geometry foundation model. This integration provides the model with depth-aware visual representations, improving its ability to understand the geometric structure of the scene and the spatial relationships among objects from RGB images alone. We evaluate our method on a set of spatially challenging tasks in both simulation and the real world. Extensive evaluations show that our method significantly improves the performance of state-of-the-art VLA models across diverse scenarios.
>
---
#### [replaced 007] Do Visual-Language Grid Maps Capture Latent Semantics?
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.10117v3](http://arxiv.org/pdf/2403.10117v3)**

> **作者:** Matti Pekkanen; Tsvetomila Mihaylova; Francesco Verdoja; Ville Kyrki
>
> **备注:** IROS 2025
>
> **摘要:** Visual-language models (VLMs) have recently been introduced in robotic mapping using the latent representations, i.e., embeddings, of the VLMs to represent semantics in the map. They allow moving from a limited set of human-created labels toward open-vocabulary scene understanding, which is very useful for robots when operating in complex real-world environments and interacting with humans. While there is anecdotal evidence that maps built this way support downstream tasks, such as navigation, rigorous analysis of the quality of the maps using these embeddings is missing. In this paper, we propose a way to analyze the quality of maps created using VLMs. We investigate two critical properties of map quality: queryability and distinctness. The evaluation of queryability addresses the ability to retrieve information from the embeddings. We investigate intra-map distinctness to study the ability of the embeddings to represent abstract semantic classes and inter-map distinctness to evaluate the generalization properties of the representation. We propose metrics to evaluate these properties and evaluate two state-of-the-art mapping methods, VLMaps and OpenScene, using two encoders, LSeg and OpenSeg, using real-world data from the Matterport3D data set. Our findings show that while 3D features improve queryability, they are not scale invariant, whereas image-based embeddings generalize to multiple map resolutions. This allows the image-based methods to maintain smaller map sizes, which can be crucial for using these methods in real-world deployments. Furthermore, we show that the choice of the encoder has an effect on the results. The results imply that properly thresholding open-vocabulary queries is an open problem.
>
---
#### [replaced 008] Diffusion-Based Approximate MPC: Fast and Consistent Imitation of Multi-Modal Action Distributions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.04603v3](http://arxiv.org/pdf/2504.04603v3)**

> **作者:** Pau Marquez Julbe; Julian Nubert; Henrik Hose; Sebastian Trimpe; Katherine J. Kuchenbecker
>
> **摘要:** Approximating model predictive control (MPC) using imitation learning (IL) allows for fast control without solving expensive optimization problems online. However, methods that use neural networks in a simple L2-regression setup fail to approximate multi-modal (set-valued) solution distributions caused by local optima found by the numerical solver or non-convex constraints, such as obstacles, significantly limiting the applicability of approximate MPC in practice. We solve this issue by using diffusion models to accurately represent the complete solution distribution (i.e., all modes) up to kilohertz sampling rates. This work shows that diffusion-based AMPC significantly outperforms L2-regression-based approximate MPC for multi-modal action distributions. In contrast to most earlier work on IL, we also focus on running the diffusion-based controller at a higher rate and in joint space instead of end-effector space. Additionally, we propose the use of gradient guidance during the denoising process to consistently pick the same mode in closed loop to prevent switching between solutions. We propose using the cost and constraint satisfaction of the original MPC problem during parallel sampling of solutions from the diffusion model to pick a better mode online. We evaluate our method on the fast and accurate control of a 7-DoF robot manipulator both in simulation and on hardware deployed at 250 Hz, achieving a speedup of more than 70 times compared to solving the MPC problem online and also outperforming the numerical optimization (used for training) in success ratio.
>
---
#### [replaced 009] ReWiND: Language-Guided Rewards Teach Robot Policies without New Demonstrations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.10911v2](http://arxiv.org/pdf/2505.10911v2)**

> **作者:** Jiahui Zhang; Yusen Luo; Abrar Anwar; Sumedh Anand Sontakke; Joseph J Lim; Jesse Thomason; Erdem Biyik; Jesse Zhang
>
> **备注:** CoRL 2025 Oral
>
> **摘要:** We introduce ReWiND, a framework for learning robot manipulation tasks solely from language instructions without per-task demonstrations. Standard reinforcement learning (RL) and imitation learning methods require expert supervision through human-designed reward functions or demonstrations for every new task. In contrast, ReWiND starts from a small demonstration dataset to learn: (1) a data-efficient, language-conditioned reward function that labels the dataset with rewards, and (2) a language-conditioned policy pre-trained with offline RL using these rewards. Given an unseen task variation, ReWiND fine-tunes the pre-trained policy using the learned reward function, requiring minimal online interaction. We show that ReWiND's reward model generalizes effectively to unseen tasks, outperforming baselines by up to 2.4x in reward generalization and policy alignment metrics. Finally, we demonstrate that ReWiND enables sample-efficient adaptation to new tasks, beating baselines by 2x in simulation and improving real-world pretrained bimanual policies by 5x, taking a step towards scalable, real-world robot learning. See website at https://rewind-reward.github.io/.
>
---
#### [replaced 010] Dexterous Grasping with Real-World Robotic Reinforcement Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.04014v2](http://arxiv.org/pdf/2503.04014v2)**

> **作者:** Dongchi Huang; Tianle Zhang; Yihang Li; Ling Zhao; Jiayi Li; Zhirui Fang; Chunhe Xia; Xiaodong He
>
> **摘要:** Dexterous grasping in the real world presents a fundamental and significant challenge for robot learning. The ability to employ affordance-aware poses to grasp objects with diverse geometries and properties in arbitrary scenarios is essential for general-purpose robots. However, existing research predominantly addresses dexterous grasping problems within simulators, which encounter difficulties when applied in real-world environments due to the domain gap between reality and simulation. This limitation hinders their generalizability and practicality in real-world applications. In this paper, we present DexGraspRL, a reinforcement learning (RL) framework that directly trains robots in real-world environments to acquire dexterous grasping skills. Specifically, DexGraspRL consists of two stages: (i) a pretraining stage that pretrains the policy using imitation learning (IL) with a limited set of expert demonstrations; (ii) a fine-tuning stage that refines the policy through direct RL in real-world scenarios. To mitigate the catastrophic forgetting phenomenon arising from the distribution shift between demonstrations and real-world environments, we design a regularization term that balances the exploitation of RL with the preservation of the pretrained policy. Our experiments with real-world tasks demonstrate that DexGraspRL successfully accomplishes diverse dexterous grasping tasks, achieving an average success rate of nearly 92%. Furthermore, by fine-tuning with RL, our method uncovers novel policies, surpassing the IL policy with a 23% reduction in average cycle time.
>
---
#### [replaced 011] FLAME: A Federated Learning Benchmark for Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01729v2](http://arxiv.org/pdf/2503.01729v2)**

> **作者:** Santiago Bou Betran; Alberta Longhini; Miguel Vasco; Yuchong Zhang; Danica Kragic
>
> **备注:** Under Review
>
> **摘要:** Recent progress in robotic manipulation has been fueled by large-scale datasets collected across diverse environments. Training robotic manipulation policies on these datasets is traditionally performed in a centralized manner, raising concerns regarding scalability, adaptability, and data privacy. While federated learning enables decentralized, privacy-preserving training, its application to robotic manipulation remains largely unexplored. We introduce FLAME (Federated Learning Across Manipulation Environments), the first benchmark designed for federated learning in robotic manipulation. FLAME consists of: (i) a set of large-scale datasets of over 160,000 expert demonstrations of multiple manipulation tasks, collected across a wide range of simulated environments; (ii) a training and evaluation framework for robotic policy learning in a federated setting. We evaluate standard federated learning algorithms in FLAME, showing their potential for distributed policy learning and highlighting key challenges. Our benchmark establishes a foundation for scalable, adaptive, and privacy-aware robotic learning.
>
---
#### [replaced 012] VF-Plan: Bridging the Art Gallery Problem and Static LiDAR Scanning with Visibility Field Optimization
- **分类: cs.RO; cs.CG**

- **链接: [http://arxiv.org/pdf/2503.01562v2](http://arxiv.org/pdf/2503.01562v2)**

> **作者:** Biao Xiong; Longjun Zhang; Ruiqi Huang; Junwei Zhou; S. R. U. N. Jafri; Bojian Wu; Fashuai Li
>
> **摘要:** Viewpoint planning is critical for efficient 3D data acquisition in applications such as 3D reconstruction, building life-cycle management, navigation, and interior decoration. However, existing methods often neglect key optimization objectives specific to static LiDAR systems, resulting in redundant or disconnected viewpoint networks. The viewpoint planning problem (VPP) extends the classical Art Gallery Problem (AGP) by requiring full coverage, strong registrability, and coherent network connectivity under constrained sensor capabilities. To address these challenges, we introduce a novel Visibility Field (VF) that accurately captures the directional and range-dependent visibility properties of static LiDAR scanners. We further observe that visibility information naturally converges onto a 1D skeleton embedded in the 2D space, enabling significant searching space reduction. Leveraging these insights, we develop a greedy optimization algorithm tailored to the VPP, which constructs a minimal yet fully connected Viewpoint Network (VPN) with low redundancy. Experimental evaluations across diverse indoor and outdoor scenarios confirm the scalability and robustness of our method. Compared to expert-designed VPNs and existing state-of-the-art approaches, our algorithm achieves comparable or fewer viewpoints while significantly enhancing connectivity. In particular, it reduces the weighted average path length by approximately 95%, demonstrating substantial improvements in compactness and structural efficiency. Code is available at https://github.com/xiongbiaostar/VFPlan.
>
---
#### [replaced 013] Improving Drone Racing Performance Through Iterative Learning MPC
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2508.01103v3](http://arxiv.org/pdf/2508.01103v3)**

> **作者:** Haocheng Zhao; Niklas Schlüter; Lukas Brunke; Angela P. Schoellig
>
> **备注:** Accepted for oral presentation at IROS 2025
>
> **摘要:** Autonomous drone racing presents a challenging control problem, requiring real-time decision-making and robust handling of nonlinear system dynamics. While iterative learning model predictive control (LMPC) offers a promising framework for iterative performance improvement, its direct application to drone racing faces challenges like real-time compatibility or the trade-off between time-optimal and safe traversal. In this paper, we enhance LMPC with three key innovations: (1) an adaptive cost function that dynamically weights time-optimal tracking against centerline adherence, (2) a shifted local safe set to prevent excessive shortcutting and enable more robust iterative updates, and (3) a Cartesian-based formulation that accommodates safety constraints without the singularities or integration errors associated with Frenet-frame transformations. Results from extensive simulation and real-world experiments demonstrate that our improved algorithm can optimize initial trajectories generated by a wide range of controllers with varying levels of tuning for a maximum improvement in lap time by 60.85%. Even applied to the most aggressively tuned state-of-the-art model-based controller, MPCC++, on a real drone, a 6.05% improvement is still achieved. Overall, the proposed method pushes the drone toward faster traversal and avoids collisions in simulation and real-world experiments, making it a practical solution to improve the peak performance of drone racing.
>
---
#### [replaced 014] DETACH: Cross-domain Learning for Long-Horizon Tasks via Mixture of Disentangled Experts
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.07842v2](http://arxiv.org/pdf/2508.07842v2)**

> **作者:** Yutong Shen; Hangxu Liu; Lei Zhang; Penghui Liu; Ruizhe Xia; Tianyi Yao; Tongtong Feng
>
> **备注:** 14 pages,8 figures. Submitted to ICRA'26
>
> **摘要:** Long-Horizon (LH) tasks in Human-Scene Interaction (HSI) are complex multi-step tasks that require continuous planning, sequential decision-making, and extended execution across domains to achieve the final goal. However, existing methods heavily rely on skill chaining by concatenating pre-trained subtasks, with environment observations and self-state tightly coupled, lacking the ability to generalize to new combinations of environments and skills, failing to complete various LH tasks across domains. To solve this problem, this paper presents DETACH, a cross-domain learning framework for LH tasks via biologically inspired dual-stream disentanglement. Inspired by the brain's "where-what" dual pathway mechanism, DETACH comprises two core modules: i) an environment learning module for spatial understanding, which captures object functions, spatial relationships, and scene semantics, achieving cross-domain transfer through complete environment-self disentanglement; ii) a skill learning module for task execution, which processes self-state information including joint degrees of freedom and motor patterns, enabling cross-skill transfer through independent motor pattern encoding. We conducted extensive experiments on various LH tasks in HSI scenes. Compared with existing methods, DETACH can achieve an average subtasks success rate improvement of 23% and average execution efficiency improvement of 29%.
>
---
#### [replaced 015] Robotic Trail Maker Platform for Rehabilitation in Neurological Conditions: Clinical Use Cases
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.19230v3](http://arxiv.org/pdf/2504.19230v3)**

> **作者:** Srikar Annamraju; Harris Nisar; Dayu Xia; Shankar A. Deka; Anne Horowitz; Nadica Miljković; Dušan M. Stipanović
>
> **备注:** The first three authors are co-first authors. This manuscript is under review with the IEEE Transactions on Neural Systems and Rehabilitation Engineering
>
> **摘要:** Patients with neurological conditions require rehabilitation to restore their motor, visual, and cognitive abilities. To meet the shortage of therapists and reduce their workload, a robotic rehabilitation platform involving the clinical trail making test is proposed. Therapists can create custom trails for each patient and the patient can trace the trails using a robotic device. The platform can track the performance of the patient and use these data to provide dynamic assistance through the robot to the patient interface. Therefore, the proposed platform not only functions as an evaluation platform, but also trains the patient in recovery. The developed platform has been validated at a rehabilitation center, with therapists and patients operating the device. It was found that patients performed poorly while using the platform compared to healthy subjects and that the assistance provided also improved performance amongst patients. Statistical analysis demonstrated that the speed of the patients was significantly enhanced with the robotic assistance. Further, neural networks are trained to classify between patients and healthy subjects and to forecast their movements using the data collected.
>
---
#### [replaced 016] ExT: Towards Scalable Autonomous Excavation via Large-Scale Multi-Task Pretraining and Fine-Tuning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.14992v2](http://arxiv.org/pdf/2509.14992v2)**

> **作者:** Yifan Zhai; Lorenzo Terenzi; Patrick Frey; Diego Garcia Soto; Pascal Egli; Marco Hutter
>
> **摘要:** Scaling up the deployment of autonomous excavators is of great economic and societal importance. Yet it remains a challenging problem, as effective systems must robustly handle unseen worksite conditions and new hardware configurations. Current state-of-the-art approaches rely on highly engineered, task-specific controllers, which require extensive manual tuning for each new scenario. In contrast, recent advances in large-scale pretrained models have shown remarkable adaptability across tasks and embodiments in domains such as manipulation and navigation, but their applicability to heavy construction machinery remains largely unexplored. In this work, we introduce ExT, a unified open-source framework for large-scale demonstration collection, pretraining, and fine-tuning of multitask excavation policies. ExT policies are first trained on large-scale demonstrations collected from a mix of experts, then fine-tuned either with supervised fine-tuning (SFT) or reinforcement learning fine-tuning (RLFT) to specialize to new tasks or operating conditions. Through both simulation and real-world experiments, we show that pretrained ExT policies can execute complete excavation cycles with centimeter-level accuracy, successfully transferring from simulation to real machine with performance comparable to specialized single-task controllers. Furthermore, in simulation, we demonstrate that ExT's fine-tuning pipelines allow rapid adaptation to new tasks, out-of-distribution conditions, and machine configurations, while maintaining strong performance on previously learned tasks. These results highlight the potential of ExT to serve as a foundation for scalable and generalizable autonomous excavation.
>
---
#### [replaced 017] Toward an Interaction-Centered Approach to Robot Trustworthiness
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.13976v2](http://arxiv.org/pdf/2508.13976v2)**

> **作者:** Carlo Mazzola; Hassan Ali; Kristína Malinovská; Igor Farkaš
>
> **备注:** In proceedings of TRUST 2025 (https://arxiv.org/abs/2509.11402), a workshop at IEEE RO-MAN 2025: https://ro-man2025.org/
>
> **摘要:** As robots get more integrated into human environments, fostering trustworthiness in embodied robotic agents becomes paramount for an effective and safe human-robot interaction (HRI). To achieve that, HRI applications must promote human trust that aligns with robot skills and avoid misplaced trust or overtrust, which can pose safety risks and ethical concerns. To achieve that, HRI applications must promote human trust that aligns with robot skills and avoid misplaced trust or overtrust, which can pose safety risks and ethical concerns. In this position paper, we outline an interaction-based framework for building trust through mutual understanding between humans and robots. We emphasize two main pillars: human awareness and transparency, referring to the robot ability to interpret human actions accurately and to clearly communicate its intentions and goals, respectively. By integrating these two pillars, robots can behave in a manner that aligns with human expectations and needs while providing their human partners with both comprehension and control over their actions. We also introduce four components that we think are important for bridging the gap between a human-perceived sense of trust and a robot true capabilities.
>
---
#### [replaced 018] Diffusion Graph Neural Networks and Dataset for Robust Olfactory Navigation in Hazard Robotics
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00455v4](http://arxiv.org/pdf/2506.00455v4)**

> **作者:** Kordel K. France; Ovidiu Daescu
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Navigation by scent is a capability in robotic systems that is rising in demand. However, current methods often suffer from ambiguities, particularly when robots misattribute odours to incorrect objects due to limitations in olfactory datasets and sensor resolutions. To address challenges in olfactory navigation, we introduce a multimodal olfaction dataset along with a novel machine learning method using diffusion-based molecular generation that can be used by itself or with automated olfactory dataset construction pipelines. This generative process of our diffusion model expands the chemical space beyond the limitations of both current olfactory datasets and training methods, enabling the identification of potential odourant molecules not previously documented. The generated molecules can then be more accurately validated using advanced olfactory sensors, enabling them to detect more compounds and inform better hardware design. By integrating visual analysis, language processing, and molecular generation, our framework enhances the ability of olfaction-vision models on robots to accurately associate odours with their correct sources, thereby improving navigation and decision-making through better sensor selection for a target compound in critical applications such as explosives detection, narcotics screening, and search and rescue. Our methodology represents a foundational advancement in the field of artificial olfaction, offering a scalable solution to challenges posed by limited olfactory data and sensor ambiguities. Code, models, and data are made available to the community at: https://huggingface.co/datasets/kordelfrance/olfaction-vision-language-dataset.
>
---
#### [replaced 019] The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.12594v2](http://arxiv.org/pdf/2509.12594v2)**

> **作者:** Titong Jiang; Xuefeng Jiang; Yuan Ma; Xin Wen; Bailin Li; Kun Zhan; Peng Jia; Yahui Liu; Sheng Sun; Xianpeng Lang
>
> **备注:** Under review. Project site: https://liauto-research.github.io/LightVLA
>
> **摘要:** We present LightVLA, a simple yet effective differentiable token pruning framework for vision-language-action (VLA) models. While VLA models have shown impressive capability in executing real-world robotic tasks, their deployment on resource-constrained platforms is often bottlenecked by the heavy attention-based computation over large sets of visual tokens. LightVLA addresses this challenge through adaptive, performance-driven pruning of visual tokens: It generates dynamic queries to evaluate visual token importance, and adopts Gumbel softmax to enable differentiable token selection. Through fine-tuning, LightVLA learns to preserve the most informative visual tokens while pruning tokens which do not contribute to task execution, thereby improving efficiency and performance simultaneously. Notably, LightVLA requires no heuristic magic numbers and introduces no additional trainable parameters, making it compatible with modern inference frameworks. Experimental results demonstrate that LightVLA outperforms different VLA models and existing token pruning methods across diverse tasks on the LIBERO benchmark, achieving higher success rates with substantially reduced computational overhead. Specifically, LightVLA reduces FLOPs and latency by 59.1% and 38.2% respectively, with a 2.6% improvement in task success rate. Meanwhile, we also investigate the learnable query-based token pruning method LightVLA* with additional trainable parameters, which also achieves satisfactory performance. Our work reveals that as VLA pursues optimal performance, LightVLA spontaneously learns to prune tokens from a performance-driven perspective. To the best of our knowledge, LightVLA is the first work to apply adaptive visual token pruning to VLA tasks with the collateral goals of efficiency and performance, marking a significant step toward more efficient, powerful and practical real-time robotic systems.
>
---
#### [replaced 020] Learning Robotic Policy with Imagined Transition: Mitigating the Trade-off between Robustness and Optimality
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10484v2](http://arxiv.org/pdf/2503.10484v2)**

> **作者:** Wei Xiao; Shangke Lyu; Zhefei Gong; Renjie Wang; Donglin Wang
>
> **摘要:** Existing quadrupedal locomotion learning paradigms usually rely on extensive domain randomization to alleviate the sim2real gap and enhance robustness. It trains policies with a wide range of environment parameters and sensor noises to perform reliably under uncertainty. However, since optimal performance under ideal conditions often conflicts with the need to handle worst-case scenarios, there is a trade-off between optimality and robustness. This trade-off forces the learned policy to prioritize stability in diverse and challenging conditions over efficiency and accuracy in ideal ones, leading to overly conservative behaviors that sacrifice peak performance. In this paper, we propose a two-stage framework that mitigates this trade-off by integrating policy learning with imagined transitions. This framework enhances the conventional reinforcement learning (RL) approach by incorporating imagined transitions as demonstrative inputs. These imagined transitions are derived from an optimal policy and a dynamics model operating within an idealized setting. Our findings indicate that this approach significantly mitigates the domain randomization-induced negative impact of existing RL algorithms. It leads to accelerated training, reduced tracking errors within the distribution, and enhanced robustness outside the distribution.
>
---
#### [replaced 021] Online Slip Detection and Friction Coefficient Estimation for Autonomous Racing
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.15423v2](http://arxiv.org/pdf/2509.15423v2)**

> **作者:** Christopher Oeltjen; Carson Sobolewski; Saleh Faghfoorian; Lorant Domokos; Giancarlo Vidal; Ivan Ruchkin
>
> **备注:** Equal contribution by the first three authors
>
> **摘要:** Accurate knowledge of the tire-road friction coefficient (TRFC) is essential for vehicle safety, stability, and performance, especially in autonomous racing, where vehicles often operate at the friction limit. However, TRFC cannot be directly measured with standard sensors, and existing estimation methods either depend on vehicle or tire models with uncertain parameters or require large training datasets. In this paper, we present a lightweight approach for online slip detection and TRFC estimation. Our approach relies solely on IMU and LiDAR measurements and the control actions, without special dynamical or tire models, parameter identification, or training data. Slip events are detected in real time by comparing commanded and measured motions, and the TRFC is then estimated directly from observed accelerations under no-slip conditions. Experiments with a 1:10-scale autonomous racing car across different friction levels demonstrate that the proposed approach achieves accurate and consistent slip detections and friction coefficients, with results closely matching ground-truth measurements. These findings highlight the potential of our simple, deployable, and computationally efficient approach for real-time slip monitoring and friction coefficient estimation in autonomous driving.
>
---
#### [replaced 022] How Much Do Large Language Models Know about Human Motion? A Case Study in 3D Avatar Control
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.21531v2](http://arxiv.org/pdf/2505.21531v2)**

> **作者:** Kunhang Li; Jason Naradowsky; Yansong Feng; Yusuke Miyao
>
> **摘要:** We explore the human motion knowledge of Large Language Models (LLMs) through 3D avatar control. Given a motion instruction, we prompt LLMs to first generate a high-level movement plan with consecutive steps (High-level Planning), then specify body part positions in each step (Low-level Planning), which we linearly interpolate into avatar animations. Using 20 representative motion instructions that cover fundamental movements and balance body part usage, we conduct comprehensive evaluations, including human and automatic scoring of both high-level movement plans and generated animations, as well as automatic comparison with oracle positions in low-level planning. Our findings show that LLMs are strong at interpreting high-level body movements but struggle with precise body part positioning. While decomposing motion queries into atomic components improves planning, LLMs face challenges in multi-step movements involving high-degree-of-freedom body parts. Furthermore, LLMs provide reasonable approximations for general spatial descriptions, but fall short in handling precise spatial specifications. Notably, LLMs demonstrate promise in conceptualizing creative motions and distinguishing culturally specific motion patterns.
>
---
#### [replaced 023] AEGIS: Automated Error Generation and Identification for Multi-Agent Systems
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.14295v2](http://arxiv.org/pdf/2509.14295v2)**

> **作者:** Fanqi Kong; Ruijie Zhang; Huaxiao Yin; Guibin Zhang; Xiaofei Zhang; Ziang Chen; Zhaowei Zhang; Xiaoyuan Zhang; Song-Chun Zhu; Xue Feng
>
> **摘要:** As Multi-Agent Systems (MAS) become increasingly autonomous and complex, understanding their error modes is critical for ensuring their reliability and safety. However, research in this area has been severely hampered by the lack of large-scale, diverse datasets with precise, ground-truth error labels. To address this bottleneck, we introduce \textbf{AEGIS}, a novel framework for \textbf{A}utomated \textbf{E}rror \textbf{G}eneration and \textbf{I}dentification for Multi-Agent \textbf{S}ystems. By systematically injecting controllable and traceable errors into initially successful trajectories, we create a rich dataset of realistic failures. This is achieved using a context-aware, LLM-based adaptive manipulator that performs sophisticated attacks like prompt injection and response corruption to induce specific, predefined error modes. We demonstrate the value of our dataset by exploring three distinct learning paradigms for the error identification task: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. Our comprehensive experiments show that models trained on AEGIS data achieve substantial improvements across all three learning paradigms. Notably, several of our fine-tuned models demonstrate performance competitive with or superior to proprietary systems an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at https://kfq20.github.io/AEGIS-Website.
>
---
#### [replaced 024] Latent Policy Steering with Embodiment-Agnostic Pretrained World Models
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.13340v2](http://arxiv.org/pdf/2507.13340v2)**

> **作者:** Yiqi Wang; Mrinal Verghese; Jeff Schneider
>
> **摘要:** Learning visuomotor policies via imitation has proven effective across a wide range of robotic domains. However, the performance of these policies is heavily dependent on the number of training demonstrations, which requires expensive data collection in the real world. In this work, we aim to reduce data collection efforts when learning visuomotor robot policies by leveraging existing or cost-effective data from a wide range of embodiments, such as public robot datasets and the datasets of humans playing with objects (human data from play). Our approach leverages two key insights. First, we use optic flow as an embodiment-agnostic action representation to train a World Model (WM) across multi-embodiment datasets, and finetune it on a small amount of robot data from the target embodiment. Second, we develop a method, Latent Policy Steering (LPS), to improve the output of a behavior-cloned policy by searching in the latent space of the WM for better action sequences. In real world experiments, we observe significant improvements in the performance of policies trained with a small amount of data (over 50% relative improvement with 30 demonstrations and over 20% relative improvement with 50 demonstrations) by combining the policy with a WM pretrained on two thousand episodes sampled from the existing Open X-embodiment dataset across different robots or a cost-effective human dataset from play.
>
---
#### [replaced 025] Shape-induced obstacle attraction and repulsion during dynamic locomotion
- **分类: physics.bio-ph; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2103.08176v2](http://arxiv.org/pdf/2103.08176v2)**

> **作者:** Yuanfeng Han; Ratan Othayoth; Yulong Wang; Chun-Cheng Hsu; Rafael de la Tijera Obert; Evains Francois; Chen Li
>
> **摘要:** Robots still struggle to dynamically traverse complex 3-D terrain with many large obstacles, an ability required for many critical applications. Body-obstacle interaction is often inevitable and induces perturbation and uncertainty in motion that challenges closed-form dynamic modeling. Here, inspired by recent discovery of a terradynamic streamlined shape, we studied how two body shapes interacting with obstacles affect turning and pitching motions of an open-loop multi-legged robot and cockroaches during dynamic locomotion. With a common cuboidal body, the robot was attracted towards obstacles, resulting in pitching up and flipping-over. By contrast, with an elliptical body, the robot was repelled by obstacles and readily traversed. The animal displayed qualitatively similar turning and pitching motions induced by these two body shapes. However, unlike the cuboidal robot, the cuboidal animal was capable of escaping obstacle attraction and subsequent high pitching and flipping over, which inspired us to develop an empirical pitch-and-turn strategy for cuboidal robots. Considering the similarity of our self-propelled body-obstacle interaction with part-feeder interaction in robotic part manipulation, we developed a quasi-static potential energy landscape model to explain the dependence of dynamic locomotion on body shape. Our experimental and modeling results also demonstrated that obstacle attraction or repulsion is an inherent property of locomotor body shape and insensitive to obstacle geometry and size. Our study expanded the concept and usefulness of terradynamic shapes for passive control of robot locomotion to traverse large obstacles using physical interaction. Our study is also a step in establishing an energy landscape approach to locomotor transitions.
>
---
#### [replaced 026] Trajectory Prediction for Autonomous Driving: Progress, Limitations, and Future Directions
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03262v3](http://arxiv.org/pdf/2503.03262v3)**

> **作者:** Nadya Abdel Madjid; Abdulrahman Ahmad; Murad Mebrahtu; Yousef Babaa; Abdelmoamen Nasser; Sumbal Malik; Bilal Hassan; Naoufel Werghi; Jorge Dias; Majid Khonji
>
> **摘要:** As the potential for autonomous vehicles to be integrated on a large scale into modern traffic systems continues to grow, ensuring safe navigation in dynamic environments is crucial for smooth integration. To guarantee safety and prevent collisions, autonomous vehicles must be capable of accurately predicting the trajectories of surrounding traffic agents. Over the past decade, significant efforts from both academia and industry have been dedicated to designing solutions for precise trajectory forecasting. These efforts have produced a diverse range of approaches, raising questions about the differences between these methods and whether trajectory prediction challenges have been fully addressed. This paper reviews a substantial portion of recent trajectory prediction methods proposing a taxonomy to classify existing solutions. A general overview of the prediction pipeline is also provided, covering input and output modalities, modeling features, and prediction paradigms existing in the literature. In addition, the paper discusses active research areas within trajectory prediction, addresses the posed research questions, and highlights the remaining research gaps and challenges.
>
---
#### [replaced 027] Equality Constrained Diffusion for Direct Trajectory Optimization
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2410.01939v2](http://arxiv.org/pdf/2410.01939v2)**

> **作者:** Vince Kurtz; Joel W. Burdick
>
> **备注:** ACC 2025, fixed typo in equations (11)-(12)
>
> **摘要:** The recent success of diffusion-based generative models in image and natural language processing has ignited interest in diffusion-based trajectory optimization for nonlinear control systems. Existing methods cannot, however, handle the nonlinear equality constraints necessary for direct trajectory optimization. As a result, diffusion-based trajectory optimizers are currently limited to shooting methods, where the nonlinear dynamics are enforced by forward rollouts. This precludes many of the benefits enjoyed by direct methods, including flexible state constraints, reduced numerical sensitivity, and easy initial guess specification. In this paper, we present a method for diffusion-based optimization with equality constraints. This allows us to perform direct trajectory optimization, enforcing dynamic feasibility with constraints rather than rollouts. To the best of our knowledge, this is the first diffusion-based optimization algorithm that supports the general nonlinear equality constraints required for direct trajectory optimization.
>
---
#### [replaced 028] I-FailSense: Towards General Robotic Failure Detection with Vision-Language Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.16072v2](http://arxiv.org/pdf/2509.16072v2)**

> **作者:** Clemence Grislain; Hamed Rahimi; Olivier Sigaud; Mohamed Chetouani
>
> **摘要:** Language-conditioned robotic manipulation in open-world settings requires not only accurate task execution but also the ability to detect failures for robust deployment in real-world environments. Although recent advances in vision-language models (VLMs) have significantly improved the spatial reasoning and task-planning capabilities of robots, they remain limited in their ability to recognize their own failures. In particular, a critical yet underexplored challenge lies in detecting semantic misalignment errors, where the robot executes a task that is semantically meaningful but inconsistent with the given instruction. To address this, we propose a method for building datasets targeting Semantic Misalignment Failures detection, from existing language-conditioned manipulation datasets. We also present I-FailSense, an open-source VLM framework with grounded arbitration designed specifically for failure detection. Our approach relies on post-training a base VLM, followed by training lightweight classification heads, called FS blocks, attached to different internal layers of the VLM and whose predictions are aggregated using an ensembling mechanism. Experiments show that I-FailSense outperforms state-of-the-art VLMs, both comparable in size and larger, in detecting semantic misalignment errors. Notably, despite being trained only on semantic misalignment detection, I-FailSense generalizes to broader robotic failure categories and effectively transfers to other simulation environments and real-world with zero-shot or minimal post-training. The datasets and models are publicly released on HuggingFace (Webpage: https://clemgris.github.io/I-FailSense/).
>
---
#### [replaced 029] Vector Field-Guided Learning Predictive Control for Motion Planning of Mobile Robots with Uncertain Dynamics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2405.08283v4](http://arxiv.org/pdf/2405.08283v4)**

> **作者:** Yang Lu; Weijia Yao; Yongqian Xiao; Xinglong Zhang; Xin Xu; Yaonan Wang; Dingbang Xiao
>
> **摘要:** In obstacle-dense scenarios, providing safe guidance for mobile robots is critical to improve the safe maneuvering capability. However, the guidance provided by standard guiding vector fields (GVFs) may limit the motion capability due to the improper curvature of the integral curve when traversing obstacles. On the other hand, robotic system dynamics are often time-varying, uncertain, and even unknown during the motion planning process. Therefore, many existing kinodynamic motion planning methods could not achieve satisfactory reliability in guaranteeing safety. To address these challenges, we propose a two-level Vector Field-guided Learning Predictive Control (VF-LPC) approach that improves safe maneuverability. The first level, the guiding level, generates safe desired trajectories using the designed kinodynamic GVF, enabling safe motion in obstacle-dense environments. The second level, the Integrated Motion Planning and Control (IMPC) level, first uses a deep Koopman operator to learn a nominal dynamics model offline and then updates the model uncertainties online using sparse Gaussian processes (GPs). The learned dynamics and a game-based safe barrier function are then incorporated into the LPC framework to generate near-optimal planning solutions. Extensive simulations and real-world experiments were conducted on quadrotor unmanned aerial vehicles and unmanned ground vehicles, demonstrating that VF-LPC enables robots to maneuver safely.
>
---
#### [replaced 030] MaskedManipulator: Versatile Whole-Body Manipulation
- **分类: cs.RO; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2505.19086v2](http://arxiv.org/pdf/2505.19086v2)**

> **作者:** Chen Tessler; Yifeng Jiang; Erwin Coumans; Zhengyi Luo; Gal Chechik; Xue Bin Peng
>
> **备注:** SIGGRAPH Asia 2025
>
> **摘要:** We tackle the challenges of synthesizing versatile, physically simulated human motions for full-body object manipulation. Unlike prior methods that are focused on detailed motion tracking, trajectory following, or teleoperation, our framework enables users to specify versatile high-level objectives such as target object poses or body poses. To achieve this, we introduce MaskedManipulator, a generative control policy distilled from a tracking controller trained on large-scale human motion capture data. This two-stage learning process allows the system to perform complex interaction behaviors, while providing intuitive user control over both character and object motions. MaskedManipulator produces goal-directed manipulation behaviors that expand the scope of interactive animation systems beyond task-specific solutions.
>
---
#### [replaced 031] RIFT: Group-Relative RL Fine-Tuning for Realistic and Controllable Traffic Simulation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.03344v3](http://arxiv.org/pdf/2505.03344v3)**

> **作者:** Keyu Chen; Wenchao Sun; Hao Cheng; Sifa Zheng
>
> **摘要:** Achieving both realism and controllability in closed-loop traffic simulation remains a key challenge in autonomous driving. Dataset-based methods reproduce realistic trajectories but suffer from covariate shift in closed-loop deployment, compounded by simplified dynamics models that further reduce reliability. Conversely, physics-based simulation methods enhance reliable and controllable closed-loop interactions but often lack expert demonstrations, compromising realism. To address these challenges, we introduce a dual-stage AV-centric simulation framework that conducts imitation learning pre-training in a data-driven simulator to capture trajectory-level realism and route-level controllability, followed by reinforcement learning fine-tuning in a physics-based simulator to enhance style-level controllability and mitigate covariate shift. In the fine-tuning stage, we propose RIFT, a novel group-relative RL fine-tuning strategy that evaluates all candidate modalities through group-relative formulation and employs a surrogate objective for stable optimization, enhancing style-level controllability and mitigating covariate shift while preserving the trajectory-level realism and route-level controllability inherited from IL pre-training. Extensive experiments demonstrate that RIFT improves realism and controllability in traffic simulation while simultaneously exposing the limitations of modern AV systems in closed-loop evaluation. Project Page: https://currychen77.github.io/RIFT/
>
---
#### [replaced 032] HyperTASR: Hypernetwork-Driven Task-Aware Scene Representations for Robust Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.18802v3](http://arxiv.org/pdf/2508.18802v3)**

> **作者:** Li Sun; Jiefeng Wu; Feng Chen; Ruizhe Liu; Yanchao Yang
>
> **摘要:** Effective policy learning for robotic manipulation requires scene representations that selectively capture task-relevant environmental features. Current approaches typically employ task-agnostic representation extraction, failing to emulate the dynamic perceptual adaptation observed in human cognition. We present HyperTASR, a hypernetwork-driven framework that modulates scene representations based on both task objectives and the execution phase. Our architecture dynamically generates representation transformation parameters conditioned on task specifications and progression state, enabling representations to evolve contextually throughout task execution. This approach maintains architectural compatibility with existing policy learning frameworks while fundamentally reconfiguring how visual features are processed. Unlike methods that simply concatenate or fuse task embeddings with task-agnostic representations, HyperTASR establishes computational separation between task-contextual and state-dependent processing paths, enhancing learning efficiency and representational quality. Comprehensive evaluations in both simulation and real-world environments demonstrate substantial performance improvements across different representation paradigms. Through ablation studies and attention visualization, we confirm that our approach selectively prioritizes task-relevant scene information, closely mirroring human adaptive perception during manipulation tasks. The project website is at https://lisunphil.github.io/HyperTASR_projectpage/.
>
---
#### [replaced 033] OTAS: Open-vocabulary Token Alignment for Outdoor Segmentation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.08851v2](http://arxiv.org/pdf/2507.08851v2)**

> **作者:** Simon Schwaiger; Stefan Thalhammer; Wilfried Wöber; Gerald Steinbauer-Wagner
>
> **摘要:** Understanding open-world semantics is critical for robotic planning and control, particularly in unstructured outdoor environments. Existing vision-language mapping approaches typically rely on object-centric segmentation priors, which often fail outdoors due to semantic ambiguities and indistinct class boundaries. We propose OTAS - an Open-vocabulary Token Alignment method for outdoor Segmentation. OTAS addresses the limitations of open-vocabulary segmentation models by extracting semantic structure directly from the output tokens of pre-trained vision models. By clustering semantically similar structures across single and multiple views and grounding them in language, OTAS reconstructs a geometrically consistent feature field that supports open-vocabulary segmentation queries. Our method operates in a zero-shot manner, without scene-specific fine-tuning, and achieves real-time performance of up to ~17 fps. On the Off-Road Freespace Detection dataset, OTAS yields a modest IoU improvement over fine-tuned and open-vocabulary 2D segmentation baselines. In 3D segmentation on TartanAir, it achieves up to a 151% relative IoU improvement compared to existing open-vocabulary mapping methods. Real-world reconstructions further demonstrate OTAS' applicability to robotic deployment. Code and a ROS 2 node are available at https://otas-segmentation.github.io/.
>
---
#### [replaced 034] X2C: A Dataset Featuring Nuanced Facial Expressions for Realistic Humanoid Imitation
- **分类: cs.RO; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.11146v2](http://arxiv.org/pdf/2505.11146v2)**

> **作者:** Peizhen Li; Longbing Cao; Xiao-Ming Wu; Runze Yang; Xiaohan Yu
>
> **摘要:** The ability to imitate realistic facial expressions is essential for humanoid robots engaged in affective human-robot communication. However, the lack of datasets containing diverse humanoid facial expressions with proper annotations hinders progress in realistic humanoid facial expression imitation. To address these challenges, we introduce X2C (Anything to Control), a dataset featuring nuanced facial expressions for realistic humanoid imitation. With X2C, we contribute: 1) a high-quality, high-diversity, large-scale dataset comprising 100,000 (image, control value) pairs. Each image depicts a humanoid robot displaying a diverse range of facial expressions, annotated with 30 control values representing the ground-truth expression configuration; 2) X2CNet, a novel human-to-humanoid facial expression imitation framework that learns the correspondence between nuanced humanoid expressions and their underlying control values from X2C. It enables facial expression imitation in the wild for different human performers, providing a baseline for the imitation task, showcasing the potential value of our dataset; 3) real-world demonstrations on a physical humanoid robot, highlighting its capability to advance realistic humanoid facial expression imitation. Code and Data: https://lipzh5.github.io/X2CNet/
>
---
#### [replaced 035] A Large Language Model-based multi-agent manufacturing system for intelligent shopfloor
- **分类: cs.AI; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.16887v2](http://arxiv.org/pdf/2405.16887v2)**

> **作者:** Zhen Zhao; Dunbing Tang; Changchun Liu; Liping Wang; Zequn Zhang; Haihua Zhu; Kai Chen; Qingwei Nie; Yuchen Ji
>
> **摘要:** As customer demand for multi-variety and small-batch production increases, dynamic disturbances place greater demands on manufacturing systems. To address such challenges, researchers proposed the multi-agent manufacturing system. However, conventional agent negotiation typically relies on pre-defined and fixed heuristic rules, which are ill-suited to managing complex and fluctuating disturbances. In current implementations, mainstream approaches based on reinforcement learning require the development of simulators and training models specific to a given shopfloor, necessitating substantial computational resources and lacking scalability. To overcome this limitation, the present study proposes a Large Language Model-based (LLM-based) multi-agent manufacturing system for intelligent shopfloor management. By defining the diverse modules of agents and their collaborative methods, this system facilitates the processing of all workpieces with minimal human intervention. The agents in this system consist of the Machine Server Module (MSM), Bid Inviter Module (BIM), Bidder Module (BM), Thinking Module (TM), and Decision Module (DM). By harnessing the reasoning capabilities of LLMs, these modules enable agents to dynamically analyze shopfloor information and select appropriate processing machines. The LLM-based modules, predefined by system prompts, provide dynamic functionality for the system without the need for pre-training. Extensive experiments were conducted in physical shopfloor settings. The results demonstrate that the proposed system exhibits strong adaptability, and achieves superior performance (makespan) and stability (as measured by sample standard deviation) compared to other approaches without requiring pre-training.
>
---
#### [replaced 036] UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.08787v4](http://arxiv.org/pdf/2505.08787v4)**

> **作者:** Hanjung Kim; Jaehyun Kang; Hyolim Kang; Meedeum Cho; Seon Joo Kim; Youngwoon Lee
>
> **备注:** CoRL 2025. Project Page: https://kimhanjung.github.io/UniSkill/
>
> **摘要:** Mimicry is a fundamental learning mechanism in humans, enabling individuals to learn new tasks by observing and imitating experts. However, applying this ability to robots presents significant challenges due to the inherent differences between human and robot embodiments in both their visual appearance and physical capabilities. While previous methods bridge this gap using cross-embodiment datasets with shared scenes and tasks, collecting such aligned data between humans and robots at scale is not trivial. In this paper, we propose UniSkill, a novel framework that learns embodiment-agnostic skill representations from large-scale cross-embodiment video data without any labels, enabling skills extracted from human video prompts to effectively transfer to robot policies trained only on robot data. Our experiments in both simulation and real-world environments show that our cross-embodiment skills successfully guide robots in selecting appropriate actions, even with unseen video prompts. The project website can be found at: https://kimhanjung.github.io/UniSkill.
>
---
#### [replaced 037] Traversing Narrow Paths: A Two-Stage Reinforcement Learning Framework for Robust and Safe Humanoid Walking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.20661v4](http://arxiv.org/pdf/2508.20661v4)**

> **作者:** TianChen Huang; Runchen Xu; Yu Wang; Wei Gao; Shiwu Zhang
>
> **备注:** Project website: https://huangtc233.github.io/Traversing-the-Narrow-Path/
>
> **摘要:** Traversing narrow paths is challenging for humanoid robots due to the sparse and safety-critical footholds required. Purely template-based or end-to-end reinforcement learning-based methods suffer from such harsh terrains. This paper proposes a two stage training framework for such narrow path traversing tasks, coupling a template-based foothold planner with a low-level foothold tracker from Stage-I training and a lightweight perception aided foothold modifier from Stage-II training. With the curriculum setup from flat ground to narrow paths across stages, the resulted controller in turn learns to robustly track and safely modify foothold targets to ensure precise foot placement over narrow paths. This framework preserves the interpretability from the physics-based template and takes advantage of the generalization capability from reinforcement learning, resulting in easy sim-to-real transfer. The learned policies outperform purely template-based or reinforcement learning-based baselines in terms of success rate, centerline adherence and safety margins. Validation on a Unitree G1 humanoid robot yields successful traversal of a 0.2m wide and 3m long beam for 20 trials without any failure.
>
---
#### [replaced 038] Sensorless Remote Center of Motion Misalignment Estimation
- **分类: cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.13011v2](http://arxiv.org/pdf/2503.13011v2)**

> **作者:** Hao Yang; Lidia Al-Zogbi; Ahmet Yildiz; Nabil Simaan; Jie Ying Wu
>
> **摘要:** Laparoscopic surgery constrains instrument motion around a fixed pivot point at the incision into a patient to minimize tissue trauma. Surgical robots achieve this through either hardware to software-based remote center of motion (RCM) constraints. However, accurate RCM alignment is difficult due to manual trocar placement, patient motion, and tissue deformation. Misalignment between the robot's RCM point and the patient incision site can cause unsafe forces at the incision site. This paper presents a sensorless force estimation-based framework for dynamically assessing and optimizing RCM misalignment in robotic surgery. Our experiments demonstrate that misalignment exceeding 20 mm can generate large enough forces to potentially damage tissue, emphasizing the need for precise RCM positioning. For misalignment $D\geq $ 20 mm, our optimization algorithm estimates the RCM offset with an absolute error within 5 mm. Accurate RCM misalignment estimation is a step toward automated RCM misalignment compensation, enhancing safety and reducing tissue damage in robotic-assisted laparoscopic surgery.
>
---
#### [replaced 039] DreamControl: Human-Inspired Whole-Body Humanoid Control for Scene Interaction via Guided Diffusion
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.14353v2](http://arxiv.org/pdf/2509.14353v2)**

> **作者:** Dvij Kalaria; Sudarshan S Harithas; Pushkal Katara; Sangkyung Kwak; Sarthak Bhagat; Shankar Sastry; Srinath Sridhar; Sai Vemprala; Ashish Kapoor; Jonathan Chung-Kuan Huang
>
> **备注:** https://genrobo.github.io/DreamControl/ (under submission)
>
> **摘要:** We introduce DreamControl, a novel methodology for learning autonomous whole-body humanoid skills. DreamControl leverages the strengths of diffusion models and Reinforcement Learning (RL): our core innovation is the use of a diffusion prior trained on human motion data, which subsequently guides an RL policy in simulation to complete specific tasks of interest (e.g., opening a drawer or picking up an object). We demonstrate that this human motion-informed prior allows RL to discover solutions unattainable by direct RL, and that diffusion models inherently promote natural looking motions, aiding in sim-to-real transfer. We validate DreamControl's effectiveness on a Unitree G1 robot across a diverse set of challenging tasks involving simultaneous lower and upper body control and object interaction.
>
---
#### [replaced 040] Sample-Efficient Reinforcement Learning with Symmetry-Guided Demonstrations for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2304.06055v2](http://arxiv.org/pdf/2304.06055v2)**

> **作者:** Amir M. Soufi Enayati; Zengjie Zhang; Kashish Gupta; Homayoun Najjaran
>
> **摘要:** Reinforcement learning (RL) suffers from low sample efficiency, particularly in high-dimensional continuous state-action spaces of complex robotic manipulation tasks. RL performance can improve by leveraging prior knowledge, even when demonstrations are limited and collected from simplified environments. To address this, we define General Abstract Symmetry (GAS) for aggregating demonstrations from symmetrical abstract partitions of the robot environment. We introduce Demo-EASE, a novel training framework using a dual-buffer architecture that stores both demonstrations and RL-generated experiences. Demo-EASE improves sample efficiency through symmetry-guided demonstrations and behavior cloning, enabling strong initialization and balanced exploration-exploitation. Demo-EASE is compatible with both on-policy and off-policy RL algorithms, supporting various training regimes. We evaluate our framework in three simulation experiments using a Kinova Gen3 robot with joint-space control within PyBullet. Our results show that Demo-EASE significantly accelerates convergence and improves final performance compared to standard RL baselines, demonstrating its potential for efficient real-world robotic manipulation learning.
>
---
#### [replaced 041] A Modular Robotic System for Autonomous Exploration and Semantic Updating in Large-Scale Indoor Environments
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.15493v3](http://arxiv.org/pdf/2409.15493v3)**

> **作者:** Sai Haneesh Allu; Itay Kadosh; Tyler Summers; Yu Xiang
>
> **备注:** 10 pages, 9 figures, 5 tables. Project page is available at https://irvlutd.github.io/SemanticMapping/
>
> **摘要:** We present a modular robotic system for autonomous exploration and semantic updating of large-scale unknown environments. Our approach enables a mobile robot to build, revisit, and update a hybrid semantic map that integrates a 2D occupancy grid for geometry with a topological graph for object semantics. Unlike prior methods that rely on manual teleoperation or precollected datasets, our two-phase approach achieves end-to-end autonomy: first, a modified frontier-based exploration algorithm with dynamic search windows constructs a geometric map; second, using a greedy trajectory planner, environments are revisited, and object semantics are updated using open-vocabulary object detection and segmentation. This modular system, compatible with any metric SLAM framework, supports continuous operation by efficiently updating the semantic graph to reflect short-term and long-term changes such as object relocation, removal, or addition. We validate the approach on a Fetch robot in real-world indoor environments of approximately $8,500$m$^2$ and $117$m$^2$, demonstrating robust and scalable semantic mapping and continuous adaptation, marking a fully autonomous integration of exploration, mapping, and semantic updating on a physical robot.
>
---
#### [replaced 042] ReasonPlan: Unified Scene Prediction and Decision Reasoning for Closed-loop Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO; 68T40(Primary), 68T45, 68T50(Secondary); I.2.9; I.2.10; I.5.1**

- **链接: [http://arxiv.org/pdf/2505.20024v2](http://arxiv.org/pdf/2505.20024v2)**

> **作者:** Xueyi Liu; Zuodong Zhong; Yuxin Guo; Yun-Fu Liu; Zhiguo Su; Qichao Zhang; Junli Wang; Yinfeng Gao; Yupeng Zheng; Qiao Lin; Huiyong Chen; Dongbin Zhao
>
> **备注:** 18 pages; 9 figures; https://github.com/Liuxueyi/ReasonPlan
>
> **摘要:** Due to the powerful vision-language reasoning and generalization abilities, multimodal large language models (MLLMs) have garnered significant attention in the field of end-to-end (E2E) autonomous driving. However, their application to closed-loop systems remains underexplored, and current MLLM-based methods have not shown clear superiority to mainstream E2E imitation learning approaches. In this work, we propose ReasonPlan, a novel MLLM fine-tuning framework designed for closed-loop driving through holistic reasoning with a self-supervised Next Scene Prediction task and supervised Decision Chain-of-Thought process. This dual mechanism encourages the model to align visual representations with actionable driving context, while promoting interpretable and causally grounded decision making. We curate a planning-oriented decision reasoning dataset, namely PDR, comprising 210k diverse and high-quality samples. Our method outperforms the mainstream E2E imitation learning method by a large margin of 19% L2 and 16.1 driving score on Bench2Drive benchmark. Furthermore, ReasonPlan demonstrates strong zero-shot generalization on unseen DOS benchmark, highlighting its adaptability in handling zero-shot corner cases. Code and dataset will be found in https://github.com/Liuxueyi/ReasonPlan.
>
---
#### [replaced 043] StableTracker: Learning to Stably Track Target via Differentiable Simulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.14147v2](http://arxiv.org/pdf/2509.14147v2)**

> **作者:** Fanxing Li; Shengyang Wang; Fangyu Sun; Shuyu Wu; Dexin Zuo; Wenxian Yu; Danping Zou
>
> **备注:** Corresponding author requires to do so
>
> **摘要:** FPV object tracking methods heavily rely on handcraft modular designs, resulting in hardware overload and cumulative error, which seriously degrades the tracking performance, especially for rapidly accelerating or decelerating targets. To address these challenges, we present \textbf{StableTracker}, a learning-based control policy that enables quadrotors to robustly follow the moving target from arbitrary perspectives. The policy is trained using backpropagation-through-time via differentiable simulation, allowing the quadrotor to maintain the target at the center of the visual field in both horizontal and vertical directions, while keeping a fixed relative distance, thereby functioning as an autonomous aerial camera. We compare StableTracker against both state-of-the-art traditional algorithms and learning baselines. Simulation experiments demonstrate that our policy achieves superior accuracy, stability and generalization across varying safe distances, trajectories, and target velocities. Furthermore, a real-world experiment on a quadrotor with an onboard computer validated practicality of the proposed approach.
>
---
#### [replaced 044] Mechanical Intelligence-Aware Curriculum Reinforcement Learning for Humanoids with Parallel Actuation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00273v2](http://arxiv.org/pdf/2507.00273v2)**

> **作者:** Yusuke Tanaka; Alvin Zhu; Quanyou Wang; Dennis Hong
>
> **备注:** Proceeding to the IEEE Humanoid Conference 2025
>
> **摘要:** Reinforcement learning (RL) has enabled advances in humanoid robot locomotion, yet most learning frameworks do not account for mechanical intelligence embedded in parallel actuation mechanisms due to limitations in simulator support for closed kinematic chains. This omission can lead to inaccurate motion modeling and suboptimal policies, particularly for robots with high actuation complexity. This paper presents general formulations and simulation methods for three types of parallel mechanisms: a differential pulley, a five-bar linkage, and a four-bar linkage, and trains a parallel-mechanism aware policy through an end-to-end curriculum RL framework for BRUCE, a kid-sized humanoid robot. Unlike prior approaches that rely on simplified serial approximations, we simulate all closed-chain constraints natively using GPU-accelerated MuJoCo (MJX), preserving the hardware's mechanical nonlinear properties during training. We benchmark our RL approach against a model predictive controller (MPC), demonstrating better surface generalization and performance in real-world zero-shot deployment. This work highlights the computational approaches and performance benefits of fully simulating parallel mechanisms in end-to-end learning pipelines for legged humanoids. Project codes with parallel mechanisms: https://github.com/alvister88/og_bruce
>
---
#### [replaced 045] A Scalable Multi-Robot Framework for Decentralized and Asynchronous Perception-Action-Communication Loops
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2309.10164v2](http://arxiv.org/pdf/2309.10164v2)**

> **作者:** Saurav Agarwal; Frederic Vatnsdal; Romina Garcia Camargo; Vijay Kumar; Alejandro Ribeiro
>
> **摘要:** Collaboration in large robot swarms to achieve a common global objective is a challenging problem in large environments due to limited sensing and communication capabilities. The robots must execute a Perception-Action-Communication (PAC) loop -- they perceive their local environment, communicate with other robots, and take actions in real time. A fundamental challenge in decentralized PAC systems is to decide what information to communicate with the neighboring robots and how to take actions while utilizing the information shared by the neighbors. Recently, this has been addressed using Graph Neural Networks (GNNs) for applications such as flocking and coverage control. Although conceptually, GNN policies are fully decentralized, the evaluation and deployment of such policies have primarily remained centralized or restrictively decentralized. Furthermore, existing frameworks assume sequential execution of perception and action inference, which is very restrictive in real-world applications. This paper proposes a framework for asynchronous PAC in robot swarms, where decentralized GNNs are used to compute navigation actions and generate messages for communication. In particular, we use aggregated GNNs, which enable the exchange of hidden layer information between robots for computational efficiency and decentralized inference of actions. Furthermore, the modules in the framework are asynchronous, allowing robots to perform sensing, extracting information, communication, action inference, and control execution at different frequencies. We demonstrate the effectiveness of GNNs executed in the proposed framework in navigating large robot swarms for collaborative coverage of large environments.
>
---
#### [replaced 046] Localized Graph-Based Neural Dynamics Models for Terrain Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.23270v2](http://arxiv.org/pdf/2503.23270v2)**

> **作者:** Chaoqi Liu; Yunzhu Li; Kris Hauser
>
> **摘要:** Predictive models can be particularly helpful for robots to effectively manipulate terrains in construction sites and extraterrestrial surfaces. However, terrain state representations become extremely high-dimensional especially to capture fine-resolution details and when depth is unknown or unbounded. This paper introduces a learning-based approach for terrain dynamics modeling and manipulation, leveraging the Graph-based Neural Dynamics (GBND) framework to represent terrain deformation as motion of a graph of particles. Based on the principle that the moving portion of a terrain is usually localized, our approach builds a large terrain graph (potentially millions of particles) but only identifies a very small active subgraph (hundreds of particles) for predicting the outcomes of robot-terrain interaction. To minimize the size of the active subgraph we introduce a learning-based approach that identifies a small region of interest (RoI) based on the robot's control inputs and the current scene. We also introduce a novel domain boundary feature encoding that allows GBNDs to perform accurate dynamics prediction in the RoI interior while avoiding particle penetration through RoI boundaries. Our proposed method is both orders of magnitude faster than naive GBND and it achieves better overall prediction accuracy. We further evaluated our framework on excavation and shaping tasks on terrain with different granularity.
>
---
#### [replaced 047] Learning Primitive Embodied World Models: Towards Scalable Robotic Learning
- **分类: cs.RO; cs.AI; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.20840v2](http://arxiv.org/pdf/2508.20840v2)**

> **作者:** Qiao Sun; Liujia Yang; Wei Tang; Wei Huang; Kaixin Xu; Yongchao Chen; Mingyu Liu; Jiange Yang; Haoyi Zhu; Yating Wang; Tong He; Yilun Chen; Xili Dai; Nanyang Ye; Qinying Gu
>
> **摘要:** While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a "GPT moment" in the embodied domain. There is a naive observation: the diversity of embodied data far exceeds the relatively small space of possible primitive motions. Based on this insight, we propose a novel paradigm for world modeling--Primitive Embodied World Models (PEWM). By restricting video generation to fixed short horizons, our approach 1) enables fine-grained alignment between linguistic concepts and visual representations of robotic actions, 2) reduces learning complexity, 3) improves data efficiency in embodied data collection, and 4) decreases inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence.
>
---
#### [replaced 048] Rapid and Safe Trajectory Planning over Diverse Scenes through Diffusion Composition
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.04384v2](http://arxiv.org/pdf/2507.04384v2)**

> **作者:** Wule Mao; Zhouheng Li; Yunhao Luo; Yilun Du; Lei Xie
>
> **摘要:** Safe trajectory planning in complex environments must balance stringent collision avoidance with real-time efficiency, which is a long-standing challenge in robotics. In this work, we present a diffusion-based trajectory planning framework that is both rapid and safe. First, we introduce a scene-agnostic, MPC-based data generation pipeline that efficiently produces large volumes of kinematically feasible trajectories. Building on this dataset, our integrated diffusion planner maps raw onboard sensor inputs directly to kinematically feasible trajectories, enabling efficient inference while maintaining strong collision avoidance. To generalize to diverse, previously unseen scenarios, we compose diffusion models at test time, enabling safe behavior without additional training. We further propose a lightweight, rule-based safety filter that, from the candidate set, selects the trajectory meeting safety and kinematic-feasibility requirements. Across seen and unseen settings, the proposed method delivers real-time-capable inference with high safety and stability. Experiments on an F1TENTH vehicle demonstrate practicality on real hardware. Project page: https://rstp-comp-diffuser.github.io/.
>
---
#### [replaced 049] The Sound of Simulation: Learning Multimodal Sim-to-Real Robot Policies with Generative Audio
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.02864v2](http://arxiv.org/pdf/2507.02864v2)**

> **作者:** Renhao Wang; Haoran Geng; Tingle Li; Feishi Wang; Gopala Anumanchipalli; Trevor Darrell; Boyi Li; Pieter Abbeel; Jitendra Malik; Alexei A. Efros
>
> **备注:** Conference on Robot Learning 2025
>
> **摘要:** Robots must integrate multiple sensory modalities to act effectively in the real world. Yet, learning such multimodal policies at scale remains challenging. Simulation offers a viable solution, but while vision has benefited from high-fidelity simulators, other modalities (e.g. sound) can be notoriously difficult to simulate. As a result, sim-to-real transfer has succeeded primarily in vision-based tasks, with multimodal transfer still largely unrealized. In this work, we tackle these challenges by introducing MultiGen, a framework that integrates large-scale generative models into traditional physics simulators, enabling multisensory simulation. We showcase our framework on the dynamic task of robot pouring, which inherently relies on multimodal feedback. By synthesizing realistic audio conditioned on simulation video, our method enables training on rich audiovisual trajectories -- without any real robot data. We demonstrate effective zero-shot transfer to real-world pouring with novel containers and liquids, highlighting the potential of generative modeling to both simulate hard-to-model modalities and close the multimodal sim-to-real gap.
>
---
