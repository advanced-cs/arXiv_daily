# 机器人 cs.RO

- **最新发布 38 篇**

- **更新 18 篇**

## 最新发布

#### [new 001] Jump-Start Reinforcement Learning with Self-Evolving Priors for Extreme Monopedal Locomotion
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人运动控制任务，解决极端欠驱动和复杂地形下的单足跳跃问题。通过自进化先验的强化学习框架，提升策略稳定性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.01243v1](http://arxiv.org/pdf/2507.01243v1)**

> **作者:** Ziang Zheng; Guojian Zhan; Shiqi Liu; Yao Lyu; Tao Zhang; Shengbo Eben Li
>
> **摘要:** Reinforcement learning (RL) has shown great potential in enabling quadruped robots to perform agile locomotion. However, directly training policies to simultaneously handle dual extreme challenges, i.e., extreme underactuation and extreme terrains, as in monopedal hopping tasks, remains highly challenging due to unstable early-stage interactions and unreliable reward feedback. To address this, we propose JumpER (jump-start reinforcement learning via self-evolving priors), an RL training framework that structures policy learning into multiple stages of increasing complexity. By dynamically generating self-evolving priors through iterative bootstrapping of previously learned policies, JumpER progressively refines and enhances guidance, thereby stabilizing exploration and policy optimization without relying on external expert priors or handcrafted reward shaping. Specifically, when integrated with a structured three-stage curriculum that incrementally evolves action modality, observation space, and task objective, JumpER enables quadruped robots to achieve robust monopedal hopping on unpredictable terrains for the first time. Remarkably, the resulting policy effectively handles challenging scenarios that traditional methods struggle to conquer, including wide gaps up to 60 cm, irregularly spaced stairs, and stepping stones with distances varying from 15 cm to 35 cm. JumpER thus provides a principled and scalable approach for addressing locomotion tasks under the dual challenges of extreme underactuation and extreme terrains.
>
---
#### [new 002] VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting
- **分类: cs.RO**

- **简介: 该论文提出VISTA系统，用于机器人在开放词汇指令下进行语义引导的探索，解决如何高效构建任务相关3D地图的问题。**

- **链接: [http://arxiv.org/pdf/2507.01125v1](http://arxiv.org/pdf/2507.01125v1)**

> **作者:** Keiko Nagami; Timothy Chen; Javier Yu; Ola Shorinwa; Maximilian Adang; Carlyn Dougherty; Eric Cristofalo; Mac Schwager
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** We present VISTA (Viewpoint-based Image selection with Semantic Task Awareness), an active exploration method for robots to plan informative trajectories that improve 3D map quality in areas most relevant for task completion. Given an open-vocabulary search instruction (e.g., "find a person"), VISTA enables a robot to explore its environment to search for the object of interest, while simultaneously building a real-time semantic 3D Gaussian Splatting reconstruction of the scene. The robot navigates its environment by planning receding-horizon trajectories that prioritize semantic similarity to the query and exploration of unseen regions of the environment. To evaluate trajectories, VISTA introduces a novel, efficient viewpoint-semantic coverage metric that quantifies both the geometric view diversity and task relevance in the 3D scene. On static datasets, our coverage metric outperforms state-of-the-art baselines, FisherRF and Bayes' Rays, in computation speed and reconstruction quality. In quadrotor hardware experiments, VISTA achieves 6x higher success rates in challenging maps, compared to baseline methods, while matching baseline performance in less challenging maps. Lastly, we show that VISTA is platform-agnostic by deploying it on a quadrotor drone and a Spot quadruped robot. Open-source code will be released upon acceptance of the paper.
>
---
#### [new 003] LANet: A Lane Boundaries-Aware Approach For Robust Trajectory Prediction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于轨迹预测任务，旨在提升自动驾驶中对驾驶环境的感知与预测能力。针对现有方法依赖车道中心线的局限，提出融合多向量地图元素的模型，并采用特征融合与连接剪枝策略，提高预测精度与效率。**

- **链接: [http://arxiv.org/pdf/2507.01308v1](http://arxiv.org/pdf/2507.01308v1)**

> **作者:** Muhammad Atta ur Rahman; Dooseop Choi; KyoungWook Min
>
> **备注:** Accepted at the 17th IEEE International Conference on Advanced Computational Intelligence (ICACI 2025)
>
> **摘要:** Accurate motion forecasting is critical for safe and efficient autonomous driving, enabling vehicles to predict future trajectories and make informed decisions in complex traffic scenarios. Most of the current designs of motion prediction models are based on the major representation of lane centerlines, which limits their capability to capture critical road environments and traffic rules and constraints. In this work, we propose an enhanced motion forecasting model informed by multiple vector map elements, including lane boundaries and road edges, that facilitates a richer and more complete representation of driving environments. An effective feature fusion strategy is developed to merge information in different vector map components, where the model learns holistic information on road structures and their interactions with agents. Since encoding more information about the road environment increases memory usage and is computationally expensive, we developed an effective pruning mechanism that filters the most relevant map connections to the target agent, ensuring computational efficiency while maintaining essential spatial and semantic relationships for accurate trajectory prediction. Overcoming the limitations of lane centerline-based models, our method provides a more informative and efficient representation of the driving environment and advances the state of the art for autonomous vehicle motion forecasting. We verify our approach with extensive experiments on the Argoverse 2 motion forecasting dataset, where our method maintains competitiveness on AV2 while achieving improved performance. Index Terms-Autonomous driving, trajectory prediction, vector map elements, road topology, connection pruning, Argoverse 2.
>
---
#### [new 004] Efficient Collision Detection for Long and Slender Robotic Links in Euclidean Distance Fields: Application to a Forestry Crane
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决长细机械臂在复杂环境中的碰撞检测问题。提出一种高效算法，提升计算效率并避免参数调整。**

- **链接: [http://arxiv.org/pdf/2507.01705v1](http://arxiv.org/pdf/2507.01705v1)**

> **作者:** Marc-Philip Ecker; Bernhard Bischof; Minh Nhat Vu; Christoph Fröhlich; Tobias Glück; Wolfgang Kemmetmüller
>
> **备注:** Accepted at IROS 2025
>
> **摘要:** Collision-free motion planning in complex outdoor environments relies heavily on perceiving the surroundings through exteroceptive sensors. A widely used approach represents the environment as a voxelized Euclidean distance field, where robots are typically approximated by spheres. However, for large-scale manipulators such as forestry cranes, which feature long and slender links, this conventional spherical approximation becomes inefficient and inaccurate. This work presents a novel collision detection algorithm specifically designed to exploit the elongated structure of such manipulators, significantly enhancing the computational efficiency of motion planning algorithms. Unlike traditional sphere decomposition methods, our approach not only improves computational efficiency but also naturally eliminates the need to fine-tune the approximation accuracy as an additional parameter. We validate the algorithm's effectiveness using real-world LiDAR data from a forestry crane application, as well as simulated environment data.
>
---
#### [new 005] LLM-based Realistic Safety-Critical Driving Video Generation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶仿真任务，旨在生成真实且安全关键的驾驶场景。通过LLM和CARLA模拟器，自动合成包含碰撞等关键事件的场景，并生成逼真视频，提升测试效果。**

- **链接: [http://arxiv.org/pdf/2507.01264v1](http://arxiv.org/pdf/2507.01264v1)**

> **作者:** Yongjie Fu; Ruijian Zha; Pei Tian; Xuan Di
>
> **摘要:** Designing diverse and safety-critical driving scenarios is essential for evaluating autonomous driving systems. In this paper, we propose a novel framework that leverages Large Language Models (LLMs) for few-shot code generation to automatically synthesize driving scenarios within the CARLA simulator, which has flexibility in scenario scripting, efficient code-based control of traffic participants, and enforcement of realistic physical dynamics. Given a few example prompts and code samples, the LLM generates safety-critical scenario scripts that specify the behavior and placement of traffic participants, with a particular focus on collision events. To bridge the gap between simulation and real-world appearance, we integrate a video generation pipeline using Cosmos-Transfer1 with ControlNet, which converts rendered scenes into realistic driving videos. Our approach enables controllable scenario generation and facilitates the creation of rare but critical edge cases, such as pedestrian crossings under occlusion or sudden vehicle cut-ins. Experimental results demonstrate the effectiveness of our method in generating a wide range of realistic, diverse, and safety-critical scenarios, offering a promising tool for simulation-based testing of autonomous vehicles.
>
---
#### [new 006] Environment-Aware and Human-Cooperative Swing Control for Lower-Limb Prostheses in Diverse Obstacle Scenarios
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于下肢假肢控制任务，旨在解决复杂地形中障碍物跨越问题。通过融合环境感知与用户意图，提出新型控制策略，提升假肢适应性与安全性。**

- **链接: [http://arxiv.org/pdf/2507.01111v1](http://arxiv.org/pdf/2507.01111v1)**

> **作者:** Haosen Xing; Haoran Ma; Sijin Zhang; Hartmut Geyer
>
> **摘要:** Current control strategies for powered lower limb prostheses often lack awareness of the environment and the user's intended interactions with it. This limitation becomes particularly apparent in complex terrains. Obstacle negotiation, a critical scenario exemplifying such challenges, requires both real-time perception of obstacle geometry and responsiveness to user intention about when and where to step over or onto, to dynamically adjust swing trajectories. We propose a novel control strategy that fuses environmental awareness and human cooperativeness: an on-board depth camera detects obstacles ahead of swing phase, prompting an elevated early-swing trajectory to ensure clearance, while late-swing control defers to natural biomechanical cues from the user. This approach enables intuitive stepping strategies without requiring unnatural movement patterns. Experiments with three non-amputee participants demonstrated 100 percent success across more than 150 step-overs and 30 step-ons with randomly placed obstacles of varying heights (4-16 cm) and distances (15-70 cm). By effectively addressing obstacle navigation -- a gateway challenge for complex terrain mobility -- our system demonstrates adaptability to both environmental constraints and user intentions, with promising applications across diverse locomotion scenarios.
>
---
#### [new 007] Dynamic System Model Generation for Online Fault Detection and Diagnosis of Robotic Systems
- **分类: cs.RO**

- **简介: 该论文属于故障检测与诊断任务，旨在解决复杂机器人系统中传统模型和历史数据不足的问题。通过实时生成动态系统模型，提高故障定位的准确性和效率。**

- **链接: [http://arxiv.org/pdf/2507.01550v1](http://arxiv.org/pdf/2507.01550v1)**

> **作者:** Johannes Kohl; Georg Muck; Georg Jäger; Sebastian Zug
>
> **备注:** Accepted for publication in Ada User Journal
>
> **摘要:** With the rapid development of more complex robots, Fault Detection and Diagnosis (FDD) becomes increasingly harder. Especially the need for predetermined models and historic data is problematic because they do not encompass the dynamic and fast-changing nature of such systems. To this end, we propose a concept that actively generates a dynamic system model at runtime and utilizes it to locate root causes. The goal is to be applicable to all kinds of robotic systems that share a similar software design. Additionally, it should exhibit minimal overhead and enhance independence from expert attention.
>
---
#### [new 008] Augmented Bridge Spinal Fixation: A New Concept for Addressing Pedicle Screw Pullout via a Steerable Drilling Robot and Flexible Pedicle Screws
- **分类: cs.RO**

- **简介: 该论文属于脊柱固定技术领域，旨在解决传统椎弓根螺钉松动问题。通过机器人钻孔和柔性螺钉构建增强桥接结构，提升固定强度。**

- **链接: [http://arxiv.org/pdf/2507.01753v1](http://arxiv.org/pdf/2507.01753v1)**

> **作者:** Yash Kulkarni; Susheela Sharma; Omid Rezayof; Siddhartha Kapuria; Jordan P. Amadio; Mohsen Khadem; Maryam Tilton; Farshid Alambeigi
>
> **摘要:** To address the screw loosening and pullout limitations of rigid pedicle screws in spinal fixation procedures, and to leverage our recently developed Concentric Tube Steerable Drilling Robot (CT-SDR) and Flexible Pedicle Screw (FPS), in this paper, we introduce the concept of Augmented Bridge Spinal Fixation (AB-SF). In this concept, two connecting J-shape tunnels are first drilled through pedicles of vertebra using the CT-SDR. Next, two FPSs are passed through this tunnel and bone cement is then injected through the cannulated region of the FPS to form an augmented bridge between two pedicles and reinforce strength of the fixated spine. To experimentally analyze and study the feasibility of AB-SF technique, we first used our robotic system (i.e., a CT-SDR integrated with a robotic arm) to create two different fixation scenarios in which two J-shape tunnels, forming a bridge, were drilled at different depth of a vertebral phantom. Next, we implanted two FPSs within the drilled tunnels and then successfully simulated the bone cement augmentation process.
>
---
#### [new 009] A Survey on Vision-Language-Action Models: An Action Tokenization Perspective
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型研究，旨在解决VLA模型中行动标记的统一理解问题，通过分类与分析不同类型的行动标记，推动模型发展。**

- **链接: [http://arxiv.org/pdf/2507.01925v1](http://arxiv.org/pdf/2507.01925v1)**

> **作者:** Yifan Zhong; Fengshuo Bai; Shaofei Cai; Xuchuan Huang; Zhang Chen; Xiaowei Zhang; Yuanfei Wang; Shaoyang Guo; Tianrui Guan; Ka Nam Lui; Zhiquan Qi; Yitao Liang; Yuanpei Chen; Yaodong Yang
>
> **备注:** 70 pages, 5 figures
>
> **摘要:** The remarkable advancements of vision and language foundation models in multimodal understanding, reasoning, and generation has sparked growing efforts to extend such intelligence to the physical world, fueling the flourishing of vision-language-action (VLA) models. Despite seemingly diverse approaches, we observe that current VLA models can be unified under a single framework: vision and language inputs are processed by a series of VLA modules, producing a chain of \textit{action tokens} that progressively encode more grounded and actionable information, ultimately generating executable actions. We further determine that the primary design choice distinguishing VLA models lies in how action tokens are formulated, which can be categorized into language description, code, affordance, trajectory, goal state, latent representation, raw action, and reasoning. However, there remains a lack of comprehensive understanding regarding action tokens, significantly impeding effective VLA development and obscuring future directions. Therefore, this survey aims to categorize and interpret existing VLA research through the lens of action tokenization, distill the strengths and limitations of each token type, and identify areas for improvement. Through this systematic review and analysis, we offer a synthesized outlook on the broader evolution of VLA models, highlight underexplored yet promising directions, and contribute guidance for future research, hoping to bring the field closer to general-purpose intelligence.
>
---
#### [new 010] A Review on Sound Source Localization in Robotics: Focusing on Deep Learning Methods
- **分类: cs.RO; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于声音源定位任务，解决机器人中声源定位问题，综述了传统方法和深度学习方法，分析了数据与训练策略，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2507.01143v1](http://arxiv.org/pdf/2507.01143v1)**

> **作者:** Reza Jalayer; Masoud Jalayer; Amirali Baniasadi
>
> **备注:** 35 pages
>
> **摘要:** Sound source localization (SSL) adds a spatial dimension to auditory perception, allowing a system to pinpoint the origin of speech, machinery noise, warning tones, or other acoustic events, capabilities that facilitate robot navigation, human-machine dialogue, and condition monitoring. While existing surveys provide valuable historical context, they typically address general audio applications and do not fully account for robotic constraints or the latest advancements in deep learning. This review addresses these gaps by offering a robotics-focused synthesis, emphasizing recent progress in deep learning methodologies. We start by reviewing classical methods such as Time Difference of Arrival (TDOA), beamforming, Steered-Response Power (SRP), and subspace analysis. Subsequently, we delve into modern machine learning (ML) and deep learning (DL) approaches, discussing traditional ML and neural networks (NNs), convolutional neural networks (CNNs), convolutional recurrent neural networks (CRNNs), and emerging attention-based architectures. The data and training strategy that are the two cornerstones of DL-based SSL are explored. Studies are further categorized by robot types and application domains to facilitate researchers in identifying relevant work for their specific contexts. Finally, we highlight the current challenges in SSL works in general, regarding environmental robustness, sound source multiplicity, and specific implementation constraints in robotics, as well as data and learning strategies in DL-based SSL. Also, we sketch promising directions to offer an actionable roadmap toward robust, adaptable, efficient, and explainable DL-based SSL for next-generation robots.
>
---
#### [new 011] AC-DiT: Adaptive Coordination Diffusion Transformer for Mobile Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于移动操作任务，旨在解决移动基座与机械臂协调控制的问题。提出AC-DiT模型，通过动态感知和条件机制提升整体控制效果。**

- **链接: [http://arxiv.org/pdf/2507.01961v1](http://arxiv.org/pdf/2507.01961v1)**

> **作者:** Sixiang Chen; Jiaming Liu; Siyuan Qian; Han Jiang; Lily Li; Renrui Zhang; Zhuoyang Liu; Chenyang Gu; Chengkai Hou; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang
>
> **摘要:** Recently, mobile manipulation has attracted increasing attention for enabling language-conditioned robotic control in household tasks. However, existing methods still face challenges in coordinating mobile base and manipulator, primarily due to two limitations. On the one hand, they fail to explicitly model the influence of the mobile base on manipulator control, which easily leads to error accumulation under high degrees of freedom. On the other hand, they treat the entire mobile manipulation process with the same visual observation modality (e.g., either all 2D or all 3D), overlooking the distinct multimodal perception requirements at different stages during mobile manipulation. To address this, we propose the Adaptive Coordination Diffusion Transformer (AC-DiT), which enhances mobile base and manipulator coordination for end-to-end mobile manipulation. First, since the motion of the mobile base directly influences the manipulator's actions, we introduce a mobility-to-body conditioning mechanism that guides the model to first extract base motion representations, which are then used as context prior for predicting whole-body actions. This enables whole-body control that accounts for the potential impact of the mobile base's motion. Second, to meet the perception requirements at different stages of mobile manipulation, we design a perception-aware multimodal conditioning strategy that dynamically adjusts the fusion weights between various 2D visual images and 3D point clouds, yielding visual features tailored to the current perceptual needs. This allows the model to, for example, adaptively rely more on 2D inputs when semantic information is crucial for action prediction, while placing greater emphasis on 3D geometric information when precise spatial understanding is required. We validate AC-DiT through extensive experiments on both simulated and real-world mobile manipulation tasks.
>
---
#### [new 012] SE(3)-Equivariant Diffusion Policy in Spherical Fourier Space
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，旨在解决扩散策略在3D场景中泛化能力差的问题。通过引入SE(3)等变的球面傅里叶空间方法提升性能。**

- **链接: [http://arxiv.org/pdf/2507.01723v1](http://arxiv.org/pdf/2507.01723v1)**

> **作者:** Xupeng Zhu; Fan Wang; Robin Walters; Jane Shi
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Diffusion Policies are effective at learning closed-loop manipulation policies from human demonstrations but generalize poorly to novel arrangements of objects in 3D space, hurting real-world performance. To address this issue, we propose Spherical Diffusion Policy (SDP), an SE(3) equivariant diffusion policy that adapts trajectories according to 3D transformations of the scene. Such equivariance is achieved by embedding the states, actions, and the denoising process in spherical Fourier space. Additionally, we employ novel spherical FiLM layers to condition the action denoising process equivariantly on the scene embeddings. Lastly, we propose a spherical denoising temporal U-net that achieves spatiotemporal equivariance with computational efficiency. In the end, SDP is end-to-end SE(3) equivariant, allowing robust generalization across transformed 3D scenes. SDP demonstrates a large performance improvement over strong baselines in 20 simulation tasks and 5 physical robot tasks including single-arm and bi-manual embodiments. Code is available at https://github.com/amazon-science/Spherical_Diffusion_Policy.
>
---
#### [new 013] 2024 NASA SUITS Report: LLM-Driven Immersive Augmented Reality User Interface for Robotics and Space Exploration
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决AR环境下机器人定位与控制问题，提出URSA系统实现非侵入式交互和实时监控。**

- **链接: [http://arxiv.org/pdf/2507.01206v1](http://arxiv.org/pdf/2507.01206v1)**

> **作者:** Kathy Zhuang; Zixun Huang; Yukun Song; Rui Li; Yinuo Zhou; Allen Y. Yang
>
> **摘要:** As modern computing advances, new interaction paradigms have emerged, particularly in Augmented Reality (AR), which overlays virtual interfaces onto physical objects. This evolution poses challenges in machine perception, especially for tasks like 3D object pose estimation in complex, dynamic environments. Our project addresses critical issues in human-robot interaction within mobile AR, focusing on non-intrusive, spatially aware interfaces. We present URSA, an LLM-driven immersive AR system developed for NASA's 2023-2024 SUITS challenge, targeting future spaceflight needs such as the Artemis missions. URSA integrates three core technologies: a head-mounted AR device (e.g., HoloLens) for intuitive visual feedback, voice control powered by large language models for hands-free interaction, and robot tracking algorithms that enable accurate 3D localization in dynamic settings. To enhance precision, we leverage digital twin localization technologies, using datasets like DTTD-Mobile and specialized hardware such as the ZED2 camera for real-world tracking under noise and occlusion. Our system enables real-time robot control and monitoring via an AR interface, even in the absence of ground-truth sensors--vital for hazardous or remote operations. Key contributions include: (1) a non-intrusive AR interface with LLM-based voice input; (2) a ZED2-based dataset tailored for non-rigid robotic bodies; (3) a Local Mission Control Console (LMCC) for mission visualization; (4) a transformer-based 6DoF pose estimator (DTTDNet) optimized for depth fusion and real-time tracking; and (5) end-to-end integration for astronaut mission support. This work advances digital twin applications in robotics, offering scalable solutions for both aerospace and industrial domains.
>
---
#### [new 014] S3D: A Spatial Steerable Surgical Drilling Framework for Robotic Spinal Fixation Procedures
- **分类: cs.RO**

- **简介: 该论文属于机器人脊柱固定手术任务，旨在解决椎体钻孔的精准导航问题。通过改进机械臂和校准流程，实现多椎体水平的可操控钻孔。**

- **链接: [http://arxiv.org/pdf/2507.01779v1](http://arxiv.org/pdf/2507.01779v1)**

> **作者:** Daniyal Maroufi; Xinyuan Huang; Yash Kulkarni; Omid Rezayof; Susheela Sharma; Vaibhav Goggela; Jordan P. Amadio; Mohsen Khadem; Farshid Alambeigi
>
> **摘要:** In this paper, we introduce S3D: A Spatial Steerable Surgical Drilling Framework for Robotic Spinal Fixation Procedures. S3D is designed to enable realistic steerable drilling while accounting for the anatomical constraints associated with vertebral access in spinal fixation (SF) procedures. To achieve this, we first enhanced our previously designed concentric tube Steerable Drilling Robot (CT-SDR) to facilitate steerable drilling across all vertebral levels of the spinal column. Additionally, we propose a four-Phase calibration, registration, and navigation procedure to perform realistic SF procedures on a spine holder phantom by integrating the CT-SDR with a seven-degree-of-freedom robotic manipulator. The functionality of this framework is validated through planar and out-of-plane steerable drilling experiments in vertebral phantoms.
>
---
#### [new 015] TriVLA: A Unified Triple-System-Based Unified Vision-Language-Action Model for General Robot Control
- **分类: cs.RO**

- **简介: 该论文提出TriVLA模型，用于机器人通用控制任务，解决静态信息忽略和动态感知不足的问题，通过三系统架构提升机器人环境理解与动作生成能力。**

- **链接: [http://arxiv.org/pdf/2507.01424v1](http://arxiv.org/pdf/2507.01424v1)**

> **作者:** Zhenyang Liu; Yongchong Gu; Sixiao Zheng; Xiangyang Xue; Yanwei Fu
>
> **摘要:** Recent advancements in vision-language models (VLMs) for common-sense reasoning have led to the development of vision-language-action (VLA) models, enabling robots to perform generalized manipulation. Although existing autoregressive VLA methods design a specific architecture like dual-system to leverage large-scale pretrained knowledge, they tend to capture static information, often neglecting the dynamic aspects vital for embodied tasks. To this end, we propose TriVLA, a unified Vision-Language-Action model with a triple-system architecture for general robot control. The vision-language module (System 2) interprets the environment through vision and language instructions. The dynamics perception module (System 3) inherently produces visual representations that encompass both current static information and predicted future dynamics, thereby providing valuable guidance for policy learning. TriVLA utilizes pre-trained VLM model and fine-tunes pre-trained video foundation model on robot datasets along with internet human manipulation data. The subsequent policy learning module (System 1) generates fluid motor actions in real time. Experimental evaluation demonstrates that TriVLA operates at approximately 36 Hz and surpasses state-of-the-art imitation learning baselines on standard simulation benchmarks as well as challenging real-world manipulation tasks.
>
---
#### [new 016] MoIRA: Modular Instruction Routing Architecture for Multi-Task Robotics
- **分类: cs.RO**

- **简介: 该论文提出MoIRA，一种模块化多任务机器人架构，解决专家系统路由问题。通过文本引导的路由机制，提升任务执行效率与灵活性。**

- **链接: [http://arxiv.org/pdf/2507.01843v1](http://arxiv.org/pdf/2507.01843v1)**

> **作者:** Dmytro Kuzmenko; Nadiya Shvai
>
> **备注:** Preprint of a manuscript submitted for peer review
>
> **摘要:** Mixture-of-Experts (MoE) approaches have recently gained traction in robotics applications due to their ability to dynamically allocate computational resources and specialize sub-networks for distinct tasks or environmental contexts, enabling more efficient decision-making. Such systems often comprise sparsely activated experts combined under a single monolithic architecture and require a well-configured internal routing mechanism, which does not allow for selective low-level expert and router customization and requires additional training. We propose MoIRA, an architecture-agnostic modular MoE framework designed to coordinate existing experts with an external text-based router. MoIRA incorporates two zero-shot routing options: embedding-based similarity and prompt-driven language model inference. In our experiments, we choose large Vision-Language-Action models, gr00t-N1 and $\pi_0$, as the underlying experts, and train low-rank adapters for low-overhead inference. We evaluate MoIRA on various GR1 Humanoid tasks and LIBERO Spatial and Goal benchmarks, where it consistently outperforms generalist models and competes with other MoE pipelines. Additionally, we analyse the robustness of the proposed approach to the variations of the instructions. While relying solely on textual descriptions of tasks and experts, MoIRA demonstrates the practical viability of modular deployment with precise, low-effort routing and provides an alternative, scalable foundation for future multi-expert robotic systems.
>
---
#### [new 017] An RRT* algorithm based on Riemannian metric model for optimal path planning
- **分类: cs.RO; math.OC; 00A69, 93C85, 14H55; I.2.9**

- **简介: 该论文属于路径规划任务，解决高维空间中最优路径问题。通过构建黎曼度量模型，将问题转化为二维平面几何问题，并提出改进的RRT*-R算法。**

- **链接: [http://arxiv.org/pdf/2507.01697v1](http://arxiv.org/pdf/2507.01697v1)**

> **作者:** Yu Zhang; Qi Zhou; Xiao-Song Yang
>
> **备注:** 27 pages
>
> **摘要:** This paper presents a Riemannian metric-based model to solve the optimal path planning problem on two-dimensional smooth submanifolds in high-dimensional space. Our model is based on constructing a new Riemannian metric on a two-dimensional projection plane, which is induced by the high-dimensional Euclidean metric on two-dimensional smooth submanifold and reflects the environmental information of the robot. The optimal path planning problem in high-dimensional space is therefore transformed into a geometric problem on the two-dimensional plane with new Riemannian metric. Based on the new Riemannian metric, we proposed an incremental algorithm RRT*-R on the projection plane. The experimental results show that the proposed algorithm is suitable for scenarios with uneven fields in multiple dimensions. The proposed algorithm can help the robot to effectively avoid areas with drastic changes in height, ground resistance and other environmental factors. More importantly, the RRT*-R algorithm shows better smoothness and optimization properties compared with the original RRT* algorithm using Euclidean distance in high-dimensional workspace. The length of the entire path by RRT*-R is a good approximation of the theoretical minimum geodesic distance on projection plane.
>
---
#### [new 018] Towards Design and Development of a Concentric Tube Steerable Drilling Robot for Creating S-shape Tunnels for Pelvic Fixation Procedures
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决骨盆固定中钻孔路径受限的问题。设计了一种四自由度的同心管可操控钻孔机器人，以实现更自然的S型钻孔路径。**

- **链接: [http://arxiv.org/pdf/2507.01811v1](http://arxiv.org/pdf/2507.01811v1)**

> **作者:** Yash Kulkarni; Susheela Sharma; Sarah Go; Jordan P. Amadio; Mohsen Khadem; Farshid Alambeigi
>
> **摘要:** Current pelvic fixation techniques rely on rigid drilling tools, which inherently constrain the placement of rigid medical screws in the complex anatomy of pelvis. These constraints prevent medical screws from following anatomically optimal pathways and force clinicians to fixate screws in linear trajectories. This suboptimal approach, combined with the unnatural placement of the excessively long screws, lead to complications such as screw misplacement, extended surgery times, and increased radiation exposure due to repeated X-ray images taken ensure to safety of procedure. To address these challenges, in this paper, we present the design and development of a unique 4 degree-of-freedom (DoF) pelvic concentric tube steerable drilling robot (pelvic CT-SDR). The pelvic CT-SDR is capable of creating long S-shaped drilling trajectories that follow the natural curvatures of the pelvic anatomy. The performance of the pelvic CT-SDR was thoroughly evaluated through several S-shape drilling experiments in simulated bone phantoms.
>
---
#### [new 019] Search-Based Robot Motion Planning With Distance-Based Adaptive Motion Primitives
- **分类: cs.RO; cs.AI; cs.CG**

- **简介: 该论文属于机器人运动规划任务，旨在解决高自由度机械臂在复杂环境中的路径规划问题。通过引入自适应运动基元（bur）提升搜索效率。**

- **链接: [http://arxiv.org/pdf/2507.01198v1](http://arxiv.org/pdf/2507.01198v1)**

> **作者:** Benjamin Kraljusic; Zlatan Ajanovic; Nermin Covic; Bakir Lacevic
>
> **备注:** 6 pages, 3 figures, submitted to a conference
>
> **摘要:** This work proposes a motion planning algorithm for robotic manipulators that combines sampling-based and search-based planning methods. The core contribution of the proposed approach is the usage of burs of free configuration space (C-space) as adaptive motion primitives within the graph search algorithm. Due to their feature to adaptively expand in free C-space, burs enable more efficient exploration of the configuration space compared to fixed-sized motion primitives, significantly reducing the time to find a valid path and the number of required expansions. The algorithm is implemented within the existing SMPL (Search-Based Motion Planning Library) library and evaluated through a series of different scenarios involving manipulators with varying number of degrees-of-freedom (DoF) and environment complexity. Results demonstrate that the bur-based approach outperforms fixed-primitive planning in complex scenarios, particularly for high DoF manipulators, while achieving comparable performance in simpler scenarios.
>
---
#### [new 020] VLAD: A VLM-Augmented Autonomous Driving Framework with Hierarchical Planning and Interpretable Decision Process
- **分类: cs.RO; cs.AI; cs.CV; cs.ET; cs.LG**

- **简介: 该论文属于自动驾驶任务，旨在提升感知、预测与规划能力。通过融合视觉语言模型，增强空间推理并生成可解释决策，降低碰撞率。**

- **链接: [http://arxiv.org/pdf/2507.01284v1](http://arxiv.org/pdf/2507.01284v1)**

> **作者:** Cristian Gariboldi; Hayato Tokida; Ken Kinjo; Yuki Asada; Alexander Carballo
>
> **备注:** 2025 IEEE 28th International Conference on Intelligent Transportation Systems (ITSC)
>
> **摘要:** Recent advancements in open-source Visual Language Models (VLMs) such as LLaVA, Qwen-VL, and Llama have catalyzed extensive research on their integration with diverse systems. The internet-scale general knowledge encapsulated within these models presents significant opportunities for enhancing autonomous driving perception, prediction, and planning capabilities. In this paper we propose VLAD, a vision-language autonomous driving model, which integrates a fine-tuned VLM with VAD, a state-of-the-art end-to-end system. We implement a specialized fine-tuning approach using custom question-answer datasets designed specifically to improve the spatial reasoning capabilities of the model. The enhanced VLM generates high-level navigational commands that VAD subsequently processes to guide vehicle operation. Additionally, our system produces interpretable natural language explanations of driving decisions, thereby increasing transparency and trustworthiness of the traditionally black-box end-to-end architecture. Comprehensive evaluation on the real-world nuScenes dataset demonstrates that our integrated system reduces average collision rates by 31.82% compared to baseline methodologies, establishing a new benchmark for VLM-augmented autonomous driving systems.
>
---
#### [new 021] SonoGym: High Performance Simulation for Challenging Surgical Tasks with Robotic Ultrasound
- **分类: cs.RO**

- **简介: 该论文提出SonoGym，用于复杂骨科手术中机器人超声的高性能仿真，解决真实环境训练困难的问题，支持DRL和IL方法训练。**

- **链接: [http://arxiv.org/pdf/2507.01152v1](http://arxiv.org/pdf/2507.01152v1)**

> **作者:** Yunke Ao; Masoud Moghani; Mayank Mittal; Manish Prajapat; Luohong Wu; Frederic Giraud; Fabio Carrillo; Andreas Krause; Philipp Fürnstahl
>
> **备注:** 21 pages, 15 figures
>
> **摘要:** Ultrasound (US) is a widely used medical imaging modality due to its real-time capabilities, non-invasive nature, and cost-effectiveness. Robotic ultrasound can further enhance its utility by reducing operator dependence and improving access to complex anatomical regions. For this, while deep reinforcement learning (DRL) and imitation learning (IL) have shown potential for autonomous navigation, their use in complex surgical tasks such as anatomy reconstruction and surgical guidance remains limited -- largely due to the lack of realistic and efficient simulation environments tailored to these tasks. We introduce SonoGym, a scalable simulation platform for complex robotic ultrasound tasks that enables parallel simulation across tens to hundreds of environments. Our framework supports realistic and real-time simulation of US data from CT-derived 3D models of the anatomy through both a physics-based and a generative modeling approach. Sonogym enables the training of DRL and recent IL agents (vision transformers and diffusion policies) for relevant tasks in robotic orthopedic surgery by integrating common robotic platforms and orthopedic end effectors. We further incorporate submodular DRL -- a recent method that handles history-dependent rewards -- for anatomy reconstruction and safe reinforcement learning for surgery. Our results demonstrate successful policy learning across a range of scenarios, while also highlighting the limitations of current methods in clinically relevant environments. We believe our simulation can facilitate research in robot learning approaches for such challenging robotic surgery applications. Dataset, codes, and videos are publicly available at https://sonogym.github.io/.
>
---
#### [new 022] Approximation-free Control of Unknown Euler-Lagrangian Systems under Input Constraints
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人控制任务，解决未知动力系统在输入约束下的跟踪问题，提出两种无需近似的控制策略以确保安全与性能。**

- **链接: [http://arxiv.org/pdf/2507.01426v1](http://arxiv.org/pdf/2507.01426v1)**

> **作者:** Ratnangshu Das; Pushpak Jagtap
>
> **摘要:** In this paper, we present a novel funnel-based tracking control algorithm for robotic systems with unknown dynamics and prescribed input constraints. The Euler-Lagrange formulation, a common modeling approach for robotic systems, has been adopted in this study to address the trade-off between performance and actuator safety. We establish feasibility conditions that ensure tracking errors evolve within predefined funnel bounds while maintaining bounded control efforts, a crucial consideration for robots with limited actuation capabilities. We propose two approximation-free control strategies for scenarios where these conditions are violated: one actively corrects the error, and the other stops further deviation. Finally, we demonstrate the robust performance and safety of the approach through simulations and experimental validations. This work represents a significant advancement in funnel-based control, enhancing its applicability to real-world robotics systems with input constraints.
>
---
#### [new 023] TypeTele: Releasing Dexterity in Teleoperation by Dexterous Manipulation Types
- **分类: cs.RO**

- **简介: 该论文属于机器人遥操作任务，旨在解决传统方法依赖人类手部动作限制的问题。通过引入操作类型，提升机械手的灵巧操作能力。**

- **链接: [http://arxiv.org/pdf/2507.01857v1](http://arxiv.org/pdf/2507.01857v1)**

> **作者:** Yuhao Lin; Yi-Lin Wei; Haoran Liao; Mu Lin; Chengyi Xing; Hao Li; Dandan Zhang; Mark Cutkosky; Wei-Shi Zheng
>
> **备注:** Project Page: https://isee-laboratory.github.io/TypeTele
>
> **摘要:** Dexterous teleoperation plays a crucial role in robotic manipulation for real-world data collection and remote robot control. Previous dexterous teleoperation mostly relies on hand retargeting to closely mimic human hand postures. However, these approaches may fail to fully leverage the inherent dexterity of dexterous hands, which can execute unique actions through their structural advantages compared to human hands. To address this limitation, we propose TypeTele, a type-guided dexterous teleoperation system, which enables dexterous hands to perform actions that are not constrained by human motion patterns. This is achieved by introducing dexterous manipulation types into the teleoperation system, allowing operators to employ appropriate types to complete specific tasks. To support this system, we build an extensible dexterous manipulation type library to cover comprehensive dexterous postures used in manipulation tasks. During teleoperation, we employ a MLLM (Multi-modality Large Language Model)-assisted type retrieval module to identify the most suitable manipulation type based on the specific task and operator commands. Extensive experiments of real-world teleoperation and imitation learning demonstrate that the incorporation of manipulation types significantly takes full advantage of the dexterous robot's ability to perform diverse and complex tasks with higher success rates.
>
---
#### [new 024] A Differentiable Distance Metric for Robotics Through Generalized Alternating Projection
- **分类: cs.RO**

- **简介: 该论文属于机器人学任务，解决传统距离度量不可导的问题。提出一种可微分距离度量，改进投影计算并确保重叠时距离为零。**

- **链接: [http://arxiv.org/pdf/2507.01181v1](http://arxiv.org/pdf/2507.01181v1)**

> **作者:** Vinicius M. Gonçalves; Shiqing Wei; Eduardo Malacarne S. de Souza; Krishnamurthy Prashanth; Anthony Tzes; Farshad Khorrami
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** In many robotics applications, it is necessary to compute not only the distance between the robot and the environment, but also its derivative - for example, when using control barrier functions. However, since the traditional Euclidean distance is not differentiable, there is a need for alternative distance metrics that possess this property. Recently, a metric with guaranteed differentiability was proposed [1]. This approach has some important drawbacks, which we address in this paper. We provide much simpler and practical expressions for the smooth projection for general convex polytopes. Additionally, as opposed to [1], we ensure that the distance vanishes as the objects overlap. We show the efficacy of the approach in experimental results. Our proposed distance metric is publicly available through the Python-based simulation package UAIBot.
>
---
#### [new 025] Quantum-Assisted Automatic Path-Planning for Robotic Quality Inspection in Industry 4.0
- **分类: cs.RO; cs.AI; cs.ET**

- **简介: 该论文研究工业4.0中机器人质量检测的路径规划问题，采用量子-经典混合算法优化基于CAD模型的轨迹，解决3D旅行商问题，提升计算效率。**

- **链接: [http://arxiv.org/pdf/2507.01462v1](http://arxiv.org/pdf/2507.01462v1)**

> **作者:** Eneko Osaba; Estibaliz Garrote; Pablo Miranda-Rodriguez; Alessia Ciacco; Itziar Cabanes; Aitziber Mancisidor
>
> **备注:** 2 pages, 1 figure, paper accepted for presentation at the IEEE International Conference on Quantum Computing and Engineering (QCE)
>
> **摘要:** This work explores the application of hybrid quantum-classical algorithms to optimize robotic inspection trajectories derived from Computer-Aided Design (CAD) models in industrial settings. By modeling the task as a 3D variant of the Traveling Salesman Problem, incorporating incomplete graphs and open-route constraints, this study evaluates the performance of two D-Wave-based solvers against classical methods such as GUROBI and Google OR-Tools. Results across five real-world cases demonstrate competitive solution quality with significantly reduced computation times, highlighting the potential of quantum approaches in automation under Industry 4.0.
>
---
#### [new 026] Self-Closing Suction Grippers for Industrial Grasping via Form-Flexible Design
- **分类: cs.RO**

- **简介: 该论文属于工业抓取任务，旨在解决传统夹爪适应性差的问题。通过设计一种自闭式吸盘夹爪，实现对不同尺寸物体的稳定抓取。**

- **链接: [http://arxiv.org/pdf/2507.01561v1](http://arxiv.org/pdf/2507.01561v1)**

> **作者:** Huijiang Wang; Holger Kunz; Timon Adler; Fumiya Iida
>
> **备注:** This manuscript has been submitted for potential consideration at IEEE publication venues
>
> **摘要:** Shape-morphing robots have shown benefits in industrial grasping. We propose form-flexible grippers for adaptive grasping. The design is based on the hybrid jamming and suction mechanism, which deforms to handle objects that vary significantly in size from the aperture, including both larger and smaller parts. Compared with traditional grippers, the gripper achieves self-closing to form an airtight seal. Under a vacuum, a wide range of grasping is realized through the passive morphing mechanism at the interface that harmonizes pressure and flow rate. This hybrid gripper showcases the capability to securely grasp an egg, as small as 54.5% of its aperture, while achieving a maximum load-to-mass ratio of 94.3.
>
---
#### [new 027] BioMARS: A Multi-Agent Robotic System for Autonomous Biological Experiments
- **分类: cs.RO; cs.AI; cs.MA; q-bio.QM**

- **简介: 该论文属于实验室自动化任务，旨在解决生物实验中协议僵化、适应性差等问题。提出BioMARS系统，融合AI与机器人技术，实现自主实验设计与执行。**

- **链接: [http://arxiv.org/pdf/2507.01485v1](http://arxiv.org/pdf/2507.01485v1)**

> **作者:** Yibo Qiu; Zan Huang; Zhiyu Wang; Handi Liu; Yiling Qiao; Yifeng Hu; Shu'ang Sun; Hangke Peng; Ronald X Xu; Mingzhai Sun
>
> **摘要:** Large language models (LLMs) and vision-language models (VLMs) have the potential to transform biological research by enabling autonomous experimentation. Yet, their application remains constrained by rigid protocol design, limited adaptability to dynamic lab conditions, inadequate error handling, and high operational complexity. Here we introduce BioMARS (Biological Multi-Agent Robotic System), an intelligent platform that integrates LLMs, VLMs, and modular robotics to autonomously design, plan, and execute biological experiments. BioMARS uses a hierarchical architecture: the Biologist Agent synthesizes protocols via retrieval-augmented generation; the Technician Agent translates them into executable robotic pseudo-code; and the Inspector Agent ensures procedural integrity through multimodal perception and anomaly detection. The system autonomously conducts cell passaging and culture tasks, matching or exceeding manual performance in viability, consistency, and morphological integrity. It also supports context-aware optimization, outperforming conventional strategies in differentiating retinal pigment epithelial cells. A web interface enables real-time human-AI collaboration, while a modular backend allows scalable integration with laboratory hardware. These results highlight the feasibility of generalizable, AI-driven laboratory automation and the transformative role of language-based reasoning in biological research.
>
---
#### [new 028] Large Language Model-Driven Closed-Loop UAV Operation with Semantic Observations
- **分类: cs.RO**

- **简介: 该论文属于无人机控制任务，旨在解决LLM在复杂决策中的可靠性问题。通过构建闭环框架，利用LLM生成和评估代码，提升操作可靠性。**

- **链接: [http://arxiv.org/pdf/2507.01930v1](http://arxiv.org/pdf/2507.01930v1)**

> **作者:** Wenhao Wang; Yanyan Li; Long Jiao; Jiawei Yuan
>
> **备注:** 10 pages
>
> **摘要:** Large Language Models (LLMs) have revolutionized robotic autonomy, including Unmanned Aerial Vehicles (UAVs). Recent studies have demonstrated the potential of LLMs for translating human instructions into executable control code for UAV operations. However, LLMs still face challenges from logical reasoning and complex decision-making, leading to concerns about the reliability of LLM-driven UAV operations. In this paper, we propose a LLM-driven closed-loop control framework that enables reliable UAV operations powered by effective feedback and refinement using two LLM modules, i.e., a Code Generator and an Evaluator. Our framework transforms numerical state observations from UAV operations into natural language trajectory descriptions to enhance the evaluator LLM's understanding of UAV dynamics for precise feedback generation. Our framework also enables a simulation-based refinement process, and hence eliminates the risks to physical UAVs caused by incorrect code execution during the refinement. Extensive experiments on UAV control tasks with different complexities are conducted. The experimental results show that our framework can achieve reliable UAV operations using LLMs, which significantly outperforms baseline approaches in terms of success rate and completeness with the increase of task complexity.
>
---
#### [new 029] What does really matter in image goal navigation?
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究图像目标导航任务，探讨是否可通过端到端强化学习解决。工作包括分析架构选择对相对位姿估计的影响，并验证其在真实场景中的迁移能力。**

- **链接: [http://arxiv.org/pdf/2507.01667v1](http://arxiv.org/pdf/2507.01667v1)**

> **作者:** Gianluca Monaci; Philippe Weinzaepfel; Christian Wolf
>
> **摘要:** Image goal navigation requires two different skills: firstly, core navigation skills, including the detection of free space and obstacles, and taking decisions based on an internal representation; and secondly, computing directional information by comparing visual observations to the goal image. Current state-of-the-art methods either rely on dedicated image-matching, or pre-training of computer vision modules on relative pose estimation. In this paper, we study whether this task can be efficiently solved with end-to-end training of full agents with RL, as has been claimed by recent work. A positive answer would have impact beyond Embodied AI and allow training of relative pose estimation from reward for navigation alone. In a large study we investigate the effect of architectural choices like late fusion, channel stacking, space-to-depth projections and cross-attention, and their role in the emergence of relative pose estimators from navigation training. We show that the success of recent methods is influenced up to a certain extent by simulator settings, leading to shortcuts in simulation. However, we also show that these capabilities can be transferred to more realistic setting, up to some extend. We also find evidence for correlations between navigation performance and probed (emerging) relative pose estimation performance, an important sub skill.
>
---
#### [new 030] RALLY: Role-Adaptive LLM-Driven Yoked Navigation for Agentic UAV Swarms
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文属于多无人机协同导航任务，解决传统方法在语义沟通和角色适应上的不足。提出RALLY算法，结合LLM与MARL，提升任务覆盖与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.01378v1](http://arxiv.org/pdf/2507.01378v1)**

> **作者:** Ziyao Wang; Rongpeng Li; Sizhao Li; Yuming Xiang; Haiping Wang; Zhifeng Zhao; Honggang Zhang
>
> **摘要:** Intelligent control of Unmanned Aerial Vehicles (UAVs) swarms has emerged as a critical research focus, and it typically requires the swarm to navigate effectively while avoiding obstacles and achieving continuous coverage over multiple mission targets. Although traditional Multi-Agent Reinforcement Learning (MARL) approaches offer dynamic adaptability, they are hindered by the semantic gap in numerical communication and the rigidity of homogeneous role structures, resulting in poor generalization and limited task scalability. Recent advances in Large Language Model (LLM)-based control frameworks demonstrate strong semantic reasoning capabilities by leveraging extensive prior knowledge. However, due to the lack of online learning and over-reliance on static priors, these works often struggle with effective exploration, leading to reduced individual potential and overall system performance. To address these limitations, we propose a Role-Adaptive LLM-Driven Yoked navigation algorithm RALLY. Specifically, we first develop an LLM-driven semantic decision framework that uses structured natural language for efficient semantic communication and collaborative reasoning. Afterward, we introduce a dynamic role-heterogeneity mechanism for adaptive role switching and personalized decision-making. Furthermore, we propose a Role-value Mixing Network (RMIX)-based assignment strategy that integrates LLM offline priors with MARL online policies to enable semi-offline training of role selection strategies. Experiments in the Multi-Agent Particle Environment (MPE) environment and a Software-In-The-Loop (SITL) platform demonstrate that RALLY outperforms conventional approaches in terms of task coverage, convergence speed, and generalization, highlighting its strong potential for collaborative navigation in agentic multi-UAV systems.
>
---
#### [new 031] Imitation Learning for Satellite Attitude Control under Unknown Perturbations
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于卫星姿态控制任务，旨在解决未知扰动下的控制问题。通过结合SAC和GAIL，提升控制系统的鲁棒性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.01161v1](http://arxiv.org/pdf/2507.01161v1)**

> **作者:** Zhizhuo Zhang; Hao Peng; Xiaoli Bai
>
> **备注:** 2025 AAS/AIAA Astrodynamics Specialist Conference
>
> **摘要:** This paper presents a novel satellite attitude control framework that integrates Soft Actor-Critic (SAC) reinforcement learning with Generative Adversarial Imitation Learning (GAIL) to achieve robust performance under various unknown perturbations. Traditional control techniques often rely on precise system models and are sensitive to parameter uncertainties and external perturbations. To overcome these limitations, we first develop a SAC-based expert controller that demonstrates improved resilience against actuator failures, sensor noise, and attitude misalignments, outperforming our previous results in several challenging scenarios. We then use GAIL to train a learner policy that imitates the expert's trajectories, thereby reducing training costs and improving generalization through expert demonstrations. Preliminary experiments under single and combined perturbations show that the SAC expert can rotate the antenna to a specified direction and keep the antenna orientation reliably stable in most of the listed perturbations. Additionally, the GAIL learner can imitate most of the features from the trajectories generated by the SAC expert. Comparative evaluations and ablation studies confirm the effectiveness of the SAC algorithm and reward shaping. The integration of GAIL further reduces sample complexity and demonstrates promising imitation capabilities, paving the way for more intelligent and autonomous spacecraft control systems.
>
---
#### [new 032] Optimal Dispersion Under Asynchrony
- **分类: cs.DC; cs.DS; cs.MA; cs.RO**

- **简介: 该论文研究移动代理在异步环境下的最优分散问题，提出一种在O(k)时间内完成且内存消耗为O(log(k+Δ))的算法，解决了时间复杂度的瓶颈。**

- **链接: [http://arxiv.org/pdf/2507.01298v1](http://arxiv.org/pdf/2507.01298v1)**

> **作者:** Debasish Pattanayak; Ajay D. Kshemkalyani; Manish Kumar; Anisur Rahaman Molla; Gokarna Sharma
>
> **备注:** 35 pages, 5 figures, 2 tables, and 6 pseudocodes
>
> **摘要:** We study the dispersion problem in anonymous port-labeled graphs: $k \leq n$ mobile agents, each with a unique ID and initially located arbitrarily on the nodes of an $n$-node graph with maximum degree $\Delta$, must autonomously relocate so that no node hosts more than one agent. Dispersion serves as a fundamental task in distributed computing of mobile agents, and its complexity stems from key challenges in local coordination under anonymity and limited memory. The goal is to minimize both the time to achieve dispersion and the memory required per agent. It is known that any algorithm requires $\Omega(k)$ time in the worst case, and $\Omega(\log k)$ bits of memory per agent. A recent result [SPAA'25] gives an optimal $O(k)$-time algorithm in the synchronous setting and an $O(k \log k)$-time algorithm in the asynchronous setting, both using $O(\log(k+\Delta))$ bits. In this paper, we close the complexity gap in the asynchronous setting by presenting the first dispersion algorithm that runs in optimal $O(k)$ time using $O(\log(k+\Delta))$ bits of memory per agent. Our solution is based on a novel technique we develop in this paper that constructs a port-one tree in anonymous graphs, which may be of independent interest.
>
---
#### [new 033] Geometry-aware 4D Video Generation for Robot Manipulation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于4D视频生成任务，旨在解决多视角下视频时空一致性和几何一致性问题。通过几何监督学习3D场景表示，实现从RGB-D观测中预测未来视频序列。**

- **链接: [http://arxiv.org/pdf/2507.01099v1](http://arxiv.org/pdf/2507.01099v1)**

> **作者:** Zeyi Liu; Shuang Li; Eric Cousineau; Siyuan Feng; Benjamin Burchfiel; Shuran Song
>
> **备注:** Project website: https://robot4dgen.github.io
>
> **摘要:** Understanding and predicting the dynamics of the physical world can enhance a robot's ability to plan and interact effectively in complex environments. While recent video generation models have shown strong potential in modeling dynamic scenes, generating videos that are both temporally coherent and geometrically consistent across camera views remains a significant challenge. To address this, we propose a 4D video generation model that enforces multi-view 3D consistency of videos by supervising the model with cross-view pointmap alignment during training. This geometric supervision enables the model to learn a shared 3D representation of the scene, allowing it to predict future video sequences from novel viewpoints based solely on the given RGB-D observations, without requiring camera poses as inputs. Compared to existing baselines, our method produces more visually stable and spatially aligned predictions across multiple simulated and real-world robotic datasets. We further show that the predicted 4D videos can be used to recover robot end-effector trajectories using an off-the-shelf 6DoF pose tracker, supporting robust robot manipulation and generalization to novel camera viewpoints.
>
---
#### [new 034] Automated Vehicles Should be Connected with Natural Language
- **分类: cs.MA; cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于智能交通任务，旨在解决多智能体协作驾驶中的通信效率与信息完整性问题。通过引入自然语言进行意图和推理交流，提升协作驾驶的主动协调能力。**

- **链接: [http://arxiv.org/pdf/2507.01059v1](http://arxiv.org/pdf/2507.01059v1)**

> **作者:** Xiangbo Gao; Keshu Wu; Hao Zhang; Kexin Tian; Yang Zhou; Zhengzhong Tu
>
> **摘要:** Multi-agent collaborative driving promises improvements in traffic safety and efficiency through collective perception and decision making. However, existing communication media -- including raw sensor data, neural network features, and perception results -- suffer limitations in bandwidth efficiency, information completeness, and agent interoperability. Moreover, traditional approaches have largely ignored decision-level fusion, neglecting critical dimensions of collaborative driving. In this paper we argue that addressing these challenges requires a transition from purely perception-oriented data exchanges to explicit intent and reasoning communication using natural language. Natural language balances semantic density and communication bandwidth, adapts flexibly to real-time conditions, and bridges heterogeneous agent platforms. By enabling the direct communication of intentions, rationales, and decisions, it transforms collaborative driving from reactive perception-data sharing into proactive coordination, advancing safety, efficiency, and transparency in intelligent transportation systems.
>
---
#### [new 035] TD-MPC-Opt: Distilling Model-Based Multi-Task Reinforcement Learning Agents
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于多任务强化学习领域，旨在解决大模型在资源受限环境中的部署问题。通过知识蒸馏将大模型压缩为小模型，并优化其性能与大小。**

- **链接: [http://arxiv.org/pdf/2507.01823v1](http://arxiv.org/pdf/2507.01823v1)**

> **作者:** Dmytro Kuzmenko; Nadiya Shvai
>
> **备注:** Preprint of a manuscript submitted for peer review
>
> **摘要:** We present a novel approach to knowledge transfer in model-based reinforcement learning, addressing the critical challenge of deploying large world models in resource-constrained environments. Our method efficiently distills a high-capacity multi-task agent (317M parameters) into a compact model (1M parameters) on the MT30 benchmark, significantly improving performance across diverse tasks. Our distilled model achieves a state-of-the-art normalized score of 28.45, surpassing the original 1M parameter model score of 18.93. This improvement demonstrates the ability of our distillation technique to capture and consolidate complex multi-task knowledge. We further optimize the distilled model through FP16 post-training quantization, reducing its size by $\sim$50\%. Our approach addresses practical deployment limitations and offers insights into knowledge representation in large world models, paving the way for more efficient and accessible multi-task reinforcement learning systems in robotics and other resource-constrained applications. Code available at https://github.com/dmytro-kuzmenko/td-mpc-opt.
>
---
#### [new 036] Cooperative Target Capture in 3D Engagements over Switched Dynamic Graphs
- **分类: eess.SY; cs.MA; cs.RO; cs.SY**

- **简介: 该论文属于协同制导任务，解决多拦截器在动态图下同步击中静止目标的问题，通过优化横向加速度实现时间一致性。**

- **链接: [http://arxiv.org/pdf/2507.01350v1](http://arxiv.org/pdf/2507.01350v1)**

> **作者:** Abhinav Sinha; Shashi Ranjan Kumar
>
> **摘要:** This paper presents a leaderless cooperative guidance strategy for simultaneous time-constrained interception of a stationary target when the interceptors exchange information over switched dynamic graphs. We specifically focus on scenarios when the interceptors lack radial acceleration capabilities, relying solely on their lateral acceleration components. This consideration aligns with their inherent kinematic turn constraints. The proposed strategy explicitly addresses the complexities of coupled 3D engagements, thereby mitigating performance degradation that typically arises when the pitch and yaw channels are decoupled into two separate, mutually orthogonal planar engagements. Moreover, our formulation incorporates modeling uncertainties associated with the time-to-go estimation into the derivation of cooperative guidance commands to ensure robustness against inaccuracies in dynamic engagement scenarios. To optimize control efficiency, we analytically derive the lateral acceleration components in the orthogonal pitch and yaw channels by solving an instantaneous optimization problem, subject to an affine constraint. We show that the proposed cooperative guidance commands guarantee consensus in time-to-go values within a predefined time, which can be prescribed as a design parameter, regardless of the interceptors' initial configurations. We provide simulations to attest to the efficacy of the proposed method.
>
---
#### [new 037] Time-Varying Coverage Control: A Distributed Tracker-Planner MPC Framework
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于多智能体协同控制任务，解决动态环境下的覆盖控制问题。通过分布式框架实现轨迹规划与跟踪，确保约束满足与安全避障。**

- **链接: [http://arxiv.org/pdf/2507.01567v1](http://arxiv.org/pdf/2507.01567v1)**

> **作者:** Patrick Benito Eberhard; Johannes Köhler; Oliver Hüsser; Melanie N. Zeilinger; Andrea Carron
>
> **摘要:** Time-varying coverage control addresses the challenge of coordinating multiple agents covering an environment where regions of interest change over time. This problem has broad applications, including the deployment of autonomous taxis and coordination in search and rescue operations. The achievement of effective coverage is complicated by the presence of time-varying density functions, nonlinear agent dynamics, and stringent system and safety constraints. In this paper, we present a distributed multi-agent control framework for time-varying coverage under nonlinear constrained dynamics. Our approach integrates a reference trajectory planner and a tracking model predictive control (MPC) scheme, which operate at different frequencies within a multi-rate framework. For periodic density functions, we demonstrate closed-loop convergence to an optimal configuration of trajectories and provide formal guarantees regarding constraint satisfaction, collision avoidance, and recursive feasibility. Additionally, we propose an efficient algorithm capable of handling nonperiodic density functions, making the approach suitable for practical applications. Finally, we validate our method through hardware experiments using a fleet of four miniature race cars.
>
---
#### [new 038] Learning to Segment for Vehicle Routing Problems
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于车辆路径规划任务，旨在解决迭代求解器计算效率低的问题。通过提出FSTA分解和L2Seg框架，识别并聚合稳定段落，提升求解速度。**

- **链接: [http://arxiv.org/pdf/2507.01037v1](http://arxiv.org/pdf/2507.01037v1)**

> **作者:** Wenbin Ouyang; Sirui Li; Yining Ma; Cathy Wu
>
> **摘要:** Iterative search heuristics are widely recognized as state-of-the-art for solving Vehicle Routing Problems (VRPs). In this work, we identify and exploit a critical observation: within these solvers, a large portion of the solution remains stable, i.e., unchanged across search iterations, causing redundant computations, especially for large-scale VRPs with long subtours. To address this, we pioneer the formal study of the First-Segment-Then-Aggregate (FSTA) decomposition technique to accelerate iterative solvers. Specifically, FSTA preserves stable solution segments during the search, aggregates nodes within each segment into fixed hypernodes, and focuses the search only on unstable portions. Yet, a key challenge lies in identifying which segments should be aggregated by FSTA. To this end, we then introduce Learning-to-Segment (L2Seg), a novel neural framework to intelligently differentiate potentially stable and unstable portions for FSTA decomposition. We present three L2Seg variants: non-autoregressive (globally comprehensive but locally indiscriminate), autoregressive (locally refined but globally deficient), and their synergy, with bespoke training and inference strategies. Empirical results on CVRP and VRPTW suggest that L2Seg accelerates state-of-the-art iterative solvers by up to 7x. Additionally, we provide in-depth analysis showing NAR and AR synergy achieves best performance by combining their complementary strengths. Notably, L2Seg is a flexible framework that is compatible with traditional, learning-based, and hybrid solvers, while supporting a broad class of VRPs.
>
---
## 更新

#### [replaced 001] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.01398v2](http://arxiv.org/pdf/2412.01398v2)**

> **作者:** Anna-Maria Halacheva; Yang Miao; Jan-Nico Zaech; Xi Wang; Luc Van Gool; Danda Pani Paudel
>
> **摘要:** 3D scene understanding is a long-standing challenge in computer vision and a key component in enabling mixed reality, wearable computing, and embodied AI. Providing a solution to these applications requires a multifaceted approach that covers scene-centric, object-centric, as well as interaction-centric capabilities. While there exist numerous datasets and algorithms approaching the former two problems, the task of understanding interactable and articulated objects is underrepresented and only partly covered in the research field. In this work, we address this shortcoming by introducing: (1) Articulate3D, an expertly curated 3D dataset featuring high-quality manual annotations on 280 indoor scenes. Articulate3D provides 8 types of annotations for articulated objects, covering parts and detailed motion information, all stored in a standardized scene representation format designed for scalable 3D content creation, exchange and seamless integration into simulation environments. (2) USDNet, a novel unified framework capable of simultaneously predicting part segmentation along with a full specification of motion attributes for articulated objects. We evaluate USDNet on Articulate3D as well as two existing datasets, demonstrating the advantage of our unified dense prediction approach. Furthermore, we highlight the value of Articulate3D through cross-dataset and cross-domain evaluations and showcase its applicability in downstream tasks such as scene editing through LLM prompting and robotic policy training for articulated object manipulation. We provide open access to our dataset, benchmark, and method's source code.
>
---
#### [replaced 002] 2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.09320v3](http://arxiv.org/pdf/2503.09320v3)**

> **作者:** Marvin Heidinger; Snehal Jauhri; Vignesh Prasad; Georgia Chalvatzaki
>
> **备注:** ICCV 2025
>
> **摘要:** When interacting with objects, humans effectively reason about which regions of objects are viable for an intended action, i.e., the affordance regions of the object. They can also account for subtle differences in object regions based on the task to be performed and whether one or two hands need to be used. However, current vision-based affordance prediction methods often reduce the problem to naive object part segmentation. In this work, we propose a framework for extracting affordance data from human activity video datasets. Our extracted 2HANDS dataset contains precise object affordance region segmentations and affordance class-labels as narrations of the activity performed. The data also accounts for bimanual actions, i.e., two hands co-ordinating and interacting with one or more objects. We present a VLM-based affordance prediction model, 2HandedAfforder, trained on the dataset and demonstrate superior performance over baselines in affordance region segmentation for various activities. Finally, we show that our predicted affordance regions are actionable, i.e., can be used by an agent performing a task, through demonstration in robotic manipulation scenarios. Project-website: https://sites.google.com/view/2handedafforder
>
---
#### [replaced 003] High-Precision Transformer-Based Visual Servoing for Humanoid Robots in Aligning Tiny Objects
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.04862v2](http://arxiv.org/pdf/2503.04862v2)**

> **作者:** Jialong Xue; Wei Gao; Yu Wang; Chao Ji; Dongdong Zhao; Shi Yan; Shiwu Zhang
>
> **备注:** for associated video, see https://b23.tv/cklF7aK
>
> **摘要:** High-precision tiny object alignment remains a common and critical challenge for humanoid robots in real-world. To address this problem, this paper proposes a vision-based framework for precisely estimating and controlling the relative position between a handheld tool and a target object for humanoid robots, e.g., a screwdriver tip and a screw head slot. By fusing images from the head and torso cameras on a robot with its head joint angles, the proposed Transformer-based visual servoing method can correct the handheld tool's positional errors effectively, especially at a close distance. Experiments on M4-M8 screws demonstrate an average convergence error of 0.8-1.3 mm and a success rate of 93\%-100\%. Through comparative analysis, the results validate that this capability of high-precision tiny object alignment is enabled by the Distance Estimation Transformer architecture and the Multi-Perception-Head mechanism proposed in this paper.
>
---
#### [replaced 004] Real-is-Sim: Bridging the Sim-to-Real Gap with a Dynamic Digital Twin
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.03597v2](http://arxiv.org/pdf/2504.03597v2)**

> **作者:** Jad Abou-Chakra; Lingfeng Sun; Krishan Rana; Brandon May; Karl Schmeckpeper; Niko Suenderhauf; Maria Vittoria Minniti; Laura Herlant
>
> **摘要:** We introduce real-is-sim, a new approach to integrating simulation into behavior cloning pipelines. In contrast to real-only methods, which lack the ability to safely test policies before deployment, and sim-to-real methods, which require complex adaptation to cross the sim-to-real gap, our framework allows policies to seamlessly switch between running on real hardware and running in parallelized virtual environments. At the center of real-is-sim is a dynamic digital twin, powered by the Embodied Gaussian simulator, that synchronizes with the real world at 60Hz. This twin acts as a mediator between the behavior cloning policy and the real robot. Policies are trained using representations derived from simulator states and always act on the simulated robot, never the real one. During deployment, the real robot simply follows the simulated robot's joint states, and the simulation is continuously corrected with real world measurements. This setup, where the simulator drives all policy execution and maintains real-time synchronization with the physical world, shifts the responsibility of crossing the sim-to-real gap to the digital twin's synchronization mechanisms, instead of the policy itself. We demonstrate real-is-sim on a long-horizon manipulation task (PushT), showing that virtual evaluations are consistent with real-world results. We further show how real-world data can be augmented with virtual rollouts and compare to policies trained on different representations derived from the simulator state including object poses and rendered images from both static and robot-mounted cameras. Our results highlight the flexibility of the real-is-sim framework across training, evaluation, and deployment stages. Videos available at https://real-is-sim.github.io.
>
---
#### [replaced 005] SKIL: Semantic Keypoint Imitation Learning for Generalizable Data-efficient Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.14400v2](http://arxiv.org/pdf/2501.14400v2)**

> **作者:** Shengjie Wang; Jiacheng You; Yihang Hu; Jiongye Li; Yang Gao
>
> **备注:** 22 pages, 22 figures
>
> **摘要:** Real-world tasks such as garment manipulation and table rearrangement demand robots to perform generalizable, highly precise, and long-horizon actions. Although imitation learning has proven to be an effective approach for teaching robots new skills, large amounts of expert demonstration data are still indispensible for these complex tasks, resulting in high sample complexity and costly data collection. To address this, we propose Semantic Keypoint Imitation Learning (SKIL), a framework which automatically obtains semantic keypoints with the help of vision foundation models, and forms the descriptor of semantic keypoints that enables efficient imitation learning of complex robotic tasks with significantly lower sample complexity. In real-world experiments, SKIL doubles the performance of baseline methods in tasks such as picking a cup or mouse, while demonstrating exceptional robustness to variations in objects, environmental changes, and distractors. For long-horizon tasks like hanging a towel on a rack where previous methods fail completely, SKIL achieves a mean success rate of 70\% with as few as 30 demonstrations. Furthermore, SKIL naturally supports cross-embodiment learning due to its semantic keypoints abstraction. Our experiments demonstrate that even human videos bring considerable improvement to the learning performance. All these results demonstrate the great success of SKIL in achieving data-efficient generalizable robotic learning. Visualizations and code are available at: https://skil-robotics.github.io/SKIL-robotics/.
>
---
#### [replaced 006] Co-design of magnetic soft robots with large deformation and contacts via material point method and topology optimization
- **分类: cs.RO; cs.CE**

- **链接: [http://arxiv.org/pdf/2503.22767v2](http://arxiv.org/pdf/2503.22767v2)**

> **作者:** Liwei Wang
>
> **摘要:** Magnetic soft robots embedded with hard magnetic particles enable untethered actuation via external magnetic fields, offering remote, rapid, and precise control, which is highly promising for biomedical applications. However, designing such systems is challenging due to the complex interplay of magneto-elastic dynamics, large deformation, solid contacts, time-varying stimuli, and posture-dependent loading. As a result, most existing research relies on heuristics and trial-and-error methods or focuses on the independent design of stimuli or structures under static conditions. We propose a topology optimization framework for magnetic soft robots that simultaneously designs structures, location-specific material magnetization and time-varying magnetic stimuli, accounting for large deformations, dynamic motion, and solid contacts. This is achieved by integrating generalized topology optimization with the magneto-elastic material point method, which supports GPU-accelerated parallel simulations and auto-differentiation for sensitivity analysis. We applied this framework to design magnetic robots for various tasks, including multi-task shape morphing and locomotion, in both 2D and 3D. The method autonomously generates optimized robotic systems to achieve target behaviors without requiring human intervention. Despite the nonlinear physics and large design space, it demonstrates high computational efficiency, completing all cases within minutes. The framework provides a computational foundation for the autonomous co-design of active soft materials in applications such as metasurfaces, drug delivery, and minimally invasive procedures.
>
---
#### [replaced 007] Embodied Instruction Following in Unknown Environments
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.11818v2](http://arxiv.org/pdf/2406.11818v2)**

> **作者:** Zhenyu Wu; Ziwei Wang; Xiuwei Xu; Hang Yin; Yinan Liang; Angyuan Ma; Jiwen Lu; Haibin Yan
>
> **备注:** Project Page: https://gary3410.github.io/eif_unknown/
>
> **摘要:** Enabling embodied agents to complete complex human instructions from natural language is crucial to autonomous systems in household services. Conventional methods can only accomplish human instructions in the known environment where all interactive objects are provided to the embodied agent, and directly deploying the existing approaches for the unknown environment usually generates infeasible plans that manipulate non-existing objects. On the contrary, we propose an embodied instruction following (EIF) method for complex tasks in the unknown environment, where the agent efficiently explores the unknown environment to generate feasible plans with existing objects to accomplish abstract instructions. Specifically, we build a hierarchical embodied instruction following framework including the high-level task planner and the low-level exploration controller with multimodal large language models. We then construct a semantic representation map of the scene with dynamic region attention to demonstrate the known visual clues, where the goal of task planning and scene exploration is aligned for human instruction. For the task planner, we generate the feasible step-by-step plans for human goal accomplishment according to the task completion process and the known visual clues. For the exploration controller, the optimal navigation or object interaction policy is predicted based on the generated step-wise plans and the known visual clues. The experimental results demonstrate that our method can achieve 45.09% success rate in 204 complex human instructions such as making breakfast and tidying rooms in large house-level scenes. Code and supplementary are available at https://gary3410.github.io/eif_unknown.
>
---
#### [replaced 008] Co-Optimizing Reconfigurable Environments and Policies for Decentralized Multi-Agent Navigation
- **分类: cs.RO; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2403.14583v2](http://arxiv.org/pdf/2403.14583v2)**

> **作者:** Zhan Gao; Guang Yang; Amanda Prorok
>
> **摘要:** This work views the multi-agent system and its surrounding environment as a co-evolving system, where the behavior of one affects the other. The goal is to take both agent actions and environment configurations as decision variables, and optimize these two components in a coordinated manner to improve some measure of interest. Towards this end, we consider the problem of decentralized multi-agent navigation in a cluttered environment, where we assume that the layout of the environment is reconfigurable. By introducing two sub-objectives -- multi-agent navigation and environment optimization -- we propose an agent-environment co-optimization problem and develop a coordinated algorithm that alternates between these sub-objectives to search for an optimal synthesis of agent actions and environment configurations; ultimately, improving the navigation performance. Due to the challenge of explicitly modeling the relation between the agents, the environment and their performance therein, we leverage policy gradient to formulate a model-free learning mechanism within the coordinated framework. A formal convergence analysis shows that our coordinated algorithm tracks the local minimum solution of an associated time-varying non-convex optimization problem. Experiments corroborate theoretical findings and show the benefits of co-optimization. Interestingly, the results also indicate that optimized environments can offer structural guidance to de-conflict agents in motion.
>
---
#### [replaced 009] NeRFs in Robotics: A Survey
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.01333v2](http://arxiv.org/pdf/2405.01333v2)**

> **作者:** Guangming Wang; Lei Pan; Songyou Peng; Shaohui Liu; Chenfeng Xu; Yanzi Miao; Wei Zhan; Masayoshi Tomizuka; Marc Pollefeys; Hesheng Wang
>
> **备注:** 31 pages, 19 figures, accepted by The International Journal of Robotics Research, 2025
>
> **摘要:** Detailed and realistic 3D environment representations have been a long-standing goal in the fields of computer vision and robotics. The recent emergence of neural implicit representations has introduced significant advances to these domains, enabling numerous novel capabilities. Among these, Neural Radiance Fields (NeRFs) have gained considerable attention because of their considerable representational advantages, such as simplified mathematical models, low memory footprint, and continuous scene representations. In addition to computer vision, NeRFs have demonstrated significant potential in robotics. Thus, we present this survey to provide a comprehensive understanding of NeRFs in the field of robotics. By exploring the advantages and limitations of NeRF as well as its current applications and future potential, we aim to provide an overview of this promising area of research. Our survey is divided into two main sections: \textit{Applications of NeRFs in Robotics} and \textit{Advances for NeRFs in Robotics}, from the perspective of how NeRF enters the field of robotics. In the first section, we introduce and analyze some works that have been or could be used in robotics for perception and interaction tasks. In the second section, we show some works related to improving NeRF's own properties, which are essential for deploying NeRFs in robotics. In the discussion section of the review, we summarize the existing challenges and provide valuable future research directions.
>
---
#### [replaced 010] SCALER: Versatile Multi-Limbed Robot for Free-Climbing in Extreme Terrains
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2312.04856v3](http://arxiv.org/pdf/2312.04856v3)**

> **作者:** Yusuke Tanaka; Yuki Shirai; Alexander Schperberg; Xuan Lin; Dennis Hong
>
> **备注:** Accepted to IEEE Transactions on Robotics (T-RO), 2025
>
> **摘要:** This paper presents SCALER, a versatile free-climbing multi-limbed robot that is designed to achieve tightly coupled simultaneous locomotion and dexterous grasping. While existing quadrupedal-limbed robots have demonstrated impressive dexterous capabilities, achieving a balance between power-demanding locomotion and precise grasping remains a critical challenge. We design a torso mechanism and a parallel-serial limb to meet the conflicting requirements that pose unique challenges in hardware design. SCALER employs underactuated two-fingered GOAT grippers that can mechanically adapt and offer seven modes of grasping, enabling SCALER to traverse extreme terrains with multi-modal grasping strategies. We study the whole-body approach, where SCALER utilizes its body and limbs to generate additional forces for stable grasping in various environments, thereby further enhancing its versatility. Furthermore, we improve the GOAT gripper actuation speed to realize more dynamic climbing in a closed-loop control fashion. With these proposed technologies, SCALER can traverse vertical, overhanging, upside-down, slippery terrains and bouldering walls with non-convex-shaped climbing holds under the Earth's gravity.
>
---
#### [replaced 011] Legged Robot State Estimation Using Invariant Neural-Augmented Kalman Filter with a Neural Compensator
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.00344v2](http://arxiv.org/pdf/2503.00344v2)**

> **作者:** Seokju Lee; Hyun-Bin Kim; Kyung-Soo Kim
>
> **备注:** 8 pages, 10 figures, Accepted to IROS 2025
>
> **摘要:** This paper presents an algorithm to improve state estimation for legged robots. Among existing model-based state estimation methods for legged robots, the contact-aided invariant extended Kalman filter defines the state on a Lie group to preserve invariance, thereby significantly accelerating convergence. It achieves more accurate state estimation by leveraging contact information as measurements for the update step. However, when the model exhibits strong nonlinearity, the estimation accuracy decreases. Such nonlinearities can cause initial errors to accumulate and lead to large drifts over time. To address this issue, we propose compensating for errors by augmenting the Kalman filter with an artificial neural network serving as a nonlinear function approximator. Furthermore, we design this neural network to respect the Lie group structure to ensure invariance, resulting in our proposed Invariant Neural-Augmented Kalman Filter (InNKF). The proposed algorithm offers improved state estimation performance by combining the strengths of model-based and learning-based approaches. Project webpage: https://seokju-lee.github.io/innkf_webpage
>
---
#### [replaced 012] EP-Diffuser: An Efficient Diffusion Model for Traffic Scene Generation and Prediction via Polynomial Representations
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.05422v2](http://arxiv.org/pdf/2504.05422v2)**

> **作者:** Yue Yao; Mohamed-Khalil Bouzidi; Daniel Goehring; Joerg Reichardt
>
> **摘要:** As the prediction horizon increases, predicting the future evolution of traffic scenes becomes increasingly difficult due to the multi-modal nature of agent motion. Most state-of-the-art (SotA) prediction models primarily focus on forecasting the most likely future. However, for the safe operation of autonomous vehicles, it is equally important to cover the distribution for plausible motion alternatives. To address this, we introduce EP-Diffuser, a novel parameter-efficient diffusion-based generative model designed to capture the distribution of possible traffic scene evolutions. Conditioned on road layout and agent history, our model acts as a predictor and generates diverse, plausible scene continuations. We benchmark EP-Diffuser against two SotA models in terms of accuracy and plausibility of predictions on the Argoverse 2 dataset. Despite its significantly smaller model size, our approach achieves both highly accurate and plausible traffic scene predictions. We further evaluate model generalization ability in an out-of-distribution (OoD) test setting using Waymo Open dataset and show superior robustness of our approach.
>
---
#### [replaced 013] Tightly-Coupled LiDAR-IMU-Leg Odometry with Online Learned Leg Kinematics Incorporating Foot Tactile Information
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.09548v2](http://arxiv.org/pdf/2506.09548v2)**

> **作者:** Taku Okawara; Kenji Koide; Aoki Takanose; Shuji Oishi; Masashi Yokozuka; Kentaro Uno; Kazuya Yoshida
>
> **备注:** Robotics and Automation Letters, 2025
>
> **摘要:** In this letter, we present tightly coupled LiDAR-IMU-leg odometry, which is robust to challenging conditions such as featureless environments and deformable terrains. We developed an online learning-based leg kinematics model named the neural leg kinematics model, which incorporates tactile information (foot reaction force) to implicitly express the nonlinear dynamics between robot feet and the ground. Online training of this model enhances its adaptability to weight load changes of a robot (e.g., assuming delivery or transportation tasks) and terrain conditions. According to the \textit{neural adaptive leg odometry factor} and online uncertainty estimation of the leg kinematics model-based motion predictions, we jointly solve online training of this kinematics model and odometry estimation on a unified factor graph to retain the consistency of both. The proposed method was verified through real experiments using a quadruped robot in two challenging situations: 1) a sandy beach, representing an extremely featureless area with a deformable terrain, and 2) a campus, including multiple featureless areas and terrain types of asphalt, gravel (deformable terrain), and grass. Experimental results showed that our odometry estimation incorporating the \textit{neural leg kinematics model} outperforms state-of-the-art works. Our project page is available for further details: https://takuokawara.github.io/RAL2025_project_page/
>
---
#### [replaced 014] Anyview: Generalizable Indoor 3D Object Detection with Variable Frames
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2310.05346v2](http://arxiv.org/pdf/2310.05346v2)**

> **作者:** Zhenyu Wu; Xiuwei Xu; Ziwei Wang; Chong Xia; Linqing Zhao; Jiwen Lu; Haibin Yan
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** In this paper, we propose a novel network framework for indoor 3D object detection to handle variable input frame numbers in practical scenarios. Existing methods only consider fixed frames of input data for a single detector, such as monocular RGB-D images or point clouds reconstructed from dense multi-view RGB-D images. While in practical application scenes such as robot navigation and manipulation, the raw input to the 3D detectors is the RGB-D images with variable frame numbers instead of the reconstructed scene point cloud. However, the previous approaches can only handle fixed frame input data and have poor performance with variable frame input. In order to facilitate 3D object detection methods suitable for practical tasks, we present a novel 3D detection framework named AnyView for our practical applications, which generalizes well across different numbers of input frames with a single model. To be specific, we propose a geometric learner to mine the local geometric features of each input RGB-D image frame and implement local-global feature interaction through a designed spatial mixture module. Meanwhile, we further utilize a dynamic token strategy to adaptively adjust the number of extracted features for each frame, which ensures consistent global feature density and further enhances the generalization after fusion. Extensive experiments on the ScanNet dataset show our method achieves both great generalizability and high detection accuracy with a simple and clean architecture containing a similar amount of parameters with the baselines.
>
---
#### [replaced 015] CU-Multi: A Dataset for Multi-Robot Data Association
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.17576v2](http://arxiv.org/pdf/2505.17576v2)**

> **作者:** Doncey Albin; Miles Mena; Annika Thomas; Harel Biggie; Xuefei Sun; Dusty Woods; Steve McGuire; Christoffer Heckman
>
> **备注:** 8 pages, 6 figures, 4 tables
>
> **摘要:** Multi-robot systems (MRSs) are valuable for tasks such as search and rescue due to their ability to coordinate over shared observations. A central challenge in these systems is aligning independently collected perception data across space and time, i.e., multi-robot data association. While recent advances in collaborative SLAM (C-SLAM), map merging, and inter-robot loop closure detection have significantly progressed the field, evaluation strategies still predominantly rely on splitting a single trajectory from single-robot SLAM datasets into multiple segments to simulate multiple robots. Without careful consideration to how a single trajectory is split, this approach will fail to capture realistic pose-dependent variation in observations of a scene inherent to multi-robot systems. To address this gap, we present CU-Multi, a multi-robot dataset collected over multiple days at two locations on the University of Colorado Boulder campus. Using a single robotic platform, we generate four synchronized runs with aligned start times and deliberate percentages of trajectory overlap. CU-Multi includes RGB-D, GPS with accurate geospatial heading, and semantically annotated LiDAR data. By introducing controlled variations in trajectory overlap and dense lidar annotations, CU-Multi offers a compelling alternative for evaluating methods in multi-robot data association. Instructions on accessing the dataset, support code, and the latest updates are publicly available at https://arpg.github.io/cumulti
>
---
#### [replaced 016] Time-Series JEPA for Predictive Remote Control under Capacity-Limited Networks
- **分类: cs.IT; cs.LG; cs.RO; math.IT**

- **链接: [http://arxiv.org/pdf/2406.04853v2](http://arxiv.org/pdf/2406.04853v2)**

> **作者:** Abanoub M. Girgis; Alvaro Valcarce; Mehdi Bennis
>
> **摘要:** In remote control systems, transmitting large data volumes (e.g., images, video frames) from wireless sensors to remote controllers is challenging when uplink capacity is limited (e.g., RedCap devices or massive wireless sensor networks). Furthermore, controllers often need only information-rich representations of the original data. To address this, we propose a semantic-driven predictive control combined with a channel-aware scheduling to enhance control performance for multiple devices under limited network capacity. At its core, the proposed framework, coined Time-Series Joint Embedding Predictive Architecture (TS-JEPA), encodes high-dimensional sensory data into low-dimensional semantic embeddings at the sensor, reducing communication overhead. Furthermore, TS-JEPA enables predictive inference by predicting future embeddings from current ones and predicted commands, which are directly used by a semantic actor model to compute control commands within the embedding space, eliminating the need to reconstruct raw data. To further enhance reliability and communication efficiency, a channel-aware scheduling is integrated to dynamically prioritize device transmissions based on channel conditions and age of information (AoI). Simulations on inverted cart-pole systems show that the proposed framework significantly outperforms conventional control baselines in communication efficiency, control cost, and predictive accuracy. It enables robust and scalable control under limited network capacity compared to traditional scheduling schemes.
>
---
#### [replaced 017] DexH2R: A Benchmark for Dynamic Dexterous Grasping in Human-to-Robot Handover
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.23152v3](http://arxiv.org/pdf/2506.23152v3)**

> **作者:** Youzhuo Wang; Jiayi Ye; Chuyang Xiao; Yiming Zhong; Heng Tao; Hang Yu; Yumeng Liu; Jingyi Yu; Yuexin Ma
>
> **备注:** Comments: Accepted by ICCV 2025. Project page: https://dexh2r.github.io/
>
> **摘要:** Handover between a human and a dexterous robotic hand is a fundamental yet challenging task in human-robot collaboration. It requires handling dynamic environments and a wide variety of objects and demands robust and adaptive grasping strategies. However, progress in developing effective dynamic dexterous grasping methods is limited by the absence of high-quality, real-world human-to-robot handover datasets. Existing datasets primarily focus on grasping static objects or rely on synthesized handover motions, which differ significantly from real-world robot motion patterns, creating a substantial gap in applicability. In this paper, we introduce DexH2R, a comprehensive real-world dataset for human-to-robot handovers, built on a dexterous robotic hand. Our dataset captures a diverse range of interactive objects, dynamic motion patterns, rich visual sensor data, and detailed annotations. Additionally, to ensure natural and human-like dexterous motions, we utilize teleoperation for data collection, enabling the robot's movements to align with human behaviors and habits, which is a crucial characteristic for intelligent humanoid robots. Furthermore, we propose an effective solution, DynamicGrasp, for human-to-robot handover and evaluate various state-of-the-art approaches, including auto-regressive models and diffusion policy methods, providing a thorough comparison and analysis. We believe our benchmark will drive advancements in human-to-robot handover research by offering a high-quality dataset, effective solutions, and comprehensive evaluation metrics.
>
---
#### [replaced 018] World-aware Planning Narratives Enhance Large Vision-Language Model Planner
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.21230v2](http://arxiv.org/pdf/2506.21230v2)**

> **作者:** Junhao Shi; Zhaoye Fei; Siyin Wang; Qipeng Guo; Jingjing Gong; Xipeng Qiu
>
> **摘要:** Large Vision-Language Models (LVLMs) show promise for embodied planning tasks but struggle with complex scenarios involving unfamiliar environments and multi-step goals. Current approaches rely on environment-agnostic imitation learning that disconnects instructions from environmental contexts, causing models to struggle with context-sensitive instructions and rely on supplementary cues rather than visual reasoning during long-horizon interactions. In this work, we propose World-Aware Planning Narrative Enhancement (WAP), a framework that infuses LVLMs with comprehensive environmental understanding through four cognitive capabilities (visual appearance modeling, spatial reasoning, functional abstraction, and syntactic grounding) while developing and evaluating models using only raw visual observations through curriculum learning. Evaluations on the EB-ALFRED benchmark demonstrate substantial improvements, with Qwen2.5-VL achieving a 60.7 absolute improvement in task success rates, particularly in commonsense reasoning (+60.0) and long-horizon planning (+70.0). Notably, our enhanced open-source models outperform proprietary systems like GPT-4o and Claude-3.5-Sonnet by a large margin.
>
---
