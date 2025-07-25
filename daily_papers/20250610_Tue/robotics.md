# 机器人 cs.RO

- **最新发布 68 篇**

- **更新 33 篇**

## 最新发布

#### [new 001] A Communication-Latency-Aware Co-Simulation Platform for Safety and Comfort Evaluation of Cloud-Controlled ICVs
- **分类: cs.RO**

- **简介: 该论文属于智能网联汽车测试任务，旨在解决云控车辆在真实通信延迟下的安全与舒适性评估问题。通过构建协同仿真平台，结合实测延迟模型和冲突模块，验证了平台的有效性。**

- **链接: [http://arxiv.org/pdf/2506.07696v1](http://arxiv.org/pdf/2506.07696v1)**

> **作者:** Yongqi Zhao; Xinrui Zhang; Tomislav Mihalj; Martin Schabauer; Luis Putzer; Erik Reichmann-Blaga; Ádám Boronyák; András Rövid; Gábor Soós; Peizhi Zhang; Lu Xiong; Jia Hu; Arno Eichberger
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Testing cloud-controlled intelligent connected vehicles (ICVs) requires simulation environments that faithfully emulate both vehicle behavior and realistic communication latencies. This paper proposes a latency-aware co-simulation platform integrating CarMaker and Vissim to evaluate safety and comfort under real-world vehicle-to-cloud (V2C) latency conditions. Two communication latency models, derived from empirical 5G measurements in China and Hungary, are incorporated and statistically modeled using Gamma distributions. A proactive conflict module (PCM) is proposed to dynamically control background vehicles and generate safety-critical scenarios. The platform is validated through experiments involving an exemplary system under test (SUT) across six testing conditions combining two PCM modes (enabled/disabled) and three latency conditions (none, China, Hungary). Safety and comfort are assessed using metrics including collision rate, distance headway, post-encroachment time, and the spectral characteristics of longitudinal acceleration. Results show that the PCM effectively increases driving environment criticality, while V2C latency primarily affects ride comfort. These findings confirm the platform's effectiveness in systematically evaluating cloud-controlled ICVs under diverse testing conditions.
>
---
#### [new 002] IRS: Instance-Level 3D Scene Graphs via Room Prior Guided LiDAR-Camera Fusion
- **分类: cs.RO**

- **简介: 该论文属于3D场景图构建任务，旨在解决室内场景理解中的实例级语义与几何融合问题。通过LiDAR与相机融合，结合视觉基础模型，提升场景图构建的效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.06804v1](http://arxiv.org/pdf/2506.06804v1)**

> **作者:** Hongming Chen; Yiyang Lin; Ziliang Li; Biyu Ye; Yuying Zhang; Ximin Lyu
>
> **摘要:** Indoor scene understanding remains a fundamental challenge in robotics, with direct implications for downstream tasks such as navigation and manipulation. Traditional approaches often rely on closed-set recognition or loop closure, limiting their adaptability in open-world environments. With the advent of visual foundation models (VFMs), open-vocabulary recognition and natural language querying have become feasible, unlocking new possibilities for 3D scene graph construction. In this paper, we propose a robust and efficient framework for instance-level 3D scene graph construction via LiDAR-camera fusion. Leveraging LiDAR's wide field of view (FOV) and long-range sensing capabilities, we rapidly acquire room-level geometric priors. Multi-level VFMs are employed to improve the accuracy and consistency of semantic extraction. During instance fusion, room-based segmentation enables parallel processing, while the integration of geometric and semantic cues significantly enhances fusion accuracy and robustness. Compared to state-of-the-art methods, our approach achieves up to an order-of-magnitude improvement in construction speed while maintaining high semantic precision. Extensive experiments in both simulated and real-world environments validate the effectiveness of our approach. We further demonstrate its practical value through a language-guided semantic navigation task, highlighting its potential for real-world robotic applications.
>
---
#### [new 003] BeliefMapNav: 3D Voxel-Based Belief Map for Zero-Shot Object Navigation
- **分类: cs.RO**

- **简介: 该论文属于零样本目标导航任务，旨在解决机器人在陌生环境中根据自然语言指令定位目标的问题。工作包括提出3D体素信念图，融合语义与空间信息，提升导航效率和准确性。**

- **链接: [http://arxiv.org/pdf/2506.06487v1](http://arxiv.org/pdf/2506.06487v1)**

> **作者:** Zibo Zhou; Yue Hu; Lingkai Zhang; Zonglin Li; Siheng Chen
>
> **摘要:** Zero-shot object navigation (ZSON) allows robots to find target objects in unfamiliar environments using natural language instructions, without relying on pre-built maps or task-specific training. Recent general-purpose models, such as large language models (LLMs) and vision-language models (VLMs), equip agents with semantic reasoning abilities to estimate target object locations in a zero-shot manner. However, these models often greedily select the next goal without maintaining a global understanding of the environment and are fundamentally limited in the spatial reasoning necessary for effective navigation. To overcome these limitations, we propose a novel 3D voxel-based belief map that estimates the target's prior presence distribution within a voxelized 3D space. This approach enables agents to integrate semantic priors from LLMs and visual embeddings with hierarchical spatial structure, alongside real-time observations, to build a comprehensive 3D global posterior belief of the target's location. Building on this 3D voxel map, we introduce BeliefMapNav, an efficient navigation system with two key advantages: i) grounding LLM semantic reasoning within the 3D hierarchical semantics voxel space for precise target position estimation, and ii) integrating sequential path planning to enable efficient global navigation decisions. Experiments on HM3D, MP3D, and HSSD benchmarks show that BeliefMapNav achieves state-of-the-art (SOTA) Success Rate (SR) and Success weighted by Path Length (SPL), with a notable 46.4% SPL improvement over the previous best SR method, validating its effectiveness and efficiency.
>
---
#### [new 004] Self-Adapting Improvement Loops for Robotic Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决机器人在未见过任务上的泛化问题。通过提出SAIL框架，利用自收集数据持续优化视频模型，提升任务执行性能。**

- **链接: [http://arxiv.org/pdf/2506.06658v1](http://arxiv.org/pdf/2506.06658v1)**

> **作者:** Calvin Luo; Zilai Zeng; Mingxi Jia; Yilun Du; Chen Sun
>
> **摘要:** Video generative models trained on expert demonstrations have been utilized as performant text-conditioned visual planners for solving robotic tasks. However, generalization to unseen tasks remains a challenge. Whereas improved generalization may be facilitated by leveraging learned prior knowledge from additional pre-collected offline data sources, such as web-scale video datasets, in the era of experience we aim to design agents that can continuously improve in an online manner from self-collected behaviors. In this work we thus propose the Self-Adapting Improvement Loop (SAIL), where an in-domain video model iteratively updates itself on self-produced trajectories, collected through adaptation with an internet-scale pretrained video model, and steadily improves its performance for a specified task of interest. We apply SAIL to a diverse suite of MetaWorld tasks, as well as two manipulation tasks on a real robot arm, and find that performance improvements continuously emerge over multiple iterations for novel tasks initially unseen during original in-domain video model training. Furthermore, we discover that SAIL is surprisingly robust regarding if and how the self-collected experience is filtered, and the quality of the initial in-domain demonstrations. Through adaptation with summarized internet-scale data, and learning through online experience, we thus demonstrate a way to iteratively bootstrap a high-performance video model for solving novel robotic tasks through self-improvement.
>
---
#### [new 005] Improving Traffic Signal Data Quality for the Waymo Open Motion Dataset
- **分类: cs.RO**

- **简介: 该论文属于数据质量提升任务，旨在解决Waymo Open Motion Dataset中交通信号状态缺失或错误的问题。通过自动化方法修复数据，显著提高了数据可靠性。**

- **链接: [http://arxiv.org/pdf/2506.07150v1](http://arxiv.org/pdf/2506.07150v1)**

> **作者:** Xintao Yan; Erdao Liang; Jiawei Wang; Haojie Zhu; Henry X. Liu
>
> **摘要:** Datasets pertaining to autonomous vehicles (AVs) hold significant promise for a range of research fields, including artificial intelligence (AI), autonomous driving, and transportation engineering. Nonetheless, these datasets often encounter challenges related to the states of traffic signals, such as missing or inaccurate data. Such issues can compromise the reliability of the datasets and adversely affect the performance of models developed using them. This research introduces a fully automated approach designed to tackle these issues by utilizing available vehicle trajectory data alongside knowledge from the transportation domain to effectively impute and rectify traffic signal information within the Waymo Open Motion Dataset (WOMD). The proposed method is robust and flexible, capable of handling diverse intersection geometries and traffic signal configurations in real-world scenarios. Comprehensive validations have been conducted on the entire WOMD, focusing on over 360,000 relevant scenarios involving traffic signals, out of a total of 530,000 real-world driving scenarios. In the original dataset, 71.7% of traffic signal states are either missing or unknown, all of which were successfully imputed by our proposed method. Furthermore, in the absence of ground-truth signal states, the accuracy of our approach is evaluated based on the rate of red-light violations among vehicle trajectories. Results show that our method reduces the estimated red-light running rate from 15.7% in the original data to 2.9%, thereby demonstrating its efficacy in rectifying data inaccuracies. This paper significantly enhances the quality of AV datasets, contributing to the wider AI and AV research communities and benefiting various downstream applications. The code and improved traffic signal data are open-sourced at https://github.com/michigan-traffic-lab/WOMD-Traffic-Signal-Data-Improvement
>
---
#### [new 006] Underwater Multi-Robot Simulation and Motion Planning in Angler
- **分类: cs.RO**

- **简介: 该论文属于水下多机器人系统任务，解决多机器人仿真与路径规划问题。提出扩展框架Angler，实现多机器人协同仿真与运动规划。**

- **链接: [http://arxiv.org/pdf/2506.06612v1](http://arxiv.org/pdf/2506.06612v1)**

> **作者:** Akshaya Agrawal; Evan Palmer; Zachary Kingston; Geoffrey A. Hollinger
>
> **备注:** Accepted for OCEANS 2025 Brest
>
> **摘要:** Deploying multi-robot systems in underwater environments is expensive and lengthy; testing algorithms and software in simulation improves development by decoupling software and hardware. However, this requires a simulation framework that closely resembles the real-world. Angler is an open-source framework that simulates low-level communication protocols for an onboard autopilot, such as ArduSub, providing a framework that is close to reality, but unfortunately lacking support for simulating multiple robots. We present an extension to Angler that supports multi-robot simulation and motion planning. Our extension has a modular architecture that creates non-conflicting communication channels between Gazebo, ArduSub Software-in-the-Loop (SITL), and MAVROS to operate multiple robots simultaneously in the same environment. Our multi-robot motion planning module interfaces with cascaded controllers via a JointTrajectory controller in ROS~2. We also provide an integration with the Open Motion Planning Library (OMPL), a collision avoidance module, and tools for procedural environment generation. Our work enables the development and benchmarking of underwater multi-robot motion planning in dynamic environments.
>
---
#### [new 007] MapBERT: Bitwise Masked Modeling for Real-Time Semantic Mapping Generation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MapBERT，用于实时语义地图生成任务，解决室内语义分布建模难题，通过位编码和掩码Transformer实现高效准确的地图重建。**

- **链接: [http://arxiv.org/pdf/2506.07350v1](http://arxiv.org/pdf/2506.07350v1)**

> **作者:** Yijie Deng; Shuaihang Yuan; Congcong Wen; Hao Huang; Anthony Tzes; Geeta Chandra Raju Bethala; Yi Fang
>
> **摘要:** Spatial awareness is a critical capability for embodied agents, as it enables them to anticipate and reason about unobserved regions. The primary challenge arises from learning the distribution of indoor semantics, complicated by sparse, imbalanced object categories and diverse spatial scales. Existing methods struggle to robustly generate unobserved areas in real time and do not generalize well to new environments. To this end, we propose \textbf{MapBERT}, a novel framework designed to effectively model the distribution of unseen spaces. Motivated by the observation that the one-hot encoding of semantic maps aligns naturally with the binary structure of bit encoding, we, for the first time, leverage a lookup-free BitVAE to encode semantic maps into compact bitwise tokens. Building on this, a masked transformer is employed to infer missing regions and generate complete semantic maps from limited observations. To enhance object-centric reasoning, we propose an object-aware masking strategy that masks entire object categories concurrently and pairs them with learnable embeddings, capturing implicit relationships between object embeddings and spatial tokens. By learning these relationships, the model more effectively captures indoor semantic distributions crucial for practical robotic tasks. Experiments on Gibson benchmarks show that MapBERT achieves state-of-the-art semantic map generation, balancing computational efficiency with accurate reconstruction of unobserved regions.
>
---
#### [new 008] Blending Participatory Design and Artificial Awareness for Trustworthy Autonomous Vehicles
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在提升自动驾驶车辆的信任度。通过用户研究建立驾驶员行为模型，以增强系统透明度与安全性。**

- **链接: [http://arxiv.org/pdf/2506.07633v1](http://arxiv.org/pdf/2506.07633v1)**

> **作者:** Ana Tanevska; Ananthapathmanabhan Ratheesh Kumar; Arabinda Ghosh; Ernesto Casablanca; Ginevra Castellano; Sadegh Soudjani
>
> **备注:** Submitted to IEEE RO-MAN 2025
>
> **摘要:** Current robotic agents, such as autonomous vehicles (AVs) and drones, need to deal with uncertain real-world environments with appropriate situational awareness (SA), risk awareness, coordination, and decision-making. The SymAware project strives to address this issue by designing an architecture for artificial awareness in multi-agent systems, enabling safe collaboration of autonomous vehicles and drones. However, these agents will also need to interact with human users (drivers, pedestrians, drone operators), which in turn requires an understanding of how to model the human in the interaction scenario, and how to foster trust and transparency between the agent and the human. In this work, we aim to create a data-driven model of a human driver to be integrated into our SA architecture, grounding our research in the principles of trustworthy human-agent interaction. To collect the data necessary for creating the model, we conducted a large-scale user-centered study on human-AV interaction, in which we investigate the interaction between the AV's transparency and the users' behavior. The contributions of this paper are twofold: First, we illustrate in detail our human-AV study and its findings, and second we present the resulting Markov chain models of the human driver computed from the study's data. Our results show that depending on the AV's transparency, the scenario's environment, and the users' demographics, we can obtain significant differences in the model's transitions.
>
---
#### [new 009] BR-MPPI: Barrier Rate guided MPPI for Enforcing Multiple Inequality Constraints with Learned Signed Distance Field
- **分类: cs.RO; math.OC**

- **简介: 该论文属于控制任务，解决多不等式约束下的最优控制问题。通过融合MPPI与CBF，提出BR-MPPI算法，提升安全边界附近的控制性能。**

- **链接: [http://arxiv.org/pdf/2506.07325v1](http://arxiv.org/pdf/2506.07325v1)**

> **作者:** Hardik Parwana; Taekyung Kim; Kehan Long; Bardh Hoxha; Hideki Okamoto; Georgios Fainekos; Dimitra Panagou
>
> **摘要:** Model Predictive Path Integral (MPPI) controller is used to solve unconstrained optimal control problems and Control Barrier Function (CBF) is a tool to impose strict inequality constraints, a.k.a, barrier constraints. In this work, we propose an integration of these two methods that employ CBF-like conditions to guide the control sampling procedure of MPPI. CBFs provide an inequality constraint restricting the rate of change of barrier functions by a classK function of the barrier itself. We instead impose the CBF condition as an equality constraint by choosing a parametric linear classK function and treating this parameter as a state in an augmented system. The time derivative of this parameter acts as an additional control input that is designed by MPPI. A cost function is further designed to reignite Nagumo's theorem at the boundary of the safe set by promoting specific values of classK parameter to enforce safety. Our problem formulation results in an MPPI subject to multiple state and control-dependent equality constraints which are non-trivial to satisfy with randomly sampled control inputs. We therefore also introduce state transformations and control projection operations, inspired by the literature on path planning for manifolds, to resolve the aforementioned issue. We show empirically through simulations and experiments on quadrotor that our proposed algorithm exhibits better sampled efficiency and enhanced capability to operate closer to the safe set boundary over vanilla MPPI.
>
---
#### [new 010] Taking Flight with Dialogue: Enabling Natural Language Control for PX4-based Drone Agent
- **分类: cs.RO; I.2.7; I.2.9; I.2.10**

- **简介: 该论文属于自然语言控制任务，旨在解决无人机自主控制难题，通过集成PX4、ROS 2和本地模型实现开放源代码的无人机智能控制。**

- **链接: [http://arxiv.org/pdf/2506.07509v1](http://arxiv.org/pdf/2506.07509v1)**

> **作者:** Shoon Kit Lim; Melissa Jia Ying Chong; Jing Huey Khor; Ting Yang Ling
>
> **备注:** Source code available at: https://github.com/limshoonkit/ros2-agent-ws
>
> **摘要:** Recent advances in agentic and physical artificial intelligence (AI) have largely focused on ground-based platforms such as humanoid and wheeled robots, leaving aerial robots relatively underexplored. Meanwhile, state-of-the-art unmanned aerial vehicle (UAV) multimodal vision-language systems typically rely on closed-source models accessible only to well-resourced organizations. To democratize natural language control of autonomous drones, we present an open-source agentic framework that integrates PX4-based flight control, Robot Operating System 2 (ROS 2) middleware, and locally hosted models using Ollama. We evaluate performance both in simulation and on a custom quadcopter platform, benchmarking four large language model (LLM) families for command generation and three vision-language model (VLM) families for scene understanding.
>
---
#### [new 011] Reproducibility in the Control of Autonomous Mobility-on-Demand Systems
- **分类: cs.RO**

- **简介: 该论文属于AMoD系统研究任务，旨在解决其可复现性问题。通过分析研究流程中的关键环节，提出改进框架与指南，提升结果的可重复性和可比性。**

- **链接: [http://arxiv.org/pdf/2506.07345v1](http://arxiv.org/pdf/2506.07345v1)**

> **作者:** Xinling Li; Meshal Alharbi; Daniele Gammelli; James Harrison; Filipe Rodrigues; Maximilian Schiffer; Marco Pavone; Emilio Frazzoli; Jinhua Zhao; Gioele Zardini
>
> **摘要:** Autonomous Mobility-on-Demand (AMoD) systems, powered by advances in robotics, control, and Machine Learning (ML), offer a promising paradigm for future urban transportation. AMoD offers fast and personalized travel services by leveraging centralized control of autonomous vehicle fleets to optimize operations and enhance service performance. However, the rapid growth of this field has outpaced the development of standardized practices for evaluating and reporting results, leading to significant challenges in reproducibility. As AMoD control algorithms become increasingly complex and data-driven, a lack of transparency in modeling assumptions, experimental setups, and algorithmic implementation hinders scientific progress and undermines confidence in the results. This paper presents a systematic study of reproducibility in AMoD research. We identify key components across the research pipeline, spanning system modeling, control problems, simulation design, algorithm specification, and evaluation, and analyze common sources of irreproducibility. We survey prevalent practices in the literature, highlight gaps, and propose a structured framework to assess and improve reproducibility. Specifically, concrete guidelines are offered, along with a "reproducibility checklist", to support future work in achieving replicable, comparable, and extensible results. While focused on AMoD, the principles and practices we advocate generalize to a broader class of cyber-physical systems that rely on networked autonomy and data-driven control. This work aims to lay the foundation for a more transparent and reproducible research culture in the design and deployment of intelligent mobility systems.
>
---
#### [new 012] Semantics-aware Predictive Inspection Path Planning
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，旨在提高工业环境中特定结构的检查效率。通过识别语义模式并预测场景变化，提出新的规划策略，提升检查速度与覆盖率。**

- **链接: [http://arxiv.org/pdf/2506.06560v1](http://arxiv.org/pdf/2506.06560v1)**

> **作者:** Mihir Dharmadhikari; Kostas Alexis
>
> **备注:** Accepted at IEEE Transactions on Field Robotics
>
> **摘要:** This paper presents a novel semantics-aware inspection path planning paradigm called "Semantics-aware Predictive Planning" (SPP). Industrial environments that require the inspection of specific objects or structures (called "semantics"), such as ballast water tanks inside ships, often present structured and repetitive spatial arrangements of the semantics of interest. Motivated by this, we first contribute an algorithm that identifies spatially repeating patterns of semantics - exact or inexact - in a semantic scene graph representation and makes predictions about the evolution of the graph in the unseen parts of the environment using these patterns. Furthermore, two inspection path planning strategies, tailored to ballast water tank inspection, that exploit these predictions are proposed. To assess the performance of the novel predictive planning paradigm, both simulation and experimental evaluations are performed. First, we conduct a simulation study comparing the method against relevant state-of-the-art techniques and further present tests showing its ability to handle imperfect patterns. Second, we deploy our method onboard a collision-tolerant aerial robot operating inside the ballast tanks of two real ships. The results, both in simulation and field experiments, demonstrate significant improvement over the state-of-the-art in terms of inspection time while maintaining equal or better semantic surface coverage. A set of videos describing the different parts of the method and the field deployments is available at https://tinyurl.com/spp-videos. The code for this work is made available at https://github.com/ntnu-arl/predictive_planning_ros.
>
---
#### [new 013] Multimodal Spatial Language Maps for Robot Navigation and Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于机器人导航与操作任务，旨在解决语言与环境空间映射不精确的问题。通过构建多模态空间语言地图，融合视觉、音频和语言信息，提升机器人对多模态指令的定位与导航能力。**

- **链接: [http://arxiv.org/pdf/2506.06862v1](http://arxiv.org/pdf/2506.06862v1)**

> **作者:** Chenguang Huang; Oier Mees; Andy Zeng; Wolfram Burgard
>
> **备注:** accepted to International Journal of Robotics Research (IJRR). 24 pages, 18 figures. The paper contains texts from VLMaps(arXiv:2210.05714) and AVLMaps(arXiv:2303.07522). The project page is https://mslmaps.github.io/
>
> **摘要:** Grounding language to a navigating agent's observations can leverage pretrained multimodal foundation models to match perceptions to object or event descriptions. However, previous approaches remain disconnected from environment mapping, lack the spatial precision of geometric maps, or neglect additional modality information beyond vision. To address this, we propose multimodal spatial language maps as a spatial map representation that fuses pretrained multimodal features with a 3D reconstruction of the environment. We build these maps autonomously using standard exploration. We present two instances of our maps, which are visual-language maps (VLMaps) and their extension to audio-visual-language maps (AVLMaps) obtained by adding audio information. When combined with large language models (LLMs), VLMaps can (i) translate natural language commands into open-vocabulary spatial goals (e.g., "in between the sofa and TV") directly localized in the map, and (ii) be shared across different robot embodiments to generate tailored obstacle maps on demand. Building upon the capabilities above, AVLMaps extend VLMaps by introducing a unified 3D spatial representation integrating audio, visual, and language cues through the fusion of features from pretrained multimodal foundation models. This enables robots to ground multimodal goal queries (e.g., text, images, or audio snippets) to spatial locations for navigation. Additionally, the incorporation of diverse sensory inputs significantly enhances goal disambiguation in ambiguous environments. Experiments in simulation and real-world settings demonstrate that our multimodal spatial language maps enable zero-shot spatial and multimodal goal navigation and improve recall by 50% in ambiguous scenarios. These capabilities extend to mobile robots and tabletop manipulators, supporting navigation and interaction guided by visual, audio, and spatial cues.
>
---
#### [new 014] Hierarchical Intention Tracking with Switching Trees for Real-Time Adaptation to Dynamic Human Intentions during Collaboration
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在实时跟踪动态人类意图。提出HIT算法，通过概率推理和树状结构实现多层级意图理解，提升协作效率与用户信任。**

- **链接: [http://arxiv.org/pdf/2506.07004v1](http://arxiv.org/pdf/2506.07004v1)**

> **作者:** Zhe Huang; Ye-Ji Mun; Fatemeh Cheraghi Pouria; Katherine Driggs-Campbell
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** During collaborative tasks, human behavior is guided by multiple levels of intentions that evolve over time, such as task sequence preferences and interaction strategies. To adapt to these changing preferences and promptly correct any inaccurate estimations, collaborative robots must accurately track these dynamic human intentions in real time. We propose a Hierarchical Intention Tracking (HIT) algorithm for collaborative robots to track dynamic and hierarchical human intentions effectively in real time. HIT represents human intentions as intention trees with arbitrary depth, and probabilistically tracks human intentions by Bayesian filtering, upward measurement propagation, and downward posterior propagation across all levels. We develop a HIT-based robotic system that dynamically switches between Interaction-Task and Verification-Task trees for a collaborative assembly task, allowing the robot to effectively coordinate human intentions at three levels: task-level (subtask goal locations), interaction-level (mode of engagement with the robot), and verification-level (confirming or correcting intention recognition). Our user study shows that our HIT-based collaborative robot system surpasses existing collaborative robot solutions by achieving a balance between efficiency, physical workload, and user comfort while ensuring safety and task completion. Post-experiment surveys further reveal that the HIT-based system enhances the user trust and minimizes interruptions to user's task flow through its effective understanding of human intentions across multiple levels.
>
---
#### [new 015] Edge-Enabled Collaborative Object Detection for Real-Time Multi-Vehicle Perception
- **分类: cs.RO; cs.AI; cs.CV; cs.MA; cs.NI; I.4.8; I.2.10; I.2.11; I.2.9; C.2.4**

- **简介: 该论文属于多车辆协同感知任务，解决传统系统精度低和云处理延迟高的问题。通过边缘计算与协作算法提升实时目标检测性能。**

- **链接: [http://arxiv.org/pdf/2506.06474v1](http://arxiv.org/pdf/2506.06474v1)**

> **作者:** Everett Richards; Bipul Thapa; Lena Mashayekhy
>
> **备注:** This paper has been accepted to IEEE EDGE 2025. The final version will be published in IEEE Xplore later this year
>
> **摘要:** Accurate and reliable object detection is critical for ensuring the safety and efficiency of Connected Autonomous Vehicles (CAVs). Traditional on-board perception systems have limited accuracy due to occlusions and blind spots, while cloud-based solutions introduce significant latency, making them unsuitable for real-time processing demands required for autonomous driving in dynamic environments. To address these challenges, we introduce an innovative framework, Edge-Enabled Collaborative Object Detection (ECOD) for CAVs, that leverages edge computing and multi-CAV collaboration for real-time, multi-perspective object detection. Our ECOD framework integrates two key algorithms: Perceptive Aggregation and Collaborative Estimation (PACE) and Variable Object Tally and Evaluation (VOTE). PACE aggregates detection data from multiple CAVs on an edge server to enhance perception in scenarios where individual CAVs have limited visibility. VOTE utilizes a consensus-based voting mechanism to improve the accuracy of object classification by integrating data from multiple CAVs. Both algorithms are designed at the edge to operate in real-time, ensuring low-latency and reliable decision-making for CAVs. We develop a hardware-based controlled testbed consisting of camera-equipped robotic CAVs and an edge server to evaluate the efficacy of our framework. Our experimental results demonstrate the significant benefits of ECOD in terms of improved object classification accuracy, outperforming traditional single-perspective onboard approaches by up to 75%, while ensuring low-latency, edge-driven real-time processing. This research highlights the potential of edge computing to enhance collaborative perception for latency-sensitive autonomous systems.
>
---
#### [new 016] Towards Terrain-Aware Task-Driven 3D Scene Graph Generation in Outdoor Environments
- **分类: cs.RO**

- **简介: 该论文属于3D场景图生成任务，旨在解决户外环境中结构化语义建模问题。通过改进方法生成适用于户外的度量-语义点云，提升机器人环境理解能力。**

- **链接: [http://arxiv.org/pdf/2506.06562v1](http://arxiv.org/pdf/2506.06562v1)**

> **作者:** Chad R Samuelson; Timothy W McLain; Joshua G Mangelson
>
> **备注:** Presented at the 2025 IEEE ICRA Workshop on Field Robotics
>
> **摘要:** High-level autonomous operations depend on a robot's ability to construct a sufficiently expressive model of its environment. Traditional three-dimensional (3D) scene representations, such as point clouds and occupancy grids, provide detailed geometric information but lack the structured, semantic organization needed for high-level reasoning. 3D scene graphs (3DSGs) address this limitation by integrating geometric, topological, and semantic relationships into a multi-level graph-based representation. By capturing hierarchical abstractions of objects and spatial layouts, 3DSGs enable robots to reason about environments in a structured manner, improving context-aware decision-making and adaptive planning. Although most recent work has focused on indoor 3DSGs, this paper investigates their construction and utility in outdoor environments. We present a method for generating a task-agnostic metric-semantic point cloud for large outdoor settings and propose modifications to existing indoor 3DSG generation techniques for outdoor applicability. Our preliminary qualitative results demonstrate the feasibility of outdoor 3DSGs and highlight their potential for future deployment in real-world field robotic applications.
>
---
#### [new 017] MorphoCopter: Design, Modeling, and Control of a New Transformable Quad-Bi Copter
- **分类: cs.RO**

- **简介: 该论文属于无人机设计任务，旨在解决传统四旋翼机硬件配置固定、适应性差的问题。通过提出可变形的MorphoCopter，实现飞行器形态快速变换，提升环境适应能力。**

- **链接: [http://arxiv.org/pdf/2506.07204v1](http://arxiv.org/pdf/2506.07204v1)**

> **作者:** Harsh Modi; Hao Su; Xiao Liang; Minghui Zheng
>
> **摘要:** This paper presents a novel morphing quadrotor, named MorphoCopter, covering its design, modeling, control, and experimental tests. It features a unique single rotary joint that enables rapid transformation into an ultra-narrow profile. Although quadrotors have seen widespread adoption in applications such as cinematography, agriculture, and disaster management with increasingly sophisticated control systems, their hardware configurations have remained largely unchanged, limiting their capabilities in certain environments. Our design addresses this by enabling the hardware configuration to change on the fly when required. In standard flight mode, the MorphoCopter adopts an X configuration, functioning as a traditional quadcopter, but can quickly fold into a stacked bicopters arrangement or any configuration in between. Existing morphing designs often sacrifice controllability in compact configurations or rely on complex multi-joint systems. Moreover, our design achieves a greater width reduction than any existing solution. We develop a new inertia and control-action aware adaptive control system that maintains robust performance across all rotary-joint configurations. The prototype can reduce its width from 447 mm to 138 mm (nearly 70\% reduction) in just a few seconds. We validated the MorphoCopter through rigorous simulations and a comprehensive series of flight experiments, including robustness tests, trajectory tracking, and narrow-gap passing tests.
>
---
#### [new 018] Real-Time Execution of Action Chunking Flow Policies
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究实时执行动作分块策略的问题，旨在解决AI系统在动态任务中的延迟与不连贯问题。提出RTC方法，实现异步流畅执行，提升任务成功率与效率。**

- **链接: [http://arxiv.org/pdf/2506.07339v1](http://arxiv.org/pdf/2506.07339v1)**

> **作者:** Kevin Black; Manuel Y. Galliker; Sergey Levine
>
> **摘要:** Modern AI systems, especially those interacting with the physical world, increasingly require real-time performance. However, the high latency of state-of-the-art generalist models, including recent vision-language action models (VLAs), poses a significant challenge. While action chunking has enabled temporal consistency in high-frequency control tasks, it does not fully address the latency problem, leading to pauses or out-of-distribution jerky movements at chunk boundaries. This paper presents a novel inference-time algorithm that enables smooth asynchronous execution of action chunking policies. Our method, real-time chunking (RTC), is applicable to any diffusion- or flow-based VLA out of the box with no re-training. It generates the next action chunk while executing the current one, "freezing" actions guaranteed to execute and "inpainting" the rest. To test RTC, we introduce a new benchmark of 12 highly dynamic tasks in the Kinetix simulator, as well as evaluate 6 challenging real-world bimanual manipulation tasks. Results demonstrate that RTC is fast, performant, and uniquely robust to inference delay, significantly improving task throughput and enabling high success rates in precise tasks $\unicode{x2013}$ such as lighting a match $\unicode{x2013}$ even in the presence of significant latency. See https://pi.website/research/real_time_chunking for videos.
>
---
#### [new 019] SARAL-Bot: Autonomous Robot for Strawberry Plant Care
- **分类: cs.RO**

- **简介: 论文介绍SARAL-Bot机器人，用于草莓种植的自主护理。属于农业机器人任务，解决劳动力短缺问题，实现植株健康监测与维护。**

- **链接: [http://arxiv.org/pdf/2506.06798v1](http://arxiv.org/pdf/2506.06798v1)**

> **作者:** Arif Ahmed; Ritvik Agarwal; Gaurav Srikar; Nathaniel Rose; Parikshit Maini
>
> **备注:** Awarded Best Written Report @ Robotics Design Challenge (Advanced), ASABE 2024
>
> **摘要:** Strawberry farming demands intensive labor for monitoring and maintaining plant health. To address this, Team SARAL develops an autonomous robot for the 2024 ASABE Student Robotics Challenge, capable of navigation, unhealthy leaf detection, and removal. The system addresses labor shortages, reduces costs, and supports sustainable farming through vision-based plant assessment. This work demonstrates the potential of robotics to modernize strawberry cultivation and enable scalable, intelligent agricultural solutions.
>
---
#### [new 020] Fast ECoT: Efficient Embodied Chain-of-Thought via Thoughts Reuse
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决ECoT推理延迟问题。通过缓存和并行生成提升效率，实现更实时的部署。**

- **链接: [http://arxiv.org/pdf/2506.07639v1](http://arxiv.org/pdf/2506.07639v1)**

> **作者:** Zhekai Duan; Yuan Zhang; Shikai Geng; Gaowen Liu; Joschka Boedecker; Chris Xiaoxuan Lu
>
> **摘要:** Embodied Chain-of-Thought (ECoT) reasoning enhances vision-language-action (VLA) models by improving performance and interpretability through intermediate reasoning steps. However, its sequential autoregressive token generation introduces significant inference latency, limiting real-time deployment. We propose Fast ECoT, an inference-time acceleration method that exploits the structured and repetitive nature of ECoT to (1) cache and reuse high-level reasoning across timesteps and (2) parallelise the generation of modular reasoning steps. Additionally, we introduce an asynchronous scheduler that decouples reasoning from action decoding, further boosting responsiveness. Fast ECoT requires no model changes or additional training and integrates easily into existing VLA pipelines. Experiments in both simulation (LIBERO) and real-world robot tasks show up to a 7.5% reduction in latency with comparable or improved task success rate and reasoning faithfulness, bringing ECoT policies closer to practical real-time deployment.
>
---
#### [new 021] Active Illumination Control in Low-Light Environments using NightHawk
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉任务，解决低光环境下图像质量差的问题。提出NightHawk框架，结合主动照明与曝光控制，优化图像质量。**

- **链接: [http://arxiv.org/pdf/2506.06394v1](http://arxiv.org/pdf/2506.06394v1)**

> **作者:** Yash Turkar; Youngjin Kim; Karthik Dantu
>
> **摘要:** Subterranean environments such as culverts present significant challenges to robot vision due to dim lighting and lack of distinctive features. Although onboard illumination can help, it introduces issues such as specular reflections, overexposure, and increased power consumption. We propose NightHawk, a framework that combines active illumination with exposure control to optimize image quality in these settings. NightHawk formulates an online Bayesian optimization problem to determine the best light intensity and exposure-time for a given scene. We propose a novel feature detector-based metric to quantify image utility and use it as the cost function for the optimizer. We built NightHawk as an event-triggered recursive optimization pipeline and deployed it on a legged robot navigating a culvert beneath the Erie Canal. Results from field experiments demonstrate improvements in feature detection and matching by 47-197% enabling more reliable visual estimation in challenging lighting conditions.
>
---
#### [new 022] UruBots Autonomous Cars Challenge Pro Team Description Paper for FIRA 2025
- **分类: cs.RO; cs.SY; eess.IV; eess.SY**

- **简介: 该论文属于自动驾驶任务，旨在解决自主导航问题。团队开发了一辆小型电动车，利用深度学习模型处理视觉信息，实现自动避障和路径规划。**

- **链接: [http://arxiv.org/pdf/2506.07348v1](http://arxiv.org/pdf/2506.07348v1)**

> **作者:** Pablo Moraes; Mónica Rodríguez; Sebastian Barcelona; Angel Da Silva; Santiago Fernandez; Hiago Sodre; Igor Nunes; Bruna Guterres; Ricardo Grando
>
> **摘要:** This paper describes the development of an autonomous car by the UruBots team for the 2025 FIRA Autonomous Cars Challenge (Pro). The project involves constructing a compact electric vehicle, approximately the size of an RC car, capable of autonomous navigation through different tracks. The design incorporates mechanical and electronic components and machine learning algorithms that enable the vehicle to make real-time navigation decisions based on visual input from a camera. We use deep learning models to process camera images and control vehicle movements. Using a dataset of over ten thousand images, we trained a Convolutional Neural Network (CNN) to drive the vehicle effectively, through two outputs, steering and throttle. The car completed the track in under 30 seconds, achieving a pace of approximately 0.4 meters per second while avoiding obstacles.
>
---
#### [new 023] Prime the search: Using large language models for guiding geometric task and motion planning by warm-starting tree search
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于几何任务与运动规划（G-TAMP）任务，旨在解决物体在障碍物中移动的路径规划问题。通过结合大语言模型和蒙特卡洛树搜索，提升规划效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.07062v1](http://arxiv.org/pdf/2506.07062v1)**

> **作者:** Dongryung Lee; Sejune Joo; Kimin Lee; Beomjoon Kim
>
> **备注:** The International Journal of Robotics Research (IJRR)
>
> **摘要:** The problem of relocating a set of objects to designated areas amidst movable obstacles can be framed as a Geometric Task and Motion Planning (G-TAMP) problem, a subclass of task and motion planning (TAMP). Traditional approaches to G-TAMP have relied either on domain-independent heuristics or on learning from planning experience to guide the search, both of which typically demand significant computational resources or data. In contrast, humans often use common sense to intuitively decide which objects to manipulate in G-TAMP problems. Inspired by this, we propose leveraging Large Language Models (LLMs), which have common sense knowledge acquired from internet-scale data, to guide task planning in G-TAMP problems. To enable LLMs to perform geometric reasoning, we design a predicate-based prompt that encodes geometric information derived from a motion planning algorithm. We then query the LLM to generate a task plan, which is then used to search for a feasible set of continuous parameters. Since LLMs are prone to mistakes, instead of committing to LLM's outputs, we extend Monte Carlo Tree Search (MCTS) to a hybrid action space and use the LLM to guide the search. Unlike the previous approach that calls an LLM at every node and incurs high computational costs, we use it to warm-start the MCTS with the nodes explored in completing the LLM's task plan. On six different G-TAMP problems, we show our method outperforms previous LLM planners and pure search algorithms. Code can be found at: https://github.com/iMSquared/prime-the-search
>
---
#### [new 024] RF-Source Seeking with Obstacle Avoidance using Real-time Modified Artificial Potential Fields in Unknown Environments
- **分类: cs.RO**

- **简介: 该论文属于无人机导航任务，解决未知环境中避障与信号源定位问题。提出一种实时改进的人工势场方法，结合RF信号获取方向角，提升导航成功率和效率。**

- **链接: [http://arxiv.org/pdf/2506.06811v1](http://arxiv.org/pdf/2506.06811v1)**

> **作者:** Shahid Mohammad Mulla; Aryan Kanakapudi; Lakshmi Narasimhan; Anuj Tiwari
>
> **备注:** 14 pages, 16 figures, 1 table, shorter version under review for IEEE ICCAS 2025 conference
>
> **摘要:** Navigation of UAVs in unknown environments with obstacles is essential for applications in disaster response and infrastructure monitoring. However, existing obstacle avoidance algorithms, such as Artificial Potential Field (APF) are unable to generalize across environments with different obstacle configurations. Furthermore, the precise location of the final target may not be available in applications such as search and rescue, in which case approaches such as RF source seeking can be used to align towards the target location. This paper proposes a real-time trajectory planning method, which involves real-time adaptation of APF through a sampling-based approach. The proposed approach utilizes only the bearing angle of the target without its precise location, and adjusts the potential field parameters according to the environment with new obstacle configurations in real time. The main contributions of the article are i) an RF source seeking algorithm to provide a bearing angle estimate using RF signal calculations based on antenna placement, and ii) a modified APF for adaptable collision avoidance in changing environments, which are evaluated separately in the simulation software Gazebo, using ROS2 for communication. Simulation results show that the RF source-seeking algorithm achieves high accuracy, with an average angular error of just 1.48 degrees, and with this estimate, the proposed navigation algorithm improves the success rate of reaching the target by 46% and reduces the trajectory length by 1.2% compared to standard potential fields.
>
---
#### [new 025] MapleGrasp: Mask-guided Feature Pooling for Language-driven Efficient Robotic Grasping
- **分类: cs.RO**

- **简介: 该论文属于语言驱动的机器人抓取任务，旨在提升未见过物体的抓取效率与准确性。通过引入掩码引导特征池化和大规模数据集，显著提升了模型性能。**

- **链接: [http://arxiv.org/pdf/2506.06535v1](http://arxiv.org/pdf/2506.06535v1)**

> **作者:** Vineet Bhat; Naman Patel; Prashanth Krishnamurthy; Ramesh Karri; Farshad Khorrami
>
> **摘要:** Robotic manipulation of unseen objects via natural language commands remains challenging. Language driven robotic grasping (LDRG) predicts stable grasp poses from natural language queries and RGB-D images. Here we introduce Mask-guided feature pooling, a lightweight enhancement to existing LDRG methods. Our approach employs a two-stage training strategy: first, a vision-language model generates feature maps from CLIP-fused embeddings, which are upsampled and weighted by text embeddings to produce segmentation masks. Next, the decoder generates separate feature maps for grasp prediction, pooling only token features within these masked regions to efficiently predict grasp poses. This targeted pooling approach reduces computational complexity, accelerating both training and inference. Incorporating mask pooling results in a 12% improvement over prior approaches on the OCID-VLG benchmark. Furthermore, we introduce RefGraspNet, an open-source dataset eight times larger than existing alternatives, significantly enhancing model generalization for open-vocabulary grasping. By extending 2D grasp predictions to 3D via depth mapping and inverse kinematics, our modular method achieves performance comparable to recent Vision-Language-Action (VLA) models on the LIBERO simulation benchmark, with improved generalization across different task suites. Real-world experiments on a 7 DoF Franka robotic arm demonstrate a 57% success rate with unseen objects, surpassing competitive baselines by 7%. Code will be released post publication.
>
---
#### [new 026] Machine Learning-Based Self-Localization Using Internal Sensors for Automating Bulldozers
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自主定位任务，解决 bulldozer 在无 RTK-GNSS 信号下的定位问题。通过机器学习与 EKF 结合，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2506.07271v1](http://arxiv.org/pdf/2506.07271v1)**

> **作者:** Hikaru Sawafuji; Ryota Ozaki; Takuto Motomura; Toyohisa Matsuda; Masanori Tojima; Kento Uchida; Shinichi Shirakawa
>
> **摘要:** Self-localization is an important technology for automating bulldozers. Conventional bulldozer self-localization systems rely on RTK-GNSS (Real Time Kinematic-Global Navigation Satellite Systems). However, RTK-GNSS signals are sometimes lost in certain mining conditions. Therefore, self-localization methods that do not depend on RTK-GNSS are required. In this paper, we propose a machine learning-based self-localization method for bulldozers. The proposed method consists of two steps: estimating local velocities using a machine learning model from internal sensors, and incorporating these estimates into an Extended Kalman Filter (EKF) for global localization. We also created a novel dataset for bulldozer odometry and conducted experiments across various driving scenarios, including slalom, excavation, and driving on slopes. The result demonstrated that the proposed self-localization method suppressed the accumulation of position errors compared to kinematics-based methods, especially when slip occurred. Furthermore, this study showed that bulldozer-specific sensors, such as blade position sensors and hydraulic pressure sensors, contributed to improving self-localization accuracy.
>
---
#### [new 027] RoboPARA: Dual-Arm Robot Planning with Parallel Allocation and Recomposition Across Tasks
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于双臂机器人任务规划领域，旨在解决多任务并行效率低的问题。提出RoboPARA框架，通过图结构优化并行性，提升协作效率。**

- **链接: [http://arxiv.org/pdf/2506.06683v1](http://arxiv.org/pdf/2506.06683v1)**

> **作者:** Shiying Duan; Pei Ren; Nanxiang Jiang; Zhengping Che; Jian Tang; Yifan Sun; Zhaoxin Fan; Wenjun Wu
>
> **摘要:** Dual-arm robots play a crucial role in improving efficiency and flexibility in complex multitasking scenarios. While existing methods have achieved promising results in task planning, they often fail to fully optimize task parallelism, limiting the potential of dual-arm collaboration. To address this issue, we propose RoboPARA, a novel large language model (LLM)-driven framework for dual-arm task parallelism planning. RoboPARA employs a two-stage process: (1) Dependency Graph-based Planning Candidates Generation, which constructs directed acyclic graphs (DAGs) to model task dependencies and eliminate redundancy, and (2) Graph Re-Traversal-based Dual-Arm Parallel Planning, which optimizes DAG traversal to maximize parallelism while maintaining task coherence. In addition, we introduce the Cross-Scenario Dual-Arm Parallel Task dataset (X-DAPT dataset), the first dataset specifically designed to evaluate dual-arm task parallelism across diverse scenarios and difficulty levels. Extensive experiments on the X-DAPT dataset demonstrate that RoboPARA significantly outperforms existing methods, achieving higher efficiency and reliability, particularly in complex task combinations. The code and dataset will be released upon acceptance.
>
---
#### [new 028] CARoL: Context-aware Adaptation for Robot Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决从先验知识中高效学习新任务的问题。提出CARoL框架，通过上下文感知提升学习效率和泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.07006v1](http://arxiv.org/pdf/2506.07006v1)**

> **作者:** Zechen Hu; Tong Xu; Xuesu Xiao; Xuan Wang
>
> **摘要:** Using Reinforcement Learning (RL) to learn new robotic tasks from scratch is often inefficient. Leveraging prior knowledge has the potential to significantly enhance learning efficiency, which, however, raises two critical challenges: how to determine the relevancy of existing knowledge and how to adaptively integrate them into learning a new task. In this paper, we propose Context-aware Adaptation for Robot Learning (CARoL), a novel framework to efficiently learn a similar but distinct new task from prior knowledge. CARoL incorporates context awareness by analyzing state transitions in system dynamics to identify similarities between the new task and prior knowledge. It then utilizes these identified similarities to prioritize and adapt specific knowledge pieces for the new task. Additionally, CARoL has a broad applicability spanning policy-based, value-based, and actor-critic RL algorithms. We validate the efficiency and generalizability of CARoL on both simulated robotic platforms and physical ground vehicles. The simulations include CarRacing and LunarLander environments, where CARoL demonstrates faster convergence and higher rewards when learning policies for new tasks. In real-world experiments, we show that CARoL enables a ground vehicle to quickly and efficiently adapt policies learned in simulation to smoothly traverse real-world off-road terrain.
>
---
#### [new 029] Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决Legged Robot运动控制中的计算效率问题，提出一种基于GPU加速的Primal-Dual iLQR方法，显著提升求解速度。**

- **链接: [http://arxiv.org/pdf/2506.07823v1](http://arxiv.org/pdf/2506.07823v1)**

> **作者:** Lorenzo Amatucci; João Sousa-Pinto; Giulio Turrisi; Dominique Orban; Victor Barasuol; Claudio Semini
>
> **摘要:** This paper introduces a novel Model Predictive Control (MPC) implementation for legged robot locomotion that leverages GPU parallelization. Our approach enables both temporal and state-space parallelization by incorporating a parallel associative scan to solve the primal-dual Karush-Kuhn-Tucker (KKT) system. In this way, the optimal control problem is solved in $\mathcal{O}(n\log{N} + m)$ complexity, instead of $\mathcal{O}(N(n + m)^3)$, where $n$, $m$, and $N$ are the dimension of the system state, control vector, and the length of the prediction horizon. We demonstrate the advantages of this implementation over two state-of-the-art solvers (acados and crocoddyl), achieving up to a 60\% improvement in runtime for Whole Body Dynamics (WB)-MPC and a 700\% improvement for Single Rigid Body Dynamics (SRBD)-MPC when varying the prediction horizon length. The presented formulation scales efficiently with the problem state dimensions as well, enabling the definition of a centralized controller for up to 16 legged robots that can be computed in less than 25 ms. Furthermore, thanks to the JAX implementation, the solver supports large-scale parallelization across multiple environments, allowing the possibility of performing learning with the MPC in the loop directly in GPU.
>
---
#### [new 030] NeSyPack: A Neuro-Symbolic Framework for Bimanual Logistics Packing
- **分类: cs.RO**

- **简介: 该论文提出NeSyPack，一个用于双臂物流打包的神经符号框架，结合数据驱动与符号推理，解决任务分解与执行问题，提升系统可解释性与效率。**

- **链接: [http://arxiv.org/pdf/2506.06567v1](http://arxiv.org/pdf/2506.06567v1)**

> **作者:** Bowei Li; Peiqi Yu; Zhenran Tang; Han Zhou; Yifan Sun; Ruixuan Liu; Changliu Liu
>
> **备注:** 10 pages, 5 figures. Accepted to the RSS 2025 Workshop on Benchmarking Robot Manipulation: Improving Interoperability and Modularity. First Prize in the WBCD competition at ICRA 2025. Equal contribution by Bowei Li and Peiqi Yu
>
> **摘要:** This paper presents NeSyPack, a neuro-symbolic framework for bimanual logistics packing. NeSyPack combines data-driven models and symbolic reasoning to build an explainable hierarchical system that is generalizable, data-efficient, and reliable. It decomposes a task into subtasks via hierarchical reasoning, and further into atomic skills managed by a symbolic skill graph. The graph selects skill parameters, robot configurations, and task-specific control strategies for execution. This modular design enables robustness, adaptability, and efficient reuse - outperforming end-to-end models that require large-scale retraining. Using NeSyPack, our team won the First Prize in the What Bimanuals Can Do (WBCD) competition at the 2025 IEEE International Conference on Robotics and Automation.
>
---
#### [new 031] Tactile MNIST: Benchmarking Active Tactile Perception
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于主动触觉感知任务，旨在解决触觉传感缺乏标准化基准的问题。作者提出了Tactile MNIST基准套件，包含仿真环境和真实数据，用于评估触觉定位、分类等任务。**

- **链接: [http://arxiv.org/pdf/2506.06361v1](http://arxiv.org/pdf/2506.06361v1)**

> **作者:** Tim Schneider; Guillaume Duret; Cristiana de Farias; Roberto Calandra; Liming Chen; Jan Peters
>
> **摘要:** Tactile perception has the potential to significantly enhance dexterous robotic manipulation by providing rich local information that can complement or substitute for other sensory modalities such as vision. However, because tactile sensing is inherently local, it is not well-suited for tasks that require broad spatial awareness or global scene understanding on its own. A human-inspired strategy to address this issue is to consider active perception techniques instead. That is, to actively guide sensors toward regions with more informative or significant features and integrate such information over time in order to understand a scene or complete a task. Both active perception and different methods for tactile sensing have received significant attention recently. Yet, despite advancements, both fields lack standardized benchmarks. To bridge this gap, we introduce the Tactile MNIST Benchmark Suite, an open-source, Gymnasium-compatible benchmark specifically designed for active tactile perception tasks, including localization, classification, and volume estimation. Our benchmark suite offers diverse simulation scenarios, from simple toy environments all the way to complex tactile perception tasks using vision-based tactile sensors. Furthermore, we also offer a comprehensive dataset comprising 13,500 synthetic 3D MNIST digit models and 153,600 real-world tactile samples collected from 600 3D printed digits. Using this dataset, we train a CycleGAN for realistic tactile simulation rendering. By providing standardized protocols and reproducible evaluation frameworks, our benchmark suite facilitates systematic progress in the fields of tactile sensing and active perception.
>
---
#### [new 032] SMaRCSim: Maritime Robotics Simulation Modules
- **分类: cs.RO; cs.GR**

- **简介: 该论文属于水下机器人仿真任务，旨在解决学习方法开发、多平台协作及任务规划集成问题，提出SMaRCSim仿真模块。**

- **链接: [http://arxiv.org/pdf/2506.07781v1](http://arxiv.org/pdf/2506.07781v1)**

> **作者:** Mart Kartašev; David Dörner; Özer Özkahraman; Petter Ögren; Ivan Stenius; John Folkesson
>
> **摘要:** Developing new functionality for underwater robots and testing them in the real world is time-consuming and resource-intensive. Simulation environments allow for rapid testing before field deployment. However, existing tools lack certain functionality for use cases in our project: i) developing learning-based methods for underwater vehicles; ii) creating teams of autonomous underwater, surface, and aerial vehicles; iii) integrating the simulation with mission planning for field experiments. A holistic solution to these problems presents great potential for bringing novel functionality into the underwater domain. In this paper we present SMaRCSim, a set of simulation packages that we have developed to help us address these issues.
>
---
#### [new 033] SpikePingpong: High-Frequency Spike Vision-based Robot Learning for Precise Striking in Table Tennis Game
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人高精度击球任务，旨在解决高速运动物体的精准控制问题。通过结合脉冲视觉与模仿学习，提出SpikePingpong系统，提升乒乓球击打的准确性和策略性。**

- **链接: [http://arxiv.org/pdf/2506.06690v1](http://arxiv.org/pdf/2506.06690v1)**

> **作者:** Hao Wang; Chengkai Hou; Xianglong Li; Yankai Fu; Chenxuan Li; Ning Chen; Gaole Dai; Jiaming Liu; Tiejun Huang; Shanghang Zhang
>
> **摘要:** Learning to control high-speed objects in the real world remains a challenging frontier in robotics. Table tennis serves as an ideal testbed for this problem, demanding both rapid interception of fast-moving balls and precise adjustment of their trajectories. This task presents two fundamental challenges: it requires a high-precision vision system capable of accurately predicting ball trajectories, and it necessitates intelligent strategic planning to ensure precise ball placement to target regions. The dynamic nature of table tennis, coupled with its real-time response requirements, makes it particularly well-suited for advancing robotic control capabilities in fast-paced, precision-critical domains. In this paper, we present SpikePingpong, a novel system that integrates spike-based vision with imitation learning for high-precision robotic table tennis. Our approach introduces two key attempts that directly address the aforementioned challenges: SONIC, a spike camera-based module that achieves millimeter-level precision in ball-racket contact prediction by compensating for real-world uncertainties such as air resistance and friction; and IMPACT, a strategic planning module that enables accurate ball placement to targeted table regions. The system harnesses a 20 kHz spike camera for high-temporal resolution ball tracking, combined with efficient neural network models for real-time trajectory correction and stroke planning. Experimental results demonstrate that SpikePingpong achieves a remarkable 91% success rate for 30 cm accuracy target area and 71% in the more challenging 20 cm accuracy task, surpassing previous state-of-the-art approaches by 38% and 37% respectively. These significant performance improvements enable the robust implementation of sophisticated tactical gameplay strategies, providing a new research perspective for robotic control in high-speed dynamic tasks.
>
---
#### [new 034] Attention-Based Convolutional Neural Network Model for Human Lower Limb Activity Recognition using sEMG
- **分类: cs.RO**

- **简介: 该论文属于人体下肢活动识别任务，旨在通过sEMG信号准确分类行走、站立和坐姿动作，提出一种轻量级注意力CNN模型以实现高效实时识别。**

- **链接: [http://arxiv.org/pdf/2506.06624v1](http://arxiv.org/pdf/2506.06624v1)**

> **作者:** Mojtaba Mollahossein; Farshad Haghgoo Daryakenari; Mohammad Hossein Rohban; Gholamreza Vossoughi
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Accurate classification of lower limb movements using surface electromyography (sEMG) signals plays a crucial role in assistive robotics and rehabilitation systems. In this study, we present a lightweight attention-based deep neural network (DNN) for real-time movement classification using multi-channel sEMG data from the publicly available BASAN dataset. The proposed model consists of only 62,876 parameters and is designed without the need for computationally expensive preprocessing, making it suitable for real-time deployment. We employed a leave-oneout validation strategy to ensure generalizability across subjects, and evaluated the model on three movement classes: walking, standing with knee flexion, and sitting with knee extension. The network achieved 86.74% accuracy on the validation set and 85.38% on the test set, demonstrating strong classification performance under realistic conditions. Comparative analysis with existing models in the literature highlights the efficiency and effectiveness of our approach, especially in scenarios where computational cost and real-time response are critical. The results indicate that the proposed model is a promising candidate for integration into upper-level controllers in human-robot interaction systems.
>
---
#### [new 035] Active Test-time Vision-Language Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉语言导航任务，解决测试时性能下降问题。提出ATENA框架，通过主动学习优化不确定性，提升导航准确性。**

- **链接: [http://arxiv.org/pdf/2506.06630v1](http://arxiv.org/pdf/2506.06630v1)**

> **作者:** Heeju Ko; Sungjune Kim; Gyeongrok Oh; Jeongyoon Yoon; Honglak Lee; Sujin Jang; Seungryong Kim; Sangpil Kim
>
> **摘要:** Vision-Language Navigation (VLN) policies trained on offline datasets often exhibit degraded task performance when deployed in unfamiliar navigation environments at test time, where agents are typically evaluated without access to external interaction or feedback. Entropy minimization has emerged as a practical solution for reducing prediction uncertainty at test time; however, it can suffer from accumulated errors, as agents may become overconfident in incorrect actions without sufficient contextual grounding. To tackle these challenges, we introduce ATENA (Active TEst-time Navigation Agent), a test-time active learning framework that enables a practical human-robot interaction via episodic feedback on uncertain navigation outcomes. In particular, ATENA learns to increase certainty in successful episodes and decrease it in failed ones, improving uncertainty calibration. Here, we propose mixture entropy optimization, where entropy is obtained from a combination of the action and pseudo-expert distributions-a hypothetical action distribution assuming the agent's selected action to be optimal-controlling both prediction confidence and action preference. In addition, we propose a self-active learning strategy that enables an agent to evaluate its navigation outcomes based on confident predictions. As a result, the agent stays actively engaged throughout all iterations, leading to well-grounded and adaptive decision-making. Extensive evaluations on challenging VLN benchmarks-REVERIE, R2R, and R2R-CE-demonstrate that ATENA successfully overcomes distributional shifts at test time, outperforming the compared baseline methods across various settings.
>
---
#### [new 036] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于自动驾驶轨迹选择任务，解决多轨迹安全评估与优选问题。提出DriveSuprim方法，通过分层筛选、数据增强和自蒸馏提升安全性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.06659v1](http://arxiv.org/pdf/2506.06659v1)**

> **作者:** Wenhao Yao; Zhenxin Li; Shiyi Lan; Zi Wang; Xinglong Sun; Jose M. Alvarez; Zuxuan Wu
>
> **备注:** 15 pages, 6 figures
>
> **摘要:** In complex driving environments, autonomous vehicles must navigate safely. Relying on a single predicted path, as in regression-based approaches, usually does not explicitly assess the safety of the predicted trajectory. Selection-based methods address this by generating and scoring multiple trajectory candidates and predicting the safety score for each, but face optimization challenges in precisely selecting the best option from thousands of possibilities and distinguishing subtle but safety-critical differences, especially in rare or underrepresented scenarios. We propose DriveSuprim to overcome these challenges and advance the selection-based paradigm through a coarse-to-fine paradigm for progressive candidate filtering, a rotation-based augmentation method to improve robustness in out-of-distribution scenarios, and a self-distillation framework to stabilize training. DriveSuprim achieves state-of-the-art performance, reaching 93.5% PDMS in NAVSIM v1 and 87.1% EPDMS in NAVSIM v2 without extra data, demonstrating superior safetycritical capabilities, including collision avoidance and compliance with rules, while maintaining high trajectory quality in various driving scenarios.
>
---
#### [new 037] CPS-Guard: Framework for Dependability Assurance of AI- and LLM-Based Cyber-Physical Systems
- **分类: cs.RO; cs.AI; cs.ET; cs.HC; cs.MA; C.3; C.4; D.2.4; D.4.6; I.2.7**

- **简介: 该论文属于AI与CPS的可靠性保障任务，解决AI组件在CPS中难以验证的问题，提出CPS-Guard框架实现自动化迭代验证。**

- **链接: [http://arxiv.org/pdf/2506.06381v1](http://arxiv.org/pdf/2506.06381v1)**

> **作者:** Trisanth Srinivasan; Santosh Patapati; Himani Musku; Idhant Gode; Aditya Arora; Samvit Bhattacharya; Abubakr Nazriev; Sanika Hirave; Zaryab Kanjiani; Srinjoy Ghose; Srinidhi Shetty
>
> **摘要:** Cyber-Physical Systems (CPS) increasingly depend on advanced AI techniques to operate in critical applications. However, traditional verification and validation methods often struggle to handle the unpredictable and dynamic nature of AI components. In this paper, we introduce CPS-Guard, a novel framework that employs multi-role orchestration to automate the iterative assurance process for AI-powered CPS. By assigning specialized roles (e.g., safety monitoring, security assessment, fault injection, and recovery planning) to dedicated agents within a simulated environment, CPS-Guard continuously evaluates and refines AI behavior against a range of dependability requirements. We demonstrate the framework through a case study involving an autonomous vehicle navigating an intersection with an AI-based planner. Our results show that CPS-Guard effectively detects vulnerabilities, manages performance impacts, and supports adaptive recovery strategies, thereby offering a structured and extensible solution for rigorous V&V in safety- and security-critical systems.
>
---
#### [new 038] Very Large-scale Multi-Robot Task Allocation in Challenging Environments via Robot Redistribution
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人任务分配问题，旨在解决复杂环境中机器人冲突与死锁问题，通过路径规划和重新分配提升任务完成效率。**

- **链接: [http://arxiv.org/pdf/2506.07293v1](http://arxiv.org/pdf/2506.07293v1)**

> **作者:** Seabin Lee; Joonyeol Sim; Changjoo Nam
>
> **备注:** 15 pages
>
> **摘要:** We consider the Multi-Robot Task Allocation (MRTA) problem that aims to optimize an assignment of multiple robots to multiple tasks in challenging environments which are with densely populated obstacles and narrow passages. In such environments, conventional methods optimizing the sum-of-cost are often ineffective because the conflicts between robots incur additional costs (e.g., collision avoidance, waiting). Also, an allocation that does not incorporate the actual robot paths could cause deadlocks, which significantly degrade the collective performance of the robots. We propose a scalable MRTA method that considers the paths of the robots to avoid collisions and deadlocks which result in a fast completion of all tasks (i.e., minimizing the \textit{makespan}). To incorporate robot paths into task allocation, the proposed method constructs a roadmap using a Generalized Voronoi Diagram. The method partitions the roadmap into several components to know how to redistribute robots to achieve all tasks with less conflicts between the robots. In the redistribution process, robots are transferred to their final destinations according to a push-pop mechanism with the first-in first-out principle. From the extensive experiments, we show that our method can handle instances with hundreds of robots in dense clutter while competitors are unable to compute a solution within a time limit.
>
---
#### [new 039] Generalized Trajectory Scoring for End-to-end Multimodal Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶中的多模态轨迹规划任务，旨在解决轨迹评分器泛化能力不足的问题。提出GTRS框架，结合粗粒度与细粒度评估，提升轨迹选择性能。**

- **链接: [http://arxiv.org/pdf/2506.06664v1](http://arxiv.org/pdf/2506.06664v1)**

> **作者:** Zhenxin Li; Wenhao Yao; Zi Wang; Xinglong Sun; Joshua Chen; Nadine Chang; Maying Shen; Zuxuan Wu; Shiyi Lan; Jose M. Alvarez
>
> **备注:** The 1st place solution of the End-to-end Driving Track at the CVPR 2025 Autonomous Grand Challenge
>
> **摘要:** End-to-end multi-modal planning is a promising paradigm in autonomous driving, enabling decision-making with diverse trajectory candidates. A key component is a robust trajectory scorer capable of selecting the optimal trajectory from these candidates. While recent trajectory scorers focus on scoring either large sets of static trajectories or small sets of dynamically generated ones, both approaches face significant limitations in generalization. Static vocabularies provide effective coarse discretization but struggle to make fine-grained adaptation, while dynamic proposals offer detailed precision but fail to capture broader trajectory distributions. To overcome these challenges, we propose GTRS (Generalized Trajectory Scoring), a unified framework for end-to-end multi-modal planning that combines coarse and fine-grained trajectory evaluation. GTRS consists of three complementary innovations: (1) a diffusion-based trajectory generator that produces diverse fine-grained proposals; (2) a vocabulary generalization technique that trains a scorer on super-dense trajectory sets with dropout regularization, enabling its robust inference on smaller subsets; and (3) a sensor augmentation strategy that enhances out-of-domain generalization while incorporating refinement training for critical trajectory discrimination. As the winning solution of the Navsim v2 Challenge, GTRS demonstrates superior performance even with sub-optimal sensor inputs, approaching privileged methods that rely on ground-truth perception. Code will be available at https://github.com/NVlabs/GTRS.
>
---
#### [new 040] Robotic Policy Learning via Human-assisted Action Preference Optimization
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决VLA模型依赖专家示范的问题。通过引入HAPO方法，利用人类辅助进行动作偏好优化，提升模型从失败中学习和适应的能力。**

- **链接: [http://arxiv.org/pdf/2506.07127v1](http://arxiv.org/pdf/2506.07127v1)**

> **作者:** Wenke xia; Yichu Yang; Hongtao Wu; Xiao Ma; Tao Kong; Di Hu
>
> **摘要:** Establishing a reliable and iteratively refined robotic system is essential for deploying real-world applications. While Vision-Language-Action (VLA) models are widely recognized as the foundation model for such robotic deployment, their dependence on expert demonstrations hinders the crucial capabilities of correction and learning from failures. To mitigate this limitation, we introduce a Human-assisted Action Preference Optimization method named HAPO, designed to correct deployment failures and foster effective adaptation through preference alignment for VLA models. This method begins with a human-robot collaboration framework for reliable failure correction and interaction trajectory collection through human intervention. These human-intervention trajectories are further employed within the action preference optimization process, facilitating VLA models to mitigate failure action occurrences while enhancing corrective action adaptation. Specifically, we propose an adaptive reweighting algorithm to address the issues of irreversible interactions and token probability mismatch when introducing preference optimization into VLA models, facilitating model learning from binary desirability signals derived from interactions. Through combining these modules, our human-assisted action preference optimization method ensures reliable deployment and effective learning from failure for VLA models. The experiments conducted in simulation and real-world scenarios prove superior generalization and robustness of our framework across a variety of manipulation tasks.
>
---
#### [new 041] RAPID Hand: A Robust, Affordable, Perception-Integrated, Dexterous Manipulation Platform for Generalist Robot Autonomy
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决低成本高灵巧平台稀缺的问题。提出RAPID Hand系统，集成硬件与感知，提升多指机器人数据收集能力。**

- **链接: [http://arxiv.org/pdf/2506.07490v1](http://arxiv.org/pdf/2506.07490v1)**

> **作者:** Zhaoliang Wan; Zetong Bi; Zida Zhou; Hao Ren; Yiming Zeng; Yihan Li; Lu Qi; Xu Yang; Ming-Hsuan Yang; Hui Cheng
>
> **摘要:** This paper addresses the scarcity of low-cost but high-dexterity platforms for collecting real-world multi-fingered robot manipulation data towards generalist robot autonomy. To achieve it, we propose the RAPID Hand, a co-optimized hardware and software platform where the compact 20-DoF hand, robust whole-hand perception, and high-DoF teleoperation interface are jointly designed. Specifically, RAPID Hand adopts a compact and practical hand ontology and a hardware-level perception framework that stably integrates wrist-mounted vision, fingertip tactile sensing, and proprioception with sub-7 ms latency and spatial alignment. Collecting high-quality demonstrations on high-DoF hands is challenging, as existing teleoperation methods struggle with precision and stability on complex multi-fingered systems. We address this by co-optimizing hand design, perception integration, and teleoperation interface through a universal actuation scheme, custom perception electronics, and two retargeting constraints. We evaluate the platform's hardware, perception, and teleoperation interface. Training a diffusion policy on collected data shows superior performance over prior works, validating the system's capability for reliable, high-quality data collection. The platform is constructed from low-cost and off-the-shelf components and will be made public to ensure reproducibility and ease of adoption.
>
---
#### [new 042] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出BridgeVLA，解决3D操作学习中样本效率低的问题，通过输入输出对齐提升性能。**

- **链接: [http://arxiv.org/pdf/2506.07961v1](http://arxiv.org/pdf/2506.07961v1)**

> **作者:** Peiyan Li; Yixiang Chen; Hongtao Wu; Xiao Ma; Xiangnan Wu; Yan Huang; Liang Wang; Tao Kong; Tieniu Tan
>
> **备注:** In Submission
>
> **摘要:** Recently, leveraging pre-trained vision-language models (VLMs) for building vision-language-action (VLA) models has emerged as a promising approach to effective robot manipulation learning. However, only few methods incorporate 3D signals into VLMs for action prediction, and they do not fully leverage the spatial structure inherent in 3D data, leading to low sample efficiency. In this paper, we introduce BridgeVLA, a novel 3D VLA model that (1) projects 3D inputs to multiple 2D images, ensuring input alignment with the VLM backbone, and (2) utilizes 2D heatmaps for action prediction, unifying the input and output spaces within a consistent 2D image space. In addition, we propose a scalable pre-training method that equips the VLM backbone with the capability to predict 2D heatmaps before downstream policy learning. Extensive experiments show the proposed method is able to learn 3D manipulation efficiently and effectively. BridgeVLA outperforms state-of-the-art baseline methods across three simulation benchmarks. In RLBench, it improves the average success rate from 81.4% to 88.2%. In COLOSSEUM, it demonstrates significantly better performance in challenging generalization settings, boosting the average success rate from 56.7% to 64.0%. In GemBench, it surpasses all the comparing baseline methods in terms of average success rate. In real-robot experiments, BridgeVLA outperforms a state-of-the-art baseline method by 32% on average. It generalizes robustly in multiple out-of-distribution settings, including visual disturbances and unseen instructions. Remarkably, it is able to achieve a success rate of 96.8% on 10+ tasks with only 3 trajectories per task, highlighting its extraordinary sample efficiency. Project Website:https://bridgevla.github.io/
>
---
#### [new 043] Design and Implementation of a Peer-to-Peer Communication, Modular and Decentral YellowCube UUV
- **分类: cs.RO**

- **简介: 该论文属于水下机器人设计任务，旨在解决现有UUV集成传感器困难的问题。提出一种模块化、去中心化的YellowCube UUV，采用P2P通信机制实现模块间协作。**

- **链接: [http://arxiv.org/pdf/2506.07924v1](http://arxiv.org/pdf/2506.07924v1)**

> **作者:** Zhizun Xu; Baozhu Jia; Weichao Shi
>
> **摘要:** The underwater Unmanned Vehicles(UUVs) are pivot tools for offshore engineering and oceanographic research. Most existing UUVs do not facilitate easy integration of new or upgraded sensors. A solution to this problem is to have a modular UUV system with changeable payload sections capable of carrying different sensor to suite different missions. The design and implementation of a modular and decentral UUV named YellowCube is presented in the paper. Instead a centralised software architecture which is adopted by the other modular underwater vehicles designs, a Peer-To-Peer(P2P) communication mechanism is implemented among the UUV's modules. The experiments in the laboratory and sea trials have been executed to verify the performances of the UUV.
>
---
#### [new 044] BitVLA: 1-bit Vision-Language-Action Models for Robotics Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在资源受限设备上的部署问题。通过设计1-bit VLA模型BitVLA，实现高效推理与低内存占用。**

- **链接: [http://arxiv.org/pdf/2506.07530v1](http://arxiv.org/pdf/2506.07530v1)**

> **作者:** Hongyu Wang; Chuyan Xiong; Ruiping Wang; Xilin Chen
>
> **备注:** Work in progress
>
> **摘要:** Vision-Language-Action (VLA) models have shown impressive capabilities across a wide range of robotics manipulation tasks. However, their growing model size poses significant challenges for deployment on resource-constrained robotic systems. While 1-bit pretraining has proven effective for enhancing the inference efficiency of large language models with minimal performance loss, its application to VLA models remains underexplored. In this work, we present BitVLA, the first 1-bit VLA model for robotics manipulation, in which every parameter is ternary, i.e., {-1, 0, 1}. To further reduce the memory footprint of the vision encoder, we propose the distillation-aware training strategy that compresses the full-precision encoder to 1.58-bit weights. During this process, a full-precision encoder serves as a teacher model to better align latent representations. Despite the lack of large-scale robotics pretraining, BitVLA achieves performance comparable to the state-of-the-art model OpenVLA-OFT with 4-bit post-training quantization on the LIBERO benchmark, while consuming only 29.8% of the memory. These results highlight BitVLA's promise for deployment on memory-constrained edge devices. We release the code and model weights in https://github.com/ustcwhy/BitVLA.
>
---
#### [new 045] Language-Grounded Hierarchical Planning and Execution with Multi-Robot 3D Scene Graphs
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多机器人系统任务，解决复杂指令执行问题。通过3D场景图融合与LLM解析，实现多机器人协同规划与定位。**

- **链接: [http://arxiv.org/pdf/2506.07454v1](http://arxiv.org/pdf/2506.07454v1)**

> **作者:** Jared Strader; Aaron Ray; Jacob Arkin; Mason B. Peterson; Yun Chang; Nathan Hughes; Christopher Bradley; Yi Xuan Jia; Carlos Nieto-Granda; Rajat Talak; Chuchu Fan; Luca Carlone; Jonathan P. How; Nicholas Roy
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** In this paper, we introduce a multi-robot system that integrates mapping, localization, and task and motion planning (TAMP) enabled by 3D scene graphs to execute complex instructions expressed in natural language. Our system builds a shared 3D scene graph incorporating an open-set object-based map, which is leveraged for multi-robot 3D scene graph fusion. This representation supports real-time, view-invariant relocalization (via the object-based map) and planning (via the 3D scene graph), allowing a team of robots to reason about their surroundings and execute complex tasks. Additionally, we introduce a planning approach that translates operator intent into Planning Domain Definition Language (PDDL) goals using a Large Language Model (LLM) by leveraging context from the shared 3D scene graph and robot capabilities. We provide an experimental assessment of the performance of our system on real-world tasks in large-scale, outdoor environments.
>
---
#### [new 046] Model Analysis And Design Of Ellipse Based Segmented Varying Curved Foot For Biped Robot Walking
- **分类: cs.RO**

- **简介: 该论文属于双足机器人步态控制任务，旨在提升行走能耗效率。通过设计椭圆分段可变曲率足，优化足部形状以降低能量消耗。**

- **链接: [http://arxiv.org/pdf/2506.07283v1](http://arxiv.org/pdf/2506.07283v1)**

> **作者:** Boyang Chen; Xizhe Zang; Chao Song; Yue Zhang; Jie Zhao
>
> **摘要:** This paper presents the modeling, design, and experimental validation of an Ellipse-based Segmented Varying Curvature (ESVC) foot for bipedal robots. Inspired by the segmented curvature rollover shape of human feet, the ESVC foot aims to enhance gait energy efficiency while maintaining analytical tractability for foot location based controller. First, we derive a complete analytical contact model for the ESVC foot by formulating spatial transformations of elliptical segments only using elementary functions. Then a nonlinear programming approach is engaged to determine optimal elliptical parameters of hind foot and fore foot based on a known mid-foot. An error compensation method is introduced to address approximation inaccuracies in rollover length calculation. The proposed ESVC foot is then integrated with a Hybrid Linear Inverted Pendulum model-based walking controller and validated through both simulation and physical experiments on the TT II biped robot. Experimental results across marking time, sagittal, and lateral walking tasks show that the ESVC foot consistently reduces energy consumption compared to line, and flat feet, with up to 18.52\% improvement in lateral walking. These findings demonstrate that the ESVC foot provides a practical and energy-efficient alternative for real-world bipedal locomotion. The proposed design methodology also lays a foundation for data-driven foot shape optimization in future research.
>
---
#### [new 047] Versatile Loco-Manipulation through Flexible Interlimb Coordination
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人自主操作任务，旨在解决复杂环境中灵活协调肢体的问题。提出ReLIC方法，通过强化学习实现灵活的肢体协调与任务执行。**

- **链接: [http://arxiv.org/pdf/2506.07876v1](http://arxiv.org/pdf/2506.07876v1)**

> **作者:** Xinghao Zhu; Yuxin Chen; Lingfeng Sun; Farzad Niroui; Simon Le CleacH; Jiuguang Wang; Kuan Fang
>
> **摘要:** The ability to flexibly leverage limbs for loco-manipulation is essential for enabling autonomous robots to operate in unstructured environments. Yet, prior work on loco-manipulation is often constrained to specific tasks or predetermined limb configurations. In this work, we present Reinforcement Learning for Interlimb Coordination (ReLIC), an approach that enables versatile loco-manipulation through flexible interlimb coordination. The key to our approach is an adaptive controller that seamlessly bridges the execution of manipulation motions and the generation of stable gaits based on task demands. Through the interplay between two controller modules, ReLIC dynamically assigns each limb for manipulation or locomotion and robustly coordinates them to achieve the task success. Using efficient reinforcement learning in simulation, ReLIC learns to perform stable gaits in accordance with the manipulation goals in the real world. To solve diverse and complex tasks, we further propose to interface the learned controller with different types of task specifications, including target trajectories, contact points, and natural language instructions. Evaluated on 12 real-world tasks that require diverse and complex coordination patterns, ReLIC demonstrates its versatility and robustness by achieving a success rate of 78.9% on average. Videos and code can be found at https://relic-locoman.github.io/.
>
---
#### [new 048] Enhancing Situational Awareness in Underwater Robotics with Multi-modal Spatial Perception
- **分类: cs.RO**

- **简介: 该论文属于水下机器人任务，旨在解决视觉退化下的空间感知问题。通过多模态传感融合，提升SLAM的鲁棒性与实时性。**

- **链接: [http://arxiv.org/pdf/2506.06476v1](http://arxiv.org/pdf/2506.06476v1)**

> **作者:** Pushyami Kaveti; Ambjorn Grimsrud Waldum; Hanumant Singh; Martin Ludvigsen
>
> **摘要:** Autonomous Underwater Vehicles (AUVs) and Remotely Operated Vehicles (ROVs) demand robust spatial perception capabilities, including Simultaneous Localization and Mapping (SLAM), to support both remote and autonomous tasks. Vision-based systems have been integral to these advancements, capturing rich color and texture at low cost while enabling semantic scene understanding. However, underwater conditions -- such as light attenuation, backscatter, and low contrast -- often degrade image quality to the point where traditional vision-based SLAM pipelines fail. Moreover, these pipelines typically rely on monocular or stereo inputs, limiting their scalability to the multi-camera configurations common on many vehicles. To address these issues, we propose to leverage multi-modal sensing that fuses data from multiple sensors-including cameras, inertial measurement units (IMUs), and acoustic devices-to enhance situational awareness and enable robust, real-time SLAM. We explore both geometric and learning-based techniques along with semantic analysis, and conduct experiments on the data collected from a work-class ROV during several field deployments in the Trondheim Fjord. Through our experimental results, we demonstrate the feasibility of real-time reliable state estimation and high-quality 3D reconstructions in visually challenging underwater conditions. We also discuss system constraints and identify open research questions, such as sensor calibration, limitations with learning-based methods, that merit further exploration to advance large-scale underwater operations.
>
---
#### [new 049] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人长时序操作任务，旨在解决现有基准在长期规划和语义推理方面的不足。提出RoboCerebra基准，包含大规模数据集、分层框架和评估协议。**

- **链接: [http://arxiv.org/pdf/2506.06677v1](http://arxiv.org/pdf/2506.06677v1)**

> **作者:** Songhao Han; Boxiang Qiu; Yue Liao; Siyuan Huang; Chen Gao; Shuicheng Yan; Si Liu
>
> **备注:** 23 pages, 18 figures
>
> **摘要:** Recent advances in vision-language models (VLMs) have enabled instruction-conditioned robotic systems with improved generalization. However, most existing work focuses on reactive System 1 policies, underutilizing VLMs' strengths in semantic reasoning and long-horizon planning. These System 2 capabilities-characterized by deliberative, goal-directed thinking-remain under explored due to the limited temporal scale and structural complexity of current benchmarks. To address this gap, we introduce RoboCerebra, a benchmark for evaluating high-level reasoning in long-horizon robotic manipulation. RoboCerebra includes: (1) a large-scale simulation dataset with extended task horizons and diverse subtask sequences in household environments; (2) a hierarchical framework combining a high-level VLM planner with a low-level vision-language-action (VLA) controller; and (3) an evaluation protocol targeting planning, reflection, and memory through structured System 1-System 2 interaction. The dataset is constructed via a top-down pipeline, where GPT generates task instructions and decomposes them into subtask sequences. Human operators execute the subtasks in simulation, yielding high-quality trajectories with dynamic object variations. Compared to prior benchmarks, RoboCerebra features significantly longer action sequences and denser annotations. We further benchmark state-of-the-art VLMs as System 2 modules and analyze their performance across key cognitive dimensions, advancing the development of more capable and generalizable robotic planners.
>
---
#### [new 050] Enhancing Robot Safety via MLLM-Based Semantic Interpretation of Failure Data
- **分类: cs.RO**

- **简介: 该论文属于机器人安全任务，旨在解决失败数据难以分析的问题。通过MLLM自动聚类失败数据，提取语义信息，提升机器人学习与安全性。**

- **链接: [http://arxiv.org/pdf/2506.06570v1](http://arxiv.org/pdf/2506.06570v1)**

> **作者:** Aryaman Gupta; Yusuf Umut Ciftci; Somil Bansal
>
> **摘要:** As robotic systems become increasingly integrated into real-world environments, ranging from autonomous vehicles to household assistants, they inevitably encounter diverse and unstructured scenarios that lead to failures. While such failures pose safety and reliability challenges, they also provide rich perceptual data for improving future performance. However, manually analyzing large-scale failure datasets is impractical. In this work, we present a method for automatically organizing large-scale robotic failure data into semantically meaningful clusters, enabling scalable learning from failure without human supervision. Our approach leverages the reasoning capabilities of Multimodal Large Language Models (MLLMs), trained on internet-scale data, to infer high-level failure causes from raw perceptual trajectories and discover interpretable structure within uncurated failure logs. These semantic clusters reveal latent patterns and hypothesized causes of failure, enabling scalable learning from experience. We demonstrate that the discovered failure modes can guide targeted data collection for policy refinement, accelerating iterative improvement in agent policies and overall safety. Additionally, we show that these semantic clusters can be employed for online failure detection, offering a lightweight yet powerful safeguard for real-time adaptation. We demonstrate that this framework enhances robot learning and robustness by transforming real-world failures into actionable and interpretable signals for adaptation.
>
---
#### [new 051] Fractional Collisions: A Framework for Risk Estimation of Counterfactual Conflicts using Autonomous Driving Behavior Simulations
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶安全评估任务，旨在解决冲突风险估计问题。通过模拟场景分析碰撞风险，评估ADS性能。**

- **链接: [http://arxiv.org/pdf/2506.07540v1](http://arxiv.org/pdf/2506.07540v1)**

> **作者:** Sreeja Roy-Singh; Sarvesh Kolekar; Daniel P. Bonny; Kyle Foss
>
> **摘要:** We present a methodology for estimating collision risk from counterfactual simulated scenarios built on sensor data from automated driving systems (ADS) or naturalistic driving databases. Two-agent conflicts are assessed by detecting and classifying conflict type, identifying the agents' roles (initiator or responder), identifying the point of reaction of the responder, and modeling their human behavioral expectations as probabilistic counterfactual trajectories. The states are used to compute velocity differentials at collision, which when combined with crash models, estimates severity of loss in terms of probabilistic injury or property damage, henceforth called fractional collisions. The probabilistic models may also be extended to include other uncertainties associated with the simulation, features, and agents. We verify the effectiveness of the methodology in a synthetic simulation environment using reconstructed trajectories from 300+ collision and near-collision scenes sourced from VTTI's SHRP2 database and Nexar dashboard camera data. Our methodology predicted fractional collisions within 1% of ground truth collisions. We then evaluate agent-initiated collision risk of an arbitrary ADS software release by replacing the naturalistic responder in these synthetic reconstructions with an ADS simulator and comparing the outcome to human-response outcomes. Our ADS reduced naturalistic collisions by 4x and fractional collision risk by ~62%. The framework's utility is also demonstrated on 250k miles of proprietary, open-loop sensor data collected on ADS test vehicles, re-simulated with an arbitrary ADS software release. The ADS initiated conflicts that caused 0.4 injury-causing and 1.7 property-damaging fractional collisions, and the ADS improved collision risk in 96% of the agent-initiated conflicts.
>
---
#### [new 052] LoopDB: A Loop Closure Dataset for Large Scale Simultaneous Localization and Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LoopDB数据集，用于大规模同时定位与地图构建中的回环检测任务，解决算法评估与训练问题。**

- **链接: [http://arxiv.org/pdf/2506.06771v1](http://arxiv.org/pdf/2506.06771v1)**

> **作者:** Mohammad-Maher Nakshbandi; Ziad Sharawy; Dorian Cojocaru; Sorin Grigorescu
>
> **摘要:** In this study, we introduce LoopDB, which is a challenging loop closure dataset comprising over 1000 images captured across diverse environments, including parks, indoor scenes, parking spaces, as well as centered around individual objects. Each scene is represented by a sequence of five consecutive images. The dataset was collected using a high resolution camera, providing suitable imagery for benchmarking the accuracy of loop closure algorithms, typically used in simultaneous localization and mapping. As ground truth information, we provide computed rotations and translations between each consecutive images. Additional to its benchmarking goal, the dataset can be used to train and fine-tune loop closure methods based on deep neural networks. LoopDB is publicly available at https://github.com/RovisLab/LoopDB.
>
---
#### [new 053] FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity
- **分类: cs.CV; cs.AI; cs.CE; cs.LG; cs.RO**

- **简介: 该论文属于3D物理建模任务，旨在无需物体先验信息学习复杂动态场景的物理运动。通过引入物理编码和无散度模块，实现精确的速度场估计与未来帧预测。**

- **链接: [http://arxiv.org/pdf/2506.07865v1](http://arxiv.org/pdf/2506.07865v1)**

> **作者:** Jinxi Li; Ziyang Song; Siyuan Zhou; Bo Yang
>
> **备注:** CVPR 2025. Code and data are available at: https://github.com/vLAR-group/FreeGave
>
> **摘要:** In this paper, we aim to model 3D scene geometry, appearance, and the underlying physics purely from multi-view videos. By applying various governing PDEs as PINN losses or incorporating physics simulation into neural networks, existing works often fail to learn complex physical motions at boundaries or require object priors such as masks or types. In this paper, we propose FreeGave to learn the physics of complex dynamic 3D scenes without needing any object priors. The key to our approach is to introduce a physics code followed by a carefully designed divergence-free module for estimating a per-Gaussian velocity field, without relying on the inefficient PINN losses. Extensive experiments on three public datasets and a newly collected challenging real-world dataset demonstrate the superior performance of our method for future frame extrapolation and motion segmentation. Most notably, our investigation into the learned physics codes reveals that they truly learn meaningful 3D physical motion patterns in the absence of any human labels in training.
>
---
#### [new 054] Safety-Aware Reinforcement Learning for Control via Risk-Sensitive Action-Value Iteration and Quantile Regression
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，解决高方差环境中策略安全性和性能平衡问题。通过引入CVaR和分位数回归，提升安全性与收敛性。**

- **链接: [http://arxiv.org/pdf/2506.06954v1](http://arxiv.org/pdf/2506.06954v1)**

> **作者:** Clinton Enwerem; Aniruddh G. Puranic; John S. Baras; Calin Belta
>
> **备注:** 13 pages, 4 figures. Submission under review
>
> **摘要:** Mainstream approximate action-value iteration reinforcement learning (RL) algorithms suffer from overestimation bias, leading to suboptimal policies in high-variance stochastic environments. Quantile-based action-value iteration methods reduce this bias by learning a distribution of the expected cost-to-go using quantile regression. However, ensuring that the learned policy satisfies safety constraints remains a challenge when these constraints are not explicitly integrated into the RL framework. Existing methods often require complex neural architectures or manual tradeoffs due to combined cost functions. To address this, we propose a risk-regularized quantile-based algorithm integrating Conditional Value-at-Risk (CVaR) to enforce safety without complex architectures. We also provide theoretical guarantees on the contraction properties of the risk-sensitive distributional Bellman operator in Wasserstein space, ensuring convergence to a unique cost distribution. Simulations of a mobile robot in a dynamic reach-avoid task show that our approach leads to more goal successes, fewer collisions, and better safety-performance trade-offs compared to risk-neutral methods.
>
---
#### [new 055] R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶仿真任务，解决动态3D资产插入问题。提出R3D2模型，实现真实光照和阴影效果的实时3D资产插入。**

- **链接: [http://arxiv.org/pdf/2506.07826v1](http://arxiv.org/pdf/2506.07826v1)**

> **作者:** William Ljungbergh; Bernardo Taveira; Wenzhao Zheng; Adam Tonderski; Chensheng Peng; Fredrik Kahl; Christoffer Petersson; Michael Felsberg; Kurt Keutzer; Masayoshi Tomizuka; Wei Zhan
>
> **摘要:** Validating autonomous driving (AD) systems requires diverse and safety-critical testing, making photorealistic virtual environments essential. Traditional simulation platforms, while controllable, are resource-intensive to scale and often suffer from a domain gap with real-world data. In contrast, neural reconstruction methods like 3D Gaussian Splatting (3DGS) offer a scalable solution for creating photorealistic digital twins of real-world driving scenes. However, they struggle with dynamic object manipulation and reusability as their per-scene optimization-based methodology tends to result in incomplete object models with integrated illumination effects. This paper introduces R3D2, a lightweight, one-step diffusion model designed to overcome these limitations and enable realistic insertion of complete 3D assets into existing scenes by generating plausible rendering effects-such as shadows and consistent lighting-in real time. This is achieved by training R3D2 on a novel dataset: 3DGS object assets are generated from in-the-wild AD data using an image-conditioned 3D generative model, and then synthetically placed into neural rendering-based virtual environments, allowing R3D2 to learn realistic integration. Quantitative and qualitative evaluations demonstrate that R3D2 significantly enhances the realism of inserted assets, enabling use-cases like text-to-3D asset insertion and cross-scene/dataset object transfer, allowing for true scalability in AD validation. To promote further research in scalable and realistic AD simulation, we will release our dataset and code, see https://research.zenseact.com/publications/R3D2/.
>
---
#### [new 056] Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于离线分层强化学习任务，旨在解决长任务中子目标选择与状态转移拼接效率低的问题。提出GAS框架，通过图搜索和时序表示提升性能。**

- **链接: [http://arxiv.org/pdf/2506.07744v1](http://arxiv.org/pdf/2506.07744v1)**

> **作者:** Seungho Baek; Taegeon Park; Jongchan Park; Seungjun Oh; Yusung Kim
>
> **备注:** ICML 2025
>
> **摘要:** Existing offline hierarchical reinforcement learning methods rely on high-level policy learning to generate subgoal sequences. However, their efficiency degrades as task horizons increase, and they lack effective strategies for stitching useful state transitions across different trajectories. We propose Graph-Assisted Stitching (GAS), a novel framework that formulates subgoal selection as a graph search problem rather than learning an explicit high-level policy. By embedding states into a Temporal Distance Representation (TDR) space, GAS clusters semantically similar states from different trajectories into unified graph nodes, enabling efficient transition stitching. A shortest-path algorithm is then applied to select subgoal sequences within the graph, while a low-level policy learns to reach the subgoals. To improve graph quality, we introduce the Temporal Efficiency (TE) metric, which filters out noisy or inefficient transition states, significantly enhancing task performance. GAS outperforms prior offline HRL methods across locomotion, navigation, and manipulation tasks. Notably, in the most stitching-critical task, it achieves a score of 88.3, dramatically surpassing the previous state-of-the-art score of 1.0. Our source code is available at: https://github.com/qortmdgh4141/GAS.
>
---
#### [new 057] UA-Pose: Uncertainty-Aware 6D Object Pose Estimation and Online Object Completion with Partial References
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于6D物体位姿估计任务，解决从部分参考信息中准确估计物体位姿和完成物体的问题。工作包括引入不确定性感知方法，提升估计精度与完整性。**

- **链接: [http://arxiv.org/pdf/2506.07996v1](http://arxiv.org/pdf/2506.07996v1)**

> **作者:** Ming-Feng Li; Xin Yang; Fu-En Wang; Hritam Basak; Yuyin Sun; Shreekant Gayaka; Min Sun; Cheng-Hao Kuo
>
> **备注:** CVPR 2025
>
> **摘要:** 6D object pose estimation has shown strong generalizability to novel objects. However, existing methods often require either a complete, well-reconstructed 3D model or numerous reference images that fully cover the object. Estimating 6D poses from partial references, which capture only fragments of an object's appearance and geometry, remains challenging. To address this, we propose UA-Pose, an uncertainty-aware approach for 6D object pose estimation and online object completion specifically designed for partial references. We assume access to either (1) a limited set of RGBD images with known poses or (2) a single 2D image. For the first case, we initialize a partial object 3D model based on the provided images and poses, while for the second, we use image-to-3D techniques to generate an initial object 3D model. Our method integrates uncertainty into the incomplete 3D model, distinguishing between seen and unseen regions. This uncertainty enables confidence assessment in pose estimation and guides an uncertainty-aware sampling strategy for online object completion, enhancing robustness in pose estimation accuracy and improving object completeness. We evaluate our method on the YCB-Video, YCBInEOAT, and HO3D datasets, including RGBD sequences of YCB objects manipulated by robots and human hands. Experimental results demonstrate significant performance improvements over existing methods, particularly when object observations are incomplete or partially captured. Project page: https://minfenli.github.io/UA-Pose/
>
---
#### [new 058] Multi-Step Guided Diffusion for Image Restoration on Edge Devices: Toward Lightweight Perception in Embodied AI
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于图像修复任务，旨在提升边缘设备上的图像恢复质量与效率。通过多步优化策略增强扩散模型性能，实验证明其在超分辨率和去模糊任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2506.07286v1](http://arxiv.org/pdf/2506.07286v1)**

> **作者:** Aditya Chakravarty
>
> **备注:** Accepted in CVPR 2025 Embodied AI Workshop
>
> **摘要:** Diffusion models have shown remarkable flexibility for solving inverse problems without task-specific retraining. However, existing approaches such as Manifold Preserving Guided Diffusion (MPGD) apply only a single gradient update per denoising step, limiting restoration fidelity and robustness, especially in embedded or out-of-distribution settings. In this work, we introduce a multistep optimization strategy within each denoising timestep, significantly enhancing image quality, perceptual accuracy, and generalization. Our experiments on super-resolution and Gaussian deblurring demonstrate that increasing the number of gradient updates per step improves LPIPS and PSNR with minimal latency overhead. Notably, we validate this approach on a Jetson Orin Nano using degraded ImageNet and a UAV dataset, showing that MPGD, originally trained on face datasets, generalizes effectively to natural and aerial scenes. Our findings highlight MPGD's potential as a lightweight, plug-and-play restoration module for real-time visual perception in embodied AI agents such as drones and mobile robots.
>
---
#### [new 059] Towards Data-Driven Model-Free Safety-Critical Control
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于机器人安全控制任务，解决模型无关CBF中参数难以确定的问题，通过神经网络学习Lyapunov函数并估计衰减率，提升控制安全性。**

- **链接: [http://arxiv.org/pdf/2506.06931v1](http://arxiv.org/pdf/2506.06931v1)**

> **作者:** Zhe Shen; Yitaek Kim; Christoffer Sloth
>
> **备注:** submitted to IROS 2025
>
> **摘要:** This paper presents a framework for enabling safe velocity control of general robotic systems using data-driven model-free Control Barrier Functions (CBFs). Model-free CBFs rely on an exponentially stable velocity controller and a design parameter (e.g. alpha in CBFs); this design parameter depends on the exponential decay rate of the controller. However, in practice, the decay rate is often unavailable, making it non-trivial to use model-free CBFs, as it requires manual tuning for alpha. To address this, a Neural Network is used to learn the Lyapunov function from data, and the maximum decay rate of the systems built-in velocity controller is subsequently estimated. Furthermore, to integrate the estimated decay rate with model-free CBFs, we derive a probabilistic safety condition that incorporates a confidence bound on the violation rate of the exponential stability condition, using Chernoff bound. This enhances robustness against uncertainties in stability violations. The proposed framework has been tested on a UR5e robot in multiple experimental settings, and its effectiveness in ensuring safe velocity control with model-free CBFs has been demonstrated.
>
---
#### [new 060] Hierarchical Scoring with 3D Gaussian Splatting for Instance Image-Goal Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于实例图像目标导航任务，解决目标匹配与视角选择问题。提出分层评分框架，结合语义与几何信息提升导航性能。**

- **链接: [http://arxiv.org/pdf/2506.07338v1](http://arxiv.org/pdf/2506.07338v1)**

> **作者:** Yijie Deng; Shuaihang Yuan; Geeta Chandra Raju Bethala; Anthony Tzes; Yu-Shen Liu; Yi Fang
>
> **摘要:** Instance Image-Goal Navigation (IIN) requires autonomous agents to identify and navigate to a target object or location depicted in a reference image captured from any viewpoint. While recent methods leverage powerful novel view synthesis (NVS) techniques, such as three-dimensional Gaussian splatting (3DGS), they typically rely on randomly sampling multiple viewpoints or trajectories to ensure comprehensive coverage of discriminative visual cues. This approach, however, creates significant redundancy through overlapping image samples and lacks principled view selection, substantially increasing both rendering and comparison overhead. In this paper, we introduce a novel IIN framework with a hierarchical scoring paradigm that estimates optimal viewpoints for target matching. Our approach integrates cross-level semantic scoring, utilizing CLIP-derived relevancy fields to identify regions with high semantic similarity to the target object class, with fine-grained local geometric scoring that performs precise pose estimation within promising regions. Extensive evaluations demonstrate that our method achieves state-of-the-art performance on simulated IIN benchmarks and real-world applicability.
>
---
#### [new 061] Towards Infant Sleep-Optimized Driving: Synergizing Wearable and Vehicle Sensing in Intelligent Cruise Control
- **分类: cs.LG; cs.ET; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于智能驾驶任务，旨在解决婴儿睡眠受车辆操作影响的问题。通过融合可穿戴设备与车辆数据，利用强化学习优化驾驶行为，提升婴儿睡眠质量。**

- **链接: [http://arxiv.org/pdf/2506.06459v1](http://arxiv.org/pdf/2506.06459v1)**

> **作者:** Ruitao Chen; Mozhang Guo; Jinge Li
>
> **摘要:** Automated driving (AD) has substantially improved vehicle safety and driving comfort, but their impact on passenger well-being, particularly infant sleep, is not sufficiently studied. Sudden acceleration, abrupt braking, and sharp maneuvers can disrupt infant sleep, compromising both passenger comfort and parental convenience. To solve this problem, this paper explores the integration of reinforcement learning (RL) within AD to personalize driving behavior and optimally balance occupant comfort and travel efficiency. In particular, we propose an intelligent cruise control framework that adapts to varying driving conditions to enhance infant sleep quality by effectively synergizing wearable sensing and vehicle data. Long short-term memory (LSTM) and transformer-based neural networks are integrated with RL to model the relationship between driving behavior and infant sleep quality under diverse traffic and road conditions. Based on the sleep quality indicators from the wearable sensors, driving action data from vehicle controllers, and map data from map applications, the model dynamically computes the optimal driving aggressiveness level, which is subsequently translated into specific AD control strategies, e.g., the magnitude and frequency of acceleration, lane change, and overtaking. Simulation results demonstrate that the proposed solution significantly improves infant sleep quality compared to baseline methods, while preserving desirable travel efficiency.
>
---
#### [new 062] QForce-RL: Quantized FPGA-Optimized Reinforcement Learning Compute Engine
- **分类: cs.AR; cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于强化学习任务，旨在解决FPGA部署资源消耗大的问题。通过量化和优化架构，提升性能并降低能耗。**

- **链接: [http://arxiv.org/pdf/2506.07046v1](http://arxiv.org/pdf/2506.07046v1)**

> **作者:** Anushka Jha; Tanushree Dewangan; Mukul Lokhande; Santosh Kumar Vishvakarma
>
> **摘要:** Reinforcement Learning (RL) has outperformed other counterparts in sequential decision-making and dynamic environment control. However, FPGA deployment is significantly resource-expensive, as associated with large number of computations in training agents with high-quality images and possess new challenges. In this work, we propose QForce-RL takes benefits of quantization to enhance throughput and reduce energy footprint with light-weight RL architecture, without significant performance degradation. QForce-RL takes advantages from E2HRL to reduce overall RL actions to learn desired policy and QuaRL for quantization based SIMD for hardware acceleration. We have also provided detailed analysis for different RL environments, with emphasis on model size, parameters, and accelerated compute ops. The architecture is scalable for resource-constrained devices and provide parametrized efficient deployment with flexibility in latency, throughput, power, and energy efficiency. The proposed QForce-RL provides performance enhancement up to 2.3x and better FPS - 2.6x compared to SoTA works.
>
---
#### [new 063] Active Lubrication of Transluminal Medical Instruments
- **分类: physics.med-ph; cs.RO**

- **简介: 该论文属于医疗设备领域，旨在解决内窥镜和导管在体内运动时的摩擦问题。通过超声振动产生润滑层，减少摩擦并防止弯曲，提升手术安全性。**

- **链接: [http://arxiv.org/pdf/2506.07225v1](http://arxiv.org/pdf/2506.07225v1)**

> **作者:** Mostafa A. Atalla; Jelte Nieuwenhuis; Alan Martin; Xuan Wang; Ahranee Canden; Matt J. Carré; Roger Lewis; Aimée Sakes; Michaël Wiertlewski
>
> **摘要:** Transluminal minimally invasive surgery uses natural orifices and small incisions to access internal anatomical structures, promoting quicker recovery and reduced morbidity. However, navigating instruments--catheters and endoscopes--through anatomical pathways creates frictional interactions with luminal walls, risking complications such as perforation, poor haptic feedback, and instrument buckling. In this paper, we present a new approach to actively lubricate transluminal instruments and dynamically reduce friction with surrounding tissues. This approach employs ultrasonic vibrations, at the instrument surface, to generate a pressurized fluid layer at the contact interface, lubricating the interface and thereby reducing friction. We implemented this approach in a prototype catheter, which we validated under dry and liquid-lubricated conditions, across rigid and soft interfaces, and along varied anatomical curvatures. In a cardiac catheter use case, active lubrication reduced friction by up to 42% on ex-vivo porcine aorta tissue and 82% on rigid substrates, denoting its potential performance on healthy and calcified tissue, respectively. Thermal imaging confirmed that temperature at the tissue-catheter interface remained within safe limits. Additionally, the system effectively prevented buckling during catheter insertion experiment, further showcasing its potential. By minimizing injury risk and enhancing procedural stability, active lubrication can drastically enhance the safety and efficacy of transluminal interventions.
>
---
#### [new 064] Curriculum Learning With Counterfactual Group Relative Policy Advantage For Multi-Agent Reinforcement Learning
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于多智能体强化学习任务，旨在解决传统方法在动态环境中的适应性差和策略次优问题。提出一种动态课程学习框架，结合反事实组相对策略优势机制，提升训练稳定性和性能。**

- **链接: [http://arxiv.org/pdf/2506.07548v1](http://arxiv.org/pdf/2506.07548v1)**

> **作者:** Weiqiang Jin; Hongyang Du; Guizhong Liu; Dong In Kim
>
> **备注:** 16 pages; 12figures
>
> **摘要:** Multi-agent reinforcement learning (MARL) has achieved strong performance in cooperative adversarial tasks. However, most existing methods typically train agents against fixed opponent strategies and rely on such meta-static difficulty conditions, which limits their adaptability to changing environments and often leads to suboptimal policies. Inspired by the success of curriculum learning (CL) in supervised tasks, we propose a dynamic CL framework for MARL that employs an self-adaptive difficulty adjustment mechanism. This mechanism continuously modulates opponent strength based on real-time agent training performance, allowing agents to progressively learn from easier to more challenging scenarios. However, the dynamic nature of CL introduces instability due to nonstationary environments and sparse global rewards. To address this challenge, we develop a Counterfactual Group Relative Policy Advantage (CGRPA), which is tightly coupled with the curriculum by providing intrinsic credit signals that reflect each agent's impact under evolving task demands. CGRPA constructs a counterfactual advantage function that isolates individual contributions within group behavior, facilitating more reliable policy updates throughout the curriculum. CGRPA evaluates each agent's contribution through constructing counterfactual action advantage function, providing intrinsic rewards that enhance credit assignment and stabilize learning under non-stationary conditions. Extensive experiments demonstrate that our method improves both training stability and final performance, achieving competitive results against state-of-the-art methods. The code is available at https://github.com/NICE-HKU/CL2MARL-SMAC.
>
---
#### [new 065] Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks
- **分类: cs.LG; cs.AI; cs.NI; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于多无人机协同控制任务，解决动态环境中无人机运动与通信的联合优化问题。提出基于大语言模型的分层协作方法，提升系统效率与安全性。**

- **链接: [http://arxiv.org/pdf/2506.06532v1](http://arxiv.org/pdf/2506.06532v1)**

> **作者:** Zijiang Yan; Hao Zhou; Jianhua Pei; Hina Tabassum
>
> **备注:** Accepted in ICML 2025 Workshop on Machine Learning for Wireless Communication and Networks (ML4Wireless)
>
> **摘要:** Unmanned aerial vehicles (UAVs) have been widely adopted in various real-world applications. However, the control and optimization of multi-UAV systems remain a significant challenge, particularly in dynamic and constrained environments. This work explores the joint motion and communication control of multiple UAVs operating within integrated terrestrial and non-terrestrial networks that include high-altitude platform stations (HAPS). Specifically, we consider an aerial highway scenario in which UAVs must accelerate, decelerate, and change lanes to avoid collisions and maintain overall traffic flow. Different from existing studies, we propose a novel hierarchical and collaborative method based on large language models (LLMs). In our approach, an LLM deployed on the HAPS performs UAV access control, while another LLM onboard each UAV handles motion planning and control. This LLM-based framework leverages the rich knowledge embedded in pre-trained models to enable both high-level strategic planning and low-level tactical decisions. This knowledge-driven paradigm holds great potential for the development of next-generation 3D aerial highway systems. Experimental results demonstrate that our proposed collaborative LLM-based method achieves higher system rewards, lower operational costs, and significantly reduced UAV collision rates compared to baseline approaches.
>
---
#### [new 066] Deep Equivariant Multi-Agent Control Barrier Functions
- **分类: eess.SY; cs.MA; cs.RO; cs.SY**

- **简介: 该论文属于多智能体系统安全控制任务，旨在解决数据驱动策略的安全性问题。通过引入对称性增强的控制屏障函数，提升系统的安全性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.07755v1](http://arxiv.org/pdf/2506.07755v1)**

> **作者:** Nikolaos Bousias; Lars Lindemann; George Pappas
>
> **摘要:** With multi-agent systems increasingly deployed autonomously at scale in complex environments, ensuring safety of the data-driven policies is critical. Control Barrier Functions have emerged as an effective tool for enforcing safety constraints, yet existing learning-based methods often lack in scalability, generalization and sampling efficiency as they overlook inherent geometric structures of the system. To address this gap, we introduce symmetries-infused distributed Control Barrier Functions, enforcing the satisfaction of intrinsic symmetries on learnable graph-based safety certificates. We theoretically motivate the need for equivariant parametrization of CBFs and policies, and propose a simple, yet efficient and adaptable methodology for constructing such equivariant group-modular networks via the compatible group actions. This approach encodes safety constraints in a distributed data-efficient manner, enabling zero-shot generalization to larger and denser swarms. Through extensive simulations on multi-robot navigation tasks, we demonstrate that our method outperforms state-of-the-art baselines in terms of safety, scalability, and task success rates, highlighting the importance of embedding symmetries in safe distributed neural policies.
>
---
#### [new 067] Reading in the Dark with Foveated Event Vision
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于OCR任务，解决低光和高速场景下文本识别难题。通过事件相机和眼动追踪，降低带宽并提升识别效果。**

- **链接: [http://arxiv.org/pdf/2506.06918v1](http://arxiv.org/pdf/2506.06918v1)**

> **作者:** Carl Brander; Giovanni Cioffi; Nico Messikommer; Davide Scaramuzza
>
> **备注:** CVPR 2025 Workshop on Event-based Vision
>
> **摘要:** Current smart glasses equipped with RGB cameras struggle to perceive the environment in low-light and high-speed motion scenarios due to motion blur and the limited dynamic range of frame cameras. Additionally, capturing dense images with a frame camera requires large bandwidth and power consumption, consequently draining the battery faster. These challenges are especially relevant for developing algorithms that can read text from images. In this work, we propose a novel event-based Optical Character Recognition (OCR) approach for smart glasses. By using the eye gaze of the user, we foveate the event stream to significantly reduce bandwidth by around 98% while exploiting the benefits of event cameras in high-dynamic and fast scenes. Our proposed method performs deep binary reconstruction trained on synthetic data and leverages multimodal LLMs for OCR, outperforming traditional OCR solutions. Our results demonstrate the ability to read text in low light environments where RGB cameras struggle while using up to 2400 times less bandwidth than a wearable RGB camera.
>
---
#### [new 068] LogoSP: Local-global Grouping of Superpoints for Unsupervised Semantic Segmentation of 3D Point Clouds
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于3D点云语义分割任务，解决无监督学习中缺乏人工标注的问题。通过结合局部与全局特征，生成高精度伪标签，提升分割性能。**

- **链接: [http://arxiv.org/pdf/2506.07857v1](http://arxiv.org/pdf/2506.07857v1)**

> **作者:** Zihui Zhang; Weisheng Dai; Hongtao Wen; Bo Yang
>
> **备注:** CVPR 2025. Code and data are available at: https://github.com/vLAR-group/LogoSP
>
> **摘要:** We study the problem of unsupervised 3D semantic segmentation on raw point clouds without needing human labels in training. Existing methods usually formulate this problem into learning per-point local features followed by a simple grouping strategy, lacking the ability to discover additional and possibly richer semantic priors beyond local features. In this paper, we introduce LogoSP to learn 3D semantics from both local and global point features. The key to our approach is to discover 3D semantic information by grouping superpoints according to their global patterns in the frequency domain, thus generating highly accurate semantic pseudo-labels for training a segmentation network. Extensive experiments on two indoor and an outdoor datasets show that our LogoSP surpasses all existing unsupervised methods by large margins, achieving the state-of-the-art performance for unsupervised 3D semantic segmentation. Notably, our investigation into the learned global patterns reveals that they truly represent meaningful 3D semantics in the absence of human labels during training.
>
---
## 更新

#### [replaced 001] LLM-HDR: Bridging LLM-based Perception and Self-Supervision for Unpaired LDR-to-HDR Image Reconstruction
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO; Artificial intelligence, Computer vision, Machine learning, Deep
  learning; I.3.3; I.4.5**

- **链接: [http://arxiv.org/pdf/2410.15068v3](http://arxiv.org/pdf/2410.15068v3)**

> **作者:** Hrishav Bakul Barua; Kalin Stefanov; Lemuel Lai En Che; Abhinav Dhall; KokSheik Wong; Ganesh Krishnasamy
>
> **摘要:** The translation of Low Dynamic Range (LDR) to High Dynamic Range (HDR) images is an important computer vision task. There is a significant amount of research utilizing both conventional non-learning methods and modern data-driven approaches, focusing on using both single-exposed and multi-exposed LDR for HDR image reconstruction. However, most current state-of-the-art methods require high-quality paired {LDR,HDR} datasets for model training. In addition, there is limited literature on using unpaired datasets for this task, that is, the model learns a mapping between domains, i.e., {LDR,HDR}. This paper proposes LLM-HDR, a method that integrates the perception of Large Language Models (LLM) into a modified semantic- and cycle-consistent adversarial architecture that utilizes unpaired {LDR,HDR} datasets for training. The method introduces novel artifact- and exposure-aware generators to address visual artifact removal and an encoder and loss to address semantic consistency, another under-explored topic. LLM-HDR is the first to use an LLM for the {LDR,HDR} translation task in a self-supervised setup. The method achieves state-of-the-art performance across several benchmark datasets and reconstructs high-quality HDR images. The official website of this work is available at: https://github.com/HrishavBakulBarua/LLM-HDR
>
---
#### [replaced 002] Unifying 2D and 3D Vision-Language Understanding
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10745v3](http://arxiv.org/pdf/2503.10745v3)**

> **作者:** Ayush Jain; Alexander Swerdlow; Yuzhou Wang; Sergio Arnaud; Ada Martin; Alexander Sax; Franziska Meier; Katerina Fragkiadaki
>
> **备注:** The first two authors contributed equally
>
> **摘要:** Progress in 3D vision-language learning has been hindered by the scarcity of large-scale 3D datasets. We introduce UniVLG, a unified architecture for 2D and 3D vision-language understanding that bridges the gap between existing 2D-centric models and the rich 3D sensory data available in embodied systems. Our approach initializes most model weights from pre-trained 2D models and trains on both 2D and 3D vision-language data. We propose a novel language-conditioned mask decoder shared across 2D and 3D modalities to ground objects effectively in both RGB and RGB-D images, outperforming box-based approaches. To further reduce the domain gap between 2D and 3D, we incorporate 2D-to-3D lifting strategies, enabling UniVLG to utilize 2D data to enhance 3D performance. With these innovations, our model achieves state-of-the-art performance across multiple 3D vision-language grounding tasks, demonstrating the potential of transferring advances from 2D vision-language learning to the data-constrained 3D domain. Furthermore, co-training on both 2D and 3D data enhances performance across modalities without sacrificing 2D capabilities. By removing the reliance on 3D mesh reconstruction and ground-truth object proposals, UniVLG sets a new standard for realistic, embodied-aligned evaluation. Code and additional visualizations are available at https://univlg.github.io .
>
---
#### [replaced 003] A Learning-based Quadcopter Controller with Extreme Adaptation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.12949v2](http://arxiv.org/pdf/2409.12949v2)**

> **作者:** Dingqi Zhang; Antonio Loquercio; Jerry Tang; Ting-Hao Wang; Jitendra Malik; Mark W. Mueller
>
> **备注:** Accepted for the Transaction on Robotics (T-RO), April 2025
>
> **摘要:** This paper introduces a learning-based low-level controller for quadcopters, which adaptively controls quadcopters with significant variations in mass, size, and actuator capabilities. Our approach leverages a combination of imitation learning and reinforcement learning, creating a fast-adapting and general control framework for quadcopters that eliminates the need for precise model estimation or manual tuning. The controller estimates a latent representation of the vehicle's system parameters from sensor-action history, enabling it to adapt swiftly to diverse dynamics. Extensive evaluations in simulation demonstrate the controller's ability to generalize to unseen quadcopter parameters, with an adaptation range up to 16 times broader than the training set. In real-world tests, the controller is successfully deployed on quadcopters with mass differences of 3.7 times and propeller constants varying by more than 100 times, while also showing rapid adaptation to disturbances such as off-center payloads and motor failures. These results highlight the potential of our controller in extreme adaptation to simplify the design process and enhance the reliability of autonomous drone operations in unpredictable environments. The video and code are at: https://github.com/muellerlab/xadapt_ctrl
>
---
#### [replaced 004] Delayed-Decision Motion Planning in the Presence of Multiple Predictions
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.20636v2](http://arxiv.org/pdf/2502.20636v2)**

> **作者:** David Isele; Alexandre Miranda Anon; Faizan M. Tariq; Goro Yeh; Avinash Singh; Sangjae Bae
>
> **摘要:** Reliable automated driving technology is challenged by various sources of uncertainties, in particular, behavioral uncertainties of traffic agents. It is common for traffic agents to have intentions that are unknown to others, leaving an automated driving car to reason over multiple possible behaviors. This paper formalizes a behavior planning scheme in the presence of multiple possible futures with corresponding probabilities. We present a maximum entropy formulation and show how, under certain assumptions, this allows delayed decision-making to improve safety. The general formulation is then turned into a model predictive control formulation, which is solved as a quadratic program or a set of quadratic programs. We discuss implementation details for improving computation and verify operation in simulation and on a mobile robot.
>
---
#### [replaced 005] Dynamic Obstacle Avoidance of Unmanned Surface Vehicles in Maritime Environments Using Gaussian Processes Based Motion Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.07664v2](http://arxiv.org/pdf/2412.07664v2)**

> **作者:** Jiawei Meng; Yuanchang Liu; Danail Stoyanov
>
> **备注:** 20 pages, 22 figures
>
> **摘要:** During recent years, unmanned surface vehicles are extensively utilised in a variety of maritime applications such as the exploration of unknown areas, autonomous transportation, offshore patrol and others. In such maritime applications, unmanned surface vehicles executing relevant missions that might collide with potential static obstacles such as islands and reefs and dynamic obstacles such as other moving unmanned surface vehicles. To successfully accomplish these missions, motion planning algorithms that can generate smooth and collision-free trajectories to avoid both these static and dynamic obstacles in an efficient manner are essential. In this article, we propose a novel motion planning algorithm named the Dynamic Gaussian process motion planner 2, which successfully extends the application scope of the Gaussian process motion planner 2 into complex and dynamic environments with both static and dynamic obstacles. First, we introduce an approach to generate safe areas for dynamic obstacles using modified multivariate Gaussian distributions. Second, we introduce an approach to integrate real-time status information of dynamic obstacles into the modified multivariate Gaussian distributions. The multivariate Gaussian distributions with real-time statuses of dynamic obstacles can be innovatively added into the optimisation process of factor graph to generate an optimised trajectory. We also develop a variant of the proposed algorithm that integrates the international regulations for preventing collisions at sea, enhancing its operational effectiveness in maritime environments. The proposed algorithms have been validated in a series of benchmark simulations and a dynamic obstacle avoidance mission in a high-fidelity maritime environment in the Robotic operating system to demonstrate the functionality and practicability.
>
---
#### [replaced 006] Active inference as a unified model of collision avoidance behavior in human drivers
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.02215v2](http://arxiv.org/pdf/2506.02215v2)**

> **作者:** Julian F. Schumann; Johan Engström; Leif Johnson; Matthew O'Kelly; Joao Messias; Jens Kober; Arkady Zgonnikov
>
> **摘要:** Collision avoidance -- involving a rapid threat detection and quick execution of the appropriate evasive maneuver -- is a critical aspect of driving. However, existing models of human collision avoidance behavior are fragmented, focusing on specific scenarios or only describing certain aspects of the avoidance behavior, such as response times. This paper addresses these gaps by proposing a novel computational cognitive model of human collision avoidance behavior based on active inference. Active inference provides a unified approach to modeling human behavior: the minimization of free energy. Building on prior active inference work, our model incorporates established cognitive mechanisms such as evidence accumulation to simulate human responses in two distinct collision avoidance scenarios: front-to-rear lead vehicle braking and lateral incursion by an oncoming vehicle. We demonstrate that our model explains a wide range of previous empirical findings on human collision avoidance behavior. Specifically, the model closely reproduces both aggregate results from meta-analyses previously reported in the literature and detailed, scenario-specific effects observed in a recent driving simulator study, including response timing, maneuver selection, and execution. Our results highlight the potential of active inference as a unified framework for understanding and modeling human behavior in complex real-life driving tasks.
>
---
#### [replaced 007] A Versatile Neural Network Configuration Space Planning and Control Strategy for Modular Soft Robot Arms
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.03483v2](http://arxiv.org/pdf/2410.03483v2)**

> **作者:** Zixi Chen; Qinghua Guan; Josie Hughes; Arianna Menciassi; Cesare Stefanini
>
> **备注:** 14 pages, 16 figures, 5 tables; accepted by IEEE T-Ro
>
> **摘要:** Modular soft robot arms (MSRAs) are composed of multiple modules connected in a sequence, and they can bend at different angles in various directions. This capability allows MSRAs to perform more intricate tasks than single-module robots. However, the modular structure also induces challenges in accurate planning and control. Nonlinearity and hysteresis complicate the physical model, while the modular structure and increased DOFs further lead to cumulative errors along the sequence. To address these challenges, we propose a versatile configuration space planning and control strategy for MSRAs, named S2C2A (State to Configuration to Action). Our approach formulates an optimization problem, S2C (State to Configuration planning), which integrates various loss functions and a forward model based on biLSTM to generate configuration trajectories based on target states. A configuration controller C2A (Configuration to Action control) based on biLSTM is implemented to follow the planned configuration trajectories, leveraging only inaccurate internal sensing feedback. We validate our strategy using a cable-driven MSRA, demonstrating its ability to perform diverse offline tasks such as position and orientation control and obstacle avoidance. Furthermore, our strategy endows MSRA with online interaction capability with targets and obstacles. Future work focuses on addressing MSRA challenges, such as more accurate physical models.
>
---
#### [replaced 008] Towards Autonomous Reinforcement Learning for Real-World Robotic Manipulation with Large Language Models
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04280v3](http://arxiv.org/pdf/2503.04280v3)**

> **作者:** Niccolò Turcato; Matteo Iovino; Aris Synodinos; Alberto Dalla Libera; Ruggero Carli; Pietro Falco
>
> **摘要:** Recent advancements in Large Language Models (LLMs) and Visual Language Models (VLMs) have significantly impacted robotics, enabling high-level semantic motion planning applications. Reinforcement Learning (RL), a complementary paradigm, enables agents to autonomously optimize complex behaviors through interaction and reward signals. However, designing effective reward functions for RL remains challenging, especially in real-world tasks where sparse rewards are insufficient and dense rewards require elaborate design. In this work, we propose Autonomous Reinforcement learning for Complex Human-Informed Environments (ARCHIE), an unsupervised pipeline leveraging GPT-4, a pre-trained LLM, to generate reward functions directly from natural language task descriptions. The rewards are used to train RL agents in simulated environments, where we formalize the reward generation process to enhance feasibility. Additionally, GPT-4 automates the coding of task success criteria, creating a fully automated, one-shot procedure for translating human-readable text into deployable robot skills. Our approach is validated through extensive simulated experiments on single-arm and bi-manual manipulation tasks using an ABB YuMi collaborative robot, highlighting its practicality and effectiveness. Tasks are demonstrated on the real robot setup.
>
---
#### [replaced 009] Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data
- **分类: cs.RO; cs.GR**

- **链接: [http://arxiv.org/pdf/2506.04120v2](http://arxiv.org/pdf/2506.04120v2)**

> **作者:** Ben Moran; Mauro Comi; Arunkumar Byravan; Steven Bohez; Tom Erez; Zhibin Li; Leonard Hasenclever
>
> **备注:** Updated version correcting inadvertent omission in author list
>
> **摘要:** Creating accurate, physical simulations directly from real-world robot motion holds great value for safe, scalable, and affordable robot learning, yet remains exceptionally challenging. Real robot data suffers from occlusions, noisy camera poses, dynamic scene elements, which hinder the creation of geometrically accurate and photorealistic digital twins of unseen objects. We introduce a novel real-to-sim framework tackling all these challenges at once. Our key insight is a hybrid scene representation merging the photorealistic rendering of 3D Gaussian Splatting with explicit object meshes suitable for physics simulation within a single representation. We propose an end-to-end optimization pipeline that leverages differentiable rendering and differentiable physics within MuJoCo to jointly refine all scene components - from object geometry and appearance to robot poses and physical parameters - directly from raw and imprecise robot trajectories. This unified optimization allows us to simultaneously achieve high-fidelity object mesh reconstruction, generate photorealistic novel views, and perform annotation-free robot pose calibration. We demonstrate the effectiveness of our approach both in simulation and on challenging real-world sequences using an ALOHA 2 bi-manual manipulator, enabling more practical and robust real-to-simulation pipelines.
>
---
#### [replaced 010] DaDu-Corki: Algorithm-Architecture Co-Design for Embodied AI-powered Robotic Manipulation
- **分类: cs.AR; cs.RO**

- **链接: [http://arxiv.org/pdf/2407.04292v5](http://arxiv.org/pdf/2407.04292v5)**

> **作者:** Yiyang Huang; Yuhui Hao; Bo Yu; Feng Yan; Yuxin Yang; Feng Min; Yinhe Han; Lin Ma; Shaoshan Liu; Qiang Liu; Yiming Gan
>
> **摘要:** Embodied AI robots have the potential to fundamentally improve the way human beings live and manufacture. Continued progress in the burgeoning field of using large language models to control robots depends critically on an efficient computing substrate, and this trend is strongly evident in manipulation tasks. In particular, today's computing systems for embodied AI robots for manipulation tasks are designed purely based on the interest of algorithm developers, where robot actions are divided into a discrete frame basis. Such an execution pipeline creates high latency and energy consumption. This paper proposes \textsc{Corki}\xspace, an algorithm-architecture co-design framework for real-time embodied AI-powered robotic manipulation applications. We aim to decouple LLM inference, robotic control, and data communication in the embodied AI robots' compute pipeline. Instead of predicting action for one single frame, \textsc{Corki}\xspace predicts the trajectory for the near future to reduce the frequency of LLM inference. The algorithm is coupled with a hardware that accelerates transforming trajectory into actual torque signals used to control robots and an execution pipeline that parallels data communication with computation. \textsc{Corki}\xspace largely reduces LLM inference frequency by up to $5.1\times$, resulting in up to $5.9\times$ speed up. The success rate improvement can be up to 13.9\%.
>
---
#### [replaced 011] PartInstruct: Part-level Instruction Following for Fine-grained Robot Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21652v2](http://arxiv.org/pdf/2505.21652v2)**

> **作者:** Yifan Yin; Zhengtao Han; Shivam Aarya; Jianxin Wang; Shuhang Xu; Jiawei Peng; Angtian Wang; Alan Yuille; Tianmin Shu
>
> **摘要:** Fine-grained robot manipulation, such as lifting and rotating a bottle to display the label on the cap, requires robust reasoning about object parts and their relationships with intended tasks. Despite recent advances in training general-purpose robot manipulation policies guided by language instructions, there is a notable lack of large-scale datasets for fine-grained manipulation tasks with part-level instructions and diverse 3D object instances annotated with part-level labels. In this work, we introduce PartInstruct, the first large-scale benchmark for training and evaluating fine-grained robot manipulation models using part-level instructions. PartInstruct comprises 513 object instances across 14 categories, each annotated with part-level information, and 1302 fine-grained manipulation tasks organized into 16 task classes. Our training set consists of over 10,000 expert demonstrations synthesized in a 3D simulator, where each demonstration is paired with a high-level task instruction, a chain of base part-based skill instructions, and ground-truth 3D information about the object and its parts. Additionally, we designed a comprehensive test suite to evaluate the generalizability of learned policies across new states, objects, and tasks. We evaluated several state-of-the-art robot manipulation approaches, including end-to-end vision-language policy learning and bi-level planning models for robot manipulation on our benchmark. The experimental results reveal that current models struggle to robustly ground part concepts and predict actions in 3D space, and face challenges when manipulating object parts in long-horizon tasks.
>
---
#### [replaced 012] SLAC: Simulation-Pretrained Latent Action Space for Whole-Body Real-World RL
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.04147v2](http://arxiv.org/pdf/2506.04147v2)**

> **作者:** Jiaheng Hu; Peter Stone; Roberto Martín-Martín
>
> **摘要:** Building capable household and industrial robots requires mastering the control of versatile, high-degree-of-freedom (DoF) systems such as mobile manipulators. While reinforcement learning (RL) holds promise for autonomously acquiring robot control policies, scaling it to high-DoF embodiments remains challenging. Direct RL in the real world demands both safe exploration and high sample efficiency, which are difficult to achieve in practice. Sim-to-real RL, on the other hand, is often brittle due to the reality gap. This paper introduces SLAC, a method that renders real-world RL feasible for complex embodiments by leveraging a low-fidelity simulator to pretrain a task-agnostic latent action space. SLAC trains this latent action space via a customized unsupervised skill discovery method designed to promote temporal abstraction, disentanglement, and safety, thereby facilitating efficient downstream learning. Once a latent action space is learned, SLAC uses it as the action interface for a novel off-policy RL algorithm to autonomously learn downstream tasks through real-world interactions. We evaluate SLAC against existing methods on a suite of bimanual mobile manipulation tasks, where it achieves state-of-the-art performance. Notably, SLAC learns contact-rich whole-body tasks in under an hour of real-world interactions, without relying on any demonstrations or hand-crafted behavior priors. More information, code, and videos at robo-rl.github.io
>
---
#### [replaced 013] A Third-Order Gaussian Process Trajectory Representation Framework with Closed-Form Kinematics for Continuous-Time Motion Estimation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.22931v4](http://arxiv.org/pdf/2410.22931v4)**

> **作者:** Thien-Minh Nguyen; Ziyu Cao; Kailai Li; William Talbot; Tongxing Jin; Shenghai Yuan; Timothy D. Barfoot; Lihua Xie
>
> **备注:** The source code has been released. All feedbacks are welcome
>
> **摘要:** In this paper, we propose a third-order, i.e., white-noise-on-jerk, Gaussian Process (GP) Trajectory Representation (TR) framework for continuous-time (CT) motion estimation (ME) tasks. Our framework features a unified trajectory representation that encapsulates the kinematic models of both $SO(3)\times\mathbb{R}^3$ and $SE(3)$ pose representations. This encapsulation strategy allows users to use the same implementation of measurement-based factors for either choice of pose representation, which facilitates experimentation and comparison to achieve the best model for the ME task. In addition, unique to our framework, we derive the kinematic models with the closed-form temporal derivatives of the local variable of $SO(3)$ and $SE(3)$, which so far has only been approximated based on the Taylor expansion in the literature. Our experiments show that these kinematic models can improve the estimation accuracy in high-speed scenarios. All analytical Jacobians of the interpolated states with respect to the support states of the trajectory representation, as well as the motion prior factors, are also provided for accelerated Gauss-Newton (GN) optimization. Our experiments demonstrate the efficacy and efficiency of the framework in various motion estimation tasks such as localization, calibration, and odometry, facilitating fast prototyping for ME researchers. We release the source code for the benefit of the community. Our project is available at https://github.com/brytsknguyen/gptr.
>
---
#### [replaced 014] A Hybrid Multi-Factor Network with Dynamic Sequence Modeling for Early Warning of Intraoperative Hypotension
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11064v4](http://arxiv.org/pdf/2409.11064v4)**

> **作者:** Mingyue Cheng; Jintao Zhang; Zhiding Liu; Chunli Liu
>
> **摘要:** Intraoperative hypotension (IOH) prediction using past physiological signals is crucial, as IOH may lead to inadequate organ perfusion and significantly elevate the risk of severe complications and mortality. However, current methods often rely on static modeling, overlooking the complex temporal dependencies and the inherently non-stationary nature of physiological signals. We propose a Hybrid Multi-Factor (HMF) network that formulates IOH prediction as a dynamic sequence forecasting task, explicitly capturing both temporal dependencies and physiological non-stationarity. We represent signal dynamics as multivariate time series and decompose them into trend and seasonal components, enabling separate modeling of long-term and periodic variations. Each component is encoded with a patch-based Transformer to balance computational efficiency and feature representation. To address distributional drift from evolving signals, we introduce a symmetric normalization mechanism. Experiments on both public and real-world clinical datasets show that HMF significantly outperforms competitive baselines. We hope HMF offers new insights into IOH prediction and ultimately promotes safer surgical care. Our code is available at https://github.com/Mingyue-Cheng/HMF.
>
---
#### [replaced 015] Modeling, control, and stiffness regulation of layer jamming-based continuum robots
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2309.04154v3](http://arxiv.org/pdf/2309.04154v3)**

> **作者:** Yeman Fan; Bowen Yi; Dikai Liu
>
> **摘要:** Continuum robots with variable compliance have gained significant attention due to their adaptability in unstructured environments. Among various stiffness modulation techniques, layer jamming (LJ) provides a simple yet effective approach for achieving tunable stiffness. However, most existing LJ-based continuum robot models rely on static or quasi-static approximations, lacking a rigorous control-oriented dynamical formulation. Consequently, they are unsuitable for real-time control tasks requiring simultaneous regulation of configuration and stiffness and fail to capture the full dynamic behavior of LJ-based continuum robots. To address this gap, this paper proposes a port-Hamiltonian formulation for LJ-based continuum robots, formally characterizing the two key phenomena -- shape locking and tunable stiffness -- within a unified energy-based framework. Based on this model, we develop a passivity-based control approach that enables decoupled regulation of stiffness and configuration with provable stability guarantees. We validate the proposed framework through comprehensive experiments on the OctRobot-I continuum robotic platform. The results demonstrate consistency between theoretical predictions and empirical data, highlighting the feasibility of our approach for real-world implementation.
>
---
#### [replaced 016] A Machine Learning Approach to Sensor Substitution from Tactile Sensing to Visual Perception for Non-Prehensile Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.09180v3](http://arxiv.org/pdf/2502.09180v3)**

> **作者:** Idil Ozdamar; Doganay Sirintuna; Arash Ajoudani
>
> **备注:** 10 pages, 6 figures, submitted to Robotics and Autonomous Systems, for associated video, see https://youtu.be/6yIRcfn2DsY
>
> **摘要:** Mobile manipulators are increasingly deployed in complex environments, requiring diverse sensors to perceive and interact with their surroundings. However, equipping every robot with every possible sensor is often impractical due to cost and physical constraints. A critical challenge arises when robots with differing sensor capabilities need to collaborate or perform similar tasks. For example, consider a scenario where a mobile manipulator equipped with high-resolution tactile skin is skilled at non-prehensile manipulation tasks like pushing. If this robot needs to be replaced or augmented by a robot lacking such tactile sensing, the learned manipulation policies become inapplicable. This paper addresses the problem of sensor substitution in non-prehensile manipulation. We propose a novel machine learning-based framework that enables a robot with a limited sensor set (e.g., LiDAR or RGB-D) to effectively perform tasks previously reliant on a richer sensor suite (e.g., tactile skin). Our approach learns a mapping between the available sensor data and the information provided by the substituted sensor, effectively synthesizing the missing sensory input. Specifically, we demonstrate the efficacy of our framework by training a model to substitute tactile skin data for the task of non-prehensile pushing using a mobile manipulator. We show that a manipulator equipped only with LiDAR or RGB-D can, after training, achieve comparable and sometimes even better pushing performance to a mobile base utilizing direct tactile feedback.
>
---
#### [replaced 017] Certified Human Trajectory Prediction
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2403.13778v2](http://arxiv.org/pdf/2403.13778v2)**

> **作者:** Mohammadhossein Bahari; Saeed Saadatnejad; Amirhossein Askari Farsangi; Seyed-Mohsen Moosavi-Dezfooli; Alexandre Alahi
>
> **备注:** CVPR 2025
>
> **摘要:** Predicting human trajectories is essential for the safe operation of autonomous vehicles, yet current data-driven models often lack robustness in case of noisy inputs such as adversarial examples or imperfect observations. Although some trajectory prediction methods have been developed to provide empirical robustness, these methods are heuristic and do not offer guaranteed robustness. In this work, we propose a certification approach tailored for trajectory prediction that provides guaranteed robustness. To this end, we address the unique challenges associated with trajectory prediction, such as unbounded outputs and multi-modality. To mitigate the inherent performance drop through certification, we propose a diffusion-based trajectory denoiser and integrate it into our method. Moreover, we introduce new certified performance metrics to reliably measure the trajectory prediction performance. Through comprehensive experiments, we demonstrate the accuracy and robustness of the certified predictors and highlight their advantages over the non-certified ones. The code is available online: https://s-attack.github.io/.
>
---
#### [replaced 018] Safe Navigation in Dynamic Environments using Density Functions
- **分类: cs.RO; math.DS; math.OC**

- **链接: [http://arxiv.org/pdf/2411.12206v2](http://arxiv.org/pdf/2411.12206v2)**

> **作者:** Sriram S. K. S Narayanan; Joseph Moyalan; Umesh Vaidya
>
> **摘要:** This work presents a density-based framework for safe navigation in dynamic environments characterized by time-varying obstacle sets and time-varying target regions. We propose an analytical construction of time-varying density functions that enables the synthesis of a feedback controller defined as the positive gradient of the resulting density field. The primary contribution of this paper is a rigorous convergence proof demonstrating almost-everywhere safe navigation under the proposed framework, specifically for systems governed by single-integrator dynamics. To the best of our knowledge, these are the first analytical guarantees of their kind for navigation in dynamic environments using density functions. We illustrate the applicability of the framework to systems with more complex dynamics, including multi-agent systems and robotic manipulators, using standard control design techniques such as backstepping and inverse dynamics. These results provide a foundation for extending density-based navigation methods to a broad class of robotic systems operating in time-varying environments.
>
---
#### [replaced 019] ASMA: An Adaptive Safety Margin Algorithm for Vision-Language Drone Navigation via Scene-Aware Control Barrier Functions
- **分类: cs.RO; cs.AI; cs.SY; eess.IV; eess.SY**

- **链接: [http://arxiv.org/pdf/2409.10283v3](http://arxiv.org/pdf/2409.10283v3)**

> **作者:** Sourav Sanyal; Kaushik Roy
>
> **摘要:** In the rapidly evolving field of vision-language navigation (VLN), ensuring safety for physical agents remains an open challenge. For a human-in-the-loop language-operated drone to navigate safely, it must understand natural language commands, perceive the environment, and simultaneously avoid hazards in real time. Control Barrier Functions (CBFs) are formal methods that enforce safe operating conditions. Model Predictive Control (MPC) is an optimization framework that plans a sequence of future actions over a prediction horizon, ensuring smooth trajectory tracking while obeying constraints. In this work, we consider a VLN-operated drone platform and enhance its safety by formulating a novel scene-aware CBF that leverages ego-centric observations from a camera which has both Red-Green-Blue as well as Depth (RGB-D) channels. A CBF-less baseline system uses a Vision-Language Encoder with cross-modal attention to convert commands into an ordered sequence of landmarks. An object detection model identifies and verifies these landmarks in the captured images to generate a planned path. To further enhance safety, an Adaptive Safety Margin Algorithm (ASMA) is proposed. ASMA tracks moving objects and performs scene-aware CBF evaluation on-the-fly, which serves as an additional constraint within the MPC framework. By continuously identifying potentially risky observations, the system performs prediction in real time about unsafe conditions and proactively adjusts its control actions to maintain safe navigation throughout the trajectory. Deployed on a Parrot Bebop2 quadrotor in the Gazebo environment using the Robot Operating System (ROS), ASMA achieves 64%-67% increase in success rates with only a slight increase (1.4%-5.8%) in trajectory lengths compared to the baseline CBF-less VLN.
>
---
#### [replaced 020] An Overview of the Burer-Monteiro Method for Certifiable Robot Perception
- **分类: cs.RO; cs.CV; cs.LG; 49, 68; I.4.0; I.5.0; J.2**

- **链接: [http://arxiv.org/pdf/2410.00117v2](http://arxiv.org/pdf/2410.00117v2)**

> **作者:** Alan Papalia; Yulun Tian; David M. Rosen; Jonathan P. How; John J. Leonard
>
> **备注:** Accepted to 2024 Robotics: Science and Systems (RSS) Safe Autonomy Workshop
>
> **摘要:** This paper presents an overview of the Burer-Monteiro method (BM), a technique that has been applied to solve robot perception problems to certifiable optimality in real-time. BM is often used to solve semidefinite programming relaxations, which can be used to perform global optimization for non-convex perception problems. Specifically, BM leverages the low-rank structure of typical semidefinite programs to dramatically reduce the computational cost of performing optimization. This paper discusses BM in certifiable perception, with three main objectives: (i) to consolidate information from the literature into a unified presentation, (ii) to elucidate the role of the linear independence constraint qualification (LICQ), a concept not yet well-covered in certifiable perception literature, and (iii) to share practical considerations that are discussed among practitioners but not thoroughly covered in the literature. Our general aim is to offer a practical primer for applying BM towards certifiable perception.
>
---
#### [replaced 021] Scene Exploration by Vision-Language Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.17641v2](http://arxiv.org/pdf/2409.17641v2)**

> **作者:** Venkatesh Sripada; Samuel Carter; Frank Guerin; Amir Ghalamzan
>
> **摘要:** Active perception enables robots to dynamically gather information by adjusting their viewpoints, a crucial capability for interacting with complex, partially observable environments. In this paper, we present AP-VLM, a novel framework that combines active perception with a Vision-Language Model (VLM) to guide robotic exploration and answer semantic queries. Using a 3D virtual grid overlaid on the scene and orientation adjustments, AP-VLM allows a robotic manipulator to intelligently select optimal viewpoints and orientations to resolve challenging tasks, such as identifying objects in occluded or inclined positions. We evaluate our system on two robotic platforms: a 7-DOF Franka Panda and a 6-DOF UR5, across various scenes with differing object configurations. Our results demonstrate that AP-VLM significantly outperforms passive perception methods and baseline models, including Toward Grounded Common Sense Reasoning (TGCSR), particularly in scenarios where fixed camera views are inadequate. The adaptability of AP-VLM in real-world settings shows promise for enhancing robotic systems' understanding of complex environments, bridging the gap between high-level semantic reasoning and low-level control.
>
---
#### [replaced 022] RT-GuIDE: Real-Time Gaussian splatting for Information-Driven Exploration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.18122v2](http://arxiv.org/pdf/2409.18122v2)**

> **作者:** Yuezhan Tao; Dexter Ong; Varun Murali; Igor Spasojevic; Pratik Chaudhari; Vijay Kumar
>
> **摘要:** We propose a framework for active mapping and exploration that leverages Gaussian splatting for constructing dense maps. Further, we develop a GPU-accelerated motion planning algorithm that can exploit the Gaussian map for real-time navigation. The Gaussian map constructed onboard the robot is optimized for both photometric and geometric quality while enabling real-time situational awareness for autonomy. We show through simulation experiments that our method yields comparable Peak Signal-to-Noise Ratio (PSNR) and similar reconstruction error to state-of-the-art approaches, while being orders of magnitude faster to compute. In real-world experiments, our algorithm achieves better map quality (at least 0.8dB higher PSNR and more than 16% higher geometric reconstruction accuracy) than maps constructed by a state-of-the-art method, enabling semantic segmentation using off-the-shelf open-set models. Experiment videos and more details can be found on our project page: https://tyuezhan.github.io/RT_GuIDE/
>
---
#### [replaced 023] AI-based Framework for Robust Model-Based Connector Mating in Robotic Wire Harness Installation
- **分类: cs.RO; cs.AI; cs.CE; cs.LG; 68T40; I.2; J.2**

- **链接: [http://arxiv.org/pdf/2503.09409v2](http://arxiv.org/pdf/2503.09409v2)**

> **作者:** Claudius Kienle; Benjamin Alt; Finn Schneider; Tobias Pertlwieser; Rainer Jäkel; Rania Rayyes
>
> **备注:** 6 pages, 6 figures, 4 tables, presented at the 2025 IEEE 21st International Conference on Automation Science and Engineering (CASE 2025)
>
> **摘要:** Despite the widespread adoption of industrial robots in automotive assembly, wire harness installation remains a largely manual process, as it requires precise and flexible manipulation. To address this challenge, we design a novel AI-based framework that automates cable connector mating by integrating force control with deep visuotactile learning. Our system optimizes search-and-insertion strategies using first-order optimization over a multimodal transformer architecture trained on visual, tactile, and proprioceptive data. Additionally, we design a novel automated data collection and optimization pipeline that minimizes the need for machine learning expertise. The framework optimizes robot programs that run natively on standard industrial controllers, permitting human experts to audit and certify them. Experimental validations on a center console assembly task demonstrate significant improvements in cycle times and robustness compared to conventional robot programming approaches. Videos are available under https://claudius-kienle.github.io/AppMuTT.
>
---
#### [replaced 024] Manual2Skill: Learning to Read Manuals and Acquire Robotic Skills for Furniture Assembly Using Vision-Language Models
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.10090v2](http://arxiv.org/pdf/2502.10090v2)**

> **作者:** Chenrui Tie; Shengxiang Sun; Jinxuan Zhu; Yiwei Liu; Jingxiang Guo; Yue Hu; Haonan Chen; Junting Chen; Ruihai Wu; Lin Shao
>
> **摘要:** Humans possess an extraordinary ability to understand and execute complex manipulation tasks by interpreting abstract instruction manuals. For robots, however, this capability remains a substantial challenge, as they cannot interpret abstract instructions and translate them into executable actions. In this paper, we present Manual2Skill, a novel framework that enables robots to perform complex assembly tasks guided by high-level manual instructions. Our approach leverages a Vision-Language Model (VLM) to extract structured information from instructional images and then uses this information to construct hierarchical assembly graphs. These graphs represent parts, subassemblies, and the relationships between them. To facilitate task execution, a pose estimation model predicts the relative 6D poses of components at each assembly step. At the same time, a motion planning module generates actionable sequences for real-world robotic implementation. We demonstrate the effectiveness of Manual2Skill by successfully assembling several real-world IKEA furniture items. This application highlights its ability to manage long-horizon manipulation tasks with both efficiency and precision, significantly enhancing the practicality of robot learning from instruction manuals. This work marks a step forward in advancing robotic systems capable of understanding and executing complex manipulation tasks in a manner akin to human capabilities.Project Page: https://owensun2004.github.io/Furniture-Assembly-Web/
>
---
#### [replaced 025] From Pixels to Predicates: Learning Symbolic World Models via Pretrained Vision-Language Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00296v2](http://arxiv.org/pdf/2501.00296v2)**

> **作者:** Ashay Athalye; Nishanth Kumar; Tom Silver; Yichao Liang; Tomás Lozano-Pérez; Leslie Pack Kaelbling
>
> **摘要:** Our aim is to learn to solve long-horizon decision-making problems in complex robotics domains given low-level skills and a handful of short-horizon demonstrations containing sequences of images. To this end, we focus on learning abstract symbolic world models that facilitate zero-shot generalization to novel goals via planning. A critical component of such models is the set of symbolic predicates that define properties of and relationships between objects. In this work, we leverage pretrained vision language models (VLMs) to propose a large set of visual predicates potentially relevant for decision-making, and to evaluate those predicates directly from camera images. At training time, we pass the proposed predicates and demonstrations into an optimization-based model-learning algorithm to obtain an abstract symbolic world model that is defined in terms of a compact subset of the proposed predicates. At test time, given a novel goal in a novel setting, we use the VLM to construct a symbolic description of the current world state, and then use a search-based planning algorithm to find a sequence of low-level skills that achieves the goal. We demonstrate empirically across experiments in both simulation and the real world that our method can generalize aggressively, applying its learned world model to solve problems with a wide variety of object types, arrangements, numbers of objects, and visual backgrounds, as well as novel goals and much longer horizons than those seen at training time.
>
---
#### [replaced 026] Multi-GraspLLM: A Multimodal LLM for Multi-Hand Semantic Guided Grasp Generation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.08468v3](http://arxiv.org/pdf/2412.08468v3)**

> **作者:** Haosheng Li; Weixin Mao; Weipeng Deng; Chenyu Meng; Haoqiang Fan; Tiancai Wang; Yoshie Osamu; Ping Tan; Hongan Wang; Xiaoming Deng
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Multi-hand semantic grasp generation aims to generate feasible and semantically appropriate grasp poses for different robotic hands based on natural language instructions. Although the task is highly valuable, due to the lack of multihand grasp datasets with fine-grained contact description between robotic hands and objects, it is still a long-standing difficult task. In this paper, we present Multi-GraspSet, the first large-scale multi-hand grasp dataset with automatically contact annotations. Based on Multi-GraspSet, we propose Multi-GraspLLM, a unified language-guided grasp generation framework, which leverages large language models (LLM) to handle variable-length sequences, generating grasp poses for diverse robotic hands in a single unified architecture. Multi-GraspLLM first aligns the encoded point cloud features and text features into a unified semantic space. It then generates grasp bin tokens that are subsequently converted into grasp pose for each robotic hand via hand-aware linear mapping. The experimental results demonstrate that our approach significantly outperforms existing methods in both real-world experiments and simulator. More information can be found on our project page https://multi-graspllm.github.io.
>
---
#### [replaced 027] MMScan: A Multi-Modal 3D Scene Dataset with Hierarchical Grounded Language Annotations
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.09401v2](http://arxiv.org/pdf/2406.09401v2)**

> **作者:** Ruiyuan Lyu; Jingli Lin; Tai Wang; Shuai Yang; Xiaohan Mao; Yilun Chen; Runsen Xu; Haifeng Huang; Chenming Zhu; Dahua Lin; Jiangmiao Pang
>
> **备注:** Follow-up of EmbodiedScan (camera-ready version). A multi-modal 3D dataset with the most-ever comprehensive language annotations for 3D-LLMs. Project page: https://tai-wang.github.io/mmscan/
>
> **摘要:** With the emergence of LLMs and their integration with other data modalities, multi-modal 3D perception attracts more attention due to its connectivity to the physical world and makes rapid progress. However, limited by existing datasets, previous works mainly focus on understanding object properties or inter-object spatial relationships in a 3D scene. To tackle this problem, this paper builds the first largest ever multi-modal 3D scene dataset and benchmark with hierarchical grounded language annotations, MMScan. It is constructed based on a top-down logic, from region to object level, from a single target to inter-target relationships, covering holistic aspects of spatial and attribute understanding. The overall pipeline incorporates powerful VLMs via carefully designed prompts to initialize the annotations efficiently and further involve humans' correction in the loop to ensure the annotations are natural, correct, and comprehensive. Built upon existing 3D scanning data, the resulting multi-modal 3D dataset encompasses 1.4M meta-annotated captions on 109k objects and 7.7k regions as well as over 3.04M diverse samples for 3D visual grounding and question-answering benchmarks. We evaluate representative baselines on our benchmarks, analyze their capabilities in different aspects, and showcase the key problems to be addressed in the future. Furthermore, we use this high-quality dataset to train state-of-the-art 3D visual grounding and LLMs and obtain remarkable performance improvement both on existing benchmarks and in-the-wild evaluation. Codes, datasets, and benchmarks will be available at https://github.com/OpenRobotLab/EmbodiedScan.
>
---
#### [replaced 028] SIS: Seam-Informed Strategy for T-shirt Unfolding
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.06990v3](http://arxiv.org/pdf/2409.06990v3)**

> **作者:** Xuzhao Huang; Akira Seino; Fuyuki Tokuda; Akinari Kobayashi; Dayuan Chen; Yasuhisa Hirata; Norman C. Tien; Kazuhiro Kosuge
>
> **备注:** 8 pages, 8 figures. To be published in IEEE Robotics and Automation Letters (RAL)
>
> **摘要:** Seams are information-rich components of garments. The presence of different types of seams and their combinations helps to select grasping points for garment handling. In this paper, we propose a new Seam-Informed Strategy (SIS) for finding actions for handling a garment, such as grasping and unfolding a T-shirt. Candidates for a pair of grasping points for a dual-arm manipulator system are extracted using the proposed Seam Feature Extraction Method (SFEM). A pair of grasping points for the robot system is selected by the proposed Decision Matrix Iteration Method (DMIM). The decision matrix is first computed by multiple human demonstrations and updated by the robot execution results to improve the grasping and unfolding performance of the robot. Note that the proposed scheme is trained on real data without relying on simulation. Experimental results demonstrate the effectiveness of the proposed strategy. The project video is available at https://github.com/lancexz/sis
>
---
#### [replaced 029] Cal or No Cal? -- Real-Time Miscalibration Detection of LiDAR and Camera Sensors
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.01040v2](http://arxiv.org/pdf/2504.01040v2)**

> **作者:** Ilir Tahiraj; Jeremialie Swadiryus; Felix Fent; Markus Lienkamp
>
> **摘要:** The goal of extrinsic calibration is the alignment of sensor data to ensure an accurate representation of the surroundings and enable sensor fusion applications. From a safety perspective, sensor calibration is a key enabler of autonomous driving. In the current state of the art, a trend from target-based offline calibration towards targetless online calibration can be observed. However, online calibration is subject to strict real-time and resource constraints which are not met by state-of-the-art methods. This is mainly due to the high number of parameters to estimate, the reliance on geometric features, or the dependence on specific vehicle maneuvers. To meet these requirements and ensure the vehicle's safety at any time, we propose a miscalibration detection framework that shifts the focus from the direct regression of calibration parameters to a binary classification of the calibration state, i.e., calibrated or miscalibrated. Therefore, we propose a contrastive learning approach that compares embedded features in a latent space to classify the calibration state of two different sensor modalities. Moreover, we provide a comprehensive analysis of the feature embeddings and challenging calibration errors that highlight the performance of our approach. As a result, our method outperforms the current state-of-the-art in terms of detection performance, inference time, and resource demand. The code is open source and available on https://github.com/TUMFTM/MiscalibrationDetection.
>
---
#### [replaced 030] ShapeICP: Iterative Category-level Object Pose and Shape Estimation from Depth
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2408.13147v2](http://arxiv.org/pdf/2408.13147v2)**

> **作者:** Yihao Zhang; Harpreet S. Sawhney; John J. Leonard
>
> **摘要:** Category-level object pose and shape estimation from a single depth image has recently drawn research attention due to its potential utility for tasks such as robotics manipulation. The task is particularly challenging because the three unknowns, object pose, object shape, and model-to-measurement correspondences, are compounded together, but only a single view of depth measurements is provided. Most of the prior work heavily relies on data-driven approaches to obtain solutions to at least one of the unknowns, and typically two, running with the risk of failing to generalize to unseen domains. The shape representations used in the prior work also mainly focus on point cloud and signed distance field (SDF). In stark contrast to the prior work, we approach the problem using an iterative estimation method that does not require learning from pose-annotated data. In addition, we adopt a novel mesh-based object active shape model that the previous literature has not explored. Our algorithm, ShapeICP, is based on the iterative closest point (ICP) algorithm but is equipped with additional features for the category-level pose and shape estimation task. Although not using pose-annotated data, ShapeICP surpasses many data-driven approaches that rely on pose data for training, opening up a new solution space for researchers to consider.
>
---
#### [replaced 031] SKiD-SLAM: Robust, Lightweight, and Distributed Multi-Robot LiDAR SLAM in Resource-Constrained Field Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.08230v2](http://arxiv.org/pdf/2505.08230v2)**

> **作者:** Hogyun Kim; Jiwon Choi; Juwon Kim; Geonmo Yang; Dongjin Cho; Hyungtae Lim; Younggun Cho
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** Distributed LiDAR SLAM is crucial for achieving efficient robot autonomy and improving the scalability of mapping. However, two issues need to be considered when applying it in field environments: one is resource limitation, and the other is inter/intra-robot association. The resource limitation issue arises when the data size exceeds the processing capacity of the network or memory, especially when utilizing communication systems or onboard computers in the field. The inter/intra-robot association issue occurs due to the narrow convergence region of ICP under large viewpoint differences, triggering many false positive loops and ultimately resulting in an inconsistent global map for multi-robot systems. To tackle these problems, we propose a distributed LiDAR SLAM framework designed for versatile field applications, called SKiD-SLAM. Extending our previous work that solely focused on lightweight place recognition and fast and robust global registration, we present a multi-robot mapping framework that focuses on robust and lightweight inter-robot loop closure in distributed LiDAR SLAM. Through various environmental experiments, we demonstrate that our method is more robust and lightweight compared to other state-of-the-art distributed SLAM approaches, overcoming resource limitation and inter/intra-robot association issues. Also, we validated the field applicability of our approach through mapping experiments in real-world planetary emulation terrain and cave environments, which are in-house datasets. Our code will be available at https://sparolab.github.io/research/skid_slam/.
>
---
#### [replaced 032] LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models
- **分类: cs.LG; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.15850v2](http://arxiv.org/pdf/2501.15850v2)**

> **作者:** Yuewen Mei; Tong Nie; Jian Sun; Ye Tian
>
> **备注:** Accepted as a regular paper at IEEE TITS 2025
>
> **摘要:** Ensuring and improving the safety of autonomous driving systems (ADS) is crucial for the deployment of highly automated vehicles, especially in safety-critical events. To address the rarity issue, adversarial scenario generation methods are developed, in which behaviors of traffic participants are manipulated to induce safety-critical events. However, existing methods still face two limitations. First, identification of the adversarial participant directly impacts the effectiveness of the generation. However, the complexity of real-world scenarios, with numerous participants and diverse behaviors, makes identification challenging. Second, the potential of generated safety-critical scenarios to continuously improve ADS performance remains underexplored. To address these issues, we propose LLM-attacker: a closed-loop adversarial scenario generation framework leveraging large language models (LLMs). Specifically, multiple LLM agents are designed and coordinated to identify optimal attackers. Then, the trajectories of the attackers are optimized to generate adversarial scenarios. These scenarios are iteratively refined based on the performance of ADS, forming a feedback loop to improve ADS. Experimental results show that LLM-attacker can create more dangerous scenarios than other methods, and the ADS trained with it achieves a collision rate half that of training with normal scenarios. This indicates the ability of LLM-attacker to test and enhance the safety and robustness of ADS. Video demonstrations are provided at: https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.
>
---
#### [replaced 033] A Skeleton-Based Topological Planner for Exploration in Complex Unknown Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.13664v3](http://arxiv.org/pdf/2412.13664v3)**

> **作者:** Haochen Niu; Xingwu Ji; Lantao Zhang; Fei Wen; Rendong Ying; Peilin Liu
>
> **备注:** 7 pages, 7 figures. Accepted to be presented at the ICRA 2025
>
> **摘要:** The capability of autonomous exploration in complex, unknown environments is important in many robotic applications. While recent research on autonomous exploration have achieved much progress, there are still limitations, e.g., existing methods relying on greedy heuristics or optimal path planning are often hindered by repetitive paths and high computational demands. To address such limitations, we propose a novel exploration framework that utilizes the global topology information of observed environment to improve exploration efficiency while reducing computational overhead. Specifically, global information is utilized based on a skeletal topological graph representation of the environment geometry. We first propose an incremental skeleton extraction method based on wavefront propagation, based on which we then design an approach to generate a lightweight topological graph that can effectively capture the environment's structural characteristics. Building upon this, we introduce a finite state machine that leverages the topological structure to efficiently plan coverage paths, which can substantially mitigate the back-and-forth maneuvers (BFMs) problem. Experimental results demonstrate the superiority of our method in comparison with state-of-the-art methods. The source code will be made publicly available at: https://github.com/Haochen-Niu/STGPlanner.
>
---
