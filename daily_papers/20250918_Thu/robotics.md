# 机器人 cs.RO

- **最新发布 70 篇**

- **更新 32 篇**

## 最新发布

#### [new 001] Pre-Manipulation Alignment Prediction with Parallel Deep State-Space and Transformer Models
- **分类: cs.RO**

- **简介: 该论文提出一种预测预操作图像与轨迹及语言指令对齐的模型，解决传统方法在操作后判断成功率导致效率低的问题。通过并行使用状态空间模型和Transformer捕捉轨迹时序特征，实验表明其优于现有方法。属于机器人操作预测任务。**

- **链接: [http://arxiv.org/pdf/2509.13839v1](http://arxiv.org/pdf/2509.13839v1)**

> **作者:** Motonari Kambara; Komei Sugiura
>
> **备注:** Published in Advanced Robotics
>
> **摘要:** In this work, we address the problem of predicting the future success of open-vocabulary object manipulation tasks. Conventional approaches typically determine success or failure after the action has been carried out. However, they make it difficult to prevent potential hazards and rely on failures to trigger replanning, thereby reducing the efficiency of object manipulation sequences. To overcome these challenges, we propose a model, which predicts the alignment between a pre-manipulation egocentric image with the planned trajectory and a given natural language instruction. We introduce a Multi-Level Trajectory Fusion module, which employs a state-of-the-art deep state-space model and a transformer encoder in parallel to capture multi-level time-series self-correlation within the end effector trajectory. Our experimental results indicate that the proposed method outperformed existing methods, including foundation models.
>
---
#### [new 002] SPAR: Scalable LLM-based PDDL Domain Generation for Aerial Robotics
- **分类: cs.RO**

- **简介: 该论文提出SPAR框架，利用大语言模型自动生成适用于无人机任务的PDDL规划域，解决手动设计耗时且易错的问题。构建了验证数据集，并通过多维度评估生成域的质量，推动空中机器人自动化规划的发展。**

- **链接: [http://arxiv.org/pdf/2509.13691v1](http://arxiv.org/pdf/2509.13691v1)**

> **作者:** Songhao Huang; Yuwei Wu; Guangyao Shi; Gaurav S. Sukhatme; Vijay Kumar
>
> **摘要:** We investigate the problem of automatic domain generation for the Planning Domain Definition Language (PDDL) using Large Language Models (LLMs), with a particular focus on unmanned aerial vehicle (UAV) tasks. Although PDDL is a widely adopted standard in robotic planning, manually designing domains for diverse applications such as surveillance, delivery, and inspection is labor-intensive and error-prone, which hinders adoption and real-world deployment. To address these challenges, we propose SPAR, a framework that leverages the generative capabilities of LLMs to automatically produce valid, diverse, and semantically accurate PDDL domains from natural language input. To this end, we first introduce a systematically formulated and validated UAV planning dataset, consisting of ground-truth PDDL domains and associated problems, each paired with detailed domain and action descriptions. Building on this dataset, we design a prompting framework that generates high-quality PDDL domains from language input. The generated domains are evaluated through syntax validation, executability, feasibility, and interpretability. Overall, this work demonstrates that LLMs can substantially accelerate the creation of complex planning domains, providing a reproducible dataset and evaluation pipeline that enables application experts without prior experience to leverage it for practical tasks and advance future research in aerial robotics and automated planning.
>
---
#### [new 003] TransforMARS: Fault-Tolerant Self-Reconfiguration for Arbitrarily Shaped Modular Aerial Robot Systems
- **分类: cs.RO; cs.MA**

- **简介: 论文提出TransforMARS框架，解决模块化空中机器人系统在多故障下稳定重构问题。通过识别可控组件并规划拆装序列，实现任意形状MARS的容错自重构，提升配置多样性和故障容忍度。**

- **链接: [http://arxiv.org/pdf/2509.14025v1](http://arxiv.org/pdf/2509.14025v1)**

> **作者:** Rui Huang; Zhiyu Gao; Siyu Tang; Jialin Zhang; Lei He; Ziqian Zhang; Lin Zhao
>
> **摘要:** Modular Aerial Robot Systems (MARS) consist of multiple drone modules that are physically bound together to form a single structure for flight. Exploiting structural redundancy, MARS can be reconfigured into different formations to mitigate unit or rotor failures and maintain stable flight. Prior work on MARS self-reconfiguration has solely focused on maximizing controllability margins to tolerate a single rotor or unit fault for rectangular-shaped MARS. We propose TransforMARS, a general fault-tolerant reconfiguration framework that transforms arbitrarily shaped MARS under multiple rotor and unit faults while ensuring continuous in-air stability. Specifically, we develop algorithms to first identify and construct minimum controllable assemblies containing faulty units. We then plan feasible disassembly-assembly sequences to transport MARS units or subassemblies to form target configuration. Our approach enables more flexible and practical feasible reconfiguration. We validate TransforMARS in challenging arbitrarily shaped MARS configurations, demonstrating substantial improvements over prior works in both the capacity of handling diverse configurations and the number of faults tolerated. The videos and source code of this work are available at the anonymous repository: https://anonymous.4open.science/r/TransforMARS-1030/
>
---
#### [new 004] FSR-VLN: Fast and Slow Reasoning for Vision-Language Navigation with Hierarchical Multi-modal Scene Graph
- **分类: cs.RO**

- **简介: 该论文提出FSR-VLN系统，解决视觉语言导航中的长距离空间推理与高延迟问题。通过结合分层多模态场景图与快慢推理机制，提升导航成功率并减少响应时间，实现高效、自然的机器人导航与交互。**

- **链接: [http://arxiv.org/pdf/2509.13733v1](http://arxiv.org/pdf/2509.13733v1)**

> **作者:** Xiaolin Zhou; Tingyang Xiao; Liu Liu; Yucheng Wang; Maiyue Chen; Xinrui Meng; Xinjie Wang; Wei Feng; Wei Sui; Zhizhong Su
>
> **备注:** 8 pages
>
> **摘要:** Visual-Language Navigation (VLN) is a fundamental challenge in robotic systems, with broad applications for the deployment of embodied agents in real-world environments. Despite recent advances, existing approaches are limited in long-range spatial reasoning, often exhibiting low success rates and high inference latency, particularly in long-range navigation tasks. To address these limitations, we propose FSR-VLN, a vision-language navigation system that combines a Hierarchical Multi-modal Scene Graph (HMSG) with Fast-to-Slow Navigation Reasoning (FSR). The HMSG provides a multi-modal map representation supporting progressive retrieval, from coarse room-level localization to fine-grained goal view and object identification. Building on HMSG, FSR first performs fast matching to efficiently select candidate rooms, views, and objects, then applies VLM-driven refinement for final goal selection. We evaluated FSR-VLN across four comprehensive indoor datasets collected by humanoid robots, utilizing 87 instructions that encompass a diverse range of object categories. FSR-VLN achieves state-of-the-art (SOTA) performance in all datasets, measured by the retrieval success rate (RSR), while reducing the response time by 82% compared to VLM-based methods on tour videos by activating slow reasoning only when fast intuition fails. Furthermore, we integrate FSR-VLN with speech interaction, planning, and control modules on a Unitree-G1 humanoid robot, enabling natural language interaction and real-time navigation.
>
---
#### [new 005] Using Visual Language Models to Control Bionic Hands: Assessment of Object Perception and Grasp Inference
- **分类: cs.RO**

- **简介: 该论文研究利用视觉语言模型（VLMs）提升义肢手的感知与抓取能力。任务是通过单张图像识别物体属性并推断抓取参数。工作包括构建统一基准，评估八种VLM在物体识别和抓取推理中的性能，分析其准确率、误差及效率，揭示其在义肢控制中的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2509.13572v1](http://arxiv.org/pdf/2509.13572v1)**

> **作者:** Ozan Karaali; Hossam Farag; Strahinja Dosen; Cedomir Stefanovic
>
> **备注:** ICAT 2025
>
> **摘要:** This study examines the potential of utilizing Vision Language Models (VLMs) to improve the perceptual capabilities of semi-autonomous prosthetic hands. We introduce a unified benchmark for end-to-end perception and grasp inference, evaluating a single VLM to perform tasks that traditionally require complex pipelines with separate modules for object detection, pose estimation, and grasp planning. To establish the feasibility and current limitations of this approach, we benchmark eight contemporary VLMs on their ability to perform a unified task essential for bionic grasping. From a single static image, they should (1) identify common objects and their key properties (name, shape, orientation, and dimensions), and (2) infer appropriate grasp parameters (grasp type, wrist rotation, hand aperture, and number of fingers). A corresponding prompt requesting a structured JSON output was employed with a dataset of 34 snapshots of common objects. Key performance metrics, including accuracy for categorical attributes (e.g., object name, shape) and errors in numerical estimates (e.g., dimensions, hand aperture), along with latency and cost, were analyzed. The results demonstrated that most models exhibited high performance in object identification and shape recognition, while accuracy in estimating dimensions and inferring optimal grasp parameters, particularly hand rotation and aperture, varied more significantly. This work highlights the current capabilities and limitations of VLMs as advanced perceptual modules for semi-autonomous control of bionic limbs, demonstrating their potential for effective prosthetic applications.
>
---
#### [new 006] Language Conditioning Improves Accuracy of Aircraft Goal Prediction in Untowered Airspace
- **分类: cs.RO**

- **简介: 该论文提出一种融合语言理解和空间推理的多模态框架，用于提升无人飞机在无塔台空域中对其他飞机目标位置的预测精度。通过语音识别和大语言模型解析飞行员通话，结合轨迹信息进行目标预测，显著提高预测准确性。属于目标预测任务，解决自主飞行安全问题。**

- **链接: [http://arxiv.org/pdf/2509.14063v1](http://arxiv.org/pdf/2509.14063v1)**

> **作者:** Sundhar Vinodh Sangeetha; Chih-Yuan Chiu; Sarah H. Q. Li; Shreyas Kousik
>
> **备注:** The last two authors advised equally. Submitted to the 2026 IEEE International Conference on Robotics and Automation. 8 pages, 6 figures
>
> **摘要:** Autonomous aircraft must safely operate in untowered airspace, where coordination relies on voice-based communication among human pilots. Safe operation requires an aircraft to predict the intent, and corresponding goal location, of other aircraft. This paper introduces a multimodal framework for aircraft goal prediction that integrates natural language understanding with spatial reasoning to improve autonomous decision-making in such environments. We leverage automatic speech recognition and large language models to transcribe and interpret pilot radio calls, identify aircraft, and extract discrete intent labels. These intent labels are fused with observed trajectories to condition a temporal convolutional network and Gaussian mixture model for probabilistic goal prediction. Our method significantly reduces goal prediction error compared to baselines that rely solely on motion history, demonstrating that language-conditioned prediction increases prediction accuracy. Experiments on a real-world dataset from an untowered airport validate the approach and highlight its potential to enable socially aware, language-conditioned robotic motion planning.
>
---
#### [new 007] Behavior Foundation Model for Humanoid Robots
- **分类: cs.RO**

- **简介: 论文提出行为基础模型（BFM），用于解决人形机器人全身控制任务中任务特定、泛化能力差的问题。通过预训练生成模型，实现跨任务行为的灵活控制与快速适应。**

- **链接: [http://arxiv.org/pdf/2509.13780v1](http://arxiv.org/pdf/2509.13780v1)**

> **作者:** Weishuai Zeng; Shunlin Lu; Kangning Yin; Xiaojie Niu; Minyue Dai; Jingbo Wang; Jiangmiao Pang
>
> **摘要:** Whole-body control (WBC) of humanoid robots has witnessed remarkable progress in skill versatility, enabling a wide range of applications such as locomotion, teleoperation, and motion tracking. Despite these achievements, existing WBC frameworks remain largely task-specific, relying heavily on labor-intensive reward engineering and demonstrating limited generalization across tasks and skills. These limitations hinder their response to arbitrary control modes and restrict their deployment in complex, real-world scenarios. To address these challenges, we revisit existing WBC systems and identify a shared objective across diverse tasks: the generation of appropriate behaviors that guide the robot toward desired goal states. Building on this insight, we propose the Behavior Foundation Model (BFM), a generative model pretrained on large-scale behavioral datasets to capture broad, reusable behavioral knowledge for humanoid robots. BFM integrates a masked online distillation framework with a Conditional Variational Autoencoder (CVAE) to model behavioral distributions, thereby enabling flexible operation across diverse control modes and efficient acquisition of novel behaviors without retraining from scratch. Extensive experiments in both simulation and on a physical humanoid platform demonstrate that BFM generalizes robustly across diverse WBC tasks while rapidly adapting to new behaviors. These results establish BFM as a promising step toward a foundation model for general-purpose humanoid control.
>
---
#### [new 008] Multi-robot Multi-source Localization in Complex Flows with Physics-Preserving Environment Models
- **分类: cs.RO; cs.LG**

- **简介: 论文提出一种多机器人分布式传感框架，用于复杂流场中的源定位。通过每个机器人携带物理保真模型，采用信息驱动采样策略，提升定位精度与效率。属于多机器人源定位任务，解决复杂流场中传感器读数稀疏、环境建模困难的问题。**

- **链接: [http://arxiv.org/pdf/2509.14228v1](http://arxiv.org/pdf/2509.14228v1)**

> **作者:** Benjamin Shaffer; Victoria Edwards; Brooks Kinch; Nathaniel Trask; M. Ani Hsieh
>
> **摘要:** Source localization in a complex flow poses a significant challenge for multi-robot teams tasked with localizing the source of chemical leaks or tracking the dispersion of an oil spill. The flow dynamics can be time-varying and chaotic, resulting in sporadic and intermittent sensor readings, and complex environmental geometries further complicate a team's ability to model and predict the dispersion. To accurately account for the physical processes that drive the dispersion dynamics, robots must have access to computationally intensive numerical models, which can be difficult when onboard computation is limited. We present a distributed mobile sensing framework for source localization in which each robot carries a machine-learned, finite element model of its environment to guide information-based sampling. The models are used to evaluate an approximate mutual information criterion to drive an infotaxis control strategy, which selects sensing regions that are expected to maximize informativeness for the source localization objective. Our approach achieves faster error reduction compared to baseline sensing strategies and results in more accurate source localization compared to baseline machine learning approaches.
>
---
#### [new 009] ASTREA: Introducing Agentic Intelligence for Orbital Thermal Autonomy
- **分类: cs.RO; cs.AI; cs.LG; cs.MA; cs.SY; eess.SY**

- **简介: 论文提出ASTREA系统，首次在飞行成熟硬件上实现基于代理智能的自主航天器热控。通过结合LLM与强化学习控制器，提升热稳定性，但轨道验证显示存在推理延迟问题，揭示了该技术在实际太空环境中的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2509.13380v1](http://arxiv.org/pdf/2509.13380v1)**

> **作者:** Alejandro D. Mousist
>
> **备注:** This preprint presents ASTREA, a multi-agent architecture combining LLM-guided semantic modulation with reinforcement learning for autonomous satellite operations. The system is validated in hardware orbital environments
>
> **摘要:** This paper presents ASTREA, the first agentic system deployed on flight-heritage hardware (TRL 9) for autonomous spacecraft operations. Using thermal control as a representative use case, we integrate a resource-constrained Large Language Model (LLM) agent with a reinforcement learning controller in an asynchronous architecture tailored for space-qualified platforms. Ground experiments show that LLM-guided supervision improves thermal stability and reduces violations, confirming the feasibility of combining semantic reasoning with adaptive control under hardware constraints. However, on-orbit validation aboard the International Space Station (ISS) reveals performance degradation caused by inference latency mismatched with the rapid thermal cycles characteristic of Low Earth Orbit (LEO) satellites. These results highlight both the opportunities and current limitations of agentic LLM-based systems in real flight environments, providing practical design guidelines for future space autonomy.
>
---
#### [new 010] SeqVLA: Sequential Task Execution for Long-Horizon Manipulation with Completion-Aware Vision-Language-Action Model
- **分类: cs.RO; 68T40**

- **简介: 论文提出SeqVLA模型，解决长时序机器人操作中子任务完成检测不足的问题。通过添加轻量检测头，实现动作生成与子任务切换的联合优化，提升多阶段任务的成功率。属于视觉-语言-动作模型在机器人操作中的应用任务。**

- **链接: [http://arxiv.org/pdf/2509.14138v1](http://arxiv.org/pdf/2509.14138v1)**

> **作者:** Ran Yang; Zijian An; Lifeng ZHou; Yiming Feng
>
> **备注:** 8 pages, 9 figures, 1 table
>
> **摘要:** Long-horizon robotic manipulation tasks require executing multiple interdependent subtasks in strict sequence, where errors in detecting subtask completion can cascade into downstream failures. Existing Vision-Language-Action (VLA) models such as $\pi_0$ excel at continuous low-level control but lack an internal signal for identifying when a subtask has finished, making them brittle in sequential settings. We propose SeqVLA, a completion-aware extension of $\pi_0$ that augments the base architecture with a lightweight detection head perceiving whether the current subtask is complete. This dual-head design enables SeqVLA not only to generate manipulation actions but also to autonomously trigger transitions between subtasks. We investigate four finetuning strategies that vary in how the action and detection heads are optimized (joint vs. sequential finetuning) and how pretrained knowledge is preserved (full finetuning vs. frozen backbone). Experiments are performed on two multi-stage tasks: salad packing with seven distinct subtasks and candy packing with four distinct subtasks. Results show that SeqVLA significantly outperforms the baseline $\pi_0$ and other strong baselines in overall success rate. In particular, joint finetuning with an unfrozen backbone yields the most decisive and statistically reliable completion predictions, eliminating sequence-related failures and enabling robust long-horizon execution. Our results highlight the importance of coupling action generation with subtask-aware detection for scalable sequential manipulation.
>
---
#### [new 011] Embracing Bulky Objects with Humanoid Robots: Whole-Body Manipulation with Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出一种基于强化学习的全身操控框架，用于人形机器人拥抱大体积物体。通过结合人类运动先验与神经符号距离场，实现稳定、协调的全身接触控制，提升负载能力和任务适应性，解决传统末端抓取在稳定性与负载上的不足。**

- **链接: [http://arxiv.org/pdf/2509.13534v1](http://arxiv.org/pdf/2509.13534v1)**

> **作者:** Chunxin Zheng; Kai Chen; Zhihai Bi; Yulin Li; Liang Pan; Jinni Zhou; Haoang Li; Jun Ma
>
> **摘要:** Whole-body manipulation (WBM) for humanoid robots presents a promising approach for executing embracing tasks involving bulky objects, where traditional grasping relying on end-effectors only remains limited in such scenarios due to inherent stability and payload constraints. This paper introduces a reinforcement learning framework that integrates a pre-trained human motion prior with a neural signed distance field (NSDF) representation to achieve robust whole-body embracing. Our method leverages a teacher-student architecture to distill large-scale human motion data, generating kinematically natural and physically feasible whole-body motion patterns. This facilitates coordinated control across the arms and torso, enabling stable multi-contact interactions that enhance the robustness in manipulation and also the load capacity. The embedded NSDF further provides accurate and continuous geometric perception, improving contact awareness throughout long-horizon tasks. We thoroughly evaluate the approach through comprehensive simulations and real-world experiments. The results demonstrate improved adaptability to diverse shapes and sizes of objects and also successful sim-to-real transfer. These indicate that the proposed framework offers an effective and practical solution for multi-contact and long-horizon WBM tasks of humanoid robots.
>
---
#### [new 012] Flexible and Foldable: Workspace Analysis and Object Manipulation Using a Soft, Interconnected, Origami-Inspired Actuator Array
- **分类: cs.RO**

- **简介: 论文提出一种基于折纸灵感的柔性分布式机械系统，通过降低执行器密度提升操作范围与灵活性，解决传统系统复杂度高、适应性差的问题，实现更高效、低成本的对象操控。**

- **链接: [http://arxiv.org/pdf/2509.13998v1](http://arxiv.org/pdf/2509.13998v1)**

> **作者:** Bailey Dacre; Rodrigo Moreno; Serhat Demirtas; Ziqiao Wang; Yuhao Jiang; Jamie Paik; Kasper Stoy; Andrés Faíña
>
> **摘要:** Object manipulation is a fundamental challenge in robotics, where systems must balance trade-offs among manipulation capabilities, system complexity, and throughput. Distributed manipulator systems (DMS) use the coordinated motion of actuator arrays to perform complex object manipulation tasks, seeing widespread exploration within the literature and in industry. However, existing DMS designs typically rely on high actuator densities and impose constraints on object-to-actuator scale ratios, limiting their adaptability. We present a novel DMS design utilizing an array of 3-DoF, origami-inspired robotic tiles interconnected by a compliant surface layer. Unlike conventional DMS, our approach enables manipulation not only at the actuator end effectors but also across a flexible surface connecting all actuators; creating a continuous, controllable manipulation surface. We analyse the combined workspace of such a system, derive simple motion primitives, and demonstrate its capabilities to translate simple geometric objects across an array of tiles. By leveraging the inter-tile connective material, our approach significantly reduces actuator density, increasing the area over which an object can be manipulated by x1.84 without an increase in the number of actuators. This design offers a lower cost and complexity alternative to traditional high-density arrays, and introduces new opportunities for manipulation strategies that leverage the flexibility of the interconnected surface.
>
---
#### [new 013] Cooperative Target Detection with AUVs: A Dual-Timescale Hierarchical MARDL Approach
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 论文提出一种双时标分层多智能体强化学习框架，用于水下无人潜航器（AUVs）的协同目标检测任务。旨在解决在敌对环境下如何实现高效协作与隐蔽操作的问题，通过高低层次协同控制提升任务效率与隐蔽性。**

- **链接: [http://arxiv.org/pdf/2509.13381v1](http://arxiv.org/pdf/2509.13381v1)**

> **作者:** Zhang Xueyao; Yang Bo; Yu Zhiwen; Cao Xuelin; George C. Alexandropoulos; Merouane Debbah; Chau Yuen
>
> **备注:** 6 pages
>
> **摘要:** Autonomous Underwater Vehicles (AUVs) have shown great potential for cooperative detection and reconnaissance. However, collaborative AUV communications introduce risks of exposure. In adversarial environments, achieving efficient collaboration while ensuring covert operations becomes a key challenge for underwater cooperative missions. In this paper, we propose a novel dual time-scale Hierarchical Multi-Agent Proximal Policy Optimization (H-MAPPO) framework. The high-level component determines the individuals participating in the task based on a central AUV, while the low-level component reduces exposure probabilities through power and trajectory control by the participating AUVs. Simulation results show that the proposed framework achieves rapid convergence, outperforms benchmark algorithms in terms of performance, and maximizes long-term cooperative efficiency while ensuring covert operations.
>
---
#### [new 014] TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出TreeIRL，结合MCTS与IRL用于自动驾驶路径规划。旨在提升城市驾驶安全性与人类行为相似性。通过模拟与真实道路测试，验证其在安全、舒适与效率上的优越性能。**

- **链接: [http://arxiv.org/pdf/2509.13579v1](http://arxiv.org/pdf/2509.13579v1)**

> **作者:** Momchil S. Tomov; Sang Uk Lee; Hansford Hendrago; Jinwook Huh; Teawon Han; Forbes Howington; Rafael da Silva; Gianmarco Bernasconi; Marc Heim; Samuel Findler; Xiaonan Ji; Alexander Boule; Michael Napoli; Kuo Chen; Jesse Miller; Boaz Floor; Yunqing Hu
>
> **摘要:** We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving.
>
---
#### [new 015] Object Pose Estimation through Dexterous Touch
- **分类: cs.RO; cs.CV**

- **简介: 论文提出一种基于触觉的物体位姿估计方法，通过双机械手协同操作与强化学习探索物体表面，利用触觉数据迭代优化物体形状与位姿。属于机器人触觉感知任务，解决视觉受限场景下物体位姿估计问题。**

- **链接: [http://arxiv.org/pdf/2509.13591v1](http://arxiv.org/pdf/2509.13591v1)**

> **作者:** Amir-Hossein Shahidzadeh; Jiyue Zhu; Kezhou Chen; Sha Yi; Cornelia Fermüller; Yiannis Aloimonos; Xiaolong Wang
>
> **摘要:** Robust object pose estimation is essential for manipulation and interaction tasks in robotics, particularly in scenarios where visual data is limited or sensitive to lighting, occlusions, and appearances. Tactile sensors often offer limited and local contact information, making it challenging to reconstruct the pose from partial data. Our approach uses sensorimotor exploration to actively control a robot hand to interact with the object. We train with Reinforcement Learning (RL) to explore and collect tactile data. The collected 3D point clouds are used to iteratively refine the object's shape and pose. In our setup, one hand holds the object steady while the other performs active exploration. We show that our method can actively explore an object's surface to identify critical pose features without prior knowledge of the object's geometry. Supplementary material and more demonstrations will be provided at https://amirshahid.github.io/BimanualTactilePose .
>
---
#### [new 016] Real World Robotic Exploration using Deep Neural Networks Trained in Photorealistic Reconstructed Environments
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种基于深度神经网络的机器人定位方法，通过改进损失函数提升定位精度，并利用实景重建数据训练模型，实现室内场景下的高精度导航。属于机器人视觉定位任务，解决真实环境中感知歧义和定位误差问题。**

- **链接: [http://arxiv.org/pdf/2509.13342v1](http://arxiv.org/pdf/2509.13342v1)**

> **作者:** Isaac Ronald Ward
>
> **备注:** This report is submitted as partial fulfilment of the requirements for the Honours Programme of the Department of Computer Science and Software Engineering, The University of Western Australia, 2019
>
> **摘要:** In this work, an existing deep neural network approach for determining a robot's pose from visual information (RGB images) is modified, improving its localization performance without impacting its ease of training. Explicitly, the network's loss function is extended in a manner which intuitively combines the positional and rotational error in order to increase robustness to perceptual aliasing. An improvement in the localization accuracy for indoor scenes is observed: with decreases of up to 9.64% and 2.99% in the median positional and rotational error respectively, when compared to the unmodified network. Additionally, photogrammetry data is used to produce a pose-labelled dataset which allows the above model to be trained on a local environment, resulting in localization accuracies of 0.11m & 0.89 degrees. This trained model forms the basis of a navigation algorithm, which is tested in real-time on a TurtleBot (a wheeled robotic device). As such, this work introduces a full pipeline for creating a robust navigational algorithm for any given real world indoor scene; the only requirement being a collection of images from the scene, which can be captured in as little as 330 seconds of
>
---
#### [new 017] The Influence of Facial Features on the Perceived Trustworthiness of a Social Robot
- **分类: cs.RO**

- **简介: 该论文研究社交机器人面部特征对其可信度的影响，属于人机交互设计任务。通过改变Furhat机器人的面部投影，验证了眼睛形状和大小对可信度感知的关键作用，为优化人机交互设计提供依据。**

- **链接: [http://arxiv.org/pdf/2509.13948v1](http://arxiv.org/pdf/2509.13948v1)**

> **作者:** Benedict Barrow; Roger K. Moore
>
> **备注:** In proceedings of TRUST 2025 (arXiv:2509.11402), a workshop at IEEE RO-MAN 2025: https://ro-man2025.org/
>
> **摘要:** Trust and the perception of trustworthiness play an important role in decision-making and our behaviour towards others, and this is true not only of human-human interactions but also of human-robot interactions. While significant advances have been made in recent years in the field of social robotics, there is still some way to go before we fully understand the factors that influence human trust in robots. This paper presents the results of a study into the first impressions created by a social robot's facial features, based on the hypothesis that a `babyface' engenders trust. By manipulating the back-projected face of a Furhat robot, the study confirms that eye shape and size have a significant impact on the perception of trustworthiness. The work thus contributes to an understanding of the design choices that need to be made when developing social robots so as to optimise the effectiveness of human-robot interaction.
>
---
#### [new 018] Energy Efficient Multi Robot Package Delivery under Capacity-Constraints via Voronoi-Constrained Networks
- **分类: cs.RO; cs.MA**

- **简介: 论文提出VCST-RCP框架，解决多机器人在容量限制下的高效配送问题。通过构建Voronoi约束网络与Steiner树优化，设计稀疏中继路径及机器人调度方案，提升能量效率，实验显示比传统方法最高提升34%。属于多机器人路径规划与物流调度任务。**

- **链接: [http://arxiv.org/pdf/2509.14127v1](http://arxiv.org/pdf/2509.14127v1)**

> **作者:** Alkesh K. Srivastava; Jared Michael Levin; Philip Dames
>
> **摘要:** We consider the problem of delivering multiple packages from a single pickup depot to distinct goal locations using a homogeneous fleet of robots with limited carrying capacity. We propose VCST-RCP, a Voronoi-Constrained Steiner Tree Relay Coordination Planning framework that constructs sparse relay trunks using Steiner tree optimization and then synthesizes robot-level pickup, relay, and delivery schedules. This framework reframes relays from incidental byproducts into central elements of coordination, offering a contrast with traditional delivery methods that rely on direct source-to-destination transport. Extensive experiments show consistent improvements of up to 34% compared to conventional baselines, underscoring the benefits of incorporating relays into the delivery process. These improvements translate directly to enhanced energy efficiency in multi-robot delivery under capacity constraints, providing a scalable framework for real-world logistics.
>
---
#### [new 019] \textsc{Gen2Real}: Towards Demo-Free Dexterous Manipulation by Harnessing Generated Video
- **分类: cs.RO**

- **简介: 该论文提出Gen2Real方法，解决机器人灵巧操作中依赖人类演示的问题。通过生成视频替代真实演示，结合轨迹优化与策略学习，实现从生成视频到真实机器人抓取任务的技能迁移，提升灵活性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.14178v1](http://arxiv.org/pdf/2509.14178v1)**

> **作者:** Kai Ye; Yuhang Wu; Shuyuan Hu; Junliang Li; Meng Liu; Yongquan Chen; Rui Huang
>
> **摘要:** Dexterous manipulation remains a challenging robotics problem, largely due to the difficulty of collecting extensive human demonstrations for learning. In this paper, we introduce \textsc{Gen2Real}, which replaces costly human demos with one generated video and drives robot skill from it: it combines demonstration generation that leverages video generation with pose and depth estimation to yield hand-object trajectories, trajectory optimization that uses Physics-aware Interaction Optimization Model (PIOM) to impose physics consistency, and demonstration learning that retargets human motions to a robot hand and stabilizes control with an anchor-based residual Proximal Policy Optimization (PPO) policy. Using only generated videos, the learned policy achieves a 77.3\% success rate on grasping tasks in simulation and demonstrates coherent executions on a real robot. We also conduct ablation studies to validate the contribution of each component and demonstrate the ability to directly specify tasks using natural language, highlighting the flexibility and robustness of \textsc{Gen2Real} in generalizing grasping skills from imagined videos to real-world execution.
>
---
#### [new 020] HGACNet: Hierarchical Graph Attention Network for Cross-Modal Point Cloud Completion
- **分类: cs.RO**

- **简介: 该论文提出HGACNet，用于解决点云补全任务，通过融合单视角RGB图像与点云的层次化几何特征，提升补全精度。采用HGA编码器与MSCF模块增强跨模态交互，并引入对比损失优化模态对齐，实验证明其在多个数据集上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.13692v1](http://arxiv.org/pdf/2509.13692v1)**

> **作者:** Yadan Zeng; Jiadong Zhou; Xiaohan Li; I-Ming Chen
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Point cloud completion is essential for robotic perception, object reconstruction and supporting downstream tasks like grasp planning, obstacle avoidance, and manipulation. However, incomplete geometry caused by self-occlusion and sensor limitations can significantly degrade downstream reasoning and interaction. To address these challenges, we propose HGACNet, a novel framework that reconstructs complete point clouds of individual objects by hierarchically encoding 3D geometric features and fusing them with image-guided priors from a single-view RGB image. At the core of our approach, the Hierarchical Graph Attention (HGA) encoder adaptively selects critical local points through graph attention-based downsampling and progressively refines hierarchical geometric features to better capture structural continuity and spatial relationships. To strengthen cross-modal interaction, we further design a Multi-Scale Cross-Modal Fusion (MSCF) module that performs attention-based feature alignment between hierarchical geometric features and structured visual representations, enabling fine-grained semantic guidance for completion. In addition, we proposed the contrastive loss (C-Loss) to explicitly align the feature distributions across modalities, improving completion fidelity under modality discrepancy. Finally, extensive experiments conducted on both the ShapeNet-ViPC benchmark and the YCB-Complete dataset confirm the effectiveness of HGACNet, demonstrating state-of-the-art performance as well as strong applicability in real-world robotic manipulation tasks.
>
---
#### [new 021] How Fly Neural Perception Mechanisms Enhance Visuomotor Control of Micro Robots
- **分类: cs.RO; cs.NE**

- **简介: 论文提出一种受果蝇神经机制启发的视觉运动控制策略，用于微机器人避障。通过模拟LPLC2神经元，设计轻量化模型，在Colias机器人上实现高效碰撞检测与规避，提升其敏捷性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.13827v1](http://arxiv.org/pdf/2509.13827v1)**

> **作者:** Renyuan Liu; Haoting Zhou; Chuankai Fang; Qinbing Fu
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Anyone who has tried to swat a fly has likely been frustrated by its remarkable agility.This ability stems from its visual neural perception system, particularly the collision-selective neurons within its small brain.For autonomous robots operating in complex and unfamiliar environments, achieving similar agility is highly desirable but often constrained by the trade-off between computational cost and performance.In this context, insect-inspired intelligence offers a parsimonious route to low-power, computationally efficient frameworks.In this paper, we propose an attention-driven visuomotor control strategy inspired by a specific class of fly visual projection neurons-the lobula plate/lobula column type-2 (LPLC2)-and their associated escape behaviors.To our knowledge, this represents the first embodiment of an LPLC2 neural model in the embedded vision of a physical mobile robot, enabling collision perception and reactive evasion.The model was simplified and optimized at 70KB in memory to suit the computational constraints of a vision-based micro robot, the Colias, while preserving key neural perception mechanisms.We further incorporated multi-attention mechanisms to emulate the distributed nature of LPLC2 responses, allowing the robot to detect and react to approaching targets both rapidly and selectively.We systematically evaluated the proposed method against a state-of-the-art locust-inspired collision detection model.Results showed that the fly-inspired visuomotor model achieved comparable robustness, at success rate of 96.1% in collision detection while producing more adaptive and elegant evasive maneuvers.Beyond demonstrating an effective collision-avoidance strategy, this work highlights the potential of fly-inspired neural models for advancing research into collective behaviors in insect intelligence.
>
---
#### [new 022] GeoAware-VLA: Implicit Geometry Aware Vision-Language-Action Model
- **分类: cs.RO**

- **简介: 该论文提出GeoAware-VLA模型，解决VLA模型在新视角下泛化能力差的问题。通过引入几何先验，提升视角不变性，在模拟和真实机器人任务中均取得显著性能提升。属于视觉-语言-动作任务。**

- **链接: [http://arxiv.org/pdf/2509.14117v1](http://arxiv.org/pdf/2509.14117v1)**

> **作者:** Ali Abouzeid; Malak Mansour; Zezhou Sun; Dezhen Song
>
> **备注:** Under Review
>
> **摘要:** Vision-Language-Action (VLA) models often fail to generalize to novel camera viewpoints, a limitation stemming from their difficulty in inferring robust 3D geometry from 2D images. We introduce GeoAware-VLA, a simple yet effective approach that enhances viewpoint invariance by integrating strong geometric priors into the vision backbone. Instead of training a visual encoder or relying on explicit 3D data, we leverage a frozen, pretrained geometric vision model as a feature extractor. A trainable projection layer then adapts these geometrically-rich features for the policy decoder, relieving it of the burden of learning 3D consistency from scratch. Through extensive evaluations on LIBERO benchmark subsets, we show GeoAware-VLA achieves substantial improvements in zero-shot generalization to novel camera poses, boosting success rates by over 2x in simulation. Crucially, these benefits translate to the physical world; our model shows a significant performance gain on a real robot, especially when evaluated from unseen camera angles. Our approach proves effective across both continuous and discrete action spaces, highlighting that robust geometric grounding is a key component for creating more generalizable robotic agents.
>
---
#### [new 023] BIM Informed Visual SLAM for Construction Monitoring
- **分类: cs.RO**

- **简介: 论文提出一种结合BIM的视觉SLAM方法，用于施工监控。该方法利用BIM作为结构先验，减少视觉SLAM在重复布局、遮挡等场景下的轨迹漂移问题，提升定位与建图精度。**

- **链接: [http://arxiv.org/pdf/2509.13972v1](http://arxiv.org/pdf/2509.13972v1)**

> **作者:** Asier Bikandi; Miguel Fernandez-Cortizas; Muhammad Shaheer; Ali Tourani; Holger Voos; Jose Luis Sanchez-Lopez
>
> **备注:** 8 pages, 5 tables, 4 figures
>
> **摘要:** Simultaneous Localization and Mapping (SLAM) is a key tool for monitoring construction sites, where aligning the evolving as-built state with the as-planned design enables early error detection and reduces costly rework. LiDAR-based SLAM achieves high geometric precision, but its sensors are typically large and power-demanding, limiting their use on portable platforms. Visual SLAM offers a practical alternative with lightweight cameras already embedded in most mobile devices. however, visually mapping construction environments remains challenging: repetitive layouts, occlusions, and incomplete or low-texture structures often cause drift in the trajectory map. To mitigate this, we propose an RGB-D SLAM system that incorporates the Building Information Model (BIM) as structural prior knowledge. Instead of relying solely on visual cues, our system continuously establishes correspondences between detected wall and their BIM counterparts, which are then introduced as constraints in the back-end optimization. The proposed method operates in real time and has been validated on real construction sites, reducing trajectory error by an average of 23.71% and map RMSE by 7.14% compared to visual SLAM baselines. These results demonstrate that BIM constraints enable reliable alignment of the digital plan with the as-built scene, even under partially constructed conditions.
>
---
#### [new 024] MIMIC-D: Multi-modal Imitation for MultI-agent Coordination with Decentralized Diffusion Policies
- **分类: cs.RO**

- **简介: 该论文提出MIMIC-D，解决多智能体在多模态任务中协调问题。采用CTDE框架，利用扩散策略实现去中心化执行，通过模仿学习从专家示范中学习多模态行为，提升多智能体协作能力。**

- **链接: [http://arxiv.org/pdf/2509.14159v1](http://arxiv.org/pdf/2509.14159v1)**

> **作者:** Dayi Dong; Maulik Bhatt; Seoyeon Choi; Negar Mehr
>
> **备注:** 9 pages, 4 figures, 5 tables
>
> **摘要:** As robots become more integrated in society, their ability to coordinate with other robots and humans on multi-modal tasks (those with multiple valid solutions) is crucial. We propose to learn such behaviors from expert demonstrations via imitation learning (IL). However, when expert demonstrations are multi-modal, standard IL approaches can struggle to capture the diverse strategies, hindering effective coordination. Diffusion models are known to be effective at handling complex multi-modal trajectory distributions in single-agent systems. Diffusion models have also excelled in multi-agent scenarios where multi-modality is more common and crucial to learning coordinated behaviors. Typically, diffusion-based approaches require a centralized planner or explicit communication among agents, but this assumption can fail in real-world scenarios where robots must operate independently or with agents like humans that they cannot directly communicate with. Therefore, we propose MIMIC-D, a Centralized Training, Decentralized Execution (CTDE) paradigm for multi-modal multi-agent imitation learning using diffusion policies. Agents are trained jointly with full information, but execute policies using only local information to achieve implicit coordination. We demonstrate in both simulation and hardware experiments that our method recovers multi-modal coordination behavior among agents in a variety of tasks and environments, while improving upon state-of-the-art baselines.
>
---
#### [new 025] CrazyMARL: Decentralized Direct Motor Control Policies for Cooperative Aerial Transport of Cable-Suspended Payloads
- **分类: cs.RO; cs.MA**

- **简介: 论文提出CrazyMARL框架，解决多无人机协同运输悬吊负载时的动态控制问题。通过强化学习实现去中心化控制，提升抗干扰与跟踪精度，并成功实现仿真到现实的迁移，适用于复杂环境下的负载运输任务。**

- **链接: [http://arxiv.org/pdf/2509.14126v1](http://arxiv.org/pdf/2509.14126v1)**

> **作者:** Viktor Lorentz; Khaled Wahba; Sayantan Auddy; Marc Toussaint; Wolfgang Hönig
>
> **备注:** This work has been submitted to IEEE for possible publication
>
> **摘要:** Collaborative transportation of cable-suspended payloads by teams of Unmanned Aerial Vehicles (UAVs) has the potential to enhance payload capacity, adapt to different payload shapes, and provide built-in compliance, making it attractive for applications ranging from disaster relief to precision logistics. However, multi-UAV coordination under disturbances, nonlinear payload dynamics, and slack--taut cable modes remains a challenging control problem. To our knowledge, no prior work has addressed these cable mode transitions in the multi-UAV context, instead relying on simplifying rigid-link assumptions. We propose CrazyMARL, a decentralized Reinforcement Learning (RL) framework for multi-UAV cable-suspended payload transport. Simulation results demonstrate that the learned policies can outperform classical decentralized controllers in terms of disturbance rejection and tracking precision, achieving an 80% recovery rate from harsh conditions compared to 44% for the baseline method. We also achieve successful zero-shot sim-to-real transfer and demonstrate that our policies are highly robust under harsh conditions, including wind, random external disturbances, and transitions between slack and taut cable dynamics. This work paves the way for autonomous, resilient UAV teams capable of executing complex payload missions in unstructured environments.
>
---
#### [new 026] Label-Efficient Grasp Joint Prediction with Point-JEPA
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究利用Point-JEPA进行3D自监督预训练，以实现标签高效的抓取关节角预测。通过点云和ShapeNet预训练模型，在低标签情况下显著降低RMSE，达到与全监督相当的效果，为数据高效的抓取学习提供新方法。**

- **链接: [http://arxiv.org/pdf/2509.13349v1](http://arxiv.org/pdf/2509.13349v1)**

> **作者:** Jed Guzelkabaagac; Boris Petrović
>
> **备注:** 4 pages, 5 figures. Submitted to IROS 2025 Workshop
>
> **摘要:** We investigate whether 3D self-supervised pretraining with a Joint-Embedding Predictive Architecture (Point-JEPA) enables label-efficient grasp joint-angle prediction. Using point clouds tokenized from meshes and a ShapeNet-pretrained Point-JEPA encoder, we train a lightweight multi-hypothesis head with winner-takes-all and evaluate by top-logit selection. On DLR-Hand II with object-level splits, Point-JEPA reduces RMSE by up to 26% in low-label regimes and reaches parity with full supervision. These results suggest JEPA-style pretraining is a practical approach for data-efficient grasp learning.
>
---
#### [new 027] Reinforcement Learning for Autonomous Point-to-Point UAV Navigation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出基于强化学习的无人机自主点对点导航方法，解决无人自主飞行路径规划问题。通过自定义奖励函数训练策略，并在真实无人机平台验证，实现高效、安全的自主导航。**

- **链接: [http://arxiv.org/pdf/2509.13943v1](http://arxiv.org/pdf/2509.13943v1)**

> **作者:** Salim Oyinlola; Nitesh Subedi; Soumik Sarkar
>
> **备注:** Presented at the Research Experience for Undergraduates (REU) Symposium at the Translational AI Centre in Iowa State University
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are increasingly used in automated inspection, delivery, and navigation tasks that require reliable autonomy. This project develops a reinforcement learning (RL) approach to enable a single UAV to autonomously navigate between predefined points without manual intervention. The drone learns navigation policies through trial-and-error interaction, using a custom reward function that encourages goal-reaching efficiency while penalizing collisions and unsafe behavior. The control system integrates ROS with a Gym-compatible training environment, enabling flexible deployment and testing. After training, the learned policy is deployed on a real UAV platform and evaluated under practical conditions. Results show that the UAV can successfully perform autonomous navigation with minimal human oversight, demonstrating the viability of RL-based control for point-to-point drone operations in real-world scenarios.
>
---
#### [new 028] MCGS-SLAM: A Multi-Camera SLAM Framework Using Gaussian Splatting for High-Fidelity Mapping
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MCGS-SLAM，一种基于多摄像头和高斯点云的SLAM框架，用于提升机器人和自动驾驶的高保真地图构建。解决单目SLAM在鲁棒性和几何覆盖上的不足，通过多视角RGB融合与优化实现更准确的轨迹和重建。**

- **链接: [http://arxiv.org/pdf/2509.14191v1](http://arxiv.org/pdf/2509.14191v1)**

> **作者:** Zhihao Cao; Hanyu Wu; Li Wa Tang; Zizhou Luo; Zihan Zhu; Wei Zhang; Marc Pollefeys; Martin R. Oswald
>
> **摘要:** Recent progress in dense SLAM has primarily targeted monocular setups, often at the expense of robustness and geometric coverage. We present MCGS-SLAM, the first purely RGB-based multi-camera SLAM system built on 3D Gaussian Splatting (3DGS). Unlike prior methods relying on sparse maps or inertial data, MCGS-SLAM fuses dense RGB inputs from multiple viewpoints into a unified, continuously optimized Gaussian map. A multi-camera bundle adjustment (MCBA) jointly refines poses and depths via dense photometric and geometric residuals, while a scale consistency module enforces metric alignment across views using low-rank priors. The system supports RGB input and maintains real-time performance at large scale. Experiments on synthetic and real-world datasets show that MCGS-SLAM consistently yields accurate trajectories and photorealistic reconstructions, usually outperforming monocular baselines. Notably, the wide field of view from multi-camera input enables reconstruction of side-view regions that monocular setups miss, critical for safe autonomous operation. These results highlight the promise of multi-camera Gaussian Splatting SLAM for high-fidelity mapping in robotics and autonomous driving.
>
---
#### [new 029] Dual-Actor Fine-Tuning of VLA Models: A Talk-and-Tweak Human-in-the-Loop Approach
- **分类: cs.RO**

- **简介: 该论文提出一种基于RL的双演员微调框架，用于提升VLA模型在复杂现实任务中的表现。通过人机交互生成语义指令数据，实现高效在线微调，在多任务实验中取得高成功率。属于机器人控制与强化学习任务。**

- **链接: [http://arxiv.org/pdf/2509.13774v1](http://arxiv.org/pdf/2509.13774v1)**

> **作者:** Piaopiao Jin; Qi Wang; Guokang Sun; Ziwen Cai; Pinjia He; Yangwei You
>
> **摘要:** Vision-language-action (VLA) models demonstrate strong generalization in robotic manipulation but face challenges in complex, real-world tasks. While supervised fine-tuning with demonstrations is constrained by data quality, reinforcement learning (RL) offers a promising alternative. We propose a human-in-the-loop dual-actor fine-tuning framework grounded in RL. The framework integrates a primary actor for robust multi-task performance with a refinement actor for latent-space adaptation. Beyond standard physical interventions, we introduce a lightweight talk-and-tweak scheme that converts human corrections into semantically grounded language commands, thereby generating a new dataset for policy learning. In real-world multi-task experiments, our approach achieves 100% success across three tasks within 101 minutes of online fine-tuning. For long-horizon tasks, it sustains a 50% success rate over 12 consecutive operations. Furthermore, the framework scales effectively to multi-robot training, achieving up to a 2 times improvement in efficiency when using dual robots. The experiment videos are available at https://sites.google.com/view/hil-daft/.
>
---
#### [new 030] Using role-play and Hierarchical Task Analysis for designing human-robot interaction
- **分类: cs.RO**

- **简介: 论文提出将角色扮演与层级任务分析应用于人机交互设计，以开发社区药店助手机器人。通过角色扮演模拟客户需求，层级任务分析确保行为建模正确，促进协同设计。旨在提升社会机器人交互的实用性与合理性。**

- **链接: [http://arxiv.org/pdf/2509.13378v1](http://arxiv.org/pdf/2509.13378v1)**

> **作者:** Mattias Wingren; Sören Andersson; Sara Rosenberg; Malin Andtfolk; Susanne Hägglund; Prashani Jayasingha Arachchige; Linda Nyholm
>
> **备注:** 11 pages. This is a preprint version of the published paper in the International Conference on Social Robotics: https://link.springer.com/chapter/10.1007/978-981-96-3522-1_28
>
> **摘要:** We present the use of two methods we believe warrant more use than they currently have in the field of human-robot interaction: role-play and Hierarchical Task Analysis. Some of its potential is showcased through our use of them in an ongoing research project which entails developing a robot application meant to assist at a community pharmacy. The two methods have provided us with several advantages. The role-playing provided a controlled and adjustable environment for understanding the customers' needs where pharmacists could act as models for the robot's behavior; and the Hierarchical Task Analysis ensured the behavior displayed was modelled correctly and aided development through facilitating co-design. Future research could focus on developing task analysis methods especially suited for social robot interaction.
>
---
#### [new 031] Barometer-Aided Attitude Estimation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种利用气压计辅助的姿态估计方法，解决GNSS不可用环境下IMU姿态估计不准确的问题。通过非线性观测器结合Riccati观测器与互补滤波器，实现几何一致且几乎全局渐近稳定的姿态估计。**

- **链接: [http://arxiv.org/pdf/2509.13649v1](http://arxiv.org/pdf/2509.13649v1)**

> **作者:** Méloné Nyoba Tchonkeu; Soulaimane Berkane; Tarek Hamel
>
> **备注:** 6 pages, 4 figures. this manuscript is submitted to IEEE Control Systems Letters (L-CSS) with American Control Conference (ACC) option
>
> **摘要:** Accurate and robust attitude estimation is a central challenge for autonomous vehicles operating in GNSS-denied or highly dynamic environments. In such cases, Inertial Measurement Units (IMUs) alone are insufficient for reliable tilt estimation due to the ambiguity between gravitational and inertial accelerations. While auxiliary velocity sensors, such as GNSS, Pitot tubes, Doppler radar, or visual odometry, are often used, they can be unavailable, intermittent, or costly. This work introduces a barometer-aided attitude estimation architecture that leverages barometric altitude measurements to infer vertical velocity and attitude within a nonlinear observer on SO(3). The design cascades a deterministic Riccati observer with a complementary filter, ensuring Almost Global Asymptotic Stability (AGAS) under a uniform observability condition while maintaining geometric consistency. The analysis highlights barometer-aided estimation as a lightweight and effective complementary modality.
>
---
#### [new 032] Motion Adaptation Across Users and Tasks for Exoskeletons via Meta-Learning
- **分类: cs.RO**

- **简介: 论文提出一种基于元学习的模仿学习方法，用于外骨骼系统的运动自适应。旨在解决个性化与任务泛化问题，通过模拟数据训练神经网络，实现快速适应新用户和任务，提升外骨骼辅助效果。**

- **链接: [http://arxiv.org/pdf/2509.13736v1](http://arxiv.org/pdf/2509.13736v1)**

> **作者:** Muyuan Ma; Long Cheng; Lijun Han; Xiuze Xia; Houcheng Li
>
> **摘要:** Wearable exoskeletons can augment human strength and reduce muscle fatigue during specific tasks. However, developing personalized and task-generalizable assistance algorithms remains a critical challenge. To address this, a meta-imitation learning approach is proposed. This approach leverages a task-specific neural network to predict human elbow joint movements, enabling effective assistance while enhancing generalization to new scenarios. To accelerate data collection, full-body keypoint motions are extracted from publicly available RGB video and motion-capture datasets across multiple tasks, and subsequently retargeted in simulation. Elbow flexion trajectories generated in simulation are then used to train the task-specific neural network within the model-agnostic meta-learning (MAML) framework, which allows the network to rapidly adapt to novel tasks and unseen users with only a few gradient updates. The adapted network outputs personalized references tracked by a gravity-compensated PD controller to ensure stable assistance. Experimental results demonstrate that the exoskeleton significantly reduces both muscle activation and metabolic cost for new users performing untrained tasks, compared to performing without exoskeleton assistance. These findings suggest that the proposed framework effectively improves task generalization and user adaptability for wearable exoskeleton systems.
>
---
#### [new 033] Leg-Arm Coordinated Operation for Curtain Wall Installation
- **分类: cs.RO**

- **简介: 论文提出基于六足机器人的肢体协同控制框架，用于幕墙安装任务，解决传统方法效率低、安全风险高的问题，通过层级优化实现臂腿协调，提升安装效率与安全性。**

- **链接: [http://arxiv.org/pdf/2509.13595v1](http://arxiv.org/pdf/2509.13595v1)**

> **作者:** Xiao Liu; Weijun Wang; Tianlun Huang; Zhiyong Wang; Wei Feng
>
> **摘要:** With the acceleration of urbanization, the number of high-rise buildings and large public facilities is increasing, making curtain walls an essential component of modern architecture with widespread applications. Traditional curtain wall installation methods face challenges such as variable on-site terrain, high labor intensity, low construction efficiency, and significant safety risks. Large panels often require multiple workers to complete installation. To address these issues, based on a hexapod curtain wall installation robot, we design a hierarchical optimization-based whole-body control framework for coordinated arm-leg planning tailored to three key tasks: wall installation, ceiling installation, and floor laying. This framework integrates the motion of the hexapod legs with the operation of the folding arm and the serial-parallel manipulator. We conduct experiments on the hexapod curtain wall installation robot to validate the proposed control method, demonstrating its capability in performing curtain wall installation tasks. Our results confirm the effectiveness of the hierarchical optimization-based arm-leg coordination framework for the hexapod robot, laying the foundation for its further application in complex construction site environments.
>
---
#### [new 034] SHaRe-RL: Structured, Interactive Reinforcement Learning for Contact-Rich Industrial Assembly Tasks
- **分类: cs.RO; I.2.9**

- **简介: 该论文提出SHaRe-RL框架，用于解决高混低量工业装配中的机器人学习问题。通过结构化技能、人类示范与在线修正、力控制等方法，实现高效安全的在线学习，提升装配任务的可靠性和经济性。**

- **链接: [http://arxiv.org/pdf/2509.13949v1](http://arxiv.org/pdf/2509.13949v1)**

> **作者:** Jannick Stranghöner; Philipp Hartmann; Marco Braun; Sebastian Wrede; Klaus Neumann
>
> **备注:** 8 pages, 5 figures, submitted to the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** High-mix low-volume (HMLV) industrial assembly, common in small and medium-sized enterprises (SMEs), requires the same precision, safety, and reliability as high-volume automation while remaining flexible to product variation and environmental uncertainty. Current robotic systems struggle to meet these demands. Manual programming is brittle and costly to adapt, while learning-based methods suffer from poor sample efficiency and unsafe exploration in contact-rich tasks. To address this, we present SHaRe-RL, a reinforcement learning framework that leverages multiple sources of prior knowledge. By (i) structuring skills into manipulation primitives, (ii) incorporating human demonstrations and online corrections, and (iii) bounding interaction forces with per-axis compliance, SHaRe-RL enables efficient and safe online learning for long-horizon, contact-rich industrial assembly tasks. Experiments on the insertion of industrial Harting connector modules with 0.2-0.4 mm clearance demonstrate that SHaRe-RL achieves reliable performance within practical time budgets. Our results show that process expertise, without requiring robotics or RL knowledge, can meaningfully contribute to learning, enabling safer, more robust, and more economically viable deployment of RL for industrial assembly.
>
---
#### [new 035] Repulsive Trajectory Modification and Conflict Resolution for Efficient Multi-Manipulator Motion Planning
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机械臂运动规划任务，旨在解决多机械臂避障与冲突协调问题。提出基于排斥轨迹修正的改进CBS方法，通过人工势场引导避障，减少冲突并提升规划效率。**

- **链接: [http://arxiv.org/pdf/2509.13882v1](http://arxiv.org/pdf/2509.13882v1)**

> **作者:** Junhwa Hong; Beomjoon Lee; Woojin Lee; Changjoo Nam
>
> **备注:** 7 pages
>
> **摘要:** We propose an efficient motion planning method designed to efficiently find collision-free trajectories for multiple manipulators. While multi-manipulator systems offer significant advantages, coordinating their motions is computationally challenging owing to the high dimensionality of their composite configuration space. Conflict-Based Search (CBS) addresses this by decoupling motion planning, but suffers from subsequent conflicts incurred by resolving existing conflicts, leading to an exponentially growing constraint tree of CBS. Our proposed method is based on repulsive trajectory modification within the two-level structure of CBS. Unlike conventional CBS variants, the low-level planner applies a gradient descent approach using an Artificial Potential Field. This field generates repulsive forces that guide the trajectory of the conflicting manipulator away from those of other robots. As a result, subsequent conflicts are less likely to occur. Additionally, we develop a strategy that, under a specific condition, directly attempts to find a conflict-free solution in a single step without growing the constraint tree. Through extensive tests including physical robot experiments, we demonstrate that our method consistently reduces the number of expanded nodes in the constraint tree, achieves a higher success rate, and finds a solution faster compared to Enhanced CBS and other state-of-the-art algorithms.
>
---
#### [new 036] A Convex Formulation of Compliant Contact between Filaments and Rigid Bodies
- **分类: cs.RO**

- **简介: 该论文提出一种凸优化框架，用于模拟细丝与刚体间的接触交互。解决细丝因维度差异导致的模拟难题，统一弹性杆模型与接触模型，实现精确摩擦仿真，并应用于软体机器人与绳索操作等场景。**

- **链接: [http://arxiv.org/pdf/2509.13434v1](http://arxiv.org/pdf/2509.13434v1)**

> **作者:** Wei-Chen Li; Glen Chou
>
> **摘要:** We present a computational framework for simulating filaments interacting with rigid bodies through contact. Filaments are challenging to simulate due to their codimensionality, i.e., they are one-dimensional structures embedded in three-dimensional space. Existing methods often assume that filaments remain permanently attached to rigid bodies. Our framework unifies discrete elastic rod (DER) modeling, a pressure field patch contact model, and a convex contact formulation to accurately simulate frictional interactions between slender filaments and rigid bodies - capabilities not previously achievable. Owing to the convex formulation of contact, each time step can be solved to global optimality, guaranteeing complementarity between contact velocity and impulse. We validate the framework by assessing the accuracy of frictional forces and comparing its physical fidelity against baseline methods. Finally, we demonstrate its applicability in both soft robotics, such as a stochastic filament-based gripper, and deformable object manipulation, such as shoelace tying, providing a versatile simulator for systems involving complex filament-filament and filament-rigid body interactions.
>
---
#### [new 037] Prompt2Auto: From Motion Prompt to Automated Control via Geometry-Invariant One-Shot Gaussian Process Learning
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文提出Prompt2Auto框架，解决机器人从单次运动提示中学习自动化控制的问题。通过几何不变的高斯过程学习，实现跨坐标变换的泛化，减少示范负担，支持多技能自主控制。**

- **链接: [http://arxiv.org/pdf/2509.14040v1](http://arxiv.org/pdf/2509.14040v1)**

> **作者:** Zewen Yang; Xiaobing Dai; Dongfa Zhang; Yu Li; Ziyang Meng; Bingkun Huang; Hamid Sadeghian; Sami Haddadin
>
> **摘要:** Learning from demonstration allows robots to acquire complex skills from human demonstrations, but conventional approaches often require large datasets and fail to generalize across coordinate transformations. In this paper, we propose Prompt2Auto, a geometry-invariant one-shot Gaussian process (GeoGP) learning framework that enables robots to perform human-guided automated control from a single motion prompt. A dataset-construction strategy based on coordinate transformations is introduced that enforces invariance to translation, rotation, and scaling, while supporting multi-step predictions. Moreover, GeoGP is robust to variations in the user's motion prompt and supports multi-skill autonomy. We validate the proposed approach through numerical simulations with the designed user graphical interface and two real-world robotic experiments, which demonstrate that the proposed method is effective, generalizes across tasks, and significantly reduces the demonstration burden. Project page is available at: https://prompt2auto.github.io
>
---
#### [new 038] MAP: End-to-End Autonomous Driving with Map-Assisted Planning
- **分类: cs.RO; cs.AI; cs.CV; I.2.9; I.2.10**

- **简介: 该论文提出MAP框架，用于端到端自动驾驶轨迹规划。通过融合地图信息与车辆状态，提升规划性能。实验表明其在多个指标上优于基线模型，效果显著。**

- **链接: [http://arxiv.org/pdf/2509.13926v1](http://arxiv.org/pdf/2509.13926v1)**

> **作者:** Huilin Yin; Yiming Kan; Daniel Watzenig
>
> **备注:** 8 pages, 2 figures, accepted by ICCVW Author list updated to match the camera-ready version, in compliance with conference policy
>
> **摘要:** In recent years, end-to-end autonomous driving has attracted increasing attention for its ability to jointly model perception, prediction, and planning within a unified framework. However, most existing approaches underutilize the online mapping module, leaving its potential to enhance trajectory planning largely untapped. This paper proposes MAP (Map-Assisted Planning), a novel map-assisted end-to-end trajectory planning framework. MAP explicitly integrates segmentation-based map features and the current ego status through a Plan-enhancing Online Mapping module, an Ego-status-guided Planning module, and a Weight Adapter based on current ego status. Experiments conducted on the DAIR-V2X-seq-SPD dataset demonstrate that the proposed method achieves a 16.6% reduction in L2 displacement error, a 56.2% reduction in off-road rate, and a 44.5% improvement in overall score compared to the UniV2X baseline, even without post-processing. Furthermore, it achieves top ranking in Track 2 of the End-to-End Autonomous Driving through V2X Cooperation Challenge of MEIS Workshop @CVPR2025, outperforming the second-best model by 39.5% in terms of overall score. These results highlight the effectiveness of explicitly leveraging semantic map features in planning and suggest new directions for improving structure design in end-to-end autonomous driving systems. Our code is available at https://gitee.com/kymkym/map.git
>
---
#### [new 039] Constraint-Consistent Control of Task-Based and Kinematic RCM Constraints for Surgical Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种约束一致的力控方法，解决手术机器人中动态条件下远程运动中心（RCM）约束难以精确满足的问题。通过将RCM视为时变完整约束，嵌入逆动力学框架，实现任务与运动学约束统一，提升手术安全性和控制精度。**

- **链接: [http://arxiv.org/pdf/2509.14075v1](http://arxiv.org/pdf/2509.14075v1)**

> **作者:** Yu Li; Hamid Sadeghian; Zewen Yang; Valentin Le Mesle; Sami Haddadin
>
> **摘要:** Robotic-assisted minimally invasive surgery (RAMIS) requires precise enforcement of the remote center of motion (RCM) constraint to ensure safe tool manipulation through a trocar. Achieving this constraint under dynamic and interactive conditions remains challenging, as existing control methods either lack robustness at the torque level or do not guarantee consistent RCM constraint satisfaction. This paper proposes a constraint-consistent torque controller that treats the RCM as a rheonomic holonomic constraint and embeds it into a projection-based inverse-dynamics framework. The method unifies task-level and kinematic formulations, enabling accurate tool-tip tracking while maintaining smooth and efficient torque behavior. The controller is validated both in simulation and on a RAMIS training platform, and is benchmarked against state-of-the-art approaches. Results show improved RCM constraint satisfaction, reduced required torque, and robust performance by improving joint torque smoothness through the consistency formulation under clinically relevant scenarios, including spiral trajectories, variable insertion depths, moving trocars, and human interaction. These findings demonstrate the potential of constraint-consistent torque control to enhance safety and reliability in surgical robotics. The project page is available at: https://rcmpc-cube.github.io
>
---
#### [new 040] Agile in the Face of Delay: Asynchronous End-to-End Learning for Real-World Aerial Navigation
- **分类: cs.RO**

- **简介: 论文提出一种异步强化学习框架，解决无人机自主导航中感知与控制频率不匹配问题，通过解耦感知与控制、引入时间编码模块，实现高频率控制与异步感知融合，提升复杂环境下的实时反应能力。**

- **链接: [http://arxiv.org/pdf/2509.13816v1](http://arxiv.org/pdf/2509.13816v1)**

> **作者:** Yude Li; Zhexuan Zhou; Huizhe Li; Youmin Gong; Jie Mei
>
> **摘要:** Robust autonomous navigation for Autonomous Aerial Vehicles (AAVs) in complex environments is a critical capability. However, modern end-to-end navigation faces a key challenge: the high-frequency control loop needed for agile flight conflicts with low-frequency perception streams, which are limited by sensor update rates and significant computational cost. This mismatch forces conventional synchronous models into undesirably low control rates. To resolve this, we propose an asynchronous reinforcement learning framework that decouples perception and control, enabling a high-frequency policy to act on the latest IMU state for immediate reactivity, while incorporating perception features asynchronously. To manage the resulting data staleness, we introduce a theoretically-grounded Temporal Encoding Module (TEM) that explicitly conditions the policy on perception delays, a strategy complemented by a two-stage curriculum to ensure stable and efficient training. Validated in extensive simulations, our method was successfully deployed in zero-shot sim-to-real transfer on an onboard NUC, where it sustains a 100~Hz control rate and demonstrates robust, agile navigation in cluttered real-world environments. Our source code will be released for community reference.
>
---
#### [new 041] InterKey: Cross-modal Intersection Keypoints for Global Localization on OpenStreetMap
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出InterKey框架，用于解决自动驾驶车辆在GNSS不可用环境下的全局定位问题。通过融合点云与OpenStreetMap数据，利用道路交叉口作为特征点，实现跨模态匹配，提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.13857v1](http://arxiv.org/pdf/2509.13857v1)**

> **作者:** Nguyen Hoang Khoi Tran; Julie Stephany Berrio; Mao Shan; Stewart Worrall
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Reliable global localization is critical for autonomous vehicles, especially in environments where GNSS is degraded or unavailable, such as urban canyons and tunnels. Although high-definition (HD) maps provide accurate priors, the cost of data collection, map construction, and maintenance limits scalability. OpenStreetMap (OSM) offers a free and globally available alternative, but its coarse abstraction poses challenges for matching with sensor data. We propose InterKey, a cross-modal framework that leverages road intersections as distinctive landmarks for global localization. Our method constructs compact binary descriptors by jointly encoding road and building imprints from point clouds and OSM. To bridge modality gaps, we introduce discrepancy mitigation, orientation determination, and area-equalized sampling strategies, enabling robust cross-modal matching. Experiments on the KITTI dataset demonstrate that InterKey achieves state-of-the-art accuracy, outperforming recent baselines by a large margin. The framework generalizes to sensors that can produce dense structural point clouds, offering a scalable and cost-effective solution for robust vehicle localization.
>
---
#### [new 042] Soft Regrasping Tool Inspired by Jamming Gripper
- **分类: cs.RO**

- **简介: 论文提出一种受夹紧现象启发的柔性夹具，用于解决机器人装配中重新抓取时的姿态不确定性问题。通过优化腔体尺寸，实现多种零件的稳定放置，实验显示其具有高成功率，为柔性夹具在自动化装配中的应用提供了新方法。**

- **链接: [http://arxiv.org/pdf/2509.13815v1](http://arxiv.org/pdf/2509.13815v1)**

> **作者:** Takuya Kiyokawa; Zhengtao Hu; Weiwei Wan; Kensuke Harada
>
> **备注:** 6 pages, 9 figures
>
> **摘要:** Regrasping on fixtures is a promising approach to reduce pose uncertainty in robotic assembly, but conventional rigid fixtures lack adaptability and require dedicated designs for each part. To overcome this limitation, we propose a soft jig inspired by the jamming transition phenomenon, which can be continuously deformed to accommodate diverse object geometries. By pressing a triangular-pyramid-shaped tool into the membrane and evacuating the enclosed air, a stable cavity is formed as a placement space. We further optimize the stamping depth to balance placement stability and gripper accessibility. In soft-jig-based regrasping, the key challenge lies in optimizing the cavity size to achieve precise dropping; once the part is reliably placed, subsequent grasping can be performed with reduced uncertainty. Accordingly, we conducted drop experiments on ten mechanical parts of varying shapes, which achieved placement success rates exceeding 80% for most objects and above 90% for cylindrical ones, while failures were mainly caused by geometric constraints and membrane properties. These results demonstrate that the proposed jig enables general-purpose, accurate, and repeatable regrasping, while also clarifying its current limitations and future potential as a practical alternative to rigid fixtures in assembly automation.
>
---
#### [new 043] Maximizing UAV Cellular Connectivity with Reinforcement Learning for BVLoS Path Planning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出基于强化学习的路径规划方法，解决无人机在超视距飞行中最大化蜂窝连接质量的问题。通过考虑实际覆盖约束和信道模型，训练智能体以生成最优路径，提升连接可靠性与飞行安全性。**

- **链接: [http://arxiv.org/pdf/2509.13336v1](http://arxiv.org/pdf/2509.13336v1)**

> **作者:** Mehran Behjati; Rosdiadee Nordin; Nor Fadzilah Abdullah
>
> **备注:** Submitted to an IEEE Conference
>
> **摘要:** This paper presents a reinforcement learning (RL) based approach for path planning of cellular connected unmanned aerial vehicles (UAVs) operating beyond visual line of sight (BVLoS). The objective is to minimize travel distance while maximizing the quality of cellular link connectivity by considering real world aerial coverage constraints and employing an empirical aerial channel model. The proposed solution employs RL techniques to train an agent, using the quality of communication links between the UAV and base stations (BSs) as the reward function. Simulation results demonstrate the effectiveness of the proposed method in training the agent and generating feasible UAV path plans. The proposed approach addresses the challenges due to limitations in UAV cellular communications, highlighting the need for investigations and considerations in this area. The RL algorithm efficiently identifies optimal paths, ensuring maximum connectivity with ground BSs to ensure safe and reliable BVLoS flight operation. Moreover, the solution can be deployed as an offline path planning module that can be integrated into future ground control systems (GCS) for UAV operations, enhancing their capabilities and safety. The method holds potential for complex long range UAV applications, advancing the technology in the field of cellular connected UAV path planning.
>
---
#### [new 044] Track Any Motions under Any Disturbances
- **分类: cs.RO**

- **简介: 该论文提出Any2Track，一种两阶段强化学习框架，用于在真实世界中跟踪各种动态运动并克服干扰。其核心组件包括通用运动跟踪器AnyTracker和在线适应模块AnyAdapter，实现了零样本模拟到现实的迁移，解决了复杂环境下动态适应问题。**

- **链接: [http://arxiv.org/pdf/2509.13833v1](http://arxiv.org/pdf/2509.13833v1)**

> **作者:** Zhikai Zhang; Jun Guo; Chao Chen; Jilong Wang; Chenghuai Lin; Yunrui Lian; Han Xue; Zhenrong Wang; Maoqi Liu; Huaping Liu; He Wang; Li Yi
>
> **摘要:** A foundational humanoid motion tracker is expected to be able to track diverse, highly dynamic, and contact-rich motions. More importantly, it needs to operate stably in real-world scenarios against various dynamics disturbances, including terrains, external forces, and physical property changes for general practical use. To achieve this goal, we propose Any2Track (Track Any motions under Any disturbances), a two-stage RL framework to track various motions under multiple disturbances in the real world. Any2Track reformulates dynamics adaptability as an additional capability on top of basic action execution and consists of two key components: AnyTracker and AnyAdapter. AnyTracker is a general motion tracker with a series of careful designs to track various motions within a single policy. AnyAdapter is a history-informed adaptation module that endows the tracker with online dynamics adaptability to overcome the sim2real gap and multiple real-world disturbances. We deploy Any2Track on Unitree G1 hardware and achieve a successful sim2real transfer in a zero-shot manner. Any2Track performs exceptionally well in tracking various motions under multiple real-world disturbances.
>
---
#### [new 045] Dense-Jump Flow Matching with Non-Uniform Time Scheduling for Robotic Policies: Mitigating Multi-Step Inference Degradation
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种改进的流匹配方法，用于机器人策略学习。针对多步推理性能下降问题，采用非均匀时间调度和密集跳跃积分，提升泛化能力与稳定性，实现优于现有方法的性能提升。**

- **链接: [http://arxiv.org/pdf/2509.13574v1](http://arxiv.org/pdf/2509.13574v1)**

> **作者:** Zidong Chen; Zihao Guo; Peng Wang; ThankGod Itua Egbe; Yan Lyu; Chenghao Qian
>
> **摘要:** Flow matching has emerged as a competitive framework for learning high-quality generative policies in robotics; however, we find that generalisation arises and saturates early along the flow trajectory, in accordance with recent findings in the literature. We further observe that increasing the number of Euler integration steps during inference counter-intuitively and universally degrades policy performance. We attribute this to (i) additional, uniformly spaced integration steps oversample the late-time region, thereby constraining actions towards the training trajectories and reducing generalisation; and (ii) the learned velocity field becoming non-Lipschitz as integration time approaches 1, causing instability. To address these issues, we propose a novel policy that utilises non-uniform time scheduling (e.g., U-shaped) during training, which emphasises both early and late temporal stages to regularise policy training, and a dense-jump integration schedule at inference, which uses a single-step integration to replace the multi-step integration beyond a jump point, to avoid unstable areas around 1. Essentially, our policy is an efficient one-step learner that still pushes forward performance through multi-step integration, yielding up to 23.7% performance gains over state-of-the-art baselines across diverse robotic tasks.
>
---
#### [new 046] VEGA: Electric Vehicle Navigation Agent via Physics-Informed Neural Operator and Proximal Policy Optimization
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出VEGA，一种基于物理信息神经算子和近端策略优化的电动汽车导航代理，用于解决充电感知路径规划问题。通过学习车辆动力学并结合强化学习，实现无需额外传感器的高效路径优化，具有良好的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.13386v1](http://arxiv.org/pdf/2509.13386v1)**

> **作者:** Hansol Lim; Minhyeok Im; Jonathan Boyack; Jee Won Lee; Jongseong Brad Choi
>
> **备注:** This work has been submitted to the 2026 IEEE International Conference on Robotics and Automation (ICRA) for possible publication
>
> **摘要:** Demands for software-defined vehicles (SDV) are rising and electric vehicles (EVs) are increasingly being equipped with powerful computers. This enables onboard AI systems to optimize charge-aware path optimization customized to reflect vehicle's current condition and environment. We present VEGA, a charge-aware EV navigation agent that plans over a charger-annotated road graph using Proximal Policy Optimization (PPO) with budgeted A* teacher-student guidance under state-of-charge (SoC) feasibility. VEGA consists of two modules. First, a physics-informed neural operator (PINO), trained on real vehicle speed and battery-power logs, uses recent vehicle speed logs to estimate aerodynamic drag, rolling resistance, mass, motor and regenerative-braking efficiencies, and auxiliary load by learning a vehicle-custom dynamics. Second, a Reinforcement Learning (RL) agent uses these dynamics to optimize a path with optimal charging stops and dwell times under SoC constraints. VEGA requires no additional sensors and uses only vehicle speed signals. It may serve as a virtual sensor for power and efficiency to potentially reduce EV cost. In evaluation on long routes like San Francisco to New York, VEGA's stops, dwell times, SoC management, and total travel time closely track Tesla Trip Planner while being slightly more conservative, presumably due to real vehicle conditions such as vehicle parameter drift due to deterioration. Although trained only in U.S. regions, VEGA was able to compute optimal charge-aware paths in France and Japan, demonstrating generalizability. It achieves practical integration of physics-informed learning and RL for EV eco-routing.
>
---
#### [new 047] Shell-Type Soft Jig for Holding Objects during Disassembly
- **分类: cs.RO**

- **简介: 该论文提出一种壳型软夹具，用于机器人拆卸任务，解决传统夹具易损、适应性差的问题。通过气球式夹持机制，实现稳定抓取，减少对高精度感知和轨迹规划的依赖，并通过实验验证其可行性与局限性。**

- **链接: [http://arxiv.org/pdf/2509.13802v1](http://arxiv.org/pdf/2509.13802v1)**

> **作者:** Takuya Kiyokawa; Ryunosuke Takebayashi; Kensuke Harada
>
> **备注:** 6 pages, 8 figures
>
> **摘要:** This study addresses a flexible holding tool for robotic disassembly. We propose a shell-type soft jig that securely and universally holds objects, mitigating the risk of component damage and adapting to diverse shapes while enabling soft fixation that is robust to recognition, planning, and control errors. The balloon-based holding mechanism ensures proper alignment and stable holding performance, thereby reducing the need for dedicated jig design, highly accurate perception, precise grasping, and finely tuned trajectory planning that are typically required with conventional fixtures. Our experimental results demonstrate the practical feasibility of the proposed jig through performance comparisons with a vise and a jamming-gripper-inspired soft jig. Tests on ten different objects further showed representative successes and failures, clarifying the jig's limitations and outlook.
>
---
#### [new 048] Dynamic Adaptive Legged Locomotion Policy via Decoupling Reaction Force Control and Gait Control
- **分类: cs.RO**

- **简介: 该论文属于腿部机器人运动控制任务，旨在解决仿真到现实的性能差异和环境不确定性问题。提出了一种解耦反应力与步态控制的框架，提升在线适应能力，实验证明其在多种复杂环境下有效。**

- **链接: [http://arxiv.org/pdf/2509.13737v1](http://arxiv.org/pdf/2509.13737v1)**

> **作者:** Renjie Wang; Shangke Lyu; Donglin Wang
>
> **摘要:** While Reinforcement Learning (RL) has achieved remarkable progress in legged locomotion control, it often suffers from performance degradation in out-of-distribution (OOD) conditions and discrepancies between the simulation and the real environments. Instead of mainly relying on domain randomization (DR) to best cover the real environments and thereby close the sim-to-real gap and enhance robustness, this work proposes an emerging decoupled framework that acquires fast online adaptation ability and mitigates the sim-to-real problems in unfamiliar environments by isolating stance-leg control and swing-leg control. Various simulation and real-world experiments demonstrate its effectiveness against horizontal force disturbances, uneven terrains, heavy and biased payloads, and sim-to-real gap.
>
---
#### [new 049] Reinforcement Learning for Robotic Insertion of Flexible Cables in Industrial Settings
- **分类: cs.RO**

- **简介: 该论文研究工业场景中柔性扁平电缆（FFC）的机器人插入任务，解决因电缆变形导致的亚毫米精度难题。提出基于强化学习的算法，结合仿真到现实迁移技术，利用基础模型SAM2和视觉语言模型实现零样本部署，减少训练时间与物理损伤风险。**

- **链接: [http://arxiv.org/pdf/2509.13731v1](http://arxiv.org/pdf/2509.13731v1)**

> **作者:** Jeongwoo Park; Seabin Lee; Changmin Park; Wonjong Lee; Changjoo Nam
>
> **摘要:** The industrial insertion of flexible flat cables (FFCs) into receptacles presents a significant challenge owing to the need for submillimeter precision when handling the deformable cables. In manufacturing processes, FFC insertion with robotic manipulators often requires laborious human-guided trajectory generation. While Reinforcement Learning (RL) offers a solution to automate this task without modeling complex properties of FFCs, the nondeterminism caused by the deformability of FFCs requires significant efforts and time on training. Moreover, training directly in a real environment is dangerous as industrial robots move fast and possess no safety measure. We propose an RL algorithm for FFC insertion that leverages a foundation model-based real-to-sim approach to reduce the training time and eliminate the risk of physical damages to robots and surroundings. Training is done entirely in simulation, allowing for random exploration without the risk of physical damages. Sim-to-real transfer is achieved through semantic segmentation masks which leave only those visual features relevant to the insertion tasks such as the geometric and spatial information of the cables and receptacles. To enhance generality, we use a foundation model, Segment Anything Model 2 (SAM2). To eleminate human intervention, we employ a Vision-Language Model (VLM) to automate the initial prompting of SAM2 to find segmentation masks. In the experiments, our method exhibits zero-shot capabilities, which enable direct deployments to real environments without fine-tuning.
>
---
#### [new 050] MetricNet: Recovering Metric Scale in Generative Navigation Policies
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决生成式导航策略中轨迹无尺度和短视问题。提出MetricNet预测航路点间距离，实现真实坐标定位，并集成至MetricNav中提升避障与导航性能。**

- **链接: [http://arxiv.org/pdf/2509.13965v1](http://arxiv.org/pdf/2509.13965v1)**

> **作者:** Abhijeet Nayak; Débora N. P. Oliveira; Samiran Gode; Cordelia Schmid; Wolfram Burgard
>
> **摘要:** Generative navigation policies have made rapid progress in improving end-to-end learned navigation. Despite their promising results, this paradigm has two structural problems. First, the sampled trajectories exist in an abstract, unscaled space without metric grounding. Second, the control strategy discards the full path, instead moving directly towards a single waypoint. This leads to short-sighted and unsafe actions, moving the robot towards obstacles that a complete and correctly scaled path would circumvent. To address these issues, we propose MetricNet, an effective add-on for generative navigation that predicts the metric distance between waypoints, grounding policy outputs in real-world coordinates. We evaluate our method in simulation with a new benchmarking framework and show that executing MetricNet-scaled waypoints significantly improves both navigation and exploration performance. Beyond simulation, we further validate our approach in real-world experiments. Finally, we propose MetricNav, which integrates MetricNet into a navigation policy to guide the robot away from obstacles while still moving towards the goal.
>
---
#### [new 051] GLIDE: A Coordinated Aerial-Ground Framework for Search and Rescue in Unknown Environments
- **分类: cs.RO**

- **简介: 该论文提出GLIDE框架，结合无人机与地面车实现未知环境下的搜救任务。通过分工协作，提升目标定位与避障效率，解决复杂地形中快速救援的问题。**

- **链接: [http://arxiv.org/pdf/2509.14210v1](http://arxiv.org/pdf/2509.14210v1)**

> **作者:** Seth Farrell; Chenghao Li; Hongzhan Yu; Hesam Mojtahedi; Sicun Gao; Henrik I. Christensen
>
> **摘要:** We present a cooperative aerial-ground search-and-rescue (SAR) framework that pairs two unmanned aerial vehicles (UAVs) with an unmanned ground vehicle (UGV) to achieve rapid victim localization and obstacle-aware navigation in unknown environments. We dub this framework Guided Long-horizon Integrated Drone Escort (GLIDE), highlighting the UGV's reliance on UAV guidance for long-horizon planning. In our framework, a goal-searching UAV executes real-time onboard victim detection and georeferencing to nominate goals for the ground platform, while a terrain-scouting UAV flies ahead of the UGV's planned route to provide mid-level traversability updates. The UGV fuses aerial cues with local sensing to perform time-efficient A* planning and continuous replanning as information arrives. Additionally, we present a hardware demonstration (using a GEM e6 golf cart as the UGV and two X500 UAVs) to evaluate end-to-end SAR mission performance and include simulation ablations to assess the planning stack in isolation from detection. Empirical results demonstrate that explicit role separation across UAVs, coupled with terrain scouting and guided planning, improves reach time and navigation safety in time-critical SAR missions.
>
---
#### [new 052] Trajectory Tracking with Reachability-Guided Quadratic Programming and Freeze-Resume
- **分类: cs.RO**

- **简介: 该论文提出一种轨迹跟踪方法，解决机器人在人类或物体干扰下安全暂停并恢复跟踪的问题。通过离线可达性检查和在线二次规划实现高效跟踪与扰动抑制，无需重新规划路径。**

- **链接: [http://arxiv.org/pdf/2509.13501v1](http://arxiv.org/pdf/2509.13501v1)**

> **作者:** Hossein Gholampour; Logan E. Beaver
>
> **摘要:** Many robotic systems must follow planned paths yet pause safely and resume when people or objects intervene. We present an output-space method for systems whose tracked output can be feedback-linearized to a double integrator (e.g., manipulators). The approach has two parts. Offline, we perform a pre-run reachability check to verify that the motion plan respects speed and acceleration magnitude limits. Online, we apply a quadratic program to track the motion plan under the same limits. We use a one-step reachability test to bound the maximum disturbance the system is capable of rejecting. When the state coincides with the reference path we recover perfect tracking in the deterministic case, and we correct errors using a KKT-inspired weight. We demonstrate that safety stops and unplanned deviations are handled efficiently, and the system returns to the motion plan without replanning. We demonstrate our system's improved performance over pure pursuit in simulation.
>
---
#### [new 053] SEG-Parking: Towards Safe, Efficient, and Generalizable Autonomous Parking via End-to-End Offline Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文提出SEG-Parking，通过端到端离线强化学习解决自动驾驶停车任务中的安全、效率与泛化问题。构建专用数据集，预训练目标条件状态编码器，并优化带保守正则化的离线策略，在CARLA仿真中验证其优越性能。**

- **链接: [http://arxiv.org/pdf/2509.13956v1](http://arxiv.org/pdf/2509.13956v1)**

> **作者:** Zewei Yang; Zengqi Peng; Jun Ma
>
> **摘要:** Autonomous parking is a critical component for achieving safe and efficient urban autonomous driving. However, unstructured environments and dynamic interactions pose significant challenges to autonomous parking tasks. To address this problem, we propose SEG-Parking, a novel end-to-end offline reinforcement learning (RL) framework to achieve interaction-aware autonomous parking. Notably, a specialized parking dataset is constructed for parking scenarios, which include those without interference from the opposite vehicle (OV) and complex ones involving interactions with the OV. Based on this dataset, a goal-conditioned state encoder is pretrained to map the fused perception information into the latent space. Then, an offline RL policy is optimized with a conservative regularizer that penalizes out-of-distribution actions. Extensive closed-loop experiments are conducted in the high-fidelity CARLA simulator. Comparative results demonstrate the superior performance of our framework with the highest success rate and robust generalization to out-of-distribution parking scenarios. The related dataset and source code will be made publicly available after the paper is accepted.
>
---
#### [new 054] FlightDiffusion: Revolutionising Autonomous Drone Training with Diffusion Models Generating FPV Video
- **分类: cs.RO**

- **简介: 论文提出FlightDiffusion框架，利用扩散模型生成FPV视频用于训练自主无人机。旨在解决真实数据成本高的问题，通过生成多样化轨迹和状态-动作对，提升策略学习与数据集规模，实现优越的导航性能与仿真到现实的迁移。**

- **链接: [http://arxiv.org/pdf/2509.14082v1](http://arxiv.org/pdf/2509.14082v1)**

> **作者:** Valerii Serpiva; Artem Lykov; Faryal Batool; Vladislav Kozlovskiy; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **备注:** Submitted to conference
>
> **摘要:** We present FlightDiffusion, a diffusion-model-based framework for training autonomous drones from first-person view (FPV) video. Our model generates realistic video sequences from a single frame, enriched with corresponding action spaces to enable reasoning-driven navigation in dynamic environments. Beyond direct policy learning, FlightDiffusion leverages its generative capabilities to synthesize diverse FPV trajectories and state-action pairs, facilitating the creation of large-scale training datasets without the high cost of real-world data collection. Our evaluation demonstrates that the generated trajectories are physically plausible and executable, with a mean position error of 0.25 m (RMSE 0.28 m) and a mean orientation error of 0.19 rad (RMSE 0.24 rad). This approach enables improved policy learning and dataset scalability, leading to superior performance in downstream navigation tasks. Results in simulated environments highlight enhanced robustness, smoother trajectory planning, and adaptability to unseen conditions. An ANOVA revealed no statistically significant difference between performance in simulation and reality (F(1, 16) = 0.394, p = 0.541), with success rates of M = 0.628 (SD = 0.162) and M = 0.617 (SD = 0.177), respectively, indicating strong sim-to-real transfer. The generated datasets provide a valuable resource for future UAV research. This work introduces diffusion-based reasoning as a promising paradigm for unifying navigation, action generation, and data synthesis in aerial robotics.
>
---
#### [new 055] PhysicalAgent: Towards General Cognitive Robotics with Foundation World Models
- **分类: cs.RO**

- **简介: 该论文提出PhysicalAgent框架，解决机器人通用操作任务中的失败恢复问题。通过视频生成与迭代执行，实现多模态、多平台下的高成功率操作，提升机器人鲁棒性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.13903v1](http://arxiv.org/pdf/2509.13903v1)**

> **作者:** Artem Lykov; Jeffrin Sam; Hung Khang Nguyen; Vladislav Kozlovskiy; Yara Mahmoud; Valerii Serpiva; Miguel Altamirano Cabrera; Mikhail Konenkov; Dzmitry Tsetserukou
>
> **备注:** submitted to IEEE conference
>
> **摘要:** We introduce PhysicalAgent, an agentic framework for robotic manipulation that integrates iterative reasoning, diffusion-based video generation, and closed-loop execution. Given a textual instruction, our method generates short video demonstrations of candidate trajectories, executes them on the robot, and iteratively re-plans in response to failures. This approach enables robust recovery from execution errors. We evaluate PhysicalAgent across multiple perceptual modalities (egocentric, third-person, and simulated) and robotic embodiments (bimanual UR3, Unitree G1 humanoid, simulated GR1), comparing against state-of-the-art task-specific baselines. Experiments demonstrate that our method consistently outperforms prior approaches, achieving up to 83% success on human-familiar tasks. Physical trials reveal that first-attempt success is limited (20-30%), yet iterative correction increases overall success to 80% across platforms. These results highlight the potential of video-based generative reasoning for general-purpose robotic manipulation and underscore the importance of iterative execution for recovering from initial failures. Our framework paves the way for scalable, adaptable, and robust robot control.
>
---
#### [new 056] Using Petri Nets for Context-Adaptive Robot Explanations
- **分类: cs.RO; I.6.0; A.0**

- **简介: 论文提出使用Petri网建模上下文信息，以实现机器人在人机交互中的自适应解释。属于人机交互任务，解决机器人通信自然性和透明性问题，通过形式化方法验证系统鲁棒性与灵活性。**

- **链接: [http://arxiv.org/pdf/2509.13861v1](http://arxiv.org/pdf/2509.13861v1)**

> **作者:** Görkem Kılınç Soylu; Neziha Akalin; Maria Riveiro
>
> **备注:** In proceedings of TRUST 2025 (arXiv:2509.11402), a workshop at IEEE RO-MAN 2025: https://www.ro-man2025.org/
>
> **摘要:** In human-robot interaction, robots must communicate in a natural and transparent manner to foster trust, which requires adapting their communication to the context. In this paper, we propose using Petri nets (PNs) to model contextual information for adaptive robot explanations. PNs provide a formal, graphical method for representing concurrent actions, causal dependencies, and system states, making them suitable for analyzing dynamic interactions between humans and robots. We demonstrate this approach through a scenario involving a robot that provides explanations based on contextual cues such as user attention and presence. Model analysis confirms key properties, including deadlock-freeness, context-sensitive reachability, boundedness, and liveness, showing the robustness and flexibility of PNs for designing and verifying context-adaptive explanations in human-robot interactions.
>
---
#### [new 057] StableTracker: Learning to Stably Track Target via Differentiable Simulation
- **分类: cs.RO**

- **简介: 论文提出StableTracker，一种基于学习的控制策略，用于无人机稳定跟踪目标。通过可微仿真训练，解决传统方法在硬件负载和误差累积下的性能问题，实现高精度、稳定的自主空中摄像功能。**

- **链接: [http://arxiv.org/pdf/2509.14147v1](http://arxiv.org/pdf/2509.14147v1)**

> **作者:** Fanxing Li; Shengyang Wang; Fangyu Sun; Shuyu Wu; Dexin Zuo; Wenxian Yu; Danping Zou
>
> **摘要:** FPV object tracking methods heavily rely on handcraft modular designs, resulting in hardware overload and cumulative error, which seriously degrades the tracking performance, especially for rapidly accelerating or decelerating targets. To address these challenges, we present \textbf{StableTracker}, a learning-based control policy that enables quadrotors to robustly follow the moving target from arbitrary perspectives. The policy is trained using backpropagation-through-time via differentiable simulation, allowing the quadrotor to maintain the target at the center of the visual field in both horizontal and vertical directions, while keeping a fixed relative distance, thereby functioning as an autonomous aerial camera. We compare StableTracker against both state-of-the-art traditional algorithms and learning baselines. Simulation experiments demonstrate that our policy achieves superior accuracy, stability and generalization across varying safe distances, trajectories, and target velocities. Furthermore, a real-world experiment on a quadrotor with an onboard computer validated practicality of the proposed approach.
>
---
#### [new 058] CDFlow: Generative Gradient Flows for Configuration Space Distance Fields via Neural ODEs
- **分类: cs.RO**

- **简介: 论文提出CDFlow框架，利用神经ODE学习配置空间距离场，解决高自由度机器人运动规划中CDF的梯度模糊和几何失真问题，提升规划效率与轨迹质量。**

- **链接: [http://arxiv.org/pdf/2509.13771v1](http://arxiv.org/pdf/2509.13771v1)**

> **作者:** Mengzhu Li; Yunyu Zhou; He Ying; F. Richard Yu
>
> **摘要:** Signed Distance Fields (SDFs) are a fundamental representation in robot motion planning. Their configuration-space counterpart, the Configuration Space Distance Field (CDF), directly encodes distances in joint space, offering a unified representation for optimization and control. However, existing CDF formulations face two major challenges in high-degree-of-freedom (DoF) robots: (1) they effectively return only a single nearest collision configuration, neglecting the multi-modal nature of minimal-distance collision configurations and leading to gradient ambiguity; and (2) they rely on sparse sampling of the collision boundary, which often fails to identify the true closest configurations, producing oversmoothed approximations and geometric distortion in high-dimensional spaces. We propose CDFlow, a novel framework that addresses these limitations by learning a continuous flow in configuration space via Neural Ordinary Differential Equations (Neural ODEs). We redefine the problem from finding a single nearest point to modeling the distribution of minimal-distance collision configurations. We also introduce an adaptive refinement sampling strategy to generate high-fidelity training data for this distribution. The resulting Neural ODE implicitly models this multi-modal distribution and produces a smooth, consistent gradient field-derived as the expected direction towards the distribution-that mitigates gradient ambiguity and preserves sharp geometric features. Extensive experiments on high-DoF motion planning tasks demonstrate that CDFlow significantly improves planning efficiency, trajectory quality, and robustness compared to existing CDF-based methods, enabling more robust and efficient planning for collision-aware robots in complex environments.
>
---
#### [new 059] EZREAL: Enhancing Zero-Shot Outdoor Robot Navigation toward Distant Targets under Varying Visibility
- **分类: cs.RO**

- **简介: 论文提出EZREAL系统，解决零样本户外机器人远距离目标导航问题。通过多尺度图像融合与区域显著性分析，在遮挡和低可见度下实现稳定导航，提升任务成功率。**

- **链接: [http://arxiv.org/pdf/2509.13720v1](http://arxiv.org/pdf/2509.13720v1)**

> **作者:** Tianle Zeng; Jianwei Peng; Hanjing Ye; Guangcheng Chen; Senzi Luo; Hong Zhang
>
> **备注:** Page:https://tianlezeng.github.io/EzReal/
>
> **摘要:** Zero-shot object navigation (ZSON) in large-scale outdoor environments faces many challenges; we specifically address a coupled one: long-range targets that reduce to tiny projections and intermittent visibility due to partial or complete occlusion. We present a unified, lightweight closed-loop system built on an aligned multi-scale image tile hierarchy. Through hierarchical target-saliency fusion, it summarizes localized semantic contrast into a stable coarse-layer regional saliency that provides the target direction and indicates target visibility. This regional saliency supports visibility-aware heading maintenance through keyframe memory, saliency-weighted fusion of historical headings, and active search during temporary invisibility. The system avoids whole-image rescaling, enables deterministic bottom-up aggregation, supports zero-shot navigation, and runs efficiently on a mobile robot. Across simulation and real-world outdoor trials, the system detects semantic targets beyond 150m, maintains a correct heading through visibility changes with 82.6% probability, and improves overall task success by 17.5% compared with the SOTA methods, demonstrating robust ZSON toward distant and intermittently observable targets.
>
---
#### [new 060] Whole-body Motion Control of an Omnidirectional Wheel-Legged Mobile Manipulator via Contact-Aware Dynamic Optimization
- **分类: cs.RO**

- **简介: 论文提出一种全向轮腿机器人全身运动控制方法，解决轮腿机构冗余自由度、复杂接触动力学及运动与操作协调问题。设计了具备灵巧机械臂的机器人平台，建立接触感知动态优化框架，实现敏捷移动与精准操作，适用于工业自动化等场景。**

- **链接: [http://arxiv.org/pdf/2509.14010v1](http://arxiv.org/pdf/2509.14010v1)**

> **作者:** Zong Chen; Shaoyang Li; Ben Liu; Min Li; Zhouping Yin; Yiqun Li
>
> **摘要:** Wheel-legged robots with integrated manipulators hold great promise for mobile manipulation in logistics, industrial automation, and human-robot collaboration. However, unified control of such systems remains challenging due to the redundancy in degrees of freedom, complex wheel-ground contact dynamics, and the need for seamless coordination between locomotion and manipulation. In this work, we present the design and whole-body motion control of an omnidirectional wheel-legged quadrupedal robot equipped with a dexterous manipulator. The proposed platform incorporates independently actuated steering modules and hub-driven wheels, enabling agile omnidirectional locomotion with high maneuverability in structured environments. To address the challenges of contact-rich interaction, we develop a contact-aware whole-body dynamic optimization framework that integrates point-contact modeling for manipulation with line-contact modeling for wheel-ground interactions. A warm-start strategy is introduced to accelerate online optimization, ensuring real-time feasibility for high-dimensional control. Furthermore, a unified kinematic model tailored for the robot's 4WIS-4WID actuation scheme eliminates the need for mode switching across different locomotion strategies, improving control consistency and robustness. Simulation and experimental results validate the effectiveness of the proposed framework, demonstrating agile terrain traversal, high-speed omnidirectional mobility, and precise manipulation under diverse scenarios, underscoring the system's potential for factory automation, urban logistics, and service robotics in semi-structured environments.
>
---
#### [new 061] DREAM: Domain-aware Reasoning for Efficient Autonomous Underwater Monitoring
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出DREAM框架，用于长期水下监测任务，解决机器人自主探索与环境感知问题。通过VLM引导，提升目标搜索效率，减少时间与步骤，在牡蛎监测和沉船场景中表现优于基线模型。**

- **链接: [http://arxiv.org/pdf/2509.13666v1](http://arxiv.org/pdf/2509.13666v1)**

> **作者:** Zhenqi Wu; Abhinav Modi; Angelos Mavrogiannis; Kaustubh Joshi; Nikhil Chopra; Yiannis Aloimonos; Nare Karapetyan; Ioannis Rekleitis; Xiaomin Lin
>
> **备注:** submitted to ICRA 2026
>
> **摘要:** The ocean is warming and acidifying, increasing the risk of mass mortality events for temperature-sensitive shellfish such as oysters. This motivates the development of long-term monitoring systems. However, human labor is costly and long-duration underwater work is highly hazardous, thus favoring robotic solutions as a safer and more efficient option. To enable underwater robots to make real-time, environment-aware decisions without human intervention, we must equip them with an intelligent "brain." This highlights the need for persistent,wide-area, and low-cost benthic monitoring. To this end, we present DREAM, a Vision Language Model (VLM)-guided autonomy framework for long-term underwater exploration and habitat monitoring. The results show that our framework is highly efficient in finding and exploring target objects (e.g., oysters, shipwrecks) without prior location information. In the oyster-monitoring task, our framework takes 31.5% less time than the previous baseline with the same amount of oysters. Compared to the vanilla VLM, it uses 23% fewer steps while covering 8.88% more oysters. In shipwreck scenes, our framework successfully explores and maps the wreck without collisions, requiring 27.5% fewer steps than the vanilla model and achieving 100% coverage, while the vanilla model achieves 60.23% average coverage in our shipwreck environments.
>
---
#### [new 062] UltraHiT: A Hierarchical Transformer Architecture for Generalizable Internal Carotid Artery Robotic Ultrasonography
- **分类: cs.RO**

- **简介: 该论文提出UltraHiT模型，用于解决颈内动脉（ICA）机器人超声定位难题。通过分层Transformer架构，结合高、低级模块处理个体差异，实现对未见个体的95%定位成功率，属于医学影像导航任务。**

- **链接: [http://arxiv.org/pdf/2509.13832v1](http://arxiv.org/pdf/2509.13832v1)**

> **作者:** Teng Wang; Haojun Jiang; Yuxuan Wang; Zhenguo Sun; Xiangjie Yan; Xiang Li; Gao Huang
>
> **摘要:** Carotid ultrasound is crucial for the assessment of cerebrovascular health, particularly the internal carotid artery (ICA). While previous research has explored automating carotid ultrasound, none has tackled the challenging ICA. This is primarily due to its deep location, tortuous course, and significant individual variations, which greatly increase scanning complexity. To address this, we propose a Hierarchical Transformer-based decision architecture, namely UltraHiT, that integrates high-level variation assessment with low-level action decision. Our motivation stems from conceptualizing individual vascular structures as morphological variations derived from a standard vascular model. The high-level module identifies variation and switches between two low-level modules: an adaptive corrector for variations, or a standard executor for normal cases. Specifically, both the high-level module and the adaptive corrector are implemented as causal transformers that generate predictions based on the historical scanning sequence. To ensure generalizability, we collected the first large-scale ICA scanning dataset comprising 164 trajectories and 72K samples from 28 subjects of both genders. Based on the above innovations, our approach achieves a 95% success rate in locating the ICA on unseen individuals, outperforming baselines and demonstrating its effectiveness. Our code will be released after acceptance.
>
---
#### [new 063] Semantic 3D Reconstructions with SLAM for Central Airway Obstruction
- **分类: cs.RO; cs.CV**

- **简介: 论文提出一种结合SLAM与语义分割的实时3D重建方法，用于中央气道阻塞的手术导航。任务是实现高精度、带语义标注的气道重建，解决传统手术风险高的问题。方法融合DROID-SLAM与分割模型，提升手术自动化水平。**

- **链接: [http://arxiv.org/pdf/2509.13541v1](http://arxiv.org/pdf/2509.13541v1)**

> **作者:** Ayberk Acar; Fangjie Li; Hao Li; Lidia Al-Zogbi; Kanyifeechukwu Jane Oguine; Susheela Sharma Stern; Jesse F. d'Almeida; Robert J. Webster III; Ipek Oguz; Jie Ying Wu
>
> **备注:** 5 pages, 2 figures, 1 table
>
> **摘要:** Central airway obstruction (CAO) is a life-threatening condition with increasing incidence, caused by tumors in and outside of the airway. Traditional treatment methods such as bronchoscopy and electrocautery can be used to remove the tumor completely; however, these methods carry a high risk of complications. Recent advances allow robotic interventions with lesser risk. The combination of robot interventions with scene understanding and mapping also opens up the possibilities for automation. We present a novel pipeline that enables real-time, semantically informed 3D reconstructions of the central airway using monocular endoscopic video. Our approach combines DROID-SLAM with a segmentation model trained to identify obstructive tissues. The SLAM module reconstructs the 3D geometry of the airway in real time, while the segmentation masks guide the annotation of obstruction regions within the reconstructed point cloud. To validate our pipeline, we evaluate the reconstruction quality using ex vivo models. Qualitative and quantitative results show high similarity between ground truth CT scans and the 3D reconstructions (0.62 mm Chamfer distance). By integrating segmentation directly into the SLAM workflow, our system produces annotated 3D maps that highlight clinically relevant regions in real time. High-speed capabilities of the pipeline allows quicker reconstructions compared to previous work, reflecting the surgical scene more accurately. To the best of our knowledge, this is the first work to integrate semantic segmentation with real-time monocular SLAM for endoscopic CAO scenarios. Our framework is modular and can generalize to other anatomies or procedures with minimal changes, offering a promising step toward autonomous robotic interventions.
>
---
#### [new 064] CLAW: A Vision-Language-Action Framework for Weight-Aware Robotic Grasping
- **分类: cs.RO; 68T40**

- **简介: 该论文提出CLAW框架，解决机器人抓取中重量感知与动作控制的结合问题。通过分离条件评估与动作生成，利用CLIP模型生成指令提示，并结合视觉语言动作策略实现精准控制，提升抓取任务性能。**

- **链接: [http://arxiv.org/pdf/2509.14143v1](http://arxiv.org/pdf/2509.14143v1)**

> **作者:** Zijian An; Ran Yang; Yiming Feng; Lifeng Zhou
>
> **备注:** 8 pages, 5 figures, 1 table
>
> **摘要:** Vision-language-action (VLA) models have recently emerged as a promising paradigm for robotic control, enabling end-to-end policies that ground natural language instructions into visuomotor actions. However, current VLAs often struggle to satisfy precise task constraints, such as stopping based on numeric thresholds, since their observation-to-action mappings are implicitly shaped by training data and lack explicit mechanisms for condition monitoring. In this work, we propose CLAW (CLIP-Language-Action for Weight), a framework that decouples condition evaluation from action generation. CLAW leverages a fine-tuned CLIP model as a lightweight prompt generator, which continuously monitors the digital readout of a scale and produces discrete directives based on task-specific weight thresholds. These prompts are then consumed by $\pi_0$, a flow-based VLA policy, which integrates the prompts with multi-view camera observations to produce continuous robot actions. This design enables CLAW to combine symbolic weight reasoning with high-frequency visuomotor control. We validate CLAW on three experimental setups: single-object grasping and mixed-object tasks requiring dual-arm manipulation. Across all conditions, CLAW reliably executes weight-aware behaviors and outperforms both raw-$\pi_0$ and fine-tuned $\pi_0$ models. We have uploaded the videos as supplementary materials.
>
---
#### [new 065] Multi-Attacker Single-Defender Target Defense in Conical Environments
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 论文研究平面锥形环境中单防御者对抗多个攻击者的任务，解决防御者捕获攻击者的问题。通过分析攻击者和防御者的策略，推导均衡策略并验证理论结果。**

- **链接: [http://arxiv.org/pdf/2509.13564v1](http://arxiv.org/pdf/2509.13564v1)**

> **作者:** Arman Pourghorban; Dipankar Maity
>
> **摘要:** We consider a variant of the target defense problem in a planar conical environment where a single defender is tasked to capture a sequence of incoming attackers. The attackers' objective is to breach the target boundary without being captured by the defender. As soon as the current attacker breaches the target or gets captured by the defender, the next attacker appears at the boundary of the environment and moves radially toward the target with maximum speed. Therefore, the defender's final location at the end of the current game becomes its initial location for the next game. The attackers pick strategies that are advantageous for the current as well as for future engagements between the defender and the remaining attackers. The attackers have their own sensors with limited range, using which they can perfectly detect if the defender is within their sensing range. We derive equilibrium strategies for all the players to optimize the capture percentage using the notions of capture distribution. Finally, the theoretical results are verified through numerical examples using Monte Carlo type random trials of experiments.
>
---
#### [new 066] Dynamic Aware: Adaptive Multi-Mode Out-of-Distribution Detection for Trajectory Prediction in Autonomous Vehicles
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶轨迹预测中的分布外检测任务，旨在解决模型在真实场景中因数据分布偏移导致的预测失效问题。提出一种自适应多模式检测框架，通过建模预测误差的动态模式，提升检测效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.13577v1](http://arxiv.org/pdf/2509.13577v1)**

> **作者:** Tongfei Guo; Lili Su
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Trajectory prediction is central to the safe and seamless operation of autonomous vehicles (AVs). In deployment, however, prediction models inevitably face distribution shifts between training data and real-world conditions, where rare or underrepresented traffic scenarios induce out-of-distribution (OOD) cases. While most prior OOD detection research in AVs has concentrated on computer vision tasks such as object detection and segmentation, trajectory-level OOD detection remains largely underexplored. A recent study formulated this problem as a quickest change detection (QCD) task, providing formal guarantees on the trade-off between detection delay and false alarms [1]. Building on this foundation, we propose a new framework that introduces adaptive mechanisms to achieve robust detection in complex driving environments. Empirical analysis across multiple real-world datasets reveals that prediction errors -- even on in-distribution samples -- exhibit mode-dependent distributions that evolve over time with dataset-specific dynamics. By explicitly modeling these error modes, our method achieves substantial improvements in both detection delay and false alarm rates. Comprehensive experiments on established trajectory prediction benchmarks show that our framework significantly outperforms prior UQ- and vision-based OOD approaches in both accuracy and computational efficiency, offering a practical path toward reliable, driving-aware autonomy.
>
---
#### [new 067] Explainable AI-Enhanced Supervisory Control for High-Precision Spacecraft Formation
- **分类: astro-ph.IM; cs.AI; cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种结合AI与监督控制的系统，用于提升航天器编队任务的精度。通过深度学习和优化算法，实现高精度X射线观测任务的参数优化与能量效率提升，解决动态不确定性和干扰问题。**

- **链接: [http://arxiv.org/pdf/2509.13331v1](http://arxiv.org/pdf/2509.13331v1)**

> **作者:** Reza Pirayeshshirazinezhad
>
> **摘要:** We use artificial intelligence (AI) and supervisory adaptive control systems to plan and optimize the mission of precise spacecraft formation. Machine learning and robust control enhance the efficiency of spacecraft precision formation of the Virtual Telescope for X-ray Observation (VTXO) space mission. VTXO is a precise formation of two separate spacecraft making a virtual telescope with a one-kilometer focal length. One spacecraft carries the lens and the other spacecraft holds the camera to observe high-energy space objects in the X-ray domain with 55 milli-arcsecond angular resolution accuracy. Timed automata for supervisory control, Monte Carlo simulations for stability and robustness evaluation, and integration of deep neural networks for optimal estimation of mission parameters, satisfy the high precision mission criteria. We integrate deep neural networks with a constrained, non-convex dynamic optimization pipeline to predict optimal mission parameters, ensuring precision mission criteria are met. AI framework provides explainability by predicting the resulting energy consumption and mission error for a given set of mission parameters. It allows for transparent, justifiable, and real-time trade-offs, a capability not present in traditional adaptive controllers. The results show reductions in energy consumption and improved mission accuracy, demonstrating the capability of the system to address dynamic uncertainties and disturbances.
>
---
#### [new 068] A Generalization of CLAP from 3D Localization to Image Processing, A Connection With RANSAC & Hough Transforms
- **分类: cs.CV; cs.RO**

- **简介: 论文将CLAP算法从2D定位推广到3D定位与图像拼接，用于处理噪声和不确定性。该工作提出了一种基于聚类的鲁棒方法，并探讨了其与RANSAC和霍夫变换的关系，适用于多个领域。**

- **链接: [http://arxiv.org/pdf/2509.13605v1](http://arxiv.org/pdf/2509.13605v1)**

> **作者:** Ruochen Hou; Gabriel I. Fernandez; Alex Xu; Dennis W. Hong
>
> **摘要:** In previous work, we introduced a 2D localization algorithm called CLAP, Clustering to Localize Across $n$ Possibilities, which was used during our championship win in RoboCup 2024, an international autonomous humanoid soccer competition. CLAP is particularly recognized for its robustness against outliers, where clustering is employed to suppress noise and mitigate against erroneous feature matches. This clustering-based strategy provides an alternative to traditional outlier rejection schemes such as RANSAC, in which candidates are validated by reprojection error across all data points. In this paper, CLAP is extended to a more general framework beyond 2D localization, specifically to 3D localization and image stitching. We also show how CLAP, RANSAC, and Hough transforms are related. The generalization of CLAP is widely applicable to many different fields and can be a useful tool to deal with noise and uncertainty.
>
---
#### [new 069] Agentic UAVs: LLM-Driven Autonomy with Integrated Tool-Calling and Cognitive Reasoning
- **分类: cs.AI; cs.RO; 68T07, 68T40, 68T42; I.2.9; I.2.11; I.2.8; I.2.10**

- **简介: 论文提出Agentic UAVs框架，通过LLM驱动实现无人机自主决策与系统集成。旨在提升无人机在复杂任务中的适应性与智能水平，解决现有系统缺乏认知推理与生态整合的问题，实验验证其在搜救场景中的优越性能。**

- **链接: [http://arxiv.org/pdf/2509.13352v1](http://arxiv.org/pdf/2509.13352v1)**

> **作者:** Anis Koubaa; Khaled Gabr
>
> **备注:** 14 pages, 1 figure
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are increasingly deployed in defense, surveillance, and disaster response, yet most systems remain confined to SAE Level 2--3 autonomy. Their reliance on rule-based control and narrow AI restricts adaptability in dynamic, uncertain missions. Existing UAV frameworks lack context-aware reasoning, autonomous decision-making, and ecosystem-level integration; critically, none leverage Large Language Model (LLM) agents with tool-calling for real-time knowledge access. This paper introduces the Agentic UAVs framework, a five-layer architecture (Perception, Reasoning, Action, Integration, Learning) that augments UAVs with LLM-driven reasoning, database querying, and third-party system interaction. A ROS2 and Gazebo-based prototype integrates YOLOv11 object detection with GPT-4 reasoning and local Gemma-3 deployment. In simulated search-and-rescue scenarios, agentic UAVs achieved higher detection confidence (0.79 vs. 0.72), improved person detection rates (91% vs. 75%), and markedly increased action recommendation (92% vs. 4.5%). These results confirm that modest computational overhead enables qualitatively new levels of autonomy and ecosystem integration.
>
---
#### [new 070] MapAnything: Universal Feed-Forward Metric 3D Reconstruction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 论文提出MapAnything，一种基于Transformer的前馈模型，统一处理多视角图像及几何输入，直接回归度量三维场景和相机参数，解决多种3D视觉任务，如结构从运动、深度估计等，实现高效联合训练。**

- **链接: [http://arxiv.org/pdf/2509.13414v1](http://arxiv.org/pdf/2509.13414v1)**

> **作者:** Nikhil Keetha; Norman Müller; Johannes Schönberger; Lorenzo Porzi; Yuchen Zhang; Tobias Fischer; Arno Knapitsch; Duncan Zauss; Ethan Weber; Nelson Antunes; Jonathon Luiten; Manuel Lopez-Antequera; Samuel Rota Bulò; Christian Richardt; Deva Ramanan; Sebastian Scherer; Peter Kontschieder
>
> **备注:** Project Page: https://map-anything.github.io/
>
> **摘要:** We introduce MapAnything, a unified transformer-based feed-forward model that ingests one or more images along with optional geometric inputs such as camera intrinsics, poses, depth, or partial reconstructions, and then directly regresses the metric 3D scene geometry and cameras. MapAnything leverages a factored representation of multi-view scene geometry, i.e., a collection of depth maps, local ray maps, camera poses, and a metric scale factor that effectively upgrades local reconstructions into a globally consistent metric frame. Standardizing the supervision and training across diverse datasets, along with flexible input augmentation, enables MapAnything to address a broad range of 3D vision tasks in a single feed-forward pass, including uncalibrated structure-from-motion, calibrated multi-view stereo, monocular depth estimation, camera localization, depth completion, and more. We provide extensive experimental analyses and model ablations demonstrating that MapAnything outperforms or matches specialist feed-forward models while offering more efficient joint training behavior, thus paving the way toward a universal 3D reconstruction backbone.
>
---
## 更新

#### [replaced 001] Fluidically Innervated Lattices Make Versatile and Durable Tactile Sensors
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.21225v2](http://arxiv.org/pdf/2507.21225v2)**

> **作者:** Annan Zhang; Miguel Flores-Acton; Andy Yu; Anshul Gupta; Maggie Yao; Daniela Rus
>
> **备注:** Accepted for publication in the proceedings of the 2025 International Symposium on Experimental Robotics (ISER)
>
> **摘要:** Tactile sensing plays a fundamental role in enabling robots to navigate dynamic and unstructured environments, particularly in applications such as delicate object manipulation, surface exploration, and human-robot interaction. In this paper, we introduce a passive soft robotic fingertip with integrated tactile sensing, fabricated using a 3D-printed elastomer lattice with embedded air channels. This sensorization approach, termed fluidic innervation, transforms the lattice into a tactile sensor by detecting pressure changes within sealed air channels, providing a simple yet robust solution to tactile sensing in robotics. Unlike conventional methods that rely on complex materials or designs, fluidic innervation offers a simple, scalable, single-material fabrication process. We characterize the sensors' response, develop a geometric model to estimate tip displacement, and train a neural network to accurately predict contact location and contact force. Additionally, we integrate the fingertip with an admittance controller to emulate spring-like behavior, demonstrate its capability for environment exploration through tactile feedback, and validate its durability under high impact and cyclic loading conditions. This tactile sensing technique offers advantages in terms of simplicity, adaptability, and durability and opens up new opportunities for versatile robotic manipulation.
>
---
#### [replaced 002] Conformal Temporal Logic Planning using Large Language Models
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2309.10092v5](http://arxiv.org/pdf/2309.10092v5)**

> **作者:** Jun Wang; Jiaming Tong; Kaiyuan Tan; Yevgeniy Vorobeychik; Yiannis Kantaros
>
> **备注:** accepted by ACM Transactions on Cyber-Physical Systems
>
> **摘要:** This paper addresses planning problems for mobile robots. We consider missions that require accomplishing multiple high-level sub-tasks, expressed in natural language (NL), in a temporal and logical order. To formally define the mission, we treat these sub-tasks as atomic predicates in a Linear Temporal Logic (LTL) formula. We refer to this task specification framework as LTL-NL. Our goal is to design plans, defined as sequences of robot actions, accomplishing LTL-NL tasks. This action planning problem cannot be solved directly by existing LTL planners because of the NL nature of atomic predicates. To address it, we propose HERACLEs, a hierarchical neuro-symbolic planner that relies on a novel integration of (i) existing symbolic planners generating high-level task plans determining the order at which the NL sub-tasks should be accomplished; (ii) pre-trained Large Language Models (LLMs) to design sequences of robot actions based on these task plans; and (iii) conformal prediction acting as a formal interface between (i) and (ii) and managing uncertainties due to LLM imperfections. We show, both theoretically and empirically, that HERACLEs can achieve user-defined mission success rates. Finally, we provide comparative experiments demonstrating that HERACLEs outperforms LLM-based planners that require the mission to be defined solely using NL. Additionally, we present examples demonstrating that our approach enhances user-friendliness compared to conventional symbolic approaches.
>
---
#### [replaced 003] NavMoE: Hybrid Model- and Learning-based Traversability Estimation for Local Navigation via Mixture of Experts
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.12747v2](http://arxiv.org/pdf/2509.12747v2)**

> **作者:** Botao He; Amir Hossein Shahidzadeh; Yu Chen; Jiayi Wu; Tianrui Guan; Guofei Chen; Howie Choset; Dinesh Manocha; Glen Chou; Cornelia Fermuller; Yiannis Aloimonos
>
> **摘要:** This paper explores traversability estimation for robot navigation. A key bottleneck in traversability estimation lies in efficiently achieving reliable and robust predictions while accurately encoding both geometric and semantic information across diverse environments. We introduce Navigation via Mixture of Experts (NAVMOE), a hierarchical and modular approach for traversability estimation and local navigation. NAVMOE combines multiple specialized models for specific terrain types, each of which can be either a classical model-based or a learning-based approach that predicts traversability for specific terrain types. NAVMOE dynamically weights the contributions of different models based on the input environment through a gating network. Overall, our approach offers three advantages: First, NAVMOE enables traversability estimation to adaptively leverage specialized approaches for different terrains, which enhances generalization across diverse and unseen environments. Second, our approach significantly improves efficiency with negligible cost of solution quality by introducing a training-free lazy gating mechanism, which is designed to minimize the number of activated experts during inference. Third, our approach uses a two-stage training strategy that enables the training for the gating networks within the hybrid MoE method that contains nondifferentiable modules. Extensive experiments show that NAVMOE delivers a better efficiency and performance balance than any individual expert or full ensemble across different domains, improving cross-domain generalization and reducing average computational cost by 81.2% via lazy gating, with less than a 2% loss in path quality.
>
---
#### [replaced 004] Disturbance-Aware Dynamical Trajectory Planning for Air-Land Bimodal Vehicles
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.05972v2](http://arxiv.org/pdf/2508.05972v2)**

> **作者:** Shaoting Liu; Wenshuai Yu; Bo Zhang; Shoubin Chen; Fei Ma; Zhou Liu; Qingquan Li
>
> **摘要:** Air-land bimodal vehicles provide a promising solution for navigating complex environments by combining the flexibility of aerial locomotion with the energy efficiency of ground mobility. However, planning dynamically feasible, smooth, collision-free, and energy-efficient trajectories remains challenging due to two key factors: 1) unknown dynamic disturbances in both aerial and terrestrial domains, and 2) the inherent complexity of managing bimodal dynamics with distinct constraint characteristics. This paper proposes a disturbance-aware motion-planning framework that addresses this challenge through real-time disturbance estimation and adaptive trajectory generation. The framework comprises two key components: 1) a disturbance-adaptive safety boundary adjustment mechanism that dynamically determines the feasible region of dynamic constraints for both air and land modes based on estimated disturbances via a disturbance observer, and 2) a constraint-adaptive bimodal motion planner that integrates disturbance-aware path searching to guide trajectories toward regions with reduced disturbances and B-spline-based trajectory optimization to refine trajectories within the established feasible constraint boundaries. Experimental validation on a self-developed air-land bimodal vehicle demonstrates substantial performance improvements across three representative disturbance scenarios, achieving an average 33.9% reduction in trajectory tracking error while still maintaining superior time-energy trade-offs compared to existing methods.
>
---
#### [replaced 005] AUTO-IceNav: A Local Navigation Strategy for Autonomous Surface Ships in Broken Ice Fields
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.17155v2](http://arxiv.org/pdf/2411.17155v2)**

> **作者:** Rodrigue de Schaetzen; Alexander Botros; Ninghan Zhong; Kevin Murrant; Robert Gash; Stephen L. Smith
>
> **备注:** 20 pages, 18 figures
>
> **摘要:** Ice conditions often require ships to reduce speed and deviate from their main course to avoid damage to the ship. In addition, broken ice fields are becoming the dominant ice conditions encountered in the Arctic, where the effects of collisions with ice are highly dependent on where contact occurs and on the particular features of the ice floes. In this paper, we present AUTO-IceNav, a framework for the autonomous navigation of ships operating in ice floe fields. Trajectories are computed in a receding-horizon manner, where we frequently replan given updated ice field data. During a planning step, we assume a nominal speed that is safe with respect to the current ice conditions, and compute a reference path. We formulate a novel cost function that minimizes the kinetic energy loss of the ship from ship-ice collisions and incorporate this cost as part of our lattice-based path planner. The solution computed by the lattice planning stage is then used as an initial guess in our proposed optimization-based improvement step, producing a locally optimal path. Extensive experiments were conducted both in simulation and in a physical testbed to validate our approach.
>
---
#### [replaced 006] TeraSim-World: Worldwide Safety-Critical Data Synthesis for End-to-End Autonomous Driving
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.13164v2](http://arxiv.org/pdf/2509.13164v2)**

> **作者:** Jiawei Wang; Haowei Sun; Xintao Yan; Shuo Feng; Jun Gao; Henry X. Liu
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Safe and scalable deployment of end-to-end (E2E) autonomous driving requires extensive and diverse data, particularly safety-critical events. Existing data are mostly generated from simulators with a significant sim-to-real gap or collected from on-road testing that is costly and unsafe. This paper presents TeraSim-World, an automated pipeline that synthesizes realistic and geographically diverse safety-critical data for E2E autonomous driving at anywhere in the world. Starting from an arbitrary location, TeraSim-World retrieves real-world maps and traffic demand from geospatial data sources. Then, it simulates agent behaviors from naturalistic driving datasets, and orchestrates diverse adversities to create corner cases. Informed by street views of the same location, it achieves photorealistic, geographically grounded sensor rendering via the frontier video generation model Cosmos-Drive. By bridging agent and sensor simulations, TeraSim-World provides a scalable and critical data synthesis framework for training and evaluation of E2E autonomous driving systems. Codes and videos are available at https://wjiawei.com/terasim-world-web/ .
>
---
#### [replaced 007] Body-terrain interaction affects large bump traversal of insects and legged robots
- **分类: physics.bio-ph; cs.RO; q-bio.QM**

- **链接: [http://arxiv.org/pdf/1911.02527v2](http://arxiv.org/pdf/1911.02527v2)**

> **作者:** Sean W. Gart; Chen Li
>
> **摘要:** Small animals and robots must often rapidly traverse large bump-like obstacles when moving through complex 3-D terrains, during which, in addition to leg-ground contact, their body inevitably comes into physical contact with the obstacles. However, we know little about the performance limits of large bump traversal and how body-terrain interaction affects traversal. To address these, we challenged the discoid cockroach and an open-loop six-legged robot to dynamically run into a large bump of varying height to discover the maximal traversal performance, and studied how locomotor modes and traversal performance are affected by body-terrain interaction. Remarkably, during rapid running, both the animal and the robot were capable of dynamically traversing a bump much higher than its hip height (up to 4 times the hip height for the animal and 3 times for the robot, respectively) at traversal speeds typical of running, with decreasing traversal probability with increasing bump height. A stability analysis using a novel locomotion energy landscape model explained why traversal was more likely when the animal or robot approached the bump with a low initial body yaw and a high initial body pitch, and why deflection was more likely otherwise. Inspired by these principles, we demonstrated a novel control strategy of active body pitching that increased the robot maximal traversable bump height by 75%. Our study is a major step in establishing the framework of locomotion energy landscapes to understand locomotion in complex 3-D terrains.
>
---
#### [replaced 008] Optimizing Active Perception for Learning Simultaneous Viewpoint Selection and Manipulation with Diffusion Policy
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.14615v3](http://arxiv.org/pdf/2409.14615v3)**

> **作者:** Xiatao Sun; Francis Fan; Yinxing Chen; Daniel Rakita
>
> **摘要:** Robotic manipulation tasks often rely on static cameras for perception, which can limit flexibility, particularly in scenarios like robotic surgery and cluttered environments where mounting static cameras is impractical. Ideally, robots could jointly learn a policy for dynamic viewpoint and manipulation. However, dynamic viewpoint control requires additional degrees of freedom and intricate coordination with manipulation, which results in more challenging policy learning than single-arm manipulation. To address this complexity, we propose an integrated learning framework that combines diffusion policy with a novel look-at inverse kinematics solver for active perception. Our framework helps better coordinating between perception and manipulation. It automatically optimizes camera orientation for viewpoint selection, while allowing the policy to focus on essential manipulation and positioning decisions. We demonstrate that our integrated approach achieves superior performance and learning efficiency compared to directly applying diffusion policies to configuration space or end-effector space with various rotation representations. Further analysis suggests that these performance differences are driven by inherent variations in the high-frequency components across different state-action spaces.
>
---
#### [replaced 009] TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.11839v2](http://arxiv.org/pdf/2509.11839v2)**

> **作者:** Jiacheng Liu; Pengxiang Ding; Qihang Zhou; Yuxuan Wu; Da Huang; Zimian Peng; Wei Xiao; Weinan Zhang; Lixin Yang; Cewu Lu; Donglin Wang
>
> **摘要:** Recent Vision-Language-Action models show potential to generalize across embodiments but struggle to quickly align with a new robot's action space when high-quality demonstrations are scarce, especially for bipedal humanoids. We present TrajBooster, a cross-embodiment framework that leverages abundant wheeled-humanoid data to boost bipedal VLA. Our key idea is to use end-effector trajectories as a morphology-agnostic interface. TrajBooster (i) extracts 6D dual-arm end-effector trajectories from real-world wheeled humanoids, (ii) retargets them in simulation to Unitree G1 with a whole-body controller trained via a heuristic-enhanced harmonized online DAgger to lift low-dimensional trajectory references into feasible high-dimensional whole-body actions, and (iii) forms heterogeneous triplets that couple source vision/language with target humanoid-compatible actions to post-pre-train a VLA, followed by only 10 minutes of teleoperation data collection on the target humanoid domain. Deployed on Unitree G1, our policy achieves beyond-tabletop household tasks, enabling squatting, cross-height manipulation, and coordinated whole-body motion with markedly improved robustness and generalization. Results show that TrajBooster allows existing wheeled-humanoid data to efficiently strengthen bipedal humanoid VLA performance, reducing reliance on costly same-embodiment data while enhancing action space understanding and zero-shot skill transfer capabilities. For more details, For more details, please refer to our \href{https://jiachengliu3.github.io/TrajBooster/}.
>
---
#### [replaced 010] Occupancy-aware Trajectory Planning for Autonomous Valet Parking in Uncertain Dynamic Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.09206v2](http://arxiv.org/pdf/2509.09206v2)**

> **作者:** Farhad Nawaz; Faizan M. Tariq; Sangjae Bae; David Isele; Avinash Singh; Nadia Figueroa; Nikolai Matni; Jovin D'sa
>
> **摘要:** Autonomous Valet Parking (AVP) requires planning under partial observability, where parking spot availability evolves as dynamic agents enter and exit spots. Existing approaches either rely only on instantaneous spot availability or make static assumptions, thereby limiting foresight and adaptability. We propose an approach that estimates probability of future spot occupancy by distinguishing initially vacant and occupied spots while leveraging nearby dynamic agent motion. We propose a probabilistic estimator that integrates partial, noisy observations from a limited Field-of-View, with the evolving uncertainty of unobserved spots. Coupled with the estimator, we design a strategy planner that balances goal-directed parking maneuvers with exploratory navigation based on information gain, and incorporates wait-and-go behaviors at promising spots. Through randomized simulations emulating large parking lots, we demonstrate that our framework significantly improves parking efficiency and trajectory smoothness over existing approaches, while maintaining safety margins.
>
---
#### [replaced 011] Hybrid Diffusion Policies with Projective Geometric Algebra for Efficient Robot Manipulation Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05695v2](http://arxiv.org/pdf/2507.05695v2)**

> **作者:** Xiatao Sun; Yuxuan Wang; Shuo Yang; Yinxing Chen; Daniel Rakita
>
> **摘要:** Diffusion policies are a powerful paradigm for robot learning, but their training is often inefficient. A key reason is that networks must relearn fundamental spatial concepts, such as translations and rotations, from scratch for every new task. To alleviate this redundancy, we propose embedding geometric inductive biases directly into the network architecture using Projective Geometric Algebra (PGA). PGA provides a unified algebraic framework for representing geometric primitives and transformations, allowing neural networks to reason about spatial structure more effectively. In this paper, we introduce hPGA-DP, a novel hybrid diffusion policy that capitalizes on these benefits. Our architecture leverages the Projective Geometric Algebra Transformer (P-GATr) as a state encoder and action decoder, while employing established U-Net or Transformer-based modules for the core denoising process. Through extensive experiments and ablation studies in both simulated and real-world environments, we demonstrate that hPGA-DP significantly improves task performance and training efficiency. Notably, our hybrid approach achieves substantially faster convergence compared to both standard diffusion policies and architectures that rely solely on P-GATr.
>
---
#### [replaced 012] Meta-Optimization and Program Search using Language Models for Task and Motion Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.03725v2](http://arxiv.org/pdf/2505.03725v2)**

> **作者:** Denis Shcherba; Eckart Cobo-Briesewitz; Cornelius V. Braun; Marc Toussaint
>
> **备注:** 8 pages main text, 11 pages appendix, accepted at the 9th Annual Conference on Robot Learning (CoRL 2025)
>
> **摘要:** Intelligent interaction with the real world requires robotic agents to jointly reason over high-level plans and low-level controls. Task and motion planning (TAMP) addresses this by combining symbolic planning and continuous trajectory generation. Recently, foundation model approaches to TAMP have presented impressive results, including fast planning times and the execution of natural language instructions. Yet, the optimal interface between high-level planning and low-level motion generation remains an open question: prior approaches are limited by either too much abstraction (e.g., chaining simplified skill primitives) or a lack thereof (e.g., direct joint angle prediction). Our method introduces a novel technique employing a form of meta-optimization to address these issues by: (i) using program search over trajectory optimization problems as an interface between a foundation model and robot control, and (ii) leveraging a zero-order method to optimize numerical parameters in the foundation model output. Results on challenging object manipulation and drawing tasks confirm that our proposed method improves over prior TAMP approaches.
>
---
#### [replaced 013] Brain Inspired Probabilistic Occupancy Grid Mapping with Vector Symbolic Architectures
- **分类: cs.RO; cs.ET**

- **链接: [http://arxiv.org/pdf/2408.09066v4](http://arxiv.org/pdf/2408.09066v4)**

> **作者:** Shay Snyder; Andrew Capodieci; David Gorsich; Maryam Parsa
>
> **摘要:** Real-time robotic systems require advanced perception, computation, and action capability. However, the main bottleneck in current autonomous systems is the trade-off between computational capability, energy efficiency and model determinism. World modeling, a key objective of many robotic systems, commonly uses occupancy grid mapping (OGM) as the first step towards building an end-to-end robotic system with perception, planning, autonomous maneuvering, and decision making capabilities. OGM divides the environment into discrete cells and assigns probability values to attributes such as occupancy and traversability. Existing methods fall into two categories: traditional methods and neural methods. Traditional methods rely on dense statistical calculations, while neural methods employ deep learning for probabilistic information processing. In this study, we propose a vector symbolic architecture-based OGM system (VSA-OGM) that retains the interpretability and stability of traditional methods with the improved computational efficiency of neural methods. Our approach, validated across multiple datasets, achieves similar accuracy to covariant traditional methods while reducing latency by approximately 45x and memory by 400x. Compared to invariant traditional methods, we see similar accuracy values while reducing latency by 5.5x. Moreover, we achieve up to 6x latency reductions compared to neural methods while eliminating the need for domain-specific model training. This work demonstrates the potential of vector symbolic architectures as a practical foundation for real-time probabilistic mapping in autonomous systems operating under strict computational and latency constraints.
>
---
#### [replaced 014] Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.08531v2](http://arxiv.org/pdf/2504.08531v2)**

> **作者:** Tommaso Galliena; Tommaso Apicella; Stefano Rosa; Pietro Morerio; Alessio Del Bue; Lorenzo Natale
>
> **备注:** 11 pages, 8 figures, 6 tables, code and test set annotations available at https://hsp-iit.github.io/embodied-captioning/
>
> **摘要:** We present a self-supervised method to improve an agent's abilities in describing arbitrary objects while actively exploring a generic environment. This is a challenging problem, as current models struggle to obtain coherent image captions due to different camera viewpoints and clutter. We propose a three-phase framework to fine-tune existing captioning models that enhances caption accuracy and consistency across views via a consensus mechanism. First, an agent explores the environment, collecting noisy image-caption pairs. Then, a consistent pseudo-caption for each object instance is distilled via consensus using a large language model. Finally, these pseudo-captions are used to fine-tune an off-the-shelf captioning model, with the addition of contrastive learning. We analyse the performance of the combination of captioning models, exploration policies, pseudo-labeling methods, and fine-tuning strategies, on our manually labeled test set. Results show that a policy can be trained to mine samples with higher disagreement compared to classical baselines. Our pseudo-captioning method, in combination with all policies, has a higher semantic similarity compared to other existing methods, and fine-tuning improves caption accuracy and consistency by a significant margin. Code and test set annotations available at https://hsp-iit.github.io/embodied-captioning/
>
---
#### [replaced 015] Learning Multimodal Attention for Manipulating Deformable Objects with Changing States
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2309.14837v2](http://arxiv.org/pdf/2309.14837v2)**

> **作者:** Namiko Saito; Mayu Tatsumi; Ayuna Kubo; Kanata Suzuki; Hiroshi Ito; Shigeki Sugano; Tetsuya Ogata
>
> **备注:** Humanoids2025
>
> **摘要:** To support humans in their daily lives, robots are required to autonomously learn, adapt to objects and environments, and perform the appropriate actions. We tackled on the task of cooking scrambled eggs using real ingredients, in which the robot needs to perceive the states of the egg and adjust stirring movement in real time, while the egg is heated and the state changes continuously. In previous works, handling changing objects was found to be challenging because sensory information includes dynamical, both important or noisy information, and the modality which should be focused on changes every time, making it difficult to realize both perception and motion generation in real time. We propose a predictive recurrent neural network with an attention mechanism that can weigh the sensor input, distinguishing how important and reliable each modality is, that realize quick and efficient perception and motion generation. The model is trained with learning from the demonstration, and allows the robot to acquire human-like skills. We validated the proposed technique using the robot, Dry-AIREC, and with our learning model, it could perform cooking eggs with unknown ingredients. The robot could change the method of stirring and direction depending on the status of the egg, as in the beginning it stirs in the whole pot, then subsequently, after the egg started being heated, it starts flipping and splitting motion targeting specific areas, although we did not explicitly indicate them.
>
---
#### [replaced 016] Nominal Evaluation Of Automatic Multi-Sections Control Potential In Comparison To A Simpler One- Or Two-Sections Alternative With Predictive Spray Switching
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2508.11573v2](http://arxiv.org/pdf/2508.11573v2)**

> **作者:** Mogens Plessen
>
> **备注:** . 16 pages plus 7 pages appendix with additional figures, 19 figures, 6 tables . In comparison to previous version: added detailed economic cost discussion
>
> **摘要:** Automatic Section Control (ASC) is a long-standing trend for spraying in agriculture. It promises to minimise spray overlap areas. The core idea is to (i) switch off spray nozzles on areas that have already been sprayed, and (ii) to dynamically adjust nozzle flow rates along the boom bar that holds the spray nozzles when velocities of boom sections vary during turn maneuvers. ASC is not possible without sensors for accurate positioning data. Spraying and the movement of modern wide boom bars are highly dynamic processes. In addition, many uncertainty factors have an effect such as cross wind drift, nozzle clogging in open-field conditions, etc. In view of this complexity, the natural question arises if a simpler alternative exist. Therefore, ASC is compared to a proposed simpler one- or two-sections alternative that uses predictive spray switching. The comparison is provided under nominal conditions. Agricultural spraying is intrinsically linked to area coverage path planning and spray switching logic. Combinations of two area coverage path planning and switching logics as well as 3 sections-setups are compared. The three sections-setups differ by controlling 48 sections, 2 sections or controlling all nozzles uniformly with the same control signal as one single section. Methods are evaluated on 10 diverse real-world field examples, including non-convex field contours, freeform mainfield lanes and multiple obstacle areas. An economic cost analysis is provided to compare the methods. A preferred method is suggested that (i) minimises area coverage pathlength, (ii) offers intermediate overlap, (iii) is suitable for manual driving by following a pre-planned predictive spray switching logic for an area coverage path plan, and (iv) and in contrast to ASC can be implemented sensor-free and at low cost. Surprisingly strong economic arguments are found to not recommend ASC for small farms.
>
---
#### [replaced 017] One-Step Model Predictive Path Integral for Manipulator Motion Planning Using Configuration Space Distance Fields
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.00836v2](http://arxiv.org/pdf/2509.00836v2)**

> **作者:** Yulin Li; Tetsuro Miyazaki; Kenji Kawashima
>
> **摘要:** Motion planning for robotic manipulators is a fundamental problem in robotics. Classical optimization-based methods typically rely on the gradients of signed distance fields (SDFs) to impose collision-avoidance constraints. However, these methods are susceptible to local minima and may fail when the SDF gradients vanish. Recently, Configuration Space Distance Fields (CDFs) have been introduced, which directly model distances in the robot's configuration space. Unlike workspace SDFs, CDFs are differentiable almost everywhere and thus provide reliable gradient information. On the other hand, gradient-free approaches such as Model Predictive Path Integral (MPPI) control leverage long-horizon rollouts to achieve collision avoidance. While effective, these methods are computationally expensive due to the large number of trajectory samples, repeated collision checks, and the difficulty of designing cost functions with heterogeneous physical units. In this paper, we propose a framework that integrates CDFs with MPPI to enable direct navigation in the robot's configuration space. Leveraging CDF gradients, we unify the MPPI cost in joint-space and reduce the horizon to one step, substantially cutting computation while preserving collision avoidance in practice. We demonstrate that our approach achieves nearly 100% success rates in 2D environments and consistently high success rates in challenging 7-DOF Franka manipulator simulations with complex obstacles. Furthermore, our method attains control frequencies exceeding 750 Hz, substantially outperforming both optimization-based and standard MPPI baselines. These results highlight the effectiveness and efficiency of the proposed CDF-MPPI framework for high-dimensional motion planning.
>
---
#### [replaced 018] PRISM-DP: Spatial Pose-based Observations for Diffusion-Policies via Segmentation, Mesh Generation, and Pose Tracking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.20359v3](http://arxiv.org/pdf/2504.20359v3)**

> **作者:** Xiatao Sun; Yinxing Chen; Daniel Rakita
>
> **摘要:** Diffusion policies generate robot motions by learning to denoise action-space trajectories conditioned on observations. These observations are commonly streams of RGB images, whose high dimensionality includes substantial task-irrelevant information, requiring large models to extract relevant patterns. In contrast, using structured observations like the spatial poses of key objects enables training more compact policies with fewer parameters. However, obtaining accurate object poses in open-set, real-world environments remains challenging, as 6D pose estimation and tracking methods often depend on markers placed on objects beforehand or pre-scanned object meshes that require manual reconstruction. We propose PRISM-DP, an approach that leverages segmentation, mesh generation, and pose tracking models to enable compact diffusion policy learning directly from the spatial poses of task-relevant objects. Crucially, by using a mesh generation model, PRISM-DP eliminates the need for manual mesh creation, improving scalability in open-set environments. Experiments in simulation and the real world show that PRISM-DP outperforms high-dimensional image-based policies and achieves performance comparable to policies trained with ground-truth state information.
>
---
#### [replaced 019] Hierarchical LLMs In-the-loop Optimization for Real-time Multi-Robot Target Tracking under Unknown Hazards
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.12274v2](http://arxiv.org/pdf/2409.12274v2)**

> **作者:** Yuwei Wu; Yuezhan Tao; Peihan Li; Guangyao Shi; Gaurav S. Sukhatme; Vijay Kumar; Lifeng Zhou
>
> **摘要:** Real-time multi-robot coordination in hazardous and adversarial environments requires fast, reliable adaptation to dynamic threats. While Large Language Models (LLMs) offer strong high-level reasoning capabilities, the lack of safety guarantees limits their direct use in critical decision-making. In this paper, we propose a hierarchical optimization framework that integrates LLMs into the decision loop for multi-robot target tracking in dynamic and hazardous environments. Rather than generating control actions directly, LLMs are used to generate task configuration and adjust parameters in a bi-level task allocation and planning problem. We formulate multi-robot coordination for tracking tasks as a bi-level optimization problem, with LLMs to reason about potential hazards in the environment and the status of the robot team and modify both the inner and outer levels of the optimization. This hierarchical approach enables real-time adjustments to the robots' behavior. Additionally, a human supervisor can offer broad guidance and assessments to address unexpected dangers, model mismatches, and performance issues arising from local minima. We validate our proposed framework in both simulation and real-world experiments with comprehensive evaluations, demonstrating its effectiveness and showcasing its capability for safe LLM integration for multi-robot systems.
>
---
#### [replaced 020] Embodied Navigation Foundation Model
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.12129v2](http://arxiv.org/pdf/2509.12129v2)**

> **作者:** Jiazhao Zhang; Anqi Li; Yunpeng Qi; Minghan Li; Jiahang Liu; Shaoan Wang; Haoran Liu; Gengze Zhou; Yuze Wu; Xingxing Li; Yuxin Fan; Wenjun Li; Zhibo Chen; Fei Gao; Qi Wu; Zhizheng Zhang; He Wang
>
> **备注:** Project Page: https://pku-epic.github.io/NavFoM-Web/
>
> **摘要:** Navigation is a fundamental capability in embodied AI, representing the intelligence required to perceive and interact within physical environments following language instructions. Despite significant progress in large Vision-Language Models (VLMs), which exhibit remarkable zero-shot performance on general vision-language tasks, their generalization ability in embodied navigation remains largely confined to narrow task settings and embodiment-specific architectures. In this work, we introduce a cross-embodiment and cross-task Navigation Foundation Model (NavFoM), trained on eight million navigation samples that encompass quadrupeds, drones, wheeled robots, and vehicles, and spanning diverse tasks such as vision-and-language navigation, object searching, target tracking, and autonomous driving. NavFoM employs a unified architecture that processes multimodal navigation inputs from varying camera configurations and navigation horizons. To accommodate diverse camera setups and temporal horizons, NavFoM incorporates identifier tokens that embed camera view information of embodiments and the temporal context of tasks. Furthermore, to meet the demands of real-world deployment, NavFoM controls all observation tokens using a dynamically adjusted sampling strategy under a limited token length budget. Extensive evaluations on public benchmarks demonstrate that our model achieves state-of-the-art or highly competitive performance across multiple navigation tasks and embodiments without requiring task-specific fine-tuning. Additional real-world experiments further confirm the strong generalization capability and practical applicability of our approach.
>
---
#### [replaced 021] Humanoid Agent via Embodied Chain-of-Action Reasoning with Multimodal Foundation Models for Zero-Shot Loco-Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.09532v2](http://arxiv.org/pdf/2504.09532v2)**

> **作者:** Congcong Wen; Geeta Chandra Raju Bethala; Yu Hao; Niraj Pudasaini; Hao Huang; Shuaihang Yuan; Baoru Huang; Anh Nguyen; Anthony Tzes; Yi Fang
>
> **备注:** website link: https://humanoid-coa.github.io/
>
> **摘要:** Humanoid loco-manipulation, which integrates whole-body locomotion with dexterous manipulation, remains a fundamental challenge in robotics. Beyond whole-body coordination and balance, a central difficulty lies in understanding human instructions and translating them into coherent sequences of embodied actions. Recent advances in foundation models provide transferable multimodal representations and reasoning capabilities, yet existing efforts remain largely restricted to either locomotion or manipulation in isolation, with limited applicability to humanoid settings. In this paper, we propose Humanoid-COA, the first humanoid agent framework that integrates foundation model reasoning with an Embodied Chain-of-Action (CoA) mechanism for zero-shot loco-manipulation. Within the perception--reasoning--action paradigm, our key contribution lies in the reasoning stage, where the proposed CoA mechanism decomposes high-level human instructions into structured sequences of locomotion and manipulation primitives through affordance analysis, spatial inference, and whole-body action reasoning. Extensive experiments on two humanoid robots, Unitree H1-2 and G1, in both an open test area and an apartment environment, demonstrate that our framework substantially outperforms prior baselines across manipulation, locomotion, and loco-manipulation tasks, achieving robust generalization to long-horizon and unstructured scenarios. Project page: https://humanoid-coa.github.io/
>
---
#### [replaced 022] Hierarchical Reactive Grasping via Task-Space Velocity Fields and Joint-Space Quadratic Programming
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.01044v2](http://arxiv.org/pdf/2509.01044v2)**

> **作者:** Yonghyeon Lee; Tzu-Yuan Lin; Alexander Alexiev; Sangbae Kim
>
> **备注:** 8 pages, 12 figures, under review
>
> **摘要:** We present a fast and reactive grasping framework that combines task-space velocity fields with joint-space Quadratic Program (QP) in a hierarchical structure. Reactive, collision-free global motion planning is particularly challenging for high-DoF systems, as simultaneous increases in state dimensionality and planning horizon trigger a combinatorial explosion of the search space, making real-time planning intractable. To address this, we plan globally in a lower-dimensional task space, such as fingertip positions, and track locally in the full joint space while enforcing all constraints. This approach is realized by constructing velocity fields in multiple task-space coordinates (or, in some cases, a subset of joint coordinates) and solving a weighted joint-space QP to compute joint velocities that track these fields with appropriately assigned priorities. Through simulation experiments and real-world tests using the recent pose-tracking algorithm FoundationPose, we verify that our method enables high-DoF arm-hand systems to perform real-time, collision-free reaching motions while adapting to dynamic environments and external disturbances.
>
---
#### [replaced 023] FlowAct: A Proactive Multimodal Human-robot Interaction System with Continuous Flow of Perception and Modular Action Sub-systems
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.15864v3](http://arxiv.org/pdf/2408.15864v3)**

> **作者:** Timothée Dhaussy; Bassam Jabaian; Fabrice Lefèvre
>
> **备注:** Paper accepted at ICPRAM 2025
>
> **摘要:** The evolution of autonomous systems in the context of human-robot interaction systems necessitates a synergy between the continuous perception of the environment and the potential actions to navigate or interact within it. We present Flowact, a proactive multimodal human-robot interaction architecture, working as an asynchronous endless loop of robot sensors into actuators and organized by two controllers, the Environment State Tracking (EST) and the Action Planner. The EST continuously collects and publishes a representation of the operative environment, ensuring a steady flow of perceptual data. This persistent perceptual flow is pivotal for our advanced Action Planner which orchestrates a collection of modular action subsystems, such as movement and speaking modules, governing their initiation or cessation based on the evolving environmental narrative. The EST employs a fusion of diverse sensory modalities to build a rich, real-time representation of the environment that is distributed to the Action Planner. This planner uses a decision-making framework to dynamically coordinate action modules, allowing them to respond proactively and coherently to changes in the environment. Through a series of real-world experiments, we exhibit the efficacy of the system in maintaining a continuous perception-action loop, substantially enhancing the responsiveness and adaptability of autonomous pro-active agents. The modular architecture of the action subsystems facilitates easy extensibility and adaptability to a broad spectrum of tasks and scenarios.
>
---
#### [replaced 024] GWM: Towards Scalable Gaussian World Models for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.17600v2](http://arxiv.org/pdf/2508.17600v2)**

> **作者:** Guanxing Lu; Baoxiong Jia; Puhao Li; Yixin Chen; Ziwei Wang; Yansong Tang; Siyuan Huang
>
> **备注:** Published at ICCV 2025. Project page: https://gaussian-world-model.github.io/
>
> **摘要:** Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model.
>
---
#### [replaced 025] Robust Docking Maneuvers for Autonomous Trolley Collection: An Optimization-Based Visual Servoing Scheme
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.07413v2](http://arxiv.org/pdf/2509.07413v2)**

> **作者:** Yuhan Pang; Bingyi Xia; Zhe Zhang; Zhirui Sun; Peijia Xie; Bike Zhu; Wenjun Xu; Jiankun Wang
>
> **摘要:** Service robots have demonstrated significant potential for autonomous trolley collection and redistribution in public spaces like airports or warehouses to improve efficiency and reduce cost. Usually, a fully autonomous system for the collection and transportation of multiple trolleys is based on a Leader-Follower formation of mobile manipulators, where reliable docking maneuvers of the mobile base are essential to align trolleys into organized queues. However, developing a vision-based robotic docking system faces significant challenges: high precision requirements, environmental disturbances, and inherent robot constraints. To address these challenges, we propose an optimization-based Visual Servoing scheme that incorporates active infrared markers for robust feature extraction across diverse lighting conditions. This framework explicitly models nonholonomic kinematics and visibility constraints within the Hybrid Visual Servoing problem, augmented with an observer for disturbance rejection to ensure precise and stable docking. Experimental results across diverse environments demonstrate the robustness of this system, with quantitative evaluations confirming high docking accuracy.
>
---
#### [replaced 026] ORCA: An Open-Source, Reliable, Cost-Effective, Anthropomorphic Robotic Hand for Uninterrupted Dexterous Task Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.04259v2](http://arxiv.org/pdf/2504.04259v2)**

> **作者:** Clemens C. Christoph; Maximilian Eberlein; Filippos Katsimalis; Arturo Roberti; Aristotelis Sympetheros; Michel R. Vogt; Davide Liconti; Chenyu Yang; Barnabas Gavin Cangan; Ronan J. Hinchet; Robert K. Katzschmann
>
> **备注:** This work has been accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** General-purpose robots should possess human-like dexterity and agility to perform tasks with the same versatility as us. A human-like form factor further enables the use of vast datasets of human-hand interactions. However, the primary bottleneck in dexterous manipulation lies not only in software but arguably even more in hardware. Robotic hands that approach human capabilities are often prohibitively expensive, bulky, or require enterprise-level maintenance, limiting their accessibility for broader research and practical applications. What if the research community could get started with reliable dexterous hands within a day? We present the open-source ORCA hand, a reliable and anthropomorphic 17-DoF tendon-driven robotic hand with integrated tactile sensors, fully assembled in less than eight hours and built for a material cost below 2,000 CHF. We showcase ORCA's key design features such as popping joints, auto-calibration, and tensioning systems that significantly reduce complexity while increasing reliability, accuracy, and robustness. We benchmark the ORCA hand across a variety of tasks, ranging from teleoperation and imitation learning to zero-shot sim-to-real reinforcement learning. Furthermore, we demonstrate its durability, withstanding more than 10,000 continuous operation cycles - equivalent to approximately 20 hours - without hardware failure, the only constraint being the duration of the experiment itself. Video is here: https://youtu.be/kUbPSYMmOds. Design files, source code, and documentation are available at https://srl.ethz.ch/orcahand.
>
---
#### [replaced 027] Efficient Tactile Perception with Soft Electrical Impedance Tomography and Pre-trained Transformer
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.02824v2](http://arxiv.org/pdf/2506.02824v2)**

> **作者:** Huazhi Dong; Ronald B. Liu; Sihao Teng; Delin Hu; Peisan; E; Francesco Giorgio-Serchi; Yunjie Yang
>
> **备注:** IEEE Transactions on Industrial Electronics
>
> **摘要:** Tactile sensing is fundamental to robotic systems, enabling interactions through physical contact in multiple tasks. Despite its importance, achieving high-resolution, large-area tactile sensing remains challenging. Electrical Impedance Tomography (EIT) has emerged as a promising approach for large-area, distributed tactile sensing with minimal electrode requirements which can lend itself to addressing complex contact problems in robotics. However, existing EIT-based tactile reconstruction methods often suffer from high computational costs or depend on extensive annotated simulation datasets, hindering its viability in real-world settings. To address this shortcoming, here we propose a Pre-trained Transformer for EIT-based Tactile Reconstruction (PTET), a learning-based framework that bridges the simulation-to-reality gap by leveraging self-supervised pretraining on simulation data and fine-tuning with limited real-world data. In simulations, PTET requires 99.44 percent fewer annotated samples than equivalent state-of-the-art approaches (2,500 vs. 450,000 samples) while achieving reconstruction performance improvements of up to 43.57 percent under identical data conditions. Fine-tuning with real-world data further enables PTET to overcome discrepancies between simulated and experimental datasets, achieving superior reconstruction and detail recovery in practical scenarios. The improved reconstruction accuracy, data efficiency, and robustness in real-world tasks establish it as a scalable and practical solution for tactile sensing systems in robotics, especially for object handling and adaptive grasping under varying pressure conditions.
>
---
#### [replaced 028] Video-Language Critic: Transferable Reward Functions for Language-Conditioned Robotics
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.19988v3](http://arxiv.org/pdf/2405.19988v3)**

> **作者:** Minttu Alakuijala; Reginald McLean; Isaac Woungang; Nariman Farsad; Samuel Kaski; Pekka Marttinen; Kai Yuan
>
> **备注:** 14 pages in the main text, 22 pages including references and supplementary materials. 3 figures and 3 tables in the main text, 6 figures and 3 tables in supplementary materials
>
> **摘要:** Natural language is often the easiest and most convenient modality for humans to specify tasks for robots. However, learning to ground language to behavior typically requires impractical amounts of diverse, language-annotated demonstrations collected on each target robot. In this work, we aim to separate the problem of what to accomplish from how to accomplish it, as the former can benefit from substantial amounts of external observation-only data, and only the latter depends on a specific robot embodiment. To this end, we propose Video-Language Critic, a reward model that can be trained on readily available cross-embodiment data using contrastive learning and a temporal ranking objective, and use it to score behavior traces from a separate actor. When trained on Open X-Embodiment data, our reward model enables 2x more sample-efficient policy training on Meta-World tasks than a sparse reward only, despite a significant domain gap. Using in-domain data but in a challenging task generalization setting on Meta-World, we further demonstrate more sample-efficient training than is possible with prior language-conditioned reward models that are either trained with binary classification, use static images, or do not leverage the temporal information present in video data.
>
---
#### [replaced 029] DECAMP: Towards Scene-Consistent Multi-Agent Motion Prediction with Disentangled Context-Aware Pre-Training
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.10426v2](http://arxiv.org/pdf/2509.10426v2)**

> **作者:** Jianxin Shi; Zengqi Peng; Xiaolong Chen; Tianyu Wo; Jun Ma
>
> **摘要:** Trajectory prediction is a critical component of autonomous driving, essential for ensuring both safety and efficiency on the road. However, traditional approaches often struggle with the scarcity of labeled data and exhibit suboptimal performance in multi-agent prediction scenarios. To address these challenges, we introduce a disentangled context-aware pre-training framework for multi-agent motion prediction, named DECAMP. Unlike existing methods that entangle representation learning with pretext tasks, our framework decouples behavior pattern learning from latent feature reconstruction, prioritizing interpretable dynamics and thereby enhancing scene representation for downstream prediction. Additionally, our framework incorporates context-aware representation learning alongside collaborative spatial-motion pretext tasks, which enables joint optimization of structural and intentional reasoning while capturing the underlying dynamic intentions. Our experiments on the Argoverse 2 benchmark showcase the superior performance of our method, and the results attained underscore its effectiveness in multi-agent motion forecasting. To the best of our knowledge, this is the first context autoencoder framework for multi-agent motion forecasting in autonomous driving. The code and models will be made publicly available.
>
---
#### [replaced 030] Search-TTA: A Multimodal Test-Time Adaptation Framework for Visual Search in the Wild
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11350v4](http://arxiv.org/pdf/2505.11350v4)**

> **作者:** Derek Ming Siang Tan; Shailesh; Boyang Liu; Alok Raj; Qi Xuan Ang; Weiheng Dai; Tanishq Duhan; Jimmy Chiun; Yuhong Cao; Florian Shkurti; Guillaume Sartoretti
>
> **备注:** Accepted for presentation at CORL 2025. Code, models, and data are available at https://search-tta.github.io/
>
> **摘要:** To perform outdoor autonomous visual navigation and search, a robot may leverage satellite imagery as a prior map. This can help inform high-level search and exploration strategies, even when such images lack sufficient resolution to allow for visual recognition of targets. However, there are limited training datasets of satellite images with annotated targets that are not directly visible. Furthermore, approaches which leverage large Vision Language Models (VLMs) for generalization may yield inaccurate outputs due to hallucination, leading to inefficient search. To address these challenges, we introduce Search-TTA, a multimodal test-time adaptation framework with a flexible plug-and-play interface compatible with various input modalities (e.g. image, text, sound) and planning methods. First, we pretrain a satellite image encoder to align with CLIP's visual encoder to output probability distributions of target presence used for visual search. Second, our framework dynamically refines CLIP's predictions during search using a test-time adaptation mechanism. Through a novel feedback loop inspired by Spatial Poisson Point Processes, uncertainty-weighted gradient updates are used to correct potentially inaccurate predictions and improve search performance. To train and evaluate Search-TTA, we curate AVS-Bench, a visual search dataset based on internet-scale ecological data that contains up to 380k training and 8k validation images (in- and out-domain). We find that Search-TTA improves planner performance by up to 30.0%, particularly in cases with poor initial CLIP predictions due to limited training data. It also performs comparably with significantly larger VLMs, and achieves zero-shot generalization to unseen modalities. Finally, we deploy Search-TTA on a real UAV via hardware-in-the-loop testing, by simulating its operation within a large-scale simulation that provides onboard sensing.
>
---
#### [replaced 031] Enhancing Generalization in Vision-Language-Action Models by Preserving Pretrained Representations
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.11417v2](http://arxiv.org/pdf/2509.11417v2)**

> **作者:** Shresth Grover; Akshay Gopalkrishnan; Bo Ai; Henrik I. Christensen; Hao Su; Xuanlin Li
>
> **备注:** Project Page: https://gen-vla.github.io/
>
> **摘要:** Vision-language-action (VLA) models finetuned from vision-language models (VLMs) hold the promise of leveraging rich pretrained representations to build generalist robots across diverse tasks and environments. However, direct fine-tuning on robot data often disrupts these representations and limits generalization. We present a framework that better preserves pretrained features while adapting them for robot manipulation. Our approach introduces three components: (i) a dual-encoder design with one frozen vision encoder to retain pretrained features and another trainable for task adaptation, (ii) a string-based action tokenizer that casts continuous actions into character sequences aligned with the model's pretraining domain, and (iii) a co-training strategy that combines robot demonstrations with vision-language datasets emphasizing spatial reasoning and affordances. Evaluations in simulation and on real robots show that our method improves robustness to visual perturbations, generalization to novel instructions and environments, and overall task success compared to baselines.
>
---
#### [replaced 032] Mining the Long Tail: A Comparative Study of Data-Centric Criticality Metrics for Robust Offline Reinforcement Learning in Autonomous Motion Planning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.18397v2](http://arxiv.org/pdf/2508.18397v2)**

> **作者:** Antonio Guillen-Perez
>
> **摘要:** Offline Reinforcement Learning (RL) presents a promising paradigm for training autonomous vehicle (AV) planning policies from large-scale, real-world driving logs. However, the extreme data imbalance in these logs, where mundane scenarios vastly outnumber rare "long-tail" events, leads to brittle and unsafe policies when using standard uniform data sampling. In this work, we address this challenge through a systematic, large-scale comparative study of data curation strategies designed to focus the learning process on information-rich samples. We investigate six distinct criticality weighting schemes which are categorized into three families: heuristic-based, uncertainty-based, and behavior-based. These are evaluated at two temporal scales, the individual timestep and the complete scenario. We train seven goal-conditioned Conservative Q-Learning (CQL) agents with a state-of-the-art, attention-based architecture and evaluate them in the high-fidelity Waymax simulator. Our results demonstrate that all data curation methods significantly outperform the baseline. Notably, data-driven curation using model uncertainty as a signal achieves the most significant safety improvements, reducing the collision rate by nearly three-fold (from 16.0% to 5.5%). Furthermore, we identify a clear trade-off where timestep-level weighting excels at reactive safety while scenario-level weighting improves long-horizon planning. Our work provides a comprehensive framework for data curation in Offline RL and underscores that intelligent, non-uniform sampling is a critical component for building safe and reliable autonomous agents.
>
---
