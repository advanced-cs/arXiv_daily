# 机器人 cs.RO

- **最新发布 26 篇**

- **更新 23 篇**

## 最新发布

#### [new 001] Hybrid Data-Driven Predictive Control for Robust and Reactive Exoskeleton Locomotion Synthesis
- **分类: cs.RO**

- **简介: 论文提出混合数据驱动预测控制框架，解决外骨骼在动态环境中的实时反应与鲁棒性问题，通过联合规划接触时间与轨迹实现高效在线重规划。**

- **链接: [http://arxiv.org/pdf/2508.10269v1](http://arxiv.org/pdf/2508.10269v1)**

> **作者:** Kejun Li; Jeeseop Kim; Maxime Brunet; Marine Pétriaux; Yisong Yue; Aaron D. Ames
>
> **备注:** 8 pages; 8 figures
>
> **摘要:** Robust bipedal locomotion in exoskeletons requires the ability to dynamically react to changes in the environment in real time. This paper introduces the hybrid data-driven predictive control (HDDPC) framework, an extension of the data-enabled predictive control, that addresses these challenges by simultaneously planning foot contact schedules and continuous domain trajectories. The proposed framework utilizes a Hankel matrix-based representation to model system dynamics, incorporating step-to-step (S2S) transitions to enhance adaptability in dynamic environments. By integrating contact scheduling with trajectory planning, the framework offers an efficient, unified solution for locomotion motion synthesis that enables robust and reactive walking through online replanning. We validate the approach on the Atalante exoskeleton, demonstrating improved robustness and adaptability.
>
---
#### [new 002] MASH: Cooperative-Heterogeneous Multi-Agent Reinforcement Learning for Single Humanoid Robot Locomotion
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 论文提出MASH方法，通过合作-异构多智能体MARL优化单人形机器人运动，解决传统方法效率低、协作差的问题，提升收敛速度和整体协作能力。**

- **链接: [http://arxiv.org/pdf/2508.10423v1](http://arxiv.org/pdf/2508.10423v1)**

> **作者:** Qi Liu; Xiaopeng Zhang; Mingshan Tan; Shuaikang Ma; Jinliang Ding; Yanjie Li
>
> **摘要:** This paper proposes a novel method to enhance locomotion for a single humanoid robot through cooperative-heterogeneous multi-agent deep reinforcement learning (MARL). While most existing methods typically employ single-agent reinforcement learning algorithms for a single humanoid robot or MARL algorithms for multi-robot system tasks, we propose a distinct paradigm: applying cooperative-heterogeneous MARL to optimize locomotion for a single humanoid robot. The proposed method, multi-agent reinforcement learning for single humanoid locomotion (MASH), treats each limb (legs and arms) as an independent agent that explores the robot's action space while sharing a global critic for cooperative learning. Experiments demonstrate that MASH accelerates training convergence and improves whole-body cooperation ability, outperforming conventional single-agent reinforcement learning methods. This work advances the integration of MARL into single-humanoid-robot control, offering new insights into efficient locomotion strategies.
>
---
#### [new 003] Few-shot Vision-based Human Activity Recognition with MLLM-based Visual Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出一种基于MLLM的视觉强化学习方法，解决少样本多模态人类活动识别中的泛化难题，通过生成候选响应并优化模型参数提升推理能力。**

- **链接: [http://arxiv.org/pdf/2508.10371v1](http://arxiv.org/pdf/2508.10371v1)**

> **作者:** Wenqi Zheng; Yutaka Arakawa
>
> **摘要:** Reinforcement learning in large reasoning models enables learning from feedback on their outputs, making it particularly valuable in scenarios where fine-tuning data is limited. However, its application in multi-modal human activity recognition (HAR) domains remains largely underexplored. Our work extends reinforcement learning to the human activity recognition domain with multimodal large language models. By incorporating visual reinforcement learning in the training process, the model's generalization ability on few-shot recognition can be greatly improved. Additionally, visual reinforcement learning can enhance the model's reasoning ability and enable explainable analysis in the inference stage. We name our few-shot human activity recognition method with visual reinforcement learning FAVOR. Specifically, our approach first utilizes a multimodal large language model (MLLM) to generate multiple candidate responses for the human activity image, each containing reasoning traces and final answers. These responses are then evaluated using reward functions, and the MLLM model is subsequently optimized using the Group Relative Policy Optimization (GRPO) algorithm. In this way, the MLLM model can be adapted to human activity recognition with only a few samples. Extensive experiments on four human activity recognition datasets and five different settings demonstrate the superiority of the proposed method.
>
---
#### [new 004] Enabling Generic Robot Skill Implementation Using Object Oriented Programming
- **分类: cs.RO; cs.SE**

- **简介: 论文提出面向对象框架解决机器人接口复杂性，通过抽象层实现通用技能部署，用Python实现原型。**

- **链接: [http://arxiv.org/pdf/2508.10497v1](http://arxiv.org/pdf/2508.10497v1)**

> **作者:** Abdullah Farrukh; Achim Wagner; Martin Ruskowski
>
> **备注:** 34th International Conference on Robotics in Alpe-Adria-Danube Region (RAAD 2025)
>
> **摘要:** Developing robotic algorithms and integrating a robotic subsystem into a larger system can be a difficult task. Particularly in small and medium-sized enterprises (SMEs) where robotics expertise is lacking, implementing, maintaining and developing robotic systems can be a challenge. As a result, many companies rely on external expertise through system integrators, which, in some cases, can lead to vendor lock-in and external dependency. In the academic research on intelligent manufacturing systems, robots play a critical role in the design of robust autonomous systems. Similar challenges are faced by researchers who want to use robotic systems as a component in a larger smart system, without having to deal with the complexity and vastness of the robot interfaces in detail. In this paper, we propose a software framework that reduces the effort required to deploy a working robotic system. The focus is solely on providing a concept for simplifying the different interfaces of a modern robot system and using an abstraction layer for different manufacturers and models. The Python programming language is used to implement a prototype of the concept. The target system is a bin-picking cell containing a Yaskawa Motoman GP4.
>
---
#### [new 005] Synthesis of Deep Neural Networks with Safe Robust Adaptive Control for Reliable Operation of Wheeled Mobile Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出结合DNN和RAC的分层控制策略，解决重型WMR在干扰下的安全可靠运行问题，确保系统稳定并符合安全标准。**

- **链接: [http://arxiv.org/pdf/2508.10634v1](http://arxiv.org/pdf/2508.10634v1)**

> **作者:** Mehdi Heydari Shahna; Jouni Mattila
>
> **摘要:** Deep neural networks (DNNs) can enable precise control while maintaining low computational costs by circumventing the need for dynamic modeling. However, the deployment of such black-box approaches remains challenging for heavy-duty wheeled mobile robots (WMRs), which are subject to strict international standards and prone to faults and disturbances. We designed a hierarchical control policy for heavy-duty WMRs, monitored by two safety layers with differing levels of authority. To this end, a DNN policy was trained and deployed as the primary control strategy, providing high-precision performance under nominal operating conditions. When external disturbances arise and reach a level of intensity such that the system performance falls below a predefined threshold, a low-level safety layer intervenes by deactivating the primary control policy and activating a model-free robust adaptive control (RAC) policy. This transition enables the system to continue operating while ensuring stability by effectively managing the inherent trade-off between system robustness and responsiveness. Regardless of the control policy in use, a high-level safety layer continuously monitors system performance during operation. It initiates a shutdown only when disturbances become sufficiently severe such that compensation is no longer viable and continued operation would jeopardize the system or its environment. The proposed synthesis of DNN and RAC policy guarantees uniform exponential stability of the entire WMR system while adhering to safety standards to some extent. The effectiveness of the proposed approach was further validated through real-time experiments using a 6,000 kg WMR.
>
---
#### [new 006] An Open-Source User-Friendly Interface for Simulating Magnetic Soft Robots using Simulation Open Framework Architecture (SOFA)
- **分类: cs.RO; cond-mat.mtrl-sci**

- **简介: 论文提出基于SOFA框架的开源仿真工具，用于磁软机器人建模，解决传统平台缺乏磁材料支持的问题，实现材料属性定义、磁场应用及实时变形观察，提升理论与实践结合效率。**

- **链接: [http://arxiv.org/pdf/2508.10686v1](http://arxiv.org/pdf/2508.10686v1)**

> **作者:** Carla Wehner; Finn Schubert; Heiko Hellkamp; Julius Hahnewald; Kilian Scheafer; Muhammad Bilal Khan; Oliver Gutfleisch
>
> **摘要:** Soft robots, particularly magnetic soft robots, require specialized simulation tools to accurately model their deformation under external magnetic fields. However, existing platforms often lack dedicated support for magnetic materials, making them difficult to use for researchers at different expertise levels. This work introduces an open-source, user-friendly simulation interface using the Simulation Open Framework Architecture (SOFA), specifically designed to model magnetic soft robots. The tool enables users to define material properties, apply magnetic fields, and observe resulting deformations in real time. By integrating intuitive controls and stress analysis capabilities, it aims to bridge the gap between theoretical modeling and practical design. Four benchmark models - a beam, three- and four-finger grippers, and a butterfly - demonstrate its functionality. The software's ease of use makes it accessible to both beginners and advanced researchers. Future improvements will refine accuracy through experimental validation and comparison with industry-standard finite element solvers, ensuring realistic and predictive simulations of magnetic soft robots.
>
---
#### [new 007] CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 论文提出一种基于自修正飞轮的视觉-语言-动作导航模型CorrectNav，通过利用错误轨迹生成自修正数据，提升模型对指令的执行精度与动态避障能力，实现新状态最优性能。**

- **链接: [http://arxiv.org/pdf/2508.10416v1](http://arxiv.org/pdf/2508.10416v1)**

> **作者:** Zhuoyuan Yu; Yuxing Long; Zihan Yang; Chengyan Zeng; Hongwei Fan; Jiyao Zhang; Hao Dong
>
> **摘要:** Existing vision-and-language navigation models often deviate from the correct trajectory when executing instructions. However, these models lack effective error correction capability, hindering their recovery from errors. To address this challenge, we propose Self-correction Flywheel, a novel post-training paradigm. Instead of considering the model's error trajectories on the training set as a drawback, our paradigm emphasizes their significance as a valuable data source. We have developed a method to identify deviations in these error trajectories and devised innovative techniques to automatically generate self-correction data for perception and action. These self-correction data serve as fuel to power the model's continued training. The brilliance of our paradigm is revealed when we re-evaluate the model on the training set, uncovering new error trajectories. At this time, the self-correction flywheel begins to spin. Through multiple flywheel iterations, we progressively enhance our monocular RGB-based VLA navigation model CorrectNav. Experiments on R2R-CE and RxR-CE benchmarks show CorrectNav achieves new state-of-the-art success rates of 65.1% and 69.3%, surpassing prior best VLA navigation models by 8.2% and 16.4%. Real robot tests in various indoor and outdoor environments demonstrate \method's superior capability of error correction, dynamic obstacle avoidance, and long instruction following.
>
---
#### [new 008] ReconVLA: Reconstructive Vision-Language-Action Model as Effective Robot Perceiver
- **分类: cs.RO; cs.CV**

- **简介: 论文提出ReconVLA模型，通过隐式方法重建图像注视区域，解决视觉注意力分配不足问题，提升机器人感知精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.10333v1](http://arxiv.org/pdf/2508.10333v1)**

> **作者:** Wenxuan Song; Ziyang Zhou; Han Zhao; Jiayi Chen; Pengxiang Ding; Haodong Yan; Yuxin Huang; Feilong Tang; Donglin Wang; Haoang Li
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) models have enabled robotic agents to integrate multimodal understanding with action execution. However, our empirical analysis reveals that current VLAs struggle to allocate visual attention to target regions. Instead, visual attention is always dispersed. To guide the visual attention grounding on the correct target, we propose ReconVLA, a reconstructive VLA model with an implicit grounding paradigm. Conditioned on the model's visual outputs, a diffusion transformer aims to reconstruct the gaze region of the image, which corresponds to the target manipulated objects. This process prompts the VLA model to learn fine-grained representations and accurately allocate visual attention, thus effectively leveraging task-specific visual information and conducting precise manipulation. Moreover, we curate a large-scale pretraining dataset comprising over 100k trajectories and 2 million data samples from open-source robotic datasets, further boosting the model's generalization in visual reconstruction. Extensive experiments in simulation and the real world demonstrate the superiority of our implicit grounding method, showcasing its capabilities of precise manipulation and generalization. Our project page is https://zionchow.github.io/ReconVLA/.
>
---
#### [new 009] A Multimodal Neural Network for Recognizing Subjective Self-Disclosure Towards Social Robots
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种多模态神经网络，用于识别社交机器人对主观自我披露的感知，解决现有建模不足问题，通过新损失函数提升准确性，实现F1分数0.83，推动社交机器人社会认知能力。**

- **链接: [http://arxiv.org/pdf/2508.10828v1](http://arxiv.org/pdf/2508.10828v1)**

> **作者:** Henry Powell; Guy Laban; Emily S. Cross
>
> **备注:** Accepted at 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Subjective self-disclosure is an important feature of human social interaction. While much has been done in the social and behavioural literature to characterise the features and consequences of subjective self-disclosure, little work has been done thus far to develop computational systems that are able to accurately model it. Even less work has been done that attempts to model specifically how human interactants self-disclose with robotic partners. It is becoming more pressing as we require social robots to work in conjunction with and establish relationships with humans in various social settings. In this paper, our aim is to develop a custom multimodal attention network based on models from the emotion recognition literature, training this model on a large self-collected self-disclosure video corpus, and constructing a new loss function, the scale preserving cross entropy loss, that improves upon both classification and regression versions of this problem. Our results show that the best performing model, trained with our novel loss function, achieves an F1 score of 0.83, an improvement of 0.48 from the best baseline model. This result makes significant headway in the aim of allowing social robots to pick up on an interaction partner's self-disclosures, an ability that will be essential in social robots with social cognition.
>
---
#### [new 010] Why Report Failed Interactions With Robots?! Towards Vignette-based Interaction Quality
- **分类: cs.RO; cs.HC**

- **简介: 论文提出使用情境化 vignettes 明确机器人交互失败，解决因上下文依赖导致的泛化难题，通过多学科视角揭示未被记录的异常行为，增强交互评估透明度。**

- **链接: [http://arxiv.org/pdf/2508.10603v1](http://arxiv.org/pdf/2508.10603v1)**

> **作者:** Agnes Axelsson; Merle Reimann; Ronald Cumbal; Hannah Pelikan; Divesh Lala
>
> **备注:** Accepted at the workshop on Real-World HRI in Public and Private Spaces: Successes, Failures, and Lessons Learned (PubRob-Fails), held at the IEEE RO-MAN Conference, 2025. 6 pages
>
> **摘要:** Although the quality of human-robot interactions has improved with the advent of LLMs, there are still various factors that cause systems to be sub-optimal when compared to human-human interactions. The nature and criticality of failures are often dependent on the context of the interaction and so cannot be generalized across the wide range of scenarios and experiments which have been implemented in HRI research. In this work we propose the use of a technique overlooked in the field of HRI, ethnographic vignettes, to clearly highlight these failures, particularly those that are rarely documented. We describe the methodology behind the process of writing vignettes and create our own based on our personal experiences with failures in HRI systems. We emphasize the strength of vignettes as the ability to communicate failures from a multi-disciplinary perspective, promote transparency about the capabilities of robots, and document unexpected behaviours which would otherwise be omitted from research reports. We encourage the use of vignettes to augment existing interaction evaluation methods.
>
---
#### [new 011] Large Model Empowered Embodied AI: A Survey on Decision-Making and Embodied Learning
- **分类: cs.RO**

- **简介: 论文综述大型模型赋能具身AI，解决传统方法在开放动态环境中的智能不足，涵盖决策与学习范式，分析层次化与端到端架构，引入VLA与世界模型，探讨其提升规划、执行及反馈的作用，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2508.10399v1](http://arxiv.org/pdf/2508.10399v1)**

> **作者:** Wenlong Liang; Rui Zhou; Yang Ma; Bing Zhang; Songlin Li; Yijia Liao; Ping Kuang
>
> **摘要:** Embodied AI aims to develop intelligent systems with physical forms capable of perceiving, decision-making, acting, and learning in real-world environments, providing a promising way to Artificial General Intelligence (AGI). Despite decades of explorations, it remains challenging for embodied agents to achieve human-level intelligence for general-purpose tasks in open dynamic environments. Recent breakthroughs in large models have revolutionized embodied AI by enhancing perception, interaction, planning and learning. In this article, we provide a comprehensive survey on large model empowered embodied AI, focusing on autonomous decision-making and embodied learning. We investigate both hierarchical and end-to-end decision-making paradigms, detailing how large models enhance high-level planning, low-level execution, and feedback for hierarchical decision-making, and how large models enhance Vision-Language-Action (VLA) models for end-to-end decision making. For embodied learning, we introduce mainstream learning methodologies, elaborating on how large models enhance imitation learning and reinforcement learning in-depth. For the first time, we integrate world models into the survey of embodied AI, presenting their design methods and critical roles in enhancing decision-making and learning. Though solid advances have been achieved, challenges still exist, which are discussed at the end of this survey, potentially as the further research directions.
>
---
#### [new 012] Systematic Constraint Formulation and Collision-Free Trajectory Planning Using Space-Time Graphs of Convex Sets
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **简介: 论文提出基于空间-时间凸集图（ST-GCS）的约束建模方法，解决动态环境中无碰撞轨迹规划问题，通过系统约束优化生成最优轨迹，无需初始猜测。**

- **链接: [http://arxiv.org/pdf/2508.10203v1](http://arxiv.org/pdf/2508.10203v1)**

> **作者:** Matthew D. Osburn; Cameron K. Peterson; John L. Salmon
>
> **备注:** 21 pages with references, 20 figures
>
> **摘要:** In this paper, we create optimal, collision-free, time-dependent trajectories through cluttered dynamic environments. The many spatial and temporal constraints make finding an initial guess for a numerical solver difficult. Graphs of Convex Sets (GCS) and the recently developed Space-Time Graphs of Convex Sets formulation (ST-GCS) enable us to generate optimal minimum distance collision-free trajectories without providing an initial guess to the solver. We also explore the derivation of general GCS-compatible constraints and document an intuitive strategy for adapting general constraints to the framework. We show that ST-GCS produces equivalent trajectories to the standard GCS formulation when the environment is static. We then show ST-GCS operating in dynamic environments to find minimum distance collision-free trajectories.
>
---
#### [new 013] WiFi-based Global Localization in Large-Scale Environments Leveraging Structural Priors from osmAG
- **分类: cs.RO**

- **简介: 论文提出一种基于WiFi的全局定位方法，利用osmAG结构先验解决室内GPS受限问题。通过结合信号传播模型与osmAG几何拓扑，实现离线AP精确定位（误差3.79m）及在线实时定位（误差3.12m），显著优于传统KNN指纹法，适用于大规模多层建筑环境，有效解决 kidnapped robot问题。**

- **链接: [http://arxiv.org/pdf/2508.10144v1](http://arxiv.org/pdf/2508.10144v1)**

> **作者:** Xu Ma; Jiajie Zhang; Fujing Xie; Sören Schwertfeger
>
> **摘要:** Global localization is essential for autonomous robotics, especially in indoor environments where the GPS signal is denied. We propose a novel WiFi-based localization framework that leverages ubiquitous wireless infrastructure and the OpenStreetMap Area Graph (osmAG) for large-scale indoor environments. Our approach integrates signal propagation modeling with osmAG's geometric and topological priors. In the offline phase, an iterative optimization algorithm localizes WiFi Access Points (APs) by modeling wall attenuation, achieving a mean localization error of 3.79 m (35.3\% improvement over trilateration). In the online phase, real-time robot localization uses the augmented osmAG map, yielding a mean error of 3.12 m in fingerprinted areas (8.77\% improvement over KNN fingerprinting) and 3.83 m in non-fingerprinted areas (81.05\% improvement). Comparison with a fingerprint-based method shows that our approach is much more space efficient and achieves superior localization accuracy, especially for positions where no fingerprint data are available. Validated across a complex 11,025 &m^2& multi-floor environment, this framework offers a scalable, cost-effective solution for indoor robotic localization, solving the kidnapped robot problem. The code and dataset are available at https://github.com/XuMa369/osmag-wifi-localization.
>
---
#### [new 014] BEASST: Behavioral Entropic Gradient based Adaptive Source Seeking for Mobile Robots
- **分类: cs.RO**

- **简介: 论文提出BEASST框架，针对复杂未知环境下的移动机器人源搜索问题，通过建模信号强度为概率代理，结合行为熵与Prelec函数动态调整探索与利用策略，实现路径优化与快速定位。**

- **链接: [http://arxiv.org/pdf/2508.10363v1](http://arxiv.org/pdf/2508.10363v1)**

> **作者:** Donipolo Ghimire; Aamodh Suresh; Carlos Nieto-Granda; Solmaz S. Kia
>
> **摘要:** This paper presents BEASST (Behavioral Entropic Gradient-based Adaptive Source Seeking for Mobile Robots), a novel framework for robotic source seeking in complex, unknown environments. Our approach enables mobile robots to efficiently balance exploration and exploitation by modeling normalized signal strength as a surrogate probability of source location. Building on Behavioral Entropy(BE) with Prelec's probability weighting function, we define an objective function that adapts robot behavior from risk-averse to risk-seeking based on signal reliability and mission urgency. The framework provides theoretical convergence guarantees under unimodal signal assumptions and practical stability under bounded disturbances. Experimental validation across DARPA SubT and multi-room scenarios demonstrates that BEASST consistently outperforms state-of-the-art methods, achieving 15% reduction in path length and 20% faster source localization through intelligent uncertainty-driven navigation that dynamically transitions between aggressive pursuit and cautious exploration.
>
---
#### [new 015] A Semantic-Aware Framework for Safe and Intent-Integrative Assistance in Upper-Limb Exoskeletons
- **分类: cs.RO**

- **简介: 论文提出一种语义感知框架，针对上肢外骨骼的辅助任务，解决传统方法缺乏任务语义理解与协同规划的问题，通过大语言模型实现安全、意图整合的辅助，结合透明模式捕捉意图、语义参数配置、异常检测与实时重规划，提升任务适应性和安全性。**

- **链接: [http://arxiv.org/pdf/2508.10378v1](http://arxiv.org/pdf/2508.10378v1)**

> **作者:** Yu Chen; Shu Miao; Chunyu Wu; Jingsong Mu; Bo OuYang; Xiang Li
>
> **摘要:** Upper-limb exoskeletons are primarily designed to provide assistive support by accurately interpreting and responding to human intentions. In home-care scenarios, exoskeletons are expected to adapt their assistive configurations based on the semantic information of the task, adjusting appropriately in accordance with the nature of the object being manipulated. However, existing solutions often lack the ability to understand task semantics or collaboratively plan actions with the user, limiting their generalizability. To address this challenge, this paper introduces a semantic-aware framework that integrates large language models into the task planning framework, enabling the delivery of safe and intent-integrative assistance. The proposed approach begins with the exoskeleton operating in transparent mode to capture the wearer's intent during object grasping. Once semantic information is extracted from the task description, the system automatically configures appropriate assistive parameters. In addition, a diffusion-based anomaly detector is used to continuously monitor the state of human-robot interaction and trigger real-time replanning in response to detected anomalies. During task execution, online trajectory refinement and impedance control are used to ensure safety and regulate human-robot interaction. Experimental results demonstrate that the proposed method effectively aligns with the wearer's cognition, adapts to semantically varying tasks, and responds reliably to anomalies.
>
---
#### [new 016] Biasing Frontier-Based Exploration with Saliency Areas
- **分类: cs.RO**

- **简介: 论文提出通过显著性区域引导探索策略，解决传统方法忽略环境重要性导致效率低的问题，采用神经网络生成的显著性图优化探索路径，提升未知区域发现效率。**

- **链接: [http://arxiv.org/pdf/2508.10689v1](http://arxiv.org/pdf/2508.10689v1)**

> **作者:** Matteo Luperto; Valerii Stakanov; Giacomo Boracchi; Nicola Basilico; Francesco Amigoni
>
> **备注:** Accepted at the European Confrence on Mobile Robots (ECMR) 2025
>
> **摘要:** Autonomous exploration is a widely studied problem where a robot incrementally builds a map of a previously unknown environment. The robot selects the next locations to reach using an exploration strategy. To do so, the robot has to balance between competing objectives, like exploring the entirety of the environment, while being as fast as possible. Most exploration strategies try to maximise the explored area to speed up exploration; however, they do not consider that parts of the environment are more important than others, as they lead to the discovery of large unknown areas. We propose a method that identifies \emph{saliency areas} as those areas that are of high interest for exploration, by using saliency maps obtained from a neural network that, given the current map, implements a termination criterion to estimate whether the environment can be considered fully-explored or not. We use saliency areas to bias some widely used exploration strategies, showing, with an extensive experimental campaign, that this knowledge can significantly influence the behavior of the robot during exploration.
>
---
#### [new 017] TLE-Based A2C Agent for Terrestrial Coverage Orbital Path Planning
- **分类: cs.RO; cs.AI**

- **简介: 本文提出基于TLE的A2C代理用于LEO卫星轨道路径规划，解决因轨道拥挤导致的覆盖难题，通过构建MDP环境优化轨道参数，验证A2C在奖励效率和收敛速度上的优势，实现高效自适应部署。**

- **链接: [http://arxiv.org/pdf/2508.10872v1](http://arxiv.org/pdf/2508.10872v1)**

> **作者:** Anantha Narayanan; Battu Bhanu Teja; Pruthwik Mishra
>
> **备注:** 8 pages, 6 figures, 5 tables
>
> **摘要:** The increasing congestion of Low Earth Orbit (LEO) poses persistent challenges to the efficient deployment and safe operation of Earth observation satellites. Mission planners must now account not only for mission-specific requirements but also for the increasing collision risk with active satellites and space debris. This work presents a reinforcement learning framework using the Advantage Actor-Critic (A2C) algorithm to optimize satellite orbital parameters for precise terrestrial coverage within predefined surface radii. By formulating the problem as a Markov Decision Process (MDP) within a custom OpenAI Gymnasium environment, our method simulates orbital dynamics using classical Keplerian elements. The agent progressively learns to adjust five of the orbital parameters - semi-major axis, eccentricity, inclination, right ascension of ascending node, and the argument of perigee-to achieve targeted terrestrial coverage. Comparative evaluation against Proximal Policy Optimization (PPO) demonstrates A2C's superior performance, achieving 5.8x higher cumulative rewards (10.0 vs 9.263025) while converging in 31.5x fewer timesteps (2,000 vs 63,000). The A2C agent consistently meets mission objectives across diverse target coordinates while maintaining computational efficiency suitable for real-time mission planning applications. Key contributions include: (1) a TLE-based orbital simulation environment incorporating physics constraints, (2) validation of actor-critic methods' superiority over trust region approaches in continuous orbital control, and (3) demonstration of rapid convergence enabling adaptive satellite deployment. This approach establishes reinforcement learning as a computationally efficient alternative for scalable and intelligent LEO mission planning.
>
---
#### [new 018] Learning Task Execution Hierarchies for Redundant Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种自动学习冗余机器人任务执行层次结构的方法，结合强化学习与遗传编程，无需人工干预，通过成本函数优化实现动态任务优先级与控制策略的自动生成，提升复杂机器人任务适应性。**

- **链接: [http://arxiv.org/pdf/2508.10780v1](http://arxiv.org/pdf/2508.10780v1)**

> **作者:** Alessandro Adami; Aris Synodinos; Matteo Iovino; Ruggero Carli; Pietro Falco
>
> **摘要:** Modern robotic systems, such as mobile manipulators, humanoids, and aerial robots with arms, often possess high redundancy, enabling them to perform multiple tasks simultaneously. Managing this redundancy is key to achieving reliable and flexible behavior. A widely used approach is the Stack of Tasks (SoT), which organizes control objectives by priority within a unified framework. However, traditional SoTs are manually designed by experts, limiting their adaptability and accessibility. This paper introduces a novel framework that automatically learns both the hierarchy and parameters of a SoT from user-defined objectives. By combining Reinforcement Learning and Genetic Programming, the system discovers task priorities and control strategies without manual intervention. A cost function based on intuitive metrics such as precision, safety, and execution time guides the learning process. We validate our method through simulations and experiments on the mobile-YuMi platform, a dual-arm mobile manipulator with high redundancy. Results show that the learned SoTs enable the robot to dynamically adapt to changing environments and inputs, balancing competing objectives while maintaining robust task execution. This approach provides a general and user-friendly solution for redundancy management in complex robots, advancing human-centered robot programming and reducing the need for expert design.
>
---
#### [new 019] The SET Perceptual Factors Framework: Towards Assured Perception for Autonomous Systems
- **分类: cs.RO; cs.AI**

- **简介: 论文提出SET框架，通过状态树与因素树分析感知影响因素，量化不确定性以提升自主系统可靠性，解决感知失误导致的安全风险。**

- **链接: [http://arxiv.org/pdf/2508.10798v1](http://arxiv.org/pdf/2508.10798v1)**

> **作者:** Troi Williams
>
> **备注:** 4 pages, 4 figures, accepted to the Workshop on Public Trust in Autonomous Systems at the 2025 IEEE International Conference on Robotics & Automation
>
> **摘要:** Future autonomous systems promise significant societal benefits, yet their deployment raises concerns about safety and trustworthiness. A key concern is assuring the reliability of robot perception, as perception seeds safe decision-making. Failures in perception are often due to complex yet common environmental factors and can lead to accidents that erode public trust. To address this concern, we introduce the SET (Self, Environment, and Target) Perceptual Factors Framework. We designed the framework to systematically analyze how factors such as weather, occlusion, or sensor limitations negatively impact perception. To achieve this, the framework employs SET State Trees to categorize where such factors originate and SET Factor Trees to model how these sources and factors impact perceptual tasks like object detection or pose estimation. Next, we develop Perceptual Factor Models using both trees to quantify the uncertainty for a given task. Our framework aims to promote rigorous safety assurances and cultivate greater public understanding and trust in autonomous systems by offering a transparent and standardized method for identifying, modeling, and communicating perceptual risks.
>
---
#### [new 020] KDPE: A Kernel Density Estimation Strategy for Diffusion Policy Trajectory Selection
- **分类: cs.RO**

- **简介: 论文提出基于核密度估计(KDE)的KDPE策略，针对Diffusion Policy生成轨迹的噪声敏感性和数据异常问题，通过曼哈顿流形感知核过滤有害轨迹，降低测试时计算开销，提升单臂任务与真实机器人实验性能。**

- **链接: [http://arxiv.org/pdf/2508.10511v1](http://arxiv.org/pdf/2508.10511v1)**

> **作者:** Andrea Rosasco; Federico Ceola; Giulia Pasquale; Lorenzo Natale
>
> **备注:** 9th Conference on Robot Learning (CoRL 2025), Seoul, Korea
>
> **摘要:** Learning robot policies that capture multimodality in the training data has been a long-standing open challenge for behavior cloning. Recent approaches tackle the problem by modeling the conditional action distribution with generative models. One of these approaches is Diffusion Policy, which relies on a diffusion model to denoise random points into robot action trajectories. While achieving state-of-the-art performance, it has two main drawbacks that may lead the robot out of the data distribution during policy execution. First, the stochasticity of the denoising process can highly impact on the quality of generated trajectory of actions. Second, being a supervised learning approach, it can learn data outliers from the dataset used for training. Recent work focuses on mitigating these limitations by combining Diffusion Policy either with large-scale training or with classical behavior cloning algorithms. Instead, we propose KDPE, a Kernel Density Estimation-based strategy that filters out potentially harmful trajectories output of Diffusion Policy while keeping a low test-time computational overhead. For Kernel Density Estimation, we propose a manifold-aware kernel to model a probability density function for actions composed of end-effector Cartesian position, orientation, and gripper state. KDPE overall achieves better performance than Diffusion Policy on simulated single-arm tasks and real robot experiments. Additional material and code are available on our project page https://hsp-iit.github.io/KDPE/.
>
---
#### [new 021] Super LiDAR Reflectance for Robotic Perception
- **分类: cs.RO**

- **简介: 论文提出基于非重复扫描LiDAR（NRS-LiDAR）的稠密反射率图像生成框架，解决低成本LiDAR数据稀疏性问题，实现机器人感知任务中的反射率重建，包含数据集、网络及应用场景。**

- **链接: [http://arxiv.org/pdf/2508.10398v1](http://arxiv.org/pdf/2508.10398v1)**

> **作者:** Wei Gao; Jie Zhang; Mingle Zhao; Zhiyuan Zhang; Shu Kong; Maani Ghaffari; Dezhen Song; Cheng-Zhong Xu; Hui Kong
>
> **摘要:** Conventionally, human intuition often defines vision as a modality of passive optical sensing, while active optical sensing is typically regarded as measuring rather than the default modality of vision. However, the situation now changes: sensor technologies and data-driven paradigms empower active optical sensing to redefine the boundaries of vision, ushering in a new era of active vision. Light Detection and Ranging (LiDAR) sensors capture reflectance from object surfaces, which remains invariant under varying illumination conditions, showcasing significant potential in robotic perception tasks such as detection, recognition, segmentation, and Simultaneous Localization and Mapping (SLAM). These applications often rely on dense sensing capabilities, typically achieved by high-resolution, expensive LiDAR sensors. A key challenge with low-cost LiDARs lies in the sparsity of scan data, which limits their broader application. To address this limitation, this work introduces an innovative framework for generating dense LiDAR reflectance images from sparse data, leveraging the unique attributes of non-repeating scanning LiDAR (NRS-LiDAR). We tackle critical challenges, including reflectance calibration and the transition from static to dynamic scene domains, facilitating the reconstruction of dense reflectance images in real-world settings. The key contributions of this work include a comprehensive dataset for LiDAR reflectance image densification, a densification network tailored for NRS-LiDAR, and diverse applications such as loop closure and traffic lane detection using the generated dense reflectance images.
>
---
#### [new 022] MLM: Learning Multi-task Loco-Manipulation Whole-Body Control for Quadruped Robot with Arm
- **分类: cs.RO**

- **简介: 论文提出基于强化学习的MLM框架，解决四足机器人多任务协同控制问题，通过融合真实与模拟数据及轨迹预测网络，实现自主或远程操作下的多任务整体控制，验证了其在仿真与实际场景中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.10538v1](http://arxiv.org/pdf/2508.10538v1)**

> **作者:** Xin Liu; Bida Ma; Chenkun Qi; Yan Ding; Zhaxizhuoma; Guorong Zhang; Pengan Chen; Kehui Liu; Zhongjie Jia; Chuyue Guan; Yule Mo; Jiaqi Liu; Feng Gao; Jiangwei Zhong; Bin Zhao; Xuelong Li
>
> **摘要:** Whole-body loco-manipulation for quadruped robots with arm remains a challenging problem, particularly in achieving multi-task control. To address this, we propose MLM, a reinforcement learning framework driven by both real-world and simulation data. It enables a six-DoF robotic arm--equipped quadruped robot to perform whole-body loco-manipulation for multiple tasks autonomously or under human teleoperation. To address the problem of balancing multiple tasks during the learning of loco-manipulation, we introduce a trajectory library with an adaptive, curriculum-based sampling mechanism. This approach allows the policy to efficiently leverage real-world collected trajectories for learning multi-task loco-manipulation. To address deployment scenarios with only historical observations and to enhance the performance of policy execution across tasks with different spatial ranges, we propose a Trajectory-Velocity Prediction policy network. It predicts unobservable future trajectories and velocities. By leveraging extensive simulation data and curriculum-based rewards, our controller achieves whole-body behaviors in simulation and zero-shot transfer to real-world deployment. Ablation studies in simulation verify the necessity and effectiveness of our approach, while real-world experiments on the Go2 robot with an Airbot robotic arm demonstrate the policy's good performance in multi-task execution.
>
---
#### [new 023] CVIRO: A Consistent and Tightly-Coupled Visual-Inertial-Ranging Odometry on Lie Groups
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出基于Lie群的CVIRO系统，解决视觉-惯性-激光雷达里程计中不一致问题，通过纳入UWB锚点状态并利用Lie群不变性确保可观测性一致性，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2508.10867v1](http://arxiv.org/pdf/2508.10867v1)**

> **作者:** Yizhi Zhou; Ziwei Kang; Jiawei Xia; Xuan Wang
>
> **摘要:** Ultra Wideband (UWB) is widely used to mitigate drift in visual-inertial odometry (VIO) systems. Consistency is crucial for ensuring the estimation accuracy of a UWBaided VIO system. An inconsistent estimator can degrade localization performance, where the inconsistency primarily arises from two main factors: (1) the estimator fails to preserve the correct system observability, and (2) UWB anchor positions are assumed to be known, leading to improper neglect of calibration uncertainty. In this paper, we propose a consistent and tightly-coupled visual-inertial-ranging odometry (CVIRO) system based on the Lie group. Our method incorporates the UWB anchor state into the system state, explicitly accounting for UWB calibration uncertainty and enabling the joint and consistent estimation of both robot and anchor states. Furthermore, observability consistency is ensured by leveraging the invariant error properties of the Lie group. We analytically prove that the CVIRO algorithm naturally maintains the system's correct unobservable subspace, thereby preserving estimation consistency. Extensive simulations and experiments demonstrate that CVIRO achieves superior localization accuracy and consistency compared to existing methods.
>
---
#### [new 024] Probabilistic Latency Analysis of the Data Distribution Service in ROS 2
- **分类: cs.NI; cs.RO**

- **简介: 论文提出概率延迟分析方法，针对ROS 2 DDS在无线网络中的延迟问题，通过离散状态模型建模可靠传输过程，计算未确认消息概率分布和重传延迟，并验证其准确性。**

- **链接: [http://arxiv.org/pdf/2508.10413v1](http://arxiv.org/pdf/2508.10413v1)**

> **作者:** Sanghoon Lee; Hyung-Seok Park; Jiyeong Chae; Kyung-Joon Park
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Robot Operating System 2 (ROS 2) is now the de facto standard for robotic communication, pairing UDP transport with the Data Distribution Service (DDS) publish-subscribe middleware. DDS achieves reliability through periodic heartbeats that solicit acknowledgments for missing samples and trigger selective retransmissions. In lossy wireless networks, the tight coupling among heartbeat period, IP fragmentation, and retransmission interval obscures end to end latency behavior and leaves practitioners with little guidance on how to tune these parameters. To address these challenges, we propose a probabilistic latency analysis (PLA) that analytically models the reliable transmission process of ROS 2 DDS communication using a discrete state approach. By systematically analyzing both middleware level and transport level events, PLA computes the steady state probability distribution of unacknowledged messages and the retransmission latency. We validate our PLA across 270 scenarios, exploring variations in packet delivery ratios, message sizes, and both publishing and retransmission intervals, demonstrating a close alignment between analytical predictions and experimental results. Our findings establish a theoretical basis to systematically optimize reliability, latency, and performance in wireless industrial robotics.
>
---
#### [new 025] Scaling Up without Fading Out: Goal-Aware Sparse GNN for RL-based Generalized Planning
- **分类: cs.AI; cs.RO**

- **简介: 论文提出稀疏目标感知的GNN，解决RL中大规模规划的稀疏性与信息过载问题，提升政策泛化与成功率。**

- **链接: [http://arxiv.org/pdf/2508.10747v1](http://arxiv.org/pdf/2508.10747v1)**

> **作者:** Sangwoo Jeon; Juchul Shin; Gyeong-Tae Kim; YeonJe Cho; Seongwoo Kim
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Generalized planning using deep reinforcement learning (RL) combined with graph neural networks (GNNs) has shown promising results in various symbolic planning domains described by PDDL. However, existing approaches typically represent planning states as fully connected graphs, leading to a combinatorial explosion in edge information and substantial sparsity as problem scales grow, especially evident in large grid-based environments. This dense representation results in diluted node-level information, exponentially increases memory requirements, and ultimately makes learning infeasible for larger-scale problems. To address these challenges, we propose a sparse, goal-aware GNN representation that selectively encodes relevant local relationships and explicitly integrates spatial features related to the goal. We validate our approach by designing novel drone mission scenarios based on PDDL within a grid world, effectively simulating realistic mission execution environments. Our experimental results demonstrate that our method scales effectively to larger grid sizes previously infeasible with dense graph representations and substantially improves policy generalization and success rates. Our findings provide a practical foundation for addressing realistic, large-scale generalized planning tasks.
>
---
#### [new 026] SpaRC-AD: A Baseline for Radar-Camera Fusion in End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 论文提出SpaRC-AD框架，通过雷达-摄像头融合解决视觉方法在恶劣条件下的局限，利用稀疏3D特征对齐和多普勒估计提升3D场景表示，改善3D检测、跟踪、运动预测等任务性能，应用于多个安全关键场景。**

- **链接: [http://arxiv.org/pdf/2508.10567v1](http://arxiv.org/pdf/2508.10567v1)**

> **作者:** Philipp Wolters; Johannes Gilg; Torben Teepe; Gerhard Rigoll
>
> **备注:** 8 pages, 4 figures, 5 tables
>
> **摘要:** End-to-end autonomous driving systems promise stronger performance through unified optimization of perception, motion forecasting, and planning. However, vision-based approaches face fundamental limitations in adverse weather conditions, partial occlusions, and precise velocity estimation - critical challenges in safety-sensitive scenarios where accurate motion understanding and long-horizon trajectory prediction are essential for collision avoidance. To address these limitations, we propose SpaRC-AD, a query-based end-to-end camera-radar fusion framework for planning-oriented autonomous driving. Through sparse 3D feature alignment, and doppler-based velocity estimation, we achieve strong 3D scene representations for refinement of agent anchors, map polylines and motion modelling. Our method achieves strong improvements over the state-of-the-art vision-only baselines across multiple autonomous driving tasks, including 3D detection (+4.8% mAP), multi-object tracking (+8.3% AMOTA), online mapping (+1.8% mAP), motion prediction (-4.0% mADE), and trajectory planning (-0.1m L2 and -9% TPC). We achieve both spatial coherence and temporal consistency on multiple challenging benchmarks, including real-world open-loop nuScenes, long-horizon T-nuScenes, and closed-loop simulator Bench2Drive. We show the effectiveness of radar-based fusion in safety-critical scenarios where accurate motion understanding and long-horizon trajectory prediction are essential for collision avoidance. The source code of all experiments is available at https://phi-wol.github.io/sparcad/
>
---
## 更新

#### [replaced 001] Motion Planning Diffusion: Learning and Adapting Robot Motion Planning with Diffusion Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.19948v3](http://arxiv.org/pdf/2412.19948v3)**

> **作者:** J. Carvalho; A. Le; P. Kicki; D. Koert; J. Peters
>
> **摘要:** The performance of optimization-based robot motion planning algorithms is highly dependent on the initial solutions, commonly obtained by running a sampling-based planner to obtain a collision-free path. However, these methods can be slow in high-dimensional and complex scenes and produce non-smooth solutions. Given previously solved path-planning problems, it is highly desirable to learn their distribution and use it as a prior for new similar problems. Several works propose utilizing this prior to bootstrap the motion planning problem, either by sampling initial solutions from it, or using its distribution in a maximum-a-posterior formulation for trajectory optimization. In this work, we introduce Motion Planning Diffusion (MPD), an algorithm that learns trajectory distribution priors with diffusion models. These generative models have shown increasing success in encoding multimodal data and have desirable properties for gradient-based motion planning, such as cost guidance. Given a motion planning problem, we construct a cost function and sample from the posterior distribution using the learned prior combined with the cost function gradients during the denoising process. Instead of learning the prior on all trajectory waypoints, we propose learning a lower-dimensional representation of a trajectory using linear motion primitives, particularly B-spline curves. This parametrization guarantees that the generated trajectory is smooth, can be interpolated at higher frequencies, and needs fewer parameters than a dense waypoint representation. We demonstrate the results of our method ranging from simple 2D to more complex tasks using a 7-dof robot arm manipulator. In addition to learning from simulated data, we also use human demonstrations on a real-world pick-and-place task.
>
---
#### [replaced 002] Implicit Safe Set Algorithm for Provably Safe Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.02754v2](http://arxiv.org/pdf/2405.02754v2)**

> **作者:** Weiye Zhao; Feihan Li; Changliu Liu
>
> **备注:** Accepted to Journal of Artificial Intelligence Research. arXiv admin note: text overlap with arXiv:2308.13140
>
> **摘要:** Deep reinforcement learning (DRL) has demonstrated remarkable performance in many continuous control tasks. However, a significant obstacle to the real-world application of DRL is the lack of safety guarantees. Although DRL agents can satisfy system safety in expectation through reward shaping, designing agents to consistently meet hard constraints (e.g., safety specifications) at every time step remains a formidable challenge. In contrast, existing work in the field of safe control provides guarantees on persistent satisfaction of hard safety constraints. However, these methods require explicit analytical system dynamics models to synthesize safe control, which are typically inaccessible in DRL settings. In this paper, we present a model-free safe control algorithm, the implicit safe set algorithm, for synthesizing safeguards for DRL agents that ensure provable safety throughout training. The proposed algorithm synthesizes a safety index (barrier certificate) and a subsequent safe control law solely by querying a black-box dynamic function (e.g., a digital twin simulator). Moreover, we theoretically prove that the implicit safe set algorithm guarantees finite time convergence to the safe set and forward invariance for both continuous-time and discrete-time systems. We validate the proposed algorithm on the state-of-the-art Safety Gym benchmark, where it achieves zero safety violations while gaining $95\% \pm 9\%$ cumulative reward compared to state-of-the-art safe DRL methods. Furthermore, the resulting algorithm scales well to high-dimensional systems with parallel computing.
>
---
#### [replaced 003] Traversability analysis with vision and terrain probing for safe legged robot navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2209.00334v2](http://arxiv.org/pdf/2209.00334v2)**

> **作者:** Garen Haddeler; Meng Yee Michael Chuah; Yangwei You; Jianle Chan; Albertus H. Adiwahono; Wei Yun Yau; Chee-Meng Chew
>
> **摘要:** Inspired by human behavior when traveling over unknown terrain, this study proposes the use of probing strategies and integrates them into a traversability analysis framework to address safe navigation on unknown rough terrain. Our framework integrates collapsibility information into our existing traversability analysis, as vision and geometric information alone could be misled by unpredictable non-rigid terrains such as soft soil, bush area, or water puddles. With the new traversability analysis framework, our robot has a more comprehensive assessment of unpredictable terrain, which is critical for its safety in outdoor environments. The pipeline first identifies the terrain's geometric and semantic properties using an RGB-D camera and desired probing locations on questionable terrains. These regions are probed using a force sensor to determine the risk of terrain collapsing when the robot steps over it. This risk is formulated as a collapsibility metric, which estimates an unpredictable region's ground collapsibility. Thereafter, the collapsibility metric, together with geometric and semantic spatial data, is combined and analyzed to produce global and local traversability grid maps. These traversability grid maps tell the robot whether it is safe to step over different regions of the map. The grid maps are then utilized to generate optimal paths for the robot to safely navigate to its goal. Our approach has been successfully verified on a quadrupedal robot in both simulation and real-world experiments.
>
---
#### [replaced 004] Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.05294v2](http://arxiv.org/pdf/2508.05294v2)**

> **作者:** Sahar Salimpour; Lei Fu; Farhad Keramat; Leonardo Militano; Giovanni Toffetti; Harry Edelman; Jorge Peña Queralta
>
> **摘要:** Foundation models, including large language models (LLMs) and vision-language models (VLMs), have recently enabled novel approaches to robot autonomy and human-robot interfaces. In parallel, vision-language-action models (VLAs) or large behavior models (LBMs) are increasing the dexterity and capabilities of robotic systems. This survey paper focuses on those works advancing towards agentic applications and architectures. This includes initial efforts exploring GPT-style interfaces to tooling, as well as more complex system where AI agents are coordinators, planners, perception actors, or generalist interfaces. Such agentic architectures allow robots to reason over natural language instructions, invoke APIs, plan task sequences, or assist in operations and diagnostics. In addition to peer-reviewed research, due to the fast-evolving nature of the field, we highlight and include community-driven projects, ROS packages, and industrial frameworks that show emerging trends. We propose a taxonomy for classifying model integration approaches and present a comparative analysis of the role that agents play in different solutions in today's literature.
>
---
#### [replaced 005] RobustDexGrasp: Robust Dexterous Grasping of General Objects
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.05287v3](http://arxiv.org/pdf/2504.05287v3)**

> **作者:** Hui Zhang; Zijian Wu; Linyi Huang; Sammy Christen; Jie Song
>
> **备注:** Camera ready for CoRL2025. Project Page: https://zdchan.github.io/Robust_DexGrasp/
>
> **摘要:** The ability to robustly grasp a variety of objects is essential for dexterous robots. In this paper, we present a framework for zero-shot dynamic dexterous grasping using single-view visual inputs, designed to be resilient to various disturbances. Our approach utilizes a hand-centric object shape representation based on dynamic distance vectors between finger joints and object surfaces. This representation captures the local shape around potential contact regions rather than focusing on detailed global object geometry, thereby enhancing generalization to shape variations and uncertainties. To address perception limitations, we integrate a privileged teacher policy with a mixed curriculum learning approach, allowing the student policy to effectively distill grasping capabilities and explore for adaptation to disturbances. Trained in simulation, our method achieves success rates of 97.0% across 247,786 simulated objects and 94.6% across 512 real objects, demonstrating remarkable generalization. Quantitative and qualitative results validate the robustness of our policy against various disturbances.
>
---
#### [replaced 006] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.24381v2](http://arxiv.org/pdf/2503.24381v2)**

> **作者:** Yuping Wang; Xiangyu Huang; Xiaokang Sun; Mingxuan Yan; Shuo Xing; Zhengzhong Tu; Jiachen Li
>
> **备注:** IEEE/CVF International Conference on Computer Vision (ICCV 2025); Project website: https://uniocc.github.io/
>
> **摘要:** We introduce UniOcc, a comprehensive, unified benchmark and toolkit for occupancy forecasting (i.e., predicting future occupancies based on historical information) and occupancy prediction (i.e., predicting current-frame occupancy from camera images. UniOcc unifies the data from multiple real-world datasets (i.e., nuScenes, Waymo) and high-fidelity driving simulators (i.e., CARLA, OpenCOOD), providing 2D/3D occupancy labels and annotating innovative per-voxel flows. Unlike existing studies that rely on suboptimal pseudo labels for evaluation, UniOcc incorporates novel evaluation metrics that do not depend on ground-truth labels, enabling robust assessment on additional aspects of occupancy quality. Through extensive experiments on state-of-the-art models, we demonstrate that large-scale, diverse training data and explicit flow information significantly enhance occupancy prediction and forecasting performance. Our data and code are available at https://uniocc.github.io/.
>
---
#### [replaced 007] WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.09560v2](http://arxiv.org/pdf/2508.09560v2)**

> **作者:** Jiahao Wen; Hang Yu; Zhedong Zheng
>
> **摘要:** Visual geo-localization for drones faces critical degradation under weather perturbations, \eg, rain and fog, where existing methods struggle with two inherent limitations: 1) Heavy reliance on limited weather categories that constrain generalization, and 2) Suboptimal disentanglement of entangled scene-weather features through pseudo weather categories. We present WeatherPrompt, a multi-modality learning paradigm that establishes weather-invariant representations through fusing the image embedding with the text context. Our framework introduces two key contributions: First, a Training-free Weather Reasoning mechanism that employs off-the-shelf large multi-modality models to synthesize multi-weather textual descriptions through human-like reasoning. It improves the scalability to unseen or complex weather, and could reflect different weather strength. Second, to better disentangle the scene and weather feature, we propose a multi-modality framework with the dynamic gating mechanism driven by the text embedding to adaptively reweight and fuse visual features across modalities. The framework is further optimized by the cross-modal objectives, including image-text contrastive learning and image-text matching, which maps the same scene with different weather conditions closer in the respresentation space. Extensive experiments validate that, under diverse weather conditions, our method achieves competitive recall rates compared to state-of-the-art drone geo-localization methods. Notably, it improves Recall@1 by +13.37\% under night conditions and by 18.69\% under fog and snow conditions.
>
---
#### [replaced 008] Advancing MAPF towards the Real World: A Scalable Multi-Agent Realistic Testbed (SMART)
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.04798v2](http://arxiv.org/pdf/2503.04798v2)**

> **作者:** Jingtian Yan; Zhifei Li; William Kang; Kevin Zheng; Yulun Zhang; Zhe Chen; Yue Zhang; Daniel Harabor; Stephen F. Smith; Jiaoyang Li
>
> **摘要:** We present Scalable Multi-Agent Realistic Testbed (SMART), a realistic and efficient software tool for evaluating Multi-Agent Path Finding (MAPF) algorithms. MAPF focuses on planning collision-free paths for a group of agents. While state-ofthe-art MAPF algorithms can plan paths for hundreds of robots in seconds, they often rely on simplified robot models, making their real-world performance unclear. Researchers typically lack access to hundreds of physical robots in laboratory settings to evaluate the algorithms. Meanwhile, industrial professionals who lack expertise in MAPF require an easy-to-use simulator to efficiently test and understand the performance of MAPF algorithms in their specific settings. SMART fills this gap with several advantages: (1) SMART uses physics-engine-based simulators to create realistic simulation environments, accounting for complex real-world factors such as robot kinodynamics and execution uncertainties, (2) SMART uses an execution monitor framework based on the Action Dependency Graph, facilitating seamless integration with various MAPF algorithms and robot models, and (3) SMART scales to thousands of robots. The code is publicly available at https://github.com/smart-mapf/smart.
>
---
#### [replaced 009] Optimizing Force Signals from Human Demonstrations of In-Contact Motions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.15608v2](http://arxiv.org/pdf/2507.15608v2)**

> **作者:** Johannes Hartwig; Fabian Viessmann; Dominik Henrich
>
> **备注:** This is a preprint of a chapter the following work (and accepted for publication): Annals of Scientific Society for Assembly, Handling and Industrial Robotics 2024. The final authenticated version is will be linked here: http://dx.doi.org/[tba]
>
> **摘要:** For non-robot-programming experts, kinesthetic guiding can be an intuitive input method, as robot programming of in-contact tasks is becoming more prominent. However, imprecise and noisy input signals from human demonstrations pose problems when reproducing motions directly or using the signal as input for machine learning methods. This paper explores optimizing force signals to correspond better to the human intention of the demonstrated signal. We compare different signal filtering methods and propose a peak detection method for dealing with first-contact deviations in the signal. The evaluation of these methods considers a specialized error criterion between the input and the human-intended signal. In addition, we analyze the critical parameters' influence on the filtering methods. The quality for an individual motion could be increased by up to \SI{20}{\percent} concerning the error criterion. The proposed contribution can improve the usability of robot programming and the interaction between humans and robots.
>
---
#### [replaced 010] Split Covariance Intersection Filter Based Visual Localization With Accurate AprilTag Map For Warehouse Robot Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2310.17879v3](http://arxiv.org/pdf/2310.17879v3)**

> **作者:** Susu Fang; Yanhao Li; Hao Li
>
> **摘要:** Accurate and efficient localization with conveniently-established map is the fundamental requirement for mobile robot operation in warehouse environments. An accurate AprilTag map can be conveniently established with the help of LiDAR-based SLAM. It is true that a LiDAR-based system is usually not commercially competitive in contrast with a vision-based system, yet fortunately for warehouse applications, only a single LiDAR-based SLAM system is needed to establish an accurate AprilTag map, whereas a large amount of visual localization systems can share this established AprilTag map for their own operations. Therefore, the cost of a LiDAR-based SLAM system is actually shared by the large amount of visual localization systems, and turns to be acceptable and even negligible for practical warehouse applications. Once an accurate AprilTag map is available, visual localization is realized as recursive estimation that fuses AprilTag measurements (i.e. AprilTag detection results) and robot motion data. AprilTag measurements may be nonlinear partial measurements; this can be handled by the well-known extended Kalman filter (EKF) in the spirit of local linearization. AprilTag measurements tend to have temporal correlation as well; however, this cannot be reasonably handled by the EKF. The split covariance intersection filter (Split CIF) is adopted to handle temporal correlation among AprilTag measurements. The Split CIF (in the spirit of local linearization) can also handle AprilTag nonlinear partial measurements. The Split CIF based visual localization system incorporates a measurement adaptive mechanism to handle outliers in AprilTag measurements and adopts a dynamic initialization mechanism to address the kidnapping problem. A comparative study in real warehouse environments demonstrates the potential and advantage of the Split CIF based visual localization solution.
>
---
#### [replaced 011] Real-time Digital Double Framework to Predict Collapsible Terrains for Legged Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2209.09508v2](http://arxiv.org/pdf/2209.09508v2)**

> **作者:** Garen Haddeler; Hari P. Palanivelu; Yung Chuen Ng; Fabien Colonnier; Albertus H. Adiwahono; Zhibin Li; Chee-Meng Chew; Meng Yee Michael Chuah
>
> **备注:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Preprint version. Accepted June 2022
>
> **摘要:** Inspired by the digital twinning systems, a novel real-time digital double framework is developed to enhance robot perception of the terrain conditions. Based on the very same physical model and motion control, this work exploits the use of such simulated digital double synchronized with a real robot to capture and extract discrepancy information between the two systems, which provides high dimensional cues in multiple physical quantities to represent differences between the modelled and the real world. Soft, non-rigid terrains cause common failures in legged locomotion, whereby visual perception solely is insufficient in estimating such physical properties of terrains. We used digital double to develop the estimation of the collapsibility, which addressed this issue through physical interactions during dynamic walking. The discrepancy in sensory measurements between the real robot and its digital double are used as input of a learning-based algorithm for terrain collapsibility analysis. Although trained only in simulation, the learned model can perform collapsibility estimation successfully in both simulation and real world. Our evaluation of results showed the generalization to different scenarios and the advantages of the digital double to reliably detect nuances in ground conditions.
>
---
#### [replaced 012] Visual SLAMMOT Considering Multiple Motion Models
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.19134v2](http://arxiv.org/pdf/2411.19134v2)**

> **作者:** Peilin Tian; Hao Li
>
> **摘要:** Simultaneous Localization and Mapping (SLAM) and Multi-Object Tracking (MOT) are pivotal tasks in the realm of autonomous driving, attracting considerable research attention. While SLAM endeavors to generate real-time maps and determine the vehicle's pose in unfamiliar settings, MOT focuses on the real-time identification and tracking of multiple dynamic objects. Despite their importance, the prevalent approach treats SLAM and MOT as independent modules within an autonomous vehicle system, leading to inherent limitations. Classical SLAM methodologies often rely on a static environment assumption, suitable for indoor rather than dynamic outdoor scenarios. Conversely, conventional MOT techniques typically rely on the vehicle's known state, constraining the accuracy of object state estimations based on this prior. To address these challenges, previous efforts introduced the unified SLAMMOT paradigm, yet primarily focused on simplistic motion patterns. In our team's previous work IMM-SLAMMOT\cite{IMM-SLAMMOT}, we present a novel methodology incorporating consideration of multiple motion models into SLAMMOT i.e. tightly coupled SLAM and MOT, demonstrating its efficacy in LiDAR-based systems. This paper studies feasibility and advantages of instantiating this methodology as visual SLAMMOT, bridging the gap between LiDAR and vision-based sensing mechanisms. Specifically, we propose a solution of visual SLAMMOT considering multiple motion models and validate the inherent advantages of IMM-SLAMMOT in the visual domain.
>
---
#### [replaced 013] Episodic Memory Verbalization using Hierarchical Representations of Life-Long Robot Experience
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.17702v3](http://arxiv.org/pdf/2409.17702v3)**

> **作者:** Leonard Bärmann; Chad DeChant; Joana Plewnia; Fabian Peller-Konrad; Daniel Bauer; Tamim Asfour; Alex Waibel
>
> **备注:** Humanoids 2025. Code, data and demo videos at https://hierarchical-emv.github.io
>
> **摘要:** Verbalization of robot experience, i.e., summarization of and question answering about a robot's past, is a crucial ability for improving human-robot interaction. Previous works applied rule-based systems or fine-tuned deep models to verbalize short (several-minute-long) streams of episodic data, limiting generalization and transferability. In our work, we apply large pretrained models to tackle this task with zero or few examples, and specifically focus on verbalizing life-long experiences. For this, we derive a tree-like data structure from episodic memory (EM), with lower levels representing raw perception and proprioception data, and higher levels abstracting events to natural language concepts. Given such a hierarchical representation built from the experience stream, we apply a large language model as an agent to interactively search the EM given a user's query, dynamically expanding (initially collapsed) tree nodes to find the relevant information. The approach keeps computational costs low even when scaling to months of robot experience data. We evaluate our method on simulated household robot data, human egocentric videos, and real-world robot recordings, demonstrating its flexibility and scalability.
>
---
#### [replaced 014] TAR: Teacher-Aligned Representations via Contrastive Learning for Quadrupedal Locomotion
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.20839v2](http://arxiv.org/pdf/2503.20839v2)**

> **作者:** Amr Mousa; Neil Karavis; Michele Caprio; Wei Pan; Richard Allmendinger
>
> **备注:** This work has been accepted for publication at the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Quadrupedal locomotion via Reinforcement Learning (RL) is commonly addressed using the teacher-student paradigm, where a privileged teacher guides a proprioceptive student policy. However, key challenges such as representation misalignment between privileged teacher and proprioceptive-only student, covariate shift due to behavioral cloning, and lack of deployable adaptation; lead to poor generalization in real-world scenarios. We propose Teacher-Aligned Representations via Contrastive Learning (TAR), a framework that leverages privileged information with self-supervised contrastive learning to bridge this gap. By aligning representations to a privileged teacher in simulation via contrastive objectives, our student policy learns structured latent spaces and exhibits robust generalization to Out-of-Distribution (OOD) scenarios, surpassing the fully privileged "Teacher". Results showed accelerated training by 2x compared to state-of-the-art baselines to achieve peak performance. OOD scenarios showed better generalization by 40% on average compared to existing methods. Moreover, TAR transitions seamlessly into learning during deployment without requiring privileged states, setting a new benchmark in sample-efficient, adaptive locomotion and enabling continual fine-tuning in real-world scenarios. Open-source code and videos are available at https://amrmousa.com/TARLoco/.
>
---
#### [replaced 015] LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.11528v3](http://arxiv.org/pdf/2505.11528v3)**

> **作者:** Yuhang Huang; Jiazhao Zhang; Shilong Zou; Xinwang Liu; Ruizhen Hu; Kai Xu
>
> **备注:** CoRL 2025
>
> **摘要:** Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
>
---
#### [replaced 016] GraspClutter6D: A Large-scale Real-world Dataset for Robust Perception and Grasping in Cluttered Scenes
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06866v3](http://arxiv.org/pdf/2504.06866v3)**

> **作者:** Seunghyeok Back; Joosoon Lee; Kangmin Kim; Heeseon Rho; Geonhyup Lee; Raeyoung Kang; Sangbeom Lee; Sangjun Noh; Youngjin Lee; Taeyeop Lee; Kyoobin Lee
>
> **备注:** Accepted at IEEE Robotics and Automation Letters (RA-L). Project Websites: https://sites.google.com/view/graspclutter6d
>
> **摘要:** Robust grasping in cluttered environments remains an open challenge in robotics. While benchmark datasets have significantly advanced deep learning methods, they mainly focus on simplistic scenes with light occlusion and insufficient diversity, limiting their applicability to practical scenarios. We present GraspClutter6D, a large-scale real-world grasping dataset featuring: (1) 1,000 highly cluttered scenes with dense arrangements (14.1 objects/scene, 62.6\% occlusion), (2) comprehensive coverage across 200 objects in 75 environment configurations (bins, shelves, and tables) captured using four RGB-D cameras from multiple viewpoints, and (3) rich annotations including 736K 6D object poses and 9.3B feasible robotic grasps for 52K RGB-D images. We benchmark state-of-the-art segmentation, object pose estimation, and grasp detection methods to provide key insights into challenges in cluttered environments. Additionally, we validate the dataset's effectiveness as a training resource, demonstrating that grasping networks trained on GraspClutter6D significantly outperform those trained on existing datasets in both simulation and real-world experiments. The dataset, toolkit, and annotation tools are publicly available on our project website: https://sites.google.com/view/graspclutter6d.
>
---
#### [replaced 017] Tactile Aware Dynamic Obstacle Avoidance in Crowded Environment with Deep Reinforcement Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2406.13434v2](http://arxiv.org/pdf/2406.13434v2)**

> **作者:** Yung Chuen Ng; Qi Wen Shervina Lim; Chun Ye Tan; Zhen Hao Gan; Meng Yee Michael Chuah
>
> **摘要:** Mobile robots operating in crowded environments require the ability to navigate among humans and surrounding obstacles efficiently while adhering to safety standards and socially compliant mannerisms. This scale of the robot navigation problem may be classified as both a local path planning and trajectory optimization problem. This work presents an array of force sensors that act as a tactile layer to complement the use of a LiDAR for the purpose of inducing awareness of contact with any surrounding objects within immediate vicinity of a mobile robot undetected by LiDARs. By incorporating the tactile layer, the robot can take more risks in its movements and possibly go right up to an obstacle or wall, and gently squeeze past it. In addition, we built up a simulation platform via Pybullet which integrates Robot Operating System (ROS) and reinforcement learning (RL) together. A touch-aware neural network model was trained on it to create an RL-based local path planner for dynamic obstacle avoidance. Our proposed method was demonstrated successfully on an omni-directional mobile robot who was able to navigate in a crowded environment with high agility and versatility in movement, while not being overly sensitive to nearby obstacles-not-in-contact.
>
---
#### [replaced 018] Detection and Tracking of MAVs Using a Rosette Scanning Pattern LiDAR
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2408.08555v3](http://arxiv.org/pdf/2408.08555v3)**

> **作者:** Sándor Gazdag; Tom Möller; Anita Keszler; András L. Majdik
>
> **摘要:** The use of commercial Micro Aerial Vehicles (MAVs) has surged in the past decade, offering societal benefits but also raising risks such as airspace violations and privacy concerns. Due to the increased security risks, the development of autonomous drone detection and tracking systems has become a priority. In this study, we tackle this challenge, by using non-repetitive rosette scanning pattern LiDARs, particularly focusing on increasing the detection distance by leveraging the characteristics of the sensor. The presented method utilizes a particle filter with a velocity component for the detection and tracking of the drone, which offers added re-detection capability. A Pan-Tilt platform is utilized to take advantage of the specific characteristics of the rosette scanning pattern LiDAR by keeping the tracked object in the center where the measurement is most dense. The detection capabilities and accuracy of the system are validated through indoor experiments, while the maximum detection distance is shown in our outdoor experiments. Our approach achieved accuracy on par with the state-of-the-art indoor method while increasing the maximum detection range by approximately 80\% beyond the state-of-the-art outdoor method.
>
---
#### [replaced 019] MPPI-Generic: A CUDA Library for Stochastic Trajectory Optimization
- **分类: cs.MS; cs.DC; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2409.07563v3](http://arxiv.org/pdf/2409.07563v3)**

> **作者:** Bogdan Vlahov; Jason Gibson; Manan Gandhi; Evangelos A. Theodorou
>
> **备注:** Added missing Acknowledgements section
>
> **摘要:** This paper introduces a new C++/CUDA library for GPU-accelerated stochastic optimization called MPPI-Generic. It provides implementations of Model Predictive Path Integral control, Tube-Model Predictive Path Integral Control, and Robust Model Predictive Path Integral Control, and allows for these algorithms to be used across many pre-existing dynamics models and cost functions. Furthermore, researchers can create their own dynamics models or cost functions following our API definitions without needing to change the actual Model Predictive Path Integral Control code. Finally, we compare computational performance to other popular implementations of Model Predictive Path Integral Control over a variety of GPUs to show the real-time capabilities our library can allow for. Library code can be found at: https://acdslab.github.io/mppi-generic-website/ .
>
---
#### [replaced 020] RINO: Accurate, Robust Radar-Inertial Odometry with Non-Iterative Estimation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.07699v4](http://arxiv.org/pdf/2411.07699v4)**

> **作者:** Shuocheng Yang; Yueming Cao; Shengbo Eben Li; Jianqiang Wang; Shaobing Xu
>
> **摘要:** Odometry in adverse weather conditions, such as fog, rain, and snow, presents significant challenges, as traditional vision and LiDAR-based methods often suffer from degraded performance. Radar-Inertial Odometry (RIO) has emerged as a promising solution due to its resilience in such environments. In this paper, we present RINO, a non-iterative RIO framework implemented in an adaptively loosely coupled manner. Building upon ORORA as the baseline for radar odometry, RINO introduces several key advancements, including improvements in keypoint extraction, motion distortion compensation, and pose estimation via an adaptive voting mechanism. This voting strategy facilitates efficient polynomial-time optimization while simultaneously quantifying the uncertainty in the radar module's pose estimation. The estimated uncertainty is subsequently integrated into the maximum a posteriori (MAP) estimation within a Kalman filter framework. Unlike prior loosely coupled odometry systems, RINO not only retains the global and robust registration capabilities of the radar component but also dynamically accounts for the real-time operational state of each sensor during fusion. Experimental results conducted on publicly available datasets demonstrate that RINO reduces translation and rotation errors by 1.06% and 0.09{\deg}/100m, respectively, when compared to the baseline method, thus significantly enhancing its accuracy. Furthermore, RINO achieves performance comparable to state-of-the-art methods.
>
---
#### [replaced 021] Robotic Ultrasound-Guided Femoral Artery Reconstruction of Anatomically-Representative Phantoms
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.06795v2](http://arxiv.org/pdf/2503.06795v2)**

> **作者:** Lidia Al-Zogbi; Deepak Raina; Vinciya Pandian; Thorsten Fleiter; Axel Krieger
>
> **摘要:** Femoral artery access is essential for numerous clinical procedures, including diagnostic angiography, therapeutic catheterization, and emergency interventions. Despite its critical role, successful vascular access remains challenging due to anatomical variability, overlying adipose tissue, and the need for precise ultrasound (US) guidance. Needle placement errors can result in severe complications, thereby limiting the procedure to highly skilled clinicians operating in controlled hospital environments. While robotic systems have shown promise in addressing these challenges through autonomous scanning and vessel reconstruction, clinical translation remains limited due to reliance on simplified phantom models that fail to capture human anatomical complexity. In this work, we present a method for autonomous robotic US scanning of bifurcated femoral arteries, and validate it on five vascular phantoms created from real patient computed tomography (CT) data. Additionally, we introduce a video-based deep learning US segmentation network tailored for vascular imaging, enabling improved 3D arterial reconstruction. The proposed network achieves a Dice score of 89.21% and an Intersection over Union of 80.54% on a new vascular dataset. The reconstructed artery centerline is evaluated against ground truth CT data, showing an average L2 error of 0.91+/-0.70 mm, with an average Hausdorff distance of 4.36+/-1.11mm. This study is the first to validate an autonomous robotic system for US scanning of the femoral artery on a diverse set of patient-specific phantoms, introducing a more advanced framework for evaluating robotic performance in vascular imaging and intervention.
>
---
#### [replaced 022] VPOcc: Exploiting Vanishing Point for 3D Semantic Occupancy Prediction
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2408.03551v2](http://arxiv.org/pdf/2408.03551v2)**

> **作者:** Junsu Kim; Junhee Lee; Ukcheol Shin; Jean Oh; Kyungdon Joo
>
> **摘要:** Understanding 3D scenes semantically and spatially is crucial for the safe navigation of robots and autonomous vehicles, aiding obstacle avoidance and accurate trajectory planning. Camera-based 3D semantic occupancy prediction, which infers complete voxel grids from 2D images, is gaining importance in robot vision for its resource efficiency compared to 3D sensors. However, this task inherently suffers from a 2D-3D discrepancy, where objects of the same size in 3D space appear at different scales in a 2D image depending on their distance from the camera due to perspective projection. To tackle this issue, we propose a novel framework called VPOcc that leverages a vanishing point (VP) to mitigate the 2D-3D discrepancy at both the pixel and feature levels. As a pixel-level solution, we introduce a VPZoomer module, which warps images by counteracting the perspective effect using a VP-based homography transformation. In addition, as a feature-level solution, we propose a VP-guided cross-attention (VPCA) module that performs perspective-aware feature aggregation, utilizing 2D image features that are more suitable for 3D space. Lastly, we integrate two feature volumes extracted from the original and warped images to compensate for each other through a spatial volume fusion (SVF) module. By effectively incorporating VP into the network, our framework achieves improvements in both IoU and mIoU metrics on SemanticKITTI and SSCBench-KITTI360 datasets. Additional details are available at https://vision3d-lab.github.io/vpocc/.
>
---
#### [replaced 023] Estimation of Payload Inertial Parameters from Human Demonstrations by Hand Guiding
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.15604v2](http://arxiv.org/pdf/2507.15604v2)**

> **作者:** Johannes Hartwig; Philipp Lienhardt; Dominik Henrich
>
> **备注:** This is a preprint of a chapter the following work (and accepted for publication): Annals of Scientific Society for Assembly, Handling and Industrial Robotics 2025. The final authenticated version is will be linked here: http://dx.doi.org/[tba]
>
> **摘要:** As the availability of cobots increases, it is essential to address the needs of users with little to no programming knowledge to operate such systems efficiently. Programming concepts often use intuitive interaction modalities, such as hand guiding, to address this. When programming in-contact motions, such frameworks require knowledge of the robot tool's payload inertial parameters (PIP) in addition to the demonstrated velocities and forces to ensure effective hybrid motion-force control. This paper aims to enable non-expert users to program in-contact motions more efficiently by eliminating the need for a dedicated PIP calibration, thereby enabling flexible robot tool changes. Since demonstrated tasks generally also contain motions with non-contact, our approach uses these parts to estimate the robot's PIP using established estimation techniques. The results show that the estimation of the payload's mass is accurate, whereas the center of mass and the inertia tensor are affected by noise and a lack of excitation. Overall, these findings show the feasibility of PIP estimation during hand guiding but also highlight the need for sufficient payload accelerations for an accurate estimation.
>
---
