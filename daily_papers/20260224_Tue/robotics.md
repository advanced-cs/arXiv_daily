# 机器人 cs.RO

- **最新发布 76 篇**

- **更新 40 篇**

## 最新发布

#### [new 001] Toward AI Autonomous Navigation for Mechanical Thrombectomy using Hierarchical Modular Multi-agent Reinforcement Learning (HM-MARL)
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机械取栓自主导航任务，旨在解决复杂血管路径中的自主导航问题。通过HM-MARL框架实现两设备的高效、泛化导航。**

- **链接: [https://arxiv.org/pdf/2602.18663v1](https://arxiv.org/pdf/2602.18663v1)**

> **作者:** Harry Robertshaw; Nikola Fischer; Lennart Karstensen; Benjamin Jackson; Xingyu Chen; S. M. Hadi Sadati; Christos Bergeles; Alejandro Granados; Thomas C Booth
>
> **备注:** Published in IEEE Robotics and Automation Letters
>
> **摘要:** Mechanical thrombectomy (MT) is typically the optimal treatment for acute ischemic stroke involving large vessel occlusions, but access is limited due to geographic and logistical barriers. Reinforcement learning (RL) shows promise in autonomous endovascular navigation, but generalization across 'long' navigation tasks remains challenging. We propose a Hierarchical Modular Multi-Agent Reinforcement Learning (HM-MARL) framework for autonomous two-device navigation in vitro, enabling efficient and generalizable navigation. HM-MARL was developed to autonomously navigate a guide catheter and guidewire from the femoral artery to the internal carotid artery (ICA). A modular multi-agent approach was used to decompose the complex navigation task into specialized subtasks, each trained using Soft Actor-Critic RL. The framework was validated in both in silico and in vitro testbeds to assess generalization and real-world feasibility. In silico, a single-vasculature model achieved 92-100% success rates on individual anatomies, while a multi-vasculature model achieved 56-80% across multiple patient anatomies. In vitro, both HM-MARL models successfully navigated 100% of trials from the femoral artery to the right common carotid artery and 80% to the right ICA but failed on the left-side vessel superhuman challenge due to the anatomy and catheter type used in navigation. This study presents the first demonstration of in vitro autonomous navigation in MT vasculature. While HM-MARL enables generalization across anatomies, the simulation-to-real transition introduces challenges. Future work will refine RL strategies using world models and validate performance on unseen in vitro data, advancing autonomous MT towards clinical translation.
>
---
#### [new 002] To Move or Not to Move: Constraint-based Planning Enables Zero-Shot Generalization for Interactive Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究交互式导航任务，解决 clutter 阻塞路径的问题。提出基于约束的规划框架，结合 LLM 与主动感知，实现零样本泛化，完成物体放置任务。**

- **链接: [https://arxiv.org/pdf/2602.20055v1](https://arxiv.org/pdf/2602.20055v1)**

> **作者:** Apoorva Vashisth; Manav Kulshrestha; Pranav Bakshi; Damon Conover; Guillaume Sartoretti; Aniket Bera
>
> **摘要:** Visual navigation typically assumes the existence of at least one obstacle-free path between start and goal, which must be discovered/planned by the robot. However, in real-world scenarios, such as home environments and warehouses, clutter can block all routes. Targeted at such cases, we introduce the Lifelong Interactive Navigation problem, where a mobile robot with manipulation abilities can move clutter to forge its own path to complete sequential object- placement tasks - each involving placing an given object (eg. Alarm clock, Pillow) onto a target object (eg. Dining table, Desk, Bed). To address this lifelong setting - where effects of environment changes accumulate and have long-term effects - we propose an LLM-driven, constraint-based planning framework with active perception. Our framework allows the LLM to reason over a structured scene graph of discovered objects and obstacles, deciding which object to move, where to place it, and where to look next to discover task-relevant information. This coupling of reasoning and active perception allows the agent to explore the regions expected to contribute to task completion rather than exhaustively mapping the environment. A standard motion planner then executes the corresponding navigate-pick-place, or detour sequence, ensuring reliable low-level control. Evaluated in physics-enabled ProcTHOR-10k simulator, our approach outperforms non-learning and learning-based baselines. We further demonstrate our approach qualitatively on real-world hardware.
>
---
#### [new 003] EEG-Driven Intention Decoding: Offline Deep Learning Benchmarking on a Robotic Rover
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于脑机接口任务，旨在解决机器人导航中用户意图解码问题。通过EEG信号与机器人控制结合，使用深度学习模型进行离线命令预测。**

- **链接: [https://arxiv.org/pdf/2602.20041v1](https://arxiv.org/pdf/2602.20041v1)**

> **作者:** Ghadah Alosaimi; Maha Alsayyari; Yixin Sun; Stamos Katsigiannis; Amir Atapour-Abarghouei; Toby P. Breckon
>
> **摘要:** Brain-computer interfaces (BCIs) provide a hands-free control modality for mobile robotics, yet decoding user intent during real-world navigation remains challenging. This work presents a brain-robot control framework for offline decoding of driving commands during robotic rover operation. A 4WD Rover Pro platform was remotely operated by 12 participants who navigated a predefined route using a joystick, executing the commands forward, reverse, left, right, and stop. Electroencephalogram (EEG) signals were recorded with a 16-channel OpenBCI cap and aligned with motor actions at Delta = 0 ms and future prediction horizons (Delta > 0 ms). After preprocessing, several deep learning models were benchmarked, including convolutional neural networks, recurrent neural networks, and Transformer architectures. ShallowConvNet achieved the highest performance for both action prediction and intent prediction. By combining real-world robotic control with multi-horizon EEG intention decoding, this study introduces a reproducible benchmark and reveals key design insights for predictive deep learning-based BCI systems.
>
---
#### [new 004] WildOS: Open-Vocabulary Object Search in the Wild
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出WildOS系统，解决长距离、开放词汇目标搜索问题。结合几何探索与语义视觉推理，提升机器人在复杂环境中的导航效率与自主性。**

- **链接: [https://arxiv.org/pdf/2602.19308v1](https://arxiv.org/pdf/2602.19308v1)**

> **作者:** Hardik Shah; Erica Tevere; Deegan Atha; Marcel Kaufmann; Shehryar Khattak; Manthan Patel; Marco Hutter; Jonas Frey; Patrick Spieler
>
> **备注:** 28 pages, 16 figures, 2 tables
>
> **摘要:** Autonomous navigation in complex, unstructured outdoor environments requires robots to operate over long ranges without prior maps and limited depth sensing. In such settings, relying solely on geometric frontiers for exploration is often insufficient. In such settings, the ability to reason semantically about where to go and what is safe to traverse is crucial for robust, efficient exploration. This work presents WildOS, a unified system for long-range, open-vocabulary object search that combines safe geometric exploration with semantic visual reasoning. WildOS builds a sparse navigation graph to maintain spatial memory, while utilizing a foundation-model-based vision module, ExploRFM, to score frontier nodes of the graph. ExploRFM simultaneously predicts traversability, visual frontiers, and object similarity in image space, enabling real-time, onboard semantic navigation tasks. The resulting vision-scored graph enables the robot to explore semantically meaningful directions while ensuring geometric safety. Furthermore, we introduce a particle-filter-based method for coarse localization of the open-vocabulary target query, that estimates candidate goal positions beyond the robot's immediate depth horizon, enabling effective planning toward distant goals. Extensive closed-loop field experiments across diverse off-road and urban terrains demonstrate that WildOS enables robust navigation, significantly outperforming purely geometric and purely vision-based baselines in both efficiency and autonomy. Our results highlight the potential of vision foundation models to drive open-world robotic behaviors that are both semantically informed and geometrically grounded. Project Page: https://leggedrobotics.github.io/wildos/
>
---
#### [new 005] CACTO-BIC: Scalable Actor-Critic Learning via Biased Sampling and GPU-Accelerated Trajectory Optimization
- **分类: cs.RO; math.OC**

- **简介: 该论文提出CACTO-BIC，解决强化学习与轨迹优化结合中的可扩展性问题，通过偏差采样和GPU加速提升效率，适用于高维实时控制任务。**

- **链接: [https://arxiv.org/pdf/2602.19699v1](https://arxiv.org/pdf/2602.19699v1)**

> **作者:** Elisa Alboni; Pietro Noah Crestaz; Elias Fontanari; Andrea Del Prete
>
> **摘要:** Trajectory Optimization (TO) and Reinforcement Learning (RL) offer complementary strengths for solving optimal control problems. TO efficiently computes locally optimal solutions but can struggle with non-convexity, while RL is more robust to non-convexity at the cost of significantly higher computational demands. CACTO (Continuous Actor-Critic with Trajectory Optimization) was introduced to combine these advantages by learning a warm-start policy that guides the TO solver towards low-cost trajectories. However, scalability remains a key limitation, as increasing system complexity significantly raises the computational cost of TO. This work introduces CACTO-BIC to address these challenges. CACTO-BIC improves data efficiency by biasing initial-state sampling leveraging a property of the value function associated with locally optimal policies; moreover, it reduces computation time by exploiting GPU acceleration. Empirical evaluations show improved sample efficiency and faster computation compared to CACTO. Comparisons with PPO demonstrate that our approach can achieve similar solutions in less time. Finally, experiments on the AlienGO quadruped robot demonstrate that CACTO-BIC can scale to high-dimensional systems and is suitable for real-time applications.
>
---
#### [new 006] Understanding Fire Through Thermal Radiation Fields for Mobile Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决火灾环境下机器人安全移动的问题。通过构建实时热辐射场，实现热安全路径规划。**

- **链接: [https://arxiv.org/pdf/2602.19108v1](https://arxiv.org/pdf/2602.19108v1)**

> **作者:** Anton R. Wagner; Madhan Balaji Rao; Xuesu Xiao; Sören Pirk
>
> **摘要:** Safely moving through environments affected by fire is a critical capability for autonomous mobile robots deployed in disaster response. In this work, we present a novel approach for mobile robots to understand fire through building real-time thermal radiation fields. We register depth and thermal images to obtain a 3D point cloud annotated with temperature values. From these data, we identify fires and use the Stefan-Boltzmann law to approximate the thermal radiation in empty spaces. This enables the construction of a continuous thermal radiation field over the environment. We show that this representation can be used for robot navigation, where we embed thermal constraints into the cost map to compute collision-free and thermally safe paths. We validate our approach on a Boston Dynamics Spot robot in controlled experimental settings. Our experiments demonstrate the robot's ability to avoid hazardous regions while still reaching navigation goals. Our approach paves the way toward mobile robots that can be autonomously deployed in fire-affected environments, with potential applications in search-and-rescue, firefighting, and hazardous material response.
>
---
#### [new 007] Scaling Law of Neural Koopman Operators
- **分类: cs.RO**

- **简介: 该论文研究数据驱动的神经Koopman算子，解决模型性能与数据量、维度间的平衡问题。通过理论分析和实验验证，建立 scaling law 并引入正则化方法提升控制效果。**

- **链接: [https://arxiv.org/pdf/2602.19943v1](https://arxiv.org/pdf/2602.19943v1)**

> **作者:** Abulikemu Abuduweili; Yuyang Pang; Feihan Li; Changliu Liu
>
> **摘要:** Data-driven neural Koopman operator theory has emerged as a powerful tool for linearizing and controlling nonlinear robotic systems. However, the performance of these data-driven models fundamentally depends on the trade-off between sample size and model dimensions, a relationship for which the scaling laws have remained unclear. This paper establishes a rigorous framework to address this challenge by deriving and empirically validating scaling laws that connect sample size, latent space dimension, and downstream control quality. We derive a theoretical upper bound on the Koopman approximation error, explicitly decomposing it into sampling error and projection error. We show that these terms decay at specific rates relative to dataset size and latent dimension, providing a rigorous basis for the scaling law. Based on the theoretical results, we introduce two lightweight regularizers for the neural Koopman operator: a covariance loss to help stabilize the learned latent features and an inverse control loss to ensure the model aligns with physical actuation. The results from systematic experiments across six robotic environments confirm that model fitting error follows the derived scaling laws, and the regularizers improve dynamic model fitting fidelity, with enhanced closed-loop control performance. Together, our results provide a simple recipe for allocating effort between data collection and model capacity when learning Koopman dynamics for control.
>
---
#### [new 008] A Checklist for Deploying Robots in Public: Articulating Tacit Knowledge in the HRI Community
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互领域，旨在解决公共场景中机器人部署的挑战。通过构建检查清单，整合社区经验，帮助研究人员避免常见错误。**

- **链接: [https://arxiv.org/pdf/2602.19038v1](https://arxiv.org/pdf/2602.19038v1)**

> **作者:** Claire Liang; Franziska Babel; Hannah Pelikan; Sydney Thompson; Xiang Zhi Tan
>
> **摘要:** Many of the challenges encountered in in-the-wild public deployments of robots remain undocumented despite sharing many common pitfalls. This creates a high barrier of entry and results in repetition of avoidable mistakes. To articulate the tacit knowledge in the HRI community, this paper presents a guideline in the form of a checklist to support researchers in preparing for robot deployments in public. Drawing on their own experience with public robot deployments, the research team collected essential topics to consider in public HRI research. These topics are represented as modular flip cards in a hierarchical table, structured into deployment phases and important domains. We interviewed six interdisciplinary researchers with expertise in public HRI and show how including community input refines the checklist. We further show the checklist in action in context of real public studies. Finally, we contribute the checklist as an open-source, customizable community resource that both collects joint expertise for continual evolution and is usable as a list, set of cards, and an interactive web tool.
>
---
#### [new 009] Visual Prompt Guided Unified Pushing Policy
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决推送策略效率与通用性不足的问题。提出一种结合视觉提示的统一推送策略，提升多场景适应能力。**

- **链接: [https://arxiv.org/pdf/2602.19193v1](https://arxiv.org/pdf/2602.19193v1)**

> **作者:** Hieu Bui; Ziyan Gao; Yuya Hosoda; Joo-Ho Lee
>
> **摘要:** As one of the simplest non-prehensile manipulation skills, pushing has been widely studied as an effective means to rearrange objects. Existing approaches, however, typically rely on multi-step push plans composed of pre-defined pushing primitives with limited application scopes, which restrict their efficiency and versatility across different scenarios. In this work, we propose a unified pushing policy that incorporates a lightweight prompting mechanism into a flow matching policy to guide the generation of reactive, multimodal pushing actions. The visual prompt can be specified by a high-level planner, enabling the reuse of the pushing policy across a wide range of planning problems. Experimental results demonstrate that the proposed unified pushing policy not only outperforms existing baselines but also effectively serves as a low-level primitive within a VLM-guided planning framework to solve table-cleaning tasks efficiently.
>
---
#### [new 010] Gait Asymmetry from Unilateral Weakness and Improvement With Ankle Assistance: a Reinforcement Learning based Simulation Study
- **分类: cs.RO**

- **简介: 该论文属于康复工程任务，旨在研究单侧肌力减弱对步态对称性的影响，并评估踝关节外骨骼辅助的效果。通过强化学习仿真框架分析步态不对称性及辅助改善效果。**

- **链接: [https://arxiv.org/pdf/2602.18862v1](https://arxiv.org/pdf/2602.18862v1)**

> **作者:** Yifei Yuan; Ghaith Androwis; Xianlian Zhou
>
> **摘要:** Unilateral muscle weakness often leads to asymmetric gait, disrupting interlimb coordination and stance timing. This study presents a reinforcement learning (RL) based musculoskeletal simulation framework to (1) quantify how progressive unilateral muscle weakness affects gait symmetry and (2) evaluate whether ankle exoskeleton assistance can improve gait symmetry under impaired conditions. The overarching goal is to establish a simulation- and learning-based workflow that supports early controller development prior to patient experiments. Asymmetric gait was induced by reducing right-leg muscle strength to 75%, 50%, and 25% of baseline. Gait asymmetry was quantified using toe-off timing, peak contact forces, and joint-level symmetry metrics. Increasing weakness produced progressively larger temporal and kinematic asymmetry, most pronounced at the ankle. Ankle range of motion symmetry degraded from near-symmetric behavior at 100% strength (symmetry index, SI = +6.4%; correlation r=0.974) to severe asymmetry at 25% strength (SI = -47.1%, r=0.889), accompanied by a load shift toward the unimpaired limb. At 50% strength, ankle exoskeleton assistance improved kinematic symmetry relative to the unassisted impaired condition, reducing the magnitude of ankle SI from 25.8% to 18.5% and increasing ankle correlation from r=0.948 to 0.966, although peak loading remained biased toward the unimpaired side. Overall, this framework supports controlled evaluation of impairment severity and assistive strategies, and provides a basis for future validation in human experiments.
>
---
#### [new 011] Path planning for unmanned surface vehicle based on predictive artificial potential field. International Journal of Advanced Robotic Systems
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，旨在解决高速无人船的路径优化问题。通过改进人工势场方法，提升路径平滑性与效率，减少航行时间和能耗。**

- **链接: [https://arxiv.org/pdf/2602.19062v1](https://arxiv.org/pdf/2602.19062v1)**

> **作者:** Jia Song; Ce Hao; Jiangcheng Su
>
> **摘要:** Path planning for high-speed unmanned surface vehicles requires more complex solutions to reduce sailing time and save energy. This article proposes a new predictive artificial potential field that incorporates time information and predictive potential to plan smoother paths. It explores the principles of the artificial potential field, considering vehicle dynamics and local minimum reachability. The study first analyzes the most advanced traditional artificial potential field and its drawbacks in global and local path planning. It then introduces three modifications to the predictive artificial potential field-angle limit, velocity adjustment, and predictive potential to enhance the feasibility and flatness of the generated path. A comparison between the traditional and predictive artificial potential fields demonstrates that the latter successfully restricts the maximum turning angle, shortens sailing time, and intelligently avoids obstacles. Simulation results further verify that the predictive artificial potential field addresses the concave local minimum problem and improves reachability in special scenarios, ultimately generating a more efficient path that reduces sailing time and conserves energy for unmanned surface vehicles.
>
---
#### [new 012] CLASH: Collision Learning via Augmented Sim-to-real Hybridization to Bridge the Reality Gap
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决模拟到现实的差距问题。通过CLASH框架，利用少量真实数据提升模拟器精度，增强策略迁移效果。**

- **链接: [https://arxiv.org/pdf/2602.18707v1](https://arxiv.org/pdf/2602.18707v1)**

> **作者:** Haotian He; Ning Guo; Siqi Shi; Qipeng Liu; Wenzhao Lian
>
> **摘要:** The sim-to-real gap, particularly in the inaccurate modeling of contact-rich dynamics like collisions, remains a primary obstacle to deploying robot policies trained in simulation. Conventional physics engines often trade accuracy for computational speed, leading to discrepancies that prevent direct policy transfer. To address this, we introduce Collision Learning via Augmented Sim-to-real Hybridization (CLASH), a data-efficient framework that creates a high-fidelity hybrid simulator by learning a surrogate collision model from a minimal set of real-world data. In CLASH, a base model is first distilled from an imperfect simulator (MuJoCo) to capture general physical priors; this model is then fine-tuned with a remarkably small number of real-world interactions (as few as 10 samples) to correct for the simulator's inherent inaccuracies. The resulting hybrid simulator not only achieves higher predictive accuracy but also reduces collision computation time by nearly 50\%. We demonstrate that policies obtained with our hybrid simulator transfer more robustly to the real world, doubling the success rate in sequential pushing tasks with reinforecement learning and significantly increase the task performance with model-based control.
>
---
#### [new 013] The Price Is Not Right: Neuro-Symbolic Methods Outperform VLAs on Structured Long-Horizon Manipulation Tasks with Significantly Lower Energy Consumption
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决长周期结构化操作中VLA模型效率低的问题。通过对比神经符号方法与VLA模型，验证了前者在成功率和能耗上的优势。**

- **链接: [https://arxiv.org/pdf/2602.19260v1](https://arxiv.org/pdf/2602.19260v1)**

> **作者:** Timothy Duggan; Pierrick Lorang; Hong Lu; Matthias Scheutz
>
> **备注:** Accepted at the 2026 IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** Vision-Language-Action (VLA) models have recently been proposed as a pathway toward generalist robotic policies capable of interpreting natural language and visual inputs to generate manipulation actions. However, their effectiveness and efficiency on structured, long-horizon manipulation tasks remain unclear. In this work, we present a head-to-head empirical comparison between a fine-tuned open-weight VLA model π0 and a neuro-symbolic architecture that combines PDDL-based symbolic planning with learned low-level control. We evaluate both approaches on structured variants of the Towers of Hanoi manipulation task in simulation while measuring both task performance and energy consumption during training and execution. On the 3-block task, the neuro-symbolic model achieves 95% success compared to 34% for the best-performing VLA. The neuro-symbolic model also generalizes to an unseen 4-block variant (78% success), whereas both VLAs fail to complete the task. During training, VLA fine-tuning consumes nearly two orders of magnitude more energy than the neuro-symbolic approach. These results highlight important trade-offs between end-to-end foundation-model approaches and structured reasoning architectures for long-horizon robotic manipulation, emphasizing the role of explicit symbolic structure in improving reliability, data efficiency, and energy efficiency. Code and models are available at https://price-is-not-right.github.io
>
---
#### [new 014] OVerSeeC: Open-Vocabulary Costmap Generation from Satellite Images and Natural Language
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出OVerSeeC框架，解决从卫星图像和自然语言生成全局成本地图的问题，实现灵活、适应任务的路径规划。**

- **链接: [https://arxiv.org/pdf/2602.18606v1](https://arxiv.org/pdf/2602.18606v1)**

> **作者:** Rwik Rana; Jesse Quattrociocchi; Dongmyeong Lee; Christian Ellis; Amanda Adkins; Adam Uccello; Garrett Warnell; Joydeep Biswas
>
> **备注:** Website : https://amrl.cs.utexas.edu/overseec/
>
> **摘要:** Aerial imagery provides essential global context for autonomous navigation, enabling route planning at scales inaccessible to onboard sensing. We address the problem of generating global costmaps for long-range planning directly from satellite imagery when entities and mission-specific traversal rules are expressed in natural language at test time. This setting is challenging since mission requirements vary, terrain entities may be unknown at deployment, and user prompts often encode compositional traversal logic. Existing approaches relying on fixed ontologies and static cost mappings cannot accommodate such flexibility. While foundation models excel at language interpretation and open-vocabulary perception, no single model can simultaneously parse nuanced mission directives, locate arbitrary entities in large-scale imagery, and synthesize them into an executable cost function for planners. We therefore propose OVerSeeC, a zero-shot modular framework that decomposes the problem into Interpret-Locate-Synthesize: (i) an LLM extracts entities and ranked preferences, (ii) an open-vocabulary segmentation pipeline identifies these entities from high-resolution imagery, and (iii) the LLM uses the user's natural language preferences and masks to synthesize executable costmap code. Empirically, OVerSeeC handles novel entities, respects ranked and compositional preferences, and produces routes consistent with human-drawn trajectories across diverse regions, demonstrating robustness to distribution shifts. This shows that modular composition of foundation models enables open-vocabulary, preference-aligned costmap generation for scalable, mission-adaptive global planning.
>
---
#### [new 015] Enhancing Goal Inference via Correction Timing
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人学习任务，旨在通过纠正时机提升目标推断。解决如何利用纠正行为的时间信息来优化任务理解与约束学习的问题。工作包括分析纠正时机对任务特征、目标推断和约束学习的影响。**

- **链接: [https://arxiv.org/pdf/2602.18603v1](https://arxiv.org/pdf/2602.18603v1)**

> **作者:** Anjiabei Wang; Shuangge Wang; Tesca Fitzgerald
>
> **备注:** 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)
>
> **摘要:** Corrections offer a natural modality for people to provide feedback to a robot, by (i) intervening in the robot's behavior when they believe the robot is failing (or will fail) the task objectives and (ii) modifying the robot's behavior to successfully fulfill the task. Each correction offers information on what the robot should and should not do, where the corrected behavior is more aligned with task objectives than the original behavior. Most prior work on learning from corrections involves interpreting a correction as a new demonstration (consisting of the modified robot behavior), or a preference (for the modified trajectory compared to the robot's original behavior). However, this overlooks one essential element of the correction feedback, which is the human's decision to intervene in the robot's behavior in the first place. This decision can be influenced by multiple factors including the robot's task progress, alignment with human expectations, dynamics, motion legibility, and optimality. In this work, we investigate whether the timing of this decision can offer a useful signal for inferring these task-relevant influences. In particular, we investigate three potential applications for this learning signal: (1) identifying features of a robot's motion that may prompt people to correct it, (2) quickly inferring the final goal of a human's correction based on the timing and initial direction of their correction motion, and (3) learning more precise constraints for task objectives. Our results indicate that correction timing results in improved learning for the first two of these applications. Overall, our work provides new insights on the value of correction timing as a signal for robot learning.
>
---
#### [new 016] GRAB: A Systematic Real-World Grasping Benchmark for Robotic Food Waste Sorting
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决食品废弃物分类中的抓取难题。通过构建GRAB基准，评估不同抓取方式在真实场景下的表现，分析影响抓取性能的关键因素。**

- **链接: [https://arxiv.org/pdf/2602.18835v1](https://arxiv.org/pdf/2602.18835v1)**

> **作者:** Moniesha Thilakarathna; Xing Wang; Min Wang; David Hinwood; Shuangzhe Liu; Damith Herath
>
> **备注:** 23 pages, 12 Figures, 3 Tables, submitted to Advanced Intelligent Systems Journal and under review
>
> **摘要:** Food waste management is critical for sustainability, yet inorganic contaminants hinder recycling potential. Robotic automation presents a compelling approach to this challenge by accelerating the sorting process through automated contaminant removal. Still, the diverse and unpredictable nature of contaminants creates major challenges for robotic grasping. Benchmarking frameworks are critical for evaluating challenges from various perspectives. However, existing protocols rely on limited simulation datasets, prioritise simple metrics such as success rate, and overlook key object and environment-related pre-grasp conditions. This paper introduces GRAB, a comprehensive Grasping Real-World Article Benchmarking framework that addresses this gap by integrating diverse deformable objects, advanced grasp-pose-estimation vision, and, importantly, pre-grasp conditions, establishing a set of critical graspability metrics. It systematically compares industrial grasping modalities through an in-depth experimental evaluation involving 1,750 food contaminant grasp attempts across four high-fidelity scenes. This large-scale evaluation provides an extensive assessment of grasp performance for food waste sorting, offering a level of depth that has rarely been explored in previous studies. The results reveal distinct gripper strengths and limitations, with object quality emerging as the dominant performance factor in cluttered environments, while vision quality and clutter levels play moderate roles. These findings highlight essential design considerations and reinforce the necessity of developing multimodal gripper technologies capable of robust cross-category performance for effective robotic food waste sorting.
>
---
#### [new 017] FORMICA: Decision-Focused Learning for Communication-Free Multi-Robot Task Allocation
- **分类: cs.RO**

- **简介: 该论文提出FORMICA方法，解决多机器人无通信任务分配问题。通过预测队友出价实现隐式协调，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2602.18622v1](https://arxiv.org/pdf/2602.18622v1)**

> **作者:** Antonio Lopez; Jack Muirhead; Carlo Pinciroli
>
> **备注:** 13 pages, 2 figures, ANTS 2026
>
> **摘要:** Most multi-robot task allocation methods rely on communication to resolve conflicts and reach consistent assignments. In environments with limited bandwidth, degraded infrastructure, or adversarial interference, existing approaches degrade sharply. We introduce a learning-based framework that achieves high-quality task allocation without any robot-to-robot communication. The key idea is that robots coordinate implicitly by predicting teammates' bids: if each robot can anticipate competition for a task, it can adjust its choices accordingly. Our method predicts bid distributions to correct systematic errors in analytical mean-field approximations. While analytical predictions assume idealized conditions (uniform distributions, known bid functions), our learned approach adapts to task clustering and spatial heterogeneity. Inspired by Smart Predict-then-Optimize (SPO), we train predictors end-to-end to minimize Task Allocation Regret rather than prediction error. To scale to large swarms, we develop a mean-field approximation where each robot predicts the distribution of competing bids rather than individual bids, reducing complexity from $O(NT)$ to $O(T)$. We call our approach FORMICA: Field-Oriented Regret-Minimizing Implicit Coordination Algorithm. Experiments show FORMICA substantially outperforms a natural analytical baseline. In scenarios with 16 robots and 64 tasks, our approach improves system reward by 17% and approaches the optimal MILP solution. When deployed on larger scenarios (256 robots, 4096 tasks), the same model improves performance by 7%, demonstrating strong generalization. Training requires only 21 seconds on a laptop, enabling rapid adaptation to new environments.
>
---
#### [new 018] Human-to-Robot Interaction: Learning from Video Demonstration for Robot Imitation
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，旨在解决从视频中提取精确操作指令的问题。通过分阶段框架实现视频理解与机器人执行，提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.19184v1](https://arxiv.org/pdf/2602.19184v1)**

> **作者:** Thanh Nguyen Canh; Thanh-Tuan Tran; Haolan Zhang; Ziyan Gao; Nak Young Chong; Xiem HoangVan
>
> **摘要:** Learning from Demonstration (LfD) offers a promising paradigm for robot skill acquisition. Recent approaches attempt to extract manipulation commands directly from video demonstrations, yet face two critical challenges: (1) general video captioning models prioritize global scene features over task-relevant objects, producing descriptions unsuitable for precise robotic execution, and (2) end-to-end architectures coupling visual understanding with policy learning require extensive paired datasets and struggle to generalize across objects and scenarios. To address these limitations, we propose a novel ``Human-to-Robot'' imitation learning pipeline that enables robots to acquire manipulation skills directly from unstructured video demonstrations, inspired by the human ability to learn by watching and imitating. Our key innovation is a modular framework that decouples the learning process into two distinct stages: (1) Video Understanding, which combines Temporal Shift Modules (TSM) with Vision-Language Models (VLMs) to extract actions and identify interacted objects, and (2) Robot Imitation, which employs TD3-based deep reinforcement learning to execute the demonstrated manipulations. We validated our approach in PyBullet simulation environments with a UR5e manipulator and in a real-world experiment with a UF850 manipulator across four fundamental actions: reach, pick, move, and put. For video understanding, our method achieves 89.97% action classification accuracy and BLEU-4 scores of 0.351 on standard objects and 0.265 on novel objects, representing improvements of 76.4% and 128.4% over the best baseline, respectively. For robot manipulation, our framework achieves an average success rate of 87.5% across all actions, with 100% success on reaching tasks and up to 90% on complex pick-and-place operations. The project website is available at https://thanhnguyencanh.github.io/LfD4hri.
>
---
#### [new 019] Botson: An Accessible and Low-Cost Platform for Social Robotics Research
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于社会机器人研究任务，旨在解决AI在人际领域信任不足的问题。通过构建低成本的社交机器人Botson，增强非语言社交线索的表达。**

- **链接: [https://arxiv.org/pdf/2602.19491v1](https://arxiv.org/pdf/2602.19491v1)**

> **作者:** Samuel Bellaire; Abdalmalek Abu-raddaha; Natalie Kim; Nathan Morhan; William Elliott; Samir Rawashdeh
>
> **备注:** 5 pages, 7 figures
>
> **摘要:** Trust remains a critical barrier to the effective integration of Artificial Intelligence (AI) into human-centric domains. Disembodied agents, such as voice assistants, often fail to establish trust due to their inability to convey non-verbal social cues. This paper introduces the architecture of Botson: an anthropomorphic social robot powered by a large language model (LLM). Botson was created as a low-cost and accessible platform for social robotics research.
>
---
#### [new 020] Design and Control of Modular Magnetic Millirobots for Multimodal Locomotion and Shape Reconfiguration
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人学领域，旨在解决模块化磁驱动机器人在生物医学中的应用问题。提出一种可变形磁微机器人系统，实现多模式运动与可靠重组。**

- **链接: [https://arxiv.org/pdf/2602.19346v1](https://arxiv.org/pdf/2602.19346v1)**

> **作者:** Erik Garcia Oyono; Jialin Lin; Dandan Zhang
>
> **备注:** Accepted by 2026 ICRA
>
> **摘要:** Modular small-scale robots offer the potential for on-demand assembly and disassembly, enabling task-specific adaptation in dynamic and constrained environments. However, existing modular magnetic platforms often depend on workspace collisions for reconfiguration, employ bulky three-dimensional electromagnetic systems, and lack robust single-module control, which limits their applicability in biomedical settings. In this work, we present a modular magnetic millirobotic platform comprising three cube-shaped modules with embedded permanent magnets, each designed for a distinct functional role: a free module that supports self-assembly and reconfiguration, a fixed module that enables flip-and-walk locomotion, and a gripper module for cargo manipulation. Locomotion and reconfiguration are actuated by programmable combinations of time-varying two-dimensional uniform and gradient magnetic field inputs. Experiments demonstrate closed-loop navigation using real-time vision feedback and A* path planning, establishing robust single-module control capabilities. Beyond locomotion, the system achieves self-assembly, multimodal transformations, and disassembly at low field strengths. Chain-to-gripper transformations succeeded in 90% of trials, while chain-to-square transformations were less consistent, underscoring the role of module geometry in reconfiguration reliability. These results establish a versatile modular robotic platform capable of multimodal behavior and robust control, suggesting a promising pathway toward scalable and adaptive task execution in confined environments.
>
---
#### [new 021] Soft Surfaced Vision-Based Tactile Sensing for Bipedal Robot Applications
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在提升双足机器人的平衡与环境感知。通过视觉触觉传感器捕捉足部接触信息，实现地形识别与姿态估计，增强机器人稳定性与适应性。**

- **链接: [https://arxiv.org/pdf/2602.18638v1](https://arxiv.org/pdf/2602.18638v1)**

> **作者:** Jaeeun Kim; Junhee Lim; Yu She
>
> **备注:** 8 pages, 11 figures, RoboSoft 2026. For the supplementary video, please visit: https://youtu.be/ceJiy9q_2Aw
>
> **摘要:** Legged locomotion benefits from embodied sensing, where perception emerges from the physical interaction between body and environment. We present a soft-surfaced, vision-based tactile foot sensor that endows a bipedal robot with a skin-like deformable layer that captures contact deformations optically, turning foot-ground interactions into rich haptic signals. From a contact image stream, our method estimates contact pose (position and orientation), visualizes shear, computes center of pressure (CoP), classifies terrain, and detects geometric features of the contact patch. We validate these capabilities on a tilting platform and in visually obscured conditions, showing that foot-borne tactile feedback improves balance control and terrain awareness beyond proprioception alone. These findings suggest that integrating tactile perception into legged robot feet improves stability, adaptability, and environmental awareness, offering a promising direction toward more compliant and intelligent locomotion systems. For the supplementary video, please visit: https://youtu.be/ceJiy9q_2Aw
>
---
#### [new 022] Habilis-$β$: A Fast-Motion and Long-Lasting On-Device Vision-Language-Action Model
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出Habilis-β，一种用于实际部署的视觉-语言-动作模型，解决快速运动与长时间运行问题。通过新评估指标PRP验证其高效性与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.18813v1](https://arxiv.org/pdf/2602.18813v1)**

> **作者:** Tommoro Robotics; :; Jesoon Kang; Taegeon Park; Jisu An; Soo Min Kimm; Jaejoon Kim; Jinu Pahk; Byungju Kim; Junseok Lee; Namheon Baek; Sungwan Ha; Hojun Baek; Eduardo Ayerve Cruz; Wontae Kim; Junghyeon Choi; Yousuk Lee; Joonmo Han; Sunghyun Cho; Sunghyun Kwon; Soyoung Lee; Jun Ki Lee; Seung-Joon Yi; Byoung-Tak Zhang; Theo Taeyeong Kim
>
> **摘要:** We introduce Habilis-$β$, a fast-motion and long-lasting on-device vision-language-action (VLA) model designed for real-world deployment. Current VLA evaluation remains largely confined to single-trial success rates under curated resets, which fails to capture the fast-motion and long-lasting capabilities essential for practical operation. To address this, we introduce the Productivity-Reliability Plane (PRP), which evaluates performance through Tasks per Hour (TPH) and Mean Time Between Intervention (MTBI) under a continuous-run protocol that demands both high-speed execution and sustained robustness. Habilis-$β$ achieves high performance by integrating language-free pre-training on large-scale play data for robust interaction priors with post-training on cyclic task demonstrations that capture state drift across consecutive task iterations. The system further employs ESPADA for phase-adaptive motion shaping to accelerate free-space transit, utilizes rectified-flow distillation to enable high-frequency control on edge devices, and incorporates classifier-free guidance (CFG) as a deployment-time knob to dynamically balance instruction adherence and learned interaction priors. In 1-hour continuous-run evaluations, Habilis-$β$ achieves strong performance under the PRP metrics, compared to $π_{0.5}$ in both simulation and real-world environments. In simulation, Habilis-$β$ achieves 572.6 TPH and 39.2 s MTBI (vs. 120.5 TPH and 30.5 s for $π_{0.5}$), while in a real-world humanoid logistics workflow it achieves 124 TPH and 137.4 s MTBI (vs. 19 TPH and 46.1 s for $π_{0.5}$). Finally, Habilis-$β$ achieves the highest reported performance on the standard RoboTwin 2.0 leaderboard across representative tasks, validating its effectiveness in complex manipulation scenarios.
>
---
#### [new 023] FruitTouch: A Perceptive Gripper for Gentle and Scalable Fruit Harvesting
- **分类: cs.RO**

- **简介: 该论文属于农业自动化任务，旨在解决水果采摘中抓取不稳定和损伤问题。提出FruitTouch gripper，集成视觉触觉传感，实现稳定、轻柔的采摘。**

- **链接: [https://arxiv.org/pdf/2602.18991v1](https://arxiv.org/pdf/2602.18991v1)**

> **作者:** Ruohan Zhang; Mohammad Amin Mirzaee; Wenzhen Yuan
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** The automation of fruit harvesting has gained increasing significance in response to rising labor shortages. A sensorized gripper is a key component of this process, which must be compact enough for confined spaces, able to stably grasp diverse fruits, and provide reliable feedback on fruit conditions for efficient harvesting. To address this need, we propose FruitTouch, a compact gripper that integrates high-resolution, vision-based tactile sensing through an optimized optical design. This configuration accommodates a wide range of fruit sizes while maintaining low cost and mechanical simplicity. Tactile images captured by an embedded camera provide rich information for real-time force estimation, slip detection, and softness prediction. We validate the gripper in real-world fruit harvesting experiments, demonstrating robust grasp stability and effective damage prevention.
>
---
#### [new 024] A User-driven Design Framework for Robotaxi
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决用户在无司机机器人出租车中的体验问题。通过访谈和实际体验，分析用户需求与挑战，提出用户驱动的设计框架。**

- **链接: [https://arxiv.org/pdf/2602.19107v1](https://arxiv.org/pdf/2602.19107v1)**

> **作者:** Yue Deng; Changyang He
>
> **摘要:** Robotaxis are emerging as a promising form of urban mobility, yet research has largely emphasized technical driving performance while leaving open how passengers experience and evaluate rides without a human driver. To address the limitations of prior work that often relies on simulated or hypothetical settings, we investigate real-world robotaxi use through 18 semi-structured interviews and autoethnographic ride experiences. We found that users were drawn to robotaxis by low cost, social recommendation, and curiosity. They valued a distinctive set of benefits, such as an increased sense of agency, and consistent driving behavioral consistency and standardized ride experiences. However, they encountered persistent challenges around limited flexibility, insufficient transparency, management difficulty, robustness concerns in edge cases, and emergency handling concerns. Robotaxi experiences were shaped by privacy, safety, ethics, and trust. Users were often privacy-indifferent yet sensitive to opaque access and leakage risks; safety perceptions were polarized; and ethical considerations surfaced round issues such as accountability, feedback responsibility and absence of human-like social norms. Based on these findings, we propose a user-driven design framework spanning the end-to-end journey, such as pre-ride configuration (hailing), context-aware pickup facilitation (pick-up) in-ride explainability (traveling), and accountable post-ride feedback (drop-off) to guide robotaxi interaction and service design.
>
---
#### [new 025] Large Language Model-Assisted UAV Operations and Communications: A Multifaceted Survey and Tutorial
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机与大语言模型融合的研究，旨在提升无人机的智能水平。通过整合LLM，解决无人机在环境理解、任务推理和协同控制等方面的问题，提出统一框架并探讨应用与伦理挑战。**

- **链接: [https://arxiv.org/pdf/2602.19534v1](https://arxiv.org/pdf/2602.19534v1)**

> **作者:** Yousef Emami; Hao Zhou; Radha Reddy; Atefeh Hajijamali Arani; Biliang Wang; Kai Li; Luis Almeida; Zhu Han
>
> **备注:** 40 pages, 10 figures, 13 tables
>
> **摘要:** Uncrewed Aerial Vehicles (UAVs) are widely deployed across diverse applications due to their mobility and agility. Recent advances in Large Language Models (LLMs) offer a transformative opportunity to enhance UAV intelligence beyond conventional optimization-based and learning-based approaches. By integrating LLMs into UAV systems, advanced environmental understanding, swarm coordination, mobility optimization, and high-level task reasoning can be achieved, thereby allowing more adaptive and context-aware aerial operations. This survey systematically explores the intersection of LLMs and UAV technologies and proposes a unified framework that consolidates existing architectures, methodologies, and applications for UAVs. We first present a structured taxonomy of LLM adaptation techniques for UAVs, including pretraining, fine-tuning, Retrieval-Augmented Generation (RAG), and prompt engineering, along with key reasoning capabilities such as Chain-of-Thought (CoT) and In-Context Learning (ICL). We then examine LLM-assisted UAV communications and operations, covering navigation, mission planning, swarm control, safety, autonomy, and network management. After that, the survey further discusses Multimodal LLMs (MLLMs) for human-swarm interaction, perception-driven navigation, and collaborative control. Finally, we address ethical considerations, including bias, transparency, accountability, and Human-in-the-Loop (HITL) strategies, and outline future research directions. Overall, this work positions LLM-assisted UAVs as a foundation for intelligent and adaptive aerial systems.
>
---
#### [new 026] Seeing Farther and Smarter: Value-Guided Multi-Path Reflection for VLM Policy Optimization
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人操作任务，解决长期规划与决策问题。提出一种新框架，通过多路径反射提升VLM策略优化，提高成功率并降低推理时间。**

- **链接: [https://arxiv.org/pdf/2602.19372v1](https://arxiv.org/pdf/2602.19372v1)**

> **作者:** Yanting Yang; Shenyuan Gao; Qingwen Bu; Li Chen; Dimitris N. Metaxas
>
> **备注:** ICRA 2026
>
> **摘要:** Solving complex, long-horizon robotic manipulation tasks requires a deep understanding of physical interactions, reasoning about their long-term consequences, and precise high-level planning. Vision-Language Models (VLMs) offer a general perceive-reason-act framework for this goal. However, previous approaches using reflective planning to guide VLMs in correcting actions encounter significant limitations. These methods rely on inefficient and often inaccurate implicit learning of state-values from noisy foresight predictions, evaluate only a single greedy future, and suffer from substantial inference latency. To address these limitations, we propose a novel test-time computation framework that decouples state evaluation from action generation. This provides a more direct and fine-grained supervisory signal for robust decision-making. Our method explicitly models the advantage of an action plan, quantified by its reduction in distance to the goal, and uses a scalable critic to estimate. To address the stochastic nature of single-trajectory evaluation, we employ beam search to explore multiple future paths and aggregate them during decoding to model their expected long-term returns, leading to more robust action generation. Additionally, we introduce a lightweight, confidence-based trigger that allows for early exit when direct predictions are reliable, invoking reflection only when necessary. Extensive experiments on diverse, unseen multi-stage robotic manipulation tasks demonstrate a 24.6% improvement in success rate over state-of-the-art baselines, while significantly reducing inference time by 56.5%.
>
---
#### [new 027] Temporal Action Representation Learning for Tactical Resource Control and Subsequent Maneuver Generation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于资源控制与策略生成任务，解决资源使用与后续动作间因果关系不明确的问题。提出TART框架，通过时间表征学习提升资源利用效率和动作连贯性。**

- **链接: [https://arxiv.org/pdf/2602.18716v1](https://arxiv.org/pdf/2602.18716v1)**

> **作者:** Hoseong Jung; Sungil Son; Daesol Cho; Jonghae Park; Changhyun Choi; H. Jin Kim
>
> **备注:** ICRA 2026, 8 pages
>
> **摘要:** Autonomous robotic systems should reason about resource control and its impact on subsequent maneuvers, especially when operating with limited energy budgets or restricted sensing. Learning-based control is effective in handling complex dynamics and represents the problem as a hybrid action space unifying discrete resource usage and continuous maneuvers. However, prior works on hybrid action space have not sufficiently captured the causal dependencies between resource usage and maneuvers. They have also overlooked the multi-modal nature of tactical decisions, both of which are critical in fast-evolving scenarios. In this paper, we propose TART, a Temporal Action Representation learning framework for Tactical resource control and subsequent maneuver generation. TART leverages contrastive learning based on a mutual information objective, designed to capture inherent temporal dependencies in resource-maneuver interactions. These learned representations are quantized into discrete codebook entries that condition the policy, capturing recurring tactical patterns and enabling multi-modal and temporally coherent behaviors. We evaluate TART in two domains where resource deployment is critical: (i) a maze navigation task where a limited budget of discrete actions provides enhanced mobility, and (ii) a high-fidelity air combat simulator in which an F-16 agent operates weapons and defensive systems in coordination with flight maneuvers. Across both domains, TART consistently outperforms hybrid-action baselines, demonstrating its effectiveness in leveraging limited resources and producing context-aware subsequent maneuvers.
>
---
#### [new 028] Chasing Ghosts: A Simulation-to-Real Olfactory Navigation Stack with Optional Vision Augmentation
- **分类: cs.RO**

- **简介: 该论文属于无人机嗅觉导航任务，旨在解决在复杂环境下自主定位气味源的问题。通过最小传感器套件和仿真训练的导航策略，实现无需地图或外部系统的真实飞行导航。**

- **链接: [https://arxiv.org/pdf/2602.19577v1](https://arxiv.org/pdf/2602.19577v1)**

> **作者:** Kordel K. France; Ovidiu Daescu; Latifur Khan; Rohith Peddi
>
> **摘要:** Autonomous odor source localization remains a challenging problem for aerial robots due to turbulent airflow, sparse and delayed sensory signals, and strict payload and compute constraints. While prior unmanned aerial vehicle (UAV)-based olfaction systems have demonstrated gas distribution mapping or reactive plume tracing, they rely on predefined coverage patterns, external infrastructure, or extensive sensing and coordination. In this work, we present a complete, open-source UAV system for online odor source localization using a minimal sensor suite. The system integrates custom olfaction hardware, onboard sensing, and a learning-based navigation policy trained in simulation and deployed on a real quadrotor. Through our minimal framework, the UAV is able to navigate directly toward an odor source without constructing an explicit gas distribution map or relying on external positioning systems. Vision is incorporated as an optional complementary modality to accelerate navigation under certain conditions. We validate the proposed system through real-world flight experiments in a large indoor environment using an ethanol source, demonstrating consistent source-finding behavior under realistic airflow conditions. The primary contribution of this work is a reproducible system and methodological framework for UAV-based olfactory navigation and source finding under minimal sensing assumptions. We elaborate on our hardware design and open source our UAV firmware, simulation code, olfaction-vision dataset, and circuit board to the community. Code, data, and designs will be made available at https://github.com/KordelFranceTech/ChasingGhosts.
>
---
#### [new 029] Systematic Analysis of Coupling Effects on Closed-Loop and Open-Loop Performance in Aerial Continuum Manipulators
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究空中连续机械臂的动力学建模，比较解耦与耦合模型的性能。任务是评估解耦模型在保持精度的同时降低计算成本的可行性。工作包括建立模型、分析开环与闭环表现，并设计控制器进行对比实验。**

- **链接: [https://arxiv.org/pdf/2602.18684v1](https://arxiv.org/pdf/2602.18684v1)**

> **作者:** Niloufar Amiri; Shayan Sepahvand; Iraj Mantegh; Farrokh Janabi-Sharifi
>
> **备注:** Submitted to the 2026 International Conference on Unmanned Aircraft Systems (ICUAS 2026)
>
> **摘要:** This paper investigates two distinct approaches to the dynamic modeling of aerial continuum manipulators (ACMs): the decoupled and the coupled formulations. Both open-loop and closed-loop behaviors of a representative ACM are analyzed. The primary objective is to determine the conditions under which the decoupled model attains accuracy comparable to the coupled model while offering reduced computational cost under identical numerical conditions. The system dynamics are first derived using the Euler--Lagrange method under the piecewise constant curvature (PCC) assumption, with explicit treatment of the near-zero curvature singularity. A decoupled model is then obtained by neglecting the coupling terms in the ACM dynamics, enabling systematic evaluation of open-loop responses under diverse actuation profiles and external wrenches. To extend the analysis to closed-loop performance, a novel dynamics-based proportional-derivative sliding mode image-based visual servoing (DPD-SM-IBVS) controller is developed for regulating image feature errors in the presence of a moving target. The controller is implemented with both coupled and decoupled models, allowing a direct comparison of their effectiveness. The open-loop simulations reveal pronounced discrepancies between the two modeling approaches, particularly under varying torque inputs and continuum arm parameters. Conversely, the closed-loop experiments demonstrate that the decoupled model achieves tracking accuracy on par with the coupled model (within subpixel error) while incurring lower computational cost.
>
---
#### [new 030] TactEx: An Explainable Multimodal Robotic Interaction Framework for Human-Like Touch and Hardness Estimation
- **分类: cs.RO**

- **简介: 该论文提出TactEx框架，解决机器人触觉与硬度估计问题。融合视觉、触觉和语言，实现可解释的人类类似触觉感知与交互。**

- **链接: [https://arxiv.org/pdf/2602.18967v1](https://arxiv.org/pdf/2602.18967v1)**

> **作者:** Felix Verstraete; Lan Wei; Wen Fan; Dandan Zhang
>
> **备注:** Accepted by 2026 ICRA
>
> **摘要:** Accurate perception of object hardness is essential for safe and dexterous contact-rich robotic manipulation. Here, we present TactEx, an explainable multimodal robotic interaction framework that unifies vision, touch, and language for human-like hardness estimation and interactive guidance. We evaluate TactEx on fruit-ripeness assessment, a representative task that requires both tactile sensing and contextual understanding. The system fuses GelSight-Mini tactile streams with RGB observations and language prompts. A ResNet50+LSTM model estimates hardness from sequential tactile data, while a cross-modal alignment module combines visual cues with guidance from a large language model (LLM). This explainable multimodal interface allows users to distinguish ripeness levels with statistically significant class separation (p < 0.01 for all fruit pairs). For touch placement, we compare YOLO with Grounded-SAM (GSAM) and find GSAM to be more robust for fine-grained segmentation and contact-site selection. A lightweight LLM parses user instructions and produces grounded natural-language explanations linked to the tactile outputs. In end-to-end evaluations, TactEx attains 90% task success on simple user queries and generalises to novel tasks without large-scale tuning. These results highlight the promise of combining pretrained visual and tactile models with language grounding to advance explainable, human-like touch perception and decision-making in robotics.
>
---
#### [new 031] Scout-Rover cooperation: online terrain strength mapping and traversal risk estimation for planetary-analog explorations
- **分类: cs.RO**

- **简介: 该论文属于行星探测任务，旨在解决松散地形下的安全导航问题。通过 scout-rover 协作，实现地形强度在线映射和路径风险评估，提升探测器在复杂地形中的作业能力。**

- **链接: [https://arxiv.org/pdf/2602.18688v1](https://arxiv.org/pdf/2602.18688v1)**

> **作者:** Shipeng Liu; J. Diego Caporale; Yifeng Zhang; Xingjue Liao; William Hoganson; Wilson Hu; Shivangi Misra; Neha Peddinti; Rachel Holladay; Ethan Fulcher; Akshay Ram Panyam; Andrik Puentes; Jordan M. Bretzfelder; Michael Zanetti; Uland Wong; Daniel E. Koditschek; Mark Yim; Douglas Jerolmack; Cynthia Sung; Feifei Qian
>
> **备注:** 8 figures
>
> **摘要:** Robot-aided exploration of planetary surfaces is essential for understanding geologic processes, yet many scientifically valuable regions, such as Martian dunes and lunar craters, remain hazardous due to loose, deformable regolith. We present a scout-rover cooperation framework that expands safe access to such terrain using a hybrid team of legged and wheeled robots. In our approach, a high-mobility legged robot serves as a mobile scout, using proprioceptive leg-terrain interactions to estimate regolith strength during locomotion and construct spatially resolved terrain maps. These maps are integrated with rover locomotion models to estimate traversal risk and inform path planning. We validate the framework through analogue missions at the NASA Ames Lunar Simulant Testbed and the White Sands Dune Field. Experiments demonstrate (1) online terrain strength mapping from legged locomotion and (2) rover-specific traversal-risk estimation enabling safe navigation to scientific targets. Results show that scout-generated terrain maps reliably capture spatial variability and predict mobility failure modes, allowing risk-aware path planning that avoids hazardous regions. By combining embodied terrain sensing with heterogeneous rover cooperation, this framework enhances operational robustness and expands the reachable science workspace in deformable planetary environments.
>
---
#### [new 032] Online Navigation Planning for Long-term Autonomous Operation of Underwater Gliders
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于水下机器人自主导航任务，解决长期自主运行中的路径规划问题。通过构建随机最短路径马尔可夫决策过程，提出基于蒙特卡洛树搜索的在线规划方法，提升导航效率与实用性。**

- **链接: [https://arxiv.org/pdf/2602.19315v1](https://arxiv.org/pdf/2602.19315v1)**

> **作者:** Victor-Alexandru Darvariu; Charlotte Z. Reed; Jan Stratmann; Bruno Lacerda; Benjamin Allsup; Stephen Woodward; Elizabeth Siddle; Trishna Saeharaseelan; Owain Jones; Dan Jones; Tobias Ferreira; Chloe Baker; Kevin Chaplin; James Kirk; Ashley Morris; Ryan Patmore; Jeff Polton; Charlotte Williams; Alexandra Kokkinaki; Alvaro Lorenzo Lopez; Justin J. H. Buck; Nick Hawes
>
> **摘要:** Underwater glider robots have become an indispensable tool for ocean sampling. Although stakeholders are calling for tools to manage increasingly large fleets of gliders, successful autonomous long-term deployments have thus far been scarce, which hints at a lack of suitable methodologies and systems. In this work, we formulate glider navigation planning as a stochastic shortest-path Markov Decision Process and propose a sample-based online planner based on Monte Carlo Tree Search. Samples are generated by a physics-informed simulator that captures uncertain execution of controls and ocean current forecasts while remaining computationally tractable. The simulator parameters are fitted using historical glider data. We integrate these methods into an autonomous command-and-control system for Slocum gliders that enables closed-loop replanning at each surfacing. The resulting system was validated in two field deployments in the North Sea totalling approximately 3 months and 1000 km of autonomous operation. Results demonstrate improved efficiency compared to straight-to-goal navigation and show the practicality of sample-based planning for long-term marine autonomy.
>
---
#### [new 033] Robotic Fruits with Tunable Stiffness and Sensing: Towards a Methodology for Developing Realistic Physical Twins of Fruits
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决软体水果抓取中因机械特性多变导致的评估与训练难题。通过设计可调节刚度的物理孪生水果，实现对真实果实的精准模拟，提升抓取系统性能。**

- **链接: [https://arxiv.org/pdf/2602.18661v1](https://arxiv.org/pdf/2602.18661v1)**

> **作者:** Saitarun Nadipineni; Keshav Pandiyan; Kaspar Althoefer; Shinichi Hirai; Thilina Dulantha Lalitharatne
>
> **备注:** 6 pages, 5 figures, 9th IEEE-RAS International Conference on Soft Robotics (RoboSoft) 2026
>
> **摘要:** The global agri-food sector faces increasing challenges from labour shortages, high consumer demand, and supply-chain disruptions, resulting in substantial losses of unharvested produce. Robotic harvesting has emerged as a promising alternative; however, evaluating and training soft grippers for delicate fruits remains difficult due to the highly variable mechanical properties of natural produce. This makes it difficult to establish reliable benchmarks or data-driven control strategies. Existing testing practices rely on large quantities of real fruit to capture this variability, leading to inefficiency, higher costs, and waste. The methodology presented in this work aims to address these limitations by developing tunable soft physical twins that emulate the stiffness characteristics of real fruits at different ripeness levels. A fiber-reinforced pneumatic physical twin of a kiwi fruit was designed and fabricated to replicate the stiffness at different ripeness levels. Experimental results show that the stiffness of the physical twin can be tuned accurately over multiple trials (97.35 - 99.43% accuracy). Gripping tasks with a commercial robotic gripper showed that sensor feedback from the physical twin can reflect the applied gripping forces. Finally, a stress test was performed over 50 cycles showed reliable maintenance of desired stiffness (0.56 - 1.10% error). This work shows promise that robotic physical twins could adjust their stiffness to resemble that of real fruits. This can provide a sustainable, controllable platform for benchmarking and training robotic grippers.
>
---
#### [new 034] Hilbert-Augmented Reinforcement Learning for Scalable Multi-Robot Coverage and Exploration
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多机器人覆盖与探索任务，旨在提升稀疏奖励环境下的效率与可扩展性。通过引入希尔伯特空间填充先验，优化DQN和PPO算法，减少冗余并加速收敛。**

- **链接: [https://arxiv.org/pdf/2602.19400v1](https://arxiv.org/pdf/2602.19400v1)**

> **作者:** Tamil Selvan Gurunathan; Aryya Gangopadhyay
>
> **摘要:** We present a coverage framework that integrates Hilbert space-filling priors into decentralized multi-robot learning and execution. We augment DQN and PPO with Hilbert-based spatial indices to structure exploration and reduce redundancy in sparse-reward environments, and we evaluate scalability in multi-robot grid coverage. We further describe a waypoint interface that converts Hilbert orderings into curvature-bounded, time-parameterized SE(2) trajectories (planar (x, y, θ)), enabling onboard feasibility on resource-constrained robots. Experiments show improvements in coverage efficiency, redundancy, and convergence speed over DQN/PPO baselines. In addition, we validate the approach on a Boston Dynamics Spot legged robot, executing the generated trajectories in indoor environments and observing reliable coverage with low redundancy. These results indicate that geometric priors improve autonomy and scalability for swarm and legged robotics.
>
---
#### [new 035] 3D Shape Control of Extensible Multi-Section Soft Continuum Robots via Visual Servoing
- **分类: cs.RO**

- **简介: 该论文属于软体机械臂形状控制任务，解决如何通过视觉伺服实现多段软体机械臂整体形状调控的问题。工作包括提出一种无需本体感知的模型化2.5D视觉伺服算法，实现全局稳定控制。**

- **链接: [https://arxiv.org/pdf/2602.19273v1](https://arxiv.org/pdf/2602.19273v1)**

> **作者:** Abhinav Gandhi; Shou-Shan Chiang; Cagdas D. Onal; Berk Calli
>
> **摘要:** In this paper, we propose a novel vision-based control algorithm for regulating the whole body shape of extensible multisection soft continuum manipulators. Contrary to existing vision-based control algorithms in the literature that regulate the robot's end effector pose, our proposed control algorithm regulates the robot's whole body configuration, enabling us to leverage its kinematic redundancy. Additionally, our model-based 2.5D shape visual servoing provides globally stable asymptotic convergence in the robot's 3D workspace compared to the closest works in the literature that report local minima. Unlike existing visual servoing algorithms in the literature, our approach does not require information from proprioceptive sensors, making it suitable for continuum manipulators without such capabilities. Instead, robot state is estimated from images acquired by an external camera that observes the robot's whole body shape and is also utilized to close the shape control loop. Traditionally, visual servoing schemes require an image of the robot at its reference pose to generate the reference features. In this work, we utilize an inverse kinematics solver to generate reference features for the desired robot configuration and do not require images of the robot at the reference. Experiments are performed on a multisection continuum manipulator demonstrating the controller's capability to regulate the robot's whole body shape while precisely positioning the robot's end effector. Results validate our controller's ability to regulate the shape of continuum robots while demonstrating a smooth transient response and a steady-state error within 1 mm. Proof-of-concept object manipulation experiments including stacking, pouring, and pulling tasks are performed to demonstrate our controller's applicability.
>
---
#### [new 036] Contextual Safety Reasoning and Grounding for Open-World Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人安全任务，解决开放环境中动态上下文下的安全行为问题。提出CORE框架，通过视觉语言模型实现在线上下文推理与空间安全约束的实时生成和执行。**

- **链接: [https://arxiv.org/pdf/2602.19983v1](https://arxiv.org/pdf/2602.19983v1)**

> **作者:** Zachary Ravichadran; David Snyder; Alexander Robey; Hamed Hassani; Vijay Kumar; George J. Pappas
>
> **摘要:** Robots are increasingly operating in open-world environments where safe behavior depends on context: the same hallway may require different navigation strategies when crowded versus empty, or during an emergency versus normal operations. Traditional safety approaches enforce fixed constraints in user-specified contexts, limiting their ability to handle the open-ended contextual variability of real-world deployment. We address this gap via CORE, a safety framework that enables online contextual reasoning, grounding, and enforcement without prior knowledge of the environment (e.g., maps or safety specifications). CORE uses a vision-language model (VLM) to continuously reason about context-dependent safety rules directly from visual observations, grounds these rules in the physical environment, and enforces the resulting spatially-defined safe sets via control barrier functions. We provide probabilistic safety guarantees for CORE that account for perceptual uncertainty, and we demonstrate through simulation and real-world experiments that CORE enforces contextually appropriate behavior in unseen environments, significantly outperforming prior semantic safety methods that lack online contextual reasoning. Ablation studies validate our theoretical guarantees and underscore the importance of both VLM-based reasoning and spatial grounding for enforcing contextual safety in novel settings. We provide additional resources at https://zacravichandran.github.io/CORE.
>
---
#### [new 037] Hydrodynamic Performance Enhancement of Unmanned Underwater Gliders with Soft Robotic Morphing Wings for Agility Improvement
- **分类: cs.RO**

- **简介: 该论文属于水下机器人性能优化任务，旨在提升无人水下滑翔机的流体动力效率。通过软体可变形翼设计，实现比传统刚性翼更高的整体效率，增强机动性与任务可行性。**

- **链接: [https://arxiv.org/pdf/2602.20054v1](https://arxiv.org/pdf/2602.20054v1)**

> **作者:** A. Giordano; G. De Meurichy; V. Telazzi; C. Mucignat; I. Lunati; D. A. L. M. Louchard; M. Iovieno; S. F. Armanini; M. Kovac
>
> **备注:** Conference paper accepted at 9th IEEE-RAS International Conference on Soft Robotics (RoboSoft 2026)
>
> **摘要:** This work assesses the hydrodynamic efficiency of Underwater Unmanned Vehicles (UUVs) equipped with soft morphing wings compared to conventional rigid wings. Unlike rigid wings, deformable counterparts can alter their aerodynamic properties on demand. Improvements in hydrodynamic efficiency extend a UUV's operational range and may determine mission feasibility. Structural and Computational Fluid Dynamics (CFD) simulations were conducted for both a soft morphing wing and a UUV incorporating it. The results show that a UUV employing soft wings achieves 9.75 percent higher overall efficiency than an equivalent vehicle with traditional rigid wings. These findings confirm the potential of soft robotics to enhance underwater vehicle performance, particularly in applications requiring pressure-agnostic operation.
>
---
#### [new 038] Design, Locomotion, and Control of Amphibious Robots: Recent Advances
- **分类: cs.RO; physics.app-ph**

- **简介: 该论文属于机器人学任务，旨在提升 amphibious 机器人的跨介质运动能力。通过分析运动策略、执行器和控制系统，解决其在复杂环境中的适应性与自主性问题。**

- **链接: [https://arxiv.org/pdf/2602.19077v1](https://arxiv.org/pdf/2602.19077v1)**

> **作者:** Yi Jin; Chang Liu; Roger D. Quinn; Robert J. Wood; C. Chase Cao
>
> **摘要:** Amphibious robots, operating seamlessly across land and water, are advancing applications in conservation, disaster response, and defense. Their performance depends on locomotion mechanisms, actuation technologies, and sensor-control integration. This review highlights recent progress in these areas, examining movement strategies, material-based actuators, and control systems for autonomy and adaptability. Challenges and opportunities are outlined to guide future research toward more efficient, resilient, and multifunctional amphibious robots.
>
---
#### [new 039] Denoising Particle Filters: Learning State Estimation with Single-Step Objectives
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出一种基于粒子滤波的去噪方法，用于机器人状态估计。解决传统端到端方法难以解释和训练成本高的问题，通过单步状态转移训练模型，提升可组合性与效率。**

- **链接: [https://arxiv.org/pdf/2602.19651v1](https://arxiv.org/pdf/2602.19651v1)**

> **作者:** Lennart Röstel; Berthold Bäuml
>
> **摘要:** Learning-based methods commonly treat state estimation in robotics as a sequence modeling problem. While this paradigm can be effective at maximizing end-to-end performance, models are often difficult to interpret and expensive to train, since training requires unrolling sequences of predictions in time. As an alternative to end-to-end trained state estimation, we propose a novel particle filtering algorithm in which models are trained from individual state transitions, fully exploiting the Markov property in robotic systems. In this framework, measurement models are learned implicitly by minimizing a denoising score matching objective. At inference, the learned denoiser is used alongside a (learned) dynamics model to approximately solve the Bayesian filtering equation at each time step, effectively guiding predicted states toward the data manifold informed by measurements. We evaluate the proposed method on challenging robotic state estimation tasks in simulation, demonstrating competitive performance compared to tuned end-to-end trained baselines. Importantly, our method offers the desirable composability of classical filtering algorithms, allowing prior information and external sensor models to be incorporated without retraining.
>
---
#### [new 040] Equivalence and Divergence of Bayesian Log-Odds and Dempster's Combination Rule for 2D Occupancy Grids
- **分类: cs.RO**

- **简介: 该论文研究贝叶斯对数似然与Dempster组合规则在二维占据网格中的比较，旨在解决融合方法公平对比的问题。通过方法改进，验证了不同匹配准则下结果的差异。**

- **链接: [https://arxiv.org/pdf/2602.18872v1](https://arxiv.org/pdf/2602.18872v1)**

> **作者:** Tatiana Berlenko; Kirill Krinkin
>
> **备注:** 29 pages, 6 figures, 6 tables. Includes complete proofs, ablation studies, and supplementary statistical analysis
>
> **摘要:** We introduce a pignistic-transform-based methodology for fair comparison of Bayesian log-odds and Dempster's combination rule in occupancy grid mapping, matching per-observation decision probabilities to isolate the fusion rule from sensor parameterization. Under BetP matching across simulation, two real lidar datasets, and downstream path planning, Bayesian fusion is consistently favored (15/15 directional consistency, p = 3.1e-5) with small absolute differences (0.001-0.022). Under normalized plausibility matching, the direction reverses, confirming the result is matching-criterion-specific. The methodology is reusable for any future Bayesian/belief function comparison.
>
---
#### [new 041] Temporal-Logic-Aware Frontier-Based Exploration
- **分类: cs.RO**

- **简介: 该论文属于自主机器人运动规划任务，解决未知环境中满足时序逻辑规范的问题。通过引入“承诺状态”，提出一种完备的前沿探索算法，确保任务进展同时保留所有可行路径。**

- **链接: [https://arxiv.org/pdf/2602.18951v1](https://arxiv.org/pdf/2602.18951v1)**

> **作者:** Azizollah Taheri; Derya Aksaray
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** This paper addresses the problem of temporal logic motion planning for an autonomous robot operating in an unknown environment. The objective is to enable the robot to satisfy a syntactically co-safe Linear Temporal Logic (scLTL) specification when the exact locations of the desired labels are not known a priori. We introduce a new type of automaton state, referred to as commit states. These states capture intermediate task progress resulting from actions whose consequences are irreversible. In other words, certain future paths to satisfaction become not feasible after taking those actions that lead to the commit states. By leveraging commit states, we propose a sound and complete frontier-based exploration algorithm that strategically guides the robot to make progress toward the task while preserving all possible ways of satisfying it. The efficacy of the proposed method is validated through simulations.
>
---
#### [new 042] Athena: An Autonomous Open-Hardware Tracked Rescue Robot Platform
- **分类: cs.RO**

- **简介: 该论文介绍Athena救援机器人平台，用于灾难响应任务。解决机器人在复杂地形中作业及多样化任务需求的问题，设计可重构履带和机械臂，实现灵活操作与低成本控制。**

- **链接: [https://arxiv.org/pdf/2602.19898v1](https://arxiv.org/pdf/2602.19898v1)**

> **作者:** Stefan Fabian; Aljoscha Schmidt; Jonas Süß; Dishant; Aum Oza; Oskar von Stryk
>
> **备注:** https://github.com/tu-darmstadt-ros-pkg/athena
>
> **摘要:** In disaster response and situation assessment, robots have great potential in reducing the risks to the safety and health of first responders. As the situations encountered and the required capabilities of the robots deployed in such missions differ wildly and are often not known in advance, heterogeneous fleets of robots are needed to cover a wide range of mission requirements. While UAVs can quickly survey the mission environment, their ability to carry heavy payloads such as sensors and manipulators is limited. UGVs can carry required payloads to assess and manipulate the mission environment, but need to be able to deal with difficult and unstructured terrain such as rubble and stairs. The ability of tracked platforms with articulated arms (flippers) to reconfigure their geometry makes them particularly effective for navigating challenging terrain. In this paper, we present Athena, an open-hardware rescue ground robot research platform with four individually reconfigurable flippers and a reliable low-cost remote emergency stop (E-Stop) solution. A novel mounting solution using an industrial PU belt and tooth inserts allows the replacement and testing of different track profiles. The manipulator with a maximum reach of 1.54m can be used to operate doors, valves, and other objects of interest. Full CAD & PCB files, as well as all low-level software, are released as open-source contributions.
>
---
#### [new 043] Distributed and Consistent Multi-Robot Visual-Inertial-Ranging Odometry on Lie Groups
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多机器人定位任务，解决GPS拒收环境下定位漂移问题。提出DC-VIRO框架，融合视觉惯性与UWB数据，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.19173v1](https://arxiv.org/pdf/2602.19173v1)**

> **作者:** Ziwei Kang; Yizhi Zhou
>
> **摘要:** Reliable localization is a fundamental requirement for multi-robot systems operating in GPS-denied environments. Visual-inertial odometry (VIO) provides lightweight and accurate motion estimation but suffers from cumulative drift in the absence of global references. Ultra-wideband (UWB) ranging offers complementary global observations, yet most existing UWB-aided VIO methods are designed for single-robot scenarios and rely on pre-calibrated anchors, which limits their robustness in practice. This paper proposes a distributed collaborative visual-inertial-ranging odometry (DC-VIRO) framework that tightly fuses VIO and UWB measurements across multiple robots. Anchor positions are explicitly included in the system state to address calibration uncertainty, while shared anchor observations are exploited through inter-robot communication to provide additional geometric constraints. By leveraging a right-invariant error formulation on Lie groups, the proposed approach preserves the observability properties of standard VIO, ensuring estimator consistency. Simulation results with multiple robots demonstrate that DC-VIRO significantly improves localization accuracy and robustness, while simultaneously enabling anchor self-calibration in distributed settings.
>
---
#### [new 044] Simulation-Ready Cluttered Scene Estimation via Physics-aware Joint Shape and Pose Optimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于真实到仿真场景重建任务，解决 cluttered 环境中物体形状与位姿估计问题。通过物理约束的联合优化方法，提升重建的鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2602.20150v1](https://arxiv.org/pdf/2602.20150v1)**

> **作者:** Wei-Cheng Huang; Jiaheng Han; Xiaohan Ye; Zherong Pan; Kris Hauser
>
> **备注:** 15 pages, 13 figures, in submission
>
> **摘要:** Estimating simulation-ready scenes from real-world observations is crucial for downstream planning and policy learning tasks. Regretfully, existing methods struggle in cluttered environments, often exhibiting prohibitive computational cost, poor robustness, and restricted generality when scaling to multiple interacting objects. We propose a unified optimization-based formulation for real-to-sim scene estimation that jointly recovers the shapes and poses of multiple rigid objects under physical constraints. Our method is built on two key technical innovations. First, we leverage the recently introduced shape-differentiable contact model, whose global differentiability permits joint optimization over object geometry and pose while modeling inter-object contacts. Second, we exploit the structured sparsity of the augmented Lagrangian Hessian to derive an efficient linear system solver whose computational cost scales favorably with scene complexity. Building on this formulation, we develop an end-to-end real-to-sim scene estimation pipeline that integrates learning-based object initialization, physics-constrained joint shape-pose optimization, and differentiable texture refinement. Experiments on cluttered scenes with up to 5 objects and 22 convex hulls demonstrate that our approach robustly reconstructs physically valid, simulation-ready object shapes and poses.
>
---
#### [new 045] Anticipate, Adapt, Act: A Hybrid Framework for Task Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划领域，旨在解决人机协作中的故障预测与应对问题。融合LLM与概率决策模型，提升任务执行的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.19518v1](https://arxiv.org/pdf/2602.19518v1)**

> **作者:** Nabanita Dash; Ayush Kaura; Shivam Singh; Ramandeep Singh; Snehasis Banerjee; Mohan Sridharan; K. Madhava Krishna
>
> **备注:** Accepted at IEEE European Conference on Mobile Robots (ECMR)
>
> **摘要:** Anticipating and adapting to failures is a key capability robots need to collaborate effectively with humans in complex domains. This continues to be a challenge despite the impressive performance of state of the art AI planning systems and Large Language Models (LLMs) because of the uncertainty associated with the tasks and their outcomes. Toward addressing this challenge, we present a hybrid framework that integrates the generic prediction capabilities of an LLM with the probabilistic sequential decision-making capability of Relational Dynamic Influence Diagram Language. For any given task, the robot reasons about the task and the capabilities of the human attempting to complete it; predicts potential failures due to lack of ability (in the human) or lack of relevant domain objects; and executes actions to prevent such failures or recover from them. Experimental evaluation in the VirtualHome 3D simulation environment demonstrates substantial improvement in performance compared with state of the art baselines.
>
---
#### [new 046] Learning to Localize Reference Trajectories in Image-Space for Visual Navigation
- **分类: cs.RO**

- **简介: 该论文提出LoTIS模型，解决视觉导航任务中的机器人泛化问题。通过在图像空间中定位参考轨迹，实现无需校准和特定机器人训练的导航指导。**

- **链接: [https://arxiv.org/pdf/2602.18803v1](https://arxiv.org/pdf/2602.18803v1)**

> **作者:** Finn Lukas Busch; Matti Vahs; Quantao Yang; Jesús Gerardo Ortega Peimbert; Yixi Cai; Jana Tumova; Olov Andersson
>
> **摘要:** We present LoTIS, a model for visual navigation that provides robot-agnostic image-space guidance by localizing a reference RGB trajectory in the robot's current view, without requiring camera calibration, poses, or robot-specific training. Instead of predicting actions tied to specific robots, we predict the image-space coordinates of the reference trajectory as they would appear in the robot's current view. This creates robot-agnostic visual guidance that easily integrates with local planning. Consequently, our model's predictions provide guidance zero-shot across diverse embodiments. By decoupling perception from action and learning to localize trajectory points rather than imitate behavioral priors, we enable a cross-trajectory training strategy for robustness to viewpoint and camera changes. We outperform state-of-the-art methods by 20-50 percentage points in success rate on conventional forward navigation, achieving 94-98% success rate across diverse sim and real environments. Furthermore, we achieve over 5x improvements on challenging tasks where baselines fail, such as backward traversal. The system is straightforward to use: we show how even a video from a phone camera directly enables different robots to navigate to any point on the trajectory. Videos, demo, and code are available at https://finnbusch.com/lotis.
>
---
#### [new 047] NovaPlan: Zero-Shot Long-Horizon Manipulation via Closed-Loop Video Language Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出NovaPlan，用于解决零样本长周期操作任务。整合视觉语言模型与视频规划，结合几何定位的机器人执行，实现复杂装配与错误恢复。**

- **链接: [https://arxiv.org/pdf/2602.20119v1](https://arxiv.org/pdf/2602.20119v1)**

> **作者:** Jiahui Fu; Junyu Nan; Lingfeng Sun; Hongyu Li; Jianing Qian; Jennifer L. Barry; Kris Kitani; George Konidaris
>
> **备注:** 25 pages, 15 figures. Project webpage: https://nova-plan.github.io/
>
> **摘要:** Solving long-horizon tasks requires robots to integrate high-level semantic reasoning with low-level physical interaction. While vision-language models (VLMs) and video generation models can decompose tasks and imagine outcomes, they often lack the physical grounding necessary for real-world execution. We introduce NovaPlan, a hierarchical framework that unifies closed-loop VLM and video planning with geometrically grounded robot execution for zero-shot long-horizon manipulation. At the high level, a VLM planner decomposes tasks into sub-goals and monitors robot execution in a closed loop, enabling the system to recover from single-step failures through autonomous re-planning. To compute low-level robot actions, we extract and utilize both task-relevant object keypoints and human hand poses as kinematic priors from the generated videos, and employ a switching mechanism to choose the better one as a reference for robot actions, maintaining stable execution even under heavy occlusion or depth inaccuracy. We demonstrate the effectiveness of NovaPlan on three long-horizon tasks and the Functional Manipulation Benchmark (FMB). Our results show that NovaPlan can perform complex assembly tasks and exhibit dexterous error recovery behaviors without any prior demonstrations or training. Project page: https://nova-plan.github.io/
>
---
#### [new 048] Bellman Value Decomposition for Task Logic in Safe Optimal Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究安全最优控制中的任务逻辑问题，旨在解决高维环境下目标与安全规范的复杂组合难题。通过贝尔曼值分解，构建价值图结构，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2602.19532v1](https://arxiv.org/pdf/2602.19532v1)**

> **作者:** William Sharpless; Oswin So; Dylan Hirsch; Sylvia Herbert; Chuchu Fan
>
> **摘要:** Real-world tasks involve nuanced combinations of goal and safety specifications. In high dimensions, the challenge is exacerbated: formal automata become cumbersome, and the combination of sparse rewards tends to require laborious tuning. In this work, we consider the innate structure of the Bellman Value as a means to naturally organize the problem for improved automatic performance. Namely, we prove the Bellman Value for a complex task defined in temporal logic can be decomposed into a graph of Bellman Values, connected by a set of well-known Bellman equations (BEs): the Reach-Avoid BE, the Avoid BE, and a novel type, the Reach-Avoid-Loop BE. To solve the Value and optimal policy, we propose VDPPO, which embeds the decomposed Value graph into a two-layer neural net, bootstrapping the implicit dependencies. We conduct a variety of simulated and hardware experiments to test our method on complex, high-dimensional tasks involving heterogeneous teams and nonlinear dynamics. Ultimately, we find this approach greatly improves performance over existing baselines, balancing safety and liveness automatically.
>
---
#### [new 049] Bumper Drone: Elastic Morphology Design for Aerial Physical Interaction
- **分类: cs.RO**

- **简介: 该论文属于无人机物理交互任务，旨在解决不确定接触力下的稳定控制问题。通过弹性结构设计，实现自适应的触碰与持续接触，提升飞行稳定性。**

- **链接: [https://arxiv.org/pdf/2602.18976v1](https://arxiv.org/pdf/2602.18976v1)**

> **作者:** Pongporn Supa; Alex Dunnett; Feng Xiao; Rui Wu; Mirko Kovac; Basaran Bahadir Kocer
>
> **备注:** Accepted to the 9th IEEE-RAS International Conference on Soft Robotics (RoboSoft) 2026
>
> **摘要:** Aerial robots are evolving from avoiding obstacles to exploiting the environmental contact interactions for navigation, exploration and manipulation. A key challenge in such aerial physical interactions lies in handling uncertain contact forces on unknown targets, which typically demand accurate sensing and active control. We present a drone platform with elastic horns that enables touch-and-go manoeuvres - a self-regulated, consecutive bumping motion that allows the drone to maintain proximity to a wall without relying on active obstacle avoidance. It leverages environmental interaction as a form of embodied control, where low-level stabilisation and near-obstacle navigation emerge from the passive dynamic responses of the drone-obstacle system that resembles a mass-spring-damper system. Experiments show that the elastic horn can absorb impact energy while maintaining vehicle stability, reducing pitch oscillations by 38% compared to the rigid horn configuration. The lower horn arrangement was found to reduce pitch oscillations by approximately 54%. In addition to intermittent contact, the platform equipped with elastic horns also demonstrates stable, sustained contact with static objects, relying on a standard attitude PID controller.
>
---
#### [new 050] TactiVerse: Generalizing Multi-Point Tactile Sensing in Soft Robotics Using Single-Point Data
- **分类: cs.RO**

- **简介: 该论文属于软体机器人触觉感知任务，解决多点触觉传感泛化问题。提出TactiVerse框架，通过单点数据实现高精度多点触觉估计。**

- **链接: [https://arxiv.org/pdf/2602.19850v1](https://arxiv.org/pdf/2602.19850v1)**

> **作者:** Junhui Lee; Hyosung Kim; Saekwang Nam
>
> **备注:** 6 pages, 7 figures, accepted at 9th IEEE-RAS International Conference on Soft Robotics (RoboSoft)
>
> **摘要:** Real-time prediction of deformation in highly compliant soft materials remains a significant challenge in soft robotics. While vision-based soft tactile sensors can track internal marker displacements, learning-based models for 3D contact estimation heavily depend on their training datasets, inherently limiting their ability to generalize to complex scenarios such as multi-point sensing. To address this limitation, we introduce TactiVerse, a U-Net-based framework that formulates contact geometry estimation as a spatial heatmap prediction task. Even when trained exclusively on a limited dataset of single-point indentations, our architecture achieves highly accurate single-point sensing, yielding a superior mean absolute error of 0.0589 mm compared to the 0.0612 mm of a conventional regression-based CNN baseline. Furthermore, we demonstrate that augmenting the training dataset with multi-point contact data substantially enhances the sensor's multi-point sensing capabilities, significantly improving the overall mean MAE for two-point discrimination from 1.214 mm to 0.383 mm. By successfully extrapolating complex contact geometries from fundamental interactions, this methodology unlocks advanced multi-point and large-area shape sensing. Ultimately, it significantly streamlines the development of marker-based soft sensors, offering a highly scalable solution for real-world tactile perception.
>
---
#### [new 051] Design and Biomechanical Evaluation of a Lightweight Low-Complexity Soft Bilateral Ankle Exoskeleton
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于康复工程任务，旨在解决传统外骨骼增加负担的问题。设计了一种轻量、低复杂度的软式双侧踝关节外骨骼，提供足底屈辅助，验证其不影响正常步态。**

- **链接: [https://arxiv.org/pdf/2602.18569v1](https://arxiv.org/pdf/2602.18569v1)**

> **作者:** Josée Mallah; Zakii Javed; Zafer Azak; Thomas Stone; Luigi G. Occhipinti
>
> **摘要:** Many people could benefit from exoskeleton assistance during gait, for either medical or nonmedical purposes. But exoskeletons bring added mass and structure, which in turn require compensating for. In this work, we present a lightweight, low-complexity, soft bilateral ankle exoskeleton for plantarflexion assistance, with a shoe attachment design that can be mounted on top of any pair of shoes. Experimental tests show no significant difference in lower limb kinematics and kinetics when wearing the exoskeleton in zero-torque mode relative to not wearing an exoskeleton, showing that our device does not obstruct healthy gait, and proving it as a compliant and comfortable device, promising to provide effective assistance. Hence, a control system was developed, and additional tests are underway.
>
---
#### [new 052] Scalable Low-Density Distributed Manipulation Using an Interconnected Actuator Array
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决小物体高效操控问题。通过设计模块化低密度致动器阵列，实现灵活、可控的表面操作，验证了其在减少致动器密度下的有效性。**

- **链接: [https://arxiv.org/pdf/2602.19653v1](https://arxiv.org/pdf/2602.19653v1)**

> **作者:** Bailey Dacre; Rodrigo Moreno; Jørn Lambertsen; Kasper Stoy; Andrés Faíña
>
> **摘要:** Distributed Manipulator Systems, composed of arrays of robotic actuators necessitate dense actuator arrays to effectively manipulate small objects. This paper presents a system composed of modular 3-DoF robotic tiles interconnected by a compliant surface layer, forming a continuous, controllable manipulation surface. The compliant layer permits increased actuator spacing without compromising object manipulation capabilities, significantly reducing actuator density while maintaining robust control, even for smaller objects. We characterize the coupled workspace of the array and develop a manipulation strategy capable of translating objects to arbitrary positions within an N X N array. The approach is validated experimentally using a minimal 2 X 2 prototype, demonstrating the successful manipulation of objects with varied shapes and sizes.
>
---
#### [new 053] TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出TOPReward，用于机器人任务进度评估，解决RL中样本效率低和奖励稀疏的问题。通过分析VLM的token概率，提升任务进展预测效果。**

- **链接: [https://arxiv.org/pdf/2602.19313v1](https://arxiv.org/pdf/2602.19313v1)**

> **作者:** Shirui Chen; Cole Harrison; Ying-Chun Lee; Angela Jin Yang; Zhongzheng Ren; Lillian J. Ratliff; Jiafei Duan; Dieter Fox; Ranjay Krishna
>
> **摘要:** While Vision-Language-Action (VLA) models have seen rapid progress in pretraining, their advancement in Reinforcement Learning (RL) remains hampered by low sample efficiency and sparse rewards in real-world settings. Developing generalizable process reward models is essential for providing the fine-grained feedback necessary to bridge this gap, yet existing temporal value functions often fail to generalize beyond their training domains. We introduce TOPReward, a novel, probabilistically grounded temporal value function that leverages the latent world knowledge of pretrained video Vision-Language Models (VLMs) to estimate robotic task progress. Unlike prior methods that prompt VLMs to directly output progress values, which are prone to numerical misrepresentation, TOPReward extracts task progress directly from the VLM's internal token logits. In zero-shot evaluations across 130+ distinct real-world tasks and multiple robot platforms (e.g., Franka, YAM, SO-100/101), TOPReward achieves 0.947 mean Value-Order Correlation (VOC) on Qwen3-VL, dramatically outperforming the state-of-the-art GVL baseline which achieves near-zero correlation on the same open-source model. We further demonstrate that TOPReward serves as a versatile tool for downstream applications, including success detection and reward-aligned behavior cloning.
>
---
#### [new 054] Infinite-Dimensional Closed-Loop Inverse Kinematics for Soft Robots via Neural Operators
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于软体机器人控制任务，解决其无限自由度下的逆运动学问题。通过神经算子学习动作到形状的映射，构建无限维闭环逆运动学算法。**

- **链接: [https://arxiv.org/pdf/2602.18655v1](https://arxiv.org/pdf/2602.18655v1)**

> **作者:** Carina Veil; Moritz Flaschel; Ellen Kuhl; Cosimo Della Santina
>
> **摘要:** While kinematic inversion is a purely geometric problem for fully actuated rigid robots, it becomes extremely challenging for underactuated soft robots with infinitely many degrees of freedom. Closed-loop inverse kinematics (CLIK) schemes address this by introducing end-to-end mappings from actuation to task space for the controller to operate on, but typically assume finite dimensions of the underlying virtual configuration space. In this work, we extend CLIK to the infinite-dimensional domain to reason about the entire soft robot shape while solving tasks. We do this by composing an actuation-to-shape map with a shape-to-task map, deriving the differential end-to-end kinematics via an infinite-dimensional chain rule, and thereby obtaining a Jacobian-based CLIK algorithm. Since the actuation-to-shape mapping is rarely available in closed form, we propose to learn it from simulation data using neural operator networks, which are differentiable. We first present an analytical study on a constant-curvature segment, and then apply the neural version of the algorithm to a three-fiber soft robotic arm whose underlying model relies on morphoelasticity and active filament theory. This opens new possibilities for differentiable control of soft robots by exploiting full-body shape information in a continuous, infinite-dimensional framework.
>
---
#### [new 055] RoboCurate: Harnessing Diversity with Action-Verified Neural Trajectory for Robot Learning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出RoboCurate，解决机器人学习中合成数据动作质量不一致的问题。通过模拟回放评估动作质量，并增强数据多样性，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2602.18742v1](https://arxiv.org/pdf/2602.18742v1)**

> **作者:** Seungku Kim; Suhyeok Jang; Byungjun Yoon; Dongyoung Kim; John Won; Jinwoo Shin
>
> **备注:** 20 pages; 6 figures; Project page is available at https://seungkukim.github.io/robocurate/
>
> **摘要:** Synthetic data generated by video generative models has shown promise for robot learning as a scalable pipeline, but it often suffers from inconsistent action quality due to imperfectly generated videos. Recently, vision-language models (VLMs) have been leveraged to validate video quality, but they have limitations in distinguishing physically accurate videos and, even then, cannot directly evaluate the generated actions themselves. To tackle this issue, we introduce RoboCurate, a novel synthetic robot data generation framework that evaluates and filters the quality of annotated actions by comparing them with simulation replay. Specifically, RoboCurate replays the predicted actions in a simulator and assesses action quality by measuring the consistency of motion between the simulator rollout and the generated video. In addition, we unlock observation diversity beyond the available dataset via image-to-image editing and apply action-preserving video-to-video transfer to further augment appearance. We observe RoboCurate's generated data yield substantial relative improvements in success rates compared to using real data only, achieving +70.1% on GR-1 Tabletop (300 demos), +16.1% on DexMimicGen in the pre-training setup, and +179.9% in the challenging real-world ALLEX humanoid dexterous manipulation setting.
>
---
#### [new 056] Towards Dexterous Embodied Manipulation via Deep Multi-Sensory Fusion and Sparse Expert Scaling
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决复杂物理交互中多感官信息融合的问题。提出DeMUSE框架，整合视觉、力觉和深度信息，提升操作精度与实时性。**

- **链接: [https://arxiv.org/pdf/2602.19764v1](https://arxiv.org/pdf/2602.19764v1)**

> **作者:** Yirui Sun; Guangyu Zhuge; Keliang Liu; Jie Gu; Zhihao xia; Qionglin Ren; Chunxu tian; Zhongxue Ga
>
> **摘要:** Realizing dexterous embodied manipulation necessitates the deep integration of heterogeneous multimodal sensory inputs. However, current vision-centric paradigms often overlook the critical force and geometric feedback essential for complex tasks. This paper presents DeMUSE, a Deep Multimodal Unified Sparse Experts framework leveraging a Diffusion Transformer to integrate RGB, depth, and 6-axis force into a unified serialized stream. Adaptive Modality-specific Normalization (AdaMN) is employed to recalibrate modality-aware features, mitigating representation imbalance and harmonizing the heterogeneous distributions of multi-sensory signals. To facilitate efficient scaling, the architecture utilizes a Sparse Mixture-of-Experts (MoE) with shared experts, increasing model capacity for physical priors while maintaining the low inference latency required for real-time control. A Joint denoising objective synchronously synthesizes environmental evolution and action sequences to ensure physical consistency. Achieving success rates of 83.2% and 72.5% in simulation and real-world trials, DeMUSE demonstrates state-of-the-art performance, validating the necessity of deep multi-sensory integration for complex physical interactions.
>
---
#### [new 057] RotorSuite: A MATLAB/Simulink Toolbox for Tilt Multi-Rotor UAV Modeling
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出RotorSuite工具箱，用于多旋翼无人机的建模与仿真，解决传统方法耗时且易错的问题，属于无人机动力学建模任务。**

- **链接: [https://arxiv.org/pdf/2602.18814v1](https://arxiv.org/pdf/2602.18814v1)**

> **作者:** Nicola Cigarini; Giulia Michieletto; Angelo Cenedese
>
> **摘要:** In recent years, aerial platforms have evolved from passive flying sensors into versatile, contact-aware robotic systems, leading to rapid advances in platform design. Standard coplanar and collinear quadrotors have been complemented by modern tilted and tilting multi-rotor platforms with enhanced maneuverability. To properly analyze, control, and validate the performance of these emerging platforms, an accurate modeling step is required; however, this can be time-consuming, user-dependent and error-prone. To address this issue, we propose a MATLAB/Simulink toolbox for modeling and simulating the dynamics of a broad class of multi-rotor platforms through both an analytical and physics-based approaches. The toolbox, named RotorSuite, is provided with comprehensive documentation and example use cases, representing a valuable tool for didactic, research, and industrial development purposes.
>
---
#### [new 058] Distributional Stability of Tangent-Linearized Gaussian Inference on Smooth Manifolds
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究机器人中流形上的高斯推断问题，解决非高斯性和几何依赖性难题，通过线性化方法建立稳定性边界，提供诊断工具以判断是否需切换推理方式。**

- **链接: [https://arxiv.org/pdf/2602.19179v1](https://arxiv.org/pdf/2602.19179v1)**

> **作者:** Junghoon Seo; Hakjin Lee; Jaehoon Sim
>
> **摘要:** Gaussian inference on smooth manifolds is central to robotics, but exact marginalization and conditioning are generally non-Gaussian and geometry-dependent. We study tangent-linearized Gaussian inference and derive explicit non-asymptotic $W_2$ stability bounds for projection marginalization and surface-measure conditioning. The bounds separate local second-order geometric distortion from nonlocal tail leakage and, for Gaussian inputs, yield closed-form diagnostics from $(μ,Σ)$ and curvature/reach surrogates. Circle and planar-pushing experiments validate the predicted calibration transition near $\sqrt{\|Σ\|_{\mathrm{op}}}/R\approx 1/6$ and indicate that normal-direction uncertainty is the dominant failure mode when locality breaks. These diagnostics provide practical triggers for switching from single-chart linearization to multi-chart or sample-based manifold inference.
>
---
#### [new 059] Vid2Sid: Videos Can Help Close the Sim2Real Gap
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出Vid2Sid，解决sim2real物理参数校准问题。通过视频分析与优化，提升仿真与现实的一致性，提供可解释的参数调整。**

- **链接: [https://arxiv.org/pdf/2602.19359v1](https://arxiv.org/pdf/2602.19359v1)**

> **作者:** Kevin Qiu; Yu Zhang; Marek Cygan; Josie Hughes
>
> **摘要:** Calibrating a robot simulator's physics parameters (friction, damping, material stiffness) to match real hardware is often done by hand or with black-box optimizers that reduce error but cannot explain which physical discrepancies drive the error. When sensing is limited to external cameras, the problem is further compounded by perception noise and the absence of direct force or state measurements. We present Vid2Sid, a video-driven system identification pipeline that couples foundation-model perception with a VLM-in-the-loop optimizer that analyzes paired sim-real videos, diagnoses concrete mismatches, and proposes physics parameter updates with natural language rationales. We evaluate our approach on a tendon-actuated finger (rigid-body dynamics in MuJoCo) and a deformable continuum tentacle (soft-body dynamics in PyElastica). On sim2real holdout controls unseen during training, Vid2Sid achieves the best average rank across all settings, matching or exceeding black-box optimizers while uniquely providing interpretable reasoning at each iteration. Sim2sim validation confirms that Vid2Sid recovers ground-truth parameters most accurately (mean relative error under 13\% vs. 28--98\%), and ablation analysis reveals three calibration regimes. VLM-guided optimization excels when perception is clean and the simulator is expressive, while model-class limitations bound performance in more challenging settings.
>
---
#### [new 060] Safe and Interpretable Multimodal Path Planning for Multi-Agent Cooperation
- **分类: cs.RO; cs.AI; cs.HC; cs.MA**

- **简介: 该论文属于多智能体协作任务，解决 agents 间路径规划的协同问题。提出 CaPE 方法，通过语言通信安全地生成和更新路径计划。**

- **链接: [https://arxiv.org/pdf/2602.19304v1](https://arxiv.org/pdf/2602.19304v1)**

> **作者:** Haojun Shi; Suyu Ye; Katherine M. Guerrerio; Jianzhi Shen; Yifan Yin; Daniel Khashabi; Chien-Ming Huang; Tianmin Shu
>
> **摘要:** Successful cooperation among decentralized agents requires each agent to quickly adapt its plan to the behavior of other agents. In scenarios where agents cannot confidently predict one another's intentions and plans, language communication can be crucial for ensuring safety. In this work, we focus on path-level cooperation in which agents must adapt their paths to one another in order to avoid collisions or perform physical collaboration such as joint carrying. In particular, we propose a safe and interpretable multimodal path planning method, CaPE (Code as Path Editor), which generates and updates path plans for an agent based on the environment and language communication from other agents. CaPE leverages a vision-language model (VLM) to synthesize a path editing program verified by a model-based planner, grounding communication to path plan updates in a safe and interpretable way. We evaluate our approach in diverse simulated and real-world scenarios, including multi-robot and human-robot cooperation in autonomous driving, household, and joint carrying tasks. Experimental results demonstrate that CaPE can be integrated into different robotic systems as a plug-and-play module, greatly enhancing a robot's ability to align its plan to language communication from other robots or humans. We also show that the combination of the VLM-based path editing program synthesis and model-based planning safety enables robots to achieve open-ended cooperation while maintaining safety and interpretability.
>
---
#### [new 061] AdaWorldPolicy: World-Model-Driven Diffusion Policy with Online Adaptive Learning for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AdaWorldPolicy，用于机器人操作任务，解决动态环境下的适应性问题。通过整合世界模型、动作专家和力预测器，实现在线自适应学习，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.20057v1](https://arxiv.org/pdf/2602.20057v1)**

> **作者:** Ge Yuan; Qiyuan Qiao; Jing Zhang; Dong Xu
>
> **备注:** Homepage: https://AdaWorldPolicy.github.io
>
> **摘要:** Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. Effective robotic manipulation requires policies that can anticipate physical outcomes and adapt to real-world environments. In this work, we introduce a unified framework, World-Model-Driven Diffusion Policy with Online Adaptive Learning (AdaWorldPolicy) to enhance robotic manipulation under dynamic conditions with minimal human involvement. Our core insight is that world models provide strong supervision signals, enabling online adaptive learning in dynamic environments, which can be complemented by force-torque feedback to mitigate dynamic force shifts. Our AdaWorldPolicy integrates a world model, an action expert, and a force predictor-all implemented as interconnected Flow Matching Diffusion Transformers (DiT). They are interconnected via the multi-modal self-attention layers, enabling deep feature exchange for joint learning while preserving their distinct modularity characteristics. We further propose a novel Online Adaptive Learning (AdaOL) strategy that dynamically switches between an Action Generation mode and a Future Imagination mode to drive reactive updates across all three modules. This creates a powerful closed-loop mechanism that adapts to both visual and physical domain shifts with minimal overhead. Across a suite of simulated and real-robot benchmarks, our AdaWorldPolicy achieves state-of-the-art performance, with dynamical adaptive capacity to out-of-distribution scenarios.
>
---
#### [new 062] Cost-Aware Diffusion Active Search
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于主动搜索任务，解决未知环境中探索与利用的平衡问题。通过扩散模型生成前瞻动作序列，提升搜索效率并减少计算成本。**

- **链接: [https://arxiv.org/pdf/2602.19538v1](https://arxiv.org/pdf/2602.19538v1)**

> **作者:** Arundhati Banerjee; Jeff Schneider
>
> **备注:** In submission
>
> **摘要:** Active search for recovering objects of interest through online, adaptive decision making with autonomous agents requires trading off exploration of unknown environments with exploitation of prior observations in the search space. Prior work has proposed information gain and Thompson sampling based myopic, greedy approaches for agents to actively decide query or search locations when the number of targets is unknown. Decision making algorithms in such partially observable environments have also shown that agents capable of lookahead over a finite horizon outperform myopic policies for active search. Unfortunately, lookahead algorithms typically rely on building a computationally expensive search tree that is simulated and updated based on the agent's observations and a model of the environment dynamics. Instead, in this work, we leverage the sequence modeling abilities of diffusion models to sample lookahead action sequences that balance the exploration-exploitation trade-off for active search without building an exhaustive search tree. We identify the optimism bias in prior diffusion based reinforcement learning approaches when applied to the active search setting and propose mitigating solutions for efficient cost-aware decision making with both single and multi-agent teams. Our proposed algorithm outperforms standard baselines in offline reinforcement learning in terms of full recovery rate and is computationally more efficient than tree search in cost-aware active decision making.
>
---
#### [new 063] When the Inference Meets the Explicitness or Why Multimodality Can Make Us Forget About the Perfect Predictor
- **分类: cs.RO; cs.AI**

- **简介: 论文研究人机协作运输任务，探讨预测系统与显式通信系统的有效性。旨在解决人类行为不确定性问题，通过实验比较不同系统表现，发现结合使用更优。**

- **链接: [https://arxiv.org/pdf/2602.18850v1](https://arxiv.org/pdf/2602.18850v1)**

> **作者:** J. E. Domínguez-Vidal; Alberto Sanfeliu
>
> **备注:** Original version submitted to the International Journal of Social Robotics. Final version available on the SORO website
>
> **摘要:** Although in the literature it is common to find predictors and inference systems that try to predict human intentions, the uncertainty of these models due to the randomness of human behavior has led some authors to start advocating the use of communication systems that explicitly elicit human intention. In this work, it is analyzed the use of four different communication systems with a human-robot collaborative object transportation task as experimental testbed: two intention predictors (one based on force prediction and another with an enhanced velocity prediction algorithm) and two explicit communication methods (a button interface and a voice-command recognition system). These systems were integrated into IVO, a custom mobile social robot equipped with force sensor to detect the force exchange between both agents and LiDAR to detect the environment. The collaborative task required transporting an object over a 5-7 meter distance with obstacles in the middle, demanding rapid decisions and precise physical coordination. 75 volunteers perform a total of 255 executions divided into three groups, testing inference systems in the first round, communication systems in the second, and the combined strategies in the third. The results show that, 1) once sufficient performance is achieved, the human no longer notices and positively assesses technical improvements; 2) the human prefers systems that are more natural to them even though they have higher failure rates; and 3) the preferred option is the right combination of both systems.
>
---
#### [new 064] Uncertainty-Aware Rank-One MIMO Q Network Framework for Accelerated Offline Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决离线强化学习中的外推误差问题。通过引入不确定性感知的MIMO Q网络框架，有效利用OOD数据并提升学习效率。**

- **链接: [https://arxiv.org/pdf/2602.19917v1](https://arxiv.org/pdf/2602.19917v1)**

> **作者:** Thanh Nguyen; Tung Luu; Tri Ton; Sungwoong Kim; Chang D. Yoo
>
> **备注:** 10 pages, 4 Figures, IEEE Access
>
> **摘要:** Offline reinforcement learning (RL) has garnered significant interest due to its safe and easily scalable paradigm. However, training under this paradigm presents its own challenge: the extrapolation error stemming from out-of-distribution (OOD) data. Existing methodologies have endeavored to address this issue through means like penalizing OOD Q-values or imposing similarity constraints on the learned policy and the behavior policy. Nonetheless, these approaches are often beset by limitations such as being overly conservative in utilizing OOD data, imprecise OOD data characterization, and significant computational overhead. To address these challenges, this paper introduces an Uncertainty-Aware Rank-One Multi-Input Multi-Output (MIMO) Q Network framework. The framework aims to enhance Offline Reinforcement Learning by fully leveraging the potential of OOD data while still ensuring efficiency in the learning process. Specifically, the framework quantifies data uncertainty and harnesses it in the training losses, aiming to train a policy that maximizes the lower confidence bound of the corresponding Q-function. Furthermore, a Rank-One MIMO architecture is introduced to model the uncertainty-aware Q-function, \TP{offering the same ability for uncertainty quantification as an ensemble of networks but with a cost nearly equivalent to that of a single network}. Consequently, this framework strikes a harmonious balance between precision, speed, and memory efficiency, culminating in improved overall performance. Extensive experimentation on the D4RL benchmark demonstrates that the framework attains state-of-the-art performance while remaining computationally efficient. By incorporating the concept of uncertainty quantification, our framework offers a promising avenue to alleviate extrapolation errors and enhance the efficiency of offline RL.
>
---
#### [new 065] IRIS-SLAM: Unified Geo-Instance Representations for Robust Semantic Localization and Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出IRIS-SLAM，解决语义定位与建图中的几何-实例统一表示问题，提升地图一致性与回环检测可靠性。**

- **链接: [https://arxiv.org/pdf/2602.18709v1](https://arxiv.org/pdf/2602.18709v1)**

> **作者:** Tingyang Xiao; Liu Liu; Wei Feng; Zhengyu Zou; Xiaolin Zhou; Wei Sui; Hao Li; Dingwen Zhang; Zhizhong Su
>
> **备注:** 15 pages
>
> **摘要:** Geometry foundation models have significantly advanced dense geometric SLAM, yet existing systems often lack deep semantic understanding and robust loop closure capabilities. Meanwhile, contemporary semantic mapping approaches are frequently hindered by decoupled architectures and fragile data association. We propose IRIS-SLAM, a novel RGB semantic SLAM system that leverages unified geometric-instance representations derived from an instance-extended foundation model. By extending a geometry foundation model to concurrently predict dense geometry and cross-view consistent instance embeddings, we enable a semantic-synergized association mechanism and instance-guided loop closure detection. Our approach effectively utilizes viewpoint-agnostic semantic anchors to bridge the gap between geometric reconstruction and open-vocabulary mapping. Experimental results demonstrate that IRIS-SLAM significantly outperforms state-of-the-art methods, particularly in map consistency and wide-baseline loop closure reliability.
>
---
#### [new 066] Rendezvous and Docking of Mobile Ground Robots for Efficient Transportation Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于机器人协同任务，解决移动机器人在运动中自主对接的问题。通过提出一种中央MPC方法，实现可靠物理耦合，提升物流效率。**

- **链接: [https://arxiv.org/pdf/2602.19862v1](https://arxiv.org/pdf/2602.19862v1)**

> **作者:** Lars Fischer; Daniel Flögel; Sören Hohmann
>
> **备注:** 8 pages, conference paper
>
> **摘要:** In-Motion physical coupling of multiple mobile ground robots has the potential to enable new applications like in-motion transfer that improves efficiency in handling and transferring goods, which tackles current challenges in logistics. A key challenge lies in achieving reliable autonomous in-motion physical coupling of two mobile ground robots starting at any initial position. Existing approaches neglect the modeling of the docking interface and the strategy for approaching it, resulting in uncontrolled collisions that make in-motion physical coupling either impossible or inefficient. To address this challenge, we propose a central mpc approach that explicitly models the dynamics and states of two omnidirectional wheeled robots, incorporates constraints related to their docking interface, and implements an approaching strategy for rendezvous and docking. This novel approach enables omnidirectional wheeled robots with a docking interface to physically couple in motion regardless of their initial position. In addition, it makes in-motion transfer possible, which is 19.75% more time- and 21.04% energy-efficient compared to a non-coupling approach in a logistic scenario.
>
---
#### [new 067] Robust Taylor-Lagrange Control for Safety-Critical Systems
- **分类: eess.SY; cs.AI; cs.RO**

- **简介: 该论文属于安全关键控制系统任务，旨在解决CBF方法的局限性和TLC的可行性问题。提出rTLC方法，通过高阶泰勒展开提升控制实时性与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.20076v1](https://arxiv.org/pdf/2602.20076v1)**

> **作者:** Wei Xiao; Christos Cassandras; Anni Li
>
> **备注:** 7 pages
>
> **摘要:** Solving safety-critical control problem has widely adopted the Control Barrier Function (CBF) method. However, the existence of a CBF is only a sufficient condition for system safety. The recently proposed Taylor-Lagrange Control (TLC) method addresses this limitation, but is vulnerable to the feasibility preservation problem (e.g., inter-sampling effect). In this paper, we propose a robust TLC (rTLC) method to address the feasibility preservation problem. Specifically, the rTLC method expands the safety function at an order higher than the relative degree of the function using Taylor's expansion with Lagrange remainder, which allows the control to explicitly show up at the current time instead of the future time in the TLC method. The rTLC method naturally addresses the feasibility preservation problem with only one hyper-parameter (the discretization time interval size during implementation), which is much less than its counterparts. Finally, we illustrate the effectiveness of the proposed rTLC method through an adaptive cruise control problem, and compare it with existing safety-critical control methods.
>
---
#### [new 068] Positioning Modular Co-Design in Future HRI Design Research
- **分类: cs.HC; cs.RO**

- **简介: 论文探讨了模块化在人机交互设计中的应用，旨在解决机器人长期陪伴中功能固定的问题。通过共设计活动，提出PAS框架，强调个性化、适应性和可持续性。**

- **链接: [https://arxiv.org/pdf/2602.19422v1](https://arxiv.org/pdf/2602.19422v1)**

> **作者:** Lingyun Chen; Qing Xiao; Zitao Zhang; Eli Blevis; Selma Šabanović
>
> **备注:** 4 pages, 1 figure, accepted by 3rd Workshop on Designerly HRI at HRI'26
>
> **摘要:** Design-oriented HRI is increasingly interested in robots as long-term companions, yet many designs still assume a fixed form and a stable set of functions. We present an ongoing design research program that treats modularity as a designerly medium - a way to make long-term human-robot relationships discussable and material through co-design. Across a series of lifespan-oriented co-design activities, participants repeatedly reconfigured the same robot for different life stages, using modular parts to express changing needs, values, and roles. From these outcomes, we articulate PAS (Personalization-Adaptability-Sustainability) as a human-centered lens on how people enact modularity in practice: configuring for self-expression, adapting across transitions, and sustaining robots through repair, reuse, and continuity. We then sketch next steps toward a fabrication-aware, community-extensible modular platform and propose evaluation criteria for designerly HRI work that prioritize expressive adequacy, lifespan plausibility, repairability-in-use, and responsible stewardship - not only usability or performance.
>
---
#### [new 069] Self-Configurable Mesh-Networks for Scalable Distributed Submodular Bandit Optimization
- **分类: eess.SY; cs.MA; cs.RO; math.OC**

- **简介: 该论文研究多智能体在受限通信下的分布式子模带-bandit协调问题，通过自配置网格网络实现高效协作，解决通信带宽、数据率和连通性限制带来的挑战。**

- **链接: [https://arxiv.org/pdf/2602.19366v1](https://arxiv.org/pdf/2602.19366v1)**

> **作者:** Zirui Xu; Vasileios Tzoumas
>
> **摘要:** We study how to scale distributed bandit submodular coordination under realistic communication constraints in bandwidth, data rate, and connectivity. We are motivated by multi-agent tasks of active situational awareness in unknown, partially-observable, and resource-limited environments, where the agents must coordinate through agent-to-agent communication. Our approach enables scalability by (i) limiting information relays to only one-hop communication and (ii) keeping inter-agent messages small, having each agent transmit only its own action information. Despite these information-access restrictions, our approach enables near-optimal action coordination by optimizing the agents' communication neighborhoods over time, through distributed online bandit optimization, subject to the agents' bandwidth constraints. Particularly, our approach enjoys an anytime suboptimality bound that is also strictly positive for arbitrary network topologies, even disconnected. To prove the bound, we define the Value of Coordination (VoC), an information-theoretic metric that quantifies for each agent the benefit of information access to its neighbors. We validate in simulations the scalability and near-optimality of our approach: it is observed to converge faster, outperform benchmarks for bandit submodular coordination, and can even outperform benchmarks that are privileged with a priori knowledge of the environment.
>
---
#### [new 070] TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于场景流估计任务，解决自监督方法在遮挡下监督不稳定的问题。通过引入多帧时间一致性监督，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.19053v1](https://arxiv.org/pdf/2602.19053v1)**

> **作者:** Qingwen Zhang; Chenhan Jiang; Xiaomeng Zhu; Yunqi Miao; Yushan Zhang; Olov Andersson; Patric Jensfelt
>
> **备注:** CVPR 2026; 15 pages, 8 figures
>
> **摘要:** Self-supervised feed-forward methods for scene flow estimation offer real-time efficiency, but their supervision from two-frame point correspondences is unreliable and often breaks down under occlusions. Multi-frame supervision has the potential to provide more stable guidance by incorporating motion cues from past frames, yet naive extensions of two-frame objectives are ineffective because point correspondences vary abruptly across frames, producing inconsistent signals. In the paper, we present TeFlow, enabling multi-frame supervision for feed-forward models by mining temporally consistent supervision. TeFlow introduces a temporal ensembling strategy that forms reliable supervisory signals by aggregating the most temporally consistent motion cues from a candidate pool built across multiple frames. Extensive evaluations demonstrate that TeFlow establishes a new state-of-the-art for self-supervised feed-forward methods, achieving performance gains of up to 33\% on the challenging Argoverse 2 and nuScenes datasets. Our method performs on par with leading optimization-based methods, yet speeds up 150 times. The code is open-sourced at https://github.com/KTH-RPL/OpenSceneFlow along with trained model weights.
>
---
#### [new 071] MeanFuser: Fast One-Step Multi-Modal Trajectory Generation and Adaptive Reconstruction via MeanFlow for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MeanFuser，解决自动驾驶轨迹生成与重建问题，通过连续表示和高效推理提升性能与效率。**

- **链接: [https://arxiv.org/pdf/2602.20060v1](https://arxiv.org/pdf/2602.20060v1)**

> **作者:** Junli Wang; Xueyi Liu; Yinan Zheng; Zebing Xing; Pengfei Li; Guang Li; Kun Ma; Guang Chen; Hangjun Ye; Zhongpu Xia; Long Chen; Qichao Zhang
>
> **摘要:** Generative models have shown great potential in trajectory planning. Recent studies demonstrate that anchor-guided generative models are effective in modeling the uncertainty of driving behaviors and improving overall performance. However, these methods rely on discrete anchor vocabularies that must sufficiently cover the trajectory distribution during testing to ensure robustness, inducing an inherent trade-off between vocabulary size and model performance. To overcome this limitation, we propose MeanFuser, an end-to-end autonomous driving method that enhances both efficiency and robustness through three key designs. (1) We introduce Gaussian Mixture Noise (GMN) to guide generative sampling, enabling a continuous representation of the trajectory space and eliminating the dependency on discrete anchor vocabularies. (2) We adapt ``MeanFlow Identity" to end-to-end planning, which models the mean velocity field between GMN and trajectory distribution instead of the instantaneous velocity field used in vanilla flow matching methods, effectively eliminating numerical errors from ODE solvers and significantly accelerating inference. (3) We design a lightweight Adaptive Reconstruction Module (ARM) that enables the model to implicitly select from all sampled proposals or reconstruct a new trajectory when none is satisfactory via attention weights. Experiments on the NAVSIM closed-loop benchmark demonstrate that MeanFuser achieves outstanding performance without the supervision of the PDM Score. and exceptional inference efficiency, offering a robust and efficient solution for end-to-end autonomous driving. Our code and model are available at https://github.com/wjl2244/MeanFuser.
>
---
#### [new 072] VLANeXt: Recipes for Building Strong VLA Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型（VLA）任务，旨在解决VLA设计不统一、效果难以比较的问题。通过系统分析设计要素，提出VLANeXt模型，并提供通用代码库促进研究。**

- **链接: [https://arxiv.org/pdf/2602.18532v1](https://arxiv.org/pdf/2602.18532v1)**

> **作者:** Xiao-Ming Wu; Bin Fan; Kang Liao; Jian-Jian Jiang; Runze Yang; Yihang Luo; Zhonghua Wu; Wei-Shi Zheng; Chen Change Loy
>
> **备注:** 17 pages, 11 figures, Project Page: https://dravenalg.github.io/VLANeXt/
>
> **摘要:** Following the rise of large foundation models, Vision-Language-Action models (VLAs) emerged, leveraging strong visual and language understanding for general-purpose policy learning. Yet, the current VLA landscape remains fragmented and exploratory. Although many groups have proposed their own VLA models, inconsistencies in training protocols and evaluation settings make it difficult to identify which design choices truly matter. To bring structure to this evolving space, we reexamine the VLA design space under a unified framework and evaluation setup. Starting from a simple VLA baseline similar to RT-2 and OpenVLA, we systematically dissect design choices along three dimensions: foundational components, perception essentials, and action modelling perspectives. From this study, we distill 12 key findings that together form a practical recipe for building strong VLA models. The outcome of this exploration is a simple yet effective model, VLANeXt. VLANeXt outperforms prior state-of-the-art methods on the LIBERO and LIBERO-plus benchmarks and demonstrates strong generalization in real-world experiments. We will release a unified, easy-to-use codebase that serves as a common platform for the community to reproduce our findings, explore the design space, and build new VLA variants on top of a shared foundation.
>
---
#### [new 073] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LaS-Comp，解决3D形状补全任务，通过零样本学习和空间一致性设计，实现跨类别的高质量补全。**

- **链接: [https://arxiv.org/pdf/2602.18735v1](https://arxiv.org/pdf/2602.18735v1)**

> **作者:** Weilong Yan; Haipeng Li; Hao Xu; Nianjin Ye; Yihao Ai; Shuaicheng Liu; Jingyu Hu
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** This paper introduces LaS-Comp, a zero-shot and category-agnostic approach that leverages the rich geometric priors of 3D foundation models to enable 3D shape completion across diverse types of partial observations. Our contributions are threefold: First, \ourname{} harnesses these powerful generative priors for completion through a complementary two-stage design: (i) an explicit replacement stage that preserves the partial observation geometry to ensure faithful completion; and (ii) an implicit refinement stage ensures seamless boundaries between the observed and synthesized regions. Second, our framework is training-free and compatible with different 3D foundation models. Third, we introduce Omni-Comp, a comprehensive benchmark combining real-world and synthetic data with diverse and challenging partial patterns, enabling a more thorough and realistic evaluation. Both quantitative and qualitative experiments demonstrate that our approach outperforms previous state-of-the-art approaches. Our code and data will be available at \href{https://github.com/DavidYan2001/LaS-Comp}{LaS-Comp}.
>
---
#### [new 074] A Very Big Video Reasoning Suite
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **简介: 该论文属于视频推理任务，旨在解决视频模型推理能力不足的问题。提出VBVR数据集和评估框架，推动视频推理研究。**

- **链接: [https://arxiv.org/pdf/2602.20159v1](https://arxiv.org/pdf/2602.20159v1)**

> **作者:** Maijunxian Wang; Ruisi Wang; Juyi Lin; Ran Ji; Thaddäus Wiedemer; Qingying Gao; Dezhi Luo; Yaoyao Qian; Lianyu Huang; Zelong Hong; Jiahui Ge; Qianli Ma; Hang He; Yifan Zhou; Lingzi Guo; Lantao Mei; Jiachen Li; Hanwen Xing; Tianqi Zhao; Fengyuan Yu; Weihang Xiao; Yizheng Jiao; Jianheng Hou; Danyang Zhang; Pengcheng Xu; Boyang Zhong; Zehong Zhao; Gaoyun Fang; John Kitaoka; Yile Xu; Hua Xu; Kenton Blacutt; Tin Nguyen; Siyuan Song; Haoran Sun; Shaoyue Wen; Linyang He; Runming Wang; Yanzhi Wang; Mengyue Yang; Ziqiao Ma; Raphaël Millière; Freda Shi; Nuno Vasconcelos; Daniel Khashabi; Alan Yuille; Yilun Du; Ziming Liu; Bo Li; Dahua Lin; Ziwei Liu; Vikash Kumar; Yijiang Li; Lei Yang; Zhongang Cai; Hokin Deng
>
> **备注:** Homepage: https://video-reason.com/
>
> **摘要:** Rapid progress in video models has largely focused on visual quality, leaving their reasoning capabilities underexplored. Video reasoning grounds intelligence in spatiotemporally consistent visual environments that go beyond what text can naturally capture, enabling intuitive reasoning over spatiotemporal structure such as continuity, interaction, and causality. However, systematically studying video reasoning and its scaling behavior is hindered by the lack of large-scale training data. To address this gap, we introduce the Very Big Video Reasoning (VBVR) Dataset, an unprecedentedly large-scale resource spanning 200 curated reasoning tasks following a principled taxonomy and over one million video clips, approximately three orders of magnitude larger than existing datasets. We further present VBVR-Bench, a verifiable evaluation framework that moves beyond model-based judging by incorporating rule-based, human-aligned scorers, enabling reproducible and interpretable diagnosis of video reasoning capabilities. Leveraging the VBVR suite, we conduct one of the first large-scale scaling studies of video reasoning and observe early signs of emergent generalization to unseen reasoning tasks. Together, VBVR lays a foundation for the next stage of research in generalizable video reasoning. The data, benchmark toolkit, and models are publicly available at https://video-reason.com/ .
>
---
#### [new 075] Universal Pose Pretraining for Generalizable Vision-Language-Action Policies
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出Pose-VLA，解决VLA模型的特征崩溃和训练效率低问题，通过分离预训练与后训练阶段，提升动作策略的泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.19710v1](https://arxiv.org/pdf/2602.19710v1)**

> **作者:** Haitao Lin; Hanyang Yu; Jingshun Huang; He Zhang; Yonggen Ling; Ping Tan; Xiangyang Xue; Yanwei Fu
>
> **摘要:** Existing Vision-Language-Action (VLA) models often suffer from feature collapse and low training efficiency because they entangle high-level perception with sparse, embodiment-specific action supervision. Since these models typically rely on VLM backbones optimized for Visual Question Answering (VQA), they excel at semantic identification but often overlook subtle 3D state variations that dictate distinct action patterns. To resolve these misalignments, we propose Pose-VLA, a decoupled paradigm that separates VLA training into a pre-training phase for extracting universal 3D spatial priors in a unified camera-centric space, and a post-training phase for efficient embodiment alignment within robot-specific action space. By introducing discrete pose tokens as a universal representation, Pose-VLA seamlessly integrates spatial grounding from diverse 3D datasets with geometry-level trajectories from robotic demonstrations. Our framework follows a two-stage pre-training pipeline, establishing fundamental spatial grounding via poses followed by motion alignment through trajectory supervision. Extensive evaluations demonstrate that Pose-VLA achieves state-of-the-art results on RoboTwin 2.0 with a 79.5% average success rate and competitive performance on LIBERO at 96.0%. Real-world experiments further showcase robust generalization across diverse objects using only 100 demonstrations per task, validating the efficiency of our pre-training paradigm.
>
---
#### [new 076] BayesFusion-SDF: Probabilistic Signed Distance Fusion with View Planning on CPU
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出BayesFusion-SDF，用于3D重建任务，解决传统方法无法有效表达不确定性的难题，通过概率方法实现更准确的几何重建与视图规划。**

- **链接: [https://arxiv.org/pdf/2602.19697v1](https://arxiv.org/pdf/2602.19697v1)**

> **作者:** Soumya Mazumdar; Vineet Kumar Rakesh; Tapas Samanta
>
> **摘要:** Key part of robotics, augmented reality, and digital inspection is dense 3D reconstruction from depth observations. Traditional volumetric fusion techniques, including truncated signed distance functions (TSDF), enable efficient and deterministic geometry reconstruction; however, they depend on heuristic weighting and fail to transparently convey uncertainty in a systematic way. Recent neural implicit methods, on the other hand, get very high fidelity but usually need a lot of GPU power for optimization and aren't very easy to understand for making decisions later on. This work presents BayesFusion-SDF, a CPU-centric probabilistic signed distance fusion framework that conceptualizes geometry as a sparse Gaussian random field with a defined posterior distribution over voxel distances. First, a rough TSDF reconstruction is used to create an adaptive narrow-band domain. Then, depth observations are combined using a heteroscedastic Bayesian formulation that is solved using sparse linear algebra and preconditioned conjugate gradients. Randomized diagonal estimators are a quick way to get an idea of posterior uncertainty. This makes it possible to extract surfaces and plan the next best view while taking into account uncertainty. Tests on a controlled ablation scene and a CO3D object sequence show that the new method is more accurate geometrically than TSDF baselines and gives useful estimates of uncertainty for active sensing. The proposed formulation provides a clear and easy-to-use alternative to GPU-heavy neural reconstruction methods while still being able to be understood in a probabilistic way and acting in a predictable way. GitHub: https://mazumdarsoumya.github.io/BayesFusionSDF
>
---
## 更新

#### [replaced 001] Adaptive Monitoring of Stochastic Fire Front Processes via Information-seeking Predictive Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于 wildfire 监测任务，旨在通过移动代理（如无人机）自适应监控火灾扩散。解决如何在随机性下融合感知、估计与控制的问题，提出一种基于贝叶斯估计和信息搜索的控制方法。**

- **链接: [https://arxiv.org/pdf/2601.11231v2](https://arxiv.org/pdf/2601.11231v2)**

> **作者:** Savvas Papaioannou; Panayiotis Kolios; Christos G. Panayiotou; Marios M. Polycarpou
>
> **备注:** 2025 IEEE 64th Conference on Decision and Control (CDC)
>
> **摘要:** We consider the problem of adaptively monitoring a wildfire front using a mobile agent (e.g., a drone), whose trajectory determines where sensor data is collected and thus influences the accuracy of fire propagation estimation. This is a challenging problem, as the stochastic nature of wildfire evolution requires the seamless integration of sensing, estimation, and control, often treated separately in existing methods. State-of-the-art methods either impose linear-Gaussian assumptions to establish optimality or rely on approximations and heuristics, often without providing explicit performance guarantees. To address these limitations, we formulate the fire front monitoring task as a stochastic optimal control problem that integrates sensing, estimation, and control. We derive an optimal recursive Bayesian estimator for a class of stochastic nonlinear elliptical-growth fire front models. Subsequently, we transform the resulting nonlinear stochastic control problem into a finite-horizon Markov decision process and design an information-seeking predictive control law obtained via a lower confidence bound-based adaptive search algorithm with asymptotic convergence to the optimal policy.
>
---
#### [replaced 002] Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于目标导向强化学习任务，解决长序列目标达成问题。通过多步拟度量学习，提升离线方法在长视野任务中的性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.07730v3](https://arxiv.org/pdf/2511.07730v3)**

> **作者:** Bill Chunyuan Zheng; Vivek Myers; Benjamin Eysenbach; Sergey Levine
>
> **摘要:** Learning how to reach goals in an environment is a longstanding challenge in AI, yet reasoning over long horizons remains a challenge for modern methods. The key question is how to estimate the temporal distance between pairs of observations. While temporal difference methods leverage local updates to provide optimality guarantees, they often perform worse than Monte Carlo methods that perform global updates (e.g., with multi-step returns), which lack such guarantees. We show how these approaches can be integrated into a practical offline GCRL method that fits a quasimetric distance using a multistep Monte-Carlo return. We show our method outperforms existing offline GCRL methods on long-horizon simulated tasks with up to 4000 steps, even with visual observations. We also demonstrate that our method can enable stitching in the real-world robotic manipulation domain (Bridge setup). Our approach is the first end-to-end offline GCRL method that enables multistep stitching in this real-world manipulation domain from an unlabeled offline dataset of visual observations and demonstrate robust horizon generalization.
>
---
#### [replaced 003] Controllable Collision Scenario Generation via Collision Pattern Prediction
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于可控碰撞场景生成任务，旨在解决如何在仿真中生成指定类型和时间的碰撞场景。工作包括构建数据集和提出预测碰撞模式的框架，以提高安全评估效果。**

- **链接: [https://arxiv.org/pdf/2510.12206v3](https://arxiv.org/pdf/2510.12206v3)**

> **作者:** Pin-Lun Chen; Chi-Hsi Kung; Che-Han Chang; Wei-Chen Chiu; Yi-Ting Chen
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Evaluating the safety of autonomous vehicles (AVs) requires diverse, safety-critical scenarios, with collisions being especially important yet rare and unsafe to collect in the real world. Therefore, the community has been focusing on generating safety-critical scenarios in simulation. However, controlling attributes such as collision type and time-to-accident (TTA) remains challenging. We introduce a new task called controllable collision scenario generation, where the goal is to produce trajectories that realize a user-specified collision type and TTA, to investigate the feasibility of automatically generating desired collision scenarios. To support this task, we present COLLIDE, a large-scale collision scenario dataset constructed by transforming real-world driving logs into diverse collisions, balanced across five representative collision types and different TTA intervals. We propose a framework that predicts Collision Pattern, a compact and interpretable representation that captures the spatial configuration of the ego and the adversarial vehicles at impact, before rolling out full adversarial trajectories. Experiments show that our approach outperforms strong baselines in both collision rate and controllability. Furthermore, generated scenarios consistently induce higher planner failure rates, revealing limitations of existing planners. We demonstrate that these scenarios fine-tune planners for robustness improvements, contributing to safer AV deployment in different collision scenarios. Additional generated scenarios are available at our project page: https://plchen86157.github.io/conditional_scenario_generation/
>
---
#### [replaced 004] Debate2Create: Robot Co-design via Multi-Agent LLM Debate
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文提出D2C框架，用于机器人协同设计，通过多智能体辩论优化形态与奖励。解决联合优化问题，提升性能并实现奖励迁移。**

- **链接: [https://arxiv.org/pdf/2510.25850v2](https://arxiv.org/pdf/2510.25850v2)**

> **作者:** Kevin Qiu; Marek Cygan
>
> **摘要:** We introduce Debate2Create (D2C), a multi-agent LLM framework that formulates robot co-design as structured, iterative debate grounded in physics-based evaluation. A design agent and control agent engage in a thesis-antithesis-synthesis loop, while pluralistic LLM judges provide multi-objective feedback to steer exploration. Across five MuJoCo locomotion benchmarks, D2C achieves up to $3.2\times$ the default Ant score and $\sim9\times$ on Swimmer, outperforming prior LLM-based methods and black-box optimization. Iterative debate yields 18--35% gains over compute-matched zero-shot generation, and D2C-generated rewards transfer to default morphologies in 4/5 tasks. Our results demonstrate that structured multi-agent debate offers an effective alternative to hand-designed objectives for joint morphology-reward optimization.
>
---
#### [replaced 005] Anomaly detection for generic failure monitoring in robotic assembly, screwing and manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人故障监测任务，旨在解决不同机器人操作中异常检测的通用性问题。通过比较自编码器方法，评估其在多种任务和控制策略中的泛化能力与数据效率。**

- **链接: [https://arxiv.org/pdf/2509.26308v3](https://arxiv.org/pdf/2509.26308v3)**

> **作者:** Niklas Grambow; Lisa-Marie Fenner; Felipe Kempkes; Philip Hotz; Dingyuan Wan; Jörg Krüger; Kevin Haninger
>
> **备注:** 8 pages, 5 figures, 4 tables, the paper has been accepted for publication in the IEEE Robotics and Automation Letters
>
> **摘要:** Out-of-distribution states in robot manipulation often lead to unpredictable robot behavior or task failure, limiting success rates and increasing risk of damage. Anomaly detection (AD) can identify deviations from expected patterns in data, which can be used to trigger failsafe behaviors and recovery strategies. Prior work has applied data-driven AD on time series data for specific robotic tasks, however the transferability of an AD approach between different robot control strategies and task types has not been shown. Leveraging time series data, such as force/torque signals, allows to directly capture robot-environment interactions, crucial for manipulation and online failure detection. As robotic tasks can have widely signal characteristics and requirements, AD methods which can be applied in the same way to a wide range of tasks is needed, ideally with good data efficiency. We examine three industrial robotic tasks, robotic cabling, screwing, and sanding, each with multi-modal time series data and several anomalies. Several autoencoderbased methods are compared, and we evaluate the generalization across different robotic tasks and control methods (diffusion policy-, position-, and impedance-controlled). This allows us to validate the integration of AD in complex tasks involving tighter tolerances and variation from both the robot and its environment. Additionally, we evaluate data efficiency, detection latency, and task characteristics which support robust detection. The results indicate reliable detection with AUROC exceeding 0.96 in failures in the cabling and screwing task, such as incorrect or misaligned parts and obstructed targets. In the polishing task, only severe failures were reliably detected, while more subtle failure types remained undetected.
>
---
#### [replaced 006] Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决人形机器人预训练与微调间的效率问题。通过结合SAC算法与模型-based方法，提升训练效率和适应性。**

- **链接: [https://arxiv.org/pdf/2601.21363v3](https://arxiv.org/pdf/2601.21363v3)**

> **作者:** Weidong Huang; Zhehan Li; Hangxin Liu; Biao Hou; Yao Su; Jingwen Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** Reinforcement learning (RL) is widely used for humanoid control, with on-policy methods such as Proximal Policy Optimization (PPO) enabling robust training via large-scale parallel simulation and, in some cases, zero-shot deployment to real robots. However, the low sample efficiency of on-policy algorithms limits safe adaptation to new environments. Although off-policy RL and model-based RL have shown improved sample efficiency, the gap between large-scale pretraining and efficient finetuning on humanoids still exists. In this paper, we find that off-policy Soft Actor-Critic (SAC), with large-batch update and a high Update-To-Data (UTD) ratio, reliably supports large-scale pretraining of humanoid locomotion policies, achieving zero-shot deployment on real robots. For adaptation, we demonstrate that these SAC-pretrained policies can be finetuned in new environments and out-of-distribution tasks using model-based methods. Data collection in the new environment executes a deterministic policy while stochastic exploration is instead confined to a physics-informed world model. This separation mitigates the risks of random exploration during adaptation while preserving exploratory coverage for improvement. Overall, the approach couples the wall-clock efficiency of large-scale simulation during pretraining with the sample efficiency of model-based learning during fine-tuning. For code and videos, see https://lift-humanoid.github.io
>
---
#### [replaced 007] SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，解决安全性和实时性问题。通过结合流匹配与控制屏障函数，提出SafeFlowMatcher框架，确保路径安全且高效。**

- **链接: [https://arxiv.org/pdf/2509.24243v3](https://arxiv.org/pdf/2509.24243v3)**

> **作者:** Jeongyong Yang; Seunghwan Jang; SooJean Han
>
> **备注:** ICLR 2026(poster)
>
> **摘要:** Generative planners based on flow matching (FM) produce high-quality paths in a single or a few ODE steps, but their sampling dynamics offer no formal safety guarantees and can yield incomplete paths near constraints. We present SafeFlowMatcher, a planning framework that couples FM with control barrier functions (CBFs) to achieve both real-time efficiency and certified safety. SafeFlowMatcher uses a two-phase (PC) integrator: (i) a prediction phase integrates the learned FM once (or a few steps) to obtain a candidate path without intervention; (ii) a correction phase refines this path with a vanishing time-scaled vector field and a CBF-based quadratic program that minimally perturbs the vector field. We prove a barrier certificate for the resulting flow system, establishing forward invariance of a robust safe set and finite-time convergence to the safe set. In addition, by enforcing safety only on the executed path, rather than all intermediate latent paths, SafeFlowMatcher avoids distributional drift and mitigates local trap problems. Moreover, SafeFlowMatcher attains faster, smoother, and safer paths than diffusion- and FM-based baselines on maze navigation, locomotion, and robot manipulation tasks. Extensive ablations corroborate the contributions of the PC integrator and the barrier certificate.
>
---
#### [replaced 008] Unleashing the Power of Discrete-Time State Representation: Ultrafast Target-based IMU-Camera Spatial-Temporal Calibration
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-惯性标定任务，解决IMU与相机间时空参数校准问题。提出基于离散时间状态表示的高效标定方法，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2509.12846v2](https://arxiv.org/pdf/2509.12846v2)**

> **作者:** Junlin Song; Antoine Richard; Miguel Olivares-Mendez
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Visual-inertial fusion is crucial for a large amount of intelligent and autonomous applications, such as robot navigation and augmented reality. To bootstrap and achieve optimal state estimation, the spatial-temporal displacements between IMU and cameras must be calibrated in advance. Most existing calibration methods adopt continuous-time state representation, more specifically the B-spline. Despite these methods achieve precise spatial-temporal calibration, they suffer from high computational cost caused by continuous-time state representation. To this end, we propose a novel and extremely efficient calibration method that unleashes the power of discrete-time state representation. Moreover, the weakness of discrete-time state representation in temporal calibration is tackled in this paper. With the increasing production of drones, cellphones and other visual-inertial platforms, if one million devices need calibration around the world, saving one minute for the calibration of each device means saving 2083 work days in total. To benefit both the research and industry communities, the open-source implementation is released at https://github.com/JunlinSong/DT-VI-Calib.
>
---
#### [replaced 009] Wonder Wins Ways: Curiosity-Driven Exploration through Multi-Agent Contextual Calibration
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于多智能体强化学习任务，旨在解决稀疏奖励环境下探索效率低的问题。提出CERMIC框架，通过动态校准内在好奇心提升探索效果。**

- **链接: [https://arxiv.org/pdf/2509.20648v3](https://arxiv.org/pdf/2509.20648v3)**

> **作者:** Yiyuan Pan; Zhe Liu; Hesheng Wang
>
> **摘要:** Autonomous exploration in complex multi-agent reinforcement learning (MARL) with sparse rewards critically depends on providing agents with effective intrinsic motivation. While artificial curiosity offers a powerful self-supervised signal, it often confuses environmental stochasticity with meaningful novelty. Moreover, existing curiosity mechanisms exhibit a uniform novelty bias, treating all unexpected observations equally. However, peer behavior novelty, which encode latent task dynamics, are often overlooked, resulting in suboptimal exploration in decentralized, communication-free MARL settings. To this end, inspired by how human children adaptively calibrate their own exploratory behaviors via observing peers, we propose a novel approach to enhance multi-agent exploration. We introduce CERMIC, a principled framework that empowers agents to robustly filter noisy surprise signals and guide exploration by dynamically calibrating their intrinsic curiosity with inferred multi-agent context. Additionally, CERMIC generates theoretically-grounded intrinsic rewards, encouraging agents to explore state transitions with high information gain. We evaluate CERMIC on benchmark suites including VMAS, Meltingpot, and SMACv2. Empirical results demonstrate that exploration with CERMIC significantly outperforms SoTA algorithms in sparse-reward environments.
>
---
#### [replaced 010] PalpAid: Multimodal Pneumatic Tactile Sensor for Tissue Palpation
- **分类: eess.SP; cs.RO**

- **简介: 论文提出PalpAid，一种多模态气动触觉传感器，用于恢复机器人手术中的触觉感知。解决手术中触觉信息缺失问题，通过压力与声学传感实现组织分类。**

- **链接: [https://arxiv.org/pdf/2512.19010v2](https://arxiv.org/pdf/2512.19010v2)**

> **作者:** Devi Yuliarti; Ravi Prakash; Hiu Ching Cheung; Amy Strong; Patrick J. Codd; Shan Lin
>
> **备注:** IEEE-RAS RoboSoft 2026
>
> **摘要:** The tactile properties of tissue, such as elasticity and stiffness, often play an important role in surgical oncology when identifying tumors and pathological tissue boundaries. Though extremely valuable, robot-assisted surgery comes at the cost of reduced sensory information to the surgeon, with vision being the primary. Sensors proposed to overcome this sensory desert are often bulky, complex, and incompatible with the surgical workflow. We present PalpAid, a multimodal pneumatic tactile sensor to restore touch in robot-assisted surgery. PalpAid is equipped with a microphone and pressure sensor, converting contact force into an internal pressure differential. The pressure sensor acts as an event detector, while the acoustic signature assists in tissue identification. We show the design, fabrication, and assembly of sensory units with characterization tests for robustness to use, repetition cycles, and integration with a robotic system. Finally, we demonstrate the sensor's ability to classify 3D-printed hard objects with varying infills and soft ex vivo tissues. We envision PalpAid to be easily retrofitted with existing surgical/general robotic systems, allowing soft tissue palpation.
>
---
#### [replaced 011] Switching Among Feedback-Linearizing Output Sets (Melds): Dwell-Time and Compatibility Guarantees
- **分类: cs.RO; math.DS**

- **简介: 该论文研究非线性系统的输出切换控制问题，解决多输出选择间坐标不匹配带来的稳定性挑战。通过分析坐标映射和引入兼容性常数，提出保证系统稳定性的停留时间条件。**

- **链接: [https://arxiv.org/pdf/2510.17448v2](https://arxiv.org/pdf/2510.17448v2)**

> **作者:** Mirko Mizzoni; Pieter van Goor; Barbara Bazzana; Antonio Franchi
>
> **摘要:** We study switching among multiple square selections of output functions (melds) drawn from a deck of candidate outputs for nonlinear systems that are static feedback linearizable via outputs. Fixing an operating point, each meld induces a distinct feedback-linearizing coordinate chart defined on a common neighborhood. Switching between melds therefore produces state-dependent coordinate mismatches that are not captured by classical switched-system analyses. We quantify this effect through Lipschitz bounds on the cross-chart maps over a compact safe set and introduce a reference-compatibility constant that measures mismatch among reference families across melds. We derive an explicit dwell-time condition depending on controller decay rates and the compatibility constant, that guarantees exponential decay of the active-output tracking errors between switches, seamless tracking of outputs shared by consecutive melds, and uniform boundedness of the state error within the safe set. A planar 3R manipulator illustrates the results.
>
---
#### [replaced 012] MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决零样本环境导航中的词汇泛化与视觉信息丢失问题。提出多模态3D场景图，保留视觉线索以提升导航性能。**

- **链接: [https://arxiv.org/pdf/2511.10376v3](https://arxiv.org/pdf/2511.10376v3)**

> **作者:** Xun Huang; Shijia Zhao; Yunxiang Wang; Xin Lu; Wanfa Zhang; Rongsheng Qu; Weixin Li; Yunhong Wang; Chenglu Wen
>
> **备注:** 18 pages, Accepted by CVPR 2026
>
> **摘要:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relation
>
---
#### [replaced 013] Geometric Model Predictive Path Integral for Agile UAV Control with Online Collision Avoidance
- **分类: cs.RO**

- **简介: 该论文属于无人机控制任务，解决轨迹跟踪与避障问题。提出GMPPI方法，在保证跟踪精度的同时提升避障能力，通过几何控制和深度图像实现高效避障。**

- **链接: [https://arxiv.org/pdf/2510.12924v2](https://arxiv.org/pdf/2510.12924v2)**

> **作者:** Pavel Pochobradský; Ondřej Procházka; Robert Pěnička; Vojtěch Vonásek; Martin Saska
>
> **备注:** This work has been accepted to the IEEE for possible publication
>
> **摘要:** In this letter, we introduce Geometric Model Predictive Path Integral (GMPPI), a sampling-based controller capable of tracking agile trajectories while avoiding obstacles. In each iteration, GMPPI generates a large number of candidate rollout trajectories and then averages them to create a nominal control to be followed by the controlled Unmanned Aerial Vehicle (UAV). Classical Model Predictive Path Integral (MPPI) faces a trade-off between tracking precision and obstacle avoidance; high-noise random rollouts are inefficient for tracking but necessary for collision avoidance. To this end, we propose leveraging geometric SE(3) control to generate a portion of GMPPI rollouts. To maximize their benefit, we introduce a UAV-tailored cost function balancing tracking performance with obstacle avoidance. All generated rollouts are projected onto depth images for collision avoidance, representing, to our knowledge, the first method utilizing depth data directly in a UAV MPPI loop. Simulations show GMPPI matches the tracking error of an obstacle-blind geometric controller while exceeding the avoidance capabilities of state-of-the-art planners and learning-based controllers. Real-world experiments demonstrate flight at speeds up to 17 m/s and obstacle avoidance up to 10 m/s.
>
---
#### [replaced 014] Neuro-Symbolic Control with Large Language Models for Language-Guided Spatial Tasks
- **分类: cs.RO**

- **简介: 该论文属于语言引导的物理控制任务，解决LLM在连续控制中不稳定、低效的问题。提出神经符号框架，结合LLM语义理解和神经控制器，提升成功率和效率。**

- **链接: [https://arxiv.org/pdf/2512.17321v2](https://arxiv.org/pdf/2512.17321v2)**

> **作者:** Momina Liaqat Ali; Muhammad Abid; Muhammad Saqlain; Jose M. Merigo
>
> **摘要:** Although large language models (LLMs) have recently become effective tools for language-conditioned control in embodied systems, instability, slow convergence, and hallucinated actions continue to limit their direct application to continuous control. A modular neuro-symbolic control framework that clearly distinguishes between low-level motion execution and high-level semantic reasoning is proposed in this work. While a lightweight neural delta controller performs bounded, incremental actions in continuous space, a locally deployed LLM interprets symbolic tasks. We assess the suggested method in a planar manipulation setting with spatial relations between objects specified by language. Numerous tasks and local language models, such as Mistral, Phi, and LLaMA-3.2, are used in extensive experiments to compare LLM-only control, neural-only control, and the suggested LLM+DL framework. In comparison to LLM-only baselines, the results show that the neuro-symbolic integration consistently increases both success rate and efficiency, achieving average step reductions exceeding 70% and speedups of up to 8.83x while remaining robust to language model quality. The suggested framework enhances interpretability, stability, and generalization without any need of reinforcement learning or costly rollouts by controlling the LLM to symbolic outputs and allocating uninterpreted execution to a neural controller trained on artificial geometric data. These outputs show empirically that neuro-symbolic decomposition offers a scalable and principled way to integrate language understanding with ongoing control, this approach promotes the creation of dependable and effective language-guided embodied systems.
>
---
#### [replaced 015] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究双臂操作任务，解决现有模型依赖大量双臂数据的问题。通过组合两个预训练单臂视觉-语言-动作模型，提出TwinVLA框架，提升数据效率和性能。**

- **链接: [https://arxiv.org/pdf/2511.05275v2](https://arxiv.org/pdf/2511.05275v2)**

> **作者:** Hokyun Im; Euijin Jeong; Andrey Kolobov; Jianlong Fu; Youngwoon Lee
>
> **备注:** Accepted to ICLR 2026 (Poster). Project webpage : https://jellyho.github.io/TwinVLA/
>
> **摘要:** Vision-language-action models (VLAs) trained on large-scale robotic datasets have demonstrated strong performance on manipulation tasks, including bimanual tasks. However, because most public datasets focus on single-arm demonstrations, adapting VLAs for bimanual tasks typically requires substantial additional bimanual data and fine-tuning. To address this challenge, we introduce TwinVLA, a modular framework that composes two copies of a pretrained single-arm VLA into a coordinated bimanual VLA. Unlike monolithic cross-embodiment models trained on mixtures of single-arm and bimanual data, TwinVLA improves both data efficiency and performance by composing pretrained single-arm policies. Across diverse bimanual tasks in real-world and simulation settings, TwinVLA outperforms a comparably-sized monolithic RDT-1B model without requiring any bimanual pretraining. Furthermore, it narrows the gap to state-of-the-art model $π_0$, which relies on extensive proprietary bimanual data and compute cost. These results establish our modular composition approach as a data-efficient and scalable path toward high-performance bimanual manipulation, leveraging public single-arm data.
>
---
#### [replaced 016] A Primer on SO(3) Action Representations in Deep Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人控制中SO(3)动作表示问题，针对深度强化学习中的旋转动作表示进行系统评估，提出有效选择和使用旋转动作的指南。**

- **链接: [https://arxiv.org/pdf/2510.11103v2](https://arxiv.org/pdf/2510.11103v2)**

> **作者:** Martin Schuck; Sherif Samy; Angela P. Schoellig
>
> **摘要:** Many robotic control tasks require policies to act on orientations, yet the geometry of SO(3) makes this nontrivial. Because SO(3) admits no global, smooth, minimal parameterization, common representations such as Euler angles, quaternions, rotation matrices, and Lie algebra coordinates introduce distinct constraints and failure modes. While these trade-offs are well studied for supervised learning, their implications for actions in reinforcement learning remain unclear. We systematically evaluate SO(3) action representations across three standard continuous control algorithms, PPO, SAC, and TD3, under dense and sparse rewards. We compare how representations shape exploration, interact with entropy regularization, and affect training stability through empirical studies and analyze the implications of different projections for obtaining valid rotations from Euclidean network outputs. Across a suite of robotics benchmarks, we quantify the practical impact of these choices and distill simple, implementation-ready guidelines for selecting and using rotation actions. Our results highlight that representation-induced geometry strongly influences exploration and optimization and show that representing actions as tangent vectors in the local frame yields the most reliable results across algorithms. The project webpage and code are available at amacati.github.io/so3 primer.
>
---
#### [replaced 017] Memory-Efficient 2D/3D Shape Assembly of Robot Swarms
- **分类: cs.RO**

- **简介: 该论文属于机器人集群形状组装任务，旨在解决传统方法内存消耗大的问题。提出一种高效的树状表示方法，设计分布式控制器实现快速、低内存的2D/3D形状组装。**

- **链接: [https://arxiv.org/pdf/2509.26518v2](https://arxiv.org/pdf/2509.26518v2)**

> **作者:** Shuoyu Yue; Pengpeng Li; Yang Xu; Kunrui Ze; Xingjian Long; Huazi Cao; Guibin Sun
>
> **摘要:** Mean-shift-based approaches have recently emerged as a representative class of methods for robot swarm shape assembly. They rely on image-based target-shape representations to compute local density gradients and perform mean-shift exploration, which constitute their core mechanism. However, such representations incur substantial memory overhead, especially for high-resolution or 3D shapes. To address this limitation, we propose a memory-efficient tree representation that hierarchically encodes user-specified shapes in both 2D and 3D. Based on this representation, we design a behavior-based distributed controller for assignment-free shape assembly. Comparative 2D and 3D simulations against a state-of-the-art mean-shift algorithm show one to two orders of magnitude lower memory usage and two to four times faster shape entry. Physical experiments with 6 to 7 UAVs further validate real-world practicality.
>
---
#### [replaced 018] Budget Allocation Policies for Real-Time Multi-Agent Path Finding
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文研究实时多智能体路径规划问题，解决在有限规划预算下如何分配资源以提高求解效率。工作包括分析不同预算分配策略，提出更有效的智能分配方法。**

- **链接: [https://arxiv.org/pdf/2507.16874v2](https://arxiv.org/pdf/2507.16874v2)**

> **作者:** Raz Beck; Roni Stern
>
> **备注:** 11 pages, 4 figures, 4 tables
>
> **摘要:** Multi-Agent Path finding (MAPF) is the problem of finding paths for a set of agents such that each agent reaches its desired destination while avoiding collisions with the other agents. This problem arises in many robotics applications, such as automated warehouses and swarms of drones. Many MAPF solvers are designed to run offline, that is, first generate paths for all agents and then execute them. In real-world scenarios, waiting for a complete solution before allowing any robot to move is often impractical. Real-time MAPF (RT-MAPF) captures this setting by assuming that agents must begin execution after a fixed planning period, referred to as the planning budget, and execute a fixed number of actions, referred to as the execution window. This results in an iterative process in which a short plan is executed, while the next execution window is planned concurrently. Existing solutions to RT-MAPF iteratively call windowed versions of MAPF algorithms in every planning period, without explicitly considering the size of the planning budget. We address this gap and explore different policies for allocating the planning budget in windowed versions of MAPF-LNS2, a state-of-the-art MAPF algorithm. Our exploration shows that the baseline approach in which all agents draw from a shared planning budget pool is ineffective in challenging scenarios. Instead, policies that intelligently distribute the planning budget among agents are able to solve more problem instances in less time.
>
---
#### [replaced 019] MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对多步骤双臂移动操作任务，解决数据生成中的可达性和可视性约束问题，提出MoMaGen框架生成多样化数据集，提升模仿学习效果。**

- **链接: [https://arxiv.org/pdf/2510.18316v2](https://arxiv.org/pdf/2510.18316v2)**

> **作者:** Chengshu Li; Mengdi Xu; Arpit Bahety; Hang Yin; Yunfan Jiang; Huang Huang; Josiah Wong; Sujay Garlanka; Cem Gokmen; Ruohan Zhang; Weiyu Liu; Jiajun Wu; Roberto Martín-Martín; Li Fei-Fei
>
> **备注:** Project website: momagen.github.io. The first four authors contribute equally. Accpeted to International Conference on Learning Representations (ICLR 2026)
>
> **摘要:** Imitation learning from large-scale, diverse human demonstrations has been shown to be effective for training robots, but collecting such data is costly and time-consuming. This challenge intensifies for multi-step bimanual mobile manipulation, where humans must teleoperate both the mobile base and two high-DoF arms. Prior X-Gen works have developed automated data generation frameworks for static (bimanual) manipulation tasks, augmenting a few human demos in simulation with novel scene configurations to synthesize large-scale datasets. However, prior works fall short for bimanual mobile manipulation tasks for two major reasons: 1) a mobile base introduces the problem of how to place the robot base to enable downstream manipulation (reachability) and 2) an active camera introduces the problem of how to position the camera to generate data for a visuomotor policy (visibility). To address these challenges, MoMaGen formulates data generation as a constrained optimization problem that satisfies hard constraints (e.g., reachability) while balancing soft constraints (e.g., visibility while navigation). This formulation generalizes across most existing automated data generation approaches and offers a principled foundation for developing future methods. We evaluate on four multi-step bimanual mobile manipulation tasks and find that MoMaGen enables the generation of much more diverse datasets than previous methods. As a result of the dataset diversity, we also show that the data generated by MoMaGen can be used to train successful imitation learning policies using a single source demo. Furthermore, the trained policy can be fine-tuned with a very small amount of real-world data (40 demos) to be succesfully deployed on real robotic hardware. More details are on our project page: momagen.github.io.
>
---
#### [replaced 020] ALOE: Action-Level Off-Policy Evaluation for Vision-Language-Action Model Post-Training
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作模型的后训练任务，旨在解决真实场景下强化学习的离策略评估问题。提出ALOE框架，通过分块时序差分方法提升学习效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.12691v2](https://arxiv.org/pdf/2602.12691v2)**

> **作者:** Rushuai Yang; Hecheng Wang; Chiming Liu; Xiaohan Yan; Yunlong Wang; Xuan Du; Shuoyu Yue; Yongcheng Liu; Chuheng Zhang; Lizhe Qi; Yi Chen; Wei Shan; Maoqing Yao
>
> **摘要:** We study how to improve large foundation vision-language-action (VLA) systems through online reinforcement learning (RL) in real-world settings. Central to this process is the value function, which provides learning signals to guide VLA learning from experience. In practice, the value function is estimated from trajectory fragments collected from different data sources, including historical policies and intermittent human interventions. Estimating the value function of current behavior quality from the mixture data is inherently an off-policy evaluation problem. However, prior work often adopts conservative on-policy estimation for stability, which avoids direct evaluation of the current high-capacity policy and limits learning effectiveness. In this paper, we propose ALOE, an action-level off-policy evaluation framework for VLA post-training. ALOE applies chunking-based temporal-difference bootstrapping to evaluate individual action sequences instead of predicting final task outcomes. This design improves effective credit assignment to critical action chunks under sparse rewards and supports stable policy improvement. We evaluate our method on three real-world manipulation tasks, including smartphone packing as a high-precision task, laundry folding as a long-horizon deformable-object task, and bimanual pick-and-place involving multi-object perception. Across all tasks, ALOE improves learning efficiency without compromising execution speed, showing that off-policy RL can be reintroduced in a reliable manner for real-world VLA post-training. Videos and additional materials are available at our project website.
>
---
#### [replaced 021] Evaluating and Improving the Robustness of LiDAR Odometry and Localization Under Real-World Corruptions
- **分类: cs.RO**

- **简介: 该论文属于LiDAR位姿估计任务，旨在提升其在真实数据损坏下的鲁棒性。通过基准测试和两种改进策略，提高系统在噪声等干扰下的性能。**

- **链接: [https://arxiv.org/pdf/2409.10824v3](https://arxiv.org/pdf/2409.10824v3)**

> **作者:** Bo Yang; Tri Minh Triet Pham; Jinqiu Yang
>
> **摘要:** LiDAR odometry and localization are two widely used and fundamental applications in robotic and autonomous driving systems. Although state-of-the-art (SOTA) systems achieve high accuracy on clean point clouds, their robustness to corrupted data remains largely unexplored. We present the first comprehensive benchmark to evaluate the robustness of LiDAR pose-estimation techniques under 18 realistic synthetic corruptions. Our results show that, under these corruptions, odometry position errors escalate from 0.5% to more than 80%, while localization performance stays consistently high. To address this sensitivity, we propose two complementary strategies. First, we design a lightweight detection-and-filter pipeline that classifies the point cloud corruption and applies a corresponding filter (e.g., bilateral filter for noise) to restore the point cloud quality. Our classifier accurately identifies each corruption type, and the filter effectively restores odometry accuracy to near-clean data levels. Second, for learning-based systems, we show that fine-tuning using the corrupted data substantially improves robustness across all tested corruptions and even boosts performance on clean point clouds on one data sequence.
>
---
#### [replaced 022] SAGE: Scalable Agentic 3D Scene Generation for Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SAGE，用于生成可扩展的3D场景，解决 embodied AI 中真实数据收集成本高、不安全的问题。通过智能代理自动构建符合物理规则的仿真环境。**

- **链接: [https://arxiv.org/pdf/2602.10116v2](https://arxiv.org/pdf/2602.10116v2)**

> **作者:** Hongchi Xia; Xuan Li; Zhaoshuo Li; Qianli Ma; Jiashu Xu; Ming-Yu Liu; Yin Cui; Tsung-Yi Lin; Wei-Chiu Ma; Shenlong Wang; Shuran Song; Fangyin Wei
>
> **备注:** Project Page: https://research.nvidia.com/labs/dir/sage/
>
> **摘要:** Real-world data collection for embodied agents remains costly and unsafe, calling for scalable, realistic, and simulator-ready 3D environments. However, existing scene-generation systems often rely on rule-based or task-specific pipelines, yielding artifacts and physically invalid scenes. We present SAGE, an agentic framework that, given a user-specified embodied task (e.g., "pick up a bowl and place it on the table"), understands the intent and automatically generates simulation-ready environments at scale. The agent couples multiple generators for layout and object composition with critics that evaluate semantic plausibility, visual realism, and physical stability. Through iterative reasoning and adaptive tool selection, it self-refines the scenes until meeting user intent and physical validity. The resulting environments are realistic, diverse, and directly deployable in modern simulators for policy training. Policies trained purely on this data exhibit clear scaling trends and generalize to unseen objects and layouts, demonstrating the promise of simulation-driven scaling for embodied AI. Code, demos, and the SAGE-10k dataset can be found on the project page here: https://research.nvidia.com/labs/dir/sage/.
>
---
#### [replaced 023] Towards Information-Optimized Multi-Agent Path Finding: A Hybrid Framework with Reduced Inter-Agent Information Sharing
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，旨在减少信息共享的同时保证路径可行性。通过引入混合框架IO-MAPF，结合强化学习与轻量协调器，有效降低信息交换量。**

- **链接: [https://arxiv.org/pdf/2510.09469v2](https://arxiv.org/pdf/2510.09469v2)**

> **作者:** Bharath Muppasani; Ritirupa Dey; Biplav Srivastava; Vignesh Narayanan
>
> **摘要:** Multi-agent pathfinding (MAPF) remains a critical problem in robotics and autonomous systems, where agents must navigate shared spaces efficiently while avoiding conflicts. Traditional centralized algorithms with global information provide high-quality solutions but scale poorly in large-scale scenarios due to the combinatorial explosion of conflicts. Conversely, distributed approaches that have local information, particularly learning-based methods, offer better scalability by operating with relaxed information availability, yet often at the cost of solution quality. In realistic deployments, information is a constrained resource: broadcasting full agent states and goals can raise privacy concerns, strain limited bandwidth, and require extra sensing and communication hardware, increasing cost and energy use. We focus on the core question of how MAPF can be solved with minimal inter-agent information sharing while preserving solution feasibility. To this end, we present an information-centric formulation of the MAPF problem and introduce a hybrid framework, IO-MAPF, that integrates decentralized path planning with a lightweight centralized coordinator. In this framework, agents use reinforcement learning (RL) to plan independently, while the central coordinator provides minimal, targeted signals, such as static conflict-cell indicators or short conflict trajectories, that are dynamically shared to support efficient conflict resolution. We introduce an Information Units (IU) metric to quantify information use and show that our alert-driven design achieves 2x to 23x reduction in information sharing, compared to the state-of-the-art algorithms, while maintaining high success rates, demonstrating that reliable MAPF is achievable under strongly information-restricted, privacy-preserving conditions. We demonstrate the effectiveness of our algorithm using simulation and hardware experiments.
>
---
#### [replaced 024] Safe and Near-Optimal Control with Online Dynamics Learning
- **分类: eess.SY; cs.LG; cs.RO; math.DS; math.OC**

- **简介: 该论文属于强化学习任务，解决未知动态下的安全与最优控制问题。通过在线学习动态模型，确保安全并实现近优策略。**

- **链接: [https://arxiv.org/pdf/2509.16650v2](https://arxiv.org/pdf/2509.16650v2)**

> **作者:** Manish Prajapat; Johannes Köhler; Melanie N. Zeilinger; Andreas Krause
>
> **摘要:** Achieving both optimality and safety under unknown system dynamics is a central challenge in real-world deployment of agents. To address this, we introduce a notion of maximum safe dynamics learning, where sufficient exploration is performed within the space of safe policies. Our method executes $\textit{pessimistically}$ safe policies while $\textit{optimistically}$ exploring informative states and, despite not reaching them due to model uncertainty, ensures continuous online learning of dynamics. The framework achieves first-of-its-kind results: learning the dynamics model sufficiently $-$ up to an arbitrary small tolerance (subject to noise) $-$ in a finite time, while ensuring provably safe operation throughout with high probability and without requiring resets. Building on this, we propose an algorithm to maximize rewards while learning the dynamics $\textit{only to the extent needed}$ to achieve close-to-optimal performance. Unlike typical reinforcement learning (RL) methods, our approach operates online in a non-episodic setting and ensures safety throughout the learning process. We demonstrate the effectiveness of our approach in challenging domains such as autonomous car racing and drone navigation under aerodynamic effects $-$ scenarios where safety is critical and accurate modeling is difficult.
>
---
#### [replaced 025] Depth-PC: A Visual Servo Framework Integrated with Cross-Modality Fusion for Sim2Real Transfer
- **分类: cs.RO**

- **简介: 该论文属于视觉伺服任务，解决Sim2Real迁移中的精度与泛化问题。提出Depth-PC框架，结合跨模态融合与图神经网络，实现零样本迁移。**

- **链接: [https://arxiv.org/pdf/2411.17195v2](https://arxiv.org/pdf/2411.17195v2)**

> **作者:** Haoyu Zhang; Yang Liu; Yimu Jiang; Weiyang Lin; Chao Ye
>
> **摘要:** Visual servoing techniques guide robotic motion using visual information to accomplish manipulation tasks, requiring high precision and robustness against noise. Traditional methods often require prior knowledge and are susceptible to external disturbances. Learning-driven alternatives, while promising, frequently struggle with the scarcity of training data and fall short in generalization. To address these challenges, we propose Depth-PC, a novel visual servoing framework that leverages decoupled simulation-based training from real-world inference, achieving zero-shot Sim2Real transfer for servo tasks. To exploit spatial and geometric information of depth and point cloud features, we introduce cross-modal feature fusion, a first in servo tasks, followed by a dedicated Graph Neural Network to establish keypoint correspondences. Through simulation and real-world experiments, our approach demonstrates superior convergence basin and accuracy compared to SOTA methods, fulfilling the requirements for robotic servo tasks while enabling zero-shot Sim2Real transfer. In addition to the enhancements achieved with our proposed framework, we have also demonstrated the effectiveness of cross-modality feature fusion within the realm of servo tasks. Code is available at https://github.com/3nnui/Depth-PC.
>
---
#### [replaced 026] Much Ado About Noising: Dispelling the Myths of Generative Robotic Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决生成模型在策略中的实际贡献问题。通过实验发现，生成模型的优势源于迭代计算而非多模态性，提出轻量级策略可达到相似效果。**

- **链接: [https://arxiv.org/pdf/2512.01809v3](https://arxiv.org/pdf/2512.01809v3)**

> **作者:** Chaoyi Pan; Giri Anantharaman; Nai-Chieh Huang; Claire Jin; Daniel Pfrommer; Chenyang Yuan; Frank Permenter; Guannan Qu; Nicholas Boffi; Guanya Shi; Max Simchowitz
>
> **摘要:** Generative models, like flows and diffusions, have recently emerged as popular and efficacious policy parameterizations in robotics. There has been much speculation as to the factors underlying their successes, ranging from capturing multi-modal action distribution to expressing more complex behaviors. In this work, we perform a comprehensive evaluation of popular generative control policies (GCPs) on common behavior cloning (BC) benchmarks. We find that GCPs do not owe their success to their ability to capture multi-modality or to express more complex observation-to-action mappings. Instead, we find that their advantage stems from iterative computation, as long as intermediate steps are supervised during training and this supervision is paired with a suitable level of stochasticity. As a validation of our findings, we show that a minimum iterative policy (MIP), a lightweight two-step regression-based policy, essentially matches the performance of flow GCPs, and often outperforms distilled shortcut models. Our results suggest that the distribution-fitting component of GCPs is less salient than commonly believed, and point toward new design spaces focusing solely on control performance. Project page: https://simchowitzlabpublic.github.io/much-ado-about-noising-project/
>
---
#### [replaced 027] The Mean of Multi-Object Trajectories
- **分类: eess.SP; cs.RO**

- **简介: 该论文属于多目标轨迹分析任务，解决多轨迹平均问题。提出基于Fréchet均值和OSPA度量的算法，用于计算多目标轨迹的共识。**

- **链接: [https://arxiv.org/pdf/2504.20391v3](https://arxiv.org/pdf/2504.20391v3)**

> **作者:** Tran Thien Dat Nguyen; Ba Tuong Vo; Ba-Ngu Vo; Hoa Van Nguyen; Changbeom Shim
>
> **摘要:** This paper introduces the concept of a mean for trajectories and multi-object trajectories (defined as sets or multi-sets of trajectories) along with algorithms for computing them. Specifically, we use the Fréchet mean, and metrics based on the optimal sub-pattern assignment (OSPA) construct, to extend the notion of average from vectors to trajectories and multi-object trajectories. Further, we develop efficient algorithms to compute these means using greedy search and Gibbs sampling. Using distributed multi-object tracking as an application, we demonstrate that the Fréchet mean approach to multi-object trajectory consensus significantly outperforms state-of-the-art distributed multi-object tracking methods.
>
---
#### [replaced 028] Human-in-the-Loop Multi-Robot Information Gathering with Inverse Submodular Maximization
- **分类: cs.RO**

- **简介: 该论文研究人机协同的多机器人信息采集任务，解决如何根据人类指令调整机器人决策参数的问题。提出逆子模最大化方法，通过优化算法实现参数调整。**

- **链接: [https://arxiv.org/pdf/2403.10991v2](https://arxiv.org/pdf/2403.10991v2)**

> **作者:** Guangyao Shi; Shipeng Liu; Ellen Novoseller; Feifei Qian; Gaurav S. Sukhatme
>
> **摘要:** We consider a new type of inverse combinatorial optimization, Inverse Submodular Maximization (ISM), for its application in human-in-the-loop multi-robot information gathering. Forward combinatorial optimization - solving a combinatorial problem given the reward (cost)-related parameters - is widely used in multi-robot coordination. In the standard pipeline, domain experts design the reward (cost)-related parameters offline. These parameters are utilized for coordinating robots online. What if non-expert human supervisors desire to change these parameters during task execution to adapt to some new requirements? We are interested in the case where human supervisors can suggest what path primitives to take, and the robots need to change the internal decision-making parameters accordingly. We study such problems from the perspective of inverse combinatorial optimization, i.e., the process of finding parameters that give certain solutions to the problem. Specifically, we propose a new formulation for ISM for a family of multi-robot information gathering scenarios, in which we aim to find a new set of parameters that minimally deviates from the current parameters while causing a greedy algorithm to output path primitives that are the same as those desired by the human supervisors. We show that for the case with a single suggestion, such problems can be formulated as a Mixed Integer Quadratic Program (MIQP), which is intractable for existing solvers when the problem size is large. We propose a new Branch $\&$ Bound algorithm to solve such problems. For the case with multiple suggestions from several human supervisors, the problem can be cast as a multi-objective optimization and can be solved using Pareto Monte Carlo Tree Search. In numerical simulations, we demonstrate how to use ISM in multi-robot scientific data collection and event detection-driven coverage control.
>
---
#### [replaced 029] Humanoid Hanoi: Investigating Shared Whole-Body Control for Skill-Based Box Rearrangement
- **分类: cs.RO**

- **简介: 该论文研究人形机器人技能化箱体重排任务，解决长时序操作中控制器一致性问题，提出共享全肢体控制器框架并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.13850v3](https://arxiv.org/pdf/2602.13850v3)**

> **作者:** Minku Kim; Kuan-Chia Chen; Aayam Shrestha; Li Fuxin; Stefan Lee; Alan Fern
>
> **备注:** 10 pages, 6 figures, Project page: https://osudrl.github.io/Humanoid_Hanoi/
>
> **摘要:** We investigate a skill-based framework for humanoid box rearrangement that enables long-horizon execution by sequencing reusable skills at the task level. In our architecture, all skills execute through a shared, task-agnostic whole-body controller (WBC), providing a consistent closed-loop interface for skill composition, in contrast to non-shared designs that use separate low-level controllers per skill. We find that naively reusing the same pretrained WBC can reduce robustness over long horizons, as new skills and their compositions induce shifted state and command distributions. We address this with a simple data aggregation procedure that augments shared-WBC training with rollouts from closed-loop skill execution under domain randomization. To evaluate the approach, we introduce Humanoid Hanoi, a long-horizon Tower-of-Hanoi box rearrangement benchmark, and report results in simulation and on the Digit V3 humanoid robot, demonstrating fully autonomous rearrangement over extended horizons and quantifying the benefits of the shared-WBC approach over non-shared baselines. Project page: https://osudrl.github.io/Humanoid_Hanoi/
>
---
#### [replaced 030] FLUID: A Fine-Grained Lightweight Urban Signalized-Intersection Dataset of Dense Conflict Trajectories
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出FLUID数据集，用于城市信号交叉口的交通参与者轨迹分析。旨在解决现有数据集在场景代表性和数据质量上的不足，通过无人机采集高精度轨迹数据，支持交通行为研究与自动驾驶发展。**

- **链接: [https://arxiv.org/pdf/2509.00497v2](https://arxiv.org/pdf/2509.00497v2)**

> **作者:** Yiyang Chen; Zhigang Wu; Guohong Zheng; Xuesong Wu; Liwen Xu; Haoyuan Tang; Zhaocheng He; Haipeng Zeng
>
> **备注:** 30 pages, 16 figures
>
> **摘要:** The trajectory data of traffic participants (TPs) is a fundamental resource for evaluating traffic conditions and optimizing policies, especially at urban intersections. Although data acquisition using drones is efficient, existing datasets still have limitations in scene representativeness, information richness, and data fidelity. This study introduces FLUID, comprising a fine-grained trajectory dataset that captures dense conflicts at typical urban signalized intersections, and a lightweight, full-pipeline framework for drone-based trajectory processing. FLUID covers three distinct intersection types, with approximately 5 hours of recording time and featuring over 20,000 TPs across 8 categories. Notably, the dataset records an average of 2.8 vehicle conflicts per minute across all scenes, with roughly 15% of all recorded motor vehicles directly involved in these conflicts. FLUID provides comprehensive data, including trajectories, traffic signals, maps, and raw videos. Comparison with the DataFromSky platform and ground-truth measurements validates its high spatio-temporal accuracy. Through a detailed classification of motor vehicle conflicts and violations, FLUID reveals a diversity of interactive behaviors, demonstrating its value for human preference mining, traffic behavior modeling, and autonomous driving research.
>
---
#### [replaced 031] Query-Based Adaptive Aggregation for Multi-Dataset Joint Training Toward Universal Visual Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉位置识别任务，旨在解决多数据集训练中特征聚合能力不足的问题。提出Query-Based Adaptive Aggregation方法，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2507.03831v4](https://arxiv.org/pdf/2507.03831v4)**

> **作者:** Jiuhong Xiao; Yang Zhou; Giuseppe Loianno
>
> **备注:** 8 pages, 4 figures, accepted at ICRA 2026
>
> **摘要:** Deep learning methods for Visual Place Recognition (VPR) have advanced significantly, largely driven by large-scale datasets. However, most existing approaches are trained on a single dataset, which can introduce dataset-specific inductive biases and limit model generalization. While multi-dataset joint training offers a promising solution for developing universal VPR models, divergences among training datasets can saturate the limited information capacity in feature aggregation layers, leading to suboptimal performance. To address these challenges, we propose Query-based Adaptive Aggregation (QAA), a novel feature aggregation technique that leverages learned queries as reference codebooks to effectively enhance information capacity without significant computational or parameter complexity. We show that computing the Cross-query Similarity (CS) between query-level image features and reference codebooks provides a simple yet effective way to generate robust descriptors. Our results demonstrate that QAA outperforms state-of-the-art models, achieving balanced generalization across diverse datasets while maintaining peak performance comparable to dataset-specific models. Ablation studies further explore QAA's mechanisms and scalability. Visualizations reveal that the learned queries exhibit diverse attention patterns across datasets. Project page: http://xjh19971.github.io/QAA.
>
---
#### [replaced 032] Continuum Robot State Estimation with Actuation Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，解决 continuum 机器人在存在驱动不确定性时的精确状态估计问题。通过结合 Cosserat 杆模型与优化方法，实现多架构机器人的实时高精度估计。**

- **链接: [https://arxiv.org/pdf/2601.04493v2](https://arxiv.org/pdf/2601.04493v2)**

> **作者:** James M. Ferguson; Alan Kuntz; Tucker Hermans
>
> **备注:** Public preprint for IEEE RAL
>
> **摘要:** Continuum robots are flexible, thin manipulators capable of navigating confined or delicate environments making them well suited for surgical applications. Previous approaches to continuum robot state estimation typically rely on simplified, deterministic actuation models. In contrast, our method jointly estimates robot shape, external loads, internal stresses, and actuation inputs. We adopt a discrete Cosserat rod formulation and show that, when paired with a midpoint integration rule, it achieves high numerical accuracy with relatively few state nodes. This discretization naturally induces a factor-graph structure for sparse nonlinear optimization on SE(3). We extend the formulation with actuation factors for tendon-driven robots and combine multiple rod graphs for parallel continuum robots with closed-loop topologies. By explicitly including actuation variables in the state, the linearized system can be reused to extract manipulator Jacobians, which we leverage in performing trajectory tracking. Finally, we validate the approach experimentally on a surgical concentric tube robot. Overall, our approach enables principled, real-time estimation across multiple continuum robot architectures, accounting for actuation uncertainty and providing direct access to manipulator Jacobians.
>
---
#### [replaced 033] Learning a Shape-adaptive Assist-as-needed Rehabilitation Policy from Therapist-informed Input
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于康复机器人任务，旨在解决远程康复中安全交互与自适应问题。通过引入治疗师反馈，学习形状自适应的辅助策略，提升运动平滑度并减少修正力。**

- **链接: [https://arxiv.org/pdf/2510.04666v3](https://arxiv.org/pdf/2510.04666v3)**

> **作者:** Zhimin Hou; Jiacheng Hou; Xiao Chen; Hamid Sadeghian; Tianyu Ren; Sami Haddadin
>
> **摘要:** Therapist-in-the-loop robotic rehabilitation has shown great promise in enhancing rehabilitation outcomes by integrating the strengths of therapists and robotic systems. However, its broader adoption remains limited due to insufficient safe interaction and limited adaptation capability. This article proposes a novel telerobotics-mediated framework that enables therapists to intuitively and safely deliver assist-as-needed~(AAN) therapy based on two primary contributions. First, our framework encodes the therapist-informed corrective force into via-points in a latent space, allowing the therapist to provide only minimal assistance while encouraging patient maintaining own motion preferences. Second, a shape-adaptive ANN rehabilitation policy is learned to partially and progressively deform the reference trajectory for movement therapy based on encoded patient motion preferences and therapist-informed via-points. The effectiveness of the proposed shape-adaptive AAN strategy was validated on a telerobotic rehabilitation system using two representative tasks. The results demonstrate its practicality for remote AAN therapy and its superiority over two state-of-the-art methods in reducing corrective force and improving movement smoothness.
>
---
#### [replaced 034] Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决传统分层策略在泛化能力上的不足。通过引入语言对齐的3D关键点预测，提升模型在新指令和环境中的适应能力。**

- **链接: [https://arxiv.org/pdf/2509.23575v2](https://arxiv.org/pdf/2509.23575v2)**

> **作者:** Jianshu Hu; Lidi Wang; Shujia Li; Yunpeng Jiang; Xiao Li; Paul Weng; Yutong Ban
>
> **备注:** Published in ICLR 2026
>
> **摘要:** Hierarchical coarse-to-fine policy, where a coarse branch predicts a region of interest to guide a fine-grained action predictor, has demonstrated significant potential in robotic 3D manipulation tasks by especially enhancing sample efficiency and enabling more precise manipulation. However, even augmented with pre-trained models, these hierarchical policies still suffer from generalization issues. To enhance generalization to novel instructions and environment variations, we propose Coarse-to-fine Language-Aligned manipulation Policy (CLAP), a framework that integrates three key components: 1) task decomposition, 2) VLM fine-tuning for 3D keypoint prediction, and 3) 3D-aware representation. Through comprehensive experiments in simulation and on a real robot, we demonstrate its superior generalization capability. Specifically, on GemBench, a benchmark designed for evaluating generalization, our approach achieves a 12\% higher average success rate than the SOTA method while using only 1/5 of the training trajectories. In real-world experiments, our policy, trained on only 10 demonstrations, successfully generalizes to novel instructions and environments.
>
---
#### [replaced 035] Find the Fruit: Zero-Shot Sim2Real RL for Occlusion-Aware Plant Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主采摘任务，解决遮挡和结构不确定性下的植物操作问题。通过模拟到现实的强化学习框架，学习策略以暴露目标果实，提升采摘成功率。**

- **链接: [https://arxiv.org/pdf/2505.16547v3](https://arxiv.org/pdf/2505.16547v3)**

> **作者:** Nitesh Subedi; Hsin-Jung Yang; Devesh K. Jha; Soumik Sarkar
>
> **摘要:** Autonomous harvesting in the open presents a complex manipulation problem. In most scenarios, an autonomous system has to deal with significant occlusion and require interaction in the presence of large structural uncertainties (every plant is different). Perceptual and modeling uncertainty make design of reliable manipulation controllers for harvesting challenging, resulting in poor performance during deployment. We present a sim2real reinforcement learning (RL) framework for occlusion-aware plant manipulation, where a policy is learned entirely in simulation to reposition stems and leaves to reveal target fruit(s). In our proposed approach, we decouple high-level kinematic planning from low-level compliant control which simplifies the sim2real transfer. This decomposition allows the learned policy to generalize across multiple plants with different stiffness and morphology. In experiments with multiple real-world plant setups, our system achieves up to 86.7% success in exposing target fruits, demonstrating robustness to occlusion variation and structural uncertainty.
>
---
#### [replaced 036] Impact-Robust Posture Optimization for Aerial Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在提升空中机械臂在碰撞时的鲁棒性。通过优化冗余机器人的姿态，减少碰撞后的状态波动，提高安全性与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.13762v2](https://arxiv.org/pdf/2602.13762v2)**

> **作者:** Amr Afifi; Ahmad Gazar; Javier Alonso-Mora; Paolo Robuffo Giordano; Antonio Franchi
>
> **摘要:** We present a novel method for optimizing the posture of kinematically redundant torque-controlled robots to improve robustness during impacts. A rigid impact model is used as the basis for a configuration-dependent metric that quantifies the variation between pre- and post-impact velocities. By finding configurations (postures) that minimize the aforementioned metric, spikes in the robot's state and input commands can be significantly reduced during impacts, improving safety and robustness. The problem of identifying impact-robust postures is posed as a min-max optimization of the aforementioned metric. To overcome the real-time intractability of the problem, we reformulate it as a gradient-based motion task that iteratively guides the robot towards configurations that minimize the proposed metric. This task is embedded within a task-space inverse dynamics (TSID) whole-body controller, enabling seamless integration with other control objectives. The method is applied to a kinematically redundant aerial manipulator performing repeated point contact tasks. We test our method inside a realistic physics simulator and compare it with the nominal TSID. Our method leads to a reduction (up to 51% w.r.t. standard TSID) of post-impact spikes in the robot's configuration and successfully avoids actuator saturation. Moreover, we demonstrate the importance of kinematic redundancy for impact robustness using additional numerical simulations on a quadruped and a humanoid robot, resulting in up to 45% reduction of post-impact spikes in the robot's state w.r.t. nominal TSID.
>
---
#### [replaced 037] Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule
- **分类: cs.RO; cs.CV; stat.AP**

- **简介: 该论文属于自动驾驶感知评估任务，解决传统指标无法反映算法不确定性的问题。提出PCD和aPCD度量感知稳定性与鲁棒性，并构建SensorRainFall数据集进行验证。**

- **链接: [https://arxiv.org/pdf/2506.09217v2](https://arxiv.org/pdf/2506.09217v2)**

> **作者:** Boyu Jiang; Liang Shi; Zhengzhi Lin; Lanxin Xiang; Loren Stowe; Feng Guo
>
> **备注:** This paper has been accepted to the CVPR 2026 Main Conference
>
> **摘要:** The safety of autonomous driving systems (ADS) depends on accurate perception across distance and driving conditions. The outputs of AI perception algorithms are stochastic, which have a major impact on decision making and safety outcomes, including time-to-collision estimation. However, current perception evaluation metrics do not reflect the stochastic nature of perception algorithms. We introduce the Perception Characteristics Distance (PCD), a novel metric incorporating model output uncertainty as represented by the farthest distance at which an object can be reliably detected. To represent a system's overall perception capability in terms of reliable detection distance, we average PCD values across multiple detection quality and probabilistic thresholds to produce the average PCD (aPCD). For empirical validation, we present the SensorRainFall dataset, collected on the Virginia Smart Road using a sensor-equipped vehicle (cameras, radar, and LiDAR) under different weather (clear and rainy) and illumination conditions (daylight, streetlight, and nighttime). The dataset includes ground-truth distances, bounding boxes, and segmentation masks for target objects. Experiments with state-of-the-art models show that aPCD captures meaningful differences across weather, daylight, and illumination conditions, which traditional evaluation metrics fail to reflect. PCD provides an uncertainty-aware measure of perception performance, supporting safer and more robust ADS operation, while the SensorRainFall dataset offers a valuable benchmark for evaluation. The SensorRainFall dataset is publicly available at https://www.kaggle.com/datasets/datadrivenwheels/sensorrainfall, and the evaluation code is available at https://github.com/datadrivenwheels/PCD_Python.
>
---
#### [replaced 038] Resource-Aware Distributed Submodular Maximization: A Paradigm for Multi-Robot Decision-Making
- **分类: math.OC; cs.AI; cs.MA; cs.RO; eess.SY**

- **简介: 该论文研究多机器人决策问题，解决资源有限下协调与优化的平衡。提出一种算法，实现分布式子模最大化，兼顾效率与资源消耗。**

- **链接: [https://arxiv.org/pdf/2204.07520v4](https://arxiv.org/pdf/2204.07520v4)**

> **作者:** Zirui Xu; Vasileios Tzoumas
>
> **备注:** Updated presentation. Accepted to the 2022 IEEE Conference on Decision and Control (CDC)
>
> **摘要:** Multi-robot decision-making is the process where multiple robots coordinate actions. In this paper, we aim for efficient and effective multi-robot decision-making despite the robots' limited on-board resources and the often resource-demanding complexity of their tasks. We introduce the first algorithm enabling the robots to choose with which few other robots to coordinate and provably balance the trade-off of centralized vs. decentralized coordination. Particularly, centralization favors globally near-optimal decision-making but at the cost of increased on-board resource requirements; whereas, decentralization favors minimal resource requirements but at a global suboptimality cost. All robots can thus afford our algorithm, irrespective of their resources. We are motivated by the future of autonomy that involves multiple robots coordinating actions to complete resource-demanding tasks, such as target tracking, area coverage, and monitoring. To provide closed-form guarantees, we focus on maximization problems involving monotone and 2nd-order submodular functions. To capture the cost of decentralization, we introduce the notion of Centralization Of Information among non-Neighbors (COIN). We validate our algorithm in simulated scenarios of image covering.
>
---
#### [replaced 039] KINESIS: Motion Imitation for Human Musculoskeletal Locomotion
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; q-bio.NC**

- **简介: 该论文提出KINESIS，属于运动模仿任务，旨在解决人体骨骼肌运动控制建模问题，通过强化学习生成符合人类EMG活动的肌肉控制模式。**

- **链接: [https://arxiv.org/pdf/2503.14637v2](https://arxiv.org/pdf/2503.14637v2)**

> **作者:** Merkourios Simos; Alberto Silvio Chiappa; Alexander Mathis
>
> **备注:** Accepted to ICRA. Here we include an appendix
>
> **摘要:** How do humans move? Advances in reinforcement learning (RL) have produced impressive results in capturing human motion using physics-based humanoid control. However, torque-controlled humanoids fail to model key aspects of human motor control such as biomechanical joint constraints \& non-linear and overactuated musculotendon control. We present KINESIS, a model-free motion imitation framework that tackles these challenges. KINESIS is trained on 1.8 hours of locomotion data and achieves strong motion imitation performance on unseen trajectories. Through a negative mining approach, KINESIS learns robust locomotion priors that we leverage to deploy the policy on several downstream tasks such as text-to-control, target point reaching, and football penalty kicks. Importantly, KINESIS learns to generate muscle activity patterns that correlate well with human EMG activity. We show that these results scale seamlessly across biomechanical model complexity, demonstrating control of up to 290 muscles. Overall, the physiological plausibility makes KINESIS a promising model for tackling challenging problems in human motor control. Code, videos and benchmarks are available at https://github.com/amathislab/Kinesis.
>
---
#### [replaced 040] Coordinated motion control of a wire arc additive manufacturing robotic system for multi-directional building parts
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决WAAM中复杂零件成形时层方向与重力不一致的问题。通过算法协调两台机器人运动，确保每层沉积方向与重力对齐，提升零件质量。**

- **链接: [https://arxiv.org/pdf/2505.14858v2](https://arxiv.org/pdf/2505.14858v2)**

> **作者:** Fernando Coutinho; Nicolas Lizarralde; Fernando Lizarralde
>
> **摘要:** This work investigates the manufacturing of complex shapes parts with wire arc additive manufacturing (WAAM). In order to guarantee the integrity and quality of each deposited layer that composes the final piece, the deposition process is usually carried out in a flat position. However, for complex geometry parts with non-flat surfaces, this strategy causes unsupported overhangs and staircase effect, which contribute to a poor surface finishing. Generally, the build direction is not constant for every deposited section or layer in complex geometry parts. As a result, there is an additional concern to ensure the build direction is aligned with gravity, thus improving the quality of the final part. This paper proposes an algorithm to control the torch motion with respect to a deposition substrate as well as the torch orientation with respect to an inertial frame. The control scheme is based on task augmentation applied to an extended kinematic chain composed by two robots, which constitutes a coordinated control problem, and allows the deposition trajectory to be planned with respect to the deposition substrate coordinate frame while aligning each layer buildup direction with gravity (or any other direction defined for an inertial frame). Parts with complex geometry aspects have been produced in a WAAM cell composed by two robots (a manipulator with a welding torch and a positioning table holding the workpiece) in order to validate the proposed approach.
>
---
