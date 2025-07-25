# 机器人 cs.RO

- **最新发布 33 篇**

- **更新 24 篇**

## 最新发布

#### [new 001] Feature Geometry for Stereo Sidescan and Forward-looking Sonar
- **分类: cs.RO**

- **简介: 该论文属于多传感器数据融合任务，旨在解决声呐图像间特征匹配问题。通过几何方法实现前视与侧扫声呐间的特征投影，提升3D信息恢复能力。**

- **链接: [http://arxiv.org/pdf/2507.05410v1](http://arxiv.org/pdf/2507.05410v1)**

> **作者:** Kalin Norman; Joshua G. Mangelson
>
> **备注:** This is a submission to a workshop and was presented at the Workshop on Field Robotics, which was a part of ICRA 2025
>
> **摘要:** In this paper, we address stereo acoustic data fusion for marine robotics and propose a geometry-based method for projecting observed features from one sonar to another for a cross-modal stereo sonar setup that consists of both a forward-looking and a sidescan sonar. Our acoustic geometry for sidescan and forward-looking sonar is inspired by the epipolar geometry for stereo cameras, and we leverage relative pose information to project where an observed feature in one sonar image will be found in the image of another sonar. Additionally, we analyze how both the feature location relative to the sonar and the relative pose between the two sonars impact the projection. From simulated results, we identify desirable stereo configurations for applications in field robotics like feature correspondence and recovery of the 3D information of the feature.
>
---
#### [new 002] Learning-Augmented Model-Based Multi-Robot Planning for Time-Critical Search and Inspection Under Uncertainty
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人协同搜索任务，解决在不确定环境下高效识别高优先级区域的问题。通过结合图神经网络与模型规划，提升搜索效率。**

- **链接: [http://arxiv.org/pdf/2507.06129v1](http://arxiv.org/pdf/2507.06129v1)**

> **作者:** Abhish Khanal; Joseph Prince Mathew; Cameron Nowzari; Gregory J. Stein
>
> **备注:** 7 pages, 6 figures, CASE 2025
>
> **摘要:** In disaster response or surveillance operations, quickly identifying areas needing urgent attention is critical, but deploying response teams to every location is inefficient or often impossible. Effective performance in this domain requires coordinating a multi-robot inspection team to prioritize inspecting locations more likely to need immediate response, while also minimizing travel time. This is particularly challenging because robots must directly observe the locations to determine which ones require additional attention. This work introduces a multi-robot planning framework for coordinated time-critical multi-robot search under uncertainty. Our approach uses a graph neural network to estimate the likelihood of PoIs needing attention from noisy sensor data and then uses those predictions to guide a multi-robot model-based planner to determine the cost-effective plan. Simulated experiments demonstrate that our planner improves performance at least by 16.3\%, 26.7\%, and 26.2\% for 1, 3, and 5 robots, respectively, compared to non-learned and learned baselines. We also validate our approach on real-world platforms using quad-copters.
>
---
#### [new 003] Fast Bilateral Teleoperation and Imitation Learning Using Sensorless Force Control via Accurate Dynamics Model
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文属于机器人 teleoperation 任务，解决低成本机械臂缺乏力反馈的问题。通过四通道双边控制和精确动力学模型，实现高速力反馈操作与模仿学习。**

- **链接: [http://arxiv.org/pdf/2507.06174v1](http://arxiv.org/pdf/2507.06174v1)**

> **作者:** Koki Yamane; Yunhan Li; Masashi Konosu; Koki Inami; Junji Oaki; Sho Sakaino; Toshiaki Tsuji
>
> **备注:** 19 pages, 8 figures, Submitted to CoRL 2025
>
> **摘要:** In recent years, the advancement of imitation learning has led to increased interest in teleoperating low-cost manipulators to collect demonstration data. However, most existing systems rely on unilateral control, which only transmits target position values. While this approach is easy to implement and suitable for slow, non-contact tasks, it struggles with fast or contact-rich operations due to the absence of force feedback. This work demonstrates that fast teleoperation with force feedback is feasible even with force-sensorless, low-cost manipulators by leveraging 4-channel bilateral control. Based on accurately identified manipulator dynamics, our method integrates nonlinear terms compensation, velocity and external force estimation, and variable gain corresponding to inertial variation. Furthermore, using data collected by 4-channel bilateral control, we show that incorporating force information into both the input and output of learned policies improves performance in imitation learning. These results highlight the practical effectiveness of our system for high-fidelity teleoperation and data collection on affordable hardware.
>
---
#### [new 004] Fast and Accurate Collision Probability Estimation for Autonomous Vehicles using Adaptive Sigma-Point Sampling
- **分类: cs.RO; cs.AI; cs.CG**

- **简介: 该论文属于自动驾驶任务，解决动态物体碰撞概率估计问题。提出自适应sigma点采样算法，提高计算速度与准确性。**

- **链接: [http://arxiv.org/pdf/2507.06149v1](http://arxiv.org/pdf/2507.06149v1)**

> **作者:** Charles Champagne Cossette; Taylor Scott Clawson; Andrew Feit
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** A novel algorithm is presented for the estimation of collision probabilities between dynamic objects with uncertain trajectories, where the trajectories are given as a sequence of poses with Gaussian distributions. We propose an adaptive sigma-point sampling scheme, which ultimately produces a fast, simple algorithm capable of estimating the collision probability with a median error of 3.5%, and a median runtime of 0.21ms, when measured on an Intel Xeon Gold 6226R Processor. Importantly, the algorithm explicitly accounts for the collision probability's temporal dependence, which is often neglected in prior work and otherwise leads to an overestimation of the collision probability. Finally, the method is tested on a diverse set of relevant real-world scenarios, consisting of 400 6-second snippets of autonomous vehicle logs, where the accuracy and latency is rigorously evaluated.
>
---
#### [new 005] Robust Speech-Workload Estimation for Intelligent Human-Robot Systems
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于人机交互任务，旨在解决实时估计语音负荷的问题，通过提出算法提升智能人机系统的适应性。**

- **链接: [http://arxiv.org/pdf/2507.05985v1](http://arxiv.org/pdf/2507.05985v1)**

> **作者:** Julian Fortune; Julie A. Adams; Jamison Heard
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Demanding task environments (e.g., supervising a remotely piloted aircraft) require performing tasks quickly and accurately; however, periods of low and high operator workload can decrease task performance. Intelligent modulation of the system's demands and interaction modality in response to changes in operator workload state may increase performance by avoiding undesirable workload states. This system requires real-time estimation of each workload component (i.e., cognitive, physical, visual, speech, and auditory) to adapt the correct modality. Existing workload systems estimate multiple workload components post-hoc, but few estimate speech workload, or function in real-time. An algorithm to estimate speech workload and mitigate undesirable workload states in real-time is presented. An analysis of the algorithm's accuracy is presented, along with the results demonstrating the algorithm's generalizability across individuals and human-machine teaming paradigms. Real-time speech workload estimation is a crucial element towards developing adaptive human-machine systems.
>
---
#### [new 006] EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Embodiment-Centric Flow
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决无动作标签视频中学习灵活操作的问题。提出EC-Flow框架，通过具身流预测实现通用操作，提升对柔性物体和遮挡场景的处理能力。**

- **链接: [http://arxiv.org/pdf/2507.06224v1](http://arxiv.org/pdf/2507.06224v1)**

> **作者:** Yixiang Chen; Peiyan Li; Yan Huang; Jiabing Yang; Kehan Chen; Liang Wang
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Current language-guided robotic manipulation systems often require low-level action-labeled datasets for imitation learning. While object-centric flow prediction methods mitigate this issue, they remain limited to scenarios involving rigid objects with clear displacement and minimal occlusion. In this work, we present Embodiment-Centric Flow (EC-Flow), a framework that directly learns manipulation from action-unlabeled videos by predicting embodiment-centric flow. Our key insight is that incorporating the embodiment's inherent kinematics significantly enhances generalization to versatile manipulation scenarios, including deformable object handling, occlusions, and non-object-displacement tasks. To connect the EC-Flow with language instructions and object interactions, we further introduce a goal-alignment module by jointly optimizing movement consistency and goal-image prediction. Moreover, translating EC-Flow to executable robot actions only requires a standard robot URDF (Unified Robot Description Format) file to specify kinematic constraints across joints, which makes it easy to use in practice. We validate EC-Flow on both simulation (Meta-World) and real-world tasks, demonstrating its state-of-the-art performance in occluded object handling (62% improvement), deformable object manipulation (45% improvement), and non-object-displacement tasks (80% improvement) than prior state-of-the-art object-centric flow methods. For more information, see our project website at https://ec-flow1.github.io .
>
---
#### [new 007] Structured Task Solving via Modular Embodied Intelligence: A Case Study on Rubik's Cube
- **分类: cs.RO**

- **简介: 该论文研究机器人解决魔方复原任务，提出Auto-RubikAI框架，结合知识库、视觉语言模型和大语言模型，实现高效、可解释的自主操作。**

- **链接: [http://arxiv.org/pdf/2507.05607v1](http://arxiv.org/pdf/2507.05607v1)**

> **作者:** Chongshan Fan; Shenghai Yuan
>
> **摘要:** This paper presents Auto-RubikAI, a modular autonomous planning framework that integrates a symbolic Knowledge Base (KB), a vision-language model (VLM), and a large language model (LLM) to solve structured manipulation tasks exemplified by Rubik's Cube restoration. Unlike traditional robot systems based on predefined scripts, or modern approaches relying on pretrained networks and large-scale demonstration data, Auto-RubikAI enables interpretable, multi-step task execution with minimal data requirements and no prior demonstrations. The proposed system employs a KB module to solve group-theoretic restoration steps, overcoming LLMs' limitations in symbolic reasoning. A VLM parses RGB-D input to construct a semantic 3D scene representation, while the LLM generates structured robotic control code via prompt chaining. This tri-module architecture enables robust performance under spatial uncertainty. We deploy Auto-RubikAI in both simulation and real-world settings using a 7-DOF robotic arm, demonstrating effective Sim-to-Real adaptation without retraining. Experiments show a 79% end-to-end task success rate across randomized configurations. Compared to CFOP, DeepCubeA, and Two-Phase baselines, our KB-enhanced method reduces average solution steps while maintaining interpretability and safety. Auto-RubikAI provides a cost-efficient, modular foundation for embodied task planning in smart manufacturing, robotics education, and autonomous execution scenarios. Code, prompts, and hardware modules will be released upon publication.
>
---
#### [new 008] Hybrid Diffusion Policies with Projective Geometric Algebra for Efficient Robot Manipulation Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人操作学习任务，旨在解决模型训练效率低的问题。通过引入PGA框架，提升空间表示和操作的几何归纳偏差，提高训练效率与性能。**

- **链接: [http://arxiv.org/pdf/2507.05695v1](http://arxiv.org/pdf/2507.05695v1)**

> **作者:** Xiatao Sun; Yuxuan Wang; Shuo Yang; Yinxing Chen; Daniel Rakita
>
> **摘要:** Diffusion policies have become increasingly popular in robot learning due to their reliable convergence in motion generation tasks. At a high level, these policies learn to transform noisy action trajectories into effective ones, conditioned on observations. However, each time such a model is trained in a robotics context, the network must relearn fundamental spatial representations and operations, such as translations and rotations, from scratch in order to ground itself and operate effectively in a 3D environment. Incorporating geometric inductive biases directly into the network can alleviate this redundancy and substantially improve training efficiency. In this paper, we introduce hPGA-DP, a diffusion policy approach that integrates a mathematical framework called Projective Geometric Algebra (PGA) to embed strong geometric inductive biases. PGA is particularly well-suited for this purpose as it provides a unified algebraic framework that naturally encodes geometric primitives, such as points, directions, and rotations, enabling neural networks to reason about spatial structure through interpretable and composable operations. Specifically, we propose a novel diffusion policy architecture that incorporates the Projective Geometric Algebra Transformer (P-GATr), leveraging its E(3)-equivariant properties established in prior work. Our approach adopts a hybrid architecture strategy, using P-GATr as both a state encoder and action decoder, while employing U-Net or Transformer-based modules for the denoising process. Several experiments and ablation studies in both simulated and real-world environments demonstrate that hPGA-DP not only improves task performance and training efficiency through the geometric bias of P-GATr, but also achieves substantially faster convergence through its hybrid model compared to architectures that rely solely on P-GATr.
>
---
#### [new 009] Communication-Efficient Module-Wise Federated Learning for Grasp Pose Detection in Cluttered Environments
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人抓取姿态检测任务，解决FL中通信开销大的问题。通过模块化分析，优化通信效率，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.05861v1](http://arxiv.org/pdf/2507.05861v1)**

> **作者:** Woonsang Kang; Joohyung Lee; Seungjun Kim; Jungchan Cho; Yoonseon Oh
>
> **备注:** 8 pages, 5 figures. Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Grasp pose detection (GPD) is a fundamental capability for robotic autonomy, but its reliance on large, diverse datasets creates significant data privacy and centralization challenges. Federated Learning (FL) offers a privacy-preserving solution, but its application to GPD is hindered by the substantial communication overhead of large models, a key issue for resource-constrained robots. To address this, we propose a novel module-wise FL framework that begins by analyzing the learning dynamics of the GPD model's functional components. This analysis identifies slower-converging modules, to which our framework then allocates additional communication effort. This is realized through a two-phase process: a standard full-model training phase is followed by a communication-efficient phase where only the identified subset of slower-converging modules is trained and their partial updates are aggregated. Extensive experiments on the GraspNet-1B dataset demonstrate that our method outperforms standard FedAvg and other baselines, achieving higher accuracy for a given communication budget. Furthermore, real-world experiments on a physical robot validate our approach, showing a superior grasp success rate compared to baseline methods in cluttered scenes. Our work presents a communication-efficient framework for training robust, generalized GPD models in a decentralized manner, effectively improving the trade-off between communication cost and model performance.
>
---
#### [new 010] Stable Tracking-in-the-Loop Control of Cable-Driven Surgical Manipulators under Erroneous Kinematic Chains
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决电缆驱动手术机械臂在运动学链错误下的稳定控制问题，设计了闭环跟踪控制器并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.05663v1](http://arxiv.org/pdf/2507.05663v1)**

> **作者:** Neelay Joglekar; Fei Liu; Florian Richter; Michael C. Yip
>
> **备注:** 8 pages, 5 figures, Submitted to RAL
>
> **摘要:** Remote Center of Motion (RCM) robotic manipulators have revolutionized Minimally Invasive Surgery, enabling precise, dexterous surgical manipulation within the patient's body cavity without disturbing the insertion point on the patient. Accurate RCM tool control is vital for incorporating autonomous subtasks like suturing, blood suction, and tumor resection into robotic surgical procedures, reducing surgeon fatigue and improving patient outcomes. However, these cable-driven systems are subject to significant joint reading errors, corrupting the kinematics computation necessary to perform control. Although visual tracking with endoscopic cameras can correct errors on in-view joints, errors in the kinematic chain prior to the insertion point are irreparable because they remain out of view. No prior work has characterized the stability of control under these conditions. We fill this gap by designing a provably stable tracking-in-the-loop controller for the out-of-view portion of the RCM manipulator kinematic chain. We additionally incorporate this controller into a bilevel control scheme for the full kinematic chain. We rigorously benchmark our method in simulated and real world settings to verify our theoretical findings. Our work provides key insights into the next steps required for the transition from teleoperated to autonomous surgery.
>
---
#### [new 011] Is Diversity All You Need for Scalable Robotic Manipulation?
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究机器人操作数据集的扩展问题，探讨数据多样性对模型性能的影响，提出改进方法提升迁移效果。**

- **链接: [http://arxiv.org/pdf/2507.06219v1](http://arxiv.org/pdf/2507.06219v1)**

> **作者:** Modi Shi; Li Chen; Jin Chen; Yuxiang Lu; Chiming Liu; Guanghui Ren; Ping Luo; Di Huang; Maoqing Yao; Hongyang Li
>
> **备注:** Code is available at https://github.com/OpenDriveLab/AgiBot-World
>
> **摘要:** Data scaling has driven remarkable success in foundation models for Natural Language Processing (NLP) and Computer Vision (CV), yet the principles of effective data scaling in robotic manipulation remain insufficiently understood. In this work, we investigate the nuanced role of data diversity in robot learning by examining three critical dimensions-task (what to do), embodiment (which robot to use), and expert (who demonstrates)-challenging the conventional intuition of "more diverse is better". Throughout extensive experiments on various robot platforms, we reveal that (1) task diversity proves more critical than per-task demonstration quantity, benefiting transfer from diverse pre-training tasks to novel downstream scenarios; (2) multi-embodiment pre-training data is optional for cross-embodiment transfer-models trained on high-quality single-embodiment data can efficiently transfer to different platforms, showing more desirable scaling property during fine-tuning than multi-embodiment pre-trained models; and (3) expert diversity, arising from individual operational preferences and stochastic variations in human demonstrations, can be confounding to policy learning, with velocity multimodality emerging as a key contributing factor. Based on this insight, we propose a distribution debiasing method to mitigate velocity ambiguity, the yielding GO-1-Pro achieves substantial performance gains of 15%, equivalent to using 2.5 times pre-training data. Collectively, these findings provide new perspectives and offer practical guidance on how to scale robotic manipulation datasets effectively.
>
---
#### [new 012] LeAD: The LLM Enhanced Planning System Converged with End-to-end Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决复杂场景下决策失误问题。通过融合E2E学习与LLM，提升场景理解和决策能力。**

- **链接: [http://arxiv.org/pdf/2507.05754v1](http://arxiv.org/pdf/2507.05754v1)**

> **作者:** Yuhang Zhang; Jiaqi Liu; Chengkai Xu; Peng Hang; Jian Sun
>
> **摘要:** A principal barrier to large-scale deployment of urban autonomous driving systems lies in the prevalence of complex scenarios and edge cases. Existing systems fail to effectively interpret semantic information within traffic contexts and discern intentions of other participants, consequently generating decisions misaligned with skilled drivers' reasoning patterns. We present LeAD, a dual-rate autonomous driving architecture integrating imitation learning-based end-to-end (E2E) frameworks with large language model (LLM) augmentation. The high-frequency E2E subsystem maintains real-time perception-planning-control cycles, while the low-frequency LLM module enhances scenario comprehension through multi-modal perception fusion with HD maps and derives optimal decisions via chain-of-thought (CoT) reasoning when baseline planners encounter capability limitations. Our experimental evaluation in the CARLA Simulator demonstrates LeAD's superior handling of unconventional scenarios, achieving 71 points on Leaderboard V1 benchmark, with a route completion of 93%.
>
---
#### [new 013] FineGrasp: Towards Robust Grasping for Delicate Objects
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决小而易碎物体抓取困难的问题。通过改进网络结构、优化标签策略和引入新数据集，提升抓取性能。**

- **链接: [http://arxiv.org/pdf/2507.05978v1](http://arxiv.org/pdf/2507.05978v1)**

> **作者:** Yun Du; Mengao Zhao; Tianwei Lin; Yiwei Jin; Chaodong Huang; Zhizhong Su
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Recent advancements in robotic grasping have led to its integration as a core module in many manipulation systems. For instance, language-driven semantic segmentation enables the grasping of any designated object or object part. However, existing methods often struggle to generate feasible grasp poses for small objects or delicate components, potentially causing the entire pipeline to fail. To address this issue, we propose a novel grasping method, FineGrasp, which introduces improvements in three key aspects. First, we introduce multiple network modifications to enhance the ability of to handle delicate regions. Second, we address the issue of label imbalance and propose a refined graspness label normalization strategy. Third, we introduce a new simulated grasp dataset and show that mixed sim-to-real training further improves grasp performance. Experimental results show significant improvements, especially in grasping small objects, and confirm the effectiveness of our system in semantic grasping.
>
---
#### [new 014] CRED: Counterfactual Reasoning and Environment Design for Active Preference Learning
- **分类: cs.RO**

- **简介: 该论文属于主动偏好学习任务，解决长时序任务中轨迹空间探索不足的问题。提出CRED方法，结合环境设计与反事实推理生成有效轨迹，提升奖励估计效果。**

- **链接: [http://arxiv.org/pdf/2507.05458v1](http://arxiv.org/pdf/2507.05458v1)**

> **作者:** Yi-Shiuan Tung; Bradley Hayes; Alessandro Roncone
>
> **摘要:** For effective real-world deployment, robots should adapt to human preferences, such as balancing distance, time, and safety in delivery routing. Active preference learning (APL) learns human reward functions by presenting trajectories for ranking. However, existing methods often struggle to explore the full trajectory space and fail to identify informative queries, particularly in long-horizon tasks. We propose CRED, a trajectory generation method for APL that improves reward estimation by jointly optimizing environment design and trajectory selection. CRED "imagines" new scenarios through environment design and uses counterfactual reasoning -- by sampling rewards from its current belief and asking "What if this reward were the true preference?" -- to generate a diverse and informative set of trajectories for ranking. Experiments in GridWorld and real-world navigation using OpenStreetMap data show that CRED improves reward learning and generalizes effectively across different environments.
>
---
#### [new 015] A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文研究多任务机器人操作，解决如何评估大型行为模型的问题。通过扩展扩散策略，验证多任务预训练提升性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.05331v1](http://arxiv.org/pdf/2507.05331v1)**

> **作者:** TRI LBM Team; Jose Barreiros; Andrew Beaulieu; Aditya Bhat; Rick Cory; Eric Cousineau; Hongkai Dai; Ching-Hsin Fang; Kunimatsu Hashimoto; Muhammad Zubair Irshad; Masha Itkina; Naveen Kuppuswamy; Kuan-Hui Lee; Katherine Liu; Dale McConachie; Ian McMahon; Haruki Nishimura; Calder Phillips-Grafflin; Charles Richter; Paarth Shah; Krishnan Srinivasan; Blake Wulfe; Chen Xu; Mengchao Zhang; Alex Alspach; Maya Angeles; Kushal Arora; Vitor Campagnolo Guizilini; Alejandro Castro; Dian Chen; Ting-Sheng Chu; Sam Creasey; Sean Curtis; Richard Denitto; Emma Dixon; Eric Dusel; Matthew Ferreira; Aimee Goncalves; Grant Gould; Damrong Guoy; Swati Gupta; Xuchen Han; Kyle Hatch; Brendan Hathaway; Allison Henry; Hillel Hochsztein; Phoebe Horgan; Shun Iwase; Donovon Jackson; Siddharth Karamcheti; Sedrick Keh; Joseph Masterjohn; Jean Mercat; Patrick Miller; Paul Mitiguy; Tony Nguyen; Jeremy Nimmer; Yuki Noguchi; Reko Ong; Aykut Onol; Owen Pfannenstiehl; Richard Poyner; Leticia Priebe Mendes Rocha; Gordon Richardson; Christopher Rodriguez; Derick Seale; Michael Sherman; Mariah Smith-Jones; David Tago; Pavel Tokmakov; Matthew Tran; Basile Van Hoorick; Igor Vasiljevic; Sergey Zakharov; Mark Zolotas; Rares Ambrus; Kerri Fetzer-Borelli; Benjamin Burchfiel; Hadas Kress-Gazit; Siyuan Feng; Stacie Ford; Russ Tedrake
>
> **摘要:** Robot manipulation has seen tremendous progress in recent years, with imitation learning policies enabling successful performance of dexterous and hard-to-model tasks. Concurrently, scaling data and model size has led to the development of capable language and vision foundation models, motivating large-scale efforts to create general-purpose robot foundation models. While these models have garnered significant enthusiasm and investment, meaningful evaluation of real-world performance remains a challenge, limiting both the pace of development and inhibiting a nuanced understanding of current capabilities. In this paper, we rigorously evaluate multitask robot manipulation policies, referred to as Large Behavior Models (LBMs), by extending the Diffusion Policy paradigm across a corpus of simulated and real-world robot data. We propose and validate an evaluation pipeline to rigorously analyze the capabilities of these models with statistical confidence. We compare against single-task baselines through blind, randomized trials in a controlled setting, using both simulation and real-world experiments. We find that multi-task pretraining makes the policies more successful and robust, and enables teaching complex new tasks more quickly, using a fraction of the data when compared to single-task baselines. Moreover, performance predictably increases as pretraining scale and diversity grows. Project page: https://toyotaresearchinstitute.github.io/lbm1/
>
---
#### [new 016] PAPRLE (Plug-And-Play Robotic Limb Environment): A Modular Ecosystem for Robotic Limbs
- **分类: cs.RO**

- **简介: 该论文提出PAPRLE，一个模块化机器人肢体环境，解决多机器人灵活控制与配置问题。通过支持多种输入设备和控制方式，提升 teleoperation 的适应性与交互性。**

- **链接: [http://arxiv.org/pdf/2507.05555v1](http://arxiv.org/pdf/2507.05555v1)**

> **作者:** Obin Kwon; Sankalp Yamsani; Noboru Myers; Sean Taylor; Jooyoung Hong; Kyungseo Park; Alex Alspach; Joohyung Kim
>
> **摘要:** We introduce PAPRLE (Plug-And-Play Robotic Limb Environment), a modular ecosystem that enables flexible placement and control of robotic limbs. With PAPRLE, a user can change the arrangement of the robotic limbs, and control them using a variety of input devices, including puppeteers, gaming controllers, and VR-based interfaces. This versatility supports a wide range of teleoperation scenarios and promotes adaptability to different task requirements. To further enhance configurability, we introduce a pluggable puppeteer device that can be easily mounted and adapted to match the target robot configurations. PAPRLE supports bilateral teleoperation through these puppeteer devices, agnostic to the type or configuration of the follower robot. By supporting both joint-space and task-space control, the system provides real-time force feedback, improving user fidelity and physical interaction awareness. The modular design of PAPRLE facilitates novel spatial arrangements of the limbs and enables scalable data collection, thereby advancing research in embodied AI and learning-based control. We validate PAPRLE in various real-world settings, demonstrating its versatility across diverse combinations of leader devices and follower robots. The system will be released as open source, including both hardware and software components, to support broader adoption and community-driven extension. Additional resources and demonstrations are available at the project website: https://uiuckimlab.github.io/paprle-pages
>
---
#### [new 017] Simultaneous Triggering and Synchronization of Sensors and Onboard Computers
- **分类: cs.RO**

- **简介: 该论文属于机器人实时数据同步任务，解决传感器与计算机时间戳不准确问题，提出一种低成本同步系统以提高在线估计精度。**

- **链接: [http://arxiv.org/pdf/2507.05717v1](http://arxiv.org/pdf/2507.05717v1)**

> **作者:** Morten Nissov; Nikhil Khedekar; Kostas Alexis
>
> **备注:** 2 pages, 3 figures. Presented at ICRA@40
>
> **摘要:** High fidelity estimation algorithms for robotics require accurate data. However, timestamping of sensor data is a key issue that rarely receives the attention it deserves. Inaccurate timestamping can be compensated for in post-processing but is imperative for online estimation. Simultaneously, even online mitigation of timing issues can be achieved through a relaxation of the tuning parameters from their otherwise more performative optimal values, but at a detriment to performance. To address the need for real-time, low-cost timestamping, a versatile system which utilizes readily-available components and established methods for synchronization is introduced. The synchronization and triggering (of both high- and low-rate sensors) capabilities of the system are demonstrated.
>
---
#### [new 018] 3DGS_LSR:Large_Scale Relocation for Autonomous Driving Based on 3D Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主驾驶中的定位任务，解决复杂环境中GNSS失效导致的定位问题。通过3D高斯泼溅技术，实现单目图像的厘米级精确定位。**

- **链接: [http://arxiv.org/pdf/2507.05661v1](http://arxiv.org/pdf/2507.05661v1)**

> **作者:** Haitao Lu; Haijier Chen; Haoze Liu; Shoujian Zhang; Bo Xu; Ziao Liu
>
> **备注:** 13 pages,7 figures,4 tables
>
> **摘要:** In autonomous robotic systems, precise localization is a prerequisite for safe navigation. However, in complex urban environments, GNSS positioning often suffers from signal occlusion and multipath effects, leading to unreliable absolute positioning. Traditional mapping approaches are constrained by storage requirements and computational inefficiency, limiting their applicability to resource-constrained robotic platforms. To address these challenges, we propose 3DGS-LSR: a large-scale relocalization framework leveraging 3D Gaussian Splatting (3DGS), enabling centimeter-level positioning using only a single monocular RGB image on the client side. We combine multi-sensor data to construct high-accuracy 3DGS maps in large outdoor scenes, while the robot-side localization requires just a standard camera input. Using SuperPoint and SuperGlue for feature extraction and matching, our core innovation is an iterative optimization strategy that refines localization results through step-by-step rendering, making it suitable for real-time autonomous navigation. Experimental validation on the KITTI dataset demonstrates our 3DGS-LSR achieves average positioning accuracies of 0.026m, 0.029m, and 0.081m in town roads, boulevard roads, and traffic-dense highways respectively, significantly outperforming other representative methods while requiring only monocular RGB input. This approach provides autonomous robots with reliable localization capabilities even in challenging urban environments where GNSS fails.
>
---
#### [new 019] Integrating Diffusion-based Multi-task Learning with Online Reinforcement Learning for Robust Quadruped Robot Control
- **分类: cs.RO**

- **简介: 该论文属于四足机器人控制任务，解决语言引导和任务切换稳定性问题。结合扩散模型与在线强化学习，提出DMLoco框架，实现高效、鲁棒的控制。**

- **链接: [http://arxiv.org/pdf/2507.05674v1](http://arxiv.org/pdf/2507.05674v1)**

> **作者:** Xinyao Qin; Xiaoteng Ma; Yang Qi; Qihan Liu; Chuanyi Xue; Ning Gui; Qinyu Dong; Jun Yang; Bin Liang
>
> **摘要:** Recent research has highlighted the powerful capabilities of imitation learning in robotics. Leveraging generative models, particularly diffusion models, these approaches offer notable advantages such as strong multi-task generalization, effective language conditioning, and high sample efficiency. While their application has been successful in manipulation tasks, their use in legged locomotion remains relatively underexplored, mainly due to compounding errors that affect stability and difficulties in task transition under limited data. Online reinforcement learning (RL) has demonstrated promising results in legged robot control in the past years, providing valuable insights to address these challenges. In this work, we propose DMLoco, a diffusion-based framework for quadruped robots that integrates multi-task pretraining with online PPO finetuning to enable language-conditioned control and robust task transitions. Our approach first pretrains the policy on a diverse multi-task dataset using diffusion models, enabling language-guided execution of various skills. Then, it finetunes the policy in simulation to ensure robustness and stable task transition during real-world deployment. By utilizing Denoising Diffusion Implicit Models (DDIM) for efficient sampling and TensorRT for optimized deployment, our policy runs onboard at 50Hz, offering a scalable and efficient solution for adaptive, language-guided locomotion on resource-constrained robotic platforms.
>
---
#### [new 020] A Learning-based Planning and Control Framework for Inertia Drift Vehicles
- **分类: cs.RO**

- **简介: 该论文属于自主赛车控制任务，解决惯性漂移中快速转向与路径跟踪问题，提出基于贝叶斯优化的学习框架以提升控制性能。**

- **链接: [http://arxiv.org/pdf/2507.05748v1](http://arxiv.org/pdf/2507.05748v1)**

> **作者:** Bei Zhou; Zhouheng Li; Lei Xie; Hongye Su; Johannes Betz
>
> **摘要:** Inertia drift is a transitional maneuver between two sustained drift stages in opposite directions, which provides valuable insights for navigating consecutive sharp corners for autonomous racing.However, this can be a challenging scenario for the drift controller to handle rapid transitions between opposing sideslip angles while maintaining accurate path tracking. Moreover, accurate drift control depends on a high-fidelity vehicle model to derive drift equilibrium points and predict vehicle states, but this is often compromised by the strongly coupled longitudinal-lateral drift dynamics and unpredictable environmental variations. To address these challenges, this paper proposes a learning-based planning and control framework utilizing Bayesian optimization (BO), which develops a planning logic to ensure a smooth transition and minimal velocity loss between inertia and sustained drift phases. BO is further employed to learn a performance-driven control policy that mitigates modeling errors for enhanced system performance. Simulation results on an 8-shape reference path demonstrate that the proposed framework can achieve smooth and stable inertia drift through sharp corners.
>
---
#### [new 021] A Physics-Based Continuum Model for Versatile, Scalable, and Fast Terramechanics Simulation
- **分类: cs.RO**

- **简介: 该论文属于地面力学仿真任务，旨在解决传统半经验方法的局限性。提出基于物理的Chrono::CRM模型，实现高效、大规模的地形交互模拟。**

- **链接: [http://arxiv.org/pdf/2507.05643v1](http://arxiv.org/pdf/2507.05643v1)**

> **作者:** Huzaifa Unjhawala; Luning Bakke; Harry Zhang; Michael Taylor; Ganesh Arivoli; Radu Serban; Dan Negrut
>
> **备注:** 32 pages, 21 figures, Submitted to Journal of Terramechanics
>
> **摘要:** This paper discusses Chrono's Continuous Representation Model (called herein Chrono::CRM), a general-purpose, scalable, and efficient simulation solution for terramechanics problems. Built on Chrono's Smoothed Particle Hydrodynamics (SPH) framework, Chrono::CRM moves beyond semi-empirical terramechanics approaches, e.g., Bekker-Wong/Janosi-Hanamoto, to provide a physics-based model able to address complex tasks such as digging, grading, as well as interaction with deformable wheels and complex grouser/lug patterns. The terramechanics model is versatile in that it allows the terrain to interact with both rigid and flexible implements simulated via the Chrono dynamics engine. We validate Chrono::CRM against experimental data from three physical tests, including one involving NASA's MGRU3 rover. In addition, the simulator is benchmarked against a high-fidelity Discrete Element Method (DEM) simulation of a digging scenario involving the Regolith Advanced Surface Systems Operations Robot (RASSOR). Being GPU-accelerated, Chrono::CRM achieves computational efficiency comparable to that of semi-empirical simulation approaches for terramechanics problems. Through an ``active domains'' implementation, Chrono::CRM can handle terrain stretches up to 10 km long with 100 million SPH particles at near interactive rates, making high-fidelity off-road simulations at large scales feasible. As a component of the Chrono package, the CRM model is open source and released under a BSD-3 license. All models and simulations used in this contribution are available in a public GitHub repository for reproducibility studies and further research.
>
---
#### [new 022] AURA-CVC: Autonomous Ultrasound-guided Robotic Assistance for Central Venous Catheterization
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决CVC操作中的精准定位与自主化问题。通过深度学习和机器人技术实现自动扫描、血管重建与针插入，提升操作安全性与成功率。**

- **链接: [http://arxiv.org/pdf/2507.05979v1](http://arxiv.org/pdf/2507.05979v1)**

> **作者:** Deepak Raina; Lidia Al-Zogbi; Brian Teixeira; Vivek Singh; Ankur Kapoor; Thorsten Fleiter; Muyinatu A. Lediju Bell; Vinciya Pandian; Axel Krieger
>
> **摘要:** Purpose: Central venous catheterization (CVC) is a critical medical procedure for vascular access, hemodynamic monitoring, and life-saving interventions. Its success remains challenging due to the need for continuous ultrasound-guided visualization of a target vessel and approaching needle, which is further complicated by anatomical variability and operator dependency. Errors in needle placement can lead to life-threatening complications. While robotic systems offer a potential solution, achieving full autonomy remains challenging. In this work, we propose an end-to-end robotic-ultrasound-guided CVC pipeline, from scan initialization to needle insertion. Methods: We introduce a deep-learning model to identify clinically relevant anatomical landmarks from a depth image of the patient's neck, obtained using RGB-D camera, to autonomously define the scanning region and paths. Then, a robot motion planning framework is proposed to scan, segment, reconstruct, and localize vessels (veins and arteries), followed by the identification of the optimal insertion zone. Finally, a needle guidance module plans the insertion under ultrasound guidance with operator's feedback. This pipeline was validated on a high-fidelity commercial phantom across 10 simulated clinical scenarios. Results: The proposed pipeline achieved 10 out of 10 successful needle placements on the first attempt. Vessels were reconstructed with a mean error of 2.15 \textit{mm}, and autonomous needle insertion was performed with an error less than or close to 1 \textit{mm}. Conclusion: To our knowledge, this is the first robotic CVC system demonstrated on a high-fidelity phantom with integrated planning, scanning, and insertion. Experimental results show its potential for clinical translation.
>
---
#### [new 023] Gaussian Process-Based Active Exploration Strategies in Vision and Touch
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，解决物体属性识别问题。通过融合视觉与触觉信息，构建GPDF模型，实现物体几何和表面属性的主动探索与建模。**

- **链接: [http://arxiv.org/pdf/2507.05522v1](http://arxiv.org/pdf/2507.05522v1)**

> **作者:** Ho Jin Choi; Nadia Figueroa
>
> **备注:** Master's Thesis, Mechanical Engineering and Applied Mechanics, University of Pennsylvania - April 2024 (https://events.seas.upenn.edu/event/meam-masters-thesis-defense-gaussian-process-based-active-exploration-strategies-in-vision-and-touch/) (https://blog.me.upenn.edu/ho-jin-choi-successfully-defends-masters-thesis/)
>
> **摘要:** Robots struggle to understand object properties like shape, material, and semantics due to limited prior knowledge, hindering manipulation in unstructured environments. In contrast, humans learn these properties through interactive multi-sensor exploration. This work proposes fusing visual and tactile observations into a unified Gaussian Process Distance Field (GPDF) representation for active perception of object properties. While primarily focusing on geometry, this approach also demonstrates potential for modeling surface properties beyond geometry. The GPDF encodes signed distance using point cloud, analytic gradient and Hessian, and surface uncertainty estimates, which are attributes that common neural network shape representation lack. By utilizing a point cloud to construct a distance function, GPDF does not need extensive pretraining on large datasets and can incorporate observations by aggregation. Starting with an initial visual shape estimate, the framework iteratively refines the geometry by integrating dense vision measurements using differentiable rendering and tactile measurements at uncertain surface regions. By quantifying multi-sensor uncertainties, it plans exploratory motions to maximize information gain for recovering precise 3D structures. For the real-world robot experiment, we utilize the Franka Research 3 robot manipulator, which is fixed on a table and has a customized DIGIT tactile sensor and an Intel Realsense D435 RGBD camera mounted on the end-effector. In these experiments, the robot explores the shape and properties of objects assumed to be static and placed on the table. To improve scalability, we investigate approximation methods like inducing point method for Gaussian Processes. This probabilistic multi-modal fusion enables active exploration and mapping of complex object geometries, extending potentially beyond geometry.
>
---
#### [new 024] DreamGrasp: Zero-Shot 3D Multi-Object Reconstruction from Partial-View Images for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D多物体重建任务，解决从部分视角图像中准确恢复物体几何的问题。通过结合生成模型和对比学习，实现复杂场景下的鲁棒重建。**

- **链接: [http://arxiv.org/pdf/2507.05627v1](http://arxiv.org/pdf/2507.05627v1)**

> **作者:** Young Hun Kim; Seungyeon Kim; Yonghyeon Lee; Frank Chongwoo Park
>
> **摘要:** Partial-view 3D recognition -- reconstructing 3D geometry and identifying object instances from a few sparse RGB images -- is an exceptionally challenging yet practically essential task, particularly in cluttered, occluded real-world settings where full-view or reliable depth data are often unavailable. Existing methods, whether based on strong symmetry priors or supervised learning on curated datasets, fail to generalize to such scenarios. In this work, we introduce DreamGrasp, a framework that leverages the imagination capability of large-scale pre-trained image generative models to infer the unobserved parts of a scene. By combining coarse 3D reconstruction, instance segmentation via contrastive learning, and text-guided instance-wise refinement, DreamGrasp circumvents limitations of prior methods and enables robust 3D reconstruction in complex, multi-object environments. Our experiments show that DreamGrasp not only recovers accurate object geometry but also supports downstream tasks like sequential decluttering and target retrieval with high success rates.
>
---
#### [new 025] Learning Agile Tensile Perching for Aerial Robots from Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于空中机器人自主抓取任务，解决 tethered tensile perching 的轨迹规划问题。通过强化学习方法实现精准控制与可靠锚定。**

- **链接: [http://arxiv.org/pdf/2507.06172v1](http://arxiv.org/pdf/2507.06172v1)**

> **作者:** Kangle Yuan; Atar Babgei; Luca Romanello; Hai-Nguyen Nguyen; Ronald Clark; Mirko Kovac; Sophie F. Armanini; Basaran Bahadir Kocer
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Perching on structures such as trees, beams, and ledges is essential for extending the endurance of aerial robots by enabling energy conservation in standby or observation modes. A tethered tensile perching mechanism offers a simple, adaptable solution that can be retrofitted to existing robots and accommodates a variety of structure sizes and shapes. However, tethered tensile perching introduces significant modelling challenges which require precise management of aerial robot dynamics, including the cases of tether slack & tension, and momentum transfer. Achieving smooth wrapping and secure anchoring by targeting a specific tether segment adds further complexity. In this work, we present a novel trajectory framework for tethered tensile perching, utilizing reinforcement learning (RL) through the Soft Actor-Critic from Demonstrations (SACfD) algorithm. By incorporating both optimal and suboptimal demonstrations, our approach enhances training efficiency and responsiveness, achieving precise control over position and velocity. This framework enables the aerial robot to accurately target specific tether segments, facilitating reliable wrapping and secure anchoring. We validate our framework through extensive simulation and real-world experiments, and demonstrate effectiveness in achieving agile and reliable trajectory generation for tensile perching.
>
---
#### [new 026] Evaluation of Habitat Robotics using Large Language Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于机器人任务研究，旨在评估大语言模型在具身机器人环境中的表现，解决模型推理能力对协作任务的影响问题。工作包括在Meta PARTNER基准上测试多个模型。**

- **链接: [http://arxiv.org/pdf/2507.06157v1](http://arxiv.org/pdf/2507.06157v1)**

> **作者:** William Li; Lei Hamilton; Kaise Al-natour; Sanjeev Mohindra
>
> **备注:** 6 pages, IEEE HPEC submission
>
> **摘要:** This paper focuses on evaluating the effectiveness of Large Language Models at solving embodied robotic tasks using the Meta PARTNER benchmark. Meta PARTNR provides simplified environments and robotic interactions within randomized indoor kitchen scenes. Each randomized kitchen scene is given a task where two robotic agents cooperatively work together to solve the task. We evaluated multiple frontier models on Meta PARTNER environments. Our results indicate that reasoning models like OpenAI o3-mini outperform non-reasoning models like OpenAI GPT-4o and Llama 3 when operating in PARTNR's robotic embodied environments. o3-mini displayed outperform across centralized, decentralized, full observability, and partial observability configurations. This provides a promising avenue of research for embodied robotic development.
>
---
#### [new 027] SCCRUB: Surface Cleaning Compliant Robot Utilizing Bristles
- **分类: cs.RO**

- **简介: 该论文属于机器人清洁任务，解决软体机器人难以产生足够摩擦力 scrub 的问题。通过设计带有刷毛的软臂并训练神经网络，实现有效清洁。**

- **链接: [http://arxiv.org/pdf/2507.06053v1](http://arxiv.org/pdf/2507.06053v1)**

> **作者:** Jakub F. Kowalewski; Keeyon Hajjafar; Alyssa Ugent; Jeffrey Ian Lipton
>
> **摘要:** Scrubbing surfaces is a physically demanding and time-intensive task. Removing adhered contamination requires substantial friction generated through pressure and torque or high lateral forces. Rigid robotic manipulators, while capable of exerting these forces, are usually confined to structured environments isolated from humans due to safety risks. In contrast, soft robot arms can safely work around humans and adapt to environmental uncertainty, but typically struggle to transmit the continuous torques or lateral forces necessary for scrubbing. Here, we demonstrate a soft robotic arm scrubbing adhered residues using torque and pressure, a task traditionally challenging for soft robots. We train a neural network to learn the arm's inverse kinematics and elasticity, which enables open-loop force and position control. Using this learned model, the robot successfully scrubbed burnt food residue from a plate and sticky fruit preserve from a toilet seat, removing an average of 99.7% of contamination. This work demonstrates how soft robots, capable of exerting continuous torque, can effectively and safely scrub challenging contamination from surfaces.
>
---
#### [new 028] DRO-EDL-MPC: Evidential Deep Learning-Based Distributionally Robust Model Predictive Control for Safe Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决感知不确定性带来的安全问题。通过结合证据深度学习与模型预测控制，提升决策的安全性与可靠性。**

- **链接: [http://arxiv.org/pdf/2507.05710v1](http://arxiv.org/pdf/2507.05710v1)**

> **作者:** Hyeongchan Ham; Heejin Ahn
>
> **备注:** 11 pages; Project page can be found at https://dro-edl-mpc.github.io
>
> **摘要:** Safety is a critical concern in motion planning for autonomous vehicles. Modern autonomous vehicles rely on neural network-based perception, but making control decisions based on these inference results poses significant safety risks due to inherent uncertainties. To address this challenge, we present a distributionally robust optimization (DRO) framework that accounts for both aleatoric and epistemic perception uncertainties using evidential deep learning (EDL). Our approach introduces a novel ambiguity set formulation based on evidential distributions that dynamically adjusts the conservativeness according to perception confidence levels. We integrate this uncertainty-aware constraint into model predictive control (MPC), proposing the DRO-EDL-MPC algorithm with computational tractability for autonomous driving applications. Validation in the CARLA simulator demonstrates that our approach maintains efficiency under high perception confidence while enforcing conservative constraints under low confidence.
>
---
#### [new 029] Comparison of Path Planning Algorithms for Autonomous Vehicle Navigation Using Satellite and Airborne LiDAR Data
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主车辆路径规划任务，旨在比较不同算法在卫星和机载LiDAR数据上的表现，解决复杂地形下的导航问题。**

- **链接: [http://arxiv.org/pdf/2507.05884v1](http://arxiv.org/pdf/2507.05884v1)**

> **作者:** Chang Liu; Zhexiong Xue; Tamas Sziranyi
>
> **备注:** 6 pages, 3 figures, 67th International Symposium ELMAR-2025 15-17 September 2025 Zadar, Croatia
>
> **摘要:** Autonomous vehicle navigation in unstructured environments, such as forests and mountainous regions, presents significant challenges due to irregular terrain and complex road conditions. This work provides a comparative evaluation of mainstream and well-established path planning algorithms applied to weighted pixel-level road networks derived from high-resolution satellite imagery and airborne LiDAR data. For 2D road-map navigation, where the weights reflect road conditions and terrain difficulty, A*, Dijkstra, RRT*, and a Novel Improved Ant Colony Optimization Algorithm (NIACO) are tested on the DeepGlobe satellite dataset. For 3D road-map path planning, 3D A*, 3D Dijkstra, RRT-Connect, and NIACO are evaluated using the Hamilton airborne LiDAR dataset, which provides detailed elevation information. All algorithms are assessed under identical start and end point conditions, focusing on path cost, computation time, and memory consumption. Results demonstrate that Dijkstra consistently offers the most stable and efficient performance in both 2D and 3D scenarios, particularly when operating on dense, pixel-level geospatial road-maps. These findings highlight the reliability of Dijkstra-based planning for static terrain navigation and establish a foundation for future research on dynamic path planning under complex environmental constraints.
>
---
#### [new 030] Assessing Linear Control Strategies for Zero-Speed Fin Roll Damping
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **简介: 该论文属于船舶控制任务，解决低速或静止时的横摇稳定问题。通过线性控制策略优化零速鳍的阻尼效果，提升稳定性。**

- **链接: [http://arxiv.org/pdf/2507.05867v1](http://arxiv.org/pdf/2507.05867v1)**

> **作者:** Nikita Savin; Elena Ambrosovskaya; Dmitry Romaev; Anton Proskurnikov
>
> **摘要:** Roll stabilization is a critical aspect of ship motion control, particularly for vessels operating in low-speed or zero-speed conditions, where traditional hydrodynamic fins lose their effectiveness. In this paper, we consider a roll damping system, developed by Navis JSC, based on two actively controlled zero-speed fins. Unlike conventional fin stabilizers, zero-speed fins employ a drag-based mechanism and active oscillations to generate stabilizing forces even when the vessel is stationary. We propose a simple linear control architecture that, however, accounts for nonlinear drag forces and actuator limitations. Simulation results on a high-fidelity vessel model used for HIL testing demonstrate the effectiveness of the proposed approach.
>
---
#### [new 031] Feature-Based vs. GAN-Based Learning from Demonstrations: When and Why
- **分类: cs.LG; cs.AI; cs.GR; cs.RO**

- **简介: 该论文属于模仿学习领域，比较特征方法与GAN方法在奖励函数结构上的差异，探讨其在运动模仿中的优劣及适用场景。**

- **链接: [http://arxiv.org/pdf/2507.05906v1](http://arxiv.org/pdf/2507.05906v1)**

> **作者:** Chenhao Li; Marco Hutter; Andreas Krause
>
> **摘要:** This survey provides a comparative analysis of feature-based and GAN-based approaches to learning from demonstrations, with a focus on the structure of reward functions and their implications for policy learning. Feature-based methods offer dense, interpretable rewards that excel at high-fidelity motion imitation, yet often require sophisticated representations of references and struggle with generalization in unstructured settings. GAN-based methods, in contrast, use implicit, distributional supervision that enables scalability and adaptation flexibility, but are prone to training instability and coarse reward signals. Recent advancements in both paradigms converge on the importance of structured motion representations, which enable smoother transitions, controllable synthesis, and improved task integration. We argue that the dichotomy between feature-based and GAN-based methods is increasingly nuanced: rather than one paradigm dominating the other, the choice should be guided by task-specific priorities such as fidelity, diversity, interpretability, and adaptability. This work outlines the algorithmic trade-offs and design considerations that underlie method selection, offering a framework for principled decision-making in learning from demonstrations.
>
---
#### [new 032] Safe Domain Randomization via Uncertainty-Aware Out-of-Distribution Detection and Policy Adaptation
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习领域，解决真实环境部署中的安全与泛化问题。提出UARL框架，通过不确定性感知的OOD检测和策略适应，提升政策安全性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.06111v1](http://arxiv.org/pdf/2507.06111v1)**

> **作者:** Mohamad H. Danesh; Maxime Wabartha; Stanley Wu; Joelle Pineau; Hsiu-Chin Lin
>
> **摘要:** Deploying reinforcement learning (RL) policies in real-world involves significant challenges, including distribution shifts, safety concerns, and the impracticality of direct interactions during policy refinement. Existing methods, such as domain randomization (DR) and off-dynamics RL, enhance policy robustness by direct interaction with the target domain, an inherently unsafe practice. We propose Uncertainty-Aware RL (UARL), a novel framework that prioritizes safety during training by addressing Out-Of-Distribution (OOD) detection and policy adaptation without requiring direct interactions in target domain. UARL employs an ensemble of critics to quantify policy uncertainty and incorporates progressive environmental randomization to prepare the policy for diverse real-world conditions. By iteratively refining over high-uncertainty regions of the state space in simulated environments, UARL enhances robust generalization to the target domain without explicitly training on it. We evaluate UARL on MuJoCo benchmarks and a quadrupedal robot, demonstrating its effectiveness in reliable OOD detection, improved performance, and enhanced sample efficiency compared to baselines.
>
---
#### [new 033] Event-RGB Fusion for Spacecraft Pose Estimation Under Harsh Lighting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于航天器姿态估计任务，解决极端光照下视觉定位精度下降的问题。通过融合RGB与事件传感器数据，提升姿态估计的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.05698v1](http://arxiv.org/pdf/2507.05698v1)**

> **作者:** Mohsi Jawaid; Marcus Märtens; Tat-Jun Chin
>
> **摘要:** Spacecraft pose estimation is crucial for autonomous in-space operations, such as rendezvous, docking and on-orbit servicing. Vision-based pose estimation methods, which typically employ RGB imaging sensors, is a compelling solution for spacecraft pose estimation, but are challenged by harsh lighting conditions, which produce imaging artifacts such as glare, over-exposure, blooming and lens flare. Due to their much higher dynamic range, neuromorphic or event sensors are more resilient to extreme lighting conditions. However, event sensors generally have lower spatial resolution and suffer from reduced signal-to-noise ratio during periods of low relative motion. This work addresses these individual sensor limitations by introducing a sensor fusion approach combining RGB and event sensors. A beam-splitter prism was employed to achieve precise optical and temporal alignment. Then, a RANSAC-based technique was developed to fuse the information from the RGB and event channels to achieve pose estimation that leveraged the strengths of the two modalities. The pipeline was complemented by dropout uncertainty estimation to detect extreme conditions that affect either channel. To benchmark the performance of the proposed event-RGB fusion method, we collected a comprehensive real dataset of RGB and event data for satellite pose estimation in a laboratory setting under a variety of challenging illumination conditions. Encouraging results on the dataset demonstrate the efficacy of our event-RGB fusion approach and further supports the usage of event sensors for spacecraft pose estimation. To support community research on this topic, our dataset will be released publicly.
>
---
## 更新

#### [replaced 001] SurgiSR4K: A High-Resolution Endoscopic Video Dataset for Robotic-Assisted Minimally Invasive Procedures
- **分类: eess.IV; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00209v2](http://arxiv.org/pdf/2507.00209v2)**

> **作者:** Fengyi Jiang; Xiaorui Zhang; Lingbo Jin; Ruixing Liang; Yuxin Chen; Adi Chola Venkatesh; Jason Culman; Tiantian Wu; Lirong Shao; Wenqing Sun; Cong Gao; Hallie McNamara; Jingpei Lu; Omid Mohareri
>
> **摘要:** High-resolution imaging is crucial for enhancing visual clarity and enabling precise computer-assisted guidance in minimally invasive surgery (MIS). Despite the increasing adoption of 4K endoscopic systems, there remains a significant gap in publicly available native 4K datasets tailored specifically for robotic-assisted MIS. We introduce SurgiSR4K, the first publicly accessible surgical imaging and video dataset captured at a native 4K resolution, representing realistic conditions of robotic-assisted procedures. SurgiSR4K comprises diverse visual scenarios including specular reflections, tool occlusions, bleeding, and soft tissue deformations, meticulously designed to reflect common challenges faced during laparoscopic and robotic surgeries. This dataset opens up possibilities for a broad range of computer vision tasks that might benefit from high resolution data, such as super resolution (SR), smoke removal, surgical instrument detection, 3D tissue reconstruction, monocular depth estimation, instance segmentation, novel view synthesis, and vision-language model (VLM) development. SurgiSR4K provides a robust foundation for advancing research in high-resolution surgical imaging and fosters the development of intelligent imaging technologies aimed at enhancing performance, safety, and usability in image-guided robotic surgeries.
>
---
#### [replaced 002] Hierarchical Vision-Language Planning for Multi-Step Humanoid Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.22827v2](http://arxiv.org/pdf/2506.22827v2)**

> **作者:** André Schakkal; Ben Zandonati; Zhutian Yang; Navid Azizan
>
> **备注:** Accepted at the RSS 2025 Workshop on Robot Planning in the Era of Foundation Models
>
> **摘要:** Enabling humanoid robots to reliably execute complex multi-step manipulation tasks is crucial for their effective deployment in industrial and household environments. This paper presents a hierarchical planning and control framework designed to achieve reliable multi-step humanoid manipulation. The proposed system comprises three layers: (1) a low-level RL-based controller responsible for tracking whole-body motion targets; (2) a mid-level set of skill policies trained via imitation learning that produce motion targets for different steps of a task; and (3) a high-level vision-language planning module that determines which skills should be executed and also monitors their completion in real-time using pretrained vision-language models (VLMs). Experimental validation is performed on a Unitree G1 humanoid robot executing a non-prehensile pick-and-place task. Over 40 real-world trials, the hierarchical system achieved a 73% success rate in completing the full manipulation sequence. These experiments confirm the feasibility of the proposed hierarchical system, highlighting the benefits of VLM-based skill planning and monitoring for multi-step manipulation scenarios. See https://vlp-humanoid.github.io/ for video demonstrations of the policy rollout.
>
---
#### [replaced 003] CLOVER: Context-aware Long-term Object Viewpoint- and Environment- Invariant Representation Learning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2407.09718v2](http://arxiv.org/pdf/2407.09718v2)**

> **作者:** Dongmyeong Lee; Amanda Adkins; Joydeep Biswas
>
> **备注:** 8 pages, 3 figures, 8 tables
>
> **摘要:** Mobile service robots can benefit from object-level understanding of their environments, including the ability to distinguish object instances and re-identify previously seen instances. Object re-identification is challenging across different viewpoints and in scenes with significant appearance variation arising from weather or lighting changes. Existing works on object re-identification either focus on specific classes or require foreground segmentation. Further, these methods, along with object re-identification datasets, have limited consideration of challenges such as outdoor scenes and illumination changes. To address this problem, we introduce CODa Re-ID: an in-the-wild object re-identification dataset containing 1,037,814 observations of 557 objects across 8 classes under diverse lighting conditions and viewpoints. Further, we propose CLOVER, a representation learning method for object observations that can distinguish between static object instances without requiring foreground segmentation. We also introduce MapCLOVER, a method for scalably summarizing CLOVER descriptors for use in object maps and matching new observations to summarized descriptors. Our results show that CLOVER achieves superior performance in static object re-identification under varying lighting conditions and viewpoint changes and can generalize to unseen instances and classes.
>
---
#### [replaced 004] Improving Trust Estimation in Human-Robot Collaboration Using Beta Reputation at Fine-grained Timescales
- **分类: cs.RO; cs.AI; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.01866v2](http://arxiv.org/pdf/2411.01866v2)**

> **作者:** Resul Dagdanov; Milan Andrejevic; Dikai Liu; Chin-Teng Lin
>
> **备注:** 8 pages, 7 figures, 1 table, published in IEEE Robotics and Automation Letters (RA-L) 2025
>
> **摘要:** When interacting with each other, humans adjust their behavior based on perceived trust. To achieve similar adaptability, robots must accurately estimate human trust at sufficiently granular timescales while collaborating with humans. Beta reputation is a popular way to formalize a mathematical estimation of human trust. However, it relies on binary performance, which updates trust estimations only after each task concludes. Additionally, manually crafting a reward function is the usual method of building a performance indicator, which is labor-intensive and time-consuming. These limitations prevent efficient capture of continuous trust changes at more granular timescales throughout the collaboration task. Therefore, this paper presents a new framework for the estimation of human trust using beta reputation at fine-grained timescales. To achieve granularity in beta reputation, we utilize continuous reward values to update trust estimates at each timestep of a task. We construct a continuous reward function using maximum entropy optimization to eliminate the need for the laborious specification of a performance indicator. The proposed framework improves trust estimations by increasing accuracy, eliminating the need to manually craft a reward function, and advancing toward the development of more intelligent robots.
>
---
#### [replaced 005] RwoR: Generating Robot Demonstrations from Human Hand Collection for Policy Learning without Robot
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.03930v2](http://arxiv.org/pdf/2507.03930v2)**

> **作者:** Liang Heng; Xiaoqi Li; Shangqing Mao; Jiaming Liu; Ruolin Liu; Jingli Wei; Yu-Kai Wang; Yueru Jia; Chenyang Gu; Rui Zhao; Shanghang Zhang; Hao Dong
>
> **摘要:** Recent advancements in imitation learning have shown promising results in robotic manipulation, driven by the availability of high-quality training data. To improve data collection efficiency, some approaches focus on developing specialized teleoperation devices for robot control, while others directly use human hand demonstrations to obtain training data. However, the former requires both a robotic system and a skilled operator, limiting scalability, while the latter faces challenges in aligning the visual gap between human hand demonstrations and the deployed robot observations. To address this, we propose a human hand data collection system combined with our hand-to-gripper generative model, which translates human hand demonstrations into robot gripper demonstrations, effectively bridging the observation gap. Specifically, a GoPro fisheye camera is mounted on the human wrist to capture human hand demonstrations. We then train a generative model on a self-collected dataset of paired human hand and UMI gripper demonstrations, which have been processed using a tailored data pre-processing strategy to ensure alignment in both timestamps and observations. Therefore, given only human hand demonstrations, we are able to automatically extract the corresponding SE(3) actions and integrate them with high-quality generated robot demonstrations through our generation pipeline for training robotic policy model. In experiments, the robust manipulation performance demonstrates not only the quality of the generated robot demonstrations but also the efficiency and practicality of our data collection method. More demonstrations can be found at: https://rwor.github.io/
>
---
#### [replaced 006] Distributionally Robust Predictive Runtime Verification under Spatio-Temporal Logic Specifications
- **分类: eess.SY; cs.LO; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2504.02964v2](http://arxiv.org/pdf/2504.02964v2)**

> **作者:** Yiqi Zhao; Emily Zhu; Bardh Hoxha; Georgios Fainekos; Jyotirmoy V. Deshmukh; Lars Lindemann
>
> **备注:** arXiv admin note: text overlap with arXiv:2311.09482
>
> **摘要:** Cyber-physical systems (CPS) designed in simulators, often consisting of multiple interacting agents (e.g. in multi-agent formations), behave differently in the real-world. We want to verify these systems during runtime when they are deployed. We thus propose robust predictive runtime verification (RPRV) algorithms for: (1) general stochastic CPS under signal temporal logic (STL) tasks, and (2) stochastic multi-agent systems (MAS) under spatio-temporal logic tasks. The RPRV problem presents the following challenges: (1) there may not be sufficient data on the behavior of the deployed CPS, (2) predictive models based on design phase system trajectories may encounter distribution shift during real-world deployment, and (3) the algorithms need to scale to the complexity of MAS and be applicable to spatio-temporal logic tasks. To address the challenges, we assume knowledge of an upper bound on the statistical distance between the trajectory distributions of the system at deployment and design time. We are motivated by our prior work [1, 2] where we proposed an accurate and an interpretable RPRV algorithm for general CPS, which we here extend to the MAS setting and spatio-temporal logic tasks. Specifically, we use a learned predictive model to estimate the system behavior at runtime and robust conformal prediction to obtain probabilistic guarantees by accounting for distribution shifts. Building on [1], we perform robust conformal prediction over the robust semantics of spatio-temporal reach and escape logic (STREL) to obtain centralized RPRV algorithms for MAS. We empirically validate our results in a drone swarm simulator, where we show the scalability of our RPRV algorithms to MAS and analyze the impact of different trajectory predictors on the verification result. To the best of our knowledge, these are the first statistically valid algorithms for MAS under distribution shift.
>
---
#### [replaced 007] Detecting and Diagnosing Faults in Autonomous Robot Swarms with an Artificial Antibody Population Model
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.19942v2](http://arxiv.org/pdf/2412.19942v2)**

> **作者:** James O'Keeffe
>
> **摘要:** An active approach to fault tolerance, the combined processes of fault detection, diagnosis, and recovery, is essential for long term autonomy in robots -- particularly multi-robot systems and swarms. Previous efforts have primarily focussed on spontaneously occurring electro-mechanical failures in the sensors and actuators of a minority sub-population of robots. While the systems that enable this function are valuable, they have not yet considered that many failures arise from gradual wear and tear with continued operation, and that this may be more challenging to detect than sudden step changes in performance. This paper presents the Artificial Antibody Population Dynamics (AAPD) model -- an immune-inspired model for the detection and diagnosis of gradual degradation in robot swarms. The AAPD model is demonstrated to reliably detect and diagnose gradual degradation, as well as spontaneous changes in performance, among swarms of robots of varying sizes while remaining tolerant of normally behaving robots. The AAPD model is distributed, offers supervised and unsupervised configurations, and demonstrates promising scalable properties. Deploying the AAPD model on a swarm of foraging robots undergoing gradual degradation enables the swarm to operate on average at between 70% - 97% of its performance in perfect conditions and is able to prevent instances of robots failing in the field during experiments in most of the cases tested.
>
---
#### [replaced 008] SRT-H: A Hierarchical Framework for Autonomous Surgery via Language Conditioned Imitation Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.10251v3](http://arxiv.org/pdf/2505.10251v3)**

> **作者:** Ji Woong Kim; Juo-Tung Chen; Pascal Hansen; Lucy X. Shi; Antony Goldenberg; Samuel Schmidgall; Paul Maria Scheikl; Anton Deguet; Brandon M. White; De Ru Tsai; Richard Cha; Jeffrey Jopling; Chelsea Finn; Axel Krieger
>
> **摘要:** Research on autonomous surgery has largely focused on simple task automation in controlled environments. However, real-world surgical applications demand dexterous manipulation over extended durations and generalization to the inherent variability of human tissue. These challenges remain difficult to address using existing logic-based or conventional end-to-end learning approaches. To address this gap, we propose a hierarchical framework for performing dexterous, long-horizon surgical steps. Our approach utilizes a high-level policy for task planning and a low-level policy for generating robot trajectories. The high-level planner plans in language space, generating task-level or corrective instructions that guide the robot through the long-horizon steps and correct for the low-level policy's errors. We validate our framework through ex vivo experiments on cholecystectomy, a commonly-practiced minimally invasive procedure, and conduct ablation studies to evaluate key components of the system. Our method achieves a 100\% success rate across eight unseen ex vivo gallbladders, operating fully autonomously without human intervention. This work demonstrates step-level autonomy in a surgical procedure, marking a milestone toward clinical deployment of autonomous surgical systems.
>
---
#### [replaced 009] Coarse-to-fine Q-Network with Action Sequence for Data-Efficient Robot Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.12155v5](http://arxiv.org/pdf/2411.12155v5)**

> **作者:** Younggyo Seo; Pieter Abbeel
>
> **备注:** 18 Pages. Website: https://younggyo.me/cqn-as/
>
> **摘要:** Predicting a sequence of actions has been crucial in the success of recent behavior cloning algorithms in robotics. Can similar ideas improve reinforcement learning (RL)? We answer affirmatively by observing that incorporating action sequences when predicting ground-truth return-to-go leads to lower validation loss. Motivated by this, we introduce Coarse-to-fine Q-Network with Action Sequence (CQN-AS), a novel value-based RL algorithm that learns a critic network that outputs Q-values over a sequence of actions, i.e., explicitly training the value function to learn the consequence of executing action sequences. Our experiments show that CQN-AS outperforms several baselines on a variety of sparse-reward humanoid control and tabletop manipulation tasks from BiGym and RLBench.
>
---
#### [replaced 010] Variational OOD State Correction for Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.00503v3](http://arxiv.org/pdf/2505.00503v3)**

> **作者:** Ke Jiang; Wen Jiang; Xiaoyang Tan
>
> **摘要:** The performance of Offline reinforcement learning is significantly impacted by the issue of state distributional shift, and out-of-distribution (OOD) state correction is a popular approach to address this problem. In this paper, we propose a novel method named Density-Aware Safety Perception (DASP) for OOD state correction. Specifically, our method encourages the agent to prioritize actions that lead to outcomes with higher data density, thereby promoting its operation within or the return to in-distribution (safe) regions. To achieve this, we optimize the objective within a variational framework that concurrently considers both the potential outcomes of decision-making and their density, thus providing crucial contextual information for safe decision-making. Finally, we validate the effectiveness and feasibility of our proposed method through extensive experimental evaluations on the offline MuJoCo and AntMaze suites.
>
---
#### [replaced 011] Dynamics and multi-stability of a rotor-actuated Twistcar robot with passive steering joint
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04846v2](http://arxiv.org/pdf/2507.04846v2)**

> **作者:** Anna Zigelman; Zitao Yu; Rom Levy; Yizhar Or
>
> **备注:** Supporting Information is available at https://yizhar.net.technion.ac.il/files/2025/06/SI-MATLAB-file-Anna-Z.zip
>
> **摘要:** The nonlinear dynamics of many under-actuated wheeled platforms are governed by nonholonomic constraints of no-skid for passively rolling wheels, coupled with momentum balance. In most of theoretical models, the shape variables, i.e. joint angles, are directly prescribed as periodic inputs, such as steering angle of the Twistcar. In this work, we study a variant of the Twistcar model where the actuation input is periodic oscillations of an inertial rotor attached to the main body, while the steering joint is passively free to rotate. Remarkably, the dynamics of this model is extremely rich, and includes multiplicity of periodic solutions, both symmetric and asymmetric, as well as stability transitions and bifurcations. We conduct numerical simulations as well as asymptotic analysis of the vehicle's reduced equations of motion. We use perturbation expansion in order to obtain leading-order dynamics under symmetric periodic solution. Then, we utilize harmonic balance and further scaling assumptions in order to approximate the conditions for symmetry-breaking pitchfork bifurcation and stability transition of the symmetric periodic solution, as a function of actuation frequency and structural parameters. The asymptotic results show good agreement with numerical simulations. The results highlight the role of passive shape variables in generating multi-stable periodic solutions for nonholonomic systems of robotic locomotion.
>
---
#### [replaced 012] Holistic Construction Automation with Modular Robots: From High-Level Task Specification to Execution
- **分类: cs.RO; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2412.20867v2](http://arxiv.org/pdf/2412.20867v2)**

> **作者:** Jonathan Külz; Michael Terzer; Marco Magri; Andrea Giusti; Matthias Althoff
>
> **备注:** Appeared in IEEE Transactions on Automation Science and Engineering https://ieeexplore.ieee.org/document/11036791
>
> **摘要:** In situ robotic automation in construction is challenging due to constantly changing environments, a shortage of robotic experts, and a lack of standardized frameworks bridging robotics and construction practices. This work proposes a holistic framework for construction task specification, optimization of robot morphology, and mission execution using a mobile modular reconfigurable robot. Users can specify and monitor the desired robot behavior through a graphical interface. In contrast to existing, monolithic solutions, we automatically identify a new task-tailored robot for every task by integrating \acf{bim}. Our framework leverages modular robot components that enable the fast adaption of robot hardware to the specific demands of the construction task. Other than previous works on modular robot optimization, we consider multiple competing objectives, which allow us to explicitly model the challenges of real-world transfer, such as calibration errors. We demonstrate our framework in simulation by optimizing robots for drilling and spray painting. Finally, experimental validation demonstrates that our approach robustly enables the autonomous execution of robotic drilling.
>
---
#### [replaced 013] Hume: Introducing System-2 Thinking in Visual-Language-Action Model
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21432v4](http://arxiv.org/pdf/2505.21432v4)**

> **作者:** Haoming Song; Delin Qu; Yuanqi Yao; Qizhi Chen; Qi Lv; Yiwen Tang; Modi Shi; Guanghui Ren; Maoqing Yao; Bin Zhao; Dong Wang; Xuelong Li
>
> **摘要:** Humans practice slow thinking before performing actual actions when handling complex tasks in the physical world. This thinking paradigm, recently, has achieved remarkable advancement in boosting Large Language Models (LLMs) to solve complex tasks in digital domains. However, the potential of slow thinking remains largely unexplored for robotic foundation models interacting with the physical world. In this work, we propose Hume: a dual-system Vision-Language-Action (VLA) model with value-guided System-2 thinking and cascaded action denoising, exploring human-like thinking capabilities of Vision-Language-Action models for dexterous robot control. System 2 of Hume implements value-Guided thinking by extending a Vision-Language-Action Model backbone with a novel value-query head to estimate the state-action value of predicted actions. The value-guided thinking is conducted by repeat sampling multiple action candidates and selecting one according to state-action value. System 1 of Hume is a lightweight reactive visuomotor policy that takes System 2 selected action and performs cascaded action denoising for dexterous robot control. At deployment time, System 2 performs value-guided thinking at a low frequency while System 1 asynchronously receives the System 2 selected action candidate and predicts fluid actions in real time. We show that Hume outperforms the existing state-of-the-art Vision-Language-Action models across multiple simulation benchmark and real-robot deployments.
>
---
#### [replaced 014] Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.15866v4](http://arxiv.org/pdf/2409.15866v4)**

> **作者:** Jiayu Chen; Chao Yu; Guosheng Li; Wenhao Tang; Shilong Ji; Xinyi Yang; Botian Xu; Huazhong Yang; Yu Wang
>
> **备注:** Published in IEEE Robotics and Automation Letters 2025
>
> **摘要:** Multi-UAV pursuit-evasion, where pursuers aim to capture evaders, poses a key challenge for UAV swarm intelligence. Multi-agent reinforcement learning (MARL) has demonstrated potential in modeling cooperative behaviors, but most RL-based approaches remain constrained to simplified simulations with limited dynamics or fixed scenarios. Previous attempts to deploy RL policy to real-world pursuit-evasion are largely restricted to two-dimensional scenarios, such as ground vehicles or UAVs at fixed altitudes. In this paper, we address multi-UAV pursuit-evasion by considering UAV dynamics and physical constraints. We introduce an evader prediction-enhanced network to tackle partial observability in cooperative strategy learning. Additionally, we propose an adaptive environment generator within MARL training, enabling higher exploration efficiency and better policy generalization across diverse scenarios. Simulations show our method significantly outperforms all baselines in challenging scenarios, generalizing to unseen scenarios with a 100% capture rate. Finally, we derive a feasible policy via a two-stage reward refinement and deploy the policy on real quadrotors in a zero-shot manner. To our knowledge, this is the first work to derive and deploy an RL-based policy using collective thrust and body rates control commands for multi-UAV pursuit-evasion in unknown environments. The open-source code and videos are available at https://sites.google.com/view/pursuit-evasion-rl.
>
---
#### [replaced 015] Visual Imitation Enables Contextual Humanoid Control
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03729v4](http://arxiv.org/pdf/2505.03729v4)**

> **作者:** Arthur Allshire; Hongsuk Choi; Junyi Zhang; David McAllister; Anthony Zhang; Chung Min Kim; Trevor Darrell; Pieter Abbeel; Jitendra Malik; Angjoo Kanazawa
>
> **备注:** Project website: https://www.videomimic.net/
>
> **摘要:** How can we teach humanoids to climb staircases and sit on chairs using the surrounding environment context? Arguably, the simplest way is to just show them-casually capture a human motion video and feed it to humanoids. We introduce VIDEOMIMIC, a real-to-sim-to-real pipeline that mines everyday videos, jointly reconstructs the humans and the environment, and produces whole-body control policies for humanoid robots that perform the corresponding skills. We demonstrate the results of our pipeline on real humanoid robots, showing robust, repeatable contextual control such as staircase ascents and descents, sitting and standing from chairs and benches, as well as other dynamic whole-body skills-all from a single policy, conditioned on the environment and global root commands. VIDEOMIMIC offers a scalable path towards teaching humanoids to operate in diverse real-world environments.
>
---
#### [replaced 016] On the Robotic Uncertainty of Fully Autonomous Traffic: From Stochastic Car-Following to Mobility-Safety Trade-Offs
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2309.12611v3](http://arxiv.org/pdf/2309.12611v3)**

> **作者:** Hangyu Li; Xiaotong Sun; Chenglin Zhuang; Xiaopeng Li
>
> **摘要:** Recent transportation research highlights the potential of autonomous vehicles (AV) to improve traffic flow mobility as they are able to maintain smaller car-following distances. However, as a unique class of ground robots, AVs are susceptible to robotic errors, particularly in their perception and control modules with imperfect sensors and actuators, leading to uncertainties in their movements and an increased risk of collisions. Consequently, conservative operational strategies, such as larger headway and slower speeds, are implemented to prioritize safety over mobility in real-world operations. To reconcile the inconsistency, this paper presents an analytical model framework that delineates the endogenous reciprocity between traffic safety and mobility that arises from AVs' robotic uncertainties. Using both realistic car-following data and a stochastic intelligent driving model (IDM), the stochastic car-following distance is derived as a key parameter, enabling analysis of single-lane capacity and collision probability. A semi-Markov process is then employed to model the dynamics of the lane capacity, and the resulting collision-inclusive capacity, representing expected lane capacity under stationary conditions, serves as the primary performance metric for fully autonomous traffic. The analytical results are further utilized to investigate the impacts of critical parameters in AV and roadway designs on traffic performance, as well as the properties of optimal speed and headway under mobility-targeted or safety-dominated management objectives. Extensions to scenarios involving multiple non-independent collisions or multi-lane traffic scenarios are also discussed, which demonstrates the robustness of the theoretical results and their practical applications.
>
---
#### [replaced 017] VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01932v4](http://arxiv.org/pdf/2502.01932v4)**

> **作者:** Zelai Xu; Ruize Zhang; Chao Yu; Huining Yuan; Xiangmin Yi; Shilong Ji; Chuqi Wang; Wenhao Tang; Feng Gao; Wenbo Ding; Xinlei Chen; Yu Wang
>
> **摘要:** Robot sports, characterized by well-defined objectives, explicit rules, and dynamic interactions, present ideal scenarios for demonstrating embodied intelligence. In this paper, we present VolleyBots, a novel robot sports testbed where multiple drones cooperate and compete in the sport of volleyball under physical dynamics. VolleyBots integrates three features within a unified platform: competitive and cooperative gameplay, turn-based interaction structure, and agile 3D maneuvering. Competitive and cooperative gameplay challenges each drone to coordinate with its teammates while anticipating and countering opposing teams' tactics. Turn-based interaction demands precise timing, accurate state prediction, and management of long-horizon temporal dependencies. Agile 3D maneuvering requires rapid accelerations, sharp turns, and precise 3D positioning despite the quadrotor's underactuated dynamics. These intertwined features yield a complex problem combining motion control and strategic play, with no available expert demonstrations. We provide a comprehensive suite of tasks ranging from single-drone drills to multi-drone cooperative and competitive tasks, accompanied by baseline evaluations of representative multi-agent reinforcement learning (MARL) and game-theoretic algorithms. Simulation results show that on-policy reinforcement learning (RL) methods outperform off-policy methods in single-agent tasks, but both approaches struggle in complex tasks that combine motion control and strategic play. We additionally design a hierarchical policy which achieves a 69.5% percent win rate against the strongest baseline in the 3 vs 3 task, underscoring its potential as an effective solution for tackling the complex interplay between low-level control and high-level strategy. The project page is at https://sites.google.com/view/thu-volleybots.
>
---
#### [replaced 018] Gradient Field-Based Dynamic Window Approach for Collision Avoidance in Complex Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.03260v2](http://arxiv.org/pdf/2504.03260v2)**

> **作者:** Ze Zhang; Yifan Xue; Nadia Figueroa; Knut Åkesson
>
> **备注:** This paper has been accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** For safe and flexible navigation in multi-robot systems, this paper presents an enhanced and predictive sampling-based trajectory planning approach in complex environments, the Gradient Field-based Dynamic Window Approach (GF-DWA). Building upon the dynamic window approach, the proposed method utilizes gradient information of obstacle distances as a new cost term to anticipate potential collisions. This enhancement enables the robot to improve awareness of obstacles, including those with non-convex shapes. The gradient field is derived from the Gaussian process distance field, which generates both the distance field and gradient field by leveraging Gaussian process regression to model the spatial structure of the environment. Through several obstacle avoidance and fleet collision avoidance scenarios, the proposed GF-DWA is shown to outperform other popular trajectory planning and control methods in terms of safety and flexibility, especially in complex environments with non-convex obstacles.
>
---
#### [replaced 019] Learning thin deformable object manipulation with a multi-sensory integrated soft hand
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.13952v2](http://arxiv.org/pdf/2411.13952v2)**

> **作者:** Chao Zhao; Chunli Jiang; Lifan Luo; Shuai Yuan; Qifeng Chen; Hongyu Yu
>
> **备注:** Accepted by T-RO, 20 pages
>
> **摘要:** Robotic manipulation has made significant advancements, with systems demonstrating high precision and repeatability. However, this remarkable precision often fails to translate into efficient manipulation of thin deformable objects. Current robotic systems lack imprecise dexterity, the ability to perform dexterous manipulation through robust and adaptive behaviors that do not rely on precise control. This paper explores the singulation and grasping of thin, deformable objects. Here, we propose a novel solution that incorporates passive compliance, touch, and proprioception into thin, deformable object manipulation. Our system employs a soft, underactuated hand that provides passive compliance, facilitating adaptive and gentle interactions to dexterously manipulate deformable objects without requiring precise control. The tactile and force/torque sensors equipped on the hand, along with a depth camera, gather sensory data required for manipulation via the proposed slip module. The manipulation policies are learned directly from raw sensory data via model-free reinforcement learning, bypassing explicit environmental and object modeling. We implement a hierarchical double-loop learning process to enhance learning efficiency by decoupling the action space. Our method was deployed on real-world robots and trained in a self-supervised manner. The resulting policy was tested on a variety of challenging tasks that were beyond the capabilities of prior studies, ranging from displaying suit fabric like a salesperson to turning pages of sheet music for violinists.
>
---
#### [replaced 020] From LLMs to Actions: Latent Codes as Bridges in Hierarchical Robot Control
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.04798v3](http://arxiv.org/pdf/2405.04798v3)**

> **作者:** Yide Shentu; Philipp Wu; Aravind Rajeswaran; Pieter Abbeel
>
> **摘要:** Hierarchical control for robotics has long been plagued by the need to have a well defined interface layer to communicate between high-level task planners and low-level policies. With the advent of LLMs, language has been emerging as a prospective interface layer. However, this has several limitations. Not all tasks can be decomposed into steps that are easily expressible in natural language (e.g. performing a dance routine). Further, it makes end-to-end finetuning on embodied data challenging due to domain shift and catastrophic forgetting. We introduce our method -- Learnable Latent Codes as Bridges (LCB) -- as an alternate architecture to overcome these limitations. \method~uses a learnable latent code to act as a bridge between LLMs and low-level policies. This enables LLMs to flexibly communicate goals in the task plan without being entirely constrained by language limitations. Additionally, it enables end-to-end finetuning without destroying the embedding space of word tokens learned during pre-training. Through experiments on Language Table and Calvin, two common language based benchmarks for embodied agents, we find that \method~outperforms baselines (including those w/ GPT-4V) that leverage pure language as the interface layer on tasks that require reasoning and multi-step behaviors.
>
---
#### [replaced 021] CFMW: Cross-modality Fusion Mamba for Robust Object Detection under Adverse Weather
- **分类: cs.CV; cs.MM; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2404.16302v2](http://arxiv.org/pdf/2404.16302v2)**

> **作者:** Haoyuan Li; Qi Hu; Binjia Zhou; You Yao; Jiacheng Lin; Kailun Yang; Peng Chen
>
> **备注:** Accepted to IEEE Transactions on Circuits and Systems for Video Technology (TCSVT). The dataset and source code will be made publicly available at https://github.com/lhy-zjut/CFMW
>
> **摘要:** Visible-infrared image pairs provide complementary information, enhancing the reliability and robustness of object detection applications in real-world scenarios. However, most existing methods face challenges in maintaining robustness under complex weather conditions, which limits their applicability. Meanwhile, the reliance on attention mechanisms in modality fusion introduces significant computational complexity and storage overhead, particularly when dealing with high-resolution images. To address these challenges, we propose the Cross-modality Fusion Mamba with Weather-removal (CFMW) to augment stability and cost-effectiveness under adverse weather conditions. Leveraging the proposed Perturbation-Adaptive Diffusion Model (PADM) and Cross-modality Fusion Mamba (CFM) modules, CFMW is able to reconstruct visual features affected by adverse weather, enriching the representation of image details. With efficient architecture design, CFMW is 3 times faster than Transformer-style fusion (e.g., CFT). To bridge the gap in relevant datasets, we construct a new Severe Weather Visible-Infrared (SWVI) dataset, encompassing diverse adverse weather scenarios such as rain, haze, and snow. The dataset contains 64,281 paired visible-infrared images, providing a valuable resource for future research. Extensive experiments on public datasets (i.e., M3FD and LLVIP) and the newly constructed SWVI dataset conclusively demonstrate that CFMW achieves state-of-the-art detection performance. Both the dataset and source code will be made publicly available at https://github.com/lhy-zjut/CFMW.
>
---
#### [replaced 022] cuVSLAM: CUDA accelerated visual odometry and mapping
- **分类: cs.RO; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2506.04359v3](http://arxiv.org/pdf/2506.04359v3)**

> **作者:** Alexander Korovko; Dmitry Slepichev; Alexander Efitorov; Aigul Dzhumamuratova; Viktor Kuznetsov; Hesam Rabeti; Joydeep Biswas; Soha Pouya
>
> **摘要:** Accurate and robust pose estimation is a key requirement for any autonomous robot. We present cuVSLAM, a state-of-the-art solution for visual simultaneous localization and mapping, which can operate with a variety of visual-inertial sensor suites, including multiple RGB and depth cameras, and inertial measurement units. cuVSLAM supports operation with as few as one RGB camera to as many as 32 cameras, in arbitrary geometric configurations, thus supporting a wide range of robotic setups. cuVSLAM is specifically optimized using CUDA to deploy in real-time applications with minimal computational overhead on edge-computing devices such as the NVIDIA Jetson. We present the design and implementation of cuVSLAM, example use cases, and empirical results on several state-of-the-art benchmarks demonstrating the best-in-class performance of cuVSLAM.
>
---
#### [replaced 023] Analysis and experiments of the dissipative Twistcar: direction reversal and asymptotic approximations
- **分类: cs.RO; math.DS**

- **链接: [http://arxiv.org/pdf/2506.19112v2](http://arxiv.org/pdf/2506.19112v2)**

> **作者:** Rom Levy; Ari Dantus; Zitao Yu; Yizhar Or
>
> **摘要:** Underactuated wheeled vehicles are commonly studied as nonholonomic systems with periodic actuation. Twistcar is a classical example inspired by a riding toy, which has been analyzed using a planar model of a dynamical system with nonholonomic constraints. Most of the previous analyses did not account for energy dissipation due to frictional resistance. In this work, we study a theoretical two-link model of the Twistcar while incorporating dissipation due to rolling resistance. We obtain asymptotic expressions for the system's small-amplitude steady-state periodic dynamics, which reveals the possibility of reversing the direction of motion upon varying the geometric and mass properties of the vehicle. Next, we design and construct a robotic prototype of the Twistcar whose center-of-mass position can be shifted by adding and removing a massive block, enabling experimental demonstration of the Twistcar's direction reversal phenomenon. We also conduct parameter fitting for the frictional resistance in order to improve agreement with experiments.
>
---
#### [replaced 024] Safe Beyond the Horizon: Efficient Sampling-based MPC with Neural Control Barrier Functions
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.15006v2](http://arxiv.org/pdf/2502.15006v2)**

> **作者:** Ji Yin; Oswin So; Eric Yang Yu; Chuchu Fan; Panagiotis Tsiotras
>
> **备注:** Accepted by RSS 2025
>
> **摘要:** A common problem when using model predictive control (MPC) in practice is the satisfaction of safety specifications beyond the prediction horizon. While theoretical works have shown that safety can be guaranteed by enforcing a suitable terminal set constraint or a sufficiently long prediction horizon, these techniques are difficult to apply and thus are rarely used by practitioners, especially in the case of general nonlinear dynamics. To solve this problem, we impose a tradeoff between exact recursive feasibility, computational tractability, and applicability to ``black-box'' dynamics by learning an approximate discrete-time control barrier function and incorporating it into a variational inference MPC (VIMPC), a sampling-based MPC paradigm. To handle the resulting state constraints, we further propose a new sampling strategy that greatly reduces the variance of the estimated optimal control, improving the sample efficiency, and enabling real-time planning on a CPU. The resulting Neural Shield-VIMPC (NS-VIMPC) controller yields substantial safety improvements compared to existing sampling-based MPC controllers, even under badly designed cost functions. We validate our approach in both simulation and real-world hardware experiments. Project website: https://mit-realm.github.io/ns-vimpc/.
>
---
