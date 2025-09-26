# 机器人 cs.RO

- **最新发布 65 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SPiDR，一种用于安全实现仿真到现实（Sim-to-Real）迁移的算法。针对仿真与现实差距带来的安全性问题，SPiDR通过悲观域随机化方法，在保证性能的同时提供可证明的安全性保障。**

- **链接: [http://arxiv.org/pdf/2509.18648v1](http://arxiv.org/pdf/2509.18648v1)**

> **作者:** Yarden As; Chengrui Qu; Benjamin Unger; Dongho Kang; Max van der Hart; Laixi Shi; Stelian Coros; Adam Wierman; Andreas Krause
>
> **摘要:** Safety remains a major concern for deploying reinforcement learning (RL) in real-world applications. Simulators provide safe, scalable training environments, but the inevitable sim-to-real gap introduces additional safety concerns, as policies must satisfy constraints in real-world conditions that differ from simulation. To address this challenge, robust safe RL techniques offer principled methods, but are often incompatible with standard scalable training pipelines. In contrast, domain randomization, a simple and popular sim-to-real technique, stands out as a promising alternative, although it often results in unsafe behaviors in practice. We present SPiDR, short for Sim-to-real via Pessimistic Domain Randomization -- a scalable algorithm with provable guarantees for safe sim-to-real transfer. SPiDR uses domain randomization to incorporate the uncertainty about the sim-to-real gap into the safety constraints, making it versatile and highly compatible with existing training pipelines. Through extensive experiments on sim-to-sim benchmarks and two distinct real-world robotic platforms, we demonstrate that SPiDR effectively ensures safety despite the sim-to-real gap while maintaining strong performance.
>
---
#### [new 002] N2M: Bridging Navigation and Manipulation by Learning Pose Preference from Rollout
- **分类: cs.RO**

- **简介: 该论文针对移动操作任务中导航与操作策略的不匹配问题，提出N2M模块，通过学习姿态偏好引导机器人到达更优初始位姿，显著提升任务成功率。实验表明其具备高效性、适应性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.18671v1](http://arxiv.org/pdf/2509.18671v1)**

> **作者:** Kaixin Chai; Hyunjun Lee; Joseph J. Lim
>
> **摘要:** In mobile manipulation, the manipulation policy has strong preferences for initial poses where it is executed. However, the navigation module focuses solely on reaching the task area, without considering which initial pose is preferable for downstream manipulation. To address this misalignment, we introduce N2M, a transition module that guides the robot to a preferable initial pose after reaching the task area, thereby substantially improving task success rates. N2M features five key advantages: (1) reliance solely on ego-centric observation without requiring global or historical information; (2) real-time adaptation to environmental changes; (3) reliable prediction with high viewpoint robustness; (4) broad applicability across diverse tasks, manipulation policies, and robot hardware; and (5) remarkable data efficiency and generalizability. We demonstrate the effectiveness of N2M through extensive simulation and real-world experiments. In the PnPCounterToCab task, N2M improves the averaged success rate from 3% with the reachability-based baseline to 54%. Furthermore, in the Toybox Handover task, N2M provides reliable predictions even in unseen environments with only 15 data samples, showing remarkable data efficiency and generalizability.
>
---
#### [new 003] Category-Level Object Shape and Pose Estimation in Less Than a Millisecond
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究目标是物体形状与姿态估计，属于机器人领域的基础任务。提出了一种快速局部求解器，利用类别级先验信息，通过自洽场迭代高效求解位姿和形状，并提供全局最优性证书，适用于实时应用。**

- **链接: [http://arxiv.org/pdf/2509.18979v1](http://arxiv.org/pdf/2509.18979v1)**

> **作者:** Lorenzo Shaikewitz; Tim Nguyen; Luca Carlone
>
> **摘要:** Object shape and pose estimation is a foundational robotics problem, supporting tasks from manipulation to scene understanding and navigation. We present a fast local solver for shape and pose estimation which requires only category-level object priors and admits an efficient certificate of global optimality. Given an RGB-D image of an object, we use a learned front-end to detect sparse, category-level semantic keypoints on the target object. We represent the target object's unknown shape using a linear active shape model and pose a maximum a posteriori optimization problem to solve for position, orientation, and shape simultaneously. Expressed in unit quaternions, this problem admits first-order optimality conditions in the form of an eigenvalue problem with eigenvector nonlinearities. Our primary contribution is to solve this problem efficiently with self-consistent field iteration, which only requires computing a 4-by-4 matrix and finding its minimum eigenvalue-vector pair at each iterate. Solving a linear system for the corresponding Lagrange multipliers gives a simple global optimality certificate. One iteration of our solver runs in about 100 microseconds, enabling fast outlier rejection. We test our method on synthetic data and a variety of real-world settings, including two public datasets and a drone tracking scenario. Code is released at https://github.com/MIT-SPARK/Fast-ShapeAndPose.
>
---
#### [new 004] SlicerROS2: A Research and Development Module for Image-Guided Robotic Interventions
- **分类: cs.RO**

- **简介: 该论文提出SlicerROS2，一个结合3D Slicer与ROS的软件模块，旨在解决医学机器人研究中图像引导与机器人系统集成的问题。通过改进设计与功能，支持实时可视化和数据交互，并展示了其在真实场景中的应用。**

- **链接: [http://arxiv.org/pdf/2509.19076v1](http://arxiv.org/pdf/2509.19076v1)**

> **作者:** Laura Connolly; Aravind S. Kumar; Kapi Ketan Mehta; Lidia Al-Zogbi; Peter Kazanzides; Parvin Mousavi; Gabor Fichtinger; Axel Krieger; Junichi Tokuda; Russell H. Taylor; Simon Leonard; Anton Deguet
>
> **摘要:** Image-guided robotic interventions involve the use of medical imaging in tandem with robotics. SlicerROS2 is a software module that combines 3D Slicer and robot operating system (ROS) in pursuit of a standard integration approach for medical robotics research. The first release of SlicerROS2 demonstrated the feasibility of using the C++ API from 3D Slicer and ROS to load and visualize robots in real time. Since this initial release, we've rewritten and redesigned the module to offer greater modularity, access to low-level features, access to 3D Slicer's Python API, and better data transfer protocols. In this paper, we introduce this new design as well as four applications that leverage the core functionalities of SlicerROS2 in realistic image-guided robotics scenarios.
>
---
#### [new 005] Assistive Decision-Making for Right of Way Navigation at Uncontrolled Intersections
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文研究无信号交叉口的优先通行权辅助决策问题，旨在提升人类驾驶的安全性。提出基于POMDP的驾驶员辅助框架，通过仿真对比四种决策方法，验证概率规划器在部分可观环境下优于规则系统，强调不确定性感知规划的重要性。**

- **链接: [http://arxiv.org/pdf/2509.18407v1](http://arxiv.org/pdf/2509.18407v1)**

> **作者:** Navya Tiwari; Joseph Vazhaeparampil; Victoria Preston
>
> **备注:** 6 pages, 5 figures. Accepted as a poster at Northeast Robotics Colloquium (NERC 2025). Extended abstract
>
> **摘要:** Uncontrolled intersections account for a significant fraction of roadway crashes due to ambiguous right-of-way rules, occlusions, and unpredictable driver behavior. While autonomous vehicle research has explored uncertainty-aware decision making, few systems exist to retrofit human-operated vehicles with assistive navigation support. We present a driver-assist framework for right-of-way reasoning at uncontrolled intersections, formulated as a Partially Observable Markov Decision Process (POMDP). Using a custom simulation testbed with stochastic traffic agents, pedestrians, occlusions, and adversarial scenarios, we evaluate four decision-making approaches: a deterministic finite state machine (FSM), and three probabilistic planners: QMDP, POMCP, and DESPOT. Results show that probabilistic planners outperform the rule-based baseline, achieving up to 97.5 percent collision-free navigation under partial observability, with POMCP prioritizing safety and DESPOT balancing efficiency and runtime feasibility. Our findings highlight the importance of uncertainty-aware planning for driver assistance and motivate future integration of sensor fusion and environment perception modules for real-time deployment in realistic traffic environments.
>
---
#### [new 006] Haptic Communication in Human-Human and Human-Robot Co-Manipulation
- **分类: cs.RO**

- **简介: 该论文研究人与人及人与机器人协作操作物体时的触觉通信。通过IMU捕捉运动数据，对比两种协作方式的流畅性与准确性，发现人类协作更流畅，并提出改进机器人触觉交互能力以提升物理任务协作效果。**

- **链接: [http://arxiv.org/pdf/2509.18327v1](http://arxiv.org/pdf/2509.18327v1)**

> **作者:** Katherine H. Allen; Chris Rogers; Elaine S. Short
>
> **备注:** 9 pages, 18 figures, ROMAN 2025
>
> **摘要:** When a human dyad jointly manipulates an object, they must communicate about their intended motion plans. Some of that collaboration is achieved through the motion of the manipulated object itself, which we call "haptic communication." In this work, we captured the motion of human-human dyads moving an object together with one participant leading a motion plan about which the follower is uninformed. We then captured the same human participants manipulating the same object with a robot collaborator. By tracking the motion of the shared object using a low-cost IMU, we can directly compare human-human shared manipulation to the motion of those same participants interacting with the robot. Intra-study and post-study questionnaires provided participant feedback on the collaborations, indicating that the human-human collaborations are significantly more fluent, and analysis of the IMU data indicates that it captures objective differences in the motion profiles of the conditions. The differences in objective and subjective measures of accuracy and fluency between the human-human and human-robot trials motivate future research into improving robot assistants for physical tasks by enabling them to send and receive anthropomorphic haptic signals.
>
---
#### [new 007] A Multimodal Stochastic Planning Approach for Navigation and Multi-Robot Coordination
- **分类: cs.RO**

- **简介: 该论文提出一种多模态随机规划方法，用于机器人导航与多机器人协同。通过交叉熵方法优化多模态策略，提升鲁棒性与探索效率，解决局部最优和死锁问题，并实现高效的分布式多机器人避障规划。**

- **链接: [http://arxiv.org/pdf/2509.19168v1](http://arxiv.org/pdf/2509.19168v1)**

> **作者:** Mark Gonzales; Ethan Oh; Joseph Moore
>
> **备注:** 8 Pages, 7 Figures
>
> **摘要:** In this paper, we present a receding-horizon, sampling-based planner capable of reasoning over multimodal policy distributions. By using the cross-entropy method to optimize a multimodal policy under a common cost function, our approach increases robustness against local minima and promotes effective exploration of the solution space. We show that our approach naturally extends to multi-robot collision-free planning, enables agents to share diverse candidate policies to avoid deadlocks, and allows teams to minimize a global objective without incurring the computational complexity of centralized optimization. Numerical simulations demonstrate that employing multiple modes significantly improves success rates in trap environments and in multi-robot collision avoidance. Hardware experiments further validate the approach's real-time feasibility and practical performance.
>
---
#### [new 008] Bi-VLA: Bilateral Control-Based Imitation Learning via Vision-Language Fusion for Action Generation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出Bi-VLA，一种基于双侧控制的模仿学习框架，通过视觉-语言融合实现多任务操作。传统方法需任务专用模型，而Bi-VLA结合关节数据、视觉与自然语言指令，提升通用性与任务成功率。**

- **链接: [http://arxiv.org/pdf/2509.18865v1](http://arxiv.org/pdf/2509.18865v1)**

> **作者:** Masato Kobayashi; Thanpimon Buamanee
>
> **摘要:** We propose Bilateral Control-Based Imitation Learning via Vision-Language Fusion for Action Generation (Bi-VLA), a novel framework that extends bilateral control-based imitation learning to handle more than one task within a single model. Conventional bilateral control methods exploit joint angle, velocity, torque, and vision for precise manipulation but require task-specific models, limiting their generality. Bi-VLA overcomes this limitation by utilizing robot joint angle, velocity, and torque data from leader-follower bilateral control with visual features and natural language instructions through SigLIP and FiLM-based fusion. We validated Bi-VLA on two task types: one requiring supplementary language cues and another distinguishable solely by vision. Real-robot experiments showed that Bi-VLA successfully interprets vision-language combinations and improves task success rates compared to conventional bilateral control-based imitation learning. Our Bi-VLA addresses the single-task limitation of prior bilateral approaches and provides empirical evidence that combining vision and language significantly enhances versatility. Experimental results validate the effectiveness of Bi-VLA in real-world tasks. For additional material, please visit the website: https://mertcookimg.github.io/bi-vla/
>
---
#### [new 009] Imitation-Guided Bimanual Planning for Stable Manipulation under Changing External Forces
- **分类: cs.RO**

- **简介: 该论文研究双臂机器人在动态环境下的稳定操作任务，旨在解决外部力变化下抓取转换不流畅的问题。提出模仿引导的双臂规划框架，结合抓取策略优化与分层运动架构，提升操作稳定性与效率。**

- **链接: [http://arxiv.org/pdf/2509.19261v1](http://arxiv.org/pdf/2509.19261v1)**

> **作者:** Kuanqi Cai; Chunfeng Wang; Zeqi Li; Haowen Yao; Weinan Chen; Luis Figueredo; Aude Billard; Arash Ajoudani
>
> **摘要:** Robotic manipulation in dynamic environments often requires seamless transitions between different grasp types to maintain stability and efficiency. However, achieving smooth and adaptive grasp transitions remains a challenge, particularly when dealing with external forces and complex motion constraints. Existing grasp transition strategies often fail to account for varying external forces and do not optimize motion performance effectively. In this work, we propose an Imitation-Guided Bimanual Planning Framework that integrates efficient grasp transition strategies and motion performance optimization to enhance stability and dexterity in robotic manipulation. Our approach introduces Strategies for Sampling Stable Intersections in Grasp Manifolds for seamless transitions between uni-manual and bi-manual grasps, reducing computational costs and regrasping inefficiencies. Additionally, a Hierarchical Dual-Stage Motion Architecture combines an Imitation Learning-based Global Path Generator with a Quadratic Programming-driven Local Planner to ensure real-time motion feasibility, obstacle avoidance, and superior manipulability. The proposed method is evaluated through a series of force-intensive tasks, demonstrating significant improvements in grasp transition efficiency and motion performance. A video demonstrating our simulation results can be viewed at \href{https://youtu.be/3DhbUsv4eDo}{\textcolor{blue}{https://youtu.be/3DhbUsv4eDo}}.
>
---
#### [new 010] Semantic-Aware Particle Filter for Reliable Vineyard Robot Localisation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对葡萄园机器人定位问题，提出一种语义感知粒子滤波方法。通过融合LiDAR和语义特征（如藤干和支撑杆），利用语义墙缓解行别混淆，并引入GPS先验保持全局一致性，提升了户外结构化环境中的定位可靠性。**

- **链接: [http://arxiv.org/pdf/2509.18342v1](http://arxiv.org/pdf/2509.18342v1)**

> **作者:** Rajitha de Silva; Jonathan Cox; James R. Heselden; Marija Popovic; Cesar Cadena; Riccardo Polvara
>
> **备注:** Sumbitted to ICRA 2026
>
> **摘要:** Accurate localisation is critical for mobile robots in structured outdoor environments, yet LiDAR-based methods often fail in vineyards due to repetitive row geometry and perceptual aliasing. We propose a semantic particle filter that incorporates stable object-level detections, specifically vine trunks and support poles into the likelihood estimation process. Detected landmarks are projected into a birds eye view and fused with LiDAR scans to generate semantic observations. A key innovation is the use of semantic walls, which connect adjacent landmarks into pseudo-structural constraints that mitigate row aliasing. To maintain global consistency in headland regions where semantics are sparse, we introduce a noisy GPS prior that adaptively supports the filter. Experiments in a real vineyard demonstrate that our approach maintains localisation within the correct row, recovers from deviations where AMCL fails, and outperforms vision-based SLAM methods such as RTAB-Map.
>
---
#### [new 011] PrioriTouch: Adapting to User Contact Preferences for Whole-Arm Physical Human-Robot Interaction
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PrioriTouch框架，用于多接触点的人机物理交互任务，解决用户不同接触偏好冲突的问题。通过学习排序与分层控制结合，实现个性化舒适阈值的适应，提升交互的安全性与舒适性。**

- **链接: [http://arxiv.org/pdf/2509.18447v1](http://arxiv.org/pdf/2509.18447v1)**

> **作者:** Rishabh Madan; Jiawei Lin; Mahika Goel; Angchen Xie; Xiaoyu Liang; Marcus Lee; Justin Guo; Pranav N. Thakkar; Rohan Banerjee; Jose Barreiros; Kate Tsui; Tom Silver; Tapomayukh Bhattacharjee
>
> **备注:** Conference on Robot Learning (CoRL)
>
> **摘要:** Physical human-robot interaction (pHRI) requires robots to adapt to individual contact preferences, such as where and how much force is applied. Identifying preferences is difficult for a single contact; with whole-arm interaction involving multiple simultaneous contacts between the robot and human, the challenge is greater because different body parts can impose incompatible force requirements. In caregiving tasks, where contact is frequent and varied, such conflicts are unavoidable. With multiple preferences across multiple contacts, no single solution can satisfy all objectives--trade-offs are inherent, making prioritization essential. We present PrioriTouch, a framework for ranking and executing control objectives across multiple contacts. PrioriTouch can prioritize from a general collection of controllers, making it applicable not only to caregiving scenarios such as bed bathing and dressing but also to broader multi-contact settings. Our method combines a novel learning-to-rank approach with hierarchical operational space control, leveraging simulation-in-the-loop rollouts for data-efficient and safe exploration. We conduct a user study on physical assistance preferences, derive personalized comfort thresholds, and incorporate them into PrioriTouch. We evaluate PrioriTouch through extensive simulation and real-world experiments, demonstrating its ability to adapt to user contact preferences, maintain task performance, and enhance safety and comfort. Website: https://emprise.cs.cornell.edu/prioritouch.
>
---
#### [new 012] World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出World4RL框架，利用扩散模型构建高保真模拟环境，用于机器人操作策略的强化学习优化。旨在解决专家数据稀缺和现实交互成本高的问题，通过在虚拟环境中直接优化策略，提升了任务成功率。**

- **链接: [http://arxiv.org/pdf/2509.19080v1](http://arxiv.org/pdf/2509.19080v1)**

> **作者:** Zhennan Jiang; Kai Liu; Yuxin Qin; Shuai Tian; Yupeng Zheng; Mingcai Zhou; Chao Yu; Haoran Li; Dongbin Zhao
>
> **摘要:** Robotic manipulation policies are commonly initialized through imitation learning, but their performance is limited by the scarcity and narrow coverage of expert data. Reinforcement learning can refine polices to alleviate this limitation, yet real-robot training is costly and unsafe, while training in simulators suffers from the sim-to-real gap. Recent advances in generative models have demonstrated remarkable capabilities in real-world simulation, with diffusion models in particular excelling at generation. This raises the question of how diffusion model-based world models can be combined to enhance pre-trained policies in robotic manipulation. In this work, we propose World4RL, a framework that employs diffusion-based world models as high-fidelity simulators to refine pre-trained policies entirely in imagined environments for robotic manipulation. Unlike prior works that primarily employ world models for planning, our framework enables direct end-to-end policy optimization. World4RL is designed around two principles: pre-training a diffusion world model that captures diverse dynamics on multi-task datasets and refining policies entirely within a frozen world model to avoid online real-world interactions. We further design a two-hot action encoding scheme tailored for robotic manipulation and adopt diffusion backbones to improve modeling fidelity. Extensive simulation and real-world experiments demonstrate that World4RL provides high-fidelity environment modeling and enables consistent policy refinement, yielding significantly higher success rates compared to imitation learning and other baselines. More visualization results are available at https://world4rl.github.io/.
>
---
#### [new 013] FUNCanon: Learning Pose-Aware Action Primitives via Functional Object Canonicalization for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出FUNCanon框架，旨在解决机器人操作任务中策略泛化性差的问题。通过将任务分解为可组合的动作块，并利用功能对齐和扩散策略，实现跨类别、跨任务的泛化与复用，提升复杂操作场景下的学习效率与适应能力。**

- **链接: [http://arxiv.org/pdf/2509.19102v1](http://arxiv.org/pdf/2509.19102v1)**

> **作者:** Hongli Xu; Lei Zhang; Xiaoyue Hu; Boyang Zhong; Kaixin Bai; Zoltán-Csaba Márton; Zhenshan Bing; Zhaopeng Chen; Alois Christian Knoll; Jianwei Zhang
>
> **备注:** project website: https://sites.google.com/view/funcanon, 11 pages
>
> **摘要:** General-purpose robotic skills from end-to-end demonstrations often leads to task-specific policies that fail to generalize beyond the training distribution. Therefore, we introduce FunCanon, a framework that converts long-horizon manipulation tasks into sequences of action chunks, each defined by an actor, verb, and object. These chunks focus policy learning on the actions themselves, rather than isolated tasks, enabling compositionality and reuse. To make policies pose-aware and category-general, we perform functional object canonicalization for functional alignment and automatic manipulation trajectory transfer, mapping objects into shared functional frames using affordance cues from large vision language models. An object centric and action centric diffusion policy FuncDiffuser trained on this aligned data naturally respects object affordances and poses, simplifying learning and improving generalization ability. Experiments on simulated and real-world benchmarks demonstrate category-level generalization, cross-task behavior reuse, and robust sim2real deployment, showing that functional canonicalization provides a strong inductive bias for scalable imitation learning in complex manipulation domains. Details of the demo and supplemental material are available on our project website https://sites.google.com/view/funcanon.
>
---
#### [new 014] MV-UMI: A Scalable Multi-View Interface for Cross-Embodiment Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出MV-UMI框架，用于跨实体学习中的机器人操作任务。针对手持夹具数据采集设备因视角局限导致场景理解不足的问题，通过融合第一人称与第三人称视角，提升任务性能约47%，保留跨实体优势。**

- **链接: [http://arxiv.org/pdf/2509.18757v1](http://arxiv.org/pdf/2509.18757v1)**

> **作者:** Omar Rayyan; John Abanes; Mahmoud Hafez; Anthony Tzes; Fares Abu-Dakka
>
> **备注:** For project website and videos, see https https://mv-umi.github.io
>
> **摘要:** Recent advances in imitation learning have shown great promise for developing robust robot manipulation policies from demonstrations. However, this promise is contingent on the availability of diverse, high-quality datasets, which are not only challenging and costly to collect but are often constrained to a specific robot embodiment. Portable handheld grippers have recently emerged as intuitive and scalable alternatives to traditional robotic teleoperation methods for data collection. However, their reliance solely on first-person view wrist-mounted cameras often creates limitations in capturing sufficient scene contexts. In this paper, we present MV-UMI (Multi-View Universal Manipulation Interface), a framework that integrates a third-person perspective with the egocentric camera to overcome this limitation. This integration mitigates domain shifts between human demonstration and robot deployment, preserving the cross-embodiment advantages of handheld data-collection devices. Our experimental results, including an ablation study, demonstrate that our MV-UMI framework improves performance in sub-tasks requiring broad scene understanding by approximately 47% across 3 tasks, confirming the effectiveness of our approach in expanding the range of feasible manipulation tasks that can be learned using handheld gripper systems, without compromising the cross-embodiment advantages inherent to such systems.
>
---
#### [new 015] Fine-Tuning Robot Policies While Maintaining User Privacy
- **分类: cs.RO**

- **简介: 该论文研究个性化机器人策略中的隐私保护问题，提出PRoP框架，通过用户唯一密钥变换模型权重，在保持原始策略结构的同时实现个性化与隐私保障。**

- **链接: [http://arxiv.org/pdf/2509.18311v1](http://arxiv.org/pdf/2509.18311v1)**

> **作者:** Benjamin A. Christie; Sagar Parekh; Dylan P. Losey
>
> **摘要:** Recent works introduce general-purpose robot policies. These policies provide a strong prior over how robots should behave -- e.g., how a robot arm should manipulate food items. But in order for robots to match an individual person's needs, users typically fine-tune these generalized policies -- e.g., showing the robot arm how to make their own preferred dinners. Importantly, during the process of personalizing robots, end-users leak data about their preferences, habits, and styles (e.g., the foods they prefer to eat). Other agents can simply roll-out the fine-tuned policy and see these personally-trained behaviors. This leads to a fundamental challenge: how can we develop robots that personalize actions while keeping learning private from external agents? We here explore this emerging topic in human-robot interaction and develop PRoP, a model-agnostic framework for personalized and private robot policies. Our core idea is to equip each user with a unique key; this key is then used to mathematically transform the weights of the robot's network. With the correct key, the robot's policy switches to match that user's preferences -- but with incorrect keys, the robot reverts to its baseline behaviors. We show the general applicability of our method across multiple model types in imitation learning, reinforcement learning, and classification tasks. PRoP is practically advantageous because it retains the architecture and behaviors of the original policy, and experimentally outperforms existing encoder-based approaches. See videos and code here: https://prop-icra26.github.io.
>
---
#### [new 016] 3D Flow Diffusion Policy: Visuomotor Policy Learning via Generating Flow in 3D Space
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出3D Flow Diffusion Policy（3D FDP），旨在解决机器人操作中视觉运动策略泛化能力不足的问题。通过引入场景级3D流作为结构化中间表示，捕捉局部运动线索，提升在接触密集任务中的表现，并在仿真和真实机器人任务中验证了其优越性。**

- **链接: [http://arxiv.org/pdf/2509.18676v1](http://arxiv.org/pdf/2509.18676v1)**

> **作者:** Sangjun Noh; Dongwoo Nam; Kangmin Kim; Geonhyup Lee; Yeonguk Yu; Raeyoung Kang; Kyoobin Lee
>
> **备注:** 7 main scripts + 2 reference pages
>
> **摘要:** Learning robust visuomotor policies that generalize across diverse objects and interaction dynamics remains a central challenge in robotic manipulation. Most existing approaches rely on direct observation-to-action mappings or compress perceptual inputs into global or object-centric features, which often overlook localized motion cues critical for precise and contact-rich manipulation. We present 3D Flow Diffusion Policy (3D FDP), a novel framework that leverages scene-level 3D flow as a structured intermediate representation to capture fine-grained local motion cues. Our approach predicts the temporal trajectories of sampled query points and conditions action generation on these interaction-aware flows, implemented jointly within a unified diffusion architecture. This design grounds manipulation in localized dynamics while enabling the policy to reason about broader scene-level consequences of actions. Extensive experiments on the MetaWorld benchmark show that 3D FDP achieves state-of-the-art performance across 50 tasks, particularly excelling on medium and hard settings. Beyond simulation, we validate our method on eight real-robot tasks, where it consistently outperforms prior baselines in contact-rich and non-prehensile scenarios. These results highlight 3D flow as a powerful structural prior for learning generalizable visuomotor policies, supporting the development of more robust and versatile robotic manipulation. Robot demonstrations, additional results, and code can be found at https://sites.google.com/view/3dfdp/home.
>
---
#### [new 017] BiGraspFormer: End-to-End Bimanual Grasp Transformer
- **分类: cs.RO**

- **简介: 该论文提出BiGraspFormer，一种端到端的Transformer框架，用于双臂抓取任务。旨在解决现有方法中单臂抓取或分阶段处理导致的协调问题，通过SGB策略直接生成协调的双臂抓取位姿，提升抓取效率与安全性。**

- **链接: [http://arxiv.org/pdf/2509.19142v1](http://arxiv.org/pdf/2509.19142v1)**

> **作者:** Kangmin Kim; Seunghyeok Back; Geonhyup Lee; Sangbeom Lee; Sangjun Noh; Kyoobin Lee
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Bimanual grasping is essential for robots to handle large and complex objects. However, existing methods either focus solely on single-arm grasping or employ separate grasp generation and bimanual evaluation stages, leading to coordination problems including collision risks and unbalanced force distribution. To address these limitations, we propose BiGraspFormer, a unified end-to-end transformer framework that directly generates coordinated bimanual grasps from object point clouds. Our key idea is the Single-Guided Bimanual (SGB) strategy, which first generates diverse single grasp candidates using a transformer decoder, then leverages their learned features through specialized attention mechanisms to jointly predict bimanual poses and quality scores. This conditioning strategy reduces the complexity of the 12-DoF search space while ensuring coordinated bimanual manipulation. Comprehensive simulation experiments and real-world validation demonstrate that BiGraspFormer consistently outperforms existing methods while maintaining efficient inference speed (<0.05s), confirming the effectiveness of our framework. Code and supplementary materials are available at https://sites.google.com/bigraspformer
>
---
#### [new 018] Query-Centric Diffusion Policy for Generalizable Robotic Assembly
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对机器人装配任务中高低层策略不匹配的问题，提出查询中心扩散策略（QDP），通过结构化查询机制结合高层规划与底层控制，提升装配精度和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.18686v1](http://arxiv.org/pdf/2509.18686v1)**

> **作者:** Ziyi Xu; Haohong Lin; Shiqi Liu; Ding Zhao
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** The robotic assembly task poses a key challenge in building generalist robots due to the intrinsic complexity of part interactions and the sensitivity to noise perturbations in contact-rich settings. The assembly agent is typically designed in a hierarchical manner: high-level multi-part reasoning and low-level precise control. However, implementing such a hierarchical policy is challenging in practice due to the mismatch between high-level skill queries and low-level execution. To address this, we propose the Query-centric Diffusion Policy (QDP), a hierarchical framework that bridges high-level planning and low-level control by utilizing queries comprising objects, contact points, and skill information. QDP introduces a query-centric mechanism that identifies task-relevant components and uses them to guide low-level policies, leveraging point cloud observations to improve the policy's robustness. We conduct comprehensive experiments on the FurnitureBench in both simulation and real-world settings, demonstrating improved performance in skill precision and long-horizon success rate. In the challenging insertion and screwing tasks, QDP improves the skill-wise success rate by over 50% compared to baselines without structured queries.
>
---
#### [new 019] Learning Geometry-Aware Nonprehensile Pushing and Pulling with Dexterous Hands
- **分类: cs.RO**

- **简介: 该论文提出GD2P方法，利用灵巧手进行非抓取式推拉操作。针对难以抓取的物体，通过学习几何感知的手部姿态，结合物理仿真和扩散模型，实现稳定、多样化的非抓取操作策略训练。**

- **链接: [http://arxiv.org/pdf/2509.18455v1](http://arxiv.org/pdf/2509.18455v1)**

> **作者:** Yunshuang Li; Yiyang Ling; Gaurav S. Sukhatme; Daniel Seita
>
> **摘要:** Nonprehensile manipulation, such as pushing and pulling, enables robots to move, align, or reposition objects that may be difficult to grasp due to their geometry, size, or relationship to the robot or the environment. Much of the existing work in nonprehensile manipulation relies on parallel-jaw grippers or tools such as rods and spatulas. In contrast, multi-fingered dexterous hands offer richer contact modes and versatility for handling diverse objects to provide stable support over the objects, which compensates for the difficulty of modeling the dynamics of nonprehensile manipulation. Therefore, we propose Geometry-aware Dexterous Pushing and Pulling (GD2P) for nonprehensile manipulation with dexterous robotic hands. We study pushing and pulling by framing the problem as synthesizing and learning pre-contact dexterous hand poses that lead to effective manipulation. We generate diverse hand poses via contact-guided sampling, filter them using physics simulation, and train a diffusion model conditioned on object geometry to predict viable poses. At test time, we sample hand poses and use standard motion planners to select and execute pushing and pulling actions. We perform 840 real-world experiments with an Allegro Hand, comparing our method to baselines. The results indicate that GD2P offers a scalable route for training dexterous nonprehensile manipulation policies. We further demonstrate GD2P on a LEAP Hand, highlighting its applicability to different hand morphologies. Our pre-trained models and dataset, including 1.3 million hand poses across 2.3k objects, will be open-source to facilitate further research. Our project website is available at: geodex2p.github.io.
>
---
#### [new 020] PEEK: Guiding and Minimal Image Representations for Zero-Shot Generalization of Robot Manipulation Policies
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出PEEK方法，旨在解决机器人操作策略的零样本泛化问题。通过微调视觉语言模型，提取关键点表示（动作路径和关注区域），将语义与视觉复杂性解耦，使策略专注于“如何执行”，从而提升跨任务和架构的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.18282v1](http://arxiv.org/pdf/2509.18282v1)**

> **作者:** Jesse Zhang; Marius Memmel; Kevin Kim; Dieter Fox; Jesse Thomason; Fabio Ramos; Erdem Bıyık; Abhishek Gupta; Anqi Li
>
> **备注:** 11 pages
>
> **摘要:** Robotic manipulation policies often fail to generalize because they must simultaneously learn where to attend, what actions to take, and how to execute them. We argue that high-level reasoning about where and what can be offloaded to vision-language models (VLMs), leaving policies to specialize in how to act. We present PEEK (Policy-agnostic Extraction of Essential Keypoints), which fine-tunes VLMs to predict a unified point-based intermediate representation: 1. end-effector paths specifying what actions to take, and 2. task-relevant masks indicating where to focus. These annotations are directly overlaid onto robot observations, making the representation policy-agnostic and transferable across architectures. To enable scalable training, we introduce an automatic annotation pipeline, generating labeled data across 20+ robot datasets spanning 9 embodiments. In real-world evaluations, PEEK consistently boosts zero-shot generalization, including a 41.4x real-world improvement for a 3D policy trained only in simulation, and 2-3.5x gains for both large VLAs and small manipulation policies. By letting VLMs absorb semantic and visual complexity, PEEK equips manipulation policies with the minimal cues they need--where, what, and how. Website at https://peek-robot.github.io/.
>
---
#### [new 021] A Counterfactual Reasoning Framework for Fault Diagnosis in Robot Perception Systems
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种基于反事实推理的机器人感知系统故障检测与隔离框架，旨在解决感知故障难以定位的问题。通过构建可靠性测试和主动控制策略（如MCTS-UCB），提升故障诊断效果，并在视觉导航任务中验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.18460v1](http://arxiv.org/pdf/2509.18460v1)**

> **作者:** Haeyoon Han; Mahdi Taheri; Soon-Jo Chung; Fred Y. Hadaegh
>
> **摘要:** Perception systems provide a rich understanding of the environment for autonomous systems, shaping decisions in all downstream modules. Hence, accurate detection and isolation of faults in perception systems is important. Faults in perception systems pose particular challenges: faults are often tied to the perceptual context of the environment, and errors in their multi-stage pipelines can propagate across modules. To address this, we adopt a counterfactual reasoning approach to propose a framework for fault detection and isolation (FDI) in perception systems. As opposed to relying on physical redundancy (i.e., having extra sensors), our approach utilizes analytical redundancy with counterfactual reasoning to construct perception reliability tests as causal outcomes influenced by system states and fault scenarios. Counterfactual reasoning generates reliability test results under hypothesized faults to update the belief over fault hypotheses. We derive both passive and active FDI methods. While the passive FDI can be achieved by belief updates, the active FDI approach is defined as a causal bandit problem, where we utilize Monte Carlo Tree Search (MCTS) with upper confidence bound (UCB) to find control inputs that maximize a detection and isolation metric, designated as Effective Information (EI). The mentioned metric quantifies the informativeness of control inputs for FDI. We demonstrate the approach in a robot exploration scenario, where a space robot performing vision-based navigation actively adjusts its attitude to increase EI and correctly isolate faults caused by sensor damage, dynamic scenes, and perceptual degradation.
>
---
#### [new 022] Pure Vision Language Action (VLA) Models: A Comprehensive Survey
- **分类: cs.RO; cs.AI**

- **简介: 该论文是一篇关于视觉语言动作模型（VLA）的综述，旨在系统梳理VLA的研究进展。论文总结了多种方法、应用场景及数据集，并分析了当前挑战与未来方向，推动通用机器人技术发展。**

- **链接: [http://arxiv.org/pdf/2509.19012v1](http://arxiv.org/pdf/2509.19012v1)**

> **作者:** Dapeng Zhang; Jin Sun; Chenghui Hu; Xiaoyan Wu; Zhenlong Yuan; Rui Zhou; Fei Shen; Qingguo Zhou
>
> **摘要:** The emergence of Vision Language Action (VLA) models marks a paradigm shift from traditional policy-based control to generalized robotics, reframing Vision Language Models (VLMs) from passive sequence generators into active agents for manipulation and decision-making in complex, dynamic environments. This survey delves into advanced VLA methods, aiming to provide a clear taxonomy and a systematic, comprehensive review of existing research. It presents a comprehensive analysis of VLA applications across different scenarios and classifies VLA approaches into several paradigms: autoregression-based, diffusion-based, reinforcement-based, hybrid, and specialized methods; while examining their motivations, core strategies, and implementations in detail. In addition, foundational datasets, benchmarks, and simulation platforms are introduced. Building on the current VLA landscape, the review further proposes perspectives on key challenges and future directions to advance research in VLA models and generalizable robotics. By synthesizing insights from over three hundred recent studies, this survey maps the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose VLA methods.
>
---
#### [new 023] Latent Action Pretraining Through World Modeling
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LAWSM框架，用于在无标签视频数据上通过世界建模进行模仿学习的自监督预训练。旨在解决现有视觉-语言-动作模型依赖大量标注数据、模型规模大、部署困难的问题。方法通过学习潜在动作表示，实现跨任务、环境和形态的有效迁移，并在基准测试中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.18428v1](http://arxiv.org/pdf/2509.18428v1)**

> **作者:** Bahey Tharwat; Yara Nasser; Ali Abouzeid; Ian Reid
>
> **摘要:** Vision-Language-Action (VLA) models have gained popularity for learning robotic manipulation tasks that follow language instructions. State-of-the-art VLAs, such as OpenVLA and $\pi_{0}$, were trained on large-scale, manually labeled action datasets collected through teleoperation. More recent approaches, including LAPA and villa-X, introduce latent action representations that enable unsupervised pretraining on unlabeled datasets by modeling abstract visual changes between frames. Although these methods have shown strong results, their large model sizes make deployment in real-world settings challenging. In this work, we propose LAWM, a model-agnostic framework to pretrain imitation learning models in a self-supervised way, by learning latent action representations from unlabeled video data through world modeling. These videos can be sourced from robot recordings or videos of humans performing actions with everyday objects. Our framework is designed to be effective for transferring across tasks, environments, and embodiments. It outperforms models trained with ground-truth robotics actions and similar pretraining methods on the LIBERO benchmark and real-world setup, while being significantly more efficient and practical for real-world settings.
>
---
#### [new 024] SOE: Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出SOE框架，用于机器人策略的样本高效自我改进。针对探索不足导致策略性能受限的问题，SOE通过在有效动作流形上进行安全、多样化的探索，提升策略表现。方法可与任意策略模型结合，适用于机器人操作任务。**

- **链接: [http://arxiv.org/pdf/2509.19292v1](http://arxiv.org/pdf/2509.19292v1)**

> **作者:** Yang Jin; Jun Lv; Han Xue; Wendi Chen; Chuan Wen; Cewu Lu
>
> **摘要:** Intelligent agents progress by continually refining their capabilities through actively exploring environments. Yet robot policies often lack sufficient exploration capability due to action mode collapse. Existing methods that encourage exploration typically rely on random perturbations, which are unsafe and induce unstable, erratic behaviors, thereby limiting their effectiveness. We propose Self-Improvement via On-Manifold Exploration (SOE), a framework that enhances policy exploration and improvement in robotic manipulation. SOE learns a compact latent representation of task-relevant factors and constrains exploration to the manifold of valid actions, ensuring safety, diversity, and effectiveness. It can be seamlessly integrated with arbitrary policy models as a plug-in module, augmenting exploration without degrading the base policy performance. Moreover, the structured latent space enables human-guided exploration, further improving efficiency and controllability. Extensive experiments in both simulation and real-world tasks demonstrate that SOE consistently outperforms prior methods, achieving higher task success rates, smoother and safer exploration, and superior sample efficiency. These results establish on-manifold exploration as a principled approach to sample-efficient policy self-improvement. Project website: https://ericjin2002.github.io/SOE
>
---
#### [new 025] RL-augmented Adaptive Model Predictive Control for Bipedal Locomotion over Challenging Terrain
- **分类: cs.RO**

- **简介: 该论文针对双足机器人在复杂地形中的运动控制问题，提出一种结合强化学习（RL）与模型预测控制（MPC）的方法。通过参数化动力学、摆动腿控制器和步态频率，提升MPC在崎岖和滑地面上的适应性与鲁棒性，并在仿真中验证了其优越性能。**

- **链接: [http://arxiv.org/pdf/2509.18466v1](http://arxiv.org/pdf/2509.18466v1)**

> **作者:** Junnosuke Kamohara; Feiyang Wu; Chinmayee Wamorkar; Seth Hutchinson; Ye Zhao
>
> **摘要:** Model predictive control (MPC) has demonstrated effectiveness for humanoid bipedal locomotion; however, its applicability in challenging environments, such as rough and slippery terrain, is limited by the difficulty of modeling terrain interactions. In contrast, reinforcement learning (RL) has achieved notable success in training robust locomotion policies over diverse terrain, yet it lacks guarantees of constraint satisfaction and often requires substantial reward shaping. Recent efforts in combining MPC and RL have shown promise of taking the best of both worlds, but they are primarily restricted to flat terrain or quadrupedal robots. In this work, we propose an RL-augmented MPC framework tailored for bipedal locomotion over rough and slippery terrain. Our method parametrizes three key components of single-rigid-body-dynamics-based MPC: system dynamics, swing leg controller, and gait frequency. We validate our approach through bipedal robot simulations in NVIDIA IsaacLab across various terrains, including stairs, stepping stones, and low-friction surfaces. Experimental results demonstrate that our RL-augmented MPC framework produces significantly more adaptive and robust behaviors compared to baseline MPC and RL.
>
---
#### [new 026] Do You Need Proprioceptive States in Visuomotor Policies?
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究基于模仿学习的视觉运动策略，针对策略过度依赖本体感知状态导致泛化能力差的问题，提出无需本体感知的状态无关策略。通过仅使用视觉输入，在多个机器人任务中实现了更强的空间泛化和数据效率。**

- **链接: [http://arxiv.org/pdf/2509.18644v1](http://arxiv.org/pdf/2509.18644v1)**

> **作者:** Juntu Zhao; Wenbo Lu; Di Zhang; Yufeng Liu; Yushen Liang; Tianluo Zhang; Yifeng Cao; Junyuan Xie; Yingdong Hu; Shengjie Wang; Junliang Guo; Dequan Wang; Yang Gao
>
> **备注:** Project page: https://statefreepolicy.github.io
>
> **摘要:** Imitation-learning-based visuomotor policies have been widely used in robot manipulation, where both visual observations and proprioceptive states are typically adopted together for precise control. However, in this study, we find that this common practice makes the policy overly reliant on the proprioceptive state input, which causes overfitting to the training trajectories and results in poor spatial generalization. On the contrary, we propose the State-free Policy, removing the proprioceptive state input and predicting actions only conditioned on visual observations. The State-free Policy is built in the relative end-effector action space, and should ensure the full task-relevant visual observations, here provided by dual wide-angle wrist cameras. Empirical results demonstrate that the State-free policy achieves significantly stronger spatial generalization than the state-based policy: in real-world tasks such as pick-and-place, challenging shirt-folding, and complex whole-body manipulation, spanning multiple robot embodiments, the average success rate improves from 0\% to 85\% in height generalization and from 6\% to 64\% in horizontal generalization. Furthermore, they also show advantages in data efficiency and cross-embodiment adaptation, enhancing their practicality for real-world deployment.
>
---
#### [new 027] PIE: Perception and Interaction Enhanced End-to-End Motion Planning for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出PIE框架，用于端到端自动驾驶运动规划。针对场景理解与交互预测的挑战，融合多模态感知与推理模块，优化轨迹生成。在NAVSIM基准上取得SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.18609v1](http://arxiv.org/pdf/2509.18609v1)**

> **作者:** Chengran Yuan; Zijian Lu; Zhanqi Zhang; Yimin Zhao; Zefan Huang; Shuo Sun; Jiawei Sun; Jiahui Li; Christina Dao Wen Lee; Dongen Li; Marcelo H. Ang Jr
>
> **摘要:** End-to-end motion planning is promising for simplifying complex autonomous driving pipelines. However, challenges such as scene understanding and effective prediction for decision-making continue to present substantial obstacles to its large-scale deployment. In this paper, we present PIE, a pioneering framework that integrates advanced perception, reasoning, and intention modeling to dynamically capture interactions between the ego vehicle and surrounding agents. It incorporates a bidirectional Mamba fusion that addresses data compression losses in multimodal fusion of camera and LiDAR inputs, alongside a novel reasoning-enhanced decoder integrating Mamba and Mixture-of-Experts to facilitate scene-compliant anchor selection and optimize adaptive trajectory inference. PIE adopts an action-motion interaction module to effectively utilize state predictions of surrounding agents to refine ego planning. The proposed framework is thoroughly validated on the NAVSIM benchmark. PIE, without using any ensemble and data augmentation techniques, achieves an 88.9 PDM score and 85.6 EPDM score, surpassing the performance of prior state-of-the-art methods. Comprehensive quantitative and qualitative analyses demonstrate that PIE is capable of reliably generating feasible and high-quality ego trajectories.
>
---
#### [new 028] Towards Robust LiDAR Localization: Deep Learning-based Uncertainty Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于LiDAR定位任务，旨在解决ICP算法在特征缺失或动态场景中定位误差大、不确定性估计不准的问题。提出了一种基于深度学习的框架，在无参考地图的情况下预估ICP配准误差协方差，提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.18954v1](http://arxiv.org/pdf/2509.18954v1)**

> **作者:** Minoo Dolatabadi; Fardin Ayar; Ehsan Javanmardi; Manabu Tsukada; Mahdi Javanmardi
>
> **摘要:** LiDAR-based localization and SLAM often rely on iterative matching algorithms, particularly the Iterative Closest Point (ICP) algorithm, to align sensor data with pre-existing maps or previous scans. However, ICP is prone to errors in featureless environments and dynamic scenes, leading to inaccurate pose estimation. Accurately predicting the uncertainty associated with ICP is crucial for robust state estimation but remains challenging, as existing approaches often rely on handcrafted models or simplified assumptions. Moreover, a few deep learning-based methods for localizability estimation either depend on a pre-built map, which may not always be available, or provide a binary classification of localizable versus non-localizable, which fails to properly model uncertainty. In this work, we propose a data-driven framework that leverages deep learning to estimate the registration error covariance of ICP before matching, even in the absence of a reference map. By associating each LiDAR scan with a reliable 6-DoF error covariance estimate, our method enables seamless integration of ICP within Kalman filtering, enhancing localization accuracy and robustness. Extensive experiments on the KITTI dataset demonstrate the effectiveness of our approach, showing that it accurately predicts covariance and, when applied to localization using a pre-built map or SLAM, reduces localization errors and improves robustness.
>
---
#### [new 029] Lang2Morph: Language-Driven Morphological Design of Robotic Hands
- **分类: cs.RO**

- **简介: 该论文提出Lang2Morph，一个基于大语言模型的语言驱动机械手形态设计框架。旨在解决传统方法依赖专家经验、计算成本高和灵活性差的问题，通过自然语言输入生成任务相关的可3D打印机械手结构。**

- **链接: [http://arxiv.org/pdf/2509.18937v1](http://arxiv.org/pdf/2509.18937v1)**

> **作者:** Yanyuan Qiao; Kieran Gilday; Yutong Xie; Josie Hughes
>
> **摘要:** Designing robotic hand morphologies for diverse manipulation tasks requires balancing dexterity, manufacturability, and task-specific functionality. While open-source frameworks and parametric tools support reproducible design, they still rely on expert heuristics and manual tuning. Automated methods using optimization are often compute-intensive, simulation-dependent, and rarely target dexterous hands. Large language models (LLMs), with their broad knowledge of human-object interactions and strong generative capabilities, offer a promising alternative for zero-shot design reasoning. In this paper, we present Lang2Morph, a language-driven pipeline for robotic hand design. It uses LLMs to translate natural-language task descriptions into symbolic structures and OPH-compatible parameters, enabling 3D-printable task-specific morphologies. The pipeline consists of: (i) Morphology Design, which maps tasks into semantic tags, structural grammars, and OPH-compatible parameters; and (ii) Selection and Refinement, which evaluates design candidates based on semantic alignment and size compatibility, and optionally applies LLM-guided refinement when needed. We evaluate Lang2Morph across varied tasks, and results show that our approach can generate diverse, task-relevant morphologies. To our knowledge, this is the first attempt to develop an LLM-based framework for task-conditioned robotic hand design.
>
---
#### [new 030] AD-VF: LLM-Automatic Differentiation Enables Fine-Tuning-Free Robot Planning from Formal Methods Feedback
- **分类: cs.RO; cs.FL**

- **简介: 该论文提出LAD-VF框架，用于无需微调的机器人规划。利用形式验证反馈优化提示，提升LLM在物理任务中的安全性和合规性，成功率达90%以上。**

- **链接: [http://arxiv.org/pdf/2509.18384v1](http://arxiv.org/pdf/2509.18384v1)**

> **作者:** Yunhao Yang; Junyuan Hong; Gabriel Jacob Perin; Zhiwen Fan; Li Yin; Zhangyang Wang; Ufuk Topcu
>
> **摘要:** Large language models (LLMs) can translate natural language instructions into executable action plans for robotics, autonomous driving, and other domains. Yet, deploying LLM-driven planning in the physical world demands strict adherence to safety and regulatory constraints, which current models often violate due to hallucination or weak alignment. Traditional data-driven alignment methods, such as Direct Preference Optimization (DPO), require costly human labeling, while recent formal-feedback approaches still depend on resource-intensive fine-tuning. In this paper, we propose LAD-VF, a fine-tuning-free framework that leverages formal verification feedback for automated prompt engineering. By introducing a formal-verification-informed text loss integrated with LLM-AutoDiff, LAD-VF iteratively refines prompts rather than model parameters. This yields three key benefits: (i) scalable adaptation without fine-tuning; (ii) compatibility with modular LLM architectures; and (iii) interpretable refinement via auditable prompts. Experiments in robot navigation and manipulation tasks demonstrate that LAD-VF substantially enhances specification compliance, improving success rates from 60% to over 90%. Our method thus presents a scalable and interpretable pathway toward trustworthy, formally-verified LLM-driven control systems.
>
---
#### [new 031] Reduced-Order Model-Guided Reinforcement Learning for Demonstration-Free Humanoid Locomotion
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ROM-GRL，一种无需示教数据的人形机器人行走强化学习框架。通过两阶段策略，先训练简化模型生成高效步态模板，再指导完整身体策略学习，解决了传统方法依赖运动捕捉或复杂奖励设计的问题，实现了稳定自然的步态控制。**

- **链接: [http://arxiv.org/pdf/2509.19023v1](http://arxiv.org/pdf/2509.19023v1)**

> **作者:** Shuai Liu; Meng Cheng Lau
>
> **备注:** 11 pages, 5 figures, 1 table, Computational Science Graduate Project
>
> **摘要:** We introduce Reduced-Order Model-Guided Reinforcement Learning (ROM-GRL), a two-stage reinforcement learning framework for humanoid walking that requires no motion capture data or elaborate reward shaping. In the first stage, a compact 4-DOF (four-degree-of-freedom) reduced-order model (ROM) is trained via Proximal Policy Optimization. This generates energy-efficient gait templates. In the second stage, those dynamically consistent trajectories guide a full-body policy trained with Soft Actor--Critic augmented by an adversarial discriminator, ensuring the student's five-dimensional gait feature distribution matches the ROM's demonstrations. Experiments at 1 meter-per-second and 4 meter-per-second show that ROM-GRL produces stable, symmetric gaits with substantially lower tracking error than a pure-reward baseline. By distilling lightweight ROM guidance into high-dimensional policies, ROM-GRL bridges the gap between reward-only and imitation-based locomotion methods, enabling versatile, naturalistic humanoid behaviors without any human demonstrations.
>
---
#### [new 032] Learning Obstacle Avoidance using Double DQN for Quadcopter Navigation
- **分类: cs.RO**

- **简介: 该论文属于无人机导航任务，旨在解决城市环境中四旋翼飞行器避障问题。作者提出使用双深度Q网络（Double DQN）结合深度相机数据，训练虚拟四旋翼机器人在模拟环境中自主避障导航。**

- **链接: [http://arxiv.org/pdf/2509.18734v1](http://arxiv.org/pdf/2509.18734v1)**

> **作者:** Nishant Doshi; Amey Sutvani; Sanket Gujar
>
> **摘要:** One of the challenges faced by Autonomous Aerial Vehicles is reliable navigation through urban environments. Factors like reduction in precision of Global Positioning System (GPS), narrow spaces and dynamically moving obstacles make the path planning of an aerial robot a complicated task. One of the skills required for the agent to effectively navigate through such an environment is to develop an ability to avoid collisions using information from onboard depth sensors. In this paper, we propose Reinforcement Learning of a virtual quadcopter robot agent equipped with a Depth Camera to navigate through a simulated urban environment.
>
---
#### [new 033] Distributionally Robust Safe Motion Planning with Contextual Information
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于安全运动规划任务，旨在解决动态障碍物场景下的碰撞避免问题。通过引入上下文信息和分布鲁棒性，利用条件核均嵌入构建分布模糊集，提出了一种更鲁棒的避撞约束方法，并在模拟中验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.18666v1](http://arxiv.org/pdf/2509.18666v1)**

> **作者:** Kaizer Rahaman; Simran Kumari; Ashish R. Hota
>
> **摘要:** We present a distributionally robust approach for collision avoidance by incorporating contextual information. Specifically, we embed the conditional distribution of future trajectory of the obstacle conditioned on the motion of the ego agent in a reproducing kernel Hilbert space (RKHS) via the conditional kernel mean embedding operator. Then, we define an ambiguity set containing all distributions whose embedding in the RKHS is within a certain distance from the empirical estimate of conditional mean embedding learnt from past data. Consequently, a distributionally robust collision avoidance constraint is formulated, and included in the receding horizon based motion planning formulation of the ego agent. Simulation results show that the proposed approach is more successful in avoiding collision compared to approaches that do not include contextual information and/or distributional robustness in their formulation in several challenging scenarios.
>
---
#### [new 034] LCMF: Lightweight Cross-Modality Mambaformer for Embodied Robotics VQA
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出LCMF框架，用于具身机器人视觉问答（VQA）任务，旨在解决多模态数据融合与计算效率问题。通过引入轻量级跨模态参数共享机制，实现高效异构信息融合，在保证性能的同时显著降低计算开销。**

- **链接: [http://arxiv.org/pdf/2509.18576v1](http://arxiv.org/pdf/2509.18576v1)**

> **作者:** Zeyi Kang; Liang He; Yanxin Zhang; Zuheng Ming; Kaixing Zhao
>
> **摘要:** Multimodal semantic learning plays a critical role in embodied intelligence, especially when robots perceive their surroundings, understand human instructions, and make intelligent decisions. However, the field faces technical challenges such as effective fusion of heterogeneous data and computational efficiency in resource-constrained environments. To address these challenges, this study proposes the lightweight LCMF cascaded attention framework, introducing a multi-level cross-modal parameter sharing mechanism into the Mamba module. By integrating the advantages of Cross-Attention and Selective parameter-sharing State Space Models (SSMs), the framework achieves efficient fusion of heterogeneous modalities and semantic complementary alignment. Experimental results show that LCMF surpasses existing multimodal baselines with an accuracy of 74.29% in VQA tasks and achieves competitive mid-tier performance within the distribution cluster of Large Language Model Agents (LLM Agents) in EQA video tasks. Its lightweight design achieves a 4.35-fold reduction in FLOPs relative to the average of comparable baselines while using only 166.51M parameters (image-text) and 219M parameters (video-text), providing an efficient solution for Human-Robot Interaction (HRI) applications in resource-constrained scenarios with strong multimodal decision generalization capabilities.
>
---
#### [new 035] DexSkin: High-Coverage Conformable Robotic Skin for Learning Contact-Rich Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出DexSkin，一种高覆盖率、可贴合的柔性电子皮肤，用于学习接触丰富的机器人操作任务。旨在解决机器人触觉感知不足的问题，通过在夹爪手指表面部署DexSkin，实现敏感、局部化触觉感知，并验证其在演示学习和强化学习中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.18830v1](http://arxiv.org/pdf/2509.18830v1)**

> **作者:** Suzannah Wistreich; Baiyu Shi; Stephen Tian; Samuel Clarke; Michael Nath; Chengyi Xu; Zhenan Bao; Jiajun Wu
>
> **备注:** Accepted to CoRL 2025
>
> **摘要:** Human skin provides a rich tactile sensing stream, localizing intentional and unintentional contact events over a large and contoured region. Replicating these tactile sensing capabilities for dexterous robotic manipulation systems remains a longstanding challenge. In this work, we take a step towards this goal by introducing DexSkin. DexSkin is a soft, conformable capacitive electronic skin that enables sensitive, localized, and calibratable tactile sensing, and can be tailored to varying geometries. We demonstrate its efficacy for learning downstream robotic manipulation by sensorizing a pair of parallel jaw gripper fingers, providing tactile coverage across almost the entire finger surfaces. We empirically evaluate DexSkin's capabilities in learning challenging manipulation tasks that require sensing coverage across the entire surface of the fingers, such as reorienting objects in hand and wrapping elastic bands around boxes, in a learning-from-demonstration framework. We then show that, critically for data-driven approaches, DexSkin can be calibrated to enable model transfer across sensor instances, and demonstrate its applicability to online reinforcement learning on real robots. Our results highlight DexSkin's suitability and practicality for learning real-world, contact-rich manipulation. Please see our project webpage for videos and visualizations: https://dex-skin.github.io/.
>
---
#### [new 036] Spectral Signature Mapping from RGB Imagery for Terrain-Aware Navigation
- **分类: cs.RO**

- **简介: 该论文提出RS-Net，一种从RGB图像预测光谱特征的神经网络，用于地形分类与摩擦估计，解决视觉相似但材质不同的地面识别问题，支持轮式和四足机器人在户外环境中的物理感知导航。**

- **链接: [http://arxiv.org/pdf/2509.19105v1](http://arxiv.org/pdf/2509.19105v1)**

> **作者:** Sarvesh Prajapati; Ananya Trivedi; Nathaniel Hanson; Bruce Maxwell; Taskin Padir
>
> **备注:** 8 pages, 10 figures, submitted to Robotic Computing & Communication
>
> **摘要:** Successful navigation in outdoor environments requires accurate prediction of the physical interactions between the robot and the terrain. To this end, several methods rely on geometric or semantic labels to classify traversable surfaces. However, such labels cannot distinguish visually similar surfaces that differ in material properties. Spectral sensors enable inference of material composition from surface reflectance measured across multiple wavelength bands. Although spectral sensing is gaining traction in robotics, widespread deployment remains constrained by the need for custom hardware integration, high sensor costs, and compute-intensive processing pipelines. In this paper, we present RGB Image to Spectral Signature Neural Network (RS-Net), a deep neural network designed to bridge the gap between the accessibility of RGB sensing and the rich material information provided by spectral data. RS-Net predicts spectral signatures from RGB patches, which we map to terrain labels and friction coefficients. The resulting terrain classifications are integrated into a sampling-based motion planner for a wheeled robot operating in outdoor environments. Likewise, the friction estimates are incorporated into a contact-force-based MPC for a quadruped robot navigating slippery surfaces. Thus, we introduce a framework that learns the task-relevant physical property once during training and thereafter relies solely on RGB sensing at test time. The code is available at https://github.com/prajapatisarvesh/RS-Net.
>
---
#### [new 037] The Case for Negative Data: From Crash Reports to Counterfactuals for Reasonable Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对自动驾驶系统在安全边界决策能力不足的问题，提出利用事故报告进行学习。通过将事故描述转化为统一的场景-动作表示，并结合反事实推理，提升系统在风险情境下的决策合理性与校准能力。属于自动驾驶决策优化任务。**

- **链接: [http://arxiv.org/pdf/2509.18626v1](http://arxiv.org/pdf/2509.18626v1)**

> **作者:** Jay Patrikar; Apoorva Sharma; Sushant Veer; Boyi Li; Sebastian Scherer; Marco Pavone
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Learning-based autonomous driving systems are trained mostly on incident-free data, offering little guidance near safety-performance boundaries. Real crash reports contain precisely the contrastive evidence needed, but they are hard to use: narratives are unstructured, third-person, and poorly grounded to sensor views. We address these challenges by normalizing crash narratives to ego-centric language and converting both logs and crashes into a unified scene-action representation suitable for retrieval. At decision time, our system adjudicates proposed actions by retrieving relevant precedents from this unified index; an agentic counterfactual extension proposes plausible alternatives, retrieves for each, and reasons across outcomes before deciding. On a nuScenes benchmark, precedent retrieval substantially improves calibration, with recall on contextually preferred actions rising from 24% to 53%. The counterfactual variant preserves these gains while sharpening decisions near risk.
>
---
#### [new 038] TacEva: A Performance Evaluation Framework For Vision-Based Tactile Sensors
- **分类: cs.RO**

- **简介: 该论文提出TacEva，一个用于视觉触觉传感器性能评估的框架，旨在解决现有传感器因结构和机制差异导致的性能难以量化的问题。通过定义指标与实验流程，实现对传感器的系统评价与优化指导。**

- **链接: [http://arxiv.org/pdf/2509.19037v1](http://arxiv.org/pdf/2509.19037v1)**

> **作者:** Qingzheng Cong; Steven Oh; Wen Fan; Shan Luo; Kaspar Althoefer; Dandan Zhang
>
> **备注:** 14 pages, 8 figures. Equal contribution: Qingzheng Cong, Steven Oh, Wen Fan. Corresponding author: Dandan Zhang (d.zhang17@imperial.ac.uk). Additional resources at http://stevenoh2003.github.io/TacEva/
>
> **摘要:** Vision-Based Tactile Sensors (VBTSs) are widely used in robotic tasks because of the high spatial resolution they offer and their relatively low manufacturing costs. However, variations in their sensing mechanisms, structural dimension, and other parameters lead to significant performance disparities between existing VBTSs. This makes it challenging to optimize them for specific tasks, as both the initial choice and subsequent fine-tuning are hindered by the lack of standardized metrics. To address this issue, TacEva is introduced as a comprehensive evaluation framework for the quantitative analysis of VBTS performance. The framework defines a set of performance metrics that capture key characteristics in typical application scenarios. For each metric, a structured experimental pipeline is designed to ensure consistent and repeatable quantification. The framework is applied to multiple VBTSs with distinct sensing mechanisms, and the results demonstrate its ability to provide a thorough evaluation of each design and quantitative indicators for each performance dimension. This enables researchers to pre-select the most appropriate VBTS on a task by task basis, while also offering performance-guided insights into the optimization of VBTS design. A list of existing VBTS evaluation methods and additional evaluations can be found on our website: https://stevenoh2003.github.io/TacEva/
>
---
#### [new 039] Growing with Your Embodied Agent: A Human-in-the-Loop Lifelong Code Generation Framework for Long-Horizon Manipulation Skills
- **分类: cs.RO**

- **简介: 该论文提出一种人机协作的终身代码生成框架，用于解决长时序机器人操作任务。针对现有方法在可重用技能学习和长序列规划中的不足，通过引入外部记忆与检索增强生成技术，提升任务成功率与修正效率。**

- **链接: [http://arxiv.org/pdf/2509.18597v1](http://arxiv.org/pdf/2509.18597v1)**

> **作者:** Yuan Meng; Zhenguo Sun; Max Fest; Xukun Li; Zhenshan Bing; Alois Knoll
>
> **备注:** upload 9 main page - v1
>
> **摘要:** Large language models (LLMs)-based code generation for robotic manipulation has recently shown promise by directly translating human instructions into executable code, but existing methods remain noisy, constrained by fixed primitives and limited context windows, and struggle with long-horizon tasks. While closed-loop feedback has been explored, corrected knowledge is often stored in improper formats, restricting generalization and causing catastrophic forgetting, which highlights the need for learning reusable skills. Moreover, approaches that rely solely on LLM guidance frequently fail in extremely long-horizon scenarios due to LLMs' limited reasoning capability in the robotic domain, where such issues are often straightforward for humans to identify. To address these challenges, we propose a human-in-the-loop framework that encodes corrections into reusable skills, supported by external memory and Retrieval-Augmented Generation with a hint mechanism for dynamic reuse. Experiments on Ravens, Franka Kitchen, and MetaWorld, as well as real-world settings, show that our framework achieves a 0.93 success rate (up to 27% higher than baselines) and a 42% efficiency improvement in correction rounds. It can robustly solve extremely long-horizon tasks such as "build a house", which requires planning over 20 primitives.
>
---
#### [new 040] The Landform Contextual Mesh: Automatically Fusing Surface and Orbital Terrain for Mars 2020
- **分类: cs.RO**

- **简介: 该论文提出一种地形上下文网格方法，用于融合火星2020任务的地面和轨道数据，生成交互式3D地形。旨在支持科学家进行战术规划与公众访问，实现了自动构建并部署到相关工具与网站。**

- **链接: [http://arxiv.org/pdf/2509.18330v1](http://arxiv.org/pdf/2509.18330v1)**

> **作者:** Marsette Vona
>
> **摘要:** The Landform contextual mesh fuses 2D and 3D data from up to thousands of Mars 2020 rover images, along with orbital elevation and color maps from Mars Reconnaissance Orbiter, into an interactive 3D terrain visualization. Contextual meshes are built automatically for each rover location during mission ground data system processing, and are made available to mission scientists for tactical and strategic planning in the Advanced Science Targeting Tool for Robotic Operations (ASTTRO). A subset of them are also deployed to the "Explore with Perseverance" public access website.
>
---
#### [new 041] Robotic Skill Diversification via Active Mutation of Reward Functions in Reinforcement Learning During a Liquid Pouring Task
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究在强化学习中通过奖励函数主动变异实现机器人技能多样化，任务为液体倾倒。提出基于高斯噪声的奖励函数变异框架，结合准确度、时间和努力等关键项，在仿真环境中验证了多种技能行为的生成，如容器清洁和混合等新技能。**

- **链接: [http://arxiv.org/pdf/2509.18463v1](http://arxiv.org/pdf/2509.18463v1)**

> **作者:** Jannick van Buuren; Roberto Giglio; Loris Roveda; Luka Peternel
>
> **摘要:** This paper explores how deliberate mutations of reward function in reinforcement learning can produce diversified skill variations in robotic manipulation tasks, examined with a liquid pouring use case. To this end, we developed a new reward function mutation framework that is based on applying Gaussian noise to the weights of the different terms in the reward function. Inspired by the cost-benefit tradeoff model from human motor control, we designed the reward function with the following key terms: accuracy, time, and effort. The study was performed in a simulation environment created in NVIDIA Isaac Sim, and the setup included Franka Emika Panda robotic arm holding a glass with a liquid that needed to be poured into a container. The reinforcement learning algorithm was based on Proximal Policy Optimization. We systematically explored how different configurations of mutated weights in the rewards function would affect the learned policy. The resulting policies exhibit a wide range of behaviours: from variations in execution of the originally intended pouring task to novel skills useful for unexpected tasks, such as container rim cleaning, liquid mixing, and watering. This approach offers promising directions for robotic systems to perform diversified learning of specific tasks, while also potentially deriving meaningful skills for future tasks.
>
---
#### [new 042] VGGT-DP: Generalizable Robot Control via Vision Foundation Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出VGGT-DP，一种基于视觉基础模型的通用机器人控制框架。针对现有方法在视觉编码和空间理解上的不足，结合3D感知模型与本体感受反馈，提升空间定位和闭环控制能力，适用于复杂操作任务。**

- **链接: [http://arxiv.org/pdf/2509.18778v1](http://arxiv.org/pdf/2509.18778v1)**

> **作者:** Shijia Ge; Yinxin Zhang; Shuzhao Xie; Weixiang Zhang; Mingcai Zhou; Zhi Wang
>
> **备注:** submitted to AAAI 2026
>
> **摘要:** Visual imitation learning frameworks allow robots to learn manipulation skills from expert demonstrations. While existing approaches mainly focus on policy design, they often neglect the structure and capacity of visual encoders, limiting spatial understanding and generalization. Inspired by biological vision systems, which rely on both visual and proprioceptive cues for robust control, we propose VGGT-DP, a visuomotor policy framework that integrates geometric priors from a pretrained 3D perception model with proprioceptive feedback. We adopt the Visual Geometry Grounded Transformer (VGGT) as the visual encoder and introduce a proprioception-guided visual learning strategy to align perception with internal robot states, improving spatial grounding and closed-loop control. To reduce inference latency, we design a frame-wise token reuse mechanism that compacts multi-view tokens into an efficient spatial representation. We further apply random token pruning to enhance policy robustness and reduce overfitting. Experiments on challenging MetaWorld tasks show that VGGT-DP significantly outperforms strong baselines such as DP and DP3, particularly in precision-critical and long-horizon scenarios.
>
---
#### [new 043] Number Adaptive Formation Flight Planning via Affine Deformable Guidance in Narrow Environments
- **分类: cs.RO**

- **简介: 该论文研究无人机编队规划任务，旨在解决狭窄环境中无人机数量变化时编队难以收敛的问题。提出基于可变形虚拟结构的方法，结合分区分配与轨迹优化，实现自适应编队调整与避障，验证了方法在复杂环境中的有效性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.18636v1](http://arxiv.org/pdf/2509.18636v1)**

> **作者:** Yuan Zhou; Jialiang Hou; Guangtong Xu; Fei Gao
>
> **摘要:** Formation maintenance with varying number of drones in narrow environments hinders the convergence of planning to the desired configurations. To address this challenge, this paper proposes a formation planning method guided by Deformable Virtual Structures (DVS) with continuous spatiotemporal transformation. Firstly, to satisfy swarm safety distance and preserve formation shape filling integrity for irregular formation geometries, we employ Lloyd algorithm for uniform $\underline{PA}$rtitioning and Hungarian algorithm for $\underline{AS}$signment (PAAS) in DVS. Subsequently, a spatiotemporal trajectory involving DVS is planned using primitive-based path search and nonlinear trajectory optimization. The DVS trajectory achieves adaptive transitions with respect to a varying number of drones while ensuring adaptability to narrow environments through affine transformation. Finally, each agent conducts distributed trajectory planning guided by desired spatiotemporal positions within the DVS, while incorporating collision avoidance and dynamic feasibility requirements. Our method enables up to 15\% of swarm numbers to join or leave in cluttered environments while rapidly restoring the desired formation shape in simulation. Compared to cutting-edge formation planning method, we demonstrate rapid formation recovery capacity and environmental adaptability. Real-world experiments validate the effectiveness and resilience of our formation planning method.
>
---
#### [new 044] End-to-End Crop Row Navigation via LiDAR-Based Deep Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出一种基于LiDAR和深度强化学习的端到端作物行导航方法，旨在解决GNSS不可靠等农业环境下的导航问题。通过体素降采样减少数据量，并在仿真中训练和验证策略，实现了高成功率的作物行跟踪。**

- **链接: [http://arxiv.org/pdf/2509.18608v1](http://arxiv.org/pdf/2509.18608v1)**

> **作者:** Ana Luiza Mineiro; Francisco Affonso; Marcelo Becker
>
> **备注:** Accepted to the 22nd International Conference on Advanced Robotics (ICAR 2025). 7 pages
>
> **摘要:** Reliable navigation in under-canopy agricultural environments remains a challenge due to GNSS unreliability, cluttered rows, and variable lighting. To address these limitations, we present an end-to-end learning-based navigation system that maps raw 3D LiDAR data directly to control commands using a deep reinforcement learning policy trained entirely in simulation. Our method includes a voxel-based downsampling strategy that reduces LiDAR input size by 95.83%, enabling efficient policy learning without relying on labeled datasets or manually designed control interfaces. The policy was validated in simulation, achieving a 100% success rate in straight-row plantations and showing a gradual decline in performance as row curvature increased, tested across varying sinusoidal frequencies and amplitudes.
>
---
#### [new 045] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人操作策略的模拟到现实迁移问题，提出一种统一的联合训练框架，通过学习领域不变且任务相关的特征空间，利用少量真实数据和大量模拟数据提升策略泛化能力，解决了仿真与现实差距导致的迁移困难。**

- **链接: [http://arxiv.org/pdf/2509.18631v1](http://arxiv.org/pdf/2509.18631v1)**

> **作者:** Shuo Cheng; Liqian Ma; Zhenyang Chen; Ajay Mandlekar; Caelan Garrett; Danfei Xu
>
> **摘要:** Behavior cloning has shown promise for robot manipulation, but real-world demonstrations are costly to acquire at scale. While simulated data offers a scalable alternative, particularly with advances in automated demonstration generation, transferring policies to the real world is hampered by various simulation and real domain gaps. In this work, we propose a unified sim-and-real co-training framework for learning generalizable manipulation policies that primarily leverages simulation and only requires a few real-world demonstrations. Central to our approach is learning a domain-invariant, task-relevant feature space. Our key insight is that aligning the joint distributions of observations and their corresponding actions across domains provides a richer signal than aligning observations (marginals) alone. We achieve this by embedding an Optimal Transport (OT)-inspired loss within the co-training framework, and extend this to an Unbalanced OT framework to handle the imbalance between abundant simulation data and limited real-world examples. We validate our method on challenging manipulation tasks, showing it can leverage abundant simulation data to achieve up to a 30% improvement in the real-world success rate and even generalize to scenarios seen only in simulation.
>
---
#### [new 046] Application Management in C-ITS: Orchestrating Demand-Driven Deployments and Reconfigurations
- **分类: cs.RO; cs.MA; cs.SE**

- **简介: 该论文研究C-ITS中的应用管理任务，旨在解决大规模动态环境中资源高效利用的问题。提出基于Kubernetes和ROS 2的框架，实现微服务的按需部署、更新与扩展，提升系统自动化与响应能力。**

- **链接: [http://arxiv.org/pdf/2509.18793v1](http://arxiv.org/pdf/2509.18793v1)**

> **作者:** Lukas Zanger; Bastian Lampe; Lennart Reiher; Lutz Eckstein
>
> **备注:** 7 pages, 2 figures, 2 tables; Accepted to be published as part of the 2025 IEEE International Conference on Intelligent Transportation Systems (ITSC 2025), Gold Coast, Australia, November 18-21, 2025
>
> **摘要:** Vehicles are becoming increasingly automated and interconnected, enabling the formation of cooperative intelligent transport systems (C-ITS) and the use of offboard services. As a result, cloud-native techniques, such as microservices and container orchestration, play an increasingly important role in their operation. However, orchestrating applications in a large-scale C-ITS poses unique challenges due to the dynamic nature of the environment and the need for efficient resource utilization. In this paper, we present a demand-driven application management approach that leverages cloud-native techniques - specifically Kubernetes - to address these challenges. Taking into account the demands originating from different entities within the C-ITS, the approach enables the automation of processes, such as deployment, reconfiguration, update, upgrade, and scaling of microservices. Executing these processes on demand can, for example, reduce computing resource consumption and network traffic. A demand may include a request for provisioning an external supporting service, such as a collective environment model. The approach handles changing and new demands by dynamically reconciling them through our proposed application management framework built on Kubernetes and the Robot Operating System (ROS 2). We demonstrate the operation of our framework in the C-ITS use case of collective environment perception and make the source code of the prototypical framework publicly available at https://github.com/ika-rwth-aachen/application_manager .
>
---
#### [new 047] Proactive-reactive detection and mitigation of intermittent faults in robot swarms
- **分类: cs.RO; cs.MA; cs.SY; eess.SY**

- **简介: 该论文研究机器人集群中间歇性故障的检测与缓解问题。针对现有方法难以应对临时网络拓扑下的间歇性故障，提出基于自组织备份路径和分布式共识的主动-反应策略，提升故障检测准确性与系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.19246v1](http://arxiv.org/pdf/2509.19246v1)**

> **作者:** Sinan Oğuz; Emanuele Garone; Marco Dorigo; Mary Katherine Heinrich
>
> **摘要:** Intermittent faults are transient errors that sporadically appear and disappear. Although intermittent faults pose substantial challenges to reliability and coordination, existing studies of fault tolerance in robot swarms focus instead on permanent faults. One reason for this is that intermittent faults are prohibitively difficult to detect in the fully self-organized ad-hoc networks typical of robot swarms, as their network topologies are transient and often unpredictable. However, in the recently introduced self-organizing nervous systems (SoNS) approach, robot swarms are able to self-organize persistent network structures for the first time, easing the problem of detecting intermittent faults. To address intermittent faults in robot swarms that have persistent networks, we propose a novel proactive-reactive strategy to detection and mitigation, based on self-organized backup layers and distributed consensus in a multiplex network. Proactively, the robots self-organize dynamic backup paths before faults occur, adapting to changes in the primary network topology and the robots' relative positions. Reactively, robots use one-shot likelihood ratio tests to compare information received along different paths in the multiplex network, enabling early fault detection. Upon detection, communication is temporarily rerouted in a self-organized way, until the detected fault resolves. We validate the approach in representative scenarios of faulty positional data occurring during formation control, demonstrating that intermittent faults are prevented from disrupting convergence to desired formations, with high fault detection accuracy and low rates of false positives.
>
---
#### [new 048] Residual Off-Policy RL for Finetuning Behavior Cloning Policies
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出一种残差离线强化学习方法，结合行为克隆（BC）与RL的优势。通过使用BC策略作为基础，利用高效的离线RL学习轻量级残差修正，解决高自由度系统在现实世界中训练RL策略的样本效率和安全性问题，在仿真和真实环境中均取得优异效果。**

- **链接: [http://arxiv.org/pdf/2509.19301v1](http://arxiv.org/pdf/2509.19301v1)**

> **作者:** Lars Ankile; Zhenyu Jiang; Rocky Duan; Guanya Shi; Pieter Abbeel; Anusha Nagabandi
>
> **摘要:** Recent advances in behavior cloning (BC) have enabled impressive visuomotor control policies. However, these approaches are limited by the quality of human demonstrations, the manual effort required for data collection, and the diminishing returns from increasing offline data. In comparison, reinforcement learning (RL) trains an agent through autonomous interaction with the environment and has shown remarkable success in various domains. Still, training RL policies directly on real-world robots remains challenging due to sample inefficiency, safety concerns, and the difficulty of learning from sparse rewards for long-horizon tasks, especially for high-degree-of-freedom (DoF) systems. We present a recipe that combines the benefits of BC and RL through a residual learning framework. Our approach leverages BC policies as black-box bases and learns lightweight per-step residual corrections via sample-efficient off-policy RL. We demonstrate that our method requires only sparse binary reward signals and can effectively improve manipulation policies on high-degree-of-freedom (DoF) systems in both simulation and the real world. In particular, we demonstrate, to the best of our knowledge, the first successful real-world RL training on a humanoid robot with dexterous hands. Our results demonstrate state-of-the-art performance in various vision-based tasks, pointing towards a practical pathway for deploying RL in the real world. Project website: https://residual-offpolicy-rl.github.io
>
---
#### [new 049] ManipForce: Force-Guided Policy Learning with Frequency-Aware Representation for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文针对接触密集型操作任务（如精密装配），提出ManipForce系统与FMT模型，通过捕捉高频率力-扭矩和视觉数据，实现跨模态融合的策略学习，提升操作精度与成功率。**

- **链接: [http://arxiv.org/pdf/2509.19047v1](http://arxiv.org/pdf/2509.19047v1)**

> **作者:** Geonhyup Lee; Yeongjin Lee; Kangmin Kim; Seongju Lee; Sangjun Noh; Seunghyeok Back; Kyoobin Lee
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Contact-rich manipulation tasks such as precision assembly require precise control of interaction forces, yet existing imitation learning methods rely mainly on vision-only demonstrations. We propose ManipForce, a handheld system designed to capture high-frequency force-torque (F/T) and RGB data during natural human demonstrations for contact-rich manipulation. Building on these demonstrations, we introduce the Frequency-Aware Multimodal Transformer (FMT). FMT encodes asynchronous RGB and F/T signals using frequency- and modality-aware embeddings and fuses them via bi-directional cross-attention within a transformer diffusion policy. Through extensive experiments on six real-world contact-rich manipulation tasks - such as gear assembly, box flipping, and battery insertion - FMT trained on ManipForce demonstrations achieves robust performance with an average success rate of 83% across all tasks, substantially outperforming RGB-only baselines. Ablation and sampling-frequency analyses further confirm that incorporating high-frequency F/T data and cross-modal integration improves policy performance, especially in tasks demanding high precision and stable contact.
>
---
#### [new 050] SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones
- **分类: cs.RO**

- **简介: 该论文提出SINGER，一种用于无人机的端到端视觉-语言导航策略。针对开放世界中缺乏大规模演示、实时控制和定位模块的问题，SINGER结合仿真生成、轨迹规划和轻量策略训练，实现了零样本迁移到真实环境，提升了目标追踪性能。**

- **链接: [http://arxiv.org/pdf/2509.18610v1](http://arxiv.org/pdf/2509.18610v1)**

> **作者:** Maximilian Adang; JunEn Low; Ola Shorinwa; Mac Schwager
>
> **摘要:** Large vision-language models have driven remarkable progress in open-vocabulary robot policies, e.g., generalist robot manipulation policies, that enable robots to complete complex tasks specified in natural language. Despite these successes, open-vocabulary autonomous drone navigation remains an unsolved challenge due to the scarcity of large-scale demonstrations, real-time control demands of drones for stabilization, and lack of reliable external pose estimation modules. In this work, we present SINGER for language-guided autonomous drone navigation in the open world using only onboard sensing and compute. To train robust, open-vocabulary navigation policies, SINGER leverages three central components: (i) a photorealistic language-embedded flight simulator with minimal sim-to-real gap using Gaussian Splatting for efficient data generation, (ii) an RRT-inspired multi-trajectory generation expert for collision-free navigation demonstrations, and these are used to train (iii) a lightweight end-to-end visuomotor policy for real-time closed-loop control. Through extensive hardware flight experiments, we demonstrate superior zero-shot sim-to-real transfer of our policy to unseen environments and unseen language-conditioned goal objects. When trained on ~700k-1M observation action pairs of language conditioned visuomotor data and deployed on hardware, SINGER outperforms a velocity-controlled semantic guidance baseline by reaching the query 23.33% more on average, and maintains the query in the field of view 16.67% more on average, with 10% fewer collisions.
>
---
#### [new 051] Eva-VLA: Evaluating Vision-Language-Action Models' Robustness Under Real-World Physical Variations
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Eva-VLA框架，用于评估视觉-语言-动作模型在真实物理变化下的鲁棒性。针对3D物体变换、光照和对抗补丁等挑战，设计连续优化方法，揭示当前模型在复杂场景中的高失败率，推动鲁棒性研究。**

- **链接: [http://arxiv.org/pdf/2509.18953v1](http://arxiv.org/pdf/2509.18953v1)**

> **作者:** Hanqing Liu; Jiahuan Long; Junqi Wu; Jiacheng Hou; Huili Tang; Tingsong Jiang; Weien Zhou; Wen Yao
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as promising solutions for robotic manipulation, yet their robustness to real-world physical variations remains critically underexplored. To bridge this gap, we propose Eva-VLA, the first unified framework that systematically evaluates the robustness of VLA models by transforming discrete physical variations into continuous optimization problems. However, comprehensively assessing VLA robustness presents two key challenges: (1) how to systematically characterize diverse physical variations encountered in real-world deployments while maintaining evaluation reproducibility, and (2) how to discover worst-case scenarios without prohibitive real-world data collection costs efficiently. To address the first challenge, we decompose real-world variations into three critical domains: object 3D transformations that affect spatial reasoning, illumination variations that challenge visual perception, and adversarial patches that disrupt scene understanding. For the second challenge, we introduce a continuous black-box optimization framework that transforms discrete physical variations into parameter optimization, enabling systematic exploration of worst-case scenarios. Extensive experiments on state-of-the-art OpenVLA models across multiple benchmarks reveal alarming vulnerabilities: all variation types trigger failure rates exceeding 60%, with object transformations causing up to 97.8% failure in long-horizon tasks. Our findings expose critical gaps between controlled laboratory success and unpredictable deployment readiness, while the Eva-VLA framework provides a practical pathway for hardening VLA-based robotic manipulation models against real-world deployment challenges.
>
---
#### [new 052] MagiClaw: A Dual-Use, Vision-Based Soft Gripper for Bridging the Human Demonstration to Robotic Deployment Gap
- **分类: cs.RO**

- **简介: 该论文提出MagiClaw，一种双用途视觉软夹爪，用于弥合人类演示与机器人执行间的领域差距。通过统一硬件设计和多模态感知，实现直观数据采集与策略部署，推动操作技能迁移。**

- **链接: [http://arxiv.org/pdf/2509.19169v1](http://arxiv.org/pdf/2509.19169v1)**

> **作者:** Tianyu Wu; Xudong Han; Haoran Sun; Zishang Zhang; Bangchao Huang; Chaoyang Song; Fang Wan
>
> **备注:** 8 pages, 4 figures, accepted to Data@CoRL2025 Workshop
>
> **摘要:** The transfer of manipulation skills from human demonstration to robotic execution is often hindered by a "domain gap" in sensing and morphology. This paper introduces MagiClaw, a versatile two-finger end-effector designed to bridge this gap. MagiClaw functions interchangeably as both a handheld tool for intuitive data collection and a robotic end-effector for policy deployment, ensuring hardware consistency and reliability. Each finger incorporates a Soft Polyhedral Network (SPN) with an embedded camera, enabling vision-based estimation of 6-DoF forces and contact deformation. This proprioceptive data is fused with exteroceptive environmental sensing from an integrated iPhone, which provides 6D pose, RGB video, and LiDAR-based depth maps. Through a custom iOS application, MagiClaw streams synchronized, multi-modal data for real-time teleoperation, offline policy learning, and immersive control via mixed-reality interfaces. We demonstrate how this unified system architecture lowers the barrier to collecting high-fidelity, contact-rich datasets and accelerates the development of generalizable manipulation policies. Please refer to the iOS app at https://apps.apple.com/cn/app/magiclaw/id6661033548 for further details.
>
---
#### [new 053] Human-Interpretable Uncertainty Explanations for Point Cloud Registration
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对点云配准任务，解决因传感器噪声和遮挡等问题导致的不确定性问题。提出GP-CA方法，通过量化并解释不确定性来源，结合主动学习提升鲁棒性，并在真实机器人实验中验证了其有效性与高效性。**

- **链接: [http://arxiv.org/pdf/2509.18786v1](http://arxiv.org/pdf/2509.18786v1)**

> **作者:** Johannes A. Gaus; Loris Schneider; Yitian Shi; Jongseok Lee; Rania Rayyes; Rudolph Triebel
>
> **摘要:** In this paper, we address the point cloud registration problem, where well-known methods like ICP fail under uncertainty arising from sensor noise, pose-estimation errors, and partial overlap due to occlusion. We develop a novel approach, Gaussian Process Concept Attribution (GP-CA), which not only quantifies registration uncertainty but also explains it by attributing uncertainty to well-known sources of errors in registration problems. Our approach leverages active learning to discover new uncertainty sources in the wild by querying informative instances. We validate GP-CA on three publicly available datasets and in our real-world robot experiment. Extensive ablations substantiate our design choices. Our approach outperforms other state-of-the-art methods in terms of runtime, high sample-efficiency with active learning, and high accuracy. Our real-world experiment clearly demonstrates its applicability. Our video also demonstrates that GP-CA enables effective failure-recovery behaviors, yielding more robust robotic perception.
>
---
#### [new 054] Spatial Envelope MPC: High Performance Driving without a Reference
- **分类: cs.RO**

- **简介: 该论文提出一种无需预设参考轨迹的基于包络模型预测控制（MPC）框架，用于高性能自动驾驶。针对传统参考轨迹方法在极限驾驶场景下的局限性，引入高效动力学模型与强化学习结合优化技术，实现实时安全规划与控制，适用于赛车、避障等复杂任务。**

- **链接: [http://arxiv.org/pdf/2509.18506v1](http://arxiv.org/pdf/2509.18506v1)**

> **作者:** Siyuan Yu; Congkai Shen; Yufei Xi; James Dallas; Michael Thompson; John Subosits; Hiroshi Yasuda; Tulga Ersal
>
> **摘要:** This paper presents a novel envelope based model predictive control (MPC) framework designed to enable autonomous vehicles to handle high performance driving across a wide range of scenarios without a predefined reference. In high performance autonomous driving, safe operation at the vehicle's dynamic limits requires a real time planning and control framework capable of accounting for key vehicle dynamics and environmental constraints when following a predefined reference trajectory is suboptimal or even infeasible. State of the art planning and control frameworks, however, are predominantly reference based, which limits their performance in such situations. To address this gap, this work first introduces a computationally efficient vehicle dynamics model tailored for optimization based control and a continuously differentiable mathematical formulation that accurately captures the entire drivable envelope. This novel model and formulation allow for the direct integration of dynamic feasibility and safety constraints into a unified planning and control framework, thereby removing the necessity for predefined references. The challenge of envelope planning, which refers to maximally approximating the safe drivable area, is tackled by combining reinforcement learning with optimization techniques. The framework is validated through both simulations and real world experiments, demonstrating its high performance across a variety of tasks, including racing, emergency collision avoidance and off road navigation. These results highlight the framework's scalability and broad applicability across a diverse set of scenarios.
>
---
#### [new 055] VLN-Zero: Rapid Exploration and Cache-Enabled Neurosymbolic Vision-Language Planning for Zero-Shot Transfer in Robot Navigation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出VLN-Zero，用于机器人零样本视觉-语言导航任务。针对现有方法泛化能力差和计算效率低的问题，设计了两阶段框架：探索阶段构建符号场景图，部署阶段结合神经符号规划与缓存执行模块，实现高效、可扩展的未知环境导航。**

- **链接: [http://arxiv.org/pdf/2509.18592v1](http://arxiv.org/pdf/2509.18592v1)**

> **作者:** Neel P. Bhatt; Yunhao Yang; Rohan Siva; Pranay Samineni; Daniel Milan; Zhangyang Wang; Ufuk Topcu
>
> **备注:** Codebase, datasets, and videos for VLN-Zero are available at: https://vln-zero.github.io/
>
> **摘要:** Rapid adaptation in unseen environments is essential for scalable real-world autonomy, yet existing approaches rely on exhaustive exploration or rigid navigation policies that fail to generalize. We present VLN-Zero, a two-phase vision-language navigation framework that leverages vision-language models to efficiently construct symbolic scene graphs and enable zero-shot neurosymbolic navigation. In the exploration phase, structured prompts guide VLM-based search toward informative and diverse trajectories, yielding compact scene graph representations. In the deployment phase, a neurosymbolic planner reasons over the scene graph and environmental observations to generate executable plans, while a cache-enabled execution module accelerates adaptation by reusing previously computed task-location trajectories. By combining rapid exploration, symbolic reasoning, and cache-enabled execution, the proposed framework overcomes the computational inefficiency and poor generalization of prior vision-language navigation methods, enabling robust and scalable decision-making in unseen environments. VLN-Zero achieves 2x higher success rate compared to state-of-the-art zero-shot models, outperforms most fine-tuned baselines, and reaches goal locations in half the time with 55% fewer VLM calls on average compared to state-of-the-art models across diverse environments. Codebase, datasets, and videos for VLN-Zero are available at: https://vln-zero.github.io/.
>
---
#### [new 056] Policy Gradient with Self-Attention for Model-Free Distributed Nonlinear Multi-Agent Games
- **分类: eess.SY; cs.MA; cs.RO; cs.SY**

- **简介: 该论文研究多智能体非线性博弈中的分布式策略学习问题，提出基于自注意力机制的策略梯度方法，在无需模型信息的情况下，实现动态通信结构下的高效协同控制。**

- **链接: [http://arxiv.org/pdf/2509.18371v1](http://arxiv.org/pdf/2509.18371v1)**

> **作者:** Eduardo Sebastián; Maitrayee Keskar; Eeman Iqbal; Eduardo Montijano; Carlos Sagüés; Nikolay Atanasov
>
> **摘要:** Multi-agent games in dynamic nonlinear settings are challenging due to the time-varying interactions among the agents and the non-stationarity of the (potential) Nash equilibria. In this paper we consider model-free games, where agent transitions and costs are observed without knowledge of the transition and cost functions that generate them. We propose a policy gradient approach to learn distributed policies that follow the communication structure in multi-team games, with multiple agents per team. Our formulation is inspired by the structure of distributed policies in linear quadratic games, which take the form of time-varying linear feedback gains. In the nonlinear case, we model the policies as nonlinear feedback gains, parameterized by self-attention layers to account for the time-varying multi-agent communication topology. We demonstrate that our distributed policy gradient approach achieves strong performance in several settings, including distributed linear and nonlinear regulation, and simulated and real multi-robot pursuit-and-evasion games.
>
---
#### [new 057] A Fast Initialization Method for Neural Network Controllers: A Case Study of Image-based Visual Servoing Control for the multicopter Interception
- **分类: eess.SY; cs.LG; cs.RO; cs.SY**

- **简介: 该论文提出一种神经网络控制器快速初始化方法，用于图像视觉伺服控制的多旋翼拦截任务。针对强化学习训练数据大、收敛慢的问题，通过构建满足稳定性条件的数据集进行初始训练，提升控制策略稳定性与性能。**

- **链接: [http://arxiv.org/pdf/2509.19110v1](http://arxiv.org/pdf/2509.19110v1)**

> **作者:** Chenxu Ke; Congling Tian; Kaichen Xu; Ye Li; Lingcong Bao
>
> **摘要:** Reinforcement learning-based controller design methods often require substantial data in the initial training phase. Moreover, the training process tends to exhibit strong randomness and slow convergence. It often requires considerable time or high computational resources. Another class of learning-based method incorporates Lyapunov stability theory to obtain a control policy with stability guarantees. However, these methods generally require an initially stable neural network control policy at the beginning of training. Evidently, a stable neural network controller can not only serve as an initial policy for reinforcement learning, allowing the training to focus on improving controller performance, but also act as an initial state for learning-based Lyapunov control methods. Although stable controllers can be designed using traditional control theory, designers still need to have a great deal of control design knowledge to address increasingly complicated control problems. The proposed neural network rapid initialization method in this paper achieves the initial training of the neural network control policy by constructing datasets that conform to the stability conditions based on the system model. Furthermore, using the image-based visual servoing control for multicopter interception as a case study, simulations and experiments were conducted to validate the effectiveness and practical performance of the proposed method. In the experiment, the trained control policy attains a final interception velocity of 15 m/s.
>
---
#### [new 058] MMCD: Multi-Modal Collaborative Decision-Making for Connected Autonomy with Knowledge Distillation
- **分类: cs.AI; cs.MA; cs.RO**

- **简介: 该论文提出MMCD框架，用于联网自动驾驶的多模态协同决策。针对传感器失效或车辆缺失导致的数据不全问题，采用知识蒸馏方法提升决策鲁棒性，实验表明可提高驾驶安全性20.7%。**

- **链接: [http://arxiv.org/pdf/2509.18198v1](http://arxiv.org/pdf/2509.18198v1)**

> **作者:** Rui Liu; Zikang Wang; Peng Gao; Yu Shen; Pratap Tokekar; Ming Lin
>
> **摘要:** Autonomous systems have advanced significantly, but challenges persist in accident-prone environments where robust decision-making is crucial. A single vehicle's limited sensor range and obstructed views increase the likelihood of accidents. Multi-vehicle connected systems and multi-modal approaches, leveraging RGB images and LiDAR point clouds, have emerged as promising solutions. However, existing methods often assume the availability of all data modalities and connected vehicles during both training and testing, which is impractical due to potential sensor failures or missing connected vehicles. To address these challenges, we introduce a novel framework MMCD (Multi-Modal Collaborative Decision-making) for connected autonomy. Our framework fuses multi-modal observations from ego and collaborative vehicles to enhance decision-making under challenging conditions. To ensure robust performance when certain data modalities are unavailable during testing, we propose an approach based on cross-modal knowledge distillation with a teacher-student model structure. The teacher model is trained with multiple data modalities, while the student model is designed to operate effectively with reduced modalities. In experiments on $\textit{connected autonomous driving with ground vehicles}$ and $\textit{aerial-ground vehicles collaboration}$, our method improves driving safety by up to ${\it 20.7}\%$, surpassing the best-existing baseline in detecting potential accidents and making safe driving decisions. More information can be found on our website https://ruiiu.github.io/mmcd.
>
---
#### [new 059] Reversible Kalman Filter for state estimation with Manifold
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文提出一种用于流形上状态估计的可逆卡尔曼滤波算法，旨在提升现有卡尔曼滤波在合成数据上的精度评估能力。通过改进数值特性，消除小速度假设限制，使精度仅受传感器噪声影响，并扩展至水下轨迹重建等实际应用场景。**

- **链接: [http://arxiv.org/pdf/2509.18224v1](http://arxiv.org/pdf/2509.18224v1)**

> **作者:** Svyatoslav Covanov; Cedric Pradalier
>
> **摘要:** This work introduces an algorithm for state estimation on manifolds within the framework of the Kalman filter. Its primary objective is to provide a methodology enabling the evaluation of the precision of existing Kalman filter variants with arbitrary accuracy on synthetic data, something that, to the best of our knowledge, has not been addressed in prior work. To this end, we develop a new filter that exhibits favorable numerical properties, thereby correcting the divergences observed in previous Kalman filter variants. In this formulation, the achievable precision is no longer constrained by the small-velocity assumption and is determined solely by sensor noise. In addition, this new filter assumes high precision on the sensors, which, in real scenarios require a detection step that we define heuristically, allowing one to extend this approach to scenarios, using either a 9-axis IMU or a combination of odometry, accelerometer, and pressure sensors. The latter configuration is designed for the reconstruction of trajectories in underwater environments.
>
---
#### [new 060] Guaranteed Robust Nonlinear MPC via Disturbance Feedback
- **分类: math.OC; cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种鲁棒非线性MPC方法，通过扰动反馈分解模型误差，保证系统在干扰下的安全性与稳定性，并实现高效求解。**

- **链接: [http://arxiv.org/pdf/2509.18760v1](http://arxiv.org/pdf/2509.18760v1)**

> **作者:** Antoine P. Leeman; Johannes Köhler; Melanie N. Zeilinger
>
> **备注:** Code: https://github.com/antoineleeman/robust-nonlinear-mpc
>
> **摘要:** Robots must satisfy safety-critical state and input constraints despite disturbances and model mismatch. We introduce a robust model predictive control (RMPC) formulation that is fast, scalable, and compatible with real-time implementation. Our formulation guarantees robust constraint satisfaction, input-to-state stability (ISS) and recursive feasibility. The key idea is to decompose the uncertain nonlinear system into (i) a nominal nonlinear dynamic model, (ii) disturbance-feedback controllers, and (iii) bounds on the model error. These components are optimized jointly using sequential convex programming. The resulting convex subproblems are solved efficiently using a recent disturbance-feedback MPC solver. The approach is validated across multiple dynamics, including a rocket-landing problem with steerable thrust. An open-source implementation is available at https://github.com/antoineleeman/robust-nonlinear-mpc.
>
---
#### [new 061] OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出OrthoLoC，用于无人机6自由度定位与校准。针对无GNSS环境下的高精度定位问题，构建了包含多模态图像的大规模数据集，并引入AdHoP技术提升匹配精度，降低定位误差。**

- **链接: [http://arxiv.org/pdf/2509.18350v1](http://arxiv.org/pdf/2509.18350v1)**

> **作者:** Oussema Dhaouadi; Riccardo Marin; Johannes Meier; Jacques Kaiser; Daniel Cremers
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Accurate visual localization from aerial views is a fundamental problem with applications in mapping, large-area inspection, and search-and-rescue operations. In many scenarios, these systems require high-precision localization while operating with limited resources (e.g., no internet connection or GNSS/GPS support), making large image databases or heavy 3D models impractical. Surprisingly, little attention has been given to leveraging orthographic geodata as an alternative paradigm, which is lightweight and increasingly available through free releases by governmental authorities (e.g., the European Union). To fill this gap, we propose OrthoLoC, the first large-scale dataset comprising 16,425 UAV images from Germany and the United States with multiple modalities. The dataset addresses domain shifts between UAV imagery and geospatial data. Its paired structure enables fair benchmarking of existing solutions by decoupling image retrieval from feature matching, allowing isolated evaluation of localization and calibration performance. Through comprehensive evaluation, we examine the impact of domain shifts, data resolutions, and covisibility on localization accuracy. Finally, we introduce a refinement technique called AdHoP, which can be integrated with any feature matcher, improving matching by up to 95% and reducing translation error by up to 63%. The dataset and code are available at: https://deepscenario.github.io/OrthoLoC.
>
---
#### [new 062] Event-guided 3D Gaussian Splatting for Dynamic Human and Scene Reconstruction
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出一种基于事件相机的动态人体与场景重建方法，利用3D高斯溅射技术，通过事件引导损失解决快速运动模糊问题，无需外部人像掩码即可实现高质量的人体-场景联合建模。**

- **链接: [http://arxiv.org/pdf/2509.18566v1](http://arxiv.org/pdf/2509.18566v1)**

> **作者:** Xiaoting Yin; Hao Shi; Kailun Yang; Jiajun Zhai; Shangwei Guo; Lin Wang; Kaiwei Wang
>
> **摘要:** Reconstructing dynamic humans together with static scenes from monocular videos remains difficult, especially under fast motion, where RGB frames suffer from motion blur. Event cameras exhibit distinct advantages, e.g., microsecond temporal resolution, making them a superior sensing choice for dynamic human reconstruction. Accordingly, we present a novel event-guided human-scene reconstruction framework that jointly models human and scene from a single monocular event camera via 3D Gaussian Splatting. Specifically, a unified set of 3D Gaussians carries a learnable semantic attribute; only Gaussians classified as human undergo deformation for animation, while scene Gaussians stay static. To combat blur, we propose an event-guided loss that matches simulated brightness changes between consecutive renderings with the event stream, improving local fidelity in fast-moving regions. Our approach removes the need for external human masks and simplifies managing separate Gaussian sets. On two benchmark datasets, ZJU-MoCap-Blur and MMHPSD-Blur, it delivers state-of-the-art human-scene reconstruction, with notable gains over strong baselines in PSNR/SSIM and reduced LPIPS, especially for high-speed subjects.
>
---
#### [new 063] Conversational Orientation Reasoning: Egocentric-to-Allocentric Navigation with Multimodal Chain-of-Thought
- **分类: cs.LG; cs.AI; cs.CL; cs.RO**

- **简介: 该论文聚焦于对话导航中的自体-他体方向推理任务，旨在解决中文对话中将自体方位（如“我的右边”）转换为绝对方向（N/E/S/W）的问题。提出了COR数据集和MCoT框架，结合语音识别与坐标信息，通过结构化三步推理实现高精度方向推断，适用于资源受限环境。**

- **链接: [http://arxiv.org/pdf/2509.18200v1](http://arxiv.org/pdf/2509.18200v1)**

> **作者:** Yu Ti Huang
>
> **摘要:** Conversational agents must translate egocentric utterances (e.g., "on my right") into allocentric orientations (N/E/S/W). This challenge is particularly critical in indoor or complex facilities where GPS signals are weak and detailed maps are unavailable. While chain-of-thought (CoT) prompting has advanced reasoning in language and vision tasks, its application to multimodal spatial orientation remains underexplored. We introduce Conversational Orientation Reasoning (COR), a new benchmark designed for Traditional Chinese conversational navigation projected from real-world environments, addressing egocentric-to-allocentric reasoning in non-English and ASR-transcribed scenarios. We propose a multimodal chain-of-thought (MCoT) framework, which integrates ASR-transcribed speech with landmark coordinates through a structured three-step reasoning process: (1) extracting spatial relations, (2) mapping coordinates to absolute directions, and (3) inferring user orientation. A curriculum learning strategy progressively builds these capabilities on Taiwan-LLM-13B-v2.0-Chat, a mid-sized model representative of resource-constrained settings. Experiments show that MCoT achieves 100% orientation accuracy on clean transcripts and 98.1% with ASR transcripts, substantially outperforming unimodal and non-structured baselines. Moreover, MCoT demonstrates robustness under noisy conversational conditions, including ASR recognition errors and multilingual code-switching. The model also maintains high accuracy in cross-domain evaluation and resilience to linguistic variation, domain shift, and referential ambiguity. These findings highlight the potential of structured MCoT spatial reasoning as a path toward interpretable and resource-efficient embodied navigation.
>
---
#### [new 064] An Extended Kalman Filter for Systems with Infinite-Dimensional Measurements
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文研究离散非线性随机系统的状态估计问题，针对有限维状态和无限维测量的情况，提出扩展卡尔曼滤波器（EKF）。重点解决视觉定位中的测量建模问题，并通过实验证明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.18749v1](http://arxiv.org/pdf/2509.18749v1)**

> **作者:** Maxwell M. Varley; Timothy L. Molloy; Girish N. Nair
>
> **备注:** 8 pages
>
> **摘要:** This article examines state estimation in discrete-time nonlinear stochastic systems with finite-dimensional states and infinite-dimensional measurements, motivated by real-world applications such as vision-based localization and tracking. We develop an extended Kalman filter (EKF) for real-time state estimation, with the measurement noise modeled as an infinite-dimensional random field. When applied to vision-based state estimation, the measurement Jacobians required to implement the EKF are shown to correspond to image gradients. This result provides a novel system-theoretic justification for the use of image gradients as features for vision-based state estimation, contrasting with their (often heuristic) introduction in many computer-vision pipelines. We demonstrate the practical utility of the EKF on a public real-world dataset involving the localization of an aerial drone using video from a downward-facing monocular camera. The EKF is shown to outperform VINS-MONO, an established visual-inertial odometry algorithm, in some cases achieving mean squared error reductions of up to an order of magnitude.
>
---
#### [new 065] Dual Iterative Learning Control for Multiple-Input Multiple-Output Dynamics with Validation in Robotic Systems
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文提出了一种用于多输入多输出（MIMO）系统的双迭代学习控制（DILC）方法，旨在无需先验模型知识和手动调参的情况下，实现运动任务的自主跟踪与模型学习。**

- **链接: [http://arxiv.org/pdf/2509.18723v1](http://arxiv.org/pdf/2509.18723v1)**

> **作者:** Jan-Hendrik Ewering; Alessandro Papa; Simon F. G. Ehlers; Thomas Seel; Michael Meindl
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Solving motion tasks autonomously and accurately is a core ability for intelligent real-world systems. To achieve genuine autonomy across multiple systems and tasks, key challenges include coping with unknown dynamics and overcoming the need for manual parameter tuning, which is especially crucial in complex Multiple-Input Multiple-Output (MIMO) systems. This paper presents MIMO Dual Iterative Learning Control (DILC), a novel data-driven iterative learning scheme for simultaneous tracking control and model learning, without requiring any prior system knowledge or manual parameter tuning. The method is designed for repetitive MIMO systems and integrates seamlessly with established iterative learning control methods. We provide monotonic convergence conditions for both reference tracking error and model error in linear time-invariant systems. The DILC scheme -- rapidly and autonomously -- solves various motion tasks in high-fidelity simulations of an industrial robot and in multiple nonlinear real-world MIMO systems, without requiring model knowledge or manually tuning the algorithm. In our experiments, many reference tracking tasks are solved within 10-20 trials, and even complex motions are learned in less than 100 iterations. We believe that, because of its rapid and autonomous learning capabilities, DILC has the potential to serve as an efficient building block within complex learning frameworks for intelligent real-world systems.
>
---
## 更新

#### [replaced 001] Operator Splitting Covariance Steering for Safe Stochastic Nonlinear Control
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2411.11211v2](http://arxiv.org/pdf/2411.11211v2)**

> **作者:** Akash Ratheesh; Vincent Pacelli; Augustinos D. Saravanos; Evangelos A. Theodorou
>
> **摘要:** This paper presents a novel algorithm for solving distribution steering problems featuring nonlinear dynamics and chance constraints. Covariance steering (CS) is an emerging methodology in stochastic optimal control that poses constraints on the first two moments of the state distribution -- thereby being more tractable than full distributional control. Nevertheless, a significant limitation of current approaches for solving nonlinear CS problems, such as sequential convex programming (SCP), is that they often generate infeasible or poor results due to the large number of constraints. In this paper, we address these challenges, by proposing an operator splitting CS approach that temporarily decouples the full problem into subproblems that can be solved in parallel. This relaxation does not require intermediate iterates to satisfy all constraints simultaneously prior to convergence, which enhances exploration and improves feasibility in such non-convex settings. Simulation results across a variety of robotics applications verify the ability of the proposed method to find better solutions even under stricter safety constraints than standard SCP. Finally, the applicability of our framework on real systems is also confirmed through hardware demonstrations
>
---
#### [replaced 002] Constrained Style Learning from Imperfect Demonstrations under Task Optimality
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.09371v2](http://arxiv.org/pdf/2507.09371v2)**

> **作者:** Kehan Wen; Chenhao Li; Junzhe He; Marco Hutter
>
> **备注:** This paper has been accepted to CoRL 2025
>
> **摘要:** Learning from demonstration has proven effective in robotics for acquiring natural behaviors, such as stylistic motions and lifelike agility, particularly when explicitly defining style-oriented reward functions is challenging. Synthesizing stylistic motions for real-world tasks usually requires balancing task performance and imitation quality. Existing methods generally depend on expert demonstrations closely aligned with task objectives. However, practical demonstrations are often incomplete or unrealistic, causing current methods to boost style at the expense of task performance. To address this issue, we propose formulating the problem as a constrained Markov Decision Process (CMDP). Specifically, we optimize a style-imitation objective with constraints to maintain near-optimal task performance. We introduce an adaptively adjustable Lagrangian multiplier to guide the agent to imitate demonstrations selectively, capturing stylistic nuances without compromising task performance. We validate our approach across multiple robotic platforms and tasks, demonstrating both robust task performance and high-fidelity style learning. On ANYmal-D hardware we show a 14.5% drop in mechanical energy and a more agile gait pattern, showcasing real-world benefits.
>
---
#### [replaced 003] Stratified Topological Autonomy for Long-Range Coordination (STALC)
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.10475v2](http://arxiv.org/pdf/2503.10475v2)**

> **作者:** Cora A. Duggan; Adam Goertz; Adam Polevoy; Mark Gonzales; Kevin C. Wolfe; Bradley Woosley; John G. Rogers III; Joseph Moore
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** In this paper, we present Stratified Topological Autonomy for Long-Range Coordination (STALC), a hierarchical planning approach for coordinated multi-robot maneuvering in real-world environments with significant inter-robot spatial and temporal dependencies. At its core, STALC consists of a multi-robot graph-based planner which combines a topological graph with a novel, computationally efficient mixed-integer programming formulation to generate highly-coupled multi-robot plans in seconds. To enable autonomous planning across different spatial and temporal scales, we construct our graphs so that they capture connectivity between free-space regions and other problem-specific features, such as traversability or risk. We then use receding-horizon planners to achieve local collision avoidance and formation control. To evaluate our approach, we consider a multi-robot reconnaissance scenario where robots must autonomously coordinate to navigate through an environment while minimizing the risk of detection by observers. Through simulation-based experiments, we show that our approach is able to scale to address complex multi-robot planning scenarios. Through hardware experiments, we demonstrate our ability to generate graphs from real-world data and successfully plan across the entire hierarchy to achieve shared objectives.
>
---
#### [replaced 004] GeoAware-VLA: Implicit Geometry Aware Vision-Language-Action Model
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.14117v2](http://arxiv.org/pdf/2509.14117v2)**

> **作者:** Ali Abouzeid; Malak Mansour; Zezhou Sun; Dezhen Song
>
> **备注:** Under Review
>
> **摘要:** Vision-Language-Action (VLA) models often fail to generalize to novel camera viewpoints, a limitation stemming from their difficulty in inferring robust 3D geometry from 2D images. We introduce GeoAware-VLA, a simple yet effective approach that enhances viewpoint invariance by integrating strong geometric priors into the vision backbone. Instead of training a visual encoder or relying on explicit 3D data, we leverage a frozen, pretrained geometric vision model as a feature extractor. A trainable projection layer then adapts these geometrically-rich features for the policy decoder, relieving it of the burden of learning 3D consistency from scratch. Through extensive evaluations on LIBERO benchmark subsets, we show GeoAware-VLA achieves substantial improvements in zero-shot generalization to novel camera poses, boosting success rates by over 2x in simulation. Crucially, these benefits translate to the physical world; our model shows a significant performance gain on a real robot, especially when evaluated from unseen camera angles. Our approach proves effective across both continuous and discrete action spaces, highlighting that robust geometric grounding is a key component for creating more generalizable robotic agents.
>
---
#### [replaced 005] L2M-Reg: Building-level Uncertainty-aware Registration of Outdoor LiDAR Point Clouds and Semantic 3D City Models
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2509.16832v2](http://arxiv.org/pdf/2509.16832v2)**

> **作者:** Ziyang Xu; Benedikt Schwab; Yihui Yang; Thomas H. Kolbe; Christoph Holst
>
> **备注:** Submitted to the ISPRS Journal of Photogrammetry and Remote Sensing
>
> **摘要:** Accurate registration between LiDAR (Light Detection and Ranging) point clouds and semantic 3D city models is a fundamental topic in urban digital twinning and a prerequisite for downstream tasks, such as digital construction, change detection and model refinement. However, achieving accurate LiDAR-to-Model registration at individual building level remains challenging, particularly due to the generalization uncertainty in semantic 3D city models at the Level of Detail 2 (LoD2). This paper addresses this gap by proposing L2M-Reg, a plane-based fine registration method that explicitly accounts for model uncertainty. L2M-Reg consists of three key steps: establishing reliable plane correspondence, building a pseudo-plane-constrained Gauss-Helmert model, and adaptively estimating vertical translation. Experiments on three real-world datasets demonstrate that L2M-Reg is both more accurate and computationally efficient than existing ICP-based and plane-based methods. Overall, L2M-Reg provides a novel building-level solution regarding LiDAR-to-Model registration when model uncertainty is present.
>
---
#### [replaced 006] V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.18053v2](http://arxiv.org/pdf/2509.18053v2)**

> **作者:** Hsu-kuang Chiu; Ryo Hachiuma; Chien-Yi Wang; Yu-Chiang Frank Wang; Min-Hung Chen; Stephen F. Smith
>
> **备注:** Our project website: https://eddyhkchiu.github.io/v2vgot.github.io/
>
> **摘要:** Current state-of-the-art autonomous vehicles could face safety-critical situations when their local sensors are occluded by large nearby objects on the road. Vehicle-to-vehicle (V2V) cooperative autonomous driving has been proposed as a means of addressing this problem, and one recently introduced framework for cooperative autonomous driving has further adopted an approach that incorporates a Multimodal Large Language Model (MLLM) to integrate cooperative perception and planning processes. However, despite the potential benefit of applying graph-of-thoughts reasoning to the MLLM, this idea has not been considered by previous cooperative autonomous driving research. In this paper, we propose a novel graph-of-thoughts framework specifically designed for MLLM-based cooperative autonomous driving. Our graph-of-thoughts includes our proposed novel ideas of occlusion-aware perception and planning-aware prediction. We curate the V2V-GoT-QA dataset and develop the V2V-GoT model for training and testing the cooperative driving graph-of-thoughts. Our experimental results show that our method outperforms other baselines in cooperative perception, prediction, and planning tasks. Our project website: https://eddyhkchiu.github.io/v2vgot.github.io/ .
>
---
#### [replaced 007] HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.16757v2](http://arxiv.org/pdf/2509.16757v2)**

> **作者:** Haoyang Weng; Yitang Li; Nikhil Sobanbabu; Zihan Wang; Zhengyi Luo; Tairan He; Deva Ramanan; Guanya Shi
>
> **备注:** website: hdmi-humanoid.github.io
>
> **摘要:** Enabling robust whole-body humanoid-object interaction (HOI) remains challenging due to motion data scarcity and the contact-rich nature. We present HDMI (HumanoiD iMitation for Interaction), a simple and general framework that learns whole-body humanoid-object interaction skills directly from monocular RGB videos. Our pipeline (i) extracts and retargets human and object trajectories from unconstrained videos to build structured motion datasets, (ii) trains a reinforcement learning (RL) policy to co-track robot and object states with three key designs: a unified object representation, a residual action space, and a general interaction reward, and (iii) zero-shot deploys the RL policies on real humanoid robots. Extensive sim-to-real experiments on a Unitree G1 humanoid demonstrate the robustness and generality of our approach: HDMI achieves 67 consecutive door traversals and successfully performs 6 distinct loco-manipulation tasks in the real world and 14 tasks in simulation. Our results establish HDMI as a simple and general framework for acquiring interactive humanoid skills from human videos.
>
---
#### [replaced 008] Low-pass sampling in Model Predictive Path Integral Control
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2503.11717v2](http://arxiv.org/pdf/2503.11717v2)**

> **作者:** Piotr Kicki
>
> **摘要:** Model Predictive Path Integral (MPPI) control is a widely used sampling-based approach for real-time control, offering flexibility in handling arbitrary dynamics and cost functions. However, the original MPPI suffers from high-frequency noise in the sampled control trajectories, leading to actuator wear and inefficient exploration. In this work, we introduce Low-Pass Model Predictive Path Integral Control (LP-MPPI), which integrates low-pass filtering into the sampling process to eliminate detrimental high-frequency components and improve the effectiveness of the control trajectories exploration. Unlike prior approaches, LP-MPPI provides direct and interpretable control over the frequency spectrum of sampled trajectories, enhancing sampling efficiency and control smoothness. Through extensive evaluations in Gymnasium environments, simulated quadruped locomotion, and real-world F1TENTH autonomous racing, we demonstrate that LP-MPPI consistently outperforms state-of-the-art MPPI variants, achieving significant performance improvements while reducing control signal chattering.
>
---
#### [replaced 009] RAVE: End-to-end Hierarchical Visual Localization with Rasterized and Vectorized HD map
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.00862v2](http://arxiv.org/pdf/2503.00862v2)**

> **作者:** Jinyu Miao; Tuopu Wen; Kun Jiang; Kangan Qian; Zheng Fu; Yunlong Wang; Zhihuang Zhang; Mengmeng Yang; Jin Huang; Zhihua Zhong; Diange Yang
>
> **备注:** 16 pages, 10 figures, 6 tables
>
> **摘要:** Accurate localization serves as an important component in autonomous driving systems. Traditional rule-based localization involves many standalone modules, which is theoretically fragile and requires costly hyperparameter tuning, therefore sacrificing the accuracy and generalization. In this paper, we propose an end-to-end visual localization system, RAVE, in which the surrounding images are associated with the HD map data to estimate pose. To ensure high-quality observations for localization, a low-rank flow-based prior fusion module (FLORA) is developed to incorporate misaligned map prior into the perceived BEV features. Pursuing a balance among efficiency, interpretability, and accuracy, a hierarchical localization module is proposed, which efficiently estimates poses through a decoupled BEV neural matching-based pose solver (DEMA) using rasterized HD map, and then refines the estimation through a Transformer-based pose regressor (POET) using vectorized HD map. The experimental results demonstrate that our method can perform robust and accurate localization under varying environmental conditions while running efficiently.
>
---
#### [replaced 010] EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.17430v2](http://arxiv.org/pdf/2509.17430v2)**

> **作者:** Gunjan Chhablani; Xiaomeng Ye; Muhammad Zubair Irshad; Zsolt Kira
>
> **备注:** 16 pages, 18 figures, paper accepted at ICCV, 2025
>
> **摘要:** The field of Embodied AI predominantly relies on simulation for training and evaluation, often using either fully synthetic environments that lack photorealism or high-fidelity real-world reconstructions captured with expensive hardware. As a result, sim-to-real transfer remains a major challenge. In this paper, we introduce EmbodiedSplat, a novel approach that personalizes policy training by efficiently capturing the deployment environment and fine-tuning policies within the reconstructed scenes. Our method leverages 3D Gaussian Splatting (GS) and the Habitat-Sim simulator to bridge the gap between realistic scene capture and effective training environments. Using iPhone-captured deployment scenes, we reconstruct meshes via GS, enabling training in settings that closely approximate real-world conditions. We conduct a comprehensive analysis of training strategies, pre-training datasets, and mesh reconstruction techniques, evaluating their impact on sim-to-real predictivity in real-world scenarios. Experimental results demonstrate that agents fine-tuned with EmbodiedSplat outperform both zero-shot baselines pre-trained on large-scale real-world datasets (HM3D) and synthetically generated datasets (HSSD), achieving absolute success rate improvements of 20% and 40% on real-world Image Navigation task. Moreover, our approach yields a high sim-vs-real correlation (0.87-0.97) for the reconstructed meshes, underscoring its effectiveness in adapting policies to diverse environments with minimal effort. Project page: https://gchhablani.github.io/embodied-splat.
>
---
#### [replaced 011] Occlusion-Aware Consistent Model Predictive Control for Robot Navigation in Occluded Obstacle-Dense Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.04563v3](http://arxiv.org/pdf/2503.04563v3)**

> **作者:** Minzhe Zheng; Lei Zheng; Lei Zhu; Jun Ma
>
> **摘要:** Ensuring safety and motion consistency for robot navigation in occluded, obstacle-dense environments is a critical challenge. In this context, this study presents an occlusion-aware Consistent Model Predictive Control (CMPC) strategy. To account for the occluded obstacles, it incorporates adjustable risk regions that represent their potential future locations. Subsequently, dynamic risk boundary constraints are developed online to ensure safety. The CMPC then constructs multiple locally optimal trajectory branches (each tailored to different risk regions) to strike a balance between safety and performance. A shared consensus segment is generated to ensure smooth transitions between branches without significant velocity fluctuations, further preserving motion consistency. To facilitate high computational efficiency and ensure coordination across local trajectories, we use the alternating direction method of multipliers (ADMM) to decompose the CMPC into manageable sub-problems for parallel solving. The proposed strategy is validated through simulations and real-world experiments on an Ackermann-steering robot platform. The results demonstrate the effectiveness of the proposed CMPC strategy through comparisons with baseline approaches in occluded, obstacle-dense environments.
>
---
#### [replaced 012] RoboSeek: You Need to Interact with Your Objects
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.17783v2](http://arxiv.org/pdf/2509.17783v2)**

> **作者:** Yibo Peng; Jiahao Yang; Shenhao Yan; Ziyu Huang; Shuang Li; Shuguang Cui; Yiming Zhao; Yatong Han
>
> **摘要:** Optimizing and refining action execution through exploration and interaction is a promising way for robotic manipulation. However, practical approaches to interaction-driven robotic learning are still underexplored, particularly for long-horizon tasks where sequential decision-making, physical constraints, and perceptual uncertainties pose significant challenges. Motivated by embodied cognition theory, we propose RoboSeek, a framework for embodied action execution that leverages interactive experience to accomplish manipulation tasks. RoboSeek optimizes prior knowledge from high-level perception models through closed-loop training in simulation and achieves robust real-world execution via a real2sim2real transfer pipeline. Specifically, we first replicate real-world environments in simulation using 3D reconstruction to provide visually and physically consistent environments, then we train policies in simulation using reinforcement learning and the cross-entropy method leveraging visual priors. The learned policies are subsequently deployed on real robotic platforms for execution. RoboSeek is hardware-agnostic and is evaluated on multiple robotic platforms across eight long-horizon manipulation tasks involving sequential interactions, tool use, and object handling. Our approach achieves an average success rate of 79%, significantly outperforming baselines whose success rates remain below 50%, highlighting its generalization and robustness across tasks and platforms. Experimental results validate the effectiveness of our training framework in complex, dynamic real-world settings and demonstrate the stability of the proposed real2sim2real transfer mechanism, paving the way for more generalizable embodied robotic learning. Project Page: https://russderrick.github.io/Roboseek/
>
---
#### [replaced 013] Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.05985v2](http://arxiv.org/pdf/2506.05985v2)**

> **作者:** Yuheng Lei; Sitong Mao; Shunbo Zhou; Hongyuan Zhang; Xuelong Li; Ping Luo
>
> **摘要:** A generalist agent must continuously learn and adapt throughout its lifetime, achieving efficient forward transfer while minimizing catastrophic forgetting. Previous work within the dominant pretrain-then-finetune paradigm has explored parameter-efficient fine-tuning for single-task adaptation, effectively steering a frozen pretrained model with a small number of parameters. However, in the context of lifelong learning, these methods rely on the impractical assumption of a test-time task identifier and restrict knowledge sharing among isolated adapters. To address these limitations, we propose Dynamic Mixture of Progressive Parameter-Efficient Expert Library (DMPEL) for lifelong robot learning. DMPEL progressively builds a low-rank expert library and employs a lightweight router to dynamically combine experts into an end-to-end policy, enabling flexible and efficient lifelong forward transfer. Furthermore, by leveraging the modular structure of the fine-tuned parameters, we introduce expert coefficient replay, which guides the router to accurately retrieve frozen experts for previously encountered tasks. This technique mitigates forgetting while being significantly more storage- and computation-efficient than experience replay over the entire policy. Extensive experiments on the lifelong robot learning benchmark LIBERO demonstrate that our framework outperforms state-of-the-art lifelong learning methods in success rates during continual adaptation, while utilizing minimal trainable parameters and storage.
>
---
#### [replaced 014] Enhancing Video-Based Robot Failure Detection Using Task Knowledge
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.18705v2](http://arxiv.org/pdf/2508.18705v2)**

> **作者:** Santosh Thoduka; Sebastian Houben; Juergen Gall; Paul G. Plöger
>
> **备注:** Accepted at ECMR 2025
>
> **摘要:** Robust robotic task execution hinges on the reliable detection of execution failures in order to trigger safe operation modes, recovery strategies, or task replanning. However, many failure detection methods struggle to provide meaningful performance when applied to a variety of real-world scenarios. In this paper, we propose a video-based failure detection approach that uses spatio-temporal knowledge in the form of the actions the robot performs and task-relevant objects within the field of view. Both pieces of information are available in most robotic scenarios and can thus be readily obtained. We demonstrate the effectiveness of our approach on three datasets that we amend, in part, with additional annotations of the aforementioned task-relevant knowledge. In light of the results, we also propose a data augmentation method that improves performance by applying variable frame rates to different parts of the video. We observe an improvement from 77.9 to 80.0 in F1 score on the ARMBench dataset without additional computational expense and an additional increase to 81.4 with test-time augmentation. The results emphasize the importance of spatio-temporal information during failure detection and suggest further investigation of suitable heuristics in future implementations. Code and annotations are available.
>
---
#### [replaced 015] Ratatouille: Imitation Learning Ingredients for Real-world Social Robot Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.17204v2](http://arxiv.org/pdf/2509.17204v2)**

> **作者:** James R. Han; Mithun Vanniasinghe; Hshmat Sahak; Nicholas Rhinehart; Timothy D. Barfoot
>
> **备注:** 8 pages
>
> **摘要:** Scaling Reinforcement Learning to in-the-wild social robot navigation is both data-intensive and unsafe, since policies must learn through direct interaction and inevitably encounter collisions. Offline Imitation learning (IL) avoids these risks by collecting expert demonstrations safely, training entirely offline, and deploying policies zero-shot. However, we find that naively applying Behaviour Cloning (BC) to social navigation is insufficient; achieving strong performance requires careful architectural and training choices. We present Ratatouille, a pipeline and model architecture that, without changing the data, reduces collisions per meter by 6 times and improves success rate by 3 times compared to naive BC. We validate our approach in both simulation and the real world, where we collected over 11 hours of data on a dense university campus. We further demonstrate qualitative results in a public food court. Our findings highlight that thoughtful IL design, rather than additional data, can substantially improve safety and reliability in real-world social navigation. Video: https://youtu.be/tOdLTXsaYLQ. Code will be released after acceptance.
>
---
#### [replaced 016] Embodied Arena: A Comprehensive, Unified, and Evolving Evaluation Platform for Embodied AI
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.15273v2](http://arxiv.org/pdf/2509.15273v2)**

> **作者:** Fei Ni; Min Zhang; Pengyi Li; Yifu Yuan; Lingfeng Zhang; Yuecheng Liu; Peilong Han; Longxin Kou; Shaojin Ma; Jinbin Qiao; David Gamaliel Arcos Bravo; Yuening Wang; Xiao Hu; Zhanguang Zhang; Xianze Yao; Yutong Li; Zhao Zhang; Ying Wen; Ying-Cong Chen; Xiaodan Liang; Liang Lin; Bin He; Haitham Bou-Ammar; He Wang; Huazhe Xu; Jiankang Deng; Shan Luo; Shuqiang Jiang; Wei Pan; Yang Gao; Stefanos Zafeiriou; Jan Peters; Yuzheng Zhuang; Yingxue Zhang; Yan Zheng; Hongyao Tang; Jianye Hao
>
> **备注:** 32 pages, 5 figures, Embodied Arena Technical Report
>
> **摘要:** Embodied AI development significantly lags behind large foundation models due to three critical challenges: (1) lack of systematic understanding of core capabilities needed for Embodied AI, making research lack clear objectives; (2) absence of unified and standardized evaluation systems, rendering cross-benchmark evaluation infeasible; and (3) underdeveloped automated and scalable acquisition methods for embodied data, creating critical bottlenecks for model scaling. To address these obstacles, we present Embodied Arena, a comprehensive, unified, and evolving evaluation platform for Embodied AI. Our platform establishes a systematic embodied capability taxonomy spanning three levels (perception, reasoning, task execution), seven core capabilities, and 25 fine-grained dimensions, enabling unified evaluation with systematic research objectives. We introduce a standardized evaluation system built upon unified infrastructure supporting flexible integration of 22 diverse benchmarks across three domains (2D/3D Embodied Q&A, Navigation, Task Planning) and 30+ advanced models from 20+ worldwide institutes. Additionally, we develop a novel LLM-driven automated generation pipeline ensuring scalable embodied evaluation data with continuous evolution for diversity and comprehensiveness. Embodied Arena publishes three real-time leaderboards (Embodied Q&A, Navigation, Task Planning) with dual perspectives (benchmark view and capability view), providing comprehensive overviews of advanced model capabilities. Especially, we present nine findings summarized from the evaluation results on the leaderboards of Embodied Arena. This helps to establish clear research veins and pinpoint critical research problems, thereby driving forward progress in the field of Embodied AI.
>
---
#### [replaced 017] Learning coordinated badminton skills for legged manipulators
- **分类: cs.RO; cs.LG; 68T40, 93C85; I.2.9; I.2.6; I.2.8**

- **链接: [http://arxiv.org/pdf/2505.22974v2](http://arxiv.org/pdf/2505.22974v2)**

> **作者:** Yuntao Ma; Andrei Cramariuc; Farbod Farshidian; Marco Hutter
>
> **备注:** Science Robotics DOI: 10.1126/scirobotics.adu3922
>
> **摘要:** Coordinating the motion between lower and upper limbs and aligning limb control with perception are substantial challenges in robotics, particularly in dynamic environments. To this end, we introduce an approach for enabling legged mobile manipulators to play badminton, a task that requires precise coordination of perception, locomotion, and arm swinging. We propose a unified reinforcement learning-based control policy for whole-body visuomotor skills involving all degrees of freedom to achieve effective shuttlecock tracking and striking. This policy is informed by a perception noise model that utilizes real-world camera data, allowing for consistent perception error levels between simulation and deployment and encouraging learned active perception behaviors. Our method includes a shuttlecock prediction model, constrained reinforcement learning for robust motion control, and integrated system identification techniques to enhance deployment readiness. Extensive experimental results in a variety of environments validate the robot's capability to predict shuttlecock trajectories, navigate the service area effectively, and execute precise strikes against human players, demonstrating the feasibility of using legged mobile manipulators in complex and dynamic sports scenarios.
>
---
#### [replaced 018] EMMA: End-to-End Multimodal Model for Autonomous Driving
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.23262v3](http://arxiv.org/pdf/2410.23262v3)**

> **作者:** Jyh-Jing Hwang; Runsheng Xu; Hubert Lin; Wei-Chih Hung; Jingwei Ji; Kristy Choi; Di Huang; Tong He; Paul Covington; Benjamin Sapp; Yin Zhou; James Guo; Dragomir Anguelov; Mingxing Tan
>
> **备注:** Accepted by TMLR. Blog post: https://waymo.com/blog/2024/10/introducing-emma/
>
> **摘要:** We introduce EMMA, an End-to-end Multimodal Model for Autonomous driving. Built upon a multi-modal large language model foundation like Gemini, EMMA directly maps raw camera sensor data into various driving-specific outputs, including planner trajectories, perception objects, and road graph elements. EMMA maximizes the utility of world knowledge from the pre-trained large language models, by representing all non-sensor inputs (e.g. navigation instructions and ego vehicle status) and outputs (e.g. trajectories and 3D locations) as natural language text. This approach allows EMMA to jointly process various driving tasks in a unified language space, and generate the outputs for each task using task-specific prompts. Empirically, we demonstrate EMMA's effectiveness by achieving state-of-the-art performance in motion planning on nuScenes as well as competitive results on the Waymo Open Motion Dataset (WOMD). EMMA also yields competitive results for camera-primary 3D object detection on the Waymo Open Dataset (WOD). We show that co-training EMMA with planner trajectories, object detection, and road graph tasks yields improvements across all three domains, highlighting EMMA's potential as a generalist model for autonomous driving applications. We hope that our results will inspire research to further evolve the state of the art in autonomous driving model architectures.
>
---
#### [replaced 019] Socially Pertinent Robots in Gerontological Healthcare
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.07560v3](http://arxiv.org/pdf/2404.07560v3)**

> **作者:** Xavier Alameda-Pineda; Angus Addlesee; Daniel Hernández García; Chris Reinke; Soraya Arias; Federica Arrigoni; Alex Auternaud; Lauriane Blavette; Cigdem Beyan; Luis Gomez Camara; Ohad Cohen; Alessandro Conti; Sébastien Dacunha; Christian Dondrup; Yoav Ellinson; Francesco Ferro; Sharon Gannot; Florian Gras; Nancie Gunson; Radu Horaud; Moreno D'Incà; Imad Kimouche; Séverin Lemaignan; Oliver Lemon; Cyril Liotard; Luca Marchionni; Mordehay Moradi; Tomas Pajdla; Maribel Pino; Michal Polic; Matthieu Py; Ariel Rado; Bin Ren; Elisa Ricci; Anne-Sophie Rigaud; Paolo Rota; Marta Romeo; Nicu Sebe; Weronika Sieińska; Pinchas Tandeitnik; Francesco Tonini; Nicolas Turro; Timothée Wintz; Yanchao Yu
>
> **摘要:** Despite the many recent achievements in developing and deploying social robotics, there are still many underexplored environments and applications for which systematic evaluation of such systems by end-users is necessary. While several robotic platforms have been used in gerontological healthcare, the question of whether or not a social interactive robot with multi-modal conversational capabilities will be useful and accepted in real-life facilities is yet to be answered. This paper is an attempt to partially answer this question, via two waves of experiments with patients and companions in a day-care gerontological facility in Paris with a full-sized humanoid robot endowed with social and conversational interaction capabilities. The software architecture, developed during the H2020 SPRING project, together with the experimental protocol, allowed us to evaluate the acceptability (AES) and usability (SUS) with more than 60 end-users. Overall, the users are receptive to this technology, especially when the robot perception and action skills are robust to environmental clutter and flexible to handle a plethora of different interactions.
>
---
