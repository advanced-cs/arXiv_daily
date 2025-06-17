# 机器人 cs.RO

- **最新发布 63 篇**

- **更新 34 篇**

## 最新发布

#### [new 001] Delayed Expansion AGT: Kinodynamic Planning with Application to Tractor-Trailer Parking
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于路径规划任务，解决复杂环境中铰接车辆的运动规划问题。提出DE-AGT算法，通过预计算运动基元和强化学习提升规划效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.13421v1](http://arxiv.org/pdf/2506.13421v1)**

> **作者:** Dongliang Zheng; Yebin Wang; Stefano Di Cairano; Panagiotis Tsiotras
>
> **摘要:** Kinodynamic planning of articulated vehicles in cluttered environments faces additional challenges arising from high-dimensional state space and complex system dynamics. Built upon [1],[2], this work proposes the DE-AGT algorithm that grows a tree using pre-computed motion primitives (MPs) and A* heuristics. The first feature of DE-AGT is a delayed expansion of MPs. In particular, the MPs are divided into different modes, which are ranked online. With the MP classification and prioritization, DE-AGT expands the most promising mode of MPs first, which eliminates unnecessary computation and finds solutions faster. To obtain the cost-to-go heuristic for nonholonomic articulated vehicles, we rely on supervised learning and train neural networks for fast and accurate cost-to-go prediction. The learned heuristic is used for online mode ranking and node selection. Another feature of DE-AGT is the improved goal-reaching. Exactly reaching a goal state usually requires a constant connection checking with the goal by solving steering problems -- non-trivial and time-consuming for articulated vehicles. The proposed termination scheme overcomes this challenge by tightly integrating a light-weight trajectory tracking controller with the search process. DE-AGT is implemented for autonomous parking of a general car-like tractor with 3-trailer. Simulation results show an average of 10x acceleration compared to a previous method.
>
---
#### [new 002] Physics-informed Neural Motion Planning via Domain Decomposition in Large Environments
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，解决大规模环境下的成本估计问题。提出FB-NTFields方法，通过潜空间表示实现高效、全局一致的路径规划。**

- **链接: [http://arxiv.org/pdf/2506.12742v1](http://arxiv.org/pdf/2506.12742v1)**

> **作者:** Yuchen Liu; Alexiy Buynitsky; Ruiqi Ni; Ahmed H. Qureshi
>
> **摘要:** Physics-informed Neural Motion Planners (PiNMPs) provide a data-efficient framework for solving the Eikonal Partial Differential Equation (PDE) and representing the cost-to-go function for motion planning. However, their scalability remains limited by spectral bias and the complex loss landscape of PDE-driven training. Domain decomposition mitigates these issues by dividing the environment into smaller subdomains, but existing methods enforce continuity only at individual spatial points. While effective for function approximation, these methods fail to capture the spatial connectivity required for motion planning, where the cost-to-go function depends on both the start and goal coordinates rather than a single query point. We propose Finite Basis Neural Time Fields (FB-NTFields), a novel neural field representation for scalable cost-to-go estimation. Instead of enforcing continuity in output space, FB-NTFields construct a latent space representation, computing the cost-to-go as a distance between the latent embeddings of start and goal coordinates. This enables global spatial coherence while integrating domain decomposition, ensuring efficient large-scale motion planning. We validate FB-NTFields in complex synthetic and real-world scenarios, demonstrating substantial improvements over existing PiNMPs. Finally, we deploy our method on a Unitree B1 quadruped robot, successfully navigating indoor environments. The supplementary videos can be found at https://youtu.be/OpRuCbLNOwM.
>
---
#### [new 003] Critical Insights about Robots for Mental Wellbeing
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于心理健康领域，探讨机器人在提升心理福祉中的应用。解决如何有效设计支持性机器人的问题，通过分析六项关键洞察，提出需结合证据与伦理的设计方法。**

- **链接: [http://arxiv.org/pdf/2506.13739v1](http://arxiv.org/pdf/2506.13739v1)**

> **作者:** Guy Laban; Micol Spitale; Minja Axelsson; Nida Itrat Abbasi; Hatice Gunes
>
> **摘要:** Social robots are increasingly being explored as tools to support emotional wellbeing, particularly in non-clinical settings. Drawing on a range of empirical studies and practical deployments, this paper outlines six key insights that highlight both the opportunities and challenges in using robots to promote mental wellbeing. These include (1) the lack of a single, objective measure of wellbeing, (2) the fact that robots don't need to act as companions to be effective, (3) the growing potential of virtual interactions, (4) the importance of involving clinicians in the design process, (5) the difference between one-off and long-term interactions, and (6) the idea that adaptation and personalization are not always necessary for positive outcomes. Rather than positioning robots as replacements for human therapists, we argue that they are best understood as supportive tools that must be designed with care, grounded in evidence, and shaped by ethical and psychological considerations. Our aim is to inform future research and guide responsible, effective use of robots in mental health and wellbeing contexts.
>
---
#### [new 004] Explosive Output to Enhance Jumping Ability: A Variable Reduction Ratio Design Paradigm for Humanoid Robots Knee Joint
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人关节设计任务，旨在提升人形机器人跳跃能力。针对膝关节传动比与跳跃需求不匹配及高速性能下降问题，提出可变减速比设计，优化动力输出，实现更高跳跃性能。**

- **链接: [http://arxiv.org/pdf/2506.12314v1](http://arxiv.org/pdf/2506.12314v1)**

> **作者:** Xiaoshuai Ma; Haoxiang Qi; Qingqing Li; Haochen Xu; Xuechao Chen; Junyao Gao; Zhangguo Yu; Qiang Huang
>
> **摘要:** Enhancing the explosive power output of the knee joints is critical for improving the agility and obstacle-crossing capabilities of humanoid robots. However, a mismatch between the knee-to-center-of-mass (CoM) transmission ratio and jumping demands, coupled with motor performance degradation at high speeds, restricts the duration of high-power output and limits jump performance. To address these problems, this paper introduces a novel knee joint design paradigm employing a dynamically decreasing reduction ratio for explosive output during jump. Analysis of motor output characteristics and knee kinematics during jumping inspired a coupling strategy in which the reduction ratio gradually decreases as the joint extends. A high initial ratio rapidly increases torque at jump initiation, while its gradual reduction minimizes motor speed increments and power losses, thereby maintaining sustained high-power output. A compact and efficient linear actuator-driven guide-rod mechanism realizes this coupling strategy, supported by parameter optimization guided by explosive jump control strategies. Experimental validation demonstrated a 63 cm vertical jump on a single-joint platform (a theoretical improvement of 28.1\% over the optimal fixed-ratio joints). Integrated into a humanoid robot, the proposed design enabled a 1.1 m long jump, a 0.5 m vertical jump, and a 0.5 m box jump.
>
---
#### [new 005] Strategic Vantage Selection for Learning Viewpoint-Agnostic Manipulation Policies
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉操控任务，解决视角依赖导致的泛化问题。通过优化视角选择，提升策略的视角无关性。**

- **链接: [http://arxiv.org/pdf/2506.12261v1](http://arxiv.org/pdf/2506.12261v1)**

> **作者:** Sreevishakh Vasudevan; Som Sagar; Ransalu Senanayake
>
> **摘要:** Vision-based manipulation has shown remarkable success, achieving promising performance across a range of tasks. However, these manipulation policies often fail to generalize beyond their training viewpoints, which is a persistent challenge in achieving perspective-agnostic manipulation, especially in settings where the camera is expected to move at runtime. Although collecting data from many angles seems a natural solution, such a naive approach is both resource-intensive and degrades manipulation policy performance due to excessive and unstructured visual diversity. This paper proposes Vantage, a framework that systematically identifies and integrates data from optimal perspectives to train robust, viewpoint-agnostic policies. By formulating viewpoint selection as a continuous optimization problem, we iteratively fine-tune policies on a few vantage points. Since we leverage Bayesian optimization to efficiently navigate the infinite space of potential camera configurations, we are able to balance exploration of novel views and exploitation of high-performing ones, thereby ensuring data collection from a minimal number of effective viewpoints. We empirically evaluate this framework on diverse standard manipulation tasks using multiple policy learning methods, demonstrating that fine-tuning with data from strategic camera placements yields substantial performance gains, achieving average improvements of up to 46.19% when compared to fixed, random, or heuristic-based strategies.
>
---
#### [new 006] ProVox: Personalization and Proactive Planning for Situated Human-Robot Collaboration
- **分类: cs.RO; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于人机协作任务，旨在提升机器人对人类意图的预判与个性化适应能力。通过ProVox框架，机器人能主动规划行为，减少用户指令需求，提高协作效率。**

- **链接: [http://arxiv.org/pdf/2506.12248v1](http://arxiv.org/pdf/2506.12248v1)**

> **作者:** Jennifer Grannen; Siddharth Karamcheti; Blake Wulfe; Dorsa Sadigh
>
> **备注:** Accepted by IEEE Robotics and Automation Letters 2025
>
> **摘要:** Collaborative robots must quickly adapt to their partner's intent and preferences to proactively identify helpful actions. This is especially true in situated settings where human partners can continually teach robots new high-level behaviors, visual concepts, and physical skills (e.g., through demonstration), growing the robot's capabilities as the human-robot pair work together to accomplish diverse tasks. In this work, we argue that robots should be able to infer their partner's goals from early interactions and use this information to proactively plan behaviors ahead of explicit instructions from the user. Building from the strong commonsense priors and steerability of large language models, we introduce ProVox ("Proactive Voice"), a novel framework that enables robots to efficiently personalize and adapt to individual collaborators. We design a meta-prompting protocol that empowers users to communicate their distinct preferences, intent, and expected robot behaviors ahead of starting a physical interaction. ProVox then uses the personalized prompt to condition a proactive language model task planner that anticipates a user's intent from the current interaction context and robot capabilities to suggest helpful actions; in doing so, we alleviate user burden, minimizing the amount of time partners spend explicitly instructing and supervising the robot. We evaluate ProVox through user studies grounded in household manipulation tasks (e.g., assembling lunch bags) that measure the efficiency of the collaboration, as well as features such as perceived helpfulness, ease of use, and reliability. Our analysis suggests that both meta-prompting and proactivity are critical, resulting in 38.7% faster task completion times and 31.9% less user burden relative to non-active baselines. Supplementary material, code, and videos can be found at https://provox-2025.github.io.
>
---
#### [new 007] KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决动态人类动作模仿问题。通过物理建模和自适应优化，提升机器人对高动态动作的跟踪能力。**

- **链接: [http://arxiv.org/pdf/2506.12851v1](http://arxiv.org/pdf/2506.12851v1)**

> **作者:** Weiji Xie; Jinrui Han; Jiakun Zheng; Huanyu Li; Xinzhe Liu; Jiyuan Shi; Weinan Zhang; Chenjia Bai; Xuelong Li
>
> **摘要:** Humanoid robots are promising to acquire various skills by imitating human behaviors. However, existing algorithms are only capable of tracking smooth, low-speed human motions, even with delicate reward and curriculum design. This paper presents a physics-based humanoid control framework, aiming to master highly-dynamic human behaviors such as Kungfu and dancing through multi-steps motion processing and adaptive motion tracking. For motion processing, we design a pipeline to extract, filter out, correct, and retarget motions, while ensuring compliance with physical constraints to the maximum extent. For motion imitation, we formulate a bi-level optimization problem to dynamically adjust the tracking accuracy tolerance based on the current tracking error, creating an adaptive curriculum mechanism. We further construct an asymmetric actor-critic framework for policy training. In experiments, we train whole-body control policies to imitate a set of highly-dynamic motions. Our method achieves significantly lower tracking errors than existing approaches and is successfully deployed on the Unitree G1 robot, demonstrating stable and expressive behaviors. The project page is https://kungfu-bot.github.io.
>
---
#### [new 008] Adaptive Model-Base Control of Quadrupeds via Online System Identification using Kalman Filter
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决腿式机器人在变负载下模型不准确的问题。通过在线卡尔曼滤波识别质量与质心，提升控制器性能。**

- **链接: [http://arxiv.org/pdf/2506.13432v1](http://arxiv.org/pdf/2506.13432v1)**

> **作者:** Jonas Haack; Franek Stark; Shubham Vyas; Frank Kirchner; Shivesh Kumar
>
> **备注:** 6 pages, 5 figures, 1 table, accepted for IEEE IROS 2025
>
> **摘要:** Many real-world applications require legged robots to be able to carry variable payloads. Model-based controllers such as model predictive control (MPC) have become the de facto standard in research for controlling these systems. However, most model-based control architectures use fixed plant models, which limits their applicability to different tasks. In this paper, we present a Kalman filter (KF) formulation for online identification of the mass and center of mass (COM) of a four-legged robot. We evaluate our method on a quadrupedal robot carrying various payloads and find that it is more robust to strong measurement noise than classical recursive least squares (RLS) methods. Moreover, it improves the tracking performance of the model-based controller with varying payloads when the model parameters are adjusted at runtime.
>
---
#### [new 009] Touch begins where vision ends: Generalizable policies for contact-rich manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操控任务，解决接触丰富的精细操作问题。提出ViTaL框架，结合视觉和触觉感知，提升策略的泛化能力与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.13762v1](http://arxiv.org/pdf/2506.13762v1)**

> **作者:** Zifan Zhao; Siddhant Haldar; Jinda Cui; Lerrel Pinto; Raunaq Bhirangi
>
> **摘要:** Data-driven approaches struggle with precise manipulation; imitation learning requires many hard-to-obtain demonstrations, while reinforcement learning yields brittle, non-generalizable policies. We introduce VisuoTactile Local (ViTaL) policy learning, a framework that solves fine-grained manipulation tasks by decomposing them into two phases: a reaching phase, where a vision-language model (VLM) enables scene-level reasoning to localize the object of interest, and a local interaction phase, where a reusable, scene-agnostic ViTaL policy performs contact-rich manipulation using egocentric vision and tactile sensing. This approach is motivated by the observation that while scene context varies, the low-level interaction remains consistent across task instances. By training local policies once in a canonical setting, they can generalize via a localize-then-execute strategy. ViTaL achieves around 90% success on contact-rich tasks in unseen environments and is robust to distractors. ViTaL's effectiveness stems from three key insights: (1) foundation models for segmentation enable training robust visual encoders via behavior cloning; (2) these encoders improve the generalizability of policies learned using residual RL; and (3) tactile sensing significantly boosts performance in contact-rich tasks. Ablation studies validate each of these insights, and we demonstrate that ViTaL integrates well with high-level VLMs, enabling robust, reusable low-level skills. Results and videos are available at https://vitalprecise.github.io.
>
---
#### [new 010] Using Behavior Trees in Risk Assessment
- **分类: cs.RO; cs.SE**

- **简介: 该论文属于风险评估任务，旨在解决早期设计阶段机器人任务安全分析不足的问题。通过行为树模型支持早期风险识别与可视化。**

- **链接: [http://arxiv.org/pdf/2506.12089v1](http://arxiv.org/pdf/2506.12089v1)**

> **作者:** Razan Ghzouli; Atieh Hanna; Endre Erös; Rebekka Wohlrab
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Cyber-physical production systems increasingly involve collaborative robotic missions, requiring more demand for robust and safe missions. Industries rely on risk assessments to identify potential failures and implement measures to mitigate their risks. Although it is recommended to conduct risk assessments early in the design of robotic missions, the state of practice in the industry is different. Safety experts often struggle to completely understand robotics missions at the early design stages of projects and to ensure that the output of risk assessments is adequately considered during implementation. This paper presents a design science study that conceived a model-based approach for early risk assessment in a development-centric way. Our approach supports risk assessment activities by using the behavior-tree model. We evaluated the approach together with five practitioners from four companies. Our findings highlight the potential of the behavior-tree model in supporting early identification, visualisation, and bridging the gap between code implementation and risk assessments' outputs. This approach is the first attempt to use the behavior-tree model to support risk assessment; thus, the findings highlight the need for further development.
>
---
#### [new 011] A Spatial Relationship Aware Dataset for Robotics
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划领域，旨在解决物体空间关系理解问题。构建了一个包含1000张室内图像的数据集，并评估了场景图生成模型的性能。**

- **链接: [http://arxiv.org/pdf/2506.12525v1](http://arxiv.org/pdf/2506.12525v1)**

> **作者:** Peng Wang; Minh Huy Pham; Zhihao Guo; Wei Zhou
>
> **备注:** 7 pages; 7 figures, 1 table
>
> **摘要:** Robotic task planning in real-world environments requires not only object recognition but also a nuanced understanding of spatial relationships between objects. We present a spatial-relationship-aware dataset of nearly 1,000 robot-acquired indoor images, annotated with object attributes, positions, and detailed spatial relationships. Captured using a Boston Dynamics Spot robot and labelled with a custom annotation tool, the dataset reflects complex scenarios with similar or identical objects and intricate spatial arrangements. We benchmark six state-of-the-art scene-graph generation models on this dataset, analysing their inference speed and relational accuracy. Our results highlight significant differences in model performance and demonstrate that integrating explicit spatial relationships into foundation models, such as ChatGPT 4o, substantially improves their ability to generate executable, spatially-aware plans for robotics. The dataset and annotation tool are publicly available at https://github.com/PengPaulWang/SpatialAwareRobotDataset, supporting further research in spatial reasoning for robotics.
>
---
#### [new 012] DoublyAware: Dual Planning and Policy Awareness for Temporal Difference Learning in Humanoid Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动学习任务，旨在解决人形机器人在不确定环境中的高效稳定学习问题。提出DoublyAware方法，分解并处理规划与策略不确定性，提升样本效率和运动可行性。**

- **链接: [http://arxiv.org/pdf/2506.12095v1](http://arxiv.org/pdf/2506.12095v1)**

> **作者:** Khang Nguyen; An T. Le; Jan Peters; Minh Nhat Vu
>
> **摘要:** Achieving robust robot learning for humanoid locomotion is a fundamental challenge in model-based reinforcement learning (MBRL), where environmental stochasticity and randomness can hinder efficient exploration and learning stability. The environmental, so-called aleatoric, uncertainty can be amplified in high-dimensional action spaces with complex contact dynamics, and further entangled with epistemic uncertainty in the models during learning phases. In this work, we propose DoublyAware, an uncertainty-aware extension of Temporal Difference Model Predictive Control (TD-MPC) that explicitly decomposes uncertainty into two disjoint interpretable components, i.e., planning and policy uncertainties. To handle the planning uncertainty, DoublyAware employs conformal prediction to filter candidate trajectories using quantile-calibrated risk bounds, ensuring statistical consistency and robustness against stochastic dynamics. Meanwhile, policy rollouts are leveraged as structured informative priors to support the learning phase with Group-Relative Policy Constraint (GRPC) optimizers that impose a group-based adaptive trust-region in the latent action space. This principled combination enables the robot agent to prioritize high-confidence, high-reward behavior while maintaining effective, targeted exploration under uncertainty. Evaluated on the HumanoidBench locomotion suite with the Unitree 26-DoF H1-2 humanoid, DoublyAware demonstrates improved sample efficiency, accelerated convergence, and enhanced motion feasibility compared to RL baselines. Our simulation results emphasize the significance of structured uncertainty modeling for data-efficient and reliable decision-making in TD-MPC-based humanoid locomotion learning.
>
---
#### [new 013] Sense and Sensibility: What makes a social robot convincing to high-school students?
- **分类: cs.RO**

- **简介: 该论文属于教育机器人研究任务，探讨社会机器人如何影响高中生决策。通过实验分析机器人说服力与学生反应，发现机器人可信度显著影响学生接受其观点的程度。**

- **链接: [http://arxiv.org/pdf/2506.12507v1](http://arxiv.org/pdf/2506.12507v1)**

> **作者:** Pablo Gonzalez-Oliveras; Olov Engwall; Ali Reza Majlesi
>
> **备注:** 14 pages; 8 figures; 3 tables; RSS 2025 (Robotics: Science & Systems)
>
> **摘要:** This study with 40 high-school students demonstrates the high influence of a social educational robot on students' decision-making for a set of eight true-false questions on electric circuits, for which the theory had been covered in the students' courses. The robot argued for the correct answer on six questions and the wrong on two, and 75% of the students were persuaded by the robot to perform beyond their expected capacity, positively when the robot was correct and negatively when it was wrong. Students with more experience of using large language models were even more likely to be influenced by the robot's stance -- in particular for the two easiest questions on which the robot was wrong -- suggesting that familiarity with AI can increase susceptibility to misinformation by AI. We further examined how three different levels of portrayed robot certainty, displayed using semantics, prosody and facial signals, affected how the students aligned with the robot's answer on specific questions and how convincing they perceived the robot to be on these questions. The students aligned with the robot's answers in 94.4% of the cases when the robot was portrayed as Certain, 82.6% when it was Neutral and 71.4% when it was Uncertain. The alignment was thus high for all conditions, highlighting students' general susceptibility to accept the robot's stance, but alignment in the Uncertain condition was significantly lower than in the Certain. Post-test questionnaire answers further show that students found the robot most convincing when it was portrayed as Certain. These findings highlight the need for educational robots to adjust their display of certainty based on the reliability of the information they convey, to promote students' critical thinking and reduce undue influence.
>
---
#### [new 014] RL from Physical Feedback: Aligning Large Motion Models with Humanoid Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于人形机器人运动控制任务，解决文本驱动运动在物理上的可行性问题。通过RLPF框架，结合物理反馈与语义对齐，生成可执行的运动轨迹。**

- **链接: [http://arxiv.org/pdf/2506.12769v1](http://arxiv.org/pdf/2506.12769v1)**

> **作者:** Junpeng Yue; Zepeng Wang; Yuxuan Wang; Weishuai Zeng; Jiangxing Wang; Xinrun Xu; Yu Zhang; Sipeng Zheng; Ziluo Ding; Zongqing Lu
>
> **摘要:** This paper focuses on a critical challenge in robotics: translating text-driven human motions into executable actions for humanoid robots, enabling efficient and cost-effective learning of new behaviors. While existing text-to-motion generation methods achieve semantic alignment between language and motion, they often produce kinematically or physically infeasible motions unsuitable for real-world deployment. To bridge this sim-to-real gap, we propose Reinforcement Learning from Physical Feedback (RLPF), a novel framework that integrates physics-aware motion evaluation with text-conditioned motion generation. RLPF employs a motion tracking policy to assess feasibility in a physics simulator, generating rewards for fine-tuning the motion generator. Furthermore, RLPF introduces an alignment verification module to preserve semantic fidelity to text instructions. This joint optimization ensures both physical plausibility and instruction alignment. Extensive experiments show that RLPF greatly outperforms baseline methods in generating physically feasible motions while maintaining semantic correspondence with text instruction, enabling successful deployment on real humanoid robots.
>
---
#### [new 015] A Survey on Imitation Learning for Contact-Rich Tasks in Robotics
- **分类: cs.RO; cs.HC; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于机器人接触密集任务领域，旨在解决复杂物理交互中的模仿学习问题，综述了示范收集与学习方法，推动相关技术发展。**

- **链接: [http://arxiv.org/pdf/2506.13498v1](http://arxiv.org/pdf/2506.13498v1)**

> **作者:** Toshiaki Tsuji; Yasuhiro Kato; Gokhan Solak; Heng Zhang; Tadej Petrič; Francesco Nori; Arash Ajoudani
>
> **备注:** 47pages, 1 figures
>
> **摘要:** This paper comprehensively surveys research trends in imitation learning for contact-rich robotic tasks. Contact-rich tasks, which require complex physical interactions with the environment, represent a central challenge in robotics due to their nonlinear dynamics and sensitivity to small positional deviations. The paper examines demonstration collection methodologies, including teaching methods and sensory modalities crucial for capturing subtle interaction dynamics. We then analyze imitation learning approaches, highlighting their applications to contact-rich manipulation. Recent advances in multimodal learning and foundation models have significantly enhanced performance in complex contact tasks across industrial, household, and healthcare domains. Through systematic organization of current research and identification of challenges, this survey provides a foundation for future advancements in contact-rich robotic manipulation.
>
---
#### [new 016] LeVERB: Humanoid Whole-Body Control with Latent Vision-Language Instruction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人全身控制任务，解决视觉语言指令与低级控制器的衔接问题。提出LeVERB框架，通过层次化潜空间实现视觉语言指令到动态命令的转换。**

- **链接: [http://arxiv.org/pdf/2506.13751v1](http://arxiv.org/pdf/2506.13751v1)**

> **作者:** Haoru Xue; Xiaoyu Huang; Dantong Niu; Qiayuan Liao; Thomas Kragerud; Jan Tommy Gravdahl; Xue Bin Peng; Guanya Shi; Trevor Darrell; Koushil Screenath; Shankar Sastry
>
> **摘要:** Vision-language-action (VLA) models have demonstrated strong semantic understanding and zero-shot generalization, yet most existing systems assume an accurate low-level controller with hand-crafted action "vocabulary" such as end-effector pose or root velocity. This assumption confines prior work to quasi-static tasks and precludes the agile, whole-body behaviors required by humanoid whole-body control (WBC) tasks. To capture this gap in the literature, we start by introducing the first sim-to-real-ready, vision-language, closed-loop benchmark for humanoid WBC, comprising over 150 tasks from 10 categories. We then propose LeVERB: Latent Vision-Language-Encoded Robot Behavior, a hierarchical latent instruction-following framework for humanoid vision-language WBC, the first of its kind. At the top level, a vision-language policy learns a latent action vocabulary from synthetically rendered kinematic demonstrations; at the low level, a reinforcement-learned WBC policy consumes these latent verbs to generate dynamics-level commands. In our benchmark, LeVERB can zero-shot attain a 80% success rate on simple visual navigation tasks, and 58.5% success rate overall, outperforming naive hierarchical whole-body VLA implementation by 7.8 times.
>
---
#### [new 017] Design and Development of a Robotic Transcatheter Delivery System for Aortic Valve Replacement
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决TAVR手术中定位精度不足的问题，设计了一种具有全方位弯曲关节和驱动系统的机器人输送系统。**

- **链接: [http://arxiv.org/pdf/2506.12082v1](http://arxiv.org/pdf/2506.12082v1)**

> **作者:** Harith S. Gallage; Bailey F. De Sousa; Benjamin I. Chesnik; Chaikel G. Brownstein; Anson Paul; Ronghuai Qi
>
> **备注:** 1 page with 2 figures. This abstract has been accepted by the 2025 International Conference on Robotics and Automation (ICRA) Workshop on Robot-Assisted Endovascular Interventions
>
> **摘要:** Minimally invasive transcatheter approaches are increasingly adopted for aortic stenosis treatment, where optimal commissural and coronary alignment is important. Achieving precise alignment remains clinically challenging, even with contemporary robotic transcatheter aortic valve replacement (TAVR) devices, as this task is still performed manually. This paper proposes the development of a robotic transcatheter delivery system featuring an omnidirectional bending joint and an actuation system designed to enhance positional accuracy and precision in TAVR procedures. The preliminary experimental results validate the functionality of this novel robotic system.
>
---
#### [new 018] A Novel ViDAR Device With Visual Inertial Encoder Odometry and Reinforcement Learning-Based Active SLAM Method
- **分类: cs.RO; cs.CV; 93C85; I.4**

- **简介: 该论文属于SLAM任务，旨在提升定位与建图精度。通过融合视觉、惯性及编码器信息，提出新型VIEO和DRL主动SLAM方法，增强系统性能。**

- **链接: [http://arxiv.org/pdf/2506.13100v1](http://arxiv.org/pdf/2506.13100v1)**

> **作者:** Zhanhua Xin; Zhihao Wang; Shenghao Zhang; Wanchao Chi; Yan Meng; Shihan Kong; Yan Xiong; Chong Zhang; Yuzhen Liu; Junzhi Yu
>
> **备注:** 12 pages, 13 figures
>
> **摘要:** In the field of multi-sensor fusion for simultaneous localization and mapping (SLAM), monocular cameras and IMUs are widely used to build simple and effective visual-inertial systems. However, limited research has explored the integration of motor-encoder devices to enhance SLAM performance. By incorporating such devices, it is possible to significantly improve active capability and field of view (FOV) with minimal additional cost and structural complexity. This paper proposes a novel visual-inertial-encoder tightly coupled odometry (VIEO) based on a ViDAR (Video Detection and Ranging) device. A ViDAR calibration method is introduced to ensure accurate initialization for VIEO. In addition, a platform motion decoupled active SLAM method based on deep reinforcement learning (DRL) is proposed. Experimental data demonstrate that the proposed ViDAR and the VIEO algorithm significantly increase cross-frame co-visibility relationships compared to its corresponding visual-inertial odometry (VIO) algorithm, improving state estimation accuracy. Additionally, the DRL-based active SLAM algorithm, with the ability to decouple from platform motion, can increase the diversity weight of the feature points and further enhance the VIEO algorithm's performance. The proposed methodology sheds fresh insights into both the updated platform design and decoupled approach of active SLAM systems in complex environments.
>
---
#### [new 019] What Matters in Learning from Large-Scale Datasets for Robot Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究机器人操作中大规模数据集的构建与使用，旨在提升数据效用和策略学习性能。通过模拟生成数据集，分析数据多样性与检索策略对任务的影响。**

- **链接: [http://arxiv.org/pdf/2506.13536v1](http://arxiv.org/pdf/2506.13536v1)**

> **作者:** Vaibhav Saxena; Matthew Bronars; Nadun Ranawaka Arachchige; Kuancheng Wang; Woo Chul Shin; Soroush Nasiriany; Ajay Mandlekar; Danfei Xu
>
> **摘要:** Imitation learning from large multi-task demonstration datasets has emerged as a promising path for building generally-capable robots. As a result, 1000s of hours have been spent on building such large-scale datasets around the globe. Despite the continuous growth of such efforts, we still lack a systematic understanding of what data should be collected to improve the utility of a robotics dataset and facilitate downstream policy learning. In this work, we conduct a large-scale dataset composition study to answer this question. We develop a data generation framework to procedurally emulate common sources of diversity in existing datasets (such as sensor placements and object types and arrangements), and use it to generate large-scale robot datasets with controlled compositions, enabling a suite of dataset composition studies that would be prohibitively expensive in the real world. We focus on two practical settings: (1) what types of diversity should be emphasized when future researchers collect large-scale datasets for robotics, and (2) how should current practitioners retrieve relevant demonstrations from existing datasets to maximize downstream policy performance on tasks of interest. Our study yields several critical insights -- for example, we find that camera poses and spatial arrangements are crucial dimensions for both diversity in collection and alignment in retrieval. In real-world robot learning settings, we find that not only do our insights from simulation carry over, but our retrieval strategies on existing datasets such as DROID allow us to consistently outperform existing training strategies by up to 70%. More results at https://robo-mimiclabs.github.io/
>
---
#### [new 020] SPLATART: Articulated Gaussian Splatting with Estimated Object Structure
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学中的物体表示任务，旨在解决复杂关节物体的几何与结构建模问题。提出SPLATART方法，分离部件分割与关节估计，提升对深层运动链的表示能力。**

- **链接: [http://arxiv.org/pdf/2506.12184v1](http://arxiv.org/pdf/2506.12184v1)**

> **作者:** Stanley Lewis; Vishal Chandra; Tom Gao; Odest Chadwicke Jenkins
>
> **备注:** 7 pages, Accepted to the 2025 RSS Workshop on Gaussian Representations for Robot Autonomy. Contact: Stanley Lewis, stanlew@umich.edu
>
> **摘要:** Representing articulated objects remains a difficult problem within the field of robotics. Objects such as pliers, clamps, or cabinets require representations that capture not only geometry and color information, but also part seperation, connectivity, and joint parametrization. Furthermore, learning these representations becomes even more difficult with each additional degree of freedom. Complex articulated objects such as robot arms may have seven or more degrees of freedom, and the depth of their kinematic tree may be notably greater than the tools, drawers, and cabinets that are the typical subjects of articulated object research. To address these concerns, we introduce SPLATART - a pipeline for learning Gaussian splat representations of articulated objects from posed images, of which a subset contains image space part segmentations. SPLATART disentangles the part separation task from the articulation estimation task, allowing for post-facto determination of joint estimation and representation of articulated objects with deeper kinematic trees than previously exhibited. In this work, we present data on the SPLATART pipeline as applied to the syntheic Paris dataset objects, and qualitative results on a real-world object under spare segmentation supervision. We additionally present on articulated serial chain manipulators to demonstrate usage on deeper kinematic tree structures.
>
---
#### [new 021] Role of Uncertainty in Model Development and Control Design for a Manufacturing Process
- **分类: cs.RO; cs.SY; eess.SY; 93B30 (Primary), 93B35 (Secondary)**

- **简介: 该论文属于制造过程中的控制设计任务，旨在解决机器人在微尺度制造中因不确定性导致的精度问题，通过多机器人控制系统降低这些不确定性。**

- **链接: [http://arxiv.org/pdf/2506.12273v1](http://arxiv.org/pdf/2506.12273v1)**

> **作者:** Rongfei Li; Francis Assadian
>
> **备注:** 35 pages, 26 figures, Book Chapter. Published in: Role of Uncertainty in Model Development and Control Design for a Manufacturing Process, IntechOpen, 2022. For published version, see this http URL: https://doi.org/10.5772/intechopen.104780
>
> **摘要:** The use of robotic technology has drastically increased in manufacturing in the 21st century. But by utilizing their sensory cues, humans still outperform machines, especially in the micro scale manufacturing, which requires high-precision robot manipulators. These sensory cues naturally compensate for high level of uncertainties that exist in the manufacturing environment. Uncertainties in performing manufacturing tasks may come from measurement noise, model inaccuracy, joint compliance (e.g., elasticity) etc. Although advanced metrology sensors and high-precision microprocessors, which are utilized in nowadays robots, have compensated for many structural and dynamic errors in robot positioning, but a well-designed control algorithm still works as a comparable and cheaper alternative to reduce uncertainties in automated manufacturing. Our work illustrates that a multi-robot control system can reduce various uncertainties to a great amount.
>
---
#### [new 022] Perspective on Utilizing Foundation Models for Laboratory Automation in Materials Research
- **分类: cs.RO; cs.CL; physics.chem-ph**

- **简介: 该论文属于材料科学与人工智能交叉任务，旨在解决实验室自动化中的智能控制问题，通过基础模型提升实验规划与硬件操作的智能化水平。**

- **链接: [http://arxiv.org/pdf/2506.12312v1](http://arxiv.org/pdf/2506.12312v1)**

> **作者:** Kan Hatakeyama-Sato; Toshihiko Nishida; Kenta Kitamura; Yoshitaka Ushiku; Koichi Takahashi; Yuta Nabae; Teruaki Hayakawa
>
> **摘要:** This review explores the potential of foundation models to advance laboratory automation in the materials and chemical sciences. It emphasizes the dual roles of these models: cognitive functions for experimental planning and data analysis, and physical functions for hardware operations. While traditional laboratory automation has relied heavily on specialized, rigid systems, foundation models offer adaptability through their general-purpose intelligence and multimodal capabilities. Recent advancements have demonstrated the feasibility of using large language models (LLMs) and multimodal robotic systems to handle complex and dynamic laboratory tasks. However, significant challenges remain, including precision manipulation of hardware, integration of multimodal data, and ensuring operational safety. This paper outlines a roadmap highlighting future directions, advocating for close interdisciplinary collaboration, benchmark establishment, and strategic human-AI integration to realize fully autonomous experimental laboratories.
>
---
#### [new 023] Constrained Optimal Planning to Minimize Battery Degradation of Autonomous Mobile Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于路径规划任务，旨在减少自主移动机器人电池老化。通过优化框架同时考虑循环和日历老化，实现电池寿命延长与任务完成的平衡。**

- **链接: [http://arxiv.org/pdf/2506.13019v1](http://arxiv.org/pdf/2506.13019v1)**

> **作者:** Jiachen Li; Jian Chu; Feiyang Zhao; Shihao Li; Wei Li; Dongmei Chen
>
> **摘要:** This paper proposes an optimization framework that addresses both cycling degradation and calendar aging of batteries for autonomous mobile robot (AMR) to minimize battery degradation while ensuring task completion. A rectangle method of piecewise linear approximation is employed to linearize the bilinear optimization problem. We conduct a case study to validate the efficiency of the proposed framework in achieving an optimal path planning for AMRs while reducing battery aging.
>
---
#### [new 024] HARMONI: Haptic-Guided Assistance for Unified Robotic Tele-Manipulation and Tele-Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人遥操作任务，旨在解决导航与操作分离导致的高认知负荷问题。通过统一框架实现无缝切换，提升操作效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.13704v1](http://arxiv.org/pdf/2506.13704v1)**

> **作者:** V. Sripada; A. Khan; J. Föcker; S. Parsa; Susmitha P; H Maior; A. Ghalamzan-E
>
> **备注:** To appear in IEEE CASE 2025
>
> **摘要:** Shared control, which combines human expertise with autonomous assistance, is critical for effective teleoperation in complex environments. While recent advances in haptic-guided teleoperation have shown promise, they are often limited to simplified tasks involving 6- or 7-DoF manipulators and rely on separate control strategies for navigation and manipulation. This increases both cognitive load and operational overhead. In this paper, we present a unified tele-mobile manipulation framework that leverages haptic-guided shared control. The system integrates a 9-DoF follower mobile manipulator and a 7-DoF leader robotic arm, enabling seamless transitions between tele-navigation and tele-manipulation through real-time haptic feedback. A user study with 20 participants under real-world conditions demonstrates that our framework significantly improves task accuracy and efficiency without increasing cognitive load. These findings highlight the potential of haptic-guided shared control for enhancing operator performance in demanding teleoperation scenarios.
>
---
#### [new 025] From Experts to a Generalist: Toward General Whole-Body Control for Humanoid Robots
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决人形机器人全身运动的通用控制问题。通过聚类与仿真到现实的适应，提出BB框架提升运动泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.12779v1](http://arxiv.org/pdf/2506.12779v1)**

> **作者:** Yuxuan Wang; Ming Yang; Weishuai Zeng; Yu Zhang; Xinrun Xu; Haobin Jiang; Ziluo Ding; Zongqing Lu
>
> **摘要:** Achieving general agile whole-body control on humanoid robots remains a major challenge due to diverse motion demands and data conflicts. While existing frameworks excel in training single motion-specific policies, they struggle to generalize across highly varied behaviors due to conflicting control requirements and mismatched data distributions. In this work, we propose BumbleBee (BB), an expert-generalist learning framework that combines motion clustering and sim-to-real adaptation to overcome these challenges. BB first leverages an autoencoder-based clustering method to group behaviorally similar motions using motion features and motion descriptions. Expert policies are then trained within each cluster and refined with real-world data through iterative delta action modeling to bridge the sim-to-real gap. Finally, these experts are distilled into a unified generalist controller that preserves agility and robustness across all motion types. Experiments on two simulations and a real humanoid robot demonstrate that BB achieves state-of-the-art general whole-body control, setting a new benchmark for agile, robust, and generalizable humanoid performance in the real world.
>
---
#### [new 026] CHARM: Considering Human Attributes for Reinforcement Modeling
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.13079v1](http://arxiv.org/pdf/2506.13079v1)**

> **作者:** Qidi Fang; Hang Yu; Shijie Fang; Jindan Huang; Qiuyu Chen; Reuben M. Aronson; Elaine S. Short
>
> **摘要:** Reinforcement Learning from Human Feedback has recently achieved significant success in various fields, and its performance is highly related to feedback quality. While much prior work acknowledged that human teachers' characteristics would affect human feedback patterns, there is little work that has closely investigated the actual effects. In this work, we designed an exploratory study investigating how human feedback patterns are associated with human characteristics. We conducted a public space study with two long horizon tasks and 46 participants. We found that feedback patterns are not only correlated with task statistics, such as rewards, but also correlated with participants' characteristics, especially robot experience and educational background. Additionally, we demonstrated that human feedback value can be more accurately predicted with human characteristics compared to only using task statistics. All human feedback and characteristics we collected, and codes for our data collection and predicting more accurate human feedback are available at https://github.com/AABL-Lab/CHARM
>
---
#### [new 027] Observability-Aware Active Calibration of Multi-Sensor Extrinsics for Ground Robots via Online Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文属于多传感器外参标定任务，旨在解决地面机器人传感器空间对齐问题。通过在线轨迹优化提升标定精度与系统智能化水平。**

- **链接: [http://arxiv.org/pdf/2506.13420v1](http://arxiv.org/pdf/2506.13420v1)**

> **作者:** Jiang Wang; Yaozhong Kang; Linya Fu; Kazuhiro Nakadai; He Kong
>
> **备注:** Accepted and to appear in the IEEE Sensors Journal
>
> **摘要:** Accurate calibration of sensor extrinsic parameters for ground robotic systems (i.e., relative poses) is crucial for ensuring spatial alignment and achieving high-performance perception. However, existing calibration methods typically require complex and often human-operated processes to collect data. Moreover, most frameworks neglect acoustic sensors, thereby limiting the associated systems' auditory perception capabilities. To alleviate these issues, we propose an observability-aware active calibration method for ground robots with multimodal sensors, including a microphone array, a LiDAR (exteroceptive sensors), and wheel encoders (proprioceptive sensors). Unlike traditional approaches, our method enables active trajectory optimization for online data collection and calibration, contributing to the development of more intelligent robotic systems. Specifically, we leverage the Fisher information matrix (FIM) to quantify parameter observability and adopt its minimum eigenvalue as an optimization metric for trajectory generation via B-spline curves. Through planning and replanning of robot trajectory online, the method enhances the observability of multi-sensor extrinsic parameters. The effectiveness and advantages of our method have been demonstrated through numerical simulations and real-world experiments. For the benefit of the community, we have also open-sourced our code and data at https://github.com/AISLAB-sustech/Multisensor-Calibration.
>
---
#### [new 028] Autonomous 3D Moving Target Encirclement and Interception with Range measurement
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于无人机自主拦截任务，旨在解决非合作目标的3D围捕与拦截问题。通过距离测量和运动控制实现目标包围与中和。**

- **链接: [http://arxiv.org/pdf/2506.13106v1](http://arxiv.org/pdf/2506.13106v1)**

> **作者:** Fen Liu; Shenghai Yuan; Thien-Minh Nguyen; Rong Su
>
> **备注:** Paper has been accepted into IROS 2025
>
> **摘要:** Commercial UAVs are an emerging security threat as they are capable of carrying hazardous payloads or disrupting air traffic. To counter UAVs, we introduce an autonomous 3D target encirclement and interception strategy. Unlike traditional ground-guided systems, this strategy employs autonomous drones to track and engage non-cooperative hostile UAVs, which is effective in non-line-of-sight conditions, GPS denial, and radar jamming, where conventional detection and neutralization from ground guidance fail. Using two noisy real-time distances measured by drones, guardian drones estimate the relative position from their own to the target using observation and velocity compensation methods, based on anti-synchronization (AS) and an X$-$Y circular motion combined with vertical jitter. An encirclement control mechanism is proposed to enable UAVs to adaptively transition from encircling and protecting a target to encircling and monitoring a hostile target. Upon breaching a warning threshold, the UAVs may even employ a suicide attack to neutralize the hostile target. We validate this strategy through real-world UAV experiments and simulated analysis in MATLAB, demonstrating its effectiveness in detecting, encircling, and intercepting hostile drones. More details: https://youtu.be/5eHW56lPVto.
>
---
#### [new 029] Adapting by Analogy: OOD Generalization of Visuomotor Policies via Functional Correspondence
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人视觉控制任务，解决OOD环境下策略泛化问题。通过功能对应关系提升策略在新环境中的表现。**

- **链接: [http://arxiv.org/pdf/2506.12678v1](http://arxiv.org/pdf/2506.12678v1)**

> **作者:** Pranay Gupta; Henny Admoni; Andrea Bajcsy
>
> **备注:** 15 pages, 11 figures
>
> **摘要:** End-to-end visuomotor policies trained using behavior cloning have shown a remarkable ability to generate complex, multi-modal low-level robot behaviors. However, at deployment time, these policies still struggle to act reliably when faced with out-of-distribution (OOD) visuals induced by objects, backgrounds, or environment changes. Prior works in interactive imitation learning solicit corrective expert demonstrations under the OOD conditions -- but this can be costly and inefficient. We observe that task success under OOD conditions does not always warrant novel robot behaviors. In-distribution (ID) behaviors can directly be transferred to OOD conditions that share functional similarities with ID conditions. For example, behaviors trained to interact with in-distribution (ID) pens can apply to interacting with a visually-OOD pencil. The key challenge lies in disambiguating which ID observations functionally correspond to the OOD observation for the task at hand. We propose that an expert can provide this OOD-to-ID functional correspondence. Thus, instead of collecting new demonstrations and re-training at every OOD encounter, our method: (1) detects the need for feedback by first checking if current observations are OOD and then identifying whether the most similar training observations show divergent behaviors, (2) solicits functional correspondence feedback to disambiguate between those behaviors, and (3) intervenes on the OOD observations with the functionally corresponding ID observations to perform deployment-time generalization. We validate our method across diverse real-world robotic manipulation tasks with a Franka Panda robotic manipulator. Our results show that test-time functional correspondences can improve the generalization of a vision-based diffusion policy to OOD objects and environment conditions with low feedback.
>
---
#### [new 030] Goal-based Self-Adaptive Generative Adversarial Imitation Learning (Goal-SAGAIL) for Multi-goal Robotic Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文针对多目标机器人操作任务，解决示范数据不足导致的学习效率低问题，提出Goal-SAGAIL框架提升模仿学习效果。**

- **链接: [http://arxiv.org/pdf/2506.12676v1](http://arxiv.org/pdf/2506.12676v1)**

> **作者:** Yingyi Kuang; Luis J. Manso; George Vogiatzis
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** Reinforcement learning for multi-goal robot manipulation tasks poses significant challenges due to the diversity and complexity of the goal space. Techniques such as Hindsight Experience Replay (HER) have been introduced to improve learning efficiency for such tasks. More recently, researchers have combined HER with advanced imitation learning methods such as Generative Adversarial Imitation Learning (GAIL) to integrate demonstration data and accelerate training speed. However, demonstration data often fails to provide enough coverage for the goal space, especially when acquired from human teleoperation. This biases the learning-from-demonstration process toward mastering easier sub-tasks instead of tackling the more challenging ones. In this work, we present Goal-based Self-Adaptive Generative Adversarial Imitation Learning (Goal-SAGAIL), a novel framework specifically designed for multi-goal robot manipulation tasks. By integrating self-adaptive learning principles with goal-conditioned GAIL, our approach enhances imitation learning efficiency, even when limited, suboptimal demonstrations are available. Experimental results validate that our method significantly improves learning efficiency across various multi-goal manipulation scenarios -- including complex in-hand manipulation tasks -- using suboptimal demonstrations provided by both simulation and human experts.
>
---
#### [new 031] Towards a Formal Specification for Self-organized Shape Formation in Swarm Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于 swarm robotics 领域，旨在解决自组织形状形成的问题。通过形式化规范方法建模，提升系统设计与实现的准确性。**

- **链接: [http://arxiv.org/pdf/2506.13453v1](http://arxiv.org/pdf/2506.13453v1)**

> **作者:** YR Darr; MA Niazi
>
> **摘要:** The self-organization of robots for the formation of structures and shapes is a stimulating application of the swarm robotic system. It involves a large number of autonomous robots of heterogeneous behavior, coordination among them, and their interaction with the dynamic environment. This process of complex structure formation is considered a complex system, which needs to be modeled by using any modeling approach. Although the formal specification approach along with other formal methods has been used to model the behavior of robots in a swarm. However, to the best of our knowledge, the formal specification approach has not been used to model the self-organization process in swarm robotic systems for shape formation. In this paper, we use a formal specification approach to model the shape formation task of swarm robots. We use Z (Zed) language of formal specification, which is a state-based language, to model the states of the entities of the systems. We demonstrate the effectiveness of Z for the self-organized shape formation. The presented formal specification model gives the outlines for designing and implementing the swarm robotic system for the formation of complex shapes and structures. It also provides the foundation for modeling the complex shape formation process for swarm robotics using a multi-agent system in a simulation-based environment. Keywords: Swarm robotics, Self-organization, Formal specification, Complex systems
>
---
#### [new 032] C2TE: Coordinated Constrained Task Execution Design for Ordering-Flexible Multi-Vehicle Platoon Merging
- **分类: cs.RO**

- **简介: 该论文属于多车协同控制任务，解决车辆灵活合并入队问题。通过分阶段优化算法实现无序排列的车队合并，确保安全与效率。**

- **链接: [http://arxiv.org/pdf/2506.13202v1](http://arxiv.org/pdf/2506.13202v1)**

> **作者:** Bin-Bin Hu; Yanxin Zhou; Henglai Wei; Shuo Cheng; Chen Lv
>
> **摘要:** In this paper, we propose a distributed coordinated constrained task execution (C2TE) algorithm that enables a team of vehicles from different lanes to cooperatively merge into an {\it ordering-flexible platoon} maneuvering on the desired lane. Therein, the platoon is flexible in the sense that no specific spatial ordering sequences of vehicles are predetermined. To attain such a flexible platoon, we first separate the multi-vehicle platoon (MVP) merging mission into two stages, namely, pre-merging regulation and {\it ordering-flexible platoon} merging, and then formulate them into distributed constraint-based optimization problems. Particularly, by encoding longitudinal-distance regulation and same-lane collision avoidance subtasks into the corresponding control barrier function (CBF) constraints, the proposed algorithm in Stage 1 can safely enlarge sufficient longitudinal distances among adjacent vehicles. Then, by encoding lateral convergence, longitudinal-target attraction, and neighboring collision avoidance subtasks into CBF constraints, the proposed algorithm in Stage~2 can efficiently achieve the {\it ordering-flexible platoon}. Note that the {\it ordering-flexible platoon} is realized through the interaction of the longitudinal-target attraction and time-varying neighboring collision avoidance constraints simultaneously. Feasibility guarantee and rigorous convergence analysis are both provided under strong nonlinear couplings induced by flexible orderings. Finally, experiments using three autonomous mobile vehicles (AMVs) are conducted to verify the effectiveness and flexibility of the proposed algorithm, and extensive simulations are performed to demonstrate its robustness, adaptability, and scalability when tackling vehicles' sudden breakdown, new appearing, different number of lanes, mixed autonomy, and large-scale scenarios, respectively.
>
---
#### [new 033] JENGA: Object selection and pose estimation for robotic grasping from a stack
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，解决堆叠物体的选取与位姿估计问题，提出基于相机-IMU的方法并构建数据集进行评估。**

- **链接: [http://arxiv.org/pdf/2506.13425v1](http://arxiv.org/pdf/2506.13425v1)**

> **作者:** Sai Srinivas Jeevanandam; Sandeep Inuganti; Shreedhar Govil; Didier Stricker; Jason Rambach
>
> **摘要:** Vision-based robotic object grasping is typically investigated in the context of isolated objects or unstructured object sets in bin picking scenarios. However, there are several settings, such as construction or warehouse automation, where a robot needs to interact with a structured object formation such as a stack. In this context, we define the problem of selecting suitable objects for grasping along with estimating an accurate 6DoF pose of these objects. To address this problem, we propose a camera-IMU based approach that prioritizes unobstructed objects on the higher layers of stacks and introduce a dataset for benchmarking and evaluation, along with a suitable evaluation metric that combines object selection with pose accuracy. Experimental results show that although our method can perform quite well, this is a challenging problem if a completely error-free solution is needed. Finally, we show results from the deployment of our method for a brick-picking application in a construction scenario.
>
---
#### [new 034] ViTaSCOPE: Visuo-tactile Implicit Representation for In-hand Pose and Extrinsic Contact Estimation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.12239v1](http://arxiv.org/pdf/2506.12239v1)**

> **作者:** Jayjun Lee; Nima Fazeli
>
> **备注:** Accepted to RSS 2025 | Project page: https://jayjunlee.github.io/vitascope/
>
> **摘要:** Mastering dexterous, contact-rich object manipulation demands precise estimation of both in-hand object poses and external contact locations$\unicode{x2013}$tasks particularly challenging due to partial and noisy observations. We present ViTaSCOPE: Visuo-Tactile Simultaneous Contact and Object Pose Estimation, an object-centric neural implicit representation that fuses vision and high-resolution tactile feedback. By representing objects as signed distance fields and distributed tactile feedback as neural shear fields, ViTaSCOPE accurately localizes objects and registers extrinsic contacts onto their 3D geometry as contact fields. Our method enables seamless reasoning over complementary visuo-tactile cues by leveraging simulation for scalable training and zero-shot transfers to the real-world by bridging the sim-to-real gap. We evaluate our method through comprehensive simulated and real-world experiments, demonstrating its capabilities in dexterous manipulation scenarios.
>
---
#### [new 035] Towards Efficient Occupancy Mapping via Gaussian Process Latent Field Shaping
- **分类: cs.RO**

- **简介: 该论文属于机器人占用映射任务，旨在提高效率。通过直接操作潜在函数，融合自由空间信息作为先验，区分自由与未知区域，提升重建精度。**

- **链接: [http://arxiv.org/pdf/2506.13640v1](http://arxiv.org/pdf/2506.13640v1)**

> **作者:** Cedric Le Gentil; Cedric Pradalier; Timothy D. Barfoot
>
> **备注:** Presented at RSS 2025 Workshop: Gaussian Representations for Robot Autonomy: Challenges and Opportunities
>
> **摘要:** Occupancy mapping has been a key enabler of mobile robotics. Originally based on a discrete grid representation, occupancy mapping has evolved towards continuous representations that can predict the occupancy status at any location and account for occupancy correlations between neighbouring areas. Gaussian Process (GP) approaches treat this task as a binary classification problem using both observations of occupied and free space. Conceptually, a GP latent field is passed through a logistic function to obtain the output class without actually manipulating the GP latent field. In this work, we propose to act directly on the latent function to efficiently integrate free space information as a prior based on the shape of the sensor's field-of-view. A major difference with existing methods is the change in the classification problem, as we distinguish between free and unknown space. The `occupied' area is the infinitesimally thin location where the class transitions from free to unknown. We demonstrate in simulated environments that our approach is sound and leads to competitive reconstruction accuracy.
>
---
#### [new 036] Multimodal Large Language Models-Enabled UAV Swarm: Towards Efficient and Intelligent Autonomous Aerial Systems
- **分类: cs.RO**

- **简介: 该论文属于智能无人机系统任务，旨在解决 UAV 在复杂环境中的自主适应问题。通过集成多模态大语言模型，提升目标识别、导航与协作能力，并以森林灭火为例进行验证。**

- **链接: [http://arxiv.org/pdf/2506.12710v1](http://arxiv.org/pdf/2506.12710v1)**

> **作者:** Yuqi Ping; Tianhao Liang; Huahao Ding; Guangyu Lei; Junwei Wu; Xuan Zou; Kuan Shi; Rui Shao; Chiya Zhang; Weizheng Zhang; Weijie Yuan; Tingting Zhang
>
> **备注:** 8 pages, 5 figures,submitted to IEEE wcm
>
> **摘要:** Recent breakthroughs in multimodal large language models (MLLMs) have endowed AI systems with unified perception, reasoning and natural-language interaction across text, image and video streams. Meanwhile, Unmanned Aerial Vehicle (UAV) swarms are increasingly deployed in dynamic, safety-critical missions that demand rapid situational understanding and autonomous adaptation. This paper explores potential solutions for integrating MLLMs with UAV swarms to enhance the intelligence and adaptability across diverse tasks. Specifically, we first outline the fundamental architectures and functions of UAVs and MLLMs. Then, we analyze how MLLMs can enhance the UAV system performance in terms of target detection, autonomous navigation, and multi-agent coordination, while exploring solutions for integrating MLLMs into UAV systems. Next, we propose a practical case study focused on the forest fire fighting. To fully reveal the capabilities of the proposed framework, human-machine interaction, swarm task planning, fire assessment, and task execution are investigated. Finally, we discuss the challenges and future research directions for the MLLMs-enabled UAV swarm. An experiment illustration video could be found online at https://youtu.be/zwnB9ZSa5A4.
>
---
#### [new 037] VLM-SFD: VLM-Assisted Siamese Flow Diffusion Framework for Dual-Arm Cooperative Manipulation
- **分类: cs.RO**

- **简介: 该论文属于双臂协作操作任务，旨在解决复杂任务泛化与动态环境适应问题。提出VLM-SFD框架，通过Siamese流扩散网络和视觉语言模型实现高效模仿学习。**

- **链接: [http://arxiv.org/pdf/2506.13428v1](http://arxiv.org/pdf/2506.13428v1)**

> **作者:** Jiaming Chen; Yiyu Jiang; Aoshen Huang; Yang Li; Wei Pan
>
> **摘要:** Dual-arm cooperative manipulation holds great promise for tackling complex real-world tasks that demand seamless coordination and adaptive dynamics. Despite substantial progress in learning-based motion planning, most approaches struggle to generalize across diverse manipulation tasks and adapt to dynamic, unstructured environments, particularly in scenarios involving interactions between two objects such as assembly, tool use, and bimanual grasping. To address these challenges, we introduce a novel VLM-Assisted Siamese Flow Diffusion (VLM-SFD) framework for efficient imitation learning in dual-arm cooperative manipulation. The proposed VLM-SFD framework exhibits outstanding adaptability, significantly enhancing the ability to rapidly adapt and generalize to diverse real-world tasks from only a minimal number of human demonstrations. Specifically, we propose a Siamese Flow Diffusion Network (SFDNet) employs a dual-encoder-decoder Siamese architecture to embed two target objects into a shared latent space, while a diffusion-based conditioning process-conditioned by task instructions-generates two-stream object-centric motion flows that guide dual-arm coordination. We further design a dynamic task assignment strategy that seamlessly maps the predicted 2D motion flows into 3D space and incorporates a pre-trained vision-language model (VLM) to adaptively assign the optimal motion to each robotic arm over time. Experiments validate the effectiveness of the proposed method, demonstrating its ability to generalize to diverse manipulation tasks while maintaining high efficiency and adaptability. The code and demo videos are publicly available on our project website https://sites.google.com/view/vlm-sfd/.
>
---
#### [new 038] Edge Nearest Neighbor in Sampling-Based Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，解决如何高效寻找最近邻的问题。通过实现一种新的邻域查找方法，提升RRT和RRG算法的效率与探索能力。**

- **链接: [http://arxiv.org/pdf/2506.13753v1](http://arxiv.org/pdf/2506.13753v1)**

> **作者:** Stav Ashur; Nancy M. Amato; Sariel Har-Peled
>
> **摘要:** Neighborhood finders and nearest neighbor queries are fundamental parts of sampling based motion planning algorithms. Using different distance metrics or otherwise changing the definition of a neighborhood produces different algorithms with unique empiric and theoretical properties. In \cite{l-pa-06} LaValle suggests a neighborhood finder for the Rapidly-exploring Random Tree RRT algorithm \cite{l-rrtnt-98} which finds the nearest neighbor of the sampled point on the swath of the tree, that is on the set of all of the points on the tree edges, using a hierarchical data structure. In this paper we implement such a neighborhood finder and show, theoretically and experimentally, that this results in more efficient algorithms, and suggest a variant of the Rapidly-exploring Random Graph RRG algorithm \cite{f-isaom-10} that better exploits the exploration properties of the newly described subroutine for finding narrow passages.
>
---
#### [new 039] Deep Fusion of Ultra-Low-Resolution Thermal Camera and Gyroscope Data for Lighting-Robust and Compute-Efficient Rotational Odometry
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于旋转里程计任务，旨在解决光照变化和计算资源受限问题。通过融合热成像与陀螺仪数据，提升鲁棒性和效率。**

- **链接: [http://arxiv.org/pdf/2506.12536v1](http://arxiv.org/pdf/2506.12536v1)**

> **作者:** Farida Mohsen; Ali Safa
>
> **摘要:** Accurate rotational odometry is crucial for autonomous robotic systems, particularly for small, power-constrained platforms such as drones and mobile robots. This study introduces thermal-gyro fusion, a novel sensor fusion approach that integrates ultra-low-resolution thermal imaging with gyroscope readings for rotational odometry. Unlike RGB cameras, thermal imaging is invariant to lighting conditions and, when fused with gyroscopic data, mitigates drift which is a common limitation of inertial sensors. We first develop a multimodal data acquisition system to collect synchronized thermal and gyroscope data, along with rotational speed labels, across diverse environments. Subsequently, we design and train a lightweight Convolutional Neural Network (CNN) that fuses both modalities for rotational speed estimation. Our analysis demonstrates that thermal-gyro fusion enables a significant reduction in thermal camera resolution without significantly compromising accuracy, thereby improving computational efficiency and memory utilization. These advantages make our approach well-suited for real-time deployment in resource-constrained robotic systems. Finally, to facilitate further research, we publicly release our dataset as supplementary material.
>
---
#### [new 040] Learning Swing-up Maneuvers for a Suspended Aerial Manipulation Platform in a Hierarchical Control Framework
- **分类: cs.RO**

- **简介: 该论文研究悬空空中操作平台的摆动上位控制任务，通过分层控制框架结合强化学习解决无法仅靠推力到达目标位置的问题。**

- **链接: [http://arxiv.org/pdf/2506.13478v1](http://arxiv.org/pdf/2506.13478v1)**

> **作者:** Hemjyoti Das; Minh Nhat Vu; Christian Ott
>
> **备注:** 6 pages, 10 figures
>
> **摘要:** In this work, we present a novel approach to augment a model-based control method with a reinforcement learning (RL) agent and demonstrate a swing-up maneuver with a suspended aerial manipulation platform. These platforms are targeted towards a wide range of applications on construction sites involving cranes, with swing-up maneuvers allowing it to perch at a given location, inaccessible with purely the thrust force of the platform. Our proposed approach is based on a hierarchical control framework, which allows different tasks to be executed according to their assigned priorities. An RL agent is then subsequently utilized to adjust the reference set-point of the lower-priority tasks to perform the swing-up maneuver, which is confined in the nullspace of the higher-priority tasks, such as maintaining a specific orientation and position of the end-effector. Our approach is validated using extensive numerical simulation studies.
>
---
#### [new 041] On-board Sonar Data Classification for Path Following in Underwater Vehicles using Fast Interval Type-2 Fuzzy Extreme Learning Machine
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.12762v1](http://arxiv.org/pdf/2506.12762v1)**

> **作者:** Adrian Rubio-Solis; Luciano Nava-Balanzar; Tomas Salgado-Jimenez
>
> **摘要:** In autonomous underwater missions, the successful completion of predefined paths mainly depends on the ability of underwater vehicles to recognise their surroundings. In this study, we apply the concept of Fast Interval Type-2 Fuzzy Extreme Learning Machine (FIT2-FELM) to train a Takagi-Sugeno-Kang IT2 Fuzzy Inference System (TSK IT2-FIS) for on-board sonar data classification using an underwater vehicle called BlueROV2. The TSK IT2-FIS is integrated into a Hierarchical Navigation Strategy (HNS) as the main navigation engine to infer local motions and provide the BlueROV2 with full autonomy to follow an obstacle-free trajectory in a water container of 2.5m x 2.5m x 3.5m. Compared to traditional navigation architectures, using the proposed method, we observe a robust path following behaviour in the presence of uncertainty and noise. We found that the proposed approach provides the BlueROV with a more complete sensory picture about its surroundings while real-time navigation planning is performed by the concurrent execution of two or more tasks.
>
---
#### [new 042] ROSA: Harnessing Robot States for Vision-Language and Action Alignment
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言-动作对齐任务，旨在解决VLA模型与机器人动作空间的对齐问题。通过引入机器人状态估计数据，提升模型性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.13679v1](http://arxiv.org/pdf/2506.13679v1)**

> **作者:** Yuqing Wen; Kefan Gu; Haoxuan Liu; Yucheng Zhao; Tiancai Wang; Haoqiang Fan; Xiaoyan Sun
>
> **摘要:** Vision-Language-Action (VLA) models have recently made significant advance in multi-task, end-to-end robotic control, due to the strong generalization capabilities of Vision-Language Models (VLMs). A fundamental challenge in developing such models is effectively aligning the vision-language space with the robotic action space. Existing approaches typically rely on directly fine-tuning VLMs using expert demonstrations. However, this strategy suffers from a spatio-temporal gap, resulting in considerable data inefficiency and heavy reliance on human labor. Spatially, VLMs operate within a high-level semantic space, whereas robotic actions are grounded in low-level 3D physical space; temporally, VLMs primarily interpret the present, while VLA models anticipate future actions. To overcome these challenges, we propose a novel training paradigm, ROSA, which leverages robot state estimation to improve alignment between vision-language and action spaces. By integrating robot state estimation data obtained via an automated process, ROSA enables the VLA model to gain enhanced spatial understanding and self-awareness, thereby boosting performance and generalization. Extensive experiments in both simulated and real-world environments demonstrate the effectiveness of ROSA, particularly in low-data regimes.
>
---
#### [new 043] Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins
- **分类: cs.RO**

- **简介: 该论文属于开放世界机器人操作任务，旨在解决VLMs在低层控制上的不足。通过结合语义推理与数字孪生，生成可行轨迹并优化控制。**

- **链接: [http://arxiv.org/pdf/2506.13761v1](http://arxiv.org/pdf/2506.13761v1)**

> **作者:** Chuanruo Ning; Kuan Fang; Wei-Chiu Ma
>
> **摘要:** Recent advancements in open-world robot manipulation have been largely driven by vision-language models (VLMs). While these models exhibit strong generalization ability in high-level planning, they struggle to predict low-level robot controls due to limited physical-world understanding. To address this issue, we propose a model predictive control framework for open-world manipulation that combines the semantic reasoning capabilities of VLMs with physically-grounded, interactive digital twins of the real-world environments. By constructing and simulating the digital twins, our approach generates feasible motion trajectories, simulates corresponding outcomes, and prompts the VLM with future observations to evaluate and select the most suitable outcome based on language instructions of the task. To further enhance the capability of pre-trained VLMs in understanding complex scenes for robotic control, we leverage the flexible rendering capabilities of the digital twin to synthesize the scene at various novel, unoccluded viewpoints. We validate our approach on a diverse set of complex manipulation tasks, demonstrating superior performance compared to baseline methods for language-conditioned robotic control using VLMs.
>
---
#### [new 044] Underwater target 6D State Estimation via UUV Attitude Enhance Observability
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.13105v1](http://arxiv.org/pdf/2506.13105v1)**

> **作者:** Fen Liu; Chengfeng Jia; Na Zhang; Shenghai Yuan; Rong Su
>
> **备注:** Paper has been accepted in IROS 2025
>
> **摘要:** Accurate relative state observation of Unmanned Underwater Vehicles (UUVs) for tracking uncooperative targets remains a significant challenge due to the absence of GPS, complex underwater dynamics, and sensor limitations. Existing localization approaches rely on either global positioning infrastructure or multi-UUV collaboration, both of which are impractical for a single UUV operating in large or unknown environments. To address this, we propose a novel persistent relative 6D state estimation framework that enables a single UUV to estimate its relative motion to a non-cooperative target using only successive noisy range measurements from two monostatic sonar sensors. Our key contribution is an observability-enhanced attitude control strategy, which optimally adjusts the UUV's orientation to improve the observability of relative state estimation using a Kalman filter, effectively mitigating the impact of sensor noise and drift accumulation. Additionally, we introduce a rigorously proven Lyapunov-based tracking control strategy that guarantees long-term stability by ensuring that the UUV maintains an optimal measurement range, preventing localization errors from diverging over time. Through theoretical analysis and simulations, we demonstrate that our method significantly improves 6D relative state estimation accuracy and robustness compared to conventional approaches. This work provides a scalable, infrastructure-free solution for UUVs tracking uncooperative targets underwater.
>
---
#### [new 045] CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding
- **分类: cs.RO**

- **简介: 该论文属于机器人多模态决策任务，旨在解决VLA模型推理速度慢的问题。通过一致性蒸馏和早停解码策略，实现高效推理。**

- **链接: [http://arxiv.org/pdf/2506.13725v1](http://arxiv.org/pdf/2506.13725v1)**

> **作者:** Wenxuan Song; Jiayi Chen; Pengxiang Ding; Yuxin Huang; Han Zhao; Donglin Wang; Haoang Li
>
> **备注:** 16 pages
>
> **摘要:** In recent years, Vision-Language-Action (VLA) models have become a vital research direction in robotics due to their impressive multimodal understanding and generalization capabilities. Despite the progress, their practical deployment is severely constrained by inference speed bottlenecks, particularly in high-frequency and dexterous manipulation tasks. While recent studies have explored Jacobi decoding as a more efficient alternative to traditional autoregressive decoding, its practical benefits are marginal due to the lengthy iterations. To address it, we introduce consistency distillation training to predict multiple correct action tokens in each iteration, thereby achieving acceleration. Besides, we design mixed-label supervision to mitigate the error accumulation during distillation. Although distillation brings acceptable speedup, we identify that certain inefficient iterations remain a critical bottleneck. To tackle this, we propose an early-exit decoding strategy that moderately relaxes convergence conditions, which further improves average inference efficiency. Experimental results show that the proposed method achieves more than 4 times inference acceleration across different baselines while maintaining high task success rates in both simulated and real-world robot tasks. These experiments validate that our approach provides an efficient and general paradigm for accelerating multimodal decision-making in robotics. Our project page is available at https://irpn-eai.github.io/CEED-VLA/.
>
---
#### [new 046] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making
- **分类: cs.RO; cs.AI; I.2.9; I.2.10; I.4.8; H.5.2**

- **简介: 该论文属于机器人决策任务，旨在解决传统方法丢失任务细节的问题。提出AntiGrounding框架，将动作直接映射到VLM空间，提升决策效果。**

- **链接: [http://arxiv.org/pdf/2506.12374v1](http://arxiv.org/pdf/2506.12374v1)**

> **作者:** Wenbo Li; Shiyi Wang; Yiteng Chen; Huiping Zhuang; Qingyao Wu
>
> **备注:** submitted to NeurIPS 2025
>
> **摘要:** Vision-Language Models (VLMs) encode knowledge and reasoning capabilities for robotic manipulation within high-dimensional representation spaces. However, current approaches often project them into compressed intermediate representations, discarding important task-specific information such as fine-grained spatial or semantic details. To address this, we propose AntiGrounding, a new framework that reverses the instruction grounding process. It lifts candidate actions directly into the VLM representation space, renders trajectories from multiple views, and uses structured visual question answering for instruction-based decision making. This enables zero-shot synthesis of optimal closed-loop robot trajectories for new tasks. We also propose an offline policy refinement module that leverages past experience to enhance long-term performance. Experiments in both simulation and real-world environments show that our method outperforms baselines across diverse robotic manipulation tasks.
>
---
#### [new 047] IKDiffuser: Fast and Diverse Inverse Kinematics Solution Generation for Multi-arm Robotic Systems
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人学任务，解决多机械臂逆运动学问题。提出IKDiffuser模型，实现快速且多样化的解生成。**

- **链接: [http://arxiv.org/pdf/2506.13087v1](http://arxiv.org/pdf/2506.13087v1)**

> **作者:** Zeyu Zhang; Ziyuan Jiao
>
> **备注:** under review
>
> **摘要:** Solving Inverse Kinematics (IK) problems is fundamental to robotics, but has primarily been successful with single serial manipulators. For multi-arm robotic systems, IK remains challenging due to complex self-collisions, coupled joints, and high-dimensional redundancy. These complexities make traditional IK solvers slow, prone to failure, and lacking in solution diversity. In this paper, we present IKDiffuser, a diffusion-based model designed for fast and diverse IK solution generation for multi-arm robotic systems. IKDiffuser learns the joint distribution over the configuration space, capturing complex dependencies and enabling seamless generalization to multi-arm robotic systems of different structures. In addition, IKDiffuser can incorporate additional objectives during inference without retraining, offering versatility and adaptability for task-specific requirements. In experiments on 6 different multi-arm systems, the proposed IKDiffuser achieves superior solution accuracy, precision, diversity, and computational efficiency compared to existing solvers. The proposed IKDiffuser framework offers a scalable, unified approach to solving multi-arm IK problems, facilitating the potential of multi-arm robotic systems in real-time manipulation tasks.
>
---
#### [new 048] Disturbance-aware minimum-time planning strategies for motorsport vehicles with probabilistic safety certificates
- **分类: cs.RO**

- **简介: 该论文属于车辆轨迹优化任务，旨在解决高精度赛车路径规划中的不确定性问题。通过引入鲁棒性，提出两种策略以确保安全性和最短圈速。**

- **链接: [http://arxiv.org/pdf/2506.13622v1](http://arxiv.org/pdf/2506.13622v1)**

> **作者:** Martino Gulisano; Matteo Masoni; Marco Gabiccini; Massimo Guiggiani
>
> **备注:** 24 pages, 11 figures, paper under review
>
> **摘要:** This paper presents a disturbance-aware framework that embeds robustness into minimum-lap-time trajectory optimization for motorsport. Two formulations are introduced. (i) Open-loop, horizon-based covariance propagation uses worst-case uncertainty growth over a finite window to tighten tire-friction and track-limit constraints. (ii) Closed-loop, covariance-aware planning incorporates a time-varying LQR feedback law in the optimizer, providing a feedback-consistent estimate of disturbance attenuation and enabling sharper yet reliable constraint tightening. Both methods yield reference trajectories for human or artificial drivers: in autonomous applications the modelled controller can replicate the on-board implementation, while for human driving accuracy increases with the extent to which the driver can be approximated by the assumed time-varying LQR policy. Computational tests on a representative Barcelona-Catalunya sector show that both schemes meet the prescribed safety probability, yet the closed-loop variant incurs smaller lap-time penalties than the more conservative open-loop solution, while the nominal (non-robust) trajectory remains infeasible under the same uncertainties. By accounting for uncertainty growth and feedback action during planning, the proposed framework delivers trajectories that are both performance-optimal and probabilistically safe, advancing minimum-time optimization toward real-world deployment in high-performance motorsport and autonomous racing.
>
---
#### [new 049] Equilibrium-Driven Smooth Separation and Navigation of Marsupial Robotic Systems
- **分类: cs.RO**

- **简介: 该论文属于机器人协同控制任务，解决载荷分离与导航问题。设计了基于平衡点的控制器，实现平稳分离与目标导航。**

- **链接: [http://arxiv.org/pdf/2506.13198v1](http://arxiv.org/pdf/2506.13198v1)**

> **作者:** Bin-Bin Hu; Bayu Jayawardhana; Ming Cao
>
> **摘要:** In this paper, we propose an equilibrium-driven controller that enables a marsupial carrier-passenger robotic system to achieve smooth carrier-passenger separation and then to navigate the passenger robot toward a predetermined target point. Particularly, we design a potential gradient in the form of a cubic polynomial for the passenger's controller as a function of the carrier-passenger and carrier-target distances in the moving carrier's frame. This introduces multiple equilibrium points corresponding to the zero state of the error dynamic system during carrier-passenger separation. The change of equilibrium points is associated with the change in their attraction regions, enabling smooth carrier-passenger separation and afterwards seamless navigation toward the target. Finally, simulations demonstrate the effectiveness and adaptability of the proposed controller in environments containing obstacles.
>
---
#### [new 050] Uncertainty-Informed Active Perception for Open Vocabulary Object Goal Navigation
- **分类: cs.RO**

- **简介: 该论文属于室内物体目标导航任务，旨在解决视觉-语言模型的语义不确定性问题。通过引入概率传感器模型和不确定性感知的探索策略，提升导航性能。**

- **链接: [http://arxiv.org/pdf/2506.13367v1](http://arxiv.org/pdf/2506.13367v1)**

> **作者:** Utkarsh Bajpai; Julius Rückin; Cyrill Stachniss; Marija Popović
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Mobile robots exploring indoor environments increasingly rely on vision-language models to perceive high-level semantic cues in camera images, such as object categories. Such models offer the potential to substantially advance robot behaviour for tasks such as object-goal navigation (ObjectNav), where the robot must locate objects specified in natural language by exploring the environment. Current ObjectNav methods heavily depend on prompt engineering for perception and do not address the semantic uncertainty induced by variations in prompt phrasing. Ignoring semantic uncertainty can lead to suboptimal exploration, which in turn limits performance. Hence, we propose a semantic uncertainty-informed active perception pipeline for ObjectNav in indoor environments. We introduce a novel probabilistic sensor model for quantifying semantic uncertainty in vision-language models and incorporate it into a probabilistic geometric-semantic map to enhance spatial understanding. Based on this map, we develop a frontier exploration planner with an uncertainty-informed multi-armed bandit objective to guide efficient object search. Experimental results demonstrate that our method achieves ObjectNav success rates comparable to those of state-of-the-art approaches, without requiring extensive prompt engineering.
>
---
#### [new 051] Cognitive Synergy Architecture: SEGO for Human-Centric Collaborative Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于人机协作机器人领域，旨在解决环境认知与语义理解问题。提出SEGO架构，融合几何感知、语义推理与解释生成，实现动态语义场景图构建。**

- **链接: [http://arxiv.org/pdf/2506.13149v1](http://arxiv.org/pdf/2506.13149v1)**

> **作者:** Jaehong Oh
>
> **摘要:** This paper presents SEGO (Semantic Graph Ontology), a cognitive mapping architecture designed to integrate geometric perception, semantic reasoning, and explanation generation into a unified framework for human-centric collaborative robotics. SEGO constructs dynamic cognitive scene graphs that represent not only the spatial configuration of the environment but also the semantic relations and ontological consistency among detected objects. The architecture seamlessly combines SLAM-based localization, deep-learning-based object detection and tracking, and ontology-driven reasoning to enable real-time, semantically coherent mapping.
>
---
#### [new 052] Constrained Diffusers for Safe Planning and Control
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于安全规划与控制任务，解决扩散模型在约束下的安全性问题。通过引入约束优化方法和控制屏障函数，提升模型的安全性和计算效率。**

- **链接: [http://arxiv.org/pdf/2506.12544v1](http://arxiv.org/pdf/2506.12544v1)**

> **作者:** Jichen Zhang; Liqun Zhao; Antonis Papachristodoulou; Jack Umenberger
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Diffusion models have shown remarkable potential in planning and control tasks due to their ability to represent multimodal distributions over actions and trajectories. However, ensuring safety under constraints remains a critical challenge for diffusion models. This paper proposes Constrained Diffusers, a novel framework that incorporates constraints into pre-trained diffusion models without retraining or architectural modifications. Inspired by constrained optimization, we apply a constrained Langevin sampling mechanism for the reverse diffusion process that jointly optimizes the trajectory and realizes constraint satisfaction through three iterative algorithms: projected method, primal-dual method and augmented Lagrangian approaches. In addition, we incorporate discrete control barrier functions as constraints for constrained diffusers to guarantee safety in online implementation. Experiments in Maze2D, locomotion, and pybullet ball running tasks demonstrate that our proposed methods achieve constraint satisfaction with less computation time, and are competitive to existing methods in environments with static and time-varying constraints.
>
---
#### [new 053] SuperPoint-SLAM3: Augmenting ORB-SLAM3 with Deep Features, Adaptive NMS, and Learning-Based Loop Closure
- **分类: cs.CV; cs.RO; I.2.10; I.4.8; I.2.9**

- **简介: 该论文属于视觉SLAM任务，旨在提升ORBSLAM3在极端条件下的精度。通过引入深度特征、自适应NMS和学习环路闭合模块实现改进。**

- **链接: [http://arxiv.org/pdf/2506.13089v1](http://arxiv.org/pdf/2506.13089v1)**

> **作者:** Shahram Najam Syed; Ishir Roongta; Kavin Ravie; Gangadhar Nageswar
>
> **备注:** 10 pages, 6 figures, code at https://github.com/shahram95/SuperPointSLAM3
>
> **摘要:** Visual simultaneous localization and mapping (SLAM) must remain accurate under extreme viewpoint, scale and illumination variations. The widely adopted ORB-SLAM3 falters in these regimes because it relies on hand-crafted ORB keypoints. We introduce SuperPoint-SLAM3, a drop-in upgrade that (i) replaces ORB with the self-supervised SuperPoint detector--descriptor, (ii) enforces spatially uniform keypoints via adaptive non-maximal suppression (ANMS), and (iii) integrates a lightweight NetVLAD place-recognition head for learning-based loop closure. On the KITTI Odometry benchmark SuperPoint-SLAM3 reduces mean translational error from 4.15% to 0.34% and mean rotational error from 0.0027 deg/m to 0.0010 deg/m. On the EuRoC MAV dataset it roughly halves both errors across every sequence (e.g., V2\_03: 1.58% -> 0.79%). These gains confirm that fusing modern deep features with a learned loop-closure module markedly improves ORB-SLAM3 accuracy while preserving its real-time operation. Implementation, pretrained weights and reproducibility scripts are available at https://github.com/shahram95/SuperPointSLAM3.
>
---
#### [new 054] Open-Set LiDAR Panoptic Segmentation Guided by Uncertainty-Aware Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于开放集LiDAR语义分割任务，解决未知物体检测问题。提出ULOPS框架，利用不确定性学习区分已知与未知物体。**

- **链接: [http://arxiv.org/pdf/2506.13265v1](http://arxiv.org/pdf/2506.13265v1)**

> **作者:** Rohit Mohan; Julia Hindel; Florian Drews; Claudius Gläser; Daniele Cattaneo; Abhinav Valada
>
> **摘要:** Autonomous vehicles that navigate in open-world environments may encounter previously unseen object classes. However, most existing LiDAR panoptic segmentation models rely on closed-set assumptions, failing to detect unknown object instances. In this work, we propose ULOPS, an uncertainty-guided open-set panoptic segmentation framework that leverages Dirichlet-based evidential learning to model predictive uncertainty. Our architecture incorporates separate decoders for semantic segmentation with uncertainty estimation, embedding with prototype association, and instance center prediction. During inference, we leverage uncertainty estimates to identify and segment unknown instances. To strengthen the model's ability to differentiate between known and unknown objects, we introduce three uncertainty-driven loss functions. Uniform Evidence Loss to encourage high uncertainty in unknown regions. Adaptive Uncertainty Separation Loss ensures a consistent difference in uncertainty estimates between known and unknown objects at a global scale. Contrastive Uncertainty Loss refines this separation at the fine-grained level. To evaluate open-set performance, we extend benchmark settings on KITTI-360 and introduce a new open-set evaluation for nuScenes. Extensive experiments demonstrate that ULOPS consistently outperforms existing open-set LiDAR panoptic segmentation methods.
>
---
#### [new 055] Can you see how I learn? Human observers' inferences about Reinforcement Learning agents' learning processes
- **分类: cs.HC; cs.AI; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决人类如何理解强化学习代理的学习过程。通过实验分析人类对RL代理学习行为的推理，提出可解释的RL系统设计建议。**

- **链接: [http://arxiv.org/pdf/2506.13583v1](http://arxiv.org/pdf/2506.13583v1)**

> **作者:** Bernhard Hilpert; Muhan Hou; Kim Baraka; Joost Broekens
>
> **摘要:** Reinforcement Learning (RL) agents often exhibit learning behaviors that are not intuitively interpretable by human observers, which can result in suboptimal feedback in collaborative teaching settings. Yet, how humans perceive and interpret RL agent's learning behavior is largely unknown. In a bottom-up approach with two experiments, this work provides a data-driven understanding of the factors of human observers' understanding of the agent's learning process. A novel, observation-based paradigm to directly assess human inferences about agent learning was developed. In an exploratory interview study (\textit{N}=9), we identify four core themes in human interpretations: Agent Goals, Knowledge, Decision Making, and Learning Mechanisms. A second confirmatory study (\textit{N}=34) applied an expanded version of the paradigm across two tasks (navigation/manipulation) and two RL algorithms (tabular/function approximation). Analyses of 816 responses confirmed the reliability of the paradigm and refined the thematic framework, revealing how these themes evolve over time and interrelate. Our findings provide a human-centered understanding of how people make sense of agent learning, offering actionable insights for designing interpretable RL systems and improving transparency in Human-Robot Interaction.
>
---
#### [new 056] Bridging Data-Driven and Physics-Based Models: A Consensus Multi-Model Kalman Filter for Robust Vehicle State Estimation
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于车辆状态估计任务，旨在解决传统模型在复杂场景下的局限性。通过融合物理模型与数据驱动模型，提升估计的准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.12862v1](http://arxiv.org/pdf/2506.12862v1)**

> **作者:** Farid Mafi; Ladan Khoshnevisan; Mohammad Pirani; Amir Khajepour
>
> **摘要:** Vehicle state estimation presents a fundamental challenge for autonomous driving systems, requiring both physical interpretability and the ability to capture complex nonlinear behaviors across diverse operating conditions. Traditional methodologies often rely exclusively on either physics-based or data-driven models, each with complementary strengths and limitations that become most noticeable during critical scenarios. This paper presents a novel consensus multi-model Kalman filter framework that integrates heterogeneous model types to leverage their complementary strengths while minimizing individual weaknesses. We introduce two distinct methodologies for handling covariance propagation in data-driven models: a Koopman operator-based linearization approach enabling analytical covariance propagation, and an ensemble-based method providing unified uncertainty quantification across model types without requiring pretraining. Our approach implements an iterative consensus fusion procedure that dynamically weighs different models based on their demonstrated reliability in current operating conditions. The experimental results conducted on an electric all-wheel-drive Equinox vehicle demonstrate performance improvements over single-model techniques, with particularly significant advantages during challenging maneuvers and varying road conditions, confirming the effectiveness and robustness of the proposed methodology for safety-critical autonomous driving applications.
>
---
#### [new 057] Trust-MARL: Trust-Based Multi-Agent Reinforcement Learning Framework for Cooperative On-Ramp Merging Control in Heterogeneous Traffic Flow
- **分类: cs.MA; cs.AI; cs.ET; cs.GT; cs.RO**

- **简介: 该论文属于智能交通中的协作变道控制任务，旨在解决异质交通流中CAV与HVs的协同问题，提出Trust-MARL框架提升安全与效率。**

- **链接: [http://arxiv.org/pdf/2506.12600v1](http://arxiv.org/pdf/2506.12600v1)**

> **作者:** Jie Pan; Tianyi Wang; Christian Claudel; Jing Shi
>
> **备注:** 34 pages, 7 figures, 4 tables
>
> **摘要:** Intelligent transportation systems require connected and automated vehicles (CAVs) to conduct safe and efficient cooperation with human-driven vehicles (HVs) in complex real-world traffic environments. However, the inherent unpredictability of human behaviour, especially at bottlenecks such as highway on-ramp merging areas, often disrupts traffic flow and compromises system performance. To address the challenge of cooperative on-ramp merging in heterogeneous traffic environments, this study proposes a trust-based multi-agent reinforcement learning (Trust-MARL) framework. At the macro level, Trust-MARL enhances global traffic efficiency by leveraging inter-agent trust to improve bottleneck throughput and mitigate traffic shockwave through emergent group-level coordination. At the micro level, a dynamic trust mechanism is designed to enable CAVs to adjust their cooperative strategies in response to real-time behaviors and historical interactions with both HVs and other CAVs. Furthermore, a trust-triggered game-theoretic decision-making module is integrated to guide each CAV in adapting its cooperation factor and executing context-aware lane-changing decisions under safety, comfort, and efficiency constraints. An extensive set of ablation studies and comparative experiments validates the effectiveness of the proposed Trust-MARL approach, demonstrating significant improvements in safety, efficiency, comfort, and adaptability across varying CAV penetration rates and traffic densities.
>
---
#### [new 058] Parallel Branch Model Predictive Control on GPUs
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于控制领域，解决模型预测控制的计算效率问题。通过GPU并行加速，提升求解速度，适用于自动驾驶等场景。**

- **链接: [http://arxiv.org/pdf/2506.13624v1](http://arxiv.org/pdf/2506.13624v1)**

> **作者:** Luyao Zhang; Chenghuai Lin; Sergio Grammatico
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** We present a parallel GPU-accelerated solver for branch Model Predictive Control problems. Based on iterative LQR methods, our solver exploits the tree-sparse structure and implements temporal parallelism using the parallel scan algorithm. Consequently, the proposed solver enables parallelism across both the prediction horizon and the scenarios. In addition, we utilize an augmented Lagrangian method to handle general inequality constraints. We compare our solver with state-of-the-art numerical solvers in two automated driving applications. The numerical results demonstrate that, compared to CPU-based solvers, our solver achieves competitive performance for problems with short horizons and small-scale trees, while outperforming other solvers on large-scale problems.
>
---
#### [new 059] Multimodal "Puppeteer": An Exploration of Robot Teleoperation Via Virtual Counterpart with LLM-Driven Voice and Gesture Interaction in Augmented Reality
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在提升AR环境下机器人远程操作的效率与体验。通过结合语音和手势的多模态交互，优化用户与机器人的协作性能。**

- **链接: [http://arxiv.org/pdf/2506.13189v1](http://arxiv.org/pdf/2506.13189v1)**

> **作者:** Yuchong Zhang; Bastian Orthmann; Shichen Ji; Michael Welle; Jonne Van Haastregt; Danica Kragic
>
> **备注:** This work has been submitted to the IEEE TVCG for possible publication
>
> **摘要:** The integration of robotics and augmented reality (AR) holds transformative potential for advancing human-robot interaction (HRI), offering enhancements in usability, intuitiveness, accessibility, and collaborative task performance. This paper introduces and evaluates a novel multimodal AR-based robot puppeteer framework that enables intuitive teleoperation via virtual counterpart through large language model (LLM)-driven voice commands and hand gesture interactions. Utilizing the Meta Quest 3, users interact with a virtual counterpart robot in real-time, effectively "puppeteering" its physical counterpart within an AR environment. We conducted a within-subject user study with 42 participants performing robotic cube pick-and-place with pattern matching tasks under two conditions: gesture-only interaction and combined voice-and-gesture interaction. Both objective performance metrics and subjective user experience (UX) measures were assessed, including an extended comparative analysis between roboticists and non-roboticists. The results provide key insights into how multimodal input influences contextual task efficiency, usability, and user satisfaction in AR-based HRI. Our findings offer practical design implications for designing effective AR-enhanced HRI systems.
>
---
#### [new 060] UAV Object Detection and Positioning in a Mining Industrial Metaverse with Custom Geo-Referenced Data
- **分类: eess.IV; cs.AI; cs.ET; cs.RO**

- **简介: 该论文属于无人机目标检测与定位任务，解决矿区高精度地理信息获取问题，结合UAV、LiDAR和深度学习实现三维重建与对象定位。**

- **链接: [http://arxiv.org/pdf/2506.13505v1](http://arxiv.org/pdf/2506.13505v1)**

> **作者:** Vasiliki Balaska; Ioannis Tsampikos Papapetros; Katerina Maria Oikonomou; Loukas Bampis; Antonios Gasteratos
>
> **摘要:** The mining sector increasingly adopts digital tools to improve operational efficiency, safety, and data-driven decision-making. One of the key challenges remains the reliable acquisition of high-resolution, geo-referenced spatial information to support core activities such as extraction planning and on-site monitoring. This work presents an integrated system architecture that combines UAV-based sensing, LiDAR terrain modeling, and deep learning-based object detection to generate spatially accurate information for open-pit mining environments. The proposed pipeline includes geo-referencing, 3D reconstruction, and object localization, enabling structured spatial outputs to be integrated into an industrial digital twin platform. Unlike traditional static surveying methods, the system offers higher coverage and automation potential, with modular components suitable for deployment in real-world industrial contexts. While the current implementation operates in post-flight batch mode, it lays the foundation for real-time extensions. The system contributes to the development of AI-enhanced remote sensing in mining by demonstrating a scalable and field-validated geospatial data workflow that supports situational awareness and infrastructure safety.
>
---
#### [new 061] Enhancing Rating-Based Reinforcement Learning to Effectively Leverage Feedback from Large Vision-Language Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决奖励函数设计难题。通过引入AI生成的反馈，提升RL的效率与自主性。**

- **链接: [http://arxiv.org/pdf/2506.12822v1](http://arxiv.org/pdf/2506.12822v1)**

> **作者:** Tung Minh Luu; Younghwan Lee; Donghoon Lee; Sunho Kim; Min Jun Kim; Chang D. Yoo
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Designing effective reward functions remains a fundamental challenge in reinforcement learning (RL), as it often requires extensive human effort and domain expertise. While RL from human feedback has been successful in aligning agents with human intent, acquiring high-quality feedback is costly and labor-intensive, limiting its scalability. Recent advancements in foundation models present a promising alternative--leveraging AI-generated feedback to reduce reliance on human supervision in reward learning. Building on this paradigm, we introduce ERL-VLM, an enhanced rating-based RL method that effectively learns reward functions from AI feedback. Unlike prior methods that rely on pairwise comparisons, ERL-VLM queries large vision-language models (VLMs) for absolute ratings of individual trajectories, enabling more expressive feedback and improved sample efficiency. Additionally, we propose key enhancements to rating-based RL, addressing instability issues caused by data imbalance and noisy labels. Through extensive experiments across both low-level and high-level control tasks, we demonstrate that ERL-VLM significantly outperforms existing VLM-based reward generation methods. Our results demonstrate the potential of AI feedback for scaling RL with minimal human intervention, paving the way for more autonomous and efficient reward learning.
>
---
#### [new 062] Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决多摄像头数据高效编码问题。提出基于三平面的token化方法，减少token数量，提升推理速度，同时保持规划精度。**

- **链接: [http://arxiv.org/pdf/2506.12251v1](http://arxiv.org/pdf/2506.12251v1)**

> **作者:** Boris Ivanovic; Cristiano Saltori; Yurong You; Yan Wang; Wenjie Luo; Marco Pavone
>
> **备注:** 12 pages, 10 figures, 5 tables
>
> **摘要:** Autoregressive Transformers are increasingly being deployed as end-to-end robot and autonomous vehicle (AV) policy architectures, owing to their scalability and potential to leverage internet-scale pretraining for generalization. Accordingly, tokenizing sensor data efficiently is paramount to ensuring the real-time feasibility of such architectures on embedded hardware. To this end, we present an efficient triplane-based multi-camera tokenization strategy that leverages recent advances in 3D neural reconstruction and rendering to produce sensor tokens that are agnostic to the number of input cameras and their resolution, while explicitly accounting for their geometry around an AV. Experiments on a large-scale AV dataset and state-of-the-art neural simulator demonstrate that our approach yields significant savings over current image patch-based tokenization strategies, producing up to 72% fewer tokens, resulting in up to 50% faster policy inference while achieving the same open-loop motion planning accuracy and improved offroad rates in closed-loop driving simulations.
>
---
#### [new 063] Block-wise Adaptive Caching for Accelerating Diffusion Policy
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决扩散策略计算成本高的问题。提出BAC方法通过缓存中间动作特征实现加速。**

- **链接: [http://arxiv.org/pdf/2506.13456v1](http://arxiv.org/pdf/2506.13456v1)**

> **作者:** Kangye Ji; Yuan Meng; Hanyun Cui; Ye Li; Shengjia Hua; Lei Chen; Zhi Wang
>
> **摘要:** Diffusion Policy has demonstrated strong visuomotor modeling capabilities, but its high computational cost renders it impractical for real-time robotic control. Despite huge redundancy across repetitive denoising steps, existing diffusion acceleration techniques fail to generalize to Diffusion Policy due to fundamental architectural and data divergences. In this paper, we propose Block-wise Adaptive Caching(BAC), a method to accelerate Diffusion Policy by caching intermediate action features. BAC achieves lossless action generation acceleration by adaptively updating and reusing cached features at the block level, based on a key observation that feature similarities vary non-uniformly across timesteps and locks. To operationalize this insight, we first propose the Adaptive Caching Scheduler, designed to identify optimal update timesteps by maximizing the global feature similarities between cached and skipped features. However, applying this scheduler for each block leads to signiffcant error surges due to the inter-block propagation of caching errors, particularly within Feed-Forward Network (FFN) blocks. To mitigate this issue, we develop the Bubbling Union Algorithm, which truncates these errors by updating the upstream blocks with signiffcant caching errors before downstream FFNs. As a training-free plugin, BAC is readily integrable with existing transformer-based Diffusion Policy and vision-language-action models. Extensive experiments on multiple robotic benchmarks demonstrate that BAC achieves up to 3x inference speedup for free.
>
---
## 更新

#### [replaced 001] Hierarchical Language Models for Semantic Navigation and Manipulation in an Aerial-Ground Robotic System
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.05020v2](http://arxiv.org/pdf/2506.05020v2)**

> **作者:** Haokun Liu; Zhaoqi Ma; Yunong Li; Junichiro Sugihara; Yicheng Chen; Jinjie Li; Moju Zhao
>
> **摘要:** Heterogeneous multi-robot systems show great potential in complex tasks requiring hybrid cooperation. However, traditional approaches relying on static models often struggle with task diversity and dynamic environments. This highlights the need for generalizable intelligence that can bridge high-level reasoning with low-level execution across heterogeneous agents. To address this, we propose a hierarchical framework integrating a prompted Large Language Model (LLM) and a GridMask-enhanced fine-tuned Vision Language Model (VLM). The LLM decomposes tasks and constructs a global semantic map, while the VLM extracts task-specified semantic labels and 2D spatial information from aerial images to support local planning. Within this framework, the aerial robot follows an optimized global semantic path and continuously provides bird-view images, guiding the ground robot's local semantic navigation and manipulation, including target-absent scenarios where implicit alignment is maintained. Experiments on real-world cube or object arrangement tasks demonstrate the framework's adaptability and robustness in dynamic environments. To the best of our knowledge, this is the first demonstration of an aerial-ground heterogeneous system integrating VLM-based perception with LLM-driven task reasoning and motion planning.
>
---
#### [replaced 002] Canonical Representation and Force-Based Pretraining of 3D Tactile for Dexterous Visuo-Tactile Policy Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.17549v3](http://arxiv.org/pdf/2409.17549v3)**

> **作者:** Tianhao Wu; Jinzhou Li; Jiyao Zhang; Mingdong Wu; Hao Dong
>
> **备注:** Accepted to ICRA 2025
>
> **摘要:** Tactile sensing plays a vital role in enabling robots to perform fine-grained, contact-rich tasks. However, the high dimensionality of tactile data, due to the large coverage on dexterous hands, poses significant challenges for effective tactile feature learning, especially for 3D tactile data, as there are no large standardized datasets and no strong pretrained backbones. To address these challenges, we propose a novel canonical representation that reduces the difficulty of 3D tactile feature learning and further introduces a force-based self-supervised pretraining task to capture both local and net force features, which are crucial for dexterous manipulation. Our method achieves an average success rate of 78% across four fine-grained, contact-rich dexterous manipulation tasks in real-world experiments, demonstrating effectiveness and robustness compared to other methods. Further analysis shows that our method fully utilizes both spatial and force information from 3D tactile data to accomplish the tasks. The codes and videos can be viewed at https://3dtacdex.github.io.
>
---
#### [replaced 003] Imagine, Verify, Execute: Memory-Guided Agentic Exploration with Vision-Language Models
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.07815v2](http://arxiv.org/pdf/2505.07815v2)**

> **作者:** Seungjae Lee; Daniel Ekpo; Haowen Liu; Furong Huang; Abhinav Shrivastava; Jia-Bin Huang
>
> **备注:** Project webpage: https://ive-robot.github.io/
>
> **摘要:** Exploration is essential for general-purpose robotic learning, especially in open-ended environments where dense rewards, explicit goals, or task-specific supervision are scarce. Vision-language models (VLMs), with their semantic reasoning over objects, spatial relations, and potential outcomes, present a compelling foundation for generating high-level exploratory behaviors. However, their outputs are often ungrounded, making it difficult to determine whether imagined transitions are physically feasible or informative. To bridge the gap between imagination and execution, we present IVE (Imagine, Verify, Execute), an agentic exploration framework inspired by human curiosity. Human exploration is often driven by the desire to discover novel scene configurations and to deepen understanding of the environment. Similarly, IVE leverages VLMs to abstract RGB-D observations into semantic scene graphs, imagine novel scenes, predict their physical plausibility, and generate executable skill sequences through action tools. We evaluate IVE in both simulated and real-world tabletop environments. The results show that IVE enables more diverse and meaningful exploration than RL baselines, as evidenced by a 4.1 to 7.8x increase in the entropy of visited states. Moreover, the collected experience supports downstream learning, producing policies that closely match or exceed the performance of those trained on human-collected demonstrations.
>
---
#### [replaced 004] Towards Infant Sleep-Optimized Driving: Synergizing Wearable and Vehicle Sensing in Intelligent Cruise Control
- **分类: cs.LG; cs.ET; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.06459v2](http://arxiv.org/pdf/2506.06459v2)**

> **作者:** Ruitao Chen; Mozhang Guo; Jinge Li
>
> **摘要:** Automated driving (AD) has substantially improved vehicle safety and driving comfort, but their impact on passenger well-being, particularly infant sleep, is not sufficiently studied. Sudden acceleration, abrupt braking, and sharp maneuvers can disrupt infant sleep, compromising both passenger comfort and parental convenience. To solve this problem, this paper explores the integration of reinforcement learning (RL) within AD to personalize driving behavior and optimally balance occupant comfort and travel efficiency. In particular, we propose an intelligent cruise control framework that adapts to varying driving conditions to enhance infant sleep quality by effectively synergizing wearable sensing and vehicle data. Long short-term memory (LSTM) and transformer-based neural networks are integrated with RL to model the relationship between driving behavior and infant sleep quality under diverse traffic and road conditions. Based on the sleep quality indicators from the wearable sensors, driving action data from vehicle controllers, and map data from map applications, the model dynamically computes the optimal driving aggressiveness level, which is subsequently translated into specific AD control strategies, e.g., the magnitude and frequency of acceleration, lane change, and overtaking. Simulation results demonstrate that the proposed solution significantly improves infant sleep quality compared to baseline methods, while preserving desirable travel efficiency.
>
---
#### [replaced 005] Structureless VIO
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.12337v2](http://arxiv.org/pdf/2505.12337v2)**

> **作者:** Junlin Song; Miguel Olivares-Mendez
>
> **备注:** Accepted by the SLAM Workshop at RSS 2025
>
> **摘要:** Visual odometry (VO) is typically considered as a chicken-and-egg problem, as the localization and mapping modules are tightly-coupled. The estimation of a visual map relies on accurate localization information. Meanwhile, localization requires precise map points to provide motion constraints. This classical design principle is naturally inherited by visual-inertial odometry (VIO). Efficient localization solutions that do not require a map have not been fully investigated. To this end, we propose a novel structureless VIO, where the visual map is removed from the odometry framework. Experimental results demonstrated that, compared to the structure-based VIO baseline, our structureless VIO not only substantially improves computational efficiency but also has advantages in accuracy.
>
---
#### [replaced 006] Towards Full-Scenario Safety Evaluation of Automated Vehicles: A Volume-Based Method
- **分类: cs.RO; cs.ET**

- **链接: [http://arxiv.org/pdf/2506.09182v2](http://arxiv.org/pdf/2506.09182v2)**

> **作者:** Hang Zhou; Chengyuan Ma; Shiyu Shen; Zhaohui Liang; Xiaopeng Li
>
> **备注:** NA
>
> **摘要:** With the rapid development of automated vehicles (AVs) in recent years, commercially available AVs are increasingly demonstrating high-level automation capabilities. However, most existing AV safety evaluation methods are primarily designed for simple maneuvers such as car-following and lane-changing. While suitable for basic tests, these methods are insufficient for assessing high-level automation functions deployed in more complex environments. First, these methods typically use crash rate as the evaluation metric, whose accuracy heavily depends on the quality and completeness of naturalistic driving environment data used to estimate scenario probabilities. Such data is often difficult and expensive to collect. Second, when applied to diverse scenarios, these methods suffer from the curse of dimensionality, making large-scale evaluation computationally intractable. To address these challenges, this paper proposes a novel framework for full-scenario AV safety evaluation. A unified model is first introduced to standardize the representation of diverse driving scenarios. This modeling approach constrains the dimension of most scenarios to a regular highway setting with three lanes and six surrounding background vehicles, significantly reducing dimensionality. To further avoid the limitations of probability-based method, we propose a volume-based evaluation method that quantifies the proportion of risky scenarios within the entire scenario space. For car-following scenarios, we prove that the set of safe scenarios is convex under specific settings, enabling exact volume computation. Experimental results validate the effectiveness of the proposed volume-based method using both AV behavior models from existing literature and six production AV models calibrated from field-test trajectory data in the Ultra-AV dataset. Code and data will be made publicly available upon acceptance of this paper.
>
---
#### [replaced 007] EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.10600v2](http://arxiv.org/pdf/2506.10600v2)**

> **作者:** Xinjie Wang; Liu Liu; Yu Cao; Ruiqi Wu; Wenkang Qin; Dehui Wang; Wei Sui; Zhizhong Su
>
> **摘要:** Constructing a physically realistic and accurately scaled simulated 3D world is crucial for the training and evaluation of embodied intelligence tasks. The diversity, realism, low cost accessibility and affordability of 3D data assets are critical for achieving generalization and scalability in embodied AI. However, most current embodied intelligence tasks still rely heavily on traditional 3D computer graphics assets manually created and annotated, which suffer from high production costs and limited realism. These limitations significantly hinder the scalability of data driven approaches. We present EmbodiedGen, a foundational platform for interactive 3D world generation. It enables the scalable generation of high-quality, controllable and photorealistic 3D assets with accurate physical properties and real-world scale in the Unified Robotics Description Format (URDF) at low cost. These assets can be directly imported into various physics simulation engines for fine-grained physical control, supporting downstream tasks in training and evaluation. EmbodiedGen is an easy-to-use, full-featured toolkit composed of six key modules: Image-to-3D, Text-to-3D, Texture Generation, Articulated Object Generation, Scene Generation and Layout Generation. EmbodiedGen generates diverse and interactive 3D worlds composed of generative 3D assets, leveraging generative AI to address the challenges of generalization and evaluation to the needs of embodied intelligence related research. Code is available at https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html.
>
---
#### [replaced 008] Cybersecurity and Embodiment Integrity for Modern Robots: A Conceptual Framework
- **分类: cs.CR; cs.RO**

- **链接: [http://arxiv.org/pdf/2401.07783v2](http://arxiv.org/pdf/2401.07783v2)**

> **作者:** Alberto Giaretta; Amy Loutfi
>
> **备注:** 18 pages, 2 figures, 4 tables
>
> **摘要:** Thanks to new technologies and communication paradigms, such as the Internet of Things (IoT) and the Robotic Operating System (ROS), modern robots can be built by combining heterogeneous standard devices in a single embodiment. Although this approach brings high degrees of modularity, it also yields uncertainty, with regard to providing cybersecurity assurances and guarantees on the integrity of the embodiment. In this paper, first we illustrate how cyberattacks on different devices can have radically different consequences on the robot's ability to complete its tasks and preserve its embodiment. We also claim that modern robots should have self-awareness for what concerns such aspects, and formulate in two propositions the different characteristics that robots should integrate for doing so. Then, we show how these propositions relate to two established cybersecurity frameworks, the NIST Cybersecurity Framework and the MITRE ATT&CK, and we argue that achieving these propositions requires that robots possess at least three properties for mapping devices and tasks. Last, we reflect on how these three properties could be achieved in a larger conceptual framework.
>
---
#### [replaced 009] XPG-RL: Reinforcement Learning with Explainable Priority Guidance for Efficiency-Boosted Mechanical Search
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.20969v2](http://arxiv.org/pdf/2504.20969v2)**

> **作者:** Yiting Zhang; Shichen Li; Elena Shrestha
>
> **备注:** Accepted to RSS 2025 Workshop on Learned Robot Representations (RoboReps)
>
> **摘要:** Mechanical search (MS) in cluttered environments remains a significant challenge for autonomous manipulators, requiring long-horizon planning and robust state estimation under occlusions and partial observability. In this work, we introduce XPG-RL, a reinforcement learning framework that enables agents to efficiently perform MS tasks through explainable, priority-guided decision-making based on raw sensory inputs. XPG-RL integrates a task-driven action prioritization mechanism with a learned context-aware switching strategy that dynamically selects from a discrete set of action primitives such as target grasping, occlusion removal, and viewpoint adjustment. Within this strategy, a policy is optimized to output adaptive threshold values that govern the discrete selection among action primitives. The perception module fuses RGB-D inputs with semantic and geometric features to produce a structured scene representation for downstream decision-making. Extensive experiments in both simulation and real-world settings demonstrate that XPG-RL consistently outperforms baseline methods in task success rates and motion efficiency, achieving up to 4.5$\times$ higher efficiency in long-horizon tasks. These results underscore the benefits of integrating domain knowledge with learnable decision-making policies for robust and efficient robotic manipulation. The project page for XPG-RL is https://yitingzhang1997.github.io/xpgrl/.
>
---
#### [replaced 010] Opt2Skill: Imitating Dynamically-feasible Whole-Body Trajectories for Versatile Humanoid Loco-Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.20514v4](http://arxiv.org/pdf/2409.20514v4)**

> **作者:** Fukang Liu; Zhaoyuan Gu; Yilin Cai; Ziyi Zhou; Hyunyoung Jung; Jaehwi Jang; Shijie Zhao; Sehoon Ha; Yue Chen; Danfei Xu; Ye Zhao
>
> **摘要:** Humanoid robots are designed to perform diverse loco-manipulation tasks. However, they face challenges due to their high-dimensional and unstable dynamics, as well as the complex contact-rich nature of the tasks. Model-based optimal control methods offer flexibility to define precise motion but are limited by high computational complexity and accurate contact sensing. On the other hand, reinforcement learning (RL) handles high-dimensional spaces with strong robustness but suffers from inefficient learning, unnatural motion, and sim-to-real gaps. To address these challenges, we introduce Opt2Skill, an end-to-end pipeline that combines model-based trajectory optimization with RL to achieve robust whole-body loco-manipulation. Opt2Skill generates dynamic feasible and contact-consistent reference motions for the Digit humanoid robot using differential dynamic programming (DDP) and trains RL policies to track these optimal trajectories. Our results demonstrate that Opt2Skill outperforms baselines that rely on human demonstrations and inverse kinematics-based references, both in motion tracking and task success rates. Furthermore, we show that incorporating trajectories with torque information improves contact force tracking in contact-involved tasks, such as wiping a table. We have successfully transferred our approach to real-world applications.
>
---
#### [replaced 011] Collaboration Between the City and Machine Learning Community is Crucial to Efficient Autonomous Vehicles Routing
- **分类: cs.MA; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.13188v2](http://arxiv.org/pdf/2502.13188v2)**

> **作者:** Anastasia Psarou; Ahmet Onur Akman; Łukasz Gorczyca; Michał Hoffmann; Grzegorz Jamróz; Rafał Kucharski
>
> **摘要:** Autonomous vehicles (AVs), possibly using Multi-Agent Reinforcement Learning (MARL) for simultaneous route optimization, may destabilize traffic networks, with human drivers potentially experiencing longer travel times. We study this interaction by simulating human drivers and AVs. Our experiments with standard MARL algorithms reveal that, both in simplified and complex networks, policies often fail to converge to an optimal solution or require long training periods. This problem is amplified by the fact that we cannot rely entirely on simulated training, as there are no accurate models of human routing behavior. At the same time, real-world training in cities risks destabilizing urban traffic systems, increasing externalities, such as $CO_2$ emissions, and introducing non-stationarity as human drivers will adapt unpredictably to AV behaviors. In this position paper, we argue that city authorities must collaborate with the ML community to monitor and critically evaluate the routing algorithms proposed by car companies toward fair and system-efficient routing algorithms and regulatory standards.
>
---
#### [replaced 012] Physics-informed Neural Mapping and Motion Planning in Unknown Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.09883v2](http://arxiv.org/pdf/2410.09883v2)**

> **作者:** Yuchen Liu; Ruiqi Ni; Ahmed H. Qureshi
>
> **备注:** Published in: IEEE Transactions on Robotics ( Volume: 41)
>
> **摘要:** Mapping and motion planning are two essential elements of robot intelligence that are interdependent in generating environment maps and navigating around obstacles. The existing mapping methods create maps that require computationally expensive motion planning tools to find a path solution. In this paper, we propose a new mapping feature called arrival time fields, which is a solution to the Eikonal equation. The arrival time fields can directly guide the robot in navigating the given environments. Therefore, this paper introduces a new approach called Active Neural Time Fields (Active NTFields), which is a physics-informed neural framework that actively explores the unknown environment and maps its arrival time field on the fly for robot motion planning. Our method does not require any expert data for learning and uses neural networks to directly solve the Eikonal equation for arrival time field mapping and motion planning. We benchmark our approach against state-of-the-art mapping and motion planning methods and demonstrate its superior performance in both simulated and real-world environments with a differential drive robot and a 6 degrees-of-freedom (DOF) robot manipulator. The supplementary videos can be found at https://youtu.be/qTPL5a6pRKk, and the implementation code repository is available at https://github.com/Rtlyc/antfields-demo.
>
---
#### [replaced 013] Object State Estimation Through Robotic Active Interaction for Biological Autonomous Drilling
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.04043v2](http://arxiv.org/pdf/2503.04043v2)**

> **作者:** Xiaofeng Lin; Enduo Zhao; Saúl Alexis Heredia Pérez; Kanako Harada
>
> **备注:** The first and second authors contribute equally to this research. 6 pages, 5 figures, submitted to RA-L
>
> **摘要:** Estimating the state of biological specimens is challenging due to limited observation through microscopic vision. For instance, during mouse skull drilling, the appearance alters little when thinning bone tissue because of its semi-transparent property and the high-magnification microscopic vision. To obtain the object's state, we introduce an object state estimation method for biological specimens through active interaction based on the deflection. The method is integrated to enhance the autonomous drilling system developed in our previous work. The method and integrated system were evaluated through 12 autonomous eggshell drilling experiment trials. The results show that the system achieved a 91.7% successful ratio and 75% detachable ratio, showcasing its potential applicability in more complex surgical procedures such as mouse skull craniotomy. This research paves the way for further development of autonomous robotic systems capable of estimating the object's state through active interaction.
>
---
#### [replaced 014] SceneComplete: Open-World 3D Scene Completion in Cluttered Real World Environments for Robot Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.23643v3](http://arxiv.org/pdf/2410.23643v3)**

> **作者:** Aditya Agarwal; Gaurav Singh; Bipasha Sen; Tomás Lozano-Pérez; Leslie Pack Kaelbling
>
> **摘要:** Careful robot manipulation in every-day cluttered environments requires an accurate understanding of the 3D scene, in order to grasp and place objects stably and reliably and to avoid colliding with other objects. In general, we must construct such a 3D interpretation of a complex scene based on limited input, such as a single RGB-D image. We describe SceneComplete, a system for constructing a complete, segmented, 3D model of a scene from a single view. SceneComplete is a novel pipeline for composing general-purpose pretrained perception modules (vision-language, segmentation, image-inpainting, image-to-3D, visual-descriptors and pose-estimation) to obtain highly accurate results. We demonstrate its accuracy and effectiveness with respect to ground-truth models in a large benchmark dataset and show that its accurate whole-object reconstruction enables robust grasp proposal generation, including for a dexterous hand. We release the code on our website https://scenecomplete.github.io/.
>
---
#### [replaced 015] General agents need world models
- **分类: cs.AI; cs.LG; cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2506.01622v2](http://arxiv.org/pdf/2506.01622v2)**

> **作者:** Jonathan Richens; David Abel; Alexis Bellot; Tom Everitt
>
> **备注:** Accepted ICML 2025
>
> **摘要:** Are world models a necessary ingredient for flexible, goal-directed behaviour, or is model-free learning sufficient? We provide a formal answer to this question, showing that any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment. We show that this model can be extracted from the agent's policy, and that increasing the agents performance or the complexity of the goals it can achieve requires learning increasingly accurate world models. This has a number of consequences: from developing safe and general agents, to bounding agent capabilities in complex environments, and providing new algorithms for eliciting world models from agents.
>
---
#### [replaced 016] Tactile MNIST: Benchmarking Active Tactile Perception
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06361v2](http://arxiv.org/pdf/2506.06361v2)**

> **作者:** Tim Schneider; Guillaume Duret; Cristiana de Farias; Roberto Calandra; Liming Chen; Jan Peters
>
> **摘要:** Tactile perception has the potential to significantly enhance dexterous robotic manipulation by providing rich local information that can complement or substitute for other sensory modalities such as vision. However, because tactile sensing is inherently local, it is not well-suited for tasks that require broad spatial awareness or global scene understanding on its own. A human-inspired strategy to address this issue is to consider active perception techniques instead. That is, to actively guide sensors toward regions with more informative or significant features and integrate such information over time in order to understand a scene or complete a task. Both active perception and different methods for tactile sensing have received significant attention recently. Yet, despite advancements, both fields lack standardized benchmarks. To bridge this gap, we introduce the Tactile MNIST Benchmark Suite, an open-source, Gymnasium-compatible benchmark specifically designed for active tactile perception tasks, including localization, classification, and volume estimation. Our benchmark suite offers diverse simulation scenarios, from simple toy environments all the way to complex tactile perception tasks using vision-based tactile sensors. Furthermore, we also offer a comprehensive dataset comprising 13,500 synthetic 3D MNIST digit models and 153,600 real-world tactile samples collected from 600 3D printed digits. Using this dataset, we train a CycleGAN for realistic tactile simulation rendering. By providing standardized protocols and reproducible evaluation frameworks, our benchmark suite facilitates systematic progress in the fields of tactile sensing and active perception.
>
---
#### [replaced 017] Uncertainty-Aware Trajectory Prediction via Rule-Regularized Heteroscedastic Deep Classification
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.13111v2](http://arxiv.org/pdf/2504.13111v2)**

> **作者:** Kumar Manas; Christian Schlauch; Adrian Paschke; Christian Wirth; Nadja Klein
>
> **备注:** 17 Pages, 9 figures. Accepted to Robotics: Science and Systems(RSS), 2025
>
> **摘要:** Deep learning-based trajectory prediction models have demonstrated promising capabilities in capturing complex interactions. However, their out-of-distribution generalization remains a significant challenge, particularly due to unbalanced data and a lack of enough data and diversity to ensure robustness and calibration. To address this, we propose SHIFT (Spectral Heteroscedastic Informed Forecasting for Trajectories), a novel framework that uniquely combines well-calibrated uncertainty modeling with informative priors derived through automated rule extraction. SHIFT reformulates trajectory prediction as a classification task and employs heteroscedastic spectral-normalized Gaussian processes to effectively disentangle epistemic and aleatoric uncertainties. We learn informative priors from training labels, which are automatically generated from natural language driving rules, such as stop rules and drivability constraints, using a retrieval-augmented generation framework powered by a large language model. Extensive evaluations over the nuScenes dataset, including challenging low-data and cross-location scenarios, demonstrate that SHIFT outperforms state-of-the-art methods, achieving substantial gains in uncertainty calibration and displacement metrics. In particular, our model excels in complex scenarios, such as intersections, where uncertainty is inherently higher. Project page: https://kumarmanas.github.io/SHIFT/.
>
---
#### [replaced 018] Real-time Seafloor Segmentation and Mapping
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.10750v3](http://arxiv.org/pdf/2504.10750v3)**

> **作者:** Michele Grimaldi; Nouf Alkaabi; Francesco Ruscio; Sebastian Realpe Rua; Rafael Garcia; Nuno Gracias
>
> **摘要:** Posidonia oceanica meadows are a species of seagrass highly dependent on rocks for their survival and conservation. In recent years, there has been a concerning global decline in this species, emphasizing the critical need for efficient monitoring and assessment tools. While deep learning-based semantic segmentation and visual automated monitoring systems have shown promise in a variety of applications, their performance in underwater environments remains challenging due to complex water conditions and limited datasets. This paper introduces a framework that combines machine learning and computer vision techniques to enable an autonomous underwater vehicle (AUV) to inspect the boundaries of Posidonia oceanica meadows autonomously. The framework incorporates an image segmentation module using an existing Mask R-CNN model and a strategy for Posidonia oceanica meadow boundary tracking. Furthermore, a new class dedicated to rocks is introduced to enhance the existing model, aiming to contribute to a comprehensive monitoring approach and provide a deeper understanding of the intricate interactions between the meadow and its surrounding environment. The image segmentation model is validated using real underwater images, while the overall inspection framework is evaluated in a realistic simulation environment, replicating actual monitoring scenarios with real underwater images. The results demonstrate that the proposed framework enables the AUV to autonomously accomplish the main tasks of underwater inspection and segmentation of rocks. Consequently, this work holds significant potential for the conservation and protection of marine environments, providing valuable insights into the status of Posidonia oceanica meadows and supporting targeted preservation efforts
>
---
#### [replaced 019] M3Depth: Wavelet-Enhanced Depth Estimation on Mars via Mutual Boosting of Dual-Modal Data
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.14159v2](http://arxiv.org/pdf/2505.14159v2)**

> **作者:** Junjie Li; Jiawei Wang; Miyu Li; Yu Liu; Yumei Wang; Haitao Xu
>
> **摘要:** Depth estimation plays a great potential role in obstacle avoidance and navigation for further Mars exploration missions. Compared to traditional stereo matching, learning-based stereo depth estimation provides a data-driven approach to infer dense and precise depth maps from stereo image pairs. However, these methods always suffer performance degradation in environments with sparse textures and lacking geometric constraints, such as the unstructured terrain of Mars. To address these challenges, we propose M3Depth, a depth estimation model tailored for Mars rovers. Considering the sparse and smooth texture of Martian terrain, which is primarily composed of low-frequency features, our model incorporates a convolutional kernel based on wavelet transform that effectively captures low-frequency response and expands the receptive field. Additionally, we introduce a consistency loss that explicitly models the complementary relationship between depth map and surface normal map, utilizing the surface normal as a geometric constraint to enhance the accuracy of depth estimation. Besides, a pixel-wise refinement module with mutual boosting mechanism is designed to iteratively refine both depth and surface normal predictions. Experimental results on synthetic Mars datasets with depth annotations show that M3Depth achieves a 16% improvement in depth estimation accuracy compared to other state-of-the-art methods in depth estimation. Furthermore, the model demonstrates strong applicability in real-world Martian scenarios, offering a promising solution for future Mars exploration missions.
>
---
#### [replaced 020] JAEGER: Dual-Level Humanoid Whole-Body Controller
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.06584v2](http://arxiv.org/pdf/2505.06584v2)**

> **作者:** Ziluo Ding; Haobin Jiang; Yuxuan Wang; Zhenguo Sun; Yu Zhang; Xiaojie Niu; Ming Yang; Weishuai Zeng; Xinrun Xu; Zongqing Lu
>
> **备注:** 15 pages, 2 figures
>
> **摘要:** This paper presents JAEGER, a dual-level whole-body controller for humanoid robots that addresses the challenges of training a more robust and versatile policy. Unlike traditional single-controller approaches, JAEGER separates the control of the upper and lower bodies into two independent controllers, so that they can better focus on their distinct tasks. This separation alleviates the dimensionality curse and improves fault tolerance. JAEGER supports both root velocity tracking (coarse-grained control) and local joint angle tracking (fine-grained control), enabling versatile and stable movements. To train the controller, we utilize a human motion dataset (AMASS), retargeting human poses to humanoid poses through an efficient retargeting network, and employ a curriculum learning approach. This method performs supervised learning for initialization, followed by reinforcement learning for further exploration. We conduct our experiments on two humanoid platforms and demonstrate the superiority of our approach against state-of-the-art methods in both simulation and real environments.
>
---
#### [replaced 021] Do We Still Need to Work on Odometry for Autonomous Driving?
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.04438v2](http://arxiv.org/pdf/2505.04438v2)**

> **作者:** Cedric Le Gentil; Daniil Lisus; Timothy D. Barfoot
>
> **备注:** Presented at the 2025 IEEE ICRA Workshop on Field Robotics
>
> **摘要:** Over the past decades, a tremendous amount of work has addressed the topic of ego-motion estimation of moving platforms based on various proprioceptive and exteroceptive sensors. At the cost of ever-increasing computational load and sensor complexity, odometry algorithms have reached impressive levels of accuracy with minimal drift in various conditions. In this paper, we question the need for more research on odometry for autonomous driving by assessing the accuracy of one of the simplest algorithms: the direct integration of wheel encoder data and yaw rate measurements from a gyroscope. We denote this algorithm as Odometer-Gyroscope (OG) odometry. This work shows that OG odometry can outperform current state-of-the-art radar-inertial SE(2) odometry for a fraction of the computational cost in most scenarios. For example, the OG odometry is on top of the Boreas leaderboard with a relative translation error of 0.20%, while the second-best method displays an error of 0.26%. Lidar-inertial approaches can provide more accurate estimates, but the computational load is three orders of magnitude higher than the OG odometry. To further the analysis, we have pushed the limits of the OG odometry by purposely violating its fundamental no-slip assumption using data collected during a heavy snowstorm with different driving behaviours. Our conclusion shows that a significant amount of slippage is required to result in non-satisfactory pose estimates from the OG odometry.
>
---
#### [replaced 022] Learning-based 3D Reconstruction in Autonomous Driving: A Comprehensive Survey
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.14537v3](http://arxiv.org/pdf/2503.14537v3)**

> **作者:** Liewen Liao; Weihao Yan; Ming Yang; Songan Zhang
>
> **摘要:** Learning-based 3D reconstruction has emerged as a transformative technique in autonomous driving, enabling precise modeling of both dynamic and static environments through advanced neural representations. Despite data augmentation, 3D reconstruction inspires pioneering solution for vital tasks in the field of autonomous driving, such as scene understanding and closed-loop simulation. We investigates the details of 3D reconstruction and conducts a multi-perspective, in-depth analysis of recent advancements. Specifically, we first provide a systematic introduction of preliminaries, including data modalities, benchmarks and technical preliminaries of learning-based 3D reconstruction, facilitating instant identification of suitable methods according to sensor suites. Then, we systematically review learning-based 3D reconstruction methods in autonomous driving, categorizing approaches by subtasks and conducting multi-dimensional analysis and summary to establish a comprehensive technical reference. The development trends and existing challenges are summarized in the context of learning-based 3D reconstruction in autonomous driving. We hope that our review will inspire future researches.
>
---
#### [replaced 023] Robust Flower Cluster Matching Using The Unscented Transform
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.20631v2](http://arxiv.org/pdf/2503.20631v2)**

> **作者:** Andy Chu; Rashik Shrestha; Yu Gu; Jason N. Gross
>
> **备注:** *CASE2025 Accepted*
>
> **摘要:** Monitoring flowers over time is essential for precision robotic pollination in agriculture. To accomplish this, a continuous spatial-temporal observation of plant growth can be done using stationary RGB-D cameras. However, image registration becomes a serious challenge due to changes in the visual appearance of the plant caused by the pollination process and occlusions from growth and camera angles. Plants flower in a manner that produces distinct clusters on branches. This paper presents a method for matching flower clusters using descriptors generated from RGB-D data and considers allowing for spatial uncertainty within the cluster. The proposed approach leverages the Unscented Transform to efficiently estimate plant descriptor uncertainty tolerances, enabling a robust image-registration process despite temporal changes. The Unscented Transform is used to handle the nonlinear transformations by propagating the uncertainty of flower positions to determine the variations in the descriptor domain. A Monte Carlo simulation is used to validate the Unscented Transform results, confirming our method's effectiveness for flower cluster matching. Therefore, it can facilitate improved robotics pollination in dynamic environments.
>
---
#### [replaced 024] Diffusion Graph Neural Networks for Robustness in Olfaction Sensors and Datasets
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00455v2](http://arxiv.org/pdf/2506.00455v2)**

> **作者:** Kordel K. France; Ovidiu Daescu
>
> **摘要:** Robotic odour source localization (OSL) is a critical capability for autonomous systems operating in complex environments. However, current OSL methods often suffer from ambiguities, particularly when robots misattribute odours to incorrect objects due to limitations in olfactory datasets and sensor resolutions. To address this challenge, we introduce a novel machine learning method using diffusion-based molecular generation to enhance odour localization accuracy that can be used by itself or with automated olfactory dataset construction pipelines with vision-language models (VLMs) This generative process of our diffusion model expands the chemical space beyond the limitations of both current olfactory datasets and the training data of VLMs, enabling the identification of potential odourant molecules not previously documented. The generated molecules can then be more accurately validated using advanced olfactory sensors which emulate human olfactory recognition through electronic sensor arrays. By integrating visual analysis, language processing, and molecular generation, our framework enhances the ability of olfaction-vision models on robots to accurately associate odours with their correct sources, thereby improving navigation and decision-making through better sensor selection for a target compound. Our methodology represents a foundational advancement in the field of artificial olfaction, offering a scalable solution to the challenges posed by limited olfactory data and sensor ambiguities.
>
---
#### [replaced 025] Zero-Shot Temporal Interaction Localization for Egocentric Videos
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.03662v3](http://arxiv.org/pdf/2506.03662v3)**

> **作者:** Erhang Zhang; Junyi Ma; Yin-Dong Zheng; Yixuan Zhou; Hesheng Wang
>
> **摘要:** Locating human-object interaction (HOI) actions within video serves as the foundation for multiple downstream tasks, such as human behavior analysis and human-robot skill transfer. Current temporal action localization methods typically rely on annotated action and object categories of interactions for optimization, which leads to domain bias and low deployment efficiency. Although some recent works have achieved zero-shot temporal action localization (ZS-TAL) with large vision-language models (VLMs), their coarse-grained estimations and open-loop pipelines hinder further performance improvements for temporal interaction localization (TIL). To address these issues, we propose a novel zero-shot TIL approach dubbed EgoLoc to locate the timings of grasp actions for human-object interaction in egocentric videos. EgoLoc introduces a self-adaptive sampling strategy to generate reasonable visual prompts for VLM reasoning. By absorbing both 2D and 3D observations, it directly samples high-quality initial guesses around the possible contact/separation timestamps of HOI according to 3D hand velocities, leading to high inference accuracy and efficiency. In addition, EgoLoc generates closed-loop feedback from visual and dynamic cues to further refine the localization results. Comprehensive experiments on the publicly available dataset and our newly proposed benchmark demonstrate that EgoLoc achieves better temporal interaction localization for egocentric videos compared to state-of-the-art baselines. We will release our code and relevant data as open-source at https://github.com/IRMVLab/EgoLoc.
>
---
#### [replaced 026] BiFold: Bimanual Cloth Folding with Language Guidance
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.16458v2](http://arxiv.org/pdf/2501.16458v2)**

> **作者:** Oriol Barbany; Adrià Colomé; Carme Torras
>
> **备注:** Accepted at ICRA 2025. Project page at https://barbany.github.io/bifold/
>
> **摘要:** Cloth folding is a complex task due to the inevitable self-occlusions of clothes, their complicated dynamics, and the disparate materials, geometries, and textures that garments can have. In this work, we learn folding actions conditioned on text commands. Translating high-level, abstract instructions into precise robotic actions requires sophisticated language understanding and manipulation capabilities. To do that, we leverage a pre-trained vision-language model and repurpose it to predict manipulation actions. Our model, BiFold, can take context into account and achieves state-of-the-art performance on an existing language-conditioned folding benchmark. To address the lack of annotated bimanual folding data, we introduce a novel dataset with automatically parsed actions and language-aligned instructions, enabling better learning of text-conditioned manipulation. BiFold attains the best performance on our dataset and demonstrates strong generalization to new instructions, garments, and environments.
>
---
#### [replaced 027] Active Perception for Tactile Sensing: A Task-Agnostic Attention-Based Approach
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.06182v2](http://arxiv.org/pdf/2505.06182v2)**

> **作者:** Tim Schneider; Cristiana de Farias; Roberto Calandra; Liming Chen; Jan Peters
>
> **备注:** 16 pages; 13 figures Under Review
>
> **摘要:** Humans make extensive use of haptic exploration to map and identify the properties of the objects that we touch. In robotics, active tactile perception has emerged as an important research domain that complements vision for tasks such as object classification, shape reconstruction, and manipulation. This work introduces TAP (Task-agnostic Active Perception) -- a novel framework that leverages reinforcement learning (RL) and transformer-based architectures to address the challenges posed by partially observable environments. TAP integrates Soft Actor-Critic (SAC) and CrossQ algorithms within a unified optimization objective, jointly training a perception module and decision-making policy. By design, TAP is completely task-agnostic and can, in principle, generalize to any active perception problem. We evaluate TAP across diverse tasks, including toy examples and realistic applications involving haptic exploration of 3D models from the Tactile MNIST benchmark. Experiments demonstrate the efficacy of TAP, achieving high accuracies on the Tactile MNIST haptic digit recognition task and a tactile pose estimation task. These findings underscore the potential of TAP as a versatile and generalizable framework for advancing active tactile perception in robotics.
>
---
#### [replaced 028] Deep Reinforcement Learning for Bipedal Locomotion: A Brief Survey
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.17070v4](http://arxiv.org/pdf/2404.17070v4)**

> **作者:** Lingfan Bao; Joseph Humphreys; Tianhu Peng; Chengxu Zhou
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Bipedal robots are gaining global recognition due to their potential applications and advancements in artificial intelligence, particularly through Deep Reinforcement Learning (DRL). While DRL has significantly advanced bipedal locomotion, the development of a unified framework capable of handling a wide range of tasks remains an ongoing challenge. This survey systematically categorises, compares, and analyses existing DRL frameworks for bipedal locomotion, organising them into end-to-end and hierarchical control schemes. End-to-end frameworks are evaluated based on their learning approaches, while hierarchical frameworks are examined in terms of layered structures that integrate learning-based or traditional model-based methods. We provide a detailed evaluation of the composition, strengths, limitations, and capabilities of each framework. Additionally, this survey identifies key research gaps and proposes future directions aimed at creating a more integrated and efficient framework for bipedal locomotion, with wide-ranging applications in real-world environments.
>
---
#### [replaced 029] ActiveSplat: High-Fidelity Scene Reconstruction through Active Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.21955v2](http://arxiv.org/pdf/2410.21955v2)**

> **作者:** Yuetao Li; Zijia Kuang; Ting Li; Qun Hao; Zike Yan; Guyue Zhou; Shaohui Zhang
>
> **备注:** Accepted to IEEE RA-L. Code: https://github.com/Li-Yuetao/ActiveSplat, Project: https://li-yuetao.github.io/ActiveSplat/
>
> **摘要:** We propose ActiveSplat, an autonomous high-fidelity reconstruction system leveraging Gaussian splatting. Taking advantage of efficient and realistic rendering, the system establishes a unified framework for online mapping, viewpoint selection, and path planning. The key to ActiveSplat is a hybrid map representation that integrates both dense information about the environment and a sparse abstraction of the workspace. Therefore, the system leverages sparse topology for efficient viewpoint sampling and path planning, while exploiting view-dependent dense prediction for viewpoint selection, facilitating efficient decision-making with promising accuracy and completeness. A hierarchical planning strategy based on the topological map is adopted to mitigate repetitive trajectories and improve local granularity given limited time budgets, ensuring high-fidelity reconstruction with photorealistic view synthesis. Extensive experiments and ablation studies validate the efficacy of the proposed method in terms of reconstruction accuracy, data coverage, and exploration efficiency. The released code will be available on our project page: https://li-yuetao.github.io/ActiveSplat/.
>
---
#### [replaced 030] Bridging the Gap between Discrete Agent Strategies in Game Theory and Continuous Motion Planning in Dynamic Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.11334v2](http://arxiv.org/pdf/2403.11334v2)**

> **作者:** Hongrui Zheng; Zhijun Zhuang; Stephanie Wu; Shuo Yang; Rahul Mangharam
>
> **备注:** Submitted to RA-L
>
> **摘要:** Generating competitive strategies and performing continuous motion planning simultaneously in an adversarial setting is a challenging problem. In addition, understanding the intent of other agents is crucial to deploying autonomous systems in adversarial multi-agent environments. Existing approaches either discretize agent action by grouping similar control inputs, sacrificing performance in motion planning, or plan in uninterpretable latent spaces, producing hard-to-understand agent behaviors. This paper proposes an agent strategy representation via Policy Characteristic Space that maps the agent policies to a pre-specified low-dimensional space. Policy Characteristic Space enables the discretization of agent policy switchings while preserving continuity in control. Also, it provides intepretability of agent policies and clear intentions of policy switchings. Then, regret-based game-theoretic approaches can be applied in the Policy Characteristic Space to obtain high performance in adversarial environments. Our proposed method is assessed by conducting experiments in an autonomous racing scenario using scaled vehicles. Statistical evidence shows that our method significantly improves the win rate of ego agent and the method also generalizes well to unseen environments.
>
---
#### [replaced 031] Pursuit-Evasion for Car-like Robots with Sensor Constraints
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2405.05372v2](http://arxiv.org/pdf/2405.05372v2)**

> **作者:** Burak M. Gonultas; Volkan Isler
>
> **备注:** Accepted for publication in the Proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) as oral presentation
>
> **摘要:** We study a pursuit-evasion game between two players with car-like dynamics and sensing limitations by formalizing it as a partially observable stochastic zero-sum game. The partial observability caused by the sensing constraints is particularly challenging. As an example, in a situation where the agents have no visibility of each other, they would need to extract information from their sensor coverage history to reason about potential locations of their opponents. However, keeping historical information greatly increases the size of the state space. To mitigate the challenges encountered with such partially observable problems, we develop a new learning-based method that encodes historical information to a belief state and uses it to generate agent actions. Through experiments we show that the learned strategies improve over existing multi-agent RL baselines by up to 16 % in terms of capture rate for the pursuer. Additionally, we present experimental results showing that learned belief states are strong state estimators for extending existing game theory solvers and demonstrate our method's competitiveness for problems where existing fully observable game theory solvers are computationally feasible. Finally, we deploy the learned policies on physical robots for a game between the F1TENTH and JetRacer platforms moving as fast as $\textbf{2 m/s}$ in indoor environments, showing that they can be executed on real-robots.
>
---
#### [replaced 032] AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.15659v2](http://arxiv.org/pdf/2501.15659v2)**

> **作者:** Yuheng Qiu; Can Xu; Yutian Chen; Shibo Zhao; Junyi Geng; Sebastian Scherer
>
> **摘要:** Inertial odometry (IO) using only Inertial Measurement Units (IMUs) offers a lightweight and cost-effective solution for Unmanned Aerial Vehicle (UAV) applications, yet existing learning-based IO models often fail to generalize to UAVs due to the highly dynamic and non-linear-flight patterns that differ from pedestrian motion. In this work, we identify that the conventional practice of transforming raw IMU data to global coordinates undermines the observability of critical kinematic information in UAVs. By preserving the body-frame representation, our method achieves substantial performance improvements, with a 66.7% average increase in accuracy across three datasets. Furthermore, explicitly encoding attitude information into the motion network results in an additional 23.8% improvement over prior results. Combined with a data-driven IMU correction model (AirIMU) and an uncertainty-aware Extended Kalman Filter (EKF), our approach ensures robust state estimation under aggressive UAV maneuvers without relying on external sensors or control inputs. Notably, our method also demonstrates strong generalizability to unseen data not included in the training set, underscoring its potential for real-world UAV applications.
>
---
#### [replaced 033] Efficient Estimation of Relaxed Model Parameters for Robust UAV Trajectory Optimization
- **分类: math.OC; cs.RO; cs.SY; eess.SY; 49N10 (Primary) 93C40 (Secondary)**

- **链接: [http://arxiv.org/pdf/2411.10941v4](http://arxiv.org/pdf/2411.10941v4)**

> **作者:** Derek Fan; David A. Copp
>
> **备注:** 8 pages, 5 figures. Published in IEEE Sustech 2025, see https://ieeexplore.ieee.org/document/11025659
>
> **摘要:** Online trajectory optimization and optimal control methods are crucial for enabling sustainable unmanned aerial vehicle (UAV) services, such as agriculture, environmental monitoring, and transportation, where available actuation and energy are limited. However, optimal controllers are highly sensitive to model mismatch, which can occur due to loaded equipment, packages to be delivered, or pre-existing variability in fundamental structural and thrust-related parameters. To circumvent this problem, optimal controllers can be paired with parameter estimators to improve their trajectory planning performance and perform adaptive control. However, UAV platforms are limited in terms of onboard processing power, oftentimes making nonlinear parameter estimation too computationally expensive to consider. To address these issues, we propose a relaxed, affine-in-parameters multirotor model along with an efficient optimal parameter estimator. We convexify the nominal Moving Horizon Parameter Estimation (MHPE) problem into a linear-quadratic form (LQ-MHPE) via an affine-in-parameter relaxation on the nonlinear dynamics, resulting in fast quadratic programs (QPs) that facilitate adaptive Model Predictve Control (MPC) in real time. We compare this approach to the equivalent nonlinear estimator in Monte Carlo simulations, demonstrating a decrease in average solve time and trajectory optimality cost by 98.2% and 23.9-56.2%, respectively.
>
---
#### [replaced 034] X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.07096v3](http://arxiv.org/pdf/2505.07096v3)**

> **作者:** Prithwish Dan; Kushal Kedia; Angela Chao; Edward Weiyi Duan; Maximus Adrian Pace; Wei-Chiu Ma; Sanjiban Choudhury
>
> **摘要:** Human videos offer a scalable way to train robot manipulation policies, but lack the action labels needed by standard imitation learning algorithms. Existing cross-embodiment approaches try to map human motion to robot actions, but often fail when the embodiments differ significantly. We propose X-Sim, a real-to-sim-to-real framework that uses object motion as a dense and transferable signal for learning robot policies. X-Sim starts by reconstructing a photorealistic simulation from an RGBD human video and tracking object trajectories to define object-centric rewards. These rewards are used to train a reinforcement learning (RL) policy in simulation. The learned policy is then distilled into an image-conditioned diffusion policy using synthetic rollouts rendered with varied viewpoints and lighting. To transfer to the real world, X-Sim introduces an online domain adaptation technique that aligns real and simulated observations during deployment. Importantly, X-Sim does not require any robot teleoperation data. We evaluate it across 5 manipulation tasks in 2 environments and show that it: (1) improves task progress by 30% on average over hand-tracking and sim-to-real baselines, (2) matches behavior cloning with 10x less data collection time, and (3) generalizes to new camera viewpoints and test-time changes. Code and videos are available at https://portal-cornell.github.io/X-Sim/.
>
---
