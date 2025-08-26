# 机器人 cs.RO

- **最新发布 51 篇**

- **更新 23 篇**

## 最新发布

#### [new 001] DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration
- **分类: cs.RO**

- **简介: 论文提出DualReg方法解决点云或图像刚性配准问题，通过双空间过滤与强化策略融合特征匹配与局部几何信息，提升精度与速度，实现在噪声和部分重叠数据下的鲁棒实时配准。**

- **链接: [http://arxiv.org/pdf/2508.17034v1](http://arxiv.org/pdf/2508.17034v1)**

> **作者:** Jiayi Li; Yuxin Yao; Qiuhang Lu; Juyong Zhang
>
> **摘要:** Rigid registration, aiming to estimate a rigid transformation to align source and target data, play a crucial role in applications such as SLAM and 3D reconstruction. However, noisy, partially overlapping data and the need for real-time processing pose major challenges for rigid registration. Considering that feature-based matching can handle large transformation differences but suffers from limited accuracy, while local geometry-based matching can achieve fine-grained local alignment but relies heavily on a good initial transformation, we propose a novel dual-space paradigm to fully leverage the strengths of both approaches. First, we introduce an efficient filtering mechanism that incorporates a computationally lightweight single-point RANSAC algorithm followed by a refinement module to eliminate unreliable feature-based correspondences. Subsequently, we treat filtered correspondences as anchor points, extract geometric proxies, and formulates an effective objective function with a tailored solver to estimate the transformation. Experiments verify our method's effectiveness, as shown by achieving up to a 32x CPU-time speedup over MAC on KITTI with comparable accuracy.
>
---
#### [new 002] LodeStar: Long-horizon Dexterity via Synthetic Data Augmentation from Human Demonstrations
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出LodeStar框架，解决长程灵巧操作任务中数据稀缺与技能序列难题。通过分解人类演示生成合成数据，结合强化学习训练技能，并用SRT策略链式组合技能，显著提升真实场景下的任务性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.17547v1](http://arxiv.org/pdf/2508.17547v1)**

> **作者:** Weikang Wan; Jiawei Fu; Xiaodi Yuan; Yifeng Zhu; Hao Su
>
> **备注:** CoRL 2025
>
> **摘要:** Developing robotic systems capable of robustly executing long-horizon manipulation tasks with human-level dexterity is challenging, as such tasks require both physical dexterity and seamless sequencing of manipulation skills while robustly handling environment variations. While imitation learning offers a promising approach, acquiring comprehensive datasets is resource-intensive. In this work, we propose a learning framework and system LodeStar that automatically decomposes task demonstrations into semantically meaningful skills using off-the-shelf foundation models, and generates diverse synthetic demonstration datasets from a few human demos through reinforcement learning. These sim-augmented datasets enable robust skill training, with a Skill Routing Transformer (SRT) policy effectively chaining the learned skills together to execute complex long-horizon manipulation tasks. Experimental evaluations on three challenging real-world long-horizon dexterous manipulation tasks demonstrate that our approach significantly improves task performance and robustness compared to previous baselines. Videos are available at lodestar-robot.github.io.
>
---
#### [new 003] GWM: Towards Scalable Gaussian World Models for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出Gaussian World Model（GWM），用于机器人操作中的世界建模。针对现有图像模型缺乏几何信息的问题，GWM通过高斯基元传播预测未来状态，结合扩散Transformer与3D变分自编码器实现精细场景重建，支持模仿学习和基于模型的强化学习，显著提升策略性能。**

- **链接: [http://arxiv.org/pdf/2508.17600v1](http://arxiv.org/pdf/2508.17600v1)**

> **作者:** Guanxing Lu; Baoxiong Jia; Puhao Li; Yixin Chen; Ziwei Wang; Yansong Tang; Siyuan Huang
>
> **备注:** Published at ICCV 2025. Project page: https://gaussian-world-model.github.io/
>
> **摘要:** Training robot policies within a learned world model is trending due to the inefficiency of real-world interactions. The established image-based world models and policies have shown prior success, but lack robust geometric information that requires consistent spatial and physical understanding of the three-dimensional world, even pre-trained on internet-scale video sources. To this end, we propose a novel branch of world model named Gaussian World Model (GWM) for robotic manipulation, which reconstructs the future state by inferring the propagation of Gaussian primitives under the effect of robot actions. At its core is a latent Diffusion Transformer (DiT) combined with a 3D variational autoencoder, enabling fine-grained scene-level future state reconstruction with Gaussian Splatting. GWM can not only enhance the visual representation for imitation learning agent by self-supervised future prediction training, but can serve as a neural simulator that supports model-based reinforcement learning. Both simulated and real-world experiments depict that GWM can precisely predict future scenes conditioned on diverse robot actions, and can be further utilized to train policies that outperform the state-of-the-art by impressive margins, showcasing the initial data scaling potential of 3D world model.
>
---
#### [new 004] Modeling and Control Framework for Autonomous Space Manipulator Handover Operations
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文研究自主空间机械臂的协作操作，针对在轨服务、装配与制造任务中机器人间物品交接问题，构建双臂动力学模型并比较多种控制策略，以提升交接过程的自主性与精度。**

- **链接: [http://arxiv.org/pdf/2508.18039v1](http://arxiv.org/pdf/2508.18039v1)**

> **作者:** Diego Quevedo; Sarah Hudson; Donghoon Kim
>
> **备注:** 14 pages, submitted to 2025 Astrodynamics Specialists Conference proceedings
>
> **摘要:** Autonomous space robotics is poised to play a vital role in future space missions, particularly for In-space Servicing, Assembly, and Manufacturing (ISAM). A key capability in such missions is the Robot-to-Robot (R2R) handover of mission-critical objects. This work presents a dynamic model of a dual-arm space manipulator system and compares various tracking control laws. The key contributions of this work are the development of a cooperative manipulator dynamic model and the comparative analysis of control laws to support autonomous R2R handovers in ISAM scenarios.
>
---
#### [new 005] DANCeRS: A Distributed Algorithm for Negotiating Consensus in Robot Swarms with Gaussian Belief Propagation
- **分类: cs.RO**

- **简介: 论文提出DANCeRS算法，利用高斯信念传播实现机器人集群在离散与连续决策空间中的统一共识。解决分布式环境下群体协作的路径规划与决策一致问题，通过因子图建模和点对点通信确保可扩展性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.18153v1](http://arxiv.org/pdf/2508.18153v1)**

> **作者:** Aalok Patwardhan; Andrew J. Davison
>
> **摘要:** Robot swarms require cohesive collective behaviour to address diverse challenges, including shape formation and decision-making. Existing approaches often treat consensus in discrete and continuous decision spaces as distinct problems. We present DANCeRS, a unified, distributed algorithm leveraging Gaussian Belief Propagation (GBP) to achieve consensus in both domains. By representing a swarm as a factor graph our method ensures scalability and robustness in dynamic environments, relying on purely peer-to-peer message passing. We demonstrate the effectiveness of our general framework through two applications where agents in a swarm must achieve consensus on global behaviour whilst relying on local communication. In the first, robots must perform path planning and collision avoidance to create shape formations. In the second, we show how the same framework can be used by a group of robots to form a consensus over a set of discrete decisions. Experimental results highlight our method's scalability and efficiency compared to recent approaches to these problems making it a promising solution for multi-robot systems requiring distributed consensus. We encourage the reader to see the supplementary video demo.
>
---
#### [new 006] LLM-based Human-like Traffic Simulation for Self-driving Tests
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶测试中的交通场景生成任务，旨在解决现有模拟器无法真实再现人类驾驶多样性的问题。作者提出HDSim框架，结合认知理论与大语言模型，构建分层驾驶模型和感知引导行为策略，显著提升安全缺陷检测率与事故可解释性。**

- **链接: [http://arxiv.org/pdf/2508.16962v1](http://arxiv.org/pdf/2508.16962v1)**

> **作者:** Wendi Li; Hao Wu; Han Gao; Bing Mao; Fengyuan Xu; Sheng Zhong
>
> **摘要:** Ensuring realistic traffic dynamics is a prerequisite for simulation platforms to evaluate the reliability of self-driving systems before deployment in the real world. Because most road users are human drivers, reproducing their diverse behaviors within simulators is vital. Existing solutions, however, typically rely on either handcrafted heuristics or narrow data-driven models, which capture only fragments of real driving behaviors and offer limited driving style diversity and interpretability. To address this gap, we introduce HDSim, an HD traffic generation framework that combines cognitive theory with large language model (LLM) assistance to produce scalable and realistic traffic scenarios within simulation platforms. The framework advances the state of the art in two ways: (i) it introduces a hierarchical driver model that represents diverse driving style traits, and (ii) it develops a Perception-Mediated Behavior Influence strategy, where LLMs guide perception to indirectly shape driver actions. Experiments reveal that embedding HDSim into simulation improves detection of safety-critical failures in self-driving systems by up to 68% and yields realism-consistent accident interpretability.
>
---
#### [new 007] A Rapid Iterative Trajectory Planning Method for Automated Parking through Differential Flatness
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种基于路径速度分解的快速迭代轨迹规划方法（RITP），用于自动驾驶泊车任务，解决快速性与避障精度冲突及换挡点控制可行性问题。通过微分平坦性和终端平滑约束提升轨迹可行性和连续性。**

- **链接: [http://arxiv.org/pdf/2508.17038v1](http://arxiv.org/pdf/2508.17038v1)**

> **作者:** Zhouheng Li; Lei Xie; Cheng Hu; Hongye Su
>
> **备注:** Published in the journal Robotics and Autonomous Systems
>
> **摘要:** As autonomous driving continues to advance, automated parking is becoming increasingly essential. However, significant challenges arise when implementing path velocity decomposition (PVD) trajectory planning for automated parking. The primary challenge is ensuring rapid and precise collision-free trajectory planning, which is often in conflict. The secondary challenge involves maintaining sufficient control feasibility of the planned trajectory, particularly at gear shifting points (GSP). This paper proposes a PVD-based rapid iterative trajectory planning (RITP) method to solve the above challenges. The proposed method effectively balances the necessity for time efficiency and precise collision avoidance through a novel collision avoidance framework. Moreover, it enhances the overall control feasibility of the planned trajectory by incorporating the vehicle kinematics model and including terminal smoothing constraints (TSC) at GSP during path planning. Specifically, the proposed method leverages differential flatness to ensure the planned path adheres to the vehicle kinematic model. Additionally, it utilizes TSC to maintain curvature continuity at GSP, thereby enhancing the control feasibility of the overall trajectory. The simulation results demonstrate superior time efficiency and tracking errors compared to model-integrated and other iteration-based trajectory planning methods. In the real-world experiment, the proposed method was implemented and validated on a ROS-based vehicle, demonstrating the applicability of the RITP method for real vehicles.
>
---
#### [new 008] Autonomous UAV Flight Navigation in Confined Spaces: A Reinforcement Learning Approach
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **简介: 论文研究无人机在无GPS环境下的避障自主导航任务，针对工业狭小空间巡检难题，比较PPO与SAC两种强化学习算法。结果表明，PPO因训练稳定更有效，验证了程序生成仿真环境的有效性。**

- **链接: [http://arxiv.org/pdf/2508.16807v1](http://arxiv.org/pdf/2508.16807v1)**

> **作者:** Marco S. Tayar; Lucas K. de Oliveira; Juliano D. Negri; Thiago H. Segreto; Ricardo V. Godoy; Marcelo Becker
>
> **摘要:** Inspecting confined industrial infrastructure, such as ventilation shafts, is a hazardous and inefficient task for humans. Unmanned Aerial Vehicles (UAVs) offer a promising alternative, but GPS-denied environments require robust control policies to prevent collisions. Deep Reinforcement Learning (DRL) has emerged as a powerful framework for developing such policies, and this paper provides a comparative study of two leading DRL algorithms for this task: the on-policy Proximal Policy Optimization (PPO) and the off-policy Soft Actor-Critic (SAC). The training was conducted with procedurally generated duct environments in Genesis simulation environment. A reward function was designed to guide a drone through a series of waypoints while applying a significant penalty for collisions. PPO learned a stable policy that completed all evaluation episodes without collision, producing smooth trajectories. By contrast, SAC consistently converged to a suboptimal behavior that traversed only the initial segments before failure. These results suggest that, in hazard-dense navigation, the training stability of on-policy methods can outweigh the nominal sample efficiency of off-policy algorithms. More broadly, the study provides evidence that procedurally generated, high-fidelity simulations are effective testbeds for developing and benchmarking robust navigation policies.
>
---
#### [new 009] A Workflow for Map Creation in Autonomous Vehicle Simulations
- **分类: cs.RO; cs.AI; cs.GR**

- **简介: 该论文针对自动驾驶仿真中地图创建困难的问题，提出一种高效、灵活的定制化工作流程，用于生成仿真用3D地图，以支持定位、路径规划与场景测试。**

- **链接: [http://arxiv.org/pdf/2508.16856v1](http://arxiv.org/pdf/2508.16856v1)**

> **作者:** Zubair Islam; Ahmaad Ansari; George Daoud; Mohamed El-Darieby
>
> **备注:** 6 pages, 12 figures. Published in the Proceedings of GEOProcessing 2025: The Seventeenth International Conference on Advanced Geographic Information Systems, Applications, and Services (IARIA)
>
> **摘要:** The fast development of technology and artificial intelligence has significantly advanced Autonomous Vehicle (AV) research, emphasizing the need for extensive simulation testing. Accurate and adaptable maps are critical in AV development, serving as the foundation for localization, path planning, and scenario testing. However, creating simulation-ready maps is often difficult and resource-intensive, especially with simulators like CARLA (CAR Learning to Act). Many existing workflows require significant computational resources or rely on specific simulators, limiting flexibility for developers. This paper presents a custom workflow to streamline map creation for AV development, demonstrated through the generation of a 3D map of a parking lot at Ontario Tech University. Future work will focus on incorporating SLAM technologies, optimizing the workflow for broader simulator compatibility, and exploring more flexible handling of latitude and longitude values to enhance map generation accuracy.
>
---
#### [new 010] Effect of Performance Feedback Timing on Motor Learning for a Surgical Training Task
- **分类: cs.RO**

- **简介: 论文研究实时与事后反馈对虚拟外科训练任务中运动学习的影响。针对机器人辅助手术培训中缺乏最优训练方法的问题，通过三组对比实验发现，实时多感官反馈显著提升学习速度和准确性，尤其在曲线路径上的位置精度和直线路径上的方向精度上表现更优。**

- **链接: [http://arxiv.org/pdf/2508.17830v1](http://arxiv.org/pdf/2508.17830v1)**

> **作者:** Mary Kate Gale; Kailana Baker-Matsuoka; Ilana Nisky; Allison Okamura
>
> **备注:** Submitted to IEEE Transactions on Biomedical Engineering
>
> **摘要:** Objective: Robot-assisted minimally invasive surgery (RMIS) has become the gold standard for a variety of surgical procedures, but the optimal method of training surgeons for RMIS is unknown. We hypothesized that real-time, rather than post-task, error feedback would better increase learning speed and reduce errors. Methods: Forty-two surgical novices learned a virtual version of the ring-on-wire task, a canonical task in RMIS training. We investigated the impact of feedback timing with multi-sensory (haptic and visual) cues in three groups: (1) real-time error feedback, (2) trial replay with error feedback, and (3) no error feedback. Results: Participant performance was evaluated based on the accuracy of ring position and orientation during the task. Participants who received real-time feedback outperformed other groups in ring orientation. Additionally, participants who received feedback in replay outperformed participants who did not receive any error feedback on ring orientation during long, straight path sections. There were no significant differences between groups for ring position overall, but participants who received real-time feedback outperformed the other groups in positional accuracy on tightly curved path sections. Conclusion: The addition of real-time haptic and visual error feedback improves learning outcomes in a virtual surgical task over error feedback in replay or no error feedback at all. Significance: This work demonstrates that multi-sensory error feedback delivered in real time leads to better training outcomes as compared to the same feedback delivered after task completion. This novel method of training may enable surgical trainees to develop skills with greater speed and accuracy.
>
---
#### [new 011] Talking to Robots: A Practical Examination of Speech Foundation Models for HRI Applications
- **分类: cs.RO; cs.AI; cs.CL; cs.HC**

- **简介: 论文研究语音识别在人机交互中的应用，解决真实场景下音频质量差、用户多样性导致的识别难题。通过评估四个前沿ASR系统在八大数据集上的表现，揭示性能差异、幻觉倾向和偏见问题，为HRI中的可靠语音交互提供实证依据。**

- **链接: [http://arxiv.org/pdf/2508.17753v1](http://arxiv.org/pdf/2508.17753v1)**

> **作者:** Theresa Pekarek Rosin; Julia Gachot; Henri-Leon Kordt; Matthias Kerzel; Stefan Wermter
>
> **备注:** Accepted at the workshop on Foundation Models for Social Robotics (FoMoSR) at ICSR 2025
>
> **摘要:** Automatic Speech Recognition (ASR) systems in real-world settings need to handle imperfect audio, often degraded by hardware limitations or environmental noise, while accommodating diverse user groups. In human-robot interaction (HRI), these challenges intersect to create a uniquely challenging recognition environment. We evaluate four state-of-the-art ASR systems on eight publicly available datasets that capture six dimensions of difficulty: domain-specific, accented, noisy, age-variant, impaired, and spontaneous speech. Our analysis demonstrates significant variations in performance, hallucination tendencies, and inherent biases, despite similar scores on standard benchmarks. These limitations have serious implications for HRI, where recognition errors can interfere with task performance, user trust, and safety.
>
---
#### [new 012] Robotic Manipulation via Imitation Learning: Taxonomy, Evolution, Benchmark, and Challenges
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在通过模仿学习提升机器人在现实环境中的物体操作能力。作者系统梳理了该领域的研究进展，包括方法分类、技术实现、基准测试与挑战分析，为后续研究提供全面参考。**

- **链接: [http://arxiv.org/pdf/2508.17449v1](http://arxiv.org/pdf/2508.17449v1)**

> **作者:** Zezeng Li; Alexandre Chapin; Enda Xiang; Rui Yang; Bruno Machado; Na Lei; Emmanuel Dellandrea; Di Huang; Liming Chen
>
> **摘要:** Robotic Manipulation (RM) is central to the advancement of autonomous robots, enabling them to interact with and manipulate objects in real-world environments. This survey focuses on RM methodologies that leverage imitation learning, a powerful technique that allows robots to learn complex manipulation skills by mimicking human demonstrations. We identify and analyze the most influential studies in this domain, selected based on community impact and intrinsic quality. For each paper, we provide a structured summary, covering the research purpose, technical implementation, hierarchical classification, input formats, key priors, strengths and limitations, and citation metrics. Additionally, we trace the chronological development of imitation learning techniques within RM policy (RMP), offering a timeline of key technological advancements. Where available, we report benchmark results and perform quantitative evaluations to compare existing methods. By synthesizing these insights, this review provides a comprehensive resource for researchers and practitioners, highlighting both the state of the art and the challenges that lie ahead in the field of robotic manipulation through imitation learning.
>
---
#### [new 013] No Need to Look! Locating and Grasping Objects by a Robot Arm Covered with Sensitive Skin
- **分类: cs.RO**

- **简介: 论文研究机器人在无视觉条件下仅靠触觉感知定位并抓取物体的任务。通过全身敏感皮肤进行粗略探索，再用末端力/扭矩传感器精确定位，成功实现对多种物体的抓取，速度比仅用末端触觉快六倍，适用于视觉受限场景。**

- **链接: [http://arxiv.org/pdf/2508.17986v1](http://arxiv.org/pdf/2508.17986v1)**

> **作者:** Karel Bartunek; Lukas Rustler; Matej Hoffmann
>
> **备注:** Submitted for review to ICRA 2026
>
> **摘要:** Locating and grasping of objects by robots is typically performed using visual sensors. Haptic feedback from contacts with the environment is only secondary if present at all. In this work, we explored an extreme case of searching for and grasping objects in complete absence of visual input, relying on haptic feedback only. The main novelty lies in the use of contacts over the complete surface of a robot manipulator covered with sensitive skin. The search is divided into two phases: (1) coarse workspace exploration with the complete robot surface, followed by (2) precise localization using the end-effector equipped with a force/torque sensor. We systematically evaluated this method in simulation and on the real robot, demonstrating that diverse objects can be located, grasped, and put in a basket. The overall success rate on the real robot for one object was 85.7\% with failures mainly while grasping specific objects. The method using whole-body contacts is six times faster compared to a baseline that uses haptic feedback only on the end-effector. We also show locating and grasping multiple objects on the table. This method is not restricted to our specific setup and can be deployed on any platform with the ability of sensing contacts over the entire body surface. This work holds promise for diverse applications in areas with challenging visual perception (due to lighting, dust, smoke, occlusion) such as in agriculture when fruits or vegetables need to be located inside foliage and picked.
>
---
#### [new 014] HumanoidVerse: A Versatile Humanoid for Vision-Language Guided Multi-Object Rearrangement
- **分类: cs.RO; cs.AI**

- **简介: 论文提出HumanoidVerse框架，用于视觉语言引导的类人机器人多物体重排任务。解决单一对象、固定场景下任务泛化难的问题，通过多阶段训练和双教师蒸馏实现复杂序列操作，显著提升成功率与环境适应性。**

- **链接: [http://arxiv.org/pdf/2508.16943v1](http://arxiv.org/pdf/2508.16943v1)**

> **作者:** Haozhuo Zhang; Jingkai Sun; Michele Caprio; Jian Tang; Shanghang Zhang; Qiang Zhang; Wei Pan
>
> **备注:** Project Page: https://haozhuo-zhang.github.io/HumanoidVerse-project-page/
>
> **摘要:** We introduce HumanoidVerse, a novel framework for vision-language guided humanoid control that enables a single physically simulated robot to perform long-horizon, multi-object rearrangement tasks across diverse scenes. Unlike prior methods that operate in fixed settings with single-object interactions, our approach supports consecutive manipulation of multiple objects, guided only by natural language instructions and egocentric camera RGB observations. HumanoidVerse is trained via a multi-stage curriculum using a dual-teacher distillation pipeline, enabling fluid transitions between sub-tasks without requiring environment resets. To support this, we construct a large-scale dataset comprising 350 multi-object tasks spanning four room layouts. Extensive experiments in the Isaac Gym simulator demonstrate that our method significantly outperforms prior state-of-the-art in both task success rate and spatial precision, and generalizes well to unseen environments and instructions. Our work represents a key step toward robust, general-purpose humanoid agents capable of executing complex, sequential tasks under real-world sensory constraints. The video visualization results can be found on the project page: https://haozhuo-zhang.github.io/HumanoidVerse-project-page/.
>
---
#### [new 015] Evolutionary Brain-Body Co-Optimization Consistently Fails to Select for Morphological Potential
- **分类: cs.RO; cs.NE**

- **简介: 论文研究进化脑体协同优化问题，旨在理解为何现有算法难以找到最优形态。作者在130万种形态中系统映射适应度景观，发现算法常因低估新形态潜力而停滞，无法有效追踪适应度梯度。**

- **链接: [http://arxiv.org/pdf/2508.17464v1](http://arxiv.org/pdf/2508.17464v1)**

> **作者:** Alican Mertan; Nick Cheney
>
> **备注:** Accepted to be presented at ALife 2025 as a talk
>
> **摘要:** Brain-body co-optimization remains a challenging problem, despite increasing interest from the community in recent years. To understand and overcome the challenges, we propose exhaustively mapping a morphology-fitness landscape to study it. To this end, we train controllers for each feasible morphology in a design space of 1,305,840 distinct morphologies, constrained by a computational budget. First, we show that this design space constitutes a good model for studying the brain-body co-optimization problem, and our attempt to exhaustively map it roughly captures the landscape. We then proceed to analyze how evolutionary brain-body co-optimization algorithms work in this design space. The complete knowledge of the morphology-fitness landscape facilitates a better understanding of the results of evolutionary brain-body co-optimization algorithms and how they unfold over evolutionary time in the morphology space. This investigation shows that the experimented algorithms cannot consistently find near-optimal solutions. The search, at times, gets stuck on morphologies that are sometimes one mutation away from better morphologies, and the algorithms cannot efficiently track the fitness gradient in the morphology-fitness landscape. We provide evidence that experimented algorithms regularly undervalue the fitness of individuals with newly mutated bodies and, as a result, eliminate promising morphologies throughout evolution. Our work provides the most concrete demonstration of the challenges of evolutionary brain-body co-optimization. Our findings ground the trends in the literature and provide valuable insights for future work.
>
---
#### [new 016] A holistic perception system of internal and external monitoring for ground autonomous vehicles: AutoTRUST paradigm
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AutoTRUST感知系统，整合车内与车外监控，解决自动驾驶车辆对内外环境的全面感知问题。通过多摄像头、大语言模型和智能传感器实现车内行为识别与舒适度优化，利用LiDAR进行高效语义分割与点云超分辨率，提升感知精度与效率。**

- **链接: [http://arxiv.org/pdf/2508.17969v1](http://arxiv.org/pdf/2508.17969v1)**

> **作者:** Alexandros Gkillas; Christos Anagnostopoulos; Nikos Piperigkos; Dimitris Tsiktsiris; Theofilos Christodoulou; Theofanis Siamatras; Dimitrios Triantafyllou; Christos Basdekis; Theoktisti Marinopoulou; Panagiotis Lepentsiotis; Elefterios Blitsis; Aggeliki Zacharaki; Nearchos Stylianidis; Leonidas Katelaris; Lamberto Salvan; Aris S. Lalos; Christos Laoudias; Antonios Lalas; Konstantinos Votis
>
> **摘要:** This paper introduces a holistic perception system for internal and external monitoring of autonomous vehicles, with the aim of demonstrating a novel AI-leveraged self-adaptive framework of advanced vehicle technologies and solutions that optimize perception and experience on-board. Internal monitoring system relies on a multi-camera setup designed for predicting and identifying driver and occupant behavior through facial recognition, exploiting in addition a large language model as virtual assistant. Moreover, the in-cabin monitoring system includes AI-empowered smart sensors that measure air-quality and perform thermal comfort analysis for efficient on and off-boarding. On the other hand, external monitoring system perceives the surrounding environment of vehicle, through a LiDAR-based cost-efficient semantic segmentation approach, that performs highly accurate and efficient super-resolution on low-quality raw 3D point clouds. The holistic perception framework is developed in the context of EU's Horizon Europe programm AutoTRUST, and has been integrated and deployed on a real electric vehicle provided by ALKE. Experimental validation and evaluation at the integration site of Joint Research Centre at Ispra, Italy, highlights increased performance and efficiency of the modular blocks of the proposed perception architecture.
>
---
#### [new 017] Scene-Agnostic Traversability Labeling and Estimation via a Multimodal Self-supervised Framework
- **分类: cs.RO; cs.CV**

- **简介: 论文提出一种多模态自监督框架，用于机器人 traversability（可通行性）估计任务。针对现有方法忽略非可通行区域和单一模态局限的问题，该工作结合LiDAR、相机与足迹数据生成标签，并训练双流网络联合学习多模态特征，提升估计准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.18249v1](http://arxiv.org/pdf/2508.18249v1)**

> **作者:** Zipeng Fang; Yanbo Wang; Lei Zhao; Weidong Chen
>
> **摘要:** Traversability estimation is critical for enabling robots to navigate across diverse terrains and environments. While recent self-supervised learning methods achieve promising results, they often fail to capture the characteristics of non-traversable regions. Moreover, most prior works concentrate on a single modality, overlooking the complementary strengths offered by integrating heterogeneous sensory modalities for more robust traversability estimation. To address these limitations, we propose a multimodal self-supervised framework for traversability labeling and estimation. First, our annotation pipeline integrates footprint, LiDAR, and camera data as prompts for a vision foundation model, generating traversability labels that account for both semantic and geometric cues. Then, leveraging these labels, we train a dual-stream network that jointly learns from different modalities in a decoupled manner, enhancing its capacity to recognize diverse traversability patterns. In addition, we incorporate sparse LiDAR-based supervision to mitigate the noise introduced by pseudo labels. Finally, extensive experiments conducted across urban, off-road, and campus environments demonstrate the effectiveness of our approach. The proposed automatic labeling method consistently achieves around 88% IoU across diverse datasets. Compared to existing self-supervised state-of-the-art methods, our multimodal traversability estimation network yields consistently higher IoU, improving by 1.6-3.5% on all evaluated datasets.
>
---
#### [new 018] Relative Navigation and Dynamic Target Tracking for Autonomous Underwater Proximity Operations
- **分类: cs.RO; cs.SY; eess.SP; eess.SY; I.2.9; I.2.8; F.2.2**

- **简介: 论文解决水下近距离自主操作中目标6-DoF运动估计难题，提出基于李群切空间的广义恒定扭转变量先验，通过二元因子和闭式雅可比矩阵实现跨表示的一致轨迹估计，提升USBL仅测量下的跟踪精度。**

- **链接: [http://arxiv.org/pdf/2508.16901v1](http://arxiv.org/pdf/2508.16901v1)**

> **作者:** David Baxter; Aldo Terán Espinoza; Antonio Terán Espinoza; Amy Loutfi; John Folkesson; Peter Sigray; Stephanie Lowry; Jakob Kuttenkeuler
>
> **备注:** 10 pages, 7 figures. Equal contribution by David Baxter and Aldo Ter\'an Espinoza. Supported by SAAB, SMaRC, and WASP. Supported by SAAB and the Swedish Maritime Robotics Centre (SMaRC), and by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation
>
> **摘要:** Estimating a target's 6-DoF motion in underwater proximity operations is difficult because the chaser lacks target-side proprioception and the available relative observations are sparse, noisy, and often partial (e.g., Ultra-Short Baseline (USBL) positions). Without a motion prior, factor-graph maximum a posteriori estimation is underconstrained: consecutive target states are weakly linked and orientation can drift. We propose a generalized constant-twist motion prior defined on the tangent space of Lie groups that enforces temporally consistent trajectories across all degrees of freedom; in SE(3) it couples translation and rotation in the body frame. We present a ternary factor and derive its closed-form Jacobians based on standard Lie group operations, enabling drop-in use for trajectories on arbitrary Lie groups. We evaluate two deployment modes: (A) an SE(3)-only representation that regularizes orientation even when only position is measured, and (B) a mode with boundary factors that switches the target representation between SE(3) and 3D position while applying the same generalized constant-twist prior across representation changes. Validation on a real-world dynamic docking scenario dataset shows consistent ego-target trajectory estimation through USBL-only and optical relative measurement segments with an improved relative tracking accuracy compared to the noisy measurements to the target. Because the construction relies on standard Lie group primitives, it is portable across state manifolds and sensing modalities.
>
---
#### [new 019] Egocentric Instruction-oriented Affordance Prediction via Large Multimodal Model
- **分类: cs.RO; cs.CV**

- **简介: 论文提出任务依赖的具身 affordance 预测方法，解决传统方法忽视指令影响的问题。构建1.5万条物体-指令-可操作性三元组数据集，设计“搜索-验证”流水线，利用大模型迭代推理实现精准预测。**

- **链接: [http://arxiv.org/pdf/2508.17922v1](http://arxiv.org/pdf/2508.17922v1)**

> **作者:** Bokai Ji; Jie Gu; Xiaokang Ma; Chu Tang; Jingmin Chen; Guangxia Li
>
> **摘要:** Affordance is crucial for intelligent robots in the context of object manipulation. In this paper, we argue that affordance should be task-/instruction-dependent, which is overlooked by many previous works. That is, different instructions can lead to different manipulation regions and directions even for the same object. According to this observation, we present a new dataset comprising fifteen thousand object-instruction-affordance triplets. All scenes in the dataset are from an egocentric viewpoint, designed to approximate the perspective of a human-like robot. Furthermore, we investigate how to enable large multimodal models (LMMs) to serve as affordance predictors by implementing a ``search against verifiers'' pipeline. An LMM is asked to progressively predict affordances, with the output at each step being verified by itself during the iterative process, imitating a reasoning process. Experiments show that our method not only unlocks new instruction-oriented affordance prediction capabilities, but also achieves outstanding performance broadly.
>
---
#### [new 020] MEVITA: Open-Source Bipedal Robot Assembled from E-Commerce Components via Sheet Metal Welding
- **分类: cs.RO**

- **简介: 该论文提出MEVITA，一个可完全用电商零件组装的开源双足机器人，通过钣金焊接减少部件数量、简化装配。旨在解决现有开源机器人依赖3D打印（结构脆弱）或金属部件难获取的问题，实现低成本、易搭建且能通过强化学习实现稳定行走的双足机器人。**

- **链接: [http://arxiv.org/pdf/2508.17684v1](http://arxiv.org/pdf/2508.17684v1)**

> **作者:** Kento Kawaharazuka; Shogo Sawaguchi; Ayumu Iwata; Keita Yoneda; Temma Suzuki; Kei Okada
>
> **备注:** Accepted at IEEE-RAS Humanoids2025, Website - https://haraduka.github.io/mevita-hardware , YouTube - https://youtu.be/_akfHkCne0s
>
> **摘要:** Various bipedal robots have been developed to date, and in recent years, there has been a growing trend toward releasing these robots as open-source platforms. This shift is fostering an environment in which anyone can freely develop bipedal robots and share their knowledge, rather than relying solely on commercial products. However, most existing open-source bipedal robots are designed to be fabricated using 3D printers, which limits their scalability in size and often results in fragile structures. On the other hand, some metal-based bipedal robots have been developed, but they typically involve a large number of components, making assembly difficult, and in some cases, the parts themselves are not readily available through e-commerce platforms. To address these issues, we developed MEVITA, an open-source bipedal robot that can be built entirely from components available via e-commerce. Aiming for the minimal viable configuration for a bipedal robot, we utilized sheet metal welding to integrate complex geometries into single parts, thereby significantly reducing the number of components and enabling easy assembly for anyone. Through reinforcement learning in simulation and Sim-to-Real transfer, we demonstrated robust walking behaviors across various environments, confirming the effectiveness of our approach. All hardware, software, and training environments can be obtained from https://github.com/haraduka/mevita .
>
---
#### [new 021] Arnold: a generalist muscle transformer policy
- **分类: cs.RO; cs.AI; cs.LG; q-bio.QM**

- **简介: 该论文提出Arnold，一个通用肌肉变换器策略，解决多任务、多机体控制难题。通过传感器运动词汇和Transformer架构，实现14项复杂任务的专家级表现，支持快速适应新任务，并揭示肌肉协同在任务间转移有限。**

- **链接: [http://arxiv.org/pdf/2508.18066v1](http://arxiv.org/pdf/2508.18066v1)**

> **作者:** Alberto Silvio Chiappa; Boshi An; Merkourios Simos; Chengkun Li; Alexander Mathis
>
> **备注:** A.S.C. and B.A. contributed equally. Code is available at https://github.com/amathislab/arnold-the-generalist
>
> **摘要:** Controlling high-dimensional and nonlinear musculoskeletal models of the human body is a foundational scientific challenge. Recent machine learning breakthroughs have heralded policies that master individual skills like reaching, object manipulation and locomotion in musculoskeletal systems with many degrees of freedom. However, these agents are merely "specialists", achieving high performance for a single skill. In this work, we develop Arnold, a generalist policy that masters multiple tasks and embodiments. Arnold combines behavior cloning and fine-tuning with PPO to achieve expert or super-expert performance in 14 challenging control tasks from dexterous object manipulation to locomotion. A key innovation is Arnold's sensorimotor vocabulary, a compositional representation of the semantics of heterogeneous sensory modalities, objectives, and actuators. Arnold leverages this vocabulary via a transformer architecture to deal with the variable observation and action spaces of each task. This framework supports efficient multi-task, multi-embodiment learning and facilitates rapid adaptation to novel tasks. Finally, we analyze Arnold to provide insights into biological motor control, corroborating recent findings on the limited transferability of muscle synergies across tasks.
>
---
#### [new 022] Integration of Computer Vision with Adaptive Control for Autonomous Driving Using ADORE
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶感知与控制融合任务，解决CV模型在恶劣天气下性能下降的问题。工作包括：将YOLO视觉模型与ADORE自适应控制框架结合，通过ROS桥接实现模块实时通信，在CARLA模拟器中验证了系统在不同天气下的鲁棒性与低延迟响应能力。**

- **链接: [http://arxiv.org/pdf/2508.17985v1](http://arxiv.org/pdf/2508.17985v1)**

> **作者:** Abu Shad Ahammed; Md Shahi Amran Hossain; Sayeri Mukherjee; Roman Obermaisser; Md. Ziaur Rahman
>
> **摘要:** Ensuring safety in autonomous driving requires a seamless integration of perception and decision making under uncertain conditions. Although computer vision (CV) models such as YOLO achieve high accuracy in detecting traffic signs and obstacles, their performance degrades in drift scenarios caused by weather variations or unseen objects. This work presents a simulated autonomous driving system that combines a context aware CV model with adaptive control using the ADORE framework. The CARLA simulator was integrated with ADORE via the ROS bridge, allowing real-time communication between perception, decision, and control modules. A simulated test case was designed in both clear and drift weather conditions to demonstrate the robust detection performance of the perception model while ADORE successfully adapted vehicle behavior to speed limits and obstacles with low response latency. The findings highlight the potential of coupling deep learning-based perception with rule-based adaptive decision making to improve automotive safety critical system.
>
---
#### [new 023] SafeBimanual: Diffusion-based Trajectory Optimization for Safe Bimanual Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 论文提出SafeBimanual框架，用于安全双臂操作任务。解决扩散模型忽视物理安全约束导致危险行为的问题。通过设计多样成本函数与视觉语言模型动态调度，优化轨迹并提升成功率。**

- **链接: [http://arxiv.org/pdf/2508.18268v1](http://arxiv.org/pdf/2508.18268v1)**

> **作者:** Haoyuan Deng; Wenkai Guo; Qianzhun Wang; Zhenyu Wu; Ziwei Wang
>
> **备注:** Project website is at: https://denghaoyuan123.github.io/SafeBimanip/
>
> **摘要:** Bimanual manipulation has been widely applied in household services and manufacturing, which enables the complex task completion with coordination requirements. Recent diffusion-based policy learning approaches have achieved promising performance in modeling action distributions for bimanual manipulation. However, they ignored the physical safety constraints of bimanual manipulation, which leads to the dangerous behaviors with damage to robots and objects. To this end, we propose a test-time trajectory optimization framework named SafeBimanual for any pre-trained diffusion-based bimanual manipulation policies, which imposes the safety constraints on bimanual actions to avoid dangerous robot behaviors with improved success rate. Specifically, we design diverse cost functions for safety constraints in different dual-arm cooperation patterns including avoidance of tearing objects and collision between arms and objects, which optimizes the manipulator trajectories with guided sampling of diffusion denoising process. Moreover, we employ a vision-language model (VLM) to schedule the cost functions by specifying keypoints and corresponding pairwise relationship, so that the optimal safety constraint is dynamically generated in the entire bimanual manipulation process. SafeBimanual demonstrates superiority on 8 simulated tasks in RoboTwin with a 13.7% increase in success rate and a 18.8% reduction in unsafe interactions over state-of-the-art diffusion-based methods. Extensive experiments on 4 real-world tasks further verify its practical value by improving the success rate by 32.5%.
>
---
#### [new 024] Morphological Cognition: Classifying MNIST Digits Through Morphological Computation Alone
- **分类: cs.RO; cs.NE**

- **简介: 论文提出“形态认知”概念，通过无神经网络的物理结构实现MNIST数字分类：零使机器人左移，一使右移。解决如何用形态过程替代神经计算完成认知任务的问题，首次展示无需神经电路的高阶认知行为。**

- **链接: [http://arxiv.org/pdf/2508.17469v1](http://arxiv.org/pdf/2508.17469v1)**

> **作者:** Alican Mertan; Nick Cheney
>
> **备注:** Accepted to be presented at ALife 2025 as a talk
>
> **摘要:** With the rise of modern deep learning, neural networks have become an essential part of virtually every artificial intelligence system, making it difficult even to imagine different models for intelligent behavior. In contrast, nature provides us with many different mechanisms for intelligent behavior, most of which we have yet to replicate. One of such underinvestigated aspects of intelligence is embodiment and the role it plays in intelligent behavior. In this work, we focus on how the simple and fixed behavior of constituent parts of a simulated physical body can result in an emergent behavior that can be classified as cognitive by an outside observer. Specifically, we show how simulated voxels with fixed behaviors can be combined to create a robot such that, when presented with an image of an MNIST digit zero, it moves towards the left; and when it is presented with an image of an MNIST digit one, it moves towards the right. Such robots possess what we refer to as ``morphological cognition'' -- the ability to perform cognitive behavior as a result of morphological processes. To the best of our knowledge, this is the first demonstration of a high-level mental faculty such as image classification performed by a robot without any neural circuitry. We hope that this work serves as a proof-of-concept and fosters further research into different models of intelligence.
>
---
#### [new 025] COSMO-Bench: A Benchmark for Collaborative SLAM Optimization
- **分类: cs.RO**

- **简介: 论文提出COSMO-Bench，一个用于多机器人协同SLAM优化的基准数据集，解决缺乏标准测试数据的问题，促进分布式优化算法研究。**

- **链接: [http://arxiv.org/pdf/2508.16731v1](http://arxiv.org/pdf/2508.16731v1)**

> **作者:** Daniel McGann; Easton R. Potokar; Michael Kaess
>
> **摘要:** Recent years have seen a focus on research into distributed optimization algorithms for multi-robot Collaborative Simultaneous Localization and Mapping (C-SLAM). Research in this domain, however, is made difficult by a lack of standard benchmark datasets. Such datasets have been used to great effect in the field of single-robot SLAM, and researchers focused on multi-robot problems would benefit greatly from dedicated benchmark datasets. To address this gap, we design and release the Collaborative Open-Source Multi-robot Optimization Benchmark (COSMO-Bench) -- a suite of 24 datasets derived from a state-of-the-art C-SLAM front-end and real-world LiDAR data. Data DOI: https://doi.org/10.1184/R1/29652158
>
---
#### [new 026] Optimizing Grasping in Legged Robots: A Deep Learning Approach to Loco-Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 论文提出一种基于深度学习的四足机器人抓取优化方法，解决动态环境中精准抓取难题。通过仿真生成带标注的抓取质量图，训练U-Net模型预测最优抓取点，实现自主导航、感知与抓取的完整任务。**

- **链接: [http://arxiv.org/pdf/2508.17466v1](http://arxiv.org/pdf/2508.17466v1)**

> **作者:** Dilermando Almeida; Guilherme Lazzarini; Juliano Negri; Thiago H. Segreto; Ricardo V. Godoy; Marcelo Becker
>
> **摘要:** Quadruped robots have emerged as highly efficient and versatile platforms, excelling in navigating complex and unstructured terrains where traditional wheeled robots might fail. Equipping these robots with manipulator arms unlocks the advanced capability of loco-manipulation to perform complex physical interaction tasks in areas ranging from industrial automation to search-and-rescue missions. However, achieving precise and adaptable grasping in such dynamic scenarios remains a significant challenge, often hindered by the need for extensive real-world calibration and pre-programmed grasp configurations. This paper introduces a deep learning framework designed to enhance the grasping capabilities of quadrupeds equipped with arms, focusing on improved precision and adaptability. Our approach centers on a sim-to-real methodology that minimizes reliance on physical data collection. We developed a pipeline within the Genesis simulation environment to generate a synthetic dataset of grasp attempts on common objects. By simulating thousands of interactions from various perspectives, we created pixel-wise annotated grasp-quality maps to serve as the ground truth for our model. This dataset was used to train a custom CNN with a U-Net-like architecture that processes multi-modal input from an onboard RGB and depth cameras, including RGB images, depth maps, segmentation masks, and surface normal maps. The trained model outputs a grasp-quality heatmap to identify the optimal grasp point. We validated the complete framework on a four-legged robot. The system successfully executed a full loco-manipulation task: autonomously navigating to a target object, perceiving it with its sensors, predicting the optimal grasp pose using our model, and performing a precise grasp. This work proves that leveraging simulated training with advanced sensing offers a scalable and effective solution for object handling.
>
---
#### [new 027] FlowVLA: Thinking in Motion with a Visual Chain of Thought
- **分类: cs.RO**

- **简介: 论文提出FlowVLA，一种基于视觉思维链的视觉-语言-动作模型，通过先预测光流再生成未来图像来分离运动与外观，提升物理推理能力和策略学习效率。**

- **链接: [http://arxiv.org/pdf/2508.18269v1](http://arxiv.org/pdf/2508.18269v1)**

> **作者:** Zhide Zhong; Haodong Yan; Junfeng Li; Xiangchen Liu; Xin Gong; Wenxuan Song; Jiayi Chen; Haoang Li
>
> **摘要:** Many Vision-Language-Action (VLA) models rely on an internal world model trained via next-frame prediction. This approach, however, struggles with physical reasoning as it entangles static appearance with dynamic motion, often resulting in implausible visual forecasts and inefficient policy learning. To address these limitations, we introduce the Visual Chain of Thought (Visual CoT): a pre-training framework that encourages a model to reason about how a scene evolves before predicting what it will look like. We instantiate this principle in FlowVLA, which predicts a future frame ($v_{t+1}$) only after generating an intermediate optical flow representation ($f_t$) that encodes motion dynamics. This ``$v_t \rightarrow f_t \rightarrow v_{t+1}$'' reasoning process is implemented within a single autoregressive Transformer, guiding the model to learn disentangled dynamics. As a result, FlowVLA produces coherent visual predictions and facilitates more efficient policy learning. Experiments on challenging robotics manipulation benchmarks demonstrate state-of-the-art performance with substantially improved sample efficiency, pointing toward a more principled foundation for world modeling. Project page: https://irpn-lab.github.io/FlowVLA/
>
---
#### [new 028] SEBVS: Synthetic Event-based Visual Servoing for Robot Navigation and Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 论文提出SEBVS，一个用于机器人导航与操作的事件相机仿真系统，解决合成事件数据在主流机器人仿真中缺失的问题。通过Gazebo实现RGB到事件流的转换，训练基于Transformer的事件驱动策略，在跟随和抓取任务中优于RGB方法，验证了事件视觉在实时机器人控制中的潜力。**

- **链接: [http://arxiv.org/pdf/2508.17643v1](http://arxiv.org/pdf/2508.17643v1)**

> **作者:** Krishna Vinod; Prithvi Jai Ramesh; Pavan Kumar B N; Bharatesh Chakravarthi
>
> **摘要:** Event cameras offer microsecond latency, high dynamic range, and low power consumption, making them ideal for real-time robotic perception under challenging conditions such as motion blur, occlusion, and illumination changes. However, despite their advantages, synthetic event-based vision remains largely unexplored in mainstream robotics simulators. This lack of simulation setup hinders the evaluation of event-driven approaches for robotic manipulation and navigation tasks. This work presents an open-source, user-friendly v2e robotics operating system (ROS) package for Gazebo simulation that enables seamless event stream generation from RGB camera feeds. The package is used to investigate event-based robotic policies (ERP) for real-time navigation and manipulation. Two representative scenarios are evaluated: (1) object following with a mobile robot and (2) object detection and grasping with a robotic manipulator. Transformer-based ERPs are trained by behavior cloning and compared to RGB-based counterparts under various operating conditions. Experimental results show that event-guided policies consistently deliver competitive advantages. The results highlight the potential of event-driven perception to improve real-time robotic navigation and manipulation, providing a foundation for broader integration of event cameras into robotic policy learning. The GitHub repo for the dataset and code: https://eventbasedvision.github.io/SEBVS/
>
---
#### [new 029] CubeDN: Real-time Drone Detection in 3D Space from Dual mmWave Radar Cubes
- **分类: cs.RO**

- **简介: 论文提出CubeDN，一种基于双毫米波雷达的实时3D无人机检测网络，解决光学传感器在恶劣环境下性能下降的问题，实现高精度无人机检测与定位。**

- **链接: [http://arxiv.org/pdf/2508.17831v1](http://arxiv.org/pdf/2508.17831v1)**

> **作者:** Yuan Fang; Fangzhan Shi; Xijia Wei; Qingchao Chen; Kevin Chetty; Simon Julier
>
> **摘要:** As drone use has become more widespread, there is a critical need to ensure safety and security. A key element of this is robust and accurate drone detection and localization. While cameras and other optical sensors like LiDAR are commonly used for object detection, their performance degrades under adverse lighting and environmental conditions. Therefore, this has generated interest in finding more reliable alternatives, such as millimeter-wave (mmWave) radar. Recent research on mmWave radar object detection has predominantly focused on 2D detection of road users. Although these systems demonstrate excellent performance for 2D problems, they lack the sensing capability to measure elevation, which is essential for 3D drone detection. To address this gap, we propose CubeDN, a single-stage end-to-end radar object detection network specifically designed for flying drones. CubeDN overcomes challenges such as poor elevation resolution by utilizing a dual radar configuration and a novel deep learning pipeline. It simultaneously detects, localizes, and classifies drones of two sizes, achieving decimeter-level tracking accuracy at closer ranges with overall $95\%$ average precision (AP) and $85\%$ average recall (AR). Furthermore, CubeDN completes data processing and inference at 10Hz, making it highly suitable for practical applications.
>
---
#### [new 030] Adaptive Output Steps: FlexiSteps Network for Dynamic Trajectory Prediction
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对轨迹预测任务，解决固定输出步长限制适应动态场景的问题。提出FlexiSteps Network（FSN），通过自适应预测模块和动态解码器实现可变长度预测，并设计评分机制平衡精度与步长。**

- **链接: [http://arxiv.org/pdf/2508.17797v1](http://arxiv.org/pdf/2508.17797v1)**

> **作者:** Yunxiang Liu; Hongkuo Niu; Jianlin Zhu
>
> **摘要:** Accurate trajectory prediction is vital for autonomous driving, robotics, and intelligent decision-making systems, yet traditional models typically rely on fixed-length output predictions, limiting their adaptability to dynamic real-world scenarios. In this paper, we introduce the FlexiSteps Network (FSN), a novel framework that dynamically adjusts prediction output time steps based on varying contextual conditions. Inspired by recent advancements addressing observation length discrepancies and dynamic feature extraction, FSN incorporates an pre-trained Adaptive Prediction Module (APM) to evaluate and adjust the output steps dynamically, ensuring optimal prediction accuracy and efficiency. To guarantee the plug-and-play of our FSN, we also design a Dynamic Decoder(DD). Additionally, to balance the prediction time steps and prediction accuracy, we design a scoring mechanism, which not only introduces the Fr\'echet distance to evaluate the geometric similarity between the predicted trajectories and the ground truth trajectories but the length of predicted steps is also considered. Extensive experiments conducted on benchmark datasets including Argoverse and INTERACTION demonstrate the effectiveness and flexibility of our proposed FSN framework.
>
---
#### [new 031] OVITA: Open-Vocabulary Interpretable Trajectory Adaptations
- **分类: cs.RO**

- **简介: 论文提出OVITA框架，解决机器人在动态环境中基于自然语言指令调整轨迹的问题。通过多LLM实现开放词汇、可解释的轨迹适配，支持用户灵活修改路径点，已在多种机器人平台验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.17260v1](http://arxiv.org/pdf/2508.17260v1)**

> **作者:** Anurag Maurya; Tashmoy Ghosh; Anh Nguyen; Ravi Prakash
>
> **备注:** Accepted to Robotics and Automation Letters 2025. Code link: https://github.com/anurag1000101/OVITA
>
> **摘要:** Adapting trajectories to dynamic situations and user preferences is crucial for robot operation in unstructured environments with non-expert users. Natural language enables users to express these adjustments in an interactive manner. We introduce OVITA, an interpretable, open-vocabulary, language-driven framework designed for adapting robot trajectories in dynamic and novel situations based on human instructions. OVITA leverages multiple pre-trained Large Language Models (LLMs) to integrate user commands into trajectories generated by motion planners or those learned through demonstrations. OVITA employs code as an adaptation policy generated by an LLM, enabling users to adjust individual waypoints, thus providing flexible control. Another LLM, which acts as a code explainer, removes the need for expert users, enabling intuitive interactions. The efficacy and significance of the proposed OVITA framework is demonstrated through extensive simulations and real-world environments with diverse tasks involving spatiotemporal variations on heterogeneous robotic platforms such as a KUKA IIWA robot manipulator, Clearpath Jackal ground robot, and CrazyFlie drone.
>
---
#### [new 032] LaGarNet: Goal-Conditioned Recurrent State-Space Models for Pick-and-Place Garment Flattening
- **分类: cs.RO**

- **简介: 论文提出LaGarNet，一种用于衣物平整化操作的目标条件循环状态空间模型，解决复杂衣物抓放任务的动态建模问题。通过少样本人类示范学习策略，在仿真与真实世界中实现四种衣物的高效平整。**

- **链接: [http://arxiv.org/pdf/2508.17070v1](http://arxiv.org/pdf/2508.17070v1)**

> **作者:** Halid Abdulrahim Kadi; Kasim Terzić
>
> **备注:** 20 pages, 11 figures and 3 tables
>
> **摘要:** We present a novel goal-conditioned recurrent state space (GC-RSSM) model capable of learning latent dynamics of pick-and-place garment manipulation. Our proposed method LaGarNet matches the state-of-the-art performance of mesh-based methods, marking the first successful application of state-space models on complex garments. LaGarNet trains on a coverage-alignment reward and a dataset collected through a general procedure supported by a random policy and a diffusion policy learned from few human demonstrations; it substantially reduces the inductive biases introduced in the previous similar methods. We demonstrate that a single-policy LaGarNet achieves flattening on four different types of garments in both real-world and simulation settings.
>
---
#### [new 033] The Effects of Communication Delay on Human Performance and Neurocognitive Responses in Mobile Robot Teleoperation
- **分类: cs.RO**

- **简介: 论文研究通信延迟对移动机器人遥操作中人类表现和神经认知的影响。通过10名参与者实验，结合EEG与机器人行为数据，发现200–300 ms延迟显著降低性能，100–200 ms为感知阈值，400 ms后认知资源饱和。成果为延迟补偿设计提供神经认知依据。**

- **链接: [http://arxiv.org/pdf/2508.18074v1](http://arxiv.org/pdf/2508.18074v1)**

> **作者:** Zhaokun Chen; Wenshuo Wang; Wenzhuo Liu; Yichen Liu; Junqiang Xi
>
> **摘要:** Communication delays in mobile robot teleoperation adversely affect human-machine collaboration. Understanding delay effects on human operational performance and neurocognition is essential for resolving this issue. However, no previous research has explored this. To fill this gap, we conduct a human-in-the-loop experiment involving 10 participants, integrating electroencephalography (EEG) and robot behavior data under varying delays (0-500 ms in 100 ms increments) to systematically investigate these effects. Behavior analysis reveals significant performance degradation at 200-300 ms delays, affecting both task efficiency and accuracy. EEG analysis discovers features with significant delay dependence: frontal $\theta/\beta$-band and parietal $\alpha$-band power. We also identify a threshold window (100-200 ms) for early perception of delay in humans, during which these EEG features first exhibit significant differences. When delay exceeds 400 ms, all features plateau, indicating saturation of cognitive resource allocation at physiological limits. These findings provide the first evidence of perceptual and cognitive delay thresholds during teleoperation tasks in humans, offering critical neurocognitive insights for the design of delay compensation strategies.
>
---
#### [new 034] Analysis of Harpy's Constrained Trotting and Jumping Maneuver
- **分类: cs.RO**

- **简介: 论文研究Harpy机器人在蹬踏与跳跃中的混合腿-推进器运动控制问题，通过实验数据分析揭示了稳定运动的机制：腿提供主要推力，推进器增强空中阶段控制，且系统具备对称性、低扭矩和抗扰动能力。**

- **链接: [http://arxiv.org/pdf/2508.18139v1](http://arxiv.org/pdf/2508.18139v1)**

> **作者:** Prathima Ananda Kumar
>
> **备注:** Master's Thesis
>
> **摘要:** This study presents an analysis of experimental data from Harpy, a thruster-assisted bipedal robot developed at Northeastern University. The study examines data sets from trotting and jumping experiments to understand the fundamental principles governing hybrid leg-thruster locomotion. Through data analysis across multiple locomotion modes, this research reveals that Harpy achieves stable locomotion with bounded trajectories and consistent foot placement through strategic leg-thruster synergy. The results demonstrate controlled joint behavior with low torques and symmetric tracking, accurate foot placement within kinematic constraints despite phase-transition perturbations, and underactuated degree-of-freedom stability without divergence. Energy level analysis reveals that legs provide primary propulsion, while the thrusters enable additional aerial phase control. The analysis identifies critical body-leg coupling dynamics during aerial phases that require phase-specific control strategies. Consistent repeatability and symmetry across experiments validate the robustness of the hybrid actuation approach.
>
---
#### [new 035] Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model
- **分类: cs.RO; cs.AI**

- **简介: 论文提出基于扩散模型的多头运动规划方法，解决自动驾驶中行为单一、缺乏指令响应能力的问题。通过预训练共享权重与GRPO微调，实现多样化策略生成，并结合LLM动态选择策略，提升规划灵活性与多样性。**

- **链接: [http://arxiv.org/pdf/2508.16947v1](http://arxiv.org/pdf/2508.16947v1)**

> **作者:** Fan Ding; Xuewen Luo; Hwa Hui Tew; Ruturaj Reddy; Xikun Wang; Junn Yong Loo
>
> **备注:** Has been submitted to AAAI 2026
>
> **摘要:** Recent advances in motion planning for autonomous driving have led to models capable of generating high-quality trajectories. However, most existing planners tend to fix their policy after supervised training, leading to consistent but rigid driving behaviors. This limits their ability to reflect human preferences or adapt to dynamic, instruction-driven demands. In this work, we propose a diffusion-based multi-head trajectory planner(M-diffusion planner). During the early training stage, all output heads share weights to learn to generate high-quality trajectories. Leveraging the probabilistic nature of diffusion models, we then apply Group Relative Policy Optimization (GRPO) to fine-tune the pre-trained model for diverse policy-specific behaviors. At inference time, we incorporate a large language model (LLM) to guide strategy selection, enabling dynamic, instruction-aware planning without switching models. Closed-loop simulation demonstrates that our post-trained planner retains strong planning capability while achieving state-of-the-art (SOTA) performance on the nuPlan val14 benchmark. Open-loop results further show that the generated trajectories exhibit clear diversity, effectively satisfying multi-modal driving behavior requirements. The code and related experiments will be released upon acceptance of the paper.
>
---
#### [new 036] A Dataset and Benchmark for Robotic Cloth Unfolding Grasp Selection: The ICRA 2024 Cloth Competition
- **分类: cs.RO**

- **简介: 论文聚焦于机器人布料展开中的抓取姿态选择任务，针对缺乏标准化基准和数据集的问题，构建了公开数据集与ICRA 2024竞赛，收集679次真实抓取演示，分析方法性能并揭示手工程方法的强表现与评估偏差。**

- **链接: [http://arxiv.org/pdf/2508.16749v1](http://arxiv.org/pdf/2508.16749v1)**

> **作者:** Victor-Louis De Gusseme; Thomas Lips; Remko Proesmans; Julius Hietala; Giwan Lee; Jiyoung Choi; Jeongil Choi; Geon Kim; Phayuth Yonrith; Domen Tabernik; Andrej Gams; Peter Nimac; Matej Urbas; Jon Muhovič; Danijel Skočaj; Matija Mavsar; Hyojeong Yu; Minseo Kwon; Young J. Kim; Yang Cong; Ronghan Chen; Yu Ren; Supeng Diao; Jiawei Weng; Jiayue Liu; Haoran Sun; Linhan Yang; Zeqing Zhang; Ning Guo; Lei Yang; Fang Wan; Chaoyang Song; Jia Pan; Yixiang Jin; Yong A; Jun Shi; Dingzhe Li; Yong Yang; Kakeru Yamasaki; Takumi Kajiwara; Yuki Nakadera; Krati Saxena; Tomohiro Shibata; Chongkun Xia; Kai Mo; Yanzhao Yu; Qihao Lin; Binqiang Ma; Uihun Sagong; JungHyun Choi; JeongHyun Park; Dongwoo Lee; Yeongmin Kim; Myun Joong Hwang; Yusuke Kuribayashi; Naoki Hiratsuka; Daisuke Tanaka; Solvi Arnold; Kimitoshi Yamazaki; Carlos Mateo-Agullo; Andreas Verleysen; Francis Wyffels
>
> **备注:** submitted to IJRR
>
> **摘要:** Robotic cloth manipulation suffers from a lack of standardized benchmarks and shared datasets for evaluating and comparing different approaches. To address this, we created a benchmark and organized the ICRA 2024 Cloth Competition, a unique head-to-head evaluation focused on grasp pose selection for in-air robotic cloth unfolding. Eleven diverse teams participated in the competition, utilizing our publicly released dataset of real-world robotic cloth unfolding attempts and a variety of methods to design their unfolding approaches. Afterwards, we also expanded our dataset with 176 competition evaluation trials, resulting in a dataset of 679 unfolding demonstrations across 34 garments. Analysis of the competition results revealed insights about the trade-off between grasp success and coverage, the surprisingly strong achievements of hand-engineered methods and a significant discrepancy between competition performance and prior work, underscoring the importance of independent, out-of-the-lab evaluation in robotic cloth manipulation. The associated dataset is a valuable resource for developing and evaluating grasp selection methods, particularly for learning-based approaches. We hope that our benchmark, dataset and competition results can serve as a foundation for future benchmarks and drive further progress in data-driven robotic cloth manipulation. The dataset and benchmarking code are available at https://airo.ugent.be/cloth_competition.
>
---
#### [new 037] Variational Shape Inference for Grasp Diffusion on SE(3)
- **分类: cs.RO**

- **简介: 论文提出基于变分形状推理的抓取扩散模型，用于机器人操作中的多模态抓取合成任务，解决几何特征学习鲁棒性差和点云稀疏问题，提升真实场景下的抓取成功率。**

- **链接: [http://arxiv.org/pdf/2508.17482v1](http://arxiv.org/pdf/2508.17482v1)**

> **作者:** S. Talha Bukhari; Kaivalya Agrawal; Zachary Kingston; Aniket Bera
>
> **摘要:** Grasp synthesis is a fundamental task in robotic manipulation which usually has multiple feasible solutions. Multimodal grasp synthesis seeks to generate diverse sets of stable grasps conditioned on object geometry, making the robust learning of geometric features crucial for success. To address this challenge, we propose a framework for learning multimodal grasp distributions that leverages variational shape inference to enhance robustness against shape noise and measurement sparsity. Our approach first trains a variational autoencoder for shape inference using implicit neural representations, and then uses these learned geometric features to guide a diffusion model for grasp synthesis on the SE(3) manifold. Additionally, we introduce a test-time grasp optimization technique that can be integrated as a plugin to further enhance grasping performance. Experimental results demonstrate that our shape inference for grasp synthesis formulation outperforms state-of-the-art multimodal grasp synthesis methods on the ACRONYM dataset by 6.3%, while demonstrating robustness to deterioration in point cloud density compared to other approaches. Furthermore, our trained model achieves zero-shot transfer to real-world manipulation of household objects, generating 34% more successful grasps than baselines despite measurement noise and point cloud calibration errors.
>
---
#### [new 038] Neural Algorithmic Reasoners informed Large Language Model for Multi-Agent Path Finding
- **分类: cs.AI; cs.RO**

- **简介: 论文提出LLM-NAR框架，用神经算法推理器（NAR）增强大语言模型（LLM）解决多智能体路径规划（MAPF）问题。通过图神经网络和交叉注意力机制融合地图信息，提升LLM在复杂协调任务中的表现。**

- **链接: [http://arxiv.org/pdf/2508.17971v1](http://arxiv.org/pdf/2508.17971v1)**

> **作者:** Pu Feng; Size Wang; Yuhong Cao; Junkang Liang; Rongye Shi; Wenjun Wu
>
> **备注:** Accepted by IJCNN 2025
>
> **摘要:** The development and application of large language models (LLM) have demonstrated that foundational models can be utilized to solve a wide array of tasks. However, their performance in multi-agent path finding (MAPF) tasks has been less than satisfactory, with only a few studies exploring this area. MAPF is a complex problem requiring both planning and multi-agent coordination. To improve the performance of LLM in MAPF tasks, we propose a novel framework, LLM-NAR, which leverages neural algorithmic reasoners (NAR) to inform LLM for MAPF. LLM-NAR consists of three key components: an LLM for MAPF, a pre-trained graph neural network-based NAR, and a cross-attention mechanism. This is the first work to propose using a neural algorithmic reasoner to integrate GNNs with the map information for MAPF, thereby guiding LLM to achieve superior performance. LLM-NAR can be easily adapted to various LLM models. Both simulation and real-world experiments demonstrate that our method significantly outperforms existing LLM-based approaches in solving MAPF problems.
>
---
#### [new 039] Physical Embodiment Enables Information Processing Beyond Explicit Sensing in Active Matter
- **分类: cond-mat.soft; cs.RO**

- **简介: 论文研究如何利用物理形态实现无需显式传感的信息处理，解决合成活性物质在未知流场中导航难题。通过强化学习控制自热泳粒子，发现其能利用自身动力学隐式感知环境并成功导航。**

- **链接: [http://arxiv.org/pdf/2508.17921v1](http://arxiv.org/pdf/2508.17921v1)**

> **作者:** Diptabrata Paul; Nikola Milosevic; Nico Scherf; Frank Cichos
>
> **摘要:** Living microorganisms have evolved dedicated sensory machinery to detect environmental perturbations, processing these signals through biochemical networks to guide behavior. Replicating such capabilities in synthetic active matter remains a fundamental challenge. Here, we demonstrate that synthetic active particles can adapt to hidden hydrodynamic perturbations through physical embodiment alone, without explicit sensing mechanisms. Using reinforcement learning to control self-thermophoretic particles, we show that they learn navigation strategies to counteract unobserved flow fields by exploiting information encoded in their physical dynamics. Remarkably, particles successfully navigate perturbations that are not included in their state inputs, revealing that embodied dynamics can serve as an implicit sensing mechanism. This discovery establishes physical embodiment as a computational resource for information processing in active matter, with implications for autonomous microrobotic systems and bio-inspired computation.
>
---
#### [new 040] Dimension-Decomposed Learning for Quadrotor Geometric Attitude Control with Almost Global Exponential Convergence on SO(3)
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **简介: 论文提出DiD-L方法用于四旋翼姿态控制中的干扰识别，通过维度分解将高维映射拆分为低维子任务，结合浅层神经网络与自适应律实现在线学习。无需持续激励条件和预训练，可在STM32上实时运行（400Hz），并证明几乎全局指数收敛。**

- **链接: [http://arxiv.org/pdf/2508.14422v1](http://arxiv.org/pdf/2508.14422v1)**

> **作者:** Tianhua Gao; Masashi Izumita; Kohji Tomita; Akiya Kamimura
>
> **摘要:** This paper introduces a lightweight and interpretable online learning approach called Dimension-Decomposed Learning (DiD-L) for disturbance identification in quadrotor geometric attitude control. As a module instance of DiD-L, we propose the Sliced Adaptive-Neuro Mapping (SANM). Specifically, to address underlying underfitting problems, the high-dimensional mapping for online identification is axially ``sliced" into multiple low-dimensional submappings (slices). In this way, the complex high-dimensional problem is decomposed into a set of simple low-dimensional subtasks addressed by shallow neural networks and adaptive laws. These neural networks and adaptive laws are updated online via Lyapunov-based adaptation without the persistent excitation (PE) condition. To enhance the interpretability of the proposed approach, we prove that the state solution of the rotational error dynamics exponentially converges into an arbitrarily small ball within an almost global attraction domain, despite time-varying disturbances and inertia uncertainties. This result is novel as it demonstrates exponential convergence without requiring pre-training for unseen disturbances and specific knowledge of the model. To our knowledge in the quadrotor control field, DiD-L is the first online learning approach that is lightweight enough to run in real-time at 400 Hz on microcontroller units (MCUs) such as STM32, and has been validated through real-world experiments.
>
---
#### [new 041] Fiducial Marker Splatting for High-Fidelity Robotics Simulations
- **分类: cs.CV; cs.RO**

- **简介: 论文提出一种融合高斯点绘与标记的仿真方法，用于提升复杂环境中机器人定位精度。针对传统网格表示和神经渲染难以结合 fiducial 标记的问题，设计高效生成GS标记的算法，在温室场景中验证其在视觉真实性和定位准确性上的优势。**

- **链接: [http://arxiv.org/pdf/2508.17012v1](http://arxiv.org/pdf/2508.17012v1)**

> **作者:** Diram Tabaa; Gianni Di Caro
>
> **摘要:** High-fidelity 3D simulation is critical for training mobile robots, but its traditional reliance on mesh-based representations often struggle in complex environments, such as densely packed greenhouses featuring occlusions and repetitive structures. Recent neural rendering methods, like Gaussian Splatting (GS), achieve remarkable visual realism but lack flexibility to incorporate fiducial markers, which are essential for robotic localization and control. We propose a hybrid framework that combines the photorealism of GS with structured marker representations. Our core contribution is a novel algorithm for efficiently generating GS-based fiducial markers (e.g., AprilTags) within cluttered scenes. Experiments show that our approach outperforms traditional image-fitting techniques in both efficiency and pose-estimation accuracy. We further demonstrate the framework's potential in a greenhouse simulation. This agricultural setting serves as a challenging testbed, as its combination of dense foliage, similar-looking elements, and occlusions pushes the limits of perception, thereby highlighting the framework's value for real-world applications.
>
---
#### [new 042] Collaborative-Online-Learning-Enabled Distributionally Robust Motion Control for Multi-Robot Systems
- **分类: math.OC; cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对多机器人系统在随机障碍物环境中的避障控制问题，提出基于协同在线学习（COOL）的分布鲁棒运动控制框架。通过分布式数据流提取障碍物运动分布信息，构建并压缩模糊集，实现概率安全轨迹生成与长期跟踪性能保障。**

- **链接: [http://arxiv.org/pdf/2508.17173v1](http://arxiv.org/pdf/2508.17173v1)**

> **作者:** Chao Ning; Han Wang; Longyan Li; Yang Shi
>
> **摘要:** This paper develops a novel COllaborative-Online-Learning (COOL)-enabled motion control framework for multi-robot systems to avoid collision amid randomly moving obstacles whose motion distributions are partially observable through decentralized data streams. To address the notable challenge of data acquisition due to occlusion, a COOL approach based on the Dirichlet process mixture model is proposed to efficiently extract motion distribution information by exchanging among robots selected learning structures. By leveraging the fine-grained local-moment information learned through COOL, a data-stream-driven ambiguity set for obstacle motion is constructed. We then introduce a novel ambiguity set propagation method, which theoretically admits the derivation of the ambiguity sets for obstacle positions over the entire prediction horizon by utilizing obstacle current positions and the ambiguity set for obstacle motion. Additionally, we develop a compression scheme with its safety guarantee to automatically adjust the complexity and granularity of the ambiguity set by aggregating basic ambiguity sets that are close in a measure space, thereby striking an attractive trade-off between control performance and computation time. Then the probabilistic collision-free trajectories are generated through distributionally robust optimization problems. The distributionally robust obstacle avoidance constraints based on the compressed ambiguity set are equivalently reformulated by deriving separating hyperplanes through tractable semi-definite programming. Finally, we establish the probabilistic collision avoidance guarantee and the long-term tracking performance guarantee for the proposed framework. The numerical simulations are used to demonstrate the efficacy and superiority of the proposed approach compared with state-of-the-art methods.
>
---
#### [new 043] M3DMap: Object-aware Multimodal 3D Mapping for Dynamic Environments
- **分类: cs.CV; cs.RO**

- **简介: 论文提出M3DMap，用于动态环境中物体感知的多模态三维地图构建。解决现有方法缺乏统一多模态表示的问题，通过模块化设计实现图像、点云等数据融合，提升地图精度与实用性。**

- **链接: [http://arxiv.org/pdf/2508.17044v1](http://arxiv.org/pdf/2508.17044v1)**

> **作者:** Dmitry Yudin
>
> **备注:** 29 pages, 3 figures, 13 tables. Preprint of the accepted article in Optical Memory and Neural Network Journal
>
> **摘要:** 3D mapping in dynamic environments poses a challenge for modern researchers in robotics and autonomous transportation. There are no universal representations for dynamic 3D scenes that incorporate multimodal data such as images, point clouds, and text. This article takes a step toward solving this problem. It proposes a taxonomy of methods for constructing multimodal 3D maps, classifying contemporary approaches based on scene types and representations, learning methods, and practical applications. Using this taxonomy, a brief structured analysis of recent methods is provided. The article also describes an original modular method called M3DMap, designed for object-aware construction of multimodal 3D maps for both static and dynamic scenes. It consists of several interconnected components: a neural multimodal object segmentation and tracking module; an odometry estimation module, including trainable algorithms; a module for 3D map construction and updating with various implementations depending on the desired scene representation; and a multimodal data retrieval module. The article highlights original implementations of these modules and their advantages in solving various practical tasks, from 3D object grounding to mobile manipulation. Additionally, it presents theoretical propositions demonstrating the positive effect of using multimodal data and modern foundational models in 3D mapping methods. Details of the taxonomy and method implementation are available at https://yuddim.github.io/M3DMap.
>
---
#### [new 044] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method
- **分类: cs.CV; cs.RO**

- **简介: 论文提出DeltaFlow，一种高效多帧场景流估计方法，解决传统方法忽略时序信息或计算成本高的问题。通过Δ方案提取时序特征，结合类别平衡损失和实例一致性损失，提升精度与泛化能力，在Argoverse 2和Waymo上表现最优。**

- **链接: [http://arxiv.org/pdf/2508.17054v1](http://arxiv.org/pdf/2508.17054v1)**

> **作者:** Qingwen Zhang; Xiaomeng Zhu; Yushan Zhang; Yixi Cai; Olov Andersson; Patric Jensfelt
>
> **备注:** 17 pages (9 main pages + 8 supp materail), 11 figures, code at https://github.com/Kin-Zhang/DeltaFlow
>
> **摘要:** Previous dominant methods for scene flow estimation focus mainly on input from two consecutive frames, neglecting valuable information in the temporal domain. While recent trends shift towards multi-frame reasoning, they suffer from rapidly escalating computational costs as the number of frames grows. To leverage temporal information more efficiently, we propose DeltaFlow ($\Delta$Flow), a lightweight 3D framework that captures motion cues via a $\Delta$ scheme, extracting temporal features with minimal computational cost, regardless of the number of frames. Additionally, scene flow estimation faces challenges such as imbalanced object class distributions and motion inconsistency. To tackle these issues, we introduce a Category-Balanced Loss to enhance learning across underrepresented classes and an Instance Consistency Loss to enforce coherent object motion, improving flow accuracy. Extensive evaluations on the Argoverse 2 and Waymo datasets show that $\Delta$Flow achieves state-of-the-art performance with up to 22% lower error and $2\times$ faster inference compared to the next-best multi-frame supervised method, while also demonstrating a strong cross-domain generalization ability. The code is open-sourced at https://github.com/Kin-Zhang/DeltaFlow along with trained model weights.
>
---
#### [new 045] A Synthetic Dataset for Manometry Recognition in Robotic Applications
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 论文提出一种混合数据合成方法，用于解决工业场景下真实数据稀缺与采集成本高的问题。通过BlenderProc生成带精确标注的逼真图像，并结合NVIDIA Cosmos-Predict2模型合成视频序列，最终在YOLO检测网络上验证了合成数据的有效性，证明1:1混合数据可提升检测性能。**

- **链接: [http://arxiv.org/pdf/2508.17468v1](http://arxiv.org/pdf/2508.17468v1)**

> **作者:** Pedro Antonio Rabelo Saraiva; Enzo Ferreira de Souza; Joao Manoel Herrera Pinheiro; Thiago H. Segreto; Ricardo V. Godoy; Marcelo Becker
>
> **摘要:** This work addresses the challenges of data scarcity and high acquisition costs for training robust object detection models in complex industrial environments, such as offshore oil platforms. The practical and economic barriers to collecting real-world data in these hazardous settings often hamper the development of autonomous inspection systems. To overcome this, in this work we propose and validate a hybrid data synthesis pipeline that combines procedural rendering with AI-driven video generation. Our methodology leverages BlenderProc to create photorealistic images with precise annotations and controlled domain randomization, and integrates NVIDIA's Cosmos-Predict2 world-foundation model to synthesize physically plausible video sequences with temporal diversity, capturing rare viewpoints and adverse conditions. We demonstrate that a YOLO-based detection network trained on a composite dataset, blending real images with our synthetic data, achieves superior performance compared to models trained exclusively on real-world data. Notably, a 1:1 mixture of real and synthetic data yielded the highest accuracy, surpassing the real-only baseline. These findings highlight the viability of a synthetic-first approach as an efficient, cost-effective, and safe alternative for developing reliable perception systems in safety-critical and resource-constrained industrial applications.
>
---
#### [new 046] Social Identity in Human-Agent Interaction: A Primer
- **分类: physics.soc-ph; cs.AI; cs.CY; cs.HC; cs.RO**

- **简介: 论文探讨社会身份理论在人机交互中的应用，旨在解决人工智能代理如何参与社会互动的问题。作者通过案例和设想说明SIT与SCT的适用性，并呼吁研究者保持批判视角。**

- **链接: [http://arxiv.org/pdf/2508.16609v1](http://arxiv.org/pdf/2508.16609v1)**

> **作者:** Katie Seaborn
>
> **备注:** 28 pages
>
> **摘要:** Social identity theory (SIT) and social categorization theory (SCT) are two facets of the social identity approach (SIA) to understanding social phenomena. SIT and SCT are models that describe and explain how people interact with one another socially, connecting the individual to the group through an understanding of underlying psychological mechanisms and intergroup behaviour. SIT, originally developed in the 1970s, and SCT, a later, more general offshoot, have been broadly applied to a range of social phenomena among people. The rise of increasingly social machines embedded in daily life has spurned efforts on understanding whether and how artificial agents can and do participate in SIA activities. As agents like social robots and chatbots powered by sophisticated large language models (LLMs) advance, understanding the real and potential roles of these technologies as social entities is crucial. Here, I provide a primer on SIA and extrapolate, through case studies and imagined examples, how SIT and SCT can apply to artificial social agents. I emphasize that not all human models and sub-theories will apply. I further argue that, given the emerging competence of these machines and our tendency to be taken in by them, we experts may need to don the hat of the uncanny killjoy, for our own good.
>
---
#### [new 047] SoK: Cybersecurity Assessment of Humanoid Ecosystem
- **分类: cs.CR; cs.RO**

- **简介: 该论文属于安全评估任务，旨在解决人形机器人因软硬件复杂性导致的跨层安全风险问题。作者构建七层安全模型与39×35攻击-防御矩阵，量化评估三款机器人安全成熟度，提供系统化评估方法与优先级指导。**

- **链接: [http://arxiv.org/pdf/2508.17481v1](http://arxiv.org/pdf/2508.17481v1)**

> **作者:** Priyanka Prakash Surve; Asaf Shabtai; Yuval Elovici
>
> **摘要:** Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics.
>
---
#### [new 048] BirdRecorder's AI on Sky: Safeguarding birds of prey by detection and classification of tiny objects around wind turbines
- **分类: cs.CV; cs.LG; cs.RO; cs.SY; eess.SY**

- **简介: 论文提出BirdRecorder系统，用于检测和分类风力发电机附近的猛禽，解决风电发展与鸟类保护冲突问题。通过AI算法与优化软硬件结合，实现实时精准识别，减少鸟类碰撞风险。**

- **链接: [http://arxiv.org/pdf/2508.18136v1](http://arxiv.org/pdf/2508.18136v1)**

> **作者:** Nico Klar; Nizam Gifary; Felix P. G. Ziegler; Frank Sehnke; Anton Kaifel; Eric Price; Aamir Ahmad
>
> **备注:** 18 pages, 1 figures, to appear in Proceedings of the 19th International Conference on Intelligent Autonomous Systems (IAS-19), Genoa, Italy, 2025
>
> **摘要:** The urgent need for renewable energy expansion, particularly wind power, is hindered by conflicts with wildlife conservation. To address this, we developed BirdRecorder, an advanced AI-based anti-collision system to protect endangered birds, especially the red kite (Milvus milvus). Integrating robotics, telemetry, and high-performance AI algorithms, BirdRecorder aims to detect, track, and classify avian species within a range of 800 m to minimize bird-turbine collisions. BirdRecorder integrates advanced AI methods with optimized hardware and software architectures to enable real-time image processing. Leveraging Single Shot Detector (SSD) for detection, combined with specialized hardware acceleration and tracking algorithms, our system achieves high detection precision while maintaining the speed necessary for real-time decision-making. By combining these components, BirdRecorder outperforms existing approaches in both accuracy and efficiency. In this paper, we summarize results on field tests and performance of the BirdRecorder system. By bridging the gap between renewable energy expansion and wildlife conservation, BirdRecorder contributes to a more sustainable coexistence of technology and nature.
>
---
#### [new 049] Observations of atypical users from a pilot deployment of a public-space social robot in a church
- **分类: cs.HC; cs.RO**

- **简介: 论文研究社交机器人在教堂等公共空间中的自然交互，针对真实环境中用户行为不可预测的问题，通过三日试点部署观察异常用户行为，提出改进策略与未来研究方向，助力社会机器人更有效地融入公共空间。**

- **链接: [http://arxiv.org/pdf/2508.16622v1](http://arxiv.org/pdf/2508.16622v1)**

> **作者:** Andrew Blair; Peggy Gregory; Mary Ellen Foster
>
> **备注:** Accepted at the workshop on Real-World HRI in Public and Private Spaces: Successes, Failures, and Lessons Learned (PubRob-Fails), held at the IEEE RO-MAN Conference, 2025
>
> **摘要:** Though a goal of HRI is the natural integration of social robots into everyday public spaces, real-world studies still occur mostly within controlled environments with predetermined participants. True public spaces present an environment which is largely unconstrained and unpredictable, frequented by a diverse range of people whose goals can often conflict with those of the robot. When combined with the general unfamiliarity most people have with social robots, this leads to unexpected human-robot interactions in these public spaces that are rarely discussed or detected in other contexts. In this paper, we describe atypical users we observed interacting with our robot, and those who did not, during a three-day pilot deployment within a large working church and visitor attraction. We then discuss theoretical future advances in the field that could address these challenges, as well as immediate practical mitigations and strategies to help improve public space human-robot interactions in the present. This work contributes empirical insights into the dynamics of human-robot interaction in public environments and offers actionable guidance for more effective future deployments for social robot designers.
>
---
#### [new 050] SEER-VAR: Semantic Egocentric Environment Reasoner for Vehicle Augmented Reality
- **分类: cs.CV; cs.RO**

- **简介: 论文提出SEER-VAR框架，解决车辆增强现实（AR）中场景动态分割与上下文感知推荐问题。通过深度引导的语义分解、双SLAM分支和LLM驱动推荐，实现精准空间对齐与自然AR渲染。**

- **链接: [http://arxiv.org/pdf/2508.17255v1](http://arxiv.org/pdf/2508.17255v1)**

> **作者:** Yuzhi Lai; Shenghai Yuan; Peizheng Li; Jun Lou; Andreas Zell
>
> **摘要:** We present SEER-VAR, a novel framework for egocentric vehicle-based augmented reality (AR) that unifies semantic decomposition, Context-Aware SLAM Branches (CASB), and LLM-driven recommendation. Unlike existing systems that assume static or single-view settings, SEER-VAR dynamically separates cabin and road scenes via depth-guided vision-language grounding. Two SLAM branches track egocentric motion in each context, while a GPT-based module generates context-aware overlays such as dashboard cues and hazard alerts. To support evaluation, we introduce EgoSLAM-Drive, a real-world dataset featuring synchronized egocentric views, 6DoF ground-truth poses, and AR annotations across diverse driving scenarios. Experiments demonstrate that SEER-VAR achieves robust spatial alignment and perceptually coherent AR rendering across varied environments. As one of the first to explore LLM-based AR recommendation in egocentric driving, we address the lack of comparable systems through structured prompting and detailed user studies. Results show that SEER-VAR enhances perceived scene understanding, overlay relevance, and driver ease, providing an effective foundation for future research in this direction. Code and dataset will be made open source.
>
---
#### [new 051] Robust Point Cloud Registration via Geometric Overlapping Guided Rotation Search
- **分类: cs.CV; cs.RO**

- **简介: 论文提出一种基于几何重叠引导的鲁棒点云配准方法，通过旋转轴的分支定界搜索与区间查询优化，实现高效高精度配准，解决高 outlier 比例下的配准难题。**

- **链接: [http://arxiv.org/pdf/2508.17427v1](http://arxiv.org/pdf/2508.17427v1)**

> **作者:** Zhao Zheng; Jingfan Fan; Long Shao; Hong Song; Danni Ai; Tianyu Fu; Deqiang Xiao; Yongtian Wang; Jian Yang
>
> **摘要:** Point cloud registration based on correspondences computes the rigid transformation that maximizes the number of inliers constrained within the noise threshold. Current state-of-the-art (SOTA) methods employing spatial compatibility graphs or branch-and-bound (BnB) search mainly focus on registration under high outlier ratios. However, graph-based methods require at least quadratic space and time complexity for graph construction, while multi-stage BnB search methods often suffer from inaccuracy due to local optima between decomposed stages. This paper proposes a geometric maximum overlapping registration framework via rotation-only BnB search. The rigid transformation is decomposed using Chasles' theorem into a translation along rotation axis and a 2D rigid transformation. The optimal rotation axis and angle are searched via BnB, with residual parameters formulated as range maximum query (RMQ) problems. Firstly, the top-k candidate rotation axes are searched within a hemisphere parameterized by cube mapping, and the translation along each axis is estimated through interval stabbing of the correspondences projected onto that axis. Secondly, the 2D registration is relaxed to 1D rotation angle search with 2D RMQ of geometric overlapping for axis-aligned rectangles, which is solved deterministically in polynomial time using sweep line algorithm with segment tree. Experimental results on 3DMatch, 3DLoMatch, and KITTI datasets demonstrate superior accuracy and efficiency over SOTA methods, while the time complexity is polynomial and the space complexity increases linearly with the number of points, even in the worst case.
>
---
## 更新

#### [replaced 001] RT-Cache: Training-Free Retrieval for Real-Time Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.09040v3](http://arxiv.org/pdf/2505.09040v3)**

> **作者:** Owen Kwon; Abraham George; Alison Bartsch; Amir Barati Farimani
>
> **备注:** 8 pages, 6 figures. 2025 IEEE-RAS 24th International Conference on Humanoid Robots
>
> **摘要:** Real robots are expected to repeat the same behavior in new environments with very little new data, yet modern controllers either incur heavy per-step inference or require deployment-time fine-tuning. We propose RT-Cache, a training-free retrieval-as-control pipeline that caches diverse image action trajectories in a unified vector memory and, at test time, embeds the current frame to retrieve and replay multi-step snippets, replacing per-step model calls. A hierarchical search keeps lookups sub-second at million scale, shifting cost from compute to storage and enabling real-time control on modest GPUs. Across real-robot tasks and large open logs, RT-Cache achieves higher success and lower completion time than strong retrieval baselines (approximately x2 higher success and ~30% faster in our settings), and a single-episode anchoring study shows immediate adaptation to a more complex, contact-rich task without fine-tuning. RT-Cache turns experience into an append-only memory, offering a simple, scalable path to few-shot deployment today and a foundation for multimodal keys and optional integration with high-level policies. Project page: https://rt-cache.github.io/.
>
---
#### [replaced 002] HOSt3R: Keypoint-free Hand-Object 3D Reconstruction from RGB images
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.16465v2](http://arxiv.org/pdf/2508.16465v2)**

> **作者:** Anilkumar Swamy; Vincent Leroy; Philippe Weinzaepfel; Jean-Sébastien Franco; Grégory Rogez
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Hand-object 3D reconstruction has become increasingly important for applications in human-robot interaction and immersive AR/VR experiences. A common approach for object-agnostic hand-object reconstruction from RGB sequences involves a two-stage pipeline: hand-object 3D tracking followed by multi-view 3D reconstruction. However, existing methods rely on keypoint detection techniques, such as Structure from Motion (SfM) and hand-keypoint optimization, which struggle with diverse object geometries, weak textures, and mutual hand-object occlusions, limiting scalability and generalization. As a key enabler to generic and seamless, non-intrusive applicability, we propose in this work a robust, keypoint detector-free approach to estimating hand-object 3D transformations from monocular motion video/images. We further integrate this with a multi-view reconstruction pipeline to accurately recover hand-object 3D shape. Our method, named HOSt3R, is unconstrained, does not rely on pre-scanned object templates or camera intrinsics, and reaches state-of-the-art performance for the tasks of object-agnostic hand-object 3D transformation and shape estimation on the SHOWMe benchmark. We also experiment on sequences from the HO3D dataset, demonstrating generalization to unseen object categories.
>
---
#### [replaced 003] BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05652v2](http://arxiv.org/pdf/2503.05652v2)**

> **作者:** Yunfan Jiang; Ruohan Zhang; Josiah Wong; Chen Wang; Yanjie Ze; Hang Yin; Cem Gokmen; Shuran Song; Jiajun Wu; Li Fei-Fei
>
> **备注:** 9th Conference on Robot Learning (CoRL 2025), Seoul, Korea. Project website: https://behavior-robot-suite.github.io/
>
> **摘要:** Real-world household tasks present significant challenges for mobile manipulation robots. An analysis of existing robotics benchmarks reveals that successful task performance hinges on three key whole-body control capabilities: bimanual coordination, stable and precise navigation, and extensive end-effector reachability. Achieving these capabilities requires careful hardware design, but the resulting system complexity further complicates visuomotor policy learning. To address these challenges, we introduce the BEHAVIOR Robot Suite (BRS), a comprehensive framework for whole-body manipulation in diverse household tasks. Built on a bimanual, wheeled robot with a 4-DoF torso, BRS integrates a cost-effective whole-body teleoperation interface for data collection and a novel algorithm for learning whole-body visuomotor policies. We evaluate BRS on five challenging household tasks that not only emphasize the three core capabilities but also introduce additional complexities, such as long-range navigation, interaction with articulated and deformable objects, and manipulation in confined spaces. We believe that BRS's integrated robotic embodiment, data collection interface, and learning framework mark a significant step toward enabling real-world whole-body manipulation for everyday household tasks. BRS is open-sourced at https://behavior-robot-suite.github.io/
>
---
#### [replaced 004] Sim-to-Real Transfer of Deep Reinforcement Learning Agents for Online Coverage Path Planning
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2406.04920v3](http://arxiv.org/pdf/2406.04920v3)**

> **作者:** Arvi Jonnarth; Ola Johansson; Jie Zhao; Michael Felsberg
>
> **备注:** Published in IEEE Access
>
> **摘要:** Coverage path planning (CPP) is the problem of finding a path that covers the entire free space of a confined area, with applications ranging from robotic lawn mowing to search-and-rescue. While for known environments, offline methods can find provably complete paths, and in some cases optimal solutions, unknown environments need to be planned online during mapping. We investigate the suitability of continuous-space reinforcement learning (RL) for this challenging problem, and propose a computationally feasible egocentric map representation based on frontiers, as well as a novel reward term based on total variation to promote complete coverage. Compared to existing classical methods, this approach allows for a flexible path space, and enables the agent to adapt to specific environment characteristics. Meanwhile, the deployment of RL models on real robot systems is difficult. Training from scratch may be infeasible due to slow convergence times, while transferring from simulation to reality, i.e. sim-to-real transfer, is a key challenge in itself. We bridge the sim-to-real gap through a semi-virtual environment, including a real robot and real-time aspects, while utilizing a simulated sensor and obstacles to enable environment randomization and automated episode resetting. We investigate what level of fine-tuning is needed for adapting to a realistic setting. Through extensive experiments, we show that our approach surpasses the performance of both previous RL-based approaches and highly specialized methods across multiple CPP variations in simulation. Meanwhile, our method successfully transfers to a real robot. Our code implementation can be found online.
>
---
#### [replaced 005] Practical Equivalence Testing and Its Application in Synthetic Pre-Crash Scenario Validation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12827v3](http://arxiv.org/pdf/2505.12827v3)**

> **作者:** Jian Wu; Ulrich Sander; Carol Flannagan; Minxiang Zhao; Jonas Bärgman
>
> **摘要:** The use of representative pre-crash scenarios is critical for assessing the safety impact of driving automation systems through simulation. However, a gap remains in the robust evaluation of the similarity between synthetic and real-world pre-crash scenarios and their crash characteristics. Without proper validation, it cannot be ensured that the synthetic test scenarios adequately represent real-world driving behaviors and crash characteristics. One reason for this validation gap is the lack of focus on methods to confirm that the synthetic test scenarios are practically equivalent to real-world ones, given the assessment scope. Traditional statistical methods, like significance testing, focus on detecting differences rather than establishing equivalence; since failure to detect a difference does not imply equivalence, they are of limited applicability for validating synthetic pre-crash scenarios and crash characteristics. This study addresses this gap by proposing an equivalence testing method based on the Bayesian Region of Practical Equivalence (ROPE) framework. This method is designed to assess the practical equivalence of scenario characteristics that are most relevant for the intended assessment, making it particularly appropriate for the domain of virtual safety assessments. We first review existing equivalence testing methods. Then we propose and demonstrate the Bayesian ROPE-based method by testing the equivalence of two rear-end pre-crash datasets. Our approach focuses on the most relevant scenario characteristics. Our analysis provides insights into the practicalities and effectiveness of equivalence testing in synthetic test scenario validation and demonstrates the importance of testing for improving the credibility of synthetic data for automated vehicle safety assessment, as well as the credibility of subsequent safety impact assessments.
>
---
#### [replaced 006] MapleGrasp: Mask-guided Feature Pooling for Language-driven Efficient Robotic Grasping
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.06535v3](http://arxiv.org/pdf/2506.06535v3)**

> **作者:** Vineet Bhat; Naman Patel; Prashanth Krishnamurthy; Ramesh Karri; Farshad Khorrami
>
> **摘要:** Robotic manipulation of unseen objects via natural language commands remains challenging. Language driven robotic grasping (LDRG) predicts stable grasp poses from natural language queries and RGB-D images. We propose MapleGrasp, a novel framework that leverages mask-guided feature pooling for efficient vision-language driven grasping. Our two-stage training first predicts segmentation masks from CLIP-based vision-language features. The second stage pools features within these masks to generate pixel-level grasp predictions, improving efficiency, and reducing computation. Incorporating mask pooling results in a 7% improvement over prior approaches on the OCID-VLG benchmark. Furthermore, we introduce RefGraspNet, an open-source dataset eight times larger than existing alternatives, significantly enhancing model generalization for open-vocabulary grasping. MapleGrasp scores a strong grasping accuracy of 89\% when compared with competing methods in the RefGraspNet benchmark. Our method achieves comparable performance to larger Vision-Language-Action models on the LIBERO benchmark, and shows significantly better generalization to unseen tasks. Real-world experiments on a Franka arm demonstrate 73% success rate with unseen objects, surpassing competitive baselines by 11%. Code is provided in our github repository.
>
---
#### [replaced 007] PixRO: Pixel-Distributed Rotational Odometry with Gaussian Belief Propagation
- **分类: cs.CV; cs.DC; cs.MA; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2406.09726v2](http://arxiv.org/pdf/2406.09726v2)**

> **作者:** Ignacio Alzugaray; Riku Murai; Andrew Davison
>
> **摘要:** Images are the standard input for most computer vision algorithms. However, their processing often reduces to parallelizable operations applied locally and independently to individual pixels. Yet, many of these low-level raw pixel readings only provide redundant or noisy information for specific high-level tasks, leading to inefficiencies in both energy consumption during their transmission off-sensor and computational resources in their subsequent processing. As novel sensors featuring advanced in-pixel processing capabilities emerge, we envision a paradigm shift toward performing increasingly complex visual processing directly in-pixel, reducing computational overhead downstream. We advocate for synthesizing high-level cues at the pixel level, enabling their off-sensor transmission to directly support downstream tasks more effectively than raw pixel readings. This paper conceptualizes a novel photometric rotation estimation algorithm to be distributed at pixel level, where each pixel estimates the global motion of the camera by exchanging information with other pixels to achieve global consensus. We employ a probabilistic formulation and leverage Gaussian Belief Propagation (GBP) for decentralized inference using messaging-passing. The proposed proposed technique is evaluated on real-world public datasets and we offer a in-depth analysis of the practicality of applying GBP to distributed rotation estimation at pixel level.
>
---
#### [replaced 008] VIN-NBV: A View Introspection Network for Next-Best-View Selection
- **分类: cs.CV; cs.RO; I.2.10; I.2.9**

- **链接: [http://arxiv.org/pdf/2505.06219v3](http://arxiv.org/pdf/2505.06219v3)**

> **作者:** Noah Frahm; Dongxu Zhao; Andrea Dunn Beltran; Ron Alterovitz; Jan-Michael Frahm; Junier Oliva; Roni Sengupta
>
> **备注:** 9 pages, 9 figures, 2 tables. Reformat into two column. Additional experiments and results
>
> **摘要:** Next Best View (NBV) algorithms aim to maximize 3D scene acquisition quality using minimal resources, e.g. number of acquisitions, time taken, or distance traversed. Prior methods often rely on coverage maximization as a proxy for reconstruction quality, but for complex scenes with occlusions and finer details, this is not always sufficient and leads to poor reconstructions. Our key insight is to train an acquisition policy that directly optimizes for reconstruction quality rather than just coverage. To achieve this, we introduce the View Introspection Network (VIN): a lightweight neural network that predicts the Relative Reconstruction Improvement (RRI) of a potential next viewpoint without making any new acquisitions. We use this network to power a simple, yet effective, sequential samplingbased greedy NBV policy. Our approach, VIN-NBV, generalizes to unseen object categories, operates without prior scene knowledge, is adaptable to resource constraints, and can handle occlusions. We show that our RRI fitness criterion leads to a ~30% gain in reconstruction quality over a coverage-based criterion using the same greedy strategy. Furthermore, VIN-NBV also outperforms deep reinforcement learning methods, Scan-RL and GenNBV, by ~40%.
>
---
#### [replaced 009] AirExo-2: Scaling up Generalizable Robotic Imitation Learning with Low-Cost Exoskeletons
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03081v3](http://arxiv.org/pdf/2503.03081v3)**

> **作者:** Hongjie Fang; Chenxi Wang; Yiming Wang; Jingjing Chen; Shangning Xia; Jun Lv; Zihao He; Xiyan Yi; Yunhan Guo; Xinyu Zhan; Lixin Yang; Weiming Wang; Cewu Lu; Hao-Shu Fang
>
> **备注:** accepted to CoRL 2025
>
> **摘要:** Scaling up robotic imitation learning for real-world applications requires efficient and scalable demonstration collection methods. While teleoperation is effective, it depends on costly and inflexible robot platforms. In-the-wild demonstrations offer a promising alternative, but existing collection devices have key limitations: handheld setups offer limited observational coverage, and whole-body systems often require fine-tuning with robot data due to domain gaps. To address these challenges, we present AirExo-2, a low-cost exoskeleton system for large-scale in-the-wild data collection, along with several adaptors that transform collected data into pseudo-robot demonstrations suitable for policy learning. We further introduce RISE-2, a generalizable imitation learning policy that fuses 3D spatial and 2D semantic perception for robust manipulations. Experiments show that RISE-2 outperforms prior state-of-the-art methods on both in-domain and generalization evaluations. Trained solely on adapted in-the-wild data produced by AirExo-2, the RISE-2 policy achieves comparable performance to the policy trained with teleoperated data, highlighting the effectiveness and potential of AirExo-2 for scalable and generalizable imitation learning.
>
---
#### [replaced 010] A Photorealistic Dataset and Vision-Based Algorithm for Anomaly Detection During Proximity Operations in Lunar Orbit
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.20435v4](http://arxiv.org/pdf/2409.20435v4)**

> **作者:** Selina Leveugle; Chang Won Lee; Svetlana Stolpner; Chris Langley; Paul Grouchy; Steven Waslander; Jonathan Kelly
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** NASA's forthcoming Lunar Gateway space station, which will be uncrewed most of the time, will need to operate with an unprecedented level of autonomy. One key challenge is enabling the Canadarm3, the Gateway's external robotic system, to detect hazards in its environment using its onboard inspection cameras. This task is complicated by the extreme and variable lighting conditions in space. In this paper, we introduce the visual anomaly detection and localization task for the space domain and establish a benchmark based on a synthetic dataset called ALLO (Anomaly Localization in Lunar Orbit). We show that state-of-the-art visual anomaly detection methods often fail in the space domain, motivating the need for new approaches. To address this, we propose MRAD (Model Reference Anomaly Detection), a statistical algorithm that leverages the known pose of the Canadarm3 and a CAD model of the Gateway to generate reference images of the expected scene appearance. Anomalies are then identified as deviations from this model-generated reference. On the ALLO dataset, MRAD surpasses state-of-the-art anomaly detection algorithms, achieving an AP score of 62.1% at the pixel level and an AUROC score of 74.9% at the image level. Given the low tolerance for risk in space operations and the lack of domain-specific data, we emphasize the need for novel, robust, and accurate anomaly detection methods to handle the challenging visual conditions found in lunar orbit and beyond.
>
---
#### [replaced 011] Large Language Model-Driven Closed-Loop UAV Operation with Semantic Observations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.01930v4](http://arxiv.org/pdf/2507.01930v4)**

> **作者:** Wenhao Wang; Yanyan Li; Long Jiao; Jiawei Yuan
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Recent advances in large Language Models (LLMs) have revolutionized mobile robots, including unmanned aerial vehicles (UAVs), enabling their intelligent operation within Internet of Things (IoT) ecosystems. However, LLMs still face challenges from logical reasoning and complex decision-making, leading to concerns about the reliability of LLM-driven UAV operations in IoT applications. In this paper, we propose a closed-loop LLM-driven UAV operation code generation framework that enables reliable UAV operations powered by effective feedback and refinement using two LLM modules, i.e., a Code Generator and an Evaluator. Our framework transforms numerical state observations from UAV operations into semantic trajectory descriptions to enhance the evaluator LLM's understanding of UAV dynamics for precise feedback generation. Our framework also enables a simulation-based refinement process, and hence eliminates the risks to physical UAVs caused by incorrect code execution during the refinement. Extensive experiments on UAV control tasks with different complexities are conducted. The experimental results show that our framework can achieve reliable UAV operations using LLMs, which significantly outperforms baseline methods in terms of success rate and completeness with the increase of task complexity.
>
---
#### [replaced 012] AutoMisty: A Multi-Agent LLM Framework for Automated Code Generation in the Misty Social Robot
- **分类: cs.RO; cs.AI; cs.HC; cs.MA**

- **链接: [http://arxiv.org/pdf/2503.06791v2](http://arxiv.org/pdf/2503.06791v2)**

> **作者:** Xiao Wang; Lu Dong; Sahana Rangasrinivasan; Ifeoma Nwogu; Srirangaraj Setlur; Venugopal Govindaraju
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** The social robot's open API allows users to customize open-domain interactions. However, it remains inaccessible to those without programming experience. In this work, we introduce AutoMisty, the first multi-agent collaboration framework powered by large language models (LLMs), to enable the seamless generation of executable Misty robot code from natural language instructions. AutoMisty incorporates four specialized agent modules to manage task decomposition, assignment, problem-solving, and result synthesis. Each agent incorporates a two-layer optimization mechanism, with self-reflection for iterative refinement and human-in-the-loop for better alignment with user preferences. AutoMisty ensures a transparent reasoning process, allowing users to iteratively refine tasks through natural language feedback for precise execution. To evaluate AutoMisty's effectiveness, we designed a benchmark task set spanning four levels of complexity and conducted experiments in a real Misty robot environment. Extensive evaluations demonstrate that AutoMisty not only consistently generates high-quality code but also enables precise code control, significantly outperforming direct reasoning with ChatGPT-4o and ChatGPT-o1. All code, optimized APIs, and experimental videos will be publicly released through the webpage: https://wangxiaoshawn.github.io/AutoMisty.html
>
---
#### [replaced 013] Using Visual Anomaly Detection for Task Execution Monitoring
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2107.14206v2](http://arxiv.org/pdf/2107.14206v2)**

> **作者:** Santosh Thoduka; Juergen Gall; Paul G. Plöger
>
> **备注:** Accepted for publication at the 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Execution monitoring is essential for robots to detect and respond to failures. Since it is impossible to enumerate all failures for a given task, we learn from successful executions of the task to detect visual anomalies during runtime. Our method learns to predict the motions that occur during the nominal execution of a task, including camera and robot body motion. A probabilistic U-Net architecture is used to learn to predict optical flow, and the robot's kinematics and 3D model are used to model camera and body motion. The errors between the observed and predicted motion are used to calculate an anomaly score. We evaluate our method on a dataset of a robot placing a book on a shelf, which includes anomalies such as falling books, camera occlusions, and robot disturbances. We find that modeling camera and body motion, in addition to the learning-based optical flow prediction, results in an improvement of the area under the receiver operating characteristic curve from 0.752 to 0.804, and the area under the precision-recall curve from 0.467 to 0.549.
>
---
#### [replaced 014] ToddlerBot: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.00893v3](http://arxiv.org/pdf/2502.00893v3)**

> **作者:** Haochen Shi; Weizhuo Wang; Shuran Song; C. Karen Liu
>
> **备注:** Project website: https://toddlerbot.github.io/
>
> **摘要:** Learning-based robotics research driven by data demands a new approach to robot hardware design-one that serves as both a platform for policy execution and a tool for embodied data collection to train policies. We introduce ToddlerBot, a low-cost, open-source humanoid robot platform designed for scalable policy learning and research in robotics and AI. ToddlerBot enables seamless acquisition of high-quality simulation and real-world data. The plug-and-play zero-point calibration and transferable motor system identification ensure a high-fidelity digital twin, enabling zero-shot policy transfer from simulation to the real world. A user-friendly teleoperation interface facilitates streamlined real-world data collection for learning motor skills from human demonstrations. Utilizing its data collection ability and anthropomorphic design, ToddlerBot is an ideal platform to perform whole-body loco-manipulation. Additionally, ToddlerBot's compact size (0.56m, 3.4kg) ensures safe operation in real-world environments. Reproducibility is achieved with an entirely 3D-printed, open-source design and commercially available components, keeping the total cost under 6,000 USD. Comprehensive documentation allows assembly and maintenance with basic technical expertise, as validated by a successful independent replication of the system. We demonstrate ToddlerBot's capabilities through arm span, payload, endurance tests, loco-manipulation tasks, and a collaborative long-horizon scenario where two robots tidy a toy session together. By advancing ML-compatibility, capability, and reproducibility, ToddlerBot provides a robust platform for scalable learning and dynamic policy execution in robotics research.
>
---
#### [replaced 015] GraphCoT-VLA: A 3D Spatial-Aware Reasoning Vision-Language-Action Model for Robotic Manipulation with Ambiguous Instructions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.07650v2](http://arxiv.org/pdf/2508.07650v2)**

> **作者:** Helong Huang; Min Cen; Kai Tan; Xingyue Quan; Guowei Huang; Hong Zhang
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Vision-language-action models have emerged as a crucial paradigm in robotic manipulation. However, existing VLA models exhibit notable limitations in handling ambiguous language instructions and unknown environmental states. Furthermore, their perception is largely constrained to static two-dimensional observations, lacking the capability to model three-dimensional interactions between the robot and its environment. To address these challenges, this paper proposes GraphCoT-VLA, an efficient end-to-end model. To enhance the model's ability to interpret ambiguous instructions and improve task planning, we design a structured Chain-of-Thought reasoning module that integrates high-level task understanding and planning, failed task feedback, and low-level imaginative reasoning about future object positions and robot actions. Additionally, we construct a real-time updatable 3D Pose-Object graph, which captures the spatial configuration of robot joints and the topological relationships between objects in 3D space, enabling the model to better understand and manipulate their interactions. We further integrates a dropout hybrid reasoning strategy to achieve efficient control outputs. Experimental results across multiple real-world robotic tasks demonstrate that GraphCoT-VLA significantly outperforms existing methods in terms of task success rate and response speed, exhibiting strong generalization and robustness in open environments and under uncertain instructions.
>
---
#### [replaced 016] 3D Feature Distillation with Object-Centric Priors
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.18742v5](http://arxiv.org/pdf/2406.18742v5)**

> **作者:** Georgios Tziafas; Yucheng Xu; Zhibin Li; Hamidreza Kasaei
>
> **摘要:** Grounding natural language to the physical world is a ubiquitous topic with a wide range of applications in computer vision and robotics. Recently, 2D vision-language models such as CLIP have been widely popularized, due to their impressive capabilities for open-vocabulary grounding in 2D images. Recent works aim to elevate 2D CLIP features to 3D via feature distillation, but either learn neural fields that are scene-specific and hence lack generalization, or focus on indoor room scan data that require access to multiple camera views, which is not practical in robot manipulation scenarios. Additionally, related methods typically fuse features at pixel-level and assume that all camera views are equally informative. In this work, we show that this approach leads to sub-optimal 3D features, both in terms of grounding accuracy, as well as segmentation crispness. To alleviate this, we propose a multi-view feature fusion strategy that employs object-centric priors to eliminate uninformative views based on semantic information, and fuse features at object-level via instance segmentation masks. To distill our object-centric 3D features, we generate a large-scale synthetic multi-view dataset of cluttered tabletop scenes, spawning 15k scenes from over 3300 unique object instances, which we make publicly available. We show that our method reconstructs 3D CLIP features with improved grounding capacity and spatial consistency, while doing so from single-view RGB-D, thus departing from the assumption of multiple camera views at test time. Finally, we show that our approach can generalize to novel tabletop domains and be re-purposed for 3D instance segmentation without fine-tuning, and demonstrate its utility for language-guided robotic grasping in clutter.
>
---
#### [replaced 017] Locomotion on Constrained Footholds via Layered Architectures and Model Predictive Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.09979v2](http://arxiv.org/pdf/2506.09979v2)**

> **作者:** Zachary Olkin; Aaron D. Ames
>
> **备注:** Accepted to Humanoids 2025
>
> **摘要:** Computing stabilizing and optimal control actions for legged locomotion in real time is difficult due to the nonlinear, hybrid, and high dimensional nature of these robots. The hybrid nature of the system introduces a combination of discrete and continuous variables which causes issues for numerical optimal control. To address these challenges, we propose a layered architecture that separates the choice of discrete variables and a smooth Model Predictive Controller (MPC). The layered formulation allows for online flexibility and optimality without sacrificing real-time performance through a combination of gradient-free and gradient-based methods. The architecture leverages a sampling-based method for determining discrete variables, and a classical smooth MPC formulation using these fixed discrete variables. We demonstrate the results on a quadrupedal robot stepping over gaps and onto terrain with varying heights. In simulation, we demonstrate the controller on a humanoid robot for gap traversal. The layered approach is shown to be more optimal and reliable than common heuristic-based approaches and faster to compute than pure sampling methods.
>
---
#### [replaced 018] CleverDistiller: Simple and Spatially Consistent Cross-modal Distillation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.09878v2](http://arxiv.org/pdf/2503.09878v2)**

> **作者:** Hariprasath Govindarajan; Maciej K. Wozniak; Marvin Klingner; Camille Maurice; B Ravi Kiran; Senthil Yogamani
>
> **备注:** Accepted to BMVC 2025
>
> **摘要:** Vision foundation models (VFMs) such as DINO have led to a paradigm shift in 2D camera-based perception towards extracting generalized features to support many downstream tasks. Recent works introduce self-supervised cross-modal knowledge distillation (KD) as a way to transfer these powerful generalization capabilities into 3D LiDAR-based models. However, they either rely on highly complex distillation losses, pseudo-semantic maps, or limit KD to features useful for semantic segmentation only. In this work, we propose CleverDistiller, a self-supervised, cross-modal 2D-to-3D KD framework introducing a set of simple yet effective design choices: Unlike contrastive approaches relying on complex loss design choices, our method employs a direct feature similarity loss in combination with a multi layer perceptron (MLP) projection head to allow the 3D network to learn complex semantic dependencies throughout the projection. Crucially, our approach does not depend on pseudo-semantic maps, allowing for direct knowledge transfer from a VFM without explicit semantic supervision. Additionally, we introduce the auxiliary self-supervised spatial task of occupancy prediction to enhance the semantic knowledge, obtained from a VFM through KD, with 3D spatial reasoning capabilities. Experiments on standard autonomous driving benchmarks for 2D-to-3D KD demonstrate that CleverDistiller achieves state-of-the-art performance in both semantic segmentation and 3D object detection (3DOD) by up to 10% mIoU, especially when fine tuning on really low data amounts, showing the effectiveness of our simple yet powerful KD strategy
>
---
#### [replaced 019] A Multimodal Handover Failure Detection Dataset and Baselines
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2402.18319v2](http://arxiv.org/pdf/2402.18319v2)**

> **作者:** Santosh Thoduka; Nico Hochgeschwender; Juergen Gall; Paul G. Plöger
>
> **备注:** Accepted at ICRA 2024
>
> **摘要:** An object handover between a robot and a human is a coordinated action which is prone to failure for reasons such as miscommunication, incorrect actions and unexpected object properties. Existing works on handover failure detection and prevention focus on preventing failures due to object slip or external disturbances. However, there is a lack of datasets and evaluation methods that consider unpreventable failures caused by the human participant. To address this deficit, we present the multimodal Handover Failure Detection dataset, which consists of failures induced by the human participant, such as ignoring the robot or not releasing the object. We also present two baseline methods for handover failure detection: (i) a video classification method using 3D CNNs and (ii) a temporal action segmentation approach which jointly classifies the human action, robot action and overall outcome of the action. The results show that video is an important modality, but using force-torque data and gripper position help improve failure detection and action segmentation accuracy.
>
---
#### [replaced 020] FetchBot: Learning Generalizable Object Fetching in Cluttered Scenes via Zero-Shot Sim2Real
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.17894v2](http://arxiv.org/pdf/2502.17894v2)**

> **作者:** Weiheng Liu; Yuxuan Wan; Jilong Wang; Yuxuan Kuang; Wenbo Cui; Xuesong Shi; Haoran Li; Dongbin Zhao; Zhizheng Zhang; He Wang
>
> **备注:** 9th Annual Conference on Robot Learning (CoRL 2025, Oral)
>
> **摘要:** Generalizable object fetching in cluttered scenes remains a fundamental and application-critical challenge in embodied AI. Closely packed objects cause inevitable occlusions, making safe action generation particularly difficult. Under such partial observability, effective policies must not only generalize across diverse objects and layouts but also reason about occlusion to avoid collisions. However, collecting large-scale real-world data for this task remains prohibitively expensive, leaving this problem largely unsolved. In this paper, we introduce FetchBot, a sim-to-real framework for this challenge. We first curate a large-scale synthetic dataset featuring 1M diverse scenes and 500k representative demonstrations. Based on this dataset, FetchBot employs a depth-conditioned method for action generation, which leverages structural cues to enable robust obstacle-aware action planning. However, depth is perfect in simulation but noisy in real-world environments. To address this sim-to-real gap, FetchBot predicts depth from RGB inputs using a foundation model and integrates local occupancy prediction as a pre-training task, providing a generalizable latent representation for sim-to-real transfer. Extensive experiments in simulation and real-world environments demonstrate the strong zero-shot sim-to-real transfer, effective clutter handling, and adaptability to novel scenarios. In cluttered environments, it achieves an average real-world success rate of 89.95%, significantly outperforming prior methods. Moreover, FetchBot demonstrates excellent robustness in challenging cases, such as fetching transparent, reflective, and irregular objects, highlighting its practical value.
>
---
#### [replaced 021] Dexterous Contact-Rich Manipulation via the Contact Trust Region
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.02291v3](http://arxiv.org/pdf/2505.02291v3)**

> **作者:** H. J. Terry Suh; Tao Pang; Tong Zhao; Russ Tedrake
>
> **摘要:** What is a good local description of contact dynamics for contact-rich manipulation, and where can we trust this local description? While many approaches often rely on the Taylor approximation of dynamics with an ellipsoidal trust region, we argue that such approaches are fundamentally inconsistent with the unilateral nature of contact. As a remedy, we present the Contact Trust Region (CTR), which captures the unilateral nature of contact while remaining efficient for computation. With CTR, we first develop a Model-Predictive Control (MPC) algorithm capable of synthesizing local contact-rich plans. Then, we extend this capability to plan globally by stitching together local MPC plans, enabling efficient and dexterous contact-rich manipulation. To verify the performance of our method, we perform comprehensive evaluations, both in high-fidelity simulation and on hardware, on two contact-rich systems: a planar IiwaBimanual system and a 3D AllegroHand system. On both systems, our method offers a significantly lower-compute alternative to existing RL-based approaches to contact-rich manipulation. In particular, our Allegro in-hand manipulation policy, in the form of a roadmap, takes fewer than 10 minutes to build offline on a standard laptop using just its CPU, with online inference taking just a few seconds. Experiment data, video and code are available at ctr.theaiinstitute.com.
>
---
#### [replaced 022] MALMM: Multi-Agent Large Language Models for Zero-Shot Robotics Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.17636v2](http://arxiv.org/pdf/2411.17636v2)**

> **作者:** Harsh Singh; Rocktim Jyoti Das; Mingfei Han; Preslav Nakov; Ivan Laptev
>
> **备注:** 48 pages
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable planning abilities across various domains, including robotics manipulation and navigation. While recent efforts in robotics have leveraged LLMs both for high-level and low-level planning, these approaches often face significant challenges, such as hallucinations in long-horizon tasks and limited adaptability due to the generation of plans in a single pass without real-time feedback. To address these limitations, we propose a novel multi-agent LLM framework, Multi-Agent Large Language Model for Manipulation (MALMM) that distributes high-level planning and low-level control code generation across specialized LLM agents, supervised by an additional agent that dynamically manages transitions. By incorporating observations from the environment after each step, our framework effectively handles intermediate failures and enables adaptive re-planning. Unlike existing methods, our approach does not rely on pre-trained skill policies or in-context learning examples and generalizes to a variety of new tasks. We evaluate our approach on nine RLBench tasks, including long-horizon tasks, and demonstrate its ability to solve robotics manipulation in a zero-shot setting, thereby overcoming key limitations of existing LLM-based manipulation methods.
>
---
#### [replaced 023] Mesh-Learner: Texturing Mesh with Spherical Harmonics
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.19938v3](http://arxiv.org/pdf/2504.19938v3)**

> **作者:** Yunfei Wan; Jianheng Liu; Chunran Zheng; Jiarong Lin; Fu Zhang
>
> **备注:** IROS2025 Accepted
>
> **摘要:** In this paper, we present a 3D reconstruction and rendering framework termed Mesh-Learner that is natively compatible with traditional rasterization pipelines. It integrates mesh and spherical harmonic (SH) texture (i.e., texture filled with SH coefficients) into the learning process to learn each mesh s view-dependent radiance end-to-end. Images are rendered by interpolating surrounding SH Texels at each pixel s sampling point using a novel interpolation method. Conversely, gradients from each pixel are back-propagated to the related SH Texels in SH textures. Mesh-Learner exploits graphic features of rasterization pipeline (texture sampling, deferred rendering) to render, which makes Mesh-Learner naturally compatible with tools (e.g., Blender) and tasks (e.g., 3D reconstruction, scene rendering, reinforcement learning for robotics) that are based on rasterization pipelines. Our system can train vast, unlimited scenes because we transfer only the SH textures within the frustum to the GPU for training. At other times, the SH textures are stored in CPU RAM, which results in moderate GPU memory usage. The rendering results on interpolation and extrapolation sequences in the Replica and FAST-LIVO2 datasets achieve state-of-the-art performance compared to existing state-of-the-art methods (e.g., 3D Gaussian Splatting and M2-Mapping). To benefit the society, the code will be available at https://github.com/hku-mars/Mesh-Learner.
>
---
