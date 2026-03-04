# 机器人 cs.RO

- **最新发布 55 篇**

- **更新 26 篇**

## 最新发布

#### [new 001] DreamFlow: Local Navigation Beyond Observation via Conditional Flow Matching in the Latent Space
- **分类: cs.RO**

- **简介: 该论文属于机器人局部导航任务，解决复杂环境中感知不足导致的导航失败问题。通过条件流匹配方法扩展感知范围，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.02976](https://arxiv.org/pdf/2603.02976)**

> **作者:** Jiwon Park; Dongkyu Lee; I Made Aswin Nahrendra; Jaeyoung Lim; Hyun Myung
>
> **摘要:** Local navigation in cluttered environments often suffers from dense obstacles and frequent local minima. Conventional local planners rely on heuristics and are prone to failure, while deep reinforcement learning(DRL)based approaches provide adaptability but are constrained by limited onboard sensing. These limitations lead to navigation failures because the robot cannot perceive structures outside its field of view. In this paper, we propose DreamFlow, a DRL-based local navigation framework that extends the robot's perceptual horizon through conditional flow matching(CFM). The proposed CFM based prediction module learns probabilistic mapping between local height map latent representation and broader spatial representation conditioned on navigation context. This enables the navigation policy to predict unobserved environmental features and proactively avoid potential local minima. Experimental results demonstrate that DreamFlow outperforms existing methods in terms of latent prediction accuracy and navigation performance in simulation. The proposed method was further validated in cluttered real world environments with a quadrupedal robot. The project page is available at this https URL.
>
---
#### [new 002] ACE-Brain-0: Spatial Intelligence as a Shared Scaffold for Universal Embodiments
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文属于多模态大模型任务，旨在解决跨身体形态的通用智能问题。通过构建ACE-Brain-0，融合空间推理与不同应用场景，提升模型的泛化能力与专业性能。**

- **链接: [https://arxiv.org/pdf/2603.03198](https://arxiv.org/pdf/2603.03198)**

> **作者:** Ziyang Gong; Zehang Luo; Anke Tang; Zhe Liu; Shi Fu; Zhi Hou; Ganlin Yang; Weiyun Wang; Xiaofeng Wang; Jianbo Liu; Gen Luo; Haolan Kang; Shuang Luo; Yue Zhou; Yong Luo; Li Shen; Xiaosong Jia; Yao Mu; Xue Yang; Chunxiao Liu; Junchi Yan; Hengshuang Zhao; Dacheng Tao; Xiaogang Wang
>
> **备注:** Code: this https URL Hugging Face: this https URL
>
> **摘要:** Universal embodied intelligence demands robust generalization across heterogeneous embodiments, such as autonomous driving, robotics, and unmanned aerial vehicles (UAVs). However, existing embodied brain in training a unified model over diverse embodiments frequently triggers long-tail data, gradient interference, and catastrophic forgetting, making it notoriously difficult to balance universal generalization with domain-specific proficiency. In this report, we introduce ACE-Brain-0, a generalist foundation brain that unifies spatial reasoning, autonomous driving, and embodied manipulation within a single multimodal large language model~(MLLM). Our key insight is that spatial intelligence serves as a universal scaffold across diverse physical embodiments: although vehicles, robots, and UAVs differ drastically in morphology, they share a common need for modeling 3D mental space, making spatial cognition a natural, domain-agnostic foundation for cross-embodiment transfer. Building on this insight, we propose the Scaffold-Specialize-Reconcile~(SSR) paradigm, which first establishes a shared spatial foundation, then cultivates domain-specialized experts, and finally harmonizes them through data-free model merging. Furthermore, we adopt Group Relative Policy Optimization~(GRPO) to strengthen the model's comprehensive capability. Extensive experiments demonstrate that ACE-Brain-0 achieves competitive and even state-of-the-art performance across 24 spatial and embodiment-related benchmarks.
>
---
#### [new 003] A Novel Modular Cable-Driven Soft Robotic Arm with Multi-Segment Reconfigurability
- **分类: cs.RO**

- **简介: 论文提出一种可多段重构的模块化柔性机械臂，解决传统结构适应性差的问题。通过模块堆叠和材料刚度调节，提升工作空间与负载能力。**

- **链接: [https://arxiv.org/pdf/2603.02468](https://arxiv.org/pdf/2603.02468)**

> **作者:** Moeen Ul Islam; Cheng Ouyang; Xinda Qi; Azlan Zahid; Xiaobo Tan; Dong Chen
>
> **备注:** 6 pages, 8 figures, Submitted to IEEE/ASME International Conference on Advanced Intelligent Mechatronics
>
> **摘要:** This paper presents a novel, modular, cable-driven soft robotic arm featuring multi-segment reconfigurability. The proposed architecture enables a stackable system with independent segment control, allowing scalable adaptation to diverse structural and application requirements. The system is fabricated from soft silicone material and incorporates embedded tendon-routing channels with a protective dual-helical tendon structure. Experimental results showed that modular stacking substantially expanded the reachable workspace: relative to the single-segment arm, the three-segment configuration achieved up to a 13-fold increase in planar workspace area and a 38.9-fold increase in workspace volume. Furthermore, this study investigated the effect of silicone stiffness on actuator performance. The results revealed a clear trade-off between compliance and stiffness: softer silicone improved bending flexibility, while stiffer silicone improved structural rigidity and load-bearing stability. These results highlight the potential of stiffness tuning to balance compliance and strength for configuring scalable, reconfigurable soft robotic arms.
>
---
#### [new 004] Generative adversarial imitation learning for robot swarms: Learning from human demonstrations and trained policies
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文属于模仿学习任务，旨在让机器人群体从人类示范中学习集体行为。通过生成对抗模仿学习框架，实现对多种任务的高效学习与真实部署。**

- **链接: [https://arxiv.org/pdf/2603.02783](https://arxiv.org/pdf/2603.02783)**

> **作者:** Mattes Kraus; Jonas Kuckling
>
> **备注:** Accepted for publication at the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** In imitation learning, robots are supposed to learn from demonstrations of the desired behavior. Most of the work in imitation learning for swarm robotics provides the demonstrations as rollouts of an existing policy. In this work, we provide a framework based on generative adversarial imitation learning that aims to learn collective behaviors from human demonstrations. Our framework is evaluated across six different missions, learning both from manual demonstrations and demonstrations derived from a PPO-trained policy. Results show that the imitation learning process is able to learn qualitatively meaningful behaviors that perform similarly well as the provided demonstrations. Additionally, we deploy the learned policies on a swarm of TurtleBot 4 robots in real-robot experiments. The exhibited behaviors preserved their visually recognizable character and their performance is comparable to the one achieved in simulation.
>
---
#### [new 005] How to Peel with a Knife: Aligning Fine-Grained Manipulation with Human Preference
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; eess.SY**

- **简介: 该论文研究机器人精细操作任务，解决自主机器人在食品加工等领域的操作难题。通过两阶段学习框架，提升操作质量与人类偏好一致。**

- **链接: [https://arxiv.org/pdf/2603.03280](https://arxiv.org/pdf/2603.03280)**

> **作者:** Toru Lin; Shuying Deng; Zhao-Heng Yin; Pieter Abbeel; Jitendra Malik
>
> **备注:** Project page can be found at this https URL
>
> **摘要:** Many essential manipulation tasks - such as food preparation, surgery, and craftsmanship - remain intractable for autonomous robots. These tasks are characterized not only by contact-rich, force-sensitive dynamics, but also by their "implicit" success criteria: unlike pick-and-place, task quality in these domains is continuous and subjective (e.g. how well a potato is peeled), making quantitative evaluation and reward engineering difficult. We present a learning framework for such tasks, using peeling with a knife as a representative example. Our approach follows a two-stage pipeline: first, we learn a robust initial policy via force-aware data collection and imitation learning, enabling generalization across object variations; second, we refine the policy through preference-based finetuning using a learned reward model that combines quantitative task metrics with qualitative human feedback, aligning policy behavior with human notions of task quality. Using only 50-200 peeling trajectories, our system achieves over 90% average success rates on challenging produce including cucumbers, apples, and potatoes, with performance improving by up to 40% through preference-based finetuning. Remarkably, policies trained on a single produce category exhibit strong zero-shot generalization to unseen in-category instances and to out-of-distribution produce from different categories while maintaining over 90% success rates.
>
---
#### [new 006] Uni-Skill: Building Self-Evolving Skill Repository for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出Uni-Skill，解决机器人操作中技能库固定、适应性差的问题。通过自动演化技能库和构建SkillFolder，实现高效技能获取与泛化。**

- **链接: [https://arxiv.org/pdf/2603.02623](https://arxiv.org/pdf/2603.02623)**

> **作者:** Senwei Xie; Yuntian Zhang; Ruiping Wang; Xilin Chen
>
> **备注:** Accepted to ICRA2026
>
> **摘要:** While skill-centric approaches leverage foundation models to enhance generalization in compositional tasks, they often rely on fixed skill libraries, limiting adaptability to new tasks without manual intervention. To address this, we propose Uni-Skill, a Unified Skill-centric framework that supports skill-aware planning and facilitates automatic skill evolution. Unlike prior methods that restrict planning to predefined skills, Uni-Skill requests for new skill implementations when existing ones are insufficient, ensuring adaptable planning with self-augmented skill library. To support automatic implementation of diverse skills requested by the planning module, we construct SkillFolder, a VerbNet-inspired repository derived from large-scale unstructured robotic videos. SkillFolder introduces a hierarchical skill taxonomy that captures diverse skill descriptions at multiple levels of abstraction. By populating this taxonomy with large-scale, automatically annotated demonstrations, Uni-Skill shifts the paradigm of skill acquisition from inefficient manual annotation to efficient offline structural retrieval. Retrieved examples provide semantic supervision over behavior patterns and fine-grained references for spatial trajectories, enabling few-shot skill inference without deployment-time demonstrations. Comprehensive experiments in both simulation and real-world settings verify the state-of-the-art performance of Uni-Skill over existing VLM-based skill-centric approaches, highlighting its advanced reasoning capabilities and strong zero-shot generalization across a wide range of novel tasks.
>
---
#### [new 007] Learning Object-Centric Spatial Reasoning for Sequential Manipulation in Cluttered Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决杂乱环境中物体抓取问题。提出Unveiler框架，通过分离空间推理与动作执行，提升效率与成功率。**

- **链接: [https://arxiv.org/pdf/2603.02511](https://arxiv.org/pdf/2603.02511)**

> **作者:** Chrisantus Eze; Ryan C Julian; Christopher Crick
>
> **摘要:** Robotic manipulation in cluttered environments presents a critical challenge for automation. Recent large-scale, end-to-end models demonstrate impressive capabilities but often lack the data efficiency and modularity required for retrieving objects in dense clutter. In this work, we argue for a paradigm of specialized, decoupled systems and present Unveiler, a framework that explicitly separates high-level spatial reasoning from low-level action execution. Unveiler's core is a lightweight, transformer-based Spatial Relationship Encoder (SRE) that sequentially identifies the most critical obstacle for removal. This discrete decision is then passed to a rotation-invariant Action Decoder for execution. We demonstrate that this decoupled architecture is not only more computationally efficient in terms of parameter count and inference time, but also significantly outperforms both classic end-to-end policies and modern, large-model-based baselines in retrieving targets from dense clutter. The SRE is trained in two stages: imitation learning from heuristic demonstrations provides sample-efficient initialization, after which PPO fine-tuning enables the policy to discover removal strategies that surpass the heuristic in dense clutter. Our results, achieving up to 97.6\% success in partially occluded and 90.0\% in fully occluded scenarios in simulation, make a case for the power of specialized, object-centric reasoning in complex manipulation tasks. Additionally, we demonstrate that the SRE's spatial reasoning transfers zero-shot to real scenes, and validate the full system on a physical robot requiring only geometric workspace calibration; no learned components are retrained.
>
---
#### [new 008] Tether: Autonomous Functional Play with Correspondence-Driven Trajectory Warping
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出Tether方法，用于机器人自主功能玩耍。解决机器人在复杂环境中自主学习的问题，通过结构化交互和视觉语言模型生成高质量数据，提升模仿学习效果。**

- **链接: [https://arxiv.org/pdf/2603.03278](https://arxiv.org/pdf/2603.03278)**

> **作者:** William Liang; Sam Wang; Hung-Ju Wang; Osbert Bastani; Yecheng Jason Ma; Dinesh Jayaraman
>
> **备注:** International Conference on Learning Representations (ICLR), 2026. Project website and code: this https URL
>
> **摘要:** The ability to conduct and learn from interaction and experience is a central challenge in robotics, offering a scalable alternative to labor-intensive human demonstrations. However, realizing such "play" requires (1) a policy robust to diverse, potentially out-of-distribution environment states, and (2) a procedure that continuously produces useful robot experience. To address these challenges, we introduce Tether, a method for autonomous functional play involving structured, task-directed interactions. First, we design a novel open-loop policy that warps actions from a small set of source demonstrations (<=10) by anchoring them to semantic keypoint correspondences in the target scene. We show that this design is extremely data-efficient and robust even under significant spatial and semantic variations. Second, we deploy this policy for autonomous functional play in the real world via a continuous cycle of task selection, execution, evaluation, and improvement, guided by the visual understanding capabilities of vision-language models. This procedure generates diverse, high-quality datasets with minimal human intervention. In a household-like multi-object setup, our method is the first to perform many hours of autonomous multi-task play in the real world starting from only a handful of demonstrations. This produces a stream of data that consistently improves the performance of closed-loop imitation policies over time, ultimately yielding over 1000 expert-level trajectories and training policies competitive with those learned from human-collected demonstrations.
>
---
#### [new 009] Instant and Reversible Adhesive-free Bonding Between Silicones and Glossy Papers for Soft Robotics
- **分类: cs.RO**

- **简介: 该论文属于软体机器人领域，解决硅胶与光面纸粘接难题，提出一种快速、可逆的无胶粘接方法，提升可重复使用性和设计灵活性。**

- **链接: [https://arxiv.org/pdf/2603.02500](https://arxiv.org/pdf/2603.02500)**

> **作者:** Takumi Shibuya; Kazuya Murakami; Akitsu Shigetou; Jun Shintake
>
> **摘要:** Integrating silicone with non-extensible materials is a common strategy used in the fabrication of fluidically-driven soft actuators, yet conventional approaches often rely on irreversible adhesives or embedding processes that are labor-intensive and difficult to modify. This work presents silicone-glossy paper bonding (SGB), a rapid, adhesive-free, and solvent-reversible bonding approach that forms robust silicone-paper interfaces simply through contact. The SGB interface withstands high mechanical loads (shear strength > 61 kPa) and can be fully detached and reassembled via ethanol immersion without loss of performance, enabling component reuse and rapid redesign. Characterization studies indicate that surface functional groups primarily govern adhesion on the glossy paper and the modulus of the silicone, while durability and environmental response clarify the conditions for reversible debonding. The results further suggest a synergistic interaction of hydrogen bonding and oligomer diffusion, yielding strong yet reconfigurable adhesion. Soft actuators fabricated using SGB design exhibit equal or greater performance compared to conventional embedded-layer design and enable programmable actuation modes, including contraction, bending, and twisting. By simplifying fabrication while supporting reuse and rapid iteration, SGB offers a scalable and sustainable platform for rapid prototyping in soft robotics.
>
---
#### [new 010] SPARC: Spatial-Aware Path Planning via Attentive Robot Communication
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多机器人路径规划任务，解决通信效率问题。提出RMHA机制，通过空间距离增强注意力，提升密集环境下的协作成功率。**

- **链接: [https://arxiv.org/pdf/2603.02845](https://arxiv.org/pdf/2603.02845)**

> **作者:** Sayang Mu; Xiangyu Wu; Bo An
>
> **摘要:** Efficient communication is critical for decentralized Multi-Robot Path Planning (MRPP), yet existing learned communication methods treat all neighboring robots equally regardless of their spatial proximity, leading to diluted attention in congested regions where coordination matters most. We propose Relation enhanced Multi Head Attention (RMHA), a communication mechanism that explicitly embeds pairwise Manhattan distances into the attention weight computation, enabling each robot to dynamically prioritize messages from spatially relevant neighbors. Combined with a distance-constrained attention mask and GRU gated message fusion, RMHA integrates seamlessly with MAPPO for stable end-to-end training. In zero-shot generalization from 8 training robots to 128 test robots on 40x40 grids, RMHA achieves approximately 75 percent success rate at 30 percent obstacle density outperforming the best baseline by over 25 percentage points. Ablation studies confirm that distance-relation encoding is the key contributor to success rate improvement in high-density environments. Index Terms-Multi-robot path planning, graph attention mechanism, multi-head attention, communication optimization, cooperative decision-making
>
---
#### [new 011] Robotic Grasping and Placement Controlled by EEG-Based Hybrid Visual and Motor Imagery
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于脑机接口与机器人控制任务，旨在通过EEG实现意图驱动的抓取与放置。工作包括构建双通道意图接口，融合视觉和运动想象，提升人机协作效率。**

- **链接: [https://arxiv.org/pdf/2603.03181](https://arxiv.org/pdf/2603.03181)**

> **作者:** Yichang Liu; Tianyu Wang; Ziyi Ye; Yawei Li; Yu-Gang Jiang; Shouyan Wang; Yanwei Fu
>
> **摘要:** We present a framework that integrates EEG-based visual and motor imagery (VI/MI) with robotic control to enable real-time, intention-driven grasping and placement. Motivated by the promise of BCI-driven robotics to enhance human-robot interaction, this system bridges neural signals with physical control by deploying offline-pretrained decoders in a zero-shot manner within an online streaming pipeline. This establishes a dual-channel intent interface that translates visual intent into robotic actions, with VI identifying objects for grasping and MI determining placement poses, enabling intuitive control over both what to grasp and where to place. The system operates solely on EEG via a cue-free imagery protocol, achieving integration and online validation. Implemented on a Base robotic platform and evaluated across diverse scenarios, including occluded targets or varying participant postures, the system achieves online decoding accuracies of 40.23% (VI) and 62.59% (MI), with an end-to-end task success rate of 20.88%. These results demonstrate that high-level visual cognition can be decoded in real time and translated into executable robot commands, bridging the gap between neural signals and physical interaction, and validating the flexibility of a purely imagery-based BCI paradigm for practical human-robot collaboration.
>
---
#### [new 012] Emerging trends in Cislunar Space for Lunar Science Exploration and Space Robotics aiding Human Spaceflight Safety
- **分类: cs.RO; astro-ph.EP**

- **简介: 论文聚焦于月球科学探索与太空机器人在载人登月中的应用，旨在提升人类深空探索的安全性与可持续性。通过整合人工智能与自主机器人技术，解决月面作业、资源利用及安全支持等问题。**

- **链接: [https://arxiv.org/pdf/2603.02878](https://arxiv.org/pdf/2603.02878)**

> **作者:** Arsalan Muhammad; Yue Wang; Hai Huang; Hao Wang
>
> **备注:** Conference Proceedings of 2nd IAA Conference on AI in and for Space (2nd IAA SPAICE), Suzhou, China, 1-3 November, 2025
>
> **摘要:** In recent years, the Moon has emerged as an unparalleled extraterrestrial testbed for advancing cuttingedge technological and scientific research critical to enabling sustained human presence on its surface and supporting future interplanetary exploration. This study identifies and investigates two pivotal research domains with substantial transformative potential for accelerating humanity interplanetary aspirations. First is Lunar Science Exploration with Artificial Intelligence and Space Robotics which focusses on AI and Space Robotics redefining the frontiers of space exploration. Second being Space Robotics aid in manned spaceflight to the Moon serving as critical assets for pre-deployment infrastructure development, In-Situ Resource Utilization, surface operations support, and astronaut safety assurance. By integrating autonomy, machine learning, and realtime sensor fusion, space robotics not only augment human capabilities but also serve as force multipliers in achieving sustainable lunar exploration, paving the way for future crewed missions to Mars and beyond.
>
---
#### [new 013] Safe Whole-Body Loco-Manipulation via Combined Model and Learning-based Control
- **分类: cs.RO; cs.HC; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决足式机器人在动态环境中安全地实现全身运动与操作的问题。通过结合模型控制与强化学习，提升机器人的安全性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.02443](https://arxiv.org/pdf/2603.02443)**

> **作者:** Alexander Schperberg; Yeping Wang; Stefano Di Cairano
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA), June 2026, in Vienna, Austria
>
> **摘要:** Simultaneous locomotion and manipulation enables robots to interact with their environment beyond the constraints of a fixed base. However, coordinating legged locomotion with arm manipulation, while considering safety and compliance during contact interaction remains challenging. To this end, we propose a whole-body controller that combines a model-based admittance control for the manipulator arm with a Reinforcement Learning (RL) policy for legged locomotion. The admittance controller maps external wrenches--such as those applied by a human during physical interaction--into desired end-effector velocities, allowing for compliant behavior. The velocities are tracked jointly by the arm and leg controllers, enabling a unified 6-DoF force response. The model-based design permits accurate force control and safety guarantees via a Reference Governor (RG), while robustness is further improved by a Kalman filter enhanced with neural networks for reliable base velocity estimation. We validate our approach in both simulation and hardware using the Unitree Go2 quadruped robot with a 6-DoF arm and wrist-mounted 6-DoF Force/Torque sensor. Results demonstrate accurate tracking of interaction-driven velocities, compliant behavior, and safe, reliable performance in dynamic settings.
>
---
#### [new 014] Goal-Oriented Semantic Communication for ISAC-Enabled Robotic Obstacle Avoidance
- **分类: cs.RO**

- **简介: 该论文针对无人机避障任务，提出GOSC框架以高效传输感知与控制信号，减少通信负担并保持100%任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.02291](https://arxiv.org/pdf/2603.02291)**

> **作者:** Wenjie Liu; Yansha Deng; Henk Wymeersch
>
> **备注:** 13 pages, 15 figures
>
> **摘要:** We investigate an integrated sensing and communication (ISAC)-enabled BS for the unmanned aerial vehicle (UAV) obstacle avoidance task, and propose a goal-oriented semantic communication (GOSC) framework for the BS to transmit sensing and command and control (C&C) signals efficiently and effectively. Our GOSC framework establishes a closed loop for sensing-C&C generation-sensing and C&C transmission: For sensing, a Kalman filter (KF) is applied to continuously predict UAV positions, mitigating the reliance of UAV position acquisition on continuous sensing signal transmission, and enhancing position estimation accuracy through sensing-prediction fusion. Based on the refined estimation position provided by the KF, we develop a Mahalanobis distance-based dynamic window approach (MD-DWA) to generate precise C&C signals under uncertainty, in which we derive the mathematical expression of the minimum Mahalanobis distance required to guarantee collision avoidance. Finally, for efficient sensing and C&C signal transmission, we propose an effectiveness-aware deep Q-network (E-DQN) to determine the transmission of sensing and C&C signals based on their value of information (VoI). The VoI of sensing signals is quantified by the reduction in uncertainty entropy of UAV's position estimation, while the VoI of C&C signals is measured by their contribution to UAV navigation improvement. Extensive simulations validate the effectiveness of our proposed GOSC framework. Compared to the conventional ISAC transmission framework that transmits sensing and C&C signals at every time slot, GOSC achieves the same 100% task success rate while reducing the number of transmitted sensing and C&C signals by 92.4% and the number of transmission time slots by 85.5%.
>
---
#### [new 015] MA-CoNav: A Master-Slave Multi-Agent Framework with Hierarchical Collaboration and Dual-Level Reflection for Long-Horizon Embodied VLN
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉语言导航任务，旨在解决长距离导航中感知偏差和决策漂移问题。提出MA-CoNav框架，通过多智能体协作与双阶段反思机制提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.03024](https://arxiv.org/pdf/2603.03024)**

> **作者:** Ling Luo; Qianqian Bai
>
> **摘要:** Vision-Language Navigation (VLN) aims to empower robots with the ability to perform long-horizon navigation in unfamiliar environments based on complex linguistic instructions. Its success critically hinges on establishing an efficient ``language-understanding -- visual-perception -- embodied-execution'' closed loop. Existing methods often suffer from perceptual distortion and decision drift in complex, long-distance tasks due to the cognitive overload of a single agent. Inspired by distributed cognition theory, this paper proposes MA-CoNav, a Multi-Agent Collaborative Navigation framework. This framework adopts a ``Master-Slave'' hierarchical agent collaboration architecture, decoupling and distributing the perception, planning, execution, and memory functions required for navigation tasks to specialized agents. Specifically, the Master Agent is responsible for global orchestration, while the Subordinate Agent group collaborates through a clear division of labor: an Observation Agent generates environment descriptions, a Planning Agent performs task decomposition and dynamic verification, an Execution Agent handles simultaneous mapping and action, and a Memory Agent manages structured experiences. Furthermore, the framework introduces a ``Local-Global'' dual-stage reflection mechanism to dynamically optimize the entire navigation pipeline. Empirical experiments were conducted using a real-world indoor dataset collected by a Limo Pro robot, with no scene-specific fine-tuning performed on the models throughout the process. The results demonstrate that MA-CoNav comprehensively outperforms existing mainstream VLN methods across multiple metrics.
>
---
#### [new 016] RL-Based Coverage Path Planning for Deformable Objects on 3D Surfaces
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决变形物体在三维表面的覆盖路径规划问题。通过强化学习方法，结合视觉与触觉反馈，提升表面擦拭效果。**

- **链接: [https://arxiv.org/pdf/2603.03137](https://arxiv.org/pdf/2603.03137)**

> **作者:** Yuhang Zhang; Jinming Ma; Feng Wu
>
> **备注:** 8 pages, 8 figures. Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Currently, manipulation tasks for deformable objects often focus on activities like folding clothes, handling ropes, and manipulating bags. However, research on contact-rich tasks involving deformable objects remains relatively underdeveloped. When humans use cloth or sponges to wipe surfaces, they rely on both vision and tactile feedback. Yet, current algorithms still face challenges with issues like occlusion, while research on tactile perception for manipulation is still evolving. Tasks such as covering surfaces with deformable objects demand not only perception but also precise robotic manipulation. To address this, we propose a method that leverages efficient and accessible simulators for task execution. Specifically, we train a reinforcement learning agent in a simulator to manipulate deformable objects for surface wiping tasks. We simplify the state representation of object surfaces using harmonic UV mapping, process contact feedback from the simulator on 2D feature maps, and use scaled grouped convolutions (SGCNN) to extract features efficiently. The agent then outputs actions in a reduced-dimensional action space to generate coverage paths. Experiments demonstrate that our method outperforms previous approaches in key metrics, including total path length and coverage area. We deploy these paths on a Kinova Gen3 manipulator to perform wiping experiments on the back of a torso model, validating the feasibility of our approach.
>
---
#### [new 017] Look Forward to Walk Backward: Efficient Terrain Memory for Backward Locomotion with Forward Vision
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决机器人后退时因视野受限导致的避障问题。通过构建地形记忆框架，利用前视信息实现高效后退运动。**

- **链接: [https://arxiv.org/pdf/2603.03138](https://arxiv.org/pdf/2603.03138)**

> **作者:** Shixin Luo; Songbo Li; Yuan Hao; Yaqi Wang; Jun Zheng; Jun Wu; Qiuguo Zhu
>
> **备注:** Accepted for 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Legged robots with egocentric forward-facing depth cameras can couple exteroception and proprioception to achieve robust forward agility on complex terrain. When these robots walk backward, the forward-only field of view provides no preview. Purely proprioceptive controllers can remain stable on moderate ground when moving backward but cannot fully exploit the robot's capabilities on complex terrain and must collide with obstacles. We present Look Forward to Walk Backward (LF2WB), an efficient terrain-memory locomotion framework that uses forward egocentric depth and proprioception to write a compact associative memory during forward motion and to retrieve it for collision-free backward locomotion without rearward vision. The memory backbone employs a delta-rule selective update that softly removes then writes the memory state along the active subspace. Training uses hardware-efficient parallel computation, and deployment runs recurrent, constant-time per-step inference with a constant-size state, making the approach suitable for onboard processors on low-cost robots. Experiments in both simulations and real-world scenarios demonstrate the effectiveness of our method, improving backward agility across complex terrains under limited sensing.
>
---
#### [new 018] Compositional Visual Planning via Inference-Time Diffusion Scaling
- **分类: cs.RO**

- **简介: 该论文属于机器人长期视觉规划任务，旨在解决扩散模型在长序列生成中的不稳定性问题。通过在推理阶段强制边界一致，实现全局一致的生成，无需额外训练。**

- **链接: [https://arxiv.org/pdf/2603.02646](https://arxiv.org/pdf/2603.02646)**

> **作者:** Yixin Zhang; Yunhao Luo; Utkarsh Aashu Mishra; Woo Chul Shin; Yongxin Chen; Danfei Xu
>
> **摘要:** Diffusion models excel at short-horizon robot planning, yet scaling them to long-horizon tasks remains challenging due to computational constraints and limited training data. Existing compositional approaches stitch together short segments by separately denoising each component and averaging overlapping regions. However, this suffers from instability as the factorization assumption breaks down in noisy data space, leading to inconsistent global plans. We propose that the key to stable compositional generation lies in enforcing boundary agreement on the estimated clean data (Tweedie estimates) rather than on noisy intermediate states. Our method formulates long-horizon planning as inference over a chain-structured factor graph of overlapping video chunks, where pretrained short-horizon video diffusion models provide local priors. At inference time, we enforce boundary agreement through a novel combination of synchronous and asynchronous message passing that operates on Tweedie estimates, producing globally consistent guidance without requiring additional training. Our training-free framework demonstrates significant improvements over existing baselines, effectively generalizing to unseen start-goal combinations that were not present in the original training data. Project website: this https URL
>
---
#### [new 019] Agentic Self-Evolutionary Replanning for Embodied Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决复杂环境中导航失败后的重规划问题。提出SERP方法，通过自进化模型提升导航成功率与效率。**

- **链接: [https://arxiv.org/pdf/2603.02772](https://arxiv.org/pdf/2603.02772)**

> **作者:** Guoliang Li; Ruihua Han; Chengyang Li; He Li; Shuai Wang; Wenchao Ding; Hong Zhang; Chengzhong Xu
>
> **备注:** 8 pages, 10 figures, 4 tables, submitted to IEEE for possible publication
>
> **摘要:** Failure is inevitable for embodied navigation in complex environments. To enhance the resilience, replanning (RP) is a viable option, where the robot is allowed to fail, but is capable of adjusting plan until success. However, existing RP approaches freeze the ego action model and miss the opportunities to explore better plans by upgrading the robot itself. To address this limitation, we propose Self-Evolutionary RePlanning, or SERP for short, which leads to a paradigm shift from frozen models towards evolving models by run-time learning from recent experiences. In contrast to existing model evolution approaches that often get stuck at predefined static parameters, we introduce agentic self-evolving action model that uses in-context learning with auto-differentiation (ILAD) for adaptive function adjustment and global parameter reset. To achieve token-efficient replanning for SERP, we also propose graph chain-of-thought (GCOT) replanning with large language model (LLM) inference over distilled graphs. Extensive simulation and real-world experiments demonstrate that SERP achieves higher success rate with lower token expenditure over various benchmarks, validating its superior robustness and efficiency across diverse environments.
>
---
#### [new 020] A Robust Simulation Framework for Verification and Validation of Autonomous Maritime Navigation in Adverse Weather and Constrained Environments
- **分类: cs.RO**

- **简介: 该论文属于自主船舶导航验证与确认任务，旨在解决恶劣天气和受限环境下的导航可靠性问题。构建了高保真仿真框架，模拟天气和水深影响，支持安全测试与评估。**

- **链接: [https://arxiv.org/pdf/2603.02487](https://arxiv.org/pdf/2603.02487)**

> **作者:** Mayur S. Patil; Nataraj Sudharsan; Anthony S. Saaiby; JiaChang Xing; Keliang Pan; Veneela Ammula; Jude Tomdio; Jin Wang; Michael Kei; Heonyong Kang; Sivakumar Rathinam; Prabhakar R. Pagilla
>
> **摘要:** Maritime Autonomous Surface Ships (MASS) have emerged as a promising solution to enhance navigational safety, operational efficiency, and long-term cost effectiveness. However, their reliable deployment requires rigorous verification and validation (V\&V) under various environmental conditions, including extreme and safety-critical scenarios. This paper presents an enhanced virtual simulation framework to support the V\&V of MASS in realistic maritime environments, with particular emphasis on the influence of weather and bathymetry on autonomous navigation performance. The framework incorporates a high-fidelity environmental modeling suite capable of simulating adverse weather conditions such as rain, fog, and wave dynamics. The key factors that affect weather, such as rain and visibility, are parameterized to affect sea-state characteristics, perception, and sensing systems, resulting in position and velocity uncertainty, reduced visibility, and degraded situational awareness. Furthermore, high-resolution bathymetric data from major U.S. ports are integrated to enable depth-aware navigation, grounding prevention capabilities, and evaluation of vessel controllability in shallow or confined waterways. The proposed framework offers extensive configurability, enabling systematic testing in a wide spectrum of maritime conditions, including scenarios that are impractical or unsafe to replicate in real-world trials, thus supporting the V\&V of MASS.
>
---
#### [new 021] HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文提出HoMMI框架，解决从人类演示中学习全身移动操作的问题。通过跨体感策略设计，缩小人机差异，实现复杂协作任务。**

- **链接: [https://arxiv.org/pdf/2603.03243](https://arxiv.org/pdf/2603.03243)**

> **作者:** Xiaomeng Xu; Jisang Park; Han Zhang; Eric Cousineau; Aditya Bhat; Jose Barreiros; Dian Wang; Shuran Song
>
> **摘要:** We present Whole-Body Mobile Manipulation Interface (HoMMI), a data collection and policy learning framework that learns whole-body mobile manipulation directly from robot-free human demonstrations. We augment UMI interfaces with egocentric sensing to capture the global context required for mobile manipulation, enabling portable, robot-free, and scalable data collection. However, naively incorporating egocentric sensing introduces a larger human-to-robot embodiment gap in both observation and action spaces, making policy transfer difficult. We explicitly bridge this gap with a cross-embodiment hand-eye policy design, including an embodiment agnostic visual representation; a relaxed head action representation; and a whole-body controller that realizes hand-eye trajectories through coordinated whole-body motion under robot-specific physical constraints. Together, these enable long-horizon mobile manipulation tasks requiring bimanual and whole-body coordination, navigation, and active perception. Results are best viewed on: this https URL
>
---
#### [new 022] CMoE: Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动控制任务，旨在解决复杂地形适应问题。通过引入CMoE框架，增强专家模型的专门化能力，提升机器人在多样化地形中的行走性能。**

- **链接: [https://arxiv.org/pdf/2603.03067](https://arxiv.org/pdf/2603.03067)**

> **作者:** Shihao Ma; Hongjin Chen; Zijun Xu; Yi Zhao; Ke Wu; Ruichen Yang; Leyao Zou; Zhongxue Gan; Wenchao Ding
>
> **摘要:** For effective deployment in real-world environments, humanoid robots must autonomously navigate a diverse range of complex terrains with abrupt transitions. While the Vanilla mixture of experts (MoE) framework is theoretically capable of modeling diverse terrain features, in practice, the gating network exhibits nearly uniform expert activations across different terrains, weakening the expert specialization and limiting the model's expressive power. To address this limitation, we introduce CMoE, a novel single-stage reinforcement learning framework that integrates contrastive learning to refine expert activation distributions. By imposing contrastive constraints, CMoE maximizes the consistency of expert activations within the same terrain while minimizing their similarity across different terrains, thereby encouraging experts to specialize in distinct terrain types. We validated our approach on the Unitree G1 humanoid robot through a series of challenging experiments. Results demonstrate that CMoE enables the robot to traverse continuous steps up to 20 cm high and gaps up to 80 cm wide, while achieving robust and natural gait across diverse mixed terrains, surpassing the limits of existing methods. To support further research and foster community development, we release our code publicly.
>
---
#### [new 023] PathSpace: Rapid continuous map approximation for efficient SLAM using B-Splines in constrained environments
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，旨在解决环境表示效率问题。提出PathSpace框架，利用B样条进行连续地图近似，提升资源利用率。**

- **链接: [https://arxiv.org/pdf/2603.02538](https://arxiv.org/pdf/2603.02538)**

> **作者:** Aduen Benjumea; Andrew Bradley; Alexander Rast; Matthias Rolf
>
> **摘要:** Simultaneous Localization and Mapping (SLAM) plays a crucial role in enabling autonomous vehicles to navigate previously unknown environments. Semantic SLAM mostly extends visual SLAM, leveraging the higher density information available to reason about the environment in a more human-like manner. This allows for better decision making by exploiting prior structural knowledge of the environment, usually in the form of labels. Current semantic SLAM techniques still mostly rely on a dense geometric representation of the environment, limiting their ability to apply constraints based on context. We propose PathSpace, a novel semantic SLAM framework that uses continuous B-splines to represent the environment in a compact manner, while also maintaining and reasoning through the continuous probability density functions required for probabilistic reasoning. This system applies the multiple strengths of B-splines in the context of SLAM to interpolate and fit otherwise discrete sparse environments. We test this framework in the context of autonomous racing, where we exploit pre-specified track characteristics to produce significantly reduced representations at comparable levels of accuracy to traditional landmark based methods and demonstrate its potential in limiting the resources used by a system with minimal accuracy loss.
>
---
#### [new 024] Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids
- **分类: cs.RO**

- **简介: 该论文属于多机械臂协同控制任务，旨在解决多机械臂物理交互中的运动匹配与动态控制问题。提出Rhythm框架，整合运动迁移、强化学习和实际部署系统，实现复杂交互行为的可靠转移。**

- **链接: [https://arxiv.org/pdf/2603.02856](https://arxiv.org/pdf/2603.02856)**

> **作者:** Hongjin Chen; Wei Zhang; Pengfei Li; Shihao Ma; Ke Ma; Yujie Jin; Zijun Xu; Xiaohui Wang; Yupeng Zheng; Zining Wang; Jieru Zhao; Yilun Chen; Wenchao Ding
>
> **摘要:** Realizing interactive whole-body control for multi-humanoid systems is critical for unlocking complex collaborative capabilities in shared environments. Although recent advancements have significantly enhanced the agility of individual robots, bridging the gap to physically coupled multi-humanoid interaction remains challenging, primarily due to severe kinematic mismatches and complex contact dynamics. To address this, we introduce Rhythm, the first unified framework enabling real-world deployment of dual-humanoid systems for complex, physically plausible interactions. Our framework integrates three core components: (1) an Interaction-Aware Motion Retargeting (IAMR) module that generates feasible humanoid interaction references from human data; (2) an Interaction-Guided Reinforcement Learning (IGRL) policy that masters coupled dynamics via graph-based rewards; and (3) a real-world deployment system that enables robust transfer of dual-humanoid interaction. Extensive experiments on physical Unitree G1 robots demonstrate that our framework achieves robust interactive whole-body control, successfully transferring diverse behaviors such as hugging and dancing from simulation to reality.
>
---
#### [new 025] CASSR: Continuous A-Star Search through Reachability for real time footstep planning
- **分类: cs.RO**

- **简介: 该论文提出CASSR框架，解决双足机器人实时步态规划问题。通过结合A*搜索与连续约束，提升规划效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.02989](https://arxiv.org/pdf/2603.02989)**

> **作者:** Jiayi Wang; Steve Tonneau
>
> **摘要:** Footstep planning involves a challenging combinatorial search. Traditional A* approaches require discretising reachability constraints, while Mixed-Integer Programming (MIP) supports continuous formulations but quickly becomes intractable, especially when rotations are included. We present CASSR, a novel framework that recursively propagates convex, continuous formulations of a robot's kinematic constraints within an A* search. Combined with a new cost-to-go heuristic based on the EPA algorithm, CASSR efficiently plans contact sequences of up to 30 footsteps in under 125 ms. Experiments on biped locomotion tasks demonstrate that CASSR outperforms traditional discretised A* by up to a factor of 100, while also surpassing a commercial MIP solver. These results show that CASSR enables fast, reliable, and real-time footstep planning for biped robots.
>
---
#### [new 026] Tracing Back Error Sources to Explain and Mitigate Pose Estimation Failures
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决姿态估计失败问题。提出模块化框架，识别误差来源并针对性修复，提升ICP的鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2603.02881](https://arxiv.org/pdf/2603.02881)**

> **作者:** Loris Schneider; Yitian Shi; Rosa Wolf; Carolin Brenner; Rudolph Triebel; Rania Rayyes
>
> **摘要:** Robust estimation of object poses in robotic manipulation is often addressed using foundational general estimators, that aim to handle diverse error sources naively within a single model. Still, they struggle due to environmental uncertainties, while requiring long inference times and heavy computation. In contrast, we propose a modular, uncertainty-aware framework that attributes pose estimation errors to specific error sources and applies targeted mitigation strategies only when necessary. Instantiated with Iterative Closest Point (ICP) as a simple and lightweight pose estimator, we leverage our framework for real-world robotic grasping tasks. By decomposing pose estimation into failure detection, error attribution, and targeted recovery, we significantly improve the robustness of ICP and achieve competitive performance compared to foundation models, while relying on a substantially simpler and faster pose estimator.
>
---
#### [new 027] Design, Modeling and Direction Control of a Wire-Driven Robotic Fish Based on a 2-DoF Crank-Slider Mechanism
- **分类: cs.RO**

- **简介: 该论文属于机器人鱼设计任务，旨在解决高速与灵活转向的矛盾。通过2-DoF曲柄滑块机构实现推进与转向解耦，提升性能。**

- **链接: [https://arxiv.org/pdf/2603.02851](https://arxiv.org/pdf/2603.02851)**

> **作者:** Yita Wang; Chen Chen; Yicheng Chen; Jinjie Li; Yuichi Motegi; Kenji Ohkuma; Toshihiro Maki; Moju Zhao
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Robotic fish have attracted growing attention in recent years owing to their biomimetic design and potential applications in environmental monitoring and biological surveys. Among robotic fish employing the Body-Caudal Fin (BCF) locomotion pattern, motor-driven actuation is widely adopted. Some approaches utilize multiple servo motors to achieve precise body curvature control, while others employ a brushless motor to drive the tail via wire or rod, enabling higher oscillation and swimming speeds. However, the former approaches typically result in limited swimming speed, whereas the latter suffer from poor maneuverability, with few capable of smooth turning. To address this trade-off, we develop a wire-driven robotic fish equipped with a 2-degree-of-freedom (DoF) crank-slider mechanism that decouples propulsion from steering, enabling both high swimming speed and agile maneuvering. In this paper, we first present the design of the robotic fish, including the elastic skeleton, waterproof structure, and the actuation mechanism that realizes the decoupling. We then establish the actuation modeling and body dynamics to analyze the locomotion behavior. Furthermore, we propose a combined feedforward-feedback control strategy to achieve independent regulation of propulsion and steering. Finally, we validate the feasibility of the design, modeling, and control through a series of prototype experiments, demonstrating swimming, turning, and directional control.
>
---
#### [new 028] Tensegrity Robot Endcap-Ground Contact Estimation with Symmetry-aware Heterogeneous Graph Neural Network
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决 tensegrity 机器人接触状态估计问题。提出 Sym-HGNN 网络，利用本体感觉信息实现无接触传感器的接触状态预测，并提升姿态估计精度。**

- **链接: [https://arxiv.org/pdf/2603.02596](https://arxiv.org/pdf/2603.02596)**

> **作者:** Wenzhe Tong; Yicheng Jiang; Chi Zhang; Maani Ghaffari; Xiaonan Huang
>
> **备注:** Preprint; 7 pages, 5 figures, 3 tables
>
> **摘要:** Tensegrity robots possess lightweight and resilient structures but present significant challenges for state estimation due to compliant and distributed ground contacts. This paper introduces a symmetry-aware heterogeneous graph neural network (Sym-HGNN) that infers contact states directly from proprioceptive measurements, including IMU and cable-length histories, without dedicated contact sensors. The network incorporates the robot's dihedral symmetry $D_3$ into the message-passing process to enhance sample efficiency and generalization. The predicted contacts are integrated into a state-of-the-art contact-aided invariant extended Kalman filter (InEKF) for improved pose estimation. Simulation results demonstrate that the proposed method achieves up to 15% higher accuracy and 5% higher F1-score using only 20% of the training data compared to the CNN and MI-HGNN baselines, while maintaining low-drift and physically consistent state estimation results comparable to ground truth contacts. This work highlights the potential of fully proprioceptive sensing for accurate and robust state estimation in tensegrity robots. Code available at: this https URL
>
---
#### [new 029] Wukong-Omni: Design, Modeling and Control of a Multi-mode Robot for Air, Land, and Underwater Exploration with All-in-One Propulsion Unit
- **分类: cs.RO**

- **简介: 该论文属于多模式机器人设计任务，旨在解决洪水救援中机器人跨域作业的问题。提出Wukong-Omni机器人，实现空、陆、水三域运行，提升 propulsion 效率与推力。**

- **链接: [https://arxiv.org/pdf/2603.02602](https://arxiv.org/pdf/2603.02602)**

> **作者:** Yufan Liu; Rixi Yu; Junjie Li; Yishuai Zeng; Zhenting Wen; Cheng Li; Haifei Zhu; Shikang Lian; Wei Meng; Fumin Zhang
>
> **备注:** 19 pages, 27 figures
>
> **摘要:** In flood disaster rescue scenarios, partially submerged buildings prevent aerial robots from accessing lower levels, limiting mission effectiveness. To address this challenge, this paper presents Wukong-Omni, a novel multimode robot capable of operating across land, air, and underwater using a unified propulsion system. The system is enabled by an innovative mechanical design that allows motor reuse and improves thrust generation. Efficiency and peak thrust are enhanced through simulation and tank-based optimization. Experimental results show a 100 percent improvement in propulsion efficiency and a 150 percent increase in maximum thrust compared with direct installation methods. Dynamic models for the three operating domains are developed, and a unified cross-domain control framework is proposed. Comprehensive experiments validate stable locomotion and smooth transition across domains. Outdoor experiments further demonstrate robustness and adaptability in real-world environments.
>
---
#### [new 030] Watch Your Step: Learning Semantically-Guided Locomotion in Cluttered Environment
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决腿式机器人在杂乱环境中误踩障碍物的问题。通过引入SemLoco框架，结合语义地图与强化学习，提升足部放置安全性。**

- **链接: [https://arxiv.org/pdf/2603.02657](https://arxiv.org/pdf/2603.02657)**

> **作者:** Denan Liang; Yuan Zhu; Ruimeng Liu; Thien-Minh Nguyen; Shenghai Yuan; Lihua Xie
>
> **备注:** Submitted to IROS 2026
>
> **摘要:** Although legged robots demonstrate impressive mobility on rough terrain, using them safely in cluttered environments remains a challenge. A key issue is their inability to avoid stepping on low-lying objects, such as high-cost small devices or cables on flat ground. This limitation arises from a disconnection between high-level semantic understanding and low-level control, combined with errors in elevation maps during real-world operation. To address this, we introduce SemLoco, a Reinforcement Learning (RL) framework designed to avoid obstacles precisely in densely cluttered environments. SemLoco uses a two-stage RL approach that combines both soft and hard constraints and performs pixel-wise foothold safety inference, enabling more accurate foot placement. Additionally, SemLoco integrates a semantic map to assign traversability costs rather than relying solely on geometric data. SemLoco significantly reduces collisions and improves safety around sensitive objects, enabling reliable navigation in situations where traditional controllers would likely cause damage. Experimental results further demonstrate that SemLoco can be effectively applied to more complex, unstructured real-world environments.
>
---
#### [new 031] Robust Tightly-Coupled Filter-Based Monocular Visual-Inertial State Estimation and Graph-Based Evaluation for Autonomous Drone Racing
- **分类: cs.RO**

- **简介: 该论文属于自主无人机竞速中的状态估计任务，解决高速运动下视觉-惯性系统精度与鲁棒性问题。提出ADR-VINS框架，结合误差状态卡尔曼滤波，提升计算效率和抗干扰能力，并引入ADR-FGO进行离线评估。**

- **链接: [https://arxiv.org/pdf/2603.02742](https://arxiv.org/pdf/2603.02742)**

> **作者:** Maulana Bisyir Azhari; Donghun Han; SungJun Park; David Hyunchul Shim
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Autonomous drone racing (ADR) demands state estimation that is simultaneously computationally efficient and resilient to the perceptual degradation experienced during extreme velocity and maneuvers. Traditional frameworks typically rely on conventional visual-inertial pipelines with loosely-coupled gate-based Perspective-n-Points (PnP) corrections that suffer from a rigid requirement for four visible features and information loss in intermediate steps. Furthermore, the absence of GNSS and Motion Capture systems in uninstrumented, competitive racing environments makes the objective evaluation of such systems remarkably difficult. To address these limitations, we propose ADR-VINS, a robust, monocular visual-inertial state estimation framework based on an Error-State Kalman Filter (ESKF) tailored for autonomous drone racing. Our approach integrates direct pixel reprojection errors from gate corners features as innovation terms within the filter. By bypassing intermediate PnP solvers, ADR-VINS maintains valid state updates with as few as two visible corners and utilizes robust reweighting instead of RANSAC-based schemes to handle outliers, enhancing computational efficiency. Furthermore, we introduce ADR-FGO, an offline Factor-Graph Optimization framework to generate high-fidelity reference trajectories that facilitate post-flight performance evaluation and analysis on uninstrumented, GNSS-denied environments. The proposed system is validated using TII-RATM dataset, where ADR-VINS achieves an average RMS translation error of 0.134 m, while ADR-FGO yields 0.060 m as a smoothing-based reference. Finally, ADR-VINS was successfully deployed in the A2RL Drone Championship Season 2, maintaining stable and robust estimation despite noisy detections during high-agility flight at top speeds of 20.9 m/s. We further utilize ADR-FGO for post-flight evaluation in uninstrumented racing environments.
>
---
#### [new 032] Self-supervised Domain Adaptation for Visual 3D Pose Estimation of Nano-drone Racing Gates by Enforcing Geometric Consistency
- **分类: cs.RO**

- **简介: 该论文研究视觉3D姿态估计任务，解决模拟到真实场景的域适应问题。通过自监督方式利用真实图像和无人机里程计数据，提升模型在真实环境中的性能。**

- **链接: [https://arxiv.org/pdf/2603.02936](https://arxiv.org/pdf/2603.02936)**

> **作者:** Nicholas Carlotti; Michele Antonazzi; Elia Cereda; Mirko Nava; Nicola Basilico; Daniele Palossi; Alessandro Giusti
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** We consider the task of visually estimating the relative pose of a drone racing gate in front of a nano-quadrotor, using a convolutional neural network pre-trained on simulated data to regress the gate's pose. Due to the sim-to-real gap, the pre-trained model underperforms in the real world and must be adapted to the target domain. We propose an unsupervised domain adaptation (UDA) approach using only real image sequences collected by the drone flying an arbitrary trajectory in front of a gate; sequences are annotated in a self-supervised fashion with the drone's odometry as measured by its onboard sensors. On this dataset, a state consistency loss enforces that two images acquired at different times yield pose predictions that are consistent with the drone's odometry. Results indicate that our approach outperforms other SoA UDA approaches, has a low mean absolute error in position (x=26, y=28, z=10 cm) and orientation ($\psi$=13${^{\circ}}$), an improvement of 40% in position and 37% in orientation over a baseline. The approach's effectiveness is appreciable with as few as 10 minutes of real-world flight data and yields models with an inference time of 30.4ms (33 fps) when deployed aboard the Crazyflie 2.1 Brushless nano-drone.
>
---
#### [new 033] MMH-Planner: Multi-Mode Hybrid Trajectory Planning Method for UAV Efficient Flight Based on Real-Time Spatial Awareness
- **分类: cs.RO**

- **简介: 该论文属于无人机轨迹规划任务，旨在解决传统算法效率低、适应性差的问题。提出多模式混合规划方法，结合实时环境感知与懒惰重规划策略，提升规划效率和飞行质量。**

- **链接: [https://arxiv.org/pdf/2603.02683](https://arxiv.org/pdf/2603.02683)**

> **作者:** Yinghao Zhao; Chenguang Dai; Liang Lyu; Zhenchao Zhang; Chaozhen Lan; Hong Xie
>
> **摘要:** Motion planning is a critical component of intelligent unmanned systems, enabling their complex autonomous operations. However, current planning algorithms still face limitations in planning efficiency due to inflexible strategies and weak adaptability. To address this, this paper proposes a multi-mode hybrid trajectory planning method for UAVs based on real-time environmental awareness, which dynamically selects the optimal planning model for high-quality trajectory generation in response to environmental changes. First, we introduce a goal-oriented spatial awareness method that rapidly assesses flight safety in the upcoming environments. Second, a multi-mode hybrid trajectory planning mechanism is proposed, which can enhance the planning efficiency by selecting the optimal planning model for trajectory generation based on prior spatial awareness. Finally, we design a lazy replanning strategy that triggers replanning only when necessary to reduce computational resource consumption while maintaining flight quality. To validate the performance of the proposed method, we conducted comprehensive comparative experiments in simulation environments. Results demonstrate that our approach outperforms existing state-of-the-art (SOTA) algorithms across multiple metrics, achieving the best performance particularly in terms of the average number of planning iterations and computational cost per iteration. Furthermore, the effectiveness of our approach is further verified through real-world flight experiments integrated with a self-developed intelligent UAV platform.
>
---
#### [new 034] Architectural HRI: Towards a Robotic Paradigm Shift in Human-Building Interaction
- **分类: cs.RO; cs.HC**

- **简介: 论文探讨建筑与人之间的互动新范式，提出将机器人技术融入建筑设计，以实现空间的动态适应。属于人机交互任务，旨在解决建筑环境智能化与可持续性问题。**

- **链接: [https://arxiv.org/pdf/2603.03052](https://arxiv.org/pdf/2603.03052)**

> **作者:** Alex Binh Vinh Duc Nguyen
>
> **摘要:** Recent advances in sensing, communication, interfaces, control, and robotics are expanding Human-Building Interaction (HBI) beyond adaptive building services and facades toward the physical actuation of architectural space. In parallel, research in robotic furniture, swarm robotics, and shape-changing spaces shows that architectural elements can now be robotically augmented to move, reconfigure, and adapt space. We propose that these advances promise a paradigm shift in HBI, in which multiple building layers physically adapt in synchrony to support occupant needs and sustainability goals more holistically. Conversely, we argue that this emerging paradigm also provides an ideal case for transferring HRI knowledge to unconventional robotic morphologies, including the interpretation of the robot as multiple architectural layers or even as a building. However, this research agenda remains challenged by the temporal, spatial, and social complexity of architectural HRI, and by fragmented knowledge across HCI, environmental psychology, cognitive science, and architecture. We therefore call for interdisciplinary research that unifies the why, what, and how of robotic actuation in architectural forms.
>
---
#### [new 035] CoFL: Continuous Flow Fields for Language-Conditioned Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出CoFL，用于语言引导的导航任务，解决传统模块化或高成本动作生成的问题。通过端到端映射鸟瞰图和语言指令到连续流场，实现平滑、实时的路径规划。**

- **链接: [https://arxiv.org/pdf/2603.02854](https://arxiv.org/pdf/2603.02854)**

> **作者:** Haokun Liu; Zhaoqi Ma; Yicheng Chen; Masaki Kitagawa; Wentao Zhang; Jinjie Li; Moju Zhao
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Language-conditioned navigation pipelines often rely on brittle modular components or costly action-sequence generation. To address these limitations, we present CoFL, an end-to-end policy that directly maps a bird's-eye view (BEV) observation and a language instruction to a continuous flow field for navigation. Instead of predicting discrete action tokens or sampling action chunks via iterative denoising, CoFL outputs instantaneous velocities that can be queried at arbitrary 2D projected locations. Trajectories are obtained by numerical integration of the predicted field, producing smooth motion that remains reactive under closed-loop execution. To enable large-scale training, we build a dataset of over 500k BEV image-instruction pairs, each procedurally annotated with a flow field and a trajectory derived from BEV semantic maps built on Matterport3D and ScanNet. By training on a mixed distribution, CoFL significantly outperforms modular Vision-Language Model (VLM)-based planners and generative policy baselines on strictly unseen scenes. Finally, we deploy CoFL zero-shot in real-world experiments with overhead BEV observations across multiple layouts, maintaining reliable closed-loop control and a high success rate.
>
---
#### [new 036] Learning Therapist Policy from Therapist-Exoskeleton-Patient Interaction
- **分类: cs.RO**

- **简介: 该论文属于康复机器人领域，旨在解决术后康复中治疗师劳动强度大的问题。通过构建PTFF和ST模型，实现对治疗师行为的模拟与辅助，提升康复效率与持续性。**

- **链接: [https://arxiv.org/pdf/2603.02458](https://arxiv.org/pdf/2603.02458)**

> **作者:** Grayson Snyder; Lorenzo Vianello; Levi Hargrove; Matthew L. Elwin; Jose Pons
>
> **备注:** Accepted at IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Post-stroke rehabilitation is often necessary for patients to regain proper walking gait. However, the typical therapy process can be exhausting and physically demanding for therapists, potentially reducing therapy intensity, duration, and consistency over time. We propose a Patient-Therapist Force Field (PTFF) to visualize therapist responses to patient kinematics and a Synthetic Therapist (ST) machine learning model to support the therapist in dyadic robot-mediated physical interaction therapy. The first encodes patient and therapist stride kinematics into a shared low-dimensional latent manifold using a Variational Autoencoder (VAE) and models their interaction through a Gaussian Mixture Model (GMM), which learns a probabilistic vector field mapping patient latent states to therapist responses. This representation visualizes patient-therapist interaction dynamics to inform therapy strategies and robot controller design. The latter is implemented as a Long Short-Term Memory (LSTM) network trained on patient-therapist interaction data to predict therapist-applied joint torques from patient kinematics. Trained and validated using leave-one-out cross-validation across eight post-stroke patients, the model was integrated into a ROS-based exoskeleton controller to generate real-time torque assistance based on predicted therapist responses. Offline results and preliminary testing indicate the potential of their use as an alternative approach to post-stroke exoskeleton therapy. The PTFF provides understanding of the therapist's actions while the ST frees the human therapist from the exoskeleton, allowing them to continuously monitor the patient's nuanced condition.
>
---
#### [new 037] From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?
- **分类: cs.RO**

- **简介: 该论文属于机器人认知任务，旨在探索LLM在具身机器人中的应用。研究提出一种认知架构，用LLM进行规划与推理，评估其在家庭环境中的任务执行能力。**

- **链接: [https://arxiv.org/pdf/2603.03148](https://arxiv.org/pdf/2603.03148)**

> **作者:** Shinas Shaji; Fabian Huppertz; Alex Mitrevski; Sebastian Houben
>
> **备注:** Accepted for publication at the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** In order to flexibly act in an everyday environment, a robotic agent needs a variety of cognitive capabilities that enable it to reason about plans and perform execution recovery. Large language models (LLMs) have been shown to demonstrate emergent cognitive aspects, such as reasoning and language understanding; however, the ability to control embodied robotic agents requires reliably bridging high-level language to low-level functionalities for perception and control. In this paper, we investigate the extent to which an LLM can serve as a core component for planning and execution reasoning in a cognitive robot architecture. For this purpose, we propose a cognitive architecture in which an agentic LLM serves as the core component for planning and reasoning, while components for working and episodic memories support learning from experience and adaptation. An instance of the architecture is then used to control a mobile manipulator in a simulated household environment, where environment interaction is done through a set of high-level tools for perception, reasoning, navigation, grasping, and placement, all of which are made available to the LLM-based agent. We evaluate our proposed system on two household tasks (object placement and object swapping), which evaluate the agent's reasoning, planning, and memory utilisation. The results demonstrate that the LLM-driven agent can complete structured tasks and exhibits emergent adaptation and memory-guided planning, but also reveal significant limitations, such as hallucinations about the task success and poor instruction following by refusing to acknowledge and complete sequential tasks. These findings highlight both the potential and challenges of employing LLMs as embodied cognitive controllers for autonomous robots.
>
---
#### [new 038] IMR-LLM: Industrial Multi-Robot Task Planning and Program Generation using Large Language Models
- **分类: cs.RO**

- **简介: 该论文属于工业多机器人任务规划与程序生成领域，解决工业场景中复杂任务的协调与执行问题。通过结合大语言模型与确定性方法，生成高效任务计划和可执行程序。**

- **链接: [https://arxiv.org/pdf/2603.02669](https://arxiv.org/pdf/2603.02669)**

> **作者:** Xiangyu Su; Juzhan Xu; Oliver van Kaick; Kai Xu; Ruizhen Hu
>
> **摘要:** In modern industrial production, multiple robots often collaborate to complete complex manufacturing tasks. Large language models (LLMs), with their strong reasoning capabilities, have shown potential in coordinating robots for simple household and manipulation tasks. However, in industrial scenarios, stricter sequential constraints and more complex dependencies within tasks present new challenges for LLMs. To address this, we propose IMR-LLM, a novel LLM-driven Industrial Multi-Robot task planning and program generation framework. Specifically, we utilize LLMs to assist in constructing disjunctive graphs and employ deterministic solving methods to obtain a feasible and efficient high-level task plan. Based on this, we use a process tree to guide LLMs to generate executable low-level programs. Additionally, we create IMR-Bench, a challenging benchmark that encompasses multi-robot industrial tasks across three levels of complexity. Experimental results indicate that our method significantly surpasses existing methods across all evaluation metrics.
>
---
#### [new 039] Give me scissors: Collision-Free Dual-Arm Surgical Assistive Robot for Instrument Delivery
- **分类: cs.RO; cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于手术机器人任务，旨在解决器械传递中的碰撞和路径规划问题。提出一种双臂机器人系统，通过视觉语言模型生成轨迹，并集成实时避障方法，实现安全高效的器械传递。**

- **链接: [https://arxiv.org/pdf/2603.02553](https://arxiv.org/pdf/2603.02553)**

> **作者:** Xuejin Luo; Shiquan Sun; Runshi Zhang; Ruizhi Zhang; Junchen Wang
>
> **备注:** 8 pages, 10 figures. Accepted by IEEE International Conference on Robotics and Automation (ICRA), 2026
>
> **摘要:** During surgery, scrub nurses are required to frequently deliver surgical instruments to surgeons, which can lead to physical fatigue and decreased focus. Robotic scrub nurses provide a promising solution that can replace repetitive tasks and enhance efficiency. Existing research on robotic scrub nurses relies on predefined paths for instrument delivery, which limits their generalizability and poses safety risks in dynamic environments. To address these challenges, we present a collision-free dual-arm surgical assistive robot capable of performing instrument delivery. A vision-language model is utilized to automatically generate the robot's grasping and delivery trajectories in a zero-shot manner based on surgeons' instructions. A real-time obstacle minimum distance perception method is proposed and integrated into a unified quadratic programming framework. This framework ensures reactive obstacle avoidance and self-collision prevention during the dual-arm robot's autonomous movement in dynamic environments. Extensive experimental validations demonstrate that the proposed robotic system achieves an 83.33% success rate in surgical instrument delivery while maintaining smooth, collision-free movement throughout all trials. The project page and source code are available at this https URL.
>
---
#### [new 040] ULTRA: Unified Multimodal Control for Autonomous Humanoid Whole-Body Loco-Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人形机器人全身运动控制任务，旨在解决自主、通用的全身操作难题。提出ULTRA框架，结合物理驱动的运动迁移和多模态控制器，实现从感知和任务指令生成行为。**

- **链接: [https://arxiv.org/pdf/2603.03279](https://arxiv.org/pdf/2603.03279)**

> **作者:** Xialin He; Sirui Xu; Xinyao Li; Runpei Dong; Liuyu Bian; Yu-Xiong Wang; Liang-Yan Gui
>
> **备注:** Project Page: this https URL
>
> **摘要:** Achieving autonomous and versatile whole-body loco-manipulation remains a central barrier to making humanoids practically useful. Yet existing approaches are fundamentally constrained: retargeted data are often scarce or low-quality; methods struggle to scale to large skill repertoires; and, most importantly, they rely on tracking predefined motion references rather than generating behavior from perception and high-level task specifications. To address these limitations, we propose ULTRA, a unified framework with two key components. First, we introduce a physics-driven neural retargeting algorithm that translates large-scale motion capture to humanoid embodiments while preserving physical plausibility for contact-rich interactions. Second, we learn a unified multimodal controller that supports both dense references and sparse task specifications, under sensing ranging from accurate motion-capture state to noisy egocentric visual inputs. We distill a universal tracking policy into this controller, compress motor skills into a compact latent space, and apply reinforcement learning finetuning to expand coverage and improve robustness under out-of-distribution scenarios. This enables coordinated whole-body behavior from sparse intent without test-time reference motions. We evaluate ULTRA in simulation and on a real Unitree G1 humanoid. Results show that ULTRA generalizes to autonomous, goal-conditioned whole-body loco-manipulation from egocentric perception, consistently outperforming tracking-only baselines with limited skills.
>
---
#### [new 041] COLREGs Compliant Collision Avoidance and Grounding Prevention for Autonomous Marine Navigation
- **分类: cs.RO; math.OC**

- **简介: 该论文属于自主船舶导航任务，解决复杂水域中碰撞避让与触礁问题。通过整合COLREGs规则和水深数据，提出一种实时优化方法，确保安全、合规的航行。**

- **链接: [https://arxiv.org/pdf/2603.02484](https://arxiv.org/pdf/2603.02484)**

> **作者:** Mayur S. Patil; Nataraj Sudharsan; Veneela Ammula; Jude Tomdio; Jin Wang; Michael Kei; Sivakumar Rathinam; Prabhakar R. Pagilla
>
> **摘要:** Maritime Autonomous Surface Ships (MASS) are increasingly regarded as a promising solution to address crew shortages, improve navigational safety, and improve operational efficiency in the maritime industry. Nevertheless, the reliable deployment of MASS in real-world environments remains a significant challenge, particularly in congested waters where the majority of maritime accidents occur. This emphasizes the need for safe and regulation-aware motion planning strategies for MASS that are capable of operating under dynamic maritime conditions. This paper presents a unified motion planning method for MASS that achieves real time collision avoidance, compliance with International Regulations for Preventing Collisions at Sea (COLREGs), and grounding prevention. The proposed work introduces a convex optimization method that integrates velocity obstacle-based (VO) collision constraints, COLREGs-based directional constraints, and bathymetry-based grounding constraints to generate computationally efficient, rule-compliant optimal velocity selection. To enhance robustness, the classical VO method is extended to consider uncertainty in the position and velocity estimates of the target vessel. Unnavigable shallow water regions obtained from bathymetric data, which are inherently nonconvex, are approximated via convex geometries using a integer linear programming (ILP), allowing grounding constraints to be incorporated into the motion planning. The resulting optimization generates optimal and dynamically feasible input velocities that meet collision avoidance, regulatory compliance, kinodynamic limits, and grounding prevention requirements. Simulation results involving multi-vessel encounters demonstrate the effectiveness of the proposed method in producing safe and regulation-compliant maneuvers, highlighting the suitability of the proposed approach for real time autonomous maritime navigation.
>
---
#### [new 042] cuNRTO: GPU-Accelerated Nonlinear Robust Trajectory Optimization
- **分类: cs.RO; cs.DC; eess.SY**

- **简介: 该论文属于轨迹优化任务，解决不确定环境下自主系统安全规划问题。提出cuNRTO框架，利用GPU加速非线性鲁棒轨迹优化，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2603.02642](https://arxiv.org/pdf/2603.02642)**

> **作者:** Jiawei Wang; Arshiya Taj Abdul; Evangelos A. Theodorou
>
> **摘要:** Robust trajectory optimization enables autonomous systems to operate safely under uncertainty by computing control policies that satisfy the constraints for all bounded disturbances. However, these problems often lead to large Second Order Conic Programming (SOCP) constraints, which are computationally expensive. In this work, we propose the CUDA Nonlinear Robust Trajectory Optimization (cuNRTO) framework by introducing two dynamic optimization architectures that have direct application to robust decision-making and are implemented on CUDA. The first architecture, NRTO-DR, leverages the Douglas-Rachford (DR) splitting method to solve the SOCP inner subproblems of NRTO, thereby significantly reducing the computational burden through parallel SOCP projections and sparse direct solves. The second architecture, NRTO-FullADMM, is a novel variant that further exploits the problem structure to improve scalability using the Alternating Direction Method of Multipliers (ADMM). Finally, we provide GPU implementation of the proposed methodologies using custom CUDA kernels for SOC projection steps and cuBLAS GEMM chains for feedback gain updates. We validate the performance of cuNRTO through simulated experiments on unicycle, quadcopter, and Franka manipulator models, demonstrating speedup up to 139.6$\times$.
>
---
#### [new 043] What Capable Agents Must Know: Selection Theorems for Robust Decision-Making under Uncertainty
- **分类: cs.LG; cs.AI; cs.RO; q-bio.NC; stat.ML**

- **简介: 该论文属于强化学习领域，研究智能体在不确定性下如何实现稳健决策。通过证明选择定理，揭示了低平均损失迫使智能体构建预测性内部状态的必要性。**

- **链接: [https://arxiv.org/pdf/2603.02491](https://arxiv.org/pdf/2603.02491)**

> **作者:** Aran Nayebi
>
> **备注:** 18 pages
>
> **摘要:** As artificial agents become increasingly capable, what internal structure is *necessary* for an agent to act competently under uncertainty? Classical results show that optimal control can be *implemented* using belief states or world models, but not that such representations are required. We prove quantitative "selection theorems" showing that low *average-case regret* on structured families of action-conditioned prediction tasks forces an agent to implement a predictive, structured internal state. Our results cover stochastic policies, partial observability, and evaluation under task distributions, without assuming optimality, determinism, or access to an explicit model. Technically, we reduce predictive modeling to binary "betting" decisions and show that regret bounds limit probability mass on suboptimal bets, enforcing the predictive distinctions needed to separate high-margin outcomes. In fully observed settings, this yields approximate recovery of the interventional transition kernel; under partial observability, it implies necessity of belief-like memory and predictive state, addressing an open question in prior world-model recovery work.
>
---
#### [new 044] Scalar-Measurement Attitude Estimation on $\mathbf{SO}(3)$ with Bias Compensation
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于姿态估计任务，解决仅使用标量测量进行可靠姿态估计的问题。提出SO(3)上的非线性观测器，实现陀螺仪偏差补偿和稳定估计。**

- **链接: [https://arxiv.org/pdf/2603.02478](https://arxiv.org/pdf/2603.02478)**

> **作者:** Alessandro Melis; Tarek Bouazza; Hassan Alnahhal; Sifeddine Benahmed; Soulaimane Berkane; Tarek Hamel
>
> **备注:** 9 pages, 4 figures. Submitted to ICRA 2026
>
> **摘要:** Attitude estimation methods typically rely on full vector measurements from inertial sensors such as accelerometers and magnetometers. This paper shows that reliable estimation can also be achieved using only scalar measurements, which naturally arise either as components of vector readings or as independent constraints from other sensing modalities. We propose nonlinear deterministic observers on $\mathbf{SO}(3)$ that incorporate gyroscope bias compensation and guarantee uniform local exponential stability under suitable observability conditions. A key feature of the framework is its robustness to partial sensing: accurate estimation is maintained even when only a subset of vector components is available. Experimental validation on the BROAD dataset confirms consistent performance across progressively reduced measurement configurations, with estimation errors remaining small even under severe information loss. To the best of our knowledge, this is the first work to establish fundamental observability results showing that two scalar measurements under suitable excitation suffice for attitude estimation, and that three are enough in the static case. These results position scalar-measurement-based observers as a practical and reliable alternative to conventional vector-based approaches.
>
---
#### [new 045] Strategic Shaping of Human Prosociality: A Latent-State POMDP Framework
- **分类: cs.HC; cs.RO; eess.SY**

- **简介: 该论文属于人机协作任务，旨在通过机器人策略影响人类的亲社会行为。工作包括构建隐状态POMDP模型，学习并优化合作策略，提升团队表现和人类合作行为。**

- **链接: [https://arxiv.org/pdf/2603.02379](https://arxiv.org/pdf/2603.02379)**

> **作者:** Zahra Zahedi; Xinyue Hu; Shashank Mehrotra; Mark Steyvers; Kumar Akash
>
> **备注:** This article has been published in IEEE Robotics and Automation Letters. this https URL
>
> **摘要:** We propose a decision-theoretic framework in which a robot strategically can shape inferred human's prosocial state during repeated interactions. Modeling the human's prosociality as a latent state that evolves over time, the robot learns to infer and influence this state through its own actions, including helping and signaling. We formalize this as a latent-state POMDP with limited observations and learn the transition and observation dynamics using expectation maximization. The resulting belief-based policy balances task and social objectives, selecting actions that maximize long-term cooperative outcomes. We evaluate the model using data from user studies and show that the learned policy outperforms baseline strategies in both team performance and increasing observed human cooperative behavior.
>
---
#### [new 046] VLMFusionOcc3D: VLM Assisted Multi-Modal 3D Semantic Occupancy Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，旨在解决自动驾驶中稀疏几何网格的语义模糊和恶劣天气下的性能下降问题。通过引入VLM增强的多模态框架，提升预测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.02609](https://arxiv.org/pdf/2603.02609)**

> **作者:** A. Enes Doruk; Hasan F. Ates
>
> **摘要:** This paper introduces VLMFusionOcc3D, a robust multimodal framework for dense 3D semantic occupancy prediction in autonomous driving. Current voxel-based occupancy models often struggle with semantic ambiguity in sparse geometric grids and performance degradation under adverse weather conditions. To address these challenges, we leverage the rich linguistic priors of Vision-Language Models (VLMs) to anchor ambiguous voxel features to stable semantic concepts. Our framework initiates with a dual-branch feature extraction pipeline that projects multi-view images and LiDAR point clouds into a unified voxel space. We propose Instance-driven VLM Attention (InstVLM), which utilizes gated cross-attention and LoRA-adapted CLIP embeddings to inject high-level semantic and geographic priors directly into the 3D voxels. Furthermore, we introduce Weather-Aware Adaptive Fusion (WeathFusion), a dynamic gating mechanism that utilizes vehicle metadata and weather-conditioned prompts to re-weight sensor contributions based on real-time environmental reliability. To ensure structural consistency, a Depth-Aware Geometric Alignment (DAGA) loss is employed to align dense camera-derived geometry with sparse, spatially accurate LiDAR returns. Extensive experiments on the nuScenes and SemanticKITTI datasets demonstrate that our plug-and-play modules consistently enhance the performance of state-of-the-art voxel-based baselines. Notably, our approach achieves significant improvements in challenging weather scenarios, offering a scalable and robust solution for complex urban navigation.
>
---
#### [new 047] Retrieval-Augmented Robots via Retrieve-Reason-Act
- **分类: cs.AI; cs.RO**

- **简介: 论文提出Retrieval-Augmented Robotics（RAR），解决机器人在零样本环境下缺乏程序知识的问题。通过检索视觉手册并生成执行计划，提升机器人任务执行能力。属于机器人任务规划与信息检索交叉领域。**

- **链接: [https://arxiv.org/pdf/2603.02688](https://arxiv.org/pdf/2603.02688)**

> **作者:** Izat Temiraliev; Diji Yang; Yi Zhang
>
> **摘要:** To achieve general-purpose utility, we argue that robots must evolve from passive executors into active Information Retrieval users. In strictly zero-shot settings where no prior demonstrations exist, robots face a critical information gap, such as the exact sequence required to assemble a complex furniture kit, that cannot be satisfied by internal parametric knowledge (common sense) or past internal memory. While recent robotic works attempt to use search before action, they primarily focus on retrieving past kinematic trajectories (analogous to searching internal memory) or text-based safety rules (searching for constraints). These approaches fail to address the core information need of active task construction: acquiring unseen procedural knowledge from external, unstructured documentation. In this paper, we define the paradigm as Retrieval-Augmented Robotics (RAR), empowering the robot with the information-seeking capability that bridges the gap between visual documentation and physical actuation. We formulate the task execution as an iterative Retrieve-Reason-Act loop: the robot or embodied agent actively retrieves relevant visual procedural manuals from an unstructured corpus, grounds the abstract 2D diagrams to 3D physical parts via cross-modal alignment, and synthesizes executable plans. We validate this paradigm on a challenging long-horizon assembly benchmark. Our experiments demonstrate that grounding robotic planning in retrieved visual documents significantly outperforms baselines relying on zero-shot reasoning or few-shot example retrieval. This work establishes the basis of RAR, extending the scope of Information Retrieval from answering user queries to driving embodied physical actions.
>
---
#### [new 048] Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures
- **分类: cs.PF; cs.AI; cs.AR; cs.RO**

- **简介: 该论文属于边缘AI任务，研究VLA模型在边缘设备上的性能瓶颈。针对高延迟问题，分析了动作生成阶段的内存限制，并探讨了未来硬件优化方向。**

- **链接: [https://arxiv.org/pdf/2603.02271](https://arxiv.org/pdf/2603.02271)**

> **作者:** Manoj Vishwanathan; Suvinay Subramanian; Anand Raghunathan
>
> **备注:** 3 Pages 4 Figures for Workshop paper
>
> **摘要:** Vision-Language-Action (VLA) models are an emerging class of workloads critical for robotics and embodied AI at the edge. As these models scale, they demonstrate significant capability gains, yet they must be deployed locally to meet the strict latency requirements of real-time applications. This paper characterizes VLA performance on two generations of edge hardware, viz. the Nvidia Jetson Orin and Thor platforms. Using MolmoAct-7B, a state-of-the-art VLA model, we identify a primary execution bottleneck: up to 75% of end-to-end latency is consumed by the memory-bound action-generation phase. Through analytical modeling and simulations, we project the hardware requirements for scaling to 100B parameter models. We also explore the impact of high-bandwidth memory technologies and processing-in-memory (PIM) as promising future pathways in edge systems for embodied AI.
>
---
#### [new 049] Chain of World: World Model Thinking in Latent Motion
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出CoWVLA，解决视觉-语言-动作模型在时间推理和动态建模上的不足。通过结合世界模型与潜在运动表示，提升机器人视觉运动学习效率。**

- **链接: [https://arxiv.org/pdf/2603.03195](https://arxiv.org/pdf/2603.03195)**

> **作者:** Fuxiang Yang; Donglin Di; Lulu Tang; Xuancheng Zhang; Lei Fan; Hao Li; Chen Wei; Tonghua Su; Baorui Ma
>
> **备注:** Accepted by CVPR2026. Project page: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models are a promising path toward embodied intelligence, yet they often overlook the predictive and temporal-causal structure underlying visual dynamics. World-model VLAs address this by predicting future frames, but waste capacity reconstructing redundant backgrounds. Latent-action VLAs encode frame-to-frame transitions compactly, but lack temporally continuous dynamic modeling and world knowledge. To overcome these limitations, we introduce CoWVLA (Chain-of-World VLA), a new "Chain of World" paradigm that unifies world-model temporal reasoning with a disentangled latent motion representation. First, a pretrained video VAE serves as a latent motion extractor, explicitly factorizing video segments into structure and motion latents. Then, during pre-training, the VLA learns from an instruction and an initial frame to infer a continuous latent motion chain and predict the segment's terminal frame. Finally, during co-fine-tuning, this latent dynamic is aligned with discrete action prediction by jointly modeling sparse keyframes and action sequences in a unified autoregressive decoder. This design preserves the world-model benefits of temporal reasoning and world knowledge while retaining the compactness and interpretability of latent actions, enabling efficient visuomotor learning. Extensive experiments on robotic simulation benchmarks show that CoWVLA outperforms existing world-model and latent-action approaches and achieves moderate computational efficiency, highlighting its potential as a more effective VLA pretraining paradigm. The project website can be found at this https URL.
>
---
#### [new 050] Improving Diffusion Planners by Self-Supervised Action Gating with Energies
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决扩散规划中因局部动态不一致导致的执行脆弱问题。提出SAGE方法，在推理时通过能量惩罚不一致计划，提升性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.02650](https://arxiv.org/pdf/2603.02650)**

> **作者:** Yuan Lu; Dongqi Han; Yansen Wang; Dongsheng Li
>
> **摘要:** Diffusion planners are a strong approach for offline reinforcement learning, but they can fail when value-guided selection favours trajectories that score well yet are locally inconsistent with the environment dynamics, resulting in brittle execution. We propose Self-supervised Action Gating with Energies (SAGE), an inference-time re-ranking method that penalises dynamically inconsistent plans using a latent consistency signal. SAGE trains a Joint-Embedding Predictive Architecture (JEPA) encoder on offline state sequences and an action-conditioned latent predictor for short horizon transitions. At test time, SAGE assigns each sampled candidate an energy given by its latent prediction error and combines this feasibility score with value estimates to select actions. SAGE can integrate into existing diffusion planning pipelines that can sample trajectories and select actions via value scoring; it requires no environment rollouts and no policy re-training. Across locomotion, navigation, and manipulation benchmarks, SAGE improves the performance and robustness of diffusion planners.
>
---
#### [new 051] TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决VLM在动态空间任务中的性能不足问题。通过引入拓扑结构增强模型的空间推理能力，提升导航效果。**

- **链接: [https://arxiv.org/pdf/2603.02972](https://arxiv.org/pdf/2603.02972)**

> **作者:** Jiaxing Liu; Zexi Zhang; Xiaoyan Li; Boyue Wang; Yongli Hu; Baocai Yin
>
> **摘要:** Vision-Language Navigation (VLN) presents a unique challenge for Large Vision-Language Models (VLMs) due to their inherent architectural mismatch: VLMs are primarily pretrained on static, disembodied vision-language tasks, which fundamentally clash with the dynamic, embodied, and spatially-structured nature of navigation. Existing large-model-based methods often resort to converting rich visual and spatial information into text, forcing models to implicitly infer complex visual-topological relationships or limiting their global action capabilities. To bridge this gap, we propose TagaVLM (Topology-Aware Global Action reasoning), an end-to-end framework that explicitly injects topological structures into the VLM backbone. To introduce topological edge information, Spatial Topology Aware Residual Attention (STAR-Att) directly integrates it into the VLM's self-attention mechanism, enabling intrinsic spatial reasoning while preserving pretrained knowledge. To enhance topological node information, an Interleaved Navigation Prompt strengthens node-level visual-text alignment. Finally, with the embedded topological graph, the model is capable of global action reasoning, allowing for robust path correction. On the R2R benchmark, TagaVLM achieves state-of-the-art performance among large-model-based methods, with a Success Rate (SR) of 51.09% and SPL of 47.18 in unseen environments, outperforming prior work by 3.39% in SR and 9.08 in SPL. This demonstrates that, for embodied spatial reasoning, targeted enhancements on smaller open-source VLMs can be more effective than brute-force model scaling. The code will be released upon this http URL page: this https URL
>
---
#### [new 052] Diffusion-MPC in Discrete Domains: Feasibility Constraints, Horizon Effects, and Critic Alignment: Case study with Tetris
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究扩散模型在离散领域（如Tetris）的模型预测控制问题，通过可行性约束采样、重排序策略和计算资源分析，提升决策质量。**

- **链接: [https://arxiv.org/pdf/2603.02348](https://arxiv.org/pdf/2603.02348)**

> **作者:** Haochuan Kevin Wang
>
> **备注:** 7 pages, 3 figures, 2 tables. Includes regret diagnostics and compute-quality frontier analysis. Code and experiment configurations available in the Diffusion-Tetris repository
>
> **摘要:** We study diffusion-based model predictive control (Diffusion-MPC) in discrete combinatorial domains using Tetris as a case study. Our planner samples candidate placement sequences with a MaskGIT-style discrete denoiser and selects actions via reranking. We analyze three key factors: (1) feasibility-constrained sampling via logit masking over valid placements, (2) reranking strategies using a heuristic score, a pretrained DQN critic, and a hybrid combination, and (3) compute scaling in candidate count and planning horizon. We find that feasibility masking is necessary in discrete domains, removing invalid action mass (46%) and yielding a 6.8% improvement in score and 5.6% improvement in survival over unconstrained sampling. Naive DQN reranking is systematically misaligned with rollout quality, producing high decision regret (mean 17.6, p90 36.6). Shorter planning horizons outperform longer ones under sparse and delayed rewards, suggesting uncertainty compounding in long imagined rollouts. Overall, compute choices (K, H) determine dominant failure modes: small K limits candidate quality, while larger H amplifies misranking and model mismatch. Our findings highlight structural challenges of diffusion planners in discrete environments and provide practical diagnostics for critic integration.
>
---
#### [new 053] LLM-MLFFN: Multi-Level Autonomous Driving Behavior Feature Fusion via Large Language Model
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶行为分类任务，旨在解决传统方法在复杂环境中的可解释性和鲁棒性不足问题。通过融合多级特征和语言模型语义抽象，提升分类准确率。**

- **链接: [https://arxiv.org/pdf/2603.02528](https://arxiv.org/pdf/2603.02528)**

> **作者:** Xiangyu Li; Tianyi Wang; Xi Cheng; Rakesh Chowdary Machineni; Zhaomiao Guo; Sikai Chen; Junfeng Jiao; Christian Claudel
>
> **摘要:** Accurate classification of autonomous vehicle (AV) driving behaviors is critical for safety validation, performance diagnosis, and traffic integration analysis. However, existing approaches primarily rely on numerical time-series modeling and often lack semantic abstraction, limiting interpretability and robustness in complex traffic environments. This paper presents LLM-MLFFN, a novel large language model (LLM)-enhanced multi-level feature fusion network designed to address the complexities of multi-dimensional driving data. The proposed LLM-MLFFN framework integrates priors from largescale pre-trained models and employs a multi-level approach to enhance classification accuracy. LLM-MLFFN comprises three core components: (1) a multi-level feature extraction module that extracts statistical, behavioral, and dynamic features to capture the quantitative aspects of driving behaviors; (2) a semantic description module that leverages LLMs to transform raw data into high-level semantic features; and (3) a dual-channel multi-level feature fusion network that combines numerical and semantic features using weighted attention mechanisms to improve robustness and prediction accuracy. Evaluation on the Waymo open trajectory dataset demonstrates the superior performance of the proposed LLM-MLFFN, achieving a classification accuracy of over 94%, surpassing existing machine learning models. Ablation studies further validate the critical contributions of multi-level fusion, feature extraction strategies, and LLM-derived semantic reasoning. These results suggest that integrating structured feature modeling with language-driven semantic abstraction provides a principled and interpretable pathway for robust autonomous driving behavior classification.
>
---
#### [new 054] Real-Time Generative Policy via Langevin-Guided Flow Matching for Autonomous Driving
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决生成式策略在自动驾驶中推理延迟高的问题。通过引入流匹配和朗之万动力学，提出DACER-F算法，在保证实时性的同时提升性能。**

- **链接: [https://arxiv.org/pdf/2603.02613](https://arxiv.org/pdf/2603.02613)**

> **作者:** Tianze Zhu; Yinuo Wang; Wenjun Zou; Tianyi Zhang; Likun Wang; Letian Tao; Feihong Zhang; Yao Lyu; Shengbo Eben Li
>
> **摘要:** Reinforcement learning (RL) is a fundamental methodology in autonomous driving systems, where generative policies exhibit considerable potential by leveraging their ability to model complex distributions to enhance exploration. However, their inherent high inference latency severely impedes their deployment in real-time decision-making and control. To address this issue, we propose diffusion actor-critic with entropy regulator via flow matching (DACER-F) by introducing flow matching into online RL, enabling the generation of competitive actions in a single inference step. By leveraging Langevin dynamics and gradients of the Q-function, DACER-F dynamically optimizes actions from experience replay toward a target distribution that balances high Q-value information with exploratory behavior. The flow policy is then trained to efficiently learn a mapping from a simple prior distribution to this dynamic target. In complex multi-lane and intersection simulations, DACER-F outperforms baselines diffusion actor-critic with entropy regulator (DACER) and distributional soft actor-critic (DSAC), while maintaining an ultra-low inference latency. DACER-F further demonstrates its scalability on standard RL benchmark DeepMind Control Suite (DMC), achieving a score of 775.8 in the humanoid-stand task and surpassing prior methods. Collectively, these results establish DACER-F as a high-performance and computationally efficient RL algorithm.
>
---
#### [new 055] The Alignment Flywheel: A Governance-Centric Hybrid MAS for Architecture-Agnostic Safety
- **分类: cs.MA; cs.LG; cs.RO**

- **简介: 该论文属于自主系统安全任务，解决高能力但不可靠系统的安全治理问题。提出一种混合多智能体架构，分离决策与安全治理，实现可审计、可更新的系统监督。**

- **链接: [https://arxiv.org/pdf/2603.02259](https://arxiv.org/pdf/2603.02259)**

> **作者:** Elias Malomgré; Pieter Simoens
>
> **摘要:** Multi-agent systems provide mature methodologies for role decomposition, coordination, and normative governance, capabilities that remain essential as increasingly powerful autonomous decision components are embedded within agent-based systems. While learned and generative models substantially expand system capability, their safety behavior is often entangled with training, making it opaque, difficult to audit, and costly to update after deployment. This paper formalizes the Alignment Flywheel as a governance-centric hybrid MAS architecture that decouples decision generation from safety governance. A Proposer, representing any autonomous decision component, generates candidate trajectories, while a Safety Oracle returns raw safety signals through a stable interface. An enforcement layer applies explicit risk policy at runtime, and a governance MAS supervises the Oracle through auditing, uncertainty-driven verification, and versioned refinement. The central engineering principle is patch locality: many newly observed safety failures can be mitigated by updating the governed oracle artifact and its release pipeline rather than retracting or retraining the underlying decision component. The architecture is implementation-agnostic with respect to both the Proposer and the Safety Oracle, and specifies the roles, artifacts, protocols, and release semantics needed for runtime gating, audit intake, signed patching, and staged rollout across distributed deployments. The result is a hybrid MAS engineering framework for integrating highly capable but fallible autonomous systems under explicit, version-controlled, and auditable oversight.
>
---
## 更新

#### [replaced 001] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于 embodied AI 领域，解决物理轨迹收集成本高的问题。通过桌面游戏数据预训练，提升机器人任务性能，实现有效迁移。**

- **链接: [https://arxiv.org/pdf/2510.05684](https://arxiv.org/pdf/2510.05684)**

> **作者:** Suhwan Choi; Jaeyoon Jung; Haebin Seong; Minchan Kim; Minyeong Kim; Yongjun Cho; Yoonshik Kim; Yubeen Park; Youngjae Yu; Yunsung Lee
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations and 1K+ hours of pseudo-labeled gameplay), our 1B-parameter model achieves 96.6% success on LIBERO manipulation and 83.3% on CANVAS navigation, matching or surpassing models up to 7x larger, such as \pi_{0} (3.3B) and OpenVLA (7B). These results demonstrate that sensorimotor primitives learned from digital interactions transfer effectively to real-world physical tasks, establishing desktop pretraining as a practical paradigm for embodied AI. All resources are publicly available at this https URL.
>
---
#### [replaced 002] Self-Improving Loops for Visual Robotic Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SILVR方法，用于视觉机器人规划任务，解决未知任务泛化问题。通过自收集数据迭代优化模型，提升性能与样本效率。**

- **链接: [https://arxiv.org/pdf/2506.06658](https://arxiv.org/pdf/2506.06658)**

> **作者:** Calvin Luo; Zilai Zeng; Mingxi Jia; Yilun Du; Chen Sun
>
> **备注:** ICLR 2026. Project Page: this https URL
>
> **摘要:** Video generative models trained on expert demonstrations have been utilized as performant text-conditioned visual planners for solving robotic tasks. However, generalization to unseen tasks remains a challenge. Whereas improved generalization may be facilitated by leveraging learned prior knowledge from additional pre-collected offline data sources, such as web-scale video datasets, in the era of experience we aim to design agents that can continuously improve in an online manner from self-collected behaviors. In this work we thus propose the Self-Improving Loops for Visual Robotic Planning (SILVR), where an in-domain video model iteratively updates itself on self-produced trajectories, and steadily improves its performance for a specified task of interest. We apply SILVR to a diverse suite of MetaWorld tasks, as well as two manipulation tasks on a real robot arm, and find that performance improvements continuously emerge over multiple iterations for novel tasks unseen during initial in-domain video model training. We demonstrate that SILVR is robust in the absence of human-provided ground-truth reward functions or expert-quality demonstrations, and is preferable to alternate approaches that utilize online experience in terms of performance and sample efficiency.
>
---
#### [replaced 003] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决大规模环境中探索效率低的问题。通过分析策略多样性对学习的影响，提出耦合策略优化方法，提升样本效率和性能。**

- **链接: [https://arxiv.org/pdf/2603.01741](https://arxiv.org/pdf/2603.01741)**

> **作者:** Naoki Shitanda; Motoki Omura; Tatsuya Harada; Takayuki Osa
>
> **备注:** In ICLR 2026. Website at this https URL
>
> **摘要:** Scaling reinforcement learning to tens of thousands of parallel environments requires overcoming the limited exploration capacity of a single policy. Ensemble-based policy gradient methods, which employ multiple policies to collect diverse samples, have recently been proposed to promote exploration. However, merely broadening the exploration space does not always enhance learning capability, since excessive exploration can reduce exploration quality or compromise training stability. In this work, we theoretically analyze the impact of inter-policy diversity on learning efficiency in policy ensembles, and propose Coupled Policy Optimization which regulates diversity through KL constraints between policies. The proposed method enables effective exploration and outperforms strong baselines such as SAPG, PBT, and PPO across multiple tasks, including challenging dexterous manipulation, in terms of both sample efficiency and final performance. Furthermore, analysis of policy diversity and effective sample size during training reveals that follower policies naturally distribute around the leader, demonstrating the emergence of structured and efficient exploratory behavior. Our results indicate that diverse exploration under appropriate regulation is key to achieving stable and sample-efficient learning in ensemble policy gradient methods. Project page at this https URL .
>
---
#### [replaced 004] Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data
- **分类: cs.RO**

- **简介: 该论文属于人形机器人控制任务，旨在解决动态运动与稳定平衡难以兼得的问题。通过融合异构数据和设计混合奖励机制，提出AMS框架，实现单一策略的多任务控制。**

- **链接: [https://arxiv.org/pdf/2511.17373](https://arxiv.org/pdf/2511.17373)**

> **作者:** Yixuan Pan; Ruoyi Qiao; Li Chen; Kashyap Chitta; Liang Pan; Haoguang Mai; Qingwen Bu; Hao Zhao; Cunyuan Zheng; Ping Luo; Hongyang Li
>
> **摘要:** Humanoid robots are envisioned to perform a wide range of tasks in human-centered environments, requiring controllers that combine agility with robust balance. Recent advances in locomotion and whole-body tracking have enabled impressive progress in either agile dynamic skills or stability-critical behaviors, but existing methods remain specialized, focusing on one capability while compromising the other. In this work, we introduce AMS (Agility Meets Stability), the first framework that unifies both dynamic motion tracking and extreme balance maintenance in a single policy. Our key insight is to leverage heterogeneous data sources: human motion capture datasets that provide rich, agile behaviors, and physically constrained synthetic balance motions that capture stability configurations. To reconcile the divergent optimization goals of agility and stability, we design a hybrid reward scheme that applies general tracking objectives across all data while injecting balance-specific priors only into synthetic motions. Further, an adaptive learning strategy with performance-driven sampling and motion-specific reward shaping enables efficient training across diverse motion distributions. We validate AMS extensively in simulation and on a real Unitree G1 humanoid. Experiments demonstrate that a single policy can execute agile skills such as dancing and running, while also performing zero-shot extreme balance motions like Ip Man's Squat, highlighting AMS as a versatile control paradigm for future humanoid applications.
>
---
#### [replaced 005] Learning Agile Gate Traversal via Analytical Optimal Policy Gradient
- **分类: cs.RO**

- **简介: 该论文属于无人机精准飞行任务，解决窄门穿越难题。通过混合框架在线调优MPC参数，提升穿越效率与抗扰能力。**

- **链接: [https://arxiv.org/pdf/2508.21592](https://arxiv.org/pdf/2508.21592)**

> **作者:** Tianchen Sun; Bingheng Wang; Nuthasith Gerdpratoom; Longbin Tang; Yichao Gao; Lin Zhao
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Traversing narrow gates presents a significant challenge and has become a standard benchmark for evaluating agile and precise quadrotor flight. Traditional modularized autonomous flight stacks require extensive design and parameter tuning, while end-to-end reinforcement learning (RL) methods often suffer from low sample efficiency, limited interpretability, and degraded disturbance rejection under unseen perturbations. In this work, we present a novel hybrid framework that adaptively fine-tunes model predictive control (MPC) parameters online using outputs from a neural network (NN) trained offline. The NN jointly predicts a reference pose and cost function weights, conditioned on the coordinates of the gate corners and the current drone state. To achieve efficient training, we derive analytical policy gradients not only for the MPC module but also for an optimization-based gate traversal detection module. Hardware experiments demonstrate that our method enables fast and accurate quadrotor traversal through narrow gates in confined environments and demonstrates effective disturbance rejection against collision-induced perturbations.
>
---
#### [replaced 006] InstructVLA: Vision-Language-Action Instruction Tuning from Understanding to Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出InstructVLA，解决机器人多模态推理与精准动作生成的矛盾问题，通过新训练范式提升操作性能和泛化能力。**

- **链接: [https://arxiv.org/pdf/2507.17520](https://arxiv.org/pdf/2507.17520)**

> **作者:** Shuai Yang; Hao Li; Bin Wang; Yilun Chen; Yang Tian; Tai Wang; Hanqing Wang; Feng Zhao; Yiyi Liao; Jiangmiao Pang
>
> **备注:** 48 pages
>
> **摘要:** To operate effectively in the real world, robots should integrate multimodal reasoning with precise action generation. However, existing vision-language-action (VLA) models often sacrifice one for the other, narrow their abilities to task-specific manipulation data, and suffer catastrophic forgetting of pre-trained vision-language capabilities. To bridge this gap, we introduce InstructVLA, an end-to-end VLA model that preserves the flexible reasoning of large vision-language models (VLMs) while delivering leading manipulation performance with the help of embodied reasoning. InstructVLA introduces a novel training paradigm, Vision-Language-Action Instruction Tuning (VLA-IT), which employs multimodal training with mixture-of-experts adaptation to jointly optimize embodied reasoning and action generation on both standard VLM corpora and a curated 650K-sample VLA-IT dataset. On in-domain SimplerEnv tasks, InstructVLA achieves 33% improvement over SpatialVLA. To evaluate generalization, we introduce SimplerEnv-Instruct, an 80-task benchmark requiring closed-loop control and high-level instruction understanding, where it outperforms a fine-tuned OpenVLA by 96% and an action expert aided by GPT-4o by 29%. Additionally, InstructVLA surpasses baseline VLMs on multimodal tasks and exhibits inference-time scaling by leveraging textual reasoning to boost manipulation performance in both simulated and real-world settings. These results demonstrate InstructVLA's potential for bridging intuitive and steerable human-robot interaction with efficient policy learning.
>
---
#### [replaced 007] On Adversarial Attacks In Acoustic Drone Localization
- **分类: cs.SD; cs.RO; eess.AS**

- **简介: 该论文研究对抗攻击对声学无人机定位的影响，属于无人机导航安全任务，旨在分析PGD攻击并提出恢复算法以减轻攻击影响。**

- **链接: [https://arxiv.org/pdf/2502.20325](https://arxiv.org/pdf/2502.20325)**

> **作者:** Tamir Shor; Chaim Baskin; Alex Bronstein
>
> **摘要:** Multi-rotor aerial autonomous vehicles (MAVs, more widely known as "drones") have been generating increased interest in recent years due to their growing applicability in a vast and diverse range of fields (e.g., agriculture, commercial delivery, search and rescue). The sensitivity of visual-based methods to lighting conditions and occlusions had prompted growing study of navigation reliant on other modalities, such as acoustic sensing. A major concern in using drones in scale for tasks in non-controlled environments is the potential threat of adversarial attacks over their navigational systems, exposing users to mission-critical failures, security breaches, and compromised safety outcomes that can endanger operators and bystanders. While previous work shows impressive progress in acoustic-based drone localization, prior research in adversarial attacks over drone navigation only addresses visual sensing-based systems. In this work, we aim to compensate for this gap by supplying a comprehensive analysis of the effect of PGD adversarial attacks over acoustic drone localization. We furthermore develop an algorithm for adversarial perturbation recovery, capable of markedly diminishing the affect of such attacks in our setting.
>
---
#### [replaced 008] ConEQsA: Concurrent and Asynchronous Embodied Questions Scheduling and Answering
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出ConEQsA系统，解决多异步问题的具身问答任务。通过共享记忆和优先级规划，提升响应效率。**

- **链接: [https://arxiv.org/pdf/2509.11663](https://arxiv.org/pdf/2509.11663)**

> **作者:** Haisheng Wang; Dong Liu; Weiming Zhi
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** This paper formulates the Embodied Questions Answering (EQsA) problem, introduces a corresponding benchmark, and proposes an agentic system to tackle the problem. Classical Embodied Question Answering (EQA) is typically formulated as answering one single question by actively exploring a 3D environment. Real deployments, however, often demand handling multiple questions that may arrive asynchronously and carry different urgencies. We formalize this setting as Embodied Questions Answering (EQsA) and present ConEQsA, an agentic framework for concurrent, urgency-aware scheduling and answering. ConEQsA leverages shared group memory to reduce redundant exploration, and a priority-planning method to dynamically schedule questions. To evaluate the EQsA setting fairly, we contribute the Concurrent Asynchronous Embodied Questions (CAEQs) benchmark containing 40 indoor scenes and five questions per scene (200 in total), featuring asynchronous follow-up questions and human-annotated urgency labels. We further propose metrics for EQsA performance: Direct Answer Rate (DAR), and Normalized Urgency-Weighted Latency (NUWL), which serve as a fair evaluation protocol for EQsA. Empirical evaluations demonstrate that ConEQsA consistently outperforms strong sequential baselines, and show that urgency-aware, concurrent scheduling is key to making embodied agents responsive and efficient under realistic, multi-question workloads. Code is available on this https URL.
>
---
#### [replaced 009] Design Framework and Manufacturing of an Active Magnetic Bearing Spindle for Micro-Milling Applications
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于机械设计任务，旨在解决微铣削主轴在高速下的摩擦与热膨胀问题，通过设计制造一种主动磁悬浮主轴。**

- **链接: [https://arxiv.org/pdf/2603.00169](https://arxiv.org/pdf/2603.00169)**

> **作者:** Kazi Sher Ahmed; Bekir Bediz
>
> **摘要:** Micro-milling spindles require high rotational speeds where conventional rolling element bearings face limitations such as friction and thermal expansion. Active magnetic bearings (AMBs) address these challenges by providing non-contact and lubrication-free operation at ultra-high speeds with the ability to actively regulate spindle dynamics. The existing literature on AMB spindles has mainly reported specific prototype realizations or control system implementations for specific spindle dynamics. Consequently, design knowledge remains fragmented across isolated successful studies. This paper addresses this gap by presenting a systematic and iterative framework to design and manufacture a micro-milling AMB spindle. The process involves a multidisciplinary design flow with a focus on critical practical aspects of manufacturing. The realized spindle is reported as a case study.
>
---
#### [replaced 010] Floating-Base Deep Lagrangian Networks
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于系统辨识任务，解决浮基系统（如人形机器人）中物理约束不足的问题。通过参数化惯性矩阵，训练神经网络预测符合物理规律的惯性参数，提升模型的准确性与可解释性。**

- **链接: [https://arxiv.org/pdf/2510.17270](https://arxiv.org/pdf/2510.17270)**

> **作者:** Lucas Schulze; Juliano Decico Negri; Victor Barasuol; Vivian Suzano Medeiros; Marcelo Becker; Jan Peters; Oleg Arenz
>
> **摘要:** Grey-box methods for system identification combine deep learning with physics-informed constraints, capturing complex dependencies while improving out-of-distribution generalization. Despite the growing importance of floating-base systems such as humanoids and quadrupeds, current grey-box models ignore their specific physical constraints. For instance, the inertia matrix is not only positive definite but also exhibits branch-induced sparsity and input independence. Moreover, the 6x6 composite spatial inertia of the floating base inherits properties of single-rigid-body inertia matrices. As we show, this includes the triangle inequality on the eigenvalues of the composite rotational inertia. To address the lack of physical consistency in deep learning models of floating-base systems, we introduce a parameterization of inertia matrices that satisfies all these constraints. Inspired by Deep Lagrangian Networks (DeLaN), we train neural networks to predict physically plausible inertia matrices that minimize inverse dynamics error under Lagrangian mechanics. For evaluation, we collected and released a dataset on multiple quadrupeds and humanoids. In these experiments, our Floating-Base Deep Lagrangian Networks (FeLaN) achieve better overall performance on both simulated and real robots, while providing greater physical interpretability.
>
---
#### [replaced 011] Integration of UWB Radar on Mobile Robots for Continuous Obstacle and Environment Mapping
- **分类: cs.RO**

- **简介: 该论文属于环境感知任务，旨在解决复杂环境下障碍物检测与地图构建问题。通过在移动机器人上集成UWB雷达，提出一种无需固定锚点的映射方法。**

- **链接: [https://arxiv.org/pdf/2512.01018](https://arxiv.org/pdf/2512.01018)**

> **作者:** Adelina Giurea; Stijn Luchie; Dieter Coppens; Jeroen Hoebeke; Eli De Poorter
>
> **摘要:** This paper presents an infrastructure-free approach for obstacle detection and environmental mapping using ultra-wideband (UWB) radar mounted on a mobile robotic platform. Traditional sensing modalities such as visual cameras and Light Detection and Ranging (LiDAR) fail in environments with poor visibility due to darkness, smoke, or reflective surfaces. In these vision-impaired conditions, UWB radar offers a promising alternative. To this end, this work explores the suitability of robot-mounted UWB radar for environmental mapping in anchor-free, unknown scenarios. The study investigates how different materials (metal, concrete and plywood) and UWB radio channels (5 and 9) influence the Channel Impulse Response (CIR). Furthermore, a processing pipeline is proposed to achieve reliable mapping of detected obstacles, consisting of 3 steps: 1) target identification (based on CIR peak detection); 2) filtering (based on peak properties, signal-to-noise score, and phase-difference of arrival); and 3) clustering (based on distance estimation and angle-of-arrival estimation). The proposed approach successfully reduces noise and multipath effects, achieving high obstacle detection performance across a range of materials. Even in challenging low-reflectivity scenarios such as concrete, the method achieves a precision of 73.42% and a recall of 83.38% on channel 9. This work offers a foundation for further development of UWB-based localisation and mapping (SLAM) systems that do not rely on visual features and, unlike conventional UWB localisation systems, do not require fixed anchor nodes for triangulation.
>
---
#### [replaced 012] KILO-EKF: Koopman-Inspired Learned Observations Extended Kalman Filter
- **分类: cs.RO**

- **简介: 该论文提出KILO-EKF，用于解决传感器融合中的定位问题，通过学习测量模型提升EKF精度，无需依赖精确的传感器模型。**

- **链接: [https://arxiv.org/pdf/2601.12463](https://arxiv.org/pdf/2601.12463)**

> **作者:** Zi Cong Guo; James R. Forbes; Timothy D. Barfoot
>
> **备注:** Submitted to IEEE/RSJ IROS. 8 pages, 9 figures, 1 table
>
> **摘要:** We present the Koopman-Inspired Learned Observations Extended Kalman Filter (KILO-EKF), which combines a standard EKF prediction step with a correction step based on a Koopman-inspired measurement model learned from data. By lifting measurements into a feature space where they are linear in the state, KILO-EKF enables flexible modeling of complex or poorly calibrated sensors while retaining the structure and efficiency of recursive filtering. The resulting linear-Gaussian measurement model is learned in closed form from groundtruth training data, without iterative optimization or reliance on an explicit parametric sensor model. At inference, KILO-EKF performs a standard EKF update using Jacobians obtained via the learned lifting. We validate the approach on a real-world quadrotor localization task using an IMU, ultra-wideband (UWB) sensors, and a downward-facing laser. We compare against multiple EKF baselines with varying levels of sensor calibration. KILO-EKF achieves better accuracy and consistency compared to data-calibrated baselines, and significantly outperforms EKFs that rely on imperfect geometric models, while maintaining real-time inference and fast training. These results demonstrate the effectiveness of Koopman-inspired measurement learning as a scalable alternative to traditional model-based calibration.
>
---
#### [replaced 013] GrandTour: A Legged Robotics Dataset in the Wild for Multi-Modal Perception and State Estimation
- **分类: cs.RO**

- **简介: 该论文提出GrandTour数据集，用于解决腿式机器人在复杂环境中的状态估计与多模态感知问题，包含多种传感器数据及高精度轨迹信息，支持SLAM与传感器融合研究。**

- **链接: [https://arxiv.org/pdf/2602.18164](https://arxiv.org/pdf/2602.18164)**

> **作者:** Jonas Frey; Turcan Tuna; Frank Fu; Katharine Patterson; Tianao Xu; Maurice Fallon; Cesar Cadena; Marco Hutter
>
> **备注:** Turcan Tuna, and Jonas Frey contributed equally. Submitted to Sage The International Journal of Robotics Research
>
> **摘要:** Accurate state estimation and multi-modal perception are prerequisites for autonomous legged robots in complex, large-scale environments. To date, no large-scale public legged-robot dataset captures the real-world conditions needed to develop and benchmark algorithms for legged-robot state estimation, perception, and navigation. To address this, we introduce the GrandTour dataset, a multi-modal legged-robotics dataset collected across challenging outdoor and indoor environments, featuring an ANYbotics ANYmal-D quadruped equipped with the Boxi multi-modal sensor payload. GrandTour spans a broad range of environments and operational scenarios across distinct test sites, ranging from alpine scenery and forests to demolished buildings and urban areas, and covers a wide variation in scale, complexity, illumination, and weather conditions. The dataset provides time-synchronized sensor data from spinning LiDARs, multiple RGB cameras with complementary characteristics, proprioceptive sensors, and stereo depth cameras. Moreover, it includes high-precision ground-truth trajectories from satellite-based RTK-GNSS and a Leica Geosystems total station. This dataset supports research in SLAM, high-precision state estimation, and multi-modal learning, enabling rigorous evaluation and development of new approaches to sensor fusion in legged robotic systems. With its extensive scope, GrandTour represents the largest open-access legged-robotics dataset to date. The dataset is available at this https URL on HuggingFace (ROS-independent), and in ROS formats, along with tools and demo resources.
>
---
#### [replaced 014] Kinematify: Open-Vocabulary Synthesis of High-DoF Articulated Objects
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Kinematify，解决高自由度可动物体的自动合成问题。通过结合MCTS和几何优化，从图像或文本生成物理合理的关节结构。**

- **链接: [https://arxiv.org/pdf/2511.01294](https://arxiv.org/pdf/2511.01294)**

> **作者:** Jiawei Wang; Dingyou Wang; Jiaming Hu; Qixuan Zhang; Jingyi Yu; Lan Xu
>
> **备注:** Project Page: this https URL
>
> **摘要:** A deep understanding of kinematic structures and movable components is essential for enabling robots to manipulate objects and model their own articulated forms. Such understanding is captured through articulated objects, which are essential for tasks such as physical simulation, motion planning, and policy learning. However, creating these models, particularly for objects with high degrees of freedom (DoF), remains a significant challenge. Existing methods typically rely on motion sequences or strong assumptions from hand-curated datasets, which hinders scalability. In this paper, we introduce Kinematify, an automated framework that synthesizes articulated objects directly from arbitrary RGB images or textual descriptions. Our method addresses two core challenges: (i) inferring kinematic topologies for high-DoF objects and (ii) estimating joint parameters from static geometry. To achieve this, we combine MCTS search for structural inference with geometry-driven optimization for joint reasoning, producing physically consistent and functionally valid descriptions. We evaluate Kinematify on diverse inputs from both synthetic and real-world environments, demonstrating improvements in registration and kinematic topology accuracy over prior work.
>
---
#### [replaced 015] Towards an Adaptive Social Game-Playing Robot: An Offline Reinforcement Learning-Based Framework
- **分类: cs.RO**

- **简介: 论文提出一种基于离线强化学习的自适应社交游戏机器人系统，旨在提升人机交互中的情感适应能力。解决传统在线学习数据需求大、用户体验差的问题，通过离线RL方法实现高效安全的策略训练。**

- **链接: [https://arxiv.org/pdf/2509.16858](https://arxiv.org/pdf/2509.16858)**

> **作者:** Soon Jynn Chu; Raju Gottumukkala; Alan Barhorst
>
> **备注:** Submitted to conference
>
> **摘要:** HRI research increasingly demands robots that go beyond task execution to respond meaningfully to user emotions. This is especially needed when supporting students with learning difficulties in game-based learning scenarios. Here, the objective of these robots is to train users with game-playing skills, and this requires robots to get input about users' interests and engagement. In this paper, we present a system for an adaptive social game-playing robot. However, creating such an agent through online RL requires extensive real-world training data and potentially be uncomfortable for users. To address this, we investigate offline RL as a safe and efficient alternative. We introduce a system architecture that integrates multimodal emotion recognition and adaptive robotic responses. We also evaluate the performance of various offline RL algorithms using a dataset collected from a real-world human-robot game-playing scenario. Our results indicate that BCQ and DDQN offer the greatest robustness to hyperparameter variations, whereas CQL is the most effective at mitigating overestimation bias. Through this research, we aim to inform the selection and design of reliable offline RL policies for real-world social robotics. Ultimately, this work provides a foundational step toward creating socially intelligent agents that can learn complex and emotion-adaptive behaviors entirely from offline datasets, ensuring both human comfort and practical scalability.
>
---
#### [replaced 016] PROFusion: Robust and Accurate Dense Reconstruction via Camera Pose Regression and Optimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，解决不稳定运动下密集重建精度不足的问题。通过结合学习初始化与优化细化，提升重建鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2509.24236](https://arxiv.org/pdf/2509.24236)**

> **作者:** Siyan Dong; Zijun Wang; Lulu Cai; Yi Ma; Yanchao Yang
>
> **备注:** ICRA 2026
>
> **摘要:** Real-time dense scene reconstruction during unstable camera motions is crucial for robotics, yet current RGB-D SLAM systems fail when cameras experience large viewpoint changes, fast motions, or sudden shaking. Classical optimization-based methods deliver high accuracy but fail with poor initialization during large motions, while learning-based approaches provide robustness but lack sufficient accuracy for dense reconstruction. We address this challenge through a combination of learning-based initialization with optimization-based refinement. Our method employs a camera pose regression network to predict metric-aware relative poses from consecutive RGB-D frames, which serve as reliable starting points for a randomized optimization algorithm that further aligns depth images with the scene geometry. Extensive experiments demonstrate promising results: our approach outperforms the best competitor on challenging benchmarks, while maintaining comparable accuracy on stable motion sequences. The system operates in real-time, showcasing that combining simple and principled techniques can achieve both robustness for unstable motions and accuracy for dense reconstruction. Code released: this https URL.
>
---
#### [replaced 017] Multimodal Sensing for Robot-Assisted Sub-Tissue Feature Detection in Physiotherapy Palpation
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，旨在解决软组织中微小结构检测问题。通过融合视觉触觉与力传感器数据，提升机器人在物理治疗中的触诊准确性。**

- **链接: [https://arxiv.org/pdf/2512.20992](https://arxiv.org/pdf/2512.20992)**

> **作者:** Tian-Ao Ren; Jorge Garcia; Seongheon Hong; Jared Grinberg; Hojung Choi; Julia Di; Hao Li; Dmitry Grinberg; Mark R. Cutkosky
>
> **备注:** Accepted by AMSE Design of Medical Device 2026
>
> **摘要:** Robotic palpation relies on force sensing, but force signals in soft-tissue environments are variable and cannot reliably reveal subtle subsurface features. We present a compact multimodal sensor that integrates high-resolution vision-based tactile imaging with a 6-axis force-torque sensor. In experiments on silicone phantoms with diverse subsurface tendon geometries, force signals alone frequently produce ambiguous responses, while tactile images reveal clear structural differences in presence, diameter, depth, crossings, and multiplicity. Yet accurate force tracking remains essential for maintaining safe, consistent contact during physiotherapeutic interaction. Preliminary results show that combining tactile and force modalities enables robust subsurface feature detection and controlled robotic palpation.
>
---
#### [replaced 018] osmAG-LLM: Zero-Shot Open-Vocabulary Object Navigation via Semantic Maps and Large Language Models Reasoning
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决动态环境中物体定位与导航问题。通过结合语义地图和大语言模型推理，实现高效、灵活的物体寻址。**

- **链接: [https://arxiv.org/pdf/2507.12753](https://arxiv.org/pdf/2507.12753)**

> **作者:** Fujing Xie; Sören Schwertfeger; Hermann Blum
>
> **备注:** accepted at RA-L 2026
>
> **摘要:** Recent open-vocabulary robot mapping methods enrich dense geometric maps with pre-trained visual-language features, achieving a high level of detail and guiding robots to find objects specified by open-vocabulary language queries. While the issue of scalability for such approaches has received some attention, another fundamental problem is that high-detail object mapping quickly becomes outdated, as objects get moved around a lot. In this work, we develop a mapping and navigation system for object-goal navigation that, from the ground up, considers the possibilities that a queried object can have moved, or may not be mapped at all. Instead of striving for high-fidelity mapping detail, we consider that the main purpose of a map is to provide environment grounding and context, which we combine with the semantic priors of LLMs to reason about object locations and deploy an active, online approach to navigate to the objects. Through simulated and real-world experiments we find that our approach tends to have higher retrieval success at shorter path lengths for static objects and by far outperforms prior approaches in cases of dynamic or unmapped object queries. We provide our code and dataset at: this https URL.
>
---
#### [replaced 019] Learning Acrobatic Flight from Preferences
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究基于偏好强化学习的飞行控制任务，解决手动设计奖励函数效果不佳的问题。提出REC框架，通过不确定性建模提升性能，成功实现真实世界飞行器的复杂动作控制。**

- **链接: [https://arxiv.org/pdf/2508.18817](https://arxiv.org/pdf/2508.18817)**

> **作者:** Colin Merk; Ismail Geles; Jiaxu Xing; Angel Romero; Giorgia Ramponi; Davide Scaramuzza
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Preference-based reinforcement learning (PbRL) enables agents to learn control policies without requiring manually designed reward functions, making it well-suited for tasks where objectives are difficult to formalize or inherently subjective. Acrobatic flight poses a particularly challenging problem due to its complex dynamics, rapid movements, and the importance of precise execution. However, manually designed reward functions for such tasks often fail to capture the qualities that matter: we find that hand-crafted rewards agree with human judgment only 60.7% of the time, underscoring the need for preference-driven approaches. In this work, we propose Reward Ensemble under Confidence (REC), a probabilistic reward learning framework for PbRL that explicitly models per-timestep reward uncertainty through an ensemble of distributional reward models. By propagating uncertainty into the preference loss and leveraging disagreement for exploration, REC achieves 88.4% of shaped reward performance on acrobatic quadrotor control, compared to 55.2% with standard Preference PPO. We train policies in simulation and successfully transfer them zero-shot to the real world, demonstrating complex acrobatic maneuvers learned purely from preference feedback. We further validate REC on a continuous control benchmark, confirming its applicability beyond the domain of aerial robotics.
>
---
#### [replaced 020] Safe Payload Transfer with Ship-Mounted Cranes: A Robust Model Predictive Control Approach
- **分类: eess.SY; cs.RO**

- **简介: 论文提出一种鲁棒模型预测控制方法，用于船舶起重机的安全负载传输。解决在复杂海况下起重机动态扰动带来的安全与精度问题，通过优化参数适应和约束处理实现高效、安全的控制。**

- **链接: [https://arxiv.org/pdf/2510.16953](https://arxiv.org/pdf/2510.16953)**

> **作者:** Ersin Das; William A. Welch; Patrick Spieler; Keenan Albee; Aurelio Noca; Jeffrey Edlund; Jonathan Becktor; Thomas Touma; Jessica Todd; Sriramya Bhamidipati; Stella Kombo; Maira Saboia; Anna Sabel; Grace Lim; Rohan Thakker; Amir Rahmani; Joel W. Burdick
>
> **摘要:** Ensuring safe real-time control of ship-mounted cranes in unstructured transportation environments requires handling multiple safety constraints while maintaining effective payload transfer performance. Unlike traditional crane systems, ship-mounted cranes are consistently subjected to significant external disturbances affecting underactuated crane dynamics due to the ship's dynamic motion response to harsh sea conditions, which can lead to robustness issues. To tackle these challenges, we propose a robust and safe model predictive control (MPC) framework and demonstrate it on a 5-DOF crane system, where a Stewart platform simulates the external disturbances that ocean surface motions would have on the supporting ship. The crane payload transfer operation must avoid obstacles and accurately place the payload within a designated target area. We use a robust zero-order control barrier function (R-ZOCBF)-based safety constraint in the nonlinear MPC to ensure safe payload positioning, while time-varying bounding boxes are utilized for collision avoidance. We introduce a new optimization-based online robustness parameter adaptation scheme to reduce the conservativeness of R-ZOCBFs. Experimental trials on a crane prototype demonstrate the overall performance of our safe control approach under significant perturbing motions of the crane base. While our focus is on crane-facilitated transfer, the methods more generally apply to safe robotically-assisted parts mating and parts insertion.
>
---
#### [replaced 021] Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决如何将视觉语言模型的知识有效应用于机器人行为的问题。通过引入可操控的策略，提升低层控制能力，增强任务泛化性。**

- **链接: [https://arxiv.org/pdf/2602.13193](https://arxiv.org/pdf/2602.13193)**

> **作者:** William Chen; Jagdeep Singh Bhatia; Catherine Glossop; Nikhil Mathihalli; Ria Doshi; Andy Tang; Danny Driess; Karl Pertsch; Sergey Levine
>
> **摘要:** Pretrained vision-language models (VLMs) can make semantic and visual inferences across diverse settings, providing valuable common-sense priors for robotic control. However, effectively grounding this knowledge in robot behaviors remains an open challenge. Prior methods often employ a hierarchical approach where VLMs reason over high-level commands to be executed by separate low-level policies, e.g., vision-language-action models (VLAs). The interface between VLMs and VLAs is usually natural language task instructions, which fundamentally limits how much VLM reasoning can steer low-level behavior. We thus introduce Steerable Policies: VLAs trained on rich synthetic commands at various levels of abstraction, like subtasks, motions, and grounded pixel coordinates. By improving low-level controllability, Steerable Policies can unlock pretrained knowledge in VLMs, enabling improved task generalization. We demonstrate this benefit by controlling our Steerable Policies with both a learned high-level embodied reasoner and an off-the-shelf VLM prompted to reason over command abstractions via in-context learning. Across extensive real-world manipulation experiments, these two novel methods outperform prior embodied reasoning VLAs and VLM-based hierarchical baselines, including on challenging generalization and long-horizon tasks. Website: this http URL
>
---
#### [replaced 022] REFLEX: Metacognitive Reasoning for Reflective Zero-Shot Robotic Planning with Large Language Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于机器人规划任务，旨在解决LLM在零样本环境下执行复杂任务的不足。通过引入元认知机制，提升机器人自主思考与创新解决问题的能力。**

- **链接: [https://arxiv.org/pdf/2505.14899](https://arxiv.org/pdf/2505.14899)**

> **作者:** Wenjie Lin; Jin Wei-Kocsis; Jiansong Zhang; Byung-Cheol Min; Dongming Gan; Paul Asunda; Ragu Athinarayanan
>
> **摘要:** While large language models (LLMs) have shown great potential across various domains, their applications in robotics remain largely limited to static prompt-based behaviors and still face challenges in complex tasks under zero-shot or few-shot settings. Inspired by human metacognitive learning and creative problem-solving, we address this limitation by exploring a fundamental question: Can LLMs be empowered with metacognitive capabilities to reason, reflect, and create, thereby enhancing their ability to perform robotic tasks with minimal demonstrations? In this paper, we present REFLEX, a framework that integrates metacognitive learning into LLM-powered multi-robot collaboration. The system equips the LLM-powered robotic agents with a skill decomposition and self-reflection mechanism that identifies modular skills from prior tasks, reflects on failures in unseen task scenarios, and synthesizes effective new solutions. We propose a more challenging robotic benchmark task and evaluate our framework on the existing benchmark and the novel task. Experimental results show that our metacognitive learning framework significantly outperforms existing baselines. Moreover, we observe that our framework can generate solutions that differ from the ground truth yet still successfully complete the tasks. These findings support our hypothesis that metacognitive learning can foster creativity in robotic planning.
>
---
#### [replaced 023] D-GVIO: A Buffer-Driven and Efficient Decentralized GNSS-Visual-Inertial State Estimator for Multi-Agent Systems
- **分类: cs.RO**

- **简介: 该论文属于多智能体协同定位任务，旨在解决资源受限平台上的实时、鲁棒、高效状态估计问题。提出D-GVIO框架，通过缓冲策略和L-IEKF提升分布式定位性能。**

- **链接: [https://arxiv.org/pdf/2603.01404](https://arxiv.org/pdf/2603.01404)**

> **作者:** Yarong Luo; Wentao Lu; Chi Guo; Ming Li
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Cooperative localization is essential for swarm applications like collaborative exploration and search-and-rescue missions. However, maintaining real-time capability, robustness, and computational efficiency on resource-constrained platforms presents significant challenges. To address these challenges, we propose D-GVIO, a buffer-driven and fully decentralized GNSS-Visual-Inertial Odometry (GVIO) framework that leverages a novel buffering strategy to support efficient and robust distributed state estimation. The proposed framework is characterized by four core mechanisms. Firstly, through covariance segmentation, covariance intersection and buffering strategy, we modularize propagation and update steps in distributed state estimation, significantly reducing computational and communication burdens. Secondly, the left-invariant extended Kalman filter (L-IEKF) is adopted for information fusion, which exhibits superior state estimation performance over the traditional extended Kalman filter (EKF) since its state transition matrix is independent of the system state. Thirdly, a buffer-based re-propagation strategy is employed to handle delayed measurements efficiently and accurately by leveraging the L-IEKF, eliminating the need for costly re-computation. Finally, an adaptive buffer-driven outlier detection method is proposed to dynamically cull GNSS outliers, enhancing robustness in GNSS-challenged environments.
>
---
#### [replaced 024] SceneStreamer: Continuous Scenario Generation as Next Token Group Prediction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SceneStreamer，用于连续场景生成，解决动态交通模拟问题。通过序列预测方法，实现长期、多变的交通场景生成，提升自动驾驶训练效果。**

- **链接: [https://arxiv.org/pdf/2506.23316](https://arxiv.org/pdf/2506.23316)**

> **作者:** Zhenghao Peng; Yuxin Liu; Bolei Zhou
>
> **摘要:** Realistic and interactive traffic simulation is essential for training and evaluating autonomous driving systems. However, most existing data-driven simulation methods rely on static initialization or log-replay data, limiting their ability to model dynamic, long-horizon scenarios with evolving agent populations. We propose SceneStreamer, a unified autoregressive framework for continuous scenario generation that represents the entire scene as a sequence of tokens, including traffic light signals, agent states, and motion vectors, and generates them step by step with a transformer model. This design enables SceneStreamer to continuously introduce and retire agents over an unbounded horizon, supporting realistic long-duration simulation. Experiments demonstrate that SceneStreamer produces realistic, diverse, and adaptive traffic behaviors. Furthermore, reinforcement learning policies trained in SceneStreamer-generated scenarios achieve superior robustness and generalization, validating its utility as a high-fidelity simulation environment for autonomous driving. More information is available at this https URL .
>
---
#### [replaced 025] SoraNav: Adaptive UAV Task-Centric Navigation via Zeroshot VLM Reasoning
- **分类: cs.RO**

- **简介: 该论文提出SoraNav，解决UAV在3D环境中基于视觉和语言的导航问题，结合零样本VLM推理与几何决策，提升导航成功率和效率。**

- **链接: [https://arxiv.org/pdf/2510.25191](https://arxiv.org/pdf/2510.25191)**

> **作者:** Hongyu Song; Rishabh Dev Yadav; Cheng Guo; Wei Pan
>
> **摘要:** Interpreting visual observations and natural language instructions for complex task execution remains a key challenge in robotics and AI. Despite recent advances, language-driven navigation is still difficult, particularly for UAVs in small-scale 3D environments. Existing Vision-Language Navigation (VLN) approaches are mostly designed for ground robots and struggle to generalize to aerial tasks that require full 3D spatial reasoning. The emergence of large Vision-Language Models (VLMs), such as GPT and Claude, enables zero-shot semantic reasoning from visual and textual inputs. However, these models lack spatial grounding and are not directly applicable to navigation. To address these limitations, SoraNav is introduced, an adaptive UAV navigation framework that integrates zero-shot VLM reasoning with geometry-aware decision-making. Geometric priors are incorporated into image annotations to constrain the VLM action space and improve decision quality. A hybrid switching strategy leverages navigation history to alternate between VLM reasoning and geometry-based exploration, mitigating dead-ends and redundant revisits. A PX4-based hardware-software platform, comprising both a digital twin and a physical micro-UAV, enables reproducible evaluation. Experimental results show that in 2.5D scenarios, our method improves Success Rate (SR) by 25.7% and Success weighted by Path Length (SPL) by 17%. In 3D scenarios, it improves SR by 29.5% and SPL by 18.5% relative to the baseline.
>
---
#### [replaced 026] CoRL-MPPI: Enhancing MPPI With Learnable Behaviours For Efficient And Provably-Safe Multi-Robot Collision Avoidance
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人导航任务，解决碰撞避免问题。通过结合强化学习与MPPI，提升路径规划效率与安全性。**

- **链接: [https://arxiv.org/pdf/2511.09331](https://arxiv.org/pdf/2511.09331)**

> **作者:** Stepan Dergachev; Artem Pshenitsyn; Aleksandr Panov; Alexey Skrynnik; Konstantin Yakovlev
>
> **备注:** The manuscript includes 9 pages, 5 figures, and 1 table. This replacement revises and extends the original submission. The updated version adds a validation in Gazebo. It also expands the experimental evaluation by adding baselines and an evaluation scenario. In addition, the cost functions in MPPI-based methods were refined, leading to improved experimental performance
>
> **摘要:** Decentralized collision avoidance is a core challenge for scalable multi-robot systems. One of the promising approaches to tackle this problem is Model Predictive Path Integral (MPPI) -- a framework that naturally handles arbitrary motion models and provides strong theoretical guarantees. Still, in practice MPPI-based controller may provide suboptimal trajectories as its performance relies heavily on uninformed random sampling. In this work, we introduce CoRL-MPPI, a novel fusion of Cooperative Reinforcement Learning and MPPI to address this limitation. We train an action policy (approximated as deep neural network) in simulation that learns local cooperative collision avoidance behaviors. This learned policy is then embedded into the MPPI framework to guide its sampling distribution, biasing it towards more intelligent and cooperative actions. Notably, CoRL-MPPI preserves all the theoretical guarantees of regular MPPI. We evaluate our approach in dense, dynamic simulation environments against state-of-the-art baselines, such as ORCA, BVC, RL-RVO-NAV and classical MPPI. Our results demonstrate that CoRL-MPPI significantly improves navigation efficiency (measured by success rate and makespan) and safety, enabling agile and robust multi-robot navigation.
>
---
