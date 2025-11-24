# 机器人 cs.RO

- **最新发布 27 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Semantic Glitch: Agency and Artistry in an Autonomous Pixel Cloud
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文研究自主机器人艺术装置的创造性表达，针对传统机器人追求精度效率的问题，提出“语义错位”框架。通过多模态大模型实现无传感器自主导航，赋予机器人基于自然语言的拟人化性格，以非精确性创造独特艺术人格，验证了低精度系统在生成有个性、可信赖陪伴体方面的有效性。**

- **链接: [https://arxiv.org/pdf/2511.16048v1](https://arxiv.org/pdf/2511.16048v1)**

> **作者:** Qing Zhang; Jing Huang; Mingyang Xu; Jun Rekimoto
>
> **备注:** NeurIPS 2025 Creative AI Track, The Thirty-Ninth Annual Conference on Neural Information Processing Systems
>
> **摘要:** While mainstream robotics pursues metric precision and flawless performance, this paper explores the creative potential of a deliberately "lo-fi" approach. We present the "Semantic Glitch," a soft flying robotic art installation whose physical form, a 3D pixel style cloud, is a "physical glitch" derived from digital archaeology. We detail a novel autonomous pipeline that rejects conventional sensors like LiDAR and SLAM, relying solely on the qualitative, semantic understanding of a Multimodal Large Language Model to navigate. By authoring a bio-inspired personality for the robot through a natural language prompt, we create a "narrative mind" that complements the "weak," historically, loaded body. Our analysis begins with a 13-minute autonomous flight log, and a follow-up study statistically validates the framework's robustness for authoring quantifiably distinct personas. The combined analysis reveals emergent behaviors, from landmark-based navigation to a compelling "plan to execution" gap, and a character whose unpredictable, plausible behavior stems from a lack of precise proprioception. This demonstrates a lo-fi framework for creating imperfect companions whose success is measured in character over efficiency.
>
---
#### [new 002] Flow-Aided Flight Through Dynamic Clutters From Point To Motion
- **分类: cs.RO**

- **简介: 该论文针对动态障碍物环境中的自主飞行任务，解决感知与避障效率问题。提出基于单LiDAR的强化学习系统，通过深度距离图与点流融合实现轻量级环境表征，利用相对运动调制的距离场隐式驱动避障策略，无需目标检测与预测，显著提升复杂动态场景下的飞行成功率与实机适应性。**

- **链接: [https://arxiv.org/pdf/2511.16372v1](https://arxiv.org/pdf/2511.16372v1)**

> **作者:** Bowen Xu; Zexuan Yan; Minghao Lu; Xiyu Fan; Yi Luo; Youshen Lin; Zhiqiang Chen; Yeke Chen; Qiyuan Qiao; Peng Lu
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L), November, 2025
>
> **摘要:** Challenges in traversing dynamic clutters lie mainly in the efficient perception of the environmental dynamics and the generation of evasive behaviors considering obstacle movement. Previous solutions have made progress in explicitly modeling the dynamic obstacle motion for avoidance, but this key dependency of decision-making is time-consuming and unreliable in highly dynamic scenarios with occlusions. On the contrary, without introducing object detection, tracking, and prediction, we empower the reinforcement learning (RL) with single LiDAR sensing to realize an autonomous flight system directly from point to motion. For exteroception, a depth sensing distance map achieving fixed-shape, low-resolution, and detail-safe is encoded from raw point clouds, and an environment change sensing point flow is adopted as motion features extracted from multi-frame observations. These two are integrated into a lightweight and easy-to-learn representation of complex dynamic environments. For action generation, the behavior of avoiding dynamic threats in advance is implicitly driven by the proposed change-aware sensing representation, where the policy optimization is indicated by the relative motion modulated distance field. With the deployment-friendly sensing simulation and dynamics model-free acceleration control, the proposed system shows a superior success rate and adaptability to alternatives, and the policy derived from the simulator can drive a real-world quadrotor with safe maneuvers.
>
---
#### [new 003] PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks
- **分类: cs.RO**

- **简介: 该论文针对多机器人协作中的“共享脑困境”，提出PIPHEN框架，通过边缘端语义蒸馏将高维感知数据压缩为紧凑物理表征，并利用哈密顿能量网络实现高效控制。解决了带宽瓶颈与决策延迟问题，显著降低数据量与延迟，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.16200v1](https://arxiv.org/pdf/2511.16200v1)**

> **作者:** Kewei Chen; Yayu Long; Mingsheng Shang
>
> **备注:** Accepted at the AAAI Conference on Artificial Intelligence (AAAI-26)
>
> **摘要:** Multi-robot systems in complex physical collaborations face a "shared brain dilemma": transmitting high-dimensional multimedia data (e.g., video streams at ~30MB/s) creates severe bandwidth bottlenecks and decision-making latency. To address this, we propose PIPHEN, an innovative distributed physical cognition-control framework. Its core idea is to replace "raw data communication" with "semantic communication" by performing "semantic distillation" at the robot edge, reconstructing high-dimensional perceptual data into compact, structured physical representations. This idea is primarily realized through two key components: (1) a novel Physical Interaction Prediction Network (PIPN), derived from large model knowledge distillation, to generate this representation; and (2) a Hamiltonian Energy Network (HEN) controller, based on energy conservation, to precisely translate this representation into coordinated actions. Experiments show that, compared to baseline methods, PIPHEN can compress the information representation to less than 5% of the original data volume and reduce collaborative decision-making latency from 315ms to 76ms, while significantly improving task success rates. This work provides a fundamentally efficient paradigm for resolving the "shared brain dilemma" in resource-constrained multi-robot systems.
>
---
#### [new 004] InEKFormer: A Hybrid State Estimator for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文针对人形机器人在复杂环境中稳定运动的状态估计问题，提出InEKFormer混合方法，结合不变扩展卡尔曼滤波（InEKF）与Transformer网络，提升状态估计精度与鲁棒性。通过在RH5机器人数据集上的实验，验证了Transformer在高维状态估计中的潜力，并强调了自回归训练的重要性。**

- **链接: [https://arxiv.org/pdf/2511.16306v1](https://arxiv.org/pdf/2511.16306v1)**

> **作者:** Lasse Hohmeyer; Mihaela Popescu; Ivan Bergonzani; Dennis Mronga; Frank Kirchner
>
> **备注:** Accepted at The 22nd International Conference on Advanced Robotics (ICAR 2025)
>
> **摘要:** Humanoid robots have great potential for a wide range of applications, including industrial and domestic use, healthcare, and search and rescue missions. However, bipedal locomotion in different environments is still a challenge when it comes to performing stable and dynamic movements. This is where state estimation plays a crucial role, providing fast and accurate feedback of the robot's floating base state to the motion controller. Although classical state estimation methods such as Kalman filters are widely used in robotics, they require expert knowledge to fine-tune the noise parameters. Due to recent advances in the field of machine learning, deep learning methods are increasingly used for state estimation tasks. In this work, we propose the InEKFormer, a novel hybrid state estimation method that incorporates an invariant extended Kalman filter (InEKF) and a Transformer network. We compare our method with the InEKF and the KalmanNet approaches on datasets obtained from the humanoid robot RH5. The results indicate the potential of Transformers in humanoid state estimation, but also highlight the need for robust autoregressive training in these high-dimensional problems.
>
---
#### [new 005] PushingBots: Collaborative Pushing via Neural Accelerated Combinatorial Hybrid Optimization
- **分类: cs.RO**

- **简介: 该论文研究多机器人协同非握持推动物体任务，解决复杂环境中任意形状物体的高效推移问题。提出基于神经加速组合混合优化的PushingBots框架，通过动态任务分配、关键帧引导的模式搜索与混合控制，实现多机器人协作推物，显著提升规划效率与适应性。**

- **链接: [https://arxiv.org/pdf/2511.15995v1](https://arxiv.org/pdf/2511.15995v1)**

> **作者:** Zili Tang; Ying Zhang; Meng Guo
>
> **备注:** 20 pages, 24 figures. Accepted to IEEE Transactions on Robotics (T-RO), 2025
>
> **摘要:** Many robots are not equipped with a manipulator and many objects are not suitable for prehensile manipulation (such as large boxes and cylinders). In these cases, pushing is a simple yet effective non-prehensile skill for robots to interact with and further change the environment. Existing work often assumes a set of predefined pushing modes and fixed-shape objects. This work tackles the general problem of controlling a robotic fleet to push collaboratively numerous arbitrary objects to respective destinations, within complex environments of cluttered and movable obstacles. It incorporates several characteristic challenges for multi-robot systems such as online task coordination under large uncertainties of cost and duration, and for contact-rich tasks such as hybrid switching among different contact modes, and under-actuation due to constrained contact forces. The proposed method is based on combinatorial hybrid optimization over dynamic task assignments and hybrid execution via sequences of pushing modes and associated forces. It consists of three main components: (I) the decomposition, ordering and rolling assignment of pushing subtasks to robot subgroups; (II) the keyframe guided hybrid search to optimize the sequence of parameterized pushing modes for each subtask; (III) the hybrid control to execute these modes and transit among them. Last but not least, a diffusion-based accelerator is adopted to predict the keyframes and pushing modes that should be prioritized during hybrid search; and further improve planning efficiency. The framework is complete under mild assumptions. Its efficiency and effectiveness under different numbers of robots and general-shaped objects are validated extensively in simulations and hardware experiments, as well as generalizations to heterogeneous robots, planar assembly and 6D pushing.
>
---
#### [new 006] InternData-A1: Pioneering High-Fidelity Synthetic Data for Pre-training Generalist Policy
- **分类: cs.RO**

- **简介: 该论文针对机器人视觉-语言-动作（VLA）模型预训练中真实数据依赖问题，提出InternData-A1合成数据集。通过全自动、解耦的仿真管道生成超630k轨迹，证明仅用合成数据即可达到最强真实数据集性能，实现零样本仿生到现实迁移，推动具身智能数据规模化生成。**

- **链接: [https://arxiv.org/pdf/2511.16651v1](https://arxiv.org/pdf/2511.16651v1)**

> **作者:** Yang Tian; Yuyin Yang; Yiman Xie; Zetao Cai; Xu Shi; Ning Gao; Hangxu Liu; Xuekun Jiang; Zherui Qiu; Feng Yuan; Yaping Li; Ping Wang; Junhao Cai; Jia Zeng; Hao Dong; Jiangmiao Pang
>
> **摘要:** Recent works explore how real and synthetic data contribute to Vision-Language-Action (VLA) models' generalization. While current VLA models have shown the strong effectiveness of large-scale real-robot pre-training, synthetic data has not previously demonstrated comparable capability at scale. This paper provides the first evidence that synthetic data alone can match the performance of the strongest $π$-dataset in pre-training a VLA model, revealing the substantial value of large-scale simulation. The resulting model also exhibits surprisingly zero-shot sim-to-real transfer on several challenging tasks. Our synthetic dataset, InternData-A1, contains over 630k trajectories and 7,433 hours across 4 embodiments, 18 skills, 70 tasks, and 227 scenes, covering rigid, articulated, deformable, and fluid-object manipulation. It is generated through a highly autonomous, fully decoupled, and compositional simulation pipeline that enables long-horizon skill composition, flexible task assembly, and heterogeneous embodiments with minimal manual tuning. Using the same architecture as $π_0$, we pre-train a model entirely on InternData-A1 and find that it matches the official $π_0$ across 49 simulation tasks, 5 real-world tasks, and 4 long-horizon dexterous tasks. We release the dataset and will open-source the generation pipeline to broaden access to large-scale robotic data and to lower the barrier to scalable data creation for embodied AI research.
>
---
#### [new 007] MagBotSim: Physics-Based Simulation and Reinforcement Learning Environments for Magnetic Robotics
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出MagBotSim，一个用于磁悬浮机器人的物理仿真平台，旨在支持智能算法开发。针对工业自动化中物料运输与操作效率低的问题，通过构建可编程磁机器人集群的仿真环境，推动磁机器人在制造系统中的协同控制与智能决策研究。**

- **链接: [https://arxiv.org/pdf/2511.16158v1](https://arxiv.org/pdf/2511.16158v1)**

> **作者:** Lara Bergmann; Cedric Grothues; Klaus Neumann
>
> **摘要:** Magnetic levitation is about to revolutionize in-machine material flow in industrial automation. Such systems are flexibly configurable and can include a large number of independently actuated shuttles (movers) that dynamically rebalance production capacity. Beyond their capabilities for dynamic transportation, these systems possess the inherent yet unexploited potential to perform manipulation. By merging the fields of transportation and manipulation into a coordinated swarm of magnetic robots (MagBots), we enable manufacturing systems to achieve significantly higher efficiency, adaptability, and compactness. To support the development of intelligent algorithms for magnetic levitation systems, we introduce MagBotSim (Magnetic Robotics Simulation): a physics-based simulation for magnetic levitation systems. By framing magnetic levitation systems as robot swarms and providing a dedicated simulation, this work lays the foundation for next generation manufacturing systems powered by Magnetic Robotics. MagBotSim's documentation, videos, experiments, and code are available at: https://ubi-coro.github.io/MagBotSim/
>
---
#### [new 008] How Robot Dogs See the Unseeable
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究机器人视觉中的遮挡问题，提出通过模仿动物“侧视”运动实现合成孔径成像。利用机器人移动时相机的运动轨迹形成宽合成孔径，计算融合图像以实现极浅景深，使背景清晰而遮挡物模糊。该方法无需额外传感器，实时高效，可提升复杂场景下的视觉理解能力，尤其增强大模型在遮挡环境中的推理性能。**

- **链接: [https://arxiv.org/pdf/2511.16262v1](https://arxiv.org/pdf/2511.16262v1)**

> **作者:** Oliver Bimber; Karl Dietrich von Ellenrieder; Michael Haller; Rakesh John Amala Arokia Nathan; Gianni Lunardi; Marco Camurri; Mohamed Youssef; Santos Miguel Orozco Soto; Jeremy E. Niven
>
> **摘要:** Peering, a side-to-side motion used by animals to estimate distance through motion parallax, offers a powerful bio-inspired strategy to overcome a fundamental limitation in robotic vision: partial occlusion. Conventional robot cameras, with their small apertures and large depth of field, render both foreground obstacles and background objects in sharp focus, causing occluders to obscure critical scene information. This work establishes a formal connection between animal peering and synthetic aperture (SA) sensing from optical imaging. By having a robot execute a peering motion, its camera describes a wide synthetic aperture. Computational integration of the captured images synthesizes an image with an extremely shallow depth of field, effectively blurring out occluding elements while bringing the background into sharp focus. This efficient, wavelength-independent technique enables real-time, high-resolution perception across various spectral bands. We demonstrate that this approach not only restores basic scene understanding but also empowers advanced visual reasoning in large multimodal models, which fail with conventionally occluded imagery. Unlike feature-dependent multi-view 3D vision methods or active sensors like LiDAR, SA sensing via peering is robust to occlusion, computationally efficient, and immediately deployable on any mobile robot. This research bridges animal behavior and robotics, suggesting that peering motions for synthetic aperture sensing are a key to advanced scene understanding in complex, cluttered environments.
>
---
#### [new 009] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型因依赖海量冗余数据导致效率低下的问题，提出FT-NCFM数据蒸馏框架。通过因果归因与程序化对比验证评估样本价值，生成模型无关、信息密集的高质量数据集，显著提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2511.16233v1](https://arxiv.org/pdf/2511.16233v1)**

> **作者:** Kewei Chen; Yayu Long; Shuai Li; Mingsheng Shang
>
> **备注:** Accepted at the AAAI Conference on Artificial Intelligence (AAAI-26)
>
> **摘要:** The powerful generalization of Vision-Language-Action (VLA) models is bottlenecked by their heavy reliance on massive, redundant, and unevenly valued datasets, hindering their widespread application. Existing model-centric optimization paths, such as model compression (which often leads to performance degradation) or policy distillation (whose products are model-dependent and lack generality), fail to fundamentally address this data-level challenge. To this end, this paper introduces FT-NCFM, a fundamentally different, data-centric generative data distillation framework. Our framework employs a self-contained Fact-Tracing (FT) engine that combines causal attribution with programmatic contrastive verification to assess the intrinsic value of samples. Guided by these assessments, an adversarial NCFM process synthesizes a model-agnostic, information-dense, and reusable data asset. Experimental results on several mainstream VLA benchmarks show that models trained on just 5% of our distilled coreset achieve a success rate of 85-90% compared with training on the full dataset, while reducing training time by over 80%. Our work demonstrates that intelligent data distillation is a highly promising new path for building efficient, high-performance VLA models.
>
---
#### [new 010] Dexterity from Smart Lenses: Multi-Fingered Robot Manipulation with In-the-Wild Human Demonstrations
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究多指机器人操作中的模仿学习任务，旨在解决人类自然环境演示与机器人执行之间的“具身差距”问题。作者提出框架AINA，利用Aria Gen 2智能眼镜采集任意环境下的人类动作数据，直接学习鲁棒的3D点云控制策略，无需机器人训练数据或仿真优化，实现跨场景、免调优的自主操作。**

- **链接: [https://arxiv.org/pdf/2511.16661v1](https://arxiv.org/pdf/2511.16661v1)**

> **作者:** Irmak Guzey; Haozhi Qi; Julen Urain; Changhao Wang; Jessica Yin; Krishna Bodduluri; Mike Lambeta; Lerrel Pinto; Akshara Rai; Jitendra Malik; Tingfan Wu; Akash Sharma; Homanga Bharadhwaj
>
> **摘要:** Learning multi-fingered robot policies from humans performing daily tasks in natural environments has long been a grand goal in the robotics community. Achieving this would mark significant progress toward generalizable robot manipulation in human environments, as it would reduce the reliance on labor-intensive robot data collection. Despite substantial efforts, progress toward this goal has been bottle-necked by the embodiment gap between humans and robots, as well as by difficulties in extracting relevant contextual and motion cues that enable learning of autonomous policies from in-the-wild human videos. We claim that with simple yet sufficiently powerful hardware for obtaining human data and our proposed framework AINA, we are now one significant step closer to achieving this dream. AINA enables learning multi-fingered policies from data collected by anyone, anywhere, and in any environment using Aria Gen 2 glasses. These glasses are lightweight and portable, feature a high-resolution RGB camera, provide accurate on-board 3D head and hand poses, and offer a wide stereo view that can be leveraged for depth estimation of the scene. This setup enables the learning of 3D point-based policies for multi-fingered hands that are robust to background changes and can be deployed directly without requiring any robot data (including online corrections, reinforcement learning, or simulation). We compare our framework against prior human-to-robot policy learning approaches, ablate our design choices, and demonstrate results across nine everyday manipulation tasks. Robot rollouts are best viewed on our website: https://aina-robot.github.io.
>
---
#### [new 011] I've Changed My Mind: Robots Adapting to Changing Human Goals during Collaboration
- **分类: cs.RO**

- **简介: 该论文研究人机协作中目标动态变化的适应问题。针对传统方法假设目标固定、无法应对中途变目标的缺陷，提出通过追踪多候选动作序列并结合策略库验证，实时检测目标变更。机器人据此调整信念，利用滚动时域规划主动选择助人动作，并设计区分性动作以揭示新目标。实验在复杂烹饪场景中验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2511.15914v1](https://arxiv.org/pdf/2511.15914v1)**

> **作者:** Debasmita Ghose; Oz Gitelson; Ryan Jin; Grace Abawe; Marynel Vazquez; Brian Scassellati
>
> **备注:** Accepted to RA-L
>
> **摘要:** For effective human-robot collaboration, a robot must align its actions with human goals, even as they change mid-task. Prior approaches often assume fixed goals, reducing goal prediction to a one-time inference. However, in real-world scenarios, humans frequently shift goals, making it challenging for robots to adapt without explicit communication. We propose a method for detecting goal changes by tracking multiple candidate action sequences and verifying their plausibility against a policy bank. Upon detecting a change, the robot refines its belief in relevant past actions and constructs Receding Horizon Planning (RHP) trees to actively select actions that assist the human while encouraging Differentiating Actions to reveal their updated goal. We evaluate our approach in a collaborative cooking environment with up to 30 unique recipes and compare it to three comparable human goal prediction algorithms. Our method outperforms all baselines, quickly converging to the correct goal after a switch, reducing task completion time, and improving collaboration efficiency.
>
---
#### [new 012] Homogeneous Proportional-Integral-Derivative Controller in Mobile Robotic Manipulators
- **分类: cs.RO; nlin.AO**

- **简介: 该论文针对移动机器人操作臂（MRMs）的协同控制难题，提出一种新型齐次比例-积分-微分（hPID）控制器。通过引入齐次控制理论，将传统PID增益推广为状态相关非线性函数，实现全局渐近稳定与有限时间收敛。实验验证其在轨迹跟踪精度、响应速度及鲁棒性上优于传统线性PID控制器。**

- **链接: [https://arxiv.org/pdf/2511.16406v1](https://arxiv.org/pdf/2511.16406v1)**

> **作者:** Luis Luna; Isaac Chairez; Andrey Polyakov
>
> **摘要:** Mobile robotic manipulators (MRMs), which integrate mobility and manipulation capabilities, present significant control challenges due to their nonlinear dynamics, underactuation, and coupling between the base and manipulator subsystems. This paper proposes a novel homogeneous Proportional-Integral-Derivative (hPID) control strategy tailored for MRMs to achieve robust and coordinated motion control. Unlike classical PID controllers, the hPID controller leverages the mathematical framework of homogeneous control theory to systematically enhance the stability and convergence properties of the closed-loop system, even in the presence of dynamic uncertainties and external disturbances involved into a system in a homogeneous way. A homogeneous PID structure is designed, ensuring improved convergence of tracking errors through a graded homogeneity approach that generalizes traditional PID gains to nonlinear, state-dependent functions. Stability analysis is conducted using Lyapunov-based methods, demonstrating that the hPID controller guarantees global asymptotic stability and finite-time convergence under mild assumptions. Experimental results on a representative MRM model validate the effectiveness of the hPID controller in achieving high-precision trajectory tracking for both the mobile base and manipulator arm, outperforming conventional linear PID controllers in terms of response time, steady-state error, and robustness to model uncertainties. This research contributes a scalable and analytically grounded control framework for enhancing the autonomy and reliability of next-generation mobile manipulation systems in structured and unstructured environments.
>
---
#### [new 013] Gimballed Rotor Mechanism for Omnidirectional Quadrotors
- **分类: cs.RO**

- **简介: 该论文针对传统四旋翼机欠驱动导致运动受限的问题，提出一种带云台转子的全驱动设计，通过在旋翼平台集成舵机实现独立倾角控制，提升六自由度机动能力。研究开发了PX4中的新型控制分配方案并完成飞行验证，实现了轻量化、模块化且易于集成的全向四旋翼系统。**

- **链接: [https://arxiv.org/pdf/2511.15909v1](https://arxiv.org/pdf/2511.15909v1)**

> **作者:** J. Cristobal; A. Z. Zain Aldeen; M. Izadi; R. Faieghi
>
> **备注:** 6 pages, 7 figures, CASE 2025
>
> **摘要:** This paper presents the design of a gimballed rotor mechanism as a modular and efficient solution for constructing omnidirectional quadrotors. Unlike conventional quadrotors, which are underactuated, this class of quadrotors achieves full actuation, enabling independent motion in all six degrees of freedom. While existing omnidirectional quadrotor designs often require significant structural modifications, the proposed gimballed rotor system maintains a lightweight and easy-to-integrate design by incorporating servo motors within the rotor platforms, allowing independent tilting of each rotor without major alterations to the central structure of a quadrotor. To accommodate this unconventional design, we develop a new control allocation scheme in PX4 Autopilot and present successful flight tests, validating the effectiveness of the proposed approach.
>
---
#### [new 014] DynaMimicGen: A Data Generation Framework for Robot Learning of Dynamic Tasks
- **分类: cs.RO**

- **简介: 该论文提出DynaMimicGen（D-MG）框架，解决动态环境下机器人操作数据收集难的问题。通过少量人类示范，利用动态运动基元生成适应环境变化的平滑、真实轨迹，支持静态与动态任务泛化。实验表明，基于该框架训练的策略在长时序、高接触任务中表现优异，显著降低对人工数据依赖，推动高效自主机器人学习。**

- **链接: [https://arxiv.org/pdf/2511.16223v1](https://arxiv.org/pdf/2511.16223v1)**

> **作者:** Vincenzo Pomponi; Paolo Franceschi; Stefano Baraldo; Loris Roveda; Oliver Avram; Luca Maria Gambardella; Anna Valente
>
> **摘要:** Learning robust manipulation policies typically requires large and diverse datasets, the collection of which is time-consuming, labor-intensive, and often impractical for dynamic environments. In this work, we introduce DynaMimicGen (D-MG), a scalable dataset generation framework that enables policy training from minimal human supervision while uniquely supporting dynamic task settings. Given only a few human demonstrations, D-MG first segments the demonstrations into meaningful sub-tasks, then leverages Dynamic Movement Primitives (DMPs) to adapt and generalize the demonstrated behaviors to novel and dynamically changing environments. Improving prior methods that rely on static assumptions or simplistic trajectory interpolation, D-MG produces smooth, realistic, and task-consistent Cartesian trajectories that adapt in real time to changes in object poses, robot states, or scene geometry during task execution. Our method supports different scenarios - including scene layouts, object instances, and robot configurations - making it suitable for both static and highly dynamic manipulation tasks. We show that robot agents trained via imitation learning on D-MG-generated data achieve strong performance across long-horizon and contact-rich benchmarks, including tasks like cube stacking and placing mugs in drawers, even under unpredictable environment changes. By eliminating the need for extensive human demonstrations and enabling generalization in dynamic settings, D-MG offers a powerful and efficient alternative to manual data collection, paving the way toward scalable, autonomous robot learning.
>
---
#### [new 015] Robot Metacognition: Decision Making with Confidence for Tool Invention
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人元认知，聚焦于通过置信度实现自主工具发明中的决策优化。针对机器人缺乏自我反思能力的问题，提出基于置信度的元认知架构，强化决策可靠性与环境适应性，提升真实物理场景下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.16390v1](https://arxiv.org/pdf/2511.16390v1)**

> **作者:** Ajith Anil Meera; Poppy Collis; Polina Arbuzova; Abián Torres; Paul F Kinghorn; Ricardo Sanz; Pablo Lanillos
>
> **备注:** under review
>
> **摘要:** Robots today often miss a key ingredient of truly intelligent behavior: the ability to reflect on their own cognitive processes and decisions. In humans, this self-monitoring or metacognition is crucial for learning, decision making and problem solving. For instance, they can evaluate how confident they are in performing a task, thus regulating their own behavior and allocating proper resources. Taking inspiration from neuroscience, we propose a robot metacognition architecture centered on confidence (a second-order judgment on decisions) and we demonstrate it on the use case of autonomous tool invention. We propose the use of confidence as a metacognitive measure within the robot decision making scheme. Confidence-informed robots can evaluate the reliability of their decisions, improving their robustness during real-world physical deployment. This form of robotic metacognition emphasizes embodied action monitoring as a means to achieve better informed decisions. We also highlight potential applications and research directions for robot metacognition.
>
---
#### [new 016] Bi-AQUA: Bilateral Control-Based Imitation Learning for Underwater Robot Arms via Lighting-Aware Action Chunking with Transformers
- **分类: cs.RO**

- **简介: 该论文针对水下机器人机械臂操纵中光照变化导致的视觉失真问题，提出Bi-AQUA框架。通过三层光照自适应机制（光照编码、FiLM调制、光照令牌），实现无标注光照感知，提升模仿学习鲁棒性。实验验证其在复杂光照下显著优于基线，推动水下力控自主操作发展。**

- **链接: [https://arxiv.org/pdf/2511.16050v1](https://arxiv.org/pdf/2511.16050v1)**

> **作者:** Takeru Tsunoori; Masato Kobayashi; Yuki Uranishi
>
> **摘要:** Underwater robotic manipulation is fundamentally challenged by extreme lighting variations, color distortion, and reduced visibility. We introduce Bi-AQUA, the first underwater bilateral control-based imitation learning framework that integrates lighting-aware visual processing for underwater robot arms. Bi-AQUA employs a hierarchical three-level lighting adaptation mechanism: a Lighting Encoder that extracts lighting representations from RGB images without manual annotation and is implicitly supervised by the imitation objective, FiLM modulation of visual backbone features for adaptive, lighting-aware feature extraction, and an explicit lighting token added to the transformer encoder input for task-aware conditioning. Experiments on a real-world underwater pick-and-place task under diverse static and dynamic lighting conditions show that Bi-AQUA achieves robust performance and substantially outperforms a bilateral baseline without lighting modeling. Ablation studies further confirm that all three lighting-aware components are critical. This work bridges terrestrial bilateral control-based imitation learning and underwater manipulation, enabling force-sensitive autonomous operation in challenging marine environments. For additional material, please check: https://mertcookimg.github.io/bi-aqua
>
---
#### [new 017] From Prompts to Printable Models: Support-Effective 3D Generation via Offset Direct Preference Optimization
- **分类: cs.RO**

- **简介: 该论文针对3D打印中支持结构浪费问题，提出SEG框架，通过偏移直接偏好优化（ODPO）在生成阶段优化模型，使其天然具备少支撑特性。工作包括集成支持模拟至训练流程，显著降低支撑体积与打印时间，提升可打印性与设计保真度。**

- **链接: [https://arxiv.org/pdf/2511.16434v1](https://arxiv.org/pdf/2511.16434v1)**

> **作者:** Chenming Wu; Xiaofan Li; Chengkai Dai
>
> **备注:** Technical report (7 pages)
>
> **摘要:** The transition from digital 3D models to physical objects via 3D printing often requires support structures to prevent overhanging features from collapsing during the fabrication process. While current slicing technologies offer advanced support strategies, they focus on post-processing optimizations rather than addressing the underlying need for support-efficient design during the model generation phase. This paper introduces SEG (\textit{\underline{S}upport-\underline{E}ffective \underline{G}eneration}), a novel framework that integrates Direct Preference Optimization with an Offset (ODPO) into the 3D generation pipeline to directly optimize models for minimal support material usage. By incorporating support structure simulation into the training process, SEG encourages the generation of geometries that inherently require fewer supports, thus reducing material waste and production time. We demonstrate SEG's effectiveness through extensive experiments on two benchmark datasets, Thingi10k-Val and GPT-3DP-Val, showing that SEG significantly outperforms baseline models such as TRELLIS, DPO, and DRO in terms of support volume reduction and printability. Qualitative results further reveal that SEG maintains high fidelity to input prompts while minimizing the need for support structures. Our findings highlight the potential of SEG to transform 3D printing by directly optimizing models during the generative process, paving the way for more sustainable and efficient digital fabrication practices.
>
---
#### [new 018] MiMo-Embodied: X-Embodied Foundation Model Technical Report
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出MiMo-Embodied，首个跨具身基础模型，融合自动驾驶与具身智能任务。通过多阶段学习与数据优化，在17项具身AI与12项自动驾驶基准上均达领先性能，验证两领域间正向迁移与协同增强。**

- **链接: [https://arxiv.org/pdf/2511.16518v1](https://arxiv.org/pdf/2511.16518v1)**

> **作者:** Xiaoshuai Hao; Lei Zhou; Zhijian Huang; Zhiwen Hou; Yingbo Tang; Lingfeng Zhang; Guang Li; Zheng Lu; Shuhuai Ren; Xianhui Meng; Yuchen Zhang; Jing Wu; Jinghui Lu; Chenxu Dang; Jiayi Guan; Jianhua Wu; Zhiyi Hou; Hanbing Li; Shumeng Xia; Mingliang Zhou; Yinan Zheng; Zihao Yue; Shuhao Gu; Hao Tian; Yuannan Shen; Jianwei Cui; Wen Zhang; Shaoqing Xu; Bing Wang; Haiyang Sun; Zeyu Zhu; Yuncheng Jiang; Zibin Guo; Chuhong Gong; Chaofan Zhang; Wenbo Ding; Kun Ma; Guang Chen; Rui Cai; Diyun Xiang; Heng Qu; Fuli Luo; Hangjun Ye; Long Chen
>
> **备注:** Code: https://github.com/XiaomiMiMo/MiMo-Embodied Model: https://huggingface.co/XiaomiMiMo/MiMo-Embodied-7B
>
> **摘要:** We open-source MiMo-Embodied, the first cross-embodied foundation model to successfully integrate and achieve state-of-the-art performance in both Autonomous Driving and Embodied AI. MiMo-Embodied sets new records across 17 embodied AI benchmarks in Task Planning, Affordance Prediction and Spatial Understanding, while also excelling in 12 autonomous driving benchmarks across Environmental Perception, Status Prediction, and Driving Planning. Across these tasks, MiMo-Embodied significantly outperforms existing open-source, closed-source, and specialized baselines. Our results indicate that through multi-stage learning, curated data construction, and CoT/RL fine-tuning, these two domains exhibit strong positive transfer and mutually reinforce one another. We provide a detailed analysis of our model design and training methodologies to facilitate further research. Code and models are available at https://github.com/XiaomiMiMo/MiMo-Embodied.
>
---
#### [new 019] The Role of Consequential and Functional Sound in Human-Robot Interaction: Toward Audio Augmented Reality Interfaces
- **分类: cs.RO**

- **简介: 该论文研究人机交互中因果性与功能性声音的影响，旨在提升机器人语音交互体验。通过实验分析声音对感知、定位及情感反应的作用，探索空间声学在任务传递与用户体验优化中的应用，推动音频增强现实接口的发展。**

- **链接: [https://arxiv.org/pdf/2511.15956v1](https://arxiv.org/pdf/2511.15956v1)**

> **作者:** Aliyah Smith; Monroe Kennedy
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** As robots become increasingly integrated into everyday environments, understanding how they communicate with humans is critical. Sound offers a powerful channel for interaction, encompassing both operational noises and intentionally designed auditory cues. In this study, we examined the effects of consequential and functional sounds on human perception and behavior, including a novel exploration of spatial sound through localization and handover tasks. Results show that consequential sounds of the Kinova Gen3 manipulator did not negatively affect perceptions, spatial localization is highly accurate for lateral cues but declines for frontal cues, and spatial sounds can simultaneously convey task-relevant information while promoting warmth and reducing discomfort. These findings highlight the potential of functional and transformative auditory design to enhance human-robot collaboration and inform future sound-based interaction strategies.
>
---
#### [new 020] Safe and Optimal Variable Impedance Control via Certified Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文针对机器人协作任务中变阻抗控制的不稳定性问题，提出基于认证高斯流形采样的强化学习框架C-GMS。通过构建稳定增益调度流形，确保策略探索始终稳定且物理可行，实现安全、最优的运动与交互控制，理论保障跟踪误差有界。**

- **链接: [https://arxiv.org/pdf/2511.16330v1](https://arxiv.org/pdf/2511.16330v1)**

> **作者:** Shreyas Kumar; Ravi Prakash
>
> **摘要:** Reinforcement learning (RL) offers a powerful approach for robots to learn complex, collaborative skills by combining Dynamic Movement Primitives (DMPs) for motion and Variable Impedance Control (VIC) for compliant interaction. However, this model-free paradigm often risks instability and unsafe exploration due to the time-varying nature of impedance gains. This work introduces Certified Gaussian Manifold Sampling (C-GMS), a novel trajectory-centric RL framework that learns combined DMP and VIC policies while guaranteeing Lyapunov stability and actuator feasibility by construction. Our approach reframes policy exploration as sampling from a mathematically defined manifold of stable gain schedules. This ensures every policy rollout is guaranteed to be stable and physically realizable, thereby eliminating the need for reward penalties or post-hoc validation. Furthermore, we provide a theoretical guarantee that our approach ensures bounded tracking error even in the presence of bounded model errors and deployment-time uncertainties. We demonstrate the effectiveness of C-GMS in simulation and verify its efficacy on a real robot, paving the way for reliable autonomous interaction in complex environments.
>
---
#### [new 021] LAOF: Robust Latent Action Learning with Optical Flow Constraints
- **分类: cs.RO**

- **简介: 该论文提出LAOF框架，解决大规模视频中隐式动作学习受干扰的问题。通过利用光学流作为伪监督信号，增强动作表示的鲁棒性，在标签稀缺时仍能有效学习高质量隐式动作表征，显著提升下游模仿学习与强化学习性能。**

- **链接: [https://arxiv.org/pdf/2511.16407v1](https://arxiv.org/pdf/2511.16407v1)**

> **作者:** Xizhou Bu; Jiexi Lyu; Fulei Sun; Ruichen Yang; Zhiqiang Ma; Wei Li
>
> **备注:** Code can be found at https://github.com/XizoB/LAOF
>
> **摘要:** Learning latent actions from large-scale videos is crucial for the pre-training of scalable embodied foundation models, yet existing methods often struggle with action-irrelevant distractors. Although incorporating action supervision can alleviate these distractions, its effectiveness is restricted by the scarcity of available action labels. Optical flow represents pixel-level motion between consecutive frames, naturally suppressing background elements and emphasizing moving objects. Motivated by this, we propose robust Latent Action learning with Optical Flow constraints, called LAOF, a pseudo-supervised framework that leverages the agent's optical flow as an action-driven signal to learn latent action representations robust to distractors. Experimental results show that the latent representations learned by LAOF outperform existing methods on downstream imitation learning and reinforcement learning tasks. This superior performance arises from optical flow constraints, which substantially stabilize training and improve the quality of latent representations under extremely label-scarce conditions, while remaining effective as the proportion of action labels increases to 10 percent. Importantly, even without action supervision, LAOF matches or surpasses action-supervised methods trained with 1 percent of action labels.
>
---
#### [new 022] Funabot-Upper: McKibben Actuated Haptic Suit Inducing Kinesthetic Perceptions in Trunk, Shoulder, Elbow, and Wrist
- **分类: cs.RO**

- **简介: 该论文属于可穿戴触觉设备任务，旨在解决多部位运动感知中感知混淆问题。通过简化设计策略，开发了可独立刺激躯干、肩、肘、腕关节的Funabot-Upper，利用气动人工肌肉实现三维形变，显著提升运动感知准确率（68.8%→94.6%），有效减少跨部位干扰。**

- **链接: [https://arxiv.org/pdf/2511.16265v1](https://arxiv.org/pdf/2511.16265v1)**

> **作者:** Haru Fukatsu; Ryoji Yasuda; Yuki Funabora; Shinji Doki
>
> **备注:** 8 pages, 8 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper presents Funabot-Upper, a wearable haptic suit that enables users to perceive 14 upper-body motions, including those of the trunk, shoulder, elbow, and wrist. Inducing kinesthetic perception through wearable haptic devices has attracted attention, and various devices have been developed in the past. However, these have been limited to verifications on single body parts, and few have applied the same method to multiple body parts as well. In our previous study, we developed a technology that uses the contraction of artificial muscles to deform clothing in three dimensions. Using this technology, we developed a haptic suit that induces kinesthetic perception of 7 motions in multiple upper body. However, perceptual mixing caused by stimulating multiple human muscles has occurred between the shoulder and the elbow. In this paper, we established a new, simplified design policy and developed a novel haptic suit that induces kinesthetic perceptions in the trunk, shoulder, elbow, and wrist by stimulating joints and muscles independently. We experimentally demonstrated the induced kinesthetic perception and examined the relationship between stimulation and perceived kinesthetic perception under the new design policy. Experiments confirmed that Funabot-Upper successfully induces kinesthetic perception across multiple joints while reducing perceptual mixing observed in previous designs. The new suit improved recognition accuracy from 68.8% to 94.6% compared to the previous Funabot-Suit, demonstrating its superiority and potential for future haptic applications.
>
---
#### [new 023] Green Resilience of Cyber-Physical Systems: Doctoral Dissertation
- **分类: cs.SE; cs.AI; cs.CV; cs.RO**

- **简介: 该论文研究在线协作人工智能系统（OL-CAIS）的绿色韧性问题，旨在平衡系统在扰动后的恢复能力与能耗。通过构建三态模型与GResilience框架，提出多目标优化、博弈决策与强化学习策略，实现高效绿色恢复，并量化韧性与绿色度。实验验证了方法有效性，揭示了灾难性遗忘现象并提出缓解措施。**

- **链接: [https://arxiv.org/pdf/2511.16593v1](https://arxiv.org/pdf/2511.16593v1)**

> **作者:** Diaeddin Rimawi
>
> **摘要:** Cyber-physical systems (CPS) combine computational and physical components. Online Collaborative AI System (OL-CAIS) is a type of CPS that learn online in collaboration with humans to achieve a common goal, which makes it vulnerable to disruptive events that degrade performance. Decision-makers must therefore restore performance while limiting energy impact, creating a trade-off between resilience and greenness. This research addresses how to balance these two properties in OL-CAIS. It aims to model resilience for automatic state detection, develop agent-based policies that optimize the greenness-resilience trade-off, and understand catastrophic forgetting to maintain performance consistency. We model OL-CAIS behavior through three operational states: steady, disruptive, and final. To support recovery during disruptions, we introduce the GResilience framework, which provides recovery strategies through multi-objective optimization (one-agent), game-theoretic decision-making (two-agent), and reinforcement learning (RL-agent). We also design a measurement framework to quantify resilience and greenness. Empirical evaluation uses real and simulated experiments with a collaborative robot learning object classification from human demonstrations. Results show that the resilience model captures performance transitions during disruptions, and that GResilience policies improve green recovery by shortening recovery time, stabilizing performance, and reducing human dependency. RL-agent policies achieve the strongest results, although with a marginal increase in CO2 emissions. We also observe catastrophic forgetting after repeated disruptions, while our policies help maintain steadiness. A comparison with containerized execution shows that containerization cuts CO2 emissions by half. Overall, this research provides models, metrics, and policies that ensure the green recovery of OL-CAIS.
>
---
#### [new 024] LEGO-SLAM: Language-Embedded Gaussian Optimization SLAM
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LEGO-SLAM，面向3DGS-based SLAM系统，解决现有方法缺乏开放词汇语义理解、内存占用高及适应性差的问题。通过自适应编码器将语言嵌入压缩至16维，实现实时开放词汇建图，支持语言引导删减与回环检测，显著降低冗余点云，提升效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.16144v1](https://arxiv.org/pdf/2511.16144v1)**

> **作者:** Sibaek Lee; Seongbo Ha; Kyeongsu Kang; Joonyeol Choi; Seungjun Tak; Hyeonwoo Yu
>
> **备注:** 18 pages
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have enabled Simultaneous Localization and Mapping (SLAM) systems to build photorealistic maps. However, these maps lack the open-vocabulary semantic understanding required for advanced robotic interaction. Integrating language features into SLAM remains a significant challenge, as storing high-dimensional features demands excessive memory and rendering overhead, while existing methods with static models lack adaptability for novel environments. To address these limitations, we propose LEGO-SLAM (Language-Embedded Gaussian Optimization SLAM), the first framework to achieve real-time, open-vocabulary mapping within a 3DGS-based SLAM system. At the core of our method is a scene-adaptive encoder-decoder that distills high-dimensional language embeddings into a compact 16-dimensional feature space. This design reduces the memory per Gaussian and accelerates rendering, enabling real-time performance. Unlike static approaches, our encoder adapts online to unseen scenes. These compact features also enable a language-guided pruning strategy that identifies semantic redundancy, reducing the map's Gaussian count by over 60\% while maintaining rendering quality. Furthermore, we introduce a language-based loop detection approach that reuses these mapping features, eliminating the need for a separate detection model. Extensive experiments demonstrate that LEGO-SLAM achieves competitive mapping quality and tracking accuracy, all while providing open-vocabulary capabilities at 15 FPS.
>
---
#### [new 025] Heterogeneous Stroke: Using Unique Vibration Cues to Improve the Wrist-Worn Spatiotemporal Tactile Display
- **分类: cs.HC; cs.RO**

- **简介: 该论文针对腕戴触觉显示中因皮肤面积小、空间分辨力低导致的触觉刺激混淆问题，提出Heterogeneous Stroke设计。通过为每个触觉执行器分配独特振动线索，提升字母和数字的识别准确率（93.8%和92.4%），且在不同手臂姿势下仍保持高精度，有效改善了腕戴式时空触觉显示的性能。**

- **链接: [https://arxiv.org/pdf/2511.16133v1](https://arxiv.org/pdf/2511.16133v1)**

> **作者:** Taejun Kim; Youngbo Aram Shim; Geehyuk Lee
>
> **备注:** ACM CHI 2021
>
> **摘要:** Beyond a simple notification of incoming calls or messages, more complex information such as alphabets and digits can be delivered through spatiotemporal tactile patterns (STPs) on a wrist-worn tactile display (WTD) with multiple tactors. However, owing to the limited skin area and spatial acuity of the wrist, frequent confusions occur between closely located tactors, resulting in a low recognition accuracy. Furthermore, the accuracies reported in previous studies have mostly been measured for a specific posture and could further decrease with free arm postures in real life. Herein, we present Heterogeneous Stroke, a design concept for improving the recognition accuracy of STPs on a WTD. By assigning unique vibrotactile stimuli to each tactor, the confusion between tactors can be reduced. Through our implementation of Heterogeneous Stroke, the alphanumeric characters could be delivered with high accuracy (93.8% for 26 alphabets and 92.4% for 10 digits) across different arm postures.
>
---
#### [new 026] The Shawshank Redemption of Embodied AI: Understanding and Benchmarking Indirect Environmental Jailbreaks
- **分类: cs.CR; cs.CY; cs.RO**

- **简介: 该论文研究 embodied AI 中的间接环境越狱（IEJ）问题，提出首个自动攻击生成框架 SHAWSHANK 与基准构建框架 SHAWSHANK-FORGE，构建首个 IEJ 基准数据集。通过在环境中植入恶意指令实现对视觉语言模型的隐蔽越狱，揭示现有防御的局限性，推动安全评估与防护发展。**

- **链接: [https://arxiv.org/pdf/2511.16347v1](https://arxiv.org/pdf/2511.16347v1)**

> **作者:** Chunyang Li; Zifeng Kang; Junwei Zhang; Zhuo Ma; Anda Cheng; Xinghua Li; Jianfeng Ma
>
> **摘要:** The adoption of Vision-Language Models (VLMs) in embodied AI agents, while being effective, brings safety concerns such as jailbreaking. Prior work have explored the possibility of directly jailbreaking the embodied agents through elaborated multi-modal prompts. However, no prior work has studied or even reported indirect jailbreaks in embodied AI, where a black-box attacker induces a jailbreak without issuing direct prompts to the embodied agent. In this paper, we propose, for the first time, indirect environmental jailbreak (IEJ), a novel attack to jailbreak embodied AI via indirect prompt injected into the environment, such as malicious instructions written on a wall. Our key insight is that embodied AI does not ''think twice'' about the instructions provided by the environment -- a blind trust that attackers can exploit to jailbreak the embodied agent. We further design and implement open-source prototypes of two fully-automated frameworks: SHAWSHANK, the first automatic attack generation framework for the proposed attack IEJ; and SHAWSHANK-FORGE, the first automatic benchmark generation framework for IEJ. Then, using SHAWSHANK-FORGE, we automatically construct SHAWSHANK-BENCH, the first benchmark for indirectly jailbreaking embodied agents. Together, our two frameworks and one benchmark answer the questions of what content can be used for malicious IEJ instructions, where they should be placed, and how IEJ can be systematically evaluated. Evaluation results show that SHAWSHANK outperforms eleven existing methods across 3,957 task-scene combinations and compromises all six tested VLMs. Furthermore, current defenses only partially mitigate our attack, and we have responsibly disclosed our findings to all affected VLM vendors.
>
---
#### [new 027] Towards a Safer and Sustainable Manufacturing Process: Material classification in Laser Cutting Using Deep Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于材料分类任务，旨在解决激光切割中因烟尘危害和材料识别不准带来的安全与效率问题。通过构建基于深度学习的卷积神经网络，利用激光散斑图实现不同材料的高精度实时识别，即使激光颜色改变也能保持稳定性能，显著提升切割过程的安全性与可持续性。**

- **链接: [https://arxiv.org/pdf/2511.16026v1](https://arxiv.org/pdf/2511.16026v1)**

> **作者:** Mohamed Abdallah Salem; Hamdy Ahmed Ashur; Ahmed Elshinnawy
>
> **摘要:** Laser cutting is a widely adopted technology in material processing across various industries, but it generates a significant amount of dust, smoke, and aerosols during operation, posing a risk to both the environment and workers' health. Speckle sensing has emerged as a promising method to monitor the cutting process and identify material types in real-time. This paper proposes a material classification technique using a speckle pattern of the material's surface based on deep learning to monitor and control the laser cutting process. The proposed method involves training a convolutional neural network (CNN) on a dataset of laser speckle patterns to recognize distinct material types for safe and efficient cutting. Previous methods for material classification using speckle sensing may face issues when the color of the laser used to produce the speckle pattern is changed. Experiments conducted in this study demonstrate that the proposed method achieves high accuracy in material classification, even when the laser color is changed. The model achieved an accuracy of 98.30 % on the training set and 96.88% on the validation set. Furthermore, the model was evaluated on a set of 3000 new images for 30 different materials, achieving an F1-score of 0.9643. The proposed method provides a robust and accurate solution for material-aware laser cutting using speckle sensing.
>
---
## 更新

#### [replaced 001] RAPID: Robust and Agile Planner Using Inverse Reinforcement Learning for Vision-Based Drone Navigation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2502.02054v2](https://arxiv.org/pdf/2502.02054v2)**

> **作者:** Minwoo Kim; Geunsik Bae; Jinwoo Lee; Woojae Shin; Changseung Kim; Myong-Yol Choi; Heejung Shin; Hyondong Oh
>
> **备注:** 18 pages, 11 figures, 58 references, and appendix is included
>
> **摘要:** This paper introduces a learning-based visual planner for agile drone flight in cluttered environments. The proposed planner generates collision-free waypoints in milliseconds, enabling drones to perform agile maneuvers in complex environments without building separate perception, mapping, and planning modules. Learning-based methods, such as behavior cloning (BC) and reinforcement learning (RL), demonstrate promising performance in visual navigation but still face inherent limitations. BC is susceptible to compounding errors due to limited expert imitation, while RL struggles with reward function design and sample inefficiency. To address these limitations, this paper proposes an inverse reinforcement learning (IRL)-based framework for high-speed visual navigation. By leveraging IRL, it is possible to reduce the number of interactions with simulation environments and improve capability to deal with high-dimensional spaces while preserving the robustness of RL policies. A motion primitive-based path planning algorithm collects an expert dataset with privileged map data from diverse environments, ensuring comprehensive scenario coverage. By leveraging both the acquired expert and learner dataset gathered from the agent's interactions with the simulation environments, a robust reward function and policy are learned across diverse states. While the proposed method is trained in a simulation environment only, it can be directly applied to real-world scenarios without additional training or tuning. The performance of the proposed method is validated in both simulation and real-world environments, including forests and various structures. The trained policy achieves an average speed of 7 m/s and a maximum speed of 8.8 m/s in real flight experiments. To the best of our knowledge, this is the first work to successfully apply an IRL framework for high-speed visual navigation of drones.
>
---
#### [replaced 002] Grounding LLMs For Robot Task Planning Using Closed-loop State Feedback
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2402.08546v3](https://arxiv.org/pdf/2402.08546v3)**

> **作者:** Vineet Bhat; Ali Umut Kaypak; Prashanth Krishnamurthy; Ramesh Karri; Farshad Khorrami
>
> **备注:** Preprint version. Accepted full paper available here: https://advanced.onlinelibrary.wiley.com/doi/10.1002/adrr.202500072
>
> **摘要:** Planning algorithms decompose complex problems into intermediate steps that can be sequentially executed by robots to complete tasks. Recent works have employed Large Language Models (LLMs) for task planning, using natural language to generate robot policies in both simulation and real-world environments. LLMs like GPT-4 have shown promising results in generalizing to unseen tasks, but their applicability is limited due to hallucinations caused by insufficient grounding in the robot environment. The robustness of LLMs in task planning can be enhanced with environmental state information and feedback. In this paper, we introduce a novel approach to task planning that utilizes two separate LLMs for high-level planning and low-level control, improving task-related success rates and goal condition recall. Our algorithm, \textit{BrainBody-LLM}, draws inspiration from the human neural system, emulating its brain-body architecture by dividing planning across two LLMs in a structured, hierarchical manner. BrainBody-LLM implements a closed-loop feedback mechanism, enabling learning from simulator errors to resolve execution errors in complex settings. We demonstrate the successful application of BrainBody-LLM in the VirtualHome simulation environment, achieving a 29\% improvement in task-oriented success rates over competitive baselines with the GPT-4 backend. Additionally, we evaluate our algorithm on seven complex tasks using a realistic physics simulator and the Franka Research 3 robotic arm, comparing it with various state-of-the-art LLMs. Our results show advancements in the reasoning capabilities of recent LLMs, which enable them to learn from raw simulator/controller errors to correct plans, making them highly effective in robotic task planning.
>
---
#### [replaced 003] Risk Map As Middleware: Towards Interpretable Cooperative End-to-end Autonomous Driving for Risk-Aware Planning
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2508.07686v2](https://arxiv.org/pdf/2508.07686v2)**

> **作者:** Mingyue Lei; Zewei Zhou; Hongchen Li; Jiaqi Ma; Jia Hu
>
> **备注:** IEEE RA-L
>
> **摘要:** End-to-end paradigm has emerged as a promising approach to autonomous driving. However, existing single-agent end-to-end pipelines are often constrained by occlusion and limited perception range, resulting in hazardous driving. Furthermore, their black-box nature prevents the interpretability of the driving behavior, leading to an untrustworthiness system. To address these limitations, we introduce Risk Map as Middleware (RiskMM) and propose an interpretable cooperative end-to-end driving framework. The risk map learns directly from the driving data and provides an interpretable spatiotemporal representation of the scenario from the upstream perception and the interactions between the ego vehicle and the surrounding environment for downstream planning. RiskMM first constructs a multi-agent spatiotemporal representation with unified Transformer-based architecture, then derives risk-aware representations by modeling interactions among surrounding environments with attention. These representations are subsequently fed into a learning-based Model Predictive Control (MPC) module. The MPC planner inherently accommodates physical constraints and different vehicle types and can provide interpretation by aligning learned parameters with explicit MPC elements. Evaluations conducted on the real-world V2XPnP-Seq dataset confirm that RiskMM achieves superior and robust performance in risk-aware trajectory planning, significantly enhancing the interpretability of the cooperative end-to-end driving framework. The codebase will be released to facilitate future research in this field.
>
---
#### [replaced 004] Non-Gaited Legged Locomotion with Monte-Carlo Tree Search and Supervised Learning
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2408.07508v4](https://arxiv.org/pdf/2408.07508v4)**

> **作者:** Ilyass Taouil; Lorenzo Amatucci; Majid Khadiv; Angela Dai; Victor Barasuol; Giulio Turrisi; Claudio Semini
>
> **摘要:** Legged robots are able to navigate complex terrains by continuously interacting with the environment through careful selection of contact sequences and timings. However, the combinatorial nature behind contact planning hinders the applicability of such optimization problems on hardware. In this work, we present a novel approach that optimizes gait sequences and respective timings for legged robots in the context of optimization-based controllers through the use of sampling-based methods and supervised learning techniques. We propose to bootstrap the search by learning an optimal value function in order to speed-up the gait planning procedure making it applicable in real-time. To validate our proposed method, we showcase its performance both in simulation and on hardware using a 22 kg electric quadruped robot. The method is assessed on different terrains, under external perturbations, and in comparison to a standard control approach where the gait sequence is fixed a priori.
>
---
#### [replaced 005] A Continuous sEMG-Based Prosthetic Hand Control System Without Motion or Force Sensors
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2407.00014v3](https://arxiv.org/pdf/2407.00014v3)**

> **作者:** Gang Liu; Ye Sun; Zhenxiang Wang; Chuanmei Xi; Ziyang He; Shanshan Guo; Rui Zhang; Dezhong Yao
>
> **备注:** 12 pages
>
> **摘要:** Regressively-based surface electromyography (sEMG) prosthetics are widely used for their ability to continuously convert muscle activity into finger force and motion. However, they typically require additional kinematic or dynamic sensors, which increases complexity and limits practical application. To address this, this paper proposes a method based on the simplified near-linear relationship between sEMG and finger force, using the near-linear model ResDD proposed in this work. By applying the principle that a line can be determined by two points, we eliminate the need for complex sensor calibration. Specifically, by recording the sEMG during maximum finger flexion and extension, and assigning corresponding forces of 1 and -1, the ResDD model can fit the simplified relationship between sEMG signals and force, enabling continuous prediction and control of finger force and gestures. Offline experiments were conducted to evaluate the model's classification accuracy and its ability to learn sufficient information. It uses interpolation analysis to open up the internal structure of the trained model and checks whether the fitted curve of the model conforms to the nearly linear relationship between sEMG and force. Finally, online control and sine wave tracking experiments were carried out to further verify the practicality of the proposed method. The results show that the method effectively extracts meaningful information from sEMG and accurately decodes them. The near-linear model sufficiently reflects the expected relationship between sEMG and finger force. Fitting this simplified near-linear relationship is adequate to achieve continuous and smooth control of finger force and gestures, confirming the feasibility and effectiveness of the proposed approach.
>
---
#### [replaced 006] Barrier-Riccati Synthesis for Nonlinear Safe Control with Expanded Region of Attraction
- **分类: eess.SY; cs.RO**

- **链接: [https://arxiv.org/pdf/2504.15453v2](https://arxiv.org/pdf/2504.15453v2)**

> **作者:** Hassan Almubarak; Maitham F. AL-Sunni; Justin T. Dubbin; Nader Sadegh; John M. Dolan; Evangelos A. Theodorou
>
> **摘要:** We present a Riccati-based framework for safety-critical nonlinear control that integrates the barrier states (BaS) methodology with the State-Dependent Riccati Equation (SDRE) approach. The BaS formulation embeds safety constraints into the system dynamics via auxiliary states, enabling safety to be treated as a control objective. To overcome the limited region of attraction in linear BaS controllers, we extend the framework to nonlinear systems using SDRE synthesis applied to the barrier-augmented dynamics and derive a matrix inequality condition that certifies forward invariance of a large region of attraction and guarantees asymptotic safe stabilization. The resulting controller is computed online via pointwise Riccati solutions. We validate the method on an unstable constrained system and cluttered quadrotor navigation tasks, demonstrating improved constraint handling, scalability, and robustness near safety boundaries. This framework offers a principled and computationally tractable solution for synthesizing nonlinear safe feedback in safety-critical environments.
>
---
#### [replaced 007] UltraDP: Generalizable Carotid Ultrasound Scanning with Force-Aware Diffusion Policy
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.15550v2](https://arxiv.org/pdf/2511.15550v2)**

> **作者:** Ruoqu Chen; Xiangjie Yan; Kangchen Lv; Gao Huang; Zheng Li; Xiang Li
>
> **摘要:** Ultrasound scanning is a critical imaging technique for real-time, non-invasive diagnostics. However, variations in patient anatomy and complex human-in-the-loop interactions pose significant challenges for autonomous robotic scanning. Existing ultrasound scanning robots are commonly limited to relatively low generalization and inefficient data utilization. To overcome these limitations, we present UltraDP, a Diffusion-Policy-based method that receives multi-sensory inputs (ultrasound images, wrist camera images, contact wrench, and probe pose) and generates actions that are fit for multi-modal action distributions in autonomous ultrasound scanning of carotid artery. We propose a specialized guidance module to enable the policy to output actions that center the artery in ultrasound images. To ensure stable contact and safe interaction between the robot and the human subject, a hybrid force-impedance controller is utilized to drive the robot to track such trajectories. Also, we have built a large-scale training dataset for carotid scanning comprising 210 scans with 460k sample pairs from 21 volunteers of both genders. By exploring our guidance module and DP's strong generalization ability, UltraDP achieves a 95% success rate in transverse scanning on previously unseen subjects, demonstrating its effectiveness.
>
---
#### [replaced 008] Statistically Assuring Safety of Control Systems using Ensembles of Safety Filters and Conformal Prediction
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2511.07899v2](https://arxiv.org/pdf/2511.07899v2)**

> **作者:** Ihab Tabbara; Yuxuan Yang; Hussein Sibai
>
> **摘要:** Safety assurance is a fundamental requirement for deploying learning-enabled autonomous systems. Hamilton-Jacobi (HJ) reachability analysis is a fundamental method for formally verifying safety and generating safe controllers. However, computing the HJ value function that characterizes the backward reachable set (BRS) of a set of user-defined failure states is computationally expensive, especially for high-dimensional systems, motivating the use of reinforcement learning approaches to approximate the value function. Unfortunately, a learned value function and its corresponding safe policy are not guaranteed to be correct. The learned value function evaluated at a given state may not be equal to the actual safety return achieved by following the learned safe policy. To address this challenge, we introduce a conformal prediction-based (CP) framework that bounds such uncertainty. We leverage CP to provide probabilistic safety guarantees when using learned HJ value functions and policies to prevent control systems from reaching failure states. Specifically, we use CP to calibrate the switching between the unsafe nominal controller and the learned HJ-based safe policy and to derive safety guarantees under this switched policy. We also investigate using an ensemble of independently trained HJ value functions as a safety filter and compare this ensemble approach to using individual value functions alone.
>
---
#### [replaced 009] Vector Quantized-Elites: Unsupervised and Problem-Agnostic Quality-Diversity Optimization
- **分类: cs.NE; cs.AI; cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2504.08057v3](https://arxiv.org/pdf/2504.08057v3)**

> **作者:** Constantinos Tsakonas; Konstantinos Chatzilygeroudis
>
> **备注:** 15 pages (+4 supplementary), 14 (+1) figures, 1 algorithm, 1 (+8) table(s), accepted at IEEE Transactions on Evolutionary Computation
>
> **摘要:** Quality-Diversity algorithms have transformed optimization by prioritizing the discovery of diverse, high-performing solutions over a single optimal result. However, traditional Quality-Diversity methods, such as MAP-Elites, rely heavily on predefined behavior descriptors and complete prior knowledge of the task to define the behavior space grid, limiting their flexibility and applicability. In this work, we introduce Vector Quantized-Elites (VQ-Elites), a novel Quality-Diversity algorithm that autonomously constructs a structured behavior space grid using unsupervised learning, eliminating the need for prior task-specific knowledge. At the core of VQ-Elites is the integration of Vector Quantized Variational Autoencoders, which enables the dynamic learning of behavior descriptors and the generation of a structured, rather than unstructured, behavior space grid -- a significant advancement over existing unsupervised Quality-Diversity approaches. This design establishes VQ-Elites as a flexible, robust, and task-agnostic optimization framework. To further enhance the performance of unsupervised Quality-Diversity algorithms, we introduce behavior space bounding and cooperation mechanisms, which significantly improve convergence and performance, as well as the Effective Diversity Ratio and Coverage Diversity Score, two novel metrics that quantify the actual diversity in the unsupervised setting. We validate VQ-Elites on robotic arm pose-reaching, mobile robot space-covering, and MiniGrid exploration tasks. The results demonstrate its ability to efficiently generate diverse, high-quality solutions, emphasizing its adaptability, scalability, robustness to hyperparameters, and potential to extend Quality-Diversity optimization to complex, previously inaccessible domains.
>
---
#### [replaced 010] Relative Pose Estimation for Nonholonomic Robot Formation with UWB-IO Measurements (Extended version)
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2411.05481v4](https://arxiv.org/pdf/2411.05481v4)**

> **作者:** Kunrui Ze; Wei Wang; Shuoyu Yue; Guibin Sun; Kexin Liu; Jinhu Lü
>
> **备注:** 17 pages, 26 figures
>
> **摘要:** This article studies the problem of distributed formation control for multiple robots by using onboard ultra wide band (UWB) distance and inertial odometer (IO) measurements. Although this problem has been widely studied, a fundamental limitation of most works is that they require each robot's pose and sensor measurements are expressed in a common reference frame. However, it is inapplicable for nonholonomic robot formations due to the practical difficulty of aligning IO measurements of individual robot in a common frame. To address this problem, firstly, a concurrent-learning based estimator is firstly proposed to achieve relative localization between neighboring robots in a local frame. Different from most relative localization methods in a global frame, both relative position and orientation in a local frame are estimated with only UWB ranging and IO measurements. Secondly, to deal with information loss caused by directed communication topology, a cooperative localization algorithm is introduced to estimate the relative pose to the leader robot. Thirdly, based on the theoretical results on relative pose estimation, a distributed formation tracking controller is proposed for nonholonomic robots. Both 3D and 2D real-world experiments conducted on aerial robots and grounded robots are provided to demonstrate the effectiveness of the proposed method.
>
---
#### [replaced 011] CleverDistiller: Simple and Spatially Consistent Cross-modal Distillation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2503.09878v3](https://arxiv.org/pdf/2503.09878v3)**

> **作者:** Hariprasath Govindarajan; Maciej K. Wozniak; Marvin Klingner; Camille Maurice; B Ravi Kiran; Senthil Yogamani
>
> **备注:** Accepted to BMVC 2025
>
> **摘要:** Vision foundation models (VFMs) such as DINO have led to a paradigm shift in 2D camera-based perception towards extracting generalized features to support many downstream tasks. Recent works introduce self-supervised cross-modal knowledge distillation (KD) as a way to transfer these powerful generalization capabilities into 3D LiDAR-based models. However, they either rely on highly complex distillation losses, pseudo-semantic maps, or limit KD to features useful for semantic segmentation only. In this work, we propose CleverDistiller, a self-supervised, cross-modal 2D-to-3D KD framework introducing a set of simple yet effective design choices: Unlike contrastive approaches relying on complex loss design choices, our method employs a direct feature similarity loss in combination with a multi layer perceptron (MLP) projection head to allow the 3D network to learn complex semantic dependencies throughout the projection. Crucially, our approach does not depend on pseudo-semantic maps, allowing for direct knowledge transfer from a VFM without explicit semantic supervision. Additionally, we introduce the auxiliary self-supervised spatial task of occupancy prediction to enhance the semantic knowledge, obtained from a VFM through KD, with 3D spatial reasoning capabilities. Experiments on standard autonomous driving benchmarks for 2D-to-3D KD demonstrate that CleverDistiller achieves state-of-the-art performance in both semantic segmentation and 3D object detection (3DOD) by up to 10% mIoU, especially when fine tuning on really low data amounts, showing the effectiveness of our simple yet powerful KD strategy
>
---
