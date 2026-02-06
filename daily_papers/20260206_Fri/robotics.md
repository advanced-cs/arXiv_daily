# 机器人 cs.RO

- **最新发布 47 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] Visuo-Tactile World Models
- **分类: cs.RO**

- **简介: 该论文提出多任务视觉触觉世界模型（VT-WM），解决机器人在接触密集任务中对物理交互理解不足的问题。通过融合视觉与触觉感知，提升物体恒常性和运动规律遵守，增强规划能力与任务适应性。**

- **链接: [https://arxiv.org/pdf/2602.06001v1](https://arxiv.org/pdf/2602.06001v1)**

> **作者:** Carolina Higuera; Sergio Arnaud; Byron Boots; Mustafa Mukadam; Francois Robert Hogan; Franziska Meier
>
> **备注:** Preprint
>
> **摘要:** We introduce multi-task Visuo-Tactile World Models (VT-WM), which capture the physics of contact through touch reasoning. By complementing vision with tactile sensing, VT-WM better understands robot-object interactions in contact-rich tasks, avoiding common failure modes of vision-only models under occlusion or ambiguous contact states, such as objects disappearing, teleporting, or moving in ways that violate basic physics. Trained across a set of contact-rich manipulation tasks, VT-WM improves physical fidelity in imagination, achieving 33% better performance at maintaining object permanence and 29% better compliance with the laws of motion in autoregressive rollouts. Moreover, experiments show that grounding in contact dynamics also translates to planning. In zero-shot real-robot experiments, VT-WM achieves up to 35% higher success rates, with the largest gains in multi-step, contact-rich tasks. Finally, VT-WM demonstrates significant downstream versatility, effectively adapting its learned contact dynamics to a novel task and achieving reliable planning success with only a limited set of demonstrations.
>
---
#### [new 002] TOLEBI: Learning Fault-Tolerant Bipedal Locomotion via Online Status Estimation and Fallibility Rewards
- **分类: cs.RO**

- **简介: 该论文属于双足机器人运动控制任务，旨在解决硬件故障下的容错问题。通过在线状态估计和奖励机制，提升机器人在故障情况下的运动稳定性。**

- **链接: [https://arxiv.org/pdf/2602.05596v1](https://arxiv.org/pdf/2602.05596v1)**

> **作者:** Hokyun Lee; Woo-Jeong Baek; Junhyeok Cha; Jaeheung Park
>
> **备注:** Accepted for Publication at IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** With the growing employment of learning algorithms in robotic applications, research on reinforcement learning for bipedal locomotion has become a central topic for humanoid robotics. While recently published contributions achieve high success rates in locomotion tasks, scarce attention has been devoted to the development of methods that enable to handle hardware faults that may occur during the locomotion process. However, in real-world settings, environmental disturbances or sudden occurrences of hardware faults might yield severe consequences. To address these issues, this paper presents TOLEBI (A faulT-tOlerant Learning framEwork for Bipedal locomotIon) that handles faults on the robot during operation. Specifically, joint locking, power loss and external disturbances are injected in simulation to learn fault-tolerant locomotion strategies. In addition to transferring the learned policy to the real robot via sim-to-real transfer, an online joint status module incorporated. This module enables to classify joint conditions by referring to the actual observations at runtime under real-world conditions. The validation experiments conducted both in real-world and simulation with the humanoid robot TOCABI highlight the applicability of the proposed approach. To our knowledge, this manuscript provides the first learning-based fault-tolerant framework for bipedal locomotion, thereby fostering the development of efficient learning methods in this field.
>
---
#### [new 003] PLATO Hand: Shaping Contact Behavior with Fingernails for Precise Manipulation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人抓取任务，旨在解决精确操作中的接触控制问题。通过设计带有刚性指甲的柔性指尖，提升抓取稳定性与力感知能力，实现复杂物体的精准操作。**

- **链接: [https://arxiv.org/pdf/2602.05156v1](https://arxiv.org/pdf/2602.05156v1)**

> **作者:** Dong Ho Kang; Aaron Kim; Mingyo Seo; Kazuto Yokoyama; Tetsuya Narita; Luis Sentis
>
> **摘要:** We present the PLATO Hand, a dexterous robotic hand with a hybrid fingertip that embeds a rigid fingernail within a compliant pulp. This design shapes contact behavior to enable diverse interaction modes across a range of object geometries. We develop a strain-energy-based bending-indentation model to guide the fingertip design and to explain how guided contact preserves local indentation while suppressing global bending. Experimental results show that the proposed robotic hand design demonstrates improved pinching stability, enhanced force observability, and successful execution of edge-sensitive manipulation tasks, including paper singulation, card picking, and orange peeling. Together, these results show that coupling structured contact geometry with a force-motion transparent mechanism provides a principled, physically embodied approach to precise manipulation.
>
---
#### [new 004] Ontology-Driven Robotic Specification Synthesis
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文提出RSTM2方法，解决机器人系统从高层目标到形式化规范的转化问题，通过本体驱动的随机定时Petri网实现自主规范合成。**

- **链接: [https://arxiv.org/pdf/2602.05456v1](https://arxiv.org/pdf/2602.05456v1)**

> **作者:** Maksym Figat; Ryan M. Mackey; Michel D. Ingham
>
> **备注:** 8 pages, 9 figures, 3 tables, journal
>
> **摘要:** This paper addresses robotic system engineering for safety- and mission-critical applications by bridging the gap between high-level objectives and formal, executable specifications. The proposed method, Robotic System Task to Model Transformation Methodology (RSTM2) is an ontology-driven, hierarchical approach using stochastic timed Petri nets with resources, enabling Monte Carlo simulations at mission, system, and subsystem levels. A hypothetical case study demonstrates how the RSTM2 method supports architectural trades, resource allocation, and performance analysis under uncertainty. Ontological concepts further enable explainable AI-based assistants, facilitating fully autonomous specification synthesis. The methodology offers particular benefits to complex multi-robot systems, such as the NASA CADRE mission, representing decentralized, resource-aware, and adaptive autonomous systems of the future.
>
---
#### [new 005] Informative Path Planning with Guaranteed Estimation Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于环境监测任务，解决路径规划中如何保证估计不确定性的问题。通过结合高斯过程与路径规划，确保在有限资源下满足监测区域的不确定性约束。**

- **链接: [https://arxiv.org/pdf/2602.05198v1](https://arxiv.org/pdf/2602.05198v1)**

> **作者:** Kalvik Jakkala; Saurav Agarwal; Jason O'Kane; Srinivas Akella
>
> **备注:** 16 pages, 11 figures, preprint
>
> **摘要:** Environmental monitoring robots often need to reconstruct spatial fields (e.g., salinity, temperature, bathymetry) under tight distance and energy constraints. Classical boustrophedon lawnmower surveys provide geometric coverage guarantees but can waste effort by oversampling predictable regions. In contrast, informative path planning (IPP) methods leverage spatial correlations to reduce oversampling, yet typically offer no guarantees on reconstruction quality. This paper bridges these approaches by addressing informative path planning with guaranteed estimation uncertainty: computing the shortest path whose measurements ensure that the Gaussian-process (GP) posterior variance -- an intrinsic uncertainty measure that lower-bounds the mean-squared prediction error under the GP model -- falls below a user-specified threshold over the monitoring region. We propose a three-stage approach: (i) learn a GP model from available prior information; (ii) transform the learned GP kernel into binary coverage maps for each candidate sensing location, indicating which locations' uncertainty can be reduced below a specified target; and (iii) plan a near-shortest route whose combined coverage satisfies the global uncertainty constraint. To address heterogeneous phenomena, we incorporate a nonstationary kernel that captures spatially varying correlation structure, and we accommodate non-convex environments with obstacles. Algorithmically, we present methods with provable approximation guarantees for sensing-location selection and for the joint selection-and-routing problem under a travel budget. Experiments on real-world topographic data show that our planners meet the uncertainty target using fewer sensing locations and shorter travel distances than a recent baseline, and field experiments with bathymetry-mapping autonomous surface and underwater vehicles demonstrate real-world feasibility.
>
---
#### [new 006] Differentiable Inverse Graphics for Zero-shot Scene Reconstruction and Robot Grasping
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人抓取与场景重建任务，解决无监督下物体姿态估计与场景重构问题。通过结合神经图形学与物理渲染，从单张RGBD图像中实现零样本场景重建和抓取。**

- **链接: [https://arxiv.org/pdf/2602.05029v1](https://arxiv.org/pdf/2602.05029v1)**

> **作者:** Octavio Arriaga; Proneet Sharma; Jichen Guo; Marc Otto; Siddhant Kadwe; Rebecca Adam
>
> **备注:** Submitted to IEEE Robotics and Automation Letters (RA-L) for review. This version includes the statement required by IEEE for preprints
>
> **摘要:** Operating effectively in novel real-world environments requires robotic systems to estimate and interact with previously unseen objects. Current state-of-the-art models address this challenge by using large amounts of training data and test-time samples to build black-box scene representations. In this work, we introduce a differentiable neuro-graphics model that combines neural foundation models with physics-based differentiable rendering to perform zero-shot scene reconstruction and robot grasping without relying on any additional 3D data or test-time samples. Our model solves a series of constrained optimization problems to estimate physically consistent scene parameters, such as meshes, lighting conditions, material properties, and 6D poses of previously unseen objects from a single RGBD image and bounding boxes. We evaluated our approach on standard model-free few-shot benchmarks and demonstrated that it outperforms existing algorithms for model-free few-shot pose estimation. Furthermore, we validated the accuracy of our scene reconstructions by applying our algorithm to a zero-shot grasping task. By enabling zero-shot, physically-consistent scene reconstruction and grasping without reliance on extensive datasets or test-time sampling, our approach offers a pathway towards more data efficient, interpretable and generalizable robot autonomy in novel environments.
>
---
#### [new 007] From Bench to Flight: Translating Drone Impact Tests into Operational Safety Limits
- **分类: cs.RO**

- **简介: 该论文属于无人机安全领域，解决如何将碰撞测试数据转化为运行安全限值的问题。通过建立测试系统、数据模型和在线执行工具，实现无人机在室内安全操作。**

- **链接: [https://arxiv.org/pdf/2602.05922v1](https://arxiv.org/pdf/2602.05922v1)**

> **作者:** Aziz Mohamed Mili; Louis Catar; Paul Gérard; Ilyass Tabiai; David St-Onge
>
> **摘要:** Indoor micro-aerial vehicles (MAVs) are increasingly used for tasks that require close proximity to people, yet practitioners lack practical methods to tune motion limits based on measured impact risk. We present an end-to-end, open toolchain that converts benchtop impact tests into deployable safety governors for drones. First, we describe a compact and replicable impact rig and protocol for capturing force-time profiles across drone classes and contact surfaces. Second, we provide data-driven models that map pre-impact speed to impulse and contact duration, enabling direct computation of speed bounds for a target force limit. Third, we release scripts and a ROS2 node that enforce these bounds online and log compliance, with support for facility-specific policies. We validate the workflow on multiple commercial off-the-shelf quadrotors and representative indoor assets, demonstrating that the derived governors preserve task throughput while meeting force constraints specified by safety stakeholders. Our contribution is a practical bridge from measured impacts to runtime limits, with shareable datasets, code, and a repeatable process that teams can adopt to certify indoor MAV operations near humans.
>
---
#### [new 008] Scalable and General Whole-Body Control for Cross-Humanoid Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决跨人体模型的通用控制问题。通过提出XHugWBC框架，实现单一策略在多种机器人上的泛化控制。**

- **链接: [https://arxiv.org/pdf/2602.05791v1](https://arxiv.org/pdf/2602.05791v1)**

> **作者:** Yufei Xue; YunFeng Lin; Wentao Dong; Yang Tang; Jingbo Wang; Jiangmiao Pang; Ming Zhou; Minghuan Liu; Weinan Zhang
>
> **摘要:** Learning-based whole-body controllers have become a key driver for humanoid robots, yet most existing approaches require robot-specific training. In this paper, we study the problem of cross-embodiment humanoid control and show that a single policy can robustly generalize across a wide range of humanoid robot designs with one-time training. We introduce XHugWBC, a novel cross-embodiment training framework that enables generalist humanoid control through: (1) physics-consistent morphological randomization, (2) semantically aligned observation and action spaces across diverse humanoid robots, and (3) effective policy architectures modeling morphological and dynamical properties. XHugWBC is not tied to any specific robot. Instead, it internalizes a broad distribution of morphological and dynamical characteristics during training. By learning motion priors from diverse randomized embodiments, the policy acquires a strong structural bias that supports zero-shot transfer to previously unseen robots. Experiments on twelve simulated humanoids and seven real-world robots demonstrate the strong generalization and robustness of the resulting universal controller.
>
---
#### [new 009] MobileManiBench: Simplifying Model Verification for Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出MobileManiBench，用于验证移动操作机器人的模型架构。解决传统数据集受限于静态场景的问题，通过仿真生成多样化操作轨迹，支持研究数据效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.05233v1](https://arxiv.org/pdf/2602.05233v1)**

> **作者:** Wenbo Wang; Fangyun Wei; QiXiu Li; Xi Chen; Yaobo Liang; Chang Xu; Jiaolong Yang; Baining Guo
>
> **摘要:** Vision-language-action models have advanced robotic manipulation but remain constrained by reliance on the large, teleoperation-collected datasets dominated by the static, tabletop scenes. We propose a simulation-first framework to verify VLA architectures before real-world deployment and introduce MobileManiBench, a large-scale benchmark for mobile-based robotic manipulation. Built on NVIDIA Isaac Sim and powered by reinforcement learning, our pipeline autonomously generates diverse manipulation trajectories with rich annotations (language instructions, multi-view RGB-depth-segmentation images, synchronized object/robot states and actions). MobileManiBench features 2 mobile platforms (parallel-gripper and dexterous-hand robots), 2 synchronized cameras (head and right wrist), 630 objects in 20 categories, 5 skills (open, close, pull, push, pick) with over 100 tasks performed in 100 realistic scenes, yielding 300K trajectories. This design enables controlled, scalable studies of robot embodiments, sensing modalities, and policy architectures, accelerating research on data efficiency and generalization. We benchmark representative VLA models and report insights into perception, reasoning, and control in complex simulated environments.
>
---
#### [new 010] Task-Oriented Robot-Human Handovers on Legged Manipulators
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究任务导向的机器人-人类交接，解决通用化交接难题。通过结合大语言模型与纹理传递，实现零样本泛化交接，提升成功率与用户体验。**

- **链接: [https://arxiv.org/pdf/2602.05760v1](https://arxiv.org/pdf/2602.05760v1)**

> **作者:** Andreea Tulbure; Carmen Scheidemann; Elias Steiner; Marco Hutter
>
> **备注:** Accepted to 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI) 2026
>
> **摘要:** Task-oriented handovers (TOH) are fundamental to effective human-robot collaboration, requiring robots to present objects in a way that supports the human's intended post-handover use. Existing approaches are typically based on object- or task-specific affordances, but their ability to generalize to novel scenarios is limited. To address this gap, we present AFT-Handover, a framework that integrates large language model (LLM)-driven affordance reasoning with efficient texture-based affordance transfer to achieve zero-shot, generalizable TOH. Given a novel object-task pair, the method retrieves a proxy exemplar from a database, establishes part-level correspondences via LLM reasoning, and texturizes affordances for feature-based point cloud transfer. We evaluate AFT-Handover across diverse task-object pairs, showing improved handover success rates and stronger generalization compared to baselines. In a comparative user study, our framework is significantly preferred over the current state-of-the-art, effectively reducing human regrasping before tool use. Finally, we demonstrate TOH on legged manipulators, highlighting the potential of our framework for real-world robot-human handovers.
>
---
#### [new 011] Affordance-Aware Interactive Decision-Making and Execution for Ambiguous Instructions
- **分类: cs.RO**

- **简介: 该论文属于机器人交互任务，旨在解决模糊指令下的任务规划与执行问题。提出AIDE框架，结合视觉语言推理与交互探索，提升机器人理解与执行能力。**

- **链接: [https://arxiv.org/pdf/2602.05273v1](https://arxiv.org/pdf/2602.05273v1)**

> **作者:** Hengxuan Xu; Fengbo Lan; Zhixin Zhao; Shengjie Wang; Mengqiao Liu; Jieqian Sun; Yu Cheng; Tao Zhang
>
> **备注:** 14 pages, 10 figures, 8 tables
>
> **摘要:** Enabling robots to explore and act in unfamiliar environments under ambiguous human instructions by interactively identifying task-relevant objects (e.g., identifying cups or beverages for "I'm thirsty") remains challenging for existing vision-language model (VLM)-based methods. This challenge stems from inefficient reasoning and the lack of environmental interaction, which hinder real-time task planning and execution. To address this, We propose Affordance-Aware Interactive Decision-Making and Execution for Ambiguous Instructions (AIDE), a dual-stream framework that integrates interactive exploration with vision-language reasoning, where Multi-Stage Inference (MSI) serves as the decision-making stream and Accelerated Decision-Making (ADM) as the execution stream, enabling zero-shot affordance analysis and interpretation of ambiguous instructions. Extensive experiments in simulation and real-world environments show that AIDE achieves the task planning success rate of over 80\% and more than 95\% accuracy in closed-loop continuous execution at 10 Hz, outperforming existing VLM-based methods in diverse open-world scenarios.
>
---
#### [new 012] From Vision to Decision: Neuromorphic Control for Autonomous Navigation and Tracking
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主导航任务，解决机器人在目标模糊时的决策问题。提出一种类脑控制框架，通过视觉输入直接生成运动指令，实现高效实时导航。**

- **链接: [https://arxiv.org/pdf/2602.05683v1](https://arxiv.org/pdf/2602.05683v1)**

> **作者:** Chuwei Wang; Eduardo Sebastián; Amanda Prorok; Anastasia Bizyaeva
>
> **摘要:** Robotic navigation has historically struggled to reconcile reactive, sensor-based control with the decisive capabilities of model-based planners. This duality becomes critical when the absence of a predominant option among goals leads to indecision, challenging reactive systems to break symmetries without computationally-intense planners. We propose a parsimonious neuromorphic control framework that bridges this gap for vision-guided navigation and tracking. Image pixels from an onboard camera are encoded as inputs to dynamic neuronal populations that directly transform visual target excitation into egocentric motion commands. A dynamic bifurcation mechanism resolves indecision by delaying commitment until a critical point induced by the environmental geometry. Inspired by recently proposed mechanistic models of animal cognition and opinion dynamics, the neuromorphic controller provides real-time autonomy with a minimal computational burden, a small number of interpretable parameters, and can be seamlessly integrated with application-specific image processing pipelines. We validate our approach in simulation environments as well as on an experimental quadrotor platform.
>
---
#### [new 013] DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出DECO框架，用于双臂灵巧操作任务，解决多模态信息融合与策略优化问题。通过解耦模态条件，结合触觉适配器提升操作精度。**

- **链接: [https://arxiv.org/pdf/2602.05513v1](https://arxiv.org/pdf/2602.05513v1)**

> **作者:** Xukun Li; Yu Sun; Lei Zhang; Bosheng Huang; Yibo Peng; Yuan Meng; Haojun Jiang; Shaoxuan Xie; Guacai Yao; Alois Knoll; Zhenshan Bing; Xinlong Wang; Zhenguo Sun
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Overview of the Proposed DECO Framework.} DECO is a DiT-based policy that decouples multimodal conditioning. Image and action tokens interact via joint self attention, while proprioceptive states and optional conditions are injected through adaptive layer normalization. Tactile signals are injected via cross attention, while a lightweight LoRA-based adapter is used to efficiently fine-tune the pretrained policy. DECO is also accompanied by DECO-50, a bimanual dexterous manipulation dataset with tactile sensing, consisting of 4 scenarios and 28 sub-tasks, covering more than 50 hours of data, approximately 5 million frames, and 8,000 successful trajectories.
>
---
#### [new 014] HiCrowd: Hierarchical Crowd Flow Alignment for Dense Human Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决密集人群中的导航难题。针对机器人冻结问题，提出HiCrowd框架，结合强化学习与模型预测控制，使机器人安全高效地跟随人流。**

- **链接: [https://arxiv.org/pdf/2602.05608v1](https://arxiv.org/pdf/2602.05608v1)**

> **作者:** Yufei Zhu; Shih-Min Yang; Martin Magnusson; Allan Wang
>
> **备注:** Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Navigating through dense human crowds remains a significant challenge for mobile robots. A key issue is the freezing robot problem, where the robot struggles to find safe motions and becomes stuck within the crowd. To address this, we propose HiCrowd, a hierarchical framework that integrates reinforcement learning (RL) with model predictive control (MPC). HiCrowd leverages surrounding pedestrian motion as guidance, enabling the robot to align with compatible crowd flows. A high-level RL policy generates a follow point to align the robot with a suitable pedestrian group, while a low-level MPC safely tracks this guidance with short horizon planning. The method combines long-term crowd aware decision making with safe short-term execution. We evaluate HiCrowd against reactive and learning-based baselines in offline setting (replaying recorded human trajectories) and online setting (human trajectories are updated to react to the robot in simulation). Experiments on a real-world dataset and a synthetic crowd dataset show that our method outperforms in navigation efficiency and safety, while reducing freezing behaviors. Our results suggest that leveraging human motion as guidance, rather than treating humans solely as dynamic obstacles, provides a powerful principle for safe and efficient robot navigation in crowds.
>
---
#### [new 015] Signal or 'Noise': Human Reactions to Robot Errors in the Wild
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，探讨真实场景中人类对机器人错误的社交反应。研究通过实际部署咖啡机器人，分析用户在群体互动中的社会信号，揭示其丰富但“嘈杂”的特性。**

- **链接: [https://arxiv.org/pdf/2602.05010v1](https://arxiv.org/pdf/2602.05010v1)**

> **作者:** Maia Stiber; Sameer Khan; Russell Taylor; Chien-Ming Huang
>
> **摘要:** In the real world, robots frequently make errors, yet little is known about people's social responses to errors outside of lab settings. Prior work has shown that social signals are reliable and useful for error management in constrained interactions, but it is unclear if this holds in the real world - especially with a non-social robot in repeated and group interactions with successive or propagated errors. To explore this, we built a coffee robot and conducted a public field deployment ($N = 49$). We found that participants consistently expressed varied social signals in response to errors and other stimuli, particularly during group interactions. Our findings suggest that social signals in the wild are rich (with participants volunteering information about the interaction), but "noisy." We discuss lessons, benefits, and challenges for using social signals in real-world HRI.
>
---
#### [new 016] Benchmarking Affordance Generalization with BusyBox
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作模型的泛化任务，旨在解决物体操作能力的泛化问题。作者提出BusyBox基准，用于评估模型对新物体的物理交互能力。**

- **链接: [https://arxiv.org/pdf/2602.05441v1](https://arxiv.org/pdf/2602.05441v1)**

> **作者:** Dean Fortier; Timothy Adamson; Tess Hellebrekers; Teresa LaScala; Kofi Ennin; Michael Murray; Andrey Kolobov; Galen Mullins
>
> **摘要:** Vision-Language-Action (VLA) models have been attracting the attention of researchers and practitioners thanks to their promise of generalization. Although single-task policies still offer competitive performance, VLAs are increasingly able to handle commands and environments unseen in their training set. While generalization in vision and language space is undoubtedly important for robust versatile behaviors, a key meta-skill VLAs need to possess is affordance generalization -- the ability to manipulate new objects with familiar physical features. In this work, we present BusyBox, a physical benchmark for systematic semi-automatic evaluation of VLAs' affordance generalization. BusyBox consists of 6 modules with switches, sliders, wires, buttons, a display, and a dial. The modules can be swapped and rotated to create a multitude of BusyBox variations with different visual appearances but the same set of affordances. We empirically demonstrate that generalization across BusyBox variants is highly challenging even for strong open-weights VLAs such as $π_{0.5}$ and GR00T-N1.6. To encourage the research community to evaluate their own VLAs on BusyBox and to propose new affordance generalization experiments, we have designed BusyBox to be easy to build in most robotics labs. We release the full set of CAD files for 3D-printing its parts as well as a bill of materials for (optionally) assembling its electronics. We also publish a dataset of language-annotated demonstrations that we collected using the common bimanual Mobile Aloha robot on the canonical BusyBox configuration. All of the released materials are available at https://microsoft.github.io/BusyBox.
>
---
#### [new 017] Virtual-Tube-Based Cooperative Transport Control for Multi-UAV Systems in Constrained Environments
- **分类: cs.RO**

- **简介: 该论文属于多无人机协同运输任务，解决受限环境中高效、稳定运输的问题。提出基于虚拟管和耗散系统理论的控制框架，实现低开销的协同运输与动态配置。**

- **链接: [https://arxiv.org/pdf/2602.05516v1](https://arxiv.org/pdf/2602.05516v1)**

> **作者:** Runxiao Liu; Pengda Mao; Xiangli Le; Shuang Gu; Yapeng Chen; Quan Quan
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** This paper proposes a novel control framework for cooperative transportation of cable-suspended loads by multiple unmanned aerial vehicles (UAVs) operating in constrained environments. Leveraging virtual tube theory and principles from dissipative systems theory, the framework facilitates efficient multi-UAV collaboration for navigating obstacle-rich areas. The proposed framework offers several key advantages. (1) It achieves tension distribution and coordinated transportation within the UAV-cable-load system with low computational overhead, dynamically adapting UAV configurations based on obstacle layouts to facilitate efficient navigation. (2) By integrating dissipative systems theory, the framework ensures high stability and robustness, essential for complex multi-UAV operations. The effectiveness of the proposed approach is validated through extensive simulations, demonstrating its scalability for large-scale multi-UAV systems. Furthermore, the method is experimentally validated in outdoor scenarios, showcasing its practical feasibility and robustness under real-world conditions.
>
---
#### [new 018] VLN-Pilot: Large Vision-Language Model as an Autonomous Indoor Drone Operator
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VLN-Pilot，利用大视觉语言模型实现室内无人机自主导航，解决传统方法依赖规则和几何规划的问题，通过语义理解和视觉感知实现高效、安全的指令执行。**

- **链接: [https://arxiv.org/pdf/2602.05552v1](https://arxiv.org/pdf/2602.05552v1)**

> **作者:** Bessie Dominguez-Dager; Sergio Suescun-Ferrandiz; Felix Escalona; Francisco Gomez-Donoso; Miguel Cazorla
>
> **摘要:** This paper introduces VLN-Pilot, a novel framework in which a large Vision-and-Language Model (VLLM) assumes the role of a human pilot for indoor drone navigation. By leveraging the multimodal reasoning abilities of VLLMs, VLN-Pilot interprets free-form natural language instructions and grounds them in visual observations to plan and execute drone trajectories in GPS-denied indoor environments. Unlike traditional rule-based or geometric path-planning approaches, our framework integrates language-driven semantic understanding with visual perception, enabling context-aware, high-level flight behaviors with minimal task-specific engineering. VLN-Pilot supports fully autonomous instruction-following for drones by reasoning about spatial relationships, obstacle avoidance, and dynamic reactivity to unforeseen events. We validate our framework on a custom photorealistic indoor simulation benchmark and demonstrate the ability of the VLLM-driven agent to achieve high success rates on complex instruction-following tasks, including long-horizon navigation with multiple semantic targets. Experimental results highlight the promise of replacing remote drone pilots with a language-guided autonomous agent, opening avenues for scalable, human-friendly control of indoor UAVs in tasks such as inspection, search-and-rescue, and facility monitoring. Our results suggest that VLLM-based pilots may dramatically reduce operator workload while improving safety and mission flexibility in constrained indoor environments.
>
---
#### [new 019] A Framework for Combining Optimization-Based and Analytic Inverse Kinematics
- **分类: cs.RO**

- **简介: 该论文属于机器人逆运动学任务，旨在解决优化方法在处理非线性与约束时的低成功率问题。通过引入解析解作为变量变换，提升优化效率与成功率。**

- **链接: [https://arxiv.org/pdf/2602.05092v1](https://arxiv.org/pdf/2602.05092v1)**

> **作者:** Thomas Cohn; Lihan Tang; Alexandre Amice; Russ Tedrake
>
> **备注:** 19 pages, 5 figures, 6 tables. Under submission
>
> **摘要:** Analytic and optimization methods for solving inverse kinematics (IK) problems have been deeply studied throughout the history of robotics. The two strategies have complementary strengths and weaknesses, but developing a unified approach to take advantage of both methods has proved challenging. A key challenge faced by optimization approaches is the complicated nonlinear relationship between the joint angles and the end-effector pose. When this must be handled concurrently with additional nonconvex constraints like collision avoidance, optimization IK algorithms may suffer high failure rates. We present a new formulation for optimization IK that uses an analytic IK solution as a change of variables, and is fundamentally easier for optimizers to solve. We test our methodology on three popular solvers, representing three different paradigms for constrained nonlinear optimization. Extensive experimental comparisons demonstrate that our new formulation achieves higher success rates than the old formulation and baseline methods across various challenging IK problems, including collision avoidance, grasp selection, and humanoid stability.
>
---
#### [new 020] Reinforcement Learning Enhancement Using Vector Semantic Representation and Symbolic Reasoning for Human-Centered Autonomous Emergency Braking
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自动驾驶任务，旨在解决深度强化学习在场景理解与奖励设计上的不足。通过引入语义向量表示和符号推理，提升决策的安全性与上下文感知能力。**

- **链接: [https://arxiv.org/pdf/2602.05079v1](https://arxiv.org/pdf/2602.05079v1)**

> **作者:** Vinal Asodia; Iman Sharifi; Saber Fallah
>
> **备注:** 12 pages, 7 figures, 5 tables
>
> **摘要:** The problem with existing camera-based Deep Reinforcement Learning approaches is twofold: they rarely integrate high-level scene context into the feature representation, and they rely on rigid, fixed reward functions. To address these challenges, this paper proposes a novel pipeline that produces a neuro-symbolic feature representation that encompasses semantic, spatial, and shape information, as well as spatially boosted features of dynamic entities in the scene, with an emphasis on safety-critical road users. It also proposes a Soft First-Order Logic (SFOL) reward function that balances human values via a symbolic reasoning module. Here, semantic and spatial predicates are extracted from segmentation maps and applied to linguistic rules to obtain reward weights. Quantitative experiments in the CARLA simulation environment show that the proposed neuro-symbolic representation and SFOL reward function improved policy robustness and safety-related performance metrics compared to baseline representations and reward formulations across varying traffic densities and occlusion levels. The findings demonstrate that integrating holistic representations and soft reasoning into Reinforcement Learning can support more context-aware and value-aligned decision-making for autonomous driving.
>
---
#### [new 021] RoboPaint: From Human Demonstration to Any Robot and Any View
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决如何从人类演示生成机器人训练数据的问题。通过真实-仿真-真实管道，将人类动作转化为机器人可执行轨迹，提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.05325v1](https://arxiv.org/pdf/2602.05325v1)**

> **作者:** Jiacheng Fan; Zhiyue Zhao; Yiqian Zhang; Chao Chen; Peide Wang; Hengdi Zhang; Zhengxue Cheng
>
> **备注:** 17 pages
>
> **摘要:** Acquiring large-scale, high-fidelity robot demonstration data remains a critical bottleneck for scaling Vision-Language-Action (VLA) models in dexterous manipulation. We propose a Real-Sim-Real data collection and data editing pipeline that transforms human demonstrations into robot-executable, environment-specific training data without direct robot teleoperation. Standardized data collection rooms are built to capture multimodal human demonstrations (synchronized 3 RGB-D videos, 11 RGB videos, 29-DoF glove joint angles, and 14-channel tactile signals). Based on these human demonstrations, we introduce a tactile-aware retargeting method that maps human hand states to robot dex-hand states via geometry and force-guided optimization. Then the retargeted robot trajectories are rendered in a photorealistic Isaac Sim environment to build robot training data. Real world experiments have demonstrated: (1) The retargeted dex-hand trajectories achieve an 84\% success rate across 10 diverse object manipulation tasks. (2) VLA policies (Pi0.5) trained exclusively on our generated data achieve 80\% average success rate on three representative tasks, i.e., pick-and-place, pushing and pouring. To conclude, robot training data can be efficiently "painted" from human demonstrations using our real-sim-real data pipeline. We offer a scalable, cost-effective alternative to teleoperation with minimal performance loss for complex dexterous manipulation.
>
---
#### [new 022] TaSA: Two-Phased Deep Predictive Learning of Tactile Sensory Attenuation for Improving In-Grasp Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，解决机器人在抓取中区分自接触与外接触的问题。提出TaSA框架，分两阶段学习自触觉动态并增强物体接触信号，提升精细操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.05468v1](https://arxiv.org/pdf/2602.05468v1)**

> **作者:** Pranav Ponnivalavan; Satoshi Funabashi; Alexander Schmitz; Tetsuya Ogata; Shigeki Sugano
>
> **备注:** 8 pages, 8 figures, 8 tables, ICRA2026 accepted
>
> **摘要:** Humans can achieve diverse in-hand manipulations, such as object pinching and tool use, which often involve simultaneous contact between the object and multiple fingers. This is still an open issue for robotic hands because such dexterous manipulation requires distinguishing between tactile sensations generated by their self-contact and those arising from external contact. Otherwise, object/robot breakage happens due to contacts/collisions. Indeed, most approaches ignore self-contact altogether, by constraining motion to avoid/ignore self-tactile information during contact. While this reduces complexity, it also limits generalization to real-world scenarios where self-contact is inevitable. Humans overcome this challenge through self-touch perception, using predictive mechanisms that anticipate the tactile consequences of their own motion, through a principle called sensory attenuation, where the nervous system differentiates predictable self-touch signals, allowing novel object stimuli to stand out as relevant. Deriving from this, we introduce TaSA, a two-phased deep predictive learning framework. In the first phase, TaSA explicitly learns self-touch dynamics, modeling how a robot's own actions generate tactile feedback. In the second phase, this learned model is incorporated into the motion learning phase, to emphasize object contact signals during manipulation. We evaluate TaSA on a set of insertion tasks, which demand fine tactile discrimination: inserting a pencil lead into a mechanical pencil, inserting coins into a slot, and fixing a paper clip onto a sheet of paper, with various orientations, positions, and sizes. Across all tasks, policies trained with TaSA achieve significantly higher success rates than baseline methods, demonstrating that structured tactile perception with self-touch based on sensory attenuation is critical for dexterous robotic manipulation.
>
---
#### [new 023] CommCP: Efficient Multi-Agent Coordination via LLM-Based Communication with Conformal Prediction
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文属于多智能体协作任务，解决多机器人在复杂场景中高效信息共享的问题。提出CommCP框架，利用大模型和置信预测提升通信可靠性与任务成功率。**

- **链接: [https://arxiv.org/pdf/2602.06038v1](https://arxiv.org/pdf/2602.06038v1)**

> **作者:** Xiaopan Zhang; Zejin Wang; Zhixu Li; Jianpeng Yao; Jiachen Li
>
> **备注:** IEEE International Conference on Robotics and Automation (ICRA 2026); Project Website: https://comm-cp.github.io/
>
> **摘要:** To complete assignments provided by humans in natural language, robots must interpret commands, generate and answer relevant questions for scene understanding, and manipulate target objects. Real-world deployments often require multiple heterogeneous robots with different manipulation capabilities to handle different assignments cooperatively. Beyond the need for specialized manipulation skills, effective information gathering is important in completing these assignments. To address this component of the problem, we formalize the information-gathering process in a fully cooperative setting as an underexplored multi-agent multi-task Embodied Question Answering (MM-EQA) problem, which is a novel extension of canonical Embodied Question Answering (EQA), where effective communication is crucial for coordinating efforts without redundancy. To address this problem, we propose CommCP, a novel LLM-based decentralized communication framework designed for MM-EQA. Our framework employs conformal prediction to calibrate the generated messages, thereby minimizing receiver distractions and enhancing communication reliability. To evaluate our framework, we introduce an MM-EQA benchmark featuring diverse, photo-realistic household scenarios with embodied questions. Experimental results demonstrate that CommCP significantly enhances the task success rate and exploration efficiency over baselines. The experiment videos, code, and dataset are available on our project website: https://comm-cp.github.io.
>
---
#### [new 024] A Hybrid Autoencoder for Robust Heightmap Generation from Fused Lidar and Depth Data for Humanoid Robot Locomotion
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人地形感知任务，旨在提升人形机器人在复杂环境中的运动能力。通过融合LiDAR与深度数据，提出一种混合自编码器结构，生成更准确的高程图。**

- **链接: [https://arxiv.org/pdf/2602.05855v1](https://arxiv.org/pdf/2602.05855v1)**

> **作者:** Dennis Bank; Joost Cordes; Thomas Seel; Simon F. G. Ehlers
>
> **摘要:** Reliable terrain perception is a critical prerequisite for the deployment of humanoid robots in unstructured, human-centric environments. While traditional systems often rely on manually engineered, single-sensor pipelines, this paper presents a learning-based framework that uses an intermediate, robot-centric heightmap representation. A hybrid Encoder-Decoder Structure (EDS) is introduced, utilizing a Convolutional Neural Network (CNN) for spatial feature extraction fused with a Gated Recurrent Unit (GRU) core for temporal consistency. The architecture integrates multimodal data from an Intel RealSense depth camera, a LIVOX MID-360 LiDAR processed via efficient spherical projection, and an onboard IMU. Quantitative results demonstrate that multimodal fusion improves reconstruction accuracy by 7.2% over depth-only and 9.9% over LiDAR-only configurations. Furthermore, the integration of a 3.2 s temporal context reduces mapping drift.
>
---
#### [new 025] Low-Cost Underwater In-Pipe Centering and Inspection Using a Minimal-Sensing Robot
- **分类: cs.RO**

- **简介: 该论文属于水下管道自主导航与检测任务，解决 confined 环境下的定位与对中问题。通过少传感器实现管道中心对准与巡检。**

- **链接: [https://arxiv.org/pdf/2602.05265v1](https://arxiv.org/pdf/2602.05265v1)**

> **作者:** Kalvik Jakkala; Jason O'Kane
>
> **摘要:** Autonomous underwater inspection of submerged pipelines is challenging due to confined geometries, turbidity, and the scarcity of reliable localization cues. This paper presents a minimal-sensing strategy that enables a free-swimming underwater robot to center itself and traverse a flooded pipe of known radius using only an IMU, a pressure sensor, and two sonars: a downward-facing single-beam sonar and a rotating 360 degree sonar. We introduce a computationally efficient method for extracting range estimates from single-beam sonar intensity data, enabling reliable wall detection in noisy and reverberant conditions. A closed-form geometric model leverages the two sonar ranges to estimate the pipe center, and an adaptive, confidence-weighted proportional-derivative (PD) controller maintains alignment during traversal. The system requires no Doppler velocity log, external tracking, or complex multi-sensor arrays. Experiments in a submerged 46 cm-diameter pipe using a Blue Robotics BlueROV2 heavy remotely operated vehicle demonstrate stable centering and successful full-pipe traversal despite ambient flow and structural deformations. These results show that reliable in-pipe navigation and inspection can be achieved with a lightweight, computationally efficient sensing and processing architecture, advancing the practicality of autonomous underwater inspection in confined environments.
>
---
#### [new 026] Residual Reinforcement Learning for Waste-Container Lifting Using Large-Scale Cranes with Underactuated Tools
- **分类: cs.RO**

- **简介: 论文研究城市环境中垃圾箱吊装任务，解决起重机与欠驱动装置间的精确轨迹跟踪和摆动抑制问题。提出残差强化学习方法，结合常规控制器与学习策略，提升吊装精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.05895v1](https://arxiv.org/pdf/2602.05895v1)**

> **作者:** Qi Li; Karsten Berns
>
> **备注:** 12 pages
>
> **摘要:** This paper studies the container lifting phase of a waste-container recycling task in urban environments, performed by a hydraulic loader crane equipped with an underactuated discharge unit, and proposes a residual reinforcement learning (RRL) approach that combines a nominal Cartesian controller with a learned residual policy. All experiments are conducted in simulation, where the task is characterized by tight geometric tolerances between the discharge-unit hooks and the container rings relative to the overall crane scale, making precise trajectory tracking and swing suppression essential. The nominal controller uses admittance control for trajectory tracking and pendulum-aware swing damping, followed by damped least-squares inverse kinematics with a nullspace posture term to generate joint velocity commands. A PPO-trained residual policy in Isaac Lab compensates for unmodeled dynamics and parameter variations, improving precision and robustness without requiring end-to-end learning from scratch. We further employ randomized episode initialization and domain randomization over payload properties, actuator gains, and passive joint parameters to enhance generalization. Simulation results demonstrate improved tracking accuracy, reduced oscillations, and higher lifting success rates compared to the nominal controller alone.
>
---
#### [new 027] Learning Soccer Skills for Humanoid Robots: A Progressive Perception-Action Framework
- **分类: cs.RO**

- **简介: 该论文属于机器人足球任务，旨在解决人形机器人感知与动作集成难题。提出PAiD框架，分阶段提升踢球和平衡能力，增强泛化与真实环境适应性。**

- **链接: [https://arxiv.org/pdf/2602.05310v1](https://arxiv.org/pdf/2602.05310v1)**

> **作者:** Jipeng Kong; Xinzhe Liu; Yuhang Lin; Jinrui Han; Sören Schwertfeger; Chenjia Bai; Xuelong Li
>
> **备注:** 13 pages, 9 figures, conference
>
> **摘要:** Soccer presents a significant challenge for humanoid robots, demanding tightly integrated perception-action capabilities for tasks like perception-guided kicking and whole-body balance control. Existing approaches suffer from inter-module instability in modular pipelines or conflicting training objectives in end-to-end frameworks. We propose Perception-Action integrated Decision-making (PAiD), a progressive architecture that decomposes soccer skill acquisition into three stages: motion-skill acquisition via human motion tracking, lightweight perception-action integration for positional generalization, and physics-aware sim-to-real transfer. This staged decomposition establishes stable foundational skills, avoids reward conflicts during perception integration, and minimizes sim-to-real gaps. Experiments on the Unitree G1 demonstrate high-fidelity human-like kicking with robust performance under diverse conditions-including static or rolling balls, various positions, and disturbances-while maintaining consistent execution across indoor and outdoor scenarios. Our divide-and-conquer strategy advances robust humanoid soccer capabilities and offers a scalable framework for complex embodied skill acquisition. The project page is available at https://soccer-humanoid.github.io/.
>
---
#### [new 028] IndustryShapes: An RGB-D Benchmark dataset for 6D object pose estimation of industrial assembly components and tools
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出IndustryShapes数据集，用于工业装配件和工具的6D位姿估计任务，解决真实工业场景下姿态估计的挑战。**

- **链接: [https://arxiv.org/pdf/2602.05555v1](https://arxiv.org/pdf/2602.05555v1)**

> **作者:** Panagiotis Sapoutzoglou; Orestis Vaggelis; Athina Zacharia; Evangelos Sartinas; Maria Pateraki
>
> **备注:** To appear in ICRA 2026
>
> **摘要:** We introduce IndustryShapes, a new RGB-D benchmark dataset of industrial tools and components, designed for both instance-level and novel object 6D pose estimation approaches. The dataset provides a realistic and application-relevant testbed for benchmarking these methods in the context of industrial robotics bridging the gap between lab-based research and deployment in real-world manufacturing scenarios. Unlike many previous datasets that focus on household or consumer products or use synthetic, clean tabletop datasets, or objects captured solely in controlled lab environments, IndustryShapes introduces five new object types with challenging properties, also captured in realistic industrial assembly settings. The dataset has diverse complexity, from simple to more challenging scenes, with single and multiple objects, including scenes with multiple instances of the same object and it is organized in two parts: the classic set and the extended set. The classic set includes a total of 4,6k images and 6k annotated poses. The extended set introduces additional data modalities to support the evaluation of model-free and sequence-based approaches. To the best of our knowledge, IndustryShapes is the first dataset to offer RGB-D static onboarding sequences. We further evaluate the dataset on a representative set of state-of-the art methods for instance-based and novel object 6D pose estimation, including also object detection, segmentation, showing that there is room for improvement in this domain. The dataset page can be found in https://pose-lab.github.io/IndustryShapes.
>
---
#### [new 029] Applying Ground Robot Fleets in Urban Search: Understanding Professionals' Operational Challenges and Design Opportunities
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于公共安全领域，研究如何通过地面机器人舰队支持城市搜索任务。解决专业人员在高压力环境下的操作挑战，通过技术手段减轻认知和体力负担。**

- **链接: [https://arxiv.org/pdf/2602.04992v1](https://arxiv.org/pdf/2602.04992v1)**

> **作者:** Puqi Zhou; Charles R. Twardy; Cynthia Lum; Myeong Lee; David J. Porfirio; Michael R. Hieb; Chris Thomas; Xuesu Xiao; Sungsoo Ray Hong
>
> **备注:** Under review
>
> **摘要:** Urban searches demand rapid, defensible decisions and sustained physical effort under high cognitive and situational load. Incident commanders must plan, coordinate, and document time-critical operations, while field searchers execute evolving tasks in uncertain environments. With recent advances in technology, ground-robot fleets paired with computer-vision-based situational awareness and LLM-powered interfaces offer the potential to ease these operational burdens. However, no dedicated studies have examined how public safety professionals perceive such technologies or envision their integration into existing practices, risking building technically sophisticated yet impractical solutions. To address this gap, we conducted focus-group sessions with eight police officers across five local departments in Virginia. Our findings show that ground robots could reduce professionals' reliance on paper references, mental calculations, and ad-hoc coordination, alleviating cognitive and physical strain in four key challenge areas: (1) partitioning the workforce across multiple search hypotheses, (2) retaining group awareness and situational awareness, (3) building route planning that fits the lost-person profile, and (4) managing cognitive and physical fatigue under uncertainty. We further identify four design opportunities and requirements for future ground-robot fleet integration in public-safety operations: (1) scalable multi-robot planning and control interfaces, (2) agency-specific route optimization, (3) real-time replanning informed by debrief updates, and (4) vision-assisted cueing that preserves operational trust while reducing cognitive workload. We conclude with design implications for deployable, accountable, and human-centered urban-search support systems
>
---
#### [new 030] InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决物理交互中全身运动的生成问题。通过构建可扩展的生成控制器，提升机器人在多样场景下的运动泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06035v1](https://arxiv.org/pdf/2602.06035v1)**

> **作者:** Sirui Xu; Samuel Schulter; Morteza Ziyadi; Xialin He; Xiaohan Fei; Yu-Xiong Wang; Liangyan Gui
>
> **备注:** Webpage: https://sirui-xu.github.io/InterPrior/
>
> **摘要:** Humans rarely plan whole-body interactions with objects at the level of explicit whole-body movements. High-level intentions, such as affordance, define the goal, while coordinated balance, contact, and manipulation can emerge naturally from underlying physical and motor priors. Scaling such priors is key to enabling humanoids to compose and generalize loco-manipulation skills across diverse contexts while maintaining physically coherent whole-body coordination. To this end, we introduce InterPrior, a scalable framework that learns a unified generative controller through large-scale imitation pretraining and post-training by reinforcement learning. InterPrior first distills a full-reference imitation expert into a versatile, goal-conditioned variational policy that reconstructs motion from multimodal observations and high-level intent. While the distilled policy reconstructs training behaviors, it does not generalize reliably due to the vast configuration space of large-scale human-object interactions. To address this, we apply data augmentation with physical perturbations, and then perform reinforcement learning finetuning to improve competence on unseen goals and initializations. Together, these steps consolidate the reconstructed latent skills into a valid manifold, yielding a motion prior that generalizes beyond the training data, e.g., it can incorporate new behaviors such as interactions with unseen objects. We further demonstrate its effectiveness for user-interactive control and its potential for real robot deployment.
>
---
#### [new 031] MerNav: A Highly Generalizable Memory-Execute-Review Framework for Zero-Shot Object Goal Navigation
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在提升零样本目标导航的成功率和泛化能力。提出Memor-Execute-Review框架，在多个数据集上取得了显著提升。**

- **链接: [https://arxiv.org/pdf/2602.05467v1](https://arxiv.org/pdf/2602.05467v1)**

> **作者:** Dekang Qi; Shuang Zeng; Xinyuan Chang; Feng Xiong; Shichao Xie; Xiaolong Wu; Mu Xu
>
> **备注:** 9 pages, 2 figures, 5 tables, conference
>
> **摘要:** Visual Language Navigation (VLN) is one of the fundamental capabilities for embodied intelligence and a critical challenge that urgently needs to be addressed. However, existing methods are still unsatisfactory in terms of both success rate (SR) and generalization: Supervised Fine-Tuning (SFT) approaches typically achieve higher SR, while Training-Free (TF) approaches often generalize better, but it is difficult to obtain both simultaneously. To this end, we propose a Memory-Execute-Review framework. It consists of three parts: a hierarchical memory module for providing information support, an execute module for routine decision-making and actions, and a review module for handling abnormal situations and correcting behavior. We validated the effectiveness of this framework on the Object Goal Navigation task. Across 4 datasets, our average SR achieved absolute improvements of 7% and 5% compared to all baseline methods under TF and Zero-Shot (ZS) settings, respectively. On the most commonly used HM3D_v0.1 and the more challenging open vocabulary dataset HM3D_OVON, the SR improved by 8% and 6%, under ZS settings. Furthermore, on the MP3D and HM3D_OVON datasets, our method not only outperformed all TF methods but also surpassed all SFT methods, achieving comprehensive leadership in both SR (5% and 2%) and generalization.
>
---
#### [new 032] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning
- **分类: cs.CR; cs.LG; cs.RO**

- **简介: 该论文属于强化学习安全任务，解决模拟器被恶意篡改导致的后门攻击问题。工作包括提出新攻击方法Daze，实现无需修改奖励的隐蔽后门植入。**

- **链接: [https://arxiv.org/pdf/2602.05089v1](https://arxiv.org/pdf/2602.05089v1)**

> **作者:** Ethan Rathbun; Wo Wei Lin; Alina Oprea; Christopher Amato
>
> **备注:** 10 pages main body, ICLR 2026
>
> **摘要:** Simulated environments are a key piece in the success of Reinforcement Learning (RL), allowing practitioners and researchers to train decision making agents without running expensive experiments on real hardware. Simulators remain a security blind spot, however, enabling adversarial developers to alter the dynamics of their released simulators for malicious purposes. Therefore, in this work we highlight a novel threat, demonstrating how simulator dynamics can be exploited to stealthily implant action-level backdoors into RL agents. The backdoor then allows an adversary to reliably activate targeted actions in an agent upon observing a predefined ``trigger'', leading to potentially dangerous consequences. Traditional backdoor attacks are limited in their strong threat models, assuming the adversary has near full control over an agent's training pipeline, enabling them to both alter and observe agent's rewards. As these assumptions are infeasible to implement within a simulator, we propose a new attack ``Daze'' which is able to reliably and stealthily implant backdoors into RL agents trained for real world tasks without altering or even observing their rewards. We provide formal proof of Daze's effectiveness in guaranteeing attack success across general RL tasks along with extensive empirical evaluations on both discrete and continuous action space domains. We additionally provide the first example of RL backdoor attacks transferring to real, robotic hardware. These developments motivate further research into securing all components of the RL training pipeline to prevent malicious attacks.
>
---
#### [new 033] A Data Driven Structural Decomposition of Dynamic Games via Best Response Maps
- **分类: cs.GT; cs.MA; cs.RO; eess.SY; math.OC**

- **简介: 该论文属于动态博弈求解任务，旨在解决多智能体决策中的均衡计算难题。通过引入数据驱动的结构分解方法，减少优化耦合，提升求解效率与一致性。**

- **链接: [https://arxiv.org/pdf/2602.05324v1](https://arxiv.org/pdf/2602.05324v1)**

> **作者:** Mahdis Rabbani; Navid Mojahed; Shima Nazari
>
> **备注:** 11 pages, 6 figures, 5 tables, Submitted to RSS 2026
>
> **摘要:** Dynamic games are powerful tools to model multi-agent decision-making, yet computing Nash (generalized Nash) equilibria remains a central challenge in such settings. Complexity arises from tightly coupled optimality conditions, nested optimization structures, and poor numerical conditioning. Existing game-theoretic solvers address these challenges by directly solving the joint game, typically requiring explicit modeling of all agents' objective functions and constraints, while learning-based approaches often decouple interaction through prediction or policy approximation, sacrificing equilibrium consistency. This paper introduces a conceptually novel formulation for dynamic games by restructuring the equilibrium computation. Rather than solving a fully coupled game or decoupling agents through prediction or policy approximation, a data-driven structural reduction of the game is proposed that removes nested optimization layers and derivative coupling by embedding an offline-compiled best-response map as a feasibility constraint. Under standard regularity conditions, when the best-response operator is exact, any converged solution of the reduced problem corresponds to a local open-loop Nash (GNE) equilibrium of the original game; with a learned surrogate, the solution is approximately equilibrium-consistent up to the best-response approximation error. The proposed formulation is supported by mathematical proofs, accompanying a large-scale Monte Carlo study in a two-player open-loop dynamic game motivated by the autonomous racing problem. Comparisons are made against state-of-the-art joint game solvers, and results are reported on solution quality, computational cost, and constraint satisfaction.
>
---
#### [new 034] Learning Event-Based Shooter Models from Virtual Reality Experiments
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于模拟与安全干预任务，旨在解决VR实验中难以大规模评估干预策略的问题。通过构建基于数据的离散事件模拟器，模拟枪手行为，实现高效策略学习与评估。**

- **链接: [https://arxiv.org/pdf/2602.06023v1](https://arxiv.org/pdf/2602.06023v1)**

> **作者:** Christopher A. McClurg; Alan R. Wagner
>
> **备注:** Preprint under review for conference publication. 9 pages, 4 figures, 4 tables
>
> **摘要:** Virtual reality (VR) has emerged as a powerful tool for evaluating school security measures in high-risk scenarios such as school shootings, offering experimental control and high behavioral fidelity. However, assessing new interventions in VR requires recruiting new participant cohorts for each condition, making large-scale or iterative evaluation difficult. These limitations are especially restrictive when attempting to learn effective intervention strategies, which typically require many training episodes. To address this challenge, we develop a data-driven discrete-event simulator (DES) that models shooter movement and in-region actions as stochastic processes learned from participant behavior in VR studies. We use the simulator to examine the impact of a robot-based shooter intervention strategy. Once shown to reproduce key empirical patterns, the DES enables scalable evaluation and learning of intervention strategies that are infeasible to train directly with human subjects. Overall, this work demonstrates a high-to-mid fidelity simulation workflow that provides a scalable surrogate for developing and evaluating autonomous school-security interventions.
>
---
#### [new 035] Optimizing Mission Planning for Multi-Debris Rendezvous Using Reinforcement Learning with Refueling and Adaptive Collision Avoidance
- **分类: cs.AI; cs.LG; cs.RO; physics.space-ph**

- **简介: 该论文属于多目标交会任务，旨在解决空间碎片清除中的碰撞避让与任务优化问题。通过强化学习框架，实现高效、安全的多碎片清除任务规划。**

- **链接: [https://arxiv.org/pdf/2602.05075v1](https://arxiv.org/pdf/2602.05075v1)**

> **作者:** Agni Bandyopadhyay; Gunther Waxenegger-Wilfing
>
> **备注:** Accpeted at Conference: 15th IAA Symposium on Small Satellites for Earth System Observation At: Berlin
>
> **摘要:** As the orbital environment around Earth becomes increasingly crowded with debris, active debris removal (ADR) missions face significant challenges in ensuring safe operations while minimizing the risk of in-orbit collisions. This study presents a reinforcement learning (RL) based framework to enhance adaptive collision avoidance in ADR missions, specifically for multi-debris removal using small satellites. Small satellites are increasingly adopted due to their flexibility, cost effectiveness, and maneuverability, making them well suited for dynamic missions such as ADR. Building on existing work in multi-debris rendezvous, the framework integrates refueling strategies, efficient mission planning, and adaptive collision avoidance to optimize spacecraft rendezvous operations. The proposed approach employs a masked Proximal Policy Optimization (PPO) algorithm, enabling the RL agent to dynamically adjust maneuvers in response to real-time orbital conditions. Key considerations include fuel efficiency, avoidance of active collision zones, and optimization of dynamic orbital parameters. The RL agent learns to determine efficient sequences for rendezvousing with multiple debris targets, optimizing fuel usage and mission time while incorporating necessary refueling stops. Simulated ADR scenarios derived from the Iridium 33 debris dataset are used for evaluation, covering diverse orbital configurations and debris distributions to demonstrate robustness and adaptability. Results show that the proposed RL framework reduces collision risk while improving mission efficiency compared to traditional heuristic approaches. This work provides a scalable solution for planning complex multi-debris ADR missions and is applicable to other multi-target rendezvous problems in autonomous space mission planning.
>
---
#### [new 036] Trojan Attacks on Neural Network Controllers for Robotic Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于安全任务，研究神经网络控制器在机器人系统中的后门攻击问题。工作是设计一种轻量级Trojan网络，可在特定条件下破坏机器人控制行为。**

- **链接: [https://arxiv.org/pdf/2602.05121v1](https://arxiv.org/pdf/2602.05121v1)**

> **作者:** Farbod Younesi; Walter Lucia; Amr Youssef
>
> **备注:** Paper submitted to the 2026 IEEE Conference on Control Technology and Applications (CCTA)
>
> **摘要:** Neural network controllers are increasingly deployed in robotic systems for tasks such as trajectory tracking and pose stabilization. However, their reliance on potentially untrusted training pipelines or supply chains introduces significant security vulnerabilities. This paper investigates backdoor (Trojan) attacks against neural controllers, using a differential-drive mobile robot platform as a case study. In particular, assuming that the robot's tracking controller is implemented as a neural network, we design a lightweight, parallel Trojan network that can be embedded within the controller. This malicious module remains dormant during normal operation but, upon detecting a highly specific trigger condition defined by the robot's pose and goal parameters, compromises the primary controller's wheel velocity commands, resulting in undesired and potentially unsafe robot behaviours. We provide a proof-of-concept implementation of the proposed Trojan network, which is validated through simulation under two different attack scenarios. The results confirm the effectiveness of the proposed attack and demonstrate that neural network-based robotic control systems are subject to potentially critical security threats.
>
---
#### [new 037] GAMMS: Graph based Adversarial Multiagent Modeling Simulator
- **分类: cs.AI; cs.RO; cs.SE**

- **简介: 该论文提出GAMMS，一个基于图的多智能体模拟框架，解决传统模拟工具计算成本高、难以扩展的问题。旨在支持快速开发与评估多智能体系统。**

- **链接: [https://arxiv.org/pdf/2602.05105v1](https://arxiv.org/pdf/2602.05105v1)**

> **作者:** Rohan Patil; Jai Malegaonkar; Xiao Jiang; Andre Dion; Gaurav S. Sukhatme; Henrik I. Christensen
>
> **摘要:** As intelligent systems and multi-agent coordination become increasingly central to real-world applications, there is a growing need for simulation tools that are both scalable and accessible. Existing high-fidelity simulators, while powerful, are often computationally expensive and ill-suited for rapid prototyping or large-scale agent deployments. We present GAMMS (Graph based Adversarial Multiagent Modeling Simulator), a lightweight yet extensible simulation framework designed to support fast development and evaluation of agent behavior in environments that can be represented as graphs. GAMMS emphasizes five core objectives: scalability, ease of use, integration-first architecture, fast visualization feedback, and real-world grounding. It enables efficient simulation of complex domains such as urban road networks and communication systems, supports integration with external tools (e.g., machine learning libraries, planning solvers), and provides built-in visualization with minimal configuration. GAMMS is agnostic to policy type, supporting heuristic, optimization-based, and learning-based agents, including those using large language models. By lowering the barrier to entry for researchers and enabling high-performance simulations on standard hardware, GAMMS facilitates experimentation and innovation in multi-agent systems, autonomous planning, and adversarial modeling. The framework is open-source and available at https://github.com/GAMMSim/GAMMS/
>
---
#### [new 038] Modelling Pedestrian Behaviour in Autonomous Vehicle Encounters Using Naturalistic Dataset
- **分类: physics.soc-ph; cs.RO**

- **简介: 该论文属于交通行为分析任务，旨在研究行人与自动驾驶汽车互动时的行为模式。通过分析NuScenes数据集，建立混合模型以理解行人的移动调整因素。**

- **链接: [https://arxiv.org/pdf/2602.05142v1](https://arxiv.org/pdf/2602.05142v1)**

> **作者:** Rulla Al-Haideri; Bilal Farooq
>
> **摘要:** Understanding how pedestrians adjust their movement when interacting with autonomous vehicles (AVs) is essential for improving safety in mixed traffic. This study examines micro-level pedestrian behaviour during midblock encounters in the NuScenes dataset using a hybrid discrete choice-machine learning framework based on the Residual Logit (ResLogit) model. The model incorporates temporal, spatial, kinematic, and perceptual indicators. These include relative speed, visual looming, remaining distance, and directional collision risk proximity (CRP) measures. Results suggest that some of these variables may meaningfully influence movement adjustments, although predictive performance remains moderate. Marginal effects and elasticities indicate strong directional asymmetries in risk perception, with frontal and rear CRP showing opposite influences. The remaining distance exhibits a possible mid-crossing threshold. Relative speed cues appear to have a comparatively less effect. These patterns may reflect multiple behavioural tendencies driven by both risk perception and movement efficiency.
>
---
#### [new 039] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于离线强化学习任务，旨在解决OOD误差和多模态策略表示问题。提出ReFORM方法，通过流政策和噪声调控提升策略性能。**

- **链接: [https://arxiv.org/pdf/2602.05051v1](https://arxiv.org/pdf/2602.05051v1)**

> **作者:** Songyuan Zhang; Oswin So; H. M. Sabbir Ahmad; Eric Yang Yu; Matthew Cleaveland; Mitchell Black; Chuchu Fan
>
> **备注:** 24 pages, 17 figures; Accepted by the fourteenth International Conference on Learning Representations (ICLR 2026)
>
> **摘要:** Offline reinforcement learning (RL) aims to learn the optimal policy from a fixed dataset generated by behavior policies without additional environment interactions. One common challenge that arises in this setting is the out-of-distribution (OOD) error, which occurs when the policy leaves the training distribution. Prior methods penalize a statistical distance term to keep the policy close to the behavior policy, but this constrains policy improvement and may not completely prevent OOD actions. Another challenge is that the optimal policy distribution can be multimodal and difficult to represent. Recent works apply diffusion or flow policies to address this problem, but it is unclear how to avoid OOD errors while retaining policy expressiveness. We propose ReFORM, an offline RL method based on flow policies that enforces the less restrictive support constraint by construction. ReFORM learns a behavior cloning (BC) flow policy with a bounded source distribution to capture the support of the action distribution, then optimizes a reflected flow that generates bounded noise for the BC flow while keeping the support, to maximize the performance. Across 40 challenging tasks from the OGBench benchmark with datasets of varying quality and using a constant set of hyperparameters for all tasks, ReFORM dominates all baselines with hand-tuned hyperparameters on the performance profile curves.
>
---
#### [new 040] VISTA: Enhancing Visual Conditioning via Track-Following Preference Optimization in Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型研究，解决视觉与动作对齐问题。通过优化跟踪任务增强视觉条件依赖，提升动作预测准确性。**

- **链接: [https://arxiv.org/pdf/2602.05049v1](https://arxiv.org/pdf/2602.05049v1)**

> **作者:** Yiye Chen; Yanan Jian; Xiaoyi Dong; Shuxin Cao; Jing Wu; Patricio Vela; Benjamin E. Lundell; Dongdong Chen
>
> **备注:** In submission. Project website: https://vista-vla.github.io/
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated strong performance across a wide range of robotic manipulation tasks. Despite the success, extending large pretrained Vision-Language Models (VLMs) to the action space can induce vision-action misalignment, where action predictions exhibit weak dependence on the current visual state, leading to unreliable action outputs. In this work, we study VLA models through the lens of visual conditioning and empirically show that successful rollouts consistently exhibit stronger visual dependence than failed ones. Motivated by this observation, we propose a training framework that explicitly strengthens visual conditioning in VLA models. Our approach first aligns action prediction with visual input via preference optimization on a track-following surrogate task, and then transfers the enhanced alignment to instruction-following task through latent-space distillation during supervised finetuning. Without introducing architectural modifications or additional data collection, our method improves both visual conditioning and task performance for discrete OpenVLA, and further yields consistent gains when extended to the continuous OpenVLA-OFT setting. Project website: https://vista-vla.github.io/ .
>
---
#### [new 041] Constrained Group Relative Policy Optimization
- **分类: cs.LG; cs.CL; cs.RO**

- **简介: 该论文提出Constrained GRPO，解决有约束的策略优化问题。通过拉格朗日方法处理约束，改进优势函数以稳定约束控制，提升机器人任务中的约束满足与成功率。**

- **链接: [https://arxiv.org/pdf/2602.05863v1](https://arxiv.org/pdf/2602.05863v1)**

> **作者:** Roger Girgis; Rodrigue de Schaetzen; Luke Rowe; Azalée Robitaille; Christopher Pal; Liam Paull
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** While Group Relative Policy Optimization (GRPO) has emerged as a scalable framework for critic-free policy learning, extending it to settings with explicit behavioral constraints remains underexplored. We introduce Constrained GRPO, a Lagrangian-based extension of GRPO for constrained policy optimization. Constraints are specified via indicator cost functions, enabling direct optimization of violation rates through a Lagrangian relaxation. We show that a naive multi-component treatment in advantage estimation can break constrained learning: mismatched component-wise standard deviations distort the relative importance of the different objective terms, which in turn corrupts the Lagrangian signal and prevents meaningful constraint enforcement. We formally derive this effect to motivate our scalarized advantage construction that preserves the intended trade-off between reward and constraint terms. Experiments in a toy gridworld confirm the predicted optimization pathology and demonstrate that scalarizing advantages restores stable constraint control. In addition, we evaluate Constrained GRPO on robotics tasks, where it improves constraint satisfaction while increasing task success, establishing a simple and effective recipe for constrained policy optimization in embodied AI domains that increasingly rely on large multimodal foundation models.
>
---
#### [new 042] Evaluating Robustness and Adaptability in Learning-Based Mission Planning for Active Debris Removal
- **分类: cs.AI; cs.LG; cs.RO; physics.space-ph**

- **简介: 该论文属于主动碎片清除任务，解决多碎片任务规划中的鲁棒性与适应性问题。比较了三种规划方法在不同约束下的表现。**

- **链接: [https://arxiv.org/pdf/2602.05091v1](https://arxiv.org/pdf/2602.05091v1)**

> **作者:** Agni Bandyopadhyay; Günther Waxenegger-Wilfing
>
> **备注:** Presented at Conference: International Conference on Space Robotics (ISPARO,2025) At: Sendai,Japan
>
> **摘要:** Autonomous mission planning for Active Debris Removal (ADR) must balance efficiency, adaptability, and strict feasibility constraints on fuel and mission duration. This work compares three planners for the constrained multi-debris rendezvous problem in Low Earth Orbit: a nominal Masked Proximal Policy Optimization (PPO) policy trained under fixed mission parameters, a domain-randomized Masked PPO policy trained across varying mission constraints for improved robustness, and a plain Monte Carlo Tree Search (MCTS) baseline. Evaluations are conducted in a high-fidelity orbital simulation with refueling, realistic transfer dynamics, and randomized debris fields across 300 test cases in nominal, reduced fuel, and reduced mission time scenarios. Results show that nominal PPO achieves top performance when conditions match training but degrades sharply under distributional shift, while domain-randomized PPO exhibits improved adaptability with only moderate loss in nominal performance. MCTS consistently handles constraint changes best due to online replanning but incurs orders-of-magnitude higher computation time. The findings underline a trade-off between the speed of learned policies and the adaptability of search-based methods, and suggest that combining training-time diversity with online planning could be a promising path for future resilient ADR mission planners.
>
---
#### [new 043] Formal Synthesis of Certifiably Robust Neural Lyapunov-Barrier Certificates
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于安全强化学习任务，旨在解决动态扰动下的控制器验证问题。通过设计鲁棒神经Lyapunov-Barrier证书，提升系统在不确定环境中的安全性与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.05311v1](https://arxiv.org/pdf/2602.05311v1)**

> **作者:** Chengxiao Wang; Haoze Wu; Gagandeep Singh
>
> **摘要:** Neural Lyapunov and barrier certificates have recently been used as powerful tools for verifying the safety and stability properties of deep reinforcement learning (RL) controllers. However, existing methods offer guarantees only under fixed ideal unperturbed dynamics, limiting their reliability in real-world applications where dynamics may deviate due to uncertainties. In this work, we study the problem of synthesizing \emph{robust neural Lyapunov barrier certificates} that maintain their guarantees under perturbations in system dynamics. We formally define a robust Lyapunov barrier function and specify sufficient conditions based on Lipschitz continuity that ensure robustness against bounded perturbations. We propose practical training objectives that enforce these conditions via adversarial training, Lipschitz neighborhood bound, and global Lipschitz regularization. We validate our approach in two practically relevant environments, Inverted Pendulum and 2D Docking. The former is a widely studied benchmark, while the latter is a safety-critical task in autonomous systems. We show that our methods significantly improve both certified robustness bounds (up to $4.6$ times) and empirical success rates under strong perturbations (up to $2.4$ times) compared to the baseline. Our results demonstrate effectiveness of training robust neural certificates for safe RL under perturbations in dynamics.
>
---
#### [new 044] PIRATR: Parametric Object Inference for Robotic Applications with Transformers in 3D Point Clouds
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出PIRATR，用于机器人3D点云中的对象检测任务。解决传统方法在复杂环境下的定位与属性估计问题，通过联合预测姿态和参数化属性，提升检测精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.05557v1](https://arxiv.org/pdf/2602.05557v1)**

> **作者:** Michael Schwingshackl; Fabio F. Oberweger; Mario Niedermeyer; Huemer Johannes; Markus Murschitz
>
> **备注:** 8 Pages, 11 Figures, Accepted at 2026 IEEE International Conference on Robotics & Automation (ICRA) Vienna
>
> **摘要:** We present PIRATR, an end-to-end 3D object detection framework for robotic use cases in point clouds. Extending PI3DETR, our method streamlines parametric 3D object detection by jointly estimating multi-class 6-DoF poses and class-specific parametric attributes directly from occlusion-affected point cloud data. This formulation enables not only geometric localization but also the estimation of task-relevant properties for parametric objects, such as a gripper's opening, where the 3D model is adjusted according to simple, predefined rules. The architecture employs modular, class-specific heads, making it straightforward to extend to novel object types without re-designing the pipeline. We validate PIRATR on an automated forklift platform, focusing on three structurally and functionally diverse categories: crane grippers, loading platforms, and pallets. Trained entirely in a synthetic environment, PIRATR generalizes effectively to real outdoor LiDAR scans, achieving a detection mAP of 0.919 without additional fine-tuning. PIRATR establishes a new paradigm of pose-aware, parameterized perception. This bridges the gap between low-level geometric reasoning and actionable world models, paving the way for scalable, simulation-trained perception systems that can be deployed in dynamic robotic environments. Code available at https://github.com/swingaxe/piratr.
>
---
#### [new 045] RFM-Pose:Reinforcement-Guided Flow Matching for Fast Category-Level 6D Pose Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于6D物体位姿估计任务，解决类别级位姿估计中采样效率低的问题。提出RFM-Pose框架，结合流匹配与强化学习，提升生成效率并优化候选位姿评估。**

- **链接: [https://arxiv.org/pdf/2602.05257v1](https://arxiv.org/pdf/2602.05257v1)**

> **作者:** Diya He; Qingchen Liu; Cong Zhang; Jiahu Qin
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Object pose estimation is a fundamental problem in computer vision and plays a critical role in virtual reality and embodied intelligence, where agents must understand and interact with objects in 3D space. Recently, score based generative models have to some extent solved the rotational symmetry ambiguity problem in category level pose estimation, but their efficiency remains limited by the high sampling cost of score-based diffusion. In this work, we propose a new framework, RFM-Pose, that accelerates category-level 6D object pose generation while actively evaluating sampled hypotheses. To improve sampling efficiency, we adopt a flow-matching generative model and generate pose candidates along an optimal transport path from a simple prior to the pose distribution. To further refine these candidates, we cast the flow-matching sampling process as a Markov decision process and apply proximal policy optimization to fine-tune the sampling policy. In particular, we interpret the flow field as a learnable policy and map an estimator to a value network, enabling joint optimization of pose generation and hypothesis scoring within a reinforcement learning framework. Experiments on the REAL275 benchmark demonstrate that RFM-Pose achieves favorable performance while significantly reducing computational cost. Moreover, similar to prior work, our approach can be readily adapted to object pose tracking and attains competitive results in this setting.
>
---
#### [new 046] Sparse Video Generation Propels Real-World Beyond-the-View Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言导航任务，旨在解决远距离未知目标导航问题。提出SparseVideoNav方法，利用视频生成模型实现高效导航，提升成功率与速度。**

- **链接: [https://arxiv.org/pdf/2602.05827v1](https://arxiv.org/pdf/2602.05827v1)**

> **作者:** Hai Zhang; Siqi Liang; Li Chen; Yuxian Li; Yukuan Xu; Yichao Zhong; Fu Zhang; Hongyang Li
>
> **摘要:** Why must vision-language navigation be bound to detailed and verbose language instructions? While such details ease decision-making, they fundamentally contradict the goal for navigation in the real-world. Ideally, agents should possess the autonomy to navigate in unknown environments guided solely by simple and high-level intents. Realizing this ambition introduces a formidable challenge: Beyond-the-View Navigation (BVN), where agents must locate distant, unseen targets without dense and step-by-step guidance. Existing large language model (LLM)-based methods, though adept at following dense instructions, often suffer from short-sighted behaviors due to their reliance on short-horimzon supervision. Simply extending the supervision horizon, however, destabilizes LLM training. In this work, we identify that video generation models inherently benefit from long-horizon supervision to align with language instructions, rendering them uniquely suitable for BVN tasks. Capitalizing on this insight, we propose introducing the video generation model into this field for the first time. Yet, the prohibitive latency for generating videos spanning tens of seconds makes real-world deployment impractical. To bridge this gap, we propose SparseVideoNav, achieving sub-second trajectory inference guided by a generated sparse future spanning a 20-second horizon. This yields a remarkable 27x speed-up compared to the unoptimized counterpart. Extensive real-world zero-shot experiments demonstrate that SparseVideoNav achieves 2.5x the success rate of state-of-the-art LLM baselines on BVN tasks and marks the first realization of such capability in challenging night scenes.
>
---
#### [new 047] Location-Aware Dispersion on Anonymous Graphs
- **分类: cs.DC; cs.DS; cs.MA; cs.RO**

- **简介: 该论文研究位置感知的分散问题，旨在让机器人在匿名图中按颜色分布到不同节点。解决了在未知网络规模下高效分散的算法设计与限制。**

- **链接: [https://arxiv.org/pdf/2602.05948v1](https://arxiv.org/pdf/2602.05948v1)**

> **作者:** Himani; Supantha Pandit; Gokarna Sharma
>
> **备注:** 3 tables, 2 figures, 6 pseudo-codes
>
> **摘要:** The well-studied DISPERSION problem is a fundamental coordination problem in distributed robotics, where a set of mobile robots must relocate so that each occupies a distinct node of a network. DISPERSION assumes that a robot can settle at any node as long as no other robot settles on that node. In this work, we introduce LOCATION-AWARE DISPERSION, a novel generalization of DISPERSION that incorporates location awareness: Let $G = (V, E)$ be an anonymous, connected, undirected graph with $n = |V|$ nodes, each labeled with a color $\sf{col}(v) \in C = \{c_1, \dots, c_t\}, t\leq n$. A set $R = \{r_1, \dots, r_k\}$ of $k \leq n$ mobile robots is given, where each robot $r_i$ has an associated color $\mathsf{col}(r_i) \in C$. Initially placed arbitrarily on the graph, the goal is to relocate the robots so that each occupies a distinct node of the same color. When $|C|=1$, LOCATION-AWARE DISPERSION reduces to DISPERSION. There is a solution to DISPERSION in graphs with any $k\leq n$ without knowing $k,n$. Like DISPERSION, the goal is to solve LOCATION-AWARE DISPERSION minimizing both time and memory requirement at each agent. We develop several deterministic algorithms with guaranteed bounds on both time and memory requirement. We also give an impossibility and a lower bound for any deterministic algorithm for LOCATION-AWARE DISPERSION. To the best of our knowledge, the presented results collectively establish the algorithmic feasibility of LOCATION-AWARE DISPERSION in anonymous networks and also highlight the challenges on getting an efficient solution compared to the solutions for DISPERSION.
>
---
## 更新

#### [replaced 001] A Sliced Learning Framework for Online Disturbance Identification in Quadrotor SO(3) Attitude Control
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于无人机姿态控制任务，旨在解决在线扰动识别问题。提出Sliced Learning框架，通过误差分解实现轻量级神经自适应，提升控制精度与效率。**

- **链接: [https://arxiv.org/pdf/2508.14422v3](https://arxiv.org/pdf/2508.14422v3)**

> **作者:** Tianhua Gao; Masashi Izumita; Kohji Tomita; Akiya Kamimura
>
> **备注:** v3: Major revision--Revised title; introduced the Sliced Learning framework; added comparative experiments, extended theoretical results, and supplementary materials (such as algorithms and proofs)
>
> **摘要:** This paper introduces a dimension-decomposed geometric learning framework called Sliced Learning for disturbance identification in quadrotor geometric attitude control. Instead of conventional learning-from-states, this framework adopts a learning-from-error strategy by using the Lie-algebraic error representation as the input feature, enabling axis-wise space decomposition (``slicing") while preserving the SO(3) structure. This is highly consistent with the geometric mechanism of cognitive control observed in neuroscience, where neural systems organize adaptive representations within structured subspaces to enable cognitive flexibility and efficiency. Based on this framework, we develop a lightweight and structurally interpretable Sliced Adaptive-Neuro Mapping (SANM) module. The high-dimensional mapping for online identification is axially ``sliced" into multiple low-dimensional submappings (``slices"), implemented by shallow neural networks and adaptive laws. These neural networks and adaptive laws are updated online via Lyapunov-based adaptation within their respective shared subspaces. To enhance interpretability, we prove exponential convergence despite time-varying disturbances and inertia uncertainties. To our knowledge, Sliced Learning is among the first frameworks to demonstrate lightweight online neural adaptation at 400 Hz on resource-constrained microcontroller units (MCUs), such as STM32, with real-world experimental validation.
>
---
#### [replaced 002] Haptic bilateral teleoperation system for free-hand dental procedures
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决自由手牙科操作中精度与安全问题。提出一种触觉双向遥操作系统，提升操作自然性、安全性和准确性。**

- **链接: [https://arxiv.org/pdf/2503.21288v3](https://arxiv.org/pdf/2503.21288v3)**

> **作者:** Lorenzo Pagliara; Vincenzo Petrone; Enrico Ferrentino; Andrea Chiacchio; Giovanni Russo
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Free-hand dental procedures are typically repetitive, time-consuming and require high precision and manual dexterity. Robots can play a key role in improving procedural accuracy and safety, enhancing patient comfort, and reducing operator workload. However, robotic solutions for free-hand procedures remain limited or completely lacking. To address this gap, we develop a haptic bilateral teleoperation system (HBTS) for free-hand dental procedures (FH-HBTS). The system includes a mechanical end-effector, compatible with standard clinical tools, and equipped with an endoscopic camera for improved visibility of the intervention site. By ensuring motion and force correspondence between the operator's and the robot's actions, monitored through visual feedback, we enhance the operator's sensory awareness and motor accuracy. Furthermore, to ensure procedural safety, we limit interaction forces by scaling the motion references provided to the admittance controller based solely on measured contact forces. This ensures effective force limitation in all contact states without requiring prior knowledge of the environment. The proposed FH-HBTS is validated both through a technical evaluation and an in-vitro pre-clinical study conducted on a dental model under clinically representative conditions. The results show that the system improves the naturalness, safety, and accuracy of teleoperation, highlighting its potential to enhance free-hand dental procedures.
>
---
#### [replaced 003] Learning to Plan & Schedule with Reinforcement-Learned Bimanual Robot Skills
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多臂机器人操作任务，解决复杂接触场景下的技能规划与调度问题。通过构建技能库和Transformer规划器，实现高效协同操作。**

- **链接: [https://arxiv.org/pdf/2510.25634v2](https://arxiv.org/pdf/2510.25634v2)**

> **作者:** Weikang Wan; Fabio Ramos; Xuning Yang; Caelan Garrett
>
> **摘要:** Long-horizon contact-rich bimanual manipulation presents a significant challenge, requiring complex coordination involving a mixture of parallel execution and sequential collaboration between arms. In this paper, we introduce a hierarchical framework that frames this challenge as an integrated skill planning & scheduling problem, going beyond purely sequential decision-making to support simultaneous skill invocation. Our approach is built upon a library of single-arm and bimanual primitive skills, each trained using Reinforcement Learning (RL) in GPU-accelerated simulation. We then train a Transformer-based planner on a dataset of skill compositions to act as a high-level scheduler, simultaneously predicting the discrete schedule of skills as well as their continuous parameters. We demonstrate that our method achieves higher success rates on complex, contact-rich tasks than end-to-end RL approaches and produces more efficient, coordinated behaviors than traditional sequential-only planners.
>
---
#### [replaced 004] Learning-Based Modeling of a Magnetically Steerable Soft Suction Device for Endoscopic Endonasal Interventions
- **分类: cs.RO**

- **简介: 该论文属于软体机器人建模任务，旨在解决磁控软吸力装置的形状预测问题。通过机器学习建立磁场参数与贝塞尔控制点之间的映射，实现高精度形状重建。**

- **链接: [https://arxiv.org/pdf/2507.15155v3](https://arxiv.org/pdf/2507.15155v3)**

> **作者:** Majid Roshanfar; Alex Zhang; Changyan He; Amir Hooshiar; Dale J. Podolsky; Thomas Looi; Eric Diller
>
> **摘要:** This paper introduces a learning-based modeling framework for a magnetically steerable soft suction device designed for endoscopic endonasal brain tumor resection. The device is miniaturized (4 mm outer diameter, 2 mm inner diameter, 40 mm length), 3D printed using biocompatible SIL 30 material, and integrates embedded Fiber Bragg Grating (FBG) sensors for real-time shape feedback. Shape reconstruction is represented using four Bezier control points, providing a compact representation of deformation. A data-driven model was trained on 5,097 experimental samples to learn the mapping from magnetic field parameters (magnitude: 0-14 mT, frequency: 0.2-1.0 Hz, vertical tip distances: 90-100 mm) to Bezier control points defining the robot's 3D shape. Both Neural Network (NN) and Random Forest (RF) architectures were compared. The RF model outperformed the NN, achieving a mean RMSE of 0.087 mm in control point prediction and 0.064 mm in shape reconstruction error. Feature importance analysis revealed that magnetic field components predominantly influence distal control points, while frequency and distance affect the base configuration. Unlike prior studies applying general machine learning to soft robotic data, this framework introduces a new paradigm linking magnetic actuation inputs directly to geometric Bezier control points, creating an interpretable, low-dimensional deformation representation. This integration of magnetic field characterization, embedded FBG sensing, and Bezier-based learning provides a unified strategy extensible to other magnetically actuated continuum robots. By enabling sub-millimeter shape prediction and real-time inference, this work advances intelligent control of magnetically actuated soft robotic tools in minimally invasive neurosurgery.
>
---
#### [replaced 005] HoRD: Robust Humanoid Control via History-Conditioned Reinforcement Learning and Online Distillation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决人形机器人在动态变化环境中的鲁棒性问题。提出HoRD框架，通过历史条件强化学习和在线蒸馏，实现零样本域适应。**

- **链接: [https://arxiv.org/pdf/2602.04412v2](https://arxiv.org/pdf/2602.04412v2)**

> **作者:** Puyue Wang; Jiawei Hu; Yan Gao; Junyan Wang; Yu Zhang; Gillian Dobbie; Tao Gu; Wafa Johal; Ting Dang; Hong Jia
>
> **摘要:** Humanoid robots can suffer significant performance drops under small changes in dynamics, task specifications, or environment setup. We propose HoRD, a two-stage learning framework for robust humanoid control under domain shift. First, we train a high-performance teacher policy via history-conditioned reinforcement learning, where the policy infers latent dynamics context from recent state--action trajectories to adapt online to diverse randomized dynamics. Second, we perform online distillation to transfer the teacher's robust control capabilities into a transformer-based student policy that operates on sparse root-relative 3D joint keypoint trajectories. By combining history-conditioned adaptation with online distillation, HoRD enables a single policy to adapt zero-shot to unseen domains without per-domain retraining. Extensive experiments show HoRD outperforms strong baselines in robustness and transfer, especially under unseen domains and external perturbations. Code and project page are available at https://tonywang-0517.github.io/hord/.
>
---
#### [replaced 006] RANGER: A Monocular Zero-Shot Semantic Navigation Framework through Contextual Adaptation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RANGER框架，解决零样本语义导航问题，通过单目相机和上下文学习，在复杂环境中高效寻找目标。**

- **链接: [https://arxiv.org/pdf/2512.24212v2](https://arxiv.org/pdf/2512.24212v2)**

> **作者:** Ming-Ming Yu; Yi Chen; Börje F. Karlsson; Wenjun Wu
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** Efficiently finding targets in complex environments is fundamental to real-world embodied applications. While recent advances in multimodal foundation models have enabled zero-shot object goal navigation, allowing robots to search for arbitrary objects without fine-tuning, existing methods face two key limitations: (1) heavy reliance on precise depth and pose information provided by simulators, which restricts applicability in real-world scenarios; and (2) lack of in-context learning (ICL) capability, making it difficult to quickly adapt to new environments, as in leveraging short videos. To address these challenges, we propose RANGER, a novel zero-shot, open-vocabulary semantic navigation framework that operates using only a monocular camera. Leveraging powerful 3D foundation models, RANGER eliminates the dependency on depth and pose while exhibiting strong ICL capability. By simply observing a short video of a new environment, the system can also significantly improve task efficiency without requiring architectural modifications or fine-tuning. The framework integrates several key components: keyframe-based 3D reconstruction, semantic point cloud generation, vision-language model (VLM)-driven exploration value estimation, high-level adaptive waypoint selection, and low-level action execution. Experiments on the HM3D benchmark and real-world environments demonstrate that RANGER achieves competitive performance in terms of navigation success rate and exploration efficiency, while showing superior ICL adaptability, with no previous 3D mapping of the environment required.
>
---
#### [replaced 007] MindDrive: A Vision-Language-Action Model for Autonomous Driving via Online Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决VLA模型在在线强化学习中的探索效率问题。提出MindDrive框架，通过语言决策与轨迹映射实现高效学习。**

- **链接: [https://arxiv.org/pdf/2512.13636v3](https://arxiv.org/pdf/2512.13636v3)**

> **作者:** Haoyu Fu; Diankun Zhang; Zongchuang Zhao; Jianfeng Cui; Hongwei Xie; Bing Wang; Guang Chen; Dingkang Liang; Xiang Bai
>
> **备注:** 16 pages, 12 figures, 6 tables; Project Page: https://xiaomi-mlab.github.io/MindDrive/
>
> **摘要:** Current Vision-Language-Action (VLA) paradigms in autonomous driving primarily rely on Imitation Learning (IL), which introduces inherent challenges such as distribution shift and causal confusion. Online Reinforcement Learning offers a promising pathway to address these issues through trial-and-error learning. However, applying online reinforcement learning to VLA models in autonomous driving is hindered by inefficient exploration in continuous action spaces. To overcome this limitation, we propose MindDrive, a VLA framework comprising a large language model (LLM) with two distinct sets of LoRA parameters. The one LLM serves as a Decision Expert for scenario reasoning and driving decision-making, while the other acts as an Action Expert that dynamically maps linguistic decisions into feasible trajectories. By feeding trajectory-level rewards back into the reasoning space, MindDrive enables trial-and-error learning over a finite set of discrete linguistic driving decisions, instead of operating directly in a continuous action space. This approach effectively balances optimal decision-making in complex scenarios, human-like driving behavior, and efficient exploration in online reinforcement learning. Using the lightweight Qwen-0.5B LLM, MindDrive achieves Driving Score (DS) of 78.04 and Success Rate (SR) of 55.09% on the challenging Bench2Drive benchmark. To the best of our knowledge, this is the first work to demonstrate the effectiveness of online reinforcement learning for the VLA model in autonomous driving.
>
---
#### [replaced 008] Do Robots Really Need Anthropomorphic Hands? -- A Comparison of Human and Robotic Hands
- **分类: cs.RO**

- **简介: 该论文属于机器人学任务，探讨是否需要仿人手设计。通过比较人类与机械手，分析其结构与功能，旨在解决机器人手设计的最优方案问题。**

- **链接: [https://arxiv.org/pdf/2508.05415v2](https://arxiv.org/pdf/2508.05415v2)**

> **作者:** Alexander Fabisch; Wadhah Zai El Amri; Chandandeep Singh; Nicolás Navarro-Guerrero
>
> **摘要:** Human manipulation skills represent a pinnacle of their voluntary motor functions, requiring the coordination of many degrees of freedom and processing of high-dimensional sensor input to achieve such a high level of dexterity. Thus, we attempt to answer whether the human hand, with its associated biomechanical properties, sensors, and control mechanisms, is an ideal that we should strive for in robotics-do we really need anthropomorphic robotic hands? This survey can help practitioners to make the trade-off between hand complexity and potential manipulation skills. We provide an overview of the human hand, a comparison of commercially available robotic and prosthetic hands, and a systematic review of hand mechanisms and skills that they are capable of. This leads to follow-up questions. What is the minimum requirement for mechanisms and sensors to implement most skills that a robot needs? What is missing to reach human-level dexterity? Can we improve upon human dexterity? Although complex five-fingered hands are often used as the ultimate goal for robotic manipulators, they are not necessary for all tasks. We found that wrist flexibility and finger abduction/adduction are often more important for manipulation capabilities. Increasing the number of fingers, actuators, or degrees of freedom is not always necessary. Three fingers often are a good compromise between simplicity and dexterity. Non-anthropomorphic hand designs with two opposing pairs of fingers or human hands with six fingers can further increase dexterity, suggesting that the human hand is not the optimum. Consequently, we argue for function-based rather than form-based biomimicry.
>
---
#### [replaced 009] FilMBot: A High-Speed Soft Parallel Robotic Micromanipulator
- **分类: cs.RO**

- **简介: 该论文介绍FilMBot，一种高速软体微操作机器人，解决软机器人速度慢的问题。通过电磁驱动和柔性结构，实现高精度与高速度操作。**

- **链接: [https://arxiv.org/pdf/2410.23059v3](https://arxiv.org/pdf/2410.23059v3)**

> **作者:** Jiangkun Yu; Houari Bettahar; Hakan Kandemir; Quan Zhou
>
> **备注:** 13 pages, 16 figures
>
> **摘要:** Soft robotic manipulators are generally slow despite their great adaptability, resilience, and compliance. This limitation also extends to current soft robotic micromanipulators. Here, we introduce FilMBot, a 3-DOF film-based, electromagnetically actuated, soft kinematic robotic micromanipulator achieving speeds up to 2117 °/s and 2456 °/s in α and \{beta} angular motions, with corresponding linear velocities of 1.61 m/s and 1.92 m/s using a 4-cm needle end-effector, 0.54 m/s along the Z axis, and 1.57 m/s during Z-axis morph switching. The robot can reach ~1.50 m/s in path-following tasks, with an operational bandwidth below ~30 Hz, and remains responsive at 50 Hz. It demonstrates high precision (~6.3 μm, or ~0.05% of its workspace) in path-following tasks, with precision remaining largely stable across frequencies. The novel combination of the low-stiffness soft kinematic film structure and strong electromagnetic actuation in FilMBot opens new avenues for soft robotics. Furthermore, its simple construction and inexpensive, readily accessible components could broaden the application of micromanipulators beyond current academic and professional users.
>
---
#### [replaced 010] Constraint-Aware Discrete-Time PID Gain Optimization for Robotic Joint Control Under Actuator Saturation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决执行器饱和下离散时间PID参数优化问题。通过分析稳定性、抗饱和策略和贝叶斯优化，提升控制性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.18639v3](https://arxiv.org/pdf/2601.18639v3)**

> **作者:** Ojasva Mishra; Xiaolong Wu; Min Xu
>
> **备注:** Pending IEEE Transactions on Robotics Publication
>
> **摘要:** The precise regulation of rotary actuation is fundamental in autonomous robotics, yet practical PID loops deviate from continuous-time theory due to discrete-time execution, actuator saturation, and small delays and measurement imperfections. We present an implementation-aware analysis and tuning workflow for saturated discrete-time joint control. We (i) derive PI stability regions under Euler and exact zero-order-hold (ZOH) discretizations using the Jury criterion, (ii) evaluate a discrete back-calculation anti-windup realization under saturation-dominant regimes, and (iii) propose a hybrid-certified Bayesian optimization workflow that screens analytically unstable candidates and behaviorally unsafe transients while optimizing a robust IAE objective with soft penalties on overshoot and saturation duty. Baseline sweeps ($τ=1.0$~s, $Δt=0.01$~s, $u\in[-10,10]$) quantify rise/settle trends for P/PI/PID. Under a randomized model family emulating uncertainty, delay, noise, quantization, and tighter saturation, robustness-oriented tuning improves median IAE from $0.843$ to $0.430$ while keeping median overshoot below $2\%$. In simulation-only tuning, the certification screen rejects $11.6\%$ of randomly sampled gains within bounds before full robust evaluation, improving sample efficiency.
>
---
#### [replaced 011] RFS: Reinforcement Learning with Residual Flow Steering for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决预训练策略泛化能力不足的问题。提出RFS框架，通过残差动作和潜在噪声优化，提升策略适应性与效率。**

- **链接: [https://arxiv.org/pdf/2602.01789v3](https://arxiv.org/pdf/2602.01789v3)**

> **作者:** Entong Su; Tyler Westenbroek; Anusha Nagabandi; Abhishek Gupta
>
> **摘要:** Imitation learning has emerged as an effective approach for bootstrapping sequential decision-making in robotics, achieving strong performance even in high-dimensional dexterous manipulation tasks. Recent behavior cloning methods further leverage expressive generative models, such as diffusion models and flow matching, to represent multimodal action distributions. However, policies pretrained in this manner often exhibit limited generalization and require additional fine-tuning to achieve robust performance at deployment time. Such adaptation must preserve the global exploration benefits of pretraining while enabling rapid correction of local execution errors. We propose Residual Flow Steering(RFS), a data-efficient reinforcement learning framework for adapting pretrained generative policies. RFS steers a pretrained flow-matching policy by jointly optimizing a residual action and a latent noise distribution, enabling complementary forms of exploration: local refinement through residual corrections and global exploration through latent-space modulation. This design allows efficient adaptation while retaining the expressive structure of the pretrained policy. We demonstrate the effectiveness of RFS on dexterous manipulation tasks, showing efficient fine-tuning in both simulation and real-world settings when adapting pretrained base policies. Project website:https://weirdlabuw.github.io/rfs.
>
---
#### [replaced 012] Improved Bag-of-Words Image Retrieval with Geometric Constraints for Ground Texture Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于图像检索任务，解决地面纹理定位问题。通过改进BoW方法，引入几何约束，提升定位精度和回环检测效果。**

- **链接: [https://arxiv.org/pdf/2505.11620v3](https://arxiv.org/pdf/2505.11620v3)**

> **作者:** Aaron Wilhelm; Nils Napp
>
> **备注:** Accepted to ICRA 2025
>
> **摘要:** Ground texture localization using a downward-facing camera offers a low-cost, high-precision localization solution that is robust to dynamic environments and requires no environmental modification. We present a significantly improved bag-of-words (BoW) image retrieval system for ground texture localization, achieving substantially higher accuracy for global localization and higher precision and recall for loop closure detection in SLAM. Our approach leverages an approximate $k$-means (AKM) vocabulary with soft assignment, and exploits the consistent orientation and constant scale constraints inherent to ground texture localization. Identifying the different needs of global localization vs. loop closure detection for SLAM, we present both high-accuracy and high-speed versions of our algorithm. We test the effect of each of our proposed improvements through an ablation study and demonstrate our method's effectiveness for both global localization and loop closure detection. With numerous ground texture localization systems already using BoW, our method can readily replace other generic BoW systems in their pipeline and immediately improve their results.
>
---
#### [replaced 013] Bench-NPIN: Benchmarking Non-prehensile Interactive Navigation
- **分类: cs.RO**

- **简介: 该论文提出Bench-NPIN，一个用于非抓取交互导航的基准测试平台，解决移动机器人在复杂环境中导航与物体互动的问题。**

- **链接: [https://arxiv.org/pdf/2505.12084v2](https://arxiv.org/pdf/2505.12084v2)**

> **作者:** Ninghan Zhong; Steven Caro; Avraiem Iskandar; Megnath Ramesh; Stephen L. Smith
>
> **备注:** This paper has been withdrawn by the authors. This paper has been superseded by arXiv:2512.11736
>
> **摘要:** Mobile robots are increasingly deployed in unstructured environments where obstacles and objects are movable. Navigation in such environments is known as interactive navigation, where task completion requires not only avoiding obstacles but also strategic interactions with movable objects. Non-prehensile interactive navigation focuses on non-grasping interaction strategies, such as pushing, rather than relying on prehensile manipulation. Despite a growing body of research in this field, most solutions are evaluated using case-specific setups, limiting reproducibility and cross-comparison. In this paper, we present Bench-NPIN, the first comprehensive benchmark for non-prehensile interactive navigation. Bench-NPIN includes multiple components: 1) a comprehensive range of simulated environments for non-prehensile interactive navigation tasks, including navigating a maze with movable obstacles, autonomous ship navigation in icy waters, box delivery, and area clearing, each with varying levels of complexity; 2) a set of evaluation metrics that capture unique aspects of interactive navigation, such as efficiency, interaction effort, and partial task completion; and 3) demonstrations using Bench-NPIN to evaluate example implementations of established baselines across environments. Bench-NPIN is an open-source Python library with a modular design. The code, documentation, and trained models can be found at https://github.com/IvanIZ/BenchNPIN.
>
---
#### [replaced 014] Dull, Dirty, Dangerous: Understanding the Past, Present, and Future of a Key Motivation for Robotics
- **分类: cs.RO**

- **简介: 该论文分析了机器人领域中“枯燥、肮脏、危险”工作的研究现状，指出定义和案例不足，提出框架以更好理解其对劳动的影响。属于机器人应用研究，解决如何准确界定和应用DDD问题。**

- **链接: [https://arxiv.org/pdf/2602.04746v2](https://arxiv.org/pdf/2602.04746v2)**

> **作者:** Nozomi Nakajima; Pedro Reynolds-Cuéllar; Caitrin Lynch; Kate Darling
>
> **摘要:** In robotics, the concept of "dull, dirty, and dangerous" (DDD) work has been used to motivate where robots might be useful. In this paper, we conduct an empirical analysis of robotics publications between 1980 and 2024 that mention DDD, and find that only 2.7% of publications define DDD and 8.7% of publications provide concrete examples of tasks or jobs that are DDD. We then review the social science literature on "dull," "dirty," and "dangerous" work to provide definitions and guidance on how to conceptualize DDD for robotics. Finally, we propose a framework that helps the robotics community consider the job context for our technology, encouraging a more informed perspective on how robotics may impact human labor.
>
---
#### [replaced 015] Doppler-SLAM: Doppler-Aided Radar-Inertial and LiDAR-Inertial Simultaneous Localization and Mapping
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，解决恶劣环境下定位与建图问题。融合雷达、LiDAR和惯性数据，提升精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2504.11634v4](https://arxiv.org/pdf/2504.11634v4)**

> **作者:** Dong Wang; Hannes Haag; Daniel Casado Herraez; Stefan May; Cyrill Stachniss; Andreas Nüchter
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Simultaneous localization and mapping (SLAM) is a critical capability for autonomous systems. Traditional SLAM approaches, which often rely on visual or LiDAR sensors, face significant challenges in adverse conditions such as low light or featureless environments. To overcome these limitations, we propose a novel Doppler-aided radar-inertial and LiDAR-inertial SLAM framework that leverages the complementary strengths of 4D radar, FMCW LiDAR, and inertial measurement units. Our system integrates Doppler velocity measurements and spatial data into a tightly-coupled front-end and graph optimization back-end to provide enhanced ego velocity estimation, accurate odometry, and robust mapping. We also introduce a Doppler-based scan-matching technique to improve front-end odometry in dynamic environments. In addition, our framework incorporates an innovative online extrinsic calibration mechanism, utilizing Doppler velocity and loop closure to dynamically maintain sensor alignment. Extensive evaluations on both public and proprietary datasets show that our system significantly outperforms state-of-the-art radar-SLAM and LiDAR-SLAM frameworks in terms of accuracy and robustness. To encourage further research, the code of our Doppler-SLAM and our dataset are available at: https://github.com/Wayne-DWA/Doppler-SLAM.
>
---
#### [replaced 016] Enriching physical-virtual interaction in AR gaming by tracking identical objects via an egocentric partial observation frame
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于AR场景中的物体重识别任务，解决动态环境中相同物体难以持续跟踪的问题。通过优化框架，利用单帧视角实现物体身份保持，提升交互流畅性。**

- **链接: [https://arxiv.org/pdf/2502.17399v2](https://arxiv.org/pdf/2502.17399v2)**

> **作者:** Liuchuan Yu; Ching-I Huang; Hsueh-Cheng Wang; Lap-Fai Yu
>
> **摘要:** Augmented reality (AR) games, particularly those designed for head-mounted displays, have grown increasingly prevalent. However, most existing systems depend on pre-scanned, static environments and rely heavily on continuous tracking or marker-based solutions, which limit adaptability in dynamic physical spaces. This is particularly problematic for AR headsets and glasses, which typically follow the user's head movement and cannot maintain a fixed, stationary view of the scene. Moreover, continuous scene observation is neither power-efficient nor practical for wearable devices, given their limited battery and processing capabilities. A persistent challenge arises when multiple identical objects are present in the environment-standard object tracking pipelines often fail to maintain consistent identities without uninterrupted observation or external sensors. These limitations hinder fluid physical-virtual interactions, especially in dynamic or occluded scenes where continuous tracking is infeasible. To address this, we introduce a novel optimization-based framework for re-identifying identical objects in AR scenes using only one partial egocentric observation frame captured by a headset. We formulate the problem as a label assignment task solved via integer programming, augmented with a Voronoi diagram-based pruning strategy to improve computational efficiency. This method reduces computation time by 50% while preserving 91% accuracy in simulated experiments. Moreover, we evaluated our approach in quantitative synthetic and quantitative real-world experiments. We also conducted three qualitative real-world experiments to demonstrate the practical utility and generalizability for enabling dynamic, markerless object interaction in AR environments. Our video demo is available at https://youtu.be/RwptEfLtW1U.
>
---
#### [replaced 017] SAP-CoPE: Social-Aware Planning using Cooperative Pose Estimation with Infrastructure Sensor Nodes
- **分类: cs.RO**

- **简介: 该论文属于室内导航任务，旨在解决自主系统在人群中的社交意识路径规划问题。通过融合协同感知与心理空间模型，提升轨迹的社交舒适性。**

- **链接: [https://arxiv.org/pdf/2504.05727v2](https://arxiv.org/pdf/2504.05727v2)**

> **作者:** Minghao Ning; Yufeng Yang; Shucheng Huang; Jiaming Zhong; Keqi Shu; Chen Sun; Ehsan Hashemi; Amir Khajepour
>
> **备注:** This paper has been submitted to the IEEE Transactions on Automation Science and Engineering
>
> **摘要:** Autonomous driving systems must operate smoothly in human-populated indoor environments, where challenges arise including limited perception and occlusions when relying only on onboard sensors, as well as the need for socially compliant motion planning that accounts for human psychological comfort zones. These factors complicate accurate recognition of human intentions and the generation of comfortable, socially aware trajectories. To address these challenges, we propose SAP-CoPE, an indoor navigation system that integrates cooperative infrastructure with a novel 3D human pose estimation method and a socially-aware model predictive control (MPC)-based motion planner. In the perception module, an optimization problem is formulated to account for uncertainty propagation in the camera projection matrix while enforcing human joint coherence. The proposed method is adaptable to both single- and multi-camera configurations and can incorporate sparse LiDAR point-cloud data. For motion planning, we integrate a psychology inspired personal-space field using the information from estimated human poses into an MPC framework to enhance socially comfort in human-populated environments. Extensive real-world evaluations demonstrate the effectiveness of the proposed approach in generating socially aware trajectories for autonomous systems.
>
---
#### [replaced 018] TACO: Temporal Consensus Optimization for Continual Neural Mapping
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决动态环境中持续学习的神经映射问题。提出TACO框架，通过时间共识优化实现无需回放的历史知识利用，平衡记忆效率与适应性。**

- **链接: [https://arxiv.org/pdf/2602.04516v2](https://arxiv.org/pdf/2602.04516v2)**

> **作者:** Xunlan Zhou; Hongrui Zhao; Negar Mehr
>
> **摘要:** Neural implicit mapping has emerged as a powerful paradigm for robotic navigation and scene understanding. However, real-world robotic deployment requires continual adaptation to changing environments under strict memory and computation constraints, which existing mapping systems fail to support. Most prior methods rely on replaying historical observations to preserve consistency and assume static scenes. As a result, they cannot adapt to continual learning in dynamic robotic settings. To address these challenges, we propose TACO (TemporAl Consensus Optimization), a replay-free framework for continual neural mapping. We reformulate mapping as a temporal consensus optimization problem, where we treat past model snapshots as temporal neighbors. Intuitively, our approach resembles a model consulting its own past knowledge. We update the current map by enforcing weighted consensus with historical representations. Our method allows reliable past geometry to constrain optimization while permitting unreliable or outdated regions to be revised in response to new observations. TACO achieves a balance between memory efficiency and adaptability without storing or replaying previous data. Through extensive simulated and real-world experiments, we show that TACO robustly adapts to scene changes, and consistently outperforms other continual learning baselines.
>
---
#### [replaced 019] Physical Human-Robot Interaction: A Critical Review of Safety Constraints
- **分类: eess.SY; cs.RO**

- **简介: 论文探讨物理人机交互中的安全约束，聚焦ISO/TS 15066标准，分析其假设与实际应用，强调能量在安全评估中的作用，旨在提升工业机器人系统的安全性与性能。**

- **链接: [https://arxiv.org/pdf/2601.19462v2](https://arxiv.org/pdf/2601.19462v2)**

> **作者:** Riccardo Zanella; Federico Califano; Stefano Stramigioli
>
> **摘要:** This paper aims to provide a clear and rigorous understanding of commonly recognized safety constraints in physical human-robot interaction, particularly regarding ISO/TS 15066. We investigate the derivation of these constraints, critically examine the underlying assumptions, and evaluate their practical implications for system-level safety and performance in industrially relevant scenarios. Key design parameters within safety-critical control architectures are identified, and numerical examples are provided to quantify performance degradation arising from typical approximations and design decisions in manufacturing environments. Within this analysis, the fundamental role of energy in safety assessment is emphasized, providing focused insights into energy-based safety methodologies for collaborative industrial robot systems.
>
---
