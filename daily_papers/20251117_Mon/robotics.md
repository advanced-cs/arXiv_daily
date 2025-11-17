# 机器人 cs.RO

- **最新发布 33 篇**

- **更新 25 篇**

## 最新发布

#### [new 001] MIGHTY: Hermite Spline-based Efficient Trajectory Planning
- **分类: cs.RO**

- **简介: 该论文提出MIGHTY，一种基于Hermite样条的轨迹规划方法，解决现有软约束方法时空优化解耦或搜索空间受限的问题。实现空间-时间联合优化，模拟中计算时间减9.3%、旅行时间减13.1%，硬件验证6.7 m/s高速飞行。**

- **链接: [https://arxiv.org/pdf/2511.10822v1](https://arxiv.org/pdf/2511.10822v1)**

> **作者:** Kota Kondo; Yuwei Wu; Vijay Kumar; Jonathan P. How
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** Hard-constraint trajectory planners often rely on commercial solvers and demand substantial computational resources. Existing soft-constraint methods achieve faster computation, but either (1) decouple spatial and temporal optimization or (2) restrict the search space. To overcome these limitations, we introduce MIGHTY, a Hermite spline-based planner that performs spatiotemporal optimization while fully leveraging the continuous search space of a spline. In simulation, MIGHTY achieves a 9.3% reduction in computation time and a 13.1% reduction in travel time over state-of-the-art baselines, with a 100% success rate. In hardware, MIGHTY completes multiple high-speed flights up to 6.7 m/s in a cluttered static environment and long-duration flights with dynamically added obstacles.
>
---
#### [new 002] Scalable Policy Evaluation with Video World Models
- **分类: cs.RO**

- **简介: 论文提出用动作条件视频生成模型解决机器人策略评估难题，避免真实世界测试的高成本和风险。通过利用互联网视频预训练，无需收集大量配对数据，实验验证了其在政策评估中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.11520v1](https://arxiv.org/pdf/2511.11520v1)**

> **作者:** Wei-Cheng Tseng; Jinwei Gu; Qinsheng Zhang; Hanzi Mao; Ming-Yu Liu; Florian Shkurti; Lin Yen-Chen
>
> **摘要:** Training generalist policies for robotic manipulation has shown great promise, as they enable language-conditioned, multi-task behaviors across diverse scenarios. However, evaluating these policies remains difficult because real-world testing is expensive, time-consuming, and labor-intensive. It also requires frequent environment resets and carries safety risks when deploying unproven policies on physical robots. Manually creating and populating simulation environments with assets for robotic manipulation has not addressed these issues, primarily due to the significant engineering effort required and the often substantial sim-to-real gap, both in terms of physics and rendering. In this paper, we explore the use of action-conditional video generation models as a scalable way to learn world models for policy evaluation. We demonstrate how to incorporate action conditioning into existing pre-trained video generation models. This allows leveraging internet-scale in-the-wild online videos during the pre-training stage, and alleviates the need for a large dataset of paired video-action data, which is expensive to collect for robotic manipulation. Our paper examines the effect of dataset diversity, pre-trained weight and common failure cases for the proposed evaluation pipeline.Our experiments demonstrate that, across various metrics, including policy ranking and the correlation between actual policy values and predicted policy values, these models offer a promising approach for evaluating policies without requiring real-world interactions.
>
---
#### [new 003] Decentralized Swarm Control via SO(3) Embeddings for 3D Trajectories
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出基于SO(3)嵌入的去中心化多智能体控制方法，解决3D周期轨迹生成问题。它仅需位置输入、消除速度需求，通过相位控制器确保均匀分离，并提供稳定性证明。实验验证了方法在复杂动态中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.10858v1](https://arxiv.org/pdf/2511.10858v1)**

> **作者:** Dimitria Silveria; Kleber Cabral; Peter Jardine; Sidney Givigi
>
> **摘要:** This paper presents a novel decentralized approach for achieving emergent behavior in multi-agent systems with minimal information sharing. Based on prior work in simple orbits, our method produces a broad class of stable, periodic trajectories by stabilizing the system around a Lie group-based geometric embedding. Employing the Lie group SO(3), we generate a wider range of periodic curves than existing quaternion-based methods. Furthermore, we exploit SO(3) properties to eliminate the need for velocity inputs, allowing agents to receive only position inputs. We also propose a novel phase controller that ensures uniform agent separation, along with a formal stability proof. Validation through simulations and experiments showcases the method's adaptability to complex low-level dynamics and disturbances.
>
---
#### [new 004] SimTac: A Physics-Based Simulator for Vision-Based Tactile Sensing with Biomorphic Structures
- **分类: cs.RO**

- **简介: 论文提出SimTac框架，解决机器人触觉传感器生物形态设计缺失问题。通过粒子变形建模、光场渲染和神经网络，实现生物启发触觉传感器的模拟与验证，并在物体分类、滑动检测等Sim2Real任务中证明有效性。**

- **链接: [https://arxiv.org/pdf/2511.11456v1](https://arxiv.org/pdf/2511.11456v1)**

> **作者:** Xuyang Zhang; Jiaqi Jiang; Zhuo Chen; Yongqiang Zhao; Tianqi Yang; Daniel Fernandes Gomes; Jianan Wang; Shan Luo
>
> **摘要:** Tactile sensing in biological organisms is deeply intertwined with morphological form, such as human fingers, cat paws, and elephant trunks, which enables rich and adaptive interactions through a variety of geometrically complex structures. In contrast, vision-based tactile sensors in robotics have been limited to simple planar geometries, with biomorphic designs remaining underexplored. To address this gap, we present SimTac, a physics-based simulation framework for the design and validation of biomorphic tactile sensors. SimTac consists of particle-based deformation modeling, light-field rendering for photorealistic tactile image generation, and a neural network for predicting mechanical responses, enabling accurate and efficient simulation across a wide range of geometries and materials. We demonstrate the versatility of SimTac by designing and validating physical sensor prototypes inspired by biological tactile structures and further demonstrate its effectiveness across multiple Sim2Real tactile tasks, including object classification, slip detection, and contact safety assessment. Our framework bridges the gap between bio-inspired design and practical realisation, expanding the design space of tactile sensors and paving the way for tactile sensing systems that integrate morphology and sensing to enable robust interaction in unstructured environments.
>
---
#### [new 005] Terrain Costmap Generation via Scaled Preference Conditioning
- **分类: cs.RO**

- **简介: 论文提出SPACER方法解决地形成本图生成任务，旨在同时实现新地形泛化和测试时快速适应相对成本。通过合成数据训练和缩放偏好条件化，SPACER在五个环境中优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.11529v1](https://arxiv.org/pdf/2511.11529v1)**

> **作者:** Luisa Mao; Garret Warnell; Peter Stone; Joydeep Biswas
>
> **摘要:** Successful autonomous robot navigation in off-road domains requires the ability to generate high-quality terrain costmaps that are able to both generalize well over a wide variety of terrains and rapidly adapt relative costs at test time to meet mission-specific needs. Existing approaches for costmap generation allow for either rapid test-time adaptation of relative costs (e.g., semantic segmentation methods) or generalization to new terrain types (e.g., representation learning methods), but not both. In this work, we present scaled preference conditioned all-terrain costmap generation (SPACER), a novel approach for generating terrain costmaps that leverages synthetic data during training in order to generalize well to new terrains, and allows for rapid test-time adaptation of relative costs by conditioning on a user-specified scaled preference context. Using large-scale aerial maps, we provide empirical evidence that SPACER outperforms other approaches at generating costmaps for terrain navigation, with the lowest measured regret across varied preferences in five of seven environments for global path planning.
>
---
#### [new 006] Collaborative Representation Learning for Alignment of Tactile, Language, and Vision Modalities
- **分类: cs.RO; cs.CV**

- **简介: 论文提出TLV-CoRe方法解决触觉-语言-视觉模态对齐任务。针对触觉传感器不标准化导致的冗余特征和跨模态交互不足，引入Sensor-Aware Modulator统一触觉特征、Unified Bridging Adapter增强三模态交互，并设计RSS评估框架。实验表明其显著提升跨模态对齐效果。**

- **链接: [https://arxiv.org/pdf/2511.11512v1](https://arxiv.org/pdf/2511.11512v1)**

> **作者:** Yiyun Zhou; Mingjing Xu; Jingwei Shi; Quanjiang Li; Jingyuan Chen
>
> **摘要:** Tactile sensing offers rich and complementary information to vision and language, enabling robots to perceive fine-grained object properties. However, existing tactile sensors lack standardization, leading to redundant features that hinder cross-sensor generalization. Moreover, existing methods fail to fully integrate the intermediate communication among tactile, language, and vision modalities. To address this, we propose TLV-CoRe, a CLIP-based Tactile-Language-Vision Collaborative Representation learning method. TLV-CoRe introduces a Sensor-Aware Modulator to unify tactile features across different sensors and employs tactile-irrelevant decoupled learning to disentangle irrelevant tactile features. Additionally, a Unified Bridging Adapter is introduced to enhance tri-modal interaction within the shared representation space. To fairly evaluate the effectiveness of tactile models, we further propose the RSS evaluation framework, focusing on Robustness, Synergy, and Stability across different methods. Experimental results demonstrate that TLV-CoRe significantly improves sensor-agnostic representation learning and cross-modal alignment, offering a new direction for multimodal tactile representation.
>
---
#### [new 007] Scalable Coverage Trajectory Synthesis on GPUs as Statistical Inference
- **分类: cs.RO**

- **简介: 论文将覆盖运动规划表述为统计推断问题，利用流匹配技术解耦轨迹梯度生成与控制合成，实现GPU并行加速，显著提升计算效率。**

- **链接: [https://arxiv.org/pdf/2511.11514v1](https://arxiv.org/pdf/2511.11514v1)**

> **作者:** Max M. Sun; Jueun Kwon; Todd Murphey
>
> **备注:** Presented at the "Workshop on Fast Motion Planning and Control in the Era of Parallelism" at Robotics: Science and Systems 2025. Workshop website: https://sites.google.com/rice.edu/parallelized-planning-control/
>
> **摘要:** Coverage motion planning is essential to a wide range of robotic tasks. Unlike conventional motion planning problems, which reason over temporal sequences of states, coverage motion planning requires reasoning over the spatial distribution of entire trajectories, making standard motion planning methods limited in computational efficiency and less amenable to modern parallelization frameworks. In this work, we formulate the coverage motion planning problem as a statistical inference problem from the perspective of flow matching, a generative modeling technique that has gained significant attention in recent years. The proposed formulation unifies commonly used statistical discrepancy measures, such as Kullback-Leibler divergence and Sinkhorn divergence, with a standard linear quadratic regulator problem. More importantly, it decouples the generation of trajectory gradients for coverage from the synthesis of control under nonlinear system dynamics, enabling significant acceleration through parallelization on modern computational architectures, particularly Graphics Processing Units (GPUs). This paper focuses on the advantages of this formulation in terms of scalability through parallelization, highlighting its computational benefits compared to conventional methods based on waypoint tracking.
>
---
#### [new 008] Simulating an Autonomous System in CARLA using ROS 2
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对自主赛车任务，解决高速下感知与规划问题。在CARLA模拟器中，基于ROS 2集成LiDAR、相机等传感器，实现35米外锥桶检测与轨迹优化，软件栈经仿真验证后移植至Jetson AGX Orin硬件。**

- **链接: [https://arxiv.org/pdf/2511.11310v1](https://arxiv.org/pdf/2511.11310v1)**

> **作者:** Joseph Abdo; Aditya Shibu; Moaiz Saeed; Abdul Maajid Aga; Apsara Sivaprazad; Mohamed Al-Musleh
>
> **摘要:** Autonomous racing offers a rigorous setting to stress test perception, planning, and control under high speed and uncertainty. This paper proposes an approach to design and evaluate a software stack for an autonomous race car in CARLA: Car Learning to Act simulator, targeting competitive driving performance in the Formula Student UK Driverless (FS-AI) 2025 competition. By utilizing a 360° light detection and ranging (LiDAR), stereo camera, global navigation satellite system (GNSS), and inertial measurement unit (IMU) sensor via ROS 2 (Robot Operating System), the system reliably detects the cones marking the track boundaries at distances of up to 35 m. Optimized trajectories are computed considering vehicle dynamics and simulated environmental factors such as visibility and lighting to navigate the track efficiently. The complete autonomous stack is implemented in ROS 2 and validated extensively in CARLA on a dedicated vehicle (ADS-DV) before being ported to the actual hardware, which includes the Jetson AGX Orin 64GB, ZED2i Stereo Camera, Robosense Helios 16P LiDAR, and CHCNAV Inertial Navigation System (INS).
>
---
#### [new 009] Miniature Testbed for Validating Multi-Agent Cooperative Autonomous Driving
- **分类: cs.RO**

- **简介: 论文设计1:15比例微型测试床CIVAT，解决协作自动驾驶中智能基础设施缺失问题，通过整合V2V/V2I通信与ROS2框架，验证了基础设施感知和交叉路口管理。**

- **链接: [https://arxiv.org/pdf/2511.11022v1](https://arxiv.org/pdf/2511.11022v1)**

> **作者:** Hyunchul Bae; Eunjae Lee; Jehyeop Han; Minhee Kang; Jaehyeon Kim; Junggeun Seo; Minkyun Noh; Heejin Ahn
>
> **备注:** 8 pages
>
> **摘要:** Cooperative autonomous driving, which extends vehicle autonomy by enabling real-time collaboration between vehicles and smart roadside infrastructure, remains a challenging yet essential problem. However, none of the existing testbeds employ smart infrastructure equipped with sensing, edge computing, and communication capabilities. To address this gap, we design and implement a 1:15-scale miniature testbed, CIVAT, for validating cooperative autonomous driving, consisting of a scaled urban map, autonomous vehicles with onboard sensors, and smart infrastructure. The proposed testbed integrates V2V and V2I communication with the publish-subscribe pattern through a shared Wi-Fi and ROS2 framework, enabling information exchange between vehicles and infrastructure to realize cooperative driving functionality. As a case study, we validate the system through infrastructure-based perception and intersection management experiments.
>
---
#### [new 010] Volumetric Ergodic Control
- **分类: cs.RO; cs.AI**

- **简介: 论文提出体积遍历控制（Volumetric Ergodic Control），解决机器人空间覆盖任务中忽略物理体积的问题。通过引入体积状态表示，优化覆盖效率超两倍，保持100%任务完成率，适用于多种机器人操作与搜索场景。**

- **链接: [https://arxiv.org/pdf/2511.11533v1](https://arxiv.org/pdf/2511.11533v1)**

> **作者:** Jueun Kwon; Max M. Sun; Todd Murphey
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Ergodic control synthesizes optimal coverage behaviors over spatial distributions for nonlinear systems. However, existing formulations model the robot as a non-volumetric point, but in practice a robot interacts with the environment through its body and sensors with physical volume. In this work, we introduce a new ergodic control formulation that optimizes spatial coverage using a volumetric state representation. Our method preserves the asymptotic coverage guarantees of ergodic control, adds minimal computational overhead for real-time control, and supports arbitrary sample-based volumetric models. We evaluate our method across search and manipulation tasks -- with multiple robot dynamics and end-effector geometries or sensor models -- and show that it improves coverage efficiency by more than a factor of two while maintaining a 100% task completion rate across all experiments, outperforming the standard ergodic control method. Finally, we demonstrate the effectiveness of our method on a robot arm performing mechanical erasing tasks.
>
---
#### [new 011] A Comparative Evaluation of Prominent Methods in Autonomous Vehicle Certification
- **分类: cs.RO; cs.CY**

- **简介: 该论文比较自动驾驶车辆认证的突出方法，解决安全验证问题，开发认证流程并确定应用阶段、角色和领域。**

- **链接: [https://arxiv.org/pdf/2511.11484v1](https://arxiv.org/pdf/2511.11484v1)**

> **作者:** Mustafa Erdem Kırmızıgül; Hasan Feyzi Doğruyol; Haluk Bayram
>
> **摘要:** The "Vision Zero" policy, introduced by the Swedish Parliament in 1997, aims to eliminate fatalities and serious injuries resulting from traffic accidents. To achieve this goal, the use of self-driving vehicles in traffic is envisioned and a roadmap for the certification of self-driving vehicles is aimed to be determined. However, it is still unclear how the basic safety requirements that autonomous vehicles must meet will be verified and certified, and which methods will be used. This paper focuses on the comparative evaluation of the prominent methods planned to be used in the certification process of autonomous vehicles. It examines the prominent methods used in the certification process, develops a pipeline for the certification process of autonomous vehicles, and determines the stages, actors, and areas where the addressed methods can be applied.
>
---
#### [new 012] Terradynamics and design of tip-extending robotic anchors
- **分类: cs.RO; cond-mat.soft**

- **简介: 该论文研究树根启发的tip-extending机器人锚设计，解决传统锚插入力大、设备笨重问题。通过颗粒力学分析，提出设计准则（延伸深度、毛发状突起等），开发轻量级锚，实现40:1锚固力/重量比，成功部署于火星模拟土壤。**

- **链接: [https://arxiv.org/pdf/2511.10901v1](https://arxiv.org/pdf/2511.10901v1)**

> **作者:** Deniz Kerimoglu; Nicholas D. Naclerio; Sean Chu; Andrew Krohn; Vineet Kupunaram; Alexander Schepelmann; Daniel I. Goldman; Elliot W. Hawkes
>
> **摘要:** Most engineered pilings require substantially more force to be driven into the ground than they can resist during extraction. This requires relatively heavy equipment for insertion, which is problematic for anchoring in hard-to-access sites, including in extraterrestrial locations. In contrast, for tree roots, the external reaction force required to extract is much greater than required to insert--little more than the weight of the seed initiates insertion. This is partly due to the mechanism by which roots insert into the ground: tip extension. Proof-of-concept robotic prototypes have shown the benefits of using this mechanism, but a rigorous understanding of the underlying granular mechanics and how they inform the design of a robotic anchor is lacking. Here, we study the terradynamics of tip-extending anchors compared to traditional piling-like intruders, develop a set of design insights, and apply these to create a deployable robotic anchor. Specifically, we identify that to increase an anchor's ratio of extraction force to insertion force, it should: (i) extend beyond a critical depth; (ii) include hair-like protrusions; (iii) extend near-vertically, and (iv) incorporate multiple smaller anchors rather than a single large anchor. Synthesizing these insights, we developed a lightweight, soft robotic, root-inspired anchoring device that inserts into the ground with a reaction force less than its weight. We demonstrate that the 300 g device can deploy a series of temperature sensors 45 cm deep into loose Martian regolith simulant while anchoring with an average of 120 N, resulting in an anchoring-to-weight ratio of 40:1.
>
---
#### [new 013] Attentive Feature Aggregation or: How Policies Learn to Stop Worrying about Robustness and Attend to Task-Relevant Visual Cues
- **分类: cs.RO; cs.CV**

- **简介: 论文针对视觉运动策略训练中预训练视觉表示导致的鲁棒性问题，提出Attentive Feature Aggregation (AFA)机制，学习关注任务相关视觉线索忽略干扰，显著提升扰动场景性能，无需额外数据增强。**

- **链接: [https://arxiv.org/pdf/2511.10762v1](https://arxiv.org/pdf/2511.10762v1)**

> **作者:** Nikolaos Tsagkas; Andreas Sochopoulos; Duolikun Danier; Sethu Vijayakumar; Alexandros Kouris; Oisin Mac Aodha; Chris Xiaoxuan Lu
>
> **备注:** This paper stems from a split of our earlier work "When Pre-trained Visual Representations Fall Short: Limitations in Visuo-Motor Robot Learning." While "The Temporal Trap" replaces the original and focuses on temporal entanglement, this companion study examines policy robustness and task-relevant visual cue selection
>
> **摘要:** The adoption of pre-trained visual representations (PVRs), leveraging features from large-scale vision models, has become a popular paradigm for training visuomotor policies. However, these powerful representations can encode a broad range of task-irrelevant scene information, making the resulting trained policies vulnerable to out-of-domain visual changes and distractors. In this work we address visuomotor policy feature pooling as a solution to the observed lack of robustness in perturbed scenes. We achieve this via Attentive Feature Aggregation (AFA), a lightweight, trainable pooling mechanism that learns to naturally attend to task-relevant visual cues, ignoring even semantically rich scene distractors. Through extensive experiments in both simulation and the real world, we demonstrate that policies trained with AFA significantly outperform standard pooling approaches in the presence of visual perturbations, without requiring expensive dataset augmentation or fine-tuning of the PVR. Our findings show that ignoring extraneous visual information is a crucial step towards deploying robust and generalisable visuomotor policies. Project Page: tsagkas.github.io/afa
>
---
#### [new 014] AdaptPNP: Integrating Prehensile and Non-Prehensile Skills for Adaptive Robotic Manipulation
- **分类: cs.RO**

- **简介: 论文提出AdaptPNP框架，利用视觉语言模型整合预握与非预握机器人操作技能，解决任务泛化和动作协调问题。通过VLM生成高层计划、数字孪生预测物体姿态，并实现在线重规划，有效提升混合操作能力。**

- **链接: [https://arxiv.org/pdf/2511.11052v1](https://arxiv.org/pdf/2511.11052v1)**

> **作者:** Jinxuan Zhu; Chenrui Tie; Xinyi Cao; Yuran Wang; Jingxiang Guo; Zixuan Chen; Haonan Chen; Junting Chen; Yangyu Xiao; Ruihai Wu; Lin Shao
>
> **摘要:** Non-prehensile (NP) manipulation, in which robots alter object states without forming stable grasps (for example, pushing, poking, or sliding), significantly broadens robotic manipulation capabilities when grasping is infeasible or insufficient. However, enabling a unified framework that generalizes across different tasks, objects, and environments while seamlessly integrating non-prehensile and prehensile (P) actions remains challenging: robots must determine when to invoke NP skills, select the appropriate primitive for each context, and compose P and NP strategies into robust, multi-step plans. We introduce ApaptPNP, a vision-language model (VLM)-empowered task and motion planning framework that systematically selects and combines P and NP skills to accomplish diverse manipulation objectives. Our approach leverages a VLM to interpret visual scene observations and textual task descriptions, generating a high-level plan skeleton that prescribes the sequence and coordination of P and NP actions. A digital-twin based object-centric intermediate layer predicts desired object poses, enabling proactive mental rehearsal of manipulation sequences. Finally, a control module synthesizes low-level robot commands, with continuous execution feedback enabling online task plan refinement and adaptive replanning through the VLM. We evaluate ApaptPNP across representative P&NP hybrid manipulation tasks in both simulation and real-world environments. These results underscore the potential of hybrid P&NP manipulation as a crucial step toward general-purpose, human-level robotic manipulation capabilities. Project Website: https://sites.google.com/view/adaptpnp/home
>
---
#### [new 015] Experiences from Benchmarking Vision-Language-Action Models for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 论文系统评估VLA模型在机器人操作任务中的性能，基准测试四种模型（ACT、OpenVLA-OFT、RDT-1B、π₀）于模拟和ALOHA Mobile平台。建立标准化框架，分析准确性、适应性及语言指令遵循，发现π₀分布外适应性好，ACT分布内稳定，提供部署实用见解。**

- **链接: [https://arxiv.org/pdf/2511.11298v1](https://arxiv.org/pdf/2511.11298v1)**

> **作者:** Yihao Zhang; Yuankai Qi; Xi Zheng
>
> **摘要:** Foundation models applied in robotics, particularly \textbf{Vision--Language--Action (VLA)} models, hold great promise for achieving general-purpose manipulation. Yet, systematic real-world evaluations and cross-model comparisons remain scarce. This paper reports our \textbf{empirical experiences} from benchmarking four representative VLAs -- \textbf{ACT}, \textbf{OpenVLA--OFT}, \textbf{RDT-1B}, and \boldmath{$π_0$} -- across four manipulation tasks conducted in both simulation and on the \textbf{ALOHA Mobile} platform. We establish a \textbf{standardized evaluation framework} that measures performance along three key dimensions: (1) \textit{accuracy and efficiency} (success rate and time-to-success), (2) \textit{adaptability} across in-distribution, spatial out-of-distribution, and instance-plus-spatial out-of-distribution settings, and (3) \textit{language instruction-following accuracy}. Through this process, we observe that \boldmath{$π_0$} demonstrates superior adaptability in out-of-distribution scenarios, while \textbf{ACT} provides the highest stability in-distribution. Further analysis highlights differences in computational demands, data-scaling behavior, and recurring failure modes such as near-miss grasps, premature releases, and long-horizon state drift. These findings reveal practical trade-offs among VLA model architectures in balancing precision, generalization, and deployment cost, offering actionable insights for selecting and deploying VLAs in real-world robotic manipulation tasks.
>
---
#### [new 016] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective
- **分类: cs.RO; cs.CV**

- **简介: 论文解决机器人抓取中对象级记忆推理问题，针对非马尔可夫环境中VLA模型因物体历史缺失导致决策失效的挑战。提出LIBERO-Mem任务套件和Embodied-SlotSSM框架，通过槽位中心建模实现时间可扩展的动作预测。**

- **链接: [https://arxiv.org/pdf/2511.11478v1](https://arxiv.org/pdf/2511.11478v1)**

> **作者:** Nhat Chung; Taisei Hanyu; Toan Nguyen; Huy Le; Frederick Bumgarner; Duy Minh Ho Nguyen; Khoa Vo; Kashu Yamazaki; Chase Rainwater; Tung Kieu; Anh Nguyen; Ngan Le
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** As embodied agents operate in increasingly complex environments, the ability to perceive, track, and reason about individual object instances over time becomes essential, especially in tasks requiring sequenced interactions with visually similar objects. In these non-Markovian settings, key decision cues are often hidden in object-specific histories rather than the current scene. Without persistent memory of prior interactions (what has been interacted with, where it has been, or how it has changed) visuomotor policies may fail, repeat past actions, or overlook completed ones. To surface this challenge, we introduce LIBERO-Mem, a non-Markovian task suite for stress-testing robotic manipulation under object-level partial observability. It combines short- and long-horizon object tracking with temporally sequenced subgoals, requiring reasoning beyond the current frame. However, vision-language-action (VLA) models often struggle in such settings, with token scaling quickly becoming intractable even for tasks spanning just a few hundred frames. We propose Embodied-SlotSSM, a slot-centric VLA framework built for temporal scalability. It maintains spatio-temporally consistent slot identities and leverages them through two mechanisms: (1) slot-state-space modeling for reconstructing short-term history, and (2) a relational encoder to align the input tokens with action decoding. Together, these components enable temporally grounded, context-aware action prediction. Experiments show Embodied-SlotSSM's baseline performance on LIBERO-Mem and general tasks, offering a scalable solution for non-Markovian reasoning in object-centric robotic policies.
>
---
#### [new 017] $\rm{A}^{\rm{SAR}}$: $\varepsilon$-Optimal Graph Search for Minimum Expected-Detection-Time Paths with Path Budget Constraints for Search and Rescue
- **分类: cs.RO**

- **简介: 该论文针对搜救（SAR）路径优化任务，提出 $\rm{A}^{\rm{SAR}}$ 算法，以 $\varepsilon$-最优方式最小化预期检测时间并满足路径预算约束。实测150秒内成功定位目标，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.10792v1](https://arxiv.org/pdf/2511.10792v1)**

> **作者:** Eric Mugford; Jonathan D. Gammell
>
> **备注:** Submitted to IEEE International Conference on Robotics and Automation (ICRA) 2026, 8 pages, 4 figures, 2 tables. The corresponding video can be found at https://www.youtube.com/watch?v=R73-YKWY78M
>
> **摘要:** Searches are conducted to find missing persons and/or objects given uncertain information, imperfect observers and large search areas in Search and Rescue (SAR). In many scenarios, such as Maritime SAR, expected survival times are short and optimal search could increase the likelihood of success. This optimization problem is complex for nontrivial problems given its probabilistic nature. Stochastic optimization methods search large problems by nondeterministically sampling the space to reduce the effective size of the problem. This has been used in SAR planning to search otherwise intractably large problems but the stochastic nature provides no formal guarantees on the quality of solutions found in finite time. This paper instead presents $\rm{A}^{\rm{SAR}}$, an $\varepsilon$-optimal search algorithm for SAR planning. It calculates a heuristic to bound the search space and uses graph-search methods to find solutions that are formally guaranteed to be within a user-specified factor, $\varepsilon$, of the optimal solution. It finds better solutions faster than existing optimization approaches in operational simulations. It is also demonstrated with a real-world field trial on Lake Ontario, Canada, where it was used to locate a drifting manikin in only 150s.
>
---
#### [new 018] Sashimi-Bot: Autonomous Tri-manual Advanced Manipulation and Cutting of Deformable Objects
- **分类: cs.RO**

- **简介: 该论文提出Sashimi-Bot系统，解决变形物体（如三文鱼片）的自主抓取与切割问题。系统采用三机器人协作，结合深度强化学习及视觉触觉反馈，实现对易变形、多变对象的鲁棒操作，为机器人变形物体处理树立新范式。**

- **链接: [https://arxiv.org/pdf/2511.11223v1](https://arxiv.org/pdf/2511.11223v1)**

> **作者:** Sverre Herland; Amit Parag; Elling Ruud Øye; Fangyi Zhang; Fouad Makiyeh; Aleksander Lillienskiold; Abhaya Pal Singh; Edward H. Adelson; Francois Chaumette; Alexandre Krupa; Peter Corke; Ekrem Misimi
>
> **摘要:** Advanced robotic manipulation of deformable, volumetric objects remains one of the greatest challenges due to their pliancy, frailness, variability, and uncertainties during interaction. Motivated by these challenges, this article introduces Sashimi-Bot, an autonomous multi-robotic system for advanced manipulation and cutting, specifically the preparation of sashimi. The objects that we manipulate, salmon loins, are natural in origin and vary in size and shape, they are limp and deformable with poorly characterized elastoplastic parameters, while also being slippery and hard to hold. The three robots straighten the loin; grasp and hold the knife; cut with the knife in a slicing motion while cooperatively stabilizing the loin during cutting; and pick up the thin slices from the cutting board or knife blade. Our system combines deep reinforcement learning with in-hand tool shape manipulation, in-hand tool cutting, and feedback of visual and tactile information to achieve robustness to the variabilities inherent in this task. This work represents a milestone in robotic manipulation of deformable, volumetric objects that may inspire and enable a wide range of other real-world applications.
>
---
#### [new 019] Collaborative Multi-Robot Non-Prehensile Manipulation via Flow-Matching Co-Generation
- **分类: cs.RO; cs.MA**

- **简介: 该论文解决协同多机器人非抓取操作任务，针对杂乱环境中多物体长时任务的协调挑战。提出统一框架，整合流匹配共生成与运动规划，使生成模型共生成接触点和轨迹，运动规划器统一机器人与对象级推理，实验优于基线。**

- **链接: [https://arxiv.org/pdf/2511.10874v1](https://arxiv.org/pdf/2511.10874v1)**

> **作者:** Yorai Shaoul; Zhe Chen; Mohamed Naveed Gul Mohamed; Federico Pecora; Maxim Likhachev; Jiaoyang Li
>
> **摘要:** Coordinating a team of robots to reposition multiple objects in cluttered environments requires reasoning jointly about where robots should establish contact, how to manipulate objects once contact is made, and how to navigate safely and efficiently at scale. Prior approaches typically fall into two extremes -- either learning the entire task or relying on privileged information and hand-designed planners -- both of which struggle to handle diverse objects in long-horizon tasks. To address these challenges, we present a unified framework for collaborative multi-robot, multi-object non-prehensile manipulation that integrates flow-matching co-generation with anonymous multi-robot motion planning. Within this framework, a generative model co-generates contact formations and manipulation trajectories from visual observations, while a novel motion planner conveys robots at scale. Crucially, the same planner also supports coordination at the object level, assigning manipulated objects to larger target structures and thereby unifying robot- and object-level reasoning within a single algorithmic framework. Experiments in challenging simulated environments demonstrate that our approach outperforms baselines in both motion planning and manipulation tasks, highlighting the benefits of generative co-design and integrated planning for scaling collaborative manipulation to complex multi-agent, multi-object settings. Visit gco-paper.github.io for code and demonstrations.
>
---
#### [new 020] Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出多阶段强化学习训练人形机器人打羽毛球，解决动态环境下的全身协调问题。通过三阶段课程学习步法、击球和优化，结合EKF轨迹预测，实现模拟中21次连续击球和真实世界高精度击球（速度10 m/s）。**

- **链接: [https://arxiv.org/pdf/2511.11218v1](https://arxiv.org/pdf/2511.11218v1)**

> **作者:** Chenhao Liu; Leyun Jiang; Yibo Wang; Kairan Yao; Jinchen Fu; Xiaoyu Ren
>
> **摘要:** Humanoid robots have demonstrated strong capability for interacting with deterministic scenes across locomotion, manipulation, and more challenging loco-manipulation tasks. Yet the real world is dynamic, quasi-static interactions are insufficient to cope with the various environmental conditions. As a step toward more dynamic interaction scenario, we present a reinforcement-learning-based training pipeline that produces a unified whole-body controller for humanoid badminton, enabling coordinated lower-body footwork and upper-body striking without any motion priors or expert demonstrations. Training follows a three-stage curriculum: first footwork acquisition, then precision-guided racket swing generation, and finally task-focused refinement, yielding motions in which both legs and arms serve the hitting objective. For deployment, we incorporate an Extended Kalman Filter (EKF) to estimate and predict shuttlecock trajectories for target striking. We also introduce a prediction-free variant that dispenses with EKF and explicit trajectory prediction. To validate the framework, we conduct five sets of experiment in both simulation and the real world. In simulation, two robots sustain a rally of 21 consecutive hits. Moreover, the prediction-free variant achieves successful hits with comparable performance relative to the target-known policy. In real-world tests, both the prediction and controller module exhibit high accuracy, and on-court hitting achieves an outgoing shuttle speed up to 10 m/s with a mean return landing distance of 3.5 m. These experiment results show that our humanoid robot can deliver highly dynamic while precise goal striking in badminton, and can be adapted to more dynamism critical domains.
>
---
#### [new 021] Latent-Space Autoregressive World Model for Efficient and Robust Image-Goal Navigation
- **分类: cs.RO**

- **简介: 论文提出LS-NWM解决图像目标导航的高计算成本问题，通过潜在空间自回归预测，将训练时间减3.2倍、规划时间减447倍，同时提升导航性能（SR+35%，SPL+11%）。**

- **链接: [https://arxiv.org/pdf/2511.11011v1](https://arxiv.org/pdf/2511.11011v1)**

> **作者:** Zhiwei Zhang; Hui Zhang; Xieyuanli Chen; Kaihong Huang; Chenghao Shi; Huimin Lu
>
> **摘要:** Traditional navigation methods rely heavily on accurate localization and mapping. In contrast, world models that capture environmental dynamics in latent space have opened up new perspectives for navigation tasks, enabling systems to move beyond traditional multi-module pipelines. However, world model often suffers from high computational costs in both training and inference. To address this, we propose LS-NWM - a lightweight latent space navigation world model that is trained and operates entirely in latent space, compared to the state-of-the-art baseline, our method reduces training time by approximately 3.2x and planning time by about 447x,while further improving navigation performance with a 35% higher SR and an 11% higher SPL. The key idea is that accurate pixel-wise environmental prediction is unnecessary for navigation. Instead, the model predicts future latent states based on current observational features and action inputs, then performs path planning and decision-making within this compact representation, significantly improving computational efficiency. By incorporating an autoregressive multi-frame prediction strategy during training, the model effectively captures long-term spatiotemporal dependencies, thereby enhancing navigation performance in complex scenarios. Experimental results demonstrate that our method achieves state-of-the-art navigation performance while maintaining a substantial efficiency advantage over existing approaches.
>
---
#### [new 022] Dexterous Manipulation Transfer via Progressive Kinematic-Dynamic Alignment
- **分类: cs.RO**

- **简介: 该论文解决机器人灵巧操纵数据稀缺问题，提出手无关转移系统。通过渐进式运动学-动力学对齐，将人类操作视频转换为高质量机器人轨迹，无需大量训练数据。实验平均成功率73%，实现高效通用数据收集。**

- **链接: [https://arxiv.org/pdf/2511.10987v1](https://arxiv.org/pdf/2511.10987v1)**

> **作者:** Wenbin Bai; Qiyu Chen; Xiangbo Lin; Jianwen Li; Quancheng Li; Hejiang Pan; Yi Sun
>
> **备注:** 13 pages, 15 figures. Accepted by AAAI 2026
>
> **摘要:** The inherent difficulty and limited scalability of collecting manipulation data using multi-fingered robot hand hardware platforms have resulted in severe data scarcity, impeding research on data-driven dexterous manipulation policy learning. To address this challenge, we present a hand-agnostic manipulation transfer system. It efficiently converts human hand manipulation sequences from demonstration videos into high-quality dexterous manipulation trajectories without requirements of massive training data. To tackle the multi-dimensional disparities between human hands and dexterous hands, as well as the challenges posed by high-degree-of-freedom coordinated control of dexterous hands, we design a progressive transfer framework: first, we establish primary control signals for dexterous hands based on kinematic matching; subsequently, we train residual policies with action space rescaling and thumb-guided initialization to dynamically optimize contact interactions under unified rewards; finally, we compute wrist control trajectories with the objective of preserving operational semantics. Using only human hand manipulation videos, our system automatically configures system parameters for different tasks, balancing kinematic matching and dynamic optimization across dexterous hands, object categories, and tasks. Extensive experimental results demonstrate that our framework can automatically generate smooth and semantically correct dexterous hand manipulation that faithfully reproduces human intentions, achieving high efficiency and strong generalizability with an average transfer success rate of 73%, providing an easily implementable and scalable method for collecting robot dexterous manipulation data.
>
---
#### [new 023] An Investigation into Dynamically Extensible and Retractable Robotic Leg Linkages for Multi-task Execution in Search and Rescue Scenarios
- **分类: cs.RO**

- **简介: 论文针对搜索和救援机器人，解决地形适应性与高力输出难以兼顾的问题。提出动态可伸缩五杆连杆腿设计，通过几何变换切换高度与力优势配置。实验验证了步长、力输出和稳定性，支持多任务执行。**

- **链接: [https://arxiv.org/pdf/2511.10816v1](https://arxiv.org/pdf/2511.10816v1)**

> **作者:** William Harris; Lucas Yager; Syler Sylvester; Elizabeth Peiros; Micheal C. Yip
>
> **摘要:** Search and rescue (SAR) robots are required to quickly traverse terrain and perform high-force rescue tasks, necessitating both terrain adaptability and controlled high-force output. Few platforms exist today for SAR, and fewer still have the ability to cover both tasks of terrain adaptability and high-force output when performing extraction. While legged robots offer significant ability to traverse uneven terrain, they typically are unable to incorporate mechanisms that provide variable high-force outputs, unlike traditional wheel-based drive trains. This work introduces a novel concept for a dynamically extensible and retractable robot leg. Leveraging a dynamically extensible and retractable five-bar linkage design, it allows for mechanically switching between height-advantaged and force-advantaged configurations via a geometric transformation. A testbed evaluated leg performance across linkage geometries and operating modes, with empirical and analytical analyses conducted on stride length, force output, and stability. The results demonstrate that the morphing leg offers a promising path toward SAR robots that can both navigate terrain quickly and perform rescue tasks effectively.
>
---
#### [new 024] From Framework to Reliable Practice: End-User Perspectives on Social Robots in Public Spaces
- **分类: cs.RO**

- **简介: 该论文研究社会机器人在公共空间的用户接受度，解决伦理、安全与可访问性问题。通过部署ARI接待机器人试点，收集35人反馈验证SecuRoPS框架，并开源GitHub模板支持可重复性与实践应用。**

- **链接: [https://arxiv.org/pdf/2511.10770v1](https://arxiv.org/pdf/2511.10770v1)**

> **作者:** Samson Oruma; Ricardo Colomo-Palacios; Vasileios Gkioulos
>
> **备注:** 26 pages, 3 figures
>
> **摘要:** As social robots increasingly enter public environments, their acceptance depends not only on technical reliability but also on ethical integrity, accessibility, and user trust. This paper reports on a pilot deployment of an ARI social robot functioning as a university receptionist, designed in alignment with the SecuRoPS framework for secure and ethical social robot deployment. Thirty-five students and staff interacted with the robot and provided structured feedback on safety, privacy, usability, accessibility, and transparency. The results show generally positive perceptions of physical safety, data protection, and ethical behavior, while also highlighting challenges related to accessibility, inclusiveness, and dynamic interaction. Beyond the empirical findings, the study demonstrates how theoretical frameworks for ethical and secure design can be implemented in real-world contexts through end-user evaluation. It also provides a public GitHub repository containing reusable templates for ARI robot applications to support reproducibility and lower the entry barrier for new researchers. By combining user perspectives with practical technical resources, this work contributes to ongoing discussions in AI and society and supports the development of trustworthy, inclusive, and ethically responsible social robots for public spaces.
>
---
#### [new 025] Dynamic Reconfiguration of Robotic Swarms: Coordination and Control for Precise Shape Formation
- **分类: cs.RO**

- **简介: 该论文解决机器人集群动态重新配置任务，针对移动路径优化问题。提出几何算法，通过控制、定位和映射技术实现无缝形状过渡，克服测量误差与控制动态挑战。**

- **链接: [https://arxiv.org/pdf/2511.10989v1](https://arxiv.org/pdf/2511.10989v1)**

> **作者:** Prab Prasertying; Paulo Garcia; Warisa Sritriratanarak
>
> **备注:** accepted at the 9th International Conference on Algorithms, Computing and Systems (ICACS 2025)
>
> **摘要:** Coordination of movement and configuration in robotic swarms is a challenging endeavor. Deciding when and where each individual robot must move is a computationally complex problem. The challenge is further exacerbated by difficulties inherent to physical systems, such as measurement error and control dynamics. Thus, how to best determine the optimal path for each robot, when moving from one configuration to another, and how to best perform such determination and effect corresponding motion remains an open problem. In this paper, we show an algorithm for such coordination of robotic swarms. Our methods allow seamless transition from one configuration to another, leveraging geometric formulations that are mapped to the physical domain through appropriate control, localization, and mapping techniques. This paves the way for novel applications of robotic swarms by enabling more sophisticated distributed behaviors.
>
---
#### [new 026] WetExplorer: Automating Wetland Greenhouse-Gas Surveys with an Autonomous Mobile Robot
- **分类: cs.RO; eess.SY**

- **简介: 论文开发WetExplorer自主机器人，解决湿地温室气体手动采样效率低问题。集成双RTK传感器融合与深度学习，实现厘米级定位精度，自动化采样流程，支持高频多点测量。**

- **链接: [https://arxiv.org/pdf/2511.10864v1](https://arxiv.org/pdf/2511.10864v1)**

> **作者:** Jose Vasquez; Xuping Zhang
>
> **备注:** To be published in 2025 IEEE International Conference on Robotics and Biomimetics
>
> **摘要:** Quantifying greenhouse-gases (GHG) in wetlands is critical for climate modeling and restoration assessment, yet manual sampling is labor-intensive, and time demanding. We present WetExplorer, an autonomous tracked robot that automates the full GHG-sampling workflow. The robot system integrates low-ground-pressure locomotion, centimeter-accurate lift placement, dual-RTK sensor fusion, obstacle avoidance planning, and deep-learning perception in a containerized ROS2 stack. Outdoor trials verified that the sensor-fusion stack maintains a mean localization error of 1.71 cm, the vision module estimates object pose with 7 mm translational and 3° rotational accuracy, while indoor trials demonstrated that the full motion-planning pipeline positions the sampling chamber within a global tolerance of 70 mm while avoiding obstacles, all without human intervention. By eliminating the manual bottleneck, WetExplorer enables high-frequency, multi-site GHG measurements and opens the door for dense, long-duration datasets in saturated wetland terrain.
>
---
#### [new 027] Drone Swarm Energy Management
- **分类: math.OC; cs.RO**

- **简介: 论文提出POMDP-DDPG融合框架解决无人机群在不确定性下的能源管理问题，通过贝叶斯滤波引入信念状态表示。仿真证明该方法显著提升任务成功率与能源效率，支持分布式决策协调。**

- **链接: [https://arxiv.org/pdf/2511.11557v1](https://arxiv.org/pdf/2511.11557v1)**

> **作者:** Michael Z. Zgurovsky; Pavlo O. Kasyanov; Liliia S. Paliichuk
>
> **备注:** 14 pages, 4 Tables, 2 Figures
>
> **摘要:** This note presents an analytical framework for decision-making in drone swarm systems operating under uncertainty, based on the integration of Partially Observable Markov Decision Processes (POMDP) with Deep Deterministic Policy Gradient (DDPG) reinforcement learning. The proposed approach enables adaptive control and cooperative behavior of unmanned aerial vehicles (UAVs) within a cognitive AI platform, where each agent learns optimal energy management and navigation policies from dynamic environmental states. We extend the standard DDPG architecture with a belief-state representation derived from Bayesian filtering, allowing for robust decision-making in partially observable environments. In this paper, for the Gaussian case, we numerically compare the performance of policies derived from DDPG to optimal policies for discretized versions of the original continuous problem. Simulation results demonstrate that the POMDP-DDPG-based swarm control model significantly improves mission success rates and energy efficiency compared to baseline methods. The developed framework supports distributed learning and decision coordination across multiple agents, providing a foundation for scalable cognitive swarm autonomy. The outcomes of this research contribute to the advancement of energy-aware control algorithms for intelligent multi-agent systems and can be applied in security, environmental monitoring, and infrastructure inspection scenarios.
>
---
#### [new 028] Phys-Liquid: A Physics-Informed Dataset for Estimating 3D Geometry and Volume of Transparent Deformable Liquids
- **分类: cs.CV; cs.RO**

- **简介: 论文针对透明液体3D几何与体积估计难题，提出Phys-Liquid数据集（含97,200个模拟图像及3D网格），覆盖多场景动态行为。通过四阶段重建管道验证，显著提升精度，助力机器人精准液体操作任务。**

- **链接: [https://arxiv.org/pdf/2511.11077v1](https://arxiv.org/pdf/2511.11077v1)**

> **作者:** Ke Ma; Yizhou Fang; Jean-Baptiste Weibel; Shuai Tan; Xinggang Wang; Yang Xiao; Yi Fang; Tian Xia
>
> **备注:** 14 pages, 19 figures. Accepted as an oral paper at AAAI-26 (Main Technical Track). Code and dataset: https://github.com/dualtransparency/Phys-Liquid-AAAI Project page: https://dualtransparency.github.io/Phys-Liquid/
>
> **摘要:** Estimating the geometric and volumetric properties of transparent deformable liquids is challenging due to optical complexities and dynamic surface deformations induced by container movements. Autonomous robots performing precise liquid manipulation tasks, such as dispensing, aspiration, and mixing, must handle containers in ways that inevitably induce these deformations, complicating accurate liquid state assessment. Current datasets lack comprehensive physics-informed simulation data representing realistic liquid behaviors under diverse dynamic scenarios. To bridge this gap, we introduce Phys-Liquid, a physics-informed dataset comprising 97,200 simulation images and corresponding 3D meshes, capturing liquid dynamics across multiple laboratory scenes, lighting conditions, liquid colors, and container rotations. To validate the realism and effectiveness of Phys-Liquid, we propose a four-stage reconstruction and estimation pipeline involving liquid segmentation, multi-view mask generation, 3D mesh reconstruction, and real-world scaling. Experimental results demonstrate improved accuracy and consistency in reconstructing liquid geometry and volume, outperforming existing benchmarks. The dataset and associated validation methods facilitate future advancements in transparent liquid perception tasks. The dataset and code are available at https://dualtransparency.github.io/Phys-Liquid/.
>
---
#### [new 029] Semantic VLM Dataset for Safe Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 论文提出CAR-Scenes数据集，用于自动驾驶视觉语言模型（VLMs）的训练与评估。标注5,192张图像，覆盖28类别、350+属性，支持场景级可解释理解与风险感知分析。提供属性共现图、基线模型及分析工具，推动安全自动驾驶研究。**

- **链接: [https://arxiv.org/pdf/2511.10701v1](https://arxiv.org/pdf/2511.10701v1)**

> **作者:** Yuankai He; Weisong Shi
>
> **备注:** 8 pages, 6 figures, 7 tables
>
> **摘要:** CAR-Scenes is a frame-level dataset for autonomous driving that enables training and evaluation of vision-language models (VLMs) for interpretable, scene-level understanding. We annotate 5,192 images drawn from Argoverse 1, Cityscapes, KITTI, and nuScenes using a 28-key category/sub-category knowledge base covering environment, road geometry, background-vehicle behavior, ego-vehicle behavior, vulnerable road users, sensor states, and a discrete severity scale (1-10), totaling 350+ leaf attributes. Labels are produced by a GPT-4o-assisted vision-language pipeline with human-in-the-loop verification; we release the exact prompts, post-processing rules, and per-field baseline model performance. CAR-Scenes also provides attribute co-occurrence graphs and JSONL records that support semantic retrieval, dataset triage, and risk-aware scenario mining across sources. To calibrate task difficulty, we include reproducible, non-benchmark baselines, notably a LoRA-tuned Qwen2-VL-2B with deterministic decoding, evaluated via scalar accuracy, micro-averaged F1 for list attributes, and severity MAE/RMSE on a fixed validation split. We publicly release the annotation and analysis scripts, including graph construction and evaluation scripts, to enable explainable, data-centric workflows for future intelligent vehicles. Dataset: https://github.com/Croquembouche/CAR-Scenes
>
---
#### [new 030] 6D Strawberry Pose Estimation: Real-time and Edge AI Solutions Using Purely Synthetic Training Data
- **分类: cs.CV; cs.RO**

- **简介: 该论文解决草莓6D姿态估计问题，利用纯合成数据训练YOLOX-6D-Pose模型，实现边缘设备实时推理。在RTX 3090和Jetson Orin Nano上验证，精度高，但未成熟草莓检测需优化。**

- **链接: [https://arxiv.org/pdf/2511.11307v1](https://arxiv.org/pdf/2511.11307v1)**

> **作者:** Saptarshi Neil Sinha; Julius Kühn; Mika Silvan Goschke; Michael Weinmann
>
> **摘要:** Automated and selective harvesting of fruits has become an important area of research, particularly due to challenges such as high costs and a shortage of seasonal labor in advanced economies. This paper focuses on 6D pose estimation of strawberries using purely synthetic data generated through a procedural pipeline for photorealistic rendering. We employ the YOLOX-6D-Pose algorithm, a single-shot approach that leverages the YOLOX backbone, known for its balance between speed and accuracy, and its support for edge inference. To address the lacking availability of training data, we introduce a robust and flexible pipeline for generating synthetic strawberry data from various 3D models via a procedural Blender pipeline, where we focus on enhancing the realism of the synthesized data in comparison to previous work to make it a valuable resource for training pose estimation algorithms. Quantitative evaluations indicate that our models achieve comparable accuracy on both the NVIDIA RTX 3090 and Jetson Orin Nano across several ADD-S metrics, with the RTX 3090 demonstrating superior processing speed. However, the Jetson Orin Nano is particularly suited for resource-constrained environments, making it an excellent choice for deployment in agricultural robotics. Qualitative assessments further confirm the model's performance, demonstrating its capability to accurately infer the poses of ripe and partially ripe strawberries, while facing challenges in detecting unripe specimens. This suggests opportunities for future improvements, especially in enhancing detection capabilities for unripe strawberries (if desired) by exploring variations in color. Furthermore, the methodology presented could be adapted easily for other fruits such as apples, peaches, and plums, thereby expanding its applicability and impact in the field of agricultural automation.
>
---
#### [new 031] RadAround: A Field-Expedient Direction Finder for Contested IoT Sensing & EM Situational Awareness
- **分类: eess.SP; cs.RO**

- **简介: RadAround是用于对抗性物联网传感的EM方向查找系统，解决EM态势感知问题。它通过机械天线和SCADA软件实时生成高分辨率EM热图，实现低成本EMI检测与设备定位，适用于战场和灾难响应。**

- **链接: [https://arxiv.org/pdf/2511.11392v1](https://arxiv.org/pdf/2511.11392v1)**

> **作者:** Owen A. Maute; Blake A. Roberts; Berker Peköz
>
> **备注:** 6 pages. Cite as O. Maute, B. A. Roberts, and B. Peköz, "RadAround: A field-expedient direction finder for contested IoT sensing & EM situational awareness," in Proc. 2025 IEEE Military Commun. Conf. (MILCOM), Los Angeles, USA, Oct. 2025, pp. 1-6
>
> **摘要:** This paper presents RadAround, a passive 2-D direction-finding system designed for adversarial IoT sensing in contested environments. Using mechanically steered narrow-beam antennas and field-deployable SCADA software, it generates high-resolution electromagnetic (EM) heatmaps using low-cost COTS or 3D-printed components. The microcontroller-deployable SCADA coordinates antenna positioning and SDR sampling in real time for resilient, on-site operation. Its modular design enables rapid adaptation for applications such as EMC testing in disaster-response deployments, battlefield spectrum monitoring, electronic intrusion detection, and tactical EM situational awareness (EMSA). Experiments show RadAround detecting computing machinery through walls, assessing utilization, and pinpointing EM interference (EMI) leakage sources from Faraday enclosures.
>
---
#### [new 032] Autonomous Vehicle Path Planning by Searching With Differentiable Simulation
- **分类: cs.AI; cs.RO**

- **简介: 论文提出DSS框架解决自主驾驶路径规划问题。利用可微分模拟器Waymax作为状态预测器和评论家，通过梯度下降优化动作序列，显著提升复杂交通场景下的规划精度，优于序列预测、模仿学习等方法。**

- **链接: [https://arxiv.org/pdf/2511.11043v1](https://arxiv.org/pdf/2511.11043v1)**

> **作者:** Asen Nachkov; Jan-Nico Zaech; Danda Pani Paudel; Xi Wang; Luc Van Gool
>
> **摘要:** Planning allows an agent to safely refine its actions before executing them in the real world. In autonomous driving, this is crucial to avoid collisions and navigate in complex, dense traffic scenarios. One way to plan is to search for the best action sequence. However, this is challenging when all necessary components - policy, next-state predictor, and critic - have to be learned. Here we propose Differentiable Simulation for Search (DSS), a framework that leverages the differentiable simulator Waymax as both a next state predictor and a critic. It relies on the simulator's hardcoded dynamics, making state predictions highly accurate, while utilizing the simulator's differentiability to effectively search across action sequences. Our DSS agent optimizes its actions using gradient descent over imagined future trajectories. We show experimentally that DSS - the combination of planning gradients and stochastic search - significantly improves tracking and path planning accuracy compared to sequence prediction, imitation learning, model-free RL, and other planning methods.
>
---
#### [new 033] DualVision ArthroNav: Investigating Opportunities to Enhance Localization and Reconstruction in Image-based Arthroscopy Navigation via External Cameras
- **分类: eess.IV; cs.CV; cs.RO**

- **简介: 论文提出DualVision ArthroNav系统，用于关节镜手术导航任务。解决纯视觉导航的漂移与尺度模糊问题，通过集成外部相机提供稳定定位，单目关节镜实现场景重建。实验验证平均轨迹误差1.09mm，注册误差2.16mm。**

- **链接: [https://arxiv.org/pdf/2511.10699v1](https://arxiv.org/pdf/2511.10699v1)**

> **作者:** Hongchao Shu; Lalithkumar Seenivasan; Mingxu Liu; Yunseo Hwang; Yu-Chun Ku; Jonathan Knopf; Alejandro Martin-Gomez; Mehran Armand; Mathias Unberath
>
> **摘要:** Arthroscopic procedures can greatly benefit from navigation systems that enhance spatial awareness, depth perception, and field of view. However, existing optical tracking solutions impose strict workspace constraints and disrupt surgical workflow. Vision-based alternatives, though less invasive, often rely solely on the monocular arthroscope camera, making them prone to drift, scale ambiguity, and sensitivity to rapid motion or occlusion. We propose DualVision ArthroNav, a multi-camera arthroscopy navigation system that integrates an external camera rigidly mounted on the arthroscope. The external camera provides stable visual odometry and absolute localization, while the monocular arthroscope video enables dense scene reconstruction. By combining these complementary views, our system resolves the scale ambiguity and long-term drift inherent in monocular SLAM and ensures robust relocalization. Experiments demonstrate that our system effectively compensates for calibration errors, achieving an average absolute trajectory error of 1.09 mm. The reconstructed scenes reach an average target registration error of 2.16 mm, with high visual fidelity (SSIM = 0.69, PSNR = 22.19). These results indicate that our system provides a practical and cost-efficient solution for arthroscopic navigation, bridging the gap between optical tracking and purely vision-based systems, and paving the way toward clinically deployable, fully vision-based arthroscopic guidance.
>
---
## 更新

#### [replaced 001] Harnessing Bounded-Support Evolution Strategies for Policy Refinement
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.09923v2](https://arxiv.org/pdf/2511.09923v2)**

> **作者:** Ethan Hirschowitz; Fabio Ramos
>
> **备注:** 10 pages, 6 figures, to be published in Australasian Conference on Robotics and Automation (ACRA 2025)
>
> **摘要:** Improving competent robot policies with on-policy RL is often hampered by noisy, low-signal gradients. We revisit Evolution Strategies (ES) as a policy-gradient proxy and localize exploration with bounded, antithetic triangular perturbations, suitable for policy refinement. We propose Triangular-Distribution ES (TD-ES) which pairs bounded triangular noise with a centered-rank finite-difference estimator to deliver stable, parallelizable, gradient-free updates. In a two-stage pipeline - PPO pretraining followed by TD-ES refinement - this preserves early sample efficiency while enabling robust late-stage gains. Across a suite of robotic manipulation tasks, TD-ES raises success rates by 26.5% relative to PPO and greatly reduces variance, offering a simple, compute-light path to reliable refinement.
>
---
#### [replaced 002] Sensory-Motor Control with Large Language Models via Iterative Policy Refinement
- **分类: cs.AI; cs.HC; cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2506.04867v3](https://arxiv.org/pdf/2506.04867v3)**

> **作者:** Jônata Tyska Carvalho; Stefano Nolfi
>
> **备注:** Article updated with results from gpt-oss:120b and gpt-oss:20b. 27 pages (13 pages are from appendix), 8 figures, 2 tables, code for experiments replication and supplementary material provided at https://github.com/jtyska/llm-robotics-article/
>
> **摘要:** We propose a method that enables large language models (LLMs) to control embodied agents through the generation of control policies that directly map continuous observation vectors to continuous action vectors. At the outset, the LLMs generate a control strategy based on a textual description of the agent, its environment, and the intended goal. This strategy is then iteratively refined through a learning process in which the LLMs are repeatedly prompted to improve the current strategy, using performance feedback and sensory-motor data collected during its evaluation. The method is validated on classic control tasks from the Gymnasium library and the inverted pendulum task from the MuJoCo library. The approach proves effective with relatively compact models such as GPT-oss:120b and Qwen2.5:72b. In most cases, it successfully identifies optimal or near-optimal solutions by integrating symbolic knowledge derived through reasoning with sub-symbolic sensory-motor data gathered as the agent interacts with its environment.
>
---
#### [replaced 003] Enhancing the NAO: Extending Capabilities of Legacy Robots for Long-Term Research
- **分类: cs.RO; cs.HC; eess.AS**

- **链接: [https://arxiv.org/pdf/2509.17760v2](https://arxiv.org/pdf/2509.17760v2)**

> **作者:** Austin Wilson; Sahar Kapasi; Zane Greene; Alexis E. Block
>
> **摘要:** Legacy (unsupported) robotic platforms often lose research utility when manufacturer support ends, preventing integration of modern sensing, speech, and interaction capabilities. We present the Enhanced NAO, a revitalized version of Aldebaran's NAO robot featuring upgraded beamforming microphones, RGB-D and thermal cameras, and additional compute resources in a fully self-contained package. This system combines cloud-based and local models for perception and dialogue, while preserving the NAO's expressive body and behaviors. In a pilot user study validating conversational performance, the Enhanced NAO delivered significantly higher conversational quality and elicited stronger user preference compared to the NAO AI Edition, without increasing response latency. The added visual and thermal sensing modalities established a foundation for future perception-driven interaction. Beyond this implementation, our framework provides a platform-agnostic strategy for extending the lifespan and research utility of legacy robots, ensuring they remain valuable tools for human-robot interaction.
>
---
#### [replaced 004] MSGNav: Unleashing the Power of Multi-modal 3D Scene Graph for Zero-Shot Embodied Navigation
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.10376v2](https://arxiv.org/pdf/2511.10376v2)**

> **作者:** Xun Huang; Shijia Zhao; Yunxiang Wang; Xin Lu; Wanfa Zhang; Rongsheng Qu; Weixin Li; Yunhong Wang; Chenglu Wen
>
> **备注:** 10 pages
>
> **摘要:** Embodied navigation is a fundamental capability for robotic agents operating. Real-world deployment requires open vocabulary generalization and low training overhead, motivating zero-shot methods rather than task-specific RL training. However, existing zero-shot methods that build explicit 3D scene graphs often compress rich visual observations into text-only relations, leading to high construction cost, irreversible loss of visual evidence, and constrained vocabularies. To address these limitations, we introduce the Multi-modal 3D Scene Graph (M3DSG), which preserves visual cues by replacing textual relation
>
---
#### [replaced 005] Large Language Model-assisted Autonomous Vehicle Recovery from Immobilization
- **分类: cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2510.26023v2](https://arxiv.org/pdf/2510.26023v2)**

> **作者:** Zhipeng Bao; Qianwen Li
>
> **备注:** 7 pages
>
> **摘要:** Despite significant advancements in recent decades, autonomous vehicles (AVs) continue to face challenges in navigating certain traffic scenarios where human drivers excel. In such situations, AVs often become immobilized, disrupting overall traffic flow. Current recovery solutions, such as remote intervention (which is costly and inefficient) and manual takeover (which excludes non-drivers and limits AV accessibility), are inadequate. This paper introduces StuckSolver, a novel Large Language Model (LLM) driven recovery framework that enables AVs to resolve immobilization scenarios through self-reasoning and/or passenger-guided decision-making. StuckSolver is designed as a plug-in add-on module that operates on top of the AV's existing perception-planning-control stack, requiring no modification to its internal architecture. Instead, it interfaces with standard sensor data streams to detect immobilization states, interpret environmental context, and generate high-level recovery commands that can be executed by the AV's native planner. We evaluate StuckSolver on the Bench2Drive benchmark and in custom-designed uncertainty scenarios. Results show that StuckSolver achieves near-state-of-the-art performance through autonomous self-reasoning alone and exhibits further improvements when passenger guidance is incorporated.
>
---
#### [replaced 006] Your Ride, Your Rules: Psychology and Cognition Enabled Automated Driving Systems
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2506.11842v3](https://arxiv.org/pdf/2506.11842v3)**

> **作者:** Zhipeng Bao; Qianwen Li
>
> **备注:** 32 pages, one colummns
>
> **摘要:** Despite rapid advances in autonomous driving technology, current autonomous vehicles (AVs) lack effective bidirectional human-machine communication, limiting their ability to personalize the riding experience and recover from uncertain or immobilized states. This limitation undermines occupant comfort and trust, potentially hindering the adoption of AV technologies. We propose PACE-ADS (Psychology and Cognition Enabled Automated Driving Systems), a human-centered autonomy framework enabling AVs to sense, interpret, and respond to both external traffic conditions and internal occupant states. PACE-ADS uses an agentic workflow where three foundation model agents collaborate: the Driver Agent interprets the external environment; the Psychologist Agent decodes passive psychological signals (e.g., EEG, heart rate, facial expressions) and active cognitive inputs (e.g., verbal commands); and the Coordinator Agent synthesizes these inputs to generate high-level decisions that enhance responsiveness and personalize the ride. PACE-ADS complements, rather than replaces, conventional AV modules. It operates at the semantic planning layer, while delegating low-level control to native systems. The framework activates only when changes in the rider's psychological state are detected or when occupant instructions are issued. It integrates into existing AV platforms with minimal adjustments, positioning PACE-ADS as a scalable enhancement. We evaluate it in closed-loop simulations across diverse traffic scenarios, including intersections, pedestrian interactions, work zones, and car-following. Results show improved ride comfort, dynamic behavioral adjustment, and safe recovery from edge-case scenarios via autonomous reasoning or rider input. PACE-ADS bridges the gap between technical autonomy and human-centered mobility.
>
---
#### [replaced 007] Towards Efficient Certification of Maritime Remote Operation Centers
- **分类: cs.CY; cs.RO**

- **链接: [https://arxiv.org/pdf/2508.00543v2](https://arxiv.org/pdf/2508.00543v2)**

> **作者:** Christian Neurohr; Marcel Saager; Lina Putze; Jan-Patrick Osterloh; Karina Rothemann; Hilko Wiards; Eckard Böde; Axel Hahn
>
> **摘要:** Additional automation being build into ships implies a shift of crew from ship to shore. However, automated ships still have to be monitored and, in some situations, controlled remotely. These tasks are carried out by human operators located in shore-based remote operation centers. In this work, we present a concept for a hazard database that supports the safeguarding and certification of such remote operation centers. The concept is based on a categorization of hazard sources which we derive from a generic functional architecture. A subsequent preliminary suitability analysis unveils which methods for hazard analysis and risk assessment can adequately fill this hazard database.
>
---
#### [replaced 008] EAST: Environment Aware Safe Tracking using Planning and Control Co-Design
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2310.01363v3](https://arxiv.org/pdf/2310.01363v3)**

> **作者:** Zhichao Li; Yinzhuang Yi; Zhuolin Niu; Nikolay Atanasov
>
> **摘要:** This paper considers the problem of autonomous mobile robot navigation in unknown environments with moving obstacles. We propose a new method to achieve environment-aware safe tracking (EAST) of robot motion plans that integrates an obstacle clearance cost for path planning, a convex reachable set for robot motion prediction, and safety constraints for dynamic obstacle avoidance. EAST adapts the motion of the robot according to the locally sensed environment geometry and dynamics, leading to fast motion in wide open areas and cautious behavior in narrow passages or near moving obstacles. Our control design uses a reference governor, a virtual dynamical system that guides the robot's motion and decouples the path tracking and safety objectives. While reference governor methods have been used for safe tracking control in static environments, our key contribution is an extension to dynamic environments using convex optimization with control barrier function (CBF) constraints. Thus, our work establishes a connection between reference governor techniques and CBF techniques for safe control in dynamic environments. We validate our approach in simulated and real-world environments, featuring complex obstacle configurations and natural dynamic obstacle motion.
>
---
#### [replaced 009] TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2508.19257v3](https://arxiv.org/pdf/2508.19257v3)**

> **作者:** Chenghao Liu; Jiachen Zhang; Chengxuan Li; Zhimu Zhou; Shixin Wu; Songfang Huang; Huiling Duan
>
> **备注:** Accepted to AAAI 2026. Camera-ready version
>
> **摘要:** Vision-Language-Action (VLA) models process visual inputs independently at each timestep, discarding valuable temporal information inherent in robotic manipulation tasks. This frame-by-frame processing makes models vulnerable to visual noise while ignoring the substantial coherence between consecutive frames in manipulation sequences. We propose Temporal Token Fusion (TTF), a training-free approach that intelligently integrates historical and current visual representations to enhance VLA inference quality. Our method employs dual-dimension detection combining efficient grayscale pixel difference analysis with attention-based semantic relevance assessment, enabling selective temporal token fusion through hard fusion strategies and keyframe anchoring to prevent error accumulation. Comprehensive experiments across LIBERO, SimplerEnv, and real robot tasks demonstrate consistent improvements: 4.0 percentage points average on LIBERO (72.4\% vs 68.4\% baseline), cross-environment validation on SimplerEnv (4.8\% relative improvement), and 8.7\% relative improvement on real robot tasks. Our approach proves model-agnostic, working across OpenVLA and VLA-Cache architectures. Notably, TTF reveals that selective Query matrix reuse in attention mechanisms enhances rather than compromises performance, suggesting promising directions for direct KQV matrix reuse strategies that achieve computational acceleration while improving task success rates.
>
---
#### [replaced 010] A Learning-Based Framework for Collision-Free Motion Planning
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2508.07502v2](https://arxiv.org/pdf/2508.07502v2)**

> **作者:** Mateus Salomão; Tianyü Ren; Alexander König
>
> **摘要:** This paper presents a learning-based extension to a Circular Field (CF)-based motion planner for efficient, collision-free trajectory generation in cluttered environments. The proposed approach overcomes the limitations of hand-tuned force field parameters by employing a deep neural network trained to infer optimal planner gains from a single depth image of the scene. The pipeline incorporates a CUDA-accelerated perception module, a predictive agent-based planning strategy, and a dataset generated through Bayesian optimization in simulation. The resulting framework enables real-time planning without manual parameter tuning and is validated both in simulation and on a Franka Emika Panda robot. Experimental results demonstrate successful task completion and improved generalization compared to classical planners.
>
---
#### [replaced 011] Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.07730v2](https://arxiv.org/pdf/2511.07730v2)**

> **作者:** Bill Chunyuan Zheng; Vivek Myers; Benjamin Eysenbach; Sergey Levine
>
> **摘要:** Learning how to reach goals in an environment is a longstanding challenge in AI, yet reasoning over long horizons remains a challenge for modern methods. The key question is how to estimate the temporal distance between pairs of observations. While temporal difference methods leverage local updates to provide optimality guarantees, they often perform worse than Monte Carlo methods that perform global updates (e.g., with multi-step returns), which lack such guarantees. We show how these approaches can be integrated into a practical GCRL method that fits a quasimetric distance using a multistep Monte-Carlo return. We show our method outperforms existing GCRL methods on long-horizon simulated tasks with up to 4000 steps, even with visual observations. We also demonstrate that our method can enable stitching in the real-world robotic manipulation domain (Bridge setup). Our approach is the first end-to-end GCRL method that enables multistep stitching in this real-world manipulation domain from an unlabeled offline dataset of visual observations.
>
---
#### [replaced 012] Leveraging Sidewalk Robots for Walkability-Related Analyses
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2507.12148v3](https://arxiv.org/pdf/2507.12148v3)**

> **作者:** Xing Tong; Michele D. Simoni; Kaj Munhoz Arfvidsson; Jonas Mårtensson
>
> **摘要:** Walkability is a key component of sustainable urban development. In walkability studies, collecting detailed pedestrian infrastructure data remains challenging due to the high costs and limited scalability of traditional methods. Sidewalk delivery robots, increasingly deployed in urban environments, offer a promising solution to these limitations. This paper explores how these robots can serve as mobile data collection platforms, capturing sidewalk-level features related to walkability in a scalable, automated, and real-time manner. A sensor-equipped robot was deployed on a sidewalk network at KTH in Stockholm, completing 101 trips covering 900 segment records. From the collected data, different typologies of features are derived, including robot trip characteristics (e.g., speed, duration), sidewalk conditions (e.g., width, surface unevenness), and sidewalk utilization (e.g., pedestrian density). Their walkability-related implications were investigated with a series of analyses. The results demonstrate that pedestrian movement patterns are strongly influenced by sidewalk characteristics, with higher density, reduced width, and surface irregularity associated with slower and more variable trajectories. Notably, robot speed closely mirrors pedestrian behavior, highlighting its potential as a proxy for assessing pedestrian dynamics. The proposed framework enables continuous monitoring of sidewalk conditions and pedestrian behavior, contributing to the development of more walkable, inclusive, and responsive urban environments.
>
---
#### [replaced 013] Zero-Shot Temporal Interaction Localization for Egocentric Videos
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2506.03662v4](https://arxiv.org/pdf/2506.03662v4)**

> **作者:** Erhang Zhang; Junyi Ma; Yin-Dong Zheng; Yixuan Zhou; Hesheng Wang
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Locating human-object interaction (HOI) actions within video serves as the foundation for multiple downstream tasks, such as human behavior analysis and human-robot skill transfer. Current temporal action localization methods typically rely on annotated action and object categories of interactions for optimization, which leads to domain bias and low deployment efficiency. Although some recent works have achieved zero-shot temporal action localization (ZS-TAL) with large vision-language models (VLMs), their coarse-grained estimations and open-loop pipelines hinder further performance improvements for temporal interaction localization (TIL). To address these issues, we propose a novel zero-shot TIL approach dubbed EgoLoc to locate the timings of grasp actions for human-object interaction in egocentric videos. EgoLoc introduces a self-adaptive sampling strategy to generate reasonable visual prompts for VLM reasoning. By absorbing both 2D and 3D observations, it directly samples high-quality initial guesses around the possible contact/separation timestamps of HOI according to 3D hand velocities, leading to high inference accuracy and efficiency. In addition, EgoLoc generates closed-loop feedback from visual and dynamic cues to further refine the localization results. Comprehensive experiments on the publicly available dataset and our newly proposed benchmark demonstrate that EgoLoc achieves better temporal interaction localization for egocentric videos compared to state-of-the-art baselines. We have released our code and relevant data as open-source at https://github.com/IRMVLab/EgoLoc.
>
---
#### [replaced 014] GELATO: Multi-Instruction Trajectory Reshaping via Geometry-Aware Multiagent-based Orchestration
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2509.06031v2](https://arxiv.org/pdf/2509.06031v2)**

> **作者:** Junhui Huang; Yuhe Gong; Changsheng Li; Xingguang Duan; Luis Figueredo
>
> **摘要:** We present GELATO -- the first language-driven trajectory reshaping framework to embed geometric environment awareness and multi-agent feedback orchestration to support multi-instruction in human-robot interaction scenarios. Unlike prior learning-based methods, our approach automatically registers scene objects as 6D geometric primitives via a VLM-assisted multi-view pipeline, and an LLM translates free-form multiple instructions into explicit, verifiable geometric constraints. These are integrated into a geometric-aware vector field optimization to adapt initial trajectories while preserving smoothness, feasibility, and clearance. We further introduce a multi-agent orchestration with observer-based refinement to handle multi-instruction inputs and interactions among objectives -- increasing success rate without retraining. Simulation and real-world experiments demonstrate our method achieves smoother, safer, and more interpretable trajectory modifications compared to state-of-the-art baselines.
>
---
#### [replaced 015] Efficient Learning-Based Control of a Legged Robot in Lunar Gravity
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2509.10128v2](https://arxiv.org/pdf/2509.10128v2)**

> **作者:** Philip Arm; Oliver Fischer; Joseph Church; Adrian Fuhrer; Hendrik Kolvenbach; Marco Hutter
>
> **摘要:** Legged robots are promising candidates for exploring challenging areas on low-gravity bodies such as the Moon, Mars, or asteroids, thanks to their advanced mobility on unstructured terrain. However, as planetary robots' power and thermal budgets are highly restricted, these robots need energy-efficient control approaches that easily transfer to multiple gravity environments. In this work, we introduce a reinforcement learning-based control approach for legged robots with gravity-scaled power-optimized reward functions. We use our approach to develop and validate a locomotion controller and a base pose controller in gravity environments from lunar gravity (1.62 m/s2) to a hypothetical super-Earth (19.62 m/s2). Our approach successfully scales across these gravity levels for locomotion and base pose control with the gravity-scaled reward functions. The power-optimized locomotion controller reached a power consumption for locomotion of 23.4 W in Earth gravity on a 15.65 kg robot at 0.4 m/s, a 23 % improvement over the baseline policy. Additionally, we designed a constant-force spring offload system that allowed us to conduct real-world experiments on legged locomotion in lunar gravity. In lunar gravity, the power-optimized control policy reached 12.2 W, 36 % less than a baseline controller which is not optimized for power efficiency. Our method provides a scalable approach to developing power-efficient locomotion controllers for legged robots across multiple gravity levels.
>
---
#### [replaced 016] Reconfigurable hydrostatics: Toward versatile and efficient load-bearing robotics
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2410.17936v2](https://arxiv.org/pdf/2410.17936v2)**

> **作者:** Jeff Denis; Frederic Laberge; Jean-Sebastien Plante; Alexandre Girard
>
> **摘要:** Wearable and legged robot designers face multiple challenges when choosing actuation. Traditional fully actuated designs using electric motors are multifunctional but oversized and inefficient for bearing conservative loads and for being backdrivable. Alternatively, quasi-passive and underactuated designs reduce the amount of motorization and energy storage, but are often designed for specific tasks. Designers of versatile and stronger wearable robots will face these challenges unless future actuators become very torque-dense, backdrivable and efficient This paper explores a design paradigm for addressing this issue: reconfigurable hydrostatics. We show that a hydrostatic actuator can integrate a passive force mechanism and a sharing mechanism in the fluid domain and still be multifunctional. First, an analytical study compares the effect of these two mechanisms on the motorization requirements in the context of a load-bearing exoskeleton. Then, the hydrostatic concept integrating these two mechanisms using hydraulic components is presented. A case study analysis shows the mass/efficiency/inertia benefits of the concept over a fully actuated one. Then, experiments are conducted on robotic legs to demonstrate that the actuator concept can meet the expected performance in terms of force tracking, versatility, and efficiency under controlled conditions. The proof-of-concept can track the vertical ground reaction force (GRF) profiles of walking, running, squatting, and jumping, and the energy consumption is 4.8x lower for walking. The transient force behaviors due to switching from one leg to the other are also analyzed along with some mitigation to improve them.
>
---
#### [replaced 017] The Temporal Trap: Entanglement in Pre-Trained Visual Representations for Visuomotor Policy Learning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2502.03270v3](https://arxiv.org/pdf/2502.03270v3)**

> **作者:** Nikolaos Tsagkas; Andreas Sochopoulos; Duolikun Danier; Chris Xiaoxuan Lu; Oisin Mac Aodha
>
> **备注:** This submission replaces our earlier work "When Pre-trained Visual Representations Fall Short: Limitations in Visuo-Motor Robot Learning." The original paper was split into two studies; this version focuses on temporal entanglement in pre-trained visual representations. The companion paper is "Attentive Feature Aggregation."
>
> **摘要:** The integration of pre-trained visual representations (PVRs) has significantly advanced visuomotor policy learning. However, effectively leveraging these models remains a challenge. We identify temporal entanglement as a critical, inherent issue when using these time-invariant models in sequential decision-making tasks. This entanglement arises because PVRs, optimised for static image understanding, struggle to represent the temporal dependencies crucial for visuomotor control. In this work, we quantify the impact of temporal entanglement, demonstrating a strong correlation between a policy's success rate and the ability of its latent space to capture task-progression cues. Based on these insights, we propose a simple, yet effective disentanglement baseline designed to mitigate temporal entanglement. Our empirical results show that traditional methods aimed at enriching features with temporal components are insufficient on their own, highlighting the necessity of explicitly addressing temporal disentanglement for robust visuomotor policy learning.
>
---
#### [replaced 018] Pelican-VL 1.0: A Foundation Brain Model for Embodied Intelligence
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.00108v2](https://arxiv.org/pdf/2511.00108v2)**

> **作者:** Yi Zhang; Che Liu; Xiancong Ren; Hanchu Ni; Shuai Zhang; Zeyuan Ding; Jiayu Hu; Hanzhe Shan; Zhenwei Niu; Zhaoyang Liu; Shuang Liu; Yue Zhao; Junbo Qi; Qinfan Zhang; Dengjie Li; Yidong Wang; Jiachen Luo; Yong Dai; Zenglin Xu; Bin Shen; Qifan Wang; Jian Tang; Xiaozhu Ju
>
> **摘要:** This report presents Pelican-VL 1.0, a new family of open-source embodied brain models with parameter scales ranging from 7 billion to 72 billion. Our explicit mission is clearly stated as: To embed powerful intelligence into various embodiments. Pelican-VL 1.0 is currently the largest-scale open-source embodied multimodal brain model. Its core advantage lies in the in-depth integration of data power and intelligent adaptive learning mechanisms. Specifically, metaloop distilled a high-quality dataset from a raw dataset containing 4+ billion tokens. Pelican-VL 1.0 is trained on a large-scale cluster of 1000+ A800 GPUs, consuming over 50k+ A800 GPU-hours per checkpoint. This translates to a 20.3% performance uplift from its base model and outperforms 100B-level open-source counterparts by 10.6%, placing it on par with leading proprietary systems on well-known embodied benchmarks. We establish a novel framework, DPPO (Deliberate Practice Policy Optimization), inspired by human metacognition to train Pelican-VL 1.0. We operationalize this as a metaloop that teaches the AI to practice deliberately, which is a RL-Refine-Diagnose-SFT loop.
>
---
#### [replaced 019] Fast Finite-Time Sliding Mode Control for Chattering-Free Trajectory Tracking of Robotic Manipulators
- **分类: eess.SY; cs.RO**

- **链接: [https://arxiv.org/pdf/2502.16867v3](https://arxiv.org/pdf/2502.16867v3)**

> **作者:** Momammad Ali Ranjbar
>
> **摘要:** Achieving precise and efficient trajectory tracking in robotic arms remains a key challenge due to system uncertainties and chattering effects in conventional sliding mode control (SMC). This paper presents a chattering-free fast terminal sliding mode control (FTSMC) strategy for a three-degree-of-freedom (3-DOF) robotic arm, designed to enhance tracking accuracy and robustness while ensuring finite-time convergence. The control framework is developed using Newton-Euler dynamics, followed by a state-space representation that captures the system's angular position and velocity. By incorporating an improved sliding surface and a Lyapunov-based stability analysis, the proposed FTSMC effectively mitigates chattering while preserving the advantages of SMC, such as fast response and strong disturbance rejection. The controller's performance is rigorously evaluated through comparisons with conventional PD sliding mode control (PDSMC) and terminal sliding mode control (TSMC). Simulation results demonstrate that the proposed approach achieves superior trajectory tracking performance, faster convergence, and enhanced stability compared to existing methods, making it a promising solution for high-precision robotic applications.
>
---
#### [replaced 020] Decoupling Torque and Stiffness: A Unified Modeling and Control Framework for Antagonistic Artificial Muscles
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.09104v2](https://arxiv.org/pdf/2511.09104v2)**

> **作者:** Amirhossein Kazemipour; Robert K. Katzschmann
>
> **摘要:** Antagonistic soft actuators built from artificial muscles (PAMs, HASELs, DEAs) promise plant-level torque-stiffness decoupling, yet existing controllers for soft muscles struggle to maintain independent control through dynamic contact transients. We present a unified framework enabling independent torque and stiffness commands in real-time for diverse soft actuator types. Our unified force law captures diverse soft muscle physics in a single model with sub-ms computation, while our cascaded controller with analytical inverse dynamics maintains decoupling despite model errors and disturbances. Using co-contraction/bias coordinates, the controller independently modulates torque via bias and stiffness via co-contraction-replicating biological impedance strategies. Simulation-based validation through contact experiments demonstrates maintained independence: 200x faster settling on soft surfaces, 81% force reduction on rigid surfaces, and stable interaction vs 22-54% stability for fixed policies. This framework provides a foundation for enabling musculoskeletal antagonistic systems to execute adaptive impedance control for safe human-robot interaction.
>
---
#### [replaced 021] A Humanoid Visual-Tactile-Action Dataset for Contact-Rich Manipulation
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2510.25725v2](https://arxiv.org/pdf/2510.25725v2)**

> **作者:** Eunju Kwon; Seungwon Oh; In-Chang Baek; Yucheon Park; Gyungbo Kim; JaeYoung Moon; Yunho Choi; Kyung-Joong Kim
>
> **摘要:** Contact-rich manipulation has become increasingly important in robot learning. However, previous studies on robot learning datasets have focused on rigid objects and underrepresented the diversity of pressure conditions for real-world manipulation. To address this gap, we present a humanoid visual-tactile-action dataset designed for manipulating deformable soft objects. The dataset was collected via teleoperation using a humanoid robot equipped with dexterous hands, capturing multi-modal interactions under varying pressure conditions. This work also motivates future research on models with advanced optimization strategies capable of effectively leveraging the complexity and diversity of tactile signals.
>
---
#### [replaced 022] Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.08086v2](https://arxiv.org/pdf/2511.08086v2)**

> **作者:** Muthukumar Pandaram; Jakob Hollenstein; David Drexel; Samuele Tosatto; Antonio Rodríguez-Sánchez; Justus Piater
>
> **摘要:** The use of learned dynamics models, also known as world models, can improve the sample efficiency of reinforcement learning. Recent work suggests that the underlying causal graphs of such dynamics models are sparsely connected, with each of the future state variables depending only on a small subset of the current state variables, and that learning may therefore benefit from sparsity priors. Similarly, temporal sparsity, i.e. sparsely and abruptly changing local dynamics, has also been proposed as a useful inductive bias. In this work, we critically examine these assumptions by analyzing ground-truth dynamics from a set of robotic reinforcement learning environments in the MuJoCo Playground benchmark suite, aiming to determine whether the proposed notions of state and temporal sparsity actually tend to hold in typical reinforcement learning tasks. We study (i) whether the causal graphs of environment dynamics are sparse, (ii) whether such sparsity is state-dependent, and (iii) whether local system dynamics change sparsely. Our results indicate that global sparsity is rare, but instead the tasks show local, state-dependent sparsity in their dynamics and this sparsity exhibits distinct structures, appearing in temporally localized clusters (e.g., during contact events) and affecting specific subsets of state dimensions. These findings challenge common sparsity prior assumptions in dynamics learning, emphasizing the need for grounded inductive biases that reflect the state-dependent sparsity structure of real-world dynamics.
>
---
#### [replaced 023] Convergent Functions, Divergent Forms
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2505.21665v2](https://arxiv.org/pdf/2505.21665v2)**

> **作者:** Hyeonseong Jeon; Ainaz Eftekhar; Aaron Walsman; Kuo-Hao Zeng; Ali Farhadi; Ranjay Krishna
>
> **摘要:** We introduce LOKI, a compute-efficient framework for co-designing morphologies and control policies that generalize across unseen tasks. Inspired by biological adaptation -- where animals quickly adjust to morphological changes -- our method overcomes the inefficiencies of traditional evolutionary and quality-diversity algorithms. We propose learning convergent functions: shared control policies trained across clusters of morphologically similar designs in a learned latent space, drastically reducing the training cost per design. Simultaneously, we promote divergent forms by replacing mutation with dynamic local search, enabling broader exploration and preventing premature convergence. The policy reuse allows us to explore 780$\times$ more designs using 78% fewer simulation steps and 40% less compute per design. Local competition paired with a broader search results in a diverse set of high-performing final morphologies. Using the UNIMAL design space and a flat-terrain locomotion task, LOKI discovers a rich variety of designs -- ranging from quadrupeds to crabs, bipedals, and spinners -- far more diverse than those produced by prior work. These morphologies also transfer better to unseen downstream tasks in agility, stability, and manipulation domains (e.g., 2$\times$ higher reward on bump and push box incline tasks). Overall, our approach produces designs that are both diverse and adaptable, with substantially greater sample efficiency than existing co-design methods. (Project website: https://loki-codesign.github.io/)
>
---
#### [replaced 024] DiAReL: Reinforcement Learning with Disturbance Awareness for Robust Sim2Real Policy Transfer in Robot Control
- **分类: cs.RO; cs.LG; eess.SY**

- **链接: [https://arxiv.org/pdf/2306.09010v2](https://arxiv.org/pdf/2306.09010v2)**

> **作者:** Mohammadhossein Malmir; Josip Josifovski; Noah Klarmann; Alois Knoll
>
> **备注:** Accepted for publication in IEEE Transactions on Control Systems Technology (TCST)
>
> **摘要:** Delayed Markov decision processes (DMDPs) fulfill the Markov property by augmenting the state space of agents with a finite time window of recently committed actions. In reliance on these state augmentations, delay-resolved reinforcement learning algorithms train policies to learn optimal interactions with environments featuring observation or action delays. Although such methods can be directly trained on the real robots, due to sample inefficiency, limited resources, or safety constraints, a common approach is to transfer models trained in simulation to the physical robot. However, robotic simulations rely on approximated models of the physical systems, which hinders the sim2real transfer. In this work, we consider various uncertainties in modeling the robot or environment dynamics as unknown intrinsic disturbances applied to the system input. We introduce the disturbance-augmented Markov decision process (DAMDP) in delayed settings as a novel representation to incorporate disturbance estimation in training on-policy reinforcement learning algorithms. The proposed method is validated across several metrics on learning robotic reaching and pushing tasks and compared with disturbance-unaware baselines. The results show that the disturbance-augmented models can achieve higher stabilization and robustness in the control response, which in turn improves the prospects of successful sim2real transfer.
>
---
#### [replaced 025] Optimal Modified Feedback Strategies in LQ Games under Control Imperfections
- **分类: cs.GT; cs.MA; cs.RO; eess.SY; math.OC**

- **链接: [https://arxiv.org/pdf/2503.19200v3](https://arxiv.org/pdf/2503.19200v3)**

> **作者:** Mahdis Rabbani; Navid Mojahed; Shima Nazari
>
> **备注:** 6 pages, 2 figures, Preprint version of a paper submitted to ACC 2026
>
> **摘要:** Game-theoretic approaches and Nash equilibrium have been widely applied across various engineering domains. However, practical challenges such as disturbances, delays, and actuator limitations can hinder the precise execution of Nash equilibrium strategies. This work investigates the impact of such implementation imperfections on game trajectories and players' costs in the context of a two-player finite-horizon linear quadratic (LQ) nonzero-sum game. Specifically, we analyze how small deviations by one player, measured or estimated at each stage, affect the state and cost function of the other player. To mitigate these effects, we propose an adjusted control policy that optimally compensates for the deviations under the stated information structure and can, under certain conditions, exploit them to improve performance. Rigorous mathematical analysis and proofs are provided, and the effectiveness of the proposed method is demonstrated through a representative numerical example.
>
---
