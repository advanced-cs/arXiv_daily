# 机器人 cs.RO

- **最新发布 83 篇**

- **更新 69 篇**

## 最新发布

#### [new 001] CounterScene: Counterfactual Causal Reasoning in Generative World Models for Safety-Critical Closed-Loop Evaluation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出CounterScene，用于安全关键场景生成，解决传统方法在真实性和对抗性间的权衡问题。通过因果推理和结构化干预，提升场景安全性与真实性。**

- **链接: [https://arxiv.org/pdf/2603.21104](https://arxiv.org/pdf/2603.21104)**

> **作者:** Bowen Jing; Ruiyang Hao; Weitao Zhou; Haibao Yu
>
> **备注:** 28 pages, 7 figures
>
> **摘要:** Generating safety-critical driving scenarios requires understanding why dangerous interactions arise, rather than merely forcing collisions. However, existing methods rely on heuristic adversarial agent selection and unstructured perturbations, lacking explicit modeling of interaction dependencies and thus exhibiting a realism--adversarial trade-off. We present CounterScene, a framework that endows closed-loop generative BEV world models with structured counterfactual reasoning for safety-critical scenario generation. Given a safe scene, CounterScene asks: what if the causally critical agent had behaved differently? To answer this, we introduce causal adversarial agent identification to identify the critical agent and classify conflict types, and develop a conflict-aware interactive world model in which a causal interaction graph is used to explicitly model dynamic inter-agent dependencies. Building on this structure, stage-adaptive counterfactual guidance performs minimal interventions on the identified agent, removing its spatial and temporal safety margins while allowing risk to emerge through natural interaction propagation. Extensive experiments on nuScenes demonstrate that CounterScene achieves the strongest adversarial effectiveness while maintaining superior trajectory realism across all horizons, improving long-horizon collision rate from 12.3% to 22.7% over the strongest baseline with better realism (ADE 1.88 vs.2.09). Notably, this advantage further widens over longer rollouts, and CounterScene generalizes zero-shot to nuPlan with state-of-the-art realism.
>
---
#### [new 002] Affordance-Guided Enveloping Grasp Demonstration Toward Non-destructive Disassembly of Pinch-Infeasible Mating Parts
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决非夹持型配合部件的拆卸问题。通过生成抓取模板并可视化增强操作者感知，提升复杂环境下抓取策略的 teaching 效率。**

- **链接: [https://arxiv.org/pdf/2603.21143](https://arxiv.org/pdf/2603.21143)**

> **作者:** Masaki Tsutsumi; Takuya Kiyokawa; Gen Sako; Kensuke Harada
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** Robotic disassembly of complex mating components often renders pinch grasping infeasible, necessitating multi-fingered enveloping grasps. However, visual occlusions and geometric constraints complicate teaching appropriate grasp motions when relying solely on 2D camera feeds. To address this, we propose an affordance-guided teleoperation method that pre-generates enveloping grasp candidates via physics simulation. These Affordance Templates (ATs) are visualized with a color gradient reflecting grasp quality to augment operator perception. Simulations demonstrate the method's generality across various components. Real-robot experiments validate that AT-based visual augmentation enables operators to effectively select and teach enveloping grasp strategies for real-world disassembly, even under severe visual and geometric constraints.
>
---
#### [new 003] EnergyAction: Unimanual to Bimanual Composition with Energy-Based Models
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决从单臂到双臂操作的迁移问题。通过能量模型，组合单臂策略并优化双臂动作协调，提升双臂任务性能。**

- **链接: [https://arxiv.org/pdf/2603.20236](https://arxiv.org/pdf/2603.20236)**

> **作者:** Mingchen Song; Xiang Deng; Jie Wei; Dongmei Jiang; Liqiang Nie; Weili Guan
>
> **摘要:** Recent advances in unimanual manipulation policies have achieved remarkable success across diverse robotic tasks through abundant training data and well-established model architectures. However, extending these capabilities to bimanual manipulation remains challenging due to the lack of bimanual demonstration data and the complexity of coordinating dual-arm actions. Existing approaches either rely on extensive bimanual datasets or fail to effectively leverage pre-trained unimanual policies. To address this limitation, we propose \textbf{EnergyAction}, a novel framework that compositionally transfers unimanual manipulation policies to bimanual tasks through the Energy-Based Models (EBMs). Specifically, our method incorporates three key innovations. First, we model individual unimanual policies as EBMs and leverage their compositional properties to compose left and right arm actions, enabling the fusion of unimanual policies into a bimanual policy. Second, we introduce an energy-based temporal-spatial coordination mechanism through energy constraints, ensuring the generated bimanual actions are both temporal coherence and spatial feasibility. Third, we propose two different energy-aware denoising strategies that dynamically adapt denoising steps based on action quality assessment. These strategies ensure the generation of high-quality actions while maintaining superior computational efficiency compared to fixed-step denoising approaches. Experimental results demonstrate that EnergyAction effectively transfers unimanual knowledge to bimanual tasks, achieving superior performance on both simulated and real-world tasks with minimal bimanual data.
>
---
#### [new 004] HyReach: Vision-Guided Hybrid Manipulator Reaching in Unseen Cluttered Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人抓取任务，解决在杂乱环境中精准抓取的问题。提出HyReach系统，结合视觉感知与柔性机械臂，实现无环境依赖的实时路径规划与控制。**

- **链接: [https://arxiv.org/pdf/2603.21421](https://arxiv.org/pdf/2603.21421)**

> **作者:** Shivani Kamtikar; Kendall Koe; Justin Wasserman; Samhita Marri; Benjamin Walt; Naveen Kumar Uppalapati; Girish Krishnan; Girish Chowdhary
>
> **备注:** 8 pages, 5 figures, 5 tables
>
> **摘要:** As robotic systems increasingly operate in unstructured, cluttered, and previously unseen environments, there is a growing need for manipulators that combine compliance, adaptability, and precise control. This work presents a real-time hybrid rigid-soft continuum manipulator system designed for robust open-world object reaching in such challenging environments. The system integrates vision-based perception and 3D scene reconstruction with shape-aware motion planning to generate safe trajectories. A learning-based controller drives the hybrid arm to arbitrary target poses, leveraging the flexibility of the soft segment while maintaining the precision of the rigid segment. The system operates without environment-specific retraining, enabling direct generalization to new scenes. Extensive real-world experiments demonstrate consistent reaching performance with errors below 2 cm across diverse cluttered setups, highlighting the potential of hybrid manipulators for adaptive and reliable operation in unstructured environments.
>
---
#### [new 005] StageCraft: Execution Aware Mitigation of Distractor and Obstruction Failures in VLA Models
- **分类: cs.RO**

- **简介: 该论文提出StageCraft，用于解决VLA模型在执行时因干扰物和障碍物导致的失败问题。通过利用大视觉语言模型的推理能力，调整初始环境状态以提升策略性能。**

- **链接: [https://arxiv.org/pdf/2603.20659](https://arxiv.org/pdf/2603.20659)**

> **作者:** Kartikay Milind Pangaonkar; Prabin Rath; Omkar Patil; Nakul Gopalan
>
> **摘要:** Large scale pre-training on text and image data along with diverse robot demonstrations has helped Vision Language Action models (VLAs) to generalize to novel tasks, objects and scenes. However, these models are still susceptible to failure in the presence of execution-time impediments such as distractors and physical obstructions in the robot's workspace. Existing policy improvement methods finetune base VLAs to improve generalization, yet they still struggle in unseen distractor settings. To address this problem, we investigate whether internet-scale pretraining of large vision-language models (VLMs) can be leveraged to reason about these impediments and mitigate policy failures. To this end, we propose StageCraft, a training-free approach to improve pretrained VLA policy performance by manipulating the environment's initial state using VLM-based in-context reasoning. StageCraft takes policy rollout videos and success labels as input and leverages VLM's reasoning ability to infer which objects in the initial state need to be manipulated to avoid anticipated execution failures. StageCraft is an extensible plug-and-play module that does not introduce additional constraints on the underlying policy, and only requires a few policy rollouts to work. We evaluate performance of state-of-the-art VLA models with StageCraft and show an absolute 40% performance improvement across three real world task domains involving diverse distractors and obstructions. Our simulation experiments in RLBench empirically show that StageCraft tailors its extent of intervention based on the strength of the underlying policy and improves its performance with more in-context samples. Videos of StageCraft in effect can be found at this https URL .
>
---
#### [new 006] Closed-Loop Verbal Reinforcement Learning for Task-Level Robotic Planning
- **分类: cs.RO**

- **简介: 该论文提出一种闭环语言强化学习框架，用于移动机器人在执行不确定性下的任务规划。解决传统强化学习缺乏解释性的问题，通过语言模型迭代优化行为树，实现可解释的策略改进与适应。**

- **链接: [https://arxiv.org/pdf/2603.22169](https://arxiv.org/pdf/2603.22169)**

> **作者:** Dmitrii Plotnikov; Iaroslav Kolomiets; Dmitrii Maliukov; Dmitrij Kosenkov; Daniia Zinniatullina; Artem Trandofilov; Georgii Gazaryan; Kirill Bogatikov; Timofei Kozlov; Igor Duchinskii; Mikhail Konenkov; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **摘要:** We propose a new Verbal Reinforcement Learning (VRL) framework for interpretable task-level planning in mobile robotic systems operating under execution uncertainty. The framework follows a closed-loop architecture that enables iterative policy improvement through interaction with the physical environment. In our framework, executable Behavior Trees are repeatedly refined by a Large Language Model actor using structured natural-language feedback produced by a Vision-Language Model critic that observes the physical robot and execution traces. Unlike conventional reinforcement learning, policy updates in VRL occur directly at the symbolic planning level, without gradient-based optimization. This enables transparent reasoning, explicit causal feedback, and human-interpretable policy evolution. We validate the proposed framework on a real mobile robot performing a multi-stage manipulation and navigation task under execution uncertainty. Experimental results show that the framework supports explainable policy improvements, closed-loop adaptation to execution failures, and reliable deployment on physical robotic systems.
>
---
#### [new 007] ToFormer: Towards Large-scale Scenario Depth Completion for Lightweight ToF Camera
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于深度补全任务，解决短距离ToF相机在大场景中的应用问题。构建了首个大规模ToF深度补全数据集LASER-ToF，并提出轻量级网络提升补全精度与实时性。**

- **链接: [https://arxiv.org/pdf/2603.20669](https://arxiv.org/pdf/2603.20669)**

> **作者:** Juncheng Chen; Tiancheng Lai; Xingpeng Wang; Bingxin Liao; Baozhe Zhang; Chao Xu; Yanjun Cao
>
> **备注:** 17 pages, 15 figures
>
> **摘要:** Time-of-Flight (ToF) cameras possess compact design and high measurement precision to be applied to various robot tasks. However, their limited sensing range restricts deployment in large-scale scenarios. Depth completion has emerged as a potential solution to expand the sensing range of ToF cameras, but existing research lacks dedicated datasets and struggles to generalize to ToF measurements. In this paper, we propose a full-stack framework that enables depth completion in large-scale scenarios for short-range ToF cameras. First, we construct a multi-sensor platform with a reconstruction-based pipeline to collect real-world ToF samples with dense large-scale ground truth, yielding the first LArge-ScalE scenaRio ToF depth completion dataset (LASER-ToF). Second, we propose a sensor-aware depth completion network that incorporates a novel 3D branch with a 3D-2D Joint Propagation Pooling (JPP) module and Multimodal Cross-Covariance Attention (MXCA), enabling effective modeling of long-range relationships and efficient 3D-2D fusion under non-uniform ToF depth sparsity. Moreover, our network can utilize the sparse point cloud from visual SLAM as a supplement to ToF depth to further improve prediction accuracy. Experiments show that our method achieves an 8.6% lower mean absolute error than the second-best method, while maintaining lightweight design to support onboard deployment. Finally, to verify the system's applicability on real robots, we deploy proposed method on a quadrotor at a 10Hz runtime, enabling reliable large-scale mapping and long-range planning in challenging environments for short-range ToF cameras.
>
---
#### [new 008] Geometrically Plausible Object Pose Refinement using Differentiable Simulation
- **分类: cs.RO**

- **简介: 该论文属于物体位姿估计任务，解决姿态估计中几何不可行的问题。通过结合物理模拟、渲染和触觉传感，优化位姿以提高准确性和物理一致性。**

- **链接: [https://arxiv.org/pdf/2603.20992](https://arxiv.org/pdf/2603.20992)**

> **作者:** Anil Zeybek; Rhys Newbury; Snehal Dikhale; Nawid Jamali; Soshi Iba; Akansel Cosgun
>
> **摘要:** State-of-the-art object pose estimation methods are prone to generating geometrically infeasible pose hypotheses. This problem is prevalent in dexterous manipulation, where estimated poses often intersect with the robotic hand or are not lying on a support surface. We propose a multi-modal pose refinement approach that combines differentiable physics simulation, differentiable rendering and visuo-tactile sensing to optimize object poses for both spatial accuracy and physical consistency. Simulated experiments show that our approach reduces the intersection volume error between the object and robotic hand by 73\% when the initial estimate is accurate and by over 87\% under high initial uncertainty, significantly outperforming standard ICP-based baselines. Furthermore, the improvement in geometric plausibility is accompanied by a concurrent reduction in translation and orientation errors. Achieving pose estimation that is grounded in physical reality while remaining faithful to multi-modal sensor inputs is a critical step toward robust in-hand manipulation.
>
---
#### [new 009] Disengagement Analysis and Field Tests of a Prototypical Open-Source Level 4 Autonomous Driving System
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶系统评估任务，旨在分析开源Level 4系统的可靠性。通过实测与分类，发现系统在复杂交通中的不足及干预原因。**

- **链接: [https://arxiv.org/pdf/2603.21926](https://arxiv.org/pdf/2603.21926)**

> **作者:** Marvin Seegert; Christian Oefinger; Korbinian Moller; Christoph Bank; Johannes Betz
>
> **备注:** 8 pages, submitted to IEEE for possible publication
>
> **摘要:** Proprietary Autonomous Driving Systems are typically evaluated through disengagements, unplanned manual interventions to alter vehicle behavior, as annually reported by the California Department of Motor Vehicles. However, the real-world capabilities of prototypical open-source Level 4 vehicles over substantial distances remain largely unexplored. This study evaluates a research vehicle running an Autoware-based software stack across 236 km of mixed traffic. By classifying 30 disengagements across 26 rides with a novel five-level criticality framework, we observed a spatial disengagement rate of 0.127 1/km. Interventions predominantly occurred at lower speeds near static objects and traffic lights. Perception and Planning failures accounted for 40% and 26.7% of disengagements, respectively, largely due to object-tracking losses and operational deadlocks caused by parked vehicles. Frequent, unnecessary interventions highlighted a lack of trust on the part of the safety driver. These results show that while open-source software enables extensive operations, disengagement analysis is vital for uncovering robustness issues missed by standard metrics.
>
---
#### [new 010] Unified Orbit-Attitude Estimation and Sensor Tasking Framework for Autonomous Cislunar Space Domain Awareness Using Multiplicative Unscented Kalman Filter
- **分类: cs.RO; physics.space-ph**

- **简介: 该论文属于自主月球轨道域感知任务，解决非开普勒动力学下的轨道与姿态估计及传感器调度问题，提出基于乘法无迹卡尔曼滤波的优化框架。**

- **链接: [https://arxiv.org/pdf/2603.20579](https://arxiv.org/pdf/2603.20579)**

> **作者:** Smriti Nandan Paul; Siwei Fan
>
> **摘要:** The cislunar regime departs from near-Earth orbital behavior through strongly non-linear, non-Keplerian dynamics, which adversely affect the accuracy of uncertainty propagation and state estimation. Additional challenges arise from long-range observation requirements, restrictive sensor-target geometry and illumination conditions, the need to monitor an expansive cislunar volume, and the large design space associated with space/ground-based sensor placement. In response to these challenges, this work introduces an advanced framework for cislunar space domain awareness (SDA) encompassing two key tasks: (1) observer architecture optimization based on a realistic cost formulation that captures key performance trade-offs, solved using the Tree of Parzen Estimators algorithm, and (2) leveraging the resulting observer architecture, a mutual information-driven sensor tasking optimization is performed at discrete tasking intervals, while orbital and attitude state estimation is carried out at a finer temporal resolution between successive tasking updates using an error-state multiplicative unscented Kalman filter. Numerical simulations demonstrate that our approach in Task 1 yields observer architectures that achieve significantly lower values of the proposed cost function than baseline random-search solutions, while using fewer sensors. Task 2 results show that translational state estimation remains satisfactory over a wide range of target-to-observer count ratios, whereas attitude estimation is significantly more sensitive to target-to-observer ratios and tasking intervals, with increased rotational-state divergence observed for high target counts and infrequent tasking updates. These results highlight important trade-offs between sensing resources, tasking cadence, and achievable state estimation performance that influence the scalability of autonomous cislunar SDA.
>
---
#### [new 011] Fusing Driver Perceived and Physical Risk for Safety Critical Scenario Screening in Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于自动驾驶安全场景筛选任务，旨在解决人工标注效率低、风险量化不准确的问题。提出一种融合驾驶员感知与物理风险的方法，提升场景风险评估的效率和准确性。**

- **链接: [https://arxiv.org/pdf/2603.20232](https://arxiv.org/pdf/2603.20232)**

> **作者:** Chen Xiong; Ziwen Wang; Deqi Wang; Cheng Wang; Yiyang Chen; He Zhang; Chao Gou
>
> **摘要:** Autonomous driving testing increasingly relies on mining safety critical scenarios from large scale naturalistic driving data, yet existing screening pipelines still depend on manual risk annotation and expensive frame by frame risk evaluation, resulting in low efficiency and weakly grounded risk quantification. To address this issue, we propose a driver risk fusion based hazardous scenario screening method for autonomous driving. During training, the method combines an improved Driver Risk Field with a dynamic cost model to generate high quality risk supervision signals, while during inference it directly predicts scenario level risk scores through fast forward passes, avoiding per frame risk computation and enabling efficient large scale ranking and retrieval. The improved Driver Risk Field introduces a new risk height function and a speed adaptive look ahead mechanism, and the dynamic cost model integrates kinetic energy, oriented bounding box constraints, and Gaussian kernel diffusion smoothing for more accurate interaction modeling. We further design a risk trajectory cross attention decoder to jointly decode risk and trajectories. Experiments on the INTERACTION and FLUID datasets show that the proposed method produces smoother and more discriminative risk estimates. On FLUID, it achieves an AUC of 0.792 and an AP of 0.825, outperforming PODAR by 9.1 percent and 5.1 percent, respectively, demonstrating its effectiveness for scalable risk labeling and hazardous scenario screening.
>
---
#### [new 012] LASER: Level-Based Asynchronous Scheduling and Execution Regime for Spatiotemporally Constrained Multi-Robot Timber Manufacturing
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出LASER框架，解决多机器人在木材制造中的时空约束调度问题，通过分层异步执行实现高效协同作业。**

- **链接: [https://arxiv.org/pdf/2603.20577](https://arxiv.org/pdf/2603.20577)**

> **作者:** Zhenxiang Huang; Lior Skoury; Tim Stark; Aaron Wagner; Hans Jakob Wagner; Thomas Wortmann; Achim Menges
>
> **备注:** to be published in ICRA 2026. Supplementary video: this https URL
>
> **摘要:** Automating large-scale manufacturing in domains like timber construction requires multi-robot systems to manage tightly coupled spatiotemporal constraints, such as collision avoidance and process-driven deadlines. This paper introduces LASER (Level-based Asynchronous Scheduling and Execution Regime), a complete framework for scheduling and executing complex assembly tasks, demonstrated on a screw-press gluing application for timber slab manufacturing. Our central contribution is to integrate a barrier-based mechanism into a constraint programming (CP) scheduling formulation that partitions tasks into spatiotemporally disjoint sets, which we define as levels. This structure enables robots to execute tasks in parallel and asynchronously within a level, synchronizing only at level barriers, which guarantees collision-free operation by construction and provides robustness to timing uncertainties. To solve this formulation for large problems, we propose two specialized algorithms: an iterative temporal-relaxation approach for heterogeneous task sequences and a bi-level decomposition for homogeneous tasks that balances workload. We validate the LASER framework by fabricating a full-scale 2.4m x 6m timber slab with a two-robot system mounted on parallel linear tracks, successfully coordinating 108 subroutines and 352 screws under tight adhesive time windows. Computational studies show our method scales steadily with size compared to a monolithic approach.
>
---
#### [new 013] Do World Action Models Generalize Better than VLAs? A Robustness Study
- **分类: cs.RO**

- **简介: 该论文属于机器人动作规划任务，旨在比较世界动作模型（WAMs）与视觉-语言-动作模型（VLAs）的泛化能力。研究通过实验验证WAMs在不同扰动下的鲁棒性更强。**

- **链接: [https://arxiv.org/pdf/2603.22078](https://arxiv.org/pdf/2603.22078)**

> **作者:** Zhanguang Zhang; Zhiyuan Li; Behnam Rahmati; Rui Heng Yang; Yintao Ma; Amir Rasouli; Sajjad Pakdamansavoji; Yangzheng Wu; Lingfeng Zhang; Tongtong Cao; Feng Wen; Xingyue Quan; Yingxue Zhang
>
> **摘要:** Robot action planning in the real world is challenging as it requires not only understanding the current state of the environment but also predicting how it will evolve in response to actions. Vision-language-action (VLA), which repurpose large-scale vision-language models for robot action generation using action experts, have achieved notable success across a variety of robotic tasks. Nevertheless, their performance remains constrained by the scope of their training data, exhibiting limited generalization to unseen scenarios and vulnerability to diverse contextual perturbations. More recently, world models have been revisited as an alternative to VLAs. These models, referred to as world action models (WAMs), are built upon world models that are trained on large corpora of video data to predict future states. With minor adaptations, their latent representation can be decoded into robot actions. It has been suggested that their explicit dynamic prediction capacity, combined with spatiotemporal priors acquired from web-scale video pretraining, enables WAMs to generalize more effectively than VLAs. In this paper, we conduct a comparative study of prominent state-of-the-art VLA policies and recently released WAMs. We evaluate their performance on the LIBERO-Plus and RoboTwin 2.0-Plus benchmarks under various visual and language perturbations. Our results show that WAMs achieve strong robustness, with LingBot-VA reaching 74.2% success rate on RoboTwin 2.0-Plus and Cosmos-Policy achieving 82.2% on LIBERO-Plus. While VLAs such as $\pi_{0.5}$ can achieve comparable robustness on certain tasks, they typically require extensive training with diverse robotic datasets and varied learning objectives. Hybrid approaches that partially incorporate video-based dynamic learning exhibit intermediate robustness, highlighting the importance of how video priors are integrated.
>
---
#### [new 014] GaussianSSC: Triplane-Guided Directional Gaussian Fields for 3D Semantic Completion
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出GaussianSSC，用于3D语义补全任务，解决单目占用估计与语义预测问题，通过高斯场与三平面引导提升精度。**

- **链接: [https://arxiv.org/pdf/2603.21487](https://arxiv.org/pdf/2603.21487)**

> **作者:** Ruiqi Xian; Jing Liang; He Yin; Xuewei Qi; Dinesh Manocha
>
> **摘要:** We present \emph{GaussianSSC}, a two-stage, grid-native and triplane-guided approach to semantic scene completion (SSC) that injects the benefits of Gaussians without replacing the voxel grid or maintaining a separate Gaussian set. We introduce \emph{Gaussian Anchoring}, a sub-pixel, Gaussian-weighted image aggregation over fused FPN features that tightens voxel--image alignment and improves monocular occupancy estimation. We further convert point-like voxel features into a learned per-voxel Gaussian field and refine triplane features via a triplane-aligned \emph{Gaussian--Triplane Refinement} module that combines \emph{local gathering} (target-centric) and \emph{global aggregation} (source-centric). This directional, anisotropic support captures surface tangency, scale, and occlusion-aware asymmetry while preserving the efficiency of triplane representations. On SemanticKITTI~\cite{behley2019semantickitti}, GaussianSSC improves Stage~1 occupancy by +1.0\% Recall, +2.0\% Precision, and +1.8\% IoU over state-of-the-art baselines, and improves Stage~2 semantic prediction by +1.8\% IoU and +0.8\% mIoU.
>
---
#### [new 015] Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动控制任务，旨在解决从人类数据到机器人运动的映射问题。通过提出NMR框架，利用神经网络学习运动分布，提升运动重定向的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.22201](https://arxiv.org/pdf/2603.22201)**

> **作者:** Qingrui Zhao; Kaiyue Yang; Xiyu Wang; Shiqi Zhao; Yi Lu; Xinfang Zhang; Wei Yin; Qiu Shen; Xiao-Xiao Long; Xun Cao
>
> **备注:** Report, 12 pages, 5 figures, 4 tables
>
> **摘要:** Humanoid robots require diverse motor skills to integrate into complex environments, but bridging the kinematic and dynamic embodiment gap from human data remains a major bottleneck. We demonstrate through Hessian analysis that traditional optimization-based retargeting is inherently non-convex and prone to local optima, leading to physical artifacts like joint jumps and self-penetration. To address this, we reformulate the targeting problem as learning data distribution rather than optimizing optimal solutions, where we propose NMR, a Neural Motion Retargeting framework that transforms static geometric mapping into a dynamics-aware learned process. We first propose Clustered-Expert Physics Refinement (CEPR), a hierarchical data pipeline that leverages VAE-based motion clustering to group heterogeneous movements into latent motifs. This strategy significantly reduces the computational overhead of massively parallel reinforcement learning experts, which project and repair noisy human demonstrations onto the robot's feasible motion manifold. The resulting high-fidelity data supervises a non-autoregressive CNN-Transformer architecture that reasons over global temporal context to suppress reconstruction noise and bypass geometric traps. Experiments on the Unitree G1 humanoid across diverse dynamic tasks (e.g., martial arts, dancing) show that NMR eliminates joint jumps and significantly reduces self-collisions compared to state-of-the-art baselines. Furthermore, NMR-generated references accelerate the convergence of downstream whole-body control policies, establishing a scalable path for bridging the human-robot embodiment gap.
>
---
#### [new 016] MEVIUS2: Practical Open-Source Quadruped Robot with Sheet Metal Welding and Multimodal Perception
- **分类: cs.RO**

- **简介: 本文介绍MEVIUS2，一款开源四足机器人，解决传统开源机器人结构脆弱、尺寸小、感知能力弱的问题。采用钣金焊接和多模态传感器，提升耐用性和环境感知能力。**

- **链接: [https://arxiv.org/pdf/2603.22031](https://arxiv.org/pdf/2603.22031)**

> **作者:** Kento Kawaharazuka; Keita Yoneda; Shintaro Inoue; Temma Suzuki; Jun Oda; Kei Okada
>
> **备注:** Accepted to IEEE Robotics and Automation Practice, Website - this https URL
>
> **摘要:** Various quadruped robots have been developed to date, and thanks to reinforcement learning, they are now capable of traversing diverse types of rough terrain. In parallel, there is a growing trend of releasing these robot designs as open-source, enabling researchers to freely build and modify robots themselves. However, most existing open-source quadruped robots have been designed with 3D printing in mind, resulting in structurally fragile systems that do not scale well in size, leading to the construction of relatively small robots. Although a few open-source quadruped robots constructed with metal components exist, they still tend to be small in size and lack multimodal sensors for perception, making them less practical. In this study, we developed MEVIUS2, an open-source quadruped robot with a size comparable to Boston Dynamics' Spot, whose structural components can all be ordered through e-commerce services. By leveraging sheet metal welding and metal machining, we achieved a large, highly durable body structure while reducing the number of individual parts. Furthermore, by integrating sensors such as LiDARs and a high dynamic range camera, the robot is capable of detailed perception of its surroundings, making it more practical than previous open-source quadruped robots. We experimentally validated that MEVIUS2 can traverse various types of rough terrain and demonstrated its environmental perception capabilities. All hardware, software, and training environments can be obtained from Supplementary Materials or this https URL.
>
---
#### [new 017] DexDrummer: In-Hand, Contact-Rich, and Long-Horizon Dexterous Robot Drumming
- **分类: cs.RO**

- **简介: 该论文提出DexDrummer，解决机器人长时序、高接触的灵巧敲击任务。通过分层策略结合轨迹规划与强化学习，提升多鼓演奏性能。**

- **链接: [https://arxiv.org/pdf/2603.22263](https://arxiv.org/pdf/2603.22263)**

> **作者:** Hung-Chieh Fang; Amber Xie; Jennifer Grannen; Kenneth Llontop; Dorsa Sadigh
>
> **备注:** Website: this https URL
>
> **摘要:** Performing in-hand, contact-rich, and long-horizon dexterous manipulation remains an unsolved challenge in robotics. Prior hand dexterity works have considered each of these three challenges in isolation, yet do not combine these skills into a single, complex task. To further test the capabilities of dexterity, we propose drumming as a testbed for dexterous manipulation. Drumming naturally integrates all three challenges: it involves in-hand control for stabilizing and adjusting the drumstick with the fingers, contact-rich interaction through repeated striking of the drum surface, and long-horizon coordination when switching between drums and sustaining rhythmic play. We present DexDrummer, a hierarchical object-centric bimanual drumming policy trained in simulation with sim-to-real transfer. The framework reduces the exploration difficulty of pure reinforcement learning by combining trajectory planning with residual RL corrections for fast transitions between drums. A dexterous manipulation policy handles contact-rich dynamics, guided by rewards that explicitly model both finger-stick and stick-drum interactions. In simulation, we show our policy can play two styles of music: multi-drum, bimanual songs and challenging, technical exercises that require increased dexterity. Across simulated bimanual tasks, our dexterous, reactive policy outperforms a fixed grasp policy by 1.87x across easy songs and 1.22x across hard songs F1 scores. In real-world tasks, we show song performance across a multi-drum setup. DexDrummer is able to play our training song and its extended version with an F1 score of 1.0.
>
---
#### [new 018] Emergency Lane-Change Simulation: A Behavioral Guidance Approach for Risky Scenario Generation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶测试任务，旨在解决传统方法生成高风险场景效率低的问题。通过行为引导方法，结合生成对抗网络和强化学习，提升风险场景生成效果。**

- **链接: [https://arxiv.org/pdf/2603.20234](https://arxiv.org/pdf/2603.20234)**

> **作者:** Chen Xiong; Cheng Wang; Yuhang Liu; Zirui Wu; Ye Tian
>
> **摘要:** In contemporary autonomous driving testing, virtual simulation has become an important approach due to its efficiency and cost effectiveness. However, existing methods usually rely on reinforcement learning to generate risky scenarios, making it difficult to efficiently learn realistic emergency behaviors. To address this issue, we propose a behavior guided method for generating high risk lane change scenarios. First, a behavior learning module based on an optimized sequence generative adversarial network is developed to learn emergency lane change behaviors from an extracted dataset. This design alleviates the limitations of existing datasets and improves learning from relatively few samples. Then, the opposing vehicle is modeled as an agent, and the road environment together with surrounding vehicles is incorporated into the operating environment. Based on the Recursive Proximal Policy Optimization strategy, the generated trajectories are used to guide the vehicle toward dangerous behaviors for more effective risk scenario exploration. Finally, the reference trajectory is combined with model predictive control as physical constraints to continuously optimize the strategy and ensure physical authenticity. Experimental results show that the proposed method can effectively learn high risk trajectory behaviors from limited data and generate high risk collision scenarios with better efficiency than traditional methods such as grid search and manual design.
>
---
#### [new 019] Architecture for Multi-Unmanned Aerial Vehicles based Autonomous Precision Agriculture Systems
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文属于农业自动化任务，旨在解决多无人机在精准农业中的协同作业问题。提出一种结构化架构，实现自主任务规划、数据采集与处理，提升系统效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.21183](https://arxiv.org/pdf/2603.21183)**

> **作者:** Ebasa Temesgen; Nathnael Minyelshowa; Lebsework Negash
>
> **摘要:** The use of unmanned aerial vehicles (UAVs) in precision agriculture has seen a huge increase recently. As such, systems that aim to apply various algorithms on the field need a structured framework of abstractions. This paper defines the various tasks of the UAVs in precision agriculture and model them into an architectural framework. The presented architecture is built on the context that there will be minimal physical intervention to do the tasks defined with multiple coordinated and cooperative UAVs. Various tasks such as image processing, path planning, communication, data acquisition, and field mapping are employed in the architecture to provide an efficient system. Besides, different limitation for applying Multi-UAVs in precision agriculture has been considered in designing the architecture. The architecture provides an autonomous end-to-end solution, starting from mission planning, data acquisition and image processing framework that is highly efficient and can enable farmers to comprehensively deploy UAVs onto their lands. Simulation and field tests shows that the architecture offers a number of advantages that include fault-tolerance, robustness, developer and user-friendliness.
>
---
#### [new 020] Memory Over Maps: 3D Object Localization Without Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于目标定位任务，解决传统3D重建耗时耗存储的问题。提出无需构建3D场景的视觉记忆方法，通过2D图像直接定位目标。**

- **链接: [https://arxiv.org/pdf/2603.20530](https://arxiv.org/pdf/2603.20530)**

> **作者:** Rui Zhou; Xander Yap; Jianwen Cao; Allison Lau; Boyang Sun; Marc Pollefeys
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Target localization is a prerequisite for embodied tasks such as navigation and manipulation. Conventional approaches rely on constructing explicit 3D scene representations to enable target localization, such as point clouds, voxel grids, or scene graphs. While effective, these pipelines incur substantial mapping time, storage overhead, and scalability limitations. Recent advances in vision-language models suggest that rich semantic reasoning can be performed directly on 2D observations, raising a fundamental question: is a complete 3D scene reconstruction necessary for object localization? In this work, we revisit object localization and propose a map-free pipeline that stores only posed RGB-D keyframes as a lightweight visual memory--without constructing any global 3D representation of the scene. At query time, our method retrieves candidate views, re-ranks them with a vision-language model, and constructs a sparse, on-demand 3D estimate of the queried target through depth backprojection and multi-view fusion. Compared to reconstruction-based pipelines, this design drastically reduces preprocessing cost, enabling scene indexing that is over two orders of magnitude faster to build while using substantially less storage. We further validate the localized targets on downstream object-goal navigation tasks. Despite requiring no task-specific training, our approach achieves strong performance across multiple benchmarks, demonstrating that direct reasoning over image-based scene memory can effectively replace dense 3D reconstruction for object-centric robot navigation. Project page: this https URL
>
---
#### [new 021] Speedup Patch: Learning a Plug-and-Play Policy to Accelerate Embodied Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决 embodied 策略执行速度慢的问题。提出 SuP 框架，通过离线数据优化调度器，提升执行效率而不降低成功率。**

- **链接: [https://arxiv.org/pdf/2603.20658](https://arxiv.org/pdf/2603.20658)**

> **作者:** Zhichao Wu; Junyin Ye; Zhilong Zhang; Yihao Sun; Haoxin Lin; Jiaheng Luo; Haoxiang Ren; Lei Yuan; Yang Yu
>
> **摘要:** While current embodied policies exhibit remarkable manipulation skills, their execution remains unsatisfactorily slow as they inherit the tardy pacing of human demonstrations. Existing acceleration methods typically require policy retraining or costly online interactions, limiting their scalability for large-scale foundation models. In this paper, we propose Speedup Patch (SuP), a lightweight, policy-agnostic framework that enables plug-and-play acceleration using solely offline data. SuP introduces an external scheduler that adaptively downsamples action chunks provided by embodied policies to eliminate redundancies. Specifically, we formalize the optimization of our scheduler as a Constrained Markov Decision Process (CMDP) aimed at maximizing efficiency without compromising task performance. Since direct success evaluation is infeasible in offline settings, SuP introduces World Model based state deviation as a surrogate metric to enforce safety constraints. By leveraging a learned world model as a virtual evaluator to predict counterfactual trajectories, the scheduler can be optimized via offline reinforcement learning. Empirical results on simulation benchmarks (Libero, Bigym) and real-world tasks validate that SuP achieves an overall 1.8x execution speedup for diverse policies while maintaining their original success rates.
>
---
#### [new 022] Dreaming the Unseen: World Model-regularized Diffusion Policy for Out-of-Distribution Robustness
- **分类: cs.RO**

- **简介: 该论文属于视觉-运动控制任务，旨在解决扩散策略在分布外扰动下的鲁棒性问题。通过集成扩散世界模型，提升策略的预测能力与抗干扰性能。**

- **链接: [https://arxiv.org/pdf/2603.21017](https://arxiv.org/pdf/2603.21017)**

> **作者:** Ziou Hu; Xiangtong Yao; Yuan Meng; Zhenshan Bing; Alois Knoll
>
> **备注:** Under review
>
> **摘要:** Diffusion policies excel at visuomotor control but often fail catastrophically under severe out-of-distribution (OOD) disturbances, such as unexpected object displacements or visual corruptions. To address this vulnerability, we introduce the Dream Diffusion Policy (DDP), a framework that deeply integrates a diffusion world model into the policy's training objective via a shared 3D visual encoder. This co-optimization endows the policy with robust state-prediction capabilities. When encountering sudden OOD anomalies during inference, DDP detects the real-imagination discrepancy and actively abandons the corrupted visual stream. Instead, it relies on its internal "imagination" (autoregressively forecasted latent dynamics) to safely bypass the disruption, generating imagined trajectories before smoothly realigning with physical reality. Extensive evaluations demonstrate DDP's exceptional resilience. Notably, DDP achieves a 73.8% OOD success rate on MetaWorld (vs. 23.9% without predictive imagination) and an 83.3% success rate under severe real-world spatial shifts (vs. 3.3% without predictive imagination). Furthermore, as a stress test, DDP maintains a 76.7% real-world success rate even when relying entirely on open-loop imagination post-initialization.
>
---
#### [new 023] PRM-as-a-Judge: A Dense Evaluation Paradigm for Fine-Grained Robotic Auditing
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人评估任务，解决传统二元成功率无法反映执行细节的问题，提出PRM-as-a-Judge方法，通过轨迹视频评估任务进度。**

- **链接: [https://arxiv.org/pdf/2603.21669](https://arxiv.org/pdf/2603.21669)**

> **作者:** Yuheng Ji; Yuyang Liu; Huajie Tan; Xuchuan Huang; Fanding Huang; Yijie Xu; Cheng Chi; Yuting Zhao; Huaihai Lyu; Peterson Co; Mingyu Cao; Qiongyu Zhang; Zhe Li; Enshen Zhou; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang; Xiaolong Zheng
>
> **摘要:** Current robotic evaluation is still largely dominated by binary success rates, which collapse rich execution processes into a single outcome and obscure critical qualities such as progress, efficiency, and stability. To address this limitation, we propose PRM-as-a-Judge, a dense evaluation paradigm that leverages Process Reward Models (PRMs) to audit policy execution directly from trajectory videos by estimating task progress from observation sequences. Central to this paradigm is the OPD (Outcome-Process-Diagnosis) metric system, which explicitly formalizes execution quality via a task-aligned progress potential. We characterize dense robotic evaluation through two axiomatic properties: macro-consistency, which requires additive and path-consistent aggregation, and micro-resolution, which requires sensitivity to fine-grained physical evolution. Under this formulation, potential-based PRM judges provide a natural instantiation of dense evaluation, with macro-consistency following directly from the induced scalar potential. We empirically validate the micro-resolution property using RoboPulse, a diagnostic benchmark specifically designed for probing micro-scale progress discrimination, where several trajectory-trained PRM judges outperform discriminative similarity-based methods and general-purpose foundation-model judges. Finally, leveraging PRM-as-a-Judge and the OPD metric system, we conduct a structured audit of mainstream policy paradigms across long-horizon tasks, revealing behavioral signatures and failure modes that are invisible to outcome-only metrics.
>
---
#### [new 024] SwiftBot: A Decentralized Platform for LLM-Powered Federated Robotic Task Execution
- **分类: cs.RO**

- **简介: 该论文提出SwiftBot，解决联邦机器人任务执行中的资源管理和指令分解问题，通过LLM和DHT实现去中心化任务协作与高效资源调度。**

- **链接: [https://arxiv.org/pdf/2603.20233](https://arxiv.org/pdf/2603.20233)**

> **作者:** YueMing Zhang; Shuai Xu; Zhengxiong Li; Fangtian Zhong; Xiaokun Yang; Hailu Xu
>
> **备注:** This paper has been accepted by IEEE CCGrid 2026. We upload to arXiv for pre-print
>
> **摘要:** Federated robotic task execution systems require bridging natural language instructions to distributed robot control while efficiently managing computational resources across heterogeneous edge devices without centralized coordination. Existing approaches face three limitations: rigid hand-coded planners requiring extensive domain engineering, centralized coordination that contradicts federated collaboration as robots scale, and static resource allocation failing to share containers across robots when workloads shift dynamically. We present SwiftBot, a federated task execution platform that integrates LLM-based task decomposition with intelligent container orchestration over a DHT overlay, enabling robots to collaboratively execute tasks without centralized control. SwiftBot achieves 94.3% decomposition accuracy across diverse tasks, reduces task startup latency by 1.5-5.4x and average training latency by 1.4-2.5x, and improves tail latency by 1.2-4.7x under high load through federated warm container migration. Evaluation on multimedia tasks validates that co-designing semantic understanding and federated resource management enables both flexibility and efficiency for robotic task control.
>
---
#### [new 025] Memory-Efficient Boundary Map for Large-Scale Occupancy Grid Mapping
- **分类: cs.RO**

- **简介: 该论文属于环境建模任务，旨在解决高分辨率大场景下占用网格映射的内存消耗问题。提出一种仅维护边界的新表示方法，减少内存使用并支持高效查询。**

- **链接: [https://arxiv.org/pdf/2603.21774](https://arxiv.org/pdf/2603.21774)**

> **作者:** Benxu Tang; Yunfan Ren; Yixi Cai; Fanze Kong; Wenyi Liu; Fangcheng Zhu; Longji Yin; Liuyu Shi; Fu Zhang
>
> **摘要:** Determining the occupancy status of locations in the environment is a fundamental task for safety-critical robotic applications. Traditional occupancy grid mapping methods subdivide the environment into a grid of voxels, each associated with one of three occupancy states: free, occupied, or unknown. These methods explicitly maintain all voxels within the mapped volume and determine the occupancy state of a location by directly querying the corresponding voxel that the location falls within. However, maintaining all grid voxels in high-resolution and large-scale scenarios requires substantial memory resources. In this paper, we introduce a novel representation that only maintains the boundary of the mapped volume. Specifically, we explicitly represent the boundary voxels, such as the occupied voxels and frontier voxels, while free and unknown voxels are automatically represented by volumes within or outside the boundary, respectively. As our representation maintains only a closed surface in two-dimensional (2D) space, instead of the entire volume in three-dimensional (3D) space, it significantly reduces memory consumption. Then, based on this 2D representation, we propose a method to determine the occupancy state of arbitrary locations in the 3D environment. We term this method as boundary map. Besides, we design a novel data structure for maintaining the boundary map, supporting efficient occupancy state queries. Theoretical analyses of the occupancy state query algorithm are also provided. Furthermore, to enable efficient construction and updates of the boundary map from the real-time sensor measurements, we propose a global-local mapping framework and corresponding update algorithms. Finally, we will make our implementation of the boundary map open-source on GitHub to benefit the community:this https URL.
>
---
#### [new 026] TRGS-SLAM: IMU-Aided Gaussian Splatting SLAM for Blurry, Rolling Shutter, and Noisy Thermal Images
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，旨在解决热成像中运动模糊、滚动快门失真和噪声问题。提出TRGS-SLAM系统，结合IMU与3DGS技术实现鲁棒定位与建图。**

- **链接: [https://arxiv.org/pdf/2603.20443](https://arxiv.org/pdf/2603.20443)**

> **作者:** Spencer Carmichael; Katherine A. Skinner
>
> **备注:** Project page: this https URL
>
> **摘要:** Thermal cameras offer several advantages for simultaneous localization and mapping (SLAM) with mobile robots: they provide a passive, low-power solution to operating in darkness, are invariant to rapidly changing or high dynamic range illumination, and can see through fog, dust, and smoke. However, uncooled microbolometer thermal cameras, the only practical option in most robotics applications, suffer from significant motion blur, rolling shutter distortions, and fixed pattern noise. In this paper, we present TRGS-SLAM, a 3D Gaussian Splatting (3DGS) based thermal inertial SLAM system uniquely capable of handling these degradations. To overcome the challenges of thermal data, we introduce a model-aware 3DGS rendering method and several general innovations to 3DGS SLAM, including B-spline trajectory optimization with a two-stage IMU loss, view-diversity-based opacity resetting, and pose drift correction schemes. Our system demonstrates accurate tracking on real-world, fast motion, and high-noise thermal data that causes all other tested SLAM methods to fail. Moreover, through offline refinement of our SLAM results, we demonstrate thermal image restoration competitive with prior work that required ground truth poses.
>
---
#### [new 027] RTD-RAX: Fast, Safe Trajectory Planning for Systems under Unknown Disturbances
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于轨迹规划任务，解决未知扰动下的安全与快速规划问题。提出RTD-RAX方法，结合非保守RTD和混合单调可达性，实现快速安全轨迹生成与修复。**

- **链接: [https://arxiv.org/pdf/2603.21635](https://arxiv.org/pdf/2603.21635)**

> **作者:** Evanns Morales-Cuadrado; Long Kiu Chung; Shreyas Kousik; Samuel Coogan
>
> **摘要:** Reachability-based Trajectory Design (RTD) is a provably safe, real-time trajectory planning framework that combines offline reachable-set computation with online trajectory optimization. However, standard RTD implementations suffer from two key limitations: conservatism induced by worst-case reachable-set overapproximations, and an inability to account for real-time disturbances during execution. This paper presents RTD-RAX, a runtime-assurance extension of RTD that utilizes a non-conservative RTD formulation to rapidly generate goal-directed candidate trajectories, and utilizes mixed monotone reachability for fast, disturbance-aware online safety certification. When proposed trajectories fail safety certification under real-time uncertainty, a repair procedure finds nearby safe trajectories that preserve progress toward the goal while guaranteeing safety under real-time disturbances.
>
---
#### [new 028] Cortical Policy: A Dual-Stream View Transformer for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出Cortical Policy，一种用于机器人操作的双流视图变换器，解决静态与动态视觉信息融合问题，提升空间推理和动态适应能力。**

- **链接: [https://arxiv.org/pdf/2603.21051](https://arxiv.org/pdf/2603.21051)**

> **作者:** Xuening Zhang; Qi Lv; Xiang Deng; Miao Zhang; Xingbo Liu; Liqiang Nie
>
> **备注:** Published as a conference paper at ICLR 2026. 10 pages, 4 figures. Appendix included
>
> **摘要:** View transformers process multi-view observations to predict actions and have shown impressive performance in robotic manipulation. Existing methods typically extract static visual representations in a view-specific manner, leading to inadequate 3D spatial reasoning ability and a lack of dynamic adaptation. Taking inspiration from how the human brain integrates static and dynamic views to address these challenges, we propose Cortical Policy, a novel dual-stream view transformer for robotic manipulation that jointly reasons from static-view and dynamic-view streams. The static-view stream enhances spatial understanding by aligning features of geometrically consistent keypoints extracted from a pretrained 3D foundation model. The dynamic-view stream achieves adaptive adjustment through position-aware pretraining of an egocentric gaze estimation model, computationally replicating the human cortical dorsal pathway. Subsequently, the complementary view representations of both streams are integrated to determine the final actions, enabling the model to handle spatially-complex and dynamically-changing tasks under language conditions. Empirical evaluations on RLBench, the challenging COLOSSEUM benchmark, and real-world tasks demonstrate that Cortical Policy outperforms state-of-the-art baselines substantially, validating the superiority of dual-stream design for visuomotor control. Our cortex-inspired framework offers a fresh perspective for robotic manipulation and holds potential for broader application in vision-based robot control.
>
---
#### [new 029] Characterizing the onset and offset of motor imagery during passive arm movements induced by an upper-body exoskeleton
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于脑机接口任务，旨在解决非侵入式BMI在康复机器人干扰下检测运动想象起止的问题。通过分析脑电数据，成功识别了运动想象的开始与结束。**

- **链接: [https://arxiv.org/pdf/2603.20885](https://arxiv.org/pdf/2603.20885)**

> **作者:** Kanishka Mitra; Frigyes Samuel Racz; Satyam Kumar; Ashish D. Deshpande; José del R. Millán
>
> **备注:** Accepted to IROS 2023. 6 pages, 6 figures. Project page available at this https URL
>
> **摘要:** Two distinct technologies have gained attention lately due to their prospects for motor rehabilitation: robotics and brain-machine interfaces (BMIs). Harnessing their combined efforts is a largely uncharted and promising direction that has immense clinical potential. However, a significant challenge is whether motor intentions from the user can be accurately detected using non-invasive BMIs in the presence of instrumental noise and passive movements induced by the rehabilitation exoskeleton. As an alternative to the straightforward continuous control approach, this study instead aims to characterize the onset and offset of motor imagery during passive arm movements induced by an upper-body exoskeleton to allow for the natural control (initiation and termination) of functional movements. Ten participants were recruited to perform kinesthetic motor imagery (MI) of the right arm while attached to the robot, simultaneously cued with LEDs indicating the initiation and termination of a goal-oriented reaching task. Using electroencephalogram signals, we built a decoder to detect the transition between i) rest and beginning MI and ii) maintaining and ending MI. Offline decoder evaluation achieved group average onset accuracy of 60.7% and 66.6% for offset accuracy, revealing that the start and stop of MI could be identified while attached to the robot. Furthermore, pseudo-online evaluation could replicate this performance, forecasting reliable online exoskeleton control in the future. Our approach showed that participants could produce quality and reliable sensorimotor rhythms regardless of noise or passive arm movements induced by wearing the exoskeleton, which opens new possibilities for BMI control of assistive devices.
>
---
#### [new 030] Conformal Koopman for Embedded Nonlinear Control with Statistical Robustness: Theory and Real-World Validation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于非线性控制任务，解决模型不确定性下的安全控制问题。提出基于Koopman框架的闭环控制方法，结合共形预测确保统计鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.21580](https://arxiv.org/pdf/2603.21580)**

> **作者:** Koki Hirano; Hiroyasu Tsukamoto
>
> **备注:** 8 pages, 6 figures. Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA). The final published version will be available via IEEE Xplore
>
> **摘要:** We propose a fully data-driven, Koopman-based framework for statistically robust control of discrete-time nonlinear systems with linear embeddings. Establishing a connection between the Koopman operator and contraction theory, it offers distribution-free probabilistic bounds on the state tracking error under Koopman modeling uncertainty. Conformal prediction is employed here to rigorously derive a bound on the state-dependent modeling uncertainty throughout the trajectory, ensuring safety and robustness without assuming a specific error prediction structure or distribution. Unlike prior approaches that merely combine conformal prediction with Koopman-based control in an open-loop setting, our method establishes a closed-loop control architecture with formal guarantees that explicitly account for both forward and inverse modeling errors. Also, by expressing the tracking error bound in terms of the control parameters and the modeling errors, our framework offers a quantitative means to formally enhance the performance of arbitrary Koopman-based control. We validate our method both in numerical simulations with the Dubins car and in real-world experiments with a highly nonlinear flapping-wing drone. The results demonstrate that our method indeed provides formal safety guarantees while maintaining accurate tracking performance under Koopman modeling uncertainty.
>
---
#### [new 031] BiPreManip: Learning Affordance-Based Bimanual Preparatory Manipulation through Anticipatory Collaboration
- **分类: cs.RO**

- **简介: 该论文研究双臂协同准备操作任务，解决物体难以直接抓取或操作的问题。提出基于视觉效用的框架，通过预判动作引导双臂协作，提升任务成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.21679](https://arxiv.org/pdf/2603.21679)**

> **作者:** Yan Shen; Feng Jiang; Zichen He; Xiaoqi Li; Yuchen Liu; Zhiyu Li; Ruihai Wu; Hao Dong
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Many everyday objects are difficult to directly grasp (e.g., a flat iPad) or manipulate functionally (e.g., opening the cap of a pen lying on a desk). Such tasks require sequential, asymmetric coordination between two arms, where one arm performs preparatory manipulation that enables the other's goal-directed action - for instance, pushing the iPad to the table's edge before picking it up, or lifting the pen body to allow the other hand to remove its cap. In this work, we introduce Collaborative Preparatory Manipulation, a class of bimanual manipulation tasks that demand understanding object semantics and geometry, anticipating spatial relationships, and planning long-horizon coordinated actions between the two arms. To tackle this challenge, we propose a visual affordance-based framework that first envisions the final goal-directed action and then guides one arm to perform a sequence of preparatory manipulations that facilitate the other arm's subsequent operation. This affordance-centric representation enables anticipatory inter-arm reasoning and coordination, generalizing effectively across various objects spanning diverse categories. Extensive experiments in both simulation and the real world demonstrate that our approach substantially improves task success rates and generalization compared to competitive baselines.
>
---
#### [new 032] Dynamic Control Barrier Function Regulation with Vision-Language Models for Safe, Adaptive, and Realtime Visual Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决动态环境中安全与效率的平衡问题。通过视觉语言模型实时调整控制屏障函数参数，提升导航安全性与效率。**

- **链接: [https://arxiv.org/pdf/2603.21142](https://arxiv.org/pdf/2603.21142)**

> **作者:** Jeffrey Chen; Rohan Chandra
>
> **摘要:** Robots operating in dynamic, unstructured environments must balance safety and efficiency under potentially limited sensing. While control barrier functions (CBFs) provide principled collision avoidance via safety filtering, their behavior is often governed by fixed parameters that can be overly conservative in benign scenes or overly permissive near hazards. We present AlphaAdj, a vision-to-control navigation framework that uses egocentric RGB input to adapt the conservativeness of a CBF safety filter in real time. A vision-language model(VLM) produces a bounded scalar risk estimate from the current camera view, which we map to dynamically update a CBF parameter that modulates how strongly safety constraints are enforced. To address asynchronous inference and non-trivial VLM latency in practice, we combine a geometric, speed-aware dynamic cap and a staleness-gated fusion policy with lightweight implementation choices that reduce end-to-end inference overhead. We evaluate AlphaAdj across multiple static and dynamic obstacle scenarios in a variety of environments, comparing against fixed-parameter and uncapped ablations. Results show that AlphaAdj maintains collision-free navigation while improving efficiency (in terms of path length and time to goal) by up to 18.5% relative to fixed settings and improving robustness and success rate relative to an uncapped baseline.
>
---
#### [new 033] DyGeoVLN: Infusing Dynamic Geometry Foundation Model into Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决动态环境中导航泛化能力不足的问题。提出DyGeoVLN框架，融合动态几何模型，提升空间表示与语义推理能力。**

- **链接: [https://arxiv.org/pdf/2603.21269](https://arxiv.org/pdf/2603.21269)**

> **作者:** Xiangchen Liu; Hanghan Zheng; Jeil Jeong; Minsung Yoon; Lin Zhao; Zhide Zhong; Haoang Li; Sung-Eui Yoon
>
> **摘要:** Vision-language Navigation (VLN) requires an agent to understand visual observations and language instructions to navigate in unseen environments. Most existing approaches rely on static scene assumptions and struggle to generalize in dynamic, real-world scenarios. To address this challenge, we propose DyGeoVLN, a dynamic geometry-aware VLN framework. Our method infuses a dynamic geometry foundation model into the VLN framework through cross-branch feature fusion to enable explicit 3D spatial representation and visual-semantic reasoning. To efficiently compress historical token information in long-horizon, dynamic navigation, we further introduce a novel pose-free and adaptive-resolution token-pruning strategy. This strategy can remove spatio-temporal redundant tokens to reduce inference cost. Extensive experiments demonstrate that our approach achieves state-of-the-art performance on multiple benchmarks and exhibits strong robustness in real-world environments.
>
---
#### [new 034] Enhancing Vision-Based Policies with Omni-View and Cross-Modality Knowledge Distillation for Mobile Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉控制任务，解决轻量级移动机器人在场景迁移、计算资源和硬件成本上的难题。通过知识蒸馏方法，将全景深度策略的知识迁移到单目策略中，提升其性能。**

- **链接: [https://arxiv.org/pdf/2603.20679](https://arxiv.org/pdf/2603.20679)**

> **作者:** Kai Li; Shiyu Zhao
>
> **摘要:** Vision-based policies are widely applied in robotics for tasks such as manipulation and locomotion. On lightweight mobile robots, however, they face a trilemma of limited scene transferability, restricted onboard computation resources, and sensor hardware cost. To address these issues, we propose a knowledge distillation approach that transfers knowledge from an information-rich, appearance invariant omniview depth policy to a lightweight monocular policy. The key idea is to train the student not only to mimic the expert actions but also to align with the latent embeddings of the omni view depth teacher. Experiments demonstrate that omni-view and depth inputs improve the scene transfer and navigation performance, and that the proposed distillation method enhances the performance of a singleview monocular policy, compared with policies solely imitating actions. Real world experiments further validate the effectiveness and practicality of our approach. Code will be released publicly.
>
---
#### [new 035] VP-VLA: Visual Prompting as an Interface for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决传统模型在空间精度和泛化能力上的不足。提出VP-VLA框架，通过视觉提示分离高阶推理与低阶执行，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.22003](https://arxiv.org/pdf/2603.22003)**

> **作者:** Zixuan Wang; Yuxin Chen; Yuqi Liu; Jinhui Ye; Pengguang Chen; Changsheng Lu; Shu Liu; Jiaya Jia
>
> **备注:** Project page: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models typically map visual observations and linguistic instructions directly to robotic control signals. This "black-box" mapping forces a single forward pass to simultaneously handle instruction interpretation, spatial grounding, and low-level control, often leading to poor spatial precision and limited robustness in out-of-distribution scenarios. To address these limitations, we propose VP-VLA, a dual-system framework that decouples high-level reasoning and low-level execution via a structured visual prompting interface. Specifically, a "System 2 Planner" decomposes complex instructions into sub-tasks and identifies relevant target objects and goal locations. These spatial anchors are then overlaid directly onto visual observations as structured visual prompts, such as crosshairs and bounding boxes. Guided by these prompts and enhanced by a novel auxiliary visual grounding objective during training, a "System 1 Controller" reliably generates precise low-level execution motions. Experiments on the Robocasa-GR1-Tabletop benchmark and SimplerEnv simulation demonstrate that VP-VLA improves success rates by 5% and 8.3%, surpassing competitive baselines including QwenOFT and GR00T-N1.6.
>
---
#### [new 036] UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos
- **分类: cs.RO**

- **简介: 该论文提出UniDex，解决通用灵巧手控制问题。构建数据集、统一动作空间和采集系统，实现跨手泛化与高效操控。**

- **链接: [https://arxiv.org/pdf/2603.22264](https://arxiv.org/pdf/2603.22264)**

> **作者:** Gu Zhang; Qicheng Xu; Haozhe Zhang; Jianhan Ma; Long He; Yiming Bao; Zeyu Ping; Zhecheng Yuan; Chenhao Lu; Chengbo Yuan; Tianhai Liang; Xiaoyu Tian; Maanping Shao; Feihong Zhang; Mingyu Ding; Yang Gao; Hao Zhao; Hang Zhao; Huazhe Xu
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Dexterous manipulation remains challenging due to the cost of collecting real-robot teleoperation data, the heterogeneity of hand embodiments, and the high dimensionality of control. We present UniDex, a robot foundation suite that couples a large-scale robot-centric dataset with a unified vision-language-action (VLA) policy and a practical human-data capture setup for universal dexterous hand control. First, we construct UniDex-Dataset, a robot-centric dataset over 50K trajectories across eight dexterous hands (6--24 DoFs), derived from egocentric human video datasets. To transform human data into robot-executable trajectories, we employ a human-in-the-loop retargeting procedure to align fingertip trajectories while preserving plausible hand-object contacts, and we operate on explicit 3D pointclouds with human hands masked to narrow kinematic and visual gaps. Second, we introduce the Function-Actuator-Aligned Space (FAAS), a unified action space that maps functionally similar actuators to shared coordinates, enabling cross-hand transfer. Leveraging FAAS as the action parameterization, we train UniDex-VLA, a 3D VLA policy pretrained on UniDex-Dataset and finetuned with task demonstrations. In addition, we build UniDex-Cap, a simple portable capture setup that records synchronized RGB-D streams and human hand poses and converts them into robot-executable trajectories to enable human-robot data co-training that reduces reliance on costly robot demonstrations. On challenging tool-use tasks across two different hands, UniDex-VLA achieves 81% average task progress and outperforms prior VLA baselines by a large margin, while exhibiting strong spatial, object, and zero-shot cross-hand generalization. Together, UniDex-Dataset, UniDex-VLA, and UniDex-Cap provide a scalable foundation suite for universal dexterous manipulation.
>
---
#### [new 037] ROBOGATE: Adaptive Failure Discovery for Safe Robot Policy Deployment via Two-Stage Boundary-Focused Sampling
- **分类: cs.RO**

- **简介: 该论文提出ROBOGATE框架，用于安全机器人策略部署中的故障发现。任务是评估和管理部署风险，解决高维参数空间中故障边界难以检测的问题。通过两阶段采样策略，结合仿真与实验，识别危险区域并建立风险模型。**

- **链接: [https://arxiv.org/pdf/2603.22126](https://arxiv.org/pdf/2603.22126)**

> **作者:** Byungjin Kim
>
> **备注:** 12 pages, 5 figures, open-source code and 30K failure pattern dataset available at this https URL
>
> **摘要:** Deploying learned robot manipulation policies in industrial settings requires rigorous pre-deployment validation, yet exhaustive testing across high-dimensional parameter spaces is intractable. We present ROBOGATE, a deployment risk management framework that combines physics-based simulation with a two-stage adaptive sampling strategy to efficiently discover failure boundaries in the operational parameter space. Stage 1 employs Latin Hypercube Sampling (LHS) across an 8-dimensional parameter space to establish a coarse failure landscape from 20,000 uniformly distributed experiments. Stage 2 applies boundary-focused sampling that concentrates 10,000 additional experiments in the 30-70% success rate transition zone, enabling precise failure boundary mapping. Using NVIDIA Isaac Sim with Newton physics, we evaluate a scripted pick-and-place controller on two robot embodiments -- Franka Panda (7-DOF) and UR5e (6-DOF) -- across 30,000 total experiments. Our logistic regression risk model achieves an AUC of 0.780 on the combined dataset (vs. 0.754 for Stage 1 alone), identifies a closed-form failure boundary equation, and reveals four universal danger zones affecting both robot platforms. We further demonstrate the framework on VLA (Vision-Language-Action) model evaluation, where Octo-Small achieves 0.0% success rate on 68 adversarial scenarios versus 100% for the scripted baseline -- a 100-point gap that underscores the challenge of deploying foundation models in industrial settings. ROBOGATE is open-source and runs on a single GPU workstation.
>
---
#### [new 038] RAFL: Generalizable Sim-to-Real of Soft Robots with Residual Acceleration Field Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出RAFL框架，解决软体机器人仿真到现实的泛化问题。通过残差加速度场学习，提升模拟精度，支持形状优化中的持续改进。**

- **链接: [https://arxiv.org/pdf/2603.22039](https://arxiv.org/pdf/2603.22039)**

> **作者:** Dong Heon Cho; Boyuan Chen
>
> **摘要:** Differentiable simulators enable gradient-based optimization of soft robots over material parameters, control, and morphology, but accurately modeling real systems remains challenging due to the sim-to-real gap. This issue becomes more pronounced when geometry is itself a design variable. System identification reduces discrepancies by fitting global material parameters to data; however, when constitutive models are misspecified or observations are sparse, identified parameters often absorb geometry-dependent effects rather than reflect intrinsic material behavior. More expressive constitutive models can improve accuracy but substantially increase computational cost, limiting practicality. We propose a residual acceleration field learning (RAFL) framework that augments a base simulator with a transferable, element-level corrective dynamics field. Operating on shared local features, the model is agnostic to global mesh topology and discretization. Trained end-to-end through a differentiable simulator using sparse marker observations, the learned residual generalizes across shapes. In both sim-to-sim and sim-to-real experiments, our method achieves consistent zero-shot improvements on unseen morphologies, while system identification frequently exhibits negative transfer. The framework also supports continual refinement, enabling simulation accuracy to accumulate during morphology optimization.
>
---
#### [new 039] Collision-Free Velocity Scheduling for Multi-Agent Systems on Predefined Routes via Inexact-Projection ADMM
- **分类: cs.RO; eess.SY; math.OC**

- **简介: 该论文属于多智能体协同任务，解决路径约束下的无碰撞速度调度问题，通过改进的ADMM算法优化通行时间，提升任务效率。**

- **链接: [https://arxiv.org/pdf/2603.21913](https://arxiv.org/pdf/2603.21913)**

> **作者:** Seungyeop Lee; Jong-Han Kim
>
> **摘要:** In structured multi-agent transportation systems, agents often must follow predefined routes, making spatial rerouting undesirable or impossible. This paper addresses route-constrained multi-agent coordination by optimizing waypoint passage times while preserving each agent's assigned waypoint order and nominal route assignment. A differentiable surrogate trajectory model maps waypoint timings to smooth position profiles and captures first-order tracking lag, enabling pairwise safety to be encoded through distance-based penalties evaluated on a dense temporal grid spanning the mission horizon. The resulting nonlinear and nonconvex velocity-scheduling problem is solved using an inexact-projection Alternating Direction Method of Multipliers (ADMM) algorithm that combines structured timing updates with gradient-based collision-correction steps and avoids explicit integer sequencing variables. Numerical experiments on random-crossing, bottleneck, and graph-based network scenarios show that the proposed method computes feasible and time-efficient schedules across a range of congestion levels and yields shorter mission completion times than a representative hierarchical baseline in the tested bottleneck cases.
>
---
#### [new 040] Bayesian Active Object Recognition and 6D Pose Estimation from Multimodal Contact Sensing
- **分类: cs.RO**

- **简介: 该论文属于物体识别与6D位姿估计任务，解决如何通过多模态触觉传感提高估计精度与稳定性的问题。工作包括构建贝叶斯框架、设计粒子滤波和运动规划方法。**

- **链接: [https://arxiv.org/pdf/2603.21410](https://arxiv.org/pdf/2603.21410)**

> **作者:** Haodong Zheng; Gabriele M. Caddeo; Andrei C. Jalba; Wijnand A. IJsselsteijn; Lorenzo Natale; Raymond H. Cuijpers
>
> **摘要:** We present an active tactile exploration framework for joint object recognition and 6D pose estimation. The proposed method integrates wrist force/torque sensing, GelSight tactile sensing, and free-space constraints within a Bayesian inference framework that maintains a belief over object class and pose during active tactile exploration. By combining contact and non-contact evidence, the framework reduces ambiguity and improves robustness in the joint class-pose estimation problem. To enable efficient inference in the large hypothesis space, we employ a customized particle filter that progressively samples particles based on new observations. The inferred belief is further used to guide active exploration by selecting informative next touches under reachability constraints. For effective data collection, a motion planning and control framework is developed to plan and execute feasible paths for tactile exploration, handle unexpected contacts and GelSight-surface alignment with tactile servoing. We evaluate the framework in simulation and on a Franka Panda robot using 11 YCB objects. Results show that incorporating tactile and free-space information substantially improves recognition and pose estimation accuracy and stability, while reducing the number of action cycles compared with force/torque-only baselines. Code, dataset, and supplementary material will be made available online.
>
---
#### [new 041] Your Robot Will Feel You Now: Empathy in Robots and Embodied Agents
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文探讨机器人和具身代理中的共情实现，分析其行为模型与类人模仿。属于人机交互任务，旨在提升机器的情感智能。**

- **链接: [https://arxiv.org/pdf/2603.20200](https://arxiv.org/pdf/2603.20200)**

> **作者:** Angelica Lim; Ö. Nilay Yalçin
>
> **备注:** Accepted manuscript. Chapter in "Empathy and Artificial Intelligence: Challenges, Advances and Ethical Considerations" edited by Anat Perry; C. Daryl Cameron
>
> **摘要:** The fields of human-robot interaction (HRI) and embodied conversational agents (ECAs) have long studied how empathy could be implemented in machines. One of the major drivers has been the goal of giving multimodal social and emotional intelligence to these artificially intelligent agents, which interact with people through facial expressions, body, gesture, and speech. What empathic behaviors and models have these fields implemented by mimicking human and animal behavior? In what ways have they explored creating machine-specific analogies? This chapter aims to review the knowledge from these studies, towards applying the lessons learned to today's ubiquitous, language-based agents such as ChatGPT.
>
---
#### [new 042] E-SocialNav: Efficient Socially Compliant Navigation with Language Models
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决语言模型在导航中社会合规性不足及效率低的问题。工作包括评估现有模型并提出高效且符合社会规范的E-SocialNav模型。**

- **链接: [https://arxiv.org/pdf/2603.20664](https://arxiv.org/pdf/2603.20664)**

> **作者:** Ling Xiao; Daeun Song; Xuesu Xiao; Toshihiko Yamasaki
>
> **备注:** Accepted by 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing, to appear. Preprint version
>
> **摘要:** Language models (LMs) are increasingly applied to robotic navigation; however, existing benchmarks primarily emphasize navigation success rates while paying limited attention to social compliance. Moreover, relying on large-scale LMs can raise efficiency concerns, as their heavy computational overhead leads to slower response times and higher energy consumption, making them impractical for real-time deployment on resource-constrained robotic platforms. In this work, we evaluate the social compliance of GPT-4o and Claude in robotic navigation and propose E-SocialNav, an efficient LM designed for socially compliant navigation. Despite being trained on a relatively small dataset, E-SocialNav consistently outperforms zero-shot baselines in generating socially compliant behaviors. By employing a two-stage training pipeline consisting of supervised fine-tuning followed by direct preference optimization, E-SocialNav achieves strong performance in both text-level semantic similarity to human annotations and action accuracy. The source code is available at this https URL.
>
---
#### [new 043] SafePilot: A Framework for Assuring LLM-enabled Cyber-Physical Systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SafePilot框架，解决LLM在CPS中因幻觉导致的安全问题，通过层次化验证确保任务规划的正确性。**

- **链接: [https://arxiv.org/pdf/2603.21523](https://arxiv.org/pdf/2603.21523)**

> **作者:** Weizhe Xu; Mengyu Liu; Fanxin Kong
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs), deep learning architectures with typically over 10 billion parameters, have recently begun to be integrated into various cyber-physical systems (CPS) such as robotics, industrial automation, and autopilot systems. The abstract knowledge and reasoning capabilities of LLMs are employed for tasks like planning and navigation. However, a significant challenge arises from the tendency of LLMs to produce "hallucinations" - outputs that are coherent yet factually incorrect or contextually unsuitable. This characteristic can lead to undesirable or unsafe actions in the CPS. Therefore, our research focuses on assuring the LLM-enabled CPS by enhancing their critical properties. We propose SafePilot, a novel hierarchical neuro-symbolic framework that provides end-to-end assurance for LLM-enabled CPS according to attribute-based and temporal specifications. Given a task and its specification, SafePilot first invokes a hierarchical planner with a discriminator that assesses task complexity. If the task is deemed manageable, it is passed directly to an LLM-based task planner with built-in verification. Otherwise, the hierarchical planner applies a divide-and-conquer strategy, decomposing the task into sub-tasks, each of which is individually planned and later merged into a final solution. The LLM-based task planner translates natural language constraints into formal specifications and verifies the LLM's output against them. If violations are detected, it identifies the flaw, adjusts the prompt accordingly, and re-invokes the LLM. This iterative process continues until a valid plan is produced or a predefined limit is reached. Our framework supports LLM-enabled CPS with both attribute-based and temporal constraints. Its effectiveness and adaptability are demonstrated through two illustrative case studies.
>
---
#### [new 044] Programming Manufacturing Robots with Imperfect AI: LLMs as Tuning Experts for FDM Print Configuration Selection
- **分类: cs.RO**

- **简介: 该论文研究如何利用LLM优化FDM 3D打印配置，解决新手用户依赖不准确AI推荐的问题。通过闭环优化方法提升打印质量与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.22118](https://arxiv.org/pdf/2603.22118)**

> **作者:** Ekta U. Samani; Christopher G. Atkeson
>
> **摘要:** We use fused deposition modeling (FDM) 3D printing as a case study of how manufacturing robots can use imperfect AI to acquire process expertise. In FDM, print configuration strongly affects output quality. Yet, novice users typically rely on default configurations, trial-and-error, or recommendations from generic AI models (e.g., ChatGPT). These strategies can produce complete prints, but they do not reliably meet specific objectives. Experts iteratively tune print configurations using evidence from prior prints. We present a modular closed-loop approach that treats an LLM as a source of tuning expertise. We embed this source of expertise within a Bayesian optimization loop. An approximate evaluator scores each print configuration and returns structured diagnostics, which the LLM uses to propose natural-language adjustments that are compiled into machine-actionable guidance for optimization. On 100 Thingi10k parts, our LLM-guided loop achieves the best configuration on 78% objects with 0% likely-to-fail cases, while single-shot AI model recommendations are rarely best and exhibit 15% likely-to-fail cases. These results suggest that LLMs provide more value as constrained decision modules in evidence-driven optimization loops than as end-to-end oracles for print configuration selection. We expect this result to extend to broader LLM-based robot programming.
>
---
#### [new 045] VisFly-Lab: Unified Differentiable Framework for First-Order Reinforcement Learning of Quadrotor Control
- **分类: cs.RO**

- **简介: 该论文提出VisFly-Lab框架，解决四旋翼控制中强化学习任务的碎片化问题，通过统一接口和改进算法提升训练效果与实际应用能力。**

- **链接: [https://arxiv.org/pdf/2603.21123](https://arxiv.org/pdf/2603.21123)**

> **作者:** Fanxing Li; Fangyu Sun; Tianbao Zhang; Shuyu Wu; Dexin Zuo; yufei Yan; Wenxian Yu; Danping Zou
>
> **摘要:** First-order reinforcement learning with differentiable simulation is promising for quadrotor control, but practical progress remains fragmented across task-specific settings. To support more systematic development and evaluation, we present a unified differentiable framework for multi-task quadrotor control. The framework is wrapped, extensible, and equipped with deployment-oriented dynamics, providing a common interface across four representative tasks: hovering, tracking, landing, and racing. We also present the suite of first-order learning algorithms, where we identify two practical bottlenecks of standard first-order training: limited state coverage caused by horizon initialization and gradient bias caused by partially non-differentiable rewards. To address these issues, we propose Amended Backpropagation Through Time (ABPT), which combines differentiable rollout optimization, a value-based auxiliary objective, and visited-state initialization to improve training robustness. Experimental results show that ABPT yields the clearest gains in tasks with partially non-differentiable rewards, while remaining competitive in fully differentiable settings. We further provide proof-of-concept real-world deployments showing initial transferability of policies learned in the proposed framework beyond simulation.
>
---
#### [new 046] GHOST: Ground-projected Hypotheses from Observed Structure-from-Motion Trajectories
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶中的轨迹分割任务，旨在从单目图像中分割可行车辆轨迹。通过自监督学习，利用车载视频生成地面掩码，训练网络预测运动条件下的路径建议。**

- **链接: [https://arxiv.org/pdf/2603.20583](https://arxiv.org/pdf/2603.20583)**

> **作者:** Tomasz Frelek; Rohan Patil; Akshar Tumu; Henrik I. Christensen
>
> **备注:** 8 pages, 27 figures, 1 table
>
> **摘要:** We present a scalable self-supervised approach for segmenting feasible vehicle trajectories from monocular images for autonomous driving in complex urban environments. Leveraging large-scale dashcam videos, we treat recorded ego-vehicle motion as implicit supervision and recover camera trajectories via monocular structure-from-motion, projecting them onto the ground plane to generate spatial masks of traversed regions without manual annotation. These automatically generated labels are used to train a deep segmentation network that predicts motion-conditioned path proposals from a single RGB image at run time, without explicit modeling of road or lane markings. Trained on diverse, unconstrained internet data, the model implicitly captures scene layout, lane topology, and intersection structure, and generalizes across varying camera configurations. We evaluate our approach on NuScenes, demonstrating reliable trajectory prediction, and further show transfer to an electric scooter platform through light fine-tuning. Our results indicate that large-scale ego-motion distillation yields structured and generalizable path proposals beyond the demonstrated trajectory, enabling trajectory hypothesis estimation via image segmentation.
>
---
#### [new 047] Evaluating Factor-Wise Auxiliary Dynamics Supervision for Latent Structure and Robustness in Simulated Humanoid Locomotion
- **分类: cs.RO**

- **简介: 该论文研究模拟人形机器人运动中因子辅助动态监督的效果，评估其是否提升潜在结构或鲁棒性。通过对比不同模型，发现辅助监督未显著提升性能，而LSTM表现更优。**

- **链接: [https://arxiv.org/pdf/2603.21268](https://arxiv.org/pdf/2603.21268)**

> **作者:** Chayanin Chamachot
>
> **备注:** 17 pages, 9 figures, 25 tables
>
> **摘要:** We evaluate whether factor-wise auxiliary dynamics supervision produces useful latent structure or improved robustness in simulated humanoid locomotion. DynaMITE -- a transformer encoder with a factored 24-d latent trained by per-factor auxiliary losses during proximal policy optimization (PPO) -- is compared against Long Short-Term Memory (LSTM), plain Transformer, and Multilayer Perceptron (MLP) baselines on a Unitree G1 humanoid across four Isaac Lab tasks. The supervised latent shows no evidence of decodable or functionally separable factor structure: probe R^2 ~ 0 for all five dynamics factors, clamping any subspace changes reward by < 0.05, and standard disentanglement metrics (MIG, DCI, SAP) are near zero. An unsupervised LSTM hidden state achieves higher probe R^2 (up to 0.10). A 2x2 factorial ablation (n = 10 seeds) isolates the contributions of the tanh bottleneck and auxiliary losses: the auxiliary losses show no measurable effect on either in-distribution (ID) reward (+0.03, p = 0.732) or severe out-of-distribution (OOD) reward (+0.03, p = 0.669), while the bottleneck shows a small, consistent advantage in both regimes (ID: +0.16, p = 0.207; OOD: +0.10, p = 0.208). The bottleneck advantage persists under severe combined perturbation but does not amplify, indicating a training-time representation benefit rather than a robustness mechanism. LSTM achieves the best nominal reward on all four tasks (p < 0.03); DynaMITE degrades less under combined-shift stress (2.3% vs. 16.7%), but this difference is attributable to the bottleneck compression, not the auxiliary supervision. For locomotion practitioners: auxiliary dynamics supervision does not produce an interpretable estimator and does not measurably improve reward or robustness beyond what the bottleneck alone provides; recurrent baselines remain the stronger choice for nominal performance.
>
---
#### [new 048] Swim2Real: VLM-Guided System Identification for Sim-to-Real Transfer
- **分类: cs.RO**

- **简介: 该论文提出Swim2Real，用于解决水下机器人仿真到现实的迁移问题，通过VLM引导的系统识别校准仿真参数，实现无需手动调整的零样本强化学习迁移。**

- **链接: [https://arxiv.org/pdf/2603.20827](https://arxiv.org/pdf/2603.20827)**

> **作者:** Kevin Qiu; Kyle Walker; Mike Y. Michelis; Marek Cygan; Josie Hughes
>
> **摘要:** We present Swim2Real, a pipeline that calibrates a 16-parameter robotic fish simulator from swimming videos using vision-language model (VLM) feedback, requiring no hand-designed search stages. Calibrating soft aquatic robots is particularly challenging because nonlinear fluid-structure coupling makes the parameter landscape chaotic, simplified fluid models introduce a persistent sim-to-real gap, and controlled aquatic experiments are difficult to reproduce. Prior work on this platform required three manually tailored stages to handle this complexity. The VLM compares simulated and real videos and proposes parameter updates. A backtracking line search then validates each step size, tripling the accept rate from 14% to 42% by recovering proposals where the direction is correct but the magnitude is too large. Swim2Real calibrates all 16 parameters simultaneously, most closely matching real fish velocities across all motor frequencies (MAE = 7.4 mm/s, 43% lower than the next-best method), with zero outlier seeds across five runs. Motor commands from the trained policy transfer to the physical fish at 50 Hz, completing the pipeline from swimming video to real-world deployment. Downstream RL policies swim 12% farther than those from BayesOpt-calibrated simulators and 90% farther than CMA-ES. These results demonstrate that VLM-guided calibration can close the sim-to-real gap for aquatic robots directly from video, enabling zero-shot RL transfer to physical swimmers without manual system identification, a step toward automated, general-purpose simulator tuning for underwater robotics.
>
---
#### [new 049] Multi-Robot Learning-Informed Task Planning Under Uncertainty
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人任务规划领域，解决未知环境下高效协作问题。通过结合学习与建模方法，提升长时序决策与协调能力。**

- **链接: [https://arxiv.org/pdf/2603.20544](https://arxiv.org/pdf/2603.20544)**

> **作者:** Abhish Khanal; Abhishek Paudel; Hung Pham; Gregory J. Stein
>
> **备注:** 8 pages, 8 figures. Accepted at ICRA 2026
>
> **摘要:** We want a multi-robot team to complete complex tasks in minimum time where the locations of task-relevant objects are not known. Effective task completion requires reasoning over long horizons about the likely locations of task-relevant objects, how individual actions contribute to overall progress, and how to coordinate team efforts. Planning in this setting is extremely challenging: even when task-relevant information is partially known, coordinating which robot performs which action and when is difficult, and uncertainty introduces a multiplicity of possible outcomes for each action, which further complicates long-horizon decision-making and coordination. To address this, we propose a multi-robot planning abstraction that integrates learning to estimate uncertain aspects of the environment with model-based planning for long-horizon coordination. We demonstrate the efficient multi-stage task planning of our approach for 1, 2, and 3 robot teams over competitive baselines in large ProcTHOR household environments. Additionally, we demonstrate the effectiveness of our approach with a team of two LoCoBot mobile robots in real household settings.
>
---
#### [new 050] A Framework for Closed-Loop Robotic Assembly, Alignment and Self-Recovery of Precision Optical Systems
- **分类: cs.RO; cs.AI; physics.optics**

- **简介: 该论文属于精密光学系统自动化任务，旨在解决手动操作效率低、精度要求高的问题。通过集成视觉、优化和工具，实现光学系统的自主构建与维护。**

- **链接: [https://arxiv.org/pdf/2603.21496](https://arxiv.org/pdf/2603.21496)**

> **作者:** Seou Choi; Sachin Vaidya; Caio Silva; Shiekh Zia Uddin; Sajib Biswas Shuvo; Shrish Choudhary; Marin Soljačić
>
> **摘要:** Robotic automation has transformed scientific workflows in domains such as chemistry and materials science, yet free-space optics, which is a high precision domain, remains largely manual. Optical systems impose strict spatial and angular tolerances, and their performance is governed by tightly coupled physical parameters, making generalizable automation particularly challenging. In this work, we present a robotics framework for the autonomous construction, alignment, and maintenance of precision optical systems. Our approach integrates hierarchical computer vision systems, optimization routines, and custom-built tools to achieve this functionality. As a representative demonstration, we perform the fully autonomous construction of a tabletop laser cavity from randomly distributed components. The system performs several tasks such as laser beam centering, spatial alignment of multiple beams, resonator alignment, laser mode selection, and self-recovery from induced misalignment and disturbances. By achieving closed-loop autonomy for highly sensitive optical systems, this work establishes a foundation for autonomous optical experiments for applications across technical domains.
>
---
#### [new 051] High-Speed, All-Terrain Autonomy: Ensuring Safety at the Limits of Mobility
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主车辆控制任务，解决高速越野时的安全问题。通过新型模型预测控制方法，提升轨迹规划的安全性与实时性。**

- **链接: [https://arxiv.org/pdf/2603.20525](https://arxiv.org/pdf/2603.20525)**

> **作者:** James R. Baxter; Bogdan I. Epureanu; Paramsothy Jayakumar; Tulga Ersal
>
> **备注:** 19 pages, 16 figures, submitted to IEEE Transactions on Robotics
>
> **摘要:** A novel local trajectory planner, capable of controlling an autonomous off-road vehicle on rugged terrain at high-speed is presented. Autonomous vehicles are currently unable to safely operate off-road at high-speed, as current approaches either fail to predict and mitigate rollovers induced by rough terrain or are not real-time feasible. To address this challenge, a novel model predictive control (MPC) formulation is developed for local trajectory planning. A new dynamics model for off-road vehicles on rough, non-planar terrain is derived and used for prediction. Extreme mobility, including tire liftoff without rollover, is safely enabled through a new energy-based constraint. The formulation is analytically shown to mitigate rollover types ignored by many state-of-the-art methods, and real-time feasibility is achieved through parallelized GPGPU computation. The planner's ability to provide safe, extreme trajectories is studied through both simulated trials and full-scale physical experiments. The results demonstrate fewer rollovers and more successes compared to a state-of-the-art baseline across several challenging scenarios that push the vehicle to its mobility limits.
>
---
#### [new 052] Implementing Robust M-Estimators with Certifiable Factor Graph Optimization
- **分类: cs.RO**

- **简介: 该论文属于机器人与计算机视觉中的参数估计任务，解决异常值和非凸优化难题。通过自适应加权M-估计结合可验证的WLS求解器，提升估计质量并保证全局最优。**

- **链接: [https://arxiv.org/pdf/2603.20932](https://arxiv.org/pdf/2603.20932)**

> **作者:** Zhexin Xu; Hanna Jiamei Zhang; Helena Calatrava; Pau Closas; David M. Rosen
>
> **备注:** The paper was accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Parameter estimation in robotics and computer vision faces formidable challenges from both outlier contamination and nonconvex optimization landscapes. While M-estimation addresses the problem of outliers through robust loss functions, it creates severely nonconvex problems that are difficult to solve globally. Adaptive reweighting schemes provide one particularly appealing strategy for implementing M-estimation in practice: these methods solve a sequence of simpler weighted least squares (WLS) subproblems, enabling both the use of standard least squares solvers and the recovery of higher-quality estimates than simple local search. However, adaptive reweighting still crucially relies upon solving the inner WLS problems effectively, a task that remains challenging in many robotics applications due to the intrinsic nonconvexity of many common parameter spaces (e.g. rotations and poses). In this paper, we show how one can easily implement adaptively reweighted M-estimators with certifiably correct solvers for the inner WLS subproblems using only fast local optimization over smooth manifolds. Our approach exploits recent work on certifiable factor graph optimization to provide global optimality certificates for the inner WLS subproblems while seamlessly integrating into existing factor graph-based software libraries and workflows. Experimental evaluation on pose-graph optimization and landmark SLAM tasks demonstrates that our adaptively reweighted certifiable estimation approach provides higher-quality estimates than alternative local search-based methods, while scaling tractably to realistic problem sizes.
>
---
#### [new 053] Towards Practical World Model-based Reinforcement Learning for Vision-Language-Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型微调中的高成本和安全风险问题。提出VLA-MBPO框架，提升模型在模拟与真实环境中的性能与效率。**

- **链接: [https://arxiv.org/pdf/2603.20607](https://arxiv.org/pdf/2603.20607)**

> **作者:** Zhilong Zhang; Haoxiang Ren; Yihao Sun; Yifei Sheng; Haonan Wang; Haoxin Lin; Zhichao Wu; Pierre-Luc Bacon; Yang Yu
>
> **摘要:** Vision-Language-Action (VLA) models show strong generalization for robotic control, but finetuning them with reinforcement learning (RL) is constrained by the high cost and safety risks of real-world interaction. Training VLA models in interactive world models avoids these issues but introduces several challenges, including pixel-level world modeling, multi-view consistency, and compounding errors under sparse rewards. Building on recent advances across large multimodal models and model-based RL, we propose VLA-MBPO, a practical framework to tackle these problems in VLA finetuning. Our approach has three key design choices: (i) adapting unified multimodal models (UMMs) for data-efficient world modeling; (ii) an interleaved view decoding mechanism to enforce multi-view consistency; and (iii) chunk-level branched rollout to mitigate error compounding. Theoretical analysis and experiments across simulation and real-world tasks demonstrate that VLA-MBPO significantly improves policy performance and sample efficiency, underscoring its robustness and scalability for real-world robotic deployment.
>
---
#### [new 054] Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决仿真到现实的迁移问题。通过注入关节扭矩扰动，提升仿生机器人在真实环境中的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.21853](https://arxiv.org/pdf/2603.21853)**

> **作者:** Junhyeok Rui Cha; Woohyun Cha; Jaeyong Shin; Donghyeon Kim; Jaeheung Park
>
> **摘要:** This paper proposes a novel alternative to existing sim-to-real methods for training control policies with simulated experiences. Unlike prior methods that typically rely on domain randomization over a fixed finite set of parameters, the proposed approach injects state-dependent perturbations into the input joint torque during forward simulation. These perturbations are designed to simulate a broader spectrum of reality gaps than standard parameter randomization without requiring additional training. By using neural networks as flexible perturbation generators, the proposed method can represent complex, state-dependent uncertainties, such as nonlinear actuator dynamics and contact compliance, that parametric randomization cannot capture. Experimental results demonstrate that the proposed approach enables humanoid locomotion policies to achieve superior robustness against complex, unseen reality gaps in both simulation and real-world deployment.
>
---
#### [new 055] GAPG: Geometry Aware Push-Grasping Synergy for Goal-Oriented Manipulation in Clutter
- **分类: cs.RO**

- **简介: 该论文属于目标导向的机械臂操作任务，解决杂乱环境中抓取困难的问题。通过结合几何信息的推抓协同框架，提升抓取的稳定性和效率。**

- **链接: [https://arxiv.org/pdf/2603.21195](https://arxiv.org/pdf/2603.21195)**

> **作者:** Lijingze Xiao; Jinhong Du; Yang Cong; Supeng Diao; Yu Ren
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Grasping target objects is a fundamental skill for robotic manipulation, but in cluttered environments with stacked or occluded objects, a single-step grasp is often insufficient. To address this, previous work has introduced pushing as an auxiliary action to create graspable space. However, these methods often struggle with both stability and efficiency because they neglect the scene's geometric information, which is essential for evaluating grasp robustness and ensuring that pushing actions are safe and effective. To this end, we propose a geometry-aware push-grasp synergy framework that leverages point cloud data to integrate grasp and push evaluation. Specifically, the grasp evaluation module analyzes the geometric relationship between the gripper's point cloud and the points enclosed within its closing region to determine grasp feasibility and stability. Guided by this, the push evaluation module predicts how pushing actions influence future graspable space, enabling the robot to select actions that reliably transform non-graspable states into graspable ones. By jointly reasoning about geometry in both grasping and pushing, our framework achieves safer, more efficient, and more reliable manipulation in cluttered settings. Our method is extensively tested in simulation and real-world environments in various scenarios. Experimental results demonstrate that our model generalizes well to real-world scenes and unseen objects.
>
---
#### [new 056] An Open Source Computer Vision and Machine Learning Framework for Affordable Life Science Robotic Automation
- **分类: cs.RO**

- **简介: 该论文提出一个开源框架，结合计算机视觉和机器学习，解决实验室自动化任务中的精准操作问题，实现低成本的菌落挑选和液体处理。**

- **链接: [https://arxiv.org/pdf/2603.20465](https://arxiv.org/pdf/2603.20465)**

> **作者:** Zachary Logan; Andrew Dudash; Daniel Negrón
>
> **摘要:** We present an open-source robotic framework that integrates computer vision and machine learning based inverse kinematics to enable low-cost laboratory automation tasks such as colony picking and liquid handling. The system uses a custom trained U-net model for semantic segmentation of microbial cultures, combined with Mixture Density Network for predicating joint angles of a simple 5-DOF robot arm. We evaluated the framework using a modified robot arm, upgraded with a custom liquid handling end-effector. Experimental results demonstrate the framework's feasibility for precise, repeatable operations, with mean positional error below 1 mm and joint angle prediction errors below 4 degrees and colony detection capabilities with IoU score of 0.537 and Dice coefficient of 0.596.
>
---
#### [new 057] Can a Robot Walk the Robotic Dog: Triple-Zero Collaborative Navigation for Heterogeneous Multi-Agent Systems
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人协同导航任务，解决异构机器人在无训练、无先验知识、无仿真环境下的路径规划问题。提出TZPP框架，实现高效、适应性强的协作导航。**

- **链接: [https://arxiv.org/pdf/2603.21723](https://arxiv.org/pdf/2603.21723)**

> **作者:** Yaxuan Wang; Yifan Xiang; Ke Li; Xun Zhang; BoWen Ye; Zhuochen Fan; Fei Wei; Tong Yang
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** We present Triple Zero Path Planning (TZPP), a collaborative framework for heterogeneous multi-robot systems that requires zero training, zero prior knowledge, and zero simulation. TZPP employs a coordinator--explorer architecture: a humanoid robot handles task coordination, while a quadruped robot explores and identifies feasible paths using guidance from a multimodal large language model. We implement TZPP on Unitree G1 and Go2 robots and evaluate it across diverse indoor and outdoor environments, including obstacle-rich and landmark-sparse settings. Experiments show that TZPP achieves robust, human-comparable efficiency and strong adaptability to unseen scenarios. By eliminating reliance on training and simulation, TZPP offers a practical path toward real-world deployment of heterogeneous robot cooperation. Our code and video are provided at: this https URL
>
---
#### [new 058] Directional Mollification for Controlled Smooth Path Generation
- **分类: cs.RO; math.DG**

- **简介: 该论文属于路径生成任务，解决非微分规划输出生成平滑路径的问题。提出方向性卷积方法，实现精确航点插值与曲率控制。**

- **链接: [https://arxiv.org/pdf/2603.21831](https://arxiv.org/pdf/2603.21831)**

> **作者:** Alfredo González-Calvin; Juan F. Jiménez; Héctor García de Marina
>
> **摘要:** Path generation, the problem of producing smooth, executable paths from discrete planning outputs, such as waypoint sequences, is a fundamental step in the control of autonomous robots, industrial robots, and CNC machines, as path following and trajectory tracking controllers impose strict differentiability requirements on their reference inputs to guarantee stability and convergence, particularly for nonholonomic systems. Mollification has been recently proposed as a computationally efficient and analytically tractable tool for path generation, offering formal smoothness and curvature guarantees with advantages over spline interpolation and optimization-based methods. However, this mollification is subject to a fundamental geometric constraint: the smoothed path is confined within the convex hull of the original path, precluding exact waypoint interpolation, even when explicitly required by mission specifications or upstream planners. We introduce directional mollification, a novel operator that resolves this limitation while retaining the analytical tractability of classical mollification. The proposed operator generates infinitely differentiable paths that strictly interpolate prescribed waypoints, converge to the original non-differentiable input with arbitrary precision, and satisfy explicit curvature bounds given by a closed-form expression, addressing the core requirements of path generation for controlled autonomous systems. We further establish a parametric family of path generation operators that contains both classical and directional mollification as special cases, providing a unifying theoretical framework for the systematic generation of smooth, feasible paths from non-differentiable planning outputs.
>
---
#### [new 059] Auction-Based Task Allocation with Energy-Conscientious Trajectory Optimization for AMR Fleets
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究多自主机器人（AMR）在非对称任务空间中的任务分配与轨迹优化问题，提出分阶段框架以实现能量高效调度。**

- **链接: [https://arxiv.org/pdf/2603.21545](https://arxiv.org/pdf/2603.21545)**

> **作者:** Jiachen Li; Soovadeep Bakshi; Jian Chu; Shihao Li; Dongmei Chen
>
> **摘要:** This paper presents a hierarchical two-stage framework for multi-robot task allocation and trajectory optimization in asymmetric task spaces: (1) a sequential auction allocates tasks using closed-form bid functions, and (2) each robot independently solves an optimal control problem for energy-minimal trajectories with a physics-based battery model, followed by a collision avoidance refinement step using pairwise proximity penalties. Event-triggered warm-start rescheduling with bounded trigger frequency handles robot faults, priority arrivals, and energy deviations. Across 505 scenarios with 2-20 robots and up to 100 tasks on three factory layouts, both energy- and distance-based auction variants achieve 11.8% average energy savings over nearest-task allocation, with rescheduling latency under 10 ms. The central finding is that bid-metric performance is regime-dependent: in uniform workspaces, distance bids outperform energy bids by 3.5% (p < 0.05, Wilcoxon) because a 15.7% closed-form approximation error degrades bid ranking accuracy to 87%; however, when workspace friction heterogeneity is sufficient (r < 0.85 energy-distance correlation), a zone-aware energy bid outperforms distance bids by 2-2.4%. These results provide practitioner guidance: use distance bids in near-uniform terrain and energy-aware bids when friction variation is significant.
>
---
#### [new 060] Anatomical Prior-Driven Framework for Autonomous Robotic Cardiac Ultrasound Standard View Acquisition
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于医学影像分析任务，旨在解决心脏超声标准视图获取依赖操作者的问题。通过整合解剖先验与强化学习，提升自主探头调整能力。**

- **链接: [https://arxiv.org/pdf/2603.21134](https://arxiv.org/pdf/2603.21134)**

> **作者:** Zhiyan Cao; Zhengxi Wu; Yiwei Wang; Pei-Hsuan Lin; Li Zhang; Zhen Xie; Huan Zhao; Han Ding
>
> **备注:** Accepted for publication at the IEEE ICRA 2026. 8 pages, 5 figures, 3 tables
>
> **摘要:** Cardiac ultrasound diagnosis is critical for cardiovascular disease assessment, but acquiring standard views remains highly operator-dependent. Existing medical segmentation models often yield anatomically inconsistent results in images with poor textural differentiation between distinct feature classes, while autonomous probe adjustment methods either rely on simplistic heuristic rules or black-box learning. To address these issues, our study proposed an anatomical prior (AP)-driven framework integrating cardiac structure segmentation and autonomous probe adjustment for standard view acquisition. A YOLO-based multi-class segmentation model augmented by a spatial-relation graph (SRG) module is designed to embed AP into the feature pyramid. Quantifiable anatomical features of standard views are extracted. Their priors are fitted to Gaussian distributions to construct probabilistic APs. The probe adjustment process of robotic ultrasound scanning is formalized as a reinforcement learning (RL) problem, with the RL state built from real-time anatomical features and the reward reflecting the AP matching. Experiments validate the efficacy of the framework. The SRG-YOLOv11s improves mAP50 by 11.3% and mIoU by 6.8% on the Special Case dataset, while the RL agent achieves a 92.5% success rate in simulation and 86.7% in phantom experiments.
>
---
#### [new 061] Optimal Solutions for the Moving Target Vehicle Routing Problem with Obstacles via Lazy Branch and Price
- **分类: cs.RO**

- **简介: 该论文研究MT-VRP-O任务，解决多代理在障碍物环境中最优路径规划问题，提出Lazy BPRC算法加速求解。**

- **链接: [https://arxiv.org/pdf/2603.21880](https://arxiv.org/pdf/2603.21880)**

> **作者:** Anoop Bhat; Geordan Gutow; Surya Singh; Zhongqiang Ren; Sivakumar Rathinam; Howie Choset
>
> **摘要:** The Moving Target Vehicle Routing Problem with Obstacles (MT-VRP-O) seeks trajectories for several agents that collectively intercept a set of moving targets. Each target has one or more time windows where it must be visited, and the agents must avoid static obstacles and satisfy speed and capacity constraints. We introduce Lazy Branch-and-Price with Relaxed Continuity (Lazy BPRC), which finds optimal solutions for the MT-VRP-O. Lazy BPRC applies the branch-and-price framework for VRPs, which alternates between a restricted master problem (RMP) and a pricing problem. The RMP aims to select a sequence of target-time window pairings (called a tour) for each agent to follow, from a limited subset of tours. The pricing problem adds tours to the limited subset. Conventionally, solving the RMP requires computing the cost for an agent to follow each tour in the limited subset. Computing these costs in the MT-VRP-O is computationally intensive, since it requires collision-free motion planning between moving targets. Lazy BPRC defers cost computations by solving the RMP using lower bounds on the costs of each tour, computed via motion planning with relaxed continuity constraints. We lazily evaluate the true costs of tours as-needed. We compute a tour's cost by searching for a shortest path on a Graph of Convex Sets (GCS), and we accelerate this search using our continuity relaxation method. We demonstrate that Lazy BPRC runs up to an order of magnitude faster than two ablations.
>
---
#### [new 062] Beyond Scalar Rewards: Distributional Reinforcement Learning with Preordered Objectives for Safe and Reliable Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决自动驾驶中多目标冲突问题。通过引入预序多目标MDP和分布强化学习，提升安全性和可靠性。**

- **链接: [https://arxiv.org/pdf/2603.20230](https://arxiv.org/pdf/2603.20230)**

> **作者:** Ahmed Abouelazm; Jonas Michel; Daniel Bogdoll; Philip Schörner; J. Marius Zöllner
>
> **备注:** First and Second authors contributed equally; Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Autonomous driving involves multiple, often conflicting objectives such as safety, efficiency, and comfort. In reinforcement learning (RL), these objectives are typically combined through weighted summation, which collapses their relative priorities and often yields policies that violate safety-critical constraints. To overcome this limitation, we introduce the Preordered Multi-Objective MDP (Pr-MOMDP), which augments standard MOMDPs with a preorder over reward components. This structure enables reasoning about actions with respect to a hierarchy of objectives rather than a scalar signal. To make this structure actionable, we extend distributional RL with a novel pairwise comparison metric, Quantile Dominance (QD), that evaluates action return distributions without reducing them into a single statistic. Building on QD, we propose an algorithm for extracting optimal subsets, the subset of actions that remain non-dominated under each objective, which allows precedence information to shape both decision-making and training targets. Our framework is instantiated with Implicit Quantile Networks (IQN), establishing a concrete implementation while preserving compatibility with a broad class of distributional RL methods. Experiments in Carla show improved success rates, fewer collisions and off-road events, and deliver statistically more robust policies than IQN and ensemble-IQN baselines. By ensuring policies respect rewards preorder, our work advances safer, more reliable autonomous driving systems.
>
---
#### [new 063] Cross-Modal Reinforcement Learning for Navigation with Degraded Depth Measurements
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决深度传感器失效时的导航问题。通过跨模态学习融合灰度与深度信息，提升在恶劣环境下的导航鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.22182](https://arxiv.org/pdf/2603.22182)**

> **作者:** Omkar Sawant; Luca Zanatta; Grzegorz Malczyk; Kostas Alexis
>
> **备注:** Accepted to the 24th European Control Conference (ECC) 2026
>
> **摘要:** This paper presents a cross-modal learning framework that exploits complementary information from depth and grayscale images for robust navigation. We introduce a Cross-Modal Wasserstein Autoencoder that learns shared latent representations by enforcing cross-modal consistency, enabling the system to infer depth-relevant features from grayscale observations when depth measurements are corrupted. The learned representations are integrated with a Reinforcement Learning-based policy for collision-free navigation in unstructured environments when depth sensors experience degradation due to adverse conditions such as poor lighting or reflective surfaces. Simulation and real-world experiments demonstrate that our approach maintains robust performance under significant depth degradation and successfully transfers to real environments.
>
---
#### [new 064] Current state of the multi-agent multi-view experimental and digital twin rendezvous (MMEDR-Autonomous) framework
- **分类: cs.RO; physics.space-ph**

- **简介: 本文提出MMEDR-Autonomous框架，解决空间目标自主交会对接问题。融合机器学习与控制方法，提升任务自主性与安全性。**

- **链接: [https://arxiv.org/pdf/2603.20575](https://arxiv.org/pdf/2603.20575)**

> **作者:** Logan Banker; Michael Wozniak; Mohanad Alameer; Smriti Nandan Paul; David Meisinger; Grant Baer; Trevor Hunting; Ryan Dunham; Jay Kamdar
>
> **摘要:** As near-Earth resident space objects proliferate, there is an increasing demand for reliable technologies in applications of on-orbit servicing, debris removal, and orbit modification. Rendezvous and docking are critical mission phases for such applications and can benefit from greater autonomy to reduce operational complexity and human workload. Machine learning-based methods can be integrated within the guidance, navigation, and control (GNC) architecture to design a robust rendezvous and docking framework. In this work, the Multi-Agent Multi-View Experimental and Digital Twin Rendezvous (MMEDR-Autonomous) is introduced as a unified framework comprising a learning-based optical navigation network, a reinforcement learning-based guidance approach under ongoing development, and a hardware-in-the-loop testbed. Navigation employs a lightweight monocular pose estimation network with multi-scale feature fusion, trained on realistic image augmentations to mitigate domain shift. The guidance component is examined with emphasis on learning stability, reward design, and systematic hyperparameter tuning under mission-relevant constraints. Prior Control Barrier Function results for Clohessy-Wiltshire dynamics are reviewed as a basis for enforcing safety and operational constraints and for guiding future nonlinear controller design within the MMEDR-Autonomous framework. The MMEDR-Autonomous framework is currently progressing toward integrated experimental validation in multi-agent rendezvous scenarios.
>
---
#### [new 065] Rheos: Modelling Continuous Motion Dynamics in Hierarchical 3D Scene Graphs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于环境建模任务，解决3D场景图动态建模不足的问题。提出Rheos框架，通过连续运动模型增强场景图的导航性。**

- **链接: [https://arxiv.org/pdf/2603.20239](https://arxiv.org/pdf/2603.20239)**

> **作者:** Iacopo Catalano; Francesco Verdoja; Javier Civera; Jorge Peña-Queralta; Julio A. Placed
>
> **摘要:** 3D Scene Graphs (3DSGs) provide hierarchical, multi-resolution abstractions that encode the geometric and semantic structure of an environment, yet their treatment of dynamics remains limited to tracking individual agents. Maps of Dynamics (MoDs) complement this by modeling aggregate motion patterns, but rely on uniform grid discretizations that lack semantic grounding and scale poorly. We present Rheos, a framework that explicitly embeds continuous directional motion models into an additional dynamics layer of a hierarchical 3DSG that enhances the navigational properties of the graph. Each dynamics node maintains a semi-wrapped Gaussian mixture model that captures multimodal directional flow as a principled probability distribution with explicit uncertainty, replacing the discrete histograms used in prior work. To enable online operation, Rheos employs reservoir sampling for bounded-memory observation buffers, parallel per-cell model updates and a principled Bayesian Information Criterion (BIC) sweep that selects the optimal number of mixture components, reducing per-update initialization cost from quadratic to linear in the number of samples. Evaluated across four spatial resolutions in a simulated pedestrian environment, Rheos consistently outperforms the discrete baseline under continuous as well as unfavorable discrete metrics. We release our implementation as open source.
>
---
#### [new 066] ROI-Driven Foveated Attention for Unified Egocentric Representations in Vision-Language-Action Systems
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作系统任务，解决数据收集成本高和跨具身对齐问题。提出ROI驱动方法，生成具身对齐的视觉表示，提升数据复用与迁移能力。**

- **链接: [https://arxiv.org/pdf/2603.20668](https://arxiv.org/pdf/2603.20668)**

> **作者:** Xinhai Sun; Xiang Shi; Menglin Zou; Wenlong Huang
>
> **摘要:** The development of embodied AI systems is increasingly constrained by the availability and structure of physical interaction data. Despite recent advances in vision-language-action (VLA) models, current pipelines suffer from high data collection cost, limited cross-embodiment alignment, and poor transfer from internet-scale visual data to robot control. We propose a region-of-interest (ROI) driven engineering workflow that introduces an egocentric, geometry-grounded data representation. By projecting end-effector poses via forward kinematics (FK) into a single external camera, we derive movement-aligned hand-centric ROIs without requiring wrist-mounted cameras or multi-view systems. Unlike directly downsampling the full frame, ROI is cropped from the original image before resizing, preserving high local information density for contact-critical regions while retaining global context. We present a reproducible pipeline covering calibration, synchronization, ROI generation, deterministic boundary handling, and metadata governance. The resulting representation is embodiment-aligned and viewpoint-normalized, enabling data reuse across heterogeneous robots. We argue that egocentric ROI serves as a practical data abstraction for scalable collection and cross-embodiment learning, bridging internet-scale perception and robot-specific control.
>
---
#### [new 067] IGV-RRT: Prior-Real-Time Observation Fusion for Active Object Search in Changing Environments
- **分类: cs.RO**

- **简介: 该论文属于目标导航任务，解决动态环境中物体移动导致的导航失效问题。提出IGV-RRT框架，融合先验知识与实时视觉语言模型信息，提升目标搜索效率与成功率。**

- **链接: [https://arxiv.org/pdf/2603.21887](https://arxiv.org/pdf/2603.21887)**

> **作者:** Wei Zhang; Ping Gong; Yujie Wang; Minghui Bai; Rongfeng Ye; Yinchuan Wang; Yachao Wang; Leilei Yao; Teng Chen; Chen Sun; Chaoqun Wang
>
> **摘要:** Object Goal Navigation (ObjectNav) in temporally changing indoor environments is challenging because object relocation can invalidate historical scene knowledge. To address this issue, we propose a probabilistic planning framework that combines uncertainty-aware scene priors with online target relevance estimates derived from a Vision Language Model (VLM). The framework contains a dual-layer semantic mapping module and a real-time planner. The mapping module includes an Information Gain Map (IGM) built from a 3D scene graph (3DSG) during prior exploration to model object co-occurrence relations and provide global guidance on likely target regions. It also maintains a VLM score map (VLM-SM) that fuses confidence-weighted semantic observations into the map for local validation of the current scene. Based on these two cues, we develop a planner that jointly exploits information gain and semantic evidence for online decision making. The planner biases tree expansion toward semantically salient regions with high prior likelihood and strong online relevance (IGV-RRT), while preserving kinematic feasibility through gradient-based analysis. Simulation and real-world experiments demonstrate that the proposed method effectively mitigates the impact of object rearrangement, achieving higher search efficiency and success rates than representative baselines in complex indoor environments.
>
---
#### [new 068] FreeArtGS: Articulated Gaussian Splatting Under Free-moving Scenario
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于 articulated object reconstruction 任务，解决自由运动下关节物体重建问题。提出 FreeArtGS 方法，结合分割与联合优化，从单目 RGB-D 视频重建物体结构与纹理。**

- **链接: [https://arxiv.org/pdf/2603.22102](https://arxiv.org/pdf/2603.22102)**

> **作者:** Hang Dai; Hongwei Fan; Han Zhang; Duojin Wu; Jiyao Zhang; Hao Dong
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** The increasing demand for augmented reality and robotics is driving the need for articulated object reconstruction with high scalability. However, existing settings for reconstructing from discrete articulation states or casual monocular videos require non-trivial axis alignment or suffer from insufficient coverage, limiting their applicability. In this paper, we introduce FreeArtGS, a novel method for reconstructing articulated objects under free-moving scenario, a new setting with a simple setup and high scalability. FreeArtGS combines free-moving part segmentation with joint estimation and end-to-end optimization, taking only a monocular RGB-D video as input. By optimizing with the priors from off-the-shelf point-tracking and feature models, the free-moving part segmentation module identifies rigid parts from relative motion under unconstrained capture. The joint estimation module calibrates the unified object-to-camera poses and recovers joint type and axis robustly from part segmentation. Finally, 3DGS-based end-to-end optimization is implemented to jointly reconstruct visual textures, geometry, and joint angles of the articulated object. We conduct experiments on two benchmarks and real-world free-moving articulated objects. Experimental results demonstrate that FreeArtGS consistently excels in reconstructing free-moving articulated objects and remains highly competitive in previous reconstruction settings, proving itself a practical and effective solution for realistic asset generation. The project page is available at: this https URL
>
---
#### [new 069] ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于视频预测任务，旨在解决长时序语义捕捉不足的问题。通过结合视觉语言模型与JEPA框架，提升预测的语义能力和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.22281](https://arxiv.org/pdf/2603.22281)**

> **作者:** Haichao Zhang; Yijiang Li; Shwai He; Tushar Nagarajan; Mingfei Chen; Jianglin Lu; Ang Li; Yun Fu
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Recent progress in latent world models (e.g., V-JEPA2) has shown promising capability in forecasting future world states from video observations. Nevertheless, dense prediction from a short observation window limits temporal context and can bias predictors toward local, low-level extrapolation, making it difficult to capture long-horizon semantics and reducing downstream utility. Vision--language models (VLMs), in contrast, provide strong semantic grounding and general knowledge by reasoning over uniformly sampled frames, but they are not ideal as standalone dense predictors due to compute-driven sparse sampling, a language-output bottleneck that compresses fine-grained interaction states into text-oriented representations, and a data-regime mismatch when adapting to small action-conditioned datasets. We propose a VLM-guided JEPA-style latent world modeling framework that combines dense-frame dynamics modeling with long-horizon semantic guidance via a dual-temporal pathway: a dense JEPA branch for fine-grained motion and interaction cues, and a uniformly sampled VLM \emph{thinker} branch with a larger temporal stride for knowledge-rich guidance. To transfer the VLM's progressive reasoning signals effectively, we introduce a hierarchical pyramid representation extraction module that aggregates multi-layer VLM representations into guidance features compatible with latent prediction. Experiments on hand-manipulation trajectory prediction show that our method outperforms both a strong VLM-only baseline and a JEPA-predictor baseline, and yields more robust long-horizon rollout behavior.
>
---
#### [new 070] Partial Attention in Deep Reinforcement Learning for Safe Multi-Agent Control
- **分类: eess.SY; cs.MA; cs.RO**

- **简介: 该论文属于多智能体安全控制任务，旨在提升自动驾驶车辆在高速公路上的合并表现。通过引入部分注意力机制和综合奖励信号，优化车辆决策，提高安全性和效率。**

- **链接: [https://arxiv.org/pdf/2603.21810](https://arxiv.org/pdf/2603.21810)**

> **作者:** Turki Bin Mohaya; Peter Seiler
>
> **备注:** This work has been accepted for publication in the proceedings of the 2026 American Control Conference (ACC), New Orleans, Louisiana, USA
>
> **摘要:** Attention mechanisms excel at learning sequential patterns by discriminating data based on relevance and importance. This provides state-of-the-art performance in advanced generative artificial intelligence models. This paper applies this concept of an attention mechanism for multi-agent safe control. We specifically consider the design of a neural network to control autonomous vehicles in a highway merging scenario. The environment is modeled as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP). Within a QMIX framework, we include partial attention for each autonomous vehicle, thus allowing each ego vehicle to focus on the most relevant neighboring vehicles. Moreover, we propose a comprehensive reward signal that considers the global objectives of the environment (e.g., safety and vehicle flow) and the individual interests of each agent. Simulations are conducted in the Simulation of Urban Mobility (SUMO). The results show better performance compared to other driving algorithms in terms of safety, driving speed, and reward.
>
---
#### [new 071] DualCoT-VLA: Visual-Linguistic Chain of Thought via Parallel Reasoning for Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DualCoT-VLA模型，解决VLA在复杂任务中的逻辑规划与空间感知问题，通过双通道CoT和并行推理提升性能。**

- **链接: [https://arxiv.org/pdf/2603.22280](https://arxiv.org/pdf/2603.22280)**

> **作者:** Zhide Zhong; Junfeng Li; Junjie He; Haodong Yan; Xin Gong; Guanyi Zhao; Yingjie Cai; Jiantao Gao; Xu Yan; Bingbing Liu; Yingcong Chen; Liuqing Yang; Haoang Li
>
> **摘要:** Vision-Language-Action (VLA) models map visual observations and language instructions directly to robotic actions. While effective for simple tasks, standard VLA models often struggle with complex, multi-step tasks requiring logical planning, as well as precise manipulations demanding fine-grained spatial perception. Recent efforts have incorporated Chain-of-Thought (CoT) reasoning to endow VLA models with a ``thinking before acting'' capability. However, current CoT-based VLA models face two critical limitations: 1) an inability to simultaneously capture low-level visual details and high-level logical planning due to their reliance on isolated, single-modal CoT; 2) high inference latency with compounding errors caused by step-by-step autoregressive decoding. To address these limitations, we propose DualCoT-VLA, a visual-linguistic CoT method for VLA models with a parallel reasoning mechanism. To achieve comprehensive multi-modal reasoning, our method integrates a visual CoT for low-level spatial understanding and a linguistic CoT for high-level task planning. Furthermore, to overcome the latency bottleneck, we introduce a parallel CoT mechanism that incorporates two sets of learnable query tokens, shifting autoregressive reasoning to single-step forward reasoning. Extensive experiments demonstrate that our DualCoT-VLA achieves state-of-the-art performance on the LIBERO and RoboCasa GR1 benchmarks, as well as in real-world platforms.
>
---
#### [new 072] Feasibility of Augmented Reality-Guided Robotic Ultrasound with Cone-Beam CT Integration for Spine Procedures
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于医学影像导航任务，旨在解决脊柱手术中精准定位问题。通过融合AR与超声技术，提升手术准确性与效率。**

- **链接: [https://arxiv.org/pdf/2603.22174](https://arxiv.org/pdf/2603.22174)**

> **作者:** Tianyu Song; Felix Pabst; Feng Li; Yordanka Velikova; Miruna-Alexandra Gafencu; Yuan Bi; Ulrich Eck; Nassir Navab
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Accurate needle placement in spine interventions is critical for effective pain management, yet it depends on reliable identification of anatomical landmarks and careful trajectory planning. Conventional imaging guidance often relies both on CT and X-ray fluoroscopy, exposing patients and staff to high dose of radiation while providing limited real-time 3D feedback. We present an optical see-through augmented reality (OST-AR)-guided robotic system for spine procedures that provides in situ visualization of spinal structures to support needle trajectory planning. We integrate a cone-beam CT (CBCT)-derived 3D spine model which is co-registered with live ultrasound, enabling users to combine global anatomical context with local, real-time imaging. We evaluated the system in a phantom user study involving two representative spine procedures: facet joint injection and lumbar puncture. Sixteen participants performed insertions under two visualization conditions: conventional screen vs. AR. Results show that AR significantly reduces execution time and across-task placement error, while also improving usability, trust, and spatial understanding and lowering cognitive workload. These findings demonstrate the feasibility of AR-guided robotic ultrasound for spine interventions, highlighting its potential to enhance accuracy, efficiency, and user experience in image-guided procedures.
>
---
#### [new 073] Scene Representation using 360° Saliency Graph and its Application in Vision-based Indoor Navigation
- **分类: cs.CV; cs.RO; eess.IV; eess.SP**

- **简介: 该论文属于视觉导航任务，旨在解决室内导航中场景表示不足的问题。提出360°显著性图表示，提升场景定位与导航效果。**

- **链接: [https://arxiv.org/pdf/2603.20353](https://arxiv.org/pdf/2603.20353)**

> **作者:** Preeti Meena; Himanshu Kumar; Sandeep Yadav
>
> **摘要:** A Scene, represented visually using different formats such as RGB-D, LiDAR scan, keypoints, rectangular, spherical, multi-views, etc., contains information implicitly embedded relevant to applications such as scene indexing, vision-based navigation. Thus, these representations may not be efficient for such applications. This paper proposes a novel 360° saliency graph representation of the scenes. This rich representation explicitly encodes the relevant visual, contextual, semantic, and geometric information of the scene as nodes, edges, edge weights, and angular position in the 360° graph. Also, this representation is robust against scene view change and addresses challenges of indoor environments such as varied illumination, occlusions, and shadows as in the case of existing traditional methods. We have utilized this rich and efficient representation for vision-based navigation and compared it with existing navigation methods using 360° scenes. However, these existing methods suffer from limitations of poor scene representation, lacking scene-specific information. This work utilizes the proposed representation first to localize the query scene in the given topological map, and then facilitate 2D navigation by estimating the next required movement directions towards the target destination in the topological map by using the embedded geometric information in the 360° saliency graph. Experimental results demonstrate the efficacy of the proposed 360° saliency graph representation in enhancing both scene localization and vision-based indoor navigation.
>
---
#### [new 074] Transparent Fragments Contour Estimation via Visual-Tactile Fusion for Autonomous Reassembly
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于透明碎片轮廓估计任务，解决其在自主重组中的挑战。通过视觉触觉融合方法，提出框架与数据集，提升轮廓估计与重组效果。**

- **链接: [https://arxiv.org/pdf/2603.20290](https://arxiv.org/pdf/2603.20290)**

> **作者:** Qihao Lin; Borui Chen; Yuping Zhou; Jianing Wu; Yulan Guo; Weishi Zheng; Chongkun Xia
>
> **备注:** 17 pages, 22 figures, submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** The contour estimation of transparent fragments is very important for autonomous reassembly, especially in the fields of precision optical instrument repair, cultural relic restoration, and identification of other precious device broken accidents. Different from general intact transparent objects, the contour estimation of transparent fragments face greater challenges due to strict optical properties, irregular shapes and edges. To address this issue, a general transparent fragments contour estimation framework based on visual-tactile fusion is proposed in this paper. First, we construct the transparent fragment dataset named TransFrag27K, which includes a multiscene synthetic data of broken fragments from multiple types of transparent objects, and a scalable synthetic data generation pipeline. Secondly, we propose a visual grasping position detection network named TransFragNet to identify, locate and segment the sampling grasping position. And, we use a two-finger gripper with Gelsight Mini sensors to obtain reconstructed tactile information of the lateral edge of the fragments. By fusing this tactile information with visual cues, a visual-tactile fusion material classifier is proposed. Inspired by the way humans estimate a fragment's contour combining vision and touch, we introduce a general transparent fragment contour estimation framework based on visual-tactile fusion, demonstrates strong performance in real-world validation. Finally, a multi-dimensional similarity metrics based contour matching and reassembly algorithm is proposed, providing a reproducible benchmark for evaluating visual-tactile contour estimation and fragment reassembly. The experimental results demonstrate the validity of the proposed framework. The dataset and codes are available at this https URL.
>
---
#### [new 075] RoboECC: Multi-Factor-Aware Edge-Cloud Collaborative Deployment for VLA Models
- **分类: cs.DC; cs.LG; cs.RO**

- **简介: 该论文属于边缘-云协同部署任务，旨在解决VLA模型推理成本高和性能不稳定的问题。提出RoboECC框架，通过模型硬件协同分割和网络感知调整提升效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.20711](https://arxiv.org/pdf/2603.20711)**

> **作者:** Zihao Zheng; Hangyu Cao; Jiayu Chen; Sicheng Tian; Chenyue Li; Maoliang Li; Xinhao Sun; Guojie Luo; Xiang Chen
>
> **备注:** This paper has been accepted by IJCNN 2026
>
> **摘要:** Vision-Language-Action (VLA) models are mainstream in embodied intelligence but face high inference costs. Edge-Cloud Collaborative (ECC) deployment offers an effective fix by easing edge-device computing pressure to meet real-time needs. However, existing ECC frameworks are suboptimal for VLA models due to two challenges: (1) Diverse model structures hinder optimal ECC segmentation point identification; (2) Even if the optimal split point is determined, changes in network bandwidth can cause performance drift. To address these issues, we propose a novel ECC deployment framework for various VLA models, termed RoboECC. Specifically, we propose a model-hardware co-aware segmentation strategy to help find the optimal segmentation point for various VLA models. Moreover, we propose a network-aware deployment adjustment approach to adapt to the network fluctuations for maintaining optimal performance. Experiments demonstrate that RoboECC achieves a speedup of up to 3.28x with only 2.55x~2.62x overhead.
>
---
#### [new 076] A Framework for Low-Latency, LLM-driven Multimodal Interaction on the Pepper Robot
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决LLM在Pepper机器人中延迟高、信息丢失及功能受限的问题。提出框架通过端到端语音模型和函数调用实现低延迟、多模态交互。**

- **链接: [https://arxiv.org/pdf/2603.21013](https://arxiv.org/pdf/2603.21013)**

> **作者:** Erich Studerus; Vivienne Jia Zhong; Stephan Vonschallen
>
> **备注:** 4 pages, 2 figures. To appear in Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI '26), Edinburgh, Scotland, March 2026
>
> **摘要:** Despite recent advances in integrating Large Language Models (LLMs) into social robotics, two weaknesses persist. First, existing implementations on platforms like Pepper often rely on cascaded Speech-to-Text (STT)->LLM->Text-to-Speech (TTS) pipelines, resulting in high latency and the loss of paralinguistic information. Second, most implementations fail to fully leverage the LLM's capabilities for multimodal perception and agentic control. We present an open-source Android framework for the Pepper robot that addresses these limitations through two key innovations. First, we integrate end-to-end Speech-to-Speech (S2S) models to achieve low-latency interaction while preserving paralinguistic cues and enabling adaptive intonation. Second, we implement extensive Function Calling capabilities that elevate the LLM to an agentic planner, orchestrating robot actions (navigation, gaze control, tablet interaction) and integrating diverse multimodal feedback (vision, touch, system state). The framework runs on the robot's tablet but can also be built to run on regular Android smartphones or tablets, decoupling development from robot hardware. This work provides the HRI community with a practical, extensible platform for exploring advanced LLM-driven embodied interaction.
>
---
#### [new 077] Does Peer Observation Help? Vision-Sharing Collaboration for Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决多智能体协作问题。通过共享观察提升导航性能，验证了框架有效性。**

- **链接: [https://arxiv.org/pdf/2603.20804](https://arxiv.org/pdf/2603.20804)**

> **作者:** Qunchao Jin; Yiliao Song; Qi Wu
>
> **摘要:** Vision-Language Navigation (VLN) systems are fundamentally constrained by partial observability, as an agent can only accumulate knowledge from locations it has personally visited. As multiple robots increasingly coexist in shared environments, a natural question arises: can agents navigating the same space benefit from each other's observations? In this work, we introduce Co-VLN, a minimalist, model-agnostic framework for systematically investigating whether and how peer observations from concurrently navigating agents can benefit VLN. When independently navigating agents identify common traversed locations, they exchange structured perceptual memory, effectively expanding each agent's receptive field at no additional exploration cost. We validate our framework on the R2R benchmark under two representative paradigms (the learning-based DUET and the zero-shot MapGPT), and conduct extensive analytical experiments to systematically reveal the underlying dynamics of peer observation sharing in VLN. Results demonstrate that vision-sharing enabled model yields substantial performance improvements across both paradigms, establishing a strong foundation for future research in collaborative embodied navigation.
>
---
#### [new 078] CataractSAM-2: A Domain-Adapted Model for Anterior Segment Surgery Segmentation and Scalable Ground-Truth Annotation
- **分类: cs.CV; cs.AI; cs.DB; cs.LG; cs.RO**

- **简介: 该论文属于医学图像分割任务，旨在解决白内障手术视频的实时精准分割及标注难题。工作包括构建领域适配模型CataractSAM-2，并开发交互式标注工具以提升标注效率。**

- **链接: [https://arxiv.org/pdf/2603.21566](https://arxiv.org/pdf/2603.21566)**

> **作者:** Mohammad Eslami; Dhanvinkumar Ganeshkumar; Saber Kazeminasab; Michael G. Morley; Michael V. Boland; Michael M. Lin; John B. Miller; David S. Friedman; Nazlee Zebardast; Lucia Sobrin; Tobias Elze
>
> **摘要:** We present CataractSAM-2, a domain-adapted extension of Meta's Segment Anything Model 2, designed for real-time semantic segmentation of cataract ophthalmic surgery videos with high accuracy. Positioned at the intersection of computer vision and medical robotics, CataractSAM-2 enables precise intraoperative perception crucial for robotic-assisted and computer-guided surgical systems. Furthermore, to alleviate the burden of manual labeling, we introduce an interactive annotation framework that combines sparse prompts with video-based mask propagation. This tool significantly reduces annotation time and facilitates the scalable creation of high-quality ground-truth masks, accelerating dataset development for ocular anterior segment surgeries. We also demonstrate the model's strong zero-shot generalization to glaucoma trabeculectomy procedures, confirming its cross-procedural utility and potential for broader surgical applications. The trained model and annotation toolkit are released as open-source resources, establishing CataractSAM-2 as a foundation for expanding anterior ophthalmic surgical datasets and advancing real-time AI-driven solutions in medical robotics, as well as surgical video understanding.
>
---
#### [new 079] Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于手-物体交互生成任务，解决传统视频缺乏物理信息和遮挡问题。提出Glove2Hand框架，从多模态手套数据生成逼真手部动作，提升手部跟踪与接触估计效果。**

- **链接: [https://arxiv.org/pdf/2603.20850](https://arxiv.org/pdf/2603.20850)**

> **作者:** Xinyu Zhang; Ziyi Kou; Chuan Qin; Mia Huang; Ergys Ristani; Ankit Kumar; Lele Chen; Kun He; Abdeslam Boularias; Li Guan
>
> **备注:** CVPR 2026
>
> **摘要:** Understanding hand-object interaction (HOI) is fundamental to computer vision, robotics, and AR/VR. However, conventional hand videos often lack essential physical information such as contact forces and motion signals, and are prone to frequent occlusions. To address the challenges, we present Glove2Hand, a framework that translates multi-modal sensing glove HOI videos into photorealistic bare hands, while faithfully preserving the underlying physical interaction dynamics. We introduce a novel 3D Gaussian hand model that ensures temporal rendering consistency. The rendered hand is seamlessly integrated into the scene using a diffusion-based hand restorer, which effectively handles complex hand-object interactions and non-rigid deformations. Leveraging Glove2Hand, we create HandSense, the first multi-modal HOI dataset featuring glove-to-hand videos with synchronized tactile and IMU signals. We demonstrate that HandSense significantly enhances downstream bare-hand applications, including video-based contact estimation and hand tracking under severe occlusion.
>
---
#### [new 080] From Singleton Obstacles to Clutter: Translation Invariant Compositional Avoid Sets
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究障碍物避让问题，通过哈密顿-雅可比方法构建不变的避让集，解决复杂环境中的碰撞避免。**

- **链接: [https://arxiv.org/pdf/2603.22146](https://arxiv.org/pdf/2603.22146)**

> **作者:** Prashant Solanki; Jasper Van Beers; Coen De Visser
>
> **摘要:** This paper studies obstacle avoidance under translation invariant dynamics using an avoid-side travel cost Hamilton Jacobi formulation. For running costs that are zero outside an obstacle and strictly negative inside it, we prove that the value function is non-positive everywhere, equals zero exactly outside the avoid set, and is strictly negative exactly on it. Under translation invariance, this yields a reuse principle: the value of any translated obstacle is obtained by translating a single template value function. We show that the pointwise minimum of translated template values exactly characterizes the union of the translated single-obstacle avoid sets and provides a conservative inner certificate of unavoidable collision in clutter. To reduce conservatism, we introduce a blockwise composition framework in which subsets of obstacles are merged and solved jointly. This yields a hierarchy of conservative certificates from singleton reuse to the exact clutter value, together with monotonicity under block merging and an exactness criterion based on the existence of a common clutter avoiding control. The framework is illustrated on a Dubins car example in a repeated clutter field.
>
---
#### [new 081] OrbitStream: Training-Free Adaptive 360-degree Video Streaming via Semantic Potential Fields
- **分类: cs.NI; cs.CV; cs.MM; cs.RO; eess.IV**

- **简介: 该论文提出OrbitStream，解决远程操作中360°视频流的视口预测和码率适应问题，结合语义场景理解和控制理论，实现无需训练的高效流媒体系统。**

- **链接: [https://arxiv.org/pdf/2603.20999](https://arxiv.org/pdf/2603.20999)**

> **作者:** Aizierjiang Aiersilan; Zhangfei Yang
>
> **摘要:** Adaptive 360° video streaming for teleoperation faces dual challenges: viewport prediction under uncertain gaze patterns and bitrate adaptation over volatile wireless channels. While data-driven and Deep Reinforcement Learning (DRL) methods achieve high Quality of Experience (QoE), their "black-box" nature and reliance on training data can limit deployment in safety-critical systems. To address this, we propose OrbitStream, a training-free framework that combines semantic scene understanding with robust control theory. We formulate viewport prediction as a Gravitational Viewport Prediction (GVP) problem, where semantic objects generate potential fields that attract user gaze. Furthermore, we employ a Saturation-Based Proportional-Derivative (PD) Controller for buffer regulation. On object-rich teleoperation traces, OrbitStream achieves a 94.7\% zero-shot viewport prediction accuracy without user-specific profiling, approaching trajectory-extrapolation baselines ($\sim$98.5\%). Across 3,600 Monte Carlo simulations on diverse network traces, OrbitStream yields a mean QoE of 2.71. It ranks second among 12 evaluated algorithms, close to the top-performing BOLA-E (2.80) while outperforming FastMPC (1.84). The system exhibits an average decision latency of 1.01 ms with minimal rebuffering events. By providing competitive QoE with interpretability and zero training overhead, OrbitStream demonstrates that physics-based control, combined with semantic modeling, offers a practical solution for 360° streaming in teleoperation.
>
---
#### [new 082] 6D Robotic OCT Scanning of Curved Tissue Surfaces
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人OCT扫描任务，解决 curved tissue surfaces 扫描中的校准与误差问题。提出六维手眼标定方法，实现精准、一致的扫描。**

- **链接: [https://arxiv.org/pdf/2603.22012](https://arxiv.org/pdf/2603.22012)**

> **作者:** Suresh Guttikonda; Maximilian Neidhardt; Vidas Raudonis; Alexander Schlaefer
>
> **备注:** Accepted at IEEE ISBI 2026
>
> **摘要:** Optical coherence tomography (OCT) is a non-invasive volumetric imaging modality with high spatial and temporal resolution. For imaging larger tissue structures, OCT probes need to be moved to scan the respective area. For handheld scanning, stitching of the acquired OCT volumes requires overlap to register the images. For robotic scanning and stitching, a typical approach is to restrict the motion to translations, as this avoids a full hand-eye calibration, which is complicated by the small field of view of most OCT probes. However, stitching by registration or by translational scanning are limited when curved tissue surfaces need to be scanned. We propose a marker for full six-dimensional hand-eye calibration of a robot mounted OCT probe. We show that the calibration results in highly repeatable estimates of the transformation. Moreover, we evaluate robotic scanning of two phantom surfaces to demonstrate that the proposed calibration allows for consistent scanning of large, curved tissue surfaces. As the proposed approach is not relying on image registration, it does not suffer from a potential accumulation of errors along a scan path. We also illustrate the improvement compared to conventional 3D-translational robotic scanning.
>
---
#### [new 083] MineRobot: A Unified Framework for Kinematics Modeling and Solving of Underground Mining Robots in Virtual Environments
- **分类: cs.GR; cs.RO**

- **简介: 该论文提出MineRobot框架，解决地下采矿机器人在虚拟环境中的运动学建模与求解问题，通过统一表示和优化算法提升实时性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.22055](https://arxiv.org/pdf/2603.22055)**

> **作者:** Shengzhe Hou; Xinming Lu; Tianyu Zhang; Changqing Yan; Xingli Zhang
>
> **摘要:** Underground mining robots are increasingly operated in virtual environments (VEs) for training, planning, and digital-twin applications, where reliable kinematics is essential for avoiding hazardous in-situ trials. Unlike typical open-chain industrial manipulators, mining robots are often closed-chain mechanisms driven by linear actuators and involving planar four-bar linkages, which makes both kinematics modeling and real-time solving challenging. We present \emph{MineRobot}, a unified framework for modeling and solving the kinematics of underground mining robots in VEs. First, we introduce the Mining Robot Description Format (MRDF), a domain-specific representation that parameterizes kinematics for mining robots with native semantics for actuators and loop closures. Second, we develop a topology-processing pipeline that contracts four-bar substructures into generalized joints and, for each actuator, extracts an Independent Topologically Equivalent Path (ITEP), which is classified into one of four canonical types. Third, leveraging ITEP independence, we compose per-type solvers into an actuator-centered sequential forward-kinematics (FK) pipeline. Building on the same decomposition, we formulate inverse kinematics (IK) as a bound-constrained optimization problem and solve it with a Gauss--Seidel-style procedure that alternates actuator-length updates. By converting coupled closed-loop kinematics into a sequence of small topology-aware solves, the framework avoids robot-specific hand derivations and supports efficient computation. Experiments demonstrate that MineRobot provides the real-time performance and robustness required by VE applications.
>
---
## 更新

#### [replaced 001] SVBRD-LLM: Self-Verifying Behavioral Rule Discovery for Autonomous Vehicle Identification
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主车辆行为分析任务，旨在解决AV行为解释性不足的问题。通过框架SVBRD-LLM，自动提取可解释的行为规则，提升AV识别准确率。**

- **链接: [https://arxiv.org/pdf/2511.14977](https://arxiv.org/pdf/2511.14977)**

> **作者:** Xiangyu Li; Tianyi Wang; Junfeng Jiao; Christian Claudel; Zhaomiao Guo
>
> **摘要:** As autonomous vehicles (AVs) are increasingly deployed on public roads, understanding their real-world behaviors is critical for traffic safety analysis and regulatory oversight. However, many data-driven methods lack interpretability and cannot provide verifiable explanations of AV behavior in mixed traffic. This paper proposes SVBRD-LLM, a self-verifying behavioral rule discovery framework that automatically extracts interpretable behavioral rules from real-world traffic videos through zero-shot large language model (LLM) reasoning. The framework first derives vehicle trajectories using YOLOv26-based detection and ByteTrack-based tracking, then computes kinematic features and contextual information. It then employs GPT-5 zero-shot prompting to perform comparative behavioral analysis between AVs and human-driven vehicles (HDVs) across lane-changing and normal driving behaviors, generating 26 structured rule hypotheses that comprises both numerical thresholds and statistical behavioral patterns. These rules are subsequently evaluated through the AV identification task using an independent validation dataset, and iteratively refined through failure case analysis to filter spurious correlations and improve robustness. The resulting rule library contains 20 high-confidence behavioral rules, each including semantic description, quantitative thresholds or behavioral patterns, applicable context, and validation confidence. Experiments conducted on over 1,500 hours of real-world traffic videos from Waymo's commercial operating area demonstrate that the proposed framework achieves 90.0% accuracy and 93.3% F1-score in AV identification, with 98.0% recall. The discovered rules capture key AV traits in smoothness, conservatism, and lane discipline, informing safety assessment, regulatory compliance, and traffic management in mixed traffic. The dataset is available at: svbrd-llm-roadside-video-av.
>
---
#### [replaced 002] StableTracker: Learning to Stably Track Target via Differentiable Simulation
- **分类: cs.RO**

- **简介: 该论文提出StableTracker，解决无人机目标跟踪问题。通过学习控制策略，实现稳定跟随目标，提升跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.14147](https://arxiv.org/pdf/2509.14147)**

> **作者:** Fanxing Li; Shengyang Wang; Fangyu Sun; Shuyu Wu; Dexin Zuo; Yufei Yan; Wenxian Yu; Danping Zou
>
> **摘要:** Existing FPV object tracking methods heavily rely on handcrafted modular pipelines, which incur high onboard computation and cumulative errors. While learning-based approaches have mitigated computational delays, most still generate only high-level trajectories (position and yaw). This loose coupling with a separate controller sacrifices precise attitude control; consequently, even if target is localized precisely, accurate target estimation does not ensure that the body-fixed camera is consistently oriented toward the target, it still probably degrades and loses target when tracking high-maneuvering target. To address these challenges, we present StableTracker, a learning-based control policy that enables quadrotors to robustly follow a moving target from arbitrary viewpoints. The policy is trained using backpropagation-through-time via differentiable simulation, allowing the quadrotor to keep a fixed relative distance while maintaining the target at the center of the visual field in both horizontal and vertical directions, thereby functioning as an autonomous aerial camera. We compare StableTracker against state-of-the-art traditional algorithms and learning baselines. Simulation results demonstrate superior accuracy, stability, and generalization across varying safe distances, trajectories, and target velocities. Furthermore, real-world experiments on a quadrotor with an onboard computer validate the practicality of the proposed approach.
>
---
#### [replaced 003] Physics-Informed Policy Optimization via Analytic Dynamics Regularization
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决机器人控制中策略样本效率低和物理不一致的问题。通过引入物理约束的正则化方法，提升策略优化的效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.14469](https://arxiv.org/pdf/2603.14469)**

> **作者:** Namai Chandra; Liu Mohan; Zhihao Gu; Lin Wang
>
> **备注:** 11 pages, 8 figures
>
> **摘要:** Reinforcement learning (RL) has achieved strong performance in robotic control; however, state-of-the-art policy learning methods, such as actor-critic methods, still suffer from high sample complexity and often produce physically inconsistent actions. This limitation stems from neural policies implicitly rediscovering complex physics from data alone, despite accurate dynamics models being readily available in simulators. In this paper, we introduce a novel physics-informed RL framework, called PIPER, that seamlessly integrates physical constraints directly into neural policy optimization with analytical soft physics constraints. At the core of our method is the integration of a differentiable Lagrangian residual as a regularization term within the actor's objective. This residual, extracted from a robot's simulator description, subtly biases policy updates towards dynamically consistent solutions. Crucially, this physics integration is realized through an additional loss term during policy optimization, requiring no alterations to existing simulators or core RL algorithms. Extensive experiments demonstrate that our method significantly improves learning efficiency, stability, and control accuracy, establishing a new paradigm for efficient and physically consistent robotic control.
>
---
#### [replaced 004] Data Scaling for Navigation in Unknown Environments
- **分类: cs.RO**

- **简介: 该论文研究端到端视觉导航任务，解决模仿学习策略在未知环境中的泛化问题。通过大规模数据实验，发现数据多样性比数量更重要，提出有效提升导航性能的方法。**

- **链接: [https://arxiv.org/pdf/2601.09444](https://arxiv.org/pdf/2601.09444)**

> **作者:** Lauri Suomela; Naoki Takahata; Sasanka Kuruppu Arachchige; Harry Edelman; Joni-Kristian Kämäräinen
>
> **备注:** Robotics and Automation Letters (RA-L) 2026
>
> **摘要:** Generalization of imitation-learned navigation policies to environments unseen in training remains a major challenge. We address this by conducting the first large-scale study of how data quantity and data diversity affect real-world generalization in end-to-end, map-free visual navigation. Using a curated 4,565-hour crowd-sourced dataset collected across 161 locations in 35 countries, we train policies for point goal navigation and evaluate their closed-loop control performance on sidewalk robots operating in four countries, covering 125 km of autonomous driving. Our results show that large-scale training data enables zero-shot navigation in unknown environments, approaching the performance of policies trained with environment-specific demonstrations. Critically, we find that data diversity is far more important than data quantity. Doubling the number of geographical locations in a training set decreases navigation errors by ~15%, while performance benefit from adding data from existing locations saturates with very little data. We also observe that, with noisy crowd-sourced data, simple regression-based models outperform generative and sequence-based architectures. We release our policies, evaluation setup and example videos at this https URL.
>
---
#### [replaced 005] A Real-Time System for Scheduling and Managing UAV Delivery in Urban Areas
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机调度任务，旨在解决城市物流中无人机高效配送问题。提出一种实时管理系统，结合AGVs、UAVs和地面人员协同作业，提升配送效率。**

- **链接: [https://arxiv.org/pdf/2412.11590](https://arxiv.org/pdf/2412.11590)**

> **作者:** Han Liu; Tian Liu; Kai Huang
>
> **备注:** ROBIO 2025
>
> **摘要:** As urban logistics demand continues to grow, UAV delivery has become a key solution to improve delivery efficiency, reduce traffic congestion, and lower logistics costs. However, to fully leverage the potential of UAV delivery networks, efficient swarm scheduling and management are crucial. In this paper, we propose a real-time scheduling and management system based on the ``Airport-Unloading Station" model, aiming to bridge the gap between high-level scheduling algorithms and low-level execution systems. This system, acting as middleware, accurately translates the requirements from the scheduling layer into specific execution instructions, ensuring that the scheduling algorithms perform effectively in real-world environments. Additionally, we implement three collaborative scheduling schemes involving autonomous ground vehicles (AGVs), unmanned aerial vehicles (UAVs), and ground staff to further optimize overall delivery efficiency. Through extensive experiments, this study demonstrates the rationality and feasibility of the proposed management system, providing practical solution for the commercial application of UAVs delivery in urban. Code: this https URL
>
---
#### [replaced 006] A Tactile-based Interactive Motion Planner for Robots in Unknown Cluttered Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动规划任务，解决未知杂乱环境中路径规划困难的问题。通过交互式感知-运动框架，增强机器人对接触模型的建模能力，提升路径规划的安全性和效率。**

- **链接: [https://arxiv.org/pdf/2509.16963](https://arxiv.org/pdf/2509.16963)**

> **作者:** Chengjin Wang; Yanmin Zhou; Zheng Yan; Feng Luan; Runjie Shen; Hongrui Sang; Zhipeng Wang; Bin He
>
> **摘要:** In unknown cluttered environments with densely stacked objects, the free-motion space is extremely barren, posing significant challenges to motion planners. Collision-free planning methods often suffer from catastrophic failures due to unexpected collisions and motion obstructions. To address this issue, this paper proposes an interactive motion planning framework (I-MP), based on a perception-motion loop. This framework empowers robots to autonomously model and reason about contact models, which in turn enables safe expansion of the free-motion space. Specifically, the robot utilizes multimodal tactile perception to acquire stimulus-response signal pairs. This enables real-time identification of objects' mechanical properties and the subsequent construction of contact models. These models are integrated as computational constraints into a reactive planner. Based on fixed-point theorems, the planner computes the spatial state toward the target in real time, thus avoiding the computational burden associated with extrapolating on high-dimensional interaction models. Furthermore, high-dimensional interaction features are linearly superposed in Cartesian space in the form of energy, and the controller achieves trajectory tracking by solving the energy gradient from the current state to the planned state. The experimental results showed that at cruising speeds ranging from 0.01 to 0.07 $m/s$, the robot's initial contact force with objects remained stable at 1.0 +- 0.7 N. In the cabinet scenario test where collision-free trajectories were unavailable, I-MP expanded the free motion space by 37.5 % through active interaction, successfully completing the environmental exploration task.
>
---
#### [replaced 007] Graph-of-Constraints Model Predictive Control for Reactive Multi-agent Task and Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于多智能体任务与运动规划（TAMP）领域，解决动态任务分配和部分有序约束下的协同问题。提出GoC-MPC框架，实现高效、鲁棒的多智能体操作。**

- **链接: [https://arxiv.org/pdf/2603.18400](https://arxiv.org/pdf/2603.18400)**

> **作者:** Anastasios Manganaris; Jeremy Lu; Ahmed H. Qureshi; Suresh Jagannathan
>
> **备注:** 8 main content pages, 4 main content figures, camera ready version submitted to IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Sequences of interdependent geometric constraints are central to many multi-agent Task and Motion Planning (TAMP) problems. However, existing methods for handling such constraint sequences struggle with partially ordered tasks and dynamic agent assignments. They typically assume static assignments and cannot adapt when disturbances alter task allocations. To overcome these limitations, we introduce Graph-of-Constraints Model Predictive Control (GoC-MPC), a generalized sequence-of-constraints framework integrated with MPC. GoC-MPC naturally supports partially ordered tasks, dynamic agent coordination, and disturbance recovery. By defining constraints over tracked 3D keypoints, our method robustly solves diverse multi-agent manipulation tasks-coordinating agents and adapting online from visual observations alone, without relying on training data or environment models. Experiments demonstrate that GoC-MPC achieves higher success rates, significantly faster TAMP computation, and shorter overall paths compared to recent baselines, establishing it as an efficient and robust solution for multi-agent manipulation under real-world disturbances. Our supplementary video and code can be found at this https URL .
>
---
#### [replaced 008] Foundation Models for Trajectory Planning in Autonomous Driving: A Review of Progress and Open Challenges
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶轨迹规划任务，探讨基于基础模型的方法解决传统设计的不足，综述37种方法并分析其优缺点。**

- **链接: [https://arxiv.org/pdf/2512.00021](https://arxiv.org/pdf/2512.00021)**

> **作者:** Kemal Oksuz; Alexandru Buburuzan; Anthony Knittel; Yuhan Yao; Puneet K. Dokania
>
> **备注:** Accepted to TMLR (Survey Certification)
>
> **摘要:** The emergence of multi-modal foundation models has markedly transformed the technology for autonomous driving, shifting away from conventional and mostly hand-crafted design choices towards unified, foundation-model-based approaches, capable of directly inferring motion trajectories from raw sensory inputs. This new class of methods can also incorporate natural language as an additional modality, with Vision-Language-Action (VLA) models serving as a representative example. In this review, we provide a comprehensive examination of such methods through a unifying taxonomy to critically evaluate their architectural design choices, methodological strengths, and their inherent capabilities and limitations. Our survey covers 37 recently proposed approaches that span the landscape of trajectory planning with foundation models. Furthermore, we assess these approaches with respect to the openness of their source code and datasets, offering valuable information to practitioners and researchers. We provide an accompanying webpage that catalogues the methods based on our taxonomy, available at: this https URL
>
---
#### [replaced 009] MSACL: Multi-Step Actor-Critic Learning with Lyapunov Certificates for Exponentially Stabilizing Control
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文提出MSACL方法，用于稳定控制任务，解决高维环境中RL效率与效果问题，通过引入Lyapunov证书和多步样本提升稳定性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.24955](https://arxiv.org/pdf/2512.24955)**

> **作者:** Yongwei Zhang; Yuanzhe Xing; Quanyi Liang; Quan Quan; Zhikun She
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** For stabilizing control tasks, model-free reinforcement learning (RL) approaches face numerous challenges, particularly regarding the issues of effectiveness and efficiency in complex high-dimensional environments with limited training data. To address these challenges, we propose Multi-Step Actor-Critic Learning with Lyapunov Certificates (MSACL), a novel approach that integrates exponential stability into off-policy maximum entropy reinforcement learning (MERL). In contrast to existing RL-based approaches that depend on elaborate reward engineering and single-step constraints, MSACL adopts intuitive reward design and exploits multi-step samples to enable exploratory actor-critic learning. Specifically, we first introduce Exponential Stability Labels (ESLs) to categorize training samples and propose a $\lambda$-weighted aggregation mechanism to learn Lyapunov certificates. Based on these certificates, we further design a stability-aware advantage function to guide policy optimization, thereby promoting rapid Lyapunov descent and robust state convergence. We evaluate MSACL across six benchmarks, comprising four stabilizing and two high-dimensional tracking tasks. Experimental results demonstrate its consistent performance improvements over both standard RL baselines and state-of-the-art Lyapunov-based RL algorithms. Beyond rapid convergence, MSACL exhibits robustness against environmental uncertainties and generalization to unseen reference signals. The source code and benchmarking environments are available at \href{this https URL}{this https URL}.
>
---
#### [replaced 010] RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboFAC框架，解决VLA在开放场景中的故障诊断与恢复问题。通过构建大规模故障数据集，开发轻量级模型提升故障分析与纠正能力。**

- **链接: [https://arxiv.org/pdf/2505.12224](https://arxiv.org/pdf/2505.12224)**

> **作者:** Zewei Ye; Weifeng Lu; Minghao Ye; Tao Lin; Shuo Yang; Junchi Yan; Bo Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have recently advanced robotic manipulation by translating natural-language instructions and visual observations into control actions. However, existing VLAs are primarily trained on successful expert demonstrations and lack structured supervision for failure diagnosis and recovery, limiting robustness in open-world scenarios. To address this limitation, we propose the Robotic Failure Analysis and Correction (RoboFAC) framework. We construct a large-scale failure-centric dataset comprising 9,440 erroneous manipulation trajectories and 78,623 QA pairs across 53 scenes in both simulation and real-world environments, with systematically categorized failure types. Leveraging this dataset, we develop a lightweight multimodal model specialized for task understanding, failure analysis, and failure correction, enabling efficient local deployment while remaining competitive with large proprietary models. Experimental results demonstrate that RoboFAC achieves a 34.1% higher failure analysis accuracy compared to GPT-4o. Furthermore, we integrated RoboFAC as an external supervisor in a real-world VLA control pipeline, yielding a 29.1% relative improvement across four tasks while significantly reducing latency relative to GPT-4o. These results demonstrate that RoboFAC enables systematic failure diagnosis and recovery, significantly enhancing VLA recovery capabilities. Our model and dataset are publicly available at this https URL.
>
---
#### [replaced 011] From 2D to 3D terrain-following area coverage path planning
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于3D地形跟随区域覆盖路径规划任务，解决农机在复杂地形中高效覆盖的问题。通过生成等距且保持特定高度的路径，提升作业效率与适应性。**

- **链接: [https://arxiv.org/pdf/2601.00614](https://arxiv.org/pdf/2601.00614)**

> **作者:** Mogens Plessen
>
> **备注:** 6 pages, 10 figures, 1 table, IEEE ICARSC 2026
>
> **摘要:** An algorithm for 3D terrain-following area coverage path planning is presented. Multiple adjacent paths are generated that are (i) locally apart from each other by a distance equal to the working width of a machinery, while (ii) simultaneously floating at a projection distance equal to a specific working height above the terrain. The complexities of the algorithm in comparison to its 2D equivalent are highlighted. These include uniformly spaced elevation data generation using an Inverse Distance Weighting-approach and a local search. Area coverage path planning results for real-world 3D data within an agricultural context are presented to validate the algorithm.
>
---
#### [replaced 012] LAOF: Robust Latent Action Learning with Optical Flow Constraints
- **分类: cs.RO**

- **简介: 该论文提出LAOF方法，解决视频中潜在动作学习问题，通过光流约束提升鲁棒性，在标签稀缺情况下表现优异。**

- **链接: [https://arxiv.org/pdf/2511.16407](https://arxiv.org/pdf/2511.16407)**

> **作者:** Xizhou Bu; Jiexi Lyu; Fulei Sun; Ruichen Yang; Zhiqiang Ma; Wei Li
>
> **备注:** CVPR 2026; Project page: this https URL
>
> **摘要:** Learning latent actions from large-scale videos is crucial for the pre-training of scalable embodied foundation models, yet existing methods often struggle with action-irrelevant distractors. Although incorporating action supervision can alleviate these distractions, its effectiveness is restricted by the scarcity of available action labels. Optical flow represents pixel-level motion between consecutive frames, naturally suppressing background elements and emphasizing moving objects. Motivated by this, we propose robust Latent Action learning with Optical Flow constraints, called LAOF, a pseudo-supervised framework that leverages the agent's optical flow as an action-driven signal to learn latent action representations robust to distractors. Experimental results show that the latent representations learned by LAOF outperform existing methods on downstream imitation learning and reinforcement learning tasks. This superior performance arises from optical flow constraints, which substantially stabilize training and improve the quality of latent representations under extremely label-scarce conditions, while remaining effective as the proportion of action labels increases to 10 percent. Importantly, even without action supervision, LAOF matches or surpasses action-supervised methods trained with 1 percent of action labels.
>
---
#### [replaced 013] Causal World Modeling for Robot Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决长期操作与数据效率问题。提出LingBot-VA框架，结合视觉与动作建模，实现高效、泛化的机器人控制。**

- **链接: [https://arxiv.org/pdf/2601.21998](https://arxiv.org/pdf/2601.21998)**

> **作者:** Lin Li; Qihang Zhang; Yiming Luo; Shuai Yang; Ruilin Wang; Fei Han; Mingrui Yu; Zelin Gao; Nan Xue; Xing Zhu; Yujun Shen; Yinghao Xu
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** This work highlights that video world modeling, alongside vision-language pre-training, establishes a fresh and independent foundation for robot learning. Intuitively, video world models provide the ability to imagine the near future by understanding the causality between actions and visual dynamics. Inspired by this, we introduce LingBot-VA, an autoregressive diffusion framework that learns frame prediction and policy execution simultaneously. Our model features three carefully crafted designs: (1) a shared latent space, integrating vision and action tokens, driven by a Mixture-of-Transformers (MoT) architecture, (2) a closed-loop rollout mechanism, allowing for ongoing acquisition of environmental feedback with ground-truth observations, (3) an asynchronous inference pipeline, parallelizing action prediction and motor execution to support efficient control. We evaluate our model on both simulation benchmarks and real-world scenarios, where it shows significant promise in long-horizon manipulation, data efficiency in post-training, and strong generalizability to novel configurations. The code and model are made publicly available to facilitate the community.
>
---
#### [replaced 014] RE-SAC: Disentangling aleatoric and epistemic risks in bus fleet control: A stable and robust ensemble DRL approach
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决公交调度中的不确定性问题。针对交通和乘客需求的随机性，提出RE-SAC方法分离aleatoric与epistemic风险，提升策略稳定性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.18396](https://arxiv.org/pdf/2603.18396)**

> **作者:** Yifan Zhang; Liang Zheng
>
> **摘要:** Bus holding control is challenging due to stochastic traffic and passenger demand. While deep reinforcement learning (DRL) shows promise, standard actor-critic algorithms suffer from Q-value instability in volatile environments. A key source of this instability is the conflation of two distinct uncertainties: aleatoric uncertainty (irreducible noise) and epistemic uncertainty (data insufficiency). Treating these as a single risk leads to value underestimation in noisy states, causing catastrophic policy collapse. We propose a robust ensemble soft actor-critic (RE-SAC) framework to explicitly disentangle these uncertainties. RE-SAC applies Integral Probability Metric (IPM)-based weight regularization to the critic network to hedge against aleatoric risk, providing a smooth analytical lower bound for the robust Bellman operator without expensive inner-loop perturbations. To address epistemic risk, a diversified Q-ensemble penalizes overconfident value estimates in sparsely covered regions. This dual mechanism prevents the ensemble variance from misidentifying noise as a data gap, a failure mode identified in our ablation study. Experiments in a realistic bidirectional bus corridor simulation demonstrate that RE-SAC achieves the highest cumulative reward (approx. -0.4e6) compared to vanilla SAC (-0.55e6). Mahalanobis rareness analysis confirms that RE-SAC reduces Oracle Q-value estimation error by up to 62% in rare out-of-distribution states (MAE of 1647 vs. 4343), demonstrating superior robustness under high traffic variability.
>
---
#### [replaced 015] Parallel, Asymptotically Optimal Algorithms for Moving Target Traveling Salesman Problems
- **分类: cs.RO**

- **简介: 该论文研究移动目标旅行商问题（MT-TSP），旨在设计高效算法解决动态目标拦截路径规划问题。提出IRG框架及两种并行算法，提升求解效率与精度。**

- **链接: [https://arxiv.org/pdf/2509.08743](https://arxiv.org/pdf/2509.08743)**

> **作者:** Anoop Bhat; Geordan Gutow; Bhaskar Vundurthy; Zhongqiang Ren; Sivakumar Rathinam; Howie Choset
>
> **摘要:** The Moving Target Traveling Salesman Problem (MT-TSP) seeks a trajectory that intercepts several moving targets, within a particular time window for each target. When generic nonlinear target trajectories or kinematic constraints on the agent are present, no prior algorithm guarantees convergence to an optimal MT-TSP solution. Therefore, we introduce the Iterated Random Generalized (IRG) TSP framework. The idea behind IRG is to alternate between randomly sampling a set of agent configuration-time points, corresponding to interceptions of targets, and finding a sequence of interception points by solving a generalized TSP (GTSP). This alternation asymptotically converges to the optimum. We introduce two parallel algorithms within the IRG framework. The first algorithm, IRG-PGLNS, solves GTSPs using PGLNS, our parallelized extension of state-of-the-art solver GLNS. The second algorithm, Parallel Communicating GTSPs (PCG), solves GTSPs for several sets of points simultaneously. We present numerical results for three MT-TSP variants: one where intercepting a target only requires coming within a particular distance, another where the agent is a variable-speed Dubins car, and a third where the agent is a robot arm. We show that IRG-PGLNS and PCG converge faster than a baseline based on prior work. We further validate our framework with physical robot experiments.
>
---
#### [replaced 016] Scalable Multi-Task Learning through Spiking Neural Networks with Adaptive Task-Switching Policy for Intelligent Autonomous Agents
- **分类: cs.NE; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于多任务学习领域，旨在解决自主智能体在资源受限下多任务学习中的任务干扰问题。提出SwitchMT方法，通过自适应任务切换策略实现高效、可扩展的多任务学习。**

- **链接: [https://arxiv.org/pdf/2504.13541](https://arxiv.org/pdf/2504.13541)**

> **作者:** Rachmad Vidya Wicaksana Putra; Avaneesh Devkota; Muhammad Shafique
>
> **备注:** Accepted at the 63rd ACM/IEEE Design Automation Conference (DAC), July 26-29, 2026 in Long Beach, CA, USA. [Codes: this https URL]
>
> **摘要:** Training resource-constrained autonomous agents on multiple tasks simultaneously is crucial for adapting to diverse real-world environments. Recent works employ reinforcement learning (RL) approach, but they still suffer from sub-optimal multi-task performance due to task interference. State-of-the-art works employ Spiking Neural Networks (SNNs) to improve RL-based multi-task learning and enable low-power/energy operations through network enhancements and spike-driven data stream processing. However, they rely on fixed task-switching intervals during its training, thus limiting its performance and scalability. To address this, we propose SwitchMT, a novel methodology that employs adaptive task-switching for effective, scalable, and simultaneous multi-task learning. SwitchMT employs the following key ideas: (1) leveraging a Deep Spiking Q-Network with active dendrites and dueling structure, that utilizes task-specific context signals to create specialized sub-networks; and (2) devising an adaptive task-switching policy that leverages both rewards and internal dynamics of the network parameters. Experimental results demonstrate that SwitchMT achieves competitive scores in multiple Atari games (i.e., Pong: -8.8, Breakout: 5.6, and Enduro: 355.2) and longer game episodes as compared to the state-of-the-art. These results also highlight the effectiveness of SwitchMT methodology in addressing task interference without increasing the network complexity, enabling intelligent autonomous agents with scalable multi-task learning capabilities.
>
---
#### [replaced 017] Efficient and Reliable Teleoperation through Real-to-Sim-to-Real Shared Autonomy
- **分类: cs.RO**

- **简介: 该论文属于机器人遥操作任务，解决真实环境中精细操作效率低、易出错的问题。通过构建真实到仿真再到真实的共享自主框架，提升操作性能。**

- **链接: [https://arxiv.org/pdf/2603.17016](https://arxiv.org/pdf/2603.17016)**

> **作者:** Shuo Sha; Yixuan Wang; Binghao Huang; Antonio Loquercio; Yunzhu Li
>
> **备注:** Project Page: this https URL
>
> **摘要:** Fine-grained, contact-rich teleoperation remains slow, error-prone, and unreliable in real-world manipulation tasks, even for experienced operators. Shared autonomy offers a promising way to improve performance by combining human intent with automated assistance, but learning effective assistance in simulation requires a faithful model of human behavior, which is difficult to obtain in practice. We propose a real-to-sim-to-real shared autonomy framework that augments human teleoperation with learned corrective behaviors, using a simple yet effective k-nearest-neighbor (kNN) human surrogate to model operator actions in simulation. The surrogate is fit from less than five minutes of real-world teleoperation data and enables stable training of a residual copilot policy with model-free reinforcement learning. The resulting copilot is deployed to assist human operators in real-world fine-grained manipulation tasks. Through simulation experiments and a user study with sixteen participants on industry-relevant tasks, including nut threading, gear meshing, and peg insertion, we show that our system improves task success for novice operators and execution efficiency for experienced operators compared to direct teleoperation and shared-autonomy baselines that rely on expert priors or behavioral-cloning pilots. In addition, copilot-assisted teleoperation produces higher-quality demonstrations for downstream imitation learning.
>
---
#### [replaced 018] AERO-MPPI: Anchor-Guided Ensemble Trajectory Optimization for Agile Mapless Drone Navigation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主导航任务，解决复杂3D环境中无人机的快速、安全路径规划问题。提出AERO-MPPI框架，融合感知与规划，实现实时高效导航。**

- **链接: [https://arxiv.org/pdf/2509.17340](https://arxiv.org/pdf/2509.17340)**

> **作者:** Xin Chen; Rui Huang; Longbin Tang; Lin Zhao
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** Agile mapless navigation in cluttered 3D environments poses significant challenges for autonomous drones. Conventional mapping-planning-control pipelines incur high computational cost and propagate estimation errors. We present AERO-MPPI, a fully GPU-accelerated framework that unifies perception and planning through an anchor-guided ensemble of Model Predictive Path Integral (MPPI) optimizers. Specifically, we design a multi-resolution LiDAR point-cloud representation that rapidly extracts spatially distributed "anchors" as look-ahead intermediate endpoints, from which we construct polynomial trajectory guides to explore distinct homotopy path classes. At each planning step, we run multiple MPPI instances in parallel and evaluate them with a two-stage multi-objective cost that balances collision avoidance and goal reaching. Implemented entirely with NVIDIA Warp GPU kernels, AERO-MPPI achieves real-time onboard operation and mitigates the local-minima failures of single-MPPI approaches. Extensive simulations in forests, verticals, and inclines demonstrate sustained reliable flight above 7 m/s, with success rates above 80% and smoother trajectories compared to state-of-the-art baselines. Real-world experiments on a LiDAR-equipped quadrotor with NVIDIA Jetson Orin NX 16G confirm that AERO-MPPI runs in real time onboard and consistently achieves safe, agile, and robust flight in complex cluttered environments. Code is available at this https URL.
>
---
#### [replaced 019] SPOT: Point Cloud Based Stereo Visual Place Recognition for Similar and Opposing Viewpoints
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉定位任务，解决相似与相反视角下的场景识别问题。提出SPOT方法，利用立体视觉里程计结构，通过双距离矩阵匹配实现高效准确的VPR。**

- **链接: [https://arxiv.org/pdf/2404.12339](https://arxiv.org/pdf/2404.12339)**

> **作者:** Spencer Carmichael; Rahul Agrawal; Ram Vasudevan; Katherine A. Skinner
>
> **备注:** Expanded version with added appendix. Published in ICRA 2024. Project page: this https URL
>
> **摘要:** Recognizing places from an opposing viewpoint during a return trip is a common experience for human drivers. However, the analogous robotics capability, visual place recognition (VPR) with limited field of view cameras under 180 degree rotations, has proven to be challenging to achieve. To address this problem, this paper presents Same Place Opposing Trajectory (SPOT), a technique for opposing viewpoint VPR that relies exclusively on structure estimated through stereo visual odometry (VO). The method extends recent advances in lidar descriptors and utilizes a novel double (similar and opposing) distance matrix sequence matching method. We evaluate SPOT on a publicly available dataset with 6.7-7.6 km routes driven in similar and opposing directions under various lighting conditions. The proposed algorithm demonstrates remarkable improvement over the state-of-the-art, achieving up to 91.7% recall at 100% precision in opposing viewpoint cases, while requiring less storage than all baselines tested and running faster than all but one. Moreover, the proposed method assumes no a priori knowledge of whether the viewpoint is similar or opposing, and also demonstrates competitive performance in similar viewpoint cases.
>
---
#### [replaced 020] Semi-Infinite Programming for Collision-Avoidance in Optimal and Model Predictive Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人路径规划任务，解决碰撞避免问题。通过半无限规划方法，处理环境点与机器人的无限约束，实现高效安全导航。**

- **链接: [https://arxiv.org/pdf/2508.12335](https://arxiv.org/pdf/2508.12335)**

> **作者:** Yunfan Gao; Florian Messerer; Niels van Duijkeren; Rashmi Dabir; Moritz Diehl
>
> **备注:** 20 pages, 17 figures
>
> **摘要:** This paper presents a novel approach for collision avoidance in optimal and model predictive control, in which the environment is represented by a large number of points and the robot as a union of padded polygons. The conditions that none of the points shall collide with the robot can be written in terms of an infinite number of constraints per obstacle point. We show that the resulting semi-infinite programming (SIP) optimal control problem (OCP) can be efficiently tackled through a combination of two methods: local reduction and an external active-set method. Specifically, this involves iteratively identifying the closest point obstacles, determining the lower-level distance minimizer among all feasible robot shape parameters, and solving the upper-level finitely-constrained subproblems. In addition, this paper addresses robust collision avoidance in the presence of ellipsoidal state uncertainties. Enforcing constraint satisfaction over all possible uncertainty realizations extends the dimension of constraint infiniteness. The infinitely many constraints arising from translational uncertainty are handled by local reduction together with the robot shape parameterization, while rotational uncertainty is addressed via a backoff reformulation. A controller implemented based on the proposed method is demonstrated on a real-world robot running at 20Hz, enabling fast and collision-free navigation in tight spaces. An application to 3D collision avoidance is also demonstrated in simulation.
>
---
#### [replaced 021] Contractive Diffusion Policies: Robust Action Diffusion via Contractive Score-Based Sampling with Differential Equations
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出一种名为收缩扩散策略（CDP）的方法，用于改进离线策略学习中的动作生成。针对扩散策略在连续控制中因误差导致的不稳定性问题，CDP通过引入收缩行为增强鲁棒性，减少动作方差，提升性能。**

- **链接: [https://arxiv.org/pdf/2601.01003](https://arxiv.org/pdf/2601.01003)**

> **作者:** Amin Abyaneh; Charlotte Morissette; Mohamad H. Danesh; Anas El Houssaini; David Meger; Gregory Dudek; Hsiu-Chin Lin
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Diffusion policies have emerged as powerful generative models for offline policy learning, whose sampling process can be rigorously characterized by a score function guiding a stochastic differential equation (SDE). However, the same score-based SDE modeling that grants diffusion policies the flexibility to learn diverse behavior also incurs solver and score-matching errors, large data requirements, and inconsistencies in action generation. While less critical in image generation, these inaccuracies compound and lead to failure in continuous control settings. We introduce contractive diffusion policies (CDPs) to induce contractive behavior in the diffusion sampling dynamics. Contraction pulls nearby flows closer to enhance robustness against solver and score-matching errors while reducing unwanted action variance. We develop an in-depth theoretical analysis along with a practical implementation recipe to incorporate CDPs into existing diffusion policy architectures with minimal modification and computational cost. We evaluate CDPs for offline learning by conducting extensive experiments in simulation and real-world settings. Across benchmarks, CDPs often outperform baseline policies, with pronounced benefits under data scarcity.
>
---
#### [replaced 022] Efficient View Planning Guided by Previous-Session Reconstruction for Repeated Plant Monitoring
- **分类: cs.RO**

- **简介: 该论文属于植物监测任务，解决重复监测中3D重建效率低的问题。通过利用前次重建结果，实现高效视点规划，提升当前会话的感知效果。**

- **链接: [https://arxiv.org/pdf/2510.07028](https://arxiv.org/pdf/2510.07028)**

> **作者:** Sicong Pan; Luca Lobefaro; Moein Taherkhani; Xuying Huang; Rohit Menon; Cyrill Stachniss; Maren Bennewitz
>
> **备注:** Submitted for review
>
> **摘要:** Repeated plant monitoring is essential for tracking crop growth, and 3D reconstruction enables consistent comparison across monitoring sessions. However, rebuilding a 3D model from scratch in every session is costly and overlooks informative geometry already observed previously. We propose efficient view planning guided by a previous-session reconstruction, which reuses a 3D model from the previous session to improve active perception in the current session. Based on this previous-session reconstruction, our method replaces iterative next-best-view planning with one-shot view planning that selects an informative set of views and computes the globally shortest execution path connecting them. Experiments on real multi-session datasets, including public single-plant scans and a newly collected greenhouse crop-row dataset, show that our method achieves comparable or higher surface coverage with fewer executed views and shorter robot paths than iterative and one-shot baselines.
>
---
#### [replaced 023] DriveCode: Domain Specific Numerical Encoding for LLM-Based Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决LLM中数值编码精度与效率问题，提出DriveCode方法将数字映射为嵌入向量，提升轨迹预测和控制性能。**

- **链接: [https://arxiv.org/pdf/2603.00919](https://arxiv.org/pdf/2603.00919)**

> **作者:** Zhiye Wang; Yanbo Jiang; Rui Zhou; Bo Zhang; Fang Zhang; Zhenhua Xu; Yaqin Zhang; Jianqiang Wang
>
> **备注:** The project page is available at this https URL
>
> **摘要:** Large language models (LLMs) have shown great promise for autonomous driving. However, discretizing numbers into tokens limits precise numerical reasoning, fails to reflect the positional significance of digits in the training objective, and makes it difficult to achieve both decoding efficiency and numerical precision. These limitations affect both the processing of sensor measurements and the generation of precise control commands, creating a fundamental barrier for deploying LLM-based autonomous driving systems. In this paper, we introduce DriveCode, a novel numerical encoding method that represents numbers as dedicated embeddings rather than discrete text tokens. DriveCode employs a number projector to map numbers into the language model's hidden space, enabling seamless integration with visual and textual features in a unified multimodal sequence. Evaluated on OmniDrive, DriveGPT4, and DriveGPT4-V2 datasets, DriveCode demonstrates superior performance in trajectory prediction and control signal generation, confirming its effectiveness for LLM-based autonomous driving systems.
>
---
#### [replaced 024] Risk-Bounded Multi-Agent Visual Navigation via Iterative Risk Allocation
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多智能体路径规划任务，解决高风险环境下安全导航问题。通过动态分配风险预算，提升任务成功率和效率。**

- **链接: [https://arxiv.org/pdf/2509.08157](https://arxiv.org/pdf/2509.08157)**

> **作者:** Viraj Parimi; Brian C. Williams
>
> **备注:** Published at ICAPS '26
>
> **摘要:** Safe navigation is essential for autonomous systems operating in hazardous environments, especially when multiple agents must coordinate using only high-dimensional visual observations. While recent approaches successfully combine Goal-Conditioned RL (GCRL) for graph construction with Conflict-Based Search (CBS) for planning, they typically rely on deleting edges with high risk before running CBS to enforce safety. This binary strategy is overly conservative, precluding feasible missions that require traversing high-risk regions, even when the aggregate risk is acceptable. To address this, we introduce a framework for Risk-Bounded Multi-Agent Path Finding ($\Delta$-MAPF), where agents share a user-specified global risk budget ($\Delta$). Rather than permanently discarding edges, our framework dynamically distributes per-agent risk budgets ($\delta_i$) during search via an Iterative Risk Allocation (IRA) layer that integrates with a standard CBS planner. We investigate two distribution strategies: a greedy surplus-deficit scheme for rapid feasibility repair, and a market-inspired mechanism that treats risk as a priced resource to guide improved allocation. The market-based mechanism yields a tunable trade-off wherein agents exploit available risk to secure shorter, more efficient paths, but revert to longer, safer detours under tighter budgets. Experiments in complex visual environments show that our dynamic allocation framework achieves higher success rates than baselines and effectively leverages the available safety budget to reduce travel time. Project website can be found at this https URL
>
---
#### [replaced 025] Implicit Maximum Likelihood Estimation for Real-time Generative Model Predictive Control
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决扩散模型推理速度慢的问题。通过引入IMLE方法，提升实时生成模型预测控制的效率，实现快速适应动态环境的路径规划。**

- **链接: [https://arxiv.org/pdf/2603.13733](https://arxiv.org/pdf/2603.13733)**

> **作者:** Grayson Lee; Minh Bui; Shuzi Zhou; Yankai Li; Mo Chen; Ke Li
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Diffusion-based models have recently shown strong performance in trajectory planning, as they are capable of capturing diverse, multimodal distributions of complex behaviors. A key limitation of these models is their slow inference speed, which results from the iterative denoising process. This makes them less suitable for real-time applications such as closed-loop model predictive control (MPC), where plans must be generated quickly and adapted continuously to a changing environment. In this paper, we investigate Implicit Maximum Likelihood Estimation (IMLE) as an alternative generative modeling approach for planning. IMLE offers strong mode coverage while enabling inference that is two orders of magnitude faster, making it particularly well suited for real-time MPC tasks. Our results demonstrate that IMLE achieves competitive performance on standard offline reinforcement learning benchmarks compared to the standard diffusion-based planner, while substantially improving planning speed in both open-loop and closed-loop settings. We further validate IMLE in a closed-loop human navigation scenario, operating in real-time, demonstrating how it enables rapid and adaptive plan generation in dynamic environments. Real-world videos and code are available at this https URL.
>
---
#### [replaced 026] ArtiSG: Functional 3D Scene Graph Construction via Human-demonstrated Articulated Objects Manipulation
- **分类: cs.RO**

- **简介: 该论文属于3D场景图构建任务，旨在解决静态场景图缺乏可操作的运动信息问题。通过人类示范数据，构建包含关节物体运动信息的3D场景图，提升机器人操作能力。**

- **链接: [https://arxiv.org/pdf/2512.24845](https://arxiv.org/pdf/2512.24845)**

> **作者:** Qiuyi Gu; Yuze Sheng; Jincheng Yu; Jiahao Tang; Xiaolong Shan; Zhaoyang Shen; Tinghao Yi; Xiaodan Liang; Xinlei Chen; Yu Wang
>
> **摘要:** 3D scene graphs have empowered robots with semantic understanding for navigation and planning. However, current functional scene graphs primarily focus on static element detection, lacking the actionable kinematic information required for physical manipulation, particularly regarding articulated objects. Existing approaches for inferring articulation mechanisms from static observations are prone to visual ambiguity, while methods that estimate parameters from state changes typically rely on constrained settings such as fixed cameras and unobstructed views. Furthermore, inconspicuous functional elements like hidden handles are frequently missed by pure visual perception. To bridge this gap, we present ArtiSG, a framework that constructs functional 3D scene graphs by encoding human demonstrations into structured robotic memory. Our approach leverages a robust data collection pipeline utilizing a portable hardware setup to accurately track 6-DoF manipulation trajectories and estimate articulation axes, even under camera ego-motion. By integrating these kinematic priors into a hierarchical, open-vocabulary graph, our system not only models how articulated objects move but also utilizes physical interaction data to discover implicit elements. Extensive real-world experiments demonstrate that ArtiSG significantly outperforms baselines in functional element recall and articulation estimation precision. Moreover, we show that the constructed graph serves as a reliable robotic memory, effectively guiding robots to perform language-directed manipulation tasks in real-world environments containing diverse articulated objects.
>
---
#### [replaced 027] CoViLLM: An Adaptive Human-Robot Collaborative Assembly Framework Using Large Language Models
- **分类: cs.RO**

- **简介: 该论文属于人机协作装配任务，旨在解决传统机器人在定制化产品装配中的灵活性不足问题。通过结合大语言模型，实现新产品的自主任务规划与协作装配。**

- **链接: [https://arxiv.org/pdf/2603.11461](https://arxiv.org/pdf/2603.11461)**

> **作者:** Jiabao Zhao; Jonghan Lim; Hongliang Li; Ilya Kovalenko
>
> **备注:** 6 pages, 7 figures. Accepted to ASME MSEC 2026
>
> **摘要:** With increasing demand for mass customization, traditional manufacturing robots that rely on rule-based operations lack the flexibility to accommodate customized or new product variants. Human-Robot Collaboration has demonstrated potential to improve system adaptability by leveraging human versatility and decision-making capabilities. However, existing Human-Robot Collaborative frameworks typically depend on predefined perception-manipulation pipelines, limiting their ability to autonomously generate task plans for new product assembly. In this work, we propose CoViLLM, an adaptive human-robot collaborative assembly framework that supports the assembly of customized and previously unseen products. CoViLLM combines depth-camera-based localization for object position estimation, human operator classification for identifying new components, and a Large Language Model for assembly task planning based on natural language instructions. The framework is validated on the NIST Assembly Task Board for known, customized, and new product cases. Experimental results show that the proposed framework enables flexible collaborative assembly by extending Human-Robot Collaboration beyond predefined product and task settings.
>
---
#### [replaced 028] Risk-Aware Obstacle Avoidance Algorithm for Real-Time Applications
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决动态环境中障碍物避让问题。提出一种融合概率建模与轨迹优化的算法，提升航行安全与适应性。**

- **链接: [https://arxiv.org/pdf/2602.09204](https://arxiv.org/pdf/2602.09204)**

> **作者:** Ozan Kaya; Emir Cem Gezer; Roger Skjetne; Ingrid Bouwer Utne
>
> **摘要:** Robust navigation in changing marine environments requires autonomous systems capable of perceiving, reasoning, and acting under uncertainty. This study introduces a hybrid risk-aware navigation architecture that integrates probabilistic modeling of obstacles along the vehicle path with smooth trajectory optimization for autonomous surface vessels. The system constructs probabilistic risk maps that capture both obstacle proximity and the behavior of dynamic objects. A risk-biased Rapidly Exploring Random Tree (RRT) planner leverages these maps to generate collision-free paths, which are subsequently refined using B-spline algorithms to ensure trajectory continuity. Three distinct RRT* rewiring modes are implemented based on the cost function: minimizing the path length, minimizing risk, and optimizing a combination of the path length and total risk. The framework is evaluated in experimental scenarios containing both static and dynamic obstacles. The results demonstrate the system's ability to navigate safely, maintain smooth trajectories, and dynamically adapt to changing environmental risks. Compared with conventional LIDAR or vision-only navigation approaches, the proposed method shows improvements in operational safety and autonomy, establishing it as a promising solution for risk-aware autonomous vehicle missions in uncertain and dynamic environments.
>
---
#### [replaced 029] Multi-Source Human-in-the-Loop Digital Twin Testbed for Connected and Autonomous Vehicles in Mixed Traffic Flow
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶测试任务，旨在解决混合交通中CAV与HDV交互的测试难题。提出MSH-MCCT测试平台，融合物理、虚拟与混合环境，实现CAV与真实驾驶员的实时互动测试。**

- **链接: [https://arxiv.org/pdf/2603.17751](https://arxiv.org/pdf/2603.17751)**

> **作者:** Jianghong Dong; Jiawei Wang; Chunying Yang; Mengchi Cai; Chaoyi Chen; Qing Xu; Jianqiang Wang; Keqiang Li
>
> **摘要:** In the emerging mixed traffic environments, Connected and Autonomous Vehicles (CAVs) have to interact with surrounding human-driven vehicles (HDVs). This paper introduces MSH-MCCT (Multi-Source Human-in-the-Loop Mixed Cloud Control Testbed), a novel CAV testbed that captures complex interactions between various CAVs and HDVs. Utilizing the Mixed Digital Twin concept, which combines Mixed Reality with Digital Twin, MSH-MCCT integrates physical, virtual, and mixed platforms, along with multi-source control inputs. Bridged by the mixed platform, MSH-MCCT allows human drivers and CAV algorithms to operate both physical and virtual vehicles within multiple fields of view. Particularly, this testbed facilitates the coexistence and real-time interaction of physical and virtual CAVs \& HDVs, significantly enhancing the experimental flexibility and scalability. Experiments on vehicle platooning in mixed traffic showcase the potential of MSH-MCCT to conduct CAV testing with multi-source real human drivers in the loop through driving simulators of diverse fidelity. The videos for the experiments are available at our project website: this https URL.
>
---
#### [replaced 030] Latent Policy Steering with Embodiment-Agnostic Pretrained World Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人视觉运动控制任务，旨在解决数据不足和实体差异问题。通过预训练世界模型并微调，提升行为克隆策略性能。**

- **链接: [https://arxiv.org/pdf/2507.13340](https://arxiv.org/pdf/2507.13340)**

> **作者:** Yiqi Wang; Mrinal Verghese; Jeff Schneider
>
> **摘要:** The performance of learned robot visuomotor policies is heavily dependent on the size and quality of the training dataset. Although large-scale robot and human datasets are increasingly available, embodiment gaps and mismatched action spaces make them difficult to leverage. Our main insight is that skills performed across different embodiments produce visual similarities in motions that can be captured using off-the-shelf action representations such as optical flow. Moreover, World Models (WMs) can leverage sub-optimal data since they focus on modeling dynamics. In this work, we aim to improve visuomotor policies in low-data regimes by first pretraining a WM using optical flow as an embodiment-agnostic action representation to leverage accessible or easily collected data from multiple embodiments (robots, humans). Given a small set of demonstrations on a target embodiment, we finetune the WM on this data to better align the WM predictions, train a base policy, and learn a robust value function. Using our finetuned WM and value function, our approach evaluates action candidates from the base policy and selects the best one to improve performance. Our approach, which we term Latent Policy Steering (LPS), improves behavior-cloned policies by 10.6% on average across four Robomimic tasks, even though most of the pretraining data comes from the real world. In the real-world experiments, LPS achieves larger gains: 70% relative improvement with 30-50 target-embodiment demonstrations, and 44% relative improvement with 60-100 demonstrations, compared to a behavior-cloned baseline. Qualitative results can be found on the website: this https URL.
>
---
#### [replaced 031] Differentiable Simulation of Hard Contacts with Soft Gradients for Learning and Control
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于机器人学习与控制任务，解决硬接触模拟中梯度不准确问题。提出DiffMJX和CFD方法，提升模拟器梯度精度并保持物理真实。**

- **链接: [https://arxiv.org/pdf/2506.14186](https://arxiv.org/pdf/2506.14186)**

> **作者:** Anselm Paulus; A. René Geist; Pierre Schumacher; Vít Musil; Simon Rappenecker; Georg Martius
>
> **摘要:** Contact forces introduce discontinuities into robot dynamics that severely limit the use of simulators for gradient-based optimization. Penalty-based simulators such as MuJoCo, soften contact resolution to enable gradient computation. However, realistically simulating hard contacts requires stiff solver settings, which leads to incorrect simulator gradients when using automatic differentiation. Contrarily, using non-stiff settings strongly increases the sim-to-real gap. We analyze penalty-based simulators to pinpoint why gradients degrade under hard contacts. Building on these insights, we propose DiffMJX, which couples adaptive time integration with penalty-based simulation to substantially improve gradient accuracy. A second challenge is that contact gradients vanish when bodies separate. To address this, we introduce contacts from distance (CFD) which combines penalty-based simulation with straight-through estimation. By applying CFD exclusively in the backward pass, we obtain informative pre-contact gradients while retaining physical realism.
>
---
#### [replaced 032] Towards a Practical Understanding of Lagrangian Methods in Safe Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于安全强化学习任务，研究拉格朗日方法中乘子λ的选择对性能与安全平衡的影响，通过实验分析约束几何和更新机制，提供优化的代价限制建议。**

- **链接: [https://arxiv.org/pdf/2510.17564](https://arxiv.org/pdf/2510.17564)**

> **作者:** Lindsay Spoor; Álvaro Serra-Gómez; Aske Plaat; Thomas Moerland
>
> **摘要:** Safe reinforcement learning addresses constrained optimization problems where maximizing performance must be balanced against safety constraints, and Lagrangian methods are a widely used approach for this purpose. However, the effectiveness of Lagrangian methods depends crucially on the choice of the Lagrange multiplier $\lambda$, which governs the multi-objective trade-off between return and cost. A common practice is to update the multiplier automatically during training. Although this approach is standard in practice, there remains limited empirical evidence on the optimally achievable trade-off between return and cost as a function of $\lambda$, and there is currently no systematic benchmark comparing automated update mechanisms to this empirical optimum. Therefore, we study (i) the constraint geometry for eight widely used safety tasks and (ii) the previously overlooked constraint-regime sensitivity of different Lagrange multiplier update mechanisms in safe reinforcement learning. Through the lens of multi-objective analysis, we present empirical Pareto frontiers that offer a complete visualization of the trade-off between return and cost in the underlying optimization problem. Our results reveal the highly sensitive nature of $\lambda$ and further show that the restrictiveness of the constraint cost can vary across different cost limits within the same task. This highlights the importance of careful cost limit selection across different regions of cost restrictiveness when evaluating safe reinforcement learning methods. We provide a recommended set of cost limits for each evaluated task and offer an open-source code base: this https URL.
>
---
#### [replaced 033] PhysMem: Self-Evolving Physical Memory for Robot Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PhysMem，解决机器人操作中物理属性理解不足的问题。通过测试假设而非直接依赖经验，提升操作成功率。属于机器人操纵任务。**

- **链接: [https://arxiv.org/pdf/2602.20323](https://arxiv.org/pdf/2602.20323)**

> **作者:** Haoyang Li; Yang You; Hao Su; Leonidas Guibas
>
> **摘要:** Reliable object manipulation requires understanding physical properties that vary across objects and environments. Vision-language model (VLM) planners can reason about friction and stability in general terms; however, they often cannot predict how a specific ball will roll on a particular surface or which stone will provide a stable foundation without direct experience. We present PhysMem, a memory framework that enables VLM robot planners to learn physical principles from interaction at test time, without updating model parameters. The system records experiences, generates candidate hypotheses, and verifies them through targeted interaction before promoting validated knowledge to guide future decisions. A central design choice is verification before application: the system tests hypotheses against new observations rather than applying retrieved experience directly, reducing rigid reliance on prior experience when physical conditions change. We evaluate PhysMem on three real-world manipulation tasks and simulation benchmarks across four VLM backbones. On a controlled brick insertion task, principled abstraction achieves 76% success compared to 23% for direct experience retrieval, and real-world experiments show consistent improvement over 30-minute deployment sessions.
>
---
#### [replaced 034] Cutting the Cord: System Architecture for Low-Cost, GPU-Accelerated Bimanual Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，旨在解决低成本、自主机器人平台的构建问题。设计了一种低功耗双臂机器人系统，实现自主导航与操作。**

- **链接: [https://arxiv.org/pdf/2603.09051](https://arxiv.org/pdf/2603.09051)**

> **作者:** Artemis Shaw; Chen Liu; Justin Costa; Rane Gray; Alina Skowronek; Kevin Diaz; Nam Bui; Nikolaus Correll
>
> **摘要:** We present a bimanual mobile manipulator built on the open-source XLeRobot with integrated onboard compute for less than \$1300. Key contributions include: (1) optimized mechanical design maximizing stiffness-to-weight ratio, (2) a Tri-Bus power topology isolating compute from motor-induced voltage transients, and (3) embedded autonomy using NVIDIA Jetson Orin Nano for untethered operation. The platform enables teleoperation, autonomous SLAM navigation, and vision-based manipulation without external dependencies, providing a low-cost alternative for research and education in robotics and robot learning.
>
---
#### [replaced 035] Video2Act: A Dual-System Video Diffusion Policy with Robotic Spatio-Motional Modeling
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决视频扩散模型中运动表示不连贯的问题。提出Video2Act框架，结合空间与运动信息提升机器人动作学习效果。**

- **链接: [https://arxiv.org/pdf/2512.03044](https://arxiv.org/pdf/2512.03044)**

> **作者:** Yueru Jia; Jiaming Liu; Shengbang Liu; Rui Zhou; Wanhe Yu; Yuyang Yan; Xiaowei Chi; Yandong Guo; Boxin Shi; Shanghang Zhang
>
> **摘要:** Robust perception and dynamics modeling are fundamental to real-world robotic policy learning. Recent methods employ video diffusion models (VDMs) to enhance robotic policies, improving their understanding and modeling of the physical world. However, existing approaches overlook the coherent and physically consistent motion representations inherently encoded across frames in VDMs. To this end, we propose Video2Act, a framework that efficiently guides robotic action learning by explicitly integrating spatial and motion-aware representations. Building on the inherent representations of VDMs, we extract foreground boundaries and inter-frame motion variations while filtering out background noise and task-irrelevant biases. These refined representations are then used as additional conditioning inputs to a diffusion transformer (DiT) action head, enabling it to reason about what to manipulate and how to move. To mitigate inference inefficiency, we propose an asynchronous dual-system design, where the VDM functions as the slow System 2 and the DiT head as the fast System 1, working collaboratively to generate adaptive actions. By providing motion-aware conditions to System 1, Video2Act maintains stable manipulation even with low-frequency updates from the VDM. For evaluation, Video2Act surpasses previous state-of-the-art VLA methods by 7.7% in simulation and 21.7% in real-world tasks in terms of average success rate, further exhibiting strong generalization capabilities.
>
---
#### [replaced 036] Stratified Topological Autonomy for Long-Range Coordination (STALC)
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出STALC，用于多机器人长距离协调的分层规划方法，解决复杂环境下的多机器人路径规划与协作问题。通过拓扑图与混合整数规划结合，实现高效、自主的协同作业。**

- **链接: [https://arxiv.org/pdf/2503.10475](https://arxiv.org/pdf/2503.10475)**

> **作者:** Cora A. Duggan; Adam Goertz; Adam Polevoy; Mark Gonzales; Kevin C. Wolfe; Bradley Woosley; John G. Rogers III; Joseph Moore
>
> **备注:** ©2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** In this paper, we present Stratified Topological Autonomy for Long-Range Coordination (STALC), a hierarchical planning approach for multi-robot coordination in real-world environments with significant inter-robot spatial and temporal dependencies. At its core, STALC consists of a multi-robot graph-based planner which combines a topological graph with a novel, computationally efficient mixed-integer programming formulation to generate highly-coupled multi-robot plans in seconds. To enable autonomous planning across different spatial and temporal scales, we construct our graphs so that they capture connectivity between free-space regions and other problem-specific features, such as traversability or risk. We then use receding-horizon planners to achieve local collision avoidance and formation control. To evaluate our approach, we consider a multi-robot reconnaissance scenario where robots must autonomously coordinate to navigate through an environment while minimizing the risk of detection by observers. Through simulation-based experiments, we show that our approach is able to scale to address complex multi-robot planning scenarios. Through hardware experiments, we demonstrate our ability to generate graphs from real-world data and successfully plan across the entire hierarchy to achieve shared objectives.
>
---
#### [replaced 037] Inverse-dynamics observer design for a linear single-track vehicle model with distributed tire dynamics
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于车辆状态估计任务，旨在准确估算侧滑角和轮胎力。通过结合线性单轨模型与分布轮胎动力学，利用动态逆方法从传感器数据中重建车辆状态。**

- **链接: [https://arxiv.org/pdf/2603.07499](https://arxiv.org/pdf/2603.07499)**

> **作者:** Luigi Romano; Ole Morten Aamo; Jan Åslund; Erik Frisk
>
> **备注:** 6 pages, 5 figures. Accepted at ECC 2026
>
> **摘要:** Accurate estimation of the vehicle's sideslip angle and tire forces is essential for enhancing safety and handling performances in unknown driving scenarios. To this end, the present paper proposes an innovative observer that combines a linear single-track model with a distributed representation of the tires and information collected from standard sensors. In particular, by adopting a comprehensive representation of the tires in terms of hyperbolic partial differential equations (PDEs), the proposed estimation strategy exploits dynamical inversion to reconstruct the lumped and distributed vehicle states solely from yaw rate and lateral acceleration measurements. Simulation results demonstrate the effectiveness of the observer in estimating the sideslip angle and tire forces even in the presence of noise and model uncertainties.
>
---
#### [replaced 038] Mixed-Integer vs. Continuous Model Predictive Control for Binary Thrusters: A Comparative Study
- **分类: eess.SY; cs.RO**

- **简介: 论文比较了混合整数MPC与连续MPC结合Delta-Sigma调制在二进制推进器控制中的性能。任务是提升航天器控制的稳定性与燃料效率，解决离散执行器控制难题。工作包括仿真测试与新方法提出。**

- **链接: [https://arxiv.org/pdf/2603.19796](https://arxiv.org/pdf/2603.19796)**

> **作者:** Franek Stark; Jakob Middelberg; Shubham Vyas
>
> **备注:** Accepted to CEAS EuroGNC 2026
>
> **摘要:** Binary on/off thrusters are commonly used for spacecraft attitude and position control during proximity operations. However, their discrete nature poses challenges for conventional continuous control methods. The control of these discrete actuators is either explicitly formulated as a mixed-integer optimization problem or handled in a two-layer approach, where a continuous controller's output is converted to binary commands using analog-to digital modulation techniques such as Delta-Sigma-modulation. This paper provides the first systematic comparison between these two paradigms for binary thruster control, contrasting continuous Model Predictive Control (MPC) with Delta-Sigma modulation against direct Mixed-Integer MPC (MIMPC) approaches. Furthermore, we propose a new variant of MPC for binary actuated systems, which is informed using the state of the Delta-Sigma Modulator. The two variations for the continuous MPC along with the MIMPC are evaluated through extensive simulations using ESA's REACSA platform. Results demonstrate that while all approaches perform similarly in high-thrust regimes, MIMPC achieves superior fuel efficiency in low-thrust conditions. Continuous MPC with modulation shows instabilities at higher thrust levels, while binary informed MPC, which incorporates modulator dynamics, improves robustness and reduces the efficiency gap to the MIMPC. It can be seen from the simulated and real-system experiments that MIMPC offers complete stability and fuel efficiency benefits, particularly for resource-constrained missions, while continuous control methods remain attractive for computationally limited applications.
>
---
#### [replaced 039] Spectral Alignment in Forward-Backward Representations via Temporal Abstraction
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决连续环境中FB表示与低秩瓶颈的谱不匹配问题。通过时间抽象降低有效秩，提升长期规划能力。**

- **链接: [https://arxiv.org/pdf/2603.20103](https://arxiv.org/pdf/2603.20103)**

> **作者:** Seyed Mahdi B. Azad; Jasper Hoffmann; Iman Nematollahi; Hao Zhu; Abhinav Valada; Joschka Boedecker
>
> **摘要:** Forward-backward (FB) representations provide a powerful framework for learning the successor representation (SR) in continuous spaces by enforcing a low-rank factorization. However, a fundamental spectral mismatch often exists between the high-rank transition dynamics of continuous environments and the low-rank bottleneck of the FB architecture, making accurate low-rank representation learning difficult. In this work, we analyze temporal abstraction as a mechanism to mitigate this mismatch. By characterizing the spectral properties of the transition operator, we show that temporal abstraction acts as a low-pass filter that suppresses high-frequency spectral components. This suppression reduces the effective rank of the induced SR while preserving a formal bound on the resulting value function error. Empirically, we show that this alignment is a key factor for stable FB learning, particularly at high discount factors where bootstrapping becomes error-prone. Our results identify temporal abstraction as a principled mechanism for shaping the spectral structure of the underlying MDP and enabling effective long-horizon representations in continuous control.
>
---
#### [replaced 040] VertiAdaptor: Online Kinodynamics Adaptation for Vertically Challenging Terrain
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决复杂地形下车辆动力学模型适应问题。提出VertiAdaptor框架，通过融合高程与语义信息实现快速在线适应。**

- **链接: [https://arxiv.org/pdf/2603.06887](https://arxiv.org/pdf/2603.06887)**

> **作者:** Tong Xu; Chenhui Pan; Aniket Datar; Xuesu Xiao
>
> **摘要:** Autonomous driving in off-road environments presents significant challenges due to the dynamic and unpredictable nature of unstructured terrain. Traditional kinodynamic models often struggle to generalize across diverse geometric and semantic terrain types, underscoring the need for real-time adaptation to ensure safe and reliable navigation. We propose VertiAdaptor (VA), a novel online adaptation framework that efficiently integrates elevation with semantic embeddings to enable terrain-aware kinodynamic modeling and planning via function encoders. VA learns a kinodynamic space spanned by a set of neural ordinary differential equation basis functions, capturing complex vehicle-terrain interactions across varied environments. After offline training, the proposed approach can rapidly adapt to new, unseen environments by identifying kinodynamics in the learned space through a computationally efficient least-squares calculation. We evaluate VA within the Verti-Bench simulator, built on the Chrono multi-physics engine, and validate its performance both in simulation and on a physical Verti-4-Wheeler platform. Our results demonstrate that VA improves prediction accuracy by up to 23.9% and achieves a 5X faster adaptation time, advancing the robustness and reliability of autonomous robots in complex and evolving off-road environments.
>
---
#### [replaced 041] DYMO-Hair: Generalizable Volumetric Dynamics Modeling for Robot Hair Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人发丝操作任务，旨在解决机器人处理复杂发丝动态的问题。提出DYMO-Hair模型，通过动态学习和3D潜空间实现通用发丝建模与造型。**

- **链接: [https://arxiv.org/pdf/2510.06199](https://arxiv.org/pdf/2510.06199)**

> **作者:** Chengyang Zhao; Uksang Yoo; Arkadeep Narayan Chaudhury; Giljoo Nam; Jonathan Francis; Jeffrey Ichnowski; Jean Oh
>
> **备注:** To appear in ICRA 2026. Project page: this https URL
>
> **摘要:** Hair care is an essential daily activity, yet it remains inaccessible to individuals with limited mobility and challenging for autonomous robot systems due to the fine-grained physical structure and complex dynamics of hair. In this work, we present DYMO-Hair, a model-based robot hair care system. We introduce a novel dynamics learning paradigm that is suited for volumetric quantities such as hair, relying on an action-conditioned latent state editing mechanism, coupled with a compact 3D latent space of diverse hairstyles to improve generalizability. This latent space is pre-trained at scale using a novel hair physics simulator, enabling generalization across previously unseen hairstyles. Using the dynamics model with a Model Predictive Path Integral (MPPI) planner, DYMO-Hair is able to perform visual goal-conditioned hair styling. Experiments in simulation demonstrate that DYMO-Hair's dynamics model outperforms baselines on capturing local deformation for diverse, unseen hairstyles. DYMO-Hair further outperforms baselines in closed-loop hair styling tasks on unseen hairstyles, with an average of 22% lower final geometric error and 42% higher success rate than the state-of-the-art system. Real-world experiments exhibit zero-shot transferability of our system to wigs, achieving consistent success on challenging unseen hairstyles where the state-of-the-art system fails. Together, these results introduce a foundation for model-based robot hair care, advancing toward more generalizable, flexible, and accessible robot hair styling in unconstrained physical environments. More details are available on our project page: this https URL.
>
---
#### [replaced 042] Articulated-Body Dynamics Network: Dynamics-Grounded Prior for Robot Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在提升策略学习效率。通过引入基于动力学的先验知识，设计ABD-Net网络，解决动力学信息未被有效利用的问题。**

- **链接: [https://arxiv.org/pdf/2603.19078](https://arxiv.org/pdf/2603.19078)**

> **作者:** Sangwoo Shin; Kunzhao Ren; Xiaobin Xiong; Josiah P. Hanna
>
> **备注:** Arxiv_r2
>
> **摘要:** Recent work in reinforcement learning has shown that incorporating structural priors for articulated robots, such as link connectivity, into policy networks improves learning efficiency. However, dynamics properties, despite their fundamental role in determining how forces and motion propagate through the body, remain largely underexplored as an inductive bias for policy learning. To address this gap, we present the Articulated-Body Dynamics Network (ABD-Net), a novel graph neural network architecture grounded in the computational structure of forward dynamics. Specifically, we adapt the inertia propagation mechanism from the Articulated Body Algorithm, systematically aggregating inertial quantities from child to parent links in a tree-structured manner, while replacing physical quantities with learnable parameters. Embedding ABD-NET into the policy actor enables dynamics-informed representations that capture how actions propagate through the body, leading to efficient and robust policy learning. Through experiments with simulated humanoid, quadruped, and hopper robots, our approach demonstrates increased sample efficiency and generalization to dynamics shifts compared to transformer-based and GNN baselines. We further validate the learned policy on real Unitree G1 and Go2 robots, state-of-the-art humanoid and quadruped platforms, generating dynamic, versatile and robust locomotion behaviors through sim-to-real transfer with real-time inference.
>
---
#### [replaced 043] Large Reward Models: Generalizable Online Robot Reward Generation with Vision-Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人强化学习任务，解决奖励函数设计困难的问题。通过将视觉语言模型转化为在线奖励生成器，提升机器人操作的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2603.16065](https://arxiv.org/pdf/2603.16065)**

> **作者:** Yanru Wu; Weiduo Yuan; Ang Qi; Vitor Guizilini; Jiageng Mao; Yue Wang
>
> **摘要:** Reinforcement Learning (RL) has shown great potential in refining robotic manipulation policies, yet its efficacy remains strongly bottlenecked by the difficulty of designing generalizable reward functions. In this paper, we propose a framework for online policy refinement by adapting foundation VLMs into online reward generators. We develop a robust, scalable reward model based on a state-of-the-art VLM, trained on a large-scale, multi-source dataset encompassing real-world robot trajectories, human-object interactions, and diverse simulated environments. Unlike prior approaches that evaluate entire trajectories post-hoc, our method leverages the VLM to formulate a multifaceted reward signal comprising process, completion, and temporal contrastive rewards based on current visual observations. Initializing with a base policy trained via Imitation Learning (IL), we employ these VLM rewards to guide the model to correct sub-optimal behaviors in a closed-loop manner. We evaluate our framework on challenging long-horizon manipulation benchmarks requiring sequential execution and precise control. Crucially, our reward model operates in a purely zero-shot manner within these test environments. Experimental results demonstrate that our method significantly improves the success rate of the initial IL policy within just 30 RL iterations, demonstrating remarkable sample efficiency. This empirical evidence highlights that VLM-generated signals can provide reliable feedback to resolve execution errors, effectively eliminating the need for manual reward engineering and facilitating efficient online refinement for robot learning.
>
---
#### [replaced 044] RoboMorph: Evolving Robot Morphology using Large Language Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于机器人设计任务，旨在解决传统设计方法耗时且计算成本高的问题。通过结合大语言模型和进化算法，自动生成并优化模块化机器人结构。**

- **链接: [https://arxiv.org/pdf/2407.08626](https://arxiv.org/pdf/2407.08626)**

> **作者:** Kevin Qiu; Władysław Pałucki; Krzysztof Ciebiera; Paweł Fijałkowski; Marek Cygan; Łukasz Kuciński
>
> **摘要:** We introduce RoboMorph, an automated approach for generating and optimizing modular robot designs using large language models (LLMs) and evolutionary algorithms. Each robot design is represented by a structured grammar, and we use LLMs to efficiently explore this design space. Traditionally, such exploration is time-consuming and computationally intensive. Using a best-shot prompting strategy combined with reinforcement learning (RL)-based control evaluation, RoboMorph iteratively refines robot designs within an evolutionary feedback loop. Across four terrain types, RoboMorph discovers diverse, terrain-specialized morphologies, including wheeled quadrupeds and hexapods, that match or outperform designs produced by Robogrammar's graph-search method. These results demonstrate that LLMs, when coupled with evolutionary selection, can serve as effective generative operators for automated robot design. Our project page and code are available at this https URL.
>
---
#### [replaced 045] sim2art: Accurate Articulated Object Modeling from a Single Video using Synthetic Training Data Only
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单视角视频中刚体物体建模任务，旨在解决真实场景下复杂关节物体的3D分割与参数恢复问题。工作包括提出sim2art框架，利用合成数据训练，无需真实标注即可准确建模。**

- **链接: [https://arxiv.org/pdf/2512.07698](https://arxiv.org/pdf/2512.07698)**

> **作者:** Arslan Artykov; Tom Ravaud; Corentin Sautier; Vincent Lepetit
>
> **摘要:** Understanding articulated objects from monocular video is a crucial yet challenging task in robotics and digital twin creation. Existing methods often rely on complex multi-view setups, high-fidelity object scans, or fragile long-term point tracks that frequently fail in casual real-world captures. In this paper, we present sim2art, a data-driven framework that recovers the 3D part segmentation and joint parameters of articulated objects from a single monocular video captured by a freely moving camera. Our core insight is a robust representation based on per-frame surface point sampling, which we augment with short-term scene flow and DINOv3 semantic features. Unlike previous works that depend on error-prone long-term correspondences, our representation is easy to obtain and exhibits a negligible difference between simulation and reality without requiring domain adaptation. Also, by construction, our method relies on single-viewpoint visibility, ensuring that the geometric representation remains consistent across synthetic and real data despite noise and occlusions. Leveraging a suitable Transformer-based architecture, sim2art is trained exclusively on synthetic data yet generalizes strongly to real-world sequences. To address the lack of standardized benchmarks in the field, we introduce two datasets featuring a significantly higher diversity of object categories and instances than prior work. Our evaluations show that sim2art effectively handles large camera motions and complex articulations, outperforming state-of-the-art optimization-based and tracking-dependent methods. sim2art offers a scalable solution that can be easily extended to new object categories without the need for cumbersome real-world annotations. Project webpage: this https URL
>
---
#### [replaced 046] Expand Your SCOPE: Semantic Cognition over Potential-Based Exploration for Embodied Visual Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于具身视觉导航任务，旨在解决代理在未知环境中基于有限知识进行长期规划的问题。提出SCOPE框架，利用前沿信息提升探索效率和决策质量。**

- **链接: [https://arxiv.org/pdf/2511.08935](https://arxiv.org/pdf/2511.08935)**

> **作者:** Ningnan Wang; Weihuang Chen; Liming Chen; Haoxuan Ji; Zhongyu Guo; Xuchong Zhang; Hongbin Sun
>
> **备注:** Accepted to AAAI 2026
>
> **摘要:** Embodied visual navigation remains a challenging task, as agents must explore unknown environments with limited knowledge. Existing zero-shot studies have shown that incorporating memory mechanisms to support goal-directed behavior can improve long-horizon planning performance. However, they overlook visual frontier boundaries, which fundamentally dictate future trajectories and observations, and fall short of inferring the relationship between partial visual observations and navigation goals. In this paper, we propose Semantic Cognition Over Potential-based Exploration (SCOPE), a zero-shot framework that explicitly leverages frontier information to drive potential-based exploration, enabling more informed and goal-relevant decisions. SCOPE estimates exploration potential with a Vision-Language Model and organizes it into a spatio-temporal potential graph, capturing boundary dynamics to support long-horizon planning. In addition, SCOPE incorporates a self-reconsideration mechanism that revisits and refines prior decisions, enhancing reliability and reducing overconfident errors. Experimental results on two diverse embodied navigation tasks show that SCOPE outperforms state-of-the-art baselines by 4.6\% in accuracy. Further analysis demonstrates that its core components lead to improved calibration, stronger generalization, and higher decision quality.
>
---
#### [replaced 047] Unified Generation-Refinement Planning: Bridging Guided Flow Matching and Sampling-Based MPC for Social Navigation
- **分类: cs.RO**

- **简介: 该论文属于社会导航任务，解决动态环境中路径规划的不确定性与实时性问题。融合生成与优化方法，提升安全性和效率。**

- **链接: [https://arxiv.org/pdf/2508.01192](https://arxiv.org/pdf/2508.01192)**

> **作者:** Kazuki Mizuta; Karen Leung
>
> **摘要:** Robust robot planning in dynamic, human-centric environments remains challenging due to multimodal uncertainty, the need for real-time adaptation, and safety requirements. Optimization-based planners enable explicit constraint handling but can be sensitive to initialization and struggle in dynamic settings. Learning-based planners capture multimodal solution spaces more naturally, but often lack reliable constraint satisfaction. In this paper, we introduce a unified generation-refinement framework that combines reward-guided conditional flow matching (CFM) with model predictive path integral (MPPI) control. Our key idea is a bidirectional information exchange between generation and optimization: reward-guided CFM produces diverse, informed trajectory priors for MPPI refinement, while the optimized MPPI trajectory warm-starts the next CFM generation step. Using autonomous social navigation as a motivating application, we demonstrate that the proposed approach improves the trade-off between safety, task performance, and computation time, while adapting to dynamic environments in real-time. The source code is publicly available at this https URL.
>
---
#### [replaced 048] Energy-Aware Reinforcement Learning for Robotic Manipulation of Articulated Components in Infrastructure Operation and Maintenance
- **分类: eess.SY; cs.AI; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决智能基础设施运维中关节部件操作的能耗问题。提出一种无特定结构的节能强化学习框架，提升操作效率与可持续性。**

- **链接: [https://arxiv.org/pdf/2602.12288](https://arxiv.org/pdf/2602.12288)**

> **作者:** Xiaowen Tao; Yinuo Wang; Haitao Ding; Yuanyang Qi; Ziyu Song
>
> **备注:** 18 pages, 5 figures, 7 tables. This version supersedes all previous preprint versions
>
> **摘要:** With the growth of intelligent civil infrastructure and smart cities, operation and maintenance (O&M) increasingly requires safe, efficient, and energy-conscious robotic manipulation of articulated components, including access doors, service drawers, and pipeline valves. However, existing robotic approaches either focus primarily on grasping or target object-specific articulated manipulation, and they rarely incorporate explicit actuation energy into multi-objective optimisation, which limits their scalability and suitability for long-term deployment in real O&M settings. Therefore, this paper proposes an articulation-agnostic and energy-aware reinforcement learning framework for robotic manipulation in intelligent infrastructure O&M. The method combines part-guided 3D perception, weighted point sampling, and PointNet-based encoding to obtain a compact geometric representation that generalises across heterogeneous articulated objects. Manipulation is formulated as a Constrained Markov Decision Process (CMDP), in which actuation energy is explicitly modelled and regulated via a Lagrangian-based constrained Soft Actor-Critic scheme. The policy is trained end-to-end under this CMDP formulation, enabling effective articulated-object operation while satisfying a long-horizon energy budget. Experiments on representative O&M tasks demonstrate 16%-30% reductions in energy consumption, 16%-32% fewer steps to success, and consistently high success rates, indicating a scalable and sustainable solution for infrastructure O&M manipulation.
>
---
#### [replaced 049] Real2Edit2Real: Generating Robotic Demonstrations via a 3D Control Interface
- **分类: cs.RO; cs.CV; cs.GR**

- **简介: 该论文提出Real2Edit2Real框架，解决机器人操作任务中数据收集成本高的问题。通过3D控制接口生成新演示，提升数据效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.19402](https://arxiv.org/pdf/2512.19402)**

> **作者:** Yujie Zhao; Hongwei Fan; Di Chen; Shengcong Chen; Liliang Chen; Xiaoqi Li; Guanghui Ren; Hao Dong
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent progress in robot learning has been driven by large-scale datasets and powerful visuomotor policy architectures, yet policy robustness remains limited by the substantial cost of collecting diverse demonstrations, particularly for spatial generalization in manipulation tasks. To reduce repetitive data collection, we present Real2Edit2Real, a framework that generates new demonstrations by bridging 3D editability with 2D visual data through a 3D control interface. Our approach first reconstructs scene geometry from multi-view RGB observations with a metric-scale 3D reconstruction model. Based on the reconstructed geometry, we perform depth-reliable 3D editing on point clouds to generate new manipulation trajectories while geometrically correcting the robot poses to recover physically consistent depth, which serves as a reliable condition for synthesizing new demonstrations. Finally, we propose a multi-conditional video generation model guided by depth as the primary control signal, together with action, edge, and ray maps, to synthesize spatially augmented multi-view manipulation videos. Experiments on four real-world manipulation tasks demonstrate that policies trained on data generated from only 1-5 source demonstrations can match or outperform those trained on 50 real-world demonstrations, improving data efficiency by up to 10-50x. Moreover, experimental results on height and texture editing demonstrate the framework's flexibility and extensibility, indicating its potential to serve as a unified data generation framework. Project website is this https URL.
>
---
#### [replaced 050] Optimal Solutions for the Moving Target Vehicle Routing Problem via Branch-and-Price with Relaxed Continuity
- **分类: cs.RO**

- **简介: 该论文研究MT-VRP任务，解决移动目标的车辆路径规划问题。提出BPRC算法，通过改进的标签算法高效求解，提升求解速度与精度。**

- **链接: [https://arxiv.org/pdf/2603.00663](https://arxiv.org/pdf/2603.00663)**

> **作者:** Anoop Bhat; Geordan Gutow; Zhongqiang Ren; Sivakumar Rathinam; Howie Choset
>
> **备注:** Accepted to ICAPS 2026
>
> **摘要:** The Moving Target Vehicle Routing Problem (MT-VRP) seeks trajectories for several agents that intercept a set of moving targets, subject to speed, time window, and capacity constraints. We introduce an exact algorithm, Branch-and-Price with Relaxed Continuity (BPRC), for the MT-VRP. The main challenge in a branch-and-price approach for the MT-VRP is the pricing subproblem, which is complicated by moving targets and time-dependent travel costs between targets. Our key contribution is a new labeling algorithm that solves this subproblem by means of a novel dominance criterion tailored for problems with moving targets. Numerical results on instances with up to 25 targets show that our algorithm finds optimal solutions more than an order of magnitude faster than a baseline based on previous work, showing particular strength in scenarios with limited agent capacities.
>
---
#### [replaced 051] Barrier-Riccati Synthesis for Nonlinear Safe Control with Expanded Region of Attraction
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于安全控制任务，解决非线性系统安全约束问题。通过结合屏障状态与SDRE方法，扩展吸引域并保证系统安全稳定。**

- **链接: [https://arxiv.org/pdf/2504.15453](https://arxiv.org/pdf/2504.15453)**

> **作者:** Hassan Almubarak; Maitham F. AL-Sunni; Justin T. Dubbin; Nader Sadegh; John M. Dolan; Evangelos A. Theodorou
>
> **备注:** This work has been accepted for publication in the proceedings of the 2026 American Control Conference (ACC), New Orleans, Louisiana, USA
>
> **摘要:** We present a Riccati-based framework for safety-critical nonlinear control that integrates the barrier states (BaS) methodology with the State-Dependent Riccati Equation (SDRE) approach. The BaS formulation embeds safety constraints into the system dynamics via auxiliary states, enabling safety to be treated as a control objective. To overcome the limited region of attraction in linear BaS controllers, we extend the framework to nonlinear systems using SDRE synthesis applied to the barrier-augmented dynamics and derive a matrix inequality condition that certifies forward invariance of a large region of attraction and guarantees asymptotic safe stabilization. The resulting controller is computed online via pointwise Riccati solutions. We validate the method on an unstable constrained system and cluttered quadrotor navigation tasks, demonstrating improved constraint handling, scalability, and robustness near safety boundaries. This framework offers a principled and computationally tractable solution for synthesizing nonlinear safe feedback in safety-critical environments.
>
---
#### [replaced 052] Goal Force: Teaching Video Models To Accomplish Physics-Conditioned Goals
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视频生成任务，旨在解决视频模型目标设定困难的问题。通过引入力向量和动态信息作为目标，训练模型理解物理交互，实现精准的物理感知规划。**

- **链接: [https://arxiv.org/pdf/2601.05848](https://arxiv.org/pdf/2601.05848)**

> **作者:** Nate Gillman; Yinghua Zhou; Zitian Tang; Evan Luo; Arjan Chakravarthy; Daksh Aggarwal; Michael Freeman; Charles Herrmann; Chen Sun
>
> **备注:** Camera ready version (CVPR 2026). Code and interactive demos at this https URL
>
> **摘要:** Recent advancements in video generation have enabled the development of ``world models'' capable of simulating potential futures for robotics and planning. However, specifying precise goals for these models remains a challenge; text instructions are often too abstract to capture physical nuances, while target images are frequently infeasible to specify for dynamic tasks. To address this, we introduce Goal Force, a novel framework that allows users to define goals via explicit force vectors and intermediate dynamics, mirroring how humans conceptualize physical tasks. We train a video generation model on a curated dataset of synthetic causal primitives-such as elastic collisions and falling dominos-teaching it to propagate forces through time and space. Despite being trained on simple physics data, our model exhibits remarkable zero-shot generalization to complex, real-world scenarios, including tool manipulation and multi-object causal chains. Our results suggest that by grounding video generation in fundamental physical interactions, models can emerge as implicit neural physics simulators, enabling precise, physics-aware planning without reliance on external engines. We release all datasets, code, model weights, and interactive video demos at our project page.
>
---
#### [replaced 053] HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在解决如何高效获取数据并精确重建未知区域的问题。通过主动学习和不确定性量化，提出一种基于神经辐射场的探索框架。**

- **链接: [https://arxiv.org/pdf/2601.07242](https://arxiv.org/pdf/2601.07242)**

> **作者:** Taekbeom Lee; Dabin Kim; Youngseok Jang; H. Jin Kim
>
> **备注:** Accepted to IEEE RA-L. The first two authors contributed equally
>
> **摘要:** We present HERE, an active 3D scene reconstruction framework based on neural radiance fields, enabling high-fidelity implicit mapping. Our approach centers around an active learning strategy for camera trajectory generation, driven by accurate identification of unseen regions, which supports efficient data acquisition and precise scene reconstruction. The key to our approach is epistemic uncertainty quantification based on evidential deep learning, which directly captures data insufficiency and exhibits a strong correlation with reconstruction errors. This allows our framework to more reliably identify unexplored or poorly reconstructed regions compared to existing methods, leading to more informed and targeted exploration. Additionally, we design a hierarchical exploration strategy that leverages learned epistemic uncertainty, where local planning extracts target viewpoints from high-uncertainty voxels based on visibility for trajectory generation, and global planning uses uncertainty to guide large-scale coverage for efficient and comprehensive reconstruction. The effectiveness of the proposed method in active 3D reconstruction is demonstrated by achieving higher reconstruction completeness compared to previous approaches on photorealistic simulated scenes across varying scales, while a hardware demonstration further validates its real-world applicability. Project page: this https URL
>
---
#### [replaced 054] Multi-Step First: A Lightweight Deep Reinforcement Learning Strategy for Robust Continuous Control with Partial Observability
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究部分可观测环境下的连续控制问题，比较了PPO、TD3和SAC在POMDP中的表现，发现PPO更具鲁棒性，并提出改进方法提升算法性能。**

- **链接: [https://arxiv.org/pdf/2209.04999](https://arxiv.org/pdf/2209.04999)**

> **作者:** Lingheng Meng; Rob Gorbet; Michael Burke; Dana Kulić
>
> **备注:** 21 pages, 12 figures. Published in Neural Networks, Vol. 199, 2026
>
> **摘要:** Deep Reinforcement Learning (DRL) has made considerable advances in simulated and physical robot control tasks, especially when problems admit a fully observed Markov Decision Process (MDP) formulation. When observations only partially capture the underlying state, the problem becomes a Partially Observable MDP (POMDP), and performance rankings between algorithms can change. We empirically compare Proximal Policy Optimization (PPO), Twin Delayed Deep Deterministic Policy Gradient (TD3), and Soft Actor-Critic (SAC) on representative POMDP variants of continuous-control benchmarks. Contrary to widely reported MDP results where TD3 and SAC typically outperform PPO, we observe an inversion: PPO attains higher robustness under partial observability. We attribute this to the stabilizing effect of multi-step bootstrapping. Furthermore, incorporating multi-step targets into TD3 (MTD3) and SAC (MSAC) improves their robustness. These findings provide practical guidance for selecting and adapting DRL algorithms in partially observable settings without requiring new theoretical machinery.
>
---
#### [replaced 055] Concept-Based Dictionary Learning for Inference-Time Safety in Vision Language Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉语言动作模型的安全防护任务，解决嵌入式系统中因指令误解引发的物理安全风险。通过概念字典学习，在推理时检测并抑制有害概念，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2602.01834](https://arxiv.org/pdf/2602.01834)**

> **作者:** Siqi Wen; Shu Yang; Shaopeng Fu; Jingfeng Zhang; Lijie Hu; Di Wang
>
> **摘要:** Vision Language Action (VLA) models close the perception action loop by translating multimodal instructions into executable behaviors, but this very capability magnifies safety risks: jailbreaks that merely yield toxic text in LLMs can trigger unsafe physical actions in embodied systems. Existing defenses alignment, filtering, or prompt hardening intervene too late or at the wrong modality, leaving fused representations exploitable. We introduce a concept based dictionary learning framework for inference time safety control. By learning sparse, interpretable dictionaries from hidden activations, our method identifies harmful concept directions and attenuates risky components when the estimated risk exceeds a threshold. Experiments on Libero-Harm, BadRobot, RoboPair, and IS-Bench show that our approach achieves state-of-the-art defense performance, cutting attack success rates by over 70\% while maintaining task success. Crucially, the framework is plug-in and model-agnostic, requiring no retraining and integrating seamlessly with diverse VLAs. To our knowledge, this is the first inference time concept based safety method for embodied systems, advancing both interpretability and safe deployment of VLA models.
>
---
#### [replaced 056] PACE: Physics Augmentation for Coordinated End-to-end Reinforcement Learning toward Versatile Humanoid Table Tennis
- **分类: cs.RO**

- **简介: 该论文研究人形机器人乒乓球任务，解决其快速感知与协调运动难题。提出一种增强物理信息的强化学习框架，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2509.21690](https://arxiv.org/pdf/2509.21690)**

> **作者:** Muqun Hu; Wenxi Chen; Wenjing Li; Falak Mandali; Zijian He; Renhong Zhang; Praveen Krisna; Katherine Christian; Leo Benaharon; Dizhi Ma; Karthik Ramani; Yan Gu
>
> **摘要:** Humanoid table tennis (TT) demands rapid perception, proactive whole-body motion, and agile footwork under strict timing--capabilities that remain difficult for end-to-end control policies. We propose a reinforcement learning (RL) framework that maps ball-position observations directly to whole-body joint commands for both arm striking and leg locomotion, strengthened by predictive signals and dense, physics-guided rewards. A lightweight learned predictor, fed with recent ball positions, estimates future ball states and augments the policy's observations for proactive decision-making. During training, a physics-based predictor supplies precise future states to construct dense, informative rewards that lead to effective exploration. The resulting policy attains strong performance across varied serve ranges (hit rate$\geq$96% and success rate$\geq$92%) in simulations. Ablation studies confirm that both the learned predictor and the predictive reward design are critical for end-to-end learning. Deployed zero-shot on a physical Booster T1 humanoid with 23 revolute joints, the policy produces coordinated lateral and forward-backward footwork with accurate, fast returns, suggesting a practical path toward versatile, competitive humanoid TT. We have open-sourced our RL training code at: this https URL
>
---
#### [replaced 057] DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
- **分类: cs.RO**

- **简介: 该论文提出DiT4DiT模型，解决机器人控制中动态与动作联合建模问题，通过视频生成提升策略学习效率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.10448](https://arxiv.org/pdf/2603.10448)**

> **作者:** Teli Ma; Jia Zheng; Zifan Wang; Chunli Jiang; Andy Cui; Junwei Liang; Shuo Yang
>
> **备注:** this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising paradigm for robot learning, but their representations are still largely inherited from static image-text pretraining, leaving physical dynamics to be learned from comparatively limited action data. Generative video models, by contrast, encode rich spatiotemporal structure and implicit physics, making them a compelling foundation for robotic manipulation. But their potentials are not fully explored in the literature. To bridge the gap, we introduce DiT4DiT, an end-to-end Video-Action Model that couples a video Diffusion Transformer with an action Diffusion Transformer in a unified cascaded framework. Instead of relying on reconstructed future frames, DiT4DiT extracts intermediate denoising features from the video generation process and uses them as temporally grounded conditions for action prediction. We further propose a dual flow-matching objective with decoupled timesteps and noise scales for video prediction, hidden-state extraction, and action inference, enabling coherent joint training of both modules. Across simulation and real-world benchmarks, DiT4DiT achieves state-of-the-art results, reaching average success rates of 98.6% on LIBERO and 50.8% on RoboCasa GR1 while using substantially less training data. On the Unitree G1 robot, it also delivers superior real-world performance and strong zero-shot generalization. Importantly, DiT4DiT improves sample efficiency by over 10x and speeds up convergence by up to 7x, demonstrating that video generation can serve as an effective scaling proxy for robot policy learning. We release code and models at this https URL.
>
---
#### [replaced 058] Towards Unified World Models for Visual Navigation via Memory-Augmented Planning and Foresight
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于视觉导航任务，旨在解决模块化系统导致的状态-动作不匹配和适应性差的问题。提出UniWM，整合视觉预测与规划，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2510.08713](https://arxiv.org/pdf/2510.08713)**

> **作者:** Yifei Dong; Fengyi Wu; Guangyu Chen; Lingdong Kong; Xu Zhu; Qiyu Hu; Yuxuan Zhou; Jingdong Sun; Jun-Yan He; Qi Dai; Alexander G. Hauptmann; Zhi-Qi Cheng
>
> **备注:** 21 pages, 12 figures, code: this https URL
>
> **摘要:** Enabling embodied agents to imagine future states is essential for robust and generalizable visual navigation. Yet, state-of-the-art systems typically rely on modular designs that decouple navigation planning from visual world modeling, which often induces state-action misalignment and weak adaptability in novel or dynamic scenarios. We propose UniWM, a unified, memory-augmented world model that integrates egocentric visual foresight and planning within a single multimodal autoregressive backbone. UniWM explicitly grounds action selection in visually imagined outcomes, tightly aligning prediction with control. Meanwhile, a hierarchical memory mechanism fuses short-term perceptual cues with longer-term trajectory context, supporting stable and coherent reasoning over extended horizons. Extensive experiments on four challenging benchmarks (Go Stanford, ReCon, SCAND, HuRoN) and the 1X Humanoid Dataset show that UniWM improves navigation success rates by up to 30%, substantially reduces trajectory errors against strong baselines, generalizes zero-shot to the unseen TartanDrive dataset, and scales naturally to high-dimensional humanoid control. These results position UniWM as a principled step toward unified, imagination-driven embodied navigation. The code and models are available at this https URL.
>
---
#### [replaced 059] Learning collision risk proactively from naturalistic driving data at scale
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于交通事故预警任务，旨在主动识别潜在碰撞风险。通过分析真实驾驶数据，提出GSSM模型，无需标注即可准确预测碰撞，提升自动驾驶安全性。**

- **链接: [https://arxiv.org/pdf/2505.13556](https://arxiv.org/pdf/2505.13556)**

> **作者:** Yiru Jiao; Simeon C. Calvert; Sander van Cranenburgh; Hans van Lint
>
> **备注:** Officially published in Nature Machine Intelligence. Equation (15) in the previous versions was wrong, which has been corrected since v4
>
> **摘要:** Accurately and proactively alerting drivers or automated systems to emerging collisions is crucial for road safety, particularly in highly interactive and complex urban environments. Existing methods either require labour-intensive annotation of sparse risk, struggle to consider varying contextual factors, or are tailored to limited scenarios. Here we present the Generalised Surrogate Safety Measure (GSSM), a data-driven approach that learns collision risk from naturalistic driving without the need for crash or risk labels. Trained over multiple datasets and evaluated on 2,591 real-world crashes and near-crashes, a basic GSSM using only instantaneous motion kinematics achieves an area under the precision-recall curve of 0.9, and secures a median time advance of 2.6 seconds to prevent potential collisions. Incorporating additional interaction patterns and contextual factors provides further performance gains. Across interaction scenarios such as rear-end, merging, and turning, GSSM consistently outperforms existing baselines in accuracy and timeliness. These results establish GSSM as a scalable, context-aware, and generalisable foundation to identify risky interactions before they become unavoidable, supporting proactive safety in autonomous driving systems and traffic incident management. Code and experiment data are openly accessible at this https URL.
>
---
#### [replaced 060] HortiMulti: A Multi-Sensor Dataset for Localisation and Mapping in Horticultural Polytunnels
- **分类: cs.RO**

- **简介: 该论文提出HortiMulti数据集，用于解决农业机器人在多季节温室中的定位与建图问题，包含多种传感器数据及真实轨迹，支持多模态SLAM研究。**

- **链接: [https://arxiv.org/pdf/2603.20150](https://arxiv.org/pdf/2603.20150)**

> **作者:** Shuoyuan Xu; Zhipeng Zhong; Tiago Barros; Matthew Coombes; Cristiano Premebida; Hao Wu; Cunjia Liu
>
> **摘要:** Agricultural robotics is gaining increasing relevance in both research and real-world deployment. As these systems are expected to operate autonomously in more complex tasks, the availability of representative real-world datasets becomes essential. While domains such as urban and forestry robotics benefit from large and established benchmarks, horticultural environments remain comparatively under-explored despite the economic significance of this sector. To address this gap, we present HortiMulti, a multimodal, cross-season dataset collected in commercial strawberry and raspberry polytunnels across an entire growing season, capturing substantial appearance variation, dynamic foliage, specular reflections from plastic covers, severe perceptual aliasing, and GNSS-unreliable conditions, all of which directly degrade existing localisation and perception algorithms. The sensor suite includes two 3D LiDARs, four RGB cameras, an IMU, GNSS, and wheel odometry. Ground truth trajectories are derived from a combination of Total Station surveying, AprilTag fiducial markers, and LiDAR-inertial odometry, spanning dense, sparse, and marker-free coverage to support evaluation under both controlled and realistic conditions. We release time-synchronised raw measurements, calibration files, reference trajectories, and baseline benchmarks for visual, LiDAR, and multi-sensor SLAM, with results confirming that current state-of-the-art methods remain inadequate for reliable polytunnel deployment, establishing HortiMulti as a one-stop resource for developing and testing robotic perception systems in horticulture environments.
>
---
#### [replaced 061] Reactive Slip Control in Multifingered Grasping: Hybrid Tactile Sensing and Internal-Force Optimization
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多指抓取任务，旨在解决抓取稳定性问题。通过融合触觉感知与内部力优化，实现快速滑动检测与抓取稳定。**

- **链接: [https://arxiv.org/pdf/2602.16127](https://arxiv.org/pdf/2602.16127)**

> **作者:** Théo Ayral; Saifeddine Aloui; Mathieu Grossard
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA), 2026
>
> **摘要:** We build a low-level reflex control layer driven by fast tactile feedback for multifinger grasp stabilization. Our hybrid approach combines learned tactile slip detection with model-based internal-force control to halt in-hand slip while preserving the object-level wrench. The multimodal tactile stack integrates piezoelectric sensing (PzE) for fast slip cues and piezoresistive arrays (PzR) for contact localization, enabling online construction of a contact-centric grasp representation without prior object knowledge. Experiments demonstrate reactive stabilization of multifingered grasps under external perturbations, without explicit friction models or direct force sensing. In controlled trials, slip onset is detected after 20.4 +/- 6 ms. The framework yields a theoretical grasp response latency on the order of 30 ms, with grasp-model updates in less than 5 ms and internal-force selection in about 4 ms. The analysis supports the feasibility of sub-50 ms tactile-driven grasp responses, aligned with human reflex baselines.
>
---
#### [replaced 062] Fast Path Planning for Autonomous Vehicle Parking with Safety-Guarantee using Hamilton-Jacobi Reachability
- **分类: cs.RO**

- **简介: 该论文属于自主车辆停车路径规划任务，解决复杂停车场景下的快速安全路径生成问题。提出HJBA*算法，结合HJ可达分析与双向A*搜索，确保路径安全且计算高效。**

- **链接: [https://arxiv.org/pdf/2310.15190](https://arxiv.org/pdf/2310.15190)**

> **作者:** Xuemin Chi; Jun Zeng; Jihao Huang; Zhitao Liu; Hongye Su
>
> **备注:** accepted by IEEE Transactions on Vehicular Technology
>
> **摘要:** We present a fast planning architecture called Hamilton-Jacobi-based bidirectional A* (HJBA*) to solve general tight parking scenarios. The algorithm is a two-layer composed of a high-level HJ-based reachability analysis and a lower-level bidirectional A* search algorithm. In high-level reachability analysis, a backward reachable tube (BRT) concerning vehicle dynamics is computed by the HJ analysis and it intersects with a safe set to get a safe reachable set. The safe set is defined by constraints of positive signed distances for obstacles in the environment and computed by solving QP optimization problems offline. For states inside the intersection set, i.e., the safe reachable set, the computed backward reachable tube ensures they are reachable subjected to system dynamics and input bounds, and the safe set guarantees they satisfy parking safety with respect to obstacles in different shapes. For online computation, randomized states are sampled from the safe reachable set, and used as heuristic guide points to be considered in the bidirectional A* search. The bidirectional A* search is paralleled for each randomized state from the safe reachable set. We show that the proposed two-level planning algorithm is able to solve different parking scenarios effectively and computationally fast for typical parking requests. We validate our algorithm through simulations in large-scale randomized parking scenarios and demonstrate it to be able to outperform other state-of-the-art parking planning algorithms.
>
---
#### [replaced 063] CAR: Cross-Vehicle Kinodynamics Adaptation via Mobility Representation
- **分类: cs.RO**

- **简介: 该论文属于自主移动任务，解决异构车辆间动力学迁移问题。提出CAR框架，通过共享潜在空间实现快速动力学适配，减少数据收集和计算成本。**

- **链接: [https://arxiv.org/pdf/2603.06866](https://arxiv.org/pdf/2603.06866)**

> **作者:** Tong Xu; Chenhui Pan; Xuesu Xiao
>
> **摘要:** Developing autonomous off-road mobility typically requires either extensive, platform-specific data collection or relies on simplified abstractions, such as unicycle or bicycle models, that fail to capture the complex kinodynamics of diverse platforms, ranging from wheeled to tracked vehicles. This limitation hinders scalability across evolving heterogeneous autonomous robot fleets. To address this challenge, we propose Cross-vehicle kinodynamics Adaptation via mobility Representation (CAR), a novel framework that enables rapid mobility transfer to new vehicles. CAR employs a Transformer encoder with Adaptive Layer Normalization to embed vehicle trajectory transitions and physical configurations into a shared mobility latent space. By identifying and extracting commonality from nearest neighbors within this latent space, our approach enables rapid kinodynamics adaptation to novel platforms with minimal data collection and computational overhead. We evaluate CAR using the Verti-Bench simulator, built on the Chrono multi-physics engine, and validate its performance on four distinct physical configurations of the Verti-4-Wheeler platform. With only one minute of new trajectory data, CAR achieves up to 67.2% reduction in prediction error compared to direct neighbor transfer across diverse unseen vehicle configurations, demonstrating the effectiveness of cross-vehicle mobility knowledge transfer in both simulated and real-world environments.
>
---
#### [replaced 064] Reward Evolution with Graph-of-Thoughts: A Bi-Level Language Model Framework for Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于强化学习领域，解决奖励函数设计难题。通过结合大语言模型与视觉语言模型，利用图思维结构进行奖励进化，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2509.16136](https://arxiv.org/pdf/2509.16136)**

> **作者:** Changwei Yao; Xinzi Liu; Chen Li; Marios Savvides
>
> **摘要:** Designing effective reward functions remains a major challenge in reinforcement learning (RL), often requiring considerable human expertise and iterative refinement. Recent advances leverage Large Language Models (LLMs) for automated reward design, but these approaches are limited by hallucinations, reliance on human feedback, and challenges with handling complex, multi-step tasks. In this work, we introduce Reward Evolution with Graph-of-Thoughts (RE-GoT), a novel bi-level framework that enhances LLMs with structured graph-based reasoning and integrates Visual Language Models (VLMs) for automated rollout evaluation. RE-GoT first decomposes tasks into text-attributed graphs, enabling comprehensive analysis and reward function generation, and then iteratively refines rewards using visual feedback from VLMs without human intervention. Extensive experiments on 10 RoboGen and 4 ManiSkill2 tasks demonstrate that RE-GoT consistently outperforms existing LLM-based baselines. On RoboGen, our method improves average task success rates by 32.25%, with notable gains on complex multi-step tasks. On ManiSkill2, RE-GoT achieves an average success rate of 93.73% across four diverse manipulation tasks, significantly surpassing prior LLM-based approaches and even exceeding expert-designed rewards. Our results indicate that combining LLMs and VLMs with graph-of-thoughts reasoning provides a scalable and effective solution for autonomous reward evolution in RL.
>
---
#### [replaced 065] A User-driven Design Framework for Robotaxi
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决 robotaxi 设计中的用户需求与体验问题。通过访谈和体验研究，提出用户驱动的设计框架，提升信任与透明度。**

- **链接: [https://arxiv.org/pdf/2602.19107](https://arxiv.org/pdf/2602.19107)**

> **作者:** Yue Deng; Changyang He
>
> **摘要:** Robotaxis are emerging as a promising form of urban mobility, but removing human drivers fundamentally reshapes passenger-vehicle interaction and raises new design challenges. To inform robotaxi design based on real-world experience, we conducted 18 semi-structured interviews and autoethnographic ride experiences to examine users' perceptions, experiences, and expectations for robotaxi design. We found that users valued benefits such as increased agency and consistent driving. However, they also encountered challenges such as limited flexibility, insufficient transparency, and emergency handling concerns. Notably, users perceived robotaxis not merely as a mode of transportation, but as autonomous, semi-private transitional spaces, which made users feel less socially intrusive to engage in personal activities. Safety perceptions were polarized: some felt anxiety about reduced control, while others viewed robotaxis as safer than humans due to their cautious, law-abiding nature. Based on the findings, we propose a user-driven design framework spanning hailing, pick-up, traveling, and drop-off phases to support trustworthy, transparent, and accountable robotaxi design.
>
---
#### [replaced 066] KeySG: Hierarchical Keyframe-Based 3D Scene Graphs
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出KeySG，用于构建层次化3D场景图，解决传统方法语义有限和扩展性差的问题，通过关键帧和多模态信息提升场景理解与推理效率。**

- **链接: [https://arxiv.org/pdf/2510.01049](https://arxiv.org/pdf/2510.01049)**

> **作者:** Abdelrhman Werby; Dennis Rotondi; Fabio Scaparro; Kai O. Arras
>
> **备注:** Code and video are available at this https URL
>
> **摘要:** In recent years, 3D scene graphs have emerged as a powerful world representation, offering both geometric accuracy and semantic richness. Combining 3D scene graphs with large language models enables robots to reason, plan, and navigate in complex human-centered environments. However, current approaches for constructing 3D scene graphs are semantically limited to a predefined set of relationships, and their serialization in large environments can easily exceed an LLM's context window. We introduce KeySG, a framework that represents 3D scenes as a hierarchical graph consisting of floors, rooms, objects, and functional elements, where nodes are augmented with multi-modal information extracted from keyframes selected to optimize geometric and visual coverage. The keyframes allow us to efficiently leverage VLMs to extract scene information, alleviating the need to explicitly model relationship edges between objects, enabling more general, task-agnostic reasoning and planning. Our approach can process complex and ambiguous queries while mitigating the scalability issues associated with large scene graphs by utilizing a hierarchical multi-modal retrieval-augmented generation (RAG) pipeline to extract relevant context from the graph. Evaluated across three distinct benchmarks, 3D object semantic segmentation, functional element segmentation, and complex query retrieval, KeySG outperforms prior approaches on most metrics, demonstrating its superior semantic richness and efficiency.
>
---
#### [replaced 067] PixelVLA: Advancing Pixel-level Understanding in Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决现有模型在像素级理解与依赖文本提示的问题。提出PixelVLA，结合多尺度像素编码与视觉提示，提升机器人控制性能。**

- **链接: [https://arxiv.org/pdf/2511.01571](https://arxiv.org/pdf/2511.01571)**

> **作者:** Wenqi Liang; Gan Sun; Yao He; Jiahua Dong; Suyan Dai; Ivan Laptev; Salman Khan; Yang Cong
>
> **备注:** 17pages,7 figures, 5 tabels
>
> **摘要:** Vision-Language-Action models (VLAs) are emerging as powerful tools for learning generalizable visuomotor control policies. However, current VLAs are mostly trained on large-scale image-text-action data and remain limited in two key ways: (i) they struggle with pixel-level scene understanding, and (ii) they rely heavily on textual prompts, which reduces their flexibility in real-world settings. To address these challenges, we introduce PixelVLA, the first VLA model designed to support both pixel-level reasoning and multimodal prompting with text and visual inputs. Our approach is built on a new visuomotor instruction tuning framework that integrates a multiscale pixel-aware encoder with a visual promptaware encoder. To train PixelVLA effectively, we further propose a two-stage automated annotation pipeline that generates Pixel-160K, a large-scale dataset with pixel-level annotations derived from existing robot data. Experiments on three standard VLA benchmarks and two VLA model variants show that PixelVLA improves manipulation success rates by 10.1%-28.7% over OpenVLA, while requiring only 1.5% of its pretraining cost. These results demonstrate that PixelVLA can be integrated into existing VLAs to enable more accurate, efficient, and versatile robot control in complex environments.
>
---
#### [replaced 068] Learning to Sample: Reinforcement Learning-Guided Sampling for Autonomous Vehicle Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于自主车辆运动规划任务，旨在解决采样效率低的问题。通过强化学习引导采样，提升轨迹可行性，减少样本数量和运行时间。**

- **链接: [https://arxiv.org/pdf/2509.24313](https://arxiv.org/pdf/2509.24313)**

> **作者:** Korbinian Moller; Roland Stroop; Mattia Piccinini; Alexander Langmann; Johannes Betz
>
> **备注:** 8 pages, submitted to the IEEE for possible publication
>
> **摘要:** Sampling-based motion planning is a well-established approach in autonomous driving, valued for its modularity and analytical tractability. In complex urban scenarios, however, uniform or heuristic sampling often produces many infeasible or irrelevant trajectories. We address this limitation with a hybrid framework that learns where to sample while keeping trajectory generation and evaluation fully analytical and verifiable. A reinforcement learning (RL) agent guides the sampling process toward regions of the action space likely to yield feasible trajectories, while evaluation and final selection remains governed by deterministic feasibility checks and cost functions. We couple the RL sampler with a world model (WM) based on a decodable deep set encoder, enabling both variable numbers of traffic participants and reconstructable latent representations. The approach is evaluated in the CommonRoad (CR) simulation environment and compared against uniform-sampling baselines, showing up to 99% fewer required samples and a runtime reduction of up to 84% while maintaining planning quality in terms of success and collision-free rates. These improvements lead to faster, more reliable decision-making for autonomous vehicles in urban environments.
>
---
#### [replaced 069] OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于接触丰富机器人操作任务，旨在解决仅靠视觉难以准确感知接触力和摩擦变化的问题。作者构建了大规模多模态数据集，并提出OmniVTA框架，融合视觉与触觉信息实现精准控制。**

- **链接: [https://arxiv.org/pdf/2603.19201](https://arxiv.org/pdf/2603.19201)**

> **作者:** Yuhang Zheng; Songen Gu; Weize Li; Yupeng Zheng; Yujie Zang; Shuai Tian; Xiang Li; Ce Hao; Chen Gao; Si Liu; Haoran Li; Yilun Chen; Shuicheng Yan; Wenchao Ding
>
> **备注:** TARS Robotics Project Page: this https URL
>
> **摘要:** Contact-rich manipulation tasks, such as wiping and assembly, require accurate perception of contact forces, friction changes, and state transitions that cannot be reliably inferred from vision alone. Despite growing interest in visuo-tactile manipulation, progress is constrained by two persistent limitations: existing datasets are small in scale and narrow in task coverage, and current methods treat tactile signals as passive observations rather than using them to model contact dynamics or enable closed-loop control explicitly. In this paper, we present \textbf{OmniViTac}, a large-scale visuo-tactile-action dataset comprising $21{,}000+$ trajectories across $86$ tasks and $100+$ objects, organized into six physics-grounded interaction patterns. Building on this dataset, we propose \textbf{OmniVTA}, a world-model-based visuo-tactile manipulation framework that integrates four tightly coupled modules: a self-supervised tactile encoder, a two-stream visuo-tactile world model for predicting short-horizon contact evolution, a contact-aware fusion policy for action generation, and a 60Hz reflexive controller that corrects deviations between predicted and observed tactile signals in a closed loop. Real-robot experiments across all six interaction categories show that OmniVTA outperforms existing methods and generalizes well to unseen objects and geometric configurations, confirming the value of combining predictive contact modeling with high-frequency tactile feedback for contact-rich manipulation. All data, models, and code will be made publicly available on the project website at this https URL.
>
---
