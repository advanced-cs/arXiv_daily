# 机器人 cs.RO

- **最新发布 77 篇**

- **更新 27 篇**

## 最新发布

#### [new 001] Navigation Around Unknown Space Objects Using Visible-Thermal Image Fusion
- **分类: cs.RO; cs.CV**

- **简介: 该论文属空间导航任务，旨在解决未知空间目标（如碎片、卫星）在光照变化下的鲁棒导航问题。提出可见光与热红外图像像素级融合方法，通过仿真数据验证其显著提升单目SLAM导航精度，优于单一模态。**

- **链接: [https://arxiv.org/pdf/2512.12203v1](https://arxiv.org/pdf/2512.12203v1)**

> **作者:** Eric J. Elias; Michael Esswein; Jonathan P. How; David W. Miller
>
> **备注:** 18 pages, 11 figures. To be published in proceedings of AIAA SCITECH 2026 Forum
>
> **摘要:** As the popularity of on-orbit operations grows, so does the need for precise navigation around unknown resident space objects (RSOs) such as other spacecraft, orbital debris, and asteroids. The use of Simultaneous Localization and Mapping (SLAM) algorithms is often studied as a method to map out the surface of an RSO and find the inspector's relative pose using a lidar or conventional camera. However, conventional cameras struggle during eclipse or shadowed periods, and lidar, though robust to lighting conditions, tends to be heavier, bulkier, and more power-intensive. Thermal-infrared cameras can track the target RSO throughout difficult illumination conditions without these limitations. While useful, thermal-infrared imagery lacks the resolution and feature-richness of visible cameras. In this work, images of a target satellite in low Earth orbit are photo-realistically simulated in both visible and thermal-infrared bands. Pixel-level fusion methods are used to create visible/thermal-infrared composites that leverage the best aspects of each camera. Navigation errors from a monocular SLAM algorithm are compared between visible, thermal-infrared, and fused imagery in various lighting and trajectories. Fused imagery yields substantially improved navigation performance over visible-only and thermal-only methods.
>
---
#### [new 002] Multi-Robot Motion Planning from Vision and Language using Heat-Inspired Diffusion
- **分类: cs.RO**

- **简介: 该论文面向多机器人运动规划任务，解决现有扩散模型在语言条件化、多机协同、计算开销大及几何可达性推理弱等方面的不足。提出LCHD框架：融合CLIP语义先验与热启发碰撞感知扩散核，实现端到端视觉-语言驱动的无碰撞轨迹生成，无需显式环境建模。**

- **链接: [https://arxiv.org/pdf/2512.13090v1](https://arxiv.org/pdf/2512.13090v1)**

> **作者:** Jebeom Chae; Junwoo Chang; Seungho Yeom; Yujin Kim; Jongeun Choi
>
> **摘要:** Diffusion models have recently emerged as powerful tools for robot motion planning by capturing the multi-modal distribution of feasible trajectories. However, their extension to multi-robot settings with flexible, language-conditioned task specifications remains limited. Furthermore, current diffusion-based approaches incur high computational cost during inference and struggle with generalization because they require explicit construction of environment representations and lack mechanisms for reasoning about geometric reachability. To address these limitations, we present Language-Conditioned Heat-Inspired Diffusion (LCHD), an end-to-end vision-based framework that generates language-conditioned, collision-free trajectories. LCHD integrates CLIP-based semantic priors with a collision-avoiding diffusion kernel serving as a physical inductive bias that enables the planner to interpret language commands strictly within the reachable workspace. This naturally handles out-of-distribution scenarios -- in terms of reachability -- by guiding robots toward accessible alternatives that match the semantic intent, while eliminating the need for explicit obstacle information at inference time. Extensive evaluations on diverse real-world-inspired maps, along with real-robot experiments, show that LCHD consistently outperforms prior diffusion-based planners in success rate, while reducing planning latency.
>
---
#### [new 003] ReGlove: A Soft Pneumatic Glove for Activities of Daily Living Assistance via Wrist-Mounted Vision
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ReGlove系统，属辅助技术任务，旨在解决上肢障碍者因EMG信号不可靠或设备昂贵而难以使用助手机器手的问题。作者将商用气动康复手套改造为视觉引导式助手机构，利用腕戴摄像头与树莓派实时YOLO模型实现高精度、低延迟抓取识别，并在ADL任务中验证有效性与低成本可行性。**

- **链接: [https://arxiv.org/pdf/2512.11824v1](https://arxiv.org/pdf/2512.11824v1)**

> **作者:** Rosh Ho; Jian Zhang
>
> **摘要:** This paper presents ReGlove, a system that converts low-cost commercial pneumatic rehabilitation gloves into vision-guided assistive orthoses. Chronic upper-limb impairment affects millions worldwide, yet existing assistive technologies remain prohibitively expensive or rely on unreliable biological signals. Our platform integrates a wrist-mounted camera with an edge-computing inference engine (Raspberry Pi 5) to enable context-aware grasping without requiring reliable muscle signals. By adapting real-time YOLO-based computer vision models, the system achieves \SI{96.73}{\percent} grasp classification accuracy with sub-\SI{40.00}{\milli\second} end-to-end latency. Physical validation using standardized benchmarks shows \SI{82.71}{\percent} success on YCB object manipulation and reliable performance across \SI{27.00}{} Activities of Daily Living (ADL) tasks. With a total cost under \$\SI{250.00}{} and exclusively commercial components, ReGlove provides a technical foundation for accessible, vision-based upper-limb assistance that could benefit populations excluded from traditional EMG-controlled devices.
>
---
#### [new 004] Modified Hybrid A* Collision-Free Path-Planning for Automated Reverse Parking
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶路径规划任务，旨在解决自动倒车泊车中狭窄空间下可行且无碰撞路径难寻的问题。作者改进Hybrid A*算法，引入单轨运动学模型生成可行轨迹，并结合膨胀占据栅格图实现静态障碍物避让。**

- **链接: [https://arxiv.org/pdf/2512.12021v1](https://arxiv.org/pdf/2512.12021v1)**

> **作者:** Xincheng Cao; Haochong Chen; Bilin Aksun-Guvenc; Levent Guvenc
>
> **摘要:** Parking a vehicle in tight spaces is a challenging task to perform due to the scarcity of feasible paths that are also collision-free. This paper presents a strategy to tackle this kind of maneuver with a modified Hybrid-A* path-planning algorithm that combines the feasibility guarantee inherent in the standard Hybrid A* algorithm with the addition of static obstacle collision avoidance. A kinematic single-track model is derived to describe the low-speed motion of the vehicle, which is subsequently used as the motion model in the Hybrid A* path-planning algorithm to generate feasible motion primitive branches. The model states are also used to reconstruct the vehicle centerline, which, in conjunction with an inflated binary occupancy map, facilitates static obstacle collision avoidance functions. Simulation study and animation are set up to test the efficacy of the approach, and the proposed algorithm proves to consistently provide kinematically feasible trajectories that are also collision-free.
>
---
#### [new 005] Evaluating the Navigation Capabilities of a Modified COAST Guidewire Robot in an Anatomical Phantom Model
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于医疗机器人任务，旨在解决手动导丝导航精度低、难度大问题。研究开发并评估了一种简化版双管COAST可操控导丝机器人，在搏动流解剖体模中验证其在迂曲血管中的有效导航能力。**

- **链接: [https://arxiv.org/pdf/2512.13477v1](https://arxiv.org/pdf/2512.13477v1)**

> **作者:** Timothy A. Brumfiel; Revanth Konda; Drew Elliott; Jaydev P. Desai
>
> **备注:** Presented at the 14th Conference on New Technologies for Computer and Robot Assisted Surgery (CRAS 2025)
>
> **摘要:** To address the issues that arise due to the manual navigation of guidewires in endovascular interventions, research in medical robotics has taken a strong interest in developing robotically steerable guidewires, which offer the possibility of enhanced maneuverability and navigation, as the tip of the guidewire can be actively steered. The COaxially Aligned STeerable (COAST) guidewire robot has the ability to generate a wide variety of motions including bending motion with different bending lengths, follow-the-leader motion, and feedforward motion. In our past studies, we have explored different designs of the COAST guidewire robot and developed modeling, control, and sensing strategies for the COAST guidewire robot. In this study, the performance of a modified COAST guidewire robot is evaluated by conducting navigation experiments in an anatomical phantom model with pulsatile flow. The modified COAST guidewire robot is a simplified version of the COAST guidewire robot and consists of two tubes as opposed to three tubes. Through this study, we demonstrate the effectiveness of the modified COAST guidewire robot in navigating the tortuous phantom vasculature.
>
---
#### [new 006] VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer
- **分类: cs.RO; eess.SY**

- **简介: 该论文面向机器人操纵任务，解决VLA模型在真实环境中安全与任务性能难以兼顾的问题。提出VLSA架构AEGIS，通过即插即用的安全约束层（基于控制屏障函数）保障碰撞规避，同时保持指令遵循能力，并构建SafeLIBERO基准验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.11891v1](https://arxiv.org/pdf/2512.11891v1)**

> **作者:** Songqiao Hu; Zeyi Liu; Shuang Liu; Jun Cen; Zihan Meng; Xiao He
>
> **备注:** 20 pages, 14 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in generalizing across diverse robotic manipulation tasks. However, deploying these models in unstructured environments remains challenging due to the critical need for simultaneous task compliance and safety assurance, particularly in preventing potential collisions during physical interactions. In this work, we introduce a Vision-Language-Safe Action (VLSA) architecture, named AEGIS, which contains a plug-and-play safety constraint (SC) layer formulated via control barrier functions. AEGIS integrates directly with existing VLA models to improve safety with theoretical guarantees, while maintaining their original instruction-following performance. To evaluate the efficacy of our architecture, we construct a comprehensive safety-critical benchmark SafeLIBERO, spanning distinct manipulation scenarios characterized by varying degrees of spatial complexity and obstacle intervention. Extensive experiments demonstrate the superiority of our method over state-of-the-art baselines. Notably, AEGIS achieves a 59.16% improvement in obstacle avoidance rate while substantially increasing the task execution success rate by 17.25%. To facilitate reproducibility and future research, we make our code, models, and the benchmark datasets publicly available at https://vlsa-aegis.github.io/.
>
---
#### [new 007] CAR-CHASE: Car-Like Robot Conflict-Aware Heuristic Adaptive Search Enhancement
- **分类: cs.RO**

- **简介: 该论文面向多智能体路径规划（MAPF）中类车机器人运动规划任务，解决CL-CBS算法因连续时间与运动学约束导致的启发式计算昂贵、缓存失效问题。提出CAR-CHASE方法，含冲突感知启发式缓存与自适应混合启发式，显著加速搜索并提升求解成功率。**

- **链接: [https://arxiv.org/pdf/2512.12243v1](https://arxiv.org/pdf/2512.12243v1)**

> **作者:** HT To; S Nguyen; NH Pham
>
> **摘要:** Multi-Agent Path Finding (MAPF) for car-like robots, addressed by algorithms such as Conflict-Based Search with Continuous Time (CL-CBS), faces significant computational challenges due to expensive kinematic heuristic calculations. Traditional heuristic caching assumes that the heuristic function depends only on the state, which is incorrect in CBS where constraints from conflict resolution make the search space context-dependent. We propose \textbf{CAR-CHASE} (Car-Like Robot Conflict-Aware Heuristic Adaptive Search Enhancement), a novel approach that combines \textbf{conflict-aware heuristic caching} -- which caches heuristic values based on both state and relevant constraint context -- with an \textbf{adaptive hybrid heuristic} that intelligently switches between fast approximate and exact computations. Our key innovations are (1) a compact \emph{conflict fingerprint} that efficiently encodes which constraints affect a state's heuristic, (2) a relevance filter using spatial, temporal, and geometric criteria, and (3) an adaptive switching strategy with theoretical quality bounds. Experimental evaluation on 480 benchmark instances with varying agent counts (10 to 30) and obstacle densities (0\% and 50\%) demonstrates a geometric mean speedup of 2.46$\times$ over the baseline CL-CBS implementation while maintaining solution optimality. The optimizations improve success rate from 77.9\% to 84.8\% (+6.9 percentage points), reduce total runtime by 70.1\%, and enable solving 33 additional instances that previously timed out. Performance gains scale with problem complexity, reaching up to 4.06$\times$ speedup for challenging 30-agent obstacle scenarios. Our techniques are general and applicable to other CBS variants.
>
---
#### [new 008] Aion: Towards Hierarchical 4D Scene Graphs with Temporal Flow Dynamics
- **分类: cs.RO; cs.CV**

- **简介: 该论文属自动驾驶场景理解任务，旨在解决动态环境中时空表征缺乏语义层次与可扩展性的问题。提出Aion框架，将稀疏图结构的运动流（MoD）嵌入分层3D场景图，实现语义感知、可解释且可扩展的4D时空建模。**

- **链接: [https://arxiv.org/pdf/2512.11903v1](https://arxiv.org/pdf/2512.11903v1)**

> **作者:** Iacopo Catalano; Eduardo Montijano; Javier Civera; Julio A. Placed; Jorge Pena-Queralta
>
> **摘要:** Autonomous navigation in dynamic environments requires spatial representations that capture both semantic structure and temporal evolution. 3D Scene Graphs (3DSGs) provide hierarchical multi-resolution abstractions that encode geometry and semantics, but existing extensions toward dynamics largely focus on individual objects or agents. In parallel, Maps of Dynamics (MoDs) model typical motion patterns and temporal regularities, yet are usually tied to grid-based discretizations that lack semantic awareness and do not scale well to large environments. In this paper we introduce Aion, a framework that embeds temporal flow dynamics directly within a hierarchical 3DSG, effectively incorporating the temporal dimension. Aion employs a graph-based sparse MoD representation to capture motion flows over arbitrary time intervals and attaches them to navigational nodes in the scene graph, yielding more interpretable and scalable predictions that improve planning and interaction in complex dynamic environments.
>
---
#### [new 009] START: Traversing Sparse Footholds with Terrain Reconstruction
- **分类: cs.RO**

- **简介: 该论文面向四足机器人稀疏落脚点地形穿越任务，解决现有方法泛化差、感知不精准或学习低效问题。提出START单阶段学习框架，仅用低成本本体视觉与本体感知重建局部高程图，显式表征关键地形特征，实现零样本跨场景自适应稳定行走。**

- **链接: [https://arxiv.org/pdf/2512.13153v1](https://arxiv.org/pdf/2512.13153v1)**

> **作者:** Ruiqi Yu; Qianshi Wang; Hongyi Li; Zheng Jun; Zhicheng Wang; Jun Wu; Qiuguo Zhu
>
> **摘要:** Traversing terrains with sparse footholds like legged animals presents a promising yet challenging task for quadruped robots, as it requires precise environmental perception and agile control to secure safe foot placement while maintaining dynamic stability. Model-based hierarchical controllers excel in laboratory settings, but suffer from limited generalization and overly conservative behaviors. End-to-end learning-based approaches unlock greater flexibility and adaptability, but existing state-of-the-art methods either rely on heightmaps that introduce noise and complex, costly pipelines, or implicitly infer terrain features from egocentric depth images, often missing accurate critical geometric cues and leading to inefficient learning and rigid gaits. To overcome these limitations, we propose START, a single-stage learning framework that enables agile, stable locomotion on highly sparse and randomized footholds. START leverages only low-cost onboard vision and proprioception to accurately reconstruct local terrain heightmap, providing an explicit intermediate representation to convey essential features relevant to sparse foothold regions. This supports comprehensive environmental understanding and precise terrain assessment, reducing exploration cost and accelerating skill acquisition. Experimental results demonstrate that START achieves zero-shot transfer across diverse real-world scenarios, showcasing superior adaptability, precise foothold placement, and robust locomotion.
>
---
#### [new 010] NL2SpaTiaL: Generating Geometric Spatio-Temporal Logic Specifications from Natural Language for Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文属自然语言到形式化逻辑的翻译任务，旨在解决现有方法忽略操作任务中多层空间关系的问题。作者构建NL2SpaTiaL数据集（含几何时空逻辑公式与对应自然语言描述），并提出带语义校验的翻译框架，提升机器人指令理解的可解释性、可验证性与组合性。**

- **链接: [https://arxiv.org/pdf/2512.13670v1](https://arxiv.org/pdf/2512.13670v1)**

> **作者:** Licheng Luo; Yu Xia; Kaier Liang; Mingyu Cai
>
> **摘要:** Spatio-Temporal Logic (SpaTiaL) offers a principled formalism for expressing geometric spatial requirements-an essential component of robotic manipulation, where object locations, neighborhood relations, pose constraints, and interactions directly determine task success. Yet prior works have largely relied on standard temporal logic (TL), which models only robot trajectories and overlooks object-level interactions. Existing datasets built from randomly generated TL formulas paired with natural-language descriptions therefore cover temporal operators but fail to represent the layered spatial relations that manipulation tasks depend on. To address this gap, we introduce a dataset generation framework that synthesizes SpaTiaL specifications and converts them into natural-language descriptions through a deterministic, semantics-preserving back-translation procedure. This pipeline produces the NL2SpaTiaL dataset, aligning natural language with multi-level spatial relations and temporal objectives to reflect the compositional structure of manipulation tasks. Building on this foundation, we propose a translation-verification framework equipped with a language-based semantic checker that ensures the generated SpaTiaL formulas faithfully encode the semantics specified by the input description. Experiments across a suite of manipulation tasks show that SpaTiaL-based representations yield more interpretable, verifiable, and compositional grounding for instruction following. Project website: https://sites.google.com/view/nl2spatial
>
---
#### [new 011] K-VARK: Kernelized Variance-Aware Residual Kalman Filter for Sensorless Force Estimation in Collaborative Robots
- **分类: cs.RO**

- **简介: 该论文属机器人感知任务，解决协作机器人无传感器力估计精度低的问题。提出K-VARK方法：用核化运动基元建模残余扭矩的均值与输入相关异方差，结合方差感知虚拟测量更新和变分贝叶斯自适应过程噪声，在6-DoF机械臂上实现RMSE降低20%以上。**

- **链接: [https://arxiv.org/pdf/2512.13009v1](https://arxiv.org/pdf/2512.13009v1)**

> **作者:** Oğuzhan Akbıyık; Naseem Alhousani; Fares J. Abu-Dakka
>
> **摘要:** Reliable estimation of contact forces is crucial for ensuring safe and precise interaction of robots with unstructured environments. However, accurate sensorless force estimation remains challenging due to inherent modeling errors and complex residual dynamics and friction. To address this challenge, in this paper, we propose K-VARK (Kernelized Variance-Aware Residual Kalman filter), a novel approach that integrates a kernelized, probabilistic model of joint residual torques into an adaptive Kalman filter framework. Through Kernelized Movement Primitives trained on optimized excitation trajectories, K-VARK captures both the predictive mean and input-dependent heteroscedastic variance of residual torques, reflecting data variability and distance-to-training effects. These statistics inform a variance-aware virtual measurement update by augmenting the measurement noise covariance, while the process noise covariance adapts online via variational Bayesian optimization to handle dynamic disturbances. Experimental validation on a 6-DoF collaborative manipulator demonstrates that K-VARK achieves over 20% reduction in RMSE compared to state-of-the-art sensorless force estimation methods, yielding robust and accurate external force/torque estimation suitable for advanced tasks such as polishing and assembly.
>
---
#### [new 012] Tackling Snow-Induced Challenges: Safe Autonomous Lane-Keeping with Robust Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属自动驾驶控制任务，旨在解决雪天道路滑移与感知模糊导致的车道保持不稳问题。提出两种动作鲁棒的深度强化学习算法：AR-RDPG（分层式，含图像去噪与中心线提取）和AR-CADPG（端到端，融合CNN与注意力机制），并在仿真与实车中验证其有效性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.12987v1](https://arxiv.org/pdf/2512.12987v1)**

> **作者:** Amin Jalal Aghdasian; Farzaneh Abdollahi; Ali Kamali Iglie
>
> **摘要:** This paper proposes two new algorithms for the lane keeping system (LKS) in autonomous vehicles (AVs) operating under snowy road conditions. These algorithms use deep reinforcement learning (DRL) to handle uncertainties and slippage. They include Action-Robust Recurrent Deep Deterministic Policy Gradient (AR-RDPG) and end-to-end Action-Robust convolutional neural network Attention Deterministic Policy Gradient (AR-CADPG), two action-robust approaches for decision-making. In the AR-RDPG method, within the perception layer, camera images are first denoised using multi-scale neural networks. Then, the centerline coefficients are extracted by a pre-trained deep convolutional neural network (DCNN). These coefficients, concatenated with the driving characteristics, are used as input to the control layer. The AR-CADPG method presents an end-to-end approach in which a convolutional neural network (CNN) and an attention mechanism are integrated within a DRL framework. Both methods are first trained in the CARLA simulator and validated under various snowy scenarios. Real-world experiments on a Jetson Nano-based autonomous vehicle confirm the feasibility and stability of the learned policies. Among the two models, the AR-CADPG approach demonstrates superior path-tracking accuracy and robustness, highlighting the effectiveness of combining temporal memory, adversarial resilience, and attention mechanisms in AVs.
>
---
#### [new 013] Sequence of Expert: Boosting Imitation Planners for Autonomous Driving through Temporal Alternation
- **分类: cs.RO; cs.AI**

- **简介: 该论文面向自动驾驶中的模仿学习（IL）任务，旨在解决IL在闭环控制中因误差累积导致性能下降的问题。提出“专家序列”（SoE）方法，通过时间交替策略提升鲁棒性，无需增大模型或增加数据，在nuPlan上显著提升多模型性能。**

- **链接: [https://arxiv.org/pdf/2512.13094v1](https://arxiv.org/pdf/2512.13094v1)**

> **作者:** Xiang Li; Gang Liu; Weitao Zhou; Hongyi Zhu; Zhong Cao
>
> **摘要:** Imitation learning (IL) has emerged as a central paradigm in autonomous driving. While IL excels in matching expert behavior in open-loop settings by minimizing per-step prediction errors, its performance degrades unexpectedly in closed-loop due to the gradual accumulation of small, often imperceptible errors over time.Over successive planning cycles, these errors compound, potentially resulting in severe failures.Current research efforts predominantly rely on increasingly sophisticated network architectures or high-fidelity training datasets to enhance the robustness of IL planners against error accumulation, focusing on the state-level robustness at a single time point. However, autonomous driving is inherently a continuous-time process, and leveraging the temporal scale to enhance robustness may provide a new perspective for addressing this issue.To this end, we propose a method termed Sequence of Experts (SoE), a temporal alternation policy that enhances closed-loop performance without increasing model size or data requirements. Our experiments on large-scale autonomous driving benchmarks nuPlan demonstrate that SoE method consistently and significantly improves the performance of all the evaluated models, and achieves state-of-the-art performance.This module may provide a key and widely applicable support for improving the training efficiency of autonomous driving models.
>
---
#### [new 014] Enabling Autonomous Navigation in a Snake Robot through Visual-Inertial Odometry and Closed-Loop Trajectory Tracking Control
- **分类: cs.RO**

- **简介: 该论文面向蛇形机器人自主导航任务，解决其在无外部定位设施下因高自由度导致的定位与跟踪难题。提出融合视觉-惯性SLAM、质心状态估计和闭环轨迹跟踪的完整自主导航方案，并通过物理实验验证多路点精准跟踪能力。**

- **链接: [https://arxiv.org/pdf/2512.11886v1](https://arxiv.org/pdf/2512.11886v1)**

> **作者:** Mohammed Irfan Ali
>
> **摘要:** Snake robots offer exceptional mobility across extreme terrain inaccessible to conventional rovers, yet their highly articulated bodies present fundamental challenges for autonomous navigation in environments lacking external tracking infrastructure. This thesis develops a complete autonomy pipeline for COBRA, an 11 degree-of-freedom modular snake robot designed for planetary exploration. While the robot's biologically inspired serpentine gaits achieve impressive mobility, prior work has relied entirely on open-loop teleoperation. This approach integrates onboard visual-inertial SLAM, reduced-order state estimation, and closed-loop trajectory tracking to enable autonomous waypoint navigation. A depth camera paired with edge computing performs real-time localization during dynamic locomotion, validated against motion-capture ground truth to characterize drift behavior and failure modes unique to snake robot platforms. A reduced-order framework estimates Center-of-Mass pose, driving a closed-loop controller that modulates CPG gait parameters through distance-dependent yaw error blending. Physical experiments validate the complete system, demonstrating accurate multi-waypoint tracking and establishing foundations for autonomous snake robot navigation.
>
---
#### [new 015] Towards Accessible Physical AI: LoRA-Based Fine-Tuning of VLA Models for Real-World Robot Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属机器人控制任务，旨在解决大参数VLA模型在低成本机器人上部署难的问题。提出基于LoRA与量化的方法，使3.1B参数模型可在8GB显存GPU运行，并适配新机械臂；在SO101机械臂上验证按钮按压任务效果。**

- **链接: [https://arxiv.org/pdf/2512.11921v1](https://arxiv.org/pdf/2512.11921v1)**

> **作者:** Abdullah Yahya Abdullah Omaisan; Ibrahim Sheikh Mohamed
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in robotic manipulation,enabling robots to execute natural language commands through end-to-end learning from visual observations.However, deploying large-scale VLA models on affordable robotic platforms remains challenging due to computational constraints and the need for efficient adaptation to new robot embodiments. This paper presents an efficient fine-tuning methodology and real-world deployment analysis for adapting VLA models to low-cost robotic manipulation systems.We propose a resource-efficient fine-tuning strategy using Low-Rank Adaptation (LoRA) and quantization techniques that enable multi-billion parameter VLA models ( 3.1B parameters) to run on consumer-grade GPUs with 8GB VRAM. Our methodology addresses the critical challenge of adapting pre-trained VLA models to new robot embodiments with limited demonstration data, focusing on the trade-offs between frozen and unfrozen vision encoders. Through real-world deployment on the SO101 robotic arm for a button-pressing manipulation task, we demonstrate that our approach achieves effective manipulation performance while maintaining computational efficiency. We provide detailed analysis of deployment challenges, failure modes, and the relationship between training data quantity and real-world performance,trained on 200 demonstration episodes. Our results show that with proper fine-tuning methodology, VLA models can be successfully deployed on affordable robotic platforms,making advanced manipulation capabilities accessible beyond expensive research robots.
>
---
#### [new 016] Learning Terrain Aware Bipedal Locomotion via Reduced Dimensional Perceptual Representations
- **分类: cs.RO**

- **简介: 该论文面向地形自适应双足行走任务，解决RL策略在复杂真实地形中泛化性与实时性不足的问题。提出分层框架：用CNN-VAE提取低维地形感知表征，结合简化动力学建模紧凑状态；引入时序历史增强鲁棒性，并通过知识蒸馏实现深度图到潜码的端到端映射，完成仿真与初步硬件验证。**

- **链接: [https://arxiv.org/pdf/2512.12993v1](https://arxiv.org/pdf/2512.12993v1)**

> **作者:** Guillermo A. Castillo; Himanshu Lodha; Ayonga Hereid
>
> **摘要:** This work introduces a hierarchical strategy for terrain-aware bipedal locomotion that integrates reduced-dimensional perceptual representations to enhance reinforcement learning (RL)-based high-level (HL) policies for real-time gait generation. Unlike end-to-end approaches, our framework leverages latent terrain encodings via a Convolutional Variational Autoencoder (CNN-VAE) alongside reduced-order robot dynamics, optimizing the locomotion decision process with a compact state. We systematically analyze the impact of latent space dimensionality on learning efficiency and policy robustness. Additionally, we extend our method to be history-aware, incorporating sequences of recent terrain observations into the latent representation to improve robustness. To address real-world feasibility, we introduce a distillation method to learn the latent representation directly from depth camera images and provide preliminary hardware validation by comparing simulated and real sensor data. We further validate our framework using the high-fidelity Agility Robotics (AR) simulator, incorporating realistic sensor noise, state estimation, and actuator dynamics. The results confirm the robustness and adaptability of our method, underscoring its potential for hardware deployment.
>
---
#### [new 017] Iterative Tuning of Nonlinear Model Predictive Control for Robotic Manufacturing Tasks
- **分类: cs.RO; cs.LG; eess.SY; math.OC**

- **简介: 该论文属机器人制造控制任务，解决NMPC权重矩阵需反复人工调参的问题。提出基于任务反馈的迭代学习框架，构建经验灵敏度矩阵自动更新Q/R权重，无需求导；在碳纤维缠绕仿真中4次在线迭代即达近似贝叶斯优化性能。**

- **链接: [https://arxiv.org/pdf/2512.13170v1](https://arxiv.org/pdf/2512.13170v1)**

> **作者:** Deepak Ingole; Valentin Bhend; Shiva Ganesh Murali; Oliver Dobrich; Alisa Rupenayan
>
> **摘要:** Manufacturing processes are often perturbed by drifts in the environment and wear in the system, requiring control re-tuning even in the presence of repetitive operations. This paper presents an iterative learning framework for automatic tuning of Nonlinear Model Predictive Control (NMPC) weighting matrices based on task-level performance feedback. Inspired by norm-optimal Iterative Learning Control (ILC), the proposed method adaptively adjusts NMPC weights Q and R across task repetitions to minimize key performance indicators (KPIs) related to tracking accuracy, control effort, and saturation. Unlike gradient-based approaches that require differentiating through the NMPC solver, we construct an empirical sensitivity matrix, enabling structured weight updates without analytic derivatives. The framework is validated through simulation on a UR10e robot performing carbon fiber winding on a tetrahedral core. Results demonstrate that the proposed approach converges to near-optimal tracking performance (RMSE within 0.3% of offline Bayesian Optimization (BO)) in just 4 online repetitions, compared to 100 offline evaluations required by BO algorithm. The method offers a practical solution for adaptive NMPC tuning in repetitive robotic tasks, combining the precision of carefully optimized controllers with the flexibility of online adaptation.
>
---
#### [new 018] WAM-Diff: A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出WAM-Diff，一种面向自动驾驶的视觉-语言-动作（VLA）框架，解决端到端轨迹生成问题。它创新性地采用离散掩码扩散模型，结合MoE架构与在线强化学习（GSPO），实现非因果、场景感知的轨迹序列迭代优化。**

- **链接: [https://arxiv.org/pdf/2512.11872v1](https://arxiv.org/pdf/2512.11872v1)**

> **作者:** Mingwang Xu; Jiahao Cui; Feipeng Cai; Hanlin Shang; Zhihao Zhu; Shan Luan; Yifang Xu; Neng Zhang; Yaoyi Li; Jia Cai; Siyu Zhu
>
> **摘要:** End-to-end autonomous driving systems based on vision-language-action (VLA) models integrate multimodal sensor inputs and language instructions to generate planning and control signals. While autoregressive large language models and continuous diffusion policies are prevalent, the potential of discrete masked diffusion for trajectory generation remains largely unexplored. This paper presents WAM-Diff, a VLA framework that employs masked diffusion to iteratively refine a discrete sequence representing future ego-trajectories. Our approach features three key innovations: a systematic adaptation of masked diffusion for autonomous driving that supports flexible, non-causal decoding orders; scalable model capacity via a sparse MoE architecture trained jointly on motion prediction and driving-oriented visual question answering (VQA); and online reinforcement learning using Group Sequence Policy Optimization (GSPO) to optimize sequence-level driving rewards. Remarkably, our model achieves 91.0 PDMS on NAVSIM-v1 and 89.7 EPDMS on NAVSIM-v2, demonstrating the effectiveness of masked diffusion for autonomous driving. The approach provides a promising alternative to autoregressive and diffusion-based policies, supporting scenario-aware decoding strategies for trajectory generation. The code for this paper will be released publicly at: https://github.com/fudan-generative-vision/WAM-Diff
>
---
#### [new 019] RoboTracer: Mastering Spatial Trace with Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboTracer，一种面向机器人空间追踪任务的3D感知视觉语言模型。旨在解决多步、度量 grounded 的空间指代与测量难题。工作包括：设计空间编码器与回归解码器（SFT）、引入过程奖励的强化微调（RFT）、构建大规模TraceSpatial数据集及基准TraceSpatial-Bench。**

- **链接: [https://arxiv.org/pdf/2512.13660v1](https://arxiv.org/pdf/2512.13660v1)**

> **作者:** Enshen Zhou; Cheng Chi; Yibo Li; Jingkun An; Jiayuan Zhang; Shanyu Rong; Yi Han; Yuheng Ji; Mengzhen Liu; Pengwei Wang; Zhongyuan Wang; Lu Sheng; Shanghang Zhang
>
> **备注:** Project page: https://zhoues.github.io/RoboTracer
>
> **摘要:** Spatial tracing, as a fundamental embodied interaction ability for robots, is inherently challenging as it requires multi-step metric-grounded reasoning compounded with complex spatial referring and real-world metric measurement. However, existing methods struggle with this compositional task. To this end, we propose RoboTracer, a 3D-aware VLM that first achieves both 3D spatial referring and measuring via a universal spatial encoder and a regression-supervised decoder to enhance scale awareness during supervised fine-tuning (SFT). Moreover, RoboTracer advances multi-step metric-grounded reasoning via reinforcement fine-tuning (RFT) with metric-sensitive process rewards, supervising key intermediate perceptual cues to accurately generate spatial traces. To support SFT and RFT training, we introduce TraceSpatial, a large-scale dataset of 30M QA pairs, spanning outdoor/indoor/tabletop scenes and supporting complex reasoning processes (up to 9 steps). We further present TraceSpatial-Bench, a challenging benchmark filling the gap to evaluate spatial tracing. Experimental results show that RoboTracer surpasses baselines in spatial understanding, measuring, and referring, with an average success rate of 79.1%, and also achieves SOTA performance on TraceSpatial-Bench by a large margin, exceeding Gemini-2.5-Pro by 36% accuracy. Notably, RoboTracer can be integrated with various control policies to execute long-horizon, dynamic tasks across diverse robots (UR5, G1 humanoid) in cluttered real-world scenes.
>
---
#### [new 020] Audio-Based Tactile Human-Robot Interaction Recognition
- **分类: cs.RO**

- **简介: 该论文属人机交互识别任务，旨在解决机器人触觉感知依赖昂贵力传感器的问题。提出用机身麦克风采集触摸声音，结合CNN分类6种触觉交互（如敲击、摩擦等），在336样本上验证了基于声学频谱特征的有效性。**

- **链接: [https://arxiv.org/pdf/2512.11873v1](https://arxiv.org/pdf/2512.11873v1)**

> **作者:** Antonia Yepes; Marie Charbonneau
>
> **备注:** 1 page, 1 figure, 1 table
>
> **摘要:** This study explores the use of microphones placed on a robot's body to detect tactile interactions via sounds produced when the hard shell of the robot is touched. This approach is proposed as an alternative to traditional methods using joint torque sensors or 6-axis force/torque sensors. Two Adafruit I2S MEMS microphones integrated with a Raspberry Pi 4 were positioned on the torso of a Pollen Robotics Reachy robot to capture audio signals from various touch types on the robot arms (tapping, knocking, rubbing, stroking, scratching, and pressing). A convolutional neural network was trained for touch classification on a dataset of 336 pre-processed samples (48 samples per touch type). The model shows high classification accuracy between touch types with distinct acoustic dominant frequencies.
>
---
#### [new 021] Robust Underwater Localization of Buoyancy Driven microFloats Using Acoustic Time-of-Flight Measurements
- **分类: cs.RO**

- **简介: 该论文针对低成本水下浮标精确定位难的问题，提出一种鲁棒、低频硬件依赖的声学ToF定位方法：采用双向传输增强测量数，结合非线性三边测量与基于几何代价和CRLB的滤波剔除多径等异常值，显著提升定位精度与轨迹一致性。**

- **链接: [https://arxiv.org/pdf/2512.12233v1](https://arxiv.org/pdf/2512.12233v1)**

> **作者:** Murad Mehrab Abrar; Trevor W. Harrison
>
> **备注:** 9 pages
>
> **摘要:** Accurate underwater localization remains a challenge for inexpensive autonomous platforms that require highfrequency position updates. In this paper, we present a robust, low-cost localization pipeline for buoyancy-driven microFloats operating in coastal waters. We build upon previous work by introducing a bidirectional acoustic Time-of-Flight (ToF) localization framework, which incorporates both float-to-buoy and buoy-to-float transmissions, thereby increasing the number of usable measurements. The method integrates nonlinear trilateration with a filtering of computed position estimates based on geometric cost and Cramer-Rao Lower Bounds (CRLB). This approach removes outliers caused by multipath effects and other acoustic errors from the ToF estimation and improves localization robustness without relying on heavy smoothing. We validate the framework in two field deployments in Puget Sound, Washington, USA. The localization pipeline achieves median positioning errors below 4 m relative to GPS positions. The filtering technique shows a reduction in mean error from 139.29 m to 12.07 m, and improved alignment of trajectories with GPS paths. Additionally, we demonstrate a Time-Difference-of-Arrival (TDoA) localization for unrecovered floats that were transmitting during the experiment. Range-based acoustic localization techniques are widely used and generally agnostic to hardware-this work aims to maximize their utility by improving positioning frequency and robustness through careful algorithmic design.
>
---
#### [new 022] Universal Dexterous Functional Grasping via Demonstration-Editing Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文面向通用灵巧功能抓取任务，解决细粒度功能抓取中目标定义难、多任务学习低效及仿真到真实迁移差的问题。提出DemoFunGrasp框架，将功能抓取解耦为抓取风格与功能可供性，通过单次示范的一步编辑式强化学习实现高效训练，并结合视觉语言模型实现自主指令执行。**

- **链接: [https://arxiv.org/pdf/2512.13380v1](https://arxiv.org/pdf/2512.13380v1)**

> **作者:** Chuan Mao; Haoqi Yuan; Ziye Huang; Chaoyi Xu; Kai Ma; Zongqing Lu
>
> **备注:** 19 pages
>
> **摘要:** Reinforcement learning (RL) has achieved great success in dexterous grasping, significantly improving grasp performance and generalization from simulation to the real world. However, fine-grained functional grasping, which is essential for downstream manipulation tasks, remains underexplored and faces several challenges: the complexity of specifying goals and reward functions for functional grasps across diverse objects, the difficulty of multi-task RL exploration, and the challenge of sim-to-real transfer. In this work, we propose DemoFunGrasp for universal dexterous functional grasping. We factorize functional grasping conditions into two complementary components - grasping style and affordance - and integrate them into an RL framework that can learn to grasp any object with any functional grasping condition. To address the multi-task optimization challenge, we leverage a single grasping demonstration and reformulate the RL problem as one-step demonstration editing, substantially enhancing sample efficiency and performance. Experimental results in both simulation and the real world show that DemoFunGrasp generalizes to unseen combinations of objects, affordances, and grasping styles, outperforming baselines in both success rate and functional grasping accuracy. In addition to strong sim-to-real capability, by incorporating a vision-language model (VLM) for planning, our system achieves autonomous instruction-following grasp execution.
>
---
#### [new 023] Reinforcement Learning based 6-DoF Maneuvers for Microgravity Intravehicular Docking: A Simulation Study with Int-Ball2 in ISS-JEM
- **分类: cs.RO**

- **简介: 该论文属自主导航任务，解决微重力舱内自由飞行器（Int-Ball2）6-DoF精准对接难题。提出基于PPO的强化学习框架，在Isaac Sim高保真JEM环境中训练控制器，建模推进器物理特性并引入域随机化与观测噪声，实现鲁棒 docking。**

- **链接: [https://arxiv.org/pdf/2512.13514v1](https://arxiv.org/pdf/2512.13514v1)**

> **作者:** Aman Arora; Matteo El-Hariry; Miguel Olivares-Mendez
>
> **备注:** Presented at AI4OPA Workshop at the International Conference on Space Robotics (iSpaRo) 2025 at Sendai, Japan
>
> **摘要:** Autonomous free-flyers play a critical role in intravehicular tasks aboard the International Space Station (ISS), where their precise docking under sensing noise, small actuation mismatches, and environmental variability remains a nontrivial challenge. This work presents a reinforcement learning (RL) framework for six-degree-of-freedom (6-DoF) docking of JAXA's Int-Ball2 robot inside a high-fidelity Isaac Sim model of the Japanese Experiment Module (JEM). Using Proximal Policy Optimization (PPO), we train and evaluate controllers under domain-randomized dynamics and bounded observation noise, while explicitly modeling propeller drag-torque effects and polarity structure. This enables a controlled study of how Int-Ball2's propulsion physics influence RL-based docking performance in constrained microgravity interiors. The learned policy achieves stable and reliable docking across varied conditions and lays the groundwork for future extensions pertaining to Int-Ball2 in collision-aware navigation, safe RL, propulsion-accurate sim-to-real transfer, and vision-based end-to-end docking.
>
---
#### [new 024] Safe Learning for Contact-Rich Robot Tasks: A Survey from Classical Learning-Based Methods to Safe Foundation Models
- **分类: cs.RO**

- **简介: 该论文是一篇综述，聚焦机器人接触丰富任务中的安全学习问题。它系统梳理了安全探索与执行方法（如约束强化学习、控制屏障函数等），并分析其在视觉语言动作模型（VLA）等基础模型中的延伸与挑战，旨在推动安全、可靠、可部署的接触式机器人学习。**

- **链接: [https://arxiv.org/pdf/2512.11908v1](https://arxiv.org/pdf/2512.11908v1)**

> **作者:** Heng Zhang; Rui Dai; Gokhan Solak; Pokuang Zhou; Yu She; Arash Ajoudani
>
> **摘要:** Contact-rich tasks pose significant challenges for robotic systems due to inherent uncertainty, complex dynamics, and the high risk of damage during interaction. Recent advances in learning-based control have shown great potential in enabling robots to acquire and generalize complex manipulation skills in such environments, but ensuring safety, both during exploration and execution, remains a critical bottleneck for reliable real-world deployment. This survey provides a comprehensive overview of safe learning-based methods for robot contact-rich tasks. We categorize existing approaches into two main domains: safe exploration and safe execution. We review key techniques, including constrained reinforcement learning, risk-sensitive optimization, uncertainty-aware modeling, control barrier functions, and model predictive safety shields, and highlight how these methods incorporate prior knowledge, task structure, and online adaptation to balance safety and efficiency. A particular emphasis of this survey is on how these safe learning principles extend to and interact with emerging robotic foundation models, especially vision-language models (VLMs) and vision-language-action models (VLAs), which unify perception, language, and control for contact-rich manipulation. We discuss both the new safety opportunities enabled by VLM/VLA-based methods, such as language-level specification of constraints and multimodal grounding of safety signals, and the amplified risks and evaluation challenges they introduce. Finally, we outline current limitations and promising future directions toward deploying reliable, safety-aligned, and foundation-model-enabled robots in complex contact-rich environments. More details and materials are available at our \href{ https://github.com/jack-sherman01/Awesome-Learning4Safe-Contact-rich-tasks}{Project GitHub Repository}.
>
---
#### [new 025] Optimized Conflict Management for Urban Air Mobility Using Swarm UAV Networks
- **分类: cs.RO**

- **简介: 该论文属空中交通管理任务，旨在解决高密度城市空域中无人机冲突实时化解难题。提出基于边缘AI的去中心化蜂群架构，采用轻量神经网络实现分布式冲突检测与 resolution，并通过仿真验证其较传统中心化方法提速3.8倍、精度更高。**

- **链接: [https://arxiv.org/pdf/2512.12632v1](https://arxiv.org/pdf/2512.12632v1)**

> **作者:** Rishit Agnihotri; Sandeep Kumar Sharma
>
> **备注:** Preprint. Under review for conference submission
>
> **摘要:** Urban Air Mobility (UAM) poses unprecedented traffic coordination challenges, especially with increasing UAV densities in dense urban corridors. This paper introduces a mathematical model using a control algorithm to optimize an Edge AI-driven decentralized swarm architecture for intelligent conflict resolution, enabling real-time decision-making with low latency. Using lightweight neural networks, the system leverages edge nodes to perform distributed conflict detection and resolution. A simulation platform was developed to evaluate the scheme under various UAV densities. Results indicate that the conflict resolution time is dramatically minimized up to 3.8 times faster, and accuracy is enhanced compared to traditional centralized control models. The proposed architecture is highly promising for scalable, efficient, and safe aerial traffic management in future UAM systems.
>
---
#### [new 026] Post-Training and Test-Time Scaling of Generative Agent Behavior Models for Interactive Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向交互式自动驾驶行为建模任务，解决模仿学习模型安全性不足、闭合环评估缺失及测试时行为不一致问题。提出GRBO（强化学习后训练提升安全性）和Warm-K（测试时采样策略增强一致性与反应性）两种方法。**

- **链接: [https://arxiv.org/pdf/2512.13262v1](https://arxiv.org/pdf/2512.13262v1)**

> **作者:** Hyunki Seong; Jeong-Kyun Lee; Heesoo Myeong; Yongho Shin; Hyun-Mook Cho; Duck Hoon Kim; Pranav Desai; Monu Surana
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Learning interactive motion behaviors among multiple agents is a core challenge in autonomous driving. While imitation learning models generate realistic trajectories, they often inherit biases from datasets dominated by safe demonstrations, limiting robustness in safety-critical cases. Moreover, most studies rely on open-loop evaluation, overlooking compounding errors in closed-loop execution. We address these limitations with two complementary strategies. First, we propose Group Relative Behavior Optimization (GRBO), a reinforcement learning post-training method that fine-tunes pretrained behavior models via group relative advantage maximization with human regularization. Using only 10% of the training dataset, GRBO improves safety performance by over 40% while preserving behavioral realism. Second, we introduce Warm-K, a warm-started Top-K sampling strategy that balances consistency and diversity in motion selection. Our Warm-K method-based test-time scaling enhances behavioral consistency and reactivity at test time without retraining, mitigating covariate shift and reducing performance discrepancies. Demo videos are available in the supplementary material.
>
---
#### [new 027] A Stochastic Approach to Terrain Maps for Safe Lunar Landing
- **分类: cs.RO**

- **简介: 该论文属地形建模任务，旨在解决月球南极阴影区安全着陆中地形不确定性建模不准的问题。提出两阶段高斯过程方法：用二级GP学习LRO DEM置信度的异方差噪声，指导一级GP建模地形，结合随机变分推断实现可扩展训练，提升地形不确定性估计精度。**

- **链接: [https://arxiv.org/pdf/2512.12058v1](https://arxiv.org/pdf/2512.12058v1)**

> **作者:** Anja Sheppard; Chris Reale; Katherine A. Skinner
>
> **备注:** Accepted to IEEE Aerospace 2026
>
> **摘要:** Safely landing on the lunar surface is a challenging task, especially in the heavily-shadowed South Pole region where traditional vision-based hazard detection methods are not reliable. The potential existence of valuable resources at the lunar South Pole has made landing in that region a high priority for many space agencies and commercial companies. However, relying on a LiDAR for hazard detection during descent is risky, as this technology is fairly untested in the lunar environment. There exists a rich log of lunar surface data from the Lunar Reconnaissance Orbiter (LRO), which could be used to create informative prior maps of the surface before descent. In this work, we propose a method for generating stochastic elevation maps from LRO data using Gaussian processes (GPs), which are a powerful Bayesian framework for non-parametric modeling that produce accompanying uncertainty estimates. In high-risk environments such as autonomous spaceflight, interpretable estimates of terrain uncertainty are critical. However, no previous approaches to stochastic elevation mapping have taken LRO Digital Elevation Model (DEM) confidence maps into account, despite this data containing key information about the quality of the DEM in different areas. To address this gap, we introduce a two-stage GP model in which a secondary GP learns spatially varying noise characteristics from DEM confidence data. This heteroscedastic information is then used to inform the noise parameters for the primary GP, which models the lunar terrain. Additionally, we use stochastic variational GPs to enable scalable training. By leveraging GPs, we are able to more accurately model the impact of heteroscedastic sensor noise on the resulting elevation map. As a result, our method produces more informative terrain uncertainty, which can be used for downstream tasks such as hazard detection and safe landing site selection.
>
---
#### [new 028] HMPCC: Human-Aware Model Predictive Coverage Control
- **分类: cs.RO**

- **简介: 该论文属多机器人协同覆盖任务，旨在解决未知环境中机器人安全覆盖与避让非合作人类的问题。提出人机感知的模型预测覆盖控制（HMPCC）框架：融合人类轨迹预测、GMM环境建模、完全去中心化决策，无需通信即可实现高效自适应覆盖。**

- **链接: [https://arxiv.org/pdf/2512.12717v1](https://arxiv.org/pdf/2512.12717v1)**

> **作者:** Mattia Catellani; Marta Gabbi; Lorenzo Sabattini
>
> **摘要:** We address the problem of coordinating a team of robots to cover an unknown environment while ensuring safe operation and avoiding collisions with non-cooperative agents. Traditional coverage strategies often rely on simplified assumptions, such as known or convex environments and static density functions, and struggle to adapt to real-world scenarios, especially when humans are involved. In this work, we propose a human-aware coverage framework based on Model Predictive Control (MPC), namely HMPCC, where human motion predictions are integrated into the planning process. By anticipating human trajectories within the MPC horizon, robots can proactively coordinate their actions %avoid redundant exploration, and adapt to dynamic conditions. The environment is modeled as a Gaussian Mixture Model (GMM), representing regions of interest. Team members operate in a fully decentralized manner, without relying on explicit communication, an essential feature in hostile or communication-limited scenarios. Our results show that human trajectory forecasting enables more efficient and adaptive coverage, improving coordination between human and robotic agents.
>
---
#### [new 029] Programmable Deformation Design of Porous Soft Actuator through Volumetric-Pattern-Induced Anisotropy
- **分类: cs.RO**

- **简介: 该论文属软体机器人设计任务，旨在解决传统气动执行器结构弱、功能单一、定制成本高的问题。提出在多孔泡沫本体上刻划特定图案以引入各向异性，实现真空驱动下的可编程弯曲、倾斜与扭转变形，并验证其可扩展性与生物启发应用。**

- **链接: [https://arxiv.org/pdf/2512.12320v1](https://arxiv.org/pdf/2512.12320v1)**

> **作者:** Canqi Meng; Weibang Bai
>
> **摘要:** Conventional soft pneumatic actuators, typically based on hollow elastomeric chambers, often suffer from small structural support and require costly geometry-specific redesigns for multimodal functionality. Porous materials such as foam, filled into chambers, can provide structural stability for the actuators. However, methods to achieve programmable deformation by tailoring the porous body itself remain underexplored. In this paper, a novel design method is presented to realize soft porous actuators with programmable deformation by incising specific patterns into the porous foam body. This approach introduces localized structural anisotropy of the foam guiding the material's deformation under a global vacuum input. Furthermore, three fundamental patterns on a cylindrical foam substrate are discussed: transverse for bending, longitudinal for tilting, and diagonal for twisting. A computational model is built with Finite Element Analysis (FEA), to investigate the mechanism of the incision-patterning method. Experiments demonstrate that with a potential optimal design of the pattern array number N, actuators can achieve bending up to $80^{\circ}$ (N=2), tilting of $18^{\circ}$ (N=1), and twisting of $115^{\circ}$ (N=8). The versatility of our approach is demonstrated via pattern transferability, scalability, and mold-less rapid prototyping of complex designs. As a comprehensive application, we translate the human hand crease map into a functional incision pattern, creating a bio-inspired soft robot hand capable of human-like adaptive grasping. Our work provides a new, efficient, and scalable paradigm for the design of multi-functional soft porous robots.
>
---
#### [new 030] Fast Policy Learning for 6-DOF Position Control of Underwater Vehicles
- **分类: cs.RO; cs.LG**

- **简介: 该论文属强化学习控制任务，旨在解决AUV在复杂水下环境中6-DOF位置控制鲁棒性差、训练慢、sim-to-real难的问题。提出基于JAX与MuJoCo-XLA的GPU加速RL训练框架，实现2分钟内训练，并零样本迁移至真实AUV，首次实现实时6-DOF轨迹跟踪与扰动抑制。**

- **链接: [https://arxiv.org/pdf/2512.13359v1](https://arxiv.org/pdf/2512.13359v1)**

> **作者:** Sümer Tunçay; Alain Andres; Ignacio Carlucho
>
> **摘要:** Autonomous Underwater Vehicles (AUVs) require reliable six-degree-of-freedom (6-DOF) position control to operate effectively in complex and dynamic marine environments. Traditional controllers are effective under nominal conditions but exhibit degraded performance when faced with unmodeled dynamics or environmental disturbances. Reinforcement learning (RL) provides a powerful alternative but training is typically slow and sim-to-real transfer remains challenging. This work introduces a GPU-accelerated RL training pipeline built in JAX and MuJoCo-XLA (MJX). By jointly JIT-compiling large-scale parallel physics simulation and learning updates, we achieve training times of under two minutes.Through systematic evaluation of multiple RL algorithms, we show robust 6-DOF trajectory tracking and effective disturbance rejection in real underwater experiments, with policies transferred zero-shot from simulation. Our results provide the first explicit real-world demonstration of RL-based AUV position control across all six degrees of freedom.
>
---
#### [new 031] Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos
- **分类: cs.RO**

- **简介: 该论文属机器人学习中的视觉-语言-动作（VLA）建模任务，旨在解决2D视觉与3D物理空间动作对齐不足的问题。提出空间感知的VLA预训练范式，利用人类视频提取3D视觉与动作监督，构建VIPA-VLA双编码器模型，增强3D空间理解，提升下游机器人策略的鲁棒性与泛化性。**

- **链接: [https://arxiv.org/pdf/2512.13080v1](https://arxiv.org/pdf/2512.13080v1)**

> **作者:** Yicheng Feng; Wanpeng Zhang; Ye Wang; Hao Luo; Haoqi Yuan; Sipeng Zheng; Zongqing Lu
>
> **摘要:** Vision-Language-Action (VLA) models provide a promising paradigm for robot learning by integrating visual perception with language-guided policy learning. However, most existing approaches rely on 2D visual inputs to perform actions in 3D physical environments, creating a significant gap between perception and action grounding. To bridge this gap, we propose a Spatial-Aware VLA Pretraining paradigm that performs explicit alignment between visual space and physical space during pretraining, enabling models to acquire 3D spatial understanding before robot policy learning. Starting from pretrained vision-language models, we leverage large-scale human demonstration videos to extract 3D visual and 3D action annotations, forming a new source of supervision that aligns 2D visual observations with 3D spatial reasoning. We instantiate this paradigm with VIPA-VLA, a dual-encoder architecture that incorporates a 3D visual encoder to augment semantic visual representations with 3D-aware features. When adapted to downstream robot tasks, VIPA-VLA achieves significantly improved grounding between 2D vision and 3D action, resulting in more robust and generalizable robotic policies.
>
---
#### [new 032] Measuring What Matters: Scenario-Driven Evaluation for Trajectory Predictors in Autonomous Driving
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属自动驾驶轨迹预测评估任务，旨在解决传统误差指标（如ADE/FDE）忽视预测对车辆决策影响的问题。提出场景驱动的双维度评估框架，动态融合准确性与多样性，更合理反映预测器对自动驾驶性能的实际贡献。**

- **链接: [https://arxiv.org/pdf/2512.12211v1](https://arxiv.org/pdf/2512.12211v1)**

> **作者:** Longchao Da; David Isele; Hua Wei; Manish Saroya
>
> **备注:** 9 Pages, 8 Figures
>
> **摘要:** Being able to anticipate the motion of surrounding agents is essential for the safe operation of autonomous driving systems in dynamic situations. While various methods have been proposed for trajectory prediction, the current evaluation practices still rely on error-based metrics (e.g., ADE, FDE), which reveal the accuracy from a post-hoc view but ignore the actual effect the predictor brings to the self-driving vehicles (SDVs), especially in complex interactive scenarios: a high-quality predictor not only chases accuracy, but should also captures all possible directions a neighbor agent might move, to support the SDVs' cautious decision-making. Given that the existing metrics hardly account for this standard, in our work, we propose a comprehensive pipeline that adaptively evaluates the predictor's performance by two dimensions: accuracy and diversity. Based on the criticality of the driving scenario, these two dimensions are dynamically combined and result in a final score for the predictor's performance. Extensive experiments on a closed-loop benchmark using real-world datasets show that our pipeline yields a more reasonable evaluation than traditional metrics by better reflecting the correlation of the predictors' evaluation with the autonomous vehicles' driving performance. This evaluation pipeline shows a robust way to select a predictor that potentially contributes most to the SDV's driving performance.
>
---
#### [new 033] VLG-Loc: Vision-Language Global Localization from Labeled Footprint Maps
- **分类: cs.RO**

- **简介: 该论文提出VLG-Loc，属机器人全局定位任务，旨在解决仅用带名称与面积的简略足迹地图（无几何/外观细节）实现鲁棒定位的难题。方法利用视觉语言模型匹配图像与地图地标，并在蒙特卡洛框架中评估位姿似然，融合视觉与扫描定位提升性能。**

- **链接: [https://arxiv.org/pdf/2512.12793v1](https://arxiv.org/pdf/2512.12793v1)**

> **作者:** Mizuho Aoki; Kohei Honda; Yasuhiro Yoshimura; Takeshi Ishita; Ryo Yonetani
>
> **摘要:** This paper presents Vision-Language Global Localization (VLG-Loc), a novel global localization method that uses human-readable labeled footprint maps containing only names and areas of distinctive visual landmarks in an environment. While humans naturally localize themselves using such maps, translating this capability to robotic systems remains highly challenging due to the difficulty of establishing correspondences between observed landmarks and those in the map without geometric and appearance details. To address this challenge, VLG-Loc leverages a vision-language model (VLM) to search the robot's multi-directional image observations for the landmarks noted in the map. The method then identifies robot poses within a Monte Carlo localization framework, where the found landmarks are used to evaluate the likelihood of each pose hypothesis. Experimental validation in simulated and real-world retail environments demonstrates superior robustness compared to existing scan-based methods, particularly under environmental changes. Further improvements are achieved through the probabilistic fusion of visual and scan-based localization.
>
---
#### [new 034] Near-Field Perception for Safety Enhancement of Autonomous Mobile Robots in Manufacturing Environments
- **分类: cs.RO**

- **简介: 该论文属机器人感知任务，旨在解决AMR在制造环境中近场小障碍物检测难的问题。提出三层近场感知框架：激光断点检测（二值感知）、激光位移测量（高度估计）、嵌入式AI视觉分类（语义理解），均部署于树莓派5实现实时运行。**

- **链接: [https://arxiv.org/pdf/2512.13561v1](https://arxiv.org/pdf/2512.13561v1)**

> **作者:** Li-Wei Shih; Ruo-Syuan Mei; Jesse Heidrich; Hui-Ping Wang; Joel Hooton; Joshua Solomon; Jorge Arinez; Guangze Li; Chenhui Shao
>
> **备注:** Submitted to the 54th SME North American Manufacturing Research Conference (NAMRC 54)
>
> **摘要:** Near-field perception is essential for the safe operation of autonomous mobile robots (AMRs) in manufacturing environments. Conventional ranging sensors such as light detection and ranging (LiDAR) and ultrasonic devices provide broad situational awareness but often fail to detect small objects near the robot base. To address this limitation, this paper presents a three-tier near-field perception framework. The first approach employs light-discontinuity detection, which projects a laser stripe across the near-field zone and identifies interruptions in the stripe to perform fast, binary cutoff sensing for obstacle presence. The second approach utilizes light-displacement measurement to estimate object height by analyzing the geometric displacement of a projected stripe in the camera image, which provides quantitative obstacle height information with minimal computational overhead. The third approach employs a computer vision-based object detection model on embedded AI hardware to classify objects, enabling semantic perception and context-aware safety decisions. All methods are implemented on a Raspberry Pi 5 system, achieving real-time performance at 25 or 50 frames per second. Experimental evaluation and comparative analysis demonstrate that the proposed hierarchy balances precision, computation, and cost, thereby providing a scalable perception solution for enabling safe operations of AMRs in manufacturing environments.
>
---
#### [new 035] Semantic Zone based 3D Map Management for Mobile Robot
- **分类: cs.RO**

- **简介: 该论文属机器人SLAM地图管理任务，解决大场景3D地图内存占用高、检索低效问题。提出语义区域划分方法，以功能空间单元（如走廊、大厅）为基本管理单位，动态调度工作/长期内存，实现内存可控、高效检索与稳定导航。**

- **链接: [https://arxiv.org/pdf/2512.12228v1](https://arxiv.org/pdf/2512.12228v1)**

> **作者:** Huichang Yun; Seungho Yoo
>
> **备注:** 12 pages, 11 figures
>
> **摘要:** Mobile robots in large-scale indoor environments, such as hospitals and logistics centers, require accurate 3D spatial representations. However, 3D maps consume substantial memory, making it difficult to maintain complete map data within limited computational resources. Existing SLAM frameworks typically rely on geometric distance or temporal metrics for memory management, often resulting in inefficient data retrieval in spatially compartmentalized environments. To address this, we propose a semantic zone-based 3D map management method that shifts the paradigm from geometry-centric to semantics-centric control. Our approach partitions the environment into meaningful spatial units (e.g., lobbies, hallways) and designates these zones as the primary unit for memory management. By dynamically loading only task-relevant zones into Working Memory (WM) and offloading inactive zones to Long-Term Memory (LTM), the system strictly enforces user-defined memory thresholds. Implemented within the RTAB-Map framework, our method demonstrates substantial reductions in unnecessary signature load/unload cycles and cumulative memory utilization compared to standard approaches. The results confirm that semantic zone-based management ensures stable, predictable memory usage while preserving map availability for navigation. Code is available at: https://github.com/huichangs/rtabmap/tree/segment
>
---
#### [new 036] Multi-directional Safe Rectangle Corridor-Based MPC for Nonholonomic Robots Navigation in Cluttered Environment
- **分类: cs.RO**

- **简介: 该论文属机器人导航任务，旨在解决非完整移动机器人在密集杂乱环境中的实时避障与路径规划问题。提出改进序贯MPC框架，含多向安全矩形走廊（MDSRC）建模自由空间，及融合屏障函数的走廊约束MPC，实现高效静态/动态避障与直接速度输出。**

- **链接: [https://arxiv.org/pdf/2512.13215v1](https://arxiv.org/pdf/2512.13215v1)**

> **作者:** Yinsong Qu; Yunxiang Li; Shanlin Zhong
>
> **备注:** 9 pages, 11 figures, conference paper for the 2025 International Conference on Advanced Robotics and Mechatronics (ICARM), accepted
>
> **摘要:** Autonomous Mobile Robots (AMRs) have become indispensable in industrial applications due to their operational flexibility and efficiency. Navigation serves as a crucial technical foundation for accomplishing complex tasks. However, navigating AMRs in dense, cluttered, and semi-structured environments remains challenging, primarily due to nonholonomic vehicle dynamics, interactions with mixed static/dynamic obstacles, and the non-convex constrained nature of such operational spaces. To solve these problems, this paper proposes an Improved Sequential Model Predictive Control (ISMPC) navigation framework that systematically reformulates navigation tasks as sequential switched optimal control problems. The framework addresses the aforementioned challenges through two key innovations: 1) Implementation of a Multi-Directional Safety Rectangular Corridor (MDSRC) algorithm, which encodes the free space through rectangular convex regions to avoid collision with static obstacles, eliminating redundant computational burdens and accelerating solver convergence; 2) A sequential MPC navigation framework that integrates corridor constraints with barrier function constraints is proposed to achieve static and dynamic obstacle avoidance. The ISMPC navigation framework enables direct velocity generation for AMRs, simplifying traditional navigation algorithm architectures. Comparative experiments demonstrate the framework's superiority in free-space utilization ( an increase of 41.05$\%$ in the average corridor area) while maintaining real-time computational performance (average corridors generation latency of 3 ms).
>
---
#### [new 037] PvP: Data-Efficient Humanoid Robot Learning with Proprioceptive-Privileged Contrastive Representations
- **分类: cs.RO; cs.LG**

- **简介: 该论文属机器人学习任务，旨在解决强化学习在人形机器人全身控制中样本效率低的问题。提出PvP对比学习框架，利用本体感知与特权状态互补性学习紧凑表征，并构建SRL4Humanoid评估框架。实验验证其显著提升样本效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.13093v1](https://arxiv.org/pdf/2512.13093v1)**

> **作者:** Mingqi Yuan; Tao Yu; Haolin Song; Bo Li; Xin Jin; Hua Chen; Wenjun Zeng
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Achieving efficient and robust whole-body control (WBC) is essential for enabling humanoid robots to perform complex tasks in dynamic environments. Despite the success of reinforcement learning (RL) in this domain, its sample inefficiency remains a significant challenge due to the intricate dynamics and partial observability of humanoid robots. To address this limitation, we propose PvP, a Proprioceptive-Privileged contrastive learning framework that leverages the intrinsic complementarity between proprioceptive and privileged states. PvP learns compact and task-relevant latent representations without requiring hand-crafted data augmentations, enabling faster and more stable policy learning. To support systematic evaluation, we develop SRL4Humanoid, the first unified and modular framework that provides high-quality implementations of representative state representation learning (SRL) methods for humanoid robot learning. Extensive experiments on the LimX Oli robot across velocity tracking and motion imitation tasks demonstrate that PvP significantly improves sample efficiency and final performance compared to baseline SRL methods. Our study further provides practical insights into integrating SRL with RL for humanoid WBC, offering valuable guidance for data-efficient humanoid robot learning.
>
---
#### [new 038] SAGA: Open-World Mobile Manipulation via Structured Affordance Grounding
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出SAGA框架，面向开放世界移动操作任务，解决跨环境、跨任务及多模态指令（语言/点选/示例）下的泛化控制问题。通过结构化可供性表征与3D视觉接地生成热图，解耦语义意图与运动控制，实现零样本执行与少样本适应。**

- **链接: [https://arxiv.org/pdf/2512.12842v1](https://arxiv.org/pdf/2512.12842v1)**

> **作者:** Kuan Fang; Yuxin Chen; Xinghao Zhu; Farzad Niroui; Lingfeng Sun; Jiuguang Wang
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** We present SAGA, a versatile and adaptive framework for visuomotor control that can generalize across various environments, task objectives, and user specifications. To efficiently learn such capability, our key idea is to disentangle high-level semantic intent from low-level visuomotor control by explicitly grounding task objectives in the observed environment. Using an affordance-based task representation, we express diverse and complex behaviors in a unified, structured form. By leveraging multimodal foundation models, SAGA grounds the proposed task representation to the robot's visual observation as 3D affordance heatmaps, highlighting task-relevant entities while abstracting away spurious appearance variations that would hinder generalization. These grounded affordances enable us to effectively train a conditional policy on multi-task demonstration data for whole-body control. In a unified framework, SAGA can solve tasks specified in different forms, including language instructions, selected points, and example demonstrations, enabling both zero-shot execution and few-shot adaptation. We instantiate SAGA on a quadrupedal manipulator and conduct extensive experiments across eleven real-world tasks. SAGA consistently outperforms end-to-end and modular baselines by substantial margins. Together, these results demonstrate that structured affordance grounding offers a scalable and effective pathway toward generalist mobile manipulation.
>
---
#### [new 039] Differentiable Material Point Method for the Control of Deformable Objects
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于可微物理仿真与控制任务，旨在解决柔性物体（如超弹性绳）变形控制难的问题。作者构建了可微分的物质点法（MPM）模拟器，并用于优化主动阻尼控制轨迹，在动能最小化上比基线MPPI更快、更低能耗且更高效。**

- **链接: [https://arxiv.org/pdf/2512.13214v1](https://arxiv.org/pdf/2512.13214v1)**

> **作者:** Diego Bolliger; Gabriele Fadini; Markus Bambach; Alisa Rupenyan
>
> **备注:** 7 Pages, 4 Figures, 1 Table
>
> **摘要:** Controlling the deformation of flexible objects is challenging due to their non-linear dynamics and high-dimensional configuration space. This work presents a differentiable Material Point Method (MPM) simulator targeted at control applications. We exploit the differentiability of the simulator to optimize a control trajectory in an active damping problem for a hyperelastic rope. The simulator effectively minimizes the kinetic energy of the rope around 2$\times$ faster than a baseline MPPI method and to a 20% lower energy level, while using about 3% of the computation time.
>
---
#### [new 040] Autonomously Unweaving Multiple Cables Using Visual Feedback
- **分类: cs.RO**

- **简介: 该论文研究多电缆自动解缠任务，旨在分离相互交织的多根柔性电缆。提出基于视觉反馈的图结构状态表征与状态转移模型，将解缠建模为带几何-拓扑约束的拾取放置问题，通过迭代感知-规划-执行实现84%平均成功率。**

- **链接: [https://arxiv.org/pdf/2512.12468v1](https://arxiv.org/pdf/2512.12468v1)**

> **作者:** Tina Tian; Xinyu Wang; Andrew L. Orekhov; Fujun Ruan; Lu Li; Oliver Kroemer; Howie Choset
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** Many cable management tasks involve separating out the different cables and removing tangles. Automating this task is challenging because cables are deformable and can have combinations of knots and multiple interwoven segments. Prior works have focused on untying knots in one cable, which is one subtask of cable management. However, in this paper, we focus on a different subtask called multi-cable unweaving, which refers to removing the intersections among multiple interwoven cables to separate them and facilitate further manipulation. We propose a method that utilizes visual feedback to unweave a bundle of loosely entangled cables. We formulate cable unweaving as a pick-and-place problem, where the grasp position is selected from discrete nodes in a graph-based cable state representation. Our cable state representation encodes both topological and geometric information about the cables from the visual image. To predict future cable states and identify valid actions, we present a novel state transition model that takes into account the straightening and bending of cables during manipulation. Using this state transition model, we select between two high-level action primitives and calculate predicted immediate costs to optimize the lower-level actions. We experimentally demonstrate that iterating the above perception-planning-action process enables unweaving electric cables and shoelaces with an 84% success rate on average.
>
---
#### [new 041] Traversability Aware Autonomous Navigation for Multi-Modal Mobility Morphobot (M4)
- **分类: cs.RO; eess.SY**

- **简介: 该论文面向非结构化环境下的自主导航任务，解决机器人实时地形可通行性评估与安全高效路径规划问题。提出基于LiDAR的 traversability-aware 导航框架：用FAST-LIO定位、生成2.5D高程图，CNN估计可通行性得分，定制A*融合地形成本、距离与能耗进行路径规划。**

- **链接: [https://arxiv.org/pdf/2512.11876v1](https://arxiv.org/pdf/2512.11876v1)**

> **作者:** Hrigved Mahesh Suryawanshi
>
> **摘要:** Autonomous navigation in unstructured environments requires robots to assess terrain difficulty in real-time and plan paths that balance efficiency with safety. This thesis presents a traversability-aware navigation framework for the M4 robot platform that uses learned terrain analy- sis to generate energy-efficient paths avoiding difficult terrain.Our approach uses FAST-LIO for real- time localization, generating 2.5D elevation maps from LiDAR point clouds. A CNN-based model processes these elevation maps to estimate traversability scores, which are converted into navigation costs for path planning. A custom A* planner incorporates these costs alongside geometric distance and energy consumption to find paths that trade modest distance increases for substantial terrain quality improvements. Before system development, a platform-agnostic study compared LiDAR- based and camera-based SLAM using OptiTrack ground truth. Point cloud comparison through ICP alignment and cloud-to-mesh distance analysis demonstrated that LiDAR-based mapping achieves centimeter-level precision essential for elevation mapping, while camera-based approaches exhib- ited significantly higher geometric error. These findings directly resulted in the selection of LiDAR as the primary sensor to generate elevation maps. The complete pipeline integrates FAST-LIO localization, GPU-accelerated elevation mapping, CNN-based traversability estimation, and Nav2 navigation with a custom traversability-aware planner. Experimental results demonstrate that the system successfully avoids low traversability regions and accepts a few longer paths to achieve a reduction in terrain cost. This work establishes a foundation for intelligent terrain-aware navigation applicable to multi-modal robotic platforms.
>
---
#### [new 042] Benchmarking Tesla's Traffic Light and Stop Sign Control: Field Dataset and Behavior Insights
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文属ADAS行为分析任务，旨在实证研究特斯拉TLSSC系统与交通控制设施（红绿灯、停车标志）的交互。作者构建了实地多场景数据集，提出行为分类体系，并标定FVDM模型量化停止、加速及跟车行为特性，揭示关键阈值与动态规律。**

- **链接: [https://arxiv.org/pdf/2512.11802v1](https://arxiv.org/pdf/2512.11802v1)**

> **作者:** Zheng Li; Peng Zhang; Shixiao Liang; Hang Zhou; Chengyuan Ma; Handong Yao; Qianwen Li; Xiaopeng Li
>
> **摘要:** Understanding how Advanced Driver-Assistance Systems (ADAS) interact with Traffic Control Devices (TCDs) is critical for assessing their influence on traffic operations, yet this interaction has received little focused empirical study. This paper presents a field dataset and behavioral analysis of Tesla's Traffic Light and Stop Sign Control (TLSSC), a mature ADAS that perceives traffic lights and stop signs. We design and execute experiments across varied speed limits and TCD types, collecting synchronized high-resolution vehicle trajectory data and driver-perspective video. From these data, we develop a taxonomy of TLSSC-TCD interaction behaviors (i.e., stopping, accelerating, and car following) and calibrate the Full Velocity Difference Model (FVDM) to quantitatively characterize each behavior mode. A novel empirical insight is the identification of a car-following threshold (~90 m). Calibration results reveal that stopping behavior is driven by strong responsiveness to both desired speed deviation and relative speed, whereas accelerating behavior is more conservative. Intersection car-following behavior exhibits smoother dynamics and tighter headways compared to standard car-following behaviors. The established dataset, behavior definitions, and model characterizations together provide a foundation for future simulation, safety evaluation, and design of ADAS-TCD interaction logic. Our dataset is available at GitHub.
>
---
#### [new 043] A Unified Framework for Automated Assembly Sequence and Production Line Planning using Graph-based Optimization
- **分类: cs.RO; cs.MS**

- **简介: 该论文提出PyCAALP框架，面向自动化装配序列与产线规划任务，解决复杂装配中序列可行性、碰撞规避与产线平衡难题；通过图建模、几何约束集成、多属性评估及MIP优化，实现ASP与PLP联合求解，并开源支持定制与协作。**

- **链接: [https://arxiv.org/pdf/2512.13219v1](https://arxiv.org/pdf/2512.13219v1)**

> **作者:** Christoph Hartmann; Marios Demetriades; Kevin Prüfer; Zichen Zhang; Klaus Spindler; Stefan Weltge
>
> **备注:** Code available at https://github.com/TUM-utg/PyCAALP (repository will be made public prior to publication)
>
> **摘要:** This paper presents PyCAALP (Python-based Computer-Aided Assembly Line Planning), a framework for automated Assembly Sequence Planning (ASP) and Production Line Planning (PLP), employing a graph-based approach to model components and joints within production modules. The framework integrates kinematic boundary conditions, such as potential part collisions, to guarantee the feasibility of automated assembly planning. The developed algorithm computes all feasible production sequences, integrating modules for detecting spatial relationships and formulating geometric constraints. The algorithm incorporates additional attributes, including handling feasibility, tolerance matching, and joint compatibility, to manage the high combinatorial complexity inherent in assembly sequence generation. Heuristics, such as Single-Piece Flow assembly and geometrical constraint enforcement, are utilized to further refine the solution space, facilitating more efficient planning for complex assemblies. The PLP stage is formulated as a Mixed-Integer Program (MIP), balancing the total times of a fixed number of manufacturing stations. While some complexity reduction techniques may sacrifice optimality, they significantly reduce the MIPs computational time. Furthermore, the framework enables customization of engineering constraints and supports a flexible trade-off between ASP and PLP. The open-source nature of the framework, available at https://github.com/TUM-utg/PyCAALP, promotes further collaboration and adoption in both industrial and production research applications.
>
---
#### [new 044] Efficient Generation of Smooth Paths with Curvature Guarantees by Mollification
- **分类: cs.RO; eess.SY**

- **简介: 该论文属机器人路径规划任务，旨在解决非光滑（如分段线性）路径无法直接用于微分驱动机器人跟踪的问题。提出基于磨光（mollification）的高效平滑方法，将任意连续路径逼近为高阶可微函数，并显式控制曲率上界，兼顾精度、实时性与算法兼容性。**

- **链接: [https://arxiv.org/pdf/2512.13183v1](https://arxiv.org/pdf/2512.13183v1)**

> **作者:** Alfredo González-Calvin; Juan F. Jiménez; Héctor García de Marina
>
> **摘要:** Most path following and trajectory tracking algorithms in mobile robotics require the desired path or trajectory to be defined by at least twice continuously differentiable functions to guarantee key properties such as global convergence, especially for nonholonomic robots like unicycles with speed constraints. Consequently, these algorithms typically exclude continuous but non-differentiable paths, such as piecewise functions. Despite this exclusion, such paths provide convenient high-level inputs for describing robot missions or behavior. While techniques such as spline interpolation or optimization-based methods are commonly used to smooth non-differentiable paths or create feasible ones from sequences of waypoints, they either can produce unnecessarily complex trajectories or are computationally expensive. In this work, we present a method to regularize non-differentiable functions and generate feasible paths through mollification. Specifically, we approximate an arbitrary path with a differentiable function that can converge to it with arbitrary precision. Additionally, we provide a systematic method for bounding the curvature of generated paths, which we demonstrate by applying it to paths resulting from linking a sequence of waypoints with segments. The proposed approach is computationally efficient, enabling real-time implementation on microcontrollers and compatibility with standard trajectory tracking and path following algorithms.
>
---
#### [new 045] B-ActiveSEAL: Scalable Uncertainty-Aware Active Exploration with Tightly Coupled Localization-Mapping
- **分类: cs.RO**

- **简介: 该论文属机器人主动探索任务，旨在解决大规模环境中定位-建图耦合不确定性导致的计算不可扩展问题。提出B-ActiveSEAL框架，通过行为熵度量，自适应平衡探索与定位不确定性，支持广义熵优化，实现可扩展、不确定性感知的主动探索。**

- **链接: [https://arxiv.org/pdf/2512.12194v1](https://arxiv.org/pdf/2512.12194v1)**

> **作者:** Min-Won Seo; Aamodh Suresh; Carlos Nieto-Granda; Solmaz S. Kia
>
> **备注:** 18 pages, 17 figures
>
> **摘要:** Active robot exploration requires decision-making processes that integrate localization and mapping under tightly coupled uncertainty. However, managing these interdependent uncertainties over long-term operations in large-scale environments rapidly becomes computationally intractable. To address this challenge, we propose B-ActiveSEAL, a scalable information-theoretic active exploration framework that explicitly accounts for coupled uncertainties-from perception through mapping-into the decision-making process. Our framework (i) adaptively balances map uncertainty (exploration) and localization uncertainty (exploitation), (ii) accommodates a broad class of generalized entropy measures, enabling flexible and uncertainty-aware active exploration, and (iii) establishes Behavioral entropy (BE) as an effective information measure for active exploration by enabling intuitive and adaptive decision-making under coupled uncertainties. We establish a theoretical foundation for propagating coupled uncertainties and integrating them into general entropy formulations, enabling uncertainty-aware active exploration under tightly coupled localization-mapping. The effectiveness of the proposed approach is validated through rigorous theoretical analysis and extensive experiments on open-source maps and ROS-Unity simulations across diverse and complex environments. The results demonstrate that B-ActiveSEAL achieves a well-balanced exploration-exploitation trade-off and produces diverse, adaptive exploration behaviors across environments, highlighting clear advantages over representative baselines.
>
---
#### [new 046] INDOOR-LiDAR: Bridging Simulation and Reality for Robot-Centric 360 degree Indoor LiDAR Perception -- A Robot-Centric Hybrid Dataset
- **分类: cs.RO**

- **简介: 该论文提出INDOOR-LiDAR混合数据集，面向机器人中心的室内360°LiDAR感知任务。旨在解决现有室内LiDAR数据规模小、标注不一致、人为干扰多等问题。工作包括构建仿真与真实机器人采集的配对点云数据，统一标注格式，支持检测、BEV、SLAM等应用。**

- **链接: [https://arxiv.org/pdf/2512.12377v1](https://arxiv.org/pdf/2512.12377v1)**

> **作者:** Haichuan Li; Changda Tian; Panos Trahanias; Tomi Westerlund
>
> **摘要:** We present INDOOR-LIDAR, a comprehensive hybrid dataset of indoor 3D LiDAR point clouds designed to advance research in robot perception. Existing indoor LiDAR datasets often suffer from limited scale, inconsistent annotation formats, and human-induced variability during data collection. INDOOR-LIDAR addresses these limitations by integrating simulated environments with real-world scans acquired using autonomous ground robots, providing consistent coverage and realistic sensor behavior under controlled variations. Each sample consists of dense point cloud data enriched with intensity measurements and KITTI-style annotations. The annotation schema encompasses common indoor object categories within various scenes. The simulated subset enables flexible configuration of layouts, point densities, and occlusions, while the real-world subset captures authentic sensor noise, clutter, and domain-specific artifacts characteristic of real indoor settings. INDOOR-LIDAR supports a wide range of applications including 3D object detection, bird's-eye-view (BEV) perception, SLAM, semantic scene understanding, and domain adaptation between simulated and real indoor domains. By bridging the gap between synthetic and real-world data, INDOOR-LIDAR establishes a scalable, realistic, and reproducible benchmark for advancing robotic perception in complex indoor environments.
>
---
#### [new 047] Unifying Quadrotor Motion Planning and Control by Chaining Different Fidelity Models
- **分类: cs.RO; eess.SY**

- **简介: 该论文属无人机运动规划与控制任务，解决长时域规划与高精度控制难以兼顾的问题。提出Unique统一MPC框架，级联高低保真模型：短时高保真控制造型，长时低保真规划；引入状态对齐、约束迁移与渐进平滑障碍处理，并用并行随机初始化提升优化质量。**

- **链接: [https://arxiv.org/pdf/2512.12427v1](https://arxiv.org/pdf/2512.12427v1)**

> **作者:** Rudolf Reiter; Chao Qin; Leonard Bauersfeld; Davide Scaramuzza
>
> **摘要:** Many aerial tasks involving quadrotors demand both instant reactivity and long-horizon planning. High-fidelity models enable accurate control but are too slow for long horizons; low-fidelity planners scale but degrade closed-loop performance. We present Unique, a unified MPC that cascades models of different fidelity within a single optimization: a short-horizon, high-fidelity model for accurate control, and a long-horizon, low-fidelity model for planning. We align costs across horizons, derive feasibility-preserving thrust and body-rate constraints for the point-mass model, and introduce transition constraints that match the different states, thrust-induced acceleration, and jerk-body-rate relations. To prevent local minima emerging from nonsmooth clutter, we propose a 3D progressive smoothing schedule that morphs norm-based obstacles along the horizon. In addition, we deploy parallel randomly initialized MPC solvers to discover lower-cost local minima on the long, low-fidelity horizon. In simulation and real flights, under equal computational budgets, Unique improves closed-loop position or velocity tracking by up to 75% compared with standard MPC and hierarchical planner-tracker baselines. Ablations and Pareto analyses confirm robust gains across horizon variations, constraint approximations, and smoothing schedules.
>
---
#### [new 048] Control of a Twin Rotor using Twin Delayed Deep Deterministic Policy Gradient (TD3)
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属控制任务，旨在解决TRAS系统在非线性、强耦合下的稳定控制与轨迹跟踪难题。提出基于TD3强化学习的无模型控制器，通过仿真验证性能，对比PID抗风扰能力，并在实物平台实验验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.13356v1](https://arxiv.org/pdf/2512.13356v1)**

> **作者:** Zeyad Gamal; Youssef Mahran; Ayman El-Badawy
>
> **备注:** This is the Author Accepted Manuscript version of a paper accepted for publication. The final published version is available via IEEE Xplore
>
> **摘要:** This paper proposes a reinforcement learning (RL) framework for controlling and stabilizing the Twin Rotor Aerodynamic System (TRAS) at specific pitch and azimuth angles and tracking a given trajectory. The complex dynamics and non-linear characteristics of the TRAS make it challenging to control using traditional control algorithms. However, recent developments in RL have attracted interest due to their potential applications in the control of multirotors. The Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm was used in this paper to train the RL agent. This algorithm is used for environments with continuous state and action spaces, similar to the TRAS, as it does not require a model of the system. The simulation results illustrated the effectiveness of the RL control method. Next, external disturbances in the form of wind disturbances were used to test the controller's effectiveness compared to conventional PID controllers. Lastly, experiments on a laboratory setup were carried out to confirm the controller's effectiveness in real-world applications.
>
---
#### [new 049] OXE-AugE: A Large-Scale Robot Augmentation of OXE for Scaling Cross-Embodiment Policy Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属机器人跨形态策略学习任务，旨在解决OXE数据集机器人类型严重失衡、泛化能力弱的问题。作者提出AugE-Toolkit增强 pipeline，构建OXE-AugE数据集，新增9种机器人形态、超440万轨迹，显著提升策略在未见机器人及分布偏移下的性能。**

- **链接: [https://arxiv.org/pdf/2512.13100v1](https://arxiv.org/pdf/2512.13100v1)**

> **作者:** Guanhua Ji; Harsha Polavaram; Lawrence Yunliang Chen; Sandeep Bajamahal; Zehan Ma; Simeon Adebola; Chenfeng Xu; Ken Goldberg
>
> **摘要:** Large and diverse datasets are needed for training generalist robot policies that have potential to control a variety of robot embodiments -- robot arm and gripper combinations -- across diverse tasks and environments. As re-collecting demonstrations and retraining for each new hardware platform are prohibitively costly, we show that existing robot data can be augmented for transfer and generalization. The Open X-Embodiment (OXE) dataset, which aggregates demonstrations from over 60 robot datasets, has been widely used as the foundation for training generalist policies. However, it is highly imbalanced: the top four robot types account for over 85\% of its real data, which risks overfitting to robot--scene combinations. We present AugE-Toolkit, a scalable robot augmentation pipeline, and OXE-AugE, a high-quality open-source dataset that augments OXE with 9 different robot embodiments. OXE-AugE provides over 4.4 million trajectories, more than triple the size of the original OXE. We conduct a systematic study of how scaling robot augmentation impacts cross-embodiment learning. Results suggest that augmenting datasets with diverse arms and grippers improves policy performance not only on the augmented robots, but also on unseen robots and even the original robots under distribution shifts. In physical experiments, we demonstrate that state-of-the-art generalist policies such as OpenVLA and $π_0$ benefit from fine-tuning on OXE-AugE, improving success rates by 24-45% on previously unseen robot--gripper combinations across four real-world manipulation tasks. Project website: https://OXE-AugE.github.io/.
>
---
#### [new 050] ALBATROSS: A robotised system for high-throughput electrolyte screening via automated electrolyte formulation, coin-cell fabrication, and electrochemical evaluation
- **分类: cs.RO**

- **简介: 该论文提出ALBATROSS系统，属电池材料高通量筛选任务，旨在解决人工制备/测试纽扣电池耗时费力、难以规模化的问题。作者研发了集成于氩气手套箱的自动化系统，实现电解液配制、电池组装与电化学表征全流程无人化，显著提升数据质量与效率。**

- **链接: [https://arxiv.org/pdf/2512.13198v1](https://arxiv.org/pdf/2512.13198v1)**

> **作者:** Hyun-Gi Lee; Jaekyeong Han; Minjun Kwon; Hyeonuk Kwon; Jooha Park; Hoe Jin Ha; Dong-Hwa Seo
>
> **摘要:** As battery technologies advance toward higher stability and energy density, the need for extensive cell-level testing across various component configurations becomes critical. To evaluate performance and understand the operating principles of batteries in laboratory scale, fabrication and evaluation of coin cells are essential processes. However, the conventional coin-cell assembly and testing processes require significant time and labor from researchers, posing challenges to high-throughput screening research. In this study, we introduce an Automated Li-ion BAttery Testing RObot SyStem (ALBATROSS), an automated system capable of electrolyte formulation, coin-cell assembly, and electrochemical evaluation. The system, integrated within a argon-filled glovebox, enables fully automated assembly and testing of up to 48 cells without researcher intervention. By incorporating custom-designed robot gripper and 3D-printed structures optimized for precise cell handling, ALBATROSS achieved high assembly reliability, yielding a relative standard deviation (RSD) of less than 1.2% in discharge capacity and a standard deviation of less than 3 Ω in EIS measurements for NCM811||Li half cells. Owing to its high reliability and automation capability, ALBATROSS allows for the acquisition of high-quality coin-cell datasets, which are expected to accelerate the development of next-generation electrolytes.
>
---
#### [new 051] Learning to Get Up Across Morphologies: Zero-Shot Recovery with a Unified Humanoid Policy
- **分类: cs.RO; cs.LG**

- **简介: 该论文属机器人控制任务，旨在解决多形态人形机器人跌倒后零样本自主起身问题。提出一种基于CrossQ训练的统一深度强化学习策略，可跨7种不同尺寸、重量与动力学特性的机器人零样本迁移，成功率86%±7%，无需针对每种形态单独训练。**

- **链接: [https://arxiv.org/pdf/2512.12230v1](https://arxiv.org/pdf/2512.12230v1)**

> **作者:** Jonathan Spraggett
>
> **备注:** Accepted at 28th RoboCup International Symposium
>
> **摘要:** Fall recovery is a critical skill for humanoid robots in dynamic environments such as RoboCup, where prolonged downtime often decides the match. Recent techniques using deep reinforcement learning (DRL) have produced robust get-up behaviors, yet existing methods require training of separate policies for each robot morphology. This paper presents a single DRL policy capable of recovering from falls across seven humanoid robots with diverse heights (0.48 - 0.81 m), weights (2.8 - 7.9 kg), and dynamics. Trained with CrossQ, the unified policy transfers zero-shot up to 86 +/- 7% (95% CI [81, 89]) on unseen morphologies, eliminating the need for robot-specific training. Comprehensive leave-one-out experiments, morph scaling analysis, and diversity ablations show that targeted morphological coverage improves zero-shot generalization. In some cases, the shared policy even surpasses the specialist baselines. These findings illustrate the practicality of morphology-agnostic control for fall recovery, laying the foundation for generalist humanoid control. The software is open-source and available at: https://github.com/utra-robosoccer/unified-humanoid-getup
>
---
#### [new 052] Humanoid Robot Running Through Random Stepping Stones and Jumping Over Obstacles: Step Adaptation Using Spring-Mass Trajectories
- **分类: cs.RO**

- **简介: 该论文面向 humanoid robot 动态步态适应任务，解决在随机地形（如不规则踏石、障碍物）中鲁棒、敏捷运行的问题。提出基于弹簧质点轨迹与死区控制增益库的步态自适应框架，含自动建库、策略选择与全身控制映射，并验证其鲁棒性与泛化性。**

- **链接: [https://arxiv.org/pdf/2512.13304v1](https://arxiv.org/pdf/2512.13304v1)**

> **作者:** Sait Sovukluk; Johannes Englsberger; Christian Ott
>
> **备注:** Accepted for publication in Biomimetic Intelligence and Robotics. Supplemental video: https://youtu.be/HlAg2nbNct4
>
> **摘要:** This study proposes a step adaptation framework for running through spring-mass trajectories and deadbeat control gain libraries. It includes four main parts: (1) Automatic spring-mass trajectory library generation; (2) Deadbeat control gain library generation through an actively controlled template model that resembles the whole-body dynamics well; (3) Trajectory selection policy development for step adaptation; (4) Mapping spring-mass trajectories to a humanoid model through a whole-body control (WBC) framework also accounting for closed-kinematic chain systems, self collisions, and reactive limb swinging. We show the inclusiveness and the robustness of the proposed framework through various challenging and agile behaviors such as running through randomly generated stepping stones, jumping over random obstacles, performing slalom motions, changing the running direction suddenly with a random leg, and rejecting significant disturbances and uncertainties through the MuJoCo physics simulator. We also perform additional simulations under a comprehensive set of uncertainties and noise to better justify the proposed method's robustness to real-world challenges, including signal noise, imprecision, modeling errors, and delays. All the aforementioned behaviors are performed with a single library and the same set of WBC control parameters without additional tuning. The spring-mass and the deadbeat control gain library are automatically computed in 4.5 seconds in total for 315 different trajectories.
>
---
#### [new 053] MPC-Guided Safe Reinforcement Learning and Lipschitz-Based Filtering for Structured Nonlinear Systems
- **分类: cs.RO; eess.SY**

- **简介: 该论文属安全强化学习任务，旨在解决RL缺乏实时约束保障、MPC计算重且依赖精确模型的问题。提出MPC-RL融合框架：训练时用MPC引导安全探索，部署时用轻量Lipschitz滤波器保障约束满足，已在非线性气动弹性翼系统验证。**

- **链接: [https://arxiv.org/pdf/2512.12855v1](https://arxiv.org/pdf/2512.12855v1)**

> **作者:** Patrick Kostelac; Xuerui Wang; Anahita Jamshidnejad
>
> **摘要:** Modern engineering systems, such as autonomous vehicles, flexible robotics, and intelligent aerospace platforms, require controllers that are robust to uncertainties, adaptive to environmental changes, and safety-aware under real-time constraints. RL offers powerful data-driven adaptability for systems with nonlinear dynamics that interact with uncertain environments. RL, however, lacks built-in mechanisms for dynamic constraint satisfaction during exploration. MPC offers structured constraint handling and robustness, but its reliance on accurate models and computationally demanding online optimization may pose significant challenges. This paper proposes an integrated MPC-RL framework that combines stability and safety guarantees of MPC with the adaptability of RL. During training, MPC defines safe control bounds that guide the RL component and that enable constraint-aware policy learning. At deployment, the learned policy operates in real time with a lightweight safety filter based on Lipschitz continuity to ensure constraint satisfaction without heavy online optimizations. The approach, which is validated on a nonlinear aeroelastic wing system, demonstrates improved disturbance rejection, reduced actuator effort, and robust performance under turbulence. The architecture generalizes to other domains with structured nonlinearities and bounded disturbances, offering a scalable solution for safe artificial-intelligence-driven control in engineering applications.
>
---
#### [new 054] Bayesian Optimization Parameter Tuning Framework for a Lyapunov Based Path Following Controller
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对非线性路径跟踪控制器在真实硬件上参数调优难、试错成本高的问题，提出基于贝叶斯优化（BO）的自动调参框架。将闭环系统视为黑箱，用高斯过程代理模型高效搜索最优增益，在本田AI-Formula机器人上32次试验内显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.12649v1](https://arxiv.org/pdf/2512.12649v1)**

> **作者:** Zhewen Zheng; Wenjing Cao; Hongkang Yu; Mo Chen; Takashi Suzuki
>
> **摘要:** Parameter tuning in real-world experiments is constrained by the limited evaluation budget available on hardware. The path-following controller studied in this paper reflects a typical situation in nonlinear geometric controller, where multiple gains influence the dynamics through coupled nonlinear terms. Such interdependence makes manual tuning inefficient and unlikely to yield satisfactory performance within a practical number of trials. To address this challenge, we propose a Bayesian optimization (BO) framework that treats the closed-loop system as a black box and selects controller gains using a Gaussian-process surrogate. BO offers model-free exploration, quantified uncertainty, and data-efficient search, making it well suited for tuning tasks where each evaluation is costly. The framework is implemented on Honda's AI-Formula three-wheeled robot and assessed through repeated full-lap experiments on a fixed test track. The results show that BO improves controller performance within 32 trials, including 15 warm-start initial evaluations, indicating that it can efficiently locate high-performing regions of the parameter space under real-world conditions. These findings demonstrate that BO provides a practical, reliable, and data-efficient tuning approach for nonlinear path-following controllers on real robotic platforms.
>
---
#### [new 055] A Review of Learning-Based Motion Planning: Toward a Data-Driven Optimal Control Approach
- **分类: cs.RO; cs.AI**

- **简介: 该论文属综述与范式创新任务，旨在解决自动驾驶运动规划中“可解释性”与“适应性”的根本矛盾。作者系统梳理学习型方法演进，提出“数据驱动最优控制”新范式，融合经典控制结构与机器学习，支持人本定制、平台自适应与系统自优化。**

- **链接: [https://arxiv.org/pdf/2512.11944v1](https://arxiv.org/pdf/2512.11944v1)**

> **作者:** Jia Hu; Yang Chang; Haoran Wang
>
> **备注:** 34 pages, 11 figures
>
> **摘要:** Motion planning for high-level autonomous driving is constrained by a fundamental trade-off between the transparent, yet brittle, nature of pipeline methods and the adaptive, yet opaque, "black-box" characteristics of modern learning-based systems. This paper critically synthesizes the evolution of the field -- from pipeline methods through imitation learning, reinforcement learning, and generative AI -- to demonstrate how this persistent dilemma has hindered the development of truly trustworthy systems. To resolve this impasse, we conduct a comprehensive review of learning-based motion planning methods. Based on this review, we outline a data-driven optimal control paradigm as a unifying framework that synergistically integrates the verifiable structure of classical control with the adaptive capacity of machine learning, leveraging real-world data to continuously refine key components such as system dynamics, cost functions, and safety constraints. We explore this framework's potential to enable three critical next-generation capabilities: "Human-Centric" customization, "Platform-Adaptive" dynamics adaptation, and "System Self-Optimization" via self-tuning. We conclude by proposing future research directions based on this paradigm, aimed at developing intelligent transportation systems that are simultaneously safe, interpretable, and capable of human-like autonomy.
>
---
#### [new 056] Data-driven Interpretable Hybrid Robot Dynamics
- **分类: cs.RO**

- **简介: 该论文属机器人动力学建模任务，旨在解决传统黑盒模型缺乏可解释性与泛化性的问题。提出数据驱动的混合动力学方法：以解析刚体模型为基础，用符号回归或SINDy学习可解释的残差扭矩闭式表达式，在仿真与真实机器人上验证其高精度、强泛化与物理意义。**

- **链接: [https://arxiv.org/pdf/2512.11900v1](https://arxiv.org/pdf/2512.11900v1)**

> **作者:** Christopher E. Mower; Rui Zong; Haitham Bou-Ammar
>
> **摘要:** We study data-driven identification of interpretable hybrid robot dynamics, where an analytical rigid-body dynamics model is complemented by a learned residual torque term. Using symbolic regression and sparse identification of nonlinear dynamics (SINDy), we recover compact closed-form expressions for this residual from joint-space data. In simulation on a 7-DoF Franka arm with known dynamics, these interpretable models accurately recover inertial, Coriolis, gravity, and viscous effects with very small relative error and outperform neural-network baselines in both accuracy and generalization. On real data from a 7-DoF WAM arm, symbolic-regression residuals generalize substantially better than SINDy and neural networks, which tend to overfit, and suggest candidate new closed-form formulations that extend the nominal dynamics model for this robot. Overall, the results indicate that interpretable residual dynamics models provide compact, accurate, and physically meaningful alternatives to black-box function approximators for torque prediction.
>
---
#### [new 057] World Models Can Leverage Human Videos for Dexterous Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出DexWM世界模型，解决灵巧操作中因数据稀缺和细粒度手部动作建模难导致的环境状态预测不准问题。通过融合900+小时人类与机器人视频训练，并引入手部一致性辅助损失，提升视觉特征之外的手部姿态建模精度，实现零样本泛化至新操作任务。**

- **链接: [https://arxiv.org/pdf/2512.13644v1](https://arxiv.org/pdf/2512.13644v1)**

> **作者:** Raktim Gautam Goswami; Amir Bar; David Fan; Tsung-Yen Yang; Gaoyue Zhou; Prashanth Krishnamurthy; Michael Rabbat; Farshad Khorrami; Yann LeCun
>
> **摘要:** Dexterous manipulation is challenging because it requires understanding how subtle hand motion influences the environment through contact with objects. We introduce DexWM, a Dexterous Manipulation World Model that predicts the next latent state of the environment conditioned on past states and dexterous actions. To overcome the scarcity of dexterous manipulation datasets, DexWM is trained on over 900 hours of human and non-dexterous robot videos. To enable fine-grained dexterity, we find that predicting visual features alone is insufficient; therefore, we introduce an auxiliary hand consistency loss that enforces accurate hand configurations. DexWM outperforms prior world models conditioned on text, navigation, and full-body actions, achieving more accurate predictions of future states. DexWM also demonstrates strong zero-shot generalization to unseen manipulation skills when deployed on a Franka Panda arm equipped with an Allegro gripper, outperforming Diffusion Policy by over 50% on average in grasping, placing, and reaching tasks.
>
---
#### [new 058] Sim2Real Reinforcement Learning for Soccer skills
- **分类: cs.RO**

- **简介: 该论文属机器人控制任务，旨在解决Sim2Real迁移难问题。提出结合课程学习与对抗运动先验（AMP）的强化学习方法，提升人形机器人踢球、行走、跳跃等技能的动态性与适应性，但实验证明策略仍难以成功迁移到真实世界。**

- **链接: [https://arxiv.org/pdf/2512.12437v1](https://arxiv.org/pdf/2512.12437v1)**

> **作者:** Jonathan Spraggett
>
> **备注:** Undergrad Thesis
>
> **摘要:** This thesis work presents a more efficient and effective approach to training control-related tasks for humanoid robots using Reinforcement Learning (RL). The traditional RL methods are limited in adapting to real-world environments, complexity, and natural motions, but the proposed approach overcomes these limitations by using curriculum training and Adversarial Motion Priors (AMP) technique. The results show that the developed RL policies for kicking, walking, and jumping are more dynamic, and adaptive, and outperformed previous methods. However, the transfer of the learned policy from simulation to the real world was unsuccessful, highlighting the limitations of current RL methods in fully adapting to real-world scenarios.
>
---
#### [new 059] Intrinsic-Motivation Multi-Robot Social Formation Navigation with Coordinated Exploration
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究多机器人社交编队导航任务，旨在解决行人行为不可预测导致的协同探索低效问题。提出一种带内在动机的多机器人RL算法，含自学习内在奖励机制和双采样模式，以缓解策略保守性并提升协同探索效率。**

- **链接: [https://arxiv.org/pdf/2512.13293v1](https://arxiv.org/pdf/2512.13293v1)**

> **作者:** Hao Fua; Wei Liu; Shuai Zhoua
>
> **摘要:** This paper investigates the application of reinforcement learning (RL) to multi-robot social formation navigation, a critical capability for enabling seamless human-robot coexistence. While RL offers a promising paradigm, the inherent unpredictability and often uncooperative dynamics of pedestrian behavior pose substantial challenges, particularly concerning the efficiency of coordinated exploration among robots. To address this, we propose a novel coordinated-exploration multi-robot RL algorithm introducing an intrinsic motivation exploration. Its core component is a self-learning intrinsic reward mechanism designed to collectively alleviate policy conservatism. Moreover, this algorithm incorporates a dual-sampling mode within the centralized training and decentralized execution framework to enhance the representation of both the navigation policy and the intrinsic reward, leveraging a two-time-scale update rule to decouple parameter updates. Empirical results on social formation navigation benchmarks demonstrate the proposed algorithm's superior performance over existing state-of-the-art methods across crucial metrics. Our code and video demos are available at: https://github.com/czxhunzi/CEMRRL.
>
---
#### [new 060] SLIM-VDB: A Real-Time 3D Probabilistic Semantic Mapping Framework
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SLIM-VDB，一种轻量级实时3D概率语义建图框架。针对现有系统缺乏对闭集类别与开放语言标签统一融合支持、且未有效利用OpenVDB提升效率的问题，它基于OpenVDB设计统一贝叶斯语义融合机制，在显著降低内存与计算开销的同时保持建图精度。**

- **链接: [https://arxiv.org/pdf/2512.12945v1](https://arxiv.org/pdf/2512.12945v1)**

> **作者:** Anja Sheppard; Parker Ewen; Joey Wilson; Advaith V. Sethuraman; Benard Adewole; Anran Li; Yuzhen Chen; Ram Vasudevan; Katherine A. Skinner
>
> **备注:** Accepted into R-AL
>
> **摘要:** This paper introduces SLIM-VDB, a new lightweight semantic mapping system with probabilistic semantic fusion for closed-set or open-set dictionaries. Advances in data structures from the computer graphics community, such as OpenVDB, have demonstrated significantly improved computational and memory efficiency in volumetric scene representation. Although OpenVDB has been used for geometric mapping in robotics applications, semantic mapping for scene understanding with OpenVDB remains unexplored. In addition, existing semantic mapping systems lack support for integrating both fixed-category and open-language label predictions within a single framework. In this paper, we propose a novel 3D semantic mapping system that leverages the OpenVDB data structure and integrates a unified Bayesian update framework for both closed- and open-set semantic fusion. Our proposed framework, SLIM-VDB, achieves significant reduction in both memory and integration times compared to current state-of-the-art semantic mapping approaches, while maintaining comparable mapping accuracy. An open-source C++ codebase with a Python interface is available at https://github.com/umfieldrobotics/slim-vdb.
>
---
#### [new 061] Making Robots Play by the Rules: The ROS 2 CLIPS-Executive
- **分类: cs.RO**

- **简介: 该论文属机器人软件集成任务，旨在解决ROS 2中缺乏规则驱动决策能力的问题。作者将CLIPS规则引擎集成到ROS 2，构建CLIPS-Executive，并支持与PDDL规划框架协同，提升自主机器人任务协调的灵活性与知识表达能力。**

- **链接: [https://arxiv.org/pdf/2512.12722v1](https://arxiv.org/pdf/2512.12722v1)**

> **作者:** Tarik Viehmann; Daniel Swoboda; Samridhi Kalra; Himanshu Grover; Gerhard Lakemeyer
>
> **摘要:** CLIPS is a rule-based programming language for building knowledge-driven applications, well suited for the complex task of coordinating autonomous robots. Inspired by the CLIPS-Executive originally developed for the lesser known Fawkes robotics framework, we present an Integration of CLIPS into the ROS ecosystem. Additionally, we show the flexibility of CLIPS by describing a PDDL-based planning framework integration.
>
---
#### [new 062] Lightweight Dynamic Modeling of Cable-Driven Continuum Robots Based on Actuation-Space Energy Formulation
- **分类: cs.RO**

- **简介: 该论文面向电缆驱动连续体机器人（CDCR）的实时动态建模任务，旨在解决现有模型计算重、难兼顾精度与效率的问题。提出轻量化的执行空间能量建模（LASEM）框架，通过变分法导出单PDE动力学方程，避免接触力建模，支持力/位移双输入模式，并实现62.3%加速。**

- **链接: [https://arxiv.org/pdf/2512.13271v1](https://arxiv.org/pdf/2512.13271v1)**

> **作者:** Fangju Yang; Hang Yang; Ibrahim Alsarraj; Yuhao Wang; Ke Wu
>
> **摘要:** Cable-driven continuum robots (CDCRs) require accurate, real-time dynamic models for high-speed dynamics prediction or model-based control, making such capability an urgent need. In this paper, we propose the Lightweight Actuation-Space Energy Modeling (LASEM) framework for CDCRs, which formulates actuation potential energy directly in actuation space to enable lightweight yet accurate dynamic modeling. Through a unified variational derivation, the governing dynamics reduce to a single partial differential equation (PDE), requiring only the Euler moment balance while implicitly incorporating the Newton force balance. By also avoiding explicit computation of cable-backbone contact forces, the formulation simplifies the model structure and improves computational efficiency while preserving geometric accuracy and physical consistency. Importantly, the proposed framework for dynamic modeling natively supports both force-input and displacement-input actuation modes, a capability seldom achieved in existing dynamic formulations. Leveraging this lightweight structure, a Galerkin space-time modal discretization with analytical time-domain derivatives of the reduced state further enables an average 62.3% computational speedup over state-of-the-art real-time dynamic modeling approaches.
>
---
#### [new 063] High Order Control Lyapunov Function - Control Barrier Function - Quadratic Programming Based Autonomous Driving Controller for Bicyclist Safety
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于自动驾驶安全控制任务，旨在解决与自行车骑行者交互中的碰撞规避难题。提出基于高阶CLF-CBF-QP的控制器，融合稳定性（CLF）与安全性（CBF）约束，通过二次规划生成最优控制指令，并在FARS典型事故场景中验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.12776v1](https://arxiv.org/pdf/2512.12776v1)**

> **作者:** Haochong Chen; Xincheng Cao; Levent Guvenc; Bilin Aksun-Guvenc
>
> **摘要:** Ensuring the safety of Vulnerable Road Users (VRUs) is a critical challenge in the development of advanced autonomous driving systems in smart cities. Among vulnerable road users, bicyclists present unique characteristics that make their safety both critical and also manageable. Vehicles often travel at significantly higher relative speeds when interacting with bicyclists as compared to their interactions with pedestrians which makes collision avoidance system design for bicyclist safety more challenging. Yet, bicyclist movements are generally more predictable and governed by clear traffic rules as compared to the sudden and sometimes erratic pedestrian motion, offering opportunities for model-based control strategies. To address bicyclist safety in complex traffic environments, this study proposes and develops a High Order Control Lyapunov Function High Order Control Barrier Function Quadratic Programming (HOCLF HOCBF QP) control framework. Through this framework, CLFs constraints guarantee system stability so that the vehicle can track its reference trajectory, whereas CBFs constraints ensure system safety by letting vehicle avoiding potential collisions region with surrounding obstacles. Then by solving a QP problem, an optimal control command that simultaneously satisfies stability and safety requirements can be calculated. Three key bicyclist crash scenarios recorded in the Fatality Analysis Reporting System (FARS) are recreated and used to comprehensively evaluate the proposed autonomous driving bicyclist safety control strategy in a simulation study. Simulation results demonstrate that the HOCLF HOCBF QP controller can help the vehicle perform robust, and collision-free maneuvers, highlighting its potential for improving bicyclist safety in complex traffic environments.
>
---
#### [new 064] MindDrive: A Vision-Language-Action Model for Autonomous Driving via Online Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MindDrive，一种面向自动驾驶的视觉-语言-动作（VLA）模型，旨在解决模仿学习导致的分布偏移与因果混淆问题。通过引入双LoRA参数的LLM——分别作为决策专家与动作专家，并将轨迹奖励反馈至语言决策空间，实现基于离散语言决策的在线强化学习，在Bench2Drive上取得领先性能。**

- **链接: [https://arxiv.org/pdf/2512.13636v1](https://arxiv.org/pdf/2512.13636v1)**

> **作者:** Haoyu Fu; Diankun Zhang; Zongchuang Zhao; Jianfeng Cui; Hongwei Xie; Bing Wang; Guang Chen; Dingkang Liang; Xiang Bai
>
> **备注:** 16 pages, 12 figures, 6 tables; Project Page: https://xiaomi-mlab.github.io/MindDrive/
>
> **摘要:** Current Vision-Language-Action (VLA) paradigms in autonomous driving primarily rely on Imitation Learning (IL), which introduces inherent challenges such as distribution shift and causal confusion. Online Reinforcement Learning offers a promising pathway to address these issues through trial-and-error learning. However, applying online reinforcement learning to VLA models in autonomous driving is hindered by inefficient exploration in continuous action spaces. To overcome this limitation, we propose MindDrive, a VLA framework comprising a large language model (LLM) with two distinct sets of LoRA parameters. The one LLM serves as a Decision Expert for scenario reasoning and driving decision-making, while the other acts as an Action Expert that dynamically maps linguistic decisions into feasible trajectories. By feeding trajectory-level rewards back into the reasoning space, MindDrive enables trial-and-error learning over a finite set of discrete linguistic driving decisions, instead of operating directly in a continuous action space. This approach effectively balances optimal decision-making in complex scenarios, human-like driving behavior, and efficient exploration in online reinforcement learning. MindDrive achieves strong closed-loop performance on the challenging Bench2Drive benchmark, with a Driving Score (DS) of 78.04 and a Success Rate (SR) of 55.09%. To the best of our knowledge, this is the first work to demonstrate the effectiveness of online reinforcement learning for the VLA model in autonomous driving.
>
---
#### [new 065] MMDrive: Interactive Scene Understanding Beyond Vision with Multi-representational Fusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向自动驾驶场景理解任务，旨在解决现有视觉语言模型受限于2D图像、缺乏3D空间感知与深度语义融合的问题。提出MMDrive框架，融合占用图、LiDAR点云和文本描述，设计文本导向调制器与跨模态抽象器，实现自适应多模态融合，在DriveLM和NuScenes-QA上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.13177v1](https://arxiv.org/pdf/2512.13177v1)**

> **作者:** Minghui Hou; Wei-Hsing Huang; Shaofeng Liang; Daizong Liu; Tai-Hao Wen; Gang Wang; Runwei Guan; Weiping Ding
>
> **摘要:** Vision-language models enable the understanding and reasoning of complex traffic scenarios through multi-source information fusion, establishing it as a core technology for autonomous driving. However, existing vision-language models are constrained by the image understanding paradigm in 2D plane, which restricts their capability to perceive 3D spatial information and perform deep semantic fusion, resulting in suboptimal performance in complex autonomous driving environments. This study proposes MMDrive, an multimodal vision-language model framework that extends traditional image understanding to a generalized 3D scene understanding framework. MMDrive incorporates three complementary modalities, including occupancy maps, LiDAR point clouds, and textual scene descriptions. To this end, it introduces two novel components for adaptive cross-modal fusion and key information extraction. Specifically, the Text-oriented Multimodal Modulator dynamically weights the contributions of each modality based on the semantic cues in the question, guiding context-aware feature integration. The Cross-Modal Abstractor employs learnable abstract tokens to generate compact, cross-modal summaries that highlight key regions and essential semantics. Comprehensive evaluations on the DriveLM and NuScenes-QA benchmarks demonstrate that MMDrive achieves significant performance gains over existing vision-language models for autonomous driving, with a BLEU-4 score of 54.56 and METEOR of 41.78 on DriveLM, and an accuracy score of 62.7% on NuScenes-QA. MMDrive effectively breaks the traditional image-only understanding barrier, enabling robust multimodal reasoning in complex driving environments and providing a new foundation for interpretable autonomous driving scene understanding.
>
---
#### [new 066] From Human Intention to Action Prediction: A Comprehensive Benchmark for Intention-driven End-to-End Autonomous Driving
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文面向意图驱动的端到端自动驾驶任务，旨在解决现有系统仅执行低级指令、无法理解高阶人类意图的问题。作者构建首个基准Intention-Drive，含意图-场景配对数据集与基于意图成功率（ISR）的新评估协议，揭示当前模型在语义意图理解上的显著不足。**

- **链接: [https://arxiv.org/pdf/2512.12302v1](https://arxiv.org/pdf/2512.12302v1)**

> **作者:** Huan Zheng; Yucheng Zhou; Tianyi Yan; Jiayi Su; Hongjun Chen; Dubing Chen; Wencheng Han; Runzhou Tao; Zhongying Qiu; Jianfei Yang; Jianbing Shen
>
> **摘要:** Current end-to-end autonomous driving systems operate at a level of intelligence akin to following simple steering commands. However, achieving genuinely intelligent autonomy requires a paradigm shift: moving from merely executing low-level instructions to understanding and fulfilling high-level, abstract human intentions. This leap from a command-follower to an intention-fulfiller, as illustrated in our conceptual framework, is hindered by a fundamental challenge: the absence of a standardized benchmark to measure and drive progress on this complex task. To address this critical gap, we introduce Intention-Drive, the first comprehensive benchmark designed to evaluate the ability to translate high-level human intent into safe and precise driving actions. Intention-Drive features two core contributions: (1) a new dataset of complex scenarios paired with corresponding natural language intentions, and (2) a novel evaluation protocol centered on the Intent Success Rate (ISR), which assesses the semantic fulfillment of the human's goal beyond simple geometric accuracy. Through an extensive evaluation of a spectrum of baseline models on Intention-Drive, we reveal a significant performance deficit, showing that the baseline model struggle to achieve the comprehensive scene and intention understanding required for this advanced task.
>
---
#### [new 067] Taylor-Lagrange Control for Safety-Critical Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文提出Taylor-Lagrange控制（TLC）方法，面向安全关键非线性系统，解决高阶约束下安全与稳定控制问题。通过结合Lie导数与带Lagrange余项的泰勒展开，显式引入控制输入，给出安全性的充要条件，并扩展CBF/CLF至复域与高阶情形，降低保守性，支持QP实时求解与事件触发实现。**

- **链接: [https://arxiv.org/pdf/2512.11999v1](https://arxiv.org/pdf/2512.11999v1)**

> **作者:** Wei Xiao; Anni Li
>
> **备注:** 13 pages
>
> **摘要:** This paper proposes a novel Taylor-Lagrange Control (TLC) method for nonlinear control systems to ensure the safety and stability through Taylor's theorem with Lagrange remainder. To achieve this, we expand a safety or stability function with respect to time along the system dynamics using the Lie derivative and Taylor's theorem. This expansion enables the control input to appear in the Taylor series at an order equivalent to the relative degree of the function. We show that the proposed TLC provides necessary and sufficient conditions for system safety and is applicable to systems and constraints of arbitrary relative degree. The TLC exhibits connections with existing Control Barrier Function (CBF) and Control Lyapunov Function (CLF) methods, and it further extends the CBF and CLF methods to the complex domain, especially for higher order cases. Compared to High-Order CBFs (HOCBFs), TLC is less restrictive as it does not require forward invariance of the intersection of a set of safe sets while HOCBFs do. We employ TLC to reformulate a constrained optimal control problem as a sequence of quadratic programs with a zero-order hold implementation method, and demonstrate the safety of zero-order hold TLC using an event-triggered control method to address inter-sampling effects. Finally, we illustrate the effectiveness of the proposed TLC method through an adaptive cruise control system and a robot control problem, and compare it with existing CBF methods.
>
---
#### [new 068] SAMAY: System for Acoustic Measurement and Analysis
- **分类: cs.SD; cs.RO**

- **简介: 该论文提出SAMAY系统，属智能环境监测任务，旨在解决野外鸟类声学数据自动采集与分析难题。工作包括设计基于STM32F407的便携式录音设备，支持4麦克风、128GB存储、太阳能供电，并实现USB/Wi-Fi实时配置，支撑物种识别、种群监测与环境影响分析。**

- **链接: [https://arxiv.org/pdf/2512.13284v1](https://arxiv.org/pdf/2512.13284v1)**

> **作者:** Adheep Arya G R; Vaibhav Pratap Singh; Mayank Kumar; Niyathi Shenoy; Tejas Suryawanshi; Ruchi Juyal; Sangit Saha; Kaushik Nanda; Hari Babu Pasupuleti; S D Sudarsan
>
> **摘要:** This paper describes an automatic bird call recording system called SAMAY, which is developed to study bird species by creating a database of large amounts of bird acoustic data. By analysing the recorded bird call data, the system can also be used for automatic classification of bird species, monitoring bird populations and analysing the impact of environmental changes. The system is driven through a powerful STM32F407 series microcontroller, supports 4 microphones, is equipped with 128 GB of storage capacity, and is powered by a 10400 mAh battery pack interfaced with a solar charger. In addition, the device is user-configurable over USB and Wi-Fi during runtime, ensuring user-friendly operation during field deployment.
>
---
#### [new 069] Cross-Level Sensor Fusion with Object Lists via Transformer for 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向3D目标检测任务，解决智能传感器与V2X模块仅提供处理后的对象列表（非原始数据）时的跨层级融合难题。提出基于Transformer的端到端方法：将对象列表作为去噪查询和高斯掩码先验融入图像特征融合，提升检测精度与收敛速度，并构建伪对象列表生成策略。**

- **链接: [https://arxiv.org/pdf/2512.12884v1](https://arxiv.org/pdf/2512.12884v1)**

> **作者:** Xiangzhong Liu; Jiajie Zhang; Hao Shen
>
> **备注:** 6 pages, 3 figures, accepted at IV2025
>
> **摘要:** In automotive sensor fusion systems, smart sensors and Vehicle-to-Everything (V2X) modules are commonly utilized. Sensor data from these systems are typically available only as processed object lists rather than raw sensor data from traditional sensors. Instead of processing other raw data separately and then fusing them at the object level, we propose an end-to-end cross-level fusion concept with Transformer, which integrates highly abstract object list information with raw camera images for 3D object detection. Object lists are fed into a Transformer as denoising queries and propagated together with learnable queries through the latter feature aggregation process. Additionally, a deformable Gaussian mask, derived from the positional and size dimensional priors from the object lists, is explicitly integrated into the Transformer decoder. This directs attention toward the target area of interest and accelerates model training convergence. Furthermore, as there is no public dataset containing object lists as a standalone modality, we propose an approach to generate pseudo object lists from ground-truth bounding boxes by simulating state noise and false positives and negatives. As the first work to conduct cross-level fusion, our approach shows substantial performance improvements over the vision-based baseline on the nuScenes dataset. It demonstrates its generalization capability over diverse noise levels of simulated object lists and real detectors.
>
---
#### [new 070] Car-following Models and Congestion Control with Followerstopper on a Ring-Road under Known Delay -- Examining Limit Cycle
- **分类: eess.SY; cs.RO; eess.SP**

- **简介: 该论文属交通流控制任务，旨在解决环形道路中IDM模型因延迟引发的拥堵波（止-走波）问题。通过动力系统分析，验证单辆Followerstopper自动驾驶车可稳定全队列、抑制波传播，即使在已知通信延迟下仍有效。**

- **链接: [https://arxiv.org/pdf/2512.11842v1](https://arxiv.org/pdf/2512.11842v1)**

> **作者:** Trevor McClain; Rahul Bhadani
>
> **备注:** Submitted to IV 2026
>
> **摘要:** This paper examines the IDM microscopic car-following model from a dynamical systems perspective, analyzing the effects of delay on congestion formation. Further, a case of mixed-autonomy is considered by controlling one car with Followerstopper in a ring road setting containing IDM vehicles as human drivers. Specifically, the stop-and-go waves phenomenon in idealized traffic from a dynamical systems perspective is examined. We show that Followerstopper-controlled vehicle is effective at eliminating emergent stop-and-go waves in the IDM traffic simulation. We show through simulation that the uniform flow manifold is unstable for the ring road simulation with IDM vehicles, and that replacing a single car with Followerstopper induces stability, allowing the cars to drive safely at a uniform speed. Additionally, the case of known delay is considered in a mixed-autonomy scenario. Our simulation result shows that while considering a known time delay, traffic waves emerge earlier than in the no-delay case. At the same time, a single-vehicle controlled using Followerstopper controller is able to prevent the emergence of traffic waves even in the presence of delay.
>
---
#### [new 071] Explainable Adversarial-Robust Vision-Language-Action Model for Robotic Manipulation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文面向智能农业机器人操作任务，解决RGB视觉系统在光度扰动（如色相、光照、噪声）下鲁棒性与可解释性不足的问题；提出基于OpenVLA-OFT的可解释对抗鲁棒视觉-语言-动作模型，引入Evidence-3模块检测扰动并生成自然语言解释，显著提升动作预测精度与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.11865v1](https://arxiv.org/pdf/2512.11865v1)**

> **作者:** Ju-Young Kim; Ji-Hong Park; Myeongjun Kim; Gun-Woo Kim
>
> **备注:** Accepted to MobieSec 2025 (poster session)
>
> **摘要:** Smart farming has emerged as a key technology for advancing modern agriculture through automation and intelligent control. However, systems relying on RGB cameras for perception and robotic manipulators for control, common in smart farming, are vulnerable to photometric perturbations such as hue, illumination, and noise changes, which can cause malfunction under adversarial attacks. To address this issue, we propose an explainable adversarial-robust Vision-Language-Action model based on the OpenVLA-OFT framework. The model integrates an Evidence-3 module that detects photometric perturbations and generates natural language explanations of their causes and effects. Experiments show that the proposed model reduces Current Action L1 loss by 21.7% and Next Actions L1 loss by 18.4% compared to the baseline, demonstrating improved action prediction accuracy and explainability under adversarial conditions.
>
---
#### [new 072] Motus: A Unified Latent Action World Model
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出Motus，一种统一的潜在动作世界模型，旨在解决现有具身智能体中理解、建模与控制模块割裂的问题。它融合多专家Transformer架构、光流驱动的潜在动作学习及三阶段训练，实现世界建模、视觉-语言-动作联合预测等多功能统一，显著提升仿真与真实机器人任务性能。**

- **链接: [https://arxiv.org/pdf/2512.13030v1](https://arxiv.org/pdf/2512.13030v1)**

> **作者:** Hongzhe Bi; Hengkai Tan; Shenghao Xie; Zeyuan Wang; Shuhe Huang; Haitian Liu; Ruowen Zhao; Yao Feng; Chendong Xiang; Yinze Rong; Hongyan Zhao; Hanyu Liu; Zhizhong Su; Lei Ma; Hang Su; Jun Zhu
>
> **摘要:** While a general embodied agent must function as a unified system, current methods are built on isolated models for understanding, world modeling, and control. This fragmentation prevents unifying multimodal generative capabilities and hinders learning from large-scale, heterogeneous data. In this paper, we propose Motus, a unified latent action world model that leverages existing general pretrained models and rich, sharable motion information. Motus introduces a Mixture-of-Transformer (MoT) architecture to integrate three experts (i.e., understanding, video generation, and action) and adopts a UniDiffuser-style scheduler to enable flexible switching between different modeling modes (i.e., world models, vision-language-action models, inverse dynamics models, video generation models, and video-action joint prediction models). Motus further leverages the optical flow to learn latent actions and adopts a recipe with three-phase training pipeline and six-layer data pyramid, thereby extracting pixel-level "delta action" and enabling large-scale action pretraining. Experiments show that Motus achieves superior performance against state-of-the-art methods in both simulation (a +15% improvement over X-VLA and a +45% improvement over Pi0.5) and real-world scenarios(improved by +11~48%), demonstrating unified modeling of all functionalities and priors significantly benefits downstream robotic tasks.
>
---
#### [new 073] D3D-VLP: Dynamic 3D Vision-Language-Planning Model for Embodied Grounding and Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向具身智能任务，解决端到端模型缺乏3D可解释性与模块化系统忽视组件协同的矛盾。提出D3D-VLP模型：引入动态3D思维链统一多任务，设计碎片化监督协同学习策略，并构建10M混合数据集，显著提升导航、目标导航与任务导向定位等基准性能。**

- **链接: [https://arxiv.org/pdf/2512.12622v1](https://arxiv.org/pdf/2512.12622v1)**

> **作者:** Zihan Wang; Seungjun Lee; Guangzhao Dai; Gim Hee Lee
>
> **摘要:** Embodied agents face a critical dilemma that end-to-end models lack interpretability and explicit 3D reasoning, while modular systems ignore cross-component interdependencies and synergies. To bridge this gap, we propose the Dynamic 3D Vision-Language-Planning Model (D3D-VLP). Our model introduces two key innovations: 1) A Dynamic 3D Chain-of-Thought (3D CoT) that unifies planning, grounding, navigation, and question answering within a single 3D-VLM and CoT pipeline; 2) A Synergistic Learning from Fragmented Supervision (SLFS) strategy, which uses a masked autoregressive loss to learn from massive and partially-annotated hybrid data. This allows different CoT components to mutually reinforce and implicitly supervise each other. To this end, we construct a large-scale dataset with 10M hybrid samples from 5K real scans and 20K synthetic scenes that are compatible with online learning methods such as RL and DAgger. Our D3D-VLP achieves state-of-the-art results on multiple benchmarks, including Vision-and-Language Navigation (R2R-CE, REVERIE-CE, NavRAG-CE), Object-goal Navigation (HM3D-OVON), and Task-oriented Sequential Grounding and Navigation (SG3D). Real-world mobile manipulation experiments further validate the effectiveness.
>
---
#### [new 074] Goal Reaching with Eikonal-Constrained Hierarchical Quasimetric Reinforcement Learning
- **分类: cs.LG; cs.RO; eess.SY; stat.ML**

- **简介: 该论文属目标导向强化学习（GCRL）任务，旨在解决奖励设计难与泛化性差问题。提出Eik-QRL（基于Eikonal PDE的连续时间拟度量RL）及分层扩展Eik-HiQRL，实现轨迹无关学习、提升OOD泛化，并在导航与操作任务中达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.12046v1](https://arxiv.org/pdf/2512.12046v1)**

> **作者:** Vittorio Giammarino; Ahmed H. Qureshi
>
> **摘要:** Goal-Conditioned Reinforcement Learning (GCRL) mitigates the difficulty of reward design by framing tasks as goal reaching rather than maximizing hand-crafted reward signals. In this setting, the optimal goal-conditioned value function naturally forms a quasimetric, motivating Quasimetric RL (QRL), which constrains value learning to quasimetric mappings and enforces local consistency through discrete, trajectory-based constraints. We propose Eikonal-Constrained Quasimetric RL (Eik-QRL), a continuous-time reformulation of QRL based on the Eikonal Partial Differential Equation (PDE). This PDE-based structure makes Eik-QRL trajectory-free, requiring only sampled states and goals, while improving out-of-distribution generalization. We provide theoretical guarantees for Eik-QRL and identify limitations that arise under complex dynamics. To address these challenges, we introduce Eik-Hierarchical QRL (Eik-HiQRL), which integrates Eik-QRL into a hierarchical decomposition. Empirically, Eik-HiQRL achieves state-of-the-art performance in offline goal-conditioned navigation and yields consistent gains over QRL in manipulation tasks, matching temporal-difference methods.
>
---
#### [new 075] A Hybrid Deep Learning Framework for Emotion Recognition in Children with Autism During NAO Robot-Mediated Interaction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属情感识别任务，旨在解决自闭症儿童在NAO机器人互动中微表情识别难的问题。提出融合ResNet-50 CNN与GCN的混合深度学习框架，利用FaceMesh几何特征与视觉特征，结合加权软标签和KL散度优化，实现七类情绪识别。基于印度首个大规模ASD儿童机器人交互数据集。**

- **链接: [https://arxiv.org/pdf/2512.12208v1](https://arxiv.org/pdf/2512.12208v1)**

> **作者:** Indranil Bhattacharjee; Vartika Narayani Srinet; Anirudha Bhattacharjee; Braj Bhushan; Bishakh Bhattacharya
>
> **备注:** 12 pages, journal paper
>
> **摘要:** Understanding emotional responses in children with Autism Spectrum Disorder (ASD) during social interaction remains a critical challenge in both developmental psychology and human-robot interaction. This study presents a novel deep learning pipeline for emotion recognition in autistic children in response to a name-calling event by a humanoid robot (NAO), under controlled experimental settings. The dataset comprises of around 50,000 facial frames extracted from video recordings of 15 children with ASD. A hybrid model combining a fine-tuned ResNet-50-based Convolutional Neural Network (CNN) and a three-layer Graph Convolutional Network (GCN) trained on both visual and geometric features extracted from MediaPipe FaceMesh landmarks. Emotions were probabilistically labeled using a weighted ensemble of two models: DeepFace's and FER, each contributing to soft-label generation across seven emotion classes. Final classification leveraged a fused embedding optimized via Kullback-Leibler divergence. The proposed method demonstrates robust performance in modeling subtle affective responses and offers significant promise for affective profiling of ASD children in clinical and therapeutic human-robot interaction contexts, as the pipeline effectively captures micro emotional cues in neurodivergent children, addressing a major gap in autism-specific HRI research. This work represents the first such large-scale, real-world dataset and pipeline from India on autism-focused emotion analysis using social robotics, contributing an essential foundation for future personalized assistive technologies.
>
---
#### [new 076] Semantic-Drive: Democratizing Long-Tail Data Curation via Open-Vocabulary Grounding and Neuro-Symbolic VLM Consensus
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属自动驾驶长尾数据挖掘任务，旨在解决罕见安全事件人工标注成本高、隐私差的问题。提出Semantic-Drive框架：本地化、神经符号融合，结合开放词汇检测（YOLOE）与推理型VLM，辅以多模型共识机制，提升召回率与风险评估精度，支持消费级硬件部署。**

- **链接: [https://arxiv.org/pdf/2512.12012v1](https://arxiv.org/pdf/2512.12012v1)**

> **作者:** Antonio Guillen-Perez
>
> **摘要:** The development of robust Autonomous Vehicles (AVs) is bottlenecked by the scarcity of "Long-Tail" training data. While fleets collect petabytes of video logs, identifying rare safety-critical events (e.g., erratic jaywalking, construction diversions) remains a manual, cost-prohibitive process. Existing solutions rely on coarse metadata search, which lacks precision, or cloud-based VLMs, which are privacy-invasive and expensive. We introduce Semantic-Drive, a local-first, neuro-symbolic framework for semantic data mining. Our approach decouples perception into two stages: (1) Symbolic Grounding via a real-time open-vocabulary detector (YOLOE) to anchor attention, and (2) Cognitive Analysis via a Reasoning VLM that performs forensic scene analysis. To mitigate hallucination, we implement a "System 2" inference-time alignment strategy, utilizing a multi-model "Judge-Scout" consensus mechanism. Benchmarked on the nuScenes dataset against the Waymo Open Dataset (WOD-E2E) taxonomy, Semantic-Drive achieves a Recall of 0.966 (vs. 0.475 for CLIP) and reduces Risk Assessment Error by 40\% compared to single models. The system runs entirely on consumer hardware (NVIDIA RTX 3090), offering a privacy-preserving alternative to the cloud.
>
---
#### [new 077] SignRAG: A Retrieval-Augmented System for Scalable Zero-Shot Road Sign Recognition
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.RO**

- **简介: 该论文面向零样本路标识别任务，解决传统深度学习因类别多、标注难导致的泛化瓶颈。提出SignRAG框架：用视觉语言模型生成文本描述，检索参考库候选，再由大语言模型推理识别。在303类美国路标上验证，效果显著。**

- **链接: [https://arxiv.org/pdf/2512.12885v1](https://arxiv.org/pdf/2512.12885v1)**

> **作者:** Minghao Zhu; Zhihao Zhang; Anmol Sidhu; Keith Redmill
>
> **备注:** Submitted to IV 2026
>
> **摘要:** Automated road sign recognition is a critical task for intelligent transportation systems, but traditional deep learning methods struggle with the sheer number of sign classes and the impracticality of creating exhaustive labeled datasets. This paper introduces a novel zero-shot recognition framework that adapts the Retrieval-Augmented Generation (RAG) paradigm to address this challenge. Our method first uses a Vision Language Model (VLM) to generate a textual description of a sign from an input image. This description is used to retrieve a small set of the most relevant sign candidates from a vector database of reference designs. Subsequently, a Large Language Model (LLM) reasons over the retrieved candidates to make a final, fine-grained recognition. We validate this approach on a comprehensive set of 303 regulatory signs from the Ohio MUTCD. Experimental results demonstrate the framework's effectiveness, achieving 95.58% accuracy on ideal reference images and 82.45% on challenging real-world road data. This work demonstrates the viability of RAG-based architectures for creating scalable and accurate systems for road sign recognition without task-specific training.
>
---
## 更新

#### [replaced 001] Enhancing Interpretability and Interactivity in Robot Manipulation: A Neurosymbolic Approach
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文提出一种神经符号架构，解决自然语言驱动的机器人操作中可解释性与交互性不足的问题。通过将语言解析为可执行的符号-神经混合程序，调用视觉、逻辑、控制等原子技能，实现任务无关的指代、问答与抓取。在仿真与真实场景中验证了其准确性、泛化性与可迁移性。**

- **链接: [https://arxiv.org/pdf/2210.00858v4](https://arxiv.org/pdf/2210.00858v4)**

> **作者:** Georgios Tziafas; Hamidreza Kasaei
>
> **备注:** Published in International Journal of Robotics Research (IJRR) (2025)
>
> **摘要:** In this paper we present a neurosymbolic architecture for coupling language-guided visual reasoning with robot manipulation. A non-expert human user can prompt the robot using unconstrained natural language, providing a referring expression (REF), a question (VQA), or a grasp action instruction. The system tackles all cases in a task-agnostic fashion through the utilization of a shared library of primitive skills. Each primitive handles an independent sub-task, such as reasoning about visual attributes, spatial relation comprehension, logic and enumeration, as well as arm control. A language parser maps the input query to an executable program composed of such primitives, depending on the context. While some primitives are purely symbolic operations (e.g. counting), others are trainable neural functions (e.g. visual grounding), therefore marrying the interpretability and systematic generalization benefits of discrete symbolic approaches with the scalability and representational power of deep networks. We generate a 3D vision-and-language synthetic dataset of tabletop scenes in a simulation environment to train our approach and perform extensive evaluations in both synthetic and real-world scenes. Results showcase the benefits of our approach in terms of accuracy, sample-efficiency, and robustness to the user's vocabulary, while being transferable to real-world scenes with few-shot visual fine-tuning. Finally, we integrate our method with a robot framework and demonstrate how it can serve as an interpretable solution for an interactive object-picking task, achieving an average success rate of 80.2\%, both in simulation and with a real robot. We make supplementary material available in https://gtziafas.github.io/neurosymbolic-manipulation.
>
---
#### [replaced 002] Ctrl-Crash: Controllable Diffusion for Realistic Car Crashes
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属可控视频生成任务，旨在解决真实车祸视频数据稀缺导致的生成不真实问题。提出Ctrl-Crash模型，以边界框、碰撞类型和初始帧为条件，结合分类器自由引导实现细粒度控制，支持反事实场景生成，在质量与物理真实性上达SOTA。**

- **链接: [https://arxiv.org/pdf/2506.00227v2](https://arxiv.org/pdf/2506.00227v2)**

> **作者:** Anthony Gosselin; Ge Ya Luo; Luis Lara; Florian Golemo; Derek Nowrouzezahrai; Liam Paull; Alexia Jolicoeur-Martineau; Christopher Pal
>
> **备注:** Under review at Pattern Recognition Letters
>
> **摘要:** Video diffusion techniques have advanced significantly in recent years; however, they struggle to generate realistic imagery of car crashes due to the scarcity of accident events in most driving datasets. Improving traffic safety requires realistic and controllable accident simulations. To tackle the problem, we propose Ctrl-Crash, a controllable car crash video generation model that conditions on signals such as bounding boxes, crash types, and an initial image frame. Our approach enables counterfactual scenario generation where minor variations in input can lead to dramatically different crash outcomes. To support fine-grained control at inference time, we leverage classifier-free guidance with independently tunable scales for each conditioning signal. Ctrl-Crash achieves state-of-the-art performance across quantitative video quality metrics (e.g., FVD and JEDi) and qualitative measurements based on a human-evaluation of physical realism and video quality compared to prior diffusion-based methods.
>
---
#### [replaced 003] End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy: VR Teleoperation Augmented by Autonomous Hand VLA Policy for Efficient Data Collection
- **分类: cs.RO; cs.AI**

- **简介: 该论文属机器人灵巧操作任务，旨在解决VLA模型因高质量数据稀缺而难以扩展的问题。提出共享自主框架：VR遥控臂部宏观运动，自主DexGrasp-VLA策略控制手部微观操作；并设计臂手特征增强模块与纠错遥操作机制，实现高效数据收集与端到端策略训练。**

- **链接: [https://arxiv.org/pdf/2511.00139v2](https://arxiv.org/pdf/2511.00139v2)**

> **作者:** Yu Cui; Yujian Zhang; Lina Tao; Yang Li; Xinyu Yi; Zhibin Li
>
> **摘要:** Achieving human-like dexterous manipulation remains a major challenge for general-purpose robots. While Vision-Language-Action (VLA) models show potential in learning skills from demonstrations, their scalability is limited by scarce high-quality training data. Existing data collection methods face inherent constraints: manual teleoperation overloads human operators, while automated planning often produces unnatural motions. We propose a Shared Autonomy framework that divides control between macro and micro motions. A human operator guides the robot's arm pose through intuitive VR teleoperation, while an autonomous DexGrasp-VLA policy handles fine-grained hand control using real-time tactile and visual feedback. This division significantly reduces cognitive load and enables efficient collection of high-quality coordinated arm-hand demonstrations. Using this data, we train an end-to-end VLA policy enhanced with our novel Arm-Hand Feature Enhancement module, which captures both distinct and shared representations of macro and micro movements for more natural coordination. Our Corrective Teleoperation system enables continuous policy improvement through human-in-the-loop failure recovery. Experiments demonstrate that our framework generates high-quality data with minimal manpower and achieves a 90% success rate across diverse objects, including unseen instances. Comprehensive evaluations validate the system's effectiveness in developing dexterous manipulation capabilities.
>
---
#### [replaced 004] LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba
- **分类: cs.RO; cs.AI; cs.CV; eess.IV; eess.SY**

- **简介: 论文提出LocoMamba，一种基于Mamba架构的端到端视觉驱动仿生运动控制方法，解决复杂地形与动态障碍下机器人高效、鲁棒步态学习问题。工作包括：多模态状态嵌入、Mamba序列建模融合、随机化训练与课程学习，显著提升性能、泛化性与训练效率。**

- **链接: [https://arxiv.org/pdf/2508.11849v3](https://arxiv.org/pdf/2508.11849v3)**

> **作者:** Yinuo Wang; Gavin Tao
>
> **备注:** 14 pages. This paper has been published in Advanced Engineering Informatics. Please cite the journal version: DOI: 10.1016/j.aei.2025.104230
>
> **摘要:** We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget.
>
---
#### [replaced 005] MSG-Loc: Multi-Label Likelihood-based Semantic Graph Matching for Object-Level Global Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向机器人全局定位任务，解决语义模糊导致的对象误匹配与位姿估计误差问题。提出MSG-Loc框架：构建多标签语义图，通过上下文感知的似然传播增强节点匹配，支持开闭集检测与大词汇量场景。**

- **链接: [https://arxiv.org/pdf/2512.03522v2](https://arxiv.org/pdf/2512.03522v2)**

> **作者:** Gihyeon Lee; Jungwoo Lee; Juwon Kim; Young-Sik Shin; Younggun Cho
>
> **备注:** Accepted in IEEE Robotics and Automation Letters (2025)
>
> **摘要:** Robots are often required to localize in environments with unknown object classes and semantic ambiguity. However, when performing global localization using semantic objects, high semantic ambiguity intensifies object misclassification and increases the likelihood of incorrect associations, which in turn can cause significant errors in the estimated pose. Thus, in this letter, we propose a multi-label likelihood-based semantic graph matching framework for object-level global localization. The key idea is to exploit multi-label graph representations, rather than single-label alternatives, to capture and leverage the inherent semantic context of object observations. Based on these representations, our approach enhances semantic correspondence across graphs by combining the likelihood of each node with the maximum likelihood of its neighbors via context-aware likelihood propagation. For rigorous validation, data association and pose estimation performance are evaluated under both closed-set and open-set detection configurations. In addition, we demonstrate the scalability of our approach to large-vocabulary object categories in both real-world indoor scenes and synthetic environments.
>
---
#### [replaced 006] RoboCOIN: An Open-Sourced Bimanual Robotic Data COllection for INtegrated Manipulation
- **分类: cs.RO**

- **简介: 该论文面向机器人双臂协同操作任务，解决多平台硬件异构导致的大规模双臂数据稀缺问题。提出开源数据集RoboCOIN（含15种平台、18万+演示、16场景）及处理框架CoRobot，引入分层能力金字塔标注与RTML语言，支持跨平台双臂学习。**

- **链接: [https://arxiv.org/pdf/2511.17441v2](https://arxiv.org/pdf/2511.17441v2)**

> **作者:** Shihan Wu; Xuecheng Liu; Shaoxuan Xie; Pengwei Wang; Xinghang Li; Bowen Yang; Zhe Li; Kai Zhu; Hongyu Wu; Yiheng Liu; Zhaoye Long; Yue Wang; Chong Liu; Dihan Wang; Ziqiang Ni; Xiang Yang; You Liu; Ruoxuan Feng; Runtian Xu; Lei Zhang; Denghang Huang; Chenghao Jin; Anlan Yin; Xinlong Wang; Zhenguo Sun; Junkai Zhao; Mengfei Du; Mingyu Cao; Xiansheng Chen; Hongyang Cheng; Xiaojie Zhang; Yankai Fu; Ning Chen; Cheng Chi; Sixiang Chen; Huaihai Lyu; Xiaoshuai Hao; Yequan Wang; Bo Lei; Dong Liu; Xi Yang; Yance Jiao; Tengfei Pan; Yunyan Zhang; Songjing Wang; Ziqian Zhang; Xu Liu; Ji Zhang; Caowei Meng; Zhizheng Zhang; Jiyang Gao; Song Wang; Xiaokun Leng; Zhiqiang Xie; Zhenzhen Zhou; Peng Huang; Wu Yang; Yandong Guo; Yichao Zhu; Suibing Zheng; Hao Cheng; Xinmin Ding; Yang Yue; Huanqian Wang; Chi Chen; Jingrui Pang; YuXi Qian; Haoran Geng; Lianli Gao; Haiyuan Li; Bin Fang; Gao Huang; Yaodong Yang; Hao Dong; He Wang; Hang Zhao; Yadong Mu; Di Hu; Hao Zhao; Tiejun Huang; Shanghang Zhang; Yonghua Lin; Zhongyuan Wang; Guocai Yao
>
> **备注:** Fixed typos
>
> **摘要:** Bimanual manipulation is essential for achieving human-like dexterity in robots, but the large-scale and diverse bimanual robot datasets remain scarce due to hardware heterogeneity across robotic platforms. To address the challenge, we present RoboCOIN, a comprehensive multi-embodiment bimanual manipulation dataset with over 180,000 demonstrations collected from 15 distinct robotic platforms. The dataset covers 16 scenarios, including residential, commercial, and working environments, with 421 tasks systematically organized by bimanual coordination patterns and object properties. Our key innovation is a hierarchical capability pyramid that provides multi-level annotations, spanning trajectory-level concepts, segment-level subtasks, and frame-level kinematics. We further develop CoRobot, a comprehensive processing framework featuring Robot Trajectory Markup Language (RTML) for quality assessment, automated annotation generation, and unified multi-embodiment management. Extensive experiments demonstrate the reliability and effectiveness of RoboCOIN in multi-embodiment bimanual learning, with significant performance improvements across various model architectures and robotic platforms. The complete dataset and framework are open-sourced and publicly available for further research purposes. Project website: https://FlagOpen.github.io/RoboCOIN/.
>
---
#### [replaced 007] SoMi-ToM: Evaluating Multi-Perspective Theory of Mind in Embodied Social Interactions
- **分类: cs.CL; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出SoMi-ToM基准，旨在评估大视觉语言模型在具身多智能体复杂社交互动中的多视角心理理论（ToM）能力。针对现有ToM评测局限于静态文本、脱离真实交互的问题，构建了含视频、图像与标注题目的多模态数据集，并开展第一/第三人称双视角评测，揭示当前LVLMs与人类存在显著性能差距。**

- **链接: [https://arxiv.org/pdf/2506.23046v3](https://arxiv.org/pdf/2506.23046v3)**

> **作者:** Xianzhe Fan; Xuhui Zhou; Chuanyang Jin; Kolby Nottingham; Hao Zhu; Maarten Sap
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** Humans continuously infer the states, goals, and behaviors of others by perceiving their surroundings in dynamic, real-world social interactions. However, most Theory of Mind (ToM) benchmarks only evaluate static, text-based scenarios, which have a significant gap compared to real interactions. We propose the SoMi-ToM benchmark, designed to evaluate multi-perspective ToM in embodied multi-agent complex social interactions. This benchmark is based on rich multimodal interaction data generated by the interaction environment SoMi, covering diverse crafting goals and social relationships. Our framework supports multi-level evaluation: (1) first-person evaluation provides multimodal (visual, dialogue, action, etc.) input from a first-person perspective during a task for real-time state inference, (2) third-person evaluation provides complete third-person perspective video and text records after a task for goal and behavior inference. This evaluation method allows for a more comprehensive examination of a model's ToM capabilities from both the subjective immediate experience and the objective global observation. We constructed a challenging dataset containing 35 third-person perspective videos, 363 first-person perspective images, and 1225 expert-annotated multiple-choice questions (three options). On this dataset, we systematically evaluated the performance of human subjects and several state-of-the-art large vision-language models (LVLMs). The results show that LVLMs perform significantly worse than humans on SoMi-ToM: the average accuracy gap between humans and models is 40.1% in first-person evaluation and 26.4% in third-person evaluation. This indicates that future LVLMs need to further improve their ToM capabilities in embodied, complex social interactions.
>
---
#### [replaced 008] Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人学习任务，旨在解决世界模型在长程预测、误差累积和sim-to-real迁移中的鲁棒性问题。提出双自回归神经网络世界模型与自监督训练方法，并构建基于想象环境的策略优化框架，提升真实场景适应性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2501.10100v5](https://arxiv.org/pdf/2501.10100v5)**

> **作者:** Chenhao Li; Andreas Krause; Marco Hutter
>
> **摘要:** Learning robust and generalizable world models is crucial for enabling efficient and scalable robotic control in real-world environments. In this work, we introduce a novel framework for learning world models that accurately capture complex, partially observable, and stochastic dynamics. The proposed method employs a dual-autoregressive mechanism and self-supervised training to achieve reliable long-horizon predictions without relying on domain-specific inductive biases, ensuring adaptability across diverse robotic tasks. We further propose a policy optimization framework that leverages world models for efficient training in imagined environments and seamless deployment in real-world systems. This work advances model-based reinforcement learning by addressing the challenges of long-horizon prediction, error accumulation, and sim-to-real transfer. By providing a scalable and robust framework, the introduced methods pave the way for adaptive and efficient robotic systems in real-world applications.
>
---
#### [replaced 009] HACTS: a Human-As-Copilot Teleoperation System for Robot Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出HACTS系统，属机器人遥操作与学习任务，旨在解决现有系统单向控制、缺乏实时状态同步的问题。作者设计了低成本、双向实时同步的“人作为副驾驶”遥操作硬件，支持无缝干预与动作-修正数据采集，提升模仿学习与强化学习性能。**

- **链接: [https://arxiv.org/pdf/2503.24070v2](https://arxiv.org/pdf/2503.24070v2)**

> **作者:** Zhiyuan Xu; Yinuo Zhao; Kun Wu; Ning Liu; Junjie Ji; Zhengping Che; Chi Harold Liu; Jian Tang
>
> **摘要:** Teleoperation is essential for autonomous robot learning, especially in manipulation tasks that require human demonstrations or corrections. However, most existing systems only offer unilateral robot control and lack the ability to synchronize the robot's status with the teleoperation hardware, preventing real-time, flexible intervention. In this work, we introduce HACTS (Human-As-Copilot Teleoperation System), a novel system that establishes bilateral, real-time joint synchronization between a robot arm and teleoperation hardware. This simple yet effective feedback mechanism, akin to a steering wheel in autonomous vehicles, enables the human copilot to intervene seamlessly while collecting action-correction data for future learning. Implemented using 3D-printed components and low-cost, off-the-shelf motors, HACTS is both accessible and scalable. Our experiments show that HACTS significantly enhances performance in imitation learning (IL) and reinforcement learning (RL) tasks, boosting IL recovery capabilities and data efficiency, and facilitating human-in-the-loop RL. HACTS paves the way for more effective and interactive human-robot collaboration and data-collection, advancing the capabilities of robot manipulation.
>
---
#### [replaced 010] Entropy-Controlled Intrinsic Motivation Reinforcement Learning for Quadruped Robot Locomotion in Complex Terrains
- **分类: cs.RO**

- **简介: 该论文属机器人强化学习任务，旨在解决四足机器人在复杂地形中策略早熟收敛、运动不稳与能耗高的问题。提出熵控内在动机（ECIM）算法，融合熵调节与内在激励以增强探索，显著提升稳定性并降低关节加速度、扭矩及能耗。**

- **链接: [https://arxiv.org/pdf/2512.06486v2](https://arxiv.org/pdf/2512.06486v2)**

> **作者:** Wanru Gong; Xinyi Zheng; Yuan Hui; Zhongjun Li; Weiqiang Wang; Xiaoqing Zhu
>
> **摘要:** Learning is the basis of both biological and artificial systems when it comes to mimicking intelligent behaviors. From the classical PPO (Proximal Policy Optimization), there is a series of deep reinforcement learning algorithms which are widely used in training locomotion policies for quadrupedal robots because of their stability and sample efficiency. However, among all these variants, experiments and simulations often converge prematurely, leading to suboptimal locomotion and reduced task performance. Therefore, in this paper, we introduce Entropy-Controlled Intrinsic Motivation (ECIM), an entropy-based reinforcement learning algorithm in contrast with the PPO series, that can reduce premature convergence by combining intrinsic motivation with adaptive exploration. For experiments, in order to parallel with other baselines, we chose to apply it in Isaac Gym across six terrain categories: upward slopes, downward slopes, uneven rough terrain, ascending stairs, descending stairs, and flat ground as widely used. For comparison, our experiments consistently achieve better performance: task rewards increase by 4--12%, peak body pitch oscillation is reduced by 23--29%, joint acceleration decreases by 20--32%, and joint torque consumption declines by 11--20%. Overall, our model ECIM, by combining entropy control and intrinsic motivation control, achieves better results in stability across different terrains for quadrupedal locomotion, and at the same time reduces energetic cost and makes it a practical choice for complex robotic control tasks.
>
---
#### [replaced 011] Human-Like Robot Impedance Regulation Skill Learning from Human-Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文属人机协作任务，旨在让机器人学习人类般的阻抗调节技能。针对物理协作中适应性差的问题，提出HIImpRSL框架：融合EMG与运动数据，通过模仿学习构建端点阻抗与轨迹联合表征，用LSTM学习策略，并由全身阻抗控制器实时调节。实验验证其在多任务中优于现有方法。**

- **链接: [https://arxiv.org/pdf/2502.13707v2](https://arxiv.org/pdf/2502.13707v2)**

> **作者:** Chenzui Li; Xi Wu; Yiming Chen; Tao Teng; Xuefeng Zhang; Sylvain Calinon; Darwin Caldwell; Fei Chen
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Humans are experts in physical collaboration by leveraging cognitive abilities such as perception, reasoning, and decision-making to regulate compliance behaviors based on their partners' states and task requirements. Equipping robots with similar cognitive-inspired collaboration skills can significantly enhance the efficiency and adaptability of human-robot collaboration (HRC). This paper introduces an innovative HumanInspired Impedance Regulation Skill Learning framework (HIImpRSL) for robotic systems to achieve leader-follower and mutual adaptation in multiple physical collaborative tasks. The proposed framework enables the robot to adapt its compliance based on human states and reference trajectories derived from human-human demonstrations. By integrating electromyography (EMG) signals and motion data, we extract endpoint impedance profiles and reference trajectories to construct a joint representation via imitation learning. An LSTM-based module then learns task-oriented impedance regulation policies, which are implemented through a whole-body impedance controller for online impedance adaptation. Experimental validation was conducted through collaborative transportation, two interactive Tai Chi pushing hands, and collaborative sawing tasks with multiple human subjects, demonstrating the ability of our framework to achieve human-like collaboration skills and the superior performance from the perspective of interactive forces compared to four other related methods.
>
---
#### [replaced 012] MDE-AgriVLN: Agricultural Vision-and-Language Navigation with Monocular Depth Estimation
- **分类: cs.RO**

- **简介: 该论文面向农业视觉-语言导航（AgriVLN）任务，解决单目相机导致的深度感知不足问题。提出MDE-AgriVLN方法，引入单目深度估计（MDE）模块生成深度特征，增强多模态推理能力，在A2A基准上显著提升成功率与导航精度。**

- **链接: [https://arxiv.org/pdf/2512.03958v2](https://arxiv.org/pdf/2512.03958v2)**

> **作者:** Xiaobei Zhao; Xingqi Lyu; Xin Chen; Xiang Li
>
> **摘要:** Agricultural robots are serving as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily relying on manual operations or railway systems for movement. The AgriVLN method and the A2A benchmark pioneeringly extended Vision-and-Language Navigation (VLN) to the agricultural domain, enabling a robot to navigate to a target position following a natural language instruction. Unlike human binocular vision, most agricultural robots are only given a single camera for monocular vision, which results in limited spatial perception. To bridge this gap, we present the method of Agricultural Vision-and-Language Navigation with Monocular Depth Estimation (MDE-AgriVLN), in which we propose the MDE module generating depth features from RGB images, to assist the decision-maker on multimodal reasoning. When evaluated on the A2A benchmark, our MDE-AgriVLN method successfully increases Success Rate from 0.23 to 0.32 and decreases Navigation Error from 4.43m to 4.08m, demonstrating the state-of-the-art performance in the agricultural VLN domain. Code: https://github.com/AlexTraveling/MDE-AgriVLN.
>
---
#### [replaced 013] DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DualMap，面向动态场景下的自然语言导航任务，解决在线开放词汇语义建图与实时环境更新难题。通过混合分割前端和物体状态检测，避免3D物体融合；采用全局抽象+局部具体双地图结构，实现高效、自适应的语义映射与语言引导导航。**

- **链接: [https://arxiv.org/pdf/2506.01950v4](https://arxiv.org/pdf/2506.01950v4)**

> **作者:** Jiajun Jiang; Yiming Zhu; Zirui Wu; Jie Song
>
> **备注:** 14 pages, 14 figures. Published in IEEE Robotics and Automation Letters (RA-L), 2025. Code: https://github.com/Eku127/DualMap Project page: https://eku127.github.io/DualMap/
>
> **摘要:** We introduce DualMap, an online open-vocabulary mapping system that enables robots to understand and navigate dynamically changing environments through natural language queries. Designed for efficient semantic mapping and adaptability to changing environments, DualMap meets the essential requirements for real-world robot navigation applications. Our proposed hybrid segmentation frontend and object-level status check eliminate the costly 3D object merging required by prior methods, enabling efficient online scene mapping. The dual-map representation combines a global abstract map for high-level candidate selection with a local concrete map for precise goal-reaching, effectively managing and updating dynamic changes in the environment. Through extensive experiments in both simulation and real-world scenarios, we demonstrate state-of-the-art performance in 3D open-vocabulary segmentation, efficient scene mapping, and online language-guided navigation. Project page: https://eku127.github.io/DualMap/
>
---
#### [replaced 014] ESPADA: Execution Speedup via Semantics Aware Demonstration Data Downsampling for Imitation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文面向模仿学习中的行为克隆任务，旨在解决人类演示动作缓慢导致机器人执行效率低的问题。提出ESPADA框架，利用VLM-LLM识别语义关键段，结合DTW传播标签，实现语义感知的演示数据降采样，在不重训练、不改模型前提下提升执行速度约2倍，同时保持成功率。**

- **链接: [https://arxiv.org/pdf/2512.07371v2](https://arxiv.org/pdf/2512.07371v2)**

> **作者:** Byungju Kim; Jinu Pahk; Chungwoo Lee; Jaejoon Kim; Jangha Lee; Theo Taeyeong Kim; Kyuhwan Shim; Jun Ki Lee; Byoung-Tak Zhang
>
> **备注:** project page: https://project-espada.github.io/espada/
>
> **摘要:** Behavior-cloning based visuomotor policies enable precise manipulation but often inherit the slow, cautious tempo of human demonstrations, limiting practical deployment. However, prior studies on acceleration methods mainly rely on statistical or heuristic cues that ignore task semantics and can fail across diverse manipulation settings. We present ESPADA, a semantic and spatially aware framework that segments demonstrations using a VLM-LLM pipeline with 3D gripper-object relations, enabling aggressive downsampling only in non-critical segments while preserving precision-critical phases, without requiring extra data or architectural modifications, or any form of retraining. To scale from a single annotated episode to the full dataset, ESPADA propagates segment labels via Dynamic Time Warping (DTW) on dynamics-only features. Across both simulation and real-world experiments with ACT and DP baselines, ESPADA achieves approximately a 2x speed-up while maintaining success rates, narrowing the gap between human demonstrations and efficient robot control.
>
---
#### [replaced 015] Learning Obstacle Avoidance using Double DQN for Quadcopter Navigation
- **分类: cs.RO**

- **简介: 该论文属于强化学习与无人机导航交叉任务，旨在解决城市环境中GPS精度下降、空间狭窄及动态障碍导致的避障难题。作者提出基于Double DQN算法，利用机载深度相机数据训练虚拟四旋翼无人机在仿真城市环境中自主避障导航。**

- **链接: [https://arxiv.org/pdf/2509.18734v2](https://arxiv.org/pdf/2509.18734v2)**

> **作者:** Nishant Doshi; Amey Sutavani; Sanket Gujar
>
> **备注:** Fixed typo in second author's name throughout the paper
>
> **摘要:** One of the challenges faced by Autonomous Aerial Vehicles is reliable navigation through urban environments. Factors like reduction in precision of Global Positioning System (GPS), narrow spaces and dynamically moving obstacles make the path planning of an aerial robot a complicated task. One of the skills required for the agent to effectively navigate through such an environment is to develop an ability to avoid collisions using information from onboard depth sensors. In this paper, we propose Reinforcement Learning of a virtual quadcopter robot agent equipped with a Depth Camera to navigate through a simulated urban environment.
>
---
#### [replaced 016] STITCHER: Real-Time Trajectory Planning with Motion Primitive Search
- **分类: cs.RO**

- **简介: 该论文属于自主导航中的实时轨迹规划任务，旨在解决高速复杂环境下优化方法计算慢、不稳定的问题。提出无优化的STITCHER框架，通过图搜索拼接运动基元实现实时、安全、长程轨迹生成。**

- **链接: [https://arxiv.org/pdf/2412.21180v3](https://arxiv.org/pdf/2412.21180v3)**

> **作者:** Helene J. Levy; Brett T. Lopez
>
> **摘要:** Autonomous high-speed navigation through large, complex environments requires real-time generation of agile trajectories that are dynamically feasible, collision-free, and satisfy constraints. Most modern trajectory planning techniques rely on numerical optimization because high-quality, expressive trajectories that satisfy constraints can be systematically computed. However, strict requirements on computation time and the risk of numerical instability can limit the use of optimization-based planners in safety-critical situations. This work presents an optimization-free planning framework called STITCHER that leverages graph search to generate long-range trajectories by stitching short trajectory segments together in real time. STITCHER is shown to outperform modern optimization-based planners through its innovative planning architecture and several algorithmic developments that make real-time planning possible. Simulation results show safe trajectories through complex environments can be generated in milliseconds that cover tens of meters.
>
---
#### [replaced 017] DRO-EDL-MPC: Evidential Deep Learning-Based Distributionally Robust Model Predictive Control for Safe Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属自动驾驶安全控制任务，解决神经感知不确定性导致的运动规划风险问题。提出DRO-EDL-MPC算法：融合证据深度学习量化感知不确定性，构建置信自适应的分布鲁棒优化框架，并嵌入MPC实现安全、高效、可计算的实时控制。**

- **链接: [https://arxiv.org/pdf/2507.05710v2](https://arxiv.org/pdf/2507.05710v2)**

> **作者:** Hyeongchan Ham; Heejin Ahn
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Safety is a critical concern in motion planning for autonomous vehicles. Modern autonomous vehicles rely on neural network-based perception, but making control decisions based on these inference results poses significant safety risks due to inherent uncertainties. To address this challenge, we present a distributionally robust optimization (DRO) framework that accounts for both aleatoric and epistemic perception uncertainties using evidential deep learning (EDL). Our approach introduces a novel ambiguity set formulation based on evidential distributions that dynamically adjusts the conservativeness according to perception confidence levels. We integrate this uncertainty-aware constraint into model predictive control (MPC), proposing the DRO-EDL-MPC algorithm with computational tractability for autonomous driving applications. Validation in the CARLA simulator demonstrates that our approach maintains efficiency under high perception confidence while enforcing conservative constraints under low confidence.
>
---
#### [replaced 018] Accelerated Rotation-Invariant Convolution for UAV Image Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向无人机图像分割任务，解决传统卷积缺乏旋转不变性导致精度下降的问题。提出GPU优化的旋转不变卷积框架，跳过im2col、利用对称滤波器共享数据，显著降低计算与内存开销，提升速度与能效，同时保持高精度。**

- **链接: [https://arxiv.org/pdf/2512.08888v2](https://arxiv.org/pdf/2512.08888v2)**

> **作者:** Manduhu Manduhu; Alexander Dow; Gerard Dooly; James Riordan
>
> **摘要:** Rotation invariance is essential for precise, object-level segmentation in UAV aerial imagery, where targets can have arbitrary orientations and exhibit fine-scale details. Conventional segmentation architectures like U-Net rely on convolution operators that are not rotation-invariant, leading to degraded segmentation accuracy across varying viewpoints. Rotation invariance can be achieved by expanding the filter bank across multiple orientations; however, this will significantly increase computational cost and memory traffic. In this paper, we introduce a GPU-optimized rotation-invariant convolution framework that eliminates the traditional data-lowering (im2col) step required for matrix-multiplication-based convolution. By exploiting structured data sharing among symmetrically rotated filters, our method achieves multi-orientation convolution with greatly reduced memory traffic and computational redundancy. We further generalize the approach to accelerate convolution with arbitrary (non-symmetric) rotation angles. Across extensive benchmarks, the proposed convolution achieves 20--55% faster training and 15--45% lower energy consumption than CUDNN, while maintaining accuracy comparable to state-of-the-art rotation-invariant methods. In the eight-orientation setting, our approach achieves up to 45% speedup and 41% energy savings on 256\(\times\)256 inputs, and 32% speedup and 23% lower energy usage on 1024\(\times\)1024 inputs. Integrated into a U-Net segmentation model, the framework yields up to 6% improvement in accuracy over the non-rotation-aware baseline. These results demonstrate that the proposed method provides an effective and highly efficient alternative to existing rotation-invariant CNN frameworks.
>
---
#### [replaced 019] BEASST: Behavioral Entropic Gradient based Adaptive Source Seeking for Mobile Robots
- **分类: cs.RO**

- **简介: 该论文提出BEASST框架，面向移动机器人源定位任务，解决未知复杂环境中高效平衡探索与开发的问题。通过行为熵与概率加权建模信号不确定性，自适应调整风险偏好，实现智能导航，并提供理论收敛性与实验验证。**

- **链接: [https://arxiv.org/pdf/2508.10363v2](https://arxiv.org/pdf/2508.10363v2)**

> **作者:** Donipolo Ghimire; Aamodh Suresh; Carlos Nieto-Granda; Solmaz S. Kia
>
> **摘要:** This paper presents BEASST (Behavioral Entropic Gradient-based Adaptive Source Seeking for Mobile Robots), a novel framework for robotic source seeking in complex, unknown environments. Our approach enables mobile robots to efficiently balance exploration and exploitation by modeling normalized signal strength as a surrogate probability of source location. Building on Behavioral Entropy(BE) with Prelec's probability weighting function, we define an objective function that adapts robot behavior from risk-averse to risk-seeking based on signal reliability and mission urgency. The framework provides theoretical convergence guarantees under unimodal signal assumptions and practical stability under bounded disturbances. Experimental validation across DARPA SubT and multi-room scenarios demonstrates that BEASST consistently outperforms state-of-the-art methods, achieving 15% reduction in path length and 20% faster source localization through intelligent uncertainty-driven navigation that dynamically transitions between aggressive pursuit and cautious exploration.
>
---
#### [replaced 020] Active 6D Pose Estimation for Textureless Objects using Multi-View RGB Frames
- **分类: cs.CV; cs.RO**

- **简介: 该论文属6D位姿估计任务，旨在解决纹理缺失物体在RGB图像中因外观模糊、对称性与遮挡导致的单视图估计不准问题。提出两步解耦框架：先估3D平移以消深度歧义，再用模板匹配估旋转；并引入主动感知策略预测最优下一视角，提升精度、减少视角数。**

- **链接: [https://arxiv.org/pdf/2503.03726v2](https://arxiv.org/pdf/2503.03726v2)**

> **作者:** Jun Yang; Wenjie Xue; Sahar Ghavidel; Steven L. Waslander
>
> **摘要:** Estimating the 6D pose of textureless objects from RGB images is an important problem in robotics. Due to appearance ambiguities, rotational symmetries, and severe occlusions, single-view based 6D pose estimators are still unable to handle a wide range of objects, motivating research towards multi-view pose estimation and next-best-view prediction that addresses these limitations. In this work, we propose a comprehensive active perception framework for estimating the 6D poses of textureless objects using only RGB images. Our approach is built upon a key idea: decoupling the 6D pose estimation into a two-step sequential process can greatly improve both accuracy and efficiency. First, we estimate the 3D translation of each object, resolving scale and depth ambiguities inherent to RGB images. These estimates are then used to simplify the subsequent task of determining the 3D orientation, which we achieve through canonical scale template matching. Building on this formulation, we then introduce an active perception strategy that predicts the next best camera viewpoint to capture an RGB image, effectively reducing object pose uncertainty and enhancing pose accuracy. We evaluate our method on the public ROBI and TOD datasets, as well as on our reconstructed transparent object dataset, T-ROBI. Under the same camera viewpoints, our multi-view pose estimation significantly outperforms state-of-the-art approaches. Furthermore, by leveraging our next-best-view strategy, our approach achieves high pose accuracy with fewer viewpoints than heuristic-based policies across all evaluated datasets. The accompanying video and T-ROBI dataset will be released on our project page: https://trailab.github.io/ActiveODPE.
>
---
#### [replaced 021] WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文面向人形机器人全身位姿-操作协同控制任务，解决现有方法缺乏操作感知式运动规划、导致工作空间受限的问题。提出统一隐式VLA框架，利用无动作视频学习，并设计专用RL策略提升运动精度与稳定性，实现大空间全身位姿-操作控制。**

- **链接: [https://arxiv.org/pdf/2512.11047v2](https://arxiv.org/pdf/2512.11047v2)**

> **作者:** Haoran Jiang; Jin Chen; Qingwen Bu; Li Chen; Modi Shi; Yanjie Zhang; Delong Li; Chuanzhe Suo; Chuang Wang; Zhihui Peng; Hongyang Li
>
> **摘要:** Humanoid robots require precise locomotion and dexterous manipulation to perform challenging loco-manipulation tasks. Yet existing approaches, modular or end-to-end, are deficient in manipulation-aware locomotion. This confines the robot to a limited workspace, preventing it from performing large-space loco-manipulation. We attribute this to: (1) the challenge of acquiring loco-manipulation knowledge due to the scarcity of humanoid teleoperation data, and (2) the difficulty of faithfully and reliably executing locomotion commands, stemming from the limited precision and stability of existing RL controllers. To acquire richer loco-manipulation knowledge, we propose a unified latent learning framework that enables Vision-Language-Action (VLA) system to learn from low-cost action-free egocentric videos. Moreover, an efficient human data collection pipeline is devised to augment the dataset and scale the benefits. To execute the desired locomotion commands more precisely, we present a loco-manipulation-oriented (LMO) RL policy specifically tailored for accurate and stable core loco-manipulation movements, such as advancing, turning, and squatting. Building on these components, we introduce WholeBodyVLA, a unified framework for humanoid loco-manipulation. To the best of our knowledge, WholeBodyVLA is one of its kind enabling large-space humanoid loco-manipulation. It is verified via comprehensive experiments on the AgiBot X2 humanoid, outperforming prior baseline by 21.3%. It also demonstrates strong generalization and high extensibility across a broad range of tasks.
>
---
#### [replaced 022] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文面向具身AI的视觉运动策略学习任务，旨在解决现有生成式策略数据效率低、推理采样慢的问题。作者提出EfficientFlow框架：引入等变性提升数据效率，设计加速度正则化 surrogate 损失加速采样，在少样本下实现高性能与快速推理。**

- **链接: [https://arxiv.org/pdf/2512.02020v2](https://arxiv.org/pdf/2512.02020v2)**

> **作者:** Jianlei Chang; Ruofeng Mei; Wei Ke; Xiangyu Xu
>
> **备注:** Accepted by AAAI 2026. Project Page: https://efficientflow.github.io/
>
> **摘要:** Generative modeling has recently shown remarkable promise for visuomotor policy learning, enabling flexible and expressive control across diverse embodied AI tasks. However, existing generative policies often struggle with data inefficiency, requiring large-scale demonstrations, and sampling inefficiency, incurring slow action generation during inference. We introduce EfficientFlow, a unified framework for efficient embodied AI with flow-based policy learning. To enhance data efficiency, we bring equivariance into flow matching. We theoretically prove that when using an isotropic Gaussian prior and an equivariant velocity prediction network, the resulting action distribution remains equivariant, leading to improved generalization and substantially reduced data demands. To accelerate sampling, we propose a novel acceleration regularization strategy. As direct computation of acceleration is intractable for marginal flow trajectories, we derive a novel surrogate loss that enables stable and scalable training using only conditional trajectories. Across a wide range of robotic manipulation benchmarks, the proposed algorithm achieves competitive or superior performance under limited data while offering dramatically faster inference. These results highlight EfficientFlow as a powerful and efficient paradigm for high-performance embodied AI.
>
---
#### [replaced 023] Symphony: A Heuristic Normalized Calibrated Advantage Actor and Critic Algorithm in application for Humanoid Robots
- **分类: cs.RO; cs.NE**

- **简介: 该论文提出Symphony算法，面向人形机器人从零开始的强化学习任务，旨在解决样本效率低、动作不安全及训练不稳定问题。工作包括：引入“襁褓”正则化约束动作强度，设计带限幅参数噪声的确定性策略，提出Fading Replay Buffer与Temporal Advantage机制，实现高效、安全、稳定的端到端训练。**

- **链接: [https://arxiv.org/pdf/2512.10477v2](https://arxiv.org/pdf/2512.10477v2)**

> **作者:** Timur Ishuov; Michele Folgheraiter; Madi Nurmanov; Goncalo Gordo; Richárd Farkas; József Dombi
>
> **备注:** https://github.com/SuspensionRailway/symphony
>
> **摘要:** In our work we not explicitly hint that it is a misconception to think that humans learn fast. Learning process takes time. Babies start learning to move in the restricted liquid area called placenta. Children often are limited by underdeveloped body. Even adults are not allowed to participate in complex competitions right away. However, with robots, when learning from scratch, we often don't have the privilege of waiting for dozen millions of steps. "Swaddling" regularization is responsible for restraining an agent in rapid but unstable development penalizing action strength in a specific way not affecting actions directly. The Symphony, Transitional-policy Deterministic Actor and Critic algorithm, is a concise combination of different ideas for possibility of training humanoid robots from scratch with Sample Efficiency, Sample Proximity and Safety of Actions in mind. It is no secret that continuous increase in Gaussian noise without appropriate smoothing is harmful for motors and gearboxes. Compared to Stochastic algorithms, we set a limited parametric noise and promote a reduced strength of actions, safely increasing entropy, since the actions are kind of immersed in weaker noise. When actions require more extreme values, actions rise above the weak noise. Training becomes empirically much safer for both the environment around and the robot's mechanisms. We use Fading Replay Buffer: using a fixed formula containing the hyperbolic tangent, we adjust the batch sampling probability: the memory contains a recent memory and a long-term memory trail. Fading Replay Buffer allows us to use Temporal Advantage when we improve the current Critic Network prediction compared to the exponential moving average. Temporal Advantage allows us to update Actor and Critic in one pass, as well as combine Actor and Critic in one Object and implement their Losses in one line.
>
---
#### [replaced 024] Camera Calibration via Circular Patterns: A Comprehensive Framework with Detection Uncertainty and Unbiased Projection Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文属相机标定任务，旨在解决圆环图案中心投影模型在镜头畸变下存在偏差、导致精度低的问题。提出无偏投影模型与基于形状分布的中心点不确定性建模方法，提升标定精度与鲁棒性，并提供实用校准指南。**

- **链接: [https://arxiv.org/pdf/2506.16842v2](https://arxiv.org/pdf/2506.16842v2)**

> **作者:** Chaehyeon Song; Dongjae Lee; Jongwoo Lim; Ayoung Kim
>
> **摘要:** Camera calibration using planar targets has been widely favored, and two types of control points have been mainly considered as measurements: the corners of the checkerboard and the centroid of circles. Since a centroid is derived from numerous pixels, the circular pattern provides more precise measurements than the checkerboard. However, the existing projection model of circle centroids is biased under lens distortion, resulting in low performance. To surmount this limitation, we propose an unbiased projection model of the circular pattern and demonstrate its superior accuracy compared to the checkerboard. Complementing this, we introduce uncertainty into circular patterns to enhance calibration robustness and completeness. Defining centroid uncertainty improves the performance of calibration components, including pattern detection, optimization, and evaluation metrics. We also provide guidelines for performing good camera calibration based on the evaluation metric. The core concept of this approach is to model the boundary points of a two-dimensional shape as a Markov random field, considering its connectivity. The shape distribution is propagated to the centroid uncertainty through an appropriate shape representation based on the Green theorem. Consequently, the resulting framework achieves marked gains in calibration accuracy and robustness. The complete source code and demonstration video are available at https://github.com/chaehyeonsong/discocal.
>
---
#### [replaced 025] A Survey of Behavior Foundation Model: Next-Generation Whole-Body Control System of Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文是综述任务，旨在解决人形机器人全身控制（WBC）中传统学习型控制器泛化差、重训练成本高的问题。作者系统梳理行为基础模型（BFMs）的发展脉络、预训练范式、应用与挑战，并提供开源论文资源库。**

- **链接: [https://arxiv.org/pdf/2506.20487v4](https://arxiv.org/pdf/2506.20487v4)**

> **作者:** Mingqi Yuan; Tao Yu; Wenqi Ge; Xiuyong Yao; Huijiang Wang; Jiayu Chen; Bo Li; Wei Zhang; Wenjun Zeng; Hua Chen; Xin Jin
>
> **备注:** 19 pages, 10 figures
>
> **摘要:** Humanoid robots are drawing significant attention as versatile platforms for complex motor control, human-robot interaction, and general-purpose physical intelligence. However, achieving efficient whole-body control (WBC) in humanoids remains a fundamental challenge due to sophisticated dynamics, underactuation, and diverse task requirements. While learning-based controllers have shown promise for complex tasks, their reliance on labor-intensive and costly retraining for new scenarios limits real-world applicability. To address these limitations, behavior(al) foundation models (BFMs) have emerged as a new paradigm that leverages large-scale pre-training to learn reusable primitive skills and broad behavioral priors, enabling zero-shot or rapid adaptation to a wide range of downstream tasks. In this paper, we present a comprehensive overview of BFMs for humanoid WBC, tracing their development across diverse pre-training pipelines. Furthermore, we discuss real-world applications, current limitations, urgent challenges, and future opportunities, positioning BFMs as a key approach toward scalable and general-purpose humanoid intelligence. Finally, we provide a curated and regularly updated collection of BFM papers and projects to facilitate more subsequent research, which is available at https://github.com/yuanmingqi/awesome-bfm-papers.
>
---
#### [replaced 026] CT-UIO: Continuous-Time UWB-Inertial-Odometer Localization Using Non-Uniform B-spline with Fewer Anchors
- **分类: cs.RO**

- **简介: 该论文提出CT-UIO系统，解决少锚点UWB定位中多传感器异步融合与可观测性不足问题；采用非均匀B样条建模连续轨迹，结合自适应EKF融合IMU/里程计，并引入多假设虚拟锚点提升可观测性，显著提升定位精度。**

- **链接: [https://arxiv.org/pdf/2502.06287v2](https://arxiv.org/pdf/2502.06287v2)**

> **作者:** Jian Sun; Wei Sun; Genwei Zhang; Kailun Yang; Song Li; Xiangqi Meng; Na Deng; Chongbin Tan
>
> **备注:** Accepted to IEEE Transactions on Mobile Computing (TMC). The codebase and datasets will be open-sourced at https://github.com/JasonSun623/CT-UIO
>
> **摘要:** Ultra-wideband (UWB) based positioning with fewer anchors has attracted significant research interest in recent years, especially under energy-constrained conditions. However, most existing methods rely on discrete-time representations and smoothness priors to infer a robot's motion states, which often struggle with ensuring multi-sensor data synchronization. In this article, we present a continuous-time UWB-Inertial-Odometer localization system (CT-UIO), utilizing a non-uniform B-spline framework with fewer anchors. Unlike traditional uniform B-spline-based continuous-time methods, we introduce an adaptive knot-span adjustment strategy for non-uniform continuous-time trajectory representation. This is accomplished by adjusting control points dynamically based on movement speed. To enable efficient fusion of {inertial measurement unit (IMU) and odometer data, we propose an improved extended Kalman filter (EKF) with innovation-based adaptive estimation to provide short-term accurate motion prior. Furthermore, to address the challenge of achieving a fully observable UWB localization system under few-anchor conditions, the virtual anchor (VA) generation method based on multiple hypotheses is proposed. At the backend, we propose an adaptive sliding window strategy for global trajectory estimation. Comprehensive experiments are conducted on three self-collected datasets with different UWB anchor numbers and motion modes. The result shows that the proposed CT-UIO achieves 0.403m, 0.150m, and 0.189m localization accuracy in corridor, exhibition hall, and office environments, yielding 17.2%, 26.1%, and 15.2% improvements compared with competing state-of-the-art UIO systems, respectively. The codebase and datasets of this work will be open-sourced at https://github.com/JasonSun623/CT-UIO.
>
---
#### [replaced 027] STITCHER: Constrained Trajectory Planning in Complex Environments with Real-Time Motion Primitive Search
- **分类: cs.RO**

- **简介: 该论文提出STITCHER，一种无优化的实时轨迹规划方法，用于高速无人机在复杂环境中的安全导航。它通过图搜索拼接短轨迹段，满足动力学、避障及非凸执行器约束，显著提升实时性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.14893v3](https://arxiv.org/pdf/2510.14893v3)**

> **作者:** Helene J. Levy; Brett T. Lopez
>
> **摘要:** Autonomous high-speed navigation through large, complex environments requires real-time generation of agile trajectories that are dynamically feasible, collision-free, and satisfy state or actuator constraints. Modern trajectory planning techniques primarily use numerical optimization, as they enable the systematic computation of high-quality, expressive trajectories that satisfy various constraints. However, stringent requirements on computation time and the risk of numerical instability can limit the use of optimization-based planners in safety-critical scenarios. This work presents an optimization-free planning framework called STITCHER that stitches short trajectory segments together with graph search to compute long-range, expressive, and near-optimal trajectories in real-time. STITCHER outperforms modern optimization-based planners through our innovative planning architecture and several algorithmic developments that make real-time planning possible. Extensive simulation testing is performed to analyze the algorithmic components that make up STITCHER, along with a thorough comparison with two state-of-the-art optimization planners. Simulation tests show that safe trajectories can be created within a few milliseconds for paths that span the entirety of two 50 m x 50 m environments. Hardware tests with a custom quadrotor verify that STITCHER can produce trackable paths in real-time while respecting nonconvex constraints, such as limits on tilt angle and motor forces, which are otherwise hard to include in optimization-based planners.
>
---
