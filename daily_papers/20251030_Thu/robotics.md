# 机器人 cs.RO

- **最新发布 37 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Combining Moving Mass Actuators and Manoeuvring Models for Underwater Vehicles: A Lagrangian Approach
- **分类: cs.RO; cs.SY; eess.SY; 93C10 (Primary) 37N35, 93C95, 70B10, 70B15 (Secondary); I.6.3; I.6.4; I.6.5; J.2**

- **简介: 该论文研究水下航行器内部移动质量作动器的运动建模问题，提出基于拉格朗日方法的牛顿-欧拉动力学模型，将移动质量动态纳入航迹操纵模型，考虑科里奥利-向心效应与静水力学，通过仿真验证模型有效性。**

- **链接: [http://arxiv.org/pdf/2510.25479v1](http://arxiv.org/pdf/2510.25479v1)**

> **作者:** Alexander B. Rambech; Ivar B. Saksvik; Vahid Hassani
>
> **备注:** \c{opyright} 2025 Alexander Rambech, Ivar Saksvik and Vahid Hassani. Accepted by IFAC for publication under a Creative Commons License CC-BY-NC-ND
>
> **摘要:** In this paper, we present a Newton-Euler formulation of the equations of motion for underwater vehicles with an interntal moving mass actuator. Furthermore, the moving mass dynamics are expressed as an extension to the manoeuvring model for underwater vehicles, originally introduced by Fossen (1991). The influence of the moving mass is described in body-frame and included as states in both an additional kinematic equation and as part of the coupled rigid-body kinetics of the underwater vehicle. The Coriolis-centripetal effects are derived from Kirchhoff's equations and the hydrostatics are derived using first principals. The proposed Newton-Euler model is validated through simulation and compared with the traditional Hamiltonian internal moving mass actuator formulation.
>
---
#### [new 002] Geometric Robot Calibration Using a Calibration Plate
- **分类: cs.RO**

- **简介: 该论文提出一种基于标定板的机器人几何标定方法，旨在解决传统激光跟踪或动作捕捉系统成本高、便携性差的问题。通过设计具有精确已知点距的标定板，利用点间相对测量值，结合最小二乘与约束优化求解误差参数，实现对机器人几何误差的建模与识别，适用于多种类型机器人。**

- **链接: [http://arxiv.org/pdf/2510.25338v1](http://arxiv.org/pdf/2510.25338v1)**

> **作者:** Bernhard Rameder; Hubert Gattringer; Andreas Mueller
>
> **备注:** pp 309-317
>
> **摘要:** In this paper a new method for geometric robot calibration is introduced, which uses a calibration plate with precisely known distances between its measuring points. The relative measurement between two points on the calibration plate is used to determine predefined error parameters of the system. In comparison to conventional measurement methods, like laser tracker or motion capture systems, the calibration plate provides a more mechanically robust and cheaper alternative, which is furthermore easier to transport due to its small size. The calibration method, the plate design, the mathematical description of the error system as well as the identification of the parameters are described in detail. For identifying the error parameters, the least squares method and a constrained optimization problem are used. The functionality of this method was demonstrated in experiments that led to promising results, correlated with one of a laser tracker calibration. The modeling and identification of the error parameters is done for a gantry machine, but is not restricted to that type of robot.
>
---
#### [new 003] An approach for combining transparency and motion assistance of a lower body exoskeleton
- **分类: cs.RO**

- **简介: 该论文研究下肢外骨骼的步态辅助任务，旨在解决透明性与运动辅助难以兼顾的问题。通过利用传动间隙实现透明模式，结合自适应振荡器学习周期性运动信号，动态提供辅助扭矩，提升行走时的自然性和支持性。**

- **链接: [http://arxiv.org/pdf/2510.25335v1](http://arxiv.org/pdf/2510.25335v1)**

> **作者:** Jakob Ziegler; Bernhard Rameder; Hubert Gattringer; Andreas Mueller
>
> **备注:** 8 pages
>
> **摘要:** In this paper, an approach for gait assistance with a lower body exoskeleton is described. Two concepts, transparency and motion assistance, are combined. The transparent mode, where the system is following the user's free motion with a minimum of perceived interaction forces, is realized by exploiting the gear backlash of the actuation units. During walking a superimposed assistance mode applies an additional torque guiding the legs to their estimated future position. The concept of adaptive oscillators is utilized to learn the quasi-periodic signals typical for locomotion. First experiments showed promising results.
>
---
#### [new 004] Scalable predictive processing framework for multitask caregiving robots
- **分类: cs.RO; cs.AI; cs.LG; q-bio.NC**

- **简介: 该论文提出一种基于预测处理的可扩展框架，用于多任务护理机器人。针对现有系统任务专用、依赖人工特征工程的问题，构建了能直接处理高维感官输入的分层递归神经网络，实现无需特征工程的多任务学习，并在仿真中验证了其自组织动态、视觉退化鲁棒性及不对称干扰下的稳定性能。**

- **链接: [http://arxiv.org/pdf/2510.25053v1](http://arxiv.org/pdf/2510.25053v1)**

> **作者:** Hayato Idei; Tamon Miyake; Tetsuya Ogata; Yuichi Yamashita
>
> **摘要:** The rapid aging of societies is intensifying demand for autonomous care robots; however, most existing systems are task-specific and rely on handcrafted preprocessing, limiting their ability to generalize across diverse scenarios. A prevailing theory in cognitive neuroscience proposes that the human brain operates through hierarchical predictive processing, which underlies flexible cognition and behavior by integrating multimodal sensory signals. Inspired by this principle, we introduce a hierarchical multimodal recurrent neural network grounded in predictive processing under the free-energy principle, capable of directly integrating over 30,000-dimensional visuo-proprioceptive inputs without dimensionality reduction. The model was able to learn two representative caregiving tasks, rigid-body repositioning and flexible-towel wiping, without task-specific feature engineering. We demonstrate three key properties: (i) self-organization of hierarchical latent dynamics that regulate task transitions, capture variability in uncertainty, and infer occluded states; (ii) robustness to degraded vision through visuo-proprioceptive integration; and (iii) asymmetric interference in multitask learning, where the more variable wiping task had little influence on repositioning, whereas learning the repositioning task led to a modest reduction in wiping performance, while the model maintained overall robustness. Although the evaluation was limited to simulation, these results establish predictive processing as a universal and scalable computational principle, pointing toward robust, flexible, and autonomous caregiving robots while offering theoretical insight into the human brain's ability to achieve flexible adaptation in uncertain real-world environments.
>
---
#### [new 005] Learning Spatial-Aware Manipulation Ordering
- **分类: cs.RO**

- **简介: 该论文针对杂乱环境中的操作顺序问题，提出OrderMind框架，通过空间感知的图神经网络学习物体间空间依赖关系，生成合理操作顺序。利用空间先验标签指导视觉语言模型生成监督信号，提升真实场景下的操作效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.25138v1](http://arxiv.org/pdf/2510.25138v1)**

> **作者:** Yuxiang Yan; Zhiyuan Zhou; Xin Gao; Guanghao Li; Shenglin Li; Jiaqi Chen; Qunyan Pu; Jian Pu
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Manipulation in cluttered environments is challenging due to spatial dependencies among objects, where an improper manipulation order can cause collisions or blocked access. Existing approaches often overlook these spatial relationships, limiting their flexibility and scalability. To address these limitations, we propose OrderMind, a unified spatial-aware manipulation ordering framework that directly learns object manipulation priorities based on spatial context. Our architecture integrates a spatial context encoder with a temporal priority structuring module. We construct a spatial graph using k-Nearest Neighbors to aggregate geometric information from the local layout and encode both object-object and object-manipulator interactions to support accurate manipulation ordering in real-time. To generate physically and semantically plausible supervision signals, we introduce a spatial prior labeling method that guides a vision-language model to produce reasonable manipulation orders for distillation. We evaluate OrderMind on our Manipulation Ordering Benchmark, comprising 163,222 samples of varying difficulty. Extensive experiments in both simulation and real-world environments demonstrate that our method significantly outperforms prior approaches in effectiveness and efficiency, enabling robust manipulation in cluttered scenes.
>
---
#### [new 006] NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies
- **分类: cs.RO**

- **简介: 该论文针对资源受限边缘设备上视觉-语言-动作（VLA）模型部署难题，提出NanoVLA轻量级架构。通过视觉语言解耦、长短动作分块与动态路由策略，实现高精度、低延迟推理，显著提升效率与可部署性。**

- **链接: [http://arxiv.org/pdf/2510.25122v1](http://arxiv.org/pdf/2510.25122v1)**

> **作者:** Jiahong Chen; Jing Wang; Long Chen; Chuwei Cai; Jinghui Lu
>
> **摘要:** Vision-language-action (VLA) models have significantly advanced robotic manipulation by integrating vision-language models (VLMs), and action decoders into a unified architecture. However, their deployment on resource-constrained edge devices, such as mobile robots or embedded systems (e.g., Jetson Orin Nano), remains challenging due to high computational demands, especially in real-world scenarios where power, latency, and computational resources are critical. To close this gap, we introduce Nano-scale Vision-Language Action (NanoVLA), a family of lightweight VLA architectures that achieve high performance with minimal resources. Our core innovations include: (1) vision-language decoupling that moves conventional early vision and language inputs fusion in VLM to late stage, achieving better performance while enabling caching and reduce inference overhead and latency; (2) long-short action chunking to ensure smooth, coherent multi-step planning without sacrificing real-time responsiveness; (3) dynamic routing that adaptively assigns lightweight or heavy backbones based on task complexity, further optimizing inference efficiency. Experimental results on several benchmarks, as well as real-world deployments, demonstrate that NanoVLA achieves up to 52x faster inference on edge devices compared to previous state-of-the-art VLA models, with 98% less parameters while maintaining or surpassing their task accuracy and generalization. Ablation studies confirm that our decoupling strategy preserves cross-task transferability, and the routing module enhances cost-performance trade-offs, enabling practical, high-precision robotic manipulation on resource-constrained hardware.
>
---
#### [new 007] Learning to Plan & Schedule with Reinforcement-Learned Bimanual Robot Skills
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对长时序、高接触复杂度的双臂操作任务，提出一种融合强化学习与Transformer的分层规划调度框架。通过预训练单臂与双臂基础技能，利用Transformer实现多技能并行调度，解决传统顺序规划效率低、端到端强化学习难以协调的问题，显著提升任务成功率与行为协同性。**

- **链接: [http://arxiv.org/pdf/2510.25634v1](http://arxiv.org/pdf/2510.25634v1)**

> **作者:** Weikang Wan; Fabio Ramos; Xuning Yang; Caelan Garrett
>
> **摘要:** Long-horizon contact-rich bimanual manipulation presents a significant challenge, requiring complex coordination involving a mixture of parallel execution and sequential collaboration between arms. In this paper, we introduce a hierarchical framework that frames this challenge as an integrated skill planning & scheduling problem, going beyond purely sequential decision-making to support simultaneous skill invocation. Our approach is built upon a library of single-arm and bimanual primitive skills, each trained using Reinforcement Learning (RL) in GPU-accelerated simulation. We then train a Transformer-based planner on a dataset of skill compositions to act as a high-level scheduler, simultaneously predicting the discrete schedule of skills as well as their continuous parameters. We demonstrate that our method achieves higher success rates on complex, contact-rich tasks than end-to-end RL approaches and produces more efficient, coordinated behaviors than traditional sequential-only planners.
>
---
#### [new 008] A Humanoid Visual-Tactile-Action Dataset for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文针对接触丰富的机器人操作任务，提出一个包含视觉、触觉与动作的多模态数据集，用于软体物体操纵。旨在解决现有数据集对压力多样性覆盖不足的问题，通过人形机器人遥操作采集真实场景下的复杂触觉信号，推动更优优化策略的模型研究。**

- **链接: [http://arxiv.org/pdf/2510.25725v1](http://arxiv.org/pdf/2510.25725v1)**

> **作者:** Eunju Kwon; Seungwon Oh; In-Chang Baek; Yucheon Park; Gyungbo Kim; JaeYoung Moon; Yunho Choi; Kyung-Joong Kim
>
> **摘要:** Contact-rich manipulation has become increasingly important in robot learning. However, previous studies on robot learning datasets have focused on rigid objects and underrepresented the diversity of pressure conditions for real-world manipulation. To address this gap, we present a humanoid visual-tactile-action dataset designed for manipulating deformable soft objects. The dataset was collected via teleoperation using a humanoid robot equipped with dexterous hands, capturing multi-modal interactions under varying pressure conditions. This work also motivates future research on models with advanced optimization strategies capable of effectively leveraging the complexity and diversity of tactile signals.
>
---
#### [new 009] Sim-to-Real Gentle Manipulation of Deformable and Fragile Objects with Stress-Guided Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文研究机器人对柔性和易损物体的温和抓取任务。针对传统方法依赖精确模型或专用设备的问题，提出基于视觉的强化学习方法，通过应力惩罚奖励机制实现损伤感知的轻柔操作，并结合离线示范与渐进式训练课程，使仿真中学得的策略可零样本迁移到真实世界，显著降低36.5%的应力，同时完成任务。**

- **链接: [http://arxiv.org/pdf/2510.25405v1](http://arxiv.org/pdf/2510.25405v1)**

> **作者:** Kei Ikemura; Yifei Dong; David Blanco-Mulero; Alberta Longhini; Li Chen; Florian T. Pokorny
>
> **备注:** Under review
>
> **摘要:** Robotic manipulation of deformable and fragile objects presents significant challenges, as excessive stress can lead to irreversible damage to the object. While existing solutions rely on accurate object models or specialized sensors and grippers, this adds complexity and often lacks generalization. To address this problem, we present a vision-based reinforcement learning approach that incorporates a stress-penalized reward to discourage damage to the object explicitly. In addition, to bootstrap learning, we incorporate offline demonstrations as well as a designed curriculum progressing from rigid proxies to deformables. We evaluate the proposed method in both simulated and real-world scenarios, showing that the policy learned in simulation can be transferred to the real world in a zero-shot manner, performing tasks such as picking up and pushing tofu. Our results show that the learned policies exhibit a damage-aware, gentle manipulation behavior, demonstrating their effectiveness by decreasing the stress applied to fragile objects by 36.5% while achieving the task goals, compared to vanilla RL policies.
>
---
#### [new 010] Integrating Legal and Logical Specifications in Perception, Prediction, and Planning for Automated Driving: A Survey of Methods
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文聚焦自动驾驶中的感知、预测与规划任务，旨在解决技术决策与法律合规性之间的矛盾。通过构建分类体系，综述了融合法律与逻辑规范的方法，重点分析了应对感知不确定性与显式法律约束的技术路径，推动可解释、可问责的自主驾驶系统发展。**

- **链接: [http://arxiv.org/pdf/2510.25386v1](http://arxiv.org/pdf/2510.25386v1)**

> **作者:** Kumar Manas; Mert Keser; Alois Knoll
>
> **备注:** Accepted to 2025 IEEE International Automated Vehicle Validation Conference (IAVVC)
>
> **摘要:** This survey provides an analysis of current methodologies integrating legal and logical specifications into the perception, prediction, and planning modules of automated driving systems. We systematically explore techniques ranging from logic-based frameworks to computational legal reasoning approaches, emphasizing their capability to ensure regulatory compliance and interpretability in dynamic and uncertain driving environments. A central finding is that significant challenges arise at the intersection of perceptual reliability, legal compliance, and decision-making justifiability. To systematically analyze these challenges, we introduce a taxonomy categorizing existing approaches by their theoretical foundations, architectural implementations, and validation strategies. We particularly focus on methods that address perceptual uncertainty and incorporate explicit legal norms, facilitating decisions that are both technically robust and legally defensible. The review covers neural-symbolic integration methods for perception, logic-driven rule representation, and norm-aware prediction strategies, all contributing toward transparent and accountable autonomous vehicle operation. We highlight critical open questions and practical trade-offs that must be addressed, offering multidisciplinary insights from engineering, logic, and law to guide future developments in legally compliant autonomous driving systems.
>
---
#### [new 011] Development of Implicit-Explicit Control Based Amphibious Centipede-Type Robot and Evaluation of its Mobile Performance
- **分类: cs.RO**

- **简介: 该论文针对多地形移动机器人在陆地与水域间切换控制复杂的问题，提出基于隐式-显式控制的蜈蚣型机器人。通过优化腿结构设计，实现单一控制策略下水陆两栖高效运动，实验验证了其在不同环境下的低滑移率与低能耗性能。**

- **链接: [http://arxiv.org/pdf/2510.25280v1](http://arxiv.org/pdf/2510.25280v1)**

> **作者:** Yusuke Tsunoda; Seiya Yamamoto; Kazuki Ito; Runze Xiao; Keisuke Naniwa; Koichi Osuka
>
> **摘要:** Multi-legged mobile robots possess high mobility performance in rough terrain environments, stemming from their high postural stability, joint flexibility, and the redundancy provided by multiple legs. In prior research on navigating between different environments such as land and water, the primary strategy employed involves switching to a controller that generates an appropriate gait for the new environment upon entering it. However, designing appropriate gaits for each complex and diverse environment and accurately determining controller switching for each environment is challenging. Therefore, this research develops a centipede-type mobile robot that navigates both aquatic and terrestrial environments with a simple, unified control scheme, based on the implicit-explicit control philosophy and by ingeniously designing the robot's body structure. In this research, we developed the robot featuring flexible joints and left and right legs on each body segment and focused on the leg structure which has extensive contact with the environment. This paper evaluates the locomotion performance on land and water using the three developed leg structures, using the robot's leg slip rate and actuator energy consumption as evaluation metrics. The experimental results confirmed the existence of an appropriate leg structure capable of navigating both aquatic and terrestrial environments under identical control.
>
---
#### [new 012] Robotic Assistant: Completing Collaborative Tasks with Dexterous Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文研究人机协作中的灵巧操作任务，旨在用少量语言指令驱动机器人完成复杂动作。通过引入FiLM条件化、意图预测头和动作后处理，提升模型对协作任务的感知与执行能力，实现在低延迟下完成“拾取-传递”等长时序行为，并揭示了训练过拟合为关键挑战。**

- **链接: [http://arxiv.org/pdf/2510.25713v1](http://arxiv.org/pdf/2510.25713v1)**

> **作者:** Boshi An; Chenyu Yang; Robert Katzschmann
>
> **摘要:** We adapt a pre-trained Vision-Language-Action (VLA) model (Open-VLA) for dexterous human-robot collaboration with minimal language prompting. Our approach adds (i) FiLM conditioning to visual backbones for task-aware perception, (ii) an auxiliary intent head that predicts collaborator hand pose and target cues, and (iii) action-space post-processing that predicts compact deltas (position/rotation) and PCA-reduced finger joints before mapping to full commands. Using a multi-view, teleoperated Franka and Mimic-hand dataset augmented with MediaPipe hand poses, we demonstrate that delta actions are well-behaved and that four principal components explain ~96% of hand-joint variance. Ablations identify action post-processing as the primary performance driver; auxiliary intent helps, FiLM is mixed, and a directional motion loss is detrimental. A real-time stack (~0.3 s latency on one RTX 4090) composes "pick-up" and "pass" into a long-horizon behavior. We surface "trainer overfitting" to specific demonstrators as the key limitation.
>
---
#### [new 013] Defect Mitigation for Robot Arm-based Additive Manufacturing Utilizing Intelligent Control and IOT
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对机器人增材制造中的缺陷问题，提出融合智能控制与物联网的闭环系统。通过6自由度机械臂、热控微控制器与Oak-D相机，实现温度实时调控、运动同步与在线缺陷检测。基于视觉反馈自动修正缺陷，无需中断打印，有效提升制造质量，适用于航空航天等高要求领域。**

- **链接: [http://arxiv.org/pdf/2510.24994v1](http://arxiv.org/pdf/2510.24994v1)**

> **作者:** Matsive Ali; Blake Gassen; Sen Liu
>
> **备注:** This Paper Has Accepted at ASME 2025 International Mechanical Engineering Congress and Exposition (IMECE 2025)
>
> **摘要:** This paper presents an integrated robotic fused deposition modeling additive manufacturing system featuring closed-loop thermal control and intelligent in-situ defect correction using a 6-degree of freedom robotic arm and an Oak-D camera. The robot arm end effector was modified to mount an E3D hotend thermally regulated by an IoT microcontroller, enabling precise temperature control through real-time feedback. Filament extrusion system was synchronized with robotic motion, coordinated via ROS2, ensuring consistent deposition along complex trajectories. A vision system based on OpenCV detects layer-wise defects position, commanding autonomous re-extrusion at identified sites. Experimental validation demonstrated successful defect mitigation in printing operations. The integrated system effectively addresses challenges real-time quality assurance. Inverse kinematics were used for motion planning, while homography transformations corrected camera perspectives for accurate defect localization. The intelligent system successfully mitigated surface anomalies without interrupting the print process. By combining real-time thermal regulation, motion control, and intelligent defect detection & correction, this architecture establishes a scalable and adaptive robotic additive manufacturing framework suitable for aerospace, biomedical, and industrial applications.
>
---
#### [new 014] Hybrid Vision Servoing with Depp Alignment and GRU-Based Occlusion Recovery
- **分类: cs.RO**

- **简介: 该论文针对机器人视觉伺服中遮挡下的鲁棒目标跟踪问题，提出一种融合深度对齐与GRU遮挡恢复的混合视觉伺服框架。通过快速全局匹配、基于VGG浅层的深度LK精调及轻量回归器，实现亚像素级精度跟踪；在低置信度时利用GRU预测运动历史，保障控制信号连续性，支持30Hz实时伺服，有效应对高达90%遮挡。**

- **链接: [http://arxiv.org/pdf/2510.25233v1](http://arxiv.org/pdf/2510.25233v1)**

> **作者:** Jee Won Lee; Hansol Lim; Sooyeun Yang; Jongseong Brad Choi
>
> **摘要:** Vision-based control systems, such as image-based visual servoing (IBVS), have been extensively explored for precise robot manipulation. A persistent challenge, however, is maintaining robust target tracking under partial or full occlusions. Classical methods like Lucas-Kanade (LK) offer lightweight tracking but are fragile to occlusion and drift, while deep learning-based approaches often require continuous visibility and intensive computation. To address these gaps, we propose a hybrid visual tracking framework that bridges advanced perception with real-time servo control. First, a fast global template matcher constrains the pose search region; next, a deep-feature Lucas-Kanade module operating on early VGG layers refines alignment to sub-pixel accuracy (<2px); then, a lightweight residual regressor corrects local misalignments caused by texture degradation or partial occlusion. When visual confidence falls below a threshold, a GRU-based predictor seamlessly extrapolates pose updates from recent motion history. Crucially, the pipeline's final outputs-translation, rotation, and scale deltas-are packaged as direct control signals for 30Hz image-based servo loops. Evaluated on handheld video sequences with up to 90% occlusion, our system sustains under 2px tracking error, demonstrating the robustness and low-latency precision essential for reliable real-world robot vision applications.
>
---
#### [new 015] Modeling Collapse of Steered Vine Robots Under Their Own Weight
- **分类: cs.RO**

- **简介: 该论文研究软体藤蔓机器人在自重下坍塌的建模问题，旨在预测其在复杂环境中的稳定性。针对机器人跨隙时易坍塌的问题，提出基于真实形状与尾部张力的坍塌模型，并通过实验验证其准确性。研究展示了气动驱动对成功跨越障碍的关键作用，为3D导航提供理论支持。**

- **链接: [http://arxiv.org/pdf/2510.25727v1](http://arxiv.org/pdf/2510.25727v1)**

> **作者:** Ciera McFarland; Margaret McGuinness
>
> **摘要:** Soft, vine-inspired growing robots that move by eversion are highly mobile in confined environments, but, when faced with gaps in the environment, they may collapse under their own weight while navigating a desired path. In this work, we present a comprehensive collapse model that can predict the collapse length of steered robots in any shape using true shape information and tail tension. We validate this model by collapsing several unsteered robots without true shape information. The model accurately predicts the trends of those experiments. We then attempt to collapse a robot steered with a single actuator at different orientations. Our models accurately predict collapse when it occurs. Finally, we demonstrate how this could be used in the field by having a robot attempt a gap-crossing task with and without inflating its actuators. The robot needs its actuators inflated to cross the gap without collapsing, which our model supports. Our model has been specifically tested on straight and series pouch motor-actuated robots made of non-stretchable material, but it could be applied to other robot variations. This work enables us to model the robot's collapse behavior in any open environment and understand the parameters it needs to succeed in 3D navigation tasks.
>
---
#### [new 016] Using VLM Reasoning to Constrain Task and Motion Planning
- **分类: cs.RO**

- **简介: 该论文研究任务与运动规划（TAMP）中的抽象下推可行性问题。针对高阶任务规划因抽象不可行导致运动规划失败的问题，提出VIZ-COAST方法，利用视觉语言模型的常识空间推理能力，预先识别并约束不可行路径，显著减少规划时间，避免反复重规划。**

- **链接: [http://arxiv.org/pdf/2510.25548v1](http://arxiv.org/pdf/2510.25548v1)**

> **作者:** Muyang Yan; Miras Mengdibayev; Ardon Floros; Weihang Guo; Lydia E. Kavraki; Zachary Kingston
>
> **备注:** 8 pages, 7 figures, 1 table. Submitted to ICRA 2026
>
> **摘要:** In task and motion planning, high-level task planning is done over an abstraction of the world to enable efficient search in long-horizon robotics problems. However, the feasibility of these task-level plans relies on the downward refinability of the abstraction into continuous motion. When a domain's refinability is poor, task-level plans that appear valid may ultimately fail during motion planning, requiring replanning and resulting in slower overall performance. Prior works mitigate this by encoding refinement issues as constraints to prune infeasible task plans. However, these approaches only add constraints upon refinement failure, expending significant search effort on infeasible branches. We propose VIZ-COAST, a method of leveraging the common-sense spatial reasoning of large pretrained Vision-Language Models to identify issues with downward refinement a priori, bypassing the need to fix these failures during planning. Experiments on two challenging TAMP domains show that our approach is able to extract plausible constraints from images and domain descriptions, drastically reducing planning times and, in some cases, eliminating downward refinement failures altogether, generalizing to a diverse range of instances from the broader domain.
>
---
#### [new 017] GET-USE: Learning Generalized Tool Usage for Bimanual Mobile Manipulation via Simulated Embodiment Extensions
- **分类: cs.RO**

- **简介: 该论文针对机器人在复杂环境中通用工具使用的问题，提出GeT-USE方法。通过模拟机器人形态扩展，学习识别最优工具几何特征，并将策略迁移至真实双臂移动机器人，实现从多个物体中选择并使用最佳工具。显著提升三类视觉引导任务的成功率30-60%。**

- **链接: [http://arxiv.org/pdf/2510.25754v1](http://arxiv.org/pdf/2510.25754v1)**

> **作者:** Bohan Wu; Paul de La Sayette; Li Fei-Fei; Roberto Martín-Martín
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** The ability to use random objects as tools in a generalizable manner is a missing piece in robots' intelligence today to boost their versatility and problem-solving capabilities. State-of-the-art robotic tool usage methods focused on procedurally generating or crowd-sourcing datasets of tools for a task to learn how to grasp and manipulate them for that task. However, these methods assume that only one object is provided and that it is possible, with the correct grasp, to perform the task; they are not capable of identifying, grasping, and using the best object for a task when many are available, especially when the optimal tool is absent. In this work, we propose GeT-USE, a two-step procedure that learns to perform real-robot generalized tool usage by learning first to extend the robot's embodiment in simulation and then transferring the learned strategies to real-robot visuomotor policies. Our key insight is that by exploring a robot's embodiment extensions (i.e., building new end-effectors) in simulation, the robot can identify the general tool geometries most beneficial for a task. This learned geometric knowledge can then be distilled to perform generalized tool usage tasks by selecting and using the best available real-world object as tool. On a real robot with 22 degrees of freedom (DOFs), GeT-USE outperforms state-of-the-art methods by 30-60% success rates across three vision-based bimanual mobile manipulation tool-usage tasks.
>
---
#### [new 018] SynHLMA:Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出SynHLMA框架，解决基于语言指令生成关节物体手部操作序列的问题。通过离散的HAOI表示与语言嵌入对齐，在共享空间中建模手物交互，结合关节感知损失，实现生成、预测与插值三类任务，支持机器人灵巧抓握应用。**

- **链接: [http://arxiv.org/pdf/2510.25268v1](http://arxiv.org/pdf/2510.25268v1)**

> **作者:** Wang zhi; Yuyan Liu; Liu Liu; Li Zhang; Ruixuan Lu; Dan Guo
>
> **摘要:** Generating hand grasps with language instructions is a widely studied topic that benefits from embodied AI and VR/AR applications. While transferring into hand articulatied object interaction (HAOI), the hand grasps synthesis requires not only object functionality but also long-term manipulation sequence along the object deformation. This paper proposes a novel HAOI sequence generation framework SynHLMA, to synthesize hand language manipulation for articulated objects. Given a complete point cloud of an articulated object, we utilize a discrete HAOI representation to model each hand object interaction frame. Along with the natural language embeddings, the representations are trained by an HAOI manipulation language model to align the grasping process with its language description in a shared representation space. A joint-aware loss is employed to ensure hand grasps follow the dynamic variations of articulated object joints. In this way, our SynHLMA achieves three typical hand manipulation tasks for articulated objects of HAOI generation, HAOI prediction and HAOI interpolation. We evaluate SynHLMA on our built HAOI-lang dataset and experimental results demonstrate the superior hand grasp sequence generation performance comparing with state-of-the-art. We also show a robotics grasp application that enables dexterous grasps execution from imitation learning using the manipulation sequence provided by our SynHLMA. Our codes and datasets will be made publicly available.
>
---
#### [new 019] STITCH 2.0: Extending Augmented Suturing with EKF Needle Estimation and Thread Management
- **分类: cs.RO**

- **简介: 该论文聚焦于机器人辅助缝合任务，针对前代系统在针头追踪不准和缝线管理差导致伤口闭合不全的问题，提出STITCH 2.0，通过改进EKF针头估计、新缝线解缠方法及自动3D缝合对齐，显著提升缝合效率与闭合率。**

- **链接: [http://arxiv.org/pdf/2510.25768v1](http://arxiv.org/pdf/2510.25768v1)**

> **作者:** Kush Hari; Ziyang Chen; Hansoul Kim; Ken Goldberg
>
> **备注:** Published in RA-L 2025
>
> **摘要:** Surgical suturing is a high-precision task that impacts patient healing and scarring. Suturing skill varies widely between surgeons, highlighting the need for robot assistance. Previous robot suturing works, such as STITCH 1.0 [1], struggle to fully close wounds due to inaccurate needle tracking and poor thread management. To address these challenges, we present STITCH 2.0, an elevated augmented dexterity pipeline with seven improvements including: improved EKF needle pose estimation, new thread untangling methods, and an automated 3D suture alignment algorithm. Experimental results over 15 trials find that STITCH 2.0 on average achieves 74.4% wound closure with 4.87 sutures per trial, representing 66% more sutures in 38% less time compared to the previous baseline. When two human interventions are allowed, STITCH 2.0 averages six sutures with 100% wound closure rate. Project website: https://stitch-2.github.io/
>
---
#### [new 020] RoadSens-4M: A Multimodal Smartphone & Camera Dataset for Holistic Road-way Analysis
- **分类: cs.RO**

- **简介: 该论文提出RoadSens-4M数据集，用于道路状况的全面分析。针对缺乏高质量、多模态道路数据的问题，整合智能手机传感器、GIS、天气与视频数据，支持交通管理与智能交通研究，推动道路安全与城市规划发展。**

- **链接: [http://arxiv.org/pdf/2510.25211v1](http://arxiv.org/pdf/2510.25211v1)**

> **作者:** Amith Khandakar; David Michelson; Shaikh Golam Rabbani; Fariya Bintay Shafi; Md. Faysal Ahamed; Khondokar Radwanur Rahman; Md Abidur Rahman; Md. Fahmidun Nabi; Mohamed Arselene Ayari; Khaled Khan; Ponnuthurai Nagaratnam Suganthan
>
> **摘要:** It's important to monitor road issues such as bumps and potholes to enhance safety and improve road conditions. Smartphones are equipped with various built-in sensors that offer a cost-effective and straightforward way to assess road quality. However, progress in this area has been slow due to the lack of high-quality, standardized datasets. This paper discusses a new dataset created by a mobile app that collects sensor data from devices like GPS, accelerometers, gyroscopes, magnetometers, gravity sensors, and orientation sensors. This dataset is one of the few that integrates Geographic Information System (GIS) data with weather information and video footage of road conditions, providing a comprehensive understanding of road issues with geographic context. The dataset allows for a clearer analysis of road conditions by compiling essential data, including vehicle speed, acceleration, rotation rates, and magnetic field intensity, along with the visual and spatial context provided by GIS, weather, and video data. Its goal is to provide funding for initiatives that enhance traffic management, infrastructure development, road safety, and urban planning. Additionally, the dataset will be publicly accessible to promote further research and innovation in smart transportation systems.
>
---
#### [new 021] Non-Invasive Calibration Of A Stewart Platform By Photogrammetry
- **分类: cs.RO**

- **简介: 该论文针对斯特林平台（Stewart platform）的高精度标定问题，提出一种基于摄影测量法的非侵入式标定方法。通过多角度图像获取运动平台位姿，结合最小二乘法进行误差补偿，无需附加设备或改动硬件，有效提升了位姿精度，解决了前向运动学求解复杂与多解难题。**

- **链接: [http://arxiv.org/pdf/2510.25072v1](http://arxiv.org/pdf/2510.25072v1)**

> **作者:** Sourabh Karmakar; Cameron J. Turner
>
> **摘要:** Accurate calibration of a Stewart platform is important for their precise and efficient operation. However, the calibration of these platforms using forward kinematics is a challenge for researchers because forward kinematics normally generates multiple feasible and unfeasible solutions for any pose of the moving platform. The complex kinematic relations among the six actuator paths connecting the fixed base to the moving platform further compound the difficulty in establishing a straightforward and efficient calibration method. The authors developed a new forward kinematics-based calibration method using Denavit-Hartenberg convention and used the Stewart platform Tiger 66.1 developed in their lab for experimenting with the photogrammetry-based calibration strategies described in this paper. This system became operational upon completion of construction, marking its inaugural use. The authors used their calibration model for estimating the errors in the system and adopted three compensation options or strategies as per Least Square method to improve the accuracy of the system. These strategies leveraged a high-resolution digital camera and off-the-shelf software to capture the poses of the moving platform's center. This process is non-invasive and does not need any additional equipment to be attached to the hexapod or any alteration of the hexapod hardware. This photogrammetry-based calibration process involves multiple high-resolution images from different angles to measure the position and orientation of the platform center in the three-dimensional space. The Target poses and Actual poses are then compared, and the error compensations are estimated using the Least-Squared methods to calculate the Predicted poses. Results from each of the three compensation approaches demonstrated noticeable enhancements in platform pose accuracies, suggesting room for further improvements.
>
---
#### [new 022] Octopus-like Reaching Motion: A Perspective Inspired by Whipping
- **分类: cs.RO; physics.bio-ph**

- **简介: 该论文研究章鱼臂伸展运动的生物力学机制，旨在揭示其是否由鞭状被动动力学驱动。通过水/空气中的平台鞭甩实验，发现仅在水中能复现类似章鱼的弯曲传播，且速度呈生物钟形而非单调下降，表明其非单纯被动鞭动，介质环境至关重要。为理解生物运动提供新视角。**

- **链接: [http://arxiv.org/pdf/2510.25520v1](http://arxiv.org/pdf/2510.25520v1)**

> **作者:** Shengyao Zhang; Yiyuan Zhang; Chenrui Zhang; Yiming Li; Wenci Xin; Yuliang Liufu; Hong Wei Ng; Cecilia Laschi
>
> **备注:** The first two listed authors contributed equally. Yiyuan Zhang is the corresponding author
>
> **摘要:** The stereotypical reaching motion of the octopus arm has drawn growing attention for its efficient control of a highly deformable body. Previous studies suggest that its characteristic bend propagation may share underlying principles with the dynamics of a whip. This work investigates whether whip-like passive dynamics in water can reproduce the kinematic features observed in biological reaching and their similarities and differences. Platform-based whipping tests were performed in water and air while systematically varying material stiffness and driving speed. Image-based quantification revealed that the Ecoflex Gel 2 arm driven at 150 rpm (motor speed) reproduced curvature propagation similar to that observed in octopus reaching. However, its bend-point velocity decreased monotonically rather than exhibiting the biological bell-shaped profile, confirming that the octopus reaching movement is not merely a passive whipping behavior. The absence of propagation in air further highlights the critical role of the surrounding medium in forming octopus-like reaching motion. This study provides a new perspective for understand biological reaching movement, and offers a potential platform for future hydrodynamic research.
>
---
#### [new 023] Collision avoidance and path finding in a robotic mobile fulfillment system using multi-objective meta-heuristics
- **分类: cs.RO; math.OC**

- **简介: 该论文研究机器人移动分拣系统中的多智能体路径规划问题，旨在解决AGV间碰撞避免与任务分配难题。提出兼顾能耗与行程时间的碰撞规避策略，并采用NSGA与ALNS两种多目标算法优化任务分配，显著提升路径规划效率与系统性能。**

- **链接: [http://arxiv.org/pdf/2510.25650v1](http://arxiv.org/pdf/2510.25650v1)**

> **作者:** Ahmad Kokhahi; Mary Kurz
>
> **摘要:** Multi-Agent Path Finding (MAPF) has gained significant attention, with most research focusing on minimizing collisions and travel time. This paper also considers energy consumption in the path planning of automated guided vehicles (AGVs). It addresses two main challenges: i) resolving collisions between AGVs and ii) assigning tasks to AGVs. We propose a new collision avoidance strategy that takes both energy use and travel time into account. For task assignment, we present two multi-objective algorithms: Non-Dominated Sorting Genetic Algorithm (NSGA) and Adaptive Large Neighborhood Search (ALNS). Comparative evaluations show that these proposed methods perform better than existing approaches in both collision avoidance and task assignment.
>
---
#### [new 024] Time-Optimal Transport of Loosely Placed Liquid Filled Cups along Prescribed Paths
- **分类: cs.RO**

- **简介: 该论文研究机器人在最短时间内沿预定路径运输装有液体的松散放置杯子的任务。针对液体晃动导致溢出的问题，将晃动动力学纳入最优控制模型，采用直接多打靶法求解，以最小化晃动，实现无溢出高效运输。**

- **链接: [http://arxiv.org/pdf/2510.25255v1](http://arxiv.org/pdf/2510.25255v1)**

> **作者:** Klaus Zauner; Hubert Gattringer; Andreas Mueller
>
> **摘要:** Handling loosely placed objects with robotic manipulators is a difficult task from the point of view of trajectory planning and control. This becomes even more challenging when the object to be handled is a container filled with liquid. This paper addresses the task of transporting a liquid-filled cup placed on a tray along a prescribed path in shortest time. The objective is to minimize swapping, thus avoiding spillage of the fluid. To this end, the sloshing dynamics is incorporated into the dynamic model used within the optimal control problem formulation. The optimization problem is solved using a direct multiple shooting approach.
>
---
#### [new 025] SCOUT: A Lightweight Framework for Scenario Coverage Assessment in Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出SCOUT框架，用于高效评估自动驾驶场景覆盖率。针对现有方法依赖昂贵人工标注或高成本大模型的问题，提出基于代理模型的轻量级方案，通过知识蒸馏从大模型标签学习，直接利用预计算感知特征进行快速预测，显著降低计算开销，实现大规模场景覆盖分析。**

- **链接: [http://arxiv.org/pdf/2510.24949v1](http://arxiv.org/pdf/2510.24949v1)**

> **作者:** Anil Yildiz; Sarah M. Thornton; Carl Hildebrandt; Sreeja Roy-Singh; Mykel J. Kochenderfer
>
> **摘要:** Assessing scenario coverage is crucial for evaluating the robustness of autonomous agents, yet existing methods rely on expensive human annotations or computationally intensive Large Vision-Language Models (LVLMs). These approaches are impractical for large-scale deployment due to cost and efficiency constraints. To address these shortcomings, we propose SCOUT (Scenario Coverage Oversight and Understanding Tool), a lightweight surrogate model designed to predict scenario coverage labels directly from an agent's latent sensor representations. SCOUT is trained through a distillation process, learning to approximate LVLM-generated coverage labels while eliminating the need for continuous LVLM inference or human annotation. By leveraging precomputed perception features, SCOUT avoids redundant computations and enables fast, scalable scenario coverage estimation. We evaluate our method across a large dataset of real-life autonomous navigation scenarios, demonstrating that it maintains high accuracy while significantly reducing computational cost. Our results show that SCOUT provides an effective and practical alternative for large-scale coverage analysis. While its performance depends on the quality of LVLM-generated training labels, SCOUT represents a major step toward efficient scenario coverage oversight in autonomous systems.
>
---
#### [new 026] One-shot Humanoid Whole-body Motion Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对人形机器人全身运动学习任务，解决现有方法依赖多样本训练的问题。提出仅需一个非行走动作样本与通用行走动作，通过保序最优传输计算序列距离，沿测地线插值生成中间姿态骨架，经避障优化与角色重定向后，在仿真环境中用强化学习训练运动策略，显著提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2510.25241v1](http://arxiv.org/pdf/2510.25241v1)**

> **作者:** Hao Huang; Geeta Chandra Raju Bethala; Shuaihang Yuan; Congcong Wen; Anthony Tzes; Yi Fang
>
> **备注:** 10 pages, 3 figures, 5 tables
>
> **摘要:** Whole-body humanoid motion represents a cornerstone challenge in robotics, integrating balance, coordination, and adaptability to enable human-like behaviors. However, existing methods typically require multiple training samples per motion category, rendering the collection of high-quality human motion datasets both labor-intensive and costly. To address this, we propose a novel approach that trains effective humanoid motion policies using only a single non-walking target motion sample alongside readily available walking motions. The core idea lies in leveraging order-preserving optimal transport to compute distances between walking and non-walking sequences, followed by interpolation along geodesics to generate new intermediate pose skeletons, which are then optimized for collision-free configurations and retargeted to the humanoid before integration into a simulated environment for policy training via reinforcement learning. Experimental evaluations on the CMU MoCap dataset demonstrate that our method consistently outperforms baselines, achieving superior performance across metrics. Code will be released upon acceptance.
>
---
#### [new 027] SoraNav: Adaptive UAV Task-Centric Navigation via Zeroshot VLM Reasoning
- **分类: cs.RO**

- **简介: 该论文提出SoraNav，一种面向无人机的零样本视觉语言导航框架。针对现有方法难以适应空中3D环境的问题，融合零样本视觉语言模型与几何先验，通过混合切换策略提升导航成功率与路径效率，在2.5D和3D场景中显著优于基线。**

- **链接: [http://arxiv.org/pdf/2510.25191v1](http://arxiv.org/pdf/2510.25191v1)**

> **作者:** Hongyu Song; Rishabh Dev Yadav; Cheng Guo; Wei Pan
>
> **摘要:** Interpreting visual observations and natural language instructions for complex task execution remains a key challenge in robotics and AI. Despite recent advances, language-driven navigation is still difficult, particularly for UAVs in small-scale 3D environments. Existing Vision-Language Navigation (VLN) approaches are mostly designed for ground robots and struggle to generalize to aerial tasks that require full 3D spatial reasoning. The emergence of large Vision-Language Models (VLMs), such as GPT and Claude, enables zero-shot semantic reasoning from visual and textual inputs. However, these models lack spatial grounding and are not directly applicable to navigation. To address these limitations, SoraNav is introduced, an adaptive UAV navigation framework that integrates zero-shot VLM reasoning with geometry-aware decision-making. Geometric priors are incorporated into image annotations to constrain the VLM action space and improve decision quality. A hybrid switching strategy leverages navigation history to alternate between VLM reasoning and geometry-based exploration, mitigating dead-ends and redundant revisits. A PX4-based hardware-software platform, comprising both a digital twin and a physical micro-UAV, enables reproducible evaluation. Experimental results show that in 2.5D scenarios, our method improves Success Rate (SR) by 25.7% and Success weighted by Path Length (SPL) by 17%. In 3D scenarios, it improves SR by 29.5% and SPL by 18.5% relative to the baseline.
>
---
#### [new 028] Solving the Right Problem with Multi-Robot Formations
- **分类: cs.RO; cs.MA**

- **简介: 该论文针对多机器人编队控制中形状抽象与实际成本函数不匹配的问题，提出一种两阶段编队规划方法。通过构建代理成本函数并优化相对位置，使编队在动态环境中更高效地最小化保护、避障等真实成本，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.25422v1](http://arxiv.org/pdf/2510.25422v1)**

> **作者:** Chaz Cornwall; Jeremy P. Bos
>
> **备注:** Submitted to SAE WCX 2026
>
> **摘要:** Formation control simplifies minimizing multi-robot cost functions by encoding a cost function as a shape the robots maintain. However, by reducing complex cost functions to formations, discrepancies arise between maintaining the shape and minimizing the original cost function. For example, a Diamond or Box formation shape is often used for protecting all members of the formation. When more information about the surrounding environment becomes available, a static shape often no longer minimizes the original protection cost. We propose a formation planner to reduce mismatch between a formation and the cost function while still leveraging efficient formation controllers. Our formation planner is a two-step optimization problem that identifies desired relative robot positions. We first solve a constrained problem to estimate non-linear and non-differentiable costs with a weighted sum of surrogate cost functions. We theoretically analyze this problem and identify situations where weights do not need to be updated. The weighted, surrogate cost function is then minimized using relative positions between robots. The desired relative positions are realized using a non-cooperative formation controller derived from Lyapunov's direct approach. We then demonstrate the efficacy of this approach for military-like costs such as protection and obstacle avoidance. In simulations, we show a formation planner can reduce a single cost by over 75%. When minimizing a variety of cost functions simultaneously, using a formation planner with adaptive weights can reduce the cost by 20-40%. Formation planning provides better performance by minimizing a surrogate cost function that closely approximates the original cost function instead of relying on a shape abstraction.
>
---
#### [new 029] Mean-Shift Theory and Its Applications in Swarm Robotics: A New Way to Enhance the Efficiency of Multi-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文研究多机器人协同中的形状形成任务，针对传统基于分配方法在大规模场景下效率低、难扩展的问题，提出无分配的均值漂移探索策略，显著提升大尺度集群协作效率，应用于智能仓储、区域探测与货物运输等场景。**

- **链接: [http://arxiv.org/pdf/2510.25086v1](http://arxiv.org/pdf/2510.25086v1)**

> **作者:** Guibin Sun; Jinhu Lü; Kexin Liu; Zhenqian Wang; Guanrong Chen
>
> **摘要:** Swarms evolving from collective behaviors among multiple individuals are commonly seen in nature, which enables biological systems to exhibit more efficient and robust collaboration. Creating similar swarm intelligence in engineered robots poses challenges to the design of collaborative algorithms that can be programmed at large scales. The assignment-based method has played an eminent role for a very long time in solving collaboration problems of robot swarms. However, it faces fundamental limitations in terms of efficiency and robustness due to its unscalability to swarm variants. This article presents a tutorial review on recent advances in assignment-free collaboration of robot swarms, focusing on the problem of shape formation. A key theoretical component is the recently developed \emph{mean-shift exploration} strategy, which improves the collaboration efficiency of large-scale swarms by dozens of times. Further, the efficiency improvement is more significant as the swarm scale increases. Finally, this article discusses three important applications of the mean-shift exploration strategy, including precise shape formation, area coverage formation, and maneuvering formation, as well as their corresponding industrial scenarios in smart warehousing, area exploration, and cargo transportation.
>
---
#### [new 030] Smooth path planning with safety margins using Piece-Wise Bezier curves
- **分类: cs.RO**

- **简介: 该论文针对移动机器人路径规划任务，提出一种基于分段二次贝塞尔曲线的高效优化方法，通过二次规划显式融入安全裕度，实现平滑、$C^1$连续路径生成。相比传统分段线性方法，显著提升路径质量与鲁棒性，适用于实时嵌入式系统。**

- **链接: [http://arxiv.org/pdf/2510.24972v1](http://arxiv.org/pdf/2510.24972v1)**

> **作者:** Iancu Andrei; Marius Kloetzer; Cristian Mahulea; Catalin Dosoftei
>
> **摘要:** In this paper, we propose a computationally efficient quadratic programming (QP) approach for generating smooth, $C^1$ continuous paths for mobile robots using piece-wise quadratic Bezier (PWB) curves. Our method explicitly incorporates safety margins within a structured optimization framework, balancing trajectory smoothness and robustness with manageable numerical complexity suitable for real-time and embedded applications. Comparative simulations demonstrate clear advantages over traditional piece-wise linear (PWL) path planning methods, showing reduced trajectory deviations, enhanced robustness, and improved overall path quality. These benefits are validated through simulations using a Pure-Pursuit controller in representative scenarios, highlighting the practical effectiveness and scalability of our approach for safe navigation.
>
---
#### [new 031] Seeing Clearly and Deeply: An RGBD Imaging Approach with a Bio-inspired Monocentric Design
- **分类: cs.CV; cs.RO; eess.IV; physics.optics**

- **简介: 该论文针对紧凑型RGBD成像中图像清晰度与深度精度难以兼顾的问题，提出生物启发的全球面单中心镜头与联合重建框架（BMI）。通过物理建模生成合成数据，结合双头多尺度网络，实现单次拍摄下高保真全焦图像与精确深度图的联合恢复，显著优于现有软硬件方案。**

- **链接: [http://arxiv.org/pdf/2510.25314v1](http://arxiv.org/pdf/2510.25314v1)**

> **作者:** Zongxi Yu; Xiaolong Qian; Shaohua Gao; Qi Jiang; Yao Gao; Kailun Yang; Kaiwei Wang
>
> **备注:** The source code will be publicly available at https://github.com/ZongxiYu-ZJU/BMI
>
> **摘要:** Achieving high-fidelity, compact RGBD imaging presents a dual challenge: conventional compact optics struggle with RGB sharpness across the entire depth-of-field, while software-only Monocular Depth Estimation (MDE) is an ill-posed problem reliant on unreliable semantic priors. While deep optics with elements like DOEs can encode depth, they introduce trade-offs in fabrication complexity and chromatic aberrations, compromising simplicity. To address this, we first introduce a novel bio-inspired all-spherical monocentric lens, around which we build the Bionic Monocentric Imaging (BMI) framework, a holistic co-design. This optical design naturally encodes depth into its depth-varying Point Spread Functions (PSFs) without requiring complex diffractive or freeform elements. We establish a rigorous physically-based forward model to generate a synthetic dataset by precisely simulating the optical degradation process. This simulation pipeline is co-designed with a dual-head, multi-scale reconstruction network that employs a shared encoder to jointly recover a high-fidelity All-in-Focus (AiF) image and a precise depth map from a single coded capture. Extensive experiments validate the state-of-the-art performance of the proposed framework. In depth estimation, the method attains an Abs Rel of 0.026 and an RMSE of 0.130, markedly outperforming leading software-only approaches and other deep optics systems. For image restoration, the system achieves an SSIM of 0.960 and a perceptual LPIPS score of 0.082, thereby confirming a superior balance between image fidelity and depth accuracy. This study illustrates that the integration of bio-inspired, fully spherical optics with a joint reconstruction algorithm constitutes an effective strategy for addressing the intrinsic challenges in high-performance compact RGBD imaging. Source code will be publicly available at https://github.com/ZongxiYu-ZJU/BMI.
>
---
#### [new 032] Incorporating Social Awareness into Control of Unknown Multi-Agent Systems: A Real-Time Spatiotemporal Tubes Approach
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文研究未知多智能体系统的实时协同控制任务，旨在实现预定时间内避障、到达目标并保持状态。针对异构社会行为与动态环境挑战，提出基于时空管的分布式控制框架，无需模型、在线生成管状轨迹，确保安全、时效性与抗扰性。**

- **链接: [http://arxiv.org/pdf/2510.25597v1](http://arxiv.org/pdf/2510.25597v1)**

> **作者:** Siddhartha Upadhyay; Ratnangshu Das; Pushpak Jagtap
>
> **摘要:** This paper presents a decentralized control framework that incorporates social awareness into multi-agent systems with unknown dynamics to achieve prescribed-time reach-avoid-stay tasks in dynamic environments. Each agent is assigned a social awareness index that quantifies its level of cooperation or self-interest, allowing heterogeneous social behaviors within the system. Building on the spatiotemporal tube (STT) framework, we propose a real-time STT framework that synthesizes tubes online for each agent while capturing its social interactions with others. A closed-form, approximation-free control law is derived to ensure that each agent remains within its evolving STT, thereby avoiding dynamic obstacles while also preventing inter-agent collisions in a socially aware manner, and reaching the target within a prescribed time. The proposed approach provides formal guarantees on safety and timing, and is computationally lightweight, model-free, and robust to unknown disturbances. The effectiveness and scalability of the framework are validated through simulation and hardware experiments on a 2D omnidirectional
>
---
#### [new 033] DrivingScene: A Multi-Task Online Feed-Forward 3D Gaussian Splatting Method for Dynamic Driving Scenes
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出DrivingScene，面向动态驾驶场景的实时高保真重建任务。针对复杂动态与稀疏视角下的重建难题，提出基于双帧图像的在线前馈框架，通过轻量级残差光流网络与静态场景先验结合，显式建模非刚性运动，实现高质量深度、场景流与3D高斯点云的在线生成。**

- **链接: [http://arxiv.org/pdf/2510.24734v1](http://arxiv.org/pdf/2510.24734v1)**

> **作者:** Qirui Hou; Wenzhang Sun; Chang Zeng; Chunfeng Wang; Hao Li; Jianxun Cui
>
> **备注:** Autonomous Driving, Novel view Synthesis, Multi task Learning
>
> **摘要:** Real-time, high-fidelity reconstruction of dynamic driving scenes is challenged by complex dynamics and sparse views, with prior methods struggling to balance quality and efficiency. We propose DrivingScene, an online, feed-forward framework that reconstructs 4D dynamic scenes from only two consecutive surround-view images. Our key innovation is a lightweight residual flow network that predicts the non-rigid motion of dynamic objects per camera on top of a learned static scene prior, explicitly modeling dynamics via scene flow. We also introduce a coarse-to-fine training paradigm that circumvents the instabilities common to end-to-end approaches. Experiments on nuScenes dataset show our image-only method simultaneously generates high-quality depth, scene flow, and 3D Gaussian point clouds online, significantly outperforming state-of-the-art methods in both dynamic reconstruction and novel view synthesis.
>
---
#### [new 034] SPADE: Sparsity Adaptive Depth Estimator for Zero-Shot, Real-Time, Monocular Depth Estimation in Underwater Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SPADE，一种用于水下环境的零样本、实时单目深度估计方法。针对水下复杂场景中感知受限问题，结合稀疏深度先验与相对深度估计，通过两阶段优化生成稠密度量深度图，提升精度与泛化性，并在嵌入式设备上实现15 FPS以上运行，助力水下自主巡检。**

- **链接: [http://arxiv.org/pdf/2510.25463v1](http://arxiv.org/pdf/2510.25463v1)**

> **作者:** Hongjie Zhang; Gideon Billings; Stefan B. Williams
>
> **摘要:** Underwater infrastructure requires frequent inspection and maintenance due to harsh marine conditions. Current reliance on human divers or remotely operated vehicles is limited by perceptual and operational challenges, especially around complex structures or in turbid water. Enhancing the spatial awareness of underwater vehicles is key to reducing piloting risks and enabling greater autonomy. To address these challenges, we present SPADE: SParsity Adaptive Depth Estimator, a monocular depth estimation pipeline that combines pre-trained relative depth estimator with sparse depth priors to produce dense, metric scale depth maps. Our two-stage approach first scales the relative depth map with the sparse depth points, then refines the final metric prediction with our proposed Cascade Conv-Deformable Transformer blocks. Our approach achieves improved accuracy and generalisation over state-of-the-art baselines and runs efficiently at over 15 FPS on embedded hardware, promising to support practical underwater inspection and intervention. This work has been submitted to IEEE Journal of Oceanic Engineering Special Issue of AUV 2026.
>
---
#### [new 035] Don't Blind Your VLA: Aligning Visual Representations for OOD Generalization
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究视觉-语言-动作（VLA）模型在任务微调中视觉表征退化问题。针对动作微调导致视觉-语言表征弱化、影响分布外泛化的问题，提出分析方法并设计对齐策略，有效保留原始视觉语言能力，提升OOD性能。**

- **链接: [http://arxiv.org/pdf/2510.25616v1](http://arxiv.org/pdf/2510.25616v1)**

> **作者:** Nikita Kachaev; Mikhail Kolosov; Daniil Zelezetsky; Alexey K. Kovalev; Aleksandr I. Panov
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** The growing success of Vision-Language-Action (VLA) models stems from the promise that pretrained Vision-Language Models (VLMs) can endow agents with transferable world knowledge and vision-language (VL) grounding, laying a foundation for action models with broader generalization. Yet when these VLMs are adapted to the action modality, it remains unclear to what extent their original VL representations and knowledge are preserved. In this work, we conduct a systematic study of representation retention during VLA fine-tuning, showing that naive action fine-tuning leads to degradation of visual representations. To characterize and measure these effects, we probe VLA's hidden representations and analyze attention maps, further, we design a set of targeted tasks and methods that contrast VLA models with their counterpart VLMs, isolating changes in VL capabilities induced by action fine-tuning. We further evaluate a range of strategies for aligning visual representations and introduce a simple yet effective method that mitigates degradation and yields improved generalization to out-of-distribution (OOD) scenarios. Taken together, our analysis clarifies the trade-off between action fine-tuning and the degradation of VL representations and highlights practical approaches to recover inherited VL capabilities. Code is publicly available: https://blind-vla-paper.github.io
>
---
#### [new 036] Point-level Uncertainty Evaluation of Mobile Laser Scanning Point Clouds
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文针对移动激光扫描点云的点级不确定性评估问题，提出基于随机森林和XGBoost的机器学习框架，利用局部几何特征预测误差。通过空间分区训练避免数据泄露，有效捕捉非线性关系，显著提升评估精度，为大规模点云质量控制提供可扩展的数据驱动方法。**

- **链接: [http://arxiv.org/pdf/2510.24773v1](http://arxiv.org/pdf/2510.24773v1)**

> **作者:** Ziyang Xu; Olaf Wysocki; Christoph Holst
>
> **摘要:** Reliable quantification of uncertainty in Mobile Laser Scanning (MLS) point clouds is essential for ensuring the accuracy and credibility of downstream applications such as 3D mapping, modeling, and change analysis. Traditional backward uncertainty modeling heavily rely on high-precision reference data, which are often costly or infeasible to obtain at large scales. To address this issue, this study proposes a machine learning-based framework for point-level uncertainty evaluation that learns the relationship between local geometric features and point-level errors. The framework is implemented using two ensemble learning models, Random Forest (RF) and XGBoost, which are trained and validated on a spatially partitioned real-world dataset to avoid data leakage. Experimental results demonstrate that both models can effectively capture the nonlinear relationships between geometric characteristics and uncertainty, achieving mean ROC-AUC values above 0.87. The analysis further reveals that geometric features describing elevation variation, point density, and local structural complexity play a dominant role in predicting uncertainty. The proposed framework offers a data-driven perspective of uncertainty evaluation, providing a scalable and adaptable foundation for future quality control and error analysis of large-scale point clouds.
>
---
#### [new 037] A Survey on Efficient Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文聚焦于高效视觉-语言-动作模型（Efficient VLAs），旨在解决大模型在机器人领域部署中计算与数据成本高的问题。通过构建统一分类框架，系统梳理了高效模型设计、训练与数据收集三方面技术，总结现状、挑战与未来方向，为该领域提供全面参考。**

- **链接: [http://arxiv.org/pdf/2510.24795v1](http://arxiv.org/pdf/2510.24795v1)**

> **作者:** Zhaoshu Yu; Bo Wang; Pengpeng Zeng; Haonan Zhang; Ji Zhang; Lianli Gao; Jingkuan Song; Nicu Sebe; Heng Tao Shen
>
> **备注:** 26 pages, 8 figures
>
> **摘要:** Vision-Language-Action models (VLAs) represent a significant frontier in embodied intelligence, aiming to bridge digital knowledge with physical-world interaction. While these models have demonstrated remarkable generalist capabilities, their deployment is severely hampered by the substantial computational and data requirements inherent to their underlying large-scale foundation models. Motivated by the urgent need to address these challenges, this survey presents the first comprehensive review of Efficient Vision-Language-Action models (Efficient VLAs) across the entire data-model-training process. Specifically, we introduce a unified taxonomy to systematically organize the disparate efforts in this domain, categorizing current techniques into three core pillars: (1) Efficient Model Design, focusing on efficient architectures and model compression; (2) Efficient Training, which reduces computational burdens during model learning; and (3) Efficient Data Collection, which addresses the bottlenecks in acquiring and utilizing robotic data. Through a critical review of state-of-the-art methods within this framework, this survey not only establishes a foundational reference for the community but also summarizes representative applications, delineates key challenges, and charts a roadmap for future research. We maintain a continuously updated project page to track our latest developments: https://evla-survey.github.io/
>
---
## 更新

#### [replaced 001] Federated Deep Reinforcement Learning for Privacy-Preserving Robotic-Assisted Surgery
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12153v2](http://arxiv.org/pdf/2505.12153v2)**

> **作者:** Sana Hafeez; Sundas Rafat Mulkana; Muhammad Ali Imran; Michele Sevegnani
>
> **备注:** 11 pages, 7 figures, conference
>
> **摘要:** The integration of Reinforcement Learning (RL) into robotic-assisted surgery (RAS) holds significant promise for advancing surgical precision, adaptability, and autonomous decision-making. However, the development of robust RL models in clinical settings is hindered by key challenges, including stringent patient data privacy regulations, limited access to diverse surgical datasets, and high procedural variability. To address these limitations, this paper presents a Federated Deep Reinforcement Learning (FDRL) framework that enables decentralized training of RL models across multiple healthcare institutions without exposing sensitive patient information. A central innovation of the proposed framework is its dynamic policy adaptation mechanism, which allows surgical robots to select and tailor patient-specific policies in real-time, thereby ensuring personalized and Optimised interventions. To uphold rigorous privacy standards while facilitating collaborative learning, the FDRL framework incorporates secure aggregation, differential privacy, and homomorphic encryption techniques. Experimental results demonstrate a 60\% reduction in privacy leakage compared to conventional methods, with surgical precision maintained within a 1.5\% margin of a centralized baseline. This work establishes a foundational approach for adaptive, secure, and patient-centric AI-driven surgical robotics, offering a pathway toward clinical translation and scalable deployment across diverse healthcare environments.
>
---
#### [replaced 002] Control Modes of Teleoperated Surgical Robotic System's Tools in Ophthalmic Surgery
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13654v2](http://arxiv.org/pdf/2507.13654v2)**

> **作者:** Haoran Wang; Yasamin Foroutani; Matthew Nepo; Mercedes Rodriguez; Ji Ma; Jean-Pierre Hubschman; Tsu-Chin Tsao; Jacob Rosen
>
> **备注:** 10 pages, 11 figures
>
> **摘要:** The introduction of a teleoperated surgical robotic system designed for minimally invasive procedures enables the emulation of two distinct control modes through a dedicated input device of the surgical console: (1) Inside Control Mode, which emulates tool manipulation near the distal end as if the surgeon was holding the tip of the instrument inside the patient's body; (2) Outside Control Mode, which emulates manipulation near the proximal end as if the surgeon was holding the tool externally. The aim of this research is to compare the surgeon's performance on these two modes of operation along with various scaling factors in a simulated vitreoretinal surgical setting. The console of Intraocular Robotic Interventional Surgical System (IRISS) was utilized but the surgical robot itself and the human eye anatomy was simulated by a virtual environment projected microscope view of an intraocular setup to a VR headset. Five experienced vitreoretinal surgeons and five subjects with no surgical experience used the system to perform four fundamental tool/tissue tasks common to vitreoretinal surgery: touch and reset; grasp and drop; inject; circular tracking. Results indicate that Inside Control outperforms Outside Control across multiple tasks and metrics. Higher scaling factors generally performed better, particularly for reducing trajectory errors and tissue damage. This improvement suggests that larger scaling factors enable more precise control, making them the preferred option for fine manipulation. However, completion time was not consistently reduced across all conditions, indicating that surgeons need to balance speed and accuracy based on surgical requirements. By optimizing control dynamics and user interface, robotic teleoperation has the potential to reduce complications, enhance dexterity, and expand the accessibility of high precision procedures to a broader range of practitioners.
>
---
#### [replaced 003] Quantum Machine Learning and Grover's Algorithm for Quantum Optimization of Robotic Manipulators
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.07216v2](http://arxiv.org/pdf/2509.07216v2)**

> **作者:** Hassen Nigatu; Shi Gaokun; Li Jituo; Wang Jin; Lu Guodong; Howard Li
>
> **摘要:** Optimizing high-degree of freedom robotic manipulators requires searching complex, high-dimensional configuration spaces, a task that is computationally challenging for classical methods. This paper introduces a quantum native framework that integrates quantum machine learning with Grover's algorithm to solve kinematic optimization problems efficiently. A parameterized quantum circuit is trained to approximate the forward kinematics model, which then constructs an oracle to identify optimal configurations. Grover's algorithm leverages this oracle to provide a quadratic reduction in search complexity. Demonstrated on simulated 1-DoF, 2-DoF, and dual-arm manipulator tasks, the method achieves significant speedups-up to 93x over classical optimizers like Nelder Mead as problem dimensionality increases. This work establishes a foundational, quantum-native framework for robot kinematic optimization, effectively bridging quantum computing and robotics problems.
>
---
#### [replaced 004] Optimal Kinematic Synthesis and Prototype Development of Knee Exoskeleton
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2409.02635v3](http://arxiv.org/pdf/2409.02635v3)**

> **作者:** Shashank Mani Gautam; Ekta Singla; Ashish Singla
>
> **摘要:** The range of rotation (RoR) in a knee exoskeleton is a critical factor in rehabilitation, as it directly influences joint mobility, muscle activation, and recovery outcomes. A well-designed RoR ensures that patients achieve near-natural knee kinematics, which is essential for restoring gait patterns and preventing compensatory movements. This paper presents optimal design of one degree of freedom knee exoskeleton. In kinematic analysis, the existing design being represented by nonlinear and nonconvex mathematical functions. To obtain feasible and optimum measurement of the links of knee exoskeleton, an optimization problem is formulated based on the kinematic analysis and average human's leg measurement. The optimized solution increases the range of motion of knee exoskeleton during sit to stand motion by $24 \%$ as compared with inspired design. Furthermore, misalignment study is conducted by comparing the trajectory of human's knee and exoskeleton's knee during sit to stand motion. The joint movement is calculated using marker and camera system. Finally, a prototype of the knee joint exoskeleton is being developed based on optimal dimensions which validate the maximum range of motion achieved during simulation.
>
---
#### [replaced 005] Redistributing Rewards Across Time and Agents for Multi-Agent Reinforcement Learning
- **分类: cs.MA; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.04864v2](http://arxiv.org/pdf/2502.04864v2)**

> **作者:** Aditya Kapoor; Kale-ab Tessera; Mayank Baranwal; Harshad Khadilkar; Jan Peters; Stefano Albrecht; Mingfei Sun
>
> **备注:** 16 pages, 4 figures, 4 tables
>
> **摘要:** Credit assignmen, disentangling each agent's contribution to a shared reward, is a critical challenge in cooperative multi-agent reinforcement learning (MARL). To be effective, credit assignment methods must preserve the environment's optimal policy. Some recent approaches attempt this by enforcing return equivalence, where the sum of distributed rewards must equal the team reward. However, their guarantees are conditional on a learned model's regression accuracy, making them unreliable in practice. We introduce Temporal-Agent Reward Redistribution (TAR$^2$), an approach that decouples credit modeling from this constraint. A neural network learns unnormalized contribution scores, while a separate, deterministic normalization step enforces return equivalence by construction. We demonstrate that this method is equivalent to a valid Potential-Based Reward Shaping (PBRS), which guarantees the optimal policy is preserved regardless of model accuracy. Empirically, on challenging SMACLite and Google Research Football (GRF) benchmarks, TAR$^2$ accelerates learning and achieves higher final performance than strong baselines. These results establish our method as an effective solution for the agent-temporal credit assignment problem.
>
---
#### [replaced 006] Taxonomy and Trends in Reinforcement Learning for Robotics and Control Systems: A Structured Review
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.21758v3](http://arxiv.org/pdf/2510.21758v3)**

> **作者:** Kumater Ter; Ore-Ofe Ajayi; Daniel Udekwe
>
> **摘要:** Reinforcement learning (RL) has become a foundational approach for enabling intelligent robotic behavior in dynamic and uncertain environments. This work presents an in-depth review of RL principles, advanced deep reinforcement learning (DRL) algorithms, and their integration into robotic and control systems. Beginning with the formalism of Markov Decision Processes (MDPs), the study outlines essential elements of the agent-environment interaction and explores core algorithmic strategies including actor-critic methods, value-based learning, and policy gradients. Emphasis is placed on modern DRL techniques such as DDPG, TD3, PPO, and SAC, which have shown promise in solving high-dimensional, continuous control tasks. A structured taxonomy is introduced to categorize RL applications across domains such as locomotion, manipulation, multi-agent coordination, and human-robot interaction, along with training methodologies and deployment readiness levels. The review synthesizes recent research efforts, highlighting technical trends, design patterns, and the growing maturity of RL in real-world robotics. Overall, this work aims to bridge theoretical advances with practical implementations, providing a consolidated perspective on the evolving role of RL in autonomous robotic systems.
>
---
#### [replaced 007] Multi-robot Motion Planning based on Nets-within-Nets Modeling and Simulation
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2304.08772v4](http://arxiv.org/pdf/2304.08772v4)**

> **作者:** Sofia Hustiu; Joaquin Ezpeleta; Cristian Mahulea; Marius Kloetzer
>
> **备注:** [Note for readers] This paper has been extended from a previous submission to 62nd IEEE Conference on Decision and Control, Dec. 13-15, 2023. This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper focuses on designing motion plans for a heterogeneous team of robots that must cooperate to fulfill a global mission. Robots move in an environment that contains some regions of interest, while the specification for the entire team can include avoidance, visits, or sequencing of these regions of interest. The mission is expressed in terms of a Petri net corresponding to an automaton, while each robot is also modeled by a state machine Petri net. The current work brings about the following contributions with respect to existing solutions for related problems. First, we propose a novel model, denoted High-Level robot team Petri Net (HLrtPN) system, to incorporate the specification and robot models into the Nets-within-Nets paradigm. A guard function, named Global Enabling Function, is designed to synchronize the firing of transitions so that robot motions do not violate the specification. Then, the solution is found by simulating the HLrtPN system in a specific software tool that accommodates Nets-within-Nets. Illustrative examples based on Linear Temporal Logic missions support the computational feasibility of the proposed framework.
>
---
#### [replaced 008] Online Adaptation for Flying Quadrotors in Tight Formations
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.17488v2](http://arxiv.org/pdf/2506.17488v2)**

> **作者:** Pei-An Hsieh; Kong Yao Chee; M. Ani Hsieh
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** The task of flying in tight formations is challenging for teams of quadrotors because the complex aerodynamic wake interactions can destabilize individual team members as well as the team. Furthermore, these aerodynamic effects are highly nonlinear and fast-paced, making them difficult to model and predict. To overcome these challenges, we present L1 KNODE-DW MPC, an adaptive, mixed expert learning based control framework that allows individual quadrotors to accurately track trajectories while adapting to time-varying aerodynamic interactions during formation flights. We evaluate L1 KNODE-DW MPC in two different three-quadrotor formations and show that it outperforms several MPC baselines. Our results show that the proposed framework is capable of enabling the three-quadrotor team to remain vertically aligned in close proximity throughout the flight. These findings show that the L1 adaptive module compensates for unmodeled disturbances most effectively when paired with an accurate dynamics model. A video showcasing our framework and the physical experiments is available here: https://youtu.be/9QX1Q5Ut9Rs
>
---
#### [replaced 009] Efficient Path Planning and Task Allocation Algorithm for Boolean Specifications
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.04881v2](http://arxiv.org/pdf/2506.04881v2)**

> **作者:** Ioana Hustiu; Roozbeh Abolpour; Cristian Mahulea; Marius Kloetzer
>
> **摘要:** This paper presents a novel path-planning and task assignment algorithm for multi-robot systems that should fulfill a global Boolean specification. The proposed method is based on Integer Linear Programming (ILP) formulations, which are combined with structural insights from Petri nets to improve scalability and computational efficiency. By proving that the \emph{constraint matrix} is totally unimodular (TU) for certain classes of problems, the ILP formulation can be relaxed into a Linear Programming (LP) problem without losing the integrality of the solution. This relaxation eliminates complex combinatorial techniques, significantly reducing computational overhead and thus ensuring scalability for large-scale systems. Using the approach proposed in this paper, we can solve path-planning problems for teams made up to 500 robots. The method guarantees computational tractability, handles collision avoidance and reduces computational demands through iterative LP optimization techniques. Case studies demonstrate the efficiency of the algorithm in generating scalable, collision-free paths for large robot teams navigating in complex environments. While the conservative nature of collision avoidance introduces additional constraints, and thus, computational requirements, the solution remains practical and impactful for diverse applications. The algorithm is particularly applicable to real-world scenarios, including warehouse logistics where autonomous robots must efficiently coordinate tasks or search-and-rescue operations in various environments. This work contributes both theoretically and practically to scalable multi-robot path planning and task allocation, offering an efficient framework for coordinating autonomous agents in shared environments.
>
---
#### [replaced 010] ES-HPC-MPC: Exponentially Stable Hybrid Perception Constrained MPC for Quadrotor with Suspended Payloads
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2504.08841v2](http://arxiv.org/pdf/2504.08841v2)**

> **作者:** Luis F. Recalde; Mrunal Sarvaiya; Giuseppe Loianno; Guanrui Li
>
> **备注:** Accepted to IEEE Robotics and Automation Letters
>
> **摘要:** Aerial transportation using quadrotors with cable-suspended payloads holds great potential for applications in disaster response, logistics, and infrastructure maintenance. However, their hybrid and underactuated dynamics pose significant control and perception challenges. Traditional approaches often assume a taut cable condition, limiting their effectiveness in real-world applications where slack-to-taut transitions occur due to disturbances. We introduce ES-HPC-MPC, a model predictive control framework that enforces exponential stability and perception-constrained control under hybrid dynamics. Our method leverages Exponentially Stabilizing Control Lyapunov Functions (ES-CLFs) to enforce stability during the tasks and Control Barrier Functions (CBFs) to maintain the payload within the onboard camera's field of view (FoV). We validate our method through both simulation and real-world experiments, demonstrating stable trajectory tracking and reliable payload perception. We validate that our method maintains stability and satisfies perception constraints while tracking dynamically infeasible trajectories and when the system is subjected to hybrid mode transitions caused by unexpected disturbances.
>
---
#### [replaced 011] Dual-Regularized Riccati Recursions for Interior-Point Optimal Control
- **分类: math.OC; cs.MS; cs.RO; cs.SY; eess.SY; 49M37, 90C51, 93B45; G.1.6**

- **链接: [http://arxiv.org/pdf/2509.16370v3](http://arxiv.org/pdf/2509.16370v3)**

> **作者:** João Sousa-Pinto; Dominique Orban
>
> **摘要:** We derive closed-form extensions of Riccati's recursions (both sequential and parallel) for solving dual-regularized LQR problems. We show how these methods can be used to solve general constrained, non-convex, discrete-time optimal control problems via a regularized interior point method, while guaranteeing that each step is a descent direction of an Augmented Barrier-Lagrangian merit function. We provide MIT-licensed implementations of our methods in C++ and JAX.
>
---
#### [replaced 012] RoboOmni: Proactive Robot Manipulation in Omni-modal Context
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23763v2](http://arxiv.org/pdf/2510.23763v2)**

> **作者:** Siyin Wang; Jinlan Fu; Feihong Liu; Xinzhe He; Huangxuan Wu; Junhao Shi; Kexin Huang; Zhaoye Fei; Jingjing Gong; Zuxuan Wu; Yugang Jiang; See-Kiong Ng; Tat-Seng Chua; Xipeng Qiu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.
>
---
#### [replaced 013] Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles
- **分类: cs.CV; cs.AI; cs.ET; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2509.09349v2](http://arxiv.org/pdf/2509.09349v2)**

> **作者:** Ian Nell; Shane Gilroy
>
> **摘要:** Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behaviour classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviours such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioural analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions.
>
---
#### [replaced 014] EquiContact: A Hierarchical SE(3) Vision-to-Force Equivariant Policy for Spatially Generalizable Contact-rich Tasks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.10961v2](http://arxiv.org/pdf/2507.10961v2)**

> **作者:** Joohwan Seo; Arvind Kruthiventy; Soomi Lee; Megan Teng; Xiang Zhang; Seoyeon Choi; Jongeun Choi; Roberto Horowitz
>
> **备注:** Submitted to RA-L
>
> **摘要:** This paper presents a framework for learning vision-based robotic policies for contact-rich manipulation tasks that generalize spatially across task configurations. We focus on achieving robust spatial generalization of the policy for the peg-in-hole (PiH) task trained from a small number of demonstrations. We propose EquiContact, a hierarchical policy composed of a high-level vision planner (Diffusion Equivariant Descriptor Field, Diff-EDF) and a novel low-level compliant visuomotor policy (Geometric Compliant ACT, G-CompACT). G-CompACT operates using only localized observations (geometrically consistent error vectors (GCEV), force-torque readings, and wrist-mounted RGB images) and produces actions defined in the end-effector frame. Through these design choices, we show that the entire EquiContact pipeline is SE(3)-equivariant, from perception to force control. We also outline three key components for spatially generalizable contact-rich policies: compliance, localized policies, and induced equivariance. Real-world experiments on PiH, screwing, and surface wiping tasks demonstrate a near-perfect success rate and robust generalization to unseen spatial configurations, validating the proposed framework and principles. The experimental videos can be found on the project website: https://sites.google.com/berkeley.edu/equicontact
>
---
#### [replaced 015] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.06677v2](http://arxiv.org/pdf/2506.06677v2)**

> **作者:** Songhao Han; Boxiang Qiu; Yue Liao; Siyuan Huang; Chen Gao; Shuicheng Yan; Si Liu
>
> **备注:** 25 pages, 18 figures, Accepted by NeurIPS 2025
>
> **摘要:** Recent advances in vision-language models (VLMs) have enabled instruction-conditioned robotic systems with improved generalization. However, most existing work focuses on reactive System 1 policies, underutilizing VLMs' strengths in semantic reasoning and long-horizon planning. These System 2 capabilities-characterized by deliberative, goal-directed thinking-remain under explored due to the limited temporal scale and structural complexity of current benchmarks. To address this gap, we introduce RoboCerebra, a benchmark for evaluating high-level reasoning in long-horizon robotic manipulation. RoboCerebra includes: (1) a large-scale simulation dataset with extended task horizons and diverse subtask sequences in household environments; (2) a hierarchical framework combining a high-level VLM planner with a low-level vision-language-action (VLA) controller; and (3) an evaluation protocol targeting planning, reflection, and memory through structured System 1-System 2 interaction. The dataset is constructed via a top-down pipeline, where GPT generates task instructions and decomposes them into subtask sequences. Human operators execute the subtasks in simulation, yielding high-quality trajectories with dynamic object variations. Compared to prior benchmarks, RoboCerebra features significantly longer action sequences and denser annotations. We further benchmark state-of-the-art VLMs as System 2 modules and analyze their performance across key cognitive dimensions, advancing the development of more capable and generalizable robotic planners.
>
---
#### [replaced 016] CAT-RRT: Motion Planning that Admits Contact One Link at a Time
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2310.06210v2](http://arxiv.org/pdf/2310.06210v2)**

> **作者:** Nataliya Nechyporenko; Caleb Escobedo; Shreyas Kadekodi; Alessandro Roncone
>
> **摘要:** Current motion planning approaches rely on binary collision checking to evaluate the validity of a state and thereby dictate where the robot is allowed to move. This approach leaves little room for robots to engage in contact with an object, as is often necessary when operating in densely cluttered spaces. In this work, we propose an alternative method that considers contact states as high-cost states that the robot should avoid but can traverse if necessary to complete a task. More specifically, we introduce Contact Admissible Transition-based Rapidly exploring Random Trees (CAT-RRT), a planner that uses a novel per-link cost heuristic to find a path by traversing high-cost obstacle regions. Through extensive testing, we find that state-of-the-art optimization planners tend to over-explore low-cost states, which leads to slow and inefficient convergence to contact regions. Conversely, CAT-RRT searches both low and high-cost regions simultaneously with an adaptive thresholding mechanism carried out at each robot link. This leads to paths with a balance between efficiency, path length, and contact cost.
>
---
#### [replaced 017] SNN-Based Online Learning of Concepts and Action Laws in an Open World
- **分类: cs.AI; cs.LG; cs.NE; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.12308v4](http://arxiv.org/pdf/2411.12308v4)**

> **作者:** Christel Grimaud; Dominique Longin; Andreas Herzig
>
> **摘要:** We present the architecture of a fully autonomous, bio-inspired cognitive agent built around a spiking neural network (SNN) implementing the agent's semantic memory. This agent explores its universe and learns concepts of objects/situations and of its own actions in a one-shot manner. While object/situation concepts are unary, action concepts are triples made up of an initial situation, a motor activity, and an outcome. They embody the agent's knowledge of its universe's action laws. Both kinds of concepts have different degrees of generality. To make decisions the agent queries its semantic memory for the expected outcomes of envisaged actions and chooses the action to take on the basis of these predictions. Our experiments show that the agent handles new situations by appealing to previously learned general concepts and rapidly modifies its concepts to adapt to environment changes.
>
---
