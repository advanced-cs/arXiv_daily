# 机器人 cs.RO

- **最新发布 35 篇**

- **更新 16 篇**

## 最新发布

#### [new 001] $\nabla$-SDF: Learning Euclidean Signed Distance Functions Online with Gradient-Augmented Octree Interpolation and Neural Residual
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出$\nabla$-SDF，用于在线、大尺度下高效重建连续可微的欧氏符号距离函数（SDF）。针对传统方法在效率与精度间的权衡问题，结合梯度增强八叉树显式结构与神经隐式残差，实现非截断SDF的高精度、低内存、实时计算，显著提升机器人感知与规划性能。**

- **链接: [http://arxiv.org/pdf/2510.18999v1](http://arxiv.org/pdf/2510.18999v1)**

> **作者:** Zhirui Dai; Qihao Qian; Tianxing Fan; Nikolay Atanasov
>
> **摘要:** Estimation of signed distance functions (SDFs) from point cloud data has been shown to benefit many robot autonomy capabilities, including localization, mapping, motion planning, and control. Methods that support online and large-scale SDF reconstruction tend to rely on discrete volumetric data structures, which affect the continuity and differentiability of the SDF estimates. Recently, using implicit features, neural network methods have demonstrated high-fidelity and differentiable SDF reconstruction but they tend to be less efficient, can experience catastrophic forgetting and memory limitations in large environments, and are often restricted to truncated SDFs. This work proposes $\nabla$-SDF, a hybrid method that combines an explicit prior obtained from gradient-augmented octree interpolation with an implicit neural residual. Our method achieves non-truncated (Euclidean) SDF reconstruction with computational and memory efficiency comparable to volumetric methods and differentiability and accuracy comparable to neural network methods. Extensive experiments demonstrate that \methodname{} outperforms the state of the art in terms of accuracy and efficiency, providing a scalable solution for downstream tasks in robotics and computer vision.
>
---
#### [new 002] GigaBrain-0: A World Model-Powered Vision-Language-Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GigaBrain-0，一种基于世界模型生成数据的视觉-语言-动作（VLA）基础模型，旨在解决真实机器人数据收集成本高、难以泛化的问题。通过生成多样化仿真数据，减少对真实数据依赖，提升跨任务泛化能力与策略鲁棒性，支持复杂操作任务。同时推出轻量版GigaBrain-0-Small，适用于边缘设备。**

- **链接: [http://arxiv.org/pdf/2510.19430v1](http://arxiv.org/pdf/2510.19430v1)**

> **作者:** GigaBrain Team; Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Haoyun Li; Jie Li; Jiagang Zhu; Lv Feng; Peng Li; Qiuping Deng; Runqi Ouyang; Wenkang Qin; Xinze Chen; Xiaofeng Wang; Yang Wang; Yifan Li; Yilong Li; Yiran Ding; Yuan Xu; Yun Ye; Yukun Zhou; Zhehao Dong; Zhenan Wang; Zhichao Liu; Zheng Zhu
>
> **备注:** https://gigabrain0.github.io/
>
> **摘要:** Training Vision-Language-Action (VLA) models for generalist robots typically requires large-scale real-world robot data, which is expensive and time-consuming to collect. The inefficiency of physical data collection severely limits the scalability, and generalization capacity of current VLA systems. To address this challenge, we introduce GigaBrain-0, a novel VLA foundation model empowered by world model-generated data (e.g., video generation, real2real transfer, human transfer, view transfer, sim2real transfer data). By leveraging world models to generate diverse data at scale, GigaBrain-0 significantly reduces reliance on real robot data while improving cross-task generalization. Our approach further improves policy robustness through RGBD input modeling and embodied Chain-of-Thought (CoT) supervision, enabling the model to reason about spatial geometry, object states, and long-horizon dependencies during task execution. This leads to substantial gains in real-world performance on dexterous, long-horizon, and mobile manipulation tasks. Extensive experiments demonstrate that GigaBrain-0 achieves superior generalization across variations in appearances (e.g., textures, colors), object placements, and camera viewpoints. Additionally, we present GigaBrain-0-Small, an optimized lightweight variant designed to run efficiently on devices such as the NVIDIA Jetson AGX Orin.
>
---
#### [new 003] Towards Proprioceptive Terrain Mapping with Quadruped Robots for Exploration in Planetary Permanently Shadowed Regions
- **分类: cs.RO**

- **简介: 该论文针对月球极区永久阴影区探索任务，提出一种基于四足机器人本体感知的地形映射方法。通过融合内部传感器数据，实时估计地形高程、足端滑移、能耗与稳定性，构建多层2.5D交互式地图，解决传统视觉传感器无法获取物理交互信息的问题。**

- **链接: [http://arxiv.org/pdf/2510.18986v1](http://arxiv.org/pdf/2510.18986v1)**

> **作者:** Alberto Sanchez-Delgado; João Carlos Virgolino Soares; Victor Barasuol; Claudio Semini
>
> **备注:** Published in the Proceedings of the International Conference on Space Robotics (iSpaRo 2025)
>
> **摘要:** Permanently Shadowed Regions (PSRs) near the lunar poles are of interest for future exploration due to their potential to contain water ice and preserve geological records. Their complex, uneven terrain favors the use of legged robots, which can traverse challenging surfaces while collecting in-situ data, and have proven effective in Earth analogs, including dark caves, when equipped with onboard lighting. While exteroceptive sensors like cameras and lidars can capture terrain geometry and even semantic information, they cannot quantify its physical interaction with the robot, a capability provided by proprioceptive sensing. We propose a terrain mapping framework for quadruped robots, which estimates elevation, foot slippage, energy cost, and stability margins from internal sensing during locomotion. These metrics are incrementally integrated into a multi-layer 2.5D gridmap that reflects terrain interaction from the robot's perspective. The system is evaluated in a simulator that mimics a lunar environment, using the 21 kg quadruped robot Aliengo, showing consistent mapping performance under lunar gravity and terrain conditions.
>
---
#### [new 004] SEA: Semantic Map Prediction for Active Exploration of Uncertain Areas
- **分类: cs.RO**

- **简介: 该论文提出SEA方法，用于机器人主动探索未知环境。针对传统方法依赖短期决策导致探索效率低的问题，提出基于语义地图预测的分层强化学习策略，通过迭代预测缺失区域并优化长期探索路径，提升地图构建精度与覆盖率，在有限步数内实现更优探索效果。**

- **链接: [http://arxiv.org/pdf/2510.19766v1](http://arxiv.org/pdf/2510.19766v1)**

> **作者:** Hongyu Ding; Xinyue Liang; Yudong Fang; You Wu; Jieqi Shi; Jing Huo; Wenbin Li; Jing Wu; Yu-Kun Lai; Yang Gao
>
> **摘要:** In this paper, we propose SEA, a novel approach for active robot exploration through semantic map prediction and a reinforcement learning-based hierarchical exploration policy. Unlike existing learning-based methods that rely on one-step waypoint prediction, our approach enhances the agent's long-term environmental understanding to facilitate more efficient exploration. We propose an iterative prediction-exploration framework that explicitly predicts the missing areas of the map based on current observations. The difference between the actual accumulated map and the predicted global map is then used to guide exploration. Additionally, we design a novel reward mechanism that leverages reinforcement learning to update the long-term exploration strategies, enabling us to construct an accurate semantic map within limited steps. Experimental results demonstrate that our method significantly outperforms state-of-the-art exploration strategies, achieving superior coverage ares of the global map within the same time constraints.
>
---
#### [new 005] LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision Language Navigation in Continuous Environments
- **分类: cs.RO**

- **简介: 该论文针对零样本视觉语言导航任务，解决现有方法在未见环境中的泛化能力与大模型推理利用之间的矛盾。提出LaViRA框架，通过语言、视觉、机器人动作的分层分解，协同多模态大模型优势，实现高效、可解释的导航，显著提升性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.19655v1](http://arxiv.org/pdf/2510.19655v1)**

> **作者:** Hongyu Ding; Ziming Xu; Yudong Fang; You Wu; Zixuan Chen; Jieqi Shi; Jing Huo; Yifan Zhang; Yang Gao
>
> **摘要:** Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires an agent to navigate unseen environments based on natural language instructions without any prior training. Current methods face a critical trade-off: either rely on environment-specific waypoint predictors that limit scene generalization, or underutilize the reasoning capabilities of large models during navigation. We introduce LaViRA, a simple yet effective zero-shot framework that addresses this dilemma by decomposing action into a coarse-to-fine hierarchy: Language Action for high-level planning, Vision Action for perceptual grounding, and Robot Action for robust navigation. This modular decomposition allows us to leverage the distinct strengths of different scales of Multimodal Large Language Models (MLLMs) at each stage, creating a system that is powerful in its reasoning, grounding and practical control. LaViRA significantly outperforms existing state-of-the-art methods on the VLN-CE benchmark, demonstrating superior generalization capabilities in unseen environments, while maintaining transparency and efficiency for real-world deployment.
>
---
#### [new 006] Using Temperature Sampling to Effectively Train Robot Learning Policies on Imbalanced Datasets
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对机器人学习中因任务相似导致动作数据不平衡的问题，提出基于温度采样的简单采样策略。通过优化低频动作的训练频率，提升模型在低资源任务上的泛化能力，同时保持高资源任务性能，有效利用多任务模型容量。**

- **链接: [http://arxiv.org/pdf/2510.19373v1](http://arxiv.org/pdf/2510.19373v1)**

> **作者:** Basavasagar Patil; Sydney Belt; Jayjun Lee; Nima Fazeli; Bernadette Bucher
>
> **摘要:** Increasingly large datasets of robot actions and sensory observations are being collected to train ever-larger neural networks. These datasets are collected based on tasks and while these tasks may be distinct in their descriptions, many involve very similar physical action sequences (e.g., 'pick up an apple' versus 'pick up an orange'). As a result, many datasets of robotic tasks are substantially imbalanced in terms of the physical robotic actions they represent. In this work, we propose a simple sampling strategy for policy training that mitigates this imbalance. Our method requires only a few lines of code to integrate into existing codebases and improves generalization. We evaluate our method in both pre-training small models and fine-tuning large foundational models. Our results show substantial improvements on low-resource tasks compared to prior state-of-the-art methods, without degrading performance on high-resource tasks. This enables more effective use of model capacity for multi-task policies. We also further validate our approach in a real-world setup on a Franka Panda robot arm across a diverse set of tasks.
>
---
#### [new 007] Fast Marker Detection for UV-Based Visual Relative Localisation in Agile UAV Swarms
- **分类: cs.RO**

- **简介: 该论文针对敏捷无人机集群的视觉相对定位任务，解决高速、低延迟标记检测难题。提出三种创新方案：优化的CPU流程、GPU着色器程序和等效的FPGA流架构，显著提升处理速度，尤其FPGA实现大幅降低端到端延迟，验证了其在低功耗嵌入式平台上的高效性与可行性。**

- **链接: [http://arxiv.org/pdf/2510.19663v1](http://arxiv.org/pdf/2510.19663v1)**

> **作者:** Vojtěch Vrba; Viktor Walter; Petr Štěpán; Martin Saska
>
> **摘要:** A novel approach for the fast onboard detection of isolated markers for visual relative localisation of multiple teammates in agile UAV swarms is introduced in this paper. As the detection forms a key component of real-time localisation systems, a three-fold innovation is presented, consisting of an optimised procedure for CPUs, a GPU shader program, and a functionally equivalent FPGA streaming architecture. For the proposed CPU and GPU solutions, the mean processing time per pixel of input camera frames was accelerated by two to three orders of magnitude compared to the state of the art. For the localisation task, the proposed FPGA architecture offered the most significant overall acceleration by minimising the total delay from camera exposure to detection results. Additionally, the proposed solutions were evaluated on various 32-bit and 64-bit embedded platforms to demonstrate their efficiency, as well as their feasibility for applications using low-end UAVs and MAVs. Thus, it has become a crucial enabling technology for agile UAV swarming.
>
---
#### [new 008] Convex Maneuver Planning for Spacecraft Collision Avoidance
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对低地球轨道卫星碰撞规避问题，提出一种基于凸优化的短时交会机动规划算法。通过将非凸二次约束问题松弛为凸半定规划，实现最小能耗规避机动的全局最优求解，并在无法满足安全概率时转为最小风险方案。验证表明该方法高效可靠。**

- **链接: [http://arxiv.org/pdf/2510.19058v1](http://arxiv.org/pdf/2510.19058v1)**

> **作者:** Fausto Vega; Jon Arrizabalaga; Ryan Watson; Zachary Manchester
>
> **备注:** 8 pages, 6 figures, Accepted to International Space Robotics Conference
>
> **摘要:** Conjunction analysis and maneuver planning for spacecraft collision avoidance remains a manual and time-consuming process, typically involving repeated forward simulations of hand-designed maneuvers. With the growing density of satellites in low-Earth orbit (LEO), autonomy is becoming essential for efficiently evaluating and mitigating collisions. In this work, we present an algorithm to design low-thrust collision-avoidance maneuvers for short-term conjunction events. We first formulate the problem as a nonconvex quadratically-constrained quadratic program (QCQP), which we then relax into a convex semidefinite program (SDP) using Shor's relaxation. We demonstrate empirically that the relaxation is tight, which enables the recovery of globally optimal solutions to the original nonconvex problem. Our formulation produces a minimum-energy solution while ensuring a desired probability of collision at the time of closest approach. Finally, if the desired probability of collision cannot be satisfied, we relax this constraint into a penalty, yielding a minimum-risk solution. We validate our algorithm with a high-fidelity simulation of a satellite conjunction in low-Earth orbit with a simulated conjunction data message (CDM), demonstrating its effectiveness in reducing collision risk.
>
---
#### [new 009] A Cross-Environment and Cross-Embodiment Path Planning Framework via a Conditional Diffusion Model
- **分类: cs.RO; cs.AI; 68T40 (Primary), 70Q05 (Secondary)**

- **简介: 该论文提出GADGET框架，解决机器人在复杂环境中跨环境、跨机械臂的高效安全路径规划问题。通过条件扩散模型，结合场景编码与控制屏障函数，实现零样本迁移，生成无碰撞关节空间轨迹，支持多类型机器人实时部署。**

- **链接: [http://arxiv.org/pdf/2510.19128v1](http://arxiv.org/pdf/2510.19128v1)**

> **作者:** Mehran Ghafarian Tamizi; Homayoun Honari; Amir Mehdi Soufi Enayati; Aleksey Nozdryn-Plotnicki; Homayoun Najjaran
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** Path planning for a robotic system in high-dimensional cluttered environments needs to be efficient, safe, and adaptable for different environments and hardware. Conventional methods face high computation time and require extensive parameter tuning, while prior learning-based methods still fail to generalize effectively. The primary goal of this research is to develop a path planning framework capable of generalizing to unseen environments and new robotic manipulators without the need for retraining. We present GADGET (Generalizable and Adaptive Diffusion-Guided Environment-aware Trajectory generation), a diffusion-based planning model that generates joint-space trajectories conditioned on voxelized scene representations as well as start and goal configurations. A key innovation is GADGET's hybrid dual-conditioning mechanism that combines classifier-free guidance via learned scene encoding with classifier-guided Control Barrier Function (CBF) safety shaping, integrating environment awareness with real-time collision avoidance directly in the denoising process. This design supports zero-shot transfer to new environments and robotic embodiments without retraining. Experimental results show that GADGET achieves high success rates with low collision intensity in spherical-obstacle, bin-picking, and shelf environments, with CBF guidance further improving safety. Moreover, comparative evaluations indicate strong performance relative to both sampling-based and learning-based baselines. Furthermore, GADGET provides transferability across Franka Panda, Kinova Gen3 (6/7-DoF), and UR5 robots, and physical execution on a Kinova Gen3 demonstrates its ability to generate safe, collision-free trajectories in real-world settings.
>
---
#### [new 010] Imitation Learning Policy based on Multi-Step Consistent Integration Shortcut Model
- **分类: cs.RO**

- **简介: 该论文针对机器人模仿学习中流匹配方法推理慢的问题，提出基于多步一致性融合的单步捷径策略。通过分解单步损失为多步损失并引入自适应梯度分配，提升推理速度与模型稳定性，在仿真与真实场景中均验证了有效性。**

- **链接: [http://arxiv.org/pdf/2510.19356v1](http://arxiv.org/pdf/2510.19356v1)**

> **作者:** Yu Fang; Xinyu Wang; Xuehe Zhang; Wanli Xue; Mingwei Zhang; Shengyong Chen; Jie Zhao
>
> **摘要:** The wide application of flow-matching methods has greatly promoted the development of robot imitation learning. However, these methods all face the problem of high inference time. To address this issue, researchers have proposed distillation methods and consistency methods, but the performance of these methods still struggles to compete with that of the original diffusion models and flow-matching models. In this article, we propose a one-step shortcut method with multi-step integration for robot imitation learning. To balance the inference speed and performance, we extend the multi-step consistency loss on the basis of the shortcut model, split the one-step loss into multi-step losses, and improve the performance of one-step inference. Secondly, to solve the problem of unstable optimization of the multi-step loss and the original flow-matching loss, we propose an adaptive gradient allocation method to enhance the stability of the learning process. Finally, we evaluate the proposed method in two simulation benchmarks and five real-world environment tasks. The experimental results verify the effectiveness of the proposed algorithm.
>
---
#### [new 011] ProTerrain: Probabilistic Physics-Informed Rough Terrain World Modeling
- **分类: cs.RO**

- **简介: 该论文针对非结构化地形中机器人运动预测的不确定性问题，提出ProTerrain框架，通过建模空间相关性异类不确定性，并结合可微分物理引擎实现概率轨迹预测，提升预测精度与可靠性。**

- **链接: [http://arxiv.org/pdf/2510.19364v1](http://arxiv.org/pdf/2510.19364v1)**

> **作者:** Golnaz Raja; Ruslan Agishev; Miloš Prágr; Joni Pajarinen; Karel Zimmermann; Arun Kumar Singh; Reza Ghabcheloo
>
> **备注:** This paper is submitted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Uncertainty-aware robot motion prediction is crucial for downstream traversability estimation and safe autonomous navigation in unstructured, off-road environments, where terrain is heterogeneous and perceptual uncertainty is high. Most existing methods assume deterministic or spatially independent terrain uncertainties, ignoring the inherent local correlations of 3D spatial data and often producing unreliable predictions. In this work, we introduce an efficient probabilistic framework that explicitly models spatially correlated aleatoric uncertainty over terrain parameters as a probabilistic world model and propagates this uncertainty through a differentiable physics engine for probabilistic trajectory forecasting. By leveraging structured convolutional operators, our approach provides high-resolution multivariate predictions at manageable computational cost. Experimental evaluation on a publicly available dataset shows significantly improved uncertainty estimation and trajectory prediction accuracy over aleatoric uncertainty estimation baselines.
>
---
#### [new 012] Underwater Dense Mapping with the First Compact 3D Sonar
- **分类: cs.RO**

- **简介: 该论文研究水下密集建图任务，针对水下缺乏可靠3D感知传感器的问题，首次评估紧凑型3D声呐的性能。提出相机-声呐外参标定方法，构建新型建图与SLAM系统，在复杂水下环境中实现百米级精确重建与定位，揭示声学传播挑战，并公开数据集以推动领域发展。**

- **链接: [http://arxiv.org/pdf/2510.18991v1](http://arxiv.org/pdf/2510.18991v1)**

> **作者:** Chinmay Burgul; Yewei Huang; Michalis Chatzispyrou; Ioannis Rekleitis; Alberto Quattrini Li; Marios Xanthidis
>
> **备注:** 8 pages, 12 figures
>
> **摘要:** In the past decade, the adoption of compact 3D range sensors, such as LiDARs, has driven the developments of robust state-estimation pipelines, making them a standard sensor for aerial, ground, and space autonomy. Unfortunately, poor propagation of electromagnetic waves underwater, has limited the visibility-independent sensing options of underwater state-estimation to acoustic range sensors, which provide 2D information including, at-best, spatially ambiguous information. This paper, to the best of our knowledge, is the first study examining the performance, capacity, and opportunities arising from the recent introduction of the first compact 3D sonar. Towards that purpose, we introduce calibration procedures for extracting the extrinsics between the 3D sonar and a camera and we provide a study on acoustic response in different surfaces and materials. Moreover, we provide novel mapping and SLAM pipelines tested in deployments in underwater cave systems and other geometrically and acoustically challenging underwater environments. Our assessment showcases the unique capacity of 3D sonars to capture consistent spatial information allowing for detailed reconstructions and localization in datasets expanding to hundreds of meters. At the same time it highlights remaining challenges related to acoustic propagation, as found also in other acoustic sensors. Datasets collected for our evaluations would be released and shared with the community to enable further research advancements.
>
---
#### [new 013] Risk Assessment of an Autonomous Underwater Snake Robot in Confined Operations
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对自主水下蛇形机器人在狭窄环境中的作业风险，提出基于贝叶斯方法的风险评估模型，旨在提升任务成功率。研究聚焦于复杂环境下机器人失控风险的量化分析，通过敏感性分析识别关键影响因素，为优化系统设计提供依据。**

- **链接: [http://arxiv.org/pdf/2510.19415v1](http://arxiv.org/pdf/2510.19415v1)**

> **作者:** Abdelrahman Sayed Sayed
>
> **备注:** 9 pages, 6 figures, Accepted for publication in OCEANS 2023 - Limerick
>
> **摘要:** The growing interest in ocean discovery imposes a need for inspection and intervention in confined and demanding environments. Eely's slender shape, in addition to its ability to change its body configurations, makes articulated underwater robots an adequate option for such environments. However, operation of Eely in such environments imposes demanding requirements on the system, as it must deal with uncertain and unstructured environments, extreme environmental conditions, and reduced navigational capabilities. This paper proposes a Bayesian approach to assess the risks of losing Eely during two mission scenarios. The goal of this work is to improve Eely's performance and the likelihood of mission success. Sensitivity analysis results are presented in order to demonstrate the causes having the highest impact on losing Eely.
>
---
#### [new 014] Learning Affordances at Inference-Time for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; 68T40; I.2.9; I.2.8**

- **简介: 该论文针对视觉语言动作模型在执行任务失败后无法动态调整的问题，提出LITEN框架。通过结合高阶视觉语言模型与低阶策略，在推理时利用过往经验迭代优化计划生成，从真实机器人轨迹中学习动作优势，提升长期任务完成能力。**

- **链接: [http://arxiv.org/pdf/2510.19752v1](http://arxiv.org/pdf/2510.19752v1)**

> **作者:** Ameesh Shah; William Chen; Adwait Godbole; Federico Mora; Sanjit A. Seshia; Sergey Levine
>
> **备注:** 7 pages and appendix
>
> **摘要:** Solving complex real-world control tasks often takes multiple tries: if we fail at first, we reflect on what went wrong, and change our strategy accordingly to avoid making the same mistake. In robotics, Vision-Language-Action models (VLAs) offer a promising path towards solving complex control tasks, but lack the ability to contextually and dynamically readjust behavior when they fail to accomplish a task. In this work, we introduce Learning from Inference-Time Execution (LITEN), which connects a VLA low-level policy to a high-level VLM that conditions on past experiences by including them in-context, allowing it to learn the affordances and capabilities of the low-level VLA. Our approach iterates between a reasoning phase that generates and executes plans for the low-level VLA, and an assessment phase that reflects on the resulting execution and draws useful conclusions to be included in future reasoning contexts. Unlike similar approaches to self-refinement in non-robotics domains, LITEN must reflect on unstructured real-world robot trajectories (e.g., raw videos), which requires structured guiderails during assessment. Our experimental results demonstrate LITEN is able to effectively learn from past experience to generate plans that use high-affordance instructions to accomplish long-horizon tasks.
>
---
#### [new 015] Optimizing Prosthetic Wrist Movement: A Model Predictive Control Approach
- **分类: cs.RO**

- **简介: 该论文针对假手腕部运动控制问题，提出一种低计算成本的模型预测控制（MPC）方法。基于欧拉-伯努利梁与拉格朗日力学建立软连续腕部的运动模型，通过仿真与实验验证，显著提升假手操作的灵巧性与自然度，推动智能假肢系统发展。**

- **链接: [http://arxiv.org/pdf/2510.19541v1](http://arxiv.org/pdf/2510.19541v1)**

> **作者:** Francesco Schetter; Shifa Sulaiman; Shoby George; Paolino De Risi; Fanny Ficuciello
>
> **备注:** International Conference on Social Robotics + AI 2025
>
> **摘要:** The integration of advanced control strategies into prosthetic hands is essential to improve their adaptability and performance. In this study, we present an implementation of a Model Predictive Control (MPC) strategy to regulate the motions of a soft continuum wrist section attached to a tendon-driven prosthetic hand with less computational effort. MPC plays a crucial role in enhancing the functionality and responsiveness of prosthetic hands. By leveraging predictive modeling, this approach enables precise movement adjustments while accounting for dynamic user interactions. This advanced control strategy allows for the anticipation of future movements and adjustments based on the current state of the prosthetic device and the intentions of the user. Kinematic and dynamic modelings are performed using Euler-Bernoulli beam and Lagrange methods respectively. Through simulation and experimental validations, we demonstrate the effectiveness of MPC in optimizing wrist articulation and user control. Our findings suggest that this technique significantly improves the prosthetic hand dexterity, making movements more natural and intuitive. This research contributes to the field of robotics and biomedical engineering by offering a promising direction for intelligent prosthetic systems.
>
---
#### [new 016] Hierarchical DLO Routing with Reinforcement Learning and In-Context Vision-language Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对柔性线状物体（如电缆）的长时序路径规划任务，提出基于视觉语言模型与强化学习的分层自主框架。通过语言理解生成多步计划，并结合失败恢复机制提升鲁棒性，显著提升复杂场景下的成功率。**

- **链接: [http://arxiv.org/pdf/2510.19268v1](http://arxiv.org/pdf/2510.19268v1)**

> **作者:** Mingen Li; Houjian Yu; Yixuan Huang; Youngjin Hong; Changhyun Choi
>
> **备注:** 8 pages, 6 figures, 3 tables
>
> **摘要:** Long-horizon routing tasks of deformable linear objects (DLOs), such as cables and ropes, are common in industrial assembly lines and everyday life. These tasks are particularly challenging because they require robots to manipulate DLO with long-horizon planning and reliable skill execution. Successfully completing such tasks demands adapting to their nonlinear dynamics, decomposing abstract routing goals, and generating multi-step plans composed of multiple skills, all of which require accurate high-level reasoning during execution. In this paper, we propose a fully autonomous hierarchical framework for solving challenging DLO routing tasks. Given an implicit or explicit routing goal expressed in language, our framework leverages vision-language models~(VLMs) for in-context high-level reasoning to synthesize feasible plans, which are then executed by low-level skills trained via reinforcement learning. To improve robustness in long horizons, we further introduce a failure recovery mechanism that reorients the DLO into insertion-feasible states. Our approach generalizes to diverse scenes involving object attributes, spatial descriptions, as well as implicit language commands. It outperforms the next best baseline method by nearly 50% and achieves an overall success rate of 92.5% across long-horizon routing scenarios.
>
---
#### [new 017] SHRUMS: Sensor Hallucination for Real-time Underwater Motion Planning with a Compact 3D Sonar
- **分类: cs.RO**

- **简介: 该论文提出SHRUMS系统，解决水下机器人在复杂3D环境中实时自主导航难题。针对3D声呐数据稀疏、能见度差等问题，引入“传感器幻觉”技术，通过虚拟传感器生成优化感知数据，提升导航鲁棒性与实时性能。首次实现基于3D声呐的完整水下自主导航。**

- **链接: [http://arxiv.org/pdf/2510.18996v1](http://arxiv.org/pdf/2510.18996v1)**

> **作者:** Susheel Vadakkekuruppath; Herman B. Amundsen; Jason M. O'Kane; Marios Xanthidis
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Autonomous navigation in 3D is a fundamental problem for autonomy. Despite major advancements in terrestrial and aerial settings due to improved range sensors including LiDAR, compact sensors with similar capabilities for underwater robots have only recently become available, in the form of 3D sonars. This paper introduces a novel underwater 3D navigation pipeline, called SHRUMS (Sensor Hallucination for Robust Underwater Motion planning with 3D Sonar). To the best of the authors' knowledge, SHRUMS is the first underwater autonomous navigation stack to integrate a 3D sonar. The proposed pipeline exhibits strong robustness while operating in complex 3D environments in spite of extremely poor visibility conditions. To accommodate the intricacies of the novel sensor data stream while achieving real-time locally optimal performance, SHRUMS introduces the concept of hallucinating sensor measurements from non-existent sensors with convenient arbitrary parameters, tailored to application specific requirements. The proposed concepts are validated with real 3D sonar sensor data, utilizing real inputs in challenging settings and local maps constructed in real-time. Field deployments validating the proposed approach in full are planned in the very near future.
>
---
#### [new 018] GRASPLAT: Enabling dexterous grasping through novel view synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对多指机器人灵巧抓取任务，解决因缺乏高质量3D数据导致的抓取失败问题。提出GRASPLAT框架，仅用RGB图像训练，通过3D高斯点云合成新视角图像，结合光度损失优化抓取姿态预测，显著提升抓取成功率。**

- **链接: [http://arxiv.org/pdf/2510.19200v1](http://arxiv.org/pdf/2510.19200v1)**

> **作者:** Matteo Bortolon; Nuno Ferreira Duarte; Plinio Moreno; Fabio Poiesi; José Santos-Victor; Alessio Del Bue
>
> **备注:** Accepted IROS 2025
>
> **摘要:** Achieving dexterous robotic grasping with multi-fingered hands remains a significant challenge. While existing methods rely on complete 3D scans to predict grasp poses, these approaches face limitations due to the difficulty of acquiring high-quality 3D data in real-world scenarios. In this paper, we introduce GRASPLAT, a novel grasping framework that leverages consistent 3D information while being trained solely on RGB images. Our key insight is that by synthesizing physically plausible images of a hand grasping an object, we can regress the corresponding hand joints for a successful grasp. To achieve this, we utilize 3D Gaussian Splatting to generate high-fidelity novel views of real hand-object interactions, enabling end-to-end training with RGB data. Unlike prior methods, our approach incorporates a photometric loss that refines grasp predictions by minimizing discrepancies between rendered and real images. We conduct extensive experiments on both synthetic and real-world grasping datasets, demonstrating that GRASPLAT improves grasp success rates up to 36.9% over existing image-based methods. Project page: https://mbortolon97.github.io/grasplat/
>
---
#### [new 019] TARMAC: A Taxonomy for Robot Manipulation in Chemistry
- **分类: cs.RO**

- **简介: 该论文提出TARMAC，一个化学实验机器人操作的分类体系，旨在解决现有自动化系统依赖人工干预、技能难以复用的问题。通过分析教学实验视频，构建了基于功能与执行需求的动作分类框架，支持机器人可执行原语与高层宏指令，促进技能迁移与长周期流程集成。**

- **链接: [http://arxiv.org/pdf/2510.19289v1](http://arxiv.org/pdf/2510.19289v1)**

> **作者:** Kefeng Huang; Jonathon Pipe; Alice E. Martin; Tianyuan Wang; Barnabas A. Franklin; Andy M. Tyrrell; Ian J. S. Fairlamb; Jihong Zhu
>
> **摘要:** Chemistry laboratory automation aims to increase throughput, reproducibility, and safety, yet many existing systems still depend on frequent human intervention. Advances in robotics have reduced this dependency, but without a structured representation of the required skills, autonomy remains limited to bespoke, task-specific solutions with little capacity to transfer beyond their initial design. Current experiment abstractions typically describe protocol-level steps without specifying the robotic actions needed to execute them. This highlights the lack of a systematic account of the manipulation skills required for robots in chemistry laboratories. To address this gap, we introduce TARMAC - a Taxonomy for Robot Manipulation in Chemistry - a domain-specific framework that defines and organizes the core manipulations needed in laboratory practice. Based on annotated teaching-lab demonstrations and supported by experimental validation, TARMAC categorizes actions according to their functional role and physical execution requirements. Beyond serving as a descriptive vocabulary, TARMAC can be instantiated as robot-executable primitives and composed into higher-level macros, enabling skill reuse and supporting scalable integration into long-horizon workflows. These contributions provide a structured foundation for more flexible and autonomous laboratory automation. More information is available at https://tarmac-paper.github.io/
>
---
#### [new 020] Using Non-Expert Data to Robustify Imitation Learning via Offline Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究机器人模仿学习中的数据效率问题。针对专家数据稀缺且难以获取的问题，提出利用非专家数据（如低质量演示）提升模仿学习鲁棒性。通过改进离线强化学习算法，有效融合非专家数据，增强策略泛化能力与任务恢复能力，显著扩展成功初始条件范围。**

- **链接: [http://arxiv.org/pdf/2510.19495v1](http://arxiv.org/pdf/2510.19495v1)**

> **作者:** Kevin Huang; Rosario Scalise; Cleah Winston; Ayush Agrawal; Yunchu Zhang; Rohan Baijal; Markus Grotz; Byron Boots; Benjamin Burchfiel; Hongkai Dai; Masha Itkina; Paarth Shah; Abhishek Gupta
>
> **摘要:** Imitation learning has proven effective for training robots to perform complex tasks from expert human demonstrations. However, it remains limited by its reliance on high-quality, task-specific data, restricting adaptability to the diverse range of real-world object configurations and scenarios. In contrast, non-expert data -- such as play data, suboptimal demonstrations, partial task completions, or rollouts from suboptimal policies -- can offer broader coverage and lower collection costs. However, conventional imitation learning approaches fail to utilize this data effectively. To address these challenges, we posit that with right design decisions, offline reinforcement learning can be used as a tool to harness non-expert data to enhance the performance of imitation learning policies. We show that while standard offline RL approaches can be ineffective at actually leveraging non-expert data under the sparse data coverage settings typically encountered in the real world, simple algorithmic modifications can allow for the utilization of this data, without significant additional assumptions. Our approach shows that broadening the support of the policy distribution can allow imitation algorithms augmented by offline RL to solve tasks robustly, showing considerably enhanced recovery and generalization behavior. In manipulation tasks, these innovations significantly increase the range of initial conditions where learned policies are successful when non-expert data is incorporated. Moreover, we show that these methods are able to leverage all collected data, including partial or suboptimal demonstrations, to bolster task-directed policy performance. This underscores the importance of algorithmic techniques for using non-expert data for robust policy learning in robotics.
>
---
#### [new 021] Safe Active Navigation and Exploration for Planetary Environments Using Proprioceptive Measurements
- **分类: cs.RO**

- **简介: 该论文针对行星环境中未知颗粒地形的自主安全导航问题，提出SAEGT框架。利用本体感知数据，通过高斯过程回归在线评估可通行性，结合反应式控制器实现实时安全探索，无需视觉输入即可可靠导航。**

- **链接: [http://arxiv.org/pdf/2510.19101v1](http://arxiv.org/pdf/2510.19101v1)**

> **作者:** Matthew Jiang; Shipeng Liu; Feifei Qian
>
> **摘要:** Legged robots can sense terrain through force interactions during locomotion, offering more reliable traversability estimates than remote sensing and serving as scouts for guiding wheeled rovers in challenging environments. However, even legged scouts face challenges when traversing highly deformable or unstable terrain. We present Safe Active Exploration for Granular Terrain (SAEGT), a navigation framework that enables legged robots to safely explore unknown granular environments using proprioceptive sensing, particularly where visual input fails to capture terrain deformability. SAEGT estimates the safe region and frontier region online from leg-terrain interactions using Gaussian Process regression for traversability assessment, with a reactive controller for real-time safe exploration and navigation. SAEGT demonstrated its ability to safely explore and navigate toward a specified goal using only proprioceptively estimated traversability in simulation.
>
---
#### [new 022] A Learning-based Model Reference Adaptive Controller Implemented on a Prosthetic Hand Wrist
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对假手腕柔性机构控制精度低、计算成本高的问题，提出一种基于神经网络的模型参考自适应控制器（NN-MRAC）。结合蒂莫申科梁理论建模，通过在线自适应优化腱力，实现高精度实时控制。仿真与实验验证了其高效性与实用性。**

- **链接: [http://arxiv.org/pdf/2510.19068v1](http://arxiv.org/pdf/2510.19068v1)**

> **作者:** Shifa Sulaiman; Mohammad Gohari; Francesco Schetter; Fanny Ficuciello
>
> **备注:** International Conference on Social Robotics + AI
>
> **摘要:** The functionality and natural motion of prosthetic hands remain limited by the challenges in controlling compliant wrist mechanisms. Current control strategies often lack adaptability and incur high computational costs, which impedes real-time deployment in assistive robotics. To address this gap, this study presents a computationally efficient Neural Network (NN)-based Model Reference Adaptive Controller (MRAC) for a tendon-driven soft continuum wrist integrated with a prosthetic hand. The dynamic modeling of the wrist is formulated using Timoshenko beam theory, capturing both shear and bending deformations. The proposed NN-MRAC estimates the required tendon forces from deflection errors and minimizes deviation from a reference model through online adaptation. Simulation results demonstrate improved precision with a root mean square error (RMSE) of $6.14 \times 10^{-4}$ m and a settling time of $3.2$s. Experimental validations confirm real-time applicability, with an average RMSE of $5.66 \times 10^{-3}$ m, steady-state error of $8.05 \times 10^{-3}$ m, and settling time of $1.58$ s. These results highlight the potential of the controller to enhance motion accuracy and responsiveness in soft prosthetic systems, thereby advancing the integration of adaptive intelligent control in wearable assistive devices.
>
---
#### [new 023] Sample-Based Hybrid Mode Control: Asymptotically Optimal Switching of Algorithmic and Non-Differentiable Control Modes
- **分类: cs.RO**

- **简介: 该论文针对机器人控制中的混合模式切换问题，提出一种基于采样的优化方法，实现非光滑与算法型控制模式间的渐近最优切换。通过整数优化建模，高效搜索切换时机与持续时间，支持复杂行为合成，并在真实机器人系统中验证了其在长期规划与高频控制间实时切换的有效性。**

- **链接: [http://arxiv.org/pdf/2510.19074v1](http://arxiv.org/pdf/2510.19074v1)**

> **作者:** Yilang Liu; Haoxiang You; Ian Abraham
>
> **摘要:** This paper investigates a sample-based solution to the hybrid mode control problem across non-differentiable and algorithmic hybrid modes. Our approach reasons about a set of hybrid control modes as an integer-based optimization problem where we select what mode to apply, when to switch to another mode, and the duration for which we are in a given control mode. A sample-based variation is derived to efficiently search the integer domain for optimal solutions. We find our formulation yields strong performance guarantees that can be applied to a number of robotics-related tasks. In addition, our approach is able to synthesize complex algorithms and policies to compound behaviors and achieve challenging tasks. Last, we demonstrate the effectiveness of our approach in real-world robotic examples that require reactive switching between long-term planning and high-frequency control.
>
---
#### [new 024] Kinematic Analysis and Integration of Vision Algorithms for a Mobile Manipulator Employed Inside a Self-Driving Laboratory
- **分类: cs.RO**

- **简介: 该论文针对自驱动实验室中移动操作臂的自主操作任务，解决复杂环境下物体抓取与姿态估计难题。通过DH建模与逆运动学求解实现精确运动控制，融合特征检测与深度引导的位姿估计算法，提升对纹理物体的动态抓取能力，支持人机协同与实验自动化。**

- **链接: [http://arxiv.org/pdf/2510.19081v1](http://arxiv.org/pdf/2510.19081v1)**

> **作者:** Shifa Sulaiman; Tobias Busk Jensen; Stefan Hein Bengtson; Simon Bøgh
>
> **备注:** International Journal of Intelligent Robotics and Applications 2025
>
> **摘要:** Recent advances in robotics and autonomous systems have broadened the use of robots in laboratory settings, including automated synthesis, scalable reaction workflows, and collaborative tasks in self-driving laboratories (SDLs). This paper presents a comprehensive development of a mobile manipulator designed to assist human operators in such autonomous lab environments. Kinematic modeling of the manipulator is carried out based on the Denavit Hartenberg (DH) convention and inverse kinematics solution is determined to enable precise and adaptive manipulation capabilities. A key focus of this research is enhancing the manipulator ability to reliably grasp textured objects as a critical component of autonomous handling tasks. Advanced vision-based algorithms are implemented to perform real-time object detection and pose estimation, guiding the manipulator in dynamic grasping and following tasks. In this work, we integrate a vision method that combines feature-based detection with homography-driven pose estimation, leveraging depth information to represent an object pose as a $2$D planar projection within $3$D space. This adaptive capability enables the system to accommodate variations in object orientation and supports robust autonomous manipulation across diverse environments. By enabling autonomous experimentation and human-robot collaboration, this work contributes to the scalability and reproducibility of next-generation chemical laboratories
>
---
#### [new 025] Motion Planning and Control of an Overactuated 4-Wheel Drive with Constrained Independent Steering
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究过驱动4轮独立转向（4WIS）车辆的运动规划与控制问题，针对车轮无法全向旋转的机械约束，提出约束数学模型与速度空间分区方法，设计兼顾路径跟踪、避障与运动平滑性的规划器，并通过局部反馈控制器实现断点处的平稳过渡。**

- **链接: [http://arxiv.org/pdf/2510.19054v1](http://arxiv.org/pdf/2510.19054v1)**

> **作者:** Shiyu Liu; Ilija Hadzic; Akshay Gupta; Aliasghar Arab
>
> **备注:** 7 pages, 5 figures, 3 tables, video available at https://youtu.be/8l9s2Wb_vec, To appear at IEEE 2025 International Conference on Advanced Robotics
>
> **摘要:** This paper addresses motion planning and con- trol of an overactuated 4-wheel drive train with independent steering (4WIS) where mechanical constraints prevent the wheels from executing full 360-degree rotations (swerve). The configuration space of such a robot is constrained and contains discontinuities that affect the smoothness of the robot motion. We introduce a mathematical formulation of the steering constraints and derive discontinuity planes that partition the velocity space into regions of smooth and efficient motion. We further design the motion planner for path tracking and ob- stacle avoidance that explicitly accounts for swerve constraints and the velocity transition smoothness. The motion controller uses local feedback to generate actuation from the desired velocity, while properly handling the discontinuity crossing by temporarily stopping the motion and repositioning the wheels. We implement the proposed motion planner as an extension to ROS Navigation package and evaluate the system in simulation and on a physical robot.
>
---
#### [new 026] Semantic World Models
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出语义世界模型，将世界建模视为未来帧的视觉问答任务，聚焦于任务相关的语义信息而非像素重建。通过在图像-动作-文本数据上微调视觉语言模型，实现更优的规划与泛化能力，解决了传统像素级预测与实际决策目标不一致的问题。**

- **链接: [http://arxiv.org/pdf/2510.19818v1](http://arxiv.org/pdf/2510.19818v1)**

> **作者:** Jacob Berg; Chuning Zhu; Yanda Bao; Ishan Durugkar; Abhishek Gupta
>
> **摘要:** Planning with world models offers a powerful paradigm for robotic control. Conventional approaches train a model to predict future frames conditioned on current frames and actions, which can then be used for planning. However, the objective of predicting future pixels is often at odds with the actual planning objective; strong pixel reconstruction does not always correlate with good planning decisions. This paper posits that instead of reconstructing future frames as pixels, world models only need to predict task-relevant semantic information about the future. For such prediction the paper poses world modeling as a visual question answering problem about semantic information in future frames. This perspective allows world modeling to be approached with the same tools underlying vision language models. Thus vision language models can be trained as "semantic" world models through a supervised finetuning process on image-action-text data, enabling planning for decision-making while inheriting many of the generalization and robustness properties from the pretrained vision-language models. The paper demonstrates how such a semantic world model can be used for policy improvement on open-ended robotics tasks, leading to significant generalization improvements over typical paradigms of reconstruction-based action-conditional world modeling. Website available at https://weirdlabuw.github.io/swm.
>
---
#### [new 027] From Forecasting to Planning: Policy World Model for Collaborative State-Action Prediction
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出政策世界模型（PWM），解决自动驾驶中世界模型与规划脱节的问题。通过整合建模与规划，利用无动作未来状态预测提升规划性能，实现协同状态-动作预测。引入动态并行令牌生成机制，仅用前视摄像头即达到领先效果。**

- **链接: [http://arxiv.org/pdf/2510.19654v1](http://arxiv.org/pdf/2510.19654v1)**

> **作者:** Zhida Zhao; Talas Fu; Yifan Wang; Lijun Wang; Huchuan Lu
>
> **备注:** Accepted by NuerIPS 2025 (Poster)
>
> **摘要:** Despite remarkable progress in driving world models, their potential for autonomous systems remains largely untapped: the world models are mostly learned for world simulation and decoupled from trajectory planning. While recent efforts aim to unify world modeling and planning in a single framework, the synergistic facilitation mechanism of world modeling for planning still requires further exploration. In this work, we introduce a new driving paradigm named Policy World Model (PWM), which not only integrates world modeling and trajectory planning within a unified architecture, but is also able to benefit planning using the learned world knowledge through the proposed action-free future state forecasting scheme. Through collaborative state-action prediction, PWM can mimic the human-like anticipatory perception, yielding more reliable planning performance. To facilitate the efficiency of video forecasting, we further introduce a dynamically enhanced parallel token generation mechanism, equipped with a context-guided tokenizer and an adaptive dynamic focal loss. Despite utilizing only front camera input, our method matches or exceeds state-of-the-art approaches that rely on multi-view and multi-modal inputs. Code and model weights will be released at https://github.com/6550Zhao/Policy-World-Model.
>
---
#### [new 028] Actor-Free Continuous Control via Structurally Maximizable Q-Functions
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文提出一种无演员的连续控制方法，通过结构化最大化Q函数实现纯价值基学习。针对传统值方法在连续动作空间中不可行的问题，引入特定架构与算法，提升稳定性与效率，在约束动作空间中表现优于经典演员-评论家方法。**

- **链接: [http://arxiv.org/pdf/2510.18828v1](http://arxiv.org/pdf/2510.18828v1)**

> **作者:** Yigit Korkmaz; Urvi Bhuwania; Ayush Jain; Erdem Bıyık
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Value-based algorithms are a cornerstone of off-policy reinforcement learning due to their simplicity and training stability. However, their use has traditionally been restricted to discrete action spaces, as they rely on estimating Q-values for individual state-action pairs. In continuous action spaces, evaluating the Q-value over the entire action space becomes computationally infeasible. To address this, actor-critic methods are typically employed, where a critic is trained on off-policy data to estimate Q-values, and an actor is trained to maximize the critic's output. Despite their popularity, these methods often suffer from instability during training. In this work, we propose a purely value-based framework for continuous control that revisits structural maximization of Q-functions, introducing a set of key architectural and algorithmic choices to enable efficient and stable learning. We evaluate the proposed actor-free Q-learning approach on a range of standard simulation tasks, demonstrating performance and sample efficiency on par with state-of-the-art baselines, without the cost of learning a separate actor. Particularly, in environments with constrained action spaces, where the value functions are typically non-smooth, our method with structural maximization outperforms traditional actor-critic methods with gradient-based maximization. We have released our code at https://github.com/USC-Lira/Q3C.
>
---
#### [new 029] Local Guidance for Configuration-Based Multi-Agent Pathfinding
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文研究配置型多智能体路径规划（MAPF）中的局部引导问题。针对全局引导的局限性，提出在每个智能体附近提供局部时空指引，通过局部重计算提升解质量。实验表明，该方法在不超时的前提下显著改善了LaCAM算法的性能，推动了MAPF领域的进展。**

- **链接: [http://arxiv.org/pdf/2510.19072v1](http://arxiv.org/pdf/2510.19072v1)**

> **作者:** Tomoki Arita; Keisuke Okumura
>
> **备注:** 10 pages
>
> **摘要:** Guidance is an emerging concept that improves the empirical performance of real-time, sub-optimal multi-agent pathfinding (MAPF) methods. It offers additional information to MAPF algorithms to mitigate congestion on a global scale by considering the collective behavior of all agents across the entire workspace. This global perspective helps reduce agents' waiting times, thereby improving overall coordination efficiency. In contrast, this study explores an alternative approach: providing local guidance in the vicinity of each agent. While such localized methods involve recomputation as agents move and may appear computationally demanding, we empirically demonstrate that supplying informative spatiotemporal cues to the planner can significantly improve solution quality without exceeding a moderate time budget. When applied to LaCAM, a leading configuration-based solver, this form of guidance establishes a new performance frontier for MAPF.
>
---
#### [new 030] A Radius of Robust Feasibility Approach to Directional Sensors in Uncertain Terrain
- **分类: math.OC; cs.RO**

- **简介: 该论文针对不确定地形下方向性传感器网络的覆盖优化问题，提出基于鲁棒可行半径的分布式贪心算法。通过精确计算鲁棒可行半径，动态调整传感器方向以提升覆盖效率与鲁棒性，有效应对定位不确定性。**

- **链接: [http://arxiv.org/pdf/2510.19407v1](http://arxiv.org/pdf/2510.19407v1)**

> **作者:** Vanshika Datta; C. Nahak
>
> **摘要:** A sensor has the ability to probe its surroundings. However, uncertainties in its exact location can significantly compromise its sensing performance. The radius of robust feasibility defines the maximum range within which robust feasibility is ensured. This work introduces a novel approach integrating it with the directional sensor networks to enhance coverage using a distributed greedy algorithm. In particular, we provide an exact formula for the radius of robust feasibility of sensors in a directional sensor network. The proposed model strategically orients the sensors in regions with high coverage potential, accounting for robustness in the face of uncertainty. We analyze the algorithm's adaptability in dynamic environments, demonstrating its ability to enhance efficiency and robustness. Experimental results validate its efficacy in maximizing coverage and optimizing sensor orientations, highlighting its practical advantages for real-world scenarios.
>
---
#### [new 031] Macroscopic EEG Reveals Discriminative Low-Frequency Oscillations in Plan-to-Grasp Visuomotor Tasks
- **分类: eess.SP; cs.RO**

- **简介: 该论文研究非侵入式脑电（EEG）在视觉抓握任务中解码抓握类型的问题。针对传统方法难以区分计划与执行阶段的局限，提出新平台结合FBCSP与SVM，发现低频振荡（0.5–8 Hz）在计划与执行阶段均能有效区分精密抓与力量抓，准确率达75.3–77.8%，显著优于MRCP方法。**

- **链接: [http://arxiv.org/pdf/2510.19057v1](http://arxiv.org/pdf/2510.19057v1)**

> **作者:** Anna Cetera; Sima Ghafoori; Ali Rabiee; Mohammad Hassan Farhadi; Yalda Shahriari; Reza Abiri
>
> **备注:** 12 pages, 8 figures, 1 table
>
> **摘要:** The vision-based grasping brain network integrates visual perception with cognitive and motor processes for visuomotor tasks. While invasive recordings have successfully decoded localized neural activity related to grasp type planning and execution, macroscopic neural activation patterns captured by noninvasive electroencephalography (EEG) remain far less understood. We introduce a novel vision-based grasping platform to investigate grasp-type-specific (precision, power, no-grasp) neural activity across large-scale brain networks using EEG neuroimaging. The platform isolates grasp-specific planning from its associated execution phases in naturalistic visuomotor tasks, where the Filter-Bank Common Spatial Pattern (FBCSP) technique was designed to extract discriminative frequency-specific features within each phase. Support vector machine (SVM) classification discriminated binary (precision vs. power, grasp vs. no-grasp) and multiclass (precision vs. power vs. no-grasp) scenarios for each phase, and were compared against traditional Movement-Related Cortical Potential (MRCP) methods. Low-frequency oscillations (0.5-8 Hz) carry grasp-related information established during planning and maintained throughout execution, with consistent classification performance across both phases (75.3-77.8\%) for precision vs. power discrimination, compared to 61.1\% using MRCP. Higher-frequency activity (12-40 Hz) showed phase-dependent results with 93.3\% accuracy for grasp vs. no-grasp classification but 61.2\% for precision vs. power discrimination. Feature importance using SVM coefficients identified discriminative features within frontoparietal networks during planning and motor networks during execution. This work demonstrated the role of low-frequency oscillations in decoding grasp type during planning using noninvasive EEG.
>
---
#### [new 032] Robust Driving QA through Metadata-Grounded Context and Task-Specific Prompts
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对自动驾驶中的高阶视觉问答任务，提出两阶段视觉语言模型框架。通过多视角图像与历史帧输入，结合链式思维与自一致性推理提升准确性；第二阶段引入场景元数据和任务专用提示，显著增强鲁棒性与性能，在噪声环境下仍保持96%准确率。**

- **链接: [http://arxiv.org/pdf/2510.19001v1](http://arxiv.org/pdf/2510.19001v1)**

> **作者:** Seungjun Yu; Junsung Park; Youngsun Lim; Hyunjung Shim
>
> **摘要:** We present a two-phase vision-language QA system for autonomous driving that answers high-level perception, prediction, and planning questions. In Phase-1, a large multimodal LLM (Qwen2.5-VL-32B) is conditioned on six-camera inputs, a short temporal window of history, and a chain-of-thought prompt with few-shot exemplars. A self-consistency ensemble (multiple sampled reasoning chains) further improves answer reliability. In Phase-2, we augment the prompt with nuScenes scene metadata (object annotations, ego-vehicle state, etc.) and category-specific question instructions (separate prompts for perception, prediction, planning tasks). In experiments on a driving QA benchmark, our approach significantly outperforms the baseline Qwen2.5 models. For example, using 5 history frames and 10-shot prompting in Phase-1 yields 65.1% overall accuracy (vs.62.61% with zero-shot); applying self-consistency raises this to 66.85%. Phase-2 achieves 67.37% overall. Notably, the system maintains 96% accuracy under severe visual corruption. These results demonstrate that carefully engineered prompts and contextual grounding can greatly enhance high-level driving QA with pretrained vision-language models.
>
---
#### [new 033] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文针对长时序、高记忆需求的具身智能任务，提出Memo框架，通过周期性摘要标记实现高效记忆压缩与检索。解决了Transformer在长上下文下内存与计算开销大的问题，提升了模型在复杂环境中的泛化性与实时性表现。**

- **链接: [http://arxiv.org/pdf/2510.19732v1](http://arxiv.org/pdf/2510.19732v1)**

> **作者:** Gunshi Gupta; Karmesh Yadav; Zsolt Kira; Yarin Gal; Rahaf Aljundi
>
> **备注:** Accepted for Spotlight Presentation at NeurIPS 2025
>
> **摘要:** To enable embodied agents to operate effectively over extended timeframes, it is crucial to develop models that form and access memories to stay contextualized in their environment. In the current paradigm of training transformer-based policies for embodied sequential decision-making tasks, visual inputs often overwhelm the context limits of transformers, while humans can maintain and utilize a lifetime of experience compressed as memories. Significant compression is possible in principle, as much of the input is irrelevant and can be abstracted. However, existing approaches predominantly focus on either recurrent models with fixed-size memory or transformers with full-context reliance. In this work, we propose Memo, a transformer-based architecture and training recipe for reinforcement learning (RL) on memory-intensive, long-horizon tasks. Memo incorporates the creation and retrieval of memory by interleaving periodic summarization tokens with the inputs of a model during training. We demonstrate Memo's effectiveness on a gridworld meta-RL benchmark and a multi-object navigation task in photo-realistic indoor settings. Memo outperforms naive long-context transformer baselines while being more compute and storage efficient. Additionally, Memo generalizes better to longer contexts at inference time and remains robust in streaming settings, where historical context must be truncated to fit inference constraints.
>
---
#### [new 034] ConvXformer: Differentially Private Hybrid ConvNeXt-Transformer for Inertial Navigation
- **分类: cs.LG; cs.CR; cs.RO; 68T07, 68T05, 68P27, 62M10; I.2.6; I.5.1; I.2.9; K.4.1; K.6.5; C.3; G.3**

- **简介: 该论文针对GPS拒止环境下惯性导航中的隐私安全问题，提出ConvXformer模型，融合ConvNeXt与Transformer架构，结合自适应梯度裁剪和梯度对齐噪声注入的差分隐私机制，实现高精度定位与隐私保护的平衡。**

- **链接: [http://arxiv.org/pdf/2510.19352v1](http://arxiv.org/pdf/2510.19352v1)**

> **作者:** Omer Tariq; Muhammad Bilal; Muneeb Ul Hassan; Dongsoo Han; Jon Crowcroft
>
> **备注:** 14 pages, 8 figures, 3 tables
>
> **摘要:** Data-driven inertial sequence learning has revolutionized navigation in GPS-denied environments, offering superior odometric resolution compared to traditional Bayesian methods. However, deep learning-based inertial tracking systems remain vulnerable to privacy breaches that can expose sensitive training data. \hl{Existing differential privacy solutions often compromise model performance by introducing excessive noise, particularly in high-frequency inertial measurements.} In this article, we propose ConvXformer, a hybrid architecture that fuses ConvNeXt blocks with Transformer encoders in a hierarchical structure for robust inertial navigation. We propose an efficient differential privacy mechanism incorporating adaptive gradient clipping and gradient-aligned noise injection (GANI) to protect sensitive information while ensuring model performance. Our framework leverages truncated singular value decomposition for gradient processing, enabling precise control over the privacy-utility trade-off. Comprehensive performance evaluations on benchmark datasets (OxIOD, RIDI, RoNIN) demonstrate that ConvXformer surpasses state-of-the-art methods, achieving more than 40% improvement in positioning accuracy while ensuring $(\epsilon,\delta)$-differential privacy guarantees. To validate real-world performance, we introduce the Mech-IO dataset, collected from the mechanical engineering building at KAIST, where intense magnetic fields from industrial equipment induce significant sensor perturbations. This demonstrated robustness under severe environmental distortions makes our framework well-suited for secure and intelligent navigation in cyber-physical systems.
>
---
#### [new 035] Background Fades, Foreground Leads: Curriculum-Guided Background Pruning for Efficient Foreground-Centric Collaborative Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对车载协同感知中的带宽受限问题，提出FadeLead框架。通过课程学习策略，将背景上下文融入前景特征中，实现高效前景主导的特征共享，提升感知性能。**

- **链接: [http://arxiv.org/pdf/2510.19250v1](http://arxiv.org/pdf/2510.19250v1)**

> **作者:** Yuheng Wu; Xiangbo Gao; Quang Tau; Zhengzhong Tu; Dongman Lee
>
> **摘要:** Collaborative perception enhances the reliability and spatial coverage of autonomous vehicles by sharing complementary information across vehicles, offering a promising solution to long-tail scenarios that challenge single-vehicle perception. However, the bandwidth constraints of vehicular networks make transmitting the entire feature map impractical. Recent methods, therefore, adopt a foreground-centric paradigm, transmitting only predicted foreground-region features while discarding the background, which encodes essential context. We propose FadeLead, a foreground-centric framework that overcomes this limitation by learning to encapsulate background context into compact foreground features during training. At the core of our design is a curricular learning strategy that leverages background cues early on but progressively prunes them away, forcing the model to internalize context into foreground representations without transmitting background itself. Extensive experiments on both simulated and real-world benchmarks show that FadeLead outperforms prior methods under different bandwidth settings, underscoring the effectiveness of context-enriched foreground sharing.
>
---
## 更新

#### [replaced 001] Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.17111v2](http://arxiv.org/pdf/2510.17111v2)**

> **作者:** Weifan Guan; Qinghao Hu; Aosheng Li; Jian Cheng
>
> **摘要:** Vision-Language-Action (VLA) models extend vision-language models to embodied control by mapping natural-language instructions and visual observations to robot actions. Despite their capabilities, VLA systems face significant challenges due to their massive computational and memory demands, which conflict with the constraints of edge platforms such as on-board mobile manipulators that require real-time performance. Addressing this tension has become a central focus of recent research. In light of the growing efforts toward more efficient and scalable VLA systems, this survey provides a systematic review of approaches for improving VLA efficiency, with an emphasis on reducing latency, memory footprint, and training and inference costs. We categorize existing solutions into four dimensions: model architecture, perception feature, action generation, and training/inference strategies, summarizing representative techniques within each category. Finally, we discuss future trends and open challenges, highlighting directions for advancing efficient embodied intelligence.
>
---
#### [replaced 002] Open-World Drone Active Tracking with Goal-Centered Rewards
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.00744v2](http://arxiv.org/pdf/2412.00744v2)**

> **作者:** Haowei Sun; Jinwu Hu; Zhirui Zhang; Haoyuan Tian; Xinze Xie; Yufeng Wang; Xiaohua Xie; Yun Lin; Zhuliang Yu; Mingkui Tan
>
> **备注:** NeurIPS 2025
>
> **摘要:** Drone Visual Active Tracking aims to autonomously follow a target object by controlling the motion system based on visual observations, providing a more practical solution for effective tracking in dynamic environments. However, accurate Drone Visual Active Tracking using reinforcement learning remains challenging due to the absence of a unified benchmark and the complexity of open-world environments with frequent interference. To address these issues, we pioneer a systematic solution. First, we propose DAT, the first open-world drone active air-to-ground tracking benchmark. It encompasses 24 city-scale scenes, featuring targets with human-like behaviors and high-fidelity dynamics simulation. DAT also provides a digital twin tool for unlimited scene generation. Additionally, we propose a novel reinforcement learning method called GC-VAT, which aims to improve the performance of drone tracking targets in complex scenarios. Specifically, we design a Goal-Centered Reward to provide precise feedback across viewpoints to the agent, enabling it to expand perception and movement range through unrestricted perspectives. Inspired by curriculum learning, we introduce a Curriculum-Based Training strategy that progressively enhances the tracking performance in complex environments. Besides, experiments on simulator and real-world images demonstrate the superior performance of GC-VAT, achieving a Tracking Success Rate of approximately 72% on the simulator. The benchmark and code are available at https://github.com/SHWplus/DAT_Benchmark.
>
---
#### [replaced 003] MoTVLA: A Vision-Language-Action Model with Unified Fast-Slow Reasoning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.18337v2](http://arxiv.org/pdf/2510.18337v2)**

> **作者:** Wenhui Huang; Changhe Chen; Han Qi; Chen Lv; Yilun Du; Heng Yang
>
> **摘要:** Integrating visual-language instructions into visuomotor policies is gaining momentum in robot learning for enhancing open-world generalization. Despite promising advances, existing approaches face two challenges: limited language steerability when no generated reasoning is used as a condition, or significant inference latency when reasoning is incorporated.In this work, we introduce MoTVLA, a mixture-of-transformers (MoT)-based vision-language-action (VLA) model that integrates fast-slow unified reasoning with behavior policy learning. MoTVLA preserves the general intelligence of pre-trained VLMs (serving as the generalist) for tasks such as perception, scene understanding, and semantic planning, while incorporating a domain expert, a second transformer that shares knowledge with the pretrained VLM, to generate domain-specific fast reasoning (e.g., robot motion decomposition), thereby improving policy execution efficiency. By conditioning the action expert on decomposed motion instructions, MoTVLA can learn diverse behaviors and substantially improve language steerability. Extensive evaluations across natural language processing benchmarks, robotic simulation environments, and real-world experiments confirm the superiority of MoTVLA in both fast-slow reasoning and manipulation task performance.
>
---
#### [replaced 004] Action Tokenizer Matters in In-Context Imitation Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01206v3](http://arxiv.org/pdf/2503.01206v3)**

> **作者:** An Dinh Vuong; Minh Nhat Vu; Dong An; Ian Reid
>
> **备注:** IROS 2025
>
> **摘要:** In-context imitation learning (ICIL) is a new paradigm that enables robots to generalize from demonstrations to unseen tasks without retraining. A well-structured action representation is the key to capturing demonstration information effectively, yet action tokenizer (the process of discretizing and encoding actions) remains largely unexplored in ICIL. In this work, we first systematically evaluate existing action tokenizer methods in ICIL and reveal a critical limitation: while they effectively encode action trajectories, they fail to preserve temporal smoothness, which is crucial for stable robotic execution. To address this, we propose LipVQ-VAE, a variational autoencoder that enforces the Lipschitz condition in the latent action space via weight normalization. By propagating smoothness constraints from raw action inputs to a quantized latent codebook, LipVQ-VAE generates more stable and smoother actions. When integrating into ICIL, LipVQ-VAE improves performance by more than 5.3% in high-fidelity simulators, with real-world experiments confirming its ability to produce smoother, more reliable trajectories. Code and checkpoints are available at https://action-tokenizer-matters.github.io/
>
---
#### [replaced 005] Towards foundational LiDAR world models with efficient latent flow matching
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.23434v2](http://arxiv.org/pdf/2506.23434v2)**

> **作者:** Tianran Liu; Shengwen Zhao; Nicholas Rhinehart
>
> **备注:** Accepted to the Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS 2025), 25 pages, 13 figures
>
> **摘要:** LiDAR-based world models offer more structured and geometry-aware representations than their image-based counterparts. However, existing LiDAR world models are narrowly trained; each model excels only in the domain for which it was built. Can we develop LiDAR world models that exhibit strong transferability across multiple domains? We conduct the first systematic domain transfer study across three demanding scenarios: (i) outdoor to indoor generalization, (ii) sparse-beam & dense-beam adaptation, and (iii) non-semantic to semantic transfer. Given different amounts of fine-tuning data, our experiments show that a single pre-trained model can achieve up to 11% absolute improvement (83% relative) over training from scratch and outperforms training from scratch in 30/36 of our comparisons. This transferability of dynamic learning significantly reduces the reliance on manually annotated data for semantic occupancy forecasting: our method exceed the previous semantic occupancy forecasting models with only 5% of the labeled training data required by prior models. We also observed inefficiencies of current LiDAR world models, mainly through their under-compression of LiDAR data and inefficient training objectives. To address this, we propose a latent conditional flow matching (CFM)-based frameworks that achieves state-of-the-art reconstruction accuracy using only half the training data and a compression ratio 6 times higher than that of prior methods. Our model achieves SOTA performance on future-trajectory-conditioned semantic occupancy forecasting while being 23x more computationally efficient (a 28x FPS speedup); and achieves SOTA performance on semantic occupancy forecasting while being 2x more computationally efficient (a 1.1x FPS speedup).
>
---
#### [replaced 006] Flow with the Force Field: Learning 3D Compliant Flow Matching Policies from Force and Demonstration-Guided Simulation Data
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.02738v2](http://arxiv.org/pdf/2510.02738v2)**

> **作者:** Tianyu Li; Yihan Li; Zizhe Zhang; Nadia Figueroa
>
> **摘要:** While visuomotor policy has made advancements in recent years, contact-rich tasks still remain a challenge. Robotic manipulation tasks that require continuous contact demand explicit handling of compliance and force. However, most visuomotor policies ignore compliance, overlooking the importance of physical interaction with the real world, often leading to excessive contact forces or fragile behavior under uncertainty. Introducing force information into vision-based imitation learning could help improve awareness of contacts, but could also require a lot of data to perform well. One remedy for data scarcity is to generate data in simulation, yet computationally taxing processes are required to generate data good enough not to suffer from the Sim2Real gap. In this work, we introduce a framework for generating force-informed data in simulation, instantiated by a single human demonstration, and show how coupling with a compliant policy improves the performance of a visuomotor policy learned from synthetic data. We validate our approach on real-robot tasks, including non-prehensile block flipping and a bi-manual object moving, where the learned policy exhibits reliable contact maintenance and adaptation to novel conditions. Project Website: https://flow-with-the-force-field.github.io/webpage/
>
---
#### [replaced 007] ComDrive: Comfort-Oriented End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.05051v2](http://arxiv.org/pdf/2410.05051v2)**

> **作者:** Junming Wang; Xingyu Zhang; Zebin Xing; Songen Gu; Xiaoyang Guo; Yang Hu; Ziying Song; Qian Zhang; Xiaoxiao Long; Wei Yin
>
> **备注:** IROS 2025
>
> **摘要:** We propose ComDrive: the first comfort-oriented end-to-end autonomous driving system to generate temporally consistent and comfortable trajectories. Recent studies have demonstrated that imitation learning-based planners and learning-based trajectory scorers can effectively generate and select safety trajectories that closely mimic expert demonstrations. However, such trajectory planners and scorers face the challenge of generating temporally inconsistent and uncomfortable trajectories. To address these issues, ComDrive first extracts 3D spatial representations through sparse perception, which then serves as conditional inputs. These inputs are used by a Conditional Denoising Diffusion Probabilistic Model (DDPM)-based motion planner to generate temporally consistent multi-modal trajectories. A dual-stream adaptive trajectory scorer subsequently selects the most comfortable trajectory from these candidates to control the vehicle. Experiments demonstrate that ComDrive achieves state-of-the-art performance in both comfort and safety, outperforming UniAD by 17% in driving comfort and reducing collision rates by 25% compared to SparseDrive. More results are available on our project page: https://jmwang0117.github.io/ComDrive/.
>
---
#### [replaced 008] RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Interactive Environmental Learning in Physical Embodied Systems
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.01415v5](http://arxiv.org/pdf/2508.01415v5)**

> **作者:** Mingcong Lei; Honghao Cai; Zezhou Cui; Liangchen Tan; Junkun Hong; Gehan Hu; Shuangyu Zhu; Yimou Wu; Shaohan Jiang; Ge Wang; Yuyuan Yang; Junyuan Tan; Zhenglin Wan; Zhen Li; Shuguang Cui; Yiming Zhao; Yatong Han
>
> **摘要:** Embodied agents face persistent challenges in real-world environments, including partial observability, limited spatial reasoning, and high-latency multi-memory integration. We present RoboMemory, a brain-inspired framework that unifies Spatial, Temporal, Episodic, and Semantic memory under a parallelized architecture for efficient long-horizon planning and interactive environmental learning. A dynamic spatial knowledge graph (KG) ensures scalable and consistent memory updates, while a closed-loop planner with a critic module supports adaptive decision-making in dynamic settings. Experiments on EmbodiedBench show that RoboMemory, built on Qwen2.5-VL-72B-Ins, improves average success rates by 25% over its baseline and exceeds the closed-source state-of-the-art (SOTA) Gemini-1.5-Pro by 3%. Real-world trials further confirm its capacity for cumulative learning, with performance improving across repeated tasks. These results highlight RoboMemory as a scalable foundation for memory-augmented embodied intelligence, bridging the gap between cognitive neuroscience and robotic autonomy.
>
---
#### [replaced 009] SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18648v4](http://arxiv.org/pdf/2509.18648v4)**

> **作者:** Yarden As; Chengrui Qu; Benjamin Unger; Dongho Kang; Max van der Hart; Laixi Shi; Stelian Coros; Adam Wierman; Andreas Krause
>
> **摘要:** Deploying reinforcement learning (RL) safely in the real world is challenging, as policies trained in simulators must face the inevitable sim-to-real gap. Robust safe RL techniques are provably safe, however difficult to scale, while domain randomization is more practical yet prone to unsafe behaviors. We address this gap by proposing SPiDR, short for Sim-to-real via Pessimistic Domain Randomization -- a scalable algorithm with provable guarantees for safe sim-to-real transfer. SPiDR uses domain randomization to incorporate the uncertainty about the sim-to-real gap into the safety constraints, making it versatile and highly compatible with existing training pipelines. Through extensive experiments on sim-to-sim benchmarks and two distinct real-world robotic platforms, we demonstrate that SPiDR effectively ensures safety despite the sim-to-real gap while maintaining strong performance.
>
---
#### [replaced 010] Improving planning and MBRL with temporally-extended actions
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.15754v2](http://arxiv.org/pdf/2505.15754v2)**

> **作者:** Palash Chatterjee; Roni Khardon
>
> **备注:** NeurIPS 2025. For project website, see https://pecey.github.io/MBRL-with-TEA/
>
> **摘要:** Continuous time systems are often modeled using discrete time dynamics but this requires a small simulation step to maintain accuracy. In turn, this requires a large planning horizon which leads to computationally demanding planning problems and reduced performance. Previous work in model-free reinforcement learning has partially addressed this issue using action repeats where a policy is learned to determine a discrete action duration. Instead we propose to control the continuous decision timescale directly by using temporally-extended actions and letting the planner treat the duration of the action as an additional optimization variable along with the standard action variables. This additional structure has multiple advantages. It speeds up simulation time of trajectories and, importantly, it allows for deep horizon search in terms of primitive actions while using a shallow search depth in the planner. In addition, in the model-based reinforcement learning (MBRL) setting, it reduces compounding errors from model learning and improves training time for models. We show that this idea is effective and that the range for action durations can be automatically selected using a multi-armed bandit formulation and integrated into the MBRL framework. An extensive experimental evaluation both in planning and in MBRL, shows that our approach yields faster planning, better solutions, and that it enables solutions to problems that are not solved in the standard formulation.
>
---
#### [replaced 011] AttentionSwarm: Reinforcement Learning with Attention Control Barier Function for Crazyflie Drones in Dynamic Environments
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2503.07376v2](http://arxiv.org/pdf/2503.07376v2)**

> **作者:** Grik Tadevosyan; Valerii Serpiva; Aleksey Fedoseev; Roohan Ahmed Khan; Demetros Aschu; Faryal Batool; Nickolay Efanov; Artem Mikhaylov; Dzmitry Tsetserukou
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** We introduce AttentionSwarm, a novel benchmark designed to evaluate safe and efficient swarm control in a dynamic drone racing scenario. Central to our approach is the Attention Model-Based Control Barrier Function (CBF) framework, which integrates attention mechanisms with safety-critical control theory to enable real-time collision avoidance and trajectory optimization. This framework dynamically prioritizes critical obstacles and agents in the swarm's vicinity using attention weights, while CBFs formally guarantee safety by enforcing collision-free constraints. The AttentionSwarm algorithm was developed and evaluated using a swarm of Crazyflie 2.1 micro quadrotors, which were tested indoors with the Vicon motion capture system to ensure precise localization and control. Experimental results show that our system achieves a 95-100% collision-free navigation rate in a dynamic multi-agent drone racing environment, underscoring its effectiveness and robustness in real-world scenarios. This work offers a promising foundation for safe, high-speed multi-robot applications in logistics, inspection, and racing.
>
---
#### [replaced 012] Coordinated Strategies in Realistic Air Combat by Hierarchical Multi-Agent Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.HC; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2510.11474v2](http://arxiv.org/pdf/2510.11474v2)**

> **作者:** Ardian Selmonaj; Giacomo Del Rio; Adrian Schneider; Alessandro Antonucci
>
> **备注:** 2025 IEEE International Conference on Agentic AI (ICA)
>
> **摘要:** Achieving mission objectives in a realistic simulation of aerial combat is highly challenging due to imperfect situational awareness and nonlinear flight dynamics. In this work, we introduce a novel 3D multi-agent air combat environment and a Hierarchical Multi-Agent Reinforcement Learning framework to tackle these challenges. Our approach combines heterogeneous agent dynamics, curriculum learning, league-play, and a newly adapted training algorithm. To this end, the decision-making process is organized into two abstraction levels: low-level policies learn precise control maneuvers, while high-level policies issue tactical commands based on mission objectives. Empirical results show that our hierarchical approach improves both learning efficiency and combat performance in complex dogfight scenarios.
>
---
#### [replaced 013] On the Importance of Tactile Sensing for Imitation Learning: A Case Study on Robotic Match Lighting
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.13618v2](http://arxiv.org/pdf/2504.13618v2)**

> **作者:** Niklas Funk; Changqi Chen; Tim Schneider; Georgia Chalvatzaki; Roberto Calandra; Jan Peters
>
> **摘要:** The field of robotic manipulation has advanced significantly in recent years. At the sensing level, several novel tactile sensors have been developed, capable of providing accurate contact information. On a methodological level, learning from demonstrations has proven an efficient paradigm to obtain performant robotic manipulation policies. The combination of both holds the promise to extract crucial contact-related information from the demonstration data and actively exploit it during policy rollouts. However, this integration has so far been underexplored, most notably in dynamic, contact-rich manipulation tasks where precision and reactivity are essential. This work therefore proposes a multimodal, visuotactile imitation learning framework that integrates a modular transformer architecture with a flow-based generative model, enabling efficient learning of fast and dexterous manipulation policies. We evaluate our framework on the dynamic, contact-rich task of robotic match lighting - a task in which tactile feedback influences human manipulation performance. The experimental results highlight the effectiveness of our approach and show that adding tactile information improves policy performance, thereby underlining their combined potential for learning dynamic manipulation from few demonstrations. Project website: https://sites.google.com/view/tactile-il .
>
---
#### [replaced 014] VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.15530v2](http://arxiv.org/pdf/2510.15530v2)**

> **作者:** Zehao Ni; Yonghao He; Lingfeng Qian; Jilei Mao; Fa Fu; Wei Sui; Hu Su; Junran Peng; Zhipeng Wang; Bin He
>
> **摘要:** In the context of imitation learning, visuomotor-based diffusion policy learning is one of the main directions in robotic manipulation. Most of these approaches rely on point clouds as observation inputs and construct scene representations through point clouds feature learning, which enables them to achieve remarkable accuracy. However, the existing literature lacks an in-depth exploration of vision-only solutions that have significant potential. In this paper, we propose a Vision-Only and single-view Diffusion Policy learning method (VO-DP) that leverages pretrained visual foundation models to achieve effective fusion of semantic and geometric features. We utilize intermediate features from VGGT incorporating semantic features from DINOv2 and geometric features from Alternating Attention blocks. Features are fused via cross-attention and spatially compressed with a CNN to form the input to the policy head. Extensive experiments demonstrate that VO-DP not only outperforms the vision-only baseline DP significantly but also exhibits distinct performance trends against the point cloud-based method DP3: in simulation tasks, VO-DP achieves an average success rate of 64.6% on par with DP3 64.0% and far higher than DP 34.8%, while in real-world tasks, it reaches 87.9%, outperforming both DP3 67.5% and DP 11.2% by a notable margin. Further robustness evaluations confirm that VO-DP remains highly stable under varying conditions including color, size, background, and lighting. Lastly, we open-source a training library for robotic manipulation. Built on Accelerate, this library supports multi-machine and multi-GPU parallel training, as well as mixed precision training. It is compatible with visuomotor policies such as DP, DP3 and VO-DP, and also supports the RoboTwin simulator.
>
---
#### [replaced 015] OmniVIC: A Self-Improving Variable Impedance Controller with Vision-Language In-Context Learning for Safe Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.17150v2](http://arxiv.org/pdf/2510.17150v2)**

> **作者:** Heng Zhang; Wei-Hsing Huang; Gokhan Solak; Arash Ajoudani
>
> **备注:** Code, video and RAG dataset are available at \url{https://sites.google.com/view/omni-vic}
>
> **摘要:** We present OmniVIC, a universal variable impedance controller (VIC) enhanced by a vision language model (VLM), which improves safety and adaptation in any contact-rich robotic manipulation task to enhance safe physical interaction. Traditional VIC have shown advantages when the robot physically interacts with the environment, but lack generalization in unseen, complex, and unstructured safe interactions in universal task scenarios involving contact or uncertainty. To this end, the proposed OmniVIC interprets task context derived reasoning from images and natural language and generates adaptive impedance parameters for a VIC controller. Specifically, the core of OmniVIC is a self-improving Retrieval-Augmented Generation(RAG) and in-context learning (ICL), where RAG retrieves relevant prior experiences from a structured memory bank to inform the controller about similar past tasks, and ICL leverages these retrieved examples and the prompt of current task to query the VLM for generating context-aware and adaptive impedance parameters for the current manipulation scenario. Therefore, a self-improved RAG and ICL guarantee OmniVIC works in universal task scenarios. The impedance parameter regulation is further informed by real-time force/torque feedback to ensure interaction forces remain within safe thresholds. We demonstrate that our method outperforms baselines on a suite of complex contact-rich tasks, both in simulation and on real-world robotic tasks, with improved success rates and reduced force violations. OmniVIC takes a step towards bridging high-level semantic reasoning and low-level compliant control, enabling safer and more generalizable manipulation. Overall, the average success rate increases from 27% (baseline) to 61.4% (OmniVIC).
>
---
#### [replaced 016] RoboGPT-R1: Enhancing Robot Planning with Reinforcement Learning
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.14828v2](http://arxiv.org/pdf/2510.14828v2)**

> **作者:** Jinrui Liu; Bingyan Nie; Boyu Li; Yaran Chen; Yuze Wang; Shunsen He; Haoran Li
>
> **摘要:** Improving the reasoning capabilities of embodied agents is crucial for robots to complete complex human instructions in long-view manipulation tasks successfully. Despite the success of large language models and vision language models based on Supervised Fine-Tuning (SFT) in planning tasks, they continue facing challenges in performing long-horizon manipulation tasks in complex real-world environments, owing to their restricted common sense and reasoning capabilities. Considering that aligning general-purpose vision language models to robotic planning tasks via supervised fine-tuning suffers from poor generalization and insufficient physical understanding, we propose RoboGPT-R1, a two-stage fine-tuning framework for embodied planning. In this framework, supervised training acquires foundational knowledge through expert sequences, followed by RL to address the model's shortcomings in visual-spatial understanding and reasoning. To achieve physical understanding and action sequence consistency in multi-step reasoning tasks, we design a rule-based reward function that simultaneously considers long-horizon performance and action constraint in the environment. The reasoning model, trained on Qwen2.5-VL-3B, significantly outperforms the larger-scale model, GPT-4o-mini, by 21.33% and surpasses other work trained on Qwen2.5-VL-7B by 20.33% on the EmbodiedBench benchmark.
>
---
