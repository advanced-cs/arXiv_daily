# 机器人 cs.RO

- **最新发布 34 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] PrediFlow: A Flow-Based Prediction-Refinement Framework for Real-Time Human Motion Prediction in Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文面向人机协作中实时人体运动预测任务，解决现有方法在真实性、交互感知性及精度-效率平衡上的不足。提出PrediFlow框架，融合人/机器人观测运动，用流匹配进行预测 refinement，在保持不确定性与多模态的同时提升精度并满足实时性。**

- **链接: [https://arxiv.org/pdf/2512.13903v1](https://arxiv.org/pdf/2512.13903v1)**

> **作者:** Sibo Tian; Minghui Zheng; Xiao Liang
>
> **摘要:** Stochastic human motion prediction is critical for safe and effective human-robot collaboration (HRC) in industrial remanufacturing, as it captures human motion uncertainties and multi-modal behaviors that deterministic methods cannot handle. While earlier works emphasize highly diverse predictions, they often generate unrealistic human motions. More recent methods focus on accuracy and real-time performance, yet there remains potential to improve prediction quality further without exceeding time budgets. Additionally, current research on stochastic human motion prediction in HRC typically considers human motion in isolation, neglecting the influence of robot motion on human behavior. To address these research gaps and enable real-time, realistic, and interaction-aware human motion prediction, we propose a novel prediction-refinement framework that integrates both human and robot observed motion to refine the initial predictions produced by a pretrained state-of-the-art predictor. The refinement module employs a Flow Matching structure to account for uncertainty. Experimental studies on the HRC desktop disassembly dataset demonstrate that our method significantly improves prediction accuracy while preserving the uncertainties and multi-modalities of human motion. Moreover, the total inference time of the proposed framework remains within the time budget, highlighting the effectiveness and practicality of our approach.
>
---
#### [new 002] Synthetic Data Pipelines for Adaptive, Mission-Ready Militarized Humanoids
- **分类: cs.RO**

- **简介: 该论文提出合成数据流水线，解决 militarized humanoid 训练慢、实测成本高、泛化性差问题。工作包括：将第一视角空间观测转化为可扩展、任务定制的合成数据集，支持自动标注、快速训练与环境/威胁适配，提升感知、导航、决策及CBRNE等任务鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.14411v1](https://arxiv.org/pdf/2512.14411v1)**

> **作者:** Mohammed Ayman Habib; Aldo Petruzzelli
>
> **备注:** 6 pages; xTech Humanoid white paper submission
>
> **摘要:** Omnia presents a synthetic data driven pipeline to accelerate the training, validation, and deployment readiness of militarized humanoids. The approach converts first-person spatial observations captured from point-of-view recordings, smart glasses, augmented reality headsets, and spatial browsing workflows into scalable, mission-specific synthetic datasets for humanoid autonomy. By generating large volumes of high-fidelity simulated scenarios and pairing them with automated labeling and model training, the pipeline enables rapid iteration on perception, navigation, and decision-making capabilities without the cost, risk, or time constraints of extensive field trials. The resulting datasets can be tuned quickly for new operational environments and threat conditions, supporting both baseline humanoid performance and advanced subsystems such as multimodal sensing, counter-detection survivability, and CBRNE-relevant reconnaissance behaviors. This work targets faster development cycles and improved robustness in complex, contested settings by exposing humanoid systems to broad scenario diversity early in the development process.
>
---
#### [new 003] CoLD Fusion: A Real-time Capable Spline-based Fusion Algorithm for Collective Lane Detection
- **分类: cs.RO**

- **简介: 该论文属自动驾驶环境感知任务，旨在解决单车因传感器局限、遮挡等导致车道检测范围不足的问题。提出CoLD Fusion算法，利用V2V通信融合多车感知数据，基于样条插值实时估计未探测路段车道，将感知范围提升至200%，并保证实时性。**

- **链接: [https://arxiv.org/pdf/2512.14355v1](https://arxiv.org/pdf/2512.14355v1)**

> **作者:** Jörg Gamerdinger; Sven Teufel; Georg Volk; Oliver Bringmann
>
> **备注:** Accepted at IEEE IV 2023
>
> **摘要:** Comprehensive environment perception is essential for autonomous vehicles to operate safely. It is crucial to detect both dynamic road users and static objects like traffic signs or lanes as these are required for safe motion planning. However, in many circumstances a complete perception of other objects or lanes is not achievable due to limited sensor ranges, occlusions, and curves. In scenarios where an accurate localization is not possible or for roads where no HD maps are available, an autonomous vehicle must rely solely on its perceived road information. Thus, extending local sensing capabilities through collective perception using vehicle-to-vehicle communication is a promising strategy that has not yet been explored for lane detection. Therefore, we propose a real-time capable approach for collective perception of lanes using a spline-based estimation of undetected road sections. We evaluate our proposed fusion algorithm in various situations and road types. We were able to achieve real-time capability and extend the perception range by up to 200%.
>
---
#### [new 004] WAM-Diff: A Masked Diffusion VLA Framework with MoE and Online Reinforcement Learning for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出WAM-Diff，一种面向自动驾驶的视觉-语言-动作（VLA）框架，解决端到端轨迹生成问题。它创新性地采用离散掩码扩散建模未来自车轨迹，结合MoE架构与在线强化学习（GSPO），提升预测精度与场景适应性。**

- **链接: [https://arxiv.org/pdf/2512.11872v1](https://arxiv.org/pdf/2512.11872v1)**

> **作者:** Mingwang Xu; Jiahao Cui; Feipeng Cai; Hanlin Shang; Zhihao Zhu; Shan Luan; Yifang Xu; Neng Zhang; Yaoyi Li; Jia Cai; Siyu Zhu
>
> **摘要:** End-to-end autonomous driving systems based on vision-language-action (VLA) models integrate multimodal sensor inputs and language instructions to generate planning and control signals. While autoregressive large language models and continuous diffusion policies are prevalent, the potential of discrete masked diffusion for trajectory generation remains largely unexplored. This paper presents WAM-Diff, a VLA framework that employs masked diffusion to iteratively refine a discrete sequence representing future ego-trajectories. Our approach features three key innovations: a systematic adaptation of masked diffusion for autonomous driving that supports flexible, non-causal decoding orders; scalable model capacity via a sparse MoE architecture trained jointly on motion prediction and driving-oriented visual question answering (VQA); and online reinforcement learning using Group Sequence Policy Optimization (GSPO) to optimize sequence-level driving rewards. Remarkably, our model achieves 91.0 PDMS on NAVSIM-v1 and 89.7 EPDMS on NAVSIM-v2, demonstrating the effectiveness of masked diffusion for autonomous driving. The approach provides a promising alternative to autoregressive and diffusion-based policies, supporting scenario-aware decoding strategies for trajectory generation. The code for this paper will be released publicly at: https://github.com/fudan-generative-vision/WAM-Diff
>
---
#### [new 005] Sample-Efficient Robot Skill Learning for Construction Tasks: Benchmarking Hierarchical Reinforcement Learning and Vision-Language-Action VLA Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文属机器人技能学习任务，旨在解决Construction机器人在少样本下快速适应新任务的难题。作者构建双接口遥操作平台采集数据，三阶段对比VLA模型与DQN等RL方法，验证VLA在泛化性、少样本和部署效率上的优势。**

- **链接: [https://arxiv.org/pdf/2512.14031v1](https://arxiv.org/pdf/2512.14031v1)**

> **作者:** Zhaofeng Hu; Hongrui Yu; Vaidhyanathan Chandramouli; Ci-Jyun Liang
>
> **摘要:** This study evaluates two leading approaches for teaching construction robots new skills to understand their applicability for construction automation: a Vision-Language-Action (VLA) model and Reinforcement Learning (RL) methods. The goal is to understand both task performance and the practical effort needed to deploy each approach on real jobs. The authors developed two teleoperation interfaces to control the robots and collect the demonstrations needed, both of which proved effective for training robots for long-horizon and dexterous tasks. In addition, the authors conduct a three-stage evaluation. First, the authors compare a Multi-Layer Perceptron (MLP) policy with a Deep Q-network (DQN) imitation model to identify the stronger RL baseline, focusing on model performance, generalization, and a pick-up experiment. Second, three different VLA models are trained in two different scenarios and compared with each other. Third, the authors benchmark the selected RL baseline against the VLA model using computational and sample-efficiency measures and then a robot experiment on a multi-stage panel installation task that includes transport and installation. The VLA model demonstrates strong generalization and few-shot capability, achieving 60% and 100% success in the pickup phase. In comparison, DQN can be made robust but needs additional noise during tuning, which increases the workload. Overall, the findings indicate that VLA offers practical advantages for changing tasks by reducing programming effort and enabling useful performance with minimal data, while DQN provides a viable baseline when sufficient tuning effort is acceptable.
>
---
#### [new 006] Fine-Tuning of Neural Network Approximate MPC without Retraining via Bayesian Optimization
- **分类: cs.RO; eess.SY**

- **简介: 该论文属控制领域任务，解决AMPC部署中需手动调参、重训练的难题。提出用贝叶斯优化自动、数据高效地在线微调AMPC策略参数，无需重新训练网络，已在倒立摆与单轮平衡机器人硬件实验中验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.14350v1](https://arxiv.org/pdf/2512.14350v1)**

> **作者:** Henrik Hose; Paul Brunzema; Alexander von Rohr; Alexander Gräfe; Angela P. Schoellig; Sebastian Trimpe
>
> **摘要:** Approximate model-predictive control (AMPC) aims to imitate an MPC's behavior with a neural network, removing the need to solve an expensive optimization problem at runtime. However, during deployment, the parameters of the underlying MPC must usually be fine-tuned. This often renders AMPC impractical as it requires repeatedly generating a new dataset and retraining the neural network. Recent work addresses this problem by adapting AMPC without retraining using approximated sensitivities of the MPC's optimization problem. Currently, this adaption must be done by hand, which is labor-intensive and can be unintuitive for high-dimensional systems. To solve this issue, we propose using Bayesian optimization to tune the parameters of AMPC policies based on experimental data. By combining model-based control with direct and local learning, our approach achieves superior performance to nominal AMPC on hardware, with minimal experimentation. This allows automatic and data-efficient adaptation of AMPC to new system instances and fine-tuning to cost functions that are difficult to directly implement in MPC. We demonstrate the proposed method in hardware experiments for the swing-up maneuver on an inverted cartpole and yaw control of an under-actuated balancing unicycle robot, a challenging control problem.
>
---
#### [new 007] EVOLVE-VLA: Test-Time Training from Environment Feedback for Vision-Language-Action Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EVOLVE-VLA，属具身智能中的视觉-语言-动作（VLA）模型自适应学习任务。旨在解决VLA模型依赖大量示范、缺乏在线环境反馈下的持续适应能力问题。通过学习进展估计器与累积平滑、渐进视野扩展机制，实现零/少样本下的测试时自主训练与跨任务泛化。**

- **链接: [https://arxiv.org/pdf/2512.14666v1](https://arxiv.org/pdf/2512.14666v1)**

> **作者:** Zechen Bai; Chen Gao; Mike Zheng Shou
>
> **备注:** 15 pages
>
> **摘要:** Achieving truly adaptive embodied intelligence requires agents that learn not just by imitating static demonstrations, but by continuously improving through environmental interaction, which is akin to how humans master skills through practice. Vision-Language-Action (VLA) models have advanced robotic manipulation by leveraging large language models, yet remain fundamentally limited by Supervised Finetuning (SFT): requiring hundreds of demonstrations per task, rigidly memorizing trajectories, and failing to adapt when deployment conditions deviate from training. We introduce EVOLVE-VLA, a test-time training framework enabling VLAs to continuously adapt through environment interaction with minimal or zero task-specific demonstrations. The key technical challenge is replacing oracle reward signals (unavailable at test time) with autonomous feedback. We address this through a learned progress estimator providing dense feedback, and critically, we design our framework to ``tame'' this inherently noisy signal via two mechanisms: (1) an accumulative progress estimation mechanism smoothing noisy point-wise estimates, and (2) a progressive horizon extension strategy enabling gradual policy evolution. EVOLVE-VLA achieves substantial gains: +8.6\% on long-horizon tasks, +22.0\% in 1-shot learning, and enables cross-task generalization -- achieving 20.8\% success on unseen tasks without task-specific demonstrations training (vs. 0\% for pure SFT). Qualitative analysis reveals emergent capabilities absent in demonstrations, including error recovery and novel strategies. This work represents a critical step toward VLAs that truly learn and adapt, moving beyond static imitation toward continuous self-improvements.
>
---
#### [new 008] Field evaluation and optimization of a lightweight lidar-based UAV navigation system for dense boreal forest environments
- **分类: cs.RO**

- **简介: 该论文属森林机器人任务，旨在解决无人机在茂密北方林下自主导航难的问题。作者基于轻量激光雷达与开源算法（IPC路径规划、LTA-OM SLAM）构建并优化四旋翼系统，开展93次实地飞行实验，提出标准化测试方法，显著提升可靠性与成功率。**

- **链接: [https://arxiv.org/pdf/2512.14340v1](https://arxiv.org/pdf/2512.14340v1)**

> **作者:** Aleksi Karhunen; Teemu Hakala; Väinö Karjalainen; Eija Honkavaara
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** The interest in the usage of uncrewed aerial vehicles (UAVs) for forest applications has increased in recent years. While above-canopy flight has reached a high level of autonomy, navigating under-canopy remains a significant challenge. The use of autonomous UAVs could reduce the burden of data collection, which has motivated the development of numerous solutions for under-canopy autonomous flight. However, the experiments conducted in the literature and their reporting lack rigor. Very rarely, the density and the difficulty of the test forests are reported, or multiple flights are flown, and the success rate of those flights is reported. The aim of this study was to implement an autonomously flying quadrotor based on a lightweight lidar using openly available algorithms and test its behavior in real forest environments. A set of rigorous experiments was conducted with a quadrotor prototype utilizing the IPC path planner and LTA-OM SLAM algorithm. Based on the results of the first 33 flights, the original system was further enhanced. With the optimized system, 60 flights were performed, resulting in a total of 93 test flights. The optimized system performed significantly better in terms of reliability and flight mission completion times, achieving success rates of 12/15 in a medium-density forest and 15/15 in a dense forest, at a target flight velocity of 1 m/s. At a target flight velocity of 2 m/s, it had a success rate of 12/15 and 5/15, respectively. Furthermore, a standardized testing setup and evaluation criteria were proposed, enabling consistent performance comparisons of autonomous under-canopy UAV systems, enhancing reproducibility, guiding system improvements, and accelerating progress in forest robotics.
>
---
#### [new 009] SUPER -- A Framework for Sensitivity-based Uncertainty-aware Performance and Risk Assessment in Visual Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文提出SUPER框架，面向视觉惯性里程计（VIO）的实时风险评估任务，解决现有方法缺乏运行时风险感知的问题。它基于敏感度传播不确定性，利用高斯-牛顿法法方程的Schur补块构建无后端依赖的风险指标，实现轨迹退化预测与主动应对，无需真值。**

- **链接: [https://arxiv.org/pdf/2512.14189v1](https://arxiv.org/pdf/2512.14189v1)**

> **作者:** Johannes A. Gaus; Daniel Häufle; Woo-Jeong Baek
>
> **摘要:** While many visual odometry (VO), visual-inertial odometry (VIO), and SLAM systems achieve high accuracy, the majority of existing methods miss to assess risks at runtime. This paper presents SUPER (Sensitivity-based Uncertainty-aware PErformance and Risk assessment) that is a generic and explainable framework that propagates uncertainties via sensitivities for real-time risk assessment in VIO. The scientific novelty lies in the derivation of a real-time risk indicator that is backend-agnostic and exploits the Schur complement blocks of the Gauss-Newton normal matrix to propagate uncertainties. Practically, the Schur complement captures the sensitivity that reflects the influence of the uncertainty on the risk occurrence. Our framework estimates risks on the basis of the residual magnitudes, geometric conditioning, and short horizon temporal trends without requiring ground truth knowledge. Our framework enables to reliably predict trajectory degradation 50 frames ahead with an improvement of 20% to the baseline. In addition, SUPER initiates a stop or relocalization policy with 89.1% recall. The framework is backend agnostic and operates in real time with less than 0.2% additional CPU cost. Experiments show that SUPER provides consistent uncertainty estimates. A SLAM evaluation highlights the applicability to long horizon mapping.
>
---
#### [new 010] E-Navi: Environmental Adaptive Navigation for UAVs on Resource Constrained Platforms
- **分类: cs.RO**

- **简介: 该论文提出E-Navi系统，面向资源受限无人机的自主导航任务，解决环境动态变化下固定配置导致计算冗余与性能下降的问题。通过量化环境复杂度，动态调整感知-规划流水线的地图分辨率与执行频率，实现计算负载自适应优化。**

- **链接: [https://arxiv.org/pdf/2512.14046v1](https://arxiv.org/pdf/2512.14046v1)**

> **作者:** Boyang Li; Zhongpeng Jin; Shuai Zhao; Jiahui Liao; Tian Liu; Han Liu; Yuanhai Zhang; Kai Huang
>
> **摘要:** The ability to adapt to changing environments is crucial for the autonomous navigation systems of Unmanned Aerial Vehicles (UAVs). However, existing navigation systems adopt fixed execution configurations without considering environmental dynamics based on available computing resources, e.g., with a high execution frequency and task workload. This static approach causes rigid flight strategies and excessive computations, ultimately degrading flight performance or even leading to failures in UAVs. Despite the necessity for an adaptive system, dynamically adjusting workloads remains challenging, due to difficulties in quantifying environmental complexity and modeling the relationship between environment and system configuration. Aiming at adapting to dynamic environments, this paper proposes E-Navi, an environmental-adaptive navigation system for UAVs that dynamically adjusts task executions on the CPUs in response to environmental changes based on available computational resources. Specifically, the perception-planning pipeline of UAVs navigation system is redesigned through dynamic adaptation of mapping resolution and execution frequency, driven by the quantitative environmental complexity evaluations. In addition, E-Navi supports flexible deployment across hardware platforms with varying levels of computing capability. Extensive Hardware-In-the-Loop and real-world experiments demonstrate that the proposed system significantly outperforms the baseline method across various hardware platforms, achieving up to 53.9% navigation task workload reduction, up to 63.8% flight time savings, and delivering more stable velocity control.
>
---
#### [new 011] WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出WAM-Flow，一种面向自动驾驶的端到端运动规划VLA模型。它将轨迹生成建模为离散流匹配任务，通过并行双向去噪实现粗到细规划，在NAVSIM上显著优于自回归与扩散基线。**

- **链接: [https://arxiv.org/pdf/2512.06112v2](https://arxiv.org/pdf/2512.06112v2)**

> **作者:** Yifang Xu; Jiahao Cui; Feipeng Cai; Zhihao Zhu; Hanlin Shang; Shan Luan; Mingwang Xu; Neng Zhang; Yaoyi Li; Jia Cai; Siyu Zhu
>
> **备注:** 18 pages, 11 figures. Code & Model: https://github.com/fudan-generative-vision/WAM-Flow
>
> **摘要:** We introduce WAM-Flow, a vision-language-action (VLA) model that casts ego-trajectory planning as discrete flow matching over a structured token space. In contrast to autoregressive decoders, WAM-Flow performs fully parallel, bidirectional denoising, enabling coarse-to-fine refinement with a tunable compute-accuracy trade-off. Specifically, the approach combines a metric-aligned numerical tokenizer that preserves scalar geometry via triplet-margin learning, a geometry-aware flow objective and a simulator-guided GRPO alignment that integrates safety, ego progress, and comfort rewards while retaining parallel generation. A multi-stage adaptation converts a pre-trained auto-regressive backbone (Janus-1.5B) from causal decoding to non-causal flow model and strengthens road-scene competence through continued multimodal pretraining. Thanks to the inherent nature of consistency model training and parallel decoding inference, WAM-Flow achieves superior closed-loop performance against autoregressive and diffusion-based VLA baselines, with 1-step inference attaining 89.1 PDMS and 5-step inference reaching 90.3 PDMS on NAVSIM v1 benchmark. These results establish discrete flow matching as a new promising paradigm for end-to-end autonomous driving. The code will be publicly available soon.
>
---
#### [new 012] CLAIM: Camera-LiDAR Alignment with Intensity and Monodepth
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出CLAIM方法，解决相机与LiDAR传感器外参标定问题。利用单目深度模型，通过粗到精搜索优化基于结构（Patch Pearson相关）和纹理（互信息）的双损失函数，无需特征提取或匹配，简单鲁棒，在KITTI等数据集上性能领先。**

- **链接: [https://arxiv.org/pdf/2512.14001v1](https://arxiv.org/pdf/2512.14001v1)**

> **作者:** Zhuo Zhang; Yonghui Liu; Meijie Zhang; Feiyang Tan; Yikang Ding
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** In this paper, we unleash the potential of the powerful monodepth model in camera-LiDAR calibration and propose CLAIM, a novel method of aligning data from the camera and LiDAR. Given the initial guess and pairs of images and LiDAR point clouds, CLAIM utilizes a coarse-to-fine searching method to find the optimal transformation minimizing a patched Pearson correlation-based structure loss and a mutual information-based texture loss. These two losses serve as good metrics for camera-LiDAR alignment results and require no complicated steps of data processing, feature extraction, or feature matching like most methods, rendering our method simple and adaptive to most scenes. We validate CLAIM on public KITTI, Waymo, and MIAS-LCEC datasets, and the experimental results demonstrate its superior performance compared with the state-of-the-art methods. The code is available at https://github.com/Tompson11/claim.
>
---
#### [new 013] CaFe-TeleVision: A Coarse-to-Fine Teleoperation System with Immersive Situated Visualization for Enhanced Ergonomics
- **分类: cs.RO**

- **简介: 该论文提出CaFe-TeleVision系统，解决远程操控中效率低、人机工效差的问题。通过粗粒度到细粒度的控制机制与按需情境可视化技术，在双臂操作任务中提升操作舒适性与成功率，显著降低认知负荷与任务负荷。**

- **链接: [https://arxiv.org/pdf/2512.14270v1](https://arxiv.org/pdf/2512.14270v1)**

> **作者:** Zixin Tang; Yiming Chen; Quentin Rouxel; Dianxi Li; Shuang Wu; Fei Chen
>
> **摘要:** Teleoperation presents a promising paradigm for remote control and robot proprioceptive data collection. Despite recent progress, current teleoperation systems still suffer from limitations in efficiency and ergonomics, particularly in challenging scenarios. In this paper, we propose CaFe-TeleVision, a coarse-to-fine teleoperation system with immersive situated visualization for enhanced ergonomics. At its core, a coarse-to-fine control mechanism is proposed in the retargeting module to bridge workspace disparities, jointly optimizing efficiency and physical ergonomics. To stream immersive feedback with adequate visual cues for human vision systems, an on-demand situated visualization technique is integrated in the perception module, which reduces the cognitive load for multi-view processing. The system is built on a humanoid collaborative robot and validated with six challenging bimanual manipulation tasks. User study among 24 participants confirms that CaFe-TeleVision enhances ergonomics with statistical significance, indicating a lower task load and a higher user acceptance during teleoperation. Quantitative results also validate the superior performance of our system across six tasks, surpassing comparative methods by up to 28.89% in success rate and accelerating by 26.81% in completion time. Project webpage: https://clover-cuhk.github.io/cafe_television/
>
---
#### [new 014] CHIP: Adaptive Compliance for Humanoid Control through Hindsight Perturbation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属机器人控制任务，旨在解决人形机器人在力控操作（如推车、擦洗）中难以兼顾末端刚度调节与动态运动跟踪的问题。提出CHIP方法，通过后见扰动实现自适应柔顺控制，无需数据增强或奖励调优，可即插即用提升力控操作能力。**

- **链接: [https://arxiv.org/pdf/2512.14689v1](https://arxiv.org/pdf/2512.14689v1)**

> **作者:** Sirui Chen; Zi-ang Cao; Zhengyi Luo; Fernando Castañeda; Chenran Li; Tingwu Wang; Ye Yuan; Linxi "Jim" Fan; C. Karen Liu; Yuke Zhu
>
> **备注:** The first two authors contributed equally. Project page: https://nvlabs.github.io/CHIP/
>
> **摘要:** Recent progress in humanoid robots has unlocked agile locomotion skills, including backflipping, running, and crawling. Yet it remains challenging for a humanoid robot to perform forceful manipulation tasks such as moving objects, wiping, and pushing a cart. We propose adaptive Compliance Humanoid control through hIsight Perturbation (CHIP), a plug-and-play module that enables controllable end-effector stiffness while preserving agile tracking of dynamic reference motions. CHIP is easy to implement and requires neither data augmentation nor additional reward tuning. We show that a generalist motion-tracking controller trained with CHIP can perform a diverse set of forceful manipulation tasks that require different end-effector compliance, such as multi-robot collaboration, wiping, box delivery, and door opening.
>
---
#### [new 015] Trajectory Tracking for Multi-Manipulator Systems in Constrained Environments
- **分类: cs.RO; eess.SY**

- **简介: 论文研究多机械臂系统在障碍密集、空间受限环境中的协同轨迹跟踪任务，旨在实现被夹持物体的高精度运动跟踪与避障。提出多速率规划控制框架：离线生成满足STL规范与无碰撞的物体轨迹及基座位姿，线上执行约束逆运动学与连续反馈控制。**

- **链接: [https://arxiv.org/pdf/2512.14206v1](https://arxiv.org/pdf/2512.14206v1)**

> **作者:** Mayank Sewlia; Christos K. Verginis; Dimos V. Dimarogonas
>
> **摘要:** We consider the problem of cooperative manipulation by a mobile multi-manipulator system operating in obstacle-cluttered and highly constrained environments under spatio-temporal task specifications. The task requires transporting a grasped object while respecting both continuous robot dynamics and discrete geometric constraints arising from obstacles and narrow passages. To address this hybrid structure, we propose a multi-rate planning and control framework that combines offline generation of an STL-satisfying object trajectory and collision-free base footprints with online constrained inverse kinematics and continuous-time feedback control. The resulting closed-loop system enables coordinated reconfiguration of multiple manipulators while tracking the desired object motion. The approach is evaluated in high-fidelity physics simulations using three Franka Emika Panda mobile manipulators rigidly grasping an object.
>
---
#### [new 016] Impact of Robot Facial-Audio Expressions on Human Robot Trust Dynamics and Trust Repair
- **分类: cs.RO**

- **简介: 该论文属人机交互（HRI）研究，旨在解决机器人协作中信任动态变化与修复问题。通过控制实验，考察机器人面部-音频表达（成功“高兴”/失败“道歉”）对人类信任的时序影响，发现道歉可部分修复信任，且效果受任务类型、年龄和态度调节。**

- **链接: [https://arxiv.org/pdf/2512.13981v1](https://arxiv.org/pdf/2512.13981v1)**

> **作者:** Hossein Naderi; Alireza Shojaei; Philip Agee; Kereshmeh Afsari; Abiola Akanmu
>
> **摘要:** Despite recent advances in robotics and human-robot collaboration in the AEC industry, trust has mostly been treated as a static factor, with little guidance on how it changes across events during collaboration. This paper investigates how a robot's task performance and its expressive responses after outcomes shape the dynamics of human trust over time. To this end, we designed a controlled within-subjects study with two construction-inspired tasks, Material Delivery (physical assistance) and Information Gathering (perceptual assistance), and measured trust repeatedly (four times per task) using the 14-item Trust Perception Scale for HRI plus a redelegation choice. The robot produced two multimodal expressions, a "glad" display with a brief confirmation after success, and a "sad" display with an apology and a request for a second chance after failure. The study was conducted in a lab environment with 30 participants and a quadruped platform, and we evaluated trust dynamics and repair across both tasks. Results show that robot success reliably increases trust, failure causes sharp drops, and apology-based expressions partially restores trust (44% recovery in Material Delivery; 38% in Information Gathering). Item-level analysis indicates that recovered trust was driven mostly by interaction and communication factors, with competence recovering partially and autonomy aspects changing least. Additionally, age group and prior attitudes moderated trust dynamics with younger participants showed larger but shorter-lived changes, mid-20s participants exhibited the most durable repair, and older participants showed most conservative dynamics. This work provides a foundation for future efforts that adapt repair strategies to task demands and user profiles to support safe, productive adoption of robots on construction sites.
>
---
#### [new 017] Context Representation via Action-Free Transformer encoder-decoder for Meta Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属元强化学习任务，旨在解决现有方法依赖动作信息导致任务推断与策略强耦合的问题。提出CRAFT模型，仅用状态-奖励序列推断任务上下文，解耦任务推断与策略优化，提升泛化与适应效率。**

- **链接: [https://arxiv.org/pdf/2512.14057v1](https://arxiv.org/pdf/2512.14057v1)**

> **作者:** Amir M. Soufi Enayati; Homayoun Honari; Homayoun Najjaran
>
> **摘要:** Reinforcement learning (RL) enables robots to operate in uncertain environments, but standard approaches often struggle with poor generalization to unseen tasks. Context-adaptive meta reinforcement learning addresses these limitations by conditioning on the task representation, yet they mostly rely on complete action information in the experience making task inference tightly coupled to a specific policy. This paper introduces Context Representation via Action Free Transformer encoder decoder (CRAFT), a belief model that infers task representations solely from sequences of states and rewards. By removing the dependence on actions, CRAFT decouples task inference from policy optimization, supports modular training, and leverages amortized variational inference for scalable belief updates. Built on a transformer encoder decoder with rotary positional embeddings, the model captures long range temporal dependencies and robustly encodes both parametric and non-parametric task variations. Experiments on the MetaWorld ML-10 robotic manipulation benchmark show that CRAFT achieves faster adaptation, improved generalization, and more effective exploration compared to context adaptive meta--RL baselines. These findings highlight the potential of action-free inference as a foundation for scalable RL in robotic control.
>
---
#### [new 018] Autonomous Construction-Site Safety Inspection Using Mobile Robots: A Multilayer VLM-LLM Pipeline
- **分类: cs.RO**

- **简介: 该论文提出一种多层VLM-LLM管道，实现移动机器人自主施工安全巡检。旨在解决人工检查低效、现有自动化方法依赖难维护的专用数据集及需人工遥操作等问题。工作包括：SLAM导航、VLM场景描述与规则检索、VLM安全评估、LLM报告生成，并在模拟环境中验证。**

- **链接: [https://arxiv.org/pdf/2512.13974v1](https://arxiv.org/pdf/2512.13974v1)**

> **作者:** Hossein Naderi; Alireza Shojaei; Philip Agee; Kereshmeh Afsari; Abiola Akanmu
>
> **摘要:** Construction safety inspection remains mostly manual, and automated approaches still rely on task-specific datasets that are hard to maintain in fast-changing construction environments due to frequent retraining. Meanwhile, field inspection with robots still depends on human teleoperation and manual reporting, which are labor-intensive. This paper aims to connect what a robot sees during autonomous navigation to the safety rules that are common in construction sites, automatically generating a safety inspection report. To this end, we proposed a multi-layer framework with two main modules: robotics and AI. On the robotics side, SLAM and autonomous navigation provide repeatable coverage and targeted revisits via waypoints. On AI side, a Vision Language Model (VLM)-based layer produces scene descriptions; a retrieval component powered grounds those descriptions in OSHA and site policies; Another VLM-based layer assesses the safety situation based on rules; and finally Large Language Model (LLM) layer generates safety reports based on previous outputs. The framework is validated with a proof-of-concept implementation and evaluated in a lab environment that simulates common hazards across three scenarios. Results show high recall with competitive precision compared to state-of-the-art closed-source models. This paper contributes a transparent, generalizable pipeline that moves beyond black-box models by exposing intermediate artifacts from each layer and keeping the human in the loop. This work provides a foundation for future extensions to additional tasks and settings within and beyond construction context.
>
---
#### [new 019] Odyssey: An Automotive Lidar-Inertial Odometry Dataset for GNSS-denied situations
- **分类: cs.RO**

- **简介: 该论文面向GNSS拒止环境下的激光雷达-惯性里程计（LIO）研究，提出首个公开的RLG级高精度惯导基准数据集Odyssey，解决现有MEMS/FOG惯导在长时无GNSS场景下误差累积严重的问题，并支持LIO、SLAM及地点识别等任务。**

- **链接: [https://arxiv.org/pdf/2512.14428v1](https://arxiv.org/pdf/2512.14428v1)**

> **作者:** Aaron Kurda; Simon Steuernagel; Lukas Jung; Marcus Baum
>
> **备注:** 9 pages, 4 figures, submitted to International Journal of Robotics Research (IJRR)
>
> **摘要:** The development and evaluation of Lidar-Inertial Odometry (LIO) and Simultaneous Localization and Mapping (SLAM) systems requires a precise ground truth. The Global Navigation Satellite System (GNSS) is often used as a foundation for this, but its signals can be unreliable in obstructed environments due to multi-path effects or loss-of-signal. While existing datasets compensate for the sporadic loss of GNSS signals by incorporating Inertial Measurement Unit (IMU) measurements, the commonly used Micro-Electro-Mechanical Systems (MEMS) or Fiber Optic Gyroscope (FOG)-based systems do not permit the prolonged study of GNSS-denied environments. To close this gap, we present Odyssey, a LIO dataset with a focus on GNSS-denied environments such as tunnels and parking garages as well as other underrepresented, yet ubiquitous situations such as stop-and-go-traffic, bumpy roads and wide open fields. Our ground truth is derived from a navigation-grade Inertial Navigation System (INS) equipped with a Ring Laser Gyroscope (RLG), offering exceptional bias stability characteristics compared to IMUs used in existing datasets and enabling the prolonged and accurate study of GNSS-denied environments. This makes Odyssey the first publicly available dataset featuring a RLG-based INS. Besides providing data for LIO, we also support other tasks, such as place recognition, through the threefold repetition of all trajectories as well as the integration of external mapping data by providing precise geodetic coordinates. All data, dataloader and other material is available online at https://odyssey.uni-goettingen.de/ .
>
---
#### [new 020] Geometric Parameter Optimization of a Novel 3-(PP(2-(UPS))) Redundant Parallel Mechanism based on Workspace Determination
- **分类: cs.RO**

- **简介: 该论文属机器人机构学中的构型设计与参数优化任务，旨在解决新型3-(PP(2-(UPS)))冗余并联机构几何参数对工作空间（体积、形状、边界完整性、姿态能力）影响不明确的问题；通过定义扭转/倾转能力指数TI₁/TI₂，结合数值仿真分析参数影响规律，指导优化设计。**

- **链接: [https://arxiv.org/pdf/2512.14434v1](https://arxiv.org/pdf/2512.14434v1)**

> **作者:** Quan Yuan; Daqian Cao; Weibang Bai
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Redundant parallel robots are normally employed in scenarios requiring good precision, high load capability, and large workspace compared to traditional parallel mechanisms. However, the elementary robotic configuration and geometric parameter optimization are still quite challenging. This paper proposes a novel 3-(PP(2-(UPS))) redundant parallel mechanism, with good generalizability first, and further investigates the kinematic optimization issue by analyzing and investigating how its key geometric parameters influence the volume, shape, boundary completeness, and orientation capabilities of its workspace. The torsional capability index TI_1 and tilting capability index TI_2 are defined to evaluate the orientation performance of the mechanism. Numerical simulation studies are completed to indicate the analysis, providing reasonable but essential references for the parameter optimization of 3-(PP(2-(UPS))) and other similar redundant parallel mechanisms.
>
---
#### [new 021] ARCADE: Adaptive Robot Control with Online Changepoint-Aware Bayesian Dynamics Learning
- **分类: cs.RO**

- **简介: 该论文面向机器人在线自适应控制任务，解决动态环境（如突变、漂移、扰动）下模型实时更新与不确定性校准难题。提出ARCADE框架：离线学习潜表示，线上贝叶斯更新；引入基于似然的变点感知机制，自动权衡记忆与遗忘，实现快速响应与鲁棒预测。**

- **链接: [https://arxiv.org/pdf/2512.14331v1](https://arxiv.org/pdf/2512.14331v1)**

> **作者:** Rishabh Dev Yadav; Avirup Das; Hongyu Song; Samuel Kaski; Wei Pan
>
> **摘要:** Real-world robots must operate under evolving dynamics caused by changing operating conditions, external disturbances, and unmodeled effects. These may appear as gradual drifts, transient fluctuations, or abrupt shifts, demanding real-time adaptation that is robust to short-term variation yet responsive to lasting change. We propose a framework for modeling the nonlinear dynamics of robotic systems that can be updated in real time from streaming data. The method decouples representation learning from online adaptation, using latent representations learned offline to support online closed-form Bayesian updates. To handle evolving conditions, we introduce a changepoint-aware mechanism with a latent variable inferred from data likelihoods that indicates continuity or shift. When continuity is likely, evidence accumulates to refine predictions; when a shift is detected, past information is tempered to enable rapid re-learning. This maintains calibrated uncertainty and supports probabilistic reasoning about transient, gradual, or structural change. We prove that the adaptive regret of the framework grows only logarithmically in time and linearly with the number of shifts, competitive with an oracle that knows timings of shift. We validate on cartpole simulations and real quadrotor flights with swinging payloads and mid-flight drops, showing improved predictive accuracy, faster recovery, and more accurate closed-loop tracking than relevant baselines.
>
---
#### [new 022] Expert Switching for Robust AAV Landing: A Dual-Detector Framework in Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向AAV视觉着陆任务，解决单模型在大尺度变化（高空小目标→低空大目标）下检测鲁棒性差的问题。提出双专家YOLOv8框架，按远/近程尺度分工训练，并用几何门控动态选优，提升着陆精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.14054v1](https://arxiv.org/pdf/2512.14054v1)**

> **作者:** Humaira Tasnim; Ashik E Rasul; Bruce Jo; Hyung-Jin Yoon
>
> **摘要:** Reliable helipad detection is essential for Autonomous Aerial Vehicle (AAV) landing, especially under GPS-denied or visually degraded conditions. While modern detectors such as YOLOv8 offer strong baseline performance, single-model pipelines struggle to remain robust across the extreme scale transitions that occur during descent, where helipads appear small at high altitude and large near touchdown. To address this limitation, we propose a scale-adaptive dual-expert perception framework that decomposes the detection task into far-range and close-range regimes. Two YOLOv8 experts are trained on scale-specialized versions of the HelipadCat dataset, enabling one model to excel at detecting small, low-resolution helipads and the other to provide high-precision localization when the target dominates the field of view. During inference, both experts operate in parallel, and a geometric gating mechanism selects the expert whose prediction is most consistent with the AAV's viewpoint. This adaptive routing prevents the degradation commonly observed in single-detector systems when operating across wide altitude ranges. The dual-expert perception module is evaluated in a closed-loop landing environment that integrates CARLA's photorealistic rendering with NASA's GUAM flight-dynamics engine. Results show substantial improvements in alignment stability, landing accuracy, and overall robustness compared to single-detector baselines. By introducing a scale-aware expert routing strategy tailored to the landing problem, this work advances resilient vision-based perception for autonomous descent and provides a foundation for future multi-expert AAV frameworks.
>
---
#### [new 023] A Comprehensive Safety Metric to Evaluate Perception in Autonomous Systems
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶感知安全评估任务，旨在解决现有感知指标忽略目标重要性差异（如速度、距离、碰撞风险等）导致的安全评价不准确问题。作者提出一种综合安全度量指标，融合多维动态参数，输出单一可解释的安全评分，并在真实与虚拟数据上验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.14367v1](https://arxiv.org/pdf/2512.14367v1)**

> **作者:** Georg Volk; Jörg Gamerdinger; Alexander von Bernuth; Oliver Bringmann
>
> **备注:** Accepted at IEEE ITSC 2020
>
> **摘要:** Complete perception of the environment and its correct interpretation is crucial for autonomous vehicles. Object perception is the main component of automotive surround sensing. Various metrics already exist for the evaluation of object perception. However, objects can be of different importance depending on their velocity, orientation, distance, size, or the potential damage that could be caused by a collision due to a missed detection. Thus, these additional parameters have to be considered for safety evaluation. We propose a new safety metric that incorporates all these parameters and returns a single easily interpretable safety assessment score for object perception. This new metric is evaluated with both real world and virtual data sets and compared to state of the art metrics.
>
---
#### [new 024] Interactive Motion Planning for Human-Robot Collaboration Based on Human-Centric Configuration Space Ergonomic Field
- **分类: cs.RO**

- **简介: 该论文属人机协作运动规划任务，旨在解决工业场景中机器人运动既避障又符合人体工学的问题。提出配置空间工效场（CSEF），构建连续可微的关节空间工效度量与梯度，集成至梯度规划器，在仿真与硬件实验中验证其提升工效性、降低肌肉激活的有效性。**

- **链接: [https://arxiv.org/pdf/2512.14111v1](https://arxiv.org/pdf/2512.14111v1)**

> **作者:** Chenzui Li; Yiming Chen; Xi Wu; Tao Teng; Sylvain Calinon; Darwin Caldwell; Fei Chen
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** Industrial human-robot collaboration requires motion planning that is collision-free, responsive, and ergonomically safe to reduce fatigue and musculoskeletal risk. We propose the Configuration Space Ergonomic Field (CSEF), a continuous and differentiable field over the human joint space that quantifies ergonomic quality and provides gradients for real-time ergonomics-aware planning. An efficient algorithm constructs CSEF from established metrics with joint-wise weighting and task conditioning, and we integrate it into a gradient-based planner compatible with impedance-controlled robots. In a 2-DoF benchmark, CSEF-based planning achieves higher success rates, lower ergonomic cost, and faster computation than a task-space ergonomic planner. Hardware experiments with a dual-arm robot in unimanual guidance, collaborative drilling, and bimanual cocarrying show faster ergonomic cost reduction, closer tracking to optimized joint targets, and lower muscle activation than a point-to-point baseline. CSEF-based planning method reduces average ergonomic scores by up to 10.31% for collaborative drilling tasks and 5.60% for bimanual co-carrying tasks while decreasing activation in key muscle groups, indicating practical benefits for real-world deployment.
>
---
#### [new 025] DRAW2ACT: Turning Depth-Encoded Trajectories into Robotic Demonstration Videos
- **分类: cs.CV; cs.RO**

- **简介: 该论文属机器人视觉-动作生成任务，旨在提升轨迹条件视频生成的可控性与一致性。提出DRAW2ACT框架：利用深度编码轨迹提取多维表征，联合生成对齐的RGB/深度视频，并通过多模态策略模型输出关节角，显著提升视觉质量与操作成功率。**

- **链接: [https://arxiv.org/pdf/2512.14217v1](https://arxiv.org/pdf/2512.14217v1)**

> **作者:** Yang Bai; Liudi Yang; George Eskandar; Fengyi Shen; Mohammad Altillawi; Ziyuan Liu; Gitta Kutyniok
>
> **摘要:** Video diffusion models provide powerful real-world simulators for embodied AI but remain limited in controllability for robotic manipulation. Recent works on trajectory-conditioned video generation address this gap but often rely on 2D trajectories or single modality conditioning, which restricts their ability to produce controllable and consistent robotic demonstrations. We present DRAW2ACT, a depth-aware trajectory-conditioned video generation framework that extracts multiple orthogonal representations from the input trajectory, capturing depth, semantics, shape and motion, and injects them into the diffusion model. Moreover, we propose to jointly generate spatially aligned RGB and depth videos, leveraging cross-modality attention mechanisms and depth supervision to enhance the spatio-temporal consistency. Finally, we introduce a multimodal policy model conditioned on the generated RGB and depth sequences to regress the robot's joint angles. Experiments on Bridge V2, Berkeley Autolab, and simulation benchmarks show that DRAW2ACT achieves superior visual fidelity and consistency while yielding higher manipulation success rates compared to existing baselines.
>
---
#### [new 026] CRISP: Contact-Guided Real2Sim from Monocular Video with Planar Scene Primitives
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 论文提出CRISP方法，解决单目视频中人体-场景联合重建不满足物理仿真需求的问题。通过平面几何拟合、接触引导的遮挡恢复和强化学习驱动的物理验证，生成清洁、凸、可仿真的场景与人体运动，显著降低跟踪失败率并提升仿真效率。**

- **链接: [https://arxiv.org/pdf/2512.14696v1](https://arxiv.org/pdf/2512.14696v1)**

> **作者:** Zihan Wang; Jiashun Wang; Jeff Tan; Yiwen Zhao; Jessica Hodgins; Shubham Tulsiani; Deva Ramanan
>
> **备注:** Project page: https://crisp-real2sim.github.io/CRISP-Real2Sim/
>
> **摘要:** We introduce CRISP, a method that recovers simulatable human motion and scene geometry from monocular video. Prior work on joint human-scene reconstruction relies on data-driven priors and joint optimization with no physics in the loop, or recovers noisy geometry with artifacts that cause motion tracking policies with scene interactions to fail. In contrast, our key insight is to recover convex, clean, and simulation-ready geometry by fitting planar primitives to a point cloud reconstruction of the scene, via a simple clustering pipeline over depth, normals, and flow. To reconstruct scene geometry that might be occluded during interactions, we make use of human-scene contact modeling (e.g., we use human posture to reconstruct the occluded seat of a chair). Finally, we ensure that human and scene reconstructions are physically-plausible by using them to drive a humanoid controller via reinforcement learning. Our approach reduces motion tracking failure rates from 55.2\% to 6.9\% on human-centric video benchmarks (EMDB, PROX), while delivering a 43\% faster RL simulation throughput. We further validate it on in-the-wild videos including casually-captured videos, Internet videos, and even Sora-generated videos. This demonstrates CRISP's ability to generate physically-valid human motion and interaction environments at scale, greatly advancing real-to-sim applications for robotics and AR/VR.
>
---
#### [new 027] A4-Agent: An Agentic Framework for Zero-Shot Affordance Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向具身AI中的零样本可供性推理任务，旨在解决现有模型泛化差、依赖标注数据的问题。提出A4-Agent框架，无需训练，通过Dreamer（可视化交互）、Thinker（识别交互部件）、Spotter（精确定位）三阶段解耦推理，利用多类基础模型协同完成语言驱动的交互区域预测。**

- **链接: [https://arxiv.org/pdf/2512.14442v1](https://arxiv.org/pdf/2512.14442v1)**

> **作者:** Zixin Zhang; Kanghao Chen; Hanqing Wang; Hongfei Zhang; Harold Haodong Chen; Chenfei Liao; Litao Guo; Ying-Cong Chen
>
> **摘要:** Affordance prediction, which identifies interaction regions on objects based on language instructions, is critical for embodied AI. Prevailing end-to-end models couple high-level reasoning and low-level grounding into a single monolithic pipeline and rely on training over annotated datasets, which leads to poor generalization on novel objects and unseen environments. In this paper, we move beyond this paradigm by proposing A4-Agent, a training-free agentic framework that decouples affordance prediction into a three-stage pipeline. Our framework coordinates specialized foundation models at test time: (1) a $\textbf{Dreamer}$ that employs generative models to visualize $\textit{how}$ an interaction would look; (2) a $\textbf{Thinker}$ that utilizes large vision-language models to decide $\textit{what}$ object part to interact with; and (3) a $\textbf{Spotter}$ that orchestrates vision foundation models to precisely locate $\textit{where}$ the interaction area is. By leveraging the complementary strengths of pre-trained models without any task-specific fine-tuning, our zero-shot framework significantly outperforms state-of-the-art supervised methods across multiple benchmarks and demonstrates robust generalization to real-world settings.
>
---
#### [new 028] Constrained Policy Optimization via Sampling-Based Weight-Space Projection
- **分类: cs.LG; cs.RO**

- **简介: 该论文属安全强化学习任务，解决策略优化中未知 rollout-based 安全约束下的参数安全更新问题。提出 SCPO 方法：在参数空间采样构建局部安全区域，通过 SOCP 投影实现安全梯度步，并提供安全归纳保证。**

- **链接: [https://arxiv.org/pdf/2512.13788v1](https://arxiv.org/pdf/2512.13788v1)**

> **作者:** Shengfan Cao; Francesco Borrelli
>
> **备注:** Submitted to IFAC World Congress 2026
>
> **摘要:** Safety-critical learning requires policies that improve performance without leaving the safe operating regime. We study constrained policy learning where model parameters must satisfy unknown, rollout-based safety constraints. We propose SCPO, a sampling-based weight-space projection method that enforces safety directly in parameter space without requiring gradient access to the constraint functions. Our approach constructs a local safe region by combining trajectory rollouts with smoothness bounds that relate parameter changes to shifts in safety metrics. Each gradient update is then projected via a convex SOCP, producing a safe first-order step. We establish a safe-by-induction guarantee: starting from any safe initialization, all intermediate policies remain safe given feasible projections. In constrained control settings with a stabilizing backup policy, our approach further ensures closed-loop stability and enables safe adaptation beyond the conservative backup. On regression with harmful supervision and a constrained double-integrator task with malicious expert, our approach consistently rejects unsafe updates, maintains feasibility throughout training, and achieves meaningful primal objective improvement.
>
---
#### [new 029] A Geometric Task-Space Port-Hamiltonian Formulation for Redundant Manipulators
- **分类: eess.SY; cs.RO**

- **简介: 该论文提出一种面向冗余机械臂的几何任务空间端口-哈密顿建模方法，解决传统哈密顿模型难以直接处理任务空间控制的问题。工作包括：构建任务/零空间动量分解的新模型，分析其性质及与拉格朗日模型的关系，并基于IDA-PBC实现7-DOF Panda机器人任务空间阻抗控制。**

- **链接: [https://arxiv.org/pdf/2512.14349v1](https://arxiv.org/pdf/2512.14349v1)**

> **作者:** Federico Califano; Camilla Rota; Riccardo Zanella; Antonio Franchi
>
> **摘要:** We present a novel geometric port-Hamiltonian formulation of redundant manipulators performing a differential kinematic task $η=J(q)\dot{q}$, where $q$ is a point on the configuration manifold, $η$ is a velocity-like task space variable, and $J(q)$ is a linear map representing the task, for example the classical analytic or geometric manipulator Jacobian matrix. The proposed model emerges from a change of coordinates from canonical Hamiltonian dynamics, and splits the standard Hamiltonian momentum variable into a task-space momentum variable and a null-space momentum variable. Properties of this model and relation to Lagrangian formulations present in the literature are highlighted. Finally, we apply the proposed model in an \textit{Interconnection and Damping Assignment Passivity-Based Control} (IDA-PBC) design to stabilize and shape the impedance of a 7-DOF Emika Panda robot in simulation.
>
---
#### [new 030] A Convex Obstacle Avoidance Formulation
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文面向自动驾驶中的实时避障任务，解决非线性MPC计算耗时、难以满足高频率需求的问题。提出首个通用凸障碍物避障公式，通过新逻辑集成方法将避障嵌入凸MPC框架，提升效率并支持短预测时域，同时在非凸不可避场景下性能不逊于主流非凸方法。**

- **链接: [https://arxiv.org/pdf/2512.13836v1](https://arxiv.org/pdf/2512.13836v1)**

> **作者:** Ricardo Tapia; Iman Soltani
>
> **备注:** 18 pages, 17 figures
>
> **摘要:** Autonomous driving requires reliable collision avoidance in dynamic environments. Nonlinear Model Predictive Controllers (NMPCs) are suitable for this task, but struggle in time-critical scenarios requiring high frequency. To meet this demand, optimization problems are often simplified via linearization, narrowing the horizon window, or reduced temporal nodes, each compromising accuracy or reliability. This work presents the first general convex obstacle avoidance formulation, enabled by a novel approach to integrating logic. This facilitates the incorporation of an obstacle avoidance formulation into convex MPC schemes, enabling a convex optimization framework with substantially improved computational efficiency relative to conventional nonconvex methods. A key property of the formulation is that obstacle avoidance remains effective even when obstacles lie outside the prediction horizon, allowing shorter horizons for real-time deployment. In scenarios where nonconvex formulations are unavoidable, the proposed method meets or exceeds the performance of representative nonconvex alternatives. The method is evaluated in autonomous vehicle applications, where system dynamics are highly nonlinear.
>
---
#### [new 031] History-Enhanced Two-Stage Transformer for Aerial Vision-and-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向空中视觉-语言导航（AVLN）任务，旨在解决无人机在大尺度城市环境中依语言指令准确定位目标时，全局推理与局部感知难以兼顾的问题。提出历史增强双阶段Transformer（HETT），通过粗粒度定位到细粒度动作优化的两阶段框架，并引入历史网格地图增强空间记忆，显著提升导航性能。**

- **链接: [https://arxiv.org/pdf/2512.14222v1](https://arxiv.org/pdf/2512.14222v1)**

> **作者:** Xichen Ding; Jianzhe Gao; Cong Pan; Wenguan Wang; Jie Qin
>
> **摘要:** Aerial Vision-and-Language Navigation (AVLN) requires Unmanned Aerial Vehicle (UAV) agents to localize targets in large-scale urban environments based on linguistic instructions. While successful navigation demands both global environmental reasoning and local scene comprehension, existing UAV agents typically adopt mono-granularity frameworks that struggle to balance these two aspects. To address this limitation, this work proposes a History-Enhanced Two-Stage Transformer (HETT) framework, which integrates the two aspects through a coarse-to-fine navigation pipeline. Specifically, HETT first predicts coarse-grained target positions by fusing spatial landmarks and historical context, then refines actions via fine-grained visual analysis. In addition, a historical grid map is designed to dynamically aggregate visual features into a structured spatial memory, enhancing comprehensive scene awareness. Additionally, the CityNav dataset annotations are manually refined to enhance data quality. Experiments on the refined CityNav dataset show that HETT delivers significant performance gains, while extensive ablation studies further verify the effectiveness of each component.
>
---
#### [new 032] Nonlinear System Identification Nano-drone Benchmark
- **分类: eess.SY; cs.RO**

- **简介: 该论文构建了一个基于Crazyflie 2.1 nano-无人机的真实数据基准，用于非线性系统辨识任务。旨在解决微型无人机在噪声、非线性与开环不稳定下的建模难题，提供了75k样本、多步预测指标及基线模型，并开源全部数据与代码。**

- **链接: [https://arxiv.org/pdf/2512.14450v1](https://arxiv.org/pdf/2512.14450v1)**

> **作者:** Riccardo Busetto; Elia Cereda; Marco Forgione; Gabriele Maroni; Dario Piga; Daniele Palossi
>
> **摘要:** We introduce a benchmark for system identification based on 75k real-world samples from the Crazyflie 2.1 Brushless nano-quadrotor, a sub-50g aerial vehicle widely adopted in robotics research. The platform presents a challenging testbed due to its multi-input, multi-output nature, open-loop instability, and nonlinear dynamics under agile maneuvers. The dataset comprises four aggressive trajectories with synchronized 4-dimensional motor inputs and 13-dimensional output measurements. To enable fair comparison of identification methods, the benchmark includes a suite of multi-horizon prediction metrics for evaluating both one-step and multi-step error propagation. In addition to the data, we provide a detailed description of the platform and experimental setup, as well as baseline models highlighting the challenge of accurate prediction under real-world noise and actuation nonlinearities. All data, scripts, and reference implementations are released as open-source at https://github.com/idsia-robotics/nanodrone-sysid-benchmark to facilitate transparent comparison of algorithms and support research on agile, miniaturized aerial robotics.
>
---
#### [new 033] Learning to Car-Follow Using an Inertia-Oriented Driving Technique: A Before-and-After Study on a Closed Circuit
- **分类: cs.HC; cs.CY; cs.RO**

- **简介: 该论文属人因工程与智能交通交叉任务，旨在验证“驾驶以保持惯性（DI）”策略能否替代传统“保持车距（DD）”策略以抑制交通振荡。通过12名驾驶员在封闭实车场景的前后测实验，证实DI培训可显著降低加减速幅度与速度波动。**

- **链接: [https://arxiv.org/pdf/2512.13694v1](https://arxiv.org/pdf/2512.13694v1)**

> **作者:** Kostantinos Mattas; Antonio Lucas-Alba; Tomer Toledo; Oscar M. Melchor; Shlomo Bekhor; Biagio Ciuffo
>
> **摘要:** For decades, car following and traffic flow models have assumed that drivers default driving strategy is to maintain a safe distance. Several previous studies have questioned whether the Driving to Keep Distance is a traffic invariant. Therefore, the acceleration deceleration torque asymmetry of drivers must necessarily determine the observed patterns of traffic oscillations. Those studies indicate that drivers can adopt alternative CF strategies, such as Driving to Keep Inertia, by following basic instructions. The present work extends the evidence from previous research by showing the effectiveness of a DI course that immediately translates into practice on a closed circuit. Twelve drivers were invited to follow a lead car that varied its speed on a real circuit. Then, the driver took a DI course and returned to the same real car following scenario. Drivers generally adopted DD as the default CF mode in the pretest, both in field and simulated PC conditions, yielding very similar results. After taking the full DI course, drivers showed significantly less acceleration, deceleration, and speed variability than did the pretest, both in the field and in the simulated conditions, which indicates that drivers adopted the DI strategy. This study is the first to show the potential of adopting a DI strategy in a real circuit.
>
---
#### [new 034] Quadratic Kalman Filter for Elliptical Extended Object Tracking based on Decoupling State Components
- **分类: eess.SP; cs.RO**

- **简介: 该论文针对椭圆形扩展目标跟踪任务，旨在同时估计目标运动学、朝向与轴长。提出基于状态分量解耦的二次型卡尔曼滤波器，降低近似误差；并设计批处理变体，提升效率与精度，经仿真和实车雷达数据验证优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.14426v1](https://arxiv.org/pdf/2512.14426v1)**

> **作者:** Simon Steuernagel; Marcus Baum
>
> **备注:** 13 pages, 8 figures, submitted to IEEE Transactions on Aerospace and Electronic Systems
>
> **摘要:** Extended object tracking involves estimating both the physical extent and kinematic parameters of a target object, where typically multiple measurements are observed per time step. In this article, we propose a deterministic closed-form elliptical extended object tracker, based on decoupling of the kinematics, orientation, and axis lengths. By disregarding potential correlations between these state components, fewer approximations are required for the individual estimators than for an overall joint solution. The resulting algorithm outperforms existing algorithms, reaching the accuracy of sampling-based procedures. Additionally, a batch-based variant is introduced, yielding highly efficient computation while outperforming all comparable state-of-the-art algorithms. This is validated both by a simulation study using common models from literature, as well as an extensive quantitative evaluation on real automotive radar data.
>
---
## 更新

#### [replaced 001] Mirror Skin: In Situ Visualization of Robot Touch Intent on Robotic Skin
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出“Mirror Skin”，属人机交互中的意图可视化任务，旨在解决机器人触碰意图缺乏空间与语义明确性的问题。通过仿生镜面皮肤实时映射人体对应部位，实现“谁、何处、何时”触碰的原位视觉反馈，并经VR专家设计探索与用户实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.11472v2](https://arxiv.org/pdf/2512.11472v2)**

> **作者:** David Wagmann; Matti Krüger; Chao Wang; Jürgen Steimle
>
> **摘要:** Effective communication of robotic touch intent is a key factor in promoting safe and predictable physical human-robot interaction (pHRI). While intent communication has been widely studied, existing approaches lack the spatial specificity and semantic depth necessary to convey robot touch actions. We present Mirror Skin, a cephalopod-inspired concept that utilizes high-resolution, mirror-like visual feedback on robotic skin. By mapping in-situ visual representations of a human's body parts onto the corresponding robot's touch region, Mirror Skin communicates who shall initiate touch, where it will occur, and when it is imminent. To inform the design of Mirror Skin, we conducted a structured design exploration with experts in virtual reality (VR), iteratively refining six key dimensions. A subsequent controlled user study demonstrated that Mirror Skin significantly enhances accuracy and reduces response times for interpreting touch intent. These findings highlight the potential of visual feedback on robotic skin to communicate human-robot touch interactions.
>
---
#### [replaced 002] Semantic-Drive: Democratizing Long-Tail Data Curation via Open-Vocabulary Grounding and Neuro-Symbolic VLM Consensus
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出Semantic-Drive，解决自动驾驶中长尾安全事件（如异常闯入）数据难挖掘的问题。它采用本地化、神经符号融合框架：先用YOLOE进行开放词汇语义定位，再通过多模型共识的推理型VLM做细粒度场景分析，在保护隐私前提下显著提升召回率与风险评估精度。**

- **链接: [https://arxiv.org/pdf/2512.12012v2](https://arxiv.org/pdf/2512.12012v2)**

> **作者:** Antonio Guillen-Perez
>
> **摘要:** The development of robust Autonomous Vehicles (AVs) is bottlenecked by the scarcity of "Long-Tail" training data. While fleets collect petabytes of video logs, identifying rare safety-critical events (e.g., erratic jaywalking, construction diversions) remains a manual, cost-prohibitive process. Existing solutions rely on coarse metadata search, which lacks precision, or cloud-based VLMs, which are privacy-invasive and expensive. We introduce Semantic-Drive, a local-first, neuro-symbolic framework for semantic data mining. Our approach decouples perception into two stages: (1) Symbolic Grounding via a real-time open-vocabulary detector (YOLOE) to anchor attention, and (2) Cognitive Analysis via a Reasoning VLM that performs forensic scene analysis. To mitigate hallucination, we implement a "System 2" inference-time alignment strategy, utilizing a multi-model "Judge-Scout" consensus mechanism. Benchmarked on the nuScenes dataset against the Waymo Open Dataset (WOD-E2E) taxonomy, Semantic-Drive achieves a Recall of 0.966 (vs. 0.475 for CLIP) and reduces Risk Assessment Error by 40% ccompared to the best single scout models. The system runs entirely on consumer hardware (NVIDIA RTX 3090), offering a privacy-preserving alternative to the cloud.
>
---
#### [replaced 003] MAPS$^2$: Multi-Robot Autonomous Motion Planning under Signal Temporal Logic Specifications
- **分类: cs.RO**

- **简介: 该论文提出MAPS²，一种面向多机器人系统的分布式运动规划算法，解决在信号时序逻辑（STL）约束下协同完成耦合任务的问题。它通过时空解耦、邻居通信与概率保证的迭代优化，生成满足STL的轨迹，兼具实时性、分布性与完整性保障。**

- **链接: [https://arxiv.org/pdf/2309.05632v3](https://arxiv.org/pdf/2309.05632v3)**

> **作者:** Mayank Sewlia; Christos K. Verginis; Dimos V. Dimarogonas
>
> **摘要:** This article presents MAPS$^2$ : a distributed algorithm that allows multi-robot systems to deliver coupled tasks expressed as Signal Temporal Logic (STL) constraints. Classical control theoretical tools addressing STL constraints either adopt a limited fragment of the STL formula or require approximations of min/max operators, whereas works maximising robustness through optimisation-based methods often suffer from local minima, relaxing any completeness arguments due to the NP-hard nature of the problem. Endowed with probabilistic guarantees, MAPS$^2$ provides an anytime algorithm that iteratively improves the robots' trajectories. The algorithm selectively imposes spatial constraints by taking advantage of the temporal properties of the STL. The algorithm is distributed, in the sense that each robot calculates its trajectory by communicating only with its immediate neighbours as defined via a communication graph. We illustrate the efficiency of MAPS$^2$ by conducting extensive simulation and experimental studies, verifying the generation of STL satisfying trajectories.
>
---
#### [replaced 004] MindDrive: A Vision-Language-Action Model for Autonomous Driving via Online Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MindDrive，属自动驾驶中的视觉-语言-动作（VLA）任务，旨在解决模仿学习导致的分布偏移与因果混淆问题。通过双LoRA微调轻量LLM，将在线强化学习从连续动作空间转为离散语言决策空间，实现高效探索与人类化驾驶行为。**

- **链接: [https://arxiv.org/pdf/2512.13636v2](https://arxiv.org/pdf/2512.13636v2)**

> **作者:** Haoyu Fu; Diankun Zhang; Zongchuang Zhao; Jianfeng Cui; Hongwei Xie; Bing Wang; Guang Chen; Dingkang Liang; Xiang Bai
>
> **备注:** 16 pages, 12 figures, 6 tables; Project Page: https://xiaomi-mlab.github.io/MindDrive/
>
> **摘要:** Current Vision-Language-Action (VLA) paradigms in autonomous driving primarily rely on Imitation Learning (IL), which introduces inherent challenges such as distribution shift and causal confusion. Online Reinforcement Learning offers a promising pathway to address these issues through trial-and-error learning. However, applying online reinforcement learning to VLA models in autonomous driving is hindered by inefficient exploration in continuous action spaces. To overcome this limitation, we propose MindDrive, a VLA framework comprising a large language model (LLM) with two distinct sets of LoRA parameters. The one LLM serves as a Decision Expert for scenario reasoning and driving decision-making, while the other acts as an Action Expert that dynamically maps linguistic decisions into feasible trajectories. By feeding trajectory-level rewards back into the reasoning space, MindDrive enables trial-and-error learning over a finite set of discrete linguistic driving decisions, instead of operating directly in a continuous action space. This approach effectively balances optimal decision-making in complex scenarios, human-like driving behavior, and efficient exploration in online reinforcement learning. Using the lightweight Qwen-0.5B LLM, MindDrive achieves Driving Score (DS) of 78.04 and Success Rate (SR) of 55.09% on the challenging Bench2Drive benchmark. To the best of our knowledge, this is the first work to demonstrate the effectiveness of online reinforcement learning for the VLA model in autonomous driving.
>
---
#### [replaced 005] Closing the Loop: Motion Prediction Models beyond Open-Loop Benchmarks
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属自动驾驶运动预测任务，旨在解决“高开环精度模型未必提升闭环驾驶性能”的问题。作者系统评估预测模型与规划器的协同效果，发现时间一致性、 planner 兼容性比单纯精度更重要，并验证轻量模型在闭环中可媲美或超越大模型。**

- **链接: [https://arxiv.org/pdf/2505.05638v2](https://arxiv.org/pdf/2505.05638v2)**

> **作者:** Mohamed-Khalil Bouzidi; Christian Schlauch; Nicole Scheuerer; Yue Yao; Nadja Klein; Daniel Göhring; Jörg Reichardt
>
> **摘要:** Fueled by motion prediction competitions and benchmarks, recent years have seen the emergence of increasingly large learning based prediction models, many with millions of parameters, focused on improving open-loop prediction accuracy by mere centimeters. However, these benchmarks fail to assess whether such improvements translate to better performance when integrated into an autonomous driving stack. In this work, we systematically evaluate the interplay between state-of-the-art motion predictors and motion planners. Our results show that higher open-loop accuracy does not always correlate with better closed-loop driving behavior and that other factors, such as temporal consistency of predictions and planner compatibility, also play a critical role. Furthermore, we investigate downsized variants of these models, and, surprisingly, find that in some cases models with up to 86% fewer parameters yield comparable or even superior closed-loop driving performance. Our code is available at https://github.com/aumovio/pred2plan.
>
---
#### [replaced 006] Decomposed Object Manipulation via Dual-Actor Policy
- **分类: cs.RO**

- **简介: 该论文面向物体操控任务，解决单策略忽视“接近—操控”两阶段特性的问题。提出双执行器策略（DAP）：基于功能性的执行器定位目标部件，基于运动流的执行器引导操作，并引入决策器切换阶段；构建含双视觉先验的仿真数据集，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.05129v2](https://arxiv.org/pdf/2511.05129v2)**

> **作者:** Bin Fan; Jian-Jian Jiang; Zhuohao Li; Xiao-Ming Wu; Yi-Xiang He; YiHan Yang; Shengbang Liu; Wei-Shi Zheng
>
> **摘要:** Object manipulation, which focuses on learning to perform tasks on similar parts across different types of objects, can be divided into an approaching stage and a manipulation stage. However, previous works often ignore this characteristic of the task and rely on a single policy to directly learn the whole process of object manipulation. To address this problem, we propose a novel Dual-Actor Policy, termed DAP, which explicitly considers different stages and leverages heterogeneous visual priors to enhance each stage. Specifically, we introduce an affordance-based actor to locate the functional part in the manipulation task, thereby improving the approaching process. Following this, we propose a motion flow-based actor to capture the movement of the component, facilitating the manipulation process. Finally, we introduce a decision maker to determine the current stage of DAP and select the corresponding actor. Moreover, existing object manipulation datasets contain few objects and lack the visual priors needed to support training. To address this, we construct a simulated dataset, the Dual-Prior Object Manipulation Dataset, which combines the two visual priors and includes seven tasks, including two challenging long-term, multi-stage tasks. Experimental results on our dataset, the RoboTwin benchmark and real-world scenarios illustrate that our method consistently outperforms the SOTA method by 5.55%, 14.7% and 10.4% on average respectively.
>
---
#### [replaced 007] Data-fused MPC with Guarantees: Application to Flying Humanoid Robots
- **分类: eess.SY; cs.RO**

- **简介: 该论文提出数据融合型模型预测控制（DFMPC），解决飞行人形机器人在未知动态、测量噪声和约束下的高精度轨迹跟踪问题。工作包括融合物理模型与数据驱动模型，引入人工平衡点与Willems引理，保证递归可行性和实用稳定性，并在iRonCub平台上验证。**

- **链接: [https://arxiv.org/pdf/2509.10353v4](https://arxiv.org/pdf/2509.10353v4)**

> **作者:** Davide Gorbani; Mohamed Elobaid; Giuseppe L'Erario; Hosameldin Awadalla Omer Mohamed; Daniele Pucci
>
> **备注:** This paper has been accepted for publication in IEEE Control Systems Letters (L-CSS)
>
> **摘要:** This paper introduces a Data-Fused Model Predictive Control (DFMPC) framework that combines physics-based models with data-driven representations of unknown dynamics. Leveraging Willems' Fundamental Lemma and an artificial equilibrium formulation, the method enables tracking of changing, potentially unreachable setpoints while explicitly handling measurement noise through slack variables and regularization. We provide guarantees of recursive feasibility and practical stability under input-output constraints for a specific class of reference signals. The approach is validated on the iRonCub flying humanoid robot, integrating analytical momentum models with data-driven turbine dynamics. Simulations show improved tracking and robustness compared to a purely model-based MPC, while maintaining real-time feasibility.
>
---
#### [replaced 008] Intrinsic-Motivation Multi-Robot Social Formation Navigation with Coordinated Exploration
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究多机器人社交编队导航任务，旨在解决行人行为不可预测导致的协同探索低效问题。提出一种带内在动机探索的多机器人RL算法，含自学习内在奖励机制和双采样模式，提升策略与奖励表征能力。**

- **链接: [https://arxiv.org/pdf/2512.13293v2](https://arxiv.org/pdf/2512.13293v2)**

> **作者:** Hao Fu; Wei Liu; Shuai Zhou
>
> **摘要:** This paper investigates the application of reinforcement learning (RL) to multi-robot social formation navigation, a critical capability for enabling seamless human-robot coexistence. While RL offers a promising paradigm, the inherent unpredictability and often uncooperative dynamics of pedestrian behavior pose substantial challenges, particularly concerning the efficiency of coordinated exploration among robots. To address this, we propose a novel coordinated-exploration multi-robot RL algorithm introducing an intrinsic motivation exploration. Its core component is a self-learning intrinsic reward mechanism designed to collectively alleviate policy conservatism. Moreover, this algorithm incorporates a dual-sampling mode within the centralized training and decentralized execution framework to enhance the representation of both the navigation policy and the intrinsic reward, leveraging a two-time-scale update rule to decouple parameter updates. Empirical results on social formation navigation benchmarks demonstrate the proposed algorithm's superior performance over existing state-of-the-art methods across crucial metrics. Our code and video demos are available at: https://github.com/czxhunzi/CEMRRL.
>
---
#### [replaced 009] Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文是一篇综述，聚焦多智能体人类轨迹预测任务，旨在解决行人交互建模难题。它系统梳理了2020–2025年基于深度学习的最新方法，按架构、输入表示和预测策略分类，重点分析ETH/UCY基准上的模型，并指出关键挑战与未来方向。**

- **链接: [https://arxiv.org/pdf/2506.14831v2](https://arxiv.org/pdf/2506.14831v2)**

> **作者:** Céline Finet; Stephane Da Silva Martins; Jean-Bernard Hayet; Ioannis Karamouzas; Javad Amirian; Sylvie Le Hégarat-Mascle; Julien Pettré; Emanuel Aldea
>
> **备注:** 45 pages
>
> **摘要:** With the emergence of powerful data-driven methods in human trajectory prediction (HTP), gaining a finer understanding of multi-agent interactions lies within hand's reach, with important implications in areas such as social robot navigation, autonomous navigation, and crowd modeling. This survey reviews some of the most recent advancements in deep learning-based multi-agent trajectory prediction, focusing on studies published between 2020 and 2025. We categorize the existing methods based on their architectural design, their input representations, and their overall prediction strategies, placing a particular emphasis on models evaluated using the ETH/UCY benchmark. Furthermore, we highlight key challenges and future research directions in the field of multi-agent HTP.
>
---
#### [replaced 010] MMDrive: Interactive Scene Understanding Beyond Vision with Multi-representational Fusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向自动驾驶场景理解任务，解决现有视觉语言模型受限于2D图像、难以融合3D空间信息的问题。提出MMDrive框架，融合占用图、LiDAR点云和文本描述，设计文本导向调制器与跨模态抽象器，实现自适应多模态融合，在DriveLM和NuScenes-QA上显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.13177v2](https://arxiv.org/pdf/2512.13177v2)**

> **作者:** Minghui Hou; Wei-Hsing Huang; Shaofeng Liang; Daizong Liu; Tai-Hao Wen; Gang Wang; Runwei Guan; Weiping Ding
>
> **摘要:** Vision-language models enable the understanding and reasoning of complex traffic scenarios through multi-source information fusion, establishing it as a core technology for autonomous driving. However, existing vision-language models are constrained by the image understanding paradigm in 2D plane, which restricts their capability to perceive 3D spatial information and perform deep semantic fusion, resulting in suboptimal performance in complex autonomous driving environments. This study proposes MMDrive, an multimodal vision-language model framework that extends traditional image understanding to a generalized 3D scene understanding framework. MMDrive incorporates three complementary modalities, including occupancy maps, LiDAR point clouds, and textual scene descriptions. To this end, it introduces two novel components for adaptive cross-modal fusion and key information extraction. Specifically, the Text-oriented Multimodal Modulator dynamically weights the contributions of each modality based on the semantic cues in the question, guiding context-aware feature integration. The Cross-Modal Abstractor employs learnable abstract tokens to generate compact, cross-modal summaries that highlight key regions and essential semantics. Comprehensive evaluations on the DriveLM and NuScenes-QA benchmarks demonstrate that MMDrive achieves significant performance gains over existing vision-language models for autonomous driving, with a BLEU-4 score of 54.56 and METEOR of 41.78 on DriveLM, and an accuracy score of 62.7% on NuScenes-QA. MMDrive effectively breaks the traditional image-only understanding barrier, enabling robust multimodal reasoning in complex driving environments and providing a new foundation for interpretable autonomous driving scene understanding.
>
---
