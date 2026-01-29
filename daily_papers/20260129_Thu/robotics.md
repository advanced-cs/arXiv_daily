# 机器人 cs.RO

- **最新发布 23 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] E2HiL: Entropy-Guided Sample Selection for Efficient Real-World Human-in-the-Loop Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决人机交互强化学习中的样本效率问题。通过熵引导的样本选择方法，减少人类干预，提升学习效率。**

- **链接: [https://arxiv.org/pdf/2601.19969v1](https://arxiv.org/pdf/2601.19969v1)**

> **作者:** Haoyuan Deng; Yuanjiang Xue; Haoyang Du; Boyang Zhou; Zhenyu Wu; Ziwei Wang
>
> **备注:** Project page: https://e2hil.github.io/
>
> **摘要:** Human-in-the-loop guidance has emerged as an effective approach for enabling faster convergence in online reinforcement learning (RL) of complex real-world manipulation tasks. However, existing human-in-the-loop RL (HiL-RL) frameworks often suffer from low sample efficiency, requiring substantial human interventions to achieve convergence and thereby leading to high labor costs. To address this, we propose a sample-efficient real-world human-in-the-loop RL framework named \method, which requires fewer human intervention by actively selecting informative samples. Specifically, stable reduction of policy entropy enables improved trade-off between exploration and exploitation with higher sample efficiency. We first build influence functions of different samples on the policy entropy, which is efficiently estimated by the covariance of action probabilities and soft advantages of policies. Then we select samples with moderate values of influence functions, where shortcut samples that induce sharp entropy drops and noisy samples with negligible effect are pruned. Extensive experiments on four real-world manipulation tasks demonstrate that \method achieves a 42.1\% higher success rate while requiring 10.1\% fewer human interventions compared to the state-of-the-art HiL-RL method, validating its effectiveness. The project page providing code, videos, and mathematical formulations can be found at https://e2hil.github.io/.
>
---
#### [new 002] A Methodology for Designing Knowledge-Driven Missions for Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人自主任务设计，旨在提升任务效率与智能性。通过构建知识图谱，优化任务规划与执行，解决传统方法在复杂环境中的决策不足问题。**

- **链接: [https://arxiv.org/pdf/2601.20797v1](https://arxiv.org/pdf/2601.20797v1)**

> **作者:** Guillermo GP-Lenza; Carmen DR. Pita-Romero; Miguel Fernandez-Cortizas; Pascual Campoy
>
> **摘要:** This paper presents a comprehensive methodology for implementing knowledge graphs in ROS 2 systems, aiming to enhance the efficiency and intelligence of autonomous robotic missions. The methodology encompasses several key steps: defining initial and target conditions, structuring tasks and subtasks, planning their sequence, representing task-related data in a knowledge graph, and designing the mission using a high-level language. Each step builds on the previous one to ensure a cohesive process from initial setup to final execution. A practical implementation within the Aerostack2 framework is demonstrated through a simulated search and rescue mission in a Gazebo environment, where drones autonomously locate a target. This implementation highlights the effectiveness of the methodology in improving decision-making and mission performance by leveraging knowledge graphs.
>
---
#### [new 003] STORM: Slot-based Task-aware Object-centric Representation for robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉基础模型缺乏明确对象结构的问题。提出STORM模块，通过多阶段训练将通用特征转化为任务感知的对象中心表示，提升操作性能和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.20381v1](https://arxiv.org/pdf/2601.20381v1)**

> **作者:** Alexandre Chapin; Emmanuel Dellandréa; Liming Chen
>
> **摘要:** Visual foundation models provide strong perceptual features for robotics, but their dense representations lack explicit object-level structure, limiting robustness and contractility in manipulation tasks. We propose STORM (Slot-based Task-aware Object-centric Representation for robotic Manipulation), a lightweight object-centric adaptation module that augments frozen visual foundation models with a small set of semantic-aware slots for robotic manipulation. Rather than retraining large backbones, STORM employs a multi-phase training strategy: object-centric slots are first stabilized through visual--semantic pretraining using language embeddings, then jointly adapted with a downstream manipulation policy. This staged learning prevents degenerate slot formation and preserves semantic consistency while aligning perception with task objectives. Experiments on object discovery benchmarks and simulated manipulation tasks show that STORM improves generalization to visual distractors, and control performance compared to directly using frozen foundation model features or training object-centric representations end-to-end. Our results highlight multi-phase adaptation as an efficient mechanism for transforming generic foundation model features into task-aware object-centric representations for robotic control.
>
---
#### [new 004] Real-Time Robot Execution with Masked Action Chunking
- **分类: cs.RO**

- **简介: 该论文属于机器人实时执行任务，解决异步推理中的执行失败问题。提出REMAC方法，通过掩码动作分块提升策略鲁棒性，确保任务高效可靠完成。**

- **链接: [https://arxiv.org/pdf/2601.20130v1](https://arxiv.org/pdf/2601.20130v1)**

> **作者:** Haoxuan Wang; Gengyu Zhang; Yan Yan; Yuzhang Shang; Ramana Rao Kompella; Gaowen Liu
>
> **备注:** ICLR 2026. Project page at https://remac-async.github.io/
>
> **摘要:** Real-time execution is essential for cyber-physical systems such as robots. These systems operate in dynamic real-world environments where even small delays can undermine responsiveness and compromise performance. Asynchronous inference has recently emerged as a system-level paradigm for real-time robot manipulation, enabling the next action chunk to be predicted while the current one is being executed. While this approach achieves real-time responsiveness, naive integration often results in execution failure. Previous methods attributed this failure to inter-chunk discontinuity and developed test-time algorithms to smooth chunk boundaries. In contrast, we identify another critical yet overlooked factor: intra-chunk inconsistency, where the robot's executed action chunk partially misaligns with its current perception. To address this, we propose REMAC, which learns corrective adjustments on the pretrained policy through masked action chunking, enabling the policy to remain resilient under mismatches between intended actions and actual execution during asynchronous inference. In addition, we introduce a prefix-preserved sampling procedure to reinforce inter-chunk continuity. Overall, our method delivers more reliable policies without incurring additional latency. Extensive experiments in both simulation and real-world settings demonstrate that our method enables faster task execution, maintains robustness across varying delays, and consistently achieves higher completion rates.
>
---
#### [new 005] TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance
- **分类: cs.RO**

- **简介: 该论文提出TouchGuide，解决机器人接触丰富操作中触觉反馈利用不足的问题。通过视觉-触觉融合和物理模型引导，提升动作的物理可行性。**

- **链接: [https://arxiv.org/pdf/2601.20239v1](https://arxiv.org/pdf/2601.20239v1)**

> **作者:** Zhemeng Zhang; Jiahua Ma; Xincheng Yang; Xin Wen; Yuzhi Zhang; Boyan Li; Yiran Qin; Jin Liu; Can Zhao; Li Kang; Haoqin Hong; Zhenfei Yin; Philip Torr; Hao Su; Ruimao Zhang; Daolin Ma
>
> **摘要:** Fine-grained and contact-rich manipulation remain challenging for robots, largely due to the underutilization of tactile feedback. To address this, we introduce TouchGuide, a novel cross-policy visuo-tactile fusion paradigm that fuses modalities within a low-dimensional action space. Specifically, TouchGuide operates in two stages to guide a pre-trained diffusion or flow-matching visuomotor policy at inference time. First, the policy produces a coarse, visually-plausible action using only visual inputs during early sampling. Second, a task-specific Contact Physical Model (CPM) provides tactile guidance to steer and refine the action, ensuring it aligns with realistic physical contact conditions. Trained through contrastive learning on limited expert demonstrations, the CPM provides a tactile-informed feasibility score to steer the sampling process toward refined actions that satisfy physical contact constraints. Furthermore, to facilitate TouchGuide training with high-quality and cost-effective data, we introduce TacUMI, a data collection system. TacUMI achieves a favorable trade-off between precision and affordability; by leveraging rigid fingertips, it obtains direct tactile feedback, thereby enabling the collection of reliable tactile data. Extensive experiments on five challenging contact-rich tasks, such as shoe lacing and chip handover, show that TouchGuide consistently and significantly outperforms state-of-the-art visuo-tactile policies.
>
---
#### [new 006] One Step Is Enough: Dispersive MeanFlow Policy Optimization
- **分类: cs.RO**

- **简介: 该论文提出DMPO框架，解决实时机器人控制中生成动作速度慢的问题。通过单步推断、分散正则化和强化学习优化，实现高效动作生成，提升控制频率。**

- **链接: [https://arxiv.org/pdf/2601.20701v1](https://arxiv.org/pdf/2601.20701v1)**

> **作者:** Guowei Zou; Haitao Wang; Hejun Wu; Yukun Qian; Yuhang Wang; Weibing Li
>
> **备注:** Code and project page: https://guowei-zou.github.io/dmpo-page/
>
> **摘要:** Real-time robotic control demands fast action generation. However, existing generative policies based on diffusion and flow matching require multi-step sampling, fundamentally limiting deployment in time-critical scenarios. We propose Dispersive MeanFlow Policy Optimization (DMPO), a unified framework that enables true one-step generation through three key components: MeanFlow for mathematically-derived single-step inference without knowledge distillation, dispersive regularization to prevent representation collapse, and reinforcement learning (RL) fine-tuning to surpass expert demonstrations. Experiments across RoboMimic manipulation and OpenAI Gym locomotion benchmarks demonstrate competitive or superior performance compared to multi-step baselines. With our lightweight model architecture and the three key algorithmic components working in synergy, DMPO exceeds real-time control requirements (>120Hz) with 5-20x inference speedup, reaching hundreds of Hertz on high-performance GPUs. Physical deployment on a Franka-Emika-Panda robot validates real-world applicability.
>
---
#### [new 007] Shallow-π: Knowledge Distillation for Flow-based VLAs
- **分类: cs.RO**

- **简介: 该论文属于视觉语言动作模型压缩任务，旨在提升模型推理速度。通过知识蒸馏减少Transformer层数，实现快速且高效的模型部署。**

- **链接: [https://arxiv.org/pdf/2601.20262v1](https://arxiv.org/pdf/2601.20262v1)**

> **作者:** Boseong Jeon; Yunho Choi; Taehan Kim
>
> **摘要:** The growing demand for real-time robotic deployment necessitates fast and on-device inference for vision-language-action (VLA) models. Within the VLA literature, efficiency has been extensively studied at the token level, such as visual token pruning. In contrast, systematic transformer layer reduction has received limited attention and, to the best of our knowledge, has not been explored for flow-based VLA models under knowledge distillation. In this work, we propose Shallow-pi, a principled knowledge distillation framework that aggressively reduces the transformer depth of both the VLM backbone and the flow-based action head, compressing the model from 18 to 6 layers. Shallow-pi achieves over two times faster inference with less than one percent absolute drop in success rate on standard manipulation benchmarks, establishing state-of-the-art performance among reduced VLA models. Crucially, we validate our approach through industrial-scale real-world experiments on Jetson Orin and Jetson Thor across multiple robot platforms, including humanoid systems, in complex and dynamic manipulation scenarios.
>
---
#### [new 008] Tactile-Force Alignment in Vision-Language-Action Models for Force-aware Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型缺乏物理直觉的问题。通过引入触觉-力对齐机制，提升接触密集任务中的力感知能力。**

- **链接: [https://arxiv.org/pdf/2601.20321v1](https://arxiv.org/pdf/2601.20321v1)**

> **作者:** Yuzhe Huang; Pei Lin; Wanlin Li; Daohan Li; Jiajun Li; Jiaming Jiang; Chenxi Xiao; Ziyuan Jiao
>
> **备注:** 17pages,9fig
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged as powerful generalists for robotic manipulation. However, due to their predominant reliance on visual modalities, they fundamentally lack the physical intuition required for contact-rich tasks that require precise force regulation and physical reasoning. Existing attempts to incorporate vision-based tactile sensing into VLA models typically treat tactile inputs as auxiliary visual textures, thereby overlooking the underlying correlation between surface deformation and interaction dynamics. To bridge this gap, we propose a paradigm shift from tactile-vision alignment to tactile-force alignment. Here, we introduce TaF-VLA, a framework that explicitly grounds high-dimensional tactile observations in physical interaction forces. To facilitate this, we develop an automated tactile-force data acquisition device and curate the TaF-Dataset, comprising over 10 million synchronized tactile observations, 6-axis force/torque, and matrix force map. To align sequential tactile observations with interaction forces, the central component of our approach is the Tactile-Force Adapter (TaF-Adapter), a tactile sensor encoder that extracts discretized latent information for encoding tactile observations. This mechanism ensures that the learned representations capture history-dependent, noise-insensitive physical dynamics rather than static visual textures. Finally, we integrate this force-aligned encoder into a VLA backbone. Extensive real-world experiments demonstrate that TaF-VLA policy significantly outperforms state-of-the-art tactile-vision-aligned and vision-only baselines on contact-rich tasks, verifying its ability to achieve robust, force-aware manipulation through cross-modal physical reasoning.
>
---
#### [new 009] Tendon-based modelling, estimation and control for a simulated high-DoF anthropomorphic hand model
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决无直接关节传感的肌腱驱动机械手的位姿估计与控制问题。通过肌腱位移和张力估计关节角度，并实现闭环控制。**

- **链接: [https://arxiv.org/pdf/2601.20682v1](https://arxiv.org/pdf/2601.20682v1)**

> **作者:** Péter Polcz; Katalin Schäffer; Miklós Koller
>
> **摘要:** Tendon-driven anthropomorphic robotic hands often lack direct joint angle sensing, as the integration of joint encoders can compromise mechanical compactness and dexterity. This paper presents a computational method for estimating joint positions from measured tendon displacements and tensions. An efficient kinematic modeling framework for anthropomorphic hands is first introduced based on the Denavit-Hartenberg convention. Using a simplified tendon model, a system of nonlinear equations relating tendon states to joint positions is derived and solved via a nonlinear optimization approach. The estimated joint angles are then employed for closed-loop control through a Jacobian-based proportional-integral (PI) controller augmented with a feedforward term, enabling gesture tracking without direct joint sensing. The effectiveness and limitations of the proposed estimation and control framework are demonstrated in the MuJoCo simulation environment using the Anatomically Correct Biomechatronic Hand, featuring five degrees of freedom for each long finger and six degrees of freedom for the thumb.
>
---
#### [new 010] Vibro-Sense: Robust Vibration-based Impulse Response Localization and Trajectory Tracking for Robotic Hands
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于机器人触觉感知任务，旨在解决低成本高精度接触定位与轨迹跟踪问题。通过振动信号分析实现机器人手部接触感知，具有高精度和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.20555v1](https://arxiv.org/pdf/2601.20555v1)**

> **作者:** Wadhah Zai El Amri; Nicolás Navarro-Guerrero
>
> **备注:** Under Review: Springer Autonomous Robots Journal
>
> **摘要:** Rich contact perception is crucial for robotic manipulation, yet traditional tactile skins remain expensive and complex to integrate. This paper presents a scalable alternative: high-accuracy whole-body touch localization via vibro-acoustic sensing. By equipping a robotic hand with seven low-cost piezoelectric microphones and leveraging an Audio Spectrogram Transformer, we decode the vibrational signatures generated during physical interaction. Extensive evaluation across stationary and dynamic tasks reveals a localization error of under 5 mm in static conditions. Furthermore, our analysis highlights the distinct influence of material properties: stiff materials (e.g., metal) excel in impulse response localization due to sharp, high-bandwidth responses, whereas textured materials (e.g., wood) provide superior friction-based features for trajectory tracking. The system demonstrates robustness to the robot's own motion, maintaining effective tracking even during active operation. Our primary contribution is demonstrating that complex physical contact dynamics can be effectively decoded from simple vibrational signals, offering a viable pathway to widespread, affordable contact perception in robotics. To accelerate research, we provide our full datasets, models, and experimental setups as open-source resources.
>
---
#### [new 011] End-to-end example-based sim-to-real RL policy transfer based on neural stylisation with application to robotic cutting
- **分类: cs.RO**

- **简介: 该论文属于机器人强化学习的sim-to-real迁移任务，旨在解决仿真与现实环境差异导致的部署难题。通过神经风格迁移生成真实数据，提升策略在物理环境中的表现。**

- **链接: [https://arxiv.org/pdf/2601.20846v1](https://arxiv.org/pdf/2601.20846v1)**

> **作者:** Jamie Hathaway; Alireza Rastegarpanah; Rustam Stolkin
>
> **备注:** 14 pages, 9 figures. Submitted to Nature Scientific Reports
>
> **摘要:** Whereas reinforcement learning has been applied with success to a range of robotic control problems in complex, uncertain environments, reliance on extensive data - typically sourced from simulation environments - limits real-world deployment due to the domain gap between simulated and physical systems, coupled with limited real-world sample availability. We propose a novel method for sim-to-real transfer of reinforcement learning policies, based on a reinterpretation of neural style transfer from image processing to synthesise novel training data from unpaired unlabelled real world datasets. We employ a variational autoencoder to jointly learn self-supervised feature representations for style transfer and generate weakly paired source-target trajectories to improve physical realism of synthesised trajectories. We demonstrate the application of our approach based on the case study of robot cutting of unknown materials. Compared to baseline methods, including our previous work, CycleGAN, and conditional variational autoencoder-based time series translation, our approach achieves improved task completion time and behavioural stability with minimal real-world data. Our framework demonstrates robustness to geometric and material variation, and highlights the feasibility of policy adaptation in challenging contact-rich tasks where real-world reward information is unavailable.
>
---
#### [new 012] RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于RF材料识别任务，旨在解决缺乏大规模数据集和基准的问题。提出RF-MatID数据集，包含142k样本，用于提升识别精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.20377v1](https://arxiv.org/pdf/2601.20377v1)**

> **作者:** Xinyan Chen; Qinchun Li; Ruiqin Ma; Jiaqi Bai; Li Yi; Jianfei Yang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Accurate material identification plays a crucial role in embodied AI systems, enabling a wide range of applications. However, current vision-based solutions are limited by the inherent constraints of optical sensors, while radio-frequency (RF) approaches, which can reveal intrinsic material properties, have received growing attention. Despite this progress, RF-based material identification remains hindered by the lack of large-scale public datasets and the limited benchmarking of learning-based approaches. In this work, we present RF-MatID, the first open-source, large-scale, wide-band, and geometry-diverse RF dataset for fine-grained material identification. RF-MatID includes 16 fine-grained categories grouped into 5 superclasses, spanning a broad frequency range from 4 to 43.5 GHz, and comprises 142k samples in both frequency- and time-domain representations. The dataset systematically incorporates controlled geometry perturbations, including variations in incidence angle and stand-off distance. We further establish a multi-setting, multi-protocol benchmark by evaluating state-of-the-art deep learning models, assessing both in-distribution performance and out-of-distribution robustness under cross-angle and cross-distance shifts. The 5 frequency-allocation protocols enable systematic frequency- and region-level analysis, thereby facilitating real-world deployment. RF-MatID aims to enable reproducible research, accelerate algorithmic advancement, foster cross-domain robustness, and support the development of real-world application in RF-based material identification.
>
---
#### [new 013] Learning From a Steady Hand: A Weakly Supervised Agent for Robot Assistance under Microscopy
- **分类: cs.RO**

- **简介: 该论文属于机器人辅助任务，解决显微镜下微操作的可靠性问题。通过弱监督框架实现无需外部标记的精准控制，提升操作效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.20776v1](https://arxiv.org/pdf/2601.20776v1)**

> **作者:** Huanyu Tian; Martin Huber; Lingyun Zeng; Zhe Han; Wayne Bennett; Giuseppe Silvestri; Gerardo Mendizabal-Ruiz; Tom Vercauteren; Alejandro Chavez-Badiola; Christos Bergeles
>
> **摘要:** This paper rethinks steady-hand robotic manipulation by using a weakly supervised framework that fuses calibration-aware perception with admittance control. Unlike conventional automation that relies on labor-intensive 2D labeling, our framework leverages reusable warm-up trajectories to extract implicit spatial information, thereby achieving calibration-aware, depth-resolved perception without the need for external fiducials or manual depth annotation. By explicitly characterizing residuals from observation and calibration models, the system establishes a task-space error budget from recorded warm-ups. The uncertainty budget yields a lateral closed-loop accuracy of approx. 49 micrometers at 95% confidence (worst-case testing subset) and a depth accuracy of <= 291 micrometers at 95% confidence bound during large in-plane moves. In a within-subject user study (N=8), the learned agent reduces overall NASA-TLX workload by 77.1% relative to the simple steady-hand assistance baseline. These results demonstrate that the weakly supervised agent improves the reliability of microscope-guided biomedical micromanipulation without introducing complex setup requirements, offering a practical framework for microscope-guided intervention.
>
---
#### [new 014] MeCo: Enhancing LLM-Empowered Multi-Robot Collaboration via Similar Task Memoization
- **分类: cs.RO**

- **简介: 该论文属于多机器人协作任务，旨在解决LLM驱动的协作系统在相似任务中重复规划效率低的问题。提出MeCo框架，通过相似任务记忆实现计划复用，提升效率与成功率。**

- **链接: [https://arxiv.org/pdf/2601.20577v1](https://arxiv.org/pdf/2601.20577v1)**

> **作者:** Baiqing Wang; Helei Cui; Bo Zhang; Xiaolong Zheng; Bin Guo; Zhiwen Yu
>
> **摘要:** Multi-robot systems have been widely deployed in real-world applications, providing significant improvements in efficiency and reductions in labor costs. However, most existing multi-robot collaboration methods rely on extensive task-specific training, which limits their adaptability to new or diverse scenarios. Recent research leverages the language understanding and reasoning capabilities of large language models (LLMs) to enable more flexible collaboration without specialized training. Yet, current LLM-empowered approaches remain inefficient: when confronted with identical or similar tasks, they must replan from scratch because they omit task-level similarities. To address this limitation, we propose MeCo, a similarity-aware multi-robot collaboration framework that applies the principle of ``cache and reuse'' (a.k.a., memoization) to reduce redundant computation. Unlike simple task repetition, identifying and reusing solutions for similar but not identical tasks is far more challenging, particularly in multi-robot settings. To this end, MeCo introduces a new similarity testing method that retrieves previously solved tasks with high relevance, enabling effective plan reuse without re-invoking LLMs. Furthermore, we present MeCoBench, the first benchmark designed to evaluate performance on similar-task collaboration scenarios. Experimental results show that MeCo substantially reduces planning costs and improves success rates compared with state-of-the-art approaches.
>
---
#### [new 015] A Taylor Series Approach to Correct Localization Errors in Robotic Field Mapping using Gaussian Processes
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人环境建模任务，解决传感器定位误差影响高斯过程预测精度的问题。通过引入二阶修正算法，提升模型准确性与效率。**

- **链接: [https://arxiv.org/pdf/2601.20149v1](https://arxiv.org/pdf/2601.20149v1)**

> **作者:** Muzaffar Qureshi; Tochukwu Elijah Ogri; Kyle Volle; Rushikesh Kamalapurkar
>
> **摘要:** Gaussian Processes (GPs) are powerful non-parametric Bayesian models for regression of scalar fields, formulated under the assumption that measurement locations are perfectly known and the corresponding field measurements have Gaussian noise. However, many real-world scalar field mapping applications rely on sensor-equipped mobile robots to collect field measurements, where imperfect localization introduces state uncertainty. Such discrepancies between the estimated and true measurement locations degrade GP mean and covariance estimates. To address this challenge, we propose a method for updating the GP models when improved estimates become available. Leveraging the differentiability of the kernel function, a second-order correction algorithm is developed using the precomputed Jacobians and Hessians of the GP mean and covariance functions for real-time refinement based on measurement location discrepancy data. Simulation results demonstrate improved prediction accuracy and computational efficiency compared to full model retraining.
>
---
#### [new 016] TRACER: Texture-Robust Affordance Chain-of-Thought for Deformable-Object Refinement
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决变形物体在复杂纹理下的功能区域识别问题。提出TRACER框架，通过语义分解与边界优化提升操作精度与连续性。**

- **链接: [https://arxiv.org/pdf/2601.20208v1](https://arxiv.org/pdf/2601.20208v1)**

> **作者:** Wanjun Jia; Kang Li; Fan Yang; Mengfei Duan; Wenrui Chen; Yiming Jiang; Hui Zhang; Kailun Yang; Zhiyong Li; Yaonan Wang
>
> **备注:** The source code and dataset will be made publicly available at https://github.com/Dikay1/TRACER
>
> **摘要:** The central challenge in robotic manipulation of deformable objects lies in aligning high-level semantic instructions with physical interaction points under complex appearance and texture variations. Due to near-infinite degrees of freedom, complex dynamics, and heterogeneous patterns, existing vision-based affordance prediction methods often suffer from boundary overflow and fragmented functional regions. To address these issues, we propose TRACER, a Texture-Robust Affordance Chain-of-thought with dEformable-object Refinement framework, which establishes a cross-hierarchical mapping from hierarchical semantic reasoning to appearance-robust and physically consistent functional region refinement. Specifically, a Tree-structured Affordance Chain-of-Thought (TA-CoT) is formulated to decompose high-level task intentions into hierarchical sub-task semantics, providing consistent guidance across various execution stages. To ensure spatial integrity, a Spatial-Constrained Boundary Refinement (SCBR) mechanism is introduced to suppress prediction spillover, guiding the perceptual response to converge toward authentic interaction manifolds. Furthermore, an Interactive Convergence Refinement Flow (ICRF) is developed to aggregate discrete pixels corrupted by appearance noise, significantly enhancing the spatial continuity and physical plausibility of the identified functional regions. Extensive experiments conducted on the Fine-AGDDO15 dataset and a real-world robotic platform demonstrate that TRACER significantly improves affordance grounding precision across diverse textures and patterns inherent to deformable objects. More importantly, it enhances the success rate of long-horizon tasks, effectively bridging the gap between high-level semantic reasoning and low-level physical execution. The source code and dataset will be made publicly available at https://github.com/Dikay1/TRACER.
>
---
#### [new 017] GPO: Growing Policy Optimization for Legged Robot Locomotion and Whole-Body Control
- **分类: cs.RO**

- **简介: 该论文提出GPO框架，用于解决腿部机器人运动与全身控制中的强化学习训练难题。通过动态调整动作空间，提升数据收集效率和政策学习效果。**

- **链接: [https://arxiv.org/pdf/2601.20668v1](https://arxiv.org/pdf/2601.20668v1)**

> **作者:** Shuhao Liao; Peizhuo Li; Xinrong Yang; Linnan Chang; Zhaoxin Fan; Qing Wang; Lei Shi; Yuhong Cao; Wenjun Wu; Guillaume Sartoretti
>
> **摘要:** Training reinforcement learning (RL) policies for legged robots remains challenging due to high-dimensional continuous actions, hardware constraints, and limited exploration. Existing methods for locomotion and whole-body control work well for position-based control with environment-specific heuristics (e.g., reward shaping, curriculum design, and manual initialization), but are less effective for torque-based control, where sufficiently exploring the action space and obtaining informative gradient signals for training is significantly more difficult. We introduce Growing Policy Optimization (GPO), a training framework that applies a time-varying action transformation to restrict the effective action space in the early stage, thereby encouraging more effective data collection and policy learning, and then progressively expands it to enhance exploration and achieve higher expected return. We prove that this transformation preserves the PPO update rule and introduces only bounded, vanishing gradient distortion, thereby ensuring stable training. We evaluate GPO on both quadruped and hexapod robots, including zero-shot deployment of simulation-trained policies on hardware. Policies trained with GPO consistently achieve better performance. These results suggest that GPO provides a general, environment-agnostic optimization framework for learning legged locomotion.
>
---
#### [new 018] Demonstration-Free Robotic Control via LLM Agents
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决传统方法依赖演示和微调的问题。工作是引入FAEA框架，利用大语言模型进行自主决策，实现无需演示的机器人操作。**

- **链接: [https://arxiv.org/pdf/2601.20334v1](https://arxiv.org/pdf/2601.20334v1)**

> **作者:** Brian Y. Tsui; Alan Y. Fang; Tiffany J. Hwu
>
> **摘要:** Robotic manipulation has increasingly adopted vision-language-action (VLA) models, which achieve strong performance but typically require task-specific demonstrations and fine-tuning, and often generalize poorly under domain shift. We investigate whether general-purpose large language model (LLM) agent frameworks, originally developed for software engineering, can serve as an alternative control paradigm for embodied manipulation. We introduce FAEA (Frontier Agent as Embodied Agent), which applies an LLM agent framework directly to embodied manipulation without modification. Using the same iterative reasoning that enables software agents to debug code, FAEA enables embodied agents to reason through manipulation strategies. We evaluate an unmodified frontier agent, Claude Agent SDK, across the LIBERO, ManiSkill3, and MetaWorld benchmarks. With privileged environment state access, FAEA achieves success rates of 84.9%, 85.7%, and 96%, respectively. This level of task success approaches that of VLA models trained with less than 100 demonstrations per task, without requiring demonstrations or fine-tuning. With one round of human feedback as an optional optimization, performance increases to 88.2% on LIBERO. This demonstration-free capability has immediate practical value: FAEA can autonomously explore novel scenarios in simulation and generate successful trajectories for training data augmentation in embodied learning. Our results indicate that general-purpose agents are sufficient for a class of manipulation tasks dominated by deliberative, task-level planning. This opens a path for robotics systems to leverage actively maintained agent infrastructure and benefit directly from ongoing advances in frontier models. Code is available at https://github.com/robiemusketeer/faea-sim
>
---
#### [new 019] Just in time Informed Trees: Manipulability-Aware Asymptotically Optimized Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决高维复杂环境中高效且安全的运动规划问题。提出JIT*算法，通过动态优化提升路径质量和运动性能。**

- **链接: [https://arxiv.org/pdf/2601.19972v1](https://arxiv.org/pdf/2601.19972v1)**

> **作者:** Kuanqi Cai; Liding Zhang; Xinwen Su; Kejia Chen; Chaoqun Wang; Sami Haddadin; Alois Knoll; Arash Ajoudani; Luis Figueredo
>
> **摘要:** In high-dimensional robotic path planning, traditional sampling-based methods often struggle to efficiently identify both feasible and optimal paths in complex, multi-obstacle environments. This challenge is intensified in robotic manipulators, where the risk of kinematic singularities and self-collisions further complicates motion efficiency and safety. To address these issues, we introduce the Just-in-Time Informed Trees (JIT*) algorithm, an enhancement over Effort Informed Trees (EIT*), designed to improve path planning through two core modules: the Just-in-Time module and the Motion Performance module. The Just-in-Time module includes "Just-in-Time Edge," which dynamically refines edge connectivity, and "Just-in-Time Sample," which adjusts sampling density in bottleneck areas to enable faster initial path discovery. The Motion Performance module balances manipulability and trajectory cost through dynamic switching, optimizing motion control while reducing the risk of singularities. Comparative analysis shows that JIT* consistently outperforms traditional sampling-based planners across $\mathbb{R}^4$ to $\mathbb{R}^{16}$ dimensions. Its effectiveness is further demonstrated in single-arm and dual-arm manipulation tasks, with experimental results available in a video at https://youtu.be/nL1BMHpMR7c.
>
---
#### [new 020] A Practical Framework of Key Performance Indicators for Multi-Robot Lunar and Planetary Field Tests
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人月球探测任务，旨在解决不同实验间性能评估不一致的问题。通过构建关键绩效指标框架，实现对多机器人系统在效率、鲁棒性和精度方面的统一评估。**

- **链接: [https://arxiv.org/pdf/2601.20529v1](https://arxiv.org/pdf/2601.20529v1)**

> **作者:** Julia Richter; David Oberacker; Gabriela Ligeza; Valentin T. Bickel; Philip Arm; William Talbot; Marvin Grosse Besselmann; Florian Kehl; Tristan Schnell; Hendrik Kolvenbach; Rüdiger Dillmann; Arne Roennau; Marco Hutter
>
> **摘要:** Robotic prospecting for critical resources on the Moon, such as ilmenite, rare earth elements, and water ice, requires robust exploration methods given the diverse terrain and harsh environmental conditions. Although numerous analog field trials address these goals, comparing their results remains challenging because of differences in robot platforms and experimental setups. These missions typically assess performance using selected, scenario-specific engineering metrics that fail to establish a clear link between field performance and science-driven objectives. In this paper, we address this gap by deriving a structured framework of KPI from three realistic multi-robot lunar scenarios reflecting scientific objectives and operational constraints. Our framework emphasizes scenario-dependent priorities in efficiency, robustness, and precision, and is explicitly designed for practical applicability in field deployments. We validated the framework in a multi-robot field test and found it practical and easy to apply for efficiency- and robustness-related KPI, whereas precision-oriented KPI require reliable ground-truth data that is not always feasible to obtain in outdoor analog environments. Overall, we propose this framework as a common evaluation standard enabling consistent, goal-oriented comparison of multi-robot field trials and supporting systematic development of robotic systems for future planetary exploration.
>
---
#### [new 021] Li-ViP3D++: Query-Gated Deformable Camera-LiDAR Fusion for End-to-End Perception and Trajectory Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶中的感知与轨迹预测任务，解决模块化系统信息受限和误差放大问题。提出Li-ViP3D++框架，通过查询空间融合相机与LiDAR数据，提升检测与预测性能。**

- **链接: [https://arxiv.org/pdf/2601.20720v1](https://arxiv.org/pdf/2601.20720v1)**

> **作者:** Matej Halinkovic; Nina Masarykova; Alexey Vinel; Marek Galinski
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** End-to-end perception and trajectory prediction from raw sensor data is one of the key capabilities for autonomous driving. Modular pipelines restrict information flow and can amplify upstream errors. Recent query-based, fully differentiable perception-and-prediction (PnP) models mitigate these issues, yet the complementarity of cameras and LiDAR in the query-space has not been sufficiently explored. Models often rely on fusion schemes that introduce heuristic alignment and discrete selection steps which prevent full utilization of available information and can introduce unwanted bias. We propose Li-ViP3D++, a query-based multimodal PnP framework that introduces Query-Gated Deformable Fusion (QGDF) to integrate multi-view RGB and LiDAR in query space. QGDF (i) aggregates image evidence via masked attention across cameras and feature levels, (ii) extracts LiDAR context through fully differentiable BEV sampling with learned per-query offsets, and (iii) applies query-conditioned gating to adaptively weight visual and geometric cues per agent. The resulting architecture jointly optimizes detection, tracking, and multi-hypothesis trajectory forecasting in a single end-to-end model. On nuScenes, Li-ViP3D++ improves end-to-end behavior and detection quality, achieving higher EPA (0.335) and mAP (0.502) while substantially reducing false positives (FP ratio 0.147), and it is faster than the prior Li-ViP3D variant (139.82 ms vs. 145.91 ms). These results indicate that query-space, fully differentiable camera-LiDAR fusion can increase robustness of end-to-end PnP without sacrificing deployability.
>
---
#### [new 022] Game-Theoretic Autonomous Driving: A Graphs of Convex Sets Approach
- **分类: cs.MA; cs.RO**

- **简介: 该论文属于多车协同自动驾驶任务，解决共享安全约束下的策略交互与混合轨迹规划问题。提出IBR-GCS方法，结合博弈论与凸集图模型，实现安全、策略一致的驾驶行为。**

- **链接: [https://arxiv.org/pdf/2601.20054v1](https://arxiv.org/pdf/2601.20054v1)**

> **作者:** Nikolaj Käfer; Ahmed Khalil; Edward Huynh; Efstathios Bakolas; David Fridovich-Keil
>
> **备注:** 16 pages for content, 2 pages for references, 2 pages for notation
>
> **摘要:** Multi-vehicle autonomous driving couples strategic interaction with hybrid (discrete-continuous) maneuver planning under shared safety constraints. We introduce IBR-GCS, an Iterative Best Response (IBR) planning approach based on the Graphs of Convex Sets (GCS) framework that models highway driving as a generalized noncooperative game. IBR-GCS integrates combinatorial maneuver reasoning, trajectory planning, and game-theoretic interaction within a unified framework. The key novelty is a vehicle-specific, strategy-dependent GCS construction. Specifically, at each best-response update, each vehicle builds its own graph conditioned on the current strategies of the other vehicles, with vertices representing lane-specific, time-varying, convex, collision-free regions and edges encoding dynamically feasible transitions. This yields a shortest-path problem in GCS for each best-response step, which admits an efficient convex relaxation that can be solved using convex optimization tools without exhaustive discrete tree search. We then apply an iterative best-response scheme in which vehicles update their trajectories sequentially and provide conditions under which the resulting inexact updates converge to an approximate generalized Nash equilibrium. Simulation results across multi-lane, multi-vehicle scenarios demonstrate that IBR-GCS produces safe trajectories and strategically consistent interactive behaviors.
>
---
#### [new 023] MemCtrl: Using MLLMs as Active Memory Controllers on Embodied Agents
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决 embodied agents 的记忆管理问题。提出 MemCtrl 框架，利用 MLLMs 在线修剪记忆，提升任务完成能力。**

- **链接: [https://arxiv.org/pdf/2601.20831v1](https://arxiv.org/pdf/2601.20831v1)**

> **作者:** Vishnu Sashank Dorbala; Dinesh Manocha
>
> **摘要:** Foundation models rely on in-context learning for personalized decision making. The limited size of this context window necessitates memory compression and retrieval systems like RAG. These systems however often treat memory as large offline storage spaces, which is unfavorable for embodied agents that are expected to operate under strict memory and compute constraints, online. In this work, we propose MemCtrl, a novel framework that uses Multimodal Large Language Models (MLLMs) for pruning memory online. MemCtrl augments MLLMs with a trainable memory head μthat acts as a gate to determine which observations or reflections to retain, update, or discard during exploration. We evaluate with training two types of μ, 1) via an offline expert, and 2) via online RL, and observe significant improvement in overall embodied task completion ability on μ-augmented MLLMs. In particular, on augmenting two low performing MLLMs with MemCtrl on multiple subsets of the EmbodiedBench benchmark, we observe that μ-augmented MLLMs show an improvement of around 16% on average, with over 20% on specific instruction subsets. Finally, we present a qualitative analysis on the memory fragments collected by μ, noting the superior performance of μaugmented MLLMs on long and complex instruction types.
>
---
## 更新

#### [replaced 001] FLOL: Fast Baselines for Real-World Low-Light Enhancement
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于低光图像增强任务，旨在解决真实场景下图像增强的效率与鲁棒性问题。提出轻量级网络FLOL，在频域和空域结合处理，实现快速且高质量的增强效果。**

- **链接: [https://arxiv.org/pdf/2501.09718v2](https://arxiv.org/pdf/2501.09718v2)**

> **作者:** Juan C. Benito; Daniel Feijoo; Alvaro Garcia; Marcos V. Conde
>
> **备注:** Journal Preprint
>
> **摘要:** Low-Light Image Enhancement (LLIE) is a key task in computational photography and imaging. The problem of enhancing images captured during night or in dark environments has been well-studied in the computer vision literature. However, current deep learning-based solutions struggle with efficiency and robustness for real-world scenarios (e.g., scenes with noise, saturated pixels). We propose a lightweight neural network that combines image processing in the frequency and spatial domains. Our baseline method, FLOL, is one of the fastest models for this task, achieving results comparable to the state-of-the-art on popular real-world benchmarks such as LOLv2, LSRW, MIT-5K and UHD-LL. Moreover, we are able to process 1080p images in real-time under 12ms. Code and models at https://github.com/cidautai/FLOL
>
---
#### [replaced 002] Progressive-Resolution Policy Distillation: Leveraging Coarse-Resolution Simulations for Time-Efficient Fine-Resolution Policy Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自主挖掘任务，解决高分辨率模拟计算耗时与低分辨率模拟精度不足的问题。通过PRPD框架，利用低分辨率策略预训练高分辨率模型，提升效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2412.07477v4](https://arxiv.org/pdf/2412.07477v4)**

> **作者:** Yuki Kadokawa; Hirotaka Tahara; Takamitsu Matsubara
>
> **备注:** accepted for IEEE Transactions on Automation Science and Engineering (T-ASE)
>
> **摘要:** In earthwork and construction, excavators often encounter large rocks mixed with various soil conditions, requiring skilled operators. This paper presents a framework for achieving autonomous excavation using reinforcement learning (RL) through a rock excavation simulator. In the simulation, resolution can be defined by the particle size/number in the whole soil space. Fine-resolution simulations closely mimic real-world behavior but demand significant calculation time and challenging sample collection, while coarse-resolution simulations enable faster sample collection but deviate from real-world behavior. To combine the advantages of both resolutions, we explore using policies developed in coarse-resolution simulations for pre-training in fine-resolution simulations. To this end, we propose a novel policy learning framework called Progressive-Resolution Policy Distillation (PRPD), which progressively transfers policies through some middle-resolution simulations with conservative policy transfer to avoid domain gaps that could lead to policy transfer failure. Validation in a rock excavation simulator and nine real-world rock environments demonstrated that PRPD reduced sampling time to less than 1/7 while maintaining task success rates comparable to those achieved through policy learning in a fine-resolution simulation. Additional videos and supplementary results are available on our project page: https://yuki-kadokawa.github.io/prpd/
>
---
#### [replaced 003] MetaVLA: Unified Meta Co-training For Efficient Embodied Adaption
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出MetaVLA，解决VLA模型泛化能力差、训练成本高的问题，通过元学习实现高效多任务适应。**

- **链接: [https://arxiv.org/pdf/2510.05580v3](https://arxiv.org/pdf/2510.05580v3)**

> **作者:** Chen Li; Zhantao Yang; Han Zhang; Fangyi Chen; Chenchen Zhu; Anudeepsekhar Bolimera; Marios Savvides
>
> **摘要:** Vision-Language-Action (VLA) models show promise in embodied reasoning, yet remain far from true generalists-they often require task-specific fine-tuning, incur high compute costs, and generalize poorly to unseen tasks. We propose MetaVLA, a unified, backbone-agnostic post-training framework for efficient and scalable alignment. MetaVLA introduces Context-Aware Meta Co-Training, which consolidates diverse target tasks into a single fine-tuning stage while leveraging structurally diverse auxiliary tasks to improve in-domain generalization. Unlike naive multi-task SFT, MetaVLA integrates a lightweight meta-learning mechanism-derived from Attentive Neural Processes-to enable rapid adaptation from diverse contexts with minimal architectural change or inference overhead. On the LIBERO benchmark, MetaVLA with six auxiliary tasks outperforms OpenVLA by up to 8.0% on long-horizon tasks, reduces training steps from 240K to 75K, and cuts GPU time by ~76%. These results show that scalable, low-resource post-training is achievable-paving the way toward general-purpose embodied agents. Code will be available.
>
---
#### [replaced 004] Memory-Maze: Scenario Driven Visual Language Navigation Benchmark for Guiding Blind People
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决盲人导航中基于人类记忆的路线指令理解问题。工作包括构建Memory-Maze基准，包含记忆获取的指令和虚拟环境。**

- **链接: [https://arxiv.org/pdf/2405.07060v2](https://arxiv.org/pdf/2405.07060v2)**

> **作者:** Masaki Kuribayashi; Kohei Uehara; Allan Wang; Daisuke Sato; Simon Chu; Shigeo Morishima
>
> **摘要:** Visual Language Navigation (VLN) powered robots have the potential to guide blind people by understanding route instructions provided by sighted passersby. This capability allows robots to operate in environments often unknown a prior. Existing VLN models are insufficient for the scenario of navigation guidance for blind people, as they need to understand routes described from human memory, which frequently contains stutters, errors, and omissions of details, as opposed to those obtained by thinking out loud, such as in the R2R dataset. However, existing benchmarks do not contain instructions obtained from human memory in natural environments. To this end, we present our benchmark, Memory-Maze, which simulates the scenario of seeking route instructions for guiding blind people. Our benchmark contains a maze-like structured virtual environment and novel route instruction data from human memory. Our analysis demonstrates that instruction data collected from memory was longer and contained more varied wording. We further demonstrate that addressing errors and ambiguities from memory-based instructions is challenging, by evaluating state-of-the-art models alongside our baseline model with modularized perception and controls.
>
---
#### [replaced 005] Indoor Positioning Based on Active Radar Sensing and Passive Reflectors: Reflector Placement Optimization
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于室内定位任务，旨在解决AMR在复杂环境中的精确定位问题。通过优化被动雷达反射器的二维布局，结合FMCW雷达实现低成本高精度定位。**

- **链接: [https://arxiv.org/pdf/2509.15613v2](https://arxiv.org/pdf/2509.15613v2)**

> **作者:** Sven Hinderer; Pascal Schlachter; Zhibin Yu; Xiaofeng Wu; Bin Yang
>
> **摘要:** We extend our work on a novel indoor positioning system (IPS) for autonomous mobile robots (AMRs) based on radar sensing of local, passive radar reflectors. Through the combination of simple reflectors and a single-channel frequency modulated continuous wave (FMCW) radar, high positioning accuracy at low system cost can be achieved. Further, a multi-objective (MO) particle swarm optimization (PSO) algorithm is presented that optimizes the 2D placement of radar reflectors in complex room settings.
>
---
#### [replaced 006] Discrete Variational Autoencoding via Policy Search
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于生成模型任务，旨在解决离散变分自编码器的训练难题。通过引入策略搜索中的自然梯度方法，提升高维数据重建效果。**

- **链接: [https://arxiv.org/pdf/2509.24716v2](https://arxiv.org/pdf/2509.24716v2)**

> **作者:** Michael Drolet; Firas Al-Hafez; Aditya Bhatt; Jan Peters; Oleg Arenz
>
> **摘要:** Discrete latent bottlenecks in variational autoencoders (VAEs) offer high bit efficiency and can be modeled with autoregressive discrete distributions, enabling parameter-efficient multimodal search with transformers. However, discrete random variables do not allow for exact differentiable parameterization; therefore, discrete VAEs typically rely on approximations, such as Gumbel-Softmax reparameterization or straight-through gradient estimates, or employ high-variance gradient-free methods such as REINFORCE that have had limited success on high-dimensional tasks such as image reconstruction. Inspired by popular techniques in policy search, we propose a training framework for discrete VAEs that leverages the natural gradient of a non-parametric encoder to update the parametric encoder without requiring reparameterization. Our method, combined with automatic step size adaptation and a transformer-based encoder, scales to challenging datasets such as ImageNet and outperforms both approximate reparameterization methods and quantization-based discrete autoencoders in reconstructing high-dimensional data from compact latent spaces.
>
---
#### [replaced 007] Legged Robot State Estimation Using Invariant Neural-Augmented Kalman Filter with a Neural Compensator
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，旨在解决腿式机器人在模型非线性下的估计精度问题。通过结合卡尔曼滤波与神经网络，提出InNKF算法提升估计性能。**

- **链接: [https://arxiv.org/pdf/2503.00344v3](https://arxiv.org/pdf/2503.00344v3)**

> **作者:** Seokju Lee; Hyun-Bin Kim; Kyung-Soo Kim
>
> **备注:** 8 pages, 10 figures, Accepted to IROS 2025
>
> **摘要:** This paper presents an algorithm to improve state estimation for legged robots. Among existing model-based state estimation methods for legged robots, the contact-aided invariant extended Kalman filter defines the state on a Lie group to preserve invariance, thereby significantly accelerating convergence. It achieves more accurate state estimation by leveraging contact information as measurements for the update step. However, when the model exhibits strong nonlinearity, the estimation accuracy decreases. Such nonlinearities can cause initial errors to accumulate and lead to large drifts over time. To address this issue, we propose compensating for errors by augmenting the Kalman filter with an artificial neural network serving as a nonlinear function approximator. Furthermore, we design this neural network to respect the Lie group structure to ensure invariance, resulting in our proposed Invariant Neural-Augmented Kalman Filter (InNKF). The proposed algorithm offers improved state estimation performance by combining the strengths of model-based and learning-based approaches. Project webpage: https://seokju-lee.github.io/innkf_webpage
>
---
#### [replaced 008] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出OmniEVA，解决 embodied intelligence 中的3D适应性和物理约束问题，通过任务自适应3D接地和具身感知推理，提升任务规划能力。**

- **链接: [https://arxiv.org/pdf/2509.09332v3](https://arxiv.org/pdf/2509.09332v3)**

> **作者:** Yuecheng Liu; Dafeng Chi; Shiguang Wu; Zhanguang Zhang; Yuzheng Zhuang; Bowen Yang; He Zhu; Lingfeng Zhang; Pengwei Xie; David Gamaliel Arcos Bravo; Yingxue Zhang; Jianye Hao; Xingyue Quan
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically infeasible. To address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: https://omnieva.github.io
>
---
#### [replaced 009] Judgelight: Trajectory-Level Post-Optimization for Multi-Agent Path Finding via Closed-Subwalk Collapsing
- **分类: cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决轨迹优化问题。针对学习型算法生成的冗余路径，提出Judgelight方法，通过压缩闭合子路径提升轨迹质量。**

- **链接: [https://arxiv.org/pdf/2601.19388v2](https://arxiv.org/pdf/2601.19388v2)**

> **作者:** Yimin Tang; Sven Koenig; Erdem Bıyık
>
> **摘要:** Multi-Agent Path Finding (MAPF) is an NP-hard problem with applications in warehouse automation and multi-robot coordination. Learning-based MAPF solvers offer fast and scalable planning but often produce feasible trajectories that contain unnecessary or oscillatory movements. We propose Judgelight, a post-optimization layer that improves trajectory quality after a MAPF solver generates a feasible schedule. Judgelight collapses closed subwalks in agents' trajectories to remove redundant movements while preserving all feasibility constraints. We formalize this process as MAPF-Collapse, prove that it is NP-hard, and present an exact optimization approach by formulating it as integer linear programming (ILP) problem. Experimental results show Judgelight consistently reduces solution cost by around 20%, particularly for learning-based solvers, producing trajectories that are better suited for real-world deployment.
>
---
#### [replaced 010] Listen, Look, Drive: Coupling Audio Instructions for User-aware VLA-based Autonomous Driving
- **分类: eess.AS; cs.MM; cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决VLA模型无法实时接收用户意图的问题。通过融合音频指令与视觉信息，提出EchoVLA模型，提升驾驶决策的准确性和情感适应性。**

- **链接: [https://arxiv.org/pdf/2601.12142v2](https://arxiv.org/pdf/2601.12142v2)**

> **作者:** Ziang Guo; Feng Yang; Xuefeng Zhang; Jiaqi Guo; Kun Zhao; Yixiao Zhou; Peng Lu; Zufeng Zhang; Sifa Zheng
>
> **备注:** Accepted by IV
>
> **摘要:** Vision Language Action (VLA) models promise an open-vocabulary interface that can translate perceptual ambiguity into semantically grounded driving decisions, yet they still treat language as a static prior fixed at inference time. As a result, the model must infer continuously shifting objectives from pixels alone, yielding delayed or overly conservative maneuvers. We argue that effective VLAs for autonomous driving need an online channel in which users can influence driving with specific intentions. To this end, we present EchoVLA, a user-aware VLA that couples camera streams with in situ audio instructions. We augment the nuScenes dataset with temporally aligned, intent-specific speech commands generated by converting ego-motion descriptions into synthetic audios. Further, we compose emotional speech-trajectory pairs into a multimodal Chain-of-Thought (CoT) for fine-tuning a Multimodal Large Model (MLM) based on Qwen2.5-Omni. Specifically, we synthesize the audio-augmented dataset with different emotion types paired with corresponding driving behaviors, leveraging the emotional cues embedded in tone, pitch, and speech tempo to reflect varying user states, such as urgent or hesitant intentions, thus enabling our EchoVLA to interpret not only the semantic content but also the emotional context of audio commands for more nuanced and emotionally adaptive driving behavior. In open-loop benchmarks, our approach reduces the average L2 error by $59.4\%$ and the collision rate by $74.4\%$ compared to the baseline of vision-only perception. More experiments on nuScenes dataset validate that EchoVLA not only steers the trajectory through audio instructions, but also modulates driving behavior in response to the emotions detected in the user's speech.
>
---
#### [replaced 011] Fusion of Visual-Inertial Odometry with LiDAR Relative Localization for Cooperative Guidance of a Micro-Scale Aerial Vehicle
- **分类: cs.RO**

- **简介: 该论文属于无人机协同导航任务，旨在解决微小无人机定位精度低的问题。通过融合LiDAR与VIO数据，提升定位准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2306.17544v3](https://arxiv.org/pdf/2306.17544v3)**

> **作者:** Václav Pritzl; Matouš Vrba; Petr Štěpán; Martin Saska
>
> **备注:** Preprint version. This work has been submitted to the IEEE for possible publication
>
> **摘要:** A novel relative localization approach for guidance of a micro-scale Unmanned Aerial Vehicle (UAV) by a well-equipped aerial robot fusing Visual-Inertial Odometry (VIO) with Light Detection and Ranging (LiDAR) is proposed in this paper. LiDAR-based localization is accurate and robust to challenging environmental conditions, but 3D LiDARs are relatively heavy and require large UAV platforms, in contrast to lightweight cameras. However, visual-based self-localization methods exhibit lower accuracy and can suffer from significant drift with respect to the global reference frame. To benefit from both sensory modalities, we focus on cooperative navigation in a heterogeneous team of a primary LiDAR-equipped UAV and a secondary micro-scale camera-equipped UAV. We propose a novel cooperative approach combining LiDAR relative localization data with VIO output on board the primary UAV to obtain an accurate pose of the secondary UAV. The pose estimate is used to precisely and reliably guide the secondary UAV along trajectories defined in the primary UAV reference frame. The experimental evaluation has shown the superior accuracy of our method to the raw VIO output, reaching the average 3D Absolute Trajectory Error (ATE) of 0.28 m, and demonstrated its capability to guide the secondary UAV along desired trajectories while mitigating VIO drift. Thus, such a heterogeneous system can explore large areas with LiDAR precision, as well as visit locations inaccessible to the large LiDAR-carrying UAV platforms, as was showcased in a real-world cooperative mapping scenario.
>
---
#### [replaced 012] Embodied AI with Foundation Models for Mobile Service Robots: A Systematic Review
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于移动服务机器人领域，探讨如何将基础模型应用于具身AI，解决指令理解、多模态感知、不确定性处理和实时部署等问题。**

- **链接: [https://arxiv.org/pdf/2505.20503v2](https://arxiv.org/pdf/2505.20503v2)**

> **作者:** Matthew Lisondra; Beno Benhabib; Goldie Nejat
>
> **备注:** v2: Expanded systematic review; resubmitted to Robotics
>
> **摘要:** Rapid advancements in foundation models, including Large Language Models, Vision-Language Models, Multimodal Large Language Models, and Vision-Language-Action Models, have opened new avenues for embodied AI in mobile service robotics. By combining foundation models with the principles of embodied AI, where intelligent systems perceive, reason, and act through physical interaction, mobile service robots can achieve more flexible understanding, adaptive behavior, and robust task execution in dynamic real-world environments. Despite this progress, embodied AI for mobile service robots continues to face fundamental challenges related to the translation of natural language instructions into executable robot actions, multimodal perception in human-centered environments, uncertainty estimation for safe decision-making, and computational constraints for real-time onboard deployment. In this paper, we present the first systematic review focused specifically on the integration of foundation models in mobile service robotics. We analyze how recent advances in foundation models address these core challenges through language-conditioned control, multimodal sensor fusion, uncertainty-aware reasoning, and efficient model scaling. We further examine real-world applications in domestic assistance, healthcare, and service automation, highlighting how foundation models enable context-aware, socially responsive, and generalizable robot behaviors. Beyond technical considerations, we discuss ethical, societal, and human-interaction implications associated with deploying foundation model-enabled service robots in human environments. Finally, we outline future research directions emphasizing reliability and lifelong adaptation, privacy-aware and resource-constrained deployment, and governance and human-in-the-loop frameworks required for safe, scalable, and trustworthy mobile service robotics.
>
---
#### [replaced 013] Equitable Routing--Rethinking the Multiple Traveling Salesman Problem
- **分类: math.OC; cs.RO**

- **简介: 该论文属于路径优化任务，解决多旅行商问题中的公平性问题。提出两种新的公平约束模型，确保路线长度均衡，同时控制总成本，并开发了保证全局最优的算法。**

- **链接: [https://arxiv.org/pdf/2404.08157v5](https://arxiv.org/pdf/2404.08157v5)**

> **作者:** Abhay Singh Bhadoriya; Deepjyoti Deka; Kaarthik Sundar
>
> **备注:** 26 pages
>
> **摘要:** The Multiple Traveling Salesman Problem (MTSP) extends the traveling salesman problem by assigning multiple salesmen to visit a set of targets from a common depot, with each target visited exactly once while minimizing total tour length. A common variant, the min-max MTSP, focuses on workload balance by minimizing the longest tour, but it is difficult to solve optimally due to weak linear relaxation bounds. This paper introduces two new parametric fairness-driven variants of the MTSP: the $\varepsilon$-Fair-MTSP and the $Δ$-Fair-MTSP, which promote equitable distribution of tour lengths while controlling overall cost. The $\varepsilon$-Fair-MTSP is formulated as a mixed-integer second-order cone program, while the $Δ$-Fair-MTSP is modeled as a mixed-integer linear program. We develop algorithms that guarantee global optimality for both formulations. Computational experiments on benchmark instances and real-world applications, including electric vehicle fleet routing, demonstrate their effectiveness. Furthermore, we show that the algorithms presented for the fairness-constrained MTSP variants can be used to obtain the pareto-front of a bi-objective optimization problem where one objective focuses on minimizing the total tour length and the other focuses on balancing the tour lengths of the individual tours. Overall, these fairness-constrained MTSP variants provide a practical and flexible alternative to the min-max MTSP.
>
---
#### [replaced 014] SG-CADVLM: A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全验证任务，旨在解决真实高风险场景生成困难的问题。通过结合上下文解码与多模态输入，生成符合事故特征的仿真场景。**

- **链接: [https://arxiv.org/pdf/2601.18442v2](https://arxiv.org/pdf/2601.18442v2)**

> **作者:** Hongyi Zhao; Shuo Wang; Qijie He; Ziyuan Pu
>
> **摘要:** Autonomous vehicle safety validation requires testing on safety-critical scenarios, but these events are rare in real-world driving and costly to test due to collision risks. Crash reports provide authentic specifications of safety-critical events, offering a vital alternative to scarce real-world collision trajectory data. This makes them valuable sources for generating realistic high-risk scenarios through simulation. Existing approaches face significant limitations because data-driven methods lack diversity due to their reliance on existing latent distributions, whereas adversarial methods often produce unrealistic scenarios lacking physical fidelity. Large Language Model (LLM) and Vision Language Model (VLM)-based methods show significant promise. However, they suffer from context suppression issues where internal parametric knowledge overrides crash specifications, producing scenarios that deviate from actual accident characteristics. This paper presents SG-CADVLM (A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation), a framework that integrates Context-Aware Decoding with multi-modal input processing to generate safety-critical scenarios from crash reports and road network diagrams. The framework mitigates VLM hallucination issues while enabling the simultaneous generation of road geometry and vehicle trajectories. The experimental results demonstrate that SG-CADVLM generates critical risk scenarios at a rate of 84.4% compared to 12.5% for the baseline methods, representing an improvement of 469%, while producing executable simulations for autonomous vehicle testing.
>
---
