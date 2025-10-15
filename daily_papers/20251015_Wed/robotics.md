# 机器人 cs.RO

- **最新发布 32 篇**

- **更新 24 篇**

## 最新发布

#### [new 001] Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型缺乏3D空间感知的问题，提出Spatial Forcing方法，通过将VLA中间视觉特征与预训练3D模型的几何表征对齐，隐式增强空间理解能力，提升动作精度，无需依赖显式3D输入或深度估计。**

- **链接: [http://arxiv.org/pdf/2510.12276v1](http://arxiv.org/pdf/2510.12276v1)**

> **作者:** Fuhao Li; Wenxuan Song; Han Zhao; Jingbo Wang; Pengxiang Ding; Donglin Wang; Long Zeng; Haoang Li
>
> **摘要:** Vision-language-action (VLA) models have recently shown strong potential in enabling robots to follow language instructions and execute precise actions. However, most VLAs are built upon vision-language models pretrained solely on 2D data, which lack accurate spatial awareness and hinder their ability to operate in the 3D physical world. Existing solutions attempt to incorporate explicit 3D sensor inputs such as depth maps or point clouds, but these approaches face challenges due to sensor noise, hardware heterogeneity, and incomplete depth coverage in existing datasets. Alternative methods that estimate 3D cues from 2D images also suffer from the limited performance of depth estimators.We propose Spatial Forcing (SF), a simple yet effective alignment strategy that implicitly forces VLA models to develop spatial comprehension capabilities without relying on explicit 3D inputs or depth estimators. SF aligns intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. By enforcing alignment at intermediate layers, SF guides VLAs to encode richer spatial representations that enhance action precision.Extensive experiments in simulation and real-world environments demonstrate that SF achieves state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further accelerates training by up to 3.8x and improves data efficiency across diverse robotic tasks. Project page is at https://spatial-forcing.github.io/
>
---
#### [new 002] Gaussian Semantic Field for One-shot LiDAR Global Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究LiDAR全局定位任务，旨在解决语义地标重复导致的误匹配问题。作者提出高斯语义场模型，用连续函数表达语义分布，构建轻量三层场景图，提升单次定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.12101v1](http://arxiv.org/pdf/2510.12101v1)**

> **作者:** Pengyu Yin; Shenghai Yuan; Haozhi Cao; Xingyu Ji; Ruofei Bai; Siyu Chen; Lihua Xie
>
> **摘要:** We present a one-shot LiDAR global localization algorithm featuring semantic disambiguation ability based on a lightweight tri-layered scene graph. While landmark semantic registration-based methods have shown promising performance improvements in global localization compared with geometric-only methods, landmarks can be repetitive and misleading for correspondence establishment. We propose to mitigate this problem by modeling semantic distributions with continuous functions learned from a population of Gaussian processes. Compared with discrete semantic labels, the continuous functions capture finer-grained geo-semantic information and also provide more detailed metric information for correspondence establishment. We insert this continuous function as the middle layer between the object layer and the metric-semantic layer, forming a tri-layered 3D scene graph, serving as a light-weight yet performant backend for one-shot localization. We term our global localization pipeline Outram-GSF (Gaussian semantic field) and conduct a wide range of experiments on publicly available data sets, validating the superior performance against the current state-of-the-art.
>
---
#### [new 003] Learning Social Navigation from Positive and Negative Demonstrations and Rule-Based Specifications
- **分类: cs.RO**

- **简介: 该论文研究机器人在人群中的导航任务，旨在平衡适应性与安全性。提出融合正负示例学习与规则约束的框架，通过师生策略学习实现高效、安全的社交导航。**

- **链接: [http://arxiv.org/pdf/2510.12215v1](http://arxiv.org/pdf/2510.12215v1)**

> **作者:** Chanwoo Kim; Jihwan Yoon; Hyeonseong Kim; Taemoon Jeong; Changwoo Yoo; Seungbeen Lee; Soohwan Byeon; Hoon Chung; Matthew Pan; Jean Oh; Kyungjae Lee; Sungjoon Choi
>
> **备注:** For more videos, see https://chanwookim971024.github.io/PioneeR/
>
> **摘要:** Mobile robot navigation in dynamic human environments requires policies that balance adaptability to diverse behaviors with compliance to safety constraints. We hypothesize that integrating data-driven rewards with rule-based objectives enables navigation policies to achieve a more effective balance of adaptability and safety. To this end, we develop a framework that learns a density-based reward from positive and negative demonstrations and augments it with rule-based objectives for obstacle avoidance and goal reaching. A sampling-based lookahead controller produces supervisory actions that are both safe and adaptive, which are subsequently distilled into a compact student policy suitable for real-time operation with uncertainty estimates. Experiments in synthetic and elevator co-boarding simulations show consistent gains in success rate and time efficiency over baselines, and real-world demonstrations with human participants confirm the practicality of deployment. A video illustrating this work can be found on our project page https://chanwookim971024.github.io/PioneeR/.
>
---
#### [new 004] Reflection-Based Task Adaptation for Self-Improving VLA
- **分类: cs.RO**

- **简介: 该论文研究视觉-语言-动作（VLA）模型在新任务中的快速自适应问题。提出“反思式自适应”框架，通过失败驱动的强化学习与成功驱动的监督微调双路径，实现无人干预的自主策略优化，提升任务成功率与收敛速度。**

- **链接: [http://arxiv.org/pdf/2510.12710v1](http://arxiv.org/pdf/2510.12710v1)**

> **作者:** Baicheng Li; Dong Wu; Zike Yan; Xinchen Liu; Zecui Zeng; Lusong Li; Hongbin Zha
>
> **摘要:** Pre-trained Vision-Language-Action (VLA) models represent a major leap towards general-purpose robots, yet efficiently adapting them to novel, specific tasks in-situ remains a significant hurdle. While reinforcement learning (RL) is a promising avenue for such adaptation, the process often suffers from low efficiency, hindering rapid task mastery. We introduce Reflective Self-Adaptation, a framework for rapid, autonomous task adaptation without human intervention. Our framework establishes a self-improving loop where the agent learns from its own experience to enhance both strategy and execution. The core of our framework is a dual-pathway architecture that addresses the full adaptation lifecycle. First, a Failure-Driven Reflective RL pathway enables rapid learning by using the VLM's causal reasoning to automatically synthesize a targeted, dense reward function from failure analysis. This provides a focused learning signal that significantly accelerates policy exploration. However, optimizing such proxy rewards introduces a potential risk of "reward hacking," where the agent masters the reward function but fails the actual task. To counteract this, our second pathway, Success-Driven Quality-Guided SFT, grounds the policy in holistic success. It identifies and selectively imitates high-quality successful trajectories, ensuring the agent remains aligned with the ultimate task goal. This pathway is strengthened by a conditional curriculum mechanism to aid initial exploration. We conduct experiments in challenging manipulation tasks. The results demonstrate that our framework achieves faster convergence and higher final success rates compared to representative baselines. Our work presents a robust solution for creating self-improving agents that can efficiently and reliably adapt to new environments.
>
---
#### [new 005] Two-stream network-driven vision-based tactile sensor for object feature extraction and fusion perception
- **分类: cs.RO; physics.app-ph**

- **简介: 该论文针对视觉触觉传感器信息冗余与特征融合不足的问题，提出双流网络策略，分别提取物体内外特征（深度与硬度），通过CNN提取并加权融合，提升识别精度，实现高效物体感知。**

- **链接: [http://arxiv.org/pdf/2510.12528v1](http://arxiv.org/pdf/2510.12528v1)**

> **作者:** Muxing Huang; Zibin Chen; Weiliang Xu; Zilan Li; Yuanzhi Zhou; Guoyuan Zhou; Wenjing Chen; Xinming Li
>
> **摘要:** Tactile perception is crucial for embodied intelligent robots to recognize objects. Vision-based tactile sensors extract object physical attributes multidimensionally using high spatial resolution; however, this process generates abundant redundant information. Furthermore, single-dimensional extraction, lacking effective fusion, fails to fully characterize object attributes. These challenges hinder the improvement of recognition accuracy. To address this issue, this study introduces a two-stream network feature extraction and fusion perception strategy for vision-based tactile systems. This strategy employs a distributed approach to extract internal and external object features. It obtains depth map information through three-dimensional reconstruction while simultaneously acquiring hardness information by measuring contact force data. After extracting features with a convolutional neural network (CNN), weighted fusion is applied to create a more informative and effective feature representation. In standard tests on objects of varying shapes and hardness, the force prediction error is 0.06 N (within a 12 N range). Hardness recognition accuracy reaches 98.0%, and shape recognition accuracy reaches 93.75%. With fusion algorithms, object recognition accuracy in actual grasping scenarios exceeds 98.5%. Focused on object physical attributes perception, this method enhances the artificial tactile system ability to transition from perception to cognition, enabling its use in embodied perception applications.
>
---
#### [new 006] Improving Generative Behavior Cloning via Self-Guidance and Adaptive Chunking
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究机器人学习中的生成式行为克隆（GBC），旨在解决扩散策略在开环控制下动作采样错误和响应延迟的问题。作者提出自引导和自适应分块两项技术，提升动作准确性和环境响应能力，在多任务操作中显著改善性能。**

- **链接: [http://arxiv.org/pdf/2510.12392v1](http://arxiv.org/pdf/2510.12392v1)**

> **作者:** Junhyuk So; Chiwoong Lee; Shinyoung Lee; Jungseul Ok; Eunhyeok Park
>
> **备注:** Accepted at NeurIPS25
>
> **摘要:** Generative Behavior Cloning (GBC) is a simple yet effective framework for robot learning, particularly in multi-task settings. Recent GBC methods often employ diffusion policies with open-loop (OL) control, where actions are generated via a diffusion process and executed in multi-step chunks without replanning. While this approach has demonstrated strong success rates and generalization, its inherent stochasticity can result in erroneous action sampling, occasionally leading to unexpected task failures. Moreover, OL control suffers from delayed responses, which can degrade performance in noisy or dynamic environments. To address these limitations, we propose two novel techniques to enhance the consistency and reactivity of diffusion policies: (1) self-guidance, which improves action fidelity by leveraging past observations and implicitly promoting future-aware behavior; and (2) adaptive chunking, which selectively updates action sequences when the benefits of reactivity outweigh the need for temporal consistency. Extensive experiments show that our approach substantially improves GBC performance across a wide range of simulated and real-world robotic manipulation tasks. Our code is available at https://github.com/junhyukso/SGAC
>
---
#### [new 007] PolygMap: A Perceptive Locomotion Framework for Humanoid Robot Stair Climbing
- **分类: cs.RO**

- **简介: 该论文针对双足机器人在未知环境中上下楼梯的感知与运动规划问题，提出PolygMap框架。通过多传感器融合构建实时多边形语义地图，实现精准足步规划，部署于NVIDIA Orin实现实时高效运动控制。**

- **链接: [http://arxiv.org/pdf/2510.12346v1](http://arxiv.org/pdf/2510.12346v1)**

> **作者:** Bingquan Li; Ning Wang; Tianwei Zhang; Zhicheng He; Yucong Wu
>
> **摘要:** Recently, biped robot walking technology has been significantly developed, mainly in the context of a bland walking scheme. To emulate human walking, robots need to step on the positions they see in unknown spaces accurately. In this paper, we present PolyMap, a perception-based locomotion planning framework for humanoid robots to climb stairs. Our core idea is to build a real-time polygonal staircase plane semantic map, followed by a footstep planar using these polygonal plane segments. These plane segmentation and visual odometry are done by multi-sensor fusion(LiDAR, RGB-D camera and IMUs). The proposed framework is deployed on a NVIDIA Orin, which performs 20-30 Hz whole-body motion planning output. Both indoor and outdoor real-scene experiments indicate that our method is efficient and robust for humanoid robot stair climbing.
>
---
#### [new 008] Maximal Adaptation, Minimal Guidance: Permissive Reactive Robot Task Planning with Humans in the Loop
- **分类: cs.RO**

- **简介: 该论文研究人机协同任务规划，解决机器人在未知人类行为下持续满足时序逻辑任务的问题。提出“最大适应、最小引导”框架，实现在线策略调整与必要时请求协作，兼顾任务保证与人类自主性。**

- **链接: [http://arxiv.org/pdf/2510.12662v1](http://arxiv.org/pdf/2510.12662v1)**

> **作者:** Oz Gitelson; Satya Prakash Nayak; Ritam Raha; Anne-Kathrin Schmuck
>
> **摘要:** We present a novel framework for human-robot \emph{logical} interaction that enables robots to reliably satisfy (infinite horizon) temporal logic tasks while effectively collaborating with humans who pursue independent and unknown tasks. The framework combines two key capabilities: (i) \emph{maximal adaptation} enables the robot to adjust its strategy \emph{online} to exploit human behavior for cooperation whenever possible, and (ii) \emph{minimal tunable feedback} enables the robot to request cooperation by the human online only when necessary to guarantee progress. This balance minimizes human-robot interference, preserves human autonomy, and ensures persistent robot task satisfaction even under conflicting human goals. We validate the approach in a real-world block-manipulation task with a Franka Emika Panda robotic arm and in the Overcooked-AI benchmark, demonstrating that our method produces rich, \emph{emergent} cooperative behaviors beyond the reach of existing approaches, while maintaining strong formal guarantees.
>
---
#### [new 009] Shape-Aware Whole-Body Control for Continuum Robots with Application in Endoluminal Surgical Robotics
- **分类: cs.RO**

- **简介: 该论文研究连续体机器人在内窥手术导航中的全机身控制任务，旨在解决传统末端控制易导致管壁碰撞、损伤组织等问题。提出一种结合物理模型与残差学习的形状感知控制框架，实现精确形状估计与实时优化控制，提升导航安全性与适应性。**

- **链接: [http://arxiv.org/pdf/2510.12332v1](http://arxiv.org/pdf/2510.12332v1)**

> **作者:** Mohammadreza Kasaei; Mostafa Ghobadi; Mohsen Khadem
>
> **摘要:** This paper presents a shape-aware whole-body control framework for tendon-driven continuum robots with direct application to endoluminal surgical navigation. Endoluminal procedures, such as bronchoscopy, demand precise and safe navigation through tortuous, patient-specific anatomy where conventional tip-only control often leads to wall contact, tissue trauma, or failure to reach distal targets. To address these challenges, our approach combines a physics-informed backbone model with residual learning through an Augmented Neural ODE, enabling accurate shape estimation and efficient Jacobian computation. A sampling-based Model Predictive Path Integral (MPPI) controller leverages this representation to jointly optimize tip tracking, backbone conformance, and obstacle avoidance under actuation constraints. A task manager further enhances adaptability by allowing real-time adjustment of objectives, such as wall clearance or direct advancement, during tele-operation. Extensive simulation studies demonstrate millimeter-level accuracy across diverse scenarios, including trajectory tracking, dynamic obstacle avoidance, and shape-constrained reaching. Real-robot experiments on a bronchoscopy phantom validate the framework, showing improved lumen-following accuracy, reduced wall contacts, and enhanced adaptability compared to joystick-only navigation and existing baselines. These results highlight the potential of the proposed framework to increase safety, reliability, and operator efficiency in minimally invasive endoluminal surgery, with broader applicability to other confined and safety-critical environments.
>
---
#### [new 010] Controlling Intent Expressiveness in Robot Motion with Diffusion Models
- **分类: cs.RO**

- **简介: 该论文研究机器人运动的意图可读性控制问题，旨在生成从清晰到模糊不同表达程度的运动轨迹。提出基于信息势场和扩散模型的两阶段框架，实现可控可读性且保持性能。**

- **链接: [http://arxiv.org/pdf/2510.12370v1](http://arxiv.org/pdf/2510.12370v1)**

> **作者:** Wenli Shi; Clemence Grislain; Olivier Sigaud; Mohamed Chetouani
>
> **备注:** Using diffusion models trained on quality diversity datasets for generating robot motions with adjustable legibility levels
>
> **摘要:** Legibility of robot motion is critical in human-robot interaction, as it allows humans to quickly infer a robot's intended goal. Although traditional trajectory generation methods typically prioritize efficiency, they often fail to make the robot's intentions clear to humans. Meanwhile, existing approaches to legible motion usually produce only a single "most legible" trajectory, overlooking the need to modulate intent expressiveness in different contexts. In this work, we propose a novel motion generation framework that enables controllable legibility across the full spectrum, from highly legible to highly ambiguous motions. We introduce a modeling approach based on an Information Potential Field to assign continuous legibility scores to trajectories, and build upon it with a two-stage diffusion framework that first generates paths at specified legibility levels and then translates them into executable robot actions. Experiments in both 2D and 3D reaching tasks demonstrate that our approach produces diverse and controllable motions with varying degrees of legibility, while achieving performance comparable to SOTA. Code and project page: https://legibility-modulator.github.io.
>
---
#### [new 011] Robot Learning: A Tutorial
- **分类: cs.RO; cs.LG**

- **简介: 该论文是一篇机器人学习综述教程，旨在介绍从强化学习、行为克隆到通用语言条件模型的发展。属于教育指导任务，解决初学者入门难题，提供基础理论与实践工具，并通过lerobot库实现示例。**

- **链接: [http://arxiv.org/pdf/2510.12403v1](http://arxiv.org/pdf/2510.12403v1)**

> **作者:** Francesco Capuano; Caroline Pascal; Adil Zouitine; Thomas Wolf; Michel Aractingi
>
> **备注:** Tutorial on Robot Learning using LeRobot, the end-to-end robot learning library developed by Hugging Face
>
> **摘要:** Robot learning is at an inflection point, driven by rapid advancements in machine learning and the growing availability of large-scale robotics data. This shift from classical, model-based methods to data-driven, learning-based paradigms is unlocking unprecedented capabilities in autonomous systems. This tutorial navigates the landscape of modern robot learning, charting a course from the foundational principles of Reinforcement Learning and Behavioral Cloning to generalist, language-conditioned models capable of operating across diverse tasks and even robot embodiments. This work is intended as a guide for researchers and practitioners, and our goal is to equip the reader with the conceptual understanding and practical tools necessary to contribute to developments in robot learning, with ready-to-use examples implemented in $\texttt{lerobot}$.
>
---
#### [new 012] T(R,O) Grasp: Efficient Graph Diffusion of Robot-Object Spatial Transformation for Cross-Embodiment Dexterous Grasping
- **分类: cs.RO**

- **简介: 该论文研究灵巧抓取任务，解决跨机器人手型的高效、通用抓取生成问题。提出T(R,O) Grasp框架，通过建模机器人-物体空间变换的图扩散方法，实现快速、准确、多样化的抓取合成，支持闭环操作且泛化性强。**

- **链接: [http://arxiv.org/pdf/2510.12724v1](http://arxiv.org/pdf/2510.12724v1)**

> **作者:** Xin Fei; Zhixuan Xu; Huaicong Fang; Tianrui Zhang; Lin Shao
>
> **备注:** 12 pages, 14 figures
>
> **摘要:** Dexterous grasping remains a central challenge in robotics due to the complexity of its high-dimensional state and action space. We introduce T(R,O) Grasp, a diffusion-based framework that efficiently generates accurate and diverse grasps across multiple robotic hands. At its core is the T(R,O) Graph, a unified representation that models spatial transformations between robotic hands and objects while encoding their geometric properties. A graph diffusion model, coupled with an efficient inverse kinematics solver, supports both unconditioned and conditioned grasp synthesis. Extensive experiments on a diverse set of dexterous hands show that T(R,O) Grasp achieves average success rate of 94.83%, inference speed of 0.21s, and throughput of 41 grasps per second on an NVIDIA A100 40GB GPU, substantially outperforming existing baselines. In addition, our approach is robust and generalizable across embodiments while significantly reducing memory consumption. More importantly, the high inference speed enables closed-loop dexterous manipulation, underscoring the potential of T(R,O) Grasp to scale into a foundation model for dexterous grasping.
>
---
#### [new 013] Residual MPC: Blending Reinforcement Learning with GPU-Parallelized Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文提出残差MPC架构，将强化学习与GPU并行模型预测控制结合，解决机器人运动控制中鲁棒性、实时性与样本效率问题。通过在扭矩层面融合两者输出，实现高频率、可解释且适应性强的控制，提升性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.12717v1](http://arxiv.org/pdf/2510.12717v1)**

> **作者:** Se Hwan Jeon; Ho Jae Lee; Seungwoo Hong; Sangbae Kim
>
> **备注:** TRO submission preprint
>
> **摘要:** Model Predictive Control (MPC) provides interpretable, tunable locomotion controllers grounded in physical models, but its robustness depends on frequent replanning and is limited by model mismatch and real-time computational constraints. Reinforcement Learning (RL), by contrast, can produce highly robust behaviors through stochastic training but often lacks interpretability, suffers from out-of-distribution failures, and requires intensive reward engineering. This work presents a GPU-parallelized residual architecture that tightly integrates MPC and RL by blending their outputs at the torque-control level. We develop a kinodynamic whole-body MPC formulation evaluated across thousands of agents in parallel at 100 Hz for RL training. The residual policy learns to make targeted corrections to the MPC outputs, combining the interpretability and constraint handling of model-based control with the adaptability of RL. The model-based control prior acts as a strong bias, initializing and guiding the policy towards desirable behavior with a simple set of rewards. Compared to standalone MPC or end-to-end RL, our approach achieves higher sample efficiency, converges to greater asymptotic rewards, expands the range of trackable velocity commands, and enables zero-shot adaptation to unseen gaits and uneven terrain.
>
---
#### [new 014] Automated Behavior Planning for Fruit Tree Pruning via Redundant Robot Manipulators: Addressing the Behavior Planning Challenge
- **分类: cs.RO**

- **简介: 该论文研究果树修剪中冗余机械臂的行为规划问题，旨在解决复杂枝条环境下的多层级运动规划。提出融合感知、建模与整体规划的完整流程，并在真实机器人上验证，提升了自动化修剪的性能。**

- **链接: [http://arxiv.org/pdf/2510.12509v1](http://arxiv.org/pdf/2510.12509v1)**

> **作者:** Gaoyuan Liu; Bas Boom; Naftali Slob; Yuri Durodié; Ann Nowé; Bram Vanderborght
>
> **摘要:** Pruning is an essential agricultural practice for orchards. Proper pruning can promote healthier growth and optimize fruit production throughout the orchard's lifespan. Robot manipulators have been developed as an automated solution for this repetitive task, which typically requires seasonal labor with specialized skills. While previous research has primarily focused on the challenges of perception, the complexities of manipulation are often overlooked. These challenges involve planning and control in both joint and Cartesian spaces to guide the end-effector through intricate, obstructive branches. Our work addresses the behavior planning challenge for a robotic pruning system, which entails a multi-level planning problem in environments with complex collisions. In this paper, we formulate the planning problem for a high-dimensional robotic arm in a pruning scenario, investigate the system's intrinsic redundancies, and propose a comprehensive pruning workflow that integrates perception, modeling, and holistic planning. In our experiments, we demonstrate that more comprehensive planning methods can significantly enhance the performance of the robotic manipulator. Finally, we implement the proposed workflow on a real-world robot. As a result, this work complements previous efforts on robotic pruning and motivates future research and development in planning for pruning applications.
>
---
#### [new 015] Achieving Meaningful Collaboration: Worker-centered Design of a Physical Human-Robot Collaborative Blending Task
- **分类: cs.RO**

- **简介: 该论文聚焦工人中心的协作机器人设计，旨在提升工业场景中人机协同作业的效率与工人福祉。针对飞机发动机维修任务，采用跨学科方法融合学术研究与工人实践经验，解决劳动力短缺与生产需求增长问题。**

- **链接: [http://arxiv.org/pdf/2510.12340v1](http://arxiv.org/pdf/2510.12340v1)**

> **作者:** Nicky Mol; Luka Peternel; Alessandro Ianniello; Denis Zatyagov; Auke Nachenius; Stephan Balvert; J. Micah Prendergast; Sara Muscolo; Olger Siebinga; Eva Verhoef; Deborah Forster; David A. Abbink
>
> **备注:** 3 pages, 1 figure, ICRA@40 (Extended abstract)
>
> **摘要:** The use of robots in industrial settings continues to grow, driven by the need to address complex societal challenges such as labor shortages, aging populations, and ever-increasing production demands. In this abstract, we advocate for (and demonstrate) a transdisciplinary approach when considering robotics in the workplace. Transdisciplinarity emphasizes the integration of academic research with pragmatic expertise and embodied experiential knowledge, that prioritize values such as worker wellbeing and job attractiveness. In the following, we describe an ongoing multi-pronged effort to explore the potential of collaborative robots in the context of airplane engine repair and maintenance operations.
>
---
#### [new 016] A Task-Efficient Reinforcement Learning Task-Motion Planner for Safe Human-Robot Cooperation
- **分类: cs.RO**

- **简介: 该论文研究人机协作中的任务与运动规划问题，旨在平衡安全性与效率。提出一种结合强化学习任务规划与交互式运动规划的混合框架，通过学习安全任务序列并实时响应人类动作，减少重规划次数和任务失败，提升协作性能。**

- **链接: [http://arxiv.org/pdf/2510.12477v1](http://arxiv.org/pdf/2510.12477v1)**

> **作者:** Gaoyuan Liu; Joris de Winter; Kelly Merckaert; Denis Steckelmacher; Ann Nowe; Bram Vanderborght
>
> **摘要:** In a Human-Robot Cooperation (HRC) environment, safety and efficiency are the two core properties to evaluate robot performance. However, safety mechanisms usually hinder task efficiency since human intervention will cause backup motions and goal failures of the robot. Frequent motion replanning will increase the computational load and the chance of failure. In this paper, we present a hybrid Reinforcement Learning (RL) planning framework which is comprised of an interactive motion planner and a RL task planner. The RL task planner attempts to choose statistically safe and efficient task sequences based on the feedback from the motion planner, while the motion planner keeps the task execution process collision-free by detecting human arm motions and deploying new paths when the previous path is not valid anymore. Intuitively, the RL agent will learn to avoid dangerous tasks, while the motion planner ensures that the chosen tasks are safe. The proposed framework is validated on the cobot in both simulation and the real world, we compare the planner with hard-coded task motion planning methods. The results show that our planning framework can 1) react to uncertain human motions at both joint and task levels; 2) reduce the times of repeating failed goal commands; 3) reduce the total number of replanning requests.
>
---
#### [new 017] Fast Visuomotor Policy for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对高频率、资源受限的机器人操作任务，提出一种名为Energy Policy的快速视觉运动策略框架。通过能量得分学习目标和高效MLP架构，实现单次前向传播的多模态动作预测，在降低计算开销的同时保持高性能。**

- **链接: [http://arxiv.org/pdf/2510.12483v1](http://arxiv.org/pdf/2510.12483v1)**

> **作者:** Jingkai Jia; Tong Yang; Xueyao Chen; Chenhuan Liu; Wenqiang Zhang
>
> **摘要:** We present a fast and effective policy framework for robotic manipulation, named Energy Policy, designed for high-frequency robotic tasks and resource-constrained systems. Unlike existing robotic policies, Energy Policy natively predicts multimodal actions in a single forward pass, enabling high-precision manipulation at high speed. The framework is built upon two core components. First, we adopt the energy score as the learning objective to facilitate multimodal action modeling. Second, we introduce an energy MLP to implement the proposed objective while keeping the architecture simple and efficient. We conduct comprehensive experiments in both simulated environments and real-world robotic tasks to evaluate the effectiveness of Energy Policy. The results show that Energy Policy matches or surpasses the performance of state-of-the-art manipulation methods while significantly reducing computational overhead. Notably, on the MimicGen benchmark, Energy Policy achieves superior performance with at a faster inference compared to existing approaches.
>
---
#### [new 018] Designing Tools with Control Confidence
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人自主设计手持工具的任务，旨在解决现有方法仅优化性能而忽略使用信心的问题。作者提出一种包含神经启发式控制信心项的优化框架，提升工具在环境不确定性下的鲁棒性，并通过CMAES算法实现高效设计。**

- **链接: [http://arxiv.org/pdf/2510.12630v1](http://arxiv.org/pdf/2510.12630v1)**

> **作者:** Ajith Anil Meera; Abian Torres; Pablo Lanillos
>
> **摘要:** Prehistoric humans invented stone tools for specialized tasks by not just maximizing the tool's immediate goal-completion accuracy, but also increasing their confidence in the tool for later use under similar settings. This factor contributed to the increased robustness of the tool, i.e., the least performance deviations under environmental uncertainties. However, the current autonomous tool design frameworks solely rely on performance optimization, without considering the agent's confidence in tool use for repeated use. Here, we take a step towards filling this gap by i) defining an optimization framework for task-conditioned autonomous hand tool design for robots, where ii) we introduce a neuro-inspired control confidence term into the optimization routine that helps the agent to design tools with higher robustness. Through rigorous simulations using a robotic arm, we show that tools designed with control confidence as the objective function are more robust to environmental uncertainties during tool use than a pure accuracy-driven objective. We further show that adding control confidence to the objective function for tool design provides a balance between the robustness and goal accuracy of the designed tools under control perturbations. Finally, we show that our CMAES-based evolutionary optimization strategy for autonomous tool design outperforms other state-of-the-art optimizers by designing the optimal tool within the fewest iterations. Code: https://github.com/ajitham123/Tool_design_control_confidence.
>
---
#### [new 019] Pretraining in Actor-Critic Reinforcement Learning for Robot Motion Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究机器人运动控制中的强化学习预训练方法，旨在解决策略从零学习效率低的问题。提出通过无任务探索数据预训练逆动力学模型，并将权重用于Actor-Critic算法的冷启动，显著提升样本效率与性能。**

- **链接: [http://arxiv.org/pdf/2510.12363v1](http://arxiv.org/pdf/2510.12363v1)**

> **作者:** Jiale Fan; Andrei Cramariuc; Tifanny Portela; Marco Hutter
>
> **备注:** Submitted to ICLR 2026
>
> **摘要:** The pretraining-finetuning paradigm has facilitated numerous transformative advancements in artificial intelligence research in recent years. However, in the domain of reinforcement learning (RL) for robot motion control, individual skills are often learned from scratch despite the high likelihood that some generalizable knowledge is shared across all task-specific policies belonging to a single robot embodiment. This work aims to define a paradigm for pretraining neural network models that encapsulate such knowledge and can subsequently serve as a basis for warm-starting the RL process in classic actor-critic algorithms, such as Proximal Policy Optimization (PPO). We begin with a task-agnostic exploration-based data collection algorithm to gather diverse, dynamic transition data, which is then used to train a Proprioceptive Inverse Dynamics Model (PIDM) through supervised learning. The pretrained weights are loaded into both the actor and critic networks to warm-start the policy optimization of actual tasks. We systematically validated our proposed method on seven distinct robot motion control tasks, showing significant benefits to this initialization strategy. Our proposed approach on average improves sample efficiency by 40.1% and task performance by 7.5%, compared to random initialization. We further present key ablation studies and empirical analyses that shed light on the mechanisms behind the effectiveness of our method.
>
---
#### [new 020] M3D-skin: Multi-material 3D-printed Tactile Sensor with Hierarchical Infill Structures for Pressure Sensing
- **分类: cs.RO**

- **简介: 该论文提出一种基于多材料3D打印和层级填充结构的触觉传感器M3D-skin，利用柔性导电与非导电材料的变形实现压力感知。旨在简化传感器制造并提升集成性，验证了其在足底运动监测、机器人手集成等应用中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.12419v1](http://arxiv.org/pdf/2510.12419v1)**

> **作者:** Shunnosuke Yoshimura; Kento Kawaharazuka; Kei Okada
>
> **备注:** Accepted to IROS2025, Website: https://ssk-yoshimura.github.io/M3D-skin/
>
> **摘要:** Tactile sensors have a wide range of applications, from utilization in robotic grippers to human motion measurement. If tactile sensors could be fabricated and integrated more easily, their applicability would further expand. In this study, we propose a tactile sensor-M3D-skin-that can be easily fabricated with high versatility by leveraging the infill patterns of a multi-material fused deposition modeling (FDM) 3D printer as the sensing principle. This method employs conductive and non-conductive flexible filaments to create a hierarchical structure with a specific infill pattern. The flexible hierarchical structure deforms under pressure, leading to a change in electrical resistance, enabling the acquisition of tactile information. We measure the changes in characteristics of the proposed tactile sensor caused by modifications to the hierarchical structure. Additionally, we demonstrate the fabrication and use of a multi-tile sensor. Furthermore, as applications, we implement motion pattern measurement on the sole of a foot, integration with a robotic hand, and tactile-based robotic operations. Through these experiments, we validate the effectiveness of the proposed tactile sensor.
>
---
#### [new 021] HYPE: Hybrid Planning with Ego Proposal-Conditioned Predictions
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究自动驾驶运动规划，旨在解决复杂城市环境中多智能体交互下的安全与适应性问题。提出HYPE方法，结合学习的轨迹先验与蒙特卡洛树搜索，通过自车提议条件下的环境预测建模双向交互，简化成本函数设计，提升规划性能。**

- **链接: [http://arxiv.org/pdf/2510.12733v1](http://arxiv.org/pdf/2510.12733v1)**

> **作者:** Hang Yu; Julian Jordan; Julian Schmidt; Silvan Lindner; Alessandro Canevaro; Wilhelm Stork
>
> **摘要:** Safe and interpretable motion planning in complex urban environments needs to reason about bidirectional multi-agent interactions. This reasoning requires to estimate the costs of potential ego driving maneuvers. Many existing planners generate initial trajectories with sampling-based methods and refine them by optimizing on learned predictions of future environment states, which requires a cost function that encodes the desired vehicle behavior. Designing such a cost function can be very challenging, especially if a wide range of complex urban scenarios has to be considered. We propose HYPE: HYbrid Planning with Ego proposal-conditioned predictions, a planner that integrates multimodal trajectory proposals from a learned proposal model as heuristic priors into a Monte Carlo Tree Search (MCTS) refinement. To model bidirectional interactions, we introduce an ego-conditioned occupancy prediction model, enabling consistent, scene-aware reasoning. Our design significantly simplifies cost function design in refinement by considering proposal-driven guidance, requiring only minimalistic grid-based cost terms. Evaluations on large-scale real-world benchmarks nuPlan and DeepUrban show that HYPE effectively achieves state-of-the-art performance, especially in safety and adaptability.
>
---
#### [new 022] Autonomous Legged Mobile Manipulation for Lunar Surface Operations via Constrained Reinforcement Learning
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究月面环境下四足移动机械臂的自主操作任务，针对非结构化地形与安全约束问题，提出一种基于约束强化学习的框架，实现整机运动与操作的协同控制，满足稳定性、避障与能耗等要求，并验证了高精度位姿跟踪能力。**

- **链接: [http://arxiv.org/pdf/2510.12684v1](http://arxiv.org/pdf/2510.12684v1)**

> **作者:** Alvaro Belmonte-Baeza; Miguel Cazorla; Gabriel J. García; Carlos J. Pérez-Del-Pulgar; Jorge Pomares
>
> **备注:** This is the authors version of the paper accepted for publication in The IEEE International Conference on Space Robotics 2025. The final version link will be added here after conference proceedings are published
>
> **摘要:** Robotics plays a pivotal role in planetary science and exploration, where autonomous and reliable systems are crucial due to the risks and challenges inherent to space environments. The establishment of permanent lunar bases demands robotic platforms capable of navigating and manipulating in the harsh lunar terrain. While wheeled rovers have been the mainstay for planetary exploration, their limitations in unstructured and steep terrains motivate the adoption of legged robots, which offer superior mobility and adaptability. This paper introduces a constrained reinforcement learning framework designed for autonomous quadrupedal mobile manipulators operating in lunar environments. The proposed framework integrates whole-body locomotion and manipulation capabilities while explicitly addressing critical safety constraints, including collision avoidance, dynamic stability, and power efficiency, in order to ensure robust performance under lunar-specific conditions, such as reduced gravity and irregular terrain. Experimental results demonstrate the framework's effectiveness in achieving precise 6D task-space end-effector pose tracking, achieving an average positional accuracy of 4 cm and orientation accuracy of 8.1 degrees. The system consistently respects both soft and hard constraints, exhibiting adaptive behaviors optimized for lunar gravity conditions. This work effectively bridges adaptive learning with essential mission-critical safety requirements, paving the way for advanced autonomous robotic explorers for future lunar missions.
>
---
#### [new 023] Learning Robust Agile Flight Control with Stability Guarantees
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究高动态环境下四旋翼飞行器的鲁棒敏捷飞行控制。针对执行器约束、扰动鲁棒性与计算效率难题，提出一种具稳定性保证的神经增强反馈控制器，实现复杂轨迹的精确跟踪，兼具高效学习与直接实机部署能力。**

- **链接: [http://arxiv.org/pdf/2510.12611v1](http://arxiv.org/pdf/2510.12611v1)**

> **作者:** Lukas Pries; Markus Ryll
>
> **摘要:** In the evolving landscape of high-speed agile quadrotor flight, achieving precise trajectory tracking at the platform's operational limits is paramount. Controllers must handle actuator constraints, exhibit robustness to disturbances, and remain computationally efficient for safety-critical applications. In this work, we present a novel neural-augmented feedback controller for agile flight control. The controller addresses individual limitations of existing state-of-the-art control paradigms and unifies their strengths. We demonstrate the controller's capabilities, including the accurate tracking of highly aggressive trajectories that surpass the feasibility of the actuators. Notably, the controller provides universal stability guarantees, enhancing its robustness and tracking performance even in exceedingly disturbance-prone settings. Its nonlinear feedback structure is highly efficient enabling fast computation at high update rates. Moreover, the learning process in simulation is both fast and stable, and the controller's inherent robustness allows direct deployment to real-world platforms without the need for training augmentations or fine-tuning.
>
---
#### [new 024] Translating Milli/Microrobots with A Value-Centered Readiness Framework
- **分类: cs.RO; cs.ET**

- **简介: 该论文综述了毫米/微米机器人向临床转化的挑战，提出以价值为中心的技术就绪度框架（mTRL），旨在对齐医疗需求与技术开发，推动机器人从实验室走向临床应用。**

- **链接: [http://arxiv.org/pdf/2510.12090v1](http://arxiv.org/pdf/2510.12090v1)**

> **作者:** Hakan Ceylan; Edoardo Sinibaldi; Sanjay Misra; Pankaj J. Pasricha; Dietmar W. Hutmacher
>
> **摘要:** Untethered mobile milli/microrobots hold transformative potential for interventional medicine by enabling more precise and entirely non-invasive diagnosis and therapy. Realizing this promise requires bridging the gap between groundbreaking laboratory demonstrations and successful clinical integration. Despite remarkable technical progress over the past two decades, most millirobots and microrobots remain confined to laboratory proof-of-concept demonstrations, with limited real-world feasibility. In this Review, we identify key factors that slow translation from bench to bedside, focusing on the disconnect between technical innovation and real-world application. We argue that the long-term impact and sustainability of the field depend on aligning development with unmet medical needs, ensuring applied feasibility, and integrating seamlessly into existing clinical workflows, which are essential pillars for delivering meaningful patient outcomes. To support this shift, we introduce a strategic milli/microrobot Technology Readiness Level framework (mTRL), which maps system development from initial conceptualization to clinical adoption through clearly defined milestones and their associated stepwise activities. The mTRL model provides a structured gauge of technological maturity, a common language for cross-disciplinary collaboration and actionable guidance to accelerate translational development toward new, safer and more efficient interventions.
>
---
#### [new 025] Hybrid Terrain-Aware Path Planning: Integrating VD--RRT\(^{*}\) Exploration and VD--D\(^{*}\) Lite Repair
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究自主车辆在非结构化地形中的实时路径规划，解决复杂地形下曲率与土壤承载力约束的路径生成问题。提出融合VD-RRT*全局探索与VD-D* Lite局部修复的混合方法，并构建解析地形代价模型，实现高效、稳定的毫米级重规划。**

- **链接: [http://arxiv.org/pdf/2510.12169v1](http://arxiv.org/pdf/2510.12169v1)**

> **作者:** Akshay Naik; William R. Norris; Dustin Nottage; Ahmet Soylemezoglu
>
> **摘要:** Autonomous ground vehicles operating off-road must plan curvature-feasible paths while accounting for spatially varying soil strength and slope hazards in real time. We present a continuous state--cost metric that combines a Bekker pressure--sinkage model with elevation-derived slope and attitude penalties. The resulting terrain cost field is analytic, bounded, and monotonic in soil modulus and slope, ensuring well-posed discretization and stable updates under sensor noise. This metric is evaluated on a lattice with exact steering primitives: Dubins and Reeds--Shepp motions for differential drive and time-parameterized bicycle arcs for Ackermann steering. Global exploration is performed using Vehicle-Dynamics RRT\(^{*}\), while local repair is managed by Vehicle-Dynamics D\(^{*}\) Lite, enabling millisecond-scale replanning without heuristic smoothing. By separating the terrain--vehicle model from the planner, the framework provides a reusable basis for deterministic, sampling-based, or learning-driven planning in deformable terrain. Hardware trials on an off-road platform demonstrate real-time navigation across soft soil and slope transitions, supporting reliable autonomy in unstructured environments.
>
---
#### [new 026] Controllable Collision Scenario Generation via Collision Pattern Prediction
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出可控碰撞场景生成任务，旨在生成指定碰撞类型和碰撞时间的自动驾驶测试场景。作者构建了COLLIDE数据集，并提出基于碰撞模式预测的框架，实现高可控性与碰撞率，用于发现规划器缺陷并提升其鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.12206v1](http://arxiv.org/pdf/2510.12206v1)**

> **作者:** Pin-Lun Chen; Chi-Hsi Kung; Che-Han Chang; Wei-Chen Chiu; Yi-Ting Chen
>
> **备注:** 8 pages, 3 figures. Submitted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Evaluating the safety of autonomous vehicles (AVs) requires diverse, safety-critical scenarios, with collisions being especially important yet rare and unsafe to collect in the real world. Therefore, the community has been focusing on generating safety-critical scenarios in simulation. However, controlling attributes such as collision type and time-to-accident (TTA) remains challenging. We introduce a new task called controllable collision scenario generation, where the goal is to produce trajectories that realize a user-specified collision type and TTA, to investigate the feasibility of automatically generating desired collision scenarios. To support this task, we present COLLIDE, a large-scale collision scenario dataset constructed by transforming real-world driving logs into diverse collisions, balanced across five representative collision types and different TTA intervals. We propose a framework that predicts Collision Pattern, a compact and interpretable representation that captures the spatial configuration of the ego and the adversarial vehicles at impact, before rolling out full adversarial trajectories. Experiments show that our approach outperforms strong baselines in both collision rate and controllability. Furthermore, generated scenarios consistently induce higher planner failure rates, revealing limitations of existing planners. We demonstrate that these scenarios fine-tune planners for robustness improvements, contributing to safer AV deployment in different collision scenarios.
>
---
#### [new 027] CoIRL-AD: Collaborative-Competitive Imitation-Reinforcement Learning in Latent World Models for Autonomous Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究自动驾驶决策控制任务，旨在解决纯模仿学习泛化性差、强化学习样本效率低的问题。提出CoIRL-AD框架，通过隐空间中双策略竞争机制，实现模仿与强化学习的协同训练，提升安全性与长尾场景表现。**

- **链接: [http://arxiv.org/pdf/2510.12560v1](http://arxiv.org/pdf/2510.12560v1)**

> **作者:** Xiaoji Zheng; Ziyuan Yang; Yanhao Chen; Yuhang Peng; Yuanrong Tang; Gengyuan Liu; Bokui Chen; Jiangtao Gong
>
> **备注:** 18 pages, 17 figures
>
> **摘要:** End-to-end autonomous driving models trained solely with imitation learning (IL) often suffer from poor generalization. In contrast, reinforcement learning (RL) promotes exploration through reward maximization but faces challenges such as sample inefficiency and unstable convergence. A natural solution is to combine IL and RL. Moving beyond the conventional two-stage paradigm (IL pretraining followed by RL fine-tuning), we propose CoIRL-AD, a competitive dual-policy framework that enables IL and RL agents to interact during training. CoIRL-AD introduces a competition-based mechanism that facilitates knowledge exchange while preventing gradient conflicts. Experiments on the nuScenes dataset show an 18% reduction in collision rate compared to baselines, along with stronger generalization and improved performance on long-tail scenarios. Code is available at: https://github.com/SEU-zxj/CoIRL-AD.
>
---
#### [new 028] A Unidirectionally Connected FAS Approach for 6-DOF Quadrotor Control
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文针对6-DOF四旋翼控制中欠驱动问题，提出单向连接全驱动系统（UC-FAS）方法，统一现有FAS变换方式，避免高阶导数估计，简化控制器设计，实现精确轨迹跟踪，弥合理论与实际应用差距。**

- **链接: [http://arxiv.org/pdf/2510.12360v1](http://arxiv.org/pdf/2510.12360v1)**

> **作者:** Weijie Ren; Haowen Liu; Guang-Ren Duan
>
> **备注:** This paper has been submitted to 2026 IFAC World Congress. Corresponding author: Guang-Ren Duan
>
> **摘要:** This paper proposes a unidirectionally connected fully actuated system (UC-FAS) approach for the sub-stabilization and tracking control of 6-DOF quadrotors, tackling limitations both in state-space and FAS framework to some extent. The framework systematically converts underactuated quadrotor dynamics into a UC-FAS model, unifying the existing different FAS transformation ways. By eliminating estimation of the high-order derivatives of control inputs, a drawback of current methods, the UC-FAS model simplifies controller design and enables direct eigenstructure assignment for closed-loop dynamics. Simulations demonstrate precise 6-DOF tracking performance. This work bridges theoretical FAS approach advancements with practical implementation needs, offering a standardized paradigm for nonlinear quadrotor control.
>
---
#### [new 029] EReLiFM: Evidential Reliability-Aware Residual Flow Meta-Learning for Open-Set Domain Generalization under Noisy Labels
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究开放集域泛化下的噪声标签问题，提出EReLiFM方法。通过证据性损失聚类提升标签可靠性，结合残差流匹配建模域与类别差异，实现更鲁棒的知识迁移。**

- **链接: [http://arxiv.org/pdf/2510.12687v1](http://arxiv.org/pdf/2510.12687v1)**

> **作者:** Kunyu Peng; Di Wen; Kailun Yang; Jia Fu; Yufan Chen; Ruiping Liu; Jiamin Wu; Junwei Zheng; M. Saquib Sarfraz; Luc Van Gool; Danda Pani Paudel; Rainer Stiefelhagen
>
> **备注:** The source code is available at https://github.com/KPeng9510/ERELIFM
>
> **摘要:** Open-Set Domain Generalization (OSDG) aims to enable deep learning models to recognize unseen categories in new domains, which is crucial for real-world applications. Label noise hinders open-set domain generalization by corrupting source-domain knowledge, making it harder to recognize known classes and reject unseen ones. While existing methods address OSDG under Noisy Labels (OSDG-NL) using hyperbolic prototype-guided meta-learning, they struggle to bridge domain gaps, especially with limited clean labeled data. In this paper, we propose Evidential Reliability-Aware Residual Flow Meta-Learning (EReLiFM). We first introduce an unsupervised two-stage evidential loss clustering method to promote label reliability awareness. Then, we propose a residual flow matching mechanism that models structured domain- and category-conditioned residuals, enabling diverse and uncertainty-aware transfer paths beyond interpolation-based augmentation. During this meta-learning process, the model is optimized such that the update direction on the clean set maximizes the loss decrease on the noisy set, using pseudo labels derived from the most confident predicted class for supervision. Experimental results show that EReLiFM outperforms existing methods on OSDG-NL, achieving state-of-the-art performance. The source code is available at https://github.com/KPeng9510/ERELIFM.
>
---
#### [new 030] EmboMatrix: A Scalable Training-Ground for Embodied Decision-Making
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出EmboMatrix，一种面向具身决策的可扩展训练平台，旨在解决大语言模型缺乏物理环境交互经验的问题。通过任务生成、高效仿真与精准奖励机制，支持模型在模拟环境中学习具身决策能力。**

- **链接: [http://arxiv.org/pdf/2510.12072v1](http://arxiv.org/pdf/2510.12072v1)**

> **作者:** Zixing Lei; Sheng Yin; Yichen Xiong; Yuanzhuo Ding; Wenhao Huang; Yuxi Wei; Qingyao Xu; Yiming Li; Weixin Li; Yunhong Wang; Siheng Chen
>
> **备注:** 10 pages 8 figures
>
> **摘要:** Embodied decision-making enables agents to translate high-level goals into executable actions through continuous interactions within the physical world, forming a cornerstone of general-purpose embodied intelligence. Large language models (LLMs), with their general decision-making capabilities, offer a promising path to realize this potential; however, LLMs trained solely on language lack exposure to physical environments, limiting their true embodied understanding. To bridge this gap, we propose the concept of a training ground: a comprehensive infrastructure that provides task and scene simulation, embodied interaction, and feedback signals, offering a one-stop solution for LLM acquire genuine embodied decision-making skills. In this work, we present EmboMatrix, the first training ground of its kind, providing massive and diverse tasks with efficient simulation and precise rewards. EmboMatrix incorporates a series of novel techniques: a multi-agent data engine for large-scale task and scene generation, a distributed heterogeneous-hardware system for scalable simulation, and a multi-level reward architecture for precise supervision. Leveraging EmboMatrix, we cultivate EmboBrain, an LLM whose embodied decision-making abilities emerge from extensive embodied interactions. Experiments show that EmboBrain-7B surpasses the 671B DeepSeek-R1 baseline by 9.5\% on two challenging embodied decision-making benchmarks, demonstrating the power of interactive, environment-grounded learning for building truly intelligent embodied agents.
>
---
#### [new 031] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning
- **分类: physics.med-ph; cs.AI; cs.RO**

- **简介: 该论文提出一种零样本大语言模型代理，用于全自动放射治疗计划。旨在解决放疗计划依赖人工、耗时且不一致的问题。通过与临床系统交互，自主优化IMRT计划，无需训练或微调，生成质量优于或媲美临床计划，实现可推广的自动化放疗规划。**

- **链接: [http://arxiv.org/pdf/2510.11754v1](http://arxiv.org/pdf/2510.11754v1)**

> **作者:** Dongrong Yang; Xin Wu; Yibo Xie; Xinyi Li; Qiuwen Wu; Jackie Wu; Yang Sheng
>
> **备注:** Accepted for poster presentation at the NeurIPS 2025 Workshop on GenAI for Health: Potential, Trust, and Policy Compliance
>
> **摘要:** Radiation therapy treatment planning is an iterative, expertise-dependent process, and the growing burden of cancer cases has made reliance on manual planning increasingly unsustainable, underscoring the need for automation. In this study, we propose a workflow that leverages a large language model (LLM)-based agent to navigate inverse treatment planning for intensity-modulated radiation therapy (IMRT). The LLM agent was implemented to directly interact with a clinical treatment planning system (TPS) to iteratively extract intermediate plan states and propose new constraint values to guide inverse optimization. The agent's decision-making process is informed by current observations and previous optimization attempts and evaluations, allowing for dynamic strategy refinement. The planning process was performed in a zero-shot inference setting, where the LLM operated without prior exposure to manually generated treatment plans and was utilized without any fine-tuning or task-specific training. The LLM-generated plans were evaluated on twenty head-and-neck cancer cases against clinical manual plans, with key dosimetric endpoints analyzed and reported. The LLM-generated plans achieved comparable organ-at-risk (OAR) sparing relative to clinical plans while demonstrating improved hot spot control (Dmax: 106.5% vs. 108.8%) and superior conformity (conformity index: 1.18 vs. 1.39 for boost PTV; 1.82 vs. 1.88 for primary PTV). This study demonstrates the feasibility of a zero-shot, LLM-driven workflow for automated IMRT treatment planning in a commercial TPS. The proposed approach provides a generalizable and clinically applicable solution that could reduce planning variability and support broader adoption of AI-based planning strategies.
>
---
#### [new 032] UniGS: Unified Geometry-Aware Gaussian Splatting for Multimodal Rendering
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出UniGS，面向多模态渲染的统一三维重建任务，解决高保真RGB、深度、法线与语义联合重建问题。通过几何感知的高斯点阵化框架，实现解析梯度优化与可微剪枝，提升精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.12174v1](http://arxiv.org/pdf/2510.12174v1)**

> **作者:** Yusen Xie; Zhenmin Huang; Jianhao Jiao; Dimitrios Kanoulas; Jun Ma
>
> **摘要:** In this paper, we propose UniGS, a unified map representation and differentiable framework for high-fidelity multimodal 3D reconstruction based on 3D Gaussian Splatting. Our framework integrates a CUDA-accelerated rasterization pipeline capable of rendering photo-realistic RGB images, geometrically accurate depth maps, consistent surface normals, and semantic logits simultaneously. We redesign the rasterization to render depth via differentiable ray-ellipsoid intersection rather than using Gaussian centers, enabling effective optimization of rotation and scale attribute through analytic depth gradients. Furthermore, we derive the analytic gradient formulation for surface normal rendering, ensuring geometric consistency among reconstructed 3D scenes. To improve computational and storage efficiency, we introduce a learnable attribute that enables differentiable pruning of Gaussians with minimal contribution during training. Quantitative and qualitative experiments demonstrate state-of-the-art reconstruction accuracy across all modalities, validating the efficacy of our geometry-aware paradigm. Source code and multimodal viewer will be available on GitHub.
>
---
## 更新

#### [replaced 001] REACT3D: Recovering Articulations for Interactive Physical 3D Scenes
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.11340v2](http://arxiv.org/pdf/2510.11340v2)**

> **作者:** Zhao Huang; Boyang Sun; Alexandros Delitzas; Jiaqi Chen; Marc Pollefeys
>
> **备注:** 8 pages
>
> **摘要:** Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks. Our contributions include: (i) openable-object detection and segmentation to extract candidate movable parts from static scenes, (ii) articulation estimation that infers joint types and motion parameters, (iii) hidden-geometry completion followed by interactive object assembly, and (iv) interactive scene integration in widely supported formats to ensure compatibility with standard simulation platforms. We achieve state-of-the-art performance on detection/segmentation and articulation metrics across diverse indoor scenes, demonstrating the effectiveness of our framework and providing a practical foundation for scalable interactive scene generation, thereby lowering the barrier to large-scale research on articulated scene understanding. Our project page is https://react3d.github.io/
>
---
#### [replaced 002] DSM: Constructing a Diverse Semantic Map for 3D Visual Grounding
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.08307v2](http://arxiv.org/pdf/2504.08307v2)**

> **作者:** Qinghongbing Xie; Zijian Liang; Fuhao Li; Long Zeng
>
> **备注:** 8 pages, 6 figures, Project Page: https://binicey.github.io/DSM
>
> **摘要:** Effective scene representation is critical for the visual grounding ability of representations, yet existing methods for 3D Visual Grounding are often constrained. They either only focus on geometric and visual cues, or, like traditional 3D scene graphs, lack the multi-dimensional attributes needed for complex reasoning. To bridge this gap, we introduce the Diverse Semantic Map (DSM) framework, a novel scene representation framework that enriches robust geometric models with a spectrum of VLM-derived semantics, including appearance, physical properties, and affordances. The DSM is first constructed online by fusing multi-view observations within a temporal sliding window, creating a persistent and comprehensive world model. Building on this foundation, we propose DSM-Grounding, a new paradigm that shifts grounding from free-form VLM queries to a structured reasoning process over the semantic-rich map, markedly improving accuracy and interpretability. Extensive evaluations validate our approach's superiority. On the ScanRefer benchmark, DSM-Grounding achieves a state-of-the-art 59.06% overall accuracy of IoU@0.5, surpassing others by 10%. In semantic segmentation, our DSM attains a 67.93% F-mIoU, outperforming all baselines, including privileged ones. Furthermore, successful deployment on physical robots for complex navigation and grasping tasks confirms the framework's practical utility in real-world scenarios.
>
---
#### [replaced 003] RoVer: Robot Reward Model as Test-Time Verifier for Vision-Language-Action Model
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.10975v2](http://arxiv.org/pdf/2510.10975v2)**

> **作者:** Mingtong Dai; Lingbo Liu; Yongjie Bai; Yang Liu; Zhouxia Wang; Rui SU; Chunjie Chen; Liang Lin; Xinyu Wu
>
> **摘要:** Vision-Language-Action (VLA) models have become a prominent paradigm for embodied intelligence, yet further performance improvements typically rely on scaling up training data and model size -- an approach that is prohibitively expensive for robotics and fundamentally limited by data collection costs. We address this limitation with $\mathbf{RoVer}$, an embodied test-time scaling framework that uses a $\mathbf{Ro}$bot Process Reward Model (PRM) as a Test-Time $\mathbf{Ver}$ifier to enhance the capabilities of existing VLA models without modifying their architectures or weights. Specifically, RoVer (i) assigns scalar-based process rewards to evaluate the reliability of candidate actions, and (ii) predicts an action-space direction for candidate expansion/refinement. During inference, RoVer generates multiple candidate actions concurrently from the base policy, expands them along PRM-predicted directions, and then scores all candidates with PRM to select the optimal action for execution. Notably, by caching shared perception features, it can amortize perception cost and evaluate more candidates under the same test-time computational budget. Essentially, our approach effectively transforms available computing resources into better action decision-making, realizing the benefits of test-time scaling without extra training overhead. Our contributions are threefold: (1) a general, plug-and-play test-time scaling framework for VLAs; (2) a PRM that jointly provides scalar process rewards and an action-space direction to guide exploration; and (3) an efficient direction-guided sampling strategy that leverages a shared perception cache to enable scalable candidate generation and selection during inference.
>
---
#### [replaced 004] Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11886v4](http://arxiv.org/pdf/2505.11886v4)**

> **作者:** Shuo Wang; Yongcai Wang; Wanting Li; Xudong Cai; Yucheng Wang; Maiyue Chen; Kaihui Wang; Zhizhong Su; Deying Li; Zhaoxin Fan
>
> **摘要:** Vision-Language Navigation (VLN) is a critical task for developing embodied agents that can follow natural language instructions to navigate in complex real-world environments. Recent advances in VLN by large pretrained models have significantly improved generalization and instruction grounding compared to traditional approaches. However, the role of reasoning strategies in navigation-an action-centric, long-horizon task-remains underexplored, despite Chain-of-Thought (CoT) reasoning's demonstrated success in static tasks like visual question answering. To address this gap, we conduct the first systematic evaluation of reasoning strategies for VLN, including No-Think (direct action prediction), Pre-Think (reason before action), and Post-Think (reason after action). Surprisingly, our findings reveal the Inference-time Reasoning Collapse issue, where inference-time reasoning degrades navigation accuracy, highlighting the challenges of integrating reasoning into VLN. Based on this insight, we propose Aux-Think, a framework that trains models to internalize structured reasoning patterns through CoT supervision, while inferring action directly without reasoning in online prediction. To support this framework, we release R2R-CoT-320k, the first Chain-of-Thought annotated dataset for VLN. Extensive experiments show that Aux-Think reduces training effort greatly and achieves the best performance under the same data scale.
>
---
#### [replaced 005] SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18648v3](http://arxiv.org/pdf/2509.18648v3)**

> **作者:** Yarden As; Chengrui Qu; Benjamin Unger; Dongho Kang; Max van der Hart; Laixi Shi; Stelian Coros; Adam Wierman; Andreas Krause
>
> **摘要:** Deploying reinforcement learning (RL) safely in the real world is challenging, as policies trained in simulators must face the inevitable sim-to-real gap. Robust safe RL techniques are provably safe, however difficult to scale, while domain randomization is more practical yet prone to unsafe behaviors. We address this gap by proposing SPiDR, short for Sim-to-real via Pessimistic Domain Randomization -- a scalable algorithm with provable guarantees for safe sim-to-real transfer. SPiDR uses domain randomization to incorporate the uncertainty about the sim-to-real gap into the safety constraints, making it versatile and highly compatible with existing training pipelines. Through extensive experiments on sim-to-sim benchmarks and two distinct real-world robotic platforms, we demonstrate that SPiDR effectively ensures safety despite the sim-to-real gap while maintaining strong performance.
>
---
#### [replaced 006] Image Quality Assessment for Embodied AI
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.16815v2](http://arxiv.org/pdf/2505.16815v2)**

> **作者:** Chunyi Li; Jiaohao Xiao; Jianbo Zhang; Farong Wen; Zicheng Zhang; Yuan Tian; Xiangyang Zhu; Xiaohong Liu; Zhengxue Cheng; Weisi Lin; Guangtao Zhai
>
> **摘要:** Embodied AI has developed rapidly in recent years, but it is still mainly deployed in laboratories, with various distortions in the Real-world limiting its application. Traditionally, Image Quality Assessment (IQA) methods are applied to predict human preferences for distorted images; however, there is no IQA method to assess the usability of an image in embodied tasks, namely, the perceptual quality for robots. To provide accurate and reliable quality indicators for future embodied scenarios, we first propose the topic: IQA for Embodied AI. Specifically, we (1) based on the Mertonian system and meta-cognitive theory, constructed a perception-cognition-decision-execution pipeline and defined a comprehensive subjective score collection process; (2) established the Embodied-IQA database, containing over 36k reference/distorted image pairs, with more than 5m fine-grained annotations provided by Vision Language Models/Vision Language Action-models/Real-world robots; (3) trained and validated the performance of mainstream IQA methods on Embodied-IQA, demonstrating the need to develop more accurate quality indicators for Embodied AI. We sincerely hope that through evaluation, we can promote the application of Embodied AI under complex distortions in the Real-world. Project page: https://github.com/lcysyzxdxc/EmbodiedIQA
>
---
#### [replaced 007] Visual Affordance Prediction: Survey and Reproducibility
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.05074v2](http://arxiv.org/pdf/2505.05074v2)**

> **作者:** Tommaso Apicella; Alessio Xompero; Andrea Cavallaro
>
> **备注:** 18 pages, 3 figures, 13 tables. Project website at https://apicis.github.io/aff-survey/
>
> **摘要:** Affordances are the potential actions an agent can perform on an object, as observed by a camera. Visual affordance prediction is formulated differently for tasks such as grasping detection, affordance classification, affordance segmentation, and hand pose estimation. This diversity in formulations leads to inconsistent definitions that prevent fair comparisons between methods. In this paper, we propose a unified formulation of visual affordance prediction by accounting for the complete information on the objects of interest and the interaction of the agent with the objects to accomplish a task. This unified formulation allows us to comprehensively and systematically review disparate visual affordance works, highlighting strengths and limitations of both methods and datasets. We also discuss reproducibility issues, such as the unavailability of methods implementation and experimental setups details, making benchmarks for visual affordance prediction unfair and unreliable. To favour transparency, we introduce the Affordance Sheet, a document that details the solution, datasets, and validation of a method, supporting future reproducibility and fairness in the community.
>
---
#### [replaced 008] PolySim: Bridging the Sim-to-Real Gap for Humanoid Control via Multi-Simulator Dynamics Randomization
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.01708v3](http://arxiv.org/pdf/2510.01708v3)**

> **作者:** Zixing Lei; Zibo Zhou; Sheng Yin; Yueru Chen; Qingyao Xu; Weixin Li; Yunhong Wang; Bowei Tang; Wei Jing; Siheng Chen
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Humanoid whole-body control (WBC) policies trained in simulation often suffer from the sim-to-real gap, which fundamentally arises from simulator inductive bias, the inherent assumptions and limitations of any single simulator. These biases lead to nontrivial discrepancies both across simulators and between simulation and the real world. To mitigate the effect of simulator inductive bias, the key idea is to train policies jointly across multiple simulators, encouraging the learned controller to capture dynamics that generalize beyond any single simulator's assumptions. We thus introduce PolySim, a WBC training platform that integrates multiple heterogeneous simulators. PolySim can launch parallel environments from different engines simultaneously within a single training run, thereby realizing dynamics-level domain randomization. Theoretically, we show that PolySim yields a tighter upper bound on simulator inductive bias than single-simulator training. In experiments, PolySim substantially reduces motion-tracking error in sim-to-sim evaluations; for example, on MuJoCo, it improves execution success by 52.8 over an IsaacSim baseline. PolySim further enables zero-shot deployment on a real Unitree G1 without additional fine-tuning, showing effective transfer from simulation to the real world. We will release the PolySim code upon acceptance of this work.
>
---
#### [replaced 009] PSN Game: Game-theoretic Prediction and Planning via a Player Selection Network
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2505.00213v2](http://arxiv.org/pdf/2505.00213v2)**

> **作者:** Tianyu Qiu; Eric Ouano; Fernando Palafox; Christian Ellis; David Fridovich-Keil
>
> **摘要:** While game-theoretic planning frameworks are effective at modeling multi-agent interactions, they require solving large optimization problems where the number of variables increases with the number of agents, resulting in long computation times that limit their use in large-scale, real-time systems. To address this issue, we propose 1) PSN Game: a learning-based, game-theoretic prediction and planning framework that reduces runtime by learning a Player Selection Network (PSN); and 2) a Goal Inference Network (GIN) that makes it possible to use the PSN in incomplete information games where agents' intentions are unknown. A PSN outputs a player selection mask that distinguishes influential players from less relevant ones, enabling the ego player to solve a smaller, masked game involving only selected players. By reducing the number of players in the game, and therefore reducing the number of variables in the corresponding optimization problem, PSN directly lowers computation time. The PSN Game framework is more flexible than existing player selection methods as it 1) relies solely on observations of players' past trajectories, without requiring full state, action, or other game-specific information; and 2) requires no online parameter tuning. Experiments in both simulated scenarios and human trajectory datasets demonstrate that PSNs outperform baseline selection methods in 1) prediction accuracy; and 2) planning safety. PSNs also generalize effectively to real-world scenarios in which agents' objectives are unknown without fine-tuning. By selecting only the most relevant players for decision-making, PSN Game offers a general mechanism for reducing planning complexity that can be seamlessly integrated into existing multi-agent planning frameworks.
>
---
#### [replaced 010] No Plan but Everything Under Control: Robustly Solving Sequential Tasks with Dynamically Composed Gradient Descent
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01732v2](http://arxiv.org/pdf/2503.01732v2)**

> **作者:** Vito Mengers; Oliver Brock
>
> **备注:** Accepted at ICRA25; Supplementary Material under https://www.tu.berlin/robotics/papers/noplan ; 7 pages + 6 figures;
>
> **摘要:** We introduce a novel gradient-based approach for solving sequential tasks by dynamically adjusting the underlying myopic potential field in response to feedback and the world's regularities. This adjustment implicitly considers subgoals encoded in these regularities, enabling the solution of long sequential tasks, as demonstrated by solving the traditional planning domain of Blocks World - without any planning. Unlike conventional planning methods, our feedback-driven approach adapts to uncertain and dynamic environments, as demonstrated by one hundred real-world trials involving drawer manipulation. These experiments highlight the robustness of our method compared to planning and show how interactive perception and error recovery naturally emerge from gradient descent without explicitly implementing them. This offers a computationally efficient alternative to planning for a variety of sequential tasks, while aligning with observations on biological problem-solving strategies.
>
---
#### [replaced 011] Dynamics-aware Diffusion Models for Planning and Control
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2504.00236v3](http://arxiv.org/pdf/2504.00236v3)**

> **作者:** Darshan Gadginmath; Fabio Pasqualetti
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** This paper addresses the problem of generating dynamically admissible trajectories for control tasks using diffusion models, particularly in scenarios where the environment is complex and system dynamics are crucial for practical application. We propose a novel framework that integrates system dynamics directly into the diffusion model's denoising process through a sequential prediction and projection mechanism. This mechanism, aligned with the diffusion model's noising schedule, ensures generated trajectories are both consistent with expert demonstrations and adhere to underlying physical constraints. Notably, our approach can generate maximum likelihood trajectories and accurately recover trajectories generated by linear feedback controllers, even when explicit dynamics knowledge is unavailable. We validate the effectiveness of our method through experiments on standard control tasks and a complex non-convex optimal control problem involving waypoint tracking and collision avoidance, demonstrating its potential for efficient trajectory generation in practical applications. Our code repository is available at www.github.com/darshangm/dynamics-aware-diffusion.
>
---
#### [replaced 012] SIG-Chat: Spatial Intent-Guided Conversational Gesture Generation Involving How, When and Where
- **分类: cs.GR; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.23852v2](http://arxiv.org/pdf/2509.23852v2)**

> **作者:** Yiheng Huang; Junran Peng; Silei Shen; Jingwei Yang; ZeJi Wei; ChenCheng Bai; Yonghao He; Wei Sui; Muyi Sun; Yan Liu; Xu-Cheng Yin; Man Zhang; Zhaoxiang Zhang; Chuanchen Luo
>
> **摘要:** The accompanying actions and gestures in dialogue are often closely linked to interactions with the environment, such as looking toward the interlocutor or using gestures to point to the described target at appropriate moments. Speech and semantics guide the production of gestures by determining their timing (WHEN) and style (HOW), while the spatial locations of interactive objects dictate their directional execution (WHERE). Existing approaches either rely solely on descriptive language to generate motions or utilize audio to produce non-interactive gestures, thereby lacking the characterization of interactive timing and spatial intent. This significantly limits the applicability of conversational gesture generation, whether in robotics or in the fields of game and animation production. To address this gap, we present a full-stack solution. We first established a unique data collection method to simultaneously capture high-precision human motion and spatial intent. We then developed a generation model driven by audio, language, and spatial data, alongside dedicated metrics for evaluating interaction timing and spatial accuracy. Finally, we deployed the solution on a humanoid robot, enabling rich, context-aware physical interactions.
>
---
#### [replaced 013] HEAL: An Empirical Study on Hallucinations in Embodied Agents Driven by Large Language Models
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.15065v2](http://arxiv.org/pdf/2506.15065v2)**

> **作者:** Trishna Chakraborty; Udita Ghosh; Xiaopan Zhang; Fahim Faisal Niloy; Yue Dong; Jiachen Li; Amit K. Roy-Chowdhury; Chengyu Song
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** Large language models (LLMs) are increasingly being adopted as the cognitive core of embodied agents. However, inherited hallucinations, which stem from failures to ground user instructions in the observed physical environment, can lead to navigation errors, such as searching for a refrigerator that does not exist. In this paper, we present the first systematic study of hallucinations in LLM-based embodied agents performing long-horizon tasks under scene-task inconsistencies. Our goal is to understand to what extent hallucinations occur, what types of inconsistencies trigger them, and how current models respond. To achieve these goals, we construct a hallucination probing set by building on an existing benchmark, capable of inducing hallucination rates up to 40x higher than base prompts. Evaluating 12 models across two simulation environments, we find that while models exhibit reasoning, they fail to resolve scene-task inconsistencies-highlighting fundamental limitations in handling infeasible tasks. We also provide actionable insights on ideal model behavior for each scenario, offering guidance for developing more robust and reliable planning strategies.
>
---
#### [replaced 014] mmWave Radar-Based Non-Line-of-Sight Pedestrian Localization at T-Junctions Utilizing Road Layout Extraction via Camera
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.02348v2](http://arxiv.org/pdf/2508.02348v2)**

> **作者:** Byeonggyu Park; Hee-Yeun Kim; Byonghyok Choi; Hansang Cho; Byungkwan Kim; Soomok Lee; Mingu Jeon; Seong-Woo Kim
>
> **摘要:** Pedestrians Localization in Non-Line-of-Sight (NLoS) regions within urban environments poses a significant challenge for autonomous driving systems. While mmWave radar has demonstrated potential for detecting objects in such scenarios, the 2D radar point cloud (PCD) data is susceptible to distortions caused by multipath reflections, making accurate spatial inference difficult. Additionally, although camera images provide high-resolution visual information, they lack depth perception and cannot directly observe objects in NLoS regions. In this paper, we propose a novel framework that interprets radar PCD through road layout inferred from camera for localization of NLoS pedestrians. The proposed method leverages visual information from the camera to interpret 2D radar PCD, enabling spatial scene reconstruction. The effectiveness of the proposed approach is validated through experiments conducted using a radar-camera system mounted on a real vehicle. The localization performance is evaluated using a dataset collected in outdoor NLoS driving environments, demonstrating the practical applicability of the method.
>
---
#### [replaced 015] Product-oriented Product-Process-Resource Asset Network and its Representation in AutomationML for Asset Administration Shell
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2510.00933v3](http://arxiv.org/pdf/2510.00933v3)**

> **作者:** Sara Strakosova; Petr Novak; Petr Kadera
>
> **备注:** \copyright 2024 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Current products, especially in the automotive sector, pose complex technical systems having a multi-disciplinary mechatronic nature. Industrial standards supporting system engineering and production typically (i) address the production phase only, but do not cover the complete product life cycle, and (ii) focus on production processes and resources rather than the products themselves. The presented approach is motivated by incorporating the impacts of the end-of-life phase of the product life cycle into the engineering phase. This paper proposes a modeling approach coming up from the Product-Process-Resource (PPR) modeling paradigm. It combines requirements on (i) respecting the product structure as a basis for the model, and (ii) incorporates repairing, remanufacturing, or upcycling within cyber-physical production systems. The proposed model called PoPAN should accompany the product during the entire life cycle as a digital shadow encapsulated within the Asset Administration Shell of a product. To facilitate the adoption of the proposed paradigm, the paper also proposes serialization of the model in the AutomationML data format. The model is demonstrated on a use-case for disassembling electric vehicle batteries to support their remanufacturing for stationary battery applications.
>
---
#### [replaced 016] MTIL: Encoding Full History with Mamba for Temporal Imitation Learning
- **分类: cs.RO; I.2.9**

- **链接: [http://arxiv.org/pdf/2505.12410v2](http://arxiv.org/pdf/2505.12410v2)**

> **作者:** Yulin Zhou; Yuankai Lin; Fanzhe Peng; Jiahui Chen; Kaiji Huang; Hua Yang; Zhouping Yin
>
> **备注:** 8 pages,5 figures.Published in IEEE Robotics and Automation Letters (RA-L), 2025
>
> **摘要:** Standard imitation learning (IL) methods have achieved considerable success in robotics, yet often rely on the Markov assumption, which falters in long-horizon tasks where history is crucial for resolving perceptual ambiguity. This limitation stems not only from a conceptual gap but also from a fundamental computational barrier: prevailing architectures like Transformers are often constrained by quadratic complexity, rendering the processing of long, high-dimensional observation sequences infeasible. To overcome this dual challenge, we introduce Mamba Temporal Imitation Learning (MTIL). Our approach represents a new paradigm for robotic learning, which we frame as a practical synthesis of World Model and Dynamical System concepts. By leveraging the linear-time recurrent dynamics of State Space Models (SSMs), MTIL learns an implicit, action-oriented world model that efficiently encodes the entire trajectory history into a compressed, evolving state. This allows the policy to be conditioned on a comprehensive temporal context, transcending the confines of Markovian approaches. Through extensive experiments on simulated benchmarks (ACT, Robomimic, LIBERO) and on challenging real-world tasks, MTIL demonstrates superior performance against SOTA methods like ACT and Diffusion Policy, particularly in resolving long-term temporal ambiguities. Our findings not only affirm the necessity of full temporal context but also validate MTIL as a powerful and a computationally feasible approach for learning long-horizon, non-Markovian behaviors from high-dimensional observations.
>
---
#### [replaced 017] GPA-RAM: Grasp-Pretraining Augmented Robotic Attention Mamba for Spatial Task Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.19683v2](http://arxiv.org/pdf/2504.19683v2)**

> **作者:** Juyi Sheng; Yangjun Liu; Sheng Xu; Zhixin Yang; Mengyuan Liu
>
> **摘要:** Most existing robot manipulation methods prioritize task learning by enhancing perception through complex deep network architectures. However, they face challenges in real-time collision-free planning. Hence, Robotic Attention Mamba (RAM) is designed for refined planning. Specifically, by integrating Mamba and parallel single-view attention, RAM aligns multi-view vision and task-related language features, ensuring efficient fine-grained task planning with linear complexity and robust real-time performance. Nevertheless, it has the potential for further improvement in high-precision grasping and manipulation. Thus, Grasp-Pretraining Augmentation (GPA) is devised, with a grasp pose feature extractor pretrained utilizing object grasp poses directly inherited from whole-task demonstrations. Subsequently, the extracted grasp features are fused with the spatially aligned planning features from RAM through attention-based Pre-trained Location Fusion, preserving high-resolution grasping cues overshadowed by an overemphasis on global planning. To summarize, we propose Grasp-Pretraining Augmented Robotic Attention Mamba (GPA-RAM), dividing spatial task learning into RAM for planning skill learning and GPA for grasping skill learning. GPA-RAM demonstrates superior performance across three robot systems with distinct camera configurations in simulation and the real world. Compared with previous state-of-the-art methods, it improves the absolute success rate by 8.2% (from 79.3% to 87.5%) on the RLBench multi-task benchmark and 40% (from 16% to 56%), 12% (from 86% to 98%) on the ALOHA bimanual manipulation tasks, while delivering notably faster inference. Furthermore, experimental results demonstrate that both RAM and GPA enhance task learning, with GPA proving robust to different architectures of pretrained grasp pose feature extractors. The project is https://logssim.github.io/GPA_RAM_website/
>
---
#### [replaced 018] Towards Safe Maneuvering of Double-Ackermann-Steering Robots with a Soft Actor-Critic Framework
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10332v2](http://arxiv.org/pdf/2510.10332v2)**

> **作者:** Kohio Deflesselle; Mélodie Daniel; Aly Magassouba; Miguel Aranda; Olivier Ly
>
> **备注:** 4 pages, 3 figures, 2 tables, Accepted for Safety of Intelligent and Autonomous Vehicles: Formal Methods vs. Machine Learning approaches for reliable navigation (SIAV-FM2L) an IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) workshop
>
> **摘要:** We present a deep reinforcement learning framework based on Soft Actor-Critic (SAC) for safe and precise maneuvering of double-Ackermann-steering mobile robots (DASMRs). Unlike holonomic or simpler non-holonomic robots such as differential-drive robots, DASMRs face strong kinematic constraints that make classical planners brittle in cluttered environments. Our framework leverages the Hindsight Experience Replay (HER) and the CrossQ overlay to encourage maneuvering efficiency while avoiding obstacles. Simulation results with a heavy four-wheel-steering rover show that the learned policy can robustly reach up to 97% of target positions while avoiding obstacles. Our framework does not rely on handcrafted trajectories or expert demonstrations.
>
---
#### [replaced 019] How Vulnerable Is My Learned Policy? Universal Adversarial Perturbation Attacks On Modern Behavior Cloning Policies
- **分类: cs.LG; cs.CR; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.03698v3](http://arxiv.org/pdf/2502.03698v3)**

> **作者:** Akansha Kalra; Basavasagar Patil; Guanhong Tao; Daniel S. Brown
>
> **摘要:** Learning from Demonstration (LfD) algorithms have shown promising results in robotic manipulation tasks, but their vulnerability to offline universal perturbation attacks remains underexplored. This paper presents a comprehensive study of adversarial attacks on both classic and recently proposed algorithms, including Behavior Cloning (BC), LSTM-GMM, Implicit Behavior Cloning (IBC), Diffusion Policy (DP), and Vector-Quantizied Behavior Transformer (VQ-BET). We study the vulnerability of these methods to universal adversarial perturbations. Our experiments on several simulated robotic manipulation tasks reveal that most of the current methods are highly vulnerable to adversarial perturbations. We also show that these attacks are often transferable across algorithms, architectures, and tasks, raising concerning security vulnerabilities to black-box attacks. To the best of our knowledge, we are the first to present a systematic study of the vulnerabilities of different LfD algorithms to both white-box and black-box attacks. Our findings highlight the vulnerabilities of modern BC algorithms, paving the way for future work in addressing such limitations.
>
---
#### [replaced 020] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07961v2](http://arxiv.org/pdf/2506.07961v2)**

> **作者:** Peiyan Li; Yixiang Chen; Hongtao Wu; Xiao Ma; Xiangnan Wu; Yan Huang; Liang Wang; Tao Kong; Tieniu Tan
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recently, leveraging pre-trained vision-language models (VLMs) for building vision-language-action (VLA) models has emerged as a promising approach to effective robot manipulation learning. However, only few methods incorporate 3D signals into VLMs for action prediction, and they do not fully leverage the spatial structure inherent in 3D data, leading to low sample efficiency. In this paper, we introduce BridgeVLA, a novel 3D VLA model that (1) projects 3D inputs to multiple 2D images, ensuring input alignment with the VLM backbone, and (2) utilizes 2D heatmaps for action prediction, unifying the input and output spaces within a consistent 2D image space. In addition, we propose a scalable pre-training method that equips the VLM backbone with the capability to predict 2D heatmaps before downstream policy learning. Extensive experiments show the proposed method is able to learn 3D manipulation efficiently and effectively. BridgeVLA outperforms state-of-the-art baseline methods across three simulation benchmarks. In RLBench, it improves the average success rate from 81.4% to 88.2%. In COLOSSEUM, it demonstrates significantly better performance in challenging generalization settings, boosting the average success rate from 56.7% to 64.0%. In GemBench, it surpasses all the comparing baseline methods in terms of average success rate. In real-robot experiments, BridgeVLA outperforms a state-of-the-art baseline method by 32% on average. It generalizes robustly in multiple out-of-distribution settings, including visual disturbances and unseen instructions. Remarkably, it is able to achieve a success rate of 96.8% on 10+ tasks with only 3 trajectories per task, highlighting its extraordinary sample efficiency. Project Website:https://bridgevla.github.io/
>
---
#### [replaced 021] InternScenes: A Large-scale Simulatable Indoor Scene Dataset with Realistic Layouts
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.10813v2](http://arxiv.org/pdf/2509.10813v2)**

> **作者:** Weipeng Zhong; Peizhou Cao; Yichen Jin; Li Luo; Wenzhe Cai; Jingli Lin; Hanqing Wang; Zhaoyang Lyu; Tai Wang; Bo Dai; Xudong Xu; Jiangmiao Pang
>
> **摘要:** The advancement of Embodied AI heavily relies on large-scale, simulatable 3D scene datasets characterized by scene diversity and realistic layouts. However, existing datasets typically suffer from limitations in data scale or diversity, sanitized layouts lacking small items, and severe object collisions. To address these shortcomings, we introduce \textbf{InternScenes}, a novel large-scale simulatable indoor scene dataset comprising approximately 40,000 diverse scenes by integrating three disparate scene sources, real-world scans, procedurally generated scenes, and designer-created scenes, including 1.96M 3D objects and covering 15 common scene types and 288 object classes. We particularly preserve massive small items in the scenes, resulting in realistic and complex layouts with an average of 41.5 objects per region. Our comprehensive data processing pipeline ensures simulatability by creating real-to-sim replicas for real-world scans, enhances interactivity by incorporating interactive objects into these scenes, and resolves object collisions by physical simulations. We demonstrate the value of InternScenes with two benchmark applications: scene layout generation and point-goal navigation. Both show the new challenges posed by the complex and realistic layouts. More importantly, InternScenes paves the way for scaling up the model training for both tasks, making the generation and navigation in such complex scenes possible. We commit to open-sourcing the data, models, and benchmarks to benefit the whole community.
>
---
#### [replaced 022] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.19764v2](http://arxiv.org/pdf/2503.19764v2)**

> **作者:** Christina Kassab; Sacha Morin; Martin Büchner; Matías Mattamala; Kumaraditya Gupta; Abhinav Valada; Liam Paull; Maurice Fallon
>
> **备注:** NeurIPS 2025
>
> **摘要:** 3D scene understanding has been transformed by open-vocabulary language models that enable interaction via natural language. However, at present the evaluation of these representations is limited to datasets with closed-set semantics that do not capture the richness of language. This work presents OpenLex3D, a dedicated benchmark for evaluating 3D open-vocabulary scene representations. OpenLex3D provides entirely new label annotations for scenes from Replica, ScanNet++, and HM3D, which capture real-world linguistic variability by introducing synonymical object categories and additional nuanced descriptions. Our label sets provide 13 times more labels per scene than the original datasets. By introducing an open-set 3D semantic segmentation task and an object retrieval task, we evaluate various existing 3D open-vocabulary methods on OpenLex3D, showcasing failure cases, and avenues for improvement. Our experiments provide insights on feature precision, segmentation, and downstream capabilities. The benchmark is publicly available at: https://openlex3d.github.io/.
>
---
#### [replaced 023] Integration of the TIAGo Robot into Isaac Sim with Mecanum Drive Modeling and Learned S-Curve Velocity Profiles
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.10273v2](http://arxiv.org/pdf/2510.10273v2)**

> **作者:** Vincent Schoenbach; Marvin Wiedemann; Raphael Memmesheimer; Malte Mosbach; Sven Behnke
>
> **备注:** In Proceedings of IEEE 21st International Conference on Automation Science and Engineering (CASE), Los Angeles, USA, August 2025
>
> **摘要:** Efficient physics simulation has significantly accelerated research progress in robotics applications such as grasping and assembly. The advent of GPU-accelerated simulation frameworks like Isaac Sim has particularly empowered learning-based methods, enabling them to tackle increasingly complex tasks. The PAL Robotics TIAGo++ Omni is a versatile mobile manipulator equipped with a mecanum-wheeled base, allowing omnidirectional movement and a wide range of task capabilities. However, until now, no model of the robot has been available in Isaac Sim. In this paper, we introduce such a model, calibrated to approximate the behavior of the real robot, with a focus on its omnidirectional drive dynamics. We present two control models for the omnidirectional drive: a physically accurate model that replicates real-world wheel dynamics and a lightweight velocity-based model optimized for learning-based applications. With these models, we introduce a learning-based calibration approach to approximate the real robot's S-shaped velocity profile using minimal trajectory data recordings. This simulation should allow researchers to experiment with the robot and perform efficient learning-based control in diverse environments. We provide the integration publicly at https://github.com/AIS-Bonn/tiago_isaac.
>
---
#### [replaced 024] ManiAgent: An Agentic Framework for General Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.11660v2](http://arxiv.org/pdf/2510.11660v2)**

> **作者:** Yi Yang; Kefan Gu; Yuqing Wen; Hebei Li; Yucheng Zhao; Tiancai Wang; Xudong Liu
>
> **备注:** 8 pages, 6 figures, conference
>
> **摘要:** While Vision-Language-Action (VLA) models have demonstrated impressive capabilities in robotic manipulation, their performance in complex reasoning and long-horizon task planning is limited by data scarcity and model capacity. To address this, we introduce ManiAgent, an agentic architecture for general manipulation tasks that achieves end-to-end output from task descriptions and environmental inputs to robotic manipulation actions. In this framework, multiple agents involve inter-agent communication to perform environmental perception, sub-task decomposition and action generation, enabling efficient handling of complex manipulation scenarios. Evaluations show ManiAgent achieves an 86.8% success rate on the SimplerEnv benchmark and 95.8% on real-world pick-and-place tasks, enabling efficient data collection that yields VLA models with performance comparable to those trained on human-annotated datasets. The project webpage is available at https://yi-yang929.github.io/ManiAgent/.
>
---
