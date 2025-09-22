# 机器人 cs.RO

- **最新发布 54 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] Distribution Estimation for Global Data Association via Approximate Bayesian Inference
- **分类: cs.RO**

- **简介: 该论文针对机器人全局数据关联任务，提出基于近似贝叶斯推断的框架，解决因环境重复或对称导致的多模态解分布问题。通过粒子表示多种假设，有效估计变换分布，适用于点云或对象地图配准。**

- **链接: [http://arxiv.org/pdf/2509.15565v1](http://arxiv.org/pdf/2509.15565v1)**

> **作者:** Yixuan Jia; Mason B. Peterson; Qingyuan Li; Yulun Tian; Jonathan P. How
>
> **备注:** 9 pages
>
> **摘要:** Global data association is an essential prerequisite for robot operation in environments seen at different times or by different robots. Repetitive or symmetric data creates significant challenges for existing methods, which typically rely on maximum likelihood estimation or maximum consensus to produce a single set of associations. However, in ambiguous scenarios, the distribution of solutions to global data association problems is often highly multimodal, and such single-solution approaches frequently fail. In this work, we introduce a data association framework that leverages approximate Bayesian inference to capture multiple solution modes to the data association problem, thereby avoiding premature commitment to a single solution under ambiguity. Our approach represents hypothetical solutions as particles that evolve according to a deterministic or randomized update rule to cover the modes of the underlying solution distribution. Furthermore, we show that our method can incorporate optimization constraints imposed by the data association formulation and directly benefit from GPU-parallelized optimization. Extensive simulated and real-world experiments with highly ambiguous data show that our method correctly estimates the distribution over transformations when registering point clouds or object maps.
>
---
#### [new 002] An MPC framework for efficient navigation of mobile robots in cluttered environments
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **简介: 该论文提出一种基于模型预测控制（MPC）的框架，用于复杂环境中移动机器人的高效导航。通过将有限段最短路径规划器与MPC轨迹优化结合，实现了动态目标收敛和避障，验证了机器人在复杂环境中的快速响应与导航能力。**

- **链接: [http://arxiv.org/pdf/2509.15917v1](http://arxiv.org/pdf/2509.15917v1)**

> **作者:** Johannes Köhler; Daniel Zhang; Raffaele Soloperto; Andrea Carron; Melanie Zeilinger
>
> **备注:** - Code available at: https://github.com/IntelligentControlSystems/ClutteredEnvironment - Supplementary video: https://youtu.be/Hn_hpAmGgq0
>
> **摘要:** We present a model predictive control (MPC) framework for efficient navigation of mobile robots in cluttered environments. The proposed approach integrates a finite-segment shortest path planner into the finite-horizon trajectory optimization of the MPC. This formulation ensures convergence to dynamically selected targets and guarantees collision avoidance, even under general nonlinear dynamics and cluttered environments. The approach is validated through hardware experiments on a small ground robot, where a human operator dynamically assigns target locations. The robot successfully navigated through complex environments and reached new targets within 2-3 seconds.
>
---
#### [new 003] Defining and Monitoring Complex Robot Activities via LLMs and Symbolic Reasoning
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出一种结合大语言模型（LLMs）与自动规划的架构，用于定义和监控复杂机器人活动。旨在解决动态环境中高阶任务的自然语言描述与执行监控问题，并在精准农业场景中进行了验证。**

- **链接: [http://arxiv.org/pdf/2509.16006v1](http://arxiv.org/pdf/2509.16006v1)**

> **作者:** Francesco Argenziano; Elena Umili; Francesco Leotta; Daniele Nardi
>
> **摘要:** Recent years have witnessed a growing interest in automating labor-intensive and complex activities, i.e., those consisting of multiple atomic tasks, by deploying robots in dynamic and unpredictable environments such as industrial and agricultural settings. A key characteristic of these contexts is that activities are not predefined: while they involve a limited set of possible tasks, their combinations may vary depending on the situation. Moreover, despite recent advances in robotics, the ability for humans to monitor the progress of high-level activities - in terms of past, present, and future actions - remains fundamental to ensure the correct execution of safety-critical processes. In this paper, we introduce a general architecture that integrates Large Language Models (LLMs) with automated planning, enabling humans to specify high-level activities (also referred to as processes) using natural language, and to monitor their execution by querying a robot. We also present an implementation of this architecture using state-of-the-art components and quantitatively evaluate the approach in a real-world precision agriculture scenario.
>
---
#### [new 004] STARC: See-Through-Wall Augmented Reality Framework for Human-Robot Collaboration in Emergency Response
- **分类: cs.RO**

- **简介: 该论文提出STARC，一种用于应急响应中人机协作的透视AR框架。通过融合移动机器人与穿戴式LiDAR数据，解决遮挡环境下救援人员视线受阻的问题，实现隐藏人员和危险的实时可视化，提升态势感知能力。**

- **链接: [http://arxiv.org/pdf/2509.15507v1](http://arxiv.org/pdf/2509.15507v1)**

> **作者:** Shenghai Yuan; Weixiang Guo; Tianxin Hu; Yu Yang; Jinyu Chen; Rui Qian; Zhongyuan Liu; Lihua Xie
>
> **摘要:** In emergency response missions, first responders must navigate cluttered indoor environments where occlusions block direct line-of-sight, concealing both life-threatening hazards and victims in need of rescue. We present STARC, a see-through AR framework for human-robot collaboration that fuses mobile-robot mapping with responder-mounted LiDAR sensing. A ground robot running LiDAR-inertial odometry performs large-area exploration and 3D human detection, while helmet- or handheld-mounted LiDAR on the responder is registered to the robot's global map via relative pose estimation. This cross-LiDAR alignment enables consistent first-person projection of detected humans and their point clouds - rendered in AR with low latency - into the responder's view. By providing real-time visualization of hidden occupants and hazards, STARC enhances situational awareness and reduces operator risk. Experiments in simulation, lab setups, and tactical field trials confirm robust pose alignment, reliable detections, and stable overlays, underscoring the potential of our system for fire-fighting, disaster relief, and other safety-critical operations. Code and design will be open-sourced upon acceptance.
>
---
#### [new 005] PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models
- **分类: cs.RO**

- **简介: 该论文提出PRIMT，一种基于偏好的强化学习框架，旨在解决机器人复杂行为训练中依赖大量人工输入及奖励学习中的模糊性和归因问题。通过融合多模态基础模型和轨迹合成技术，提升了反馈可靠性和任务性能。**

- **链接: [http://arxiv.org/pdf/2509.15607v1](http://arxiv.org/pdf/2509.15607v1)**

> **作者:** Ruiqi Wang; Dezhong Zhao; Ziqin Yuan; Tianyu Shao; Guohua Chen; Dominic Kao; Sungeun Hong; Byung-Cheol Min
>
> **摘要:** Preference-based reinforcement learning (PbRL) has emerged as a promising paradigm for teaching robots complex behaviors without reward engineering. However, its effectiveness is often limited by two critical challenges: the reliance on extensive human input and the inherent difficulties in resolving query ambiguity and credit assignment during reward learning. In this paper, we introduce PRIMT, a PbRL framework designed to overcome these challenges by leveraging foundation models (FMs) for multimodal synthetic feedback and trajectory synthesis. Unlike prior approaches that rely on single-modality FM evaluations, PRIMT employs a hierarchical neuro-symbolic fusion strategy, integrating the complementary strengths of large language models and vision-language models in evaluating robot behaviors for more reliable and comprehensive feedback. PRIMT also incorporates foresight trajectory generation, which reduces early-stage query ambiguity by warm-starting the trajectory buffer with bootstrapped samples, and hindsight trajectory augmentation, which enables counterfactual reasoning with a causal auxiliary loss to improve credit assignment. We evaluate PRIMT on 2 locomotion and 6 manipulation tasks on various benchmarks, demonstrating superior performance over FM-based and scripted baselines.
>
---
#### [new 006] Learning Safety for Obstacle Avoidance via Control Barrier Functions
- **分类: cs.RO**

- **简介: 该论文研究机器人避障安全控制问题，针对复杂几何形状机器人在密集环境中的障碍物规避任务。提出基于残差神经网络预测安全距离，并结合离散高阶控制屏障函数（DHOCBF）与优化框架，实现高效、安全的避障控制，提升CBF方法的适用性与计算效率。**

- **链接: [http://arxiv.org/pdf/2509.16037v1](http://arxiv.org/pdf/2509.16037v1)**

> **作者:** Shuo Liu; Zhe Huang; Calin A. Belta
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Obstacle avoidance is central to safe navigation, especially for robots with arbitrary and nonconvex geometries operating in cluttered environments. Existing Control Barrier Function (CBF) approaches often rely on analytic clearance computations, which are infeasible for complex geometries, or on polytopic approximations, which become intractable when robot configurations are unknown. To address these limitations, this paper trains a residual neural network on a large dataset of robot-obstacle configurations to enable fast and tractable clearance prediction, even at unseen configurations. The predicted clearance defines the radius of a Local Safety Ball (LSB), which ensures continuous-time collision-free navigation. The LSB boundary is encoded as a Discrete-Time High-Order CBF (DHOCBF), whose constraints are incorporated into a nonlinear optimization framework. To improve feasibility, a novel relaxation technique is applied. The resulting framework ensure that the robot's rigid-body motion between consecutive time steps remains collision-free, effectively bridging discrete-time control and continuous-time safety. We show that the proposed method handles arbitrary, including nonconvex, robot geometries and generates collision-free, dynamically feasible trajectories in cluttered environments. Experiments demonstrate millisecond-level solve times and high prediction accuracy, highlighting both safety and efficiency beyond existing CBF-based methods.
>
---
#### [new 007] Improving Robotic Manipulation with Efficient Geometry-Aware Vision Encoder
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决传统视觉编码器缺乏3D理解的问题。研究将几何感知视觉编码器引入模仿学习，并提出高效版本eVGGT，在保证性能的同时提升计算效率。**

- **链接: [http://arxiv.org/pdf/2509.15880v1](http://arxiv.org/pdf/2509.15880v1)**

> **作者:** An Dinh Vuong; Minh Nhat Vu; Ian Reid
>
> **备注:** 9 figures, 7 tables. Project page: https://evggt.github.io/
>
> **摘要:** Existing RGB-based imitation learning approaches typically employ traditional vision encoders such as ResNet or ViT, which lack explicit 3D reasoning capabilities. Recent geometry-grounded vision models, such as VGGT~\cite{wang2025vggt}, provide robust spatial understanding and are promising candidates to address this limitation. This work investigates the integration of geometry-aware visual representations into robotic manipulation. Our results suggest that incorporating the geometry-aware vision encoder into imitation learning frameworks, including ACT and DP, yields up to 6.5% improvement over standard vision encoders in success rate across single- and bi-manual manipulation tasks in both simulation and real-world settings. Despite these benefits, most geometry-grounded models require high computational cost, limiting their deployment in practical robotic systems. To address this challenge, we propose eVGGT, an efficient geometry-aware encoder distilled from VGGT. eVGGT is nearly 9 times faster and 5 times smaller than VGGT, while preserving strong 3D reasoning capabilities. Code and pretrained models will be released to facilitate further research in geometry-aware robotics.
>
---
#### [new 008] Trust-Aware Embodied Bayesian Persuasion for Mixed-Autonomy
- **分类: cs.RO**

- **简介: 该论文提出Trust-Aware Embodied Bayesian Persuasion (TA-EBP)框架，用于提升自动驾驶与人类驾驶车辆在混合交通中的安全与效率。解决传统博弈论模型因缺乏信任导致的长期影响力衰减问题。通过引入信任参数和物理动作空间建模，验证了TA-EBP能有效引导人类谨慎驾驶，减少碰撞并优化交通流。**

- **链接: [http://arxiv.org/pdf/2509.15404v1](http://arxiv.org/pdf/2509.15404v1)**

> **作者:** Shaoting Peng; Katherine Driggs-Campbell; Roy Dong
>
> **摘要:** Safe and efficient interaction between autonomous vehicles (AVs) and human-driven vehicles (HVs) is a critical challenge for future transportation systems. While game-theoretic models capture how AVs influence HVs, they often suffer from a long-term decay of influence and can be perceived as manipulative, eroding the human's trust. This can paradoxically lead to riskier human driving behavior over repeated interactions. In this paper, we address this challenge by proposing the Trust-Aware Embodied Bayesian Persuasion (TA-EBP) framework. Our work makes three key contributions: First, we apply Bayesian persuasion to model communication at traffic intersections, offering a transparent alternative to traditional game-theoretic models. Second, we introduce a trust parameter to the persuasion framework, deriving a theorem for the minimum trust level required for influence. Finally, we ground the abstract signals of Bayesian persuasion theory into a continuous, physically meaningful action space, deriving a second theorem for the optimal signal magnitude, realized as an AV's forward nudge. Additionally, we validate our framework in a mixed-autonomy traffic simulation, demonstrating that TA-EBP successfully persuades HVs to drive more cautiously, eliminating collisions and improving traffic flow compared to baselines that either ignore trust or lack communication. Our work provides a transparent and non-strategic framework for influence in human-robot interaction, enhancing both safety and efficiency.
>
---
#### [new 009] SMART: Scalable Multi-Agent Reasoning and Trajectory Planning in Dense Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出SMART，一种用于密集环境中多车辆轨迹规划的可扩展框架。任务是解决大规模实时协同中的非凸碰撞约束问题。通过结合优先级搜索与分布式优化，有效提升规划效率与成功率，确保运动可行性与避碰。**

- **链接: [http://arxiv.org/pdf/2509.15737v1](http://arxiv.org/pdf/2509.15737v1)**

> **作者:** Heye Huang; Yibin Yang; Wang Chen; Tiantian Chen; Xiaopeng Li; Sikai Chen
>
> **摘要:** Multi-vehicle trajectory planning is a non-convex problem that becomes increasingly difficult in dense environments due to the rapid growth of collision constraints. Efficient exploration of feasible behaviors and resolution of tight interactions are essential for real-time, large-scale coordination. This paper introduces SMART, Scalable Multi-Agent Reasoning and Trajectory Planning, a hierarchical framework that combines priority-based search with distributed optimization to achieve efficient and feasible multi-vehicle planning. The upper layer explores diverse interaction modes using reinforcement learning-based priority estimation and large-step hybrid A* search, while the lower layer refines solutions via parallelizable convex optimization. By partitioning space among neighboring vehicles and constructing robust feasible corridors, the method decouples the joint non-convex problem into convex subproblems solved efficiently in parallel. This design alleviates the step-size trade-off while ensuring kinematic feasibility and collision avoidance. Experiments show that SMART consistently outperforms baselines. On 50 m x 50 m maps, it sustains over 90% success within 1 s up to 25 vehicles, while baselines often drop below 50%. On 100 m x 100 m maps, SMART achieves above 95% success up to 50 vehicles and remains feasible up to 90 vehicles, with runtimes more than an order of magnitude faster than optimization-only approaches. Built on vehicle-to-everything communication, SMART incorporates vehicle-infrastructure cooperation through roadside sensing and agent coordination, improving scalability and safety. Real-world experiments further validate this design, achieving planning times as low as 0.014 s while preserving cooperative behaviors.
>
---
#### [new 010] DSPv2: Improved Dense Policy for Effective and Generalizable Whole-body Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出DSPv2，用于解决全身移动机械臂的模仿学习问题。针对复杂观测处理、泛化能力及动作连贯性挑战，改进了密集策略架构，融合3D与多视角2D语义特征，实现更有效和泛化的全身控制。**

- **链接: [http://arxiv.org/pdf/2509.16063v1](http://arxiv.org/pdf/2509.16063v1)**

> **作者:** Yue Su; Chubin Zhang; Sijin Chen; Liufan Tan; Yansong Tang; Jianan Wang; Xihui Liu
>
> **摘要:** Learning whole-body mobile manipulation via imitation is essential for generalizing robotic skills to diverse environments and complex tasks. However, this goal is hindered by significant challenges, particularly in effectively processing complex observation, achieving robust generalization, and generating coherent actions. To address these issues, we propose DSPv2, a novel policy architecture. DSPv2 introduces an effective encoding scheme that aligns 3D spatial features with multi-view 2D semantic features. This fusion enables the policy to achieve broad generalization while retaining the fine-grained perception necessary for precise control. Furthermore, we extend the Dense Policy paradigm to the whole-body mobile manipulation domain, demonstrating its effectiveness in generating coherent and precise actions for the whole-body robotic platform. Extensive experiments show that our method significantly outperforms existing approaches in both task performance and generalization ability. Project page is available at: https://selen-suyue.github.io/DSPv2Net/.
>
---
#### [new 011] Implicit Kinodynamic Motion Retargeting for Human-to-humanoid Imitation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出隐式运动重定向（IKMR）框架，用于人到类人的模仿学习任务。旨在解决大规模人类动作转换为机器人可行轨迹的效率与可扩展性问题。通过结合运动学与动力学，实现高效、实时的全身控制器训练与部署。**

- **链接: [http://arxiv.org/pdf/2509.15443v1](http://arxiv.org/pdf/2509.15443v1)**

> **作者:** Xingyu Chen; Hanyu Wu; Sikai Wu; Mingliang Zhou; Diyun Xiang; Haodong Zhang
>
> **摘要:** Human-to-humanoid imitation learning aims to learn a humanoid whole-body controller from human motion. Motion retargeting is a crucial step in enabling robots to acquire reference trajectories when exploring locomotion skills. However, current methods focus on motion retargeting frame by frame, which lacks scalability. Could we directly convert large-scale human motion into robot-executable motion through a more efficient approach? To address this issue, we propose Implicit Kinodynamic Motion Retargeting (IKMR), a novel efficient and scalable retargeting framework that considers both kinematics and dynamics. In kinematics, IKMR pretrains motion topology feature representation and a dual encoder-decoder architecture to learn a motion domain mapping. In dynamics, IKMR integrates imitation learning with the motion retargeting network to refine motion into physically feasible trajectories. After fine-tuning using the tracking results, IKMR can achieve large-scale physically feasible motion retargeting in real time, and a whole-body controller could be directly trained and deployed for tracking its retargeted trajectories. We conduct our experiments both in the simulator and the real robot on a full-size humanoid robot. Extensive experiments and evaluation results verify the effectiveness of our proposed framework.
>
---
#### [new 012] Imagination at Inference: Synthesizing In-Hand Views for Robust Visuomotor Policy Inference
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉与控制任务，旨在解决缺乏手部视角摄像头时策略性能下降的问题。通过微调扩散模型实现推理时生成手部视角图像，提升策略鲁棒性，已在仿真和真实草莓采摘任务中验证效果。**

- **链接: [http://arxiv.org/pdf/2509.15717v1](http://arxiv.org/pdf/2509.15717v1)**

> **作者:** Haoran Ding; Anqing Duan; Zezhou Sun; Dezhen Song; Yoshihiko Nakamura
>
> **备注:** Submitted to IEEE for possible publication, under review
>
> **摘要:** Visual observations from different viewpoints can significantly influence the performance of visuomotor policies in robotic manipulation. Among these, egocentric (in-hand) views often provide crucial information for precise control. However, in some applications, equipping robots with dedicated in-hand cameras may pose challenges due to hardware constraints, system complexity, and cost. In this work, we propose to endow robots with imaginative perception - enabling them to 'imagine' in-hand observations from agent views at inference time. We achieve this via novel view synthesis (NVS), leveraging a fine-tuned diffusion model conditioned on the relative pose between the agent and in-hand views cameras. Specifically, we apply LoRA-based fine-tuning to adapt a pretrained NVS model (ZeroNVS) to the robotic manipulation domain. We evaluate our approach on both simulation benchmarks (RoboMimic and MimicGen) and real-world experiments using a Unitree Z1 robotic arm for a strawberry picking task. Results show that synthesized in-hand views significantly enhance policy inference, effectively recovering the performance drop caused by the absence of real in-hand cameras. Our method offers a scalable and hardware-light solution for deploying robust visuomotor policies, highlighting the potential of imaginative visual reasoning in embodied agents.
>
---
#### [new 013] Swarm Oracle: Trustless Blockchain Agreements through Robot Swarms
- **分类: cs.RO**

- **简介: 论文提出Swarm Oracle，一种基于自主机器人集群的区块链预言机系统，旨在解决区块链获取可信现实世界数据的问题。通过拜占庭容错协议和声誉机制，实现多方协作下的去信任共识，提升数据验证的可靠性和抗攻击能力。**

- **链接: [http://arxiv.org/pdf/2509.15956v1](http://arxiv.org/pdf/2509.15956v1)**

> **作者:** Alexandre Pacheco; Hanqing Zhao; Volker Strobel; Tarik Roukny; Gregory Dudek; Andreagiovanni Reina; Marco Dorigo
>
> **摘要:** Blockchain consensus, rooted in the principle ``don't trust, verify'', limits access to real-world data, which may be ambiguous or inaccessible to some participants. Oracles address this limitation by supplying data to blockchains, but existing solutions may reduce autonomy, transparency, or reintroduce the need for trust. We propose Swarm Oracle: a decentralized network of autonomous robots -- that is, a robot swarm -- that use onboard sensors and peer-to-peer communication to collectively verify real-world data and provide it to smart contracts on public blockchains. Swarm Oracle leverages the built-in decentralization, fault tolerance and mobility of robot swarms, which can flexibly adapt to meet information requests on-demand, even in remote locations. Unlike typical cooperative robot swarms, Swarm Oracle integrates robots from multiple stakeholders, protecting the system from single-party biases but also introducing potential adversarial behavior. To ensure the secure, trustless and global consensus required by blockchains, we employ a Byzantine fault-tolerant protocol that enables robots from different stakeholders to operate together, reaching social agreements of higher quality than the estimates of individual robots. Through extensive experiments using both real and simulated robots, we showcase how consensus on uncertain environmental information can be achieved, despite several types of attacks orchestrated by large proportions of the robots, and how a reputation system based on blockchain tokens lets Swarm Oracle autonomously recover from faults and attacks, a requirement for long-term operation.
>
---
#### [new 014] I-FailSense: Towards General Robotic Failure Detection with Vision-Language Models
- **分类: cs.RO**

- **简介: 该论文提出I-FailSense，一种基于视觉-语言模型的通用机器人故障检测框架，旨在解决语义错位等任务执行错误问题。通过构建针对性数据集和轻量分类模块，实现了高效、泛化的失败检测，并在多种环境中验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.16072v1](http://arxiv.org/pdf/2509.16072v1)**

> **作者:** Clemence Grislain; Hamed Rahimi; Olivier Sigaud; Mohamed Chetouani
>
> **摘要:** Language-conditioned robotic manipulation in open-world settings requires not only accurate task execution but also the ability to detect failures for robust deployment in real-world environments. Although recent advances in vision-language models (VLMs) have significantly improved the spatial reasoning and task-planning capabilities of robots, they remain limited in their ability to recognize their own failures. In particular, a critical yet underexplored challenge lies in detecting semantic misalignment errors, where the robot executes a task that is semantically meaningful but inconsistent with the given instruction. To address this, we propose a method for building datasets targeting Semantic Misalignment Failures detection, from existing language-conditioned manipulation datasets. We also present I-FailSense, an open-source VLM framework with grounded arbitration designed specifically for failure detection. Our approach relies on post-training a base VLM, followed by training lightweight classification heads, called FS blocks, attached to different internal layers of the VLM and whose predictions are aggregated using an ensembling mechanism. Experiments show that I-FailSense outperforms state-of-the-art VLMs, both comparable in size and larger, in detecting semantic misalignment errors. Notably, despite being trained only on semantic misalignment detection, I-FailSense generalizes to broader robotic failure categories and effectively transfers to other simulation environments and real-world with zero-shot or minimal post-training. The datasets and models are publicly released on HuggingFace (Webpage: https://clemgris.github.io/I-FailSense/).
>
---
#### [new 015] A Matter of Height: The Impact of a Robotic Object on Human Compliance
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究了机器人高度对人类服从性的影响。通过设计可调节高度的非人形服务机器人，测试其请求参与者完成任务时的服从情况。结果显示矮机器人比高机器人更易获得服从，与人类互动模式相反。研究强调不能直接将人类社交动态应用于机器人设计。**

- **链接: [http://arxiv.org/pdf/2509.16032v1](http://arxiv.org/pdf/2509.16032v1)**

> **作者:** Michael Faber; Andrey Grishko; Julian Waksberg; David Pardo; Tomer Leivy; Yuval Hazan; Emanuel Talmansky; Benny Megidish; Hadas Erel
>
> **备注:** 8 pages, 6 figures, 1 table, submitted to IEEE RO-MAN 2025
>
> **摘要:** Robots come in various forms and have different characteristics that may shape the interaction with them. In human-human interactions, height is a characteristic that shapes human dynamics, with taller people typically perceived as more persuasive. In this work, we aspired to evaluate if the same impact replicates in a human-robot interaction and specifically with a highly non-humanoid robotic object. The robot was designed with modules that could be easily added or removed, allowing us to change its height without altering other design features. To test the impact of the robot's height, we evaluated participants' compliance with its request to volunteer to perform a tedious task. In the experiment, participants performed a cognitive task on a computer, which was framed as the main experiment. When done, they were informed that the experiment was completed. While waiting to receive their credits, the robotic object, designed as a mobile robotic service table, entered the room, carrying a tablet that invited participants to complete a 300-question questionnaire voluntarily. We compared participants' compliance in two conditions: A Short robot composed of two modules and 95cm in height and a Tall robot consisting of three modules and 132cm in height. Our findings revealed higher compliance with the Short robot's request, demonstrating an opposite pattern to human dynamics. We conclude that while height has a substantial social impact on human-robot interactions, it follows a unique pattern of influence. Our findings suggest that designers cannot simply adopt and implement elements from human social dynamics to robots without testing them first.
>
---
#### [new 016] Real-Time Planning and Control with a Vortex Particle Model for Fixed-Wing UAVs in Unsteady Flows
- **分类: cs.RO**

- **简介: 该论文研究固定翼无人机在非定常气流中的实时规划与控制问题，提出基于涡旋粒子模型和策略优化方法，以提升无人机在复杂气动环境下的机动性能。**

- **链接: [http://arxiv.org/pdf/2509.16079v1](http://arxiv.org/pdf/2509.16079v1)**

> **作者:** Ashwin Gupta; Kevin Wolfe; Gino Perrotta; Joseph Moore
>
> **摘要:** Unsteady aerodynamic effects can have a profound impact on aerial vehicle flight performance, especially during agile maneuvers and in complex aerodynamic environments. In this paper, we present a real-time planning and control approach capable of reasoning about unsteady aerodynamics. Our approach relies on a lightweight vortex particle model, parallelized to allow GPU acceleration, and a sampling-based policy optimization strategy capable of leveraging the vortex particle model for predictive reasoning. We demonstrate, through both simulation and hardware experiments, that by replanning with our unsteady aerodynamics model, we can improve the performance of aggressive post-stall maneuvers in the presence of unsteady environmental flow disturbances.
>
---
#### [new 017] Momentum-constrained Hybrid Heuristic Trajectory Optimization Framework with Residual-enhanced DRL for Visually Impaired Scenarios
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出MHHTOF框架，结合动量约束轨迹优化与残差增强深度强化学习，用于视障人群的辅助导航任务。旨在提升路径规划的鲁棒性、安全性与实时性，通过双阶段成本建模和LSTM-ResB-PPO算法实现高效轨迹选择与优化。**

- **链接: [http://arxiv.org/pdf/2509.15582v1](http://arxiv.org/pdf/2509.15582v1)**

> **作者:** Yuting Zeng; Zhiwen Zheng; You Zhou; JiaLing Xiao; Yongbin Yu; Manping Fan; Bo Gong; Liyong Ren
>
> **备注:** 20 pages, 16 figures
>
> **摘要:** This paper proposes a momentum-constrained hybrid heuristic trajectory optimization framework (MHHTOF) tailored for assistive navigation in visually impaired scenarios, integrating trajectory sampling generation, optimization and evaluation with residual-enhanced deep reinforcement learning (DRL). In the first stage, heuristic trajectory sampling cluster (HTSC) is generated in the Frenet coordinate system using third-order interpolation with fifth-order polynomials and momentum-constrained trajectory optimization (MTO) constraints to ensure smoothness and feasibility. After first stage cost evaluation, the second stage leverages a residual-enhanced actor-critic network with LSTM-based temporal feature modeling to adaptively refine trajectory selection in the Cartesian coordinate system. A dual-stage cost modeling mechanism (DCMM) with weight transfer aligns semantic priorities across stages, supporting human-centered optimization. Experimental results demonstrate that the proposed LSTM-ResB-PPO achieves significantly faster convergence, attaining stable policy performance in approximately half the training iterations required by the PPO baseline, while simultaneously enhancing both reward outcomes and training stability. Compared to baseline method, the selected model reduces average cost and cost variance by 30.3% and 53.3%, and lowers ego and obstacle risks by over 77%. These findings validate the framework's effectiveness in enhancing robustness, safety, and real-time feasibility in complex assistive planning tasks.
>
---
#### [new 018] Latent Conditioned Loco-Manipulation Using Motion Priors
- **分类: cs.RO**

- **简介: 该论文研究人形和四足机器人的运动控制任务，旨在解决复杂任务中单一技能控制方法效率低的问题。提出通过模仿学习训练多功能运动策略，并引入潜在空间控制以适应高阶目标与物理约束，验证了在仿真和真实四足机器人上的效果。**

- **链接: [http://arxiv.org/pdf/2509.16061v1](http://arxiv.org/pdf/2509.16061v1)**

> **作者:** Maciej Stępień; Rafael Kourdis; Constant Roux; Olivier Stasse
>
> **备注:** https://gepetto.github.io/LaCoLoco/
>
> **摘要:** Although humanoid and quadruped robots provide a wide range of capabilities, current control methods, such as Deep Reinforcement Learning, focus mainly on single skills. This approach is inefficient for solving more complicated tasks where high-level goals, physical robot limitations and desired motion style might all need to be taken into account. A more effective approach is to first train a multipurpose motion policy that acquires low-level skills through imitation, while providing latent space control over skill execution. Then, this policy can be used to efficiently solve downstream tasks. This method has already been successful for controlling characters in computer graphics. In this work, we apply the approach to humanoid and quadrupedal loco-manipulation by imitating either simple synthetic motions or kinematically retargeted dog motions. We extend the original formulation to handle constraints, ensuring deployment safety, and use a diffusion discriminator for better imitation quality. We verify our methods by performing loco-manipulation in simulation for the H1 humanoid and Solo12 quadruped, as well as deploying policies on Solo12 hardware. Videos and code are available at https://gepetto.github.io/LaCoLoco/
>
---
#### [new 019] GP3: A 3D Geometry-Aware Policy with Multi-View Images for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出GP3，一种用于机器人操作的3D几何感知策略。它通过多视角图像输入，结合空间编码器和语言指令生成连续动作，在无需深度传感器的情况下实现高效、实用的机器人操作。**

- **链接: [http://arxiv.org/pdf/2509.15733v1](http://arxiv.org/pdf/2509.15733v1)**

> **作者:** Quanhao Qian; Guoyang Zhao; Gongjie Zhang; Jiuniu Wang; Ran Xu; Junlong Gao; Deli Zhao
>
> **摘要:** Effective robotic manipulation relies on a precise understanding of 3D scene geometry, and one of the most straightforward ways to acquire such geometry is through multi-view observations. Motivated by this, we present GP3 -- a 3D geometry-aware robotic manipulation policy that leverages multi-view input. GP3 employs a spatial encoder to infer dense spatial features from RGB observations, which enable the estimation of depth and camera parameters, leading to a compact yet expressive 3D scene representation tailored for manipulation. This representation is fused with language instructions and translated into continuous actions via a lightweight policy head. Comprehensive experiments demonstrate that GP3 consistently outperforms state-of-the-art methods on simulated benchmarks. Furthermore, GP3 transfers effectively to real-world robots without depth sensors or pre-mapped environments, requiring only minimal fine-tuning. These results highlight GP3 as a practical, sensor-agnostic solution for geometry-aware robotic manipulation.
>
---
#### [new 020] Distributed Nash Equilibrium Seeking Algorithm in Aggregative Games for Heterogeneous Multi-Robot Systems
- **分类: cs.RO**

- **简介: 该论文研究多机器人系统的纳什均衡求解问题，提出一种分布式算法，通过优化与输出控制实现异构机器人的协同决策。算法利用邻居信息，确保收敛并提升效率，经仿真和实验证明有效。**

- **链接: [http://arxiv.org/pdf/2509.15597v1](http://arxiv.org/pdf/2509.15597v1)**

> **作者:** Yi Dong; Zhongguo Li; Sarvapali D. Ramchurn; Xiaowei Huang
>
> **摘要:** This paper develops a distributed Nash Equilibrium seeking algorithm for heterogeneous multi-robot systems. The algorithm utilises distributed optimisation and output control to achieve the Nash equilibrium by leveraging information shared among neighbouring robots. Specifically, we propose a distributed optimisation algorithm that calculates the Nash equilibrium as a tailored reference for each robot and designs output control laws for heterogeneous multi-robot systems to track it in an aggregative game. We prove that our algorithm is guaranteed to converge and result in efficient outcomes. The effectiveness of our approach is demonstrated through numerical simulations and empirical testing with physical robots.
>
---
#### [new 021] Efficient Detection of Objects Near a Robot Manipulator via Miniature Time-of-Flight Sensors
- **分类: cs.RO**

- **简介: 该论文提出一种基于微型飞行时间传感器的机器人附近物体检测方法，旨在解决传感器误将机器人自身识别为障碍物的问题。通过建立机器人自身的测量模型，实现轻量级计算下的物体检测与定位，提升人机交互安全性。**

- **链接: [http://arxiv.org/pdf/2509.16122v1](http://arxiv.org/pdf/2509.16122v1)**

> **作者:** Carter Sifferman; Mohit Gupta; Michael Gleicher
>
> **摘要:** We provide a method for detecting and localizing objects near a robot arm using arm-mounted miniature time-of-flight sensors. A key challenge when using arm-mounted sensors is differentiating between the robot itself and external objects in sensor measurements. To address this challenge, we propose a computationally lightweight method which utilizes the raw time-of-flight information captured by many off-the-shelf, low-resolution time-of-flight sensor. We build an empirical model of expected sensor measurements in the presence of the robot alone, and use this model at runtime to detect objects in proximity to the robot. In addition to avoiding robot self-detections in common sensor configurations, the proposed method enables extra flexibility in sensor placement, unlocking configurations which achieve more efficient coverage of a radius around the robot arm. Our method can detect small objects near the arm and localize the position of objects along the length of a robot link to reasonable precision. We evaluate the performance of the method with respect to object type, location, and ambient light level, and identify limiting factors on performance inherent in the measurement principle. The proposed method has potential applications in collision avoidance and in facilitating safe human-robot interaction.
>
---
#### [new 022] Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion
- **分类: cs.RO**

- **简介: 该论文提出Omni-LIVO，首个紧耦合多相机LIVO系统，解决LiDAR与相机视场不匹配问题。通过跨视角直接跟踪和改进ESIKF实现多视角融合，提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.15673v1](http://arxiv.org/pdf/2509.15673v1)**

> **作者:** Yinong Cao; Xin He; Yuwei Chen; Chenyang Zhang; Chengyu Pu; Bingtao Wang; Kaile Wu; Shouzheng Zhu; Fei Han; Shijie Liu; Chunlai Li; Jianyu Wang
>
> **摘要:** Wide field-of-view (FoV) LiDAR sensors provide dense geometry across large environments, but most existing LiDAR-inertial-visual odometry (LIVO) systems rely on a single camera, leading to limited spatial coverage and degraded robustness. We present Omni-LIVO, the first tightly coupled multi-camera LIVO system that bridges the FoV mismatch between wide-angle LiDAR and conventional cameras. Omni-LIVO introduces a Cross-View direct tracking strategy that maintains photometric consistency across non-overlapping views, and extends the Error-State Iterated Kalman Filter (ESIKF) with multi-view updates and adaptive covariance weighting. The system is evaluated on public benchmarks and our custom dataset, showing improved accuracy and robustness over state-of-the-art LIVO, LIO, and visual-inertial baselines. Code and dataset will be released upon publication.
>
---
#### [new 023] Coordinated Multi-Drone Last-mile Delivery: Learning Strategies for Energy-aware and Timely Operations
- **分类: cs.RO**

- **简介: 该论文研究多无人机协同的最后一公里配送问题，旨在解决能量感知和时效性要求。通过K-means聚类、强化学习优化飞行范围及路径规划方法，提出基于多智能体深度强化学习的算法，实现高效节能的配送调度。**

- **链接: [http://arxiv.org/pdf/2509.15830v1](http://arxiv.org/pdf/2509.15830v1)**

> **作者:** Chuhao Qin; Arun Narayanan; Evangelos Pournaras
>
> **备注:** 12 pages, 8 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Drones have recently emerged as a faster, safer, and cost-efficient way for last-mile deliveries of parcels, particularly for urgent medical deliveries highlighted during the pandemic. This paper addresses a new challenge of multi-parcel delivery with a swarm of energy-aware drones, accounting for time-sensitive customer requirements. Each drone plans an optimal multi-parcel route within its battery-restricted flight range to minimize delivery delays and reduce energy consumption. The problem is tackled by decomposing it into three sub-problems: (1) optimizing depot locations and service areas using K-means clustering; (2) determining the optimal flight range for drones through reinforcement learning; and (3) planning and selecting multi-parcel delivery routes via a new optimized plan selection approach. To integrate these solutions and enhance long-term efficiency, we propose a novel algorithm leveraging actor-critic-based multi-agent deep reinforcement learning. Extensive experimentation using realistic delivery datasets demonstrate an exceptional performance of the proposed algorithm. We provide new insights into economic efficiency (minimize energy consumption), rapid operations (reduce delivery delays and overall execution time), and strategic guidance on depot deployment for practical logistics applications.
>
---
#### [new 024] High-Bandwidth Tactile-Reactive Control for Grasp Adjustment
- **分类: cs.RO**

- **简介: 该论文研究抓取调整任务，旨在解决视觉抓取系统因感知误差导致的接触不确定性问题。提出了一种纯触觉反馈的控制器，无需物体几何或精确姿态，即可提升抓取稳定性，并通过仿真和实验证明其有效性。**

- **链接: [http://arxiv.org/pdf/2509.15876v1](http://arxiv.org/pdf/2509.15876v1)**

> **作者:** Yonghyeon Lee; Tzu-Yuan Lin; Alexander Alexiev; Sangbae Kim
>
> **备注:** 8 pages; 12 figures
>
> **摘要:** Vision-only grasping systems are fundamentally constrained by calibration errors, sensor noise, and grasp pose prediction inaccuracies, leading to unavoidable contact uncertainty in the final stage of grasping. High-bandwidth tactile feedback, when paired with a well-designed tactile-reactive controller, can significantly improve robustness in the presence of perception errors. This paper contributes to controller design by proposing a purely tactile-feedback grasp-adjustment algorithm. The proposed controller requires neither prior knowledge of the object's geometry nor an accurate grasp pose, and is capable of refining a grasp even when starting from a crude, imprecise initial configuration and uncertain contact points. Through simulation studies and real-world experiments on a 15-DoF arm-hand system (featuring an 8-DoF hand) equipped with fingertip tactile sensors operating at 200 Hz, we demonstrate that our tactile-reactive grasping framework effectively improves grasp stability.
>
---
#### [new 025] Modeling Elastic-Body Dynamics of Fish Swimming Using a Variational Framework
- **分类: cs.RO**

- **简介: 该论文属于仿生机器人动力学建模任务，旨在解决柔性鱼体游泳动力学的高效建模问题。基于变分原理，构建了包含弹性形变与流固耦合的全鱼体动力学模型，并研究了运动参数对游速和能耗的影响，为软体水下机器人设计提供理论支持。**

- **链接: [http://arxiv.org/pdf/2509.16145v1](http://arxiv.org/pdf/2509.16145v1)**

> **作者:** Zhiheng Chen; Wei Wang
>
> **备注:** Under review at IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Fish-inspired aquatic robots are gaining increasing attention in research communities due to their high swimming speeds and efficient propulsion enabled by flexible bodies that generate undulatory motions. To support the design optimizations and control of such systems, accurate, interpretable, and computationally tractable modeling of the underlying swimming dynamics is indispensable. In this letter, we present a full-body dynamics model for fish swimming, rigorously derived from Hamilton's principle. The model captures the continuously distributed elasticity of a deformable fish body undergoing large deformations and incorporates fluid-structure coupling effects, enabling self-propelled motion without prescribing kinematics. A preliminary parameter study explores the influence of actuation frequency and body stiffness on swimming speed and cost of transport (COT). Simulation results indicate that swimming speed and energy efficiency exhibit opposing trends with tail-beat frequency and that both body stiffness and body length have distinct optimal values. These findings provide insights into biological swimming mechanisms and inform the design of high-performance soft robotic swimmers.
>
---
#### [new 026] GiAnt: A Bio-Inspired Hexapod for Adaptive Terrain Navigation and Object Detection
- **分类: cs.RO**

- **简介: 该论文提出GiAnt，一种仿生六足机器人，旨在解决复杂地形适应与物体识别问题。设计轻量化结构和单自由度腿部机构，并结合Arduino控制与机器学习技术，实现高效运动与81类物体检测，适用于科研与勘探任务。**

- **链接: [http://arxiv.org/pdf/2509.15264v1](http://arxiv.org/pdf/2509.15264v1)**

> **作者:** Aasfee Mosharraf Bhuiyan; Md Luban Mehda; Md. Thawhid Hasan Puspo; Jubayer Amin Pritom
>
> **摘要:** This paper presents the design, development and testing of GiAnt, an affordable hexapod which is inspired by the efficient motions of ants. The decision to model GiAnt after ants rather than other insects is rooted in ants' natural adaptability to a variety of terrains. This bio-inspired approach gives it a significant advantage in outdoor applications, offering terrain flexibility along with efficient energy use. It features a lightweight 3D-printed and laser cut structure weighing 1.75 kg with dimensions of 310 mm x 200 mm x 120 mm. Its legs have been designed with a simple Single Degree of Freedom (DOF) using a link and crank mechanism. It is great for conquering challenging terrains such as grass, rocks, and steep surfaces. Unlike traditional robots using four wheels for motion, its legged design gives superior adaptability to uneven and rough surfaces. GiAnt's control system is built on Arduino, allowing manual operation. An effective way of controlling the legs of GiAnt was achieved by gait analysis. It can move up to 8 cm of height easily with its advanced leg positioning system. Furthermore, equipped with machine learning and image processing technology, it can identify 81 different objects in a live monitoring system. It represents a significant step towards creating accessible hexapod robots for research, exploration, and surveying, offering unique advantages in adaptability and control simplicity.
>
---
#### [new 027] Measurement and Potential Field-Based Patient Modeling for Model-Mediated Tele-ultrasound
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究模型介导的远程超声任务，旨在解决通信延迟下力反馈不准确的问题。提出基于测量数据更新内部势场模型的方法，提升远程操作的透明度与力估计精度。**

- **链接: [http://arxiv.org/pdf/2509.15325v1](http://arxiv.org/pdf/2509.15325v1)**

> **作者:** Ryan S. Yeung; David G. Black; Septimiu E. Salcudean
>
> **摘要:** Teleoperated ultrasound can improve diagnostic medical imaging access for remote communities. Having accurate force feedback is important for enabling sonographers to apply the appropriate probe contact force to optimize ultrasound image quality. However, large time delays in communication make direct force feedback impractical. Prior work investigated using point cloud-based model-mediated teleoperation and internal potential field models to estimate contact forces and torques. We expand on this by introducing a method to update the internal potential field model of the patient with measured positions and forces for more transparent model-mediated tele-ultrasound. We first generate a point cloud model of the patient's surface and transmit this to the sonographer in a compact data structure. This is converted to a static voxelized volume where each voxel contains a potential field value. These values determine the forces and torques, which are rendered based on overlap between the voxelized volume and a point shell model of the ultrasound transducer. We solve for the potential field using a convex quadratic that combines the spatial Laplace operator with measured forces. This was evaluated on volunteer patients ($n=3$) by computing the accuracy of rendered forces. Results showed the addition of measured forces to the model reduced the force magnitude error by an average of 7.23 N and force vector angle error by an average of 9.37$^{\circ}$ compared to using only Laplace's equation.
>
---
#### [new 028] A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出VLAC模型，用于解决机器人真实世界强化学习中稀疏奖励和低效探索问题。基于视觉-语言-动作模型，通过大规模异构数据训练，实现任务通用的奖励生成与策略统一，结合人机协作机制显著提升成功率与样本效率。**

- **链接: [http://arxiv.org/pdf/2509.15937v1](http://arxiv.org/pdf/2509.15937v1)**

> **作者:** Shaopeng Zhai; Qi Zhang; Tianyi Zhang; Fuxian Huang; Haoran Zhang; Ming Zhou; Shengzhe Zhang; Litao Liu; Sixu Lin; Jiangmiao Pang
>
> **备注:** 26 pages,10 figures
>
> **摘要:** Robotic real-world reinforcement learning (RL) with vision-language-action (VLA) models is bottlenecked by sparse, handcrafted rewards and inefficient exploration. We introduce VLAC, a general process reward model built upon InternVL and trained on large scale heterogeneous datasets. Given pairwise observations and a language goal, it outputs dense progress delta and done signal, eliminating task-specific reward engineering, and supports one-shot in-context transfer to unseen tasks and environments. VLAC is trained on vision-language datasets to strengthen perception, dialogic and reasoning capabilities, together with robot and human trajectories data that ground action generation and progress estimation, and additionally strengthened to reject irrelevant prompts as well as detect regression or stagnation by constructing large numbers of negative and semantically mismatched samples. With prompt control, a single VLAC model alternately generating reward and action tokens, unifying critic and policy. Deployed inside an asynchronous real-world RL loop, we layer a graded human-in-the-loop protocol (offline demonstration replay, return and explore, human guided explore) that accelerates exploration and stabilizes early learning. Across four distinct real-world manipulation tasks, VLAC lifts success rates from about 30\% to about 90\% within 200 real-world interaction episodes; incorporating human-in-the-loop interventions yields a further 50% improvement in sample efficiency and achieves up to 100% final success.
>
---
#### [new 029] Miniature soft robot with magnetically reprogrammable surgical functions
- **分类: cs.RO**

- **简介: 该论文提出一种毫米级软体机器人，通过磁性重编程实现五种手术功能，并具备六自由度运动能力。旨在解决现有微型机器人功能单一、控制受限的问题，推动微创手术技术发展。**

- **链接: [http://arxiv.org/pdf/2509.15610v1](http://arxiv.org/pdf/2509.15610v1)**

> **作者:** Chelsea Shan Xian Ng; Yu Xuan Yeoh; Nicholas Yong Wei Foo; Keerthana Radhakrishnan; Guo Zhan Lum
>
> **备注:** First three listed authors are equally contributing authors. Correspondence to: gzlum@ntu.edu.sg
>
> **摘要:** Miniature robots are untethered actuators, which have significant potential to make existing minimally invasive surgery considerably safer and painless, and enable unprecedented treatments because they are much smaller and dexterous than existing surgical robots. Of the miniature robots, the magnetically actuated ones are the most functional and dexterous. However, existing magnetic miniature robots are currently impractical for surgery because they are either restricted to possessing at most two on-board functionalities or having limited five degrees-of-freedom (DOF) locomotion. Some of these actuators are also only operational under specialized environments where actuation from strong external magnets must be at very close proximity (< 4 cm away). Here we present a millimeter-scale soft robot where its magnetization profile can be reprogrammed upon command to perform five surgical functionalities: drug-dispensing, cutting through biological tissues (simulated with gelatin), gripping, storing (biological) samples and remote heating. By possessing full six-DOF motions, including the sixth-DOF rotation about its net magnetic moment, our soft robot can also roll and two-anchor crawl across challenging unstructured environments, which are impassable by its five-DOF counterparts. Because our actuating magnetic fields are relatively uniform and weak (at most 65 mT and 1.5 T/m), such fields can theoretically penetrate through biological tissues harmlessly and allow our soft robot to remain controllable within the depths of the human body. We envision that this work marks a major milestone for the advancement of soft actuators, and towards revolutionizing minimally invasive treatments with untethered miniature robots that have unprecedented functionalities.
>
---
#### [new 030] Explainable AI-Enhanced Supervisory Control for Robust Multi-Agent Robotic Systems
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文提出一种可解释AI增强的监督控制框架，用于多智能体机器人系统。旨在解决复杂动态环境下的安全、鲁棒控制问题。结合定时自动机、Lyapunov和滑模控制器，并通过可解释预测器实现透明决策优化，验证了在航天器编队和水下机器人中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.15491v1](http://arxiv.org/pdf/2509.15491v1)**

> **作者:** Reza Pirayeshshirazinezhad; Nima Fathi
>
> **摘要:** We present an explainable AI-enhanced supervisory control framework for multi-agent robotics that combines (i) a timed-automata supervisor for safe, auditable mode switching, (ii) robust continuous control (Lyapunov-based controller for large-angle maneuver; sliding-mode controller (SMC) with boundary layers for precision and disturbance rejection), and (iii) an explainable predictor that maps mission context to gains and expected performance (energy, error). Monte Carlo-driven optimization provides the training data, enabling transparent real-time trade-offs. We validated the approach in two contrasting domains, spacecraft formation flying and autonomous underwater vehicles (AUVs). Despite different environments (gravity/actuator bias vs. hydrodynamic drag/currents), both share uncertain six degrees of freedom (6-DOF) rigid-body dynamics, relative motion, and tight tracking needs, making them representative of general robotic systems. In the space mission, the supervisory logic selects parameters that meet mission criteria. In AUV leader-follower tests, the same SMC structure maintains a fixed offset under stochastic currents with bounded steady error. In spacecraft validation, the SMC controller achieved submillimeter alignment with 21.7% lower tracking error and 81.4% lower energy consumption compared to Proportional-Derivative PD controller baselines. At the same time, in AUV tests, SMC maintained bounded errors under stochastic currents. These results highlight both the portability and the interpretability of the approach for safety-critical, resource-constrained multi-agent robotics.
>
---
#### [new 031] Right-Side-Out: Learning Zero-Shot Sim-to-Real Garment Reversal
- **分类: cs.RO**

- **简介: 该论文研究衣物翻面任务，解决高度动态、严重遮挡下的操作难题。提出零样本Sim-to-Real框架Right-Side-Out，通过任务分解和高保真仿真，实现无需人工示教的自动策略训练，实测成功率高达81.3%。**

- **链接: [http://arxiv.org/pdf/2509.15953v1](http://arxiv.org/pdf/2509.15953v1)**

> **作者:** Chang Yu; Siyu Ma; Wenxin Du; Zeshun Zong; Han Xue; Wendi Chen; Cewu Lu; Yin Yang; Xuchen Han; Joseph Masterjohn; Alejandro Castro; Chenfanfu Jiang
>
> **备注:** More details and supplementary material are on the website: https://right-side-out.github.io
>
> **摘要:** Turning garments right-side out is a challenging manipulation task: it is highly dynamic, entails rapid contact changes, and is subject to severe visual occlusion. We introduce Right-Side-Out, a zero-shot sim-to-real framework that effectively solves this challenge by exploiting task structures. We decompose the task into Drag/Fling to create and stabilize an access opening, followed by Insert&Pull to invert the garment. Each step uses a depth-inferred, keypoint-parameterized bimanual primitive that sharply reduces the action space while preserving robustness. Efficient data generation is enabled by our custom-built, high-fidelity, GPU-parallel Material Point Method (MPM) simulator that models thin-shell deformation and provides robust and efficient contact handling for batched rollouts. Built on the simulator, our fully automated pipeline scales data generation by randomizing garment geometry, material parameters, and viewpoints, producing depth, masks, and per-primitive keypoint labels without any human annotations. With a single depth camera, policies trained entirely in simulation deploy zero-shot on real hardware, achieving up to 81.3% success rate. By employing task decomposition and high fidelity simulation, our framework enables tackling highly dynamic, severely occluded tasks without laborious human demonstrations.
>
---
#### [new 032] DIPP: Discriminative Impact Point Predictor for Catching Diverse In-Flight Objects
- **分类: cs.RO**

- **简介: 该论文研究四足机器人用篮子接空中物体的任务，旨在解决因气动复杂性导致的落点预测难题。作者构建了包含8000条轨迹的数据集，并提出DIPP方法，通过特征嵌入和落点预测模块提升早期预测精度，在仿真与实验证明了有效性。**

- **链接: [http://arxiv.org/pdf/2509.15254v1](http://arxiv.org/pdf/2509.15254v1)**

> **作者:** Ngoc Huy Nguyen; Kazuki Shibata; Takamitsu Matsubara
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** In this study, we address the problem of in-flight object catching using a quadruped robot with a basket. Our objective is to accurately predict the impact point, defined as the object's landing position. This task poses two key challenges: the absence of public datasets capturing diverse objects under unsteady aerodynamics, which are essential for training reliable predictors; and the difficulty of accurate early-stage impact point prediction when trajectories appear similar across objects. To overcome these issues, we construct a real-world dataset of 8,000 trajectories from 20 objects, providing a foundation for advancing in-flight object catching under complex aerodynamics. We then propose the Discriminative Impact Point Predictor (DIPP), consisting of two modules: (i) a Discriminative Feature Embedding (DFE) that separates trajectories by dynamics to enable early-stage discrimination and generalization, and (ii) an Impact Point Predictor (IPP) that estimates the impact point from these features. Two IPP variants are implemented: an Neural Acceleration Estimator (NAE)-based method that predicts trajectories and derives the impact point, and a Direct Point Estimator (DPE)-based method that directly outputs it. Experimental results show that our dataset is more diverse and complex than existing dataset, and that our method outperforms baselines on both 15 seen and 5 unseen objects. Furthermore, we show that improved early-stage prediction enhances catching success in simulation and demonstrate the effectiveness of our approach through real-world experiments. The demonstration is available at https://sites.google.com/view/robot-catching-2025.
>
---
#### [new 033] Indoor Positioning Based on Active Radar Sensing and Passive Reflectors: Reflector Placement Optimization
- **分类: cs.RO; eess.SP**

- **简介: 该论文研究基于雷达感知的室内定位系统，针对低成本高精度定位问题，提出结合被动反射器与FMCW雷达，并采用多目标粒子群算法优化反射器布局。**

- **链接: [http://arxiv.org/pdf/2509.15613v1](http://arxiv.org/pdf/2509.15613v1)**

> **作者:** Sven Hinderer; Pascal Schlachter; Zhibin Yu; Xiaofeng Wu; Bin Yang
>
> **摘要:** We extend our work on a novel indoor positioning system (IPS) for autonomous mobile robots (AMRs) based on radar sensing of local, passive radar reflectors. Through the combination of simple reflectors and a single-channel frequency modulated continuous wave (FMCW) radar, high positioning accuracy at low system cost can be achieved. Further, a multi-objective (MO) particle swarm optimization (PSO) algorithm is presented that optimizes the 2D placement of radar reflectors in complex room settings.
>
---
#### [new 034] Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories
- **分类: cs.RO**

- **简介: 该论文提出ACDC系统，通过自然语言指令驱动无人机自主完成室内航拍。任务为自主航拍生成，解决传统人工设点效率低、效果不一致的问题。工作包括：利用大模型解析指令生成路径，并结合优化与运动规划实现安全美观的航拍轨迹。**

- **链接: [http://arxiv.org/pdf/2509.16176v1](http://arxiv.org/pdf/2509.16176v1)**

> **作者:** Yifan Lin; Sophie Ziyu Liu; Ran Qi; George Z. Xue; Xinping Song; Chao Qin; Hugh H. -T. Liu
>
> **摘要:** We present Agentic Aerial Cinematography: From Dialogue Cues to Cinematic Trajectories (ACDC), an autonomous drone cinematography system driven by natural language communication between human directors and drones. The main limitation of previous drone cinematography workflows is that they require manual selection of waypoints and view angles based on predefined human intent, which is labor-intensive and yields inconsistent performance. In this paper, we propose employing large language models (LLMs) and vision foundation models (VFMs) to convert free-form natural language prompts directly into executable indoor UAV video tours. Specifically, our method comprises a vision-language retrieval pipeline for initial waypoint selection, a preference-based Bayesian optimization framework that refines poses using aesthetic feedback, and a motion planner that generates safe quadrotor trajectories. We validate ACDC through both simulation and hardware-in-the-loop experiments, demonstrating that it robustly produces professional-quality footage across diverse indoor scenes without requiring expertise in robotics or cinematography. These results highlight the potential of embodied AI agents to close the loop from open-vocabulary dialogue to real-world autonomous aerial cinematography.
>
---
#### [new 035] Online Slip Detection and Friction Coefficient Estimation for Autonomous Racing
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究自动驾驶赛车中的滑移检测与轮胎-路面摩擦系数估计问题。针对现有方法依赖模型或大量数据的不足，提出一种轻量级在线方法，仅使用IMU、LiDAR和控制指令实现实时滑移检测与摩擦系数估计，实验验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.15423v1](http://arxiv.org/pdf/2509.15423v1)**

> **作者:** Christopher Oeltjen; Carson Sobolewski; Saleh Faghfoorian; Lorant Domokos; Giancarlo Vidal; Ivan Ruchkin
>
> **备注:** Equal contribution by the first three authors
>
> **摘要:** Accurate knowledge of the tire-road friction coefficient (TRFC) is essential for vehicle safety, stability, and performance, especially in autonomous racing, where vehicles often operate at the friction limit. However, TRFC cannot be directly measured with standard sensors, and existing estimation methods either depend on vehicle or tire models with uncertain parameters or require large training datasets. In this paper, we present a lightweight approach for online slip detection and TRFC estimation. Our approach relies solely on IMU and LiDAR measurements and the control actions, without special dynamical or tire models, parameter identification, or training data. Slip events are detected in real time by comparing commanded and measured motions, and the TRFC is then estimated directly from observed accelerations under no-slip conditions. Experiments with a 1:10-scale autonomous racing car across different friction levels demonstrate that the proposed approach achieves accurate and consistent slip detections and friction coefficients, with results closely matching ground-truth measurements. These findings highlight the potential of our simple, deployable, and computationally efficient approach for real-time slip monitoring and friction coefficient estimation in autonomous driving.
>
---
#### [new 036] Sym2Real: Symbolic Dynamics with Residual Learning for Data-Efficient Adaptive Control
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出Sym2Real框架，用于数据高效的自适应控制。它结合符号回归与残差学习，在仅使用少量真实轨迹的情况下，实现四旋翼和赛车的鲁棒控制，解决了现实噪声和模型退化问题。**

- **链接: [http://arxiv.org/pdf/2509.15412v1](http://arxiv.org/pdf/2509.15412v1)**

> **作者:** Easop Lee; Samuel A. Moore; Boyuan Chen
>
> **摘要:** We present Sym2Real, a fully data-driven framework that provides a principled way to train low-level adaptive controllers in a highly data-efficient manner. Using only about 10 trajectories, we achieve robust control of both a quadrotor and a racecar in the real world, without expert knowledge or simulation tuning. Our approach achieves this data efficiency by bringing symbolic regression to real-world robotics while addressing key challenges that prevent its direct application, including noise sensitivity and model degradation that lead to unsafe control. Our key observation is that the underlying physics is often shared for a system regardless of internal or external changes. Hence, we strategically combine low-fidelity simulation data with targeted real-world residual learning. Through experimental validation on quadrotor and racecar platforms, we demonstrate consistent data-efficient adaptation across six out-of-distribution sim2sim scenarios and successful sim2real transfer across five real-world conditions. More information and videos can be found at at http://generalroboticslab.com/Sym2Real
>
---
#### [new 037] Reward Evolution with Graph-of-Thoughts: A Bi-Level Language Model Framework for Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文提出RE-GoT，一种结合LLM与VLM的双层框架，用于自动化设计强化学习奖励函数。通过图结构推理和视觉反馈迭代优化奖励，解决了传统方法依赖人工、易幻觉的问题，在多任务中显著提升了成功率。**

- **链接: [http://arxiv.org/pdf/2509.16136v1](http://arxiv.org/pdf/2509.16136v1)**

> **作者:** Changwei Yao; Xinzi Liu; Chen Li; Marios Savvides
>
> **摘要:** Designing effective reward functions remains a major challenge in reinforcement learning (RL), often requiring considerable human expertise and iterative refinement. Recent advances leverage Large Language Models (LLMs) for automated reward design, but these approaches are limited by hallucinations, reliance on human feedback, and challenges with handling complex, multi-step tasks. In this work, we introduce Reward Evolution with Graph-of-Thoughts (RE-GoT), a novel bi-level framework that enhances LLMs with structured graph-based reasoning and integrates Visual Language Models (VLMs) for automated rollout evaluation. RE-GoT first decomposes tasks into text-attributed graphs, enabling comprehensive analysis and reward function generation, and then iteratively refines rewards using visual feedback from VLMs without human intervention. Extensive experiments on 10 RoboGen and 4 ManiSkill2 tasks demonstrate that RE-GoT consistently outperforms existing LLM-based baselines. On RoboGen, our method improves average task success rates by 32.25%, with notable gains on complex multi-step tasks. On ManiSkill2, RE-GoT achieves an average success rate of 93.73% across four diverse manipulation tasks, significantly surpassing prior LLM-based approaches and even exceeding expert-designed rewards. Our results indicate that combining LLMs and VLMs with graph-of-thoughts reasoning provides a scalable and effective solution for autonomous reward evolution in RL.
>
---
#### [new 038] CoReVLA: A Dual-Stage End-to-End Autonomous Driving Framework for Long-Tail Scenarios via Collect-and-Refine
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出CoReVLA，一种面向长尾场景的端到端自动驾驶框架。通过数据收集与行为优化双阶段流程，提升模型在罕见安全关键场景中的性能，解决了传统方法在数据质量和学习效率上的不足。**

- **链接: [http://arxiv.org/pdf/2509.15968v1](http://arxiv.org/pdf/2509.15968v1)**

> **作者:** Shiyu Fang; Yiming Cui; Haoyang Liang; Chen Lv; Peng Hang; Jian Sun
>
> **摘要:** Autonomous Driving (AD) systems have made notable progress, but their performance in long-tail, safety-critical scenarios remains limited. These rare cases contribute a disproportionate number of accidents. Vision-Language Action (VLA) models have strong reasoning abilities and offer a potential solution, but their effectiveness is limited by the lack of high-quality data and inefficient learning in such conditions. To address these challenges, we propose CoReVLA, a continual learning end-to-end autonomous driving framework that improves the performance in long-tail scenarios through a dual-stage process of data Collection and behavior Refinement. First, the model is jointly fine-tuned on a mixture of open-source driving QA datasets, allowing it to acquire a foundational understanding of driving scenarios. Next, CoReVLA is deployed within the Cave Automatic Virtual Environment (CAVE) simulation platform, where driver takeover data is collected from real-time interactions. Each takeover indicates a long-tail scenario that CoReVLA fails to handle reliably. Finally, the model is refined via Direct Preference Optimization (DPO), allowing it to learn directly from human preferences and thereby avoid reward hacking caused by manually designed rewards. Extensive open-loop and closed-loop experiments demonstrate that the proposed CoReVLA model can accurately perceive driving scenarios and make appropriate decisions. On the Bench2Drive benchmark, CoReVLA achieves a Driving Score (DS) of 72.18 and a Success Rate (SR) of 50%, outperforming state-of-the-art methods by 7.96 DS and 15% SR under long-tail, safety-critical scenarios. Furthermore, case studies demonstrate the model's ability to continually improve its performance in similar failure-prone scenarios by leveraging past takeover experiences. All codea and preprocessed datasets are available at: https://github.com/FanGShiYuu/CoReVLA
>
---
#### [new 039] Compose by Focus: Scene Graph-based Atomic Skills
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究通用机器人在长时序任务中的组合泛化问题。针对视觉-运动策略在场景组合下鲁棒性差的问题，提出基于场景图的技能表示与学习框架，结合图神经网络和扩散模仿学习，并与视觉语言模型任务规划器集成，提升了复杂任务的成功率与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.16053v1](http://arxiv.org/pdf/2509.16053v1)**

> **作者:** Han Qi; Changhe Chen; Heng Yang
>
> **摘要:** A key requirement for generalist robots is compositional generalization - the ability to combine atomic skills to solve complex, long-horizon tasks. While prior work has primarily focused on synthesizing a planner that sequences pre-learned skills, robust execution of the individual skills themselves remains challenging, as visuomotor policies often fail under distribution shifts induced by scene composition. To address this, we introduce a scene graph-based representation that focuses on task-relevant objects and relations, thereby mitigating sensitivity to irrelevant variation. Building on this idea, we develop a scene-graph skill learning framework that integrates graph neural networks with diffusion-based imitation learning, and further combine "focused" scene-graph skills with a vision-language model (VLM) based task planner. Experiments in both simulation and real-world manipulation tasks demonstrate substantially higher success rates than state-of-the-art baselines, highlighting improved robustness and compositional generalization in long-horizon tasks.
>
---
#### [new 040] FlyKites: Human-centric Interactive Exploration and Assistance under Limited Communication
- **分类: cs.RO**

- **简介: 该论文提出FlyKites框架，用于多机器人系统在通信受限环境下的任务。它解决人类协助延迟问题，通过分布式探索、中继优化和人机交互执行三个模块，提升复杂场景下（如洞穴）的协作效率与响应速度。**

- **链接: [http://arxiv.org/pdf/2509.15807v1](http://arxiv.org/pdf/2509.15807v1)**

> **作者:** Yuyang Zhang; Zhuoli Tian; Jinsheng Wei; Meng Guo
>
> **摘要:** Fleets of autonomous robots have been deployed for exploration of unknown scenes for features of interest, e.g., subterranean exploration, reconnaissance, search and rescue missions. During exploration, the robots may encounter un-identified targets, blocked passages, interactive objects, temporary failure, or other unexpected events, all of which require consistent human assistance with reliable communication for a time period. This however can be particularly challenging if the communication among the robots is severely restricted to only close-range exchange via ad-hoc networks, especially in extreme environments like caves and underground tunnels. This paper presents a novel human-centric interactive exploration and assistance framework called FlyKites, for multi-robot systems under limited communication. It consists of three interleaved components: (I) the distributed exploration and intermittent communication (called the "spread mode"), where the robots collaboratively explore the environment and exchange local data among the fleet and with the operator; (II) the simultaneous optimization of the relay topology, the operator path, and the assignment of robots to relay roles (called the "relay mode"), such that all requested assistance can be provided with minimum delay; (III) the human-in-the-loop online execution, where the robots switch between different roles and interact with the operator adaptively. Extensive human-in-the-loop simulations and hardware experiments are performed over numerous challenging scenes.
>
---
#### [new 041] ORB: Operating Room Bot, Automating Operating Room Logistics through Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出ORB，一个用于医院手术室物流自动化的机器人系统。针对手术室物品递送的效率与无菌要求，设计了基于行为树的模块化架构，并结合实时目标识别和运动规划，实现了80%的取物成功率和96%的补货成功率。**

- **链接: [http://arxiv.org/pdf/2509.15600v1](http://arxiv.org/pdf/2509.15600v1)**

> **作者:** Jinkai Qiu; Yungjun Kim; Gaurav Sethia; Tanmay Agarwal; Siddharth Ghodasara; Zackory Erickson; Jeffrey Ichnowski
>
> **备注:** 7 pages, 5 figures, accepted as a regular conference paper in IEEE CASE 2025
>
> **摘要:** Efficiently delivering items to an ongoing surgery in a hospital operating room can be a matter of life or death. In modern hospital settings, delivery robots have successfully transported bulk items between rooms and floors. However, automating item-level operating room logistics presents unique challenges in perception, efficiency, and maintaining sterility. We propose the Operating Room Bot (ORB), a robot framework to automate logistics tasks in hospital operating rooms (OR). ORB leverages a robust, hierarchical behavior tree (BT) architecture to integrate diverse functionalities of object recognition, scene interpretation, and GPU-accelerated motion planning. The contributions of this paper include: (1) a modular software architecture facilitating robust mobile manipulation through behavior trees; (2) a novel real-time object recognition pipeline integrating YOLOv7, Segment Anything Model 2 (SAM2), and Grounded DINO; (3) the adaptation of the cuRobo parallelized trajectory optimization framework to real-time, collision-free mobile manipulation; and (4) empirical validation demonstrating an 80% success rate in OR supply retrieval and a 96% success rate in restocking operations. These contributions establish ORB as a reliable and adaptable system for autonomous OR logistics.
>
---
#### [new 042] Bench-RNR: Dataset for Benchmarking Repetitive and Non-repetitive Scanning LiDAR for Infrastructure-based Vehicle Localization
- **分类: cs.RO; eess.SP**

- **简介: 该论文针对基础设施车辆定位任务，研究重复与非重复扫描LiDAR的性能差异。为解决非重复扫描LiDAR在该领域应用不足的问题，作者构建了一个包含5445帧点云数据的公开数据集Bench-RNR，并建立基准实验进行对比分析。**

- **链接: [http://arxiv.org/pdf/2509.15583v1](http://arxiv.org/pdf/2509.15583v1)**

> **作者:** Runxin Zhao; Chunxiang Wang; Hanyang Zhuang; Ming Yang
>
> **摘要:** Vehicle localization using roadside LiDARs can provide centimeter-level accuracy for cloud-controlled vehicles while simultaneously serving multiple vehicles, enhanc-ing safety and efficiency. While most existing studies rely on repetitive scanning LiDARs, non-repetitive scanning LiDAR offers advantages such as eliminating blind zones and being more cost-effective. However, its application in roadside perception and localization remains limited. To address this, we present a dataset for infrastructure-based vehicle localization, with data collected from both repetitive and non-repetitive scanning LiDARs, in order to benchmark the performance of different LiDAR scanning patterns. The dataset contains 5,445 frames of point clouds across eight vehicle trajectory sequences, with diverse trajectory types. Our experiments establish base-lines for infrastructure-based vehicle localization and compare the performance of these methods using both non-repetitive and repetitive scanning LiDARs. This work offers valuable insights for selecting the most suitable LiDAR scanning pattern for infrastruc-ture-based vehicle localization. Our dataset is a signifi-cant contribution to the scientific community, supporting advancements in infrastructure-based perception and vehicle localization. The dataset and source code are publicly available at: https://github.com/sjtu-cyberc3/BenchRNR.
>
---
#### [new 043] Embodied Arena: A Comprehensive, Unified, and Evolving Evaluation Platform for Embodied AI
- **分类: cs.RO**

- **简介: 该论文提出Embodied Arena，一个综合、统一的具身AI评估平台，旨在解决其发展滞后问题。通过构建能力体系、标准化评估系统及自动化数据生成方法，推动具身AI研究目标明确与模型性能提升。**

- **链接: [http://arxiv.org/pdf/2509.15273v1](http://arxiv.org/pdf/2509.15273v1)**

> **作者:** Fei Ni; Min Zhang; Pengyi Li; Yifu Yuan; Lingfeng Zhang; Yuecheng Liu; Peilong Han; Longxin Kou; Shaojin Ma; Jinbin Qiao; David Gamaliel Arcos Bravo; Yuening Wang; Xiao Hu; Zhanguang Zhang; Xianze Yao; Yutong Li; Zhao Zhang; Ying Wen; Ying-Cong Chen; Xiaodan Liang; Liang Lin; Bin He; Haitham Bou-Ammar; He Wang; Huazhe Xu; Jiankang Deng; Shan Luo; Shuqiang Jiang; Wei Pan; Yang Gao; Stefanos Zafeiriou; Jan Peters; Yuzheng Zhuang; Yingxue Zhang; Yan Zheng; Hongyao Tang; Jianye Hao
>
> **备注:** 32 pages, 5 figures, Embodied Arena Technical Report
>
> **摘要:** Embodied AI development significantly lags behind large foundation models due to three critical challenges: (1) lack of systematic understanding of core capabilities needed for Embodied AI, making research lack clear objectives; (2) absence of unified and standardized evaluation systems, rendering cross-benchmark evaluation infeasible; and (3) underdeveloped automated and scalable acquisition methods for embodied data, creating critical bottlenecks for model scaling. To address these obstacles, we present Embodied Arena, a comprehensive, unified, and evolving evaluation platform for Embodied AI. Our platform establishes a systematic embodied capability taxonomy spanning three levels (perception, reasoning, task execution), seven core capabilities, and 25 fine-grained dimensions, enabling unified evaluation with systematic research objectives. We introduce a standardized evaluation system built upon unified infrastructure supporting flexible integration of 22 diverse benchmarks across three domains (2D/3D Embodied Q&A, Navigation, Task Planning) and 30+ advanced models from 20+ worldwide institutes. Additionally, we develop a novel LLM-driven automated generation pipeline ensuring scalable embodied evaluation data with continuous evolution for diversity and comprehensiveness. Embodied Arena publishes three real-time leaderboards (Embodied Q&A, Navigation, Task Planning) with dual perspectives (benchmark view and capability view), providing comprehensive overviews of advanced model capabilities. Especially, we present nine findings summarized from the evaluation results on the leaderboards of Embodied Arena. This helps to establish clear research veins and pinpoint critical research problems, thereby driving forward progress in the field of Embodied AI.
>
---
#### [new 044] Towards Sharper Object Boundaries in Self-Supervised Depth Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对单目深度估计任务，旨在解决现有方法在物体边界处深度模糊的问题。提出了一种基于混合分布的自监督方法，通过不确定性建模提升边界锐度，实验表明显著提升了KITTI和VKITTIv2数据集上的深度边界清晰度和点云质量。**

- **链接: [http://arxiv.org/pdf/2509.15987v1](http://arxiv.org/pdf/2509.15987v1)**

> **作者:** Aurélien Cecille; Stefan Duffner; Franck Davoine; Rémi Agier; Thibault Neveu
>
> **备注:** BMVC 2025 Oral, 10 pages, 6 figures
>
> **摘要:** Accurate monocular depth estimation is crucial for 3D scene understanding, but existing methods often blur depth at object boundaries, introducing spurious intermediate 3D points. While achieving sharp edges usually requires very fine-grained supervision, our method produces crisp depth discontinuities using only self-supervision. Specifically, we model per-pixel depth as a mixture distribution, capturing multiple plausible depths and shifting uncertainty from direct regression to the mixture weights. This formulation integrates seamlessly into existing pipelines via variance-aware loss functions and uncertainty propagation. Extensive evaluations on KITTI and VKITTIv2 show that our method achieves up to 35% higher boundary sharpness and improves point cloud quality compared to state-of-the-art baselines.
>
---
#### [new 045] A Nascent Taxonomy of Machine Learning in Intelligent Robotic Process Automation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于文献综述任务，旨在解决传统RPA在复杂任务中的局限性问题。通过研究机器学习与RPA的结合，构建了一个包含两个元特征和八个维度的智能RPA分类体系。**

- **链接: [http://arxiv.org/pdf/2509.15730v1](http://arxiv.org/pdf/2509.15730v1)**

> **作者:** Lukas Laakmann; Seyyid A. Ciftci; Christian Janiesch
>
> **摘要:** Robotic process automation (RPA) is a lightweight approach to automating business processes using software robots that emulate user actions at the graphical user interface level. While RPA has gained popularity for its cost-effective and timely automation of rule-based, well-structured tasks, its symbolic nature has inherent limitations when approaching more complex tasks currently performed by human agents. Machine learning concepts enabling intelligent RPA provide an opportunity to broaden the range of automatable tasks. In this paper, we conduct a literature review to explore the connections between RPA and machine learning and organize the joint concept intelligent RPA into a taxonomy. Our taxonomy comprises the two meta-characteristics RPA-ML integration and RPA-ML interaction. Together, they comprise eight dimensions: architecture and ecosystem, capabilities, data basis, intelligence level, and technical depth of integration as well as deployment environment, lifecycle phase, and user-robot relation.
>
---
#### [new 046] Hierarchical Reinforcement Learning with Low-Level MPC for Multi-Agent Control
- **分类: eess.SY; cs.AI; cs.RO; cs.SY; math.OC**

- **简介: 该论文针对多智能体控制中的安全与协调问题，提出一种结合强化学习（RL）与模型预测控制（MPC）的分层框架。高层RL负责决策，低层MPC确保动态可行与安全，提升了样本效率与可靠性。**

- **链接: [http://arxiv.org/pdf/2509.15799v1](http://arxiv.org/pdf/2509.15799v1)**

> **作者:** Max Studt; Georg Schildbach
>
> **摘要:** Achieving safe and coordinated behavior in dynamic, constraint-rich environments remains a major challenge for learning-based control. Pure end-to-end learning often suffers from poor sample efficiency and limited reliability, while model-based methods depend on predefined references and struggle to generalize. We propose a hierarchical framework that combines tactical decision-making via reinforcement learning (RL) with low-level execution through Model Predictive Control (MPC). For the case of multi-agent systems this means that high-level policies select abstract targets from structured regions of interest (ROIs), while MPC ensures dynamically feasible and safe motion. Tested on a predator-prey benchmark, our approach outperforms end-to-end and shielding-based RL baselines in terms of reward, safety, and consistency, underscoring the benefits of combining structured learning with model-based control.
>
---
#### [new 047] A CARLA-based Simulation of Electrically Driven Forklifts
- **分类: cs.CE; cs.RO**

- **简介: 该论文属于物流仿真领域，旨在模拟电动叉车在仓储环境中的运行。工作包括基于CARLA构建3D仓库场景、实现叉车路径规划与任务调度，并通过回放真实数据和电池模型分析交通密度及充电站布局优化问题。**

- **链接: [http://arxiv.org/pdf/2509.15909v1](http://arxiv.org/pdf/2509.15909v1)**

> **作者:** David Claus; Christiane Thielemann; Hans-Georg Stark
>
> **摘要:** This paper presents the simulation of the operation of an electric forklift fleet within an intralogistics scenario. For this purpose, the open source simulation tool CARLA is used; according to our knowledge this is a novel approach in the context of logistics simulation. First, CARLA is used to generate and visualize a realistic 3D outdoor warehouse scenario, incorporating a number of randomly moving forklifts. In a next step, intralogistics transport tasks, such as pick-and-place, are simulated for the forklift fleet, including shortest-path finding. Furthermore, the capability to play back localization data, previously recorded from a ''real'' forklift fleet, is demonstrated.This play back is done in the original recreated environment, thereby enabling the visualization of the forklifts movements. Finally, the energy consumption of the forklift trucks is simulated by integrating a physical battery model that generates the state of charge (SOC) of each truck as a function of load and activity. To demonstrate the wide range of possible applications for the CARLA simulation platform, we describe two use cases. The first deals with the problem of detecting regions with critically high traffic densities, the second with optimal placement of charging stations for the forklift trucks. Both use cases are calculated for an exemplary warehouse model.
>
---
#### [new 048] Exploring multimodal implicit behavior learning for vehicle navigation in simulated cities
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文针对车辆导航中的多模态决策问题，提出数据增强的隐式行为克隆（DA-IBC），利用能量模型学习专家动作分布。实验表明其在CARLA模拟器中优于标准方法。**

- **链接: [http://arxiv.org/pdf/2509.15400v1](http://arxiv.org/pdf/2509.15400v1)**

> **作者:** Eric Aislan Antonelo; Gustavo Claudio Karl Couto; Christian Möller
>
> **备注:** ENIAC conference
>
> **摘要:** Standard Behavior Cloning (BC) fails to learn multimodal driving decisions, where multiple valid actions exist for the same scenario. We explore Implicit Behavioral Cloning (IBC) with Energy-Based Models (EBMs) to better capture this multimodality. We propose Data-Augmented IBC (DA-IBC), which improves learning by perturbing expert actions to form the counterexamples of IBC training and using better initialization for derivative-free inference. Experiments in the CARLA simulator with Bird's-Eye View inputs demonstrate that DA-IBC outperforms standard IBC in urban driving tasks designed to evaluate multimodal behavior learning in a test environment. The learned energy landscapes are able to represent multimodal action distributions, which BC fails to achieve.
>
---
#### [new 049] Uncertainty-Based Smooth Policy Regularisation for Reinforcement Learning with Few Demonstrations
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文研究强化学习中稀疏奖励下的策略优化任务，旨在解决何时模仿少量示教数据的问题。提出SPReD框架，通过建模Q值分布与不确定性，实现连续、自适应的策略正则化，提升学习效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.15981v1](http://arxiv.org/pdf/2509.15981v1)**

> **作者:** Yujie Zhu; Charles A. Hepburn; Matthew Thorpe; Giovanni Montana
>
> **摘要:** In reinforcement learning with sparse rewards, demonstrations can accelerate learning, but determining when to imitate them remains challenging. We propose Smooth Policy Regularisation from Demonstrations (SPReD), a framework that addresses the fundamental question: when should an agent imitate a demonstration versus follow its own policy? SPReD uses ensemble methods to explicitly model Q-value distributions for both demonstration and policy actions, quantifying uncertainty for comparisons. We develop two complementary uncertainty-aware methods: a probabilistic approach estimating the likelihood of demonstration superiority, and an advantage-based approach scaling imitation by statistical significance. Unlike prevailing methods (e.g. Q-filter) that make binary imitation decisions, SPReD applies continuous, uncertainty-proportional regularisation weights, reducing gradient variance during training. Despite its computational simplicity, SPReD achieves remarkable gains in experiments across eight robotics tasks, outperforming existing approaches by up to a factor of 14 in complex tasks while maintaining robustness to demonstration quality and quantity. Our code is available at https://github.com/YujieZhu7/SPReD.
>
---
#### [new 050] How Good are Foundation Models in Step-by-Step Embodied Reasoning?
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究基础模型在具身推理任务中的表现，提出FoMER基准测试，评估LMMs在复杂现实环境中的分步决策能力。工作包括构建多任务数据集、设计评估框架，并分析当前模型的优劣，推动机器人智能发展。**

- **链接: [http://arxiv.org/pdf/2509.15293v1](http://arxiv.org/pdf/2509.15293v1)**

> **作者:** Dinura Dissanayake; Ahmed Heakl; Omkar Thawakar; Noor Ahsan; Ritesh Thawkar; Ketan More; Jean Lahoud; Rao Anwer; Hisham Cholakkal; Ivan Laptev; Fahad Shahbaz Khan; Salman Khan
>
> **摘要:** Embodied agents operating in the physical world must make decisions that are not only effective but also safe, spatially coherent, and grounded in context. While recent advances in large multimodal models (LMMs) have shown promising capabilities in visual understanding and language generation, their ability to perform structured reasoning for real-world embodied tasks remains underexplored. In this work, we aim to understand how well foundation models can perform step-by-step reasoning in embodied environments. To this end, we propose the Foundation Model Embodied Reasoning (FoMER) benchmark, designed to evaluate the reasoning capabilities of LMMs in complex embodied decision-making scenarios. Our benchmark spans a diverse set of tasks that require agents to interpret multimodal observations, reason about physical constraints and safety, and generate valid next actions in natural language. We present (i) a large-scale, curated suite of embodied reasoning tasks, (ii) a novel evaluation framework that disentangles perceptual grounding from action reasoning, and (iii) empirical analysis of several leading LMMs under this setting. Our benchmark includes over 1.1k samples with detailed step-by-step reasoning across 10 tasks and 8 embodiments, covering three different robot types. Our results highlight both the potential and current limitations of LMMs in embodied reasoning, pointing towards key challenges and opportunities for future research in robot intelligence. Our data and code will be made publicly available.
>
---
#### [new 051] SAMPO:Scale-wise Autoregression with Motion PrOmpt for generative world models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SAMPO，用于生成世界模型，解决视觉一致性差、解码效率低和运动建模不足的问题。结合自回归与因果建模，引入多尺度编码器和轨迹感知运动提示模块，提升动态场景预测质量与效率。**

- **链接: [http://arxiv.org/pdf/2509.15536v1](http://arxiv.org/pdf/2509.15536v1)**

> **作者:** Sen Wang; Jingyi Tian; Le Wang; Zhimin Liao; Jiayi Li; Huaiyi Dong; Kun Xia; Sanping Zhou; Wei Tang; Hua Gang
>
> **备注:** 22 pages,15 figures
>
> **摘要:** World models allow agents to simulate the consequences of actions in imagined environments for planning, control, and long-horizon decision-making. However, existing autoregressive world models struggle with visually coherent predictions due to disrupted spatial structure, inefficient decoding, and inadequate motion modeling. In response, we propose \textbf{S}cale-wise \textbf{A}utoregression with \textbf{M}otion \textbf{P}r\textbf{O}mpt (\textbf{SAMPO}), a hybrid framework that combines visual autoregressive modeling for intra-frame generation with causal modeling for next-frame generation. Specifically, SAMPO integrates temporal causal decoding with bidirectional spatial attention, which preserves spatial locality and supports parallel decoding within each scale. This design significantly enhances both temporal consistency and rollout efficiency. To further improve dynamic scene understanding, we devise an asymmetric multi-scale tokenizer that preserves spatial details in observed frames and extracts compact dynamic representations for future frames, optimizing both memory usage and model performance. Additionally, we introduce a trajectory-aware motion prompt module that injects spatiotemporal cues about object and robot trajectories, focusing attention on dynamic regions and improving temporal consistency and physical realism. Extensive experiments show that SAMPO achieves competitive performance in action-conditioned video prediction and model-based control, improving generation quality with 4.4$\times$ faster inference. We also evaluate SAMPO's zero-shot generalization and scaling behavior, demonstrating its ability to generalize to unseen tasks and benefit from larger model sizes.
>
---
#### [new 052] KoopCast: Trajectory Forecasting via Koopman Operators
- **分类: cs.LG; cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出KoopCast，用于动态环境中的轨迹预测任务。针对非线性运动建模问题，利用Koopman算子理论，设计两阶段模型：目标估计与轨迹优化，实现高精度、可解释且低延迟的轨迹预测。**

- **链接: [http://arxiv.org/pdf/2509.15513v1](http://arxiv.org/pdf/2509.15513v1)**

> **作者:** Jungjin Lee; Jaeuk Shin; Gihwan Kim; Joonho Han; Insoon Yang
>
> **摘要:** We present KoopCast, a lightweight yet efficient model for trajectory forecasting in general dynamic environments. Our approach leverages Koopman operator theory, which enables a linear representation of nonlinear dynamics by lifting trajectories into a higher-dimensional space. The framework follows a two-stage design: first, a probabilistic neural goal estimator predicts plausible long-term targets, specifying where to go; second, a Koopman operator-based refinement module incorporates intention and history into a nonlinear feature space, enabling linear prediction that dictates how to go. This dual structure not only ensures strong predictive accuracy but also inherits the favorable properties of linear operators while faithfully capturing nonlinear dynamics. As a result, our model offers three key advantages: (i) competitive accuracy, (ii) interpretability grounded in Koopman spectral theory, and (iii) low-latency deployment. We validate these benefits on ETH/UCY, the Waymo Open Motion Dataset, and nuScenes, which feature rich multi-agent interactions and map-constrained nonlinear motion. Across benchmarks, KoopCast consistently delivers high predictive accuracy together with mode-level interpretability and practical efficiency.
>
---
#### [new 053] CoPAD : Multi-source Trajectory Fusion and Cooperative Trajectory Prediction with Anchor-oriented Decoder in V2X Scenarios
- **分类: cs.CV; cs.MA; cs.RO**

- **简介: 该论文提出CoPAD，用于V2X场景下的协同轨迹预测任务。针对单车感知不稳定的问题，设计融合模块与注意力机制，结合多源轨迹数据，提升预测的完整性和准确性。**

- **链接: [http://arxiv.org/pdf/2509.15984v1](http://arxiv.org/pdf/2509.15984v1)**

> **作者:** Kangyu Wu; Jiaqi Qiao; Ya Zhang
>
> **备注:** 7 pages, 4 pages, IROS2025
>
> **摘要:** Recently, data-driven trajectory prediction methods have achieved remarkable results, significantly advancing the development of autonomous driving. However, the instability of single-vehicle perception introduces certain limitations to trajectory prediction. In this paper, a novel lightweight framework for cooperative trajectory prediction, CoPAD, is proposed. This framework incorporates a fusion module based on the Hungarian algorithm and Kalman filtering, along with the Past Time Attention (PTA) module, mode attention module and anchor-oriented decoder (AoD). It effectively performs early fusion on multi-source trajectory data from vehicles and road infrastructure, enabling the trajectories with high completeness and accuracy. The PTA module can efficiently capture potential interaction information among historical trajectories, and the mode attention module is proposed to enrich the diversity of predictions. Additionally, the decoder based on sparse anchors is designed to generate the final complete trajectories. Extensive experiments show that CoPAD achieves the state-of-the-art performance on the DAIR-V2X-Seq dataset, validating the effectiveness of the model in cooperative trajectory prediction in V2X scenarios.
>
---
#### [new 054] All-Electric Heavy-Duty Robotic Manipulator: Actuator Configuration Optimization and Sensorless Control
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文研究全电动重型机械臂的执行器配置优化与无传感器控制，建立了动力学模型，提出基于NSGA-II和深度神经网络的优化方法，并设计了无传感控制框架。**

- **链接: [http://arxiv.org/pdf/2509.15778v1](http://arxiv.org/pdf/2509.15778v1)**

> **作者:** Mohammad Bahari; Amir Hossein Barjini; Pauli Mustalahti; Jouni Mattila
>
> **摘要:** This paper presents a unified framework that integrates modeling, optimization, and sensorless control of an all-electric heavy-duty robotic manipulator (HDRM) driven by electromechanical linear actuators (EMLAs). An EMLA model is formulated to capture motor electromechanics and direction-dependent transmission efficiencies, while a mathematical model of the HDRM, incorporating both kinematics and dynamics, is established to generate joint-space motion profiles for prescribed TCP trajectories. A safety-ensured trajectory generator, tailored to this model, maps Cartesian goals to joint space while enforcing joint-limit and velocity margins. Based on the resulting force and velocity demands, a multi-objective Non-dominated Sorting Genetic Algorithm II (NSGA-II) is employed to select the optimal EMLA configuration. To accelerate this optimization, a deep neural network, trained with EMLA parameters, is embedded in the optimization process to predict steady-state actuator efficiency from trajectory profiles. For the chosen EMLA design, a physics-informed Kriging surrogate, anchored to the analytic model and refined with experimental data, learns residuals of EMLA outputs to support force and velocity sensorless control. The actuator model is further embedded in a hierarchical virtual decomposition control (VDC) framework that outputs voltage commands. Experimental validation on a one-degree-of-freedom EMLA testbed confirms accurate trajectory tracking and effective sensorless control under varying loads.
>
---
## 更新

#### [replaced 001] Towards Interactive and Learnable Cooperative Driving Automation: a Large Language Model-Driven Decision-Making Framework
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.12812v3](http://arxiv.org/pdf/2409.12812v3)**

> **作者:** Shiyu Fang; Jiaqi Liu; Mingyu Ding; Yiming Cui; Chen Lv; Peng Hang; Jian Sun
>
> **摘要:** At present, Connected Autonomous Vehicles (CAVs) have begun to open road testing around the world, but their safety and efficiency performance in complex scenarios is still not satisfactory. Cooperative driving leverages the connectivity ability of CAVs to achieve synergies greater than the sum of their parts, making it a promising approach to improving CAV performance in complex scenarios. However, the lack of interaction and continuous learning ability limits current cooperative driving to single-scenario applications and specific Cooperative Driving Automation (CDA). To address these challenges, this paper proposes CoDrivingLLM, an interactive and learnable LLM-driven cooperative driving framework, to achieve all-scenario and all-CDA. First, since Large Language Models(LLMs) are not adept at handling mathematical calculations, an environment module is introduced to update vehicle positions based on semantic decisions, thus avoiding potential errors from direct LLM control of vehicle positions. Second, based on the four levels of CDA defined by the SAE J3216 standard, we propose a Chain-of-Thought (COT) based reasoning module that includes state perception, intent sharing, negotiation, and decision-making, enhancing the stability of LLMs in multi-step reasoning tasks. Centralized conflict resolution is then managed through a conflict coordinator in the reasoning process. Finally, by introducing a memory module and employing retrieval-augmented generation, CAVs are endowed with the ability to learn from their past experiences. We validate the proposed CoDrivingLLM through ablation experiments on the negotiation module, reasoning with different shots experience, and comparison with other cooperative driving methods.
>
---
#### [replaced 002] Online Learning of Deceptive Policies under Intermittent Observation
- **分类: cs.RO; cs.MA; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.14453v2](http://arxiv.org/pdf/2509.14453v2)**

> **作者:** Gokul Puthumanaillam; Ram Padmanabhan; Jose Fuentes; Nicole Cruz; Paulo Padrao; Ruben Hernandez; Hao Jiang; William Schafer; Leonardo Bobadilla; Melkior Ornik
>
> **摘要:** In supervisory control settings, autonomous systems are not monitored continuously. Instead, monitoring often occurs at sporadic intervals within known bounds. We study the problem of deception, where an agent pursues a private objective while remaining plausibly compliant with a supervisor's reference policy when observations occur. Motivated by the behavior of real, human supervisors, we situate the problem within Theory of Mind: the representation of what an observer believes and expects to see. We show that Theory of Mind can be repurposed to steer online reinforcement learning (RL) toward such deceptive behavior. We model the supervisor's expectations and distill from them a single, calibrated scalar -- the expected evidence of deviation if an observation were to happen now. This scalar combines how unlike the reference and current action distributions appear, with the agent's belief that an observation is imminent. Injected as a state-dependent weight into a KL-regularized policy improvement step within an online RL loop, this scalar informs a closed-form update that smoothly trades off self-interest and compliance, thus sidestepping hand-crafted or heuristic policies. In real-world, real-time hardware experiments on marine (ASV) and aerial (UAV) navigation, our ToM-guided RL runs online, achieves high return and success with observed-trace evidence calibrated to the supervisor's expectations.
>
---
#### [replaced 003] MapAnything: Universal Feed-Forward Metric 3D Reconstruction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.13414v2](http://arxiv.org/pdf/2509.13414v2)**

> **作者:** Nikhil Keetha; Norman Müller; Johannes Schönberger; Lorenzo Porzi; Yuchen Zhang; Tobias Fischer; Arno Knapitsch; Duncan Zauss; Ethan Weber; Nelson Antunes; Jonathon Luiten; Manuel Lopez-Antequera; Samuel Rota Bulò; Christian Richardt; Deva Ramanan; Sebastian Scherer; Peter Kontschieder
>
> **备注:** Project Page: https://map-anything.github.io/
>
> **摘要:** We introduce MapAnything, a unified transformer-based feed-forward model that ingests one or more images along with optional geometric inputs such as camera intrinsics, poses, depth, or partial reconstructions, and then directly regresses the metric 3D scene geometry and cameras. MapAnything leverages a factored representation of multi-view scene geometry, i.e., a collection of depth maps, local ray maps, camera poses, and a metric scale factor that effectively upgrades local reconstructions into a globally consistent metric frame. Standardizing the supervision and training across diverse datasets, along with flexible input augmentation, enables MapAnything to address a broad range of 3D vision tasks in a single feed-forward pass, including uncalibrated structure-from-motion, calibrated multi-view stereo, monocular depth estimation, camera localization, depth completion, and more. We provide extensive experimental analyses and model ablations demonstrating that MapAnything outperforms or matches specialist feed-forward models while offering more efficient joint training behavior, thus paving the way toward a universal 3D reconstruction backbone.
>
---
#### [replaced 004] StageACT: Stage-Conditioned Imitation for Robust Humanoid Door Opening
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.13200v2](http://arxiv.org/pdf/2509.13200v2)**

> **作者:** Moonyoung Lee; Dong Ki Kim; Jai Krishna Bandi; Max Smith; Aileen Liao; Ali-akbar Agha-mohammadi; Shayegan Omidshafiei
>
> **备注:** 7 pages
>
> **摘要:** Humanoid robots promise to operate in everyday human environments without requiring modifications to the surroundings. Among the many skills needed, opening doors is essential, as doors are the most common gateways in built spaces and often limit where a robot can go. Door opening, however, poses unique challenges as it is a long-horizon task under partial observability, such as reasoning about the door's unobservable latch state that dictates whether the robot should rotate the handle or push the door. This ambiguity makes standard behavior cloning prone to mode collapse, yielding blended or out-of-sequence actions. We introduce StageACT, a stage-conditioned imitation learning framework that augments low-level policies with task-stage inputs. This effective addition increases robustness to partial observability, leading to higher success rates and shorter completion times. On a humanoid operating in a real-world office environment, StageACT achieves a 55% success rate on previously unseen doors, more than doubling the best baseline. Moreover, our method supports intentional behavior guidance through stage prompting, enabling recovery behaviors. These results highlight stage conditioning as a lightweight yet powerful mechanism for long-horizon humanoid loco-manipulation.
>
---
#### [replaced 005] Set Phasers to Stun: Beaming Power and Control to Mobile Robots with Laser Light
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17865v2](http://arxiv.org/pdf/2504.17865v2)**

> **作者:** Charles J. Carver; Hadleigh Schwartz; Toma Itagaki; Zachary Englhardt; Kechen Liu; Megan Graciela Nauli Manik; Chun-Cheng Chang; Vikram Iyer; Brian Plancher; Xia Zhou
>
> **备注:** 8 pages, 7 figures, accepted to IROS 2025
>
> **摘要:** We present Phaser, a flexible system that directs narrow-beam laser light to moving robots for concurrent wireless power delivery and communication. We design a semi-automatic calibration procedure to enable fusion of stereo-vision-based 3D robot tracking with high-power beam steering, and a low-power optical communication scheme that reuses the laser light as a data channel. We fabricate a Phaser prototype using off-the-shelf hardware and evaluate its performance with battery-free autonomous robots. Phaser delivers optical power densities of over 110 mW/cm$^2$ and error-free data to mobile robots at multi-meter ranges, with on-board decoding drawing 0.3 mA ($97\%$ less current than Bluetooth Low Energy). We demonstrate Phaser fully powering gram-scale battery-free robots to nearly 2x higher speeds than prior work while simultaneously controlling them to navigate around obstacles and along paths. Code, an open-source design guide, and a demonstration video of Phaser is available at https://mobilex.cs.columbia.edu/phaser.
>
---
#### [replaced 006] Model-Free and Real-Time Unicycle-Based Source Seeking with Differential Wheeled Robotic Experiments
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2501.02184v5](http://arxiv.org/pdf/2501.02184v5)**

> **作者:** Ahmed A. Elgohary; Sameh A. Eisa; Shivam Bajpai
>
> **摘要:** Many autonomous robots aimed at source-seeking are studied, and their controls designed, using unicycle modeling and formulation. This is true not only for model-based controllers, but also for model-free, real-time control methods such as extremum seeking control (ESC). In this paper, we propose a unicycle-based ESC design applicable to differential wheeled robots that: (1) is very simple design, based on one simple control-affine law, and without state integrators; (2) attenuates oscillations known to persist in ESC designs (i.e., fully stop at the source); and (3) operates in a model-free, real-time setting, tolerating environmental/sensor noise. We provide simulation and real-world robotic experimental results for fixed and moving light source seeking by a differential wheeled robot using our proposed design. Results indicate clear advantages of our proposed design when compared to the literature, including attenuation of undesired oscillations, improved convergence speed, and better handling of noise.
>
---
#### [replaced 007] Dynamic Neural Curiosity Enhances Learning Flexibility for Autonomous Goal Discovery
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.00152v2](http://arxiv.org/pdf/2412.00152v2)**

> **作者:** Quentin Houbre; Roel Pieters
>
> **摘要:** The autonomous learning of new goals in robotics remains a complex issue to address. Here, we propose a model where curiosity influence learning flexibility. To do so, this paper proposes to root curiosity and attention together by taking inspiration from the Locus Coeruleus-Norepinephrine system along with various cognitive processes such as cognitive persistence and visual habituation. We apply our approach by experimenting with a simulated robotic arm on a set of objects with varying difficulty. The robot first discovers new goals via bottom-up attention through motor babbling with an inhibition of return mechanism, then engage to the learning of goals due to neural activity arising within the curiosity mechanism. The architecture is modelled with dynamic neural fields and the learning of goals such as pushing the objects in diverse directions is supported by the use of forward and inverse models implemented by multi-layer perceptrons. The adoption of dynamic neural fields to model curiosity, habituation and persistence allows the robot to demonstrate various learning trajectories depending on the object. In addition, the approach exhibits interesting properties regarding the learning of similar goals as well as the continuous switch between exploration and exploitation.
>
---
#### [replaced 008] Runtime Learning of Quadruped Robots in Wild Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.04794v2](http://arxiv.org/pdf/2503.04794v2)**

> **作者:** Yihao Cai; Yanbing Mao; Lui Sha; Hongpeng Cao; Marco Caccamo
>
> **摘要:** This paper presents a runtime learning framework for quadruped robots, enabling them to learn and adapt safely in dynamic wild environments. The framework integrates sensing, navigation, and control, forming a closed-loop system for the robot. The core novelty of this framework lies in two interactive and complementary components within the control module: the high-performance (HP)-Student and the high-assurance (HA)-Teacher. HP-Student is a deep reinforcement learning (DRL) agent that engages in self-learning and teaching-to-learn to develop a safe and high-performance action policy. HA-Teacher is a simplified yet verifiable physics-model-based controller, with the role of teaching HP-Student about safety while providing a backup for the robot's safe locomotion. HA-Teacher is innovative due to its real-time physics model, real-time action policy, and real-time control goals, all tailored to respond effectively to real-time wild environments, ensuring safety. The framework also includes a coordinator who effectively manages the interaction between HP-Student and HA-Teacher. Experiments involving a Unitree Go2 robot in Nvidia Isaac Gym and comparisons with state-of-the-art safe DRLs demonstrate the effectiveness of the proposed runtime learning framework.
>
---
#### [replaced 009] FlightDiffusion: Revolutionising Autonomous Drone Training with Diffusion Models Generating FPV Video
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.14082v2](http://arxiv.org/pdf/2509.14082v2)**

> **作者:** Valerii Serpiva; Artem Lykov; Faryal Batool; Vladislav Kozlovskiy; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **备注:** Submitted to conference
>
> **摘要:** We present FlightDiffusion, a diffusion-model-based framework for training autonomous drones from first-person view (FPV) video. Our model generates realistic video sequences from a single frame, enriched with corresponding action spaces to enable reasoning-driven navigation in dynamic environments. Beyond direct policy learning, FlightDiffusion leverages its generative capabilities to synthesize diverse FPV trajectories and state-action pairs, facilitating the creation of large-scale training datasets without the high cost of real-world data collection. Our evaluation demonstrates that the generated trajectories are physically plausible and executable, with a mean position error of 0.25 m (RMSE 0.28 m) and a mean orientation error of 0.19 rad (RMSE 0.24 rad). This approach enables improved policy learning and dataset scalability, leading to superior performance in downstream navigation tasks. Results in simulated environments highlight enhanced robustness, smoother trajectory planning, and adaptability to unseen conditions. An ANOVA revealed no statistically significant difference between performance in simulation and reality (F(1, 16) = 0.394, p = 0.541), with success rates of M = 0.628 (SD = 0.162) and M = 0.617 (SD = 0.177), respectively, indicating strong sim-to-real transfer. The generated datasets provide a valuable resource for future UAV research. This work introduces diffusion-based reasoning as a promising paradigm for unifying navigation, action generation, and data synthesis in aerial robotics.
>
---
#### [replaced 010] Using Natural Language for Human-Robot Collaboration in the Real World
- **分类: cs.RO; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.11759v2](http://arxiv.org/pdf/2508.11759v2)**

> **作者:** Peter Lindes; Kaoutar Skiker
>
> **备注:** 34 pages, 11 figures, 5 tables. Submitted for publication (2026) in W.F. Lawless, Ranjeev Mittu, Shannon P. McGrarry, & Marco Brambilla (Eds.), Generative AI Risks and Benefits within Human-Machine Teams, Elsevier, Chapter 6
>
> **摘要:** We have a vision of a day when autonomous robots can collaborate with humans as assistants in performing complex tasks in the physical world. This vision includes that the robots will have the ability to communicate with their human collaborators using language that is natural to the humans. Traditional Interactive Task Learning (ITL) systems have some of this ability, but the language they can understand is very limited. The advent of large language models (LLMs) provides an opportunity to greatly improve the language understanding of robots, yet integrating the language abilities of LLMs with robots that operate in the real physical world is a challenging problem. In this chapter we first review briefly a few commercial robot products that work closely with humans, and discuss how they could be much better collaborators with robust language abilities. We then explore how an AI system with a cognitive agent that controls a physical robot at its core, interacts with both a human and an LLM, and accumulates situational knowledge through its experiences, can be a possible approach to reach that vision. We focus on three specific challenges of having the robot understand natural language, and present a simple proof-of-concept experiment using ChatGPT for each. Finally, we discuss what it will take to turn these simple experiments into an operational system where LLM-assisted language understanding is a part of an integrated robotic assistant that uses language to collaborate with humans.
>
---
#### [replaced 011] Ask-to-Clarify: Resolving Instruction Ambiguity through Multi-turn Dialogue
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.15061v2](http://arxiv.org/pdf/2509.15061v2)**

> **作者:** Xingyao Lin; Xinghao Zhu; Tianyi Lu; Sicheng Xie; Hui Zhang; Xipeng Qiu; Zuxuan Wu; Yu-Gang Jiang
>
> **备注:** 9 pages, 4 figures, 7 tables
>
> **摘要:** The ultimate goal of embodied agents is to create collaborators that can interact with humans, not mere executors that passively follow instructions. This requires agents to communicate, coordinate, and adapt their actions based on human feedback. Recently, advances in VLAs have offered a path toward this goal. However, most current VLA-based embodied agents operate in a one-way mode: they receive an instruction and execute it without feedback. This approach fails in real-world scenarios where instructions are often ambiguous. In this paper, we address this problem with the Ask-to-Clarify framework. Our framework first resolves ambiguous instructions by asking questions in a multi-turn dialogue. Then it generates low-level actions end-to-end. Specifically, the Ask-to-Clarify framework consists of two components, one VLM for collaboration and one diffusion for action. We also introduce a connection module that generates conditions for the diffusion based on the output of the VLM. This module adjusts the observation by instructions to create reliable conditions. We train our framework with a two-stage knowledge-insulation strategy. First, we fine-tune the collaboration component using ambiguity-solving dialogue data to handle ambiguity. Then, we integrate the action component while freezing the collaboration one. This preserves the interaction abilities while fine-tuning the diffusion to generate actions. The training strategy guarantees our framework can first ask questions, then generate actions. During inference, a signal detector functions as a router that helps our framework switch between asking questions and taking actions. We evaluate the Ask-to-Clarify framework in 8 real-world tasks, where it outperforms existing state-of-the-art VLAs. The results suggest that our proposed framework, along with the training strategy, provides a path toward collaborative embodied agents.
>
---
#### [replaced 012] Affordance-Based Disambiguation of Surgical Instructions for Collaborative Robot-Assisted Surgery
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2509.14967v2](http://arxiv.org/pdf/2509.14967v2)**

> **作者:** Ana Davila; Jacinto Colan; Yasuhisa Hasegawa
>
> **备注:** To be presented at the 1st Workshop on Intelligent Cobodied Assistance and Robotic Empowerment (iCARE). 2025 Conference on Robot Learning (CoRL)
>
> **摘要:** Effective human-robot collaboration in surgery is affected by the inherent ambiguity of verbal communication. This paper presents a framework for a robotic surgical assistant that interprets and disambiguates verbal instructions from a surgeon by grounding them in the visual context of the operating field. The system employs a two-level affordance-based reasoning process that first analyzes the surgical scene using a multimodal vision-language model and then reasons about the instruction using a knowledge base of tool capabilities. To ensure patient safety, a dual-set conformal prediction method is used to provide a statistically rigorous confidence measure for robot decisions, allowing it to identify and flag ambiguous commands. We evaluated our framework on a curated dataset of ambiguous surgical requests from cholecystectomy videos, demonstrating a general disambiguation rate of 60% and presenting a method for safer human-robot interaction in the operating room.
>
---
#### [replaced 013] Multi-Quadruped Cooperative Object Transport: Learning Decentralized Pinch-Lift-Move
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.14342v2](http://arxiv.org/pdf/2509.14342v2)**

> **作者:** Bikram Pandit; Aayam Kumar Shrestha; Alan Fern
>
> **摘要:** We study decentralized cooperative transport using teams of N-quadruped robots with arm that must pinch, lift, and move ungraspable objects through physical contact alone. Unlike prior work that relies on rigid mechanical coupling between robots and objects, we address the more challenging setting where mechanically independent robots must coordinate through contact forces alone without any communication or centralized control. To this end, we employ a hierarchical policy architecture that separates base locomotion from arm control, and propose a constellation reward formulation that unifies position and orientation tracking to enforce rigid contact behavior. The key insight is encouraging robots to behave as if rigidly connected to the object through careful reward design and training curriculum rather than explicit mechanical constraints. Our approach enables coordination through shared policy parameters and implicit synchronization cues - scaling to arbitrary team sizes without retraining. We show extensive simulation experiments to demonstrate robust transport across 2-10 robots on diverse object geometries and masses, along with sim2real transfer results on lightweight objects.
>
---
#### [replaced 014] Advances in Multimodal Adaptation and Generalization: From Traditional Approaches to Foundation Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.18592v4](http://arxiv.org/pdf/2501.18592v4)**

> **作者:** Hao Dong; Moru Liu; Kaiyang Zhou; Eleni Chatzi; Juho Kannala; Cyrill Stachniss; Olga Fink
>
> **备注:** Project page: https://github.com/donghao51/Awesome-Multimodal-Adaptation
>
> **摘要:** In real-world scenarios, achieving domain adaptation and generalization poses significant challenges, as models must adapt to or generalize across unknown target distributions. Extending these capabilities to unseen multimodal distributions, i.e., multimodal domain adaptation and generalization, is even more challenging due to the distinct characteristics of different modalities. Significant progress has been made over the years, with applications ranging from action recognition to semantic segmentation. Besides, the recent advent of large-scale pre-trained multimodal foundation models, such as CLIP, has inspired works leveraging these models to enhance adaptation and generalization performances or adapting them to downstream tasks. This survey provides the first comprehensive review of recent advances from traditional approaches to foundation models, covering: (1) Multimodal domain adaptation; (2) Multimodal test-time adaptation; (3) Multimodal domain generalization; (4) Domain adaptation and generalization with the help of multimodal foundation models; and (5) Adaptation of multimodal foundation models. For each topic, we formally define the problem and thoroughly review existing methods. Additionally, we analyze relevant datasets and applications, highlighting open challenges and potential future research directions. We maintain an active repository that contains up-to-date literature at https://github.com/donghao51/Awesome-Multimodal-Adaptation.
>
---
#### [replaced 015] SymBridge: A Human-in-the-Loop Cyber-Physical Interactive System for Adaptive Human-Robot Symbiosis
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.07358v2](http://arxiv.org/pdf/2502.07358v2)**

> **作者:** Haoran Chen; Yiteng Xu; Yiming Ren; Yaoqin Ye; Xinran Li; Ning Ding; Yuxuan Wu; Yaoze Liu; Peishan Cong; Ziyi Wang; Bushi Liu; Yuhan Chen; Zhiyang Dou; Xiaokun Leng; Manyi Li; Yuexin Ma; Changhe Tu
>
> **摘要:** The development of intelligent robots seeks to seamlessly integrate them into the human world, providing assistance and companionship in daily life and work, with the ultimate goal of achieving human-robot symbiosis. This requires robots with intelligent interaction abilities to work naturally and effectively with humans. However, current robotic simulators fail to support real human participation, limiting their ability to provide authentic interaction experiences and gather valuable human feedback essential for enhancing robotic capabilities. In this paper, we introduce SymBridge, the first human-in-the-loop cyber-physical interactive system designed to enable the safe and efficient development, evaluation, and optimization of human-robot interaction methods. Specifically, we employ augmented reality technology to enable real humans to interact with virtual robots in physical environments, creating an authentic interactive experience. Building on this, we propose a novel robotic interaction model that generates responsive, precise robot actions in real time through continuous human behavior observation. The model incorporates multi-resolution human motion features and environmental affordances, ensuring contextually adaptive robotic responses. Additionally, SymBridge enables continuous robot learning by collecting human feedback and dynamically adapting the robotic interaction model. By leveraging a carefully designed system architecture and modules, SymBridge builds a bridge between humans and robots, as well as between cyber and physical spaces, providing a natural and realistic online interaction experience while facilitating the continuous evolution of robotic intelligence. Extensive experiments, user studies, and real robot testing demonstrate the promising performance of the system and highlight its potential to significantly advance research on human-robot symbiosis.
>
---
