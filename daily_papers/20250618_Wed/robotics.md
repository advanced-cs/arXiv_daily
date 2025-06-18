# 机器人 cs.RO

- **最新发布 55 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] ClutterDexGrasp: A Sim-to-Real System for General Dexterous Grasping in Cluttered Scenes
- **分类: cs.RO**

- **简介: 该论文属于机械臂抓取任务，解决杂乱场景下灵巧抓取问题。提出ClutterDexGrasp系统，通过模拟到现实的迁移学习实现零样本部署。**

- **链接: [http://arxiv.org/pdf/2506.14317v1](http://arxiv.org/pdf/2506.14317v1)**

> **作者:** Zeyuan Chen; Qiyang Yan; Yuanpei Chen; Tianhao Wu; Jiyao Zhang; Zihan Ding; Jinzhou Li; Yaodong Yang; Hao Dong
>
> **摘要:** Dexterous grasping in cluttered scenes presents significant challenges due to diverse object geometries, occlusions, and potential collisions. Existing methods primarily focus on single-object grasping or grasp-pose prediction without interaction, which are insufficient for complex, cluttered scenes. Recent vision-language-action models offer a potential solution but require extensive real-world demonstrations, making them costly and difficult to scale. To address these limitations, we revisit the sim-to-real transfer pipeline and develop key techniques that enable zero-shot deployment in reality while maintaining robust generalization. We propose ClutterDexGrasp, a two-stage teacher-student framework for closed-loop target-oriented dexterous grasping in cluttered scenes. The framework features a teacher policy trained in simulation using clutter density curriculum learning, incorporating both a novel geometry and spatially-embedded scene representation and a comprehensive safety curriculum, enabling general, dynamic, and safe grasping behaviors. Through imitation learning, we distill the teacher's knowledge into a student 3D diffusion policy (DP3) that operates on partial point cloud observations. To the best of our knowledge, this represents the first zero-shot sim-to-real closed-loop system for target-oriented dexterous grasping in cluttered scenes, demonstrating robust performance across diverse objects and layouts. More details and videos are available at https://clutterdexgrasp.github.io/.
>
---
#### [new 002] A Point Cloud Completion Approach for the Grasping of Partially Occluded Objects and Its Applications in Robotic Strawberry Harvesting
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决部分遮挡物体的抓取问题。通过点云补全和分割，提升草莓采摘的抓取成功率与安全性。**

- **链接: [http://arxiv.org/pdf/2506.14066v1](http://arxiv.org/pdf/2506.14066v1)**

> **作者:** Ali Abouzeid; Malak Mansour; Chengsong Hu; Dezhen Song
>
> **摘要:** In robotic fruit picking applications, managing object occlusion in unstructured settings poses a substantial challenge for designing grasping algorithms. Using strawberry harvesting as a case study, we present an end-to-end framework for effective object detection, segmentation, and grasp planning to tackle this issue caused by partially occluded objects. Our strategy begins with point cloud denoising and segmentation to accurately locate fruits. To compensate for incomplete scans due to occlusion, we apply a point cloud completion model to create a dense 3D reconstruction of the strawberries. The target selection focuses on ripe strawberries while categorizing others as obstacles, followed by converting the refined point cloud into an occupancy map for collision-aware motion planning. Our experimental results demonstrate high shape reconstruction accuracy, with the lowest Chamfer Distance compared to state-of-the-art methods with 1.10 mm, and significantly improved grasp success rates of 79.17%, yielding an overall success-to-attempt ratio of 89.58\% in real-world strawberry harvesting. Additionally, our method reduces the obstacle hit rate from 43.33% to 13.95%, highlighting its effectiveness in improving both grasp quality and safety compared to prior approaches. This pipeline substantially improves autonomous strawberry harvesting, advancing more efficient and reliable robotic fruit picking systems.
>
---
#### [new 003] Steering Robots with Inference-Time Interactions
- **分类: cs.RO; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决预训练策略在部署时出错难以修正的问题。通过用户交互在推理时引导策略，无需微调即可纠正错误。**

- **链接: [http://arxiv.org/pdf/2506.14287v1](http://arxiv.org/pdf/2506.14287v1)**

> **作者:** Yanwei Wang
>
> **备注:** MIT Robotics PhD Thesis
>
> **摘要:** Imitation learning has driven the development of generalist policies capable of autonomously solving multiple tasks. However, when a pretrained policy makes errors during deployment, there are limited mechanisms for users to correct its behavior. While collecting additional data for finetuning can address such issues, doing so for each downstream use case is inefficient at deployment. My research proposes an alternative: keeping pretrained policies frozen as a fixed skill repertoire while allowing user interactions to guide behavior generation toward user preferences at inference time. By making pretrained policies steerable, users can help correct policy errors when the model struggles to generalize-without needing to finetune the policy. Specifically, I propose (1) inference-time steering, which leverages user interactions to switch between discrete skills, and (2) task and motion imitation, which enables user interactions to edit continuous motions while satisfying task constraints defined by discrete symbolic plans. These frameworks correct misaligned policy predictions without requiring additional training, maximizing the utility of pretrained models while achieving inference-time user objectives.
>
---
#### [new 004] SENIOR: Efficient Query Selection and Preference-Guided Exploration in Preference-based Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于偏好强化学习任务，旨在提升人类反馈效率和策略收敛速度。提出SENIOR方法，通过有效查询选择和偏好引导探索解决样本效率低的问题。**

- **链接: [http://arxiv.org/pdf/2506.14648v1](http://arxiv.org/pdf/2506.14648v1)**

> **作者:** Hexian Ni; Tao Lu; Haoyuan Hu; Yinghao Cai; Shuo Wang
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Preference-based Reinforcement Learning (PbRL) methods provide a solution to avoid reward engineering by learning reward models based on human preferences. However, poor feedback- and sample- efficiency still remain the problems that hinder the application of PbRL. In this paper, we present a novel efficient query selection and preference-guided exploration method, called SENIOR, which could select the meaningful and easy-to-comparison behavior segment pairs to improve human feedback-efficiency and accelerate policy learning with the designed preference-guided intrinsic rewards. Our key idea is twofold: (1) We designed a Motion-Distinction-based Selection scheme (MDS). It selects segment pairs with apparent motion and different directions through kernel density estimation of states, which is more task-related and easy for human preference labeling; (2) We proposed a novel preference-guided exploration method (PGE). It encourages the exploration towards the states with high preference and low visits and continuously guides the agent achieving the valuable samples. The synergy between the two mechanisms could significantly accelerate the progress of reward and policy learning. Our experiments show that SENIOR outperforms other five existing methods in both human feedback-efficiency and policy convergence speed on six complex robot manipulation tasks from simulation and four real-worlds.
>
---
#### [new 005] ReLCP: Scalable Complementarity-Based Collision Resolution for Smooth Rigid Bodies
- **分类: cs.RO; cond-mat.soft; physics.comp-ph**

- **简介: 该论文属于物理模拟任务，解决刚体碰撞分辨率问题。提出ReLCP算法，通过自适应生成线性互补问题，提高计算效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.14097v1](http://arxiv.org/pdf/2506.14097v1)**

> **作者:** Bryce Palmer; Hasan Metin Aktulga; Tong Gao
>
> **摘要:** We present a complementarity-based collision resolution algorithm for smooth, non-spherical, rigid bodies. Unlike discrete surface representation approaches, which approximate surfaces using discrete elements (e.g., tessellations or sub-spheres) with constraints between nearby faces, edges, nodes, or sub-objects, our algorithm solves a recursively generated linear complementarity problem (ReLCP) to adaptively identify potential collision locations during the collision resolution procedure. Despite adaptively and in contrast to Newton-esque schemes, we prove conditions under which the resulting solution exists and the center of mass translational and rotational dynamics are unique. Our ReLCP also converges to classical LCP-based collision resolution for sufficiently small timesteps. Because increasing the surface resolution in discrete representation methods necessitates subdividing geometry into finer elements -- leading to a super-linear increase in the number of collision constraints -- these approaches scale poorly with increased surface resolution. In contrast, our adaptive ReLCP framework begins with a single constraint per pair of nearby bodies and introduces new constraints only when unconstrained motion would lead to overlap, circumventing the oversampling required by discrete methods. By requiring one to two orders of magnitude fewer collision constraints to achieve the same surface resolution, we observe 10-100x speedup in densely packed applications. We validate our ReLCP method against multisphere and single-constraint methods, comparing convergence in a two-ellipsoid collision test, scalability and performance in a compacting ellipsoid suspension and growing bacterial colony, and stability in a taut chainmail network, highlighting our ability to achieve high-fidelity surface representations without suffering from poor scalability or artificial surface roughness.
>
---
#### [new 006] Whole-Body Control Framework for Humanoid Robots with Heavy Limbs: A Model-Based Approach
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决 humanoid 机器人因重肢导致的平衡问题。提出一种基于模型的全身控制框架，结合运动规划与优化算法，提升动态行走和环境适应能力。**

- **链接: [http://arxiv.org/pdf/2506.14278v1](http://arxiv.org/pdf/2506.14278v1)**

> **作者:** Tianlin Zhang; Linzhu Yue; Hongbo Zhang; Lingwei Zhang; Xuanqi Zeng; Zhitao Song; Yun-Hui Liu
>
> **摘要:** Humanoid robots often face significant balance issues due to the motion of their heavy limbs. These challenges are particularly pronounced when attempting dynamic motion or operating in environments with irregular terrain. To address this challenge, this manuscript proposes a whole-body control framework for humanoid robots with heavy limbs, using a model-based approach that combines a kino-dynamics planner and a hierarchical optimization problem. The kino-dynamics planner is designed as a model predictive control (MPC) scheme to account for the impact of heavy limbs on mass and inertia distribution. By simplifying the robot's system dynamics and constraints, the planner enables real-time planning of motion and contact forces. The hierarchical optimization problem is formulated using Hierarchical Quadratic Programming (HQP) to minimize limb control errors and ensure compliance with the policy generated by the kino-dynamics planner. Experimental validation of the proposed framework demonstrates its effectiveness. The humanoid robot with heavy limbs controlled by the proposed framework can achieve dynamic walking speeds of up to 1.2~m/s, respond to external disturbances of up to 60~N, and maintain balance on challenging terrains such as uneven surfaces, and outdoor environments.
>
---
#### [new 007] A Cooperative Contactless Object Transport with Acoustic Robots
- **分类: cs.RO**

- **简介: 该论文属于协同无接触物体运输任务，解决多机器人协同控制问题。通过声学系统实现空中物体的精确操控与运输。**

- **链接: [http://arxiv.org/pdf/2506.13957v1](http://arxiv.org/pdf/2506.13957v1)**

> **作者:** Narsimlu Kemsaram; Akin Delibasi; James Hardwick; Bonot Gautam; Diego Martinez Plasencia; Sriram Subramanian
>
> **备注:** This paper has been accepted for publication in the Proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) as oral presentation, 8 pages with 8 figures
>
> **摘要:** Cooperative transport, the simultaneous movement of an object by multiple agents, has been widely observed in biological systems such as ant colonies, which improve efficiency and adaptability in dynamic environments. Inspired by these natural phenomena, we present a novel acoustic robotic system for the transport of contactless objects in mid-air. Our system leverages phased ultrasonic transducers and a robotic control system onboard to generate localized acoustic pressure fields, enabling precise manipulation of airborne particles and robots. We categorize contactless object-transport strategies into independent transport (uncoordinated) and forward-facing cooperative transport (coordinated), drawing parallels with biological systems to optimize efficiency and robustness. The proposed system is experimentally validated by evaluating levitation stability using a microphone in the measurement lab, transport efficiency through a phase-space motion capture system, and clock synchronization accuracy via an oscilloscope. The results demonstrate the feasibility of both independent and cooperative airborne object transport. This research contributes to the field of acoustophoretic robotics, with potential applications in contactless material handling, micro-assembly, and biomedical applications.
>
---
#### [new 008] Beyond the Plane: A 3D Representation of Human Personal Space for Socially-Aware Robotics
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决机器人在三维空间中理解并尊重人类个人空间的问题。工作包括提出一种新的三维个人空间模型，结合高度和水平距离计算不适感。**

- **链接: [http://arxiv.org/pdf/2506.13937v1](http://arxiv.org/pdf/2506.13937v1)**

> **作者:** Caio C. G. Ribeiro; Douglas G. Macharet
>
> **备注:** Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (ROMAN)
>
> **摘要:** The increasing presence of robots in human environments requires them to exhibit socially appropriate behavior, adhering to social norms. A critical aspect in this context is the concept of personal space, a psychological boundary around an individual that influences their comfort based on proximity. This concept extends to human-robot interaction, where robots must respect personal space to avoid causing discomfort. While much research has focused on modeling personal space in two dimensions, almost none have considered the vertical dimension. In this work, we propose a novel three-dimensional personal space model that integrates both height (introducing a discomfort function along the Z-axis) and horizontal proximity (via a classic XY-plane formulation) to quantify discomfort. To the best of our knowledge, this is the first work to compute discomfort in 3D space at any robot component's position, considering the person's configuration and height.
>
---
#### [new 009] ATK: Automatic Task-driven Keypoint Selection for Robust Policy Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉控制任务，旨在解决环境变化导致的策略性能下降问题。通过自动选择关键点提升策略鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.13867v1](http://arxiv.org/pdf/2506.13867v1)**

> **作者:** Yunchu Zhang; Shubham Mittal; Zhengyu Zhang; Liyiming Ke; Siddhartha Srinivasa; Abhishek Gupta
>
> **摘要:** Visuomotor policies often suffer from perceptual challenges, where visual differences between training and evaluation environments degrade policy performance. Policies relying on state estimations, like 6D pose, require task-specific tracking and are difficult to scale, while raw sensor-based policies may lack robustness to small visual disturbances.In this work, we leverage 2D keypoints - spatially consistent features in the image frame - as a flexible state representation for robust policy learning and apply it to both sim-to-real transfer and real-world imitation learning. However, the choice of which keypoints to use can vary across objects and tasks. We propose a novel method, ATK, to automatically select keypoints in a task-driven manner so that the chosen keypoints are predictive of optimal behavior for the given task. Our proposal optimizes for a minimal set of keypoints that focus on task-relevant parts while preserving policy performance and robustness. We distill expert data (either from an expert policy in simulation or a human expert) into a policy that operates on RGB images while tracking the selected keypoints. By leveraging pre-trained visual modules, our system effectively encodes states and transfers policies to the real-world evaluation scenario despite wide scene variations and perceptual challenges such as transparent objects, fine-grained tasks, and deformable objects manipulation. We validate ATK on various robotic tasks, demonstrating that these minimal keypoint representations significantly improve robustness to visual disturbances and environmental variations. See all experiments and more details on our website.
>
---
#### [new 010] Automatic Cannulation of Femoral Vessels in a Porcine Shock Model
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决创伤中快速建立血管通路的问题。通过自动化技术实现猪模型中的股静脉和动脉穿刺。**

- **链接: [http://arxiv.org/pdf/2506.14467v1](http://arxiv.org/pdf/2506.14467v1)**

> **作者:** Nico Zevallos; Cecilia G. Morales; Andrew Orekhov; Tejas Rane; Hernando Gomez; Francis X. Guyette; Michael R. Pinsky; John Galeotti; Artur Dubrawski; Howie Choset
>
> **备注:** 2 pages, 2 figures, conference
>
> **摘要:** Rapid and reliable vascular access is critical in trauma and critical care. Central vascular catheterization enables high-volume resuscitation, hemodynamic monitoring, and advanced interventions like ECMO and REBOA. While peripheral access is common, central access is often necessary but requires specialized ultrasound-guided skills, posing challenges in prehospital settings. The complexity arises from deep target vessels and the precision needed for needle placement. Traditional techniques, like the Seldinger method, demand expertise to avoid complications. Despite its importance, ultrasound-guided central access is underutilized due to limited field expertise. While autonomous needle insertion has been explored for peripheral vessels, only semi-autonomous methods exist for femoral access. This work advances toward full automation, integrating robotic ultrasound for minimally invasive emergency procedures. Our key contribution is the successful femoral vein and artery cannulation in a porcine hemorrhagic shock model.
>
---
#### [new 011] Sequence Modeling for Time-Optimal Quadrotor Trajectory Optimization with Sampling-based Robustness Analysis
- **分类: cs.RO**

- **简介: 该论文属于无人机轨迹优化任务，解决时间最优轨迹计算效率低的问题，通过学习模型加速轨迹生成并提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.13915v1](http://arxiv.org/pdf/2506.13915v1)**

> **作者:** Katherine Mao; Hongzhan Yu; Ruipeng Zhang; Igor Spasojevic; M Ani Hsieh; Sicun Gao; Vijay Kumar
>
> **摘要:** Time-optimal trajectories drive quadrotors to their dynamic limits, but computing such trajectories involves solving non-convex problems via iterative nonlinear optimization, making them prohibitively costly for real-time applications. In this work, we investigate learning-based models that imitate a model-based time-optimal trajectory planner to accelerate trajectory generation. Given a dataset of collision-free geometric paths, we show that modeling architectures can effectively learn the patterns underlying time-optimal trajectories. We introduce a quantitative framework to analyze local analytic properties of the learned models, and link them to the Backward Reachable Tube of the geometric tracking controller. To enhance robustness, we propose a data augmentation scheme that applies random perturbations to the input paths. Compared to classical planners, our method achieves substantial speedups, and we validate its real-time feasibility on a hardware quadrotor platform. Experiments demonstrate that the learned models generalize to previously unseen path lengths. The code for our approach can be found here: https://github.com/maokat12/lbTOPPQuad
>
---
#### [new 012] Data Driven Approach to Input Shaping for Vibration Suppression in a Flexible Robot Arm
- **分类: cs.RO**

- **简介: 该论文属于振动控制任务，旨在解决柔性机械臂残余振动问题。通过数据驱动方法自适应调整输入整形参数，有效抑制振动。**

- **链接: [http://arxiv.org/pdf/2506.14405v1](http://arxiv.org/pdf/2506.14405v1)**

> **作者:** Jarkko Kotaniemi; Janne Saukkoriipi; Shuai Li; Markku Suomalainen
>
> **备注:** 6 pages, 11 figures, robosoft2025 conference
>
> **摘要:** This paper presents a simple and effective method for setting parameters for an input shaper to suppress the residual vibrations in flexible robot arms using a data-driven approach. The parameters are adaptively tuned in the workspace of the robot by interpolating previously measured data of the robot's residual vibrations. Input shaping is a simple and robust technique to generate vibration-reduced shaped commands by a convolution of an impulse sequence with the desired input command. The generated impulses create waves in the material countering the natural vibrations of the system. The method is demonstrated with a flexible 3D-printed robot arm with multiple different materials, achieving a significant reduction in the residual vibrations.
>
---
#### [new 013] Barrier Method for Inequality Constrained Factor Graph Optimization with Application to Model Predictive Control
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文将障碍法引入因子图优化，解决机器人控制中的约束处理问题，实现高效求解带不等式约束的最优控制问题。**

- **链接: [http://arxiv.org/pdf/2506.14341v1](http://arxiv.org/pdf/2506.14341v1)**

> **作者:** Anas Abdelkarim; Holger Voos; Daniel Görges
>
> **摘要:** Factor graphs have demonstrated remarkable efficiency for robotic perception tasks, particularly in localization and mapping applications. However, their application to optimal control problems -- especially Model Predictive Control (MPC) -- has remained limited due to fundamental challenges in constraint handling. This paper presents a novel integration of the Barrier Interior Point Method (BIPM) with factor graphs, implemented as an open-source extension to the widely adopted g2o framework. Our approach introduces specialized inequality factor nodes that encode logarithmic barrier functions, thereby overcoming the quadratic-form limitations of conventional factor graph formulations. To the best of our knowledge, this is the first g2o-based implementation capable of efficiently handling both equality and inequality constraints within a unified optimization backend. We validate the method through a multi-objective adaptive cruise control application for autonomous vehicles. Benchmark comparisons with state-of-the-art constraint-handling techniques demonstrate faster convergence and improved computational efficiency. (Code repository: https://github.com/snt-arg/bipm_g2o)
>
---
#### [new 014] Latent Action Diffusion for Cross-Embodiment Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决不同机械臂间技能迁移困难的问题。通过学习统一的潜在动作空间，实现跨形态机器人的高效控制与数据共享。**

- **链接: [http://arxiv.org/pdf/2506.14608v1](http://arxiv.org/pdf/2506.14608v1)**

> **作者:** Erik Bauer; Elvis Nava; Robert K. Katzschmann
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** End-to-end learning approaches offer great potential for robotic manipulation, but their impact is constrained by data scarcity and heterogeneity across different embodiments. In particular, diverse action spaces across different end-effectors create barriers for cross-embodiment learning and skill transfer. We address this challenge through diffusion policies learned in a latent action space that unifies diverse end-effector actions. We first show that we can learn a semantically aligned latent action space for anthropomorphic robotic hands, a human hand, and a parallel jaw gripper using encoders trained with a contrastive loss. Second, we show that by using our proposed latent action space for co-training on manipulation data from different end-effectors, we can utilize a single policy for multi-robot control and obtain up to 13% improved manipulation success rates, indicating successful skill transfer despite a significant embodiment gap. Our approach using latent cross-embodiment policies presents a new method to unify different action spaces across embodiments, enabling efficient multi-robot control and data sharing across robot setups. This unified representation significantly reduces the need for extensive data collection for each new robot morphology, accelerates generalization across embodiments, and ultimately facilitates more scalable and efficient robotic learning.
>
---
#### [new 015] Robust Adaptive Time-Varying Control Barrier Function with Application to Robotic Surface Treatment
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决动态环境下时间变化约束的鲁棒性问题，提出一种考虑不确定性和扰动的自适应控制方法。**

- **链接: [http://arxiv.org/pdf/2506.14249v1](http://arxiv.org/pdf/2506.14249v1)**

> **作者:** Yitaek Kim; Christoffer Sloth
>
> **备注:** This work has been accepted to ECC 2025
>
> **摘要:** Set invariance techniques such as control barrier functions (CBFs) can be used to enforce time-varying constraints such as keeping a safe distance from dynamic objects. However, existing methods for enforcing time-varying constraints often overlook model uncertainties. To address this issue, this paper proposes a CBFs-based robust adaptive controller design endowing time-varying constraints while considering parametric uncertainty and additive disturbances. To this end, we first leverage Robust adaptive Control Barrier Functions (RaCBFs) to handle model uncertainty, along with the concept of Input-to-State Safety (ISSf) to ensure robustness towards input disturbances. Furthermore, to alleviate the inherent conservatism in robustness, we also incorporate a set membership identification scheme. We demonstrate the proposed method on robotic surface treatment that requires time-varying force bounds to ensure uniform quality, in numerical simulation and real robotic setup, showing that the quality is formally guaranteed within an acceptable range.
>
---
#### [new 016] RobotSmith: Generative Robotic Tool Design for Acquisition of Complex Manipulation Skills
- **分类: cs.RO**

- **简介: 该论文属于机器人工具设计任务，旨在解决机器人难以处理复杂操作的问题。工作包括提出RobotSmith系统，结合视觉语言模型和物理模拟生成有效工具并优化使用策略。**

- **链接: [http://arxiv.org/pdf/2506.14763v1](http://arxiv.org/pdf/2506.14763v1)**

> **作者:** Chunru Lin; Haotian Yuan; Yian Wang; Xiaowen Qiu; Tsun-Hsuan Wang; Minghao Guo; Bohan Wang; Yashraj Narang; Dieter Fox; Chuang Gan
>
> **摘要:** Endowing robots with tool design abilities is critical for enabling them to solve complex manipulation tasks that would otherwise be intractable. While recent generative frameworks can automatically synthesize task settings, such as 3D scenes and reward functions, they have not yet addressed the challenge of tool-use scenarios. Simply retrieving human-designed tools might not be ideal since many tools (e.g., a rolling pin) are difficult for robotic manipulators to handle. Furthermore, existing tool design approaches either rely on predefined templates with limited parameter tuning or apply generic 3D generation methods that are not optimized for tool creation. To address these limitations, we propose RobotSmith, an automated pipeline that leverages the implicit physical knowledge embedded in vision-language models (VLMs) alongside the more accurate physics provided by physics simulations to design and use tools for robotic manipulation. Our system (1) iteratively proposes tool designs using collaborative VLM agents, (2) generates low-level robot trajectories for tool use, and (3) jointly optimizes tool geometry and usage for task performance. We evaluate our approach across a wide range of manipulation tasks involving rigid, deformable, and fluid objects. Experiments show that our method consistently outperforms strong baselines in terms of both task success rate and overall performance. Notably, our approach achieves a 50.0\% average success rate, significantly surpassing other baselines such as 3D generation (21.4%) and tool retrieval (11.1%). Finally, we deploy our system in real-world settings, demonstrating that the generated tools and their usage plans transfer effectively to physical execution, validating the practicality and generalization capabilities of our approach.
>
---
#### [new 017] GAMORA: A Gesture Articulated Meta Operative Robotic Arm for Hazardous Material Handling in Containment-Level Environments
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出GAMORA系统，用于危险物质处理，解决高风险实验室中人机交互与精准操作问题，通过VR和机器人技术实现安全、精确的远程操控。**

- **链接: [http://arxiv.org/pdf/2506.14513v1](http://arxiv.org/pdf/2506.14513v1)**

> **作者:** Farha Abdul Wasay; Mohammed Abdul Rahman; Hania Ghouse
>
> **摘要:** The convergence of robotics and virtual reality (VR) has enabled safer and more efficient workflows in high-risk laboratory settings, particularly virology labs. As biohazard complexity increases, minimizing direct human exposure while maintaining precision becomes essential. We propose GAMORA (Gesture Articulated Meta Operative Robotic Arm), a novel VR-guided robotic system that enables remote execution of hazardous tasks using natural hand gestures. Unlike existing scripted automation or traditional teleoperation, GAMORA integrates the Oculus Quest 2, NVIDIA Jetson Nano, and Robot Operating System (ROS) to provide real-time immersive control, digital twin simulation, and inverse kinematics-based articulation. The system supports VR-based training and simulation while executing precision tasks in physical environments via a 3D-printed robotic arm. Inverse kinematics ensure accurate manipulation for delicate operations such as specimen handling and pipetting. The pipeline includes Unity-based 3D environment construction, real-time motion planning, and hardware-in-the-loop testing. GAMORA achieved a mean positional discrepancy of 2.2 mm (improved from 4 mm), pipetting accuracy within 0.2 mL, and repeatability of 1.2 mm across 50 trials. Integrated object detection via YOLOv8 enhances spatial awareness, while energy-efficient operation (50% reduced power output) ensures sustainable deployment. The system's digital-physical feedback loop enables safe, precise, and repeatable automation of high-risk lab tasks. GAMORA offers a scalable, immersive solution for robotic control and biosafety in biomedical research environments.
>
---
#### [new 018] Factor-Graph-Based Passive Acoustic Navigation for Decentralized Cooperative Localization Using Bearing Elevation Depth Difference
- **分类: cs.RO**

- **简介: 该论文属于水下多智能体协同定位任务，解决水下通信受限下的精准定位问题，提出基于因子图的BEDD方法实现AUVs协作导航。**

- **链接: [http://arxiv.org/pdf/2506.14690v1](http://arxiv.org/pdf/2506.14690v1)**

> **作者:** Kalliyan Velasco; Timothy W. McLain; Joshua G. Mangelson
>
> **摘要:** Accurate and scalable underwater multi-agent localization remains a critical challenge due to the constraints of underwater communication. In this work, we propose a multi-agent localization framework using a factor-graph representation that incorporates bearing, elevation, and depth difference (BEDD). Our method leverages inverted ultra-short baseline (inverted-USBL) derived azimuth and elevation measurements from incoming acoustic signals and relative depth measurements to enable cooperative localization for a multi-robot team of autonomous underwater vehicles (AUVs). We validate our approach in the HoloOcean underwater simulator with a fleet of AUVs, demonstrating improved localization accuracy compared to dead reckoning. Additionally, we investigate the impact of azimuth and elevation measurement outliers, highlighting the need for robust outlier rejection techniques for acoustic signals.
>
---
#### [new 019] TUM Teleoperation: Open Source Software for Remote Driving and Assistance of Automated Vehicles
- **分类: cs.RO; cs.HC; cs.SE**

- **简介: 该论文属于自动驾驶领域，解决远程操控与辅助的集成问题。提出一个开源软件栈，支持远程驾驶和高阶交互，实现与真实车辆的集成测试。**

- **链接: [http://arxiv.org/pdf/2506.13933v1](http://arxiv.org/pdf/2506.13933v1)**

> **作者:** Tobias Kerbl; David Brecht; Nils Gehrke; Nijinshan Karunainayagam; Niklas Krauss; Florian Pfab; Richard Taupitz; Ines Trautmannsheimer; Xiyan Su; Maria-Magdalena Wolf; Frank Diermeyer
>
> **摘要:** Teleoperation is a key enabler for future mobility, supporting Automated Vehicles in rare and complex scenarios beyond the capabilities of their automation. Despite ongoing research, no open source software currently combines Remote Driving, e.g., via steering wheel and pedals, Remote Assistance through high-level interaction with automated driving software modules, and integration with a real-world vehicle for practical testing. To address this gap, we present a modular, open source teleoperation software stack that can interact with an automated driving software, e.g., Autoware, enabling Remote Assistance and Remote Driving. The software featuresstandardized interfaces for seamless integration with various real-world and simulation platforms, while allowing for flexible design of the human-machine interface. The system is designed for modularity and ease of extension, serving as a foundation for collaborative development on individual software components as well as realistic testing and user studies. To demonstrate the applicability of our software, we evaluated the latency and performance of different vehicle platforms in simulation and real-world. The source code is available on GitHub
>
---
#### [new 020] AMPLIFY: Actionless Motion Priors for Robot Learning from Videos
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人学习任务，解决动作标签数据稀缺问题。通过AMPLITFY框架，利用无动作视频数据学习运动先验，提升策略泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.14198v1](http://arxiv.org/pdf/2506.14198v1)**

> **作者:** Jeremy A. Collins; Loránd Cheng; Kunal Aneja; Albert Wilcox; Benjamin Joffe; Animesh Garg
>
> **摘要:** Action-labeled data for robotics is scarce and expensive, limiting the generalization of learned policies. In contrast, vast amounts of action-free video data are readily available, but translating these observations into effective policies remains a challenge. We introduce AMPLIFY, a novel framework that leverages large-scale video data by encoding visual dynamics into compact, discrete motion tokens derived from keypoint trajectories. Our modular approach separates visual motion prediction from action inference, decoupling the challenges of learning what motion defines a task from how robots can perform it. We train a forward dynamics model on abundant action-free videos and an inverse dynamics model on a limited set of action-labeled examples, allowing for independent scaling. Extensive evaluations demonstrate that the learned dynamics are both accurate, achieving up to 3.7x better MSE and over 2.5x better pixel prediction accuracy compared to prior approaches, and broadly useful. In downstream policy learning, our dynamics predictions enable a 1.2-2.2x improvement in low-data regimes, a 1.4x average improvement by learning from action-free human videos, and the first generalization to LIBERO tasks from zero in-distribution action data. Beyond robotic control, we find the dynamics learned by AMPLIFY to be a versatile latent world model, enhancing video prediction quality. Our results present a novel paradigm leveraging heterogeneous data sources to build efficient, generalizable world models. More information can be found at https://amplify-robotics.github.io/.
>
---
#### [new 021] Haptic-Based User Authentication for Tele-robotic System
- **分类: cs.RO**

- **简介: 该论文属于安全认证任务，旨在解决远程操作机器人中的身份验证问题。通过分析触觉反馈中的用户行为特征，提出一种抗欺骗和重放攻击的认证方法。**

- **链接: [http://arxiv.org/pdf/2506.14116v1](http://arxiv.org/pdf/2506.14116v1)**

> **作者:** Rongyu Yu; Kan Chen; Zeyu Deng; Chen Wang; Burak Kizilkaya; Liying Emma Li
>
> **摘要:** Tele-operated robots rely on real-time user behavior mapping for remote tasks, but ensuring secure authentication remains a challenge. Traditional methods, such as passwords and static biometrics, are vulnerable to spoofing and replay attacks, particularly in high-stakes, continuous interactions. This paper presents a novel anti-spoofing and anti-replay authentication approach that leverages distinctive user behavioral features extracted from haptic feedback during human-robot interactions. To evaluate our authentication approach, we collected a time-series force feedback dataset from 15 participants performing seven distinct tasks. We then developed a transformer-based deep learning model to extract temporal features from the haptic signals. By analyzing user-specific force dynamics, our method achieves over 90 percent accuracy in both user identification and task classification, demonstrating its potential for enhancing access control and identity assurance in tele-robotic systems.
>
---
#### [new 022] Quadrotor Morpho-Transition: Learning vs Model-Based Control Strategies
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究无人机在空中转换为地面形态的控制问题，比较了强化学习与模型预测控制方法，旨在提升飞行中敏捷动作的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.14039v1](http://arxiv.org/pdf/2506.14039v1)**

> **作者:** Ioannis Mandralis; Richard M. Murray; Morteza Gharib
>
> **摘要:** Quadrotor Morpho-Transition, or the act of transitioning from air to ground through mid-air transformation, involves complex aerodynamic interactions and a need to operate near actuator saturation, complicating controller design. In recent work, morpho-transition has been studied from a model-based control perspective, but these approaches remain limited due to unmodeled dynamics and the requirement for planning through contacts. Here, we train an end-to-end Reinforcement Learning (RL) controller to learn a morpho-transition policy and demonstrate successful transfer to hardware. We find that the RL control policy achieves agile landing, but only transfers to hardware if motor dynamics and observation delays are taken into account. On the other hand, a baseline MPC controller transfers out-of-the-box without knowledge of the actuator dynamics and delays, at the cost of reduced recovery from disturbances in the event of unknown actuator failures. Our work opens the way for more robust control of agile in-flight quadrotor maneuvers that require mid-air transformation.
>
---
#### [new 023] NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决通用模型与专用模型集成中的异步问题。提出NetRoller适配器，实现高效融合与性能提升。**

- **链接: [http://arxiv.org/pdf/2506.14589v1](http://arxiv.org/pdf/2506.14589v1)**

> **作者:** Ren Xin; Hongji Liu; Xiaodong Mei; Wenru Liu; Maosheng Ye; Zhili Chen; Jun Ma
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Integrating General Models (GMs) such as Large Language Models (LLMs), with Specialized Models (SMs) in autonomous driving tasks presents a promising approach to mitigating challenges in data diversity and model capacity of existing specialized driving models. However, this integration leads to problems of asynchronous systems, which arise from the distinct characteristics inherent in GMs and SMs. To tackle this challenge, we propose NetRoller, an adapter that incorporates a set of novel mechanisms to facilitate the seamless integration of GMs and specialized driving models. Specifically, our mechanisms for interfacing the asynchronous GMs and SMs are organized into three key stages. NetRoller first harvests semantically rich and computationally efficient representations from the reasoning processes of LLMs using an early stopping mechanism, which preserves critical insights on driving context while maintaining low overhead. It then applies learnable query embeddings, nonsensical embeddings, and positional layer embeddings to facilitate robust and efficient cross-modality translation. At last, it employs computationally efficient Query Shift and Feature Shift mechanisms to enhance the performance of SMs through few-epoch fine-tuning. Based on the mechanisms formalized in these three stages, NetRoller enables specialized driving models to operate at their native frequencies while maintaining situational awareness of the GM. Experiments conducted on the nuScenes dataset demonstrate that integrating GM through NetRoller significantly improves human similarity and safety in planning tasks, and it also achieves noticeable precision improvements in detection and mapping tasks for end-to-end autonomous driving. The code and models are available at https://github.com/Rex-sys-hk/NetRoller .
>
---
#### [new 024] Tactile Beyond Pixels: Multisensory Touch Representations for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，旨在提升机器人操作的精确性与鲁棒性。通过多模态触觉表示Sparsh-X，融合图像、音频、运动和压力信息，增强对物理属性的感知能力。**

- **链接: [http://arxiv.org/pdf/2506.14754v1](http://arxiv.org/pdf/2506.14754v1)**

> **作者:** Carolina Higuera; Akash Sharma; Taosha Fan; Chaithanya Krishna Bodduluri; Byron Boots; Michael Kaess; Mike Lambeta; Tingfan Wu; Zixi Liu; Francois Robert Hogan; Mustafa Mukadam
>
> **摘要:** We present Sparsh-X, the first multisensory touch representations across four tactile modalities: image, audio, motion, and pressure. Trained on ~1M contact-rich interactions collected with the Digit 360 sensor, Sparsh-X captures complementary touch signals at diverse temporal and spatial scales. By leveraging self-supervised learning, Sparsh-X fuses these modalities into a unified representation that captures physical properties useful for robot manipulation tasks. We study how to effectively integrate real-world touch representations for both imitation learning and tactile adaptation of sim-trained policies, showing that Sparsh-X boosts policy success rates by 63% over an end-to-end model using tactile images and improves robustness by 90% in recovering object states from touch. Finally, we benchmark Sparsh-X ability to make inferences about physical properties, such as object-action identification, material-quantity estimation, and force estimation. Sparsh-X improves accuracy in characterizing physical properties by 48% compared to end-to-end approaches, demonstrating the advantages of multisensory pretraining for capturing features essential for dexterous manipulation.
>
---
#### [new 025] Enhancing Object Search in Indoor Spaces via Personalized Object-factored Ontologies
- **分类: cs.RO**

- **简介: 该论文属于室内物体搜索任务，旨在提升服务机器人在长期环境中的个性化搜索能力。通过构建个性化本体和自适应推理策略，提高多物体搜索效果。**

- **链接: [http://arxiv.org/pdf/2506.14422v1](http://arxiv.org/pdf/2506.14422v1)**

> **作者:** Akash Chikhalikar; Ankit A. Ravankar; Jose Victorio Salazar Luces; Yasuhisa Hirata
>
> **备注:** 8 pages, 9 figures. Accepted for publication in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems
>
> **摘要:** Personalization is critical for the advancement of service robots. Robots need to develop tailored understandings of the environments they are put in. Moreover, they need to be aware of changes in the environment to facilitate long-term deployment. Long-term understanding as well as personalization is necessary to execute complex tasks like prepare dinner table or tidy my room. A precursor to such tasks is that of Object Search. Consequently, this paper focuses on locating and searching multiple objects in indoor environments. In this paper, we propose two crucial novelties. Firstly, we propose a novel framework that can enable robots to deduce Personalized Ontologies of indoor environments. Our framework consists of a personalization schema that enables the robot to tune its understanding of ontologies. Secondly, we propose an Adaptive Inferencing strategy. We integrate Dynamic Belief Updates into our approach which improves performance in multi-object search tasks. The cumulative effect of personalization and adaptive inferencing is an improved capability in long-term object search. This framework is implemented on top of a multi-layered semantic map. We conduct experiments in real environments and compare our results against various state-of-the-art (SOTA) methods to demonstrate the effectiveness of our approach. Additionally, we show that personalization can act as a catalyst to enhance the performance of SOTAs. Video Link: https://bit.ly/3WHk9i9
>
---
#### [new 026] Diffusion-based Inverse Observation Model for Artificial Skin
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.13986v1](http://arxiv.org/pdf/2506.13986v1)**

> **作者:** Ante Maric; Julius Jankowski; Giammarco Caroleo; Alessandro Albini; Perla Maiolino; Sylvain Calinon
>
> **备注:** Accepted to RSS 2025 workshop on Navigating Contact Dynamics in Robotics
>
> **摘要:** Contact-based estimation of object pose is challenging due to discontinuities and ambiguous observations that can correspond to multiple possible system states. This multimodality makes it difficult to efficiently sample valid hypotheses while respecting contact constraints. Diffusion models can learn to generate samples from such multimodal probability distributions through denoising algorithms. We leverage these probabilistic modeling capabilities to learn an inverse observation model conditioned on tactile measurements acquired from a distributed artificial skin. We present simulated experiments demonstrating efficient sampling of contact hypotheses for object pose estimation through touch.
>
---
#### [new 027] Can Pretrained Vision-Language Embeddings Alone Guide Robot Navigation?
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，探讨是否仅用预训练视觉-语言嵌入即可引导机器人导航。工作是通过行为克隆策略验证其可行性，并分析其性能与局限。**

- **链接: [http://arxiv.org/pdf/2506.14507v1](http://arxiv.org/pdf/2506.14507v1)**

> **作者:** Nitesh Subedi; Adam Haroon; Shreyan Ganguly; Samuel T. K. Tetteh; Prajwal Koirala; Cody Fleming; Soumik Sarkar
>
> **备注:** 6 figures, 2 tables, Accepted to Robotics: Science and Systems (RSS) 2025 Workshop on Robot Planning in the Era of Foundation Models (FM4RoboPlan)
>
> **摘要:** Foundation models have revolutionized robotics by providing rich semantic representations without task-specific training. While many approaches integrate pretrained vision-language models (VLMs) with specialized navigation architectures, the fundamental question remains: can these pretrained embeddings alone successfully guide navigation without additional fine-tuning or specialized modules? We present a minimalist framework that decouples this question by training a behavior cloning policy directly on frozen vision-language embeddings from demonstrations collected by a privileged expert. Our approach achieves a 74% success rate in navigation to language-specified targets, compared to 100% for the state-aware expert, though requiring 3.2 times more steps on average. This performance gap reveals that pretrained embeddings effectively support basic language grounding but struggle with long-horizon planning and spatial reasoning. By providing this empirical baseline, we highlight both the capabilities and limitations of using foundation models as drop-in representations for embodied tasks, offering critical insights for robotics researchers facing practical design tradeoffs between system complexity and performance in resource-constrained scenarios. Our code is available at https://github.com/oadamharoon/text2nav
>
---
#### [new 028] DynaGuide: Steering Diffusion Polices with Active Dynamic Guidance
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决复杂策略在现实中的适应性问题。提出DynaGuide方法，通过外部动态模型引导扩散策略，提升策略的可引导性和多样性。**

- **链接: [http://arxiv.org/pdf/2506.13922v1](http://arxiv.org/pdf/2506.13922v1)**

> **作者:** Maximilian Du; Shuran Song
>
> **备注:** 9 pages main, 21 pages with appendix and citations. 9 figures. Submitted to Neurips 2025
>
> **摘要:** Deploying large, complex policies in the real world requires the ability to steer them to fit the needs of a situation. Most common steering approaches, like goal-conditioning, require training the robot policy with a distribution of test-time objectives in mind. To overcome this limitation, we present DynaGuide, a steering method for diffusion policies using guidance from an external dynamics model during the diffusion denoising process. DynaGuide separates the dynamics model from the base policy, which gives it multiple advantages, including the ability to steer towards multiple objectives, enhance underrepresented base policy behaviors, and maintain robustness on low-quality objectives. The separate guidance signal also allows DynaGuide to work with off-the-shelf pretrained diffusion policies. We demonstrate the performance and features of DynaGuide against other steering approaches in a series of simulated and real experiments, showing an average steering success of 70% on a set of articulated CALVIN tasks and outperforming goal-conditioning by 5.4x when steered with low-quality objectives. We also successfully steer an off-the-shelf real robot policy to express preference for particular objects and even create novel behavior. Videos and more can be found on the project website: https://dynaguide.github.io
>
---
#### [new 029] GAF: Gaussian Action Field as a Dvnamic World Model for Robotic Mlanipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决视觉引导下动作推断不准确的问题。提出GAF框架，通过4D动态表示提升动作推理精度。**

- **链接: [http://arxiv.org/pdf/2506.14135v1](http://arxiv.org/pdf/2506.14135v1)**

> **作者:** Ying Chai; Litao Deng; Ruizhi Shao; Jiajun Zhang; Liangjun Xing; Hongwen Zhang; Yebin Liu
>
> **备注:** http://chaiying1.github.io/GAF.github.io/project_page/
>
> **摘要:** Accurate action inference is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we propose a V-4D-A framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing simultaneous modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF supports three key query types: reconstruction of the current scene, prediction of future frames, and estimation of initial action via robot motion. Furthermore, the high-quality current and future frames generated by GAF facilitate manipulation action refinement through a GAF-guided diffusion model. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average success rate in robotic manipulation tasks by 10.33% over state-of-the-art methods. Project page: http://chaiying1.github.io/GAF.github.io/project_page/
>
---
#### [new 030] GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决无人机在复杂环境中根据自然语言指令自主导航的问题。通过轻量级模型和高效学习方法，实现可靠的语言引导飞行。**

- **链接: [http://arxiv.org/pdf/2506.14009v1](http://arxiv.org/pdf/2506.14009v1)**

> **作者:** Qianzhong Chen; Naixiang Gao; Suning Huang; JunEn Low; Timothy Chen; Jiankai Sun; Mac Schwager
>
> **摘要:** Autonomous drones capable of interpreting and executing high-level language instructions in unstructured environments remain a long-standing goal. Yet existing approaches are constrained by their dependence on hand-crafted skills, extensive parameter tuning, or computationally intensive models unsuitable for onboard use. We introduce GRaD-Nav++, a lightweight Vision-Language-Action (VLA) framework that runs fully onboard and follows natural-language commands in real time. Our policy is trained in a photorealistic 3D Gaussian Splatting (3DGS) simulator via Differentiable Reinforcement Learning (DiffRL), enabling efficient learning of low-level control from visual and linguistic inputs. At its core is a Mixture-of-Experts (MoE) action head, which adaptively routes computation to improve generalization while mitigating forgetting. In multi-task generalization experiments, GRaD-Nav++ achieves a success rate of 83% on trained tasks and 75% on unseen tasks in simulation. When deployed on real hardware, it attains 67% success on trained tasks and 50% on unseen ones. In multi-environment adaptation experiments, GRaD-Nav++ achieves an average success rate of 81% across diverse simulated environments and 67% across varied real-world settings. These results establish a new benchmark for fully onboard Vision-Language-Action (VLA) flight and demonstrate that compact, efficient models can enable reliable, language-guided navigation without relying on external infrastructure.
>
---
#### [new 031] Socially Aware Robot Crowd Navigation via Online Uncertainty-Driven Risk Adaptation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决复杂人群中的安全与社交意识问题。提出LR-MPC算法，结合风险学习与在线适应，提升导航效率与社会兼容性。**

- **链接: [http://arxiv.org/pdf/2506.14305v1](http://arxiv.org/pdf/2506.14305v1)**

> **作者:** Zhirui Sun; Xingrong Diao; Yao Wang; Bi-Ke Zhu; Jiankun Wang
>
> **摘要:** Navigation in human-robot shared crowded environments remains challenging, as robots are expected to move efficiently while respecting human motion conventions. However, many existing approaches emphasize safety or efficiency while overlooking social awareness. This article proposes Learning-Risk Model Predictive Control (LR-MPC), a data-driven navigation algorithm that balances efficiency, safety, and social awareness. LR-MPC consists of two phases: an offline risk learning phase, where a Probabilistic Ensemble Neural Network (PENN) is trained using risk data from a heuristic MPC-based baseline (HR-MPC), and an online adaptive inference phase, where local waypoints are sampled and globally guided by a Multi-RRT planner. Each candidate waypoint is evaluated for risk by PENN, and predictions are filtered using epistemic and aleatoric uncertainty to ensure robust decision-making. The safest waypoint is selected as the MPC input for real-time navigation. Extensive experiments demonstrate that LR-MPC outperforms baseline methods in success rate and social awareness, enabling robots to navigate complex crowds with high adaptability and low disruption. A website about this work is available at https://sites.google.com/view/lr-mpc.
>
---
#### [new 032] Public Acceptance of Cybernetic Avatars in the service sector: Evidence from a Large-Scale Survey in Dubai
- **分类: cs.RO; cs.ET; cs.HC**

- **简介: 该论文属于人机交互研究，探讨迪拜社会对服务领域仿生虚拟人的接受度。通过大规模调查分析不同外观、场景和任务下的接受情况，旨在提升技术的社会接受度。**

- **链接: [http://arxiv.org/pdf/2506.14268v1](http://arxiv.org/pdf/2506.14268v1)**

> **作者:** Laura Aymerich-Franch; Tarek Taha; Takahiro Miyashita; Hiroko Kamide; Hiroshi Ishiguro; Paolo Dario
>
> **备注:** 25 pages, 3 Figures
>
> **摘要:** Cybernetic avatars are hybrid interaction robots or digital representations that combine autonomous capabilities with teleoperated control. This study investigates the acceptance of cybernetic avatars in the highly multicultural society of Dubai, with particular emphasis on robotic avatars for customer service. Specifically, we explore how acceptance varies as a function of robot appearance (e.g., android, robotic-looking, cartoonish), deployment settings (e.g., shopping malls, hotels, hospitals), and functional tasks (e.g., providing information, patrolling). To this end, we conducted a large-scale survey with over 1,000 participants. Overall, cybernetic avatars received a high level of acceptance, with physical robot avatars receiving higher acceptance than digital avatars. In terms of appearance, robot avatars with a highly anthropomorphic robotic appearance were the most accepted, followed by cartoonish designs and androids. Animal-like appearances received the lowest level of acceptance. Among the tasks, providing information and guidance was rated as the most valued. Shopping malls, airports, public transport stations, and museums were the settings with the highest acceptance, whereas healthcare-related spaces received lower levels of support. An analysis by community cluster revealed among others that Emirati respondents showed significantly greater acceptance of android appearances compared to the overall sample, while participants from the 'Other Asia' cluster were significantly more accepting of cartoonish appearances. Our study underscores the importance of incorporating citizen feedback into the design and deployment of cybernetic avatars from the early stages to enhance acceptance of this technology in society.
>
---
#### [new 033] GMT: General Motion Tracking for Humanoid Whole-Body Control
- **分类: cs.RO**

- **简介: 该论文属于机器人运动跟踪任务，旨在解决人形机器人在真实世界中跟踪多样化全身运动的问题。提出GMT框架，结合自适应采样和混合专家架构，实现统一策略的高效运动跟踪。**

- **链接: [http://arxiv.org/pdf/2506.14770v1](http://arxiv.org/pdf/2506.14770v1)**

> **作者:** Zixuan Chen; Mazeyu Ji; Xuxin Cheng; Xuanbin Peng; Xue Bin Peng; Xiaolong Wang
>
> **摘要:** The ability to track general whole-body motions in the real world is a useful way to build general-purpose humanoid robots. However, achieving this can be challenging due to the temporal and kinematic diversity of the motions, the policy's capability, and the difficulty of coordination of the upper and lower bodies. To address these issues, we propose GMT, a general and scalable motion-tracking framework that trains a single unified policy to enable humanoid robots to track diverse motions in the real world. GMT is built upon two core components: an Adaptive Sampling strategy and a Motion Mixture-of-Experts (MoE) architecture. The Adaptive Sampling automatically balances easy and difficult motions during training. The MoE ensures better specialization of different regions of the motion manifold. We show through extensive experiments in both simulation and the real world the effectiveness of GMT, achieving state-of-the-art performance across a broad spectrum of motions using a unified general policy. Videos and additional information can be found at https://gmt-humanoid.github.io.
>
---
#### [new 034] Pose State Perception of Interventional Robot for Cardio-cerebrovascular Procedures
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于医疗机器人姿态感知任务，旨在解决血管介入手术中机器人精确定位问题。通过视觉方法实现机器人姿态的准确识别与控制。**

- **链接: [http://arxiv.org/pdf/2506.14201v1](http://arxiv.org/pdf/2506.14201v1)**

> **作者:** Shunhan Ji; Yanxi Chen; Zhongyu Yang; Quan Zhang; Xiaohang Nie; Jingqian Sun; Yichao Tang
>
> **摘要:** In response to the increasing demand for cardiocerebrovascular interventional surgeries, precise control of interventional robots has become increasingly important. Within these complex vascular scenarios, the accurate and reliable perception of the pose state for interventional robots is particularly crucial. This paper presents a novel vision-based approach without the need of additional sensors or markers. The core of this paper's method consists of a three-part framework: firstly, a dual-head multitask U-Net model for simultaneous vessel segment and interventional robot detection; secondly, an advanced algorithm for skeleton extraction and optimization; and finally, a comprehensive pose state perception system based on geometric features is implemented to accurately identify the robot's pose state and provide strategies for subsequent control. The experimental results demonstrate the proposed method's high reliability and accuracy in trajectory tracking and pose state perception.
>
---
#### [new 035] Non-Overlap-Aware Egocentric Pose Estimation for Collaborative Perception in Connected Autonomy
- **分类: cs.RO**

- **简介: 该论文属于多机器人协作感知任务，解决非重叠视角下的自车位姿估计问题。提出NOPE方法，在有限通信下实现准确位姿估计。**

- **链接: [http://arxiv.org/pdf/2506.14180v1](http://arxiv.org/pdf/2506.14180v1)**

> **作者:** Hong Huang; Dongkuan Xu; Hao Zhang; Peng Gao
>
> **备注:** IROS 2025
>
> **摘要:** Egocentric pose estimation is a fundamental capability for multi-robot collaborative perception in connected autonomy, such as connected autonomous vehicles. During multi-robot operations, a robot needs to know the relative pose between itself and its teammates with respect to its own coordinates. However, different robots usually observe completely different views that contains similar objects, which leads to wrong pose estimation. In addition, it is unrealistic to allow robots to share their raw observations to detect overlap due to the limited communication bandwidth constraint. In this paper, we introduce a novel method for Non-Overlap-Aware Egocentric Pose Estimation (NOPE), which performs egocentric pose estimation in a multi-robot team while identifying the non-overlap views and satifying the communication bandwidth constraint. NOPE is built upon an unified hierarchical learning framework that integrates two levels of robot learning: (1) high-level deep graph matching for correspondence identification, which allows to identify if two views are overlapping or not, (2) low-level position-aware cross-attention graph learning for egocentric pose estimation. To evaluate NOPE, we conduct extensive experiments in both high-fidelity simulation and real-world scenarios. Experimental results have demonstrated that NOPE enables the novel capability for non-overlapping-aware egocentric pose estimation and achieves state-of-art performance compared with the existing methods. Our project page at https://hongh0.github.io/NOPE/.
>
---
#### [new 036] Uncertainty-Driven Radar-Inertial Fusion for Instantaneous 3D Ego-Velocity Estimation
- **分类: cs.RO; cs.AI; eess.SP**

- **简介: 该论文属于自主导航中的运动估计任务，解决传统雷达测速精度不足的问题，通过融合雷达与惯性数据，利用神经网络估计速度及不确定性，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2506.14294v1](http://arxiv.org/pdf/2506.14294v1)**

> **作者:** Prashant Kumar Rai; Elham Kowsari; Nataliya Strokina; Reza Ghabcheloo
>
> **备注:** This paper has been accepted for presentation at the 28th International Conference on Information Fusion (Fusion 2025)
>
> **摘要:** We present a method for estimating ego-velocity in autonomous navigation by integrating high-resolution imaging radar with an inertial measurement unit. The proposed approach addresses the limitations of traditional radar-based ego-motion estimation techniques by employing a neural network to process complex-valued raw radar data and estimate instantaneous linear ego-velocity along with its associated uncertainty. This uncertainty-aware velocity estimate is then integrated with inertial measurement unit data using an Extended Kalman Filter. The filter leverages the network-predicted uncertainty to refine the inertial sensor's noise and bias parameters, improving the overall robustness and accuracy of the ego-motion estimation. We evaluated the proposed method on the publicly available ColoRadar dataset. Our approach achieves significantly lower error compared to the closest publicly available method and also outperforms both instantaneous and scan matching-based techniques.
>
---
#### [new 037] A Hierarchical Test Platform for Vision Language Model (VLM)-Integrated Real-World Autonomous Driving
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶任务，旨在解决VLM在真实驾驶环境中的适配问题。提出一个分层测试平台，支持VLM集成系统的有效评估与实验。**

- **链接: [http://arxiv.org/pdf/2506.14100v1](http://arxiv.org/pdf/2506.14100v1)**

> **作者:** Yupeng Zhou; Can Cui; Juntong Peng; Zichong Yang; Juanwu Lu; Jitesh H Panchal; Bin Yao; Ziran Wang
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated notable promise in autonomous driving by offering the potential for multimodal reasoning through pretraining on extensive image-text pairs. However, adapting these models from broad web-scale data to the safety-critical context of driving presents a significant challenge, commonly referred to as domain shift. Existing simulation-based and dataset-driven evaluation methods, although valuable, often fail to capture the full complexity of real-world scenarios and cannot easily accommodate repeatable closed-loop testing with flexible scenario manipulation. In this paper, we introduce a hierarchical real-world test platform specifically designed to evaluate VLM-integrated autonomous driving systems. Our approach includes a modular, low-latency on-vehicle middleware that allows seamless incorporation of various VLMs, a clearly separated perception-planning-control architecture that can accommodate both VLM-based and conventional modules, and a configurable suite of real-world testing scenarios on a closed track that facilitates controlled yet authentic evaluations. We demonstrate the effectiveness of the proposed platform`s testing and evaluation ability with a case study involving a VLM-enabled autonomous vehicle, highlighting how our test framework supports robust experimentation under diverse conditions.
>
---
#### [new 038] Narrate2Nav: Real-Time Visual Navigation with Implicit Language Reasoning in Human-Centric Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉导航任务，旨在提升人机环境中的实时导航性能。针对VLM计算复杂和响应慢的问题，提出Narrate2Nav模型，结合视觉与语言信息实现高效导航。**

- **链接: [http://arxiv.org/pdf/2506.14233v1](http://arxiv.org/pdf/2506.14233v1)**

> **作者:** Amirreza Payandeh; Anuj Pokhrel; Daeun Song; Marcos Zampieri; Xuesu Xiao
>
> **摘要:** Large Vision-Language Models (VLMs) have demonstrated potential in enhancing mobile robot navigation in human-centric environments by understanding contextual cues, human intentions, and social dynamics while exhibiting reasoning capabilities. However, their computational complexity and limited sensitivity to continuous numerical data impede real-time performance and precise motion control. To this end, we propose Narrate2Nav, a novel real-time vision-action model that leverages a novel self-supervised learning framework based on the Barlow Twins redundancy reduction loss to embed implicit natural language reasoning, social cues, and human intentions within a visual encoder-enabling reasoning in the model's latent space rather than token space. The model combines RGB inputs, motion commands, and textual signals of scene context during training to bridge from robot observations to low-level motion commands for short-horizon point-goal navigation during deployment. Extensive evaluation of Narrate2Nav across various challenging scenarios in both offline unseen dataset and real-world experiments demonstrates an overall improvement of 52.94 percent and 41.67 percent, respectively, over the next best baseline. Additionally, qualitative comparative analysis of Narrate2Nav's visual encoder attention map against four other baselines demonstrates enhanced attention to navigation-critical scene elements, underscoring its effectiveness in human-centric navigation tasks.
>
---
#### [new 039] ros2 fanuc interface: Design and Evaluation of a Fanuc CRX Hardware Interface in ROS2
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决Fanuc CRX与ROS2的接口集成问题，实现运动规划与控制，验证了路径跟踪与避障性能。**

- **链接: [http://arxiv.org/pdf/2506.14487v1](http://arxiv.org/pdf/2506.14487v1)**

> **作者:** Paolo Franceschi; Marco Faroni; Stefano Baraldo; Anna Valente
>
> **摘要:** This paper introduces the ROS2 control and the Hardware Interface (HW) integration for the Fanuc CRX- robot family. It explains basic implementation details and communication protocols, and its integration with the Moveit2 motion planning library. We conducted a series of experiments to evaluate relevant performances in the robotics field. We tested the developed ros2_fanuc_interface for four relevant robotics cases: step response, trajectory tracking, collision avoidance integrated with Moveit2, and dynamic velocity scaling, respectively. Results show that, despite a non-negligible delay between command and feedback, the robot can track the defined path with negligible errors (if it complies with joint velocity limits), ensuring collision avoidance. Full code is open source and available at https://github.com/paolofrance/ros2_fanuc_interface.
>
---
#### [new 040] Socially-aware Object Transportation by a Mobile Manipulator in Static Planar Environments with Obstacles
- **分类: cs.RO**

- **简介: 该论文属于移动机械臂的社交导航任务，解决其在人群环境中安全运输物体的问题。提出基于Risk-RRT*的方法，协调移动与操作，实现避障和减少社交不适。**

- **链接: [http://arxiv.org/pdf/2506.13953v1](http://arxiv.org/pdf/2506.13953v1)**

> **作者:** Caio C. G. Ribeiro; Leonardo R. D. Paes; Douglas G. Macharet
>
> **备注:** Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (ROMAN)
>
> **摘要:** Socially-aware robotic navigation is essential in environments where humans and robots coexist, ensuring both safety and comfort. However, most existing approaches have been primarily developed for mobile robots, leaving a significant gap in research that addresses the unique challenges posed by mobile manipulators. In this paper, we tackle the challenge of navigating a robotic mobile manipulator, carrying a non-negligible load, within a static human-populated environment while adhering to social norms. Our goal is to develop a method that enables the robot to simultaneously manipulate an object and navigate between locations in a socially-aware manner. We propose an approach based on the Risk-RRT* framework that enables the coordinated actuation of both the mobile base and manipulator. This approach ensures collision-free navigation while adhering to human social preferences. We compared our approach in a simulated environment to socially-aware mobile-only methods applied to a mobile manipulator. The results highlight the necessity for mobile manipulator-specific techniques, with our method outperforming mobile-only approaches. Our method enabled the robot to navigate, transport an object, avoid collisions, and minimize social discomfort effectively.
>
---
#### [new 041] Hard Contacts with Soft Gradients: Refining Differentiable Simulators for Learning and Control
- **分类: cs.RO; cs.LG; cs.SY; eess.SY; I.2.9; I.2.6; I.6.4; G.1.6**

- **简介: 该论文属于机器人学习与控制任务，解决硬接触下梯度计算不准确的问题。通过改进模拟器和引入CFD机制，提升梯度质量并保持物理真实性。**

- **链接: [http://arxiv.org/pdf/2506.14186v1](http://arxiv.org/pdf/2506.14186v1)**

> **作者:** Anselm Paulus; A. René Geist; Pierre Schumacher; Vít Musil; Georg Martius
>
> **摘要:** Contact forces pose a major challenge for gradient-based optimization of robot dynamics as they introduce jumps in the system's velocities. Penalty-based simulators, such as MuJoCo, simplify gradient computation by softening the contact forces. However, realistically simulating hard contacts requires very stiff contact settings, which leads to incorrect gradients when using automatic differentiation. On the other hand, using non-stiff settings strongly increases the sim-to-real gap. We analyze the contact computation of penalty-based simulators to identify the causes of gradient errors. Then, we propose DiffMJX, which combines adaptive integration with MuJoCo XLA, to notably improve gradient quality in the presence of hard contacts. Finally, we address a key limitation of contact gradients: they vanish when objects do not touch. To overcome this, we introduce Contacts From Distance (CFD), a mechanism that enables the simulator to generate informative contact gradients even before objects are in contact. To preserve physical realism, we apply CFD only in the backward pass using a straight-through trick, allowing us to compute useful gradients without modifying the forward simulation.
>
---
#### [new 042] Lasso Gripper: A String Shooting-Retracting Mechanism for Shape-Adaptive Grasping
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决传统夹爪在抓取大型、异形或易损物体时的局限性。提出Lasso Gripper，通过弹射和回收绳索实现自适应抓取。**

- **链接: [http://arxiv.org/pdf/2506.14163v1](http://arxiv.org/pdf/2506.14163v1)**

> **作者:** Qiyuan Qiao; Yu Wang; Xiyu Fan; Peng Lu
>
> **备注:** 6 pages, 13 figures
>
> **摘要:** Handling oversized, variable-shaped, or delicate objects in transportation, grasping tasks is extremely challenging, mainly due to the limitations of the gripper's shape and size. This paper proposes a novel gripper, Lasso Gripper. Inspired by traditional tools like the lasso and the uurga, Lasso Gripper captures objects by launching and retracting a string. Contrary to antipodal grippers, which concentrate force on a limited area, Lasso Gripper applies uniform pressure along the length of the string for a more gentle grasp. The gripper is controlled by four motors-two for launching the string inward and two for launching it outward. By adjusting motor speeds, the size of the string loop can be tuned to accommodate objects of varying sizes, eliminating the limitations imposed by the maximum gripper separation distance. To address the issue of string tangling during rapid retraction, a specialized mechanism was incorporated. Additionally, a dynamic model was developed to estimate the string's curve, providing a foundation for the kinematic analysis of the workspace. In grasping experiments, Lasso Gripper, mounted on a robotic arm, successfully captured and transported a range of objects, including bull and horse figures as well as delicate vegetables. The demonstration video is available here: https://youtu.be/PV1J76mNP9Y.
>
---
#### [new 043] Casper: Inferring Diverse Intents for Assistive Teleoperation with Vision Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于辅助遥控任务，旨在解决机器人从用户输入中推断多样意图的问题。工作包括引入Casper系统，利用视觉语言模型实现意图推理和灵活操作。**

- **链接: [http://arxiv.org/pdf/2506.14727v1](http://arxiv.org/pdf/2506.14727v1)**

> **作者:** Huihan Liu; Rutav Shah; Shuijing Liu; Jack Pittenger; Mingyo Seo; Yuchen Cui; Yonatan Bisk; Roberto Martín-Martín; Yuke Zhu
>
> **摘要:** Assistive teleoperation, where control is shared between a human and a robot, enables efficient and intuitive human-robot collaboration in diverse and unstructured environments. A central challenge in real-world assistive teleoperation is for the robot to infer a wide range of human intentions from user control inputs and to assist users with correct actions. Existing methods are either confined to simple, predefined scenarios or restricted to task-specific data distributions at training, limiting their support for real-world assistance. We introduce Casper, an assistive teleoperation system that leverages commonsense knowledge embedded in pre-trained visual language models (VLMs) for real-time intent inference and flexible skill execution. Casper incorporates an open-world perception module for a generalized understanding of novel objects and scenes, a VLM-powered intent inference mechanism that leverages commonsense reasoning to interpret snippets of teleoperated user input, and a skill library that expands the scope of prior assistive teleoperation systems to support diverse, long-horizon mobile manipulation tasks. Extensive empirical evaluation, including human studies and system ablations, demonstrates that Casper improves task performance, reduces human cognitive load, and achieves higher user satisfaction than direct teleoperation and assistive teleoperation baselines.
>
---
#### [new 044] TACS-Graphs: Traversability-Aware Consistent Scene Graphs for Ground Robot Indoor Localization and Mapping
- **分类: cs.RO**

- **简介: 该论文属于室内定位与建图任务，解决场景图分割不一致问题，提出TACS-Graphs框架，结合可通行性提升分割一致性与定位精度。**

- **链接: [http://arxiv.org/pdf/2506.14178v1](http://arxiv.org/pdf/2506.14178v1)**

> **作者:** Jeewon Kim; Minho Oh; Hyun Myung
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Scene graphs have emerged as a powerful tool for robots, providing a structured representation of spatial and semantic relationships for advanced task planning. Despite their potential, conventional 3D indoor scene graphs face critical limitations, particularly under- and over-segmentation of room layers in structurally complex environments. Under-segmentation misclassifies non-traversable areas as part of a room, often in open spaces, while over-segmentation fragments a single room into overlapping segments in complex environments. These issues stem from naive voxel-based map representations that rely solely on geometric proximity, disregarding the structural constraints of traversable spaces and resulting in inconsistent room layers within scene graphs. To the best of our knowledge, this work is the first to tackle segmentation inconsistency as a challenge and address it with Traversability-Aware Consistent Scene Graphs (TACS-Graphs), a novel framework that integrates ground robot traversability with room segmentation. By leveraging traversability as a key factor in defining room boundaries, the proposed method achieves a more semantically meaningful and topologically coherent segmentation, effectively mitigating the inaccuracies of voxel-based scene graph approaches in complex environments. Furthermore, the enhanced segmentation consistency improves loop closure detection efficiency in the proposed Consistent Scene Graph-leveraging Loop Closure Detection (CoSG-LCD) leading to higher pose estimation accuracy. Experimental results confirm that the proposed approach outperforms state-of-the-art methods in terms of scene graph consistency and pose graph optimization performance.
>
---
#### [new 045] A Survey on World Models Grounded in Acoustic Physical Information
- **分类: cs.SD; cs.AI; cs.RO; eess.AS; physics.app-ph; 68T07, 35L05, 78A45; I.2.6; H.5.5; I.2.9**

- **简介: 该论文属于感知与建模任务，旨在利用声学物理信息构建高保真环境模型，解决动态事件预测与因果推理问题，提出相关方法并探讨其应用与挑战。**

- **链接: [http://arxiv.org/pdf/2506.13833v1](http://arxiv.org/pdf/2506.13833v1)**

> **作者:** Xiaoliang Chen; Le Chang; Xin Yu; Yunhe Huang; Xianling Tu
>
> **备注:** 28 pages,11 equations
>
> **摘要:** This survey provides a comprehensive overview of the emerging field of world models grounded in the foundation of acoustic physical information. It examines the theoretical underpinnings, essential methodological frameworks, and recent technological advancements in leveraging acoustic signals for high-fidelity environmental perception, causal physical reasoning, and predictive simulation of dynamic events. The survey explains how acoustic signals, as direct carriers of mechanical wave energy from physical events, encode rich, latent information about material properties, internal geometric structures, and complex interaction dynamics. Specifically, this survey establishes the theoretical foundation by explaining how fundamental physical laws govern the encoding of physical information within acoustic signals. It then reviews the core methodological pillars, including Physics-Informed Neural Networks (PINNs), generative models, and self-supervised multimodal learning frameworks. Furthermore, the survey details the significant applications of acoustic world models in robotics, autonomous driving, healthcare, and finance. Finally, it systematically outlines the important technical and ethical challenges while proposing a concrete roadmap for future research directions toward robust, causal, uncertainty-aware, and responsible acoustic intelligence. These elements collectively point to a research pathway towards embodied active acoustic intelligence, empowering AI systems to construct an internal "intuitive physics" engine through sound.
>
---
#### [new 046] DiFuse-Net: RGB and Dual-Pixel Depth Estimation using Window Bi-directional Parallax Attention and Cross-modal Transfer Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于深度估计任务，解决传统方法在成本和鲁棒性上的不足，提出DiFuse-Net模型结合RGB与双像素信息提升深度预测精度。**

- **链接: [http://arxiv.org/pdf/2506.14709v1](http://arxiv.org/pdf/2506.14709v1)**

> **作者:** Kunal Swami; Debtanu Gupta; Amrit Kumar Muduli; Chirag Jaiswal; Pankaj Kumar Bajpai
>
> **备注:** Accepted in IROS 2025
>
> **摘要:** Depth estimation is crucial for intelligent systems, enabling applications from autonomous navigation to augmented reality. While traditional stereo and active depth sensors have limitations in cost, power, and robustness, dual-pixel (DP) technology, ubiquitous in modern cameras, offers a compelling alternative. This paper introduces DiFuse-Net, a novel modality decoupled network design for disentangled RGB and DP based depth estimation. DiFuse-Net features a window bi-directional parallax attention mechanism (WBiPAM) specifically designed to capture the subtle DP disparity cues unique to smartphone cameras with small aperture. A separate encoder extracts contextual information from the RGB image, and these features are fused to enhance depth prediction. We also propose a Cross-modal Transfer Learning (CmTL) mechanism to utilize large-scale RGB-D datasets in the literature to cope with the limitations of obtaining large-scale RGB-DP-D dataset. Our evaluation and comparison of the proposed method demonstrates its superiority over the DP and stereo-based baseline methods. Additionally, we contribute a new, high-quality, real-world RGB-DP-D training dataset, named Dual-Camera Dual-Pixel (DCDP) dataset, created using our novel symmetric stereo camera hardware setup, stereo calibration and rectification protocol, and AI stereo disparity estimation method.
>
---
#### [new 047] Markov Regime-Switching Intelligent Driver Model for Interpretable Car-Following Behavior
- **分类: stat.AP; cs.LG; cs.RO**

- **简介: 该论文属于交通仿真与自动驾驶领域，旨在解决传统模型无法准确解释驾驶行为的问题。通过引入马尔可夫切换框架，提升模型对复杂驾驶行为的解释性和准确性。**

- **链接: [http://arxiv.org/pdf/2506.14762v1](http://arxiv.org/pdf/2506.14762v1)**

> **作者:** Chengyuan Zhang; Cathy Wu; Lijun Sun
>
> **摘要:** Accurate and interpretable car-following models are essential for traffic simulation and autonomous vehicle development. However, classical models like the Intelligent Driver Model (IDM) are fundamentally limited by their parsimonious and single-regime structure. They fail to capture the multi-modal nature of human driving, where a single driving state (e.g., speed, relative speed, and gap) can elicit many different driver actions. This forces the model to average across distinct behaviors, reducing its fidelity and making its parameters difficult to interpret. To overcome this, we introduce a regime-switching framework that allows driving behavior to be governed by different IDM parameter sets, each corresponding to an interpretable behavioral mode. This design enables the model to dynamically switch between interpretable behavioral modes, rather than averaging across diverse driving contexts. We instantiate the framework using a Factorial Hidden Markov Model with IDM dynamics (FHMM-IDM), which explicitly separates intrinsic driving regimes (e.g., aggressive acceleration, steady-state following) from external traffic scenarios (e.g., free-flow, congestion, stop-and-go) through two independent latent Markov processes. Bayesian inference via Markov chain Monte Carlo (MCMC) is used to jointly estimate the regime-specific parameters, transition dynamics, and latent state trajectories. Experiments on the HighD dataset demonstrate that FHMM-IDM uncovers interpretable structure in human driving, effectively disentangling internal driver actions from contextual traffic conditions and revealing dynamic regime-switching patterns. This framework provides a tractable and principled solution to modeling context-dependent driving behavior under uncertainty, offering improvements in the fidelity of traffic simulations, the efficacy of safety analyses, and the development of more human-centric ADAS.
>
---
#### [new 048] KDMOS:Knowledge Distillation for Motion Segmentation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于运动目标分割任务，旨在提升分割精度与实时性。通过知识蒸馏方法，优化模型结构，减少参数量，提高分割效果。**

- **链接: [http://arxiv.org/pdf/2506.14130v1](http://arxiv.org/pdf/2506.14130v1)**

> **作者:** Chunyu Cao; Jintao Cheng; Zeyu Chen; Linfan Zhan; Rui Fan; Zhijian He; Xiaoyu Tang
>
> **摘要:** Motion Object Segmentation (MOS) is crucial for autonomous driving, as it enhances localization, path planning, map construction, scene flow estimation, and future state prediction. While existing methods achieve strong performance, balancing accuracy and real-time inference remains a challenge. To address this, we propose a logits-based knowledge distillation framework for MOS, aiming to improve accuracy while maintaining real-time efficiency. Specifically, we adopt a Bird's Eye View (BEV) projection-based model as the student and a non-projection model as the teacher. To handle the severe imbalance between moving and non-moving classes, we decouple them and apply tailored distillation strategies, allowing the teacher model to better learn key motion-related features. This approach significantly reduces false positives and false negatives. Additionally, we introduce dynamic upsampling, optimize the network architecture, and achieve a 7.69% reduction in parameter count, mitigating overfitting. Our method achieves a notable IoU of 78.8% on the hidden test set of the SemanticKITTI-MOS dataset and delivers competitive results on the Apollo dataset. The KDMOS implementation is available at https://github.com/SCNU-RISLAB/KDMOS.
>
---
#### [new 049] Scaling Algorithm Distillation for Continuous Control with Mamba
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决ICRL在连续控制中的效率问题，通过引入Mamba模型提升长序列建模能力，实现更优的算法蒸馏效果。**

- **链接: [http://arxiv.org/pdf/2506.13892v1](http://arxiv.org/pdf/2506.13892v1)**

> **作者:** Samuel Beaussant; Mehdi Mounsif
>
> **摘要:** Algorithm Distillation (AD) was recently proposed as a new approach to perform In-Context Reinforcement Learning (ICRL) by modeling across-episodic training histories autoregressively with a causal transformer model. However, due to practical limitations induced by the attention mechanism, experiments were bottlenecked by the transformer's quadratic complexity and limited to simple discrete environments with short time horizons. In this work, we propose leveraging the recently proposed Selective Structured State Space Sequence (S6) models, which achieved state-of-the-art (SOTA) performance on long-range sequence modeling while scaling linearly in sequence length. Through four complex and continuous Meta Reinforcement Learning environments, we demonstrate the overall superiority of Mamba, a model built with S6 layers, over a transformer model for AD. Additionally, we show that scaling AD to very long contexts can improve ICRL performance and make it competitive even with a SOTA online meta RL baseline.
>
---
#### [new 050] A Novel Indicator for Quantifying and Minimizing Information Utility Loss of Robot Teams
- **分类: cs.DC; cs.RO**

- **简介: 该论文属于机器人协作任务，旨在解决信息传输受限导致的协作效率问题。提出LoIU指标并优化传输策略以提升信息新鲜度与实用性。**

- **链接: [http://arxiv.org/pdf/2506.14237v1](http://arxiv.org/pdf/2506.14237v1)**

> **作者:** Xiyu Zhao; Qimei Cui; Wei Ni; Quan Z. Sheng; Abbas Jamalipour; Guoshun Nan; Xiaofeng Tao; Ping Zhang
>
> **摘要:** The timely exchange of information among robots within a team is vital, but it can be constrained by limited wireless capacity. The inability to deliver information promptly can result in estimation errors that impact collaborative efforts among robots. In this paper, we propose a new metric termed Loss of Information Utility (LoIU) to quantify the freshness and utility of information critical for cooperation. The metric enables robots to prioritize information transmissions within bandwidth constraints. We also propose the estimation of LoIU using belief distributions and accordingly optimize both transmission schedule and resource allocation strategy for device-to-device transmissions to minimize the time-average LoIU within a robot team. A semi-decentralized Multi-Agent Deep Deterministic Policy Gradient framework is developed, where each robot functions as an actor responsible for scheduling transmissions among its collaborators while a central critic periodically evaluates and refines the actors in response to mobility and interference. Simulations validate the effectiveness of our approach, demonstrating an enhancement of information freshness and utility by 98%, compared to alternative methods.
>
---
#### [new 051] ASMR: Augmenting Life Scenario using Large Generative Models for Robotic Action Reflection
- **分类: cs.CL; cs.AI; cs.RO**

- **简介: 该论文属于多模态分类任务，旨在解决机器人理解用户意图时数据不足的问题。通过生成对话和图像数据增强训练集，提升机器人动作选择能力。**

- **链接: [http://arxiv.org/pdf/2506.13956v1](http://arxiv.org/pdf/2506.13956v1)**

> **作者:** Shang-Chi Tsai; Seiya Kawano; Angel Garcia Contreras; Koichiro Yoshino; Yun-Nung Chen
>
> **备注:** IWSDS 2024 Best Paper Award
>
> **摘要:** When designing robots to assist in everyday human activities, it is crucial to enhance user requests with visual cues from their surroundings for improved intent understanding. This process is defined as a multimodal classification task. However, gathering a large-scale dataset encompassing both visual and linguistic elements for model training is challenging and time-consuming. To address this issue, our paper introduces a novel framework focusing on data augmentation in robotic assistance scenarios, encompassing both dialogues and related environmental imagery. This approach involves leveraging a sophisticated large language model to simulate potential conversations and environmental contexts, followed by the use of a stable diffusion model to create images depicting these environments. The additionally generated data serves to refine the latest multimodal models, enabling them to more accurately determine appropriate actions in response to user interactions with the limited target data. Our experimental results, based on a dataset collected from real-world scenarios, demonstrate that our methodology significantly enhances the robot's action selection capabilities, achieving the state-of-the-art performance.
>
---
#### [new 052] Adaptive Reinforcement Learning for Unobservable Random Delays
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决动态环境中不可观测随机延迟问题。提出交互层框架和ACDA算法，适应性处理时间变化的延迟。**

- **链接: [http://arxiv.org/pdf/2506.14411v1](http://arxiv.org/pdf/2506.14411v1)**

> **作者:** John Wikman; Alexandre Proutiere; David Broman
>
> **摘要:** In standard Reinforcement Learning (RL) settings, the interaction between the agent and the environment is typically modeled as a Markov Decision Process (MDP), which assumes that the agent observes the system state instantaneously, selects an action without delay, and executes it immediately. In real-world dynamic environments, such as cyber-physical systems, this assumption often breaks down due to delays in the interaction between the agent and the system. These delays can vary stochastically over time and are typically unobservable, meaning they are unknown when deciding on an action. Existing methods deal with this uncertainty conservatively by assuming a known fixed upper bound on the delay, even if the delay is often much lower. In this work, we introduce the interaction layer, a general framework that enables agents to adaptively and seamlessly handle unobservable and time-varying delays. Specifically, the agent generates a matrix of possible future actions to handle both unpredictable delays and lost action packets sent over networks. Building on this framework, we develop a model-based algorithm, Actor-Critic with Delay Adaptation (ACDA), which dynamically adjusts to delay patterns. Our method significantly outperforms state-of-the-art approaches across a wide range of locomotion benchmark environments.
>
---
#### [new 053] CDP: Towards Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人视觉-运动策略学习任务，解决数据质量差和实时性限制导致的策略失效问题。提出CDP模型，通过历史动作序列提升预测精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.14769v1](http://arxiv.org/pdf/2506.14769v1)**

> **作者:** Jiahua Ma; Yiran Qin; Yixiong Li; Xuanqi Liao; Yulan Guo; Ruimao Zhang
>
> **摘要:** Diffusion Policy (DP) enables robots to learn complex behaviors by imitating expert demonstrations through action diffusion. However, in practical applications, hardware limitations often degrade data quality, while real-time constraints restrict model inference to instantaneous state and scene observations. These limitations seriously reduce the efficacy of learning from expert demonstrations, resulting in failures in object localization, grasp planning, and long-horizon task execution. To address these challenges, we propose Causal Diffusion Policy (CDP), a novel transformer-based diffusion model that enhances action prediction by conditioning on historical action sequences, thereby enabling more coherent and context-aware visuomotor policy learning. To further mitigate the computational cost associated with autoregressive inference, a caching mechanism is also introduced to store attention key-value pairs from previous timesteps, substantially reducing redundant computations during execution. Extensive experiments in both simulated and real-world environments, spanning diverse 2D and 3D manipulation tasks, demonstrate that CDP uniquely leverages historical action sequences to achieve significantly higher accuracy than existing methods. Moreover, even when faced with degraded input observation quality, CDP maintains remarkable precision by reasoning through temporal continuity, which highlights its practical robustness for robotic control under realistic, imperfect conditions.
>
---
#### [new 054] VisLanding: Monocular 3D Perception for UAV Safe Landing via Depth-Normal Synergy
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机安全着陆任务，解决复杂环境中着陆区域识别问题。通过深度-法线协同优化，构建端到端安全着陆区估计框架。**

- **链接: [http://arxiv.org/pdf/2506.14525v1](http://arxiv.org/pdf/2506.14525v1)**

> **作者:** Zhuoyue Tan; Boyong He; Yuxiang Ji; Liaoni Wu
>
> **备注:** Accepted by IROS2025
>
> **摘要:** This paper presents VisLanding, a monocular 3D perception-based framework for safe UAV (Unmanned Aerial Vehicle) landing. Addressing the core challenge of autonomous UAV landing in complex and unknown environments, this study innovatively leverages the depth-normal synergy prediction capabilities of the Metric3D V2 model to construct an end-to-end safe landing zones (SLZ) estimation framework. By introducing a safe zone segmentation branch, we transform the landing zone estimation task into a binary semantic segmentation problem. The model is fine-tuned and annotated using the WildUAV dataset from a UAV perspective, while a cross-domain evaluation dataset is constructed to validate the model's robustness. Experimental results demonstrate that VisLanding significantly enhances the accuracy of safe zone identification through a depth-normal joint optimization mechanism, while retaining the zero-shot generalization advantages of Metric3D V2. The proposed method exhibits superior generalization and robustness in cross-domain testing compared to other approaches. Furthermore, it enables the estimation of landing zone area by integrating predicted depth and normal information, providing critical decision-making support for practical applications.
>
---
#### [new 055] AGENTSAFE: Benchmarking the Safety of Embodied Agents on Hazardous Instructions
- **分类: cs.CR; cs.RO**

- **简介: 该论文属于安全评估任务，旨在解决 embodied agents 在危险指令下的安全性问题。提出 AGENTSAFE 基准，包含风险指令数据集与对抗场景，用于系统测试与提升安全性。**

- **链接: [http://arxiv.org/pdf/2506.14697v1](http://arxiv.org/pdf/2506.14697v1)**

> **作者:** Aishan Liu; Zonghao Ying; Le Wang; Junjie Mu; Jinyang Guo; Jiakai Wang; Yuqing Ma; Siyuan Liang; Mingchuan Zhang; Xianglong Liu; Dacheng Tao
>
> **备注:** 11 pages
>
> **摘要:** The rapid advancement of vision-language models (VLMs) and their integration into embodied agents have unlocked powerful capabilities for decision-making. However, as these systems are increasingly deployed in real-world environments, they face mounting safety concerns, particularly when responding to hazardous instructions. In this work, we propose AGENTSAFE, the first comprehensive benchmark for evaluating the safety of embodied VLM agents under hazardous instructions. AGENTSAFE simulates realistic agent-environment interactions within a simulation sandbox and incorporates a novel adapter module that bridges the gap between high-level VLM outputs and low-level embodied controls. Specifically, it maps recognized visual entities to manipulable objects and translates abstract planning into executable atomic actions in the environment. Building on this, we construct a risk-aware instruction dataset inspired by Asimovs Three Laws of Robotics, including base risky instructions and mutated jailbroken instructions. The benchmark includes 45 adversarial scenarios, 1,350 hazardous tasks, and 8,100 hazardous instructions, enabling systematic testing under adversarial conditions ranging from perception, planning, and action execution stages.
>
---
## 更新

#### [replaced 001] Human-robot collaborative transport personalization via Dynamic Movement Primitives and velocity scaling
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.09697v2](http://arxiv.org/pdf/2506.09697v2)**

> **作者:** Paolo Franceschi; Andrea Bussolan; Vincenzo Pomponi; Oliver Avram; Stefano Baraldo; Anna Valente
>
> **摘要:** Nowadays, industries are showing a growing interest in human-robot collaboration, particularly for shared tasks. This requires intelligent strategies to plan a robot's motions, considering both task constraints and human-specific factors such as height and movement preferences. This work introduces a novel approach to generate personalized trajectories using Dynamic Movement Primitives (DMPs), enhanced with real-time velocity scaling based on human feedback. The method was rigorously tested in industrial-grade experiments, focusing on the collaborative transport of an engine cowl lip section. Comparative analysis between DMP-generated trajectories and a state-of-the-art motion planner (BiTRRT) highlights their adaptability combined with velocity scaling. Subjective user feedback further demonstrates a clear preference for DMP- based interactions. Objective evaluations, including physiological measurements from brain and skin activity, reinforce these findings, showcasing the advantages of DMPs in enhancing human-robot interaction and improving user experience.
>
---
#### [replaced 002] IKDiffuser: Fast and Diverse Inverse Kinematics Solution Generation for Multi-arm Robotic Systems
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.13087v2](http://arxiv.org/pdf/2506.13087v2)**

> **作者:** Zeyu Zhang; Ziyuan Jiao
>
> **备注:** under review
>
> **摘要:** Solving Inverse Kinematics (IK) problems is fundamental to robotics, but has primarily been successful with single serial manipulators. For multi-arm robotic systems, IK remains challenging due to complex self-collisions, coupled joints, and high-dimensional redundancy. These complexities make traditional IK solvers slow, prone to failure, and lacking in solution diversity. In this paper, we present IKDiffuser, a diffusion-based model designed for fast and diverse IK solution generation for multi-arm robotic systems. IKDiffuser learns the joint distribution over the configuration space, capturing complex dependencies and enabling seamless generalization to multi-arm robotic systems of different structures. In addition, IKDiffuser can incorporate additional objectives during inference without retraining, offering versatility and adaptability for task-specific requirements. In experiments on 6 different multi-arm systems, the proposed IKDiffuser achieves superior solution accuracy, precision, diversity, and computational efficiency compared to existing solvers. The proposed IKDiffuser framework offers a scalable, unified approach to solving multi-arm IK problems, facilitating the potential of multi-arm robotic systems in real-time manipulation tasks.
>
---
#### [replaced 003] Design and Evaluation of an Uncertainty-Aware Shared-Autonomy System with Hierarchical Conservative Skill Inference
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2312.02488v2](http://arxiv.org/pdf/2312.02488v2)**

> **作者:** Taewoo Kim; Donghyung Kim; Minsu Jang; Jaehong Kim
>
> **备注:** ArXiv 2024
>
> **摘要:** Shared-autonomy imitation learning lets a human correct a robot in real time, mitigating covariate-shift errors. Yet existing approaches ignore two critical factors: (i) the operator's cognitive load and (ii) the risk created by delayed or erroneous interventions. We present an uncertainty-aware shared-autonomy system in which the robot modulates its behaviour according to a learned estimate of latent-space skill uncertainty. A hierarchical policy first infers a conservative skill embedding and then decodes it into low-level actions, enabling rapid task execution while automatically slowing down when uncertainty is high. We detail a full, open-source VR-teleoperation pipeline that is compatible with multi-configuration manipulators such as UR-series arms. Experiments on pouring and pick-and-place tasks demonstrate 70-90% success in dynamic scenes with moving targets, and a qualitative study shows a marked reduction in collision events compared with a non-conservative baseline. Although a dedicated ablation that isolates uncertainty is impractical on hardware for safety and cost reasons, the reported gains in stability and operator workload already validate the design and motivate future large-scale studies.
>
---
#### [replaced 004] ClearDepth: Enhanced Stereo Perception of Transparent Objects for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.08926v2](http://arxiv.org/pdf/2409.08926v2)**

> **作者:** Kaixin Bai; Huajian Zeng; Lei Zhang; Yiwen Liu; Hongli Xu; Zhaopeng Chen; Jianwei Zhang
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** Transparent object depth perception poses a challenge in everyday life and logistics, primarily due to the inability of standard 3D sensors to accurately capture depth on transparent or reflective surfaces. This limitation significantly affects depth map and point cloud-reliant applications, especially in robotic manipulation. We developed a vision transformer-based algorithm for stereo depth recovery of transparent objects. This approach is complemented by an innovative feature post-fusion module, which enhances the accuracy of depth recovery by structural features in images. To address the high costs associated with dataset collection for stereo camera-based perception of transparent objects, our method incorporates a parameter-aligned, domain-adaptive, and physically realistic Sim2Real simulation for efficient data generation, accelerated by AI algorithm. Our experimental results demonstrate the model's exceptional Sim2Real generalizability in real-world scenarios, enabling precise depth mapping of transparent objects to assist in robotic manipulation. Project details are available at https://sites.google.com/view/cleardepth/ .
>
---
#### [replaced 005] Fast Contact Detection via Fusion of Joint and Inertial Sensors for Parallel Robots in Human-Robot Collaboration
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.08334v2](http://arxiv.org/pdf/2505.08334v2)**

> **作者:** Aran Mohammad; Jan Piosik; Dustin Lehmann; Thomas Seel; Moritz Schappler
>
> **备注:** Preprint of a publication accepted for IEEE Robotics and Automation Letters
>
> **摘要:** Fast contact detection is crucial for safe human-robot collaboration. Observers based on proprioceptive information can be used for contact detection but have first-order error dynamics, which results in delays. Sensor fusion based on inertial measurement units (IMUs) consisting of accelerometers and gyroscopes is advantageous for reducing delays. The acceleration estimation enables the direct calculation of external forces. For serial robots, the installation of multiple accelerometers and gyroscopes is required for dynamics modeling since the joint coordinates are the minimal coordinates. Alternatively, parallel robots (PRs) offer the potential to use only one IMU on the end-effector platform, which already presents the minimal coordinates of the PR. This work introduces a sensor-fusion method for contact detection using encoders and only one low-cost, consumer-grade IMU for a PR. The end-effector accelerations are estimated by an extended Kalman filter and incorporated into the dynamics to calculate external forces. In real-world experiments with a planar PR, we demonstrate that this approach reduces the detection duration by up to 50% compared to a momentum observer and enables the collision and clamping detection within 3-39ms.
>
---
#### [replaced 006] SRT-H: A Hierarchical Framework for Autonomous Surgery via Language Conditioned Imitation Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.10251v2](http://arxiv.org/pdf/2505.10251v2)**

> **作者:** Ji Woong Kim; Juo-Tung Chen; Pascal Hansen; Lucy X. Shi; Antony Goldenberg; Samuel Schmidgall; Paul Maria Scheikl; Anton Deguet; Brandon M. White; De Ru Tsai; Richard Cha; Jeffrey Jopling; Chelsea Finn; Axel Krieger
>
> **摘要:** Research on autonomous surgery has largely focused on simple task automation in controlled environments. However, real-world surgical applications demand dexterous manipulation over extended durations and robust generalization to the inherent variability of human tissue. These challenges remain difficult to address using existing logic-based or conventional end-to-end learning strategies. To address this gap, we propose a hierarchical framework for performing dexterous, long-horizon surgical steps. Our approach utilizes a high-level policy for task planning and a low-level policy for generating low-level trajectories. The high-level planner plans in language space, generating task or corrective instructions to guide the robot through the long-horizon steps and correct for the low-level policy's errors. We validate our framework through ex vivo experiments on cholecystectomy, a commonly-practiced minimally invasive procedure, and conduct ablation studies to evaluate key components of the system. Our method achieves a 100% success rate across n=8 different ex vivo gallbladders, operating fully autonomously without human intervention. The hierarchical approach improves the policy's ability to recover from suboptimal states that are inevitable in the highly dynamic environment of realistic surgical applications. This work demonstrates step-level autonomy in a surgical procedure, marking a milestone toward clinical deployment of autonomous surgical systems.
>
---
#### [replaced 007] H$^3$DP: Triply-Hierarchical Diffusion Policy for Visuomotor Learning
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07819v2](http://arxiv.org/pdf/2505.07819v2)**

> **作者:** Yiyang Lu; Yufeng Tian; Zhecheng Yuan; Xianbang Wang; Pu Hua; Zhengrong Xue; Huazhe Xu
>
> **摘要:** Visuomotor policy learning has witnessed substantial progress in robotic manipulation, with recent approaches predominantly relying on generative models to model the action distribution. However, these methods often overlook the critical coupling between visual perception and action prediction. In this work, we introduce $\textbf{Triply-Hierarchical Diffusion Policy}~(\textbf{H$^{\mathbf{3}}$DP})$, a novel visuomotor learning framework that explicitly incorporates hierarchical structures to strengthen the integration between visual features and action generation. H$^{3}$DP contains $\mathbf{3}$ levels of hierarchy: (1) depth-aware input layering that organizes RGB-D observations based on depth information; (2) multi-scale visual representations that encode semantic features at varying levels of granularity; and (3) a hierarchically conditioned diffusion process that aligns the generation of coarse-to-fine actions with corresponding visual features. Extensive experiments demonstrate that H$^{3}$DP yields a $\mathbf{+27.5\%}$ average relative improvement over baselines across $\mathbf{44}$ simulation tasks and achieves superior performance in $\mathbf{4}$ challenging bimanual real-world manipulation tasks. Project Page: https://lyy-iiis.github.io/h3dp/.
>
---
#### [replaced 008] NGD-SLAM: Towards Real-Time Dynamic SLAM without GPU
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.07392v3](http://arxiv.org/pdf/2405.07392v3)**

> **作者:** Yuhao Zhang; Mihai Bujanca; Mikel Luján
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Many existing visual SLAM methods can achieve high localization accuracy in dynamic environments by leveraging deep learning to mask moving objects. However, these methods incur significant computational overhead as the camera tracking needs to wait for the deep neural network to generate mask at each frame, and they typically require GPUs for real-time operation, which restricts their practicality in real-world robotic applications. Therefore, this paper proposes a real-time dynamic SLAM system that runs exclusively on a CPU. Our approach incorporates a mask propagation mechanism that decouples camera tracking and deep learning-based masking for each frame. We also introduce a hybrid tracking strategy that integrates ORB features with optical flow methods, enhancing both robustness and efficiency by selectively allocating computational resources to input frames. Compared to previous methods, our system maintains high localization accuracy in dynamic environments while achieving a tracking frame rate of 60 FPS on a laptop CPU. These results demonstrate the feasibility of utilizing deep learning for dynamic SLAM without GPU support. Since most existing dynamic SLAM systems are not open-source, we make our code publicly available at: https://github.com/yuhaozhang7/NGD-SLAM
>
---
#### [replaced 009] SmartWay: Enhanced Waypoint Prediction and Backtracking for Zero-Shot Vision-and-Language Navigation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.10069v2](http://arxiv.org/pdf/2503.10069v2)**

> **作者:** Xiangyu Shi; Zerui Li; Wenqi Lyu; Jiatong Xia; Feras Dayoub; Yanyuan Qiao; Qi Wu
>
> **备注:** Accepted by IROS 2025. Project website: https://sxyxs.github.io/smartway/
>
> **摘要:** Vision-and-Language Navigation (VLN) in continuous environments requires agents to interpret natural language instructions while navigating unconstrained 3D spaces. Existing VLN-CE frameworks rely on a two-stage approach: a waypoint predictor to generate waypoints and a navigator to execute movements. However, current waypoint predictors struggle with spatial awareness, while navigators lack historical reasoning and backtracking capabilities, limiting adaptability. We propose a zero-shot VLN-CE framework integrating an enhanced waypoint predictor with a Multi-modal Large Language Model (MLLM)-based navigator. Our predictor employs a stronger vision encoder, masked cross-attention fusion, and an occupancy-aware loss for better waypoint quality. The navigator incorporates history-aware reasoning and adaptive path planning with backtracking, improving robustness. Experiments on R2R-CE and MP3D benchmarks show our method achieves state-of-the-art (SOTA) performance in zero-shot settings, demonstrating competitive results compared to fully supervised methods. Real-world validation on Turtlebot 4 further highlights its adaptability.
>
---
#### [replaced 010] PartInstruct: Part-level Instruction Following for Fine-grained Robot Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21652v3](http://arxiv.org/pdf/2505.21652v3)**

> **作者:** Yifan Yin; Zhengtao Han; Shivam Aarya; Jianxin Wang; Shuhang Xu; Jiawei Peng; Angtian Wang; Alan Yuille; Tianmin Shu
>
> **摘要:** Fine-grained robot manipulation, such as lifting and rotating a bottle to display the label on the cap, requires robust reasoning about object parts and their relationships with intended tasks. Despite recent advances in training general-purpose robot manipulation policies guided by language instructions, there is a notable lack of large-scale datasets for fine-grained manipulation tasks with part-level instructions and diverse 3D object instances annotated with part-level labels. In this work, we introduce PartInstruct, the first large-scale benchmark for training and evaluating fine-grained robot manipulation models using part-level instructions. PartInstruct comprises 513 object instances across 14 categories, each annotated with part-level information, and 1302 fine-grained manipulation tasks organized into 16 task classes. Our training set consists of over 10,000 expert demonstrations synthesized in a 3D simulator, where each demonstration is paired with a high-level task instruction, a chain of base part-based skill instructions, and ground-truth 3D information about the object and its parts. Additionally, we designed a comprehensive test suite to evaluate the generalizability of learned policies across new states, objects, and tasks. We evaluated several state-of-the-art robot manipulation approaches, including end-to-end vision-language policy learning and bi-level planning models for robot manipulation on our benchmark. The experimental results reveal that current models struggle to robustly ground part concepts and predict actions in 3D space, and face challenges when manipulating object parts in long-horizon tasks.
>
---
#### [replaced 011] Opt2Skill: Imitating Dynamically-feasible Whole-Body Trajectories for Versatile Humanoid Loco-Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.20514v5](http://arxiv.org/pdf/2409.20514v5)**

> **作者:** Fukang Liu; Zhaoyuan Gu; Yilin Cai; Ziyi Zhou; Hyunyoung Jung; Jaehwi Jang; Shijie Zhao; Sehoon Ha; Yue Chen; Danfei Xu; Ye Zhao
>
> **摘要:** Humanoid robots are designed to perform diverse loco-manipulation tasks. However, they face challenges due to their high-dimensional and unstable dynamics, as well as the complex contact-rich nature of the tasks. Model-based optimal control methods offer flexibility to define precise motion but are limited by high computational complexity and accurate contact sensing. On the other hand, reinforcement learning (RL) handles high-dimensional spaces with strong robustness but suffers from inefficient learning, unnatural motion, and sim-to-real gaps. To address these challenges, we introduce Opt2Skill, an end-to-end pipeline that combines model-based trajectory optimization with RL to achieve robust whole-body loco-manipulation. Opt2Skill generates dynamic feasible and contact-consistent reference motions for the Digit humanoid robot using differential dynamic programming (DDP) and trains RL policies to track these optimal trajectories. Our results demonstrate that Opt2Skill outperforms baselines that rely on human demonstrations and inverse kinematics-based references, both in motion tracking and task success rates. Furthermore, we show that incorporating trajectories with torque information improves contact force tracking in contact-involved tasks, such as wiping a table. We have successfully transferred our approach to real-world applications.
>
---
#### [replaced 012] Language and Planning in Robotic Navigation: A Multilingual Evaluation of State-of-the-Art Models
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.05478v2](http://arxiv.org/pdf/2501.05478v2)**

> **作者:** Malak Mansour; Ahmed Aly; Bahey Tharwat; Sarim Hashmi; Dong An; Ian Reid
>
> **备注:** This work has been accepted for presentation at LM4Plan@AAAI'25. For more details, please check: https://llmforplanning.github.io/
>
> **摘要:** Large Language Models (LLMs) such as GPT-4, trained on huge amount of datasets spanning multiple domains, exhibit significant reasoning, understanding, and planning capabilities across various tasks. This study presents the first-ever work in Arabic language integration within the Vision-and-Language Navigation (VLN) domain in robotics, an area that has been notably underexplored in existing research. We perform a comprehensive evaluation of state-of-the-art multi-lingual Small Language Models (SLMs), including GPT-4o mini, Llama 3 8B, and Phi-3 medium 14B, alongside the Arabic-centric LLM, Jais. Our approach utilizes the NavGPT framework, a pure LLM-based instruction-following navigation agent, to assess the impact of language on navigation reasoning through zero-shot sequential action prediction using the R2R dataset. Through comprehensive experiments, we demonstrate that our framework is capable of high-level planning for navigation tasks when provided with instructions in both English and Arabic. However, certain models struggled with reasoning and planning in the Arabic language due to inherent limitations in their capabilities, sub-optimal performance, and parsing issues. These findings highlight the importance of enhancing planning and reasoning capabilities in language models for effective navigation, emphasizing this as a key area for further development while also unlocking the potential of Arabic-language models for impactful real-world applications.
>
---
#### [replaced 013] LBAP: Improved Uncertainty Alignment of LLM Planners using Bayesian Inference
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.13198v3](http://arxiv.org/pdf/2403.13198v3)**

> **作者:** James F. Mullen Jr.; Dinesh Manocha
>
> **摘要:** Large language models (LLMs) showcase many desirable traits for intelligent and helpful robots. However, they are also known to hallucinate predictions. This issue is exacerbated in robotics where LLM hallucinations may result in robots confidently executing plans that are contrary to user goals or relying more frequently on human assistance. In this work, we present LBAP, a novel approach for utilizing off-the-shelf LLMs, alongside Bayesian inference for uncertainty Alignment in robotic Planners that minimizes hallucinations and human intervention. Our key finding is that we can use Bayesian inference to more accurately calibrate a robots confidence measure through accounting for both scene grounding and world knowledge. This process allows us to mitigate hallucinations and better align the LLM's confidence measure with the probability of success. Through experiments in both simulation and the real world on tasks with a variety of ambiguities, we show that LBAP significantly increases success rate and decreases the amount of human intervention required relative to prior art. For example, in our real-world testing paradigm, LBAP decreases the human help rate of previous methods by over 33% at a success rate of 70%.
>
---
#### [replaced 014] Hierarchical Intention Tracking with Switching Trees for Real-Time Adaptation to Dynamic Human Intentions during Collaboration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.07004v2](http://arxiv.org/pdf/2506.07004v2)**

> **作者:** Zhe Huang; Ye-Ji Mun; Fatemeh Cheraghi Pouria; Katherine Driggs-Campbell
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** During collaborative tasks, human behavior is guided by multiple levels of intentions that evolve over time, such as task sequence preferences and interaction strategies. To adapt to these changing preferences and promptly correct any inaccurate estimations, collaborative robots must accurately track these dynamic human intentions in real time. We propose a Hierarchical Intention Tracking (HIT) algorithm for collaborative robots to track dynamic and hierarchical human intentions effectively in real time. HIT represents human intentions as intention trees with arbitrary depth, and probabilistically tracks human intentions by Bayesian filtering, upward measurement propagation, and downward posterior propagation across all levels. We develop a HIT-based robotic system that dynamically switches between Interaction-Task and Verification-Task trees for a collaborative assembly task, allowing the robot to effectively coordinate human intentions at three levels: task-level (subtask goal locations), interaction-level (mode of engagement with the robot), and verification-level (confirming or correcting intention recognition). Our user study shows that our HIT-based collaborative robot system surpasses existing collaborative robot solutions by achieving a balance between efficiency, physical workload, and user comfort while ensuring safety and task completion. Post-experiment surveys further reveal that the HIT-based system enhances the user trust and minimizes interruptions to user's task flow through its effective understanding of human intentions across multiple levels.
>
---
#### [replaced 015] AssistantX: An LLM-Powered Proactive Assistant in Collaborative Human-Populated Environment
- **分类: cs.RO; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2409.17655v2](http://arxiv.org/pdf/2409.17655v2)**

> **作者:** Nan Sun; Bo Mao; Yongchang Li; Di Guo; Huaping Liu
>
> **备注:** 8 pages, 10 figures, 6 tables
>
> **摘要:** Current service robots suffer from limited natural language communication abilities, heavy reliance on predefined commands, ongoing human intervention, and, most notably, a lack of proactive collaboration awareness in human-populated environments. This results in narrow applicability and low utility. In this paper, we introduce AssistantX, an LLM-powered proactive assistant designed for autonomous operation in realworld scenarios with high accuracy. AssistantX employs a multi-agent framework consisting of 4 specialized LLM agents, each dedicated to perception, planning, decision-making, and reflective review, facilitating advanced inference capabilities and comprehensive collaboration awareness, much like a human assistant by your side. We built a dataset of 210 real-world tasks to validate AssistantX, which includes instruction content and status information on whether relevant personnel are available. Extensive experiments were conducted in both text-based simulations and a real office environment over the course of a month and a half. Our experiments demonstrate the effectiveness of the proposed framework, showing that AssistantX can reactively respond to user instructions, actively adjust strategies to adapt to contingencies, and proactively seek assistance from humans to ensure successful task completion. More details and videos can be found at https://assistantx-agent. github.io/AssistantX/.
>
---
#### [replaced 016] Semantic Enhancement for Object SLAM with Heterogeneous Multimodal Large Language Model Agents
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.06752v2](http://arxiv.org/pdf/2411.06752v2)**

> **作者:** Jungseok Hong; Ran Choi; John J. Leonard
>
> **摘要:** Object Simultaneous Localization and Mapping (SLAM) systems struggle to correctly associate semantically similar objects in close proximity, especially in cluttered indoor environments and when scenes change. We present Semantic Enhancement for Object SLAM (SEO-SLAM), a novel framework that enhances semantic mapping by integrating heterogeneous multimodal large language model (MLLM) agents. Our method enables scene adaptation while maintaining a semantically rich map. To improve computational efficiency, we propose an asynchronous processing scheme that significantly reduces the agents' inference time without compromising semantic accuracy or SLAM performance. Additionally, we introduce a multi-data association strategy using a cost matrix that combines semantic and Mahalanobis distances, formulating the problem as a Linear Assignment Problem (LAP) to alleviate perceptual aliasing. Experimental results demonstrate that SEO-SLAM consistently achieves higher semantic accuracy and reduces false positives compared to baselines, while our asynchronous MLLM agents significantly improve processing efficiency over synchronous setups. We also demonstrate that SEO-SLAM has the potential to improve downstream tasks such as robotic assistance. Our dataset is publicly available at: jungseokhong.com/SEO-SLAM.
>
---
#### [replaced 017] Sketch-Plan-Generalize: Learning and Planning with Neuro-Symbolic Programmatic Representations for Inductive Spatial Concepts
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2404.07774v3](http://arxiv.org/pdf/2404.07774v3)**

> **作者:** Namasivayam Kalithasan; Sachit Sachdeva; Himanshu Gaurav Singh; Vishal Bindal; Arnav Tuli; Gurarmaan Singh Panjeta; Harsh Himanshu Vora; Divyanshu Aggarwal; Rohan Paul; Parag Singla
>
> **备注:** Programmatic Representations for Agent Learning Worskop, ICML 2025
>
> **摘要:** Effective human-robot collaboration requires the ability to learn personalized concepts from a limited number of demonstrations, while exhibiting inductive generalization, hierarchical composition, and adaptability to novel constraints. Existing approaches that use code generation capabilities of pre-trained large (vision) language models as well as purely neural models show poor generalization to \emph{a-priori} unseen complex concepts. Neuro-symbolic methods (Grand et al., 2023) offer a promising alternative by searching in program space, but face challenges in large program spaces due to the inability to effectively guide the search using demonstrations. Our key insight is to factor inductive concept learning as: (i) {\it Sketch:} detecting and inferring a coarse signature of a new concept (ii) {\it Plan:} performing an MCTS search over grounded action sequences guided by human demonstrations (iii) {\it Generalize:} abstracting out grounded plans as inductive programs. Our pipeline facilitates generalization and modular re-use, enabling continual concept learning. Our approach combines the benefits of code generation ability of large language models (LLMs) along with grounded neural representations, resulting in neuro-symbolic programs that show stronger inductive generalization on the task of constructing complex structures vis-\'a-vis LLM-only and purely neural approaches. Further, we demonstrate reasoning and planning capabilities with learned concepts for embodied instruction following.
>
---
