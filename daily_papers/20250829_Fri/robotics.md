# 机器人 cs.RO

- **最新发布 30 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] Rapid Mismatch Estimation via Neural Network Informed Variational Inference
- **分类: cs.RO**

- **简介: 论文研究机器人动态环境中的动力学模型不匹配问题，提出RME框架，结合神经网络与变分推理在线估计末端执行器参数，无需外部传感器，在400ms内实现快速适应。**

- **链接: [http://arxiv.org/pdf/2508.21007v1](http://arxiv.org/pdf/2508.21007v1)**

> **作者:** Mateusz Jaszczuk; Nadia Figueroa
>
> **备注:** Accepted at 9th Annual Conference on Robot Learning. Project Website - https://mateusz-jaszczuk.github.io/rme/
>
> **摘要:** With robots increasingly operating in human-centric environments, ensuring soft and safe physical interactions, whether with humans, surroundings, or other machines, is essential. While compliant hardware can facilitate such interactions, this work focuses on impedance controllers that allow torque-controlled robots to safely and passively respond to contact while accurately executing tasks. From inverse dynamics to quadratic programming-based controllers, the effectiveness of these methods relies on accurate dynamics models of the robot and the object it manipulates. Any model mismatch results in task failures and unsafe behaviors. Thus, we introduce Rapid Mismatch Estimation (RME), an adaptive, controller-agnostic, probabilistic framework that estimates end-effector dynamics mismatches online, without relying on external force-torque sensors. From the robot's proprioceptive feedback, a Neural Network Model Mismatch Estimator generates a prior for a Variational Inference solver, which rapidly converges to the unknown parameters while quantifying uncertainty. With a real 7-DoF manipulator driven by a state-of-the-art passive impedance controller, RME adapts to sudden changes in mass and center of mass at the end-effector in $\sim400$ ms, in static and dynamic settings. We demonstrate RME in a collaborative scenario where a human attaches an unknown basket to the robot's end-effector and dynamically adds/removes heavy items, showcasing fast and safe adaptation to changing dynamics during physical interaction without any external sensory system.
>
---
#### [new 002] Task Allocation for Autonomous Machines using Computational Intelligence and Deep Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文综述了计算智能与深度强化学习在自主机器任务分配中的应用，分析其优缺点并提出未来研究方向，旨在提升自主机器在复杂环境中的协作效率与性能。**

- **链接: [http://arxiv.org/pdf/2508.20688v1](http://arxiv.org/pdf/2508.20688v1)**

> **作者:** Thanh Thi Nguyen; Quoc Viet Hung Nguyen; Jonathan Kua; Imran Razzak; Dung Nguyen; Saeid Nahavandi
>
> **备注:** Accepted for publication in the Proceedings of the 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC)
>
> **摘要:** Enabling multiple autonomous machines to perform reliably requires the development of efficient cooperative control algorithms. This paper presents a survey of algorithms that have been developed for controlling and coordinating autonomous machines in complex environments. We especially focus on task allocation methods using computational intelligence (CI) and deep reinforcement learning (RL). The advantages and disadvantages of the surveyed methods are analysed thoroughly. We also propose and discuss in detail various future research directions that shed light on how to improve existing algorithms or create new methods to enhance the employability and performance of autonomous machines in real-world applications. The findings indicate that CI and deep RL methods provide viable approaches to addressing complex task allocation problems in dynamic and uncertain environments. The recent development of deep RL has greatly contributed to the literature on controlling and coordinating autonomous machines, and it has become a growing trend in this area. It is envisaged that this paper will provide researchers and engineers with a comprehensive overview of progress in machine learning research related to autonomous machines. It also highlights underexplored areas, identifies emerging methodologies, and suggests new avenues for exploration in future research within this domain.
>
---
#### [new 003] Traversing the Narrow Path: A Two-Stage Reinforcement Learning Framework for Humanoid Beam Walking
- **分类: cs.RO**

- **简介: 论文提出两阶段强化学习框架，用于人形机器人在狭窄横梁上行走，解决稀疏接触与策略脆弱性问题，结合物理模板与残差优化，提升安全性和成功率。**

- **链接: [http://arxiv.org/pdf/2508.20661v1](http://arxiv.org/pdf/2508.20661v1)**

> **作者:** TianChen Huang; Wei Gao; Runchen Xu; Shiwu Zhang
>
> **摘要:** Traversing narrow beams is challenging for humanoids due to sparse, safety-critical contacts and the fragility of purely learned policies. We propose a physically grounded, two-stage framework that couples an XCoM/LIPM footstep template with a lightweight residual planner and a simple low-level tracker. Stage-1 is trained on flat ground: the tracker learns to robustly follow footstep targets by adding small random perturbations to heuristic footsteps, without any hand-crafted centerline locking, so it acquires stable contact scheduling and strong target-tracking robustness. Stage-2 is trained in simulation on a beam: a high-level planner predicts a body-frame residual (Delta x, Delta y, Delta psi) for the swing foot only, refining the template step to prioritize safe, precise placement under narrow support while preserving interpretability. To ease deployment, sensing is kept minimal and consistent between simulation and hardware: the planner consumes compact, forward-facing elevation cues together with onboard IMU and joint signals. On a Unitree G1, our system reliably traverses a 0.2 m-wide, 3 m-long beam. Across simulation and real-world studies, residual refinement consistently outperforms template-only and monolithic baselines in success rate, centerline adherence, and safety margins, while the structured footstep interface enables transparent analysis and low-friction sim-to-real transfer.
>
---
#### [new 004] Task-Oriented Edge-Assisted Cross-System Design for Real-Time Human-Robot Interaction in Industrial Metaverse
- **分类: cs.RO; cs.AI; cs.GR**

- **简介: 论文针对工业元宇宙中实时人机交互的高计算、带宽与延迟问题，提出边缘辅助的跨系统框架，通过数字孪生分离视觉与控制模块，并结合HITL-MAML算法优化预测，实现轨迹绘制与核废料处理任务中的高精度交互，提升空间精度与视觉质量。**

- **链接: [http://arxiv.org/pdf/2508.20664v1](http://arxiv.org/pdf/2508.20664v1)**

> **作者:** Kan Chen; Zhen Meng; Xiangmin Xu; Jiaming Yang; Emma Li; Philip G. Zhao
>
> **备注:** This paper has submitted to IEEE Transactions on Mobile Computing
>
> **摘要:** Real-time human-device interaction in industrial Metaverse faces challenges such as high computational load, limited bandwidth, and strict latency. This paper proposes a task-oriented edge-assisted cross-system framework using digital twins (DTs) to enable responsive interactions. By predicting operator motions, the system supports: 1) proactive Metaverse rendering for visual feedback, and 2) preemptive control of remote devices. The DTs are decoupled into two virtual functions-visual display and robotic control-optimizing both performance and adaptability. To enhance generalizability, we introduce the Human-In-The-Loop Model-Agnostic Meta-Learning (HITL-MAML) algorithm, which dynamically adjusts prediction horizons. Evaluation on two tasks demonstrates the framework's effectiveness: in a Trajectory-Based Drawing Control task, it reduces weighted RMSE from 0.0712 m to 0.0101 m; in a real-time 3D scene representation task for nuclear decommissioning, it achieves a PSNR of 22.11, SSIM of 0.8729, and LPIPS of 0.1298. These results show the framework's capability to ensure spatial precision and visual fidelity in real-time, high-risk industrial environments.
>
---
#### [new 005] Non-expert to Expert Motion Translation Using Generative Adversarial Networks
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文通过GAN实现非专家到专家的运动翻译，解决机器人技能迁移中任务适应性差的问题，提出灵活的运动翻译方法，使用户可通过输入数据教机器人执行任务。**

- **链接: [http://arxiv.org/pdf/2508.20740v1](http://arxiv.org/pdf/2508.20740v1)**

> **作者:** Yuki Tanaka; Seiichiro Katsura
>
> **摘要:** Decreasing skilled workers is a very serious problem in the world. To deal with this problem, the skill transfer from experts to robots has been researched. These methods which teach robots by human motion are called imitation learning. Experts' skills generally appear in not only position data, but also force data. Thus, position and force data need to be saved and reproduced. To realize this, a lot of research has been conducted in the framework of a motion-copying system. Recent research uses machine learning methods to generate motion commands. However, most of them could not change tasks by following human intention. Some of them can change tasks by conditional training, but the labels are limited. Thus, we propose the flexible motion translation method by using Generative Adversarial Networks. The proposed method enables users to teach robots tasks by inputting data, and skills by a trained model. We evaluated the proposed system with a 3-DOF calligraphy robot.
>
---
#### [new 006] Model-Free Hovering and Source Seeking via Extremum Seeking Control: Experimental Demonstration
- **分类: cs.RO; math.OC**

- **简介: 该论文提出利用极值搜索控制（ESC）实现扑翼机器人无模型、实时悬停与寻源控制，通过实验验证ESC作为生物仿生控制方法的可行性。**

- **链接: [http://arxiv.org/pdf/2508.20836v1](http://arxiv.org/pdf/2508.20836v1)**

> **作者:** Ahmed A. Elgohary; Rohan Palanikumar; Sameh A. Eisa
>
> **摘要:** In a recent effort, we successfully proposed a categorically novel approach to mimic the phenomenoa of hovering and source seeking by flapping insects and hummingbirds using a new extremum seeking control (ESC) approach. Said ESC approach was shown capable of characterizing the physics of hovering and source seeking by flapping systems, providing at the same time uniquely novel opportunity for a model-free, real-time biomimicry control design. In this paper, we experimentally test and verify, for the first time in the literature, the potential of ESC in flapping robots to achieve model-free, real-time controlled hovering and source seeking. The results of this paper, while being restricted to 1D, confirm the premise of introducing ESC as a natural control method and biomimicry mechanism to the field of flapping flight and robotics.
>
---
#### [new 007] HITTER: A HumanoId Table TEnnis Robot via Hierarchical Planning and Learning
- **分类: cs.RO**

- **简介: 论文研究人形机器人乒乓球任务，解决高速动态环境下的快速反应与精准控制问题，提出分层框架融合模型预测与强化学习，整合规划与控制，引入人类动作参考，实现亚秒级连续击球。**

- **链接: [http://arxiv.org/pdf/2508.21043v1](http://arxiv.org/pdf/2508.21043v1)**

> **作者:** Zhi Su; Bike Zhang; Nima Rahmanian; Yuman Gao; Qiayuan Liao; Caitlin Regan; Koushil Sreenath; S. Shankar Sastry
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Humanoid robots have recently achieved impressive progress in locomotion and whole-body control, yet they remain constrained in tasks that demand rapid interaction with dynamic environments through manipulation. Table tennis exemplifies such a challenge: with ball speeds exceeding 5 m/s, players must perceive, predict, and act within sub-second reaction times, requiring both agility and precision. To address this, we present a hierarchical framework for humanoid table tennis that integrates a model-based planner for ball trajectory prediction and racket target planning with a reinforcement learning-based whole-body controller. The planner determines striking position, velocity and timing, while the controller generates coordinated arm and leg motions that mimic human strikes and maintain stability and agility across consecutive rallies. Moreover, to encourage natural movements, human motion references are incorporated during training. We validate our system on a general-purpose humanoid robot, achieving up to 106 consecutive shots with a human opponent and sustained exchanges against another humanoid. These results demonstrate real-world humanoid table tennis with sub-second reactive control, marking a step toward agile and interactive humanoid behaviors.
>
---
#### [new 008] UltraTac: Integrated Ultrasound-Augmented Visuotactile Sensor for Enhanced Robotic Perception
- **分类: cs.RO**

- **简介: 论文设计UltraTac，整合超声波与视觉触觉传感器，解决传统传感器无法感知材料特性的难题，通过结构共享和动态调整实现多模态感知，验证其在材料分类和表面检测中的高精度。**

- **链接: [http://arxiv.org/pdf/2508.20982v1](http://arxiv.org/pdf/2508.20982v1)**

> **作者:** Junhao Gong; Kit-Wa Sou; Shoujie Li; Changqing Guo; Yan Huang; Chuqiao Lyu; Ziwu Song; Wenbo Ding
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Visuotactile sensors provide high-resolution tactile information but are incapable of perceiving the material features of objects. We present UltraTac, an integrated sensor that combines visuotactile imaging with ultrasound sensing through a coaxial optoacoustic architecture. The design shares structural components and achieves consistent sensing regions for both modalities. Additionally, we incorporate acoustic matching into the traditional visuotactile sensor structure, enabling integration of the ultrasound sensing modality without compromising visuotactile performance. Through tactile feedback, we dynamically adjust the operating state of the ultrasound module to achieve flexible functional coordination. Systematic experiments demonstrate three key capabilities: proximity sensing in the 3-8 cm range ($R^2=0.90$), material classification (average accuracy: 99.20%), and texture-material dual-mode object recognition achieving 92.11% accuracy on a 15-class task. Finally, we integrate the sensor into a robotic manipulation system to concurrently detect container surface patterns and internal content, which verifies its potential for advanced human-machine interaction and precise robotic manipulation.
>
---
#### [new 009] SimShear: Sim-to-Real Shear-based Tactile Servoing
- **分类: cs.RO**

- **简介: 论文提出SimShear，用于触觉伺服控制，解决仿真中剪切模拟难题，通过shPix2pix生成器将无剪切仿真图像转换为真实效果，并验证其在两个任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.20561v1](http://arxiv.org/pdf/2508.20561v1)**

> **作者:** Kipp McAdam Freud; Yijiong Lin; Nathan F. Lepora
>
> **备注:** 2025 Conference on Robot Learning (CoRL)
>
> **摘要:** We present SimShear, a sim-to-real pipeline for tactile control that enables the use of shear information without explicitly modeling shear dynamics in simulation. Shear, arising from lateral movements across contact surfaces, is critical for tasks involving dynamic object interactions but remains challenging to simulate. To address this, we introduce shPix2pix, a shear-conditioned U-Net GAN that transforms simulated tactile images absent of shear, together with a vector encoding shear information, into realistic equivalents with shear deformations. This method outperforms baseline pix2pix approaches in simulating tactile images and in pose/shear prediction. We apply SimShear to two control tasks using a pair of low-cost desktop robotic arms equipped with a vision-based tactile sensor: (i) a tactile tracking task, where a follower arm tracks a surface moved by a leader arm, and (ii) a collaborative co-lifting task, where both arms jointly hold an object while the leader follows a prescribed trajectory. Our method maintains contact errors within 1 to 2 mm across varied trajectories where shear sensing is essential, validating the feasibility of sim-to-real shear modeling with rigid-body simulators and opening new directions for simulation in tactile robotics.
>
---
#### [new 010] Uncertainty Aware-Predictive Control Barrier Functions: Safer Human Robot Interaction through Probabilistic Motion Forecasting
- **分类: cs.RO; cs.AI**

- **简介: 论文针对人机协作中的安全与灵活性矛盾，提出结合概率运动预测与控制屏障函数的UA-PCBF框架，动态调整安全边距，提升交互安全性与流畅性。**

- **链接: [http://arxiv.org/pdf/2508.20812v1](http://arxiv.org/pdf/2508.20812v1)**

> **作者:** Lorenzo Busellato; Federico Cunico; Diego Dall'Alba; Marco Emporio; Andrea Giachetti; Riccardo Muradore; Marco Cristani
>
> **摘要:** To enable flexible, high-throughput automation in settings where people and robots share workspaces, collaborative robotic cells must reconcile stringent safety guarantees with the need for responsive and effective behavior. A dynamic obstacle is the stochastic, task-dependent variability of human motion: when robots fall back on purely reactive or worst-case envelopes, they brake unnecessarily, stall task progress, and tamper with the fluidity that true Human-Robot Interaction demands. In recent years, learning-based human-motion prediction has rapidly advanced, although most approaches produce worst-case scenario forecasts that often do not treat prediction uncertainty in a well-structured way, resulting in over-conservative planning algorithms, limiting their flexibility. We introduce Uncertainty-Aware Predictive Control Barrier Functions (UA-PCBFs), a unified framework that fuses probabilistic human hand motion forecasting with the formal safety guarantees of Control Barrier Functions. In contrast to other variants, our framework allows for dynamic adjustment of the safety margin thanks to the human motion uncertainty estimation provided by a forecasting module. Thanks to uncertainty estimation, UA-PCBFs empower collaborative robots with a deeper understanding of future human states, facilitating more fluid and intelligent interactions through informed motion planning. We validate UA-PCBFs through comprehensive real-world experiments with an increasing level of realism, including automated setups (to perform exactly repeatable motions) with a robotic hand and direct human-robot interactions (to validate promptness, usability, and human confidence). Relative to state-of-the-art HRI architectures, UA-PCBFs show better performance in task-critical metrics, significantly reducing the number of violations of the robot's safe space during interaction with respect to the state-of-the-art.
>
---
#### [new 011] PLUME: Procedural Layer Underground Modeling Engine
- **分类: cs.RO**

- **简介: 论文提出PLUME，一种程序化生成地下环境模型的框架，解决太阳系地下环境多样性不足的问题，通过灵活结构生成3D环境用于AI训练、机器人测试等。**

- **链接: [http://arxiv.org/pdf/2508.20926v1](http://arxiv.org/pdf/2508.20926v1)**

> **作者:** Gabriel Manuel Garcia; Antoine Richard; Miguel Olivares-Mendez
>
> **摘要:** As space exploration advances, underground environments are becoming increasingly attractive due to their potential to provide shelter, easier access to resources, and enhanced scientific opportunities. Although such environments exist on Earth, they are often not easily accessible and do not accurately represent the diversity of underground environments found throughout the solar system. This paper presents PLUME, a procedural generation framework aimed at easily creating 3D underground environments. Its flexible structure allows for the continuous enhancement of various underground features, aligning with our expanding understanding of the solar system. The environments generated using PLUME can be used for AI training, evaluating robotics algorithms, 3D rendering, and facilitating rapid iteration on developed exploration algorithms. In this paper, it is demonstrated that PLUME has been used along with a robotic simulator. PLUME is open source and has been released on Github. https://github.com/Gabryss/P.L.U.M.E
>
---
#### [new 012] Deep Fuzzy Optimization for Batch-Size and Nearest Neighbors in Optimal Robot Motion Planning
- **分类: cs.RO**

- **简介: 论文针对机器人运动规划中的参数优化问题，提出LIT*算法，结合深度模糊学习动态调整批量大小和最近邻参数，提升高维空间下的规划效率与路径质量。**

- **链接: [http://arxiv.org/pdf/2508.20884v1](http://arxiv.org/pdf/2508.20884v1)**

> **作者:** Liding Zhang; Qiyang Zong; Yu Zhang; Zhenshan Bing; Alois Knoll
>
> **摘要:** Efficient motion planning algorithms are essential in robotics. Optimizing essential parameters, such as batch size and nearest neighbor selection in sampling-based methods, can enhance performance in the planning process. However, existing approaches often lack environmental adaptability. Inspired by the method of the deep fuzzy neural networks, this work introduces Learning-based Informed Trees (LIT*), a sampling-based deep fuzzy learning-based planner that dynamically adjusts batch size and nearest neighbor parameters to obstacle distributions in the configuration spaces. By encoding both global and local ratios via valid and invalid states, LIT* differentiates between obstacle-sparse and obstacle-dense regions, leading to lower-cost paths and reduced computation time. Experimental results in high-dimensional spaces demonstrate that LIT* achieves faster convergence and improved solution quality. It outperforms state-of-the-art single-query, sampling-based planners in environments ranging from R^8 to R^14 and is successfully validated on a dual-arm robot manipulation task. A video showcasing our experimental results is available at: https://youtu.be/NrNs9zebWWk
>
---
#### [new 013] Language-Enhanced Mobile Manipulation for Efficient Object Search in Indoor Environments
- **分类: cs.RO**

- **简介: 该论文针对室内环境中高效物体搜索任务，解决传统方法在复杂场景中语义理解与路径规划能力不足的问题，提出融合大语言模型与启发式规划的分层导航框架GODHS，提升搜索效率。**

- **链接: [http://arxiv.org/pdf/2508.20899v1](http://arxiv.org/pdf/2508.20899v1)**

> **作者:** Liding Zhang; Zeqi Li; Kuanqi Cai; Qian Huang; Zhenshan Bing; Alois Knoll
>
> **摘要:** Enabling robots to efficiently search for and identify objects in complex, unstructured environments is critical for diverse applications ranging from household assistance to industrial automation. However, traditional scene representations typically capture only static semantics and lack interpretable contextual reasoning, limiting their ability to guide object search in completely unfamiliar settings. To address this challenge, we propose a language-enhanced hierarchical navigation framework that tightly integrates semantic perception and spatial reasoning. Our method, Goal-Oriented Dynamically Heuristic-Guided Hierarchical Search (GODHS), leverages large language models (LLMs) to infer scene semantics and guide the search process through a multi-level decision hierarchy. Reliability in reasoning is achieved through the use of structured prompts and logical constraints applied at each stage of the hierarchy. For the specific challenges of mobile manipulation, we introduce a heuristic-based motion planner that combines polar angle sorting with distance prioritization to efficiently generate exploration paths. Comprehensive evaluations in Isaac Sim demonstrate the feasibility of our framework, showing that GODHS can locate target objects with higher search efficiency compared to conventional, non-semantic search strategies. Website and Video are available at: https://drapandiger.github.io/GODHS
>
---
#### [new 014] SPGrasp: Spatiotemporal Prompt-driven Grasp Synthesis in Dynamic Scenes
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出SPGrasp，解决动态物体实时抓取的低延迟与提示性矛盾，通过时空上下文融合实现59ms低延迟、90.6%准确率。**

- **链接: [http://arxiv.org/pdf/2508.20547v1](http://arxiv.org/pdf/2508.20547v1)**

> **作者:** Yunpeng Mei; Hongjie Cao; Yinqiu Xia; Wei Xiao; Zhaohan Feng; Gang Wang; Jie Chen
>
> **摘要:** Real-time interactive grasp synthesis for dynamic objects remains challenging as existing methods fail to achieve low-latency inference while maintaining promptability. To bridge this gap, we propose SPGrasp (spatiotemporal prompt-driven dynamic grasp synthesis), a novel framework extending segment anything model v2 (SAMv2) for video stream grasp estimation. Our core innovation integrates user prompts with spatiotemporal context, enabling real-time interaction with end-to-end latency as low as 59 ms while ensuring temporal consistency for dynamic objects. In benchmark evaluations, SPGrasp achieves instance-level grasp accuracies of 90.6% on OCID and 93.8% on Jacquard. On the challenging GraspNet-1Billion dataset under continuous tracking, SPGrasp achieves 92.0% accuracy with 73.1 ms per-frame latency, representing a 58.5% reduction compared to the prior state-of-the-art promptable method RoG-SAM while maintaining competitive accuracy. Real-world experiments involving 13 moving objects demonstrate a 94.8% success rate in interactive grasping scenarios. These results confirm SPGrasp effectively resolves the latency-interactivity trade-off in dynamic grasp synthesis. Code is available at https://github.com/sejmoonwei/SPGrasp.
>
---
#### [new 015] Genetic Informed Trees (GIT*): Path Planning via Reinforced Genetic Programming Heuristics
- **分类: cs.RO**

- **简介: 论文研究路径规划任务，解决传统方法忽略环境数据的问题，提出GIT*结合强化遗传编程优化启发式函数，提升搜索效率与解质量。**

- **链接: [http://arxiv.org/pdf/2508.20871v1](http://arxiv.org/pdf/2508.20871v1)**

> **作者:** Liding Zhang; Kuanqi Cai; Zhenshan Bing; Chaoqun Wang; Alois Knoll
>
> **摘要:** Optimal path planning involves finding a feasible state sequence between a start and a goal that optimizes an objective. This process relies on heuristic functions to guide the search direction. While a robust function can improve search efficiency and solution quality, current methods often overlook available environmental data and simplify the function structure due to the complexity of information relationships. This study introduces Genetic Informed Trees (GIT*), which improves upon Effort Informed Trees (EIT*) by integrating a wider array of environmental data, such as repulsive forces from obstacles and the dynamic importance of vertices, to refine heuristic functions for better guidance. Furthermore, we integrated reinforced genetic programming (RGP), which combines genetic programming with reward system feedback to mutate genotype-generative heuristic functions for GIT*. RGP leverages a multitude of data types, thereby improving computational efficiency and solution quality within a set timeframe. Comparative analyses demonstrate that GIT* surpasses existing single-query, sampling-based planners in problems ranging from R^4 to R^16 and was tested on a real-world mobile manipulation task. A video showcasing our experimental results is available at https://youtu.be/URjXbc_BiYg
>
---
#### [new 016] A Soft Fabric-Based Thermal Haptic Device for VR and Teleoperation
- **分类: cs.RO**

- **简介: 论文提出一种基于柔软织物的热触觉装置，用于VR和远程操作，整合气动与导电材料，实现轻量化热力反馈，实验验证其高效温度识别和操作改进。**

- **链接: [http://arxiv.org/pdf/2508.20831v1](http://arxiv.org/pdf/2508.20831v1)**

> **作者:** Rui Chen; Domenico Chiaradia; Antonio Frisoli; Daniele Leonardis
>
> **摘要:** This paper presents a novel fabric-based thermal-haptic interface for virtual reality and teleoperation. It integrates pneumatic actuation and conductive fabric with an innovative ultra-lightweight design, achieving only 2~g for each finger unit. By embedding heating elements within textile pneumatic chambers, the system delivers modulated pressure and thermal stimuli to fingerpads through a fully soft, wearable interface. Comprehensive characterization demonstrates rapid thermal modulation with heating rates up to 3$^{\circ}$C/s, enabling dynamic thermal feedback for virtual or teleoperation interactions. The pneumatic subsystem generates forces up to 8.93~N at 50~kPa, while optimization of fingerpad-actuator clearance enhances cooling efficiency with minimal force reduction. Experimental validation conducted with two different user studies shows high temperature identification accuracy (0.98 overall) across three thermal levels, and significant manipulation improvements in a virtual pick-and-place tasks. Results show enhanced success rates (88.5\% to 96.4\%, p = 0.029) and improved force control precision (p = 0.013) when haptic feedback is enabled, validating the effectiveness of the integrated thermal-haptic approach for advanced human-machine interaction applications.
>
---
#### [new 017] Scaling Fabric-Based Piezoresistive Sensor Arrays for Whole-Body Tactile Sensing
- **分类: cs.RO; eess.SP**

- **简介: 论文提出一种全身体触觉传感系统，解决布线复杂、数据通量和可靠性问题，通过织物传感器、定制电子和新型SPI拓扑实现高精度实时反馈，成功应用于机器人抓取任务。**

- **链接: [http://arxiv.org/pdf/2508.20959v1](http://arxiv.org/pdf/2508.20959v1)**

> **作者:** Curtis C. Johnson; Daniel Webb; David Hill; Marc D. Killpack
>
> **备注:** In submission to IEEE Sensors
>
> **摘要:** Scaling tactile sensing for robust whole-body manipulation is a significant challenge, often limited by wiring complexity, data throughput, and system reliability. This paper presents a complete architecture designed to overcome these barriers. Our approach pairs open-source, fabric-based sensors with custom readout electronics that reduce signal crosstalk to less than 3.3% through hardware-based mitigation. Critically, we introduce a novel, daisy-chained SPI bus topology that avoids the practical limitations of common wireless protocols and the prohibitive wiring complexity of USB hub-based systems. This architecture streams synchronized data from over 8,000 taxels across 1 square meter of sensing area at update rates exceeding 50 FPS, confirming its suitability for real-time control. We validate the system's efficacy in a whole-body grasping task where, without feedback, the robot's open-loop trajectory results in an uncontrolled application of force that slowly crushes a deformable cardboard box. With real-time tactile feedback, the robot transforms this motion into a gentle, stable grasp, successfully manipulating the object without causing structural damage. This work provides a robust and well-characterized platform to enable future research in advanced whole-body control and physical human-robot interaction.
>
---
#### [new 018] Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation
- **分类: cs.RO**

- **简介: 该论文针对机器人策略在仿真与现实间的迁移问题，提出基于可微分模拟的在线自适应框架，通过实时动态建模与策略更新，在5秒内适应环境扰动，提升四旋翼控制精度。**

- **链接: [http://arxiv.org/pdf/2508.21065v1](http://arxiv.org/pdf/2508.21065v1)**

> **作者:** Jiahe Pan; Jiaxu Xing; Rudolf Reiter; Yifan Zhai; Elie Aljalbout; Davide Scaramuzza
>
> **摘要:** Learning control policies in simulation enables rapid, safe, and cost-effective development of advanced robotic capabilities. However, transferring these policies to the real world remains difficult due to the sim-to-real gap, where unmodeled dynamics and environmental disturbances can degrade policy performance. Existing approaches, such as domain randomization and Real2Sim2Real pipelines, can improve policy robustness, but either struggle under out-of-distribution conditions or require costly offline retraining. In this work, we approach these problems from a different perspective. Instead of relying on diverse training conditions before deployment, we focus on rapidly adapting the learned policy in the real world in an online fashion. To achieve this, we propose a novel online adaptive learning framework that unifies residual dynamics learning with real-time policy adaptation inside a differentiable simulation. Starting from a simple dynamics model, our framework refines the model continuously with real-world data to capture unmodeled effects and disturbances such as payload changes and wind. The refined dynamics model is embedded in a differentiable simulation framework, enabling gradient backpropagation through the dynamics and thus rapid, sample-efficient policy updates beyond the reach of classical RL methods like PPO. All components of our system are designed for rapid adaptation, enabling the policy to adjust to unseen disturbances within 5 seconds of training. We validate the approach on agile quadrotor control under various disturbances in both simulation and the real world. Our framework reduces hovering error by up to 81% compared to L1-MPC and 55% compared to DATT, while also demonstrating robustness in vision-based control without explicit state estimation.
>
---
#### [new 019] Learning Fast, Tool aware Collision Avoidance for Collaborative Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文针对协作机器人在动态环境中的工具变化和障碍物运动导致的碰撞问题，提出工具感知的实时避障系统，结合学习感知模型与约束强化学习控制策略，实现实时高效避障，提升动态环境中的安全性与效率。**

- **链接: [http://arxiv.org/pdf/2508.20457v1](http://arxiv.org/pdf/2508.20457v1)**

> **作者:** Joonho Lee; Yunho Kim; Seokjoon Kim; Quan Nguyen; Youngjin Heo
>
> **摘要:** Ensuring safe and efficient operation of collaborative robots in human environments is challenging, especially in dynamic settings where both obstacle motion and tasks change over time. Current robot controllers typically assume full visibility and fixed tools, which can lead to collisions or overly conservative behavior. In our work, we introduce a tool-aware collision avoidance system that adjusts in real time to different tool sizes and modes of tool-environment interaction. Using a learned perception model, our system filters out robot and tool components from the point cloud, reasons about occluded area, and predicts collision under partial observability. We then use a control policy trained via constrained reinforcement learning to produce smooth avoidance maneuvers in under 10 milliseconds. In simulated and real-world tests, our approach outperforms traditional approaches (APF, MPPI) in dynamic environments, while maintaining sub-millimeter accuracy. Moreover, our system operates with approximately 60% lower computational cost compared to a state-of-the-art GPU-based planner. Our approach provides modular, efficient, and effective collision avoidance for robots operating in dynamic environments. We integrate our method into a collaborative robot application and demonstrate its practical use for safe and responsive operation.
>
---
#### [new 020] Learning Primitive Embodied World Models: Towards Scalable Robotic Learning
- **分类: cs.RO; cs.AI; cs.MM**

- **简介: 该论文针对具身世界建模中的数据瓶颈与对齐难题，提出Primitive Embodied World Models（PEWM），通过限制生成时长、结合VLM与SGG机制，实现细粒度语言-动作对齐，提升数据效率与泛化能力，推动可扩展的机器人学习。**

- **链接: [http://arxiv.org/pdf/2508.20840v1](http://arxiv.org/pdf/2508.20840v1)**

> **作者:** Qiao Sun; Liujia Yang; Wei Tang; Wei Huang; Kaixin Xu; Yongchao Chen; Mingyu Liu; Jiange Yang; Haoyi Zhu; Yating Wang; Tong He; Yilun Chen; Xili Dai; Nanyang Ye; Qinying Gu
>
> **摘要:** While video-generation-based embodied world models have gained increasing attention, their reliance on large-scale embodied interaction data remains a key bottleneck. The scarcity, difficulty of collection, and high dimensionality of embodied data fundamentally limit the alignment granularity between language and actions and exacerbate the challenge of long-horizon video generation--hindering generative models from achieving a "GPT moment" in the embodied domain. There is a naive observation: the diversity of embodied data far exceeds the relatively small space of possible primitive motions. Based on this insight, we propose a novel paradigm for world modeling--Primitive Embodied World Models (PEWM). By restricting video generation to fixed short horizons, our approach 1) enables fine-grained alignment between linguistic concepts and visual representations of robotic actions, 2) reduces learning complexity, 3) improves data efficiency in embodied data collection, and 4) decreases inference latency. By equipping with a modular Vision-Language Model (VLM) planner and a Start-Goal heatmap Guidance mechanism (SGG), PEWM further enables flexible closed-loop control and supports compositional generalization of primitive-level policies over extended, complex tasks. Our framework leverages the spatiotemporal vision priors in video models and the semantic awareness of VLMs to bridge the gap between fine-grained physical interaction and high-level reasoning, paving the way toward scalable, interpretable, and general-purpose embodied intelligence.
>
---
#### [new 021] ActLoc: Learning to Localize on the Move via Active Viewpoint Selection
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 论文针对机器人移动定位中视角信息不均问题，提出ActLoc框架，通过注意力模型主动选择最优视角，结合地图与姿态预测定位准确性，提升路径规划的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.20981v1](http://arxiv.org/pdf/2508.20981v1)**

> **作者:** Jiajie Li; Boyang Sun; Luca Di Giammarino; Hermann Blum; Marc Pollefeys
>
> **摘要:** Reliable localization is critical for robot navigation, yet most existing systems implicitly assume that all viewing directions at a location are equally informative. In practice, localization becomes unreliable when the robot observes unmapped, ambiguous, or uninformative regions. To address this, we present ActLoc, an active viewpoint-aware planning framework for enhancing localization accuracy for general robot navigation tasks. At its core, ActLoc employs a largescale trained attention-based model for viewpoint selection. The model encodes a metric map and the camera poses used during map construction, and predicts localization accuracy across yaw and pitch directions at arbitrary 3D locations. These per-point accuracy distributions are incorporated into a path planner, enabling the robot to actively select camera orientations that maximize localization robustness while respecting task and motion constraints. ActLoc achieves stateof-the-art results on single-viewpoint selection and generalizes effectively to fulltrajectory planning. Its modular design makes it readily applicable to diverse robot navigation and inspection tasks.
>
---
#### [new 022] CoCoL: A Communication Efficient Decentralized Collaborative Method for Multi-Robot Systems
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 论文针对多机器人协同学习中的通信开销和数据异质性问题，提出CoCoL方法，采用镜像下降框架和梯度追踪，实现高效通信与低计算成本，实验验证其在减少通信轮次和带宽消耗方面优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.20898v1](http://arxiv.org/pdf/2508.20898v1)**

> **作者:** Jiaxi Huang; Yan Huang; Yixian Zhao; Wenchao Meng; Jinming Xu
>
> **备注:** Accepted by IROS2025
>
> **摘要:** Collaborative learning enhances the performance and adaptability of multi-robot systems in complex tasks but faces significant challenges due to high communication overhead and data heterogeneity inherent in multi-robot tasks. To this end, we propose CoCoL, a Communication efficient decentralized Collaborative Learning method tailored for multi-robot systems with heterogeneous local datasets. Leveraging a mirror descent framework, CoCoL achieves remarkable communication efficiency with approximate Newton-type updates by capturing the similarity between objective functions of robots, and reduces computational costs through inexact sub-problem solutions. Furthermore, the integration of a gradient tracking scheme ensures its robustness against data heterogeneity. Experimental results on three representative multi robot collaborative learning tasks show the superiority of the proposed CoCoL in significantly reducing both the number of communication rounds and total bandwidth consumption while maintaining state-of-the-art accuracy. These benefits are particularly evident in challenging scenarios involving non-IID (non-independent and identically distributed) data distribution, streaming data, and time-varying network topologies.
>
---
#### [new 023] Prompt-to-Product: Generative Assembly via Bimanual Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Prompt-to-Product，通过双臂机器人将自然语言提示转化为乐高装配产品，解决从创意到实物的自动化生成问题，降低手动努力。**

- **链接: [http://arxiv.org/pdf/2508.21063v1](http://arxiv.org/pdf/2508.21063v1)**

> **作者:** Ruixuan Liu; Philip Huang; Ava Pun; Kangle Deng; Shobhit Aggarwal; Kevin Tang; Michelle Liu; Deva Ramanan; Jun-Yan Zhu; Jiaoyang Li; Changliu Liu
>
> **备注:** 12 pages, 10 figures, 2 tables
>
> **摘要:** Creating assembly products demands significant manual effort and expert knowledge in 1) designing the assembly and 2) constructing the product. This paper introduces Prompt-to-Product, an automated pipeline that generates real-world assembly products from natural language prompts. Specifically, we leverage LEGO bricks as the assembly platform and automate the process of creating brick assembly structures. Given the user design requirements, Prompt-to-Product generates physically buildable brick designs, and then leverages a bimanual robotic system to construct the real assembly products, bringing user imaginations into the real world. We conduct a comprehensive user study, and the results demonstrate that Prompt-to-Product significantly lowers the barrier and reduces manual effort in creating assembly products from imaginative ideas.
>
---
#### [new 024] COMETH: Convex Optimization for Multiview Estimation and Tracking of Humans
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对多视角人体姿态估计与跟踪的实时性与精度问题，提出COMETH算法，结合凸优化和状态观测器，提升时空一致性与准确性，适用于工业场景。**

- **链接: [http://arxiv.org/pdf/2508.20920v1](http://arxiv.org/pdf/2508.20920v1)**

> **作者:** Enrico Martini; Ho Jin Choi; Nadia Figueroa; Nicola Bombieri
>
> **备注:** Submitted to Information Fusion
>
> **摘要:** In the era of Industry 5.0, monitoring human activity is essential for ensuring both ergonomic safety and overall well-being. While multi-camera centralized setups improve pose estimation accuracy, they often suffer from high computational costs and bandwidth requirements, limiting scalability and real-time applicability. Distributing processing across edge devices can reduce network bandwidth and computational load. On the other hand, the constrained resources of edge devices lead to accuracy degradation, and the distribution of computation leads to temporal and spatial inconsistencies. We address this challenge by proposing COMETH (Convex Optimization for Multiview Estimation and Tracking of Humans), a lightweight algorithm for real-time multi-view human pose fusion that relies on three concepts: it integrates kinematic and biomechanical constraints to increase the joint positioning accuracy; it employs convex optimization-based inverse kinematics for spatial fusion; and it implements a state observer to improve temporal consistency. We evaluate COMETH on both public and industrial datasets, where it outperforms state-of-the-art methods in localization, detection, and tracking accuracy. The proposed fusion pipeline enables accurate and scalable human motion tracking, making it well-suited for industrial and safety-critical applications. The code is publicly available at https://github.com/PARCO-LAB/COMETH.
>
---
#### [new 025] Encoding Tactile Stimuli for Organoid Intelligence in Braille Recognition
- **分类: cs.NE; cs.ET; cs.RO**

- **简介: 该论文通过编码触觉刺激，使神经类器官执行盲文识别任务，解决类器官处理触觉信息的挑战。通过优化电刺激参数并结合多类器官，实现高准确率和抗噪能力，展示其作为生物混合计算元件的潜力。**

- **链接: [http://arxiv.org/pdf/2508.20850v1](http://arxiv.org/pdf/2508.20850v1)**

> **作者:** Tianyi Liu; Hemma Philamore; Benjamin Ward-Cherrier
>
> **摘要:** This study proposes a generalizable encoding strategy that maps tactile sensor data to electrical stimulation patterns, enabling neural organoids to perform an open-loop artificial tactile Braille classification task. Human forebrain organoids cultured on a low-density microelectrode array (MEA) are systematically stimulated to characterize the relationship between electrical stimulation parameters (number of pulse, phase amplitude, phase duration, and trigger delay) and organoid responses, measured as spike activity and spatial displacement of the center of activity. Implemented on event-based tactile inputs recorded from the Evetac sensor, our system achieved an average Braille letter classification accuracy of 61 percent with a single organoid, which increased significantly to 83 percent when responses from a three-organoid ensemble were combined. Additionally, the multi-organoid configuration demonstrated enhanced robustness against various types of artificially introduced noise. This research demonstrates the potential of organoids as low-power, adaptive bio-hybrid computational elements and provides a foundational encoding framework for future scalable bio-hybrid computing architectures.
>
---
#### [new 026] SKGE-SWIN: End-To-End Autonomous Vehicle Waypoint Prediction and Navigation Using Skip Stage Swin Transformer
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 论文提出SKGE-Swin架构，结合Swin Transformer与跳过阶段机制，用于自动驾驶车辆的端到端路径预测与导航，提升复杂环境下的像素级上下文理解能力，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.20762v1](http://arxiv.org/pdf/2508.20762v1)**

> **作者:** Fachri Najm Noer Kartiman; Rasim; Yaya Wihardi; Nurul Hasanah; Oskar Natan; Bambang Wahono; Taufik Ibnu Salim
>
> **备注:** keywords-multitask learning, autonomous driving, end-to-end learning, skip connections, swin transformer, self-attention mechanism. 12 pages
>
> **摘要:** Focusing on the development of an end-to-end autonomous vehicle model with pixel-to-pixel context awareness, this research proposes the SKGE-Swin architecture. This architecture utilizes the Swin Transformer with a skip-stage mechanism to broaden feature representation globally and at various network levels. This approach enables the model to extract information from distant pixels by leveraging the Swin Transformer's Shifted Window-based Multi-head Self-Attention (SW-MSA) mechanism and to retain critical information from the initial to the final stages of feature extraction, thereby enhancing its capability to comprehend complex patterns in the vehicle's surroundings. The model is evaluated on the CARLA platform using adversarial scenarios to simulate real-world conditions. Experimental results demonstrate that the SKGE-Swin architecture achieves a superior Driving Score compared to previous methods. Furthermore, an ablation study will be conducted to evaluate the contribution of each architectural component, including the influence of skip connections and the use of the Swin Transformer, in improving model performance.
>
---
#### [new 027] To New Beginnings: A Survey of Unified Perception in Autonomous Vehicle Software
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶感知中模块化方法的误差积累与任务协同不足问题，综述统一感知范式，提出系统分类框架，总结三种统一模式及现有方法，指导未来研究。**

- **链接: [http://arxiv.org/pdf/2508.20892v1](http://arxiv.org/pdf/2508.20892v1)**

> **作者:** Loïc Stratil; Felix Fent; Esteban Rivera; Markus Lienkamp
>
> **摘要:** Autonomous vehicle perception typically relies on modular pipelines that decompose the task into detection, tracking, and prediction. While interpretable, these pipelines suffer from error accumulation and limited inter-task synergy. Unified perception has emerged as a promising paradigm that integrates these sub-tasks within a shared architecture, potentially improving robustness, contextual reasoning, and efficiency while retaining interpretable outputs. In this survey, we provide a comprehensive overview of unified perception, introducing a holistic and systemic taxonomy that categorizes methods along task integration, tracking formulation, and representation flow. We define three paradigms -Early, Late, and Full Unified Perception- and systematically review existing methods, their architectures, training strategies, datasets used, and open-source availability, while highlighting future research directions. This work establishes the first comprehensive framework for understanding and advancing unified perception, consolidates fragmented efforts, and guides future research toward more robust, generalizable, and interpretable perception.
>
---
#### [new 028] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification
- **分类: cs.CV; cs.RO**

- **简介: 论文提出CogVLA，解决Vision-Language-Action（VLA）模型训练成本高、推理慢的问题，通过指令驱动的路由与稀疏化技术，提升效率与性能，在LIBERO基准和机器人任务中达到SOTA效果。**

- **链接: [http://arxiv.org/pdf/2508.21046v1](http://arxiv.org/pdf/2508.21046v1)**

> **作者:** Wei Li; Renshan Zhang; Rui Shao; Jie He; Liqiang Nie
>
> **备注:** 23 pages, 8 figures, Project Page: https://jiutian-vl.github.io/CogVLA-page
>
> **摘要:** Recent Vision-Language-Action (VLA) models built on pre-trained Vision-Language Models (VLMs) require extensive post-training, resulting in high computational overhead that limits scalability and deployment.We propose CogVLA, a Cognition-Aligned Vision-Language-Action framework that leverages instruction-driven routing and sparsification to improve both efficiency and performance. CogVLA draws inspiration from human multimodal coordination and introduces a 3-stage progressive architecture. 1) Encoder-FiLM based Aggregation Routing (EFA-Routing) injects instruction information into the vision encoder to selectively aggregate and compress dual-stream visual tokens, forming a instruction-aware latent representation. 2) Building upon this compact visual encoding, LLM-FiLM based Pruning Routing (LFP-Routing) introduces action intent into the language model by pruning instruction-irrelevant visually grounded tokens, thereby achieving token-level sparsity. 3) To ensure that compressed perception inputs can still support accurate and coherent action generation, we introduce V-L-A Coupled Attention (CAtten), which combines causal vision-language attention with bidirectional action parallel decoding. Extensive experiments on the LIBERO benchmark and real-world robotic tasks demonstrate that CogVLA achieves state-of-the-art performance with success rates of 97.4% and 70.0%, respectively, while reducing training costs by 2.5-fold and decreasing inference latency by 2.8-fold compared to OpenVLA. CogVLA is open-sourced and publicly available at https://github.com/JiuTian-VL/CogVLA.
>
---
#### [new 029] Regulation-Aware Game-Theoretic Motion Planning for Autonomous Racing
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文针对自动驾驶赛车中的法规合规运动规划任务，解决车辆交互中的安全与规则约束问题。通过将赛车规则编码为混合逻辑动态约束，构建广义纳什均衡模型，提出Regulation-Aware Game-Theoretic Planner，实现安全且非保守的超车策略。**

- **链接: [http://arxiv.org/pdf/2508.20203v1](http://arxiv.org/pdf/2508.20203v1)**

> **作者:** Francesco Prignoli; Francesco Borrelli; Paolo Falcone; Mark Pustilnik
>
> **备注:** Accepted for presentation at the IEEE International Conference on Intelligent Transportation Systems (ITSC 2025)
>
> **摘要:** This paper presents a regulation-aware motion planning framework for autonomous racing scenarios. Each agent solves a Regulation-Compliant Model Predictive Control problem, where racing rules - such as right-of-way and collision avoidance responsibilities - are encoded using Mixed Logical Dynamical constraints. We formalize the interaction between vehicles as a Generalized Nash Equilibrium Problem (GNEP) and approximate its solution using an Iterative Best Response scheme. Building on this, we introduce the Regulation-Aware Game-Theoretic Planner (RA-GTP), in which the attacker reasons over the defender's regulation-constrained behavior. This game-theoretic layer enables the generation of overtaking strategies that are both safe and non-conservative. Simulation results demonstrate that the RA-GTP outperforms baseline methods that assume non-interacting or rule-agnostic opponent models, leading to more effective maneuvers while consistently maintaining compliance with racing regulations.
>
---
#### [new 030] Train-Once Plan-Anywhere Kinodynamic Motion Planning via Diffusion Trees
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文针对动力学运动规划任务，解决高效安全轨迹生成问题，提出DiTree框架，结合扩散策略与采样规划，实现快速且泛化的解决方案，实验验证效果更好。**

- **链接: [http://arxiv.org/pdf/2508.21001v1](http://arxiv.org/pdf/2508.21001v1)**

> **作者:** Yaniv Hassidof; Tom Jurgenson; Kiril Solovey
>
> **备注:** Accepted to CoRL 2025. Project page: https://sites.google.com/view/ditree
>
> **摘要:** Kinodynamic motion planning is concerned with computing collision-free trajectories while abiding by the robot's dynamic constraints. This critical problem is often tackled using sampling-based planners (SBPs) that explore the robot's high-dimensional state space by constructing a search tree via action propagations. Although SBPs can offer global guarantees on completeness and solution quality, their performance is often hindered by slow exploration due to uninformed action sampling. Learning-based approaches can yield significantly faster runtimes, yet they fail to generalize to out-of-distribution (OOD) scenarios and lack critical guarantees, e.g., safety, thus limiting their deployment on physical robots. We present Diffusion Tree (DiTree): a \emph{provably-generalizable} framework leveraging diffusion policies (DPs) as informed samplers to efficiently guide state-space search within SBPs. DiTree combines DP's ability to model complex distributions of expert trajectories, conditioned on local observations, with the completeness of SBPs to yield \emph{provably-safe} solutions within a few action propagation iterations for complex dynamical systems. We demonstrate DiTree's power with an implementation combining the popular RRT planner with a DP action sampler trained on a \emph{single environment}. In comprehensive evaluations on OOD scenarios, % DiTree has comparable runtimes to a standalone DP (3x faster than classical SBPs), while improving the average success rate over DP and SBPs. DiTree is on average 3x faster than classical SBPs, and outperforms all other approaches by achieving roughly 30\% higher success rate. Project webpage: https://sites.google.com/view/ditree.
>
---
## 更新

#### [replaced 001] Uncertainty-Aware Trajectory Prediction via Rule-Regularized Heteroscedastic Deep Classification
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.13111v3](http://arxiv.org/pdf/2504.13111v3)**

> **作者:** Kumar Manas; Christian Schlauch; Adrian Paschke; Christian Wirth; Nadja Klein
>
> **备注:** 17 Pages, 9 figures. Accepted to Robotics: Science and Systems(RSS), 2025
>
> **摘要:** Deep learning-based trajectory prediction models have demonstrated promising capabilities in capturing complex interactions. However, their out-of-distribution generalization remains a significant challenge, particularly due to unbalanced data and a lack of enough data and diversity to ensure robustness and calibration. To address this, we propose SHIFT (Spectral Heteroscedastic Informed Forecasting for Trajectories), a novel framework that uniquely combines well-calibrated uncertainty modeling with informative priors derived through automated rule extraction. SHIFT reformulates trajectory prediction as a classification task and employs heteroscedastic spectral-normalized Gaussian processes to effectively disentangle epistemic and aleatoric uncertainties. We learn informative priors from training labels, which are automatically generated from natural language driving rules, such as stop rules and drivability constraints, using a retrieval-augmented generation framework powered by a large language model. Extensive evaluations over the nuScenes dataset, including challenging low-data and cross-location scenarios, demonstrate that SHIFT outperforms state-of-the-art methods, achieving substantial gains in uncertainty calibration and displacement metrics. In particular, our model excels in complex scenarios, such as intersections, where uncertainty is inherently higher. Project page: https://kumarmanas.github.io/SHIFT/.
>
---
#### [replaced 002] Enhanced Trust Region Sequential Convex Optimization for Multi-Drone Thermal Screening Trajectory Planning in Urban Environments
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **链接: [http://arxiv.org/pdf/2506.06012v3](http://arxiv.org/pdf/2506.06012v3)**

> **作者:** Kaiyuan Chen; Zhengjie Hu; Shaolin Zhang; Yuanqing Xia; Wannian Liang; Shuo Wang
>
> **摘要:** The rapid detection of abnormal body temperatures in urban populations is essential for managing public health risks, especially during outbreaks of infectious diseases. Multi-drone thermal screening systems offer promising solutions for fast, large-scale, and non-intrusive human temperature monitoring. However, trajectory planning for multiple drones in complex urban environments poses significant challenges, including collision avoidance, coverage efficiency, and constrained flight environments. In this study, we propose an enhanced trust region sequential convex optimization (TR-SCO) algorithm for optimal trajectory planning of multiple drones performing thermal screening tasks. Our improved algorithm integrates a refined convex optimization formulation within a trust region framework, effectively balancing trajectory smoothness, obstacle avoidance, altitude constraints, and maximum screening coverage. Simulation results demonstrate that our approach significantly improves trajectory optimality and computational efficiency compared to conventional convex optimization methods. This research provides critical insights and practical contributions toward deploying efficient multi-drone systems for real-time thermal screening in urban areas. For reader who are interested in our research, we release our source code at https://github.com/Cherry0302/Enhanced-TR-SCO.
>
---
#### [replaced 003] From Tabula Rasa to Emergent Abilities: Discovering Robot Skills via Real-World Unsupervised Quality-Diversity
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.19172v3](http://arxiv.org/pdf/2508.19172v3)**

> **作者:** Luca Grillotti; Lisa Coiffard; Oscar Pang; Maxence Faldor; Antoine Cully
>
> **备注:** Accepted at CoRL 2025
>
> **摘要:** Autonomous skill discovery aims to enable robots to acquire diverse behaviors without explicit supervision. Learning such behaviors directly on physical hardware remains challenging due to safety and data efficiency constraints. Existing methods, including Quality-Diversity Actor-Critic (QDAC), require manually defined skill spaces and carefully tuned heuristics, limiting real-world applicability. We propose Unsupervised Real-world Skill Acquisition (URSA), an extension of QDAC that enables robots to autonomously discover and master diverse, high-performing skills directly in the real world. We demonstrate that URSA successfully discovers diverse locomotion skills on a Unitree A1 quadruped in both simulation and the real world. Our approach supports both heuristic-driven skill discovery and fully unsupervised settings. We also show that the learned skill repertoire can be reused for downstream tasks such as real-world damage adaptation, where URSA outperforms all baselines in 5 out of 9 simulated and 3 out of 5 real-world damage scenarios. Our results establish a new framework for real-world robot learning that enables continuous skill discovery with limited human intervention, representing a significant step toward more autonomous and adaptable robotic systems. Demonstration videos are available at https://adaptive-intelligent-robotics.github.io/URSA.
>
---
#### [replaced 004] Pixel Motion as Universal Representation for Robot Control
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.07817v2](http://arxiv.org/pdf/2505.07817v2)**

> **作者:** Kanchana Ranasinghe; Xiang Li; E-Ro Nguyen; Cristina Mata; Jongwoo Park; Michael S Ryoo
>
> **摘要:** We present LangToMo, a vision-language-action framework structured as a dual-system architecture that uses pixel motion forecasts as intermediate representations. Our high-level System 2, an image diffusion model, generates text-conditioned pixel motion sequences from a single frame to guide robot control. Pixel motion-a universal, interpretable, and motion-centric representation-can be extracted from videos in a weakly-supervised manner, enabling diffusion model training on any video-caption data. Treating generated pixel motion as learned universal representations, our low level System 1 module translates these into robot actions via motion-to-action mapping functions, which can be either hand-crafted or learned with minimal supervision. System 2 operates as a high-level policy applied at sparse temporal intervals, while System 1 acts as a low-level policy at dense temporal intervals. This hierarchical decoupling enables flexible, scalable, and generalizable robot control under both unsupervised and supervised settings, bridging the gap between language, motion, and action. Checkout https://kahnchana.github.io/LangToMo
>
---
#### [replaced 005] On the complexity of constrained reconfiguration and motion planning
- **分类: cs.CC; cs.DM; cs.DS; cs.RO; math.CO**

- **链接: [http://arxiv.org/pdf/2508.13032v3](http://arxiv.org/pdf/2508.13032v3)**

> **作者:** Nicolas Bousquet; Remy El Sabeh; Amer E. Mouawad; Naomi Nishimura
>
> **摘要:** Coordinating the motion of multiple agents in constrained environments is a fundamental challenge in robotics, motion planning, and scheduling. A motivating example involves $n$ robotic arms, each represented as a line segment. The objective is to rotate each arm to its vertical orientation, one at a time (clockwise or counterclockwise), without collisions nor rotating any arm more than once. This scenario is an example of the more general $k$-Compatible Ordering problem, where $n$ agents, each capable of $k$ state-changing actions, must transition to specific target states under constraints encoded as a set $\mathcal{G}$ of $k$ pairs of directed graphs. We show that $k$-Compatible Ordering is $\mathsf{NP}$-complete, even when $\mathcal{G}$ is planar, degenerate, or acyclic. On the positive side, we provide polynomial-time algorithms for cases such as when $k = 1$ or $\mathcal{G}$ has bounded treewidth. We also introduce generalized variants supporting multiple state-changing actions per agent, broadening the applicability of our framework. These results extend to a wide range of scheduling, reconfiguration, and motion planning applications in constrained environments.
>
---
#### [replaced 006] UAV-UGV Cooperative Trajectory Optimization and Task Allocation for Medical Rescue Tasks in Post-Disaster Environments
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.06136v2](http://arxiv.org/pdf/2506.06136v2)**

> **作者:** Kaiyuan Chen; Wanpeng Zhao; Yongxi Liu; Yuanqing Xia; Wannian Liang; Shuo Wang
>
> **摘要:** In post-disaster scenarios, rapid and efficient delivery of medical resources is critical and challenging due to severe damage to infrastructure. To provide an optimized solution, we propose a cooperative trajectory optimization and task allocation framework leveraging unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs). This study integrates a Genetic Algorithm (GA) for efficient task allocation among multiple UAVs and UGVs, and employs an informed-RRT* (Rapidly-exploring Random Tree Star) algorithm for collision-free trajectory generation. Further optimization of task sequencing and path efficiency is conducted using Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Simulation experiments conducted in a realistic post-disaster environment demonstrate that our proposed approach significantly improves the overall efficiency of medical rescue operations compared to traditional strategies. Specifically, our method reduces the total mission completion time to 26.7 minutes for a 15-task scenario, outperforming K-Means clustering and random allocation by over 73%. Furthermore, the framework achieves a substantial 15.1% reduction in total traveled distance after CMA-ES optimization. The cooperative utilization of UAVs and UGVs effectively balances their complementary advantages, highlighting the system's scalability and practicality for real-world deployment.
>
---
#### [replaced 007] CogNav: Cognitive Process Modeling for Object Goal Navigation with LLMs
- **分类: cs.CV; cs.RO; I.2; I.4**

- **链接: [http://arxiv.org/pdf/2412.10439v3](http://arxiv.org/pdf/2412.10439v3)**

> **作者:** Yihan Cao; Jiazhao Zhang; Zhinan Yu; Shuzhen Liu; Zheng Qin; Qin Zou; Bo Du; Kai Xu
>
> **摘要:** Object goal navigation (ObjectNav) is a fundamental task in embodied AI, requiring an agent to locate a target object in previously unseen environments. This task is particularly challenging because it requires both perceptual and cognitive processes, including object recognition and decision-making. While substantial advancements in perception have been driven by the rapid development of visual foundation models, progress on the cognitive aspect remains constrained, primarily limited to either implicit learning through simulator rollouts or explicit reliance on predefined heuristic rules. Inspired by neuroscientific findings demonstrating that humans maintain and dynamically update fine-grained cognitive states during object search tasks in novel environments, we propose CogNav, a framework designed to mimic this cognitive process using large language models. Specifically, we model the cognitive process using a finite state machine comprising fine-grained cognitive states, ranging from exploration to identification. Transitions between states are determined by a large language model based on a dynamically constructed heterogeneous cognitive map, which contains spatial and semantic information about the scene being explored. Extensive evaluations on the HM3D, MP3D, and RoboTHOR benchmarks demonstrate that our cognitive process modeling significantly improves the success rate of ObjectNav at least by relative 14% over the state-of-the-arts.
>
---
#### [replaced 008] Safe and Efficient Social Navigation through Explainable Safety Regions Based on Topological Features
- **分类: cs.RO; cs.AI; math.GN**

- **链接: [http://arxiv.org/pdf/2503.16441v2](http://arxiv.org/pdf/2503.16441v2)**

> **作者:** Victor Toscano-Duran; Sara Narteni; Alberto Carlevaro; Jérôme Guzzi Rocio Gonzalez-Diaz; Maurizio Mongelli
>
> **摘要:** The recent adoption of artificial intelligence in robotics has driven the development of algorithms that enable autonomous systems to adapt to complex social environments. In particular, safe and efficient social navigation is a key challenge, requiring AI not only to avoid collisions and deadlocks but also to interact intuitively and predictably with its surroundings. Methods based on probabilistic models and the generation of conformal safety regions have shown promising results in defining safety regions with a controlled margin of error, primarily relying on classification approaches and explicit rules to describe collision-free navigation conditions. This work extends the existing perspective by investigating how topological features can contribute to the creation of explainable safety regions in social navigation scenarios, enabling the classification and characterization of different simulation behaviors. Rather than relying on behaviors parameters to generate safety regions, we leverage topological features through topological data analysis. We first utilize global rule-based classification to provide interpretable characterizations of different simulation behaviors, distinguishing between safe and unsafe scenarios based on topological properties. Next, we define safety regions, $S_\varepsilon$, representing zones in the topological feature space where collisions are avoided with a maximum classification error of $\varepsilon$. These regions are constructed using adjustable SVM classifiers and order statistics, ensuring a robust and scalable decision boundary. Our approach initially separates simulations with and without collisions, outperforming methods that not incorporate topological features. We further refine safety regions to ensure deadlock-free simulations and integrate both aspects to define a compliant simulation space that guarantees safe and efficient navigation.
>
---
#### [replaced 009] Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.19958v2](http://arxiv.org/pdf/2508.19958v2)**

> **作者:** Yiguo Fan; Pengxiang Ding; Shuanghao Bai; Xinyang Tong; Yuyang Zhu; Hongchao Lu; Fengqi Dai; Wei Zhao; Yang Liu; Siteng Huang; Zhaoxin Fan; Badong Chen; Donglin Wang
>
> **备注:** Accepted to CoRL 2025; Github Page: https://long-vla.github.io
>
> **摘要:** Vision-Language-Action (VLA) models have become a cornerstone in robotic policy learning, leveraging large-scale multimodal data for robust and scalable control. However, existing VLA frameworks primarily address short-horizon tasks, and their effectiveness on long-horizon, multi-step robotic manipulation remains limited due to challenges in skill chaining and subtask dependencies. In this work, we introduce Long-VLA, the first end-to-end VLA model specifically designed for long-horizon robotic tasks. Our approach features a novel phase-aware input masking strategy that adaptively segments each subtask into moving and interaction phases, enabling the model to focus on phase-relevant sensory cues and enhancing subtask compatibility. This unified strategy preserves the scalability and data efficiency of VLA training, and our architecture-agnostic module can be seamlessly integrated into existing VLA models. We further propose the L-CALVIN benchmark to systematically evaluate long-horizon manipulation. Extensive experiments on both simulated and real-world tasks demonstrate that Long-VLA significantly outperforms prior state-of-the-art methods, establishing a new baseline for long-horizon robotic control.
>
---
#### [replaced 010] HERMES: Human-to-Robot Embodied Learning from Multi-Source Motion Data for Mobile Dexterous Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.20085v2](http://arxiv.org/pdf/2508.20085v2)**

> **作者:** Zhecheng Yuan; Tianming Wei; Langzhe Gu; Pu Hua; Tianhai Liang; Yuanpei Chen; Huazhe Xu
>
> **摘要:** Leveraging human motion data to impart robots with versatile manipulation skills has emerged as a promising paradigm in robotic manipulation. Nevertheless, translating multi-source human hand motions into feasible robot behaviors remains challenging, particularly for robots equipped with multi-fingered dexterous hands characterized by complex, high-dimensional action spaces. Moreover, existing approaches often struggle to produce policies capable of adapting to diverse environmental conditions. In this paper, we introduce HERMES, a human-to-robot learning framework for mobile bimanual dexterous manipulation. First, HERMES formulates a unified reinforcement learning approach capable of seamlessly transforming heterogeneous human hand motions from multiple sources into physically plausible robotic behaviors. Subsequently, to mitigate the sim2real gap, we devise an end-to-end, depth image-based sim2real transfer method for improved generalization to real-world scenarios. Furthermore, to enable autonomous operation in varied and unstructured environments, we augment the navigation foundation model with a closed-loop Perspective-n-Point (PnP) localization mechanism, ensuring precise alignment of visual goals and effectively bridging autonomous navigation and dexterous manipulation. Extensive experimental results demonstrate that HERMES consistently exhibits generalizable behaviors across diverse, in-the-wild scenarios, successfully performing numerous complex mobile bimanual dexterous manipulation tasks. Project Page:https://gemcollector.github.io/HERMES/.
>
---
#### [replaced 011] Unscented Kalman Filter with a Nonlinear Propagation Model for Navigation Applications
- **分类: cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.10082v3](http://arxiv.org/pdf/2507.10082v3)**

> **作者:** Amit Levy; Itzik Klein
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** The unscented Kalman filter is a nonlinear estimation algorithm commonly used in navigation applications. The prediction of the mean and covariance matrix is crucial to the stable behavior of the filter. This prediction is done by propagating the sigma points according to the dynamic model at hand. In this paper, we introduce an innovative method to propagate the sigma points according to the nonlinear dynamic model of the navigation error state vector. This improves the filter accuracy and navigation performance. We demonstrate the benefits of our proposed approach using real sensor data recorded by an autonomous underwater vehicle during several scenarios.
>
---
#### [replaced 012] RSRNav: Reasoning Spatial Relationship for Image-Goal Navigation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.17991v2](http://arxiv.org/pdf/2504.17991v2)**

> **作者:** Zheng Qin; Le Wang; Yabing Wang; Sanping Zhou; Gang Hua; Wei Tang
>
> **摘要:** Recent image-goal navigation (ImageNav) methods learn a perception-action policy by separately capturing semantic features of the goal and egocentric images, then passing them to a policy network. However, challenges remain: (1) Semantic features often fail to provide accurate directional information, leading to superfluous actions, and (2) performance drops significantly when viewpoint inconsistencies arise between training and application. To address these challenges, we propose RSRNav, a simple yet effective method that reasons spatial relationships between the goal and current observations as navigation guidance. Specifically, we model the spatial relationship by constructing correlations between the goal and current observations, which are then passed to the policy network for action prediction. These correlations are progressively refined using fine-grained cross-correlation and direction-aware correlation for more precise navigation. Extensive evaluation of RSRNav on three benchmark datasets demonstrates superior navigation performance, particularly in the "user-matched goal" setting, highlighting its potential for real-world applications.
>
---
#### [replaced 013] Learning Complex Motion Plans using Neural ODEs with Safety and Stability Guarantees
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2308.00186v4](http://arxiv.org/pdf/2308.00186v4)**

> **作者:** Farhad Nawaz; Tianyu Li; Nikolai Matni; Nadia Figueroa
>
> **备注:** accepted to ICRA 2024
>
> **摘要:** We propose a Dynamical System (DS) approach to learn complex, possibly periodic motion plans from kinesthetic demonstrations using Neural Ordinary Differential Equations (NODE). To ensure reactivity and robustness to disturbances, we propose a novel approach that selects a target point at each time step for the robot to follow, by combining tools from control theory and the target trajectory generated by the learned NODE. A correction term to the NODE model is computed online by solving a quadratic program that guarantees stability and safety using control Lyapunov functions and control barrier functions, respectively. Our approach outperforms baseline DS learning techniques on the LASA handwriting dataset and complex periodic trajectories. It is also validated on the Franka Emika robot arm to produce stable motions for wiping and stirring tasks that do not have a single attractor, while being robust to perturbations and safe around humans and obstacles.
>
---
#### [replaced 014] TacCompress: A Benchmark for Multi-Point Tactile Data Compression in Dexterous Hand
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.16289v2](http://arxiv.org/pdf/2505.16289v2)**

> **作者:** Yan Zhao; Yang Li; Zhengxue Cheng; Hengdi Zhang; Li Song
>
> **备注:** 9 pages, 10 figures, 2 tables
>
> **摘要:** Though robotic dexterous manipulation has progressed substantially recently, challenges like in-hand occlusion still necessitate fine-grained tactile perception, leading to the integration of more tactile sensors into robotic hands. Consequently, the increased data volume imposes substantial bandwidth pressure on signal transmission from the hand's controller. However, the acquisition and compression of multi-point tactile signals based on the dexterous hands' physical structures have not been thoroughly explored. In this paper, our contributions are twofold. First, we introduce a Multi-Point Tactile Dataset for Dexterous Hand Grasping (Dex-MPTD). This dataset captures tactile signals from multiple contact sensors across various objects and grasping poses, offering a comprehensive benchmark for advancing dexterous robotic manipulation research. Second, we investigate both lossless and lossy compression on Dex-MPTD by converting tactile data into images and applying six lossless and five lossy image codecs for efficient compression. Experimental results demonstrate that tactile data can be losslessly compressed to as low as 0.0364 bits per sub-sample (bpss), achieving approximately 200$\times$ compression ratio compared to the raw tactile data. Efficient lossy compressors like HM and VTM can achieve about 1000$\times$ data reductions while preserving acceptable data fidelity. The exploration of lossy compression also reveals that screen-content-targeted coding tools outperform general-purpose codecs in compressing tactile data.
>
---
#### [replaced 015] Staircase Recognition and Location Based on Polarization Vision
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19026v3](http://arxiv.org/pdf/2505.19026v3)**

> **作者:** Weifeng Kong; Zhiying Tan
>
> **摘要:** Staircase is one of the most common structures in artificial scenes. However, it is difficult for humanoid robots and people with lower limb disabilities or visual impairment to cross the scene without the help of sensors and intelligent algorithms. Staircase scene perception technology is a prerequisite for recognition and localization. This technology is of great significance for the mode switching of the robot and the calculation of the footprint position to adapt to the discontinuous terrain. However, there are still many problems that constrain the application of this technology, such as low recognition accuracy, high initial noise from sensors, unstable output signals and high computational requirements. In terms of scene reconstruction, the binocular and time of flight (TOF) reconstruction of the scene can be easily affected by environmental light and the surface material of the target object. In contrast, due to the special structure of the polarizer, the polarization can selectively transmit polarized light in a specific direction and this reconstruction method relies on the polarization information of the object surface. So the advantages of polarization reconstruction are reflected, which are less affected by environmental light and not dependent on the texture information of the object surface. In this paper, in order to achieve the detection of staircase, this paper proposes a contrast enhancement algorithm that integrates polarization and light intensity information, and integrates point cloud segmentation based on YOLOv11. To realize the high-quality reconstruction, we proposed a method of fusing polarized binocular and TOF depth information to realize the three-dimensional (3D) reconstruction of the staircase. Besides, it also proposes a joint calibration algorithm of monocular camera and TOF camera based on ICP registration and improved gray wolf optimization algorithm.
>
---
#### [replaced 016] Omni-Perception: Omnidirectional Collision Avoidance for Legged Locomotion in Dynamic Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19214v2](http://arxiv.org/pdf/2505.19214v2)**

> **作者:** Zifan Wang; Teli Ma; Yufei Jia; Xun Yang; Jiaming Zhou; Wenlong Ouyang; Qiang Zhang; Junwei Liang
>
> **摘要:** Agile locomotion in complex 3D environments requires robust spatial awareness to safely avoid diverse obstacles such as aerial clutter, uneven terrain, and dynamic agents. Depth-based perception approaches often struggle with sensor noise, lighting variability, computational overhead from intermediate representations (e.g., elevation maps), and difficulties with non-planar obstacles, limiting performance in unstructured environments. In contrast, direct integration of LiDAR sensing into end-to-end learning for legged locomotion remains underexplored. We propose Omni-Perception, an end-to-end locomotion policy that achieves 3D spatial awareness and omnidirectional collision avoidance by directly processing raw LiDAR point clouds. At its core is PD-RiskNet (Proximal-Distal Risk-Aware Hierarchical Network), a novel perception module that interprets spatio-temporal LiDAR data for environmental risk assessment. To facilitate efficient policy learning, we develop a high-fidelity LiDAR simulation toolkit with realistic noise modeling and fast raycasting, compatible with platforms such as Isaac Gym, Genesis, and MuJoCo, enabling scalable training and effective sim-to-real transfer. Learning reactive control policies directly from raw LiDAR data enables the robot to navigate complex environments with static and dynamic obstacles more robustly than approaches relying on intermediate maps or limited sensing. We validate Omni-Perception through real-world experiments and extensive simulation, demonstrating strong omnidirectional avoidance capabilities and superior locomotion performance in highly dynamic environments.
>
---
#### [replaced 017] A Simple Approach to Constraint-Aware Imitation Learning with Application to Autonomous Racing
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.07737v2](http://arxiv.org/pdf/2503.07737v2)**

> **作者:** Shengfan Cao; Eunhyek Joa; Francesco Borrelli
>
> **备注:** Accepted for publication at IROS 2025
>
> **摘要:** Guaranteeing constraint satisfaction is challenging in imitation learning (IL), particularly in tasks that require operating near a system's handling limits. Traditional IL methods, such as Behavior Cloning (BC), often struggle to enforce constraints, leading to suboptimal performance in high-precision tasks. In this paper, we present a simple approach to incorporating safety into the IL objective. Through simulations, we empirically validate our approach on an autonomous racing task with both full-state and image feedback, demonstrating improved constraint satisfaction and greater consistency in task performance compared to BC.
>
---
#### [replaced 018] Dimension-Decomposed Learning for Quadrotor Geometric Attitude Control with Almost Global Exponential Convergence on SO(3)
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **链接: [http://arxiv.org/pdf/2508.14422v2](http://arxiv.org/pdf/2508.14422v2)**

> **作者:** Tianhua Gao; Masashi Izumita; Kohji Tomita; Akiya Kamimura
>
> **备注:** v2: Corrected methodology naming typo; provided TeX source files
>
> **摘要:** This paper introduces a lightweight and interpretable online learning approach called Dimension-Decomposed Learning (DiD-L) for disturbance identification in quadrotor geometric attitude control. As a module instance of DiD-L, we propose the Sliced Adaptive-Neuro Mapping (SANM). Specifically, to address underlying underfitting problems, the high-dimensional mapping for online identification is axially ``sliced" into multiple low-dimensional submappings (slices). In this way, the complex high-dimensional problem is decomposed into a set of simple low-dimensional subtasks addressed by shallow neural networks and adaptive laws. These neural networks and adaptive laws are updated online via Lyapunov-based adaptation without the persistent excitation (PE) condition. To enhance the interpretability of the proposed approach, we prove that the state solution of the rotational error dynamics exponentially converges into an arbitrarily small ball within an almost global attraction domain, despite time-varying disturbances and inertia uncertainties. This result is novel as it demonstrates exponential convergence without requiring pre-training for unseen disturbances and specific knowledge of the model. To our knowledge in the quadrotor control field, DiD-L is the first online learning approach that is lightweight enough to run in real-time at 400 Hz on microcontroller units (MCUs) such as STM32, and has been validated through real-world experiments.
>
---
#### [replaced 019] Learning to Drive Ethically: Embedding Moral Reasoning into Autonomous Driving
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.14926v2](http://arxiv.org/pdf/2508.14926v2)**

> **作者:** Dianzhao Li; Ostap Okhrin
>
> **摘要:** Autonomous vehicles hold great promise for reducing traffic fatalities and improving transportation efficiency, yet their widespread adoption hinges on embedding robust ethical reasoning into routine and emergency maneuvers, particularly to protect vulnerable road users (VRUs) such as pedestrians and cyclists. Here, we present a hierarchical Safe Reinforcement Learning (Safe RL) framework that explicitly integrates moral considerations with standard driving objectives. At the decision level, a Safe RL agent is trained using a composite ethical risk cost, combining collision probability and harm severity, to generate high-level motion targets. A dynamic Prioritized Experience Replay mechanism amplifies learning from rare but critical, high-risk events. At the execution level, polynomial path planning coupled with Proportional-Integral-Derivative (PID) and Stanley controllers translates these targets into smooth, feasible trajectories, ensuring both accuracy and comfort. We train and validate our approach on rich, real-world traffic datasets encompassing diverse vehicles, cyclists, and pedestrians, and demonstrate that it outperforms baseline methods in reducing ethical risk and maintaining driving performance. To our knowledge, this is the first study of ethical decision-making for autonomous vehicles via Safe RL evaluated on real-world, human-mixed traffic scenarios. Our results highlight the potential of combining formal control theory and data-driven learning to advance ethically accountable autonomy that explicitly protects those most at risk in urban traffic environments.
>
---
#### [replaced 020] Residual Neural Terminal Constraint for MPC-based Collision Avoidance in Dynamic Environments
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2508.03428v2](http://arxiv.org/pdf/2508.03428v2)**

> **作者:** Bojan Derajić; Mohamed-Khalil Bouzidi; Sebastian Bernhard; Wolfgang Hönig
>
> **摘要:** In this paper, we propose a hybrid MPC local planner that uses a learning-based approximation of a time-varying safe set, derived from local observations and applied as the MPC terminal constraint. This set can be represented as a zero-superlevel set of the value function computed via Hamilton-Jacobi (HJ) reachability analysis, which is infeasible in real-time. We exploit the property that the HJ value function can be expressed as a difference of the corresponding signed distance function (SDF) and a non-negative residual function. The residual component is modeled as a neural network with non-negative output and subtracted from the computed SDF, resulting in a real-time value function estimate that is at least as safe as the SDF by design. Additionally, we parametrize the neural residual by a hypernetwork to improve real-time performance and generalization properties. The proposed method is compared with three state-of-the-art methods in simulations and hardware experiments, achieving up to 30\% higher success rates compared to the best baseline while requiring a similar computational effort and producing high-quality (low travel-time) solutions.
>
---
#### [replaced 021] FFHFlow: Diverse and Uncertainty-Aware Dexterous Grasp Generation via Flow Variational Inference
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.15161v3](http://arxiv.org/pdf/2407.15161v3)**

> **作者:** Qian Feng; Jianxiang Feng; Zhaopeng Chen; Rudolph Triebel; Alois Knoll
>
> **备注:** First two authors contributed equally, whose ordering decided via coin-tossing. Accepted for CoRL 2025
>
> **摘要:** Synthesizing diverse, uncertainty-aware grasps for multi-fingered hands from partial observations remains a critical challenge in robot learning. Prior generative methods struggle to model the intricate grasp distribution of dexterous hands and often fail to reason about shape uncertainty inherent in partial point clouds, leading to unreliable or overly conservative grasps. We propose FFHFlow, a flow-based variational framework that generates diverse, robust multi-finger grasps while explicitly quantifying perceptual uncertainty in the partial point clouds. Our approach leverages a normalizing flow-based deep latent variable model to learn a hierarchical grasp manifold, overcoming the mode collapse and rigid prior limitations of conditional Variational Autoencoders (cVAEs). By exploiting the invertibility and exact likelihoods of flows, FFHFlow introspects shape uncertainty in partial observations and identifies novel object structures, enabling risk-aware grasp synthesis. To further enhance reliability, we integrate a discriminative grasp evaluator with the flow likelihoods, formulating an uncertainty-aware ranking strategy that prioritizes grasps robust to shape ambiguity. Extensive experiments in simulation and real-world setups demonstrate that FFHFlow outperforms state-of-the-art baselines (including diffusion models) in grasp diversity and success rate, while achieving run-time efficient sampling. We also showcase its practical value in cluttered and confined environments, where diversity-driven sampling excels by mitigating collisions (Project Page: https://sites.google.com/view/ffhflow/home/).
>
---
