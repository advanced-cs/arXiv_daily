# 机器人 cs.RO

- **最新发布 45 篇**

- **更新 25 篇**

## 最新发布

#### [new 001] From Theory to Practice: Advancing Multi-Robot Path Planning Algorithms and Applications
- **分类: cs.RO**

- **简介: 该论文属于多机器人路径规划任务，解决高效避障路径规划问题。提出新方法提升算法效率与实用性，应用于仓储、停车等场景。**

- **链接: [http://arxiv.org/pdf/2506.09914v1](http://arxiv.org/pdf/2506.09914v1)**

> **作者:** Teng Guo
>
> **备注:** Ph.D. thesis
>
> **摘要:** The labeled MRPP (Multi-Robot Path Planning) problem involves routing robots from start to goal configurations efficiently while avoiding collisions. Despite progress in solution quality and runtime, its complexity and industrial relevance continue to drive research. This dissertation introduces scalable MRPP methods with provable guarantees and practical heuristics. First, we study dense MRPP on 2D grids, relevant to warehouse and parcel systems. We propose the Rubik Table method, achieving $(1 + \delta)$-optimal makespan (with $\delta \in (0, 0.5]$) for up to $\frac{m_1 m_2}{2}$ robots, solving large instances efficiently and setting a new theoretical benchmark. Next, we address real-world MRPP. We design optimal layouts for structured environments (e.g., warehouses, parking systems) and propose a puzzle-based system for dense, deadlock-free autonomous vehicle parking. We also extend MRPP to Reeds-Shepp robots, introducing motion primitives and smoothing techniques to ensure feasible, efficient paths under nonholonomic constraints. Simulations and real-world tests validate the approach in urban driving and robotic transport scenarios.
>
---
#### [new 002] Design of an innovative robotic surgical instrument for circular stapling
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决传统手术工具精度低、并发症多的问题，设计了一种新型机械臂辅助的圆形吻合器。**

- **链接: [http://arxiv.org/pdf/2506.09444v1](http://arxiv.org/pdf/2506.09444v1)**

> **作者:** Paul Tucan; Nadim Al Hajjar; Calin Vaida; Alexandru Pusca; Tiberiu Antal; Corina Radu; Daniel Jucan; Adrian Pisla; Damien Chablat; Doina Pisla
>
> **摘要:** Esophageal cancer remains a highly aggressive malignancy with low survival rates, requiring advanced surgical interventions like esophagectomy. Traditional manual techniques, including circular staplers, face challenges such as limited precision, prolonged recovery times, and complications like leaks and tissue misalignment. This paper presents a novel robotic circular stapler designed to enhance the dexterity in confined spaces, improve tissue alignment, and reduce post-operative risks. Integrated with a cognitive robot that serves as a surgeon's assistant, the surgical stapler uses three actuators to perform anvil motion, cutter/stapler motion and allows a 75-degree bending of the cartridge (distal tip). Kinematic analysis is used to compute the stapler tip's position, ensuring synchronization with a robotic system.
>
---
#### [new 003] Enhancing Human-Robot Collaboration: A Sim2Real Domain Adaptation Algorithm for Point Cloud Segmentation in Industrial Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于点云语义分割任务，旨在解决工业环境中人机协作的3D环境理解问题。通过提出一种双流网络模型，实现从模拟到现实的域适应，提升分割精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.09552v1](http://arxiv.org/pdf/2506.09552v1)**

> **作者:** Fatemeh Mohammadi Amin; Darwin G. Caldwell; Hans Wernher van de Venn
>
> **备注:** Preprint, Journal of Intelligent & Robotic Systems
>
> **摘要:** The robust interpretation of 3D environments is crucial for human-robot collaboration (HRC) applications, where safety and operational efficiency are paramount. Semantic segmentation plays a key role in this context by enabling a precise and detailed understanding of the environment. Considering the intense data hunger for real-world industrial annotated data essential for effective semantic segmentation, this paper introduces a pioneering approach in the Sim2Real domain adaptation for semantic segmentation of 3D point cloud data, specifically tailored for HRC. Our focus is on developing a network that robustly transitions from simulated environments to real-world applications, thereby enhancing its practical utility and impact on a safe HRC. In this work, we propose a dual-stream network architecture (FUSION) combining Dynamic Graph Convolutional Neural Networks (DGCNN) and Convolutional Neural Networks (CNN) augmented with residual layers as a Sim2Real domain adaptation algorithm for an industrial environment. The proposed model was evaluated on real-world HRC setups and simulation industrial point clouds, it showed increased state-of-the-art performance, achieving a segmentation accuracy of 97.76%, and superior robustness compared to existing methods.
>
---
#### [new 004] Aucamp: An Underwater Camera-Based Multi-Robot Platform with Low-Cost, Distributed, and Robust Localization
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于水下多机器人定位任务，解决低成本、分布式、鲁棒的定位问题，提出Aucamp平台，结合单目视觉与分布式算法提升定位精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.09876v1](http://arxiv.org/pdf/2506.09876v1)**

> **作者:** Jisheng Xu; Ding Lin; Pangkit Fong; Chongrong Fang; Xiaoming Duan; Jianping He
>
> **摘要:** This paper introduces an underwater multi-robot platform, named Aucamp, characterized by cost-effective monocular-camera-based sensing, distributed protocol and robust orientation control for localization. We utilize the clarity feature to measure the distance, present the monocular imaging model, and estimate the position of the target object. We achieve global positioning in our platform by designing a distributed update protocol. The distributed algorithm enables the perception process to simultaneously cover a broader range, and greatly improves the accuracy and robustness of the positioning. Moreover, the explicit dynamics model of the robot in our platform is obtained, based on which, we propose a robust orientation control framework. The control system ensures that the platform maintains a balanced posture for each robot, thereby ensuring the stability of the localization system. The platform can swiftly recover from an forced unstable state to a stable horizontal posture. Additionally, we conduct extensive experiments and application scenarios to evaluate the performance of our platform. The proposed new platform may provide support for extensive marine exploration by underwater sensor networks.
>
---
#### [new 005] Analytic Task Scheduler: Recursive Least Squares Based Method for Continual Learning in Embodied Foundation Models
- **分类: cs.RO**

- **简介: 该论文属于持续学习任务，旨在解决机器人在学习新技能时的灾难性遗忘问题。提出ATS框架，通过递归最小二乘法实现任务识别与模型选择，有效防止参数干扰。**

- **链接: [http://arxiv.org/pdf/2506.09623v1](http://arxiv.org/pdf/2506.09623v1)**

> **作者:** Lipei Xie; Yingxin Li; Huiping Zhuang
>
> **摘要:** Embodied foundation models are crucial for Artificial Intelligence (AI) interacting with the physical world by integrating multi-modal inputs, such as proprioception, vision and language, to understand human intentions and generate actions to control robots. While these models demonstrate strong generalization and few-shot learning capabilities, they face significant challenges in continually acquiring new skills without forgetting previously learned skills, a problem known as catastrophic forgetting. To address this issue, we propose the Analytic Task Scheduler (ATS), a novel framework for continual learning in embodied foundation models. ATS consists of a task-specific model library, where each model is fine-tuned independently on a single task, and an analytic scheduler trained using recursive least squares (RLS) to learn the mapping between language instructions and task-specific models. This architecture enables accurate task recognition and dynamic model selection while fundamentally avoiding parameter interference across tasks. The scheduler updates its parameters incrementally using only statistics (autocorrelation and cross-correlation matrices), enabling forgetting-resistant learning without the need to revisit historical data. We validate ATS on a real-world robot platform (RM65B), demonstrating superior resistance to forgetting and strong adaptability to task variations. The results highlight ATS as an effective, scalable, and deployable solution for continual learning in embodied foundation models operating in complex, dynamic environments. Our code will be available at https://github.com/MIAA-Embodied-AI/AnalyticTaskScheduler
>
---
#### [new 006] Locomotion on Constrained Footholds via Layered Architectures and Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文属于机器人步态控制任务，解决腿式机器人在受限足点上的稳定运动问题。通过分层架构与模型预测控制，提升控制的优化性与实时性。**

- **链接: [http://arxiv.org/pdf/2506.09979v1](http://arxiv.org/pdf/2506.09979v1)**

> **作者:** Zachary Olkin; Aaron D. Ames
>
> **备注:** Submitted to Humanoids 2025
>
> **摘要:** Computing stabilizing and optimal control actions for legged locomotion in real time is difficult due to the nonlinear, hybrid, and high dimensional nature of these robots. The hybrid nature of the system introduces a combination of discrete and continuous variables which causes issues for numerical optimal control. To address these challenges, we propose a layered architecture that separates the choice of discrete variables and a smooth Model Predictive Controller (MPC). The layered formulation allows for online flexibility and optimality without sacrificing real-time performance through a combination of gradient-free and gradient-based methods. The architecture leverages a sampling-based method for determining discrete variables, and a classical smooth MPC formulation using these fixed discrete variables. We demonstrate the results on a quadrupedal robot stepping over gaps and onto terrain with varying heights. In simulation, we demonstrate the controller on a humanoid robot for gap traversal. The layered approach is shown to be more optimal and reliable than common heuristic-based approaches and faster to compute than pure sampling methods.
>
---
#### [new 007] Bipedal Balance Control with Whole-body Musculoskeletal Standing and Falling Simulations
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文属于人体平衡控制任务，旨在解决静态平衡与跌倒机制的理解问题，通过全身心肌系统模拟分析平衡动态及肌肉损伤影响。**

- **链接: [http://arxiv.org/pdf/2506.09383v1](http://arxiv.org/pdf/2506.09383v1)**

> **作者:** Chengtian Ma; Yunyue Wei; Chenhui Zuo; Chen Zhang; Yanan Sui
>
> **摘要:** Balance control is important for human and bipedal robotic systems. While dynamic balance during locomotion has received considerable attention, quantitative understanding of static balance and falling remains limited. This work presents a hierarchical control pipeline for simulating human balance via a comprehensive whole-body musculoskeletal system. We identified spatiotemporal dynamics of balancing during stable standing, revealed the impact of muscle injury on balancing behavior, and generated fall contact patterns that aligned with clinical data. Furthermore, our simulated hip exoskeleton assistance demonstrated improvement in balance maintenance and reduced muscle effort under perturbation. This work offers unique muscle-level insights into human balance dynamics that are challenging to capture experimentally. It could provide a foundation for developing targeted interventions for individuals with balance impairments and support the advancement of humanoid robotic systems.
>
---
#### [new 008] Towards Full-Scenario Safety Evaluation of Automated Vehicles: A Volume-Based Method
- **分类: cs.RO; cs.ET**

- **简介: 该论文属于自动驾驶安全评估任务，旨在解决复杂场景下安全评估不足的问题。提出一种基于体积的评估方法，简化场景维度并提高计算效率。**

- **链接: [http://arxiv.org/pdf/2506.09182v1](http://arxiv.org/pdf/2506.09182v1)**

> **作者:** Hang Zhou; Chengyuan Ma; Shiyu Shen; Xiaopeng Li
>
> **备注:** NA
>
> **摘要:** With the rapid development of automated vehicles (AVs) in recent years, commercially available AVs are increasingly demonstrating high-level automation capabilities. However, most existing AV safety evaluation methods are primarily designed for simple maneuvers such as car-following and lane-changing. While suitable for basic tests, these methods are insufficient for assessing high-level automation functions deployed in more complex environments. First, these methods typically use crash rate as the evaluation metric, whose accuracy heavily depends on the quality and completeness of naturalistic driving environment data used to estimate scenario probabilities. Such data is often difficult and expensive to collect. Second, when applied to diverse scenarios, these methods suffer from the curse of dimensionality, making large-scale evaluation computationally intractable. To address these challenges, this paper proposes a novel framework for full-scenario AV safety evaluation. A unified model is first introduced to standardize the representation of diverse driving scenarios. This modeling approach constrains the dimension of most scenarios to a regular highway setting with three lanes and six surrounding background vehicles, significantly reducing dimensionality. To further avoid the limitations of probability-based method, we propose a volume-based evaluation method that quantifies the proportion of risky scenarios within the entire scenario space. For car-following scenarios, we prove that the set of safe scenarios is convex under specific settings, enabling exact volume computation. Experimental results validate the effectiveness of the proposed volume-based method using both AV behavior models from existing literature and six production AV models calibrated from field-test trajectory data in the Ultra-AV dataset. Code and data will be made publicly available upon acceptance of this paper.
>
---
#### [new 009] Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation
- **分类: cs.RO; cs.AI; cs.GR**

- **简介: 该论文属于自动驾驶安全测试任务，旨在解决真实数据中稀有事故场景不足的问题。通过提出Adv-BMT框架生成多样化且真实的碰撞场景。**

- **链接: [http://arxiv.org/pdf/2506.09485v1](http://arxiv.org/pdf/2506.09485v1)**

> **作者:** Yuxin Liu; Zhenghao Peng; Xuanhao Cui; Bolei Zhou
>
> **摘要:** Scenario-based testing is essential for validating the performance of autonomous driving (AD) systems. However, such testing is limited by the scarcity of long-tailed, safety-critical scenarios in existing datasets collected in the real world. To tackle the data issue, we propose the Adv-BMT framework, which augments real-world scenarios with diverse and realistic adversarial interactions. The core component of Adv-BMT is a bidirectional motion transformer (BMT) model to perform inverse traffic motion predictions, which takes agent information in the last time step of the scenario as input, and reconstruct the traffic in the inverse of chronological order until the initial time step. The Adv-BMT framework is a two-staged pipeline: it first conducts adversarial initializations and then inverse motion predictions. Different from previous work, we do not need any collision data for pretraining, and are able to generate realistic and diverse collision interactions. Our experimental results validate the quality of generated collision scenarios by Adv-BMT: training in our augmented dataset would reduce episode collision rates by 20\% compared to previous work.
>
---
#### [new 010] Learning to Optimize Package Picking for Large-Scale, Real-World Robot Induction
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于仓库自动化任务，旨在优化机器人拣选效率。通过机器学习提升拣选成功率，解决传统方法依赖启发式策略的问题。**

- **链接: [http://arxiv.org/pdf/2506.09765v1](http://arxiv.org/pdf/2506.09765v1)**

> **作者:** Shuai Li; Azarakhsh Keipour; Sicong Zhao; Srinath Rajagopalan; Charles Swan; Kostas E. Bekris
>
> **备注:** The 19th International Symposium on Experimental Robotics (ISER 2025); 6-10 July 2025, Santa Fe, New Mexico, USA; 10 pages
>
> **摘要:** Warehouse automation plays a pivotal role in enhancing operational efficiency, minimizing costs, and improving resilience to workforce variability. While prior research has demonstrated the potential of machine learning (ML) models to increase picking success rates in large-scale robotic fleets by prioritizing high-probability picks and packages, these efforts primarily focused on predicting success probabilities for picks sampled using heuristic methods. Limited attention has been given, however, to leveraging data-driven approaches to directly optimize sampled picks for better performance at scale. In this study, we propose an ML-based framework that predicts transform adjustments as well as improving the selection of suction cups for multi-suction end effectors for sampled picks to enhance their success probabilities. The framework was integrated and evaluated in test workcells that resemble the operations of Amazon Robotics' Robot Induction (Robin) fleet, which is used for package manipulation. Evaluated on over 2 million picks, the proposed method achieves a 20\% reduction in pick failure rates compared to a heuristic-based pick sampling baseline, demonstrating its effectiveness in large-scale warehouse automation scenarios.
>
---
#### [new 011] Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation with Heterogeneous Constraints
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决动态环境中带异构约束的路径规划问题。提出一种分层框架，结合强化学习与优化方法，提升导航安全与效率。**

- **链接: [http://arxiv.org/pdf/2506.09859v1](http://arxiv.org/pdf/2506.09859v1)**

> **作者:** Huajian Liu; Yixuan Feng; Wei Dong; Kunpeng Fan; Chao Wang; Yongzhuo Gao
>
> **摘要:** In this paper, we propose a novel hierarchical framework for robot navigation in dynamic environments with heterogeneous constraints. Our approach leverages a graph neural network trained via reinforcement learning (RL) to efficiently estimate the robot's cost-to-go, formulated as local goal recommendations. A spatio-temporal path-searching module, which accounts for kinematic constraints, is then employed to generate a reference trajectory to facilitate solving the non-convex optimization problem used for explicit constraint enforcement. More importantly, we introduce an incremental action-masking mechanism and a privileged learning strategy, enabling end-to-end training of the proposed planner. Both simulation and real-world experiments demonstrate that the proposed method effectively addresses local planning in complex dynamic environments, achieving state-of-the-art (SOTA) performance. Compared with existing learning-optimization hybrid methods, our approach eliminates the dependency on high-fidelity simulation environments, offering significant advantages in computational efficiency and training scalability. The code will be released as open-source upon acceptance of the paper.
>
---
#### [new 012] Hearing the Slide: Acoustic-Guided Constraint Learning for Fast Non-Prehensile Transport
- **分类: cs.RO**

- **简介: 该论文属于机器人非抓取运输任务，旨在解决高速运动中摩擦模型不准确导致物体滑动的问题。通过声学传感学习动态摩擦系数，提升运输稳定性。**

- **链接: [http://arxiv.org/pdf/2506.09169v1](http://arxiv.org/pdf/2506.09169v1)**

> **作者:** Yuemin Mao; Bardienus P. Duisterhof; Moonyoung Lee; Jeffrey Ichnowski
>
> **摘要:** Object transport tasks are fundamental in robotic automation, emphasizing the importance of efficient and secure methods for moving objects. Non-prehensile transport can significantly improve transport efficiency, as it enables handling multiple objects simultaneously and accommodating objects unsuitable for parallel-jaw or suction grasps. Existing approaches incorporate constraints based on the Coulomb friction model, which is imprecise during fast motions where inherent mechanical vibrations occur. Imprecise constraints can cause transported objects to slide or even fall off the tray. To address this limitation, we propose a novel method to learn a friction model using acoustic sensing that maps a tray's motion profile to a dynamically conditioned friction coefficient. This learned model enables an optimization-based motion planner to adjust the friction constraint at each control step according to the planned motion at that step. In experiments, we generate time-optimized trajectories for a UR5e robot to transport various objects with constraints using both the standard Coulomb friction model and the learned friction model. Results suggest that the learned friction model reduces object displacement by up to 86.0% compared to the baseline, highlighting the effectiveness of acoustic sensing in learning real-world friction constraints.
>
---
#### [new 013] Time-Unified Diffusion Policy with Action Discrimination for Robotic Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操作任务，解决扩散模型生成动作效率低、精度不足的问题。提出TUDP方法，通过时间统一和动作判别提升动作生成效率与准确性。**

- **链接: [http://arxiv.org/pdf/2506.09422v1](http://arxiv.org/pdf/2506.09422v1)**

> **作者:** Ye Niu; Sanping Zhou; Yizhe Li; Ye Den; Le Wang
>
> **摘要:** In many complex scenarios, robotic manipulation relies on generative models to estimate the distribution of multiple successful actions. As the diffusion model has better training robustness than other generative models, it performs well in imitation learning through successful robot demonstrations. However, the diffusion-based policy methods typically require significant time to iteratively denoise robot actions, which hinders real-time responses in robotic manipulation. Moreover, existing diffusion policies model a time-varying action denoising process, whose temporal complexity increases the difficulty of model training and leads to suboptimal action accuracy. To generate robot actions efficiently and accurately, we present the Time-Unified Diffusion Policy (TUDP), which utilizes action recognition capabilities to build a time-unified denoising process. On the one hand, we build a time-unified velocity field in action space with additional action discrimination information. By unifying all timesteps of action denoising, our velocity field reduces the difficulty of policy learning and speeds up action generation. On the other hand, we propose an action-wise training method, which introduces an action discrimination branch to supply additional action discrimination information. Through action-wise training, the TUDP implicitly learns the ability to discern successful actions to better denoising accuracy. Our method achieves state-of-the-art performance on RLBench with the highest success rate of 82.6% on a multi-view setup and 83.8% on a single-view setup. In particular, when using fewer denoising iterations, TUDP achieves a more significant improvement in success rate. Additionally, TUDP can produce accurate actions for a wide range of real-world tasks.
>
---
#### [new 014] Advances on Affordable Hardware Platforms for Human Demonstration Acquisition in Agricultural Applications
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决农业场景下低成本示范采集问题。通过优化样本获取和轨迹生成，提升示范数据质量与可靠性。**

- **链接: [http://arxiv.org/pdf/2506.09494v1](http://arxiv.org/pdf/2506.09494v1)**

> **作者:** Alberto San-Miguel-Tello; Gennaro Scarati; Alejandro Hernández; Mario Cavero-Vidal; Aakash Maroti; Néstor García
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** This paper presents advances on the Universal Manipulation Interface (UMI), a low-cost hand-held gripper for robot Learning from Demonstration (LfD), for complex in-the-wild scenarios found in agricultural settings. The focus is on improving the acquisition of suitable samples with minimal additional setup. Firstly, idle times and user's cognitive load are reduced through the extraction of individual samples from a continuous demonstration considering task events. Secondly, reliability on the generation of task sample's trajectories is increased through the combination on-board inertial measurements and external visual marker localization usage using Extended Kalman Filtering (EKF). Results are presented for a fruit harvesting task, outperforming the default pipeline.
>
---
#### [new 015] R-CARLA: High-Fidelity Sensor Simulations with Interchangeable Dynamics for Autonomous Racing
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶领域，解决仿真环境不足的问题。提出R-CARLA，融合高精度车辆动力学与传感器模拟，减少真实世界差距。**

- **链接: [http://arxiv.org/pdf/2506.09629v1](http://arxiv.org/pdf/2506.09629v1)**

> **作者:** Maurice Brunner; Edoardo Ghignone; Nicolas Baumann; Michele Magno
>
> **摘要:** Autonomous racing has emerged as a crucial testbed for autonomous driving algorithms, necessitating a simulation environment for both vehicle dynamics and sensor behavior. Striking the right balance between vehicle dynamics and sensor accuracy is crucial for pushing vehicles to their performance limits. However, autonomous racing developers often face a trade-off between accurate vehicle dynamics and high-fidelity sensor simulations. This paper introduces R-CARLA, an enhancement of the CARLA simulator that supports holistic full-stack testing, from perception to control, using a single system. By seamlessly integrating accurate vehicle dynamics with sensor simulations, opponents simulation as NPCs, and a pipeline for creating digital twins from real-world robotic data, R-CARLA empowers researchers to push the boundaries of autonomous racing development. Furthermore, it is developed using CARLA's rich suite of sensor simulations. Our results indicate that incorporating the proposed digital-twin framework into R-CARLA enables more realistic full-stack testing, demonstrating a significant reduction in the Sim-to-Real gap of car dynamics simulation by 42% and by 82% in the case of sensor simulation across various testing scenarios.
>
---
#### [new 016] UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决无监督泛化问题。通过UAD方法，从基础模型中蒸馏可操作性知识，无需人工标注，提升机器人对未知对象和任务的适应能力。**

- **链接: [http://arxiv.org/pdf/2506.09284v1](http://arxiv.org/pdf/2506.09284v1)**

> **作者:** Yihe Tang; Wenlong Huang; Yingke Wang; Chengshu Li; Roy Yuan; Ruohan Zhang; Jiajun Wu; Li Fei-Fei
>
> **摘要:** Understanding fine-grained object affordances is imperative for robots to manipulate objects in unstructured environments given open-ended task instructions. However, existing methods of visual affordance predictions often rely on manually annotated data or conditions only on a predefined set of tasks. We introduce UAD (Unsupervised Affordance Distillation), a method for distilling affordance knowledge from foundation models into a task-conditioned affordance model without any manual annotations. By leveraging the complementary strengths of large vision models and vision-language models, UAD automatically annotates a large-scale dataset with detailed $<$instruction, visual affordance$>$ pairs. Training only a lightweight task-conditioned decoder atop frozen features, UAD exhibits notable generalization to in-the-wild robotic scenes and to various human activities, despite only being trained on rendered objects in simulation. Using affordance provided by UAD as the observation space, we show an imitation learning policy that demonstrates promising generalization to unseen object instances, object categories, and even variations in task instructions after training on as few as 10 demonstrations. Project website: https://unsup-affordance.github.io/
>
---
#### [new 017] Human-robot collaborative transport personalization via Dynamic Movement Primitives and velocity scaling
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决个性化轨迹规划问题。通过动态运动基元和速度调整，生成适应人类特征的运输路径。**

- **链接: [http://arxiv.org/pdf/2506.09697v1](http://arxiv.org/pdf/2506.09697v1)**

> **作者:** Paolo Franceschi; Andrea Bussolan; Vincenzo Pomponi; Oliver Avram; Stefano Baraldo; Anna Valente
>
> **摘要:** Nowadays, industries are showing a growing interest in human-robot collaboration, particularly for shared tasks. This requires intelligent strategies to plan a robot's motions, considering both task constraints and human-specific factors such as height and movement preferences. This work introduces a novel approach to generate personalized trajectories using Dynamic Movement Primitives (DMPs), enhanced with real-time velocity scaling based on human feedback. The method was rigorously tested in industrial-grade experiments, focusing on the collaborative transport of an engine cowl lip section. Comparative analysis between DMP-generated trajectories and a state-of-the-art motion planner (BiTRRT) highlights their adaptability combined with velocity scaling. Subjective user feedback further demonstrates a clear preference for DMP- based interactions. Objective evaluations, including physiological measurements from brain and skin activity, reinforce these findings, showcasing the advantages of DMPs in enhancing human-robot interaction and improving user experience.
>
---
#### [new 018] Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule
- **分类: cs.RO; cs.CV; stat.AP**

- **简介: 该论文属于自动驾驶感知系统评估任务，旨在解决静态指标无法反映动态环境下的感知稳定性问题。提出PCD度量，结合不确定性分析，评估检测距离可靠性。**

- **链接: [http://arxiv.org/pdf/2506.09217v1](http://arxiv.org/pdf/2506.09217v1)**

> **作者:** Boyu Jiang; Liang Shi; Zhengzhi Lin; Loren Stowe; Feng Guo
>
> **摘要:** The performance of perception systems in autonomous driving systems (ADS) is strongly influenced by object distance, scene dynamics, and environmental conditions such as weather. AI-based perception outputs are inherently stochastic, with variability driven by these external factors, while traditional evaluation metrics remain static and event-independent, failing to capture fluctuations in confidence over time. In this work, we introduce the Perception Characteristics Distance (PCD) -- a novel evaluation metric that quantifies the farthest distance at which an object can be reliably detected, incorporating uncertainty in model outputs. To support this, we present the SensorRainFall dataset, collected on the Virginia Smart Road using a sensor-equipped vehicle (cameras, radar, LiDAR) under controlled daylight-clear and daylight-rain scenarios, with precise ground-truth distances to the target objects. Statistical analysis reveals the presence of change points in the variance of detection confidence score with distance. By averaging the PCD values across a range of detection quality thresholds and probabilistic thresholds, we compute the mean PCD (mPCD), which captures the overall perception characteristics of a system with respect to detection distance. Applying state-of-the-art perception models shows that mPCD captures meaningful reliability differences under varying weather conditions -- differences that static metrics overlook. PCD provides a principled, distribution-aware measure of perception performance, supporting safer and more robust ADS operation, while the SensorRainFall dataset offers a valuable benchmark for evaluation. The SensorRainFall dataset is publicly available at https://www.kaggle.com/datasets/datadrivenwheels/sensorrainfall, and the evaluation code is open-sourced at https://github.com/datadrivenwheels/PCD_Python.
>
---
#### [new 019] SAFE: Multitask Failure Detection for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决VLAs在新任务中失败检测的问题。提出SAFE模型，利用VLA内部特征进行多任务失败预测，提升检测准确性和效率。**

- **链接: [http://arxiv.org/pdf/2506.09937v1](http://arxiv.org/pdf/2506.09937v1)**

> **作者:** Qiao Gu; Yuanliang Ju; Shengxiang Sun; Igor Gilitschenski; Haruki Nishimura; Masha Itkina; Florian Shkurti
>
> **备注:** Project Page: https://vla-safe.github.io/
>
> **摘要:** While vision-language-action models (VLAs) have shown promising robotic behaviors across a diverse set of manipulation tasks, they achieve limited success rates when deployed on novel tasks out-of-the-box. To allow these policies to safely interact with their environments, we need a failure detector that gives a timely alert such that the robot can stop, backtrack, or ask for help. However, existing failure detectors are trained and tested only on one or a few specific tasks, while VLAs require the detector to generalize and detect failures also in unseen tasks and novel environments. In this paper, we introduce the multitask failure detection problem and propose SAFE, a failure detector for generalist robot policies such as VLAs. We analyze the VLA feature space and find that VLAs have sufficient high-level knowledge about task success and failure, which is generic across different tasks. Based on this insight, we design SAFE to learn from VLA internal features and predict a single scalar indicating the likelihood of task failure. SAFE is trained on both successful and failed rollouts, and is evaluated on unseen tasks. SAFE is compatible with different policy architectures. We test it on OpenVLA, $\pi_0$, and $\pi_0$-FAST in both simulated and real-world environments extensively. We compare SAFE with diverse baselines and show that SAFE achieves state-of-the-art failure detection performance and the best trade-off between accuracy and detection time using conformal prediction. More qualitative results can be found at https://vla-safe.github.io/.
>
---
#### [new 020] Attention-Based Map Encoding for Learning Generalized Legged Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决腿式机器人在复杂地形中的泛化行走问题。通过学习注意力地图编码，提升控制器的鲁棒性和精度。**

- **链接: [http://arxiv.org/pdf/2506.09588v1](http://arxiv.org/pdf/2506.09588v1)**

> **作者:** Junzhe He; Chong Zhang; Fabian Jenelten; Ruben Grandia; Moritz BÄcher; Marco Hutter
>
> **备注:** Original draft prior to peer review. Significant revisions and new materials are expected after formal publication release
>
> **摘要:** Dynamic locomotion of legged robots is a critical yet challenging topic in expanding the operational range of mobile robots. It requires precise planning when possible footholds are sparse, robustness against uncertainties and disturbances, and generalizability across diverse terrains. While traditional model-based controllers excel at planning on complex terrains, they struggle with real-world uncertainties. Learning-based controllers offer robustness to such uncertainties but often lack precision on terrains with sparse steppable areas. Hybrid methods achieve enhanced robustness on sparse terrains by combining both methods but are computationally demanding and constrained by the inherent limitations of model-based planners. To achieve generalized legged locomotion on diverse terrains while preserving the robustness of learning-based controllers, this paper proposes to learn an attention-based map encoding conditioned on robot proprioception, which is trained as part of the end-to-end controller using reinforcement learning. We show that the network learns to focus on steppable areas for future footholds when the robot dynamically navigates diverse and challenging terrains. We synthesize behaviors that exhibit robustness against uncertainties while enabling precise and agile traversal of sparse terrains. Additionally, our method offers a way to interpret the topographical perception of a neural network. We have trained two controllers for a 12-DoF quadrupedal robot and a 23-DoF humanoid robot respectively and tested the resulting controllers in the real world under various challenging indoor and outdoor scenarios, including ones unseen during training.
>
---
#### [new 021] WD-DETR: Wavelet Denoising-Enhanced Real-Time Object Detection Transformer for Robot Perception with Event Cameras
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于实时目标检测任务，针对事件相机中密集事件表示的噪声问题，提出WD-DETR网络，结合小波去噪和Transformer提升检测性能。**

- **链接: [http://arxiv.org/pdf/2506.09098v1](http://arxiv.org/pdf/2506.09098v1)**

> **作者:** Yangjie Cui; Boyang Gao; Yiwei Zhang; Xin Dong; Jinwu Xiang; Daochun Li; Zhan Tu
>
> **备注:** https://youtu.be/AQAgVdrx1DE
>
> **摘要:** Previous studies on event camera sensing have demonstrated certain detection performance using dense event representations. However, the accumulated noise in such dense representations has received insufficient attention, which degrades the representation quality and increases the likelihood of missed detections. To address this challenge, we propose the Wavelet Denoising-enhanced DEtection TRansformer, i.e., WD-DETR network, for event cameras. In particular, a dense event representation is presented first, which enables real-time reconstruction of events as tensors. Then, a wavelet transform method is designed to filter noise in the event representations. Such a method is integrated into the backbone for feature extraction. The extracted features are subsequently fed into a transformer-based network for object prediction. To further reduce inference time, we incorporate the Dynamic Reorganization Convolution Block (DRCB) as a fusion module within the hybrid encoder. The proposed method has been evaluated on three event-based object detection datasets, i.e., DSEC, Gen1, and 1Mpx. The results demonstrate that WD-DETR outperforms tested state-of-the-art methods. Additionally, we implement our approach on a common onboard computer for robots, the NVIDIA Jetson Orin NX, achieving a high frame rate of approximately 35 FPS using TensorRT FP16, which is exceptionally well-suited for real-time perception of onboard robotic systems.
>
---
#### [new 022] Reinforced Refinement with Self-Aware Expansion for End-to-End Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于端到端自动驾驶任务，解决模型泛化能力差和缺乏修正反馈的问题。提出R2SE方法，通过强化学习提升模型在复杂场景下的表现与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.09800v1](http://arxiv.org/pdf/2506.09800v1)**

> **作者:** Haochen Liu; Tianyu Li; Haohan Yang; Li Chen; Caojun Wang; Ke Guo; Haochen Tian; Hongchen Li; Hongyang Li; Chen Lv
>
> **摘要:** End-to-end autonomous driving has emerged as a promising paradigm for directly mapping sensor inputs to planning maneuvers using learning-based modular integrations. However, existing imitation learning (IL)-based models suffer from generalization to hard cases, and a lack of corrective feedback loop under post-deployment. While reinforcement learning (RL) offers a potential solution to tackle hard cases with optimality, it is often hindered by overfitting to specific driving cases, resulting in catastrophic forgetting of generalizable knowledge and sample inefficiency. To overcome these challenges, we propose Reinforced Refinement with Self-aware Expansion (R2SE), a novel learning pipeline that constantly refines hard domain while keeping generalizable driving policy for model-agnostic end-to-end driving systems. Through reinforcement fine-tuning and policy expansion that facilitates continuous improvement, R2SE features three key components: 1) Generalist Pretraining with hard-case allocation trains a generalist imitation learning (IL) driving system while dynamically identifying failure-prone cases for targeted refinement; 2) Residual Reinforced Specialist Fine-tuning optimizes residual corrections using reinforcement learning (RL) to improve performance in hard case domain while preserving global driving knowledge; 3) Self-aware Adapter Expansion dynamically integrates specialist policies back into the generalist model, enhancing continuous performance improvement. Experimental results in closed-loop simulation and real-world datasets demonstrate improvements in generalization, safety, and long-horizon policy robustness over state-of-the-art E2E systems, highlighting the effectiveness of reinforce refinement for scalable autonomous driving.
>
---
#### [new 023] Tightly-Coupled LiDAR-IMU-Leg Odometry with Online Learned Leg Kinematics Incorporating Foot Tactile Information
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人定位任务，解决复杂地形下的里程计问题。通过融合LiDAR、IMU和足部触觉信息，提出在线学习的腿部运动模型，提升定位精度与适应性。**

- **链接: [http://arxiv.org/pdf/2506.09548v1](http://arxiv.org/pdf/2506.09548v1)**

> **作者:** Taku Okawara; Kenji Koide; Aoki Takanose; Shuji Oishi; Masashi Yokozuka; Kentaro Uno; Kazuya Yoshida
>
> **备注:** Robotics and Automation Letters
>
> **摘要:** In this letter, we present tightly coupled LiDAR-IMU-leg odometry, which is robust to challenging conditions such as featureless environments and deformable terrains. We developed an online learning-based leg kinematics model named the neural leg kinematics model, which incorporates tactile information (foot reaction force) to implicitly express the nonlinear dynamics between robot feet and the ground. Online training of this model enhances its adaptability to weight load changes of a robot (e.g., assuming delivery or transportation tasks) and terrain conditions. According to the \textit{neural adaptive leg odometry factor} and online uncertainty estimation of the leg kinematics model-based motion predictions, we jointly solve online training of this kinematics model and odometry estimation on a unified factor graph to retain the consistency of both. The proposed method was verified through real experiments using a quadruped robot in two challenging situations: 1) a sandy beach, representing an extremely featureless area with a deformable terrain, and 2) a campus, including multiple featureless areas and terrain types of asphalt, gravel (deformable terrain), and grass. Experimental results showed that our odometry estimation incorporating the \textit{neural leg kinematics model} outperforms state-of-the-art works. Our project page is available for further details: https://takuokawara.github.io/RAL2025_project_page/
>
---
#### [new 024] VAULT: A Mobile Mapping System for ROS 2-based Autonomous Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，旨在解决户外环境下的精准定位与建图问题。通过融合多种传感器和算法，提出VAULT系统实现可靠3D定位与地图构建。**

- **链接: [http://arxiv.org/pdf/2506.09583v1](http://arxiv.org/pdf/2506.09583v1)**

> **作者:** Miguel Á. González-Santamarta; Francisco J. Rodríguez-Lera; Vicente Matellán-Olivera
>
> **备注:** 15 pages, 5 figures, Submitted to WAF 2023: Workshop de Agentes Fisicos
>
> **摘要:** Localization plays a crucial role in the navigation capabilities of autonomous robots, and while indoor environments can rely on wheel odometry and 2D LiDAR-based mapping, outdoor settings such as agriculture and forestry, present unique challenges that necessitate real-time localization and consistent mapping. Addressing this need, this paper introduces the VAULT prototype, a ROS 2-based mobile mapping system (MMS) that combines various sensors to enable robust outdoor and indoor localization. The proposed solution harnesses the power of Global Navigation Satellite System (GNSS) data, visual-inertial odometry (VIO), inertial measurement unit (IMU) data, and the Extended Kalman Filter (EKF) to generate reliable 3D odometry. To further enhance the localization accuracy, Visual SLAM (VSLAM) is employed, resulting in the creation of a comprehensive 3D point cloud map. By leveraging these sensor technologies and advanced algorithms, the prototype offers a comprehensive solution for outdoor localization in autonomous mobile robots, enabling them to navigate and map their surroundings with confidence and precision.
>
---
#### [new 025] Chain-of-Action: Trajectory Autoregressive Modeling for Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出CoA，一种基于轨迹自回归建模的视觉-运动策略，用于机器人操作任务。旨在解决轨迹生成与目标约束问题，通过逆向推理生成全局一致的行动序列。**

- **链接: [http://arxiv.org/pdf/2506.09990v1](http://arxiv.org/pdf/2506.09990v1)**

> **作者:** Wenbo Zhang; Tianrun Hu; Yanyuan Qiao; Hanbo Zhang; Yuchu Qin; Yang Li; Jiajun Liu; Tao Kong; Lingqiao Liu; Xiao Ma
>
> **摘要:** We present Chain-of-Action (CoA), a novel visuo-motor policy paradigm built upon Trajectory Autoregressive Modeling. Unlike conventional approaches that predict next step action(s) forward, CoA generates an entire trajectory by explicit backward reasoning with task-specific goals through an action-level Chain-of-Thought (CoT) process. This process is unified within a single autoregressive structure: (1) the first token corresponds to a stable keyframe action that encodes the task-specific goals; and (2) subsequent action tokens are generated autoregressively, conditioned on the initial keyframe and previously predicted actions. This backward action reasoning enforces a global-to-local structure, allowing each local action to be tightly constrained by the final goal. To further realize the action reasoning structure, CoA incorporates four complementary designs: continuous action token representation; dynamic stopping for variable-length trajectory generation; reverse temporal ensemble; and multi-token prediction to balance action chunk modeling with global structure. As a result, CoA gives strong spatial generalization capabilities while preserving the flexibility and simplicity of a visuo-motor policy. Empirically, we observe CoA achieves the state-of-the-art performance across 60 RLBench tasks and 8 real-world manipulation tasks.
>
---
#### [new 026] Scoop-and-Toss: Dynamic Object Collection for Quadrupedal Systems
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人动态物体收集任务，解决四足机器人无额外执行器抓取物体的问题。通过腿部附加装置和分层策略实现物体 scooping 和 tossing。**

- **链接: [http://arxiv.org/pdf/2506.09406v1](http://arxiv.org/pdf/2506.09406v1)**

> **作者:** Minji Kang; Chanwoo Baek; Yoonsang Lee
>
> **摘要:** Quadruped robots have made significant advances in locomotion, extending their capabilities from controlled environments to real-world applications. Beyond movement, recent work has explored loco-manipulation using the legs to perform tasks such as pressing buttons or opening doors. While these efforts demonstrate the feasibility of leg-based manipulation, most have focused on relatively static tasks. In this work, we propose a framework that enables quadruped robots to collect objects without additional actuators by leveraging the agility of their legs. By attaching a simple scoop-like add-on to one leg, the robot can scoop objects and toss them into a collection tray mounted on its back. Our method employs a hierarchical policy structure comprising two expert policies-one for scooping and tossing, and one for approaching object positions-and a meta-policy that dynamically switches between them. The expert policies are trained separately, followed by meta-policy training for coordinated multi-object collection. This approach demonstrates how quadruped legs can be effectively utilized for dynamic object manipulation, expanding their role beyond locomotion.
>
---
#### [new 027] Fluoroscopic Shape and Pose Tracking of Catheters with Custom Radiopaque Markers
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于医学影像中的导管跟踪任务，旨在解决微导管形状与姿态估计问题。通过定制放射显影标记，实现高精度的导管跟踪。**

- **链接: [http://arxiv.org/pdf/2506.09934v1](http://arxiv.org/pdf/2506.09934v1)**

> **作者:** Jared Lawson; Rohan Chitale; Nabil Simaan
>
> **备注:** 8 pages, 5 figures, accepted in Robotics and Automation Letters
>
> **摘要:** Safe navigation of steerable and robotic catheters in the cerebral vasculature requires awareness of the catheters shape and pose. Currently, a significant perception burden is placed on interventionalists to mentally reconstruct and predict catheter motions from biplane fluoroscopy images. Efforts to track these catheters are limited to planar segmentation or bulky sensing instrumentation, which are incompatible with microcatheters used in neurointervention. In this work, a catheter is equipped with custom radiopaque markers arranged to enable simultaneous shape and pose estimation under biplane fluoroscopy. A design measure is proposed to guide the arrangement of these markers to minimize sensitivity to marker tracking uncertainty. This approach was deployed for microcatheters smaller than 2mm OD navigating phantom vasculature with shape tracking errors less than 1mm and catheter roll errors below 40 degrees. This work can enable steerable catheters to autonomously navigate under biplane imaging.
>
---
#### [new 028] Integrating Quantized LLMs into Robotics Systems as Edge AI to Leverage their Natural Language Processing Capabilities
- **分类: cs.RO**

- **简介: 该论文属于机器人与AI融合任务，旨在解决资源受限环境下LLM部署问题，通过llama_ros工具实现量化LLM的高效边缘计算应用。**

- **链接: [http://arxiv.org/pdf/2506.09581v1](http://arxiv.org/pdf/2506.09581v1)**

> **作者:** Miguel Á. González-Santamarta; Francisco J. Rodríguez-Lera; David Sobrín-Hidalgo; Ángel Manuel Guerrero-Higueras; Vicente MatellÁn-Olivera
>
> **备注:** 10 pages, 4 figures, Submitted to 3rd edition of the Workshop on Ontologies and Standards for Robotics and Automation (WOSRA) at ICRA 2024
>
> **摘要:** Large Language Models (LLMs) have experienced great advancements in the last year resulting in an increase of these models in several fields to face natural language tasks. The integration of these models in robotics can also help to improve several aspects such as human-robot interaction, navigation, planning and decision-making. Therefore, this paper introduces llama\_ros, a tool designed to integrate quantized Large Language Models (LLMs) into robotic systems using ROS 2. Leveraging llama.cpp, a highly optimized runtime engine, llama\_ros enables the efficient execution of quantized LLMs as edge artificial intelligence (AI) in robotics systems with resource-constrained environments, addressing the challenges of computational efficiency and memory limitations. By deploying quantized LLMs, llama\_ros empowers robots to leverage the natural language understanding and generation for enhanced decision-making and interaction which can be paired with prompt engineering, knowledge graphs, ontologies or other tools to improve the capabilities of autonomous robots. Additionally, this paper provides insights into some use cases of using llama\_ros for planning and explainability in robotics.
>
---
#### [new 029] From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型研究，旨在解决机器人策略泛化能力不足的问题。通过构建统一的仿真任务集评估VLA模型，发现其感知能力强但执行不稳定，需进一步优化。**

- **链接: [http://arxiv.org/pdf/2506.09930v1](http://arxiv.org/pdf/2506.09930v1)**

> **作者:** Irving Fang; Juexiao Zhang; Shengbang Tong; Chen Feng
>
> **备注:** Under review
>
> **摘要:** One promise that Vision-Language-Action (VLA) models hold over traditional imitation learning for robotics is to leverage the broad generalization capabilities of large Vision-Language Models (VLMs) to produce versatile, "generalist" robot policies. However, current evaluations of VLAs remain insufficient. Traditional imitation learning benchmarks are unsuitable due to the lack of language instructions. Emerging benchmarks for VLAs that incorporate language often come with limited evaluation tasks and do not intend to investigate how much VLM pretraining truly contributes to the generalization capabilities of the downstream robotic policy. Meanwhile, much research relies on real-world robot setups designed in isolation by different institutions, which creates a barrier for reproducibility and accessibility. To address this gap, we introduce a unified probing suite of 50 simulation-based tasks across 10 subcategories spanning language instruction, vision, and objects. We systematically evaluate several state-of-the-art VLA architectures on this suite to understand their generalization capability. Our results show that while VLM backbones endow VLAs with robust perceptual understanding and high level planning, which we refer to as good intentions, this does not reliably translate into precise motor execution: when faced with out-of-distribution observations, policies often exhibit coherent intentions, but falter in action execution. Moreover, finetuning on action data can erode the original VLM's generalist reasoning abilities. We release our task suite and evaluation code to serve as a standardized benchmark for future VLAs and to drive research on closing the perception-to-action gap. More information, including the source code, can be found at https://ai4ce.github.io/INT-ACT/
>
---
#### [new 030] SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出SkillBlender，解决人形机器人在多样化任务中灵活性不足的问题，通过技能融合实现高效运动与操作控制。**

- **链接: [http://arxiv.org/pdf/2506.09366v1](http://arxiv.org/pdf/2506.09366v1)**

> **作者:** Yuxuan Kuang; Haoran Geng; Amine Elhafsi; Tan-Dzung Do; Pieter Abbeel; Jitendra Malik; Marco Pavone; Yue Wang
>
> **摘要:** Humanoid robots hold significant potential in accomplishing daily tasks across diverse environments thanks to their flexibility and human-like morphology. Recent works have made significant progress in humanoid whole-body control and loco-manipulation leveraging optimal control or reinforcement learning. However, these methods require tedious task-specific tuning for each task to achieve satisfactory behaviors, limiting their versatility and scalability to diverse tasks in daily scenarios. To that end, we introduce SkillBlender, a novel hierarchical reinforcement learning framework for versatile humanoid loco-manipulation. SkillBlender first pretrains goal-conditioned task-agnostic primitive skills, and then dynamically blends these skills to accomplish complex loco-manipulation tasks with minimal task-specific reward engineering. We also introduce SkillBench, a parallel, cross-embodiment, and diverse simulated benchmark containing three embodiments, four primitive skills, and eight challenging loco-manipulation tasks, accompanied by a set of scientific evaluation metrics balancing accuracy and feasibility. Extensive simulated experiments show that our method significantly outperforms all baselines, while naturally regularizing behaviors to avoid reward hacking, resulting in more accurate and feasible movements for diverse loco-manipulation tasks in our daily scenarios. Our code and benchmark will be open-sourced to the community to facilitate future research. Project page: https://usc-gvl.github.io/SkillBlender-web/.
>
---
#### [new 031] DCIRNet: Depth Completion with Iterative Refinement for Dexterous Grasping of Transparent and Reflective Objects
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于深度补全任务，旨在解决透明和反光物体深度信息缺失问题。通过融合RGB图像与深度图，提出DCIRNet模型提升深度估计质量，并在抓取任务中取得显著效果。**

- **链接: [http://arxiv.org/pdf/2506.09491v1](http://arxiv.org/pdf/2506.09491v1)**

> **作者:** Guanghu Xie; Zhiduo Jiang; Yonglong Zhang; Yang Liu; Zongwu Xie; Baoshi Cao; Hong Liu
>
> **摘要:** Transparent and reflective objects in everyday environments pose significant challenges for depth sensors due to their unique visual properties, such as specular reflections and light transmission. These characteristics often lead to incomplete or inaccurate depth estimation, which severely impacts downstream geometry-based vision tasks, including object recognition, scene reconstruction, and robotic manipulation. To address the issue of missing depth information in transparent and reflective objects, we propose DCIRNet, a novel multimodal depth completion network that effectively integrates RGB images and depth maps to enhance depth estimation quality. Our approach incorporates an innovative multimodal feature fusion module designed to extract complementary information between RGB images and incomplete depth maps. Furthermore, we introduce a multi-stage supervision and depth refinement strategy that progressively improves depth completion and effectively mitigates the issue of blurred object boundaries. We integrate our depth completion model into dexterous grasping frameworks and achieve a $44\%$ improvement in the grasp success rate for transparent and reflective objects. We conduct extensive experiments on public datasets, where DCIRNet demonstrates superior performance. The experimental results validate the effectiveness of our approach and confirm its strong generalization capability across various transparent and reflective objects.
>
---
#### [new 032] Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于人机操作任务，旨在解决人手到机械手运动迁移中的关键目标分析问题。通过实验比较和优化目标设计，提升操作精度与效果。**

- **链接: [http://arxiv.org/pdf/2506.09384v1](http://arxiv.org/pdf/2506.09384v1)**

> **作者:** Chendong Xin; Mingrui Yu; Yongpeng Jiang; Zhefeng Zhang; Xiang Li
>
> **摘要:** Kinematic retargeting from human hands to robot hands is essential for transferring dexterity from humans to robots in manipulation teleoperation and imitation learning. However, due to mechanical differences between human and robot hands, completely reproducing human motions on robot hands is impossible. Existing works on retargeting incorporate various optimization objectives, focusing on different aspects of hand configuration. However, the lack of experimental comparative studies leaves the significance and effectiveness of these objectives unclear. This work aims to analyze these retargeting objectives for dexterous manipulation through extensive real-world comparative experiments. Specifically, we propose a comprehensive retargeting objective formulation that integrates intuitively crucial factors appearing in recent approaches. The significance of each factor is evaluated through experimental ablation studies on the full objective in kinematic posture retargeting and real-world teleoperated manipulation tasks. Experimental results and conclusions provide valuable insights for designing more accurate and effective retargeting algorithms for real-world dexterous manipulation.
>
---
#### [new 033] eFlesh: Highly customizable Magnetic Touch Sensing using Cut-Cell Microstructures
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人触觉感知任务，旨在解决缺乏通用、可定制触觉传感器的问题。提出eFlesh磁力触觉传感器，低成本、易制造，适用于多种应用场景。**

- **链接: [http://arxiv.org/pdf/2506.09994v1](http://arxiv.org/pdf/2506.09994v1)**

> **作者:** Venkatesh Pattabiraman; Zizhou Huang; Daniele Panozzo; Denis Zorin; Lerrel Pinto; Raunaq Bhirangi
>
> **摘要:** If human experience is any guide, operating effectively in unstructured environments -- like homes and offices -- requires robots to sense the forces during physical interaction. Yet, the lack of a versatile, accessible, and easily customizable tactile sensor has led to fragmented, sensor-specific solutions in robotic manipulation -- and in many cases, to force-unaware, sensorless approaches. With eFlesh, we bridge this gap by introducing a magnetic tactile sensor that is low-cost, easy to fabricate, and highly customizable. Building an eFlesh sensor requires only four components: a hobbyist 3D printer, off-the-shelf magnets (<$5), a CAD model of the desired shape, and a magnetometer circuit board. The sensor is constructed from tiled, parameterized microstructures, which allow for tuning the sensor's geometry and its mechanical response. We provide an open-source design tool that converts convex OBJ/STL files into 3D-printable STLs for fabrication. This modular design framework enables users to create application-specific sensors, and to adjust sensitivity depending on the task. Our sensor characterization experiments demonstrate the capabilities of eFlesh: contact localization RMSE of 0.5 mm, and force prediction RMSE of 0.27 N for normal force and 0.12 N for shear force. We also present a learned slip detection model that generalizes to unseen objects with 95% accuracy, and visuotactile control policies that improve manipulation performance by 40% over vision-only baselines -- achieving 91% average success rate for four precise tasks that require sub-mm accuracy for successful completion. All design files, code and the CAD-to-eFlesh STL conversion tool are open-sourced and available on https://e-flesh.com.
>
---
#### [new 034] ReSim: Reliable World Simulation for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶仿真任务，解决真实数据不足导致的非专家行为模拟问题。通过融合模拟器数据构建可控世界模型，提升仿真可靠性和控制能力。**

- **链接: [http://arxiv.org/pdf/2506.09981v1](http://arxiv.org/pdf/2506.09981v1)**

> **作者:** Jiazhi Yang; Kashyap Chitta; Shenyuan Gao; Long Chen; Yuqian Shao; Xiaosong Jia; Hongyang Li; Andreas Geiger; Xiangyu Yue; Li Chen
>
> **备注:** Project page: https://opendrivelab.com/ReSim
>
> **摘要:** How can we reliably simulate future driving scenarios under a wide range of ego driving behaviors? Recent driving world models, developed exclusively on real-world driving data composed mainly of safe expert trajectories, struggle to follow hazardous or non-expert behaviors, which are rare in such data. This limitation restricts their applicability to tasks such as policy evaluation. In this work, we address this challenge by enriching real-world human demonstrations with diverse non-expert data collected from a driving simulator (e.g., CARLA), and building a controllable world model trained on this heterogeneous corpus. Starting with a video generator featuring a diffusion transformer architecture, we devise several strategies to effectively integrate conditioning signals and improve prediction controllability and fidelity. The resulting model, ReSim, enables Reliable Simulation of diverse open-world driving scenarios under various actions, including hazardous non-expert ones. To close the gap between high-fidelity simulation and applications that require reward signals to judge different actions, we introduce a Video2Reward module that estimates a reward from ReSim's simulated future. Our ReSim paradigm achieves up to 44% higher visual fidelity, improves controllability for both expert and non-expert actions by over 50%, and boosts planning and policy selection performance on NAVSIM by 2% and 25%, respectively.
>
---
#### [new 035] CheckManual: A New Challenge and Benchmark for Manual-based Appliance Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决基于说明书的家电操作问题。通过构建基准数据集CheckManual和提出ManualPlan模型，提升机器人理解与执行手册指导的能力。**

- **链接: [http://arxiv.org/pdf/2506.09343v1](http://arxiv.org/pdf/2506.09343v1)**

> **作者:** Yuxing Long; Jiyao Zhang; Mingjie Pan; Tianshu Wu; Taewhan Kim; Hao Dong
>
> **备注:** CVPR 2025 Highlight
>
> **摘要:** Correct use of electrical appliances has significantly improved human life quality. Unlike simple tools that can be manipulated with common sense, different parts of electrical appliances have specific functions defined by manufacturers. If we want the robot to heat bread by microwave, we should enable them to review the microwave manual first. From the manual, it can learn about component functions, interaction methods, and representative task steps about appliances. However, previous manual-related works remain limited to question-answering tasks while existing manipulation researchers ignore the manual's important role and fail to comprehend multi-page manuals. In this paper, we propose the first manual-based appliance manipulation benchmark CheckManual. Specifically, we design a large model-assisted human-revised data generation pipeline to create manuals based on CAD appliance models. With these manuals, we establish novel manual-based manipulation challenges, metrics, and simulator environments for model performance evaluation. Furthermore, we propose the first manual-based manipulation planning model ManualPlan to set up a group of baselines for the CheckManual benchmark.
>
---
#### [new 036] Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于交互式模仿学习任务，旨在降低人类监督的认知负担。提出自适应干预机制（AIM），通过代理Q函数动态请求人类示范，提升学习效率与安全性。**

- **链接: [http://arxiv.org/pdf/2506.09176v1](http://arxiv.org/pdf/2506.09176v1)**

> **作者:** Haoyuan Cai; Zhenghao Peng; Bolei Zhou
>
> **备注:** ICML 2025 Poster
>
> **摘要:** Interactive Imitation Learning (IIL) allows agents to acquire desired behaviors through human interventions, but current methods impose high cognitive demands on human supervisors. We propose the Adaptive Intervention Mechanism (AIM), a novel robot-gated IIL algorithm that learns an adaptive criterion for requesting human demonstrations. AIM utilizes a proxy Q-function to mimic the human intervention rule and adjusts intervention requests based on the alignment between agent and human actions. By assigning high Q-values when the agent deviates from the expert and decreasing these values as the agent becomes proficient, the proxy Q-function enables the agent to assess the real-time alignment with the expert and request assistance when needed. Our expert-in-the-loop experiments reveal that AIM significantly reduces expert monitoring efforts in both continuous and discrete control tasks. Compared to the uncertainty-based baseline Thrifty-DAgger, our method achieves a 40% improvement in terms of human take-over cost and learning efficiency. Furthermore, AIM effectively identifies safety-critical states for expert assistance, thereby collecting higher-quality expert demonstrations and reducing overall expert data and environment interactions needed. Code and demo video are available at https://github.com/metadriverse/AIM.
>
---
#### [new 037] Adaptive event-triggered robust tracking control of soft robots
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于软体机器人控制任务，旨在解决不确定环境下跟踪控制问题。通过设计事件触发策略和自适应算法，提升控制精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.09523v1](http://arxiv.org/pdf/2506.09523v1)**

> **作者:** Renjie Ma; Ziyao Qu; Zhijian Hu; Dong Zhao; Marios M. Polycarpou
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Soft robots manufactured with flexible materials can be highly compliant and adaptive to their surroundings, which facilitates their application in areas such as dexterous manipulation and environmental exploration. This paper aims at investigating the tracking control problem for soft robots under uncertainty such as unmodeled dynamics and external disturbance. First, we establish a novel switching function and design the compensated tracking error dynamics by virtue of the command filter. Then, based on the backstepping methodology, the virtual controllers and the adaptive logic estimating the supremum of uncertainty impacts are developed for synthesizing an event-triggered control strategy. In addition, the uniformed finite-time stability certification is derived for different scenarios of the switching function. Finally, we perform a case study of a soft robot to illustrate the effectiveness of the proposed control algorithm.
>
---
#### [new 038] BG-HOP: A Bimanual Generative Hand-Object Prior
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于3D手-物交互任务，旨在解决数据不足的问题，通过扩展单手生成先验，构建双臂交互模型并生成抓取。**

- **链接: [http://arxiv.org/pdf/2506.09068v1](http://arxiv.org/pdf/2506.09068v1)**

> **作者:** Sriram Krishna; Sravan Chittupalli; Sungjae Park
>
> **备注:** Presented at Agents in Interaction, from Humans to Robots, CVPR 2025
>
> **摘要:** In this work, we present BG-HOP, a generative prior that seeks to model bimanual hand-object interactions in 3D. We address the challenge of limited bimanual interaction data by extending existing single-hand generative priors, demonstrating preliminary results in capturing the joint distribution of hands and objects. Our experiments showcase the model's capability to generate bimanual interactions and synthesize grasps for given objects. We make code and models publicly available.
>
---
#### [new 039] V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视频理解与机器人规划任务，旨在通过自监督学习结合网络视频和少量机器人数据，提升模型对物理世界的理解、预测与规划能力。**

- **链接: [http://arxiv.org/pdf/2506.09985v1](http://arxiv.org/pdf/2506.09985v1)**

> **作者:** Mido Assran; Adrien Bardes; David Fan; Quentin Garrido; Russell Howes; Mojtaba; Komeili; Matthew Muckley; Ammar Rizvi; Claire Roberts; Koustuv Sinha; Artem Zholus; Sergio Arnaud; Abha Gejji; Ada Martin; Francois Robert Hogan; Daniel Dugas; Piotr Bojanowski; Vasil Khalidov; Patrick Labatut; Francisco Massa; Marc Szafraniec; Kapil Krishnakumar; Yong Li; Xiaodong Ma; Sarath Chandar; Franziska Meier; Yann LeCun; Michael Rabbat; Nicolas Ballas
>
> **备注:** 48 pages, 19 figures
>
> **摘要:** A major challenge for modern AI is to learn to understand the world and learn to act largely by observation. This paper explores a self-supervised approach that combines internet-scale video data with a small amount of interaction data (robot trajectories), to develop models capable of understanding, predicting, and planning in the physical world. We first pre-train an action-free joint-embedding-predictive architecture, V-JEPA 2, on a video and image dataset comprising over 1 million hours of internet video. V-JEPA 2 achieves strong performance on motion understanding (77.3 top-1 accuracy on Something-Something v2) and state-of-the-art performance on human action anticipation (39.7 recall-at-5 on Epic-Kitchens-100) surpassing previous task-specific models. Additionally, after aligning V-JEPA 2 with a large language model, we demonstrate state-of-the-art performance on multiple video question-answering tasks at the 8 billion parameter scale (e.g., 84.0 on PerceptionTest, 76.9 on TempCompass). Finally, we show how self-supervised learning can be applied to robotic planning tasks by post-training a latent action-conditioned world model, V-JEPA 2-AC, using less than 62 hours of unlabeled robot videos from the Droid dataset. We deploy V-JEPA 2-AC zero-shot on Franka arms in two different labs and enable picking and placing of objects using planning with image goals. Notably, this is achieved without collecting any data from the robots in these environments, and without any task-specific training or reward. This work demonstrates how self-supervised learning from web-scale data and a small amount of robot interaction data can yield a world model capable of planning in the physical world.
>
---
#### [new 040] Efficient Preference-Based Reinforcement Learning: Randomized Exploration Meets Experimental Design
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文属于基于人类反馈的强化学习任务，解决如何高效选择偏好查询以学习奖励函数的问题。提出一种随机探索算法，提升查询效率并保证理论性能。**

- **链接: [http://arxiv.org/pdf/2506.09508v1](http://arxiv.org/pdf/2506.09508v1)**

> **作者:** Andreas Schlaginhaufen; Reda Ouhamma; Maryam Kamgarpour
>
> **摘要:** We study reinforcement learning from human feedback in general Markov decision processes, where agents learn from trajectory-level preference comparisons. A central challenge in this setting is to design algorithms that select informative preference queries to identify the underlying reward while ensuring theoretical guarantees. We propose a meta-algorithm based on randomized exploration, which avoids the computational challenges associated with optimistic approaches and remains tractable. We establish both regret and last-iterate guarantees under mild reinforcement learning oracle assumptions. To improve query complexity, we introduce and analyze an improved algorithm that collects batches of trajectory pairs and applies optimal experimental design to select informative comparison queries. The batch structure also enables parallelization of preference queries, which is relevant in practical deployment as feedback can be gathered concurrently. Empirical evaluation confirms that the proposed method is competitive with reward-based reinforcement learning while requiring a small number of preference queries.
>
---
#### [new 041] How attention simplifies mental representations for planning
- **分类: q-bio.NC; cs.AI; cs.RO**

- **简介: 该论文研究人类规划中注意力如何简化心理表征，属于认知科学任务。它探讨注意力如何影响环境表征，并通过虚拟迷宫实验验证空间接近性与任务相关性的作用。**

- **链接: [http://arxiv.org/pdf/2506.09520v1](http://arxiv.org/pdf/2506.09520v1)**

> **作者:** Jason da Silva Castanheira; Nicholas Shea; Stephen M. Fleming
>
> **摘要:** Human planning is efficient -- it frugally deploys limited cognitive resources to accomplish difficult tasks -- and flexible -- adapting to novel problems and environments. Computational approaches suggest that people construct simplified mental representations of their environment, balancing the complexity of a task representation with its utility. These models imply a nested optimisation in which planning shapes perception, and perception shapes planning -- but the perceptual and attentional mechanisms governing how this interaction unfolds remain unknown. Here, we harness virtual maze navigation to characterise how spatial attention controls which aspects of a task representation enter subjective awareness and are available for planning. We find that spatial proximity governs which aspects of a maze are available for planning, and that when task-relevant information follows natural (lateralised) contours of attention, people can more easily construct simplified and useful maze representations. This influence of attention varies considerably across individuals, explaining differences in people's task representations and behaviour. Inspired by the 'spotlight of attention' analogy, we incorporate the effects of visuospatial attention into existing computational accounts of value-guided construal. Together, our work bridges computational perspectives on perception and decision-making to better understand how individuals represent their environments in aid of planning.
>
---
#### [new 042] Hierarchical Image Matching for UAV Absolute Visual Localization via Semantic and Structural Constraints
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机绝对视觉定位任务，解决GNSS信号缺失下的定位问题。通过引入分层图像匹配方法，结合语义与结构约束，提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.09748v1](http://arxiv.org/pdf/2506.09748v1)**

> **作者:** Xiangkai Zhang; Xiang Zhou; Mao Chen; Yuchen Lu; Xu Yang; Zhiyong Liu
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Absolute localization, aiming to determine an agent's location with respect to a global reference, is crucial for unmanned aerial vehicles (UAVs) in various applications, but it becomes challenging when global navigation satellite system (GNSS) signals are unavailable. Vision-based absolute localization methods, which locate the current view of the UAV in a reference satellite map to estimate its position, have become popular in GNSS-denied scenarios. However, existing methods mostly rely on traditional and low-level image matching, suffering from difficulties due to significant differences introduced by cross-source discrepancies and temporal variations. To overcome these limitations, in this paper, we introduce a hierarchical cross-source image matching method designed for UAV absolute localization, which integrates a semantic-aware and structure-constrained coarse matching module with a lightweight fine-grained matching module. Specifically, in the coarse matching module, semantic features derived from a vision foundation model first establish region-level correspondences under semantic and structural constraints. Then, the fine-grained matching module is applied to extract fine features and establish pixel-level correspondences. Building upon this, a UAV absolute visual localization pipeline is constructed without any reliance on relative localization techniques, mainly by employing an image retrieval module before the proposed hierarchical image matching modules. Experimental evaluations on public benchmark datasets and a newly introduced CS-UAV dataset demonstrate superior accuracy and robustness of the proposed method under various challenging conditions, confirming its effectiveness.
>
---
#### [new 043] HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios
- **分类: cs.CV; cs.LG; cs.MM; cs.RO; eess.IV**

- **简介: 该论文属于多人群体中基于文本引用的人体动作分割任务，解决多人群体动作识别中目标人物定位与分割问题，提出HopaDIFF框架提升分割精度。**

- **链接: [http://arxiv.org/pdf/2506.09650v1](http://arxiv.org/pdf/2506.09650v1)**

> **作者:** Kunyu Peng; Junchao Huang; Xiangsheng Huang; Di Wen; Junwei Zheng; Yufan Chen; Kailun Yang; Jiamin Wu; Chongqing Hao; Rainer Stiefelhagen
>
> **备注:** The code is available at https://github.com/KPeng9510/HopaDIFF.git
>
> **摘要:** Action segmentation is a core challenge in high-level video understanding, aiming to partition untrimmed videos into segments and assign each a label from a predefined action set. Existing methods primarily address single-person activities with fixed action sequences, overlooking multi-person scenarios. In this work, we pioneer textual reference-guided human action segmentation in multi-person settings, where a textual description specifies the target person for segmentation. We introduce the first dataset for Referring Human Action Segmentation, i.e., RHAS133, built from 133 movies and annotated with 137 fine-grained actions with 33h video data, together with textual descriptions for this new task. Benchmarking existing action recognition methods on RHAS133 using VLM-based feature extractors reveals limited performance and poor aggregation of visual cues for the target person. To address this, we propose a holistic-partial aware Fourier-conditioned diffusion framework, i.e., HopaDIFF, leveraging a novel cross-input gate attentional xLSTM to enhance holistic-partial long-range reasoning and a novel Fourier condition to introduce more fine-grained control to improve the action segmentation generation. HopaDIFF achieves state-of-the-art results on RHAS133 in diverse evaluation settings. The code is available at https://github.com/KPeng9510/HopaDIFF.git.
>
---
#### [new 044] OctoNav: Towards Generalist Embodied Navigation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于具身导航任务，旨在解决多模态、多能力导航的通用性问题。提出OctoNav-Bench和OctoNav-R1，通过混合训练提升模型推理能力。**

- **链接: [http://arxiv.org/pdf/2506.09839v1](http://arxiv.org/pdf/2506.09839v1)**

> **作者:** Chen Gao; Liankai Jin; Xingyu Peng; Jiazhao Zhang; Yue Deng; Annan Li; He Wang; Si Liu
>
> **备注:** 31 pages, 25 figures
>
> **摘要:** Embodied navigation stands as a foundation pillar within the broader pursuit of embodied AI. However, previous navigation research is divided into different tasks/capabilities, e.g., ObjNav, ImgNav and VLN, where they differ in task objectives and modalities, making datasets and methods are designed individually. In this work, we take steps toward generalist navigation agents, which can follow free-form instructions that include arbitrary compounds of multi-modal and multi-capability. To achieve this, we propose a large-scale benchmark and corresponding method, termed OctoNav-Bench and OctoNav-R1. Specifically, OctoNav-Bench features continuous environments and is constructed via a designed annotation pipeline. We thoroughly craft instruction-trajectory pairs, where instructions are diverse in free-form with arbitrary modality and capability. Also, we construct a Think-Before-Action (TBA-CoT) dataset within OctoNav-Bench to provide the thinking process behind actions. For OctoNav-R1, we build it upon MLLMs and adapt it to a VLA-type model, which can produce low-level actions solely based on 2D visual observations. Moreover, we design a Hybrid Training Paradigm (HTP) that consists of three stages, i.e., Action-/TBA-SFT, Nav-GPRO, and Online RL stages. Each stage contains specifically designed learning policies and rewards. Importantly, for TBA-SFT and Nav-GRPO designs, we are inspired by the OpenAI-o1 and DeepSeek-R1, which show impressive reasoning ability via thinking-before-answer. Thus, we aim to investigate how to achieve thinking-before-action in the embodied navigation field, to improve model's reasoning ability toward generalists. Specifically, we propose TBA-SFT to utilize the TBA-CoT dataset to fine-tune the model as a cold-start phrase and then leverage Nav-GPRO to improve its thinking ability. Finally, OctoNav-R1 shows superior performance compared with previous methods.
>
---
#### [new 045] UFM: A Simple Path towards Unified Dense Correspondence with Flow
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于图像对应任务，解决宽基线和光流估计的统一问题。提出UFM模型，通过统一训练提升准确性和效率。**

- **链接: [http://arxiv.org/pdf/2506.09278v1](http://arxiv.org/pdf/2506.09278v1)**

> **作者:** Yuchen Zhang; Nikhil Keetha; Chenwei Lyu; Bhuvan Jhamb; Yutian Chen; Yuheng Qiu; Jay Karhade; Shreyas Jha; Yaoyu Hu; Deva Ramanan; Sebastian Scherer; Wenshan Wang
>
> **备注:** Project Page: https://uniflowmatch.github.io/
>
> **摘要:** Dense image correspondence is central to many applications, such as visual odometry, 3D reconstruction, object association, and re-identification. Historically, dense correspondence has been tackled separately for wide-baseline scenarios and optical flow estimation, despite the common goal of matching content between two images. In this paper, we develop a Unified Flow & Matching model (UFM), which is trained on unified data for pixels that are co-visible in both source and target images. UFM uses a simple, generic transformer architecture that directly regresses the (u,v) flow. It is easier to train and more accurate for large flows compared to the typical coarse-to-fine cost volumes in prior work. UFM is 28% more accurate than state-of-the-art flow methods (Unimatch), while also having 62% less error and 6.7x faster than dense wide-baseline matchers (RoMa). UFM is the first to demonstrate that unified training can outperform specialized approaches across both domains. This result enables fast, general-purpose correspondence and opens new directions for multi-modal, long-range, and real-time correspondence tasks.
>
---
## 更新

#### [replaced 001] Teaching Physical Awareness to LLMs through Sounds
- **分类: cs.SD; cs.AI; cs.MM; cs.RO; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08524v2](http://arxiv.org/pdf/2506.08524v2)**

> **作者:** Weiguo Wang; Andy Nie; Wenrui Zhou; Yi Kai; Chengchen Hu
>
> **备注:** ICML 2025
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities in text and multimodal processing, yet they fundamentally lack physical awareness--understanding of real-world physical phenomena. In this work, we present ACORN, a framework that teaches LLMs physical awareness through sound, focusing on fundamental physical phenomena like the Doppler effect, multipath effect, and spatial relationships. To overcome data scarcity, ACORN introduce a physics-based simulator combining real-world sound sources with controlled physical channels to generate diverse training data. Using this simulator, we build AQA-PHY, a comprehensive Audio Question-Answer dataset, and propose an audio encoder that processes both magnitude and phase information. By connecting our audio encoder to state-of-the-art LLMs, we demonstrate reasonable results in both simulated and real-world tasks, such as line-of-sight detection, Doppler effect estimation, and Direction-of-Arrival estimation, paving the way for enabling LLMs to understand physical world.
>
---
#### [replaced 002] Sim-to-Real Causal Transfer: A Metric Learning Approach to Causally-Aware Interaction Representations
- **分类: cs.LG; cs.AI; cs.CV; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2312.04540v2](http://arxiv.org/pdf/2312.04540v2)**

> **作者:** Ahmad Rahimi; Po-Chien Luan; Yuejiang Liu; Frano Rajič; Alexandre Alahi
>
> **备注:** CVPR 2025
>
> **摘要:** Modeling spatial-temporal interactions among neighboring agents is at the heart of multi-agent problems such as motion forecasting and crowd navigation. Despite notable progress, it remains unclear to which extent modern representations can capture the causal relationships behind agent interactions. In this work, we take an in-depth look at the causal awareness of these representations, from computational formalism to real-world practice. First, we cast doubt on the notion of non-causal robustness studied in the recent CausalAgents benchmark. We show that recent representations are already partially resilient to perturbations of non-causal agents, and yet modeling indirect causal effects involving mediator agents remains challenging. To address this challenge, we introduce a metric learning approach that regularizes latent representations with causal annotations. Our controlled experiments show that this approach not only leads to higher degrees of causal awareness but also yields stronger out-of-distribution robustness. To further operationalize it in practice, we propose a sim-to-real causal transfer method via cross-domain multi-task learning. Experiments on pedestrian datasets show that our method can substantially boost generalization, even in the absence of real-world causal annotations. We hope our work provides a new perspective on the challenges and pathways towards causally-aware representations of multi-agent interactions. Our code is available at https://github.com/vita-epfl/CausalSim2Real.
>
---
#### [replaced 003] ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22094v3](http://arxiv.org/pdf/2505.22094v3)**

> **作者:** Tonghe Zhang; Chao Yu; Sichang Su; Yu Wang
>
> **备注:** 30 pages, 13 figures, 10 tables
>
> **摘要:** We propose ReinFlow, a simple yet effective online reinforcement learning (RL) framework that fine-tunes a family of flow matching policies for continuous robotic control. Derived from rigorous RL theory, ReinFlow injects learnable noise into a flow policy's deterministic path, converting the flow into a discrete-time Markov Process for exact and straightforward likelihood computation. This conversion facilitates exploration and ensures training stability, enabling ReinFlow to fine-tune diverse flow model variants, including Rectified Flow [35] and Shortcut Models [19], particularly at very few or even one denoising step. We benchmark ReinFlow in representative locomotion and manipulation tasks, including long-horizon planning with visual input and sparse reward. The episode reward of Rectified Flow policies obtained an average net growth of 135.36% after fine-tuning in challenging legged locomotion tasks while saving denoising steps and 82.63% of wall time compared to state-of-the-art diffusion RL fine-tuning method DPPO [43]. The success rate of the Shortcut Model policies in state and visual manipulation tasks achieved an average net increase of 40.34% after fine-tuning with ReinFlow at four or even one denoising step, whose performance is comparable to fine-tuned DDIM policies while saving computation time for an average of 23.20%. Project webpage: https://reinflow.github.io/
>
---
#### [replaced 004] Trailblazer: Learning offroad costmaps for long range planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.09739v2](http://arxiv.org/pdf/2505.09739v2)**

> **作者:** Kasi Viswanath; Felix Sanchez; Timothy Overbye; Jason M. Gregory; Srikanth Saripalli
>
> **摘要:** Autonomous navigation in off-road environments remains a significant challenge in field robotics, particularly for Unmanned Ground Vehicles (UGVs) tasked with search and rescue, exploration, and surveillance. Effective long-range planning relies on the integration of onboard perception systems with prior environmental knowledge, such as satellite imagery and LiDAR data. This work introduces Trailblazer, a novel framework that automates the conversion of multi-modal sensor data into costmaps, enabling efficient path planning without manual tuning. Unlike traditional approaches, Trailblazer leverages imitation learning and a differentiable A* planner to learn costmaps directly from expert demonstrations, enhancing adaptability across diverse terrains. The proposed methodology was validated through extensive real-world testing, achieving robust performance in dynamic and complex environments, demonstrating Trailblazer's potential for scalable, efficient autonomous navigation.
>
---
#### [replaced 005] Generalizable and Fast Surrogates: Model Predictive Control of Articulated Soft Robots using Physics-Informed Neural Networks
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01916v2](http://arxiv.org/pdf/2502.01916v2)**

> **作者:** Tim-Lukas Habich; Aran Mohammad; Simon F. G. Ehlers; Martin Bensch; Thomas Seel; Moritz Schappler
>
> **摘要:** Soft robots can revolutionize several applications with high demands on dexterity and safety. When operating these systems, real-time estimation and control require fast and accurate models. However, prediction with first-principles (FP) models is slow, and learned black-box models have poor generalizability. Physics-informed machine learning offers excellent advantages here, but it is currently limited to simple, often simulated systems without considering changes after training. We propose physics-informed neural networks (PINNs) for articulated soft robots (ASRs) with a focus on data efficiency. The amount of expensive real-world training data is reduced to a minimum -- one dataset in one system domain. Two hours of data in different domains are used for a comparison against two gold-standard approaches: In contrast to a recurrent neural network, the PINN provides a high generalizability. The prediction speed of an accurate FP model is exceeded with the PINN by up to a factor of 467 at slightly reduced accuracy. This enables nonlinear model predictive control (MPC) of a pneumatic ASR. Accurate position tracking with the MPC running at 47 Hz is achieved in six dynamic experiments.
>
---
#### [replaced 006] Mixed Reality Tele-Ultrasound over 750 km: A Feasibility Study
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.13058v2](http://arxiv.org/pdf/2409.13058v2)**

> **作者:** Ryan Yeung; David Black; Patrick B. Chen; Victoria Lessoway; Janice Reid; Sergio Rangel-Suarez; Silvia D. Chang; Septimiu E. Salcudean
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** To address the lack of access to ultrasound in remote communities, previous work introduced human teleoperation, a mixed reality and haptics-based tele-ultrasound system. In this approach, a novice takes the role of a cognitive robot controlled remotely by an expert through mixed reality. In this manuscript we summarize new developments to this system and describe a feasibility study assessing its use for long-distance remote abdominal ultrasound examinations. To provide simple but effective haptic feedback, we used an ellipsoid model of the patient with its parameters calibrated using our system's position and force sensors. We tested the system in Skidegate, Haida Gwaii, Canada, with the experts positioned 754 km away in Vancouver, Canada. We performed 11 total scans with 10 novices and 2 sonographers. The sonographers were tasked with acquiring 5 target images in the epigastric region. The image acquisition quality was assessed by 2 radiologists. We collected alignment data and the novices completed task load and usability questionnaires. Both the novices and sonographers provided written and verbal feedback to inform future design iterations. 92% of the acquired images had sufficient quality for interpretation by both radiologists. The mean task load reported by the novices was below reference values reported in literature and the usability was unanimously positive. No correlation was found between image quality and the follower's alignment error with the virtual transducer. Overall, we show that human teleoperation enables sonographers to perform remote abdominal ultrasound imaging with high performance, even across large distances and with novice followers. Future work will compare human teleoperation to conventional, robotic and tele-mentored ultrasound.
>
---
#### [replaced 007] PatchPilot: A Cost-Efficient Software Engineering Agent with Early Attempts on Formal Verification
- **分类: cs.RO; cs.AI; cs.CR**

- **链接: [http://arxiv.org/pdf/2502.02747v2](http://arxiv.org/pdf/2502.02747v2)**

> **作者:** Hongwei Li; Yuheng Tang; Shiqi Wang; Wenbo Guo
>
> **摘要:** Recent research builds various patching agents that combine large language models (LLMs) with non-ML tools and achieve promising results on the state-of-the-art (SOTA) software patching benchmark, SWE-bench. Based on how to determine the patching workflows, existing patching agents can be categorized as agent-based planning methods, which rely on LLMs for planning, and rule-based planning methods, which follow a pre-defined workflow. At a high level, agent-based planning methods achieve high patching performance but with a high cost and limited stability. Rule-based planning methods, on the other hand, are more stable and efficient but have key workflow limitations that compromise their patching performance. In this paper, we propose PatchPilot, an agentic patcher that strikes a balance between patching efficacy, stability, and cost-efficiency. PatchPilot proposes a novel rule-based planning workflow with five components: reproduction, localization, generation, validation, and refinement (where refinement is unique to PatchPilot). We introduce novel and customized designs to each component to optimize their effectiveness and efficiency. Through extensive experiments on the SWE-bench benchmarks, PatchPilot shows a superior performance than existing open-source methods while maintaining low cost (less than 1$ per instance) and ensuring higher stability. We also conduct a detailed ablation study to validate the key designs in each component. Our code is available at https://github.com/ucsb-mlsec/PatchPilot.
>
---
#### [replaced 008] Zero-Shot Temporal Interaction Localization for Egocentric Videos
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.03662v2](http://arxiv.org/pdf/2506.03662v2)**

> **作者:** Erhang Zhang; Junyi Ma; Yin-Dong Zheng; Yixuan Zhou; Hesheng Wang
>
> **摘要:** Locating human-object interaction (HOI) actions within video serves as the foundation for multiple downstream tasks, such as human behavior analysis and human-robot skill transfer. Current temporal action localization methods typically rely on annotated action and object categories of interactions for optimization, which leads to domain bias and low deployment efficiency. Although some recent works have achieved zero-shot temporal action localization (ZS-TAL) with large vision-language models (VLMs), their coarse-grained estimations and open-loop pipelines hinder further performance improvements for temporal interaction localization (TIL). To address these issues, we propose a novel zero-shot TIL approach dubbed EgoLoc to locate the timings of grasp actions for human-object interaction in egocentric videos. EgoLoc introduces a self-adaptive sampling strategy to generate reasonable visual prompts for VLM reasoning. By absorbing both 2D and 3D observations, it directly samples high-quality initial guesses around the possible contact/separation timestamps of HOI according to 3D hand velocities, leading to high inference accuracy and efficiency. In addition, EgoLoc generates closed-loop feedback from visual and dynamic cues to further refine the localization results. Comprehensive experiments on the publicly available dataset and our newly proposed benchmark demonstrate that EgoLoc achieves better temporal interaction localization for egocentric videos compared to state-of-the-art baselines. We will release our code and relevant data as open-source at https://github.com/IRMVLab/EgoLoc.
>
---
#### [replaced 009] Occlusion-Aware Ground Target Tracking by a Dubins Vehicle using Visibility Volumes
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2506.03400v2](http://arxiv.org/pdf/2506.03400v2)**

> **作者:** Collin Hague; Artur Wolek
>
> **备注:** 28 pages, 14 figures, 1 table
>
> **摘要:** This paper considers the problem of tracking a point of interest (POI) moving along a known trajectory on the ground with an uncrewed aerial vehicle (UAV) modeled as a Dubins vehicle using a line-of-sight (LOS) sensor through an urban environment that may occlude the POI. A visibility volume (VV) encodes a time-varying, three-dimensional representation of the sensing constraints for a particular POI position. A constant-altitude, translating, and radially time-varying circular standoff orbit is then inscribed within the dynamically changing VV centered at the POI position. The time-varying VV is approximated by placing static VVs along the POI's trajectory using an adaptive metric that restricts the volume change of consecutive VVs to below a specified rate. The time-varying circular standoff orbit is proven to be feasible for a Dubins vehicle and approximated with a piecewise set of linearly interpolated circular orbits inside the static VVs. A steering controller is derived that drives the UAV to the time-varying standoff orbit. Numerical simulations and a flight test illustrate the proposed approach.
>
---
#### [replaced 010] Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.23450v2](http://arxiv.org/pdf/2505.23450v2)**

> **作者:** Zhejian Yang; Yongchao Chen; Xueyang Zhou; Jiangyue Yan; Dingjie Song; Yinuo Liu; Yuting Li; Yu Zhang; Pan Zhou; Hechang Chen; Lichao Sun
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Long-horizon robotic manipulation poses significant challenges for autonomous systems, requiring extended reasoning, precise execution, and robust error recovery across complex sequential tasks. Current approaches, whether based on static planning or end-to-end visuomotor policies, suffer from error accumulation and lack effective verification mechanisms during execution, limiting their reliability in real-world scenarios. We present Agentic Robot, a brain-inspired framework that addresses these limitations through Standardized Action Procedure (SAP)--a novel coordination protocol governing component interactions throughout manipulation tasks. Drawing inspiration from Standardized Operating Procedures (SOPs) in human organizations, SAP establishes structured workflows for planning, execution, and verification phases. Our architecture comprises three specialized components: (1) a large reasoning model that decomposes high-level instructions into semantically coherent subgoals, (2) a vision-language-action executor that generates continuous control commands from real-time visual inputs, and (3) a temporal verifier that enables autonomous progression and error recovery through introspective assessment. This SAP-driven closed-loop design supports dynamic self-verification without external supervision. On the LIBERO benchmark, Agentic Robot achieves state-of-the-art performance with an average success rate of 79.6%, outperforming SpatialVLA by 6.1% and OpenVLA by 7.4% on long-horizon tasks. These results demonstrate that SAP-driven coordination between specialized components enhances both performance and interpretability in sequential manipulation, suggesting significant potential for reliable autonomous systems. Project Github: https://agentic-robot.github.io.
>
---
#### [replaced 011] Mem2Ego: Empowering Vision-Language Models with Global-to-Ego Memory for Long-Horizon Embodied Navigation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14254v2](http://arxiv.org/pdf/2502.14254v2)**

> **作者:** Lingfeng Zhang; Yuecheng Liu; Zhanguang Zhang; Matin Aghaei; Yaochen Hu; Hongjian Gu; Mohammad Ali Alomrani; David Gamaliel Arcos Bravo; Raika Karimi; Atia Hamidizadeh; Haoping Xu; Guowei Huang; Zhanpeng Zhang; Tongtong Cao; Weichao Qiu; Xingyue Quan; Jianye Hao; Yuzheng Zhuang; Yingxue Zhang
>
> **摘要:** Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have made them powerful tools in embodied navigation, enabling agents to leverage commonsense and spatial reasoning for efficient exploration in unfamiliar environments. Existing LLM-based approaches convert global memory, such as semantic or topological maps, into language descriptions to guide navigation. While this improves efficiency and reduces redundant exploration, the loss of geometric information in language-based representations hinders spatial reasoning, especially in intricate environments. To address this, VLM-based approaches directly process ego-centric visual inputs to select optimal directions for exploration. However, relying solely on a first-person perspective makes navigation a partially observed decision-making problem, leading to suboptimal decisions in complex environments. In this paper, we present a novel vision-language model (VLM)-based navigation framework that addresses these challenges by adaptively retrieving task-relevant cues from a global memory module and integrating them with the agent's egocentric observations. By dynamically aligning global contextual information with local perception, our approach enhances spatial reasoning and decision-making in long-horizon tasks. Experimental results demonstrate that the proposed method surpasses previous state-of-the-art approaches in object navigation tasks, providing a more effective and scalable solution for embodied navigation.
>
---
#### [replaced 012] V2I-Calib++: A Multi-terminal Spatial Calibration Approach in Urban Intersections for Collaborative Perception
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.11008v3](http://arxiv.org/pdf/2410.11008v3)**

> **作者:** Qianxin Qu; Xinyu Zhang; Yifan Cheng; Yijin Xiong; Chen Xia; Qian Peng; Ziqiang Song; Kang Liu; Xin Wu; Jun Li
>
> **摘要:** Urban intersections, dense with pedestrian and vehicular traffic and compounded by GPS signal obstructions from high-rise buildings, are among the most challenging areas in urban traffic systems. Traditional single-vehicle intelligence systems often perform poorly in such environments due to a lack of global traffic flow information and the ability to respond to unexpected events. Vehicle-to-Everything (V2X) technology, through real-time communication between vehicles (V2V) and vehicles to infrastructure (V2I), offers a robust solution. However, practical applications still face numerous challenges. Calibration among heterogeneous vehicle and infrastructure endpoints in multi-end LiDAR systems is crucial for ensuring the accuracy and consistency of perception system data. Most existing multi-end calibration methods rely on initial calibration values provided by positioning systems, but the instability of GPS signals due to high buildings in urban canyons poses severe challenges to these methods. To address this issue, this paper proposes a novel multi-end LiDAR system calibration method that does not require positioning priors to determine initial external parameters and meets real-time requirements. Our method introduces an innovative multi-end perception object association technique, utilizing a new Overall Distance metric (oDist) to measure the spatial association between perception objects, and effectively combines global consistency search algorithms with optimal transport theory. By this means, we can extract co-observed targets from object association results for further external parameter computation and optimization. Extensive comparative and ablation experiments conducted on the simulated dataset V2X-Sim and the real dataset DAIR-V2X confirm the effectiveness and efficiency of our method. The code for this method can be accessed at: https://github.com/MassimoQu/v2i-calib.
>
---
#### [replaced 013] STREAMS: An Assistive Multimodal AI Framework for Empowering Biosignal Based Robotic Controls
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.03486v2](http://arxiv.org/pdf/2410.03486v2)**

> **作者:** Ali Rabiee; Sima Ghafoori; Xiangyu Bai; Sarah Ostadabbas; Reza Abiri
>
> **摘要:** End-effector based assistive robots face persistent challenges in generating smooth and robust trajectories when controlled by human's noisy and unreliable biosignals such as muscle activities and brainwaves. The produced endpoint trajectories are often jerky and imprecise to perform complex tasks such as stable robotic grasping. We propose STREAMS (Self-Training Robotic End-to-end Adaptive Multimodal Shared autonomy) as a novel framework leveraged deep reinforcement learning to tackle this challenge in biosignal based robotic control systems. STREAMS blends environmental information and synthetic user input into a Deep Q Learning Network (DQN) pipeline for an interactive end-to-end and self-training mechanism to produce smooth trajectories for the control of end-effector based robots. The proposed framework achieved a high-performance record of 98% in simulation with dynamic target estimation and acquisition without any pre-existing datasets. As a zero-shot sim-to-real user study with five participants controlling a physical robotic arm with noisy head movements, STREAMS (as an assistive mode) demonstrated significant improvements in trajectory stabilization, user satisfaction, and task performance reported as a success rate of 83% compared to manual mode which was 44% without any task support. STREAMS seeks to improve biosignal based assistive robotic controls by offering an interactive, end-to-end solution that stabilizes end-effector trajectories, enhancing task performance and accuracy.
>
---
#### [replaced 014] TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.08440v2](http://arxiv.org/pdf/2506.08440v2)**

> **作者:** Zengjue Chen; Runliang Niu; He Kong; Qi Wang
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) model have demonstrated strong generalization capabilities across diverse scenes, tasks, and robotic platforms when pretrained at large-scale datasets. However, these models still require task-specific fine-tuning in novel environments, a process that relies almost exclusively on supervised fine-tuning (SFT) using static trajectory datasets. Such approaches neither allow robot to interact with environment nor do they leverage feedback from live execution. Also, their success is critically dependent on the size and quality of the collected trajectories. Reinforcement learning (RL) offers a promising alternative by enabling closed-loop interaction and aligning learned policies directly with task objectives. In this work, we draw inspiration from the ideas of GRPO and propose the Trajectory-wise Group Relative Policy Optimization (TGRPO) method. By fusing step-level and trajectory-level advantage signals, this method improves GRPO's group-level advantage estimation, thereby making the algorithm more suitable for online reinforcement learning training of VLA. Experimental results on ten manipulation tasks from the libero-object benchmark demonstrate that TGRPO consistently outperforms various baseline methods, capable of generating more robust and efficient policies across multiple tested scenarios. Our source codes are available at: https://github.com/hahans/TGRPO
>
---
#### [replaced 015] Toward Reliable AR-Guided Surgical Navigation: Interactive Deformation Modeling with Data-Driven Biomechanics and Prompts
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08048v2](http://arxiv.org/pdf/2506.08048v2)**

> **作者:** Zheng Han; Jun Zhou; Jialun Pei; Jing Qin; Yingfang Fan; Qi Dou
>
> **摘要:** In augmented reality (AR)-guided surgical navigation, preoperative organ models are superimposed onto the patient's intraoperative anatomy to visualize critical structures such as vessels and tumors. Accurate deformation modeling is essential to maintain the reliability of AR overlays by ensuring alignment between preoperative models and the dynamically changing anatomy. Although the finite element method (FEM) offers physically plausible modeling, its high computational cost limits intraoperative applicability. Moreover, existing algorithms often fail to handle large anatomical changes, such as those induced by pneumoperitoneum or ligament dissection, leading to inaccurate anatomical correspondences and compromised AR guidance. To address these challenges, we propose a data-driven biomechanics algorithm that preserves FEM-level accuracy while improving computational efficiency. In addition, we introduce a novel human-in-the-loop mechanism into the deformation modeling process. This enables surgeons to interactively provide prompts to correct anatomical misalignments, thereby incorporating clinical expertise and allowing the model to adapt dynamically to complex surgical scenarios. Experiments on a publicly available dataset demonstrate that our algorithm achieves a mean target registration error of 3.42 mm. Incorporating surgeon prompts through the interactive framework further reduces the error to 2.78 mm, surpassing state-of-the-art methods in volumetric accuracy. These results highlight the ability of our framework to deliver efficient and accurate deformation modeling while enhancing surgeon-algorithm collaboration, paving the way for safer and more reliable computer-assisted surgeries.
>
---
#### [replaced 016] FROG: A new people detection dataset for knee-high 2D range finders
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2306.08531v2](http://arxiv.org/pdf/2306.08531v2)**

> **作者:** Fernando Amodeo; Noé Pérez-Higueras; Luis Merino; Fernando Caballero
>
> **备注:** Code and data are publicly available at: https://github.com/robotics-upo/2DLaserPeopleBenchmark
>
> **摘要:** Mobile robots require knowledge of the environment, especially of humans located in its vicinity. While the most common approaches for detecting humans involve computer vision, an often overlooked hardware feature of robots for people detection are their 2D range finders. These were originally intended for obstacle avoidance and mapping/SLAM tasks. In most robots, they are conveniently located at a height approximately between the ankle and the knee, so they can be used for detecting people too, and with a larger field of view and depth resolution compared to cameras. In this paper, we present a new dataset for people detection using knee-high 2D range finders called FROG. This dataset has greater laser resolution, scanning frequency, and more complete annotation data compared to existing datasets such as DROW. Particularly, the FROG dataset contains annotations for 100% of its laser scans (unlike DROW which only annotates 5%), 17x more annotated scans, 100x more people annotations, and over twice the distance traveled by the robot. We propose a benchmark based on the FROG dataset, and analyze a collection of state-of-the-art people detectors based on 2D range finder data. We also propose and evaluate a new end-to-end deep learning approach for people detection. Our solution works with the raw sensor data directly (not needing hand-crafted input data features), thus avoiding CPU preprocessing and releasing the developer of understanding specific domain heuristics. Experimental results show how the proposed people detector attains results comparable to the state of the art, while an optimized implementation for ROS can operate at more than 500 Hz.
>
---
#### [replaced 017] RMP-YOLO: A Robust Motion Predictor for Partially Observable Scenarios even if You Only Look Once
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11696v2](http://arxiv.org/pdf/2409.11696v2)**

> **作者:** Jiawei Sun; Jiahui Li; Tingchen Liu; Chengran Yuan; Shuo Sun; Zefan Huang; Anthony Wong; Keng Peng Tee; Marcelo H. Ang Jr
>
> **摘要:** We introduce RMP-YOLO, a unified framework designed to provide robust motion predictions even with incomplete input data. Our key insight stems from the observation that complete and reliable historical trajectory data plays a pivotal role in ensuring accurate motion prediction. Therefore, we propose a new paradigm that prioritizes the reconstruction of intact historical trajectories before feeding them into the prediction modules. Our approach introduces a novel scene tokenization module to enhance the extraction and fusion of spatial and temporal features. Following this, our proposed recovery module reconstructs agents' incomplete historical trajectories by leveraging local map topology and interactions with nearby agents. The reconstructed, clean historical data is then integrated into the downstream prediction modules. Our framework is able to effectively handle missing data of varying lengths and remains robust against observation noise, while maintaining high prediction accuracy. Furthermore, our recovery module is compatible with existing prediction models, ensuring seamless integration. Extensive experiments validate the effectiveness of our approach, and deployment in real-world autonomous vehicles confirms its practical utility. In the 2024 Waymo Motion Prediction Competition, our method, RMP-YOLO, achieves state-of-the-art performance, securing third place.
>
---
#### [replaced 018] Benchmarking Population-Based Reinforcement Learning across Robotic Tasks with GPU-Accelerated Simulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.03336v4](http://arxiv.org/pdf/2404.03336v4)**

> **作者:** Asad Ali Shahid; Yashraj Narang; Vincenzo Petrone; Enrico Ferrentino; Ankur Handa; Dieter Fox; Marco Pavone; Loris Roveda
>
> **备注:** Accepted for publication at 2025 IEEE 21st International Conference on Automation Science and Engineering
>
> **摘要:** In recent years, deep reinforcement learning (RL) has shown its effectiveness in solving complex continuous control tasks. However, this comes at the cost of an enormous amount of experience required for training, exacerbated by the sensitivity of learning efficiency and the policy performance to hyperparameter selection, which often requires numerous trials of time-consuming experiments. This work leverages a Population-Based Reinforcement Learning (PBRL) approach and a GPU-accelerated physics simulator to enhance the exploration capabilities of RL by concurrently training multiple policies in parallel. The PBRL framework is benchmarked against three state-of-the-art RL algorithms -- PPO, SAC, and DDPG -- dynamically adjusting hyperparameters based on the performance of learning agents. The experiments are performed on four challenging tasks in Isaac Gym -- Anymal Terrain, Shadow Hand, Humanoid, Franka Nut Pick -- by analyzing the effect of population size and mutation mechanisms for hyperparameters. The results show that PBRL agents achieve superior performance, in terms of cumulative reward, compared to non-evolutionary baseline agents. Moreover, the trained agents are finally deployed in the real world for a Franka Nut Pick task. To our knowledge, this is the first sim-to-real attempt for deploying PBRL agents on real hardware. Code and videos of the learned policies are available on our project website (https://sites.google.com/view/pbrl).
>
---
#### [replaced 019] Versatile Loco-Manipulation through Flexible Interlimb Coordination
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.07876v2](http://arxiv.org/pdf/2506.07876v2)**

> **作者:** Xinghao Zhu; Yuxin Chen; Lingfeng Sun; Farzad Niroui; Simon Le Cleac'h; Jiuguang Wang; Kuan Fang
>
> **摘要:** The ability to flexibly leverage limbs for loco-manipulation is essential for enabling autonomous robots to operate in unstructured environments. Yet, prior work on loco-manipulation is often constrained to specific tasks or predetermined limb configurations. In this work, we present Reinforcement Learning for Interlimb Coordination (ReLIC), an approach that enables versatile loco-manipulation through flexible interlimb coordination. The key to our approach is an adaptive controller that seamlessly bridges the execution of manipulation motions and the generation of stable gaits based on task demands. Through the interplay between two controller modules, ReLIC dynamically assigns each limb for manipulation or locomotion and robustly coordinates them to achieve the task success. Using efficient reinforcement learning in simulation, ReLIC learns to perform stable gaits in accordance with the manipulation goals in the real world. To solve diverse and complex tasks, we further propose to interface the learned controller with different types of task specifications, including target trajectories, contact points, and natural language instructions. Evaluated on 12 real-world tasks that require diverse and complex coordination patterns, ReLIC demonstrates its versatility and robustness by achieving a success rate of 78.9% on average. Videos and code can be found at https://relic-locoman.rai-inst.com.
>
---
#### [replaced 020] SACA: A Scenario-Aware Collision Avoidance Framework for Autonomous Vehicles Integrating LLMs-Driven Reasoning
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.00115v2](http://arxiv.org/pdf/2504.00115v2)**

> **作者:** Shiyue Zhao; Junzhi Zhang; Neda Masoud; Heye Huang; Xiaohui Hou; Chengkun He
>
> **备注:** 11 pages,10 figures. This work has been submitted to the IEEE TVT for possible publication
>
> **摘要:** Reliable collision avoidance under extreme situations remains a critical challenge for autonomous vehicles. While large language models (LLMs) offer promising reasoning capabilities, their application in safety-critical evasive maneuvers is limited by latency and robustness issues. Even so, LLMs stand out for their ability to weigh emotional, legal, and ethical factors, enabling socially responsible and context-aware collision avoidance. This paper proposes a scenario-aware collision avoidance (SACA) framework for extreme situations by integrating predictive scenario evaluation, data-driven reasoning, and scenario-preview-based deployment to improve collision avoidance decision-making. SACA consists of three key components. First, a predictive scenario analysis module utilizes obstacle reachability analysis and motion intention prediction to construct a comprehensive situational prompt. Second, an online reasoning module refines decision-making by leveraging prior collision avoidance knowledge and fine-tuning with scenario data. Third, an offline evaluation module assesses performance and stores scenarios in a memory bank. Additionally, A precomputed policy method improves deployability by previewing scenarios and retrieving or reasoning policies based on similarity and confidence levels. Real-vehicle tests show that, compared with baseline methods, SACA effectively reduces collision losses in extreme high-risk scenarios and lowers false triggering under complex conditions. Project page: https://sean-shiyuez.github.io/SACA/.
>
---
#### [replaced 021] Design and Validation of an Intention-Aware Probabilistic Framework for Trajectory Prediction: Integrating COLREGS, Grounding Hazards, and Planned Routes
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.00731v2](http://arxiv.org/pdf/2504.00731v2)**

> **作者:** Dhanika Mahipala; Trym Tengesdal; Børge Rokseth; Tor Arne Johansen
>
> **备注:** IMPORTANT: This preprint is not the final version. The peer-reviewed and updated version is published in Ocean Engineering journal [https://doi.org/10.1016/j.oceaneng.2025.121564]
>
> **摘要:** Collision avoidance capability is an essential component in an autonomous vessel navigation system. To this end, an accurate prediction of dynamic obstacle trajectories is vital. Traditional approaches to trajectory prediction face limitations in generalizability and often fail to account for the intentions of other vessels. While recent research has considered incorporating the intentions of dynamic obstacles, these efforts are typically based on the own-ship's interpretation of the situation. The current state-of-the-art in this area is a Dynamic Bayesian Network (DBN) model, which infers target vessel intentions by considering multiple underlying causes and allowing for different interpretations of the situation by different vessels. However, since its inception, there have not been any significant structural improvements to this model. In this paper, we propose enhancing the DBN model by incorporating considerations for grounding hazards and vessel waypoint information. The proposed model is validated using real vessel encounters extracted from historical Automatic Identification System (AIS) data.
>
---
#### [replaced 022] HiBerNAC: Hierarchical Brain-emulated Robotic Neural Agent Collective for Disentangling Complex Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08296v2](http://arxiv.org/pdf/2506.08296v2)**

> **作者:** Hongjun Wu; Heng Zhang; Pengsong Zhang; Jin Wang; Cong Wang
>
> **备注:** 31 pages,5 figures
>
> **摘要:** Recent advances in multimodal vision-language-action (VLA) models have revolutionized traditional robot learning, enabling systems to interpret vision, language, and action in unified frameworks for complex task planning. However, mastering complex manipulation tasks remains an open challenge, constrained by limitations in persistent contextual memory, multi-agent coordination under uncertainty, and dynamic long-horizon planning across variable sequences. To address this challenge, we propose \textbf{HiBerNAC}, a \textbf{Hi}erarchical \textbf{B}rain-\textbf{e}mulated \textbf{r}obotic \textbf{N}eural \textbf{A}gent \textbf{C}ollective, inspired by breakthroughs in neuroscience, particularly in neural circuit mechanisms and hierarchical decision-making. Our framework combines: (1) multimodal VLA planning and reasoning with (2) neuro-inspired reflection and multi-agent mechanisms, specifically designed for complex robotic manipulation tasks. By leveraging neuro-inspired functional modules with decentralized multi-agent collaboration, our approach enables robust and enhanced real-time execution of complex manipulation tasks. In addition, the agentic system exhibits scalable collective intelligence via dynamic agent specialization, adapting its coordination strategy to variable task horizons and complexity. Through extensive experiments on complex manipulation tasks compared with state-of-the-art VLA models, we demonstrate that \textbf{HiBerNAC} reduces average long-horizon task completion time by 23\%, and achieves non-zero success rates (12\textendash 31\%) on multi-path tasks where prior state-of-the-art VLA models consistently fail. These results provide indicative evidence for bridging biological cognition and robotic learning mechanisms.
>
---
#### [replaced 023] Reactive and Safety-Aware Path Replanning for Collaborative Applications
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.07192v2](http://arxiv.org/pdf/2503.07192v2)**

> **作者:** Cesare Tonola; Marco Faroni; Saeed Abdolshah; Mazin Hamad; Sami Haddadin; Nicola Pedrocchi; Manuel Beschi
>
> **备注:** Submitted to IEEE
>
> **摘要:** This paper addresses motion replanning in human-robot collaborative scenarios, emphasizing reactivity and safety-compliant efficiency. While existing human-aware motion planners are effective in structured environments, they often struggle with unpredictable human behavior, leading to safety measures that limit robot performance and throughput. In this study, we combine reactive path replanning and a safety-aware cost function, allowing the robot to adjust its path to changes in the human state. This solution reduces the execution time and the need for trajectory slowdowns without sacrificing safety. Simulations and real-world experiments show the method's effectiveness compared to standard human-robot cooperation approaches, with efficiency enhancements of up to 60\%.
>
---
#### [replaced 024] 4D Radar-Inertial Odometry based on Gaussian Modeling and Multi-Hypothesis Scan Matching
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.13639v2](http://arxiv.org/pdf/2412.13639v2)**

> **作者:** Fernando Amodeo; Luis Merino; Fernando Caballero
>
> **备注:** Our code and results can be publicly accessed at: https://github.com/robotics-upo/gaussian-rio-cpp
>
> **摘要:** 4D millimeter-wave (mmWave) radars are sensors that provide robustness against adverse weather conditions (rain, snow, fog, etc.), and as such they are increasingly being used for odometry and SLAM applications. However, the noisy and sparse nature of the returned scan data proves to be a challenging obstacle for existing point cloud matching based solutions, especially those originally intended for more accurate sensors such as LiDAR. Inspired by visual odometry research around 3D Gaussian Splatting, in this paper we propose using freely positioned 3D Gaussians to create a summarized representation of a radar point cloud tolerant to sensor noise, and subsequently leverage its inherent probability distribution function for registration (similar to NDT). Moreover, we propose simultaneously optimizing multiple scan matching hypotheses in order to further increase the robustness of the system against local optima of the function. Finally, we fuse our Gaussian modeling and scan matching algorithms into an EKF radar-inertial odometry system designed after current best practices. Experiments using publicly available 4D radar datasets show that our Gaussian-based odometry is comparable to existing registration algorithms, outperforming them in several sequences.
>
---
#### [replaced 025] STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.03863v2](http://arxiv.org/pdf/2506.03863v2)**

> **作者:** Hao Li; Qi Lv; Rui Shao; Xiang Deng; Yinchuan Li; Jianye Hao; Liqiang Nie
>
> **备注:** Accepted by ICML 2025 Spotlight
>
> **摘要:** Transforming complex actions into discrete skill abstractions has demonstrated strong potential for robotic manipulation. Existing approaches mainly leverage latent variable models, e.g., VQ-VAE, to learn skill abstractions through learned vectors (codebooks), while they suffer from codebook collapse and modeling the causal relationship between learned skills. To address these limitations, we present \textbf{S}kill \textbf{T}raining with \textbf{A}ugmented \textbf{R}otation (\textbf{STAR}), a framework that advances both skill learning and composition to complete complex behaviors. Specifically, to prevent codebook collapse, we devise rotation-augmented residual skill quantization (RaRSQ). It encodes relative angles between encoder outputs into the gradient flow by rotation-based gradient mechanism. Points within the same skill code are forced to be either pushed apart or pulled closer together depending on gradient directions. Further, to capture the causal relationship between skills, we present causal skill transformer (CST) which explicitly models dependencies between skill representations through an autoregressive mechanism for coherent action generation. Extensive experiments demonstrate the superiority of STAR on both LIBERO benchmark and realworld tasks, with around 12\% improvement over the baselines.
>
---
