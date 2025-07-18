# 机器人 cs.RO

- **最新发布 29 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] One For All: LLM-based Heterogeneous Mission Planning in Precision Agriculture
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于农业机器人任务规划，解决非技术人员操控多种机器人的问题。通过自然语言接口和大语言模型，实现复杂任务的自动化执行。**

- **链接: [http://arxiv.org/pdf/2506.10106v1](http://arxiv.org/pdf/2506.10106v1)**

> **作者:** Marcos Abel Zuzuárregui; Mustafa Melih Toslak; Stefano Carpin
>
> **备注:** Accepted to International Federation of Automatic Control (IFAC) Sensing, Control and Automation Technologies for Agriculture - 8th AGRICONTROL 2025
>
> **摘要:** Artificial intelligence is transforming precision agriculture, offering farmers new tools to streamline their daily operations. While these technological advances promise increased efficiency, they often introduce additional complexity and steep learning curves that are particularly challenging for non-technical users who must balance tech adoption with existing workloads. In this paper, we present a natural language (NL) robotic mission planner that enables non-specialists to control heterogeneous robots through a common interface. By leveraging large language models (LLMs) and predefined primitives, our architecture seamlessly translates human language into intermediate descriptions that can be executed by different robotic platforms. With this system, users can formulate complex agricultural missions without writing any code. In the work presented in this paper, we extend our previous system tailored for wheeled robot mission planning through a new class of experiments involving robotic manipulation and computer vision tasks. Our results demonstrate that the architecture is both general enough to support a diverse set of robots and powerful enough to execute complex mission requests. This work represents a significant step toward making robotic automation in precision agriculture more accessible to non-technical users.
>
---
#### [new 002] Leveraging LLMs for Mission Planning in Precision Agriculture
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于农业机器人任务规划领域，旨在解决用户难以编程控制机器人的问题。通过LLMs将自然语言指令转化为标准任务计划，提升机器人自主执行复杂任务的能力。**

- **链接: [http://arxiv.org/pdf/2506.10093v1](http://arxiv.org/pdf/2506.10093v1)**

> **作者:** Marcos Abel Zuzuárregui; Stefano Carpin
>
> **备注:** Published in Proceedings of 2025 International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Robotics and artificial intelligence hold significant potential for advancing precision agriculture. While robotic systems have been successfully deployed for various tasks, adapting them to perform diverse missions remains challenging, particularly because end users often lack technical expertise. In this paper, we present an end-to-end system that leverages large language models (LLMs), specifically ChatGPT, to enable users to assign complex data collection tasks to autonomous robots using natural language instructions. To enhance reusability, mission plans are encoded using an existing IEEE task specification standard, and are executed on robots via ROS2 nodes that bridge high-level mission descriptions with existing ROS libraries. Through extensive experiments, we highlight the strengths and limitations of LLMs in this context, particularly regarding spatial reasoning and solving complex routing challenges, and show how our proposed implementation overcomes them.
>
---
#### [new 003] A Navigation Framework Utilizing Vision-Language Models
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决复杂环境中的导航问题。通过模块化框架结合预训练模型与轻量规划逻辑，提升导航效率与适应性。**

- **链接: [http://arxiv.org/pdf/2506.10172v1](http://arxiv.org/pdf/2506.10172v1)**

> **作者:** Yicheng Duan; Kaiyu tang
>
> **摘要:** Vision-and-Language Navigation (VLN) presents a complex challenge in embodied AI, requiring agents to interpret natural language instructions and navigate through visually rich, unfamiliar environments. Recent advances in large vision-language models (LVLMs), such as CLIP and Flamingo, have significantly improved multimodal understanding but introduced new challenges related to computational cost and real-time deployment. In this project, we propose a modular, plug-and-play navigation framework that decouples vision-language understanding from action planning. By integrating a frozen vision-language model, Qwen2.5-VL-7B-Instruct, with lightweight planning logic, we aim to achieve flexible, fast, and adaptable navigation without extensive model fine-tuning. Our framework leverages prompt engineering, structured history management, and a two-frame visual input strategy to enhance decision-making continuity across navigation steps. We evaluate our system on the Room-to-Room benchmark within the VLN-CE setting using the Matterport3D dataset and Habitat-Lab simulation environment. Although our initial results reveal challenges in generalizing to unseen environments under strict evaluation settings, our modular approach lays a foundation for scalable and efficient navigation systems, highlighting promising directions for future improvement through enhanced environmental priors and expanded multimodal input integration.
>
---
#### [new 004] Eye, Robot: Learning to Look to Act with a BC-RL Perception-Action Loop
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EyeRobot系统，解决机器人手眼协调问题，通过BC-RL循环训练，使机器人根据视觉主动调整动作。**

- **链接: [http://arxiv.org/pdf/2506.10968v1](http://arxiv.org/pdf/2506.10968v1)**

> **作者:** Justin Kerr; Kush Hari; Ethan Weber; Chung Min Kim; Brent Yi; Tyler Bonnen; Ken Goldberg; Angjoo Kanazawa
>
> **备注:** Project page: https://www.eyerobot.net/
>
> **摘要:** Humans do not passively observe the visual world -- we actively look in order to act. Motivated by this principle, we introduce EyeRobot, a robotic system with gaze behavior that emerges from the need to complete real-world tasks. We develop a mechanical eyeball that can freely rotate to observe its surroundings and train a gaze policy to control it using reinforcement learning. We accomplish this by first collecting teleoperated demonstrations paired with a 360 camera. This data is imported into a simulation environment that supports rendering arbitrary eyeball viewpoints, allowing episode rollouts of eye gaze on top of robot demonstrations. We then introduce a BC-RL loop to train the hand and eye jointly: the hand (BC) agent is trained from rendered eye observations, and the eye (RL) agent is rewarded when the hand produces correct action predictions. In this way, hand-eye coordination emerges as the eye looks towards regions which allow the hand to complete the task. EyeRobot implements a foveal-inspired policy architecture allowing high resolution with a small compute budget, which we find also leads to the emergence of more stable fixation as well as improved ability to track objects and ignore distractors. We evaluate EyeRobot on five panoramic workspace manipulation tasks requiring manipulation in an arc surrounding the robot arm. Our experiments suggest EyeRobot exhibits hand-eye coordination behaviors which effectively facilitate manipulation over large workspaces with a single camera. See project site for videos: https://www.eyerobot.net/
>
---
#### [new 005] Invariant Extended Kalman Filter for Autonomous Surface Vessels with Partial Orientation Measurements
- **分类: cs.RO**

- **简介: 该论文属于自主水面航行器状态估计任务，解决开放海域中部分姿态测量下的定位问题，提出改进的InEKF框架以提升估计精度。**

- **链接: [http://arxiv.org/pdf/2506.10850v1](http://arxiv.org/pdf/2506.10850v1)**

> **作者:** Derek Benham; Easton Potokar; Joshua G. Mangelson
>
> **备注:** Presented at the 2025 IEEE ICRA Workshop on Field Robotics. 8 pages, 4 figures, 2 tables
>
> **摘要:** Autonomous surface vessels (ASVs) are increasingly vital for marine science, offering robust platforms for underwater mapping and inspection. Accurate state estimation, particularly of vehicle pose, is paramount for precise seafloor mapping, as even small surface deviations can have significant consequences when sensing the seafloor below. To address this challenge, we propose an Invariant Extended Kalman Filter (InEKF) framework designed to integrate partial orientation measurements. While conventional estimation often relies on relative position measurements to fixed landmarks, open ocean ASVs primarily observe a receding horizon. We leverage forward-facing monocular cameras to estimate roll and pitch with respect to this horizon, which provides yaw-ambiguous partial orientation information. To effectively utilize these measurements within the InEKF, we introduce a novel framework for incorporating such partial orientation data. This approach contrasts with traditional InEKF implementations that assume full orientation measurements and is particularly relevant for planar vehicle motion constrained to a "seafaring plane." This paper details the developed InEKF framework; its integration with horizon-based roll/pitch observations and dual-antenna GPS heading measurements for ASV state estimation; and provides a comparative analysis against the InEKF using full orientation and a Multiplicative EKF (MEKF). Our results demonstrate the efficacy and robustness of the proposed partial orientation measurements for accurate ASV state estimation in open ocean environments.
>
---
#### [new 006] RICE: Reactive Interaction Controller for Cluttered Canopy Environment
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人导航任务，解决在杂乱植物冠层中安全导航的问题。通过设计一种反应式控制器，利用末端位置和触觉反馈实现无损移动。**

- **链接: [http://arxiv.org/pdf/2506.10383v1](http://arxiv.org/pdf/2506.10383v1)**

> **作者:** Nidhi Homey Parayil; Thierry Peynot; Chris Lehnert
>
> **备注:** This work has been submitted to the IEEE RAL for possible publication
>
> **摘要:** Robotic navigation in dense, cluttered environments such as agricultural canopies presents significant challenges due to physical and visual occlusion caused by leaves and branches. Traditional vision-based or model-dependent approaches often fail in these settings, where physical interaction without damaging foliage and branches is necessary to reach a target. We present a novel reactive controller that enables safe navigation for a robotic arm in a contact-rich, cluttered, deformable environment using end-effector position and real-time tactile feedback. Our proposed framework's interaction strategy is based on a trade-off between minimizing disturbance by maneuvering around obstacles and pushing through them to move towards the target. We show that over 35 trials in 3 experimental plant setups with an occluded target, the proposed controller successfully reached the target in all trials without breaking any branch and outperformed the state-of-the-art model-free controller in robustness and adaptability. This work lays the foundation for safe, adaptive interaction in cluttered, contact-rich deformable environments, enabling future agricultural tasks such as pruning and harvesting in plant canopies.
>
---
#### [new 007] Estimating the Joint Probability of Scenario Parameters with Gaussian Mixture Copula Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于场景参数联合概率建模任务，旨在提升自动驾驶系统安全验证的准确性。通过引入高斯混合Copula模型，解决场景参数依赖关系建模问题，并优于传统方法。**

- **链接: [http://arxiv.org/pdf/2506.10098v1](http://arxiv.org/pdf/2506.10098v1)**

> **作者:** Christian Reichenbächer; Philipp Rank; Jochen Hipp; Oliver Bringmann
>
> **备注:** 8 pages, 4 figures; This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper presents the first application of Gaussian Mixture Copula Models to the statistical modeling of driving scenarios for the safety validation of automated driving systems. Knowledge of the joint probability distribution of scenario parameters is essential for scenario-based safety assessment, where risk quantification depends on the likelihood of concrete parameter combinations. Gaussian Mixture Copula Models bring together the multimodal expressivity of Gaussian Mixture Models and the flexibility of copulas, enabling separate modeling of marginal distributions and dependencies. We benchmark Gaussian Mixture Copula Models against previously proposed approaches - Gaussian Mixture Models and Gaussian Copula Models - using real-world driving data drawn from scenarios defined in United Nations Regulation No. 157. Our evaluation across 18 million scenario instances demonstrates that Gaussian Mixture Copula Models provide a better fit to the data in terms of both likelihood and Sinkhorn distance. These results suggest that Gaussian Mixture Copula Models are a compelling foundation for future scenario-based validation frameworks.
>
---
#### [new 008] Learning Safe Control via On-the-Fly Bandit Exploration
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于安全控制任务，解决高模型不确定性下的安全控制问题。通过结合控制屏障函数与高斯过程带宽算法，实现安全探索与控制，无需备用控制器。**

- **链接: [http://arxiv.org/pdf/2506.10279v1](http://arxiv.org/pdf/2506.10279v1)**

> **作者:** Alexandre Capone; Ryan Cosner; Aaaron Ames; Sandra Hirche
>
> **备注:** arXiv admin note: text overlap with arXiv:2311.02133
>
> **摘要:** Control tasks with safety requirements under high levels of model uncertainty are increasingly common. Machine learning techniques are frequently used to address such tasks, typically by leveraging model error bounds to specify robust constraint-based safety filters. However, if the learned model uncertainty is very high, the corresponding filters are potentially invalid, meaning no control input satisfies the constraints imposed by the safety filter. While most works address this issue by assuming some form of safe backup controller, ours tackles it by collecting additional data on the fly using a Gaussian process bandit-type algorithm. We combine a control barrier function with a learned model to specify a robust certificate that ensures safety if feasible. Whenever infeasibility occurs, we leverage the control barrier function to guide exploration, ensuring the collected data contributes toward the closed-loop system safety. By combining a safety filter with exploration in this manner, our method provably achieves safety in a setting that allows for a zero-mean prior dynamics model, without requiring a backup controller. To the best of our knowledge, it is the first safe learning-based control method that achieves this.
>
---
#### [new 009] Innovative Adaptive Imaged Based Visual Servoing Control of 6 DoFs Industrial Robot Manipulators
- **分类: cs.RO; cs.SY; eess.SY; 93B52 (Primary), 93C85 (Secondary)**

- **简介: 该论文属于工业机器人视觉伺服控制任务，解决无3D点特征时的位姿控制问题，提出自适应控制算法提升系统稳定性与精度。**

- **链接: [http://arxiv.org/pdf/2506.10240v1](http://arxiv.org/pdf/2506.10240v1)**

> **作者:** Rongfei Li; Francis Assadian
>
> **备注:** 22 pages, 13 figures. To appear in: Innovative Adaptive Image-Based Visual Servoing Control of 6 DoFs Industrial Robot Manipulators, IntechOpen, 2024. For published version, see this http URL: https://doi.org/10.5772/intechopen.1004857
>
> **摘要:** Image-based visual servoing (IBVS) methods have been well developed and used in many applications, especially in pose (position and orientation) alignment. However, most research papers focused on developing control solutions when 3D point features can be detected inside the field of view. This work proposes an innovative feedforward-feedback adaptive control algorithm structure with the Youla Parameterization method. A designed feature estimation loop ensures stable and fast motion control when point features are outside the field of view. As 3D point features move inside the field of view, the IBVS feedback loop preserves the precision of the pose at the end of the control period. Also, an adaptive controller is developed in the feedback loop to stabilize the system in the entire range of operations. The nonlinear camera and robot manipulator model is linearized and decoupled online by an adaptive algorithm. The adaptive controller is then computed based on the linearized model evaluated at current linearized point. The proposed solution is robust and easy to implement in different industrial robotic systems. Various scenarios are used in simulations to validate the effectiveness and robust performance of the proposed controller.
>
---
#### [new 010] In-Hand Object Pose Estimation via Visual-Tactile Fusion
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取中的位姿估计任务，旨在解决视觉遮挡导致的定位不准问题，通过融合视觉与触觉信息提升估计精度。**

- **链接: [http://arxiv.org/pdf/2506.10787v1](http://arxiv.org/pdf/2506.10787v1)**

> **作者:** Felix Nonnengießer; Alap Kshirsagar; Boris Belousov; Jan Peters
>
> **备注:** 8 pages
>
> **摘要:** Accurate in-hand pose estimation is crucial for robotic object manipulation, but visual occlusion remains a major challenge for vision-based approaches. This paper presents an approach to robotic in-hand object pose estimation, combining visual and tactile information to accurately determine the position and orientation of objects grasped by a robotic hand. We address the challenge of visual occlusion by fusing visual information from a wrist-mounted RGB-D camera with tactile information from vision-based tactile sensors mounted on the fingertips of a robotic gripper. Our approach employs a weighting and sensor fusion module to combine point clouds from heterogeneous sensor types and control each modality's contribution to the pose estimation process. We use an augmented Iterative Closest Point (ICP) algorithm adapted for weighted point clouds to estimate the 6D object pose. Our experiments show that incorporating tactile information significantly improves pose estimation accuracy, particularly when occlusion is high. Our method achieves an average pose estimation error of 7.5 mm and 16.7 degrees, outperforming vision-only baselines by up to 20%. We also demonstrate the ability of our method to perform precise object manipulation in a real-world insertion task.
>
---
#### [new 011] GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决真实场景下策略泛化问题。提出GenManip平台与基准，评估模块化与端到端策略的泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.10966v1](http://arxiv.org/pdf/2506.10966v1)**

> **作者:** Ning Gao; Yilun Chen; Shuai Yang; Xinyi Chen; Yang Tian; Hao Li; Haifeng Huang; Hanqing Wang; Tai Wang; Jiangmiao Pang
>
> **摘要:** Robotic manipulation in real-world settings remains challenging, especially regarding robust generalization. Existing simulation platforms lack sufficient support for exploring how policies adapt to varied instructions and scenarios. Thus, they lag behind the growing interest in instruction-following foundation models like LLMs, whose adaptability is crucial yet remains underexplored in fair comparisons. To bridge this gap, we introduce GenManip, a realistic tabletop simulation platform tailored for policy generalization studies. It features an automatic pipeline via LLM-driven task-oriented scene graph to synthesize large-scale, diverse tasks using 10K annotated 3D object assets. To systematically assess generalization, we present GenManip-Bench, a benchmark of 200 scenarios refined via human-in-the-loop corrections. We evaluate two policy types: (1) modular manipulation systems integrating foundation models for perception, reasoning, and planning, and (2) end-to-end policies trained through scalable data collection. Results show that while data scaling benefits end-to-end methods, modular systems enhanced with foundation models generalize more effectively across diverse scenarios. We anticipate this platform to facilitate critical insights for advancing policy generalization in realistic conditions. Project Page: https://genmanip.axi404.top/.
>
---
#### [new 012] A Novel Feedforward Youla Parameterization Method for Avoiding Local Minima in Stereo Image Based Visual Servoing Control
- **分类: cs.RO; cs.SY; eess.SY; 93B52 (Primary), 93C85 (Secondary)**

- **简介: 该论文属于视觉伺服控制任务，解决立体视觉中因局部极小值导致的位姿估计偏差问题。通过结合前馈控制器与Youla参数化反馈控制器，提高定位精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.10252v1](http://arxiv.org/pdf/2506.10252v1)**

> **作者:** Rongfei Li; Francis Assadian
>
> **备注:** 36 pages, 19 figures, Journal, Published in: Applied Sciences, 2025, vol. 15, article 4991. For published version, see this http URL: https://doi.org/10.3390/app15094991
>
> **摘要:** In robot navigation and manipulation, accurately determining the camera's pose relative to the environment is crucial for effective task execution. In this paper, we systematically prove that this problem corresponds to the Perspective-3-Point (P3P) formulation, where exactly three known 3D points and their corresponding 2D image projections are used to estimate the pose of a stereo camera. In image-based visual servoing (IBVS) control, the system becomes overdetermined, as the 6 degrees of freedom (DoF) of the stereo camera must align with 9 observed 2D features in the scene. When more constraints are imposed than available DoFs, global stability cannot be guaranteed, as the camera may become trapped in a local minimum far from the desired configuration during servoing. To address this issue, we propose a novel control strategy for accurately positioning a calibrated stereo camera. Our approach integrates a feedforward controller with a Youla parameterization-based feedback controller, ensuring robust servoing performance. Through simulations, we demonstrate that our method effectively avoids local minima and enables the camera to reach the desired pose accurately and efficiently.
>
---
#### [new 013] Towards more efficient quantitative safety validation of residual risk for assisted and automated driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全验证任务，旨在提高残余风险量化验证的效率。针对FOT测试效率低的问题，评估了现有缩减方法，提出模型并分析其局限性。**

- **链接: [http://arxiv.org/pdf/2506.10363v1](http://arxiv.org/pdf/2506.10363v1)**

> **作者:** Daniel Betschinske; Malte Schrimpf; Steven Peters; Kamil Klonecki; Jan Peter Karch; Moritz Lippert
>
> **摘要:** The safety validation of Advanced Driver Assistance Systems (ADAS) and Automated Driving Systems (ADS) increasingly demands efficient and reliable methods to quantify residual risk while adhering to international standards such as ISO 21448. Traditionally, Field Operational Testing (FOT) has been pivotal for macroscopic safety validation of automotive driving functions up to SAE automation level 2. However, state-of-the-art derivations for empirical safety demonstrations using FOT often result in impractical testing efforts, particularly at higher automation levels. Even at lower automation levels, this limitation - coupled with the substantial costs associated with FOT - motivates the exploration of approaches to enhance the efficiency of FOT-based macroscopic safety validation. Therefore, this publication systematically identifies and evaluates state-of-the-art Reduction Approaches (RAs) for FOT, including novel methods reported in the literature. Based on an analysis of ISO 21448, two models are derived: a generic model capturing the argumentation components of the standard, and a base model, exemplarily applied to Automatic Emergency Braking (AEB) systems, establishing a baseline for the real-world driving requirement for a Quantitative Safety Validation of Residual Risk (QSVRR). Subsequently, the RAs are assessed using four criteria: quantifiability, threats to validity, missing links, and black box compatibility, highlighting potential benefits, inherent limitations, and identifying key areas for further research. Our evaluation reveals that, while several approaches offer potential, none are free from missing links or other substantial shortcomings. Moreover, no identified alternative can fully replace FOT, reflecting its crucial role in the safety validation of ADAS and ADS.
>
---
#### [new 014] An $O(n$)-Algorithm for the Higher-Order Kinematics and Inverse Dynamics of Serial Manipulators using Spatial Representation of Twists
- **分类: cs.RO; cs.SC; math.GR; math.OC; physics.class-ph**

- **简介: 该论文属于机器人动力学与运动学计算任务，解决串联系统的高阶运动学和逆动力学问题，提出一种基于空间表示的O(n)算法。**

- **链接: [http://arxiv.org/pdf/2506.10686v1](http://arxiv.org/pdf/2506.10686v1)**

> **作者:** Andreas Mueller
>
> **摘要:** Optimal control in general, and flatness-based control in particular, of robotic arms necessitate to compute the first and second time derivatives of the joint torques/forces required to achieve a desired motion. In view of the required computational efficiency, recursive $O(n)$-algorithms were proposed to this end. Aiming at compact yet efficient formulations, a Lie group formulation was recently proposed, making use of body-fixed and hybrid representation of twists and wrenches. In this paper a formulation is introduced using the spatial representation. The second-order inverse dynamics algorithm is accompanied by a fourth-order forward and inverse kinematics algorithm. An advantage of all Lie group formulations is that they can be parameterized in terms of vectorial quantities that are readily available. The method is demonstrated for the 7 DOF Franka Emika Panda robot.
>
---
#### [new 015] RationalVLA: A Rational Vision-Language-Action Model with Dual System
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉-语言-动作任务，旨在解决指令不明确或不可行时的机器人操作问题。提出RationalVLA模型，结合视觉语言与低级控制，提升指令理解和执行能力。**

- **链接: [http://arxiv.org/pdf/2506.10826v1](http://arxiv.org/pdf/2506.10826v1)**

> **作者:** Wenxuan Song; Jiayi Chen; Wenxue Li; Xu He; Han Zhao; Pengxiang Ding Shiyan Su; Feilong Tang; Xuelian Cheng; Donglin Wang; Zongyuan Ge; Xinhu Zheng; Zhe Liu; Hesheng Wang; Yunhui Liu; Haoang Li
>
> **备注:** 14 pages
>
> **摘要:** A fundamental requirement for real-world robotic deployment is the ability to understand and respond to natural language instructions. Existing language-conditioned manipulation tasks typically assume that instructions are perfectly aligned with the environment. This assumption limits robustness and generalization in realistic scenarios where instructions may be ambiguous, irrelevant, or infeasible. To address this problem, we introduce RAtional MAnipulation (RAMA), a new benchmark that challenges models with both unseen executable instructions and defective ones that should be rejected. In RAMA, we construct a dataset with over 14,000 samples, including diverse defective instructions spanning six dimensions: visual, physical, semantic, motion, safety, and out-of-context. We further propose the Rational Vision-Language-Action model (RationalVLA). It is a dual system for robotic arms that integrates the high-level vision-language model with the low-level manipulation policy by introducing learnable latent space embeddings. This design enables RationalVLA to reason over instructions, reject infeasible commands, and execute manipulation effectively. Experiments demonstrate that RationalVLA outperforms state-of-the-art baselines on RAMA by a 14.5% higher success rate and 0.94 average task length, while maintaining competitive performance on standard manipulation tasks. Real-world trials further validate its effectiveness and robustness in practical applications. Our project page is https://irpn-eai.github.io/rationalvla.
>
---
#### [new 016] Modeling Trust Dynamics in Robot-Assisted Delivery: Impact of Trust Repair Strategies
- **分类: cs.RO**

- **简介: 该论文研究机器人辅助配送中的信任动态，解决如何通过修复策略影响人类信任的问题。通过IOHMM模型分析人类行为与信任变化。**

- **链接: [http://arxiv.org/pdf/2506.10884v1](http://arxiv.org/pdf/2506.10884v1)**

> **作者:** Dong Hae Mangalindan; Karthik Kandikonda; Ericka Rovira; Vaibhav Srivastava
>
> **摘要:** With increasing efficiency and reliability, autonomous systems are becoming valuable assistants to humans in various tasks. In the context of robot-assisted delivery, we investigate how robot performance and trust repair strategies impact human trust. In this task, while handling a secondary task, humans can choose to either send the robot to deliver autonomously or manually control it. The trust repair strategies examined include short and long explanations, apology and promise, and denial. Using data from human participants, we model human behavior using an Input-Output Hidden Markov Model (IOHMM) to capture the dynamics of trust and human action probabilities. Our findings indicate that humans are more likely to deploy the robot autonomously when their trust is high. Furthermore, state transition estimates show that long explanations are the most effective at repairing trust following a failure, while denial is most effective at preventing trust loss. We also demonstrate that the trust estimates generated by our model are isomorphic to self-reported trust values, making them interpretable. This model lays the groundwork for developing optimal policies that facilitate real-time adjustment of human trust in autonomous systems.
>
---
#### [new 017] Grounded Vision-Language Navigation for UAVs with Open-Vocabulary Goal Understanding
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉语言导航任务，解决UAV在复杂环境中根据语言指令导航的问题。提出VLFly框架，实现开放词汇目标理解和连续控制。**

- **链接: [http://arxiv.org/pdf/2506.10756v1](http://arxiv.org/pdf/2506.10756v1)**

> **作者:** Yuhang Zhang; Haosheng Yu; Jiaping Xiao; Mir Feroskhan
>
> **摘要:** Vision-and-language navigation (VLN) is a long-standing challenge in autonomous robotics, aiming to empower agents with the ability to follow human instructions while navigating complex environments. Two key bottlenecks remain in this field: generalization to out-of-distribution environments and reliance on fixed discrete action spaces. To address these challenges, we propose Vision-Language Fly (VLFly), a framework tailored for Unmanned Aerial Vehicles (UAVs) to execute language-guided flight. Without the requirement for localization or active ranging sensors, VLFly outputs continuous velocity commands purely from egocentric observations captured by an onboard monocular camera. The VLFly integrates three modules: an instruction encoder based on a large language model (LLM) that reformulates high-level language into structured prompts, a goal retriever powered by a vision-language model (VLM) that matches these prompts to goal images via vision-language similarity, and a waypoint planner that generates executable trajectories for real-time UAV control. VLFly is evaluated across diverse simulation environments without additional fine-tuning and consistently outperforms all baselines. Moreover, real-world VLN tasks in indoor and outdoor environments under direct and indirect instructions demonstrate that VLFly achieves robust open-vocabulary goal understanding and generalized navigation capabilities, even in the presence of abstract language input.
>
---
#### [new 018] Multi-Timescale Dynamics Model Bayesian Optimization for Plasma Stabilization in Tokamaks
- **分类: cs.RO**

- **简介: 该论文属于控制任务，旨在解决核聚变中等离子体不稳定问题。提出多时间尺度贝叶斯优化方法，结合动态模型与高斯过程，提升控制效果。**

- **链接: [http://arxiv.org/pdf/2506.10287v1](http://arxiv.org/pdf/2506.10287v1)**

> **作者:** Rohit Sonker; Alexandre Capone; Andrew Rothstein; Hiro Josep Farre Kaga; Egemen Kolemen; Jeff Schneider
>
> **摘要:** Machine learning algorithms often struggle to control complex real-world systems. In the case of nuclear fusion, these challenges are exacerbated, as the dynamics are notoriously complex, data is poor, hardware is subject to failures, and experiments often affect dynamics beyond the experiment's duration. Existing tools like reinforcement learning, supervised learning, and Bayesian optimization address some of these challenges but fail to provide a comprehensive solution. To overcome these limitations, we present a multi-scale Bayesian optimization approach that integrates a high-frequency data-driven dynamics model with a low-frequency Gaussian process. By updating the Gaussian process between experiments, the method rapidly adapts to new data, refining the predictions of the less reliable dynamical model. We validate our approach by controlling tearing instabilities in the DIII-D nuclear fusion plant. Offline testing on historical data shows that our method significantly outperforms several baselines. Results on live experiments on the DIII-D tokamak, conducted under high-performance plasma scenarios prone to instabilities, shows a 50% success rate, marking a 117% improvement over historical outcomes.
>
---
#### [new 019] EmbodiedGen: Towards a Generative 3D World Engine for Embodied Intelligence
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EmbodiedGen，用于生成高质量、可控制的3D世界，解决 embodied AI 数据不足和成本高的问题。**

- **链接: [http://arxiv.org/pdf/2506.10600v1](http://arxiv.org/pdf/2506.10600v1)**

> **作者:** Wang Xinjie; Liu Liu; Cao Yu; Wu Ruiqi; Qin Wenkang; Wang Dehui; Sui Wei; Su Zhizhong
>
> **摘要:** Constructing a physically realistic and accurately scaled simulated 3D world is crucial for the training and evaluation of embodied intelligence tasks. The diversity, realism, low cost accessibility and affordability of 3D data assets are critical for achieving generalization and scalability in embodied AI. However, most current embodied intelligence tasks still rely heavily on traditional 3D computer graphics assets manually created and annotated, which suffer from high production costs and limited realism. These limitations significantly hinder the scalability of data driven approaches. We present EmbodiedGen, a foundational platform for interactive 3D world generation. It enables the scalable generation of high-quality, controllable and photorealistic 3D assets with accurate physical properties and real-world scale in the Unified Robotics Description Format (URDF) at low cost. These assets can be directly imported into various physics simulation engines for fine-grained physical control, supporting downstream tasks in training and evaluation. EmbodiedGen is an easy-to-use, full-featured toolkit composed of six key modules: Image-to-3D, Text-to-3D, Texture Generation, Articulated Object Generation, Scene Generation and Layout Generation. EmbodiedGen generates diverse and interactive 3D worlds composed of generative 3D assets, leveraging generative AI to address the challenges of generalization and evaluation to the needs of embodied intelligence related research. Code is available at https://horizonrobotics.github.io/robot_lab/embodied_gen/index.html.
>
---
#### [new 020] A Unified Framework for Probabilistic Dynamic-, Trajectory- and Vision-based Virtual Fixtures
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，解决如何动态切换虚拟夹具模式的问题。提出统一框架，实现手动、半自动与全自主控制的无缝切换。**

- **链接: [http://arxiv.org/pdf/2506.10239v1](http://arxiv.org/pdf/2506.10239v1)**

> **作者:** Maximilian Mühlbauer; Freek Stulp; Sylvain Calinon; Alin Albu-Schäffer; João Silvério
>
> **摘要:** Probabilistic Virtual Fixtures (VFs) enable the adaptive selection of the most suitable haptic feedback for each phase of a task, based on learned or perceived uncertainty. While keeping the human in the loop remains essential, for instance, to ensure high precision, partial automation of certain task phases is critical for productivity. We present a unified framework for probabilistic VFs that seamlessly switches between manual fixtures, semi-automated fixtures (with the human handling precise tasks), and full autonomy. We introduce a novel probabilistic Dynamical System-based VF for coarse guidance, enabling the robot to autonomously complete certain task phases while keeping the human operator in the loop. For tasks requiring precise guidance, we extend probabilistic position-based trajectory fixtures with automation allowing for seamless human interaction as well as geometry-awareness and optimal impedance gains. For manual tasks requiring very precise guidance, we also extend visual servoing fixtures with the same geometry-awareness and impedance behaviour. We validate our approach experimentally on different robots, showcasing multiple operation modes and the ease of programming fixtures.
>
---
#### [new 021] Using Language and Road Manuals to Inform Map Reconstruction for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶中的车道拓扑预测任务，旨在提升地图重建的准确性。通过融合语言信息与道路设计手册，改进了模型对道路结构的理解和检测能力。**

- **链接: [http://arxiv.org/pdf/2506.10317v1](http://arxiv.org/pdf/2506.10317v1)**

> **作者:** Akshar Tumu; Henrik I. Christensen; Marcell Vazquez-Chanlatte; Chikao Tsuchiya; Dhaval Bhanderi
>
> **备注:** 4 pages, 3 figures, Accepted at RSS 2025 Workshop - RobotEvaluation@RSS2025
>
> **摘要:** Lane-topology prediction is a critical component of safe and reliable autonomous navigation. An accurate understanding of the road environment aids this task. We observe that this information often follows conventions encoded in natural language, through design codes that reflect the road structure and road names that capture the road functionality. We augment this information in a lightweight manner to SMERF, a map-prior-based online lane-topology prediction model, by combining structured road metadata from OSM maps and lane-width priors from Road design manuals with the road centerline encodings. We evaluate our method on two geo-diverse complex intersection scenarios. Our method shows improvement in both lane and traffic element detection and their association. We report results using four topology-aware metrics to comprehensively assess the model performance. These results demonstrate the ability of our approach to generalize and scale to diverse topologies and conditions.
>
---
#### [new 022] Are We Generalizing from the Exception? An In-the-Wild Study on Group-Sensitive Conversation Design in Human-Agent Interactions
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，研究群体敏感对话设计对多人互动的影响，通过实验证明现有方法的局限性，并提出多模态策略的重要性。**

- **链接: [http://arxiv.org/pdf/2506.10462v1](http://arxiv.org/pdf/2506.10462v1)**

> **作者:** Ana Müller; Sabina Jeschke; Anja Richert
>
> **备注:** Accepted as a regular paper at the 2025 IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). \c{opyright} IEEE. This is the preprint version. The final version will appear in the IEEE proceedings
>
> **摘要:** This paper investigates the impact of a group-adaptive conversation design in two socially interactive agents (SIAs) through two real-world studies. Both SIAs - Furhat, a social robot, and MetaHuman, a virtual agent - were equipped with a conversational artificial intelligence (CAI) backend combining hybrid retrieval and generative models. The studies were carried out in an in-the-wild setting with a total of $N = 188$ participants who interacted with the SIAs - in dyads, triads or larger groups - at a German museum. Although the results did not reveal a significant effect of the group-sensitive conversation design on perceived satisfaction, the findings provide valuable insights into the challenges of adapting CAI for multi-party interactions and across different embodiments (robot vs.\ virtual agent), highlighting the need for multimodal strategies beyond linguistic pluralization. These insights contribute to the fields of Human-Agent Interaction (HAI), Human-Robot Interaction (HRI), and broader Human-Machine Interaction (HMI), providing insights for future research on effective dialogue adaptation in group settings.
>
---
#### [new 023] Vib2Move: In-Hand Object Reconfiguration via Fingertip Micro-Vibrations
- **分类: cs.RO**

- **简介: 该论文属于物体重新定位任务，解决平面物体精确定位问题。通过指尖微振动和重力实现高精度操控。**

- **链接: [http://arxiv.org/pdf/2506.10923v1](http://arxiv.org/pdf/2506.10923v1)**

> **作者:** Xili Yi; Nima Fazeli
>
> **备注:** 11 pages, 12 figures
>
> **摘要:** We introduce Vib2Move, a novel approach for in-hand object reconfiguration that uses fingertip micro-vibrations and gravity to precisely reposition planar objects. Our framework comprises three key innovations. First, we design a vibration-based actuator that dynamically modulates the effective finger-object friction coefficient, effectively emulating changes in gripping force. Second, we derive a sliding motion model for objects clamped in a parallel gripper with two symmetric, variable-friction contact patches. Third, we propose a motion planner that coordinates end-effector finger trajectories and fingertip vibrations to achieve the desired object pose. In real-world trials, Vib2Move consistently yields final positioning errors below 6 mm, demonstrating reliable, high-precision manipulation across a variety of planar objects. For more results and information, please visit https://vib2move.github.io.
>
---
#### [new 024] Data-Driven Prediction of Dynamic Interactions Between Robot Appendage and Granular Material
- **分类: cs.RO; cs.AI; cs.LG; cs.NA; math.NA**

- **简介: 该论文属于机器人与颗粒材料交互预测任务，旨在提升计算效率与预测精度。通过数据驱动方法结合降维、代理模型和数据同化，实现高效准确的动态交互预测。**

- **链接: [http://arxiv.org/pdf/2506.10875v1](http://arxiv.org/pdf/2506.10875v1)**

> **作者:** Guanjin Wang; Xiangxue Zhao; Shapour Azarm; Balakumar Balachandran
>
> **摘要:** An alternative data-driven modeling approach has been proposed and employed to gain fundamental insights into robot motion interaction with granular terrain at certain length scales. The approach is based on an integration of dimension reduction (Sequentially Truncated Higher-Order Singular Value Decomposition), surrogate modeling (Gaussian Process), and data assimilation techniques (Reduced Order Particle Filter). This approach can be used online and is based on offline data, obtained from the offline collection of high-fidelity simulation data and a set of sparse experimental data. The results have shown that orders of magnitude reduction in computational time can be obtained from the proposed data-driven modeling approach compared with physics-based high-fidelity simulations. With only simulation data as input, the data-driven prediction technique can generate predictions that have comparable accuracy as simulations. With both simulation data and sparse physical experimental measurement as input, the data-driven approach with its embedded data assimilation techniques has the potential in outperforming only high-fidelity simulations for the long-horizon predictions. In addition, it is demonstrated that the data-driven modeling approach can also reproduce the scaling relationship recovered by physics-based simulations for maximum resistive forces, which may indicate its general predictability beyond a case-by-case basis. The results are expected to help robot navigation and exploration in unknown and complex terrains during both online and offline phases.
>
---
#### [new 025] Demonstrating Multi-Suction Item Picking at Scale via Multi-Modal Learning of Pick Success
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人抓取任务，解决多吸盘抓取成功率预测问题。通过多模态学习提升抓取性能，利用RGB、深度和语义信息进行训练与微调。**

- **链接: [http://arxiv.org/pdf/2506.10359v1](http://arxiv.org/pdf/2506.10359v1)**

> **作者:** Che Wang; Jeroen van Baar; Chaitanya Mitash; Shuai Li; Dylan Randle; Weiyao Wang; Sumedh Sontakke; Kostas E. Bekris; Kapil Katyal
>
> **备注:** Accepted to Robotics: Science and Systems (RSS 2025), 15 pages
>
> **摘要:** This work demonstrates how autonomously learning aspects of robotic operation from sparsely-labeled, real-world data of deployed, engineered solutions at industrial scale can provide with solutions that achieve improved performance. Specifically, it focuses on multi-suction robot picking and performs a comprehensive study on the application of multi-modal visual encoders for predicting the success of candidate robotic picks. Picking diverse items from unstructured piles is an important and challenging task for robot manipulation in real-world settings, such as warehouses. Methods for picking from clutter must work for an open set of items while simultaneously meeting latency constraints to achieve high throughput. The demonstrated approach utilizes multiple input modalities, such as RGB, depth and semantic segmentation, to estimate the quality of candidate multi-suction picks. The strategy is trained from real-world item picking data, with a combination of multimodal pretrain and finetune. The manuscript provides comprehensive experimental evaluation performed over a large item-picking dataset, an item-picking dataset targeted to include partial occlusions, and a package-picking dataset, which focuses on containers, such as boxes and envelopes, instead of unpackaged items. The evaluation measures performance for different item configurations, pick scenes, and object types. Ablations help to understand the effects of in-domain pretraining, the impact of different modalities and the importance of finetuning. These ablations reveal both the importance of training over multiple modalities but also the ability of models to learn during pretraining the relationship between modalities so that during finetuning and inference, only a subset of them can be used as input.
>
---
#### [new 026] Impacts between multibody systems and deformable structures
- **分类: physics.class-ph; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于多体系统与弹性结构碰撞建模任务，旨在解决精确模拟碰撞力学的问题，通过分析单边接触和弹性元件的参数进行数值研究。**

- **链接: [http://arxiv.org/pdf/2506.10034v1](http://arxiv.org/pdf/2506.10034v1)**

> **作者:** Lipinski Krzysztof
>
> **备注:** 20 pages, 11 figures, submitted to Virtual Conference Proceeding of 12th ECCOMAS Thematic Conference on Multibody Dynamics - Innsbruck July 13-18, 2025 and to the journal of Multibody System Dynamics
>
> **摘要:** Collisions and impacts are the principal reasons for impulsive motions, which we frequently see in dynamic responses of systems. Precise modelling of impacts is a challenging problem due to the lack of the accurate and commonly accepted constitutive law that governs their mechanics. Rigid-body approach and soft contact methods are discussed in this paper and examined in the presented numerical examples. The main focus is set to impacts in systems with multiple unilateral contacts and collisions with elastic elements of the reference. Parameters of interconnecting unilateral springs are under discussion.
>
---
#### [new 027] Cybernetic Marionette: Channeling Collective Agency Through a Wearable Robot in a Live Dancer-Robot Duet
- **分类: cs.HC; cs.RO**

- **简介: 该论文描述DANCE^2项目，探讨观众通过投票影响舞蹈机器人表演的互动机制。任务是研究集体代理在人机协作中的体现，解决如何设计交互以平衡感知与实际控制的问题。**

- **链接: [http://arxiv.org/pdf/2506.10079v1](http://arxiv.org/pdf/2506.10079v1)**

> **作者:** Anup Sathya; Jiasheng Li; Zeyu Yan; Adriane Fang; Bill Kules; Jonathan David Martin; Huaishu Peng
>
> **摘要:** We describe DANCE^2, an interactive dance performance in which audience members channel their collective agency into a dancer-robot duet by voting on the behavior of a wearable robot affixed to the dancer's body. At key moments during the performance, the audience is invited to either continue the choreography or override it, shaping the unfolding interaction through real-time collective input. While post-performance surveys revealed that participants felt their choices meaningfully influenced the performance, voting data across four public performances exhibited strikingly consistent patterns. This tension between what audience members do, what they feel, and what actually changes highlights a complex interplay between agentive behavior, the experience of agency, and power. We reflect on how choreography, interaction design, and the structure of the performance mediate this relationship, offering a live analogy for algorithmically curated digital systems where agency is felt, but not exercised.
>
---
#### [new 028] EQ-TAA: Equivariant Traffic Accident Anticipation via Diffusion-Based Accident Video Synthesis
- **分类: cs.MM; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于交通事故预测任务，旨在解决数据偏差和背景混淆问题。提出AVD模型生成事故视频，结合EQ-TAA方法提升预测性能。**

- **链接: [http://arxiv.org/pdf/2506.10002v1](http://arxiv.org/pdf/2506.10002v1)**

> **作者:** Jianwu Fang; Lei-Lei Li; Zhedong Zheng; Hongkai Yu; Jianru Xue; Zhengguo Li; Tat-Seng Chua
>
> **备注:** Accepted by IEEE-TMM
>
> **摘要:** Traffic Accident Anticipation (TAA) in traffic scenes is a challenging problem for achieving zero fatalities in the future. Current approaches typically treat TAA as a supervised learning task needing the laborious annotation of accident occurrence duration. However, the inherent long-tailed, uncertain, and fast-evolving nature of traffic scenes has the problem that real causal parts of accidents are difficult to identify and are easily dominated by data bias, resulting in a background confounding issue. Thus, we propose an Attentive Video Diffusion (AVD) model that synthesizes additional accident video clips by generating the causal part in dashcam videos, i.e., from normal clips to accident clips. AVD aims to generate causal video frames based on accident or accident-free text prompts while preserving the style and content of frames for TAA after video generation. This approach can be trained using datasets collected from various driving scenes without any extra annotations. Additionally, AVD facilitates an Equivariant TAA (EQ-TAA) with an equivariant triple loss for an anchor accident-free video clip, along with the generated pair of contrastive pseudo-normal and pseudo-accident clips. Extensive experiments have been conducted to evaluate the performance of AVD and EQ-TAA, and competitive performance compared to state-of-the-art methods has been obtained.
>
---
#### [new 029] Provable Sim-to-Real Transfer via Offline Domain Randomization
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，解决模拟到现实的迁移问题。通过改进域随机化方法，提升策略在真实环境中的表现。**

- **链接: [http://arxiv.org/pdf/2506.10133v1](http://arxiv.org/pdf/2506.10133v1)**

> **作者:** Arnaud Fickinger; Abderrahim Bendahi; Stuart Russell
>
> **摘要:** Reinforcement-learning agents often struggle when deployed from simulation to the real-world. A dominant strategy for reducing the sim-to-real gap is domain randomization (DR) which trains the policy across many simulators produced by sampling dynamics parameters, but standard DR ignores offline data already available from the real system. We study offline domain randomization (ODR), which first fits a distribution over simulator parameters to an offline dataset. While a growing body of empirical work reports substantial gains with algorithms such as DROPO, the theoretical foundations of ODR remain largely unexplored. In this work, we (i) formalize ODR as a maximum-likelihood estimation over a parametric simulator family, (ii) prove consistency of this estimator under mild regularity and identifiability conditions, showing it converges to the true dynamics as the dataset grows, (iii) derive gap bounds demonstrating ODRs sim-to-real error is up to an O(M) factor tighter than uniform DR in the finite-simulator case (and analogous gains in the continuous setting), and (iv) introduce E-DROPO, a new version of DROPO which adds an entropy bonus to prevent variance collapse, yielding broader randomization and more robust zero-shot transfer in practice.
>
---
## 更新

#### [replaced 001] Passivity-Centric Safe Reinforcement Learning for Contact-Rich Robotic Tasks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.00287v2](http://arxiv.org/pdf/2503.00287v2)**

> **作者:** Heng Zhang; Gokhan Solak; Sebastian Hjorth; Arash Ajoudani
>
> **备注:** revision version
>
> **摘要:** Reinforcement learning (RL) has achieved remarkable success in various robotic tasks; however, its deployment in real-world scenarios, particularly in contact-rich environments, often overlooks critical safety and stability aspects. Policies without passivity guarantees can result in system instability, posing risks to robots, their environments, and human operators. In this work, we investigate the limitations of traditional RL policies when deployed in contact-rich tasks and explore the combination of energy-based passive control with safe RL in both training and deployment to answer these challenges. Firstly, we reveal the discovery that standard RL policy does not satisfy stability in contact-rich scenarios. Secondly, we introduce a \textit{passivity-aware} RL policy training with energy-based constraints in our safe RL formulation. Lastly, a passivity filter is exerted on the policy output for \textit{passivity-ensured} control during deployment. We conduct comparative studies on a contact-rich robotic maze exploration task, evaluating the effects of learning passivity-aware policies and the importance of passivity-ensured control. The experiments demonstrate that a passivity-agnostic RL policy easily violates energy constraints in deployment, even though it achieves high task completion in training. The results show that our proposed approach guarantees control stability through passivity filtering and improves the energy efficiency through passivity-aware training. A video of real-world experiments is available as supplementary material. We also release the checkpoint model and offline data for pre-training at \href{https://huggingface.co/Anonymous998/passiveRL/tree/main}{Hugging Face}.
>
---
#### [replaced 002] An energy-efficient learning solution for the Agile Earth Observation Satellite Scheduling Problem
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04803v2](http://arxiv.org/pdf/2503.04803v2)**

> **作者:** Antonio M. Mercado-Martínez; Beatriz Soret; Antonio Jurado-Navas
>
> **备注:** This paper has been accepted for presentation at the IEEE International Conference on Machine Learning for Communication and Networking (ICMLCN) Special Sessions 2025
>
> **摘要:** The Agile Earth Observation Satellite Scheduling Problem (AEOSSP) entails finding the subset of observation targets to be scheduled along the satellite's orbit while meeting operational constraints of time, energy and memory. The problem of deciding what and when to observe is inherently complex, and becomes even more challenging when considering several issues that compromise the quality of the captured images, such as cloud occlusion, atmospheric turbulence, and image resolution. This paper presents a Deep Reinforcement Learning (DRL) approach for addressing the AEOSSP with time-dependent profits, integrating these three factors to optimize the use of energy and memory resources. The proposed method involves a dual decision-making process: selecting the sequence of targets and determining the optimal observation time for each. Our results demonstrate that the proposed algorithm reduces the capture of images that fail to meet quality requirements by > 60% and consequently decreases energy waste from attitude maneuvers by up to 78%, all while maintaining strong observation performance.
>
---
#### [replaced 003] Safety-Ensured Robotic Control Framework for Cutting Task Automation in Endoscopic Submucosal Dissection
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.08214v2](http://arxiv.org/pdf/2503.08214v2)**

> **作者:** Yitaek Kim; Iñigo Iturrate; Christoffer Sloth; Hansoul Kim
>
> **备注:** This article has been accepted for publication in IEEE Access. This is the author's version which has not been fully edited and content may change prior to final publication. Citation information: DOI 10.1109/ACCESS.2025.3578607
>
> **摘要:** There is growing interest in automating surgical tasks using robotic systems, such as endoscopy for treating gastrointestinal (GI) cancer. However, previous studies have primarily focused on detecting and analyzing objects or robots, with limited attention to ensuring safety, which is critical for clinical applications, where accidents can be caused by unsafe robot motions. In this study, we propose a new control framework that can formally ensure the safety of automating the cutting task in endoscopic submucosal dissection (ESD), a representative endoscopic surgical method for the treatment of early GI cancer, by using an endoscopic robot. The proposed framework utilizes Control Barrier Functions (CBFs) to accurately identify the boundaries of individual tumors, even in close proximity within the GI tract, ensuring precise treatment and removal while preserving the surrounding normal tissue. Additionally, by adopting a model-free control scheme, safety assurance is made possible even in endoscopic robotic systems where dynamic modeling is challenging. We demonstrate the proposed framework in a simulation-based experimental environment, where the tumors to be removed are close to each other, and show that the safety constraints are enforced. We show that the model-free CBF-based controlled robot eliminates one tumor completely without damaging it, while not invading another nearby tumor.
>
---
#### [replaced 004] Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20619v2](http://arxiv.org/pdf/2505.20619v2)**

> **作者:** Tianhu Peng; Lingfan Bao; Chengxu Zhou
>
> **摘要:** We present a unified gait-conditioned reinforcement learning framework that enables humanoid robots to perform standing, walking, running, and smooth transitions within a single recurrent policy. A compact reward routing mechanism dynamically activates gait-specific objectives based on a one-hot gait ID, mitigating reward interference and supporting stable multi-gait learning. Human-inspired reward terms promote biomechanically natural motions, such as straight-knee stance and coordinated arm-leg swing, without requiring motion capture data. A structured curriculum progressively introduces gait complexity and expands command space over multiple phases. In simulation, the policy successfully achieves robust standing, walking, running, and gait transitions. On the real Unitree G1 humanoid, we validate standing, walking, and walk-to-stand transitions, demonstrating stable and coordinated locomotion. This work provides a scalable, reference-free solution toward versatile and naturalistic humanoid control across diverse modes and environments.
>
---
#### [replaced 005] Constrained Human-AI Cooperation: An Inclusive Embodied Social Intelligence Challenge
- **分类: cs.AI; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.01796v3](http://arxiv.org/pdf/2411.01796v3)**

> **作者:** Weihua Du; Qiushi Lyu; Jiaming Shan; Zhenting Qi; Hongxin Zhang; Sunli Chen; Andi Peng; Tianmin Shu; Kwonjoon Lee; Behzad Dariush; Chuang Gan
>
> **备注:** NeurIPS 2024 Dataset and Benchmark Track. The first two authors contributed equally. Project Website at https://umass-embodied-agi.github.io/CHAIC/
>
> **摘要:** We introduce Constrained Human-AI Cooperation (CHAIC), an inclusive embodied social intelligence challenge designed to test social perception and cooperation in embodied agents. In CHAIC, the goal is for an embodied agent equipped with egocentric observations to assist a human who may be operating under physical constraints -- e.g., unable to reach high places or confined to a wheelchair -- in performing common household or outdoor tasks as efficiently as possible. To achieve this, a successful helper must: (1) infer the human's intents and constraints by following the human and observing their behaviors (social perception), and (2) make a cooperative plan tailored to the human partner to solve the task as quickly as possible, working together as a team (cooperative planning). To benchmark this challenge, we create four new agents with real physical constraints and eight long-horizon tasks featuring both indoor and outdoor scenes with various constraints, emergency events, and potential risks. We benchmark planning- and learning-based baselines on the challenge and introduce a new method that leverages large language models and behavior modeling. Empirical evaluations demonstrate the effectiveness of our benchmark in enabling systematic assessment of key aspects of machine social intelligence. Our benchmark and code are publicly available at https://github.com/UMass-Embodied-AGI/CHAIC.
>
---
#### [replaced 006] Active inference as a unified model of collision avoidance behavior in human drivers
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.02215v4](http://arxiv.org/pdf/2506.02215v4)**

> **作者:** Julian F. Schumann; Johan Engström; Leif Johnson; Matthew O'Kelly; Joao Messias; Jens Kober; Arkady Zgonnikov
>
> **摘要:** Collision avoidance -- involving a rapid threat detection and quick execution of the appropriate evasive maneuver -- is a critical aspect of driving. However, existing models of human collision avoidance behavior are fragmented, focusing on specific scenarios or only describing certain aspects of the avoidance behavior, such as response times. This paper addresses these gaps by proposing a novel computational cognitive model of human collision avoidance behavior based on active inference. Active inference provides a unified approach to modeling human behavior: the minimization of free energy. Building on prior active inference work, our model incorporates established cognitive mechanisms such as evidence accumulation to simulate human responses in two distinct collision avoidance scenarios: front-to-rear lead vehicle braking and lateral incursion by an oncoming vehicle. We demonstrate that our model explains a wide range of previous empirical findings on human collision avoidance behavior. Specifically, the model closely reproduces both aggregate results from meta-analyses previously reported in the literature and detailed, scenario-specific effects observed in a recent driving simulator study, including response timing, maneuver selection, and execution. Our results highlight the potential of active inference as a unified framework for understanding and modeling human behavior in complex real-life driving tasks.
>
---
#### [replaced 007] Simultaneous Localization and Affordance Prediction of Tasks from Egocentric Video
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2407.13856v2](http://arxiv.org/pdf/2407.13856v2)**

> **作者:** Zachary Chavis; Hyun Soo Park; Stephen J. Guy
>
> **摘要:** Vision-Language Models (VLMs) have shown great success as foundational models for downstream vision and natural language applications in a variety of domains. However, these models are limited to reasoning over objects and actions currently visible on the image plane. We present a spatial extension to the VLM, which leverages spatially-localized egocentric video demonstrations to augment VLMs in two ways -- through understanding spatial task-affordances, i.e. where an agent must be for the task to physically take place, and the localization of that task relative to the egocentric viewer. We show our approach outperforms the baseline of using a VLM to map similarity of a task's description over a set of location-tagged images. Our approach has less error both on predicting where a task may take place and on predicting what tasks are likely to happen at the current location. The resulting representation will enable robots to use egocentric sensing to navigate to, or around, physical regions of interest for novel tasks specified in natural language.
>
---
#### [replaced 008] Robotic Policy Learning via Human-assisted Action Preference Optimization
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07127v2](http://arxiv.org/pdf/2506.07127v2)**

> **作者:** Wenke Xia; Yichu Yang; Hongtao Wu; Xiao Ma; Tao Kong; Di Hu
>
> **摘要:** Establishing a reliable and iteratively refined robotic system is essential for deploying real-world applications. While Vision-Language-Action (VLA) models are widely recognized as the foundation model for such robotic deployment, their dependence on expert demonstrations hinders the crucial capabilities of correction and learning from failures. To mitigate this limitation, we introduce a Human-assisted Action Preference Optimization method named HAPO, designed to correct deployment failures and foster effective adaptation through preference alignment for VLA models. This method begins with a human-robot collaboration framework for reliable failure correction and interaction trajectory collection through human intervention. These human-intervention trajectories are further employed within the action preference optimization process, facilitating VLA models to mitigate failure action occurrences while enhancing corrective action adaptation. Specifically, we propose an adaptive reweighting algorithm to address the issues of irreversible interactions and token probability mismatch when introducing preference optimization into VLA models, facilitating model learning from binary desirability signals derived from interactions. Through combining these modules, our human-assisted action preference optimization method ensures reliable deployment and effective learning from failure for VLA models. The experiments conducted in simulation and real-world scenarios prove superior generalization and robustness of our framework across a variety of manipulation tasks.
>
---
#### [replaced 009] Help or Hindrance: Understanding the Impact of Robot Communication in Action Teams
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08892v2](http://arxiv.org/pdf/2506.08892v2)**

> **作者:** Tauhid Tanjim; Jonathan St. George; Kevin Ching; Angelique Taylor
>
> **备注:** This is the author's original submitted version of the paper accepted to the 2025 IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). \c{opyright} 2025 IEEE. Personal use of this material is permitted. For any other use, please contact IEEE
>
> **摘要:** The human-robot interaction (HRI) field has recognized the importance of enabling robots to interact with teams. Human teams rely on effective communication for successful collaboration in time-sensitive environments. Robots can play a role in enhancing team coordination through real-time assistance. Despite significant progress in human-robot teaming research, there remains an essential gap in how robots can effectively communicate with action teams using multimodal interaction cues in time-sensitive environments. This study addresses this knowledge gap in an experimental in-lab study to investigate how multimodal robot communication in action teams affects workload and human perception of robots. We explore team collaboration in a medical training scenario where a robotic crash cart (RCC) provides verbal and non-verbal cues to help users remember to perform iterative tasks and search for supplies. Our findings show that verbal cues for object search tasks and visual cues for task reminders reduce team workload and increase perceived ease of use and perceived usefulness more effectively than a robot with no feedback. Our work contributes to multimodal interaction research in the HRI field, highlighting the need for more human-robot teaming research to understand best practices for integrating collaborative robots in time-sensitive environments such as in hospitals, search and rescue, and manufacturing applications.
>
---
#### [replaced 010] Tightly Coupled SLAM with Imprecise Architectural Plans
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.01737v3](http://arxiv.org/pdf/2408.01737v3)**

> **作者:** Muhammad Shaheer; Jose Andres Millan-Romera; Hriday Bavle; Marco Giberna; Jose Luis Sanchez-Lopez; Javier Civera; Holger Voos
>
> **摘要:** Robots navigating indoor environments often have access to architectural plans, which can serve as prior knowledge to enhance their localization and mapping capabilities. While some SLAM algorithms leverage these plans for global localization in real-world environments, they typically overlook a critical challenge: the "as-planned" architectural designs frequently deviate from the "as-built" real-world environments. To address this gap, we present a novel algorithm that tightly couples LIDAR-based simultaneous localization and mapping with architectural plans under the presence of deviations. Our method utilizes a multi-layered semantic representation to not only localize the robot, but also to estimate global alignment and structural deviations between "as-planned" and as-built environments in real-time. To validate our approach, we performed experiments in simulated and real datasets demonstrating robustness to structural deviations up to 35 cm and 15 degrees. On average, our method achieves 43% less localization error than baselines in simulated environments, while in real environments, the as-built 3D maps show 7% lower average alignment error
>
---
#### [replaced 011] AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.15298v3](http://arxiv.org/pdf/2505.15298v3)**

> **作者:** Kangan Qian; Sicong Jiang; Yang Zhong; Ziang Luo; Zilin Huang; Tianze Zhu; Kun Jiang; Mengmeng Yang; Zheng Fu; Jinyu Miao; Yining Shi; He Zhe Lim; Li Liu; Tianbao Zhou; Huang Yu; Yifei Hu; Guang Li; Guang Chen; Hao Ye; Lijun Sun; Diange Yang
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Vision-Language Models (VLMs) show promise for autonomous driving, yet their struggle with hallucinations, inefficient reasoning, and limited real-world validation hinders accurate perception and robust step-by-step reasoning. To overcome this, we introduce AgentThink, a pioneering unified framework that, for the first time, integrates Chain-of-Thought (CoT) reasoning with dynamic, agent-style tool invocation for autonomous driving tasks. AgentThink's core innovations include: (i) Structured Data Generation, by establishing an autonomous driving tool library to automatically construct structured, self-verified reasoning data explicitly incorporating tool usage for diverse driving scenarios; (ii) A Two-stage Training Pipeline, employing Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to equip VLMs with the capability for autonomous tool invocation; and (iii) Agent-style Tool-Usage Evaluation, introducing a novel multi-tool assessment protocol to rigorously evaluate the model's tool invocation and utilization. Experiments on the DriveLMM-o1 benchmark demonstrate AgentThink significantly boosts overall reasoning scores by 53.91% and enhances answer accuracy by 33.54%, while markedly improving reasoning quality and consistency. Furthermore, ablation studies and robust zero-shot/few-shot generalization experiments across various benchmarks underscore its powerful capabilities. These findings highlight a promising trajectory for developing trustworthy and tool-aware autonomous driving models.
>
---
#### [replaced 012] MoRE: Mixture of Residual Experts for Humanoid Lifelike Gaits Learning on Complex Terrains
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08840v2](http://arxiv.org/pdf/2506.08840v2)**

> **作者:** Dewei Wang; Xinmiao Wang; Xinzhe Liu; Jiyuan Shi; Yingnan Zhao; Chenjia Bai; Xuelong Li
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Humanoid robots have demonstrated robust locomotion capabilities using Reinforcement Learning (RL)-based approaches. Further, to obtain human-like behaviors, existing methods integrate human motion-tracking or motion prior in the RL framework. However, these methods are limited in flat terrains with proprioception only, restricting their abilities to traverse challenging terrains with human-like gaits. In this work, we propose a novel framework using a mixture of latent residual experts with multi-discriminators to train an RL policy, which is capable of traversing complex terrains in controllable lifelike gaits with exteroception. Our two-stage training pipeline first teaches the policy to traverse complex terrains using a depth camera, and then enables gait-commanded switching between human-like gait patterns. We also design gait rewards to adjust human-like behaviors like robot base height. Simulation and real-world experiments demonstrate that our framework exhibits exceptional performance in traversing complex terrains, and achieves seamless transitions between multiple human-like gait patterns.
>
---
#### [replaced 013] EAST: Environment Aware Safe Tracking using Planning and Control Co-Design
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2310.01363v2](http://arxiv.org/pdf/2310.01363v2)**

> **作者:** Zhichao Li; Yinzhuang Yi; Zhuolin Niu; Nikolay Atanasov
>
> **摘要:** This paper considers the problem of autonomous mobile robot navigation in unknown environments with moving obstacles. We propose a new method to achieve environment-aware safe tracking (EAST) of robot motion plans that integrates an obstacle clearance cost for path planning, a convex reachable set for robot motion prediction, and safety constraints for dynamic obstacle avoidance. EAST adapts the motion of the robot according to the locally sensed environment geometry and dynamics, leading to fast motion in wide open areas and cautious behavior in narrow passages or near moving obstacles. Our control design uses a reference governor, a virtual dynamical system that guides the robot's motion and decouples the path tracking and safety objectives. While reference governor methods have been used for safe tracking control in static environments, our key contribution is an extension to dynamic environments using convex optimization with control barrier function (CBF) constraints. Thus, our work establishes a connection between reference governor techniques and CBF techniques for safe control in dynamic environments. We validate our approach in simulated and real-world environments, featuring complex obstacle configurations and natural dynamic obstacle motion.
>
---
#### [replaced 014] APEX: Action Priors Enable Efficient Exploration for Skill Imitation on Articulated Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.10022v2](http://arxiv.org/pdf/2505.10022v2)**

> **作者:** Shivam Sood; Laukik B Nakhwa; Yuhong Cao; Sun Ge; Guillaume Sartoretti
>
> **摘要:** Learning by imitation provides an effective way for robots to develop well-regulated complex behaviors and directly benefit from natural demonstrations. State-of-the-art imitation learning (IL) approaches typically leverage Adversarial Motion Priors (AMP), which, despite their impressive results, suffer from two key limitations. They are prone to mode collapse, which often leads to overfitting to the simulation environment and thus increased sim-to-real gap, and they struggle to learn diverse behaviors effectively. To overcome these limitations, we introduce APEX (Action Priors enable Efficient eXploration): a simple yet versatile IL framework that integrates demonstrations directly into reinforcement learning (RL), maintaining high exploration while grounding behavior with expert-informed priors. We achieve this through a combination of decaying action priors, which initially bias exploration toward expert demonstrations but gradually allow the policy to explore independently. This is complemented by a multi-critic RL framework that effectively balances stylistic consistency with task performance. Our approach achieves sample-efficient IL and enables the acquisition of diverse skills within a single policy. APEX generalizes to varying velocities and preserves reference-like styles across complex tasks such as navigating rough terrain and climbing stairs, utilizing only flat-terrain kinematic motion data as a prior. We validate our framework through extensive hardware experiments on the Unitree Go2 quadruped. There, APEX yields diverse and agile locomotion gaits, inherent gait transitions, and the highest reported speed for the platform to the best of our knowledge (peak velocity of ~3.3 m/s on hardware). Our results establish APEX as a compelling alternative to existing IL methods, offering better efficiency, adaptability, and real-world performance. https://marmotlab.github.io/APEX/
>
---
#### [replaced 015] Nocturnal eye inspired liquid to gas phase change soft actuator with Laser-Induced-Graphene: enhanced environmental light harvesting and photothermal conversion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.11930v4](http://arxiv.org/pdf/2501.11930v4)**

> **作者:** Maina Sogabe; Youhyun Kim; Hiroki Miyazako; Kenji Kawashima
>
> **备注:** 33pages, 10 figures, journal paper
>
> **摘要:** Robotic systems' mobility is constrained by power sources and wiring. While pneumatic actuators remain tethered to air supplies, we developed a new actuator utilizing light energy. Inspired by nocturnal animals' eyes, we designed a bilayer soft actuator incorporating Laser-Induced Graphene (LIG) on the inner surface of a silicone layer. This design maintains silicone's transparency and flexibility while achieving 54% faster response time compared to conventional actuators through enhanced photothermal conversion.
>
---
#### [replaced 016] PhysNav-DG: A Novel Adaptive Framework for Robust VLM-Sensor Fusion in Navigation Applications
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.01881v2](http://arxiv.org/pdf/2505.01881v2)**

> **作者:** Trisanth Srinivasan; Santosh Patapati
>
> **备注:** 9 pages, 5 figures. CVPRW 2025
>
> **摘要:** Robust navigation in diverse environments and domains requires both accurate state estimation and transparent decision making. We present PhysNav-DG, a novel framework that integrates classical sensor fusion with the semantic power of vision-language models. Our dual-branch architecture predicts navigation actions from multi-sensor inputs while simultaneously generating detailed chain-of-thought explanations. A modified Adaptive Kalman Filter dynamically adjusts its noise parameters based on environmental context. It leverages several streams of raw sensor data along with semantic insights from models such as LLaMA 3.2 11B and BLIP-2. To evaluate our approach, we introduce the MD-NEX Benchmark, a novel multi-domain dataset that unifies indoor navigation, autonomous driving, and social navigation tasks with ground-truth actions and human-validated explanations. Extensive experiments and ablations show that PhysNav-DG improves navigation success rates by over 20% and achieves high efficiency, with explanations that are both highly grounded and clear. This work connects high-level semantic reasoning and geometric planning for safer and more trustworthy autonomous systems.
>
---
#### [replaced 017] Automated Generation of Precedence Graphs in Digital Value Chains for Automotive Production
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.19835v2](http://arxiv.org/pdf/2504.19835v2)**

> **作者:** Cornelius Hake; Christian Friedrich
>
> **备注:** \c{opyright}2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** This study examines the digital value chain in automotive manufacturing, focusing on the identification, software flashing, customization, and commissioning of electronic control units in vehicle networks. A novel precedence graph design is proposed to optimize this process chain using an automated scheduling algorithm, which combines structured data extraction from heterogeneous sources via natural language processing and classification techniques with mixed integer linear programming for efficient graph generation. The results show significant improvements in key metrics. The algorithm reduces the number of production stations equipped with expensive hardware and software to execute digital value chain processes, while also increasing capacity utilization through efficient scheduling and reduced idle time. Task parallelization is optimized, resulting in streamlined workflows and increased throughput. Compared to the traditional scheduling method, the automated approach has reduced preparation time by 50% and reduced scheduling activities, as it now takes two minutes to create the precedence graph. The flexibility of the algorithm's constraints allows for vehicle-specific configurations while maintaining high responsiveness, eliminating backup stations and facilitating the integration of new topologies. Automated scheduling significantly outperforms manual methods in efficiency, functionality, and adaptability.
>
---
