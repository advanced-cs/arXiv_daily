# 机器人 cs.RO

- **最新发布 80 篇**

- **更新 37 篇**

## 最新发布

#### [new 001] Active Semantic Mapping of Horticultural Environments Using Gaussian Splatting
- **分类: cs.RO**

- **简介: 该论文属于农业环境的语义重建任务，旨在解决传统方法效率低、精度不足的问题。通过结合Octomap与3D高斯点云，实现高效精准的场景重建与果实计数。**

- **链接: [https://arxiv.org/pdf/2601.12122v1](https://arxiv.org/pdf/2601.12122v1)**

> **作者:** Jose Cuaran; Naveen K. Upalapati; Girish Chowdhary
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Semantic reconstruction of agricultural scenes plays a vital role in tasks such as phenotyping and yield estimation. However, traditional approaches that rely on manual scanning or fixed camera setups remain a major bottleneck in this process. In this work, we propose an active 3D reconstruction framework for horticultural environments using a mobile manipulator. The proposed system integrates the classical Octomap representation with 3D Gaussian Splatting to enable accurate and efficient target-aware mapping. While a low-resolution Octomap provides probabilistic occupancy information for informative viewpoint selection and collision-free planning, 3D Gaussian Splatting leverages geometric, photometric, and semantic information to optimize a set of 3D Gaussians for high-fidelity scene reconstruction. We further introduce simple yet effective strategies to enhance robustness against segmentation noise and reduce memory consumption. Simulation experiments demonstrate that our method outperforms purely occupancy-based approaches in both runtime efficiency and reconstruction accuracy, enabling precise fruit counting and volume estimation. Compared to a 0.01m-resolution Octomap, our approach achieves an improvement of 6.6% in fruit-level F1 score under noise-free conditions, and up to 28.6% under segmentation noise. Additionally, it achieves a 50% reduction in runtime, highlighting its potential for scalable, real-time semantic reconstruction in agricultural robotics.
>
---
#### [new 002] Visual-Language-Guided Task Planning for Horticultural Robots
- **分类: cs.RO**

- **简介: 该论文属于农业机器人任务规划领域，旨在解决作物监测中高阶推理不足的问题。通过引入视觉语言模型，实现任务规划与动作指令的结合，并构建了基准测试。研究揭示了VLM在长期任务中的局限性。**

- **链接: [https://arxiv.org/pdf/2601.11906v1](https://arxiv.org/pdf/2601.11906v1)**

> **作者:** Jose Cuaran; Kendall Koe; Aditya Potnis; Naveen Kumar Uppalapati; Girish Chowdhary
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Crop monitoring is essential for precision agriculture, but current systems lack high-level reasoning. We introduce a novel, modular framework that uses a Visual Language Model (VLM) to guide robotic task planning, interleaving input queries with action primitives. We contribute a comprehensive benchmark for short- and long-horizon crop monitoring tasks in monoculture and polyculture environments. Our main results show that VLMs perform robustly for short-horizon tasks (comparable to human success), but exhibit significant performance degradation in challenging long-horizon tasks. Critically, the system fails when relying on noisy semantic maps, demonstrating a key limitation in current VLM context grounding for sustained robotic operations. This work offers a deployable framework and critical insights into VLM capabilities and shortcomings for complex agricultural robotics.
>
---
#### [new 003] Enabling High-Curvature Navigation in Eversion Robots through Buckle-Inducing Constrictive Bands
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决eversion机器人在高曲率路径中导航困难的问题。通过引入收缩带降低弯曲刚度，提升其通过能力。**

- **链接: [https://arxiv.org/pdf/2601.12523v1](https://arxiv.org/pdf/2601.12523v1)**

> **作者:** Cem Suulker; Muhie Al Haimus; Thomas Mack; Mohammad Sheikhsofla; Neri Niccolò Dei; Reza Kashef; Hadi Sadati; Federica Barontini; Fanny Ficuciello; Alberto Arezzo; Bruno Siciliano; Sebastien Ourselin; Kaspar Althoefer
>
> **摘要:** Tip-growing eversion robots are renowned for their ability to access remote spaces through narrow passages. However, achieving reliable navigation remains a significant challenge. Existing solutions often rely on artificial muscles integrated into the robot body or active tip-steering mechanisms. While effective, these additions introduce structural complexity and compromise the defining advantages of eversion robots: their inherent softness and compliance. In this paper, we propose a passive approach to reduce bending stiffness by purposefully introducing buckling points along the robot's outer wall. We achieve this by integrating inextensible diameter-reducing circumferential bands at regular intervals along the robot body facilitating forward motion through tortuous, obstacle cluttered paths. Rather than relying on active steering, our approach leverages the robot's natural interaction with the environment, allowing for smooth, compliant navigation. We present a Cosserat rod-based mathematical model to quantify this behavior, capturing the local stiffness reductions caused by the constricting bands and their impact on global bending mechanics. Experimental results demonstrate that these bands reduce the robot's stiffness when bent at the tip by up to 91 percent, enabling consistent traversal of 180 degree bends with a bending radius of as low as 25 mm-notably lower than the 35 mm achievable by standard eversion robots under identical conditions. The feasibility of the proposed method is further demonstrated through a case study in a colon phantom. By significantly improving maneuverability without sacrificing softness or increasing mechanical complexity, this approach expands the applicability of eversion robots in highly curved pathways, whether in relation to pipe inspection or medical procedures such as colonoscopy.
>
---
#### [new 004] BiKC+: Bimanual Hierarchical Imitation with Keypose-Conditioned Coordination-Aware Consistency Policies
- **分类: cs.RO**

- **简介: 该论文属于双臂操作任务，解决多阶段协作难题。提出一种基于关键姿态的协调一致性策略，提升任务成功率和效率。**

- **链接: [https://arxiv.org/pdf/2601.12116v1](https://arxiv.org/pdf/2601.12116v1)**

> **作者:** Hang Xu; Yizhou Chen; Dongjie Yu; Yi Ren; Jia PanI
>
> **备注:** Accepted by IEEE Transactions on Automation Science and Engineering 2025
>
> **摘要:** Robots are essential in industrial manufacturing due to their reliability and efficiency. They excel in performing simple and repetitive unimanual tasks but still face challenges with bimanual manipulation. This difficulty arises from the complexities of coordinating dual arms and handling multi-stage processes. Recent integration of generative models into imitation learning (IL) has made progress in tackling specific challenges. However, few approaches explicitly consider the multi-stage nature of bimanual tasks while also emphasizing the importance of inference speed. In multi-stage tasks, failures or delays at any stage can cascade over time, impacting the success and efficiency of subsequent sub-stages and ultimately hindering overall task performance. In this paper, we propose a novel keypose-conditioned coordination-aware consistency policy tailored for bimanual manipulation. Our framework instantiates hierarchical imitation learning with a high-level keypose predictor and a low-level trajectory generator. The predicted keyposes serve as sub-goals for trajectory generation, indicating targets for individual sub-stages. The trajectory generator is formulated as a consistency model, generating action sequences based on historical observations and predicted keyposes in a single inference step. In particular, we devise an innovative approach for identifying bimanual keyposes, considering both robot-centric action features and task-centric operation styles. Simulation and real-world experiments illustrate that our approach significantly outperforms baseline methods in terms of success rates and operational efficiency. Implementation codes can be found at https://github.com/JoanaHXU/BiKC-plus.
>
---
#### [new 005] Static Is Not Enough: A Comparative Study of VR and SpaceMouse in Static and Dynamic Teleoperation Tasks
- **分类: cs.RO**

- **简介: 该论文研究VR与SpaceMouse在静态和动态遥操作任务中的表现，旨在解决接口选择对数据质量的影响问题。通过实验对比两种设备的性能与用户体验。**

- **链接: [https://arxiv.org/pdf/2601.13042v1](https://arxiv.org/pdf/2601.13042v1)**

> **作者:** Yijun Zhou; Muhan Hou; Kim Baraka
>
> **备注:** 5 pages, 5 figures. Accepted in HRI'26 (Late-Breaking Reports track) in 12 Jan, 2026
>
> **摘要:** Imitation learning relies on high-quality demonstrations, and teleoperation is a primary way to collect them, making teleoperation interface choice crucial for the data. Prior work mainly focused on static tasks, i.e., discrete, segmented motions, yet demonstrations also include dynamic tasks requiring reactive control. As dynamic tasks impose fundamentally different interface demands, insights from static-task evaluations cannot generalize. To address this gap, we conduct a within-subjects study comparing a VR controller and a SpaceMouse across two static and two dynamic tasks ($N=25$). We assess success rate, task duration, cumulative success, alongside NASA-TLX, SUS, and open-ended feedback. Results show statistically significant advantages for VR: higher success rates, particularly on dynamic tasks, shorter successful execution times across tasks, and earlier successes across attempts, with significantly lower workload and higher usability. As existing VR teleoperation systems are rarely open-source or suited for dynamic tasks, we release our VR interface to fill this gap.
>
---
#### [new 006] Dynamic Hand Gesture Recognition for Robot Manipulator Tasks
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于人机交互任务，旨在解决动态手部手势识别问题。通过无监督模型实现机器人操作任务手势的实时准确识别。**

- **链接: [https://arxiv.org/pdf/2601.12918v1](https://arxiv.org/pdf/2601.12918v1)**

> **作者:** Dharmendra Sharma; Peeyush Thakur; Sandeep Gupta; Narendra Kumar Dhar; Laxmidhar Behera
>
> **摘要:** This paper proposes a novel approach to recognizing dynamic hand gestures facilitating seamless interaction between humans and robots. Here, each robot manipulator task is assigned a specific gesture. There may be several such tasks, hence, several gestures. These gestures may be prone to several dynamic variations. All such variations for different gestures shown to the robot are accurately recognized in real-time using the proposed unsupervised model based on the Gaussian Mixture model. The accuracy during training and real-time testing prove the efficacy of this methodology.
>
---
#### [new 007] Learning Legged MPC with Smooth Neural Surrogates
- **分类: cs.RO**

- **简介: 该论文属于腿式机器人控制任务，解决学习模型与在线规划集成的问题。通过引入平滑神经代理和重尾似然训练，提升模型的可靠性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.12169v1](https://arxiv.org/pdf/2601.12169v1)**

> **作者:** Samuel A. Moore; Easop Lee; Boyuan Chen
>
> **摘要:** Deep learning and model predictive control (MPC) can play complementary roles in legged robotics. However, integrating learned models with online planning remains challenging. When dynamics are learned with neural networks, three key difficulties arise: (1) stiff transitions from contact events may be inherited from the data; (2) additional non-physical local nonsmoothness can occur; and (3) training datasets can induce non-Gaussian model errors due to rapid state changes. We address (1) and (2) by introducing the smooth neural surrogate, a neural network with tunable smoothness designed to provide informative predictions and derivatives for trajectory optimization through contact. To address (3), we train these models using a heavy-tailed likelihood that better matches the empirical error distributions observed in legged-robot dynamics. Together, these design choices substantially improve the reliability, scalability, and generalizability of learned legged MPC. Across zero-shot locomotion tasks of increasing difficulty, smooth neural surrogates with robust learning yield consistent reductions in cumulative cost on simple, well-conditioned behaviors (typically 10-50%), while providing substantially larger gains in regimes where standard neural dynamics often fail outright. In these regimes, smoothing enables reliable execution (from 0/5 to 5/5 success) and produces about 2-50x lower cumulative cost, reflecting orders-of-magnitude absolute improvements in robustness rather than incremental performance gains.
>
---
#### [new 008] A General One-Shot Multimodal Active Perception Framework for Robotic Manipulation: Learning to Predict Optimal Viewpoint
- **分类: cs.RO**

- **简介: 该论文属于机器人操作中的主动感知任务，旨在解决传统方法依赖迭代优化和任务耦合的问题。提出了一种一次性多模态框架，直接预测最优视角，提升抓取成功率。**

- **链接: [https://arxiv.org/pdf/2601.13639v1](https://arxiv.org/pdf/2601.13639v1)**

> **作者:** Deyun Qin; Zezhi Liu; Hanqian Luo; Xiao Liang; Yongchun Fang
>
> **摘要:** Active perception in vision-based robotic manipulation aims to move the camera toward more informative observation viewpoints, thereby providing high-quality perceptual inputs for downstream tasks. Most existing active perception methods rely on iterative optimization, leading to high time and motion costs, and are tightly coupled with task-specific objectives, which limits their transferability. In this paper, we propose a general one-shot multimodal active perception framework for robotic manipulation. The framework enables direct inference of optimal viewpoints and comprises a data collection pipeline and an optimal viewpoint prediction network. Specifically, the framework decouples viewpoint quality evaluation from the overall architecture, supporting heterogeneous task requirements. Optimal viewpoints are defined through systematic sampling and evaluation of candidate viewpoints, after which large-scale training datasets are constructed via domain randomization. Moreover, a multimodal optimal viewpoint prediction network is developed, leveraging cross-attention to align and fuse multimodal features and directly predict camera pose adjustments. The proposed framework is instantiated in robotic grasping under viewpoint-constrained environments. Experimental results demonstrate that active perception guided by the framework significantly improves grasp success rates. Notably, real-world evaluations achieve nearly double the grasp success rate and enable seamless sim-to-real transfer without additional fine-tuning, demonstrating the effectiveness of the proposed framework.
>
---
#### [new 009] Active Informative Planning for UAV-based Weed Mapping using Discrete Gaussian Process Representations
- **分类: cs.RO**

- **简介: 该论文属于农业无人机路径规划任务，旨在提升 Weed Mapping 的效率与准确性。通过研究不同离散化方法对 GP 映射的影响，优化 UAV 采样路径。**

- **链接: [https://arxiv.org/pdf/2601.13196v1](https://arxiv.org/pdf/2601.13196v1)**

> **作者:** Jacob Swindell; Marija Popović; Riccardo Polvara
>
> **摘要:** Accurate agricultural weed mapping using unmanned aerial vehicles (UAVs) is crucial for precision farming. While traditional methods rely on rigid, pre-defined flight paths and intensive offline processing, informative path planning (IPP) offers a way to collect data adaptively where it is most needed. Gaussian process (GP) mapping provides a continuous model of weed distribution with built-in uncertainty. However, GPs must be discretised for practical use in autonomous planning. Many discretisation techniques exist, but the impact of discrete representation choice remains poorly understood. This paper investigates how different discrete GP representations influence both mapping quality and mission-level performance in UAV-based weed mapping. Considering a UAV equipped with a downward-facing camera, we implement a receding-horizon IPP strategy that selects sampling locations based on the map uncertainty, travel cost, and coverage penalties. We investigate multiple discretisation strategies for representing the GP posterior and use their induced map partitions to generate candidate viewpoints for planning. Experiments on real-world weed distributions show that representation choice significantly affects exploration behaviour and efficiency. Overall, our results demonstrate that discretisation is not only a representational detail but a key design choice that shapes planning dynamics, coverage efficiency, and computational load in online UAV weed mapping.
>
---
#### [new 010] PlannerRFT: Reinforcing Diffusion Planners through Closed-Loop and Sample-Efficient Fine-Tuning
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶轨迹生成任务，解决扩散模型在强化微调中生成多样性不足的问题。提出PlannerRFT框架，提升轨迹多样性和适应性。**

- **链接: [https://arxiv.org/pdf/2601.12901v1](https://arxiv.org/pdf/2601.12901v1)**

> **作者:** Hongchen Li; Tianyu Li; Jiazhi Yang; Haochen Tian; Caojun Wang; Lei Shi; Mingyang Shang; Zengrong Lin; Gaoqiang Wu; Zhihui Hao; Xianpeng Lang; Jia Hu; Hongyang Li
>
> **摘要:** Diffusion-based planners have emerged as a promising approach for human-like trajectory generation in autonomous driving. Recent works incorporate reinforcement fine-tuning to enhance the robustness of diffusion planners through reward-oriented optimization in a generation-evaluation loop. However, they struggle to generate multi-modal, scenario-adaptive trajectories, hindering the exploitation efficiency of informative rewards during fine-tuning. To resolve this, we propose PlannerRFT, a sample-efficient reinforcement fine-tuning framework for diffusion-based planners. PlannerRFT adopts a dual-branch optimization that simultaneously refines the trajectory distribution and adaptively guides the denoising process toward more promising exploration, without altering the original inference pipeline. To support parallel learning at scale, we develop nuMax, an optimized simulator that achieves 10 times faster rollout compared to native nuPlan. Extensive experiments shows that PlannerRFT yields state-of-the-art performance with distinct behaviors emerging during the learning process.
>
---
#### [new 011] VR$^2$: A Co-Located Dual-Headset Platform for Touch-Enabled Human-Robot Interaction Research
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出VR2VR平台，解决触觉人机交互研究中的成本高、实验难问题，通过双头显实现共享空间内的触觉互动与行为模拟。**

- **链接: [https://arxiv.org/pdf/2601.12395v1](https://arxiv.org/pdf/2601.12395v1)**

> **作者:** Chao Wang; Anna Belardinelli; Michael Gienger
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Touch-rich human-robot interaction (HRI) is difficult to study: building and programming physical robots is costly and slow, while VR-based robot prototypes often remove physical contact or break the tight coupling between an agent's body and the user's felt touch. We present VR2VR, a co-located dual VR-headset platform for HRI research in which a participant and a hidden operator share the same physical space while experiencing different virtual embodiments. The participant sees an expressive virtual robot that interacts face-to-face in a shared virtual environment. In real time, the robot's upper-body gestures, head and gaze behaviors, and facial expressions are mapped from the operator's tracked motion and face signals. Because the operator is physically co-present and calibrated into the same coordinate frame, the operator can also physically touch the participant, enabling the participant to perceive robot touch aligned with the robot's hands; finger and hand motion are mapped to the robot using inverse kinematics to support precise contact. Beyond faithful motion retargeting for limb teleoperation, our VR2VR system supports experimental control by retargeting or selectively enabling nonverbal channels (e.g., head only vs. head+eyes vs. head+eyes+facial expressions) while keeping physical interaction constant. We detail the system design, calibration workflow, and safety considerations, and demonstrate the platform through a touch-based Wizard-of-Oz HRI study, illustrating how VR2VR lowers barriers for rapidly prototyping and rigorously evaluating embodied, touch-centric robot behaviors.
>
---
#### [new 012] Model selection and real-time skill assessment for suturing in robotic surgery
- **分类: cs.RO**

- **简介: 该论文属于手术技能评估任务，旨在实时预测外科医生技能水平。通过多模态深度学习模型融合运动和视觉数据，提升评估准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.12012v1](https://arxiv.org/pdf/2601.12012v1)**

> **作者:** Zhaoyang Jacopo Hu; Alex Ranne; Alaa Eldin Abdelaal; Kiran Bhattacharyya; Etienne Burdet; Allison M. Okamura; Ferdinando Rodriguez y Baena
>
> **摘要:** Automated feedback systems have the potential to provide objective skill assessment for training and evaluation in robot-assisted surgery. In this study, we examine methods to achieve real-time prediction of surgical skill level in real-time based on Objective Structured Assessment of Technical Skills (OSATS) scores. Using data acquired from the da Vinci Surgical System, we carry out three main analyses, focusing on model design, their real-time performance, and their skill-level-based cross-validation training. For the model design, we evaluate the effectiveness of multimodal deep learning models for predicting surgical skill levels using synchronized kinematic and vision data. Our models include separate unimodal baselines and fusion architectures that integrate features from both modalities and are evaluated using mean Spearman's correlation coefficients, demonstrating that the fusion model consistently outperforms unimodal models for real-time predictions. For the real-time performance, we observe the prediction's trend over time and highlight correlation with the surgeon's gestures. For the skill-level-based cross-validation, we separately trained models on surgeons with different skill levels, which showed that high-skill demonstrations allow for better performance than those trained on low-skilled ones and generalize well to similarly skilled participants. Our findings show that multimodal learning allows more stable fine-grained evaluation of surgical performance and highlights the value of expert-level training data for model generalization.
>
---
#### [new 013] FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出FRoM-W1框架，解决人形机器人通过自然语言指令实现全身运动控制的问题。通过语言生成和运动控制两阶段方法，提升机器人动作的通用性和稳定性。**

- **链接: [https://arxiv.org/pdf/2601.12799v1](https://arxiv.org/pdf/2601.12799v1)**

> **作者:** Peng Li; Zihan Zhuang; Yangfan Gao; Yi Dong; Sixian Li; Changhao Jiang; Shihan Dou; Zhiheng Xi; Enyu Zhou; Jixuan Huang; Hui Li; Jingjing Gong; Xingjun Ma; Tao Gui; Zuxuan Wu; Qi Zhang; Xuanjing Huang; Yu-Gang Jiang; Xipeng Qiu
>
> **备注:** Project Page: https://openmoss.github.io/FRoM-W1
>
> **摘要:** Humanoid robots are capable of performing various actions such as greeting, dancing and even backflipping. However, these motions are often hard-coded or specifically trained, which limits their versatility. In this work, we present FRoM-W1, an open-source framework designed to achieve general humanoid whole-body motion control using natural language. To universally understand natural language and generate corresponding motions, as well as enable various humanoid robots to stably execute these motions in the physical world under gravity, FRoM-W1 operates in two stages: (a) H-GPT: utilizing massive human data, a large-scale language-driven human whole-body motion generation model is trained to generate diverse natural behaviors. We further leverage the Chain-of-Thought technique to improve the model's generalization in instruction understanding. (b) H-ACT: After retargeting generated human whole-body motions into robot-specific actions, a motion controller that is pretrained and further fine-tuned through reinforcement learning in physical simulation enables humanoid robots to accurately and stably perform corresponding actions. It is then deployed on real robots via a modular simulation-to-reality module. We extensively evaluate FRoM-W1 on Unitree H1 and G1 robots. Results demonstrate superior performance on the HumanML3D-X benchmark for human whole-body motion generation, and our introduced reinforcement learning fine-tuning consistently improves both motion tracking accuracy and task success rates of these humanoid robots. We open-source the entire FRoM-W1 framework and hope it will advance the development of humanoid intelligence.
>
---
#### [new 014] AI for Green Spaces: Leveraging Autonomous Navigation and Computer Vision for Park Litter Removal
- **分类: cs.RO; cs.AI; cs.CV; eess.SY**

- **简介: 该论文属于自主导航与计算机视觉任务，旨在解决公园草地垃圾清理问题。通过STC算法、RTK GPS和ResNet50模型实现自主导航与垃圾识别，设计新型拾取装置，成功率达80%。**

- **链接: [https://arxiv.org/pdf/2601.11876v1](https://arxiv.org/pdf/2601.11876v1)**

> **作者:** Christopher Kao; Akhil Pathapati; James Davis
>
> **备注:** Published in IEEE/SICE SII 2025
>
> **摘要:** There are 50 billion pieces of litter in the U.S. alone. Grass fields contribute to this problem because picnickers tend to leave trash on the field. We propose building a robot that can autonomously navigate, identify, and pick up trash in parks. To autonomously navigate the park, we used a Spanning Tree Coverage (STC) algorithm to generate a coverage path the robot could follow. To navigate this path, we successfully used Real-Time Kinematic (RTK) GPS, which provides a centimeter-level reading every second. For computer vision, we utilized the ResNet50 Convolutional Neural Network (CNN), which detects trash with 94.52% accuracy. For trash pickup, we tested multiple design concepts. We select a new pickup mechanism that specifically targets the trash we encounter on the field. Our solution achieved an overall success rate of 80%, demonstrating that autonomous trash pickup robots on grass fields are a viable solution.
>
---
#### [new 015] AirHunt: Bridging VLM Semantics and Continuous Planning for Efficient Aerial Object Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AirHunt系统，解决无人机在户外环境中基于自然语言指令高效导航的问题。通过融合视觉-语言模型与连续路径规划，提升导航成功率与效率。**

- **链接: [https://arxiv.org/pdf/2601.12742v1](https://arxiv.org/pdf/2601.12742v1)**

> **作者:** Xuecheng Chen; Zongzhuo Liu; Jianfa Ma; Bang Du; Tiantian Zhang; Xueqian Wang; Boyu Zhou
>
> **摘要:** Recent advances in large Vision-Language Models (VLMs) have provided rich semantic understanding that empowers drones to search for open-set objects via natural language instructions. However, prior systems struggle to integrate VLMs into practical aerial systems due to orders-of-magnitude frequency mismatch between VLM inference and real-time planning, as well as VLMs' limited 3D scene understanding. They also lack a unified mechanism to balance semantic guidance with motion efficiency in large-scale environments. To address these challenges, we present AirHunt, an aerial object navigation system that efficiently locates open-set objects with zero-shot generalization in outdoor environments by seamlessly fusing VLM semantic reasoning with continuous path planning. AirHunt features a dual-pathway asynchronous architecture that establishes a synergistic interface between VLM reasoning and path planning, enabling continuous flight with adaptive semantic guidance that evolves through motion. Moreover, we propose an active dual-task reasoning module that exploits geometric and semantic redundancy to enable selective VLM querying, and a semantic-geometric coherent planning module that dynamically reconciles semantic priorities and motion efficiency in a unified framework, enabling seamless adaptation to environmental heterogeneity. We evaluate AirHunt across diverse object navigation tasks and environments, demonstrating a higher success rate with lower navigation error and reduced flight time compared to state-of-the-art methods. Real-world experiments further validate AirHunt's practical capability in complex and challenging environments. Code and dataset will be made publicly available before publication.
>
---
#### [new 016] Optimal Thruster Configuration for 6-DOF Control of a Small Satellite
- **分类: cs.RO; eess.SY**

- **简介: 论文研究小卫星六自由度控制的最优推进器配置问题，旨在通过减少推进器数量实现有效姿态与轨道控制，提升任务灵活性。**

- **链接: [https://arxiv.org/pdf/2601.11802v1](https://arxiv.org/pdf/2601.11802v1)**

> **作者:** Suguru Sato; Jinaykumar Patel; Kamesh Subbarao
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** With the growing deployment of small satellites (such as CubeSats, Nanosats, Picosats, and Femtosats) in Low Earth Orbit (LEO) for targeted applications like imaging, communication, data storage, and rendezvous-docking mission, there is increasing attention on orbit maintenance and attitude control. A common approach for active orbit control involves the use of multiple thrusters, which, when properly arranged, can also generate the required torque for attitude control. Starting from a 24-thruster configuration, this paper presents a set of thruster configurations (referred to as a viable configuration group) that enable full six degrees of freedom (6-DOF) control. Further, configuration group that requires minimum total thrust to achieve 6-DOF commands are found among the viable configuration group. One configuration from each of these groups is further evaluated for its attitude control performance through a representative rendezvous-docking mission, demonstrating that even with a reduced thruster count, sufficient maneuverability can be achieved.
>
---
#### [new 017] LogicEnvGen: Task-Logic Driven Generation of Diverse Simulated Environments for Embodied AI
- **分类: cs.RO**

- **简介: 该论文提出LogicEnvGen，用于生成逻辑多样化的模拟环境，解决传统方法忽视逻辑多样性的问题，提升Agent测试效果。**

- **链接: [https://arxiv.org/pdf/2601.13556v1](https://arxiv.org/pdf/2601.13556v1)**

> **作者:** Jianan Wang; Siyang Zhang; Bin Li; Juan Chen; Jingtao Qi; Zhuo Zhang; Chen Qian
>
> **备注:** 19 pages, 15 figures, 6 tables
>
> **摘要:** Simulated environments play an essential role in embodied AI, functionally analogous to test cases in software engineering. However, existing environment generation methods often emphasize visual realism (e.g., object diversity and layout coherence), overlooking a crucial aspect: logical diversity from the testing perspective. This limits the comprehensive evaluation of agent adaptability and planning robustness in distinct simulated environments. To bridge this gap, we propose LogicEnvGen, a novel method driven by Large Language Models (LLMs) that adopts a top-down paradigm to generate logically diverse simulated environments as test cases for agents. Given an agent task, LogicEnvGen first analyzes its execution logic to construct decision-tree-structured behavior plans and then synthesizes a set of logical trajectories. Subsequently, it adopts a heuristic algorithm to refine the trajectory set, reducing redundant simulation. For each logical trajectory, which represents a potential task situation, LogicEnvGen correspondingly instantiates a concrete environment. Notably, it employs constraint solving for physical plausibility. Furthermore, we introduce LogicEnvEval, a novel benchmark comprising four quantitative metrics for environment evaluation. Experimental results verify the lack of logical diversity in baselines and demonstrate that LogicEnvGen achieves 1.04-2.61x greater diversity, significantly improving the performance in revealing agent faults by 4.00%-68.00%.
>
---
#### [new 018] Active Cross-Modal Visuo-Tactile Perception of Deformable Linear Objects
- **分类: cs.RO**

- **简介: 该论文属于3D形状重建任务，旨在解决视觉受限下柔性线状物体的感知问题。通过融合视觉与触觉信息，实现准确的电缆重构。**

- **链接: [https://arxiv.org/pdf/2601.13979v1](https://arxiv.org/pdf/2601.13979v1)**

> **作者:** Raffaele Mazza; Ciro Natale; Pietro Falco
>
> **摘要:** This paper presents a novel cross-modal visuo-tactile perception framework for the 3D shape reconstruction of deformable linear objects (DLOs), with a specific focus on cables subject to severe visual occlusions. Unlike existing methods relying predominantly on vision, whose performance degrades under varying illumination, background clutter, or partial visibility, the proposed approach integrates foundation-model-based visual perception with adaptive tactile exploration. The visual pipeline exploits SAM for instance segmentation and Florence for semantic refinement, followed by skeletonization, endpoint detection, and point-cloud extraction. Occluded cable segments are autonomously identified and explored with a tactile sensor, which provides local point clouds that are merged with the visual data through Euclidean clustering and topology-preserving fusion. A B-spline interpolation driven by endpoint-guided point sorting yields a smooth and complete reconstruction of the cable shape. Experimental validation using a robotic manipulator equipped with an RGB-D camera and a tactile pad demonstrates that the proposed framework accurately reconstructs both simple and highly curved single or multiple cable configurations, even when large portions are occluded. These results highlight the potential of foundation-model-enhanced cross-modal perception for advancing robotic manipulation of deformable objects.
>
---
#### [new 019] SandWorm: Event-based Visuotactile Perception with Active Vibration for Screw-Actuated Robot in Granular Media
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，解决在颗粒介质中感知与运动难题。提出SandWorm机器人和SWTac传感器，实现高精度触觉与视觉感知。**

- **链接: [https://arxiv.org/pdf/2601.14128v1](https://arxiv.org/pdf/2601.14128v1)**

> **作者:** Shoujie Li; Changqing Guo; Junhao Gong; Chenxin Liang; Wenhua Ding; Wenbo Ding
>
> **备注:** Accepted by IEEE Transactions on Robotics
>
> **摘要:** Perception in granular media remains challenging due to unpredictable particle dynamics. To address this challenge, we present SandWorm, a biomimetic screw-actuated robot augmented by peristaltic motion to enhance locomotion, and SWTac, a novel event-based visuotactile sensor with an actively vibrated elastomer. The event camera is mechanically decoupled from vibrations by a spring isolation mechanism, enabling high-quality tactile imaging of both dynamic and stationary objects. For algorithm design, we propose an IMU-guided temporal filter to enhance imaging consistency, improving MSNR by 24%. Moreover, we systematically optimize SWTac with vibration parameters, event camera settings and elastomer properties. Motivated by asymmetric edge features, we also implement contact surface estimation by U-Net. Experimental validation demonstrates SWTac's 0.2 mm texture resolution, 98% stone classification accuracy, and 0.15 N force estimation error, while SandWorm demonstrates versatile locomotion (up to 12.5 mm/s) in challenging terrains, successfully executes pipeline dredging and subsurface exploration in complex granular media (observed 90% success rate). Field experiments further confirm the system's practical performance.
>
---
#### [new 020] An Efficient and Multi-Modal Navigation System with One-Step World Model
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决学习方法在3D空间推理和物理动态理解上的不足。提出一种轻量级世界模型，采用单步生成和3D U-Net结构，提升导航效率与性能。**

- **链接: [https://arxiv.org/pdf/2601.12277v1](https://arxiv.org/pdf/2601.12277v1)**

> **作者:** Wangtian Shen; Ziyang Meng; Jinming Ma; Mingliang Zhou; Diyun Xiang
>
> **摘要:** Navigation is a fundamental capability for mobile robots. While the current trend is to use learning-based approaches to replace traditional geometry-based methods, existing end-to-end learning-based policies often struggle with 3D spatial reasoning and lack a comprehensive understanding of physical world dynamics. Integrating world models-which predict future observations conditioned on given actions-with iterative optimization planning offers a promising solution due to their capacity for imagination and flexibility. However, current navigation world models, typically built on pure transformer architectures, often rely on multi-step diffusion processes and autoregressive frame-by-frame generation. These mechanisms result in prohibitive computational latency, rendering real-time deployment impossible. To address this bottleneck, we propose a lightweight navigation world model that adopts a one-step generation paradigm and a 3D U-Net backbone equipped with efficient spatial-temporal attention. This design drastically reduces inference latency, enabling high-frequency control while achieving superior predictive performance. We also integrate this model into an optimization-based planning framework utilizing anchor-based initialization to handle multi-modal goal navigation tasks. Extensive closed-loop experiments in both simulation and real-world environments demonstrate our system's superior efficiency and robustness compared to state-of-the-art baselines.
>
---
#### [new 021] MATTERIX: toward a digital twin for robotics-assisted chemistry laboratory automation
- **分类: cs.RO**

- **简介: 该论文属于机器人化学实验室自动化任务，旨在解决实验流程开发成本高、效率低的问题。通过构建高保真数字孪生框架MATTERIX，实现虚拟仿真与真实环境的高效迁移。**

- **链接: [https://arxiv.org/pdf/2601.13232v1](https://arxiv.org/pdf/2601.13232v1)**

> **作者:** Kourosh Darvish; Arjun Sohal; Abhijoy Mandal; Hatem Fakhruldeen; Nikola Radulov; Zhengxue Zhou; Satheeshkumar Veeramani; Joshua Choi; Sijie Han; Brayden Zhang; Jeeyeoun Chae; Alex Wright; Yijie Wang; Hossein Darvish; Yuchi Zhao; Gary Tom; Han Hao; Miroslav Bogdanovic; Gabriella Pizzuto; Andrew I. Cooper; Alán Aspuru-Guzik; Florian Shkurti; Animesh Garg
>
> **备注:** Darvish, K., Sohal, A., Mandal, A. et al. MATTERIX: toward a digital twin for robotics-assisted chemistry laboratory automation. Nat Comput Sci (2025)
>
> **摘要:** Accelerated materials discovery is critical for addressing global challenges. However, developing new laboratory workflows relies heavily on real-world experimental trials, and this can hinder scalability because of the need for numerous physical make-and-test iterations. Here we present MATTERIX, a multiscale, graphics processing unit-accelerated robotic simulation framework designed to create high-fidelity digital twins of chemistry laboratories, thus accelerating workflow development. This multiscale digital twin simulates robotic physical manipulation, powder and liquid dynamics, device functionalities, heat transfer and basic chemical reaction kinetics. This is enabled by integrating realistic physics simulation and photorealistic rendering with a modular graphics processing unit-accelerated semantics engine, which models logical states and continuous behaviors to simulate chemistry workflows across different levels of abstraction. MATTERIX streamlines the creation of digital twin environments through open-source asset libraries and interfaces, while enabling flexible workflow design via hierarchical plan definition and a modular skill library that incorporates learning-based methods. Our approach demonstrates sim-to-real transfer in robotic chemistry setups, reducing reliance on costly real-world experiments and enabling the testing of hypothetical automated workflows in silico. The project website is available at https://accelerationconsortium.github.io/Matterix/ .
>
---
#### [new 022] DroneVLA: VLA based Aerial Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于航空操作任务，旨在解决非专家用户自然指令控制无人机的问题。提出DroneVLA系统，结合视觉语言模型与导航算法，实现物体抓取与安全交接。**

- **链接: [https://arxiv.org/pdf/2601.13809v1](https://arxiv.org/pdf/2601.13809v1)**

> **作者:** Fawad Mehboob; Monijesu James; Amir Habel; Jeffrin Sam; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **备注:** This paper has been accepted for publication at LBR of HRI 2026 conference
>
> **摘要:** As aerial platforms evolve from passive observers to active manipulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally. This work introduces a novel concept of autonomous aerial manipulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user. The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom-built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera. VLA performs semantic reasoning to interpret the intent of a user prompt and generates a prioritized task queue for grasping of relevant objects in the scene. Grounding DINO and dynamic A* planning algorithm are used to navigate and safely relocate the object. To ensure safe and natural interaction during the handover phase, the system employs a human-centric controller driven by MediaPipe. This module provides real-time human pose estimation, allowing the drone to employ visual servoing to maintain a stable, distinct position directly in front of the user, facilitating a comfortable handover. We demonstrate the system's efficacy through real-world experiments for localization and navigation, which resulted in a 0.164m, 0.070m, and 0.084m of max, mean euclidean, and root-mean squared errors, respectively, highlighting the feasibility of VLA for aerial manipulation operations.
>
---
#### [new 023] Three Dimensional Hydrodynamic Flow-Based Collision Avoidance for UAV Formations Facing Emergent Dynamic Obstacles
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机编队避障任务，解决动态障碍物下的实时避障问题。通过流体力学方法和虚拟刚体策略，实现安全、平滑的编队避让。**

- **链接: [https://arxiv.org/pdf/2601.11832v1](https://arxiv.org/pdf/2601.11832v1)**

> **作者:** Suguru Sato; Kamesh Subbarao
>
> **备注:** 18 pages, 15 figures
>
> **摘要:** This paper presents a three-dimensional, hydrodynamics-inspired collision avoidance framework for uncrewed aerial vehicle (UAV) formations operating in dynamic environments. When moving obstacles enter a UAV's sensing region, they are modeled as three dimensional doublets or ellipsoids that generate local velocity fields, guiding nearby UAVs to execute smooth, collision-free maneuvers without trajectory discontinuities or explicit trajectory replanning. This flow-based approach enables real-time operation and interpretable behavior by leveraging the nature of fluid flow around obstacles via the harmonic properties of Laplace's equation, inherently avoiding local minima common in traditional potential field methods. To establish and maintain coordination among the UAVs, a Virtual Rigid Body (VRB) formation strategy is integrated, ensuring that formation geometry and trajectory tracking are preserved. Simulation results demonstrate the feasibility and scalability of the method for both individual and multi-UAV scenarios with multiple formation geometries encountering moving obstacles. The proposed approach achieves safe, smooth, and computationally efficient avoidance maneuvers suitable for real-time and practical applications.
>
---
#### [new 024] Helical Tendon-Driven Continuum Robot with Programmable Follow-the-Leader Operation
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决脊髓刺激电极精准定位问题。通过设计一种具有跟随运动功能的连续体机器人，实现对脊髓目标区域的精确导航。**

- **链接: [https://arxiv.org/pdf/2601.13177v1](https://arxiv.org/pdf/2601.13177v1)**

> **作者:** Behnam Moradkhani; Raghav Sankaranarayanan; Pejman Kheradmand; Harshith Jella; Nicholas Ahn; Ajmal Zemmar; Yash Chitalia
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Spinal cord stimulation (SCS) is primarily utilized for pain management and has recently demonstrated efficacy in promoting functional recovery in patients with spinal cord injury. Effective stimulation of motor neurons ideally requires the placement of SCS leads in the ventral or lateral epidural space where the corticospinal and rubrospinal motor fibers are located. This poses significant challenges with the current standard of manual steering. In this study, we present a static modeling approach for the ExoNav, a steerable robotic tool designed to facilitate precise navigation to the ventral and lateral epidural space. Cosserat rod framework is employed to establish the relationship between tendon actuation forces and the robot's overall shape. The effects of gravity, as an example of an external load, are investigated and implemented in the model and simulation. The experimental results indicate RMSE values of 1.76mm, 2.33mm, 2.18mm, and 1.33mm across four tested prototypes. Based on the helical shape of the ExoNav upon actuation, it is capable of performing follow-the-leader (FTL) motion by adding insertion and rotation DoFs to this robotic system, which is shown in simulation and experimentally. The proposed simulation has the capability to calculate optimum tendon tensions to follow the desired FTL paths while gravity-induced robot deformations are present. Three FTL experimental trials are conducted and the end-effector position showed repeatable alignments with the desired path with maximum RMSE value of 3.75mm. Ultimately, a phantom model demonstration is conducted where the teleoperated robot successfully navigated to the lateral and ventral spinal cord targets. Additionally, the user was able to navigate to the dorsal root ganglia, illustrating ExoNav's potential in both motor function recovery and pain management.
>
---
#### [new 025] Exploiting Light To Enhance The Endurance and Navigation of Lighter-Than-Air Micro-Drones
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机任务，旨在解决LTA微无人机续航短和导航难的问题。通过光能供能与导航系统，实现可持续自主飞行。**

- **链接: [https://arxiv.org/pdf/2601.13088v1](https://arxiv.org/pdf/2601.13088v1)**

> **作者:** Harry Huang; Talia Xu; Marco Zúñiga Zamalloa
>
> **摘要:** Micro-Unmanned Aerial Vehicles (UAVs) are rapidly expanding into tasks from inventory to environmental sensing, yet their short endurance and unreliable navigation in GPS-denied spaces limit deployment. Lighter-Than-Air (LTA) drones offer an energy-efficient alternative: they use a helium envelope to provide buoyancy, which enables near-zero-power drain during hovering and much longer operation. LTAs are promising, but their design is complex, and they lack integrated solutions to enable sustained autonomous operations and navigation with simple, low-infrastructure. We propose a compact, self-sustaining LTA drone that uses light for both energy harvesting and navigation. Our contributions are threefold: (i) a high-fidelity simulation framework to analyze LTA aerodynamics and select a stable, efficient configuration; (ii) a framework to integrate solar cells on the envelope to provide net-positive energy; and (iii) a point-and-go navigation system with three light-seeking algorithms operating on a single light beacon. Our LTA-analysis, together with the integrated solar panels, not only saves energy while flying, but also enables sustainable operation: providing 1 minute of flying time for every 4 minutes of energy harvesting, under illuminations of 80klux. We also demonstrate robust single-beacon navigation towards a light source that can be up to 7m away, in indoor and outdoor environments, even with moderate winds. The resulting system indicates a plausible path toward persistent, autonomous operation for indoor and outdoor monitoring. More broadly, this work provides a practical pathway for translating the promise of LTA drones into a persistent, self-sustaining aerial system.
>
---
#### [new 026] RPT*: Global Planning with Probabilistic Terminals for Target Search in Complex Environments
- **分类: cs.RO; cs.CG**

- **简介: 该论文研究目标搜索中的HPP-PT问题，即在不确定环境下优化路径以最小化期望成本。提出RPT*算法及HATS系统，解决路径规划与探索的平衡问题。**

- **链接: [https://arxiv.org/pdf/2601.12701v1](https://arxiv.org/pdf/2601.12701v1)**

> **作者:** Yunpeng Lyu; Chao Cao; Ji Zhang; Howie Choset; Zhongqiang Ren
>
> **摘要:** Routing problems such as Hamiltonian Path Problem (HPP), seeks a path to visit all the vertices in a graph while minimizing the path cost. This paper studies a variant, HPP with Probabilistic Terminals (HPP-PT), where each vertex has a probability representing the likelihood that the robot's path terminates there, and the objective is to minimize the expected path cost. HPP-PT arises in target object search, where a mobile robot must visit all candidate locations to find an object, and prior knowledge of the object's location is expressed as vertex probabilities. While routing problems have been studied for decades, few of them consider uncertainty as required in this work. The challenge lies not only in optimally ordering the vertices, as in standard HPP, but also in handling history dependency: the expected path cost depends on the order in which vertices were previously visited. This makes many existing methods inefficient or inapplicable. To address the challenge, we propose a search-based approach RPT* with solution optimality guarantees, which leverages dynamic programming in a new state space to bypass the history dependency and novel heuristics to speed up the computation. Building on RPT*, we design a Hierarchical Autonomous Target Search (HATS) system that combines RPT* with either Bayesian filtering for lifelong target search with noisy sensors, or autonomous exploration to find targets in unknown environments. Experiments in both simulation and real robot show that our approach can naturally balance between exploitation and exploration, thereby finding targets more quickly on average than baseline methods.
>
---
#### [new 027] Highly Deformable Proprioceptive Membrane for Real-Time 3D Shape Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决复杂环境下3D形状重建问题。提出一种基于光学波导的柔性 proprioceptive 膜，实现高精度实时变形恢复。**

- **链接: [https://arxiv.org/pdf/2601.13574v1](https://arxiv.org/pdf/2601.13574v1)**

> **作者:** Guanyu Xu; Jiaqi Wang; Dezhong Tong; Xiaonan Huang
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Reconstructing the three-dimensional (3D) geometry of object surfaces is essential for robot perception, yet vision-based approaches are generally unreliable under low illumination or occlusion. This limitation motivates the design of a proprioceptive membrane that conforms to the surface of interest and infers 3D geometry by reconstructing its own deformation. Conventional shape-aware membranes typically rely on resistive, capacitive, or magneto-sensitive mechanisms. However, these methods often encounter challenges such as structural complexity, limited compliance during large-scale deformation, and susceptibility to electromagnetic interference. This work presents a soft, flexible, and stretchable proprioceptive silicone membrane based on optical waveguide sensing. The membrane sensor integrates edge-mounted LEDs and centrally distributed photodiodes (PDs), interconnected via liquid-metal traces embedded within a multilayer elastomeric composite. Rich deformation-dependent light intensity signals are decoded by a data-driven model to recover the membrane geometry as a 3D point cloud. On a customized 140 mm square membrane, real-time reconstruction of large-scale out-of-plane deformation is achieved at 90 Hz with an average reconstruction error of 1.3 mm, measured by Chamfer distance, while maintaining accuracy for indentations up to 25 mm. The proposed framework provides a scalable, robust, and low-profile solution for global shape perception in deformable robotic systems.
>
---
#### [new 028] GuideTouch: An Obstacle Avoidance Device for Visually Impaired
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于辅助导航任务，旨在解决视觉障碍者检测头部障碍物的问题。研究开发了GuideTouch设备，通过传感器和触觉反馈实现障碍物避让。**

- **链接: [https://arxiv.org/pdf/2601.13813v1](https://arxiv.org/pdf/2601.13813v1)**

> **作者:** Timofei Kozlov; Artem Trandofilov; Georgii Gazaryan; Issatay Tokmurziyev; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **备注:** This paper has been accepted for publication at LBR of HRI 2026 conference
>
> **摘要:** Safe navigation for the visually impaired individuals remains a critical challenge, especially concerning head-level obstacles, which traditional mobility aids often fail to detect. We introduce GuideTouch, a compact, affordable, standalone wearable device designed for autonomous obstacle avoidance. The system integrates two vertically aligned Time-of-Flight (ToF) sensors, enabling three-dimensional environmental perception, and four vibrotactile actuators that provide directional haptic feedback. Proximity and direction information is communicated via an intuitive 4-point vibrotactile feedback system located across the user's shoulders and upper chest. For real-world robustness, the device includes a unique centrifugal self-cleaning optical cover mechanism and a sound alarm system for location if the device is dropped. We evaluated the haptic perception accuracy across 22 participants (17 male and 5 female, aged 21-48, mean 25.7, sd 6.1). Statistical analysis confirmed a significant difference between the perception accuracy of different patterns. The system demonstrated high recognition accuracy, achieving an average of 92.9% for single and double motor (primary directional) patterns. Furthermore, preliminary experiments with 14 visually impaired users validated this interface, showing a recognition accuracy of 93.75% for primary directional cues. The results demonstrate that GuideTouch enables intuitive spatial perception and could significantly improve the safety, confidence, and autonomy of users with visual impairments during independent navigation.
>
---
#### [new 029] Active Inference-Driven World Modeling for Adaptive UAV Swarm Trajectory Design
- **分类: cs.RO; cs.AI; eess.SP**

- **简介: 该论文属于智能无人机编队控制任务，解决动态环境下的轨迹设计问题。通过主动推理框架，结合概率推理与自学习，实现高效、安全的编队导航。**

- **链接: [https://arxiv.org/pdf/2601.12939v1](https://arxiv.org/pdf/2601.12939v1)**

> **作者:** Kaleem Arshid; Ali Krayani; Lucio Marcenaro; David Martin Gomez; Carlo Regazzoni
>
> **备注:** This paper has been accepted for presentation at the 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (IEEE ICASSP 2026) Workshop: 'Multi-Modal Signal Processing and AI for Communications and Sensing in 6G and Beyond (MuSiC-6GB)'
>
> **摘要:** This paper proposes an Active Inference-based framework for autonomous trajectory design in UAV swarms. The method integrates probabilistic reasoning and self-learning to enable distributed mission allocation, route ordering, and motion planning. Expert trajectories generated using a Genetic Algorithm with Repulsion Forces (GA-RF) are employed to train a hierarchical World Model capturing swarm behavior across mission, route, and motion levels. During online operation, UAVs infer actions by minimizing divergence between current beliefs and model-predicted states, enabling adaptive responses to dynamic environments. Simulation results show faster convergence, higher stability, and safer navigation than Q-Learning, demonstrating the scalability and cognitive grounding of the proposed framework for intelligent UAV swarm control.
>
---
#### [new 030] ForeDiffusion: Foresight-Conditioned Diffusion Policy via Future View Construction for Robot Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决扩散策略在复杂任务中成功率低的问题。通过引入未来视图引导扩散过程，提出ForeDiffusion方法，提升抓取准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.12925v1](https://arxiv.org/pdf/2601.12925v1)**

> **作者:** Weize Xie; Yi Ding; Ying He; Leilei Wang; Binwen Bai; Zheyi Zhao; Chenyang Wang; F. Richard Yu
>
> **摘要:** Diffusion strategies have advanced visual motor control by progressively denoising high-dimensional action sequences, providing a promising method for robot manipulation. However, as task complexity increases, the success rate of existing baseline models decreases considerably. Analysis indicates that current diffusion strategies are confronted with two limitations. First, these strategies only rely on short-term observations as conditions. Second, the training objective remains limited to a single denoising loss, which leads to error accumulation and causes grasping deviations. To address these limitations, this paper proposes Foresight-Conditioned Diffusion (ForeDiffusion), by injecting the predicted future view representation into the diffusion process. As a result, the policy is guided to be forward-looking, enabling it to correct trajectory deviations. Following this design, ForeDiffusion employs a dual loss mechanism, combining the traditional denoising loss and the consistency loss of future observations, to achieve the unified optimization. Extensive evaluation on the Adroit suite and the MetaWorld benchmark demonstrates that ForeDiffusion achieves an average success rate of 80% for the overall task, significantly outperforming the existing mainstream diffusion methods by 23% in complex tasks, while maintaining more stable performance across the entire tasks.
>
---
#### [new 031] Neural Process-Based Reactive Controller for Autonomous Racing
- **分类: cs.RO**

- **简介: 该论文属于自主驾驶控制任务，解决实时安全决策问题。提出基于AttNP的反应式控制器，并引入物理先验提升性能，结合CBF确保碰撞避免。**

- **链接: [https://arxiv.org/pdf/2601.12143v1](https://arxiv.org/pdf/2601.12143v1)**

> **作者:** Devin Hunter; Chinwendu Enyioha
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** Attention-based neural architectures have become central to state-of-the-art methods in real-time nonlinear control. As these data-driven models continue to be integrated into increasingly safety-critical domains, ensuring statistically grounded and provably safe decision-making becomes essential. This paper introduces a novel reactive control framework for gap-based navigation using the Attentive Neural Process (AttNP) and a physics-informed extension, the PI-AttNP. Both models are evaluated in a simulated F1TENTH-style Ackermann steering racecar environment, chosen as a fast-paced proxy for safety-critical autonomous driving scenarios. The PI-AttNP augments the AttNP architecture with approximate model-based priors to inject physical inductive bias, enabling faster convergence and improved prediction accuracy suited for real-time control. To further ensure safety, we derive and implement a control barrier function (CBF)-based filtering mechanism that analytically enforces collision avoidance constraints. This CBF formulation is fully compatible with the learned AttNP controller and generalizes across a wide range of racing scenarios, providing a lightweight and certifiable safety layer. Our results demonstrate competitive closed-loop performance while ensuring real-time constraint satisfaction.
>
---
#### [new 032] LLM-VLM Fusion Framework for Autonomous Maritime Port Inspection using a Heterogeneous UAV-USV System
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主海上港口检测任务，旨在解决传统方法在可扩展性和上下文理解上的不足。通过融合LLM和VLM，实现无人机与无人水面艇的协同自动检测与报告生成。**

- **链接: [https://arxiv.org/pdf/2601.13096v1](https://arxiv.org/pdf/2601.13096v1)**

> **作者:** Muhayy Ud Din; Waseem Akram; Ahsan B. Bakht; Irfan Hussain
>
> **备注:** submitted in AEJ
>
> **摘要:** Maritime port inspection plays a critical role in ensuring safety, regulatory compliance, and operational efficiency in complex maritime environments. However, existing inspection methods often rely on manual operations and conventional computer vision techniques that lack scalability and contextual understanding. This study introduces a novel integrated engineering framework that utilizes the synergy between Large Language Models (LLMs) and Vision Language Models (VLMs) to enable autonomous maritime port inspection using cooperative aerial and surface robotic platforms. The proposed framework replaces traditional state-machine mission planners with LLM-driven symbolic planning and improved perception pipelines through VLM-based semantic inspection, enabling context-aware and adaptive monitoring. The LLM module translates natural language mission instructions into executable symbolic plans with dependency graphs that encode operational constraints and ensure safe UAV-USV coordination. Meanwhile, the VLM module performs real-time semantic inspection and compliance assessment, generating structured reports with contextual reasoning. The framework was validated using the extended MBZIRC Maritime Simulator with realistic port infrastructure and further assessed through real-world robotic inspection trials. The lightweight on-board design ensures suitability for resource-constrained maritime platforms, advancing the development of intelligent, autonomous inspection systems. Project resources (code and videos) can be found here: https://github.com/Muhayyuddin/llm-vlm-fusion-port-inspection
>
---
#### [new 033] Diffusion-Guided Backdoor Attacks in Real-World Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于安全任务，解决真实世界强化学习中的后门攻击问题。提出DGBA框架，通过扩散模型生成触发器，在关键状态注入，实现有效攻击同时保持正常性能。**

- **链接: [https://arxiv.org/pdf/2601.14104v1](https://arxiv.org/pdf/2601.14104v1)**

> **作者:** Tairan Huang; Qingqing Ye; Yulin Jin; Jiawei Lian; Yi Wang; Haibo Hu
>
> **摘要:** Backdoor attacks embed hidden malicious behaviors in reinforcement learning (RL) policies and activate them using triggers at test time. Most existing attacks are validated only in simulation, while their effectiveness in real-world robotic systems remains unclear. In physical deployment, safety-constrained control pipelines such as velocity limiting, action smoothing, and collision avoidance suppress abnormal actions, causing strong attenuation of conventional backdoor attacks. We study this previously overlooked problem and propose a diffusion-guided backdoor attack framework (DGBA) for real-world RL. We design small printable visual patch triggers placed on the floor and generate them using a conditional diffusion model that produces diverse patch appearances under real-world visual variations. We treat the robot control stack as a black-box system. We further introduce an advantage-based poisoning strategy that injects triggers only at decision-critical training states. We evaluate our method on a TurtleBot3 mobile robot and demonstrate reliable activation of targeted attacks while preserving normal task performance. Demo videos and code are available in the supplementary material.
>
---
#### [new 034] Language-Based Swarm Perception: Decentralized Person Re-Identification via Natural Language Descriptions
- **分类: cs.RO**

- **简介: 该论文属于机器人协同感知任务，旨在解决去中心化的人体再识别问题。通过自然语言描述代替视觉特征向量，实现机器人间的协作与解释性感知。**

- **链接: [https://arxiv.org/pdf/2601.12479v1](https://arxiv.org/pdf/2601.12479v1)**

> **作者:** Miquel Kegeleirs; Lorenzo Garattoni; Gianpiero Francesca; Mauro Birattari
>
> **摘要:** We introduce a method for decentralized person re-identification in robot swarms that leverages natural language as the primary representational modality. Unlike traditional approaches that rely on opaque visual embeddings -- high-dimensional feature vectors extracted from images -- the proposed method uses human-readable language to represent observations. Each robot locally detects and describes individuals using a vision-language model (VLM), producing textual descriptions of appearance instead of feature vectors. These descriptions are compared and clustered across the swarm without centralized coordination, allowing robots to collaboratively group observations of the same individual. Each cluster is distilled into a representative description by a language model, providing an interpretable, concise summary of the swarm's collective perception. This approach enables natural-language querying, enhances transparency, and supports explainable swarm behavior. Preliminary experiments demonstrate competitive performance in identity consistency and interpretability compared to embedding-based methods, despite current limitations in text similarity and computational load. Ongoing work explores refined similarity metrics, semantic navigation, and the extension of language-based perception to environmental elements. This work prioritizes decentralized perception and communication, while active navigation remains an open direction for future study.
>
---
#### [new 035] Robustness and Resilience Evaluation of Eco-Driving Strategies at Signalized Intersections
- **分类: cs.RO**

- **简介: 该论文属于智能驾驶任务，旨在评估生态驾驶策略在信号交叉口的鲁棒性和韧性。通过构建框架和指标，分析不同控制器在扰动下的性能表现。**

- **链接: [https://arxiv.org/pdf/2601.13389v1](https://arxiv.org/pdf/2601.13389v1)**

> **作者:** Zhaohui Liang; Chengyuan Ma; Keke Long; Xiaopeng Li
>
> **摘要:** Eco-driving strategies have demonstrated substantial potential for improving energy efficiency and reducing emissions, especially at signalized intersections. However, evaluations of eco-driving methods typically rely on simplified simulation or experimental conditions, where certain assumptions are made to manage complexity and experimental control. This study introduces a unified framework to evaluate eco-driving strategies through the lens of two complementary criteria: control robustness and environmental resilience. We define formal indicators that quantify performance degradation caused by internal execution variability and external environmental disturbances, respectively. These indicators are then applied to assess multiple eco-driving controllers through real-world vehicle experiments. The results reveal key tradeoffs between tracking accuracy and adaptability, showing that optimization-based controllers offer more consistent performance across varying disturbance levels, while analytical controllers may perform comparably under nominal conditions but exhibit greater sensitivity to execution and timing variability.
>
---
#### [new 036] FocusNav: Spatial Selective Attention with Waypoint Guidance for Humanoid Local Navigation
- **分类: cs.RO**

- **简介: 该论文属于人形机器人局部导航任务，旨在解决动态环境中路径规划与运动稳定性的平衡问题。提出FocusNav框架，结合路径引导注意力和稳定性感知门控机制，提升导航成功率与安全性。**

- **链接: [https://arxiv.org/pdf/2601.12790v1](https://arxiv.org/pdf/2601.12790v1)**

> **作者:** Yang Zhang; Jianming Ma; Liyun Yan; Zhanxiang Cao; Yazhou Zhang; Haoyang Li; Yue Gao
>
> **备注:** 12 pages, 11 figures
>
> **摘要:** Robust local navigation in unstructured and dynamic environments remains a significant challenge for humanoid robots, requiring a delicate balance between long-range navigation targets and immediate motion stability. In this paper, we propose FocusNav, a spatial selective attention framework that adaptively modulates the robot's perceptual field based on navigational intent and real-time stability. FocusNav features a Waypoint-Guided Spatial Cross-Attention (WGSCA) mechanism that anchors environmental feature aggregation to a sequence of predicted collision-free waypoints, ensuring task-relevant perception along the planned trajectory. To enhance robustness in complex terrains, the Stability-Aware Selective Gating (SASG) module autonomously truncates distal information when detecting instability, compelling the policy to prioritize immediate foothold safety. Extensive experiments on the Unitree G1 humanoid robot demonstrate that FocusNav significantly improves navigation success rates in challenging scenarios, outperforming baselines in both collision avoidance and motion stability, achieving robust navigation in dynamic and complex environments.
>
---
#### [new 037] Diffusion-based Inverse Model of a Distributed Tactile Sensor for Object Pose Estimation
- **分类: cs.RO**

- **简介: 该论文属于物体位姿估计任务，解决触觉数据在视觉受限环境下的位姿估计问题。通过学习逆触觉模型并结合粒子滤波，提升估计精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.13250v1](https://arxiv.org/pdf/2601.13250v1)**

> **作者:** Ante Marić; Giammarco Caroleo; Alessandro Albini; Julius Jankowski; Perla Maiolino; Sylvain Calinon
>
> **摘要:** Tactile sensing provides a promising sensing modality for object pose estimation in manipulation settings where visual information is limited due to occlusion or environmental effects. However, efficiently leveraging tactile data for estimation remains a challenge due to partial observability, with single observations corresponding to multiple possible contact configurations. This limits conventional estimation approaches largely tailored to vision. We propose to address these challenges by learning an inverse tactile sensor model using denoising diffusion. The model is conditioned on tactile observations from a distributed tactile sensor and trained in simulation using a geometric sensor model based on signed distance fields. Contact constraints are enforced during inference through single-step projection using distance and gradient information from the signed distance field. For online pose estimation, we integrate the inverse model with a particle filter through a proposal scheme that combines generated hypotheses with particles from the prior belief. Our approach is validated in simulated and real-world planar pose estimation settings, without access to visual data or tight initial pose priors. We further evaluate robustness to unmodeled contact and sensor dynamics for pose tracking in a box-pushing scenario. Compared to local sampling baselines, the inverse sensor model improves sampling efficiency and estimation accuracy while preserving multimodal beliefs across objects with varying tactile discriminability.
>
---
#### [new 038] Zero-shot adaptable task planning for autonomous construction robots: a comparative study of lightweight single and multi-AI agent systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究自主建造机器人任务规划，解决动态任务适应性问题。采用轻量级AI模型，对比单/多智能体系统，提升规划效率与通用性。**

- **链接: [https://arxiv.org/pdf/2601.14091v1](https://arxiv.org/pdf/2601.14091v1)**

> **作者:** Hossein Naderi; Alireza Shojaei; Lifu Huang; Philip Agee; Kereshmeh Afsari; Abiola Akanmu
>
> **摘要:** Robots are expected to play a major role in the future construction industry but face challenges due to high costs and difficulty adapting to dynamic tasks. This study explores the potential of foundation models to enhance the adaptability and generalizability of task planning in construction robots. Four models are proposed and implemented using lightweight, open-source large language models (LLMs) and vision language models (VLMs). These models include one single agent and three multi-agent teams that collaborate to create robot action plans. The models are evaluated across three construction roles: Painter, Safety Inspector, and Floor Tiling. Results show that the four-agent team outperforms the state-of-the-art GPT-4o in most metrics while being ten times more cost-effective. Additionally, teams with three and four agents demonstrate the improved generalizability. By discussing how agent behaviors influence outputs, this study enhances the understanding of AI teams and supports future research in diverse unstructured environments beyond construction.
>
---
#### [new 039] Imitation learning-based spacecraft rendezvous and docking method with Expert Demonstration
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于航天控制任务，旨在解决空间器对接中模型依赖强、鲁棒性不足的问题。通过模仿学习框架IL-SRD，直接从专家示范中学习控制策略，提升对接的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.12952v1](https://arxiv.org/pdf/2601.12952v1)**

> **作者:** Shibo Shao; Dong Zhou; Guanghui Sun; Liwen Zhang; Mingxuan Jiang
>
> **备注:** 6 figures, 4 tables. Focus on 6-DOF spacecraft rendezvous and docking control using imitation learning-based control method
>
> **摘要:** Existing spacecraft rendezvous and docking control methods largely rely on predefined dynamic models and often exhibit limited robustness in realistic on-orbit environments. To address this issue, this paper proposes an Imitation Learning-based spacecraft rendezvous and docking control framework (IL-SRD) that directly learns control policies from expert demonstrations, thereby reducing dependence on accurate modeling. We propose an anchored decoder target mechanism, which conditions the decoder queries on state-related anchors to explicitly constrain the control generation process. This mechanism enforces physically consistent control evolution and effectively suppresses implausible action deviations in sequential prediction, enabling reliable six-degree-of-freedom (6-DOF) rendezvous and docking control. To further enhance stability, a temporal aggregation mechanism is incorporated to mitigate error accumulation caused by the sequential prediction nature of Transformer-based models, where small inaccuracies at each time step can propagate and amplify over long horizons. Extensive simulation results demonstrate that the proposed IL-SRD framework achieves accurate and energy-efficient model-free rendezvous and docking control. Robustness evaluations further confirm its capability to maintain competitive performance under significant unknown disturbances. The source code is available at https://github.com/Dongzhou-1996/IL-SRD.
>
---
#### [new 040] KILO-EKF: Koopman-Inspired Learned Observations Extended Kalman Filter
- **分类: cs.RO**

- **简介: 该论文提出KILO-EKF，用于解决传感器融合中的定位问题。通过学习测量模型提升EKF精度，无需依赖精确的传感器模型。**

- **链接: [https://arxiv.org/pdf/2601.12463v1](https://arxiv.org/pdf/2601.12463v1)**

> **作者:** Zi Cong Guo; James R. Forbes; Timothy D. Barfoot
>
> **备注:** Submitted to RA-L. 9 pages, 9 figures, 1 table. Note: version submitted to RA-L did not include the Appendix section present in this arXiv version
>
> **摘要:** We present the Koopman-Inspired Learned Observations Extended Kalman Filter (KILO-EKF), which combines a standard EKF prediction step with a correction step based on a Koopman-inspired measurement model learned from data. By lifting measurements into a feature space where they are linear in the state, KILO-EKF enables flexible modeling of complex or poorly calibrated sensors while retaining the structure and efficiency of recursive filtering. The resulting linear-Gaussian measurement model is learned in closed form from groundtruth training data, without iterative optimization or reliance on an explicit parametric sensor model. At inference, KILO-EKF performs a standard EKF update using Jacobians obtained via the learned lifting. We validate the approach on a real-world quadrotor localization task using an IMU, ultra-wideband (UWB) sensors, and a downward-facing laser. We compare against multiple EKF baselines with varying levels of sensor calibration. KILO-EKF achieves better accuracy and consistency compared to data-calibrated baselines, and significantly outperforms EKFs that rely on imperfect geometric models, while maintaining real-time inference and fast training. These results demonstrate the effectiveness of Koopman-inspired measurement learning as a scalable alternative to traditional model-based calibration.
>
---
#### [new 041] From Shallow Waters to Mariana Trench: A Survey of Bio-inspired Underwater Soft Robots
- **分类: cs.RO**

- **简介: 论文属于海洋探索任务，旨在解决传统水下机器人环境适应性差的问题。通过分析仿生软体机器人的设计与应用，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2601.12353v1](https://arxiv.org/pdf/2601.12353v1)**

> **作者:** Jie Wang; Peng Du; Yiyuan Zhang; Zhexin Xie; Cecilia Laschi
>
> **备注:** Provisional accepted by Bioinspiration & Biomimetics
>
> **摘要:** Sample Exploring the ocean environment holds profound significance in areas such as resource exploration and ecological protection. Underwater robots struggle with extreme water pressure and often cause noise and damage to the underwater ecosystem, while bio-inspired soft robots draw inspiration from aquatic creatures to address these challenges. These bio-inspired approaches enable robots to withstand high water pressure, minimize drag, operate with efficient manipulation and sensing systems, and interact with the environment in an eco-friendly manner. Consequently, bio-inspired soft robots have emerged as a promising field for ocean exploration. This paper reviews recent advancements in underwater bio-inspired soft robots, analyses their design considerations when facing different desired functions, bio-inspirations, ambient pressure, temperature, light, and biodiversity , and finally explores the progression from bio-inspired principles to practical applications in the field and suggests potential directions for developing the next generation of underwater soft robots.
>
---
#### [new 042] A Comprehensive Review of Bio-Inspired Approaches to Coordination, Communication, and System Architecture in Underwater Swarm Robotics
- **分类: cs.RO; cs.NE**

- **简介: 该论文属于 underwater swarm robotics 领域，旨在解决协同、通信与系统架构问题。综述了生物启发的协调机制、通信策略及系统设计，分析算法与硬件进展，提出分类框架以指导未来研究。**

- **链接: [https://arxiv.org/pdf/2601.12244v1](https://arxiv.org/pdf/2601.12244v1)**

> **作者:** Shyalan Ramesh; Scott Mann; Alex Stumpf
>
> **备注:** Published as part of the Special Issue: Wide Application of Marine Robotic Systems, in the Journal of Marine Science and Engineering
>
> **摘要:** The increasing complexity of marine operations has intensified the need for intelligent robotic systems to support ocean observation, exploration, and resource management. Underwater swarm robotics offers a promising framework that extends the capabilities of individual autonomous platforms through collective coordination. Inspired by natural systems, such as fish schools and insect colonies, bio-inspired swarm approaches enable distributed decision-making, adaptability, and resilience under challenging marine conditions. Yet research in this field remains fragmented, with limited integration across algorithmic, communication, and hardware design perspectives. This review synthesises bio-inspired coordination mechanisms, communication strategies, and system design considerations for underwater swarm robotics. It examines key marine-specific algorithms, including the Artificial Fish Swarm Algorithm, Whale Optimisation Algorithm, Coral Reef Optimisation, and Marine Predators Algorithm, highlighting their applications in formation control, task allocation, and environmental interaction. The review also analyses communication constraints unique to the underwater domain and emerging acoustic, optical, and hybrid solutions that support cooperative operation. Additionally, it examines hardware and system design advances that enhance system efficiency and scalability. A multi-dimensional classification framework evaluates existing approaches across communication dependency, environmental adaptability, energy efficiency, and swarm scalability. Through this integrated analysis, the review unifies bio-inspired coordination algorithms, communication modalities, and system design approaches. It also identifies converging trends, key challenges, and future research directions for real-world deployment of underwater swarm systems.
>
---
#### [new 043] R-VoxelMap: Accurate Voxel Mapping with Recursive Plane Fitting for Online LiDAR Odometry
- **分类: cs.RO**

- **简介: 该论文属于LiDAR里程计任务，解决 voxel map 中平面拟合误差问题，提出 R-VoxelMap 方法通过递归平面拟合提升定位精度。**

- **链接: [https://arxiv.org/pdf/2601.12377v1](https://arxiv.org/pdf/2601.12377v1)**

> **作者:** Haobo Xi; Shiyong Zhang; Qianli Dong; Yunze Tong; Songyang Wu; Jing Yuan; Xuebo Zhang
>
> **摘要:** This paper proposes R-VoxelMap, a novel voxel mapping method that constructs accurate voxel maps using a geometry-driven recursive plane fitting strategy to enhance the localization accuracy of online LiDAR odometry. VoxelMap and its variants typically fit and check planes using all points in a voxel, which may lead to plane parameter deviation caused by outliers, over segmentation of large planes, and incorrect merging across different physical planes. To address these issues, R-VoxelMap utilizes a geometry-driven recursive construction strategy based on an outlier detect-and-reuse pipeline. Specifically, for each voxel, accurate planes are first fitted while separating outliers using random sample consensus (RANSAC). The remaining outliers are then propagated to deeper octree levels for recursive processing, ensuring a detailed representation of the environment. In addition, a point distribution-based validity check algorithm is devised to prevent erroneous plane merging. Extensive experiments on diverse open-source LiDAR(-inertial) simultaneous localization and mapping (SLAM) datasets validate that our method achieves higher accuracy than other state-of-the-art approaches, with comparable efficiency and memory usage. Code will be available on GitHub.
>
---
#### [new 044] The OncoReach Stylet for Brachytherapy: Design Evaluation and Pilot Study
- **分类: cs.RO**

- **简介: 该论文属于医疗设备设计任务，旨在解决传统直针限制手术路径的问题，通过开发可操控的OncoReach风格针，提升宫颈癌放射治疗的精准性。**

- **链接: [https://arxiv.org/pdf/2601.13529v1](https://arxiv.org/pdf/2601.13529v1)**

> **作者:** Pejman Kheradmand; Kent K. Yamamoto; Emma Webster; Keith Sowards; Gianna Hatheway; Katharine L. Jackson; Sabino Zani; Julie A. Raffi; Diandra N. Ayala-Peacock; Scott R. Silva; Joanna Deaton Bertram; Yash Chitalia
>
> **摘要:** Cervical cancer accounts for a significant portion of the global cancer burden among women. Interstitial brachytherapy (ISBT) is a standard procedure for treating cervical cancer; it involves placing a radioactive source through a straight hollow needle within or in close proximity to the tumor and surrounding tissue. However, the use of straight needles limits surgical planning to a linear needle path. We present the OncoReach stylet, a handheld, tendon-driven steerable stylet designed for compatibility with standard ISBT 15- and 13-gauge needles. Building upon our prior work, we evaluated design parameters like needle gauge, spherical joint count and spherical joint placement, including an asymmetric disk design to identify a configuration that maximizes bending compliance while retaining axial stiffness. Free space experiments quantified tip deflection across configurations, and a two-tube Cosserat rod model accurately predicted the centerline shape of the needle for most trials. The best performing configuration was integrated into a reusable handheld prototype that enables manual actuation. A patient-derived, multi-composite phantom model of the uterus and pelvis was developed to conduct a pilot study of the OncoReach steerable stylet with one expert user. Results showed the ability to steer from less-invasive, medial entry points to reach the lateral-most targets, underscoring the significance of steerable stylets.
>
---
#### [new 045] Autonomous Navigation at the Nano-Scale: Algorithms, Architectures, and Constraints
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于纳米级无人机自主导航任务，解决SWaP约束下的导航与控制问题，探讨了算法、架构及硬件协同设计，提出融合传统控制与数据驱动的混合方案。**

- **链接: [https://arxiv.org/pdf/2601.13252v1](https://arxiv.org/pdf/2601.13252v1)**

> **作者:** Mahmud S. Zango; Jianglin Lan
>
> **备注:** 28 pages, 5 figures, 1 table. Review article
>
> **摘要:** Autonomous navigation for nano-scale unmanned aerial vehicles (nano-UAVs) is governed by extreme Size, Weight, and Power (SWaP) constraints (with the weight < 50 g and sub-100 mW onboard processor), distinguishing it fundamentally from standard robotic paradigms. This review synthesizes the state-of-the-art in sensing, computing, and control architectures designed specifically for these sub- 100mW computational envelopes. We critically analyse the transition from classical geometry-based methods to emerging "Edge AI" paradigms, including quantized deep neural networks deployed on ultra-low-power System-on-Chips (SoCs) and neuromorphic event-based control. Beyond algorithms, we evaluate the hardware-software co-design requisite for autonomy, covering advancements in dense optical flow, optimized Simultaneous Localization and Mapping (SLAM), and learning-based flight control. While significant progress has been observed in visual navigation and relative pose estimation, our analysis reveals persistent gaps in long-term endurance, robust obstacle avoidance in dynamic environments, and the "Sim-to-Real" transfer of reinforcement learning policies. This survey provides a roadmap for bridging these gaps, advocating for hybrid architectures that fuse lightweight classical control with data-driven perception to enable fully autonomous, agile nano-UAVs in GPS-denied environments.
>
---
#### [new 046] HoverAI: An Embodied Aerial Agent for Natural Human-Drone Interaction
- **分类: cs.RO**

- **简介: 该论文提出HoverAI，解决人机无人机交互中的沟通不足问题。整合视觉、语音与对话AI，实现自然交互，提升无人机的社交响应能力。**

- **链接: [https://arxiv.org/pdf/2601.13801v1](https://arxiv.org/pdf/2601.13801v1)**

> **作者:** Yuhua Jin; Nikita Kuzmin; Georgii Demianchuk; Mariya Lezina; Fawad Mehboob; Issatay Tokmurziyev; Miguel Altamirano Cabrera; Muhammad Ahsan Mustafa; Dzmitry Tsetserukou
>
> **备注:** This paper has been accepted for publication at LBR HRI 2026 conference
>
> **摘要:** Drones operating in human-occupied spaces suffer from insufficient communication mechanisms that create uncertainty about their intentions. We present HoverAI, an embodied aerial agent that integrates drone mobility, infrastructure-independent visual projection, and real-time conversational AI into a unified platform. Equipped with a MEMS laser projector, onboard semi-rigid screen, and RGB camera, HoverAI perceives users through vision and voice, responding via lip-synced avatars that adapt appearance to user demographics. The system employs a multimodal pipeline combining VAD, ASR (Whisper), LLM-based intent classification, RAG for dialogue, face analysis for personalization, and voice synthesis (XTTS v2). Evaluation demonstrates high accuracy in command recognition (F1: 0.90), demographic estimation (gender F1: 0.89, age MAE: 5.14 years), and speech transcription (WER: 0.181). By uniting aerial robotics with adaptive conversational AI and self-contained visual output, HoverAI introduces a new class of spatially-aware, socially responsive embodied agents for applications in guidance, assistance, and human-centered interaction.
>
---
#### [new 047] Learning Diverse Skills for Behavior Models with Mixture of Experts
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，解决多任务场景下模型性能下降的问题。通过引入Mixture of Experts框架，让每个专家专注于不同观测区域，提升模型多样性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.12397v1](https://arxiv.org/pdf/2601.12397v1)**

> **作者:** Wangtian Shen; Jinming Ma; Mingliang Zhou; Ziyang Meng
>
> **摘要:** Imitation learning has demonstrated strong performance in robotic manipulation by learning from large-scale human demonstrations. While existing models excel at single-task learning, it is observed in practical applications that their performance degrades in the multi-task setting, where interference across tasks leads to an averaging effect. To address this issue, we propose to learn diverse skills for behavior models with Mixture of Experts, referred to as Di-BM. Di-BM associates each expert with a distinct observation distribution, enabling experts to specialize in sub-regions of the observation space. Specifically, we employ energy-based models to represent expert-specific observation distributions and jointly train them alongside the corresponding action models. Our approach is plug-and-play and can be seamlessly integrated into standard imitation learning methods. Extensive experiments on multiple real-world robotic manipulation tasks demonstrate that Di-BM significantly outperforms state-of-the-art baselines. Moreover, fine-tuning the pretrained Di-BM on novel tasks exhibits superior data efficiency and the reusable of expert-learned knowledge. Code is available at https://github.com/robotnav-bot/Di-BM.
>
---
#### [new 048] Communication-Free Collective Navigation for a Swarm of UAVs via LiDAR-Based Deep Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG; cs.MA**

- **简介: 该论文属于无人机群协同导航任务，解决通信受限环境下的群体导航问题。通过LiDAR与深度强化学习，实现无通信的隐式领航机制。**

- **链接: [https://arxiv.org/pdf/2601.13657v1](https://arxiv.org/pdf/2601.13657v1)**

> **作者:** Myong-Yol Choi; Hankyoul Ko; Hanse Cho; Changseung Kim; Seunghwan Kim; Jaemin Seo; Hyondong Oh
>
> **摘要:** This paper presents a deep reinforcement learning (DRL) based controller for collective navigation of unmanned aerial vehicle (UAV) swarms in communication-denied environments, enabling robust operation in complex, obstacle-rich environments. Inspired by biological swarms where informed individuals guide groups without explicit communication, we employ an implicit leader-follower framework. In this paradigm, only the leader possesses goal information, while follower UAVs learn robust policies using only onboard LiDAR sensing, without requiring any inter-agent communication or leader identification. Our system utilizes LiDAR point clustering and an extended Kalman filter for stable neighbor tracking, providing reliable perception independent of external positioning systems. The core of our approach is a DRL controller, trained in GPU-accelerated Nvidia Isaac Sim, that enables followers to learn complex emergent behaviors - balancing flocking and obstacle avoidance - using only local perception. This allows the swarm to implicitly follow the leader while robustly addressing perceptual challenges such as occlusion and limited field-of-view. The robustness and sim-to-real transfer of our approach are confirmed through extensive simulations and challenging real-world experiments with a swarm of five UAVs, which successfully demonstrated collective navigation across diverse indoor and outdoor environments without any communication or external localization.
>
---
#### [new 049] RobotDesignGPT: Automated Robot Design Synthesis using Vision Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RobotDesignGPT，用于自动化机器人设计，解决传统设计依赖专家和手动反馈的问题。通过视觉语言模型，从用户提示和参考图生成高质量设计。**

- **链接: [https://arxiv.org/pdf/2601.11801v1](https://arxiv.org/pdf/2601.11801v1)**

> **作者:** Nitish Sontakke; K. Niranjan Kumar; Sehoon Ha
>
> **摘要:** Robot design is a nontrivial process that involves careful consideration of multiple criteria, including user specifications, kinematic structures, and visual appearance. Therefore, the design process often relies heavily on domain expertise and significant human effort. The majority of current methods are rule-based, requiring the specification of a grammar or a set of primitive components and modules that can be composed to create a design. We propose a novel automated robot design framework, RobotDesignGPT, that leverages the general knowledge and reasoning capabilities of large pre-trained vision-language models to automate the robot design synthesis process. Our framework synthesizes an initial robot design from a simple user prompt and a reference image. Our novel visual feedback approach allows us to greatly improve the design quality and reduce unnecessary manual feedback. We demonstrate that our framework can design visually appealing and kinematically valid robots inspired by nature, ranging from legged animals to flying creatures. We justify the proposed framework by conducting an ablation study and a user study.
>
---
#### [new 050] Sparse ActionGen: Accelerating Diffusion Policy with Real-time Pruning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，解决扩散策略实时性不足的问题。提出SAG方法，通过动态剪枝与缓存重用提升生成速度，实现高效实时决策。**

- **链接: [https://arxiv.org/pdf/2601.12894v1](https://arxiv.org/pdf/2601.12894v1)**

> **作者:** Kangye Ji; Yuan Meng; Zhou Jianbo; Ye Li; Hanyun Cui; Zhi Wang
>
> **摘要:** Diffusion Policy has dominated action generation due to its strong capabilities for modeling multi-modal action distributions, but its multi-step denoising processes make it impractical for real-time visuomotor control. Existing caching-based acceleration methods typically rely on $\textit{static}$ schedules that fail to adapt to the $\textit{dynamics}$ of robot-environment interactions, thereby leading to suboptimal performance. In this paper, we propose $\underline{\textbf{S}}$parse $\underline{\textbf{A}}$ction$\underline{\textbf{G}}$en ($\textbf{SAG}$) for extremely sparse action generation. To accommodate the iterative interactions, SAG customizes a rollout-adaptive prune-then-reuse mechanism that first identifies prunable computations globally and then reuses cached activations to substitute them during action diffusion. To capture the rollout dynamics, SAG parameterizes an observation-conditioned diffusion pruner for environment-aware adaptation and instantiates it with a highly parameter- and inference-efficient design for real-time prediction. Furthermore, SAG introduces a one-for-all reusing strategy that reuses activations across both timesteps and blocks in a zig-zag manner, minimizing the global redundancy. Extensive experiments on multiple robotic benchmarks demonstrate that SAG achieves up to 4$\times$ generation speedup without sacrificing performance. Project Page: https://sparse-actiongen.github.io/.
>
---
#### [new 051] RIM Hand : A Robotic Hand with an Accurate Carpometacarpal Joint and Nitinol-Supported Skeletal Structure
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人手设计任务，旨在提升仿生手的灵活性与抓取能力。通过精确模拟腕掌关节并使用镍钛合金结构，实现更自然的运动和更强的抓握性能。**

- **链接: [https://arxiv.org/pdf/2601.13737v1](https://arxiv.org/pdf/2601.13737v1)**

> **作者:** Joon Lee; Jeongyoon Han; Doyoung Kim; Seokhwan Jeong
>
> **备注:** Soft Robotics
>
> **摘要:** This paper presents the flexible RIM Hand, a biomimetic robotic hand that precisely replicates the carpometacarpal (CMC) joints and employs superelastic Nitinol wires throughout its skeletal framework. By modeling the full carpal-to-metacarpal anatomy, the design enables realistic palm deformation through tendon-driven fingers while enhancing joint restoration and supports skeletal structure with Nitinol-based dorsal extensors. A flexible silicone skin further increases contact friction and contact area, enabling stable grasps for diverse objects. Experiments show that the palm can deform up to 28%, matching human hand flexibility, while achieving more than twice the payload capacity and three times the contact area compared to a rigid palm design. The RIM Hand thus offers improved dexterity, compliance, and anthropomorphism, making it promising for prosthetic and service-robot applications.
>
---
#### [new 052] TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，解决VLA模型在保持语义理解与学习细粒度操作技能间的矛盾。提出TwinBrainVLA架构，结合通用与专用VLM，提升操控性能并保留视觉理解能力。**

- **链接: [https://arxiv.org/pdf/2601.14133v1](https://arxiv.org/pdf/2601.14133v1)**

> **作者:** Bin Yu; Shijie Lian; Xiaopeng Lin; Yuliang Wei; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Xinming Wang; Bailing Wang; Cong Huang; Kai Chen
>
> **备注:** GitHub: https://github.com/ZGC-EmbodyAI/TwinBrainVLA
>
> **摘要:** Standard Vision-Language-Action (VLA) models typically fine-tune a monolithic Vision-Language Model (VLM) backbone explicitly for robotic control. However, this approach creates a critical tension between maintaining high-level general semantic understanding and learning low-level, fine-grained sensorimotor skills, often leading to "catastrophic forgetting" of the model's open-world capabilities. To resolve this conflict, we introduce TwinBrainVLA, a novel architecture that coordinates a generalist VLM retaining universal semantic understanding and a specialist VLM dedicated to embodied proprioception for joint robotic control. TwinBrainVLA synergizes a frozen "Left Brain", which retains robust general visual reasoning, with a trainable "Right Brain", specialized for embodied perception, via a novel Asymmetric Mixture-of-Transformers (AsyMoT) mechanism. This design allows the Right Brain to dynamically query semantic knowledge from the frozen Left Brain and fuse it with proprioceptive states, providing rich conditioning for a Flow-Matching Action Expert to generate precise continuous controls. Extensive experiments on SimplerEnv and RoboCasa benchmarks demonstrate that TwinBrainVLA achieves superior manipulation performance compared to state-of-the-art baselines while explicitly preserving the comprehensive visual understanding capabilities of the pre-trained VLM, offering a promising direction for building general-purpose robots that simultaneously achieve high-level semantic understanding and low-level physical dexterity.
>
---
#### [new 053] CLEAR: A Semantic-Geometric Terrain Abstraction for Large-Scale Unstructured Environments
- **分类: cs.RO**

- **简介: 该论文提出CLEAR，用于大范围非结构化环境的语义几何地形抽象，解决导航路径规划效率与可靠性问题。**

- **链接: [https://arxiv.org/pdf/2601.13361v1](https://arxiv.org/pdf/2601.13361v1)**

> **作者:** Pranay Meshram; Charuvahan Adhivarahan; Ehsan Tarkesh Esfahani; Souma Chowdhury; Chen Wang; Karthik Dantu
>
> **备注:** Under review for an IEEE conference
>
> **摘要:** Long-horizon navigation in unstructured environments demands terrain abstractions that scale to tens of km$^2$ while preserving semantic and geometric structure, a combination existing methods fail to achieve. Grids scale poorly; quadtrees misalign with terrain boundaries; neither encodes landcover semantics essential for traversability-aware planning. This yields infeasible or unreliable paths for autonomous ground vehicles operating over 10+ km$^2$ under real-time constraints. CLEAR (Connected Landcover Elevation Abstract Representation) couples boundary-aware spatial decomposition with recursive plane fitting to produce convex, semantically aligned regions encoded as a terrain-aware graph. Evaluated on maps spanning 9-100~km$^2$ using a physics-based simulator, CLEAR achieves up to 10x faster planning than raw grids with only 6.7% cost overhead and delivers 6-9% shorter, more reliable paths than other abstraction baselines. These results highlight CLEAR's scalability and utility for long-range navigation in applications such as disaster response, defense, and planetary exploration.
>
---
#### [new 054] Sample Efficient Learning of Body-Environment Interaction of an Under-Actuated System
- **分类: cs.RO**

- **简介: 该论文属于机器人运动学习任务，旨在解决欠驱动系统与环境交互的高效建模问题。通过比较不同方法在预测身体速度上的表现，探索数据量对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2601.13777v1](https://arxiv.org/pdf/2601.13777v1)**

> **作者:** Zvi Chapnik; Yizhar Or; Shai Revzen
>
> **摘要:** Geometric mechanics provides valuable insights into how biological and robotic systems use changes in shape to move by mechanically interacting with their environment. In high-friction environments it provides that the entire interaction is captured by the ``motility map''. Here we compare methods for learning the motility map from motion tracking data of a physical robot created specifically to test these methods by having under-actuated degrees of freedom and a hard to model interaction with its substrate. We compared four modeling approaches in terms of their ability to predict body velocity from shape change within the same gait, across gaits, and across speeds. Our results show a trade-off between simpler methods which are superior on small training datasets, and more sophisticated methods, which are superior when more training data is available.
>
---
#### [new 055] Event-based Heterogeneous Information Processing for Online Vision-based Obstacle Detection and Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉导航任务，旨在解决动态环境中的障碍物检测与定位问题。通过融合ANN与SNN，实现高效、准确的实时处理。**

- **链接: [https://arxiv.org/pdf/2601.13451v1](https://arxiv.org/pdf/2601.13451v1)**

> **作者:** Reza Ahmadvand; Sarah Safura Sharif; Yaser Mike Banad
>
> **摘要:** This paper introduces a novel framework for robotic vision-based navigation that integrates Hybrid Neural Networks (HNNs) with Spiking Neural Network (SNN)-based filtering to enhance situational awareness for unmodeled obstacle detection and localization. By leveraging the complementary strengths of Artificial Neural Networks (ANNs) and SNNs, the system achieves both accurate environmental understanding and fast, energy-efficient processing. The proposed architecture employs a dual-pathway approach: an ANN component processes static spatial features at low frequency, while an SNN component handles dynamic, event-based sensor data in real time. Unlike conventional hybrid architectures that rely on domain conversion mechanisms, our system incorporates a pre-developed SNN-based filter that directly utilizes spike-encoded inputs for localization and state estimation. Detected anomalies are validated using contextual information from the ANN pathway and continuously tracked to support anticipatory navigation strategies. Simulation results demonstrate that the proposed method offers acceptable detection accuracy while maintaining computational efficiency close to SNN-only implementations, which operate at a fraction of the resource cost. This framework represents a significant advancement in neuromorphic navigation systems for robots operating in unpredictable and dynamic environments.
>
---
#### [new 056] Group-Invariant Unsupervised Skill Discovery: Symmetry-aware Skill Representations for Generalizable Behavior
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于技能发现任务，旨在解决环境对称性被忽略导致的冗余行为问题。提出GISD框架，通过嵌入群结构提升技能泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.14000v1](https://arxiv.org/pdf/2601.14000v1)**

> **作者:** Junwoo Chang; Joseph Park; Roberto Horowitz; Jongmin Lee; Jongeun Choi
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Unsupervised skill discovery aims to acquire behavior primitives that improve exploration and accelerate downstream task learning. However, existing approaches often ignore the geometric symmetries of physical environments, leading to redundant behaviors and sample inefficiency. To address this, we introduce Group-Invariant Skill Discovery (GISD), a framework that explicitly embeds group structure into the skill discovery objective. Our approach is grounded in a theoretical guarantee: we prove that in group-symmetric environments, the standard Wasserstein dependency measure admits a globally optimal solution comprised of an equivariant policy and a group-invariant scoring function. Motivated by this, we formulate the Group-Invariant Wasserstein dependency measure, which restricts the optimization to this symmetry-aware subspace without loss of optimality. Practically, we parameterize the scoring function using a group Fourier representation and define the intrinsic reward via the alignment of equivariant latent features, ensuring that the discovered skills generalize systematically under group transformations. Experiments on state-based and pixel-based locomotion benchmarks demonstrate that GISD achieves broader state-space coverage and improved efficiency in downstream task learning compared to a strong baseline.
>
---
#### [new 057] Being-H0.5: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization
- **分类: cs.RO**

- **简介: 该论文提出Being-H0.5，解决机器人跨形态泛化问题，通过人机交互数据提升多平台技能迁移能力。**

- **链接: [https://arxiv.org/pdf/2601.12993v1](https://arxiv.org/pdf/2601.12993v1)**

> **作者:** Hao Luo; Ye Wang; Wanpeng Zhang; Sipeng Zheng; Ziheng Xi; Chaoyi Xu; Haiweng Xu; Haoqi Yuan; Chi Zhang; Yiqing Wang; Yicheng Feng; Zongqing Lu
>
> **备注:** 44 pages
>
> **摘要:** We introduce Being-H0.5, a foundational Vision-Language-Action (VLA) model designed for robust cross-embodiment generalization across diverse robotic platforms. While existing VLAs often struggle with morphological heterogeneity and data scarcity, we propose a human-centric learning paradigm that treats human interaction traces as a universal "mother tongue" for physical interaction. To support this, we present UniHand-2.0, the largest embodied pre-training recipe to date, comprising over 35,000 hours of multimodal data across 30 distinct robotic embodiments. Our approach introduces a Unified Action Space that maps heterogeneous robot controls into semantically aligned slots, enabling low-resource robots to bootstrap skills from human data and high-resource platforms. Built upon this human-centric foundation, we design a unified sequential modeling and multi-task pre-training paradigm to bridge human demonstrations and robotic execution. Architecturally, Being-H0.5 utilizes a Mixture-of-Transformers design featuring a novel Mixture-of-Flow (MoF) framework to decouple shared motor primitives from specialized embodiment-specific experts. Finally, to make cross-embodiment policies stable in the real world, we introduce Manifold-Preserving Gating for robustness under sensory shift and Universal Async Chunking to universalize chunked control across embodiments with different latency and control profiles. We empirically demonstrate that Being-H0.5 achieves state-of-the-art results on simulated benchmarks, such as LIBERO (98.9%) and RoboCasa (53.9%), while also exhibiting strong cross-embodiment capabilities on five robotic platforms.
>
---
#### [new 058] ReWorld: Multi-Dimensional Reward Modeling for Embodied World Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ReWorld框架，解决视频世界模型物理真实性不足的问题。通过强化学习提升模型的物理拟真、任务逻辑和视觉质量，适用于复杂操作任务。**

- **链接: [https://arxiv.org/pdf/2601.12428v1](https://arxiv.org/pdf/2601.12428v1)**

> **作者:** Baorui Peng; Wenyao Zhang; Liang Xu; Zekun Qi; Jiazhao Zhang; Hongsi Liu; Wenjun Zeng; Xin Jin
>
> **摘要:** Recently, video-based world models that learn to simulate the dynamics have gained increasing attention in robot learning. However, current approaches primarily emphasize visual generative quality while overlooking physical fidelity, dynamic consistency, and task logic, especially for contact-rich manipulation tasks, which limits their applicability to downstream tasks. To this end, we introduce ReWorld, a framework aimed to employ reinforcement learning to align the video-based embodied world models with physical realism, task completion capability, embodiment plausibility and visual quality. Specifically, we first construct a large-scale (~235K) video preference dataset and employ it to train a hierarchical reward model designed to capture multi-dimensional reward consistent with human preferences. We further propose a practical alignment algorithm that post-trains flow-based world models using this reward through a computationally efficient PPO-style algorithm. Comprehensive experiments and theoretical analysis demonstrate that ReWorld significantly improves the physical fidelity, logical coherence, embodiment and visual quality of generated rollouts, outperforming previous methods.
>
---
#### [new 059] Contact-Aware Neural Dynamics
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决sim-to-real对齐问题。通过引入接触感知的神经动力学模型，利用真实接触信息提升模拟精度与政策性能。**

- **链接: [https://arxiv.org/pdf/2601.12796v1](https://arxiv.org/pdf/2601.12796v1)**

> **作者:** Changwei Jing; Jai Krishna Bandi; Jianglong Ye; Yan Duan; Pieter Abbeel; Xiaolong Wang; Sha Yi
>
> **备注:** 8 pages
>
> **摘要:** High-fidelity physics simulation is essential for scalable robotic learning, but the sim-to-real gap persists, especially for tasks involving complex, dynamic, and discontinuous interactions like physical contacts. Explicit system identification, which tunes explicit simulator parameters, is often insufficient to align the intricate, high-dimensional, and state-dependent dynamics of the real world. To overcome this, we propose an implicit sim-to-real alignment framework that learns to directly align the simulator's dynamics with contact information. Our method treats the off-the-shelf simulator as a base prior and learns a contact-aware neural dynamics model to refine simulated states using real-world observations. We show that using tactile contact information from robotic hands can effectively model the non-smooth discontinuities inherent in contact-rich tasks, resulting in a neural dynamics model grounded by real-world data. We demonstrate that this learned forward dynamics model improves state prediction accuracy and can be effectively used to predict policy performance and refine policies trained purely in standard simulators, offering a scalable, data-driven approach to sim-to-real alignment.
>
---
#### [new 060] Efficient Coordination with the System-Level Shared State: An Embodied-AI Native Modular Framework
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出ANCHOR框架，解决Embodied AI系统在部署中的耦合与可靠性问题，通过显式共享状态和通信机制实现模块化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.13945v1](https://arxiv.org/pdf/2601.13945v1)**

> **作者:** Yixuan Deng; Tongrun Wu; Donghao Wu; Zeyu Wei; Jiayuan Wang; Zhenglong Sun; Yuqing Tang; Xiaoqiang Ji
>
> **摘要:** As Embodied AI systems move from research prototypes to real world deployments, they tend to evolve rapidly while remaining reliable under workload changes and partial failures. In practice, many deployments are only partially decoupled: middleware moves messages, but shared context and feedback semantics are implicit, causing interface drift, cross-module interference, and brittle recovery at scale. We present ANCHOR, a modular framework that makes decoupling and robustness explicit system-level primitives. ANCHOR separates (i) Canonical Records, an evolvable contract for the standardized shared state, from (ii) a communication bus for many-to-many dissemination and feedback-oriented coordination, forming an inspectable end-to-end loop. We validate closed-loop feasibility on a de-identified workflow instantiation, characterize latency distributions under varying payload sizes and publish rates, and demonstrate automatic stream resumption after hard crashes and restarts even with shared-memory loss. Overall, ANCHOR turns ad-hoc integration glue into explicit contracts, enabling controlled degradation under load and self-healing recovery for scalable deployment of closed-loop AI systems.
>
---
#### [new 061] SUNSET -- A Sensor-fUsioN based semantic SegmEnTation exemplar for ROS-based self-adaptation
- **分类: cs.RO**

- **简介: 该论文提出SUNSET，一个基于ROS2的自适应系统示例，用于研究传感器融合语义分割中的自我修复与优化问题。**

- **链接: [https://arxiv.org/pdf/2601.13732v1](https://arxiv.org/pdf/2601.13732v1)**

> **作者:** Andreas Wiedholz; Rafael Paintner; Julian Gleißner; Alwin Hoffmann; Tobias Huber
>
> **摘要:** The fact that robots are getting deployed more often in dynamic environments, together with the increasing complexity of their software systems, raises the need for self-adaptive approaches. In these environments robotic software systems increasingly operate amid (1) uncertainties, where symptoms are easy to observe but root causes are ambiguous, or (2) multiple uncertainties appear concurrently. We present SUNSET, a ROS2-based exemplar that enables rigorous, repeatable evaluation of architecture-based self-adaptation in such conditions. It implements a sensor fusion semantic-segmentation pipeline driven by a trained Machine Learning (ML) model whose input preprocessing can be perturbed to induce realistic performance degradations. The exemplar exposes five observable symptoms, where each can be caused by different root causes and supports concurrent uncertainties spanning self-healing and self-optimisation. SUNSET includes the segmentation pipeline, a trained ML model, uncertainty-injection scripts, a baseline controller, and step-by-step integration and evaluation documentation to facilitate reproducible studies and fair comparison.
>
---
#### [new 062] OpenNavMap: Structure-Free Topometric Mapping via Large-Scale Collaborative Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出OpenNavMap，解决大规模、结构自由的拓扑度量地图构建问题，通过协作定位和3D几何模型实现高效、准确的导航。**

- **链接: [https://arxiv.org/pdf/2601.12291v1](https://arxiv.org/pdf/2601.12291v1)**

> **作者:** Jianhao Jiao; Changkun Liu; Jingwen Yu; Boyi Liu; Qianyi Zhang; Yue Wang; Dimitrios Kanoulas
>
> **备注:** 21 pages, 20 figures
>
> **摘要:** Scalable and maintainable map representations are fundamental to enabling large-scale visual navigation and facilitating the deployment of robots in real-world environments. While collaborative localization across multi-session mapping enhances efficiency, traditional structure-based methods struggle with high maintenance costs and fail in feature-less environments or under significant viewpoint changes typical of crowd-sourced data. To address this, we propose OPENNAVMAP, a lightweight, structure-free topometric system leveraging 3D geometric foundation models for on-demand reconstruction. Our method unifies dynamic programming-based sequence matching, geometric verification, and confidence-calibrated optimization to robust, coarse-to-fine submap alignment without requiring pre-built 3D models. Evaluations on the Map-Free benchmark demonstrate superior accuracy over structure-from-motion and regression baselines, achieving an average translation error of 0.62m. Furthermore, the system maintains global consistency across 15km of multi-session data with an absolute trajectory error below 3m for map merging. Finally, we validate practical utility through 12 successful autonomous image-goal navigation tasks on simulated and physical robots. Code and datasets will be publicly available in https://rpl-cs-ucl.github.io/OpenNavMap_page.
>
---
#### [new 063] Physics-Constrained Denoising Autoencoders for Data-Scarce Wildfire UAV Sensing
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文属于数据稀疏下的野火监测任务，解决UAV传感器数据噪声问题。提出物理约束的去噪自编码器PC²DAE，提升浓度估计准确性。**

- **链接: [https://arxiv.org/pdf/2601.11794v1](https://arxiv.org/pdf/2601.11794v1)**

> **作者:** Abdelrahman Ramadan; Zahra Dorbeigi Namaghi; Emily Taylor; Lucas Edwards; Xan Giuliani; David S. McLagan; Sidney Givigi; Melissa Greeff
>
> **摘要:** Wildfire monitoring requires high-resolution atmospheric measurements, yet low-cost sensors on Unmanned Aerial Vehicles (UAVs) exhibit baseline drift, cross-sensitivity, and response lag that corrupt concentration estimates. Traditional deep learning denoising approaches demand large datasets impractical to obtain from limited UAV flight campaigns. We present PC$^2$DAE, a physics-informed denoising autoencoder that addresses data scarcity by embedding physical constraints directly into the network architecture. Non-negative concentration estimates are enforced via softplus activations and physically plausible temporal smoothing, ensuring outputs are physically admissible by construction rather than relying on loss function penalties. The architecture employs hierarchical decoder heads for Black Carbon, Gas, and CO$_2$ sensor families, with two variants: PC$^2$DAE-Lean (21k parameters) for edge deployment and PC$^2$DAE-Wide (204k parameters) for offline processing. We evaluate on 7,894 synchronized 1 Hz samples collected from UAV flights during prescribed burns in Saskatchewan, Canada (approximately 2.2 hours of flight data), two orders of magnitude below typical deep learning requirements. PC$^2$DAE-Lean achieves 67.3\% smoothness improvement and 90.7\% high-frequency noise reduction with zero physics violations. Five baselines (LSTM-AE, U-Net, Transformer, CBDAE, DeSpaWN) produce 15--23\% negative outputs. The lean variant outperforms wide (+5.6\% smoothness), suggesting reduced capacity with strong inductive bias prevents overfitting in data-scarce regimes. Training completes in under 65 seconds on consumer hardware.
>
---
#### [new 064] Learning Fine-Grained Correspondence with Cross-Perspective Perception for Open-Vocabulary 6D Object Pose Estimation
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于6D物体位姿估计任务，解决开放词汇下物体位姿估计中全局匹配模糊的问题。通过引入细粒度对应机制和跨视角感知模块，提升位姿估计的精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.13565v1](https://arxiv.org/pdf/2601.13565v1)**

> **作者:** Yu Qin; Shimeng Fan; Fan Yang; Zixuan Xue; Zijie Mai; Wenrui Chen; Kailun Yang; Zhiyong Li
>
> **备注:** The source code will be made publicly available at https://github.com/zjjqinyu/FiCoP
>
> **摘要:** Open-vocabulary 6D object pose estimation empowers robots to manipulate arbitrary unseen objects guided solely by natural language. However, a critical limitation of existing approaches is their reliance on unconstrained global matching strategies. In open-world scenarios, trying to match anchor features against the entire query image space introduces excessive ambiguity, as target features are easily confused with background distractors. To resolve this, we propose Fine-grained Correspondence Pose Estimation (FiCoP), a framework that transitions from noise-prone global matching to spatially-constrained patch-level correspondence. Our core innovation lies in leveraging a patch-to-patch correlation matrix as a structural prior to narrowing the matching scope, effectively filtering out irrelevant clutter to prevent it from degrading pose estimation. Firstly, we introduce an object-centric disentanglement preprocessing to isolate the semantic target from environmental noise. Secondly, a Cross-Perspective Global Perception (CPGP) module is proposed to fuse dual-view features, establishing structural consensus through explicit context reasoning. Finally, we design a Patch Correlation Predictor (PCP) that generates a precise block-wise association map, acting as a spatial filter to enforce fine-grained, noise-resilient matching. Experiments on the REAL275 and Toyota-Light datasets demonstrate that FiCoP improves Average Recall by 8.0% and 6.1%, respectively, compared to the state-of-the-art method, highlighting its capability to deliver robust and generalized perception for robotic agents operating in complex, unconstrained open-world environments. The source code will be made publicly available at https://github.com/zjjqinyu/FiCoP.
>
---
#### [new 065] Towards Natural Language Environment: Understanding Seamless Natural-Language-Based Human-Multi-Robot Interactions
- **分类: cs.HC; cs.RO**

- **简介: 该论文探讨自然语言环境下的多机器人协作问题，旨在理解人与多异构机器人的自然语言交互设计空间，通过角色扮演研究分析任务协调与机器人自主性等关键问题。**

- **链接: [https://arxiv.org/pdf/2601.13338v1](https://arxiv.org/pdf/2601.13338v1)**

> **作者:** Ziyi Liu; Xinyi Wang; Shao-Kang Hsia; Chenfei Zhu; Zhengze Zhu; Xiyun Hu; Anastasia Kouvaras Ostrowski; Karthik Ramani
>
> **摘要:** As multiple robots are expected to coexist in future households, natural language is increasingly envisioned as a primary medium for human-robot and robot-robot communication. This paper introduces the concept of a Natural Language Environment (NLE), defined as an interaction space in which humans and multiple heterogeneous robots coordinate primarily through natural language. Rather than proposing a deployable system, this work aims to explore the design space of such environments. We first synthesize prior work on language-based human-robot interaction to derive a preliminary design space for NLEs. We then conduct a role-playing study in virtual reality to investigate how people conceptualize, negotiate, and coordinate human-multi-robot interactions within this imagined environment. Based on qualitative and quantitative analysis, we refine the preliminary design space and derive design implications that highlight key tensions and opportunities around task coordination dominance, robot autonomy, and robot personality in Natural Language Environments.
>
---
#### [new 066] Domain-specific Hardware Acceleration for Model Predictive Path Integral Control
- **分类: cs.AR; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决MPPI控制计算负载高、能耗大的问题。提出一种专用硬件加速器，提升轨迹精度与能效。**

- **链接: [https://arxiv.org/pdf/2601.12089v1](https://arxiv.org/pdf/2601.12089v1)**

> **作者:** Erwan Tanguy-Legac; Tommaso Belvedere; Gianluca Corsini; Marco Tognon; Marcello Traiola
>
> **备注:** 7 pages, 11 figures
>
> **摘要:** Accurately controlling a robotic system in real time is a challenging problem. To address this, the robotics community has adopted various algorithms, such as Model Predictive Control (MPC) and Model Predictive Path Integral (MPPI) control. The first is difficult to implement on non-linear systems such as unmanned aerial vehicles, whilst the second requires a heavy computational load. GPUs have been successfully used to accelerate MPPI implementations; however, their power consumption is often excessive for autonomous or unmanned targets, especially when battery-powered. On the other hand, custom designs, often implemented on FPGAs, have been proposed to accelerate robotic algorithms while consuming considerably less energy than their GPU (or CPU) implementation. However, no MPPI custom accelerator has been proposed so far. In this work, we present a hardware accelerator for MPPI control and simulate its execution. Results show that the MPPI custom accelerator allows more accurate trajectories than GPU-based MPPI implementations.
>
---
#### [new 067] UAV-Based Infrastructure Inspections: A Literature Review and Proposed Framework for AEC+FM
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于基础设施检测任务，旨在解决传统检测方法效率低的问题。通过综述150余篇文献，提出融合多模态数据的UAV检测框架，提升结构缺陷识别的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.11665v1](https://arxiv.org/pdf/2601.11665v1)**

> **作者:** Amir Farzin Nikkhah; Dong Chen; Bradford Campbell; Somayeh Asadi; Arsalan Heydarian
>
> **备注:** Accepted for publication at the International Conference on Construction Engineering and Management (I3CE 2025)
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are transforming infrastructure inspections in the Architecture, Engineering, Construction, and Facility Management (AEC+FM) domain. By synthesizing insights from over 150 studies, this review paper highlights UAV-based methodologies for data acquisition, photogrammetric modeling, defect detection, and decision-making support. Key innovations include path optimization, thermal integration, and advanced machine learning (ML) models such as YOLO and Faster R-CNN for anomaly detection. UAVs have demonstrated value in structural health monitoring (SHM), disaster response, urban infrastructure management, energy efficiency evaluations, and cultural heritage preservation. Despite these advancements, challenges in real-time processing, multimodal data fusion, and generalizability remain. A proposed workflow framework, informed by literature and a case study, integrates RGB imagery, LiDAR, and thermal sensing with transformer-based architectures to improve accuracy and reliability in detecting structural defects, thermal anomalies, and geometric inconsistencies. The proposed framework ensures precise and actionable insights by fusing multimodal data and dynamically adapting path planning for complex environments, presented as a comprehensive step-by-step guide to address these challenges effectively. This paper concludes with future research directions emphasizing lightweight AI models, adaptive flight planning, synthetic datasets, and richer modality fusion to streamline modern infrastructure inspections.
>
---
#### [new 068] A Hybrid Soft Haptic Display for Rendering Lump Stiffness in Remote Palpation
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于触觉反馈任务，旨在提升远程触诊中硬块感知的准确性。通过混合触觉显示技术，解决现有设备无法同时呈现大尺度力和精细空间细节的问题。**

- **链接: [https://arxiv.org/pdf/2601.11807v1](https://arxiv.org/pdf/2601.11807v1)**

> **作者:** Pijuan Yu; Anzu Kawazoe; Alexis Urquhart; Thomas K. Ferris; M. Cynthia Hipwell; Rebecca F. Friesen
>
> **备注:** Paper manuscript has been accepted by 2026 IEEE Haptics Symposium
>
> **摘要:** Remote palpation enables noninvasive tissue examination in telemedicine, yet current tactile displays often lack the fidelity to convey both large-scale forces and fine spatial details. This study introduces a hybrid fingertip display comprising a rigid platform and a $4\times4$ soft pneumatic tactile display (4.93 mm displacement and 1.175 N per single pneumatic chamber) to render a hard lump beneath soft tissue. This study compares three rendering strategies: a Platform-Only baseline that renders the total interaction force; a Hybrid A (Position + Force Feedback) strategy that adds a dynamic, real-time soft spatial cue; and a Hybrid B (Position + Preloaded Stiffness Feedback) strategy that provides a constant, pre-calculated soft spatial cue. In a 12-participant lump detection study, both hybrid methods dramatically improved accuracy over the Platform-Only baseline (from 50\% to over 95\%). While the Hybrid B was highlighted qualitatively for realism, its event-based averaging is expected to increase interaction latency in real-time operation. This suggests a trade-off between perceived lump realism and real-time responsiveness, such that rendering choices that enhance realism may conflict with those that minimize latency.
>
---
#### [new 069] PointSLAM++: Robust Dense Neural Gaussian Point Cloud-based SLAM
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于SLAM任务，旨在解决深度噪声下的结构一致性与定位精度问题。提出PointSLAM++，结合神经高斯表示和动态图优化，提升3D重建与渲染效果。**

- **链接: [https://arxiv.org/pdf/2601.11617v1](https://arxiv.org/pdf/2601.11617v1)**

> **作者:** Xu Wang; Boyao Han; Xiaojun Chen; Ying Liu; Ruihui Li
>
> **摘要:** Real-time 3D reconstruction is crucial for robotics and augmented reality, yet current simultaneous localization and mapping(SLAM) approaches often struggle to maintain structural consistency and robust pose estimation in the presence of depth noise. This work introduces PointSLAM++, a novel RGB-D SLAM system that leverages a hierarchically constrained neural Gaussian representation to preserve structural relationships while generating Gaussian primitives for scene mapping. It also employs progressive pose optimization to mitigate depth sensor noise, significantly enhancing localization accuracy. Furthermore, it utilizes a dynamic neural representation graph that adjusts the distribution of Gaussian nodes based on local geometric complexity, enabling the map to adapt to intricate scene details in real time. This combination yields high-precision 3D mapping and photorealistic scene rendering. Experimental results show PointSLAM++ outperforms existing 3DGS-based SLAM methods in reconstruction accuracy and rendering quality, demonstrating its advantages for large-scale AR and robotics.
>
---
#### [new 070] User-to-Vehicle Interaction in Smart Mobility: The GO-DRiVeS Autonomous Ride-Sharing Application
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于移动应用开发任务，旨在解决学生和员工长途步行及耗时问题。开发了GO-DRiVeS应用，支持实时追踪与多请求处理。**

- **链接: [https://arxiv.org/pdf/2601.12367v1](https://arxiv.org/pdf/2601.12367v1)**

> **作者:** Hana E. Elmalah; Catherine M. Elias
>
> **摘要:** This paper introduces the GO-DRiVeS application, an on demand ride sharing and requesting mobile application tailored specifically to save long walks and challenges which are time consuming and tiring especially during hot days or when carrying heavy items, faced by university students and staff. The GO-DRiVeS application was developed following the Agile methodology for its flexibility. In addition to, using the mobile application system architecture and client-server architecture. GO-DRiVeS was implemented using React Native (Expo) for the frontend, Node.js and Express for the backend, and MongoDB as the database; based on a detailed analyses to the existing transportation application, comparing their frameworks and identifying their essential functionalities. GO-DRiVeS supports core features like user registration, ride requesting and real-time tracking.In addition to handling multiple requests at the same time in a first come first serve manner. The application was developed based on these features, and the results were conducted in the form of multiple experiments that demonstrated stable behavior in handling the requests, as presented in the Methodology and Results chapters.
>
---
#### [new 071] FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决传统方法在推理与实时性上的不足。提出FantasyVLN框架，在不增加显式token的情况下实现高效多模态推理。**

- **链接: [https://arxiv.org/pdf/2601.13976v1](https://arxiv.org/pdf/2601.13976v1)**

> **作者:** Jing Zuo; Lingzhou Mu; Fan Jiang; Chengcheng Ma; Mu Xu; Yonggang Qi
>
> **摘要:** Achieving human-level performance in Vision-and-Language Navigation (VLN) requires an embodied agent to jointly understand multimodal instructions and visual-spatial context while reasoning over long action sequences. Recent works, such as NavCoT and NavGPT-2, demonstrate the potential of Chain-of-Thought (CoT) reasoning for improving interpretability and long-horizon planning. Moreover, multimodal extensions like OctoNav-R1 and CoT-VLA further validate CoT as a promising pathway toward human-like navigation reasoning. However, existing approaches face critical drawbacks: purely textual CoTs lack spatial grounding and easily overfit to sparse annotated reasoning steps, while multimodal CoTs incur severe token inflation by generating imagined visual observations, making real-time navigation impractical. In this work, we propose FantasyVLN, a unified implicit reasoning framework that preserves the benefits of CoT reasoning without explicit token overhead. Specifically, imagined visual tokens are encoded into a compact latent space using a pretrained Visual AutoRegressor (VAR) during CoT reasoning training, and the model jointly learns from textual, visual, and multimodal CoT modes under a unified multi-CoT strategy. At inference, our model performs direct instruction-to-action mapping while still enjoying reasoning-aware representations. Extensive experiments on LH-VLN show that our approach achieves reasoning-aware yet real-time navigation, improving success rates and efficiency while reducing inference latency by an order of magnitude compared to explicit CoT methods.
>
---
#### [new 072] CD-TWINSAFE: A ROS-enabled Digital Twin for Scene Understanding and Safety Emerging V2I Technology
- **分类: cs.CV; cs.HC; cs.RO**

- **简介: 该论文提出CD-TWINSAFE，一个基于V2I的数字孪生系统，用于自动驾驶场景理解和安全预警。解决实时场景感知与安全评估问题，通过双栈架构实现车辆与环境的协同感知与反馈。**

- **链接: [https://arxiv.org/pdf/2601.12373v1](https://arxiv.org/pdf/2601.12373v1)**

> **作者:** Amro Khaled; Farah Khaled; Omar Riad; Catherine M. Elias
>
> **摘要:** In this paper, the CD-TWINSAFE is introduced, a V2I-based digital twin for Autonomous Vehicles. The proposed architecture is composed of two stacks running simultaneously, an on-board driving stack that includes a stereo camera for scene understanding, and a digital twin stack that runs an Unreal Engine 5 replica of the scene viewed by the camera as well as returning safety alerts to the cockpit. The on-board stack is implemented on the vehicle side including 2 main autonomous modules; localization and perception. The position and orientation of the ego vehicle are obtained using on-board sensors. Furthermore, the perception module is responsible for processing 20-fps images from stereo camera and understands the scene through two complementary pipelines. The pipeline are working on object detection and feature extraction including object velocity, yaw and the safety metrics time-to-collision and time-headway. The collected data form the driving stack are sent to the infrastructure side through the ROS-enabled architecture in the form of custom ROS2 messages and sent over UDP links that ride a 4G modem for V2I communication. The environment is monitored via the digital twin through the shared messages which update the information of the spawned ego vehicle and detected objects based on the real-time localization and perception data. Several tests with different driving scenarios to confirm the validity and real-time response of the proposed architecture.
>
---
#### [new 073] Q-learning with Adjoint Matching
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文提出QAM算法，解决连续动作强化学习中策略优化难题，通过邻接匹配技术实现稳定高效的策略更新。**

- **链接: [https://arxiv.org/pdf/2601.14234v1](https://arxiv.org/pdf/2601.14234v1)**

> **作者:** Qiyang Li; Sergey Levine
>
> **备注:** 32 pages, 8 figures, 7 tables
>
> **摘要:** We propose Q-learning with Adjoint Matching (QAM), a novel TD-based reinforcement learning (RL) algorithm that tackles a long-standing challenge in continuous-action RL: efficient optimization of an expressive diffusion or flow-matching policy with respect to a parameterized Q-function. Effective optimization requires exploiting the first-order information of the critic, but it is challenging to do so for flow or diffusion policies because direct gradient-based optimization via backpropagation through their multi-step denoising process is numerically unstable. Existing methods work around this either by only using the value and discarding the gradient information, or by relying on approximations that sacrifice policy expressivity or bias the learned policy. QAM sidesteps both of these challenges by leveraging adjoint matching, a recently proposed technique in generative modeling, which transforms the critic's action gradient to form a step-wise objective function that is free from unstable backpropagation, while providing an unbiased, expressive policy at the optimum. Combined with temporal-difference backup for critic learning, QAM consistently outperforms prior approaches on hard, sparse reward tasks in both offline and offline-to-online RL.
>
---
#### [new 074] Learning-Augmented Online TRP on a Line
- **分类: cs.DS; cs.RO**

- **简介: 该论文研究在线旅行维修工问题，旨在利用机器学习预测提升算法性能。在预测准确时，算法竞争比为约3.732；预测有误时，竞争比可达4。**

- **链接: [https://arxiv.org/pdf/2601.13494v1](https://arxiv.org/pdf/2601.13494v1)**

> **作者:** Swapnil Guragain; Gokarna Sharma
>
> **备注:** 8 pages, 5 figures, 3 tables, and 2 pseudocodes
>
> **摘要:** We study the online traveling repairperson problem on a line within the recently proposed learning-augmented framework, which provides predictions on the requests to be served via machine learning. In the original model (with no predictions), there is a stream of requests released over time along the line. The goal is to minimize the sum (or average) of the completion times of the requests. In the original model, the state-of-the-art competitive ratio lower bound is $1+\sqrt{2} > 2.414$ for any deterministic algorithm and the state-of-the-art competitive ratio upper bound is 4 for a deterministic algorithm. Our prediction model involves predicted positions, possibly error-prone, of each request in the stream known a priori but the arrival times of requests are not known until their arrival. We first establish a 3-competitive lower bound which extends to the original model. We then design a deterministic algorithm that is $(2+\sqrt{3})\approx 3.732$-competitive when predictions are perfect. With imperfect predictions (maximum error $δ> 0$), we show that our deterministic algorithm becomes $\min\{3.732+4δ,4\}$-competitive, knowing $δ$. To the best of our knowledge, these are the first results for online traveling repairperson problem in the learning-augmented framework.
>
---
#### [new 075] From Prompts to Pavement: LMMs-based Agentic Behavior-Tree Generation Framework for Autonomous Vehicles
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决传统行为树静态、难以适应复杂环境的问题。通过LLMs和LVMs动态生成可执行行为树，提升自主车辆的适应能力。**

- **链接: [https://arxiv.org/pdf/2601.12358v1](https://arxiv.org/pdf/2601.12358v1)**

> **作者:** Omar Y. Goba; Ahmed Y. Gado; Catherine M. Elias; Ahmed Hussein
>
> **摘要:** Autonomous vehicles (AVs) require adaptive behavior planners to navigate unpredictable, real-world environments safely. Traditional behavior trees (BTs) offer structured decision logic but are inherently static and demand labor-intensive manual tuning, limiting their applicability at SAE Level 5 autonomy. This paper presents an agentic framework that leverages large language models (LLMs) and multi-modal vision models (LVMs) to generate and adapt BTs on the fly. A specialized Descriptor agent applies chain-of-symbols prompting to assess scene criticality, a Planner agent constructs high-level sub-goals via in-context learning, and a Generator agent synthesizes executable BT sub-trees in XML format. Integrated into a CARLA+Nav2 simulation, our system triggers only upon baseline BT failure, demonstrating successful navigation around unexpected obstacles (e.g., street blockage) with no human intervention. Compared to a static BT baseline, this approach is a proof-of-concept that extends to diverse driving scenarios.
>
---
#### [new 076] Affective Translation: Material and Virtual Embodiments of Kinetic Textile Robots
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在比较物理与虚拟纺织机器人的情感体验。通过实验分析运动、形态和材质对情感反应的影响，为设计情感化人机交互系统提供依据。**

- **链接: [https://arxiv.org/pdf/2601.11543v1](https://arxiv.org/pdf/2601.11543v1)**

> **作者:** Berfin Ataman; Rodrigo Gallardo; Qilmeg Doudatcz
>
> **摘要:** This study presents a comparative framework for evaluating emotional engagement with textile soft robots and their augmented-reality (AR) counterparts. Four robotic sculptures were developed, each embodying nature-inspired dynamic behaviors such as breathing and gradual deformation. Using a between-subjects design, two independent groups, one experiencing the physical installations and one engaging with their virtual (AR) twins, follow identical protocols and complete the same self-assessment survey on affective and perceptual responses. This approach minimizes carryover and novelty effects while enabling a direct comparison of sensations such as calmness, curiosity, and discomfort across modalities. The analysis explores how motion, form, and material behavior shape emotional interpretation in physical versus digital contexts, informing the design of hybrid systems that evoke meaningful, emotionally legible interactions between humans, robots, and digital twins.
>
---
#### [new 077] Reframing Conversational Design in HRI: Deliberate Design with AI Scaffolds
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，解决对话设计效率低、效果差的问题。提出ACE系统，通过AI辅助生成和优化对话提示，提升交互质量。**

- **链接: [https://arxiv.org/pdf/2601.12084v1](https://arxiv.org/pdf/2601.12084v1)**

> **作者:** Shiye Cao; Jiwon Moon; Yifan Xu; Anqi Liu; Chien-Ming Huang
>
> **摘要:** Large language models (LLMs) have enabled conversational robots to move beyond constrained dialogue toward free-form interaction. However, without context-specific adaptation, generic LLM outputs can be ineffective or inappropriate. This adaptation is often attempted through prompt engineering, which is non-intuitive and tedious. Moreover, predominant design practice in HRI relies on impression-based, trial-and-error refinement without structured methods or tools, making the process inefficient and inconsistent. To address this, we present the AI-Aided Conversation Engine (ACE), a system that supports the deliberate design of human-robot conversations. ACE contributes three key innovations: 1) an LLM-powered voice agent that scaffolds initial prompt creation to overcome the "blank page problem," 2) an annotation interface that enables the collection of granular and grounded feedback on conversational transcripts, and 3) using LLMs to translate user feedback into prompt refinements. We evaluated ACE through two user studies, examining both designs' experience and end users' interactions with robots designed using ACE. Results show that ACE facilitates the creation of robot behavior prompts with greater clarity and specificity, and that the prompts generated with ACE lead to higher-quality human-robot conversational interactions.
>
---
#### [new 078] From Design to Deorbit: A Solar-Electric Autonomous Module for Multi-Debris Remediation
- **分类: cs.DC; cs.RO**

- **简介: 该论文属于空间碎片清理任务，旨在解决传统燃料依赖方法的局限性。研究设计了一种太阳能驱动的自主模块，集成机械抓取与导航系统，实现多目标碎片清除。**

- **链接: [https://arxiv.org/pdf/2601.12830v1](https://arxiv.org/pdf/2601.12830v1)**

> **作者:** Om Mishra; Jayesh Patil; Sathwik Narkedimilli; G Srikantha Sharma; Ananda S; Manjunath K Vanahalli
>
> **备注:** 6 pages, 13 Figures, 2 tables
>
> **摘要:** The escalating accumulation of orbital debris threatens the sustainability of space operations, necessitating active removal solutions that overcome the limitations of current fuel-dependent methods. To address this, this study introduces a novel remediation architecture that integrates a mechanical clamping system for secure capture with a high-efficiency, solar-powered NASA Evolutionary Xenon Thruster (NEXT) and autonomous navigation protocols. High-fidelity simulations validate the architecture's capabilities, demonstrating a successful retrograde deorbit from 800 km to 100 km, <10m position Root Mean Square Errors (RMSE) via radar-based Extended Kalman Filter (EKF) navigation, and a 93\% data delivery efficiency within 1 second using Delay/Disruption Tolerant Network (DTN) protocols. This approach significantly advances orbital management by establishing a benchmark for renewable solar propulsion that minimizes reliance on conventional fuels and extends mission longevity for multi-target removal.
>
---
#### [new 079] DC-VLAQ: Query-Residual Aggregation for Robust Visual Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决VPR中的鲁棒性问题。通过融合不同模型的互补信息并改进全局聚合方法，提升识别性能。**

- **链接: [https://arxiv.org/pdf/2601.12729v1](https://arxiv.org/pdf/2601.12729v1)**

> **作者:** Hanyu Zhu; Zhihao Zhan; Yuhang Ming; Liang Li; Dibo Hou; Javier Civera; Wanzeng Kong
>
> **备注:** 10 pages, 4 figures, 5 tables
>
> **摘要:** One of the central challenges in visual place recognition (VPR) is learning a robust global representation that remains discriminative under large viewpoint changes, illumination variations, and severe domain shifts. While visual foundation models (VFMs) provide strong local features, most existing methods rely on a single model, overlooking the complementary cues offered by different VFMs. However, exploiting such complementary information inevitably alters token distributions, which challenges the stability of existing query-based global aggregation schemes. To address these challenges, we propose DC-VLAQ, a representation-centric framework that integrates the fusion of complementary VFMs and robust global aggregation. Specifically, we first introduce a lightweight residual-guided complementary fusion that anchors representations in the DINOv2 feature space while injecting complementary semantics from CLIP through a learned residual correction. In addition, we propose the Vector of Local Aggregated Queries (VLAQ), a query--residual global aggregation scheme that encodes local tokens by their residual responses to learnable queries, resulting in improved stability and the preservation of fine-grained discriminative cues. Extensive experiments on standard VPR benchmarks, including Pitts30k, Tokyo24/7, MSLS, Nordland, SPED, and AmsterTime, demonstrate that DC-VLAQ consistently outperforms strong baselines and achieves state-of-the-art performance, particularly under challenging domain shifts and long-term appearance changes.
>
---
#### [new 080] Listen, Look, Drive: Coupling Audio Instructions for User-aware VLA-based Autonomous Driving
- **分类: eess.AS; cs.MM; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决单一视觉输入导致的决策延迟问题。通过融合音频指令与视觉信息，提出EchoVLA模型，提升驾驶行为的适应性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.12142v1](https://arxiv.org/pdf/2601.12142v1)**

> **作者:** Ziang Guo; Feng Yang; Xuefeng Zhang; Jiaqi Guo; Kun Zhao; Peng Lu; Zufeng Zhang; Sifa Zheng
>
> **备注:** Accepted by IV
>
> **摘要:** Vision Language Action (VLA) models promise an open-vocabulary interface that can translate perceptual ambiguity into semantically grounded driving decisions, yet they still treat language as a static prior fixed at inference time. As a result, the model must infer continuously shifting objectives from pixels alone, yielding delayed or overly conservative maneuvers. We argue that effective VLAs for autonomous driving need an online channel in which users can influence driving with specific intentions. To this end, we present EchoVLA, a user-aware VLA that couples camera streams with in situ audio instructions. We augment the nuScenes dataset with temporally aligned, intent-specific speech commands generated by converting ego-motion descriptions into synthetic audios. Further, we compose emotional speech-trajectory pairs into a multimodal Chain-of-Thought (CoT) for fine-tuning a Multimodal Large Model (MLM) based on Qwen2.5-Omni. Specifically, we synthesize the audio-augmented dataset with different emotion types paired with corresponding driving behaviors, leveraging the emotional cues embedded in tone, pitch, and speech tempo to reflect varying user states, such as urgent or hesitant intentions, thus enabling our EchoVLA to interpret not only the semantic content but also the emotional context of audio commands for more nuanced and emotionally adaptive driving behavior. In open-loop benchmarks, our approach reduces the average L2 error by $59.4\%$ and the collision rate by $74.4\%$ compared to the baseline of vision-only perception. More experiments on nuScenes dataset validate that EchoVLA not only steers the trajectory through audio instructions, but also modulates driving behavior in response to the emotions detected in the user's speech.
>
---
## 更新

#### [replaced 001] Floor Plan-Guided Visual Navigation Incorporating Depth and Directional Cues
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，旨在解决导航效率低的问题。通过结合深度信息和楼层平面图，提出GlocDiff框架，提升导航效果与效率。**

- **链接: [https://arxiv.org/pdf/2511.01493v3](https://arxiv.org/pdf/2511.01493v3)**

> **作者:** Weiqi Huang; Jiaxin Li; Zan Wang; Huijun Di; Wei Liang; Zhu Yang
>
> **摘要:** Current visual navigation strategies mainly follow an exploration-first and then goal-directed navigation paradigm. This exploratory phase inevitably compromises the overall efficiency of navigation. Recent studies propose leveraging floor plans alongside RGB inputs to guide agents, aiming for rapid navigation without prior exploration or mapping. Key issues persist despite early successes. The modal gap and content misalignment between floor plans and RGB images necessitate an efficient approach to extract the most salient and complementary features from both for reliable navigation. Here, we propose GlocDiff, a novel framework that employs a diffusion-based policy to continuously predict future waypoints. This policy is conditioned on two complementary information streams: (1) local depth cues derived from the current RGB observation, and (2) global directional guidance extracted from the floor plan. The former handles immediate navigation safety by capturing surrounding geometry, while the latter ensures goal-directed efficiency by offering definitive directional cues. Extensive evaluations on the FloNa benchmark demonstrate that GlocDiff achieves superior efficiency and effectiveness. Furthermore, its successful deployment in real-world scenarios underscores its strong potential for broad practical application.
>
---
#### [replaced 002] Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion
- **分类: cs.RO**

- **简介: 该论文提出Omni-LIVO，解决多相机LiDAR惯性里程计问题，通过跨视图对齐和ESIKF融合提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.15673v3](https://arxiv.org/pdf/2509.15673v3)**

> **作者:** Yinong Cao; Chenyang Zhang; Xin He; Yuwei Chen; Chengyu Pu; Bingtao Wang; Kaile Wu; Shouzheng Zhu; Fei Han; Shijie Liu; Chunlai Li; Jianyu Wang
>
> **摘要:** Wide field-of-view (FoV) LiDAR sensors provide dense geometry across large environments, but existing LiDAR-inertial-visual odometry (LIVO) systems generally rely on a single camera, limiting their ability to fully exploit LiDAR-derived depth for photometric alignment and scene colorization. We present Omni-LIVO, a tightly coupled multi-camera LIVO system that leverages multi-view observations to comprehensively utilize LiDAR geometric information across extended spatial regions. Omni-LIVO introduces a Cross-View direct alignment strategy that maintains photometric consistency across non-overlapping views, and extends the Error-State Iterated Kalman Filter (ESIKF) with multi-view updates and adaptive covariance. The system is evaluated on public benchmarks and our custom dataset, showing improved accuracy and robustness over state-of-the-art LIVO, LIO, and visual-inertial SLAM baselines. Code and dataset will be released upon publication.
>
---
#### [replaced 003] Safety Not Found (404): Hidden Risks of LLM-Based Robotics Decision Making
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于安全评估任务，旨在解决LLM在机器人决策中的潜在风险问题。通过设计不同场景的任务，评估LLM在安全关键系统中的可靠性，揭示其严重缺陷。**

- **链接: [https://arxiv.org/pdf/2601.05529v3](https://arxiv.org/pdf/2601.05529v3)**

> **作者:** Jua Han; Jaeyoon Seo; Jungbin Min; Jihie Kim; Jean Oh
>
> **备注:** Corrected author order in metadata; manuscript unchanged
>
> **摘要:** One mistake by an AI system in a safety-critical setting can cost lives. As Large Language Models (LLMs) become integral to robotics decision-making, the physical dimension of risk grows; a single wrong instruction can directly endanger human safety. This paper addresses the urgent need to systematically evaluate LLM performance in scenarios where even minor errors are catastrophic. Through a qualitative evaluation of a fire evacuation scenario, we identified critical failure cases in LLM-based decision-making. Based on these, we designed seven tasks for quantitative assessment, categorized into: Complete Information, Incomplete Information, and Safety-Oriented Spatial Reasoning (SOSR). Complete information tasks utilize ASCII maps to minimize interpretation ambiguity and isolate spatial reasoning from visual processing. Incomplete information tasks require models to infer missing context, testing for spatial continuity versus hallucinations. SOSR tasks use natural language to evaluate safe decision-making in life-threatening contexts. We benchmark various LLMs and Vision-Language Models (VLMs) across these tasks. Beyond aggregate performance, we analyze the implications of a 1% failure rate, highlighting how "rare" errors escalate into catastrophic outcomes. Results reveal serious vulnerabilities: several models achieved a 0% success rate in ASCII navigation, while in a simulated fire drill, models instructed robots to move toward hazardous areas instead of emergency exits. Our findings lead to a sobering conclusion: current LLMs are not ready for direct deployment in safety-critical systems. A 99% accuracy rate is dangerously misleading in robotics, as it implies one out of every hundred executions could result in catastrophic harm. We demonstrate that even state-of-the-art models cannot guarantee safety, and absolute reliance on them creates unacceptable risks.
>
---
#### [replaced 004] Message passing-based inference in an autoregressive active inference agent
- **分类: cs.AI; cs.LG; cs.RO; eess.SY; stat.ML**

- **简介: 该论文提出一种基于消息传递的自回归主动推理代理，用于机器人导航任务，解决如何在不确定环境中平衡探索与利用的问题。**

- **链接: [https://arxiv.org/pdf/2509.25482v2](https://arxiv.org/pdf/2509.25482v2)**

> **作者:** Wouter M. Kouw; Tim N. Nisslbeck; Wouter L. N. Nuijten
>
> **备注:** 14 pages, 4 figures, proceedings of the International Workshop on Active Inference 2025. Erratum v1: in Eq. (50), $p(y_t, Θ, u_t \mid y_{*}, \mathcal{D}_k)$ should have been $p(y_t, Θ\mid u_t, y_{*}, \mathcal{D}_k)$
>
> **摘要:** We present the design of an autoregressive active inference agent in the form of message passing on a factor graph. Expected free energy is derived and distributed across a planning graph. The proposed agent is validated on a robot navigation task, demonstrating exploration and exploitation in a continuous-valued observation space with bounded continuous-valued actions. Compared to a classical optimal controller, the agent modulates action based on predictive uncertainty, arriving later but with a better model of the robot's dynamics.
>
---
#### [replaced 005] Beyond Task and Motion Planning: Hierarchical Robot Planning with General-Purpose Skills
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人规划领域，解决传统方法无法有效整合复杂技能的问题。提出TASP框架，通过CIPs将多样技能与运动规划结合，提升长周期任务的执行能力。**

- **链接: [https://arxiv.org/pdf/2504.17901v2](https://arxiv.org/pdf/2504.17901v2)**

> **作者:** Benned Hedegaard; Yichen Wei; Ahmed Jaafar; Stefanie Tellex; George Konidaris; Naman Shah
>
> **摘要:** Task and motion planning is a well-established approach for solving long-horizon robot planning problems. However, traditional methods assume that each task-level robot action, or skill, can be reduced to kinematic motion planning. We address the challenge of combining motion planning with closed-loop motor controllers that go beyond mere kinematic considerations. We propose a novel framework that integrates these policies into motion planning using Composable Interaction Primitives (CIPs), enabling the use of diverse, non-composable pre-learned skills in hierarchical robot planning. We validate our Task and Skill Planning (TASP) approach through real-world experiments on a bimanual manipulator and a mobile manipulator, demonstrating that CIPs allow diverse robots to combine motion planning with general-purpose skills to solve complex, long-horizon tasks.
>
---
#### [replaced 006] Safety on the Fly: Constructing Robust Safety Filters via Policy Control Barrier Functions at Runtime
- **分类: math.OC; cs.RO**

- **简介: 该论文属于安全控制任务，旨在解决高相对阶系统在扰动和输入约束下的安全性问题。通过在线构建鲁棒策略控制屏障函数（RPCBF），提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2410.11157v3](https://arxiv.org/pdf/2410.11157v3)**

> **作者:** Luzia Knoedler; Oswin So; Ji Yin; Mitchell Black; Zachary Serlin; Panagiotis Tsiotras; Javier Alonso-Mora; Chuchu Fan
>
> **备注:** Accepted in RAL. The project page can be found at www.oswinso.xyz/rpcbf/
>
> **摘要:** Control Barrier Functions (CBFs) have proven to be an effective tool for performing safe control synthesis for nonlinear systems. However, guaranteeing safety in the presence of disturbances and input constraints for high relative degree systems is a difficult problem. In this work, we propose the Robust Policy CBF (RPCBF), a practical approach for constructing robust CBF approximations online via the estimation of a value function. We establish conditions under which the approximation qualifies as a valid CBF and demonstrate the effectiveness of the RPCBF-safety filter in simulation on a variety of high relative degree input-constrained systems. Finally, we demonstrate the benefits of our method in compensating for model errors on a hardware quadcopter platform by treating the model errors as disturbances. Website including code: www.oswinso.xyz/rpcbf/
>
---
#### [replaced 007] Shape Completion with Prediction of Uncertain Regions
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于形状补全任务，解决部分观测下几何不确定性问题。提出两种方法预测不确定区域，提升形状补全和抓取质量。**

- **链接: [https://arxiv.org/pdf/2308.00377v2](https://arxiv.org/pdf/2308.00377v2)**

> **作者:** Matthias Humt; Dominik Winkelbauer; Ulrich Hillenbrand
>
> **备注:** 7 pages, 5 figures, Published in IROS 2023. Project page: https://hummat.github.io/2023-iros-uncertain/
>
> **摘要:** Shape completion, i.e., predicting the complete geometry of an object from a partial observation, is highly relevant for several downstream tasks, most notably robotic manipulation. When basing planning or prediction of real grasps on object shape reconstruction, an indication of severe geometric uncertainty is indispensable. In particular, there can be an irreducible uncertainty in extended regions about the presence of entire object parts when given ambiguous object views. To treat this important case, we propose two novel methods for predicting such uncertain regions as straightforward extensions of any method for predicting local spatial occupancy, one through postprocessing occupancy scores, the other through direct prediction of an uncertainty indicator. We compare these methods together with two known approaches to probabilistic shape completion. Moreover, we generate a dataset, derived from ShapeNet, of realistically rendered depth images of object views with ground-truth annotations for the uncertain regions. We train on this dataset and test each method in shape completion and prediction of uncertain regions for known and novel object instances and on synthetic and real data. While direct uncertainty prediction is by far the most accurate in the segmentation of uncertain regions, both novel methods outperform the two baselines in shape completion and uncertain region prediction, and avoiding the predicted uncertain regions increases the quality of grasps for all tested methods.
>
---
#### [replaced 008] Event-Grounding Graph: Unified Spatio-Temporal Scene Graph from Robotic Observations
- **分类: cs.RO**

- **简介: 该论文提出事件锚定图（EGG），解决机器人场景中空间与动态事件关联的问题。通过构建统一的时空场景图，提升机器人对环境和事件的理解与响应能力。**

- **链接: [https://arxiv.org/pdf/2510.18697v2](https://arxiv.org/pdf/2510.18697v2)**

> **作者:** Phuoc Nguyen; Francesco Verdoja; Ville Kyrki
>
> **备注:** Submitted to RA-L
>
> **摘要:** A fundamental aspect for building intelligent autonomous robots that can assist humans in their daily lives is the construction of rich environmental representations. While advances in semantic scene representations have enriched robotic scene understanding, current approaches lack a connection between spatial features and dynamic events; e.g., connecting the blue mug to the event washing a mug. In this work, we introduce the event-grounding graph (EGG), a framework grounding event interactions to spatial features of a scene. This representation allows robots to perceive, reason, and respond to complex spatio-temporal queries. Experiments using real robotic data demonstrate EGG's capability to retrieve relevant information and respond accurately to human inquiries concerning the environment and events within. Furthermore, the EGG framework's source code and evaluation dataset are released as open-source at: https://github.com/aalto-intelligent-robotics/EGG.
>
---
#### [replaced 009] Tube-Based Robust Control Strategy for Vision-Guided Autonomous Vehicles
- **分类: eess.SY; cs.CV; cs.RO**

- **简介: 该论文属于自主车辆控制任务，旨在提升高速弯道下的鲁棒性。提出itube-CILQR算法，优化控制性能并减少计算时间。**

- **链接: [https://arxiv.org/pdf/2503.18752v2](https://arxiv.org/pdf/2503.18752v2)**

> **作者:** Der-Hau Lee
>
> **备注:** 15 pages, 16 figures
>
> **摘要:** A robust control strategy for autonomous vehicles can improve system stability, enhance riding comfort, and prevent driving accidents. This paper presents a novel interpolation-tube-based constrained iterative linear quadratic regulator (itube-CILQR) algorithm for autonomous computer-vision-based vehicle lane-keeping. The goal of the algorithm is to enhance robustness during high-speed cornering on tight turns. Compared with standard tube-based approaches, the proposed itube-CILQR algorithm reduces system conservatism and exhibits higher computational speed. Numerical simulations and vision-based experiments were conducted to examine the feasibility of using the proposed algorithm for controlling autonomous vehicles. The results indicated that the proposed algorithm achieved superior vehicle lane-keeping performance to variational CILQR-based methods and model predictive control (MPC) approaches involving the use of a classical interior-point optimizer. Specifically, itube-CILQR required an average runtime of 3.45 ms to generate a control signal for guiding a self-driving vehicle. By comparison, itube-MPC typically required a 4.32 times longer computation time to complete the same task. Moreover, the influence of conservatism on system behavior was investigated by exploring the variations in the interpolation variables derived using the proposed itube-CILQR algorithm during lane-keeping maneuvers.
>
---
#### [replaced 010] Knot So Simple: A Minimalistic Environment for Spatial Reasoning
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出KnotGym，一个用于空间推理与操作的环境，解决基于视觉的复杂绳结任务。通过不同复杂度的挑战，测试感知、推理与操作能力。**

- **链接: [https://arxiv.org/pdf/2505.18028v3](https://arxiv.org/pdf/2505.18028v3)**

> **作者:** Zizhao Chen; Yoav Artzi
>
> **备注:** Fix camera ready footer
>
> **摘要:** We propose KnotGym, an interactive environment for complex, spatial reasoning and manipulation. KnotGym includes goal-oriented rope manipulation tasks with varying levels of complexity, all requiring acting from pure image observations. Tasks are defined along a clear and quantifiable axis of complexity based on the number of knot crossings, creating a natural generalization test. KnotGym has a simple observation space, allowing for scalable development, yet it highlights core challenges in integrating acute perception, spatial reasoning, and grounded manipulation. We evaluate methods of different classes, including model-based RL, model-predictive control, and chain-of-thought reasoning, and illustrate the challenges KnotGym presents. KnotGym is available at https://github.com/lil-lab/knotgym.
>
---
#### [replaced 011] LLM-Glasses: GenAI-driven Glasses with Haptic Feedback for Navigation of Visually Impaired People
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出LLM-Glasses，一种为视障人士设计的导航系统，结合视觉识别、AI推理和触觉反馈，解决无障碍导航问题。**

- **链接: [https://arxiv.org/pdf/2503.16475v2](https://arxiv.org/pdf/2503.16475v2)**

> **作者:** Issatay Tokmurziyev; Miguel Altamirano Cabrera; Muhammad Haris Khan; Yara Mahmoud; Dzmitry Tsetserukou
>
> **摘要:** LLM-Glasses is a wearable navigation system which assists visually impaired people by utilizing YOLO-World object detection, GPT-4o-based reasoning, and haptic feedback for real-time guidance. The device translates visual scene understanding into intuitive tactile feedback on the temples, allowing hands-free navigation. Three studies evaluate the system: recognition of 13 haptic patterns with an average recognition rate of 81.3%, VICON-based guidance with predefined paths using haptic cues, and an LLM-guided scene evaluation with decision accuracies of 91.8% without obstacles, 84.6% with static obstacles, and 81.5% with dynamic obstacles. These results show that LLM-Glasses can deliver reliable navigation support in controlled environments and motivate further work on responsiveness and deployment in more complex real-world scenarios.
>
---
#### [replaced 012] CAHC:A General Conflict-Aware Heuristic Caching Framework for Multi-Agent Path Finding
- **分类: cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决传统启发式缓存无法适应约束环境的问题，提出CAHC框架，结合冲突信息提升搜索效率。**

- **链接: [https://arxiv.org/pdf/2512.12243v2](https://arxiv.org/pdf/2512.12243v2)**

> **作者:** HT To; S Nguyen; NH Pham
>
> **摘要:** Multi-Agent Path Finding (MAPF) algorithms, including those for car-like robots and grid-based scenarios, face significant computational challenges due to expensive heuristic calculations. Traditional heuristic caching assumes that the heuristic function depends only on the state, which is incorrect in constraint-based search algorithms (e.g., CBS, MAPF-LNS, MAP2) where constraints from conflict resolution make the search space context-dependent. We propose \textbf{CAHC} (Conflict-Aware Heuristic Caching), a general framework that caches heuristic values based on both state and relevant constraint context, addressing this fundamental limitation. We demonstrate CAHC through a case study on CL-CBS for car-like robots, where we combine conflict-aware caching with an adaptive hybrid heuristic in \textbf{CAR-CHASE} (Car-Like Robot Conflict-Aware Heuristic Adaptive Search Enhancement). Our key innovations are (1) a compact \emph{conflict fingerprint} that efficiently encodes which constraints affect a state's heuristic, (2) a domain-adaptable relevance filter using spatial, temporal, and geometric criteria, and (3) a modular architecture that enables systematic application to diverse MAPF algorithms. Experimental evaluation on 480 CL-CBS benchmark instances demonstrates a geometric mean speedup of 2.46$\times$ while maintaining solution optimality. The optimizations improve success rate from 77.9\% to 84.8\% (+6.9 percentage points), reduce total runtime by 70.1\%, and enable solving 33 additional instances. The framework's general architecture makes it applicable as a reliable optimization technique for MAP2, MAPF-LNS, and other constraint-based MAPF algorithms.
>
---
#### [replaced 013] EmoBipedNav: Emotion-aware Social Navigation for Bipedal Robots with Deep Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人导航任务，旨在解决 bipedal 机器人在社交环境中的安全导航问题。通过 DRL 方法，结合情感与社会交互信息，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2503.12538v2](https://arxiv.org/pdf/2503.12538v2)**

> **作者:** Wei Zhu; Abirath Raju; Abdulaziz Shamsah; Anqi Wu; Seth Hutchinson; Ye Zhao
>
> **备注:** 13 pages
>
> **摘要:** This study presents an emotion-aware navigation framework -- EmoBipedNav -- using deep reinforcement learning (DRL) for bipedal robots walking in socially interactive environments. The inherent locomotion constraints of bipedal robots challenge their safe maneuvering capabilities in dynamic environments. When combined with the intricacies of social environments, including pedestrian interactions and social cues, such as emotions, these challenges become even more pronounced. To address these coupled problems, we propose a two-stage pipeline that considers both bipedal locomotion constraints and complex social environments. Specifically, social navigation scenarios are represented using sequential LiDAR grid maps (LGMs), from which we extract latent features, including collision regions, emotion-related discomfort zones, social interactions, and the spatio-temporal dynamics of evolving environments. The extracted features are directly mapped to the actions of reduced-order models (ROMs) through a DRL architecture. Furthermore, the proposed framework incorporates full-order dynamics and locomotion constraints during training, effectively accounting for tracking errors and restrictions of the locomotion controller while planning the trajectory with ROMs. Comprehensive experiments demonstrate that our approach exceeds both model-based planners and DRL-based baselines. The hardware videos and open-source code are available at https://gatech-lidar.github.io/emobipednav.github.io/.
>
---
#### [replaced 014] Safe Navigation under State Uncertainty: Online Adaptation for Robust Control Barrier Functions
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于安全控制任务，解决状态不确定性下的导航问题。通过改进鲁棒控制屏障函数，提出在线参数自适应方法，提升系统安全性与效率。**

- **链接: [https://arxiv.org/pdf/2508.19159v2](https://arxiv.org/pdf/2508.19159v2)**

> **作者:** Ersin Das; Rahal Nanayakkara; Xiao Tan; Ryan M. Bena; Joel W. Burdick; Paulo Tabuada; Aaron D. Ames
>
> **摘要:** Measurements and state estimates are often imperfect in control practice, posing challenges for safety-critical applications, where safety guarantees rely on accurate state information. In the presence of estimation errors, several prior robust control barrier function (R-CBF) formulations have imposed strict conditions on the input. These methods can be overly conservative and can introduce issues such as infeasibility, high control effort, etc. This work proposes a systematic method to improve R-CBFs, and demonstrates its advantages on a tracked vehicle that navigates among multiple obstacles. A primary contribution is a new optimization-based online parameter adaptation scheme that reduces the conservativeness of existing R-CBFs. In order to reduce the complexity of the parameter optimization, we merge several safety constraints into one unified numerical CBF via Poisson's equation. We further address the dual relative degree issue that typically causes difficulty in vehicle tracking. Experimental trials demonstrate the overall performance improvement of our approach over existing formulations.
>
---
#### [replaced 015] PERSEUS: Perception with Semantic Endoscopic Understanding and SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手术视觉感知任务，旨在解决自然孔道手术中的可视化与定位问题。通过整合语义分割、深度估计和SLAM，实现高精度实时场景重建与定位。**

- **链接: [https://arxiv.org/pdf/2509.13541v2](https://arxiv.org/pdf/2509.13541v2)**

> **作者:** Ayberk Acar; Fangjie Li; Susheela Sharma Stern; Lidia Al-Zogbi; Hao Li; Kanyifeechukwu Jane Oguine; Dilara Isik; Brendan Burkhart; Jesse F. d'Almeida; Robert J. Webster; Ipek Oguz; Jie Ying Wu
>
> **备注:** 13 pages, 6 figures, 2 tables. Under review for The 17th International Conference on Information Processing in Computer-Assisted Interventions (IPCAI 2026)
>
> **摘要:** Purpose: Natural orifice surgeries minimize the need for incisions and reduce the recovery time compared to open surgery; however, they require a higher level of expertise due to visualization and orientation challenges. We propose a perception pipeline for these surgeries that allows semantic scene understanding. Methods: We bring learning-based segmentation, depth estimation, and 3D reconstruction modules together to create real-time segmented maps of the surgical scenes. Additionally, we use registration with robot poses to solve the scale ambiguity of mapping from monocular images, and allow the use of semantically informed real-time reconstructions in robotic surgeries. Results: We achieve sub-milimeter reconstruction accuracy based on average one-sided Chamfer distances, average pose registration RMSE of 0.9 mm, and an estimated scale within 2% of ground truth. Conclusion: We present a modular perception pipeline, integrating semantic segmentation with real-time monocular SLAM for natural orifice surgeries. This pipeline offers a promising solution for scene understanding that can facilitate automation or surgeon guidance.
>
---
#### [replaced 016] Sequentially Teaching Sequential Tasks $(ST)^2$: Teaching Robots Long-horizon Manipulation Skills
- **分类: cs.RO**

- **简介: 该论文研究机器人长时序操作技能的教学问题，提出$(ST)^2$方法，通过分步教学提升教学效率与用户体验。**

- **链接: [https://arxiv.org/pdf/2510.21046v2](https://arxiv.org/pdf/2510.21046v2)**

> **作者:** Zlatan Ajanović; Ravi Prakash; Leandro de Souza Rosa; Jens Kober
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Magazine
>
> **摘要:** Learning from demonstration has proved itself useful for teaching robots complex skills with high sample efficiency. However, teaching long-horizon tasks with multiple skills is challenging as deviations tend to accumulate, the distributional shift becomes more evident, and human teachers become fatigued over time, thereby increasing the likelihood of failure. To address these challenges, we introduce $(ST)^2$, a sequential method for learning long-horizon manipulation tasks that allows users to control the teaching flow by specifying key points, enabling structured and incremental demonstrations. Using this framework, we study how users respond to two teaching paradigms: (i) a traditional monolithic approach, in which users demonstrate the entire task trajectory at once, and (ii) a sequential approach, in which the task is segmented and demonstrated step by step. We conducted an extensive user study on the restocking task with $16$ participants in a realistic retail store environment, evaluating the user preferences and effectiveness of the methods. User-level analysis showed superior performance for the sequential approach in most cases (10 users), compared with the monolithic approach (5 users), with one tie. Our subjective results indicate that some teachers prefer sequential teaching -- as it allows them to teach complicated tasks iteratively -- or others prefer teaching in one go due to its simplicity.
>
---
#### [replaced 017] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出A0模型，解决机器人操作中的空间可及性理解问题。通过分层结构实现高效、通用的操控，适用于多种机器人平台。**

- **链接: [https://arxiv.org/pdf/2504.12636v5](https://arxiv.org/pdf/2504.12636v5)**

> **作者:** Rongtao Xu; Jian Zhang; Minghao Guo; Youpeng Wen; Haoting Yang; Min Lin; Jianzheng Huang; Zhe Li; Kaidong Zhang; Liqiong Wang; Yuxuan Kuang; Meng Cao; Feng Zheng; Xiaodan Liang
>
> **摘要:** Robotic manipulation faces critical challenges in understanding spatial affordances--the "where" and "how" of object interactions--essential for complex manipulation tasks like wiping a board or stacking objects. Existing methods, including modular-based and end-to-end approaches, often lack robust spatial reasoning capabilities. Unlike recent point-based and flow-based affordance methods that focus on dense spatial representations or trajectory modeling, we propose A0, a hierarchical affordance-aware diffusion model that decomposes manipulation tasks into high-level spatial affordance understanding and low-level action execution. A0 leverages the Embodiment-Agnostic Affordance Representation, which captures object-centric spatial affordances by predicting contact points and post-contact trajectories. A0 is pre-trained on 1 million contact points data and fine-tuned on annotated trajectories, enabling generalization across platforms. Key components include Position Offset Attention for motion-aware feature extraction and a Spatial Information Aggregation Layer for precise coordinate mapping. The model's output is executed by the action execution module. Experiments on multiple robotic systems (Franka, Kinova, Realman, and Dobot) demonstrate A0's superior performance in complex tasks, showcasing its efficiency, flexibility, and real-world applicability.
>
---
#### [replaced 018] AnyTask: an Automated Task and Data Generation Framework for Advancing Sim-to-Real Policy Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AnyTask框架，解决机器人学习中数据不足与仿真到现实迁移困难的问题，通过自动化生成任务和数据提升策略泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.17853v2](https://arxiv.org/pdf/2512.17853v2)**

> **作者:** Ran Gong; Xiaohan Zhang; Jinghuan Shang; Maria Vittoria Minniti; Jigarkumar Patel; Valerio Pepe; Riedana Yan; Ahmet Gundogdu; Ivan Kapelyukh; Ali Abbas; Xiaoqiang Yan; Harsh Patel; Laura Herlant; Karl Schmeckpeper
>
> **备注:** 28 pages, 25 figures. The first four authors contributed equally
>
> **摘要:** Generalist robot learning remains constrained by data: large-scale, diverse, and high-quality interaction data are expensive to collect in the real world. While simulation has become a promising way for scaling up data collection, the related tasks, including simulation task design, task-aware scene generation, expert demonstration synthesis, and sim-to-real transfer, still demand substantial human effort. We present AnyTask, an automated framework that pairs massively parallel GPU simulation with foundation models to design diverse manipulation tasks and synthesize robot data. We introduce three AnyTask agents for generating expert demonstrations aiming to solve as many tasks as possible: 1) ViPR, a novel task and motion planning agent with VLM-in-the-loop Parallel Refinement; 2) ViPR-Eureka, a reinforcement learning agent with generated dense rewards and LLM-guided contact sampling; 3) ViPR-RL, a hybrid planning and learning approach that jointly produces high-quality demonstrations with only sparse rewards. We train behavior cloning policies on generated data, validate them in simulation, and deploy them directly on real robot hardware. The policies generalize to novel object poses, achieving 44% average success across a suite of real-world pick-and-place, drawer opening, contact-rich pushing, and long-horizon manipulation tasks. Our project website is at https://anytask.rai-inst.com .
>
---
#### [replaced 019] From Human Bias to Robot Choice: How Occupational Contexts and Racial Priming Shape Robot Selection
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究人类偏见如何影响对机器人选择，属于社会心理学与人机交互任务，旨在揭示职业情境和种族提示对机器人选择的影响。**

- **链接: [https://arxiv.org/pdf/2512.20951v3](https://arxiv.org/pdf/2512.20951v3)**

> **作者:** Jiangen He; Wanqi Zhang; Jessica Barfield
>
> **备注:** HRI '26
>
> **摘要:** As artificial agents increasingly integrate into professional environments, fundamental questions have emerged about how societal biases influence human-robot selection decisions. We conducted two comprehensive experiments (N = 1,038) examining how occupational contexts and stereotype activation shape robotic agent choices across construction, healthcare, educational, and athletic domains. Participants made selections from artificial agents that varied systematically in skin tone and anthropomorphic characteristics. Our study revealed distinct context-dependent patterns. Healthcare and educational scenarios demonstrated strong favoritism toward lighter-skinned artificial agents, while construction and athletic contexts showed greater acceptance of darker-toned alternatives. Participant race was associated with systematic differences in selection patterns across professional domains. The second experiment demonstrated that exposure to human professionals from specific racial backgrounds systematically shifted later robotic agent preferences in stereotype-consistent directions. These findings show that occupational biases and color-based discrimination transfer directly from human-human to human-robot evaluation contexts. The results highlight mechanisms through which robotic deployment may unintentionally perpetuate existing social inequalities.
>
---
#### [replaced 020] Large Multimodal Models for Embodied Intelligent Driving: The Next Frontier in Self-Driving?
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决传统模块化设计在开放场景中的不足。通过融合大模态模型与深度强化学习，提出一种语义与策略双驱动框架，提升决策能力和持续学习效果。**

- **链接: [https://arxiv.org/pdf/2601.08434v3](https://arxiv.org/pdf/2601.08434v3)**

> **作者:** Long Zhang; Yuchen Xia; Bingqing Wei; Zhen Liu; Shiwen Mao; Zhu Han; Mohsen Guizani
>
> **摘要:** The advent of Large Multimodal Models (LMMs) offers a promising technology to tackle the limitations of modular design in autonomous driving, which often falters in open-world scenarios requiring sustained environmental understanding and logical reasoning. Besides, embodied artificial intelligence facilitates policy optimization through closed-loop interactions to achieve the continuous learning capability, thereby advancing autonomous driving toward embodied intelligent (El) driving. However, such capability will be constrained by relying solely on LMMs to enhance EI driving without joint decision-making. This article introduces a novel semantics and policy dual-driven hybrid decision framework to tackle this challenge, ensuring continuous learning and joint decision. The framework merges LMMs for semantic understanding and cognitive representation, and deep reinforcement learning (DRL) for real-time policy optimization. We start by introducing the foundational principles of EI driving and LMMs. Moreover, we examine the emerging opportunities this framework enables, encompassing potential benefits and representative use cases. A case study is conducted experimentally to validate the performance superiority of our framework in completing lane-change planning task. Finally, several future research directions to empower EI driving are identified to guide subsequent work.
>
---
#### [replaced 021] FlyPose: Towards Robust Human Pose Estimation From Aerial Views
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人体姿态估计任务，解决从空中视角准确检测和估计人体姿态的问题。通过多数据集训练提升性能，并发布新数据集FlyPose-104。**

- **链接: [https://arxiv.org/pdf/2601.05747v2](https://arxiv.org/pdf/2601.05747v2)**

> **作者:** Hassaan Farooq; Marvin Brenner; Peter Stütz
>
> **备注:** 11 pages, 9 figures, IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are increasingly deployed in close proximity to humans for applications such as parcel delivery, traffic monitoring, disaster response and infrastructure inspections. Ensuring safe and reliable operation in these human-populated environments demands accurate perception of human poses and actions from an aerial viewpoint. This perspective challenges existing methods with low resolution, steep viewing angles and (self-)occlusion, especially if the application demands realtime feasibile models. We train and deploy FlyPose, a lightweight top-down human pose estimation pipeline for aerial imagery. Through multi-dataset training, we achieve an average improvement of 6.8 mAP in person detection across the test-sets of Manipal-UAV, VisDrone, HIT-UAV as well as our custom dataset. For 2D human pose estimation we report an improvement of 16.3 mAP on the challenging UAV-Human dataset. FlyPose runs with an inference latency of ~20 milliseconds including preprocessing on a Jetson Orin AGX Developer Kit and is deployed onboard a quadrotor UAV during flight experiments. We also publish FlyPose-104, a small but challenging aerial human pose estimation dataset, that includes manual annotations from difficult aerial perspectives: https://github.com/farooqhassaan/FlyPose.
>
---
#### [replaced 022] Towards Accessible Robot Control: Comparing Kinesthetic Teaching, SpaceMouse Teleoperation, and a Mixed Reality Interface
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决非专家用户在机器人控制中的性能差距问题。通过比较三种控制方式，评估其在复杂任务中的表现与用户体验。**

- **链接: [https://arxiv.org/pdf/2409.18394v3](https://arxiv.org/pdf/2409.18394v3)**

> **作者:** Aliyah Smith; Monroe Kennedy
>
> **备注:** 32 pages, 12 figures
>
> **摘要:** Teleoperation interfaces are essential tools for enabling human control of robotic systems. Although a wide range of interfaces has been developed, a persistent gap remains between the level of performance humans can achieve through these interfaces and the capabilities afforded by direct human-guided robot control. This gap is further exacerbated when users are inexperienced or unfamiliar with the robotic platform or control interface. In this work, we aim to better characterize this performance gap for non-expert users by comparing two teleoperation approaches, SpaceMouse teleoperation and a Mixed Reality (MR) interface, against kinesthetic teaching as a non-teleoperation baseline. All three approaches were evaluated in a comprehensive user study involving two robotic platforms and six complex manipulation tasks. Quantitative results show that the SpaceMouse and MR interfaces performed comparably, with significant differences in task completion observed for only two tasks, and success rates declining as task complexity increased. Qualitative analysis reflected these trends, highlighting differences in Physical Demand and identifying interface attributes that influence users' ability to perform, learn, and understand. This study quantifies the limitations of current teleoperation methods and incorporates subjective feedback from 25 participants. The results highlight the critical need to design and rigorously evaluate teleoperation systems for non-expert users, particularly in contexts where autonomous robots are deployed in personal or everyday environments, to ensure usability, efficiency, and accessibility.
>
---
#### [replaced 023] Can the Waymo Open Motion Dataset Support Realistic Behavioral Modeling? A Validation Study with Naturalistic Trajectories
- **分类: cs.RO; cs.AI; cs.LG; eess.SY; stat.AP**

- **简介: 该论文属于行为建模任务，旨在验证Waymo数据集是否适合真实驾驶行为分析。通过对比独立数据集，发现WOMD存在偏差，需谨慎使用。**

- **链接: [https://arxiv.org/pdf/2509.03515v2](https://arxiv.org/pdf/2509.03515v2)**

> **作者:** Yanlin Zhang; Sungyong Chung; Nachuan Li; Dana Monzer; Hani S. Mahmassani; Samer H. Hamdar; Alireza Talebpour
>
> **摘要:** The Waymo Open Motion Dataset (WOMD) has become a popular resource for data-driven modeling of autonomous vehicles (AVs) behavior. However, its validity for behavioral analysis remains uncertain due to proprietary post-processing, the absence of error quantification, and the segmentation of trajectories into 20-second clips. This study examines whether WOMD accurately captures the dynamics and interactions observed in real-world AV operations. Leveraging an independently collected naturalistic dataset from Level 4 AV operations in Phoenix, Arizona (PHX), we perform comparative analyses across three representative urban driving scenarios: discharging at signalized intersections, car-following, and lane-changing behaviors. For the discharging analysis, headways are manually extracted from aerial video to ensure negligible measurement error. For the car-following and lane-changing cases, we apply the Simulation-Extrapolation (SIMEX) method to account for empirically estimated error in the PHX data and use Dynamic Time Warping (DTW) distances to quantify behavioral differences. Results across all scenarios consistently show that behavior in PHX falls outside the behavioral envelope of WOMD. Notably, WOMD underrepresents short headways and abrupt decelerations. These findings suggest that behavioral models calibrated solely on WOMD may systematically underestimate the variability, risk, and complexity of naturalistic driving. Caution is therefore warranted when using WOMD for behavior modeling without proper validation against independently collected data.
>
---
#### [replaced 024] Generation of Real-time Robotic Emotional Expressions Learning from Human Demonstration in Mixed Reality
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于机器人情感表达生成任务，旨在通过混合现实捕捉人类示范，实现机器人实时、多样化的表情生成。**

- **链接: [https://arxiv.org/pdf/2508.08999v2](https://arxiv.org/pdf/2508.08999v2)**

> **作者:** Chao Wang; Michael Gienger; Fan Zhang
>
> **备注:** 5
>
> **摘要:** Expressive behaviors in robots are critical for effectively conveying their emotional states during interactions with humans. In this work, we present a framework that autonomously generates realistic and diverse robotic emotional expressions based on expert human demonstrations captured in Mixed Reality (MR). Our system enables experts to teleoperate a virtual robot from a first-person perspective, capturing their facial expressions, head movements, and upper-body gestures, and mapping these behaviors onto corresponding robotic components including eyes, ears, neck, and arms. Leveraging a flow-matching-based generative process, our model learns to produce coherent and varied behaviors in real-time in response to moving objects, conditioned explicitly on given emotional states. A preliminary test validated the effectiveness of our approach for generating autonomous expressions.
>
---
#### [replaced 025] MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control
- **分类: cs.GR; cs.LG; cs.RO**

- **简介: 论文提出MimicKit，一个用于运动模仿与控制的强化学习框架，解决计算机图形学和机器人领域的运动控制问题。该框架提供模块化工具和标准化结构，支持研究与应用。**

- **链接: [https://arxiv.org/pdf/2510.13794v4](https://arxiv.org/pdf/2510.13794v4)**

> **作者:** Xue Bin Peng
>
> **摘要:** MimicKit is an open-source framework for training motion controllers using motion imitation and reinforcement learning. The codebase provides implementations of commonly-used motion-imitation techniques and RL algorithms. This framework is intended to support research and applications in computer graphics and robotics by providing a unified training framework, along with standardized environment, agent, and data structures. The codebase is designed to be modular and easily configurable, enabling convenient modification and extension to new characters and tasks. The open-source codebase is available at: https://github.com/xbpeng/MimicKit.
>
---
#### [replaced 026] Gauss-Newton accelerated MPPI Control
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制领域，针对MPPI在高维问题中效率低下的问题，提出Gauss-Newton加速的MPPI方法，提升其可扩展性和计算效率。**

- **链接: [https://arxiv.org/pdf/2512.04579v2](https://arxiv.org/pdf/2512.04579v2)**

> **作者:** Hannes Homburger; Katrin Baumgärtner; Moritz Diehl; Johannes Reuter
>
> **备注:** 6 pages, 3 figures, submitted to the IFAC World Congress 2026
>
> **摘要:** Model Predictive Path Integral (MPPI) control is a sampling-based optimization method that has recently attracted attention, particularly in the robotics and reinforcement learning communities. MPPI has been widely applied as a GPU-accelerated random search method to deterministic direct single-shooting optimal control problems arising in model predictive control (MPC) formulations. MPPI offers several key advantages, including flexibility, robustness, ease of implementation, and inherent parallelizability. However, its performance can deteriorate in high-dimensional settings since the optimal control problem is solved via Monte Carlo sampling. To address this limitation, this paper proposes an enhanced MPPI method that incorporates a Jacobian reconstruction technique and the second-order Generalized Gauss-Newton method. This novel approach is called \textit{Gauss-Newton accelerated MPPI}. The numerical results show that the Gauss-Newton accelerated MPPI approach substantially improves MPPI scalability and computational efficiency while preserving the key benefits of the classical MPPI framework, making it a promising approach even for high-dimensional problems.
>
---
#### [replaced 027] Robotic Tele-Operation for Upper Aerodigestive Tract Microsurgery: System Design and Validation
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决UADT手术中手动操作力臂的局限性。设计了新型末端执行器和远程操控系统，提升手术精度与医生操作舒适度。**

- **链接: [https://arxiv.org/pdf/2601.06617v3](https://arxiv.org/pdf/2601.06617v3)**

> **作者:** Giovani Braglia; José Jair Alves Mendes Junior; Augusto Tetsuo Prado Inafuco; Federico Mariano; Leonardo S. Mattos
>
> **摘要:** Upper aerodigestive tract (UADT) treatments frequently employ transoral laser microsurgery (TLM) for procedures such as the removal of tumors or polyps. In TLM, a laser beam is used to cut target tissue, while forceps are employed to grasp, manipulate, and stabilize tissue within the UADT. Although TLM systems may rely on different technologies and interfaces, forceps manipulation is still predominantly performed manually, introducing limitations in ergonomics, precision, and controllability. This paper proposes a novel robotic system for tissue manipulation in UADT procedures, based on a novel end-effector designed for forceps control. The system is integrated within a teleoperation framework that employs a robotic manipulator with a programmed remote center of motion (RCM), enabling precise and constrained instrument motion while improving surgeon ergonomics. The proposed approach is validated through two experimental studies and a dedicated usability evaluation, demonstrating its effectiveness and suitability for UADT surgical applications.
>
---
#### [replaced 028] Learning with pyCub: A Simulation and Exercise Framework for Humanoid Robotics
- **分类: cs.RO**

- **简介: 论文提出pyCub，一个基于Python的人形机器人仿真与教学框架，解决传统仿真需C++和YARP的问题，提供易用的机器人控制练习。**

- **链接: [https://arxiv.org/pdf/2506.01756v2](https://arxiv.org/pdf/2506.01756v2)**

> **作者:** Lukas Rustler; Matej Hoffmann
>
> **备注:** Submitted for RiE 2026
>
> **摘要:** We present pyCub, an open-source physics-based simulation of the humanoid robot iCub, along with exercises to teach students the basics of humanoid robotics. Compared to existing iCub simulators (iCub SIM, iCub Gazebo), which require C++ code and YARP as middleware, pyCub works without YARP and with Python code. The complete robot with all articulations has been simulated, with two cameras in the eyes and the unique sensitive skin of the iCub comprising 4000 receptors on its body surface. The exercises range from basic control of the robot in velocity, joint, and Cartesian space to more complex tasks like gazing, grasping, or reactive control. The whole framework is written and controlled with Python, thus allowing to be used even by people with small or almost no programming practice. The exercises can be scaled to different difficulty levels. We tested the framework in two runs of a course on humanoid robotics. The simulation, exercises, documentation, Docker images, and example videos are publicly available at https://rustlluk.github.io/pyCub.
>
---
#### [replaced 029] Prespecified-Performance Kinematic Tracking Control for Aerial Manipulation
- **分类: cs.RO**

- **简介: 该论文属于空中机械臂的运动控制任务，旨在解决传统方法无法在预定时间内精确跟踪的问题。提出一种结合预设轨迹和二次规划的新控制框架，确保末端执行器在限定时间内准确到达目标位置。**

- **链接: [https://arxiv.org/pdf/2509.10065v3](https://arxiv.org/pdf/2509.10065v3)**

> **作者:** Huazi Cao; Jiahao Shen; Zhengzhen Li; Qinquan Ren; Shiyu Zhao
>
> **摘要:** This paper studies the kinematic tracking control problem for aerial manipulators. Existing kinematic tracking control methods, which typically employ proportional-derivative feedback or tracking-error-based feedback strategies, may fail to achieve tracking objectives within specified time constraints. To address this limitation, we propose a novel control framework comprising two key components: end-effector tracking control based on a user-defined preset trajectory and quadratic programming-based reference allocation. Compared with state-of-the-art approaches, the proposed method has several attractive features. First, it ensures that the end-effector reaches the desired position within a preset time while keeping the tracking error within a performance envelope that reflects task requirements. Second, quadratic programming is employed to allocate the references of the quadcopter base and the Delta arm, while considering the physical constraints of the aerial manipulator, thus preventing solutions that may violate physical limitations. The proposed approach is validated through three experiments. Experimental results demonstrate the effectiveness of the proposed algorithm and its capability to guarantee that the target position is reached within the preset time.
>
---
#### [replaced 030] DAPPER: Discriminability-Aware Policy-to-Policy Preference-Based Reinforcement Learning for Query-Efficient Robot Skill Acquisition
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决PbRL查询效率低的问题。通过引入偏好可区分性，利用多策略比较提升轨迹多样性，提出DAPPER方法以提高学习效率。**

- **链接: [https://arxiv.org/pdf/2505.06357v3](https://arxiv.org/pdf/2505.06357v3)**

> **作者:** Yuki Kadokawa; Jonas Frey; Takahiro Miki; Takamitsu Matsubara; Marco Hutter
>
> **备注:** Accepted for IEEE Robotics & Automation Magazine (RAM)
>
> **摘要:** Preference-based Reinforcement Learning (PbRL) enables policy learning through simple queries comparing trajectories from a single policy. While human responses to these queries make it possible to learn policies aligned with human preferences, PbRL suffers from low query efficiency, as policy bias limits trajectory diversity and reduces the number of discriminable queries available for learning preferences. This paper identifies preference discriminability, which quantifies how easily a human can judge which trajectory is closer to their ideal behavior, as a key metric for improving query efficiency. To address this, we move beyond comparisons within a single policy and instead generate queries by comparing trajectories from multiple policies, as training them from scratch promotes diversity without policy bias. We propose Discriminability-Aware Policy-to-Policy Preference-Based Efficient Reinforcement Learning (DAPPER), which integrates preference discriminability with trajectory diversification achieved by multiple policies. DAPPER trains new policies from scratch after each reward update and employs a discriminator that learns to estimate preference discriminability, enabling the prioritized sampling of more discriminable queries. During training, it jointly maximizes the preference reward and preference discriminability score, encouraging the discovery of highly rewarding and easily distinguishable policies. Experiments in simulated and real-world legged robot environments demonstrate that DAPPER outperforms previous methods in query efficiency, particularly under challenging preference discriminability conditions. A supplementary video that facilitates understanding of the proposed framework and its experimental results is available at: https://youtu.be/lRwX8FNN8n4
>
---
#### [replaced 031] A Two-Stage Reactive Auction Framework for the Multi-Depot Rural Postman Problem with Dynamic Vehicle Failures
- **分类: cs.RO; cs.CC; cs.MA**

- **简介: 该论文研究多仓库农村邮递员问题中的车辆故障实时调度，提出两阶段拍卖框架以提升应急调度效率与质量。**

- **链接: [https://arxiv.org/pdf/2411.04073v2](https://arxiv.org/pdf/2411.04073v2)**

> **作者:** Eashwar Sathyamurthy; Jeffrey W. Herrmann; Shapour Azarm
>
> **摘要:** Although unmanned vehicle fleets offer efficiency in transportation, logistics and inspection, their susceptibility to failures poses a significant challenge to mission continuity. We study the Multi-Depot Rural Postman Problem with Rechargeable and Reusable Vehicles (MD-RPP-RRV) with vehicle failures, where unmanned rechargeable vehicles placed at multiple depots with capacity constraints may fail while serving arc-based demands. To address unexpected vehicle breakdowns during operation, we propose a two-stage real-time rescheduling framework. First, a centralized auction quickly generates a feasible rescheduling solution; for this stage, we derive a theoretical additive bound that establishes an analytical guarantee on the worst-case rescheduling penalty. Second, a peer auction refines this baseline through a problem-specific magnetic field router for local schedule repair, utilizing parameters calibrated via sensitivity analysis to ensure controlled computational growth. We benchmark this approach against a simulated annealing metaheuristic to evaluate solution quality and execution speed. Experimental results on 257 diverse failure scenarios demonstrate that the framework achieves an average runtime reduction of over 95\% relative to the metaheuristic baseline, cutting rescheduling times from hours to seconds while maintaining high solution quality. The two-stage framework excels on large-scale instances, surpassing the centralized auction in nearly 80\% of scenarios with an average solution improvement exceeding 12\%. Moreover, it outperforms the simulated annealing mean and best results in 59\% and 28\% of scenarios, respectively, offering the robust speed-quality trade-off required for real-time mission continuity.
>
---
#### [replaced 032] Monotone Subsystem Decomposition for Efficient Multi-Objective Robot Design
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，解决多目标组件选择问题。通过单调子系统分解方法，高效计算帕累托最优解，提升设计效率与规模。**

- **链接: [https://arxiv.org/pdf/2505.11624v2](https://arxiv.org/pdf/2505.11624v2)**

> **作者:** Andrew Wilhelm; Nils Napp
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2025
>
> **摘要:** Automating design minimizes errors, accelerates the design process, and reduces cost. However, automating robot design is challenging due to recursive constraints, multiple design objectives, and cross-domain design complexity possibly spanning multiple abstraction layers. Here we look at the problem of component selection, a combinatorial optimization problem in which a designer, given a robot model, must select compatible components from an extensive catalog. The goal is to satisfy high-level task specifications while optimally balancing trade-offs between competing design objectives. In this paper, we extend our previous constraint programming approach to multi-objective design problems and propose the novel technique of monotone subsystem decomposition to efficiently compute a Pareto front of solutions for large-scale problems. We prove that subsystems can be optimized for their Pareto fronts and, under certain conditions, these results can be used to determine a globally optimal Pareto front. Furthermore, subsystems serve as an intuitive design abstraction and can be reused across various design problems. Using an example quadcopter design problem, we compare our method to a linear programming approach and demonstrate our method scales better for large catalogs, solving a multi-objective problem of 10^25 component combinations in seconds. We then expand the original problem and solve a task-oriented, multi-objective design problem to build a fleet of quadcopters to deliver packages. We compute a Pareto front of solutions in seconds where each solution contains an optimal component-level design and an optimal package delivery schedule for each quadcopter.
>
---
#### [replaced 033] PolyFly: Polytopic Optimal Planning for Collision-Free Cable-Suspended Aerial Payload Transportation
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决空中吊运系统在复杂环境中的高效避障问题。通过建模各物理组件为独立多面体，提升规划精度与速度。**

- **链接: [https://arxiv.org/pdf/2510.15226v2](https://arxiv.org/pdf/2510.15226v2)**

> **作者:** Mrunal Sarvaiya; Guanrui Li; Giuseppe Loianno
>
> **摘要:** Aerial transportation robots using suspended cables have emerged as versatile platforms for disaster response and rescue operations. To maximize the capabilities of these systems, robots need to aggressively fly through tightly constrained environments, such as dense forests and structurally unsafe buildings, while minimizing flight time and avoiding obstacles. Existing methods geometrically over-approximate the vehicle and obstacles, leading to conservative maneuvers and increased flight times. We eliminate these restrictions by proposing PolyFly, an optimal global planner which considers a non-conservative representation for aerial transportation by modeling each physical component of the environment, and the robot (quadrotor, cable and payload), as independent polytopes. We further increase the model accuracy by incorporating the attitude of the physical components by constructing orientation-aware polytopes. The resulting optimal control problem is efficiently solved by converting the polytope constraints into smooth differentiable constraints via duality theory. We compare our method against the existing state-of-the-art approach in eight maze-like environments and show that PolyFly produces faster trajectories in each scenario. We also experimentally validate our proposed approach on a real quadrotor with a suspended payload, demonstrating the practical reliability and accuracy of our method.
>
---
#### [replaced 034] Reflection-Based Task Adaptation for Self-Improving VLA
- **分类: cs.RO**

- **简介: 该论文属于机器人任务适应领域，解决VLA模型在新任务中高效适应的问题。提出Reflective Self-Adaptation框架，通过双路径机制实现快速、自主学习与优化。**

- **链接: [https://arxiv.org/pdf/2510.12710v2](https://arxiv.org/pdf/2510.12710v2)**

> **作者:** Baicheng Li; Dong Wu; Zike Yan; Xinchen Liu; Zecui Zeng; Lusong Li; Hongbin Zha
>
> **摘要:** Pre-trained Vision-Language-Action (VLA) models represent a major leap towards general-purpose robots, yet efficiently adapting them to novel, specific tasks in-situ remains a significant hurdle. While reinforcement learning (RL) is a promising avenue for such adaptation, the process often suffers from low efficiency, hindering rapid task mastery. We introduce Reflective Self-Adaptation, a framework for rapid, autonomous task adaptation without human intervention. Our framework establishes a self-improving loop where the agent learns from its own experience to enhance both strategy and execution. The core of our framework is a dual-pathway architecture that addresses the full adaptation lifecycle. First, a Failure-Driven Reflective RL pathway enables rapid learning by using the VLM's causal reasoning to automatically synthesize a targeted, dense reward function from failure analysis. This provides a focused learning signal that significantly accelerates policy exploration. However, optimizing such proxy rewards introduces a potential risk of "reward hacking," where the agent masters the reward function but fails the actual task. To counteract this, our second pathway, Success-Driven Quality-Guided SFT, grounds the policy in holistic success. It identifies and selectively imitates high-quality successful trajectories, ensuring the agent remains aligned with the ultimate task goal. This pathway is strengthened by a conditional curriculum mechanism to aid initial exploration. We conduct experiments in challenging manipulation tasks. The results demonstrate that our framework achieves faster convergence and higher final success rates compared to representative baselines. Our work presents a robust solution for creating self-improving agents that can efficiently and reliably adapt to new environments.
>
---
#### [replaced 035] Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人抓取任务，解决无先验知识下多指手的灵活抓取问题。通过结合形状补全与抓取预测，实现快速高精度抓取。**

- **链接: [https://arxiv.org/pdf/2310.20350v2](https://arxiv.org/pdf/2310.20350v2)**

> **作者:** Matthias Humt; Dominik Winkelbauer; Ulrich Hillenbrand; Berthold Bäuml
>
> **备注:** 8 pages, 10 figures, 3 tables, 1 algorithm. Published in Humanoids 2023. Project page: https://aidx-lab.org/grasping/humanoids23
>
> **摘要:** Grasping objects with limited or no prior knowledge about them is a highly relevant skill in assistive robotics. Still, in this general setting, it has remained an open problem, especially when it comes to only partial observability and versatile grasping with multi-fingered hands. We present a novel, fast, and high fidelity deep learning pipeline consisting of a shape completion module that is based on a single depth image, and followed by a grasp predictor that is based on the predicted object shape. The shape completion network is based on VQDIF and predicts spatial occupancy values at arbitrary query points. As grasp predictor, we use our two-stage architecture that first generates hand poses using an autoregressive model and then regresses finger joint configurations per pose. Critical factors turn out to be sufficient data realism and augmentation, as well as special attention to difficult cases during training. Experiments on a physical robot platform demonstrate successful grasping of a wide range of household objects based on a depth image from a single viewpoint. The whole pipeline is fast, taking only about 1 s for completing the object's shape (0.7 s) and generating 1000 grasps (0.3 s).
>
---
#### [replaced 036] Genie Centurion: Accelerating Scalable Real-World Robot Training with Human Rewind-and-Refine Guidance
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决真实场景下机器人策略训练数据收集成本高、效率低的问题。提出GCENT框架，通过人类回退与修正指导提升训练效率。**

- **链接: [https://arxiv.org/pdf/2505.18793v2](https://arxiv.org/pdf/2505.18793v2)**

> **作者:** Wenhao Wang; Jianheng Song; Chiming Liu; Jiayao Ma; Siyuan Feng; Jingyuan Wang; Yuxin Jiang; Kylin Chen; Sikang Zhan; Yi Wang; Tong Meng; Modi Shi; Xindong He; Guanghui Ren; Yang Yang; Maoqing Yao
>
> **摘要:** While Vision-Language-Action (VLA) models show strong generalizability in various tasks, real-world deployment of robotic policy still requires large-scale, high-quality human expert demonstrations. However, data collection via human teleoperation requires continuous operator attention, which is costly, hard to scale. To address this, we propose Genie Centurion (GCENT), a scalable and general data collection paradigm based on human rewind-and-refine guidance, enabling robots' interactive learning in deployment. GCENT starts at an imperfect policy and improves over time. When the robot execution failures occur, GCENT allows robots to revert to a previous state with a rewind mechanism, after which a teleoperator provides corrective demonstrations to refine the policy. This framework supports a one-human-to-many-robots supervision scheme with a Task Sentinel module, which autonomously predicts task success and solicits human intervention when necessary. Empirical results show that GCENT achieves up to 40% higher task success rates than state-of-the-art data collection methods, and reaches comparable performance using less than half the data in long-horizon and precise tasks. We also quantify the data yield-to-effort ratio under multi-robot scenarios, demonstrating GCENT's potential for scalable and cost-efficient robot policy training in real-world environments.
>
---
#### [replaced 037] Astra: Efficient Transformer Architecture and Contrastive Dynamics Learning for Embodied Instruction Following
- **分类: cs.RO**

- **简介: 该论文属于具身指令跟随任务，解决多模态序列处理效率问题。提出Astra架构与对比动态学习，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2408.01147v2](https://arxiv.org/pdf/2408.01147v2)**

> **作者:** Yueen Ma; Dafeng Chi; Shiguang Wu; Yuecheng Liu; Yuzheng Zhuang; Irwin King
>
> **备注:** Accepted to EMNLP 2025 (main). Published version: https://aclanthology.org/2025.emnlp-main.688/ Code available at: https://github.com/yueen-ma/Astra
>
> **摘要:** Vision-language-action models have gained significant attention for their ability to model multimodal sequences in embodied instruction following tasks. However, most existing models rely on causal attention, which we find suboptimal for processing sequences composed of interleaved segments from different modalities. In this paper, we introduce Astra, a novel Transformer architecture featuring trajectory attention and learnable action queries, designed to efficiently process segmented multimodal trajectories and predict actions for imitation learning. Furthermore, we propose a contrastive dynamics learning objective to enhance the model's understanding of environment dynamics and multimodal alignment, complementing the primary behavior cloning objective. Through extensive experiments on three large-scale robot manipulation benchmarks, Astra demonstrates substantial performance improvements over previous models.
>
---
