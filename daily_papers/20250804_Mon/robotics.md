# 机器人 cs.RO

- **最新发布 25 篇**

- **更新 26 篇**

## 最新发布

#### [new 001] OmniUnet: A Multimodal Network for Unstructured Terrain Segmentation on Planetary Rovers Using RGB, Depth, and Thermal Imagery
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出 OmniUnet 作为多模态网络用于火星探测器对复杂地形的语义分割，解决无结构环境下的导航安全问题，通过RGB、深度和热成像数据训练模型并验证其性能，开发了专用硬件实现，取得80.37%的像素准确率和673ms推理速度。**

- **链接: [http://arxiv.org/pdf/2508.00580v1](http://arxiv.org/pdf/2508.00580v1)**

> **作者:** Raul Castilla-Arquillo; Carlos Perez-del-Pulgar; Levin Gerdes; Alfonso Garcia-Cerezo; Miguel A. Olivares-Mendez
>
> **摘要:** Robot navigation in unstructured environments requires multimodal perception systems that can support safe navigation. Multimodality enables the integration of complementary information collected by different sensors. However, this information must be processed by machine learning algorithms specifically designed to leverage heterogeneous data. Furthermore, it is necessary to identify which sensor modalities are most informative for navigation in the target environment. In Martian exploration, thermal imagery has proven valuable for assessing terrain safety due to differences in thermal behaviour between soil types. This work presents OmniUnet, a transformer-based neural network architecture for semantic segmentation using RGB, depth, and thermal (RGB-D-T) imagery. A custom multimodal sensor housing was developed using 3D printing and mounted on the Martian Rover Testbed for Autonomy (MaRTA) to collect a multimodal dataset in the Bardenas semi-desert in northern Spain. This location serves as a representative environment of the Martian surface, featuring terrain types such as sand, bedrock, and compact soil. A subset of this dataset was manually labeled to support supervised training of the network. The model was evaluated both quantitatively and qualitatively, achieving a pixel accuracy of 80.37% and demonstrating strong performance in segmenting complex unstructured terrain. Inference tests yielded an average prediction time of 673 ms on a resource-constrained computer (Jetson Orin Nano), confirming its suitability for on-robot deployment. The software implementation of the network and the labeled dataset have been made publicly available to support future research in multimodal terrain perception for planetary robotics.
>
---
#### [new 002] On Learning Closed-Loop Probabilistic Multi-Agent Simulator
- **分类: cs.RO**

- **简介: 该论文提出了一种基于概率的多智能体仿真框架NIVA，解决复杂交通场景下的闭环模拟问题，通过将序列生成模型与贝叶斯推理相结合，实现了对车辆意图与驾驶风格的控制优化。**

- **链接: [http://arxiv.org/pdf/2508.00384v1](http://arxiv.org/pdf/2508.00384v1)**

> **作者:** Juanwu Lu; Rohit Gupta; Ahmadreza Moradipari; Kyungtae Han; Ruqi Zhang; Ziran Wang
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025. Source Code: https://github.com/juanwulu/niva
>
> **摘要:** The rapid iteration of autonomous vehicle (AV) deployments leads to increasing needs for building realistic and scalable multi-agent traffic simulators for efficient evaluation. Recent advances in this area focus on closed-loop simulators that enable generating diverse and interactive scenarios. This paper introduces Neural Interactive Agents (NIVA), a probabilistic framework for multi-agent simulation driven by a hierarchical Bayesian model that enables closed-loop, observation-conditioned simulation through autoregressive sampling from a latent, finite mixture of Gaussian distributions. We demonstrate how NIVA unifies preexisting sequence-to-sequence trajectory prediction models and emerging closed-loop simulation models trained on Next-token Prediction (NTP) from a Bayesian inference perspective. Experiments on the Waymo Open Motion Dataset demonstrate that NIVA attains competitive performance compared to the existing method while providing embellishing control over intentions and driving styles.
>
---
#### [new 003] SubCDM: Collective Decision-Making with a Swarm Subset
- **分类: cs.RO**

- **简介: 该论文旨在解决传统自主机器人群体决策中资源浪费的问题，通过子集决策机制实现动态优化，减少冗余机器人使用，提升效率。**

- **链接: [http://arxiv.org/pdf/2508.00467v1](http://arxiv.org/pdf/2508.00467v1)**

> **作者:** Samratul Fuady; Danesh Tarapore; Mohammad D. Soorati
>
> **备注:** 6 pages, 7 figures. This paper has been accepted for presentation at the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Collective decision-making is a key function of autonomous robot swarms, enabling them to reach a consensus on actions based on environmental features. Existing strategies require the participation of all robots in the decision-making process, which is resource-intensive and prevents the swarm from allocating the robots to any other tasks. We propose Subset-Based Collective Decision-Making (SubCDM), which enables decisions using only a swarm subset. The construction of the subset is dynamic and decentralized, relying solely on local information. Our method allows the swarm to adaptively determine the size of the subset for accurate decision-making, depending on the difficulty of reaching a consensus. Simulation results using one hundred robots show that our approach achieves accuracy comparable to using the entire swarm while reducing the number of robots required to perform collective decision-making, making it a resource-efficient solution for collective decision-making in swarm robotics.
>
---
#### [new 004] A Whole-Body Motion Imitation Framework from Human Data for Full-Size Humanoid Robot
- **分类: cs.RO**

- **简介: 该论文提出了一种基于人类数据的全尺寸 humanoid 机器人运动模仿框架，旨在解决骨骼与动态差异带来的运动准确性与平衡性问题。通过接触感知机制和非线性模型预测控制器，实现了高精度轨迹参考值并增强了外部扰动鲁棒性，验证了其在真实环境下的适应性与有效性。**

- **链接: [http://arxiv.org/pdf/2508.00362v1](http://arxiv.org/pdf/2508.00362v1)**

> **作者:** Zhenghan Chen; Haodong Zhang; Dongqi Wang; Jiyu Yu; Haocheng Xu; Yue Wang; Rong Xiong
>
> **摘要:** Motion imitation is a pivotal and effective approach for humanoid robots to achieve a more diverse range of complex and expressive movements, making their performances more human-like. However, the significant differences in kinematics and dynamics between humanoid robots and humans present a major challenge in accurately imitating motion while maintaining balance. In this paper, we propose a novel whole-body motion imitation framework for a full-size humanoid robot. The proposed method employs contact-aware whole-body motion retargeting to mimic human motion and provide initial values for reference trajectories, and the non-linear centroidal model predictive controller ensures the motion accuracy while maintaining balance and overcoming external disturbances in real time. The assistance of the whole-body controller allows for more precise torque control. Experiments have been conducted to imitate a variety of human motions both in simulation and in a real-world humanoid robot. These experiments demonstrate the capability of performing with accuracy and adaptability, which validates the effectiveness of our approach.
>
---
#### [new 005] On-Device Diffusion Transformer Policy for Efficient Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文旨在开发高效机器人操控的扩散策略框架LightDP，解决移动端部署时因计算效率低和内存占用大导致的应用瓶颈，通过网络压缩、采样步长优化及一致性蒸馏技术实现实时动作预测。**

- **链接: [http://arxiv.org/pdf/2508.00697v1](http://arxiv.org/pdf/2508.00697v1)**

> **作者:** Yiming Wu; Huan Wang; Zhenghao Chen; Jianxin Pang; Dong Xu
>
> **备注:** ICCV 2025
>
> **摘要:** Diffusion Policies have significantly advanced robotic manipulation tasks via imitation learning, but their application on resource-constrained mobile platforms remains challenging due to computational inefficiency and extensive memory footprint. In this paper, we propose LightDP, a novel framework specifically designed to accelerate Diffusion Policies for real-time deployment on mobile devices. LightDP addresses the computational bottleneck through two core strategies: network compression of the denoising modules and reduction of the required sampling steps. We first conduct an extensive computational analysis on existing Diffusion Policy architectures, identifying the denoising network as the primary contributor to latency. To overcome performance degradation typically associated with conventional pruning methods, we introduce a unified pruning and retraining pipeline, optimizing the model's post-pruning recoverability explicitly. Furthermore, we combine pruning techniques with consistency distillation to effectively reduce sampling steps while maintaining action prediction accuracy. Experimental evaluations on the standard datasets, \ie, PushT, Robomimic, CALVIN, and LIBERO, demonstrate that LightDP achieves real-time action prediction on mobile devices with competitive performance, marking an important step toward practical deployment of diffusion-based policies in resource-limited environments. Extensive real-world experiments also show the proposed LightDP can achieve performance comparable to state-of-the-art Diffusion Policies.
>
---
#### [new 006] TOP: Time Optimization Policy for Stable and Accurate Standing Manipulation with Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文提出了一种时间优化策略（TOP），用于提升人类体征学机器人在快速上肢动作中的稳定性和精度，解决了高维关节控制与鲁棒性不足的问题。通过分离上肢与下肢控制，并结合VAE训练方法，有效减少了快速上肢运动带来的平衡负担。**

- **链接: [http://arxiv.org/pdf/2508.00355v1](http://arxiv.org/pdf/2508.00355v1)**

> **作者:** Zhenghan Chen; Haocheng Xu; Haodong Zhang; Liang Zhang; He Li; Dongqi Wang; Jiyu Yu; Yifei Yang; Zhongxiang Zhou; Rong Xiong
>
> **摘要:** Humanoid robots have the potential capability to perform a diverse range of manipulation tasks, but this is based on a robust and precise standing controller. Existing methods are either ill-suited to precisely control high-dimensional upper-body joints, or difficult to ensure both robustness and accuracy, especially when upper-body motions are fast. This paper proposes a novel time optimization policy (TOP), to train a standing manipulation control model that ensures balance, precision, and time efficiency simultaneously, with the idea of adjusting the time trajectory of upper-body motions but not only strengthening the disturbance resistance of the lower-body. Our approach consists of three parts. Firstly, we utilize motion prior to represent upper-body motions to enhance the coordination ability between the upper and lower-body by training a variational autoencoder (VAE). Then we decouple the whole-body control into an upper-body PD controller for precision and a lower-body RL controller to enhance robust stability. Finally, we train TOP method in conjunction with the decoupled controller and VAE to reduce the balance burden resulting from fast upper-body motions that would destabilize the robot and exceed the capabilities of the lower-body RL policy. The effectiveness of the proposed approach is evaluated via both simulation and real world experiments, which demonstrate the superiority on standing manipulation tasks stably and accurately. The project page can be found at https://anonymous.4open.science/w/top-258F/.
>
---
#### [new 007] A control scheme for collaborative object transportation between a human and a quadruped robot using the MIGHTY suction cup
- **分类: cs.RO**

- **简介: 该论文提出了一种基于admittance控制的协作搬运方案，旨在通过四足机器人与人类协同操作时避免物体脱落，利用MIGHTY吸盘作为抓取和传感装置。研究解决了人机协作下物体稳定性与控制性能的平衡问题，并验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2508.00584v1](http://arxiv.org/pdf/2508.00584v1)**

> **作者:** Konstantinos Plotas; Emmanouil Papadakis; Drosakis Drosakis; Panos Trahanias; Dimitrios Papageorgiou
>
> **备注:** Please find the citation info @ Zenodo, ArXiv or Zenodo, as the proceedings of ICRA are no longer sent to IEEE Xplore
>
> **摘要:** In this work, a control scheme for human-robot collaborative object transportation is proposed, considering a quadruped robot equipped with the MIGHTY suction cup that serves both as a gripper for holding the object and a force/torque sensor. The proposed control scheme is based on the notion of admittance control, and incorporates a variable damping term aiming towards increasing the controllability of the human and, at the same time, decreasing her/his effort. Furthermore, to ensure that the object is not detached from the suction cup during the collaboration, an additional control signal is proposed, which is based on a barrier artificial potential. The proposed control scheme is proven to be passive and its performance is demonstrated through experimental evaluations conducted using the Unitree Go1 robot equipped with the MIGHTY suction cup.
>
---
#### [new 008] Topology-Inspired Morphological Descriptor for Soft Continuum Robots
- **分类: cs.RO**

- **简介: 该论文旨在开发一种基于拓扑启发的软连续体机器人形态描述方法，利用伪刚体模型与Morse理论结合，通过计数方向投影的临界点实现多模态配置的量化描述与分类，并将其应用于形态控制，优化目标形状参数以生成具有特定拓扑特征的平衡形态，提升其在医疗领域的精确性和适应性。**

- **链接: [http://arxiv.org/pdf/2508.00258v1](http://arxiv.org/pdf/2508.00258v1)**

> **作者:** Zhiwei Wu; Siyi Wei; Jiahao Luo; Jinhui Zhang
>
> **摘要:** This paper presents a topology-inspired morphological descriptor for soft continuum robots by combining a pseudo-rigid-body (PRB) model with Morse theory to achieve a quantitative characterization of robot morphologies. By counting critical points of directional projections, the proposed descriptor enables a discrete representation of multimodal configurations and facilitates morphological classification. Furthermore, we apply the descriptor to morphology control by formulating the target configuration as an optimization problem to compute actuation parameters that generate equilibrium shapes with desired topological features. The proposed framework provides a unified methodology for quantitative morphology description, classification, and control of soft continuum robots, with the potential to enhance their precision and adaptability in medical applications such as minimally invasive surgery and endovascular interventions.
>
---
#### [new 009] HannesImitation: Grasping with the Hannes Prosthetic Hand via Imitation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究了如何利用模仿学习控制Hannes手，解决了传统方法依赖手动示例的问题，提出了HannesImitationPolicy并构建了数据集，通过单个扩散模型实现了自主操作与动态调整，提升了手部控制能力。**

- **链接: [http://arxiv.org/pdf/2508.00491v1](http://arxiv.org/pdf/2508.00491v1)**

> **作者:** Carlo Alessi; Federico Vasile; Federico Ceola; Giulia Pasquale; Nicolò Boccardo; Lorenzo Natale
>
> **备注:** Paper accepted at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Recent advancements in control of prosthetic hands have focused on increasing autonomy through the use of cameras and other sensory inputs. These systems aim to reduce the cognitive load on the user by automatically controlling certain degrees of freedom. In robotics, imitation learning has emerged as a promising approach for learning grasping and complex manipulation tasks while simplifying data collection. Its application to the control of prosthetic hands remains, however, largely unexplored. Bridging this gap could enhance dexterity restoration and enable prosthetic devices to operate in more unconstrained scenarios, where tasks are learned from demonstrations rather than relying on manually annotated sequences. To this end, we present HannesImitationPolicy, an imitation learning-based method to control the Hannes prosthetic hand, enabling object grasping in unstructured environments. Moreover, we introduce the HannesImitationDataset comprising grasping demonstrations in table, shelf, and human-to-prosthesis handover scenarios. We leverage such data to train a single diffusion policy and deploy it on the prosthetic hand to predict the wrist orientation and hand closure for grasping. Experimental evaluation demonstrates successful grasps across diverse objects and conditions. Finally, we show that the policy outperforms a segmentation-based visual servo controller in unstructured scenarios. Additional material is provided on our project page: https://hsp-iit.github.io/HannesImitation
>
---
#### [new 010] Video Generators are Robot Policies
- **分类: cs.RO**

- **简介: 该论文旨在解决视觉运动策略泛化能力和数据效率问题，提出视频生成作为代理的方法，构建端到端的模块框架以提升鲁棒性和效率，实验表明其在模拟和真实场景中均优于传统行为克隆技术。**

- **链接: [http://arxiv.org/pdf/2508.00795v1](http://arxiv.org/pdf/2508.00795v1)**

> **作者:** Junbang Liang; Pavel Tokmakov; Ruoshi Liu; Sruthi Sudhakar; Paarth Shah; Rares Ambrus; Carl Vondrick
>
> **摘要:** Despite tremendous progress in dexterous manipulation, current visuomotor policies remain fundamentally limited by two challenges: they struggle to generalize under perceptual or behavioral distribution shifts, and their performance is constrained by the size of human demonstration data. In this paper, we use video generation as a proxy for robot policy learning to address both limitations simultaneously. We propose Video Policy, a modular framework that combines video and action generation that can be trained end-to-end. Our results demonstrate that learning to generate videos of robot behavior allows for the extraction of policies with minimal demonstration data, significantly improving robustness and sample efficiency. Our method shows strong generalization to unseen objects, backgrounds, and tasks, both in simulation and the real world. We further highlight that task success is closely tied to the generated video, with action-free video data providing critical benefits for generalizing to novel tasks. By leveraging large-scale video generative models, we achieve superior performance compared to traditional behavior cloning, paving the way for more scalable and data-efficient robot policy learning.
>
---
#### [new 011] OpenScout v1.1 mobile robot: a case study on open hardware continuation
- **分类: cs.RO**

- **简介: 该论文为OpenScout v1.1硬件改进提供案例研究，验证了扩展后的简化计算模块、ROS2/Gozebo仿真功能及实验结果的应用价值，旨在评估新型硬件对科研与工业的实际贡献。**

- **链接: [http://arxiv.org/pdf/2508.00625v1](http://arxiv.org/pdf/2508.00625v1)**

> **作者:** Bartosz Krawczyk; Ahmed Elbary; Robbie Cato; Jagdish Patil; Kaung Myat; Anyeh Ndi-Tah; Nivetha Sakthivel; Mark Crampton; Gautham Das; Charles Fox
>
> **备注:** 6 pages, 4 figures, a TAROS2025 short paper
>
> **摘要:** OpenScout is an Open Source Hardware (OSH) mobile robot for research and industry. It is extended to v1.1 which includes simplified, cheaper and more powerful onboard compute hardware; a simulated ROS2 interface; and a Gazebo simulation. Changes, their rationale, project methodology, and results are reported as an OSH case study.
>
---
#### [new 012] TopoDiffuser: A Diffusion-Based Multimodal Trajectory Prediction Model with Topometric Maps
- **分类: cs.RO**

- **简介: 该论文提出TopoDiffuser框架，解决多模态轨迹预测问题，通过融合LiDAR、历史运动与拓扑结构信息，利用扩散模型生成道路合规的未来轨迹，提升几何一致性并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.00303v1](http://arxiv.org/pdf/2508.00303v1)**

> **作者:** Zehui Xu; Junhui Wang; Yongliang Shi; Chao Gao; Guyue Zhou
>
> **摘要:** This paper introduces TopoDiffuser, a diffusion-based framework for multimodal trajectory prediction that incorporates topometric maps to generate accurate, diverse, and road-compliant future motion forecasts. By embedding structural cues from topometric maps into the denoising process of a conditional diffusion model, the proposed approach enables trajectory generation that naturally adheres to road geometry without relying on explicit constraints. A multimodal conditioning encoder fuses LiDAR observations, historical motion, and route information into a unified bird's-eye-view (BEV) representation. Extensive experiments on the KITTI benchmark demonstrate that TopoDiffuser outperforms state-of-the-art methods, while maintaining strong geometric consistency. Ablation studies further validate the contribution of each input modality, as well as the impact of denoising steps and the number of trajectory samples. To support future research, we publicly release our code at https://github.com/EI-Nav/TopoDiffuser.
>
---
#### [new 013] XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出XRoboToolkit框架，解决多平台机器人操作数据不足的问题，构建了基于OpenXR的标准系统，通过低延迟立体视觉、优化逆运动学和多种追踪模态，实现跨平台无缝集成与精度操控验证。**

- **链接: [http://arxiv.org/pdf/2508.00097v1](http://arxiv.org/pdf/2508.00097v1)**

> **作者:** Zhigen Zhao; Liuchuan Yu; Ke Jing; Ning Yang
>
> **备注:** 6 pages, 6 figures, project link: https://github.com/XR-Robotics
>
> **摘要:** The rapid advancement of Vision-Language-Action models has created an urgent need for large-scale, high-quality robot demonstration datasets. Although teleoperation is the predominant method for data collection, current approaches suffer from limited scalability, complex setup procedures, and suboptimal data quality. This paper presents XRoboToolkit, a cross-platform framework for extended reality based robot teleoperation built on the OpenXR standard. The system features low-latency stereoscopic visual feedback, optimization-based inverse kinematics, and support for diverse tracking modalities including head, controller, hand, and auxiliary motion trackers. XRoboToolkit's modular architecture enables seamless integration across robotic platforms and simulation environments, spanning precision manipulators, mobile robots, and dexterous hands. We demonstrate the framework's effectiveness through precision manipulation tasks and validate data quality by training VLA models that exhibit robust autonomous performance.
>
---
#### [new 014] Towards Data-Driven Adaptive Exoskeleton Assistance for Post-stroke Gait
- **分类: cs.RO**

- **简介: 该研究旨在开发基于数据驱动的辅助假肢，解决术后肌萎缩患者步态适应问题。通过训练多任务TCN模型，利用IMU数据和健康数据集，实现实时踝关节扭矩估计并验证原型可行性。**

- **链接: [http://arxiv.org/pdf/2508.00691v1](http://arxiv.org/pdf/2508.00691v1)**

> **作者:** Fabian C. Weigend; Dabin K. Choe; Santiago Canete; Conor J. Walsh
>
> **备注:** 8 pages, 6 figures, 2 tables
>
> **摘要:** Recent work has shown that exoskeletons controlled through data-driven methods can dynamically adapt assistance to various tasks for healthy young adults. However, applying these methods to populations with neuromotor gait deficits, such as post-stroke hemiparesis, is challenging. This is due not only to high population heterogeneity and gait variability but also to a lack of post-stroke gait datasets to train accurate models. Despite these challenges, data-driven methods offer a promising avenue for control, potentially allowing exoskeletons to function safely and effectively in unstructured community settings. This work presents a first step towards enabling adaptive plantarflexion and dorsiflexion assistance from data-driven torque estimation during post-stroke walking. We trained a multi-task Temporal Convolutional Network (TCN) using collected data from four post-stroke participants walking on a treadmill ($R^2$ of $0.74 \pm 0.13$). The model uses data from three inertial measurement units (IMU) and was pretrained on healthy walking data from 6 participants. We implemented a wearable prototype for our ankle torque estimation approach for exoskeleton control and demonstrated the viability of real-time sensing, estimation, and actuation with one post-stroke participant.
>
---
#### [new 015] Omni-Scan: Creating Visually-Accurate Digital Twin Object Models Using a Bimanual Robot with Handover and Gaussian Splat Merging
- **分类: cs.RO; cs.CV**

- **简介: 该论文旨在解决传统3DGS扫描受限于空间的问题，通过双手机器人实现高精度3D数字孪生建模，利用深度学习与GANs进行数据融合，并优化训练流程以支持多视角扫描，验证了 Omni-Scan 在工业缺陷检测中的有效性（100字）。**

- **链接: [http://arxiv.org/pdf/2508.00354v1](http://arxiv.org/pdf/2508.00354v1)**

> **作者:** Tianshuang Qiu; Zehan Ma; Karim El-Refai; Hiya Shah; Chung Min Kim; Justin Kerr; Ken Goldberg
>
> **摘要:** 3D Gaussian Splats (3DGSs) are 3D object models derived from multi-view images. Such "digital twins" are useful for simulations, virtual reality, marketing, robot policy fine-tuning, and part inspection. 3D object scanning usually requires multi-camera arrays, precise laser scanners, or robot wrist-mounted cameras, which have restricted workspaces. We propose Omni-Scan, a pipeline for producing high-quality 3D Gaussian Splat models using a bi-manual robot that grasps an object with one gripper and rotates the object with respect to a stationary camera. The object is then re-grasped by a second gripper to expose surfaces that were occluded by the first gripper. We present the Omni-Scan robot pipeline using DepthAny-thing, Segment Anything, as well as RAFT optical flow models to identify and isolate objects held by a robot gripper while removing the gripper and the background. We then modify the 3DGS training pipeline to support concatenated datasets with gripper occlusion, producing an omni-directional (360 degree view) model of the object. We apply Omni-Scan to part defect inspection, finding that it can identify visual or geometric defects in 12 different industrial and household objects with an average accuracy of 83%. Interactive videos of Omni-Scan 3DGS models can be found at https://berkeleyautomation.github.io/omni-scan/
>
---
#### [new 016] CHILD (Controller for Humanoid Imitation and Live Demonstration): a Whole-Body Humanoid Teleoperation System
- **分类: cs.RO**

- **简介: 该论文提出了一种名为CHILD的人形机器人全身体重操作系统，旨在解决传统半体式控制方法对关节级控制的不足，通过紧凑硬件实现多关节协同操控，结合自适应反馈增强操作体验并防止意外运动。**

- **链接: [http://arxiv.org/pdf/2508.00162v1](http://arxiv.org/pdf/2508.00162v1)**

> **作者:** Noboru Myers; Obin Kwon; Sankalp Yamsani; Joohyung Kim
>
> **摘要:** Recent advances in teleoperation have demonstrated robots performing complex manipulation tasks. However, existing works rarely support whole-body joint-level teleoperation for humanoid robots, limiting the diversity of tasks that can be accomplished. This work presents Controller for Humanoid Imitation and Live Demonstration (CHILD), a compact reconfigurable teleoperation system that enables joint level control over humanoid robots. CHILD fits within a standard baby carrier, allowing the operator control over all four limbs, and supports both direct joint mapping for full-body control and loco-manipulation. Adaptive force feedback is incorporated to enhance operator experience and prevent unsafe joint movements. We validate the capabilities of this system by conducting loco-manipulation and full-body control examples on a humanoid robot and multiple dual-arm systems. Lastly, we open-source the design of the hardware promoting accessibility and reproducibility. Additional details and open-source information are available at our project website: https://uiuckimlab.github.io/CHILD-pages.
>
---
#### [new 017] UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents
- **分类: cs.RO; cs.CV**

- **简介: 该论文旨在构建一个开放世界下的无人机目标导航基准，解决传统视觉语言导航受限于细节指导的问题，通过高语义目标与模块化策略实现自主探索，对比多种方法验证其在复杂环境中的挑战性。**

- **链接: [http://arxiv.org/pdf/2508.00288v1](http://arxiv.org/pdf/2508.00288v1)**

> **作者:** Jianqiang Xiao; Yuexuan Sun; Yixin Shao; Boxi Gan; Rongqiang Liu; Yanjing Wu; Weili Gua; Xiang Deng
>
> **备注:** Accepted to ACM MM Dataset Track 2025
>
> **摘要:** Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments.
>
---
#### [new 018] Petri Net Modeling and Deadlock-Free Scheduling of Attachable Heterogeneous AGV Systems
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文研究了attachable heterogeneous AGV系统的动态调度问题，提出基于Petri网的deadlock-free方法并开发了适应性元启发式算法，有效解决了协作AGV中同步耦合与死锁风险，验证了其在工程实践中的有效性。**

- **链接: [http://arxiv.org/pdf/2508.00724v1](http://arxiv.org/pdf/2508.00724v1)**

> **作者:** Boyu Li; Zhengchen Li; Weimin Wu; Mengchu Zhou
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** The increasing demand for automation and flexibility drives the widespread adoption of heterogeneous automated guided vehicles (AGVs). This work intends to investigate a new scheduling problem in a material transportation system consisting of attachable heterogeneous AGVs, namely carriers and shuttles. They can flexibly attach to and detach from each other to cooperatively execute complex transportation tasks. While such collaboration enhances operational efficiency, the attachment-induced synchronization and interdependence render the scheduling coupled and susceptible to deadlock. To tackle this challenge, Petri nets are introduced to model AGV schedules, well describing the concurrent and sequential task execution and carrier-shuttle synchronization. Based on Petri net theory, a firing-driven decoding method is proposed, along with deadlock detection and prevention strategies to ensure deadlock-free schedules. Furthermore, a Petri net-based metaheuristic is developed in an adaptive large neighborhood search framework and incorporates an effective acceleration method to enhance computational efficiency. Finally, numerical experiments using real-world industrial data validate the effectiveness of the proposed algorithm against the scheduling policy applied in engineering practice, an exact solver, and four state-of-the-art metaheuristics. A sensitivity analysis is also conducted to provide managerial insights.
>
---
#### [new 019] Context-based Motion Retrieval using Open Vocabulary Methods for Autonomous Driving
- **分类: cs.CV; cs.CL; cs.IR; cs.RO; 68T45, 68P20, 68T10, 68T50, 68T07, 68T40; I.2.10; I.4.8; I.2.9; H.3.3**

- **简介: 该论文旨在构建一种基于开放语料库的自主驾驶场景检索方法，解决传统方法在长尾数据中的挑战，通过结合SMPL运动序列与文本查询实现高效检索。**

- **链接: [http://arxiv.org/pdf/2508.00589v1](http://arxiv.org/pdf/2508.00589v1)**

> **作者:** Stefan Englmeier; Max A. Büttner; Katharina Winter; Fabian B. Flohr
>
> **备注:** 9 pages, 10 figure, project page https://iv.ee.hm.edu/contextmotionclip/, submitted to IEEE Transactions on Intelligent Vehicles (T-IV), This work has been submitted to the IEEE for possible publication
>
> **摘要:** Autonomous driving systems must operate reliably in safety-critical scenarios, particularly those involving unusual or complex behavior by Vulnerable Road Users (VRUs). Identifying these edge cases in driving datasets is essential for robust evaluation and generalization, but retrieving such rare human behavior scenarios within the long tail of large-scale datasets is challenging. To support targeted evaluation of autonomous driving systems in diverse, human-centered scenarios, we propose a novel context-aware motion retrieval framework. Our method combines Skinned Multi-Person Linear (SMPL)-based motion sequences and corresponding video frames before encoding them into a shared multimodal embedding space aligned with natural language. Our approach enables the scalable retrieval of human behavior and their context through text queries. This work also introduces our dataset WayMoCo, an extension of the Waymo Open Dataset. It contains automatically labeled motion and scene context descriptions derived from generated pseudo-ground-truth SMPL sequences and corresponding image data. Our approach outperforms state-of-the-art models by up to 27.5% accuracy in motion-context retrieval, when evaluated on the WayMoCo dataset.
>
---
#### [new 020] Data-Driven Motion Planning for Uncertain Nonlinear Systems
- **分类: eess.SY; cs.LG; cs.RO; cs.SY; math.OC**

- **简介: 该论文提出了一种基于数据驱动的不确定性非线性系统运动规划方法，解决了动态可变约束下的路径规划问题。通过构建覆盖不变多面体序列、利用凸优化学习状态反馈增益及凸壳近似，实现了安全、动态可行的路径规划，无需传统系统建模。**

- **链接: [http://arxiv.org/pdf/2508.00154v1](http://arxiv.org/pdf/2508.00154v1)**

> **作者:** Babak Esmaeili; Hamidreza Modares; Stefano Di Cairano
>
> **摘要:** This paper proposes a data-driven motion-planning framework for nonlinear systems that constructs a sequence of overlapping invariant polytopes. Around each randomly sampled waypoint, the algorithm identifies a convex admissible region and solves data-driven linear-matrix-inequality problems to learn several ellipsoidal invariant sets together with their local state-feedback gains. The convex hull of these ellipsoids, still invariant under a piece-wise-affine controller obtained by interpolating the gains, is then approximated by a polytope. Safe transitions between nodes are ensured by verifying the intersection of consecutive convex-hull polytopes and introducing an intermediate node for a smooth transition. Control gains are interpolated in real time via simplex-based interpolation, keeping the state inside the invariant polytopes throughout the motion. Unlike traditional approaches that rely on system dynamics models, our method requires only data to compute safe regions and design state-feedback controllers. The approach is validated through simulations, demonstrating the effectiveness of the proposed method in achieving safe, dynamically feasible paths for complex nonlinear systems.
>
---
#### [new 021] Reducing the gap between general purpose data and aerial images in concentrated solar power plants
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文旨在解决通用数据集难以泛化到复杂太阳能场的问题，提出AerialCSP虚拟数据集作为替代方案，通过生成仿真实验数据降低手动标注成本，提升CSP相关视觉任务的性能。**

- **链接: [http://arxiv.org/pdf/2508.00440v1](http://arxiv.org/pdf/2508.00440v1)**

> **作者:** M. A. Pérez-Cutiño; J. Valverde; J. Capitán; J. M. Díaz-Báñez
>
> **摘要:** In the context of Concentrated Solar Power (CSP) plants, aerial images captured by drones present a unique set of challenges. Unlike urban or natural landscapes commonly found in existing datasets, solar fields contain highly reflective surfaces, and domain-specific elements that are uncommon in traditional computer vision benchmarks. As a result, machine learning models trained on generic datasets struggle to generalize to this setting without extensive retraining and large volumes of annotated data. However, collecting and labeling such data is costly and time-consuming, making it impractical for rapid deployment in industrial applications. To address this issue, we propose a novel approach: the creation of AerialCSP, a virtual dataset that simulates aerial imagery of CSP plants. By generating synthetic data that closely mimic real-world conditions, our objective is to facilitate pretraining of models before deployment, significantly reducing the need for extensive manual labeling. Our main contributions are threefold: (1) we introduce AerialCSP, a high-quality synthetic dataset for aerial inspection of CSP plants, providing annotated data for object detection and image segmentation; (2) we benchmark multiple models on AerialCSP, establishing a baseline for CSP-related vision tasks; and (3) we demonstrate that pretraining on AerialCSP significantly improves real-world fault detection, particularly for rare and small defects, reducing the need for extensive manual labeling. AerialCSP is made publicly available at https://mpcutino.github.io/aerialcsp/.
>
---
#### [new 022] The Monado SLAM Dataset for Egocentric Visual-Inertial Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文旨在解决复杂环境下的视觉-惯性跟踪问题，通过Monado SLAM数据集提供实测序列，弥补现有方法在高动态、低纹理等场景中的不足，推动VIO/SLAM技术发展。**

- **链接: [http://arxiv.org/pdf/2508.00088v1](http://arxiv.org/pdf/2508.00088v1)**

> **作者:** Mateo de Mayo; Daniel Cremers; Taihú Pire
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Humanoid robots and mixed reality headsets benefit from the use of head-mounted sensors for tracking. While advancements in visual-inertial odometry (VIO) and simultaneous localization and mapping (SLAM) have produced new and high-quality state-of-the-art tracking systems, we show that these are still unable to gracefully handle many of the challenging settings presented in the head-mounted use cases. Common scenarios like high-intensity motions, dynamic occlusions, long tracking sessions, low-textured areas, adverse lighting conditions, saturation of sensors, to name a few, continue to be covered poorly by existing datasets in the literature. In this way, systems may inadvertently overlook these essential real-world issues. To address this, we present the Monado SLAM dataset, a set of real sequences taken from multiple virtual reality headsets. We release the dataset under a permissive CC BY 4.0 license, to drive advancements in VIO/SLAM research and development.
>
---
#### [new 023] Towards Efficient Certification of Maritime Remote Operation Centers
- **分类: cs.CY; cs.RO**

- **简介: 该论文旨在构建支持远程操作中心认证的危险数据库，解决自动化船舶监控与远程控制的安全性问题，通过分类风险源并评估方法可行性进行前期分析。**

- **链接: [http://arxiv.org/pdf/2508.00543v1](http://arxiv.org/pdf/2508.00543v1)**

> **作者:** Christian Neurohr; Marcel Saager; Lina Putze; Jan-Patrick Osterloh; Karina Rothemann; Hilko Wiards; Eckard Böde; Axel Hahn
>
> **摘要:** Additional automation being build into ships implies a shift of crew from ship to shore. However, automated ships still have to be monitored and, in some situations, controlled remotely. These tasks are carried out by human operators located in shore-based remote operation centers. In this work, we present a concept for a hazard database that supports the safeguarding and certification of such remote operation centers. The concept is based on a categorization of hazard sources which we derive from a generic functional architecture. A subsequent preliminary suitability analysis unveils which methods for hazard analysis and risk assessment can adequately fill this hazard database.
>
---
#### [new 024] IGL-Nav: Incremental 3D Gaussian Localization for Image-goal Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文旨在解决视觉导航中图像作为目标时的高效定位问题，构建基于增量式3D-Gaussian（3DGS）的框架，通过改进的离散空间匹配算法实现3D-aware定位，同时支持自由视图场景并可部署于移动设备。**

- **链接: [http://arxiv.org/pdf/2508.00823v1](http://arxiv.org/pdf/2508.00823v1)**

> **作者:** Wenxuan Guo; Xiuwei Xu; Hang Yin; Ziwei Wang; Jianjiang Feng; Jie Zhou; Jiwen Lu
>
> **备注:** Accepted to ICCV 2025. Project page: https://gwxuan.github.io/IGL-Nav/
>
> **摘要:** Visual navigation with an image as goal is a fundamental and challenging problem. Conventional methods either rely on end-to-end RL learning or modular-based policy with topological graph or BEV map as memory, which cannot fully model the geometric relationship between the explored 3D environment and the goal image. In order to efficiently and accurately localize the goal image in 3D space, we build our navigation system upon the renderable 3D gaussian (3DGS) representation. However, due to the computational intensity of 3DGS optimization and the large search space of 6-DoF camera pose, directly leveraging 3DGS for image localization during agent exploration process is prohibitively inefficient. To this end, we propose IGL-Nav, an Incremental 3D Gaussian Localization framework for efficient and 3D-aware image-goal navigation. Specifically, we incrementally update the scene representation as new images arrive with feed-forward monocular prediction. Then we coarsely localize the goal by leveraging the geometric information for discrete space matching, which can be equivalent to efficient 3D convolution. When the agent is close to the goal, we finally solve the fine target pose with optimization via differentiable rendering. The proposed IGL-Nav outperforms existing state-of-the-art methods by a large margin across diverse experimental configurations. It can also handle the more challenging free-view image-goal setting and be deployed on real-world robotic platform using a cellphone to capture goal image at arbitrary pose. Project page: https://gwxuan.github.io/IGL-Nav/.
>
---
#### [new 025] Controllable Pedestrian Video Editing for Multi-View Driving Scenarios via Motion Sequence
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文旨在解决自动驾驶中行人检测模型在多视角场景下的鲁棒性不足问题，提出一种基于运动序列控制的视频剪辑框架，通过融合视频补全与人体姿态引导实现跨视角行人编辑，提升视觉真实性和时空一致性。**

- **链接: [http://arxiv.org/pdf/2508.00299v1](http://arxiv.org/pdf/2508.00299v1)**

> **作者:** Danzhen Fu; Jiagao Hu; Daiguo Zhou; Fei Wang; Zepeng Wang; Wenhua Liao
>
> **备注:** ICCV 2025 Workshop (HiGen)
>
> **摘要:** Pedestrian detection models in autonomous driving systems often lack robustness due to insufficient representation of dangerous pedestrian scenarios in training datasets. To address this limitation, we present a novel framework for controllable pedestrian video editing in multi-view driving scenarios by integrating video inpainting and human motion control techniques. Our approach begins by identifying pedestrian regions of interest across multiple camera views, expanding detection bounding boxes with a fixed ratio, and resizing and stitching these regions into a unified canvas while preserving cross-view spatial relationships. A binary mask is then applied to designate the editable area, within which pedestrian editing is guided by pose sequence control conditions. This enables flexible editing functionalities, including pedestrian insertion, replacement, and removal. Extensive experiments demonstrate that our framework achieves high-quality pedestrian editing with strong visual realism, spatiotemporal coherence, and cross-view consistency. These results establish the proposed method as a robust and versatile solution for multi-view pedestrian video generation, with broad potential for applications in data augmentation and scenario simulation in autonomous driving.
>
---
## 更新

#### [replaced 001] Data-driven tool wear prediction in milling, based on a process-integrated single-sensor approach
- **分类: cs.LG; cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2412.19950v4](http://arxiv.org/pdf/2412.19950v4)**

> **作者:** Eric Hirsch; Christian Friedrich
>
> **备注:** This work has been submitted to the IEEE Transactions on Automation Science and Engineering for possible publication. ,14 pages, 12 figures
>
> **摘要:** Accurate tool wear prediction is essential for maintaining productivity and minimizing costs in machining. However, the complex nature of the tool wear process poses significant challenges to achieving reliable predictions. This study explores data-driven methods, in particular deep learning, for tool wear prediction. Traditional data-driven approaches often focus on a single process, relying on multi-sensor setups and extensive data generation, which limits generalization to new settings. Moreover, multi-sensor integration is often impractical in industrial environments. To address these limitations, this research investigates the transferability of predictive models using minimal training data, validated across two processes. Furthermore, it uses a simple setup with a single acceleration sensor to establish a low-cost data generation approach that facilitates the generalization of models to other processes via transfer learning. The study evaluates several machine learning models, including transformer-inspired convolutional neural networks (CNN), long short-term memory networks (LSTM), support vector machines (SVM), and decision trees, trained on different input formats such as feature vectors and short-time Fourier transform (STFT). The performance of the models is evaluated on two machines and on different amounts of training data, including scenarios with significantly reduced datasets, providing insight into their effectiveness under constrained data conditions. The results demonstrate the potential of specific models and configurations for effective tool wear prediction, contributing to the development of more adaptable and efficient predictive maintenance strategies in machining. Notably, the ConvNeXt model has an exceptional performance, achieving 99.1\% accuracy in identifying tool wear using data from only four milling tools operated until they are worn.
>
---
#### [replaced 002] AugInsert: Learning Robust Visual-Force Policies via Data Augmentation for Object Assembly Tasks
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.14968v2](http://arxiv.org/pdf/2410.14968v2)**

> **作者:** Ryan Diaz; Adam Imdieke; Vivek Veeriah; Karthik Desingh
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Operating in unstructured environments like households requires robotic policies that are robust to out-of-distribution conditions. Although much work has been done in evaluating robustness for visuomotor policies, the robustness evaluation of a multisensory approach that includes force-torque sensing remains largely unexplored. This work introduces a novel, factor-based evaluation framework with the goal of assessing the robustness of multisensory policies in a peg-in-hole assembly task. To this end, we develop a multisensory policy framework utilizing the Perceiver IO architecture to learn the task. We investigate which factors pose the greatest generalization challenges in object assembly and explore a simple multisensory data augmentation technique to enhance out-of-distribution performance. We provide a simulation environment enabling controlled evaluation of these factors. Our results reveal that multisensory variations such as Grasp Pose present the most significant challenges for robustness, and naive unisensory data augmentation applied independently to each sensory modality proves insufficient to overcome them. Additionally, we find force-torque sensing to be the most informative modality for our contact-rich assembly task, with vision being the least informative. Finally, we briefly discuss supporting real-world experimental results. For additional experiments and qualitative results, we refer to the project webpage https://rpm-lab-umn.github.io/auginsert/ .
>
---
#### [replaced 003] Cooperative Payload Estimation by a Team of Mocobots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.04600v3](http://arxiv.org/pdf/2502.04600v3)**

> **作者:** Haoxuan Zhang; C. Lin Liu; Matthew L. Elwin; Randy A. Freeman; Kevin M. Lynch
>
> **备注:** 8 pages, 6 figures. Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** For high-performance autonomous manipulation of a payload by a mobile manipulator team, or for collaborative manipulation with the human, robots should be able to discover where other robots are attached to the payload, as well as the payload's mass and inertial properties. In this paper, we describe a method for the robots to autonomously discover this information. The robots cooperatively manipulate the payload, and the twist, twist derivative, and wrench data at their grasp frames are used to estimate the transformation matrices between the grasp frames, the location of the payload's center of mass, and the payload's inertia matrix. The method is validated experimentally with a team of three mobile cobots, or mocobots.
>
---
#### [replaced 004] Learning Goal-Directed Object Pushing in Cluttered Scenes With Location-Based Attention
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.17667v3](http://arxiv.org/pdf/2403.17667v3)**

> **作者:** Nils Dengler; Juan Del Aguila Ferrandis; João Moura; Sethu Vijayakumar; Maren Bennewitz
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)2025
>
> **摘要:** In complex scenarios where typical pick-and-place techniques are insufficient, often non-prehensile manipulation can ensure that a robot is able to fulfill its task. However, non-prehensile manipulation is challenging due to its underactuated nature with hybrid-dynamics, where a robot needs to reason about an object's long-term behavior and contact-switching, while being robust to contact uncertainty. The presence of clutter in the workspace further complicates this task, introducing the need to include more advanced spatial analysis to avoid unwanted collisions. Building upon prior work on reinforcement learning with multimodal categorical exploration for planar pushing, we propose to incorporate location-based attention to enable robust manipulation in cluttered scenes. Unlike previous approaches addressing this obstacle avoiding pushing task, our framework requires no predefined global paths and considers the desired target orientation of the manipulated object. Experimental results in simulation as well as with a real KUKA iiwa robot arm demonstrate that our learned policy manipulates objects successfully while avoiding collisions through complex obstacle configurations, including dynamic obstacles, to reach the desired target pose.
>
---
#### [replaced 005] TopoRec: Point Cloud Recognition Using Topological Data Analysis
- **分类: cs.RO; cs.CG; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.18725v2](http://arxiv.org/pdf/2506.18725v2)**

> **作者:** Anirban Ghosh; Iliya Kulbaka; Ian Dahlin; Ayan Dutta
>
> **摘要:** Point cloud-based object/place recognition remains a problem of interest in applications such as autonomous driving, scene reconstruction, and localization. Extracting a meaningful global descriptor from a query point cloud that can be matched with the descriptors of the database point clouds is a challenging problem. Furthermore, when the query point cloud is noisy or has been transformed (e.g., rotated), it adds to the complexity. To this end, we propose a novel methodology, named TopoRec, which utilizes Topological Data Analysis (TDA) for extracting local descriptors from a point cloud, thereby eliminating the need for resource-intensive GPU-based machine learning training. More specifically, we used the ATOL vectorization method to generate vectors for point clouds. To test the quality of the proposed TopoRec technique, we have implemented it on multiple real-world (e.g., Oxford RobotCar, NCLT) and realistic (e.g., ShapeNet) point cloud datasets for large-scale place and object recognition, respectively. Unlike existing learning-based approaches such as PointNetVLAD and PCAN, our method does not require extensive training, making it easily adaptable to new environments. Despite this, it consistently outperforms both state-of-the-art learning-based and handcrafted baselines (e.g., M2DP, ScanContext) on standard benchmark datasets, demonstrating superior accuracy and strong generalization.
>
---
#### [replaced 006] Cooperative and Asynchronous Transformer-based Mission Planning for Heterogeneous Teams of Mobile Robots
- **分类: cs.RO; cs.AI; I.2.9; I.2.11**

- **链接: [http://arxiv.org/pdf/2410.06372v3](http://arxiv.org/pdf/2410.06372v3)**

> **作者:** Milad Farjadnasab; Shahin Sirouspour
>
> **摘要:** Cooperative mission planning for heterogeneous teams of mobile robots presents a unique set of challenges, particularly when operating under communication constraints and limited computational resources. To address these challenges, we propose the Cooperative and Asynchronous Transformer-based Mission Planning (CATMiP) framework, which leverages multi-agent reinforcement learning (MARL) to coordinate distributed decision making among agents with diverse sensing, motion, and actuation capabilities, operating under sporadic ad hoc communication. A Class-based Macro-Action Decentralized Partially Observable Markov Decision Process (CMacDec-POMDP) is also formulated to effectively model asynchronous decision-making for heterogeneous teams of agents. The framework utilizes an asynchronous centralized training and distributed execution scheme, enabled by the proposed Asynchronous Multi-Agent Transformer (AMAT) architecture. This design allows a single trained model to generalize to larger environments and accommodate varying team sizes and compositions. We evaluate CATMiP in a 2D grid-world simulation environment and compare its performance against planning-based exploration methods. Results demonstrate CATMiP's superior efficiency, scalability, and robustness to communication dropouts and input noise, highlighting its potential for real-world heterogeneous mobile robot systems. The code is available at https://github.com/mylad13/CATMiP
>
---
#### [replaced 007] Learning to Push, Group, and Grasp: A Diffusion Policy Approach for Multi-Object Delivery
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.08452v3](http://arxiv.org/pdf/2502.08452v3)**

> **作者:** Takahiro Yonemaru; Weiwei Wan; Tatsuki Nishimura; Kensuke Harada
>
> **摘要:** Simultaneously grasping and delivering multiple objects can significantly enhance robotic work efficiency and has been a key research focus for decades. The primary challenge lies in determining how to push objects, group them, and execute simultaneous grasping for respective groups while considering object distribution and the hardware constraints of the robot. Traditional rule-based methods struggle to flexibly adapt to diverse scenarios. To address this challenge, this paper proposes an imitation learning-based approach. We collect a series of expert demonstrations through teleoperation and train a diffusion policy network, enabling the robot to dynamically generate action sequences for pushing, grouping, and grasping, thereby facilitating efficient multi-object grasping and delivery. We conducted experiments to evaluate the method under different training dataset sizes, varying object quantities, and real-world object scenarios. The results demonstrate that the proposed approach can effectively and adaptively generate multi-object grouping and grasping strategies. With the support of more training data, imitation learning is expected to be an effective approach for solving the multi-object grasping problem.
>
---
#### [replaced 008] Multi-robot LiDAR SLAM: a practical case study in underground tunnel environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.21553v3](http://arxiv.org/pdf/2507.21553v3)**

> **作者:** Federica Di Lauro; Domenico G. Sorrenti; Miguel Angel Sotelo
>
> **备注:** 14 pages, 14 figures
>
> **摘要:** Multi-robot SLAM aims at localizing and building a map with multiple robots, interacting with each other. In the work described in this article, we analyze the pipeline of a decentralized LiDAR SLAM system to study the current limitations of the state of the art, and we discover a significant source of failures, i.e., that the loop detection is the source of too many false positives. We therefore develop and propose a new heuristic to overcome these limitations. The environment taken as reference in this work is the highly challenging case of underground tunnels. We also highlight potential new research areas still under-explored.
>
---
#### [replaced 009] Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.13587v4](http://arxiv.org/pdf/2411.13587v4)**

> **作者:** Taowen Wang; Cheng Han; James Chenhao Liang; Wenhao Yang; Dongfang Liu; Luna Xinyu Zhang; Qifan Wang; Jiebo Luo; Ruixiang Tang
>
> **备注:** ICCV camera ready; Github: https://github.com/William-wAng618/roboticAttack Homepage: https://vlaattacker.github.io/
>
> **摘要:** Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. Despite their significant capabilities, VLA models introduce new attack surfaces. This paper systematically evaluates their robustness. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce two untargeted attack objectives that leverage spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, we advance both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for continuously developing robust defense strategies prior to physical-world deployments.
>
---
#### [replaced 010] SHIELD: Safety on Humanoids via CBFs In Expectation on Learned Dynamics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11494v2](http://arxiv.org/pdf/2505.11494v2)**

> **作者:** Lizhi Yang; Blake Werner; Ryan K. Cosner; David Fridovich-Keil; Preston Culbertson; Aaron D. Ames
>
> **备注:** Video at https://youtu.be/-Qv1wR4jfj4. To appear at IROS 2025
>
> **摘要:** Robot learning has produced remarkably effective ``black-box'' controllers for complex tasks such as dynamic locomotion on humanoids. Yet ensuring dynamic safety, i.e., constraint satisfaction, remains challenging for such policies. Reinforcement learning (RL) embeds constraints heuristically through reward engineering, and adding or modifying constraints requires retraining. Model-based approaches, like control barrier functions (CBFs), enable runtime constraint specification with formal guarantees but require accurate dynamics models. This paper presents SHIELD, a layered safety framework that bridges this gap by: (1) training a generative, stochastic dynamics residual model using real-world data from hardware rollouts of the nominal controller, capturing system behavior and uncertainties; and (2) adding a safety layer on top of the nominal (learned locomotion) controller that leverages this model via a stochastic discrete-time CBF formulation enforcing safety constraints in probability. The result is a minimally-invasive safety layer that can be added to the existing autonomy stack to give probabilistic guarantees of safety that balance risk and performance. In hardware experiments on an Unitree G1 humanoid, SHIELD enables safe navigation (obstacle avoidance) through varied indoor and outdoor environments using a nominal (unknown) RL controller and onboard perception.
>
---
#### [replaced 011] Trends in Motion Prediction Toward Deployable and Generalizable Autonomy: A Revisit and Perspectives
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.09074v3](http://arxiv.org/pdf/2505.09074v3)**

> **作者:** Letian Wang; Marc-Antoine Lavoie; Sandro Papais; Barza Nisar; Yuxiao Chen; Wenhao Ding; Boris Ivanovic; Hao Shao; Abulikemu Abuduweili; Evan Cook; Yang Zhou; Peter Karkus; Jiachen Li; Changliu Liu; Marco Pavone; Steven Waslander
>
> **备注:** Updated draft. 163 pages, 40 figures, 13 tables
>
> **摘要:** Motion prediction, the anticipation of future agent states or scene evolution, is rooted in human cognition, bridging perception and decision-making. It enables intelligent systems, such as robots and self-driving cars, to act safely in dynamic, human-involved environments, and informs broader time-series reasoning challenges. With advances in methods, representations, and datasets, the field has seen rapid progress, reflected in quickly evolving benchmark results. Yet, when state-of-the-art methods are deployed in the real world, they often struggle to generalize to open-world conditions and fall short of deployment standards. This reveals a gap between research benchmarks, which are often idealized or ill-posed, and real-world complexity. To address this gap, this survey revisits the generalization and deployability of motion prediction models, with an emphasis on the applications of robotics, autonomous driving, and human motion. We first offer a comprehensive taxonomy of motion prediction methods, covering representations, modeling strategies, application domains, and evaluation protocols. We then study two key challenges: (1) how to push motion prediction models to be deployable to realistic deployment standards, where motion prediction does not act in a vacuum, but functions as one module of closed-loop autonomy stacks - it takes input from the localization and perception, and informs downstream planning and control. 2) how to generalize motion prediction models from limited seen scenarios/datasets to the open-world settings. Throughout the paper, we highlight critical open challenges to guide future work, aiming to recalibrate the community's efforts, fostering progress that is not only measurable but also meaningful for real-world applications. The project webpage corresponding to this paper can be found here https://trends-in-motion-prediction-2025.github.io/.
>
---
#### [replaced 012] SCORE: Saturated Consensus Relocalization in Semantic Line Maps
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03254v2](http://arxiv.org/pdf/2503.03254v2)**

> **作者:** Haodong Jiang; Xiang Zheng; Yanglin Zhang; Qingcheng Zeng; Yiqian Li; Ziyang Hong; Junfeng Wu
>
> **备注:** 12 pages, 13 figurs, arxiv version for paper published at IROS 2025
>
> **摘要:** We present SCORE, a visual relocalization system that achieves unprecedented map compactness by adopting semantically labeled 3D line maps. SCORE requires only 0.01\%-0.1\% of the storage needed by structure-based or learning-based baselines, while maintaining practical accuracy and comparable runtime. The key innovation is a novel robust estimation mechanism, Saturated Consensus Maximization (Sat-CM), which generalizes classical Consensus Maximization (CM) by assigning diminishing weights to inlier associations according to maximum likelihood with probabilistic justification. Under extreme outlier ratios (up to 99.5\%) arising from one-to-many ambiguity in semantic matching, Sat-CM enables accurate estimation when CM fails. To ensure computational efficiency, we propose an accelerating framework for globally solving Sat-CM formulations and specialize it for the Perspective-n-Lines problem at the core of SCORE.
>
---
#### [replaced 013] PIPE Planner: Pathwise Information Gain with Map Predictions for Indoor Robot Exploration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.07504v2](http://arxiv.org/pdf/2503.07504v2)**

> **作者:** Seungjae Baek; Brady Moon; Seungchan Kim; Muqing Cao; Cherie Ho; Sebastian Scherer; Jeong hwan Jeon
>
> **备注:** 8 pages, 8 figures, IROS 2025
>
> **摘要:** Autonomous exploration in unknown environments requires estimating the information gain of an action to guide planning decisions. While prior approaches often compute information gain at discrete waypoints, pathwise integration offers a more comprehensive estimation but is often computationally challenging or infeasible and prone to overestimation. In this work, we propose the Pathwise Information Gain with Map Prediction for Exploration (PIPE) planner, which integrates cumulative sensor coverage along planned trajectories while leveraging map prediction to mitigate overestimation. To enable efficient pathwise coverage computation, we introduce a method to efficiently calculate the expected observation mask along the planned path, significantly reducing computational overhead. We validate PIPE on real-world floorplan datasets, demonstrating its superior performance over state-of-the-art baselines. Our results highlight the benefits of integrating predictive mapping with pathwise information gain for efficient and informed exploration. Website: https://pipe-planner.github.io
>
---
#### [replaced 014] Flow Matching Policy Gradients
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.21053v2](http://arxiv.org/pdf/2507.21053v2)**

> **作者:** David McAllister; Songwei Ge; Brent Yi; Chung Min Kim; Ethan Weber; Hongsuk Choi; Haiwen Feng; Angjoo Kanazawa
>
> **备注:** See our blog post at https://flowreinforce.github.io
>
> **摘要:** Flow-based generative models, including diffusion models, excel at modeling continuous distributions in high-dimensional spaces. In this work, we introduce Flow Policy Optimization (FPO), a simple on-policy reinforcement learning algorithm that brings flow matching into the policy gradient framework. FPO casts policy optimization as maximizing an advantage-weighted ratio computed from the conditional flow matching loss, in a manner compatible with the popular PPO-clip framework. It sidesteps the need for exact likelihood computation while preserving the generative capabilities of flow-based models. Unlike prior approaches for diffusion-based reinforcement learning that bind training to a specific sampling method, FPO is agnostic to the choice of diffusion or flow integration at both training and inference time. We show that FPO can train diffusion-style policies from scratch in a variety of continuous control tasks. We find that flow-based models can capture multimodal action distributions and achieve higher performance than Gaussian policies, particularly in under-conditioned settings.
>
---
#### [replaced 015] DiFuse-Net: RGB and Dual-Pixel Depth Estimation using Window Bi-directional Parallax Attention and Cross-modal Transfer Learning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.14709v2](http://arxiv.org/pdf/2506.14709v2)**

> **作者:** Kunal Swami; Debtanu Gupta; Amrit Kumar Muduli; Chirag Jaiswal; Pankaj Kumar Bajpai
>
> **备注:** Accepted in IROS 2025
>
> **摘要:** Depth estimation is crucial for intelligent systems, enabling applications from autonomous navigation to augmented reality. While traditional stereo and active depth sensors have limitations in cost, power, and robustness, dual-pixel (DP) technology, ubiquitous in modern cameras, offers a compelling alternative. This paper introduces DiFuse-Net, a novel modality decoupled network design for disentangled RGB and DP based depth estimation. DiFuse-Net features a window bi-directional parallax attention mechanism (WBiPAM) specifically designed to capture the subtle DP disparity cues unique to smartphone cameras with small aperture. A separate encoder extracts contextual information from the RGB image, and these features are fused to enhance depth prediction. We also propose a Cross-modal Transfer Learning (CmTL) mechanism to utilize large-scale RGB-D datasets in the literature to cope with the limitations of obtaining large-scale RGB-DP-D dataset. Our evaluation and comparison of the proposed method demonstrates its superiority over the DP and stereo-based baseline methods. Additionally, we contribute a new, high-quality, real-world RGB-DP-D training dataset, named Dual-Camera Dual-Pixel (DCDP) dataset, created using our novel symmetric stereo camera hardware setup, stereo calibration and rectification protocol, and AI stereo disparity estimation method.
>
---
#### [replaced 016] H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.23523v2](http://arxiv.org/pdf/2507.23523v2)**

> **作者:** Hongzhe Bi; Lingxuan Wu; Tianwei Lin; Hengkai Tan; Zhizhong Su; Hang Su; Jun Zhu
>
> **摘要:** Imitation learning for robotic manipulation faces a fundamental challenge: the scarcity of large-scale, high-quality robot demonstration data. Recent robotic foundation models often pre-train on cross-embodiment robot datasets to increase data scale, while they face significant limitations as the diverse morphologies and action spaces across different robot embodiments make unified training challenging. In this paper, we present H-RDT (Human to Robotics Diffusion Transformer), a novel approach that leverages human manipulation data to enhance robot manipulation capabilities. Our key insight is that large-scale egocentric human manipulation videos with paired 3D hand pose annotations provide rich behavioral priors that capture natural manipulation strategies and can benefit robotic policy learning. We introduce a two-stage training paradigm: (1) pre-training on large-scale egocentric human manipulation data, and (2) cross-embodiment fine-tuning on robot-specific data with modular action encoders and decoders. Built on a diffusion transformer architecture with 2B parameters, H-RDT uses flow matching to model complex action distributions. Extensive evaluations encompassing both simulation and real-world experiments, single-task and multitask scenarios, as well as few-shot learning and robustness assessments, demonstrate that H-RDT outperforms training from scratch and existing state-of-the-art methods, including Pi0 and RDT, achieving significant improvements of 13.9% and 40.5% over training from scratch in simulation and real-world experiments, respectively. The results validate our core hypothesis that human manipulation data can serve as a powerful foundation for learning bimanual robotic manipulation policies.
>
---
#### [replaced 017] E2E Parking Dataset: An Open Benchmark for End-to-End Autonomous Parking
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10812v2](http://arxiv.org/pdf/2504.10812v2)**

> **作者:** Kejia Gao; Liguo Zhou; Mingjun Liu; Alois Knoll
>
> **摘要:** End-to-end learning has shown great potential in autonomous parking, yet the lack of publicly available datasets limits reproducibility and benchmarking. While prior work introduced a visual-based parking model and a pipeline for data generation, training, and close-loop test, the dataset itself was not released. To bridge this gap, we create and open-source a high-quality dataset for end-to-end autonomous parking. Using the original model, we achieve an overall success rate of 85.16% with lower average position and orientation errors (0.24 meters and 0.34 degrees).
>
---
#### [replaced 018] Learning to Drift with Individual Wheel Drive: Maneuvering Autonomous Vehicle at the Handling Limits
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.23339v2](http://arxiv.org/pdf/2507.23339v2)**

> **作者:** Yihan Zhou; Yiwen Lu; Bo Yang; Jiayun Li; Yilin Mo
>
> **摘要:** Drifting, characterized by controlled vehicle motion at high sideslip angles, is crucial for safely handling emergency scenarios at the friction limits. While recent reinforcement learning approaches show promise for drifting control, they struggle with the significant simulation-to-reality gap, as policies that perform well in simulation often fail when transferred to physical systems. In this paper, we present a reinforcement learning framework with GPU-accelerated parallel simulation and systematic domain randomization that effectively bridges the gap. The proposed approach is validated on both simulation and a custom-designed and open-sourced 1/10 scale Individual Wheel Drive (IWD) RC car platform featuring independent wheel speed control. Experiments across various scenarios from steady-state circular drifting to direction transitions and variable-curvature path following demonstrate that our approach achieves precise trajectory tracking while maintaining controlled sideslip angles throughout complex maneuvers in both simulated and real-world environments.
>
---
#### [replaced 019] Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.23172v2](http://arxiv.org/pdf/2507.23172v2)**

> **作者:** Viraj Joshi; Zifan Xu; Bo Liu; Peter Stone; Amy Zhang
>
> **备注:** RLC 2025
>
> **摘要:** Multi-task Reinforcement Learning (MTRL) has emerged as a critical training paradigm for applying reinforcement learning (RL) to a set of complex real-world robotic tasks, which demands a generalizable and robust policy. At the same time, \emph{massively parallelized training} has gained popularity, not only for significantly accelerating data collection through GPU-accelerated simulation but also for enabling diverse data collection across multiple tasks by simulating heterogeneous scenes in parallel. However, existing MTRL research has largely been limited to off-policy methods like SAC in the low-parallelization regime. MTRL could capitalize on the higher asymptotic performance of on-policy algorithms, whose batches require data from the current policy, and as a result, take advantage of massive parallelization offered by GPU-accelerated simulation. To bridge this gap, we introduce a massively parallelized $\textbf{M}$ulti-$\textbf{T}$ask $\textbf{Bench}$mark for robotics (MTBench), an open-sourced benchmark featuring a broad distribution of 50 manipulation tasks and 20 locomotion tasks, implemented using the GPU-accelerated simulator IsaacGym. MTBench also includes four base RL algorithms combined with seven state-of-the-art MTRL algorithms and architectures, providing a unified framework for evaluating their performance. Our extensive experiments highlight the superior speed of evaluating MTRL approaches using MTBench, while also uncovering unique challenges that arise from combining massive parallelism with MTRL. Code is available at https://github.com/Viraj-Joshi/MTBench
>
---
#### [replaced 020] Scalable Outdoors Autonomous Drone Flight with Visual-Inertial SLAM and Dense Submaps Built without LiDAR
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.09596v2](http://arxiv.org/pdf/2403.09596v2)**

> **作者:** Sebastián Barbas Laina; Simon Boche; Sotiris Papatheodorou; Dimos Tzoumanikas; Simon Schaefer; Hanzhi Chen; Stefan Leutenegger
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Autonomous navigation is needed for several robotics applications. In this paper we present an autonomous Micro Aerial Vehicle (MAV) system which purely relies on cost-effective and light-weight passive visual and inertial sensors to perform large-scale autonomous navigation in outdoor,unstructured and cluttered environments. We leverage visual-inertial simultaneous localization and mapping (VI-SLAM) for accurate MAV state estimates and couple it with a volumetric occupancy submapping system to achieve a scalable mapping framework which can be directly used for path planning. To ensure the safety of the MAV during navigation, we also propose a novel reference trajectory anchoring scheme that deforms the reference trajectory the MAV is tracking upon state updates from the VI-SLAM system in a consistent way, even upon large state updates due to loop-closures. We thoroughly validate our system in both real and simulated forest environments and at peak velocities up to 3 m/s while not encountering a single collision or system failure. To the best of our knowledge, this is the first system which achieves this level of performance in such an unstructured environment using low-cost passive visual sensors and fully on-board computation, including VI-SLAM.
>
---
#### [replaced 021] Safe Navigation in Uncertain Crowded Environments Using Risk Adaptive CVaR Barrier Functions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.06513v5](http://arxiv.org/pdf/2504.06513v5)**

> **作者:** Xinyi Wang; Taekyung Kim; Bardh Hoxha; Georgios Fainekos; Dimitra Panagou
>
> **备注:** 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). Project page: {https://lawliet9666.github.io/cvarbf/}
>
> **摘要:** Robot navigation in dynamic, crowded environments poses a significant challenge due to the inherent uncertainties in the obstacle model. In this work, we propose a risk-adaptive approach based on the Conditional Value-at-Risk Barrier Function (CVaR-BF), where the risk level is automatically adjusted to accept the minimum necessary risk, achieving a good performance in terms of safety and optimization feasibility under uncertainty. Additionally, we introduce a dynamic zone-based barrier function which characterizes the collision likelihood by evaluating the relative state between the robot and the obstacle. By integrating risk adaptation with this new function, our approach adaptively expands the safety margin, enabling the robot to proactively avoid obstacles in highly dynamic environments. Comparisons and ablation studies demonstrate that our method outperforms existing social navigation approaches, and validate the effectiveness of our proposed framework.
>
---
#### [replaced 022] Risk-Aware Autonomous Driving with Linear Temporal Logic Specifications
- **分类: eess.SY; cs.FL; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2409.09769v3](http://arxiv.org/pdf/2409.09769v3)**

> **作者:** Shuhao Qi; Zengjie Zhang; Zhiyong Sun; Sofie Haesaert
>
> **摘要:** Human drivers naturally balance the risks of different concerns while driving, including traffic rule violations, minor accidents, and fatalities. However, achieving the same behavior in autonomous driving systems remains an open problem. This paper extends a risk metric that has been verified in human-like driving studies to encompass more complex driving scenarios specified by linear temporal logic (LTL) that go beyond just collision risks. This extension incorporates the timing and severity of events into LTL specifications, thereby reflecting a human-like risk awareness. Without sacrificing expressivity for traffic rules, we adopt LTL specifications composed of safety and co-safety formulas, allowing the control synthesis problem to be reformulated as a reachability problem. By leveraging occupation measures, we further formulate a linear programming (LP) problem for this LTL-based risk metric. Consequently, the synthesized policy balances different types of driving risks, including both collision risks and traffic rule violations. The effectiveness of the proposed approach is validated by three typical traffic scenarios in Carla simulator.
>
---
#### [replaced 023] FalconGym: A Photorealistic Simulation Framework for Zero-Shot Sim-to-Real Vision-Based Quadrotor Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.02198v2](http://arxiv.org/pdf/2503.02198v2)**

> **作者:** Yan Miao; Will Shen; Sayan Mitra
>
> **备注:** Accepted in IROS 2025
>
> **摘要:** We present a novel framework demonstrating zero-shot sim-to-real transfer of visual control policies learned in a Neural Radiance Field (NeRF) environment for quadrotors to fly through racing gates. Robust transfer from simulation to real flight poses a major challenge, as standard simulators often lack sufficient visual fidelity. To address this, we construct a photorealistic simulation environment of quadrotor racing tracks, called FalconGym, which provides effectively unlimited synthetic images for training. Within FalconGym, we develop a pipelined approach for crossing gates that combines (i) a Neural Pose Estimator (NPE) coupled with a Kalman filter to reliably infer quadrotor poses from single-frame RGB images and IMU data, and (ii) a self-attention-based multi-modal controller that adaptively integrates visual features and pose estimation. This multi-modal design compensates for perception noise and intermittent gate visibility. We train this controller purely in FalconGym with imitation learning and deploy the resulting policy to real hardware with no additional fine-tuning. Simulation experiments on three distinct tracks (circle, U-turn and figure-8) demonstrate that our controller outperforms a vision-only state-of-the-art baseline in both success rate and gate-crossing accuracy. In 30 live hardware flights spanning three tracks and 120 gates, our controller achieves a 95.8% success rate and an average error of just 10 cm when flying through 38 cm-radius gates.
>
---
#### [replaced 024] PC-SRIF: Preconditioned Cholesky-based Square Root Information Filter for Vision-aided Inertial Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11372v3](http://arxiv.org/pdf/2409.11372v3)**

> **作者:** Tong Ke; Parth Agrawal; Yun Zhang; Weikun Zhen; Chao X. Guo; Toby Sharp; Ryan C. Dutoit
>
> **备注:** This work has been accepted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** In this paper, we introduce a novel estimator for vision-aided inertial navigation systems (VINS), the Preconditioned Cholesky-based Square Root Information Filter (PC-SRIF). When solving linear systems, employing Cholesky decomposition offers superior efficiency but can compromise numerical stability. Due to this, existing VINS utilizing (Square Root) Information Filters often opt for QR decomposition on platforms where single precision is preferred, avoiding the numerical challenges associated with Cholesky decomposition. While these issues are often attributed to the ill-conditioned information matrix in VINS, our analysis reveals that this is not an inherent property of VINS but rather a consequence of specific parameterizations. We identify several factors that contribute to an ill-conditioned information matrix and propose a preconditioning technique to mitigate these conditioning issues. Building on this analysis, we present PC-SRIF, which exhibits remarkable stability in performing Cholesky decomposition in single precision when solving linear systems in VINS. Consequently, PC-SRIF achieves superior theoretical efficiency compared to alternative estimators. To validate the efficiency advantages and numerical stability of PC-SRIF based VINS, we have conducted well controlled experiments, which provide empirical evidence in support of our theoretical findings. Remarkably, in our VINS implementation, PC-SRIF's runtime is 41% faster than QR-based SRIF.
>
---
#### [replaced 025] Evaluating Uncertainty and Quality of Visual Language Action-enabled Robots
- **分类: cs.SE; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.17049v2](http://arxiv.org/pdf/2507.17049v2)**

> **作者:** Pablo Valle; Chengjie Lu; Shaukat Ali; Aitor Arrieta
>
> **摘要:** Visual Language Action (VLA) models are a multi-modal class of Artificial Intelligence (AI) systems that integrate visual perception, natural language understanding, and action planning to enable agents to interpret their environment, comprehend instructions, and perform embodied tasks autonomously. Recently, significant progress has been made to advance this field. These kinds of models are typically evaluated through task success rates, which fail to capture the quality of task execution and the mode's confidence in its decisions. In this paper, we propose eight uncertainty metrics and five quality metrics specifically designed for VLA models for robotic manipulation tasks. We assess their effectiveness through a large-scale empirical study involving 908 successful task executions from three state-of-the-art VLA models across four representative robotic manipulation tasks. Human domain experts manually labeled task quality, allowing us to analyze the correlation between our proposed metrics and expert judgments. The results reveal that several metrics show moderate to strong correlation with human assessments, highlighting their utility for evaluating task quality and model confidence. Furthermore, we found that some of the metrics can discriminate between high-, medium-, and low-quality executions from unsuccessful tasks, which can be interesting when test oracles are not available. Our findings challenge the adequacy of current evaluation practices that rely solely on binary success rates and pave the way for improved real-time monitoring and adaptive enhancement of VLA-enabled robotic systems.
>
---
#### [replaced 026] A Segmented Robot Grasping Perception Neural Network for Edge AI
- **分类: cs.RO; cs.AI; I.2; I.2.9; I.2.10**

- **链接: [http://arxiv.org/pdf/2507.13970v2](http://arxiv.org/pdf/2507.13970v2)**

> **作者:** Casper Bröcheler; Thomas Vroom; Derrick Timmermans; Alan van den Akker; Guangzhi Tang; Charalampos S. Kouzinopoulos; Rico Möckel
>
> **备注:** Accepted by SMC 2025
>
> **摘要:** Robotic grasping, the ability of robots to reliably secure and manipulate objects of varying shapes, sizes and orientations, is a complex task that requires precise perception and control. Deep neural networks have shown remarkable success in grasp synthesis by learning rich and abstract representations of objects. When deployed at the edge, these models can enable low-latency, low-power inference, making real-time grasping feasible in resource-constrained environments. This work implements Heatmap-Guided Grasp Detection, an end-to-end framework for the detection of 6-Dof grasp poses, on the GAP9 RISC-V System-on-Chip. The model is optimised using hardware-aware techniques, including input dimensionality reduction, model partitioning, and quantisation. Experimental evaluation on the GraspNet-1Billion benchmark validates the feasibility of fully on-chip inference, highlighting the potential of low-power MCUs for real-time, autonomous manipulation.
>
---
