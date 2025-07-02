# 机器人 cs.RO

- **最新发布 34 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Control-Optimized Deep Reinforcement Learning for Artificially Intelligent Autonomous Systems
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文属于强化学习任务，解决实际系统中动作执行不匹配的问题。提出一种控制优化的DRL框架，提升智能体在真实环境中的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.00268v1](http://arxiv.org/pdf/2507.00268v1)**

> **作者:** Oren Fivel; Matan Rudman; Kobi Cohen
>
> **备注:** 27 pages, 10 figures
>
> **摘要:** Deep reinforcement learning (DRL) has become a powerful tool for complex decision-making in machine learning and AI. However, traditional methods often assume perfect action execution, overlooking the uncertainties and deviations between an agent's selected actions and the actual system response. In real-world applications, such as robotics, mechatronics, and communication networks, execution mismatches arising from system dynamics, hardware constraints, and latency can significantly degrade performance. This work advances AI by developing a novel control-optimized DRL framework that explicitly models and compensates for action execution mismatches, a challenge largely overlooked in existing methods. Our approach establishes a structured two-stage process: determining the desired action and selecting the appropriate control signal to ensure proper execution. It trains the agent while accounting for action mismatches and controller corrections. By incorporating these factors into the training process, the AI agent optimizes the desired action with respect to both the actual control signal and the intended outcome, explicitly considering execution errors. This approach enhances robustness, ensuring that decision-making remains effective under real-world uncertainties. Our approach offers a substantial advancement for engineering practice by bridging the gap between idealized learning and real-world implementation. It equips intelligent agents operating in engineering environments with the ability to anticipate and adjust for actuation errors and system disturbances during training. We evaluate the framework in five widely used open-source mechanical simulation environments we restructured and developed to reflect real-world operating conditions, showcasing its robustness against uncertainties and offering a highly practical and efficient solution for control-oriented applications.
>
---
#### [new 002] Learning Steerable Imitation Controllers from Unstructured Animal Motions
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在通过动物运动数据生成可操控的仿生行为。工作包括数据转换、运动合成与强化学习控制，实现灵活步态切换和精准速度跟踪。**

- **链接: [http://arxiv.org/pdf/2507.00677v1](http://arxiv.org/pdf/2507.00677v1)**

> **作者:** Dongho Kang; Jin Cheng; Fatemeh Zargarbashi; Taerim Yoon; Sungjoon Choi; Stelian Coros
>
> **备注:** The supplementary video is available at https://youtu.be/DukyUGNYf5A
>
> **摘要:** This paper presents a control framework for legged robots that leverages unstructured real-world animal motion data to generate animal-like and user-steerable behaviors. Our framework learns to follow velocity commands while reproducing the diverse gait patterns in the original dataset. To begin with, animal motion data is transformed into a robot-compatible database using constrained inverse kinematics and model predictive control, bridging the morphological and physical gap between the animal and the robot. Subsequently, a variational autoencoder-based motion synthesis module captures the diverse locomotion patterns in the motion database and generates smooth transitions between them in response to velocity commands. The resulting kinematic motions serve as references for a reinforcement learning-based feedback controller deployed on physical robots. We show that this approach enables a quadruped robot to adaptively switch gaits and accurately track user velocity commands while maintaining the stylistic coherence of the motion data. Additionally, we provide component-wise evaluations to analyze the system's behavior in depth and demonstrate the efficacy of our method for more accurate and reliable motion imitation.
>
---
#### [new 003] VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VQ-VLA，解决视觉-语言-动作模型中的动作分词问题，通过大规模数据提升模型性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.01016v1](http://arxiv.org/pdf/2507.01016v1)**

> **作者:** Yating Wang; Haoyi Zhu; Mingyu Liu; Jiange Yang; Hao-Shu Fang; Tong He
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** In this paper, we introduce an innovative vector quantization based action tokenizer built upon the largest-scale action trajectory dataset to date, leveraging over 100 times more data than previous approaches. This extensive dataset enables our tokenizer to capture rich spatiotemporal dynamics, resulting in a model that not only accelerates inference but also generates smoother and more coherent action outputs. Once trained, the tokenizer can be seamlessly adapted to a wide range of downstream tasks in a zero-shot manner, from short-horizon reactive behaviors to long-horizon planning. A key finding of our work is that the domain gap between synthetic and real action trajectories is marginal, allowing us to effectively utilize a vast amount of synthetic data during training without compromising real-world performance. To validate our approach, we conducted extensive experiments in both simulated environments and on real robotic platforms. The results demonstrate that as the volume of synthetic trajectory data increases, the performance of our tokenizer on downstream tasks improves significantly-most notably, achieving up to a 30% higher success rate on two real-world tasks in long-horizon scenarios. These findings highlight the potential of our action tokenizer as a robust and scalable solution for real-time embodied intelligence systems, paving the way for more efficient and reliable robotic control in diverse application domains.Project website: https://xiaoxiao0406.github.io/vqvla.github.io
>
---
#### [new 004] Parallel Transmission Aware Co-Design: Enhancing Manipulator Performance Through Actuation-Space Optimization
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，解决传统设计与行为优化分离的问题。通过引入并联耦合约束，优化执行器空间以提升机械臂动态负载能力。**

- **链接: [http://arxiv.org/pdf/2507.00644v1](http://arxiv.org/pdf/2507.00644v1)**

> **作者:** Rohit Kumar; Melya Boukheddimi; Dennis Mronga; Shivesh Kumar; Frank Kirchner
>
> **摘要:** In robotics, structural design and behavior optimization have long been considered separate processes, resulting in the development of systems with limited capabilities. Recently, co-design methods have gained popularity, where bi-level formulations are used to simultaneously optimize the robot design and behavior for specific tasks. However, most implementations assume a serial or tree-type model of the robot, overlooking the fact that many robot platforms incorporate parallel mechanisms. In this paper, we present a novel co-design approach that explicitly incorporates parallel coupling constraints into the dynamic model of the robot. In this framework, an outer optimization loop focuses on the design parameters, in our case the transmission ratios of a parallel belt-driven manipulator, which map the desired torques from the joint space to the actuation space. An inner loop performs trajectory optimization in the actuation space, thus exploiting the entire dynamic range of the manipulator. We compare the proposed method with a conventional co-design approach based on a simplified tree-type model. By taking advantage of the actuation space representation, our approach leads to a significant increase in dynamic payload capacity compared to the conventional co-design implementation.
>
---
#### [new 005] Box Pose and Shape Estimation and Domain Adaptation for Large-Scale Warehouse Automation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人视觉任务，解决仓库自动化中箱体位姿与形状估计问题，提出自监督域适应方法，利用未标注数据提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00984v1](http://arxiv.org/pdf/2507.00984v1)**

> **作者:** Xihang Yu; Rajat Talak; Jingnan Shi; Ulrich Viereck; Igor Gilitschenski; Luca Carlone
>
> **备注:** 12 pages, 6 figures. This work will be presented at the 19th International Symposium on Experimental Robotics (ISER2025)
>
> **摘要:** Modern warehouse automation systems rely on fleets of intelligent robots that generate vast amounts of data -- most of which remains unannotated. This paper develops a self-supervised domain adaptation pipeline that leverages real-world, unlabeled data to improve perception models without requiring manual annotations. Our work focuses specifically on estimating the pose and shape of boxes and presents a correct-and-certify pipeline for self-supervised box pose and shape estimation. We extensively evaluate our approach across a range of simulated and real industrial settings, including adaptation to a large-scale real-world dataset of 50,000 images. The self-supervised model significantly outperforms models trained solely in simulation and shows substantial improvements over a zero-shot 3D bounding box estimation baseline.
>
---
#### [new 006] DIJE: Dense Image Jacobian Estimation for Robust Robotic Self-Recognition and Visual Servoing
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉伺服任务，解决机器人自我识别与运动控制问题。提出DIJE算法实时估计图像雅可比矩阵，用于区分自身运动与外部干扰，并实现精准控制。**

- **链接: [http://arxiv.org/pdf/2507.00446v1](http://arxiv.org/pdf/2507.00446v1)**

> **作者:** Yasunori Toshimitsu; Kento Kawaharazuka; Akihiro Miki; Kei Okada; Masayuki Inaba
>
> **备注:** 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** For robots to move in the real world, they must first correctly understand the state of its own body and the tools that it holds. In this research, we propose DIJE, an algorithm to estimate the image Jacobian for every pixel. It is based on an optical flow calculation and a simplified Kalman Filter that can be efficiently run on the whole image in real time. It does not rely on markers nor knowledge of the robotic structure. We use the DIJE in a self-recognition process which can robustly distinguish between movement by the robot and by external entities, even when the motion overlaps. We also propose a visual servoing controller based on DIJE, which can learn to control the robot's body to conduct reaching movements or bimanual tool-tip control. The proposed algorithms were implemented on a physical musculoskeletal robot and its performance was verified. We believe that such global estimation of the visuomotor policy has the potential to be extended into a more general framework for manipulation.
>
---
#### [new 007] I Move Therefore I Learn: Experience-Based Traversability in Outdoor Robotics
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决户外地形可通行性估计问题。通过经验学习方法，使机器人自主识别可行走区域，无需预标注数据。**

- **链接: [http://arxiv.org/pdf/2507.00882v1](http://arxiv.org/pdf/2507.00882v1)**

> **作者:** Miguel Ángel de Miguel; Jorge Beltrán; Juan S. Cely; Francisco Martín; Juan Carlos Manzanares; Alberto García
>
> **摘要:** Accurate traversability estimation is essential for safe and effective navigation of outdoor robots operating in complex environments. This paper introduces a novel experience-based method that allows robots to autonomously learn which terrains are traversable based on prior navigation experience, without relying on extensive pre-labeled datasets. The approach integrates elevation and texture data into multi-layered grid maps, which are processed using a variational autoencoder (VAE) trained on a generic texture dataset. During an initial teleoperated phase, the robot collects sensory data while moving around the environment. These experiences are encoded into compact feature vectors and clustered using the BIRCH algorithm to represent traversable terrain areas efficiently. In deployment, the robot compares new terrain patches to its learned feature clusters to assess traversability in real time. The proposed method does not require training with data from the targeted scenarios, generalizes across diverse surfaces and platforms, and dynamically adapts as new terrains are encountered. Extensive evaluations on both synthetic benchmarks and real-world scenarios with wheeled and legged robots demonstrate its effectiveness, robustness, and superior adaptability compared to state-of-the-art approaches.
>
---
#### [new 008] Edge Computing and its Application in Robotics: A Survey
- **分类: cs.RO; cs.DC; cs.NI**

- **简介: 该论文属于综述任务，旨在探讨边缘计算在机器人领域的应用，解决集成边缘计算带来的挑战与机遇，分析其优势及未来方向。**

- **链接: [http://arxiv.org/pdf/2507.00523v1](http://arxiv.org/pdf/2507.00523v1)**

> **作者:** Nazish Tahir; Ramviyas Parasuraman
>
> **摘要:** The Edge computing paradigm has gained prominence in both academic and industry circles in recent years. By implementing edge computing facilities and services in robotics, it becomes a key enabler in the deployment of artificial intelligence applications to robots. Time-sensitive robotics applications benefit from the reduced latency, mobility, and location awareness provided by the edge computing paradigm, which enables real-time data processing and intelligence at the network's edge. While the advantages of integrating edge computing into robotics are numerous, there has been no recent survey that comprehensively examines these benefits. This paper aims to bridge that gap by highlighting important work in the domain of edge robotics, examining recent advancements, and offering deeper insight into the challenges and motivations behind both current and emerging solutions. In particular, this article provides a comprehensive evaluation of recent developments in edge robotics, with an emphasis on fundamental applications, providing in-depth analysis of the key motivations, challenges, and future directions in this rapidly evolving domain. It also explores the importance of edge computing in real-world robotics scenarios where rapid response times are critical. Finally, the paper outlines various open research challenges in the field of edge robotics.
>
---
#### [new 009] DexWrist: A Robotic Wrist for Constrained and Dynamic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，旨在解决受限环境下的动态操作问题。作者设计了DexWrist，提升机械臂的灵活性与数据收集效率。**

- **链接: [http://arxiv.org/pdf/2507.01008v1](http://arxiv.org/pdf/2507.01008v1)**

> **作者:** Martin Peticco; Gabriella Ulloa; John Marangola; Pulkit Agrawal
>
> **备注:** More details about the wrist can be found at: dexwrist.csail.mit.edu
>
> **摘要:** We present the DexWrist, a compliant robotic wrist designed to advance robotic manipulation in highly-constrained environments, enable dynamic tasks, and speed up data collection. DexWrist is designed to be close to the functional capabilities of the human wrist and achieves mechanical compliance and a greater workspace as compared to existing robotic wrist designs. The DexWrist can supercharge policy learning by (i) enabling faster teleoperation and therefore making data collection more scalable; (ii) completing tasks in fewer steps which reduces trajectory lengths and therefore can ease policy learning; (iii) DexWrist is designed to be torque transparent with easily simulatable kinematics for simulated data collection; and (iv) most importantly expands the workspace of manipulation for approaching highly cluttered scenes and tasks. More details about the wrist can be found at: dexwrist.csail.mit.edu.
>
---
#### [new 010] A Miniature High-Resolution Tension Sensor Based on a Photo-Reflector for Robotic Hands and Grippers
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决高分辨率张力检测问题。设计了一种微型光电反射式张力传感器，用于机械手和夹爪，具有高精度和良好耐用性。**

- **链接: [http://arxiv.org/pdf/2507.00464v1](http://arxiv.org/pdf/2507.00464v1)**

> **作者:** Hyun-Bin Kim; Kyung-Soo Kim
>
> **摘要:** This paper presents a miniature tension sensor using a photo-reflector, designed for compact tendon-driven grippers and robotic hands. The proposed sensor has a small form factor of 13~mm x 7~mm x 6.5~mm and is capable of measuring tensile forces up to 200~N. A symmetric elastomer structure incorporating fillets and flexure hinges is designed based on Timoshenko beam theory and verified via FEM analysis, enabling improved sensitivity and mechanical durability while minimizing torsional deformation. The sensor utilizes a compact photo-reflector (VCNT2020) to measure displacement in the near-field region, eliminating the need for light-absorbing materials or geometric modifications required in photo-interrupter-based designs. A 16-bit analog-to-digital converter (ADC) and CAN-FD (Flexible Data-rate) communication enable efficient signal acquisition with up to 5~kHz sampling rate. Calibration experiments demonstrate a resolution of 9.9~mN (corresponding to over 14-bit accuracy) and a root mean square error (RMSE) of 0.455~N. Force control experiments using a twisted string actuator and PI control yield RMSEs as low as 0.073~N. Compared to previous research using photo-interrupter, the proposed method achieves more than tenfold improvement in resolution while also reducing nonlinearity and hysteresis. The design is mechanically simple, lightweight, easy to assemble, and suitable for integration into robotic and prosthetic systems requiring high-resolution force feedback.
>
---
#### [new 011] PI-WAN: A Physics-Informed Wind-Adaptive Network for Quadrotor Dynamics Prediction in Unknown Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机动力学预测任务，解决未知环境中模型泛化与鲁棒性问题。提出PI-WAN网络，融合物理约束与数据驱动方法，提升预测精度与控制性能。**

- **链接: [http://arxiv.org/pdf/2507.00816v1](http://arxiv.org/pdf/2507.00816v1)**

> **作者:** Mengyun Wang; Bo Wang; Yifeng Niu; Chang Wang
>
> **摘要:** Accurate dynamics modeling is essential for quadrotors to achieve precise trajectory tracking in various applications. Traditional physical knowledge-driven modeling methods face substantial limitations in unknown environments characterized by variable payloads, wind disturbances, and external perturbations. On the other hand, data-driven modeling methods suffer from poor generalization when handling out-of-distribution (OoD) data, restricting their effectiveness in unknown scenarios. To address these challenges, we introduce the Physics-Informed Wind-Adaptive Network (PI-WAN), which combines knowledge-driven and data-driven modeling methods by embedding physical constraints directly into the training process for robust quadrotor dynamics learning. Specifically, PI-WAN employs a Temporal Convolutional Network (TCN) architecture that efficiently captures temporal dependencies from historical flight data, while a physics-informed loss function applies physical principles to improve model generalization and robustness across previously unseen conditions. By incorporating real-time prediction results into a model predictive control (MPC) framework, we achieve improvements in closed-loop tracking performance. Comprehensive simulations and real-world flight experiments demonstrate that our approach outperforms baseline methods in terms of prediction accuracy, tracking precision, and robustness to unknown environments.
>
---
#### [new 012] HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人双臂灵巧操作任务，解决仿真数据不足问题。通过LLM生成操作约束和空间标注，构建高质量数据集。**

- **链接: [http://arxiv.org/pdf/2507.00833v1](http://arxiv.org/pdf/2507.00833v1)**

> **作者:** Zhi Jing; Siyuan Yang; Jicong Ao; Ting Xiao; Yugang Jiang; Chenjia Bai
>
> **备注:** Project Page: https://openhumanoidgen.github.io
>
> **摘要:** For robotic manipulation, existing robotics datasets and simulation benchmarks predominantly cater to robot-arm platforms. However, for humanoid robots equipped with dual arms and dexterous hands, simulation tasks and high-quality demonstrations are notably lacking. Bimanual dexterous manipulation is inherently more complex, as it requires coordinated arm movements and hand operations, making autonomous data collection challenging. This paper presents HumanoidGen, an automated task creation and demonstration collection framework that leverages atomic dexterous operations and LLM reasoning to generate relational constraints. Specifically, we provide spatial annotations for both assets and dexterous hands based on the atomic operations, and perform an LLM planner to generate a chain of actionable spatial constraints for arm movements based on object affordances and scenes. To further improve planning ability, we employ a variant of Monte Carlo tree search to enhance LLM reasoning for long-horizon tasks and insufficient annotation. In experiments, we create a novel benchmark with augmented scenarios to evaluate the quality of the collected data. The results show that the performance of the 2D and 3D diffusion policies can scale with the generated dataset. Project page is https://openhumanoidgen.github.io.
>
---
#### [new 013] Sim2Real Diffusion: Learning Cross-Domain Adaptive Representations for Transferable Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶领域的sim2real任务，旨在解决模拟到现实的迁移问题。通过生成式扩散模型学习跨域自适应表示，提升算法在真实环境中的性能。**

- **链接: [http://arxiv.org/pdf/2507.00236v1](http://arxiv.org/pdf/2507.00236v1)**

> **作者:** Chinmay Vilas Samak; Tanmay Vilas Samak; Bing Li; Venkat Krovi
>
> **摘要:** Simulation-based design, optimization, and validation of autonomous driving algorithms have proven to be crucial for their iterative improvement over the years. Nevertheless, the ultimate measure of effectiveness is their successful transition from simulation to reality (sim2real). However, existing sim2real transfer methods struggle to comprehensively address the autonomy-oriented requirements of balancing: (i) conditioned domain adaptation, (ii) robust performance with limited examples, (iii) modularity in handling multiple domain representations, and (iv) real-time performance. To alleviate these pain points, we present a unified framework for learning cross-domain adaptive representations for sim2real transferable autonomous driving algorithms using conditional latent diffusion models. Our framework offers options to leverage: (i) alternate foundation models, (ii) a few-shot fine-tuning pipeline, and (iii) textual as well as image prompts for mapping across given source and target domains. It is also capable of generating diverse high-quality samples when diffusing across parameter spaces such as times of day, weather conditions, seasons, and operational design domains. We systematically analyze the presented framework and report our findings in the form of critical quantitative metrics and ablation studies, as well as insightful qualitative examples and remarks. Additionally, we demonstrate the serviceability of the proposed approach in bridging the sim2real gap for end-to-end autonomous driving using a behavioral cloning case study. Our experiments indicate that the proposed framework is capable of bridging the perceptual sim2real gap by over 40%. We hope that our approach underscores the potential of generative diffusion models in sim2real transfer, offering a pathway toward more robust and adaptive autonomous driving.
>
---
#### [new 014] RaGNNarok: A Light-Weight Graph Neural Network for Enhancing Radar Point Clouds on Unmanned Ground Vehicles
- **分类: cs.RO; cs.AR; cs.CV; cs.LG**

- **简介: 该论文属于机器人感知任务，旨在解决雷达点云稀疏和噪声问题。提出RaGNNarok框架，提升雷达数据质量，适用于低成本移动机器人。**

- **链接: [http://arxiv.org/pdf/2507.00937v1](http://arxiv.org/pdf/2507.00937v1)**

> **作者:** David Hunt; Shaocheng Luo; Spencer Hallyburton; Shafii Nillongo; Yi Li; Tingjun Chen; Miroslav Pajic
>
> **备注:** 8 pages, accepted by IROS 2025
>
> **摘要:** Low-cost indoor mobile robots have gained popularity with the increasing adoption of automation in homes and commercial spaces. However, existing lidar and camera-based solutions have limitations such as poor performance in visually obscured environments, high computational overhead for data processing, and high costs for lidars. In contrast, mmWave radar sensors offer a cost-effective and lightweight alternative, providing accurate ranging regardless of visibility. However, existing radar-based localization suffers from sparse point cloud generation, noise, and false detections. Thus, in this work, we introduce RaGNNarok, a real-time, lightweight, and generalizable graph neural network (GNN)-based framework to enhance radar point clouds, even in complex and dynamic environments. With an inference time of just 7.3 ms on the low-cost Raspberry Pi 5, RaGNNarok runs efficiently even on such resource-constrained devices, requiring no additional computational resources. We evaluate its performance across key tasks, including localization, SLAM, and autonomous navigation, in three different environments. Our results demonstrate strong reliability and generalizability, making RaGNNarok a robust solution for low-cost indoor mobile robots.
>
---
#### [new 015] Generation of Indoor Open Street Maps for Robot Navigation from CAD Files
- **分类: cs.RO**

- **简介: 该论文属于室内地图生成任务，旨在解决传统SLAM方法在动态大场景中效率低、易失效的问题。通过CAD文件自动生成结构化OSM地图，提升机器人导航的鲁棒性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2507.00552v1](http://arxiv.org/pdf/2507.00552v1)**

> **作者:** Jiajie Zhang; Shenrui Wu; Xu Ma; Sören Schwertfeger
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** The deployment of autonomous mobile robots is predicated on the availability of environmental maps, yet conventional generation via SLAM (Simultaneous Localization and Mapping) suffers from significant limitations in time, labor, and robustness, particularly in dynamic, large-scale indoor environments where map obsolescence can lead to critical localization failures. To address these challenges, this paper presents a complete and automated system for converting architectural Computer-Aided Design (CAD) files into a hierarchical topometric OpenStreetMap (OSM) representation, tailored for robust life-long robot navigation. Our core methodology involves a multi-stage pipeline that first isolates key structural layers from the raw CAD data and then employs an AreaGraph-based topological segmentation to partition the building layout into a hierarchical graph of navigable spaces. This process yields a comprehensive and semantically rich map, further enhanced by automatically associating textual labels from the CAD source and cohesively merging multiple building floors into a unified, topologically-correct model. By leveraging the permanent structural information inherent in CAD files, our system circumvents the inefficiencies and fragility of SLAM, offering a practical and scalable solution for deploying robots in complex indoor spaces. The software is encapsulated within an intuitive Graphical User Interface (GUI) to facilitate practical use. The code and dataset are available at https://github.com/jiajiezhang7/osmAG-from-cad.
>
---
#### [new 016] Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，解决无需物理演示的机器人学习问题。通过生成视频并提取轨迹，实现机器人复杂操作。**

- **链接: [http://arxiv.org/pdf/2507.00990v1](http://arxiv.org/pdf/2507.00990v1)**

> **作者:** Shivansh Patel; Shraddhaa Mohan; Hanlin Mai; Unnat Jain; Svetlana Lazebnik; Yunzhu Li
>
> **备注:** Project Page: https://rigvid-robot.github.io/
>
> **摘要:** This work introduces Robots Imitating Generated Videos (RIGVid), a system that enables robots to perform complex manipulation tasks--such as pouring, wiping, and mixing--purely by imitating AI-generated videos, without requiring any physical demonstrations or robot-specific training. Given a language command and an initial scene image, a video diffusion model generates potential demonstration videos, and a vision-language model (VLM) automatically filters out results that do not follow the command. A 6D pose tracker then extracts object trajectories from the video, and the trajectories are retargeted to the robot in an embodiment-agnostic fashion. Through extensive real-world evaluations, we show that filtered generated videos are as effective as real demonstrations, and that performance improves with generation quality. We also show that relying on generated videos outperforms more compact alternatives such as keypoint prediction using VLMs, and that strong 6D pose tracking outperforms other ways to extract trajectories, such as dense feature point tracking. These findings suggest that videos produced by a state-of-the-art off-the-shelf model can offer an effective source of supervision for robotic manipulation.
>
---
#### [new 017] Novel Design of 3D Printed Tumbling Microrobots for in vivo Targeted Drug Delivery
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决靶向药物输送问题。设计3D打印的翻滚微机器人，通过磁力驱动，在体内实现精准给药。**

- **链接: [http://arxiv.org/pdf/2507.00166v1](http://arxiv.org/pdf/2507.00166v1)**

> **作者:** Aaron C. Davis; Siting Zhang; Adalyn Meeks; Diya Sakhrani; Luis Carlos Sanjuan Acosta; D. Ethan Kelley; Emma Caldwell; Luis Solorio; Craig J. Goergen; David J. Cappelleri
>
> **摘要:** This paper presents innovative designs for 3D-printed tumbling microrobots, specifically engineered for targeted in vivo drug delivery applications. The microrobot designs, created using stereolithography 3D printing technologies, incorporate permanent micro-magnets to enable actuation via a rotating magnetic field actuator system. The experimental framework encompasses a series of locomotion characterization tests to evaluate microrobot performance under various conditions. Testing variables include variations in microrobot geometries, actuation frequencies, and environmental conditions, such as dry and wet environments, and temperature changes. The paper outlines designs for three drug loading methods, along with comprehensive assessments thermal drug release using a focused ultrasound system, as well as biocompatibility tests. Animal model testing involves tissue phantoms and in vivo rat models, ensuring a thorough evaluation of the microrobots' performance and compatibility. The results highlight the robustness and adaptability of the proposed microrobot designs, showcasing the potential for efficient and targeted in vivo drug delivery. This novel approach addresses current limitations in existing tumbling microrobot designs and paves the way for advancements in targeted drug delivery within the large intestine.
>
---
#### [new 018] Novel Pigeon-inspired 3D Obstacle Detection and Avoidance Maneuver for Multi-UAV Systems
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多无人机避障任务，旨在解决城市环境中静态与动态障碍物的避让问题。通过仿生方法设计了3D避障框架，实现了多无人机的协同控制与路径规划。**

- **链接: [http://arxiv.org/pdf/2507.00443v1](http://arxiv.org/pdf/2507.00443v1)**

> **作者:** Reza Ahmadvand; Sarah Safura Sharif; Yaser Mike Banad
>
> **备注:** 11 Pages, 11 Pictures, 1 Table, 3 Algorithms
>
> **摘要:** Recent advances in multi-agent systems manipulation have demonstrated a rising demand for the implementation of multi-UAV systems in urban areas, which are always subjected to the presence of static and dynamic obstacles. Inspired by the collective behavior of tilapia fish and pigeons, the focus of the presented research is on the introduction of a nature-inspired collision-free formation control for a multi-UAV system, considering the obstacle avoidance maneuvers. The developed framework in this study utilizes a semi-distributed control approach, in which, based on a probabilistic Lloyd's algorithm, a centralized guidance algorithm works for optimal positioning of the UAVs, while a distributed control approach has been used for the intervehicle collision and obstacle avoidance. Further, the presented framework has been extended to the 3D space with a novel definition of 3D maneuvers. Finally, the presented framework has been applied to multi-UAV systems in 2D and 3D scenarios, and the obtained results demonstrated the validity of the presented method in dynamic environments with stationary and moving obstacles.
>
---
#### [new 019] A Survey: Learning Embodied Intelligence from Physical Simulators and World Models
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在提升智能体的环境感知与决策能力。通过结合物理模拟器和世界模型，解决真实世界部署中的适应性与泛化问题。**

- **链接: [http://arxiv.org/pdf/2507.00917v1](http://arxiv.org/pdf/2507.00917v1)**

> **作者:** Xiaoxiao Long; Qingrui Zhao; Kaiwen Zhang; Zihao Zhang; Dingrui Wang; Yumeng Liu; Zhengjie Shu; Yi Lu; Shouzheng Wang; Xinzhe Wei; Wei Li; Wei Yin; Yao Yao; Jia Pan; Qiu Shen; Ruigang Yang; Xun Cao; Qionghai Dai
>
> **备注:** https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey
>
> **摘要:** The pursuit of artificial general intelligence (AGI) has placed embodied intelligence at the forefront of robotics research. Embodied intelligence focuses on agents capable of perceiving, reasoning, and acting within the physical world. Achieving robust embodied intelligence requires not only advanced perception and control, but also the ability to ground abstract cognition in real-world interactions. Two foundational technologies, physical simulators and world models, have emerged as critical enablers in this quest. Physical simulators provide controlled, high-fidelity environments for training and evaluating robotic agents, allowing safe and efficient development of complex behaviors. In contrast, world models empower robots with internal representations of their surroundings, enabling predictive planning and adaptive decision-making beyond direct sensory input. This survey systematically reviews recent advances in learning embodied AI through the integration of physical simulators and world models. We analyze their complementary roles in enhancing autonomy, adaptability, and generalization in intelligent robots, and discuss the interplay between external simulation and internal modeling in bridging the gap between simulated training and real-world deployment. By synthesizing current progress and identifying open challenges, this survey aims to provide a comprehensive perspective on the path toward more capable and generalizable embodied AI systems. We also maintain an active repository that contains up-to-date literature and open-source projects at https://github.com/NJU3DV-LoongGroup/Embodied-World-Models-Survey.
>
---
#### [new 020] When Digital Twins Meet Large Language Models: Realistic, Interactive, and Editable Simulation for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶仿真任务，旨在解决动态真实、渲染逼真、场景相关和实时性能的平衡问题。通过融合物理与数据驱动方法，构建高保真数字孪生系统，并引入大语言模型实现自然语言编辑场景。**

- **链接: [http://arxiv.org/pdf/2507.00319v1](http://arxiv.org/pdf/2507.00319v1)**

> **作者:** Tanmay Vilas Samak; Chinmay Vilas Samak; Bing Li; Venkat Krovi
>
> **摘要:** Simulation frameworks have been key enablers for the development and validation of autonomous driving systems. However, existing methods struggle to comprehensively address the autonomy-oriented requirements of balancing: (i) dynamical fidelity, (ii) photorealistic rendering, (iii) context-relevant scenario orchestration, and (iv) real-time performance. To address these limitations, we present a unified framework for creating and curating high-fidelity digital twins to accelerate advancements in autonomous driving research. Our framework leverages a mix of physics-based and data-driven techniques for developing and simulating digital twins of autonomous vehicles and their operating environments. It is capable of reconstructing real-world scenes and assets (real2sim) with geometric and photorealistic accuracy and infusing them with various physical properties to enable real-time dynamical simulation of the ensuing driving scenarios. Additionally, it also incorporates a large language model (LLM) interface to flexibly edit the driving scenarios online via natural language prompts. We analyze the presented framework in terms of its fidelity, performance, and serviceability. Results indicate that our framework can reconstruct 3D scenes and assets with up to 97% structural similarity, while maintaining frame rates above 60 Hz. We also demonstrate that it can handle natural language prompts to generate diverse driving scenarios with up to 95% repeatability and 85% generalizability.
>
---
#### [new 021] RoboEval: Where Robotic Manipulation Meets Structured and Scalable Evaluation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出RoboEval，用于评估双臂机器人操作性能。解决传统评估指标不足的问题，通过结构化任务和细粒度指标揭示策略缺陷。**

- **链接: [http://arxiv.org/pdf/2507.00435v1](http://arxiv.org/pdf/2507.00435v1)**

> **作者:** Yi Ru Wang; Carter Ung; Grant Tannert; Jiafei Duan; Josephine Li; Amy Le; Rishabh Oswal; Markus Grotz; Wilbert Pumacay; Yuquan Deng; Ranjay Krishna; Dieter Fox; Siddhartha Srinivasa
>
> **备注:** Project page: https://robo-eval.github.io
>
> **摘要:** We present RoboEval, a simulation benchmark and structured evaluation framework designed to reveal the limitations of current bimanual manipulation policies. While prior benchmarks report only binary task success, we show that such metrics often conceal critical weaknesses in policy behavior -- such as poor coordination, slipping during grasping, or asymmetric arm usage. RoboEval introduces a suite of tiered, semantically grounded tasks decomposed into skill-specific stages, with variations that systematically challenge spatial, physical, and coordination capabilities. Tasks are paired with fine-grained diagnostic metrics and 3000+ human demonstrations to support imitation learning. Our experiments reveal that policies with similar success rates diverge in how tasks are executed -- some struggle with alignment, others with temporally consistent bimanual control. We find that behavioral metrics correlate with success in over half of task-metric pairs, and remain informative even when binary success saturates. By pinpointing when and how policies fail, RoboEval enables a deeper, more actionable understanding of robotic manipulation -- and highlights the need for evaluation tools that go beyond success alone.
>
---
#### [new 022] Rethink 3D Object Detection from Physical World
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D目标检测任务，解决实时系统中速度与精度的权衡问题。提出L-AP和P-AP新指标，优化模型与硬件选择。**

- **链接: [http://arxiv.org/pdf/2507.00190v1](http://arxiv.org/pdf/2507.00190v1)**

> **作者:** Satoshi Tanaka; Koji Minoda; Fumiya Watanabe; Takamasa Horibe
>
> **备注:** 15 pages, 10 figures
>
> **摘要:** High-accuracy and low-latency 3D object detection is essential for autonomous driving systems. While previous studies on 3D object detection often evaluate performance based on mean average precision (mAP) and latency, they typically fail to address the trade-off between speed and accuracy, such as 60.0 mAP at 100 ms vs 61.0 mAP at 500 ms. A quantitative assessment of the trade-offs between different hardware devices and accelerators remains unexplored, despite being critical for real-time applications. Furthermore, they overlook the impact on collision avoidance in motion planning, for example, 60.0 mAP leading to safer motion planning or 61.0 mAP leading to high-risk motion planning. In this paper, we introduce latency-aware AP (L-AP) and planning-aware AP (P-AP) as new metrics, which consider the physical world such as the concept of time and physical constraints, offering a more comprehensive evaluation for real-time 3D object detection. We demonstrate the effectiveness of our metrics for the entire autonomous driving system using nuPlan dataset, and evaluate 3D object detection models accounting for hardware differences and accelerators. We also develop a state-of-the-art performance model for real-time 3D object detection through latency-aware hyperparameter optimization (L-HPO) using our metrics. Additionally, we quantitatively demonstrate that the assumption "the more point clouds, the better the recognition performance" is incorrect for real-time applications and optimize both hardware and model selection using our metrics.
>
---
#### [new 023] Mechanical Intelligence-Aware Curriculum Reinforcement Learning for Humanoids with Parallel Actuation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决平行驱动机构在强化学习中的建模问题。通过端到端课程强化学习框架，提升人形机器人运动性能。**

- **链接: [http://arxiv.org/pdf/2507.00273v1](http://arxiv.org/pdf/2507.00273v1)**

> **作者:** Yusuke Tanaka; Alvin Zhu; Quanyou Wang; Dennis Hong
>
> **摘要:** Reinforcement learning (RL) has enabled significant advances in humanoid robot locomotion, yet most learning frameworks do not account for mechanical intelligence embedded in parallel actuation mechanisms due to limitations in simulator support for closed kinematic chains. This omission can lead to inaccurate motion modeling and suboptimal policies, particularly for robots with high actuation complexity. This paper presents an end-to-end curriculum RL framework for BRUCE, a kid-sized humanoid robot featuring three distinct parallel mechanisms in its legs: a differential pulley, a 5-bar linkage, and a 4-bar linkage. Unlike prior approaches that rely on simplified serial approximations, we simulate all closed-chain constraints natively using GPU-accelerated MJX (MuJoCo), preserving the hardware's physical properties during training. We benchmark our RL approach against a Model Predictive Controller (MPC), demonstrating better surface generalization and performance in real-world zero-shot deployment. This work highlights the computational approaches and performance benefits of fully simulating parallel mechanisms in end-to-end learning pipelines for legged humanoids.
>
---
#### [new 024] Evo-0: Vision-Language-Action Model with Implicit Spatial Understanding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决VLA模型缺乏精确空间理解的问题。通过引入隐式3D几何特征提升模型性能。**

- **链接: [http://arxiv.org/pdf/2507.00416v1](http://arxiv.org/pdf/2507.00416v1)**

> **作者:** Tao Lin; Gen Li; Yilei Zhong; Yanwen Zou; Bo Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising framework for enabling generalist robots capable of perceiving, reasoning, and acting in the real world. These models usually build upon pretrained Vision-Language Models (VLMs), which excel at semantic understanding due to large-scale text pretraining. However, VLMs typically lack precise spatial understanding capabilities, as they are primarily tuned on 2D image-text pairs without 3D supervision. To address this limitation, recent approaches have incorporated explicit 3D inputs such as point clouds or depth maps, but this necessitates additional depth sensors or defective estimation. In contrast, our work introduces a plug-and-play module that implicitly injects 3D geometry features into VLA models by leveraging an off-the-shelf visual geometry foundation models. We design five spatially challenging tasks that require precise spatial understanding ability to validate effectiveness of our method. Extensive evaluations show that our method significantly improves the performance of state-of-the-art VLA models across diverse scenarios.
>
---
#### [new 025] Stable Tracking of Eye Gaze Direction During Ophthalmic Surgery
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文属于眼动追踪任务，旨在解决手术中眼位估计不准确的问题。通过结合机器学习与传统算法，实现稳定的眼球定位与跟踪。**

- **链接: [http://arxiv.org/pdf/2507.00635v1](http://arxiv.org/pdf/2507.00635v1)**

> **作者:** Tinghe Hong; Shenlin Cai; Boyang Li; Kai Huang
>
> **备注:** Accepted by ICRA 2025
>
> **摘要:** Ophthalmic surgical robots offer superior stability and precision by reducing the natural hand tremors of human surgeons, enabling delicate operations in confined surgical spaces. Despite the advancements in developing vision- and force-based control methods for surgical robots, preoperative navigation remains heavily reliant on manual operation, limiting the consistency and increasing the uncertainty. Existing eye gaze estimation techniques in the surgery, whether traditional or deep learning-based, face challenges including dependence on additional sensors, occlusion issues in surgical environments, and the requirement for facial detection. To address these limitations, this study proposes an innovative eye localization and tracking method that combines machine learning with traditional algorithms, eliminating the requirements of landmarks and maintaining stable iris detection and gaze estimation under varying lighting and shadow conditions. Extensive real-world experiment results show that our proposed method has an average estimation error of 0.58 degrees for eye orientation estimation and 2.08-degree average control error for the robotic arm's movement based on the calculated orientation.
>
---
#### [new 026] GaussianVLM: Scene-centric 3D Vision-Language Models using Language-aligned Gaussian Splats for Embodied Reasoning and Beyond
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GaussianVLM，解决3D场景理解中的多模态对齐问题，通过语言对齐高斯点云实现场景表示，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00886v1](http://arxiv.org/pdf/2507.00886v1)**

> **作者:** Anna-Maria Halacheva; Jan-Nico Zaech; Xi Wang; Danda Pani Paudel; Luc Van Gool
>
> **摘要:** As multimodal language models advance, their application to 3D scene understanding is a fast-growing frontier, driving the development of 3D Vision-Language Models (VLMs). Current methods show strong dependence on object detectors, introducing processing bottlenecks and limitations in taxonomic flexibility. To address these limitations, we propose a scene-centric 3D VLM for 3D Gaussian splat scenes that employs language- and task-aware scene representations. Our approach directly embeds rich linguistic features into the 3D scene representation by associating language with each Gaussian primitive, achieving early modality alignment. To process the resulting dense representations, we introduce a dual sparsifier that distills them into compact, task-relevant tokens via task-guided and location-guided pathways, producing sparse, task-aware global and local scene tokens. Notably, we present the first Gaussian splatting-based VLM, leveraging photorealistic 3D representations derived from standard RGB images, demonstrating strong generalization: it improves performance of prior 3D VLM five folds, in out-of-the-domain settings.
>
---
#### [new 027] Residual Reward Models for Preference-based Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于偏好强化学习任务，旨在解决传统方法收敛慢的问题。通过引入残差奖励模型，结合先验奖励与学习奖励，提升策略学习效率。**

- **链接: [http://arxiv.org/pdf/2507.00611v1](http://arxiv.org/pdf/2507.00611v1)**

> **作者:** Chenyang Cao; Miguel Rogel-García; Mohamed Nabail; Xueqian Wang; Nicholas Rhinehart
>
> **备注:** 26 pages, 22 figures
>
> **摘要:** Preference-based Reinforcement Learning (PbRL) provides a way to learn high-performance policies in environments where the reward signal is hard to specify, avoiding heuristic and time-consuming reward design. However, PbRL can suffer from slow convergence speed since it requires training in a reward model. Prior work has proposed learning a reward model from demonstrations and fine-tuning it using preferences. However, when the model is a neural network, using different loss functions for pre-training and fine-tuning can pose challenges to reliable optimization. In this paper, we propose a method to effectively leverage prior knowledge with a Residual Reward Model (RRM). An RRM assumes that the true reward of the environment can be split into a sum of two parts: a prior reward and a learned reward. The prior reward is a term available before training, for example, a user's ``best guess'' reward function, or a reward function learned from inverse reinforcement learning (IRL), and the learned reward is trained with preferences. We introduce state-based and image-based versions of RRM and evaluate them on several tasks in the Meta-World environment suite. Experimental results show that our method substantially improves the performance of a common PbRL method. Our method achieves performance improvements for a variety of different types of prior rewards, including proxy rewards, a reward obtained from IRL, and even a negated version of the proxy reward. We also conduct experiments with a Franka Panda to show that our method leads to superior performance on a real robot. It significantly accelerates policy learning for different tasks, achieving success in fewer steps than the baseline. The videos are presented at https://sunlighted.github.io/RRM-web/.
>
---
#### [new 028] Towards Open-World Human Action Segmentation Using Graph Convolutional Networks
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人类动作分割任务，解决开放世界中新动作识别与分割问题。提出一种基于图卷积网络的框架，结合数据增强和时序聚类损失，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.00756v1](http://arxiv.org/pdf/2507.00756v1)**

> **作者:** Hao Xing; Kai Zhe Boey; Gordon Cheng
>
> **备注:** 8 pages, 3 figures, accepted in IROS25, Hangzhou, China
>
> **摘要:** Human-object interaction segmentation is a fundamental task of daily activity understanding, which plays a crucial role in applications such as assistive robotics, healthcare, and autonomous systems. Most existing learning-based methods excel in closed-world action segmentation, they struggle to generalize to open-world scenarios where novel actions emerge. Collecting exhaustive action categories for training is impractical due to the dynamic diversity of human activities, necessitating models that detect and segment out-of-distribution actions without manual annotation. To address this issue, we formally define the open-world action segmentation problem and propose a structured framework for detecting and segmenting unseen actions. Our framework introduces three key innovations: 1) an Enhanced Pyramid Graph Convolutional Network (EPGCN) with a novel decoder module for robust spatiotemporal feature upsampling. 2) Mixup-based training to synthesize out-of-distribution data, eliminating reliance on manual annotations. 3) A novel Temporal Clustering loss that groups in-distribution actions while distancing out-of-distribution samples. We evaluate our framework on two challenging human-object interaction recognition datasets: Bimanual Actions and 2 Hands and Object (H2O) datasets. Experimental results demonstrate significant improvements over state-of-the-art action segmentation models across multiple open-set evaluation metrics, achieving 16.9% and 34.6% relative gains in open-set segmentation (F1@50) and out-of-distribution detection performances (AUROC), respectively. Additionally, we conduct an in-depth ablation study to assess the impact of each proposed component, identifying the optimal framework configuration for open-world action segmentation.
>
---
#### [new 029] User Concerns Regarding Social Robots for Mood Regulation: A Case Study on the "Sunday Blues"
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互领域，探讨用户对社交机器人用于情绪调节的担忧。通过案例研究，分析用户对“周日低落”情境下机器人应用的看法，提出设计改进方向。**

- **链接: [http://arxiv.org/pdf/2507.00271v1](http://arxiv.org/pdf/2507.00271v1)**

> **作者:** Zhuochao Peng; Jiaxin Xu; Jun Hu; Haian Xue; Laurens A. G. Kolks; Pieter M. A. Desmet
>
> **备注:** Accepted to International Conference on Social Robotics + AI (ICSR 2025)
>
> **摘要:** While recent research highlights the potential of social robots to support mood regulation, little is known about how prospective users view their integration into everyday life. To explore this, we conducted an exploratory case study that used a speculative robot concept "Mora" to provoke reflection and facilitate meaningful discussion about using social robots to manage subtle, day-to-day emotional experiences. We focused on the "Sunday Blues," a common dip in mood that occurs at the end of the weekend, as a relatable context in which to explore individuals' insights. Using a video prototype and a co-constructing stories method, we engaged 15 participants in imagining interactions with Mora and discussing their expectations, doubts, and concerns. The study surfaced a range of nuanced reflections around the attributes of social robots like empathy, intervention effectiveness, and ethical boundaries, which we translated into design considerations for future research and development in human-robot interaction.
>
---
#### [new 030] Time Invariant Sensor Tasking for Catalog Maintenance of LEO Space objects using Stochastic Geometry
- **分类: astro-ph.IM; cs.RO; cs.SY; eess.SY; math-ph; math.MP**

- **简介: 该论文属于空间目标跟踪任务，旨在解决有限地面传感器下LEO物体目录维护问题，通过随机几何方法优化传感器调度。**

- **链接: [http://arxiv.org/pdf/2507.00076v1](http://arxiv.org/pdf/2507.00076v1)**

> **作者:** Partha Chowdhury; Harsha M; Chinni Prabhunath Georg; Arun Balaji Buduru; Sanat K Biswas
>
> **备注:** This work has been accepted and presented at the 35th AAS/AIAA Space Flight Mechanics Meeting, 2025, Kaua'i, Hawai
>
> **摘要:** Catalog maintenance of space objects by limited number of ground-based sensors presents a formidable challenging task to the space community. This article presents a methodology for time-invariant tracking and surveillance of space objects in low Earth orbit (LEO) by optimally directing ground sensors. Our methodology aims to maximize the expected number of space objects from a set of ground stations by utilizing concepts from stochastic geometry, particularly the Poisson point process. We have provided a systematic framework to understand visibility patterns and enhance the efficiency of tracking multiple objects simultaneously. Our approach contributes to more informed decision-making in space operations, ultimately supporting efforts to maintain safety and sustainability in LEO.
>
---
#### [new 031] SurgiSR4K: A High-Resolution Endoscopic Video Dataset for Robotic-Assisted Minimally Invasive Procedures
- **分类: eess.IV; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出SurgiSR4K数据集，解决机器人辅助微创手术中高分辨率影像数据不足的问题，支持多种计算机视觉任务。**

- **链接: [http://arxiv.org/pdf/2507.00209v1](http://arxiv.org/pdf/2507.00209v1)**

> **作者:** Fengyi Jiang; Xiaorui Zhang; Lingbo Jin; Ruixing Liang; Yuxin Chen; Adi Chola Venkatesh; Jason Culman; Tiantian Wu; Lirong Shao; Wenqing Sun; Cong Gao; Hallie McNamara; Jingpei Lu; Omid Mohareri
>
> **摘要:** High-resolution imaging is crucial for enhancing visual clarity and enabling precise computer-assisted guidance in minimally invasive surgery (MIS). Despite the increasing adoption of 4K endoscopic systems, there remains a significant gap in publicly available native 4K datasets tailored specifically for robotic-assisted MIS. We introduce SurgiSR4K, the first publicly accessible surgical imaging and video dataset captured at a native 4K resolution, representing realistic conditions of robotic-assisted procedures. SurgiSR4K comprises diverse visual scenarios including specular reflections, tool occlusions, bleeding, and soft tissue deformations, meticulously designed to reflect common challenges faced during laparoscopic and robotic surgeries. This dataset opens up possibilities for a broad range of computer vision tasks that might benefit from high resolution data, such as super resolution (SR), smoke removal, surgical instrument detection, 3D tissue reconstruction, monocular depth estimation, instance segmentation, novel view synthesis, and vision-language model (VLM) development. SurgiSR4K provides a robust foundation for advancing research in high-resolution surgical imaging and fosters the development of intelligent imaging technologies aimed at enhancing performance, safety, and usability in image-guided robotic surgeries.
>
---
#### [new 032] Multi-Modal Graph Convolutional Network with Sinusoidal Encoding for Robust Human Action Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人体动作分割任务，旨在解决因噪声导致的过分割问题。通过多模态图卷积网络和正弦编码等方法提升分割精度。**

- **链接: [http://arxiv.org/pdf/2507.00752v1](http://arxiv.org/pdf/2507.00752v1)**

> **作者:** Hao Xing; Kai Zhe Boey; Yuankai Wu; Darius Burschka; Gordon Cheng
>
> **备注:** 7 pages, 4 figures, accepted in IROS25, Hangzhou, China
>
> **摘要:** Accurate temporal segmentation of human actions is critical for intelligent robots in collaborative settings, where a precise understanding of sub-activity labels and their temporal structure is essential. However, the inherent noise in both human pose estimation and object detection often leads to over-segmentation errors, disrupting the coherence of action sequences. To address this, we propose a Multi-Modal Graph Convolutional Network (MMGCN) that integrates low-frame-rate (e.g., 1 fps) visual data with high-frame-rate (e.g., 30 fps) motion data (skeleton and object detections) to mitigate fragmentation. Our framework introduces three key contributions. First, a sinusoidal encoding strategy that maps 3D skeleton coordinates into a continuous sin-cos space to enhance spatial representation robustness. Second, a temporal graph fusion module that aligns multi-modal inputs with differing resolutions via hierarchical feature aggregation, Third, inspired by the smooth transitions inherent to human actions, we design SmoothLabelMix, a data augmentation technique that mixes input sequences and labels to generate synthetic training examples with gradual action transitions, enhancing temporal consistency in predictions and reducing over-segmentation artifacts. Extensive experiments on the Bimanual Actions Dataset, a public benchmark for human-object interaction understanding, demonstrate that our approach outperforms state-of-the-art methods, especially in action segmentation accuracy, achieving F1@10: 94.5% and F1@25: 92.8%.
>
---
#### [new 033] Audio-3DVG: Unified Audio - Point Cloud Fusion for 3D Visual Grounding
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于3D视觉定位任务，解决音频引导的3D物体定位问题。提出Audio-3DVG框架，融合音频与点云信息，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2507.00669v1](http://arxiv.org/pdf/2507.00669v1)**

> **作者:** Duc Cao-Dinh; Khai Le-Duc; Anh Dao; Bach Phan Tat; Chris Ngo; Duy M. H. Nguyen; Nguyen X. Khanh; Thanh Nguyen-Tang
>
> **备注:** Work in progress, 42 pages
>
> **摘要:** 3D Visual Grounding (3DVG) involves localizing target objects in 3D point clouds based on natural language. While prior work has made strides using textual descriptions, leveraging spoken language-known as Audio-based 3D Visual Grounding-remains underexplored and challenging. Motivated by advances in automatic speech recognition (ASR) and speech representation learning, we propose Audio-3DVG, a simple yet effective framework that integrates audio and spatial information for enhanced grounding. Rather than treating speech as a monolithic input, we decompose the task into two complementary components. First, we introduce Object Mention Detection, a multi-label classification task that explicitly identifies which objects are referred to in the audio, enabling more structured audio-scene reasoning. Second, we propose an Audio-Guided Attention module that captures interactions between candidate objects and relational speech cues, improving target discrimination in cluttered scenes. To support benchmarking, we synthesize audio descriptions for standard 3DVG datasets, including ScanRefer, Sr3D, and Nr3D. Experimental results demonstrate that Audio-3DVG not only achieves new state-of-the-art performance in audio-based grounding, but also competes with text-based methods-highlighting the promise of integrating spoken language into 3D vision tasks.
>
---
#### [new 034] Social Robots for People with Dementia: A Literature Review on Deception from Design to Perception
- **分类: cs.HC; cs.CY; cs.RO**

- **简介: 该论文属于文献综述任务，探讨社会机器人在阿尔茨海默病护理中的欺骗问题，分析设计线索如何引发误解，并提出基于认知机制的欺骗定义。**

- **链接: [http://arxiv.org/pdf/2507.00963v1](http://arxiv.org/pdf/2507.00963v1)**

> **作者:** Fan Wang; Giulia Perugia; Yuan Feng; Wijnand IJsselsteijn
>
> **摘要:** As social robots increasingly enter dementia care, concerns about deception, intentional or not, are gaining attention. Yet, how robotic design cues might elicit misleading perceptions in people with dementia, and how these perceptions arise, remains insufficiently understood. In this scoping review, we examined 26 empirical studies on interactions between people with dementia and physical social robots. We identify four key design cue categories that may influence deceptive impressions: cues resembling physiological signs (e.g., simulated breathing), social intentions (e.g., playful movement), familiar beings (e.g., animal-like form and sound), and, to a lesser extent, cues that reveal artificiality. Thematic analysis of user responses reveals that people with dementia often attribute biological, social, and mental capacities to robots, dynamically shifting between awareness and illusion. These findings underscore the fluctuating nature of ontological perception in dementia contexts. Existing definitions of robotic deception often rest on philosophical or behaviorist premises, but rarely engage with the cognitive mechanisms involved. We propose an empirically grounded definition: robotic deception occurs when Type 1 (automatic, heuristic) processing dominates over Type 2 (deliberative, analytic) reasoning, leading to misinterpretation of a robot's artificial nature. This dual-process perspective highlights the ethical complexity of social robots in dementia care and calls for design approaches that are not only engaging, but also epistemically respectful.
>
---
## 更新

#### [replaced 001] Flatness-based Finite-Horizon Multi-UAV Formation Trajectory Planning and Directionally Aware Collision Avoidance Tracking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.23129v2](http://arxiv.org/pdf/2506.23129v2)**

> **作者:** Hossein B. Jond; Logan Beaver; Martin Jiroušek; Naiemeh Ahmadlou; Veli Bakırcıoğlu; Martin Saska
>
> **备注:** Accepted for Journal of the Franklin Institute
>
> **摘要:** Optimal collision-free formation control of the unmanned aerial vehicle (UAV) is a challenge. The state-of-the-art optimal control approaches often rely on numerical methods sensitive to initial guesses. This paper presents an innovative collision-free finite-time formation control scheme for multiple UAVs leveraging the differential flatness of the UAV dynamics, eliminating the need for numerical methods. We formulate a finite-time optimal control problem to plan a formation trajectory for feasible initial states. This optimal control problem in formation trajectory planning involves a collective performance index to meet the formation requirements to achieve relative positions and velocity consensus. It is solved by applying Pontryagin's principle. Subsequently, a collision-constrained regulating problem is addressed to ensure collision-free tracking of the planned formation trajectory. The tracking problem incorporates a directionally aware collision avoidance strategy that prioritizes avoiding UAVs in the forward path and relative approach. It assigns lower priority to those on the sides with an oblique relative approach, disregarding UAVs behind and not in the relative approach. The high-fidelity simulation results validate the effectiveness of the proposed control scheme.
>
---
#### [replaced 002] ParticleFormer: A 3D Point Cloud World Model for Multi-Object, Multi-Material Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.23126v2](http://arxiv.org/pdf/2506.23126v2)**

> **作者:** Suning Huang; Qianzhong Chen; Xiaohan Zhang; Jiankai Sun; Mac Schwager
>
> **摘要:** 3D world models (i.e., learning-based 3D dynamics models) offer a promising approach to generalizable robotic manipulation by capturing the underlying physics of environment evolution conditioned on robot actions. However, existing 3D world models are primarily limited to single-material dynamics using a particle-based Graph Neural Network model, and often require time-consuming 3D scene reconstruction to obtain 3D particle tracks for training. In this work, we present ParticleFormer, a Transformer-based point cloud world model trained with a hybrid point cloud reconstruction loss, supervising both global and local dynamics features in multi-material, multi-object robot interactions. ParticleFormer captures fine-grained multi-object interactions between rigid, deformable, and flexible materials, trained directly from real-world robot perception data without an elaborate scene reconstruction. We demonstrate the model's effectiveness both in 3D scene forecasting tasks, and in downstream manipulation tasks using a Model Predictive Control (MPC) policy. In addition, we extend existing dynamics learning benchmarks to include diverse multi-material, multi-object interaction scenarios. We validate our method on six simulation and three real-world experiments, where it consistently outperforms leading baselines by achieving superior dynamics prediction accuracy and less rollout error in downstream visuomotor tasks. Experimental videos are available at https://particleformer.github.io/.
>
---
#### [replaced 003] Building Rome with Convex Optimization
- **分类: cs.RO; cs.CV; math.OC**

- **链接: [http://arxiv.org/pdf/2502.04640v4](http://arxiv.org/pdf/2502.04640v4)**

> **作者:** Haoyu Han; Heng Yang
>
> **摘要:** Global bundle adjustment is made easy by depth prediction and convex optimization. We (i) propose a scaled bundle adjustment (SBA) formulation that lifts 2D keypoint measurements to 3D with learned depth, (ii) design an empirically tight convex semidfinite program (SDP) relaxation that solves SBA to certfiable global optimality, (iii) solve the SDP relaxations at extreme scale with Burer-Monteiro factorization and a CUDA-based trust-region Riemannian optimizer (dubbed XM), (iv) build a structure from motion (SfM) pipeline with XM as the optimization engine and show that XM-SfM compares favorably with existing pipelines in terms of reconstruction quality while being significantly faster, more scalable, and initialization-free.
>
---
#### [replaced 004] CoCMT: Communication-Efficient Cross-Modal Transformer for Collaborative Perception
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.13504v2](http://arxiv.org/pdf/2503.13504v2)**

> **作者:** Rujia Wang; Xiangbo Gao; Hao Xiang; Runsheng Xu; Zhengzhong Tu
>
> **摘要:** Multi-agent collaborative perception enhances each agent perceptual capabilities by sharing sensing information to cooperatively perform robot perception tasks. This approach has proven effective in addressing challenges such as sensor deficiencies, occlusions, and long-range perception. However, existing representative collaborative perception systems transmit intermediate feature maps, such as bird-eye view (BEV) representations, which contain a significant amount of non-critical information, leading to high communication bandwidth requirements. To enhance communication efficiency while preserving perception capability, we introduce CoCMT, an object-query-based collaboration framework that optimizes communication bandwidth by selectively extracting and transmitting essential features. Within CoCMT, we introduce the Efficient Query Transformer (EQFormer) to effectively fuse multi-agent object queries and implement a synergistic deep supervision to enhance the positive reinforcement between stages, leading to improved overall performance. Experiments on OPV2V and V2V4Real datasets show CoCMT outperforms state-of-the-art methods while drastically reducing communication needs. On V2V4Real, our model (Top-50 object queries) requires only 0.416 Mb bandwidth, 83 times less than SOTA methods, while improving AP70 by 1.1 percent. This efficiency breakthrough enables practical collaborative perception deployment in bandwidth-constrained environments without sacrificing detection accuracy.
>
---
#### [replaced 005] AirV2X: Unified Air-Ground Vehicle-to-Everything Collaboration
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19283v2](http://arxiv.org/pdf/2506.19283v2)**

> **作者:** Xiangbo Gao; Yuheng Wu; Xuewen Luo; Keshu Wu; Xinghao Chen; Yuping Wang; Chenxi Liu; Yang Zhou; Zhengzhong Tu
>
> **摘要:** While multi-vehicular collaborative driving demonstrates clear advantages over single-vehicle autonomy, traditional infrastructure-based V2X systems remain constrained by substantial deployment costs and the creation of "uncovered danger zones" in rural and suburban areas. We present AirV2X-Perception, a large-scale dataset that leverages Unmanned Aerial Vehicles (UAVs) as a flexible alternative or complement to fixed Road-Side Units (RSUs). Drones offer unique advantages over ground-based perception: complementary bird's-eye-views that reduce occlusions, dynamic positioning capabilities that enable hovering, patrolling, and escorting navigation rules, and significantly lower deployment costs compared to fixed infrastructure. Our dataset comprises 6.73 hours of drone-assisted driving scenarios across urban, suburban, and rural environments with varied weather and lighting conditions. The AirV2X-Perception dataset facilitates the development and standardized evaluation of Vehicle-to-Drone (V2D) algorithms, addressing a critical gap in the rapidly expanding field of aerial-assisted autonomous driving systems. The dataset and development kits are open-sourced at https://github.com/taco-group/AirV2X-Perception.
>
---
#### [replaced 006] Generating and Customizing Robotic Arm Trajectories using Neural Networks
- **分类: cs.RO; cs.AI; 68T40, 93C85, 70E60; I.2.9**

- **链接: [http://arxiv.org/pdf/2506.20259v2](http://arxiv.org/pdf/2506.20259v2)**

> **作者:** Andrej Lúčny; Matilde Antonj; Carlo Mazzola; Hana Hornáčková; Igor Farkaš
>
> **备注:** The code is released at https://github.com/andylucny/nico2/tree/main/generate
>
> **摘要:** We introduce a neural network approach for generating and customizing the trajectory of a robotic arm, that guarantees precision and repeatability. To highlight the potential of this novel method, we describe the design and implementation of the technique and show its application in an experimental setting of cognitive robotics. In this scenario, the NICO robot was characterized by the ability to point to specific points in space with precise linear movements, increasing the predictability of the robotic action during its interaction with humans. To achieve this goal, the neural network computes the forward kinematics of the robot arm. By integrating it with a generator of joint angles, another neural network was developed and trained on an artificial dataset created from suitable start and end poses of the robotic arm. Through the computation of angular velocities, the robot was characterized by its ability to perform the movement, and the quality of its action was evaluated in terms of shape and accuracy. Thanks to its broad applicability, our approach successfully generates precise trajectories that could be customized in their shape and adapted to different settings.
>
---
#### [replaced 007] Unified Manipulability and Compliance Analysis of Modular Soft-Rigid Hybrid Fingers
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.13800v2](http://arxiv.org/pdf/2504.13800v2)**

> **作者:** Jianshu Zhou; Boyuan Liang; Junda Huang; Masayoshi Tomizuka
>
> **摘要:** This paper presents a unified framework to analyze the manipulability and compliance of modular soft-rigid hybrid robotic fingers. The approach applies to both hydraulic and pneumatic actuation systems. A Jacobian-based formulation maps actuator inputs to joint and task-space responses. Hydraulic actuators are modeled under incompressible assumptions, while pneumatic actuators are described using nonlinear pressure-volume relations. The framework enables consistent evaluation of manipulability ellipsoids and compliance matrices across actuation modes. We validate the analysis using two representative hands: DexCo (hydraulic) and Edgy-2 (pneumatic). Results highlight actuation-dependent trade-offs in dexterity and passive stiffness. These findings provide insights for structure-aware design and actuator selection in soft-rigid robotic fingers.
>
---
#### [replaced 008] Learning Attentive Neural Processes for Planning with Pushing Actions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.17924v3](http://arxiv.org/pdf/2504.17924v3)**

> **作者:** Atharv Jain; Seiji Shaw; Nicholas Roy
>
> **备注:** Presented at 2025 RoboReps Workshop
>
> **摘要:** Our goal is to enable robots to plan sequences of tabletop actions to push a block with unknown physical properties to a desired goal pose. We approach this problem by learning the constituent models of a Partially-Observable Markov Decision Process (POMDP), where the robot can observe the outcome of a push, but the physical properties of the block that govern the dynamics remain unknown. A common solution approach is to train an observation model in a supervised fashion, and do inference with a general inference technique such as particle filters. However, supervised training requires knowledge of the relevant physical properties that determine the problem dynamics, which we do not assume to be known. Planning also requires simulating many belief updates, which becomes expensive when using particle filters to represent the belief. We propose to learn an Attentive Neural Process that computes the belief over a learned latent representation of the relevant physical properties given a history of actions. To address the pushing planning problem, we integrate a trained Neural Process with a double-progressive widening sampling strategy. Simulation results indicate that Neural Process Tree with Double Progressive Widening (NPT-DPW) generates better-performing plans faster than traditional particle-filter methods that use a supervised-trained observation model, even in complex pushing scenarios.
>
---
#### [replaced 009] Da Yu: Towards USV-Based Image Captioning for Waterway Surveillance and Scene Understanding
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19288v2](http://arxiv.org/pdf/2506.19288v2)**

> **作者:** Runwei Guan; Ningwei Ouyang; Tianhao Xu; Shaofeng Liang; Wei Dai; Yafeng Sun; Shang Gao; Songning Lai; Shanliang Yao; Xuming Hu; Ryan Wen Liu; Yutao Yue; Hui Xiong
>
> **备注:** 14 pages, 13 figures
>
> **摘要:** Automated waterway environment perception is crucial for enabling unmanned surface vessels (USVs) to understand their surroundings and make informed decisions. Most existing waterway perception models primarily focus on instance-level object perception paradigms (e.g., detection, segmentation). However, due to the complexity of waterway environments, current perception datasets and models fail to achieve global semantic understanding of waterways, limiting large-scale monitoring and structured log generation. With the advancement of vision-language models (VLMs), we leverage image captioning to introduce WaterCaption, the first captioning dataset specifically designed for waterway environments. WaterCaption focuses on fine-grained, multi-region long-text descriptions, providing a new research direction for visual geo-understanding and spatial scene cognition. Exactly, it includes 20.2k image-text pair data with 1.8 million vocabulary size. Additionally, we propose Da Yu, an edge-deployable multi-modal large language model for USVs, where we propose a novel vision-to-language projector called Nano Transformer Adaptor (NTA). NTA effectively balances computational efficiency with the capacity for both global and fine-grained local modeling of visual features, thereby significantly enhancing the model's ability to generate long-form textual outputs. Da Yu achieves an optimal balance between performance and efficiency, surpassing state-of-the-art models on WaterCaption and several other captioning benchmarks.
>
---
#### [replaced 010] Diffuse-CLoC: Guided Diffusion for Physics-based Character Look-ahead Control
- **分类: cs.GR; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.11801v2](http://arxiv.org/pdf/2503.11801v2)**

> **作者:** Xiaoyu Huang; Takara Truong; Yunbo Zhang; Fangzhou Yu; Jean Pierre Sleiman; Jessica Hodgins; Koushil Sreenath; Farbod Farshidian
>
> **摘要:** We present Diffuse-CLoC, a guided diffusion framework for physics-based look-ahead control that enables intuitive, steerable, and physically realistic motion generation. While existing kinematics motion generation with diffusion models offer intuitive steering capabilities with inference-time conditioning, they often fail to produce physically viable motions. In contrast, recent diffusion-based control policies have shown promise in generating physically realizable motion sequences, but the lack of kinematics prediction limits their steerability. Diffuse-CLoC addresses these challenges through a key insight: modeling the joint distribution of states and actions within a single diffusion model makes action generation steerable by conditioning it on the predicted states. This approach allows us to leverage established conditioning techniques from kinematic motion generation while producing physically realistic motions. As a result, we achieve planning capabilities without the need for a high-level planner. Our method handles a diverse set of unseen long-horizon downstream tasks through a single pre-trained model, including static and dynamic obstacle avoidance, motion in-betweening, and task-space control. Experimental results show that our method significantly outperforms the traditional hierarchical framework of high-level motion diffusion and low-level tracking.
>
---
#### [replaced 011] Adapt Your Body: Mitigating Proprioception Shifts in Imitation Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.23944v2](http://arxiv.org/pdf/2506.23944v2)**

> **作者:** Fuhang Kuang; Jiacheng You; Yingdong Hu; Tong Zhang; Chuan Wen; Yang Gao
>
> **备注:** Need further modification
>
> **摘要:** Imitation learning models for robotic tasks typically rely on multi-modal inputs, such as RGB images, language, and proprioceptive states. While proprioception is intuitively important for decision-making and obstacle avoidance, simply incorporating all proprioceptive states leads to a surprising degradation in imitation learning performance. In this work, we identify the underlying issue as the proprioception shift problem, where the distributions of proprioceptive states diverge significantly between training and deployment. To address this challenge, we propose a domain adaptation framework that bridges the gap by utilizing rollout data collected during deployment. Using Wasserstein distance, we quantify the discrepancy between expert and rollout proprioceptive states and minimize this gap by adding noise to both sets of states, proportional to the Wasserstein distance. This strategy enhances robustness against proprioception shifts by aligning the training and deployment distributions. Experiments on robotic manipulation tasks demonstrate the efficacy of our method, enabling the imitation policy to leverage proprioception while mitigating its adverse effects. Our approach outperforms the naive solution which discards proprioception, and other baselines designed to address distributional shifts.
>
---
#### [replaced 012] DG16M: A Large-Scale Dataset for Dual-Arm Grasping with Force-Optimized Grasps
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.08358v2](http://arxiv.org/pdf/2503.08358v2)**

> **作者:** Md Faizal Karim; Mohammed Saad Hashmi; Shreya Bollimuntha; Mahesh Reddy Tapeti; Gaurav Singh; Nagamanikandan Govindan; K Madhava Krishna
>
> **摘要:** Dual-arm robotic grasping is crucial for handling large objects that require stable and coordinated manipulation. While single-arm grasping has been extensively studied, datasets tailored for dual-arm settings remain scarce. We introduce a large-scale dataset of 16 million dual-arm grasps, evaluated under improved force-closure constraints. Additionally, we develop a benchmark dataset containing 300 objects with approximately 30,000 grasps, evaluated in a physics simulation environment, providing a better grasp quality assessment for dual-arm grasp synthesis methods. Finally, we demonstrate the effectiveness of our dataset by training a Dual-Arm Grasp Classifier network that outperforms the state-of-the-art methods by 15\%, achieving higher grasp success rates and improved generalization across objects.
>
---
#### [replaced 013] DexH2R: A Benchmark for Dynamic Dexterous Grasping in Human-to-Robot Handover
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.23152v2](http://arxiv.org/pdf/2506.23152v2)**

> **作者:** Youzhuo Wang; Jiayi Ye; Chuyang Xiao; Yiming Zhong; Heng Tao; Hang Yu; Yumeng Liu; Jingyi Yu; Yuexin Ma
>
> **摘要:** Handover between a human and a dexterous robotic hand is a fundamental yet challenging task in human-robot collaboration. It requires handling dynamic environments and a wide variety of objects and demands robust and adaptive grasping strategies. However, progress in developing effective dynamic dexterous grasping methods is limited by the absence of high-quality, real-world human-to-robot handover datasets. Existing datasets primarily focus on grasping static objects or rely on synthesized handover motions, which differ significantly from real-world robot motion patterns, creating a substantial gap in applicability. In this paper, we introduce DexH2R, a comprehensive real-world dataset for human-to-robot handovers, built on a dexterous robotic hand. Our dataset captures a diverse range of interactive objects, dynamic motion patterns, rich visual sensor data, and detailed annotations. Additionally, to ensure natural and human-like dexterous motions, we utilize teleoperation for data collection, enabling the robot's movements to align with human behaviors and habits, which is a crucial characteristic for intelligent humanoid robots. Furthermore, we propose an effective solution, DynamicGrasp, for human-to-robot handover and evaluate various state-of-the-art approaches, including auto-regressive models and diffusion policy methods, providing a thorough comparison and analysis. We believe our benchmark will drive advancements in human-to-robot handover research by offering a high-quality dataset, effective solutions, and comprehensive evaluation metrics.
>
---
#### [replaced 014] Autonomous Robotic Bone Micro-Milling System with Automatic Calibration and 3D Surface Fitting
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.04038v2](http://arxiv.org/pdf/2503.04038v2)**

> **作者:** Enduo Zhao; Xiaofeng Lin; Yifan Wang; Kanako Harada
>
> **备注:** 8 pages, 8 figures, submitted to RA-L
>
> **摘要:** Automating bone micro-milling using a robotic system presents challenges due to the uncertainties in both the external and internal features of bone tissue. For example, during mouse cranial window creation, a circular path with a radius of 2 to 4 mm needs to be milled on the mouse skull using a microdrill. The uneven surface and non-uniform thickness of the mouse skull make it difficult to fully automate this process, requiring the system to possess advanced perceptual and adaptive capabilities. In this study, we address this challenge by integrating a Microscopic Stereo Camera System (MSCS) into the robotic bone micro-milling system and proposing a novel pre-measurement pipeline for the target surface. Starting from uncalibrated cameras, the pipeline enables automatic calibration and 3D surface fitting through a convolutional neural network (CNN)-based keypoint detection. Combined with the existing feedback-based system, we develop the world's first autonomous robotic bone micro-milling system capable of rapidly, in real-time, and accurately perceiving and adapting to surface unevenness and non-uniform thickness, thereby enabling an end-to-end autonomous cranial window creation workflow without human assistance. Validation experiments on euthanized mice demonstrate that the improved system achieves a success rate of 85.7% and an average milling time of 2.1 minutes, showing not only significant performance improvements over the previous system but also exceptional accuracy, speed, and stability compared to human operators.
>
---
