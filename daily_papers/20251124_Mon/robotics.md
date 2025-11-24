# 机器人 cs.RO

- **最新发布 32 篇**

- **更新 11 篇**

## 最新发布

#### [new 001] Agility Meets Stability: Versatile Humanoid Control with Heterogeneous Data
- **分类: cs.RO**

- **简介: 该论文针对人形机器人在动态任务与平衡稳定性间的矛盾，提出AMS框架。通过融合人类动作捕捉数据与物理约束的合成平衡数据，设计混合奖励机制与自适应学习策略，实现单一策略下敏捷运动与极端平衡的统一控制，显著提升机器人多功能性。**

- **链接: [https://arxiv.org/pdf/2511.17373v1](https://arxiv.org/pdf/2511.17373v1)**

> **作者:** Yixuan Pan; Ruoyi Qiao; Li Chen; Kashyap Chitta; Liang Pan; Haoguang Mai; Qingwen Bu; Hao Zhao; Cunyuan Zheng; Ping Luo; Hongyang Li
>
> **摘要:** Humanoid robots are envisioned to perform a wide range of tasks in human-centered environments, requiring controllers that combine agility with robust balance. Recent advances in locomotion and whole-body tracking have enabled impressive progress in either agile dynamic skills or stability-critical behaviors, but existing methods remain specialized, focusing on one capability while compromising the other. In this work, we introduce AMS (Agility Meets Stability), the first framework that unifies both dynamic motion tracking and extreme balance maintenance in a single policy. Our key insight is to leverage heterogeneous data sources: human motion capture datasets that provide rich, agile behaviors, and physically constrained synthetic balance motions that capture stability configurations. To reconcile the divergent optimization goals of agility and stability, we design a hybrid reward scheme that applies general tracking objectives across all data while injecting balance-specific priors only into synthetic motions. Further, an adaptive learning strategy with performance-driven sampling and motion-specific reward shaping enables efficient training across diverse motion distributions. We validate AMS extensively in simulation and on a real Unitree G1 humanoid. Experiments demonstrate that a single policy can execute agile skills such as dancing and running, while also performing zero-shot extreme balance motions like Ip Man's Squat, highlighting AMS as a versatile control paradigm for future humanoid applications.
>
---
#### [new 002] Single-Pixel Tactile Skin via Compressive Sampling
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出单像素触觉皮肤（SPTS），解决电子皮肤大面积极限的布线复杂与数据瓶颈问题。通过压缩采样在硬件级实现分布式传感，仅用单一输出通道重建全阵列触觉信息，支持快速定位与渐进高保真重构，显著提升响应速度与系统可扩展性，适用于机器人与人机交互。**

- **链接: [https://arxiv.org/pdf/2511.16898v1](https://arxiv.org/pdf/2511.16898v1)**

> **作者:** Ariel Slepyan; Laura Xing; Rudy Zhang; Nitish Thakor
>
> **备注:** 24 pages, 6 main figures, 6 supplemental figures
>
> **摘要:** Development of large-area, high-speed electronic skins is a grand challenge for robotics, prosthetics, and human-machine interfaces, but is fundamentally limited by wiring complexity and data bottlenecks. Here, we introduce Single-Pixel Tactile Skin (SPTS), a paradigm that uses compressive sampling to reconstruct rich tactile information from an entire sensor array via a single output channel. This is achieved through a direct circuit-level implementation where each sensing element, equipped with a miniature microcontroller, contributes a dynamically weighted analog signal to a global sum, performing distributed compressed sensing in hardware. Our flexible, daisy-chainable design simplifies wiring to a few input lines and one output, and significantly reduces measurement requirements compared to raster scanning methods. We demonstrate the system's performance by achieving object classification at an effective 3500 FPS and by capturing transient dynamics, resolving an 8 ms projectile impact into 23 frames. A key feature is the support for adaptive reconstruction, where sensing fidelity scales with measurement time. This allows for rapid contact localization using as little as 7% of total data, followed by progressive refinement to a high-fidelity image - a capability critical for responsive robotic systems. This work offers an efficient pathway towards large-scale tactile intelligence for robotics and human-machine interfaces.
>
---
#### [new 003] HALO: High-Altitude Language-Conditioned Monocular Aerial Exploration and Navigation
- **分类: cs.RO**

- **简介: 该论文提出HALO系统，解决高空单目视觉下实时稠密3D语义建图与导航问题。通过融合视觉、GPS与IMU，实现大尺度室外环境的精准建图与自然语言驱动的任务探索，可在真实无人机上实现实时运行，显著提升探索效率。**

- **链接: [https://arxiv.org/pdf/2511.17497v1](https://arxiv.org/pdf/2511.17497v1)**

> **作者:** Yuezhan Tao; Dexter Ong; Fernando Cladera; Jason Hughes; Camillo J. Taylor; Pratik Chaudhari; Vijay Kumar
>
> **摘要:** We demonstrate real-time high-altitude aerial metric-semantic mapping and exploration using a monocular camera paired with a global positioning system (GPS) and an inertial measurement unit (IMU). Our system, named HALO, addresses two key challenges: (i) real-time dense 3D reconstruction using vision at large distances, and (ii) mapping and exploration of large-scale outdoor environments with accurate scene geometry and semantics. We demonstrate that HALO can plan informative paths that exploit this information to complete missions with multiple tasks specified in natural language. In simulation-based evaluation across large-scale environments of size up to 78,000 sq. m., HALO consistently completes tasks with less exploration time and achieves up to 68% higher competitive ratio in terms of the distance traveled compared to the state-of-the-art semantic exploration baseline. We use real-world experiments on a custom quadrotor platform to demonstrate that (i) all modules can run onboard the robot, and that (ii) in diverse environments HALO can support effective autonomous execution of missions covering up to 24,600 sq. m. area at an altitude of 40 m. Experiment videos and more details can be found on our project page: https://tyuezhan.github.io/halo/.
>
---
#### [new 004] Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文研究视觉语言导航任务，针对现有模型忽视观察与指令序列单调共进关系的问题，提出Progress-Think框架。通过三阶段训练，实现从视觉历史中推理语义进度，提升导航一致性与效率，显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.17097v1](https://arxiv.org/pdf/2511.17097v1)**

> **作者:** Shuo Wang; Yucheng Wang; Guoxin Lian; Yongcai Wang; Maiyue Chen; Kaihui Wang; Bo Zhang; Zhizhong Su; Yutian Zhou; Wanting Li; Deying Li; Zhaoxin Fan
>
> **摘要:** Vision-Language Navigation requires agents to act coherently over long horizons by understanding not only local visual context but also how far they have advanced within a multi-step instruction. However, recent Vision-Language-Action models focus on direct action prediction and earlier progress methods predict numeric achievements; both overlook the monotonic co-progression property of the observation and instruction sequences. Building on this insight, Progress-Think introduces semantic progress reasoning, predicting instruction-style progress from visual observations to enable more accurate navigation. To achieve this without expensive annotations, we propose a three-stage framework. In the initial stage, Self-Aligned Progress Pretraining bootstraps a reasoning module via a novel differentiable alignment between visual history and instruction prefixes. Then, Progress-Guided Policy Pretraining injects learned progress states into the navigation context, guiding the policy toward consistent actions. Finally, Progress-Policy Co-Finetuning jointly optimizes both modules with tailored progress-aware reinforcement objectives. Experiments on R2R-CE and RxR-CE show state-of-the-art success and efficiency, demonstrating that semantic progress yields a more consistent representation of navigation advancement.
>
---
#### [new 005] RoboCOIN: An Open-Sourced Bimanual Robotic Data COllection for INtegrated Manipulation
- **分类: cs.RO**

- **简介: 该论文针对双臂机器人操作数据稀缺问题，提出RoboCOIN开源数据集，涵盖15种机器人平台的18万条示范。通过分层能力金字塔与RTML框架实现多层级标注与统一管理，支持跨平台双臂协作学习，显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.17441v1](https://arxiv.org/pdf/2511.17441v1)**

> **作者:** Shihan Wu; Xuecheng Liu; Shaoxuan Xie; Pengwei Wang; Xinghang Li; Bowen Yang; Zhe Li; Kai Zhu; Hongyu Wu; Yiheng Liu; Zhaoye Long; Yue Wang; Chong Liu; Dihan Wang; Ziqiang Ni; Xiang Yang; You Liu; Ruoxuan Feng; Runtian Xu; Lei Zhang; Denghang Huang; Chenghao Jin; Anlan Yin; Xinlong Wang; Zhenguo Sun; Junkai Zhao; Mengfei Du; Mingyu Cao; Xiansheng Chen; Hongyang Cheng; Xiaojie Zhang; Yankai Fu; Ning Chen; Cheng Chi; Sixiang Chen; Huaihai Lyu; Xiaoshuai Hao; Yankai Fu; Yequan Wang; Bo Lei; Dong Liu; Xi Yang; Yance Jiao; Tengfei Pan; Yunyan Zhang; Songjing Wang; Ziqian Zhang; Xu Liu; Ji Zhang; Caowei Meng; Zhizheng Zhang; Jiyang Gao; Song Wang; Xiaokun Leng; Zhiqiang Xie; Zhenzhen Zhou; Peng Huang; Wu Yang; Yandong Guo; Yichao Zhu; Suibing Zheng; Hao Cheng; Xinmin Ding; Yang Yue; Huanqian Wang; Chi Chen; Jingrui Pang; YuXi Qian; Haoran Geng; Lianli Gao; Haiyuan Li; Bin Fang; Gao Huang; Yaodong Yang; Hao Dong; He Wang; Hang Zhao; Yadong Mu; Di Hu; Hao Zhao; Tiejun Huang; Shanghang Zhang; Yonghua Lin; Zhongyuan Wang; Guocai Yao
>
> **摘要:** Bimanual manipulation is essential for achieving human-like dexterity in robots, but the large-scale and diverse bimanual robot datasets remain scarce due to hardware heterogeneity across robotic platforms. To address the challenge, we present RoboCOIN, a comprehensive multi-embodiment bimanual manipulation dataset with over 180,000 demonstrations collected from 15 distinct robotic platforms. The dataset covers 16 scenarios, including residential, commercial, and working environments, with 421 tasks systematically organized by bimanual coordination patterns and object properties. Our key innovation is a hierarchical capability pyramid that provides multi-level annotations, spanning trajectory-level concepts, segment-level subtasks, and frame-level kinematics. We further develop CoRobot, a comprehensive processing framework featuring Robot Trajectory Markup Language (RTML) for quality assessment, automated annotation generation, and unified multi-embodiment management. Extensive experiments demonstrate the reliability and effectiveness of RoboCOIN in multi-embodiment bimanual learning, with significant performance improvements across various model architectures and robotic platforms. The complete dataset and framework are open-sourced and publicly available for further research purposes. Project website: https://FlagOpen.github.io/RoboCOIN/.
>
---
#### [new 006] METIS: Multi-Source Egocentric Training for Integrated Dexterous Vision-Language-Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对灵巧操作中缺乏大规模动作标注数据的问题，提出METIS模型。通过整合多源第一人称数据构建EgoAtlas，并引入运动感知动态表示，实现视觉-语言-动作统一建模，显著提升灵巧操作的泛化与鲁棒性，在六项真实任务中达到最高成功率。**

- **链接: [https://arxiv.org/pdf/2511.17366v1](https://arxiv.org/pdf/2511.17366v1)**

> **作者:** Yankai Fu; Ning Chen; Junkai Zhao; Shaozhe Shan; Guocai Yao; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang
>
> **摘要:** Building a generalist robot that can perceive, reason, and act across diverse tasks remains an open challenge, especially for dexterous manipulation. A major bottleneck lies in the scarcity of large-scale, action-annotated data for dexterous skills, as teleoperation is difficult and costly. Human data, with its vast scale and diverse manipulation behaviors, provides rich priors for learning robotic actions. While prior works have explored leveraging human demonstrations, they are often constrained by limited scenarios and a large visual gap between human and robots. To eliminate these limitations, we propose METIS, a vision-language-action (VLA) model for dexterous manipulation pretrained on multi-source egocentric datasets. We first construct EgoAtlas, which integrates large-scale human and robotic data from multiple sources, all unified under a consistent action space. We further extract motion-aware dynamics, a compact and discretized motion representation, which provides efficient and expressive supervision for VLA training. Built upon them, METIS integrates reasoning and acting into a unified framework, enabling effective deployment to downstream dexterous manipulation tasks. Our method demonstrates exceptional dexterous manipulation capabilities, achieving highest average success rate in six real-world tasks. Experimental results also highlight the superior generalization and robustness to out-of-distribution scenarios. These findings emphasize METIS as a promising step toward a generalist model for dexterous manipulation.
>
---
#### [new 007] SPEAR-1: Scaling Beyond Robot Demonstrations via 3D Understanding
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出SPEAR-1，一种基于3D理解的机器人基础模型。针对现有模型在3D空间推理能力不足的问题，通过增强非机器人图像数据的3D标注，训练出具备3D感知能力的VLM，并将其用于机器人控制。实验表明，SPEAR-1在少量机器人演示下优于或媲美先进模型，提升了泛化能力与控制可靠性。**

- **链接: [https://arxiv.org/pdf/2511.17411v1](https://arxiv.org/pdf/2511.17411v1)**

> **作者:** Nikolay Nikolov; Giuliano Albanese; Sombit Dey; Aleksandar Yanev; Luc Van Gool; Jan-Nico Zaech; Danda Pani Paudel
>
> **摘要:** Robotic Foundation Models (RFMs) hold great promise as generalist, end-to-end systems for robot control. Yet their ability to generalize across new environments, tasks, and embodiments remains limited. We argue that a major bottleneck lies in their foundations: most RFMs are built by fine-tuning internet-pretrained Vision-Language Models (VLMs). However, these VLMs are trained on 2D image-language tasks and lack the 3D spatial reasoning inherently required for embodied control in the 3D world. Bridging this gap directly with large-scale robotic data is costly and difficult to scale. Instead, we propose to enrich easy-to-collect non-robotic image data with 3D annotations and enhance a pretrained VLM with 3D understanding capabilities. Following this strategy, we train SPEAR-VLM, a 3D-aware VLM that infers object coordinates in 3D space from a single 2D image. Building on SPEAR-VLM, we introduce our main contribution, $~\textbf{SPEAR-1}$: a robotic foundation model that integrates grounded 3D perception with language-instructed embodied control. Trained on $\sim$45M frames from 24 Open X-Embodiment datasets, SPEAR-1 outperforms or matches state-of-the-art models such as $π_0$-FAST and $π_{0.5}$, while it uses 20$\times$ fewer robot demonstrations. This carefully-engineered training strategy unlocks new VLM capabilities and as a consequence boosts the reliability of embodied control beyond what is achievable with only robotic data. We make our model weights and 3D-annotated datasets publicly available.
>
---
#### [new 008] H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文针对机器人操作中观测与动作预测不一致的问题，提出H-GAR框架。通过目标驱动的分层交互机制，先生成目标观测与粗略动作，再通过协同模块实现观测与动作的精细化迭代优化，提升预测一致性与操作准确性。**

- **链接: [https://arxiv.org/pdf/2511.17079v1](https://arxiv.org/pdf/2511.17079v1)**

> **作者:** Yijie Zhu; Rui Shao; Ziyang Liu; Jie He; Jizhihui Liu; Jiuru Wang; Zitong Yu
>
> **备注:** Accepted to AAAI 2026 (Oral), Project Page: https://github.com/JiuTian-VL/H-GAR
>
> **摘要:** Unified video and action prediction models hold great potential for robotic manipulation, as future observations offer contextual cues for planning, while actions reveal how interactions shape the environment. However, most existing approaches treat observation and action generation in a monolithic and goal-agnostic manner, often leading to semantically misaligned predictions and incoherent behaviors. To this end, we propose H-GAR, a Hierarchical interaction framework via Goal-driven observation-Action Refinement.To anchor prediction to the task objective, H-GAR first produces a goal observation and a coarse action sketch that outline a high-level route toward the goal. To enable explicit interaction between observation and action under the guidance of the goal observation for more coherent decision-making, we devise two synergistic modules. (1) Goal-Conditioned Observation Synthesizer (GOS) synthesizes intermediate observations based on the coarse-grained actions and the predicted goal observation. (2) Interaction-Aware Action Refiner (IAAR) refines coarse actions into fine-grained, goal-consistent actions by leveraging feedback from the intermediate observations and a Historical Action Memory Bank that encodes prior actions to ensure temporal consistency. By integrating goal grounding with explicit action-observation interaction in a coarse-to-fine manner, H-GAR enables more accurate manipulation. Extensive experiments on both simulation and real-world robotic manipulation tasks demonstrate that H-GAR achieves state-of-the-art performance.
>
---
#### [new 009] Vector Cost Behavioral Planning for Autonomous Robotic Systems with Contemporary Validation Strategies
- **分类: cs.RO; cs.GT**

- **简介: 该论文针对自主机器人多目标行为规划问题，提出基于向量代价双矩阵博弈的规划方法，克服传统加权求和法的局限性。通过引入可解释AI与参数空间探索技术，实现性能分析与敏感性研究，构建了集成化仿真验证流程，显著提升规划效果的可解释性与通用性。**

- **链接: [https://arxiv.org/pdf/2511.17375v1](https://arxiv.org/pdf/2511.17375v1)**

> **作者:** Benjamin R. Toaz; Quentin Goss; John Thompson; Seta Boğosyan; Shaunak D. Bopardikar; Mustafa İlhan Akbaş; Metin Gökaşan
>
> **备注:** Technical report associated with submission to Journal of Intelligent & Robotic Systems, currently under review
>
> **摘要:** The vector cost bimatrix game is a method for multi-objective decision making that enables autonomous robotic systems to optimize for multiple goals at once while avoiding worst-case scenarios in neglected objectives. We expand this approach to arbitrary numbers of objectives and compare its performance to scalar weighted sum methods during competitive motion planning. Explainable Artificial Intelligence (XAI) software is used to aid in the analysis of high dimensional decision-making data. State-space Exploration of Multidimensional Boundaries using Adherence Strategies (SEMBAS) is applied to explore performance modes in the parameter space as a sensitivity study for the baseline and proposed frameworks. While some works have explored aspects of game theoretic planning and intelligent systems validation separately, we combine each of these into a novel and comprehensive simulation pipeline. This integration demonstrates a dramatic improvement of the vector cost method over scalarization and offers an interpretable and generalizable framework for robotic behavioral planning. Code available at https://github.com/toazbenj/race_simulation. The video companion to this work is available at https://tinyurl.com/vectorcostvideo.
>
---
#### [new 010] A ROS2 Interface for Universal Robots Collaborative Manipulators Based on ur_rtde
- **分类: cs.RO; cs.SE**

- **简介: 该论文提出一种基于ur_rtde的ROS2驱动，用于Universal Robots协作机械臂。旨在解决机械臂与ROS2系统集成的灵活性问题，通过高阶指令和插件机制支持多种应用。实现了基于路径点的运动控制，并开源发布。**

- **链接: [https://arxiv.org/pdf/2511.17237v1](https://arxiv.org/pdf/2511.17237v1)**

> **作者:** Alessio Saccuti; Riccardo Monica; Jacopo Aleotti
>
> **摘要:** In this paper a novel ROS2 driver for UR robot manipulators is presented, based on the ur_rtde C++ library. The proposed driver aims to be a flexible solution, adaptable to a wide range of applications. The driver exposes the high-level commands of Universal Robots URScripts, and custom commands can be added using a plugin system. Several commands have been implemented, including motion execution along a waypoint-based path. The driver is published as open source.
>
---
#### [new 011] MDG: Masked Denoising Generation for Multi-Agent Behavior Modeling in Traffic Environments
- **分类: cs.RO; cs.MA**

- **简介: 该论文针对自动驾驶中多智能体行为建模任务，解决现有方法效率低、通用性差的问题。提出掩码去噪生成（MDG）框架，通过连续噪声掩码实现单次前向传播的轨迹生成，统一支持开环预测、闭环模拟、规划与条件生成，显著提升效率与可控性。**

- **链接: [https://arxiv.org/pdf/2511.17496v1](https://arxiv.org/pdf/2511.17496v1)**

> **作者:** Zhiyu Huang; Zewei Zhou; Tianhui Cai; Yun Zhang; Jiaqi Ma
>
> **摘要:** Modeling realistic and interactive multi-agent behavior is critical to autonomous driving and traffic simulation. However, existing diffusion and autoregressive approaches are limited by iterative sampling, sequential decoding, or task-specific designs, which hinder efficiency and reuse. We propose Masked Denoising Generation (MDG), a unified generative framework that reformulates multi-agent behavior modeling as the reconstruction of independently noised spatiotemporal tensors. Instead of relying on diffusion time steps or discrete tokenization, MDG applies continuous, per-agent and per-timestep noise masks that enable localized denoising and controllable trajectory generation in a single or few forward passes. This mask-driven formulation generalizes across open-loop prediction, closed-loop simulation, motion planning, and conditional generation within one model. Trained on large-scale real-world driving datasets, MDG achieves competitive closed-loop performance on the Waymo Sim Agents and nuPlan Planning benchmarks, while providing efficient, consistent, and controllable open-loop multi-agent trajectory generation. These results position MDG as a simple yet versatile paradigm for multi-agent behavior modeling.
>
---
#### [new 012] Efficient Robot Design with Multi-Objective Black-Box Optimization and Large Language Models
- **分类: cs.RO**

- **简介: 该论文属于机器人设计优化任务，旨在解决黑箱优化采样效率低的问题。通过结合大语言模型（LLMs）与黑箱优化，在采样过程中利用LLM提供问题设定和反馈，实现更高效的解空间探索，提升设计效率。**

- **链接: [https://arxiv.org/pdf/2511.17178v1](https://arxiv.org/pdf/2511.17178v1)**

> **作者:** Kento Kawaharazuka; Yoshiki Obinata; Naoaki Kanazawa; Haoyu Jia; Kei Okada
>
> **摘要:** Various methods for robot design optimization have been developed so far. These methods are diverse, ranging from numerical optimization to black-box optimization. While numerical optimization is fast, it is not suitable for cases involving complex structures or discrete values, leading to frequent use of black-box optimization instead. However, black-box optimization suffers from low sampling efficiency and takes considerable sampling iterations to obtain good solutions. In this study, we propose a method to enhance the efficiency of robot body design based on black-box optimization by utilizing large language models (LLMs). In parallel with the sampling process based on black-box optimization, sampling is performed using LLMs, which are provided with problem settings and extensive feedback. We demonstrate that this method enables more efficient exploration of design solutions and discuss its characteristics and limitations.
>
---
#### [new 013] FORWARD: Dataset of a forwarder operating in rough terrain
- **分类: cs.RO; cs.AI; cs.CE; cs.LG; physics.app-ph**

- **简介: 该论文提出FORWARD数据集，记录大型伐木前移机在瑞典粗糙地形中的高分辨率多模态运行数据。旨在支持森林机械自主控制、感知与交通能力研究，解决复杂环境下作业效率、安全与环保问题。工作包括采集传感器数据、标注作业环节、设计实验场景，用于AI算法开发与仿真系统构建。**

- **链接: [https://arxiv.org/pdf/2511.17318v1](https://arxiv.org/pdf/2511.17318v1)**

> **作者:** Mikael Lundbäck; Erik Wallin; Carola Häggström; Mattias Nyström; Andreas Grönlund; Mats Richardson; Petrus Jönsson; William Arnvik; Lucas Hedström; Arvid Fälldin; Martin Servin
>
> **备注:** 25 pages, 22 figures
>
> **摘要:** We present FORWARD, a high-resolution multimodal dataset of a cut-to-length forwarder operating in rough terrain on two harvest sites in the middle part of Sweden. The forwarder is a large Komatsu model equipped with a variety of sensors, including RTK-GNSS, 360-camera, operator vibration sensors, internal CAN-bus signal recording, and multiple IMUs. The data includes event time logs recorded in 5 Hz with e.g., driving speed, fuel consumption, vehicle position with centimeter accuracy, and crane use while the vehicle operates in forest areas laser-scanned with very high-resolution, $\sim$1500 points per square meter. Production log files (StanForD standard) with time-stamped machine events, extensive video material, and terrain data in various formats are included as well. About 18 hours of regular wood extraction work during three days is annotated from 360-video material into individual work elements and included in the dataset. We also include scenario specifications of conducted experiments on forest roads and in terrain. Scenarios include repeatedly driving the same routes with and without steel tracks, different load weight, and different target driving speeds. The dataset is intended for developing models and algorithms for trafficability, perception, and autonomous control of forest machines using artificial intelligence, simulation, and experiments on physical testbeds. In part, we focus on forwarders traversing terrain, avoiding obstacles, and loading or unloading logs, with consideration for efficiency, fuel consumption, safety, and environmental impact. Other benefits of the open dataset include the ability to explore auto-generation and calibration of forestry machine simulators and automation scenario descriptions using the data recorded in the field.
>
---
#### [new 014] Simulation of Active Soft Nets for Capture of Space Debris
- **分类: cs.RO**

- **简介: 该论文针对空间碎片自主捕获任务，提出基于MuJoCo的软体网捕获仿真系统。解决传统方法依赖预抛射、控制策略不足的问题。通过模拟不同柔顺性网体与滑模控制，实现对Envisat碎片的高效捕获，显著提升接触面积与成功率。**

- **链接: [https://arxiv.org/pdf/2511.17266v1](https://arxiv.org/pdf/2511.17266v1)**

> **作者:** Leone Costi; Dario Izzo
>
> **摘要:** In this work, we propose a simulator, based on the open-source physics engine MuJoCo, for the design and control of soft robotic nets for the autonomous removal of space debris. The proposed simulator includes net dynamics, contact between the net and the debris, self-contact of the net, orbital mechanics, and a controller that can actuate thrusters on the four satellites at the corners of the net. It showcases the case of capturing Envisat, a large ESA satellite that remains in orbit as space debris following the end of its mission. This work investigates different mechanical models, which can be used to simulate the net dynamics, simulating various degrees of compliance, and different control strategies to achieve the capture of the debris, depending on the relative position of the net and the target. Unlike previous works on this topic, we do not assume that the net has been previously ballistically thrown toward the target, and we start from a relatively static configuration. The results show that a more compliant net achieves higher performance when attempting the capture of Envisat. Moreover, when paired with a sliding mode controller, soft nets are able to achieve successful capture in 100% of the tested cases, whilst also showcasing a higher effective area at contact and a higher number of contact points between net and Envisat.
>
---
#### [new 015] Robot Confirmation Generation and Action Planning Using Long-context Q-Former Integrated with Multimodal LLM
- **分类: cs.RO; cs.CL; cs.CV; cs.SD; eess.AS**

- **简介: 该论文聚焦人机协作中的动作确认与规划任务，针对现有方法忽略长视频上下文依赖、文本信息抽象过度的问题，提出融合左右上下文的长程Q-former与文本条件化机制，通过VideoLLaMA3提升多模态理解能力，显著改善动作确认与规划准确率。**

- **链接: [https://arxiv.org/pdf/2511.17335v1](https://arxiv.org/pdf/2511.17335v1)**

> **作者:** Chiori Hori; Yoshiki Masuyama; Siddarth Jain; Radu Corcodel; Devesh Jha; Diego Romeres; Jonathan Le Roux
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Human-robot collaboration towards a shared goal requires robots to understand human action and interaction with the surrounding environment. This paper focuses on human-robot interaction (HRI) based on human-robot dialogue that relies on the robot action confirmation and action step generation using multimodal scene understanding. The state-of-the-art approach uses multimodal transformers to generate robot action steps aligned with robot action confirmation from a single clip showing a task composed of multiple micro steps. Although actions towards a long-horizon task depend on each other throughout an entire video, the current approaches mainly focus on clip-level processing and do not leverage long-context information. This paper proposes a long-context Q-former incorporating left and right context dependency in full videos. Furthermore, this paper proposes a text-conditioning approach to feed text embeddings directly into the LLM decoder to mitigate the high abstraction of the information in text by Q-former. Experiments with the YouCook2 corpus show that the accuracy of confirmation generation is a major factor in the performance of action planning. Furthermore, we demonstrate that the long-context Q-former improves the confirmation and action planning by integrating VideoLLaMA3.
>
---
#### [new 016] Leveraging CVAE for Joint Configuration Estimation of Multifingered Grippers from Point Cloud Data
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对多指灵巧手关节配置估计任务，解决仅从点云数据中精确获取关节状态的难题。提出基于条件变分自编码器（CVAE）的方法，直接从点云输入重建关节配置，避免传统逆运动学的复杂计算与后处理，实现高效高精度估计。**

- **链接: [https://arxiv.org/pdf/2511.17276v1](https://arxiv.org/pdf/2511.17276v1)**

> **作者:** Julien Merand; Boris Meden; Mathieu Grossard
>
> **摘要:** This paper presents an efficient approach for determining the joint configuration of a multifingered gripper solely from the point cloud data of its poly-articulated chain, as generated by visual sensors, simulations or even generative neural networks. Well-known inverse kinematics (IK) techniques can provide mathematically exact solutions (when they exist) for joint configuration determination based solely on the fingertip pose, but often require post-hoc decision-making by considering the positions of all intermediate phalanges in the gripper's fingers, or rely on algorithms to numerically approximate solutions for more complex kinematics. In contrast, our method leverages machine learning to implicitly overcome these challenges. This is achieved through a Conditional Variational Auto-Encoder (CVAE), which takes point cloud data of key structural elements as input and reconstructs the corresponding joint configurations. We validate our approach on the MultiDex grasping dataset using the Allegro Hand, operating within 0.05 milliseconds and achieving accuracy comparable to state-of-the-art methods. This highlights the effectiveness of our pipeline for joint configuration estimation within the broader context of AI-driven techniques for grasp planning.
>
---
#### [new 017] MobileOcc: A Human-Aware Semantic Occupancy Dataset for Mobile Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对移动机器人在人流密集环境中的3D语义占据感知难题，提出MobileOcc数据集。通过结合静态物体标注与新型人体网格优化框架，实现从2D图像和LiDAR数据中重建并优化人体几何。构建了占据预测与行人速度预测的基准，验证了方法在不同数据集上的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.16949v1](https://arxiv.org/pdf/2511.16949v1)**

> **作者:** Junseo Kim; Guido Dumont; Xinyu Gao; Gang Chen; Holger Caesar; Javier Alonso-Mora
>
> **摘要:** Dense 3D semantic occupancy perception is critical for mobile robots operating in pedestrian-rich environments, yet it remains underexplored compared to its application in autonomous driving. To address this gap, we present MobileOcc, a semantic occupancy dataset for mobile robots operating in crowded human environments. Our dataset is built using an annotation pipeline that incorporates static object occupancy annotations and a novel mesh optimization framework explicitly designed for human occupancy modeling. It reconstructs deformable human geometry from 2D images and subsequently refines and optimizes it using associated LiDAR point data. Using MobileOcc, we establish benchmarks for two tasks, i) Occupancy prediction and ii) Pedestrian velocity prediction, using different methods including monocular, stereo, and panoptic occupancy, with metrics and baseline implementations for reproducible comparison. Beyond occupancy prediction, we further assess our annotation method on 3D human pose estimation datasets. Results demonstrate that our method exhibits robust performance across different datasets.
>
---
#### [new 018] Multi-UAV Swarm Obstacle Avoidance Based on Potential Field Optimization
- **分类: cs.RO; cs.MA**

- **简介: 该论文针对多无人机编队避障中路径冗余、航向突变及碰撞风险问题，提出一种融合改进MRF-IAPF与优化单机路径的混合算法。通过引入三类交互力与动态辅助目标策略，提升路径效率与航向稳定性，有效避免障碍并快速恢复编队。**

- **链接: [https://arxiv.org/pdf/2511.16911v1](https://arxiv.org/pdf/2511.16911v1)**

> **作者:** Yendo Hu; Yiliang Wu; Weican Chen
>
> **备注:** 12 pages, 13 figures, and 2 tables
>
> **摘要:** In multi UAV scenarios,the traditional Artificial Potential Field (APF) method often leads to redundant flight paths and frequent abrupt heading changes due to unreasonable obstacle avoidance path planning,and is highly prone to inter UAV collisions during the obstacle avoidance process.To address these issues,this study proposes a novel hybrid algorithm that combines the improved Multi-Robot Formation Obstacle Avoidance (MRF IAPF) algorithm with an enhanced APF optimized for single UAV path planning.Its core ideas are as follows:first,integrating three types of interaction forces from MRF IAPF obstacle repulsion force,inter UAV interaction force,and target attraction force;second,incorporating a refined single UAV path optimization mechanism,including collision risk assessment and an auxiliary sub goal strategy.When a UAV faces a high collision threat,temporary waypoints are generated to guide obstacle avoidance,ensuring eventual precise arrival at the actual target.Simulation results demonstrate that compared with traditional APF based formation algorithms,the proposed algorithm achieves significant improvements in path length optimization and heading stability,can effectively avoid obstacles and quickly restore the formation configuration,thus verifying its applicability and effectiveness in static environments with unknown obstacles.
>
---
#### [new 019] Human Imitated Bipedal Locomotion with Frequency Based Gait Generator Network
- **分类: cs.RO**

- **简介: 该论文研究机器人仿人行走任务，旨在解决复杂地形下步态鲁棒性差的问题。通过融合人类运动谱特征的轻量级步态生成网络与PPO强化学习控制器，实现仅在平坦地面训练却能泛化至陡坡和崎岖地形的自然行走，显著降低训练成本。**

- **链接: [https://arxiv.org/pdf/2511.17387v1](https://arxiv.org/pdf/2511.17387v1)**

> **作者:** Yusuf Baran Ates; Omer Morgul
>
> **摘要:** Learning human-like, robust bipedal walking remains difficult due to hybrid dynamics and terrain variability. We propose a lightweight framework that combines a gait generator network learned from human motion with Proximal Policy Optimization (PPO) controller for torque control. Despite being trained only on flat or mildly sloped ground, the learned policies generalize to steeper ramps and rough surfaces. Results suggest that pairing spectral motion priors with Deep Reinforcement Learning (DRL) offers a practical path toward natural and robust bipedal locomotion with modest training cost.
>
---
#### [new 020] TP-MDDN: Task-Preferenced Multi-Demand-Driven Navigation with Autonomous Decision-Making
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出TP-MDDN任务，解决多需求、带偏好长时程导航问题。针对真实场景中需同时处理多个目标与个人偏好的挑战，设计AWMSystem决策系统与MASMap空间记忆，结合双节奏动作生成与自适应纠错机制，显著提升导航准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.17225v1](https://arxiv.org/pdf/2511.17225v1)**

> **作者:** Shanshan Li; Da Huang; Yu He; Yanwei Fu; Yu-Gang Jiang; Xiangyang Xue
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** In daily life, people often move through spaces to find objects that meet their needs, posing a key challenge in embodied AI. Traditional Demand-Driven Navigation (DDN) handles one need at a time but does not reflect the complexity of real-world tasks involving multiple needs and personal choices. To bridge this gap, we introduce Task-Preferenced Multi-Demand-Driven Navigation (TP-MDDN), a new benchmark for long-horizon navigation involving multiple sub-demands with explicit task preferences. To solve TP-MDDN, we propose AWMSystem, an autonomous decision-making system composed of three key modules: BreakLLM (instruction decomposition), LocateLLM (goal selection), and StatusMLLM (task monitoring). For spatial memory, we design MASMap, which combines 3D point cloud accumulation with 2D semantic mapping for accurate and efficient environmental understanding. Our Dual-Tempo action generation framework integrates zero-shot planning with policy-based fine control, and is further supported by an Adaptive Error Corrector that handles failure cases in real time. Experiments demonstrate that our approach outperforms state-of-the-art baselines in both perception accuracy and navigation robustness.
>
---
#### [new 021] A*-based Temporal Logic Path Planning with User Preferences on Relaxed Task Satisfaction
- **分类: cs.RO**

- **简介: 该论文研究机器人在大型环境中的时序逻辑路径规划问题。当无法完全满足任务时，通过引入用户偏好对任务松弛进行优化，提出基于A*的规划框架，结合高效启发式算法，在保证近似最优解的同时显著提升计算效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.16844v1](https://arxiv.org/pdf/2511.16844v1)**

> **作者:** Disha Kamale; Xi Yu; Cristian-Ioan Vasile
>
> **摘要:** In this work, we consider the problem of planning for temporal logic tasks in large robot environments. When full task compliance is unattainable, we aim to achieve the best possible task satisfaction by integrating user preferences for relaxation into the planning process. Utilizing the automata-based representations for temporal logic goals and user preferences, we propose an A*-based planning framework. This approach effectively tackles large-scale problems while generating near-optimal high-level trajectories. To facilitate this, we propose a simple, efficient heuristic that allows for planning over large robot environments in a fraction of time and search memory as compared to uninformed search algorithms. We present extensive case studies to demonstrate the scalability, runtime analysis as well as empirical bounds on the suboptimality of the proposed heuristic.
>
---
#### [new 022] Stable Offline Hand-Eye Calibration for any Robot with Just One Mark
- **分类: cs.RO**

- **简介: 该论文针对机器人手眼标定中相机外参难以获取的问题，提出CalibAll方法，仅需一个末端标记即可实现无需训练、稳定准确的外参估计。通过视觉基础模型匹配标记，结合PnP与渲染优化，提升标定精度与泛化性，适用于多种机器人平台。**

- **链接: [https://arxiv.org/pdf/2511.17001v1](https://arxiv.org/pdf/2511.17001v1)**

> **作者:** Sicheng Xie; Lingchen Meng; Zhiying Du; Shuyuan Tu; Haidong Cao; Jiaqi Leng; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Imitation learning has achieved remarkable success in a variety of robotic tasks by learning a mapping function from camera-space observations to robot-space actions. Recent work indicates that the use of robot-to-camera transformation information ({\ie}, camera extrinsics) benefits the learning process and produces better results. However, camera extrinsics are oftentimes unavailable and estimation methods usually suffer from local minima and poor generalizations. In this paper, we present CalibAll, a simple yet effective method that \textbf{requires only a single mark} and performs training-free, stable, and accurate camera extrinsic estimation across diverse robots and datasets through a coarse-to-fine calibration pipeline. In particular, we annotate a single mark on an end-effector (EEF), and leverage the correspondence ability emerged from vision foundation models (VFM) to automatically localize the corresponding mark across robots in diverse datasets. Using this mark, together with point tracking and the 3D EEF trajectory, we obtain a coarse camera extrinsic via temporal Perspective-n-Point (PnP). This estimate is further refined through a rendering-based optimization that aligns rendered and ground-true masks, yielding accurate and stable camera extrinsic. Experimental results demonstrate that our method outperforms state-of-the-art approaches, showing strong robustness and general effectiveness across three robot platforms. It also produces useful auxiliary annotations such as depth maps, link-wise masks, and end-effector 2D trajectories, which can further support downstream tasks.
>
---
#### [new 023] MonoSpheres: Large-Scale Monocular SLAM-Based UAV Exploration through Perception-Coupled Mapping and Planning
- **分类: cs.RO**

- **简介: 该论文针对仅配备单目相机的无人机在未知环境中进行大尺度3D自主探索的任务，解决稀疏深度、自由空间不确定性等问题。提出MonoSpheres方法，通过感知耦合的建图与规划，在真实复杂环境中实现安全探索，首次实现在真实非结构化室外环境中的单目3D探索。**

- **链接: [https://arxiv.org/pdf/2511.17299v1](https://arxiv.org/pdf/2511.17299v1)**

> **作者:** Tomáš Musil; Matěj Petrlík; Martin Saska
>
> **备注:** 8 pages, 9 figures, submitted to RA-L
>
> **摘要:** Autonomous exploration of unknown environments is a key capability for mobile robots, but it is largely unsolved for robots equipped with only a single monocular camera and no dense range sensors. In this paper, we present a novel approach to monocular vision-based exploration that can safely cover large-scale unstructured indoor and outdoor 3D environments by explicitly accounting for the properties of a sparse monocular SLAM frontend in both mapping and planning. The mapping module solves the problems of sparse depth data, free-space gaps, and large depth uncertainty by oversampling free space in texture-sparse areas and keeping track of obstacle position uncertainty. The planning module handles the added free-space uncertainty through rapid replanning and perception-aware heading control. We further show that frontier-based exploration is possible with sparse monocular depth data when parallax requirements and the possibility of textureless surfaces are taken into account. We evaluate our approach extensively in diverse real-world and simulated environments, including ablation studies. To the best of the authors' knowledge, the proposed method is the first to achieve 3D monocular exploration in real-world unstructured outdoor environments. We open-source our implementation to support future research.
>
---
#### [new 024] IndustryNav: Exploring Spatial Reasoning of Embodied Agents in Dynamic Industrial Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出IndustryNav，首个动态工业导航基准，用于评估视觉大模型在复杂动态环境中的空间推理能力。针对现有基准多聚焦静态家居环境、缺乏动态真实场景的问题，构建12个高保真Unity仓库场景，引入碰撞率与预警率等安全指标，评估模型的路径规划、避障与主动探索能力，揭示当前模型在动态环境中仍存在显著不足。**

- **链接: [https://arxiv.org/pdf/2511.17384v1](https://arxiv.org/pdf/2511.17384v1)**

> **作者:** Yifan Li; Lichi Li; Anh Dao; Xinyu Zhou; Yicheng Qiao; Zheda Mai; Daeun Lee; Zichen Chen; Zhen Tan; Mohit Bansal; Yu Kong
>
> **摘要:** While Visual Large Language Models (VLLMs) show great promise as embodied agents, they continue to face substantial challenges in spatial reasoning. Existing embodied benchmarks largely focus on passive, static household environments and evaluate only isolated capabilities, failing to capture holistic performance in dynamic, real-world complexity. To fill this gap, we present IndustryNav, the first dynamic industrial navigation benchmark for active spatial reasoning. IndustryNav leverages 12 manually created, high-fidelity Unity warehouse scenarios featuring dynamic objects and human movement. Our evaluation employs a PointGoal navigation pipeline that effectively combines egocentric vision with global odometry to assess holistic local-global planning. Crucially, we introduce the "collision rate" and "warning rate" metrics to measure safety-oriented behaviors and distance estimation. A comprehensive study of nine state-of-the-art VLLMs (including models such as GPT-5-mini, Claude-4.5, and Gemini-2.5) reveals that closed-source models maintain a consistent advantage; however, all agents exhibit notable deficiencies in robust path planning, collision avoidance and active exploration. This highlights a critical need for embodied research to move beyond passive perception and toward tasks that demand stable planning, active exploration, and safe behavior in dynamic, real-world environment.
>
---
#### [new 025] RynnVLA-002: A Unified Vision-Language-Action and World Model
- **分类: cs.RO**

- **简介: 该论文提出RynnVLA-002，一个统一的视觉-语言-动作与世界模型。旨在联合学习环境动态与动作规划，解决传统模型分离训练导致的性能瓶颈。通过双向增强：世界模型预测未来图像以优化动作，VLA模型提升视觉理解支持生成，显著提升仿真与真实机器人任务表现。**

- **链接: [https://arxiv.org/pdf/2511.17502v1](https://arxiv.org/pdf/2511.17502v1)**

> **作者:** Jun Cen; Siteng Huang; Yuqian Yuan; Hangjie Yuan; Chaohui Yu; Yuming Jiang; Jiayan Guo; Kehan Li; Hao Luo; Fan Wang; Xin Li; Deli Zhao; Hao Chen
>
> **摘要:** We introduce RynnVLA-002, a unified Vision-Language-Action (VLA) and world model. The world model leverages action and visual inputs to predict future image states, learning the underlying physics of the environment to refine action generation. Conversely, the VLA model produces subsequent actions from image observations, enhancing visual understanding and supporting the world model's image generation. The unified framework of RynnVLA-002 enables joint learning of environmental dynamics and action planning. Our experiments show that RynnVLA-002 surpasses individual VLA and world models, demonstrating their mutual enhancement. We evaluate RynnVLA-002 in both simulation and real-world robot tasks. RynnVLA-002 achieves 97.4% success rate on the LIBERO simulation benchmark without pretraining, while in real-world LeRobot experiments, its integrated world model boosts the overall success rate by 50%.
>
---
#### [new 026] Reflection-Based Relative Localization for Cooperative UAV Teams Using Active Markers
- **分类: cs.RO**

- **简介: 该论文针对多无人机团队在未知环境中进行机载相对定位的问题，提出一种利用主动标记反射的新型方法。无需预先知道无人机尺寸或标记布局，可应对非平面表面（如动态水面）带来的不确定性，显著提升定位范围与精度，适用于异构微型飞行器集群的海洋等复杂环境协同作业。**

- **链接: [https://arxiv.org/pdf/2511.17166v1](https://arxiv.org/pdf/2511.17166v1)**

> **作者:** Tim Lakemann; Daniel Bonilla Licea; Viktor Walter; Martin Saska
>
> **摘要:** Reflections of active markers in the environment are a common source of ambiguity in onboard visual relative localization. This work presents a novel approach for onboard relative localization in multi-robot teams that exploits these typically unwanted reflections of active markers in the environment. It operates without prior knowledge of robot size or predefined marker configurations and remains independent of surface properties, an essential feature for heterogeneous micro-aerial swarms cooperating in unknown environments. It explicitly accounts for uncertainties caused by non-flat surfaces, with a particular focus on dynamic water surfaces, which are especially relevant for marine deployments. We validated the approach in both indoor and outdoor experiments, demonstrating that the proposed reflection-based localization system operates reliably without prior knowledge of team member size and achieves greater effective range (above 30 m) and accuracy than state-of-the-art methods. The video and source code of this work will be made publicly available after publication.
>
---
#### [new 027] MfNeuPAN: Proactive End-to-End Navigation in Dynamic Environments via Direct Multi-Frame Point Constraints
- **分类: cs.RO**

- **简介: 该论文针对动态环境中机器人导航的障碍物避让问题，提出MfNeuPAN框架。通过引入多帧点约束（含未来帧预测），实现基于预测的主动端到端导航，提升复杂动态场景下的导航鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2511.17013v1](https://arxiv.org/pdf/2511.17013v1)**

> **作者:** Yiwen Ying; Hanjing Ye; Senzi Luo; Luyao Liu; Yu Zhan; Li He; Hong Zhang
>
> **备注:** 6 pages, 9 figures, accepted at IEEE ROBIO 2025
>
> **摘要:** Obstacle avoidance in complex and dynamic environments is a critical challenge for real-time robot navigation. Model-based and learning-based methods often fail in highly dynamic scenarios because traditional methods assume a static environment and cannot adapt to real-time changes, while learning-based methods rely on single-frame observations for motion constraint estimation, limiting their adaptability. To overcome these limitations, this paper proposes a novel framework that leverages multi-frame point constraints, including current and future frames predicted by a dedicated module, to enable proactive end-to-end navigation. By incorporating a prediction module that forecasts the future path of moving obstacles based on multi-frame observations, our method allows the robot to proactively anticipate and avoid potential dangers. This proactive planning capability significantly enhances navigation robustness and efficiency in unknown dynamic environments. Simulations and real-world experiments validate the effectiveness of our approach.
>
---
#### [new 028] Feasibility of Embodied Dynamics Based Bayesian Learning for Continuous Pursuit Motion Control of Assistive Mobile Robots in the Built Environment
- **分类: cs.RO; cs.HC**

- **简介: 该论文针对脑机接口（BCI）中缺乏连续追迹运动控制的问题，提出基于具身动力学的贝叶斯学习框架，通过解码加速度级运动表征实现连续速度与方向调控。利用公开数据集验证，相比传统方法，预测误差降低72%，支持更自然、稳定的轮椅导航。**

- **链接: [https://arxiv.org/pdf/2511.17401v1](https://arxiv.org/pdf/2511.17401v1)**

> **作者:** Xiaoshan Zhou; Carol C. Menassa; Vineet R. Kamat
>
> **备注:** 37 pages, 9 figures, and 7 tables
>
> **摘要:** Non-invasive electroencephalography (EEG)-based brain-computer interfaces (BCIs) offer an intuitive means for individuals with severe motor impairments to independently operate assistive robotic wheelchairs and navigate built environments. Despite considerable progress in BCI research, most current motion control systems are limited to discrete commands, rather than supporting continuous pursuit, where users can freely adjust speed and direction in real time. Such natural mobility control is, however, essential for wheelchair users to navigate complex public spaces, such as transit stations, airports, hospitals, and indoor corridors, to interact socially with the dynamic populations with agility, and to move flexibly and comfortably as autonomous driving is refined to allow movement at will. In this study, we address the gap of continuous pursuit motion control in BCIs by proposing and validating a brain-inspired Bayesian inference framework, where embodied dynamics in acceleration-based motor representations are decoded. This approach contrasts with conventional kinematics-level decoding and deep learning-based methods. Using a public dataset with sixteen hours of EEG from four subjects performing motor imagery-based target-following, we demonstrate that our method, utilizing Automatic Relevance Determination for feature selection and continual online learning, reduces the normalized mean squared error between predicted and true velocities by 72% compared to autoregressive and EEGNet-based methods in a session-accumulative transfer learning setting. Theoretically, these findings empirically support embodied cognition theory and reveal the brain's intrinsic motor control dynamics in an embodied and predictive nature. Practically, grounding EEG decoding in the same dynamical principles that govern biological motion offers a promising path toward more stable and intuitive BCI control.
>
---
#### [new 029] A segment anchoring-based balancing algorithm for agricultural multi-robot task allocation with energy constraints
- **分类: cs.MA; cs.RO**

- **简介: 该论文针对智能农业中电动机器人多任务分配问题，解决能量约束下动态负载与充电导致的调度中断难题。提出SABA算法，通过路径锚定与比例分割重构机制，有效平衡任务使能时间与运输成本，显著提升解的质量与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.17076v1](https://arxiv.org/pdf/2511.17076v1)**

> **作者:** Peng Chen; Jing Liang; Kang-Jia Qiao; Hui Song; Tian-lei Ma; Kun-Jie Yu; Cai-Tong Yue; Ponnuthurai Nagaratnam Suganthan; Witold Pedryc
>
> **摘要:** Multi-robot systems have emerged as a key technology for addressing the efficiency and cost challenges in labor-intensive industries. In the representative scenario of smart farming, planning efficient harvesting schedules for a fleet of electric robots presents a highly challenging frontier problem. The complexity arises not only from the need to find Pareto-optimal solutions for the conflicting objectives of makespan and transportation cost, but also from the necessity to simultaneously manage payload constraints and finite battery capacity. When robot loads are dynamically updated during planned multi-trip operations, a mandatory recharge triggered by energy constraints introduces an unscheduled load reset. This interaction creates a complex cascading effect that disrupts the entire schedule and renders traditional optimization methods ineffective. To address this challenge, this paper proposes the segment anchoring-based balancing algorithm (SABA). The core of SABA lies in the organic combination of two synergistic mechanisms: the sequential anchoring and balancing mechanism, which leverages charging decisions as `anchors' to systematically reconstruct disrupted routes, while the proportional splitting-based rebalancing mechanism is responsible for the fine-grained balancing and tuning of the final solutions' makespans. Extensive comparative experiments, conducted on a real-world case study and a suite of benchmark instances, demonstrate that SABA comprehensively outperforms 6 state-of-the-art algorithms in terms of both solution convergence and diversity. This research provides a novel theoretical perspective and an effective solution for the multi-robot task allocation problem under energy constraints.
>
---
#### [new 030] BOP-ASK: Object-Interaction Reasoning for Vision-Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出BOP-ASK数据集，解决视觉语言模型在物体交互理解上的不足。针对现有基准忽略细粒度空间关系的问题，构建包含150k图像和33M问答对的大规模数据集，涵盖抓取、路径规划等六项任务，支持训练与评估。实验表明模型在复杂场景中具备精准定位、规划等新能力。**

- **链接: [https://arxiv.org/pdf/2511.16857v1](https://arxiv.org/pdf/2511.16857v1)**

> **作者:** Vineet Bhat; Sungsu Kim; Valts Blukis; Greg Heinrich; Prashanth Krishnamurthy; Ramesh Karri; Stan Birchfield; Farshad Khorrami; Jonathan Tremblay
>
> **摘要:** Vision Language Models (VLMs) have achieved impressive performance on spatial reasoning benchmarks, yet these evaluations mask critical weaknesses in understanding object interactions. Current benchmarks test high level relationships ('left of,' 'behind', etc.) but ignore fine-grained spatial understanding needed for real world applications: precise 3D localization, physical compatibility between objects, object affordances and multi step spatial planning. In this work, we present BOP-ASK, a novel large scale dataset for object interaction reasoning for both training and benchmarking. Our data generation pipeline leverages 6D object poses from the Benchmark for Object Pose Estimation (BOP) datasets from which we derive fine grained annotations such as grasp poses, referred object poses, path planning trajectories, relative spatial and depth relationships, and object-to-object relationships. BOP-ASK comprises over 150k images and 33M question answer pairs spanning six tasks (four novel), providing a rich resource for training and evaluating VLMs. We evaluate proprietary and open sourced VLMs, and conduct human evaluations on BOP-ASK-core, a contributed test benchmark. We also release BOP-ASK-lab, an out-of-distribution benchmark with images not sourced from BOP, enabling testing of generalization. Our experiments demonstrate that models trained on BOP-ASK outperform baselines and exhibit emergent capabilities such as precise object and grasp pose estimation, trajectory planning, and fine-grained object-centric spatial reasoning in cluttered environments. We will publicly release our datasets and dataset generation pipeline.
>
---
#### [new 031] SING3R-SLAM: Submap-based Indoor Monocular Gaussian SLAM with 3D Reconstruction Priors
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SING3R-SLAM，面向室内单目SLAM任务，解决传统方法中漂移严重、点云冗余、效率低的问题。通过构建局部一致的子地图并融合为全局高斯表示，实现几何与位姿联合优化，提升跟踪精度与重建质量，同时保持紧凑内存占用。**

- **链接: [https://arxiv.org/pdf/2511.17207v1](https://arxiv.org/pdf/2511.17207v1)**

> **作者:** Kunyi Li; Michael Niemeyer; Sen Wang; Stefano Gasperini; Nassir Navab; Federico Tombari
>
> **摘要:** Recent advances in dense 3D reconstruction enable the accurate capture of local geometry; however, integrating them into SLAM is challenging due to drift and redundant point maps, which limit efficiency and downstream tasks, such as novel view synthesis. To address these issues, we propose SING3R-SLAM, a globally consistent and compact Gaussian-based dense RGB SLAM framework. The key idea is to combine locally consistent 3D reconstructions with a unified global Gaussian representation that jointly refines scene geometry and camera poses, enabling efficient and versatile 3D mapping for multiple downstream applications. SING3R-SLAM first builds locally consistent submaps through our lightweight tracking and reconstruction module, and then progressively aligns and fuses them into a global Gaussian map that enforces cross-view geometric consistency. This global map, in turn, provides feedback to correct local drift and enhance the robustness of tracking. Extensive experiments demonstrate that SING3R-SLAM achieves state-of-the-art tracking, 3D reconstruction, and novel view rendering, resulting in over 12% improvement in tracking and producing finer, more detailed geometry, all while maintaining a compact and memory-efficient global representation on real-world datasets.
>
---
#### [new 032] QueryOcc: Query-based Self-Supervision for 3D Semantic Occupancy
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D语义占据估计任务，解决自监督学习中精度与可扩展性不足的问题。提出QueryOcc框架，通过4D时空查询实现连续3D语义占据的直接自监督学习，结合合同性场景表示，支持长距离推理，显著提升性能并保持高效运行。**

- **链接: [https://arxiv.org/pdf/2511.17221v1](https://arxiv.org/pdf/2511.17221v1)**

> **作者:** Adam Lilja; Ji Lan; Junsheng Fu; Lars Hammarstrand
>
> **摘要:** Learning 3D scene geometry and semantics from images is a core challenge in computer vision and a key capability for autonomous driving. Since large-scale 3D annotation is prohibitively expensive, recent work explores self-supervised learning directly from sensor data without manual labels. Existing approaches either rely on 2D rendering consistency, where 3D structure emerges only implicitly, or on discretized voxel grids from accumulated lidar point clouds, limiting spatial precision and scalability. We introduce QueryOcc, a query-based self-supervised framework that learns continuous 3D semantic occupancy directly through independent 4D spatio-temporal queries sampled across adjacent frames. The framework supports supervision from either pseudo-point clouds derived from vision foundation models or raw lidar data. To enable long-range supervision and reasoning under constant memory, we introduce a contractive scene representation that preserves near-field detail while smoothly compressing distant regions. QueryOcc surpasses previous camera-based methods by 26% in semantic RayIoU on the self-supervised Occ3D-nuScenes benchmark while running at 11.6 FPS, demonstrating that direct 4D query supervision enables strong self-supervised occupancy learning. https://research.zenseact.com/publications/queryocc/
>
---
## 更新

#### [replaced 001] CleverDistiller: Simple and Spatially Consistent Cross-modal Distillation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2503.09878v4](https://arxiv.org/pdf/2503.09878v4)**

> **作者:** Hariprasath Govindarajan; Maciej K. Wozniak; Marvin Klingner; Camille Maurice; B Ravi Kiran; Senthil Yogamani
>
> **备注:** Accepted to BMVC 2025
>
> **摘要:** Vision foundation models (VFMs) such as DINO have led to a paradigm shift in 2D camera-based perception towards extracting generalized features to support many downstream tasks. Recent works introduce self-supervised cross-modal knowledge distillation (KD) as a way to transfer these powerful generalization capabilities into 3D LiDAR-based models. However, they either rely on highly complex distillation losses, pseudo-semantic maps, or limit KD to features useful for semantic segmentation only. In this work, we propose CleverDistiller, a self-supervised, cross-modal 2D-to-3D KD framework introducing a set of simple yet effective design choices: Unlike contrastive approaches relying on complex loss design choices, our method employs a direct feature similarity loss in combination with a multi layer perceptron (MLP) projection head to allow the 3D network to learn complex semantic dependencies throughout the projection. Crucially, our approach does not depend on pseudo-semantic maps, allowing for direct knowledge transfer from a VFM without explicit semantic supervision. Additionally, we introduce the auxiliary self-supervised spatial task of occupancy prediction to enhance the semantic knowledge, obtained from a VFM through KD, with 3D spatial reasoning capabilities. Experiments on standard autonomous driving benchmarks for 2D-to-3D KD demonstrate that CleverDistiller achieves state-of-the-art performance in both semantic segmentation and 3D object detection (3DOD) by up to 10% mIoU, especially when fine tuning on really low data amounts, showing the effectiveness of our simple yet powerful KD strategy
>
---
#### [replaced 002] Bootstrap Off-policy with World Model
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2511.00423v2](https://arxiv.org/pdf/2511.00423v2)**

> **作者:** Guojian Zhan; Likun Wang; Xiangteng Zhang; Jiaxin Gao; Masayoshi Tomizuka; Shengbo Eben Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Online planning has proven effective in reinforcement learning (RL) for improving sample efficiency and final performance. However, using planning for environment interaction inevitably introduces a divergence between the collected data and the policy's actual behaviors, degrading both model learning and policy improvement. To address this, we propose BOOM (Bootstrap Off-policy with WOrld Model), a framework that tightly integrates planning and off-policy learning through a bootstrap loop: the policy initializes the planner, and the planner refines actions to bootstrap the policy through behavior alignment. This loop is supported by a jointly learned world model, which enables the planner to simulate future trajectories and provides value targets to facilitate policy improvement. The core of BOOM is a likelihood-free alignment loss that bootstraps the policy using the planner's non-parametric action distribution, combined with a soft value-weighted mechanism that prioritizes high-return behaviors and mitigates variability in the planner's action quality within the replay buffer. Experiments on the high-dimensional DeepMind Control Suite and Humanoid-Bench show that BOOM achieves state-of-the-art results in both training stability and final performance. The code is accessible at https://github.com/molumitu/BOOM_MBRL.
>
---
#### [replaced 003] OpenVLN: Open-world Aerial Vision-Language Navigation
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.06182v2](https://arxiv.org/pdf/2511.06182v2)**

> **作者:** Peican Lin; Gan Sun; Chenxi Liu; Fazeng Li; Weihong Ren; Yang Cong
>
> **备注:** Content: 8 pages 4 figures, conference paper under review
>
> **摘要:** Vision-language models (VLMs) have been widely-applied in ground-based vision-language navigation (VLN). However, the vast complexity of outdoor aerial environments compounds data acquisition challenges and imposes long-horizon trajectory planning requirements on Unmanned Aerial Vehicles (UAVs), introducing novel complexities for aerial VLN. To address these challenges, we propose a data-efficient Open-world aerial Vision-Language Navigation (i.e., OpenVLN) framework, which could execute language-guided flight with limited data constraints and enhance long-horizon trajectory planning capabilities in complex aerial environments. Specifically, we reconfigure a reinforcement learning framework to optimize the VLM for UAV navigation tasks, which can efficiently fine-tune VLM by using rule-based policies under limited training data. Concurrently, we introduce a long-horizon planner for trajectory synthesis that dynamically generates precise UAV actions via value-based rewards. To the end, we conduct sufficient navigation experiments on the TravelUAV benchmark with dataset scaling across diverse reward settings. Our method demonstrates consistent performance gains of up to 4.34% in Success Rate, 6.19% in Oracle Success Rate, and 4.07% in Success weighted by Path Length over baseline methods, validating its deployment efficacy for long-horizon UAV navigation in complex aerial environments.
>
---
#### [replaced 004] DynoSAM: Open-Source Smoothing and Mapping Framework for Dynamic SLAM
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2501.11893v3](https://arxiv.org/pdf/2501.11893v3)**

> **作者:** Jesse Morris; Yiduo Wang; Mikolaj Kliniewski; Viorela Ila
>
> **备注:** 20 pages, 10 figures. Submitted to T-RO Visual SLAM SI 2025
>
> **摘要:** Traditional Visual Simultaneous Localization and Mapping (vSLAM) systems focus solely on static scene structures, overlooking dynamic elements in the environment. Although effective for accurate visual odometry in complex scenarios, these methods discard crucial information about moving objects. By incorporating this information into a Dynamic SLAM framework, the motion of dynamic entities can be estimated, enhancing navigation whilst ensuring accurate localization. However, the fundamental formulation of Dynamic SLAM remains an open challenge, with no consensus on the optimal approach for accurate motion estimation within a SLAM pipeline. Therefore, we developed DynoSAM, an open-source framework for Dynamic SLAM that enables the efficient implementation, testing, and comparison of various Dynamic SLAM optimization formulations. DynoSAM integrates static and dynamic measurements into a unified optimization problem solved using factor graphs, simultaneously estimating camera poses, static scene, object motion or poses, and object structures. We evaluate DynoSAM across diverse simulated and real-world datasets, achieving state-of-the-art motion estimation in indoor and outdoor environments, with substantial improvements over existing systems. Additionally, we demonstrate DynoSAM utility in downstream applications, including 3D reconstruction of dynamic scenes and trajectory prediction, thereby showcasing potential for advancing dynamic object-aware SLAM systems. DynoSAM is open-sourced at https://github.com/ACFR-RPG/DynOSAM.
>
---
#### [replaced 005] VLM-SFD: VLM-Assisted Siamese Flow Diffusion Framework for Dual-Arm Cooperative Manipulation
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2506.13428v2](https://arxiv.org/pdf/2506.13428v2)**

> **作者:** Jiaming Chen; Yiyu Jiang; Aoshen Huang; Yang Li; Wei Pan
>
> **备注:** Accepted by IEEE RA-L
>
> **摘要:** Dual-arm cooperative manipulation holds great promise for tackling complex real-world tasks that demand seamless coordination and adaptive dynamics. Despite substantial progress in learning-based motion planning, most approaches struggle to generalize across diverse manipulation tasks and adapt to dynamic, unstructured environments, particularly in scenarios involving interactions between two objects such as assembly, tool use, and bimanual grasping. To address these challenges, we introduce a novel VLM-Assisted Siamese Flow Diffusion (VLM-SFD) framework for efficient imitation learning in dual-arm cooperative manipulation. The proposed VLM-SFD framework exhibits outstanding adaptability, significantly enhancing the ability to rapidly adapt and generalize to diverse real-world tasks from only a minimal number of human demonstrations. Specifically, we propose a Siamese Flow Diffusion Network (SFDNet) employs a dual-encoder-decoder Siamese architecture to embed two target objects into a shared latent space, while a diffusion-based conditioning process - conditioned by task instructions - generates two-stream object-centric motion flows that guide dual-arm coordination. We further design a dynamic task assignment strategy that seamlessly maps the predicted 2D motion flows into 3D space and incorporates a pre-trained vision-language model (VLM) to adaptively assign the optimal motion to each robotic arm over time. Experiments validate the effectiveness of the proposed method, demonstrating its ability to generalize to diverse manipulation tasks while maintaining high efficiency and adaptability. The code and demo videos are publicly available on our project website https://sites.google.com/view/vlm-sfd/.
>
---
#### [replaced 006] Mask2IV: Interaction-Centric Video Generation via Mask Trajectories
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2510.03135v2](https://arxiv.org/pdf/2510.03135v2)**

> **作者:** Gen Li; Bo Zhao; Jianfei Yang; Laura Sevilla-Lara
>
> **备注:** AAAI 2026. Project page: https://reagan1311.github.io/mask2iv
>
> **摘要:** Generating interaction-centric videos, such as those depicting humans or robots interacting with objects, is crucial for embodied intelligence, as they provide rich and diverse visual priors for robot learning, manipulation policy training, and affordance reasoning. However, existing methods often struggle to model such complex and dynamic interactions. While recent studies show that masks can serve as effective control signals and enhance generation quality, obtaining dense and precise mask annotations remains a major challenge for real-world use. To overcome this limitation, we introduce Mask2IV, a novel framework specifically designed for interaction-centric video generation. It adopts a decoupled two-stage pipeline that first predicts plausible motion trajectories for both actor and object, then generates a video conditioned on these trajectories. This design eliminates the need for dense mask inputs from users while preserving the flexibility to manipulate the interaction process. Furthermore, Mask2IV supports versatile and intuitive control, allowing users to specify the target object of interaction and guide the motion trajectory through action descriptions or spatial position cues. To support systematic training and evaluation, we curate two benchmarks covering diverse action and object categories across both human-object interaction and robotic manipulation scenarios. Extensive experiments demonstrate that our method achieves superior visual realism and controllability compared to existing baselines.
>
---
#### [replaced 007] Perception, Control and Hardware for In-Hand Slip-Aware Object Manipulation with Parallel Grippers
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2410.19660v2](https://arxiv.org/pdf/2410.19660v2)**

> **作者:** Gabriel Arslan Waltersson; Yiannis Karayiannidis
>
> **摘要:** Dexterous in-hand manipulation offers significant potential to enhance robotic manipulator capabilities. This paper presents a sensori-motor architecture for in-hand slip-aware control, being embodied in a sensorized gripper. The gripper in our architecture features rapid closed-loop, low-level force control, and is equipped with sensors capable of independently measuring contact forces and sliding velocities. Our system can quickly estimate essential object properties during pick-up using only in-hand sensing, without relying on prior object information. We introduce four distinct slippage controllers: gravity-assisted trajectory following for both rotational and linear slippage, a hinge controller that maintains the object's orientation while the gripper rotates, and a slip-avoidance controller. The gripper is mounted on a robot arm and validated through extensive experiments involving a diverse range of objects, demonstrating the architecture's novel capabilities for manipulating objects with flat surfaces.
>
---
#### [replaced 008] PhyBlock: A Progressive Benchmark for Physical Understanding and Planning via 3D Block Assembly
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [https://arxiv.org/pdf/2506.08708v2](https://arxiv.org/pdf/2506.08708v2)**

> **作者:** Liang Ma; Jiajun Wen; Min Lin; Rongtao Xu; Xiwen Liang; Bingqian Lin; Jun Ma; Yongxin Wang; Ziming Wei; Haokun Lin; Mingfei Han; Meng Cao; Bokui Chen; Ivan Laptev; Xiaodan Liang
>
> **摘要:** While vision-language models (VLMs) have demonstrated promising capabilities in reasoning and planning for embodied agents, their ability to comprehend physical phenomena, particularly within structured 3D environments, remains severely limited. To close this gap, we introduce PhyBlock, a progressive benchmark designed to assess VLMs on physical understanding and planning through robotic 3D block assembly tasks. PhyBlock integrates a novel four-level cognitive hierarchy assembly task alongside targeted Visual Question Answering (VQA) samples, collectively aimed at evaluating progressive spatial reasoning and fundamental physical comprehension, including object properties, spatial relationships, and holistic scene understanding. PhyBlock includes 2600 block tasks (400 assembly tasks, 2200 VQA tasks) and evaluates models across three key dimensions: partial completion, failure diagnosis, and planning robustness. We benchmark 21 state-of-the-art VLMs, highlighting their strengths and limitations in physically grounded, multi-step planning. Our empirical findings indicate that the performance of VLMs exhibits pronounced limitations in high-level planning and reasoning capabilities, leading to a notable decline in performance for the growing complexity of the tasks. Error analysis reveals persistent difficulties in spatial orientation and dependency reasoning. Surprisingly, chain-of-thought prompting offers minimal improvements, suggesting spatial tasks heavily rely on intuitive model comprehension. We position PhyBlock as a unified testbed to advance embodied reasoning, bridging vision-language understanding and real-world physical problem-solving.
>
---
#### [replaced 009] AgriChrono: A Multi-modal Dataset Capturing Crop Growth and Lighting Variability with a Field Robot
- **分类: cs.RO; cs.AI; eess.SY**

- **链接: [https://arxiv.org/pdf/2508.18694v2](https://arxiv.org/pdf/2508.18694v2)**

> **作者:** Jaehwan Jeong; Tuan-Anh Vu; Mohammad Jony; Shahab Ahmad; Md. Mukhlesur Rahman; Sangpil Kim; M. Khalid Jawed
>
> **摘要:** Advances in AI and Robotics have accelerated significant initiatives in agriculture, particularly in the areas of robot navigation and 3D digital twin creation. A significant bottleneck impeding this progress is the critical lack of "in-the-wild" datasets that capture the full complexities of real farmland, including non-rigid motion from wind, drastic illumination variance, and morphological changes resulting from growth. This data gap fundamentally limits research on robust AI models for autonomous field navigation and scene-level dynamic 3D reconstruction. In this paper, we present AgriChrono, a modular robotic data collection platform and multi-modal dataset designed to capture these dynamic farmland conditions. Our platform integrates multiple sensors, enabling remote, time-synchronized acquisition of RGB, Depth, LiDAR, IMU, and Pose data for efficient and repeatable long-term data collection in real-world agricultural environments. We successfully collected 18TB of data over one month, documenting the entire growth cycle of Canola under diverse illumination conditions. We benchmark state-of-the-art 3D reconstruction methods on AgriChrono, revealing the profound challenge of reconstructing high-fidelity, dynamic non-rigid scenes in such farmland settings. This benchmark validates AgriChrono as a critical asset for advancing model generalization, and its public release is expected to significantly accelerate research and development in precision agriculture. The code and dataset are publicly available at: https://github.com/StructuresComp/agri-chrono
>
---
#### [replaced 010] Searching in Space and Time: Unified Memory-Action Loops for Open-World Object Retrieval
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.14004v3](https://arxiv.org/pdf/2511.14004v3)**

> **作者:** Taijing Chen; Sateesh Kumar; Junhong Xu; Georgios Pavlakos; Joydeep Biswas; Roberto Martín-Martín
>
> **备注:** https://amrl.cs.utexas.edu/STAR/
>
> **摘要:** Service robots must retrieve objects in dynamic, open-world settings where requests may reference attributes ("the red mug"), spatial context ("the mug on the table"), or past states ("the mug that was here yesterday"). Existing approaches capture only parts of this problem: scene graphs capture spatial relations but ignore temporal grounding, temporal reasoning methods model dynamics but do not support embodied interaction, and dynamic scene graphs handle both but remain closed-world with fixed vocabularies. We present STAR (SpatioTemporal Active Retrieval), a framework that unifies memory queries and embodied actions within a single decision loop. STAR leverages non-parametric long-term memory and a working memory to support efficient recall, and uses a vision-language model to select either temporal or spatial actions at each step. We introduce STARBench, a benchmark of spatiotemporal object search tasks across simulated and real environments. Experiments in STARBench and on a Tiago robot show that STAR consistently outperforms scene-graph and memory-only baselines, demonstrating the benefits of treating search in time and search in space as a unified problem. For more information: https://amrl.cs.utexas.edu/STAR.
>
---
#### [replaced 011] AeroVerse: UAV-Agent Benchmark Suite for Simulating, Pre-training, Finetuning, and Evaluating Aerospace Embodied World Models
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2408.15511v2](https://arxiv.org/pdf/2408.15511v2)**

> **作者:** Fanglong Yao; Yuanchang Yue; Youzhi Liu; Xian Sun; Kun Fu
>
> **摘要:** Aerospace embodied intelligence aims to empower unmanned aerial vehicles (UAVs) and other aerospace platforms to achieve autonomous perception, cognition, and action, as well as egocentric active interaction with humans and the environment. The aerospace embodied world model serves as an effective means to realize the autonomous intelligence of UAVs and represents a necessary pathway toward aerospace embodied intelligence. However, existing embodied world models primarily focus on ground-level intelligent agents in indoor scenarios, while research on UAV intelligent agents remains unexplored. To address this gap, we construct the first large-scale real-world image-text pre-training dataset, AerialAgent-Ego10k, featuring urban drones from a first-person perspective. We also create a virtual image-text-pose alignment dataset, CyberAgent Ego500k, to facilitate the pre-training of the aerospace embodied world model. For the first time, we clearly define 5 downstream tasks, i.e., aerospace embodied scene awareness, spatial reasoning, navigational exploration, task planning, and motion decision, and construct corresponding instruction datasets, i.e., SkyAgent-Scene3k, SkyAgent-Reason3k, SkyAgent-Nav3k and SkyAgent-Plan3k, and SkyAgent-Act3k, for fine-tuning the aerospace embodiment world model. Simultaneously, we develop SkyAgentEval, the downstream task evaluation metrics based on GPT-4, to comprehensively, flexibly, and objectively assess the results, revealing the potential and limitations of 2D/3D visual language models in UAV-agent tasks. Furthermore, we integrate over 10 2D/3D visual-language models, 2 pre-training datasets, 5 finetuning datasets, more than 10 evaluation metrics, and a simulator into the benchmark suite, i.e., AeroVerse, which will be released to the community to promote exploration and development of aerospace embodied intelligence.
>
---
