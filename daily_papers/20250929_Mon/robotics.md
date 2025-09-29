# 机器人 cs.RO

- **最新发布 62 篇**

- **更新 24 篇**

## 最新发布

#### [new 001] Improved Vehicle Maneuver Prediction using Game Theoretic Priors
- **分类: cs.RO**

- **简介: 该论文属于车辆行为预测任务，旨在解决传统方法在缺乏场景信息时无法准确预测变道的问题。提出结合博弈论推理生成先验信息，与传统模型融合以提升预测准确性，用于辅助驾驶决策系统。**

- **链接: [http://arxiv.org/pdf/2509.21873v1](http://arxiv.org/pdf/2509.21873v1)**

> **作者:** Nishant Doshi
>
> **摘要:** Conventional maneuver prediction methods use some sort of classification model on temporal trajectory data to predict behavior of agents over a set time horizon. Despite of having the best precision and recall, these models cannot predict a lane change accurately unless they incorporate information about the entire scene. Level-k game theory can leverage the human-like hierarchical reasoning to come up with the most rational decisions each agent can make in a group. This can be leveraged to model interactions between different vehicles in presence of each other and hence compute the most rational decisions each agent would make. The result of game theoretic evaluation can be used as a "prior" or combined with a traditional motion-based classification model to achieve more accurate predictions. The proposed approach assumes that the states of the vehicles around the target lead vehicle are known. The module will output the most rational maneuver prediction of the target vehicle based on an online optimization solution. These predictions are instrumental in decision making systems like Adaptive Cruise Control (ACC) or Traxen's iQ-Cruise further improving the resulting fuel savings.
>
---
#### [new 002] An Adaptive ICP LiDAR Odometry Based on Reliable Initial Pose
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对LiDAR里程计在动态环境中易收敛到局部最优的问题，提出一种基于可靠初始位姿的自适应ICP方法。通过密度过滤粗配准、运动预测优化初始位姿，并结合历史误差动态调整阈值，提升点云配准精度。实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.22058v1](http://arxiv.org/pdf/2509.22058v1)**

> **作者:** Qifeng Wang; Weigang Li; Lei Nie; Xin Xu; Wenping Liu; Zhe Xu
>
> **摘要:** As a key technology for autonomous navigation and positioning in mobile robots, light detection and ranging (LiDAR) odometry is widely used in autonomous driving applications. The Iterative Closest Point (ICP)-based methods have become the core technique in LiDAR odometry due to their efficient and accurate point cloud registration capability. However, some existing ICP-based methods do not consider the reliability of the initial pose, which may cause the method to converge to a local optimum. Furthermore, the absence of an adaptive mechanism hinders the effective handling of complex dynamic environments, resulting in a significant degradation of registration accuracy. To address these issues, this paper proposes an adaptive ICP-based LiDAR odometry method that relies on a reliable initial pose. First, distributed coarse registration based on density filtering is employed to obtain the initial pose estimation. The reliable initial pose is then selected by comparing it with the motion prediction pose, reducing the initial error between the source and target point clouds. Subsequently, by combining the current and historical errors, the adaptive threshold is dynamically adjusted to accommodate the real-time changes in the dynamic environment. Finally, based on the reliable initial pose and the adaptive threshold, point-to-plane adaptive ICP registration is performed from the current frame to the local map, achieving high-precision alignment of the source and target point clouds. Extensive experiments on the public KITTI dataset demonstrate that the proposed method outperforms existing approaches and significantly enhances the accuracy of LiDAR odometry.
>
---
#### [new 003] Towards Versatile Humanoid Table Tennis: Unified Reinforcement Learning with Prediction Augmentation
- **分类: cs.RO**

- **简介: 该论文研究双足机器人打乒乓球任务，旨在解决统一控制快速感知、全身运动和敏捷步态的问题。提出融合预测增强的强化学习框架，通过预测球的状态提升决策与探索效率，实验证明方法在仿真和真实机器人中均表现优异。**

- **链接: [http://arxiv.org/pdf/2509.21690v1](http://arxiv.org/pdf/2509.21690v1)**

> **作者:** Muqun Hu; Wenxi Chen; Wenjing Li; Falak Mandali; Zijian He; Renhong Zhang; Praveen Krisna; Katherine Christian; Leo Benaharon; Dizhi Ma; Karthik Ramani; Yan Gu
>
> **摘要:** Humanoid table tennis (TT) demands rapid perception, proactive whole-body motion, and agile footwork under strict timing -- capabilities that remain difficult for unified controllers. We propose a reinforcement learning framework that maps ball-position observations directly to whole-body joint commands for both arm striking and leg locomotion, strengthened by predictive signals and dense, physics-guided rewards. A lightweight learned predictor, fed with recent ball positions, estimates future ball states and augments the policy's observations for proactive decision-making. During training, a physics-based predictor supplies precise future states to construct dense, informative rewards that lead to effective exploration. The resulting policy attains strong performance across varied serve ranges (hit rate $\geq$ 96% and success rate $\geq$ 92%) in simulations. Ablation studies confirm that both the learned predictor and the predictive reward design are critical for end-to-end learning. Deployed zero-shot on a physical Booster T1 humanoid with 23 revolute joints, the policy produces coordinated lateral and forward-backward footwork with accurate, fast returns, suggesting a practical path toward versatile, competitive humanoid TT.
>
---
#### [new 004] Learning-Based Collaborative Control for Bi-Manual Tactile-Reactive Grasping
- **分类: cs.RO**

- **简介: 该论文研究机器人双臂协作抓取任务，旨在解决柔性及复杂物体抓取稳定性差的问题。提出基于触觉反馈的多智能体MPC控制器，结合实时触觉信息实现自适应控制，提升抓取成功率与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.22421v1](http://arxiv.org/pdf/2509.22421v1)**

> **作者:** Leonel Giacobbe; Jingdao Chen; Chuangchuang Sun
>
> **摘要:** Grasping is a core task in robotics with various applications. However, most current implementations are primarily designed for rigid items, and their performance drops considerably when handling fragile or deformable materials that require real-time feedback. Meanwhile, tactile-reactive grasping focuses on a single agent, which limits their ability to grasp and manipulate large, heavy objects. To overcome this, we propose a learning-based, tactile-reactive multi-agent Model Predictive Controller (MPC) for grasping a wide range of objects with different softness and shapes, beyond the capabilities of preexisting single-agent implementations. Our system uses two Gelsight Mini tactile sensors [1] to extract real-time information on object texture and stiffness. This rich tactile feedback is used to estimate contact dynamics and object compliance in real time, enabling the system to adapt its control policy to diverse object geometries and stiffness profiles. The learned controller operates in a closed loop, leveraging tactile encoding to predict grasp stability and adjust force and position accordingly. Our key technical contributions include a multi-agent MPC formulation trained on real contact interactions, a tactile-data driven method for inferring grasping states, and a coordination strategy that enables collaborative control. By combining tactile sensing and a learning-based multi-agent MPC, our method offers a robust, intelligent solution for collaborative grasping in complex environments, significantly advancing the capabilities of multi-agent systems. Our approach is validated through extensive experiments against independent PD and MPC baselines. Our pipeline outperforms the baselines regarding success rates in achieving and maintaining stable grasps across objects of varying sizes and stiffness.
>
---
#### [new 005] Beyond Detection -- Orchestrating Human-Robot-Robot Assistance via an Internet of Robotic Things Paradigm
- **分类: cs.RO**

- **简介: 该论文提出基于物联网的机器人系统，用于医院中预防患者跌倒。通过热感测预测患者离床意图，并协调两个机器人提供主动帮助，解决传统系统反应滞后和隐私问题，实现个性化、实时的护理辅助。**

- **链接: [http://arxiv.org/pdf/2509.22296v1](http://arxiv.org/pdf/2509.22296v1)**

> **作者:** Joseph Hunt; Koyo Fujii; Aly Magassouba; Praminda Caleb-Solly
>
> **备注:** ICSR 2025, 8 pages, 3 figures
>
> **摘要:** Hospital patient falls remain a critical and costly challenge worldwide. While conventional fall prevention systems typically rely on post-fall detection or reactive alerts, they also often suffer from high false positive rates and fail to address the underlying patient needs that lead to bed-exit attempts. This paper presents a novel system architecture that leverages the Internet of Robotic Things (IoRT) to orchestrate human-robot-robot interaction for proactive and personalized patient assistance. The system integrates a privacy-preserving thermal sensing model capable of real-time bed-exit prediction, with two coordinated robotic agents that respond dynamically based on predicted intent and patient input. This orchestrated response could not only reduce fall risk but also attend to the patient's underlying motivations for movement, such as thirst, discomfort, or the need for assistance, before a hazardous situation arises. Our contributions with this pilot study are three-fold: (1) a modular IoRT-based framework enabling distributed sensing, prediction, and multi-robot coordination; (2) a demonstration of low-resolution thermal sensing for accurate, privacy-preserving preemptive bed-exit detection; and (3) results from a user study and systematic error analysis that inform the design of situationally aware, multi-agent interactions in hospital settings. The findings highlight how interactive and connected robotic systems can move beyond passive monitoring to deliver timely, meaningful assistance, empowering safer, more responsive care environments.
>
---
#### [new 006] One-DoF Robotic Design of Overconstrained Limbs with Energy-Efficient, Self-Collision-Free Motion
- **分类: cs.RO**

- **简介: 该论文提出一种计算方法，设计单自由度（1-DoF）超约束机械肢，解决其在全周期运动中自碰撞和能耗高的问题。通过几何优化与轨迹生成，实现高效、无碰撞的运动，并验证于仿生六足机器人，展示了优异的能效表现。**

- **链接: [http://arxiv.org/pdf/2509.22002v1](http://arxiv.org/pdf/2509.22002v1)**

> **作者:** Yuping Gu; Bangchao Huang; Haoran Sun; Ronghan Xu; Jiayi Yin; Wei Zhang; Fang Wan; Jia Pan; Chaoyang Song
>
> **备注:** 23 pages, 11 figures, 2 tables. Accepted by Fundamental Research. For Supplementary Videos, see https://bionicdl.ancorasir.com/?p=1668
>
> **摘要:** While it is expected to build robotic limbs with multiple degrees of freedom (DoF) inspired by nature, a single DoF design remains fundamental, providing benefits that include, but are not limited to, simplicity, robustness, cost-effectiveness, and efficiency. Mechanisms, especially those with multiple links and revolute joints connected in closed loops, play an enabling factor in introducing motion diversity for 1-DoF systems, which are usually constrained by self-collision during a full-cycle range of motion. This study presents a novel computational approach to designing one-degree-of-freedom (1-DoF) overconstrained robotic limbs for a desired spatial trajectory, while achieving energy-efficient, self-collision-free motion in full-cycle rotations. Firstly, we present the geometric optimization problem of linkage-based robotic limbs in a generalized formulation for self-collision-free design. Next, we formulate the spatial trajectory generation problem with the overconstrained linkages by optimizing the similarity and dynamic-related metrics. We further optimize the geometric shape of the overconstrained linkage to ensure smooth and collision-free motion driven by a single actuator. We validated our proposed method through various experiments, including personalized automata and bio-inspired hexapod robots. The resulting hexapod robot, featuring overconstrained robotic limbs, demonstrated outstanding energy efficiency during forward walking.
>
---
#### [new 007] Learnable Conformal Prediction with Context-Aware Nonconformity Functions for Robotic Planning and Perception
- **分类: cs.RO; cs.LG; math.ST; stat.TH**

- **简介: 该论文提出Learnable Conformal Prediction (LCP)，用于机器人规划与感知任务，旨在解决传统方法在不确定性量化中忽略上下文信息的问题。通过引入轻量级神经网络生成上下文感知的不确定性集，LCP在多个基准上提升了预测精度和安全性，同时保持低计算开销。**

- **链接: [http://arxiv.org/pdf/2509.21955v1](http://arxiv.org/pdf/2509.21955v1)**

> **作者:** Divake Kumar; Sina Tayebati; Francesco Migliarba; Ranganath Krishnan; Amit Ranjan Trivedi
>
> **摘要:** Deep learning models in robotics often output point estimates with poorly calibrated confidences, offering no native mechanism to quantify predictive reliability under novel, noisy, or out-of-distribution inputs. Conformal prediction (CP) addresses this gap by providing distribution-free coverage guarantees, yet its reliance on fixed nonconformity scores ignores context and can yield intervals that are overly conservative or unsafe. We address this with Learnable Conformal Prediction (LCP), which replaces fixed scores with a lightweight neural function that leverages geometric, semantic, and task-specific features to produce context-aware uncertainty sets. LCP maintains CP's theoretical guarantees while reducing prediction set sizes by 18% in classification, tightening detection intervals by 52%, and improving path planning safety from 72% to 91% success with minimal overhead. Across three robotic tasks on seven benchmarks, LCP consistently outperforms Standard CP and ensemble baselines. In classification on CIFAR-100 and ImageNet, it achieves smaller set sizes (4.7-9.9% reduction) at target coverage. For object detection on COCO, BDD100K, and Cityscapes, it produces 46-54% tighter bounding boxes. In path planning through cluttered environments, it improves success to 91.5% with only 4.5% path inflation, compared to 12.2% for Standard CP. The method is lightweight (approximately 4.8% runtime overhead, 42 KB memory) and supports online adaptation, making it well suited to resource-constrained autonomous systems. Hardware evaluation shows LCP adds less than 1% memory and 15.9% inference overhead, yet sustains 39 FPS on detection tasks while being 7.4 times more energy-efficient than ensembles.
>
---
#### [new 008] Leveraging Large Language Models for Robot-Assisted Learning of Morphological Structures in Preschool Children with Language Vulnerabilities
- **分类: cs.RO; cs.AI; cs.HC; I.2.7; H.5.2; K.3.1; J.4**

- **简介: 该论文探讨利用大型语言模型（LLM）和对话机器人TalBot辅助语言弱势学龄前儿童学习形态结构。通过游戏“Alias”提高语言技能，解决教育者在实时生成语法结构和维持互动中的挑战，目标是开发跨语言的LLM驱动语言干预系统。**

- **链接: [http://arxiv.org/pdf/2509.22287v1](http://arxiv.org/pdf/2509.22287v1)**

> **作者:** Stina Sundstedt; Mattias Wingren; Susanne Hägglund; Daniel Ventus
>
> **备注:** 12 pages, 2 figures, Preprint of: Sundstedt, S., Wingren, M., H\"agglund, S. & Ventus, D. (2025). Leveraging Large Language Models for Robot-Assisted Learning of Morphological Structures in Preschool Children with Language Vulnerabilities. In: Stephanidis, C., Antona, M., Ntoa, S. & Salvendy, G. (eds.), Communications in Computer and Information Science, vol. 2523, pp. 415-425. Springer
>
> **摘要:** Preschool children with language vulnerabilities -- such as developmental language disorders or immigration related language challenges -- often require support to strengthen their expressive language skills. Based on the principle of implicit learning, speech-language therapists (SLTs) typically embed target morphological structures (e.g., third person -s) into everyday interactions or game-based learning activities. Educators are recommended by SLTs to do the same. This approach demands precise linguistic knowledge and real-time production of various morphological forms (e.g., "Daddy wears these when he drives to work"). The task becomes even more demanding when educators or parent also must keep children engaged and manage turn-taking in a game-based activity. In the TalBot project our multiprofessional team have developed an application in which the Furhat conversational robot plays the word retrieval game "Alias" with children to improve language skills. Our application currently employs a large language model (LLM) to manage gameplay, dialogue, affective responses, and turn-taking. Our next step is to further leverage the capacity of LLMs so the robot can generate and deliver specific morphological targets during the game. We hypothesize that a robot could outperform humans at this task. Novel aspects of this approach are that the robot could ultimately serve as a model and tutor for both children and professionals and that using LLM capabilities in this context would support basic communication needs for children with language vulnerabilities. Our long-term goal is to create a robust LLM-based Robot-Assisted Language Learning intervention capable of teaching a variety of morphological structures across different languages.
>
---
#### [new 009] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Robotic Bimanual Manipulation
- **分类: cs.RO**

- **简介: 该论文提出VLBiMan，用于双臂机器人操作任务。针对现有方法需大量示例且适应性差的问题，通过单次示范结合视觉-语言锚定实现高效学习与泛化，提升机器人在复杂环境中的灵活性和跨平台迁移能力。**

- **链接: [http://arxiv.org/pdf/2509.21723v1](http://arxiv.org/pdf/2509.21723v1)**

> **作者:** Huayi Zhou; Kui Jia
>
> **备注:** under review
>
> **摘要:** Achieving generalizable bimanual manipulation requires systems that can learn efficiently from minimal human input while adapting to real-world uncertainties and diverse embodiments. Existing approaches face a dilemma: imitation policy learning demands extensive demonstrations to cover task variations, while modular methods often lack flexibility in dynamic scenes. We introduce VLBiMan, a framework that derives reusable skills from a single human example through task-aware decomposition, preserving invariant primitives as anchors while dynamically adapting adjustable components via vision-language grounding. This adaptation mechanism resolves scene ambiguities caused by background changes, object repositioning, or visual clutter without policy retraining, leveraging semantic parsing and geometric feasibility constraints. Moreover, the system inherits human-like hybrid control capabilities, enabling mixed synchronous and asynchronous use of both arms. Extensive experiments validate VLBiMan across tool-use and multi-object tasks, demonstrating: (1) a drastic reduction in demonstration requirements compared to imitation baselines, (2) compositional generalization through atomic skill splicing for long-horizon tasks, (3) robustness to novel but semantically similar objects and external disturbances, and (4) strong cross-embodiment transfer, showing that skills learned from human demonstrations can be instantiated on different robotic platforms without retraining. By bridging human priors with vision-language anchored adaptation, our work takes a step toward practical and versatile dual-arm manipulation in unstructured settings.
>
---
#### [new 010] WoW: Towards a World omniscient World model Through Embodied Interaction
- **分类: cs.RO; cs.CV; cs.MM**

- **简介: 该论文提出WoW，一个通过200万条机器人交互轨迹训练的140亿参数生成式世界模型，旨在解决视频模型缺乏物理因果理解的问题。通过SOPHIA机制约束生成结果，并构建逆动力学模型实现从想象到行动的闭环，最终在物理一致性与因果推理任务中取得SOTA表现。**

- **链接: [http://arxiv.org/pdf/2509.22642v1](http://arxiv.org/pdf/2509.22642v1)**

> **作者:** Xiaowei Chi; Peidong Jia; Chun-Kai Fan; Xiaozhu Ju; Weishi Mi; Kevin Zhang; Zhiyuan Qin; Wanxin Tian; Kuangzhi Ge; Hao Li; Zezhong Qian; Anthony Chen; Qiang Zhou; Yueru Jia; Jiaming Liu; Yong Dai; Qingpo Wuwu; Chengyu Bai; Yu-Kai Wang; Ying Li; Lizhang Chen; Yong Bao; Zhiyuan Jiang; Jiacheng Zhu; Kai Tang; Ruichuan An; Yulin Luo; Qiuxuan Feng; Siyuan Zhou; Chi-min Chan; Chengkai Hou; Wei Xue; Sirui Han; Yike Guo; Shanghang Zhang; Jian Tang
>
> **摘要:** Humans develop an understanding of intuitive physics through active interaction with the world. This approach is in stark contrast to current video models, such as Sora, which rely on passive observation and therefore struggle with grasping physical causality. This observation leads to our central hypothesis: authentic physical intuition of the world model must be grounded in extensive, causally rich interactions with the real world. To test this hypothesis, we present WoW, a 14-billion-parameter generative world model trained on 2 million robot interaction trajectories. Our findings reveal that the model's understanding of physics is a probabilistic distribution of plausible outcomes, leading to stochastic instabilities and physical hallucinations. Furthermore, we demonstrate that this emergent capability can be actively constrained toward physical realism by SOPHIA, where vision-language model agents evaluate the DiT-generated output and guide its refinement by iteratively evolving the language instructions. In addition, a co-trained Inverse Dynamics Model translates these refined plans into executable robotic actions, thus closing the imagination-to-action loop. We establish WoWBench, a new benchmark focused on physical consistency and causal reasoning in video, where WoW achieves state-of-the-art performance in both human and autonomous evaluation, demonstrating strong ability in physical causality, collision dynamics, and object permanence. Our work provides systematic evidence that large-scale, real-world interaction is a cornerstone for developing physical intuition in AI. Models, data, and benchmarks will be open-sourced.
>
---
#### [new 011] Language-in-the-Loop Culvert Inspection on the Erie Canal
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VISION系统，用于运河涵洞的自主检测。针对人工检测困难的问题，结合视觉-语言模型与路径规划，实现自动识别、定位并拍摄关键区域，提升检测效率与准确性。**

- **链接: [http://arxiv.org/pdf/2509.21370v1](http://arxiv.org/pdf/2509.21370v1)**

> **作者:** Yashom Dighe; Yash Turkar; Karthik Dantu
>
> **备注:** First two authors contributed equally
>
> **摘要:** Culverts on canals such as the Erie Canal, built originally in 1825, require frequent inspections to ensure safe operation. Human inspection of culverts is challenging due to age, geometry, poor illumination, weather, and lack of easy access. We introduce VISION, an end-to-end, language-in-the-loop autonomy system that couples a web-scale vision-language model (VLM) with constrained viewpoint planning for autonomous inspection of culverts. Brief prompts to the VLM solicit open-vocabulary ROI proposals with rationales and confidences, stereo depth is fused to recover scale, and a planner -- aware of culvert constraints -- commands repositioning moves to capture targeted close-ups. Deployed on a quadruped in a culvert under the Erie Canal, VISION closes the see, decide, move, re-image loop on-board and produces high-resolution images for detailed reporting without domain-specific fine-tuning. In an external evaluation by New York Canal Corporation personnel, initial ROI proposals achieved 61.4\% agreement with subject-matter experts, and final post-re-imaging assessments reached 80\%, indicating that VISION converts tentative hypotheses into grounded, expert-aligned findings.
>
---
#### [new 012] SAGE: Scene Graph-Aware Guidance and Execution for Long-Horizon Manipulation Tasks
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SAGE框架，用于长视野操作任务。针对现有方法在任务规划和图像控制中的不足，SAGE利用语义场景图连接高层推理与底层控制，并通过结构化图像编辑生成子目标图像，实现了端到端的任务执行与控制。**

- **链接: [http://arxiv.org/pdf/2509.21928v1](http://arxiv.org/pdf/2509.21928v1)**

> **作者:** Jialiang Li; Wenzheng Wu; Gaojing Zhang; Yifan Han; Wenzhao Lian
>
> **摘要:** Successfully solving long-horizon manipulation tasks remains a fundamental challenge. These tasks involve extended action sequences and complex object interactions, presenting a critical gap between high-level symbolic planning and low-level continuous control. To bridge this gap, two essential capabilities are required: robust long-horizon task planning and effective goal-conditioned manipulation. Existing task planning methods, including traditional and LLM-based approaches, often exhibit limited generalization or sparse semantic reasoning. Meanwhile, image-conditioned control methods struggle to adapt to unseen tasks. To tackle these problems, we propose SAGE, a novel framework for Scene Graph-Aware Guidance and Execution in Long-Horizon Manipulation Tasks. SAGE utilizes semantic scene graphs as a structural representation for scene states. A structural scene graph enables bridging task-level semantic reasoning and pixel-level visuo-motor control. This also facilitates the controllable synthesis of accurate, novel sub-goal images. SAGE consists of two key components: (1) a scene graph-based task planner that uses VLMs and LLMs to parse the environment and reason about physically-grounded scene state transition sequences, and (2) a decoupled structural image editing pipeline that controllably converts each target sub-goal graph into a corresponding image through image inpainting and composition. Extensive experiments have demonstrated that SAGE achieves state-of-the-art performance on distinct long-horizon tasks.
>
---
#### [new 013] MINT-RVAE: Multi-Cues Intention Prediction of Human-Robot Interaction using Human Pose and Emotion Information from RGB-only Camera Data
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MINT-RVAE，用于基于RGB相机数据的人机交互意图预测任务。旨在解决真实场景中类别不平衡问题，通过合成序列生成和新损失函数提升模型性能与泛化能力，实现帧级精度的意图检测。**

- **链接: [http://arxiv.org/pdf/2509.22573v1](http://arxiv.org/pdf/2509.22573v1)**

> **作者:** Farida Mohsen; Ali Safa
>
> **摘要:** Efficiently detecting human intent to interact with ubiquitous robots is crucial for effective human-robot interaction (HRI) and collaboration. Over the past decade, deep learning has gained traction in this field, with most existing approaches relying on multimodal inputs, such as RGB combined with depth (RGB-D), to classify time-sequence windows of sensory data as interactive or non-interactive. In contrast, we propose a novel RGB-only pipeline for predicting human interaction intent with frame-level precision, enabling faster robot responses and improved service quality. A key challenge in intent prediction is the class imbalance inherent in real-world HRI datasets, which can hinder the model's training and generalization. To address this, we introduce MINT-RVAE, a synthetic sequence generation method, along with new loss functions and training strategies that enhance generalization on out-of-sample data. Our approach achieves state-of-the-art performance (AUROC: 0.95) outperforming prior works (AUROC: 0.90-0.912), while requiring only RGB input and supporting precise frame onset prediction. Finally, to support future research, we openly release our new dataset with frame-level labeling of human interaction intent.
>
---
#### [new 014] See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出See, Point, Fly（SPF），一种无需训练的视觉-语言导航框架，用于无人机在复杂环境中根据自然语言指令自主导航。其核心是将动作预测转化为2D空间定位任务，并结合3D位移控制无人机，实现了高效、通用的闭环导航。**

- **链接: [http://arxiv.org/pdf/2509.22653v1](http://arxiv.org/pdf/2509.22653v1)**

> **作者:** Chih Yao Hu; Yang-Sen Lin; Yuna Lee; Chih-Hai Su; Jie-Ying Lee; Shr-Ruei Tsai; Chin-Yang Lin; Kuan-Wen Chen; Tsung-Wei Ke; Yu-Lun Liu
>
> **备注:** CoRL 2025. Project page: https://spf-web.pages.dev
>
> **摘要:** We present See, Point, Fly (SPF), a training-free aerial vision-and-language navigation (AVLN) framework built atop vision-language models (VLMs). SPF is capable of navigating to any goal based on any type of free-form instructions in any kind of environment. In contrast to existing VLM-based approaches that treat action prediction as a text generation task, our key insight is to consider action prediction for AVLN as a 2D spatial grounding task. SPF harnesses VLMs to decompose vague language instructions into iterative annotation of 2D waypoints on the input image. Along with the predicted traveling distance, SPF transforms predicted 2D waypoints into 3D displacement vectors as action commands for UAVs. Moreover, SPF also adaptively adjusts the traveling distance to facilitate more efficient navigation. Notably, SPF performs navigation in a closed-loop control manner, enabling UAVs to follow dynamic targets in dynamic environments. SPF sets a new state of the art in DRL simulation benchmark, outperforming the previous best method by an absolute margin of 63%. In extensive real-world evaluations, SPF outperforms strong baselines by a large margin. We also conduct comprehensive ablation studies to highlight the effectiveness of our design choice. Lastly, SPF shows remarkable generalization to different VLMs. Project page: https://spf-web.pages.dev
>
---
#### [new 015] IMU-Preintegrated Radar Factors for Asynchronous Radar-LiDAR-Inertial SLAM
- **分类: cs.RO**

- **简介: 该论文研究雷达-激光雷达-惯性导航的SLAM任务，旨在解决传感器异步导致的状态节点过多、优化成本高的问题。提出基于IMU预积分的雷达因子方法，将LiDAR状态传播至雷达时间戳，降低状态数50%，优化时间减少56%。**

- **链接: [http://arxiv.org/pdf/2509.22288v1](http://arxiv.org/pdf/2509.22288v1)**

> **作者:** Johan Hatleskog; Morten Nissov; Kostas Alexis
>
> **备注:** 8 pages, 7 figures, accepted by The 22nd International Conference on Advanced Robotics (ICAR 2025). Supplementary video: https://youtu.be/95jeWXBMN7c
>
> **摘要:** Fixed-lag Radar-LiDAR-Inertial smoothers conventionally create one factor graph node per measurement to compensate for the lack of time synchronization between radar and LiDAR. For a radar-LiDAR sensor pair with equal rates, this strategy results in a state creation rate of twice the individual sensor frequencies. This doubling of the number of states per second yields high optimization costs, inhibiting real-time performance on resource-constrained hardware. We introduce IMU-preintegrated radar factors that use high-rate inertial data to propagate the most recent LiDAR state to the radar measurement timestamp. This strategy maintains the node creation rate at the LiDAR measurement frequency. Assuming equal sensor rates, this lowers the number of nodes by 50 % and consequently the computational costs. Experiments on a single board computer (which has 4 cores each of 2.2 GHz A73 and 2 GHz A53 with 8 GB RAM) show that our method preserves the absolute pose error of a conventional baseline while simultaneously lowering the aggregated factor graph optimization time by up to 56 %.
>
---
#### [new 016] Ontological foundations for contrastive explanatory narration of robot plans
- **分类: cs.RO; cs.AI; cs.IR; cs.LO**

- **简介: 该论文研究机器人计划的对比解释生成任务，旨在解决机器人如何向人类清晰解释不同计划选择的问题。提出了一种新的本体模型和算法，用于建模、区分并构建对比性解释，提升解释效果。**

- **链接: [http://arxiv.org/pdf/2509.22493v1](http://arxiv.org/pdf/2509.22493v1)**

> **作者:** Alberto Olivares-Alarcos; Sergi Foix; Júlia Borràs; Gerard Canal; Guillem Alenyà
>
> **备注:** This version was submitted to the journal Information Sciences and is under review since October 2024
>
> **摘要:** Mutual understanding of artificial agents' decisions is key to ensuring a trustworthy and successful human-robot interaction. Hence, robots are expected to make reasonable decisions and communicate them to humans when needed. In this article, the focus is on an approach to modeling and reasoning about the comparison of two competing plans, so that robots can later explain the divergent result. First, a novel ontological model is proposed to formalize and reason about the differences between competing plans, enabling the classification of the most appropriate one (e.g., the shortest, the safest, the closest to human preferences, etc.). This work also investigates the limitations of a baseline algorithm for ontology-based explanatory narration. To address these limitations, a novel algorithm is presented, leveraging divergent knowledge between plans and facilitating the construction of contrastive narratives. Through empirical evaluation, it is observed that the explanations excel beyond the baseline method.
>
---
#### [new 017] HELIOS: Hierarchical Exploration for Language-grounded Interaction in Open Scenes
- **分类: cs.RO**

- **简介: 该论文提出HELIOS方法，用于解决语言指定的移动机械臂在开放场景中的操作任务。面对部分可观测环境和语义信息对齐问题，HELIOS构建了层次化场景表示，并设计搜索目标以平衡探索与利用，实现了高效的拾取与放置操作。**

- **链接: [http://arxiv.org/pdf/2509.22498v1](http://arxiv.org/pdf/2509.22498v1)**

> **作者:** Katrina Ashton; Chahyon Ku; Shrey Shah; Wen Jiang; Kostas Daniilidis; Bernadette Bucher
>
> **摘要:** Language-specified mobile manipulation tasks in novel environments simultaneously face challenges interacting with a scene which is only partially observed, grounding semantic information from language instructions to the partially observed scene, and actively updating knowledge of the scene with new observations. To address these challenges, we propose HELIOS, a hierarchical scene representation and associated search objective to perform language specified pick and place mobile manipulation tasks. We construct 2D maps containing the relevant semantic and occupancy information for navigation while simultaneously actively constructing 3D Gaussian representations of task-relevant objects. We fuse observations across this multi-layered representation while explicitly modeling the multi-view consistency of the detections of each object. In order to efficiently search for the target object, we formulate an objective function balancing exploration of unobserved or uncertain regions with exploitation of scene semantic information. We evaluate HELIOS on the OVMM benchmark in the Habitat simulator, a pick and place benchmark in which perception is challenging due to large and complex scenes with comparatively small target objects. HELIOS achieves state-of-the-art results on OVMM. As our approach is zero-shot, HELIOS can also transfer to the real world without requiring additional data, as we illustrate by demonstrating it in a real world office environment on a Spot robot.
>
---
#### [new 018] Pixel Motion Diffusion is What We Need for Robot Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DAWN，一个基于扩散模型的统一框架，用于语言条件下的机器人操作任务。它通过结构化像素运动表示，连接高层意图与低层动作，实现端到端控制。实验在CALVIN和MetaWorld上展示了其优越的多任务性能和现实迁移能力。**

- **链接: [http://arxiv.org/pdf/2509.22652v1](http://arxiv.org/pdf/2509.22652v1)**

> **作者:** E-Ro Nguyen; Yichi Zhang; Kanchana Ranasinghe; Xiang Li; Michael S. Ryoo
>
> **备注:** 16 pages, 7 figures
>
> **摘要:** We present DAWN (Diffusion is All We Need for robot control), a unified diffusion-based framework for language-conditioned robotic manipulation that bridges high-level motion intent and low-level robot action via structured pixel motion representation. In DAWN, both the high-level and low-level controllers are modeled as diffusion processes, yielding a fully trainable, end-to-end system with interpretable intermediate motion abstractions. DAWN achieves state-of-the-art results on the challenging CALVIN benchmark, demonstrating strong multi-task performance, and further validates its effectiveness on MetaWorld. Despite the substantial domain gap between simulation and reality and limited real-world data, we demonstrate reliable real-world transfer with only minimal finetuning, illustrating the practical viability of diffusion-based motion abstractions for robotic control. Our results show the effectiveness of combining diffusion modeling with motion-centric representations as a strong baseline for scalable and robust robot learning. Project page: https://nero1342.github.io/DAWN/
>
---
#### [new 019] Multi-stage robust nonlinear model predictive control of a lower-limb exoskeleton robot
- **分类: cs.RO**

- **简介: 该论文提出一种多阶段鲁棒非线性模型预测控制（RNMPC）方法，用于提高下肢外骨骼机器人的控制鲁棒性。针对人机系统不确定性及动力学非线性问题，通过多场景优化降低交互力与跟踪误差，实验验证其在未知负载和扰动下的优越性能。**

- **链接: [http://arxiv.org/pdf/2509.22120v1](http://arxiv.org/pdf/2509.22120v1)**

> **作者:** Alireza Aliyari; Gholamreza Vossoughi
>
> **备注:** 12 pages, 11 figures, 2 tables, under review at the journal of "Transactions of the Canadian Society for Mechanical Engineering"
>
> **摘要:** The use of exoskeleton robots is increasing due to the rising number of musculoskeletal injuries. However, their effectiveness depends heavily on the design of control systems. Designing robust controllers is challenging because of uncertainties in human-robot systems. Among various control strategies, Model Predictive Control (MPC) is a powerful approach due to its ability to handle constraints and optimize performance. Previous studies have used linearization-based methods to implement robust MPC on exoskeletons, but these can degrade performance due to nonlinearities in the robot's dynamics. To address this gap, this paper proposes a Robust Nonlinear Model Predictive Control (RNMPC) method, called multi-stage NMPC, to control a two-degree-of-freedom exoskeleton by solving a nonlinear optimization problem. This method uses multiple scenarios to represent system uncertainties. The study focuses on minimizing human-robot interaction forces during the swing phase, particularly when the robot carries unknown loads. Simulations and experimental tests show that the proposed method significantly improves robustness, outperforming non-robust NMPC. It achieves lower tracking errors and interaction forces under various uncertainties. For instance, when a 2 kg unknown payload is combined with external disturbances, the RMS values of thigh and shank interaction forces for multi-stage NMPC are reduced by 77 and 94 percent, respectively, compared to non-robust NMPC.
>
---
#### [new 020] Wall Inspector: Quadrotor Control in Wall-proximity Through Model Compensation
- **分类: cs.RO**

- **简介: 该论文针对四旋翼无人机在近墙环境中因气动效应导致的不稳定问题，提出了一种基于物理模型补偿的SC-MPC控制方法，提升了轨迹跟踪精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.21496v1](http://arxiv.org/pdf/2509.21496v1)**

> **作者:** Peiwen Yang; Weisong Wen; Runqiu Yang; Yingming Chen; Cheuk Chi Tsang
>
> **摘要:** The safe operation of quadrotors in near-wall urban or indoor environments (e.g., inspection and search-and-rescue missions) is challenged by unmodeled aerodynamic effects arising from wall-proximity. It generates complex vortices that induce destabilizing suction forces, potentially leading to hazardous vibrations or collisions. This paper presents a comprehensive solution featuring (1) a physics-based suction force model that explicitly characterizes the dependency on both rotor speed and wall distance, and (2) a suction-compensated model predictive control (SC-MPC) framework designed to ensure accurate and stable trajectory tracking during wall-proximity operations. The proposed SC-MPC framework incorporates an enhanced dynamics model that accounts for suction force effects, formulated as a factor graph optimization problem integrating system dynamics constraints, trajectory tracking objectives, control input smoothness requirements, and actuator physical limitations. The suction force model parameters are systematically identified through extensive experimental measurements across varying operational conditions. Experimental validation demonstrates SC-MPC's superior performance, achieving 2.1 cm root mean squared error (RMSE) in X-axis and 2.0 cm RMSE in Y-axis position control - representing 74% and 79% improvements over cascaded proportional-integral-derivative (PID) control, and 60% and 53% improvements over standard MPC respectively. The corresponding mean absolute error (MAE) metrics (1.2 cm X-axis, 1.4 cm Y-axis) similarly outperform both baselines. The evaluation platform employs a ducted quadrotor design that provides collision protection while maintaining aerodynamic efficiency. To facilitate reproducibility and community adoption, we have open-sourced our complete implementation, available at https://anonymous.4open.science/r/SC-MPC-6A61.
>
---
#### [new 021] The Turkish Ice Cream Robot: Examining Playful Deception in Social Human-Robot Interactions
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究社交人机交互中的“趣味性欺骗”问题，设计了一个受土耳其冰淇淋小贩启发的机器人，通过延迟传递动作探讨欺骗对用户信任、愉悦和参与度的影响。实验表明，这种欺骗能提升娱乐性和参与感，但会降低安全感知与信任。**

- **链接: [http://arxiv.org/pdf/2509.21776v1](http://arxiv.org/pdf/2509.21776v1)**

> **作者:** Hyeonseong Kim; Roy El-Helou; Seungbeen Lee; Sungjoon Choi; Matthew Pan
>
> **备注:** for more videos, see https://hyeonseong-kim98.github.io/turkish-ice-cream-robot/
>
> **摘要:** Playful deception, a common feature in human social interactions, remains underexplored in Human-Robot Interaction (HRI). Inspired by the Turkish Ice Cream (TIC) vendor routine, we investigate how bounded, culturally familiar forms of deception influence user trust, enjoyment, and engagement during robotic handovers. We design a robotic manipulator equipped with a custom end-effector and implement five TIC-inspired trick policies that deceptively delay the handover of an ice cream-shaped object. Through a mixed-design user study with 91 participants, we evaluate the effects of playful deception and interaction duration on user experience. Results reveal that TIC-inspired deception significantly enhances enjoyment and engagement, though reduces perceived safety and trust, suggesting a structured trade-off across the multi-dimensional aspects. Our findings demonstrate that playful deception can be a valuable design strategy for interactive robots in entertainment and engagement-focused contexts, while underscoring the importance of deliberate consideration of its complex trade-offs. You can find more information, including demonstration videos, on https://hyeonseong-kim98.github.io/turkish-ice-cream-robot/ .
>
---
#### [new 022] DemoGrasp: Universal Dexterous Grasping from a Single Demonstration
- **分类: cs.RO**

- **简介: 该论文提出DemoGrasp，旨在解决通用灵巧抓取问题。通过单次示范轨迹编辑（调整腕部位姿和手部关节），结合强化学习优化策略，在仿真和现实环境中实现高成功率的多对象、多场景抓取，支持视觉输入与语言引导。**

- **链接: [http://arxiv.org/pdf/2509.22149v1](http://arxiv.org/pdf/2509.22149v1)**

> **作者:** Haoqi Yuan; Ziye Huang; Ye Wang; Chuan Mao; Chaoyi Xu; Zongqing Lu
>
> **摘要:** Universal grasping with multi-fingered dexterous hands is a fundamental challenge in robotic manipulation. While recent approaches successfully learn closed-loop grasping policies using reinforcement learning (RL), the inherent difficulty of high-dimensional, long-horizon exploration necessitates complex reward and curriculum design, often resulting in suboptimal solutions across diverse objects. We propose DemoGrasp, a simple yet effective method for learning universal dexterous grasping. We start from a single successful demonstration trajectory of grasping a specific object and adapt to novel objects and poses by editing the robot actions in this trajectory: changing the wrist pose determines where to grasp, and changing the hand joint angles determines how to grasp. We formulate this trajectory editing as a single-step Markov Decision Process (MDP) and use RL to optimize a universal policy across hundreds of objects in parallel in simulation, with a simple reward consisting of a binary success term and a robot-table collision penalty. In simulation, DemoGrasp achieves a 95% success rate on DexGraspNet objects using the Shadow Hand, outperforming previous state-of-the-art methods. It also shows strong transferability, achieving an average success rate of 84.6% across diverse dexterous hand embodiments on six unseen object datasets, while being trained on only 175 objects. Through vision-based imitation learning, our policy successfully grasps 110 unseen real-world objects, including small, thin items. It generalizes to spatial, background, and lighting changes, supports both RGB and depth inputs, and extends to language-guided grasping in cluttered scenes.
>
---
#### [new 023] An Intention-driven Lane Change Framework Considering Heterogeneous Dynamic Cooperation in Mixed-traffic Environment
- **分类: cs.RO**

- **简介: 该论文提出一种意图驱动的变道框架，用于混合交通环境中异构车辆的协同交互。通过驾驶风格识别、合作感知决策和轨迹规划，解决自动驾驶中因人类驾驶不可预测性带来的安全高效变道问题，提升了变道识别性能。**

- **链接: [http://arxiv.org/pdf/2509.22550v1](http://arxiv.org/pdf/2509.22550v1)**

> **作者:** Xiaoyun Qiu; Haichao Liu; Yue Pan; Jun Ma; Xinhu Zheng
>
> **摘要:** In mixed-traffic environments, where autonomous vehicles (AVs) interact with diverse human-driven vehicles (HVs), unpredictable intentions and heterogeneous behaviors make safe and efficient lane change maneuvers highly challenging. Existing methods often oversimplify these interactions by assuming uniform patterns. We propose an intention-driven lane change framework that integrates driving-style recognition, cooperation-aware decision-making, and coordinated motion planning. A deep learning classifier trained on the NGSIM dataset identifies human driving styles in real time. A cooperation score with intrinsic and interactive components estimates surrounding drivers' intentions and quantifies their willingness to cooperate with the ego vehicle. Decision-making combines behavior cloning with inverse reinforcement learning to determine whether a lane change should be initiated. For trajectory generation, model predictive control is integrated with IRL-based intention inference to produce collision-free and socially compliant maneuvers. Experiments show that the proposed model achieves 94.2\% accuracy and 94.3\% F1-score, outperforming rule-based and learning-based baselines by 4-15\% in lane change recognition. These results highlight the benefit of modeling inter-driver heterogeneity and demonstrate the potential of the framework to advance context-aware and human-like autonomous driving in complex traffic environments.
>
---
#### [new 024] Real-Time Indoor Object SLAM with LLM-Enhanced Priors
- **分类: cs.RO**

- **简介: 该论文研究对象级SLAM任务，旨在解决因观测稀疏导致的优化约束不足问题。提出利用大语言模型提供物体几何先验（尺寸与朝向），增强SLAM系统的鲁棒性与精度，并在真实场景中实现实时性能。**

- **链接: [http://arxiv.org/pdf/2509.21602v1](http://arxiv.org/pdf/2509.21602v1)**

> **作者:** Yang Jiao; Yiding Qiu; Henrik I. Christensen
>
> **摘要:** Object-level Simultaneous Localization and Mapping (SLAM), which incorporates semantic information for high-level scene understanding, faces challenges of under-constrained optimization due to sparse observations. Prior work has introduced additional constraints using commonsense knowledge, but obtaining such priors has traditionally been labor-intensive and lacks generalizability across diverse object categories. We address this limitation by leveraging large language models (LLMs) to provide commonsense knowledge of object geometric attributes, specifically size and orientation, as prior factors in a graph-based SLAM framework. These priors are particularly beneficial during the initial phase when object observations are limited. We implement a complete pipeline integrating these priors, achieving robust data association on sparse object-level features and enabling real-time object SLAM. Our system, evaluated on the TUM RGB-D and 3RScan datasets, improves mapping accuracy by 36.8\% over the latest baseline. Additionally, we present real-world experiments in the supplementary video, demonstrating its real-time performance.
>
---
#### [new 025] WAVE: Worm Gear-based Adaptive Variable Elasticity for Decoupling Actuators from External Forces
- **分类: cs.RO**

- **简介: 该论文提出WAVE，一种基于蜗轮的可变刚度执行器，旨在解决机械臂在接触任务中安全性和适应性问题。通过解耦电机与外力，实现精准力控与刚度调节，并验证其模型和应用效果。**

- **链接: [http://arxiv.org/pdf/2509.21878v1](http://arxiv.org/pdf/2509.21878v1)**

> **作者:** Moses Gladson Selvamuthu; Tomoya Takahashi; Riichiro Tadakuma; Kazutoshi Tanaka
>
> **摘要:** Robotic manipulators capable of regulating both compliance and stiffness offer enhanced operational safety and versatility. Here, we introduce Worm Gear-based Adaptive Variable Elasticity (WAVE), a variable stiffness actuator (VSA) that integrates a non-backdrivable worm gear. By decoupling the driving motor from external forces using this gear, WAVE enables precise force transmission to the joint, while absorbing positional discrepancies through compliance. WAVE is protected from excessive loads by converting impact forces into elastic energy stored in a spring. In addition, the actuator achieves continuous joint stiffness modulation by changing the spring's precompression length. We demonstrate these capabilities, experimentally validate the proposed stiffness model, show that motor loads approach zero at rest--even under external loading--and present applications using a manipulator with WAVE. This outcome showcases the successful decoupling of external forces. The protective attributes of this actuator allow for extended operation in contact-intensive tasks, and for robust robotic applications in challenging environments.
>
---
#### [new 026] DHAGrasp: Synthesizing Affordance-Aware Dual-Hand Grasps with Text Instructions
- **分类: cs.RO**

- **简介: 该论文聚焦于双手机器人抓取任务，旨在解决数据稀缺导致的语义感知不足问题。提出SymOpt管道生成大规模双手抓取数据集，并设计文本引导的DHAGrasp模型，实现对未见过物体的语义一致抓取生成。**

- **链接: [http://arxiv.org/pdf/2509.22175v1](http://arxiv.org/pdf/2509.22175v1)**

> **作者:** Quanzhou Li; Zhonghua Wu; Jingbo Wang; Chen Change Loy; Bo Dai
>
> **摘要:** Learning to generate dual-hand grasps that respect object semantics is essential for robust hand-object interaction but remains largely underexplored due to dataset scarcity. Existing grasp datasets predominantly focus on single-hand interactions and contain only limited semantic part annotations. To address these challenges, we introduce a pipeline, SymOpt, that constructs a large-scale dual-hand grasp dataset by leveraging existing single-hand datasets and exploiting object and hand symmetries. Building on this, we propose a text-guided dual-hand grasp generator, DHAGrasp, that synthesizes Dual-Hand Affordance-aware Grasps for unseen objects. Our approach incorporates a novel dual-hand affordance representation and follows a two-stage design, which enables effective learning from a small set of segmented training objects while scaling to a much larger pool of unsegmented data. Extensive experiments demonstrate that our method produces diverse and semantically consistent grasps, outperforming strong baselines in both grasp quality and generalization to unseen objects. The project page is at https://quanzhou-li.github.io/DHAGrasp/.
>
---
#### [new 027] Uncertainty-Aware Multi-Robot Task Allocation With Strongly Coupled Inter-Robot Rewards
- **分类: cs.RO**

- **简介: 该论文研究多机器人任务分配问题，针对任务需求不确定的环境，提出一种考虑机器人间强耦合奖励的市场机制算法，通过概率模型实现互补技能分配，有效降低任务失败率。**

- **链接: [http://arxiv.org/pdf/2509.22469v1](http://arxiv.org/pdf/2509.22469v1)**

> **作者:** Ben Rossano; Jaein Lim; Jonathan P. How
>
> **备注:** 8 pages
>
> **摘要:** This paper proposes a task allocation algorithm for teams of heterogeneous robots in environments with uncertain task requirements. We model these requirements as probability distributions over capabilities and use this model to allocate tasks such that robots with complementary skills naturally position near uncertain tasks, proactively mitigating task failures without wasting resources. We introduce a market-based approach that optimizes the joint team objective while explicitly capturing coupled rewards between robots, offering a polynomial-time solution in decentralized settings with strict communication assumptions. Comparative experiments against benchmark algorithms demonstrate the effectiveness of our approach and highlight the challenges of incorporating coupled rewards in a decentralized formulation.
>
---
#### [new 028] Effect of Gait Design on Proprioceptive Sensing of Terrain Properties in a Quadrupedal Robot
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究四足机器人步态设计对地形感知的影响，旨在解决机器人在松软地表移动时如何准确感知地形性质的问题。通过对比两种步态（Crawl N' Sense和Trot-Walk），评估其在测量地表强度与纹理上的表现，为“运动中感知”提供设计依据。**

- **链接: [http://arxiv.org/pdf/2509.22065v1](http://arxiv.org/pdf/2509.22065v1)**

> **作者:** Ethan Fulcher; J. Diego Caporale; Yifeng Zhang; John Ruck; Feifei Qian
>
> **备注:** 7+1 pages, 5 figures, ICRA Submission This work has been submitted to the IEEE for possible publication
>
> **摘要:** In-situ robotic exploration is an important tool for advancing knowledge of geological processes that describe the Earth and other Planetary bodies. To inform and enhance operations for these roving laboratories, it is imperative to understand the terramechanical properties of their environments, especially for traversing on loose, deformable substrates. Recent research suggested that legged robots with direct-drive and low-gear ratio actuators can sensitively detect external forces, and therefore possess the potential to measure terrain properties with their legs during locomotion, providing unprecedented sampling speed and density while accessing terrains previously too risky to sample. This paper explores these ideas by investigating the impact of gait on proprioceptive terrain sensing accuracy, particularly comparing a sensing-oriented gait, Crawl N' Sense, with a locomotion-oriented gait, Trot-Walk. Each gait's ability to measure the strength and texture of deformable substrate is quantified as the robot locomotes over a laboratory transect consisting of a rigid surface, loose sand, and loose sand with synthetic surface crusts. Our results suggest that with both the sensing-oriented crawling gait and locomotion-oriented trot gait, the robot can measure a consistent difference in the strength (in terms of penetration resistance) between the low- and high-resistance substrates; however, the locomotion-oriented trot gait contains larger magnitude and variance in measurements. Furthermore, the slower crawl gait can detect brittle ruptures of the surface crusts with significantly higher accuracy than the faster trot gait. Our results offer new insights that inform legged robot "sensing during locomotion" gait design and planning for scouting the terrain and producing scientific measurements on other worlds to advance our understanding of their geology and formation.
>
---
#### [new 029] Developing Vision-Language-Action Model from Egocentric Videos
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究视觉-语言-动作模型（VLA）的训练，旨在解决如何直接从第一视角视频中学习物体操作的问题。提出EgoScaler框架，无需辅助标注即可提取6DoF轨迹，并构建大规模预训练数据集，实验表明其显著提升任务成功率，具备可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.21986v1](http://arxiv.org/pdf/2509.21986v1)**

> **作者:** Tomoya Yoshida; Shuhei Kurita; Taichi Nishimura; Shinsuke Mori
>
> **摘要:** Egocentric videos capture how humans manipulate objects and tools, providing diverse motion cues for learning object manipulation. Unlike the costly, expert-driven manual teleoperation commonly used in training Vision-Language-Action models (VLAs), egocentric videos offer a scalable alternative. However, prior studies that leverage such videos for training robot policies typically rely on auxiliary annotations, such as detailed hand-pose recordings. Consequently, it remains unclear whether VLAs can be trained directly from raw egocentric videos. In this work, we address this challenge by leveraging EgoScaler, a framework that extracts 6DoF object manipulation trajectories from egocentric videos without requiring auxiliary recordings. We apply EgoScaler to four large-scale egocentric video datasets and automatically refine noisy or incomplete trajectories, thereby constructing a new large-scale dataset for VLA pre-training. Our experiments with a state-of-the-art $\pi_0$ architecture in both simulated and real-robot environments yield three key findings: (i) pre-training on our dataset improves task success rates by over 20\% compared to training from scratch, (ii) the performance is competitive with that achieved using real-robot datasets, and (iii) combining our dataset with real-robot data yields further improvements. These results demonstrate that egocentric videos constitute a promising and scalable resource for advancing VLA research.
>
---
#### [new 030] RoboView-Bias: Benchmarking Visual Bias in Embodied Agents for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboView-Bias，首个系统量化机器人操作中视觉偏见的基准。针对现有基准缺乏对视觉偏见的系统评估问题，构建了2,127个任务实例，并验证了视觉因素对决策的影响，提出了缓解策略。**

- **链接: [http://arxiv.org/pdf/2509.22356v1](http://arxiv.org/pdf/2509.22356v1)**

> **作者:** Enguang Liu; Siyuan Liang; Liming Lu; Xiyu Zeng; Xiaochun Cao; Aishan Liu; Shuchao Pang
>
> **摘要:** The safety and reliability of embodied agents rely on accurate and unbiased visual perception. However, existing benchmarks mainly emphasize generalization and robustness under perturbations, while systematic quantification of visual bias remains scarce. This gap limits a deeper understanding of how perception influences decision-making stability. To address this issue, we propose RoboView-Bias, the first benchmark specifically designed to systematically quantify visual bias in robotic manipulation, following a principle of factor isolation. Leveraging a structured variant-generation framework and a perceptual-fairness validation protocol, we create 2,127 task instances that enable robust measurement of biases induced by individual visual factors and their interactions. Using this benchmark, we systematically evaluate three representative embodied agents across two prevailing paradigms and report three key findings: (i) all agents exhibit significant visual biases, with camera viewpoint being the most critical factor; (ii) agents achieve their highest success rates on highly saturated colors, indicating inherited visual preferences from underlying VLMs; and (iii) visual biases show strong, asymmetric coupling, with viewpoint strongly amplifying color-related bias. Finally, we demonstrate that a mitigation strategy based on a semantic grounding layer substantially reduces visual bias by approximately 54.5\% on MOKA. Our results highlight that systematic analysis of visual bias is a prerequisite for developing safe and reliable general-purpose embodied agents.
>
---
#### [new 031] Action-aware Dynamic Pruning for Efficient Vision-Language-Action Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人操作中的推理效率问题，提出ADP方法。通过结合文本驱动的token选择与动作感知的轨迹门控机制，动态剪枝冗余视觉token，从而降低计算成本并提升推理速度，同时保持较高成功率。**

- **链接: [http://arxiv.org/pdf/2509.22093v1](http://arxiv.org/pdf/2509.22093v1)**

> **作者:** Xiaohuan Pei; Yuxing Chen; Siyu Xu; Yunke Wang; Yuheng Shi; Chang Xu
>
> **摘要:** Robotic manipulation with Vision-Language-Action models requires efficient inference over long-horizon multi-modal context, where attention to dense visual tokens dominates computational cost. Existing methods optimize inference speed by reducing visual redundancy within VLA models, but they overlook the varying redundancy across robotic manipulation stages. We observe that the visual token redundancy is higher in coarse manipulation phase than in fine-grained operations, and is strongly correlated with the action dynamic. Motivated by this observation, we propose \textbf{A}ction-aware \textbf{D}ynamic \textbf{P}runing (\textbf{ADP}), a multi-modal pruning framework that integrates text-driven token selection with action-aware trajectory gating. Our method introduces a gating mechanism that conditions the pruning signal on recent action trajectories, using past motion windows to adaptively adjust token retention ratios in accordance with dynamics, thereby balancing computational efficiency and perceptual precision across different manipulation stages. Extensive experiments on the LIBERO suites and diverse real-world scenarios demonstrate that our method significantly reduces FLOPs and action inference latency (\textit{e.g.} $1.35 \times$ speed up on OpenVLA-OFT) while maintaining competitive success rates (\textit{e.g.} 25.8\% improvements with OpenVLA) compared to baselines, thereby providing a simple plug-in path to efficient robot policies that advances the efficiency and performance frontier of robotic manipulation. Our project website is: \href{https://vla-adp.github.io/}{ADP.com}.
>
---
#### [new 032] FlowDrive: moderated flow matching with data balancing for trajectory planning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对轨迹规划任务中数据分布不均导致模型性能下降的问题，提出FlowDrive方法。通过轨迹模式重加权平衡数据，并结合改进的流匹配和引导策略，提升了轨迹多样性与场景一致性，在多个基准上取得最优效果。**

- **链接: [http://arxiv.org/pdf/2509.21961v1](http://arxiv.org/pdf/2509.21961v1)**

> **作者:** Lingguang Wang; Ömer Şahin Taş; Marlon Steiner; Christoph Stiller
>
> **摘要:** Learning-based planners are sensitive to the long-tailed distribution of driving data. Common maneuvers dominate datasets, while dangerous or rare scenarios are sparse. This imbalance can bias models toward the frequent cases and degrade performance on critical scenarios. To tackle this problem, we compare balancing strategies for sampling training data and find reweighting by trajectory pattern an effective approach. We then present FlowDrive, a flow-matching trajectory planner that learns a conditional rectified flow to map noise directly to trajectory distributions with few flow-matching steps. We further introduce moderated, in-the-loop guidance that injects small perturbation between flow steps to systematically increase trajectory diversity while remaining scene-consistent. On nuPlan and the interaction-focused interPlan benchmarks, FlowDrive achieves state-of-the-art results among learning-based planners and approaches methods with rule-based refinements. After adding moderated guidance and light post-processing (FlowDrive*), it achieves overall state-of-the-art performance across nearly all benchmark splits.
>
---
#### [new 033] VLA-Reasoner: Empowering Vision-Language-Action Models with Reasoning via Online Monte Carlo Tree Search
- **分类: cs.RO**

- **简介: 该论文提出VLA-Reasoner，通过在线蒙特卡洛树搜索增强视觉-语言-动作模型的推理能力，解决长时序任务中预测偏差问题，提升机器人操作性能。**

- **链接: [http://arxiv.org/pdf/2509.22643v1](http://arxiv.org/pdf/2509.22643v1)**

> **作者:** Wenkai Guo; Guanxing Lu; Haoyuan Deng; Zhenyu Wu; Yansong Tang; Ziwei Wang
>
> **备注:** 9 pages
>
> **摘要:** Vision-Language-Action models (VLAs) achieve strong performance in general robotic manipulation tasks by scaling imitation learning. However, existing VLAs are limited to predicting short-sighted next-action, which struggle with long-horizon trajectory tasks due to incremental deviations. To address this problem, we propose a plug-in framework named VLA-Reasoner that effectively empowers off-the-shelf VLAs with the capability of foreseeing future states via test-time scaling. Specifically, VLA-Reasoner samples and rolls out possible action trajectories where involved actions are rationales to generate future states via a world model, which enables VLA-Reasoner to foresee and reason potential outcomes and search for the optimal actions. We further leverage Monte Carlo Tree Search (MCTS) to improve search efficiency in large action spaces, where stepwise VLA predictions seed the root. Meanwhile, we introduce a confidence sampling mechanism based on Kernel Density Estimation (KDE), to enable efficient exploration in MCTS without redundant VLA queries. We evaluate intermediate states in MCTS via an offline reward shaping strategy, to score predicted futures and correct deviations with long-term feedback. We conducted extensive experiments in both simulators and the real world, demonstrating that our proposed VLA-Reasoner achieves significant improvements over the state-of-the-art VLAs. Our method highlights a potential pathway toward scalable test-time computation of robotic manipulation.
>
---
#### [new 034] Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting
- **分类: cs.RO**

- **简介: 该论文研究视觉-语言-动作模型（VLA）的训练，旨在解决微调过程中VLM性能退化问题。提出VLM2VLA方法，通过语言表示动作，减少数据分布差异，使用LoRA微调，保留VLM核心能力，实现零样本泛化和多语言指令理解。**

- **链接: [http://arxiv.org/pdf/2509.22195v1](http://arxiv.org/pdf/2509.22195v1)**

> **作者:** Asher J. Hancock; Xindi Wu; Lihan Zha; Olga Russakovsky; Anirudha Majumdar
>
> **摘要:** Fine-tuning vision-language models (VLMs) on robot teleoperation data to create vision-language-action (VLA) models is a promising paradigm for training generalist policies, but it suffers from a fundamental tradeoff: learning to produce actions often diminishes the VLM's foundational reasoning and multimodal understanding, hindering generalization to novel scenarios, instruction following, and semantic understanding. We argue that this catastrophic forgetting is due to a distribution mismatch between the VLM's internet-scale pretraining corpus and the robotics fine-tuning data. Inspired by this observation, we introduce VLM2VLA: a VLA training paradigm that first resolves this mismatch at the data level by representing low-level actions with natural language. This alignment makes it possible to train VLAs solely with Low-Rank Adaptation (LoRA), thereby minimally modifying the VLM backbone and averting catastrophic forgetting. As a result, the VLM can be fine-tuned on robot teleoperation data without fundamentally altering the underlying architecture and without expensive co-training on internet-scale VLM datasets. Through extensive Visual Question Answering (VQA) studies and over 800 real-world robotics experiments, we demonstrate that VLM2VLA preserves the VLM's core capabilities, enabling zero-shot generalization to novel tasks that require open-world semantic reasoning and multilingual instruction following.
>
---
#### [new 035] Developing a Mono-Actuated Compliant GeoGami Robot
- **分类: cs.RO**

- **简介: 该论文提出一种单驱动软硬结合的GeoGami机器人，利用折纸结构实现形状变换与运动。针对折纸高自由度需多驱动的问题，通过表面柔顺性提升重复性，并设计几何柔顺骨架与齿轮箱机制，验证其变形与滚动能力。**

- **链接: [http://arxiv.org/pdf/2509.21445v1](http://arxiv.org/pdf/2509.21445v1)**

> **作者:** Archie Webster; Lee Skull; Seyed Amir Tafrishi
>
> **备注:** 8 pages, 12 figures, under-review
>
> **摘要:** This paper presents the design of a new soft-rigid robotic platform, "GeoGami". We leverage origami surface capabilities to achieve shape contraction and to support locomotion with underactuated forms. A key challenge is that origami surfaces have high degrees of freedom and typically require many actuators; we address repeatability by integrating surface compliance. We propose a mono-actuated GeoGami mobile platform that combines origami surface compliance with a geometric compliant skeleton, enabling the robot to transform and locomote using a single actuator. We demonstrate the robot, develop a stiffness model, and describe the central gearbox mechanism. We also analyze alternative cable-driven actuation methods for the skeleton to enable surface transformation. Finally, we evaluate the GeoGami platform for capabilities, including shape transformation and rolling. This platform opens new capabilities for robots that change shape to access different environments and that use shape transformation for locomotion.
>
---
#### [new 036] Hybrid Diffusion for Simultaneous Symbolic and Continuous Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出一种混合扩散方法，用于同时生成符号计划和连续轨迹，解决长周期任务中决策复杂的问题。**

- **链接: [http://arxiv.org/pdf/2509.21983v1](http://arxiv.org/pdf/2509.21983v1)**

> **作者:** Sigmund Hennum Høeg; Aksel Vaaler; Chaoqi Liu; Olav Egeland; Yilun Du
>
> **备注:** 10 pages, 11 figures. This work has been submitted to the IEEE for possible publication. See https://sigmundhh.com/hybrid_diffusion/ for the project website
>
> **摘要:** Constructing robots to accomplish long-horizon tasks is a long-standing challenge within artificial intelligence. Approaches using generative methods, particularly Diffusion Models, have gained attention due to their ability to model continuous robotic trajectories for planning and control. However, we show that these models struggle with long-horizon tasks that involve complex decision-making and, in general, are prone to confusing different modes of behavior, leading to failure. To remedy this, we propose to augment continuous trajectory generation by simultaneously generating a high-level symbolic plan. We show that this requires a novel mix of discrete variable diffusion and continuous diffusion, which dramatically outperforms the baselines. In addition, we illustrate how this hybrid diffusion process enables flexible trajectory synthesis, allowing us to condition synthesized actions on partial and complete symbolic conditions.
>
---
#### [new 037] PL-VIWO2: A Lightweight, Fast and Robust Visual-Inertial-Wheel Odometry Using Points and Lines
- **分类: cs.RO**

- **简介: 该论文提出PL-VIWO2，一种轻量、快速且鲁棒的视觉-惯性-轮速里程计系统，用于自动驾驶。针对复杂户外环境中视觉里程计性能下降的问题，结合IMU、轮编码器和相机，引入线特征处理、SE(2)约束预积分和运动一致性检查，提升长期定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.21563v1](http://arxiv.org/pdf/2509.21563v1)**

> **作者:** Zhixin Zhang; Liang Zhao; Pawel Ladosz
>
> **备注:** 16 pages
>
> **摘要:** Vision-based odometry has been widely adopted in autonomous driving owing to its low cost and lightweight setup; however, its performance often degrades in complex outdoor urban environments. To address these challenges, we propose PL-VIWO2, a filter-based visual-inertial-wheel odometry system that integrates an IMU, wheel encoder, and camera (supporting both monocular and stereo) for long-term robust state estimation. The main contributions are: (i) a novel line feature processing framework that exploits the geometric relationship between 2D feature points and lines, enabling fast and robust line tracking and triangulation while ensuring real-time performance; (ii) an SE(2)-constrained SE(3) wheel pre-integration method that leverages the planar motion characteristics of ground vehicles for accurate wheel updates; and (iii) an efficient motion consistency check (MCC) that filters out dynamic features by jointly using IMU and wheel measurements. Extensive experiments on Monte Carlo simulations and public autonomous driving datasets demonstrate that PL-VIWO2 outperforms state-of-the-art methods in terms of accuracy, efficiency, and robustness.
>
---
#### [new 038] An Ontology for Unified Modeling of Tasks, Actions, Environments, and Capabilities in Personal Service Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出OntoBOT本体，用于统一建模服务机器人任务、动作、环境与能力，解决现有方案平台耦合性强、缺乏通用性的问题，支持上下文感知推理与知识共享。**

- **链接: [http://arxiv.org/pdf/2509.22434v1](http://arxiv.org/pdf/2509.22434v1)**

> **作者:** Margherita Martorana; Francesca Urgese; Ilaria Tiddi; Stefan Schlobach
>
> **摘要:** Personal service robots are increasingly used in domestic settings to assist older adults and people requiring support. Effective operation involves not only physical interaction but also the ability to interpret dynamic environments, understand tasks, and choose appropriate actions based on context. This requires integrating both hardware components (e.g. sensors, actuators) and software systems capable of reasoning about tasks, environments, and robot capabilities. Frameworks such as the Robot Operating System (ROS) provide open-source tools that help connect low-level hardware with higher-level functionalities. However, real-world deployments remain tightly coupled to specific platforms. As a result, solutions are often isolated and hard-coded, limiting interoperability, reusability, and knowledge sharing. Ontologies and knowledge graphs offer a structured way to represent tasks, environments, and robot capabilities. Existing ontologies, such as the Socio-physical Model of Activities (SOMA) and the Descriptive Ontology for Linguistic and Cognitive Engineering (DOLCE), provide models for activities, spatial relationships, and reasoning structures. However, they often focus on specific domains and do not fully capture the connection between environment, action, robot capabilities, and system-level integration. In this work, we propose the Ontology for roBOts and acTions (OntoBOT), which extends existing ontologies to provide a unified representation of tasks, actions, environments, and capabilities. Our contributions are twofold: (1) we unify these aspects into a cohesive ontology to support formal reasoning about task execution, and (2) we demonstrate its generalizability by evaluating competency questions across four embodied agents - TIAGo, HSR, UR3, and Stretch - showing how OntoBOT enables context-aware reasoning, task-oriented execution, and knowledge sharing in service robotics.
>
---
#### [new 039] DroneFL: Federated Learning for Multi-UAV Visual Target Tracking
- **分类: cs.RO**

- **简介: 该论文提出DroneFL，首个面向多无人机目标跟踪的联邦学习框架。针对计算资源受限、数据异质性及轨迹预测耦合问题，设计轻量模型与位置无关架构，实现实时高效的目标跟踪，显著降低预测误差和跟踪距离。**

- **链接: [http://arxiv.org/pdf/2509.21523v1](http://arxiv.org/pdf/2509.21523v1)**

> **作者:** Xiaofan Yu; Yuwei Wu; Katherine Mao; Ye Tian; Vijay Kumar; Tajana Rosing
>
> **摘要:** Multi-robot target tracking is a fundamental problem that requires coordinated monitoring of dynamic entities in applications such as precision agriculture, environmental monitoring, disaster response, and security surveillance. While Federated Learning (FL) has the potential to enhance learning across multiple robots without centralized data aggregation, its use in multi-Unmanned Aerial Vehicle (UAV) target tracking remains largely underexplored. Key challenges include limited onboard computational resources, significant data heterogeneity in FL due to varying targets and the fields of view, and the need for tight coupling between trajectory prediction and multi-robot planning. In this paper, we introduce DroneFL, the first federated learning framework specifically designed for efficient multi-UAV target tracking. We design a lightweight local model to predict target trajectories from sensor inputs, using a frozen YOLO backbone and a shallow transformer for efficient onboard training. The updated models are periodically aggregated in the cloud for global knowledge sharing. To alleviate the data heterogeneity that hinders FL convergence, DroneFL introduces a position-invariant model architecture with altitude-based adaptive instance normalization. Finally, we fuse predictions from multiple UAVs in the cloud and generate optimal trajectories that balance target prediction accuracy and overall tracking performance. Our results show that DroneFL reduces prediction error by 6%-83% and tracking distance by 0.4%-4.6% compared to a distributed non-FL framework. In terms of efficiency, DroneFL runs in real time on a Raspberry Pi 5 and has on average just 1.56 KBps data rate to the cloud.
>
---
#### [new 040] From Watch to Imagine: Steering Long-horizon Manipulation via Human Demonstration and Future Envisionment
- **分类: cs.RO**

- **简介: 该论文研究零样本长时序机器人操作任务，旨在解决从静态视觉输入中分解高层指令的难题。提出Super-Mimic框架，通过解析人类演示视频生成语言子任务，并结合动态预测模型，显著提升了零样本操作性能。**

- **链接: [http://arxiv.org/pdf/2509.22205v1](http://arxiv.org/pdf/2509.22205v1)**

> **作者:** Ke Ye; Jiaming Zhou; Yuanfeng Qiu; Jiayi Liu; Shihui Zhou; Kun-Yu Lin; Junwei Liang
>
> **摘要:** Generalizing to long-horizon manipulation tasks in a zero-shot setting remains a central challenge in robotics. Current multimodal foundation based approaches, despite their capabilities, typically fail to decompose high-level commands into executable action sequences from static visual input alone. To address this challenge, we introduce Super-Mimic, a hierarchical framework that enables zero-shot robotic imitation by directly inferring procedural intent from unscripted human demonstration videos. Our framework is composed of two sequential modules. First, a Human Intent Translator (HIT) parses the input video using multimodal reasoning to produce a sequence of language-grounded subtasks. These subtasks then condition a Future Dynamics Predictor (FDP), which employs a generative model that synthesizes a physically plausible video rollout for each step. The resulting visual trajectories are dynamics-aware, explicitly modeling crucial object interactions and contact points to guide the low-level controller. We validate this approach through extensive experiments on a suite of long-horizon manipulation tasks, where Super-Mimic significantly outperforms state-of-the-art zero-shot methods by over 20\%. These results establish that coupling video-driven intent parsing with prospective dynamics modeling is a highly effective strategy for developing general-purpose robotic systems.
>
---
#### [new 041] MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出MimicDreamer框架，旨在解决人类演示视频与机器人执行间的领域差距问题。通过视觉对齐、视角稳定和动作映射，将低成本的人类演示转化为机器人可用的监督信号，用于高效训练VLA模型，提升其在现实任务中的表现。**

- **链接: [http://arxiv.org/pdf/2509.22199v1](http://arxiv.org/pdf/2509.22199v1)**

> **作者:** Haoyun Li; Ivan Zhang; Runqi Ouyang; Xiaofeng Wang; Zheng Zhu; Zhiqin Yang; Zhentao Zhang; Boyuan Wang; Chaojun Ni; Wenkang Qin; Xinze Chen; Yun Ye; Guan Huang; Zhenbo Song; Xingang Wang
>
> **摘要:** Vision Language Action (VLA) models derive their generalization capability from diverse training data, yet collecting embodied robot interaction data remains prohibitively expensive. In contrast, human demonstration videos are far more scalable and cost-efficient to collect, and recent studies confirm their effectiveness in training VLA models. However, a significant domain gap persists between human videos and robot-executed videos, including unstable camera viewpoints, visual discrepancies between human hands and robotic arms, and differences in motion dynamics. To bridge this gap, we propose MimicDreamer, a framework that turns fast, low-cost human demonstrations into robot-usable supervision by jointly aligning vision, viewpoint, and actions to directly support policy training. For visual alignment, we propose H2R Aligner, a video diffusion model that generates high-fidelity robot demonstration videos by transferring motion from human manipulation footage. For viewpoint stabilization, EgoStabilizer is proposed, which canonicalizes egocentric videos via homography and inpaints occlusions and distortions caused by warping. For action alignment, we map human hand trajectories to the robot frame and apply a constrained inverse kinematics solver to produce feasible, low-jitter joint commands with accurate pose tracking. Empirically, VLA models trained purely on our synthesized human-to-robot videos achieve few-shot execution on real robots. Moreover, scaling training with human data significantly boosts performance compared to models trained solely on real robot data; our approach improves the average success rate by 14.7\% across six representative manipulation tasks.
>
---
#### [new 042] UnderwaterVLA: Dual-brain Vision-Language-Action architecture for Autonomous Underwater Navigation
- **分类: cs.RO**

- **简介: 该论文提出UnderwaterVLA，一种用于水下自主导航的双脑视觉-语言-动作架构。针对水下通信受限、感知退化等问题，引入了任务推理与控制解耦、VLA模型应用及流体感知MPC控制三项创新，提升了导航鲁棒性和任务完成率。**

- **链接: [http://arxiv.org/pdf/2509.22441v1](http://arxiv.org/pdf/2509.22441v1)**

> **作者:** Zhangyuan Wang; Yunpeng Zhu; Yuqi Yan; Xiaoyuan Tian; Xinhao Shao; Meixuan Li; Weikun Li; Guangsheng Su; Weicheng Cui; Dixia Fan
>
> **备注:** This paper introduces the first VLA framework for AUVs, featuring a dual-brain architecture and zero-data MPC for real-world underwater navigation
>
> **摘要:** This paper presents UnderwaterVLA, a novel framework for autonomous underwater navigation that integrates multimodal foundation models with embodied intelligence systems. Underwater operations remain difficult due to hydrodynamic disturbances, limited communication bandwidth, and degraded sensing in turbid waters. To address these challenges, we introduce three innovations. First, a dual-brain architecture decouples high-level mission reasoning from low-level reactive control, enabling robust operation under communication and computational constraints. Second, we apply Vision-Language-Action(VLA) models to underwater robotics for the first time, incorporating structured chain-of-thought reasoning for interpretable decision-making. Third, a hydrodynamics-informed Model Predictive Control(MPC) scheme compensates for fluid effects in real time without costly task-specific training. Experimental results in field tests show that UnderwaterVLA reduces navigation errors in degraded visual conditions while maintaining higher task completion by 19% to 27% over baseline. By minimizing reliance on underwater-specific training data and improving adaptability across environments, UnderwaterVLA provides a scalable and cost-effective path toward the next generation of intelligent AUVs.
>
---
#### [new 043] Generating Stable Placements via Physics-guided Diffusion Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对机器人操作中多物体场景下稳定放置的问题，提出一种基于物理引导的扩散模型方法。通过结合几何先验与稳定性损失，直接在采样过程中生成稳定的放置方案，无需额外训练，提升了放置的鲁棒性并减少了运行时间。**

- **链接: [http://arxiv.org/pdf/2509.21664v1](http://arxiv.org/pdf/2509.21664v1)**

> **作者:** Philippe Nadeau; Miguel Rogel; Ivan Bilić; Ivan Petrović; Jonathan Kelly
>
> **备注:** Submitted to the IEEE International Conference on Robotics and Automation 2026, Vienna, Austria, June 1-5, 2026
>
> **摘要:** Stably placing an object in a multi-object scene is a fundamental challenge in robotic manipulation, as placements must be penetration-free, establish precise surface contact, and result in a force equilibrium. To assess stability, existing methods rely on running a simulation engine or resort to heuristic, appearance-based assessments. In contrast, our approach integrates stability directly into the sampling process of a diffusion model. To this end, we query an offline sampling-based planner to gather multi-modal placement labels and train a diffusion model to generate stable placements. The diffusion model is conditioned on scene and object point clouds, and serves as a geometry-aware prior. We leverage the compositional nature of score-based generative models to combine this learned prior with a stability-aware loss, thereby increasing the likelihood of sampling from regions of high stability. Importantly, this strategy requires no additional re-training or fine-tuning, and can be directly applied to off-the-shelf models. We evaluate our method on four benchmark scenes where stability can be accurately computed. Our physics-guided models achieve placements that are 56% more robust to forceful perturbations while reducing runtime by 47% compared to a state-of-the-art geometric method.
>
---
#### [new 044] Autonomous UAV-Quadruped Docking in Complex Terrains via Active Posture Alignment and Constraint-Aware Control
- **分类: cs.RO**

- **简介: 该论文研究无人机与四足机器人在复杂地形下的自主对接任务，旨在解决四足机器人姿态变化导致对接不稳定的问题。提出HIM-HA模型和三阶段控制策略，实现无GPS环境下的稳定对接，验证了在楼梯和陡坡等场景的有效性。**

- **链接: [http://arxiv.org/pdf/2509.21571v1](http://arxiv.org/pdf/2509.21571v1)**

> **作者:** HaoZhe Xu; Cheng Cheng; HongRui Sang; Zhipeng Wang; Qiyong He; Xiuxian Li; Bin He
>
> **摘要:** Autonomous docking between Unmanned Aerial Vehicles (UAVs) and ground robots is essential for heterogeneous systems, yet most existing approaches target wheeled platforms whose limited mobility constrains exploration in complex terrains. Quadruped robots offer superior adaptability but undergo frequent posture variations, making it difficult to provide a stable landing surface for UAVs. To address these challenges, we propose an autonomous UAV-quadruped docking framework for GPS-denied environments. On the quadruped side, a Hybrid Internal Model with Horizontal Alignment (HIM-HA), learned via deep reinforcement learning, actively stabilizes the torso to provide a level platform. On the UAV side, a three-phase strategy is adopted, consisting of long-range acquisition with a median-filtered YOLOv8 detector, close-range tracking with a constraint-aware controller that integrates a Nonsingular Fast Terminal Sliding Mode Controller (NFTSMC) and a logarithmic Barrier Function (BF) to guarantee finite-time error convergence under field-of-view (FOV) constraints, and terminal descent guided by a Safety Period (SP) mechanism that jointly verifies tracking accuracy and platform stability. The proposed framework is validated in both simulation and real-world scenarios, successfully achieving docking on outdoor staircases higher than 17 cm and rough slopes steeper than 30 degrees. Supplementary materials and videos are available at: https://uav-quadruped-docking.github.io.
>
---
#### [new 045] Learning Multi-Skill Legged Locomotion Using Conditional Adversarial Motion Priors
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决四足机器人多技能学习与平滑切换的问题。提出了基于条件对抗运动先验（CAMP）的框架，通过新技能判别器和奖励设计，实现从专家演示中高效学习多种运动技能并支持主动控制与复用。**

- **链接: [http://arxiv.org/pdf/2509.21810v1](http://arxiv.org/pdf/2509.21810v1)**

> **作者:** Ning Huang; Zhentao Xie; Qinchuan Li
>
> **摘要:** Despite growing interest in developing legged robots that emulate biological locomotion for agile navigation of complex environments, acquiring a diverse repertoire of skills remains a fundamental challenge in robotics. Existing methods can learn motion behaviors from expert data, but they often fail to acquire multiple locomotion skills through a single policy and lack smooth skill transitions. We propose a multi-skill learning framework based on Conditional Adversarial Motion Priors (CAMP), with the aim of enabling quadruped robots to efficiently acquire a diverse set of locomotion skills from expert demonstrations. Precise skill reconstruction is achieved through a novel skill discriminator and skill-conditioned reward design. The overall framework supports the active control and reuse of multiple skills, providing a practical solution for learning generalizable policies in complex environments.
>
---
#### [new 046] EgoDemoGen: Novel Egocentric Demonstration Generation Enables Viewpoint-Robust Manipulation
- **分类: cs.RO**

- **简介: 该论文提出EgoDemoGen框架，用于生成新的视角鲁棒的自我中心演示数据，解决机器人在视角变化下操作性能下降的问题。通过动作重定向和视频合成模型EgoViewTransfer提升策略表现，在仿真和真实环境中均取得显著效果。**

- **链接: [http://arxiv.org/pdf/2509.22578v1](http://arxiv.org/pdf/2509.22578v1)**

> **作者:** Yuan Xu; Jiabing Yang; Xiaofeng Wang; Yixiang Chen; Zheng Zhu; Bowen Fang; Guan Huang; Xinze Chen; Yun Ye; Qiang Zhang; Peiyan Li; Xiangnan Wu; Kai Wang; Bing Zhan; Shuo Lu; Jing Liu; Nianfeng Liu; Yan Huang; Liang Wang
>
> **摘要:** Imitation learning based policies perform well in robotic manipulation, but they often degrade under *egocentric viewpoint shifts* when trained from a single egocentric viewpoint. To address this issue, we present **EgoDemoGen**, a framework that generates *paired* novel egocentric demonstrations by retargeting actions in the novel egocentric frame and synthesizing the corresponding egocentric observation videos with proposed generative video repair model **EgoViewTransfer**, which is conditioned by a novel-viewpoint reprojected scene video and a robot-only video rendered from the retargeted joint actions. EgoViewTransfer is finetuned from a pretrained video generation model using self-supervised double reprojection strategy. We evaluate EgoDemoGen on both simulation (RoboTwin2.0) and real-world robot. After training with a mixture of EgoDemoGen-generated novel egocentric demonstrations and original standard egocentric demonstrations, policy success rate improves **absolutely** by **+17.0%** for standard egocentric viewpoint and by **+17.7%** for novel egocentric viewpoints in simulation. On real-world robot, the **absolute** improvements are **+18.3%** and **+25.8%**. Moreover, performance continues to improve as the proportion of EgoDemoGen-generated demonstrations increases, with diminishing returns. These results demonstrate that EgoDemoGen provides a practical route to egocentric viewpoint-robust robotic manipulation.
>
---
#### [new 047] Plan2Evolve: LLM Self-Evolution for Improved Planning Capability via Automated Domain Generation
- **分类: cs.RO**

- **简介: 该论文提出Plan2Evolve框架，旨在提升大语言模型（LLM）的规划能力。通过自动生成符号规划领域及问题-计划对，并转化为链式推理轨迹，实现无需人工标注的数据增强与模型微调，从而提高规划成功率和泛化性。**

- **链接: [http://arxiv.org/pdf/2509.21543v1](http://arxiv.org/pdf/2509.21543v1)**

> **作者:** Jinbang Huang; Zhiyuan Li; Zhanguang Zhang; Xingyue Quan; Jianye Hao; Yingxue Zhang
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) have recently shown strong potential in robotic task planning, particularly through automatic planning domain generation that integrates symbolic search. Prior approaches, however, have largely treated these domains as search utilities, with limited attention to their potential as scalable sources of reasoning data. At the same time, progress in reasoning LLMs has been driven by chain-of-thought (CoT) supervision, whose application in robotics remains dependent on costly, human-curated datasets. We propose Plan2Evolve, an LLM self-evolving framework in which the base model generates planning domains that serve as engines for producing symbolic problem-plan pairs as reasoning traces. These pairs are then transformed into extended CoT trajectories by the same model through natural-language explanations, thereby explicitly aligning symbolic planning structures with natural language reasoning. The resulting data extend beyond the model's intrinsic planning capacity, enabling model fine-tuning that yields a planning-enhanced LLM with improved planning success, stronger cross-task generalization, and reduced inference costs.
>
---
#### [new 048] ShipwreckFinder: A QGIS Tool for Shipwreck Detection in Multibeam Sonar Data
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出ShipwreckFinder，一个QGIS插件，用于从多波束声呐数据中自动检测沉船。针对传统人工分析耗时且依赖专家的问题，该工具结合深度学习和合成数据生成，实现了高效的沉船分割与定位。**

- **链接: [http://arxiv.org/pdf/2509.21386v1](http://arxiv.org/pdf/2509.21386v1)**

> **作者:** Anja Sheppard; Tyler Smithline; Andrew Scheffer; David Smith; Advaith V. Sethuraman; Ryan Bird; Sabrina Lin; Katherine A. Skinner
>
> **备注:** Accepted to OCEANS 2025 Great Lakes
>
> **摘要:** In this paper, we introduce ShipwreckFinder, an open-source QGIS plugin that detects shipwrecks from multibeam sonar data. Shipwrecks are an important historical marker of maritime history, and can be discovered through manual inspection of bathymetric data. However, this is a time-consuming process and often requires expert analysis. Our proposed tool allows users to automatically preprocess bathymetry data, perform deep learning inference, threshold model outputs, and produce either pixel-wise segmentation masks or bounding boxes of predicted shipwrecks. The backbone of this open-source tool is a deep learning model, which is trained on a variety of shipwreck data from the Great Lakes and the coasts of Ireland. Additionally, we employ synthetic data generation in order to increase the size and diversity of our dataset. We demonstrate superior segmentation performance with our open-source tool and training pipeline as compared to a deep learning-based ArcGIS toolkit and a more classical inverse sinkhole detection method. The open-source tool can be found at https://github.com/umfieldrobotics/ShipwreckFinderQGISPlugin.
>
---
#### [new 049] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MesaTask，一个基于LLM的框架，用于任务驱动的桌面场景生成。针对传统方法在布局合理性与任务对齐上的不足，构建了包含10,700个合成场景的数据集MesaTask-10K，并引入空间推理链以提升生成效果。**

- **链接: [http://arxiv.org/pdf/2509.22281v1](http://arxiv.org/pdf/2509.22281v1)**

> **作者:** Jinkun Hao; Naifu Liang; Zhen Luo; Xudong Xu; Weipeng Zhong; Ran Yi; Yichen Jin; Zhaoyang Lyu; Feng Zheng; Lizhuang Ma; Jiangmiao Pang
>
> **备注:** Accepted by NeurIPS 2025; Project page: https://mesatask.github.io/
>
> **摘要:** The ability of robots to interpret human instructions and execute manipulation tasks necessitates the availability of task-relevant tabletop scenes for training. However, traditional methods for creating these scenes rely on time-consuming manual layout design or purely randomized layouts, which are limited in terms of plausibility or alignment with the tasks. In this paper, we formulate a novel task, namely task-oriented tabletop scene generation, which poses significant challenges due to the substantial gap between high-level task instructions and the tabletop scenes. To support research on such a challenging task, we introduce MesaTask-10K, a large-scale dataset comprising approximately 10,700 synthetic tabletop scenes with manually crafted layouts that ensure realistic layouts and intricate inter-object relations. To bridge the gap between tasks and scenes, we propose a Spatial Reasoning Chain that decomposes the generation process into object inference, spatial interrelation reasoning, and scene graph construction for the final 3D layout. We present MesaTask, an LLM-based framework that utilizes this reasoning chain and is further enhanced with DPO algorithms to generate physically plausible tabletop scenes that align well with given task descriptions. Exhaustive experiments demonstrate the superior performance of MesaTask compared to baselines in generating task-conforming tabletop scenes with realistic layouts. Project page is at https://mesatask.github.io/
>
---
#### [new 050] Human Autonomy and Sense of Agency in Human-Robot Interaction: A Systematic Literature Review
- **分类: cs.HC; cs.RO**

- **简介: 该论文是一篇系统综述，旨在探讨人机交互中如何保障人类自主性与能动感。通过分析22项实证研究，识别出五个影响因素，并指出当前研究的局限性，为设计以人为本的机器人提供理论支持。**

- **链接: [http://arxiv.org/pdf/2509.22271v1](http://arxiv.org/pdf/2509.22271v1)**

> **作者:** Felix Glawe; Tim Schmeckel; Philipp Brauner; Martina Ziefle
>
> **摘要:** Human autonomy and sense of agency are increasingly recognised as critical for user well-being, motivation, and the ethical deployment of robots in human-robot interaction (HRI). Given the rapid development of artificial intelligence, robot capabilities and their potential to function as colleagues and companions are growing. This systematic literature review synthesises 22 empirical studies selected from an initial pool of 728 articles published between 2011 and 2024. Articles were retrieved from major scientific databases and identified based on empirical focus and conceptual relevance, namely, how to preserve and promote human autonomy and sense of agency in HRI. Derived through thematic synthesis, five clusters of potentially influential factors are revealed: robot adaptiveness, communication style, anthropomorphism, presence of a robot and individual differences. Measured through psychometric scales or the intentional binding paradigm, perceptions of autonomy and agency varied across industrial, educational, healthcare, care, and hospitality settings. The review underscores the theoretical differences between both concepts, but their yet entangled use in HRI. Despite increasing interest, the current body of empirical evidence remains limited and fragmented, underscoring the necessity for standardised definitions, more robust operationalisations, and further exploratory and qualitative research. By identifying existing gaps and highlighting emerging trends, this review contributes to the development of human-centered, autonomy-supportive robot design strategies that uphold ethical and psychological principles, ultimately supporting well-being in human-robot interaction.
>
---
#### [new 051] Learning to Ball: Composing Policies for Long-Horizon Basketball Moves
- **分类: cs.GR; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对长时域篮球动作控制任务，研究如何无缝组合不同技能策略的问题。提出了一种新的策略集成框架和高层软路由机制，以应对子任务间状态不明确的挑战，实现根据实时指令完成复杂篮球动作。**

- **链接: [http://arxiv.org/pdf/2509.22442v1](http://arxiv.org/pdf/2509.22442v1)**

> **作者:** Pei Xu; Zhen Wu; Ruocheng Wang; Vishnu Sarukkai; Kayvon Fatahalian; Ioannis Karamouzas; Victor Zordan; C. Karen Liu
>
> **备注:** ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia 2025). Website: http://pei-xu.github.io/basketball. Video: https://youtu.be/2RBFIjjmR2I. Code: https://github.com/xupei0610/basketball
>
> **摘要:** Learning a control policy for a multi-phase, long-horizon task, such as basketball maneuvers, remains challenging for reinforcement learning approaches due to the need for seamless policy composition and transitions between skills. A long-horizon task typically consists of distinct subtasks with well-defined goals, separated by transitional subtasks with unclear goals but critical to the success of the entire task. Existing methods like the mixture of experts and skill chaining struggle with tasks where individual policies do not share significant commonly explored states or lack well-defined initial and terminal states between different phases. In this paper, we introduce a novel policy integration framework to enable the composition of drastically different motor skills in multi-phase long-horizon tasks with ill-defined intermediate states. Based on that, we further introduce a high-level soft router to enable seamless and robust transitions between the subtasks. We evaluate our framework on a set of fundamental basketball skills and challenging transitions. Policies trained by our approach can effectively control the simulated character to interact with the ball and accomplish the long-horizon task specified by real-time user commands, without relying on ball trajectory references.
>
---
#### [new 052] A Multi-Modality Evaluation of the Reality Gap in Autonomous Driving Systems
- **分类: cs.SE; cs.RO**

- **简介: 该论文研究自动驾驶系统的“现实差距”问题，通过多模态测试（SiL、ViL、MR和实车测试），评估感知、执行与行为保真度差异。旨在提升测试结果的可迁移性与系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.22379v1](http://arxiv.org/pdf/2509.22379v1)**

> **作者:** Stefano Carlo Lambertenghi; Mirena Flores Valdez; Andrea Stocco
>
> **备注:** In proceedings of the 40th IEEE/ACM International Conference on Automated Software Engineering (ASE '25)
>
> **摘要:** Simulation-based testing is a cornerstone of Autonomous Driving System (ADS) development, offering safe and scalable evaluation across diverse driving scenarios. However, discrepancies between simulated and real-world behavior, known as the reality gap, challenge the transferability of test results to deployed systems. In this paper, we present a comprehensive empirical study comparing four representative testing modalities: Software-in-the-Loop (SiL), Vehicle-in-the-Loop (ViL), Mixed-Reality (MR), and full real-world testing. Using a small-scale physical vehicle equipped with real sensors (camera and LiDAR) and its digital twin, we implement each setup and evaluate two ADS architectures (modular and end-to-end) across diverse indoor driving scenarios involving real obstacles, road topologies, and indoor environments. We systematically assess the impact of each testing modality along three dimensions of the reality gap: actuation, perception, and behavioral fidelity. Our results show that while SiL and ViL setups simplify critical aspects of real-world dynamics and sensing, MR testing improves perceptual realism without compromising safety or control. Importantly, we identify the conditions under which failures do not transfer across testing modalities and isolate the underlying dimensions of the gap responsible for these discrepancies. Our findings offer actionable insights into the respective strengths and limitations of each modality and outline a path toward more robust and transferable validation of autonomous driving systems.
>
---
#### [new 053] Log2Plan: An Adaptive GUI Automation Framework Integrated with Task Mining Approach
- **分类: cs.AI; cs.HC; cs.MA; cs.RO; 68N19, 68T09; H.5.2; D.2.2**

- **简介: 该论文提出Log2Plan，一个结合任务挖掘的自适应GUI自动化框架。针对现有方法泛化差、延迟高、长流程不连贯的问题，通过双层规划和用户日志分析，实现鲁棒且可复用的GUI任务自动化。**

- **链接: [http://arxiv.org/pdf/2509.22137v1](http://arxiv.org/pdf/2509.22137v1)**

> **作者:** Seoyoung Lee; Seonbin Yoon; Seongbeen Lee; Hyesoo Kim; Joo Yong Sim
>
> **摘要:** GUI task automation streamlines repetitive tasks, but existing LLM or VLM-based planner-executor agents suffer from brittle generalization, high latency, and limited long-horizon coherence. Their reliance on single-shot reasoning or static plans makes them fragile under UI changes or complex tasks. Log2Plan addresses these limitations by combining a structured two-level planning framework with a task mining approach over user behavior logs, enabling robust and adaptable GUI automation. Log2Plan constructs high-level plans by mapping user commands to a structured task dictionary, enabling consistent and generalizable automation. To support personalization and reuse, it employs a task mining approach from user behavior logs that identifies user-specific patterns. These high-level plans are then grounded into low-level action sequences by interpreting real-time GUI context, ensuring robust execution across varying interfaces. We evaluated Log2Plan on 200 real-world tasks, demonstrating significant improvements in task success rate and execution time. Notably, it maintains over 60.0% success rate even on long-horizon task sequences, highlighting its robustness in complex, multi-step workflows.
>
---
#### [new 054] ReLAM: Learning Anticipation Model for Rewarding Visual Robotic Manipulation
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对视觉强化学习中奖励设计的瓶颈问题，提出ReLAM框架。通过从视频演示中自动生成结构化奖励，利用关键点预测子目标，加速复杂操作任务的学习，提升性能。**

- **链接: [http://arxiv.org/pdf/2509.22402v1](http://arxiv.org/pdf/2509.22402v1)**

> **作者:** Nan Tang; Jing-Cheng Pang; Guanlin Li; Chao Qian; Yang Yu
>
> **摘要:** Reward design remains a critical bottleneck in visual reinforcement learning (RL) for robotic manipulation. In simulated environments, rewards are conventionally designed based on the distance to a target position. However, such precise positional information is often unavailable in real-world visual settings due to sensory and perceptual limitations. In this study, we propose a method that implicitly infers spatial distances through keypoints extracted from images. Building on this, we introduce Reward Learning with Anticipation Model (ReLAM), a novel framework that automatically generates dense, structured rewards from action-free video demonstrations. ReLAM first learns an anticipation model that serves as a planner and proposes intermediate keypoint-based subgoals on the optimal path to the final goal, creating a structured learning curriculum directly aligned with the task's geometric objectives. Based on the anticipated subgoals, a continuous reward signal is provided to train a low-level, goal-conditioned policy under the hierarchical reinforcement learning (HRL) framework with provable sub-optimality bound. Extensive experiments on complex, long-horizon manipulation tasks show that ReLAM significantly accelerates learning and achieves superior performance compared to state-of-the-art methods.
>
---
#### [new 055] Residual Vector Quantization For Communication-Efficient Multi-Agent Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究多智能体协作感知任务，旨在解决通信带宽限制问题。提出ReVQom方法，通过残差向量量化压缩特征，实现高效通信，在低带宽下保持感知精度，推动V2X应用落地。**

- **链接: [http://arxiv.org/pdf/2509.21464v1](http://arxiv.org/pdf/2509.21464v1)**

> **作者:** Dereje Shenkut; B. V. K Vijaya Kumar
>
> **备注:** 5 pages
>
> **摘要:** Multi-agent collaborative perception (CP) improves scene understanding by sharing information across connected agents such as autonomous vehicles, unmanned aerial vehicles, and robots. Communication bandwidth, however, constrains scalability. We present ReVQom, a learned feature codec that preserves spatial identity while compressing intermediate features. ReVQom is an end-to-end method that compresses feature dimensions via a simple bottleneck network followed by multi-stage residual vector quantization (RVQ). This allows only per-pixel code indices to be transmitted, reducing payloads from 8192 bits per pixel (bpp) of uncompressed 32-bit float features to 6-30 bpp per agent with minimal accuracy loss. On DAIR-V2X real-world CP dataset, ReVQom achieves 273x compression at 30 bpp to 1365x compression at 6 bpp. At 18 bpp (455x), ReVQom matches or outperforms raw-feature CP, and at 6-12 bpp it enables ultra-low-bandwidth operation with graceful degradation. ReVQom allows efficient and accurate multi-agent collaborative perception with a step toward practical V2X deployment.
>
---
#### [new 056] Object Identification Under Known Dynamics: A PIRNN Approach for UAV Classification
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于无人机分类任务，旨在解决已知动力学条件下的目标识别问题。提出了一种基于物理信息残差神经网络（PIRNN）的框架，结合物理模型与深度学习，实现状态映射和分类，提升了识别准确率并减少了训练时间。**

- **链接: [http://arxiv.org/pdf/2509.21405v1](http://arxiv.org/pdf/2509.21405v1)**

> **作者:** Nyi Nyi Aung; Neil Muralles; Adrian Stein
>
> **备注:** 2025 International Conference on Machine Learning and Applications (ICMLA)
>
> **摘要:** This work addresses object identification under known dynamics in unmanned aerial vehicle applications, where learning and classification are combined through a physics-informed residual neural network. The proposed framework leverages physics-informed learning for state mapping and state-derivative prediction, while a softmax layer enables multi-class confidence estimation. Quadcopter, fixed-wing, and helicopter aerial vehicles are considered as case studies. The results demonstrate high classification accuracy with reduced training time, offering a promising solution for system identification problems in domains where the underlying dynamics are well understood.
>
---
#### [new 057] SGAligner++: Cross-Modal Language-Aided 3D Scene Graph Alignment
- **分类: cs.GR; cs.CV; cs.RO**

- **简介: 该论文提出SGAligner++，用于3D场景图对齐任务。针对现有方法依赖单一模态数据、处理噪声和低重叠场景效果差的问题，设计了一个跨模态语言辅助框架，通过联合嵌入空间提升对齐精度与鲁棒性，在真实数据集上表现优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.20401v1](http://arxiv.org/pdf/2509.20401v1)**

> **作者:** Binod Singh; Sayan Deb Sarkar; Iro Armeni
>
> **摘要:** Aligning 3D scene graphs is a crucial initial step for several applications in robot navigation and embodied perception. Current methods in 3D scene graph alignment often rely on single-modality point cloud data and struggle with incomplete or noisy input. We introduce SGAligner++, a cross-modal, language-aided framework for 3D scene graph alignment. Our method addresses the challenge of aligning partially overlapping scene observations across heterogeneous modalities by learning a unified joint embedding space, enabling accurate alignment even under low-overlap conditions and sensor noise. By employing lightweight unimodal encoders and attention-based fusion, SGAligner++ enhances scene understanding for tasks such as visual localization, 3D reconstruction, and navigation, while ensuring scalability and minimal computational overhead. Extensive evaluations on real-world datasets demonstrate that SGAligner++ outperforms state-of-the-art methods by up to 40% on noisy real-world reconstructions, while enabling cross-modal generalization.
>
---
#### [new 058] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出JanusVLN，针对视觉-语言导航任务，解决现有方法因显式语义记忆导致的空间信息丢失和计算冗余问题。通过构建双隐式神经记忆，分别建模空间几何与视觉语义信息，实现高效增量更新，提升导航成功率。**

- **链接: [http://arxiv.org/pdf/2509.22548v1](http://arxiv.org/pdf/2509.22548v1)**

> **作者:** Shuang Zeng; Dekang Qi; Xinyuan Chang; Feng Xiong; Shichao Xie; Xiaolong Wu; Shiyi Liang; Mu Xu; Xing Wei
>
> **备注:** Project page: https://miv-xjtu.github.io/JanusVLN.github.io/
>
> **摘要:** Vision-and-Language Navigation requires an embodied agent to navigate through unseen environments, guided by natural language instructions and a continuous video stream. Recent advances in VLN have been driven by the powerful semantic understanding of Multimodal Large Language Models. However, these methods typically rely on explicit semantic memory, such as building textual cognitive maps or storing historical visual frames. This type of method suffers from spatial information loss, computational redundancy, and memory bloat, which impede efficient navigation. Inspired by the implicit scene representation in human navigation, analogous to the left brain's semantic understanding and the right brain's spatial cognition, we propose JanusVLN, a novel VLN framework featuring a dual implicit neural memory that models spatial-geometric and visual-semantic memory as separate, compact, and fixed-size neural representations. This framework first extends the MLLM to incorporate 3D prior knowledge from the spatial-geometric encoder, thereby enhancing the spatial reasoning capabilities of models based solely on RGB input. Then, the historical key-value caches from the spatial-geometric and visual-semantic encoders are constructed into a dual implicit memory. By retaining only the KVs of tokens in the initial and sliding window, redundant computation is avoided, enabling efficient incremental updates. Extensive experiments demonstrate that JanusVLN outperforms over 20 recent methods to achieve SOTA performance. For example, the success rate improves by 10.5-35.5 compared to methods using multiple data types as input and by 3.6-10.8 compared to methods using more RGB training data. This indicates that the proposed dual implicit neural memory, as a novel paradigm, explores promising new directions for future VLN research. Ours project page: https://miv-xjtu.github.io/JanusVLN.github.io/.
>
---
#### [new 059] EMMA: Generalizing Real-World Robot Manipulation via Generative Visual Transfer
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出EMMA框架，旨在解决机器人操作中真实数据收集昂贵的问题。通过生成视觉多视角一致的操视频，并结合自适应训练策略AdaMix，提升VLA模型在未见物体和环境中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.22407v1](http://arxiv.org/pdf/2509.22407v1)**

> **作者:** Zhehao Dong; Xiaofeng Wang; Zheng Zhu; Yirui Wang; Yang Wang; Yukun Zhou; Boyuan Wang; Chaojun Ni; Runqi Ouyang; Wenkang Qin; Xinze Chen; Yun Ye; Guan Huang
>
> **摘要:** Vision-language-action (VLA) models increasingly rely on diverse training data to achieve robust generalization. However, collecting large-scale real-world robot manipulation data across varied object appearances and environmental conditions remains prohibitively time-consuming and expensive. To overcome this bottleneck, we propose Embodied Manipulation Media Adaptation (EMMA), a VLA policy enhancement framework that integrates a generative data engine with an effective training pipeline. We introduce DreamTransfer, a diffusion Transformer-based framework for generating multi-view consistent, geometrically grounded embodied manipulation videos. DreamTransfer enables text-controlled visual editing of robot videos, transforming foreground, background, and lighting conditions without compromising 3D structure or geometrical plausibility. Furthermore, we explore hybrid training with real and generated data, and introduce AdaMix, a hard-sample-aware training strategy that dynamically reweights training batches to focus optimization on perceptually or kinematically challenging samples. Extensive experiments show that videos generated by DreamTransfer significantly outperform prior video generation methods in multi-view consistency, geometric fidelity, and text-conditioning accuracy. Crucially, VLAs trained with generated data enable robots to generalize to unseen object categories and novel visual domains using only demonstrations from a single appearance. In real-world robotic manipulation tasks with zero-shot visual domains, our approach achieves over a 200% relative performance gain compared to training on real data alone, and further improves by 13% with AdaMix, demonstrating its effectiveness in boosting policy generalization.
>
---
#### [new 060] Trust and Human Autonomy after Cobot Failures: Communication is Key for Industry 5.0
- **分类: cs.HC; cs.RO**

- **简介: 该论文研究工业5.0中协作机器人（cobot）故障对工人信任与自主性的影响。通过VR实验发现，故障会降低信任和自主性，严重故障更影响信任。透明沟通可部分恢复两者。任务是探索故障后的恢复机制，解决人机协作中的信任与自主问题。**

- **链接: [http://arxiv.org/pdf/2509.22298v1](http://arxiv.org/pdf/2509.22298v1)**

> **作者:** Felix Glawe; Laura Kremer; Luisa Vervier; Philipp Brauner; Martina Ziefle
>
> **摘要:** Collaborative robots (cobots) are a core technology of Industry 4.0. Industry 4.0 uses cyber-physical systems, IoT and smart automation to improve efficiency and data-driven decision-making. Cobots, as cyber-physical systems, enable the introduction of lightweight automation to smaller companies through their flexibility, low cost and ability to work alongside humans, while keeping humans and their skills in the loop. Industry 5.0, the evolution of Industry 4.0, places the worker at the centre of its principles: The physical and mental well-being of the worker is the main goal of new technology design, not just productivity, efficiency and safety standards. Within this concept, human trust in cobots and human autonomy are important. While trust is essential for effective and smooth interaction, the workers' perception of autonomy is key to intrinsic motivation and overall well-being. As failures are an inevitable part of technological systems, this study aims to answer the question of how system failures affect trust in cobots as well as human autonomy, and how they can be recovered afterwards. Therefore, a VR experiment (n = 39) was set up to investigate the influence of a cobot failure and its severity on human autonomy and trust in the cobot. Furthermore, the influence of transparent communication about the failure and next steps was investigated. The results show that both trust and autonomy suffer after cobot failures, with the severity of the failure having a stronger negative impact on trust, but not on autonomy. Both trust and autonomy can be partially restored by transparent communication.
>
---
#### [new 061] Lightweight Structured Multimodal Reasoning for Clinical Scene Understanding in Robotics
- **分类: cs.CV; cs.AI; cs.HC; cs.RO**

- **简介: 该论文提出一种轻量级多模态框架，用于机器人临床场景理解。针对现有视觉-语言模型在时序推理和结构化输出的不足，结合Qwen2.5-VL与SmolAgent，实现动态工具调用与可解释推理，提升医疗机器人安全性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.22014v1](http://arxiv.org/pdf/2509.22014v1)**

> **作者:** Saurav Jha; Stefan K. Ehrlich
>
> **备注:** 11 pages, 3 figures
>
> **摘要:** Healthcare robotics requires robust multimodal perception and reasoning to ensure safety in dynamic clinical environments. Current Vision-Language Models (VLMs) demonstrate strong general-purpose capabilities but remain limited in temporal reasoning, uncertainty estimation, and structured outputs needed for robotic planning. We present a lightweight agentic multimodal framework for video-based scene understanding. Combining the Qwen2.5-VL-3B-Instruct model with a SmolAgent-based orchestration layer, it supports chain-of-thought reasoning, speech-vision fusion, and dynamic tool invocation. The framework generates structured scene graphs and leverages a hybrid retrieval module for interpretable and adaptive reasoning. Evaluations on the Video-MME benchmark and a custom clinical dataset show competitive accuracy and improved robustness compared to state-of-the-art VLMs, demonstrating its potential for applications in robot-assisted surgery, patient monitoring, and decision support.
>
---
#### [new 062] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DynaNav，用于高效视觉导航任务。针对现有模型计算开销大、可解释性差的问题，设计动态特征与层选择机制，结合稀疏操作和早退出机制，显著降低计算成本并提升性能。**

- **链接: [http://arxiv.org/pdf/2509.21930v1](http://arxiv.org/pdf/2509.21930v1)**

> **作者:** Jiahui Wang; Changhao Chen
>
> **备注:** Accepted as a poster in NeurIPS 2025
>
> **摘要:** Visual navigation is essential for robotics and embodied AI. However, existing foundation models, particularly those with transformer decoders, suffer from high computational overhead and lack interpretability, limiting their deployment in resource-tight scenarios. To address this, we propose DynaNav, a Dynamic Visual Navigation framework that adapts feature and layer selection based on scene complexity. It employs a trainable hard feature selector for sparse operations, enhancing efficiency and interpretability. Additionally, we integrate feature selection into an early-exit mechanism, with Bayesian Optimization determining optimal exit thresholds to reduce computational cost. Extensive experiments in real-world-based datasets and simulated environments demonstrate the effectiveness of DynaNav. Compared to ViNT, DynaNav achieves a 2.26x reduction in FLOPs, 42.3% lower inference time, and 32.8% lower memory usage, while improving navigation performance across four public datasets.
>
---
## 更新

#### [replaced 001] Empowering Multi-Robot Cooperation via Sequential World Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.13095v2](http://arxiv.org/pdf/2509.13095v2)**

> **作者:** Zijie Zhao; Honglei Guo; Shengqian Chen; Kaixuan Xu; Bo Jiang; Yuanheng Zhu; Dongbin Zhao
>
> **摘要:** Model-based reinforcement learning (MBRL) has shown significant potential in robotics due to its high sample efficiency and planning capability. However, extending MBRL to multi-robot cooperation remains challenging due to the complexity of joint dynamics and the reliance on synchronous communication. SeqWM employs independent, autoregressive agent-wise world models to represent joint dynamics, where each agent generates its future trajectory and plans its actions based on the predictions of its predecessors. This design lowers modeling complexity, alleviates the reliance on communication synchronization, and enables the emergence of advanced cooperative behaviors through explicit intention sharing. Experiments in challenging simulated environments (Bi-DexHands and Multi-Quad) demonstrate that SeqWM outperforms existing state-of-the-art model-based and model-free baselines in both overall performance and sample efficiency, while exhibiting advanced cooperative behaviors such as predictive adaptation, temporal alignment, and role division. Furthermore, SeqWM has been success fully deployed on physical quadruped robots, demonstrating its effectiveness in real-world multi-robot systems. Demos and code are available at: https://sites.google.com/view/seqwm-marl
>
---
#### [replaced 002] Adaptive Diffusion Constrained Sampling for Bimanual Robot Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.13667v3](http://arxiv.org/pdf/2505.13667v3)**

> **作者:** Haolei Tong; Yuezhe Zhang; Sophie Lueth; Georgia Chalvatzaki
>
> **摘要:** Coordinated multi-arm manipulation requires satisfying multiple simultaneous geometric constraints across high-dimensional configuration spaces, which poses a significant challenge for traditional planning and control methods. In this work, we propose Adaptive Diffusion Constrained Sampling (ADCS), a generative framework that flexibly integrates both equality (e.g., relative and absolute pose constraints) and structured inequality constraints (e.g., proximity to object surfaces) into an energy-based diffusion model. Equality constraints are modeled using dedicated energy networks trained on pose differences in Lie algebra space, while inequality constraints are represented via Signed Distance Functions (SDFs) and encoded into learned constraint embeddings, allowing the model to reason about complex spatial regions. A key innovation of our method is a Transformer-based architecture that learns to weight constraint-specific energy functions at inference time, enabling flexible and context-aware constraint integration. Moreover, we adopt a two-phase sampling strategy that improves precision and sample diversity by combining Langevin dynamics with resampling and density-aware re-weighting. Experimental results on dual-arm manipulation tasks show that ADCS significantly improves sample diversity and generalization across settings demanding precise coordination and adaptive constraint handling.
>
---
#### [replaced 003] AVR: Active Vision-Driven Precise Robot Manipulation with Viewpoint and Focal Length Optimization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01439v4](http://arxiv.org/pdf/2503.01439v4)**

> **作者:** Yushan Liu; Shilong Mu; Xintao Chao; Zizhen Li; Yao Mu; Tianxing Chen; Shoujie Li; Chuqiao Lyu; Xiao-Ping Zhang; Wenbo Ding
>
> **备注:** Project Page: https://AVR-robot.github.io
>
> **摘要:** Robotic manipulation in complex scenes demands precise perception of task-relevant details, yet fixed or suboptimal viewpoints often impair fine-grained perception and induce occlusions, constraining imitation-learned policies. We present AVR (Active Vision-driven Robotics), a bimanual teleoperation and learning framework that unifies head-tracked viewpoint control (HMD-to-2-DoF gimbal) with motorized optical zoom to keep targets centered at an appropriate scale during data collection and deployment. In simulation, an AVR plugin augments RoboTwin demonstrations by emulating active vision (ROI-conditioned viewpoint change, aspect-ratio-preserving crops with explicit zoom ratios, and super-resolution), yielding 5-17% gains in task success across diverse manipulations. On our real-world platform, AVR improves success on most tasks, with over 25% gains compared to the static-view baseline, and extended studies further demonstrate robustness under occlusion, clutter, and lighting disturbances, as well as generalization to unseen environments and objects. These results pave the way for future robotic precision manipulation methods in the pursuit of human-level dexterity and precision.
>
---
#### [replaced 004] Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.17659v3](http://arxiv.org/pdf/2505.17659v3)**

> **作者:** Xiaolong Tang; Meina Kan; Shiguang Shan; Xilin Chen
>
> **摘要:** Safe and feasible trajectory planning is critical for real-world autonomous driving systems. However, existing learning-based planners rely heavily on expert demonstrations, which not only lack explicit safety awareness but also risk inheriting undesirable behaviors such as speeding from suboptimal human driving data. Inspired by the success of large language models, we propose Plan-R1, a two-stage trajectory planning framework that decouples principle alignment from behavior learning. In the first stage, a general trajectory predictor is pre-trained on expert data to capture diverse, human-like driving behaviors. In the second stage, the model is fine-tuned with rule-based rewards using Group Relative Policy Optimization (GRPO), explicitly aligning ego planning with principles such as safety, comfort, and traffic rule compliance. This two-stage paradigm retains human-like behaviors while enhancing safety awareness and discarding undesirable patterns from demonstrations. Furthermore, we identify a key limitation of directly applying GRPO to planning: group-wise normalization erases cross-group scale differences, causing rare, high-variance safety-violation groups to have similar advantages as abundant low-variance safe groups, thereby suppressing optimization for safety-critical objectives. To address this, we propose Variance-Decoupled GRPO (VD-GRPO), which replaces normalization with centering and fixed scaling to preserve absolute reward magnitudes, ensuring that safety-critical objectives remain dominant throughout training. Experiments on the nuPlan benchmark demonstrate that Plan-R1 significantly improves planning safety and feasibility, achieving state-of-the-art performance, particularly in realistic reactive settings. Our code is available at https://github.com/XiaolongTang23/Plan-R1.
>
---
#### [replaced 005] MapExRL: Human-Inspired Indoor Exploration with Predicted Environment Context and Reinforcement Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01548v2](http://arxiv.org/pdf/2503.01548v2)**

> **作者:** Narek Harutyunyan; Brady Moon; Seungchan Kim; Cherie Ho; Adam Hung; Sebastian Scherer
>
> **备注:** 8 pages, 6 figures, ICAR 2025
>
> **摘要:** Path planning for robotic exploration is challenging, requiring reasoning over unknown spaces and anticipating future observations. Efficient exploration requires selecting budget-constrained paths that maximize information gain. Despite advances in autonomous exploration, existing algorithms still fall short of human performance, particularly in structured environments where predictive cues exist but are underutilized. Guided by insights from our user study, we introduce MapExRL, which improves robot exploration efficiency in structured indoor environments by enabling longer-horizon planning through a learned policy and global map predictions. Unlike many learning-based exploration methods that use motion primitives as the action space, our approach leverages frontiers for more efficient model learning and longer horizon reasoning. Our framework generates global map predictions from the observed map, which our policy utilizes, along with the prediction uncertainty, estimated sensor coverage, frontier distance, and remaining distance budget, to assess the strategic long-term value of frontiers. By leveraging multiple frontier scoring methods and additional context, our policy makes more informed decisions at each stage of the exploration. We evaluate our framework on a real-world indoor map dataset, achieving up to an 18.8% improvement over the strongest state-of-the-art baseline, with even greater gains compared to conventional frontier-based algorithms. Website: https://mapexrl.github.io
>
---
#### [replaced 006] Multi-CAP: A Multi-Robot Connectivity-Aware Hierarchical Coverage Path Planning Algorithm for Unknown Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.14941v2](http://arxiv.org/pdf/2509.14941v2)**

> **作者:** Zongyuan Shen; Burhanuddin Shirose; Prasanna Sriganesh; Bhaskar Vundurthy; Howie Choset; Matthew Travers
>
> **摘要:** Efficient coordination of multiple robots for coverage of large, unknown environments is a significant challenge that involves minimizing the total coverage path length while reducing inter-robot conflicts. In this paper, we introduce a Multi-robot Connectivity-Aware Planner (Multi-CAP), a hierarchical coverage path planning algorithm that facilitates multi-robot coordination through a novel connectivity-aware approach. The algorithm constructs and dynamically maintains an adjacency graph that represents the environment as a set of connected subareas. Critically, we make the assumption that the environment, while unknown, is bounded. This allows for incremental refinement of the adjacency graph online to ensure its structure represents the physical layout of the space, both in observed and unobserved areas of the map as robots explore the environment. We frame the task of assigning subareas to robots as a Vehicle Routing Problem (VRP), a well-studied problem for finding optimal routes for a fleet of vehicles. This is used to compute disjoint tours that minimize redundant travel, assigning each robot a unique, non-conflicting set of subareas. Each robot then executes its assigned tour, independently adapting its coverage strategy within each subarea to minimize path length based on real-time sensor observations of the subarea. We demonstrate through simulations and multi-robot hardware experiments that Multi-CAP significantly outperforms state-of-the-art methods in key metrics, including coverage time, total path length, and path overlap ratio. Ablation studies further validate the critical role of our connectivity-aware graph and the global tour planner in achieving these performance gains.
>
---
#### [replaced 007] Gaussian-Process-based Adaptive Tracking Control with Dynamic Active Learning for Autonomous Ground Vehicles
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2501.14672v2](http://arxiv.org/pdf/2501.14672v2)**

> **作者:** Kristóf Floch; Tamás Péni; Roland Tóth
>
> **备注:** Submitted to IEEE Transactions on Control Systems Technology (revised)
>
> **摘要:** This article proposes an active-learning-based adaptive trajectory tracking control method for autonomous ground vehicles to compensate for modeling errors and unmodeled dynamics. The nominal vehicle model is decoupled into lateral and longitudinal subsystems, which are augmented with online Gaussian Processes (GPs), using measurement data. The estimated mean functions of the GPs are used to construct a feedback compensator, which, together with an LPV state feedback controller designed for the nominal system, gives the adaptive control structure. To assist exploration of the dynamics, the paper proposes a new, dynamic active learning method to collect the most informative samples to accelerate the training process. To analyze the performance of the overall learning tool-chain provided controller, a novel iterative, counterexample-based algorithm is proposed for calculating the induced L2 gain between the reference trajectory and the tracking error. The analysis can be executed for a set of possible realizations of the to-be-controlled system, giving robust performance certificate of the learning method under variation of the vehicle dynamics. The efficiency of the proposed control approach is shown on a high-fidelity physics simulator and in real experiments using a 1/10 scale F1TENTH electric car.
>
---
#### [replaced 008] DSPv2: Improved Dense Policy for Effective and Generalizable Whole-body Mobile Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.16063v2](http://arxiv.org/pdf/2509.16063v2)**

> **作者:** Yue Su; Chubin Zhang; Sijin Chen; Liufan Tan; Yansong Tang; Jianan Wang; Xihui Liu
>
> **摘要:** Learning whole-body mobile manipulation via imitation is essential for generalizing robotic skills to diverse environments and complex tasks. However, this goal is hindered by significant challenges, particularly in effectively processing complex observation, achieving robust generalization, and generating coherent actions. To address these issues, we propose DSPv2, a novel policy architecture. DSPv2 introduces an effective encoding scheme that aligns 3D spatial features with multi-view 2D semantic features. This fusion enables the policy to achieve broad generalization while retaining the fine-grained perception necessary for precise control. Furthermore, we extend the Dense Policy paradigm to the whole-body mobile manipulation domain, demonstrating its effectiveness in generating coherent and precise actions for the whole-body robotic platform. Extensive experiments show that our method significantly outperforms existing approaches in both task performance and generalization ability. Project page is available at: https://selen-suyue.github.io/DSPv2Net/.
>
---
#### [replaced 009] The need for and feasibility of alternative ground robots to traverse sandy and rocky extraterrestrial terrain
- **分类: cs.RO; cs.SY; eess.SY; physics.bio-ph**

- **链接: [http://arxiv.org/pdf/2201.11984v3](http://arxiv.org/pdf/2201.11984v3)**

> **作者:** Chen Li; Kevin Lewis
>
> **摘要:** Robotic spacecraft have helped expand our reach for many planetary exploration missions. Most ground mobile planetary exploration robots use wheeled or modified wheeled platforms. Although extraordinarily successful at completing intended mission goals, because of the limitations of wheeled locomotion, they have been largely limited to benign, solid terrain and avoided extreme terrain with loose soil/sand and large rocks. Unfortunately, such challenging terrain is often scientifically interesting for planetary geology. Although many animals traverse such terrain at ease, robots have not matched their performance and robustness. This is in major part due to a lack of fundamental understanding of how effective locomotion can be generated from controlled interaction with complex terrain on the same level of flight aerodynamics and underwater vehicle hydrodynamics. Early fundamental understanding of legged and limbless locomotor-ground interaction has already enabled stable and efficient bio-inspired robot locomotion on relatively flat ground with small obstacles. Recent progress in the new field of terradynamics of locomotor-terrain interaction begins to reveal the principles of bio-inspired locomotion on loose soil/sand and over large obstacles. Multi-legged and limbless platforms using terradynamics insights hold the promise for serving as robust alternative platforms for traversing extreme extraterrestrial terrain and expanding our reach in planetary exploration.
>
---
#### [replaced 010] Residual Off-Policy RL for Finetuning Behavior Cloning Policies
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.19301v2](http://arxiv.org/pdf/2509.19301v2)**

> **作者:** Lars Ankile; Zhenyu Jiang; Rocky Duan; Guanya Shi; Pieter Abbeel; Anusha Nagabandi
>
> **备注:** Project website: https://residual-offpolicy-rl.github.io
>
> **摘要:** Recent advances in behavior cloning (BC) have enabled impressive visuomotor control policies. However, these approaches are limited by the quality of human demonstrations, the manual effort required for data collection, and the diminishing returns from offline data. In comparison, reinforcement learning (RL) trains an agent through autonomous interaction with the environment and has shown remarkable success in various domains. Still, training RL policies directly on real-world robots remains challenging due to sample inefficiency, safety concerns, and the difficulty of learning from sparse rewards for long-horizon tasks, especially for high-degree-of-freedom (DoF) systems. We present a recipe that combines the benefits of BC and RL through a residual learning framework. Our approach leverages BC policies as black-box bases and learns lightweight per-step residual corrections via sample-efficient off-policy RL. We demonstrate that our method requires only sparse binary reward signals and can effectively improve manipulation policies on high-degree-of-freedom (DoF) systems in both simulation and the real world. In particular, we demonstrate, to the best of our knowledge, the first successful real-world RL training on a humanoid robot with dexterous hands. Our results demonstrate state-of-the-art performance in various vision-based tasks, pointing towards a practical pathway for deploying RL in the real world.
>
---
#### [replaced 011] DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21969v4](http://arxiv.org/pdf/2505.21969v4)**

> **作者:** Tianjun Gu; Linfeng Li; Xuhong Wang; Chenghua Gong; Jingyu Gong; Zhizhong Zhang; Yuan Xie; Lizhuang Ma; Xin Tan
>
> **摘要:** Adaptive navigation in unfamiliar environments is crucial for household service robots but remains challenging due to the need for both low-level path planning and high-level scene understanding. While recent vision-language model (VLM) based zero-shot approaches reduce dependence on prior maps and scene-specific training data, they face significant limitations: spatiotemporal discontinuity from discrete observations, unstructured memory representations, and insufficient task understanding leading to navigation failures. We propose DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation), a novel cognitive-inspired framework consisting of Ventral and Dorsal Streams that mimics human navigation capabilities. The Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology Map to handle spatiotemporal discontinuities, while the Ventral Stream combines RAG-VLM and Policy-VLM to improve decision-making. Our approach also develops Nav-Ensurance to ensure navigation safety and efficiency. We evaluate DORAEMON on the HM3D, MP3D, and GOAT datasets, where it achieves state-of-the-art performance on both success rate (SR) and success weighted by path length (SPL) metrics, significantly outperforming existing methods. We also introduce a new evaluation metric (AORI) to assess navigation intelligence better. Comprehensive experiments demonstrate DORAEMON's effectiveness in zero-shot autonomous navigation without requiring prior map building or pre-training.
>
---
#### [replaced 012] STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.20470v3](http://arxiv.org/pdf/2405.20470v3)**

> **作者:** Jiuhong Xiao; Ning Zhang; Daniel Tortei; Giuseppe Loianno
>
> **备注:** 8 pages, 7 figures. Accepted for IEEE Robotics and Automation Letters
>
> **摘要:** Accurate geo-localization of Unmanned Aerial Vehicles (UAVs) is crucial for outdoor applications including search and rescue operations, power line inspections, and environmental monitoring. The vulnerability of Global Navigation Satellite Systems (GNSS) signals to interference and spoofing necessitates the development of additional robust localization methods for autonomous navigation. Visual Geo-localization (VG), leveraging onboard cameras and reference satellite maps, offers a promising solution for absolute localization. Specifically, Thermal Geo-localization (TG), which relies on image-based matching between thermal imagery with satellite databases, stands out by utilizing infrared cameras for effective nighttime localization. However, the efficiency and effectiveness of current TG approaches, are hindered by dense sampling on satellite maps and geometric noises in thermal query images. To overcome these challenges, we introduce STHN, a novel UAV thermal geo-localization approach that employs a coarse-to-fine deep homography estimation method. This method attains reliable thermal geo-localization within a 512-meter radius of the UAV's last known location even with a challenging 11% size ratio between thermal and satellite images, despite the presence of indistinct textures and self-similar patterns. We further show how our research significantly enhances UAV thermal geo-localization performance and robustness against geometric noises under low-visibility conditions in the wild. The code is made publicly available.
>
---
#### [replaced 013] AnchDrive: Bootstrapping Diffusion Policies with Hybrid Trajectory Anchors for End-to-End Driving
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.20253v2](http://arxiv.org/pdf/2509.20253v2)**

> **作者:** Jinhao Chai; Anqing Jiang; Hao Jiang; Shiyi Mu; Zichong Gu; Hao Sun; Shugong Xu
>
> **摘要:** End-to-end multi-modal planning has become a transformative paradigm in autonomous driving, effectively addressing behavioral multi-modality and the generalization challenge in long-tail scenarios. We propose AnchDrive, a framework for end-to-end driving that effectively bootstraps a diffusion policy to mitigate the high computational cost of traditional generative models. Rather than denoising from pure noise, AnchDrive initializes its planner with a rich set of hybrid trajectory anchors. These anchors are derived from two complementary sources: a static vocabulary of general driving priors and a set of dynamic, context-aware trajectories. The dynamic trajectories are decoded in real-time by a Transformer that processes dense and sparse perceptual features. The diffusion model then learns to refine these anchors by predicting a distribution of trajectory offsets, enabling fine-grained refinement. This anchor-based bootstrapping design allows for efficient generation of diverse, high-quality trajectories. Experiments on the NAVSIM benchmark confirm that AnchDrive sets a new state-of-the-art and shows strong generalizability
>
---
#### [replaced 014] Excavating in the Wild: The GOOSE-Ex Dataset for Semantic Segmentation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.18788v2](http://arxiv.org/pdf/2409.18788v2)**

> **作者:** Raphael Hagmanns; Peter Mortimer; Miguel Granero; Thorsten Luettel; Janko Petereit
>
> **备注:** Accepted for publication at 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** The successful deployment of deep learning-based techniques for autonomous systems is highly dependent on the data availability for the respective system in its deployment environment. Especially for unstructured outdoor environments, very few datasets exist for even fewer robotic platforms and scenarios. In an earlier work, we presented the German Outdoor and Offroad Dataset (GOOSE) framework along with 10000 multimodal frames from an offroad vehicle to enhance the perception capabilities in unstructured environments. In this work, we address the generalizability of the GOOSE framework. To accomplish this, we open-source the GOOSE-Ex dataset, which contains additional 5000 labeled multimodal frames from various completely different environments, recorded on a robotic excavator and a quadruped platform. We perform a comprehensive analysis of the semantic segmentation performance on different platforms and sensor modalities in unseen environments. In addition, we demonstrate how the combined datasets can be utilized for different downstream applications or competitions such as offroad navigation, object manipulation or scene completion. The dataset, its platform documentation and pre-trained state-of-the-art models for offroad perception will be made available on https://goose-dataset.de/. \
>
---
#### [replaced 015] Learning Personalized Driving Styles via Reinforcement Learning from Human Feedback
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10434v2](http://arxiv.org/pdf/2503.10434v2)**

> **作者:** Derun Li; Changye Li; Yue Wang; Jianwei Ren; Xin Wen; Pengxiang Li; Leimeng Xu; Kun Zhan; Peng Jia; Xianpeng Lang; Ningyi Xu; Hang Zhao
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Generating human-like and adaptive trajectories is essential for autonomous driving in dynamic environments. While generative models have shown promise in synthesizing feasible trajectories, they often fail to capture the nuanced variability of personalized driving styles due to dataset biases and distributional shifts. To address this, we introduce TrajHF, a human feedback-driven finetuning framework for generative trajectory models, designed to align motion planning with diverse driving styles. TrajHF incorporates multi-conditional denoiser and reinforcement learning with human feedback to refine multi-modal trajectory generation beyond conventional imitation learning. This enables better alignment with human driving preferences while maintaining safety and feasibility constraints. TrajHF achieves performance comparable to the state-of-the-art on NavSim benchmark. TrajHF sets a new paradigm for personalized and adaptable trajectory generation in autonomous driving.
>
---
#### [replaced 016] Learning Hierarchical Domain Models Through Environment-Grounded Interaction
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.13497v2](http://arxiv.org/pdf/2505.13497v2)**

> **作者:** Claudius Kienle; Benjamin Alt; Oleg Arenz; Jan Peters
>
> **摘要:** Domain models enable autonomous agents to solve long-horizon tasks by producing interpretable plans. However, in open-world environments, a single general domain model cannot capture the variety of tasks, so agents must generate suitable task-specific models on the fly. Large Language Models (LLMs), with their implicit common knowledge, can generate such domains, but suffer from high error rates that limit their applicability. Hence, related work relies on extensive human feed-back or prior knowledge, which undermines autonomous, open-world deployment. In this work, we propose LODGE, a framework for autonomous domain learning from LLMs and environment grounding. LODGE builds on hierarchical abstractions and automated simulations to identify and correct inconsistencies between abstraction layers and between the model and environment. Our framework is task-agnostic, as it generates predicates, operators, and their preconditions and effects, while only assuming access to a simulator and a set of generic, executable low-level skills. Experiments on two International Planning Competition ( IPC) domains and a robotic assembly domain show that LODGE yields more accurate domain models and higher task success than existing methods, requiring remarkably few environment interactions and no human feedback or demonstrations.
>
---
#### [replaced 017] DexWrist: A Robotic Wrist for Constrained and Dynamic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.01008v2](http://arxiv.org/pdf/2507.01008v2)**

> **作者:** Martin Peticco; Gabriella Ulloa; John Marangola; Nitish Dashora; Pulkit Agrawal
>
> **备注:** 9 pages, 8 figures. Submitted to ICRA 2026
>
> **摘要:** Development of dexterous manipulation hardware has primarily focused on hands and grippers. However, robotic wrists are equally critical, often playing a greater role than the end effector itself. Many conventional wrist designs fall short in human environments because they are too large or rely on rigid, high-reduction actuators that cannot support dynamic, contact-rich tasks. Some designs address these issues using backdrivable quasi-direct drive (QDD) actuators and compact form factors. However, they are often difficult to model and control due to coupled kinematics or high mechanical inertia. We present DexWrist, a robotic wrist that is designed to advance robotic manipulation in highly constrained environments, enable dynamic and contact-rich tasks, and simplify policy learning. DexWrist provides low-impedance actuation, low inertia, integrated proprioception, high speed, and a large workspace. Together, these capabilities support robust learning-based manipulation. DexWrist accelerates policy learning by: (i) enabling faster teleoperation for scalable data collection, (ii) simplifying the learned function through shorter trajectories and decoupled degrees of freedom (DOFs), (iii) providing natural backdrivability for safe contact without complex compliant controllers, and (iv) expanding the manipulation workspace in cluttered scenes. In our experiments, DexWrist improved policy success rates by 50-55% and reduced task completion times by a factor of 3-5. More details about the wrist can be found at https://dexwrist.csail.mit.edu.
>
---
#### [replaced 018] One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.18382v3](http://arxiv.org/pdf/2505.18382v3)**

> **作者:** Jinbang Huang; Yixin Xiao; Zhanguang Zhang; Mark Coates; Jianye Hao; Yingxue Zhang
>
> **备注:** 35 pages, 10 figures
>
> **摘要:** Pre-trained large language models (LLMs) show promise for robotic task planning but often struggle to guarantee correctness in long-horizon problems. Task and motion planning (TAMP) addresses this by grounding symbolic plans in low-level execution, yet it relies heavily on manually engineered planning domains. To improve long-horizon planning reliability and reduce human intervention, we present Planning Domain Derivation with LLMs (PDDLLM), a framework that automatically induces symbolic predicates and actions directly from demonstration trajectories by combining LLM reasoning with physical simulation roll-outs. Unlike prior domain-inference methods that rely on partially predefined or language descriptions of planning domains, PDDLLM constructs domains without manual domain initialization and automatically integrates them with motion planners to produce executable plans, enhancing long-horizon planning automation. Across 1,200 tasks in nine environments, PDDLLM outperforms six LLM-based planning baselines, achieving at least 20\% higher success rates, reduced token costs, and successful deployment on multiple physical robot platforms.
>
---
#### [replaced 019] Mobi-$π$: Mobilizing Your Robot Learning Policy
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.23692v2](http://arxiv.org/pdf/2505.23692v2)**

> **作者:** Jingyun Yang; Isabella Huang; Brandon Vu; Max Bajracharya; Rika Antonova; Jeannette Bohg
>
> **备注:** CoRL 2025. Project website: https://mobipi.github.io/
>
> **摘要:** Learned visuomotor policies are capable of performing increasingly complex manipulation tasks. However, most of these policies are trained on data collected from limited robot positions and camera viewpoints. This leads to poor generalization to novel robot positions, which limits the use of these policies on mobile platforms, especially for precise tasks like pressing buttons or turning faucets. In this work, we formulate the policy mobilization problem: find a mobile robot base pose in a novel environment that is in distribution with respect to a manipulation policy trained on a limited set of camera viewpoints. Compared to retraining the policy itself to be more robust to unseen robot base pose initializations, policy mobilization decouples navigation from manipulation and thus does not require additional demonstrations. Crucially, this problem formulation complements existing efforts to improve manipulation policy robustness to novel viewpoints and remains compatible with them. We propose a novel approach for policy mobilization that bridges navigation and manipulation by optimizing the robot's base pose to align with an in-distribution base pose for a learned policy. Our approach utilizes 3D Gaussian Splatting for novel view synthesis, a score function to evaluate pose suitability, and sampling-based optimization to identify optimal robot poses. To understand policy mobilization in more depth, we also introduce the Mobi-$\pi$ framework, which includes: (1) metrics that quantify the difficulty of mobilizing a given policy, (2) a suite of simulated mobile manipulation tasks based on RoboCasa to evaluate policy mobilization, and (3) visualization tools for analysis. In both our developed simulation task suite and the real world, we show that our approach outperforms baselines, demonstrating its effectiveness for policy mobilization.
>
---
#### [replaced 020] Decentralized Aerial Manipulation of a Cable-Suspended Load using Multi-Agent Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.MA; I.2.9; I.2.11; I.2.6**

- **链接: [http://arxiv.org/pdf/2508.01522v2](http://arxiv.org/pdf/2508.01522v2)**

> **作者:** Jack Zeng; Andreu Matoses Gimenez; Eugene Vinitsky; Javier Alonso-Mora; Sihao Sun
>
> **摘要:** This paper presents the first decentralized method to enable real-world 6-DoF manipulation of a cable-suspended load using a team of Micro-Aerial Vehicles (MAVs). Our method leverages multi-agent reinforcement learning (MARL) to train an outer-loop control policy for each MAV. Unlike state-of-the-art controllers that utilize a centralized scheme, our policy does not require global states, inter-MAV communications, nor neighboring MAV information. Instead, agents communicate implicitly through load pose observations alone, which enables high scalability and flexibility. It also significantly reduces computing costs during inference time, enabling onboard deployment of the policy. In addition, we introduce a new action space design for the MAVs using linear acceleration and body rates. This choice, combined with a robust low-level controller, enables reliable sim-to-real transfer despite significant uncertainties caused by cable tension during dynamic 3D motion. We validate our method in various real-world experiments, including full-pose control under load model uncertainties, showing setpoint tracking performance comparable to the state-of-the-art centralized method. We also demonstrate cooperation amongst agents with heterogeneous control policies, and robustness to the complete in-flight loss of one MAV. Videos of experiments: https://autonomousrobots.nl/paper_websites/aerial-manipulation-marl
>
---
#### [replaced 021] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.13019v2](http://arxiv.org/pdf/2507.13019v2)**

> **作者:** Liuyi Wang; Xinyuan Xia; Hui Zhao; Hanqing Wang; Tai Wang; Yilun Chen; Chengju Liu; Qijun Chen; Jiangmiao Pang
>
> **备注:** Accepted by ICCV 2025
>
> **摘要:** Recent Vision-and-Language Navigation (VLN) advancements are promising, but their idealized assumptions about robot movement and control fail to reflect physically embodied deployment challenges. To bridge this gap, we introduce VLN-PE, a physically realistic VLN platform supporting humanoid, quadruped, and wheeled robots. For the first time, we systematically evaluate several ego-centric VLN methods in physical robotic settings across different technical pipelines, including classification models for single-step discrete action prediction, a diffusion model for dense waypoint prediction, and a train-free, map-based large language model (LLM) integrated with path planning. Our results reveal significant performance degradation due to limited robot observation space, environmental lighting variations, and physical challenges like collisions and falls. This also exposes locomotion constraints for legged robots in complex environments. VLN-PE is highly extensible, allowing seamless integration of new scenes beyond MP3D, thereby enabling more comprehensive VLN evaluation. Despite the weak generalization of current models in physical deployment, VLN-PE provides a new pathway for improving cross-embodiment's overall adaptability. We hope our findings and tools inspire the community to rethink VLN limitations and advance robust, practical VLN models. The code is available at https://crystalsixone.github.io/vln_pe.github.io/.
>
---
#### [replaced 022] Flexible and Foldable: Workspace Analysis and Object Manipulation Using a Soft, Interconnected, Origami-Inspired Actuator Array
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.13998v2](http://arxiv.org/pdf/2509.13998v2)**

> **作者:** Bailey Dacre; Rodrigo Moreno; Serhat Demirtas; Ziqiao Wang; Yuhao Jiang; Jamie Paik; Kasper Stoy; Andrés Faíña
>
> **摘要:** Object manipulation is a fundamental challenge in robotics, where systems must balance trade-offs among manipulation capabilities, system complexity, and throughput. Distributed manipulator systems (DMS) use the coordinated motion of actuator arrays to perform complex object manipulation tasks, seeing widespread exploration within the literature and in industry. However, existing DMS designs typically rely on high actuator densities and impose constraints on object-to-actuator scale ratios, limiting their adaptability. We present a novel DMS design utilizing an array of 3-DoF, origami-inspired robotic tiles interconnected by a compliant surface layer. Unlike conventional DMS, our approach enables manipulation not only at the actuator end effectors but also across a flexible surface connecting all actuators; creating a continuous, controllable manipulation surface. We analyse the combined workspace of such a system, derive simple motion primitives, and demonstrate its capabilities to translate simple geometric objects across an array of tiles. By leveraging the inter-tile connective material, our approach significantly reduces actuator density, increasing the area over which an object can be manipulated by x1.84 without an increase in the number of actuators. This design offers a lower cost and complexity alternative to traditional high-density arrays, and introduces new opportunities for manipulation strategies that leverage the flexibility of the interconnected surface.
>
---
#### [replaced 023] mindmap: Spatial Memory in Deep Feature Maps for 3D Action Policies
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.20297v2](http://arxiv.org/pdf/2509.20297v2)**

> **作者:** Remo Steiner; Alexander Millane; David Tingdahl; Clemens Volk; Vikram Ramasamy; Xinjie Yao; Peter Du; Soha Pouya; Shiwei Sheng
>
> **备注:** Accepted to CoRL 2025 Workshop RemembeRL
>
> **摘要:** End-to-end learning of robot control policies, structured as neural networks, has emerged as a promising approach to robotic manipulation. To complete many common tasks, relevant objects are required to pass in and out of a robot's field of view. In these settings, spatial memory - the ability to remember the spatial composition of the scene - is an important competency. However, building such mechanisms into robot learning systems remains an open research problem. We introduce mindmap (Spatial Memory in Deep Feature Maps for 3D Action Policies), a 3D diffusion policy that generates robot trajectories based on a semantic 3D reconstruction of the environment. We show in simulation experiments that our approach is effective at solving tasks where state-of-the-art approaches without memory mechanisms struggle. We release our reconstruction system, training code, and evaluation tasks to spur research in this direction.
>
---
#### [replaced 024] GraphSCENE: On-Demand Critical Scenario Generation for Autonomous Vehicles in Simulation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.13514v3](http://arxiv.org/pdf/2410.13514v3)**

> **作者:** Efimia Panagiotaki; Georgi Pramatarov; Lars Kunze; Daniele De Martini
>
> **备注:** Accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Testing and validating Autonomous Vehicle (AV) performance in safety-critical and diverse scenarios is crucial before real-world deployment. However, manually creating such scenarios in simulation remains a significant and time-consuming challenge. This work introduces a novel method that generates dynamic temporal scene graphs corresponding to diverse traffic scenarios, on-demand, tailored to user-defined preferences, such as AV actions, sets of dynamic agents, and criticality levels. A temporal Graph Neural Network (GNN) model learns to predict relationships between ego-vehicle, agents, and static structures, guided by real-world spatiotemporal interaction patterns and constrained by an ontology that restricts predictions to semantically valid links. Our model consistently outperforms the baselines in accurately generating links corresponding to the requested scenarios. We render the predicted scenarios in simulation to further demonstrate their effectiveness as testing environments for AV agents.
>
---
