# 机器人 cs.RO

- **最新发布 24 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] Ground-Aware Octree-A* Hybrid Path Planning for Memory-Efficient 3D Navigation of Ground Vehicles
- **分类: cs.RO**

- **简介: 该论文提出一种结合A*算法与八叉树结构的3D路径规划方法，解决地面车辆在复杂地形中高效导航问题。通过优化成本函数和压缩3D网格地图，提升路径规划的实时性与内存效率，实现最优路径生成。**

- **链接: [http://arxiv.org/pdf/2509.04950v1](http://arxiv.org/pdf/2509.04950v1)**

> **作者:** Byeong-Il Ham; Hyun-Bin Kim; Kyung-Soo Kim
>
> **备注:** 6 pages, 3 figures. Accepted at The 25th International Conference on Control, Automation, and Systems (ICCAS 2025). This is arXiv v1 (pre-revision); the camera-ready has been submitted
>
> **摘要:** In this paper, we propose a 3D path planning method that integrates the A* algorithm with the octree structure. Unmanned Ground Vehicles (UGVs) and legged robots have been extensively studied, enabling locomotion across a variety of terrains. Advances in mobility have enabled obstacles to be regarded not only as hindrances to be avoided, but also as navigational aids when beneficial. A modified 3D A* algorithm generates an optimal path by leveraging obstacles during the planning process. By incorporating a height-based penalty into the cost function, the algorithm enables the use of traversable obstacles to aid locomotion while avoiding those that are impassable, resulting in more efficient and realistic path generation. The octree-based 3D grid map achieves compression by merging high-resolution nodes into larger blocks, especially in obstacle-free or sparsely populated areas. This reduces the number of nodes explored by the A* algorithm, thereby improving computational efficiency and memory usage, and supporting real-time path planning in practical environments. Benchmark results demonstrate that the use of octree structure ensures an optimal path while significantly reducing memory usage and computation time.
>
---
#### [new 002] Lyapunov-Based Deep Learning Control for Robots with Unknown Jacobian
- **分类: cs.RO**

- **简介: 论文提出基于Lyapunov的深度学习控制框架，解决机器人控制中雅可比未知导致的稳定性问题。通过模块化学习实时更新权重，确保系统稳定性，并在工业机器人上验证方法的有效性，实现安全实时控制。**

- **链接: [http://arxiv.org/pdf/2509.04984v1](http://arxiv.org/pdf/2509.04984v1)**

> **作者:** Koji Matsuno; Chien Chern Cheah
>
> **摘要:** Deep learning, with its exceptional learning capabilities and flexibility, has been widely applied in various applications. However, its black-box nature poses a significant challenge in real-time robotic applications, particularly in robot control, where trustworthiness and robustness are critical in ensuring safety. In robot motion control, it is essential to analyze and ensure system stability, necessitating the establishment of methodologies that address this need. This paper aims to develop a theoretical framework for end-to-end deep learning control that can be integrated into existing robot control theories. The proposed control algorithm leverages a modular learning approach to update the weights of all layers in real time, ensuring system stability based on Lyapunov-like analysis. Experimental results on industrial robots are presented to illustrate the performance of the proposed deep learning controller. The proposed method offers an effective solution to the black-box problem in deep learning, demonstrating the possibility of deploying real-time deep learning strategies for robot kinematic control in a stable manner. This achievement provides a critical foundation for future advancements in deep learning based real-time robotic applications.
>
---
#### [new 003] A Knowledge-Driven Diffusion Policy for End-to-End Autonomous Driving Based on Expert Routing
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对端到端自动驾驶中多模态动作生成、时间一致性及泛化能力不足的问题，提出KDP方法，结合扩散模型与稀疏专家路由机制，提升控制平滑度与成功率。**

- **链接: [http://arxiv.org/pdf/2509.04853v1](http://arxiv.org/pdf/2509.04853v1)**

> **作者:** Chengkai Xu; Jiaqi Liu; Yicheng Guo; Peng Hang; Jian Sun
>
> **备注:** https://perfectxu88.github.io/KDP-AD/
>
> **摘要:** End-to-end autonomous driving remains constrained by the need to generate multi-modal actions, maintain temporal stability, and generalize across diverse scenarios. Existing methods often collapse multi-modality, struggle with long-horizon consistency, or lack modular adaptability. This paper presents KDP, a knowledge-driven diffusion policy that integrates generative diffusion modeling with a sparse mixture-of-experts routing mechanism. The diffusion component generates temporally coherent and multi-modal action sequences, while the expert routing mechanism activates specialized and reusable experts according to context, enabling modular knowledge composition. Extensive experiments across representative driving scenarios demonstrate that KDP achieves consistently higher success rates, reduced collision risk, and smoother control compared to prevailing paradigms. Ablation studies highlight the effectiveness of sparse expert activation and the Transformer backbone, and activation analyses reveal structured specialization and cross-scenario reuse of experts. These results establish diffusion with expert routing as a scalable and interpretable paradigm for knowledge-driven end-to-end autonomous driving.
>
---
#### [new 004] Pointing-Guided Target Estimation via Transformer-Based Attention
- **分类: cs.RO; cs.AI; cs.CV; I.2.9; I.2.10; I.2.6**

- **简介: 该论文提出MM-ITF模型，通过Transformer注意力机制将指向手势与视觉数据结合，解决人机协作中机器人理解人类意图的目标预测问题，提升交互准确性。**

- **链接: [http://arxiv.org/pdf/2509.05031v1](http://arxiv.org/pdf/2509.05031v1)**

> **作者:** Luca Müller; Hassan Ali; Philipp Allgeuer; Lukáš Gajdošech; Stefan Wermter
>
> **备注:** Accepted at the 34th International Conference on Artificial Neural Networks (ICANN) 2025,12 pages,4 figures,1 table; work was co-funded by Horizon Europe project TERAIS under Grant agreement number 101079338
>
> **摘要:** Deictic gestures, like pointing, are a fundamental form of non-verbal communication, enabling humans to direct attention to specific objects or locations. This capability is essential in Human-Robot Interaction (HRI), where robots should be able to predict human intent and anticipate appropriate responses. In this work, we propose the Multi-Modality Inter-TransFormer (MM-ITF), a modular architecture to predict objects in a controlled tabletop scenario with the NICOL robot, where humans indicate targets through natural pointing gestures. Leveraging inter-modality attention, MM-ITF maps 2D pointing gestures to object locations, assigns a likelihood score to each, and identifies the most likely target. Our results demonstrate that the method can accurately predict the intended object using monocular RGB data, thus enabling intuitive and accessible human-robot collaboration. To evaluate the performance, we introduce a patch confusion matrix, providing insights into the model's predictions across candidate object locations. Code available at: https://github.com/lucamuellercode/MMITF.
>
---
#### [new 005] Action Chunking with Transformers for Image-Based Spacecraft Guidance and Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ACT方法，利用Transformer和动作分块，在有限专家演示下实现航天器对接控制，优于元强化学习基线，提升轨迹平滑度与样本效率。**

- **链接: [http://arxiv.org/pdf/2509.04628v1](http://arxiv.org/pdf/2509.04628v1)**

> **作者:** Alejandro Posadas-Nava; Andrea Scorsoglio; Luca Ghilardi; Roberto Furfaro; Richard Linares
>
> **备注:** 12 pages, 6 figures, 2025 AAS/AIAA Astrodynamics Specialist Conference
>
> **摘要:** We present an imitation learning approach for spacecraft guidance, navigation, and control(GNC) that achieves high performance from limited data. Using only 100 expert demonstrations, equivalent to 6,300 environment interactions, our method, which implements Action Chunking with Transformers (ACT), learns a control policy that maps visual and state observations to thrust and torque commands. ACT generates smoother, more consistent trajectories than a meta-reinforcement learning (meta-RL) baseline trained with 40 million interactions. We evaluate ACT on a rendezvous task: in-orbit docking with the International Space Station (ISS). We show that our approach achieves greater accuracy, smoother control, and greater sample efficiency.
>
---
#### [new 006] Towards an Accurate and Effective Robot Vision (The Problem of Topological Localization for Mobile Robots)
- **分类: cs.RO; cs.CV**

- **简介: 论文针对移动机器人拓扑定位问题，通过比较多种视觉描述符（如SIFT、Bag-of-Visual-Words）在办公室环境中的性能，优化配置以提升定位准确性与鲁棒性，适应不同光照条件。**

- **链接: [http://arxiv.org/pdf/2509.04948v1](http://arxiv.org/pdf/2509.04948v1)**

> **作者:** Emanuela Boros
>
> **备注:** Master's thesis
>
> **摘要:** Topological localization is a fundamental problem in mobile robotics, since robots must be able to determine their position in order to accomplish tasks. Visual localization and place recognition are challenging due to perceptual ambiguity, sensor noise, and illumination variations. This work addresses topological localization in an office environment using only images acquired with a perspective color camera mounted on a robot platform, without relying on temporal continuity of image sequences. We evaluate state-of-the-art visual descriptors, including Color Histograms, SIFT, ASIFT, RGB-SIFT, and Bag-of-Visual-Words approaches inspired by text retrieval. Our contributions include a systematic, quantitative comparison of these features, distance measures, and classifiers. Performance was analyzed using standard evaluation metrics and visualizations, extending previous experiments. Results demonstrate the advantages of proper configurations of appearance descriptors, similarity measures, and classifiers. The quality of these configurations was further validated in the Robot Vision task of the ImageCLEF evaluation campaign, where the system identified the most likely location of novel image sequences. Future work will explore hierarchical models, ranking methods, and feature combinations to build more robust localization systems, reducing training and runtime while avoiding the curse of dimensionality. Ultimately, this aims toward integrated, real-time localization across varied illumination and longer routes.
>
---
#### [new 007] Surformer v2: A Multimodal Classifier for Surface Understanding from Touch and Vision
- **分类: cs.RO**

- **简介: 该论文提出Surformer v2，解决多模态表面材料分类问题，通过决策级融合视觉（CNN）与触觉（Transformer）数据，提升机器人触觉感知，适用于实时应用。**

- **链接: [http://arxiv.org/pdf/2509.04658v1](http://arxiv.org/pdf/2509.04658v1)**

> **作者:** Manish Kansana; Sindhuja Penchala; Shahram Rahimi; Noorbakhsh Amiri Golilarz
>
> **备注:** 6 pages
>
> **摘要:** Multimodal surface material classification plays a critical role in advancing tactile perception for robotic manipulation and interaction. In this paper, we present Surformer v2, an enhanced multi-modal classification architecture designed to integrate visual and tactile sensory streams through a late(decision level) fusion mechanism. Building on our earlier Surformer v1 framework [1], which employed handcrafted feature extraction followed by mid-level fusion architecture with multi-head cross-attention layers, Surformer v2 integrates the feature extraction process within the model itself and shifts to late fusion. The vision branch leverages a CNN-based classifier(Efficient V-Net), while the tactile branch employs an encoder-only transformer model, allowing each modality to extract modality-specific features optimized for classification. Rather than merging feature maps, the model performs decision-level fusion by combining the output logits using a learnable weighted sum, enabling adaptive emphasis on each modality depending on data context and training dynamics. We evaluate Surformer v2 on the Touch and Go dataset [2], a multi-modal benchmark comprising surface images and corresponding tactile sensor readings. Our results demonstrate that Surformer v2 performs well, maintaining competitive inference speed, suitable for real-time robotic applications. These findings underscore the effectiveness of decision-level fusion and transformer-based tactile modeling for enhancing surface understanding in multi-modal robotic perception.
>
---
#### [new 008] Shared Autonomy through LLMs and Reinforcement Learning for Applications to Ship Hull Inspections
- **分类: cs.RO**

- **简介: 该论文开发基于LLMs和强化学习的共享自主系统，用于船舶 hull 检查，解决复杂高风险环境下的高效人机协作问题，整合LLM任务指定、人机交互框架及行为树任务管理器，通过模拟和实地测试验证其降低认知负担、提升透明度和适应性的效果。**

- **链接: [http://arxiv.org/pdf/2509.05042v1](http://arxiv.org/pdf/2509.05042v1)**

> **作者:** Cristiano Caissutti; Estelle Gerbier; Ehsan Khorrambakht; Paolo Marinelli; Andrea Munafo'; Andrea Caiti
>
> **摘要:** Shared autonomy is a promising paradigm in robotic systems, particularly within the maritime domain, where complex, high-risk, and uncertain environments necessitate effective human-robot collaboration. This paper investigates the interaction of three complementary approaches to advance shared autonomy in heterogeneous marine robotic fleets: (i) the integration of Large Language Models (LLMs) to facilitate intuitive high-level task specification and support hull inspection missions, (ii) the implementation of human-in-the-loop interaction frameworks in multi-agent settings to enable adaptive and intent-aware coordination, and (iii) the development of a modular Mission Manager based on Behavior Trees to provide interpretable and flexible mission control. Preliminary results from simulation and real-world lake-like environments demonstrate the potential of this multi-layered architecture to reduce operator cognitive load, enhance transparency, and improve adaptive behaviour alignment with human intent. Ongoing work focuses on fully integrating these components, refining coordination mechanisms, and validating the system in operational port scenarios. This study contributes to establishing a modular and scalable foundation for trustworthy, human-collaborative autonomy in safety-critical maritime robotics applications.
>
---
#### [new 009] FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies
- **分类: cs.RO**

- **简介: 该论文提出FLOWER模型，解决高效视觉-语言-动作（VLA）策略开发难题，通过模态融合与参数优化减少计算成本，实现跨任务机器人控制，达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2509.04996v1](http://arxiv.org/pdf/2509.04996v1)**

> **作者:** Moritz Reuss; Hongyi Zhou; Marcel Rühle; Ömer Erdinç Yağmurlu; Fabian Otto; Rudolf Lioutikov
>
> **备注:** Published at CoRL 2025
>
> **摘要:** Developing efficient Vision-Language-Action (VLA) policies is crucial for practical robotics deployment, yet current approaches face prohibitive computational costs and resource requirements. Existing diffusion-based VLA policies require multi-billion-parameter models and massive datasets to achieve strong performance. We tackle this efficiency challenge with two contributions: intermediate-modality fusion, which reallocates capacity to the diffusion head by pruning up to $50\%$ of LLM layers, and action-specific Global-AdaLN conditioning, which cuts parameters by $20\%$ through modular adaptation. We integrate these advances into a novel 950 M-parameter VLA called FLOWER. Pretrained in just 200 H100 GPU hours, FLOWER delivers competitive performance with bigger VLAs across $190$ tasks spanning ten simulation and real-world benchmarks and demonstrates robustness across diverse robotic embodiments. In addition, FLOWER achieves a new SoTA of 4.53 on the CALVIN ABC benchmark. Demos, code and pretrained weights are available at https://intuitive-robots.github.io/flower_vla/.
>
---
#### [new 010] Robust Model Predictive Control Design for Autonomous Vehicles with Perception-based Observers
- **分类: cs.RO; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出一种鲁棒MPC框架，针对自动驾驶车辆感知模块的非高斯噪声问题，结合约束区间状态估计与线性规划优化，确保稳定性，实验验证其在重尾噪声下的控制性能优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.05201v1](http://arxiv.org/pdf/2509.05201v1)**

> **作者:** Nariman Niknejad; Gokul S. Sankar; Bahare Kiumarsi; Hamidreza Modares
>
> **摘要:** This paper presents a robust model predictive control (MPC) framework that explicitly addresses the non-Gaussian noise inherent in deep learning-based perception modules used for state estimation. Recognizing that accurate uncertainty quantification of the perception module is essential for safe feedback control, our approach departs from the conventional assumption of zero-mean noise quantification of the perception error. Instead, it employs set-based state estimation with constrained zonotopes to capture biased, heavy-tailed uncertainties while maintaining bounded estimation errors. To improve computational efficiency, the robust MPC is reformulated as a linear program (LP), using a Minkowski-Lyapunov-based cost function with an added slack variable to prevent degenerate solutions. Closed-loop stability is ensured through Minkowski-Lyapunov inequalities and contractive zonotopic invariant sets. The largest stabilizing terminal set and its corresponding feedback gain are then derived via an ellipsoidal approximation of the zonotopes. The proposed framework is validated through both simulations and hardware experiments on an omnidirectional mobile robot along with a camera and a convolutional neural network-based perception module implemented within a ROS2 framework. The results demonstrate that the perception-aware MPC provides stable and accurate control performance under heavy-tailed noise conditions, significantly outperforming traditional Gaussian-noise-based designs in terms of both state estimation error bounding and overall control performance.
>
---
#### [new 011] COMMET: A System for Human-Induced Conflicts in Mobile Manipulation of Everyday Tasks
- **分类: cs.RO**

- **简介: 该论文提出COMMET系统，解决移动机器人日常任务中因人类活动引发的冲突检测与处理问题。通过混合检测方法和用户偏好建模，实现动态环境下的冲突识别与个性化解决方案，支持实际部署与研究。**

- **链接: [http://arxiv.org/pdf/2509.04836v1](http://arxiv.org/pdf/2509.04836v1)**

> **作者:** Dongping Li; Shaoting Peng; John Pohovey; Katherine Rose Driggs-Campbell
>
> **摘要:** Continuous advancements in robotics and AI are driving the integration of robots from industry into everyday environments. However, dynamic and unpredictable human activities in daily lives would directly or indirectly conflict with robot actions. Besides, due to the social attributes of such human-induced conflicts, solutions are not always unique and depend highly on the user's personal preferences. To address these challenges and facilitate the development of household robots, we propose COMMET, a system for human-induced COnflicts in Mobile Manipulation of Everyday Tasks. COMMET employs a hybrid detection approach, which begins with multi-modal retrieval and escalates to fine-tuned model inference for low-confidence cases. Based on collected user preferred options and settings, GPT-4o will be used to summarize user preferences from relevant cases. In preliminary studies, our detection module shows better accuracy and latency compared with GPT models. To facilitate future research, we also design a user-friendly interface for user data collection and demonstrate an effective workflow for real-world deployments.
>
---
#### [new 012] Bootstrapping Reinforcement Learning with Sub-optimal Policies for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **简介: 该论文针对自动驾驶中RL的样本效率与探索难题，提出结合规则车道变更控制器与SAC算法，利用次优策略引导学习，提升驾驶性能与扩展性。**

- **链接: [http://arxiv.org/pdf/2509.04712v1](http://arxiv.org/pdf/2509.04712v1)**

> **作者:** Zhihao Zhang; Chengyang Peng; Ekim Yurtsever; Keith A. Redmill
>
> **摘要:** Automated vehicle control using reinforcement learning (RL) has attracted significant attention due to its potential to learn driving policies through environment interaction. However, RL agents often face training challenges in sample efficiency and effective exploration, making it difficult to discover an optimal driving strategy. To address these issues, we propose guiding the RL driving agent with a demonstration policy that need not be a highly optimized or expert-level controller. Specifically, we integrate a rule-based lane change controller with the Soft Actor Critic (SAC) algorithm to enhance exploration and learning efficiency. Our approach demonstrates improved driving performance and can be extended to other driving scenarios that can similarly benefit from demonstration-based guidance.
>
---
#### [new 013] Planning from Point Clouds over Continuous Actions for Multi-object Rearrangement
- **分类: cs.RO**

- **简介: 该论文提出SPOT方法，解决多物体重新排列中传统离散化限制问题，通过学习模型引导连续动作空间搜索，无需离散化，实现在模拟与现实环境中的高效规划，优于策略学习方法。**

- **链接: [http://arxiv.org/pdf/2509.04645v1](http://arxiv.org/pdf/2509.04645v1)**

> **作者:** Kallol Saha; Amber Li; Angela Rodriguez-Izquierdo; Lifan Yu; Ben Eisner; Maxim Likhachev; David Held
>
> **备注:** Conference on Robot Learning (CoRL) 2025 (https://planning-from-point-clouds.github.io/)
>
> **摘要:** Long-horizon planning for robot manipulation is a challenging problem that requires reasoning about the effects of a sequence of actions on a physical 3D scene. While traditional task planning methods are shown to be effective for long-horizon manipulation, they require discretizing the continuous state and action space into symbolic descriptions of objects, object relationships, and actions. Instead, we propose a hybrid learning-and-planning approach that leverages learned models as domain-specific priors to guide search in high-dimensional continuous action spaces. We introduce SPOT: Search over Point cloud Object Transformations, which plans by searching for a sequence of transformations from an initial scene point cloud to a goal-satisfying point cloud. SPOT samples candidate actions from learned suggesters that operate on partially observed point clouds, eliminating the need to discretize actions or object relationships. We evaluate SPOT on multi-object rearrangement tasks, reporting task planning success and task execution success in both simulation and real-world environments. Our experiments show that SPOT generates successful plans and outperforms a policy-learning approach. We also perform ablations that highlight the importance of search-based planning.
>
---
#### [new 014] In-Context Policy Adaptation via Cross-Domain Skill Diffusion
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出跨域策略适应框架ICPAD，解决长视野多任务环境下有限目标域数据和无模型更新约束下的策略迁移问题。通过跨域一致扩散学习领域无关原型技能与适配器，并结合动态提示增强适应效果，实现在机器人操作和自动驾驶等跨域任务中的高效迁移。**

- **链接: [http://arxiv.org/pdf/2509.04535v1](http://arxiv.org/pdf/2509.04535v1)**

> **作者:** Minjong Yoo; Woo Kyung Kim; Honguk Woo
>
> **备注:** 9 pages
>
> **摘要:** In this work, we present an in-context policy adaptation (ICPAD) framework designed for long-horizon multi-task environments, exploring diffusion-based skill learning techniques in cross-domain settings. The framework enables rapid adaptation of skill-based reinforcement learning policies to diverse target domains, especially under stringent constraints on no model updates and only limited target domain data. Specifically, the framework employs a cross-domain skill diffusion scheme, where domain-agnostic prototype skills and a domain-grounded skill adapter are learned jointly and effectively from an offline dataset through cross-domain consistent diffusion processes. The prototype skills act as primitives for common behavior representations of long-horizon policies, serving as a lingua franca to bridge different domains. Furthermore, to enhance the in-context adaptation performance, we develop a dynamic domain prompting scheme that guides the diffusion-based skill adapter toward better alignment with the target domain. Through experiments with robotic manipulation in Metaworld and autonomous driving in CARLA, we show that our $\oursol$ framework achieves superior policy adaptation performance under limited target domain data conditions for various cross-domain configurations including differences in environment dynamics, agent embodiment, and task horizon.
>
---
#### [new 015] Imitation Learning Based on Disentangled Representation Learning of Behavioral Characteristics
- **分类: cs.RO**

- **简介: 该论文针对机器人根据人类定性指令实时调整动作的任务，提出基于解耦表示学习的模仿学习方法，通过弱监督标签和短序列演示学习指令到动作映射，实现在线调整，优于传统批处理方法。**

- **链接: [http://arxiv.org/pdf/2509.04737v1](http://arxiv.org/pdf/2509.04737v1)**

> **作者:** Ryoga Oishi; Sho Sakaino; Toshiaki Tsuji
>
> **备注:** 16 pages, 5 figures, Accepted at CoRL2025
>
> **摘要:** In the field of robot learning, coordinating robot actions through language instructions is becoming increasingly feasible. However, adapting actions to human instructions remains challenging, as such instructions are often qualitative and require exploring behaviors that satisfy varying conditions. This paper proposes a motion generation model that adapts robot actions in response to modifier directives human instructions imposing behavioral conditions during task execution. The proposed method learns a mapping from modifier directives to actions by segmenting demonstrations into short sequences, assigning weakly supervised labels corresponding to specific modifier types. We evaluated our method in wiping and pick and place tasks. Results show that it can adjust motions online in response to modifier directives, unlike conventional batch-based methods that cannot adapt during execution.
>
---
#### [new 016] DeGuV: Depth-Guided Visual Reinforcement Learning for Generalization and Interpretability in Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 论文提出DeGuV框架，通过深度引导的掩码网络和对比学习，提升视觉强化学习在机器人操作中的泛化能力与样本效率，增强训练稳定性。**

- **链接: [http://arxiv.org/pdf/2509.04970v1](http://arxiv.org/pdf/2509.04970v1)**

> **作者:** Tien Pham; Xinyun Chi; Khang Nguyen; Manfred Huber; Angelo Cangelosi
>
> **摘要:** Reinforcement learning (RL) agents can learn to solve complex tasks from visual inputs, but generalizing these learned skills to new environments remains a major challenge in RL application, especially robotics. While data augmentation can improve generalization, it often compromises sample efficiency and training stability. This paper introduces DeGuV, an RL framework that enhances both generalization and sample efficiency. In specific, we leverage a learnable masker network that produces a mask from the depth input, preserving only critical visual information while discarding irrelevant pixels. Through this, we ensure that our RL agents focus on essential features, improving robustness under data augmentation. In addition, we incorporate contrastive learning and stabilize Q-value estimation under augmentation to further enhance sample efficiency and training stability. We evaluate our proposed method on the RL-ViGen benchmark using the Franka Emika robot and demonstrate its effectiveness in zero-shot sim-to-real transfer. Our results show that DeGuV outperforms state-of-the-art methods in both generalization and sample efficiency while also improving interpretability by highlighting the most relevant regions in the visual input
>
---
#### [new 017] Hierarchical Reduced-Order Model Predictive Control for Robust Locomotion on Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文提出分层降阶MPC框架，解决人形机器人在复杂地形中的稳健行走问题，通过优化步态与整合上肢躯干动力学提升稳定性，经仿真与实验验证有效。**

- **链接: [http://arxiv.org/pdf/2509.04722v1](http://arxiv.org/pdf/2509.04722v1)**

> **作者:** Adrian B. Ghansah; Sergio A. Esteban; Aaron D. Ames
>
> **备注:** 8 pages, 6 figures, accepted to IEEE-RAS International Conference on Humanoid Robots 2025
>
> **摘要:** As humanoid robots enter real-world environments, ensuring robust locomotion across diverse environments is crucial. This paper presents a computationally efficient hierarchical control framework for humanoid robot locomotion based on reduced-order models -- enabling versatile step planning and incorporating arm and torso dynamics to better stabilize the walking. At the high level, we use the step-to-step dynamics of the ALIP model to simultaneously optimize over step periods, step lengths, and ankle torques via nonlinear MPC. The ALIP trajectories are used as references to a linear MPC framework that extends the standard SRB-MPC to also include simplified arm and torso dynamics. We validate the performance of our approach through simulation and hardware experiments on the Unitree G1 humanoid robot. In the proposed framework the high-level step planner runs at 40 Hz and the mid-level MPC at 500 Hz using the onboard mini-PC. Adaptive step timing increased the push recovery success rate by 36%, and the upper body control improved the yaw disturbance rejection. We also demonstrate robust locomotion across diverse indoor and outdoor terrains, including grass, stone pavement, and uneven gym mats.
>
---
#### [new 018] Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 本论文针对3D点云分类任务，解决ModelNet40数据集的标签不一致、2D数据等问题，提出改进数据集ModelNet-R和轻量网络Point-SkipNet，提升分类准确率并降低计算开销。**

- **链接: [http://arxiv.org/pdf/2509.05198v1](http://arxiv.org/pdf/2509.05198v1)**

> **作者:** Mohammad Saeid; Amir Salarpour; Pedram MohajerAnsari
>
> **备注:** This paper has been accepted for presentation at the 7th International Conference on Pattern Recognition and Image Analysis (IPRIA 2025)
>
> **摘要:** The classification of 3D point clouds is crucial for applications such as autonomous driving, robotics, and augmented reality. However, the commonly used ModelNet40 dataset suffers from limitations such as inconsistent labeling, 2D data, size mismatches, and inadequate class differentiation, which hinder model performance. This paper introduces ModelNet-R, a meticulously refined version of ModelNet40 designed to address these issues and serve as a more reliable benchmark. Additionally, this paper proposes Point-SkipNet, a lightweight graph-based neural network that leverages efficient sampling, neighborhood grouping, and skip connections to achieve high classification accuracy with reduced computational overhead. Extensive experiments demonstrate that models trained in ModelNet-R exhibit significant performance improvements. Notably, Point-SkipNet achieves state-of-the-art accuracy on ModelNet-R with a substantially lower parameter count compared to contemporary models. This research highlights the crucial role of dataset quality in optimizing model efficiency for 3D point cloud classification. For more details, see the code at: https://github.com/m-saeid/ModeNetR_PointSkipNet.
>
---
#### [new 019] Domain Adaptation for Different Sensor Configurations in 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶中不同传感器配置导致的3D目标检测领域偏差问题，提出下游微调与部分层微调方法，通过联合训练提升跨配置模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.04711v1](http://arxiv.org/pdf/2509.04711v1)**

> **作者:** Satoshi Tanaka; Kok Seang Tan; Isamu Yamashita
>
> **摘要:** Recent advances in autonomous driving have underscored the importance of accurate 3D object detection, with LiDAR playing a central role due to its robustness under diverse visibility conditions. However, different vehicle platforms often deploy distinct sensor configurations, causing performance degradation when models trained on one configuration are applied to another because of shifts in the point cloud distribution. Prior work on multi-dataset training and domain adaptation for 3D object detection has largely addressed environmental domain gaps and density variation within a single LiDAR; in contrast, the domain gap for different sensor configurations remains largely unexplored. In this work, we address domain adaptation across different sensor configurations in 3D object detection. We propose two techniques: Downstream Fine-tuning (dataset-specific fine-tuning after multi-dataset training) and Partial Layer Fine-tuning (updating only a subset of layers to improve cross-configuration generalization). Using paired datasets collected in the same geographic region with multiple sensor configurations, we show that joint training with Downstream Fine-tuning and Partial Layer Fine-tuning consistently outperforms naive joint training for each configuration. Our findings provide a practical and scalable solution for adapting 3D object detection models to the diverse vehicle platforms.
>
---
#### [new 020] Analyzing Gait Adaptation with Hemiplegia Simulation Suits and Digital Twins
- **分类: cs.ET; cs.RO**

- **简介: 该论文旨在通过半瘫痪模拟服和数字孪生技术研究步态适应，解决早期康复机器人测试的安全风险。研究收集生物力学数据，构建数字孪生模型，利用机器学习分析步态变化及人-辅助设备交互，识别关键传感器模态以优化康复机器人设计。**

- **链接: [http://arxiv.org/pdf/2509.05116v1](http://arxiv.org/pdf/2509.05116v1)**

> **作者:** Jialin Chen; Jeremie Clos; Dominic Price; Praminda Caleb-Solly
>
> **备注:** 7 pages, accepted at EMBC 2025, presented at the conference
>
> **摘要:** To advance the development of assistive and rehabilitation robots, it is essential to conduct experiments early in the design cycle. However, testing early prototypes directly with users can pose safety risks. To address this, we explore the use of condition-specific simulation suits worn by healthy participants in controlled environments as a means to study gait changes associated with various impairments and support rapid prototyping. This paper presents a study analyzing the impact of a hemiplegia simulation suit on gait. We collected biomechanical data using a Vicon motion capture system and Delsys Trigno EMG and IMU sensors under four walking conditions: with and without a rollator, and with and without the simulation suit. The gait data was integrated into a digital twin model, enabling machine learning analyses to detect the use of the simulation suit and rollator, identify turning behavior, and evaluate how the suit affects gait over time. Our findings show that the simulation suit significantly alters movement and muscle activation patterns, prompting users to compensate with more abrupt motions. We also identify key features and sensor modalities that are most informative for accurately capturing gait dynamics and modeling human-rollator interaction within the digital twin framework.
>
---
#### [new 021] Language-Driven Hierarchical Task Structures as Explicit World Models for Multi-Agent Learning
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO; 68T05, 90C40, 91A26, 68T42, 93E35; I.2.11; I.2.6; I.2.8; I.2.9; I.2.7**

- **简介: 该论文提出通过语言驱动的层次化任务结构构建显式环境模型，解决多智能体学习中因复杂任务和稀疏奖励导致的探索效率低问题，利用大语言模型动态生成任务分层框架，提升智能体策略学习效率。**

- **链接: [http://arxiv.org/pdf/2509.04731v1](http://arxiv.org/pdf/2509.04731v1)**

> **作者:** Brennen Hill
>
> **摘要:** The convergence of Language models, Agent models, and World models represents a critical frontier for artificial intelligence. While recent progress has focused on scaling Language and Agent models, the development of sophisticated, explicit World Models remains a key bottleneck, particularly for complex, long-horizon multi-agent tasks. In domains such as robotic soccer, agents trained via standard reinforcement learning in high-fidelity but structurally-flat simulators often fail due to intractable exploration spaces and sparse rewards. This position paper argues that the next frontier in developing capable agents lies in creating environments that possess an explicit, hierarchical World Model. We contend that this is best achieved through hierarchical scaffolding, where complex goals are decomposed into structured, manageable subgoals. Drawing evidence from a systematic review of 2024 research in multi-agent soccer, we identify a clear and decisive trend towards integrating symbolic and hierarchical methods with multi-agent reinforcement learning (MARL). These approaches implicitly or explicitly construct a task-based world model to guide agent learning. We then propose a paradigm shift: leveraging Large Language Models to dynamically generate this hierarchical scaffold, effectively using language to structure the World Model on the fly. This language-driven world model provides an intrinsic curriculum, dense and meaningful learning signals, and a framework for compositional learning, enabling Agent Models to acquire sophisticated, strategic behaviors with far greater sample efficiency. By building environments with explicit, language-configurable task layers, we can bridge the gap between low-level reactive behaviors and high-level strategic team play, creating a powerful and generalizable framework for training the next generation of intelligent agents.
>
---
#### [new 022] PRREACH: Probabilistic Risk Assessment Using Reachability for UAV Control
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文提出PRReach方法，通过可达性分析实现无人机风险概率评估，优化控制策略以降低风险，解决现有框架数据不足及缺乏风险缓解的问题，实现实时与离线风险控制。**

- **链接: [http://arxiv.org/pdf/2509.04451v1](http://arxiv.org/pdf/2509.04451v1)**

> **作者:** Nicole Fronda; Hariharan Narayanan; Sadia Afrin Ananna; Steven Weber; Houssam Abbas
>
> **备注:** Accepted to IEEE International Conference on Intelligent Transportation Systems (ITSC) 2025
>
> **摘要:** We present a new approach for designing risk-bounded controllers for Uncrewed Aerial Vehicles (UAVs). Existing frameworks for assessing risk of UAV operations rely on knowing the conditional probability of an incident occurring given different causes. Limited data for computing these probabilities makes real-world implementation of these frameworks difficult. Furthermore, existing frameworks do not include control methods for risk mitigation. Our approach relies on UAV dynamics, and employs reachability analysis for a probabilistic risk assessment over all feasible UAV trajectories. We use this holistic risk assessment to formulate a control optimization problem that minimally changes a UAV's existing control law to be bounded by an accepted risk threshold. We call our approach PRReach. Public and readily available UAV dynamics models and open source spatial data for mapping hazard outcomes enables practical implementation of PRReach for both offline pre-flight and online in-flight risk assessment and mitigation. We evaluate PRReach through simulation experiments on real-world data. Results show that PRReach controllers reduce risk by up to 24% offline, and up to 53% online from classical controllers.
>
---
#### [new 023] UAV-Based Intelligent Traffic Surveillance System: Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis
- **分类: cs.CV; cs.ET; cs.RO; cs.SY; eess.IV; eess.SY**

- **简介: 论文提出基于无人机的智能交通监控系统，解决传统方法覆盖不足、适应性差的问题，实现实时车辆检测、分类、跟踪及违规行为分析，通过多技术融合和案例验证系统有效性。**

- **链接: [http://arxiv.org/pdf/2509.04624v1](http://arxiv.org/pdf/2509.04624v1)**

> **作者:** Ali Khanpour; Tianyi Wang; Afra Vahidi-Shams; Wim Ectors; Farzam Nakhaie; Amirhossein Taheri; Christian Claudel
>
> **备注:** 15 pages, 8 figures, 2 tables
>
> **摘要:** Traffic congestion and violations pose significant challenges for urban mobility and road safety. Traditional traffic monitoring systems, such as fixed cameras and sensor-based methods, are often constrained by limited coverage, low adaptability, and poor scalability. To address these challenges, this paper introduces an advanced unmanned aerial vehicle (UAV)-based traffic surveillance system capable of accurate vehicle detection, classification, tracking, and behavioral analysis in real-world, unconstrained urban environments. The system leverages multi-scale and multi-angle template matching, Kalman filtering, and homography-based calibration to process aerial video data collected from altitudes of approximately 200 meters. A case study in urban area demonstrates robust performance, achieving a detection precision of 91.8%, an F1-score of 90.5%, and tracking metrics (MOTA/MOTP) of 92.1% and 93.7%, respectively. Beyond precise detection, the system classifies five vehicle types and automatically detects critical traffic violations, including unsafe lane changes, illegal double parking, and crosswalk obstructions, through the fusion of geofencing, motion filtering, and trajectory deviation analysis. The integrated analytics module supports origin-destination tracking, vehicle count visualization, inter-class correlation analysis, and heatmap-based congestion modeling. Additionally, the system enables entry-exit trajectory profiling, vehicle density estimation across road segments, and movement direction logging, supporting comprehensive multi-scale urban mobility analytics. Experimental results confirms the system's scalability, accuracy, and practical relevance, highlighting its potential as an enforcement-aware, infrastructure-independent traffic monitoring solution for next-generation smart cities.
>
---
#### [new 024] The best approximation pair problem relative to two subsets in a normed space
- **分类: math.OC; cs.GR; cs.RO; math.FA; math.MG; 41A50, 41A52, 41A65, 90C25, 46N10, 90C26, 46B20, 68U05, 65D18; G.1.6; G.1.2; I.3.5**

- **简介: 该论文研究赋范空间中两个子集的最佳逼近对问题，探讨解的存在性与唯一性条件，提出几何结构相关的新条件，并扩展了现有算法的应用范围。**

- **链接: [http://arxiv.org/pdf/2403.18767v2](http://arxiv.org/pdf/2403.18767v2)**

> **作者:** Daniel Reem; Yair Censor
>
> **备注:** Major revision, Introduction and abstract were rewritten, added Theorem 4.8 and Remark 4(ii), minor changes here and there such as in MSC and in the proof of Lemma 3.2 and in Theorem 5.1 (added one sufficient condition, two were removed), Remark 5.2 was extended, added several figures and many more references, added acknowledgements
>
> **摘要:** In the classical best approximation pair (BAP) problem, one is given two nonempty, closed, convex and disjoint subsets in a finite- or an infinite-dimensional Hilbert space, and the goal is to find a pair of points, each from each subset, which realizes the distance between the subsets. We discuss the problem in more general normed spaces and with possibly non-convex subsets, and focus our attention on the issues of uniqueness and existence of the solution to the problem. As far as we know, these fundamental issues have not received much attention. We present several sufficient geometric conditions for the (at most) uniqueness of a BAP. These conditions are related to the structure and the relative orientation of the boundaries of the subsets and to the norm. We also present many sufficient conditions for the existence of a BAP. Our results significantly extend the horizon of a recent algorithm for solving the BAP problem [Censor, Mansour, Reem, J. Approx. Theory (2024)]. The paper also shows, perhaps for the first time, how wide is the scope of the BAP problem in terms of the scientific communities which are involved in it (frequently independently) and in terms of its applications.
>
---
## 更新

#### [replaced 001] HyperTASR: Hypernetwork-Driven Task-Aware Scene Representations for Robust Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.18802v2](http://arxiv.org/pdf/2508.18802v2)**

> **作者:** Li Sun; Jiefeng Wu; Feng Chen; Ruizhe Liu; Yanchao Yang
>
> **摘要:** Effective policy learning for robotic manipulation requires scene representations that selectively capture task-relevant environmental features. Current approaches typically employ task-agnostic representation extraction, failing to emulate the dynamic perceptual adaptation observed in human cognition. We present HyperTASR, a hypernetwork-driven framework that modulates scene representations based on both task objectives and the execution phase. Our architecture dynamically generates representation transformation parameters conditioned on task specifications and progression state, enabling representations to evolve contextually throughout task execution. This approach maintains architectural compatibility with existing policy learning frameworks while fundamentally reconfiguring how visual features are processed. Unlike methods that simply concatenate or fuse task embeddings with task-agnostic representations, HyperTASR establishes computational separation between task-contextual and state-dependent processing paths, enhancing learning efficiency and representational quality. Comprehensive evaluations in both simulation and real-world environments demonstrate substantial performance improvements across different representation paradigms. Through ablation studies and attention visualization, we confirm that our approach selectively prioritizes task-relevant scene information, closely mirroring human adaptive perception during manipulation tasks. The project website is at [HyperTASR](https://lisunphil.github.io/HyperTASR_projectpage/ "lisunphil.github.io/HyperTASR_projectpage/").
>
---
#### [replaced 002] InteLiPlan: An Interactive Lightweight LLM-Based Planner for Domestic Robot Autonomy
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.14506v3](http://arxiv.org/pdf/2409.14506v3)**

> **作者:** Kim Tien Ly; Kai Lu; Ioannis Havoutis
>
> **摘要:** We introduce an interactive LLM-based framework designed to enhance the autonomy and robustness of domestic robots, targeting embodied intelligence. Our approach reduces reliance on large-scale data and incorporates a robot-agnostic pipeline that embodies an LLM. Our framework, InteLiPlan, ensures that the LLM's decision-making capabilities are effectively aligned with robotic functions, enhancing operational robustness and adaptability, while our human-in-the-loop mechanism allows for real-time human intervention when user instruction is required. We evaluate our method in both simulation and on the real Toyota Human Support Robot and Anymal D-Unitree Z1 platforms. Our method achieves a 95% success rate in the 'fetch me' task completion with failure recovery, highlighting its capability in both failure reasoning and task planning. InteLiPlan achieves comparable performance to state-of-the-art large-scale LLM-based robotics planners, while using only real-time onboard computing.
>
---
#### [replaced 003] IA-TIGRIS: An Incremental and Adaptive Sampling-Based Planner for Online Informative Path Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.15961v2](http://arxiv.org/pdf/2502.15961v2)**

> **作者:** Brady Moon; Nayana Suvarna; Andrew Jong; Satrajit Chatterjee; Junbin Yuan; Muqing Cao; Sebastian Scherer
>
> **备注:** 18 pages, 19 figures
>
> **摘要:** Planning paths that maximize information gain for robotic platforms has wide-ranging applications and significant potential impact. To effectively adapt to real-time data collection, informative path planning must be computed online and be responsive to new observations. In this work, we present IA-TIGRIS, an incremental and adaptive sampling-based informative path planner designed for real-time onboard execution. Our approach leverages past planning efforts through incremental refinement while continuously adapting to updated belief maps. We additionally present detailed implementation and optimization insights to facilitate real-world deployment, along with an array of reward functions tailored to specific missions and behaviors. Extensive simulation results demonstrate IA-TIGRIS generates higher-quality paths compared to baseline methods. We validate our planner on two distinct hardware platforms: a hexarotor UAV and a fixed-wing UAV, each having different motion models and configuration spaces. Our results show up to a 41% improvement in information gain compared to baseline methods, highlighting the planner's potential for deployment in real-world applications. Project website and video: https://ia-tigris.github.io
>
---
#### [replaced 004] Imitating and Finetuning Model Predictive Control for Robust and Symmetric Quadrupedal Locomotion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2311.02304v2](http://arxiv.org/pdf/2311.02304v2)**

> **作者:** Donghoon Youm; Hyunyoung Jung; Hyeongjun Kim; Jemin Hwangbo; Hae-Won Park; Sehoon Ha
>
> **摘要:** Control of legged robots is a challenging problem that has been investigated by different approaches, such as model-based control and learning algorithms. This work proposes a novel Imitating and Finetuning Model Predictive Control (IFM) framework to take the strengths of both approaches. Our framework first develops a conventional model predictive controller (MPC) using Differential Dynamic Programming and Raibert heuristic, which serves as an expert policy. Then we train a clone of the MPC using imitation learning to make the controller learnable. Finally, we leverage deep reinforcement learning with limited exploration for further finetuning the policy on more challenging terrains. By conducting comprehensive simulation and hardware experiments, we demonstrate that the proposed IFM framework can significantly improve the performance of the given MPC controller on rough, slippery, and conveyor terrains that require careful coordination of footsteps. We also showcase that IFM can efficiently produce more symmetric, periodic, and energy-efficient gaits compared to Vanilla RL with a minimal burden of reward shaping.
>
---
#### [replaced 005] Teleoperation of Continuum Instruments: Task-Priority Analysis of Linear Angular Command Interplay
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.06035v3](http://arxiv.org/pdf/2412.06035v3)**

> **作者:** Ehsan Nasiri; Long Wang
>
> **备注:** 27 pages (single Column Version), published by ASME Journal of Mechanisms and Robotics,2025
>
> **摘要:** This paper addresses the challenge of teleoperating continuum instruments for minimally invasive surgery (MIS). We develop and adopt a novel task-priority-based kinematic formulation to quantitatively investigate teleoperation commands for continuum instruments under remote center of motion (RCM) constraints. Using redundancy resolution methods, we investigate the kinematic performance during teleoperation, comparing linear and angular commands within a task-priority scheme. For experimental validation, an instrument module (IM) was designed and integrated with a 7-DoF manipulator. Assessments, simulations, and experimental validations demonstrated the effectiveness of the proposed framework. The experiments involved several tasks: trajectory tracking of the IM tip along multiple paths with varying priorities for linear and angular teleoperation commands, pushing a ball along predefined paths on a silicon board, following a pattern on a pegboard, and guiding the continuum tip through rings on a ring board using a standard surgical kit.
>
---
#### [replaced 006] Behavior Synthesis via Contact-Aware Fisher Information Maximization
- **分类: cs.RO; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2505.12214v2](http://arxiv.org/pdf/2505.12214v2)**

> **作者:** Hrishikesh Sathyanarayan; Ian Abraham
>
> **备注:** In Robotics Science and Systems 2025
>
> **摘要:** Contact dynamics hold immense amounts of information that can improve a robot's ability to characterize and learn about objects in their environment through interactions. However, collecting information-rich contact data is challenging due to its inherent sparsity and non-smooth nature, requiring an active approach to maximize the utility of contacts for learning. In this work, we investigate an optimal experimental design approach to synthesize robot behaviors that produce contact-rich data for learning. Our approach derives a contact-aware Fisher information measure that characterizes information-rich contact behaviors that improve parameter learning. We observe emergent robot behaviors that are able to excite contact interactions that efficiently learns object parameters across a range of parameter learning examples. Last, we demonstrate the utility of contact-awareness for learning parameters through contact-seeking behaviors on several robotic experiments.
>
---
#### [replaced 007] Align-Then-stEer: Adapting the Vision-Language Action Models through Unified Latent Guidance
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.02055v2](http://arxiv.org/pdf/2509.02055v2)**

> **作者:** Yang Zhang; Chenwei Wang; Ouyang Lu; Yuan Zhao; Yunfei Ge; Zhenglong Sun; Xiu Li; Chi Zhang; Chenjia Bai; Xuelong Li
>
> **备注:** The first three authors contributed equally
>
> **摘要:** Vision-Language-Action (VLA) models pre-trained on large, diverse datasets show remarkable potential for general-purpose robotic manipulation. However, a primary bottleneck remains in adapting these models to downstream tasks, especially when the robot's embodiment or the task itself differs from the pre-training data. This discrepancy leads to a significant mismatch in action distributions, demanding extensive data and compute for effective fine-tuning. To address this challenge, we introduce \textbf{Align-Then-stEer (\texttt{ATE})}, a novel, data-efficient, and plug-and-play adaptation framework. \texttt{ATE} first aligns disparate action spaces by constructing a unified latent space, where a variational autoencoder constrained by reverse KL divergence embeds adaptation actions into modes of the pre-training action latent distribution. Subsequently, it steers the diffusion- or flow-based VLA's generation process during fine-tuning via a guidance mechanism that pushes the model's output distribution towards the target domain. We conduct extensive experiments on cross-embodiment and cross-task manipulation in both simulation and real world. Compared to direct fine-tuning of representative VLAs, our method improves the average multi-task success rate by up to \textbf{9.8\%} in simulation and achieves a striking \textbf{32\% success rate gain} in a real-world cross-embodiment setting. Our work presents a general and lightweight solution that greatly enhances the practicality of deploying VLA models to new robotic platforms and tasks.
>
---
#### [replaced 008] Sensing environmental physical interaction to traverse cluttered obstacles
- **分类: cs.RO; cs.SY; eess.SY; physics.bio-ph**

- **链接: [http://arxiv.org/pdf/2401.13062v3](http://arxiv.org/pdf/2401.13062v3)**

> **作者:** Yaqing Wang; Ling Xu; Chen Li
>
> **摘要:** The long-standing, dominant approach to robotic obstacle negotiation relies on mapping environmental geometry to avoid obstacles. However, this approach does not allow for traversal of cluttered obstacles, hindering applications such as search and rescue operations through earthquake rubble and exploration across lunar and Martian rocks. To overcome this challenge, robots must further sense and utilize environmental physical interactions to control themselves to traverse obstacles. Recently, a physics-based approach has been established towards this vision. Self-propelled robots interacting with obstacles results in a potential energy landscape. On this landscape, to traverse obstacles, a robot must escape from certain landscape basins that attract it into failure modes, to reach other basins that lead to successful modes. Thus, sensing the potential energy landscape is crucial. Here, we developed new methods and performed systematic experiments to demonstrate that the potential energy landscape can be estimated by sensing environmental physical interaction. We developed a minimalistic robot capable of sensing obstacle contact forces and torques for systematic experiments over a wide range of parameter space. Surprisingly, although these forces and torques are not fully conservative, they match the potential energy landscape gradients that are conservative forces and torques, enabling an accurate estimation of the potential energy landscape. Additionally, a bio-inspired strategy further enhanced estimation accuracy. Our results provided a foundation for further refining these methods for use in free-locomoting robots. Our study is a key step in establishing a new physics-based approach for robots to traverse clustered obstacles to advance their mobility in complex, real-world environments.
>
---
#### [replaced 009] Learning Multi-Stage Pick-and-Place with a Legged Mobile Manipulator
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.03859v2](http://arxiv.org/pdf/2509.03859v2)**

> **作者:** Haichao Zhang; Haonan Yu; Le Zhao; Andrew Choi; Qinxun Bai; Yiqing Yang; Wei Xu
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L). arXiv admin note: substantial text overlap with arXiv:2501.09905
>
> **摘要:** Quadruped-based mobile manipulation presents significant challenges in robotics due to the diversity of required skills, the extended task horizon, and partial observability. After presenting a multi-stage pick-and-place task as a succinct yet sufficiently rich setup that captures key desiderata for quadruped-based mobile manipulation, we propose an approach that can train a visuo-motor policy entirely in simulation, and achieve nearly 80\% success in the real world. The policy efficiently performs search, approach, grasp, transport, and drop into actions, with emerged behaviors such as re-grasping and task chaining. We conduct an extensive set of real-world experiments with ablation studies highlighting key techniques for efficient training and effective sim-to-real transfer. Additional experiments demonstrate deployment across a variety of indoor and outdoor environments. Demo videos and additional resources are available on the project page: https://horizonrobotics.github.io/gail/SLIM.
>
---
#### [replaced 010] Graph-based Decentralized Task Allocation for Multi-Robot Target Localization
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2309.08896v2](http://arxiv.org/pdf/2309.08896v2)**

> **作者:** Juntong Peng; Hrishikesh Viswanath; Aniket Bera
>
> **摘要:** We introduce a new graph neural operator-based approach for task allocation in a system of heterogeneous robots composed of Unmanned Ground Vehicles (UGVs) and Unmanned Aerial Vehicles (UAVs). The proposed model, \texttt{\method}, or \textbf{G}raph \textbf{A}ttention \textbf{T}ask \textbf{A}llocato\textbf{R} aggregates information from neighbors in the multi-robot system, with the aim of achieving globally optimal target localization. Being decentralized, our method is highly robust and adaptable to situations where the number of robots and the number of tasks may change over time. We also propose a heterogeneity-aware preprocessing technique to model the heterogeneity of the system. The experimental results demonstrate the effectiveness and scalability of the proposed approach in a range of simulated scenarios generated by varying the number of UGVs and UAVs and the number and location of the targets. We show that a single model can handle a heterogeneous robot team with a number of robots ranging between 2 and 12 while outperforming the baseline architectures.
>
---
#### [replaced 011] Exploring persuasive interactions with generative social robots: An experimental framework
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.03231v2](http://arxiv.org/pdf/2509.03231v2)**

> **作者:** Stephan Vonschallen; Larissa Julia Corina Finsler; Theresa Schmiedel; Friederike Eyssel
>
> **备注:** A shortened version of this paper was accepted as poster for the Thirteenth International Conference on Human-Agent Interaction (HAI2025)
>
> **摘要:** Integrating generative AI such as Large Language Models into social robots has improved their ability to engage in natural, human-like communication. This study presents a method to examine their persuasive capabilities. We designed an experimental framework focused on decision making and tested it in a pilot that varied robot appearance and self-knowledge. Using qualitative analysis, we evaluated interaction quality, persuasion effectiveness, and the robot's communicative strategies. Participants generally experienced the interaction positively, describing the robot as competent, friendly, and supportive, while noting practical limits such as delayed responses and occasional speech-recognition errors. Persuasiveness was highly context dependent and shaped by robot behavior: Participants responded well to polite, reasoned suggestions and expressive gestures, but emphasized the need for more personalized, context-aware arguments and clearer social roles. These findings suggest that generative social robots can influence user decisions, but their effectiveness depends on communicative nuance and contextual relevance. We propose refinements to the framework to further study persuasive dynamics between robots and human users.
>
---
#### [replaced 012] Controller Design and Implementation of a New Quadrotor Manipulation System
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/1904.08498v2](http://arxiv.org/pdf/1904.08498v2)**

> **作者:** Ahmed Khalifa
>
> **备注:** Ph.D. Thesis. Supervisors: Prof. Mohamed Fanni (EJUST), Prof. Toru Namerikawa (Keio University)
>
> **摘要:** The previously introduced aerial manipulation systems suffer from either limited end-effector DOF or small payload capacity. In this dissertation, a quadrotor with a 2-DOF manipulator is investigated that has a unique topology to enable the end-effector to track 6-DOF trajectory with the minimum possible number of actuators/links and hence, maximize the payload and/or mission time. The proposed system is designed, modeled, and constructed. An identification process is carried out to find the system parameters. An experimental setup is proposed with a 6-DOF state measurement and estimation scheme. The system feasibility is validated via numerical and experimental results. The inverse kinematics require a solution of complicated algebraic-differential equations. Therefore, an algorithm is developed to get an approximate solution of these equations.
>
---
#### [replaced 013] ActiveGAMER: Active GAussian Mapping through Efficient Rendering
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.06897v3](http://arxiv.org/pdf/2501.06897v3)**

> **作者:** Liyan Chen; Huangying Zhan; Kevin Chen; Xiangyu Xu; Qingan Yan; Changjiang Cai; Yi Xu
>
> **备注:** Accepted to CVPR2025. Project page: https://oppo-us-research.github.io/ActiveGAMER-website/. Code: https://github.com/oppo-us-research/ActiveGAMER
>
> **摘要:** We introduce ActiveGAMER, an active mapping system that utilizes 3D Gaussian Splatting (3DGS) to achieve high-quality, real-time scene mapping and exploration. Unlike traditional NeRF-based methods, which are computationally demanding and restrict active mapping performance, our approach leverages the efficient rendering capabilities of 3DGS, allowing effective and efficient exploration in complex environments. The core of our system is a rendering-based information gain module that dynamically identifies the most informative viewpoints for next-best-view planning, enhancing both geometric and photometric reconstruction accuracy. ActiveGAMER also integrates a carefully balanced framework, combining coarse-to-fine exploration, post-refinement, and a global-local keyframe selection strategy to maximize reconstruction completeness and fidelity. Our system autonomously explores and reconstructs environments with state-of-the-art geometric and photometric accuracy and completeness, significantly surpassing existing approaches in both aspects. Extensive evaluations on benchmark datasets such as Replica and MP3D highlight ActiveGAMER's effectiveness in active mapping tasks.
>
---
#### [replaced 014] Find Everything: A General Vision Language Model Approach to Multi-Object Search
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.00388v3](http://arxiv.org/pdf/2410.00388v3)**

> **作者:** Daniel Choi; Angus Fung; Haitong Wang; Aaron Hao Tan
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** The Multi-Object Search (MOS) problem involves navigating to a sequence of locations to maximize the likelihood of finding target objects while minimizing travel costs. In this paper, we introduce a novel approach to the MOS problem, called Finder, which leverages vision language models (VLMs) to locate multiple objects across diverse environments. Specifically, our approach introduces multi-channel score maps to track and reason about multiple objects simultaneously during navigation, along with a score map technique that combines scene-level and object-level semantic correlations. Experiments in both simulated and real-world settings showed that Finder outperforms existing methods using deep reinforcement learning and VLMs. Ablation and scalability studies further validated our design choices and robustness with increasing numbers of target objects, respectively. Website: https://find-all-my-things.github.io/
>
---
#### [replaced 015] Train-Once Plan-Anywhere Kinodynamic Motion Planning via Diffusion Trees
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.21001v2](http://arxiv.org/pdf/2508.21001v2)**

> **作者:** Yaniv Hassidof; Tom Jurgenson; Kiril Solovey
>
> **备注:** Accepted to CoRL 2025, Project page: https://sites.google.com/view/ditree. v2: Abstract updated
>
> **摘要:** Kinodynamic motion planning is concerned with computing collision-free trajectories while abiding by the robot's dynamic constraints. This critical problem is often tackled using sampling-based planners (SBPs) that explore the robot's high-dimensional state space by constructing a search tree via action propagations. Although SBPs can offer global guarantees on completeness and solution quality, their performance is often hindered by slow exploration due to uninformed action sampling. Learning-based approaches can yield significantly faster runtimes, yet they fail to generalize to out-of-distribution (OOD) scenarios and lack critical guarantees, e.g., safety, thus limiting their deployment on physical robots. We present Diffusion Tree (DiTree): a provably-generalizable framework leveraging diffusion policies (DPs) as informed samplers to efficiently guide state-space search within SBPs. DiTree combines DP's ability to model complex distributions of expert trajectories, conditioned on local observations, with the completeness of SBPs to yield provably-safe solutions within a few action propagation iterations for complex dynamical systems. We demonstrate DiTree's power with an implementation combining the popular RRT planner with a DP action sampler trained on a single environment. In comprehensive evaluations on OOD scenarios, DiTree achieves on average a 30% higher success rate compared to standalone DP or SBPs, on a dynamic car and Mujoco's ant robot settings (for the latter, SBPs fail completely). Beyond simulation, real-world car experiments confirm DiTree's applicability, demonstrating superior trajectory quality and robustness even under severe sim-to-real gaps. Project webpage: https://sites.google.com/view/ditree.
>
---
#### [replaced 016] Multimodal LLM Guided Exploration and Active Mapping using Fisher Information
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.17422v3](http://arxiv.org/pdf/2410.17422v3)**

> **作者:** Wen Jiang; Boshu Lei; Katrina Ashton; Kostas Daniilidis
>
> **备注:** ICCV 2025
>
> **摘要:** We present an active mapping system that plans for both long-horizon exploration goals and short-term actions using a 3D Gaussian Splatting (3DGS) representation. Existing methods either do not take advantage of recent developments in multimodal Large Language Models (LLM) or do not consider challenges in localization uncertainty, which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based objective. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.
>
---
#### [replaced 017] Global Contact-Rich Planning with Sparsity-Rich Semidefinite Relaxations
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2502.02829v4](http://arxiv.org/pdf/2502.02829v4)**

> **作者:** Shucheng Kang; Guorui Liu; Heng Yang
>
> **备注:** Website: https://computationalrobotics.seas.harvard.edu/project-spot/
>
> **摘要:** We show that contact-rich motion planning is also sparsity-rich when viewed as polynomial optimization (POP). We can exploit not only the correlative and term sparsity patterns that are general to all POPs, but also specialized sparsity patterns from the robot kinematic structure and the separability of contact modes. Such sparsity enables the design of high-order but sparse semidefinite programming (SDPs) relaxations--building upon Lasserre's moment and sums of squares hierarchy--that (i) can be solved in seconds by off-the-shelf SDP solvers, and (ii) compute near globally optimal solutions to the nonconvex contact-rich planning problems with small certified suboptimality. Through extensive experiments both in simulation (Push Bot, Push Box, Push Box with Obstacles, and Planar Hand) and real world (Push T), we demonstrate the power of using convex SDP relaxations to generate global contact-rich motion plans. As a contribution of independent interest, we release the Sparse Polynomial Optimization Toolbox (SPOT)--implemented in C++ with interfaces to both Python and Matlab--that automates sparsity exploitation for robotics and beyond.
>
---
#### [replaced 018] Learning to Coordinate: Distributed Meta-Trajectory Optimization Via Differentiable ADMM-DDP
- **分类: cs.LG; cs.MA; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.01630v2](http://arxiv.org/pdf/2509.01630v2)**

> **作者:** Bingheng Wang; Yichao Gao; Tianchen Sun; Lin Zhao
>
> **摘要:** Distributed trajectory optimization via ADMM-DDP is a powerful approach for coordinating multi-agent systems, but it requires extensive tuning of tightly coupled hyperparameters that jointly govern local task performance and global coordination. In this paper, we propose Learning to Coordinate (L2C), a general framework that meta-learns these hyperparameters, modeled by lightweight agent-wise neural networks, to adapt across diverse tasks and agent configurations. L2C differentiates end-to-end through the ADMM-DDP pipeline in a distributed manner. It also enables efficient meta-gradient computation by reusing DDP components such as Riccati recursions and feedback gains. These gradients correspond to the optimal solutions of distributed matrix-valued LQR problems, coordinated across agents via an auxiliary ADMM framework that becomes convex under mild assumptions. Training is further accelerated by truncating iterations and meta-learning ADMM penalty parameters optimized for rapid residual reduction, with provable Lipschitz-bounded gradient errors. On a challenging cooperative aerial transport task, L2C generates dynamically feasible trajectories in high-fidelity simulation using IsaacSIM, reconfigures quadrotor formations for safe 6-DoF load manipulation in tight spaces, and adapts robustly to varying team sizes and task conditions, while achieving up to $88\%$ faster gradient computation than state-of-the-art methods.
>
---
#### [replaced 019] ApexNav: An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.14478v3](http://arxiv.org/pdf/2504.14478v3)**

> **作者:** Mingjie Zhang; Yuheng Du; Chengkai Wu; Jinni Zhou; Zhenchao Qi; Jun Ma; Boyu Zhou
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RAL), August, 2025
>
> **摘要:** Navigating unknown environments to find a target object is a significant challenge. While semantic information is crucial for navigation, relying solely on it for decision-making may not always be efficient, especially in environments with weak semantic cues. Additionally, many methods are susceptible to misdetections, especially in environments with visually similar objects. To address these limitations, we propose ApexNav, a zero-shot object navigation framework that is both more efficient and reliable. For efficiency, ApexNav adaptively utilizes semantic information by analyzing its distribution in the environment, guiding exploration through semantic reasoning when cues are strong, and switching to geometry-based exploration when they are weak. For reliability, we propose a target-centric semantic fusion method that preserves long-term memory of the target and similar objects, enabling robust object identification even under noisy detections. We evaluate ApexNav on the HM3Dv1, HM3Dv2, and MP3D datasets, where it outperforms state-of-the-art methods in both SR and SPL metrics. Comprehensive ablation studies further demonstrate the effectiveness of each module. Furthermore, real-world experiments validate the practicality of ApexNav in physical environments. The code will be released at https://github.com/Robotics-STAR-Lab/ApexNav.
>
---
#### [replaced 020] Enhanced Mean Field Game for Interactive Decision-Making with Varied Stylish Multi-Vehicles
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.00981v2](http://arxiv.org/pdf/2509.00981v2)**

> **作者:** Liancheng Zheng; Zhen Tian; Yangfan He; Shuo Liu; Huilin Chen; Fujiang Yuan; Yanhong Peng
>
> **摘要:** This paper presents an MFG-based decision-making framework for autonomous driving in heterogeneous traffic. To capture diverse human behaviors, we propose a quantitative driving style representation that maps abstract traits to parameters such as speed, safety factors, and reaction time. These parameters are embedded into the MFG through a spatial influence field model. To ensure safe operation in dense traffic, we introduce a safety-critical lane-changing algorithm that leverages dynamic safety margins, time-to-collision analysis, and multi-layered constraints. Real-world NGSIM data is employed for style calibration and empirical validation. Experimental results demonstrate zero collisions across six style combinations, two 15-vehicle scenarios, and NGSIM-based trials, consistently outperforming conventional game-theoretic baselines. Overall, our approach provides a scalable, interpretable, and behavior-aware planning framework for real-world autonomous driving applications.
>
---
#### [replaced 021] Cutting Sequence Diffuser: Sim-to-Real Transferable Planning for Object Shaping by Grinding
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.14417v2](http://arxiv.org/pdf/2412.14417v2)**

> **作者:** Takumi Hachimine; Jun Morimoto; Takamitsu Matsubara
>
> **备注:** 8 pages, Accepted by Robotics and Automation Letter
>
> **摘要:** Automating object shaping by grinding with a robot is a crucial industrial process that involves removing material with a rotating grinding belt. This process generates removal resistance depending on such process conditions as material type, removal volume, and robot grinding posture, all of which complicate the analytical modeling of shape transitions. Additionally, a data-driven approach based on real-world data is challenging due to high data collection costs and the irreversible nature of the process. This paper proposes a Cutting Sequence Diffuser (CSD) for object shaping by grinding. The CSD, which only requires simple simulation data for model learning, offers an efficient way to plan long-horizon action sequences transferable to the real world. Our method designs a smooth action space with constrained small removal volumes to suppress the complexity of the shape transitions caused by removal resistance, thus reducing the reality gap in simulations. Moreover, by using a diffusion model to generate long-horizon action sequences, our approach reduces the planning time and allows for grinding the target shape while adhering to the constraints of a small removal volume per step. Through evaluations in both simulation and real robot experiments, we confirmed that our CSD was effective for grinding to different materials and various target shapes in a short time.
>
---
