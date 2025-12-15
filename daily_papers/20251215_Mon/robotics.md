# 机器人 cs.RO

- **最新发布 27 篇**

- **更新 23 篇**

## 最新发布

#### [new 001] BLURR: A Boosted Low-Resource Inference for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型推理过重的问题，提出BLURR轻量推理框架。无需重训练，通过缓存、混合精度和单步预测降低计算开销，在保持任务成功率的同时提升推理效率，适用于低资源部署场景。**

- **链接: [https://arxiv.org/pdf/2512.11769v1](https://arxiv.org/pdf/2512.11769v1)**

> **作者:** Xiaoyu Ma; Zhengqing Yuan; Zheyuan Zhang; Kaiwen Shi; Lichao Sun; Yanfang Ye
>
> **备注:** 10 pages, 3 figures. Code and integration scripts will be released at this http URL: https://github.com/JijiKing-Sam/BLURR-A-Boosted-Low-Resource-Inference-for-Vision-Language-Action-Model
>
> **摘要:** Vision-language-action (VLA) models enable impressive zero shot manipulation, but their inference stacks are often too heavy for responsive web demos or high frequency robot control on commodity GPUs. We present BLURR, a lightweight inference wrapper that can be plugged into existing VLA controllers without retraining or changing model checkpoints. Instantiated on the pi-zero VLA controller, BLURR keeps the original observation interfaces and accelerates control by combining an instruction prefix key value cache, mixed precision execution, and a single step rollout schedule that reduces per step computation. In our SimplerEnv based evaluation, BLURR maintains task success rates comparable to the original controller while significantly lowering effective FLOPs and wall clock latency. We also build an interactive web demo that allows users to switch between controllers and toggle inference options in real time while watching manipulation episodes. This highlights BLURR as a practical approach for deploying modern VLA policies under tight compute budgets.
>
---
#### [new 002] Incremental Validation of Automated Driving Functions using Generic Volumes in Micro- Operational Design Domains
- **分类: cs.RO; eess.SY**

- **简介: 该论文属自动驾驶验证任务，旨在解决感知系统在真实场景中测试不完整的问题。提出将运行设计域细分为微ODD，用抽象障碍物生成测试案例，结合仿真验证感知性能，提升测试系统性和安全性论证。**

- **链接: [https://arxiv.org/pdf/2512.11351v1](https://arxiv.org/pdf/2512.11351v1)**

> **作者:** Steffen Schäfer; Martin Cichon
>
> **摘要:** The validation of highly automated, perception-based driving systems must ensure that they function correctly under the full range of real-world conditions. Scenario-based testing is a prominent approach to addressing this challenge, as it involves the systematic simulation of objects and environments. Operational Design Domains (ODDs) are usually described using a taxonomy of qualitative designations for individual objects. However, the process of transitioning from taxonomy to concrete test cases remains unstructured, and completeness is theoretical. This paper introduces a structured method of subdividing the ODD into manageable sections, termed micro-ODDs (mODDs), and deriving test cases with abstract object representations. This concept is demonstrated using a one-dimensional, laterally guided manoeuvre involving a shunting locomotive within a constrained ODD. In this example, mODDs are defined and refined into narrow taxonomies that enable test case generation. Obstacles are represented as generic cubes of varying sizes, providing a simplified yet robust means of evaluating perception performance. A series of tests were conducted in a closed-loop, co-simulated virtual environment featuring photorealistic rendering and simulated LiDAR, GNSS and camera sensors. The results demonstrate how edge cases in obstacle detection can be systematically explored and how perception quality can be evaluated based on observed vehicle behaviour, using crash versus safe stop as the outcome metrics. These findings support the development of a standardised framework for safety argumentation and offer a practical step towards the validation and authorisation of automated driving functions.
>
---
#### [new 003] Bench-Push: Benchmarking Pushing-based Navigation and Manipulation Tasks for Mobile Robots
- **分类: cs.RO**

- **简介: 该论文针对移动机器人在复杂环境中需通过推动物体完成任务的问题，提出首个统一基准Bench-Push，包含多种仿真环境、评估指标及基线测试，支持可复现与跨方法比较，推动推挤式导航与操作技术发展。**

- **链接: [https://arxiv.org/pdf/2512.11736v1](https://arxiv.org/pdf/2512.11736v1)**

> **作者:** Ninghan Zhong; Steven Caro; Megnath Ramesh; Rishi Bhatnagar; Avraiem Iskandar; Stephen L. Smith
>
> **备注:** Under review for ICRA 2026
>
> **摘要:** Mobile robots are increasingly deployed in cluttered environments with movable objects, posing challenges for traditional methods that prohibit interaction. In such settings, the mobile robot must go beyond traditional obstacle avoidance, leveraging pushing or nudging strategies to accomplish its goals. While research in pushing-based robotics is growing, evaluations rely on ad hoc setups, limiting reproducibility and cross-comparison. To address this, we present Bench-Push, the first unified benchmark for pushing-based mobile robot navigation and manipulation tasks. Bench-Push includes multiple components: 1) a comprehensive range of simulated environments that capture the fundamental challenges in pushing-based tasks, including navigating a maze with movable obstacles, autonomous ship navigation in ice-covered waters, box delivery, and area clearing, each with varying levels of complexity; 2) novel evaluation metrics to capture efficiency, interaction effort, and partial task completion; and 3) demonstrations using Bench-Push to evaluate example implementations of established baselines across environments. Bench-Push is open-sourced as a Python library with a modular design. The code, documentation, and trained models can be found at https://github.com/IvanIZ/BenchNPIN.
>
---
#### [new 004] Seeing to Act, Prompting to Specify: A Bayesian Factorization of Vision Language Action Policy
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉-语言-动作（VLA）模型的泛化问题，解决因模态不平衡导致的语言遗忘。提出BayesVLA，通过贝叶斯分解将策略分为视觉-动作先验和语言条件似然，提升指令跟随与跨场景泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.11218v1](https://arxiv.org/pdf/2512.11218v1)**

> **作者:** Kechun Xu; Zhenjie Zhu; Anzhe Chen; Shuqi Zhao; Qing Huang; Yifei Yang; Haojian Lu; Rong Xiong; Masayoshi Tomizuka; Yue Wang
>
> **摘要:** The pursuit of out-of-distribution generalization in Vision-Language-Action (VLA) models is often hindered by catastrophic forgetting of the Vision-Language Model (VLM) backbone during fine-tuning. While co-training with external reasoning data helps, it requires experienced tuning and data-related overhead. Beyond such external dependencies, we identify an intrinsic cause within VLA datasets: modality imbalance, where language diversity is much lower than visual and action diversity. This imbalance biases the model toward visual shortcuts and language forgetting. To address this, we introduce BayesVLA, a Bayesian factorization that decomposes the policy into a visual-action prior, supporting seeing-to-act, and a language-conditioned likelihood, enabling prompt-to-specify. This inherently preserves generalization and promotes instruction following. We further incorporate pre- and post-contact phases to better leverage pre-trained foundation models. Information-theoretic analysis formally validates our effectiveness in mitigating shortcut learning. Extensive experiments show superior generalization to unseen instructions, objects, and environments compared to existing methods. Project page is available at: https://xukechun.github.io/papers/BayesVLA.
>
---
#### [new 005] Cross-Entropy Optimization of Physically Grounded Task and Motion Plans
- **分类: cs.RO**

- **简介: 该论文研究任务与运动规划（TAMP），旨在解决传统方法忽略动力学和接触影响导致计划不可行的问题。作者利用GPU并行物理仿真和交叉熵优化，联合优化动作与控制器参数，生成可直接执行的、考虑真实动力学的可行计划。**

- **链接: [https://arxiv.org/pdf/2512.11571v1](https://arxiv.org/pdf/2512.11571v1)**

> **作者:** Andreu Matoses Gimenez; Nils Wilde; Chris Pek; Javier Alonso-Mora
>
> **备注:** Preprint
>
> **摘要:** Autonomously performing tasks often requires robots to plan high-level discrete actions and continuous low-level motions to realize them. Previous TAMP algorithms have focused mainly on computational performance, completeness, or optimality by making the problem tractable through simplifications and abstractions. However, this comes at the cost of the resulting plans potentially failing to account for the dynamics or complex contacts necessary to reliably perform the task when object manipulation is required. Additionally, approaches that ignore effects of the low-level controllers may not obtain optimal or feasible plan realizations for the real system. We investigate the use of a GPU-parallelized physics simulator to compute realizations of plans with motion controllers, explicitly accounting for dynamics, and considering contacts with the environment. Using cross-entropy optimization, we sample the parameters of the controllers, or actions, to obtain low-cost solutions. Since our approach uses the same controllers as the real system, the robot can directly execute the computed plans. We demonstrate our approach for a set of tasks where the robot is able to exploit the environment's geometry to move an object. Website and code: https://andreumatoses.github.io/research/parallel-realization
>
---
#### [new 006] WholeBodyVLA: Towards Unified Latent VLA for Whole-Body Loco-Manipulation Control
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文聚焦人形机器人全身运动-操作控制任务，旨在解决现有方法在操作感知行走和大空间运动中的局限。提出统一的隐式学习框架WholeBodyVLA，结合视觉-语言-动作模型与专用强化学习策略，提升运动精度与泛化能力，实现高效大规模数据训练与复杂任务执行。**

- **链接: [https://arxiv.org/pdf/2512.11047v1](https://arxiv.org/pdf/2512.11047v1)**

> **作者:** Haoran Jiang; Jin Chen; Qingwen Bu; Li Chen; Modi Shi; Yanjie Zhang; Delong Li; Chuanzhe Suo; Chuang Wang; Zhihui Peng; Hongyang Li
>
> **摘要:** Humanoid robots require precise locomotion and dexterous manipulation to perform challenging loco-manipulation tasks. Yet existing approaches, modular or end-to-end, are deficient in manipulation-aware locomotion. This confines the robot to a limited workspace, preventing it from performing large-space loco-manipulation. We attribute this to: (1) the challenge of acquiring loco-manipulation knowledge due to the scarcity of humanoid teleoperation data, and (2) the difficulty of faithfully and reliably executing locomotion commands, stemming from the limited precision and stability of existing RL controllers. To acquire richer loco-manipulation knowledge, we propose a unified latent learning framework that enables Vision-Language-Action (VLA) system to learn from low-cost action-free egocentric videos. Moreover, an efficient human data collection pipeline is devised to augment the dataset and scale the benefits. To more precisely execute the desired locomotion commands, we present a loco-manipulation-oriented (LMO) RL policy specifically tailored for accurate and stable core loco-manipulation movements, such as advancing, turning, and squatting. Building on these components, we introduce WholeBodyVLA, a unified framework for humanoid loco-manipulation. To the best of our knowledge, WholeBodyVLA is one of its kind enabling large-space humanoid loco-manipulation. It is verified via comprehensive experiments on the AgiBot X2 humanoid, outperforming prior baseline by 21.3%. It also demonstrates strong generalization and high extensibility across a broad range of tasks.
>
---
#### [new 007] AnchorDream: Repurposing Video Diffusion for Embodiment-Aware Robot Data Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人模仿学习数据生成任务，旨在解决真实数据采集成本高、仿真数据多样性不足的问题。作者提出AnchorDream，利用预训练视频扩散模型，以机器人动作为锚点生成具身一致的多样化演示数据，显著提升策略学习性能。**

- **链接: [https://arxiv.org/pdf/2512.11797v1](https://arxiv.org/pdf/2512.11797v1)**

> **作者:** Junjie Ye; Rong Xue; Basile Van Hoorick; Pavel Tokmakov; Muhammad Zubair Irshad; Yue Wang; Vitor Guizilini
>
> **备注:** Project page: https://jay-ye.github.io/AnchorDream/
>
> **摘要:** The collection of large-scale and diverse robot demonstrations remains a major bottleneck for imitation learning, as real-world data acquisition is costly and simulators offer limited diversity and fidelity with pronounced sim-to-real gaps. While generative models present an attractive solution, existing methods often alter only visual appearances without creating new behaviors, or suffer from embodiment inconsistencies that yield implausible motions. To address these limitations, we introduce AnchorDream, an embodiment-aware world model that repurposes pretrained video diffusion models for robot data synthesis. AnchorDream conditions the diffusion process on robot motion renderings, anchoring the embodiment to prevent hallucination while synthesizing objects and environments consistent with the robot's kinematics. Starting from only a handful of human teleoperation demonstrations, our method scales them into large, diverse, high-quality datasets without requiring explicit environment modeling. Experiments show that the generated data leads to consistent improvements in downstream policy learning, with relative gains of 36.4% in simulator benchmarks and nearly double performance in real-world studies. These results suggest that grounding generative world models in robot motion provides a practical path toward scaling imitation learning.
>
---
#### [new 008] Architecting Large Action Models for Human-in-the-Loop Intelligent Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文致力于构建可验证的神经符号化大型动作模型，解决智能机器人在感知、推理与行动中的可靠性问题。通过组合现成基础模型并引入符号封装与PDDL生成，实现可解释、可验证的人机协同控制。**

- **链接: [https://arxiv.org/pdf/2512.11620v1](https://arxiv.org/pdf/2512.11620v1)**

> **作者:** Kanisorn Sangchai; Methasit Boonpun; Withawin Kraipetchara; Paulo Garcia
>
> **摘要:** The realization of intelligent robots, operating autonomously and interacting with other intelligent agents, human or artificial, requires the integration of environment perception, reasoning, and action. Classic Artificial Intelligence techniques for this purpose, focusing on symbolic approaches, have long-ago hit the scalability wall on compute and memory costs. Advances in Large Language Models in the past decade (neural approaches) have resulted in unprecedented displays of capability, at the cost of control, explainability, and interpretability. Large Action Models aim at extending Large Language Models to encompass the full perception, reasoning, and action cycle; however, they typically require substantially more comprehensive training and suffer from the same deficiencies in reliability. Here, we show it is possible to build competent Large Action Models by composing off-the-shelf foundation models, and that their control, interpretability, and explainability can be effected by incorporating symbolic wrappers and associated verification on their outputs, achieving verifiable neuro-symbolic solutions for intelligent robots. Our experiments on a multi-modal robot demonstrate that Large Action Model intelligence does not require massive end-to-end training, but can be achieved by integrating efficient perception models with a logic-driven core. We find that driving action execution through the generation of Planning Domain Definition Language (PDDL) code enables a human-in-the-loop verification stage that effectively mitigates action hallucinations. These results can support practitioners in the design and development of robotic Large Action Models across novel industries, and shed light on the ongoing challenges that must be addressed to ensure safety in the field.
>
---
#### [new 009] ProbeMDE: Uncertainty-Guided Active Proprioception for Monocular Depth Estimation in Surgical Robotics
- **分类: cs.RO**

- **简介: 该论文研究单目深度估计在手术机器人中的应用，旨在解决纹理缺失、反光等问题导致的深度预测不准确。提出ProbeMDE框架，结合稀疏本体感知测量与RGB图像，通过不确定性引导主动触觉采样，提升深度估计精度，减少测量次数。**

- **链接: [https://arxiv.org/pdf/2512.11773v1](https://arxiv.org/pdf/2512.11773v1)**

> **作者:** Britton Jordan; Jordan Thompson; Jesse F. d'Almeida; Hao Li; Nithesh Kumar; Susheela Sharma Stern; Ipek Oguz; Robert J. Webster; Daniel Brown; Alan Kuntz; James Ferguson
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Monocular depth estimation (MDE) provides a useful tool for robotic perception, but its predictions are often uncertain and inaccurate in challenging environments such as surgical scenes where textureless surfaces, specular reflections, and occlusions are common. To address this, we propose ProbeMDE, a cost-aware active sensing framework that combines RGB images with sparse proprioceptive measurements for MDE. Our approach utilizes an ensemble of MDE models to predict dense depth maps conditioned on both RGB images and on a sparse set of known depth measurements obtained via proprioception, where the robot has touched the environment in a known configuration. We quantify predictive uncertainty via the ensemble's variance and measure the gradient of the uncertainty with respect to candidate measurement locations. To prevent mode collapse while selecting maximally informative locations to propriocept (touch), we leverage Stein Variational Gradient Descent (SVGD) over this gradient map. We validate our method in both simulated and physical experiments on central airway obstruction surgical phantoms. Our results demonstrate that our approach outperforms baseline methods across standard depth estimation metrics, achieving higher accuracy while minimizing the number of required proprioceptive measurements.
>
---
#### [new 010] Optimal Control and Structurally-Informed Gradient Optimization of a Custom 4-DOF Rigid-Body Manipulator
- **分类: cs.RO**

- **简介: 该论文研究四自由度刚体机械臂的最优控制问题。提出结合简化PMP控制器与物理信息梯度优化的方法，生成符合动力学约束的轨迹与时间规划，并通过逆动力学模型输出控制输入，兼顾控制理论结构与计算效率。**

- **链接: [https://arxiv.org/pdf/2512.11250v1](https://arxiv.org/pdf/2512.11250v1)**

> **作者:** Brock Marcinczyk; Logan E. Beaver
>
> **备注:** 6 pages + 18 page appendix
>
> **摘要:** This work develops a control-centric framework for a custom 4-DOF rigid-body manipulator by coupling a reduced-order Pontryagin's Maximum Principle (PMP) controller with a physics-informed Gradient Descent stage. The reduced PMP model provides a closed-form optimal control law for the joint accelerations, while the Gradient Descent module determines the corresponding time horizons by minimizing a cost functional built directly from the full Rigid-Body Dynamics. Structural-mechanics reaction analysis is used only to initialize feasible joint velocities-most critically the azimuthal component-ensuring that the optimizer begins in a physically admissible region. The resulting kinematic trajectories and dynamically consistent time horizons are then supplied to the symbolic Euler-Lagrange model to yield closed-form inverse-dynamics inputs. This pipeline preserves a strict control-theoretic structure while embedding the physical constraints and loading behavior of the manipulator in a computationally efficient way.
>
---
#### [new 011] Towards Logic-Aware Manipulation: A Knowledge Primitive for VLM-Based Assistants in Smart Manufacturing
- **分类: cs.RO**

- **简介: 该论文面向智能制造业中VLM助手机械操作的逻辑感知问题，提出一种八字段元组τ的知识原语，用于传递操作关键参数。工作包括构建τ schema、实例化于3D打印机耗材移除任务，并支持数据增强与检索增强推理，提升规划质量。**

- **链接: [https://arxiv.org/pdf/2512.11275v1](https://arxiv.org/pdf/2512.11275v1)**

> **作者:** Suchang Chen; Daqiang Guo
>
> **备注:** 8 pages, 2 figures, submitted to the 2026 IFAC World Congress
>
> **摘要:** Existing pipelines for vision-language models (VLMs) in robotic manipulation prioritize broad semantic generalization from images and language, but typically omit execution-critical parameters required for contact-rich actions in manufacturing cells. We formalize an object-centric manipulation-logic schema, serialized as an eight-field tuple τ, which exposes object, interface, trajectory, tolerance, and force/impedance information as a first-class knowledge signal between human operators, VLM-based assistants, and robot controllers. We instantiate τ and a small knowledge base (KB) on a 3D-printer spool-removal task in a collaborative cell, and analyze τ-conditioned VLM planning using plan-quality metrics adapted from recent VLM/LLM planning benchmarks, while demonstrating how the same schema supports taxonomy-tagged data augmentation at training time and logic-aware retrieval-augmented prompting at test time as a building block for assistant systems in smart manufacturing enterprises.
>
---
#### [new 012] Learning Category-level Last-meter Navigation from RGB Demonstrations of a Single-instance
- **分类: cs.RO**

- **简介: 该论文研究类别级末端导航任务，解决仅用RGB图像实现移动操作机器人精确定位的问题。提出一种基于单实例示范的模仿学习框架，通过语言驱动分割与空间评分解码，实现跨实例、跨环境的厘米级定位，无需深度、激光或地图先验。**

- **链接: [https://arxiv.org/pdf/2512.11173v1](https://arxiv.org/pdf/2512.11173v1)**

> **作者:** Tzu-Hsien Lee; Fidan Mahmudova; Karthik Desingh
>
> **摘要:** Achieving precise positioning of the mobile manipulator's base is essential for successful manipulation actions that follow. Most of the RGB-based navigation systems only guarantee coarse, meter-level accuracy, making them less suitable for the precise positioning phase of mobile manipulation. This gap prevents manipulation policies from operating within the distribution of their training demonstrations, resulting in frequent execution failures. We address this gap by introducing an object-centric imitation learning framework for last-meter navigation, enabling a quadruped mobile manipulator robot to achieve manipulation-ready positioning using only RGB observations from its onboard cameras. Our method conditions the navigation policy on three inputs: goal images, multi-view RGB observations from the onboard cameras, and a text prompt specifying the target object. A language-driven segmentation module and a spatial score-matrix decoder then supply explicit object grounding and relative pose reasoning. Using real-world data from a single object instance within a category, the system generalizes to unseen object instances across diverse environments with challenging lighting and background conditions. To comprehensively evaluate this, we introduce two metrics: an edge-alignment metric, which uses ground truth orientation, and an object-alignment metric, which evaluates how well the robot visually faces the target. Under these metrics, our policy achieves 73.47% success in edge-alignment and 96.94% success in object-alignment when positioning relative to unseen target objects. These results show that precise last-meter navigation can be achieved at a category-level without depth, LiDAR, or map priors, enabling a scalable pathway toward unified mobile manipulation. Project page: https://rpm-lab-umn.github.io/category-level-last-meter-nav/
>
---
#### [new 013] UniBYD: A Unified Framework for Learning Robotic Manipulation Across Embodiments Beyond Imitation of Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文针对机器人手与人手形态差异导致的模仿学习性能受限问题，提出UniBYD框架，结合统一形态表示与动态强化学习，实现跨形态操作策略学习，并构建新基准UniManip，显著提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2512.11609v1](https://arxiv.org/pdf/2512.11609v1)**

> **作者:** Tingyu Yuan; Biaoliang Guan; Wen Ye; Ziyan Tian; Yi Yang; Weijie Zhou; Yan Huang; Peng Wang; Chaoyang Zhao; Jinqiao Wang
>
> **摘要:** In embodied intelligence, the embodiment gap between robotic and human hands brings significant challenges for learning from human demonstrations. Although some studies have attempted to bridge this gap using reinforcement learning, they remain confined to merely reproducing human manipulation, resulting in limited task performance. In this paper, we propose UniBYD, a unified framework that uses a dynamic reinforcement learning algorithm to discover manipulation policies aligned with the robot's physical characteristics. To enable consistent modeling across diverse robotic hand morphologies, UniBYD incorporates a unified morphological representation (UMR). Building on UMR, we design a dynamic PPO with an annealed reward schedule, enabling reinforcement learning to transition from imitation of human demonstrations to explore policies adapted to diverse robotic morphologies better, thereby going beyond mere imitation of human hands. To address the frequent failures of learning human priors in the early training stage, we design a hybrid Markov-based shadow engine that enables reinforcement learning to imitate human manipulations in a fine-grained manner. To evaluate UniBYD comprehensively, we propose UniManip, the first benchmark encompassing robotic manipulation tasks spanning multiple hand morphologies. Experiments demonstrate a 67.90% improvement in success rate over the current state-of-the-art. Upon acceptance of the paper, we will release our code and benchmark at https://github.com/zhanheng-creator/UniBYD.
>
---
#### [new 014] CarlaNCAP: A Framework for Quantifying the Safety of Vulnerable Road Users in Infrastructure-Assisted Collective Perception Using EuroNCAP Scenarios
- **分类: cs.RO**

- **简介: 该论文提出CarlaNCAP框架，旨在评估基础设施辅助集体感知对弱势道路使用者的安全提升效果。通过构建包含11k帧的EuroNCAP场景数据集，验证了该技术可显著降低事故率，最高实现100%事故避免。**

- **链接: [https://arxiv.org/pdf/2512.11551v1](https://arxiv.org/pdf/2512.11551v1)**

> **作者:** Jörg Gamerdinger; Sven Teufel; Simon Roller; Oliver Bringmann
>
> **摘要:** The growing number of road users has significantly increased the risk of accidents in recent years. Vulnerable Road Users (VRUs) are particularly at risk, especially in urban environments where they are often occluded by parked vehicles or buildings. Autonomous Driving (AD) and Collective Perception (CP) are promising solutions to mitigate these risks. In particular, infrastructure-assisted CP, where sensor units are mounted on infrastructure elements such as traffic lights or lamp posts, can help overcome perceptual limitations by providing enhanced points of view, which significantly reduces occlusions. To encourage decision makers to adopt this technology, comprehensive studies and datasets demonstrating safety improvements for VRUs are essential. In this paper, we propose a framework for evaluating the safety improvement by infrastructure-based CP specifically targeted at VRUs including a dataset with safety-critical EuroNCAP scenarios (CarlaNCAP) with 11k frames. Using this dataset, we conduct an in-depth simulation study and demonstrate that infrastructure-assisted CP can significantly reduce accident rates in safety-critical scenarios, achieving up to 100% accident avoidance compared to a vehicle equipped with sensors with only 33%. Code is available at https://github.com/ekut-es/carla_ncap
>
---
#### [new 015] Taxonomy and Modular Tool System for Versatile and Effective Non-Prehensile Manipulations
- **分类: cs.RO**

- **简介: 该论文针对通用夹爪在非抓握操作（如按压、刮擦）中效果有限的问题，提出一种分类法及基于此的模块化工具系统，使标准二指夹爪能灵活执行多样化的抓握与非抓握任务，提升操作通用性与有效性。**

- **链接: [https://arxiv.org/pdf/2512.11080v1](https://arxiv.org/pdf/2512.11080v1)**

> **作者:** Cedric-Pascal Sommer; Robert J. Wood; Justin Werfel
>
> **备注:** 34 pages, 10 figures, 2 tables, supplementary videos: https://youtu.be/Hcefy53PY0M, https://youtu.be/nFF9k91hsfU, https://youtu.be/EulPLskNIZQ
>
> **摘要:** General-purpose robotic end-effectors of limited complexity, like the parallel-jaw gripper, are appealing for their balance of simplicity and effectiveness in a wide range of manipulation tasks. However, while many such manipulators offer versatility in grasp-like interactions, they are not optimized for non-prehensile actions like pressing, rubbing, or scraping -- manipulations needed for many common tasks. To perform such tasks, humans use a range of different body parts or tools with different rigidity, friction, etc., according to the properties most effective for a given task. Here, we discuss a taxonomy for the key properties of a non-actuated end-effector, laying the groundwork for a systematic understanding of the affordances of non-prehensile manipulators. We then present a modular tool system, based on the taxonomy, that can be used by a standard two-fingered gripper to extend its versatility and effectiveness in performing such actions. We demonstrate the application of the tool system in aerospace and household scenarios that require a range of non-prehensile and prehensile manipulations.
>
---
#### [new 016] Design and Experimental Validation of Closed-Form CBF-Based Safe Control for Stewart Platform Under Multiple Constraints
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究Stewart平台的安全控制任务，解决多约束下实时性与安全性兼顾的问题。提出基于控制屏障函数（CBF）的闭式解析解控制方法，避免在线求解二次规划，显著提升计算效率，并通过仿真与实验验证了安全性和有效性。**

- **链接: [https://arxiv.org/pdf/2512.11125v1](https://arxiv.org/pdf/2512.11125v1)**

> **作者:** Benedictus C. G. Cinun; Tua A. Tamba; Immanuel R. Santjoko; Xiaofeng Wang; Michael A. Gunarso; Bin Hu
>
> **备注:** 9 pages
>
> **摘要:** This letter presents a closed-form solution of Control Barrier Function (CBF) framework for enforcing safety constraints on a Stewart robotic platform. The proposed method simultaneously handles multiple position and velocity constraints through an explicit closed-form control law, eliminating the need to solve a Quadratic Program (QP) at every control step and enabling efficient real-time implementation. This letter derives necessary and sufficient conditions under which the closed-form expression remains non-singular, thereby ensuring well-posedness of the CBF solution to multi-constraint problem. The controller is validated in both simulation and hardware experiments on a custom-built Stewart platform prototype, demonstrating safetyguaranteed performance that is comparable to the QP-based formulation, while reducing computation time by more than an order of magnitude. The results confirm that the proposed approach provides a reliable and computationally lightweight framework for real-time safe control of parallel robotic systems. The experimental videos are available on the project website. (https://nail-uh.github.io/StewartPlatformSafeControl.github.io/)
>
---
#### [new 017] Elevation Aware 2D/3D Co-simulation Framework for Large-scale Traffic Flow and High-fidelity Vehicle Dynamics
- **分类: cs.RO; cs.MA**

- **简介: 该论文针对自动驾驶仿真中缺乏真实地形的问题，提出一种高程感知的2D/3D联合仿真框架，融合OpenStreetMap与USGS数据，实现SUMO与CARLA的协同仿真，构建含复杂地形的大规模、高保真城市交通环境。**

- **链接: [https://arxiv.org/pdf/2512.11249v1](https://arxiv.org/pdf/2512.11249v1)**

> **作者:** Chandra Raskoti; Weizi Li
>
> **摘要:** Reliable testing of autonomous driving systems requires simulation environments that combine large-scale traffic modeling with realistic 3D perception and terrain. Existing tools rarely capture real-world elevation, limiting their usefulness in cities with complex topography. This paper presents an automated, elevation-aware co-simulation framework that integrates SUMO with CARLA using a pipeline that fuses OpenStreetMap road networks and USGS elevation data into physically consistent 3D environments. The system generates smooth elevation profiles, validates geometric accuracy, and enables synchronized 2D-3D simulation across platforms. Demonstrations on multiple regions of San Francisco show the framework's scalability and ability to reproduce steep and irregular terrain. The result is a practical foundation for high-fidelity autonomous vehicle testing in realistic, elevation-rich urban settings.
>
---
#### [new 018] An Anatomy of Vision-Language-Action Models: From Modules to Milestones and Challenges
- **分类: cs.RO**

- **简介: 该论文属于综述任务，旨在梳理视觉-语言-动作（VLA）模型的发展脉络。它系统分析了VLA的模块、里程碑与核心挑战，聚焦表征、执行、泛化、安全及数据评估五大问题，为初学者提供指南，为研究者指明方向。**

- **链接: [https://arxiv.org/pdf/2512.11362v1](https://arxiv.org/pdf/2512.11362v1)**

> **作者:** Chao Xu; Suyu Zhang; Yang Liu; Baigui Sun; Weihong Chen; Bo Xu; Qi Liu; Juncheng Wang; Shujun Wang; Shan Luo; Jan Peters; Athanasios V. Vasilakos; Stefanos Zafeiriou; Jiankang Deng
>
> **摘要:** Vision-Language-Action (VLA) models are driving a revolution in robotics, enabling machines to understand instructions and interact with the physical world. This field is exploding with new models and datasets, making it both exciting and challenging to keep pace with. This survey offers a clear and structured guide to the VLA landscape. We design it to follow the natural learning path of a researcher: we start with the basic Modules of any VLA model, trace the history through key Milestones, and then dive deep into the core Challenges that define recent research frontier. Our main contribution is a detailed breakdown of the five biggest challenges in: (1) Representation, (2) Execution, (3) Generalization, (4) Safety, and (5) Dataset and Evaluation. This structure mirrors the developmental roadmap of a generalist agent: establishing the fundamental perception-action loop, scaling capabilities across diverse embodiments and environments, and finally ensuring trustworthy deployment-all supported by the essential data infrastructure. For each of them, we review existing approaches and highlight future opportunities. We position this paper as both a foundational guide for newcomers and a strategic roadmap for experienced researchers, with the dual aim of accelerating learning and inspiring new ideas in embodied intelligence. A live version of this survey, with continuous updates, is maintained on our \href{https://suyuz1.github.io/Survery/}{project page}.
>
---
#### [new 019] Agile Flight Emerges from Multi-Agent Competitive Racing
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文研究多智能体竞速中的敏捷飞行控制，旨在通过稀疏任务奖励（赢比赛）替代传统密集行为奖励，解决复杂环境中策略与控制协同演化及仿真到现实迁移难题。作者在仿真与真实无人机上验证了方法有效性。**

- **链接: [https://arxiv.org/pdf/2512.11781v1](https://arxiv.org/pdf/2512.11781v1)**

> **作者:** Vineet Pasumarti; Lorenzo Bianchi; Antonio Loquercio
>
> **摘要:** Through multi-agent competition and the sparse high-level objective of winning a race, we find that both agile flight (e.g., high-speed motion pushing the platform to its physical limits) and strategy (e.g., overtaking or blocking) emerge from agents trained with reinforcement learning. We provide evidence in both simulation and the real world that this approach outperforms the common paradigm of training agents in isolation with rewards that prescribe behavior, e.g., progress on the raceline, in particular when the complexity of the environment increases, e.g., in the presence of obstacles. Moreover, we find that multi-agent competition yields policies that transfer more reliably to the real world than policies trained with a single-agent progress-based reward, despite the two methods using the same simulation environment, randomization strategy, and hardware. In addition to improved sim-to-real transfer, the multi-agent policies also exhibit some degree of generalization to opponents unseen at training time. Overall, our work, following in the tradition of multi-agent competitive game-play in digital domains, shows that sparse task-level rewards are sufficient for training agents capable of advanced low-level control in the physical world. Code: https://github.com/Jirl-upenn/AgileFlight_MultiAgent
>
---
#### [new 020] The Influence of Human-like Appearance on Expected Robot Explanations
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究机器人拟人化外观对用户预期解释的影响，探讨外观如何引发人类心理投射，并影响人们对机器人行为解释的期望。通过三类不同外观机器人的对比实验，发现外观越像人，用户越倾向于给出拟人化解释。**

- **链接: [https://arxiv.org/pdf/2512.11746v1](https://arxiv.org/pdf/2512.11746v1)**

> **作者:** Hana Kopecka; Jose Such
>
> **摘要:** A robot's appearance is a known factor influencing user's mental model and human-robot interaction, that has not been studied in the context of its influence in expected robot explanations. In this study, we investigate whether and to what extent the human-like appearance of robots elicits anthropomorphism, which is conceptualised as an attribution of mental capacities, and how the level of anthropomorphism is revealed in explanations that people expect to receive. We designed a between-subject study comprising conditions with visual stimuli of three domestic service robots with varying human-like appearance, and we prompted respondents to provide explanations they would expect to receive from the robot for the same robot actions. We found that most explanations were anthropomorphic across all conditions. However, there is a positive correlation between the anthropomorphic explanations and human-like appearance. We also report on more nuanced trends observed in non-anthropomorphic explanations and trends in robot descriptions.
>
---
#### [new 021] Vision-Language Models for Infrared Industrial Sensing in Additive Manufacturing Scene Description
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究零样本工业红外感知任务，解决现有视觉语言模型无法理解红外图像的问题。提出VLM-IRIS框架，将红外图像转为RGB兼容的伪彩色图像，结合CLIP模型实现无需训练的工件存在检测，适用于增材制造中的无标签热监控。**

- **链接: [https://arxiv.org/pdf/2512.11098v1](https://arxiv.org/pdf/2512.11098v1)**

> **作者:** Nazanin Mahjourian; Vinh Nguyen
>
> **摘要:** Many manufacturing environments operate in low-light conditions or within enclosed machines where conventional vision systems struggle. Infrared cameras provide complementary advantages in such environments. Simultaneously, supervised AI systems require large labeled datasets, which makes zero-shot learning frameworks more practical for applications including infrared cameras. Recent advances in vision-language foundation models (VLMs) offer a new path in zero-shot predictions from paired image-text representations. However, current VLMs cannot understand infrared camera data since they are trained on RGB data. This work introduces VLM-IRIS (Vision-Language Models for InfraRed Industrial Sensing), a zero-shot framework that adapts VLMs to infrared data by preprocessing infrared images captured by a FLIR Boson sensor into RGB-compatible inputs suitable for CLIP-based encoders. We demonstrate zero-shot workpiece presence detection on a 3D printer bed where temperature differences between the build plate and workpieces make the task well-suited for thermal imaging. VLM-IRIS converts the infrared images to magma representation and applies centroid prompt ensembling with a CLIP ViT-B/32 encoder to achieve high accuracy on infrared images without any model retraining. These findings demonstrate that the proposed improvements to VLMs can be effectively extended to thermal applications for label-free monitoring.
>
---
#### [new 022] Two-dimensional Decompositions of High-dimensional Configurations for Efficient Multi-vehicle Coordination at Intelligent Intersections
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究智能交叉口多车协同轨迹规划，旨在解决高维构型空间下避障与效率难题。提出二维分解算法，将高维问题转为序列2D图搜索，降低计算复杂度，并结合NMPC实现安全平滑控制，较传统MILP方法更高效。**

- **链接: [https://arxiv.org/pdf/2512.11713v1](https://arxiv.org/pdf/2512.11713v1)**

> **作者:** Amirreza Akbari; Johan Thunberg
>
> **摘要:** For multi-vehicle complex traffic scenarios in shared spaces such as intelligent intersections, safe coordination and trajectory planning is challenging due to computational complexity. To meet this challenge, we introduce a computationally efficient method for generating collision-free trajectories along predefined vehicle paths. We reformulate a constrained minimum-time trajectory planning problem as a problem in a high-dimensional configuration space, where conflict zones are modeled by high-dimensional polyhedra constructed from two-dimensional rectangles. Still, in such a formulation, as the number of vehicles involved increases, the computational complexity increases significantly. To address this, we propose two algorithms for near-optimal local optimization that significantly reduce the computational complexity by decomposing the high-dimensional problem into a sequence of 2D graph search problems. The resulting trajectories are then incorporated into a Nonlinear Model Predictive Control (NMPC) framework to ensure safe and smooth vehicle motion. We furthermore show in numerical evaluation that this approach significantly outperforms existing MILP-based time-scheduling; both in terms of objective-value and computational time.
>
---
#### [new 023] Symmetry-Aware Steering of Equivariant Diffusion Policies: Benefits and Limits
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究对称性感知的扩散策略引导，解决标准强化学习忽略几何对称性导致的低效与不稳定问题。提出对称性保持的引导框架，理论证明扩散过程的等变性，并验证其在样本效率和策略提升上的优势。**

- **链接: [https://arxiv.org/pdf/2512.11345v1](https://arxiv.org/pdf/2512.11345v1)**

> **作者:** Minwoo Park; Junwoo Chang; Jongeun Choi; Roberto Horowitz
>
> **摘要:** Equivariant diffusion policies (EDPs) combine the generative expressivity of diffusion models with the strong generalization and sample efficiency afforded by geometric symmetries. While steering these policies with reinforcement learning (RL) offers a promising mechanism for fine-tuning beyond demonstration data, directly applying standard (non-equivariant) RL can be sample-inefficient and unstable, as it ignores the symmetries that EDPs are designed to exploit. In this paper, we theoretically establish that the diffusion process of an EDP is equivariant, which in turn induces a group-invariant latent-noise MDP that is well-suited for equivariant diffusion steering. Building on this theory, we introduce a principled symmetry-aware steering framework and compare standard, equivariant, and approximately equivariant RL strategies through comprehensive experiments across tasks with varying degrees of symmetry. While we identify the practical boundaries of strict equivariance under symmetry breaking, we show that exploiting symmetry during the steering process yields substantial benefits-enhancing sample efficiency, preventing value divergence, and achieving strong policy improvements even when EDPs are trained from extremely limited demonstrations.
>
---
#### [new 024] Atomic Action Slicing: Planner-Aligned Options for Generalist VLA Agents
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型泛化能力差的问题，提出原子动作切片（AAS）方法，将长程任务分解为 planner 对齐的短原子动作。基于此构建 GATE-VLAP 数据集，并验证其提升 CLIP-RT+ 模型在 LIBERO 任务上的表现。**

- **链接: [https://arxiv.org/pdf/2512.11584v1](https://arxiv.org/pdf/2512.11584v1)**

> **作者:** Stefan Tabakov; Asen Popov; Dimitar Dimitrov; S. Ensiye Kiyamousavi; Vladimir Hristov; Boris Kraychev
>
> **备注:** The 41st ACM/SIGAPP Symposium On Applied Computing
>
> **摘要:** Current vision-language-action (VLA) models generalize poorly, particularly when tasks require new compositions of skills or objects. We introduce Atomic Action Slicing (AAS), a planner-aligned approach that decomposes long-horizon demonstrations into short, typed atomic actions that are easier for planners to use and policies to learn. Using LIBERO demonstrations, AAS produces a validated dataset of 2,124 atomic segments labeled with action type, temporal span, and confidence. A stronger segmenter (Gemini 2.5 Pro) closely matches planner-defined plans and remains robust under keyframe jitter, while smaller models perform worse on multi-object tasks. Fine-tuning CLIP-RT+ on our atomic dataset improves task success from 94.2% to 95.3% on LIBERO-Goal and 83.8% to 88.8% on LIBERO-Long. We publicly release the GATE-VLAP dataset on HuggingFace(https://huggingface.co/datasets/gate-institute/GATE-VLAP-datasets)
>
---
#### [new 025] Mirror Skin: In Situ Visualization of Robot Touch Intent on Robotic Skin
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出Mirror Skin，旨在解决人机物理交互中触碰意图传达不明确的问题。通过机器人皮肤上的镜像视觉反馈，实时映射人体接触部位，实现对触碰对象、位置和时机的直观示意，并经专家设计探索与用户实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.11472v1](https://arxiv.org/pdf/2512.11472v1)**

> **作者:** David Wagmann; Matti Krüger; Chao Wang; Jürgen Steimle
>
> **摘要:** Effective communication of robotic touch intent is a key factor in promoting safe and predictable physical human-robot interaction (pHRI). While intent communication has been widely studied, existing approaches lack the spatial specificity and semantic depth necessary to convey robot touch actions. We present Mirror Skin, a cephalopod-inspired concept that utilizes high-resolution, mirror-like visual feedback on robotic skin. By mapping in-situ visual representations of a human's body parts onto the corresponding robot's touch region, Mirror Skin communicates who shall initiate touch, where it will occur, and when it is imminent. To inform the design of Mirror Skin, we conducted a structured design exploration with experts in virtual reality (VR), iteratively refining six key dimensions. A subsequent controlled user study demonstrated that Mirror Skin significantly enhances accuracy and reduces response times for interpreting touch intent. These findings highlight the potential of visual feedback on robotic skin to communicate human-robot touch interactions.
>
---
#### [new 026] Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于立体匹配任务，旨在解决现有模型无法兼顾实时性与零样本泛化能力的问题。作者提出Fast-FoundationStereo，通过知识蒸馏、神经架构搜索和结构化剪枝实现高效加速，并构建真实场景数据集提升性能，首次实现实时零样本高精度立体匹配。**

- **链接: [https://arxiv.org/pdf/2512.11130v1](https://arxiv.org/pdf/2512.11130v1)**

> **作者:** Bowen Wen; Shaurya Dewan; Stan Birchfield
>
> **摘要:** Stereo foundation models achieve strong zero-shot generalization but remain computationally prohibitive for real-time applications. Efficient stereo architectures, on the other hand, sacrifice robustness for speed and require costly per-domain fine-tuning. To bridge this gap, we present Fast-FoundationStereo, a family of architectures that achieve, for the first time, strong zero-shot generalization at real-time frame rate. We employ a divide-and-conquer acceleration strategy with three components: (1) knowledge distillation to compress the hybrid backbone into a single efficient student; (2) blockwise neural architecture search for automatically discovering optimal cost filtering designs under latency budgets, reducing search complexity exponentially; and (3) structured pruning for eliminating redundancy in the iterative refinement module. Furthermore, we introduce an automatic pseudo-labeling pipeline used to curate 1.4M in-the-wild stereo pairs to supplement synthetic training data and facilitate knowledge distillation. The resulting model can run over 10x faster than FoundationStereo while closely matching its zero-shot accuracy, thus establishing a new state-of-the-art among real-time methods. Project page: https://nvlabs.github.io/Fast-FoundationStereo/
>
---
#### [new 027] Toward a Decision Support System for Energy-Efficient Ferry Operation on Lake Constance based on Optimal Control
- **分类: eess.SY; cs.HC; cs.RO**

- **简介: 该论文设计基于最优控制的决策支持系统，解决博登湖渡轮节能航行问题。通过建立渡轮动力学模型并考虑水流、风等环境干扰，采用滚动时域优化框架，为船员提供实时操作指引，在保证航行时间的同时提升能效。**

- **链接: [https://arxiv.org/pdf/2512.11786v1](https://arxiv.org/pdf/2512.11786v1)**

> **作者:** Hannes Homburger; Bastian Jäckl; Stefan Wirtensohn; Christian Stopp; Maximilian T. Fischer; Moritz Diehl; Daniel A. Keim; Johannes Reuter
>
> **备注:** 6 pages, 8 figures
>
> **摘要:** The maritime sector is undergoing a disruptive technological change driven by three main factors: autonomy, decarbonization, and digital transformation. Addressing these factors necessitates a reassessment of inland vessel operations. This paper presents the design and development of a decision support system for ferry operations based on a shrinking-horizon optimal control framework. The problem formulation incorporates a mathematical model of the ferry's dynamics and environmental disturbances, specifically water currents and wind, which can significantly influence the dynamics. Real-world data and illustrative scenarios demonstrate the potential of the proposed system to effectively support ferry crews by providing real-time guidance. This enables enhanced operational efficiency while maintaining predefined maneuver durations. The findings suggest that optimal control applications hold substantial promise for advancing future ferry operations on inland waters. A video of the real-world ferry MS Insel Mainau operating on Lake Constance is available at: https://youtu.be/i1MjCdbEQyE
>
---
## 更新

#### [replaced 001] See What I Mean? Expressiveness and Clarity in Robot Display Design
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究人机协作中非语言视觉符号的设计对沟通效果的影响。通过在公开场所开展37人参与的导航实验，比较动态与静态图标及拟人眼的效果，发现静态图标更易理解，静态眼提高任务成功率，而动画增强信任与满意度，提出结合静态符号优化沟通的设计建议。**

- **链接: [https://arxiv.org/pdf/2506.16643v2](https://arxiv.org/pdf/2506.16643v2)**

> **作者:** Matthew Ebisu; Hang Yu; Reuben Aronson; Elaine Short
>
> **摘要:** Nonverbal visual symbols and displays play an important role in communication when humans and robots work collaboratively. However, few studies have investigated how different types of non-verbal cues affect objective task performance, especially in a dynamic environment that requires real time decision-making. In this work, we designed a collaborative navigation task where the user and the robot only had partial information about the map on each end and thus the users were forced to communicate with a robot to complete the task. We conducted our study in a public space and recruited 37 participants who randomly passed by our setup. Each participant collaborated with a robot utilizing either animated anthropomorphic eyes and animated icons, or static anthropomorphic eyes and static icons. We found that participants that interacted with a robot with animated displays reported the greatest level of trust and satisfaction; that participants interpreted static icons the best; and that participants with a robot with static eyes had the highest completion success. These results suggest that while animation can foster trust with robots, human-robot communication can be optimized by the addition of familiar static icons that may be easier for users to interpret. We published our code, designed symbols, and collected results online at: https://github.com/mattufts/huamn_Cozmo_interaction.
>
---
#### [replaced 002] In-situ Value-aligned Human-Robot Interactions with Physical Constraints
- **分类: cs.RO**

- **简介: 该论文研究人机交互中任务执行与人类偏好对齐的问题，提出ICLHF框架，结合人类反馈与物理约束，使机器人在完成任务时兼顾用户偏好。构建了家庭活动基准并验证了方法有效性。**

- **链接: [https://arxiv.org/pdf/2508.07606v2](https://arxiv.org/pdf/2508.07606v2)**

> **作者:** Hongtao Li; Ziyuan Jiao; Xiaofeng Liu; Hangxin Liu; Zilong Zheng
>
> **备注:** 8 pages, 7 figures. Accepted by IROS 2025
>
> **摘要:** Equipped with Large Language Models (LLMs), human-centered robots are now capable of performing a wide range of tasks that were previously deemed challenging or unattainable. However, merely completing tasks is insufficient for cognitive robots, who should learn and apply human preferences to future scenarios. In this work, we propose a framework that combines human preferences with physical constraints, requiring robots to complete tasks while considering both. Firstly, we developed a benchmark of everyday household activities, which are often evaluated based on specific preferences. We then introduced In-Context Learning from Human Feedback (ICLHF), where human feedback comes from direct instructions and adjustments made intentionally or unintentionally in daily life. Extensive sets of experiments, testing the ICLHF to generate task plans and balance physical constraints with preferences, have demonstrated the efficiency of our approach. Project page: https://iclhf.github.io .
>
---
#### [replaced 003] Social Mediation through Robots -- A Scoping Review on Improving Group Interactions through Directed Robot Action using an Extended Group Process Model
- **分类: cs.RO; cs.HC**

- **简介: 该论文属综述任务，旨在解决社交机器人如何通过干预改善群体互动的问题。作者提出“中介I-P-O模型”，分析1633篇文献，提炼89个概念与11种机器人干预策略，构建理论框架以推动领域标准化研究。**

- **链接: [https://arxiv.org/pdf/2409.06557v2](https://arxiv.org/pdf/2409.06557v2)**

> **作者:** Thomas H. Weisswange; Hifza Javed; Manuel Dietrich; Malte F. Jung; Nawid Jamali
>
> **备注:** Early version of the published journal paper: Weisswange, T. H., Javed, H., Dietrich, M., Jung, M. F., & Jamali, N. (2025). Design Implications for Robots that Facilitate Groups-A Scoping Review on Improving Group Interactions through Directed Robot Action. ACM Transactions on Human-Robot Interaction
>
> **摘要:** Group processes refer to the dynamics that occur within a group and are critical for understanding how groups function. With robots being increasingly placed within small groups, improving these processes has emerged as an important application of social robotics. Social Mediation Robots elicit behavioral change within groups by deliberately influencing the processes of groups. While research in this field has demonstrated that robots can effectively affect interpersonal dynamics, there is a notable gap in integrating these insights to develop coherent understanding and theory. We present a scoping review of literature targeting changes in social interactions between multiple humans through intentional action from robotic agents. To guide our review, we adapt the classical Input-Process-Output (I-P-O) models that we call "Mediation I-P-O model". We evaluated 1633 publications, which yielded 89 distinct social mediation concepts. We construct 11 mediation approaches robots can use to shape processes in small groups and teams. This work strives to produce generalizable insights and evaluate the extent to which the potential of social mediation through robots has been realized thus far. We hope that the proposed framework encourages a holistic approach to the study of social mediation and provides a foundation to standardize future reporting in the domain.
>
---
#### [replaced 004] 3DSGrasp: 3D Shape-Completion for Robotic Grasp
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人抓取任务，旨在解决因点云数据不完整导致抓取姿态错误的问题。提出3DSGrasp方法，通过基于Transformer的形状补全网络恢复缺失几何，提升抓取成功率。**

- **链接: [https://arxiv.org/pdf/2301.00866v2](https://arxiv.org/pdf/2301.00866v2)**

> **作者:** Seyed S. Mohammadi; Nuno F. Duarte; Dimitris Dimou; Yiming Wang; Matteo Taiana; Pietro Morerio; Atabak Dehban; Plinio Moreno; Alexandre Bernardino; Alessio Del Bue; Jose Santos-Victor
>
> **摘要:** Real-world robotic grasping can be done robustly if a complete 3D Point Cloud Data (PCD) of an object is available. However, in practice, PCDs are often incomplete when objects are viewed from few and sparse viewpoints before the grasping action, leading to the generation of wrong or inaccurate grasp poses. We propose a novel grasping strategy, named 3DSGrasp, that predicts the missing geometry from the partial PCD to produce reliable grasp poses. Our proposed PCD completion network is a Transformer-based encoder-decoder network with an Offset-Attention layer. Our network is inherently invariant to the object pose and point's permutation, which generates PCDs that are geometrically consistent and completed properly. Experiments on a wide range of partial PCD show that 3DSGrasp outperforms the best state-of-the-art method on PCD completion tasks and largely improves the grasping success rate in real-world scenarios. The code and dataset will be made available upon acceptance.
>
---
#### [replaced 005] MOFU: Development of a MOrphing Fluffy Unit with Expansion and Contraction Capabilities and Evaluation of the Animacy of Its Movements
- **分类: cs.RO**

- **简介: 该论文旨在提升社交机器人的生命感。针对体积变化运动对拟人性影响的研究不足，设计了可整体膨胀收缩的绒毛机器人MOFU，并通过实验验证其动作能显著增强人类感知的生动性，表明体积变化是社交机器人设计的重要因素。**

- **链接: [https://arxiv.org/pdf/2509.09613v2](https://arxiv.org/pdf/2509.09613v2)**

> **作者:** Taisei Mogi; Mari Saito; Yoshihiro Nakata
>
> **摘要:** Robots designed for therapy and social interaction aim to evoke a sense of animacy in humans. While many studies have focused on life like appearance or joint based movements, the effect of whole body volume changing movements commonly observed in living organisms has received little attention. In this study, we developed MOFU MOrphing Fluffy Unit, a mobile robot capable of whole body expansion and contraction using a single motor enclosed in a fluffy exterior. MOFU employs a Jitterbug geometric transformation mechanism that enables smooth diameter changes from approximately 210 mm to 280 mm with a single actuator, and is equipped with a differential two wheel drive mechanism for locomotion. We conducted an online survey using videos of MOFU behaviors and evaluated perceived animacy using the Godspeed Questionnaire Series. First, we compared stationary conditions with and without expansion contraction and with and without rotational motion. Both expansion contraction and rotation independently increased perceived animacy. Second, we examined whether presenting two MOFUs simultaneously would further enhance animacy perception, but no significant difference was observed. Exploratory analyses were also conducted across four dual robot motion conditions. Third, when expansion contraction was combined with locomotion, animacy ratings were higher than for locomotion alone. These results suggest that whole body volume changing movements enhance perceived animacy in robots, indicating that physical volume change is an important design element for future social and therapeutic robots.
>
---
#### [replaced 006] Fast Multi-Party Open-Ended Conversation with a Social Robot
- **分类: cs.HC; cs.RO**

- **简介: 该论文研究多用户开放对话任务，旨在解决机器人在多人交互中识别说话人、分配话语权和生成连贯回应的问题。作者结合多模态感知与大语言模型，在Furhat机器人上实现系统，并通过两种场景实验验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2503.15496v2](https://arxiv.org/pdf/2503.15496v2)**

> **作者:** Giulio Antonio Abbo; Maria Jose Pinto-Bernal; Martijn Catrycke; Tony Belpaeme
>
> **备注:** 15 pages, 5 figures, 4 tables; 2 appendices
>
> **摘要:** Multi-party open-ended conversation remains a major challenge in human-robot interaction, particularly when robots must recognise speakers, allocate turns, and respond coherently under overlapping or rapidly shifting dialogue. This paper presents a multi-party conversational system that combines multimodal perception (voice direction of arrival, speaker diarisation, face recognition) with a large language model for response generation. Implemented on the Furhat robot, the system was evaluated with 30 participants across two scenarios: (i) parallel, separate conversations and (ii) shared group discussion. Results show that the system maintains coherent and engaging conversations, achieving high addressee accuracy in parallel settings (92.6%) and strong face recognition reliability (80-94%). Participants reported clear social presence and positive engagement, although technical barriers such as audio-based speaker recognition errors and response latency affected the fluidity of group interactions. The results highlight both the promise and limitations of LLM-based multi-party interaction and outline concrete directions for improving multimodal cue integration and responsiveness in future social robots.
>
---
#### [replaced 007] Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge
- **分类: cs.RO**

- **简介: 该论文针对2025 BEHAVIOR挑战赛，解决机器人在真实家庭环境中执行长周期操作任务的问题。基于π₀.₅模型，通过系统优化训练方法与数据，提升性能，在比赛中获第二名，为具身智能提供实用设计建议。**

- **链接: [https://arxiv.org/pdf/2512.10071v2](https://arxiv.org/pdf/2512.10071v2)**

> **作者:** Junjie Bai; Yu-Wei Chao; Qizhi Chen; Jinwei Gu; Moo Jin Kim; Zhaoshuo Li; Xuan Li; Tsung-Yi Lin; Ming-Yu Liu; Nic Ma; Kaichun Mo; Delin Qu; Shangkun Sun; Hongchi Xia; Fangyin Wei; Xiaohui Zeng
>
> **备注:** preprint
>
> **摘要:** The 2025 BEHAVIOR Challenge is designed to rigorously track progress toward solving long-horizon tasks by physical agents in simulated environments. BEHAVIOR-1K focuses on everyday household tasks that people most want robots to assist with and these tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. This report presents our solution to the 2025 BEHAVIOR Challenge in a very close 2nd place and substantially outperforms the rest of the submissions. Building on $π_{0.5}$, we focus on systematically building our solution by studying the effects of training techniques and data. Through careful ablations, we show the scaling power in pre-training and post-training phases for competitive performance. We summarize our practical lessons and design recommendations that we hope will provide actionable insights for the broader embodied AI community when adapting powerful foundation models to complex embodied scenarios.
>
---
#### [replaced 008] Adaptive Compressive Tactile Subsampling: Enabling High Spatiotemporal Resolution in Scalable Robotic Skin
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出自适应压缩触觉欠采样（ACTS），解决高分辨率触觉传感因数据量大导致帧率低的问题。通过自适应采样与稀疏重建，实现千赫兹级高速触觉感知，支持实时闭环控制与动态交互应用。**

- **链接: [https://arxiv.org/pdf/2410.13847v3](https://arxiv.org/pdf/2410.13847v3)**

> **作者:** Ariel Slepyan; Dian Li; Hongjun Cai; Ryan McGovern; Aidan Aug; Sriramana Sankar; Trac Tran; Nitish Thakor
>
> **备注:** 51 pages, 11 main figures, 16 supplemental figures, Videos can be accessed at https://tinyurl.com/ACTS-videos
>
> **摘要:** Robots require full-body, high-resolution tactile sensing to operate safely in unstructured environments, enabling reflexive responses and closed-loop control. However, the pixel counts needed for dense, large-area coverage limit readout rates of most tactile arrays to <100 Hz, hindering their use in high-speed tasks. We present Adaptive Compressive Tactile Subsampling (ACTS), a scalable and data-driven method that greatly enhances traditional tactile matrices by leveraging adaptive sensor sampling and sparse recovery. By adaptively allocating measurements to informative regions, ACTS is especially effective for spatially sparse signals common in real-world interactions. Tested on a 1024-pixel tactile sensor array (32x32), ACTS achieved frame rates up to 1,000 Hz, an 18X improvement over conventional raster scanning, with minimal reconstruction error. For the first time, ACTS enables wearable, large-area, high-density tactile sensing systems that can deliver high-speed results. We demonstrate rapid object classification within 20 ms of contact, high-speed projectile detection, ricochet angle estimation, and soft deformation tracking, in tactile and robotics applications, all using flexible, high-density tactile arrays. These include high-resolution tactile gloves, pressure insoles, and full-body configurations covering robotic arms and human-sized mannequins. We further showcase tactile-based closed-loop control by guiding a metallic ball to trace letters using tactile feedback and by executing tactile-only whole-hand reflexes on a fully sensorized LEAP hand to stabilize grasps, prevent slip, and avoid sharp objects, validating ACTS for real-time interaction and motion control. ACTS transforms standard, low-cost, and robust tactile sensors into high-speed systems enabling scalable, responsive, and adaptive tactile perception for robots and wearables operating in dynamic environments.
>
---
#### [replaced 009] An effective control of large systems of active particles: An application to evacuation problem
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究大规模活性粒子系统的控制，旨在解决疏散难题。通过结合强化学习与人工力场，提出基于领导者引导的高效控制策略，并应用于机器人救援员引导人群疏散，显著提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.19972v3](https://arxiv.org/pdf/2509.19972v3)**

> **作者:** Albina Klepach; Egor E. Nuzhin; Alexey A. Tsukanov; Nikolay V. Brilliantov
>
> **摘要:** Manipulation of large systems of active particles is a serious challenge across diverse domains, including crowd management, control of robotic swarms, and coordinated material transport. The development of advanced control strategies for complex scenarios is hindered, however, by the lack of scalability and robustness of the existing methods, in particular, due to the need of an individual control for each agent. One possible solution involves controlling a system through a leader or a group of leaders, which other agents tend to follow. Using such an approach we develop an effective control strategy for a leader, combining reinforcement learning (RL) with artificial forces acting on the system. To describe the guidance of active particles by a leader we introduce the generalized Vicsek model. This novel method is then applied to the problem of the effective evacuation by a robot-rescuer (leader) of large groups of people from hazardous places. We demonstrate, that while a straightforward application of RL yields suboptimal results, even for advanced architectures, our approach provides a robust and efficient evacuation strategy. The source code supporting this study is publicly available at: https://github.com/cinemere/evacuation.
>
---
#### [replaced 010] Model-Based Lookahead Reinforcement Learning for in-hand manipulation
- **分类: cs.RO**

- **简介: 该论文研究灵巧手操作中的强化学习方法，旨在提升任务性能。提出一种结合模型基与无模型的混合框架，利用动态模型和值函数指导策略，通过轨迹优化提高控制精度，并验证其在不同物体属性下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.08884v2](https://arxiv.org/pdf/2510.08884v2)**

> **作者:** Alexandre Lopes; Catarina Barata; Plinio Moreno
>
> **摘要:** In-Hand Manipulation, as many other dexterous tasks, remains a difficult challenge in robotics by combining complex dynamic systems with the capability to control and manoeuvre various objects using its actuators. This work presents the application of a previously developed hybrid Reinforcement Learning (RL) Framework to In-Hand Manipulation task, verifying that it is capable of improving the performance of the task. The model combines concepts of both Model-Free and Model-Based Reinforcement Learning, by guiding a trained policy with the help of a dynamic model and value-function through trajectory evaluation, as done in Model Predictive Control. This work evaluates the performance of the model by comparing it with the policy that will be guided. To fully explore this, various tests are performed using both fully-actuated and under-actuated simulated robotic hands to manipulate different objects for a given task. The performance of the model will also be tested for generalization tests, by changing the properties of the objects in which both the policy and dynamic model were trained, such as density and size, and additionally by guiding a trained policy in a certain object to perform the same task in a different one. The results of this work show that, given a policy with high average reward and an accurate dynamic model, the hybrid framework improves the performance of in-hand manipulation tasks for most test cases, even when the object properties are changed. However, this improvement comes at the expense of increasing the computational cost, due to the complexity of trajectory evaluation.
>
---
#### [replaced 011] WARPD: World model Assisted Reactive Policy Diffusion
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决扩散策略在高频率控制中推理慢、轨迹误差累积的问题。作者提出WARPD方法，通过世界模型辅助，在参数空间学习闭环策略，实现长视野、抗干扰且低计算成本的控制。**

- **链接: [https://arxiv.org/pdf/2410.14040v4](https://arxiv.org/pdf/2410.14040v4)**

> **作者:** Shashank Hegde; Satyajeet Das; Gautam Salhotra; Gaurav S. Sukhatme
>
> **备注:** Outstanding Paper Award at the Embodied World Models for Decision Making Workshop at NeurIPS 2025
>
> **摘要:** With the increasing availability of open-source robotic data, imitation learning has become a promising approach for both manipulation and locomotion. Diffusion models are now widely used to train large, generalized policies that predict controls or trajectories, leveraging their ability to model multimodal action distributions. However, this generality comes at the cost of larger model sizes and slower inference, an acute limitation for robotic tasks requiring high control frequencies. Moreover, Diffusion Policy (DP), a popular trajectory-generation approach, suffers from a trade-off between performance and action horizon: fewer diffusion queries lead to larger trajectory chunks, which in turn accumulate tracking errors. To overcome these challenges, we introduce WARPD (World model Assisted Reactive Policy Diffusion), a method that generates closed-loop policies (weights for neural policies) directly, instead of open-loop trajectories. By learning behavioral distributions in parameter space rather than trajectory space, WARPD offers two major advantages: (1) extended action horizons with robustness to perturbations, while maintaining high task performance, and (2) significantly reduced inference costs. Empirically, WARPD outperforms DP in long-horizon and perturbed environments, and achieves multitask performance on par with DP while requiring only ~ 1/45th of the inference-time FLOPs per step.
>
---
#### [replaced 012] UMArm: Untethered, Modular, Portable, Soft Pneumatic Arm
- **分类: cs.RO**

- **简介: 该论文提出一种新型气动刚柔混合机械臂UMArm，旨在解决现有软体机械臂自由度低、依赖外部调压装置等问题。通过集成自调节McKibben驱动器，实现高负载、高精度、无束缚、可重构的便携式系统，提升其在非结构化环境中的适应性与实用性。**

- **链接: [https://arxiv.org/pdf/2505.11476v2](https://arxiv.org/pdf/2505.11476v2)**

> **作者:** Runze Zuo; Dong Heon Han; Richard Li; Saima Jamal; Daniel Bruder
>
> **摘要:** Robotic arms are essential to modern industries, however, their adaptability to unstructured environments remains limited. Soft robotic arms, particularly those actuated pneumatically, offer greater adaptability in unstructured environments and enhanced safety for human-robot interaction. However, current pneumatic soft arms are constrained by limited degrees of freedom, precision, payload capacity, and reliance on bulky external pressure regulators. In this work, a novel pneumatically driven rigid-soft hybrid arm, ``UMArm'', is presented. The shortcomings of pneumatically actuated soft arms are addressed by densely integrating high-force-to-weight-ratio, self-regulated McKibben actuators onto a lightweight rigid spine structure. The modified McKibben actuators incorporate valves and controllers directly inside, eliminating the need for individual pressure lines and external regulators, significantly reducing system weight and complexity. Full untethered operation, high payload capacity, precision, and directionally tunable compliance are achieved by the UMArm. Portability is demonstrated through a wearable assistive arm experiment, and versatility is showcased by reconfiguring the system into an inchworm robot. The results of this work show that the high-degree-of-freedom, external-regulator-free pneumatically driven arm systems like the UMArm possess great potential for real-world unstructured environments.
>
---
#### [replaced 013] Communication-Efficient Module-Wise Federated Learning for Grasp Pose Detection in Cluttered Environments
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究抓取姿态检测（GPD）任务，旨在解决联邦学习中通信开销大的问题。提出模块级联邦学习框架，通过分析模型组件学习动态，分阶段训练慢收敛模块，减少通信量。实验表明其在有限通信下提升准确率与抓取成功率。**

- **链接: [https://arxiv.org/pdf/2507.05861v2](https://arxiv.org/pdf/2507.05861v2)**

> **作者:** Woonsang Kang; Joohyung Lee; Seungjun Kim; Jungchan Cho; Yoonseon Oh
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L). 8 pages, 5 figures
>
> **摘要:** Grasp pose detection (GPD) is a fundamental capability for robotic autonomy, but its reliance on large, diverse datasets creates significant data privacy and centralization challenges. Federated Learning (FL) offers a privacy-preserving solution, but its application to GPD is hindered by the substantial communication overhead of large models, a key issue for resource-constrained robots. To address this, we propose a novel module-wise FL framework that begins by analyzing the learning dynamics of the GPD model's functional components. This analysis identifies slower-converging modules, to which our framework then allocates additional communication effort. This is realized through a two-phase process: a standard full-model training phase is followed by a communication-efficient phase where only the identified subset of slower-converging modules is trained and their partial updates are aggregated. Extensive experiments on the GraspNet-1B dataset demonstrate that our method outperforms standard FedAvg and other baselines, achieving higher accuracy for a given communication budget. Furthermore, real-world experiments on a physical robot validate our approach, showing a superior grasp success rate compared to baseline methods in cluttered scenes. Our work presents a communication-efficient framework for training robust, generalized GPD models in a decentralized manner, effectively improving the trade-off between communication cost and model performance.
>
---
#### [replaced 014] Bio-inspired reconfigurable stereo vision for robotics using omnidirectional cameras
- **分类: cs.RO**

- **简介: 该论文提出一种受生物启发的可重构立体视觉系统，用于机器人。针对传统立体视觉视野窄、相机布局固定的问题，设计了动态可调的双目配置，并结合深度神经网络与几何方法实现精准测距。系统在变形机器人上验证，支持多种视觉模式切换，适应不同任务需求。**

- **链接: [https://arxiv.org/pdf/2410.08691v2](https://arxiv.org/pdf/2410.08691v2)**

> **作者:** Suchang Chen; Dongliang Fan; Huijuan Feng; Jian S Dai
>
> **备注:** 7 pages, 8 figures, submitted to IEEE ICRA 2025
>
> **摘要:** This work introduces a novel bio-inspired reconfigurable stereo vision system for robotics, leveraging omnidirectional cameras and a novel algorithm to achieve flexible visual capabilities. Inspired by the adaptive vision of various species, our visual system addresses traditional stereo vision limitations, i.e., immutable camera alignment with narrow fields of view, by introducing a reconfigurable stereo vision system to robotics. Our key innovations include the reconfigurable stereo vision strategy that allows dynamic camera alignment, a robust depth measurement system utilizing a nonrectified geometrical method combined with a deep neural network for feature matching, and a geometrical compensation technique to enhance visual accuracy. Implemented on a metamorphic robot, this vision system demonstrates its great adaptability to various scenarios by switching its configurations of 316° monocular with 79° binocular field for fast target seeking and 242° monocular with 150° binocular field for detailed close inspection.
>
---
#### [replaced 015] Driving Through Uncertainty: Risk-Averse Control with LLM Commonsense for Autonomous Driving under Perception Deficits
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究自动驾驶中感知缺失下的风险控制问题，提出LLM-RCO框架，利用大语言模型融入驾驶常识，实现兼顾安全与灵活的决策。构建DriveLM-Deficit数据集并验证其在复杂场景中的有效性。**

- **链接: [https://arxiv.org/pdf/2503.07020v2](https://arxiv.org/pdf/2503.07020v2)**

> **作者:** Yuting Hu; Chenhui Xu; Ruiyang Qin; Dancheng Liu; Amir Nassereldine; Yiyu Shi; Jinjun Xiong
>
> **摘要:** Partial perception deficits can compromise autonomous vehicle safety by disrupting environmental understanding. Existing protocols typically default to entirely risk-avoidant actions such as immediate stops, which are detrimental to navigation goals and lack flexibility for rare driving scenarios. Yet, in cases of minor risk, halting the vehicle may be unnecessary, and more adaptive responses are preferable. In this paper, we propose LLM-RCO, a risk-averse framework leveraging large language models (LLMs) to integrate human-like driving commonsense into autonomous systems facing perception deficits. LLM-RCO features four key modules interacting with the dynamic driving environment: hazard inference, short-term motion planner, action condition verifier, and safety constraint generator, enabling proactive and context-aware actions in such challenging conditions. To enhance the driving decision-making of LLMs, we construct DriveLM-Deficit, a dataset of 53,895 video clips featuring deficits of safety-critical objects, annotated for LLM fine-tuning in hazard detection and motion planning. Extensive experiments in adverse driving conditions with the CARLA simulator demonstrate that LLM-RCO promotes proactive maneuvers over purely risk-averse actions in perception deficit scenarios, underscoring its value for boosting autonomous driving resilience against perception loss challenges.
>
---
#### [replaced 016] Aligning Humans and Robots via Reinforcement Learning from Implicit Human Feedback
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究人机对齐的强化学习，旨在解决稀疏奖励下策略学习困难及传统人类反馈干扰交互的问题。提出基于隐式脑电（ErrPs）反馈的RL框架，实现无需主动干预的高效策略学习，在机器人抓取任务中取得与密集人工奖励相当的效果。**

- **链接: [https://arxiv.org/pdf/2507.13171v2](https://arxiv.org/pdf/2507.13171v2)**

> **作者:** Suzie Kim; Hye-Bin Shin; Seong-Whan Lee
>
> **备注:** Accepted to IEEE Int. Conf. Syst., Man, Cybern. (SMC) 2025
>
> **摘要:** Conventional reinforcement learning (RL) ap proaches often struggle to learn effective policies under sparse reward conditions, necessitating the manual design of complex, task-specific reward functions. To address this limitation, rein forcement learning from human feedback (RLHF) has emerged as a promising strategy that complements hand-crafted rewards with human-derived evaluation signals. However, most existing RLHF methods depend on explicit feedback mechanisms such as button presses or preference labels, which disrupt the natural interaction process and impose a substantial cognitive load on the user. We propose a novel reinforcement learning from implicit human feedback (RLIHF) framework that utilizes non-invasive electroencephalography (EEG) signals, specifically error-related potentials (ErrPs), to provide continuous, implicit feedback without requiring explicit user intervention. The proposed method adopts a pre-trained decoder to transform raw EEG signals into probabilistic reward components, en abling effective policy learning even in the presence of sparse external rewards. We evaluate our approach in a simulation environment built on the MuJoCo physics engine, using a Kinova Gen2 robotic arm to perform a complex pick-and-place task that requires avoiding obstacles while manipulating target objects. The results show that agents trained with decoded EEG feedback achieve performance comparable to those trained with dense, manually designed rewards. These findings validate the potential of using implicit neural feedback for scalable and human-aligned reinforcement learning in interactive robotics.
>
---
#### [replaced 017] Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究语言条件下的行为克隆任务，旨在解决序列动作中累积误差导致的执行不连续与语义-物理错位问题。提出CCoL框架，通过视觉-语言-动作连续协同学习与双向跨注意力实现语义-物理对齐，提升动作克隆的连贯性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.14396v3](https://arxiv.org/pdf/2511.14396v3)**

> **作者:** Xiuxiu Qi; Yu Yang; Jiannong Cao; Luyao Bai; Chongshan Fan; Chengtai Cao; Hongpeng Wang
>
> **备注:** Accepted at AAAI 2026, the Project website is available at https://qhemu.github.io/CCoL/
>
> **摘要:** Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states.
>
---
#### [replaced 018] Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation -- Adios low-level controllers
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究模型预测控制在机器人实时控制中的应用，旨在降低计算开销并提升控制频率。作者提出Feedback-MPPI，通过引入基于灵敏度分析的局部反馈增益，实现快速闭环校正，无需每步重优化，显著提高动态任务中的稳定性与响应速度。**

- **链接: [https://arxiv.org/pdf/2506.14855v3](https://arxiv.org/pdf/2506.14855v3)**

> **作者:** Tommaso Belvedere; Michael Ziegltrum; Giulio Turrisi; Valerio Modugno
>
> **摘要:** Model Predictive Path Integral control is a powerful sampling-based approach suitable for complex robotic tasks due to its flexibility in handling nonlinear dynamics and non-convex costs. However, its applicability in real-time, highfrequency robotic control scenarios is limited by computational demands. This paper introduces Feedback-MPPI (F-MPPI), a novel framework that augments standard MPPI by computing local linear feedback gains derived from sensitivity analysis inspired by Riccati-based feedback used in gradient-based MPC. These gains allow for rapid closed-loop corrections around the current state without requiring full re-optimization at each timestep. We demonstrate the effectiveness of F-MPPI through simulations and real-world experiments on two robotic platforms: a quadrupedal robot performing dynamic locomotion on uneven terrain and a quadrotor executing aggressive maneuvers with onboard computation. Results illustrate that incorporating local feedback significantly improves control performance and stability, enabling robust, high-frequency operation suitable for complex robotic systems.
>
---
#### [replaced 019] Real-Time QP Solvers: A Concise Review and Practical Guide Towards Legged Robots
- **分类: cs.RO**

- **简介: 该论文综述并评测了面向腿式机器人的实时QP求解器，旨在解决资源受限下高效、鲁棒的优化问题。工作包括分类主流算法、分析性能指标，并提供基于速度、精度与能效的选型指南。**

- **链接: [https://arxiv.org/pdf/2510.21773v2](https://arxiv.org/pdf/2510.21773v2)**

> **作者:** Van Nam Dinh
>
> **备注:** 12 pages, 1 figure, 2 tables
>
> **摘要:** Quadratic programming (QP) underpins real-time robotics by enabling efficient, constrained optimization in state estimation, motion planning, and control. In legged locomotion and manipulation, essential modules like inverse dynamics, Model Predictive Control (MPC), and Whole-Body Control (WBC) are inherently QP-based, demanding reliable solutions amid tight timing, energy, and computational resources on embedded platforms. This paper presents a comprehensive analysis and benchmarking study of QP solvers for legged robotics. We begin by formulating the standard convex QP and classify solvers into principal algorithmic approaches: interior-point methods, active-set strategies, operator-splitting schemes, and augmented Lagrangian/proximal approaches, while also discussing solver code generation for fixed-structure QPs. Each solver is examined in terms of algorithmic structure, computational characteristics, and its ability to exploit problem structure and warm-starting. Performance is reviewed using publicly available benchmarks, with a focus on metrics such as computation time, constraint satisfaction, and robustness under perturbations. Unified comparison tables yield practical guidance for solver selection, underscoring trade-offs in speed, accuracy, and energy efficiency. Our findings emphasize the synergy between solvers, tasks, and hardware -- e.g., sparse structured IPMs for long-horizon MPC and dense active-set for high-frequency WBC to advance agile, autonomous legged systems, with emerging trends toward ill-conditioned, conic, and code-generated deployments.
>
---
#### [replaced 020] Decoupled Q-Chunking
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文研究离线强化学习中的价值函数学习，旨在解决TD方法的引导偏差与长动作块策略建模难、反应性差的问题。提出解耦Q分块算法，通过蒸馏机制构建短动作块策略，保留多步备份优势，提升性能。**

- **链接: [https://arxiv.org/pdf/2512.10926v2](https://arxiv.org/pdf/2512.10926v2)**

> **作者:** Qiyang Li; Seohong Park; Sergey Levine
>
> **备注:** 76 pages, 14 figures
>
> **摘要:** Temporal-difference (TD) methods learn state and action values efficiently by bootstrapping from their own future value predictions, but such a self-bootstrapping mechanism is prone to bootstrapping bias, where the errors in the value targets accumulate across steps and result in biased value estimates. Recent work has proposed to use chunked critics, which estimate the value of short action sequences ("chunks") rather than individual actions, speeding up value backup. However, extracting policies from chunked critics is challenging: policies must output the entire action chunk open-loop, which can be sub-optimal for environments that require policy reactivity and also challenging to model especially when the chunk length grows. Our key insight is to decouple the chunk length of the critic from that of the policy, allowing the policy to operate over shorter action chunks. We propose a novel algorithm that achieves this by optimizing the policy against a distilled critic for partial action chunks, constructed by optimistically backing up from the original chunked critic to approximate the maximum value achievable when a partial action chunk is extended to a complete one. This design retains the benefits of multi-step value propagation while sidestepping both the open-loop sub-optimality and the difficulty of learning action chunking policies for long action chunks. We evaluate our method on challenging, long-horizon offline goal-conditioned tasks and show that it reliably outperforms prior methods. Code: github.com/ColinQiyangLi/dqc.
>
---
#### [replaced 021] Iterative Compositional Data Generation for Robot Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究机器人控制中的数据生成任务，旨在解决多对象、多环境场景下演示数据昂贵且难覆盖所有任务组合的问题。提出语义组合扩散Transformer模型，通过分解状态转移并利用注意力学习组件交互，实现对未见任务的零样本生成，并结合迭代自优化提升性能。**

- **链接: [https://arxiv.org/pdf/2512.10891v2](https://arxiv.org/pdf/2512.10891v2)**

> **作者:** Anh-Quan Pham; Marcel Hussing; Shubhankar P. Patankar; Dani S. Bassett; Jorge Mendez-Mendez; Eric Eaton
>
> **备注:** Corrected reference chronological order and added acknowledgements; results unchanged
>
> **摘要:** Collecting robotic manipulation data is expensive, making it impractical to acquire demonstrations for the combinatorially large space of tasks that arise in multi-object, multi-robot, and multi-environment settings. While recent generative models can synthesize useful data for individual tasks, they do not exploit the compositional structure of robotic domains and struggle to generalize to unseen task combinations. We propose a semantic compositional diffusion transformer that factorizes transitions into robot-, object-, obstacle-, and objective-specific components and learns their interactions through attention. Once trained on a limited subset of tasks, we show that our model can zero-shot generate high-quality transitions from which we can learn control policies for unseen task combinations. Then, we introduce an iterative self-improvement procedure in which synthetic data is validated via offline reinforcement learning and incorporated into subsequent training rounds. Our approach substantially improves zero-shot performance over monolithic and hard-coded compositional baselines, ultimately solving nearly all held-out tasks and demonstrating the emergence of meaningful compositional structure in the learned representations.
>
---
#### [replaced 022] Gaze on the Prize: Shaping Visual Attention with Return-Guided Contrastive Learning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉强化学习任务，旨在解决高维图像中无关像素干扰导致的样本效率低问题。作者提出“Gaze on the Prize”框架，通过返回引导的对比学习，训练可学习的注视机制聚焦任务相关特征，提升样本效率并解决基线难以收敛的任务。**

- **链接: [https://arxiv.org/pdf/2510.08442v2](https://arxiv.org/pdf/2510.08442v2)**

> **作者:** Andrew Lee; Ian Chuang; Dechen Gao; Kai Fukazawa; Iman Soltani
>
> **备注:** Project page: https://andrewcwlee.github.io/gaze-on-the-prize
>
> **摘要:** Visual Reinforcement Learning (RL) agents must learn to act based on high-dimensional image data where only a small fraction of the pixels is task-relevant. This forces agents to waste exploration and computational resources on irrelevant features, leading to sample-inefficient and unstable learning. To address this, inspired by human visual foveation, we introduce Gaze on the Prize. This framework augments visual RL with a learnable foveal attention mechanism (Gaze), guided by a self-supervised signal derived from the agent's experience pursuing higher returns (the Prize). Our key insight is that return differences reveal what matters most: If two similar representations produce different outcomes, their distinguishing features are likely task-relevant, and the gaze should focus on them accordingly. This is realized through return-guided contrastive learning that trains the attention to distinguish between the features relevant to success and failure. We group similar visual representations into positives and negatives based on their return differences and use the resulting labels to construct contrastive triplets. These triplets provide the training signal that teaches the attention mechanism to produce distinguishable representations for states associated with different outcomes. Our method achieves up to 2.52x improvement in sample efficiency and can solve challenging tasks from the ManiSkill3 benchmark that the baseline fails to learn, without modifying the underlying algorithm or hyperparameters.
>
---
#### [replaced 023] From "Thumbs Up" to "10 out of 10": Reconsidering Scalar Feedback in Interactive Reinforcement Learning
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究交互式强化学习中的人类反馈形式，比较标量与二元反馈的效果。针对标量反馈噪声大、不稳定的问题，提出STEADY方法，通过重构正负反馈分布并重缩放，提升学习性能，验证了标量反馈的潜力。**

- **链接: [https://arxiv.org/pdf/2311.10284v2](https://arxiv.org/pdf/2311.10284v2)**

> **作者:** Hang Yu; Reuben M. Aronson; Katherine H. Allen; Elaine Schaertl Short
>
> **摘要:** Learning from human feedback is an effective way to improve robotic learning in exploration-heavy tasks. Compared to the wide application of binary human feedback, scalar human feedback has been used less because it is believed to be noisy and unstable. In this paper, we compare scalar and binary feedback, and demonstrate that scalar feedback benefits learning when properly handled. We collected binary or scalar feedback respectively from two groups of crowdworkers on a robot task. We found that when considering how consistently a participant labeled the same data, scalar feedback led to less consistency than binary feedback; however, the difference vanishes if small mismatches are allowed. Additionally, scalar and binary feedback show no significant differences in their correlations with key Reinforcement Learning targets. We then introduce Stabilizing TEacher Assessment DYnamics (STEADY) to improve learning from scalar feedback. Based on the idea that scalar feedback is muti-distributional, STEADY re-constructs underlying positive and negative feedback distributions and re-scales scalar feedback based on feedback statistics. We show that models trained with \textit{scalar feedback + STEADY } outperform baselines, including binary feedback and raw scalar feedback, in a robot reaching task with non-expert human feedback. Our results show that both binary feedback and scalar feedback are dynamic, and scalar feedback is a promising signal for use in interactive Reinforcement Learning.
>
---
