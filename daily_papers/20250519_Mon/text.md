# 自然语言处理 cs.CL

- **最新发布 81 篇**

- **更新 72 篇**

## 最新发布

#### [new 001] Learning When to Think: Shaping Adaptive Reasoning in R1-Style Models via Multi-Stage RL
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **简介: 该论文属于自适应推理优化任务，旨在解决大型推理模型(LRMs)因过度思考导致的计算效率低下问题。提出AutoThink框架，通过多阶段强化学习训练模型动态选择是否生成推理步骤，在R1模型基础上利用省略符触发控制，实现复杂问题显式推理、简单任务简洁回答。实验表明该方法在精度与效率间取得更优平衡。**

- **链接: [http://arxiv.org/pdf/2505.10832v1](http://arxiv.org/pdf/2505.10832v1)**

> **作者:** Songjun Tu; Jiahao Lin; Qichao Zhang; Xiangyu Tian; Linjing Li; Xiangyuan Lan; Dongbin Zhao
>
> **备注:** Project Page: https://github.com/TU2021/AutoThink
>
> **摘要:** Large reasoning models (LRMs) are proficient at generating explicit, step-by-step reasoning sequences before producing final answers. However, such detailed reasoning can introduce substantial computational overhead and latency, particularly for simple problems. To address this over-thinking problem, we explore how to equip LRMs with adaptive thinking capabilities: enabling them to dynamically decide whether or not to engage in explicit reasoning based on problem complexity. Building on R1-style distilled models, we observe that inserting a simple ellipsis ("...") into the prompt can stochastically trigger either a thinking or no-thinking mode, revealing a latent controllability in the reasoning behavior. Leveraging this property, we propose AutoThink, a multi-stage reinforcement learning (RL) framework that progressively optimizes reasoning policies via stage-wise reward shaping. AutoThink learns to invoke explicit reasoning only when necessary, while defaulting to succinct responses for simpler tasks. Experiments on five mainstream mathematical benchmarks demonstrate that AutoThink achieves favorable accuracy-efficiency trade-offs compared to recent prompting and RL-based pruning methods. It can be seamlessly integrated into any R1-style model, including both distilled and further fine-tuned variants. Notably, AutoThink improves relative accuracy by 6.4 percent while reducing token usage by 52 percent on DeepSeek-R1-Distill-Qwen-1.5B, establishing a scalable and adaptive reasoning paradigm for LRMs.
>
---
#### [new 002] GeoGrid-Bench: Can Foundation Models Understand Multimodal Gridded Geo-Spatial Data?
- **分类: cs.CL**

- **简介: 该论文属于多模态地理空间数据分析任务，旨在评估基础模型对网格化多模态地理数据的理解能力。为解决模型处理密集数值、时空依赖和多模态表示的挑战，构建了GeoGrid-Bench基准，包含16气候变量、150地点的3200个专家生成问答对。通过测试发现视觉语言模型表现最佳，并分析了不同模型在复杂时空任务中的优劣。**

- **链接: [http://arxiv.org/pdf/2505.10714v1](http://arxiv.org/pdf/2505.10714v1)**

> **作者:** Bowen Jiang; Yangxinyu Xie; Xiaomeng Wang; Jiashu He; Joshua Bergerson; John K Hutchison; Jordan Branham; Camillo J Taylor; Tanwi Mallick
>
> **摘要:** We present GeoGrid-Bench, a benchmark designed to evaluate the ability of foundation models to understand geo-spatial data in the grid structure. Geo-spatial datasets pose distinct challenges due to their dense numerical values, strong spatial and temporal dependencies, and unique multimodal representations including tabular data, heatmaps, and geographic visualizations. To assess how foundation models can support scientific research in this domain, GeoGrid-Bench features large-scale, real-world data covering 16 climate variables across 150 locations and extended time frames. The benchmark includes approximately 3,200 question-answer pairs, systematically generated from 8 domain expert-curated templates to reflect practical tasks encountered by human scientists. These range from basic queries at a single location and time to complex spatiotemporal comparisons across regions and periods. Our evaluation reveals that vision-language models perform best overall, and we provide a fine-grained analysis of the strengths and limitations of different foundation models in different geo-spatial tasks. This benchmark offers clearer insights into how foundation models can be effectively applied to geo-spatial data analysis and used to support scientific research.
>
---
#### [new 003] OntoURL: A Benchmark for Evaluating Large Language Models on Symbolic Ontological Understanding, Reasoning and Learning
- **分类: cs.CL**

- **简介: 该论文提出首个评估大语言模型（LLMs）处理形式化本体能力的基准OntoURL，属于符号知识处理任务。针对LLMs在结构化知识理解、推理和学习上的不足，作者构建包含15个任务/58,981问题的分类体系，测试20个开源模型，揭示其在推理和学习环节的显著缺陷，为LLM与形式知识融合提供评估标准。**

- **链接: [http://arxiv.org/pdf/2505.11031v1](http://arxiv.org/pdf/2505.11031v1)**

> **作者:** Xiao Zhang; Huiyuan Lai; Qianru Meng; Johan Bos
>
> **备注:** Paper submitted to NeruoIPS 2025 dataset and benchmark track
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across a range of natural language processing tasks, yet their ability to process structured symbolic knowledge remains underexplored. To address this gap, we propose a taxonomy of LLMs' ontological capabilities and introduce OntoURL, the first comprehensive benchmark designed to systematically evaluate LLMs' proficiency in handling ontologies -- formal, symbolic representations of domain knowledge through concepts, relationships, and instances. Based on the proposed taxonomy, OntoURL systematically assesses three dimensions: understanding, reasoning, and learning through 15 distinct tasks comprising 58,981 questions derived from 40 ontologies across 8 domains. Experiments with 20 open-source LLMs reveal significant performance differences across models, tasks, and domains, with current LLMs showing proficiency in understanding ontological knowledge but substantial weaknesses in reasoning and learning tasks. These findings highlight fundamental limitations in LLMs' capability to process symbolic knowledge and establish OntoURL as a critical benchmark for advancing the integration of LLMs with formal knowledge representations.
>
---
#### [new 004] Review-Instruct: A Review-Driven Multi-Turn Conversations Generation Method for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Review-Instruct框架，针对大语言模型在多轮对话中上下文连贯性不足的问题，通过"提问-响应-评审"多代理协同迭代生成对话数据，提升指令多样性与难度。基于Alpaca数据集构建多轮对话库并微调LLaMA2-13B模型，实验验证其在多项基准测试中超越现有方法，证实评审机制和多评审者设计对增强对话质量的关键作用。**

- **链接: [http://arxiv.org/pdf/2505.11010v1](http://arxiv.org/pdf/2505.11010v1)**

> **作者:** Jiangxu Wu; Cong Wang; TianHuang Su; Jun Yang; Haozhi Lin; Chao Zhang; Ming Peng; Kai Shi; SongPan Yang; BinQing Pan; ZiXian Li; Ni Yang; ZhenYu Yang
>
> **备注:** ACL2025 Accepted
>
> **摘要:** The effectiveness of large language models (LLMs) in conversational AI is hindered by their reliance on single-turn supervised fine-tuning (SFT) data, which limits contextual coherence in multi-turn dialogues. Existing methods for generating multi-turn dialogue data struggle to ensure both diversity and quality in instructions. To address this, we propose Review-Instruct, a novel framework that synthesizes multi-turn conversations through an iterative "Ask-Respond-Review" process involving three agent roles: a Candidate, multiple Reviewers, and a Chairman. The framework iteratively refines instructions by incorporating Reviewer feedback, enhancing dialogue diversity and difficulty. We construct a multi-turn dataset using the Alpaca dataset and fine-tune the LLaMA2-13B model. Evaluations on MT-Bench, MMLU-Pro, and Auto-Arena demonstrate significant improvements, achieving absolute gains of 2.9\% on MMLU-Pro and 2\% on MT-Bench compared to prior state-of-the-art models based on LLaMA2-13B. Ablation studies confirm the critical role of the Review stage and the use of multiple Reviewers in boosting instruction diversity and difficulty. Our work highlights the potential of review-driven, multi-agent frameworks for generating high-quality conversational data at scale.
>
---
#### [new 005] Tracr-Injection: Distilling Algorithms into Pre-trained Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型增强任务，旨在解决transformer架构理论符号能力与无监督数据实际学习效果间的差距。研究者提出tracr-injection方法，将RASP编程语言编写的算法直接蒸馏至预训练语言模型，通过注入三类算法验证其有效性，构建了可解释的子空间并提升模型分布外性能。**

- **链接: [http://arxiv.org/pdf/2505.10719v1](http://arxiv.org/pdf/2505.10719v1)**

> **作者:** Tomás Vergara-Browne; Álvaro Soto
>
> **备注:** ACL Findings 2025
>
> **摘要:** Motivated by the surge of large language models, there has been a push to formally characterize the symbolic abilities intrinsic to the transformer architecture. A programming language, called RASP, has been proposed, which can be directly compiled into transformer weights to implement these algorithms. However, the tasks that can be implemented in RASP are often uncommon to learn from natural unsupervised data, showing a mismatch between theoretical capabilities of the transformer architecture, and the practical learnability of these capabilities from unsupervised data. We propose tracr-injection, a method that allows us to distill algorithms written in RASP directly into a pre-trained language model. We showcase our method by injecting 3 different algorithms into a language model. We show how our method creates an interpretable subspace within the model's residual stream, which can be decoded into the variables present in the code of the RASP algorithm. Additionally, we found that the proposed method can improve out of distribution performance compared to our baseline, indicating that indeed a more symbolic mechanism is taking place in the inner workings of the model. We release the code used to run our experiments.
>
---
#### [new 006] Scaling Reasoning can Improve Factuality in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究如何通过扩展推理提升大语言模型在开放域问答中的事实准确性。针对长推理链是否增强事实性的问题，作者从大模型中提取推理轨迹并融合知识图谱路径，微调不同规模模型（含Qwen2.5架构）。实验表明小模型单次推理效果提升，增加测试计算量使准确率提高2-8%，验证了推理扩展的有效性。**

- **链接: [http://arxiv.org/pdf/2505.11140v1](http://arxiv.org/pdf/2505.11140v1)**

> **作者:** Mike Zhang; Johannes Bjerva; Russa Biswas
>
> **摘要:** Recent studies on large language model (LLM) reasoning capabilities have demonstrated promising improvements in model performance by leveraging a lengthy thinking process and additional computational resources during inference, primarily in tasks involving mathematical reasoning (Muennighoff et al., 2025). However, it remains uncertain if longer reasoning chains inherently enhance factual accuracy, particularly beyond mathematical contexts. In this work, we thoroughly examine LLM reasoning within complex open-domain question-answering (QA) scenarios. We initially distill reasoning traces from advanced, large-scale reasoning models (QwQ-32B and DeepSeek-R1-671B), then fine-tune a variety of models ranging from smaller, instruction-tuned variants to larger architectures based on Qwen2.5. To enrich reasoning traces, we introduce factual information from knowledge graphs in the form of paths into our reasoning traces. Our experimental setup includes four baseline approaches and six different instruction-tuned models evaluated across a benchmark of six datasets, encompassing over 22.6K questions. Overall, we carry out 168 experimental runs and analyze approximately 1.7 million reasoning traces. Our findings indicate that, within a single run, smaller reasoning models achieve noticeable improvements in factual accuracy compared to their original instruction-tuned counterparts. Moreover, our analysis demonstrates that adding test-time compute and token budgets factual accuracy consistently improves by 2-8%, further confirming the effectiveness of test-time scaling for enhancing performance and consequently improving reasoning accuracy in open-domain QA tasks. We release all the experimental artifacts for further research.
>
---
#### [new 007] Illusion or Algorithm? Investigating Memorization, Emergence, and Symbolic Processing in In-Context Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型机制分析任务，探究大语言模型上下文学习（ICL）的本质是数据记忆还是算法能力。通过构建实验任务和Pythia模型套件的多维度分析，证明ICL既非单纯记忆训练数据，也未形成独立符号算法，揭示了训练动态与模型能力对其的影响，为模型优化和安全评估提供依据。**

- **链接: [http://arxiv.org/pdf/2505.11004v1](http://arxiv.org/pdf/2505.11004v1)**

> **作者:** Jingcheng Niu; Subhabrata Dutta; Ahmed Elshabrawy; Harish Tayyar Madabushi; Iryna Gurevych
>
> **摘要:** Large-scale Transformer language models (LMs) trained solely on next-token prediction with web-scale data can solve a wide range of tasks after seeing just a few examples. The mechanism behind this capability, known as in-context learning (ICL), remains both controversial and poorly understood. Some studies argue that it is merely the result of memorizing vast amounts of data, while others contend that it reflects a fundamental, symbolic algorithmic development in LMs. In this work, we introduce a suite of investigative tasks and a novel method to systematically investigate ICL by leveraging the full Pythia scaling suite, including interim checkpoints that capture progressively larger amount of training data. By carefully exploring ICL performance on downstream tasks and simultaneously conducting a mechanistic analysis of the residual stream's subspace, we demonstrate that ICL extends beyond mere "memorization" of the training corpus, yet does not amount to the implementation of an independent symbolic algorithm. Our results also clarify several aspects of ICL, including the influence of training dynamics, model capabilities, and elements of mechanistic interpretability. Overall, our work advances the understanding of ICL and its implications, offering model developers insights into potential improvements and providing AI security practitioners with a basis for more informed guidelines.
>
---
#### [new 008] Reconstructing Syllable Sequences in Abugida Scripts with Incomplete Inputs
- **分类: cs.CL; cs.LG; I.2.7**

- **简介: 该论文研究Abugida文字的音节序列预测任务，解决不完整输入（如缺失辅音、元音或音节）下的重建问题。基于Transformer模型对六种亚洲语言实验，发现辅音序列对预测准确率贡献显著，而元音重建更具挑战。研究验证了模型在部分缺失和遮盖音节场景的鲁棒性，为文本纠错等应用提供参考。**

- **链接: [http://arxiv.org/pdf/2505.11008v1](http://arxiv.org/pdf/2505.11008v1)**

> **作者:** Ye Kyaw Thu; Thazin Myint Oo
>
> **备注:** 14 pages, 2 figures, 6 tables, 1 listing
>
> **摘要:** This paper explores syllable sequence prediction in Abugida languages using Transformer-based models, focusing on six languages: Bengali, Hindi, Khmer, Lao, Myanmar, and Thai, from the Asian Language Treebank (ALT) dataset. We investigate the reconstruction of complete syllable sequences from various incomplete input types, including consonant sequences, vowel sequences, partial syllables (with random character deletions), and masked syllables (with fixed syllable deletions). Our experiments reveal that consonant sequences play a critical role in accurate syllable prediction, achieving high BLEU scores, while vowel sequences present a significantly greater challenge. The model demonstrates robust performance across tasks, particularly in handling partial and masked syllable reconstruction, with strong results for tasks involving consonant information and syllable masking. This study advances the understanding of sequence prediction for Abugida languages and provides practical insights for applications such as text prediction, spelling correction, and data augmentation in these scripts.
>
---
#### [new 009] Is Compression Really Linear with Code Intelligence?
- **分类: cs.CL**

- **简介: 该论文属于代码智能评估领域，研究数据压缩与代码大模型(LLMs)性能的关系。针对现有线性关系假设的局限性，提出多语言/多任务评测框架，并设计Format Annealing训练方法，通过GitHub数据验证发现代码智能与压缩效能(bits-per-character)呈对数关系，修正了线性假设的认知偏差。**

- **链接: [http://arxiv.org/pdf/2505.11441v1](http://arxiv.org/pdf/2505.11441v1)**

> **作者:** Xianzhen Luo; Shijie Xuyang; Tianhao Cheng; Zheng Chu; Houyi Li; ziqi wang; Siming Huang; Qingfu Zhu; Qiufeng Wang; Xiangyu Zhang; Shuigeng Zhou; Wanxiang Che
>
> **备注:** work in progress
>
> **摘要:** Understanding the relationship between data compression and the capabilities of Large Language Models (LLMs) is crucial, especially in specialized domains like code intelligence. Prior work posited a linear relationship between compression and general intelligence. However, it overlooked the multifaceted nature of code that encompasses diverse programming languages and tasks, and struggled with fair evaluation of modern Code LLMs. We address this by evaluating a diverse array of open-source Code LLMs on comprehensive multi-language, multi-task code benchmarks. To address the challenge of efficient and fair evaluation of pre-trained LLMs' code intelligence, we introduce \textit{Format Annealing}, a lightweight, transparent training methodology designed to assess the intrinsic capabilities of these pre-trained models equitably. Compression efficacy, measured as bits-per-character (BPC), is determined using a novel, large-scale, and previously unseen code validation set derived from GitHub. Our empirical results reveal a fundamental logarithmic relationship between measured code intelligence and BPC. This finding refines prior hypotheses of linearity, which we suggest are likely observations of the logarithmic curve's tail under specific, limited conditions. Our work provides a more nuanced understanding of compression's role in developing code intelligence and contributes a robust evaluation framework in the code domain.
>
---
#### [new 010] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对规则检索任务，解决现有方法因查询事实与规则语义差距导致的低准确率问题。提出SIAR方法利用大语言模型从查询中归纳潜在推理规则以增强检索，并设计R³机制通过评估规则可实例化性和推理帮助性重排结果，提升检索质量与下游推理性能。**

- **链接: [http://arxiv.org/pdf/2505.10870v1](http://arxiv.org/pdf/2505.10870v1)**

> **作者:** Ziyang Huang; Wangtao Sun; Jun Zhao; Kang Liu
>
> **备注:** ACL 2025
>
> **摘要:** This paper systematically addresses the challenges of rule retrieval, a crucial yet underexplored area. Vanilla retrieval methods using sparse or dense retrievers to directly search for relevant rules to support downstream reasoning, often suffer from low accuracy. This is primarily due to a significant semantic gap between the instantiated facts in the queries and the abstract representations of the rules. Such misalignment results in suboptimal retrieval quality, which in turn negatively impacts reasoning performance. To overcome these challenges, we propose Self-Induction Augmented Retrieval (SIAR), a novel approach that utilizes Large Language Models (LLMs) to induce potential inferential rules that might offer benefits for reasoning by abstracting the underlying knowledge and logical structure in queries. These induced rules are then used for query augmentation to improve retrieval effectiveness. Additionally, we introduce Rule Relevance ReEstimate (R$^3$), a method that re-estimates the relevance of retrieved rules by assessing whether the abstract knowledge they contain can be instantiated to align with the facts in the queries and the helpfulness for reasoning. Extensive experiments across various settings demonstrate the effectiveness and versatility of our proposed methods.
>
---
#### [new 011] Connecting the Dots: A Chain-of-Collaboration Prompting Framework for LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出Cochain框架，解决大语言模型在业务工作流中单代理协作困难及多代理资源浪费问题。通过集成知识图谱和动态提示树，降低协作成本，提升任务执行效率。实验证明其性能优于传统提示工程和多代理系统，小模型结合Cochain甚至超越GPT-4。**

- **链接: [http://arxiv.org/pdf/2505.10936v1](http://arxiv.org/pdf/2505.10936v1)**

> **作者:** Jiaxing Zhao; Hongbin Xie; Yuzhen Lei; Xuan Song; Zhuoran Shi; Lianxin Li; Shuangxue Liu; Haoran Zhang
>
> **备注:** 34 pages, 20 figures
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive performance in executing complex reasoning tasks. Chain-of-thought effectively enhances reasoning capabilities by unlocking the potential of large models, while multi-agent systems provide more comprehensive solutions by integrating collective intelligence of multiple agents. However, both approaches face significant limitations. Single-agent with chain-of-thought, due to the inherent complexity of designing cross-domain prompts, faces collaboration challenges. Meanwhile, multi-agent systems consume substantial tokens and inevitably dilute the primary problem, which is particularly problematic in business workflow tasks. To address these challenges, we propose Cochain, a collaboration prompting framework that effectively solves business workflow collaboration problem by combining knowledge and prompts at a reduced cost. Specifically, we construct an integrated knowledge graph that incorporates knowledge from multiple stages. Furthermore, by maintaining and retrieving a prompts tree, we can obtain prompt information relevant to other stages of the business workflow. We perform extensive evaluations of Cochain across multiple datasets, demonstrating that Cochain outperforms all baselines in both prompt engineering and multi-agent LLMs. Additionally, expert evaluation results indicate that the use of a small model in combination with Cochain outperforms GPT-4.
>
---
#### [new 012] Disentangling Reasoning and Knowledge in Medical Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学大模型评估任务，旨在区分医学推理与知识记忆。现有基准混淆二者，作者使用PubMedBERT将11个数据集拆分为知识/推理子集（准确率81%），发现仅32.8%问题需复杂推理。通过测试发现模型推理能力普遍弱于知识表现，提出BioMed-R1模型（强化学习+微调），提升了推理性能。**

- **链接: [http://arxiv.org/pdf/2505.11462v1](http://arxiv.org/pdf/2505.11462v1)**

> **作者:** Rahul Thapa; Qingyang Wu; Kevin Wu; Harrison Zhang; Angela Zhang; Eric Wu; Haotian Ye; Suhana Bedi; Nevin Aresh; Joseph Boen; Shriya Reddy; Ben Athiwaratkun; Shuaiwen Leon Song; James Zou
>
> **摘要:** Medical reasoning in large language models (LLMs) aims to emulate clinicians' diagnostic thinking, but current benchmarks such as MedQA-USMLE, MedMCQA, and PubMedQA often mix reasoning with factual recall. We address this by separating 11 biomedical QA benchmarks into reasoning- and knowledge-focused subsets using a PubMedBERT classifier that reaches 81 percent accuracy, comparable to human performance. Our analysis shows that only 32.8 percent of questions require complex reasoning. We evaluate biomedical models (HuatuoGPT-o1, MedReason, m1) and general-domain models (DeepSeek-R1, o4-mini, Qwen3), finding consistent gaps between knowledge and reasoning performance. For example, m1 scores 60.5 on knowledge but only 47.1 on reasoning. In adversarial tests where models are misled with incorrect initial reasoning, biomedical models degrade sharply, while larger or RL-trained general models show more robustness. To address this, we train BioMed-R1 using fine-tuning and reinforcement learning on reasoning-heavy examples. It achieves the strongest performance among similarly sized models. Further gains may come from incorporating clinical case reports and training with adversarial and backtracking scenarios.
>
---
#### [new 013] Have Multimodal Large Language Models (MLLMs) Really Learned to Tell the Time on Analog Clocks?
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于多模态理解任务，探究多模态大语言模型（MLLMs）能否真正理解模拟时钟时间。针对模型因训练数据缺乏时钟图像导致的识别缺陷，研究通过测试GPT-4.1分析失败原因，验证微调效果，并设计多样化时钟实验揭示模型抽象推理与泛化能力的局限性。**

- **链接: [http://arxiv.org/pdf/2505.10862v1](http://arxiv.org/pdf/2505.10862v1)**

> **作者:** Tairan Fu; Miguel González; Javier Conde; Elena Merino-Gómez; Pedro Reviriego
>
> **备注:** 6 pages, 5 figures, 2 tables
>
> **摘要:** Multimodal Large Language Models which can answer complex questions on an image struggle to tell the time on analog clocks. This is probably due to the lack of images with clocks at different times in their training set. In this work we explore this issue with one of the latest MLLMs: GPT-4.1 to understand why MLLMs fail to tell the time and whether fine-tuning can solve the problem. The results show how models are making progress in reading the time on analog clocks. But have they really learned to do it, or have they only learned patterns in their training datasets? In this work we put the models to the test with different clocks to illustrate the limitations of MLLMs to abstract and generalize.
>
---
#### [new 014] Multimodal Event Detection: Current Approaches and Defining the New Playground through LLMs and VLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究社交媒体多模态事件检测任务，解决传统单模态系统应对多模态数据传播的不足。通过对比单模态模型、多模态融合模型及生成模型（如GPT-4o），发现多模态方法优于单模态，但生成模型因无法准确生成事件类别，精度落后于监督方法。同时揭示生成模型擅长处理网络语言现象而监督方法难以应对的特性。**

- **链接: [http://arxiv.org/pdf/2505.10836v1](http://arxiv.org/pdf/2505.10836v1)**

> **作者:** Abhishek Dey; Aabha Bothera; Samhita Sarikonda; Rishav Aryan; Sanjay Kumar Podishetty; Akshay Havalgi; Gaurav Singh; Saurabh Srivastava
>
> **备注:** Accepted at NLDB 2025
>
> **摘要:** In this paper, we study the challenges of detecting events on social media, where traditional unimodal systems struggle due to the rapid and multimodal nature of data dissemination. We employ a range of models, including unimodal ModernBERT and ConvNeXt-V2, multimodal fusion techniques, and advanced generative models like GPT-4o, and LLaVA. Additionally, we also study the effect of providing multimodal generative models (such as GPT-4o) with a single modality to assess their efficacy. Our results indicate that while multimodal approaches notably outperform unimodal counterparts, generative approaches despite having a large number of parameters, lag behind supervised methods in precision. Furthermore, we also found that they lag behind instruction-tuned models because of their inability to generate event classes correctly. During our error analysis, we discovered that common social media issues such as leet speak, text elongation, etc. are effectively handled by generative approaches but are hard to tackle using supervised approaches.
>
---
#### [new 015] Improving Assembly Code Performance with Large Language Models via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.PF; cs.PL; cs.SE**

- **简介: 该论文研究利用强化学习训练大语言模型（LLMs）优化汇编代码性能的任务。针对LLM在底层代码优化潜力未被充分挖掘的问题，提出基于PPO的强化学习框架，结合功能正确性和执行速度设计奖励函数，构建8,072程序基准集。模型Qwen2.5-Coder-7B-PPO在保持96%正确率的同时实现1.47倍于gcc -O3的加速，验证了LLMs作为汇编优化器的有效性。**

- **链接: [http://arxiv.org/pdf/2505.11480v1](http://arxiv.org/pdf/2505.11480v1)**

> **作者:** Anjiang Wei; Tarun Suresh; Huanmi Tan; Yinglun Xu; Gagandeep Singh; Ke Wang; Alex Aiken
>
> **摘要:** Large language models (LLMs) have demonstrated strong performance across a wide range of programming tasks, yet their potential for code optimization remains underexplored. This work investigates whether LLMs can optimize the performance of assembly code, where fine-grained control over execution enables improvements that are difficult to express in high-level languages. We present a reinforcement learning framework that trains LLMs using Proximal Policy Optimization (PPO), guided by a reward function that considers both functional correctness, validated through test cases, and execution performance relative to the industry-standard compiler gcc -O3. To support this study, we introduce a benchmark of 8,072 real-world programs. Our model, Qwen2.5-Coder-7B-PPO, achieves 96.0% test pass rates and an average speedup of 1.47x over the gcc -O3 baseline, outperforming all 20 other models evaluated, including Claude-3.7-sonnet. These results indicate that reinforcement learning can unlock the potential of LLMs to serve as effective optimizers for assembly code performance.
>
---
#### [new 016] The Way We Prompt: Conceptual Blending, Neural Dynamics, and Prompt-Induced Transitions in LLMs
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文属于认知科学与AI交叉研究，旨在揭示大语言模型（LLMs）通过提示混合语义的机制。通过概念融合理论框架，分析提示诱导的认知转变和幻觉现象，对比人工与生物认知结构差异，提出提示工程可作为探究语义深层结构的科学方法。**

- **链接: [http://arxiv.org/pdf/2505.10948v1](http://arxiv.org/pdf/2505.10948v1)**

> **作者:** Makoto Sato
>
> **摘要:** Large language models (LLMs), inspired by neuroscience, exhibit behaviors that often evoke a sense of personality and intelligence-yet the mechanisms behind these effects remain elusive. Here, we operationalize Conceptual Blending Theory (CBT) as an experimental framework, using prompt-based methods to reveal how LLMs blend and compress meaning. By systematically investigating Prompt-Induced Transitions (PIT) and Prompt-Induced Hallucinations (PIH), we uncover structural parallels and divergences between artificial and biological cognition. Our approach bridges linguistics, neuroscience, and empirical AI research, demonstrating that human-AI collaboration can serve as a living prototype for the future of cognitive science. This work proposes prompt engineering not just as a technical tool, but as a scientific method for probing the deep structure of meaning itself.
>
---
#### [new 017] Ranked Voting based Self-Consistency of Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对链式思维推理中自洽性评估的不足，提出基于排名投票的方法，提升大语言模型的推理性能。通过生成多答案并采用三种排名投票策略（即时复选、博尔达计数、平均倒数排名），解决传统多数投票忽略潜在答案的问题。实验在六类问答任务中验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.10772v1](http://arxiv.org/pdf/2505.10772v1)**

> **作者:** Weiqin Wang; Yile Wang; Hui Huang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Majority voting is considered an effective method to enhance chain-of-thought reasoning, as it selects the answer with the highest "self-consistency" among different reasoning paths (Wang et al., 2023). However, previous chain-of-thought reasoning methods typically generate only a single answer in each trial, thereby ignoring the possibility of other potential answers. As a result, these alternative answers are often overlooked in subsequent voting processes. In this work, we propose to generate ranked answers in each reasoning process and conduct ranked voting among multiple ranked answers from different responses, thereby making the overall self-consistency more reliable. Specifically, we use three ranked voting methods: Instant-runoff voting, Borda count voting, and mean reciprocal rank voting. We validate our methods on six datasets, including three multiple-choice and three open-ended question-answering tasks, using both advanced open-source and closed-source large language models. Extensive experimental results indicate that our proposed method outperforms the baselines, showcasing the potential of leveraging the information of ranked answers and using ranked voting to improve reasoning performance. The code is available at https://github.com/szu-tera/RankedVotingSC.
>
---
#### [new 018] HelpSteer3-Preference: Open Human-Annotated Preference Data across Diverse Tasks and Languages
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理领域，聚焦通过增强偏好数据集质量提升语言模型的强化学习效果。针对现有公开偏好数据多样性不足的问题，提出了覆盖STEM、编程和多语言场景的4万标注样本数据集HelpSteer3-Preference。基于该数据训练的奖励模型在基准测试中取得突破性性能（82.4% RM-Bench），较现有模型提升约10%，并验证了其在生成模型与策略对齐中的应用价值。**

- **链接: [http://arxiv.org/pdf/2505.11475v1](http://arxiv.org/pdf/2505.11475v1)**

> **作者:** Zhilin Wang; Jiaqi Zeng; Olivier Delalleau; Hoo-Chang Shin; Felipe Soares; Alexander Bukharin; Ellie Evans; Yi Dong; Oleksii Kuchaiev
>
> **备注:** 38 pages, 2 figures
>
> **摘要:** Preference datasets are essential for training general-domain, instruction-following language models with Reinforcement Learning from Human Feedback (RLHF). Each subsequent data release raises expectations for future data collection, meaning there is a constant need to advance the quality and diversity of openly available preference data. To address this need, we introduce HelpSteer3-Preference, a permissively licensed (CC-BY-4.0), high-quality, human-annotated preference dataset comprising of over 40,000 samples. These samples span diverse real-world applications of large language models (LLMs), including tasks relating to STEM, coding and multilingual scenarios. Using HelpSteer3-Preference, we train Reward Models (RMs) that achieve top performance on RM-Bench (82.4%) and JudgeBench (73.7%). This represents a substantial improvement (~10% absolute) over the previously best-reported results from existing RMs. We demonstrate HelpSteer3-Preference can also be applied to train Generative RMs and how policy models can be aligned with RLHF using our RMs. Dataset (CC-BY-4.0): https://huggingface.co/datasets/nvidia/HelpSteer3#preference
>
---
#### [new 019] NoPE: The Counting Power of Transformers with No Positional Encodings
- **分类: cs.CL; cs.FL; cs.LG**

- **简介: 该论文属理论计算语言学，研究无位置编码的Transformer（NoPE）表达能力。通过平均硬注意力机制，NoPE可表达半代数集（多元多项式方程的非负整数解），远超正则语言，但无法处理简单计数如奇偶性。证明了其表达能力与不可判定问题的关联，并对比电路复杂度类TC⁰，填补了现有模型的理论空白。**

- **链接: [http://arxiv.org/pdf/2505.11199v1](http://arxiv.org/pdf/2505.11199v1)**

> **作者:** Chris Köcher; Alexander Kozachinskiy; Anthony Widjaja Lin; Marco Sälzer; Georg Zetzsche
>
> **摘要:** Positional Encodings (PEs) seem to be indispensable for ensuring expressiveness of transformers; without them attention transformers reduce to a bag-of-word model. NoPE-transformers (i.e. with No PEs) with unique hard attention mechanisms were very recently shown to only be able to express regular languages, i.e., with limited counting ability. This paper shows that, with average hard attention mechanisms, NoPE-transformers are still surprisingly expressive: they can express counting languages corresponding to nonnegative integer solutions to multivariate polynomial equations (i.e. Diophantine equations), reasoning about which is well-known to be undecidable. In fact, we provide a precise characterization of languages expressible by Average Hard Attention NoPE-Transformers (NoPE-AHATs): they correspond precisely to what we call \emph{semi-algebraic sets}, i.e., finite unions of sets of nonnegative integer solutions to systems of multivariate polynomial inequations. We obtain several interesting consequences of our characterization. Firstly, NoPE-transformers can express counting properties that are far more complex than established models like simplified counter machines and Petri nets, but cannot express a very simple counting property of PARITY. Secondly, the problem of analyzing NoPE-transformers is undecidable, e.g., whether a given NoPE transformer classifies all input strings in one class. To complement our results, we exhibit a counting language that is not expressible by average hard attention transformers even with arbitrary PEs but is expressible in the circuit complexity class TC$^0$, answering an open problem.
>
---
#### [new 020] LegoSLM: Connecting LLM with Speech Encoder using CTC Posteriors
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音语言处理任务，旨在解决预训练语音编码器与LLM结合性能不佳的问题。提出LegoSLM框架，通过CTC后验矩阵将语音特征转换为LLM词表概率分布，重构伪音频嵌入并与文本嵌入融合，实现ASR与语音翻译性能提升（49% WERR）。支持模块化替换语音编码器，并通过温度控制调节多模态权重。**

- **链接: [http://arxiv.org/pdf/2505.11352v1](http://arxiv.org/pdf/2505.11352v1)**

> **作者:** Rao Ma; Tongzhou Chen; Kartik Audhkhasi; Bhuvana Ramabhadran
>
> **摘要:** Recently, large-scale pre-trained speech encoders and Large Language Models (LLMs) have been released, which show state-of-the-art performance on a range of spoken language processing tasks including Automatic Speech Recognition (ASR). To effectively combine both models for better performance, continuous speech prompts, and ASR error correction have been adopted. However, these methods are prone to suboptimal performance or are inflexible. In this paper, we propose a new paradigm, LegoSLM, that bridges speech encoders and LLMs using the ASR posterior matrices. The speech encoder is trained to generate Connectionist Temporal Classification (CTC) posteriors over the LLM vocabulary, which are used to reconstruct pseudo-audio embeddings by computing a weighted sum of the LLM input embeddings. These embeddings are concatenated with text embeddings in the LLM input space. Using the well-performing USM and Gemma models as an example, we demonstrate that our proposed LegoSLM method yields good performance on both ASR and speech translation tasks. By connecting USM with Gemma models, we can get an average of 49% WERR over the USM-CTC baseline on 8 MLS testsets. The trained model also exhibits modularity in a range of settings -- after fine-tuning the Gemma model weights, the speech encoder can be switched and combined with the LLM in a zero-shot fashion. Additionally, we propose to control the decode-time influence of the USM and LLM using a softmax temperature, which shows effectiveness in domain adaptation.
>
---
#### [new 021] When Thinking Fails: The Pitfalls of Reasoning for Instruction-Following in LLMs
- **分类: cs.CL**

- **简介: 该论文研究增强推理的大语言模型（RLLMs）在指令遵循任务中的缺陷，发现显式思维链（CoT）推理会降低准确性。通过评估15个模型和注意力分析，揭示推理导致注意力偏移的关键问题，并提出选择性推理策略（如分类器筛选）有效缓解性能损失。**

- **链接: [http://arxiv.org/pdf/2505.11423v1](http://arxiv.org/pdf/2505.11423v1)**

> **作者:** Xiaomin Li; Zhou Yu; Zhiwei Zhang; Xupeng Chen; Ziji Zhang; Yingying Zhuang; Narayanan Sadagopan; Anurag Beniwal
>
> **摘要:** Reasoning-enhanced large language models (RLLMs), whether explicitly trained for reasoning or prompted via chain-of-thought (CoT), have achieved state-of-the-art performance on many complex reasoning tasks. However, we uncover a surprising and previously overlooked phenomenon: explicit CoT reasoning can significantly degrade instruction-following accuracy. Evaluating 15 models on two benchmarks: IFEval (with simple, rule-verifiable constraints) and ComplexBench (with complex, compositional constraints), we consistently observe performance drops when CoT prompting is applied. Through large-scale case studies and an attention-based analysis, we identify common patterns where reasoning either helps (e.g., with formatting or lexical precision) or hurts (e.g., by neglecting simple constraints or introducing unnecessary content). We propose a metric, constraint attention, to quantify model focus during generation and show that CoT reasoning often diverts attention away from instruction-relevant tokens. To mitigate these effects, we introduce and evaluate four strategies: in-context learning, self-reflection, self-selective reasoning, and classifier-selective reasoning. Our results demonstrate that selective reasoning strategies, particularly classifier-selective reasoning, can substantially recover lost performance. To our knowledge, this is the first work to systematically expose reasoning-induced failures in instruction-following and offer practical mitigation strategies.
>
---
#### [new 022] GuideBench: Benchmarking Domain-Oriented Guideline Following for LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出GuideBench基准，用于评估大语言模型（LLM）代理在特定领域遵循动态规则的能力。任务属于领域导向的指令遵循评测，旨在解决现有基准缺乏对领域规则冲突、频繁更新及人类偏好对齐的评估问题。通过测试LLM在多样化规则遵守、规则更新鲁棒性和人类偏好匹配三方面的表现，揭示其提升空间。**

- **链接: [http://arxiv.org/pdf/2505.11368v1](http://arxiv.org/pdf/2505.11368v1)**

> **作者:** Lingxiao Diao; Xinyue Xu; Wanxuan Sun; Cheng Yang; Zhuosheng Zhang
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Large language models (LLMs) have been widely deployed as autonomous agents capable of following user instructions and making decisions in real-world applications. Previous studies have made notable progress in benchmarking the instruction following capabilities of LLMs in general domains, with a primary focus on their inherent commonsense knowledge. Recently, LLMs have been increasingly deployed as domain-oriented agents, which rely on domain-oriented guidelines that may conflict with their commonsense knowledge. These guidelines exhibit two key characteristics: they consist of a wide range of domain-oriented rules and are subject to frequent updates. Despite these challenges, the absence of comprehensive benchmarks for evaluating the domain-oriented guideline following capabilities of LLMs presents a significant obstacle to their effective assessment and further development. In this paper, we introduce GuideBench, a comprehensive benchmark designed to evaluate guideline following performance of LLMs. GuideBench evaluates LLMs on three critical aspects: (i) adherence to diverse rules, (ii) robustness to rule updates, and (iii) alignment with human preferences. Experimental results on a range of LLMs indicate substantial opportunities for improving their ability to follow domain-oriented guidelines.
>
---
#### [new 023] SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought Reasoning
- **分类: cs.CL**

- **简介: 该论文属于测试时扩展任务，旨在提升推理性能而不修改模型参数。针对连续潜在空间推理中路径多样性受限的问题，提出SoftCoT++，通过多初始令牌扰动和对比学习增强软思维表征多样性。实验证明其在五个基准上超越现有方法，并兼容传统扩展技术。**

- **链接: [http://arxiv.org/pdf/2505.11484v1](http://arxiv.org/pdf/2505.11484v1)**

> **作者:** Yige Xu; Xu Guo; Zhiwei Zeng; Chunyan Miao
>
> **备注:** 14 pages
>
> **摘要:** Test-Time Scaling (TTS) refers to approaches that improve reasoning performance by allocating extra computation during inference, without altering the model's parameters. While existing TTS methods operate in a discrete token space by generating more intermediate steps, recent studies in Coconut and SoftCoT have demonstrated that thinking in the continuous latent space can further enhance the reasoning performance. Such latent thoughts encode informative thinking without the information loss associated with autoregressive token generation, sparking increased interest in continuous-space reasoning. Unlike discrete decoding, where repeated sampling enables exploring diverse reasoning paths, latent representations in continuous space are fixed for a given input, which limits diverse exploration, as all decoded paths originate from the same latent thought. To overcome this limitation, we introduce SoftCoT++ to extend SoftCoT to the Test-Time Scaling paradigm by enabling diverse exploration of thinking paths. Specifically, we perturb latent thoughts via multiple specialized initial tokens and apply contrastive learning to promote diversity among soft thought representations. Experiments across five reasoning benchmarks and two distinct LLM architectures demonstrate that SoftCoT++ significantly boosts SoftCoT and also outperforms SoftCoT with self-consistency scaling. Moreover, it shows strong compatibility with conventional scaling techniques such as self-consistency. Source code is available at https://github.com/xuyige/SoftCoT.
>
---
#### [new 024] A Survey on the Safety and Security Threats of Computer-Using Agents: JARVIS or Ultron?
- **分类: cs.CL; cs.AI; cs.CR; cs.CV; cs.SE**

- **简介: 该论文为系统化综述，研究计算机使用代理（CUAs）的安全威胁。通过文献分析，定义CUA安全分析框架，分类现有威胁，提出防御策略，总结评估指标，为未来研究提供基础。**

- **链接: [http://arxiv.org/pdf/2505.10924v1](http://arxiv.org/pdf/2505.10924v1)**

> **作者:** Ada Chen; Yongjiang Wu; Junyuan Zhang; Shu Yang; Jen-tse Huang; Kun Wang; Wenxuan Wang; Shuai Wang
>
> **摘要:** Recently, AI-driven interactions with computing devices have advanced from basic prototype tools to sophisticated, LLM-based systems that emulate human-like operations in graphical user interfaces. We are now witnessing the emergence of \emph{Computer-Using Agents} (CUAs), capable of autonomously performing tasks such as navigating desktop applications, web pages, and mobile apps. However, as these agents grow in capability, they also introduce novel safety and security risks. Vulnerabilities in LLM-driven reasoning, with the added complexity of integrating multiple software components and multimodal inputs, further complicate the security landscape. In this paper, we present a systematization of knowledge on the safety and security threats of CUAs. We conduct a comprehensive literature review and distill our findings along four research objectives: \textit{\textbf{(i)}} define the CUA that suits safety analysis; \textit{\textbf{(ii)} } categorize current safety threats among CUAs; \textit{\textbf{(iii)}} propose a comprehensive taxonomy of existing defensive strategies; \textit{\textbf{(iv)}} summarize prevailing benchmarks, datasets, and evaluation metrics used to assess the safety and performance of CUAs. Building on these insights, our work provides future researchers with a structured foundation for exploring unexplored vulnerabilities and offers practitioners actionable guidance in designing and deploying secure Computer-Using Agents.
>
---
#### [new 025] No Gold Standard, No Problem: Reference-Free Evaluation of Taxonomies
- **分类: cs.CL**

- **简介: 该论文研究分类法质量评估任务，解决无黄金标准时的评价问题。作者提出两个无参考指标：基于语义与分类相似性相关性的鲁棒性评估，以及利用自然语言推理的逻辑合理性检测。方法在五个分类数据集上验证，与人工标准F1分数呈良好相关性。**

- **链接: [http://arxiv.org/pdf/2505.11470v1](http://arxiv.org/pdf/2505.11470v1)**

> **作者:** Pascal Wullschleger; Majid Zarharan; Donnacha Daly; Marc Pouly; Jennifer Foster
>
> **摘要:** We introduce two reference-free metrics for quality evaluation of taxonomies. The first metric evaluates robustness by calculating the correlation between semantic and taxonomic similarity, covering a type of error not handled by existing metrics. The second uses Natural Language Inference to assess logical adequacy. Both metrics are tested on five taxonomies and are shown to correlate well with F1 against gold-standard taxonomies.
>
---
#### [new 026] Benchmarking Critical Questions Generation: A Challenging Reasoning Task for Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究批判性问题生成（CQs-Gen）任务，旨在通过提问揭示论证缺陷以提升批判性思维。针对领域缺乏数据集和评估标准的问题，作者构建了首个大规模人工标注数据集，提出基于LLM的自动评估方法（与人类判断高度相关），并通过零样本测试11个LLM建立基准，揭示任务难度，推动自动化推理与人类思维研究。**

- **链接: [http://arxiv.org/pdf/2505.11341v1](http://arxiv.org/pdf/2505.11341v1)**

> **作者:** Banca Calvo Figueras; Rodrigo Agerri
>
> **摘要:** The task of Critical Questions Generation (CQs-Gen) aims to foster critical thinking by enabling systems to generate questions that expose assumptions and challenge the reasoning in arguments. Despite growing interest in this area, progress has been hindered by the lack of suitable datasets and automatic evaluation standards. This work presents a comprehensive approach to support the development and benchmarking of systems for this task. We construct the first large-scale manually-annotated dataset. We also investigate automatic evaluation methods and identify a reference-based technique using large language models (LLMs) as the strategy that best correlates with human judgments. Our zero-shot evaluation of 11 LLMs establishes a strong baseline while showcasing the difficulty of the task. Data, code, and a public leaderboard are provided to encourage further research not only in terms of model performance, but also to explore the practical benefits of CQs-Gen for both automated reasoning and human critical thinking.
>
---
#### [new 027] A computational system to handle the orthographic layer of tajwid in contemporary Quranic Orthography
- **分类: cs.CL**

- **简介: 该论文属于计算语言学任务，旨在解决《古兰经》正字法中音标规则层（tajwid）的自动化处理问题。研究者开发了Python模块，通过编码开罗版《古兰经》数字文本，实现添加/移除音标层功能，构建跨手稿对齐框架，以分析语音标记系统及比较不同版本经文。**

- **链接: [http://arxiv.org/pdf/2505.11379v1](http://arxiv.org/pdf/2505.11379v1)**

> **作者:** Alicia González Martínez
>
> **摘要:** Contemporary Quranic Orthography (CQO) relies on a precise system of phonetic notation that can be traced back to the early stages of Islam, when the Quran was mainly oral in nature and the first written renderings of it served as memory aids for this oral tradition. The early systems of diacritical marks created on top of the Quranic Consonantal Text (QCT) motivated the creation and further development of a fine-grained system of phonetic notation that represented tajwid-the rules of recitation. We explored the systematicity of the rules of tajwid, as they are encountered in the Cairo Quran, using a fully and accurately encoded digital edition of the Quranic text. For this purpose, we developed a python module that can remove or add the orthographic layer of tajwid from a Quranic text in CQO. The interesting characteristic of these two sets of rules is that they address the complete Quranic text of the Cairo Quran, so they can be used as precise witnesses to study its phonetic and prosodic processes. From a computational point of view, the text of the Cairo Quran can be used as a linchpin to align and compare Quranic manuscripts, due to its richness and completeness. This will let us create a very powerful framework to work with the Arabic script, not just within an isolated text, but automatically exploring a specific textual phenomenon in other connected manuscripts. Having all the texts mapped among each other can serve as a powerful tool to study the nature of the notation systems of diacritics added to the consonantal skeleton.
>
---
#### [new 028] Probing Subphonemes in Morphology Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理的形态学建模任务，探究Transformer在跨语言形态变化中泛化能力受限的原因。通过语言无关的探测方法，分析七种语言中音系特征（如土耳其语尾音清化、元音和谐）的编码机制，发现局部特征由音素嵌入捕获，长距离依赖则依赖编码器表征，为优化形态模型的次音素特征学习提供依据。**

- **链接: [http://arxiv.org/pdf/2505.11297v1](http://arxiv.org/pdf/2505.11297v1)**

> **作者:** Gal Astrach; Yuval Pinter
>
> **摘要:** Transformers have achieved state-of-the-art performance in morphological inflection tasks, yet their ability to generalize across languages and morphological rules remains limited. One possible explanation for this behavior can be the degree to which these models are able to capture implicit phenomena at the phonological and subphonemic levels. We introduce a language-agnostic probing method to investigate phonological feature encoding in transformers trained directly on phonemes, and perform it across seven morphologically diverse languages. We show that phonological features which are local, such as final-obstruent devoicing in Turkish, are captured well in phoneme embeddings, whereas long-distance dependencies like vowel harmony are better represented in the transformer's encoder. Finally, we discuss how these findings inform empirical strategies for training morphological models, particularly regarding the role of subphonemic feature acquisition.
>
---
#### [new 029] Artificial Intelligence Bias on English Language Learners in Automatic Scoring
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究自动评分系统对英语学习者（ELLs）的潜在偏见，属于公平性分析任务。通过微调BERT模型并对比不同训练数据（平衡/不平衡），发现当ELL样本充足（≥1000）时无显著偏差，但小样本（200）可能导致评分差异。**

- **链接: [http://arxiv.org/pdf/2505.10643v1](http://arxiv.org/pdf/2505.10643v1)**

> **作者:** Shuchen Guo; Yun Wang; Jichao Yu; Xuansheng Wu; Bilgehan Ayik; Field M. Watts; Ehsan Latif; Ninghao Liu; Lei Liu; Xiaoming Zhai
>
> **摘要:** This study investigated potential scoring biases and disparities toward English Language Learners (ELLs) when using automatic scoring systems for middle school students' written responses to science assessments. We specifically focus on examining how unbalanced training data with ELLs contributes to scoring bias and disparities. We fine-tuned BERT with four datasets: responses from (1) ELLs, (2) non-ELLs, (3) a mixed dataset reflecting the real-world proportion of ELLs and non-ELLs (unbalanced), and (4) a balanced mixed dataset with equal representation of both groups. The study analyzed 21 assessment items: 10 items with about 30,000 ELL responses, five items with about 1,000 ELL responses, and six items with about 200 ELL responses. Scoring accuracy (Acc) was calculated and compared to identify bias using Friedman tests. We measured the Mean Score Gaps (MSGs) between ELLs and non-ELLs and then calculated the differences in MSGs generated through both the human and AI models to identify the scoring disparities. We found that no AI bias and distorted disparities between ELLs and non-ELLs were found when the training dataset was large enough (ELL = 30,000 and ELL = 1,000), but concerns could exist if the sample size is limited (ELL = 200).
>
---
#### [new 030] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出GODBench基准，用于评估多模态大语言模型在视频评论艺术中的创意表达能力，解决现有基准模态单一、类别不足的问题，并设计Ripple of Thought推理框架提升模型创造力。通过实验验证现有模型在幽默讽刺生成上的不足及新方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.11436v1](http://arxiv.org/pdf/2505.11436v1)**

> **作者:** Chenkai Zhang; Yiming Lei; Zeming Liu; Haitao Leng; Shaoguo Liu; Tingting Gao; Qingjie Liu; Yunhong Wang
>
> **备注:** 69 pages, 66 figures, accepted by ACL 2025
>
> **摘要:** Video Comment Art enhances user engagement by providing creative content that conveys humor, satire, or emotional resonance, requiring a nuanced and comprehensive grasp of cultural and contextual subtleties. Although Multimodal Large Language Models (MLLMs) and Chain-of-Thought (CoT) have demonstrated strong reasoning abilities in STEM tasks (e.g. mathematics and coding), they still struggle to generate creative expressions such as resonant jokes and insightful satire. Moreover, existing benchmarks are constrained by their limited modalities and insufficient categories, hindering the exploration of comprehensive creativity in video-based Comment Art creation. To address these limitations, we introduce GODBench, a novel benchmark that integrates video and text modalities to systematically evaluate MLLMs' abilities to compose Comment Art. Furthermore, inspired by the propagation patterns of waves in physics, we propose Ripple of Thought (RoT), a multi-step reasoning framework designed to enhance the creativity of MLLMs. Extensive experiments reveal that existing MLLMs and CoT methods still face significant challenges in understanding and generating creative video comments. In contrast, RoT provides an effective approach to improve creative composing, highlighting its potential to drive meaningful advancements in MLLM-based creativity. GODBench is publicly available at https://github.com/stan-lei/GODBench-ACL2025.
>
---
#### [new 031] CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs
- **分类: cs.CL**

- **简介: 该论文属于医疗大语言模型安全评估任务，针对现有基准缺乏临床针对性、攻击覆盖不足的问题，构建CARES评估框架，包含多维医疗提示、安全评分体系和对抗攻击检测方法，分析模型漏洞并提出轻量级防御策略，提升医疗LLMs在对抗性场景下的安全性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.11413v1](http://arxiv.org/pdf/2505.11413v1)**

> **作者:** Sijia Chen; Xiaomin Li; Mengxue Zhang; Eric Hanchen Jiang; Qingcheng Zeng; Chen-Hsiang Yu
>
> **摘要:** Large language models (LLMs) are increasingly deployed in medical contexts, raising critical concerns about safety, alignment, and susceptibility to adversarial manipulation. While prior benchmarks assess model refusal capabilities for harmful prompts, they often lack clinical specificity, graded harmfulness levels, and coverage of jailbreak-style attacks. We introduce CARES (Clinical Adversarial Robustness and Evaluation of Safety), a benchmark for evaluating LLM safety in healthcare. CARES includes over 18,000 prompts spanning eight medical safety principles, four harm levels, and four prompting styles: direct, indirect, obfuscated, and role-play, to simulate both malicious and benign use cases. We propose a three-way response evaluation protocol (Accept, Caution, Refuse) and a fine-grained Safety Score metric to assess model behavior. Our analysis reveals that many state-of-the-art LLMs remain vulnerable to jailbreaks that subtly rephrase harmful prompts, while also over-refusing safe but atypically phrased queries. Finally, we propose a mitigation strategy using a lightweight classifier to detect jailbreak attempts and steer models toward safer behavior via reminder-based conditioning. CARES provides a rigorous framework for testing and improving medical LLM safety under adversarial and ambiguous conditions.
>
---
#### [new 032] StRuCom: A Novel Dataset of Structured Code Comments in Russian
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文属于代码注释生成任务，旨在解决俄语结构化文档注释生成性能差的问题。作者构建了首个大规模俄语数据集StRuCom（含15.3万样本），融合人工编写与合成注释，通过自动验证确保符合多语言标准。实验表明，使用该数据集微调的Qwen2.5-Coder模型在自动评估指标上显著优于基线。**

- **链接: [http://arxiv.org/pdf/2505.11026v1](http://arxiv.org/pdf/2505.11026v1)**

> **作者:** Maria Dziuba; Valentin Malykh
>
> **摘要:** Structured code comments in docstring format are essential for code comprehension and maintenance, but existing machine learning models for their generation perform poorly for Russian compared to English. To bridge this gap, we present StRuCom - the first large-scale dataset (153K examples) specifically designed for Russian code documentation. Unlike machine-translated English datasets that distort terminology (e.g., technical loanwords vs. literal translations) and docstring structures, StRuCom combines human-written comments from Russian GitHub repositories with synthetically generated ones, ensuring compliance with Python, Java, JavaScript, C#, and Go standards through automated validation. Fine-tuning Qwen2.5-Coder models (0.5B-7B) on StRuCom shows statistically significant improvements of chrf++ and BERTScore over baseline models.
>
---
#### [new 033] Modeling cognitive processes of natural reading with transformer-based Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于认知建模任务，旨在探究transformer语言模型对人类阅读中眼动行为（凝视时长）的解释能力。研究比较了GPT2、LLaMA系列与早期模型在预测西班牙语读者眼动数据时的表现，发现transformer模型虽优于传统方法，但仍无法完全匹配人类语言预测机制，揭示了AI与人类认知处理的差异。**

- **链接: [http://arxiv.org/pdf/2505.11485v1](http://arxiv.org/pdf/2505.11485v1)**

> **作者:** Bruno Bianchi; Fermín Travi; Juan E. Kamienkowski
>
> **摘要:** Recent advances in Natural Language Processing (NLP) have led to the development of highly sophisticated language models for text generation. In parallel, neuroscience has increasingly employed these models to explore cognitive processes involved in language comprehension. Previous research has shown that models such as N-grams and LSTM networks can partially account for predictability effects in explaining eye movement behaviors, specifically Gaze Duration, during reading. In this study, we extend these findings by evaluating transformer-based models (GPT2, LLaMA-7B, and LLaMA2-7B) to further investigate this relationship. Our results indicate that these architectures outperform earlier models in explaining the variance in Gaze Durations recorded from Rioplantense Spanish readers. However, similar to previous studies, these models still fail to account for the entirety of the variance captured by human predictability. These findings suggest that, despite their advancements, state-of-the-art language models continue to predict language in ways that differ from human readers.
>
---
#### [new 034] Towards Cultural Bridge by Bahnaric-Vietnamese Translation Using Transfer Learning of Sequence-To-Sequence Pre-training Language Model
- **分类: cs.CL**

- **简介: 该论文研究巴拿语-越南语机器翻译任务，旨在解决因巴拿语资源匮乏（词汇、双语语料等）导致的翻译难题。通过迁移学习构建序列到序列预训练模型，复用越南语模型特征，结合有限双语数据微调，并采用数据增强与启发式方法优化翻译效果，促进语言保护及跨文化沟通。**

- **链接: [http://arxiv.org/pdf/2505.11421v1](http://arxiv.org/pdf/2505.11421v1)**

> **作者:** Phan Tran Minh Dat; Vo Hoang Nhat Khang; Quan Thanh Tho
>
> **摘要:** This work explores the journey towards achieving Bahnaric-Vietnamese translation for the sake of culturally bridging the two ethnic groups in Vietnam. However, translating from Bahnaric to Vietnamese also encounters some difficulties. The most prominent challenge is the lack of available original Bahnaric resources source language, including vocabulary, grammar, dialogue patterns and bilingual corpus, which hinders the data collection process for training. To address this, we leverage a transfer learning approach using sequence-to-sequence pre-training language model. First of all, we leverage a pre-trained Vietnamese language model to capture the characteristics of this language. Especially, to further serve the purpose of machine translation, we aim for a sequence-to-sequence model, not encoder-only like BERT or decoder-only like GPT. Taking advantage of significant similarity between the two languages, we continue training the model with the currently limited bilingual resources of Vietnamese-Bahnaric text to perform the transfer learning from language model to machine translation. Thus, this approach can help to handle the problem of imbalanced resources between two languages, while also optimizing the training and computational processes. Additionally, we also enhanced the datasets using data augmentation to generate additional resources and defined some heuristic methods to help the translation more precise. Our approach has been validated to be highly effective for the Bahnaric-Vietnamese translation model, contributing to the expansion and preservation of languages, and facilitating better mutual understanding between the two ethnic people.
>
---
#### [new 035] AI-enhanced semantic feature norms for 786 concepts
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于认知科学中语义特征规范构建任务，旨在解决传统方法在概念覆盖与质量验证间的矛盾。研究者结合人类数据与大型语言模型生成特征，创建了AI增强的NOVA数据集，验证其质量后证实其预测人类语义相似性的能力优于纯人工数据集，证明合理验证的LLMs能有效扩展认知研究工具。**

- **链接: [http://arxiv.org/pdf/2505.10718v1](http://arxiv.org/pdf/2505.10718v1)**

> **作者:** Siddharth Suresh; Kushin Mukherjee; Tyler Giallanza; Xizheng Yu; Mia Patil; Jonathan D. Cohen; Timothy T. Rogers
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Semantic feature norms have been foundational in the study of human conceptual knowledge, yet traditional methods face trade-offs between concept/feature coverage and verifiability of quality due to the labor-intensive nature of norming studies. Here, we introduce a novel approach that augments a dataset of human-generated feature norms with responses from large language models (LLMs) while verifying the quality of norms against reliable human judgments. We find that our AI-enhanced feature norm dataset, NOVA: Norms Optimized Via AI, shows much higher feature density and overlap among concepts while outperforming a comparable human-only norm dataset and word-embedding models in predicting people's semantic similarity judgments. Taken together, we demonstrate that human conceptual knowledge is richer than captured in previous norm datasets and show that, with proper validation, LLMs can serve as powerful tools for cognitive science research.
>
---
#### [new 036] CAMEO: Collection of Multilingual Emotional Speech Corpora
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 本文提出CAMEO多语言情感语音数据集，用于语音情感识别任务，解决数据分散、复现困难和缺乏统一基准的问题。通过筛选、标准化处理多语种数据，构建公开平台并提供模型性能评估，建立跨语言/情感的标准化评测基准。**

- **链接: [http://arxiv.org/pdf/2505.11051v1](http://arxiv.org/pdf/2505.11051v1)**

> **作者:** Iwona Christop; Maciej Czajka
>
> **备注:** Under review at NeurIPS
>
> **摘要:** This paper presents CAMEO -- a curated collection of multilingual emotional speech datasets designed to facilitate research in emotion recognition and other speech-related tasks. The main objectives were to ensure easy access to the data, to allow reproducibility of the results, and to provide a standardized benchmark for evaluating speech emotion recognition (SER) systems across different emotional states and languages. The paper describes the dataset selection criteria, the curation and normalization process, and provides performance results for several models. The collection, along with metadata, and a leaderboard, is publicly available via the Hugging Face platform.
>
---
#### [new 037] Reasoning with OmniThought: A Large CoT Dataset with Verbosity and Cognitive Difficulty Annotations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的推理模型优化任务，旨在解决现有思维链(CoT)数据集规模小、缺乏多维标注的问题。研究者构建了包含200万标注RV（合理长度）与CD（认知难度）的OmniThought数据集，通过双教师模型生成验证，并基于此训练出具备更强推理能力的模型系列。**

- **链接: [http://arxiv.org/pdf/2505.10937v1](http://arxiv.org/pdf/2505.10937v1)**

> **作者:** Wenrui Cai; Chengyu Wang; Junbing Yan; Jun Huang; Xiangzhong Fang
>
> **摘要:** The emergence of large reasoning models (LRMs) has transformed Natural Language Processing by excelling in complex tasks such as mathematical problem-solving and code generation. These models leverage chain-of-thought (CoT) processes, enabling them to emulate human-like reasoning strategies. However, the advancement of LRMs is hindered by the lack of comprehensive CoT datasets. Current resources often fail to provide extensive reasoning problems with coherent CoT processes distilled from multiple teacher models and do not account for multifaceted properties describing the internal characteristics of CoTs. To address these challenges, we introduce OmniThought, a large-scale dataset featuring 2 million CoT processes generated and validated by two powerful LRMs as teacher models. Each CoT process in OmniThought is annotated with novel Reasoning Verbosity (RV) and Cognitive Difficulty (CD) scores, which describe the appropriateness of CoT verbosity and cognitive difficulty level for models to comprehend these reasoning processes. We further establish a self-reliant pipeline to curate this dataset. Extensive experiments using Qwen2.5 models of various sizes demonstrate the positive impact of our proposed scores on LRM training effectiveness. Based on the proposed OmniThought dataset, we further train and release a series of high-performing LRMs, specifically equipped with stronger reasoning abilities and optimal CoT output length and difficulty level. Our contributions significantly enhance the development and training of LRMs for solving complex tasks.
>
---
#### [new 038] SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于多语言事实核查任务，旨在解决虚假信息检测中多语言和低资源语言被忽视的问题。研究组织了SemEval-2025共享任务，包含单语言和跨语言两个子赛道，要求从社交媒体检索匹配的事实核查声明。通过分析179名参与者的52份提交结果，总结了最优系统及有效方法，为自动化事实核查提供数据集和参考方案。**

- **链接: [http://arxiv.org/pdf/2505.10740v1](http://arxiv.org/pdf/2505.10740v1)**

> **作者:** Qiwei Peng; Robert Moro; Michal Gregor; Ivan Srba; Simon Ostermann; Marian Simko; Juraj Podroužek; Matúš Mesarčík; Jaroslav Kopčan; Anders Søgaard
>
> **摘要:** The rapid spread of online disinformation presents a global challenge, and machine learning has been widely explored as a potential solution. However, multilingual settings and low-resource languages are often neglected in this field. To address this gap, we conducted a shared task on multilingual claim retrieval at SemEval 2025, aimed at identifying fact-checked claims that match newly encountered claims expressed in social media posts across different languages. The task includes two subtracks: (1) a monolingual track, where social posts and claims are in the same language, and (2) a crosslingual track, where social posts and claims might be in different languages. A total of 179 participants registered for the task contributing to 52 test submissions. 23 out of 31 teams have submitted their system papers. In this paper, we report the best-performing systems as well as the most common and the most effective approaches across both subtracks. This shared task, along with its dataset and participating systems, provides valuable insights into multilingual claim retrieval and automated fact-checking, supporting future research in this field.
>
---
#### [new 039] Survey of End-to-End Multi-Speaker Automatic Speech Recognition for Monaural Audio
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文属多说话人语音识别综述，针对单通道音频中数据稀缺、重叠语音下说话人及内容识别难题，系统梳理端到端架构（SIMO/SISO范式）的技术演进，分析算法改进、长语音处理策略，并通过基准评估对比方法性能，总结挑战与未来方向。**

- **链接: [http://arxiv.org/pdf/2505.10975v1](http://arxiv.org/pdf/2505.10975v1)**

> **作者:** Xinlu He; Jacob Whitehill
>
> **备注:** 13 pages. Submitted to IEEE/ACM Transaction on Audio Speech and Language Processing (TASLP)
>
> **摘要:** Monaural multi-speaker automatic speech recognition (ASR) remains challenging due to data scarcity and the intrinsic difficulty of recognizing and attributing words to individual speakers, particularly in overlapping speech. Recent advances have driven the shift from cascade systems to end-to-end (E2E) architectures, which reduce error propagation and better exploit the synergy between speech content and speaker identity. Despite rapid progress in E2E multi-speaker ASR, the field lacks a comprehensive review of recent developments. This survey provides a systematic taxonomy of E2E neural approaches for multi-speaker ASR, highlighting recent advances and comparative analysis. Specifically, we analyze: (1) architectural paradigms (SIMO vs.~SISO) for pre-segmented audio, analyzing their distinct characteristics and trade-offs; (2) recent architectural and algorithmic improvements based on these two paradigms; (3) extensions to long-form speech, including segmentation strategy and speaker-consistent hypothesis stitching. Further, we (4) evaluate and compare methods across standard benchmarks. We conclude with a discussion of open challenges and future research directions towards building robust and scalable multi-speaker ASR.
>
---
#### [new 040] SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLMs）长文本处理能力不足的问题，提出短长偏好优化框架SoLoPO。通过分解长上下文优化为短上下文训练（提升知识利用）和长短奖励对齐（保持任务一致性），解决数据质量差、训练效率低的问题，提升模型在长文本场景的泛化能力和计算效率。属于自然语言处理的长上下文优化任务。**

- **链接: [http://arxiv.org/pdf/2505.11166v1](http://arxiv.org/pdf/2505.11166v1)**

> **作者:** Huashan Sun; Shengyi Liao; Yansen Han; Yu Bai; Yang Gao; Cheng Fu; Weizhou Shen; Fanqi Wan; Ming Yan; Ji Zhang; Fei Huang
>
> **摘要:** Despite advances in pretraining with extended context lengths, large language models (LLMs) still face challenges in effectively utilizing real-world long-context information, primarily due to insufficient long-context alignment caused by data quality issues, training inefficiencies, and the lack of well-designed optimization objectives. To address these limitations, we propose a framework named $\textbf{S}$h$\textbf{o}$rt-to-$\textbf{Lo}$ng $\textbf{P}$reference $\textbf{O}$ptimization ($\textbf{SoLoPO}$), decoupling long-context preference optimization (PO) into two components: short-context PO and short-to-long reward alignment (SoLo-RA), supported by both theoretical and empirical evidence. Specifically, short-context PO leverages preference pairs sampled from short contexts to enhance the model's contextual knowledge utilization ability. Meanwhile, SoLo-RA explicitly encourages reward score consistency utilization for the responses when conditioned on both short and long contexts that contain identical task-relevant information. This facilitates transferring the model's ability to handle short contexts into long-context scenarios. SoLoPO is compatible with mainstream preference optimization algorithms, while substantially improving the efficiency of data construction and training processes. Experimental results show that SoLoPO enhances all these algorithms with respect to stronger length and domain generalization abilities across various long-context benchmarks, while achieving notable improvements in both computational and memory efficiency.
>
---
#### [new 041] Temporal fine-tuning for early risk detection
- **分类: cs.CL**

- **简介: 该论文针对早期风险检测（ERD）任务，旨在通过优化模型精度与减少检测延迟，快速识别用户健康风险（如抑郁、饮食障碍）。传统方法依赖多目标优化，而本文提出时间微调策略，将时间因素融入Transformer模型训练，通过分析用户完整发帖历史并采用时间相关指标，统一优化检测速度与准确性，在西班牙语任务中取得竞争性结果。**

- **链接: [http://arxiv.org/pdf/2505.11280v1](http://arxiv.org/pdf/2505.11280v1)**

> **作者:** Horacio Thompson; Esaú Villatoro-Tello; Manuel Montes-y-Gómez; Marcelo Errecalde
>
> **备注:** In: Proceedings of the 53rd JAIIO / 50th CLEI - ASAID, 2024, p. 137. ISSN: 2451-7496
>
> **摘要:** Early Risk Detection (ERD) on the Web aims to identify promptly users facing social and health issues. Users are analyzed post-by-post, and it is necessary to guarantee correct and quick answers, which is particularly challenging in critical scenarios. ERD involves optimizing classification precision and minimizing detection delay. Standard classification metrics may not suffice, resorting to specific metrics such as ERDE(theta) that explicitly consider precision and delay. The current research focuses on applying a multi-objective approach, prioritizing classification performance and establishing a separate criterion for decision time. In this work, we propose a completely different strategy, temporal fine-tuning, which allows tuning transformer-based models by explicitly incorporating time within the learning process. Our method allows us to analyze complete user post histories, tune models considering different contexts, and evaluate training performance using temporal metrics. We evaluated our proposal in the depression and eating disorders tasks for the Spanish language, achieving competitive results compared to the best models of MentalRiskES 2023. We found that temporal fine-tuning optimized decisions considering context and time progress. In this way, by properly taking advantage of the power of transformers, it is possible to address ERD by combining precision and speed as a single objective.
>
---
#### [new 042] Semantic Caching of Contextual Summaries for Efficient Question-Answering with Language Models
- **分类: cs.CL; cs.AI; cs.IR; cs.LG; I.2.7**

- **简介: 该论文针对大语言模型（LLM）在实时问答中处理长上下文时的高计算与带宽开销，提出语义缓存方法存储复用上下文摘要，减少冗余计算。属于高效自然语言处理任务，通过缓存中间结果实现资源优化，在多个数据集验证中降低50-60%算力且保持准确性，适用于实时AI助手场景。**

- **链接: [http://arxiv.org/pdf/2505.11271v1](http://arxiv.org/pdf/2505.11271v1)**

> **作者:** Camille Couturier; Spyros Mastorakis; Haiying Shen; Saravan Rajmohan; Victor Rühle
>
> **备注:** Preprint. Paper accepted at ICCCN 2025, the final version will appear in the proceedings
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed across edge and cloud platforms for real-time question-answering and retrieval-augmented generation. However, processing lengthy contexts in distributed systems incurs high computational overhead, memory usage, and network bandwidth. This paper introduces a novel semantic caching approach for storing and reusing intermediate contextual summaries, enabling efficient information reuse across similar queries in LLM-based QA workflows. Our method reduces redundant computations by up to 50-60% while maintaining answer accuracy comparable to full document processing, as demonstrated on NaturalQuestions, TriviaQA, and a synthetic ArXiv dataset. This approach balances computational cost and response quality, critical for real-time AI assistants.
>
---
#### [new 043] A Systematic Analysis of Base Model Choice for Reward Modeling
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究奖励建模中基模型选择对性能的影响（任务）。针对基模型选型常被忽视且选择困难的问题，通过系统分析发现优化基模型可提升14%性能，结合多基准测试提升18%选择效果，并验证后训练步骤和数据分布对预测误差的影响。**

- **链接: [http://arxiv.org/pdf/2505.10775v1](http://arxiv.org/pdf/2505.10775v1)**

> **作者:** Kian Ahrabian; Pegah Jandaghi; Negar Mokhberian; Sai Praneeth Karimireddy; Jay Pujara
>
> **备注:** 19 pages, 13 figures, 5 tables
>
> **摘要:** Reinforcement learning from human feedback (RLHF) and, at its core, reward modeling have become a crucial part of training powerful large language models (LLMs). One commonly overlooked factor in training high-quality reward models (RMs) is the effect of the base model, which is becoming more challenging to choose given the rapidly growing pool of LLMs. In this work, we present a systematic analysis of the effect of base model selection on reward modeling performance. Our results show that the performance can be improved by up to 14% compared to the most common (i.e., default) choice. Moreover, we showcase the strong statistical relation between some existing benchmarks and downstream performances. We also demonstrate that the results from a small set of benchmarks could be combined to boost the model selection ($+$18% on average in the top 5-10). Lastly, we illustrate the impact of different post-training steps on the final performance and explore using estimated data distributions to reduce performance prediction error.
>
---
#### [new 044] Accurate KV Cache Quantization with Outlier Tokens Tracing
- **分类: cs.CL**

- **简介: 该论文研究大语言模型部署优化的KV Cache量化任务，解决异常词元破坏量化精度问题。通过追踪解码过程的离群词元并单独处理，改进了2位量化精度，在保持通道/词元量化策略基础上减少内存占用6.4倍，提升推理吞吐量2.3倍。**

- **链接: [http://arxiv.org/pdf/2505.10938v1](http://arxiv.org/pdf/2505.10938v1)**

> **作者:** Yi Su; Yuechi Zhou; Quantong Qiu; Juntao Li; Qingrong Xia; Ping Li; Xinyu Duan; Zhefeng Wang; Min Zhang
>
> **备注:** ACL2025 Main
>
> **摘要:** The impressive capabilities of Large Language Models (LLMs) come at the cost of substantial computational resources during deployment. While KV Cache can significantly reduce recomputation during inference, it also introduces additional memory overhead. KV Cache quantization presents a promising solution, striking a good balance between memory usage and accuracy. Previous research has shown that the Keys are distributed by channel, while the Values are distributed by token. Consequently, the common practice is to apply channel-wise quantization to the Keys and token-wise quantization to the Values. However, our further investigation reveals that a small subset of unusual tokens exhibit unique characteristics that deviate from this pattern, which can substantially impact quantization accuracy. To address this, we develop a simple yet effective method to identify these tokens accurately during the decoding process and exclude them from quantization as outlier tokens, significantly improving overall accuracy. Extensive experiments show that our method achieves significant accuracy improvements under 2-bit quantization and can deliver a 6.4 times reduction in memory usage and a 2.3 times increase in throughput.
>
---
#### [new 045] Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究检索增强推理任务，解决大语言模型（LLM）检索信息低质导致推理不准的问题。提出AutoRefine框架，通过强化学习引入“搜索-精炼”机制，迭代优化外部知识筛选与整合，结合检索质量与答案正确性双奖励机制。实验证明其在复杂多跳问答中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.11277v1](http://arxiv.org/pdf/2505.11277v1)**

> **作者:** Yaorui Shi; Shihan Li; Chang Wu; Zhiyuan Liu; Junfeng Fang; Hengxing Cai; An Zhang; Xiang Wang
>
> **摘要:** Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
>
---
#### [new 046] GenKnowSub: Improving Modularity and Reusability of LLMs through General Knowledge Subtraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的模型模块化研究，旨在解决大语言模型中通用知识与任务适配的耦合问题。提出GenKnowSub框架，通过分解通用领域和任务专用的LoRA模块，利用知识减法和动态路由算法，提升跨任务、跨语言的零样本泛化能力，并在多语言基准测试中验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.10939v1](http://arxiv.org/pdf/2505.10939v1)**

> **作者:** Mohammadtaha Bagherifard; Sahar Rajabi; Ali Edalat; Yadollah Yaghoobzadeh
>
> **备注:** Accepted to ACL 2025 (main conference, short paper), 10 pages
>
> **摘要:** Large language models often struggle with zero-shot generalization, and several modular approaches have been proposed to address this challenge. Yet, we hypothesize that a key limitation remains: the entanglement of general knowledge and task-specific adaptations. To overcome this, we propose a modular framework that disentangles these components by constructing a library of task-specific LoRA modules alongside a general-domain LoRA. By subtracting this general knowledge component from each task-specific module, we obtain residual modules that focus more exclusively on task-relevant information, a method we call general knowledge subtraction (GenKnowSub). Leveraging the refined task-specific modules and the Arrow routing algorithm \citep{ostapenko2024towards}, we dynamically select and combine modules for new inputs without additional training. Our studies on the Phi-3 model and standard Arrow as baselines reveal that using general knowledge LoRAs derived from diverse languages, including English, French, and German, yields consistent performance gains in both monolingual and cross-lingual settings across a wide set of benchmarks. Further experiments on Phi-2 demonstrate how GenKnowSub generalizes to weaker LLMs. The complete code and data are available at https://github.com/saharsamr/Modular-LLM.
>
---
#### [new 047] Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization
- **分类: cs.CL**

- **简介: 该论文研究大语言模型自动提示优化任务，解决传统方法依赖随机评估数据导致效果差的问题。提出IPOMP方法：先通过语义聚类和边界分析选取代表性样本，再结合实时性能迭代替换冗余数据，提升优化效果和稳定性。实验显示其效果提升1.6%-5.3%，计算开销低于1%。**

- **链接: [http://arxiv.org/pdf/2505.10736v1](http://arxiv.org/pdf/2505.10736v1)**

> **作者:** Ximing Dong; Shaowei Wang; Dayi Lin; Ahmed E. Hassan
>
> **摘要:** Optimizing Large Language Model (LLM) performance requires well-crafted prompts, but manual prompt engineering is labor-intensive and often ineffective. Automated prompt optimization techniques address this challenge but the majority of them rely on randomly selected evaluation subsets, which fail to represent the full dataset, leading to unreliable evaluations and suboptimal prompts. Existing coreset selection methods, designed for LLM benchmarking, are unsuitable for prompt optimization due to challenges in clustering similar samples, high data collection costs, and the unavailability of performance data for new or private datasets. To overcome these issues, we propose IPOMP, an Iterative evaluation data selection for effective Prompt Optimization using real-time Model Performance. IPOMP is a two-stage approach that selects representative and diverse samples using semantic clustering and boundary analysis, followed by iterative refinement with real-time model performance data to replace redundant samples. Evaluations on the BIG-bench dataset show that IPOMP improves effectiveness by 1.6% to 5.3% and stability by at least 57% compared with SOTA baselines, with minimal computational overhead below 1%. Furthermore, the results demonstrate that our real-time performance-guided refinement approach can be universally applied to enhance existing coreset selection methods.
>
---
#### [new 048] Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对检索增强生成（RAG）中因检索内容不相关导致语言模型幻觉的问题，提出Finetune-RAG方法。通过构建模拟现实不完美检索的训练数据集微调模型，提升事实准确性21.2%，并设计Bench-RAG评估框架测试模型抗干扰能力，属于生成模型鲁棒性优化任务。**

- **链接: [http://arxiv.org/pdf/2505.10792v1](http://arxiv.org/pdf/2505.10792v1)**

> **作者:** Zhan Peng Lee; Andre Lin; Calvin Tan
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a powerful framework to improve factuality in large language models (LLMs) by grounding their outputs in retrieved documents. However, ensuring perfect retrieval of relevant information remains challenging, and when irrelevant content is passed downstream to an LLM, it can lead to hallucinations. In this work, we propose Finetune-RAG, a simple and effective fine-tuning approach that features the first-of-its-kind RAG training dataset constructed to mimic real-world imperfections. Experimental results show that Finetune-RAG improves factual accuracy by 21.2% over the base model. We also propose a Bench-RAG, an LLM-as-a-judge evaluation pipeline that stress tests models under realistic imperfect retrieval scenarios. Our codebase and dataset are fully open sourced for community use.
>
---
#### [new 049] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-lingual Transfer
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究跨语言迁移任务，解决传统方法因词汇替换导致目标语言表达能力受限的问题。提出SALT方法，通过回收目标语言预训练模型的嵌入，基于词汇重叠相似性建立回归线处理非重叠词嵌入，实验证明其在多语言迁移中收敛更快、效果更优。**

- **链接: [http://arxiv.org/pdf/2505.10945v1](http://arxiv.org/pdf/2505.10945v1)**

> **作者:** Seungyoon Lee; Seongtae Hong; Hyeonseok Moon; Heuiseok Lim
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) increasingly incorporate multilingual capabilities, fueling the demand to transfer them into target language-specific models. However, most approaches, which blend the source model's embedding by replacing the source vocabulary with the target language-specific vocabulary, may constrain expressive capacity in the target language since the source model is predominantly trained on English data. In this paper, we propose Semantic Aware Linear Transfer (SALT), a novel cross-lingual transfer technique that recycles embeddings from target language Pre-trained Language Models (PLMs) to transmit the deep representational strengths of PLM-derived embedding to LLMs. SALT derives unique regression lines based on the similarity in the overlap of the source and target vocabularies, to handle each non-overlapping token's embedding space. Our extensive experiments show that SALT significantly outperforms other transfer methods and achieves lower loss with accelerating faster convergence during language adaptation. Notably, SALT obtains remarkable performance in cross-lingual understanding setups compared to other methods. Furthermore, we highlight the scalable use of PLMs to enhance the functionality of contemporary LLMs by conducting experiments with varying architectures.
>
---
#### [new 050] Towards Better Evaluation for Generated Patent Claims
- **分类: cs.CL**

- **简介: 该论文属于专利权利要求生成评估任务，旨在解决自动评估指标与专家评估不一致的问题。作者构建了首个专家标注的Patent-CE基准（含五项关键标准），并提出多维度评估方法PatClaimEval，实验证明其与人工评估相关性最优，为自动化专利生成系统提供了更准确的评估框架。**

- **链接: [http://arxiv.org/pdf/2505.11095v1](http://arxiv.org/pdf/2505.11095v1)**

> **作者:** Lekang Jiang; Pascal A Scherz; Stephan Goetz
>
> **备注:** Accepted to ACL 2025. 14 pages, 8 tables
>
> **摘要:** Patent claims define the scope of protection and establish the legal boundaries of an invention. Drafting these claims is a complex and time-consuming process that usually requires the expertise of skilled patent attorneys, which can form a large access barrier for many small enterprises. To solve these challenges, researchers have investigated the use of large language models (LLMs) for automating patent claim generation. However, existing studies highlight inconsistencies between automated evaluation metrics and human expert assessments. To bridge this gap, we introduce Patent-CE, the first comprehensive benchmark for evaluating patent claims. Patent-CE includes comparative claim evaluations annotated by patent experts, focusing on five key criteria: feature completeness, conceptual clarity, terminology consistency, logical linkage, and overall quality. Additionally, we propose PatClaimEval, a novel multi-dimensional evaluation method specifically designed for patent claims. Our experiments demonstrate that PatClaimEval achieves the highest correlation with human expert evaluations across all assessment criteria among all tested metrics. This research provides the groundwork for more accurate evaluations of automated patent claim generation systems.
>
---
#### [new 051] BLEUBERI: BLEU is a surprisingly effective reward for instruction following
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究语言模型对齐任务，旨在解决传统奖励模型训练成本高的问题。通过发现BLEU指标与人类偏好高度一致，提出BLEUBERI方法：筛选困难指令后，用BLEU作为奖励函数进行策略优化。实验证明其效果与复杂奖励模型相当，且生成内容更事实准确。**

- **链接: [http://arxiv.org/pdf/2505.11080v1](http://arxiv.org/pdf/2505.11080v1)**

> **作者:** Yapei Chang; Yekyung Kim; Michael Krumdick; Amir Zadeh; Chuan Li; Chris Tanner; Mohit Iyyer
>
> **备注:** 28 pages, 11 figures, 15 tables
>
> **摘要:** Reward models are central to aligning LLMs with human preferences, but they are costly to train, requiring large-scale human-labeled preference data and powerful pretrained LLM backbones. Meanwhile, the increasing availability of high-quality synthetic instruction-following datasets raises the question: can simpler, reference-based metrics serve as viable alternatives to reward models during RL-based alignment? In this paper, we show first that BLEU, a basic string-matching metric, surprisingly matches strong reward models in agreement with human preferences on general instruction-following datasets. Based on this insight, we develop BLEUBERI, a method that first identifies challenging instructions and then applies Group Relative Policy Optimization (GRPO) using BLEU directly as the reward function. We demonstrate that BLEUBERI-trained models are competitive with models trained via reward model-guided RL across four challenging instruction-following benchmarks and three different base language models. A human evaluation further supports that the quality of BLEUBERI model outputs is on par with those from reward model-aligned models. Moreover, BLEUBERI models generate outputs that are more factually grounded than competing methods. Overall, we show that given access to high-quality reference outputs (easily obtained via existing instruction-following datasets or synthetic data generation), string matching-based metrics are cheap yet effective proxies for reward models during alignment. We release our code and data at https://github.com/lilakk/BLEUBERI.
>
---
#### [new 052] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床自然语言处理任务，旨在解决大型模型部署成本高、临床数据敏感及小型模型（SLMs）领域适应难的问题。提出模块化框架：通过预指令调整、模型合并和任务对齐优化3.8B参数的SLMs（MediPhi），扩展CLUE+基准并生成合成数据集MediFlow，使模型在医疗实体识别等任务中性能超越GPT-4达14%。**

- **链接: [http://arxiv.org/pdf/2505.10717v1](http://arxiv.org/pdf/2505.10717v1)**

> **作者:** Jean-Philippe Corbeil; Amin Dada; Jean-Michel Attendu; Asma Ben Abacha; Alessandro Sordoni; Lucas Caccia; François Beaulieu; Thomas Lin; Jens Kleesiek; Paul Vozila
>
> **摘要:** High computation costs and latency of large language models such as GPT-4 have limited their deployment in clinical settings. Small language models (SLMs) offer a cost-effective alternative, but their limited capacity requires biomedical domain adaptation, which remains challenging. An additional bottleneck is the unavailability and high sensitivity of clinical data. To address these challenges, we propose a novel framework for adapting SLMs into high-performing clinical models. We introduce the MediPhi collection of 3.8B-parameter SLMs developed with our novel framework: pre-instruction tuning of experts on relevant medical and clinical corpora (PMC, Medical Guideline, MedWiki, etc.), model merging, and clinical-tasks alignment. To cover most clinical tasks, we extended the CLUE benchmark to CLUE+, doubling its size. Our expert models deliver relative improvements on this benchmark over the base model without any task-specific fine-tuning: 64.3% on medical entities, 49.5% on radiology reports, and 44% on ICD-10 coding (outperforming GPT-4-0125 by 14%). We unify the expert models into MediPhi via model merging, preserving gains across benchmarks. Furthermore, we built the MediFlow collection, a synthetic dataset of 2.5 million high-quality instructions on 14 medical NLP tasks, 98 fine-grained document types, and JSON format support. Alignment of MediPhi using supervised fine-tuning and direct preference optimization achieves further gains of 18.9% on average.
>
---
#### [new 053] HAPO: Training Language Models to Reason Concisely via History-Aware Policy Optimization
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型高效推理优化任务，解决大模型生成冗长、成本高的问题。提出HAPO方法，通过历史感知策略记录问题最短正确解，设计奖励机制联合优化正确性和简洁性，训练模型逐步生成更短答案。实验表明在数学基准上实现33-59%长度缩减，精度仅降2-5%。**

- **链接: [http://arxiv.org/pdf/2505.11225v1](http://arxiv.org/pdf/2505.11225v1)**

> **作者:** Chengyu Huang; Zhengxin Zhang; Claire Cardie
>
> **摘要:** While scaling the length of responses at test-time has been shown to markedly improve the reasoning abilities and performance of large language models (LLMs), it often results in verbose outputs and increases inference cost. Prior approaches for efficient test-time scaling, typically using universal budget constraints or query-level length optimization, do not leverage historical information from previous encounters with the same problem during training. We hypothesize that this limits their ability to progressively make solutions more concise over time. To address this, we present History-Aware Policy Optimization (HAPO), which keeps track of a history state (e.g., the minimum length over previously generated correct responses) for each problem. HAPO employs a novel length reward function based on this history state to incentivize the discovery of correct solutions that are more concise than those previously found. Crucially, this reward structure avoids overly penalizing shorter incorrect responses with the goal of facilitating exploration towards more efficient solutions. By combining this length reward with a correctness reward, HAPO jointly optimizes for correctness and efficiency. We use HAPO to train DeepSeek-R1-Distill-Qwen-1.5B, DeepScaleR-1.5B-Preview, and Qwen-2.5-1.5B-Instruct, and evaluate HAPO on several math benchmarks that span various difficulty levels. Experiment results demonstrate that HAPO effectively induces LLMs' concise reasoning abilities, producing length reductions of 33-59% with accuracy drops of only 2-5%.
>
---
#### [new 054] Relation Extraction Across Entire Books to Reconstruct Community Networks: The AffilKG Datasets
- **分类: cs.CL**

- **简介: 该论文属于知识图谱构建任务，旨在解决现有标注数据集无法有效评估知识图谱下游分析准确性的问题。作者提出了AffilKG数据集，包含6个基于完整书籍构建的标注知识图谱，涵盖人物-组织隶属关系及扩展关系，支持分析提取错误对图分析的影响并验证知识图谱提取方法，推动社会科学研究应用。**

- **链接: [http://arxiv.org/pdf/2505.10798v1](http://arxiv.org/pdf/2505.10798v1)**

> **作者:** Erica Cai; Sean McQuade; Kevin Young; Brendan O'Connor
>
> **摘要:** When knowledge graphs (KGs) are automatically extracted from text, are they accurate enough for downstream analysis? Unfortunately, current annotated datasets can not be used to evaluate this question, since their KGs are highly disconnected, too small, or overly complex. To address this gap, we introduce AffilKG (https://doi.org/10.5281/zenodo.15427977), which is a collection of six datasets that are the first to pair complete book scans with large, labeled knowledge graphs. Each dataset features affiliation graphs, which are simple KGs that capture Member relationships between Person and Organization entities -- useful in studies of migration, community interactions, and other social phenomena. In addition, three datasets include expanded KGs with a wider variety of relation types. Our preliminary experiments demonstrate significant variability in model performance across datasets, underscoring AffilKG's ability to enable two critical advances: (1) benchmarking how extraction errors propagate to graph-level analyses (e.g., community structure), and (2) validating KG extraction methods for real-world social science research.
>
---
#### [new 055] XtraGPT: LLMs for Human-AI Collaboration on Controllable Academic Paper Revision
- **分类: cs.CL**

- **简介: 该论文研究学术论文修订中的人机协作任务，解决现有大模型在科学写作深层修订（如概念连贯性）及迭代支持不足的问题。通过构建包含7040篇论文和14万修订标注的数据集，开发了开源模型XtraGPT，提供上下文感知的写作辅助，实验验证其效果接近商用系统。**

- **链接: [http://arxiv.org/pdf/2505.11336v1](http://arxiv.org/pdf/2505.11336v1)**

> **作者:** Nuo Chen; Andre Lin HuiKai; Jiaying Wu; Junyi Hou; Zining Zhang; Qian Wang; Xidong Wang; Bingsheng He
>
> **备注:** preprint
>
> **摘要:** Despite the growing adoption of large language models (LLMs) in academic workflows, their capabilities remain limited when it comes to supporting high-quality scientific writing. Most existing systems are designed for general-purpose scientific text generation and fail to meet the sophisticated demands of research communication beyond surface-level polishing, such as conceptual coherence across sections. Furthermore, academic writing is inherently iterative and revision-driven, a process not well supported by direct prompting-based paradigms. To address these scenarios, we propose a human-AI collaboration framework for academic paper revision. We first introduce a comprehensive dataset of 7,040 research papers from top-tier venues annotated with over 140,000 instruction-response pairs that reflect realistic, section-level scientific revisions. Building on the dataset, we develop XtraGPT, the first suite of open-source LLMs, designed to provide context-aware, instruction-guided writing assistance, ranging from 1.5B to 14B parameters. Extensive experiments validate that XtraGPT significantly outperforms same-scale baselines and approaches the quality of proprietary systems. Both automated preference assessments and human evaluations confirm the effectiveness of our models in improving scientific drafts.
>
---
#### [new 056] Enhancing Low-Resource Minority Language Translation with LLMs and Retrieval-Augmented Generation for Cultural Nuances
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对低资源少数民族语言翻译任务，解决文化术语和语法准确性不足的问题。通过结合大语言模型（LLMs）与检索增强生成（RAG），测试多种配置模型，最佳方案（Model 4）显著提升词汇覆盖和流畅度，BLEU达31%。研究验证了动态检索和迭代修正优于静态词典方法，强调需融合领域知识及社区协作以保障文化适配。**

- **链接: [http://arxiv.org/pdf/2505.10829v1](http://arxiv.org/pdf/2505.10829v1)**

> **作者:** Chen-Chi Chang; Chong-Fu Li; Chu-Hsuan Lee; Hung-Shin Lee
>
> **备注:** Accepted to IntelliSys 2025
>
> **摘要:** This study investigates the challenges of translating low-resource languages by integrating Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG). Various model configurations were tested on Hakka translations, with BLEU scores ranging from 12% (dictionary-only) to 31% (RAG with Gemini 2.0). The best-performing model (Model 4) combined retrieval and advanced language modeling, improving lexical coverage, particularly for specialized or culturally nuanced terms, and enhancing grammatical coherence. A two-stage method (Model 3) using dictionary outputs refined by Gemini 2.0 achieved a BLEU score of 26%, highlighting iterative correction's value and the challenges of domain-specific expressions. Static dictionary-based approaches struggled with context-sensitive content, demonstrating the limitations of relying solely on predefined resources. These results emphasize the need for curated resources, domain knowledge, and ethical collaboration with local communities, offering a framework that improves translation accuracy and fluency while supporting cultural preservation.
>
---
#### [new 057] Low-Resource Language Processing: An OCR-Driven Summarization and Translation Pipeline
- **分类: cs.CL; cs.AI; 68T50 (Natural language processing), 68U10 (Image processing)**

- **简介: 该论文属于多语言文档处理任务，旨在解决低资源语言图像文档的信息获取难题。研究构建了一个端到端系统：通过OCR提取文本（英语/印地语/泰米尔语），结合大模型（Gemini）实现跨语言翻译、摘要生成，并集成情感分析、主题分类等模块，最终通过Grado界面提升多语言图像媒体的信息可及性。**

- **链接: [http://arxiv.org/pdf/2505.11177v1](http://arxiv.org/pdf/2505.11177v1)**

> **作者:** Hrishit Madhavi; Jacob Cherian; Yuvraj Khamkar; Dhananjay Bhagat
>
> **备注:** 8 pages, 7 figures, direct arXiv submission
>
> **摘要:** This paper presents an end-to-end suite for multilingual information extraction and processing from image-based documents. The system uses Optical Character Recognition (Tesseract) to extract text in languages such as English, Hindi, and Tamil, and then a pipeline involving large language model APIs (Gemini) for cross-lingual translation, abstractive summarization, and re-translation into a target language. Additional modules add sentiment analysis (TensorFlow), topic classification (Transformers), and date extraction (Regex) for better document comprehension. Made available in an accessible Gradio interface, the current research shows a real-world application of libraries, models, and APIs to close the language gap and enhance access to information in image media across different linguistic environments
>
---
#### [new 058] SelfBudgeter: Adaptive Token Allocation for Efficient LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于高效大语言模型推理优化任务，旨在解决模型处理不同复杂度查询时资源浪费和延迟高的问题。提出SelfBudgeter方法：通过双阶段训练（预估计推理成本+预算引导强化学习）动态分配计算资源，在压缩响应长度（MATH基准达74.47%）的同时保持精度，允许用户预判时间并控制推理预算。**

- **链接: [http://arxiv.org/pdf/2505.11274v1](http://arxiv.org/pdf/2505.11274v1)**

> **作者:** Zheng Li; Qingxiu Dong; Jingyuan Ma; Di Zhang; Zhifang Sui
>
> **摘要:** Recently, large reasoning models demonstrate exceptional performance on various tasks. However, reasoning models inefficiently over-process both trivial and complex queries, leading to resource waste and prolonged user latency. To address this challenge, we propose SelfBudgeter - a self-adaptive controllable reasoning strategy for efficient reasoning. Our approach adopts a dual-phase training paradigm: first, the model learns to pre-estimate the reasoning cost based on the difficulty of the query. Then, we introduce budget-guided GPRO for reinforcement learning, which effectively maintains accuracy while reducing output length. SelfBudgeter allows users to anticipate generation time and make informed decisions about continuing or interrupting the process. Furthermore, our method enables direct manipulation of reasoning length via pre-filling token budget. Experimental results demonstrate that SelfBudgeter can rationally allocate budgets according to problem complexity, achieving up to 74.47% response length compression on the MATH benchmark while maintaining nearly undiminished accuracy.
>
---
#### [new 059] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出首个长上下文视觉语言模型（LCVLM）评测基准MMLongBench，覆盖5类任务、13,331样本及多类型图像，通过标准化输入长度（8K-128K）评估46个模型，揭示现有模型在长上下文任务中的不足，为改进模型提供诊断基础。**

- **链接: [http://arxiv.org/pdf/2505.10610v1](http://arxiv.org/pdf/2505.10610v1)**

> **作者:** Zhaowei Wang; Wenhao Yu; Xiyu Ren; Jipeng Zhang; Yu Zhao; Rohit Saxena; Liang Cheng; Ginny Wong; Simon See; Pasquale Minervini; Yangqiu Song; Mark Steedman
>
> **备注:** Work in progress
>
> **摘要:** The rapid extension of context windows in large vision-language models has given rise to long-context vision-language models (LCVLMs), which are capable of handling hundreds of images with interleaved text tokens in a single forward pass. In this work, we introduce MMLongBench, the first benchmark covering a diverse set of long-context vision-language tasks, to evaluate LCVLMs effectively and thoroughly. MMLongBench is composed of 13,331 examples spanning five different categories of downstream tasks, such as Visual RAG and Many-Shot ICL. It also provides broad coverage of image types, including various natural and synthetic images. To assess the robustness of the models to different input lengths, all examples are delivered at five standardized input lengths (8K-128K tokens) via a cross-modal tokenization scheme that combines vision patches and text tokens. Through a thorough benchmarking of 46 closed-source and open-source LCVLMs, we provide a comprehensive analysis of the current models' vision-language long-context ability. Our results show that: i) performance on a single task is a weak proxy for overall long-context capability; ii) both closed-source and open-source models face challenges in long-context vision-language tasks, indicating substantial room for future improvement; iii) models with stronger reasoning ability tend to exhibit better long-context performance. By offering wide task coverage, various image types, and rigorous length control, MMLongBench provides the missing foundation for diagnosing and advancing the next generation of LCVLMs.
>
---
#### [new 060] Two Minds Better Than One: Collaborative Reward Modeling for LLM Alignment
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型（LLM）对齐任务，针对人类反馈中噪声偏好导致奖励模型泛化差的问题，提出协作奖励建模框架（CRM）。通过双模型并行评估筛选噪声数据，结合课程学习优化训练稳定性，实验表明其在噪声环境下显著提升模型性能。**

- **链接: [http://arxiv.org/pdf/2505.10597v1](http://arxiv.org/pdf/2505.10597v1)**

> **作者:** Jiazheng Zhang; Wenqing Jing; Zizhuo Zhang; Zhiheng Xi; Shihan Dou; Rongxiang Weng; Jiahuan Li; Jingang Wang; MingXu Cai; Shibo Hong; Tao Gui; Qi Zhang
>
> **摘要:** Reward models (RMs) are essential for aligning large language models (LLMs) with human values. However, noisy preferences in human feedback often lead to reward misgeneralization, where RMs overfit to spurious patterns and provide misleading signals during policy optimization. We systematically analyze the training dynamics of preference pairs and identify that noisy examples are harder to fit and introduce instability. Empirical evidence shows that LLMs optimized using reward models trained on full noisy datasets perform worse than those trained on filtered, high-quality preferences. To address this, we propose Collaborative Reward Modeling (CRM), an online framework that enhances robustness by combining peer review and curriculum learning. Two reward models are trained in parallel and assess each other's data selections to filter out potential noise. Curriculum learning structures the preference data from easy to hard, ensuring synchronized training and stable feedback. Extensive experiments demonstrate that CRM improves generalization, with up to 9.94 points of accuracy gain on RewardBench under 40 percent label noise. CRM is also compatible with implicit-reward alignment methods, offering a practical and versatile strategy for robust alignment.
>
---
#### [new 061] Large Language Model Use Impact Locus of Control
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于AI与心理学的交叉研究，探究使用大语言模型协作写作对用户心理控制感的影响。通过462人实验发现，就业状况显著影响AI依赖程度：在职者更依赖AI且内控倾向增强，失业者则产生自我效能感下降。研究结合实证数据揭示了AI工具重塑个人能动性的潜在机制。**

- **链接: [http://arxiv.org/pdf/2505.11406v1](http://arxiv.org/pdf/2505.11406v1)**

> **作者:** Jenny Xiyu Fu; Brennan Antone; Kowe Kadoma; Malte Jung
>
> **摘要:** As AI tools increasingly shape how we write, they may also quietly reshape how we perceive ourselves. This paper explores the psychological impact of co-writing with AI on people's locus of control. Through an empirical study with 462 participants, we found that employment status plays a critical role in shaping users' reliance on AI and their locus of control. Current results demonstrated that employed participants displayed higher reliance on AI and a shift toward internal control, while unemployed users tended to experience a reduction in personal agency. Through quantitative results and qualitative observations, this study opens a broader conversation about AI's role in shaping personal agency and identity.
>
---
#### [new 062] UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable Emotional Text-to-Speech
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于可控情感文本到语音合成任务，旨在解决传统方法依赖离散情感标签导致情感表达不连续、数据分布不均衡的问题。通过提出UDDETTS模型，结合离散标签与基于唤醒-支配-效价的三维连续情感空间，采用半监督训练策略融合多源数据，实现了线性情感控制和高质量情感语音合成。**

- **链接: [http://arxiv.org/pdf/2505.10599v1](http://arxiv.org/pdf/2505.10599v1)**

> **作者:** Jiaxuan Liu; Zhenhua Ling
>
> **备注:** Under review
>
> **摘要:** Recent neural codec language models have made great progress in the field of text-to-speech (TTS), but controllable emotional TTS still faces many challenges. Traditional methods rely on predefined discrete emotion labels to control emotion categories and intensities, which can't capture the complexity and continuity of human emotional perception and expression. The lack of large-scale emotional speech datasets with balanced emotion distributions and fine-grained emotion annotations often causes overfitting in synthesis models and impedes effective emotion control. To address these issues, we propose UDDETTS, a neural codec language model unifying discrete and dimensional emotions for controllable emotional TTS. This model introduces the interpretable Arousal-Dominance-Valence (ADV) space for dimensional emotion description and supports emotion control driven by either discrete emotion labels or nonlinearly quantified ADV values. Furthermore, a semi-supervised training strategy is designed to comprehensively utilize diverse speech datasets with different types of emotion annotations to train the UDDETTS. Experiments show that UDDETTS achieves linear emotion control along the three dimensions of ADV space, and exhibits superior end-to-end emotional speech synthesis capabilities.
>
---
#### [new 063] MatTools: Benchmarking Large Language Models for Materials Science Tools
- **分类: cond-mat.mtrl-sci; cs.CL; cs.DB**

- **简介: 该论文属于评估任务，旨在解决大语言模型（LLMs）在材料科学工具中生成可靠代码的能力评估问题。通过构建MatTools基准（含材料模拟QA库和真实任务集），评估LLM生成物理计算代码的准确性，发现通用模型优于专用模型等关键结论，为提升科学AI系统提供标准化框架。**

- **链接: [http://arxiv.org/pdf/2505.10852v1](http://arxiv.org/pdf/2505.10852v1)**

> **作者:** Siyu Liu; Jiamin Xu; Beilin Ye; Bo Hu; David J. Srolovitz; Tongqi Wen
>
> **备注:** 27 pages, 23 figures
>
> **摘要:** Large language models (LLMs) are increasingly applied to materials science questions, including literature comprehension, property prediction, materials discovery and alloy design. At the same time, a wide range of physics-based computational approaches have been developed in which materials properties can be calculated. Here, we propose a benchmark application to evaluate the proficiency of LLMs to answer materials science questions through the generation and safe execution of codes based on such physics-based computational materials science packages. MatTools is built on two complementary components: a materials simulation tool question-answer (QA) benchmark and a real-world tool-usage benchmark. We designed an automated methodology to efficiently collect real-world materials science tool-use examples. The QA benchmark, derived from the pymatgen (Python Materials Genomics) codebase and documentation, comprises 69,225 QA pairs that assess the ability of an LLM to understand materials science tools. The real-world benchmark contains 49 tasks (138 subtasks) requiring the generation of functional Python code for materials property calculations. Our evaluation of diverse LLMs yields three key insights: (1)Generalists outshine specialists;(2)AI knows AI; and (3)Simpler is better. MatTools provides a standardized framework for assessing and improving LLM capabilities for materials science tool applications, facilitating the development of more effective AI systems for materials science and general scientific research.
>
---
#### [new 064] CompAlign: Improving Compositional Text-to-Image Generation with a Complex Benchmark and Fine-Grained Feedback
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于组合文本到图像生成任务，旨在解决现有模型对多对象、属性和空间关系描述不准确的问题。提出了包含900复杂提示的基准CompAlign，开发了基于多模态大模型的可解释评估框架CompQuest（分解提示并提供细粒度反馈），并通过反馈信号改进扩散模型的对齐能力。实验表明模型在复杂3D空间关系任务中表现差且开源/闭源模型差距大，经对齐优化后生成准确性显著提升。**

- **链接: [http://arxiv.org/pdf/2505.11178v1](http://arxiv.org/pdf/2505.11178v1)**

> **作者:** Yixin Wan; Kai-Wei Chang
>
> **摘要:** State-of-the-art T2I models are capable of generating high-resolution images given textual prompts. However, they still struggle with accurately depicting compositional scenes that specify multiple objects, attributes, and spatial relations. We present CompAlign, a challenging benchmark with an emphasis on assessing the depiction of 3D-spatial relationships, for evaluating and improving models on compositional image generation. CompAlign consists of 900 complex multi-subject image generation prompts that combine numerical and 3D-spatial relationships with varied attribute bindings. Our benchmark is remarkably challenging, incorporating generation tasks with 3+ generation subjects with complex 3D-spatial relationships. Additionally, we propose CompQuest, an interpretable and accurate evaluation framework that decomposes complex prompts into atomic sub-questions, then utilizes a MLLM to provide fine-grained binary feedback on the correctness of each aspect of generation elements in model-generated images. This enables precise quantification of alignment between generated images and compositional prompts. Furthermore, we propose an alignment framework that uses CompQuest's feedback as preference signals to improve diffusion models' compositional image generation abilities. Using adjustable per-image preferences, our method is easily scalable and flexible for different tasks. Evaluation of 9 T2I models reveals that: (1) models remarkable struggle more with compositional tasks with more complex 3D-spatial configurations, and (2) a noticeable performance gap exists between open-source accessible models and closed-source commercial models. Further empirical study on using CompAlign for model alignment yield promising results: post-alignment diffusion models achieve remarkable improvements in compositional accuracy, especially on complex generation tasks, outperforming previous approaches.
>
---
#### [new 065] CROC: Evaluating and Training T2I Metrics with Pseudo- and Human-Labeled Contrastive Robustness Checks
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于文本到图像生成（T2I）的评估指标研究，旨在解决现有评价指标鲁棒性差且人工验证成本高的问题。提出CROC框架，通过合成百万级对比样本（CROC$^{syn}$）自动量化指标缺陷，并训练出SOTA指标CROCScore，同时构建人工标注基准（CROC$^{hum}$）验证复杂场景，揭示现有指标在否定语义、身体部位等场景的显著缺陷。**

- **链接: [http://arxiv.org/pdf/2505.11314v1](http://arxiv.org/pdf/2505.11314v1)**

> **作者:** Christoph Leiter; Yuki M. Asano; Margret Keuper; Steffen Eger
>
> **备注:** preprint
>
> **摘要:** The assessment of evaluation metrics (meta-evaluation) is crucial for determining the suitability of existing metrics in text-to-image (T2I) generation tasks. Human-based meta-evaluation is costly and time-intensive, and automated alternatives are scarce. We address this gap and propose CROC: a scalable framework for automated Contrastive Robustness Checks that systematically probes and quantifies metric robustness by synthesizing contrastive test cases across a comprehensive taxonomy of image properties. With CROC, we generate a pseudo-labeled dataset (CROC$^{syn}$) of over one million contrastive prompt-image pairs to enable a fine-grained comparison of evaluation metrics. We also use the dataset to train CROCScore, a new metric that achieves state-of-the-art performance among open-source methods, demonstrating an additional key application of our framework. To complement this dataset, we introduce a human-supervised benchmark (CROC$^{hum}$) targeting especially challenging categories. Our results highlight robustness issues in existing metrics: for example, many fail on prompts involving negation, and all tested open-source metrics fail on at least 25% of cases involving correct identification of body parts.
>
---
#### [new 066] Maximizing Asynchronicity in Event-based Neural Networks
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于事件相机数据处理任务，旨在解决异步稀疏事件流难以适配传统同步机器学习框架的问题。提出了EVA框架，通过借鉴语言模型的线性注意力与自监督学习，将事件逐次编码为高表达力、强泛化的同步表示，在识别和检测任务中性能超越现有方法，最高达47.7 mAP。**

- **链接: [http://arxiv.org/pdf/2505.11165v1](http://arxiv.org/pdf/2505.11165v1)**

> **作者:** Haiqing Hao; Nikola Zubić; Weihua He; Zhipeng Sui; Davide Scaramuzza; Wenhui Wang
>
> **备注:** 18 pages, 5 figures, 9 tables
>
> **摘要:** Event cameras deliver visual data with high temporal resolution, low latency, and minimal redundancy, yet their asynchronous, sparse sequential nature challenges standard tensor-based machine learning (ML). While the recent asynchronous-to-synchronous (A2S) paradigm aims to bridge this gap by asynchronously encoding events into learned representations for ML pipelines, existing A2S approaches often sacrifice representation expressivity and generalizability compared to dense, synchronous methods. This paper introduces EVA (EVent Asynchronous representation learning), a novel A2S framework to generate highly expressive and generalizable event-by-event representations. Inspired by the analogy between events and language, EVA uniquely adapts advances from language modeling in linear attention and self-supervised learning for its construction. In demonstration, EVA outperforms prior A2S methods on recognition tasks (DVS128-Gesture and N-Cars), and represents the first A2S framework to successfully master demanding detection tasks, achieving a remarkable 47.7 mAP on the Gen1 dataset. These results underscore EVA's transformative potential for advancing real-time event-based vision applications.
>
---
#### [new 067] LARGO: Latent Adversarial Reflection through Gradient Optimization for Jailbreaking LLMs
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文提出LARGO方法，属于大语言模型安全漏洞检测任务，旨在解决离散语言空间下梯度攻击效率低的问题。通过优化连续潜在向量并递归解码生成自然语言，实现高效、流畅的越狱攻击，在基准测试中攻击成功率比现有方法提升44%。**

- **链接: [http://arxiv.org/pdf/2505.10838v1](http://arxiv.org/pdf/2505.10838v1)**

> **作者:** Ran Li; Hao Wang; Chengzhi Mao
>
> **摘要:** Efficient red-teaming method to uncover vulnerabilities in Large Language Models (LLMs) is crucial. While recent attacks often use LLMs as optimizers, the discrete language space make gradient-based methods struggle. We introduce LARGO (Latent Adversarial Reflection through Gradient Optimization), a novel latent self-reflection attack that reasserts the power of gradient-based optimization for generating fluent jailbreaking prompts. By operating within the LLM's continuous latent space, LARGO first optimizes an adversarial latent vector and then recursively call the same LLM to decode the latent into natural language. This methodology yields a fast, effective, and transferable attack that produces fluent and stealthy prompts. On standard benchmarks like AdvBench and JailbreakBench, LARGO surpasses leading jailbreaking techniques, including AutoDAN, by 44 points in attack success rate. Our findings demonstrate a potent alternative to agentic LLM prompting, highlighting the efficacy of interpreting and attacking LLM internals through gradient optimization.
>
---
#### [new 068] EmotionHallucer: Evaluating Emotion Hallucinations in Multimodal Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态大语言模型（MLLMs）的幻觉评估任务，旨在解决模型在情绪理解中生成无关/错误内容的问题。作者提出首个情绪幻觉基准EmotionHallucer，结合心理学知识与多模态感知，通过对抗性QA框架评估38个模型，发现普遍存在情绪幻觉且闭源模型更优，最终提出PEP-MEK框架提升检测效果。**

- **链接: [http://arxiv.org/pdf/2505.11405v1](http://arxiv.org/pdf/2505.11405v1)**

> **作者:** Bohao Xing; Xin Liu; Guoying Zhao; Chengyu Liu; Xiaolan Fu; Heikki Kälviäinen
>
> **摘要:** Emotion understanding is a critical yet challenging task. Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced their capabilities in this area. However, MLLMs often suffer from hallucinations, generating irrelevant or nonsensical content. To the best of our knowledge, despite the importance of this issue, there has been no dedicated effort to evaluate emotion-related hallucinations in MLLMs. In this work, we introduce EmotionHallucer, the first benchmark for detecting and analyzing emotion hallucinations in MLLMs. Unlike humans, whose emotion understanding stems from the interplay of biology and social learning, MLLMs rely solely on data-driven learning and lack innate emotional instincts. Fortunately, emotion psychology provides a solid foundation of knowledge about human emotions. Building on this, we assess emotion hallucinations from two dimensions: emotion psychology knowledge and real-world multimodal perception. To support robust evaluation, we utilize an adversarial binary question-answer (QA) framework, which employs carefully crafted basic and hallucinated pairs to assess the emotion hallucination tendencies of MLLMs. By evaluating 38 LLMs and MLLMs on EmotionHallucer, we reveal that: i) most current models exhibit substantial issues with emotion hallucinations; ii) closed-source models outperform open-source ones in detecting emotion hallucinations, and reasoning capability provides additional advantages; iii) existing models perform better in emotion psychology knowledge than in multimodal emotion perception. As a byproduct, these findings inspire us to propose the PEP-MEK framework, which yields an average improvement of 9.90% in emotion hallucination detection across selected models. Resources will be available at https://github.com/xxtars/EmotionHallucer.
>
---
#### [new 069] Visual Planning: Let's Think Only with Images
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出视觉规划任务，解决现有模型依赖文本进行空间推理的问题。通过纯视觉表示（图像序列）执行规划，开发强化学习框架VPRL训练视觉模型，在FrozenLake等导航任务中超越文本推理方法，验证了图像推理的有效性。**

- **链接: [http://arxiv.org/pdf/2505.11409v1](http://arxiv.org/pdf/2505.11409v1)**

> **作者:** Yi Xu; Chengzu Li; Han Zhou; Xingchen Wan; Caiqi Zhang; Anna Korhonen; Ivan Vulić
>
> **备注:** 10 pages, 6 figures, 1 table (26 pages, 12 figures, 8 tables including references and appendices)
>
> **摘要:** Recent advancements in Large Language Models (LLMs) and their multimodal extensions (MLLMs) have substantially enhanced machine reasoning across diverse tasks. However, these models predominantly rely on pure text as the medium for both expressing and structuring reasoning, even when visual information is present. In this work, we argue that language may not always be the most natural or effective modality for reasoning, particularly in tasks involving spatial and geometrical information. Motivated by this, we propose a new paradigm, Visual Planning, which enables planning through purely visual representations, independent of text. In this paradigm, planning is executed via sequences of images that encode step-by-step inference in the visual domain, akin to how humans sketch or visualize future actions. We introduce a novel reinforcement learning framework, Visual Planning via Reinforcement Learning (VPRL), empowered by GRPO for post-training large vision models, leading to substantial improvements in planning in a selection of representative visual navigation tasks, FrozenLake, Maze, and MiniBehavior. Our visual planning paradigm outperforms all other planning variants that conduct reasoning in the text-only space. Our results establish Visual Planning as a viable and promising alternative to language-based reasoning, opening new avenues for tasks that benefit from intuitive, image-based inference.
>
---
#### [new 070] MPMA: Preference Manipulation Attack Against Model Context Protocol
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于网络安全任务，研究模型上下文协议（MCP）在开放生态中的安全漏洞。针对第三方MCP服务器可能被攻击者操控以劫持LLM工具优先级的问题，提出两种偏好操纵攻击：直接修改工具描述的DPMA和基于遗传算法的隐蔽攻击GAPMA。实验表明GAPMA在效果与隐蔽性间取得平衡，揭示了MCP生态亟需防御机制。**

- **链接: [http://arxiv.org/pdf/2505.11154v1](http://arxiv.org/pdf/2505.11154v1)**

> **作者:** Zihan Wang; Hongwei Li; Rui Zhang; Yu Liu; Wenbo Jiang; Wenshu Fan; Qingchuan Zhao; Guowen Xu
>
> **摘要:** Model Context Protocol (MCP) standardizes interface mapping for large language models (LLMs) to access external data and tools, which revolutionizes the paradigm of tool selection and facilitates the rapid expansion of the LLM agent tool ecosystem. However, as the MCP is increasingly adopted, third-party customized versions of the MCP server expose potential security vulnerabilities. In this paper, we first introduce a novel security threat, which we term the MCP Preference Manipulation Attack (MPMA). An attacker deploys a customized MCP server to manipulate LLMs, causing them to prioritize it over other competing MCP servers. This can result in economic benefits for attackers, such as revenue from paid MCP services or advertising income generated from free servers. To achieve MPMA, we first design a Direct Preference Manipulation Attack ($\mathtt{DPMA}$) that achieves significant effectiveness by inserting the manipulative word and phrases into the tool name and description. However, such a direct modification is obvious to users and lacks stealthiness. To address these limitations, we further propose Genetic-based Advertising Preference Manipulation Attack ($\mathtt{GAPMA}$). $\mathtt{GAPMA}$ employs four commonly used strategies to initialize descriptions and integrates a Genetic Algorithm (GA) to enhance stealthiness. The experiment results demonstrate that $\mathtt{GAPMA}$ balances high effectiveness and stealthiness. Our study reveals a critical vulnerability of the MCP in open ecosystems, highlighting an urgent need for robust defense mechanisms to ensure the fairness of the MCP ecosystem.
>
---
#### [new 071] Creating General User Models from Computer Use
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于用户建模任务，旨在解决现有模型碎片化、无法跨应用灵活推理的问题。提出了通用用户模型(GUM)，通过多模态观察学习用户行为与偏好，构建动态更新的用户知识库，并开发了基于GUM的智能助手系统，可主动预测需求，实验验证其推断准确性。**

- **链接: [http://arxiv.org/pdf/2505.10831v1](http://arxiv.org/pdf/2505.10831v1)**

> **作者:** Omar Shaikh; Shardul Sapkota; Shan Rizvi; Eric Horvitz; Joon Sung Park; Diyi Yang; Michael S. Bernstein
>
> **备注:** 22 pages, 6 figures, 1 table; see https://generalusermodels.github.io/
>
> **摘要:** Human-computer interaction has long imagined technology that understands us-from our preferences and habits, to the timing and purpose of our everyday actions. Yet current user models remain fragmented, narrowly tailored to specific apps, and incapable of the flexible reasoning required to fulfill these visions. This paper presents an architecture for a general user model (GUM) that learns about you by observing any interaction you have with your computer. The GUM takes as input any unstructured observation of a user (e.g., device screenshots) and constructs confidence-weighted propositions that capture that user knowledge and preferences. GUMs can infer that a user is preparing for a wedding they're attending from messages with a friend. Or recognize that a user is struggling with a collaborator's feedback on a draft by observing multiple stalled edits and a switch to reading related work. GUMs introduce an architecture that infers new propositions about a user from multimodal observations, retrieves related propositions for context, and continuously revises existing propositions. To illustrate the breadth of applications that GUMs enable, we demonstrate how they augment chat-based assistants with context, manage OS notifications to selectively surface important information, and enable interactive agents that adapt to preferences across apps. We also instantiate proactive assistants (GUMBOs) that discover and execute useful suggestions on a user's behalf using their GUM. In our evaluations, we find that GUMs make calibrated and accurate inferences about users, and that assistants built on GUMs proactively identify and perform actions that users wouldn't think to request explicitly. Altogether, GUMs introduce methods that leverage multimodal models to understand unstructured context, enabling long-standing visions of HCI and entirely new interactive systems that anticipate user needs.
>
---
#### [new 072] On Next-Token Prediction in LLMs: How End Goals Determine the Consistency of Decoding Algorithms
- **分类: stat.ML; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型（LLM）中不同解码算法（如贪婪、随机采样等）与目标损失函数的一致性，属于解码策略分析任务。通过理论分析发现，随机采样能模拟真实分布，但其他算法仅在特定分布下最优，揭示了信息检索与创意生成目标间的算法选择差异，强调解码算法需根据任务目标优化。**

- **链接: [http://arxiv.org/pdf/2505.11183v1](http://arxiv.org/pdf/2505.11183v1)**

> **作者:** Jacob Trauger; Ambuj Tewari
>
> **备注:** 23 pages
>
> **摘要:** Probabilistic next-token prediction trained using cross-entropy loss is the basis of most large language models. Given a sequence of previous values, next-token prediction assigns a probability to each possible next value in the vocabulary. There are many ways to use next-token prediction to output token sequences. This paper examines a few of these algorithms (greedy, lookahead, random sampling, and temperature-scaled random sampling) and studies their consistency with respect to various goals encoded as loss functions. Although consistency of surrogate losses with respect to a target loss function is a well researched topic, we are the first to study it in the context of LLMs (to the best of our knowledge). We find that, so long as next-token prediction converges to its true probability distribution, random sampling is consistent with outputting sequences that mimic sampling from the true probability distribution. For the other goals, such as minimizing the 0-1 loss on the entire sequence, we show no polynomial-time algorithm is optimal for all probability distributions and all decoding algorithms studied are only optimal for a subset of probability distributions. When analyzing these results, we see that there is a dichotomy created between the goals of information retrieval and creative generation for the decoding algorithms. This shows that choosing the correct decoding algorithm based on the desired goal is extremely important and many of the ones used are lacking theoretical grounding in numerous scenarios.
>
---
#### [new 073] Audio Turing Test: Benchmarking the Human-likeness of Large Language Model-based Text-to-Speech Systems in Chinese
- **分类: cs.SD; cs.AI; cs.CL; cs.HC; cs.LG; eess.AS**

- **简介: 该论文属于文本到语音（TTS）评估任务，旨在解决中文TTS系统评测中传统方法（如MOS）的主观性、维度单一等问题。提出多维度中文数据集ATT-Corpus及基于图灵测试的评测协议，通过人类判听简化评估，并开发自动评测工具Auto-ATT，验证其与人工评价的一致性，提升评测效率和客观性。**

- **链接: [http://arxiv.org/pdf/2505.11200v1](http://arxiv.org/pdf/2505.11200v1)**

> **作者:** Xihuai Wang; Ziyi Zhao; Siyu Ren; Shao Zhang; Song Li; Xiaoyu Li; Ziwen Wang; Lin Qiu; Guanglu Wan; Xuezhi Cao; Xunliang Cai; Weinan Zhang
>
> **备注:** Under Review
>
> **摘要:** Recent advances in large language models (LLMs) have significantly improved text-to-speech (TTS) systems, enhancing control over speech style, naturalness, and emotional expression, which brings TTS Systems closer to human-level performance. Although the Mean Opinion Score (MOS) remains the standard for TTS System evaluation, it suffers from subjectivity, environmental inconsistencies, and limited interpretability. Existing evaluation datasets also lack a multi-dimensional design, often neglecting factors such as speaking styles, context diversity, and trap utterances, which is particularly evident in Chinese TTS evaluation. To address these challenges, we introduce the Audio Turing Test (ATT), a multi-dimensional Chinese corpus dataset ATT-Corpus paired with a simple, Turing-Test-inspired evaluation protocol. Instead of relying on complex MOS scales or direct model comparisons, ATT asks evaluators to judge whether a voice sounds human. This simplification reduces rating bias and improves evaluation robustness. To further support rapid model development, we also finetune Qwen2-Audio-Instruct with human judgment data as Auto-ATT for automatic evaluation. Experimental results show that ATT effectively differentiates models across specific capability dimensions using its multi-dimensional design. Auto-ATT also demonstrates strong alignment with human evaluations, confirming its value as a fast and reliable assessment tool. The white-box ATT-Corpus and Auto-ATT can be found in ATT Hugging Face Collection (https://huggingface.co/collections/meituan/audio-turing-test-682446320368164faeaf38a4).
>
---
#### [new 074] Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究大语言模型测试时扩展中提示策略的作用，属模型优化任务。针对多数投票场景下复杂提示策略扩展性能差的问题，通过实验和概率分析，提出性能预测方法及改进策略，证明简单提示在计算增加时更优。**

- **链接: [http://arxiv.org/pdf/2505.10981v1](http://arxiv.org/pdf/2505.10981v1)**

> **作者:** Yexiang Liu; Zekun Li; Zhi Fang; Nan Xu; Ran He; Tieniu Tan
>
> **备注:** ACL 2025 Main
>
> **摘要:** Recently, scaling test-time compute on Large Language Models (LLM) has garnered wide attention. However, there has been limited investigation of how various reasoning prompting strategies perform as scaling. In this paper, we focus on a standard and realistic scaling setting: majority voting. We systematically conduct experiments on 6 LLMs $\times$ 8 prompting strategies $\times$ 6 benchmarks. Experiment results consistently show that as the sampling time and computational overhead increase, complicated prompting strategies with superior initial performance gradually fall behind simple Chain-of-Thought. We analyze this phenomenon and provide theoretical proofs. Additionally, we propose a method according to probability theory to quickly and accurately predict the scaling performance and select the best strategy under large sampling times without extra resource-intensive inference in practice. It can serve as the test-time scaling law for majority voting. Furthermore, we introduce two ways derived from our theoretical analysis to significantly improve the scaling performance. We hope that our research can promote to re-examine the role of complicated prompting, unleash the potential of simple prompting strategies, and provide new insights for enhancing test-time scaling performance.
>
---
#### [new 075] Understanding Gen Alpha Digital Language: Evaluation of LLM Safety Systems for Content Moderation
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; I.2; I.2.7; K.4.2**

- **简介: 该论文属于AI内容审核评估任务，旨在解决现有安全系统无法识别Gen Alpha（2010-2024出生）数字语言中隐蔽有害内容的问题。研究评估了GPT-4等4个模型对游戏/社交媒体中100条新型表达的检测能力，发现系统存在理解缺陷。通过构建首个Gen Alpha数据集、提出改进框架及多视角（AI/人类/家长/青少年）分析，揭示了语言差异加剧青少年风险，呼吁优化安全系统以适应其独特沟通方式。**

- **链接: [http://arxiv.org/pdf/2505.10588v1](http://arxiv.org/pdf/2505.10588v1)**

> **作者:** Manisha Mehta; Fausto Giunchiglia
>
> **备注:** Accepted to ACM FAccT 2025. To be presented in Athens, June 2025, and published in the conference proceedings. Preprint version; final version will appear in the ACM Digital Library
>
> **摘要:** This research offers a unique evaluation of how AI systems interpret the digital language of Generation Alpha (Gen Alpha, born 2010-2024). As the first cohort raised alongside AI, Gen Alpha faces new forms of online risk due to immersive digital engagement and a growing mismatch between their evolving communication and existing safety tools. Their distinct language, shaped by gaming, memes, and AI-driven trends, often conceals harmful interactions from both human moderators and automated systems. We assess four leading AI models (GPT-4, Claude, Gemini, and Llama 3) on their ability to detect masked harassment and manipulation within Gen Alpha discourse. Using a dataset of 100 recent expressions from gaming platforms, social media, and video content, the study reveals critical comprehension failures with direct implications for online safety. This work contributes: (1) a first-of-its-kind dataset capturing Gen Alpha expressions; (2) a framework to improve AI moderation systems for youth protection; (3) a multi-perspective evaluation including AI systems, human moderators, and parents, with direct input from Gen Alpha co-researchers; and (4) an analysis of how linguistic divergence increases youth vulnerability. Findings highlight the urgent need to redesign safety systems attuned to youth communication, especially given Gen Alpha reluctance to seek help when adults fail to understand their digital world. This study combines the insight of a Gen Alpha researcher with systematic academic analysis to address critical digital safety challenges.
>
---
#### [new 076] Creativity or Brute Force? Using Brainteasers as a Window into the Problem-Solving Abilities of Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的模型推理能力评估任务，旨在探究大语言模型（LLMs）解决复杂问题的策略是创造性还是机械化的。通过构建基于叙事型脑筋急转弯的评估基准，从语义解析、解法生成、自我纠正等五个层面分析LLMs的推理质量，发现其虽具备一定创造性解题能力，但仍存在依赖低效方法的局限性，为优化模型推理指明了方向。（99字）**

- **链接: [http://arxiv.org/pdf/2505.10844v1](http://arxiv.org/pdf/2505.10844v1)**

> **作者:** Simeng Han; Stephen Xia; Grant Zhang; Howard Dai; Chen Liu; Lichang Chen; Hoang Huy Nguyen; Hongyuan Mei; Jiayuan Mao; R. Thomas McCoy
>
> **备注:** 13 Tables; 5 Figures
>
> **摘要:** Accuracy remains a standard metric for evaluating AI systems, but it offers limited insight into how models arrive at their solutions. In this work, we introduce a benchmark based on brainteasers written in long narrative form to probe more deeply into the types of reasoning strategies that models use. Brainteasers are well-suited for this goal because they can be solved with multiple approaches, such as a few-step solution that uses a creative insight or a longer solution that uses more brute force. We investigate large language models (LLMs) across multiple layers of reasoning, focusing not only on correctness but also on the quality and creativity of their solutions. We investigate many aspects of the reasoning process: (1) semantic parsing of the brainteasers into precise mathematical competition style formats; (2) generating solutions from these mathematical forms; (3) self-correcting solutions based on gold solutions; (4) producing step-by-step sketches of solutions; and (5) making use of hints. We find that LLMs are in many cases able to find creative, insightful solutions to brainteasers, suggesting that they capture some of the capacities needed to solve novel problems in creative ways. Nonetheless, there also remain situations where they rely on brute force despite the availability of more efficient, creative solutions, highlighting a potential direction for improvement in the reasoning abilities of LLMs.
>
---
#### [new 077] $\mathcal{A}LLM4ADD$: Unlocking the Capabilities of Audio Large Language Models for Audio Deepfake Detection
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决现有音频大语言模型（ALLM）在零样本下检测伪造音频效果差的问题。研究提出ALLM4ADD框架，将检测任务重构为音频问答问题，通过监督微调使ALLM识别音频真实性，在数据稀缺场景中实现高效检测，为开发更优检测系统提供新思路。**

- **链接: [http://arxiv.org/pdf/2505.11079v1](http://arxiv.org/pdf/2505.11079v1)**

> **作者:** Hao Gu; Jiangyan Yi; Chenglong Wang; Jianhua Tao; Zheng Lian; Jiayi He; Yong Ren; Yujie Chen; Zhengqi Wen
>
> **摘要:** Audio deepfake detection (ADD) has grown increasingly important due to the rise of high-fidelity audio generative models and their potential for misuse. Given that audio large language models (ALLMs) have made significant progress in various audio processing tasks, a heuristic question arises: Can ALLMs be leveraged to solve ADD?. In this paper, we first conduct a comprehensive zero-shot evaluation of ALLMs on ADD, revealing their ineffectiveness in detecting fake audio. To enhance their performance, we propose $\mathcal{A}LLM4ADD$, an ALLM-driven framework for ADD. Specifically, we reformulate ADD task as an audio question answering problem, prompting the model with the question: "Is this audio fake or real?". We then perform supervised fine-tuning to enable the ALLM to assess the authenticity of query audio. Extensive experiments are conducted to demonstrate that our ALLM-based method can achieve superior performance in fake audio detection, particularly in data-scarce scenarios. As a pioneering study, we anticipate that this work will inspire the research community to leverage ALLMs to develop more effective ADD systems.
>
---
#### [new 078] Phare: A Safety Probe for Large Language Models
- **分类: cs.CY; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于大语言模型安全评估任务，旨在解决现有评测忽视故障模式识别的问题。研究者提出了Phare框架，通过检测幻觉、偏见、有害内容三个维度，系统性评估17个主流模型的漏洞（如奉承行为、刻板印象），为构建安全可靠的语言系统提供改进方向。**

- **链接: [http://arxiv.org/pdf/2505.11365v1](http://arxiv.org/pdf/2505.11365v1)**

> **作者:** Pierre Le Jeune; Benoît Malésieux; Weixuan Xiao; Matteo Dora
>
> **摘要:** Ensuring the safety of large language models (LLMs) is critical for responsible deployment, yet existing evaluations often prioritize performance over identifying failure modes. We introduce Phare, a multilingual diagnostic framework to probe and evaluate LLM behavior across three critical dimensions: hallucination and reliability, social biases, and harmful content generation. Our evaluation of 17 state-of-the-art LLMs reveals patterns of systematic vulnerabilities across all safety dimensions, including sycophancy, prompt sensitivity, and stereotype reproduction. By highlighting these specific failure modes rather than simply ranking models, Phare provides researchers and practitioners with actionable insights to build more robust, aligned, and trustworthy language systems.
>
---
#### [new 079] Towards Automated Situation Awareness: A RAG-Based Framework for Peacebuilding Reports
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于自动化文本生成任务，解决人工分析海量数据效率低导致决策延迟的问题。提出基于检索增强生成（RAG）的动态系统，整合多源实时数据自动生成和平建设报告，通过语义评估、专家验证和LLM评审的三级框架确保报告质量，经真实场景验证有效，并开源代码工具。**

- **链接: [http://arxiv.org/pdf/2505.10586v1](http://arxiv.org/pdf/2505.10586v1)**

> **作者:** Poli A. Nemkova; Suleyman O. Polat; Rafid I. Jahan; Sagnik Ray Choudhury; Sun-joo Lee; Shouryadipta Sarkar; Mark V. Albert
>
> **摘要:** Timely and accurate situation awareness is vital for decision-making in humanitarian response, conflict monitoring, and early warning and early action. However, the manual analysis of vast and heterogeneous data sources often results in delays, limiting the effectiveness of interventions. This paper introduces a dynamic Retrieval-Augmented Generation (RAG) system that autonomously generates situation awareness reports by integrating real-time data from diverse sources, including news articles, conflict event databases, and economic indicators. Our system constructs query-specific knowledge bases on demand, ensuring timely, relevant, and accurate insights. To ensure the quality of generated reports, we propose a three-level evaluation framework that combines semantic similarity metrics, factual consistency checks, and expert feedback. The first level employs automated NLP metrics to assess coherence and factual accuracy. The second level involves human expert evaluation to verify the relevance and completeness of the reports. The third level utilizes LLM-as-a-Judge, where large language models provide an additional layer of assessment to ensure robustness. The system is tested across multiple real-world scenarios, demonstrating its effectiveness in producing coherent, insightful, and actionable reports. By automating report generation, our approach reduces the burden on human analysts and accelerates decision-making processes. To promote reproducibility and further research, we openly share our code and evaluation tools with the community via GitHub.
>
---
#### [new 080] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文研究具身智能体在任务规划中理解模糊人类指令的问题，属于机器人任务规划领域。针对现有大模型规划器假设指令清晰、无法处理模糊指称表达（REs）的缺陷，构建了REI-Bench基准测试，发现模糊REs可使成功率骤降77.9%。提出任务导向的上下文认知方法，通过生成清晰指令提升性能，助力非专业用户的人机交互。**

- **链接: [http://arxiv.org/pdf/2505.10872v1](http://arxiv.org/pdf/2505.10872v1)**

> **作者:** Chenxi Jiang; Chuhao Zhou; Jianfei Yang
>
> **备注:** Submitted to CoRL 2025, under review
>
> **摘要:** Robot task planning decomposes human instructions into executable action sequences that enable robots to complete a series of complex tasks. Although recent large language model (LLM)-based task planners achieve amazing performance, they assume that human instructions are clear and straightforward. However, real-world users are not experts, and their instructions to robots often contain significant vagueness. Linguists suggest that such vagueness frequently arises from referring expressions (REs), whose meanings depend heavily on dialogue context and environment. This vagueness is even more prevalent among the elderly and children, who robots should serve more. This paper studies how such vagueness in REs within human instructions affects LLM-based robot task planning and how to overcome this issue. To this end, we propose the first robot task planning benchmark with vague REs (REI-Bench), where we discover that the vagueness of REs can severely degrade robot planning performance, leading to success rate drops of up to 77.9%. We also observe that most failure cases stem from missing objects in planners. To mitigate the REs issue, we propose a simple yet effective approach: task-oriented context cognition, which generates clear instructions for robots, achieving state-of-the-art performance compared to aware prompt and chains of thought. This work contributes to the research community of human-robot interaction (HRI) by making robot task planning more practical, particularly for non-expert users, e.g., the elderly and children.
>
---
#### [new 081] Relative Drawing Identification Complexity is Invariant to Modality in Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉-语言模型中概念识别复杂度是否跨模态一致，属于多模态表征分析任务。通过比较图像位图和笔画坐标两种模态在Quick, Draw!数据集上的教学效率，发现图像模态教学更高效，但概念复杂度排序跨模态呈现显著一致性，表明概念简繁具有独立于表征模态的内在属性。**

- **链接: [http://arxiv.org/pdf/2505.10583v1](http://arxiv.org/pdf/2505.10583v1)**

> **作者:** Diogo Freitas; Brigt Håvardstun; Cèsar Ferri; Darío Garigliotti; Jan Arne Telle; José Hernández-Orallo
>
> **备注:** 54 pages (42 pages of appendix)
>
> **摘要:** Large language models have become multimodal, and many of them are said to integrate their modalities using common representations. If this were true, a drawing of a car as an image, for instance, should map to the similar area in the latent space as a textual description of the strokes that conform the drawing. To explore this in a black-box access regime to these models, we propose the use of machine teaching, a theory that studies the minimal set of examples a teacher needs to choose so that the learner captures the concept. In this paper we evaluate the complexity of teaching visual-language models a subset of objects in the Quick, Draw! dataset using two presentations: raw images as bitmaps and trace coordinates in TikZ format. The results indicate that image-based representations generally require fewer segments and achieve higher accuracy than coordinate-based representations. But, surprisingly, the teaching size usually ranks concepts similarly across both modalities, even when controlling for (a human proxy of) concept priors, suggesting that the simplicity of concepts may be an inherent property that transcends modality representations.
>
---
## 更新

#### [replaced 001] How Good is Your Wikipedia? Auditing Data Quality for Low-resource and Multilingual NLP
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.05527v2](http://arxiv.org/pdf/2411.05527v2)**

> **作者:** Kushal Tatariya; Artur Kulmizev; Wessel Poelman; Esther Ploeger; Marcel Bollmann; Johannes Bjerva; Jiaming Luo; Heather Lent; Miryam de Lhoneux
>
> **摘要:** Wikipedia's perceived high quality and broad language coverage have established it as a fundamental resource in multilingual NLP. In the context of low-resource languages, however, these quality assumptions are increasingly being scrutinised. This paper critically examines the data quality of Wikipedia in a non-English setting by subjecting it to various quality filtering techniques, revealing widespread issues such as a high percentage of one-line articles and duplicate articles. We evaluate the downstream impact of quality filtering on Wikipedia and find that data quality pruning is an effective means for resource-efficient training without hurting performance, especially for low-resource languages. Moreover, we advocate for a shift in perspective from seeking a general definition of data quality towards a more language- and task-specific one. Ultimately, we aim for this study to serve as a guide to using Wikipedia for pretraining in a multilingual setting.
>
---
#### [replaced 002] Unveiling Attractor Cycles in Large Language Models: A Dynamical Systems View of Successive Paraphrasing
- **分类: cs.CL; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.15208v2](http://arxiv.org/pdf/2502.15208v2)**

> **作者:** Zhilin Wang; Yafu Li; Jianhao Yan; Yu Cheng; Yue Zhang
>
> **备注:** 9 pages
>
> **摘要:** Dynamical systems theory provides a framework for analyzing iterative processes and evolution over time. Within such systems, repetitive transformations can lead to stable configurations, known as attractors, including fixed points and limit cycles. Applying this perspective to large language models (LLMs), which iteratively map input text to output text, provides a principled approach to characterizing long-term behaviors. Successive paraphrasing serves as a compelling testbed for exploring such dynamics, as paraphrases re-express the same underlying meaning with linguistic variation. Although LLMs are expected to explore a diverse set of paraphrases in the text space, our study reveals that successive paraphrasing converges to stable periodic states, such as 2-period attractor cycles, limiting linguistic diversity. This phenomenon is attributed to the self-reinforcing nature of LLMs, as they iteratively favour and amplify certain textual forms over others. This pattern persists with increasing generation randomness or alternating prompts and LLMs. These findings underscore inherent constraints in LLM generative capability, while offering a novel dynamical systems perspective for studying their expressive potential.
>
---
#### [replaced 003] Structured Preference Optimization for Vision-Language Long-Horizon Task Planning
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20742v3](http://arxiv.org/pdf/2502.20742v3)**

> **作者:** Xiwen Liang; Min Lin; Weiqi Ruan; Rongtao Xu; Yuecheng Liu; Jiaqi Chen; Bingqian Lin; Yuzheng Zhuang; Xiaodan Liang
>
> **备注:** 18 pages
>
> **摘要:** Existing methods for vision-language task planning excel in short-horizon tasks but often fall short in complex, long-horizon planning within dynamic environments. These challenges primarily arise from the difficulty of effectively training models to produce high-quality reasoning processes for long-horizon tasks. To address this, we propose Structured Preference Optimization (SPO), which aims to enhance reasoning and action selection in long-horizon task planning through structured preference evaluation and optimized training strategies. Specifically, SPO introduces: 1) Preference-Based Scoring and Optimization, which systematically evaluates reasoning chains based on task relevance, visual grounding, and historical consistency; and 2) Curriculum-Guided Training, where the model progressively adapts from simple to complex tasks, improving its generalization ability in long-horizon scenarios and enhancing reasoning robustness. To advance research in vision-language long-horizon task planning, we introduce ExtendaBench, a comprehensive benchmark covering 1,509 tasks across VirtualHome and Habitat 2.0, categorized into ultra-short, short, medium, and long tasks. Experimental results demonstrate that SPO significantly improves reasoning quality and final decision accuracy, outperforming prior methods on long-horizon tasks and underscoring the effectiveness of preference-driven optimization in vision-language task planning. Specifically, SPO achieves a +5.98% GCR and +4.68% SR improvement in VirtualHome and a +3.30% GCR and +2.11% SR improvement in Habitat over the best-performing baselines.
>
---
#### [replaced 004] XRAG: eXamining the Core -- Benchmarking Foundational Components in Advanced Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.15529v3](http://arxiv.org/pdf/2412.15529v3)**

> **作者:** Qianren Mao; Yangyifei Luo; Qili Zhang; Yashuo Luo; Zhilong Cao; Jinlong Zhang; HanWen Hao; Zhijun Chen; Weifeng Jiang; Junnan Liu; Xiaolong Wang; Zhenting Huang; Zhixing Tan; Sun Jie; Bo Li; Xudong Liu; Richong Zhang; Jianxin Li
>
> **摘要:** Retrieval-augmented generation (RAG) synergizes the retrieval of pertinent data with the generative capabilities of Large Language Models (LLMs), ensuring that the generated output is not only contextually relevant but also accurate and current. We introduce XRAG, an open-source, modular codebase that facilitates exhaustive evaluation of the performance of foundational components of advanced RAG modules. These components are systematically categorized into four core phases: pre-retrieval, retrieval, post-retrieval, and generation. We systematically analyse them across reconfigured datasets, providing a comprehensive benchmark for their effectiveness. As the complexity of RAG systems continues to escalate, we underscore the critical need to identify potential failure points in RAG systems. We formulate a suite of experimental methodologies and diagnostic testing protocols to dissect the failure points inherent in RAG engineering. Subsequently, we proffer bespoke solutions aimed at bolstering the overall performance of these modules. Our work thoroughly evaluates the performance of advanced core components in RAG systems, providing insights into optimizations for prevalent failure points.
>
---
#### [replaced 005] DRA-GRPO: Exploring Diversity-Aware Reward Adjustment for R1-Zero-Like Training of Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.09655v2](http://arxiv.org/pdf/2505.09655v2)**

> **作者:** Xiwen Chen; Wenhui Zhu; Peijie Qiu; Xuanzhao Dong; Hao Wang; Haiyu Wu; Huayu Li; Aristeidis Sotiras; Yalin Wang; Abolfazl Razi
>
> **摘要:** Recent advances in reinforcement learning for language model post-training, such as Group Relative Policy Optimization (GRPO), have shown promise in low-resource settings. However, GRPO typically relies on solution-level and scalar reward signals that fail to capture the semantic diversity among sampled completions. This leads to what we identify as a diversity-quality inconsistency, where distinct reasoning paths may receive indistinguishable rewards. To address this limitation, we propose $\textit{Diversity-aware Reward Adjustment}$ (DRA), a method that explicitly incorporates semantic diversity into the reward computation. DRA uses Submodular Mutual Information (SMI) to downweight redundant completions and amplify rewards for diverse ones. This encourages better exploration during learning, while maintaining stable exploitation of high-quality samples. Our method integrates seamlessly with both GRPO and its variant DR.~GRPO, resulting in $\textit{DRA-GRPO}$ and $\textit{DGA-DR.~GRPO}$. We evaluate our method on five mathematical reasoning benchmarks and find that it outperforms recent strong baselines. It achieves state-of-the-art performance with an average accuracy of 58.2%, using only 7,000 fine-tuning samples and a total training cost of approximately $55. The code is available at https://github.com/xiwenc1/DRA-GRPO.
>
---
#### [replaced 006] Do Theory of Mind Benchmarks Need Explicit Human-like Reasoning in Language Models?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.01698v3](http://arxiv.org/pdf/2504.01698v3)**

> **作者:** Yi-Long Lu; Chunhui Zhang; Jiajun Song; Lifeng Fan; Wei Wang
>
> **摘要:** Theory of Mind (ToM), the ability to attribute mental states to others, is fundamental for human social intelligence and a critical capability for advanced Artificial Intelligence. Recent advancements in Large Language Models (LLMs) have shown promising performance on ToM benchmarks, raising the question: Do these benchmarks necessitate explicit human-like reasoning processes, or can models succeed through alternative strategies? We investigate this question empirically by applying Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) to LLMs of varying scales (0.5B to 7B parameters) and evaluating them across multiple ToM datasets. Our results reveal a scale-dependent impact of RL: while RL significantly improves accuracy and fosters high-quality, interpretable, and transferable belief-tracking reasoning in larger models (7B), it leads to "reasoning collapse" in smaller models ($\leq$3B), where high accuracy and generalization ability are achieved via drastically shortened, less meaningful responses. Surprisingly, further SFT achieves competitive and generalizable performance across these benchmarks, often matching or exceeding RL models in accuracy, despite not being explicitly trained to produce structured reasoning traces. These findings highlight a critical discrepancy between benchmark accuracy and the nature of learned reasoning. Our work suggests that current ToM benchmarks may be solvable without requiring the explicit, human-like simulation of mental states they were designed to probe. LLMs, particularly when scale is limited or training signals focus solely on output correctness, may leverage alternative rules effective for benchmark data structures.
>
---
#### [replaced 007] COBIAS: Assessing the Contextual Reliability of Bias Benchmarks for Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2402.14889v4](http://arxiv.org/pdf/2402.14889v4)**

> **作者:** Priyanshul Govil; Hemang Jain; Vamshi Krishna Bonagiri; Aman Chadha; Ponnurangam Kumaraguru; Manas Gaur; Sanorita Dey
>
> **摘要:** Large Language Models (LLMs) often inherit biases from the web data they are trained on, which contains stereotypes and prejudices. Current methods for evaluating and mitigating these biases rely on bias-benchmark datasets. These benchmarks measure bias by observing an LLM's behavior on biased statements. However, these statements lack contextual considerations of the situations they try to present. To address this, we introduce a contextual reliability framework, which evaluates model robustness to biased statements by considering the various contexts in which they may appear. We develop the Context-Oriented Bias Indicator and Assessment Score (COBIAS) to measure a biased statement's reliability in detecting bias, based on the variance in model behavior across different contexts. To evaluate the metric, we augmented 2,291 stereotyped statements from two existing benchmark datasets by adding contextual information. We show that COBIAS aligns with human judgment on the contextual reliability of biased statements (Spearman's $\rho = 0.65, p = 3.4 * 10^{-60}$) and can be used to create reliable benchmarks, which would assist bias mitigation works.
>
---
#### [replaced 008] ShifCon: Enhancing Non-Dominant Language Capabilities with a Shift-based Contrastive Framework
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.19453v5](http://arxiv.org/pdf/2410.19453v5)**

> **作者:** Hengyuan Zhang; Chenming Shang; Sizhe Wang; Dongdong Zhang; Feng Yao; Renliang Sun; Yiyao Yu; Yujiu Yang; Furu Wei
>
> **备注:** 23 pages, 11 figures
>
> **摘要:** Although fine-tuning Large Language Models (LLMs) with multilingual data can rapidly enhance the multilingual capabilities of LLMs, they still exhibit a performance gap between the dominant language (e.g., English) and non-dominant ones due to the imbalance of training data across languages. To further enhance the performance of non-dominant languages, we propose ShifCon, a Shift-based Contrastive framework that aligns the internal forward process of other languages toward that of the dominant one. Specifically, it shifts the representations of non-dominant languages into the dominant language subspace, allowing them to access relatively rich information encoded in the model parameters. The enriched representations are then shifted back into their original language subspace before generation. Moreover, we introduce a subspace distance metric to pinpoint the optimal layer area for shifting representations and employ multilingual contrastive learning to further enhance the alignment of representations within this area. Experiments demonstrate that our ShifCon framework significantly enhances the performance of non-dominant languages, particularly for low-resource ones. Further analysis offers extra insights to verify the effectiveness of ShifCon and propel future research
>
---
#### [replaced 009] MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.14731v2](http://arxiv.org/pdf/2410.14731v2)**

> **作者:** Bokai Lin; Zihao Zeng; Zipeng Xiao; Siqi Kou; Tianqi Hou; Xiaofeng Gao; Hao Zhang; Zhijie Deng
>
> **摘要:** KV cache has become a de facto technique for the inference of large language models (LLMs), where tensors of shape (layer number, head number, sequence length, feature dimension) are introduced to cache historical information for self-attention. As the size of the model and data grows, the KV cache can quickly become a bottleneck within the system in both storage and memory transfer. To address this, prior studies usually focus on the first three axes of the cache tensors for compression. This paper supplements them, focusing on the feature dimension axis, by utilizing low-rank projection matrices to transform the cache features into spaces with reduced dimensions. We begin by investigating the canonical orthogonal projection method for data compression through principal component analysis (PCA). We observe the issue with PCA projection where significant performance degradation is observed at low compression rates. To bridge the gap, we propose to directly tune the orthogonal projection matrices with a distillation objective using an elaborate Matryoshka training strategy. After training, we adaptively search for the optimal compression rates for various layers and heads given varying compression budgets. Compared to previous works, our method can easily embrace pre-trained LLMs and hold a smooth tradeoff between performance and compression rate. We empirically witness the high data efficiency of our training procedure and find that our method can sustain over 90% performance with an average KV cache compression rate of 60% (and up to 75% in certain extreme scenarios) for popular LLMs like LLaMA2-7B-base and Mistral-7B-v0.3-base.
>
---
#### [replaced 010] Med-R$^2$: Crafting Trustworthy LLM Physicians via Retrieval and Reasoning of Evidence-Based Medicine
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.11885v4](http://arxiv.org/pdf/2501.11885v4)**

> **作者:** Keer Lu; Zheng Liang; Zhuoran Zhang; Da Pan; Shusen Zhang; Xin Wu; Zenan Zhou; Guosheng Dong; Bin Cui; Tengjiao Wang; Wentao Zhang
>
> **摘要:** Large Language Models (LLMs) have exhibited remarkable capabilities in clinical scenarios. Despite their potential, existing works face challenges when applying LLMs to medical settings. Strategies relying on training with medical datasets are highly cost-intensive and may suffer from outdated training data. Leveraging external knowledge bases is a suitable alternative, yet it faces obstacles such as limited retrieval precision and poor effectiveness in answer extraction. These issues collectively prevent LLMs from demonstrating the expected level of proficiency in mastering medical expertise. To address these challenges, we introduce Med-R^2, a novel LLM physician framework that adheres to the Evidence-Based Medicine (EBM) process, efficiently integrating retrieval mechanisms as well as the selection and reasoning processes of evidence, thereby enhancing the problem-solving capabilities of LLMs in healthcare scenarios and fostering a trustworthy LLM physician. Our comprehensive experiments indicate that Med-R^2 achieves a 14.74\% improvement over vanilla RAG methods and even a 3.32\% enhancement compared to fine-tuning strategies, without incurring additional training costs.
>
---
#### [replaced 011] Mixture of Routers
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.23362v2](http://arxiv.org/pdf/2503.23362v2)**

> **作者:** Jia-Chen Zhang; Yu-Jie Xiong; Xi-He Qiu; Chun-Ming Xia; Fei Dai
>
> **备注:** 10 pages,4 figures
>
> **摘要:** Supervised fine-tuning (SFT) is a milestone in aligning large language models with human instructions and adapting them to downstream tasks. In particular, Low-Rank Adaptation (LoRA) has gained widespread attention due to its parameter efficiency. However, its impact on improving the performance of large models remains limited. Recent studies suggest that combining LoRA with Mixture-of-Experts (MoE) can significantly enhance fine-tuning performance. MoE adapts to the diversity and complexity of datasets by dynamically selecting the most suitable experts, thereby improving task accuracy and efficiency. Despite impressive results, recent studies reveal issues in the MoE routing mechanism, such as incorrect assignments and imbalanced expert allocation. Inspired by the principles of Redundancy and Fault Tolerance Theory. We innovatively integrate the concept of Mixture of Experts into the routing mechanism and propose an efficient fine-tuning method called Mixture of Routers (MoR). It employs multiple sub-routers for joint selection and uses a learnable main router to determine the weights of the sub-routers. The results show that MoR outperforms baseline models on most tasks, achieving an average performance improvement of 1%. MoR can serve as a plug-and-play, parameter-efficient fine-tuning method suitable for a wide range of applications. Our code is available here: https://anonymous.4open.science/r/MoR-DFC6.
>
---
#### [replaced 012] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.07019v3](http://arxiv.org/pdf/2411.07019v3)**

> **作者:** Zhiqiang Liu; Yin Hua; Mingyang Chen; Zhuo Chen; Ziqi Liu; Lei Liang; Huajun Chen; Wen Zhang
>
> **摘要:** Beyond-triple fact representations including hyper-relational facts with auxiliary key-value pairs, temporal facts with additional timestamps, and nested facts implying relationships between facts, are gaining significant attention. However, constrained by complex fact representation forms, existing link prediction models for beyond-triple facts have difficulty achieving hierarchical fact modeling and generalizing the modules for one specific facts to other fact types. To overcome this limitation, we propose a Unified Hierarchical Representation learning framework (UniHR) for unified knowledge graph link prediction. It consists of a unified Hierarchical Data Representation (HiDR) module and a unified Hierarchical Structure Learning (HiSL) module as graph encoder. The HiDR module unifies hyper-relational KGs, temporal KGs, and nested factual KGs into triple-based representations. Then HiSL incorporates intra-fact and inter-fact message passing, focusing on enhancing the semantic information within individual facts and enriching the structural information between facts. Empirical results demonstrate the effectiveness of UniHR and highlight the strong potential of unified representations. Code and data are available at https://github.com/Lza12a/UniHR.
>
---
#### [replaced 013] Safety in Large Reasoning Models: A Survey
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17704v2](http://arxiv.org/pdf/2504.17704v2)**

> **作者:** Cheng Wang; Yue Liu; Baolong Bi; Duzhen Zhang; Zhongzhi Li; Junfeng Fang; Bryan Hooi
>
> **摘要:** Large Reasoning Models (LRMs) have exhibited extraordinary prowess in tasks like mathematics and coding, leveraging their advanced reasoning capabilities. Nevertheless, as these capabilities progress, significant concerns regarding their vulnerabilities and safety have arisen, which can pose challenges to their deployment and application in real-world settings. This paper presents a comprehensive survey of LRMs, meticulously exploring and summarizing the newly emerged safety risks, attacks, and defense strategies. By organizing these elements into a detailed taxonomy, this work aims to offer a clear and structured understanding of the current safety landscape of LRMs, facilitating future research and development to enhance the security and reliability of these powerful models.
>
---
#### [replaced 014] Can Authorship Attribution Models Distinguish Speakers in Speech Transcripts?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2311.07564v4](http://arxiv.org/pdf/2311.07564v4)**

> **作者:** Cristina Aggazzotti; Nicholas Andrews; Elizabeth Allyn Smith
>
> **备注:** Published in Transactions of the Association for Computational Linguistics; 1st revision includes additional experiments and evaluations; 2nd revision includes minor tweak to TFIDF table numbers
>
> **摘要:** Authorship verification is the task of determining if two distinct writing samples share the same author and is typically concerned with the attribution of written text. In this paper, we explore the attribution of transcribed speech, which poses novel challenges. The main challenge is that many stylistic features, such as punctuation and capitalization, are not informative in this setting. On the other hand, transcribed speech exhibits other patterns, such as filler words and backchannels (e.g., 'um', 'uh-huh'), which may be characteristic of different speakers. We propose a new benchmark for speaker attribution focused on human-transcribed conversational speech transcripts. To limit spurious associations of speakers with topic, we employ both conversation prompts and speakers participating in the same conversation to construct verification trials of varying difficulties. We establish the state of the art on this new benchmark by comparing a suite of neural and non-neural baselines, finding that although written text attribution models achieve surprisingly good performance in certain settings, they perform markedly worse as conversational topic is increasingly controlled. We present analyses of the impact of transcription style on performance as well as the ability of fine-tuning on speech transcripts to improve performance.
>
---
#### [replaced 015] Can Your Uncertainty Scores Detect Hallucinated Entity?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11948v2](http://arxiv.org/pdf/2502.11948v2)**

> **作者:** Min-Hsuan Yeh; Max Kamachee; Seongheon Park; Yixuan Li
>
> **摘要:** To mitigate the impact of hallucination nature of LLMs, many studies propose detecting hallucinated generation through uncertainty estimation. However, these approaches predominantly operate at the sentence or paragraph level, failing to pinpoint specific spans or entities responsible for hallucinated content. This lack of granularity is especially problematic for long-form outputs that mix accurate and fabricated information. To address this limitation, we explore entity-level hallucination detection. We propose a new data set, HalluEntity, which annotates hallucination at the entity level. Based on the dataset, we comprehensively evaluate uncertainty-based hallucination detection approaches across 17 modern LLMs. Our experimental results show that uncertainty estimation approaches focusing on individual token probabilities tend to over-predict hallucinations, while context-aware methods show better but still suboptimal performance. Through an in-depth qualitative study, we identify relationships between hallucination tendencies and linguistic properties and highlight important directions for future research. HalluEntity: https://huggingface.co/datasets/samuelyeh/HalluEntity
>
---
#### [replaced 016] On the Role of Speech Data in Reducing Toxicity Detection Bias
- **分类: cs.CL; cs.AI; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2411.08135v2](http://arxiv.org/pdf/2411.08135v2)**

> **作者:** Samuel J. Bell; Mariano Coria Meglioli; Megan Richards; Eduardo Sánchez; Christophe Ropers; Skyler Wang; Adina Williams; Levent Sagun; Marta R. Costa-jussà
>
> **备注:** Accepted at NAACL 2025
>
> **摘要:** Text toxicity detection systems exhibit significant biases, producing disproportionate rates of false positives on samples mentioning demographic groups. But what about toxicity detection in speech? To investigate the extent to which text-based biases are mitigated by speech-based systems, we produce a set of high-quality group annotations for the multilingual MuTox dataset, and then leverage these annotations to systematically compare speech- and text-based toxicity classifiers. Our findings indicate that access to speech data during inference supports reduced bias against group mentions, particularly for ambiguous and disagreement-inducing samples. Our results also suggest that improving classifiers, rather than transcription pipelines, is more helpful for reducing group bias. We publicly release our annotations and provide recommendations for future toxicity dataset construction.
>
---
#### [replaced 017] Hallucination, Monofacts, and Miscalibration: An Empirical Investigation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.08666v2](http://arxiv.org/pdf/2502.08666v2)**

> **作者:** Miranda Muqing Miao; Michael Kearns
>
> **备注:** Code available at https://github.com/mmiao2/Hallucination.git
>
> **摘要:** Hallucinated facts in large language models (LLMs) have recently been shown to obey a statistical lower bound determined by the monofact rate (related to the classical Good-Turing missing mass estimator) minus model miscalibration (Kalai & Vempala, 2024). We present the first empirical investigation of this three-way relationship in classical n-gram models and fine-tuned encoder-decoder Transformers. By generating training data from Pareto distributions with varying shape parameters, we systematically control the monofact rates and establish its positive relationship with hallucination. To bridge theory and practice, we derive an empirical analog of the hallucination bound by replacing the population miscalibration term (Section 2.1) with an empirical bin-wise KL divergence and confirm its practical viability. We then introduce selective upweighting -- a simple yet effective technique that strategically repeats as little as 5% of training examples -- to deliberately inject miscalibration into the model. This intervention reduces hallucination by up to 40%, challenging universal deduplication policies. Our experiments reveal a critical trade-off: selective upweighting maintains pre-injection levels of accuracy while substantially reducing hallucination, whereas standard training gradually improves accuracy but fails to address persistently high hallucination, indicating an inherent tension in optimization objectives.
>
---
#### [replaced 018] Investigating Language Preference of Multilingual RAG Systems
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11175v2](http://arxiv.org/pdf/2502.11175v2)**

> **作者:** Jeonghyun Park; Hwanhee Lee
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Multilingual Retrieval-Augmented Generation (mRAG) systems enhance language models by integrating external multilingual information to produce context-aware responses. However, mRAG systems struggle with retrieving relevant information due to linguistic variations between queries and documents, generating inconsistent responses when multilingual sources conflict. In this work, we systematically investigate language preferences in both retrieval and generation of mRAG through a series of experiments. Our analysis indicates that retrievers tend to prefer high-resource and query languages, yet this preference does not consistently improve generation performance. Moreover, we observe that generators prefer the query language or Latin scripts, leading to inconsistent outputs. To overcome these issues, we propose Dual Knowledge Multilingual RAG (DKM-RAG), a simple yet effective framework that fuses translated multilingual passages with complementary model knowledge. Empirical results demonstrate that DKM-RAG mitigates language preference in generation and enhances performance across diverse linguistic settings.
>
---
#### [replaced 019] DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.14609v2](http://arxiv.org/pdf/2410.14609v2)**

> **作者:** Simon Lupart; Mohammad Aliannejadi; Evangelos Kanoulas
>
> **备注:** 11 pages, 6 figures. SIGIR '25 Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval July 13--18, 2025 Padua, Italy
>
> **摘要:** Conversational Search (CS) involves retrieving relevant documents from a corpus while considering the conversational context, integrating retrieval with context modeling. Recent advancements in Large Language Models (LLMs) have significantly enhanced CS by enabling query rewriting based on conversational context. However, employing LLMs during inference poses efficiency challenges. Existing solutions mitigate this issue by distilling embeddings derived from human-rewritten queries, focusing primarily on learning the context modeling task. These methods, however, often separate the contrastive retrieval task from the distillation process, treating it as an independent loss term. To overcome these limitations, we introduce DiSCo (Distillation of Sparse Conversational retrieval), a novel approach that unifies retrieval and context modeling through a relaxed distillation objective. Instead of relying exclusively on representation learning, our method distills similarity scores between conversations and documents, providing more freedom in the representation space and better leveraging the contrastive nature of document relevance. Extensive experiments on Learned Sparse Retrieval (LSR) across five CS datasets demonstrate that DiSCo achieves substantial improvements in both in-domain and out-of-domain retrieval tasks, achieving up to a six-point gain in recall for out-of-domain datasets over state-of-the-art methods. Additionally, DiSCo employs a multi-teacher distillation strategy, using multiple LLMs as teachers, further enhancing performance and surpassing the individual teachers in in-domain settings. Furthermore, analysis of model sparsity reveals that DiSCo allows for more effective control over the sparsity of the trained models.
>
---
#### [replaced 020] Training of Scaffolded Language Models with Language Supervision: A Survey
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.16392v2](http://arxiv.org/pdf/2410.16392v2)**

> **作者:** Matthieu Lin; Jenny Sheng; Andrew Zhao; Shenzhi Wang; Yang Yue; Victor Shea Jay Huang; Huan Liu; Jun Liu; Gao Huang; Yong-Jin Liu
>
> **摘要:** This survey organizes the intricate literature on the design and optimization of emerging structures around post-trained LMs. We refer to this overarching structure as scaffolded LMs and focus on LMs that are integrated into multi-step processes with tools. We view scaffolded LMs as semi-parametric models wherein we train non-parametric variables, including the prompt, tools, and scaffold's code. In particular, they interpret instructions, use tools, and receive feedback all in language. Recent works use an LM as an optimizer to interpret language supervision and update non-parametric variables according to intricate objectives. In this survey, we refer to this paradigm as training of scaffolded LMs with language supervision. A key feature of non-parametric training is the ability to learn from language. Parametric training excels in learning from demonstration (supervised learning), exploration (reinforcement learning), or observations (unsupervised learning), using well-defined loss functions. Language-based optimization enables rich, interpretable, and expressive objectives, while mitigating issues like catastrophic forgetting and supporting compatibility with closed-source models. Furthermore, agents are increasingly deployed as co-workers in real-world applications such as Copilot in Office tools or software development. In these mixed-autonomy settings, where control and decision-making are shared between human and AI, users point out errors or suggest corrections. Accordingly, we discuss agents that continuously improve by learning from this real-time, language-based feedback and refer to this setting as streaming learning from language supervision.
>
---
#### [replaced 021] Linear Attention Sequence Parallelism
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.02882v3](http://arxiv.org/pdf/2404.02882v3)**

> **作者:** Weigao Sun; Zhen Qin; Dong Li; Xuyang Shen; Yu Qiao; Yiran Zhong
>
> **备注:** Accepted by TMLR, 23 pages
>
> **摘要:** Sequence parallelism (SP) serves as a prevalent strategy to handle long sequences that exceed the memory limit of a single device. However, for linear sequence modeling methods like linear attention, existing SP approaches do not take advantage of their right-product-first feature, resulting in sub-optimal communication efficiency and usability. In this paper, we introduce Linear Attention Sequence Parallelism (LASP), an efficient SP approach designed for linear attention-based transformer models. Specifically, we design an efficient point-to-point ring-style communication mechanism to leverage the right-product kernel trick of linear attention, which sharply decreases the communication overhead, comparing with existing SP methods. We enhance the computation efficiency of LASP by performing kernel fusion and intermediate state caching, making the implementation of LASP hardware-friendly on GPUs. Furthermore, we meticulously ensure the compatibility of sequence-level LASP with all types of batch-level data parallel methods, which is vital for distributed training on large clusters with very-long sequences. We also discuss the generalization of LASP on other linear sequence modeling methods. Extensive experiments on linear attention-based models are conducted with varying sequence lengths from 2K to 4096K. LASP scales sequence length up to 4096K on 128 GPUs, which is 8$\times$ longer than existing SP methods. Code is available at: https://github.com/OpenNLPLab/LASP.
>
---
#### [replaced 022] iAgent: LLM Agent as a Shield between User and Recommender Systems
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.14662v2](http://arxiv.org/pdf/2502.14662v2)**

> **作者:** Wujiang Xu; Yunxiao Shi; Zujie Liang; Xuying Ning; Kai Mei; Kun Wang; Xi Zhu; Min Xu; Yongfeng Zhang
>
> **备注:** Findings of ACL 2025 and WWW2025@HCRS
>
> **摘要:** Traditional recommender systems usually take the user-platform paradigm, where users are directly exposed under the control of the platform's recommendation algorithms. However, the defect of recommendation algorithms may put users in very vulnerable positions under this paradigm. First, many sophisticated models are often designed with commercial objectives in mind, focusing on the platform's benefits, which may hinder their ability to protect and capture users' true interests. Second, these models are typically optimized using data from all users, which may overlook individual user's preferences. Due to these shortcomings, users may experience several disadvantages under the traditional user-platform direct exposure paradigm, such as lack of control over the recommender system, potential manipulation by the platform, echo chamber effects, or lack of personalization for less active users due to the dominance of active users during collaborative learning. Therefore, there is an urgent need to develop a new paradigm to protect user interests and alleviate these issues. Recently, some researchers have introduced LLM agents to simulate user behaviors, these approaches primarily aim to optimize platform-side performance, leaving core issues in recommender systems unresolved. To address these limitations, we propose a new user-agent-platform paradigm, where agent serves as the protective shield between user and recommender system that enables indirect exposure.
>
---
#### [replaced 023] Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines
- **分类: cs.CL; cs.AI; cs.GT; cs.IR; econ.TH**

- **链接: [http://arxiv.org/pdf/2501.00745v2](http://arxiv.org/pdf/2501.00745v2)**

> **作者:** Xiyang Hu
>
> **摘要:** The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design.
>
---
#### [replaced 024] Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.10316v3](http://arxiv.org/pdf/2501.10316v3)**

> **作者:** Suvodip Dey; Yi-Jyun Sun; Gokhan Tur; Dilek Hakkani-Tur
>
> **备注:** Accepted at ACL 2025 Main Conference
>
> **摘要:** Recent LLMs have enabled significant advancements for conversational agents. However, they are also well known to hallucinate, producing responses that seem plausible but are factually incorrect. On the other hand, users tend to over-rely on LLM-based AI agents, accepting AI's suggestion even when it is wrong. Adding positive friction, such as explanations or getting user confirmations, has been proposed as a mitigation in AI-supported decision-making systems. In this paper, we propose an accountability model for LLM-based task-oriented dialogue agents to address user overreliance via friction turns in cases of model uncertainty and errors associated with dialogue state tracking (DST). The accountability model is an augmented LLM with an additional accountability head that functions as a binary classifier to predict the relevant slots of the dialogue state mentioned in the conversation. We perform our experiments with multiple backbone LLMs on two established benchmarks (MultiWOZ and Snips). Our empirical findings demonstrate that the proposed approach not only enables reliable estimation of AI agent errors but also guides the decoder in generating more accurate actions. We observe around 3% absolute improvement in joint goal accuracy (JGA) of DST output by incorporating accountability heads into modern LLMs. Self-correcting the detected errors further increases the JGA from 67.13 to 70.51, achieving state-of-the-art DST performance. Finally, we show that error correction through user confirmations (friction turn) achieves a similar performance gain, highlighting its potential to reduce user overreliance.
>
---
#### [replaced 025] Evaluating Vision-Language Models as Evaluators in Path Planning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.18711v4](http://arxiv.org/pdf/2411.18711v4)**

> **作者:** Mohamed Aghzal; Xiang Yue; Erion Plaku; Ziyu Yao
>
> **备注:** Accepted to the 2025 IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)
>
> **摘要:** Despite their promise to perform complex reasoning, large language models (LLMs) have been shown to have limited effectiveness in end-to-end planning. This has inspired an intriguing question: if these models cannot plan well, can they still contribute to the planning framework as a helpful plan evaluator? In this work, we generalize this question to consider LLMs augmented with visual understanding, i.e., Vision-Language Models (VLMs). We introduce PathEval, a novel benchmark evaluating VLMs as plan evaluators in complex path-planning scenarios. Succeeding in the benchmark requires a VLM to be able to abstract traits of optimal paths from the scenario description, demonstrate precise low-level perception on each path, and integrate this information to decide the better path. Our analysis of state-of-the-art VLMs reveals that these models face significant challenges on the benchmark. We observe that the VLMs can precisely abstract given scenarios to identify the desired traits and exhibit mixed performance in integrating the provided information. Yet, their vision component presents a critical bottleneck, with models struggling to perceive low-level details about a path. Our experimental results show that this issue cannot be trivially addressed via end-to-end fine-tuning; rather, task-specific discriminative adaptation of these vision encoders is needed for these VLMs to become effective path evaluators.
>
---
#### [replaced 026] Towards Multi-Agent Reasoning Systems for Collaborative Expertise Delegation: An Exploratory Design Study
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07313v2](http://arxiv.org/pdf/2505.07313v2)**

> **作者:** Baixuan Xu; Chunyang Li; Weiqi Wang; Wei Fan; Tianshi Zheng; Haochen Shi; Tao Fan; Yangqiu Song; Qiang Yang
>
> **备注:** 19 pages
>
> **摘要:** Designing effective collaboration structure for multi-agent LLM systems to enhance collective reasoning is crucial yet remains under-explored. In this paper, we systematically investigate how collaborative reasoning performance is affected by three key design dimensions: (1) Expertise-Domain Alignment, (2) Collaboration Paradigm (structured workflow vs. diversity-driven integration), and (3) System Scale. Our findings reveal that expertise alignment benefits are highly domain-contingent, proving most effective for contextual reasoning tasks. Furthermore, collaboration focused on integrating diverse knowledge consistently outperforms rigid task decomposition. Finally, we empirically explore the impact of scaling the multi-agent system with expertise specialization and study the computational trade off, highlighting the need for more efficient communication protocol design. This work provides concrete guidelines for configuring specialized multi-agent system and identifies critical architectural trade-offs and bottlenecks for scalable multi-agent reasoning. The code will be made available upon acceptance.
>
---
#### [replaced 027] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10354v2](http://arxiv.org/pdf/2505.10354v2)**

> **作者:** Yile Wang; Zhanyu Shen; Hui Huang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Semantic text representation is a fundamental task in the field of natural language processing. Existing text embedding (e.g., SimCSE and LLM2Vec) have demonstrated excellent performance, but the values of each dimension are difficult to trace and interpret. Bag-of-words, as classic sparse interpretable embeddings, suffers from poor performance. Recently, Benara et al. (2024) propose interpretable text embeddings using large language models, which forms "0/1" embeddings based on responses to a series of questions. These interpretable text embeddings are typically high-dimensional (larger than 10,000). In this work, we propose Low-dimensional (lower than 500) Dense and Interpretable text embeddings with Relative representations (LDIR). The numerical values of its dimensions indicate semantic relatedness to different anchor texts through farthest point sampling, offering both semantic representation as well as a certain level of traceability and interpretability. We validate LDIR on multiple semantic textual similarity, retrieval, and clustering tasks. Extensive experimental results show that LDIR performs close to the black-box baseline models and outperforms the interpretable embeddings baselines with much fewer dimensions. Code is available at https://github.com/szu-tera/LDIR.
>
---
#### [replaced 028] Do we really have to filter out random noise in pre-training data for language models?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06604v2](http://arxiv.org/pdf/2502.06604v2)**

> **作者:** Jinghan Ru; Yuxin Xie; Xianwei Zhuang; Yuguo Yin; Zhihui Guo; Zhiming Liu; Qianli Ren; Yuexian Zou
>
> **摘要:** Web-scale pre-training datasets are the cornerstone of LLMs' success. However, text data curated from the Internet inevitably contains random noise caused by decoding errors or unregulated web content. In contrast to previous works that focus on low quality or synthetic data, our study \textbf{provides the first systematic investigation of such random noise through a cohesive ``What-Why-How'' framework.} Surprisingly, we observed that the resulting increase in the loss of next-token prediction (NTP) was significantly lower than the proportion of random noise even when the model was scaled up to 2.7B. We provide a theoretical justification for this phenomenon, which also elucidates the success of multilingual models and can be applied to multimodal models. On the other hand, experiments show that the model's performance in downstream tasks is not based solely on the NTP loss, which means that random noise may result in degraded downstream performance. To address the potential adverse effects, we introduce a novel plug-and-play Local Gradient Matching loss, which explicitly enhances the denoising capability of the downstream task head by aligning the gradient of normal and perturbed features without requiring knowledge of the model's parameters. Additional experiments on 8 language and 14 vision benchmarks further validate its effectiveness.
>
---
#### [replaced 029] KVShare: An LLM Service System with Efficient and Effective Multi-Tenant KV Cache Reuse
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.16525v2](http://arxiv.org/pdf/2503.16525v2)**

> **作者:** Huan Yang; Renji Zhang; Mingzhe Huang; Weijun Wang; Yin Tang; Yuanchun Li; Yunxin Liu; Deyu Zhang
>
> **摘要:** Recent advances in long-text understanding have pushed the context length of large language models (LLMs) up to one million tokens. It boosts LLMs's accuracy and reasoning capacity but causes exorbitant computational costs and unsatisfactory Time to First Token (TTFT). KV cache reuse, which reuses the exact same KV cache of prefixes and templates or shares similar ones but with extra selective recomputation, offers a promising way to tackle this issue. However, prior studies overlook the cross-request KV reuse and the attention deviations introduced by new tokens during the decoding stage. In this paper, we present a KV cache management module that shares the KV cache across requests under multi-tenant scenarios without sacrificing model accuracy. Our system, KVShare, enables accurate and efficient LLM serving by 1) a Dual-Stage High Deviation algorithm (DHD) that conditionally selects a small portion of KV cache to be recomputed during both prefill and decode phases, and 2) a cache-aware scheduler that prioritizes requests based on their KV cache hit rates and orchestrates continuous batching to achieve enhanced system efficiency and faster TTFT. Multi-task experiments conducted on models such as Qwen2.5-7B,Llama3.1-8B and Yi1.5-9B demonstrate that KVShare reduces TTFT by up to 9.39x and increases 1.2x of the throughput compared to the full KV recompute. Moreover, KVShare achieves 20.38% boost in terms of accuracy compared to SOTA methods.
>
---
#### [replaced 030] AI Idea Bench 2025: AI Research Idea Generation Benchmark
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14191v2](http://arxiv.org/pdf/2504.14191v2)**

> **作者:** Yansheng Qiu; Haoquan Zhang; Zhaopan Xu; Ming Li; Diping Song; Zheng Wang; Kaipeng Zhang
>
> **摘要:** Large-scale Language Models (LLMs) have revolutionized human-AI interaction and achieved significant success in the generation of novel ideas. However, current assessments of idea generation overlook crucial factors such as knowledge leakage in LLMs, the absence of open-ended benchmarks with grounded truth, and the limited scope of feasibility analysis constrained by prompt design. These limitations hinder the potential of uncovering groundbreaking research ideas. In this paper, we present AI Idea Bench 2025, a framework designed to quantitatively evaluate and compare the ideas generated by LLMs within the domain of AI research from diverse perspectives. The framework comprises a comprehensive dataset of 3,495 AI papers and their associated inspired works, along with a robust evaluation methodology. This evaluation system gauges idea quality in two dimensions: alignment with the ground-truth content of the original papers and judgment based on general reference material. AI Idea Bench 2025's benchmarking system stands to be an invaluable resource for assessing and comparing idea-generation techniques, thereby facilitating the automation of scientific discovery.
>
---
#### [replaced 031] MIR-Bench: Can Your LLM Recognize Complicated Patterns via Many-Shot In-Context Reasoning?
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09933v4](http://arxiv.org/pdf/2502.09933v4)**

> **作者:** Kai Yan; Zhan Ling; Kang Liu; Yifan Yang; Ting-Han Fan; Lingfeng Shen; Zhengyin Du; Jiecao Chen
>
> **备注:** 36 pages, 11 figures. The last version adds more experiments and modifies name for better summary of the work
>
> **摘要:** The ability to recognize patterns from examples and apply them to new ones is a primal ability for general intelligence, and is widely studied by psychology and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually <10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations often focus on classification, and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context reasoning benchmark for pattern recognition that asks LLM to predict output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for many-shot in-context reasoning, and acquired many insightful findings including scaling effect, robustness, inductive vs. transductive reasoning, retrieval Augmented Generation (RAG), coding for inductive reasoning, cross-domain generalizability, etc.
>
---
#### [replaced 032] Towards understanding evolution of science through language model series
- **分类: cs.CL; cs.CY; cs.DL**

- **链接: [http://arxiv.org/pdf/2409.09636v2](http://arxiv.org/pdf/2409.09636v2)**

> **作者:** Junjie Dong; Zhuoqi Lyu; Qing Ke
>
> **摘要:** We introduce AnnualBERT, a series of language models designed specifically to capture the temporal evolution of scientific text. Deviating from the prevailing paradigms of subword tokenizations and "one model to rule them all", AnnualBERT adopts whole words as tokens and is composed of a base RoBERTa model pretrained from scratch on the full-text of 1.7 million arXiv papers published until 2008 and a collection of progressively trained models on arXiv papers at an annual basis. We demonstrate the effectiveness of AnnualBERT models by showing that they not only have comparable performances in standard tasks but also achieve state-of-the-art performances on domain-specific NLP tasks as well as link prediction tasks in the arXiv citation network. We then utilize probing tasks to quantify the models' behavior in terms of representation learning and forgetting as time progresses. Our approach enables the pretrained models to not only improve performances on scientific text processing tasks but also to provide insights into the development of scientific discourse over time. The series of the models is available at https://huggingface.co/jd445/AnnualBERTs.
>
---
#### [replaced 033] Intervention-Aware Forecasting: Breaking Historical Limits from a System Perspective
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.13522v3](http://arxiv.org/pdf/2405.13522v3)**

> **作者:** Zhijian Xu; Hao Wang; Qiang Xu
>
> **摘要:** Traditional time series forecasting methods predominantly rely on historical data patterns, neglecting external interventions that significantly shape future dynamics. Through control-theoretic analysis, we show that the implicit "self-stimulation" assumption limits the accuracy of these forecasts. To overcome this limitation, we propose an Intervention-Aware Time Series Forecasting (IATSF) framework explicitly designed to incorporate external interventions. We particularly emphasize textual interventions due to their unique capability to represent qualitative or uncertain influences inadequately captured by conventional exogenous variables. We propose a leak-free benchmark composed of temporally synchronized textual intervention data across synthetic and real-world scenarios. To rigorously evaluate IATSF, we develop FIATS, a lightweight forecasting model that integrates textual interventions through Channel-Aware Adaptive Sensitivity Modeling (CASM) and Channel-Aware Parameter Sharing (CAPS) mechanisms, enabling the model to adjust its sensitivity to interventions and historical data in a channel-specific manner. Extensive empirical evaluations confirm that FIATS surpasses state-of-the-art methods, highlighting that forecasting improvements stem explicitly from modeling external interventions rather than increased model complexity alone.
>
---
#### [replaced 034] TigerLLM -- A Family of Bangla Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.10995v2](http://arxiv.org/pdf/2503.10995v2)**

> **作者:** Nishat Raihan; Marcos Zampieri
>
> **摘要:** The development of Large Language Models (LLMs) remains heavily skewed towards English and a few other high-resource languages. This linguistic disparity is particularly evident for Bangla - the 5th most spoken language. A few initiatives attempted to create open-source Bangla LLMs with performance still behind high-resource languages and limited reproducibility. To address this gap, we introduce TigerLLM - a family of Bangla LLMs. Our results demonstrate that these models surpass all open-source alternatives and also outperform larger proprietary models like GPT3.5 across standard benchmarks, establishing TigerLLM as the new baseline for future Bangla language modeling.
>
---
#### [replaced 035] What External Knowledge is Preferred by LLMs? Characterizing and Exploring Chain of Evidence in Imperfect Context
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.12632v2](http://arxiv.org/pdf/2412.12632v2)**

> **作者:** Zhiyuan Chang; Mingyang Li; Xiaojun Jia; Junjie Wang; Yuekai Huang; Qing Wang; Yihao Huang; Yang Liu
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Incorporating external knowledge into large language models (LLMs) has emerged as a promising approach to mitigate outdated knowledge and hallucination in LLMs. However, external knowledge is often imperfect. In addition to useful knowledge, external knowledge is rich in irrelevant or misinformation in the context that can impair the reliability of LLM responses. This paper focuses on LLMs' preferred external knowledge in imperfect contexts when handling multi-hop QA. Inspired by criminal procedural law's Chain of Evidence (CoE), we characterize that knowledge preferred by LLMs should maintain both relevance to the question and mutual support among knowledge pieces. Accordingly, we propose an automated CoE discrimination approach and evaluate LLMs' effectiveness, faithfulness and robustness with CoE, including its application in the Retrieval-Augmented Generation (RAG). Tests on five LLMs show CoE improves generation accuracy, answer faithfulness, robustness to knowledge conflicts, and boosts the performance of existing approaches in three practical RAG scenarios.
>
---
#### [replaced 036] Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods
- **分类: stat.ML; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01384v2](http://arxiv.org/pdf/2502.01384v2)**

> **作者:** Oussama Zekri; Nicolas Boullé
>
> **备注:** 30 pages, 8 figures, 8 tables
>
> **摘要:** Discrete diffusion models have recently gained significant attention due to their ability to process complex discrete structures for language modeling. However, fine-tuning these models with policy gradient methods, as is commonly done in Reinforcement Learning from Human Feedback (RLHF), remains a challenging task. We propose an efficient, broadly applicable, and theoretically justified policy gradient algorithm, called Score Entropy Policy Optimization (SEPO), for fine-tuning discrete diffusion models over non-differentiable rewards. Our numerical experiments across several discrete generative tasks demonstrate the scalability and efficiency of our method. Our code is available at https://github.com/ozekri/SEPO.
>
---
#### [replaced 037] ViTextVQA: A Large-Scale Visual Question Answering Dataset for Evaluating Vietnamese Text Comprehension in Images
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.10652v3](http://arxiv.org/pdf/2404.10652v3)**

> **作者:** Quan Van Nguyen; Dan Quang Tran; Huy Quang Pham; Thang Kien-Bao Nguyen; Nghia Hieu Nguyen; Kiet Van Nguyen; Ngan Luu-Thuy Nguyen
>
> **摘要:** Visual Question Answerinng (VQA) is a complicated task that requires the capability of simultaneously processing natural language and images. This task was initially researched with a focus on developing methods to help machines understand objects and scene contexts in images. However, some scene text that carries explicit information about the full content of the image is not mentioned. Along with the continuous development of the AI era, there have been many studies on the reading comprehension ability of VQA models in the world. Therefore, we introduce the first large-scale dataset in Vietnamese specializing in the ability to understand scene text, we call it ViTextVQA (\textbf{Vi}etnamese \textbf{Text}-based \textbf{V}isual \textbf{Q}uestion \textbf{A}nswering dataset) which contains \textbf{over 16,000} images and \textbf{over 50,000} questions with answers. To tackle this task efficiently, we propose ViTextBLIP-2, an novel multimodal feature fusion Method, which optimizes Vietnamese OCR-based VQA by integrating a frozen Vision Transformer, SwinTextSpotter OCR, and ViT5 LLM with a trainable Q-Former for multimodal feature fusion. Through experiments with various state-of-the-art models, we uncover the significance of the order in which tokens in OCR text are processed and selected to formulate answers. This finding helped us significantly improve the performance of the baseline models on the ViTextVQA dataset. Our dataset is available (https://github.com/minhquan6203/ViTextVQA-Dataset) for research purposes.
>
---
#### [replaced 038] An AI-Powered Research Assistant in the Lab: A Practical Guide for Text Analysis Through Iterative Collaboration with LLMs
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.09724v2](http://arxiv.org/pdf/2505.09724v2)**

> **作者:** Gino Carmona-Díaz; William Jiménez-Leal; María Alejandra Grisales; Chandra Sripada; Santiago Amaya; Michael Inzlicht; Juan Pablo Bermúdez
>
> **备注:** 31 pages, 1 figure
>
> **摘要:** Analyzing texts such as open-ended responses, headlines, or social media posts is a time- and labor-intensive process highly susceptible to bias. LLMs are promising tools for text analysis, using either a predefined (top-down) or a data-driven (bottom-up) taxonomy, without sacrificing quality. Here we present a step-by-step tutorial to efficiently develop, test, and apply taxonomies for analyzing unstructured data through an iterative and collaborative process between researchers and LLMs. Using personal goals provided by participants as an example, we demonstrate how to write prompts to review datasets and generate a taxonomy of life domains, evaluate and refine the taxonomy through prompt and direct modifications, test the taxonomy and assess intercoder agreements, and apply the taxonomy to categorize an entire dataset with high intercoder reliability. We discuss the possibilities and limitations of using LLMs for text analysis.
>
---
#### [replaced 039] PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.09921v2](http://arxiv.org/pdf/2505.09921v2)**

> **作者:** Yidan Wang; Yanan Cao; Yubing Ren; Fang Fang; Zheng Lin; Binxing Fang
>
> **备注:** Accepted to ACL 2025 (main)
>
> **摘要:** Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at https://github.com/redwyd/PrivacyJailbreak.
>
---
#### [replaced 040] ZeroSearch: Incentivize the Search Capability of LLMs without Searching
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.04588v2](http://arxiv.org/pdf/2505.04588v2)**

> **作者:** Hao Sun; Zile Qiao; Jiayan Guo; Xuanbo Fan; Yingyan Hou; Yong Jiang; Pengjun Xie; Yan Zhang; Fei Huang; Jingren Zhou
>
> **摘要:** Effective information searching is essential for enhancing the reasoning and generation capabilities of large language models (LLMs). Recent research has explored using reinforcement learning (RL) to improve LLMs' search capabilities by interacting with live search engines in real-world environments. While these approaches show promising results, they face two major challenges: (1) Uncontrolled Document Quality: The quality of documents returned by search engines is often unpredictable, introducing noise and instability into the training process. (2) Prohibitively High API Costs: RL training requires frequent rollouts, potentially involving hundreds of thousands of search requests, which incur substantial API expenses and severely constrain scalability. To address these challenges, we introduce ZeroSearch, a novel RL framework that incentivizes the capabilities of LLMs to use a real search engine with simulated searches during training. Our approach begins with lightweight supervised fine-tuning to transform the LLM into a retrieval module capable of generating both useful and noisy documents in response to a query. During RL training, we employ a curriculum-based rollout strategy that incrementally degrades the quality of generated documents, progressively eliciting the model's reasoning ability by exposing it to increasingly challenging retrieval scenarios. Extensive experiments demonstrate that ZeroSearch effectively incentivizes the search capabilities of LLMs using a 3B LLM as the retrieval module. Remarkably, a 7B retrieval module achieves comparable performance to the real search engine, while a 14B retrieval module even surpasses it. Furthermore, it generalizes well across both base and instruction-tuned models of various parameter sizes and is compatible with a wide range of RL algorithms.
>
---
#### [replaced 041] Parameterized Synthetic Text Generation with SimpleStories
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.09184v2](http://arxiv.org/pdf/2504.09184v2)**

> **作者:** Lennart Finke; Chandan Sreedhara; Thomas Dooms; Mat Allen; Emerald Zhang; Juan Diego Rodriguez; Noa Nabeshima; Thomas Marshall; Dan Braun
>
> **摘要:** We present SimpleStories, a large synthetic story dataset in simple language, consisting of 2 million samples each in English and Japanese. Through parameterizing prompts at multiple levels of abstraction, we achieve control over story characteristics at scale, inducing syntactic and semantic diversity. Ablations on a newly trained model suite show improved sample efficiency and model interpretability compared to the TinyStories dataset. We open-source all constituent parts of model creation, hoping to enable novel ways to study the end-to-end training process. As a byproduct, we move the frontier regarding the fewest-parameter language model that outputs grammatical natural language.
>
---
#### [replaced 042] Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms
- **分类: cs.CL; cs.AI; cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2501.13977v2](http://arxiv.org/pdf/2501.13977v2)**

> **作者:** Rajvardhan Oak; Muhammad Haroon; Claire Jo; Magdalena Wojcieszak; Anshuman Chhabra
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Social media platforms utilize Machine Learning (ML) and Artificial Intelligence (AI) powered recommendation algorithms to maximize user engagement, which can result in inadvertent exposure to harmful content. Current moderation efforts, reliant on classifiers trained with extensive human-annotated data, struggle with scalability and adapting to new forms of harm. To address these challenges, we propose a novel re-ranking approach using Large Language Models (LLMs) in zero-shot and few-shot settings. Our method dynamically assesses and re-ranks content sequences, effectively mitigating harmful content exposure without requiring extensive labeled data. Alongside traditional ranking metrics, we also introduce two new metrics to evaluate the effectiveness of re-ranking in reducing exposure to harmful content. Through experiments on three datasets, three models and across three configurations, we demonstrate that our LLM-based approach significantly outperforms existing proprietary moderation approaches, offering a scalable and adaptable solution for harm mitigation.
>
---
#### [replaced 043] Fourier Transformer: Fast Long Range Modeling by Removing Sequence Redundancy with FFT Operator
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2305.15099v2](http://arxiv.org/pdf/2305.15099v2)**

> **作者:** Ziwei He; Meng Yang; Minwei Feng; Jingcheng Yin; Xinbing Wang; Jingwen Leng; Zhouhan Lin
>
> **摘要:** The transformer model is known to be computationally demanding, and prohibitively costly for long sequences, as the self-attention module uses a quadratic time and space complexity with respect to sequence length. Many researchers have focused on designing new forms of self-attention or introducing new parameters to overcome this limitation, however a large portion of them prohibits the model to inherit weights from large pretrained models. In this work, the transformer's inefficiency has been taken care of from another perspective. We propose Fourier Transformer, a simple yet effective approach by progressively removing redundancies in hidden sequence using the ready-made Fast Fourier Transform (FFT) operator to perform Discrete Cosine Transformation (DCT). Fourier Transformer is able to significantly reduce computational costs while retain the ability to inherit from various large pretrained models. Experiments show that our model achieves state-of-the-art performances among all transformer-based models on the long-range modeling benchmark LRA with significant improvement in both speed and space. For generative seq-to-seq tasks including CNN/DailyMail and ELI5, by inheriting the BART weights our model outperforms the standard BART and other efficient models. Our code is publicly available at https://github.com/LUMIA-Group/FourierTransformer
>
---
#### [replaced 044] Insertion Language Models: Sequence Generation with Arbitrary-Position Insertions
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05755v2](http://arxiv.org/pdf/2505.05755v2)**

> **作者:** Dhruvesh Patel; Aishwarya Sahoo; Avinash Amballa; Tahira Naseem; Tim G. J. Rudner; Andrew McCallum
>
> **备注:** Corrected a typo in author names
>
> **摘要:** Autoregressive models (ARMs), which predict subsequent tokens one-by-one ``from left to right,'' have achieved significant success across a wide range of sequence generation tasks. However, they struggle to accurately represent sequences that require satisfying sophisticated constraints or whose sequential dependencies are better addressed by out-of-order generation. Masked Diffusion Models (MDMs) address some of these limitations, but the process of unmasking multiple tokens simultaneously in MDMs can introduce incoherences, and MDMs cannot handle arbitrary infilling constraints when the number of tokens to be filled in is not known in advance. In this work, we introduce Insertion Language Models (ILMs), which learn to insert tokens at arbitrary positions in a sequence -- that is, they select jointly both the position and the vocabulary element to be inserted. By inserting tokens one at a time, ILMs can represent strong dependencies between tokens, and their ability to generate sequences in arbitrary order allows them to accurately model sequences where token dependencies do not follow a left-to-right sequential structure. To train ILMs, we propose a tailored network parameterization and use a simple denoising objective. Our empirical evaluation demonstrates that ILMs outperform both ARMs and MDMs on common planning tasks. Furthermore, we show that ILMs outperform MDMs and perform on par with ARMs in an unconditional text generation task while offering greater flexibility than MDMs in arbitrary-length text infilling.
>
---
#### [replaced 045] TreeKV: Smooth Key-Value Cache Compression with Tree Structures
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.04987v3](http://arxiv.org/pdf/2501.04987v3)**

> **作者:** Ziwei He; Jian Yuan; Haoli Bai; Jingwen Leng; Bo Jiang
>
> **摘要:** Efficient key-value (KV) cache compression is critical for scaling transformer-based Large Language Models (LLMs) in long sequences and resource-limited settings. Existing methods evict tokens based on their positions or importance scores, but position-based strategies can miss crucial information outside predefined regions, while those relying on global importance scores resulting in strong regional biases, limiting the KV cache's overall context retention and potentially impairing the performance of LLMs on complex tasks. Our wavelet analysis reveals that as tokens approach the end of sequence, their contributions to generation gradually increase and tends to diverge more from neighboring tokens, indicating a smooth transition with increasing complexity and variability from distant to nearby context. Motivated by this observation, we propose TreeKV, an intuitive, training-free method that employs a tree structure for smooth cache compression. TreeKV maintains a fixed cache size, allowing LLMs to deliver high-quality output even in long text scenarios. Unlike most compression methods, TreeKV is applicable to both the generation and prefilling stages. TreeKV consistently surpasses all baseline models in language modeling tasks on PG19 and OpenWebText2, allowing LLMs trained with short context window to generalize to longer window with a 16x cache reduction. On the Longbench benchmark, TreeKV achieves the best performance with only 6\% of the budget at optimal efficiency.
>
---
#### [replaced 046] Call for Rigor in Reporting Quality of Instruction Tuning Data
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.04807v3](http://arxiv.org/pdf/2503.04807v3)**

> **作者:** Hyeonseok Moon; Jaehyung Seo; Heuiseok Lim
>
> **备注:** Accepted to the ACL2025-main
>
> **摘要:** Instruction tuning is crucial for adapting large language models (LLMs) to align with user intentions. Numerous studies emphasize the significance of the quality of instruction tuning (IT) data, revealing a strong correlation between IT data quality and the alignment performance of LLMs. In these studies, the quality of IT data is typically assessed by evaluating the performance of LLMs trained with that data. However, we identified a prevalent issue in such practice: hyperparameters for training models are often selected arbitrarily without adequate justification. We observed significant variations in hyperparameters applied across different studies, even when training the same model with the same data. In this study, we demonstrate the potential problems arising from this practice and emphasize the need for careful consideration in verifying data quality. Through our experiments on the quality of LIMA data and a selected set of 1,000 Alpaca data points, we demonstrate that arbitrary hyperparameter decisions can make any arbitrary conclusion.
>
---
#### [replaced 047] Safety Evaluation and Enhancement of DeepSeek Models in Chinese Contexts
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2503.16529v2](http://arxiv.org/pdf/2503.16529v2)**

> **作者:** Wenjing Zhang; Xuejiao Lei; Zhaoxiang Liu; Limin Han; Jiaojiao Zhao; Junting Guo; Zhenhong Long; Shu Yang; Meijuan An; Beibei Huang; Rongjia Du; Ning Wang; Kai Wang; Shiguo Lian
>
> **备注:** 21 pages, 13 figures, 4 tables
>
> **摘要:** DeepSeek-R1, renowned for its exceptional reasoning capabilities and open-source strategy, is significantly influencing the global artificial intelligence landscape. However, it exhibits notable safety shortcomings. Recent research conducted by Robust Intelligence, a subsidiary of Cisco, in collaboration with the University of Pennsylvania, revealed that DeepSeek-R1 achieves a 100\% attack success rate when processing harmful prompts. Furthermore, multiple security firms and research institutions have identified critical security vulnerabilities within the model. Although China Unicom has uncovered safety vulnerabilities of R1 in Chinese contexts, the safety capabilities of the remaining distilled models in the R1 series have not yet been comprehensively evaluated. To address this gap, this study utilizes the comprehensive Chinese safety benchmark CHiSafetyBench to conduct an in-depth safety evaluation of the DeepSeek-R1 series distilled models. The objective is to assess the safety capabilities of these models in Chinese contexts both before and after distillation, and to further elucidate the adverse effects of distillation on model safety. Building on these findings, we implement targeted safety enhancements for the entire DeepSeek-R1 model series. Evaluation results indicate that the enhanced models achieve significant improvements in safety while maintaining reasoning capabilities without notable degradation. We open-source the safety-enhanced models at https://github.com/UnicomAI/DeepSeek-R1-Safe to serve as a valuable resource for future research and optimization of DeepSeek models.
>
---
#### [replaced 048] Mask-Enhanced Autoregressive Prediction: Pay Less Attention to Learn More
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.07490v2](http://arxiv.org/pdf/2502.07490v2)**

> **作者:** Xialie Zhuang; Zhikai Jia; Jianjin Li; Zhenyu Zhang; Li Shen; Zheng Cao; Shiwei Liu
>
> **备注:** 17 pages,7 figures
>
> **摘要:** Large Language Models (LLMs) are discovered to suffer from accurately retrieving key information. To address this, we propose Mask-Enhanced Autoregressive Prediction (MEAP), a simple yet effective training paradigm that seamlessly integrates Masked Language Modeling (MLM) into Next-Token Prediction (NTP) to enhance the latter's in-context retrieval capabilities. Specifically, MEAP first randomly masks a small fraction of input tokens and then directly performs the standard next-token prediction autoregressive using a decoder-only Transformer. MEAP eliminates the need for bidirectional attention or encoder-decoder architectures for MLM, incurring no additional computational overhead during pre-training or inference. Intensive experiments demonstrate that MEAP substantially outperforms NTP on key information retrieval and long-context reasoning tasks, while performing on par or better on commonsense reasoning tasks. The benefits of MEAP also extend to supervised fine-tuning, where it shows remarkable advantages in lost-in-the-middle scenarios, outperforming NTP by 11.77 percentage points. Our analysis indicates that MEAP's effectiveness arises from its ability to promote more distinguishable attention scores by concentrating on a reduced set of non-masked tokens. This mechanism improves the model's focus on task-relevant signals while mitigating the influence of peripheral context. These findings position MEAP as a promising training paradigm for large language models.
>
---
#### [replaced 049] FOReCAst: The Future Outcome Reasoning and Confidence Assessment Benchmark
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.19676v4](http://arxiv.org/pdf/2502.19676v4)**

> **作者:** Zhangdie Yuan; Zifeng Ding; Andreas Vlachos
>
> **摘要:** Forecasting is an important task in many domains, such as technology and economics. However existing forecasting benchmarks largely lack comprehensive confidence assessment, focus on limited question types, and often consist of artificial questions that do not align with real-world human forecasting needs. To address these gaps, we introduce FOReCAst (Future Outcome Reasoning and Confidence Assessment), a benchmark that evaluates models' ability to make predictions and their confidence in them. FOReCAst spans diverse forecasting scenarios involving Boolean questions, timeframe prediction, and quantity estimation, enabling a comprehensive evaluation of both prediction accuracy and confidence calibration for real-world applications.
>
---
#### [replaced 050] Divided by discipline? A systematic literature review on the quantification of online sexism and misogyny using a semi-automated approach
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2409.20204v2](http://arxiv.org/pdf/2409.20204v2)**

> **作者:** Aditi Dutta; Susan Banducci; Chico Q. Camargo
>
> **摘要:** Several computational tools have been developed to detect and identify sexism, misogyny, and gender-based hate speech, particularly on online platforms. These tools draw on insights from both social science and computer science. Given the increasing concern over gender-based discrimination in digital spaces, the contested definitions and measurements of sexism, and the rise of interdisciplinary efforts to understand its online manifestations, a systematic literature review is essential for capturing the current state and trajectory of this evolving field. In this review, we make four key contributions: (1) we synthesize the literature into five core themes: definitions of sexism and misogyny, disciplinary divergences, automated detection methods, associated challenges, and design-based interventions; (2) we adopt an interdisciplinary lens, bridging theoretical and methodological divides across disciplines; (3) we highlight critical gaps, including the need for intersectional approaches, the under-representation of non-Western languages and perspectives, and the limited focus on proactive design strategies beyond text classification; and (4) we offer a methodological contribution by applying a rigorous semi-automated systematic review process guided by PRISMA, establishing a replicable standard for future work in this domain. Our findings reveal a clear disciplinary divide in how sexism and misogyny are conceptualized and measured. Through an evidence-based synthesis, we examine how existing studies have attempted to bridge this gap through interdisciplinary collaboration. Drawing on both social science theories and computational modeling practices, we assess the strengths and limitations of current methodologies. Finally, we outline key challenges and future directions for advancing research on the detection and mitigation of online sexism and misogyny.
>
---
#### [replaced 051] Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions
- **分类: cs.GT; cs.AI; cs.CL; q-fin.CP**

- **链接: [http://arxiv.org/pdf/2410.00031v2](http://arxiv.org/pdf/2410.00031v2)**

> **作者:** Ryan Y. Lin; Siddhartha Ojha; Kevin Cai; Maxwell F. Chen
>
> **摘要:** Machine-learning technologies are seeing increased deployment in real-world market scenarios. In this work, we explore the strategic behaviors of large language models (LLMs) when deployed as autonomous agents in multi-commodity markets, specifically within Cournot competition frameworks. We examine whether LLMs can independently engage in anti-competitive practices such as collusion or, more specifically, market division. Our findings demonstrate that LLMs can effectively monopolize specific commodities by dynamically adjusting their pricing and resource allocation strategies, thereby maximizing profitability without direct human input or explicit collusion commands. These results pose unique challenges and opportunities for businesses looking to integrate AI into strategic roles and for regulatory bodies tasked with maintaining fair and competitive markets. The study provides a foundation for further exploration into the ramifications of deferring high-stakes decisions to LLM-based agents.
>
---
#### [replaced 052] Does Liking Yellow Imply Driving a School Bus? Semantic Leakage in Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.06518v3](http://arxiv.org/pdf/2408.06518v3)**

> **作者:** Hila Gonen; Terra Blevins; Alisa Liu; Luke Zettlemoyer; Noah A. Smith
>
> **摘要:** Despite their wide adoption, the biases and unintended behaviors of language models remain poorly understood. In this paper, we identify and characterize a phenomenon never discussed before, which we call semantic leakage, where models leak irrelevant information from the prompt into the generation in unexpected ways. We propose an evaluation setting to detect semantic leakage both by humans and automatically, curate a diverse test suite for diagnosing this behavior, and measure significant semantic leakage in 13 flagship models. We also show that models exhibit semantic leakage in languages besides English and across different settings and generation scenarios. This discovery highlights yet another type of bias in language models that affects their generation patterns and behavior.
>
---
#### [replaced 053] When to Speak, When to Abstain: Contrastive Decoding with Abstention
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12527v3](http://arxiv.org/pdf/2412.12527v3)**

> **作者:** Hyuhng Joon Kim; Youna Kim; Sang-goo Lee; Taeuk Kim
>
> **备注:** ACL 2025 (main)
>
> **摘要:** Large Language Models (LLMs) demonstrate exceptional performance across diverse tasks by leveraging pre-trained (i.e., parametric) and external (i.e., contextual) knowledge. While substantial efforts have been made to enhance the utilization of both forms of knowledge, situations in which models lack relevant information remain underexplored. To investigate this challenge, we first present a controlled testbed featuring four distinct knowledge access scenarios, including the aforementioned edge case, revealing that conventional LLM usage exhibits insufficient robustness in handling all instances. Addressing this limitation, we propose Contrastive Decoding with Abstention (CDA), a novel training-free decoding method that allows LLMs to generate responses when relevant knowledge is available and to abstain otherwise. CDA estimates the relevance of both knowledge sources for a given input, adaptively deciding which type of information to prioritize and which to exclude. Through extensive experiments, we demonstrate that CDA can effectively perform accurate generation and abstention simultaneously, enhancing reliability and preserving user trust.
>
---
#### [replaced 054] Sparsing Law: Towards Large Language Models with Greater Activation Sparsity
- **分类: cs.LG; cs.CL; stat.ML; I.2.7**

- **链接: [http://arxiv.org/pdf/2411.02335v3](http://arxiv.org/pdf/2411.02335v3)**

> **作者:** Yuqi Luo; Chenyang Song; Xu Han; Yingfa Chen; Chaojun Xiao; Zhiyuan Liu; Maosong Sun
>
> **备注:** 23 pages, 13 figures, 6 tables
>
> **摘要:** Activation sparsity denotes the existence of substantial weakly-contributed elements within activation outputs that can be eliminated, benefiting many important applications concerned with large language models (LLMs). Although promoting greater activation sparsity within LLMs deserves deep studies, existing works lack comprehensive and quantitative research on the correlation between activation sparsity and potentially influential factors. In this paper, we present a comprehensive study on the quantitative scaling properties and influential factors of the activation sparsity within decoder-only Transformer-based LLMs. Specifically, we propose PPL-$p\%$ sparsity, a precise and performance-aware activation sparsity metric that is applicable to any activation function. Through extensive experiments, we find several important phenomena. Firstly, different activation functions exhibit comparable performance but opposite training-time sparsity trends. The activation ratio (i.e., $1-\mathrm{sparsity\ ratio}$) evolves as a convergent increasing power-law and decreasing logspace power-law with the amount of training data for SiLU-activated and ReLU-activated LLMs, respectively. These demonstrate that ReLU is more efficient as the activation function than SiLU and can leverage more training data to improve activation sparsity. Secondly, the activation ratio linearly increases with the width-depth ratio below a certain bottleneck point, indicating the potential advantage of a deeper architecture at a fixed parameter scale. Finally, at similar width-depth ratios, we surprisingly find that the limit value of activation sparsity varies weakly with the parameter scale, i.e., the activation patterns within LLMs are insensitive to the parameter scale. These empirical laws towards LLMs with greater activation sparsity have important implications for making LLMs more efficient and interpretable.
>
---
#### [replaced 055] LLM Content Moderation and User Satisfaction: Evidence from Response Refusals in Chatbot Arena
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.SI**

- **链接: [http://arxiv.org/pdf/2501.03266v2](http://arxiv.org/pdf/2501.03266v2)**

> **作者:** Stefan Pasch
>
> **摘要:** LLM safety and ethical alignment are widely discussed, but the impact of content moderation on user satisfaction remains underexplored. In particular, little is known about how users respond when models refuse to answer a prompt-one of the primary mechanisms used to enforce ethical boundaries in LLMs. We address this gap by analyzing nearly 50,000 model comparisons from Chatbot Arena, a platform where users indicate their preferred LLM response in pairwise matchups, providing a large-scale setting for studying real-world user preferences. Using a novel RoBERTa-based refusal classifier fine-tuned on a hand-labeled dataset, we distinguish between refusals due to ethical concerns and technical limitations. Our results reveal a substantial refusal penalty: ethical refusals yield significantly lower win rates than both technical refusals and standard responses, indicating that users are especially dissatisfied when models decline a task for ethical reasons. However, this penalty is not uniform. Refusals receive more favorable evaluations when the underlying prompt is highly sensitive (e.g., involving illegal content), and when the refusal is phrased in a detailed and contextually aligned manner. These findings underscore a core tension in LLM design: safety-aligned behaviors may conflict with user expectations, calling for more adaptive moderation strategies that account for context and presentation.
>
---
#### [replaced 056] TestAgent: A Framework for Domain-Adaptive Evaluation of LLMs via Dynamic Benchmark Construction and Exploratory Interaction
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.11507v4](http://arxiv.org/pdf/2410.11507v4)**

> **作者:** Wanying Wang; Zeyu Ma; Pengfei Liu; Mingang Chen
>
> **摘要:** As large language models (LLMs) are increasingly deployed to various vertical domains, automatically evaluating their performance across different domains remains a critical challenge. Current evaluation methods often rely on static and resource-intensive datasets that are not aligned with real-world requirements and lack cross-domain adaptability. To address these limitations, we revisit the evaluation process and introduce two key concepts: \textbf{Benchmark+}, which extends the traditional question-answer benchmark into a more flexible ``strategy-criterion'' format; and \textbf{Assessment+}, which enhances the interaction process to facilitate deeper exploration and comprehensive analysis from multiple perspectives. We propose \textbf{\textsc{TestAgent}}, an agent-based evaluation framework that implements these concepts using retrieval-augmented generation and reinforcement learning. \textsc{TestAgent} enables automatic dynamic benchmark generation and in-depth assessment across diverse vertical domains. Experiments on tasks ranging from constructing multiple vertical domain evaluations to transforming static benchmarks into dynamic forms demonstrate the effectiveness of \textsc{TestAgent}. This work provides a novel perspective on automatic evaluation methods for domain-specific LLMs, offering a pathway for domain-adaptive dynamic benchmark construction and exploratory assessment.
>
---
#### [replaced 057] Thousand Voices of Trauma: A Large-Scale Synthetic Dataset for Modeling Prolonged Exposure Therapy Conversations
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; cs.LG; 68T50; I.2.7; H.5.2**

- **链接: [http://arxiv.org/pdf/2504.13955v4](http://arxiv.org/pdf/2504.13955v4)**

> **作者:** Suhas BN; Andrew M. Sherrill; Rosa I. Arriaga; Chris W. Wiese; Saeed Abdullah
>
> **备注:** 22 pages, 6 figures Updated Appendix with example model responses
>
> **摘要:** The advancement of AI systems for mental health support is hindered by limited access to therapeutic conversation data, particularly for trauma treatment. We present Thousand Voices of Trauma, a synthetic benchmark dataset of 3,000 therapy conversations based on Prolonged Exposure therapy protocols for Post-traumatic Stress Disorder (PTSD). The dataset comprises 500 unique cases, each explored through six conversational perspectives that mirror the progression of therapy from initial anxiety to peak distress to emotional processing. We incorporated diverse demographic profiles (ages 18-80, M=49.3, 49.4% male, 44.4% female, 6.2% non-binary), 20 trauma types, and 10 trauma-related behaviors using deterministic and probabilistic generation methods. Analysis reveals realistic distributions of trauma types (witnessing violence 10.6%, bullying 10.2%) and symptoms (nightmares 23.4%, substance abuse 20.8%). Clinical experts validated the dataset's therapeutic fidelity, highlighting its emotional depth while suggesting refinements for greater authenticity. We also developed an emotional trajectory benchmark with standardized metrics for evaluating model responses. This privacy-preserving dataset addresses critical gaps in trauma-focused mental health data, offering a valuable resource for advancing both patient-facing applications and clinician training tools.
>
---
#### [replaced 058] AD-LLM: Benchmarking Large Language Models for Anomaly Detection
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.11142v3](http://arxiv.org/pdf/2412.11142v3)**

> **作者:** Tiankai Yang; Yi Nian; Shawn Li; Ruiyao Xu; Yuangang Li; Jiaqi Li; Zhuo Xiao; Xiyang Hu; Ryan Rossi; Kaize Ding; Xia Hu; Yue Zhao
>
> **摘要:** Anomaly detection (AD) is an important machine learning task with many real-world uses, including fraud detection, medical diagnosis, and industrial monitoring. Within natural language processing (NLP), AD helps detect issues like spam, misinformation, and unusual user activity. Although large language models (LLMs) have had a strong impact on tasks such as text generation and summarization, their potential in AD has not been studied enough. This paper introduces AD-LLM, the first benchmark that evaluates how LLMs can help with NLP anomaly detection. We examine three key tasks: (i) zero-shot detection, using LLMs' pre-trained knowledge to perform AD without tasks-specific training; (ii) data augmentation, generating synthetic data and category descriptions to improve AD models; and (iii) model selection, using LLMs to suggest unsupervised AD models. Through experiments with different datasets, we find that LLMs can work well in zero-shot AD, that carefully designed augmentation methods are useful, and that explaining model selection for specific datasets remains challenging. Based on these results, we outline six future research directions on LLMs for AD.
>
---
#### [replaced 059] DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07233v2](http://arxiv.org/pdf/2505.07233v2)**

> **作者:** Jiashuo Sun; Xianrui Zhong; Sizhe Zhou; Jiawei Han
>
> **备注:** 24 pages, 7 figures, 15 tables
>
> **摘要:** Retrieval-augmented generation (RAG) systems combine large language models (LLMs) with external knowledge retrieval, making them highly effective for knowledge-intensive tasks. A crucial but often under-explored component of these systems is the reranker. Since irrelevant documents in RAG systems can mislead the generator, the reranker plays a vital role in refining retrieved documents to enhance generation quality and explainability. However, it is challenging to determine the appropriate number of documents ($k$) that the reranker should select: too few may result in missing critical information, while too many introduce noise and inefficiencies. Although recent studies have explored LLM-based rerankers, they primarily leverage internal model knowledge and overlook the rich supervisory signals that LLMs can provide, such as using response quality as feedback for optimizing reranking decisions. In this paper, we propose DynamicRAG, a novel RAG framework where the reranker dynamically adjusts both the order and number of retrieved documents based on the query. We model the reranker as an agent optimized through reinforcement learning (RL), using rewards derived from LLM output quality. Across seven knowledge-intensive datasets, DynamicRAG demonstrates superior performance, achieving state-of-the-art results among models of same parameter sizes. The model, data and code are available at https://github.com/GasolSun36/DynamicRAG.
>
---
#### [replaced 060] Shuttle Between the Instructions and the Parameters of Large Language Models
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02315v3](http://arxiv.org/pdf/2502.02315v3)**

> **作者:** Wangtao Sun; Haotian Xu; Huanxuan Liao; Xuanqing Yu; Zhongtao Jiang; Shizhu He; Jun Zhao; Kang Liu
>
> **摘要:** The interaction with Large Language Models (LLMs) through instructions has been extensively investigated in the research community. While instructions have been widely used as the guidelines for task solving, this paper further notices that both instructions and parameters are the compression of task data. Therefore, they could be strongly correlated and can be learned to predict one from the other. This paper proposes a novel neural network framework, SHIP (\textbf{Sh}uttle between the \textbf{I}nstructions and the \textbf{P}arameters), to model and learn the mutual mappings between the instructions and the parameters of LLMs. We verify that SHIP can effectively map one of the instructions/parameters to the other by evaluating it on the tasks of instruction deduction and induction. The results show that SHIP performs better than existing baseline methods in terms of deductive capabilities while significantly surpassing them in inductive capabilities. Moreover, SHIP can effectively combine the two mapping processes to perform excellent inductive reasoning. The code and data for this paper are released at https://anonymous.4open.science/r/Shuttle-Between-Instructions-Parameters/.
>
---
#### [replaced 061] Mix Data or Merge Models? Balancing the Helpfulness, Honesty, and Harmlessness of Large Language Model via Model Merging
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.06876v3](http://arxiv.org/pdf/2502.06876v3)**

> **作者:** Jinluan Yang; Dingnan Jin; Anke Tang; Li Shen; Didi Zhu; Zhengyu Chen; Ziyu Zhao; Daixin Wang; Qing Cui; Zhiqiang Zhang; Jun Zhou; Fei Wu; Kun Kuang
>
> **摘要:** Achieving balanced alignment of large language models (LLMs) in terms of Helpfulness, Honesty, and Harmlessness (3H optimization) constitutes a cornerstone of responsible AI. Existing methods like data mixture strategies face limitations, including heavy reliance on expert knowledge and conflicting optimization signals. While model merging offers parameter-level conflict-resolution strategies through integrating specialized models' parameters, its potential for 3H optimization remains underexplored. This paper systematically compares the effectiveness of model merging and data mixture methods in constructing 3H-aligned LLMs for the first time, revealing previously overlooked collaborative and conflict relationships among the 3H dimensions and discussing the advantages and drawbacks of data mixture (\textit{data-level}) and model merging (\textit{parameter-level}) methods in mitigating the conflict for balanced 3H optimization. Specially, we propose a novel \textbf{R}eweighting \textbf{E}nhanced task \textbf{S}ingular \textbf{M}erging method, \textbf{RESM}, through outlier weighting and sparsity-aware rank selection strategies to address the challenges of preference noise accumulation and layer sparsity adaptation inherent in 3H-aligned LLM merging. Extensive evaluations can verify the effectiveness and robustness of RESM compared to previous data mixture (2\%-5\% gain) and model merging (1\%-3\% gain) methods in achieving balanced LLM alignment. We release our models through \href{https://huggingface.co/Jinluan}{3H\_Merging} for further investigations.
>
---
#### [replaced 062] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13837v2](http://arxiv.org/pdf/2504.13837v2)**

> **作者:** Yang Yue; Zhiqi Chen; Rui Lu; Andrew Zhao; Zhaokai Wang; Yang Yue; Shiji Song; Gao Huang
>
> **备注:** 30 pages, 27 figures
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated notable success in enhancing the reasoning performance of large language models (LLMs), particularly on mathematics and programming tasks. Similar to how traditional RL helps agents explore and learn new strategies, RLVR is believed to enable LLMs to continuously self-improve, thus acquiring novel reasoning abilities beyond those of the corresponding base models. In this study we critically examine the current state of RLVR by systematically probing the reasoning capability boundaries of RLVR-trained LLMs across various model families, RL algorithms, and math, coding, and visual reasoning benchmarks, using pass@k at large k values as the evaluation metric. Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns. While RLVR-trained models outperform their base models at small k (e.g., k = 1), the base models achieve a higher pass@k score when k is large. Coverage and perplexity analyses show that the observed reasoning abilities originate from and are bounded by the base model. Treating the base model as an upper bound, our quantitative analysis shows that six popular RLVR algorithms perform similarly and remain far from optimal in leveraging the potential of the base model. By contrast, we find that distillation can introduce new reasoning patterns from the teacher and genuinely expand the model's reasoning capabilities. Overall, our findings suggest that current RLVR methods have not yet realized the potential of RL to elicit truly novel reasoning abilities in LLMs. This highlights the need for improved RL paradigms, such as continual scaling and multi-turn agent-environment interaction, to unlock this potential.
>
---
#### [replaced 063] From Rankings to Insights: Evaluation Should Shift Focus from Leaderboard to Feedback
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.06698v2](http://arxiv.org/pdf/2505.06698v2)**

> **作者:** Zongqi Wang; Tianle Gu; Chen Gong; Xin Tian; Siqi Bao; Yujiu Yang
>
> **摘要:** Automatic evaluation benchmarks such as MT-Bench, Arena-Hard, and Auto-Arena are seeing growing adoption for the evaluation of Large Language Models (LLMs). Existing research has primarily focused on approximating human-based model rankings using limited data and LLM-as-a-Judge. However, the fundamental premise of these studies, which attempts to replicate human rankings, is flawed. Specifically, these benchmarks typically offer only overall scores, limiting their utility to leaderboard rankings, rather than providing feedback that can guide model optimization and support model profiling. Therefore, we advocate for an evaluation paradigm shift from approximating human-based model rankings to providing feedback with analytical value. To this end, we introduce \textbf{Feedbacker}, an evaluation framework that provides comprehensive and fine-grained results, thereby enabling thorough identification of a model's specific strengths and weaknesses. Such feedback not only supports the targeted optimization of the model but also enhances the understanding of its behavior. Feedbacker comprises three key components: an extensible tree-based query taxonomy builder, an automated query synthesis scheme, and a suite of visualization and analysis tools. Furthermore, we propose a novel LLM-as-a-Judge method: PC$^{2}$ (Pre-Comparison-derived Criteria) pointwise evaluation. This method derives evaluation criteria by pre-comparing the differences between several auxiliary responses, achieving the accuracy of pairwise evaluation while maintaining the time complexity of pointwise evaluation. Finally, leveraging the evaluation results of 17 mainstream LLMs, we demonstrate the usage of Feedbacker and highlight its effectiveness and potential. Our project homepage and dataset are available at https://liudan193.github.io/Feedbacker.
>
---
#### [replaced 064] From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models
- **分类: cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2505.09924v2](http://arxiv.org/pdf/2505.09924v2)**

> **作者:** Yidan Wang; Yubing Ren; Yanan Cao; Binxing Fang
>
> **备注:** Accepted to ACL 2025 (main)
>
> **摘要:** The rise of Large Language Models (LLMs) has heightened concerns about the misuse of AI-generated text, making watermarking a promising solution. Mainstream watermarking schemes for LLMs fall into two categories: logits-based and sampling-based. However, current schemes entail trade-offs among robustness, text quality, and security. To mitigate this, we integrate logits-based and sampling-based schemes, harnessing their respective strengths to achieve synergy. In this paper, we propose a versatile symbiotic watermarking framework with three strategies: serial, parallel, and hybrid. The hybrid framework adaptively embeds watermarks using token entropy and semantic entropy, optimizing the balance between detectability, robustness, text quality, and security. Furthermore, we validate our approach through comprehensive experiments on various datasets and models. Experimental results indicate that our method outperforms existing baselines and achieves state-of-the-art (SOTA) performance. We believe this framework provides novel insights into diverse watermarking paradigms. Our code is available at https://github.com/redwyd/SymMark.
>
---
#### [replaced 065] Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.06326v5](http://arxiv.org/pdf/2406.06326v5)**

> **作者:** Xiaoying Zhang; Baolin Peng; Ye Tian; Jingyan Zhou; Yipeng Zhang; Haitao Mi; Helen Meng
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large language models (LLMs) often struggle to provide up-to-date information due to their one-time training and the constantly evolving nature of the world. To keep LLMs current, existing approaches typically involve continued pre-training on new documents. However, they frequently face difficulties in extracting stored knowledge. Motivated by the remarkable success of the Feynman Technique in efficient human learning, we introduce Self-Tuning, a learning framework aimed at improving an LLM's ability to effectively acquire new knowledge from unseen raw documents through self-teaching. Specifically, we develop a Self-Teaching strategy that augments the documents with a set of knowledge-intensive tasks created in a self-supervised manner, focusing on three crucial aspects: memorization, comprehension, and self-reflection. Additionally, we introduce three Wiki-Newpages-2023-QA datasets to facilitate an in-depth analysis of an LLM's knowledge acquisition ability concerning memorization, extraction, and reasoning. Extensive experimental results on various models, e.g., Llama2-7B reveal that Self-Tuning consistently exhibits superior performance across all knowledge acquisition tasks and excels in preserving previous knowledge.
>
---
#### [replaced 066] Item-Language Model for Conversational Recommendation
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.02844v2](http://arxiv.org/pdf/2406.02844v2)**

> **作者:** Li Yang; Anushya Subbiah; Hardik Patel; Judith Yue Li; Yanwei Song; Reza Mirghaderi; Vikram Aggarwal; Qifan Wang
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Large-language Models (LLMs) have been extremely successful at tasks like complex dialogue understanding, reasoning and coding due to their emergent abilities. These emergent abilities have been extended with multi-modality to include image, audio, and video capabilities. Recommender systems, on the other hand, have been critical for information seeking and item discovery needs. Recently, there have been attempts to apply LLMs for recommendations. One difficulty of current attempts is that the underlying LLM is usually not trained on the recommender system data, which largely contains user interaction signals and is often not publicly available. Another difficulty is user interaction signals often have a different pattern from natural language text, and it is currently unclear if the LLM training setup can learn more non-trivial knowledge from interaction signals compared with traditional recommender system methods. Finally, it is difficult to train multiple LLMs for different use-cases, and to retain the original language and reasoning abilities when learning from recommender system data. To address these three limitations, we propose an Item-Language Model (ILM), which is composed of an item encoder to produce text-aligned item representations that encode user interaction signals, and a frozen LLM that can understand those item representations with preserved pretrained knowledge. We conduct extensive experiments which demonstrate both the importance of the language-alignment and of user interaction knowledge in the item encoder.
>
---
#### [replaced 067] Towards Adapting Open-Source Large Language Models for Expert-Level Clinical Note Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.00715v5](http://arxiv.org/pdf/2405.00715v5)**

> **作者:** Hanyin Wang; Chufan Gao; Bolun Liu; Qiping Xu; Guleid Hussein; Mohamad El Labban; Kingsley Iheasirim; Hariprasad Korsapati; Chuck Outcalt; Jimeng Sun
>
> **摘要:** Proprietary Large Language Models (LLMs) such as GPT-4 and Gemini have demonstrated promising capabilities in clinical text summarization tasks. However, due to patient data privacy concerns and computational costs, many healthcare providers prefer using small, locally-hosted models over external generic LLMs. This study presents a comprehensive domain- and task-specific adaptation process for the open-source LLaMA-2 13 billion parameter model, enabling it to generate high-quality clinical notes from outpatient patient-doctor dialogues. Our process incorporates continued pre-training, supervised fine-tuning, and reinforcement learning from both AI and human feedback. We introduced a new approach, DistillDirect, for performing on-policy reinforcement learning with Gemini 1.0 Pro as the teacher model. Our resulting model, LLaMA-Clinic, can generate clinical notes comparable in quality to those authored by physicians. In a blinded physician reader study, the majority (90.4%) of individual evaluations rated the notes generated by LLaMA-Clinic as "acceptable" or higher across all three criteria: real-world readiness, completeness, and accuracy. In the more challenging "Assessment and Plan" section, LLaMA-Clinic scored higher (4.2/5) in real-world readiness than physician-authored notes (4.1/5). We highlight key considerations for future clinical note-generation tasks, emphasizing the importance of pre-defining a best-practice note format, rather than relying on LLMs to determine this for clinical practice.
>
---
#### [replaced 068] Tales of the 2025 Los Angeles Fire: Hotwash for Public Health Concerns in Reddit via LLM-Enhanced Topic Modeling
- **分类: cs.SI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.09665v2](http://arxiv.org/pdf/2505.09665v2)**

> **作者:** Sulong Zhou; Qunying Huang; Shaoheng Zhou; Yun Hang; Xinyue Ye; Aodong Mei; Kathryn Phung; Yuning Ye; Uma Govindswamy; Zehan Li
>
> **备注:** Corrected capitalization errors in the section subtitle 3.4, 4.3, step 1 in section 3.3.2, and Supplementary Information. Fix typo with "Weighting" for step 4 in section 3.3.2
>
> **摘要:** Wildfires have become increasingly frequent, irregular, and severe in recent years. Understanding how affected populations perceive and respond during wildfire crises is critical for timely and empathetic disaster response. Social media platforms offer a crowd-sourced channel to capture evolving public discourse, providing hyperlocal information and insight into public sentiment. This study analyzes Reddit discourse during the 2025 Los Angeles wildfires, spanning from the onset of the disaster to full containment. We collect 385 posts and 114,879 comments related to the Palisades and Eaton fires. We adopt topic modeling methods to identify the latent topics, enhanced by large language models (LLMs) and human-in-the-loop (HITL) refinement. Furthermore, we develop a hierarchical framework to categorize latent topics, consisting of two main categories, Situational Awareness (SA) and Crisis Narratives (CN). The volume of SA category closely aligns with real-world fire progressions, peaking within the first 2-5 days as the fires reach the maximum extent. The most frequent co-occurring category set of public health and safety, loss and damage, and emergency resources expands on a wide range of health-related latent topics, including environmental health, occupational health, and one health. Grief signals and mental health risks consistently accounted for 60 percentage and 40 percentage of CN instances, respectively, with the highest total volume occurring at night. This study contributes the first annotated social media dataset on the 2025 LA fires, and introduces a scalable multi-layer framework that leverages topic modeling for crisis discourse analysis. By identifying persistent public health concerns, our results can inform more empathetic and adaptive strategies for disaster response, public health communication, and future research in comparable climate-related disaster events.
>
---
#### [replaced 069] Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2403.11083v2](http://arxiv.org/pdf/2403.11083v2)**

> **作者:** Xiaohao Xu; Yunkang Cao; Huaxin Zhang; Nong Sang; Xiaonan Huang
>
> **备注:** Best Student Paper Award at IEEE International Conference on Computer Supported Cooperative Work in Design, 2025
>
> **摘要:** Anomaly detection is vital in various industrial scenarios, including the identification of unusual patterns in production lines and the detection of manufacturing defects for quality control. Existing techniques tend to be specialized in individual scenarios and lack generalization capacities. In this study, our objective is to develop a generic anomaly detection model that can be applied in multiple scenarios. To achieve this, we custom-build generic visual language foundation models that possess extensive knowledge and robust reasoning abilities as anomaly detectors and reasoners. Specifically, we introduce a multi-modal prompting strategy that incorporates domain knowledge from experts as conditions to guide the models. Our approach considers diverse prompt types, including task descriptions, class context, normality rules, and reference images. In addition, we unify the input representation of multi-modality into a 2D image format, enabling multi-modal anomaly detection and reasoning. Our preliminary studies demonstrate that combining visual and language prompts as conditions for customizing the models enhances anomaly detection performance. The customized models showcase the ability to detect anomalies across different data modalities such as images, point clouds, and videos. Qualitative case studies further highlight the anomaly detection and reasoning capabilities, particularly for multi-object scenes and temporal data. Our code is publicly available at https://github.com/Xiaohao-Xu/Customizable-VLM
>
---
#### [replaced 070] Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities
- **分类: stat.ML; cs.AI; cs.CL; cs.IT; cs.LG; math.IT**

- **链接: [http://arxiv.org/pdf/2501.02406v4](http://arxiv.org/pdf/2501.02406v4)**

> **作者:** Tara Radvand; Mojtaba Abdolmaleki; Mohamed Mostagir; Ambuj Tewari
>
> **摘要:** Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. In this paper, we answer the following question: Given a piece of text, can we identify whether it was produced by a particular LLM or not? We model LLM-generated text as a sequential stochastic process with complete dependence on history. We then design zero-shot statistical tests to (i) distinguish between text generated by two different known sets of LLMs $A$ (non-sanctioned) and $B$ (in-house), and (ii) identify whether text was generated by a known LLM or generated by any unknown model, e.g., a human or some other language generation process. We prove that the type I and type II errors of our test decrease exponentially with the length of the text. For that, we show that if $B$ generates the text, then except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. We then present experiments using LLMs with white-box access to support our theoretical results and empirically examine the robustness of our results to black-box settings and adversarial attacks. In the black-box setting, our method achieves an average TPR of 82.5\% at a fixed FPR of 5\%. Under adversarial perturbations, our minimum TPR is 48.6\% at the same FPR threshold. Both results outperform all non-commercial baselines. See https://github.com/TaraRadvand74/llm-text-detection for code, data, and an online demo of the project.
>
---
#### [replaced 071] Grounding Synthetic Data Evaluations of Language Models in Unsupervised Document Corpora
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08905v2](http://arxiv.org/pdf/2505.08905v2)**

> **作者:** Michael Majurski; Cynthia Matuszek
>
> **摘要:** Language Models (LMs) continue to advance, improving response quality and coherence. Given Internet-scale training datasets, LMs have likely encountered much of what users may ask them to generate in some form during their training. A plethora of evaluation benchmarks have been constructed to assess model quality, response appropriateness, and reasoning capabilities. However, the human effort required for benchmark construction is rapidly being outpaced by the size and scope of the models under evaluation. Having humans build a benchmark for every possible domain of interest is impractical. Therefore, we propose a methodology for automating the construction of fact-based synthetic data model evaluations grounded in document populations. This work leverages the same LMs to evaluate domain-specific knowledge automatically, using only grounding documents (e.g., a textbook) as input. This synthetic data benchmarking approach corresponds well with human curated questions producing a Spearman ranking correlation of 0.97 and a benchmark evaluation Pearson accuracy correlation of 0.75. This novel approach supports generating both multiple choice and open-ended synthetic data questions to gain diagnostic insight of LM capability. We apply this methodology to evaluate model performance on two recent arXiv preprints, discovering a surprisingly strong performance from Gemma-3 models on open-ended questions. Code is available at https://github.com/mmajurski/grounded-synth-lm-benchmark
>
---
#### [replaced 072] Words in Motion: Extracting Interpretable Control Vectors for Motion Transformers
- **分类: cs.LG; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.11624v5](http://arxiv.org/pdf/2406.11624v5)**

> **作者:** Omer Sahin Tas; Royden Wagner
>
> **备注:** ICLR 2025 final version. Our implementation is available at https://github.com/kit-mrt/future-motion
>
> **摘要:** Transformer-based models generate hidden states that are difficult to interpret. In this work, we analyze hidden states and modify them at inference, with a focus on motion forecasting. We use linear probing to analyze whether interpretable features are embedded in hidden states. Our experiments reveal high probing accuracy, indicating latent space regularities with functionally important directions. Building on this, we use the directions between hidden states with opposing features to fit control vectors. At inference, we add our control vectors to hidden states and evaluate their impact on predictions. Remarkably, such modifications preserve the feasibility of predictions. We further refine our control vectors using sparse autoencoders (SAEs). This leads to more linear changes in predictions when scaling control vectors. Our approach enables mechanistic interpretation as well as zero-shot generalization to unseen dataset characteristics with negligible computational overhead.
>
---
