# 自然语言处理 cs.CL

- **最新发布 61 篇**

- **更新 44 篇**

## 最新发布

#### [new 001] An AI-Powered Research Assistant in the Lab: A Practical Guide for Text Analysis Through Iterative Collaboration with LLMs
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于文本分析任务，旨在解决传统方法效率低、易偏差的问题。提出研究者与LLMs迭代协作的流程，指导开发、测试和应用分类法处理非结构化数据。通过生成生活领域分类案例，演示提示编写、评估优化及高一致性分类实现，并讨论LLMs的潜力与限制。**

- **链接: [http://arxiv.org/pdf/2505.09724v1](http://arxiv.org/pdf/2505.09724v1)**

> **作者:** Gino Carmona-Díaz; William Jiménez-Leal; María Alejandra Grisales; Chandra Sripada; Santiago Amaya; Michael Inzlicht; Juan Pablo Bermúdez
>
> **备注:** 31 pages, 1 figure
>
> **摘要:** Analyzing texts such as open-ended responses, headlines, or social media posts is a time- and labor-intensive process highly susceptible to bias. LLMs are promising tools for text analysis, using either a predefined (top-down) or a data-driven (bottom-up) taxonomy, without sacrificing quality. Here we present a step-by-step tutorial to efficiently develop, test, and apply taxonomies for analyzing unstructured data through an iterative and collaborative process between researchers and LLMs. Using personal goals provided by participants as an example, we demonstrate how to write prompts to review datasets and generate a taxonomy of life domains, evaluate and refine the taxonomy through prompt and direct modifications, test the taxonomy and assess intercoder agreements, and apply the taxonomy to categorize an entire dataset with high intercoder reliability. We discuss the possibilities and limitations of using LLMs for text analysis.
>
---
#### [new 002] Exploring the generalization of LLM truth directions on conversational formats
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型（LLM）可解释性任务，研究真实性检测方向的跨对话格式泛化问题。针对现有线性探针在长对话格式（谎言出现在输入前端）中泛化差的问题，提出通过在对话末尾添加固定关键词的方法，有效提升了检测效果，但揭示了LLM谎言检测器适应新场景的挑战。**

- **链接: [http://arxiv.org/pdf/2505.09807v1](http://arxiv.org/pdf/2505.09807v1)**

> **作者:** Timour Ichmoukhamedov; David Martens
>
> **摘要:** Several recent works argue that LLMs have a universal truth direction where true and false statements are linearly separable in the activation space of the model. It has been demonstrated that linear probes trained on a single hidden state of the model already generalize across a range of topics and might even be used for lie detection in LLM conversations. In this work we explore how this truth direction generalizes between various conversational formats. We find good generalization between short conversations that end on a lie, but poor generalization to longer formats where the lie appears earlier in the input prompt. We propose a solution that significantly improves this type of generalization by adding a fixed key phrase at the end of each conversation. Our results highlight the challenges towards reliable LLM lie detectors that generalize to new settings.
>
---
#### [new 003] Automated Detection of Clinical Entities in Lung and Breast Cancer Reports Using NLP Techniques
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床信息抽取任务，旨在解决癌症报告中人工提取数据效率低、易出错的问题。研究利用NLP技术（uQuery工具及微调RoBERTa模型），通过命名实体识别自动抽取肺癌和乳腺癌电子病历中的关键临床实体（如MET/PAT），在600份西班牙语报告数据集上验证了有效性，但低频实体识别仍存挑战。**

- **链接: [http://arxiv.org/pdf/2505.09794v1](http://arxiv.org/pdf/2505.09794v1)**

> **作者:** J. Moreno-Casanova; J. M. Auñón; A. Mártinez-Pérez; M. E. Pérez-Martínez; M. E. Gas-López
>
> **摘要:** Research projects, including those focused on cancer, rely on the manual extraction of information from clinical reports. This process is time-consuming and prone to errors, limiting the efficiency of data-driven approaches in healthcare. To address these challenges, Natural Language Processing (NLP) offers an alternative for automating the extraction of relevant data from electronic health records (EHRs). In this study, we focus on lung and breast cancer due to their high incidence and the significant impact they have on public health. Early detection and effective data management in both types of cancer are crucial for improving patient outcomes. To enhance the accuracy and efficiency of data extraction, we utilized GMV's NLP tool uQuery, which excels at identifying relevant entities in clinical texts and converting them into standardized formats such as SNOMED and OMOP. uQuery not only detects and classifies entities but also associates them with contextual information, including negated entities, temporal aspects, and patient-related details. In this work, we explore the use of NLP techniques, specifically Named Entity Recognition (NER), to automatically identify and extract key clinical information from EHRs related to these two cancers. A dataset from Health Research Institute Hospital La Fe (IIS La Fe), comprising 200 annotated breast cancer and 400 lung cancer reports, was used, with eight clinical entities manually labeled using the Doccano platform. To perform NER, we fine-tuned the bsc-bio-ehr-en3 model, a RoBERTa-based biomedical linguistic model pre-trained in Spanish. Fine-tuning was performed using the Transformers architecture, enabling accurate recognition of clinical entities in these cancer types. Our results demonstrate strong overall performance, particularly in identifying entities like MET and PAT, although challenges remain with less frequent entities like EVOL.
>
---
#### [new 004] Hierarchical Document Refinement for Long-context Retrieval-augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对长文本检索增强生成（RAG）中冗余信息多、计算成本高的问题，提出LongRefiner模型。通过分层文档结构解析和双级查询分析，结合多任务学习实现自适应优化。实验表明其以10倍低消耗达到基线性能，适用于现实长文本场景。**

- **链接: [http://arxiv.org/pdf/2505.10413v1](http://arxiv.org/pdf/2505.10413v1)**

> **作者:** Jiajie Jin; Xiaoxi Li; Guanting Dong; Yuyao Zhang; Yutao Zhu; Yongkang Wu; Zhonghua Li; Qi Ye; Zhicheng Dou
>
> **摘要:** Real-world RAG applications often encounter long-context input scenarios, where redundant information and noise results in higher inference costs and reduced performance. To address these challenges, we propose LongRefiner, an efficient plug-and-play refiner that leverages the inherent structural characteristics of long documents. LongRefiner employs dual-level query analysis, hierarchical document structuring, and adaptive refinement through multi-task learning on a single foundation model. Experiments on seven QA datasets demonstrate that LongRefiner achieves competitive performance in various scenarios while using 10x fewer computational costs and latency compared to the best baseline. Further analysis validates that LongRefiner is scalable, efficient, and effective, providing practical insights for real-world long-text RAG applications. Our code is available at https://github.com/ignorejjj/LongRefiner.
>
---
#### [new 005] Can You Really Trust Code Copilots? Evaluating Large Language Models from a Code Security Perspective
- **分类: cs.CL**

- **简介: 该论文属于代码安全评估任务，针对现有基准单一、缺乏多维度分析的问题，提出多任务评测框架CoV-Eval和改进审查模型VC-Judge，评估20个LLMs在安全编码、漏洞识别与修复的能力，揭示模型生成不安全代码的缺陷及修复难点，为优化方向提供依据。**

- **链接: [http://arxiv.org/pdf/2505.10494v1](http://arxiv.org/pdf/2505.10494v1)**

> **作者:** Yutao Mou; Xiao Deng; Yuxiao Luo; Shikun Zhang; Wei Ye
>
> **备注:** Accepted by ACL2025 Main Conference
>
> **摘要:** Code security and usability are both essential for various coding assistant applications driven by large language models (LLMs). Current code security benchmarks focus solely on single evaluation task and paradigm, such as code completion and generation, lacking comprehensive assessment across dimensions like secure code generation, vulnerability repair and discrimination. In this paper, we first propose CoV-Eval, a multi-task benchmark covering various tasks such as code completion, vulnerability repair, vulnerability detection and classification, for comprehensive evaluation of LLM code security. Besides, we developed VC-Judge, an improved judgment model that aligns closely with human experts and can review LLM-generated programs for vulnerabilities in a more efficient and reliable way. We conduct a comprehensive evaluation of 20 proprietary and open-source LLMs. Overall, while most LLMs identify vulnerable codes well, they still tend to generate insecure codes and struggle with recognizing specific vulnerability types and performing repairs. Extensive experiments and qualitative analyses reveal key challenges and optimization directions, offering insights for future research in LLM code security.
>
---
#### [new 006] From Questions to Clinical Recommendations: Large Language Models Driving Evidence-Based Clinical Decision Making
- **分类: cs.CL**

- **简介: 该论文属于临床决策支持任务，旨在解决临床证据整合效率低的问题。研究开发了基于大语言模型的系统Quicker，自动化完成证据合成、生成建议，并通过基准测试验证其高效性和准确性，协助医生快速制定循证决策。**

- **链接: [http://arxiv.org/pdf/2505.10282v1](http://arxiv.org/pdf/2505.10282v1)**

> **作者:** Dubai Li; Nan Jiang; Kangping Huang; Ruiqi Tu; Shuyu Ouyang; Huayu Yu; Lin Qiao; Chen Yu; Tianshu Zhou; Danyang Tong; Qian Wang; Mengtao Li; Xiaofeng Zeng; Yu Tian; Xinping Tian; Jingsong Li
>
> **摘要:** Clinical evidence, derived from rigorous research and data analysis, provides healthcare professionals with reliable scientific foundations for informed decision-making. Integrating clinical evidence into real-time practice is challenging due to the enormous workload, complex professional processes, and time constraints. This highlights the need for tools that automate evidence synthesis to support more efficient and accurate decision making in clinical settings. This study introduces Quicker, an evidence-based clinical decision support system powered by large language models (LLMs), designed to automate evidence synthesis and generate clinical recommendations modeled after standard clinical guideline development processes. Quicker implements a fully automated chain that covers all phases, from questions to clinical recommendations, and further enables customized decision-making through integrated tools and interactive user interfaces. To evaluate Quicker's capabilities, we developed the Q2CRBench-3 benchmark dataset, based on clinical guideline development records for three different diseases. Experimental results highlighted Quicker's strong performance, with fine-grained question decomposition tailored to user preferences, retrieval sensitivities comparable to human experts, and literature screening performance approaching comprehensive inclusion of relevant studies. In addition, Quicker-assisted evidence assessment effectively supported human reviewers, while Quicker's recommendations were more comprehensive and logically coherent than those of clinicians. In system-level testing, collaboration between a single reviewer and Quicker reduced the time required for recommendation development to 20-40 minutes. In general, our findings affirm the potential of Quicker to help physicians make quicker and more reliable evidence-based clinical decisions.
>
---
#### [new 007] Personalizing Large Language Models using Retrieval Augmented Generation and Knowledge Graph
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于个性化语言模型任务，旨在解决大模型生成不准确、缺乏个性化信息的问题。通过结合检索增强生成（RAG）和知识图谱（KG），利用KG结构化存储实时更新的个人数据（如日历），提升生成准确性和个性化。实验表明该方法优于基线模型，响应时间适度降低。**

- **链接: [http://arxiv.org/pdf/2505.09945v1](http://arxiv.org/pdf/2505.09945v1)**

> **作者:** Deeksha Prahlad; Chanhee Lee; Dongha Kim; Hokeun Kim
>
> **备注:** To appear in the Companion Proceedings of the ACM Web Conference 2025 (WWW Companion '25)
>
> **摘要:** The advent of large language models (LLMs) has allowed numerous applications, including the generation of queried responses, to be leveraged in chatbots and other conversational assistants. Being trained on a plethora of data, LLMs often undergo high levels of over-fitting, resulting in the generation of extra and incorrect data, thus causing hallucinations in output generation. One of the root causes of such problems is the lack of timely, factual, and personalized information fed to the LLM. In this paper, we propose an approach to address these problems by introducing retrieval augmented generation (RAG) using knowledge graphs (KGs) to assist the LLM in personalized response generation tailored to the users. KGs have the advantage of storing continuously updated factual information in a structured way. While our KGs can be used for a variety of frequently updated personal data, such as calendar, contact, and location data, we focus on calendar data in this paper. Our experimental results show that our approach works significantly better in understanding personal information and generating accurate responses compared to the baseline LLMs using personal data as text inputs, with a moderate reduction in response time.
>
---
#### [new 008] Designing and Contextualising Probes for African Languages
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理模型可解释性任务，旨在探究非洲语言预训练模型(PLMs)的语言学知识编码机制。通过分层探测六种非洲语言，结合控制任务验证模型性能，发现适应非洲语言的PLMs比多语言模型编码更多目标语言信息，证实句法特征集中于中后层，语义信息分布全层，并证明模型表现源自内部知识而非探针记忆。**

- **链接: [http://arxiv.org/pdf/2505.10081v1](http://arxiv.org/pdf/2505.10081v1)**

> **作者:** Wisdom Aduah; Francois Meyer
>
> **摘要:** Pretrained language models (PLMs) for African languages are continually improving, but the reasons behind these advances remain unclear. This paper presents the first systematic investigation into probing PLMs for linguistic knowledge about African languages. We train layer-wise probes for six typologically diverse African languages to analyse how linguistic features are distributed. We also design control tasks, a way to interpret probe performance, for the MasakhaPOS dataset. We find PLMs adapted for African languages to encode more linguistic information about target languages than massively multilingual PLMs. Our results reaffirm previous findings that token-level syntactic information concentrates in middle-to-last layers, while sentence-level semantic information is distributed across all layers. Through control tasks and probing baselines, we confirm that performance reflects the internal knowledge of PLMs rather than probe memorisation. Our study applies established interpretability techniques to African-language PLMs. In doing so, we highlight the internal mechanisms underlying the success of strategies like active learning and multilingual adaptation.
>
---
#### [new 009] KRISTEVA: Close Reading as a Novel Task for Benchmarking Interpretive Reasoning
- **分类: cs.CL**

- **简介: 该论文提出首个评估大语言模型文学细读能力的基准测试KRISTEVA，解决现有基准缺乏文学分析任务的问题。通过构建1331道改编自教学数据的多选题，设计三阶段渐进任务（风格特征提取、上下文检索、多跳推理），测试发现先进模型具备基础细读能力（49.7%-69.7%准确率），但多数任务仍落后人类表现。**

- **链接: [http://arxiv.org/pdf/2505.09825v1](http://arxiv.org/pdf/2505.09825v1)**

> **作者:** Peiqi Sui; Juan Diego Rodriguez; Philippe Laban; Dean Murphy; Joseph P. Dexter; Richard Jean So; Samuel Baker; Pramit Chaudhuri
>
> **摘要:** Each year, tens of millions of essays are written and graded in college-level English courses. Students are asked to analyze literary and cultural texts through a process known as close reading, in which they gather textual details to formulate evidence-based arguments. Despite being viewed as a basis for critical thinking and widely adopted as a required element of university coursework, close reading has never been evaluated on large language models (LLMs), and multi-discipline benchmarks like MMLU do not include literature as a subject. To fill this gap, we present KRISTEVA, the first close reading benchmark for evaluating interpretive reasoning, consisting of 1331 multiple-choice questions adapted from classroom data. With KRISTEVA, we propose three progressively more difficult sets of tasks to approximate different elements of the close reading process, which we use to test how well LLMs may seem to understand and reason about literary works: 1) extracting stylistic features, 2) retrieving relevant contextual information from parametric knowledge, and 3) multi-hop reasoning between style and external contexts. Our baseline results find that, while state-of-the-art LLMs possess some college-level close reading competency (accuracy 49.7% - 69.7%), their performances still trail those of experienced human evaluators on 10 out of our 11 tasks.
>
---
#### [new 010] Beyond 'Aha!': Toward Systematic Meta-Abilities Alignment in Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文属于大型推理模型能力优化任务，旨在解决其推理行为不可控、不可靠的问题。研究提出显式对齐演绎、归纳、溯因三种元能力的方法，通过自验证任务的三阶段训练流程（个体对齐/参数合并/领域强化学习），相比基线提升10%性能，并在数学/编程/科学任务中额外提升2%，建立可扩展的推理基础。**

- **链接: [http://arxiv.org/pdf/2505.10554v1](http://arxiv.org/pdf/2505.10554v1)**

> **作者:** Zhiyuan Hu; Yibo Wang; Hanze Dong; Yuhui Xu; Amrita Saha; Caiming Xiong; Bryan Hooi; Junnan Li
>
> **备注:** In Progress
>
> **摘要:** Large reasoning models (LRMs) already possess a latent capacity for long chain-of-thought reasoning. Prior work has shown that outcome-based reinforcement learning (RL) can incidentally elicit advanced reasoning behaviors such as self-correction, backtracking, and verification phenomena often referred to as the model's "aha moment". However, the timing and consistency of these emergent behaviors remain unpredictable and uncontrollable, limiting the scalability and reliability of LRMs' reasoning capabilities. To address these limitations, we move beyond reliance on prompts and coincidental "aha moments". Instead, we explicitly align models with three meta-abilities: deduction, induction, and abduction, using automatically generated, self-verifiable tasks. Our three stage-pipeline individual alignment, parameter-space merging, and domain-specific reinforcement learning, boosting performance by over 10\% relative to instruction-tuned baselines. Furthermore, domain-specific RL from the aligned checkpoint yields an additional 2\% average gain in the performance ceiling across math, coding, and science benchmarks, demonstrating that explicit meta-ability alignment offers a scalable and dependable foundation for reasoning. Code is available at: https://github.com/zhiyuanhubj/Meta-Ability-Alignment
>
---
#### [new 011] DRA-GRPO: Exploring Diversity-Aware Reward Adjustment for R1-Zero-Like Training of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大型语言模型的强化学习调优任务，旨在解决传统方法因忽视语义多样性导致的探索不足问题。作者提出多样性感知奖励调整方法DRA，通过子模互信息动态调节奖励，抑制冗余结果并增强多样性探索。结合GRPO框架后，在低资源数学推理任务中取得58.2%的平均准确率（SOTA），仅需7000样本和55美元训练成本。**

- **链接: [http://arxiv.org/pdf/2505.09655v1](http://arxiv.org/pdf/2505.09655v1)**

> **作者:** Xiwen Chen; Wenhui Zhu; Peijie Qiu; Xuanzhao Dong; Hao Wang; Haiyu Wu; Huayu Li; Aristeidis Sotiras; Yalin Wang; Abolfazl Razi
>
> **摘要:** Recent advances in reinforcement learning for language model post-training, such as Group Relative Policy Optimization (GRPO), have shown promise in low-resource settings. However, GRPO typically relies on solution-level and scalar reward signals that fail to capture the semantic diversity among sampled completions. This leads to what we identify as a diversity-quality inconsistency, where distinct reasoning paths may receive indistinguishable rewards. To address this limitation, we propose $\textit{Diversity-aware Reward Adjustment}$ (DRA), a method that explicitly incorporates semantic diversity into the reward computation. DRA uses Submodular Mutual Information (SMI) to downweight redundant completions and amplify rewards for diverse ones. This encourages better exploration during learning, while maintaining stable exploitation of high-quality samples. Our method integrates seamlessly with both GRPO and its variant DR.~GRPO, resulting in $\textit{DRA-GRPO}$ and $\textit{DGA-DR.~GRPO}$. We evaluate our method on five mathematical reasoning benchmarks and find that it outperforms recent strong baselines. It achieves state-of-the-art performance with an average accuracy of 58.2%, using only 7,000 fine-tuning samples and a total training cost of approximately $55. The code is available at https://github.com/xiwenc1/DRA-GRPO.
>
---
#### [new 012] Large Language Models Are More Persuasive Than Incentivized Human Persuaders
- **分类: cs.CL; I.2.7; H.1.2; K.4.1; H.5.2**

- **简介: 该论文属于AI能力评估任务，旨在比较大语言模型与激励人类的说服力差异。通过在线测验实验发现，Claude模型在真实/欺骗性劝说中的效果均显著优于受金钱激励的人类，既能提升答题准确率（正确引导）也可误导降低收益（错误引导），揭示了AI说服力超越人类的事实及治理紧迫性。**

- **链接: [http://arxiv.org/pdf/2505.09662v1](http://arxiv.org/pdf/2505.09662v1)**

> **作者:** Philipp Schoenegger; Francesco Salvi; Jiacheng Liu; Xiaoli Nan; Ramit Debnath; Barbara Fasolo; Evelina Leivada; Gabriel Recchia; Fritz Günther; Ali Zarifhonarvar; Joe Kwon; Zahoor Ul Islam; Marco Dehnert; Daryl Y. H. Lee; Madeline G. Reinecke; David G. Kamper; Mert Kobaş; Adam Sandford; Jonas Kgomo; Luke Hewitt; Shreya Kapoor; Kerem Oktar; Eyup Engin Kucuk; Bo Feng; Cameron R. Jones; Izzy Gainsburg; Sebastian Olschewski; Nora Heinzelmann; Francisco Cruz; Ben M. Tappin; Tao Ma; Peter S. Park; Rayan Onyonka; Arthur Hjorth; Peter Slattery; Qingcheng Zeng; Lennart Finke; Igor Grossmann; Alessandro Salatiello; Ezra Karger
>
> **摘要:** We directly compare the persuasion capabilities of a frontier large language model (LLM; Claude Sonnet 3.5) against incentivized human persuaders in an interactive, real-time conversational quiz setting. In this preregistered, large-scale incentivized experiment, participants (quiz takers) completed an online quiz where persuaders (either humans or LLMs) attempted to persuade quiz takers toward correct or incorrect answers. We find that LLM persuaders achieved significantly higher compliance with their directional persuasion attempts than incentivized human persuaders, demonstrating superior persuasive capabilities in both truthful (toward correct answers) and deceptive (toward incorrect answers) contexts. We also find that LLM persuaders significantly increased quiz takers' accuracy, leading to higher earnings, when steering quiz takers toward correct answers, and significantly decreased their accuracy, leading to lower earnings, when steering them toward incorrect answers. Overall, our findings suggest that AI's persuasion capabilities already exceed those of humans that have real-money bonuses tied to performance. Our findings of increasingly capable AI persuaders thus underscore the urgency of emerging alignment and governance frameworks.
>
---
#### [new 013] WorldPM: Scaling Human Preference Modeling
- **分类: cs.CL**

- **简介: 该论文研究人类偏好建模的扩展性（任务），探索如何利用数据和模型规模提升性能。通过收集公共论坛数据，训练1.5B-72B参数模型，发现对抗/客观指标随规模提升而增强，主观指标无此规律。提出WorldPM框架验证其扩展潜力，集成到强化学习微调（RLHF）中提升多项基准任务性能，泛化增益超5%，内部评估提升4-8%。**

- **链接: [http://arxiv.org/pdf/2505.10527v1](http://arxiv.org/pdf/2505.10527v1)**

> **作者:** Binghai Wang; Runji Lin; Keming Lu; Le Yu; Zhenru Zhang; Fei Huang; Chujie Zheng; Kai Dang; Yang Fan; Xingzhang Ren; An Yang; Binyuan Hui; Dayiheng Liu; Tao Gui; Qi Zhang; Xuanjing Huang; Yu-Gang Jiang; Bowen Yu; Jingren Zhou; Junyang Lin
>
> **摘要:** Motivated by scaling laws in language modeling that demonstrate how test loss scales as a power law with model and dataset sizes, we find that similar laws exist in preference modeling. We propose World Preference Modeling$ (WorldPM) to emphasize this scaling potential, where World Preference embodies a unified representation of human preferences. In this paper, we collect preference data from public forums covering diverse user communities, and conduct extensive training using 15M-scale data across models ranging from 1.5B to 72B parameters. We observe distinct patterns across different evaluation metrics: (1) Adversarial metrics (ability to identify deceptive features) consistently scale up with increased training data and base model size; (2) Objective metrics (objective knowledge with well-defined answers) show emergent behavior in larger language models, highlighting WorldPM's scalability potential; (3) Subjective metrics (subjective preferences from a limited number of humans or AI) do not demonstrate scaling trends. Further experiments validate the effectiveness of WorldPM as a foundation for preference fine-tuning. Through evaluations on 7 benchmarks with 20 subtasks, we find that WorldPM broadly improves the generalization performance across human preference datasets of varying sizes (7K, 100K and 800K samples), with performance gains exceeding 5% on many key subtasks. Integrating WorldPM into our internal RLHF pipeline, we observe significant improvements on both in-house and public evaluation sets, with notable gains of 4% to 8% in our in-house evaluations.
>
---
#### [new 014] VQ-Logits: Compressing the Output Bottleneck of Large Language Models via Vector Quantized Logits
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型输出层参数量大、计算成本高的问题。提出VQ-Logits方法，用向量量化将词汇表映射到小码本，大幅减少输出层参数（99%）并加速计算（6倍），仅轻微增加困惑度（4%）。核心是用共享码本替代传统大矩阵，通过散射机制预测词汇分布。**

- **链接: [http://arxiv.org/pdf/2505.10202v1](http://arxiv.org/pdf/2505.10202v1)**

> **作者:** Jintian Shao; Hongyi Huang; Jiayi Wu; YiMing Cheng; ZhiYu Wu; You Shan; MingKai Zheng
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success but face significant computational and memory challenges, particularly due to their extensive output vocabularies. The final linear projection layer, mapping hidden states to vocabulary-sized logits, often constitutes a substantial portion of the model's parameters and computational cost during inference. Existing methods like adaptive softmax or hierarchical softmax introduce structural complexities. In this paper, we propose VQ-Logits, a novel approach that leverages Vector Quantization (VQ) to drastically reduce the parameter count and computational load of the LLM output layer. VQ-Logits replaces the large V * dmodel output embedding matrix with a small, shared codebook of K embedding vectors (K << V ). Each token in the vocabulary is mapped to one of these K codebook vectors. The LLM predicts logits over this compact codebook, which are then efficiently "scattered" to the full vocabulary space using the learned or preassigned mapping. We demonstrate through extensive experiments on standard language modeling benchmarks (e.g., WikiText-103, C4) that VQ-Logits can achieve up to 99% parameter reduction in the output layer and 6x speedup in logit computation, with only a marginal 4% increase in perplexity compared to full softmax baselines. We further provide detailed ablation studies on codebook size, initialization, and learning strategies, showcasing the robustness and effectiveness of our approach.
>
---
#### [new 015] Coherent Language Reconstruction from Brain Recordings with Flexible Multi-Modal Input Stimuli
- **分类: cs.CL**

- **简介: 该论文属于多模态语言重建任务，旨在解决传统脑信号解码依赖单模态输入的问题。通过构建灵活框架整合视觉、听觉和文本多模态刺激，利用视觉语言模型与模态专家协同解码大脑记录，实现连贯语言重建。实验验证了方法的有效性和扩展性，推进了脑机交互的生态化应用。**

- **链接: [http://arxiv.org/pdf/2505.10356v1](http://arxiv.org/pdf/2505.10356v1)**

> **作者:** Chunyu Ye; Shaonan Wang
>
> **摘要:** Decoding thoughts from brain activity offers valuable insights into human cognition and enables promising applications in brain-computer interaction. While prior studies have explored language reconstruction from fMRI data, they are typically limited to single-modality inputs such as images or audio. In contrast, human thought is inherently multimodal. To bridge this gap, we propose a unified and flexible framework for reconstructing coherent language from brain recordings elicited by diverse input modalities-visual, auditory, and textual. Our approach leverages visual-language models (VLMs), using modality-specific experts to jointly interpret information across modalities. Experiments demonstrate that our method achieves performance comparable to state-of-the-art systems while remaining adaptable and extensible. This work advances toward more ecologically valid and generalizable mind decoding.
>
---
#### [new 016] Are LLM-generated plain language summaries truly understandable? A large-scale crowdsourced evaluation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本生成评估任务，旨在验证大语言模型(LLM)生成的医学简明摘要(PLS)的真实可理解性。通过150人众包实验结合主观评分与客观测试，发现人工撰写的PLS虽与LLM产出的主观评分相当，但实际理解效果显著更优，且自动评估指标与人类判断脱节，揭示了现有评估体系的局限性。**

- **链接: [http://arxiv.org/pdf/2505.10409v1](http://arxiv.org/pdf/2505.10409v1)**

> **作者:** Yue Guo; Jae Ho Sohn; Gondy Leroy; Trevor Cohen
>
> **摘要:** Plain language summaries (PLSs) are essential for facilitating effective communication between clinicians and patients by making complex medical information easier for laypeople to understand and act upon. Large language models (LLMs) have recently shown promise in automating PLS generation, but their effectiveness in supporting health information comprehension remains unclear. Prior evaluations have generally relied on automated scores that do not measure understandability directly, or subjective Likert-scale ratings from convenience samples with limited generalizability. To address these gaps, we conducted a large-scale crowdsourced evaluation of LLM-generated PLSs using Amazon Mechanical Turk with 150 participants. We assessed PLS quality through subjective Likert-scale ratings focusing on simplicity, informativeness, coherence, and faithfulness; and objective multiple-choice comprehension and recall measures of reader understanding. Additionally, we examined the alignment between 10 automated evaluation metrics and human judgments. Our findings indicate that while LLMs can generate PLSs that appear indistinguishable from human-written ones in subjective evaluations, human-written PLSs lead to significantly better comprehension. Furthermore, automated evaluation metrics fail to reflect human judgment, calling into question their suitability for evaluating PLSs. This is the first study to systematically evaluate LLM-generated PLSs based on both reader preferences and comprehension outcomes. Our findings highlight the need for evaluation frameworks that move beyond surface-level quality and for generation methods that explicitly optimize for layperson comprehension.
>
---
#### [new 017] Crossing Borders Without Crossing Boundaries: How Sociolinguistic Awareness Can Optimize User Engagement with Localized Spanish AI Models Across Hispanophone Countries
- **分类: cs.CL**

- **简介: 该论文属于AI语言模型本地化任务，旨在解决西班牙语方言差异导致的社会语言不协调问题。通过分析拉丁美洲和西班牙的书面西班牙变体差异，提出构建五种区域子方言模型，以优化本地化策略，增强用户信任与文化包容性，推动可持续用户增长。**

- **链接: [http://arxiv.org/pdf/2505.09902v1](http://arxiv.org/pdf/2505.09902v1)**

> **作者:** Martin Capdevila; Esteban Villa Turek; Ellen Karina Chumbe Fernandez; Luis Felipe Polo Galvez; Luis Cadavid; Andrea Marroquin; Rebeca Vargas Quesada; Johanna Crew; Nicole Vallejo Galarraga; Christopher Rodriguez; Diego Gutierrez; Radhi Datla
>
> **摘要:** Large language models are, by definition, based on language. In an effort to underscore the critical need for regional localized models, this paper examines primary differences between variants of written Spanish across Latin America and Spain, with an in-depth sociocultural and linguistic contextualization therein. We argue that these differences effectively constitute significant gaps in the quotidian use of Spanish among dialectal groups by creating sociolinguistic dissonances, to the extent that locale-sensitive AI models would play a pivotal role in bridging these divides. In doing so, this approach informs better and more efficient localization strategies that also serve to more adequately meet inclusivity goals, while securing sustainable active daily user growth in a major low-risk investment geographic area. Therefore, implementing at least the proposed five sub variants of Spanish addresses two lines of action: to foment user trust and reliance on AI language models while also demonstrating a level of cultural, historical, and sociolinguistic awareness that reflects positively on any internationalization strategy.
>
---
#### [new 018] VeriFact: Enhancing Long-Form Factuality Evaluation with Refined Fact Extraction and Reference Facts
- **分类: cs.CL**

- **简介: 该论文属于长文本生成的事实性评估任务，旨在解决现有方法无法有效捕捉上下文及关联事实、忽略召回率评估的问题。提出VeriFact框架改进事实提取完整性，并构建FactRBench基准，结合参考事实集实现精确率与召回率的综合测评。**

- **链接: [http://arxiv.org/pdf/2505.09701v1](http://arxiv.org/pdf/2505.09701v1)**

> **作者:** Xin Liu; Lechen Zhang; Sheza Munir; Yiyang Gu; Lu Wang
>
> **摘要:** Large language models (LLMs) excel at generating long-form responses, but evaluating their factuality remains challenging due to complex inter-sentence dependencies within the generated facts. Prior solutions predominantly follow a decompose-decontextualize-verify pipeline but often fail to capture essential context and miss key relational facts. In this paper, we introduce VeriFact, a factuality evaluation framework designed to enhance fact extraction by identifying and resolving incomplete and missing facts to support more accurate verification results. Moreover, we introduce FactRBench , a benchmark that evaluates both precision and recall in long-form model responses, whereas prior work primarily focuses on precision. FactRBench provides reference fact sets from advanced LLMs and human-written answers, enabling recall assessment. Empirical evaluations show that VeriFact significantly enhances fact completeness and preserves complex facts with critical relational information, resulting in more accurate factuality evaluation. Benchmarking various open- and close-weight LLMs on FactRBench indicate that larger models within same model family improve precision and recall, but high precision does not always correlate with high recall, underscoring the importance of comprehensive factuality assessment.
>
---
#### [new 019] J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM评估优化任务，旨在提升AI评判模型（LLM-as-a-Judge）的思维链推理能力和减少判断偏差。通过强化学习方法J1，将各类提示转化为含可验证奖励的评判任务，激励模型生成评估标准、对比参考答案并重新验证回答正确性。实验表明J1在8B/70B规模下超越同类模型，甚至优于更大模型的部分表现。**

- **链接: [http://arxiv.org/pdf/2505.10320v1](http://arxiv.org/pdf/2505.10320v1)**

> **作者:** Chenxi Whitehouse; Tianlu Wang; Ping Yu; Xian Li; Jason Weston; Ilia Kulikov; Swarnadeep Saha
>
> **备注:** 10 pages, 8 tables, 11 figures
>
> **摘要:** The progress of AI is bottlenecked by the quality of evaluation, and powerful LLM-as-a-Judge models have proved to be a core solution. Improved judgment ability is enabled by stronger chain-of-thought reasoning, motivating the need to find the best recipes for training such models to think. In this work we introduce J1, a reinforcement learning approach to training such models. Our method converts both verifiable and non-verifiable prompts to judgment tasks with verifiable rewards that incentivize thinking and mitigate judgment bias. In particular, our approach outperforms all other existing 8B or 70B models when trained at those sizes, including models distilled from DeepSeek-R1. J1 also outperforms o1-mini, and even R1 on some benchmarks, despite training a smaller model. We provide analysis and ablations comparing Pairwise-J1 vs Pointwise-J1 models, offline vs online training recipes, reward strategies, seed prompts, and variations in thought length and content. We find that our models make better judgments by learning to outline evaluation criteria, comparing against self-generated reference answers, and re-evaluating the correctness of model responses.
>
---
#### [new 020] RAIDEN-R1: Improving Role-awareness of LLMs via GRPO with Verifiable Reward
- **分类: cs.CL**

- **简介: 该论文属于角色扮演对话代理（RPCA）任务，旨在解决角色一致性不足的问题。提出RAIDEN-R1框架，通过可验证角色感知奖励（VRAR）和GRPO强化学习方法量化角色关键点评估，构建多LLM协作的思维链数据集提升推理连贯性。实验显示其14B模型在基准测试中准确率超88%，优于基线模型，并增强了上下文冲突解决与角色叙事稳定性。**

- **链接: [http://arxiv.org/pdf/2505.10218v1](http://arxiv.org/pdf/2505.10218v1)**

> **作者:** Zongsheng Wang; Kaili Sun; Bowen Wu; Qun Yu; Ying Li; Baoxun Wang
>
> **摘要:** Role-playing conversational agents (RPCAs) face persistent challenges in maintaining role consistency. To address this, we propose RAIDEN-R1, a novel reinforcement learning framework that integrates Verifiable Role-Awareness Reward (VRAR). The method introduces both singular and multi-term mining strategies to generate quantifiable rewards by assessing role-specific keys. Additionally, we construct a high-quality, role-aware Chain-of-Thought dataset through multi-LLM collaboration, and implement experiments to enhance reasoning coherence. Experiments on the RAIDEN benchmark demonstrate RAIDEN-R1's superiority: our 14B-GRPO model achieves 88.04% and 88.65% accuracy on Script-Based Knowledge and Conversation Memory metrics, respectively, outperforming baseline models while maintaining robustness. Case analyses further reveal the model's enhanced ability to resolve conflicting contextual cues and sustain first-person narrative consistency. This work bridges the non-quantifiability gap in RPCA training and provides insights into role-aware reasoning patterns, advancing the development of RPCAs.
>
---
#### [new 021] What Does Neuro Mean to Cardio? Investigating the Role of Clinical Specialty Data in Medical LLMs
- **分类: cs.CL**

- **简介: 该论文属于医学问答任务，研究临床专科数据对医学大语言模型的影响。通过构建S-MedQA数据集验证知识注入假设，发现专科训练未必提升对应领域表现，模型改进主要源于领域迁移而非专科知识注入，主张重新评估微调数据作用，并开源了数据集和代码。**

- **链接: [http://arxiv.org/pdf/2505.10113v1](http://arxiv.org/pdf/2505.10113v1)**

> **作者:** Xinlan Yan; Di Wu; Yibin Lei; Christof Monz; Iacer Calixto
>
> **摘要:** In this paper, we introduce S-MedQA, an English medical question-answering (QA) dataset for benchmarking large language models in fine-grained clinical specialties. We use S-MedQA to check the applicability of a popular hypothesis related to knowledge injection in the knowledge-intense scenario of medical QA, and show that: 1) training on data from a speciality does not necessarily lead to best performance on that specialty and 2) regardless of the specialty fine-tuned on, token probabilities of clinically relevant terms for all specialties increase consistently. Thus, we believe improvement gains come mostly from domain shifting (e.g., general to medical) rather than knowledge injection and suggest rethinking the role of fine-tuning data in the medical domain. We release S-MedQA and all code needed to reproduce all our experiments to the research community.
>
---
#### [new 022] Mining Hidden Thoughts from Texts: Evaluating Continual Pretraining with Synthetic Data for LLM Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型的持续预训练方法，解决传统推理训练数据受限、跨领域泛化差的问题。提出Reasoning CPT，利用合成数据（STEM/法律语料隐含思维）重构文本生成逻辑，在Gemma2-9B上验证。实验表明该方法提升MMLU基准全领域性能，难题表现提升8%，且具备跨领域推理迁移能力。**

- **链接: [http://arxiv.org/pdf/2505.10182v1](http://arxiv.org/pdf/2505.10182v1)**

> **作者:** Yoichi Ishibashi; Taro Yano; Masafumi Oyamada
>
> **摘要:** Large Language Models (LLMs) have demonstrated significant improvements in reasoning capabilities through supervised fine-tuning and reinforcement learning. However, when training reasoning models, these approaches are primarily applicable to specific domains such as mathematics and programming, which imposes fundamental constraints on the breadth and scalability of training data. In contrast, continual pretraining (CPT) offers the advantage of not requiring task-specific signals. Nevertheless, how to effectively synthesize training data for reasoning and how such data affect a wide range of domains remain largely unexplored. This study provides a detailed evaluation of Reasoning CPT, a form of CPT that uses synthetic data to reconstruct the hidden thought processes underlying texts, based on the premise that texts are the result of the author's thinking process. Specifically, we apply Reasoning CPT to Gemma2-9B using synthetic data with hidden thoughts derived from STEM and Law corpora, and compare it to standard CPT on the MMLU benchmark. Our analysis reveals that Reasoning CPT consistently improves performance across all evaluated domains. Notably, reasoning skills acquired in one domain transfer effectively to others; the performance gap with conventional methods widens as problem difficulty increases, with gains of up to 8 points on the most challenging problems. Furthermore, models trained with hidden thoughts learn to adjust the depth of their reasoning according to problem difficulty.
>
---
#### [new 023] XRAG: Cross-lingual Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文提出XRAG基准，用于评估大语言模型在跨语言检索增强生成中的表现，解决用户语言与检索结果不匹配时的生成难题。通过构建基于新闻的多语言数据集，分析单语/多语检索场景，揭示模型在语言正确性和跨语言推理的缺陷，为LLM推理能力研究提供新方向。**

- **链接: [http://arxiv.org/pdf/2505.10089v1](http://arxiv.org/pdf/2505.10089v1)**

> **作者:** Wei Liu; Sony Trenous; Leonardo F. R. Ribeiro; Bill Byrne; Felix Hieber
>
> **摘要:** We propose XRAG, a novel benchmark designed to evaluate the generation abilities of LLMs in cross-lingual Retrieval-Augmented Generation (RAG) settings where the user language does not match the retrieval results. XRAG is constructed from recent news articles to ensure that its questions require external knowledge to be answered. It covers the real-world scenarios of monolingual and multilingual retrieval, and provides relevancy annotations for each retrieved document. Our novel dataset construction pipeline results in questions that require complex reasoning, as evidenced by the significant gap between human and LLM performance. Consequently, XRAG serves as a valuable benchmark for studying LLM reasoning abilities, even before considering the additional cross-lingual complexity. Experimental results on five LLMs uncover two previously unreported challenges in cross-lingual RAG: 1) in the monolingual retrieval setting, all evaluated models struggle with response language correctness; 2) in the multilingual retrieval setting, the main challenge lies in reasoning over retrieved information across languages rather than generation of non-English text.
>
---
#### [new 024] Multi-domain Multilingual Sentiment Analysis in Industry: Predicting Aspect-based Opinion Quadruples
- **分类: cs.CL**

- **简介: 该论文属于多领域多语言方面级情感分析任务，旨在解决传统模型需针对单一领域单独训练的问题。研究通过微调大语言模型，构建统一模型同时提取文本中的四元组（方面类别、情感极性、目标及观点表达），验证了多领域模型性能接近单领域专用模型，并降低了部署复杂度，还探讨了非抽取式预测与错误评估方法。**

- **链接: [http://arxiv.org/pdf/2505.10389v1](http://arxiv.org/pdf/2505.10389v1)**

> **作者:** Benjamin White; Anastasia Shimorina
>
> **摘要:** This paper explores the design of an aspect-based sentiment analysis system using large language models (LLMs) for real-world use. We focus on quadruple opinion extraction -- identifying aspect categories, sentiment polarity, targets, and opinion expressions from text data across different domains and languages. Using internal datasets, we investigate whether a single fine-tuned model can effectively handle multiple domain-specific taxonomies simultaneously. We demonstrate that a combined multi-domain model achieves performance comparable to specialized single-domain models while reducing operational complexity. We also share lessons learned for handling non-extractive predictions and evaluating various failure modes when developing LLM-based systems for structured prediction tasks.
>
---
#### [new 025] The Devil Is in the Word Alignment Details: On Translation-Based Cross-Lingual Transfer for Token Classification Tasks
- **分类: cs.CL**

- **简介: 该论文研究跨语言迁移（XLT）中基于翻译的标记分类任务，聚焦标签投影问题。通过系统分析词对齐工具（WA）的低层设计（如标签映射算法、噪声过滤及预分词策略），优化WA性能，使其达到与标记方法相当水平，并提出集成翻译训练与测试预测的新策略，显著提升效果并增强鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.10507v1](http://arxiv.org/pdf/2505.10507v1)**

> **作者:** Benedikt Ebing; Goran Glavaš
>
> **摘要:** Translation-based strategies for cross-lingual transfer XLT such as translate-train -- training on noisy target language data translated from the source language -- and translate-test -- evaluating on noisy source language data translated from the target language -- are competitive XLT baselines. In XLT for token classification tasks, however, these strategies include label projection, the challenging step of mapping the labels from each token in the original sentence to its counterpart(s) in the translation. Although word aligners (WAs) are commonly used for label projection, the low-level design decisions for applying them to translation-based XLT have not been systematically investigated. Moreover, recent marker-based methods, which project labeled spans by inserting tags around them before (or after) translation, claim to outperform WAs in label projection for XLT. In this work, we revisit WAs for label projection, systematically investigating the effects of low-level design decisions on token-level XLT: (i) the algorithm for projecting labels between (multi-)token spans, (ii) filtering strategies to reduce the number of noisily mapped labels, and (iii) the pre-tokenization of the translated sentences. We find that all of these substantially impact translation-based XLT performance and show that, with optimized choices, XLT with WA offers performance at least comparable to that of marker-based methods. We then introduce a new projection strategy that ensembles translate-train and translate-test predictions and demonstrate that it substantially outperforms the marker-based projection. Crucially, we show that our proposed ensembling also reduces sensitivity to low-level WA design choices, resulting in more robust XLT for token classification tasks.
>
---
#### [new 026] GE-Chat: A Graph Enhanced RAG Framework for Evidential Response Generation of LLMs
- **分类: cs.CL; 68T50, 68T30; I.2.7; I.2.4; H.3.3**

- **简介: 该论文属于可信AI任务，旨在解决大语言模型（LLM）生成不可靠或虚假回答的问题。通过构建知识图谱增强的检索生成框架（GE-Chat），结合链式推理、多跳子图检索和蕴含式句子生成，提升回答的证据支持能力，使结论可溯源以增强可信度。**

- **链接: [http://arxiv.org/pdf/2505.10143v1](http://arxiv.org/pdf/2505.10143v1)**

> **作者:** Longchao Da; Parth Mitesh Shah; Kuan-Ru Liou; Jiaxing Zhang; Hua Wei
>
> **备注:** 5 pages, 4 figures, accepted to IJCAI2025 demo track
>
> **摘要:** Large Language Models are now key assistants in human decision-making processes. However, a common note always seems to follow: "LLMs can make mistakes. Be careful with important info." This points to the reality that not all outputs from LLMs are dependable, and users must evaluate them manually. The challenge deepens as hallucinated responses, often presented with seemingly plausible explanations, create complications and raise trust issues among users. To tackle such issue, this paper proposes GE-Chat, a knowledge Graph enhanced retrieval-augmented generation framework to provide Evidence-based response generation. Specifically, when the user uploads a material document, a knowledge graph will be created, which helps construct a retrieval-augmented agent, enhancing the agent's responses with additional knowledge beyond its training corpus. Then we leverage Chain-of-Thought (CoT) logic generation, n-hop sub-graph searching, and entailment-based sentence generation to realize accurate evidence retrieval. We demonstrate that our method improves the existing models' performance in terms of identifying the exact evidence in a free-form context, providing a reliable way to examine the resources of LLM's conclusion and help with the judgment of the trustworthiness.
>
---
#### [new 027] DIF: A Framework for Benchmarking and Verifying Implicit Bias in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的公平性评估任务，旨在解决大型语言模型（LLMs）隐式偏见缺乏标准化评测方法的问题。作者提出DIF框架，通过结合社会人口统计角色与逻辑/数学问题数据集，量化模型偏见，验证偏见存在及其与回答准确性的负相关性。**

- **链接: [http://arxiv.org/pdf/2505.10013v1](http://arxiv.org/pdf/2505.10013v1)**

> **作者:** Lake Yin; Fan Huang
>
> **备注:** 7 pages, 1 figure
>
> **摘要:** As Large Language Models (LLMs) have risen in prominence over the past few years, there has been concern over the potential biases in LLMs inherited from the training data. Previous studies have examined how LLMs exhibit implicit bias, such as when response generation changes when different social contexts are introduced. We argue that this implicit bias is not only an ethical, but also a technical issue, as it reveals an inability of LLMs to accommodate extraneous information. However, unlike other measures of LLM intelligence, there are no standard methods to benchmark this specific subset of LLM bias. To bridge this gap, we developed a method for calculating an easily interpretable benchmark, DIF (Demographic Implicit Fairness), by evaluating preexisting LLM logic and math problem datasets with sociodemographic personas. We demonstrate that this method can statistically validate the presence of implicit bias in LLM behavior and find an inverse trend between question answering accuracy and implicit bias, supporting our argument.
>
---
#### [new 028] Multi-Token Prediction Needs Registers
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于语言模型优化任务，旨在提升多令牌预测在微调等场景的泛化性。提出MuToR方法：通过插入可学习的寄存器令牌预测多步目标，保持与原模型兼容性，仅少量参数即可支持扩展预测范围，并在语言/视觉任务的监督微调、PEFT及预训练中验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.10518v1](http://arxiv.org/pdf/2505.10518v1)**

> **作者:** Anastasios Gerontopoulos; Spyros Gidaris; Nikos Komodakis
>
> **摘要:** Multi-token prediction has emerged as a promising objective for improving language model pretraining, but its benefits have not consistently generalized to other settings such as fine-tuning. In this paper, we propose MuToR, a simple and effective approach to multi-token prediction that interleaves learnable register tokens into the input sequence, each tasked with predicting future targets. Compared to existing methods, MuToR offers several key advantages: it introduces only a negligible number of additional parameters, requires no architectural changes--ensuring compatibility with off-the-shelf pretrained language models--and remains aligned with the next-token pretraining objective, making it especially well-suited for supervised fine-tuning. Moreover, it naturally supports scalable prediction horizons. We demonstrate the effectiveness and versatility of MuToR across a range of use cases, including supervised fine-tuning, parameter-efficient fine-tuning (PEFT), and pretraining, on challenging generative tasks in both language and vision domains. Our code will be available at: https://github.com/nasosger/MuToR.
>
---
#### [new 029] Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）能否预测暴力冲突，属于冲突趋势预测任务。通过对比参数化知识（预训练权重）与非参数化能力（结合外部数据检索增强），评估LLMs在非洲和中东地区的冲突趋势与伤亡预测效果，揭示模型依赖内部知识或外部信息的性能差异及优劣。**

- **链接: [http://arxiv.org/pdf/2505.09852v1](http://arxiv.org/pdf/2505.09852v1)**

> **作者:** Apollinaire Poli Nemkova; Sarath Chandra Lingareddy; Sagnik Ray Choudhury; Mark V. Albert
>
> **摘要:** Large Language Models (LLMs) have shown impressive performance across natural language tasks, but their ability to forecast violent conflict remains underexplored. We investigate whether LLMs possess meaningful parametric knowledge-encoded in their pretrained weights-to predict conflict escalation and fatalities without external data. This is critical for early warning systems, humanitarian planning, and policy-making. We compare this parametric knowledge with non-parametric capabilities, where LLMs access structured and unstructured context from conflict datasets (e.g., ACLED, GDELT) and recent news reports via Retrieval-Augmented Generation (RAG). Incorporating external information could enhance model performance by providing up-to-date context otherwise missing from pretrained weights. Our two-part evaluation framework spans 2020-2024 across conflict-prone regions in the Horn of Africa and the Middle East. In the parametric setting, LLMs predict conflict trends and fatalities relying only on pretrained knowledge. In the non-parametric setting, models receive summaries of recent conflict events, indicators, and geopolitical developments. We compare predicted conflict trend labels (e.g., Escalate, Stable Conflict, De-escalate, Peace) and fatalities against historical data. Our findings highlight the strengths and limitations of LLMs for conflict forecasting and the benefits of augmenting them with structured external knowledge.
>
---
#### [new 030] CL-RAG: Bridging the Gap in Retrieval-Augmented Generation with Curriculum Learning
- **分类: cs.CL**

- **简介: 该论文针对检索增强生成（RAG）系统中检索文档质量参差不齐影响训练的问题，提出基于课程学习的多阶段训练框架CL-RAG。通过构建分难度样本并分阶段优化检索器和生成器，提升系统泛化能力，在开放域QA任务中性能优于现有方法2%-4%。**

- **链接: [http://arxiv.org/pdf/2505.10493v1](http://arxiv.org/pdf/2505.10493v1)**

> **作者:** Shaohan Wang; Licheng Zhang; Zheren Fu; Zhendong Mao
>
> **摘要:** Retrieval-Augmented Generation (RAG) is an effective method to enhance the capabilities of large language models (LLMs). Existing methods focus on optimizing the retriever or generator in the RAG system by directly utilizing the top-k retrieved documents. However, the documents effectiveness are various significantly across user queries, i.e. some documents provide valuable knowledge while others totally lack critical information. It hinders the retriever and generator's adaptation during training. Inspired by human cognitive learning, curriculum learning trains models using samples progressing from easy to difficult, thus enhancing their generalization ability, and we integrate this effective paradigm to the training of the RAG system. In this paper, we propose a multi-stage Curriculum Learning based RAG system training framework, named CL-RAG. We first construct training data with multiple difficulty levels for the retriever and generator separately through sample evolution. Then, we train the model in stages based on the curriculum learning approach, thereby optimizing the overall performance and generalization of the RAG system more effectively. Our CL-RAG framework demonstrates consistent effectiveness across four open-domain QA datasets, achieving performance gains of 2% to 4% over multiple advanced methods.
>
---
#### [new 031] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文提出扩散横向思维链（DCoLT），用于增强扩散语言模型的推理能力。针对传统链式思维（CoT）线性推理的局限，DCoLT通过反向扩散过程将中间步骤建模为潜在思考动作，利用基于结果的强化学习优化推理轨迹，支持双向非线性推理并放宽语法限制。实验表明，该方法在数学和代码生成任务中显著提升扩散模型性能，强化后的LLaDA模型在多个基准任务上准确率最高提升19.5%。**

- **链接: [http://arxiv.org/pdf/2505.10446v1](http://arxiv.org/pdf/2505.10446v1)**

> **作者:** Zemin Huang; Zhiyang Chen; Zijun Wang; Tiancheng Li; Guo-Jun Qi
>
> **摘要:** We introduce the \emph{Diffusion Chain of Lateral Thought (DCoLT)}, a reasoning framework for diffusion language models. DCoLT treats each intermediate step in the reverse diffusion process as a latent "thinking" action and optimizes the entire reasoning trajectory to maximize the reward on the correctness of the final answer with outcome-based Reinforcement Learning (RL). Unlike traditional Chain-of-Thought (CoT) methods that follow a causal, linear thinking process, DCoLT allows bidirectional, non-linear reasoning with no strict rule on grammatical correctness amid its intermediate steps of thought. We implement DCoLT on two representative Diffusion Language Models (DLMs). First, we choose SEDD as a representative continuous-time discrete diffusion model, where its concrete score derives a probabilistic policy to maximize the RL reward over the entire sequence of intermediate diffusion steps. We further consider the discrete-time masked diffusion language model -- LLaDA, and find that the order to predict and unmask tokens plays an essential role to optimize its RL action resulting from the ranking-based Unmasking Policy Module (UPM) defined by the Plackett-Luce model. Experiments on both math and code generation tasks show that using only public data and 16 H800 GPUs, DCoLT-reinforced DLMs outperform other DLMs trained by SFT or RL or even both. Notably, DCoLT-reinforced LLaDA boosts its reasoning accuracy by +9.8%, +5.7%, +11.4%, +19.5% on GSM8K, MATH, MBPP, and HumanEval.
>
---
#### [new 032] Rethinking Repetition Problems of LLMs in Code Generation
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **简介: 该论文针对大语言模型（LLMs）在代码生成任务中出现的结构性重复问题，提出基于语法规则的解码方法RPG。通过语法分析识别重复模式，衰减关键token概率抑制重复生成，并构建CodeRepetEval数据集验证有效性，显著降低了代码冗余并提升生成质量。**

- **链接: [http://arxiv.org/pdf/2505.10402v1](http://arxiv.org/pdf/2505.10402v1)**

> **作者:** Yihong Dong; Yuchen Liu; Xue Jiang; Zhi Jin; Ge Li
>
> **备注:** Accepted to ACL 2025 (main)
>
> **摘要:** With the advent of neural language models, the performance of code generation has been significantly boosted. However, the problem of repetitions during the generation process continues to linger. Previous work has primarily focused on content repetition, which is merely a fraction of the broader repetition problem in code generation. A more prevalent and challenging problem is structural repetition. In structural repetition, the repeated code appears in various patterns but possesses a fixed structure, which can be inherently reflected in grammar. In this paper, we formally define structural repetition and propose an efficient decoding approach called RPG, which stands for Repetition Penalization based on Grammar, to alleviate the repetition problems in code generation for LLMs. Specifically, RPG first leverages grammar rules to identify repetition problems during code generation, and then strategically decays the likelihood of critical tokens that contribute to repetitions, thereby mitigating them in code generation. To facilitate this study, we construct a new dataset CodeRepetEval to comprehensively evaluate approaches for mitigating the repetition problems in code generation. Extensive experimental results demonstrate that RPG substantially outperforms the best-performing baselines on CodeRepetEval dataset as well as HumanEval and MBPP benchmarks, effectively reducing repetitions and enhancing the quality of generated code.
>
---
#### [new 033] The Evolving Landscape of Generative Large Language Models and Traditional Natural Language Processing in Medicine
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对比分析任务，旨在明确生成式大语言模型与传统NLP在医学应用中的差异。通过分析19,123项研究，发现生成式LLMs擅长开放型任务，传统NLP在信息提取与分析更具优势，同时强调技术发展中伦理规范的必要性。**

- **链接: [http://arxiv.org/pdf/2505.10261v1](http://arxiv.org/pdf/2505.10261v1)**

> **作者:** Rui Yang; Huitao Li; Matthew Yu Heng Wong; Yuhe Ke; Xin Li; Kunyu Yu; Jingchi Liao; Jonathan Chong Kai Liew; Sabarinath Vinod Nair; Jasmine Chiat Ling Ong; Irene Li; Douglas Teodoro; Chuan Hong; Daniel Shu Wei Ting; Nan Liu
>
> **摘要:** Natural language processing (NLP) has been traditionally applied to medicine, and generative large language models (LLMs) have become prominent recently. However, the differences between them across different medical tasks remain underexplored. We analyzed 19,123 studies, finding that generative LLMs demonstrate advantages in open-ended tasks, while traditional NLP dominates in information extraction and analysis tasks. As these technologies advance, ethical use of them is essential to ensure their potential in medical applications.
>
---
#### [new 034] The CoT Encyclopedia: Analyzing, Predicting, and Controlling how a Reasoning Model will Think
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型推理分析任务，旨在解决传统方法无法全面理解模型思维策略的问题。提出了CoT Encyclopedia框架，通过自动提取、聚类和解释推理模式，实现策略预测与控制，并揭示训练数据格式对推理行为的关键影响。**

- **链接: [http://arxiv.org/pdf/2505.10185v1](http://arxiv.org/pdf/2505.10185v1)**

> **作者:** Seongyun Lee; Seungone Kim; Minju Seo; Yongrae Jo; Dongyoung Go; Hyeonbin Hwang; Jinho Park; Xiang Yue; Sean Welleck; Graham Neubig; Moontae Lee; Minjoon Seo
>
> **备注:** Work in progress
>
> **摘要:** Long chain-of-thought (CoT) is an essential ingredient in effective usage of modern large language models, but our understanding of the reasoning strategies underlying these capabilities remains limited. While some prior works have attempted to categorize CoTs using predefined strategy types, such approaches are constrained by human intuition and fail to capture the full diversity of model behaviors. In this work, we introduce the CoT Encyclopedia, a bottom-up framework for analyzing and steering model reasoning. Our method automatically extracts diverse reasoning criteria from model-generated CoTs, embeds them into a semantic space, clusters them into representative categories, and derives contrastive rubrics to interpret reasoning behavior. Human evaluations show that this framework produces more interpretable and comprehensive analyses than existing methods. Moreover, we demonstrate that this understanding enables performance gains: we can predict which strategy a model is likely to use and guide it toward more effective alternatives. Finally, we provide practical insights, such as that training data format (e.g., free-form vs. multiple-choice) has a far greater impact on reasoning behavior than data domain, underscoring the importance of format-aware model design.
>
---
#### [new 035] Rethinking Prompt Optimizers: From Prompt Merits to Optimization
- **分类: cs.CL**

- **简介: 该论文属于提示优化任务，旨在解决现有方法依赖大模型导致轻量模型效果下降、成本高的问题。作者提出MePO优化器，通过定义通用质量指标并训练轻量模型生成简洁提示，实现跨模型兼容、低成本部署，实验证明其有效性。**

- **链接: [http://arxiv.org/pdf/2505.09930v1](http://arxiv.org/pdf/2505.09930v1)**

> **作者:** Zixiao Zhu; Hanzhang Zhou; Zijian Feng; Tianjiao Li; Chua Jia Jim Deryl; Mak Lee Onn; Gee Wah Ng; Kezhi Mao
>
> **备注:** 20 pages, 14 figures
>
> **摘要:** Prompt optimization (PO) offers a practical alternative to fine-tuning large language models (LLMs), enabling performance improvements without altering model weights. Existing methods typically rely on advanced, large-scale LLMs like GPT-4 to generate optimized prompts. However, due to limited downward compatibility, verbose, instruction-heavy prompts from advanced LLMs can overwhelm lightweight inference models and degrade response quality. In this work, we rethink prompt optimization through the lens of interpretable design. We first identify a set of model-agnostic prompt quality merits and empirically validate their effectiveness in enhancing prompt and response quality. We then introduce MePO, a merit-guided, lightweight, and locally deployable prompt optimizer trained on our preference dataset built from merit-aligned prompts generated by a lightweight LLM. Unlike prior work, MePO avoids online optimization reliance, reduces cost and privacy concerns, and, by learning clear, interpretable merits, generalizes effectively to both large-scale and lightweight inference models. Experiments demonstrate that MePO achieves better results across diverse tasks and model types, offering a scalable and robust solution for real-world deployment. Our model and dataset are available at: https://github.com/MidiyaZhu/MePO
>
---
#### [new 036] Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究提升预训练语言模型的分词器灵活性，解决固定分词方案导致的多语言/专业场景低效问题。提出Tokenadapt混合启发式初始化新词嵌入（局部子词分解+全局语义相似），并引入超令牌学习增强压缩，减少再训练需求，在困惑度等指标上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.09738v1](http://arxiv.org/pdf/2505.09738v1)**

> **作者:** Shaurya Sharthak; Vinayak Pahalwan; Adithya Kamath; Adarsh Shirawalmath
>
> **摘要:** Pretrained language models (LLMs) are often constrained by their fixed tokenization schemes, leading to inefficiencies and performance limitations, particularly for multilingual or specialized applications. This tokenizer lock-in presents significant challenges. standard methods to overcome this often require prohibitive computational resources. Although tokenizer replacement with heuristic initialization aims to reduce this burden, existing methods often require exhaustive residual fine-tuning and still may not fully preserve semantic nuances or adequately address the underlying compression inefficiencies. Our framework introduces two innovations: first, Tokenadapt, a model-agnostic tokenizer transplantation method, and second, novel pre-tokenization learning for multi-word Supertokens to enhance compression and reduce fragmentation. Tokenadapt initializes new unique token embeddings via a hybrid heuristic that combines two methods: a local estimate based on subword decomposition using the old tokenizer, and a global estimate utilizing the top-k semantically similar tokens from the original vocabulary. This methodology aims to preserve semantics while significantly minimizing retraining requirements. Empirical investigations validate both contributions: the transplantation heuristic successfully initializes unique tokens, markedly outperforming conventional baselines and sophisticated methods including Transtokenizer and ReTok, while our Supertokens achieve notable compression gains. Our zero-shot perplexity results demonstrate that the TokenAdapt hybrid initialization consistently yields lower perplexity ratios compared to both ReTok and TransTokenizer baselines across different base models and newly trained target tokenizers. TokenAdapt typically reduced the overall perplexity ratio significantly compared to ReTok, yielding at least a 2-fold improvement in these aggregate scores.
>
---
#### [new 037] CAFE: Retrieval Head-based Coarse-to-Fine Information Seeking to Enhance Multi-Document QA Capability
- **分类: cs.CL**

- **简介: 该论文针对多文档问答任务，解决大模型在长上下文中检索精度与召回率失衡的问题，提出两阶段方法CAFE：先粗粒度过滤相关文档，再细粒度引导注意力聚焦证据内容，减少干扰信息影响，实验显示其在基准测试中显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.10063v1](http://arxiv.org/pdf/2505.10063v1)**

> **作者:** Han Peng; Jinhao Jiang; Zican Dong; Wayne Xin Zhao; Lei Fang
>
> **摘要:** Advancements in Large Language Models (LLMs) have extended their input context length, yet they still struggle with retrieval and reasoning in long-context inputs. Existing methods propose to utilize the prompt strategy and retrieval head to alleviate this limitation. However, they still face challenges in balancing retrieval precision and recall, impacting their efficacy in answering questions. To address this, we introduce $\textbf{CAFE}$, a two-stage coarse-to-fine method to enhance multi-document question-answering capacities. By gradually eliminating the negative impacts of background and distracting documents, CAFE makes the responses more reliant on the evidence documents. Initially, a coarse-grained filtering method leverages retrieval heads to identify and rank relevant documents. Then, a fine-grained steering method guides attention to the most relevant content. Experiments across benchmarks show CAFE outperforms baselines, achieving up to 22.1% and 13.7% SubEM improvement over SFT and RAG methods on the Mistral model, respectively.
>
---
#### [new 038] System Prompt Optimization with Meta-Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型的系统提示优化，属于元学习任务。针对现有方法忽视跨任务通用系统提示的问题，提出双层优化框架：通过多数据集元学习系统提示，同时迭代更新用户提示以增强协同性。实验表明优化后的系统提示能泛化至新任务，并提升少步调优性能。**

- **链接: [http://arxiv.org/pdf/2505.09666v1](http://arxiv.org/pdf/2505.09666v1)**

> **作者:** Yumin Choi; Jinheon Baek; Sung Ju Hwang
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities, with optimizing their input prompts playing a pivotal role in maximizing their performance. However, while LLM prompts consist of both the task-agnostic system prompts and task-specific user prompts, existing work on prompt optimization has focused on user prompts specific to individual queries or tasks, and largely overlooked the system prompt that is, once optimized, applicable across different tasks and domains. Motivated by this, we introduce the novel problem of bilevel system prompt optimization, whose objective is to design system prompts that are robust to diverse user prompts and transferable to unseen tasks. To tackle this problem, we then propose a meta-learning framework, which meta-learns the system prompt by optimizing it over various user prompts across multiple datasets, while simultaneously updating the user prompts in an iterative manner to ensure synergy between them. We conduct experiments on 14 unseen datasets spanning 5 different domains, on which we show that our approach produces system prompts that generalize effectively to diverse user prompts. Also, our findings reveal that the optimized system prompt enables rapid adaptation even to unseen tasks, requiring fewer optimization steps for test-time user prompts while achieving improved performance.
>
---
#### [new 039] From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models
- **分类: cs.CL; cs.CR**

- **简介: 该论文针对大语言模型生成文本的滥用问题，提出一种协同水印框架以平衡鲁棒性、质量和安全性。通过融合基于logits和采样的水印方法，设计串行、并行及混合策略，利用熵自适应嵌入水印，实验表明其性能超越现有方案，实现多指标优化。**

- **链接: [http://arxiv.org/pdf/2505.09924v1](http://arxiv.org/pdf/2505.09924v1)**

> **作者:** Yidan Wang; Yubing Ren; Yanan Cao; Binxing Fang
>
> **摘要:** The rise of Large Language Models (LLMs) has heightened concerns about the misuse of AI-generated text, making watermarking a promising solution. Mainstream watermarking schemes for LLMs fall into two categories: logits-based and sampling-based. However, current schemes entail trade-offs among robustness, text quality, and security. To mitigate this, we integrate logits-based and sampling-based schemes, harnessing their respective strengths to achieve synergy. In this paper, we propose a versatile symbiotic watermarking framework with three strategies: serial, parallel, and hybrid. The hybrid framework adaptively embeds watermarks using token entropy and semantic entropy, optimizing the balance between detectability, robustness, text quality, and security. Furthermore, we validate our approach through comprehensive experiments on various datasets and models. Experimental results indicate that our method outperforms existing baselines and achieves state-of-the-art (SOTA) performance. We believe this framework provides novel insights into diverse watermarking paradigms. Our code is available at \href{https://github.com/redwyd/SymMark}{https://github.com/redwyd/SymMark}.
>
---
#### [new 040] Comparing LLM Text Annotation Skills: A Study on Human Rights Violations in Social Media Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多个先进大语言模型（LLMs）在俄乌社交媒体数据中识别人权侵犯的二分类标注任务，通过零样本/小样本测试对比其与人类标注的差异，分析模型跨语言适应性和错误模式，评估LLMs在敏感多语言场景的可靠性及主观语境处理能力。**

- **链接: [http://arxiv.org/pdf/2505.10260v1](http://arxiv.org/pdf/2505.10260v1)**

> **作者:** Poli Apollinaire Nemkova; Solomon Ubani; Mark V. Albert
>
> **摘要:** In the era of increasingly sophisticated natural language processing (NLP) systems, large language models (LLMs) have demonstrated remarkable potential for diverse applications, including tasks requiring nuanced textual understanding and contextual reasoning. This study investigates the capabilities of multiple state-of-the-art LLMs - GPT-3.5, GPT-4, LLAMA3, Mistral 7B, and Claude-2 - for zero-shot and few-shot annotation of a complex textual dataset comprising social media posts in Russian and Ukrainian. Specifically, the focus is on the binary classification task of identifying references to human rights violations within the dataset. To evaluate the effectiveness of these models, their annotations are compared against a gold standard set of human double-annotated labels across 1000 samples. The analysis includes assessing annotation performance under different prompting conditions, with prompts provided in both English and Russian. Additionally, the study explores the unique patterns of errors and disagreements exhibited by each model, offering insights into their strengths, limitations, and cross-linguistic adaptability. By juxtaposing LLM outputs with human annotations, this research contributes to understanding the reliability and applicability of LLMs for sensitive, domain-specific tasks in multilingual contexts. It also sheds light on how language models handle inherently subjective and context-dependent judgments, a critical consideration for their deployment in real-world scenarios.
>
---
#### [new 041] LDIR: Low-Dimensional Dense and Interpretable Text Embeddings with Relative Representations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义文本表示任务，旨在解决现有文本嵌入模型（如SimCSE）维度不可解释或高维的问题。提出LDIR方法，通过最远点采样生成低维（<500）密集向量，以锚文本语义相关度实现可解释性，在保持性能的同时超越可解释基线模型。**

- **链接: [http://arxiv.org/pdf/2505.10354v1](http://arxiv.org/pdf/2505.10354v1)**

> **作者:** Yile Wang; Zhanyu Shen; Hui Huang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Semantic text representation is a fundamental task in the field of natural language processing. Existing text embedding (e.g., SimCSE and LLM2Vec) have demonstrated excellent performance, but the values of each dimension are difficult to trace and interpret. Bag-of-words, as classic sparse interpretable embeddings, suffers from poor performance. Recently, Benara et al. (2024) propose interpretable text embeddings using large language models, which forms "0/1" embeddings based on responses to a series of questions. These interpretable text embeddings are typically high-dimensional (larger than 10,000). In this work, we propose Low-dimensional (lower than 500) Dense and Interpretable text embeddings with Relative representations (LDIR). The numerical values of its dimensions indicate semantic relatedness to different anchor texts through farthest point sampling, offering both semantic representation as well as a certain level of traceability and interpretability. We validate LDIR on multiple semantic textual similarity, retrieval, and clustering tasks. Extensive experimental results show that LDIR performs close to the black-box baseline models and outperforms the interpretable embeddings baselines with much fewer dimensions. Code is available at https://github.com/szu-tera/LDIR.
>
---
#### [new 042] Dark LLMs: The Growing Threat of Unaligned AI Models
- **分类: cs.CL; cs.AI; cs.CR; cs.LG; 68T50, 68T05, 68P25; I.2.7**

- **简介: 该论文属于AI安全领域，研究LLM的越狱漏洞问题。针对未对齐AI模型因训练数据含不良内容导致的伦理风险，提出通用攻击方法可突破主流模型的安全限制，揭示行业防护措施不足，呼吁加强监管防止技术滥用。**

- **链接: [http://arxiv.org/pdf/2505.10066v1](http://arxiv.org/pdf/2505.10066v1)**

> **作者:** Michael Fire; Yitzhak Elbazis; Adi Wasenstein; Lior Rokach
>
> **摘要:** Large Language Models (LLMs) rapidly reshape modern life, advancing fields from healthcare to education and beyond. However, alongside their remarkable capabilities lies a significant threat: the susceptibility of these models to jailbreaking. The fundamental vulnerability of LLMs to jailbreak attacks stems from the very data they learn from. As long as this training data includes unfiltered, problematic, or 'dark' content, the models can inherently learn undesirable patterns or weaknesses that allow users to circumvent their intended safety controls. Our research identifies the growing threat posed by dark LLMs models deliberately designed without ethical guardrails or modified through jailbreak techniques. In our research, we uncovered a universal jailbreak attack that effectively compromises multiple state-of-the-art models, enabling them to answer almost any question and produce harmful outputs upon request. The main idea of our attack was published online over seven months ago. However, many of the tested LLMs were still vulnerable to this attack. Despite our responsible disclosure efforts, responses from major LLM providers were often inadequate, highlighting a concerning gap in industry practices regarding AI safety. As model training becomes more accessible and cheaper, and as open-source LLMs proliferate, the risk of widespread misuse escalates. Without decisive intervention, LLMs may continue democratizing access to dangerous knowledge, posing greater risks than anticipated.
>
---
#### [new 043] Next Word Suggestion using Graph Neural Network
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的语言建模任务，聚焦上下文嵌入子任务。针对现有大规模模型依赖海量数据与算力的问题，提出结合图卷积网络（GNN）提取上下文特征与LSTM协同预测后续词的轻量化方法，在有限资源下验证了有效性。**

- **链接: [http://arxiv.org/pdf/2505.09649v1](http://arxiv.org/pdf/2505.09649v1)**

> **作者:** Abisha Thapa Magar; Anup Shakya
>
> **摘要:** Language Modeling is a prevalent task in Natural Language Processing. The currently existing most recent and most successful language models often tend to build a massive model with billions of parameters, feed in a tremendous amount of text data, and train with enormous computation resources which require millions of dollars. In this project, we aim to address an important sub-task in language modeling, i.e., context embedding. We propose an approach to exploit the Graph Convolution operation in GNNs to encode the context and use it in coalition with LSTMs to predict the next word given a local context of preceding words. We test this on the custom Wikipedia text corpus using a very limited amount of resources and show that this approach works fairly well to predict the next word.
>
---
#### [new 044] RouteNator: A Router-Based Multi-Modal Architecture for Generating Synthetic Training Data for Function Calling LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于LLM微调任务，解决功能调用场景中真实数据缺失导致的合成训练数据质量低问题。提出基于路由器的多模态架构，整合元数据、知识图谱和多模态模型生成高多样性合成数据，其路由机制匹配真实分布，实验证明模型在函数分类和API参数选择上优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.10495v1](http://arxiv.org/pdf/2505.10495v1)**

> **作者:** Vibha Belavadi; Tushar Vatsa; Dewang Sultania; Suhas Suresha; Ishita Verma; Cheng Chen; Tracy Holloway King; Michael Friedrich
>
> **备注:** Proceedings of the 4th International Workshop on Knowledge-Augmented Methods for Natural Language Processing
>
> **摘要:** This paper addresses fine-tuning Large Language Models (LLMs) for function calling tasks when real user interaction data is unavailable. In digital content creation tools, where users express their needs through natural language queries that must be mapped to API calls, the lack of real-world task-specific data and privacy constraints for training on it necessitate synthetic data generation. Existing approaches to synthetic data generation fall short in diversity and complexity, failing to replicate real-world data distributions and leading to suboptimal performance after LLM fine-tuning. We present a novel router-based architecture that leverages domain resources like content metadata and structured knowledge graphs, along with text-to-text and vision-to-text language models to generate high-quality synthetic training data. Our architecture's flexible routing mechanism enables synthetic data generation that matches observed real-world distributions, addressing a fundamental limitation of traditional approaches. Evaluation on a comprehensive set of real user queries demonstrates significant improvements in both function classification accuracy and API parameter selection. Models fine-tuned with our synthetic data consistently outperform traditional approaches, establishing new benchmarks for function calling tasks.
>
---
#### [new 045] A Survey on Large Language Models in Multimodal Recommender Systems
- **分类: cs.IR; cs.CL**

- **简介: 该论文为综述性研究，探讨大语言模型（LLMs）在多模态推荐系统（MRS）中的应用。针对LLMs整合中存在的可扩展性、模型访问性等问题，系统回顾了提示策略、微调方法及数据适应技术，提出新分类法归纳整合模式，总结评估指标与数据集，并展望未来方向，旨在推动LLMs在跨模态推荐领域的发展。**

- **链接: [http://arxiv.org/pdf/2505.09777v1](http://arxiv.org/pdf/2505.09777v1)**

> **作者:** Alejo Lopez-Avila; Jinhua Du
>
> **备注:** 30 pages, 6 figures
>
> **摘要:** Multimodal recommender systems (MRS) integrate heterogeneous user and item data, such as text, images, and structured information, to enhance recommendation performance. The emergence of large language models (LLMs) introduces new opportunities for MRS by enabling semantic reasoning, in-context learning, and dynamic input handling. Compared to earlier pre-trained language models (PLMs), LLMs offer greater flexibility and generalisation capabilities but also introduce challenges related to scalability and model accessibility. This survey presents a comprehensive review of recent work at the intersection of LLMs and MRS, focusing on prompting strategies, fine-tuning methods, and data adaptation techniques. We propose a novel taxonomy to characterise integration patterns, identify transferable techniques from related recommendation domains, provide an overview of evaluation metrics and datasets, and point to possible future directions. We aim to clarify the emerging role of LLMs in multimodal recommendation and support future research in this rapidly evolving field.
>
---
#### [new 046] Parallel Scaling Law for Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究语言模型高效扩展方法，属于模型优化任务。针对传统参数或推理扩展的高资源消耗问题，提出并行扩展范式ParScale：通过并行执行多输入变换并动态聚合结果，复用参数实现计算效率提升。理论推导新扩展定律并通过实验验证，证明并行扩展在同等性能下比参数扩展减少22倍内存和6倍延迟增长，支持小数据微调迁移预训练模型，为低资源部署提供新方案。**

- **链接: [http://arxiv.org/pdf/2505.10475v1](http://arxiv.org/pdf/2505.10475v1)**

> **作者:** Mouxiang Chen; Binyuan Hui; Zeyu Cui; Jiaxi Yang; Dayiheng Liu; Jianling Sun; Junyang Lin; Zhongxin Liu
>
> **摘要:** It is commonly believed that scaling language models should commit a significant space or time cost, by increasing the parameters (parameter scaling) or output tokens (inference-time scaling). We introduce the third and more inference-efficient scaling paradigm: increasing the model's parallel computation during both training and inference time. We apply $P$ diverse and learnable transformations to the input, execute forward passes of the model in parallel, and dynamically aggregate the $P$ outputs. This method, namely parallel scaling (ParScale), scales parallel computation by reusing existing parameters and can be applied to any model structure, optimization procedure, data, or task. We theoretically propose a new scaling law and validate it through large-scale pre-training, which shows that a model with $P$ parallel streams is similar to scaling the parameters by $O(\log P)$ while showing superior inference efficiency. For example, ParScale can use up to 22$\times$ less memory increase and 6$\times$ less latency increase compared to parameter scaling that achieves the same performance improvement. It can also recycle an off-the-shelf pre-trained model into a parallelly scaled one by post-training on a small amount of tokens, further reducing the training budget. The new scaling law we discovered potentially facilitates the deployment of more powerful models in low-resource scenarios, and provides an alternative perspective for the role of computation in machine learning.
>
---
#### [new 047] ComplexFormer: Disruptively Advancing Transformer Inference Ability via Head-Specific Complex Vector Attention
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ComplexFormer模型，改进Transformer的多头注意力机制。针对现有方法在位置编码与语义融合灵活性不足的问题，设计了复数向量注意力（CMHA），将语义和位置差异统一建模为复数平面的旋转缩放，通过每头独立的欧拉变换和自适应旋转机制增强表示能力。应用于语言建模、代码生成等任务，提升了推理性能和长上下文连贯性。**

- **链接: [http://arxiv.org/pdf/2505.10222v1](http://arxiv.org/pdf/2505.10222v1)**

> **作者:** Jintian Shao; Hongyi Huang; Jiayi Wu; Beiwen Zhang; ZhiYu Wu; You Shan; MingKai Zheng
>
> **摘要:** Transformer models rely on self-attention to capture token dependencies but face challenges in effectively integrating positional information while allowing multi-head attention (MHA) flexibility. Prior methods often model semantic and positional differences disparately or apply uniform positional adjustments across heads, potentially limiting representational capacity. This paper introduces ComplexFormer, featuring Complex Multi-Head Attention-CMHA. CMHA empowers each head to independently model semantic and positional differences unified within the complex plane, representing interactions as rotations and scaling. ComplexFormer incorporates two key improvements: (1) a per-head Euler transformation, converting real-valued query/key projections into polar-form complex vectors for head-specific complex subspace operation; and (2) a per-head adaptive differential rotation mechanism, exp[i(Adapt(ASmn,i) + Delta(Pmn),i)], allowing each head to learn distinct strategies for integrating semantic angle differences (ASmn,i) with relative positional encodings (Delta(Pmn),i). Extensive experiments on language modeling, text generation, code generation, and mathematical reasoning show ComplexFormer achieves superior performance, significantly lower generation perplexity , and improved long-context coherence compared to strong baselines like RoPE-Transformers. ComplexFormer demonstrates strong parameter efficiency, offering a more expressive, adaptable attention mechanism.
>
---
#### [new 048] Towards a Deeper Understanding of Reasoning Capabilities in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文评估大型语言模型在动态环境中的推理能力，属模型评估任务。通过实验不同提示策略，发现模型性能受规模与策略影响，但存在规划、推理等局限，表明需超越静态基准以揭示真实推理缺陷。**

- **链接: [http://arxiv.org/pdf/2505.10543v1](http://arxiv.org/pdf/2505.10543v1)**

> **作者:** Annie Wong; Thomas Bäck; Aske Plaat; Niki van Stein; Anna V. Kononova
>
> **摘要:** While large language models demonstrate impressive performance on static benchmarks, the true potential of large language models as self-learning and reasoning agents in dynamic environments remains unclear. This study systematically evaluates the efficacy of self-reflection, heuristic mutation, and planning as prompting techniques to test the adaptive capabilities of agents. We conduct experiments with various open-source language models in dynamic environments and find that larger models generally outperform smaller ones, but that strategic prompting can close this performance gap. Second, a too-long prompt can negatively impact smaller models on basic reactive tasks, while larger models show more robust behaviour. Third, advanced prompting techniques primarily benefit smaller models on complex games, but offer less improvement for already high-performing large language models. Yet, we find that advanced reasoning methods yield highly variable outcomes: while capable of significantly improving performance when reasoning and decision-making align, they also introduce instability and can lead to big performance drops. Compared to human performance, our findings reveal little evidence of true emergent reasoning. Instead, large language model performance exhibits persistent limitations in crucial areas such as planning, reasoning, and spatial coordination, suggesting that current-generation large language models still suffer fundamental shortcomings that may not be fully overcome through self-reflective prompting alone. Reasoning is a multi-faceted task, and while reasoning methods like Chain of thought improves multi-step reasoning on math word problems, our findings using dynamic benchmarks highlight important shortcomings in general reasoning capabilities, indicating a need to move beyond static benchmarks to capture the complexity of reasoning.
>
---
#### [new 049] StoryReasoning Dataset: Using Chain-of-Thought for Scene Understanding and Grounded Story Generation
- **分类: cs.CV; cs.CL; I.2.10; I.2.7**

- **简介: 该论文属于视觉叙事任务，旨在解决角色一致性差和指代幻觉问题。通过构建StoryReasoning数据集（含结构化场景分析和视觉关联故事），结合跨帧目标重识别、链式推理及视觉实体关联方法，提出Qwen Storyteller模型，将幻觉率降低12.3%。**

- **链接: [http://arxiv.org/pdf/2505.10292v1](http://arxiv.org/pdf/2505.10292v1)**

> **作者:** Daniel A. P. Oliveira; David Martins de Matos
>
> **备注:** 31 pages, 14 figures
>
> **摘要:** Visual storytelling systems struggle to maintain character identity across frames and link actions to appropriate subjects, frequently leading to referential hallucinations. These issues can be addressed through grounding of characters, objects, and other entities on the visual elements. We propose StoryReasoning, a dataset containing 4,178 stories derived from 52,016 movie images, with both structured scene analyses and grounded stories. Each story maintains character and object consistency across frames while explicitly modeling multi-frame relationships through structured tabular representations. Our approach features cross-frame object re-identification using visual similarity and face recognition, chain-of-thought reasoning for explicit narrative modeling, and a grounding scheme that links textual elements to visual entities across multiple frames. We establish baseline performance by fine-tuning Qwen2.5-VL 7B, creating Qwen Storyteller, which performs end-to-end object detection, re-identification, and landmark detection while maintaining consistent object references throughout the story. Evaluation demonstrates a reduction from 4.06 to 3.56 (-12.3%) hallucinations on average per story when compared to a non-fine-tuned model.
>
---
#### [new 050] Advanced Crash Causation Analysis for Freeway Safety: A Large Language Model Approach to Identifying Key Contributing Factors
- **分类: cs.LG; cs.CL; stat.AP**

- **简介: 该论文属于交通安全的文本分析任务，旨在解决传统方法难以捕捉事故复杂因素的问题。通过微调Llama3 8B模型分析226项研究数据，实现无标注条件下的高速事故成因识别（如酒驾、超速），结合事件数据提升解释性，模型结论获88.89%专家认可，为制定安全措施提供依据。**

- **链接: [http://arxiv.org/pdf/2505.09949v1](http://arxiv.org/pdf/2505.09949v1)**

> **作者:** Ahmed S. Abdelrahman; Mohamed Abdel-Aty; Samgyu Yang; Abdulrahman Faden
>
> **摘要:** Understanding the factors contributing to traffic crashes and developing strategies to mitigate their severity is essential. Traditional statistical methods and machine learning models often struggle to capture the complex interactions between various factors and the unique characteristics of each crash. This research leverages large language model (LLM) to analyze freeway crash data and provide crash causation analysis accordingly. By compiling 226 traffic safety studies related to freeway crashes, a training dataset encompassing environmental, driver, traffic, and geometric design factors was created. The Llama3 8B model was fine-tuned using QLoRA to enhance its understanding of freeway crashes and their contributing factors, as covered in these studies. The fine-tuned Llama3 8B model was then used to identify crash causation without pre-labeled data through zero-shot classification, providing comprehensive explanations to ensure that the identified causes were reasonable and aligned with existing research. Results demonstrate that LLMs effectively identify primary crash causes such as alcohol-impaired driving, speeding, aggressive driving, and driver inattention. Incorporating event data, such as road maintenance, offers more profound insights. The model's practical applicability and potential to improve traffic safety measures were validated by a high level of agreement among researchers in the field of traffic safety, as reflected in questionnaire results with 88.89%. This research highlights the complex nature of traffic crashes and how LLMs can be used for comprehensive analysis of crash causation and other contributing factors. Moreover, it provides valuable insights and potential countermeasures to aid planners and policymakers in developing more effective and efficient traffic safety practices.
>
---
#### [new 051] Comparing Exploration-Exploitation Strategies of LLMs and Humans: Insights from Standard Multi-armed Bandit Tasks
- **分类: cs.LG; cs.AI; cs.CL; cs.HC**

- **简介: 该论文研究LLMs与人类在动态决策中的探索-利用策略差异，属于行为对比任务。通过多臂老虎机实验，分析LLMs能否模拟人类决策及性能优劣。研究发现推理能力使LLMs更接近人类混合探索模式，但在复杂非稳态任务中适应性弱于人类，揭示了LLMs模拟人类行为的潜力与局限。**

- **链接: [http://arxiv.org/pdf/2505.09901v1](http://arxiv.org/pdf/2505.09901v1)**

> **作者:** Ziyuan Zhang; Darcy Wang; Ningyuan Chen; Rodrigo Mansur; Vahid Sarhangian
>
> **摘要:** Large language models (LLMs) are increasingly used to simulate or automate human behavior in complex sequential decision-making tasks. A natural question is then whether LLMs exhibit similar decision-making behavior to humans, and can achieve comparable (or superior) performance. In this work, we focus on the exploration-exploitation (E&E) tradeoff, a fundamental aspect of dynamic decision-making under uncertainty. We employ canonical multi-armed bandit (MAB) tasks introduced in the cognitive science and psychiatry literature to conduct a comparative study of the E&E strategies of LLMs, humans, and MAB algorithms. We use interpretable choice models to capture the E&E strategies of the agents and investigate how explicit reasoning, through both prompting strategies and reasoning-enhanced models, shapes LLM decision-making. We find that reasoning shifts LLMs toward more human-like behavior, characterized by a mix of random and directed exploration. In simple stationary tasks, reasoning-enabled LLMs exhibit similar levels of random and directed exploration compared to humans. However, in more complex, non-stationary environments, LLMs struggle to match human adaptability, particularly in effective directed exploration, despite achieving similar regret in certain scenarios. Our findings highlight both the promise and limits of LLMs as simulators of human behavior and tools for automated decision-making and point to potential areas of improvements.
>
---
#### [new 052] Superposition Yields Robust Neural Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究神经缩放定律（模型性能随尺寸增长的规律），属于机器学习理论分析。通过构建叠加表示和特征频率分布的玩具模型，揭示了模型损失与尺寸呈幂律关系的机制：弱叠加时损失依赖特征分布，强叠加时损失与维度成反比。实验验证主流大语言模型符合强叠加理论，为优化模型效率提供理论依据。**

- **链接: [http://arxiv.org/pdf/2505.10465v1](http://arxiv.org/pdf/2505.10465v1)**

> **作者:** Yizhou liu; Ziming Liu; Jeff Gore
>
> **备注:** 30 pages, 23 figures
>
> **摘要:** The success of today's large language models (LLMs) depends on the observation that larger models perform better. However, the origin of this neural scaling law -- the finding that loss decreases as a power law with model size -- remains unclear. Starting from two empirical principles -- that LLMs represent more things than the model dimensions (widths) they have (i.e., representations are superposed), and that words or concepts in language occur with varying frequencies -- we constructed a toy model to study the loss scaling with model size. We found that when superposition is weak, meaning only the most frequent features are represented without interference, the scaling of loss with model size depends on the underlying feature frequency; if feature frequencies follow a power law, so does the loss. In contrast, under strong superposition, where all features are represented but overlap with each other, the loss becomes inversely proportional to the model dimension across a wide range of feature frequency distributions. This robust scaling behavior is explained geometrically: when many more vectors are packed into a lower dimensional space, the interference (squared overlaps) between vectors scales inversely with that dimension. We then analyzed four families of open-sourced LLMs and found that they exhibit strong superposition and quantitatively match the predictions of our toy model. The Chinchilla scaling law turned out to also agree with our results. We conclude that representation superposition is an important mechanism underlying the observed neural scaling laws. We anticipate that these insights will inspire new training strategies and model architectures to achieve better performance with less computation and fewer parameters.
>
---
#### [new 053] MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态数学推理任务，旨在解决现有模型因缺乏数学图表细节而推理受限的问题。通过代码监督实现视觉与代码跨模态对齐，开发了FigCodifier模型和ImgCode-8.6M数据集，并合成MM-MathInstruct-3M指令数据。最终训练的MathCoder-VL模型在MathVista几何任务上超越GPT-4o和Claude 3.5，达到开源SOTA。**

- **链接: [http://arxiv.org/pdf/2505.10557v1](http://arxiv.org/pdf/2505.10557v1)**

> **作者:** Ke Wang; Junting Pan; Linda Wei; Aojun Zhou; Weikang Shi; Zimu Lu; Han Xiao; Yunqiao Yang; Houxing Ren; Mingjie Zhan; Hongsheng Li
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Natural language image-caption datasets, widely used for training Large Multimodal Models, mainly focus on natural scenarios and overlook the intricate details of mathematical figures that are critical for problem-solving, hindering the advancement of current LMMs in multimodal mathematical reasoning. To this end, we propose leveraging code as supervision for cross-modal alignment, since code inherently encodes all information needed to generate corresponding figures, establishing a precise connection between the two modalities. Specifically, we co-develop our image-to-code model and dataset with model-in-the-loop approach, resulting in an image-to-code model, FigCodifier and ImgCode-8.6M dataset, the largest image-code dataset to date. Furthermore, we utilize FigCodifier to synthesize novel mathematical figures and then construct MM-MathInstruct-3M, a high-quality multimodal math instruction fine-tuning dataset. Finally, we present MathCoder-VL, trained with ImgCode-8.6M for cross-modal alignment and subsequently fine-tuned on MM-MathInstruct-3M for multimodal math problem solving. Our model achieves a new open-source SOTA across all six metrics. Notably, it surpasses GPT-4o and Claude 3.5 Sonnet in the geometry problem-solving subset of MathVista, achieving improvements of 8.9% and 9.2%. The dataset and models will be released at https://github.com/mathllm/MathCoder.
>
---
#### [new 054] From Text to Network: Constructing a Knowledge Graph of Taiwan-Based China Studies Using Generative AI
- **分类: cs.AI; cs.CL; I.2.4; H.3.3; J.5**

- **简介: 该论文属于知识图谱构建任务，旨在解决台湾中国研究领域文献分散、缺乏系统整合的问题。通过生成式AI和大型语言模型从1367篇论文中提取结构化三元组，构建可视化知识图谱及向量数据库，实现非结构化文本到网络化知识导航的转换，助力学术发现与领域分析。**

- **链接: [http://arxiv.org/pdf/2505.10093v1](http://arxiv.org/pdf/2505.10093v1)**

> **作者:** Hsuan-Lei Shao
>
> **备注:** 4 pages, 4 figures
>
> **摘要:** Taiwanese China Studies (CS) has developed into a rich, interdisciplinary research field shaped by the unique geopolitical position and long standing academic engagement with Mainland China. This study responds to the growing need to systematically revisit and reorganize decades of Taiwan based CS scholarship by proposing an AI assisted approach that transforms unstructured academic texts into structured, interactive knowledge representations. We apply generative AI (GAI) techniques and large language models (LLMs) to extract and standardize entity relation triples from 1,367 peer reviewed CS articles published between 1996 and 2019. These triples are then visualized through a lightweight D3.js based system, forming the foundation of a domain specific knowledge graph and vector database for the field. This infrastructure allows users to explore conceptual nodes and semantic relationships across the corpus, revealing previously uncharted intellectual trajectories, thematic clusters, and research gaps. By decomposing textual content into graph structured knowledge units, our system enables a paradigm shift from linear text consumption to network based knowledge navigation. In doing so, it enhances scholarly access to CS literature while offering a scalable, data driven alternative to traditional ontology construction. This work not only demonstrates how generative AI can augment area studies and digital humanities but also highlights its potential to support a reimagined scholarly infrastructure for regional knowledge systems.
>
---
#### [new 055] Learning Virtual Machine Scheduling in Cloud Computing through Language Agents
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究云计算的虚拟机调度（在线动态多维装箱任务），解决传统方法适应性差、策略僵化及学习模型泛化不足的问题。提出分层语言代理框架MiCo，利用大语言模型生成启发式策略：Option Miner挖掘非上下文策略，Option Composer整合上下文策略。实验表明其在超万级规模下保持96.9%竞争比。**

- **链接: [http://arxiv.org/pdf/2505.10117v1](http://arxiv.org/pdf/2505.10117v1)**

> **作者:** JieHao Wu; Ziwei Wang; Junjie Sheng; Wenhao Li; Xiangfei Wang; Jun Luo
>
> **摘要:** In cloud services, virtual machine (VM) scheduling is a typical Online Dynamic Multidimensional Bin Packing (ODMBP) problem, characterized by large-scale complexity and fluctuating demands. Traditional optimization methods struggle to adapt to real-time changes, domain-expert-designed heuristic approaches suffer from rigid strategies, and existing learning-based methods often lack generalizability and interpretability. To address these limitations, this paper proposes a hierarchical language agent framework named MiCo, which provides a large language model (LLM)-driven heuristic design paradigm for solving ODMBP. Specifically, ODMBP is formulated as a Semi-Markov Decision Process with Options (SMDP-Option), enabling dynamic scheduling through a two-stage architecture, i.e., Option Miner and Option Composer. Option Miner utilizes LLMs to discover diverse and useful non-context-aware strategies by interacting with constructed environments. Option Composer employs LLMs to discover a composing strategy that integrates the non-context-aware strategies with the contextual ones. Extensive experiments on real-world enterprise datasets demonstrate that MiCo achieves a 96.9\% competitive ratio in large-scale scenarios involving more than 10,000 virtual machines. It maintains high performance even under nonstationary request flows and diverse configurations, thus validating its effectiveness in complex and large-scale cloud environments.
>
---
#### [new 056] On the Interplay of Human-AI Alignment,Fairness, and Performance Trade-offs in Medical Imaging
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究医学影像中AI的公平性、人机协同与性能平衡问题，属于医疗AI系统优化任务。针对模型偏见导致的跨群体公平差距，提出通过校准人机协作策略（融入专家知识）减少不公平性并提升泛化能力，同时避免过度对齐引发的性能下降，旨在构建兼顾公平、鲁棒与效率的医疗AI。**

- **链接: [http://arxiv.org/pdf/2505.10231v1](http://arxiv.org/pdf/2505.10231v1)**

> **作者:** Haozhe Luo; Ziyu Zhou; Zixin Shu; Aurélie Pahud de Mortanges; Robert Berke; Mauricio Reyes
>
> **摘要:** Deep neural networks excel in medical imaging but remain prone to biases, leading to fairness gaps across demographic groups. We provide the first systematic exploration of Human-AI alignment and fairness in this domain. Our results show that incorporating human insights consistently reduces fairness gaps and enhances out-of-domain generalization, though excessive alignment can introduce performance trade-offs, emphasizing the need for calibrated strategies. These findings highlight Human-AI alignment as a promising approach for developing fair, robust, and generalizable medical AI systems, striking a balance between expert guidance and automated efficiency. Our code is available at https://github.com/Roypic/Aligner.
>
---
#### [new 057] PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.09921v1](http://arxiv.org/pdf/2505.09921v1)**

> **作者:** Yidan Wang; Yanan Cao; Yubing Ren; Fang Fang; Zheng Lin; Binxing Fang
>
> **摘要:** Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at \href{https://github.com/redwyd/PrivacyJailbreak}{https://github.com/redwyd/PrivacyJailbreak}.
>
---
#### [new 058] Predictability Shapes Adaptation: An Evolutionary Perspective on Modes of Learning in Transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究Transformer模型中的权重学习（IWL）与上下文学习（ICL）的平衡机制，借鉴进化生物学的遗传编码与表型可塑性理论，探究环境可预测性（稳定性/线索可靠性）对两种模式的影响。通过实验验证：高稳定性强化IWL，高线索可靠性促进ICL，并揭示任务类型驱动学习动态（如IWL→ICL转变）。其成果为理解模型适应性策略提供了理论框架。**

- **链接: [http://arxiv.org/pdf/2505.09855v1](http://arxiv.org/pdf/2505.09855v1)**

> **作者:** Alexander Y. Ku; Thomas L. Griffiths; Stephanie C. Y. Chan
>
> **摘要:** Transformer models learn in two distinct modes: in-weights learning (IWL), encoding knowledge into model weights, and in-context learning (ICL), adapting flexibly to context without weight modification. To better understand the interplay between these learning modes, we draw inspiration from evolutionary biology's analogous adaptive strategies: genetic encoding (akin to IWL, adapting over generations and fixed within an individual's lifetime) and phenotypic plasticity (akin to ICL, enabling flexible behavioral responses to environmental cues). In evolutionary biology, environmental predictability dictates the balance between these strategies: stability favors genetic encoding, while reliable predictive cues promote phenotypic plasticity. We experimentally operationalize these dimensions of predictability and systematically investigate their influence on the ICL/IWL balance in Transformers. Using regression and classification tasks, we show that high environmental stability decisively favors IWL, as predicted, with a sharp transition at maximal stability. Conversely, high cue reliability enhances ICL efficacy, particularly when stability is low. Furthermore, learning dynamics reveal task-contingent temporal evolution: while a canonical ICL-to-IWL shift occurs in some settings (e.g., classification with many classes), we demonstrate that scenarios with easier IWL (e.g., fewer classes) or slower ICL acquisition (e.g., regression) can exhibit an initial IWL phase later yielding to ICL dominance. These findings support a relative-cost hypothesis for explaining these learning mode transitions, establishing predictability as a critical factor governing adaptive strategies in Transformers, and offering novel insights for understanding ICL and guiding training methodologies.
>
---
#### [new 059] Why 1 + 1 < 1 in Visual Token Pruning: Beyond Naive Integration via Multi-Objective Balanced Covering
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉令牌剪枝任务，解决现有方法因静态策略忽视多目标动态权衡导致的性能不稳定问题。通过理论分析提出误差界与最优目标平衡条件，设计多目标平衡覆盖方法（MoB），将剪枝转化为预算分配问题，实现高效自适应剪枝。实验验证其在多模态大模型中显著降低计算量且保持性能。**

- **链接: [http://arxiv.org/pdf/2505.10118v1](http://arxiv.org/pdf/2505.10118v1)**

> **作者:** Yangfu Li; Hongjian Zhan; Tianyi Chen; Qi Liu; Yue Lu
>
> **备注:** 31 pages,9 figures,conference
>
> **摘要:** Existing visual token pruning methods target prompt alignment and visual preservation with static strategies, overlooking the varying relative importance of these objectives across tasks, which leads to inconsistent performance. To address this, we derive the first closed-form error bound for visual token pruning based on the Hausdorff distance, uniformly characterizing the contributions of both objectives. Moreover, leveraging $\epsilon$-covering theory, we reveal an intrinsic trade-off between these objectives and quantify their optimal attainment levels under a fixed budget. To practically handle this trade-off, we propose Multi-Objective Balanced Covering (MoB), which reformulates visual token pruning as a bi-objective covering problem. In this framework, the attainment trade-off reduces to budget allocation via greedy radius trading. MoB offers a provable performance bound and linear scalability with respect to the number of input visual tokens, enabling adaptation to challenging pruning scenarios. Extensive experiments show that MoB preserves 96.4% of performance for LLaVA-1.5-7B using only 11.1% of the original visual tokens and accelerates LLaVA-Next-7B by 1.3-1.5$\times$ with negligible performance loss. Additionally, evaluations on Qwen2-VL and Video-LLaVA confirm that MoB integrates seamlessly into advanced MLLMs and diverse vision-language tasks.
>
---
#### [new 060] Tales of the 2025 Los Angeles Fire: Hotwash for Public Health Concerns in Reddit via LLM-Enhanced Topic Modeling
- **分类: cs.SI; cs.CL**

- **简介: 该论文属于社交媒体危机分析任务，旨在通过LLM增强的主题建模揭示野火灾难中的公众健康关切。研究分析2025年洛杉矶火灾期间Reddit的11万+评论，构建分层框架（态势感知与危机叙事），识别环境健康、心理风险等主题时序特征，为灾难响应提供数据驱动策略。**

- **链接: [http://arxiv.org/pdf/2505.09665v1](http://arxiv.org/pdf/2505.09665v1)**

> **作者:** Sulong Zhou; Qunying Huang; Shaoheng Zhou; Yun Hang; Xinyue Ye; Aodong Mei; Kathryn Phung; Yuning Ye; Uma Govindswamy; Zehan Li
>
> **摘要:** Wildfires have become increasingly frequent, irregular, and severe in recent years. Understanding how affected populations perceive and respond during wildfire crises is critical for timely and empathetic disaster response. Social media platforms offer a crowd-sourced channel to capture evolving public discourse, providing hyperlocal information and insight into public sentiment. This study analyzes Reddit discourse during the 2025 Los Angeles wildfires, spanning from the onset of the disaster to full containment. We collect 385 posts and 114,879 comments related to the Palisades and Eaton fires. We adopt topic modeling methods to identify the latent topics, enhanced by large language models (LLMs) and human-in-the-loop (HITL) refinement. Furthermore, we develop a hierarchical framework to categorize latent topics, consisting of two main categories, Situational Awareness (SA) and Crisis Narratives (CN). The volume of SA category closely aligns with real-world fire progressions, peaking within the first 2-5 days as the fires reach the maximum extent. The most frequent co-occurring category set of public health and safety, loss and damage, and emergency resources expands on a wide range of health-related latent topics, including environmental health, occupational health, and one health. Grief signals and mental health risks consistently accounted for 60 percentage and 40 percentage of CN instances, respectively, with the highest total volume occurring at night. This study contributes the first annotated social media dataset on the 2025 LA fires, and introduces a scalable multi-layer framework that leverages topic modeling for crisis discourse analysis. By identifying persistent public health concerns, our results can inform more empathetic and adaptive strategies for disaster response, public health communication, and future research in comparable climate-related disaster events.
>
---
#### [new 061] MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文针对视觉语言模型（VLM）推测解码加速任务，解决小语言模型无法处理视觉输入且预测不匹配的问题。提出MASSV方法，通过轻量级投影器连接视觉编码器，并利用目标模型自蒸馏对齐预测，提升推理速度1.46倍，兼容现有VLM架构。**

- **链接: [http://arxiv.org/pdf/2505.10526v1](http://arxiv.org/pdf/2505.10526v1)**

> **作者:** Mugilan Ganesan; Shane Segal; Ankur Aggarwal; Nish Sinnadurai; Sean Lie; Vithursan Thangarasa
>
> **备注:** Main paper: 11 pp., 4 figs., 3 tabs.; Supplementary: 2 pp
>
> **摘要:** Speculative decoding significantly accelerates language model inference by enabling a lightweight draft model to propose multiple tokens that a larger target model verifies simultaneously. However, applying this technique to vision-language models (VLMs) presents two fundamental challenges: small language models that could serve as efficient drafters lack the architectural components to process visual inputs, and their token predictions fail to match those of VLM target models that consider visual context. We introduce Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models (MASSV), which transforms existing small language models into effective multimodal drafters through a two-phase approach. MASSV first connects the target VLM's vision encoder to the draft model via a lightweight trainable projector, then applies self-distilled visual instruction tuning using responses generated by the target VLM to align token predictions. Comprehensive experiments across the Qwen2.5-VL and Gemma3 model families demonstrate that MASSV increases accepted length by up to 30% and delivers end-to-end inference speedups of up to 1.46x on visually-grounded tasks. MASSV provides a scalable, architecture-compatible method for accelerating both current and future VLMs.
>
---
## 更新

#### [replaced 001] Latent Action Pretraining from Videos
- **分类: cs.RO; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.11758v2](http://arxiv.org/pdf/2410.11758v2)**

> **作者:** Seonghyeon Ye; Joel Jang; Byeongguk Jeon; Sejune Joo; Jianwei Yang; Baolin Peng; Ajay Mandlekar; Reuben Tan; Yu-Wei Chao; Bill Yuchen Lin; Lars Liden; Kimin Lee; Jianfeng Gao; Luke Zettlemoyer; Dieter Fox; Minjoon Seo
>
> **备注:** ICLR 2025 Website: https://latentactionpretraining.github.io
>
> **摘要:** We introduce Latent Action Pretraining for general Action models (LAPA), an unsupervised method for pretraining Vision-Language-Action (VLA) models without ground-truth robot action labels. Existing Vision-Language-Action models require action labels typically collected by human teleoperators during pretraining, which significantly limits possible data sources and scale. In this work, we propose a method to learn from internet-scale videos that do not have robot action labels. We first train an action quantization model leveraging VQ-VAE-based objective to learn discrete latent actions between image frames, then pretrain a latent VLA model to predict these latent actions from observations and task descriptions, and finally finetune the VLA on small-scale robot manipulation data to map from latent to robot actions. Experimental results demonstrate that our method significantly outperforms existing techniques that train robot manipulation policies from large-scale videos. Furthermore, it outperforms the state-of-the-art VLA model trained with robotic action labels on real-world manipulation tasks that require language conditioning, generalization to unseen objects, and semantic generalization to unseen instructions. Training only on human manipulation videos also shows positive transfer, opening up the potential for leveraging web-scale data for robotics foundation model.
>
---
#### [replaced 002] PersLLM: A Personified Training Approach for Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2407.12393v5](http://arxiv.org/pdf/2407.12393v5)**

> **作者:** Zheni Zeng; Jiayi Chen; Huimin Chen; Yukun Yan; Yuxuan Chen; Zhenghao Liu; Zhiyuan Liu; Maosong Sun
>
> **备注:** 8 pages for main text, 5 figures
>
> **摘要:** Large language models (LLMs) exhibit human-like intelligence, enabling them to simulate human behavior and support various applications that require both humanized communication and extensive knowledge reserves. Efforts are made to personify LLMs with special training data or hand-crafted prompts, while correspondingly faced with challenges such as insufficient data usage or rigid behavior patterns. Consequently, personified LLMs fail to capture personified knowledge or express persistent opinion. To fully unlock the potential of LLM personification, we propose PersLLM, a framework for better data construction and model tuning. For insufficient data usage, we incorporate strategies such as Chain-of-Thought prompting and anti-induction, improving the quality of data construction and capturing the personality experiences, knowledge, and thoughts more comprehensively. For rigid behavior patterns, we design the tuning process and introduce automated DPO to enhance the specificity and dynamism of the models' personalities, which leads to a more natural opinion communication. Both automated metrics and expert human evaluations demonstrate the effectiveness of our approach. Case studies in human-machine interactions and multi-agent systems further suggest potential application scenarios and future directions for LLM personification.
>
---
#### [replaced 003] Not All Adapters Matter: Selective Adapter Freezing for Memory-Efficient Fine-Tuning of Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.03587v2](http://arxiv.org/pdf/2412.03587v2)**

> **作者:** Hyegang Son; Yonglak Son; Changhoon Kim; Young Geun Kim
>
> **备注:** URL: https://aclanthology.org/2025.naacl-long.480/ Volume: Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) Year: 2025 Address: Albuquerque, New Mexico
>
> **摘要:** Transformer-based large-scale pre-trained models achieve great success. Fine-tuning is the standard practice for leveraging these models in downstream tasks. Among the fine-tuning methods, adapter-tuning provides a parameter-efficient fine-tuning by introducing lightweight trainable modules while keeping most pre-trained parameters frozen. However, existing adapter-tuning methods still impose substantial resource usage. Through our investigation, we show that each adapter unequally contributes to both task performance and resource usage. Motivated by this insight, we propose Selective Adapter FrEezing (SAFE), which gradually freezes less important adapters early to reduce unnecessary resource usage while maintaining performance. In our experiments, SAFE reduces memory usage, computation amount, and training time by 42.85\%, 34.59\%, and 11.82\%, respectively, while achieving comparable or better task performance compared to the baseline. We also demonstrate that SAFE induces regularization effect, thereby smoothing the loss landscape, which enables the model to generalize better by avoiding sharp minima.
>
---
#### [replaced 004] Disentangling Memory and Reasoning Ability in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.13504v3](http://arxiv.org/pdf/2411.13504v3)**

> **作者:** Mingyu Jin; Weidi Luo; Sitao Cheng; Xinyi Wang; Wenyue Hua; Ruixiang Tang; William Yang Wang; Yongfeng Zhang
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong performance in handling complex tasks requiring both extensive knowledge and reasoning abilities. However, the existing LLM inference pipeline operates as an opaque process without explicit separation between knowledge retrieval and reasoning steps, making the model's decision-making process unclear and disorganized. This ambiguity can lead to issues such as hallucinations and knowledge forgetting, which significantly impact the reliability of LLMs in high-stakes domains. In this paper, we propose a new inference paradigm that decomposes the complex inference process into two distinct and clear actions: (1) memory recall: which retrieves relevant knowledge, and (2) reasoning: which performs logical steps based on the recalled knowledge. To facilitate this decomposition, we introduce two special tokens memory and reason, guiding the model to distinguish between steps that require knowledge retrieval and those that involve reasoning. Our experiment results show that this decomposition not only improves model performance but also enhances the interpretability of the inference process, enabling users to identify sources of error and refine model responses effectively. The code is available at https://github.com/MingyuJ666/Disentangling-Memory-and-Reasoning.
>
---
#### [replaced 005] Hypernym Mercury: Token Optimization Through Semantic Field Constriction And Reconstruction From Hypernyms. A New Text Compression Method
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08058v2](http://arxiv.org/pdf/2505.08058v2)**

> **作者:** Chris Forrester; Octavia Sulea
>
> **摘要:** Compute optimization using token reduction of LLM prompts is an emerging task in the fields of NLP and next generation, agentic AI. In this white paper, we introduce a novel (patent pending) text representation scheme and a first-of-its-kind word-level semantic compression of paragraphs that can lead to over 90% token reduction, while retaining high semantic similarity to the source text. We explain how this novel compression technique can be lossless and how the detail granularity is controllable. We discuss benchmark results over open source data (i.e. Bram Stoker's Dracula available through Project Gutenberg) and show how our results hold at the paragraph level, across multiple genres and models.
>
---
#### [replaced 006] Behind Maya: Building a Multilingual Vision Language Model
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08910v2](http://arxiv.org/pdf/2505.08910v2)**

> **作者:** Nahid Alam; Karthik Reddy Kanjula; Surya Guthikonda; Timothy Chung; Bala Krishna S Vegesna; Abhipsha Das; Anthony Susevski; Ryan Sze-Yin Chan; S M Iftekhar Uddin; Shayekh Bin Islam; Roshan Santhosh; Snegha A; Drishti Sharma; Chen Liu; Isha Chaturvedi; Genta Indra Winata; Ashvanth. S; Snehanshu Mukherjee; Alham Fikri Aji
>
> **备注:** Accepted at VLMs4ALL CVPR 2025 Workshop; corrected workshop name spelling
>
> **摘要:** In recent times, we have seen a rapid development of large Vision-Language Models (VLMs). They have shown impressive results on academic benchmarks, primarily in widely spoken languages but lack performance on low-resource languages and varied cultural contexts. To address these limitations, we introduce Maya, an open-source Multilingual VLM. Our contributions are: 1) a multilingual image-text pretraining dataset in eight languages, based on the LLaVA pretraining dataset; and 2) a multilingual image-text model supporting these languages, enhancing cultural and linguistic comprehension in vision-language tasks. Code available at https://github.com/nahidalam/maya.
>
---
#### [replaced 007] Mitigating Modality Bias in Multi-modal Entity Alignment from a Causal Perspective
- **分类: cs.MM; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2504.19458v3](http://arxiv.org/pdf/2504.19458v3)**

> **作者:** Taoyu Su; Jiawei Sheng; Duohe Ma; Xiaodong Li; Juwei Yue; Mengxiao Song; Yingkai Tang; Tingwen Liu
>
> **备注:** Accepted by SIGIR 2025, 11 pages, 10 figures, 4 tables,
>
> **摘要:** Multi-Modal Entity Alignment (MMEA) aims to retrieve equivalent entities from different Multi-Modal Knowledge Graphs (MMKGs), a critical information retrieval task. Existing studies have explored various fusion paradigms and consistency constraints to improve the alignment of equivalent entities, while overlooking that the visual modality may not always contribute positively. Empirically, entities with low-similarity images usually generate unsatisfactory performance, highlighting the limitation of overly relying on visual features. We believe the model can be biased toward the visual modality, leading to a shortcut image-matching task. To address this, we propose a counterfactual debiasing framework for MMEA, termed CDMEA, which investigates visual modality bias from a causal perspective. Our approach aims to leverage both visual and graph modalities to enhance MMEA while suppressing the direct causal effect of the visual modality on model predictions. By estimating the Total Effect (TE) of both modalities and excluding the Natural Direct Effect (NDE) of the visual modality, we ensure that the model predicts based on the Total Indirect Effect (TIE), effectively utilizing both modalities and reducing visual modality bias. Extensive experiments on 9 benchmark datasets show that CDMEA outperforms 14 state-of-the-art methods, especially in low-similarity, high-noise, and low-resource data scenarios.
>
---
#### [replaced 008] Time Awareness in Large Language Models: Benchmarking Fact Recall Across Time
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.13338v3](http://arxiv.org/pdf/2409.13338v3)**

> **作者:** David Herel; Vojtech Bartek; Jiri Jirak; Tomas Mikolov
>
> **摘要:** Who is the US President? The answer changes depending on when the question is asked. While large language models (LLMs) are evaluated on various reasoning tasks, they often miss a crucial dimension: time. In real-world scenarios, the correctness of answers is frequently tied to temporal context. To address this gap, we present a novel framework and dataset spanning over 8,000 events from 2018 to 2024, annotated with day-level granularity and sourced globally across domains such as politics, science, and business. Our TimeShift evaluation method systematically probes LLMs for temporal reasoning, revealing that base models often outperform instruction-tuned and synthetic-trained counterparts on time-sensitive recall. Additionally, we find that even large-scale models exhibit brittleness in handling paraphrased facts, highlighting unresolved challenges in temporal consistency. By identifying these limitations, our work provides a significant step toward advancing time-aware language models capable of adapting to the dynamic nature of real-world knowledge.
>
---
#### [replaced 009] Simple and Provable Scaling Laws for the Test-Time Compute of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19477v3](http://arxiv.org/pdf/2411.19477v3)**

> **作者:** Yanxi Chen; Xuchen Pan; Yaliang Li; Bolin Ding; Jingren Zhou
>
> **摘要:** We propose two simple, principled and practical algorithms that enjoy provable scaling laws for the test-time compute of large language models (LLMs). The first one is a two-stage knockout-style algorithm: given an input problem, it first generates multiple candidate solutions, and then aggregate them via a knockout tournament for the final output. Assuming that the LLM can generate a correct solution with non-zero probability and do better than a random guess in comparing a pair of correct and incorrect solutions, we prove theoretically that the failure probability of this algorithm decays to zero exponentially or by a power law (depending on the specific way of scaling) as its test-time compute grows. The second one is a two-stage league-style algorithm, where each candidate is evaluated by its average win rate against multiple opponents, rather than eliminated upon loss to a single opponent. Under analogous but more robust assumptions, we prove that its failure probability also decays to zero exponentially with more test-time compute. Both algorithms require a black-box LLM and nothing else (e.g., no verifier or reward model) for a minimalistic implementation, which makes them appealing for practical applications and easy to adapt for different tasks. Through extensive experiments with diverse models and datasets, we validate the proposed theories and demonstrate the outstanding scaling properties of both algorithms.
>
---
#### [replaced 010] PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.02069v4](http://arxiv.org/pdf/2406.02069v4)**

> **作者:** Zefan Cai; Yichi Zhang; Bofei Gao; Yuliang Liu; Yucheng Li; Tianyu Liu; Keming Lu; Wayne Xiong; Yue Dong; Junjie Hu; Wen Xiao
>
> **摘要:** In this study, we investigate whether attention-based information flow inside large language models (LLMs) is aggregated through noticeable patterns for long context processing. Our observations reveal that LLMs aggregate information through Pyramidal Information Funneling where attention is scattering widely in lower layers, progressively consolidating within specific contexts, and ultimately focusing on critical tokens (a.k.a massive activation or attention sink) in higher layers. Motivated by these insights, we developed PyramidKV, a novel and effective KV cache compression method. This approach dynamically adjusts the KV cache size across different layers, allocating more cache in lower layers and less in higher ones, diverging from traditional methods that maintain a uniform KV cache size. Our experimental evaluations, utilizing the LongBench benchmark, show that PyramidKV matches the performance of models with a full KV cache while retaining only 12% of the KV cache, thus significantly reducing memory usage. In scenarios emphasizing memory efficiency, where only 0.7% of the KV cache is maintained, PyramidKV surpasses other KV cache compression techniques, achieving up to a 20.5 absolute accuracy improvement on TREC dataset. In the Needle-in-a-Haystack experiment, PyramidKV outperforms competing methods in maintaining long-context comprehension in LLMs; notably, retaining just 128 KV cache entries enables the LLAMA-3-70B model to achieve 100.0 Acc. performance.
>
---
#### [replaced 011] How Does Knowledge Selection Help Retrieval Augmented Generation?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.13258v3](http://arxiv.org/pdf/2410.13258v3)**

> **作者:** Xiangci Li; Jessica Ouyang
>
> **摘要:** Retrieval-augmented generation (RAG) is a powerful method for enhancing natural language generation by integrating external knowledge into a model's output. While prior work has demonstrated the importance of improving knowledge retrieval for boosting generation quality, the role of knowledge selection remains less clear. This paper empirically analyzes how knowledge selection influences downstream generation performance in RAG systems. By simulating different retrieval and selection conditions through a controlled mixture of gold and distractor knowledge, we assess the impact of these factors on generation outcomes. Our findings indicate that the downstream generator model's capability, as well as the complexity of the task and dataset, significantly influence the impact of knowledge selection on the overall RAG system performance. In typical scenarios, improving the knowledge recall score is key to enhancing generation outcomes, with the knowledge selector providing limited benefit when a strong generator model is used on clear, well-defined tasks. For weaker generator models or more ambiguous tasks and datasets, the knowledge F1 score becomes a critical factor, and the knowledge selector plays a more prominent role in improving overall performance.
>
---
#### [replaced 012] Data-Driven Calibration of Prediction Sets in Large Vision-Language Models Based on Inductive Conformal Prediction
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.17671v3](http://arxiv.org/pdf/2504.17671v3)**

> **作者:** Yuanchang Ye; Weiyan Wen
>
> **备注:** Accepted by ICIPCA 2025
>
> **摘要:** This study addresses the critical challenge of hallucination mitigation in Large Vision-Language Models (LVLMs) for Visual Question Answering (VQA) tasks through a Split Conformal Prediction (SCP) framework. While LVLMs excel in multi-modal reasoning, their outputs often exhibit hallucinated content with high confidence, posing risks in safety-critical applications. We propose a model-agnostic uncertainty quantification method that integrates dynamic threshold calibration and cross-modal consistency verification. By partitioning data into calibration and test sets, the framework computes nonconformity scores to construct prediction sets with statistical guarantees under user-defined risk levels ($\alpha$). Key innovations include: (1) rigorous control of \textbf{marginal coverage} to ensure empirical error rates remain strictly below $\alpha$; (2) dynamic adjustment of prediction set sizes inversely with $\alpha$, filtering low-confidence outputs; (3) elimination of prior distribution assumptions and retraining requirements. Evaluations on benchmarks (ScienceQA, MMMU) with eight LVLMs demonstrate that SCP enforces theoretical guarantees across all $\alpha$ values. The framework achieves stable performance across varying calibration-to-test split ratios, underscoring its robustness for real-world deployment in healthcare, autonomous systems, and other safety-sensitive domains. This work bridges the gap between theoretical reliability and practical applicability in multi-modal AI systems, offering a scalable solution for hallucination detection and uncertainty-aware decision-making.
>
---
#### [replaced 013] KwaiChat: A Large-Scale Video-Driven Multilingual Mixed-Type Dialogue Corpus
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.06899v2](http://arxiv.org/pdf/2503.06899v2)**

> **作者:** Xiaoming Shi; Zeming Liu; Yiming Lei; Chenkai Zhang; Haitao Leng; Chuan Wang; Qingjie Liu; Wanxiang Che; Shaoguo Liu; Size Li; Yunhong Wang
>
> **摘要:** Video-based dialogue systems, such as education assistants, have compelling application value, thereby garnering growing interest. However, the current video-based dialogue systems are limited by their reliance on a single dialogue type, which hinders their versatility in practical applications across a range of scenarios, including question-answering, emotional dialog, etc. In this paper, we identify this challenge as how to generate video-driven multilingual mixed-type dialogues. To mitigate this challenge, we propose a novel task and create a human-to-human video-driven multilingual mixed-type dialogue corpus, termed KwaiChat, containing a total of 93,209 videos and 246,080 dialogues, across 4 dialogue types, 30 domains, 4 languages, and 13 topics. Additionally, we establish baseline models on KwaiChat. An extensive analysis of 7 distinct LLMs on KwaiChat reveals that GPT-4o achieves the best performance but still cannot perform well in this situation even with the help of in-context learning and fine-tuning, which indicates that the task is not trivial and needs further research.
>
---
#### [replaced 014] Phase Diagram of Vision Large Language Models Inference: A Perspective from Interaction across Image and Instruction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.00646v2](http://arxiv.org/pdf/2411.00646v2)**

> **作者:** Houjing Wei; Yuting Shi; Naoya Inoue
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** Vision Large Language Models (VLLMs) usually take input as a concatenation of image token embeddings and text token embeddings and conduct causal modeling. However, their internal behaviors remain underexplored, raising the question of interaction among two types of tokens. To investigate such multimodal interaction during model inference, in this paper, we measure the contextualization among the hidden state vectors of tokens from different modalities. Our experiments uncover a four-phase inference dynamics of VLLMs against the depth of Transformer-based LMs, including (I) Alignment: In very early layers, contextualization emerges between modalities, suggesting a feature space alignment. (II) Intra-modal Encoding: In early layers, intra-modal contextualization is enhanced while inter-modal interaction is suppressed, suggesting a local encoding within modalities. (III) Inter-modal Encoding: In later layers, contextualization across modalities is enhanced, suggesting a deeper fusion across modalities. (IV) Output Preparation: In very late layers, contextualization is reduced globally, and hidden states are aligned towards the unembedding space.
>
---
#### [replaced 015] RM-R1: Reward Modeling as Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02387v2](http://arxiv.org/pdf/2505.02387v2)**

> **作者:** Xiusi Chen; Gaotang Li; Ziqi Wang; Bowen Jin; Cheng Qian; Yu Wang; Hongru Wang; Yu Zhang; Denghui Zhang; Tong Zhang; Hanghang Tong; Heng Ji
>
> **备注:** 24 pages, 8 figures
>
> **摘要:** Reward modeling is essential for aligning large language models (LLMs) with human preferences through reinforcement learning (RL). To provide accurate reward signals, a reward model (RM) should stimulate deep thinking and conduct interpretable reasoning before assigning a score or a judgment. Inspired by recent advances of long chain-of-thought (CoT) on reasoning-intensive tasks, we hypothesize and validate that integrating reasoning capabilities into reward modeling significantly enhances RM's interpretability and performance. To this end, we introduce a new class of generative reward models -- Reasoning Reward Models (ReasRMs) -- which formulate reward modeling as a reasoning task. We propose a reasoning-oriented training pipeline and train a family of ReasRMs, RM-R1. RM-R1 features a chain-of-rubrics (CoR) mechanism -- self-generating sample-level chat rubrics or math/code solutions, and evaluating candidate responses against them. The training of M-R1 consists of two key stages: (1) distillation of high-quality reasoning chains and (2) reinforcement learning with verifiable rewards. Empirically, our models achieve state-of-the-art performance across three reward model benchmarks on average, outperforming much larger open-weight models (e.g., INF-ORM-Llama3.1-70B) and proprietary ones (e.g., GPT-4o) by up to 4.9%. Beyond final performance, we perform thorough empirical analysis to understand the key ingredients of successful ReasRM training. To facilitate future research, we release six ReasRM models along with code and data at https://github.com/RM-R1-UIUC/RM-R1.
>
---
#### [replaced 016] Beyond Next Token Prediction: Patch-Level Training for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.12665v3](http://arxiv.org/pdf/2407.12665v3)**

> **作者:** Chenze Shao; Fandong Meng; Jie Zhou
>
> **备注:** ICLR 2025 Spotlight
>
> **摘要:** The prohibitive training costs of Large Language Models (LLMs) have emerged as a significant bottleneck in the development of next-generation LLMs. In this paper, we show that it is possible to significantly reduce the training costs of LLMs without sacrificing their performance. Specifically, we introduce patch-level training for LLMs, in which multiple tokens are aggregated into a unit of higher information density, referred to as a `patch', to serve as the fundamental text unit for training LLMs. During patch-level training, we feed the language model shorter sequences of patches and train it to predict the next patch, thereby processing the majority of the training data at a significantly reduced cost. Following this, the model continues token-level training on the remaining training data to align with the inference mode. Experiments on a diverse range of models (370M-2.7B parameters) demonstrate that patch-level training can reduce the overall training costs to 0.5$\times$, without compromising the model performance compared to token-level training. Source code: https://github.com/shaochenze/PatchTrain.
>
---
#### [replaced 017] Model Utility Law: Evaluating LLMs beyond Performance through Mechanism Interpretable Metric
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07440v2](http://arxiv.org/pdf/2504.07440v2)**

> **作者:** Yixin Cao; Jiahao Ying; Yaoning Wang; Xipeng Qiu; Xuanjing Huang; Yugang Jiang
>
> **摘要:** Large Language Models (LLMs) have become indispensable across academia, industry, and daily applications, yet current evaluation methods struggle to keep pace with their rapid development. One core challenge of evaluation in the large language model (LLM) era is the generalization issue: how to infer a model's near-unbounded abilities from inevitably bounded benchmarks. We address this challenge by proposing Model Utilization Index (MUI), a mechanism interpretability enhanced metric that complements traditional performance scores. MUI quantifies the effort a model expends on a task, defined as the proportion of activated neurons or features during inference. Intuitively, a truly capable model should achieve higher performance with lower effort. Extensive experiments across popular LLMs reveal a consistent inverse logarithmic relationship between MUI and performance, which we formulate as the Utility Law. From this law we derive four practical corollaries that (i) guide training diagnostics, (ii) expose data contamination issue, (iii) enable fairer model comparisons, and (iv) design model-specific dataset diversity. Our code can be found at https://github.com/ALEX-nlp/MUI-Eva.
>
---
#### [replaced 018] ARR: Question Answering with Large Language Models via Analyzing, Retrieving, and Reasoning
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.04689v3](http://arxiv.org/pdf/2502.04689v3)**

> **作者:** Yuwei Yin; Giuseppe Carenini
>
> **备注:** 21 pages. Code: https://github.com/YuweiYin/ARR
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities on complex evaluation benchmarks, many of which are formulated as question-answering (QA) tasks. Enhancing the performance of LLMs in QA contexts is becoming increasingly vital for advancing their development and applicability. This paper introduces ARR, an intuitive, effective, and general QA solving method that explicitly incorporates three key steps: analyzing the intent of the question, retrieving relevant information, and reasoning step by step. Notably, this paper is the first to introduce intent analysis in QA, which plays a vital role in ARR. Comprehensive evaluations across 10 diverse QA tasks demonstrate that ARR consistently outperforms the baseline methods. Ablation and case studies further validate the positive contributions of each ARR component. Furthermore, experiments involving variations in prompt design indicate that ARR maintains its effectiveness regardless of the specific prompt formulation. Additionally, extensive evaluations across various model sizes, LLM series, and generation settings solidify the effectiveness, robustness, and generalizability of ARR.
>
---
#### [replaced 019] FitCF: A Framework for Automatic Feature Importance-guided Counterfactual Example Generation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00777v2](http://arxiv.org/pdf/2501.00777v2)**

> **作者:** Qianli Wang; Nils Feldhus; Simon Ostermann; Luis Felipe Villa-Arenas; Sebastian Möller; Vera Schmitt
>
> **备注:** ACL 2025 Findings; camera-ready version
>
> **摘要:** Counterfactual examples are widely used in natural language processing (NLP) as valuable data to improve models, and in explainable artificial intelligence (XAI) to understand model behavior. The automated generation of counterfactual examples remains a challenging task even for large language models (LLMs), despite their impressive performance on many tasks. In this paper, we first introduce ZeroCF, a faithful approach for leveraging important words derived from feature attribution methods to generate counterfactual examples in a zero-shot setting. Second, we present a new framework, FitCF, which further verifies aforementioned counterfactuals by label flip verification and then inserts them as demonstrations for few-shot prompting, outperforming two state-of-the-art baselines. Through ablation studies, we identify the importance of each of FitCF's core components in improving the quality of counterfactuals, as assessed through flip rate, perplexity, and similarity measures. Furthermore, we show the effectiveness of LIME and Integrated Gradients as backbone attribution methods for FitCF and find that the number of demonstrations has the largest effect on performance. Finally, we reveal a strong correlation between the faithfulness of feature attribution scores and the quality of generated counterfactuals.
>
---
#### [replaced 020] RoBERTa-BiLSTM: A Context-Aware Hybrid Model for Sentiment Analysis
- **分类: cs.CL; cs.AI; cs.CE**

- **链接: [http://arxiv.org/pdf/2406.00367v2](http://arxiv.org/pdf/2406.00367v2)**

> **作者:** Md. Mostafizer Rahman; Ariful Islam Shiplu; Yutaka Watanobe; Md. Ashad Alam
>
> **摘要:** Effectively analyzing the comments to uncover latent intentions holds immense value in making strategic decisions across various domains. However, several challenges hinder the process of sentiment analysis including the lexical diversity exhibited in comments, the presence of long dependencies within the text, encountering unknown symbols and words, and dealing with imbalanced datasets. Moreover, existing sentiment analysis tasks mostly leveraged sequential models to encode the long dependent texts and it requires longer execution time as it processes the text sequentially. In contrast, the Transformer requires less execution time due to its parallel processing nature. In this work, we introduce a novel hybrid deep learning model, RoBERTa-BiLSTM, which combines the Robustly Optimized BERT Pretraining Approach (RoBERTa) with Bidirectional Long Short-Term Memory (BiLSTM) networks. RoBERTa is utilized to generate meaningful word embedding vectors, while BiLSTM effectively captures the contextual semantics of long-dependent texts. The RoBERTa-BiLSTM hybrid model leverages the strengths of both sequential and Transformer models to enhance performance in sentiment analysis. We conducted experiments using datasets from IMDb, Twitter US Airline, and Sentiment140 to evaluate the proposed model against existing state-of-the-art methods. Our experimental findings demonstrate that the RoBERTa-BiLSTM model surpasses baseline models (e.g., BERT, RoBERTa-base, RoBERTa-GRU, and RoBERTa-LSTM), achieving accuracies of 80.74%, 92.36%, and 82.25% on the Twitter US Airline, IMDb, and Sentiment140 datasets, respectively. Additionally, the model achieves F1-scores of 80.73%, 92.35%, and 82.25% on the same datasets, respectively.
>
---
#### [replaced 021] Concise Reasoning via Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05185v2](http://arxiv.org/pdf/2504.05185v2)**

> **作者:** Mehdi Fatemi; Banafsheh Rafiee; Mingjie Tang; Kartik Talamadupula
>
> **摘要:** Despite significant advancements in large language models (LLMs), a major drawback of reasoning models is their enormous token usage, which increases computational cost, resource requirements, and response time. In this work, we revisit the core principles of reinforcement learning (RL) and, through mathematical analysis, demonstrate that the tendency to generate lengthy responses arises inherently from RL-based optimization during training. This finding questions the prevailing assumption that longer responses inherently improve reasoning accuracy. Instead, we uncover a natural correlation between conciseness and accuracy that has been largely overlooked. We show that introducing a secondary phase of RL training, using a very small set of problems, can significantly reduce chains of thought while maintaining or even enhancing accuracy. Additionally, we demonstrate that, while GRPO shares some interesting properties of PPO, it suffers from collapse modes, which limit its reliability for concise reasoning. Finally, we validate our conclusions through extensive experimental results.
>
---
#### [replaced 022] Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information
- **分类: cs.CL; cs.LG; 68T50**

- **链接: [http://arxiv.org/pdf/2505.06046v2](http://arxiv.org/pdf/2505.06046v2)**

> **作者:** Joshua Harris; Fan Grayson; Felix Feldman; Timothy Laurence; Toby Nonnenmacher; Oliver Higgins; Leo Loman; Selina Patel; Thomas Finnie; Samuel Collins; Michael Borowitz
>
> **备注:** 24 pages, 10 pages main text
>
> **摘要:** As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, currently little is known about LLM knowledge of UK Government public health information. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries. To create PubHealthBench we extract free text from 687 current UK government guidance documents and implement an automated pipeline for generating MCQA samples. Assessing 24 LLMs on PubHealthBench we find the latest private LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% accuracy in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Importantly we find in both setups LLMs have higher accuracy on guidance intended for the general public. Therefore, there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, but additional safeguards or tools may still be needed when providing free form responses on public health topics.
>
---
#### [replaced 023] MultiMed: Multilingual Medical Speech Recognition via Attention Encoder Decoder
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.14074v3](http://arxiv.org/pdf/2409.14074v3)**

> **作者:** Khai Le-Duc; Phuc Phan; Tan-Hanh Pham; Bach Phan Tat; Minh-Huong Ngo; Chris Ngo; Thanh Nguyen-Tang; Truong-Son Hy
>
> **备注:** ACL 2025, 38 pages
>
> **摘要:** Multilingual automatic speech recognition (ASR) in the medical domain serves as a foundational task for various downstream applications such as speech translation, spoken language understanding, and voice-activated assistants. This technology improves patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we introduce MultiMed, the first multilingual medical ASR dataset, along with the first collection of small-to-large end-to-end medical ASR models, spanning five languages: Vietnamese, English, German, French, and Mandarin Chinese. To our best knowledge, MultiMed stands as the world's largest medical ASR dataset across all major benchmarks: total duration, number of recording conditions, number of accents, and number of speaking roles. Furthermore, we present the first multilinguality study for medical ASR, which includes reproducible empirical baselines, a monolinguality-multilinguality analysis, Attention Encoder Decoder (AED) vs Hybrid comparative study and a linguistic analysis. We present practical ASR end-to-end training schemes optimized for a fixed number of trainable parameters that are common in industry settings. All code, data, and models are available online: https://github.com/leduckhai/MultiMed/tree/master/MultiMed.
>
---
#### [replaced 024] TopoLM: brain-like spatio-functional organization in a topographic language model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.11516v3](http://arxiv.org/pdf/2410.11516v3)**

> **作者:** Neil Rathi; Johannes Mehrer; Badr AlKhamissi; Taha Binhuraib; Nicholas M. Blauch; Martin Schrimpf
>
> **摘要:** Neurons in the brain are spatially organized such that neighbors on tissue often exhibit similar response profiles. In the human language system, experimental studies have observed clusters for syntactic and semantic categories, but the mechanisms underlying this functional organization remain unclear. Here, building on work from the vision literature, we develop TopoLM, a transformer language model with an explicit two-dimensional spatial representation of model units. By combining a next-token prediction objective with a spatial smoothness loss, representations in this model assemble into clusters that correspond to semantically interpretable groupings of text and closely match the functional organization in the brain's language system. TopoLM successfully predicts the emergence of the spatio-functional organization of a cortical language system as well as the organization of functional clusters selective for fine-grained linguistic features empirically observed in human cortex. Our results suggest that the functional organization of the human language system is driven by a unified spatial objective, and provide a functionally and spatially aligned model of language processing in the brain.
>
---
#### [replaced 025] Harnessing Multiple Large Language Models: A Survey on LLM Ensemble
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18036v4](http://arxiv.org/pdf/2502.18036v4)**

> **作者:** Zhijun Chen; Jingzheng Li; Pengpeng Chen; Zhuoran Li; Kai Sun; Yuankai Luo; Qianren Mao; Dingqi Yang; Hailong Sun; Philip S. Yu
>
> **备注:** 9 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
>
> **摘要:** LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference'', and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
>
---
#### [replaced 026] SceneGenAgent: Precise Industrial Scene Generation with Coding Agent
- **分类: cs.CL; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2410.21909v2](http://arxiv.org/pdf/2410.21909v2)**

> **作者:** Xiao Xia; Dan Zhang; Zibo Liao; Zhenyu Hou; Tianrui Sun; Jing Li; Ling Fu; Yuxiao Dong
>
> **摘要:** The modeling of industrial scenes is essential for simulations in industrial manufacturing. While large language models (LLMs) have shown significant progress in generating general 3D scenes from textual descriptions, generating industrial scenes with LLMs poses a unique challenge due to their demand for precise measurements and positioning, requiring complex planning over spatial arrangement. To address this challenge, we introduce SceneGenAgent, an LLM-based agent for generating industrial scenes through C# code. SceneGenAgent ensures precise layout planning through a structured and calculable format, layout verification, and iterative refinement to meet the quantitative requirements of industrial scenarios. Experiment results demonstrate that LLMs powered by SceneGenAgent exceed their original performance, reaching up to 81.0% success rate in real-world industrial scene generation tasks and effectively meeting most scene generation requirements. To further enhance accessibility, we construct SceneInstruct, a dataset designed for fine-tuning open-source LLMs to integrate into SceneGenAgent. Experiments show that fine-tuning open-source LLMs on SceneInstruct yields significant performance improvements, with Llama3.1-70B approaching the capabilities of GPT-4o. Our code and data are available at https://github.com/THUDM/SceneGenAgent .
>
---
#### [replaced 027] Temporal Scaling Law for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2404.17785v3](http://arxiv.org/pdf/2404.17785v3)**

> **作者:** Yizhe Xiong; Xiansheng Chen; Xin Ye; Hui Chen; Zijia Lin; Haoran Lian; Zhenpeng Su; Wei Huang; Jianwei Niu; Jungong Han; Guiguang Ding
>
> **备注:** Preprint, Currently under review
>
> **摘要:** Recently, Large Language Models (LLMs) have been widely adopted in a wide range of tasks, leading to increasing attention towards the research on how scaling LLMs affects their performance. Existing works, termed Scaling Laws, have discovered that the final test loss of LLMs scales as power-laws with model size, computational budget, and dataset size. However, the temporal change of the test loss of an LLM throughout its pre-training process remains unexplored, though it is valuable in many aspects, such as selecting better hyperparameters \textit{directly} on the target LLM. In this paper, we propose the novel concept of Temporal Scaling Law, studying how the test loss of an LLM evolves as the training steps scale up. In contrast to modeling the test loss as a whole in a coarse-grained manner, we break it down and dive into the fine-grained test loss of each token position, and further develop a dynamic hyperbolic-law. Afterwards, we derive the much more precise temporal scaling law by studying the temporal patterns of the parameters in the dynamic hyperbolic-law. Results on both in-distribution (ID) and out-of-distribution (OOD) validation datasets demonstrate that our temporal scaling law accurately predicts the test loss of LLMs across training steps. Our temporal scaling law has broad practical applications. First, it enables direct and efficient hyperparameter selection on the target LLM, such as data mixture proportions. Secondly, viewing the LLM pre-training dynamics from the token position granularity provides some insights to enhance the understanding of LLM pre-training.
>
---
#### [replaced 028] Benchmarking Generative AI for Scoring Medical Student Interviews in Objective Structured Clinical Examinations (OSCEs)
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2501.13957v2](http://arxiv.org/pdf/2501.13957v2)**

> **作者:** Jadon Geathers; Yann Hicke; Colleen Chan; Niroop Rajashekar; Justin Sewell; Susannah Cornes; Rene F. Kizilcec; Dennis Shung
>
> **备注:** 12 pages + 3 pages of references, 4 figures
>
> **摘要:** Objective Structured Clinical Examinations (OSCEs) are widely used to assess medical students' communication skills, but scoring interview-based assessments is time-consuming and potentially subject to human bias. This study explored the potential of large language models (LLMs) to automate OSCE evaluations using the Master Interview Rating Scale (MIRS). We compared the performance of four state-of-the-art LLMs (GPT-4o, Claude 3.5, Llama 3.1, and Gemini 1.5 Pro) in evaluating OSCE transcripts across all 28 items of the MIRS under the conditions of zero-shot, chain-of-thought (CoT), few-shot, and multi-step prompting. The models were benchmarked against a dataset of 10 OSCE cases with 174 expert consensus scores available. Model performance was measured using three accuracy metrics (exact, off-by-one, thresholded). Averaging across all MIRS items and OSCE cases, LLMs performed with low exact accuracy (0.27 to 0.44), and moderate to high off-by-one accuracy (0.67 to 0.87) and thresholded accuracy (0.75 to 0.88). A zero temperature parameter ensured high intra-rater reliability ({\alpha} = 0.98 for GPT-4o). CoT, few-shot, and multi-step techniques proved valuable when tailored to specific assessment items. The performance was consistent across MIRS items, independent of encounter phases and communication domains. We demonstrated the feasibility of AI-assisted OSCE evaluation and provided benchmarking of multiple LLMs across multiple prompt techniques. Our work provides a baseline performance assessment for LLMs that lays a foundation for future research into automated assessment of clinical communication skills.
>
---
#### [replaced 029] Understanding In-context Learning of Addition via Activation Subspaces
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05145v2](http://arxiv.org/pdf/2505.05145v2)**

> **作者:** Xinyan Hu; Kayo Yin; Michael I. Jordan; Jacob Steinhardt; Lijie Chen
>
> **备注:** 20 pages
>
> **摘要:** To perform in-context learning, language models must extract signals from individual few-shot examples, aggregate these into a learned prediction rule, and then apply this rule to new examples. How is this implemented in the forward pass of modern transformer models? To study this, we consider a structured family of few-shot learning tasks for which the true prediction rule is to add an integer $k$ to the input. We find that Llama-3-8B attains high accuracy on this task for a range of $k$, and localize its few-shot ability to just three attention heads via a novel optimization approach. We further show the extracted signals lie in a six-dimensional subspace, where four of the dimensions track the unit digit and the other two dimensions track overall magnitude. We finally examine how these heads extract information from individual few-shot examples, identifying a self-correction mechanism in which mistakes from earlier examples are suppressed by later examples. Our results demonstrate how tracking low-dimensional subspaces across a forward pass can provide insight into fine-grained computational structures.
>
---
#### [replaced 030] TensorLLM: Tensorising Multi-Head Attention for Enhanced Reasoning and Compression in LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.15674v2](http://arxiv.org/pdf/2501.15674v2)**

> **作者:** Yuxuan Gu; Wuyang Zhou; Giorgos Iacovides; Danilo Mandic
>
> **备注:** Accpeted for IEEE International Joint Conference on Neural Networks (IJCNN 2025). The code is available at https://github.com/guyuxuan9/TensorLLM
>
> **摘要:** The reasoning abilities of Large Language Models (LLMs) can be improved by structurally denoising their weights, yet existing techniques primarily focus on denoising the feed-forward network (FFN) of the transformer block, and can not efficiently utilise the Multi-head Attention (MHA) block, which is the core of transformer architectures. To address this issue, we propose a novel intuitive framework that, at its very core, performs MHA compression through a multi-head tensorisation process and the Tucker decomposition. This enables both higher-dimensional structured denoising and compression of the MHA weights, by enforcing a shared higher-dimensional subspace across the weights of the multiple attention heads. We demonstrate that this approach consistently enhances the reasoning capabilities of LLMs across multiple benchmark datasets, and for both encoder-only and decoder-only architectures, while achieving compression rates of up to $\sim 250$ times in the MHA weights, all without requiring any additional data, training, or fine-tuning. Furthermore, we show that the proposed method can be seamlessly combined with existing FFN-only-based denoising techniques to achieve further improvements in LLM reasoning performance.
>
---
#### [replaced 031] SAS-Bench: A Fine-Grained Benchmark for Evaluating Short Answer Scoring with Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.07247v2](http://arxiv.org/pdf/2505.07247v2)**

> **作者:** Peichao Lai; Kexuan Zhang; Yi Lin; Linyihan Zhang; Feiyang Ye; Jinhao Yan; Yanwei Xu; Conghui He; Yilei Wang; Wentao Zhang; Bin Cui
>
> **摘要:** Subjective Answer Grading (SAG) plays a crucial role in education, standardized testing, and automated assessment systems, particularly for evaluating short-form responses in Short Answer Scoring (SAS). However, existing approaches often produce coarse-grained scores and lack detailed reasoning. Although large language models (LLMs) have demonstrated potential as zero-shot evaluators, they remain susceptible to bias, inconsistencies with human judgment, and limited transparency in scoring decisions. To overcome these limitations, we introduce SAS-Bench, a benchmark specifically designed for LLM-based SAS tasks. SAS-Bench provides fine-grained, step-wise scoring, expert-annotated error categories, and a diverse range of question types derived from real-world subject-specific exams. This benchmark facilitates detailed evaluation of model reasoning processes and explainability. We also release an open-source dataset containing 1,030 questions and 4,109 student responses, each annotated by domain experts. Furthermore, we conduct comprehensive experiments with various LLMs, identifying major challenges in scoring science-related questions and highlighting the effectiveness of few-shot prompting in improving scoring accuracy. Our work offers valuable insights into the development of more robust, fair, and educationally meaningful LLM-based evaluation systems.
>
---
#### [replaced 032] Construction and Application of Materials Knowledge Graph in Multidisciplinary Materials Science via Large Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.03080v5](http://arxiv.org/pdf/2404.03080v5)**

> **作者:** Yanpeng Ye; Jie Ren; Shaozhou Wang; Yuwei Wan; Imran Razzak; Bram Hoex; Haofen Wang; Tong Xie; Wenjie Zhang
>
> **备注:** Accepted by 38th Conference on Neural Information Processing Systems (NeurIPS 2024)
>
> **摘要:** Knowledge in materials science is widely dispersed across extensive scientific literature, posing significant challenges to the efficient discovery and integration of new materials. Traditional methods, often reliant on costly and time-consuming experimental approaches, further complicate rapid innovation. Addressing these challenges, the integration of artificial intelligence with materials science has opened avenues for accelerating the discovery process, though it also demands precise annotation, data extraction, and traceability of information. To tackle these issues, this article introduces the Materials Knowledge Graph (MKG), which utilizes advanced natural language processing techniques integrated with large language models to extract and systematically organize a decade's worth of high-quality research into structured triples, contains 162,605 nodes and 731,772 edges. MKG categorizes information into comprehensive labels such as Name, Formula, and Application, structured around a meticulously designed ontology, thus enhancing data usability and integration. By implementing network-based algorithms, MKG not only facilitates efficient link prediction but also significantly reduces reliance on traditional experimental methods. This structured approach not only streamlines materials research but also lays the groundwork for more sophisticated science knowledge graphs.
>
---
#### [replaced 033] uDistil-Whisper: Label-Free Data Filtering for Knowledge Distillation in Low-Data Regimes
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.01257v5](http://arxiv.org/pdf/2407.01257v5)**

> **作者:** Abdul Waheed; Karima Kadaoui; Bhiksha Raj; Muhammad Abdul-Mageed
>
> **备注:** Accepted to NAACL'25 main conference
>
> **摘要:** Recent work on distilling Whisper's knowledge into small models using pseudo-labels shows promising performance while reducing the size by up to 50%. This results in small, efficient, and dedicated models. However, a critical step of distillation using pseudo-labels involves filtering high-quality predictions and using only those during training. This step requires ground truth labels to compare with and filter low-quality examples, making the process dependent on human labels. Additionally, the distillation process requires a large amount of data thereby limiting its applicability in low-resource settings. To address this, we propose a distillation framework that does not require any labeled data. Through experimentation, we show that our best-distilled models outperform the teacher model by 5-7 WER points and are on par with or outperform similar supervised data filtering setups. When scaling the data, our models significantly outperform all zero-shot and supervised models. Our models are also 25-50% more compute- and memory-efficient while maintaining performance equal to or better than that of the teacher model. For more details about our models, dataset, and other resources, please visit our GitHub page: https://github.com/UBC-NLP/uDistilWhisper.
>
---
#### [replaced 034] CLASH: Evaluating Language Models on Judging High-Stakes Dilemmas from Multiple Perspectives
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.10823v2](http://arxiv.org/pdf/2504.10823v2)**

> **作者:** Ayoung Lee; Ryan Sungmo Kwon; Peter Railton; Lu Wang
>
> **摘要:** Navigating high-stakes dilemmas involving conflicting values is challenging even for humans, let alone for AI. Yet prior work in evaluating the reasoning capabilities of large language models (LLMs) in such situations has been limited to everyday scenarios. To close this gap, this work first introduces CLASH (Character perspective-based LLM Assessments in Situations with High-stakes), a meticulously curated dataset consisting of 345 high-impact dilemmas along with 3,795 individual perspectives of diverse values. In particular, we design CLASH in a way to support the study of critical aspects of value-based decision-making processes which are missing from prior work, including understanding decision ambivalence and psychological discomfort as well as capturing the temporal shifts of values in characters' perspectives. By benchmarking 10 open and closed frontier models, we uncover several key findings. (1) Even the strongest models, such as GPT-4o and Claude-Sonnet, achieve less than 50% accuracy in identifying situations where the decision should be ambivalent, while they perform significantly better in clear-cut scenarios. (2) While LLMs reasonably predict psychological discomfort as marked by human, they inadequately comprehend perspectives involving value shifts, indicating a need for LLMs to reason over complex values. (3) Our experiments also reveal a significant correlation between LLMs' value preferences and their steerability towards a given value. (4) Finally, LLMs exhibit greater steerability when engaged in value reasoning from a third-party perspective, compared to a first-person setup, though certain value pairs benefit uniquely from the first-person framing.
>
---
#### [replaced 035] Conversational Query Reformulation with the Guidance of Retrieved Documents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.12363v5](http://arxiv.org/pdf/2407.12363v5)**

> **作者:** Jeonghyun Park; Hwanhee Lee
>
> **备注:** 18 pages, 3 figures, 16 tables
>
> **摘要:** Conversational search seeks to retrieve relevant passages for the given questions in conversational question answering. Conversational Query Reformulation (CQR) improves conversational search by refining the original queries into de-contextualized forms to resolve the issues in the original queries, such as omissions and coreferences. Previous CQR methods focus on imitating human written queries which may not always yield meaningful search results for the retriever. In this paper, we introduce GuideCQR, a framework that refines queries for CQR by leveraging key information from the initially retrieved documents. Specifically, GuideCQR extracts keywords and generates expected answers from the retrieved documents, then unifies them with the queries after filtering to add useful information that enhances the search process. Experimental results demonstrate that our proposed method achieves state-of-the-art performance across multiple datasets, outperforming previous CQR methods. Additionally, we show that GuideCQR can get additional performance gains in conversational search using various types of queries, even for queries written by humans.
>
---
#### [replaced 036] Pose Priors from Language Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2405.03689v2](http://arxiv.org/pdf/2405.03689v2)**

> **作者:** Sanjay Subramanian; Evonne Ng; Lea Müller; Dan Klein; Shiry Ginosar; Trevor Darrell
>
> **备注:** CVPR 2025
>
> **摘要:** Language is often used to describe physical interaction, yet most 3D human pose estimation methods overlook this rich source of information. We bridge this gap by leveraging large multimodal models (LMMs) as priors for reconstructing contact poses, offering a scalable alternative to traditional methods that rely on human annotations or motion capture data. Our approach extracts contact-relevant descriptors from an LMM and translates them into tractable losses to constrain 3D human pose optimization. Despite its simplicity, our method produces compelling reconstructions for both two-person interactions and self-contact scenarios, accurately capturing the semantics of physical and social interactions. Our results demonstrate that LMMs can serve as powerful tools for contact prediction and pose estimation, offering an alternative to costly manual human annotations or motion capture data. Our code is publicly available at https://prosepose.github.io.
>
---
#### [replaced 037] The Mosaic Memory of Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2405.15523v2](http://arxiv.org/pdf/2405.15523v2)**

> **作者:** Igor Shilov; Matthieu Meeus; Yves-Alexandre de Montjoye
>
> **摘要:** As Large Language Models (LLMs) become widely adopted, understanding how they learn from, and memorize, training data becomes crucial. Memorization in LLMs is widely assumed to only occur as a result of sequences being repeated in the training data. Instead, we show that LLMs memorize by assembling information from similar sequences, a phenomena we call mosaic memory. We show major LLMs to exhibit mosaic memory, with fuzzy duplicates contributing to memorization as much as 0.8 of an exact duplicate and even heavily modified sequences contributing substantially to memorization. Despite models display reasoning capabilities, we somewhat surprisingly show memorization to be predominantly syntactic rather than semantic. We finally show fuzzy duplicates to be ubiquitous in real-world data, untouched by deduplication techniques. Taken together, our results challenge widely held beliefs and show memorization to be a more complex, mosaic process, with real-world implications for privacy, confidentiality, model utility and evaluation.
>
---
#### [replaced 038] Tokenization Matters! Degrading Large Language Models through Challenging Their Tokenization
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.17067v2](http://arxiv.org/pdf/2405.17067v2)**

> **作者:** Dixuan Wang; Yanda Li; Junyuan Jiang; Zepeng Ding; Ziqin Luo; Guochao Jiang; Jiaqing Liang; Deqing Yang
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities in language understanding and generation. Nonetheless, it was also witnessed that LLMs tend to produce inaccurate responses to specific queries. This deficiency can be traced to the tokenization step LLMs must undergo, which is an inevitable limitation inherent to all LLMs. In fact, incorrect tokenization is the critical point that hinders LLMs in understanding the input precisely, thus leading to unsatisfactory output. This defect is more obvious in Chinese scenarios. To demonstrate this flaw of LLMs, we construct an adversarial dataset, named as $\textbf{ADT (Adversarial Dataset for Tokenizer)}$, which draws upon the vocabularies of various open-source LLMs to challenge LLMs' tokenization. ADT consists of two subsets: the manually constructed ADT-Human and the automatically generated ADT-Auto. Our empirical results reveal that our ADT is highly effective on challenging the tokenization of leading LLMs, including GPT-4o, Llama-3, Deepseek-R1 and so on, thus degrading these LLMs' capabilities. Moreover, our method of automatic data generation has been proven efficient and robust, which can be applied to any open-source LLMs. In this paper, we substantially investigate LLMs' vulnerability in terms of challenging their token segmentation, which will shed light on the subsequent research of improving LLMs' capabilities through optimizing their tokenization process and algorithms.
>
---
#### [replaced 039] Natural Language Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.14251v2](http://arxiv.org/pdf/2411.14251v2)**

> **作者:** Xidong Feng; Bo Liu; Ziyu Wan; Haotian Fu; Girish A. Koushik; Zhiyuan Hu; Mengyue Yang; Ying Wen; Jun Wang
>
> **备注:** Accepted at ICLR 2025 Workshop SSI-FM
>
> **摘要:** Reinforcement Learning (RL) mathematically formulates decision-making with Markov Decision Process (MDP). With MDPs, researchers have achieved remarkable breakthroughs across various domains, including games, robotics, and language models. This paper seeks a new possibility, Natural Language Reinforcement Learning (NLRL), by extending traditional MDP to natural language-based representation space. Specifically, NLRL innovatively redefines RL principles, including task objectives, policy, value function, Bellman equation, and policy iteration, into their language counterparts. With recent advancements in large language models (LLMs), NLRL can be practically implemented to achieve RL-like policy and value improvement by either pure prompting or gradient-based training. Experiments over Maze, Breakthrough, and Tic-Tac-Toe games demonstrate the effectiveness, efficiency, and interpretability of the NLRL framework among diverse use cases.
>
---
#### [replaced 040] Compensate Quantization Errors+: Quantized Models Are Inquisitive Learners
- **分类: cs.CL; cs.AI; I.2.7**

- **链接: [http://arxiv.org/pdf/2407.15508v3](http://arxiv.org/pdf/2407.15508v3)**

> **作者:** Yifei Gao; Jie Ou; Lei Wang; Jun Cheng; Mengchu Zhou
>
> **备注:** Effecient Quantization Methods for LLMs
>
> **摘要:** The quantization of large language models (LLMs) has been a prominent research area aimed at enabling their lightweight deployment in practice. Existing research about LLM's quantization has mainly explored the interplay between weights and activations, or employing auxiliary components while neglecting the necessity of adjusting weights during quantization. Consequently, original weight distributions frequently fail to yield desired results after round-to-nearest (RTN) quantization. Even though incorporating techniques such as mixed precision and low-rank error approximation in LLM's quantization can yield improved results, they inevitably introduce additional computational overhead. On the other hand, traditional techniques for weight quantization, such as Generative Post-Training Quantization, rely on manually tweaking weight distributions to minimize local errors, but they fall short of achieving globally optimal outcomes. Although the recently proposed Learnable Singular-value Increment improves global weight quantization by modifying weight distributions, it disrupts the original distribution considerably. This introduces pronounced bias toward the training data and can degrade downstream task performance. In this paper, we introduce Singular-value Diagonal Expansion, a more nuanced approach to refining weight distributions to achieve better quantization alignment. Furthermore, we introduce Cross-layer Learning that improves overall quantization outcomes by distributing errors more evenly across layers. Our plug-and-play weight-quantization methods demonstrate substantial performance improvements over state-of-the-art approaches, including OmniQuant, DuQuant, and PrefixQuant.
>
---
#### [replaced 041] KBAlign: Efficient Self Adaptation on Specific Knowledge Bases
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.14790v4](http://arxiv.org/pdf/2411.14790v4)**

> **作者:** Zheni Zeng; Yuxuan Chen; Shi Yu; Ruobing Wang; Yukun Yan; Zhenghao Liu; Shuo Wang; Xu Han; Zhiyuan Liu; Maosong Sun
>
> **摘要:** Although retrieval-augmented generation (RAG) remains essential for knowledge-based question answering (KBQA), current paradigms face critical challenges under specific domains. Existing methods struggle with targeted adaptation on small-scale KBs: vanilla unsupervised training exhibits poor effectiveness, while fine-tuning incurs prohibitive costs of external signals. We present KBAlign, a self-supervised framework that enhances RAG systems through efficient model adaptation. Our key insight is to leverage the model's intrinsic capabilities for knowledge alignment through two innovative mechanisms: multi-grained self-annotation that captures global knowledge for data construction, and iterative tuning that accelerates convergence through self verification. This framework enables cost-effective model adaptation to specific textual KBs, without human supervision or external model assistance. Experiments demonstrate that KBAlign can achieve 90\% of the performance gain obtained through GPT-4-supervised adaptation, while relying entirely on self-annotation of much smaller models. KBAlign significantly improves downstream QA accuracy across multiple domains with tiny costs, particularly benefiting scenarios requiring deep knowledge integration from specialized corpora. We release our experimental data, models, and process analyses to the community for further exploration (https://github.com/thunlp/KBAlign).
>
---
#### [replaced 042] 100 Days After DeepSeek-R1: A Survey on Replication Studies and More Directions for Reasoning Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00551v3](http://arxiv.org/pdf/2505.00551v3)**

> **作者:** Chong Zhang; Yue Deng; Xiang Lin; Bin Wang; Dianwen Ng; Hai Ye; Xingxuan Li; Yao Xiao; Zhanfeng Mo; Qi Zhang; Lidong Bing
>
> **摘要:** The recent development of reasoning language models (RLMs) represents a novel evolution in large language models. In particular, the recent release of DeepSeek-R1 has generated widespread social impact and sparked enthusiasm in the research community for exploring the explicit reasoning paradigm of language models. However, the implementation details of the released models have not been fully open-sourced by DeepSeek, including DeepSeek-R1-Zero, DeepSeek-R1, and the distilled small models. As a result, many replication studies have emerged aiming to reproduce the strong performance achieved by DeepSeek-R1, reaching comparable performance through similar training procedures and fully open-source data resources. These works have investigated feasible strategies for supervised fine-tuning (SFT) and reinforcement learning from verifiable rewards (RLVR), focusing on data preparation and method design, yielding various valuable insights. In this report, we provide a summary of recent replication studies to inspire future research. We primarily focus on SFT and RLVR as two main directions, introducing the details for data construction, method design and training procedure of current replication studies. Moreover, we conclude key findings from the implementation details and experimental results reported by these studies, anticipating to inspire future research. We also discuss additional techniques of enhancing RLMs, highlighting the potential of expanding the application scope of these models, and discussing the challenges in development. By this survey, we aim to help researchers and developers of RLMs stay updated with the latest advancements, and seek to inspire new ideas to further enhance RLMs.
>
---
#### [replaced 043] ChronoFact: Timeline-based Temporal Fact Verification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.14964v2](http://arxiv.org/pdf/2410.14964v2)**

> **作者:** Anab Maulana Barik; Wynne Hsu; Mong Li Lee
>
> **摘要:** Temporal claims, often riddled with inaccuracies, are a significant challenge in the digital misinformation landscape. Fact-checking systems that can accurately verify such claims are crucial for combating misinformation. Current systems struggle with the complexities of evaluating the accuracy of these claims, especially when they include multiple, overlapping, or recurring events. We introduce a novel timeline-based fact verification framework that identify events from both claim and evidence and organize them into their respective chronological timelines. The framework systematically examines the relationships between the events in both claim and evidence to predict the veracity of each claim event and their chronological accuracy. This allows us to accurately determine the overall veracity of the claim. We also introduce a new dataset of complex temporal claims involving timeline-based reasoning for the training and evaluation of our proposed framework. Experimental results demonstrate the effectiveness of our approach in handling the intricacies of temporal claim verification.
>
---
#### [replaced 044] FAMMA: A Benchmark for Financial Domain Multilingual Multimodal Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.04526v4](http://arxiv.org/pdf/2410.04526v4)**

> **作者:** Siqiao Xue; Xiaojing Li; Fan Zhou; Qingyang Dai; Zhixuan Chu; Hongyuan Mei
>
> **摘要:** In this paper, we introduce FAMMA, an open-source benchmark for \underline{f}in\underline{a}ncial \underline{m}ultilingual \underline{m}ultimodal question \underline{a}nswering (QA). Our benchmark aims to evaluate the abilities of large language models (LLMs) in answering complex reasoning questions that require advanced financial knowledge. The benchmark has two versions: FAMMA-Basic consists of 1,945 questions extracted from university textbooks and exams, along with human-annotated answers and rationales; FAMMA-LivePro consists of 103 novel questions created by human domain experts, with answers and rationales held out from the public for a contamination-free evaluation. These questions cover advanced knowledge of 8 major subfields in finance (e.g., corporate finance, derivatives, and portfolio management). Some are in Chinese or French, while a majority of them are in English. Each question has some non-text data such as charts, diagrams, or tables. Our experiments reveal that FAMMA poses a significant challenge on LLMs, including reasoning models such as GPT-o1 and DeepSeek-R1. Additionally, we curated 1,270 reasoning trajectories of DeepSeek-R1 on the FAMMA-Basic data, and fine-tuned a series of open-source Qwen models using this reasoning data. We found that training a model on these reasoning trajectories can significantly improve its performance on FAMMA-LivePro. We released our leaderboard, data, code, and trained models at https://famma-bench.github.io/famma/.
>
---
