# 自然语言处理 cs.CL

- **最新发布 62 篇**

- **更新 52 篇**

## 最新发布

#### [new 001] Fast-Decoding Diffusion Language Models via Progress-Aware Confidence Schedules
- **分类: cs.CL**

- **简介: 该论文针对扩散语言模型（dLLM）解码慢的问题，提出无需训练的SchED算法，通过进度感知置信度阈值实现早期退出。在多任务基准上，显著提升速度（最高4倍），同时保持99%以上性能，有效加速长文本生成。**

- **链接: [https://arxiv.org/pdf/2512.02892v1](https://arxiv.org/pdf/2512.02892v1)**

> **作者:** Amr Mohamed; Yang Zhang; Michalis Vazirgiannis; Guokan Shang
>
> **摘要:** Diffusion large language models (dLLMs) offer a promising alternative to autoregressive models, but their practical utility is severely hampered by slow, iterative sampling. We present SchED, a training-free, model-agnostic early-exit algorithm that aggregates full-span logit margins and halts decoding once a smooth, progress-dependent confidence threshold is met. We evaluated SchED on two dLLM families (Dream and LLaDA), in base and instruction-tuned variants across ten benchmarks spanning downstream tasks including multiple-choice question answering (MCQ), math, long-form QA/summarization, and translation. SchED delivers large, stable accelerations: on instruction-tuned models, it achieves $3.8$-$4.0\times$ speedups while retaining $99.8$-$100\%$ of the baseline score on average. On base models, SchED yields consistent speedup gains with $99.1$-$100\%$ performance retention, with up to $2.34\times$ under more aggressive settings. Using a conservative speed metric that heavily penalizes quality loss (QPS, $γ{=}4$), we show that SchED is robust and clearly outperforms prior confidence-based early-exit methods, which break down on long-form generation. An entropy analysis of the model's token predictions reveals that instruction tuning speeds up the decay of predictive entropy. By turning genuine confidence stabilization into computational savings, SchED makes dLLM decoding substantially more efficient.
>
---
#### [new 002] Bangla Hate Speech Classification with Fine-tuned Transformer Models
- **分类: cs.CL**

- **简介: 该论文针对低资源语言孟加拉语的仇恨言论分类任务，解决数据稀缺与语言多样性难题。研究参与BLP 2025共享任务，对比传统模型与多种Transformer模型，发现专为孟加拉语预训练的BanglaBERT表现最优，验证了领域特定预训练的重要性。**

- **链接: [https://arxiv.org/pdf/2512.02845v1](https://arxiv.org/pdf/2512.02845v1)**

> **作者:** Yalda Keivan Jafari; Krishno Dey
>
> **摘要:** Hate speech recognition in low-resource lan- guages remains a difficult problem due to in- sufficient datasets, orthographic heterogeneity, and linguistic variety. Bangla is spoken by more than 230 million people of Bangladesh and India (West Bengal). Despite the grow- ing need for automated moderation on social media platforms, Bangla is significantly under- represented in computational resources. In this work, we study Subtask 1A and Subtask 1B of the BLP 2025 Shared Task on hate speech detection. We reproduce the official base- lines (e.g., Majority, Random, Support Vec- tor Machine) and also produce and consider Logistic Regression, Random Forest, and De- cision Tree as baseline methods. We also uti- lized transformer-based models such as Dis- tilBERT, BanglaBERT, m-BERT, and XLM- RoBERTa for hate speech classification. All the transformer-based models outperformed base- line methods for the subtasks, except for Distil- BERT. Among the transformer-based models, BanglaBERT produces the best performance for both subtasks. Despite being smaller in size, BanglaBERT outperforms both m-BERT and XLM-RoBERTa, which suggests language- specific pre-training is very important. Our results highlight the potential and need for pre- trained language models for the low-resource Bangla language.
>
---
#### [new 003] TriLex: A Framework for Multilingual Sentiment Analysis in Low-Resource South African Languages
- **分类: cs.CL**

- **简介: 该论文针对低资源南非语言在情感分析中词汇覆盖不足的问题，提出TriLex框架，通过三阶段方法融合语料抽取、跨语言映射与检索增强生成，扩展情感词典。实验验证了其对AfroXLMR和AfriBERTa模型的有效性，显著提升多语言情感分析性能。**

- **链接: [https://arxiv.org/pdf/2512.02799v1](https://arxiv.org/pdf/2512.02799v1)**

> **作者:** Mike Nkongolo; Hilton Vorster; Josh Warren; Trevor Naick; Deandre Vanmali; Masana Mashapha; Luke Brand; Alyssa Fernandes; Janco Calitz; Sibusiso Makhoba
>
> **摘要:** Low-resource African languages remain underrepresented in sentiment analysis, limiting both lexical coverage and the performance of multilingual Natural Language Processing (NLP) systems. This study proposes TriLex, a three-stage retrieval augmented framework that unifies corpus-based extraction, cross lingual mapping, and retrieval augmented generation (RAG) driven lexical refinement to systematically expand sentiment lexicons for low-resource languages. Using the enriched lexicon, the performance of two prominent African pretrained language models (AfroXLMR and AfriBERTa) is evaluated across multiple case studies. Results demonstrate that AfroXLMR delivers superior performance, achieving F1-scores above 80% for isiXhosa and isiZulu and exhibiting strong cross-lingual stability. Although AfriBERTa lacks pre-training on these target languages, it still achieves reliable F1-scores around 64%, validating its utility in computationally constrained settings. Both models outperform traditional machine learning baselines, and ensemble analyses further enhance precision and robustness. The findings establish TriLex as a scalable and effective framework for multilingual sentiment lexicon expansion and sentiment modeling in low-resource South African languages.
>
---
#### [new 004] Towards Unification of Hallucination Detection and Fact Verification for Large Language Models
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对大语言模型的幻觉问题，提出统一框架UniFact，整合幻觉检测（HD）与事实验证（FV）任务。通过动态生成输出与标签，实现两者在实例级的直接比较，揭示其互补性，并证明融合方法性能最优，推动两领域协同发展。**

- **链接: [https://arxiv.org/pdf/2512.02772v1](https://arxiv.org/pdf/2512.02772v1)**

> **作者:** Weihang Su; Jianming Long; Changyue Wang; Shiyu Lin; Jingyan Xu; Ziyi Ye; Qingyao Ai; Yiqun Liu
>
> **摘要:** Large Language Models (LLMs) frequently exhibit hallucinations, generating content that appears fluent and coherent but is factually incorrect. Such errors undermine trust and hinder their adoption in real-world applications. To address this challenge, two distinct research paradigms have emerged: model-centric Hallucination Detection (HD) and text-centric Fact Verification (FV). Despite sharing the same goal, these paradigms have evolved in isolation, using distinct assumptions, datasets, and evaluation protocols. This separation has created a research schism that hinders their collective progress. In this work, we take a decisive step toward bridging this divide. We introduce UniFact, a unified evaluation framework that enables direct, instance-level comparison between FV and HD by dynamically generating model outputs and corresponding factuality labels. Through large-scale experiments across multiple LLM families and detection methods, we reveal three key findings: (1) No paradigm is universally superior; (2) HD and FV capture complementary facets of factual errors; and (3) hybrid approaches that integrate both methods consistently achieve state-of-the-art performance. Beyond benchmarking, we provide the first in-depth analysis of why FV and HD diverged, as well as empirical evidence supporting the need for their unification. The comprehensive experimental results call for a new, integrated research agenda toward unifying Hallucination Detection and Fact Verification in LLMs. We have open-sourced all the code, data, and baseline implementation at: https://github.com/oneal2000/UniFact/
>
---
#### [new 005] Lightweight Latent Reasoning for Narrative Tasks
- **分类: cs.CL**

- **简介: 该论文针对叙事任务中大模型推理过程冗长、计算成本高的问题，提出LiteReason方法。通过轻量级投影模块生成连续隐式令牌，使模型可跳过部分推理步骤。结合强化学习，在保持性能的同时显著缩短推理长度77-92%，提升效率。**

- **链接: [https://arxiv.org/pdf/2512.02240v1](https://arxiv.org/pdf/2512.02240v1)**

> **作者:** Alexander Gurung; Nikolay Malkin; Mirella Lapata
>
> **摘要:** Large language models (LLMs) tackle complex tasks by generating long chains of thought or "reasoning traces" that act as latent variables in the generation of an output given a query. A model's ability to generate such traces can be optimized with reinforcement learning (RL) to improve their utility in predicting an answer. This optimization comes at a high computational cost, especially for narrative-related tasks that involve retrieving and processing many tokens. To this end, we propose LiteReason, a latent reasoning method that can be interleaved with standard token sampling and easily combined with RL techniques. LiteReason employs a lightweight Reasoning Projector module, trained to produce continuous latent tokens that help the model 'skip' reasoning steps. During RL, the policy model decides when to activate the projector, switching between latent and discrete reasoning as needed. Experimental results on plot hole detection and book chapter generation show that our method outperforms latent reasoning baselines and comes close to matching non-latent RL training, while reducing final reasoning length by 77-92%. Overall, LiteReason guides RL training to a more efficient part of the performance-computation tradeoff curve.
>
---
#### [new 006] Reversing Large Language Models for Efficient Training and Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLM）训练与微调中内存消耗大的问题，提出基于可逆架构的高效方法。通过利用时间可逆动力学，无需存储中间激活，显著降低内存使用，支持更大批量处理。同时提出将现有非可逆模型转换为可逆架构的微调方法，实现对预训练模型的高效利用，在保持性能的同时大幅减少计算成本。**

- **链接: [https://arxiv.org/pdf/2512.02056v1](https://arxiv.org/pdf/2512.02056v1)**

> **作者:** Eshed Gal; Moshe Eliasof; Javier Turek; Uri Ascher; Eran Treister; Eldad Haber
>
> **摘要:** Large Language Models (LLMs) are known for their expensive and time-consuming training. Thus, oftentimes, LLMs are fine-tuned to address a specific task, given the pretrained weights of a pre-trained LLM considered a foundation model. In this work, we introduce memory-efficient, reversible architectures for LLMs, inspired by symmetric and symplectic differential equations, and investigate their theoretical properties. Different from standard, baseline architectures that store all intermediate activations, the proposed models use time-reversible dynamics to retrieve hidden states during backpropagation, relieving the need to store activations. This property allows for a drastic reduction in memory consumption, allowing for the processing of larger batch sizes for the same available memory, thereby offering improved throughput. In addition, we propose an efficient method for converting existing, non-reversible LLMs into reversible architectures through fine-tuning, rendering our approach practical for exploiting existing pre-trained models. Our results show comparable or improved performance on several datasets and benchmarks, on several LLMs, building a scalable and efficient path towards reducing the memory and computational costs associated with both training from scratch and fine-tuning of LLMs.
>
---
#### [new 007] Memory-Augmented Knowledge Fusion with Safety-Aware Decoding for Domain-Adaptive Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对服务场景下领域自适应问答中知识融合与安全性的挑战，提出KARMA框架。通过双编码器融合异构知识，动态记忆单元调控外部信息，安全感知解码机制保障输出可靠性，有效提升答案准确率与安全性。**

- **链接: [https://arxiv.org/pdf/2512.02363v1](https://arxiv.org/pdf/2512.02363v1)**

> **作者:** Lei Fu; Xiang Chen; Kaige Gao Xinyue Huang; Kejian Tong
>
> **摘要:** Domain-specific question answering (QA) systems for services face unique challenges in integrating heterogeneous knowledge sources while ensuring both accuracy and safety. Existing large language models often struggle with factual consistency and context alignment in sensitive domains such as healthcare policies and government welfare. In this work, we introduce Knowledge-Aware Reasoning and Memory-Augmented Adaptation (KARMA), a novel framework designed to enhance QA performance in care scenarios. KARMA incorporates a dual-encoder architecture to fuse structured and unstructured knowledge sources, a gated memory unit to dynamically regulate external knowledge integration, and a safety-aware controllable decoder that mitigates unsafe outputs using safety classification and guided generation techniques. Extensive experiments on a proprietary QA dataset demonstrate that KARMA outperforms strong baselines in both answer quality and safety. This study offers a comprehensive solution for building trustworthy and adaptive QA systems in service contexts.
>
---
#### [new 008] A Concise Review of Hallucinations in LLMs and their Mitigation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的问题分析与综述任务，旨在探讨大语言模型（LLMs）中的幻觉现象。论文系统梳理了幻觉的类型、成因，并总结了当前主流的缓解方法，为理解与应对幻觉提供简明指南。**

- **链接: [https://arxiv.org/pdf/2512.02527v1](https://arxiv.org/pdf/2512.02527v1)**

> **作者:** Parth Pulkundwar; Vivek Dhanawade; Rohit Yadav; Minal Sonkar; Medha Asurlekar; Sarita Rathod
>
> **备注:** 7 pages
>
> **摘要:** Traditional language models face a challenge from hallucinations. Their very presence casts a large, dangerous shadow over the promising realm of natural language processing. It becomes crucial to understand the various kinds of hallucinations that occur nowadays, their origins, and ways of reducing them. This document provides a concise and straightforward summary of that. It serves as a one-stop resource for a general understanding of hallucinations and how to mitigate them.
>
---
#### [new 009] TaleFrame: An Interactive Story Generation System with Fine-Grained Control and Large Language Models
- **分类: cs.CL; cs.HC**

- **简介: 该论文提出TaleFrame系统，针对故事生成中用户意图难以精准实现的问题，结合大语言模型与人机交互，将故事分解为实体、事件、关系和大纲四要素，通过结构化数据生成可精细控制的连贯故事，支持交互式编辑与多维度评估，提升生成效果与用户参与度。**

- **链接: [https://arxiv.org/pdf/2512.02402v1](https://arxiv.org/pdf/2512.02402v1)**

> **作者:** Yunchao Wang; Guodao Sun; Zihang Fu; Zhehao Liu; Kaixing Du; Haidong Gao; Ronghua Liang
>
> **备注:** 11 pages
>
> **摘要:** With the advancement of natural language generation (NLG) technologies, creative story generation systems have gained increasing attention. However, current systems often fail to accurately translate user intent into satisfactory story outputs due to a lack of fine-grained control and unclear input specifications, limiting their applicability. To address this, we propose TaleFrame, a system that combines large language models (LLMs) with human-computer interaction (HCI) to generate stories through structured information, enabling precise control over the generation process. The innovation of TaleFrame lies in decomposing the story structure into four basic units: entities, events, relationships, and story outline. We leverage the Tinystories dataset, parsing and constructing a preference dataset consisting of 9,851 JSON-formatted entries, which is then used to fine-tune a local Llama model. By employing this JSON2Story approach, structured data is transformed into coherent stories. TaleFrame also offers an intuitive interface that supports users in creating and editing entities and events and generates stories through the structured framework. Users can control these units through simple interactions (e.g., drag-and-drop, attach, and connect), thus influencing the details and progression of the story. The generated stories can be evaluated across seven dimensions (e.g., creativity, structural integrity), with the system providing suggestions for refinement based on these evaluations. Users can iteratively adjust the story until a satisfactory result is achieved. Finally, we conduct quantitative evaluation and user studies that demonstrate the usefulness of TaleFrame. Dataset available at https://huggingface.co/datasets/guodaosun/tale-frame.
>
---
#### [new 010] Fine-Tuned Large Language Models for Logical Translation: Reducing Hallucinations with Lang2Logic
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对自然语言到逻辑的精准翻译任务，解决大模型在逻辑翻译中易产生幻觉的问题。通过结合自定义语法、符号计算与微调语言模型，将英文语句转化为可满足性求解的合取范式（CNF），有效减少错误输出，提升逻辑表达的可靠性。**

- **链接: [https://arxiv.org/pdf/2512.02987v1](https://arxiv.org/pdf/2512.02987v1)**

> **作者:** Muyu Pan; Dheeraj Kodakandla; Mahfuza Farooque
>
> **备注:** IEEE ISNCC 2025
>
> **摘要:** Recent advances in natural language processing (NLP), particularly large language models (LLMs), have motivated the automatic translation of natural language statements into formal logic without human intervention. This enables automated reasoning and facilitates debugging, finding loop invariants, and adhering to specifications in software systems. However, hallucinations-incorrect outputs generated by LLMs are challenging, particularly for logical translation tasks requiring precision. This work introduces a novel framework that inputs English sentences, converts them into logical expressions, and then translates them into Conjunctive Normal Form (CNF) for satisfiability solving. It employs classical NLP techniques with self-defined grammar, symbolic computation libraries, and a fine-tuned language model to reduce hallucinations. In the early experiments, we observed that the fine-tuned model, trained on different grammar settings, could intentionally correct the same types of hallucinations made by the original model. Thus, it provides reliable CNF generation.
>
---
#### [new 011] An Empirical Survey of Model Merging Algorithms for Social Bias Mitigation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的社会偏见缓解任务。针对模型继承并放大训练数据中的偏见问题，作者对七种模型合并算法进行实证比较，使用多类开源模型和多个评测数据集，发现部分方法虽能有效减缓偏见，但会损害模型在阅读理解等任务上的性能，其中SLERP在适度插值下表现最均衡。**

- **链接: [https://arxiv.org/pdf/2512.02689v1](https://arxiv.org/pdf/2512.02689v1)**

> **作者:** Daiki Shirafuji; Tatsuhiko Saito; Yasutomo Kimura
>
> **备注:** Accepted in PACLIC 2025
>
> **摘要:** Large language models (LLMs) are known to inherit and even amplify societal biases present in their pre-training corpora, threatening fairness and social trust. To address this issue, recent work has explored ``editing'' LLM parameters to mitigate social bias with model merging approaches; however, there is no empirical comparison. In this work, we empirically survey seven algorithms: Linear, Karcher Mean, SLERP, NuSLERP, TIES, DELLA, and Nearswap, applying 13 open weight models in the GPT, LLaMA, and Qwen families. We perform a comprehensive evaluation using three bias datasets (BBQ, BOLD, and HONEST) and measure the impact of these techniques on LLM performance in downstream tasks of the SuperGLUE benchmark. We find a trade-off between bias reduction and downstream performance: methods achieving greater bias mitigation degrade accuracy, particularly on tasks requiring reading comprehension and commonsense and causal reasoning. Among the merging algorithms, Linear, SLERP, and Nearswap consistently reduce bias while maintaining overall performance, with SLERP at moderate interpolation weights emerging as the most balanced choice. These results highlight the potential of model merging algorithms for bias mitigation, while indicating that excessive debiasing or inappropriate merging methods may lead to the degradation of important linguistic abilities.
>
---
#### [new 012] Beyond Confidence: Adaptive and Coherent Decoding for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文针对扩散语言模型（DLM）推理中因依赖局部指标导致生成轨迹不一致、质量低的问题，提出自适应协同解码（CCD）框架。通过历史上下文增强序列一致性，并设计动态采样策略优化解码预算分配，显著提升生成质量与速度，在多个基准上实现3.48倍加速和3.91%性能提升。**

- **链接: [https://arxiv.org/pdf/2512.02044v1](https://arxiv.org/pdf/2512.02044v1)**

> **作者:** Kecheng Chen; Ziru Liu; Xijia Tao; Hui Liu; Xinyu Fu; Suiyun Zhang; Dandan Tu; Lingpeng Kong; Rui Liu; Haoliang Li
>
> **摘要:** Diffusion Language Models (DLMs) have recently achieved significant success due to their any-order generation capabilities. However, existing inference methods typically rely on local, immediate-step metrics such as confidence or entropy which inherently lack a more reliable perspective. This limitation frequently leads to inconsistent sampling trajectories and suboptimal generation quality. To address this, we propose Coherent Contextual Decoding (CCD), a novel inference framework built upon two core innovations. First, CCD employs a trajectory rectification mechanism that leverages historical context to enhance sequence coherence, enabling the early rejection of suboptimal paths. We demonstrate that this mechanism is theoretically equivalent to modeling the consistency of historical steps via the conditional mutual information between context and token predictions. Building on this theoretical insight, we further address the inefficiency of conventional uniform decoding budgets. Instead of rigid allocations based on diffusion steps, we introduce an adaptive sampling strategy that dynamically adjusts the unmasking budget for each step according to our consistency metric. Consequently, our method significantly improves the quality of generation trajectories while accelerating the sampling process. Empirically, our method achieves a simultaneous enhancement in both inference speed and performance across diverse benchmarks on Dream and LLaDA, delivering up to 3.48x speedup alongside 3.91% performance improvement.
>
---
#### [new 013] Swivuriso: The South African Next Voices Multilingual Speech Dataset
- **分类: cs.CL**

- **简介: 该论文提出Swivuriso，一个3000小时的多语言语音数据集，涵盖7种南非语言，聚焦农业、医疗和通用领域。旨在解决非洲语言ASR数据匮乏问题，支持语音识别技术发展。论文阐述数据设计与伦理规范，提供基线模型结果，验证数据有效性。**

- **链接: [https://arxiv.org/pdf/2512.02201v1](https://arxiv.org/pdf/2512.02201v1)**

> **作者:** Vukosi Marivatee; Kayode Olaleye; Sitwala Mundia; Andinda Bakainga; Unarine Netshifhefhe; Mahmooda Milanzie; Tsholofelo Hope Mogale; Thapelo Sindane; Zainab Abdulrasaq; Kesego Mokgosi; Chijioke Okorie; Nia Zion Van Wyk; Graham Morrissey; Dale Dunbar; Francois Smit; Tsosheletso Chidi; Rooweither Mabuya; Andiswa Bukula; Respect Mlambo; Tebogo Macucwa; Idris Abdulmumin; and Seani Rananga
>
> **备注:** Work in Progress
>
> **摘要:** This paper introduces Swivuriso, a 3000-hour multilingual speech dataset developed as part of the African Next Voices project, to support the development and benchmarking of automatic speech recognition (ASR) technologies in seven South African languages. Covering agriculture, healthcare, and general domain topics, Swivuriso addresses significant gaps in existing ASR datasets. We describe the design principles, ethical considerations, and data collection procedures that guided the dataset creation. We present baseline results of training/finetuning ASR models with this data and compare to other ASR datasets for the langauges concerned.
>
---
#### [new 014] DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出DeepSeek-V3.2，面向开放大语言模型的高效推理与智能代理任务。针对长文本计算效率低、复杂任务泛化能力弱的问题，提出稀疏注意力机制、可扩展强化学习框架及大规模代理任务合成管道，显著提升推理与工具使用性能，在数学与信息学竞赛中达顶尖水平。**

- **链接: [https://arxiv.org/pdf/2512.02556v1](https://arxiv.org/pdf/2512.02556v1)**

> **作者:** DeepSeek-AI; Aixin Liu; Aoxue Mei; Bangcai Lin; Bing Xue; Bingxuan Wang; Bingzheng Xu; Bochao Wu; Bowei Zhang; Chaofan Lin; Chen Dong; Chengda Lu; Chenggang Zhao; Chengqi Deng; Chenhao Xu; Chong Ruan; Damai Dai; Daya Guo; Dejian Yang; Deli Chen; Erhang Li; Fangqi Zhou; Fangyun Lin; Fucong Dai; Guangbo Hao; Guanting Chen; Guowei Li; H. Zhang; Hanwei Xu; Hao Li; Haofen Liang; Haoran Wei; Haowei Zhang; Haowen Luo; Haozhe Ji; Honghui Ding; Hongxuan Tang; Huanqi Cao; Huazuo Gao; Hui Qu; Hui Zeng; Jialiang Huang; Jiashi Li; Jiaxin Xu; Jiewen Hu; Jingchang Chen; Jingting Xiang; Jingyang Yuan; Jingyuan Cheng; Jinhua Zhu; Jun Ran; Junguang Jiang; Junjie Qiu; Junlong Li; Junxiao Song; Kai Dong; Kaige Gao; Kang Guan; Kexin Huang; Kexing Zhou; Kezhao Huang; Kuai Yu; Lean Wang; Lecong Zhang; Lei Wang; Liang Zhao; Liangsheng Yin; Lihua Guo; Lingxiao Luo; Linwang Ma; Litong Wang; Liyue Zhang; M. S. Di; M. Y Xu; Mingchuan Zhang; Minghua Zhang; Minghui Tang; Mingxu Zhou; Panpan Huang; Peixin Cong; Peiyi Wang; Qiancheng Wang; Qihao Zhu; Qingyang Li; Qinyu Chen; Qiushi Du; Ruiling Xu; Ruiqi Ge; Ruisong Zhang; Ruizhe Pan; Runji Wang; Runqiu Yin; Runxin Xu; Ruomeng Shen; Ruoyu Zhang; S. H. Liu; Shanghao Lu; Shangyan Zhou; Shanhuang Chen; Shaofei Cai; Shaoyuan Chen; Shengding Hu; Shengyu Liu; Shiqiang Hu; Shirong Ma; Shiyu Wang; Shuiping Yu; Shunfeng Zhou; Shuting Pan; Songyang Zhou; Tao Ni; Tao Yun; Tian Pei; Tian Ye; Tianyuan Yue; Wangding Zeng; Wen Liu; Wenfeng Liang; Wenjie Pang; Wenjing Luo; Wenjun Gao; Wentao Zhang; Xi Gao; Xiangwen Wang; Xiao Bi; Xiaodong Liu; Xiaohan Wang; Xiaokang Chen; Xiaokang Zhang; Xiaotao Nie; Xin Cheng; Xin Liu; Xin Xie; Xingchao Liu; Xingkai Yu; Xingyou Li; Xinyu Yang; Xinyuan Li; Xu Chen; Xuecheng Su; Xuehai Pan; Xuheng Lin; Xuwei Fu; Y. Q. Wang; Yang Zhang; Yanhong Xu; Yanru Ma; Yao Li; Yao Li; Yao Zhao; Yaofeng Sun; Yaohui Wang; Yi Qian; Yi Yu; Yichao Zhang; Yifan Ding; Yifan Shi; Yiliang Xiong; Ying He; Ying Zhou; Yinmin Zhong; Yishi Piao; Yisong Wang; Yixiao Chen; Yixuan Tan; Yixuan Wei; Yiyang Ma; Yiyuan Liu; Yonglun Yang; Yongqiang Guo; Yongtong Wu; Yu Wu; Yuan Cheng; Yuan Ou; Yuanfan Xu; Yuduan Wang; Yue Gong; Yuhan Wu; Yuheng Zou; Yukun Li; Yunfan Xiong; Yuxiang Luo; Yuxiang You; Yuxuan Liu; Yuyang Zhou; Z. F. Wu; Z. Z. Ren; Zehua Zhao; Zehui Ren; Zhangli Sha; Zhe Fu; Zhean Xu; Zhenda Xie; Zhengyan Zhang; Zhewen Hao; Zhibin Gou; Zhicheng Ma; Zhigang Yan; Zhihong Shao; Zhixian Huang; Zhiyu Wu; Zhuoshu Li; Zhuping Zhang; Zian Xu; Zihao Wang; Zihui Gu; Zijia Zhu; Zilin Li; Zipeng Zhang; Ziwei Xie; Ziyi Gao; Zizheng Pan; Zongqing Yao; Bei Feng; Hui Li; J. L. Cai; Jiaqi Ni; Lei Xu; Meng Li; Ning Tian; R. J. Chen; R. L. Jin; S. S. Li; Shuang Zhou; Tianyu Sun; X. Q. Li; Xiangyue Jin; Xiaojin Shen; Xiaosha Chen; Xinnan Song; Xinyi Zhou; Y. X. Zhu; Yanping Huang; Yaohui Li; Yi Zheng; Yuchen Zhu; Yunxian Ma; Zhen Huang; Zhipeng Xu; Zhongyu Zhang; Dongjie Ji; Jian Liang; Jianzhong Guo; Jin Chen; Leyi Xia; Miaojun Wang; Mingming Li; Peng Zhang; Ruyi Chen; Shangmian Sun; Shaoqing Wu; Shengfeng Ye; T. Wang; W. L. Xiao; Wei An; Xianzu Wang; Xiaowen Sun; Xiaoxiang Wang; Ying Tang; Yukun Zha; Zekai Zhang; Zhe Ju; Zhen Zhang; Zihua Qu
>
> **摘要:** We introduce DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance. The key technical breakthroughs of DeepSeek-V3.2 are as follows: (1) DeepSeek Sparse Attention (DSA): We introduce DSA, an efficient attention mechanism that substantially reduces computational complexity while preserving model performance in long-context scenarios. (2) Scalable Reinforcement Learning Framework: By implementing a robust reinforcement learning protocol and scaling post-training compute, DeepSeek-V3.2 performs comparably to GPT-5. Notably, our high-compute variant, DeepSeek-V3.2-Speciale, surpasses GPT-5 and exhibits reasoning proficiency on par with Gemini-3.0-Pro, achieving gold-medal performance in both the 2025 International Mathematical Olympiad (IMO) and the International Olympiad in Informatics (IOI). (3) Large-Scale Agentic Task Synthesis Pipeline: To integrate reasoning into tool-use scenarios, we developed a novel synthesis pipeline that systematically generates training data at scale. This methodology facilitates scalable agentic post-training, yielding substantial improvements in generalization and instruction-following robustness within complex, interactive environments.
>
---
#### [new 015] CAIRNS: Balancing Readability and Scientific Accuracy in Climate Adaptation Question Answering
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出CAIRNS框架，解决气候适应问答中可读性与科学准确性难以兼顾的问题。针对农业领域专家从复杂网络数据获取可信答案的需求，通过结构化提示增强可读性与引用可靠性，并采用一致性加权混合评估器实现无需微调的鲁棒评价，显著提升问答质量。**

- **链接: [https://arxiv.org/pdf/2512.02251v1](https://arxiv.org/pdf/2512.02251v1)**

> **作者:** Liangji Kong; Aditya Joshi; Sarvnaz Karimi
>
> **备注:** Short Paper; Under Review at The WebConf 2026 (single-blind submission)
>
> **摘要:** Climate adaptation strategies are proposed in response to climate change. They are practised in agriculture to sustain food production. These strategies can be found in unstructured data (for example, scientific literature from the Elsevier website) or structured (heterogeneous climate data via government APIs). We present Climate Adaptation question-answering with Improved Readability and Noted Sources (CAIRNS), a framework that enables experts -- farmer advisors -- to obtain credible preliminary answers from complex evidence sources from the web. It enhances readability and citation reliability through a structured ScholarGuide prompt and achieves robust evaluation via a consistency-weighted hybrid evaluator that leverages inter-model agreement with experts. Together, these components enable readable, verifiable, and domain-grounded question-answering without fine-tuning or reinforcement learning. Using a previously reported dataset of expert-curated question-answers, we show that CAIRNS outperforms the baselines on most of the metrics. Our thorough ablation study confirms the results on all metrics. To validate our LLM-based evaluation, we also report an analysis of correlations against human judgment.
>
---
#### [new 016] Cross-Lingual Prompt Steerability: Towards Accurate and Robust LLM Behavior across Languages
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文研究多语言大模型系统提示的可引导性，旨在实现跨语言准确与鲁棒推理。针对现有工作局限于英语的局限，提出四维评估框架，通过大规模实验发现特定提示成分（如思维链、情绪、场景）有助于提升多语言表现，并开发优化框架自动发现高效提示，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.02841v1](https://arxiv.org/pdf/2512.02841v1)**

> **作者:** Lechen Zhang; Yusheng Zhou; Tolga Ergen; Lajanugen Logeswaran; Moontae Lee; David Jurgens
>
> **摘要:** System prompts provide a lightweight yet powerful mechanism for conditioning large language models (LLMs) at inference time. While prior work has focused on English-only settings, real-world deployments benefit from having a single prompt to operate reliably across languages. This paper presents a comprehensive study of how different system prompts steer models toward accurate and robust cross-lingual behavior. We propose a unified four-dimensional evaluation framework to assess system prompts in multilingual environments. Through large-scale experiments on five languages, three LLMs, and three benchmarks, we uncover that certain prompt components, such as CoT, emotion, and scenario, correlate with robust multilingual behavior. We develop a prompt optimization framework for multilingual settings and show it can automatically discover prompts that improve all metrics by 5-10%. Finally, we analyze over 10 million reasoning units and find that more performant system prompts induce more structured and consistent reasoning patterns, while reducing unnecessary language-switching. Together, we highlight system prompt optimization as a scalable path to accurate and robust multilingual LLM behavior.
>
---
#### [new 017] Making Dialogue Grounding Data Rich: A Three-Tier Data Synthesis Framework for Generalized Referring Expression Comprehension
- **分类: cs.CL**

- **简介: 该论文针对对话式泛化指代消解（GREC）任务，解决训练与测试域分布偏移及标注数据稀缺问题。提出三层数据合成框架，生成兼具真实性和可控性的对话接地数据，通过微调显著提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.02791v1](https://arxiv.org/pdf/2512.02791v1)**

> **作者:** Juexi Shao; Siyou Li; Yujian Gan; Chris Madge; Vanja Karan; Massimo Poesio
>
> **摘要:** Dialogue-Based Generalized Referring Expressions Comprehension (GREC) requires models to ground the expression and unlimited targets in complex visual scenes while resolving coreference across a long dialogue context. However, existing systems struggle under distribution shift between training and evaluation domains, a gap exacerbated by the scarcity of annotated dialogue grounding data. We address this challenge with a three-tier data-synthesis method that balances realism and controllability to produce scalable supervision for dialogue-conditioned grounding. Fine-tuning on the synthesized data yields consistent, substantial improvements over prior approaches across standard evaluation metrics.
>
---
#### [new 018] SurveyEval: Towards Comprehensive Evaluation of LLM-Generated Academic Surveys
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型生成学术调查问卷的评估难题，提出SurveyEval基准。它从整体质量、结构连贯性、参考准确性三方面进行综合评估，并引入人类参考增强评价对齐。研究发现专用生成系统表现优于通用写作系统，为自动调查系统提供了可扩展的评估框架。**

- **链接: [https://arxiv.org/pdf/2512.02763v1](https://arxiv.org/pdf/2512.02763v1)**

> **作者:** Jiahao Zhao; Shuaixing Zhang; Nan Xu; Lei Wang
>
> **摘要:** LLM-based automatic survey systems are transforming how users acquire information from the web by integrating retrieval, organization, and content synthesis into end-to-end generation pipelines. While recent works focus on developing new generation pipelines, how to evaluate such complex systems remains a significant challenge. To this end, we introduce SurveyEval, a comprehensive benchmark that evaluates automatically generated surveys across three dimensions: overall quality, outline coherence, and reference accuracy. We extend the evaluation across 7 subjects and augment the LLM-as-a-Judge framework with human references to strengthen evaluation-human alignment. Evaluation results show that while general long-text or paper-writing systems tend to produce lower-quality surveys, specialized survey-generation systems are able to deliver substantially higher-quality results. We envision SurveyEval as a scalable testbed to understand and improve automatic survey systems across diverse subjects and evaluation criteria.
>
---
#### [new 019] What Signals Really Matter for Misinformation Tasks? Evaluating Fake-News Detection and Virality Prediction under Real-World Constraints
- **分类: cs.CL**

- **简介: 该论文研究在线虚假信息的检测与传播预测任务，针对实时响应需求，比较文本嵌入与轻量级数值特征的效果。结果表明，文本内容对虚假新闻检测有效，数值特征在资源受限时仍可行；病毒式传播预测更难，且受标签定义影响大，需谨慎处理时间截断问题。**

- **链接: [https://arxiv.org/pdf/2512.02552v1](https://arxiv.org/pdf/2512.02552v1)**

> **作者:** Francesco Paolo Savatteri; Chahan Vidal-Gorène; Florian Cafiero
>
> **摘要:** We present an evaluation-driven study of two practical tasks regarding online misinformation: (i) fake-news detection and (ii) virality prediction in the context of operational settings, with the necessity for rapid reaction. Using the EVONS and FakeNewsNet datasets, we compare textual embeddings (RoBERTa; with a control using Mistral) against lightweight numeric features (timing, follower counts, verification, likes) and sequence models (GRU, gating architectures, Transformer encoders). We show that textual content alone is a strong discriminator for fake-news detection, while numeric-only pipelines remain viable when language models are unavailable or compute is constrained. Virality prediction is markedly harder than fake-news detection and is highly sensitive to label construction; in our setup, a median-based ''viral'' split (<50 likes) is pragmatic but underestimates real-world virality, and time-censoring for engagement features is desirable yet difficult under current API limits. Dimensionality-reduction analyses suggest non-linear structure is more informative for virality than for fake-news detection (t-SNE > PCA on numeric features). Swapping RoBERTa for Mistral embeddings yields only modest deltas, leaving conclusions unchanged. We discuss implications for evaluation design and report reproducibility constraints that realistically affect the field. We release splits and code where possible and provide guidance for metric selection.
>
---
#### [new 020] CREST: Universal Safety Guardrails Through Cluster-Guided Cross-Lingual Transfer
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出CREST，一种基于聚类引导的跨语言迁移安全分类模型，解决大语言模型在低资源语言中缺乏通用安全防护的问题。通过仅在13种高资源语言上训练，实现对100种语言的高效安全检测，仅用0.5B参数即超越多数更大模型，推动构建全球可扩展的语言无关安全系统。**

- **链接: [https://arxiv.org/pdf/2512.02711v1](https://arxiv.org/pdf/2512.02711v1)**

> **作者:** Lavish Bansal; Naman Mishra
>
> **备注:** 8 Pages, 5 Figures, Under Review
>
> **摘要:** Ensuring content safety in large language models (LLMs) is essential for their deployment in real-world applications. However, existing safety guardrails are predominantly tailored for high-resource languages, leaving a significant portion of the world's population underrepresented who communicate in low-resource languages. To address this, we introduce CREST (CRoss-lingual Efficient Safety Transfer), a parameter-efficient multilingual safety classification model that supports 100 languages with only 0.5B parameters. By training on a strategically chosen subset of only 13 high-resource languages, our model utilizes cluster-based cross-lingual transfer from a few to 100 languages, enabling effective generalization to both unseen high-resource and low-resource languages. This approach addresses the challenge of limited training data in low-resource settings. We conduct comprehensive evaluations across six safety benchmarks to demonstrate that CREST outperforms existing state-of-the-art guardrails of comparable scale and achieves competitive results against models with significantly larger parameter counts (2.5B parameters and above). Our findings highlight the limitations of language-specific guardrails and underscore the importance of developing universal, language-agnostic safety systems that can scale effectively to serve global populations.
>
---
#### [new 021] Spoken Conversational Agents with Large Language Models
- **分类: cs.CL; cs.MA; cs.NE; cs.SD; eess.AS**

- **简介: 该论文研究语音对话系统，聚焦将文本大模型适配至语音场景。解决从分步式到端到端系统的演进难题，开展跨模态对齐与联合训练，对比不同架构设计，提出可复现基准，推动语音助手在隐私、安全与评估方面的进展。**

- **链接: [https://arxiv.org/pdf/2512.02593v1](https://arxiv.org/pdf/2512.02593v1)**

> **作者:** Chao-Han Huck Yang; Andreas Stolcke; Larry Heck
>
> **备注:** Accepted to EMNLP 2025 Tutorial
>
> **摘要:** Spoken conversational agents are converging toward voice-native LLMs. This tutorial distills the path from cascaded ASR/NLU to end-to-end, retrieval-and vision-grounded systems. We frame adaptation of text LLMs to audio, cross-modal alignment, and joint speech-text training; review datasets, metrics, and robustness across accents and compare design choices (cascaded vs. E2E, post-ASR correction, streaming). We link industrial assistants to current open-domain and task-oriented agents, highlight reproducible baselines, and outline open problems in privacy, safety, and evaluation. Attendees leave with practical recipes and a clear systems-level roadmap.
>
---
#### [new 022] The Moral Consistency Pipeline: Continuous Ethical Evaluation for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MoCoP框架，解决大模型道德一致性动态评估难题。通过自闭环机制实现无数据集、持续性的伦理行为监测，揭示伦理与毒性负相关、与响应延迟无关的稳定特征，推动可复现的持续审计。**

- **链接: [https://arxiv.org/pdf/2512.03026v1](https://arxiv.org/pdf/2512.03026v1)**

> **作者:** Saeid Jamshidi; Kawser Wazed Nafi; Arghavan Moradi Dakhel; Negar Shahabi; Foutse Khomh
>
> **摘要:** The rapid advancement and adaptability of Large Language Models (LLMs) highlight the need for moral consistency, the capacity to maintain ethically coherent reasoning across varied contexts. Existing alignment frameworks, structured approaches designed to align model behavior with human ethical and social norms, often rely on static datasets and post-hoc evaluations, offering limited insight into how ethical reasoning may evolve across different contexts or temporal scales. This study presents the Moral Consistency Pipeline (MoCoP), a dataset-free, closed-loop framework for continuously evaluating and interpreting the moral stability of LLMs. MoCoP combines three supporting layers: (i) lexical integrity analysis, (ii) semantic risk estimation, and (iii) reasoning-based judgment modeling within a self-sustaining architecture that autonomously generates, evaluates, and refines ethical scenarios without external supervision. Our empirical results on GPT-4-Turbo and DeepSeek suggest that MoCoP effectively captures longitudinal ethical behavior, revealing a strong inverse relationship between ethical and toxicity dimensions (correlation rET = -0.81, p value less than 0.001) and a near-zero association with response latency (correlation rEL approximately equal to 0). These findings demonstrate that moral coherence and linguistic safety tend to emerge as stable and interpretable characteristics of model behavior rather than short-term fluctuations. Furthermore, by reframing ethical evaluation as a dynamic, model-agnostic form of moral introspection, MoCoP offers a reproducible foundation for scalable, continuous auditing and advances the study of computational morality in autonomous AI systems.
>
---
#### [new 023] PEFT-Factory: Unified Parameter-Efficient Fine-Tuning of Autoregressive Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型参数高效微调（PEFT）方法难以复现与比较的问题，提出PEFT-Factory统一框架。它集成19种PEFT方法、27个数据集及多类评估指标，提供可复现、可扩展的微调环境，提升PEFT方法的可比性与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.02764v1](https://arxiv.org/pdf/2512.02764v1)**

> **作者:** Robert Belanec; Ivan Srba; Maria Bielikova
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) methods address the increasing size of Large Language Models (LLMs). Currently, many newly introduced PEFT methods are challenging to replicate, deploy, or compare with one another. To address this, we introduce PEFT-Factory, a unified framework for efficient fine-tuning LLMs using both off-the-shelf and custom PEFT methods. While its modular design supports extensibility, it natively provides a representative set of 19 PEFT methods, 27 classification and text generation datasets addressing 12 tasks, and both standard and PEFT-specific evaluation metrics. As a result, PEFT-Factory provides a ready-to-use, controlled, and stable environment, improving replicability and benchmarking of PEFT methods. PEFT-Factory is a downstream framework that originates from the popular LLaMA-Factory, and is publicly available at https://github.com/kinit-sk/PEFT-Factory
>
---
#### [new 024] ADORE: Autonomous Domain-Oriented Relevance Engine for E-commerce
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对电商搜索中相关性建模的语义鸿沟与数据稀缺问题，提出ADORE框架。通过规则感知判别、错误类型感知数据生成和关键属性增强的知识蒸馏，实现自动标注、对抗样本生成与模型优化，提升模型推理能力与工业适用性。**

- **链接: [https://arxiv.org/pdf/2512.02555v1](https://arxiv.org/pdf/2512.02555v1)**

> **作者:** Zheng Fang; Donghao Xie; Ming Pang; Chunyuan Yuan; Xue Jiang; Changping Peng; Zhangang Lin; Zheng Luo
>
> **备注:** Accepted by SIGIR 2025
>
> **摘要:** Relevance modeling in e-commerce search remains challenged by semantic gaps in term-matching methods (e.g., BM25) and neural models' reliance on the scarcity of domain-specific hard samples. We propose ADORE, a self-sustaining framework that synergizes three innovations: (1) A Rule-aware Relevance Discrimination module, where a Chain-of-Thought LLM generates intent-aligned training data, refined via Kahneman-Tversky Optimization (KTO) to align with user behavior; (2) An Error-type-aware Data Synthesis module that auto-generates adversarial examples to harden robustness; and (3) A Key-attribute-enhanced Knowledge Distillation module that injects domain-specific attribute hierarchies into a deployable student model. ADORE automates annotation, adversarial generation, and distillation, overcoming data scarcity while enhancing reasoning. Large-scale experiments and online A/B testing verify the effectiveness of ADORE. The framework establishes a new paradigm for resource-efficient, cognitively aligned relevance modeling in industrial applications.
>
---
#### [new 025] Think in Parallel, Answer as One: Logit Averaging for Open-Ended Reasoning
- **分类: cs.CL**

- **简介: 该论文针对开放性推理任务（如代码生成、深度研究）中无法使用多数投票的问题，提出无需训练的ThinkMerge方法。通过并行运行K条推理路径，在同步点平均下一词概率，生成连贯输出。实验显示其在代码和网页研究任务上显著提升性能，实现了高效且通用的并行推理。**

- **链接: [https://arxiv.org/pdf/2512.02874v1](https://arxiv.org/pdf/2512.02874v1)**

> **作者:** Haonan Wang; Chao Du; Kenji Kawaguchi; Tianyu Pang
>
> **摘要:** Majority voting has proven effective for close-ended question answering by aggregating parallel reasoning traces. However, it is not directly applicable to open-ended reasoning, such as code generation and web-based deep research, where a "majority" over complete solutions is ill-defined. We introduce ThinkMerge, a training-free, plug-and-play decoding strategy that runs K parallel reasoning traces and averages their next-token logits at synchronization points to produce a single coherent output. ThinkMerge integrates seamlessly with vLLM/SGLang and remains compatible with standard decoding techniques such as Top-p/Top-k. Empirically, it matches or surpasses majority voting on AIME and GPQA, while delivering consistent gains on open-ended coding tasks: on LiveCodeBench (hard), pass@1 improves by +8.28% for DeepCoder-14B-Preview and +7.58% for Qwen3-8B. Beyond code, we further show that ThinkMerge improves web-based deep-research agents (e.g., WebSailor-7B/32B) across GAIA, BrowseComp-en/zh, and XbenchDeepSearch. These results demonstrate that parallel test-time scaling can benefit open-ended reasoning without relying on voting over complete outputs.
>
---
#### [new 026] Dialect Identification Using Resource-Efficient Fine-Tuning Approaches
- **分类: cs.CL; cs.SD**

- **简介: 该论文研究方言识别任务，针对语音模型微调计算成本高、内存占用大的问题，探索适用于预训练语音模型的内存高效微调（MEFT）方法。以Whisper模型在KeSpeech数据集上识别六种普通话子方言为例，实验表明MEFT可降低73.25%显存使用，提速2.1倍，同时保持与传统方法相当的准确率。**

- **链接: [https://arxiv.org/pdf/2512.02074v1](https://arxiv.org/pdf/2512.02074v1)**

> **作者:** Zirui Lin; Haris Gulzar; Monnika Roslianna Busto; Akiko Masaki; Takeharu Eda; Kazuhiro Nakadai
>
> **备注:** Published in APSIPA ASC 2025
>
> **摘要:** Dialect Identification (DI) is a task to recognize different dialects within the same language from a speech signal. DI can help to improve the downstream speech related tasks even when speakers have a strong dialect. However, fine-tuning a speech model for tasks like DI is expensive in terms of computation cost and memory requirement. Recent studies have explored fine-tuning pre-trained speech models for tasks like DI using Parameter-Efficient Fine-Tuning (PEFT) methods, which offer parameter efficiency but limited improvement in memory efficiency and training speed. To address these challenges, we explore Memory-Efficient Fine-Tuning (MEFT) methods, originally proposed for language processing, and apply them to the general-purpose pre-trained speech model. We then comprehensively analyze the GPU memory usage and fine-tuning speed based on various MEFT methods. As a case study, we fine-tune the Whisper model to identify six Mandarin subdialects from the KeSpeech dataset, reducing GPU memory usage by up to 73.25% and accelerating training speed by a factor of 2.1, while maintaining accuracy comparable to vanilla fine-tuning and PEFT methods.
>
---
#### [new 027] Input Order Shapes LLM Semantic Alignment in Multi-Document Summarization
- **分类: cs.CL**

- **简介: 该论文研究多文档摘要中输入顺序对大语言模型语义对齐的影响。针对摘要任务，探究输入顺序是否导致模型偏好首篇文档。通过构造文章三元组并改变输入顺序，实验发现模型生成摘要更契合首篇内容，存在显著首因效应，揭示了模型在信息整合中的偏倚风险。**

- **链接: [https://arxiv.org/pdf/2512.02665v1](https://arxiv.org/pdf/2512.02665v1)**

> **作者:** Jing Ma
>
> **备注:** 9 pages, 3 figures, 2 tables
>
> **摘要:** Large language models (LLMs) are now used in settings such as Google's AI Overviews, where it summarizes multiple long documents. However, it remains unclear whether they weight all inputs equally. Focusing on abortion-related news, we construct 40 pro-neutral-con article triplets, permute each triplet into six input orders, and prompt Gemini 2.5 Flash to generate a neutral overview. We evaluate each summary against its source articles using ROUGE-L (lexical overlap), BERTScore (semantic similarity), and SummaC (factual consistency). One-way ANOVA reveals a significant primacy effect for BERTScore across all stances, indicating that summaries are more semantically aligned with the first-seen article. Pairwise comparisons further show that Position 1 differs significantly from Positions 2 and 3, while the latter two do not differ from each other, confirming a selective preference for the first document. The findings present risks for applications that rely on LLM-generated overviews and for agentic AI systems, where the steps involving LLMs can disproportionately influence downstream actions.
>
---
#### [new 028] Emergent Bayesian Behaviour and Optimal Cue Combination in LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; q-bio.NC**

- **简介: 该论文研究大模型在多模态感知中的隐式贝叶斯行为。通过构建心理物理学基准BayesBench，评估九个LLMs在四类感知任务中的不确定性处理与最优线索整合能力，发现模型准确率高但策略不鲁棒，揭示了准确性与贝叶斯一致性间的分离，提出一致性评分以识别隐含策略。**

- **链接: [https://arxiv.org/pdf/2512.02719v1](https://arxiv.org/pdf/2512.02719v1)**

> **作者:** Julian Ma; Jun Wang; Zafeirios Fountas
>
> **摘要:** Large language models (LLMs) excel at explicit reasoning, but their implicit computational strategies remain underexplored. Decades of psychophysics research show that humans intuitively process and integrate noisy signals using near-optimal Bayesian strategies in perceptual tasks. We ask whether LLMs exhibit similar behaviour and perform optimal multimodal integration without explicit training or instruction. Adopting the psychophysics paradigm, we infer computational principles of LLMs from systematic behavioural studies. We introduce a behavioural benchmark - BayesBench: four magnitude estimation tasks (length, location, distance, and duration) over text and image, inspired by classic psychophysics, and evaluate a diverse set of nine LLMs alongside human judgments for calibration. Through controlled ablations of noise, context, and instruction prompts, we measure performance, behaviour and efficiency in multimodal cue-combination. Beyond accuracy and efficiency metrics, we introduce a Bayesian Consistency Score that detects Bayes-consistent behavioural shifts even when accuracy saturates. Our results show that while capable models often adapt in Bayes-consistent ways, accuracy does not guarantee robustness. Notably, GPT-5 Mini achieves perfect text accuracy but fails to integrate visual cues efficiently. This reveals a critical dissociation between capability and strategy, suggesting accuracy-centric benchmarks may over-index on performance while missing brittle uncertainty handling. These findings reveal emergent principled handling of uncertainty and highlight the correlation between accuracy and Bayesian tendencies. We release our psychophysics benchmark and consistency metric (https://bayes-bench.github.io) as evaluation tools and to inform future multimodal architecture designs.
>
---
#### [new 029] HealthContradict: Evaluating Biomedical Knowledge Conflicts in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对医学语言模型在矛盾上下文中的推理能力，提出HealthContradict数据集，评估模型如何利用正确信息并抵抗错误信息。通过对比不同上下文设置下的回答，揭示模型依赖上下文而非仅靠预训练知识的特性，推动医学问答中因果推理的评测发展。**

- **链接: [https://arxiv.org/pdf/2512.02299v1](https://arxiv.org/pdf/2512.02299v1)**

> **作者:** Boya Zhang; Alban Bornet; Rui Yang; Nan Liu; Douglas Teodoro
>
> **摘要:** How do language models use contextual information to answer health questions? How are their responses impacted by conflicting contexts? We assess the ability of language models to reason over long, conflicting biomedical contexts using HealthContradict, an expert-verified dataset comprising 920 unique instances, each consisting of a health-related question, a factual answer supported by scientific evidence, and two documents presenting contradictory stances. We consider several prompt settings, including correct, incorrect or contradictory context, and measure their impact on model outputs. Compared to existing medical question-answering evaluation benchmarks, HealthContradict provides greater distinctions of language models' contextual reasoning capabilities. Our experiments show that the strength of fine-tuned biomedical language models lies not only in their parametric knowledge from pretraining, but also in their ability to exploit correct context while resisting incorrect context.
>
---
#### [new 030] DETAIL Matters: Measuring the Impact of Prompt Specificity on Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究提示词具体性对大模型推理性能的影响，属于自然语言处理中的提示工程任务。针对提示具体性影响未被充分研究的问题，提出DETAIL框架，通过生成多级提示、量化具体性并评估正确性，发现具体性提升准确率，尤其对小模型和程序类任务更显著。**

- **链接: [https://arxiv.org/pdf/2512.02246v1](https://arxiv.org/pdf/2512.02246v1)**

> **作者:** Olivia Kim
>
> **摘要:** Prompt design plays a critical role in the reasoning performance of large language models (LLMs), yet the impact of prompt specificity - how detailed or vague a prompt is - remains understudied. This paper introduces DETAIL, a framework for evaluating LLM performance across varying levels of prompt specificity. We generate multi-level prompts using GPT-4, quantify specificity via perplexity, and assess correctness using GPT-based semantic equivalence. Experiments on 30 novel reasoning tasks across GPT-4 and O3-mini reveal that specificity improves accuracy, especially for smaller models and procedural tasks. Our results highlight the need for adaptive prompting strategies and provide tools and data to support further research.
>
---
#### [new 031] promptolution: A Unified, Modular Framework for Prompt Optimization
- **分类: cs.CL**

- **简介: 该论文提出promptolution框架，解决大语言模型提示优化中工具分散、难维护的问题。通过统一、模块化设计，集成多种离散提示优化方法，支持灵活扩展，提升实践与研究效率。**

- **链接: [https://arxiv.org/pdf/2512.02840v1](https://arxiv.org/pdf/2512.02840v1)**

> **作者:** Tom Zehle; Timo Heiß; Moritz Schlager; Matthias Aßenmacher; Matthias Feurer
>
> **摘要:** Prompt optimization has become crucial for enhancing the performance of large language models (LLMs) across a broad range of tasks. Although many research papers show its effectiveness, practical adoption is hindered as existing implementations are often tied to unmaintained and isolated research codebases. To address this, we introduce promptolution, a unified and modular open-source framework that provides all components required for prompt optimization within a single extensible system for both practitioners and researchers. It integrates multiple contemporary discrete prompt optimizers while remaining agnostic to the underlying LLM implementation.
>
---
#### [new 032] Think Before You Prune: Self-Reflective Structured Pruning for Reasoning Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对推理型大模型（RLM）的高效部署问题，提出自省式结构化剪枝框架RESP。针对现有剪枝方法在RLM上导致性能崩溃的问题，利用模型自生成推理轨迹作为校准信号，结合解码时梯度重要性评估与渐进式重构，实现高稀疏度下保持推理准确性。**

- **链接: [https://arxiv.org/pdf/2512.02185v1](https://arxiv.org/pdf/2512.02185v1)**

> **作者:** Ziyan Wang; Enmao Diao; Qi Le; Pu Wang; Guanchu Wang; Minwoo Lee; Shu-ping Yeh; Li Yang
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Reasoning LLMs (RLMs) such as OpenAI o1, DeepSeek-R1, and Qwen3 deliver strong multi-step reasoning through chain-of-thought generation, but their large model sizes and lengthy decode-time outputs make them costly to deploy and unsuitable for resource-constrained settings. To reduce computing and memory cost, pruning offers a promising solution by removing unimportant parameters. However, despite their success on standard LLMs, existing pruning methods severely damage RLMs, as even moderate sparsity (e.g., 20%) can collapse accuracy and completely disrupt the model's reasoning coherence. We begin by analyzing why existing pruning pipelines fail on reasoning LLMs and find that their brittleness largely stems from a mismatch between the calibration data, the pruning objective, and the model's decode-time reasoning behavior. Our study further shows that the most reliable calibration signal comes not from human-written labels but from the model's own self-generated reasoning traces, which more accurately reflect its inference distribution. Guided by these insights, we introduce RESP, a self-reflective structured pruning framework that aligns pruning decisions with the model's reasoning dynamics through self-generated calibration, decode-only gradient-based importance estimation, and progressive regeneration that maintains calibration fidelity as sparsity increases. Experiments on Qwen3-8B demonstrate that RESP markedly outperforms existing structured pruning methods on both GSM8K and MathQA, preserving near-dense accuracy at 20-30% sparsity and substantially mitigating performance collapse at higher sparsity levels. At 40% sparsity, RESP attains 81.3% accuracy on GSM8K and 59.6% on MathQA, surpassing the strongest baselines by 66.87% and 47%, respectively.
>
---
#### [new 033] When Does Verification Pay Off? A Closer Look at LLMs as Solution Verifiers
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）作为解题验证器的有效性，聚焦于验证机制如何提升解题性能。通过系统实验，比较自验证、同族与跨族验证，提出“验证增益”指标，揭示跨族验证更优，后训练增强跨族效果，数学逻辑任务最易验证。**

- **链接: [https://arxiv.org/pdf/2512.02304v1](https://arxiv.org/pdf/2512.02304v1)**

> **作者:** Jack Lu; Ryan Teehan; Jinran Jin; Mengye Ren
>
> **摘要:** Large language models (LLMs) can act as both problem solvers and solution verifiers, with verifiers improving solver performance by selecting high-quality answers from a pool of candidates. However, prior studies of solver-verifier interactions have been limited, focusing mainly on self-verification and rarely examining how verifiers judge outputs from models in their own or in another model family. Modern LLMs also undergo extensive post-training, but its effect on verification remains unclear. We present a systematic study across 37 models spanning multiple families, sizes, and base vs. post-trained variants, evaluated on 9 benchmarks covering logical reasoning, structured puzzles, symbolic computation, mathematics, commonsense, factual recall, and domain knowledge. We compare self-verification with verification within the same family and across different families. To support this, we introduce and empirically validate verifier gain, a metric that predicts the performance improvements from test-time verifier-based rejection sampling. We analyze how metrics like verifier gain and false positive rate scale with model size and post-training, and characterize differences in dataset verifiability. Our findings show that cross-family verification is especially effective; post-training reduces self-improvement but strengthens cross-family improvement; and mathematical and logical tasks exhibit the highest inherent verifiability.
>
---
#### [new 034] Feature Selection Empowered BERT for Detection of Hate Speech with Vocabulary Augmentation
- **分类: cs.CL; cs.AI; cs.NE**

- **简介: 该论文针对社交媒体仇恨言论检测任务，解决传统方法依赖大量数据且难以应对新式俚语和隐晦表达的问题。提出基于TF-IDF的样本筛选与领域词汇增强策略，减少训练数据量并提升BERT对新型仇恨言论的识别能力，实现高效精准检测。**

- **链接: [https://arxiv.org/pdf/2512.02141v1](https://arxiv.org/pdf/2512.02141v1)**

> **作者:** Pritish N. Desai; Tanay Kewalramani; Srimanta Mandal
>
> **摘要:** Abusive speech on social media poses a persistent and evolving challenge, driven by the continuous emergence of novel slang and obfuscated terms designed to circumvent detection systems. In this work, we present a data efficient strategy for fine tuning BERT on hate speech classification by significantly reducing training set size without compromising performance. Our approach employs a TF IDF-based sample selection mechanism to retain only the most informative 75 percent of examples, thereby minimizing training overhead. To address the limitations of BERT's native vocabulary in capturing evolving hate speech terminology, we augment the tokenizer with domain-specific slang and lexical variants commonly found in abusive contexts. Experimental results on a widely used hate speech dataset demonstrate that our method achieves competitive performance while improving computational efficiency, highlighting its potential for scalable and adaptive abusive content moderation.
>
---
#### [new 035] Human-Level and Beyond: Benchmarking Large Language Models Against Clinical Pharmacists in Prescription Review
- **分类: cs.CL; cs.CY**

- **简介: 该论文针对临床处方审核任务，构建了RxBench基准，涵盖14类常见用药错误。通过评估18个大模型，发现部分模型已超越人类药师水平，并基于此对模型进行微调，提升了其在短答案任务中的表现，推动了可信临床AI工具的发展。**

- **链接: [https://arxiv.org/pdf/2512.02024v1](https://arxiv.org/pdf/2512.02024v1)**

> **作者:** Yan Yang; Mouxiao Bian; Peiling Li; Bingjian Wen; Ruiyao Chen; Kangkun Mao; Xiaojun Ye; Tianbin Li; Pengcheng Chen; Bing Han; Jie Xu; Kaifeng Qiu; Junyan Wu
>
> **摘要:** The rapid advancement of large language models (LLMs) has accelerated their integration into clinical decision support, particularly in prescription review. To enable systematic and fine-grained evaluation, we developed RxBench, a comprehensive benchmark that covers common prescription review categories and consolidates 14 frequent types of prescription errors drawn from authoritative pharmacy references. RxBench consists of 1,150 single-choice, 230 multiple-choice, and 879 short-answer items, all reviewed by experienced clinical pharmacists. We benchmarked 18 state-of-the-art LLMs and identified clear stratification of performance across tasks. Notably, Gemini-2.5-pro-preview-05-06, Grok-4-0709, and DeepSeek-R1-0528 consistently formed the first tier, outperforming other models in both accuracy and robustness. Comparisons with licensed pharmacists indicated that leading LLMs can match or exceed human performance in certain tasks. Furthermore, building on insights from our benchmark evaluation, we performed targeted fine-tuning on a mid-tier model, resulting in a specialized model that rivals leading general-purpose LLMs in performance on short-answer question tasks. The main contribution of RxBench lies in establishing a standardized, error-type-oriented framework that not only reveals the capabilities and limitations of frontier LLMs in prescription review but also provides a foundational resource for building more reliable and specialized clinical tools.
>
---
#### [new 036] Mirror, Mirror on the Wall -- Which is the Best Model of Them All?
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）选型难题，聚焦量化评估维度，分析现有排行榜与基准测试，以医疗领域为案例研究模型性能演变。提出模型选择方法论（MSM），系统指导根据具体任务选择最优模型。**

- **链接: [https://arxiv.org/pdf/2512.02043v1](https://arxiv.org/pdf/2512.02043v1)**

> **作者:** Dina Sayed; Heiko Schuldt
>
> **摘要:** Large Language Models (LLMs) have become one of the most transformative tools across many applications, as they have significantly boosted productivity and achieved impressive results in various domains such as finance, healthcare, education, telecommunications, and law, among others. Typically, state-of-the-art (SOTA) foundation models are developed by large corporations based on large data collections and substantial computational and financial resources required to pretrain such models from scratch. These foundation models then serve as the basis for further development and domain adaptation for specific use cases or tasks. However, given the dynamic and fast-paced nature of launching new foundation models, the process of selecting the most suitable model for a particular use case, application, or domain becomes increasingly complex. We argue that there are two main dimensions that need to be taken into consideration when selecting a model for further training: a qualitative dimension (which model is best suited for a task based on information, for instance, taken from model cards) and a quantitative dimension (which is the best performing model). The quantitative performance of models is assessed through leaderboards, which rank models based on standardized benchmarks and provide a consistent framework for comparing different LLMs. In this work, we address the analysis of the quantitative dimension by exploring the current leaderboards and benchmarks. To illustrate this analysis, we focus on the medical domain as a case study, demonstrating the evolution, current landscape, and practical significance of this quantitative evaluation dimension. Finally, we propose a Model Selection Methodology (MSM), a systematic approach designed to guide the navigation, prioritization, and selection of the model that best aligns with a given use case.
>
---
#### [new 037] BOOM: Beyond Only One Modality KIT's Multimodal Multilingual Lecture Companion
- **分类: cs.CL**

- **简介: 该论文提出BOOM系统，解决多模态多语言讲座内容本地化问题。针对讲座中音频、文本与幻灯片需同步翻译的挑战，实现三模态联合翻译：生成译文、保留视觉元素的本地化幻灯片及合成语音。实验表明其提升摘要与问答性能，代码已开源。**

- **链接: [https://arxiv.org/pdf/2512.02817v1](https://arxiv.org/pdf/2512.02817v1)**

> **作者:** Sai Koneru; Fabian Retkowski; Christian Huber; Lukas Hilgert; Seymanur Akti; Enes Yavuz Ugan; Alexander Waibel; Jan Niehues
>
> **备注:** Under review
>
> **摘要:** The globalization of education and rapid growth of online learning have made localizing educational content a critical challenge. Lecture materials are inherently multimodal, combining spoken audio with visual slides, which requires systems capable of processing multiple input modalities. To provide an accessible and complete learning experience, translations must preserve all modalities: text for reading, slides for visual understanding, and speech for auditory learning. We present \textbf{BOOM}, a multimodal multilingual lecture companion that jointly translates lecture audio and slides to produce synchronized outputs across three modalities: translated text, localized slides with preserved visual elements, and synthesized speech. This end-to-end approach enables students to access lectures in their native language while aiming to preserve the original content in its entirety. Our experiments demonstrate that slide-aware transcripts also yield cascading benefits for downstream tasks such as summarization and question answering. We release our Slide Translation code at https://github.com/saikoneru/image-translator and integrate it in Lecture Translator at https://gitlab.kit.edu/kit/isl-ai4lt/lt-middleware/ltpipeline}\footnote{All released code and models are licensed under the MIT License.
>
---
#### [new 038] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks
- **分类: cs.CL**

- **简介: 该论文针对大语言模型后训练中优势信号混杂导致指导模糊的问题，提出CAPO机制。通过先用正向优势样本进行模仿学习建立基础，再逐步引入负向信号提升判别能力，增强跨领域推理性能。适用于数学与多模态GUI推理任务，兼容多种优化方法，显著提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.02580v1](https://arxiv.org/pdf/2512.02580v1)**

> **作者:** Changpeng Yang; Jinyang Wu; Yuchen Liu; Shuai Zhang; Yang Li; Qiliang Liang; Hongzhen Wang; Shuai Nie; Jiaming Xu; Runyu Shi; Ying Huang; Guoquan Zhang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Reinforcement learning has emerged as a paradigm for post-training large language models, boosting their reasoning capabilities. Such approaches compute an advantage value for each sample, reflecting better or worse performance than expected, thereby yielding both positive and negative signals for training. However, the indiscriminate mixing of the two signals in existing methods, especially from the early stages, may lead to ambiguous guidance and limited gains. To address this issue, we propose **CAPO** (**C**urriculum **A**dvantage **P**olicy **O**ptimization), an adaptive curriculum mechanism based on advantage signals. The proposed mechanism bootstraps imitation learning with positive-only advantage samples to establish robust foundations, and subsequently introduces negative signals to cultivate discriminative capabilities, thereby improving generalization across complex scenarios. Compatible with diverse optimization methods including GRPO, PPO, RLOO, and Reinforce++, our method consistently achieves stable and significant improvements in mathematical reasoning tasks, and further generalizes effectively to multimodal Graphical User Interface (GUI) reasoning scenarios, establishing itself as a versatile and robust optimization framework.
>
---
#### [new 039] Deep Research: A Systematic Survey
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文聚焦深度研究（Deep Research）任务，旨在解决大语言模型在复杂开放问题上缺乏多源信息整合与可验证输出的问题。论文提出三阶段路线图，系统梳理查询规划、信息获取、记忆管理与答案生成四大核心组件，总结优化方法与评估挑战，为构建高效研究型智能体提供框架指导。**

- **链接: [https://arxiv.org/pdf/2512.02038v1](https://arxiv.org/pdf/2512.02038v1)**

> **作者:** Zhengliang Shi; Yiqun Chen; Haitao Li; Weiwei Sun; Shiyu Ni; Yougang Lyu; Run-Ze Fan; Bowen Jin; Yixuan Weng; Minjun Zhu; Qiujie Xie; Xinyu Guo; Qu Yang; Jiayi Wu; Jujia Zhao; Xiaqiang Tang; Xinbei Ma; Cunxiang Wang; Jiaxin Mao; Qingyao Ai; Jen-Tse Huang; Wenxuan Wang; Yue Zhang; Yiming Yang; Zhaopeng Tu; Zhaochun Ren
>
> **摘要:** Large language models (LLMs) have rapidly evolved from text generators into powerful problem solvers. Yet, many open tasks demand critical thinking, multi-source, and verifiable outputs, which are beyond single-shot prompting or standard retrieval-augmented generation. Recently, numerous studies have explored Deep Research (DR), which aims to combine the reasoning capabilities of LLMs with external tools, such as search engines, thereby empowering LLMs to act as research agents capable of completing complex, open-ended tasks. This survey presents a comprehensive and systematic overview of deep research systems, including a clear roadmap, foundational components, practical implementation techniques, important challenges, and future directions. Specifically, our main contributions are as follows: (i) we formalize a three-stage roadmap and distinguish deep research from related paradigms; (ii) we introduce four key components: query planning, information acquisition, memory management, and answer generation, each paired with fine-grained sub-taxonomies; (iii) we summarize optimization techniques, including prompting, supervised fine-tuning, and agentic reinforcement learning; and (iv) we consolidate evaluation criteria and open challenges, aiming to guide and facilitate future development. As the field of deep research continues to evolve rapidly, we are committed to continuously updating this survey to reflect the latest progress in this area.
>
---
#### [new 040] A Knowledge-Based Language Model: Deducing Grammatical Knowledge in a Multi-Agent Language Acquisition Simulation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言习得中的语法知识获取，提出基于多智能体的MODOMA系统。通过成人与儿童智能体的交互，实现无监督语言学习，成功使儿童智能体习得离散的语法类别。实验验证了系统在机器生成数据上可复现人类语言规律，证明了该方法的有效性。**

- **链接: [https://arxiv.org/pdf/2512.02195v1](https://arxiv.org/pdf/2512.02195v1)**

> **作者:** David Ph. Shakouri; Crit Cremers; Niels O. Schiller
>
> **备注:** 23 pages, 7 figures, 11 tables. Related work: arXiv:2503.18702. This is the peer-reviewed publisher's version, downloadable from: https://www.clinjournal.org/clinj/article/view/193
>
> **摘要:** This paper presents an initial study performed by the MODOMA system. The MODOMA is a computational multi-agent laboratory environment for unsupervised language acquisition experiments such that acquisition is based on the interaction between two language models, an adult and a child agent. Although this framework employs statistical as well as rule-based procedures, the result of language acquisition is a knowledge-based language model, which can be used to generate and parse new utterances of the target language. This system is fully parametrized and researchers can control all aspects of the experiments while the results of language acquisition, that is, the acquired grammatical knowledge, are explicitly represented and can be consulted. Thus, this system introduces novel possibilities for conducting computational language acquisition experiments. The experiments presented by this paper demonstrate that functional and content categories can be acquired and represented by the daughter agent based on training and test data containing different amounts of exemplars generated by the adult agent. Interestingly, similar patterns, which are well-established for human-generated data, are also found for these machine-generated data. As the procedures resulted in the successful acquisition of discrete grammatical categories by the child agent, these experiments substantiate the validity of the MODOMA approach to modelling language acquisition.
>
---
#### [new 041] AutoNeural: Co-Designing Vision-Language Models for NPU Inference
- **分类: cs.CL**

- **简介: 该论文针对边缘设备上视觉-语言模型（VLM）在NPU推理效率低的问题，提出AutoNeural架构。通过替换ViT为轻量卷积主干并结合状态空间模型，实现整数量化与线性计算，显著降低延迟与内存开销，提升推理速度与上下文长度，验证了硬件感知的模型协同设计对边缘多模态智能的关键作用。**

- **链接: [https://arxiv.org/pdf/2512.02924v1](https://arxiv.org/pdf/2512.02924v1)**

> **作者:** Wei Chen; Liangmin Wu; Yunhai Hu; Zhiyuan Li; Zhiyuan Cheng; Yicheng Qian; Lingyue Zhu; Zhipeng Hu; Luoyi Liang; Qiang Tang; Zhen Liu; Han Yang
>
> **摘要:** While Neural Processing Units (NPUs) offer high theoretical efficiency for edge AI, state-of-the-art Vision--Language Models (VLMs) tailored for GPUs often falter on these substrates. We attribute this hardware-model mismatch to two primary factors: the quantization brittleness of Vision Transformers (ViTs) and the I/O-bound nature of autoregressive attention mechanisms, which fail to utilize the high arithmetic throughput of NPUs. To bridge this gap, we propose AutoNeural, an NPU-native VLM architecture co-designed for integer-only inference. We replace the standard ViT encoder with a MobileNetV5-style backbone utilizing depthwise separable convolutions, which ensures bounded activation distributions for stable INT4/8/16 quantization. Complementing this, our language backbone integrates State-Space Model (SSM) principles with Transformer layers, employing efficient gated convolutions to achieve linear-time complexity. This hybrid design eliminates the heavy memory I/O overhead of Key-Value caching during generation. Our approach delivers substantial efficiency gains, reducing quantization error of vision encoder by up to 7x and end-to-end latency by 14x compared to conventional baselines. The AutoNeural also delivers 3x decoding speed and 4x longer context window than the baseline. We validate these improvements via a real-world automotive case study on the Qualcomm SA8295P SoC, demonstrating real-time performance for cockpit applications. Our results highlight that rethinking model topology specifically for NPU constraints is a prerequisite for robust multi-modal edge intelligence.
>
---
#### [new 042] SR-GRPO: Stable Rank as an Intrinsic Geometric Reward for Large Language Model Alignment
- **分类: cs.CL**

- **简介: 该论文针对大模型对齐中依赖外部监督的问题，提出基于模型内部表示的稳定秩（Stable Rank）作为无监督质量信号。通过衡量隐藏状态的有效维度，构建无需标注的奖励机制，实现无需外部监督的强化学习对齐。实验表明，该方法显著提升模型在数学和STEM任务上的表现。**

- **链接: [https://arxiv.org/pdf/2512.02807v1](https://arxiv.org/pdf/2512.02807v1)**

> **作者:** Yixuan Tang; Yi Yang
>
> **摘要:** Aligning Large Language Models (LLMs) with human preferences typically relies on external supervision, which faces critical limitations: human annotations are scarce and subjective, reward models are vulnerable to reward hacking, and self-evaluation methods suffer from prompt sensitivity and biases. In this work, we propose stable rank, an intrinsic, annotation-free quality signal derived from model representations. Stable rank measures the effective dimensionality of hidden states by computing the ratio of total variance to dominant-direction variance, capturing quality through how information distributes across representation dimensions. Empirically, stable rank achieves 84.04% accuracy on RewardBench and improves task accuracy by an average of 11.3 percentage points over greedy decoding via Best-of-N sampling. Leveraging this insight, we introduce Stable Rank Group Relative Policy Optimization (SR-GRPO), which uses stable rank as a reward signal for reinforcement learning. Without external supervision, SR-GRPO improves Qwen2.5-1.5B-Instruct by 10% on STEM and 19% on mathematical reasoning, outperforming both learned reward models and self-evaluation baselines. Our findings demonstrate that quality signals can be extracted from internal model geometry, offering a path toward scalable alignment without external supervision.
>
---
#### [new 043] A benchmark dataset for evaluating Syndrome Differentiation and Treatment in large language models
- **分类: cs.CL**

- **简介: 该论文针对中医“辨证论治”（SDT）的评估难题，构建了首个临床病例驱动的基准数据集TCM-BEST4SDT。通过专家标注与多机制评估框架，涵盖知识、伦理、安全及辨证治疗四类任务，有效衡量LLMs在中医诊疗中的综合能力，并公开数据以推动智能中医发展。**

- **链接: [https://arxiv.org/pdf/2512.02816v1](https://arxiv.org/pdf/2512.02816v1)**

> **作者:** Kunning Li; Jianbin Guo; Zhaoyang Shang; Yiqing Liu; Hongmin Du; Lingling Liu; Yuping Zhao; Lifeng Dong
>
> **摘要:** The emergence of Large Language Models (LLMs) within the Traditional Chinese Medicine (TCM) domain presents an urgent need to assess their clinical application capabilities. However, such evaluations are challenged by the individualized, holistic, and diverse nature of TCM's "Syndrome Differentiation and Treatment" (SDT). Existing benchmarks are confined to knowledge-based question-answering or the accuracy of syndrome differentiation, often neglecting assessment of treatment decision-making. Here, we propose a comprehensive, clinical case-based benchmark spearheaded by TCM experts, and a specialized reward model employed to quantify prescription-syndrome congruence. Data annotation follows a rigorous pipeline. This benchmark, designated TCM-BEST4SDT, encompasses four tasks, including TCM Basic Knowledge, Medical Ethics, LLM Content Safety, and SDT. The evaluation framework integrates three mechanisms, namely selected-response evaluation, judge model evaluation, and reward model evaluation. The effectiveness of TCM-BEST4SDT was corroborated through experiments on 15 mainstream LLMs, spanning both general and TCM domains. To foster the development of intelligent TCM research, TCM-BEST4SDT is now publicly available.
>
---
#### [new 044] ReVSeg: Incentivizing the Reasoning Chain for Video Segmentation with Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对视频对象分割中推理链不透明的问题，提出ReVSeg方法。通过强化学习驱动预训练视觉语言模型，分步执行语义理解、时序证据选择与空间定位，显式建模复杂动态推理，提升分割性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.02835v1](https://arxiv.org/pdf/2512.02835v1)**

> **作者:** Yifan Li; Yingda Yin; Lingting Zhu; Weikai Chen; Shengju Qian; Xin Wang; Yanwei Fu
>
> **摘要:** Reasoning-centric video object segmentation is an inherently complex task: the query often refers to dynamics, causality, and temporal interactions, rather than static appearances. Yet existing solutions generally collapse these factors into simplified reasoning with latent embeddings, rendering the reasoning chain opaque and essentially intractable. We therefore adopt an explicit decomposition perspective and introduce ReVSeg, which executes reasoning as sequential decisions in the native interface of pretrained vision language models (VLMs). Rather than folding all reasoning into a single-step prediction, ReVSeg executes three explicit operations -- semantics interpretation, temporal evidence selection, and spatial grounding -- aligning pretrained capabilities. We further employ reinforcement learning to optimize the multi-step reasoning chain, enabling the model to self-refine its decision quality from outcome-driven signals. Experimental results demonstrate that ReVSeg attains state-of-the-art performances on standard video object segmentation benchmarks and yields interpretable reasoning trajectories. Project page is available at https://clementine24.github.io/ReVSeg/ .
>
---
#### [new 045] LeechHijack: Covert Computational Resource Exploitation in Intelligent Agent Systems
- **分类: cs.CR; cs.CL**

- **简介: 该论文针对LLM代理系统中因信任第三方工具带来的安全风险，提出“隐性毒性”攻击模型LeechHijack。通过嵌入隐蔽后门，利用MCP协议在合法权限内窃取计算资源，实现资源劫持。研究验证了其在四大LLM框架上的有效性，揭示了对计算溯源与资源认证机制的迫切需求。**

- **链接: [https://arxiv.org/pdf/2512.02321v1](https://arxiv.org/pdf/2512.02321v1)**

> **作者:** Yuanhe Zhang; Weiliu Wang; Zhenhong Zhou; Kun Wang; Jie Zhang; Li Sun; Yang Liu; Sen Su
>
> **摘要:** Large Language Model (LLM)-based agents have demonstrated remarkable capabilities in reasoning, planning, and tool usage. The recently proposed Model Context Protocol (MCP) has emerged as a unifying framework for integrating external tools into agent systems, enabling a thriving open ecosystem of community-built functionalities. However, the openness and composability that make MCP appealing also introduce a critical yet overlooked security assumption -- implicit trust in third-party tool providers. In this work, we identify and formalize a new class of attacks that exploit this trust boundary without violating explicit permissions. We term this new attack vector implicit toxicity, where malicious behaviors occur entirely within the allowed privilege scope. We propose LeechHijack, a Latent Embedded Exploit for Computation Hijacking, in which an adversarial MCP tool covertly expropriates the agent's computational resources for unauthorized workloads. LeechHijack operates through a two-stage mechanism: an implantation stage that embeds a benign-looking backdoor in a tool, and an exploitation stage where the backdoor activates upon predefined triggers to establish a command-and-control channel. Through this channel, the attacker injects additional tasks that the agent executes as if they were part of its normal workflow, effectively parasitizing the user's compute budget. We implement LeechHijack across four major LLM families. Experiments show that LeechHijack achieves an average success rate of 77.25%, with a resource overhead of 18.62% compared to the baseline. This study highlights the urgent need for computational provenance and resource attestation mechanisms to safeguard the emerging MCP ecosystem.
>
---
#### [new 046] OmniGuard: Unified Omni-Modal Guardrails with Deliberate Reasoning
- **分类: cs.AI; cs.CL; cs.CR; cs.CV; cs.LG**

- **简介: 该论文针对多模态大模型的安全防护问题，提出OmniGuard统一框架，解决传统单模态安全机制在跨模态场景下泛化能力差的问题。通过构建超21万样本的多模态安全数据集，结合专家模型提炼安全判别与批判性反馈，实现对文本、图像、视频、音频等全模态的精细化安全管控。**

- **链接: [https://arxiv.org/pdf/2512.02306v1](https://arxiv.org/pdf/2512.02306v1)**

> **作者:** Boyu Zhu; Xiaofei Wen; Wenjie Jacky Mo; Tinghui Zhu; Yanan Xie; Peng Qi; Muhao Chen
>
> **摘要:** Omni-modal Large Language Models (OLLMs) that process text, images, videos, and audio introduce new challenges for safety and value guardrails in human-AI interaction. Prior guardrail research largely targets unimodal settings and typically frames safeguarding as binary classification, which limits robustness across diverse modalities and tasks. To address this gap, we propose OmniGuard, the first family of omni-modal guardrails that performs safeguarding across all modalities with deliberate reasoning ability. To support the training of OMNIGUARD, we curate a large, comprehensive omni-modal safety dataset comprising over 210K diverse samples, with inputs that cover all modalities through both unimodal and cross-modal samples. Each sample is annotated with structured safety labels and carefully curated safety critiques from expert models through targeted distillation. Extensive experiments on 15 benchmarks show that OmniGuard achieves strong effectiveness and generalization across a wide range of multimodal safety scenarios. Importantly, OmniGuard provides a unified framework that enforces policies and mitigates risks in omni-modalities, paving the way toward building more robust and capable omnimodal safeguarding systems.
>
---
#### [new 047] See, Think, Learn: A Self-Taught Multimodal Reasoner
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对视觉语言模型在多模态推理中感知与推理能力不足的问题，提出自训练框架See-Think-Learn（STL）。通过结构化推理模板先提取视觉属性再推理，并引入负向解释增强判别力，实现感知与推理的联合优化，显著提升多模态推理性能。**

- **链接: [https://arxiv.org/pdf/2512.02456v1](https://arxiv.org/pdf/2512.02456v1)**

> **作者:** Sourabh Sharma; Sonam Gupta; Sadbhawna
>
> **备注:** Winter Conference on Applications of Computer Vision 2026
>
> **摘要:** Vision-Language Models (VLMs) have achieved remarkable progress in integrating visual perception with language understanding. However, effective multimodal reasoning requires both accurate perception and robust reasoning, and weakness in either limits the performance of VLMs. Prior efforts to enhance reasoning often depend on high-quality chain-of-thought (CoT) data, obtained via labor-intensive human annotations, costly proprietary models, or self-training methods that overlook perception. To address these limitations, we propose a simple yet effective self-training framework called See-Think-Learn (STL). At its core, STL introduces a structured reasoning template that encourages the model to see before thinking, first extracting visual attributes in textual form, then using them to guide reasoning. The framework jointly improves perception and reasoning by having the model generate and learn from its own structured rationales in a self-training loop. Furthermore, we augment the training data with negative rationales, i.e. explanations that justify why certain answer choices are incorrect, to enhance the model's ability to distinguish between correct and misleading responses. This fosters more discriminative and robust learning. Experiments across diverse domains show that STL consistently outperforms baselines trained directly only on answers or self-generated reasoning, while qualitative analysis confirms the high quality of its rationales. STL thus provides a cost-effective solution to enhance multimodal reasoning ability of VLMs.
>
---
#### [new 048] Process-Centric Analysis of Agentic Software Systems
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文研究 agentic 软件系统的流程分析，针对传统评估仅关注结果而忽视过程的问题，提出 Graphectory 框架编码轨迹的时空语义关系。通过分析 4000 条轨迹，揭示了提示复杂度、模型强度与策略演化对过程质量的影响，发现成功系统仍存在效率低下问题。**

- **链接: [https://arxiv.org/pdf/2512.02393v1](https://arxiv.org/pdf/2512.02393v1)**

> **作者:** Shuyang Liu; Yang Chen; Rahul Krishna; Saurabh Sinha; Jatin Ganhotra; Reyhan Jabbarvand
>
> **摘要:** Agentic systems are modern software systems: they consist of orchestrated modules, expose interfaces, and are deployed in software pipelines. Unlike conventional programs, their execution (i.e., trajectories) is inherently stochastic and adaptive to the problem they are solving. Evaluation of such systems is often outcome-centric, judging their performance based on success or failure at the final step. This narrow focus overlooks detailed insights about such systems, failing to explain how agents reason, plan, act, or change their strategies over time. Inspired by the structured representation of conventional software systems as graphs, we introduce Graphectory to systematically encode the temporal and semantic relations in such software systems. Graphectory facilitates the design of process-centric metrics and analyses to assess the quality of agentic workflows independent of final success. Using Graphectory, we analyze 4000 trajectories of two dominant agentic programming workflows, namely SWE-agent and OpenHands, with a combination of four backbone Large Language Models (LLMs), attempting to resolve SWE-bench Verified issues. Our fully automated analyses reveal that: (1) agents using richer prompts or stronger LLMs exhibit more complex Graphectory, reflecting deeper exploration, broader context gathering, and more thorough validation before patch submission; (2) agents' problem-solving strategies vary with both problem difficulty and the underlying LLM -- for resolved issues, the strategies often follow coherent localization-patching-validation steps, while unresolved ones exhibit chaotic, repetitive, or backtracking behaviors; (3) even when successful, agentic programming systems often display inefficient processes, leading to unnecessarily prolonged trajectories.
>
---
#### [new 049] Probabilistic energy profiler for statically typed JVM-based programming languages
- **分类: cs.PL; cs.CL**

- **简介: 该论文针对静态类型JVM语言（如Java、Scala）的能效预测问题，提出基于贝叶斯统计的随机能量分析方法。通过测量字节码模式能耗，构建包含数据大小、类型、操作及设备四因素的统计模型，实现对源代码级能耗的分布预测，提升可解释性与统计推断能力。**

- **链接: [https://arxiv.org/pdf/2512.02738v1](https://arxiv.org/pdf/2512.02738v1)**

> **作者:** Joel Nyholm; Wojciech Mostowski; Christoph Reichenbach
>
> **摘要:** Energy consumption is a growing concern in several fields, from mobile devices to large data centers. Developers need detailed data on the energy consumption of their software to mitigate consumption issues. Previous approaches have a broader focus, such as on specific functions or programs, rather than source code statements. They primarily focus on estimating the CPU's energy consumption using point estimates, thereby disregarding other hardware effects and limiting their use for statistical reasoning and explainability. We developed a novel methodology to address the limitations of measuring only the CPU's consumption and using point estimates, focusing on predicting the energy usage of statically typed JVM-based programming languages, such as Java and Scala. We measure the energy consumption of Bytecode patterns, the translation from the programming language's source code statement to their Java Bytecode representation. With the energy measurements, we construct a statistical model using Bayesian statistics, which allows us to predict the energy consumption through statistical distributions and analyze individual factors. The model includes three factors we obtain statically from the code: data size, data type, operation, and one factor about the hardware platform the code executes on: device. To validate our methodology, we implemented it for Java and evaluated its energy predictions on unseen programs. We observe that all four factors are influential, notably that two devices of the same model may differ in energy consumption and that the operations and data types cause consumption differences. The experiments also show that the energy prediction of programs closely follows the program's real energy consumption, validating our approach. Our work presents a methodology for constructing an energy model that future work, such as verification tools, can use for their energy estimates.
>
---
#### [new 050] Do Large Language Models Walk Their Talk? Measuring the Gap Between Implicit Associations, Self-Report, and Behavioral Altruism
- **分类: cs.LG; cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究大语言模型（LLM）的利他行为一致性，旨在解决“模型自我认知与实际行为是否一致”的问题。通过隐式关联测试、行为选择任务和自评量表，发现模型虽具利他倾向但普遍高估自身行为，存在显著“美德信号差距”。提出以“校准差距”作为对齐评估新指标。**

- **链接: [https://arxiv.org/pdf/2512.01568v1](https://arxiv.org/pdf/2512.01568v1)**

> **作者:** Sandro Andric
>
> **备注:** 14 pages, 7 figures, 7 tables. Code and data available at https://github.com/sandroandric/LLMs_Altruism_Study_Code
>
> **摘要:** We investigate whether Large Language Models (LLMs) exhibit altruistic tendencies, and critically, whether their implicit associations and self-reports predict actual altruistic behavior. Using a multi-method approach inspired by human social psychology, we tested 24 frontier LLMs across three paradigms: (1) an Implicit Association Test (IAT) measuring implicit altruism bias, (2) a forced binary choice task measuring behavioral altruism, and (3) a self-assessment scale measuring explicit altruism beliefs. Our key findings are: (1) All models show strong implicit pro-altruism bias (mean IAT = 0.87, p < .0001), confirming models "know" altruism is good. (2) Models behave more altruistically than chance (65.6% vs. 50%, p < .0001), but with substantial variation (48-85%). (3) Implicit associations do not predict behavior (r = .22, p = .29). (4) Most critically, models systematically overestimate their own altruism, claiming 77.5% altruism while acting at 65.6% (p < .0001, Cohen's d = 1.08). This "virtue signaling gap" affects 75% of models tested. Based on these findings, we recommend the Calibration Gap (the discrepancy between self-reported and behavioral values) as a standardized alignment metric. Well-calibrated models are more predictable and behaviorally consistent; only 12.5% of models achieve the ideal combination of high prosocial behavior and accurate self-knowledge.
>
---
#### [new 051] When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究长上下文大语言模型代理的安全性问题。针对现有评估忽视代理在长上下文中的表现，研究发现模型在10万+令牌时性能下降超50%，拒绝有害请求的能力不稳定，暴露安全机制不稳风险，揭示了当前评估范式在长任务中的局限性。**

- **链接: [https://arxiv.org/pdf/2512.02445v1](https://arxiv.org/pdf/2512.02445v1)**

> **作者:** Tsimur Hadeliya; Mohammad Ali Jauhar; Nidhi Sakpal; Diogo Cruz
>
> **备注:** 12 pages, 11 figures. Accepted at AAAI 2026 TrustAgent Workshop
>
> **摘要:** Solving complex or long-horizon problems often requires large language models (LLMs) to use external tools and operate over a significantly longer context window. New LLMs enable longer context windows and support tool calling capabilities. Prior works have focused mainly on evaluation of LLMs on long-context prompts, leaving agentic setup relatively unexplored, both from capability and safety perspectives. Our work addresses this gap. We find that LLM agents could be sensitive to length, type, and placement of the context, exhibiting unexpected and inconsistent shifts in task performance and in refusals to execute harmful requests. Models with 1M-2M token context windows show severe degradation already at 100K tokens, with performance drops exceeding 50\% for both benign and harmful tasks. Refusal rates shift unpredictably: GPT-4.1-nano increases from $\sim$5\% to $\sim$40\% while Grok 4 Fast decreases from $\sim$80\% to $\sim$10\% at 200K tokens. Our work shows potential safety issues with agents operating on longer context and opens additional questions on the current metrics and paradigm for evaluating LLM agent safety on long multi-step tasks. In particular, our results on LLM agents reveal a notable divergence in both capability and safety performance compared to prior evaluations of LLMs on similar criteria.
>
---
#### [new 052] Contextual Image Attack: How Visual Context Exposes Multimodal Safety Vulnerabilities
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文针对多模态大模型的安全漏洞，提出图像中心的越狱攻击方法CIA。通过多智能体系统在视觉上下文中隐匿有害指令，结合四种可视化策略与上下文增强技术，显著提升攻击成功率，实验证明其对GPT-4o和Qwen2.5-VL-72B的攻击成功率达91.07%，有效揭示了视觉模态的潜在安全风险。**

- **链接: [https://arxiv.org/pdf/2512.02973v1](https://arxiv.org/pdf/2512.02973v1)**

> **作者:** Yuan Xiong; Ziqi Miao; Lijun Li; Chen Qian; Jie Li; Jing Shao
>
> **摘要:** While Multimodal Large Language Models (MLLMs) show remarkable capabilities, their safety alignments are susceptible to jailbreak attacks. Existing attack methods typically focus on text-image interplay, treating the visual modality as a secondary prompt. This approach underutilizes the unique potential of images to carry complex, contextual information. To address this gap, we propose a new image-centric attack method, Contextual Image Attack (CIA), which employs a multi-agent system to subtly embeds harmful queries into seemingly benign visual contexts using four distinct visualization strategies. To further enhance the attack's efficacy, the system incorporate contextual element enhancement and automatic toxicity obfuscation techniques. Experimental results on the MMSafetyBench-tiny dataset show that CIA achieves high toxicity scores of 4.73 and 4.83 against the GPT-4o and Qwen2.5-VL-72B models, respectively, with Attack Success Rates (ASR) reaching 86.31\% and 91.07\%. Our method significantly outperforms prior work, demonstrating that the visual modality itself is a potent vector for jailbreaking advanced MLLMs.
>
---
#### [new 053] OptPO: Optimal Rollout Allocation for Test-time Policy Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型测试时优化中的计算冗余问题，提出OptPO框架。通过贝叶斯序贯检验动态停止采样，自适应分配推理预算，减少滚动生成开销，同时利用保留样本进行策略更新。在多个推理基准上实现更低开销与更高或相当精度。**

- **链接: [https://arxiv.org/pdf/2512.02882v1](https://arxiv.org/pdf/2512.02882v1)**

> **作者:** Youkang Wang; Jian Wang; Rubing Chen; Tianyi Zeng; Xiao-Yong Wei; Qing Li
>
> **备注:** Work in Progress
>
> **摘要:** Test-time policy optimization enables large language models (LLMs) to adapt to distribution shifts by leveraging feedback from self-generated rollouts. However, existing methods rely on fixed-budget majority voting to estimate rewards, incurring substantial computational redundancy. We propose Optimal Rollout Allocation for Test-time Policy Optimization (OptPO), a principled framework that adaptively allocates inference budgets. By formulating the voting process as a Bayesian sequential probability ratio test, OptPO dynamically halts sampling once the posterior confidence in a consensus answer exceeds a specified threshold. Crucially, it utilizes the retained rollouts for on-policy updates, seamlessly integrating with algorithms like PPO or GRPO without requiring ground-truth labels. Across diverse reasoning benchmarks, OptPO significantly reduces rollout overhead compared to fixed-sample baselines while preserving or improving accuracy. By unifying statistically optimal stopping with test-time learning, OptPO offers a computationally efficient paradigm for test-time adaptation. The source code will be open upon acceptance at https://open-upon-acceptance.
>
---
#### [new 054] Story2MIDI: Emotionally Aligned Music Generation from Text
- **分类: cs.SD; cs.AI; cs.CL**

- **简介: 该论文提出Story2MIDI，一个基于Transformer的序列到序列模型，旨在从文本生成情感一致的音乐。针对文本与音乐情感对齐问题，构建了包含文本-音乐配对的情感数据集，通过客观指标与人类听觉实验验证，模型能有效捕捉并生成符合文本情绪的多样化音乐。**

- **链接: [https://arxiv.org/pdf/2512.02192v1](https://arxiv.org/pdf/2512.02192v1)**

> **作者:** Mohammad Shokri; Alexandra C. Salem; Gabriel Levine; Johanna Devaney; Sarah Ita Levitan
>
> **备注:** 8 pages (6 pages of main text + 2 pages of references and appendices), 4 figures, 1 table. Presented at IEEE Big Data 2025 3rd Workshop on AI Music Generation (AIMG 2025)
>
> **摘要:** In this paper, we introduce Story2MIDI, a sequence-to-sequence Transformer-based model for generating emotion-aligned music from a given piece of text. To develop this model, we construct the Story2MIDI dataset by merging existing datasets for sentiment analysis from text and emotion classification in music. The resulting dataset contains pairs of text blurbs and music pieces that evoke the same emotions in the reader or listener. Despite the small scale of our dataset and limited computational resources, our results indicate that our model effectively learns emotion-relevant features in music and incorporates them into its generation process, producing samples with diverse emotional responses. We evaluate the generated outputs using objective musical metrics and a human listening study, confirming the model's ability to capture intended emotional cues.
>
---
#### [new 055] Guided Self-Evolving LLMs with Minimal Human Supervision
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出R-Few框架，解决大模型自演化中因缺乏监督导致的性能退化问题。通过轻量级人类引导的挑战者-求解器协同机制，实现稳定、可控的自我进化，在数学与通用推理任务上显著提升模型性能，减少对人工数据的依赖。**

- **链接: [https://arxiv.org/pdf/2512.02472v1](https://arxiv.org/pdf/2512.02472v1)**

> **作者:** Wenhao Yu; Zhenwen Liang; Chengsong Huang; Kishan Panaganti; Tianqing Fang; Haitao Mi; Dong Yu
>
> **摘要:** AI self-evolution has long been envisioned as a path toward superintelligence, where models autonomously acquire, refine, and internalize knowledge from their own learning experiences. Yet in practice, unguided self-evolving systems often plateau quickly or even degrade as training progresses. These failures arise from issues such as concept drift, diversity collapse, and mis-evolution, as models reinforce their own biases and converge toward low-entropy behaviors. To enable models to self-evolve in a stable and controllable manner while minimizing reliance on human supervision, we introduce R-Few, a guided Self-Play Challenger-Solver framework that incorporates lightweight human oversight through in-context grounding and mixed training. At each iteration, the Challenger samples a small set of human-labeled examples to guide synthetic question generation, while the Solver jointly trains on human and synthetic examples under an online, difficulty-based curriculum. Across math and general reasoning benchmarks, R-Few achieves consistent and iterative improvements. For example, Qwen3-8B-Base improves by +3.0 points over R-Zero on math tasks and achieves performance on par with General-Reasoner, despite the latter being trained on 20 times more human data. Ablation studies confirm the complementary contributions of grounded challenger training and curriculum-based solver training, and further analysis shows that R-Few mitigates drift, yielding more stable and controllable co-evolutionary dynamics.
>
---
#### [new 056] Factor(T,U): Factored Cognition Strengthens Monitoring of Untrusted AI
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究AI安全控制问题，针对未受信任AI在复杂任务中隐藏风险的挑战，提出基于分解认知（factored cognition）的监控协议。通过将任务分解为子任务由可信模型监督、不可信模型独立求解，提升监控有效性。实验显示安全率从41%提升至63%，且不影响后门攻击难度，适用于更复杂任务。**

- **链接: [https://arxiv.org/pdf/2512.02157v1](https://arxiv.org/pdf/2512.02157v1)**

> **作者:** Aaron Sandoval; Cody Rushing
>
> **备注:** 7 pages body; 19 pages total; 8 figures
>
> **摘要:** The field of AI Control seeks to develop robust control protocols, deployment safeguards for untrusted AI which may be intentionally subversive. However, existing protocols that rely on weaker monitors to detect unsafe behavior often fail on complex tasks beyond the monitor's comprehension. We develop control protocols based on factored cognition, in which a trusted model decomposes a task, an untrusted model solves each resultant child task in isolation, and the results are reassembled into a full solution. These protocols may improve safety by several means, such as by simplifying the context for monitors, or by obscuring vulnerabilities in the environment. We implement our protocols in the APPS coding setting and red team them against backdoor attempts from an adversarial GPT-4.1 Nano. We find that: (i) Adding factored cognition to a trusted monitoring protocol can boost safety from 41% to 63%; (ii) Safety improves because monitor performance improves; (iii) Factored cognition makes it no harder for capable LLMs to write backdoors in APPS. While our protocols show low usefulness in APPS, they hold promise for more complex tasks.
>
---
#### [new 057] Misalignment of LLM-Generated Personas with Human Perceptions in Low-Resource Settings
- **分类: cs.CY; cs.CL**

- **简介: 该论文属于自然语言处理中的社会角色建模任务，旨在解决LLM生成的虚拟人格在低资源环境下与真实人类认知不一致的问题。研究通过对比八类LLM personas与孟加拉国真实人群在文化相关问题上的回答，发现LLM在共情、可信度及情感真实性上显著落后，且存在过度积极倾向，揭示其生成内容缺乏真实社会感知。**

- **链接: [https://arxiv.org/pdf/2512.02058v1](https://arxiv.org/pdf/2512.02058v1)**

> **作者:** Tabia Tanzin Prama; Christopher M. Danforth; Peter Sheridan Dodds
>
> **摘要:** Recent advances enable Large Language Models (LLMs) to generate AI personas, yet their lack of deep contextual, cultural, and emotional understanding poses a significant limitation. This study quantitatively compared human responses with those of eight LLM-generated social personas (e.g., Male, Female, Muslim, Political Supporter) within a low-resource environment like Bangladesh, using culturally specific questions. Results show human responses significantly outperform all LLMs in answering questions, and across all matrices of persona perception, with particularly large gaps in empathy and credibility. Furthermore, LLM-generated content exhibited a systematic bias along the lines of the ``Pollyanna Principle'', scoring measurably higher in positive sentiment ($Φ_{avg} = 5.99$ for LLMs vs. $5.60$ for Humans). These findings suggest that LLM personas do not accurately reflect the authentic experience of real people in resource-scarce environments. It is essential to validate LLM personas against real-world human data to ensure their alignment and reliability before deploying them in social science research.
>
---
#### [new 058] The brain-AI convergence: Predictive and generative world models for general-purpose computation
- **分类: q-bio.NC; cs.AI; cs.CL; cs.NE**

- **简介: 该论文探讨大脑与人工智能在构建世界模型上的共性，指出两者均通过预测-误差学习形成内部世界模型，实现感知与生成的多功能性。研究揭示了生物智能与人工系统在计算机制上的深层相似性，为理解通用智能提供了统一理论框架。**

- **链接: [https://arxiv.org/pdf/2512.02419v1](https://arxiv.org/pdf/2512.02419v1)**

> **作者:** Shogo Ohmae; Keiko Ohmae
>
> **备注:** 22 pages, 4 figures. Related to our earlier preprint "The brain versus AI" (arXiv:2411.16075) but a distinct article. The earlier work surveyed broad brain-AI parallels; here we focus on world-model-based computation and convergent evolution between the brain and AI, especially large language models
>
> **摘要:** Recent advances in general-purpose AI systems with attention-based transformers offer a potential window into how the neocortex and cerebellum, despite their relatively uniform circuit architectures, give rise to diverse functions and, ultimately, to human intelligence. This Perspective provides a cross-domain comparison between the brain and AI that goes beyond the traditional focus on visual processing, adopting the emerging perspecive of world-model-based computation. Here, we identify shared computational mechanisms in the attention-based neocortex and the non-attentional cerebellum: both predict future world events from past inputs and construct internal world models through prediction-error learning. These predictive world models are repurposed for seemingly distinct functions--understanding in sensory processing and generation in motor processing-- enabling the brain to achieve multi-domain capabilities and human-like adaptive intelligence. Notably, attention-based AI has independently converged on a similar learning paradigm and world-model-based computation. We conclude that these shared mechanisms in both biological and artificial systems constitute a core computational foundation for realizing diverse functions including high-level intelligence, despite their relatively uniform circuit structures. Our theoretical insights bridge neuroscience and AI, advancing our understanding of the computational essence of intelligence.
>
---
#### [new 059] LORE: A Large Generative Model for Search Relevance
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对电商搜索相关性任务，提出LORE框架，解决大模型在相关性评估中因任务单一化导致的性能瓶颈。通过分解相关性为知识推理、多模态匹配和规则遵循三大能力，设计两阶段训练与评估基准RAIR，实现线上GoodRate提升27%，并提出频率分层部署策略，形成可复用的方法论。**

- **链接: [https://arxiv.org/pdf/2512.03025v1](https://arxiv.org/pdf/2512.03025v1)**

> **作者:** Chenji Lu; Zhuo Chen; Hui Zhao; Zhiyuan Zeng; Gang Zhao; Junjie Ren; Ruicong Xu; Haoran Li; Songyan Liu; Pengjie Wang; Jian Xu; Bo Zheng
>
> **摘要:** Achievement. We introduce LORE, a systematic framework for Large Generative Model-based relevance in e-commerce search. Deployed and iterated over three years, LORE achieves a cumulative +27\% improvement in online GoodRate metrics. This report shares the valuable experience gained throughout its development lifecycle, spanning data, features, training, evaluation, and deployment. Insight. While existing works apply Chain-of-Thought (CoT) to enhance relevance, they often hit a performance ceiling. We argue this stems from treating relevance as a monolithic task, lacking principled deconstruction. Our key insight is that relevance comprises distinct capabilities: knowledge and reasoning, multi-modal matching, and rule adherence. We contend that a qualitative-driven decomposition is essential for breaking through current performance bottlenecks. Contributions. LORE provides a complete blueprint for the LLM relevance lifecycle. Key contributions include: (1) A two-stage training paradigm combining progressive CoT synthesis via SFT with human preference alignment via RL. (2) A comprehensive benchmark, RAIR, designed to evaluate these core capabilities. (3) A query frequency-stratified deployment strategy that efficiently transfers offline LLM capabilities to the online system. LORE serves as both a practical solution and a methodological reference for other vertical domains.
>
---
#### [new 060] Martingale Score: An Unsupervised Metric for Bayesian Rationality in LLM Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出“鞅分”（Martingale Score）作为无监督度量，评估大语言模型在推理中是否具备贝叶斯理性。针对迭代推理易引发信念固化的问题，利用鞅性质检测信念更新的可预测性，发现多类模型与方法普遍存在信念固化现象，并验证该分数能有效预测真实准确性。**

- **链接: [https://arxiv.org/pdf/2512.02914v1](https://arxiv.org/pdf/2512.02914v1)**

> **作者:** Zhonghao He; Tianyi Qiu; Hirokazu Shirado; Maarten Sap
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recent advances in reasoning techniques have substantially improved the performance of large language models (LLMs), raising expectations for their ability to provide accurate, truthful, and reliable information. However, emerging evidence suggests that iterative reasoning may foster belief entrenchment and confirmation bias, rather than enhancing truth-seeking behavior. In this study, we propose a systematic evaluation framework for belief entrenchment in LLM reasoning by leveraging the Martingale property from Bayesian statistics. This property implies that, under rational belief updating, the expected value of future beliefs should remain equal to the current belief, i.e., belief updates are unpredictable from the current belief. We propose the unsupervised, regression-based Martingale Score to measure violations of this property, which signal deviation from the Bayesian ability of updating on new evidence. In open-ended problem domains including event forecasting, value-laden questions, and academic paper review, we find such violations to be widespread across models and setups, where the current belief positively predicts future belief updates, a phenomenon which we term belief entrenchment. We identify the models, reasoning techniques, and domains more prone to belief entrenchment. Finally, we validate the Martingale Score by showing that it predicts ground-truth accuracy on problem domains where ground truth labels are available. This indicates that, while designed as an unsupervised metric that operates even in domains without access to ground truth, the Martingale Score is a useful proxy of the truth-seeking ability of a reasoning process.
>
---
#### [new 061] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG**

- **简介: 该论文针对长视频问答任务，解决模型因上下文容量有限导致的细节丢失与多尺度事件理解困难问题。提出WorldMM，一种动态多模态记忆代理，融合文本、语义与视觉记忆，通过自适应检索实现多粒度信息获取，显著提升长视频推理性能。**

- **链接: [https://arxiv.org/pdf/2512.02425v1](https://arxiv.org/pdf/2512.02425v1)**

> **作者:** Woongyeong Yeo; Kangsan Kim; Jaehong Yoon; Sung Ju Hwang
>
> **备注:** Project page : https://worldmm.github.io
>
> **摘要:** Recent advances in video large language models have demonstrated strong capabilities in understanding short clips. However, scaling them to hours- or days-long videos remains highly challenging due to limited context capacity and the loss of critical visual details during abstraction. Existing memory-augmented methods mitigate this by leveraging textual summaries of video segments, yet they heavily rely on text and fail to utilize visual evidence when reasoning over complex scenes. Moreover, retrieving from fixed temporal scales further limits their flexibility in capturing events that span variable durations. To address this, we introduce WorldMM, a novel multimodal memory agent that constructs and retrieves from multiple complementary memories, encompassing both textual and visual representations. WorldMM comprises three types of memory: episodic memory indexes factual events across multiple temporal scales, semantic memory continuously updates high-level conceptual knowledge, and visual memory preserves detailed information about scenes. During inference, an adaptive retrieval agent iteratively selects the most relevant memory source and leverages multiple temporal granularities based on the query, continuing until it determines that sufficient information has been gathered. WorldMM significantly outperforms existing baselines across five long video question-answering benchmarks, achieving an average 8.4% performance gain over previous state-of-the-art methods, showing its effectiveness on long video reasoning.
>
---
#### [new 062] On the Difficulty of Token-Level Modeling of Dysfluency and Fluency Shaping Artifacts
- **分类: eess.AS; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对语音识别中口吃与流畅性修饰的建模难题，提出轻量级适配方法，将非流利现象作为特殊标记纳入转录。通过多步微调与语言自适应预训练，在英德语数据上提升识别效果，揭示现有系统对非英语数据的偏差问题。**

- **链接: [https://arxiv.org/pdf/2512.02027v1](https://arxiv.org/pdf/2512.02027v1)**

> **作者:** Kashaf Gulzar; Dominik Wagner; Sebastian P. Bayerl; Florian Hönig; Tobias Bocklet; Korbinian Riedhammer
>
> **备注:** 6 pages, 1 figure. Accepted to ASRU 2025. This is the arXiv preprint of the accepted paper
>
> **摘要:** Automatic transcription of stuttered speech remains a challenge, even for modern end-to-end (E2E) automatic speech recognition (ASR) frameworks. Dysfluencies and fluency-shaping artifacts are often overlooked, resulting in non-verbatim transcriptions with limited clinical and research value. We propose a parameter-efficient adaptation method to decode dysfluencies and fluency modifications as special tokens within transcriptions, evaluated on simulated (LibriStutter, English) and natural (KSoF, German) stuttered speech datasets. To mitigate ASR performance disparities and bias towards English, we introduce a multi-step fine-tuning strategy with language-adaptive pretraining. Tokenization analysis further highlights the tokenizer's English-centric bias, which poses challenges for improving performance on German data. Our findings demonstrate the effectiveness of lightweight adaptation techniques for dysfluency-aware ASR while exposing key limitations in multilingual E2E systems.
>
---
## 更新

#### [replaced 001] Computational Fact-Checking of Online Discourse: Scoring scientific accuracy in climate change related news articles
- **分类: cs.CL**

- **简介: 该论文属于科学事实核查任务，旨在半自动量化气候相关新闻的科学准确性。通过语义化处理媒体内容并对比知识图谱，利用LLM提取陈述并进行分析，构建了可评估可信度的流程。但现有气候知识图谱不足，难以实现大规模精细标注，需完善FAIR标准的基准数据与评估指标。**

- **链接: [https://arxiv.org/pdf/2505.07409v2](https://arxiv.org/pdf/2505.07409v2)**

> **作者:** Tim Wittenborg; Constantin Sebastian Tremel; Markus Stocker; Sören Auer
>
> **备注:** 8 pages, 7 figures, accepted at ICKG 2025
>
> **摘要:** Democratic societies need reliable information. Misinformation in popular media, such as news articles or videos, threatens to impair civic discourse. Citizens are, unfortunately, not equipped to verify the flood of content consumed daily at increasing rates. This work aims to quantify the scientific accuracy of online media semi-automatically. We investigate the state of the art of climate-related ground truth knowledge representation. By semantifying media content of unknown veracity, their statements can be compared against these ground truth knowledge graphs. We implemented a workflow using LLM-based statement extraction and knowledge graph analysis. Our implementation can streamline content processing towards state-of-the-art knowledge representation and veracity quantification. Developed and evaluated with the help of 27 experts and detailed interviews with 10, the tool evidently provides a beneficial veracity indication. These findings are supported by 43 anonymous participants from a parallel user survey. This initial step, however, is unable to annotate public media at the required granularity and scale. Additionally, the identified state of climate change knowledge graphs is vastly insufficient to support this neurosymbolic fact-checking approach. Further work towards a FAIR (Findable, Accessible, Interoperable, Reusable) ground truth and complementary metrics is required to support civic discourse scientifically.
>
---
#### [replaced 002] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences via Tournament Graph Reconstruction
- **分类: cs.CL**

- **简介: 该论文针对大模型评估中非传递性偏好问题，提出ELSPR框架，通过锦标赛图重构与强连通分量分析识别低质数据，基于结构熵量化偏好模糊性，过滤非传递性样本。实验表明，该方法显著提升评估一致性与模型性能。**

- **链接: [https://arxiv.org/pdf/2505.17691v3](https://arxiv.org/pdf/2505.17691v3)**

> **作者:** Yan Yu; Yilun Liu; Minggui He; Shimin Tao; Weibin Meng; Xinhua Yang; Li Zhang; Hongxia Ma; Dengye Li; Daimeng Wei; Boxing Chen; Fuliang Li
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Pairwise evaluation of large language models (LLMs) has become the dominant paradigm for benchmarking open-ended tasks, yet non-transitive preferences, where evaluators prefer A over B, B over C, but C over A, fundamentally undermine ranking reliability. We show that this critical issue stems largely from low-quality data that contains inherently ambiguous preference pairs. To address this challenge, we propose ELSPR, a principled graph-theoretic framework that models pairwise preferences as tournament graphs and systematically identifies problematic training data. ELSPR quantifies non-transitivity through strongly connected components (SCCs) analysis and measures overall preference clarity using a novel normalized directed graph structural entropy metric. Our filtering methodology selectively removes preference data that induce non-transitivity while preserving transitive preferences. Extensive experiments on the AlpacaEval benchmark demonstrate that models fine-tuned on ELSPR-filtered data achieve substantial improvements: a 13.8% reduction in non-transitivity, a 0.088 decrease in structural entropy, and significantly enhanced discriminative power in real-world evaluation systems. Human validation confirms that discarded data exhibit dramatically lower inter-annotator agreement (34.4% vs. 52.6%) and model-human consistency (51.2% vs. 80.6%) compared to cleaned data. These findings establish ELSPR as an effective data self-purification approach for developing more robust, consistent, and human-aligned LLM evaluation systems.
>
---
#### [replaced 003] Unifying Linear-Time Attention via Latent Probabilistic Modelling
- **分类: cs.CL; stat.ML**

- **简介: 该论文针对Transformer模型在长序列建模中因二次注意力机制导致的计算瓶颈问题，提出一种基于潜在概率建模的线性时间注意力机制。通过引入有向参数化和递归查询/键设计，增强语言建模的因果与顺序特性，实现高效且高性能的序列建模。**

- **链接: [https://arxiv.org/pdf/2402.17512v5](https://arxiv.org/pdf/2402.17512v5)**

> **作者:** Rares Dolga; Lucas Maystre; Marius Cobzarenco; David Barber
>
> **摘要:** Transformers have achieved state-of-the-art results across a range of domains, but their quadratic attention mechanism poses significant challenges for long-sequence modelling. Recent efforts to design linear-time attention mechanisms have yielded more scalable alternatives, yet often at the cost of performance, particularly on discrete data such as language. In this work, we revisit linear attention through the lens of probabilistic graphical models. We first show that standard linear attention can be interpreted as an undirected latent variable model, revealing a key limitation: the absence of directionality. To address this, we propose a novel directed parameterisation of linear attention that introduces an asymmetric structure, enabling an interpretation aligned with the causal and sequential nature of language. Our formulation integrates global latent-variable attention with local standard attention in a fully probabilistic framework. Additionally, we introduce a recurrent parameterisation of queries and keys that avoids reliance on relative positional encodings, often incompatible with linear attention. Experiments on language modelling benchmarks demonstrate that our model achieves competitive performance with standard attention and outperforms existing linear attention variants.
>
---
#### [replaced 004] Pre-trained Language Models Improve the Few-shot Prompt Ability of Decision Transformer
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究离线强化学习中的少样本提示问题。针对决策变压器（DT）依赖大量数据、任务区分能力弱的问题，提出基于预训练语言模型初始化的LPDT框架，结合LoRA微调与提示正则化，显著提升少样本性能。实验表明，仅用10%数据即达Prompt-DT效果。**

- **链接: [https://arxiv.org/pdf/2408.01402v2](https://arxiv.org/pdf/2408.01402v2)**

> **作者:** Yu Yang; Pan Xu
>
> **备注:** 2 figures, 10 tables. Published in Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Decision Transformer (DT) has emerged as a promising class of algorithms in offline reinforcement learning (RL) tasks, leveraging pre-collected datasets and Transformer's capability to model long sequences. Recent works have demonstrated that using parts of trajectories from training tasks as prompts in DT enhances its performance on unseen tasks, giving rise to Prompt-DT methods. However, collecting data from specific environments can be both costly and unsafe in many scenarios, leading to suboptimal performance and limited few-shot prompt abilities due to the data-hungry nature of Transformer-based models. Additionally, the limited datasets used in pre-training make it challenging for Prompt-DT type of methods to distinguish between various RL tasks through prompts alone. To address these challenges, we introduce the Language model-initialized Prompt Decision Transformer (LPDT) framework, which leverages pretrained language models providing rich prior knowledge for RL tasks and fine-tunes the sequence model using Low-rank Adaptation (LoRA) for meta-RL problems. We further incorporate prompt regularization to effectively differentiate between tasks based on prompt feature representations. Comprehensive empirical studies demonstrate that initializing with a pre-trained language model provides the prior knowledge and achieves a similar performance with Prompt-DT under only $10\%$ data in some MuJoCo control tasks. We also provide a thorough ablation study to validate the effectiveness of each component, including sequence modeling, language models, prompt regularizations, and prompt strategies.
>
---
#### [replaced 005] WebMall - A Multi-Shop Benchmark for Evaluating Web Agents [Technical Report]
- **分类: cs.CL**

- **简介: 该论文提出WebMall，首个离线多店铺电商基准，用于评估基于大模型的网络代理在复杂比价任务中的表现。针对现有基准或在线、或仅支持单店且任务简单的问题，WebMall通过模拟四个异构店铺，涵盖从搜索到结算的多样化挑战性任务，验证显示顶级代理完成率不足55%，凸显其难度。**

- **链接: [https://arxiv.org/pdf/2508.13024v2](https://arxiv.org/pdf/2508.13024v2)**

> **作者:** Ralph Peeters; Aaron Steiner; Luca Schwarz; Julian Yuya Caspary; Christian Bizer
>
> **摘要:** LLM-based web agents have the potential to automate long-running web tasks, such as searching for products in multiple e-shops and subsequently ordering the cheapest products that meet the users needs. Benchmarks for evaluating web agents either require agents to perform tasks online using the live Web or offline using simulated environments, which allow for the exact reproduction of the experimental setup. While DeepShop provides an online benchmark that requires agents to perform challenging shopping tasks, existing offline benchmarks such as WebShop, WebArena, or Mind2Web cover only comparatively simple e-commerce tasks that need to be performed against a single shop containing product data from a single source. What is missing is an e-commerce benchmark that simulates multiple shops containing heterogeneous product data and requires agents to perform complex tasks. We fill this gap by introducing WebMall, the first offline multi-shop benchmark for evaluating web agents on challenging comparison shopping tasks. WebMall consists of four simulated shops populated with product data extracted from the Common Crawl. The WebMall tasks range from specific product searches and price comparisons to advanced queries for complementary or substitute products, as well as checkout processes. We validate WebMall using eight agents that differ in observation space, availability of short-term memory, and the employed LLM. The validation highlights the difficulty of the benchmark, with even the best-performing agents achieving task completion rates below 55% in the task categories cheapest product search and vague product search.
>
---
#### [replaced 006] Look Before you Leap: Estimating LLM Benchmark Scores from Descriptions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究大语言模型性能的文本描述预测任务，旨在通过任务描述和配置预估其在基准测试中的得分。针对评估瓶颈问题，作者构建了PRECOG数据集，实现无需实际运行模型的性能预测，支持早期实验设计与资源分配决策。**

- **链接: [https://arxiv.org/pdf/2509.20645v2](https://arxiv.org/pdf/2509.20645v2)**

> **作者:** Jungsoo Park; Ethan Mendes; Gabriel Stanovsky; Alan Ritter
>
> **备注:** 32 pages, 7 figures
>
> **摘要:** Progress in large language models is constrained by an evaluation bottleneck: build a benchmark, run models, then iterate. We ask a question: can we forecast outcomes before running any experiments to inform earlier study design? For example, a team building an AI assistant for a certain task can estimate whether expected performance is around 50 or closer to 80, evidence that supports whether to proceed to a pilot study, how to scope it, and how to allocate resources. We study text-only performance forecasting, where a model predicts a score from a redacted task description and intended configuration, with no access to dataset instances. To support systematic study, we curate PRECOG, a corpus of redacted description-performance pairs spanning diverse tasks, domains, and metrics. We scrape task and configuration descriptions from arXiv, yielding 2,290 instances covering 1,519 papers, and construct a leakage free test split using papers published after the knowledge cutoff of the evaluated models. Experiments show the task is challenging but feasible: reasoning models achieve moderate prediction performance with well calibrated uncertainty, reaching mean absolute error as low as 9.9 at high confidence thresholds. We further test a zero-leakage setting, forecasting on newly released datasets or experiments before their papers are indexed, where GPT5 with built in web search still attains nontrivial prediction accuracy. Overall, our corpus and analyses offer an initial step toward open ended anticipatory evaluation, supporting difficulty estimation and smarter experiment prioritization.
>
---
#### [replaced 007] Unleashing Hour-Scale Video Training for Long Video-Language Understanding
- **分类: cs.CV; cs.CL**

- **简介: 该论文针对长视频语言理解中训练数据稀缺问题，提出VideoMarathon数据集（9700小时，3-60分钟视频）和Hour-LLaVA模型。通过支持小时级视频训练与推理，解决长期视频理解难题，实现多任务高效建模，显著提升长视频语言理解性能。**

- **链接: [https://arxiv.org/pdf/2506.05332v2](https://arxiv.org/pdf/2506.05332v2)**

> **作者:** Jingyang Lin; Jialian Wu; Ximeng Sun; Ze Wang; Jiang Liu; Yusheng Su; Xiaodong Yu; Hao Chen; Jiebo Luo; Zicheng Liu; Emad Barsoum
>
> **备注:** NeurIPS 2025, Project page: https://videomarathon.github.io/
>
> **摘要:** Recent long-form video-language understanding benchmarks have driven progress in video large multimodal models (Video-LMMs). However, the scarcity of well-annotated long videos has left the training of hour-long Video-LMMs underexplored. To close this gap, we present VideoMarathon, a large-scale hour-long video instruction-following dataset. This dataset includes around 9,700 hours of long videos sourced from diverse domains, ranging from 3 to 60 minutes per video. Specifically, it contains 3.3M high-quality QA pairs, spanning six fundamental topics: temporality, spatiality, object, action, scene, and event. Compared to existing video instruction datasets, VideoMarathon significantly extends training video durations up to 1 hour, and supports 22 diverse tasks requiring both short- and long-term video comprehension. Building on VideoMarathon, we propose Hour-LLaVA, a powerful and efficient Video-LMM for hour-scale video-language modeling. It enables hour-long video training and inference at 1-FPS sampling by leveraging a memory augmentation module, which adaptively integrates question-relevant and spatiotemporally informative semantics from the cached full video context. In our experiments, Hour-LLaVA achieves the best performance on multiple representative long video-language benchmarks, demonstrating the high quality of the VideoMarathon dataset and the superiority of the Hour-LLaVA model.
>
---
#### [replaced 008] SkyLadder: Better and Faster Pretraining via Context Window Scheduling
- **分类: cs.CL**

- **简介: 该论文针对大模型预训练中长上下文窗口导致效率低下的问题，提出SkyLadder调度策略，通过短到长的上下文窗口渐进式训练，在固定词元预算下提升模型性能与训练速度。实验表明，该方法在1B和3B参数模型上显著优于基线，性能提升最高达3.7%，训练速度加快22%。**

- **链接: [https://arxiv.org/pdf/2503.15450v2](https://arxiv.org/pdf/2503.15450v2)**

> **作者:** Tongyao Zhu; Qian Liu; Haonan Wang; Shiqi Chen; Xiangming Gu; Tianyu Pang; Min-Yen Kan
>
> **备注:** Accepted to NeurIPS 2025. 10 pages
>
> **摘要:** Recent advancements in LLM pretraining have featured ever-expanding context windows to process longer sequences. However, our pilot study reveals that models pretrained with shorter context windows consistently outperform their long-context counterparts under a fixed token budget. This finding motivates us to explore an optimal context window scheduling strategy to better balance long-context capability with pretraining efficiency. To this end, we propose SkyLadder, a simple yet effective approach that implements a short-to-long context window transition. SkyLadder preserves strong standard benchmark performance, while matching or exceeding baseline results on long context tasks. Through extensive experiments, we pre-train 1B-parameter models (up to 32K context) and 3B-parameter models (8K context) on 100B tokens, demonstrating that SkyLadder yields consistent gains of up to 3.7% on common benchmarks, while achieving up to 22% faster training speeds compared to baselines. The code is at https://github.com/sail-sg/SkyLadder.
>
---
#### [replaced 009] Between Help and Harm: An Evaluation of Mental Health Crisis Handling by LLMs
- **分类: cs.CL; cs.CY**

- **简介: 该论文针对大模型在心理危机干预中的安全与有效性问题，构建了六类危机分类体系，建立了2252条标注数据集，并设计临床评估协议。通过测试五款模型，发现其在识别隐晦信号和生成安全回应方面存在不足，强调需加强上下文理解与安全对齐。**

- **链接: [https://arxiv.org/pdf/2509.24857v2](https://arxiv.org/pdf/2509.24857v2)**

> **作者:** Adrian Arnaiz-Rodriguez; Miguel Baidal; Erik Derner; Jenn Layton Annable; Mark Ball; Mark Ince; Elvira Perez Vallejos; Nuria Oliver
>
> **摘要:** Large language model-powered chatbots have transformed how people seek information, especially in high-stakes contexts like mental health. Despite their support capabilities, safe detection and response to crises such as suicidal ideation and self-harm are still unclear, hindered by the lack of unified crisis taxonomies and clinical evaluation standards. We address this by creating: (1) a taxonomy of six crisis categories; (2) a dataset of over 2,000 inputs from 12 mental health datasets, classified into these categories; and (3) a clinical response assessment protocol. We also use LLMs to identify crisis inputs and audit five models for response safety and appropriateness. First, we built a clinical-informed crisis taxonomy and evaluation protocol. Next, we curated 2,252 relevant examples from over 239,000 user inputs, then tested three LLMs for automatic classification. In addition, we evaluated five models for the appropriateness of their responses to a user's crisis, graded on a 5-point Likert scale from harmful (1) to appropriate (5). While some models respond reliably to explicit crises, risks still exist. Many outputs, especially in self-harm and suicidal categories, are inappropriate or unsafe. Different models perform variably; some, like gpt-5-nano and deepseek-v3.2-exp, have low harm rates, but others, such as gpt-4o-mini and grok-4-fast, generate more unsafe responses. All models struggle with indirect signals, default replies, and context misalignment. These results highlight the urgent need for better safeguards, crisis detection, and context-aware responses in LLMs. They also show that alignment and safety practices, beyond scale, are crucial for reliable crisis support. Our taxonomy, datasets, and evaluation methods support ongoing AI mental health research, aiming to reduce harm and protect vulnerable users.
>
---
#### [replaced 010] LLM-as-a-Supervisor: Mistaken Therapeutic Behaviors Trigger Targeted Supervisory Feedback
- **分类: cs.CL**

- **简介: 该论文提出“LLM-as-a-Supervisor”范式，解决心理治疗中缺乏明确反馈标准的问题。通过构建含典型错误的对话数据集，让LLM监督者识别并纠正治疗错误，实现对真实治疗师的高效培训。**

- **链接: [https://arxiv.org/pdf/2508.09042v2](https://arxiv.org/pdf/2508.09042v2)**

> **作者:** Chen Xu; Zhenyu Lv; Tian Lan; Xianyang Wang; Luyao Ji; Leyang Cui; Minqiang Yang; Jian Shen; Qunxi Dong; Xiuling Liu; Juan Wang; Bin Hu
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Although large language models (LLMs) hold significant promise in psychotherapy, their direct application in patient-facing scenarios raises ethical and safety concerns. Therefore, this work shifts towards developing an LLM as a supervisor to train real therapists. In addition to the privacy of clinical therapist training data, a fundamental contradiction complicates the training of therapeutic behaviors: clear feedback standards are necessary to ensure a controlled training system, yet there is no absolute "gold standard" for appropriate therapeutic behaviors in practice. In contrast, many common therapeutic mistakes are universal and identifiable, making them effective triggers for targeted feedback that can serve as clearer evidence. Motivated by this, we create a novel therapist-training paradigm: (1) guidelines for mistaken behaviors and targeted correction strategies are first established as standards; (2) a human-in-the-loop dialogue-feedback dataset is then constructed, where a mistake-prone agent intentionally makes standard mistakes during interviews naturally, and a supervisor agent locates and identifies mistakes and provides targeted feedback; (3) after fine-tuning on this dataset, the final supervisor model is provided for real therapist training. The detailed experimental results of automated, human and downstream assessments demonstrate that models fine-tuned on our dataset MATE, can provide high-quality feedback according to the clinical guideline, showing significant potential for the therapist training scenario.
>
---
#### [replaced 011] XISM: an eXploratory and Interactive Graph Tool to Visualize and Evaluate Semantic Map Models
- **分类: cs.CL**

- **简介: 该论文提出XISM，一种交互式语义地图构建工具，旨在解决现有方法在可扩展性与可解释性间的矛盾。通过结合数据驱动推断与专家知识，实现可视化迭代优化，提升决策透明度与效率，支持语言类型学中的语义关系建模。**

- **链接: [https://arxiv.org/pdf/2507.04070v2](https://arxiv.org/pdf/2507.04070v2)**

> **作者:** Zhu Liu; Zhen Hu; Lei Dai; Yu Xuan; Ying Liu
>
> **备注:** Paper under review
>
> **摘要:** Semantic map models visualize systematic relations among semantic functions through graph structures and are widely used in linguistic typology. However, existing construction methods either depend on labor-intensive expert reasoning or on fully automated systems lacking expert involvement, creating a tension between scalability and interpretability. We introduce \textbf{XISM}, an interactive system that combines data-driven inference with expert knowledge. XISM generates candidate maps via a top-down procedure and allows users to iteratively refine edges in a visual interface, with real-time metric feedback. Experiments in three semantic domains and expert interviews show that XISM improves linguistic decision transparency and controllability in semantic-map construction while maintaining computational efficiency. XISM provides a collaborative approach for scalable and interpretable semantic-map building. The system\footnote{https://app.xism2025.xin/} , source code\footnote{https://github.com/hank317/XISM} , and demonstration video\footnote{https://youtu.be/m5laLhGn6Ys} are publicly available.
>
---
#### [replaced 012] From Word Vectors to Multimodal Embeddings: Techniques, Applications, and Future Directions For Large Language Models
- **分类: cs.CL**

- **简介: 该论文综述词向量到多模态嵌入的技术演进，涵盖静态与上下文嵌入、跨语言与个性化应用，探讨其在视觉、机器人等领域的融合。旨在解决语言模型的表示、可解释性与多模态对齐问题，提出未来在训练效率、可解释性及非文本模态融合方面的研究方向。**

- **链接: [https://arxiv.org/pdf/2411.05036v3](https://arxiv.org/pdf/2411.05036v3)**

> **作者:** Charles Zhang; Benji Peng; Xintian Sun; Qian Niu; Junyu Liu; Keyu Chen; Ming Li; Pohsun Feng; Ziqian Bi; Ming Liu; Yichao Zhang; Xinyuan Song; Cheng Fei; Caitlyn Heqi Yin; Lawrence KQ Yan; Hongyang He; Tianyang Wang
>
> **备注:** 21 pages
>
> **摘要:** Word embeddings and language models have transformed natural language processing (NLP) by facilitating the representation of linguistic elements in continuous vector spaces. This review visits foundational concepts such as the distributional hypothesis and contextual similarity, tracing the evolution from sparse representations like one-hot encoding to dense embeddings including Word2Vec, GloVe, and fastText. We examine both static and contextualized embeddings, underscoring advancements in models such as ELMo, BERT, and GPT and their adaptations for cross-lingual and personalized applications. The discussion extends to sentence and document embeddings, covering aggregation methods and generative topic models, along with the application of embeddings in multimodal domains, including vision, robotics, and cognitive science. Advanced topics such as model compression, interpretability, numerical encoding, and bias mitigation are analyzed, addressing both technical challenges and ethical implications. Additionally, we identify future research directions, emphasizing the need for scalable training techniques, enhanced interpretability, and robust grounding in non-textual modalities. By synthesizing current methodologies and emerging trends, this survey offers researchers and practitioners an in-depth resource to push the boundaries of embedding-based language models.
>
---
#### [replaced 013] AI Text Detectors and the Misclassification of Slightly Polished Arabic Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究AI文本检测器对轻微润色阿拉伯语文本的误判问题。针对现有检测模型在阿拉伯语场景下易将人类写作经AI润色后的内容误判为AI生成的问题，构建两个数据集，评估14个LLM及商业检测器性能。结果表明，多数模型在轻微润色后准确率大幅下降，暴露其可靠性缺陷。**

- **链接: [https://arxiv.org/pdf/2511.16690v2](https://arxiv.org/pdf/2511.16690v2)**

> **作者:** Saleh Almohaimeed; Saad Almohaimeed; Mousa Jari; Khaled A. Alobaid; Fahad Alotaibi
>
> **备注:** (Submitted) Journal of Big Data
>
> **摘要:** Many AI detection models have been developed to counter the presence of articles created by artificial intelligence (AI). However, if a human-authored article is slightly polished by AI, a shift will occur in the borderline decision of these AI detection models, leading them to consider it as AI-generated article. This misclassification may result in falsely accusing authors of AI plagiarism and harm the credibility of AI detectors. In English, some efforts were made to meet this challenge, but not in Arabic. In this paper, we generated two datasets. The first dataset contains 800 Arabic articles, half AI-generated and half human-authored. We used it to evaluate 14 Large Language models (LLMs) and commercial AI detectors to assess their ability in distinguishing between human-authored and AI-generated articles. The best 8 models were chosen to act as detectors for our primary concern, which is whether they would consider slightly polished human-authored text as AI-generated. The second dataset, Ar-APT, contains 400 Arabic human-authored articles polished by 10 LLMs using 4 polishing settings, totaling 16400 samples. We use it to evaluate the 8 nominated models and determine whether slight polishing will affect their performance. The results reveal that all AI detectors incorrectly attribute a significant number of articles to AI. The best performing LLM, Claude-4 Sonnet, achieved 83.51\%, its performance decreased to 57.63\% for articles slightly polished by LLaMA-3. Whereas the best performing commercial model, originality.AI, achieves 92\% accuracy, dropped to 12\% for articles slightly polished by Mistral or Gemma-3.
>
---
#### [replaced 014] Eka-Eval: An Evaluation Framework for Low-Resource Multilingual Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出Eka-Eval框架，解决低资源多语言大模型评估缺乏统一、灵活工具的问题。集成50+多语言基准，支持本地与私有模型，具备11项核心功能，实现可扩展的多语言评估，显著提升易用性与可复现性。**

- **链接: [https://arxiv.org/pdf/2507.01853v4](https://arxiv.org/pdf/2507.01853v4)**

> **作者:** Samridhi Raj Sinha; Rajvee Sheth; Abhishek Upperwal; Mayank Singh
>
> **摘要:** The rapid evolution of Large Language Models' has underscored the need for evaluation frameworks that are globally applicable, flexible, and modular, and that support a wide range of tasks, model types, and linguistic settings. We introduce EKA-EVAL, a unified, end- to-end framework that combines a zero-code web interface and an interactive CLI to ensure broad accessibility. It integrates 50+ multilingual benchmarks across nine evaluation categories, supports local and proprietary models, and provides 11 core capabilities through a modular, plug-and-play architecture. Designed for scalable, multilingual evaluation with support for low-resource multilingual languages, EKA-EVAL is, to the best of our knowledge, the first suite to offer comprehensive coverage in a single platform. Comparisons against five existing baselines indicate improvements of at least 2x better on key usability measures, with the highest user satisfaction, faster setup times, and consistent benchmark reproducibility. The framework is open-source and publicly available at https://github.com/lingo-iitgn/eka-eval.
>
---
#### [replaced 015] LLMEval-3: A Large-Scale Longitudinal Study on Robust and Fair Evaluation of Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对大语言模型评估中静态基准导致的数据污染和排行榜过拟合问题，提出动态评估框架LLMEval-3。通过22万条研究生级题目动态抽样、抗作弊架构与人工对齐的评分机制，实现更公平、鲁棒的模型评估，揭示了知识记忆的性能上限及静态评估的漏洞。**

- **链接: [https://arxiv.org/pdf/2508.05452v3](https://arxiv.org/pdf/2508.05452v3)**

> **作者:** Ming Zhang; Yujiong Shen; Jingyi Deng; Yuhui Wang; Yue Zhang; Junzhe Wang; Shichun Liu; Shihan Dou; Huayu Sha; Qiyuan Peng; Changhao Jiang; Jingqi Tong; Yilong Wu; Zhihao Zhang; Mingqi Wu; Zhiheng Xi; Mingxu Chai; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Existing evaluation of Large Language Models (LLMs) on static benchmarks is vulnerable to data contamination and leaderboard overfitting, critical issues that obscure true model capabilities. To address this, we introduce LLMEval-3, a framework for dynamic evaluation of LLMs. LLMEval-3 is built on a proprietary bank of 220k graduate-level questions, from which it dynamically samples unseen test sets for each evaluation run. Its automated pipeline ensures integrity via contamination-resistant data curation, a novel anti-cheating architecture, and a calibrated LLM-as-a-judge process achieving 90% agreement with human experts, complemented by a relative ranking system for fair comparison. An 20-month longitudinal study of nearly 50 leading models reveals a performance ceiling on knowledge memorization and exposes data contamination vulnerabilities undetectable by static benchmarks. The framework demonstrates exceptional robustness in ranking stability and consistency, providing strong empirical validation for the dynamic evaluation paradigm. LLMEval-3 offers a robust and credible methodology for assessing the true capabilities of LLMs beyond leaderboard scores, promoting the development of more trustworthy evaluation standards.
>
---
#### [replaced 016] Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型（LLM）内部表示不透明的问题，提出超维度探针（Hyperdimensional Probe），融合符号表示与神经探测，统一输入导向与输出导向的可解释性方法。通过向量符号架构实现跨模型、跨规模的概念提取，支持联合输入输出分析，深化对语义表征的理解。**

- **链接: [https://arxiv.org/pdf/2509.25045v2](https://arxiv.org/pdf/2509.25045v2)**

> **作者:** Marco Bronzini; Carlo Nicolini; Bruno Lepri; Jacopo Staiano; Andrea Passerini
>
> **摘要:** Despite their capabilities, Large Language Models (LLMs) remain opaque with limited understanding of their internal representations. Current interpretability methods either focus on input-oriented feature extraction, such as supervised probes and Sparse Autoencoders (SAEs), or on output distribution inspection, such as logit-oriented approaches. A full understanding of LLM vector spaces, however, requires integrating both perspectives, something existing approaches struggle with due to constraints on latent feature definitions. We introduce the Hyperdimensional Probe, a hybrid supervised probe that combines symbolic representations with neural probing. Leveraging Vector Symbolic Architectures (VSAs) and hypervector algebra, it unifies prior methods: the top-down interpretability of supervised probes, SAE's sparsity-driven proxy space, and output-oriented logit investigation. This allows deeper input-focused feature extraction while supporting output-oriented investigation. Our experiments show that our method consistently extracts meaningful concepts across LLMs, embedding sizes, and setups, uncovering concept-driven patterns in analogy-oriented inference and QA-focused text generation. By supporting joint input-output analysis, this work advances semantic understanding of neural representations while unifying the complementary perspectives of prior methods.
>
---
#### [replaced 017] Evolution and compression in LLMs: On the emergence of human-aligned categorization
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）能否自发形成人类对齐的语义分类系统。针对“语义分类是否可实现信息瓶颈最优压缩”这一核心问题，作者以颜色命名为例，通过实验发现：大模型在指令微调后能提升与人类的对齐度；在模拟文化演化中，仅具备强上下文学习能力的模型能逼近人类的高效分类模式，揭示了语义效率源于相同的基本原理。**

- **链接: [https://arxiv.org/pdf/2509.08093v3](https://arxiv.org/pdf/2509.08093v3)**

> **作者:** Nathaniel Imel; Noga Zaslavsky
>
> **备注:** Accepted at CogInterp: Interpreting Cognition in Deep Learning Models Workshop at NeurIPS 2025
>
> **摘要:** Converging evidence suggests that human systems of semantic categories achieve near-optimal compression via the Information Bottleneck (IB) complexity-accuracy tradeoff. Large language models (LLMs) are not trained for this objective, which raises the question: are LLMs capable of evolving efficient human-aligned semantic systems? To address this question, we focus on color categorization -- a key testbed of cognitive theories of categorization with uniquely rich human data -- and replicate with LLMs two influential human studies. First, we conduct an English color-naming study, showing that LLMs vary widely in their complexity and English-alignment, with larger instruction-tuned models achieving better alignment and IB-efficiency. Second, to test whether these LLMs simply mimic patterns in their training data or actually exhibit a human-like inductive bias toward IB-efficiency, we simulate cultural evolution of pseudo color-naming systems in LLMs via a method we refer to as Iterated in-Context Language Learning (IICLL). We find that akin to humans, LLMs iteratively restructure initially random systems towards greater IB-efficiency. However, only a model with strongest in-context capabilities (Gemini 2.0) is able to recapitulate the wide range of near-optimal IB-tradeoffs observed in humans, while other state-of-the-art models converge to low-complexity solutions. These findings demonstrate how human-aligned semantic categories can emerge in LLMs via the same fundamental principle that underlies semantic efficiency in humans.
>
---
#### [replaced 018] From Code Foundation Models to Agents and Applications: A Practical Guide to Code Intelligence
- **分类: cs.SE; cs.CL**

- **简介: 该论文聚焦代码智能任务，旨在解决大模型在代码生成中的准确性、安全性和实际应用落地问题。通过系统分析代码预训练、微调与强化学习等技术，对比通用与专用模型性能，揭示研究与实践差距，并提供可复现的实验指导，推动代码AI从理论向工程应用转化。**

- **链接: [https://arxiv.org/pdf/2511.18538v3](https://arxiv.org/pdf/2511.18538v3)**

> **作者:** Jian Yang; Xianglong Liu; Weifeng Lv; Ken Deng; Shawn Guo; Lin Jing; Yizhi Li; Shark Liu; Xianzhen Luo; Yuyu Luo; Changzai Pan; Ensheng Shi; Yingshui Tan; Renshuai Tao; Jiajun Wu; Xianjie Wu; Zhenhe Wu; Daoguang Zan; Chenchen Zhang; Wei Zhang; He Zhu; Terry Yue Zhuo; Kerui Cao; Xianfu Cheng; Jun Dong; Shengjie Fang; Zhiwei Fei; Xiangyuan Guan; Qipeng Guo; Zhiguang Han; Joseph James; Tianqi Luo; Renyuan Li; Yuhang Li; Yiming Liang; Congnan Liu; Jiaheng Liu; Qian Liu; Ruitong Liu; Tyler Loakman; Xiangxin Meng; Chuang Peng; Tianhao Peng; Jiajun Shi; Mingjie Tang; Boyang Wang; Haowen Wang; Yunli Wang; Fanglin Xu; Zihan Xu; Fei Yuan; Ge Zhang; Jiayi Zhang; Xinhao Zhang; Wangchunshu Zhou; Hualei Zhu; King Zhu; Bryan Dai; Aishan Liu; Zhoujun Li; Chenghua Lin; Tianyu Liu; Chao Peng; Kai Shen; Libo Qin; Shuangyong Song; Zizheng Zhan; Jiajun Zhang; Jie Zhang; Zhaoxiang Zhang; Bo Zheng
>
> **摘要:** Large language models (LLMs) have fundamentally transformed automated software development by enabling direct translation of natural language descriptions into functional code, driving commercial adoption through tools like Github Copilot (Microsoft), Cursor (Anysphere), Trae (ByteDance), and Claude Code (Anthropic). While the field has evolved dramatically from rule-based systems to Transformer-based architectures, achieving performance improvements from single-digit to over 95\% success rates on benchmarks like HumanEval. In this work, we provide a comprehensive synthesis and practical guide (a series of analytic and probing experiments) about code LLMs, systematically examining the complete model life cycle from data curation to post-training through advanced prompting paradigms, code pre-training, supervised fine-tuning, reinforcement learning, and autonomous coding agents. We analyze the code capability of the general LLMs (GPT-4, Claude, LLaMA) and code-specialized LLMs (StarCoder, Code LLaMA, DeepSeek-Coder, and QwenCoder), critically examining the techniques, design decisions, and trade-offs. Further, we articulate the research-practice gap between academic research (e.g., benchmarks and tasks) and real-world deployment (e.g., software-related code tasks), including code correctness, security, contextual awareness of large codebases, and integration with development workflows, and map promising research directions to practical needs. Last, we conduct a series of experiments to provide a comprehensive analysis of code pre-training, supervised fine-tuning, and reinforcement learning, covering scaling law, framework selection, hyperparameter sensitivity, model architectures, and dataset comparisons.
>
---
#### [replaced 019] Animating Language Practice: Engagement with Stylized Conversational Agents in Japanese Learning
- **分类: cs.HC; cs.CL**

- **简介: 该论文研究面向日语学习的动漫风格对话代理系统Jouzu，旨在提升语言练习的参与度与自然性。通过集成大语言模型与拟人化角色，结合语音表达，增强学习者在真实场景中的互动体验。基于52名学习者的两周实地测试，发现该设计显著提升学习积极性，尤其促进高级学习者使用多样语体。研究揭示了风格化设计对情感投入与持续学习的关键作用，为文化相关对话AI在语言学习中的应用提供新思路。**

- **链接: [https://arxiv.org/pdf/2507.06483v2](https://arxiv.org/pdf/2507.06483v2)**

> **作者:** Zackary Rackauckas; Julia Hirschberg
>
> **摘要:** We explore Jouzu, a Japanese language learning application that integrates large language models with anime-inspired conversational agents. Designed to address challenges learners face in practicing natural and expressive dialogue, Jouzu combines stylized character personas with expressive text-to-speech to create engaging conversational scenarios. We conducted a two-week in-the-wild deployment with 52 Japanese learners to examine how such stylized agents influence engagement and learner experience. Our findings show that participants interacted frequently and creatively, with advanced learners demonstrating greater use of expressive forms. Participants reported that the anime-inspired style made practice more enjoyable and encouraged experimenting with different registers. We discuss how stylization shapes willingness to engage, the role of affect in sustaining practice, and design opportunities for culturally grounded conversational AI in computer-assisted language learning (CALL). By framing our findings as an exploration of design and engagement, we highlight opportunities for generalization beyond Japanese contexts and contribute to international HCI scholarship.
>
---
#### [replaced 020] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文针对大模型多学科复杂推理能力不足的问题，提出 DESIGNER 数据合成框架。通过提取12万+“设计逻辑”，基于书本与网络文本生成跨75个学科的高难度、多样化问题。构建DLR-Book和DLR-Web数据集，经微调后显著提升Qwen3与Llama3模型的多学科推理能力，甚至超越其官方优化版本。**

- **链接: [https://arxiv.org/pdf/2508.12726v5](https://arxiv.org/pdf/2508.12726v5)**

> **作者:** Weize Liu; Yongchi Zhao; Yijia Luo; Mingyu Xu; Jiaheng Liu; Yanan Li; Xiguo Hu; Zhiqi Bai; Yuchi Xu; Wenbo Su; Bo Zheng
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in many natural language tasks but still struggle with complex, multi-step reasoning, particularly across diverse disciplines. Existing reasoning datasets often lack disciplinary breadth, reasoning depth, and diversity, as well as guiding principles for question synthesis. We propose DESIGNER: a DESIGN-logic-guidEd Reasoning data synthesis pipeline that leverages naturally available, extensive raw documents (e.g., book corpus and web corpus) to generate multidisciplinary challenging questions. We introduce the concept of "design logic" and instruct LLMs to mimic human educators' question-creation process, enabling the automated synthesis of large-scale, high-difficulty questions. We use LLMs to reverse-engineer and abstract over 120,000 design logics from existing questions across various disciplines. By matching these design logics with source documents, we are able to generate reasoning questions with controllable question types and difficulty levels. Using this pipeline, we synthesized two large-scale reasoning datasets that span 75 disciplines: DLR-Book (3.04 million questions from the book corpus) and DLR-Web (1.66 million questions from the web corpus). Data analysis indicates that the questions synthesized by our method exhibit greater difficulty and diversity compared to those in the baseline datasets. We validate our synthesized data through supervised fine-tuning (SFT) on the Qwen3 and Llama3 model families. Our data substantially enhances their multidisciplinary reasoning capabilities, outperforming existing datasets. Notably, by applying SFT on the base versions of these models using only our data, we even surpass their official final models that have undergone the full post-training process.
>
---
#### [replaced 021] Just-in-time and distributed task representations in language models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型在上下文学习中任务表征的形成机制，聚焦“即时性”与“分布式”表征。针对新任务如何被编码的问题，提出并分析了可迁移的任务向量与任务类别表征。发现可迁移表征具非单调、局部性特征，仅在特定位置激活，而任务身份则持续存在。**

- **链接: [https://arxiv.org/pdf/2509.04466v3](https://arxiv.org/pdf/2509.04466v3)**

> **作者:** Yuxuan Li; Declan Campbell; Stephanie C. Y. Chan; Andrew Kyle Lampinen
>
> **摘要:** Many of language models' impressive capabilities originate from their in-context learning: based on instructions or examples, they can infer and perform new tasks without weight updates. In this work, we investigate when representations for new tasks are formed in language models, and how these representations change over the course of context. We study two different task representations: those that are ''transferrable'' -- vector representations that can transfer task contexts to another model instance, even without the full prompt -- and simpler representations of high-level task categories. We show that transferrable task representations evolve in non-monotonic and sporadic ways, while task identity representations persist throughout the context. Specifically, transferrable task representations exhibit a two-fold locality. They successfully condense evidence when more examples are provided in the context. But this evidence accrual process exhibits strong temporal locality along the sequence dimension, coming online only at certain tokens -- despite task identity being reliably decodable throughout the context. In some cases, transferrable task representations also show semantic locality, capturing a small task ''scope'' such as an independent subtask. Language models thus represent new tasks on the fly through both an inert, sustained sensitivity to the task and an active, just-in-time representation to support inference.
>
---
#### [replaced 022] SQLBarber: A System Leveraging Large Language Models to Generate Customized and Realistic SQL Workloads
- **分类: cs.DB; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SQLBarber系统，旨在生成定制化且符合真实场景的SQL工作负载。针对现有方法难以获取真实SQL、定制性差的问题，利用大语言模型与贝叶斯优化，实现自然语言驱动的模板生成与成本分布匹配，显著提升生成效率与真实性。**

- **链接: [https://arxiv.org/pdf/2507.06192v2](https://arxiv.org/pdf/2507.06192v2)**

> **作者:** Jiale Lao; Immanuel Trummer
>
> **备注:** Accepted by SIGMOD 2026; extended version with appendix
>
> **摘要:** Database research and development often require a large number of SQL queries for benchmarking purposes. However, acquiring real-world SQL queries is challenging due to privacy concerns, and existing SQL generation methods are limited in customization and in satisfying realistic constraints. To address this issue, we present SQLBarber, a system based on Large Language Models (LLMs) to generate customized and realistic SQL workloads. SQLBarber (i) eliminates the need for users to manually craft SQL templates in advance, while providing the flexibility to accept natural language specifications to constrain SQL templates, (ii) scales efficiently to generate large volumes of queries matching any user-defined cost distribution (e.g., cardinality and execution plan cost), and (iii) uses execution statistics from Amazon Redshift and Snowflake to derive SQL template specifications and query cost distributions that reflect real-world query characteristics. SQLBarber introduces (i) a declarative interface for users to effortlessly generate customized SQL templates, (ii) an LLM-powered pipeline augmented with a self-correction module that profiles, refines, and prunes SQL templates based on query costs, and (iii) a Bayesian Optimizer to efficiently explore different predicate values and identify a set of queries that satisfy the target cost distribution. We construct and open-source ten benchmarks of varying difficulty levels and target query cost distributions based on real-world statistics from Snowflake and Amazon Redshift. Extensive experiments on these benchmarks show that SQLBarber is the only system that can generate customized SQL templates. It reduces query generation time by one to three orders of magnitude, and significantly improves alignment with the target cost distribution, compared with existing methods.
>
---
#### [replaced 023] Evaluating LLMs on Sequential API Call Through Automated Test Generation
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文针对大模型调用API时缺乏有效评估的问题，提出StateGen框架，自动生成包含序列API调用的测试任务。通过状态机约束与采样技术生成可执行代码，并转化为自然语言任务描述，构建了包含120个验证用例的StateEval基准，用于评估和改进大模型在复杂API交互中的表现。**

- **链接: [https://arxiv.org/pdf/2507.09481v2](https://arxiv.org/pdf/2507.09481v2)**

> **作者:** Yuheng Huang; Jiayang Song; Da Song; Zhenlan Ji; Wenhan Wang; Shuai Wang; Lei Ma
>
> **摘要:** By integrating tools from external APIs, Large Language Models (LLMs) have expanded their promising capabilities in a diverse spectrum of complex real-world tasks. However, testing, evaluation, and analysis of LLM tool use remain in their early stages. Most existing benchmarks rely on manually collected test cases, many of which cannot be automatically checked for semantic correctness and instead depend on static methods such as string matching. Additionally, these benchmarks often overlook the complex interactions that occur between sequential API calls, which are common in real-world applications. To fill the gap, in this paper, we introduce StateGen, an automated framework designed to generate diverse coding tasks involving sequential API interactions. StateGen combines state-machine-based API constraint solving and validation, energy-based sampling, and control-flow injection to generate executable programs. These programs are then translated into human-like natural language task descriptions through a collaboration of two LLM agents. Utilizing StateGen, we construct StateEval, a benchmark encompassing 120 verified test cases spanning across three representative scenarios: Session Service, Tensor Operation, and ElevenLabs MCP. Experimental results confirm that StateGen can effectively generate challenging and realistic API-oriented tasks, highlighting areas for improvement in current LLMs incorporating APIs.We make our framework and benchmark publicly available to support future research.
>
---
#### [replaced 024] Tiny but Mighty: A Software-Hardware Co-Design Approach for Efficient Multimodal Inference on Battery-Powered Small Devices
- **分类: cs.DC; cs.AI; cs.CL; eess.SP**

- **简介: 该论文针对电池供电小设备上运行大型多模态模型（LMM）时资源浪费与高延迟问题，提出软硬件协同设计框架NANOMIND。通过将模型拆分为模块化“积木”，动态调度至最优加速器，结合低比特计算与智能内存管理，实现端侧高效推理，显著提升能效与吞吐量。**

- **链接: [https://arxiv.org/pdf/2510.05109v3](https://arxiv.org/pdf/2510.05109v3)**

> **作者:** Yilong Li; Shuai Zhang; Yijing Zeng; Hao Zhang; Xinmiao Xiong; Jingyu Liu; Pan Hu; Suman Banerjee
>
> **摘要:** Large Multimodal Models (LMMs) are inherently modular, consisting of vision and audio encoders, projectors, and large language models. Yet, they are almost always executed monolithically, which underutilizes the heterogeneous accelerators (NPUs, GPUs, DSPs) in modern SoCs and leads to high end-to-end latency. In this paper, we present NANOMIND, a hardware--software co-design inference framework for Large Multimodal Models (LMMs) that breaks large models into modular ``bricks'' (vision, language, audio, etc.) and maps each to its ideal accelerator. The key insight is that large models can be broken into modular components and scheduled to run on the most appropriate compute units. It performs module-level dynamic offloading across accelerators on unified-memory SoCs. By combining customized hardware design, system-level scheduling, and optimized low-bit computation kernels, we demonstrate our framework with a compact, battery-powered device capable of running LMMs entirely on device. This prototype functions as a self-contained intelligent assistant that requires no network connectivity, while achieving higher throughput and superior power efficiency under strict resource constraints. The design further bypasses CPU bottlenecks and reduces redundant memory usage through token-aware buffer management and module-level coordination. Our system outperforms existing implementations in resource efficiency, cutting energy consumption by 42.3\% and GPU memory usage by 11.2\%. This enables a battery-powered device to run LLaVA-OneVision with a camera for nearly 20.8 hours.
>
---
#### [replaced 025] H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文研究大语言模型中的幻觉现象，聚焦于幻觉相关神经元（H-Neurons）的识别、影响与起源。通过系统分析发现，极少数神经元（<0.1%）可精准预测幻觉，且其行为具有因果性，起源于预训练阶段。工作揭示了幻觉的微观神经机制，为提升模型可靠性提供新思路。**

- **链接: [https://arxiv.org/pdf/2512.01797v2](https://arxiv.org/pdf/2512.01797v2)**

> **作者:** Cheng Gao; Huimin Chen; Chaojun Xiao; Zhiyi Chen; Zhiyuan Liu; Maosong Sun
>
> **备注:** 20 pages, 4 figures
>
> **摘要:** Large language models (LLMs) frequently generate hallucinations -- plausible but factually incorrect outputs -- undermining their reliability. While prior work has examined hallucinations from macroscopic perspectives such as training data and objectives, the underlying neuron-level mechanisms remain largely unexplored. In this paper, we conduct a systematic investigation into hallucination-associated neurons (H-Neurons) in LLMs from three perspectives: identification, behavioral impact, and origins. Regarding their identification, we demonstrate that a remarkably sparse subset of neurons (less than $0.1\%$ of total neurons) can reliably predict hallucination occurrences, with strong generalization across diverse scenarios. In terms of behavioral impact, controlled interventions reveal that these neurons are causally linked to over-compliance behaviors. Concerning their origins, we trace these neurons back to the pre-trained base models and find that these neurons remain predictive for hallucination detection, indicating they emerge during pre-training. Our findings bridge macroscopic behavioral patterns with microscopic neural mechanisms, offering insights for developing more reliable LLMs.
>
---
#### [replaced 026] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MAS-ZERO，一种无需监督的多智能体系统自演化框架。针对现有MAS依赖人工设计、缺乏适应性与动态调整能力的问题，提出基于元级反馈的迭代设计机制，实现推理时动态分解任务、组合智能体并简化系统，显著提升复杂任务求解性能，且无需验证集。**

- **链接: [https://arxiv.org/pdf/2505.14996v3](https://arxiv.org/pdf/2505.14996v3)**

> **作者:** Zixuan Ke; Austin Xu; Yifei Ming; Xuan-Phi Nguyen; Ryan Chin; Caiming Xiong; Shafiq Joty
>
> **备注:** SEA@NeurIPS (Oral) 2025
>
> **摘要:** Multi-agent systems (MAS) leveraging the impressive capabilities of Large Language Models (LLMs) hold significant potential for tackling complex tasks. However, most current MAS depend on manually designed agent roles and communication protocols. These manual designs often fail to align with the underlying LLMs' strengths and struggle to adapt to novel tasks. Recent automatic MAS approaches attempt to mitigate these limitations but typically necessitate a validation set for tuning and yield static MAS designs lacking adaptability during inference, while also removing the flexibility to reduce to simpler systems. We introduce MAS-ZERO, the first self-evolved, inference-time framework for automatic MAS design. MAS-ZERO employs meta-level design to iteratively design, critique, and refine MAS configurations tailored to each problem instance, without requiring a validation set. Critically, it enables dynamic problem decomposition and agent composition through meta-feedback on solvability and completeness, and reduction to simpler systems when appropriate. Experiments across reasoning (math and graduate-level QA), coding, and agentic (search-based) benchmarks, using both closed-source and open-source LLM backbones of varying sizes, demonstrate that MAS-ZERO outperforms strong manual and automatic MAS baselines. It achieves substantial average accuracy improvements of up to 16.69% on reasoning, 16.66% on coding, and 5.45% on agentic tasks, while maintaining cost efficiency.
>
---
#### [replaced 027] Scaling Towards the Information Boundary of Instruction Set: InfinityInstruct-Subject Technical Report
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对指令数据集覆盖广度与深度不足的问题，提出闭环式框架，通过分层标注、种子选择、演化合成与缺陷诊断，构建了包含150万条指令的InfinityInstruct-Subject数据集，显著提升模型指令遵循能力与泛化性能。**

- **链接: [https://arxiv.org/pdf/2507.06968v2](https://arxiv.org/pdf/2507.06968v2)**

> **作者:** Li Du; Hanyu Zhao; Yiming Ju; Tengfei Pan
>
> **摘要:** Instruction tuning has become a foundation for unlocking the capabilities of large-scale pretrained models and improving their performance on complex tasks. Thus, the construction of high-quality instruction datasets is crucial for enhancing model performance and generalizability. Although current instruction datasets have reached tens of millions of samples, models finetuned on them may still struggle with complex instruction following and tasks in rare domains. This is primarily due to limited expansion in both ``coverage'' (coverage of task types and knowledge areas) and ``depth'' (instruction complexity) of the instruction set. To address this issue, we propose a systematic instruction data construction framework, which integrates a hierarchical labeling system, an informative seed selection algorithm, an evolutionary data synthesis process, and a model deficiency diagnosis with targeted data generation. These components form an iterative closed-loop to continuously enhance the coverage and depth of instruction data. Based on this framework, we construct InfinityInstruct-Subject, a high-quality dataset containing ~1.5 million instructions. Experiments on multiple foundation models and benchmark tasks demonstrate its effectiveness in improving instruction-following capabilities. Further analyses suggest that InfinityInstruct-Subject shows enlarged coverage and depth compared to comparable synthesized instruction datasets. Our work lays a theoretical and practical foundation for the efficient, continuous evolution of instruction datasets, moving from data quantity expansion to qualitative improvement.
>
---
#### [replaced 028] OpenGVL -- Benchmarking Visual Temporal Progress for Data Curation
- **分类: cs.RO; cs.CL**

- **简介: 该论文提出OpenGVL，一个用于评估视觉时间进度的基准，解决机器人数据稀缺与大规模数据标注难题。通过对比开源与闭源模型在任务进度预测中的表现，揭示开源模型显著落后，并展示其在自动化数据筛选中的应用价值。**

- **链接: [https://arxiv.org/pdf/2509.17321v3](https://arxiv.org/pdf/2509.17321v3)**

> **作者:** Paweł Budzianowski; Emilia Wiśnios; Gracjan Góral; Igor Kulakov; Viktor Petrenko; Krzysztof Walas
>
> **备注:** Workshop on Making Sense of Data in Robotics: Composition, Curation, and Interpretability at Scale at CoRL 2025
>
> **摘要:** Data scarcity remains one of the most limiting factors in driving progress in robotics. However, the amount of available robotics data in the wild is growing exponentially, creating new opportunities for large-scale data utilization. Reliable temporal task completion prediction could help automatically annotate and curate this data at scale. The Generative Value Learning (GVL) approach was recently proposed, leveraging the knowledge embedded in vision-language models (VLMs) to predict task progress from visual observations. Building upon GVL, we propose OpenGVL, a comprehensive benchmark for estimating task progress across diverse challenging manipulation tasks involving both robotic and human embodiments. We evaluate the capabilities of publicly available open-source foundation models, showing that open-source model families significantly underperform closed-source counterparts, achieving only approximately $70\%$ of their performance on temporal progress prediction tasks. Furthermore, we demonstrate how OpenGVL can serve as a practical tool for automated data curation and filtering, enabling efficient quality assessment of large-scale robotics datasets. We release the benchmark along with the complete codebase at \href{github.com/budzianowski/opengvl}{OpenGVL}.
>
---
#### [replaced 029] Kardia-R1: Unleashing LLMs to Reason toward Understanding and Empathy for Emotional Support via Rubric-as-Judge Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦于情感支持对话任务，针对现有系统缺乏用户身份感知和透明奖励机制的问题，构建了大规模用户锚定数据集KardiaBench，并提出Kardia-R1框架，通过可解释的评分标准强化学习实现可追溯的情感推理与共情生成，显著提升模型在情感准确性、共情能力等方面的表现。**

- **链接: [https://arxiv.org/pdf/2512.01282v2](https://arxiv.org/pdf/2512.01282v2)**

> **作者:** Jiahao Yuan; Zhiqing Cui; Hanqing Wang; Yuansheng Gao; Yucheng Zhou; Usman Naseem
>
> **摘要:** As web platforms evolve towards greater personalization and emotional complexity, conversational agents must transcend superficial empathy to demonstrate identity-aware emotional reasoning. However, existing systems face two limitations: (1) reliance on situation-centric datasets lacking persistent user identity, which hampers the capture of personalized affective nuances; and (2) dependence on opaque, coarse reward signals that hinder development of verifiable empathetic reasoning. To address these gaps, we introduce KardiaBench, a large-scale user-grounded benchmark comprising 178,080 QA pairs across 22,080 multi-turn conversations anchored to 671 real-world profiles. The dataset is constructed via a model-in-the-loop pipeline with iterative rubric-guided refinement to ensure psychological plausibility and persona consistency. This progressive empathy pipeline that integrates user comprehension, contextual reasoning, and emotion perception into conversations, followed by iterative critique and rubric-based refinement to ensure psychological plausibility, emotional fidelity, and persona consistency. Building on this, we propose Kardia-R1, a framework that trains models for interpretable, stepwise empathetic cognition. Kardia-R1 leverages Rubric-as-Judge Empathetic Reinforcement Learning (Rubric-ERL), a GRPO-based method that uses explainable, human-aligned rubric rewards to tightly couple user understanding, emotional inference, and supportive response generation. Extensive experiments across four LLM backbones demonstrate that Kardia-R1 consistently outperforms othet methods in emotion accuracy, empathy, relevance, persona consistency, and safety. Our dataset and model will be released at https://github.com/JhCircle/Kardia-R1.
>
---
#### [replaced 030] On SkipGram Word Embedding Models with Negative Sampling: Unified Framework and Impact of Noise Distributions
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究词嵌入任务，针对SkipGram负采样模型（SGN）的局限性，提出统一的“词-上下文分类”（WCC）框架，通过理论分析与实验验证，发现使用数据分布作为噪声分布能提升嵌入性能与收敛速度，并由此发现若干优于现有模型的新方法。**

- **链接: [https://arxiv.org/pdf/2009.04413v2](https://arxiv.org/pdf/2009.04413v2)**

> **作者:** Dezhi Liu; Richong Zhang; Ziqiao Wang
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** SkipGram word embedding models with negative sampling, or SGN in short, is an elegant family of word embedding models. In this paper, we formulate a framework for word embedding, referred to as Word-Context Classification (WCC), that generalizes SGN to a wide family of models. The framework, which uses some ``noise examples'', is justified through theoretical analysis. The impact of noise distribution on the learning of the WCC embedding models is studied experimentally, suggesting that the best noise distribution is, in fact, the data distribution, in terms of both the embedding performance and the speed of convergence during training. Along our way, we discover several novel embedding models that outperform existing WCC models.
>
---
#### [replaced 031] HLPD: Aligning LLMs to Human Language Preference for Machine-Revised Text Detection
- **分类: cs.CL; cs.CR**

- **简介: 该论文针对机器修订文本检测难题，提出HLPD方法。基于人类语言偏好假设，通过奖励对齐优化模型，提升对人类写作风格的敏感度，有效识别经高级LLM修订的文本，在黑盒环境下显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.06942v3](https://arxiv.org/pdf/2511.06942v3)**

> **作者:** Fangqi Dai; Xingjian Jiang; Zizhuang Deng
>
> **备注:** 20 pages, 10 figures, accepted by AAAI'26
>
> **摘要:** To prevent misinformation and social issues arising from trustworthy-looking content generated by LLMs, it is crucial to develop efficient and reliable methods for identifying the source of texts. Previous approaches have demonstrated exceptional performance in detecting texts fully generated by LLMs. However, these methods struggle when confronting more advanced LLM output or text with adversarial multi-task machine revision, especially in the black-box setting, where the generating model is unknown. To address this challenge, grounded in the hypothesis that human writing possesses distinctive stylistic patterns, we propose Human Language Preference Detection (HLPD). HLPD employs a reward-based alignment process, Human Language Preference Optimization (HLPO), to shift the scoring model's token distribution toward human-like writing, making the model more sensitive to human writing, therefore enhancing the identification of machine-revised text. We test HLPD in an adversarial multi-task evaluation framework that leverages a five-dimensional prompt generator and multiple advanced LLMs to create diverse revision scenarios. When detecting texts revised by GPT-series models, HLPD achieves a 15.11% relative improvement in AUROC over ImBD, surpassing Fast-DetectGPT by 45.56%. When evaluated on texts generated by advanced LLMs, HLPD achieves the highest average AUROC, exceeding ImBD by 5.53% and Fast-DetectGPT by 34.14%. Code will be made available at https://github.com/dfq2021/HLPD.
>
---
#### [replaced 032] Reasoning Up the Instruction Ladder for Controllable Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中的指令优先级问题，旨在解决多源指令冲突时模型的可控性与可靠性。通过构建VerIH数据集并采用轻量强化学习，将指令层级解析转化为推理任务，提升模型对高优先级指令的遵循能力，增强安全性和抗攻击性。**

- **链接: [https://arxiv.org/pdf/2511.04694v3](https://arxiv.org/pdf/2511.04694v3)**

> **作者:** Zishuo Zheng; Vidhisha Balachandran; Chan Young Park; Faeze Brahman; Sachin Kumar
>
> **摘要:** As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises ~7K aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks, achieving roughly a 20% improvement on the IHEval conflict setup. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks, providing up to a 20% reduction in attack success rate (ASR). These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior.
>
---
#### [replaced 033] Multilingual Pretraining for Pixel Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言预训练在像素语言模型中的应用。针对像素语言模型在非拉丁语系上表现不足的问题，提出PIXEL-M4模型，在英、印、乌、中四语言图像上预训练。实验证明其在跨语言任务中优于单语模型，能有效捕捉多语言语义特征，提升对多样语言的支持能力。**

- **链接: [https://arxiv.org/pdf/2505.21265v2](https://arxiv.org/pdf/2505.21265v2)**

> **作者:** Ilker Kesen; Jonas F. Lotz; Ingo Ziegler; Phillip Rust; Desmond Elliott
>
> **备注:** EMNLP 2025
>
> **摘要:** Pixel language models operate directly on images of rendered text, eliminating the need for a fixed vocabulary. While these models have demonstrated strong capabilities for downstream cross-lingual transfer, multilingual pretraining remains underexplored. We introduce PIXEL-M4, a model pretrained on four visually and linguistically diverse languages: English, Hindi, Ukrainian, and Simplified Chinese. Multilingual evaluations on semantic and syntactic tasks show that PIXEL-M4 outperforms an English-only counterpart on non-Latin scripts. Word-level probing analyses confirm that PIXEL-M4 captures rich linguistic features, even in languages not seen during pretraining. Furthermore, an analysis of its hidden representations shows that multilingual pretraining yields a semantic embedding space closely aligned across the languages used for pretraining. This work demonstrates that multilingual pretraining substantially enhances the capability of pixel language models to effectively support a diverse set of languages.
>
---
#### [replaced 034] Geometric Uncertainty for Detecting and Correcting Hallucinations in LLMs
- **分类: cs.CL**

- **简介: 该论文针对大语言模型（LLM）幻觉问题，提出基于几何框架的不确定性量化方法。通过响应嵌入的原型分析，引入全局“几何体积”和局部“几何怀疑度”，实现对幻觉的检测与纠正。该方法在医疗等高风险任务中表现优异，具有理论支持的连续语义边界，优于现有离散比较方法。**

- **链接: [https://arxiv.org/pdf/2509.13813v2](https://arxiv.org/pdf/2509.13813v2)**

> **作者:** Edward Phillips; Sean Wu; Soheila Molaei; Danielle Belgrave; Anshul Thakur; David Clifton
>
> **备注:** Revision. Clarified positioning as a unified geometric framework for global and local uncertainty in LLMs. Added baselines (Degree, Eccentricity) and expanded comparison to related methods. Included ablations (PCA dimension, number of archetypes, number of samples) and complexity analysis. Extended discussion of medical QA results and model-specific behaviour
>
> **摘要:** Large language models demonstrate impressive results across diverse tasks but are still known to hallucinate, generating linguistically plausible but incorrect answers to questions. Uncertainty quantification has been proposed as a strategy for hallucination detection, requiring estimates for both global uncertainty (attributed to a batch of responses) and local uncertainty (attributed to individual responses). While recent black-box approaches have shown some success, they often rely on disjoint heuristics or graph-theoretic approximations that lack a unified geometric interpretation. We introduce a geometric framework to address this, based on archetypal analysis of batches of responses sampled with only black-box model access. At the global level, we propose Geometric Volume, which measures the convex hull volume of archetypes derived from response embeddings. At the local level, we propose Geometric Suspicion, which leverages the spatial relationship between responses and these archetypes to rank reliability, enabling hallucination reduction through preferential response selection. Unlike prior methods that rely on discrete pairwise comparisons, our approach provides continuous semantic boundary points which have utility for attributing reliability to individual responses. Experiments show that our framework performs comparably to or better than prior methods on short form question-answering datasets, and achieves superior results on medical datasets where hallucinations carry particularly critical risks. We also provide theoretical justification by proving a link between convex hull volume and entropy.
>
---
#### [replaced 035] The Necessity of Imperfection:Reversing Model Collapse via Simulating Cognitive Boundedness
- **分类: cs.AI; cs.CL; cs.CY; cs.LG; q-fin.TR**

- **简介: 该论文针对合成数据导致模型崩溃的问题，提出模拟人类认知局限的PMCSF框架。通过认知状态解码与扰动生成，使合成文本具备人类典型不完美特征，在认知相似性与金融策略性能上显著优于传统方法，解决了合成数据缺乏认知真实性的问题。**

- **链接: [https://arxiv.org/pdf/2512.01354v2](https://arxiv.org/pdf/2512.01354v2)**

> **作者:** Zhongjie Jiang
>
> **备注:** 38 pages,5 figures. Extended technical disclosure (Version 2.0) is attached as ancillary files, containing raw forensic logs of the "Silent Rupture"detection [May 2025], proprietary GARCH parameter ranges, and the linguistic micro-chaos injection protocols
>
> **摘要:** Although synthetic data is widely promoted as a remedy, its prevailing production paradigm -- one optimizing for statistical smoothness -- systematically removes the long-tail, cognitively grounded irregularities that characterize human text. Prolonged training on such statistically optimal but cognitively impoverished data accelerates model collapse. This paper proposes a paradigm shift: instead of imitating the surface properties of data, we simulate the cognitive processes that generate human text. We introduce the Prompt-driven Cognitive Computing Framework (PMCSF), whose core consists of a Cognitive State Decoder (CSD) that reverse-engineers unstructured text into structured cognitive vectors, and a Cognitive Text Encoder (CTE) that re-materializes these states into text enriched with human-typical imperfections via mathematically defined Cognitive Perturbation Operators. The framework is validated through a two-stage objective evaluation pipeline. First, in cognitive codec verification, CTE text yields a Jensen-Shannon divergence of 0.0614 from human text (vs. 0.4431 for standard LLM output), passes double-blind professional media review, and achieves an intraclass correlation coefficient ICC > 0.9 for cognitive profile alignment across heterogeneous models. Second, in functional gain evaluation, isomorphic stress tests in the A-share market show that strategies incorporating CTE-generated data reduce maximum drawdown by 47.4% during the 2015 crash and deliver 8.6% Defensive Alpha, exceeding transaction costs by a factor of 33. Our findings demonstrate that modelling human cognitive limitations -- not copying surface data -- enables synthetic data with genuine functional gain, offering a viable technical pathway toward resolving the AI data-collapse crisis.
>
---
#### [replaced 036] Facilitating Long Context Understanding via Supervised Chain-of-Thought Reasoning
- **分类: cs.CL**

- **简介: 该论文针对长文本理解中模型推理能力不足的问题，提出基于监督式思维链（CoT）的解决方案。通过构建金融领域合成数据集LongFinanceQA及属性驱动的代理推理框架PAI，增强模型在长上下文中的显式推理能力。实验表明，该方法显著提升了模型在长文本任务上的准确率与可解释性。**

- **链接: [https://arxiv.org/pdf/2502.13127v2](https://arxiv.org/pdf/2502.13127v2)**

> **作者:** Jingyang Lin; Andy Wong; Tian Xia; Shenghua He; Hui Wei; Mei Han; Jiebo Luo
>
> **备注:** Main Conference of EMNLP 2025, Project Page: https://long-pai.github.io/
>
> **摘要:** Recent advances in Large Language Models (LLMs) have enabled them to process increasingly longer sequences, ranging from 2K to 2M tokens and even beyond. However, simply extending the input sequence length does not necessarily lead to effective long-context understanding. In this study, we integrate Chain-of-Thought (CoT) reasoning into LLMs in a supervised manner to facilitate effective long-context understanding. To achieve this, we introduce LongFinanceQA, a synthetic dataset in the financial domain designed to improve long-context reasoning. Unlike existing long-context synthetic data, LongFinanceQA includes intermediate CoT reasoning before the final conclusion, which encourages LLMs to perform explicit reasoning, improving accuracy and interpretability in long-context understanding. To generate synthetic CoT reasoning, we propose Property-based Agentic Inference (PAI), an agentic framework that simulates human-like reasoning steps, including property extraction, retrieval, and summarization. We evaluate PAI's reasoning capabilities by assessing GPT-4o-mini w/ PAI on the Loong benchmark, outperforming standard GPT-4o-mini by 20.0%. Furthermore, we fine-tune LLaMA-3.1-8B-Instruct on LongFinanceQA, achieving a 28.0% gain on Loong's financial subset.
>
---
#### [replaced 037] Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型中的重复诅咒问题，指出诱导头的“毒性”是导致重复生成的机制根源。通过分析注意力头的作用，提出用注意力头正则化来削弱诱导头的主导性，从而提升生成多样性与连贯性。属于自然语言生成优化任务。**

- **链接: [https://arxiv.org/pdf/2505.13514v2](https://arxiv.org/pdf/2505.13514v2)**

> **作者:** Shuxun Wang; Qingyu Yin; Chak Tou Leong; Qiang Zhang; Linyi Yang
>
> **备注:** Need to be refined
>
> **摘要:** Repetition curse is a phenomenon where Large Language Models (LLMs) generate repetitive sequences of tokens or cyclic sequences. While the repetition curse has been widely observed, its underlying mechanisms remain poorly understood. In this work, we investigate the role of induction heads--a specific type of attention head known for their ability to perform in-context learning--in driving this repetitive behavior. Specifically, we focus on the "toxicity" of induction heads, which we define as their tendency to dominate the model's output logits during repetition, effectively excluding other attention heads from contributing to the generation process. Our findings have important implications for the design and training of LLMs. By identifying induction heads as a key driver of the repetition curse, we provide a mechanistic explanation for this phenomenon and suggest potential avenues for mitigation. We also propose a technique with attention head regularization that could be employed to reduce the dominance of induction heads during generation, thereby promoting more diverse and coherent outputs.
>
---
#### [replaced 038] iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文针对大语言模型推理中多智能体辩论（MAD）效率低、误判风险高的问题，提出iMAD框架。通过分析单代理的自省式批判文本，提取41个语义特征，利用轻量级分类器与焦点校准损失，智能判断是否触发辩论，实现高效准确推理，在六组视觉问答任务上显著降低92%的令牌消耗并提升13.5%准确率。**

- **链接: [https://arxiv.org/pdf/2511.11306v2](https://arxiv.org/pdf/2511.11306v2)**

> **作者:** Wei Fan; JinYi Yoon; Bo Ji
>
> **备注:** Accepted in AAAI 2026 (Oral)
>
> **摘要:** Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%).
>
---
#### [replaced 039] CT-GLIP: 3D Grounded Language-Image Pretraining with CT Scans and Radiology Reports for Full-Body Scenarios
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出CT-GLIP，一种用于全身体部CT扫描的3D医学视觉语言预训练模型。针对现有方法因全局对齐导致关键细节丢失的问题，通过构建细粒度的CT-报告配对，实现基于语义的跨模态对比学习，显著提升器官与病灶的零样本识别与分割性能。**

- **链接: [https://arxiv.org/pdf/2404.15272v4](https://arxiv.org/pdf/2404.15272v4)**

> **作者:** Jingyang Lin; Yingda Xia; Jianpeng Zhang; Ke Yan; Kai Cao; Le Lu; Jiebo Luo; Ling Zhang
>
> **摘要:** 3D medical vision-language (VL) pretraining has shown potential in radiology by leveraging large-scale multimodal datasets with CT-report pairs. However, existing methods primarily rely on a global VL alignment directly adapted from 2D scenarios. The entire 3D image is transformed into one global embedding, resulting in a loss of sparse but critical semantics essential for accurately aligning with the corresponding diagnosis. To address this limitation, we propose CT-GLIP, a 3D Grounded Language-Image Pretrained model that constructs fine-grained CT-report pairs to enhance \textit{grounded} cross-modal contrastive learning, effectively aligning grounded visual features with precise textual descriptions. Leveraging the grounded cross-modal alignment, CT-GLIP improves performance across diverse downstream tasks and can even identify organs and abnormalities in a zero-shot manner using natural language. CT-GLIP is trained on a multimodal CT dataset comprising 44,011 organ-level CT-report pairs from 17,702 patients, covering 104 organs. Evaluation is conducted on four downstream tasks: zero-shot organ recognition (OR), zero-shot abnormality detection (AD), tumor detection (TD), and tumor segmentation (TS). Empirical results show that it outperforms its counterparts with global VL alignment. Compared to vanilla CLIP, CT-GLIP achieves average performance improvements of 15.1% of F1 score, 1.9% of AUC, and 3.2% of DSC for zero-shot AD, TD, and TS tasks, respectively. This study highlights the significance of grounded VL alignment in enabling 3D medical VL foundation models to understand sparse representations within CT scans.
>
---
#### [replaced 040] A Reproducible Framework for Neural Topic Modeling in Focus Group Analysis
- **分类: cs.CL; cs.HC; cs.LG**

- **简介: 该论文针对焦点小组文本分析中人工编码效率低、可复现性差的问题，提出基于BERTopic的可复现建模框架。通过系统调参与多指标评估，构建了7主题模型，显著提升主题质量与可解释性，验证了其在小规模语料中的有效性，强调需多维度评估模型质量。**

- **链接: [https://arxiv.org/pdf/2511.18843v2](https://arxiv.org/pdf/2511.18843v2)**

> **作者:** Heger Arfaoui; Mohammed Iheb Hergli; Beya Benzina; Slimane BenMiled
>
> **摘要:** Focus group discussions generate rich qualitative data but their analysis traditionally relies on labor-intensive manual coding that limits scalability and reproducibility. We present a systematic framework for applying BERTopic to focus group transcripts using data from ten focus groups exploring HPV vaccine perceptions in Tunisia (1,075 utterances). We conducted comprehensive hyperparameter exploration across 27 configurations, evaluating each through bootstrap stability analysis, performance metrics, and comparison with LDA baseline. Bootstrap analysis revealed that stability metrics (NMI and ARI) exhibited strong disagreement (r = -0.691) and showed divergent relationships with coherence, demonstrating that stability is multifaceted rather than monolithic. Our multi-criteria selection framework yielded a 7-topic model achieving 18\% higher coherence than optimized LDA (0.573 vs. 0.486) with interpretable topics validated through independent human evaluation (ICC = 0.700, weighted Cohen's kappa = 0.678). These findings demonstrate that transformer-based topic modeling can extract interpretable themes from small focus group transcript corpora when systematically configured and validated, while revealing that quality metrics capture distinct, sometimes conflicting constructs requiring multi-criteria evaluation. We provide complete documentation and code to support reproducibility.
>
---
#### [replaced 041] HeavyWater and SimplexWater: Distortion-Free LLM Watermarks for Low-Entropy Next-Token Predictions
- **分类: cs.CR; cs.AI; cs.CL; cs.CY; cs.IT; cs.LG**

- **简介: 该论文研究大语言模型文本水印技术，针对低熵生成任务（如编程）中水印易失真问题，提出HeavyWater与SimplexWater两种无失真水印方法。通过优化框架，实现高检测率与低文本质量损失的平衡，适用于任意LLM且与侧信息生成方式无关。**

- **链接: [https://arxiv.org/pdf/2506.06409v2](https://arxiv.org/pdf/2506.06409v2)**

> **作者:** Dor Tsur; Carol Xuan Long; Claudio Mayrink Verdun; Hsiang Hsu; Chen-Fu Chen; Haim Permuter; Sajani Vithana; Flavio P. Calmon
>
> **备注:** Presented at NeurIPS2025
>
> **摘要:** Large language model (LLM) watermarks enable authentication of text provenance, curb misuse of machine-generated text, and promote trust in AI systems. Current watermarks operate by changing the next-token predictions output by an LLM. The updated (i.e., watermarked) predictions depend on random side information produced, for example, by hashing previously generated tokens. LLM watermarking is particularly challenging in low-entropy generation tasks -- such as coding -- where next-token predictions are near-deterministic. In this paper, we propose an optimization framework for watermark design. Our goal is to understand how to most effectively use random side information in order to maximize the likelihood of watermark detection and minimize the distortion of generated text. Our analysis informs the design of two new watermarks: HeavyWater and SimplexWater. Both watermarks are tunable, gracefully trading-off between detection accuracy and text distortion. They can also be applied to any LLM and are agnostic to side information generation. We examine the performance of HeavyWater and SimplexWater through several benchmarks, demonstrating that they can achieve high watermark detection accuracy with minimal compromise of text generation quality, particularly in the low-entropy regime. Our theoretical analysis also reveals surprising new connections between LLM watermarking and coding theory. The code implementation can be found in https://github.com/DorTsur/HeavyWater_SimplexWater
>
---
#### [replaced 042] BountyBench: Dollar Impact of AI Agent Attackers and Defenders on Real-World Cybersecurity Systems
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出BountyBench框架，评估AI代理在真实系统中攻防能力。针对漏洞生命周期，设计检测、利用、修复三类任务，构建25个含40个真实漏洞的系统，量化攻击者与防御者对网络安全的经济影响，评估10个AI代理，揭示其攻防能力差异。**

- **链接: [https://arxiv.org/pdf/2505.15216v3](https://arxiv.org/pdf/2505.15216v3)**

> **作者:** Andy K. Zhang; Joey Ji; Celeste Menders; Riya Dulepet; Thomas Qin; Ron Y. Wang; Junrong Wu; Kyleen Liao; Jiliang Li; Jinghan Hu; Sara Hong; Nardos Demilew; Shivatmica Murgai; Jason Tran; Nishka Kacheria; Ethan Ho; Denis Liu; Lauren McLane; Olivia Bruvik; Dai-Rong Han; Seungwoo Kim; Akhil Vyas; Cuiyuanxiu Chen; Ryan Li; Weiran Xu; Jonathan Z. Ye; Prerit Choudhary; Siddharth M. Bhatia; Vikram Sivashankar; Yuxuan Bao; Dawn Song; Dan Boneh; Daniel E. Ho; Percy Liang
>
> **备注:** 113 pages
>
> **摘要:** AI agents have the potential to significantly alter the cybersecurity landscape. Here, we introduce the first framework to capture offensive and defensive cyber-capabilities in evolving real-world systems. Instantiating this framework with BountyBench, we set up 25 systems with complex, real-world codebases. To capture the vulnerability lifecycle, we define three task types: Detect (detecting a new vulnerability), Exploit (exploiting a given vulnerability), and Patch (patching a given vulnerability). For Detect, we construct a new success indicator, which is general across vulnerability types and provides localized evaluation. We manually set up the environment for each system, including installing packages, setting up server(s), and hydrating database(s). We add 40 bug bounties, which are vulnerabilities with monetary awards from \$10 to \$30,485, covering 9 of the OWASP Top 10 Risks. To modulate task difficulty, we devise a new strategy based on information to guide detection, interpolating from identifying a zero day to exploiting a given vulnerability. We evaluate 10 agents: Claude Code, OpenAI Codex CLI with o3-high and o4-mini, and custom agents with o3-high, GPT-4.1, Gemini 2.5 Pro Preview, Claude 3.7 Sonnet Thinking, Qwen3 235B A22B, Llama 4 Maverick, and DeepSeek-R1. Given up to three attempts, the top-performing agents are Codex CLI: o3-high (12.5% on Detect, mapping to \$3,720; 90% on Patch, mapping to \$14,152), Custom Agent: Claude 3.7 Sonnet Thinking (67.5% on Exploit), and Codex CLI: o4-mini (90% on Patch, mapping to \$14,422). Codex CLI: o3-high, Codex CLI: o4-mini, and Claude Code are more capable at defense, achieving higher Patch scores of 90%, 90%, and 87.5%, compared to Exploit scores of 47.5%, 32.5%, and 57.5% respectively; while the custom agents are relatively balanced between offense and defense, achieving Exploit scores of 17.5-67.5% and Patch scores of 25-60%.
>
---
#### [replaced 043] On the Temporal Question-Answering Capabilities of Large Language Models Over Anonymized Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型在匿名化数据上的时间推理能力，聚焦自然语言中的17类时间推理任务。针对训练外数据的推理难题，构建RATA数据集并对比多种先进方法，发现单一LLM不足，需集成策略以实现可靠、可扩展的时序问答。**

- **链接: [https://arxiv.org/pdf/2504.07646v2](https://arxiv.org/pdf/2504.07646v2)**

> **作者:** Alfredo Garrachón Ruiz; Tomás de la Rosa; Daniel Borrajo
>
> **备注:** 18 pages, 7 tables, 5 figures
>
> **摘要:** The applicability of Large Language Models (LLMs) in temporal reasoning tasks over data that is not present during training is still a field that remains to be explored. In this paper we work on this topic, focusing on structured and semi-structured anonymized data. We not only develop a direct LLM pipeline, but also compare various methodologies and conduct an in-depth analysis. We identified and examined seventeen common temporal reasoning tasks in natural language, focusing on their algorithmic components. To assess LLM performance, we created the \textit{Reasoning and Answering Temporal Ability} dataset (RATA), featuring semi-structured anonymized data to ensure reliance on reasoning rather than on prior knowledge. We compared several methodologies, involving SoTA techniques such as Tree-of-Thought, self-reflexion and code execution, tuned specifically for this scenario. Our results suggest that achieving scalable and reliable solutions requires more than just standalone LLMs, highlighting the need for integrated approaches.
>
---
#### [replaced 044] ParlAI Vote: A Web Platform for Analyzing Gender and Political Bias in Large Language Models
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文提出ParlAI Vote平台，用于分析大语言模型在欧洲议会辩论与投票中的性别和政治偏见。通过整合辩论、演讲与投票数据，实现投票预测与偏差可视化，揭示LLMs在不同人口统计群体上的系统性性能偏差，支持研究、教育与公众参与。**

- **链接: [https://arxiv.org/pdf/2509.16264v3](https://arxiv.org/pdf/2509.16264v3)**

> **作者:** Wenjie Lin; Hange Liu; Yingying Zhuang; Xutao Mao; Jingwei Shi; Xudong Han; Tianyu Shi; Jinrui Yang
>
> **备注:** online demo: https://euro-parl-vote-demo.vercel.app/; Video: https://www.youtube.com/@Jinrui-sf2jg
>
> **摘要:** We present ParlAI Vote, an interactive web platform for exploring European Parliament debates and votes, and for testing LLMs on vote prediction and bias analysis. This web system connects debate topics, speeches, and roll-call outcomes, and includes rich demographic data such as gender, age, country, and political group. Users can browse debates, inspect linked speeches, compare real voting outcomes with predictions from frontier LLMs, and view error breakdowns by demographic group. Visualizing the EuroParlVote benchmark and its core tasks of gender classification and vote prediction, ParlAI Vote highlights systematic performance bias in state-of-the-art LLMs. It unifies data, models, and visual analytics in a single interface, lowering the barrier for reproducing findings, auditing behavior, and running counterfactual scenarios. This web platform also shows model reasoning, helping users see why errors occur and what cues the models rely on. It supports research, education, and public engagement with legislative decision-making, while making clear both the strengths and the limitations of current LLMs in political analysis.
>
---
#### [replaced 045] From Atomic to Composite: Reinforcement Learning Enables Generalization in Complementary Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究强化学习（RL）在复杂推理中的作用，针对“RL是增强已有行为还是合成新技能”这一问题。通过分解互补推理任务为参数与上下文两种原子技能，发现仅靠监督微调（SFT）无法实现泛化，而RL在掌握原子技能后可有效合成复杂策略，突破零样本泛化瓶颈，提出分阶段训练范式以实现可扩展的通用推理。**

- **链接: [https://arxiv.org/pdf/2512.01970v2](https://arxiv.org/pdf/2512.01970v2)**

> **作者:** Sitao Cheng; Xunjian Yin; Ruiwen Zhou; Yuxuan Li; Xinyi Wang; Liangming Pan; William Yang Wang; Victor Zhong
>
> **备注:** Work in Progress. Code and data will be available at https://github.com/sitaocheng/from_atomic_to_composite
>
> **摘要:** The mechanism by which RL contributes to reasoning capabilities-whether it incentivizes the synthesis of new skills or merely amplifies existing behaviors-remains a subject of intense debate. In this work, we investigate this question through the lens of Complementary Reasoning, a complex task that requires integrating internal parametric knowledge with external contextual information. Using a controlled synthetic dataset of human biographies, we strictly decouple this ability into two atomic skills: Parametric Reasoning (relying on internal knowledge) and Contextual Reasoning (depending on external information). To rigorously assess capability boundaries, we evaluate generalization across three distinct levels of difficulty: I.I.D., Composition, and Zero-shot settings. We find that while SFT is sufficient for in-distribution performance, it struggles with O.O.D. generalization, particularly in Zero-shot settings where relational combinations are novel. Crucially, we identify the SFT Generalization Paradox: Models supervised solely on the composite task achieve near-perfect in-distribution accuracy but collapse on out-of-distribution generalization, indicating their reliance on rote memorization of path shortcuts. In contrast, we find that RL acts as a reasoning synthesizer rather than a probability amplifier. However, we uncover a strict atomic prerequisite: RL can only synthesize these complex strategies if the base model has first mastered the independent atomic skills (Parametric and Contextual) via SFT. These findings challenge the view of RL as a mere amplifier, suggesting that given sufficient atomic foundations, RL can actively synthesize complex reasoning strategies from learned primitives without explicit supervision on such complex strategies. This indicates that decoupled atomic training followed by RL offers a scalable path to generalization for complex reasoning tasks.
>
---
#### [replaced 046] Mechanisms of Symbol Processing for In-Context Learning in Transformer Networks
- **分类: cs.AI; cs.CL; cs.NE; cs.SC**

- **简介: 该论文研究Transformer网络在上下文学习中进行符号处理的机制，针对其能否实现抽象符号操作这一核心问题。通过构建可解释的生产系统语言PSL和编译器，将符号程序精确映射至Transformer，验证其计算能力（图灵完备），揭示了其在符号处理中的潜力与局限，但未涉及学习能力。**

- **链接: [https://arxiv.org/pdf/2410.17498v2](https://arxiv.org/pdf/2410.17498v2)**

> **作者:** Paul Smolensky; Roland Fernandez; Zhenghao Herbert Zhou; Mattia Opper; Adam Davies; Jianfeng Gao
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive abilities in symbol processing through in-context learning (ICL). This success flies in the face of decades of critiques asserting that artificial neural networks cannot master abstract symbol manipulation. We seek to understand the mechanisms that can enable robust symbol processing in transformer networks, illuminating both the unanticipated success, and the significant limitations, of transformers in symbol processing. Borrowing insights from symbolic AI and cognitive science on the power of Production System architectures, we develop a high-level Production System Language, PSL, that allows us to write symbolic programs to do complex, abstract symbol processing, and create compilers that precisely implement PSL programs in transformer networks which are, by construction, 100% mechanistically interpretable. The work is driven by study of a purely abstract (semantics-free) symbolic task that we develop, Templatic Generation (TGT). Although developed through study of TGT, PSL is, we demonstrate, highly general: it is Turing Universal. The new type of transformer architecture that we compile from PSL programs suggests a number of paths for enhancing transformers' capabilities at symbol processing. We note, however, that the work we report addresses computability, and not learnability, by transformer networks. Note: The first section provides an extended synopsis of the entire paper.
>
---
#### [replaced 047] Apertus: Democratizing Open and Compliant LLMs for Global Language Environments
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Apertus，一个全开源的多语言大模型系列，旨在解决开放模型生态中数据合规性差与多语言覆盖不足的问题。通过使用可复现的公开数据、遵守robots.txt、过滤有害内容，并采用Goldfish目标减少记忆风险，实现合规与安全。模型在15T tokens、1800+语言上训练，支持多语言任务，性能媲美顶尖开源模型，所有开发资源均公开。**

- **链接: [https://arxiv.org/pdf/2509.14233v2](https://arxiv.org/pdf/2509.14233v2)**

> **作者:** Project Apertus; Alejandro Hernández-Cano; Alexander Hägele; Allen Hao Huang; Angelika Romanou; Antoni-Joan Solergibert; Barna Pasztor; Bettina Messmer; Dhia Garbaya; Eduard Frank Ďurech; Ido Hakimi; Juan García Giraldo; Mete Ismayilzada; Negar Foroutan; Skander Moalla; Tiancheng Chen; Vinko Sabolčec; Yixuan Xu; Michael Aerni; Badr AlKhamissi; Inés Altemir Mariñas; Mohammad Hossein Amani; Matin Ansaripour; Ilia Badanin; Harold Benoit; Emanuela Boros; Nicholas Browning; Fabian Bösch; Maximilian Böther; Niklas Canova; Camille Challier; Clement Charmillot; Jonathan Coles; Jan Deriu; Arnout Devos; Lukas Drescher; Daniil Dzenhaliou; Maud Ehrmann; Dongyang Fan; Simin Fan; Silin Gao; Miguel Gila; María Grandury; Diba Hashemi; Alexander Hoyle; Jiaming Jiang; Mark Klein; Andrei Kucharavy; Anastasiia Kucherenko; Frederike Lübeck; Roman Machacek; Theofilos Manitaras; Andreas Marfurt; Kyle Matoba; Simon Matrenok; Henrique Mendonça; Fawzi Roberto Mohamed; Syrielle Montariol; Luca Mouchel; Sven Najem-Meyer; Jingwei Ni; Gennaro Oliva; Matteo Pagliardini; Elia Palme; Andrei Panferov; Léo Paoletti; Marco Passerini; Ivan Pavlov; Auguste Poiroux; Kaustubh Ponkshe; Nathan Ranchin; Javi Rando; Mathieu Sauser; Jakhongir Saydaliev; Muhammad Ali Sayfiddinov; Marian Schneider; Stefano Schuppli; Marco Scialanga; Andrei Semenov; Kumar Shridhar; Raghav Singhal; Anna Sotnikova; Alexander Sternfeld; Ayush Kumar Tarun; Paul Teiletche; Jannis Vamvas; Xiaozhe Yao; Hao Zhao; Alexander Ilic; Ana Klimovic; Andreas Krause; Caglar Gulcehre; David Rosenthal; Elliott Ash; Florian Tramèr; Joost VandeVondele; Livio Veraldi; Martin Rajman; Thomas Schulthess; Torsten Hoefler; Antoine Bosselut; Martin Jaggi; Imanol Schlag
>
> **摘要:** We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting `robots.txt` exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension.
>
---
#### [replaced 048] Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs
- **分类: cs.CL; cs.CR; cs.LG**

- **简介: 该论文研究大模型中隐写协同攻击的涌现与防范。针对因奖励设计不当导致的隐蔽信息传递问题，提出两种强化学习方法生成语言隐写文本，发现现有检测与改写手段无效，警示需加强监测并创新防御机制。**

- **链接: [https://arxiv.org/pdf/2410.03768v2](https://arxiv.org/pdf/2410.03768v2)**

> **作者:** Yohan Mathew; Ollie Matthews; Robert McCarthy; Joan Velja; Christian Schroeder de Witt; Dylan Cope; Nandi Schoots
>
> **备注:** Camera-ready version. Oral presentation at IJCNLP-AACL 2025 (14th International Joint Conference on Natural Language Processing and 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics), Mumbai, India, December 20-24, 2025
>
> **摘要:** The rapid proliferation of frontier model agents promises significant societal advances but also raises concerns about systemic risks arising from unsafe interactions. Collusion to the disadvantage of others has been identified as a central form of undesirable agent cooperation. The use of information hiding (steganography) in agent communications could render such collusion practically undetectable. This underscores the need for investigations into the possibility of such behaviours emerging and the robustness corresponding countermeasures. To investigate this problem we design two approaches -- a gradient-based reinforcement learning (GBRL) method and an in-context reinforcement learning (ICRL) method -- for reliably eliciting sophisticated LLM-generated linguistic text steganography. We demonstrate, for the first time, that unintended steganographic collusion in LLMs can arise due to mispecified reward incentives during training. Additionally, we find that standard mitigations -- both passive oversight of model outputs and active mitigation through communication paraphrasing -- are not fully effective at preventing this steganographic communication. Our findings imply that (i) emergence of steganographic collusion is a plausible concern that should be monitored and researched, and (ii) preventing emergence may require innovation in mitigation techniques.
>
---
#### [replaced 049] Stabilizing Reinforcement Learning with LLMs: Formulation and Practices
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究大语言模型在强化学习中的稳定训练问题，旨在解决政策僵化与训练-推理差异导致的优化不稳。提出基于一阶近似的代理目标理论，解释重要性采样、裁剪和路由重放等技术的必要性，并通过大规模实验验证其有效性，为稳定RL训练提供理论依据与实践方案。**

- **链接: [https://arxiv.org/pdf/2512.01374v2](https://arxiv.org/pdf/2512.01374v2)**

> **作者:** Chujie Zheng; Kai Dang; Bowen Yu; Mingze Li; Huiqiang Jiang; Junrong Lin; Yuqiong Liu; Hao Lin; Chencan Wu; Feng Hu; An Yang; Jingren Zhou; Junyang Lin
>
> **摘要:** This paper proposes a novel formulation for reinforcement learning (RL) with large language models, explaining why and under what conditions the true sequence-level reward can be optimized via a surrogate token-level objective in policy gradient methods such as REINFORCE. Specifically, through a first-order approximation, we show that this surrogate becomes increasingly valid only when both the training-inference discrepancy and policy staleness are minimized. This insight provides a principled explanation for the crucial role of several widely adopted techniques in stabilizing RL training, including importance sampling correction, clipping, and particularly Routing Replay for Mixture-of-Experts (MoE) models. Through extensive experiments with a 30B MoE model totaling hundreds of thousands of GPU hours, we show that for on-policy training, the basic policy gradient algorithm with importance sampling correction achieves the highest training stability. When off-policy updates are introduced to accelerate convergence, combining clipping and Routing Replay becomes essential to mitigate the instability caused by policy staleness. Notably, once training is stabilized, prolonged optimization consistently yields comparable final performance regardless of cold-start initialization. We hope that the shared insights and the developed recipes for stable RL training will facilitate future research.
>
---
#### [replaced 050] Bridging Human and LLM Judgments: Understanding and Narrowing the Gap
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文针对大语言模型（LLM）作为评判者时与人类判断存在系统性偏差的问题，提出统一统计框架Bridge。通过建模人类偏好得分及LLM评分的线性偏差，实现对齐与校准，提升评估一致性，有效缩小人机判断差距。**

- **链接: [https://arxiv.org/pdf/2508.12792v2](https://arxiv.org/pdf/2508.12792v2)**

> **作者:** Felipe Maia Polo; Xinhe Wang; Mikhail Yurochkin; Gongjun Xu; Moulinath Banerjee; Yuekai Sun
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large language models are increasingly used as judges (LLM-as-a-judge) to evaluate model outputs at scale, but their assessments often diverge systematically from human judgments. We present Bridge, a unified statistical framework that explicitly bridges human and LLM evaluations under both absolute scoring and pairwise comparison paradigms. Bridge posits a latent human preference score for each prompt-response pair and models LLM deviations as linear transformations of covariates that capture sources of discrepancies. This offers a simple and principled framework for refining LLM ratings and characterizing systematic discrepancies between humans and LLMs. We provide an efficient fitting algorithm with asymptotic guarantees for statistical inference. Using six LLM judges and two benchmarks (BigGen Bench and Chatbot Arena), Bridge achieves higher agreement with human ratings (accuracy, calibration, and KL divergence) and exposes systematic human-LLM gaps.
>
---
#### [replaced 051] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI
- **分类: cs.CL; q-bio.NC**

- **简介: 该论文探究大语言模型（LLM）与人脑在句子级语言处理上的计算机制是否相似。通过对比14个LLM的分层嵌入与人类fMRI数据，构建神经预测模型，发现模型性能提升促使表示架构向脑区激活模式演化，尤其在高阶语义抽象层面表现出更强对应性。研究揭示了LLM与人脑在语言处理上的潜在共性。**

- **链接: [https://arxiv.org/pdf/2505.22563v3](https://arxiv.org/pdf/2505.22563v3)**

> **作者:** Yu Lei; Xingyang Ge; Yi Zhang; Yiming Yang; Bolei Ma
>
> **备注:** AAAI 2026
>
> **摘要:** Understanding whether large language models (LLMs) and the human brain converge on similar computational principles remains a fundamental and important question in cognitive neuroscience and AI. Do the brain-like patterns observed in LLMs emerge simply from scaling, or do they reflect deeper alignment with the architecture of human language processing? This study focuses on the sentence-level neural mechanisms of language models, systematically investigating how layer-wise representations in LLMs align with the dynamic neural responses during human sentence comprehension. By comparing hierarchical embeddings from 14 publicly available LLMs with fMRI data collected from participants, who were exposed to a naturalistic narrative story, we constructed sentence-level neural prediction models to identify the model layers most significantly correlated with brain region activations. Results show that improvements in model performance drive the evolution of representational architectures toward brain-like hierarchies, particularly achieving stronger functional and anatomical correspondence at higher semantic abstraction levels. These findings advance our understanding of the computational parallels between LLMs and the human brain, highlighting the potential of LLMs as models for human language processing.
>
---
#### [replaced 052] OmniBench: Towards The Future of Universal Omni-Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出OmniBench基准，评估模型在视觉、听觉与文本三模态下的理解与推理能力，旨在推动通用全语言模型（OLMs）发展。针对现有模型在多模态指令遵循和推理上的不足，构建高质量标注数据集并提出OmniInstruct训练数据，以提升模型三模态融合能力。**

- **链接: [https://arxiv.org/pdf/2409.15272v5](https://arxiv.org/pdf/2409.15272v5)**

> **作者:** Yizhi Li; Ge Zhang; Yinghao Ma; Ruibin Yuan; Kang Zhu; Hangyu Guo; Yiming Liang; Jiaheng Liu; Zekun Wang; Jian Yang; Siwei Wu; Xingwei Qu; Jinjie Shi; Xinyue Zhang; Zhenzhu Yang; Xiangzhou Wang; Zhaoxiang Zhang; Zachary Liu; Emmanouil Benetos; Wenhao Huang; Chenghua Lin
>
> **摘要:** Recent advancements in multimodal large language models (MLLMs) have focused on integrating multiple modalities, yet their ability to simultaneously process and reason across different inputs remains underexplored. We introduce OmniBench, a novel benchmark designed to evaluate models' ability to recognize, interpret, and reason across visual, acoustic, and textual inputs simultaneously. We define language models capable of such tri-modal processing as omni-language models (OLMs). OmniBench features high-quality human annotations that require integrated understanding across all modalities. Our evaluation reveals that: i) open-source OLMs show significant limitations in instruction-following and reasoning in tri-modal contexts; and ii) most baseline models perform poorly (around 50% accuracy) even with textual alternatives to image/audio inputs. To address these limitations, we develop OmniInstruct, an 96K-sample instruction tuning dataset for training OLMs. We advocate for developing more robust tri-modal integration techniques and training strategies to enhance OLM performance. Codes and data could be found at our repo (https://github.com/multimodal-art-projection/OmniBench).
>
---
