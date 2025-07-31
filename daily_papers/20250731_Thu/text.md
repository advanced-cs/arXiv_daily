# 自然语言处理 cs.CL

- **最新发布 51 篇**

- **更新 52 篇**

## 最新发布

#### [new 001] A Benchmark Dataset and Evaluation Framework for Vietnamese Large Language Models in Customer Support
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决越南语大语言模型在客服领域缺乏领域评估和基准数据的问题。论文构建了包含9000多问答对的越南语客服对话数据集CSConDa，并提出评估框架，对11个轻量级开源越南语模型进行系统评估，以支持模型选择与改进。**

- **链接: [http://arxiv.org/pdf/2507.22542v1](http://arxiv.org/pdf/2507.22542v1)**

> **作者:** Long S. T. Nguyen; Truong P. Hua; Thanh M. Nguyen; Toan Q. Pham; Nam K. Ngo; An X. Nguyen; Nghi D. M. Pham; Nghia H. Nguyen; Tho T. Quan
>
> **备注:** Under review at ICCCI 2025
>
> **摘要:** With the rapid growth of Artificial Intelligence, Large Language Models (LLMs) have become essential for Question Answering (QA) systems, improving efficiency and reducing human workload in customer service. The emergence of Vietnamese LLMs (ViLLMs) highlights lightweight open-source models as a practical choice for their accuracy, efficiency, and privacy benefits. However, domain-specific evaluations remain limited, and the absence of benchmark datasets reflecting real customer interactions makes it difficult for enterprises to select suitable models for support applications. To address this gap, we introduce the Customer Support Conversations Dataset (CSConDa), a curated benchmark of over 9,000 QA pairs drawn from real interactions with human advisors at a large Vietnamese software company. Covering diverse topics such as pricing, product availability, and technical troubleshooting, CSConDa provides a representative basis for evaluating ViLLMs in practical scenarios. We further present a comprehensive evaluation framework, benchmarking 11 lightweight open-source ViLLMs on CSConDa with both automatic metrics and syntactic analysis to reveal model strengths, weaknesses, and linguistic patterns. This study offers insights into model behavior, explains performance differences, and identifies key areas for improvement, supporting the development of next-generation ViLLMs. By establishing a robust benchmark and systematic evaluation, our work enables informed model selection for customer service QA and advances research on Vietnamese LLMs. The dataset is publicly available at https://huggingface.co/datasets/ura-hcmut/Vietnamese-Customer-Support-QA.
>
---
#### [new 002] NeedleChain: Measuring Intact Long-Context Reasoning Capability of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在评估大语言模型对长上下文的理解能力。现有“针在 haystack”基准可能高估模型能力，因此作者提出新基准 NeedleChain，使用全相关上下文测试模型理解。此外，作者提出 ROPE Contraction 方法提升模型表现，揭示模型处理与理解长上下文的差距。**

- **链接: [http://arxiv.org/pdf/2507.22411v1](http://arxiv.org/pdf/2507.22411v1)**

> **作者:** Hyeonseok Moon; Heuiseok Lim
>
> **备注:** 13 pages
>
> **摘要:** The Needle-in-a-Haystack (NIAH) benchmark is widely used to evaluate Large Language Models' (LLMs) ability to understand long contexts (LC). It evaluates the capability to identify query-relevant context within extensive query-irrelevant passages. Although this method serves as a widely accepted standard for evaluating long-context understanding, our findings suggest it may overestimate the true LC capability of LLMs. We demonstrate that even state-of-the-art models such as GPT-4o struggle to intactly incorporate given contexts made up of solely query-relevant ten sentences. In response, we introduce a novel benchmark, \textbf{NeedleChain}, where the context consists entirely of query-relevant information, requiring the LLM to fully grasp the input to answer correctly. Our benchmark allows for flexible context length and reasoning order, offering a more comprehensive analysis of LLM performance. Additionally, we propose an extremely simple yet compelling strategy to improve LC understanding capability of LLM: ROPE Contraction. Our experiments with various advanced LLMs reveal a notable disparity between their ability to process large contexts and their capacity to fully understand them. Source code and datasets are available at https://github.com/hyeonseokk/NeedleChain
>
---
#### [new 003] IndoPref: A Multi-Domain Pairwise Preference Dataset for Indonesian
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决印尼语在大语言模型偏好研究中的代表性不足问题。作者构建了首个印尼语多领域偏好数据集IndoPref，用于评估生成文本的自然性和质量，并通过多模型实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.22159v1](http://arxiv.org/pdf/2507.22159v1)**

> **作者:** Vanessa Rebecca Wiyono; David Anugraha; Ayu Purwarianti; Genta Indra Winata
>
> **备注:** Preprint
>
> **摘要:** Over 200 million people speak Indonesian, yet the language remains significantly underrepresented in preference-based research for large language models (LLMs). Most existing multilingual datasets are derived from English translations, often resulting in content that lacks cultural and linguistic authenticity. To address this gap, we introduce IndoPref, the first fully human-authored and multi-domain Indonesian preference dataset specifically designed to evaluate the naturalness and quality of LLM-generated text. All annotations are natively written in Indonesian and evaluated using Krippendorff's alpha, demonstrating strong inter-annotator agreement. Additionally, we benchmark the dataset across multiple LLMs and assess the output quality of each model.
>
---
#### [new 004] How Well Does First-Token Entropy Approximate Word Entropy as a Psycholinguistic Predictor?
- **分类: cs.CL**

- **简介: 该论文研究语言模型中词熵的近似方法，探讨首子词熵与真实词熵的差异。任务是比较二者在心理语言学预测中的效果。作者使用蒙特卡洛方法估计真实词熵，通过回归实验发现首子词熵存在低估和偏差，提示需谨慎使用该近似方法。**

- **链接: [http://arxiv.org/pdf/2507.22209v1](http://arxiv.org/pdf/2507.22209v1)**

> **作者:** Christian Clark; Byung-Doh Oh; William Schuler
>
> **摘要:** Contextual entropy is a psycholinguistic measure capturing the anticipated difficulty of processing a word just before it is encountered. Recent studies have tested for entropy-related effects as a potential complement to well-known effects from surprisal. For convenience, entropy is typically estimated based on a language model's probability distribution over a word's first subword token. However, this approximation results in underestimation and potential distortion of true word entropy. To address this, we generate Monte Carlo (MC) estimates of word entropy that allow words to span a variable number of tokens. Regression experiments on reading times show divergent results between first-token and MC word entropy, suggesting a need for caution in using first-token approximations of contextual entropy.
>
---
#### [new 005] AI-generated stories favour stability over change: homogeneity and cultural stereotyping in narratives generated by gpt-4o-mini
- **分类: cs.CL; cs.AI; H.1.2; I.2.4; I.2.0; I.2.7**

- **简介: 该论文研究语言模型生成故事的文化相关性，属于自然语言处理与文化研究的交叉任务。旨在解决AI生成内容是否存在文化偏见与叙事同质化问题。工作包括用gpt-4o-mini生成236国故事，分析其叙事结构，发现AI偏好稳定、传统情节，缺乏现实冲突与多样性，提出叙事标准化是一种新型AI偏见。**

- **链接: [http://arxiv.org/pdf/2507.22445v1](http://arxiv.org/pdf/2507.22445v1)**

> **作者:** Jill Walker Rettberg; Hermann Wigers
>
> **备注:** This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement number 101142306. The project is also supported by the Center for Digital Narrative, which is funded by the Research Council of Norway through its Centres of Excellence scheme, project number 332643
>
> **摘要:** Can a language model trained largely on Anglo-American texts generate stories that are culturally relevant to other nationalities? To find out, we generated 11,800 stories - 50 for each of 236 countries - by sending the prompt "Write a 1500 word potential {demonym} story" to OpenAI's model gpt-4o-mini. Although the stories do include surface-level national symbols and themes, they overwhelmingly conform to a single narrative plot structure across countries: a protagonist lives in or returns home to a small town and resolves a minor conflict by reconnecting with tradition and organising community events. Real-world conflicts are sanitised, romance is almost absent, and narrative tension is downplayed in favour of nostalgia and reconciliation. The result is a narrative homogenisation: an AI-generated synthetic imaginary that prioritises stability above change and tradition above growth. We argue that the structural homogeneity of AI-generated narratives constitutes a distinct form of AI bias, a narrative standardisation that should be acknowledged alongside the more familiar representational bias. These findings are relevant to literary studies, narratology, critical AI studies, NLP research, and efforts to improve the cultural alignment of generative AI.
>
---
#### [new 006] DBLPLink 2.0 -- An Entity Linker for the DBLP Scholarly Knowledge Graph
- **分类: cs.CL**

- **简介: 论文提出DBLPLink 2.0，一个用于DBLP学术知识图谱的实体链接工具。该工作属于自然语言处理中的实体链接任务，旨在解决学术数据中实体歧义问题。相比之前版本，其创新点在于采用大语言模型（LLM）实现零样本学习，并通过LLM倒数第二层“yes”标记的概率对候选实体进行重排序，提升链接准确性。**

- **链接: [http://arxiv.org/pdf/2507.22811v1](http://arxiv.org/pdf/2507.22811v1)**

> **作者:** Debayan Banerjee; Tilahun Abedissa Taffa; Ricardo Usbeck
>
> **摘要:** In this work we present an entity linker for DBLP's 2025 version of RDF-based Knowledge Graph. Compared to the 2022 version, DBLP now considers publication venues as a new entity type called dblp:Stream. In the earlier version of DBLPLink, we trained KG-embeddings and re-rankers on a dataset to produce entity linkings. In contrast, in this work, we develop a zero-shot entity linker using LLMs using a novel method, where we re-rank candidate entities based on the log-probabilities of the "yes" token output at the penultimate layer of the LLM.
>
---
#### [new 007] A Comprehensive Taxonomy of Negation for NLP and Neural Retrievers
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理任务，旨在解决神经信息检索模型在处理否定语句时表现不佳的问题。论文提出了一个基于哲学、语言学和逻辑学的否定分类体系，生成了两个用于评估和优化模型的基准数据集，并设计了一个基于逻辑的分类机制，用于分析模型在不同否定类型上的表现，从而提升模型对否定语句的泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.22337v1](http://arxiv.org/pdf/2507.22337v1)**

> **作者:** Roxana Petcu; Samarth Bhargav; Maarten de Rijke; Evangelos Kanoulas
>
> **摘要:** Understanding and solving complex reasoning tasks is vital for addressing the information needs of a user. Although dense neural models learn contextualised embeddings, they still underperform on queries containing negation. To understand this phenomenon, we study negation in both traditional neural information retrieval and LLM-based models. We (1) introduce a taxonomy of negation that derives from philosophical, linguistic, and logical definitions; (2) generate two benchmark datasets that can be used to evaluate the performance of neural information retrieval models and to fine-tune models for a more robust performance on negation; and (3) propose a logic-based classification mechanism that can be used to analyze the performance of retrieval models on existing datasets. Our taxonomy produces a balanced data distribution over negation types, providing a better training setup that leads to faster convergence on the NevIR dataset. Moreover, we propose a classification schema that reveals the coverage of negation types in existing datasets, offering insights into the factors that might affect the generalization of fine-tuned models on negation.
>
---
#### [new 008] The role of media memorability in facilitating startups' access to venture capital funding
- **分类: cs.CL; cs.SI; physics.soc-ph; I.2.7; J.4; H.4.0**

- **简介: 论文探讨媒体记忆度对初创企业获取风投的影响，属于创业金融与媒体合法化研究任务。旨在解决媒体报道如何影响风投决策的问题，通过分析197家英国微纳米科技初创企业的数据，发现媒体记忆度（即企业名称在投资者记忆中的留存能力）显著影响融资结果，建议初创企业应强化品牌独特性和行业关联性以提高记忆度。**

- **链接: [http://arxiv.org/pdf/2507.22201v1](http://arxiv.org/pdf/2507.22201v1)**

> **作者:** L. Toschi; S. Torrisi; A. Fronzetti Colladon
>
> **摘要:** Media reputation plays an important role in attracting venture capital investment. However, prior research has focused too narrowly on general media exposure, limiting our understanding of how media truly influences funding decisions. As informed decision-makers, venture capitalists respond to more nuanced aspects of media content. We introduce the concept of media memorability - the media's ability to imprint a startup's name in the memory of relevant investors. Using data from 197 UK startups in the micro and nanotechnology sector (funded between 1995 and 2004), we show that media memorability significantly influences investment outcomes. Our findings suggest that venture capitalists rely on detailed cues such as a startup's distinctiveness and connectivity within news semantic networks. This contributes to research on entrepreneurial finance and media legitimation. In practice, startups should go beyond frequent media mentions to strengthen brand memorability through more targeted, meaningful coverage highlighting their uniqueness and relevance within the broader industry conversation.
>
---
#### [new 009] Unveiling the Influence of Amplifying Language-Specific Neurons
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究放大语言特定神经元对多语言大模型的影响，属于自然语言处理任务。旨在解决如何通过干预模型神经元来引导语言输出的问题。论文通过放大语言特定神经元进行多语言干预实验，提出了LSS评估指标，并在多个下游任务上测试效果。结果显示放大干预能有效引导语言输出，对低资源语言有帮助，但跨语言迁移效果有限。**

- **链接: [http://arxiv.org/pdf/2507.22581v1](http://arxiv.org/pdf/2507.22581v1)**

> **作者:** Inaya Rahmanisa; Lyzander Marciano Andrylie; Krisna Mahardika Ihsani; Alfan Farizki Wicaksono; Haryo Akbarianto Wibowo; Alham Fikri Aji
>
> **备注:** Our code and dataset are made available at https://github.com/tauimbz/lang-task-neuron
>
> **摘要:** Language-specific neurons in LLMs that strongly correlate with individual languages have been shown to influence model behavior by deactivating them. However, their role in amplification remains underexplored. This work investigates the effect of amplifying language-specific neurons through interventions across 18 languages, including low-resource ones, using three models primarily trained in different languages. We compare amplification factors by their effectiveness in steering to the target language using a proposed Language Steering Shift (LSS) evaluation score, then evaluate it on downstream tasks: commonsense reasoning (XCOPA, XWinograd), knowledge (Include), and translation (FLORES). The optimal amplification factors effectively steer output toward nearly all tested languages. Intervention using this factor on downstream tasks improves self-language performance in some cases but generally degrades cross-language results. These findings highlight the effect of language-specific neurons in multilingual behavior, where amplification can be beneficial especially for low-resource languages, but provides limited advantage for cross-lingual transfer.
>
---
#### [new 010] Persona-Augmented Benchmarking: Evaluating LLMs Across Diverse Writing Styles
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决当前LLM评估基准缺乏写作风格多样性的问题。作者通过基于角色的提示重写，模拟不同写作风格，发现风格变化显著影响模型表现，提出了可扩展的增强基准方法。**

- **链接: [http://arxiv.org/pdf/2507.22168v1](http://arxiv.org/pdf/2507.22168v1)**

> **作者:** Kimberly Le Truong; Riccardo Fogliato; Hoda Heidari; Zhiwei Steven Wu
>
> **摘要:** Current benchmarks for evaluating Large Language Models (LLMs) often do not exhibit enough writing style diversity, with many adhering primarily to standardized conventions. Such benchmarks do not fully capture the rich variety of communication patterns exhibited by humans. Thus, it is possible that LLMs, which are optimized on these benchmarks, may demonstrate brittle performance when faced with "non-standard" input. In this work, we test this hypothesis by rewriting evaluation prompts using persona-based LLM prompting, a low-cost method to emulate diverse writing styles. Our results show that, even with identical semantic content, variations in writing style and prompt formatting significantly impact the estimated performance of the LLM under evaluation. Notably, we identify distinct writing styles that consistently trigger either low or high performance across a range of models and tasks, irrespective of model family, size, and recency. Our work offers a scalable approach to augment existing benchmarks, improving the external validity of the assessments they provide for measuring LLM performance across linguistic variations.
>
---
#### [new 011] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CliCARE框架，旨在通过将大型语言模型（LLMs）与临床指南结合，提升癌症电子健康记录（EHR）的决策支持。任务是解决LLMs在处理长文本、多语言EHR时的局限性，降低临床幻觉风险，并提高决策建议的准确性。方法是构建患者特定的时间知识图谱（TKG），并与规范指南图谱对齐，生成高保真临床摘要和建议。实验验证显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.22533v1](http://arxiv.org/pdf/2507.22533v1)**

> **作者:** Dongchen Li; Jitao Liang; Wei Li; Xiaoyu Wang; Longbing Cao; Kun Yu
>
> **摘要:** Large Language Models (LLMs) hold significant promise for improving clinical decision support and reducing physician burnout by synthesizing complex, longitudinal cancer Electronic Health Records (EHRs). However, their implementation in this critical field faces three primary challenges: the inability to effectively process the extensive length and multilingual nature of patient records for accurate temporal analysis; a heightened risk of clinical hallucination, as conventional grounding techniques such as Retrieval-Augmented Generation (RAG) do not adequately incorporate process-oriented clinical guidelines; and unreliable evaluation metrics that hinder the validation of AI systems in oncology. To address these issues, we propose CliCARE, a framework for Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records. The framework operates by transforming unstructured, longitudinal EHRs into patient-specific Temporal Knowledge Graphs (TKGs) to capture long-range dependencies, and then grounding the decision support process by aligning these real-world patient trajectories with a normative guideline knowledge graph. This approach provides oncologists with evidence-grounded decision support by generating a high-fidelity clinical summary and an actionable recommendation. We validated our framework using large-scale, longitudinal data from a private Chinese cancer dataset and the public English MIMIC-IV dataset. In these diverse settings, CliCARE significantly outperforms strong baselines, including leading long-context LLMs and Knowledge Graph-enhanced RAG methods. The clinical validity of our results is supported by a robust evaluation protocol, which demonstrates a high correlation with assessments made by expert oncologists.
>
---
#### [new 012] Intent Recognition and Out-of-Scope Detection using LLMs in Multi-party Conversations
- **分类: cs.CL; cs.LG**

- **简介: 论文研究任务导向对话系统中的意图识别与超出范围意图检测。针对传统方法依赖大量标注数据的问题，提出结合BERT与大语言模型的混合方法，在零样本和小样本场景下提升多轮对话场景中的意图识别与OOS检测效果。**

- **链接: [http://arxiv.org/pdf/2507.22289v1](http://arxiv.org/pdf/2507.22289v1)**

> **作者:** Galo Castillo-López; Gaël de Chalendar; Nasredine Semmar
>
> **备注:** Accepted for publication at SIGDIAL 2025
>
> **摘要:** Intent recognition is a fundamental component in task-oriented dialogue systems (TODS). Determining user intents and detecting whether an intent is Out-of-Scope (OOS) is crucial for TODS to provide reliable responses. However, traditional TODS require large amount of annotated data. In this work we propose a hybrid approach to combine BERT and LLMs in zero and few-shot settings to recognize intents and detect OOS utterances. Our approach leverages LLMs generalization power and BERT's computational efficiency in such scenarios. We evaluate our method on multi-party conversation corpora and observe that sharing information from BERT outputs to LLMs leads to system performance improvement.
>
---
#### [new 013] Reducing Hallucinations in Summarization via Reinforcement Learning with Entity Hallucination Index
- **分类: cs.CL; cs.AI; 68T50; I.2.7**

- **简介: 该论文属于文本摘要生成任务，旨在减少生成式摘要中的实体幻觉问题。通过引入实体幻觉指数（EHI）作为奖励信号，利用强化学习对预训练语言模型进行微调，优化生成结果的实体准确性和真实性，从而提升摘要的可信度与实用性。**

- **链接: [http://arxiv.org/pdf/2507.22744v1](http://arxiv.org/pdf/2507.22744v1)**

> **作者:** Praveenkumar Katwe; Rakesh Chandra; Balabantaray Kali; Prasad Vittala
>
> **备注:** 8
>
> **摘要:** Reducing hallucinations in abstractive summarization remains a critical challenge for deploying language models (LMs) in real-world settings. In this work, we introduce a rewarddriven fine-tuning framework that explicitly optimizes for Entity Hallucination Index (EHI), a metric designed to quantify the presence, correctness, and grounding of named entities in generated summaries. Given a corpus of meeting transcripts, we first generate baseline summaries using a pre-trained LM and compute EHI scores via automatic entity extraction and matching. We then apply reinforcement learning to fine-tune the model parameters, using EHI as a reward signal to bias generation toward entity-faithful outputs. Our approach does not rely on human-written factuality annotations, enabling scalable fine-tuning. Experiments demonstrate consistent improvements in EHI across datasets, with qualitative analysis revealing a significant reduction in entity-level hallucinations without degradation in fluency or informativeness. We release a reproducible Colab pipeline, facilitating further research on hallucination-aware model fine-tuning using lightweight, hallucintion metrics like EHI.
>
---
#### [new 014] Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于人工智能安全任务，旨在揭示大语言模型（LLM）安全机制的漏洞。它提出了一种利用认知偏差协同效应的攻击框架CognitiveAttack，通过结合监督微调与强化学习生成攻击性提示，有效绕过安全防护，暴露出当前防御机制的不足，为构建更安全AI系统提供新视角。**

- **链接: [http://arxiv.org/pdf/2507.22564v1](http://arxiv.org/pdf/2507.22564v1)**

> **作者:** Xikang Yang; Biyu Zhou; Xuehai Tang; Jizhong Han; Songlin Hu
>
> **摘要:** Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.
>
---
#### [new 015] Multilingual Political Views of Large Language Models: Identification and Steering
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的政治倾向，属于自然语言处理任务。旨在解决现有研究覆盖模型和语言有限、缺乏对偏见控制的问题。作者评估了七个多语言模型在14种语言中的政治立场，并尝试通过激活干预技术操控模型的政治倾向。**

- **链接: [http://arxiv.org/pdf/2507.22623v1](http://arxiv.org/pdf/2507.22623v1)**

> **作者:** Daniil Gurgurov; Katharina Trinley; Ivan Vykopal; Josef van Genabith; Simon Ostermann; Roberto Zamparelli
>
> **备注:** pre-print
>
> **摘要:** Large language models (LLMs) are increasingly used in everyday tools and applications, raising concerns about their potential influence on political views. While prior research has shown that LLMs often exhibit measurable political biases--frequently skewing toward liberal or progressive positions--key gaps remain. Most existing studies evaluate only a narrow set of models and languages, leaving open questions about the generalizability of political biases across architectures, scales, and multilingual settings. Moreover, few works examine whether these biases can be actively controlled. In this work, we address these gaps through a large-scale study of political orientation in modern open-source instruction-tuned LLMs. We evaluate seven models, including LLaMA-3.1, Qwen-3, and Aya-Expanse, across 14 languages using the Political Compass Test with 11 semantically equivalent paraphrases per statement to ensure robust measurement. Our results reveal that larger models consistently shift toward libertarian-left positions, with significant variations across languages and model families. To test the manipulability of political stances, we utilize a simple center-of-mass activation intervention technique and show that it reliably steers model responses toward alternative ideological positions across multiple languages. Our code is publicly available at https://github.com/d-gurgurov/Political-Ideologies-LLMs.
>
---
#### [new 016] MASCA: LLM based-Multi Agents System for Credit Assessment
- **分类: cs.CL; cs.CE; cs.LG**

- **简介: 该论文属于金融信用评估任务，旨在解决传统方法在信用评估中的局限性。作者提出了MASCA，一种基于大语言模型的多智能体系统，通过分层架构和对比学习优化风险与奖励评估，并引入博弈论视角与偏差分析，提升信用评分的准确性与公平性。**

- **链接: [http://arxiv.org/pdf/2507.22758v1](http://arxiv.org/pdf/2507.22758v1)**

> **作者:** Gautam Jajoo; Pranjal A Chitale; Saksham Agarwal
>
> **备注:** Accepted at ACL REALM Workshop. Work in Progress
>
> **摘要:** Recent advancements in financial problem-solving have leveraged LLMs and agent-based systems, with a primary focus on trading and financial modeling. However, credit assessment remains an underexplored challenge, traditionally dependent on rule-based methods and statistical models. In this paper, we introduce MASCA, an LLM-driven multi-agent system designed to enhance credit evaluation by mirroring real-world decision-making processes. The framework employs a layered architecture where specialized LLM-based agents collaboratively tackle sub-tasks. Additionally, we integrate contrastive learning for risk and reward assessment to optimize decision-making. We further present a signaling game theory perspective on hierarchical multi-agent systems, offering theoretical insights into their structure and interactions. Our paper also includes a detailed bias analysis in credit assessment, addressing fairness concerns. Experimental results demonstrate that MASCA outperforms baseline approaches, highlighting the effectiveness of hierarchical LLM-based multi-agent systems in financial applications, particularly in credit scoring.
>
---
#### [new 017] Language Arithmetics: Towards Systematic Language Neuron Identification and Manipulation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在探究大型语言模型中的语言特定神经机制。作者通过分析多语言模型，识别控制语言行为的特定神经元，并提出语言算术方法，系统操控神经元激活以引导模型行为。工作包括语言特定神经元定位、跨语言干预实验及效果评估，提升多语言模型控制能力。**

- **链接: [http://arxiv.org/pdf/2507.22608v1](http://arxiv.org/pdf/2507.22608v1)**

> **作者:** Daniil Gurgurov; Katharina Trinley; Yusser Al Ghussin; Tanja Baeumel; Josef van Genabith; Simon Ostermann
>
> **备注:** preprint
>
> **摘要:** Large language models (LLMs) exhibit strong multilingual abilities, yet the neural mechanisms behind language-specific processing remain unclear. We analyze language-specific neurons in Llama-3.1-8B, Mistral-Nemo-12B, and Aya-Expanse-8B & 32B across 21 typologically diverse languages, identifying neurons that control language behavior. Using the Language Activation Probability Entropy (LAPE) method, we show that these neurons cluster in deeper layers, with non-Latin scripts showing greater specialization. Related languages share overlapping neurons, reflecting internal representations of linguistic proximity. Through language arithmetics, i.e. systematic activation addition and multiplication, we steer models to deactivate unwanted languages and activate desired ones, outperforming simpler replacement approaches. These interventions effectively guide behavior across five multilingual tasks: language forcing, translation, QA, comprehension, and NLI. Manipulation is more successful for high-resource languages, while typological similarity improves effectiveness. We also demonstrate that cross-lingual neuron steering enhances downstream performance and reveal internal "fallback" mechanisms for language selection when neurons are progressively deactivated. Our code is made publicly available at https://github.com/d-gurgurov/Language-Neurons-Manipulation.
>
---
#### [new 018] RL from Teacher-Model Refinement: Gradual Imitation Learning for Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器翻译任务，旨在解决现有偏好学习方法依赖大量人工标注数据且泛化能力差的问题。作者提出RLfR框架，通过引入外部教师模型（如GPT-4o）提供连续高质量反馈，使翻译过程逐步优化，提升语义准确性和实体保留效果。实验表明该方法在多个翻译方向上优于基线模型。**

- **链接: [http://arxiv.org/pdf/2507.22219v1](http://arxiv.org/pdf/2507.22219v1)**

> **作者:** Dongyub Jude Lee; Zhenyi Ye; Pengcheng He
>
> **摘要:** Preference-learning methods for machine translation (MT)--such as Direct Preference Optimization (DPO)--have achieved impressive gains but depend heavily on large, carefully curated triplet datasets and often struggle to generalize beyond their tuning domains. We propose Reinforcement Learning from Teacher-Model Refinement (RLfR), a novel framework that removes reliance on static triplets by leveraging continuous, high-quality feedback from an external teacher model (GPT-4o). RLfR frames each translation step as a micro-tutorial: the actor generates a hypothesis, the teacher refines it, and the actor is rewarded based on how closely it aligns with the teacher's refinement. Guided by two complementary signals--(i) negative edit distance, promoting lexical and structural fidelity, and (ii) COMET score, ensuring semantic adequacy--the actor progressively learns to emulate the teacher, mirroring a human learning process through incremental, iterative improvement. On the FLORES-200 benchmark (English to and from German, Spanish, Chinese, Korean, and Japanese), RLfR consistently outperforms both MT-SFT and preference-based baselines, significantly improving COMET (semantic adequacy) and M-ETA (entity preservation) scores.
>
---
#### [new 019] Opportunities and Challenges of LLMs in Education: An NLP Perspective
- **分类: cs.CL**

- **简介: 该论文探讨了大语言模型（LLMs）在教育中的应用机会与挑战，聚焦于自然语言处理（NLP）视角。论文任务是分析LLMs在教学辅助与评估两个主要场景中的影响，并从阅读、写作、口语和辅导四个维度展开讨论，提出新方向与关键挑战。**

- **链接: [http://arxiv.org/pdf/2507.22753v1](http://arxiv.org/pdf/2507.22753v1)**

> **作者:** Sowmya Vajjala; Bashar Alhafni; Stefano Bannò; Kaushal Kumar Maurya; Ekaterina Kochmar
>
> **备注:** Pre-print
>
> **摘要:** Interest in the role of large language models (LLMs) in education is increasing, considering the new opportunities they offer for teaching, learning, and assessment. In this paper, we examine the impact of LLMs on educational NLP in the context of two main application scenarios: {\em assistance} and {\em assessment}, grounding them along the four dimensions -- reading, writing, speaking, and tutoring. We then present the new directions enabled by LLMs, and the key challenges to address. We envision that this holistic overview would be useful for NLP researchers and practitioners interested in exploring the role of LLMs in developing language-focused and NLP-enabled educational applications of the future.
>
---
#### [new 020] What is an "Abstract Reasoner"? Revisiting Experiments and Arguments about Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文探讨大型语言模型（LLMs）是否具备“抽象推理”能力。通过重新审视零样本表现差的实验，作者发现微调部分参数可显著提升性能，但效果不跨数据集迁移。论文旨在重新讨论“抽象推理者”的定义及LLMs是否符合该标准。**

- **链接: [http://arxiv.org/pdf/2507.22457v1](http://arxiv.org/pdf/2507.22457v1)**

> **作者:** Tian Yun; Chen Sun; Ellie Pavlick
>
> **备注:** CONLL 2025. Project webpage: https://abstract-reasoner-llm.github.io/
>
> **摘要:** Recent work has argued that large language models (LLMs) are not "abstract reasoners", citing their poor zero-shot performance on a variety of challenging tasks as evidence. We revisit these experiments in order to add nuance to the claim. First, we show that while LLMs indeed perform poorly in a zero-shot setting, even tuning a small subset of parameters for input encoding can enable near-perfect performance. However, we also show that this finetuning does not necessarily transfer across datasets. We take this collection of empirical results as an invitation to (re-)open the discussion of what it means to be an "abstract reasoner", and why it matters whether LLMs fit the bill.
>
---
#### [new 021] Listening to the Unspoken: Exploring 365 Aspects of Multimodal Interview Performance Assessment
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于多模态面试表现评估任务，旨在解决如何全面、公正地评估候选人面试表现的问题。论文提出了一个包含365个评估方面的框架，融合视频、音频和文本三种模态，使用特征提取、共享压缩多层感知机和两层集成学习策略，实现对五个评估维度的预测，提升了评估的自动化与准确性。**

- **链接: [http://arxiv.org/pdf/2507.22676v1](http://arxiv.org/pdf/2507.22676v1)**

> **作者:** Jia Li; Yang Wang; Wenhao Qian; Zhenzhen Hu; Richang Hong; Meng Wang
>
> **备注:** 8 pages, 4 figures, ACM MM 2025. github:https://github.com/MSA-LMC/365Aspects
>
> **摘要:** Interview performance assessment is essential for determining candidates' suitability for professional positions. To ensure holistic and fair evaluations, we propose a novel and comprehensive framework that explores ``365'' aspects of interview performance by integrating \textit{three} modalities (video, audio, and text), \textit{six} responses per candidate, and \textit{five} key evaluation dimensions. The framework employs modality-specific feature extractors to encode heterogeneous data streams and subsequently fused via a Shared Compression Multilayer Perceptron. This module compresses multimodal embeddings into a unified latent space, facilitating efficient feature interaction. To enhance prediction robustness, we incorporate a two-level ensemble learning strategy: (1) independent regression heads predict scores for each response, and (2) predictions are aggregated across responses using a mean-pooling mechanism to produce final scores for the five target dimensions. By listening to the unspoken, our approach captures both explicit and implicit cues from multimodal data, enabling comprehensive and unbiased assessments. Achieving a multi-dimensional average MSE of 0.1824, our framework secured first place in the AVI Challenge 2025, demonstrating its effectiveness and robustness in advancing automated and multimodal interview performance assessment. The full implementation is available at https://github.com/MSA-LMC/365Aspects.
>
---
#### [new 022] From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs
- **分类: cs.CL**

- **简介: 该论文属于检索增强推理任务，旨在解决现有方法忽视中间推理质量的问题。提出了TIRESRAG-R1框架，通过思考-检索-反思过程和多维奖励系统提升推理质量，包括充分性奖励、推理质量奖励和反思奖励。实验表明其在多跳问答数据集上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.22716v1](http://arxiv.org/pdf/2507.22716v1)**

> **作者:** Jie He; Victor Gutierrez Basulto; Jeff Z. Pan
>
> **摘要:** Reinforcement learning-based retrieval-augmented generation (RAG) methods enhance the reasoning abilities of large language models (LLMs). However, most rely only on final-answer rewards, overlooking intermediate reasoning quality. This paper analyzes existing RAG reasoning models and identifies three main failure patterns: (1) information insufficiency, meaning the model fails to retrieve adequate support; (2) faulty reasoning, where logical or content-level flaws appear despite sufficient information; and (3) answer-reasoning inconsistency, where a valid reasoning chain leads to a mismatched final answer. We propose TIRESRAG-R1, a novel framework using a think-retrieve-reflect process and a multi-dimensional reward system to improve reasoning and stability. TIRESRAG-R1 introduces: (1) a sufficiency reward to encourage thorough retrieval; (2) a reasoning quality reward to assess the rationality and accuracy of the reasoning chain; and (3) a reflection reward to detect and revise errors. It also employs a difficulty-aware reweighting strategy and training sample filtering to boost performance on complex tasks. Experiments on four multi-hop QA datasets show that TIRESRAG-R1 outperforms prior RAG methods and generalizes well to single-hop tasks. The code and data are available at: https://github.com/probe2/TIRESRAG-R1.
>
---
#### [new 023] Beyond Natural Language Plans: Structure-Aware Planning for Query-Focused Table Summarization
- **分类: cs.CL**

- **简介: 该论文属于表格摘要生成任务，旨在解决自然语言计划在复杂推理中的歧义和结构缺失问题。作者提出结构化计划TaSoF和框架SPaGe，通过结构化规划、图执行和摘要生成三个阶段，提升多表任务下的查询聚焦摘要效果，增强可靠性和可扩展性。**

- **链接: [http://arxiv.org/pdf/2507.22829v1](http://arxiv.org/pdf/2507.22829v1)**

> **作者:** Weijia Zhang; Songgaojun Deng; Evangelos Kanoulas
>
> **备注:** 10 pages, 4 figures, and 5 tables
>
> **摘要:** Query-focused table summarization requires complex reasoning, often approached through step-by-step natural language (NL) plans. However, NL plans are inherently ambiguous and lack structure, limiting their conversion into executable programs like SQL and hindering scalability, especially for multi-table tasks. To address this, we propose a paradigm shift to structured representations. We introduce a new structured plan, TaSoF, inspired by formalism in traditional multi-agent systems, and a framework, SPaGe, that formalizes the reasoning process in three phases: 1) Structured Planning to generate TaSoF from a query, 2) Graph-based Execution to convert plan steps into SQL and model dependencies via a directed cyclic graph for parallel execution, and 3) Summary Generation to produce query-focused summaries. Our method explicitly captures complex dependencies and improves reliability. Experiments on three public benchmarks show that SPaGe consistently outperforms prior models in both single- and multi-table settings, demonstrating the advantages of structured representations for robust and scalable summarization.
>
---
#### [new 024] Resource-Efficient Adaptation of Large Language Models for Text Embeddings via Prompt Engineering and Contrastive Fine-tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本嵌入任务。旨在解决如何高效适配大语言模型（LLMs）以生成高质量文本嵌入，克服传统池化方法丢失信息的问题。工作包括：探索聚合策略、任务提示工程和对比微调方法，结合后在MTEB英文聚类任务上达到SOTA，并通过注意力分析验证有效性。**

- **链接: [http://arxiv.org/pdf/2507.22729v1](http://arxiv.org/pdf/2507.22729v1)**

> **作者:** Benedikt Roth; Stephan Rappensperger; Tianming Qiu; Hamza Imamović; Julian Wörmann; Hao Shen
>
> **摘要:** Large Language Models (LLMs) have become a cornerstone in Natural Language Processing (NLP), achieving impressive performance in text generation. Their token-level representations capture rich, human-aligned semantics. However, pooling these vectors into a text embedding discards crucial information. Nevertheless, many non-generative downstream tasks, such as clustering, classification, or retrieval, still depend on accurate and controllable sentence- or document-level embeddings. We explore several adaptation strategies for pre-trained, decoder-only LLMs: (i) various aggregation techniques for token embeddings, (ii) task-specific prompt engineering, and (iii) text-level augmentation via contrastive fine-tuning. Combining these components yields state-of-the-art performance on the English clustering track of the Massive Text Embedding Benchmark (MTEB). An analysis of the attention map further shows that fine-tuning shifts focus from prompt tokens to semantically relevant words, indicating more effective compression of meaning into the final hidden state. Our experiments demonstrate that LLMs can be effectively adapted as text embedding models through a combination of prompt engineering and resource-efficient contrastive fine-tuning on synthetically generated positive pairs.
>
---
#### [new 025] BALSAM: A Platform for Benchmarking Arabic Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出了BALSAM，一个用于评估阿拉伯语大语言模型的基准平台。旨在解决阿拉伯语模型发展滞后、评估数据不足等问题。论文工作包括构建包含78项任务的数据集，并提供透明的盲测平台，以推动阿拉伯语LLM的研究与标准化。**

- **链接: [http://arxiv.org/pdf/2507.22603v1](http://arxiv.org/pdf/2507.22603v1)**

> **作者:** Rawan Al-Matham; Kareem Darwish; Raghad Al-Rasheed; Waad Alshammari; Muneera Alhoshan; Amal Almazrua; Asma Al Wazrah; Mais Alheraki; Firoj Alam; Preslav Nakov; Norah Alzahrani; Eman alBilali; Nizar Habash; Abdelrahman El-Sheikh; Muhammad Elmallah; Haonan Li; Hamdy Mubarak; Mohamed Anwar; Zaid Alyafeai; Ahmed Abdelali; Nora Altwairesh; Maram Hasanain; Abdulmohsen Al Thubaity; Shady Shehata; Bashar Alhafni; Injy Hamed; Go Inoue; Khalid Elmadani; Ossama Obeid; Fatima Haouari; Tamer Elsayed; Emad Alghamdi; Khalid Almubarak; Saied Alshahrani; Ola Aljarrah; Safa Alajlan; Areej Alshaqarawi; Maryam Alshihri; Sultana Alghurabi; Atikah Alzeghayer; Afrah Altamimi; Abdullah Alfaifi; Abdulrahman AlOsaimy
>
> **摘要:** The impressive advancement of Large Language Models (LLMs) in English has not been matched across all languages. In particular, LLM performance in Arabic lags behind, due to data scarcity, linguistic diversity of Arabic and its dialects, morphological complexity, etc. Progress is further hindered by the quality of Arabic benchmarks, which typically rely on static, publicly available data, lack comprehensive task coverage, or do not provide dedicated platforms with blind test sets. This makes it challenging to measure actual progress and to mitigate data contamination. Here, we aim to bridge these gaps. In particular, we introduce BALSAM, a comprehensive, community-driven benchmark aimed at advancing Arabic LLM development and evaluation. It includes 78 NLP tasks from 14 broad categories, with 52K examples divided into 37K test and 15K development, and a centralized, transparent platform for blind evaluation. We envision BALSAM as a unifying platform that sets standards and promotes collaborative research to advance Arabic LLM capabilities.
>
---
#### [new 026] A Scalable Pipeline for Estimating Verb Frame Frequencies Using Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决动词框架频率（VFF）估计问题。现有方法在规模、准确性或可访问性上受限。论文提出一种基于大语言模型的自动化流程，生成包含476个英语动词的语料库，并利用LLM分析句法结构，实现高效、可扩展的VFF估计，构建了覆盖更广、区分更细的新VFF数据库。**

- **链接: [http://arxiv.org/pdf/2507.22187v1](http://arxiv.org/pdf/2507.22187v1)**

> **作者:** Adam M. Morgan; Adeen Flinker
>
> **摘要:** We present an automated pipeline for estimating Verb Frame Frequencies (VFFs), the frequency with which a verb appears in particular syntactic frames. VFFs provide a powerful window into syntax in both human and machine language systems, but existing tools for calculating them are limited in scale, accuracy, or accessibility. We use large language models (LLMs) to generate a corpus of sentences containing 476 English verbs. Next, by instructing an LLM to behave like an expert linguist, we had it analyze the syntactic structure of the sentences in this corpus. This pipeline outperforms two widely used syntactic parsers across multiple evaluation datasets. Furthermore, it requires far fewer resources than manual parsing (the gold-standard), thereby enabling rapid, scalable VFF estimation. Using the LLM parser, we produce a new VFF database with broader verb coverage, finer-grained syntactic distinctions, and explicit estimates of the relative frequencies of structural alternates commonly studied in psycholinguistics. The pipeline is easily customizable and extensible to new verbs, syntactic frames, and even other languages. We present this work as a proof of concept for automated frame frequency estimation, and release all code and data to support future research.
>
---
#### [new 027] CUS-QA: Local-Knowledge-Oriented Open-Ended Question Answering Dataset
- **分类: cs.CL**

- **简介: 该论文提出了CUS-QA数据集，用于开放性区域问答任务，包含文本和视觉模态问题，旨在评估大语言模型的区域知识、跨语言生成一致性和评测指标开发。论文通过人工构建多语言问答对，并分析模型表现与评测方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22752v1](http://arxiv.org/pdf/2507.22752v1)**

> **作者:** Jindřich Libovický; Jindřich Helcl; Andrei Manea; Gianluca Vico
>
> **摘要:** We introduce a benchmark for open-ended regional question answering that encompasses both textual and visual modalities. We also provide strong baselines using state-of-the-art large language models (LLMs). Our dataset consists of manually curated questions and answers grounded in Wikipedia, created by native speakers from Czechia, Slovakia, and Ukraine, with accompanying English translations. It includes both purely textual questions and those requiring visual understanding. As a baseline, we evaluate state-of-the-art LLMs through prompting and complement this with human judgments of answer correctness. Using these human evaluations, we analyze the reliability of existing automatic evaluation metrics. Our baseline results highlight a significant gap in regional knowledge among current LLMs. Moreover, apart from LLM-based evaluation, there is minimal correlation between automated metrics and human judgment. We release this dataset as a resource to (1) assess regional knowledge in LLMs, (2) study cross-lingual generation consistency in a challenging setting, and (3) advance the development of evaluation metrics for open-ended question answering.
>
---
#### [new 028] Meaning-infused grammar: Gradient Acceptability Shapes the Geometric Representations of Constructions in LLMs
- **分类: cs.CL; cs.AI; 68T50**

- **简介: 该论文属于自然语言处理任务，旨在验证大语言模型（LLMs）是否学习到意义驱动的、具有梯度接受度的构式表征。研究者分析了Pythia-1.4B中英语与格构式的神经表征，发现人类偏好强度影响模型中构式表征的几何分离性，支持基于构式的梯度意义表征。**

- **链接: [http://arxiv.org/pdf/2507.22286v1](http://arxiv.org/pdf/2507.22286v1)**

> **作者:** Supantho Rakshit; Adele Goldberg
>
> **备注:** 5 pages, 3 figures, Accepted for publication at the Second International Workshop on Construction Grammars and NLP at the 16th International Conference for Computational Semantics (IWCS) 2025
>
> **摘要:** The usage-based constructionist (UCx) approach posits that language comprises a network of learned form-meaning pairings (constructions) whose use is largely determined by their meanings or functions, requiring them to be graded and probabilistic. This study investigates whether the internal representations in Large Language Models (LLMs) reflect the proposed function-infused gradience. We analyze the neural representations of the English dative constructions (Double Object and Prepositional Object) in Pythia-$1.4$B, using a dataset of $5000$ sentence pairs systematically varied for human-rated preference strength. A macro-level geometric analysis finds that the separability between construction representations, as measured by Energy Distance or Jensen-Shannon Divergence, is systematically modulated by gradient preference strength. More prototypical exemplars of each construction occupy more distinct regions in the activation space of LLMs. These results provide strong evidence that LLMs learn rich, meaning-infused, graded representations of constructions and offer support for geometric measures of basic constructionist principles in LLMs.
>
---
#### [new 029] Where to show Demos in Your Prompt: A Positional Bias of In-Context Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLM）在上下文学习（ICL）中因示例（demos）位置变化导致的预测偏差问题，称为“DPP偏差”。通过系统评估不同任务和模型，发现将示例置于提示开头效果最佳，提升了准确性和稳定性。**

- **链接: [http://arxiv.org/pdf/2507.22887v1](http://arxiv.org/pdf/2507.22887v1)**

> **作者:** Kwesi Cobbina; Tianyi Zhou
>
> **摘要:** In-context learning (ICL) is a critical emerging capability of large language models (LLMs), enabling few-shot learning during inference by including a few demonstrations (demos) in the prompt. However, it has been found that ICL's performance can be sensitive to the choices of demos and their order. This paper investigates an unexplored new positional bias of ICL for the first time: we observe that the predictions and accuracy can drift drastically when the positions of demos, the system prompt, and the user message in LLM input are varied. We refer to this bias as DEMOS' POSITION IN PROMPT (DPP) bias. We design a systematic evaluation pipeline to study this type of positional bias across classification, question answering, summarization, and reasoning tasks. We introduce two metrics, ACCURACY-CHANGE and PREDICTION-CHANGE, to quantify net gains and output volatility induced by changes in the demos' position. Extensive experiments on ten LLMs from four open-source model families (QWEN, LLAMA3, MISTRAL, COHERE) verify that the bias significantly affects their accuracy and predictions: placing demos at the start of the prompt yields the most stable and accurate outputs with gains of up to +6 points. In contrast, placing demos at the end of the user message flips over 30\% of predictions without improving correctness on QA tasks. Smaller models are most affected by this sensitivity, though even large models remain marginally affected on complex tasks.
>
---
#### [new 030] ControlMed: Adding Reasoning Control to Medical Language Model
- **分类: cs.CL**

- **简介: 该论文属于医疗领域的大语言模型任务，旨在解决现有推理模型生成冗长推理过程、计算开销大的问题。作者提出ControlMed，通过细粒度控制标记让用户在推理时控制推理长度，结合预训练、监督微调与强化学习提升效果，在多语言医疗任务上验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2507.22545v1](http://arxiv.org/pdf/2507.22545v1)**

> **作者:** Sung-Min Lee; Siyoon Lee; Juyeon Kim; Kyungmin Roh
>
> **备注:** 13 pages
>
> **摘要:** Reasoning Large Language Models (LLMs) with enhanced accuracy and explainability are increasingly being adopted in the medical domain, as the life-critical nature of clinical decision-making demands reliable support. Despite these advancements, existing reasoning LLMs often generate unnecessarily lengthy reasoning processes, leading to significant computational overhead and response latency. These limitations hinder their practical deployment in real-world clinical environments. To address these challenges, we introduce \textbf{ControlMed}, a medical language model that enables users to actively control the length of the reasoning process at inference time through fine-grained control markers. ControlMed is trained through a three-stage pipeline: 1) pre-training on a large-scale synthetic medical instruction dataset covering both \textit{direct} and \textit{reasoning responses}; 2) supervised fine-tuning with multi-length reasoning data and explicit length-control markers; and 3) reinforcement learning with model-based reward signals to enhance factual accuracy and response quality. Experimental results on a variety of English and Korean medical benchmarks demonstrate that our model achieves similar or better performance compared to state-of-the-art models. Furthermore, users can flexibly balance reasoning accuracy and computational efficiency by controlling the reasoning length as needed. These findings demonstrate that ControlMed is a practical and adaptable solution for clinical question answering and medical information analysis.
>
---
#### [new 031] Falcon-H1: A Family of Hybrid-Head Language Models Redefining Efficiency and Performance
- **分类: cs.CL**

- **简介: 论文提出Falcon-H1系列语言模型，结合Transformer与SSM架构，提升性能与效率，解决大模型参数多、训练成本高的问题。涵盖多种参数规模与多语言支持，适用于广泛任务。**

- **链接: [http://arxiv.org/pdf/2507.22448v1](http://arxiv.org/pdf/2507.22448v1)**

> **作者:** Jingwei Zuo; Maksim Velikanov; Ilyas Chahed; Younes Belkada; Dhia Eddine Rhayem; Guillaume Kunsch; Hakim Hacid; Hamza Yous; Brahim Farhat; Ibrahim Khadraoui; Mugariya Farooq; Giulia Campesan; Ruxandra Cojocaru; Yasser Djilali; Shi Hu; Iheb Chaabane; Puneesh Khanna; Mohamed El Amine Seddik; Ngoc Dung Huynh; Phuc Le Khac; Leen AlQadi; Billel Mokeddem; Mohamed Chami; Abdalgader Abubaker; Mikhail Lubinets; Kacper Piskorski; Slim Frikha
>
> **备注:** Technical report of Falcon-H1 model series
>
> **摘要:** In this report, we introduce Falcon-H1, a new series of large language models (LLMs) featuring hybrid architecture designs optimized for both high performance and efficiency across diverse use cases. Unlike earlier Falcon models built solely on Transformer or Mamba architectures, Falcon-H1 adopts a parallel hybrid approach that combines Transformer-based attention with State Space Models (SSMs), known for superior long-context memory and computational efficiency. We systematically revisited model design, data strategy, and training dynamics, challenging conventional practices in the field. Falcon-H1 is released in multiple configurations, including base and instruction-tuned variants at 0.5B, 1.5B, 1.5B-deep, 3B, 7B, and 34B parameters. Quantized instruction-tuned models are also available, totaling over 30 checkpoints on Hugging Face Hub. Falcon-H1 models demonstrate state-of-the-art performance and exceptional parameter and training efficiency. The flagship Falcon-H1-34B matches or outperforms models up to 70B scale, such as Qwen3-32B, Qwen2.5-72B, and Llama3.3-70B, while using fewer parameters and less data. Smaller models show similar trends: the Falcon-H1-1.5B-Deep rivals current leading 7B-10B models, and Falcon-H1-0.5B performs comparably to typical 7B models from 2024. These models excel across reasoning, mathematics, multilingual tasks, instruction following, and scientific knowledge. With support for up to 256K context tokens and 18 languages, Falcon-H1 is suitable for a wide range of applications. All models are released under a permissive open-source license, underscoring our commitment to accessible and impactful AI research.
>
---
#### [new 032] Investigating Hallucination in Conversations for Low Resource Languages
- **分类: cs.CL**

- **简介: 该论文研究低资源语言对话中的大模型幻觉问题，分析GPT-3.5、GPT-4o等模型在Hindi、Farsi和Mandarin中的事实与语言错误。任务是识别和比较不同语言中的幻觉现象，旨在提升模型在多语言场景下的可靠性。**

- **链接: [http://arxiv.org/pdf/2507.22720v1](http://arxiv.org/pdf/2507.22720v1)**

> **作者:** Amit Das; Md. Najib Hasan; Souvika Sarkar; Zheng Zhang; Fatemeh Jamshidi; Tathagata Bhattacharya; Nilanjana Raychawdhury; Dongji Feng; Vinija Jain; Aman Chadha
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable proficiency in generating text that closely resemble human writing. However, they often generate factually incorrect statements, a problem typically referred to as 'hallucination'. Addressing hallucination is crucial for enhancing the reliability and effectiveness of LLMs. While much research has focused on hallucinations in English, our study extends this investigation to conversational data in three languages: Hindi, Farsi, and Mandarin. We offer a comprehensive analysis of a dataset to examine both factual and linguistic errors in these languages for GPT-3.5, GPT-4o, Llama-3.1, Gemma-2.0, DeepSeek-R1 and Qwen-3. We found that LLMs produce very few hallucinated responses in Mandarin but generate a significantly higher number of hallucinations in Hindi and Farsi.
>
---
#### [new 033] IFEvalCode: Controlled Code Generation
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决代码大模型在遵循详细要求（如风格、结构）方面的不足。论文提出了前向与后向约束生成方法，并构建了多语言评测基准IFEvalCode，包含1.6K测试样例，评估模型在代码正确性和指令遵循方面的能力。**

- **链接: [http://arxiv.org/pdf/2507.22462v1](http://arxiv.org/pdf/2507.22462v1)**

> **作者:** Jian Yang; Wei Zhang; Shukai Liu; Linzheng Chai; Yingshui Tan; Jiaheng Liu; Ge Zhang; Wangchunshu Zhou; Guanglin Niu; Zhoujun Li; Binyuan Hui; Junyang Lin
>
> **备注:** 10 pages
>
> **摘要:** Code large language models (Code LLMs) have made significant progress in code generation by translating natural language descriptions into functional code; however, real-world applications often demand stricter adherence to detailed requirements such as coding style, line count, and structural constraints, beyond mere correctness. To address this, the paper introduces forward and backward constraints generation to improve the instruction-following capabilities of Code LLMs in controlled code generation, ensuring outputs align more closely with human-defined guidelines. The authors further present IFEvalCode, a multilingual benchmark comprising 1.6K test samples across seven programming languages (Python, Java, JavaScript, TypeScript, Shell, C++, and C#), with each sample featuring both Chinese and English queries. Unlike existing benchmarks, IFEvalCode decouples evaluation into two metrics: correctness (Corr.) and instruction-following (Instr.), enabling a more nuanced assessment. Experiments on over 40 LLMs reveal that closed-source models outperform open-source ones in controllable code generation and highlight a significant gap between the models' ability to generate correct code versus code that precisely follows instructions.
>
---
#### [new 034] PATENTWRITER: A Benchmarking Study for Patent Drafting with LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决专利撰写效率低的问题。研究提出PATENTWRITER基准框架，评估多种大语言模型在专利摘要生成中的表现，涵盖生成质量、鲁棒性及适用性。**

- **链接: [http://arxiv.org/pdf/2507.22387v1](http://arxiv.org/pdf/2507.22387v1)**

> **作者:** Homaira Huda Shomee; Suman Kalyan Maity; Sourav Medya
>
> **摘要:** Large language models (LLMs) have emerged as transformative approaches in several important fields. This paper aims for a paradigm shift for patent writing by leveraging LLMs to overcome the tedious patent-filing process. In this work, we present PATENTWRITER, the first unified benchmarking framework for evaluating LLMs in patent abstract generation. Given the first claim of a patent, we evaluate six leading LLMs -- including GPT-4 and LLaMA-3 -- under a consistent setup spanning zero-shot, few-shot, and chain-of-thought prompting strategies to generate the abstract of the patent. Our benchmark PATENTWRITER goes beyond surface-level evaluation: we systematically assess the output quality using a comprehensive suite of metrics -- standard NLP measures (e.g., BLEU, ROUGE, BERTScore), robustness under three types of input perturbations, and applicability in two downstream patent classification and retrieval tasks. We also conduct stylistic analysis to assess length, readability, and tone. Experimental results show that modern LLMs can generate high-fidelity and stylistically appropriate patent abstracts, often surpassing domain-specific baselines. Our code and dataset are open-sourced to support reproducibility and future research.
>
---
#### [new 035] SLM-SQL: An Exploration of Small Language Models for Text-to-SQL
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与数据库交互任务，旨在提升小语言模型在Text-to-SQL任务中的表现。论文通过构建新数据集、应用后训练技术和改进推理方法，显著提升小模型性能，验证其在边缘部署中的潜力。**

- **链接: [http://arxiv.org/pdf/2507.22478v1](http://arxiv.org/pdf/2507.22478v1)**

> **作者:** Lei Sheng; Shuai-Shuai Xu
>
> **备注:** 16 pages, 2 figures, work in progress
>
> **摘要:** Large language models (LLMs) have demonstrated strong performance in translating natural language questions into SQL queries (Text-to-SQL). In contrast, small language models (SLMs) ranging from 0.5B to 1.5B parameters currently underperform on Text-to-SQL tasks due to their limited logical reasoning capabilities. However, SLMs offer inherent advantages in inference speed and suitability for edge deployment. To explore their potential in Text-to-SQL applications, we leverage recent advancements in post-training techniques. Specifically, we used the open-source SynSQL-2.5M dataset to construct two derived datasets: SynSQL-Think-916K for SQL generation and SynSQL-Merge-Think-310K for SQL merge revision. We then applied supervised fine-tuning and reinforcement learning-based post-training to the SLM, followed by inference using a corrective self-consistency approach. Experimental results validate the effectiveness and generalizability of our method, SLM-SQL. On the BIRD development set, the five evaluated models achieved an average improvement of 31.4 points. Notably, the 0.5B model reached 56.87\% execution accuracy (EX), while the 1.5B model achieved 67.08\% EX. We will release our dataset, model, and code to github: https://github.com/CycloneBoy/slm_sql.
>
---
#### [new 036] Traits Run Deep: Enhancing Personality Assessment via Psychology-Guided LLM Representations and Multimodal Apparent Behaviors
- **分类: cs.CL; cs.MM**

- **简介: 该论文属于多模态性格评估任务，旨在解决传统方法难以建模性格语义和跨模态理解的问题。论文提出“Traits Run Deep”框架，结合心理学引导的提示与多模态表征，提升性格评估准确性。实验表明方法有效，在AVI Challenge 2025中排名第一。**

- **链接: [http://arxiv.org/pdf/2507.22367v1](http://arxiv.org/pdf/2507.22367v1)**

> **作者:** Jia Li; Yichao He; Jiacheng Xu; Tianhao Luo; Zhenzhen Hu; Richang Hong; Meng Wang
>
> **备注:** 8 pages, 3 figures, ACM MM 2025
>
> **摘要:** Accurate and reliable personality assessment plays a vital role in many fields, such as emotional intelligence, mental health diagnostics, and personalized education. Unlike fleeting emotions, personality traits are stable, often subconsciously leaked through language, facial expressions, and body behaviors, with asynchronous patterns across modalities. It was hard to model personality semantics with traditional superficial features and seemed impossible to achieve effective cross-modal understanding. To address these challenges, we propose a novel personality assessment framework called \textit{\textbf{Traits Run Deep}}. It employs \textit{\textbf{psychology-informed prompts}} to elicit high-level personality-relevant semantic representations. Besides, it devises a \textit{\textbf{Text-Centric Trait Fusion Network}} that anchors rich text semantics to align and integrate asynchronous signals from other modalities. To be specific, such fusion module includes a Chunk-Wise Projector to decrease dimensionality, a Cross-Modal Connector and a Text Feature Enhancer for effective modality fusion and an ensemble regression head to improve generalization in data-scarce situations. To our knowledge, we are the first to apply personality-specific prompts to guide large language models (LLMs) in extracting personality-aware semantics for improved representation quality. Furthermore, extracting and fusing audio-visual apparent behavior features further improves the accuracy. Experimental results on the AVI validation set have demonstrated the effectiveness of the proposed components, i.e., approximately a 45\% reduction in mean squared error (MSE). Final evaluations on the test set of the AVI Challenge 2025 confirm our method's superiority, ranking first in the Personality Assessment track. The source code will be made available at https://github.com/MSA-LMC/TraitsRunDeep.
>
---
#### [new 037] Question Generation for Assessing Early Literacy Reading Comprehension
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理与教育技术交叉任务，旨在解决低年级英语学习者阅读理解评估问题。作者提出一种生成理解问题的新方法，可覆盖学习内容、适应学生水平，并生成多样化问题以全面评估。他们基于FairytaleQA数据集评估了多种语言模型的表现。**

- **链接: [http://arxiv.org/pdf/2507.22410v1](http://arxiv.org/pdf/2507.22410v1)**

> **作者:** Xiaocheng Yang; Sumuk Shashidhar; Dilek Hakkani-Tur
>
> **备注:** 2 pages, 1 figure, accepted by SLaTE 2025
>
> **摘要:** Assessment of reading comprehension through content-based interactions plays an important role in the reading acquisition process. In this paper, we propose a novel approach for generating comprehension questions geared to K-2 English learners. Our method ensures complete coverage of the underlying material and adaptation to the learner's specific proficiencies, and can generate a large diversity of question types at various difficulty levels to ensure a thorough evaluation. We evaluate the performance of various language models in this framework using the FairytaleQA dataset as the source material. Eventually, the proposed approach has the potential to become an important part of autonomous AI-driven English instructors.
>
---
#### [new 038] Pre-trained Models Perform the Best When Token Distributions Follow Zipf's Law
- **分类: cs.LG; cs.CL; I.2.6; I.2.7**

- **简介: 该论文属于自然语言处理及相关序列建模任务，旨在解决如何选择最优词汇量大小的问题。通过分析词频分布是否符合齐普夫定律（Zipf's Law），提出一种合理选择词汇量的方法。实验表明，当词频分布符合齐普夫定律时，模型在多个任务中表现最佳，验证了该方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22543v1](http://arxiv.org/pdf/2507.22543v1)**

> **作者:** Yanjin He; Qingkai Zeng; Meng Jiang
>
> **摘要:** Tokenization is a fundamental step in natural language processing (NLP) and other sequence modeling domains, where the choice of vocabulary size significantly impacts model performance. Despite its importance, selecting an optimal vocabulary size remains underexplored, typically relying on heuristics or dataset-specific choices. In this work, we propose a principled method for determining the vocabulary size by analyzing token frequency distributions through Zipf's law. We show that downstream task performance correlates with how closely token distributions follow power-law behavior, and that aligning with Zipfian scaling improves both model efficiency and effectiveness. Extensive experiments across NLP, genomics, and chemistry demonstrate that models consistently achieve peak performance when the token distribution closely adheres to Zipf's law, establishing Zipfian alignment as a robust and generalizable criterion for vocabulary size selection.
>
---
#### [new 039] Explainability Through Systematicity: The Hard Systematicity Challenge for Artificial Intelligence
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 论文探讨人工智能的“系统性”问题，超越传统的“系统性挑战”，提出“硬系统性挑战”。它分析系统性在思想整合、一致性与科学性中的作用，区分四种系统性概念，缓解系统性与联结主义间的矛盾，并提出五个系统化原则评估AI模型，明确系统化需求的动态标准。任务属AI理论与哲学分析。**

- **链接: [http://arxiv.org/pdf/2507.22197v1](http://arxiv.org/pdf/2507.22197v1)**

> **作者:** Matthieu Queloz
>
> **备注:** 39 pages; final, published version
>
> **摘要:** This paper argues that explainability is only one facet of a broader ideal that shapes our expectations towards artificial intelligence (AI). Fundamentally, the issue is to what extent AI exhibits systematicity--not merely in being sensitive to how thoughts are composed of recombinable constituents, but in striving towards an integrated body of thought that is consistent, coherent, comprehensive, and parsimoniously principled. This richer conception of systematicity has been obscured by the long shadow of the "systematicity challenge" to connectionism, according to which network architectures are fundamentally at odds with what Fodor and colleagues termed "the systematicity of thought." I offer a conceptual framework for thinking about "the systematicity of thought" that distinguishes four senses of the phrase. I use these distinctions to defuse the perceived tension between systematicity and connectionism and show that the conception of systematicity that historically shaped our sense of what makes thought rational, authoritative, and scientific is more demanding than the Fodorian notion. To determine whether we have reason to hold AI models to this ideal of systematicity, I then argue, we must look to the rationales for systematization and explore to what extent they transfer to AI models. I identify five such rationales and apply them to AI. This brings into view the "hard systematicity challenge." However, the demand for systematization itself needs to be regulated by the rationales for systematization. This yields a dynamic understanding of the need to systematize thought, which tells us how systematic we need AI models to be and when.
>
---
#### [new 040] CoEx -- Co-evolving World-model and Exploration
- **分类: cs.AI; cs.CL**

- **简介: 论文提出CoEx架构，属于智能体规划与探索任务。解决现有LLM代理依赖静态世界模型导致计划偏离真实世界的问题。工作是设计分层状态抽象结构，使LLM规划与动态更新的世界模型协同演化，并通过文本推理与符号记忆持续整合经验，提升复杂任务表现。**

- **链接: [http://arxiv.org/pdf/2507.22281v1](http://arxiv.org/pdf/2507.22281v1)**

> **作者:** Minsoo Kim; Seung-won Hwang
>
> **摘要:** Planning in modern LLM agents relies on the utilization of LLM as an internal world model, acquired during pretraining. However, existing agent designs fail to effectively assimilate new observations into dynamic updates of the world model. This reliance on the LLM's static internal world model is progressively prone to misalignment with the underlying true state of the world, leading to the generation of divergent and erroneous plans. We introduce a hierarchical agent architecture, CoEx, in which hierarchical state abstraction allows LLM planning to co-evolve with a dynamically updated model of the world. CoEx plans and interacts with the world by using LLM reasoning to orchestrate dynamic plans consisting of subgoals, and its learning mechanism continuously incorporates these subgoal experiences into a persistent world model in the form of a neurosymbolic belief state, comprising textual inferences and code-based symbolic memory. We evaluate our agent across a diverse set of agent scenarios involving rich environments and complex tasks including ALFWorld, PDDL, and Jericho. Our experiments show that CoEx outperforms existing agent paradigms in planning and exploration.
>
---
#### [new 041] RecGPT Technical Report
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于推荐系统任务，旨在解决传统推荐系统过度依赖历史行为数据导致的用户兴趣建模不足、推荐同质化等问题。论文提出RecGPT框架，将大语言模型引入推荐流程，以用户意图为中心，提升推荐多样性与满意度，并已在淘宝部署取得良好效果。**

- **链接: [http://arxiv.org/pdf/2507.22879v1](http://arxiv.org/pdf/2507.22879v1)**

> **作者:** Chao Yi; Dian Chen; Gaoyang Guo; Jiakai Tang; Jian Wu; Jing Yu; Sunhao Dai; Wen Chen; Wenjun Yang; Yuning Jiang; Zhujin Gao; Bo Zheng; Chi Li; Dimin Wang; Dixuan Wang; Fan Li; Fan Zhang; Haibin Chen; Haozhuang Liu; Jialin Zhu; Jiamang Wang; Jiawei Wu; Jin Cui; Ju Huang; Kai Zhang; Kan Liu; Lang Tian; Liang Rao; Longbin Li; Lulu Zhao; Mao Zhang; Na He; Peiyang Wang; Qiqi Huang; Tao Luo; Wenbo Su; Xiaoxiao He; Xin Tong; Xu Chen; Xunke Xi; Yang Li; Yaxuan Wu; Yeqiu Yang; Yi Hu; Yinnan Song; Yuchen Li; Yujie Luo; Yujin Yuan; Yuliang Yan; Zhengyang Wang; Zhibo Xiao; Zhixin Ma; Zile Zhou
>
> **摘要:** Recommender systems are among the most impactful applications of artificial intelligence, serving as critical infrastructure connecting users, merchants, and platforms. However, most current industrial systems remain heavily reliant on historical co-occurrence patterns and log-fitting objectives, i.e., optimizing for past user interactions without explicitly modeling user intent. This log-fitting approach often leads to overfitting to narrow historical preferences, failing to capture users' evolving and latent interests. As a result, it reinforces filter bubbles and long-tail phenomena, ultimately harming user experience and threatening the sustainability of the whole recommendation ecosystem. To address these challenges, we rethink the overall design paradigm of recommender systems and propose RecGPT, a next-generation framework that places user intent at the center of the recommendation pipeline. By integrating large language models (LLMs) into key stages of user interest mining, item retrieval, and explanation generation, RecGPT transforms log-fitting recommendation into an intent-centric process. To effectively align general-purpose LLMs to the above domain-specific recommendation tasks at scale, RecGPT incorporates a multi-stage training paradigm, which integrates reasoning-enhanced pre-alignment and self-training evolution, guided by a Human-LLM cooperative judge system. Currently, RecGPT has been fully deployed on the Taobao App. Online experiments demonstrate that RecGPT achieves consistent performance gains across stakeholders: users benefit from increased content diversity and satisfaction, merchants and the platform gain greater exposure and conversions. These comprehensive improvement results across all stakeholders validates that LLM-driven, intent-centric design can foster a more sustainable and mutually beneficial recommendation ecosystem.
>
---
#### [new 042] Prompt Optimization and Evaluation for LLM Automated Red Teaming
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于自然语言处理与安全评估任务，旨在解决大型语言模型应用中的系统漏洞识别问题。通过优化攻击生成器提示，利用攻击成功率（ASR）和多次实验测得攻击的可发现性，从而提升自动化红队攻击的效果与评估精度。**

- **链接: [http://arxiv.org/pdf/2507.22133v1](http://arxiv.org/pdf/2507.22133v1)**

> **作者:** Michael Freenor; Lauren Alvarez; Milton Leal; Lily Smith; Joel Garrett; Yelyzaveta Husieva; Madeline Woodruff; Ryan Miller; Erich Kummerfeld; Rafael Medeiros; Sander Schulhoff
>
> **备注:** 9 pages, 5 Figures, and 1 Appendix item
>
> **摘要:** Applications that use Large Language Models (LLMs) are becoming widespread, making the identification of system vulnerabilities increasingly important. Automated Red Teaming accelerates this effort by using an LLM to generate and execute attacks against target systems. Attack generators are evaluated using the Attack Success Rate (ASR) the sample mean calculated over the judgment of success for each attack. In this paper, we introduce a method for optimizing attack generator prompts that applies ASR to individual attacks. By repeating each attack multiple times against a randomly seeded target, we measure an attack's discoverability the expectation of the individual attack success. This approach reveals exploitable patterns that inform prompt optimization, ultimately enabling more robust evaluation and refinement of generators.
>
---
#### [new 043] CIMR: Contextualized Iterative Multimodal Reasoning for Robust Instruction Following in LVLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出CIMR框架，用于提升大型视觉语言模型（LVLMs）在复杂多模态指令跟随任务中的推理与纠错能力。任务是多模态动作规划（MAP），旨在解决现有模型在逻辑推理、反馈整合和迭代修正方面的不足。通过迭代推理与动态融合模块，CIMR在VIT数据集上微调后表现出更高的准确率。**

- **链接: [http://arxiv.org/pdf/2507.22074v1](http://arxiv.org/pdf/2507.22074v1)**

> **作者:** Yangshu Yuan; Heng Chen; Xinyi Jiang; Christian Ng; Kexin Qiu
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) has enhanced our ability to process and generate human language and visual information. However, these models often struggle with complex, multi-step multi-modal instructions that require logical reasoning, dynamic feedback integration, and iterative self-correction. To address this, we propose CIMR: Contextualized Iterative Multimodal Reasoning, a novel framework that introduces a context-aware iterative reasoning and self-correction module. CIMR operates in two stages: initial reasoning and response generation, followed by iterative refinement using parsed multi-modal feedback. A dynamic fusion module deeply integrates textual, visual, and contextual features at each step. We fine-tune LLaVA-1.5-7B on the Visual Instruction Tuning (VIT) dataset and evaluate CIMR on the newly introduced Multi-modal Action Planning (MAP) dataset. CIMR achieves 91.5% accuracy, outperforming state-of-the-art models such as GPT-4V (89.2%), LLaVA-1.5 (78.5%), MiniGPT-4 (75.3%), and InstructBLIP (72.8%), demonstrating the efficacy of its iterative reasoning and self-correction capabilities in complex tasks.
>
---
#### [new 044] CodeEvo: Interaction-Driven Synthesis of Code-centric Data through Hybrid and Iterative Feedback
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 论文提出CodeEvo，用于代码生成的高质量指令-代码对合成框架。通过两个LLM代理（Coder和Reviewer）的迭代交互，结合编译器验证与生成反馈，解决现有方法数据质量低、缺乏验证的问题，提升代码生成效果。**

- **链接: [http://arxiv.org/pdf/2507.22080v1](http://arxiv.org/pdf/2507.22080v1)**

> **作者:** Qiushi Sun; Jinyang Gong; Lei Li; Qipeng Guo; Fei Yuan
>
> **备注:** Work in progress
>
> **摘要:** Acquiring high-quality instruction-code pairs is essential for training Large Language Models (LLMs) for code generation. Manually curated data is expensive and inherently limited in scale, motivating the development of code-centric synthesis methods. Yet, current approaches either focus on augmenting existing code or rely on predefined heuristics, both lacking rigorous data validation, which results in synthetic data that is ungrounded, repetitive, or overly simplistic. Inspired by collaborative programming practices, we propose CodeEvo, a framework that synthesizes code data through iterative interactions between two LLM agents: a Coder, which generates candidate code and test cases based on given instructions, and a Reviewer, which guides the synthesis process by producing new instructions and feedback. We further introduce a hybrid feedback mechanism that combines compiler determinism with the generative flexibility of agents, enabling automatic quality control throughout synthesis. Extensive experiments demonstrate that models fine-tuned on CodeEvo data significantly outperform established baselines across code generation benchmarks with various difficulties. In-depth analyses further provide insights from multiple perspectives into effective code-centric data synthesis.
>
---
#### [new 045] Next Tokens Denoising for Speech Synthesis
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文提出Dragon-FM模型，用于语音合成任务。为解决自回归模型生成慢和扩散模型缓存难的问题，结合两者优势，采用分块自回归与块内流匹配方法，实现高效高质量语音生成。**

- **链接: [http://arxiv.org/pdf/2507.22746v1](http://arxiv.org/pdf/2507.22746v1)**

> **作者:** Yanqing Liu; Ruiqing Xue; Chong Zhang; Yufei Liu; Gang Wang; Bohan Li; Yao Qian; Lei He; Shujie Liu; Sheng Zhao
>
> **摘要:** While diffusion and autoregressive (AR) models have significantly advanced generative modeling, they each present distinct limitations. AR models, which rely on causal attention, cannot exploit future context and suffer from slow generation speeds. Conversely, diffusion models struggle with key-value (KV) caching. To overcome these challenges, we introduce Dragon-FM, a novel text-to-speech (TTS) design that unifies AR and flow-matching. This model processes 48 kHz audio codec tokens in chunks at a compact 12.5 tokens per second rate. This design enables AR modeling across chunks, ensuring global coherence, while parallel flow-matching within chunks facilitates fast iterative denoising. Consequently, the proposed model can utilize KV-cache across chunks and incorporate future context within each chunk. Furthermore, it bridges continuous and discrete feature modeling, demonstrating that continuous AR flow-matching can predict discrete tokens with finite scalar quantizers. This efficient codec and fast chunk-autoregressive architecture also makes the proposed model particularly effective for generating extended content. Experiment for demos of our work} on podcast datasets demonstrate its capability to efficiently generate high-quality zero-shot podcasts.
>
---
#### [new 046] GeoOutageKG: A Multimodal Geospatiotemporal Knowledge Graph for Multiresolution Power Outage Analysis
- **分类: cs.IR; cs.CL; cs.CY**

- **简介: 该论文属于知识图谱与电力系统分析任务，旨在解决现有停电数据空间分辨率低、难以捕捉局部模式的问题。作者构建了GeoOutageKG，一个融合夜间灯光卫星图像、高分辨率停电地图和县级停电报告的多模态时空知识图谱，支持多分辨率停电分析，提升停电检测与预测能力。**

- **链接: [http://arxiv.org/pdf/2507.22878v1](http://arxiv.org/pdf/2507.22878v1)**

> **作者:** Ethan Frakes; Yinghui Wu; Roger H. French; Mengjie Li
>
> **备注:** Accepted to the 24th International Semantic Web Conference Resource Track (ISWC 2025)
>
> **摘要:** Detecting, analyzing, and predicting power outages is crucial for grid risk assessment and disaster mitigation. Numerous outages occur each year, exacerbated by extreme weather events such as hurricanes. Existing outage data are typically reported at the county level, limiting their spatial resolution and making it difficult to capture localized patterns. However, it offers excellent temporal granularity. In contrast, nighttime light satellite image data provides significantly higher spatial resolution and enables a more comprehensive spatial depiction of outages, enhancing the accuracy of assessing the geographic extent and severity of power loss after disaster events. However, these satellite data are only available on a daily basis. Integrating spatiotemporal visual and time-series data sources into a unified knowledge representation can substantially improve power outage detection, analysis, and predictive reasoning. In this paper, we propose GeoOutageKG, a multimodal knowledge graph that integrates diverse data sources, including nighttime light satellite image data, high-resolution spatiotemporal power outage maps, and county-level timeseries outage reports in the U.S. We describe our method for constructing GeoOutageKG by aligning source data with a developed ontology, GeoOutageOnto. Currently, GeoOutageKG includes over 10.6 million individual outage records spanning from 2014 to 2024, 300,000 NTL images spanning from 2012 to 2024, and 15,000 outage maps. GeoOutageKG is a novel, modular and reusable semantic resource that enables robust multimodal data integration. We demonstrate its use through multiresolution analysis of geospatiotemporal power outages.
>
---
#### [new 047] Efficient Differentially Private Fine-Tuning of LLMs via Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 论文提出RLDP框架，通过强化学习解决大型语言模型微调中的差分隐私优化问题，旨在平衡隐私保护与模型性能。任务属自然语言处理与隐私保护交叉领域，解决传统DP-SGD方法效率低、控制僵化的问题，实现更优的隐私预算分配和训练加速。**

- **链接: [http://arxiv.org/pdf/2507.22565v1](http://arxiv.org/pdf/2507.22565v1)**

> **作者:** Afshin Khadangi; Amir Sartipi; Igor Tchappi; Ramin Bahmani; Gilbert Fridgen
>
> **摘要:** The tension between data privacy and model utility has become the defining bottleneck for the practical deployment of large language models (LLMs) trained on sensitive corpora including healthcare. Differentially private stochastic gradient descent (DP-SGD) guarantees formal privacy, yet it does so at a pronounced cost: gradients are forcibly clipped and perturbed with noise, degrading sample efficiency and final accuracy. Numerous variants have been proposed to soften this trade-off, but they all share a handicap: their control knobs are hard-coded, global, and oblivious to the evolving optimization landscape. Consequently, practitioners are forced either to over-spend privacy budget in pursuit of utility, or to accept mediocre models in order to stay within privacy constraints. We present RLDP, the first framework to cast DP optimization itself as a closed-loop control problem amenable to modern deep reinforcement learning (RL). RLDP continuously senses rich statistics of the learning dynamics and acts by selecting fine-grained per parameter gradient-clipping thresholds as well as the magnitude of injected Gaussian noise. A soft actor-critic (SAC) hyper-policy is trained online during language model fine-tuning; it learns, from scratch, how to allocate the privacy budget where it matters and when it matters. Across more than 1,600 ablation experiments on GPT2-small, Llama-1B, Llama-3B, and Mistral-7B, RLDP delivers perplexity reductions of 1.3-30.5% (mean 5.4%) and an average 5.6% downstream utility gain. RLDP reaches each baseline's final utility after only 13-43% of the gradient-update budget (mean speed-up 71%), all while honoring the same ($\epsilon$, $\delta$)-DP contract and exhibiting equal or lower susceptibility to membership-inference and canary-extraction attacks.
>
---
#### [new 048] LLM-Crowdsourced: A Benchmark-Free Paradigm for Mutual Evaluation of Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大语言模型（LLM）评估任务，旨在解决现有评估方法存在的数据污染、黑箱操作和主观偏好等问题。作者提出了一种无需基准数据集的评估范式LLM-Crowdsourced，通过让LLMs自动生成问题、独立回答并互相评价，实现动态、透明、客观且专业的评估，并验证了其在数学与编程任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.22359v1](http://arxiv.org/pdf/2507.22359v1)**

> **作者:** Qianhong Guo; Wei Xie; Xiaofang Cai; Enze Wang; Shuoyoucheng Ma; Kai Chen; Xiaofeng Wang; Baosheng Wang
>
> **摘要:** Although large language models (LLMs) demonstrate remarkable capabilities across various tasks, evaluating their capabilities remains a challenging task. Existing evaluation methods suffer from issues such as data contamination, black-box operation, and subjective preference. These issues make it difficult to evaluate the LLMs' true capabilities comprehensively. To tackle these challenges, we propose a novel benchmark-free evaluation paradigm, LLM-Crowdsourced. It utilizes LLMs to generate questions, answer independently, and evaluate mutually. This method integrates four key evaluation criteria: dynamic, transparent, objective, and professional, which existing evaluation methods cannot satisfy simultaneously. Experiments on eight mainstream LLMs across mathematics and programming verify the advantages of our method in distinguishing LLM performance. Furthermore, our study reveals several novel findings that are difficult for traditional methods to detect, including but not limited to: (1) Gemini demonstrates the highest original and professional question-design capabilities among others; (2) Some LLMs exhibit ''memorization-based answering'' by misrecognizing questions as familiar ones with a similar structure; (3) LLM evaluation results demonstrate high consistency (robustness).
>
---
#### [new 049] VL-Cogito: Progressive Curriculum Reinforcement Learning for Advanced Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决现有模型在不同领域和难度任务中表现不稳定的问题。作者提出了VL-Cogito模型，基于多阶段渐进课程强化学习（PCuRL）框架，引入难度软加权和动态长度奖励机制，提升模型在数学、科学、逻辑等多模态任务中的推理能力与效率。**

- **链接: [http://arxiv.org/pdf/2507.22607v1](http://arxiv.org/pdf/2507.22607v1)**

> **作者:** Ruifeng Yuan; Chenghao Xiao; Sicong Leng; Jianyu Wang; Long Li; Weiwen Xu; Hou Pong Chan; Deli Zhao; Tingyang Xu; Zhongyu Wei; Hao Zhang; Yu Rong
>
> **备注:** 21 pages, 5 figures, 6 tables. Work in progress
>
> **摘要:** Reinforcement learning has proven its effectiveness in enhancing the reasoning capabilities of large language models. Recent research efforts have progressively extended this paradigm to multimodal reasoning tasks. Due to the inherent complexity and diversity of multimodal tasks, especially in semantic content and problem formulations, existing models often exhibit unstable performance across various domains and difficulty levels. To address these limitations, we propose VL-Cogito, an advanced multimodal reasoning model trained via a novel multi-stage Progressive Curriculum Reinforcement Learning (PCuRL) framework. PCuRL systematically guides the model through tasks of gradually increasing difficulty, substantially improving its reasoning abilities across diverse multimodal contexts. The framework introduces two key innovations: (1) an online difficulty soft weighting mechanism, dynamically adjusting training difficulty across successive RL training stages; and (2) a dynamic length reward mechanism, which encourages the model to adaptively regulate its reasoning path length according to task complexity, thus balancing reasoning efficiency with correctness. Experimental evaluations demonstrate that VL-Cogito consistently matches or surpasses existing reasoning-oriented models across mainstream multimodal benchmarks spanning mathematics, science, logic, and general understanding, validating the effectiveness of our approach.
>
---
#### [new 050] The Incomplete Bridge: How AI Research (Mis)Engages with Psychology
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文分析了1,006篇AI顶会论文及引用的2,544篇心理学文献，旨在探讨人工智能与心理学的跨学科融合情况。任务是绘制二者互动的图谱，识别心理学理论的应用模式与误用问题，并指导更有效的整合，以促进AI系统设计与理解。**

- **链接: [http://arxiv.org/pdf/2507.22847v1](http://arxiv.org/pdf/2507.22847v1)**

> **作者:** Han Jiang; Pengda Wang; Xiaoyuan Yi; Xing Xie; Ziang Xiao
>
> **摘要:** Social sciences have accumulated a rich body of theories and methodologies for investigating the human mind and behaviors, while offering valuable insights into the design and understanding of Artificial Intelligence (AI) systems. Focusing on psychology as a prominent case, this study explores the interdisciplinary synergy between AI and the field by analyzing 1,006 LLM-related papers published in premier AI venues between 2023 and 2025, along with the 2,544 psychology publications they cite. Through our analysis, we identify key patterns of interdisciplinary integration, locate the psychology domains most frequently referenced, and highlight areas that remain underexplored. We further examine how psychology theories/frameworks are operationalized and interpreted, identify common types of misapplication, and offer guidance for more effective incorporation. Our work provides a comprehensive map of interdisciplinary engagement between AI and psychology, thereby facilitating deeper collaboration and advancing AI systems.
>
---
#### [new 051] Strategic Deflection: Defending LLMs from Logit Manipulation
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的安全防御任务，旨在解决大语言模型遭受的logit级攻击问题。作者提出了“策略性偏转”方法，使模型在面对攻击时生成语义相关但无害的回答，从而降低攻击成功率，同时保持对正常请求的响应性能。**

- **链接: [http://arxiv.org/pdf/2507.22160v1](http://arxiv.org/pdf/2507.22160v1)**

> **作者:** Yassine Rachidy; Jihad Rbaiti; Youssef Hmamouche; Faissal Sehbaoui; Amal El Fallah Seghrouchni
>
> **备注:** 20 pages
>
> **摘要:** With the growing adoption of Large Language Models (LLMs) in critical areas, ensuring their security against jailbreaking attacks is paramount. While traditional defenses primarily rely on refusing malicious prompts, recent logit-level attacks have demonstrated the ability to bypass these safeguards by directly manipulating the token-selection process during generation. We introduce Strategic Deflection (SDeflection), a defense that redefines the LLM's response to such advanced attacks. Instead of outright refusal, the model produces an answer that is semantically adjacent to the user's request yet strips away the harmful intent, thereby neutralizing the attacker's harmful intent. Our experiments demonstrate that SDeflection significantly lowers Attack Success Rate (ASR) while maintaining model performance on benign queries. This work presents a critical shift in defensive strategies, moving from simple refusal to strategic content redirection to neutralize advanced threats.
>
---
## 更新

#### [replaced 001] ECG-Byte: A Tokenizer for End-to-End Generative Electrocardiogram Language Modeling
- **分类: cs.CL; eess.SP; I.2.7; J.3**

- **链接: [http://arxiv.org/pdf/2412.14373v3](http://arxiv.org/pdf/2412.14373v3)**

> **作者:** William Han; Chaojing Duan; Michael A. Rosenberg; Emerson Liu; Ding Zhao
>
> **备注:** 38 pages, 9 figures; Accepted to MLHC 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated exceptional versatility across domains, including applications to electrocardiograms (ECGs). A growing body of work focuses on generating text from multi-channeled ECG signals and corresponding textual prompts. Existing approaches often involve a two-stage process: pretraining an ECG-specific encoder with a self-supervised learning (SSL) objective, followed by finetuning an LLM for natural language generation (NLG) using encoder-derived features. However, these methods face two key limitations: inefficiency due to multi-stage training and challenges in interpreting encoder-generated features. To overcome these issues, we propose ECG-Byte, an adapted byte pair encoding (BPE) tokenizer pipeline for autoregressive language modeling of ECGs. ECG-Byte compresses and encodes ECG signals into tokens, enabling direct end-to-end LLM training by combining ECG and text tokens. This approach enhances interpretability, as ECG tokens can be directly mapped back to the original signals. Leveraging ECG-Byte, we achieve competitive NLG performance while training 3 times faster and using just 48\% of the data required by traditional two-stage methods.
>
---
#### [replaced 002] Natural Language Processing for the Legal Domain: A Survey of Tasks, Datasets, Models, and Challenges
- **分类: cs.CL; cs.AI; A.1; I.2.7; J.1**

- **链接: [http://arxiv.org/pdf/2410.21306v3](http://arxiv.org/pdf/2410.21306v3)**

> **作者:** Farid Ariai; Joel Mackenzie; Gianluca Demartini
>
> **备注:** 35 pages
>
> **摘要:** Natural Language Processing (NLP) is revolutionising the way both professionals and laypersons operate in the legal field. The considerable potential for NLP in the legal sector, especially in developing computational assistance tools for various legal processes, has captured the interest of researchers for years. This survey follows the Preferred Reporting Items for Systematic Reviews and Meta-Analyses framework, reviewing 154 studies, with a final selection of 131 after manual filtering. It explores foundational concepts related to NLP in the legal domain, illustrating the unique aspects and challenges of processing legal texts, such as extensive document lengths, complex language, and limited open legal datasets. We provide an overview of NLP tasks specific to legal text, such as Document Summarisation, Named Entity Recognition, Question Answering, Argument Mining, Text Classification, and Judgement Prediction. Furthermore, we analyse both developed legal-oriented language models, and approaches for adapting general-purpose language models to the legal domain. Additionally, we identify sixteen open research challenges, including the detection and mitigation of bias in artificial intelligence applications, the need for more robust and interpretable models, and improving explainability to handle the complexities of legal language and reasoning.
>
---
#### [replaced 003] FRED: Financial Retrieval-Enhanced Detection and Editing of Hallucinations in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.20930v2](http://arxiv.org/pdf/2507.20930v2)**

> **作者:** Likun Tan; Kuan-Wei Huang; Kevin Wu
>
> **摘要:** Hallucinations in large language models pose a critical challenge for applications requiring factual reliability, particularly in high-stakes domains such as finance. This work presents an effective approach for detecting and editing factually incorrect content in model-generated responses based on the provided context. Given a user-defined domain-specific error taxonomy, we construct a synthetic dataset by inserting tagged errors into financial question-answering corpora and then fine-tune four language models, Phi-4, Phi-4-mini, Qwen3-4B, and Qwen3-14B, to detect and edit these factual inaccuracies. Our best-performing model, fine-tuned Phi-4, achieves an 8% improvement in binary F1 score and a 30% gain in overall detection performance compared to OpenAI-o3. Notably, our fine-tuned Phi-4-mini model, despite having only 4 billion parameters, maintains competitive performance with just a 2% drop in binary detection and a 0.1% decline in overall detection compared to OpenAI-o3. Our work provides a practical solution for detecting and editing factual inconsistencies in financial text generation while introducing a generalizable framework that can enhance the trustworthiness and alignment of large language models across diverse applications beyond finance. Our code and data are available at https://github.com/pegasi-ai/shield.
>
---
#### [replaced 004] Exploring the Frontier of Vision-Language Models: A Survey of Current Methodologies and Future Directions
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2404.07214v3](http://arxiv.org/pdf/2404.07214v3)**

> **作者:** Akash Ghosh; Arkadeep Acharya; Sriparna Saha; Vinija Jain; Aman Chadha
>
> **备注:** One of the first survey on Visual Language Models
>
> **摘要:** The advent of Large Language Models (LLMs) has significantly reshaped the trajectory of the AI revolution. Nevertheless, these LLMs exhibit a notable limitation, as they are primarily adept at processing textual information. To address this constraint, researchers have endeavored to integrate visual capabilities with LLMs, resulting in the emergence of Vision-Language Models (VLMs). These advanced models are instrumental in tackling more intricate tasks such as image captioning and visual question answering. In our comprehensive survey paper, we delve into the key advancements within the realm of VLMs. Our classification organizes VLMs into three distinct categories: models dedicated to vision-language understanding, models that process multimodal inputs to generate unimodal (textual) outputs and models that both accept and produce multimodal inputs and outputs.This classification is based on their respective capabilities and functionalities in processing and generating various modalities of data.We meticulously dissect each model, offering an extensive analysis of its foundational architecture, training data sources, as well as its strengths and limitations wherever possible, providing readers with a comprehensive understanding of its essential components. We also analyzed the performance of VLMs in various benchmark datasets. By doing so, we aim to offer a nuanced understanding of the diverse landscape of VLMs. Additionally, we underscore potential avenues for future research in this dynamic domain, anticipating further breakthroughs and advancements.
>
---
#### [replaced 005] GneissWeb: Preparing High Quality Data for LLMs at Scale
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.14907v2](http://arxiv.org/pdf/2502.14907v2)**

> **作者:** Hajar Emami Gohari; Swanand Ravindra Kadhe; Syed Yousaf Shah; Constantin Adam; Abdulhamid Adebayo; Praneet Adusumilli; Farhan Ahmed; Nathalie Baracaldo Angel; Santosh Subhashrao Borse; Yuan-Chi Chang; Xuan-Hong Dang; Nirmit Desai; Revital Eres; Ran Iwamoto; Alexei Karve; Yan Koyfman; Wei-Han Lee; Changchang Liu; Boris Lublinsky; Takuyo Ohko; Pablo Pesce; Maroun Touma; Shiqiang Wang; Shalisha Witherspoon; Herbert Woisetschläger; David Wood; Kun-Lung Wu; Issei Yoshida; Syed Zawad; Petros Zerfos; Yi Zhou; Bishwaranjan Bhattacharjee
>
> **摘要:** Data quantity and quality play a vital role in determining the performance of Large Language Models (LLMs). High-quality data, in particular, can significantly boost the LLM's ability to generalize on a wide range of downstream tasks. Large pre-training datasets for leading LLMs remain inaccessible to the public, whereas many open datasets are small in size (less than 5 trillion tokens), limiting their suitability for training large models. In this paper, we introduce GneissWeb, a large dataset yielding around 10 trillion tokens that caters to the data quality and quantity requirements of training LLMs. Our GneissWeb recipe that produced the dataset consists of sharded exact sub-string deduplication and a judiciously constructed ensemble of quality filters. GneissWeb achieves a favorable trade-off between data quality and quantity, producing models that outperform models trained on state-of-the-art open large datasets (5+ trillion tokens). We show that models trained using GneissWeb dataset outperform those trained on FineWeb-V1.1.0 by 2.73 percentage points in terms of average score computed on a set of 11 commonly used benchmarks (both zero-shot and few-shot) for pre-training dataset evaluation. When the evaluation set is extended to 20 benchmarks (both zero-shot and few-shot), models trained using GneissWeb still achieve a 1.75 percentage points advantage over those trained on FineWeb-V1.1.0.
>
---
#### [replaced 006] Spatial Language Likelihood Grounding Network for Bayesian Fusion of Human-Robot Observations
- **分类: cs.RO; cs.CL; cs.IT; cs.LG; cs.SY; eess.SY; math.IT**

- **链接: [http://arxiv.org/pdf/2507.19947v2](http://arxiv.org/pdf/2507.19947v2)**

> **作者:** Supawich Sitdhipol; Waritwong Sukprasongdee; Ekapol Chuangsuwanich; Rina Tse
>
> **备注:** Accepted to the 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC); Supplementary video: https://cu-asl.github.io/fp-lgn/
>
> **摘要:** Fusing information from human observations can help robots overcome sensing limitations in collaborative tasks. However, an uncertainty-aware fusion framework requires a grounded likelihood representing the uncertainty of human inputs. This paper presents a Feature Pyramid Likelihood Grounding Network (FP-LGN) that grounds spatial language by learning relevant map image features and their relationships with spatial relation semantics. The model is trained as a probability estimator to capture aleatoric uncertainty in human language using three-stage curriculum learning. Results showed that FP-LGN matched expert-designed rules in mean Negative Log-Likelihood (NLL) and demonstrated greater robustness with lower standard deviation. Collaborative sensing results demonstrated that the grounded likelihood successfully enabled uncertainty-aware fusion of heterogeneous human language observations and robot sensor measurements, achieving significant improvements in human-robot collaborative task performance.
>
---
#### [replaced 007] SpatialViz-Bench: Automatically Generated Spatial Visualization Reasoning Tasks for MLLMs
- **分类: cs.CV; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2507.07610v3](http://arxiv.org/pdf/2507.07610v3)**

> **作者:** Siting Wang; Luoyang Sun; Cheng Deng; Kun Shao; Minnan Pei; Zheng Tian; Haifeng Zhang; Jun Wang
>
> **摘要:** Humans can directly imagine and manipulate visual images in their minds, a capability known as spatial visualization. While multi-modal Large Language Models (MLLMs) support imagination-based reasoning, spatial visualization remains insufficiently evaluated, typically embedded within broader mathematical and logical assessments. Existing evaluations often rely on IQ tests or math competitions that may overlap with training data, compromising assessment reliability. To this end, we introduce SpatialViz-Bench, a comprehensive multi-modal benchmark for spatial visualization with 12 tasks across 4 sub-abilities, comprising 1,180 automatically generated problems. Our evaluation of 33 state-of-the-art MLLMs not only reveals wide performance variations and demonstrates the benchmark's strong discriminative power, but also uncovers counter-intuitive findings: models show difficulty perception misaligned with human intuition, exhibit dramatic 2Dto-3D performance cliffs, default to formulaic derivation over visualization, and paradoxically suffer performance degradation from Chain-of-Thought prompting in open-source models. Through statistical and qualitative analysis of error types, SpatialViz-Bench demonstrates that state-of-the-art MLLMs continue to exhibit deficiencies in spatial visualization tasks, thereby addressing a significant lacuna in the field. The benchmark data and evaluation code are publicly available.
>
---
#### [replaced 008] ReverBERT: A State Space Model for Efficient Text-Driven Speech Style Transfer
- **分类: cs.GR; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.20992v2](http://arxiv.org/pdf/2503.20992v2)**

> **作者:** Michael Brown; Sofia Martinez; Priya Singh
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship and affiliation
>
> **摘要:** Text-driven speech style transfer aims to mold the intonation, pace, and timbre of a spoken utterance to match stylistic cues from text descriptions. While existing methods leverage large-scale neural architectures or pre-trained language models, the computational costs often remain high. In this paper, we present \emph{ReverBERT}, an efficient framework for text-driven speech style transfer that draws inspiration from a state space model (SSM) paradigm, loosely motivated by the image-based method of Wang and Liu~\cite{wang2024stylemamba}. Unlike image domain techniques, our method operates in the speech space and integrates a discrete Fourier transform of latent speech features to enable smooth and continuous style modulation. We also propose a novel \emph{Transformer-based SSM} layer for bridging textual style descriptors with acoustic attributes, dramatically reducing inference time while preserving high-quality speech characteristics. Extensive experiments on benchmark speech corpora demonstrate that \emph{ReverBERT} significantly outperforms baselines in terms of naturalness, expressiveness, and computational efficiency. We release our model and code publicly to foster further research in text-driven speech style transfer.
>
---
#### [replaced 009] Modeling Story Expectations to Understand Engagement: A Generative Framework Using LLMs
- **分类: cs.CL; cs.AI; econ.GN; q-fin.EC; stat.ME; 68T50, 91F20; H.3.1; I.2.7**

- **链接: [http://arxiv.org/pdf/2412.15239v3](http://arxiv.org/pdf/2412.15239v3)**

> **作者:** Hortense Fong; George Gui
>
> **摘要:** Understanding when and why consumers engage with stories is crucial for content creators and platforms. While existing theories suggest that audience beliefs of what is going to happen should play an important role in engagement decisions, empirical work has mostly focused on developing techniques to directly extract features from actual content, rather than capturing forward-looking beliefs, due to the lack of a principled way to model such beliefs in unstructured narrative data. To complement existing feature extraction techniques, this paper introduces a novel framework that leverages large language models to model audience forward-looking beliefs about how stories might unfold. Our method generates multiple potential continuations for each story and extracts features related to expectations, uncertainty, and surprise using established content analysis techniques. Applying our method to over 30,000 book chapters, we demonstrate that our framework complements existing feature engineering techniques by amplifying their marginal explanatory power on average by 31%. The results reveal that different types of engagement-continuing to read, commenting, and voting-are driven by distinct combinations of current and anticipated content features. Our framework provides a novel way to study and explore how audience forward-looking beliefs shape their engagement with narrative media, with implications for marketing strategy in content-focused industries.
>
---
#### [replaced 010] Instruction-tuned Large Language Models for Machine Translation in the Medical Domain
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.16440v2](http://arxiv.org/pdf/2408.16440v2)**

> **作者:** Miguel Rios
>
> **备注:** Citation: Miguel Rios. 2025. Instruction-tuned Large Language Models for Machine Translation in the Medical Domain. In Proceedings of Machine Translation Summit XX Volume 1, pages 162-172
>
> **摘要:** Large Language Models (LLMs) have shown promising results on machine translation for high resource language pairs and domains. However, in specialised domains (e.g. medical) LLMs have shown lower performance compared to standard neural machine translation models. The consistency in the machine translation of terminology is crucial for users, researchers, and translators in specialised domains. In this study, we compare the performance between baseline LLMs and instruction-tuned LLMs in the medical domain. In addition, we introduce terminology from specialised medical dictionaries into the instruction formatted datasets for fine-tuning LLMs. The instruction-tuned LLMs significantly outperform the baseline models with automatic metrics.
>
---
#### [replaced 011] FineMedLM-o1: Enhancing Medical Knowledge Reasoning Ability of LLM from Supervised Fine-Tuning to Test-Time Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.09213v3](http://arxiv.org/pdf/2501.09213v3)**

> **作者:** Hongzhou Yu; Tianhao Cheng; Yingwen Wang; Wen He; Qing Wang; Ying Cheng; Yuejie Zhang; Rui Feng; Xiaobo Zhang
>
> **摘要:** Recent advancements in large language models (LLMs) have shown promise in medical applications such as disease diagnosis and treatment planning. However, most existing medical LLMs struggle with the deep reasoning required for complex medical problems, such as differential diagnosis and medication recommendations. We propose FineMedLM-o1, which leverages high-quality medical synthetic data and long-form reasoning data for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), enabling advanced dialogue and deep reasoning capabilities. Additionally, we introduce Test-Time Training (TTT) in the medical domain for the first time, facilitating domain adaptation and ensuring reliable, accurate reasoning. Experimental results demonstrate that FineMedLM-o1 achieves a 23% average performance improvement over prior models on key medical benchmarks. Furthermore, the introduction of TTT provides an additional 14% performance boost, highlighting its effectiveness in enhancing medical reasoning capabilities. To support this process, we also propose a novel method for synthesizing medical dialogue. Compared to other open-source datasets, our dataset stands out as superior in both quality and complexity. The project and data will be released on GitHub.
>
---
#### [replaced 012] UI-E2I-Synth: Advancing GUI Grounding with Large-Scale Instruction Synthesis
- **分类: cs.HC; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.11257v4](http://arxiv.org/pdf/2504.11257v4)**

> **作者:** Xinyi Liu; Xiaoyi Zhang; Ziyun Zhang; Yan Lu
>
> **摘要:** Recent advancements in Large Vision-Language Models are accelerating the development of Graphical User Interface (GUI) agents that utilize human-like vision perception capabilities to enhance productivity on digital devices. Compared to approaches predicated on GUI metadata, which are platform-dependent and vulnerable to implementation variations, vision-based approaches offer broader applicability. In this vision-based paradigm, the GUI instruction grounding, which maps user instruction to the location of corresponding element on the given screenshot, remains a critical challenge, particularly due to limited public training dataset and resource-intensive manual instruction data annotation. In this paper, we delve into unexplored challenges in this task including element-to-screen ratio, unbalanced element type, and implicit instruction. To address these challenges, we introduce a large-scale data synthesis pipeline UI-E2I-Synth for generating varying complex instruction datasets using GPT-4o instead of human annotators. Furthermore, we propose a new GUI instruction grounding benchmark UI-I2E-Bench, which is designed to address the limitations of existing benchmarks by incorporating diverse annotation aspects. Our model, trained on the synthesized data, achieves superior performance in GUI instruction grounding, demonstrating the advancements of proposed data synthesis pipeline. The proposed benchmark, accompanied by extensive analyses, provides practical insights for future research in GUI grounding. We will release corresponding artifacts at https://microsoft.github.io/FIVE-UI-Evol/ .
>
---
#### [replaced 013] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.21391v2](http://arxiv.org/pdf/2507.21391v2)**

> **作者:** Shijie Zhou; Ruiyi Zhang; Huaisheng Zhu; Branislav Kveton; Yufan Zhou; Jiuxiang Gu; Jian Chen; Changyou Chen
>
> **备注:** Accepted at ICCV 2025. Code available at https://github.com/sjz5202/LLaVA-Reward
>
> **摘要:** We introduce LLaVA-Reward, an efficient reward model designed to automatically evaluate text-to-image (T2I) generations across multiple perspectives, leveraging pretrained multimodal large language models (MLLMs). Existing MLLM-based approaches require instruction-following data for supervised fine-tuning and evaluate generation quality on analyzing text response, which is time-consuming and difficult to train. To address this problem, we propose LLaVA-Reward, which directly utilizes the hidden states of MLLMs given text-image pairs. To enhance the bidirectional interaction between visual and textual representations in decoder-only MLLMs, we further propose adding a Skip-connection Cross Attention (SkipCA) module. This design enhances text-image correlation reasoning by connecting early-layer visual features with later-layer hidden representations. In addition, LLaVA-Reward supports different types of preference data for efficient fine-tuning, including paired preference data and unpaired data. We train LLaVA-Reward on four evaluation perspectives: text-image alignment, fidelity/artifact, safety, and overall ranking. Empirical results demonstrate that LLaVA-Reward outperforms conventional and MLLM-based methods in generating human-aligned scores for automatic evaluations and inference-time scaling in text-to-image generations.
>
---
#### [replaced 014] UI-AGILE: Advancing GUI Agents with Effective Reinforcement Learning and Precise Inference-Time Grounding
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.22025v2](http://arxiv.org/pdf/2507.22025v2)**

> **作者:** Shuquan Lian; Yuhang Wu; Jia Ma; Zihan Song; Bingqi Chen; Xiawu Zheng; Hui Li
>
> **摘要:** The emergence of Multimodal Large Language Models (MLLMs) has driven significant advances in Graphical User Interface (GUI) agent capabilities. Nevertheless, existing GUI agent training and inference techniques still suffer from a dilemma for reasoning designs, ineffective reward, and visual noise. To address these issues, we introduce UI-AGILE, a comprehensive framework enhancing GUI agents at both the training and inference stages. For training, we propose a suite of improvements to the Supervised Fine-Tuning (SFT) process: 1) a Continuous Reward function to incentivize high-precision grounding; 2) a "Simple Thinking" reward to balance planning with speed and grounding accuracy; and 3) a Cropping-based Resampling strategy to mitigate the sparse reward problem and improve learning on complex tasks. For inference, we present Decomposed Grounding with Selection, a novel method that dramatically improves grounding accuracy on high-resolution displays by breaking the image into smaller, manageable parts. Experiments show that UI-AGILE achieves the state-of-the-art performance on two benchmarks ScreenSpot-Pro and ScreenSpot-v2. For instance, using both our proposed training and inference enhancement methods brings 23% grounding accuracy improvement over the best baseline on ScreenSpot-Pro.
>
---
#### [replaced 015] Past Meets Present: Creating Historical Analogy with Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.14820v2](http://arxiv.org/pdf/2409.14820v2)**

> **作者:** Nianqi Li; Siyu Yuan; Jiangjie Chen; Jiaqing Liang; Feng Wei; Zujie Liang; Deqing Yang; Yanghua Xiao
>
> **备注:** Accepted to ACL 2025 (Outstanding Paper Award)
>
> **摘要:** Historical analogies, which compare known past events with contemporary but unfamiliar events, are important abilities that help people make decisions and understand the world. However, research in applied history suggests that people have difficulty finding appropriate analogies. And previous studies in the AI community have also overlooked historical analogies. To fill this gap, in this paper, we focus on the historical analogy acquisition task, which aims to acquire analogous historical events for a given event. We explore retrieval and generation methods for acquiring historical analogies based on different large language models (LLMs). Furthermore, we propose a self-reflection method to mitigate hallucinations and stereotypes when LLMs generate historical analogies. Through human evaluations and our specially designed automatic multi-dimensional assessment, we find that LLMs generally have a good potential for historical analogies. And the performance of the models can be further improved by using our self-reflection method.
>
---
#### [replaced 016] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15586v4](http://arxiv.org/pdf/2507.15586v4)**

> **作者:** Xinping Zhao; Shouzheng Huang; Yan Zhong; Xinshuo Hu; Meishan Zhang; Baotian Hu; Min Zhang
>
> **备注:** 16 pages, 7 Figures, 10 Tables
>
> **摘要:** Retrieval-Augmented Generation (RAG) effectively improves the accuracy of Large Language Models (LLMs). However, retrieval noises significantly impact the quality of LLMs' generation, necessitating the development of denoising mechanisms. Previous methods extract evidence straightforwardly without explicit thinking, which risks filtering out key clues and struggles with generalization. To this end, we propose EviOmni, which learns to extract rational evidence by (1) explicitly reasoning to identify potential cues within retrieval contents first, and then (2) consciously extracting to avoid omitting any key cues helpful for answering questions. Specifically, we frame evidence reasoning and evidence extraction into one unified response for end-to-end training; apply knowledge token masks for disentanglement to derive reasoning-based and extraction-based answers; and devise three types of verifiable reward functions, including answer, length, and format, to update the model via the policy optimization algorithm. Extensive experiments on three benchmark datasets show the effectiveness of EviOmni, providing compact and high-quality evidence, improving the accuracy of downstream tasks, and promoting effective application in online RAG systems.
>
---
#### [replaced 017] MuSciClaims: Multimodal Scientific Claim Verification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04585v2](http://arxiv.org/pdf/2506.04585v2)**

> **作者:** Yash Kumar Lal; Manikanta Bandham; Mohammad Saqib Hasan; Apoorva Kashi; Mahnaz Koupaee; Niranjan Balasubramanian
>
> **摘要:** Assessing scientific claims requires identifying, extracting, and reasoning with multimodal data expressed in information-rich figures in scientific literature. Despite the large body of work in scientific QA, figure captioning, and other multimodal reasoning tasks over chart-based data, there are no readily usable multimodal benchmarks that directly test claim verification abilities. To remedy this gap, we introduce a new benchmark MuSciClaims accompanied by diagnostics tasks. We automatically extract supported claims from scientific articles, which we manually perturb to produce contradicted claims. The perturbations are designed to test for a specific set of claim verification capabilities. We also introduce a suite of diagnostic tasks that help understand model failures. Our results show most vision-language models are poor (~0.3-0.5 F1), with even the best model only achieving 0.72 F1. They are also biased towards judging claims as supported, likely misunderstanding nuanced perturbations within the claims. Our diagnostics show models are bad at localizing correct evidence within figures, struggle with aggregating information across modalities, and often fail to understand basic components of the figure.
>
---
#### [replaced 018] BERSting at the Screams: A Benchmark for Distanced, Emotional and Shouted Speech Recognition
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.00059v2](http://arxiv.org/pdf/2505.00059v2)**

> **作者:** Paige Tuttösí; Mantaj Dhillon; Luna Sang; Shane Eastwood; Poorvi Bhatia; Quang Minh Dinh; Avni Kapoor; Yewon Jin; Angelica Lim
>
> **备注:** Accepted to Computer Speech and Language, Special issue: Multi-Speaker, Multi-Microphone, and Multi-Modal Distant Speech Recognition. Project Webpage and Data access : https://huggingface.co/datasets/Rosie-Lab/BERSt
>
> **摘要:** Some speech recognition tasks, such as automatic speech recognition (ASR), are approaching or have reached human performance in many reported metrics. Yet, they continue to struggle in complex, real-world, situations, such as with distanced speech. Previous challenges have released datasets to address the issue of distanced ASR, however, the focus remains primarily on distance, specifically relying on multi-microphone array systems. Here we present the B(asic) E(motion) R(andom phrase) S(hou)t(s) (BERSt) dataset. The dataset contains almost 4 hours of English speech from 98 actors with varying regional and non-native accents. The data was collected on smartphones in the actors homes and therefore includes at least 98 different acoustic environments. The data also includes 7 different emotion prompts and both shouted and spoken utterances. The smartphones were places in 19 different positions, including obstructions and being in a different room than the actor. This data is publicly available for use and can be used to evaluate a variety of speech recognition tasks, including: ASR, shout detection, and speech emotion recognition (SER). We provide initial benchmarks for ASR and SER tasks, and find that ASR degrades both with an increase in distance and shout level and shows varied performance depending on the intended emotion. Our results show that the BERSt dataset is challenging for both ASR and SER tasks and continued work is needed to improve the robustness of such systems for more accurate real-world use.
>
---
#### [replaced 019] Reservoir Computing as a Language Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.15779v2](http://arxiv.org/pdf/2507.15779v2)**

> **作者:** Felix Köster; Atsushi Uchida
>
> **备注:** 8 pages, 5 figures, 1 table Code available at: https://github.com/fekoester/Shakespeare_Res
>
> **摘要:** Large Language Models (LLM) have dominated the science and media landscape duo to their impressive performance on processing large chunks of data and produce human-like levels of text. Nevertheless, their huge energy demand and slow processing still a bottleneck for further increasing quality while also making the models accessible to everyone. To solve this bottleneck, we will investigate how reservoir computing performs on natural text processing, which could enable fast and energy efficient hardware implementations. Studies investigating the use of reservoir computing as a language model remain sparse. In this paper, we compare three distinct approaches for character-level language modeling, two different reservoir computing approaches, where only an output layer is trainable, and the well-known transformer-based architectures, which fully learn an attention-based sequence representation. We explore the performance, computational cost and prediction accuracy for both paradigms by equally varying the number of trainable parameters for all models. Using a consistent pipeline for all three approaches, we demonstrate that transformers excel in prediction quality, whereas reservoir computers remain highly efficient reducing the training and inference speed. Furthermore, we investigate two types of reservoir computing: a traditional reservoir with a static linear readout, and an attention-enhanced reservoir that dynamically adapts its output weights via an attention mechanism. Our findings underline how these paradigms scale and offer guidelines to balance resource constraints with performance.
>
---
#### [replaced 020] MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral Reasoning of LLMs through Hate Speech Multi-hop Explanations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.19073v2](http://arxiv.org/pdf/2506.19073v2)**

> **作者:** Jackson Trager; Diego Alves; Matteo Guida; Mikel K. Ngueajio; Ameeta Agrawal; Flor Plaza-del-Arco; Yalda Daryanai; Farzan Karimi-Malekabadi; Francielle Vargas
>
> **备注:** Under Review
>
> **摘要:** Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning.
>
---
#### [replaced 021] Training language models to be warm and empathetic makes them less reliable and more sycophantic
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2507.21919v2](http://arxiv.org/pdf/2507.21919v2)**

> **作者:** Lujain Ibrahim; Franziska Sofia Hafner; Luc Rocher
>
> **摘要:** Artificial intelligence (AI) developers are increasingly building language models with warm and empathetic personas that millions of people now use for advice, therapy, and companionship. Here, we show how this creates a significant trade-off: optimizing language models for warmth undermines their reliability, especially when users express vulnerability. We conducted controlled experiments on five language models of varying sizes and architectures, training them to produce warmer, more empathetic responses, then evaluating them on safety-critical tasks. Warm models showed substantially higher error rates (+10 to +30 percentage points) than their original counterparts, promoting conspiracy theories, providing incorrect factual information, and offering problematic medical advice. They were also significantly more likely to validate incorrect user beliefs, particularly when user messages expressed sadness. Importantly, these effects were consistent across different model architectures, and occurred despite preserved performance on standard benchmarks, revealing systematic risks that current evaluation practices may fail to detect. As human-like AI systems are deployed at an unprecedented scale, our findings indicate a need to rethink how we develop and oversee these systems that are reshaping human relationships and social interaction.
>
---
#### [replaced 022] Enhancing Ultra-Low-Bit Quantization of Large Language Models Through Saliency-Aware Partial Retraining
- **分类: cs.LG; cs.CL; 68T50, 68T07, 68T09, 68U15; I.2.7; I.2.6; I.2.4**

- **链接: [http://arxiv.org/pdf/2504.13932v3](http://arxiv.org/pdf/2504.13932v3)**

> **作者:** Deyu Cao; Samin Aref
>
> **备注:** This is a post-peer-review accepted manuscript from the proceedings of the 22nd International Conference on Modeling Decisions for Artificial Intelligence (MDAI'25). The publisher authenticated version and full citation details are available on Springer's website (LNAI 15957). https://doi.org/10.1007/978-3-032-00891-6_28
>
> **摘要:** The growing use of large language models has raised environmental and economic concerns about their intensity of resource usage during inference. Serving these models to each user requires substantial energy and water for cooling. Model compression techniques like quantization can shrink large language models and make them more resource efficient at the cost of potential performance degradation. Quantization methods compress model size through replacing their high-precision parameters by quantized values of lower precision. Among existing methods, the ApiQ method achieves superior accuracy preservation at minimal memory and time overhead. We investigate two ideas to extend performance in ultra-low-bit quantization beyond ApiQ's level. First, we look into combining existing quantization-aware training techniques with ApiQ's partial training. We show that this does not outperform the baseline ApiQ method with limited training data and frozen weights. This leads to two key insights: (1) The substantial representational capacity that is gained through full retraining is unlikely to be feasible through partial training. (2) This gain may depend on using a large and diverse dataset in quantization-aware training. Second, through a novel approach informed by the two insights, we propose an ultra-low-bit quantization method that builds upon ApiQ and extends its performance without the need for full retraining. This publicly available method relies on a saliency-aware regularization term that prioritizes preserving the most impactful parameters during quantization. Our experiments on LLaMA 7B and 13B benchmarks demonstrate that our method reduces the ApiQ's accuracy degradation by 10.85% and 7.54% respectively. A Python implementation of the proposed quantization method is publicly available on GitHub https://github.com/TokuyuSou/ULB-SAPR.
>
---
#### [replaced 023] Scaling RL to Long Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.07966v3](http://arxiv.org/pdf/2507.07966v3)**

> **作者:** Yukang Chen; Wei Huang; Baifeng Shi; Qinghao Hu; Hanrong Ye; Ligeng Zhu; Zhijian Liu; Pavlo Molchanov; Jan Kautz; Xiaojuan Qi; Sifei Liu; Hongxu Yin; Yao Lu; Song Han
>
> **备注:** Code at https://github.com/NVlabs/Long-RL and model at https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B
>
> **摘要:** We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 104K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In our experiments, LongVILA-R1-7B achieves strong performance on video benchmarks, reaching 65.1% and 71.1% accuracy on VideoMME without and with subtitles, respectively, and consistently outperforming LongVILA-7B across multiple benchmarks. Moreover, LongVILA-R1-7B supports processing up to 8,192 video frames per video, and configurable FPS settings. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames).
>
---
#### [replaced 024] QE4PE: Word-level Quality Estimation for Human Post-Editing
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2503.03044v2](http://arxiv.org/pdf/2503.03044v2)**

> **作者:** Gabriele Sarti; Vilém Zouhar; Grzegorz Chrupała; Ana Guerberof-Arenas; Malvina Nissim; Arianna Bisazza
>
> **备注:** Accepted by TACL (pre-MIT Press publication version); Code: https://github.com/gsarti/qe4pe. Dataset: https://huggingface.co/datasets/gsarti/qe4pe
>
> **摘要:** Word-level quality estimation (QE) methods aim to detect erroneous spans in machine translations, which can direct and facilitate human post-editing. While the accuracy of word-level QE systems has been assessed extensively, their usability and downstream influence on the speed, quality and editing choices of human post-editing remain understudied. In this study, we investigate the impact of word-level QE on machine translation (MT) post-editing in a realistic setting involving 42 professional post-editors across two translation directions. We compare four error-span highlight modalities, including supervised and uncertainty-based word-level QE methods, for identifying potential errors in the outputs of a state-of-the-art neural MT model. Post-editing effort and productivity are estimated from behavioral logs, while quality improvements are assessed by word- and segment-level human annotation. We find that domain, language and editors' speed are critical factors in determining highlights' effectiveness, with modest differences between human-made and automated QE highlights underlining a gap between accuracy and usability in professional workflows.
>
---
#### [replaced 025] MiniLongBench: The Low-cost Long Context Understanding Benchmark for Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19959v2](http://arxiv.org/pdf/2505.19959v2)**

> **作者:** Zhongzhan Huang; Guoming Ling; Shanshan Zhong; Hefeng Wu; Liang Lin
>
> **备注:** Accepted by ACL'25 main track
>
> **摘要:** Long Context Understanding (LCU) is a critical area for exploration in current large language models (LLMs). However, due to the inherently lengthy nature of long-text data, existing LCU benchmarks for LLMs often result in prohibitively high evaluation costs, like testing time and inference expenses. Through extensive experimentation, we discover that existing LCU benchmarks exhibit significant redundancy, which means the inefficiency in evaluation. In this paper, we propose a concise data compression method tailored for long-text data with sparse information characteristics. By pruning the well-known LCU benchmark LongBench, we create MiniLongBench. This benchmark includes only 237 test samples across six major task categories and 21 distinct tasks. Through empirical analysis of over 60 LLMs, MiniLongBench achieves an average evaluation cost reduced to only 4.5% of the original while maintaining an average rank correlation coefficient of 0.97 with LongBench results. Therefore, our MiniLongBench, as a low-cost benchmark, holds great potential to substantially drive future research into the LCU capabilities of LLMs. See https://github.com/MilkThink-Lab/MiniLongBench for our code, data and tutorial.
>
---
#### [replaced 026] Towards the Law of Capacity Gap in Distilling Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2311.07052v4](http://arxiv.org/pdf/2311.07052v4)**

> **作者:** Chen Zhang; Qiuchi Li; Dawei Song; Zheyu Ye; Yan Gao; Yan Hu
>
> **备注:** 32 pages, 10 figures, 15 tables, accepted to ACL 2025. Code and checkpoints are available at https://github.com/GeneZC/MiniMA
>
> **摘要:** Language model (LM) distillation aims at distilling the knowledge in a large teacher LM to a small student one. As a critical issue facing LM distillation, a superior student often arises from a teacher of a relatively small scale instead of a larger one, especially in the presence of substantial capacity gap between the teacher and student. This issue, often referred to as the \textit{curse of capacity gap}, suggests that there is likely an optimal teacher yielding the best-performing student along the scaling course of the teacher. Consequently, distillation trials on teachers of a wide range of scales are called for to determine the optimal teacher, which becomes computationally intensive in the context of large LMs (LLMs). This paper addresses this critical bottleneck by providing the \textit{law of capacity gap} inducted from a preliminary study on distilling a broad range of small-scale (<3B) LMs, where the optimal teacher consistently scales linearly with the student scale across different model and data scales. By extending the law to LLM distillation on a larger scale (7B), we succeed in obtaining versatile LLMs that outperform a wide array of competitors.
>
---
#### [replaced 027] Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.15038v2](http://arxiv.org/pdf/2505.15038v2)**

> **作者:** Haiyan Zhao; Xuansheng Wu; Fan Yang; Bo Shen; Ninghao Liu; Mengnan Du
>
> **备注:** 12 pages, 4 figures, 4 tables
>
> **摘要:** Linear concept vectors effectively steer LLMs, but existing methods suffer from noisy features in diverse datasets that undermine steering robustness. We propose Sparse Autoencoder-Denoised Concept Vectors (SDCV), which selectively keep the most discriminative SAE latents while reconstructing hidden representations. Our key insight is that concept-relevant signals can be explicitly separated from dataset noise by scaling up activations of top-k latents that best differentiate positive and negative samples. Applied to linear probing and difference-in-mean, SDCV consistently improves steering success rates by 4-16\% across six challenging concepts, while maintaining topic relevance.
>
---
#### [replaced 028] Voices of Freelance Professional Writers on AI: Limitations, Expectations, and Fears
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.05008v2](http://arxiv.org/pdf/2504.05008v2)**

> **作者:** Anastasiia Ivanova; Natalia Fedorova; Sergei Tilga; Ekaterina Artemova
>
> **摘要:** The rapid development of AI-driven tools, particularly large language models (LLMs), is reshaping professional writing. Still, key aspects of their adoption such as languages support, ethics, and long-term impact on writers voice and creativity remain underexplored. In this work, we conducted a questionnaire (N = 301) and an interactive survey (N = 36) targeting professional writers regularly using AI. We examined LLM-assisted writing practices across 25+ languages, ethical concerns, and user expectations. The findings of the survey demonstrate important insights, reflecting upon the importance of: LLMs adoption for non-English speakers; the degree of misinformation, domain and style adaptation; usability and key features of LLMs. These insights can guide further development, benefiting both writers and a broader user base.
>
---
#### [replaced 029] Prompt-Reverse Inconsistency: LLM Self-Inconsistency Beyond Generative Randomness and Prompt Paraphrasing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.01282v2](http://arxiv.org/pdf/2504.01282v2)**

> **作者:** Jihyun Janice Ahn; Wenpeng Yin
>
> **备注:** accepted in COLM2025, 9 pages
>
> **摘要:** While the inconsistency of LLMs is not a novel topic, prior research has predominantly addressed two types of generative inconsistencies: i) Randomness Inconsistency: running the same LLM multiple trials, yielding varying responses; ii) Paraphrase Inconsistency: paraphrased prompts result in different responses from the same LLM. Randomness Inconsistency arises from the inherent randomness due to stochastic sampling in generative models, while Paraphrase Inconsistency is a consequence of the language modeling objectives, where paraphrased prompts alter the distribution of vocabulary logits. This research discovers Prompt-Reverse Inconsistency (PRIN), a new form of LLM self-inconsistency: given a question and a couple of LLM-generated answer candidates, the LLM often has conflicting responses when prompted "Which are correct answers?" and "Which are incorrect answers?". PRIN poses a big concern as it undermines the credibility of LLM-as-a-judge, and suggests a challenge for LLMs to adhere to basic logical rules. We conduct a series of experiments to investigate PRIN, examining the extent of PRIN across different LLMs, methods to mitigate it, potential applications, and its relationship with Randomness Inconsistency and Paraphrase Inconsistency. As the first study to explore PRIN, our findings offer valuable insights into the inner workings of LLMs and contribute to advancing trustworthy AI.
>
---
#### [replaced 030] Cross-Modal State-Space Graph Reasoning for Structured Summarization
- **分类: cs.CL; cs.GR**

- **链接: [http://arxiv.org/pdf/2503.20988v2](http://arxiv.org/pdf/2503.20988v2)**

> **作者:** Hannah Kim; Sofia Martinez; Jason Lee
>
> **备注:** arXiv admin note: This paper has been withdrawn by arXiv due to disputed and unverifiable authorship and affiliation
>
> **摘要:** The ability to extract compact, meaningful summaries from large-scale and multimodal data is critical for numerous applications, ranging from video analytics to medical reports. Prior methods in cross-modal summarization have often suffered from high computational overheads and limited interpretability. In this paper, we propose a \textit{Cross-Modal State-Space Graph Reasoning} (\textbf{CSS-GR}) framework that incorporates a state-space model with graph-based message passing, inspired by prior work on efficient state-space models. Unlike existing approaches relying on purely sequential models, our method constructs a graph that captures inter- and intra-modal relationships, allowing more holistic reasoning over both textual and visual streams. We demonstrate that our approach significantly improves summarization quality and interpretability while maintaining computational efficiency, as validated on standard multimodal summarization benchmarks. We also provide a thorough ablation study to highlight the contributions of each component.
>
---
#### [replaced 031] Co-AttenDWG: Co-Attentive Dimension-Wise Gating and Expert Fusion for Multi-Modal Offensive Content Detection
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19010v2](http://arxiv.org/pdf/2505.19010v2)**

> **作者:** Md. Mithun Hossain; Md. Shakil Hossain; Sudipto Chaki; M. F. Mridha
>
> **摘要:** Multi-modal learning has emerged as a crucial research direction, as integrating textual and visual information can substantially enhance performance in tasks such as classification, retrieval, and scene understanding. Despite advances with large pre-trained models, existing approaches often suffer from insufficient cross-modal interactions and rigid fusion strategies, failing to fully harness the complementary strengths of different modalities. To address these limitations, we propose Co-AttenDWG, co-attention with dimension-wise gating, and expert fusion. Our approach first projects textual and visual features into a shared embedding space, where a dedicated co-attention mechanism enables simultaneous, fine-grained interactions between modalities. This is further strengthened by a dimension-wise gating network, which adaptively modulates feature contributions at the channel level to emphasize salient information. In parallel, dual-path encoders independently refine modality-specific representations, while an additional cross-attention layer aligns the modalities further. The resulting features are aggregated via an expert fusion module that integrates learned gating and self-attention, yielding a robust unified representation. Experimental results on the MIMIC and SemEval Memotion 1.0 datasets show that Co-AttenDWG achieves state-of-the-art performance and superior cross-modal alignment, highlighting its effectiveness for diverse multi-modal applications.
>
---
#### [replaced 032] Neutral Residues: Revisiting Adapters for Model Extension
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.02744v2](http://arxiv.org/pdf/2410.02744v2)**

> **作者:** Franck Signe Talla; Edouard Grave; Hervé Jégou
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** We address the problem of extending a pretrained large language model to a new domain that was not seen during training. Standard techniques, such as finetuning or low-rank adaptation (LoRA) are successful at domain adaptation, but do not formally add capacity to the model. This often leads to a trade-off, between performing well on the new domain vs. degrading performance on the original domain. Here, we revisit and improve adapters to extend LLMs from three angles: data, architecture and training procedure, which are advantageously considered jointly. The resulting method, called neutral residues, modifies adapters in a way that leads each new residual block to output near-zeros on the original domain. This solution leads to strong results when adapting a state-of-the-art model originally trained on English to a new language. Neutral residues significantly outperform competing approaches such as finetuning, LoRA or vanilla adapters in terms of the trade-off between learning the new language and not forgetting English.
>
---
#### [replaced 033] Leveraging Large Language Models for Bengali Math Word Problem Solving with Chain of Thought Reasoning
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21354v2](http://arxiv.org/pdf/2505.21354v2)**

> **作者:** Bidyarthi Paul; Jalisha Jashim Era; Mirazur Rahman Zim; Tahmid Sattar Aothoi; Faisal Muhammad Shah
>
> **摘要:** Solving Bengali Math Word Problems (MWPs) remains a major challenge in natural language processing (NLP) due to the language's low-resource status and the multi-step reasoning required. Existing models struggle with complex Bengali MWPs, largely because no human-annotated Bengali dataset has previously addressed this task. This gap has limited progress in Bengali mathematical reasoning. To address this, we created SOMADHAN, a dataset of 8792 complex Bengali MWPs with manually written, step-by-step solutions. We designed this dataset to support reasoning-focused evaluation and model development in a linguistically underrepresented context. Using SOMADHAN, we evaluated a range of large language models (LLMs) - including GPT-4o, GPT-3.5 Turbo, LLaMA series models, Deepseek, and Qwen - through both zero-shot and few-shot prompting with and without Chain of Thought (CoT) reasoning. CoT prompting consistently improved performance over standard prompting, especially in tasks requiring multi-step logic. LLaMA-3.3 70B achieved the highest accuracy of 88% with few-shot CoT prompting. We also applied Low-Rank Adaptation (LoRA) to fine-tune models efficiently, enabling them to adapt to Bengali MWPs with minimal computational cost. Our work fills a critical gap in Bengali NLP by providing a high-quality reasoning dataset and a scalable framework for solving complex MWPs. We aim to advance equitable research in low-resource languages and enhance reasoning capabilities in educational and language technologies.
>
---
#### [replaced 034] Scoring Verifiers: Evaluating Synthetic Verification for Code and Reasoning
- **分类: cs.AI; cs.CL; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2502.13820v3](http://arxiv.org/pdf/2502.13820v3)**

> **作者:** Aleksander Ficek; Somshubra Majumdar; Vahid Noroozi; Boris Ginsburg
>
> **备注:** COLM 2025
>
> **摘要:** Synthetic verification techniques such as generating test cases and reward modelling are common ways to enhance the coding capabilities of large language models (LLM) beyond predefined tests. Additionally, code verification has recently found great success as a critical component in improving reasoning capability of LLMs via reinforcement learning. In this paper, we propose an approach which can transform existing coding benchmarks into scoring and ranking datasets to evaluate the effectiveness of synthetic verifiers. We also propose multiple metrics to measure different aspects of the synthetic verifiers with the proposed benchmarks. By employing the proposed approach, we release four new benchmarks (HE-R, HE-R+, MBPP-R, and MBPP-R+), and analyzed synthetic verification methods with standard, reasoning-based, and reward-based LLMs. Our experiments show that reasoning can significantly improve test case generation and that scaling the number of test cases enhances the verification accuracy.
>
---
#### [replaced 035] Rationale-guided Prompting for Knowledge-based Visual Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.16936v2](http://arxiv.org/pdf/2412.16936v2)**

> **作者:** Zhongjian Hu; Peng Yang; Bing Li; Fengyuan Liu
>
> **摘要:** Recently, Large Language Models (LLMs) have been used for knowledge-based Visual Question Answering (VQA). Despite the encouraging results of previous studies, prior methods prompt LLMs to predict answers directly, neglecting intermediate thought processes. We argue that prior methods do not sufficiently activate the capacities of LLMs. We propose a framework called PLRH that Prompts LLMs with Rationale Heuristics for knowledge-based VQA. The PLRH prompts LLMs with Chain of Thought (CoT) to generate rationale heuristics, i.e., intermediate thought processes, and then leverages the rationale heuristics to inspire LLMs to predict answers. Experiments show that our approach outperforms the existing baselines by more than 2.2 and 2.1 on OK-VQA and A-OKVQA, respectively.
>
---
#### [replaced 036] Basic Reading Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.19741v2](http://arxiv.org/pdf/2507.19741v2)**

> **作者:** Zhi Zhou; Sirui Miao; Xiangyu Duan; Hao Yang; Min Zhang
>
> **备注:** Accepted by ACL2025
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable abilities in various natural language processing areas, but they demand high computation resources which limits their deployment in real-world. Distillation is one technique to solve this problem through either knowledge distillation or task distillation. Both distillation approaches train small models to imitate specific features of LLMs, but they all neglect basic reading education for small models on generic texts that are \emph{unrelated} to downstream tasks. In this paper, we propose basic reading distillation (BRD) which educates a small model to imitate LLMs basic reading behaviors, such as named entity recognition, question raising and answering, on each sentence. After such basic education, we apply the small model on various tasks including language inference benchmarks and BIG-bench tasks. It shows that the small model can outperform or perform comparable to over 20x bigger LLMs. Analysis reveals that BRD effectively influences the probability distribution of the small model, and has orthogonality to either knowledge distillation or task distillation.
>
---
#### [replaced 037] Efficient Continual Learning for Small Language Models with a Discrete Key-Value Bottleneck
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.08528v2](http://arxiv.org/pdf/2412.08528v2)**

> **作者:** Andor Diera; Lukas Galke; Fabian Karl; Ansgar Scherp
>
> **摘要:** Continual learning remains a challenge across various natural language processing (NLP) tasks, as models updated with new training data often risk catastrophic forgetting of previously acquired knowledge. We introduce a discrete key-value bottleneck (DKVB) for encoder-only language models, enabling efficient continual learning through localized updates. Inspired by a discrete key-value bottleneck in vision, we consider new and NLP-specific challenges. We compare different bottleneck architectures for NLP and introduce a new, task-independent initialization technique for the discrete keys. We evaluate our DKVB for NLP in four continual learning scenarios and show that it alleviates catastrophic forgetting. Our experiments demonstrate that the proposed approach achieves competitive performance compared to popular continual learning methods while incurring lower computational costs. Furthermore, we show that DKVB remains effective even in challenging single-head continual learning scenarios where no task ID is provided.
>
---
#### [replaced 038] IterKey: Iterative Keyword Generation with LLMs for Enhanced Retrieval Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.08450v2](http://arxiv.org/pdf/2505.08450v2)**

> **作者:** Kazuki Hayashi; Hidetaka Kamigaito; Shinya Kouda; Taro Watanabe
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a way to complement the in-context knowledge of Large Language Models (LLMs) by integrating external documents. However, real-world applications demand not only accuracy but also interpretability. While dense retrieval methods provide high accuracy, they lack interpretability; conversely, sparse retrieval methods offer transparency but often fail to capture the full intent of queries due to their reliance on keyword matching. To address these issues, we introduce IterKey, an LLM-driven iterative keyword generation framework that enhances RAG via sparse retrieval. IterKey consists of three LLM-driven stages: generating keywords for retrieval, generating answers based on retrieved documents, and validating the answers. If validation fails, the process iteratively repeats with refined keywords. Across four QA tasks, experimental results show that IterKey achieves 5% to 20% accuracy improvements over BM25-based RAG and simple baselines. Its performance is comparable to dense retrieval-based RAG and prior iterative query refinement methods using dense models. In summary, IterKey is a novel BM25-based approach leveraging LLMs to iteratively refine RAG, effectively balancing accuracy with interpretability.
>
---
#### [replaced 039] Yankari: A Monolingual Yoruba Dataset
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.03334v2](http://arxiv.org/pdf/2412.03334v2)**

> **作者:** Maro Akpobi
>
> **备注:** 6 pages
>
> **摘要:** This paper presents Yankari, a large-scale monolingual dataset for the Yoruba language, aimed at addressing the critical gap in Natural Language Processing (NLP) resources for this important West African language. Despite being spoken by over 30 million people, Yoruba has been severely underrepresented in NLP research and applications. We detail our methodology for creating this dataset, which includes careful source selection, automated quality control, and rigorous data cleaning processes. The Yankari dataset comprises 51,407 documents from 13 diverse sources, totaling over 30 million tokens. Our approach focuses on ethical data collection practices, avoiding problematic sources and addressing issues prevalent in existing datasets. We provide thorough automated evaluations of the dataset, demonstrating its quality compared to existing resources. The Yankari dataset represents a significant advancement in Yoruba language resources, providing a foundation for developing more accurate NLP models, supporting comparative linguistic studies, and contributing to the digital accessibility of the Yoruba language.
>
---
#### [replaced 040] Can adversarial attacks by large language models be attributed?
- **分类: cs.AI; cs.CL; cs.CY; cs.FL**

- **链接: [http://arxiv.org/pdf/2411.08003v3](http://arxiv.org/pdf/2411.08003v3)**

> **作者:** Manuel Cebrian; Andres Abeliuk; Jan Arne Telle
>
> **备注:** 22 pages, 5 figures, 2 tables
>
> **摘要:** Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.
>
---
#### [replaced 041] Mmm whatcha say? Uncovering distal and proximal context effects in first and second-language word perception using psychophysical reverse correlation
- **分类: cs.SD; cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2406.05515v2](http://arxiv.org/pdf/2406.05515v2)**

> **作者:** Paige Tuttösí; H. Henny Yeung; Yue Wang; Fenqi Wang; Guillaume Denis; Jean-Julien Aucouturier; Angelica Lim
>
> **备注:** Accepted to INTERSPEECH 2024 Project Webpage : https://rosielab.github.io/vocal_ambiguity/ Code: https://github.com/neuro-team-femto/vocal_ambiguity Data : https://zenodo.org/records/12761242
>
> **摘要:** Acoustic context effects, where surrounding changes in pitch, rate or timbre influence the perception of a sound, are well documented in speech perception, but how they interact with language background remains unclear. Using a reverse-correlation approach, we systematically varied the pitch and speech rate in phrases around different pairs of vowels for second language (L2) speakers of English (/i/-/I/) and French (/u/-/y/), thus reconstructing, in a data-driven manner, the prosodic profiles that bias their perception. Testing English and French speakers (n=25), we showed that vowel perception is in fact influenced by conflicting effects from the surrounding pitch and speech rate: a congruent proximal effect 0.2s pre-target and a distal contrastive effect up to 1s before; and found that L1 and L2 speakers exhibited strikingly similar prosodic profiles in perception. We provide a novel method to investigate acoustic context effects across stimuli, timescales, and acoustic domain.
>
---
#### [replaced 042] OWLViz: An Open-World Benchmark for Visual Question Answering
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.07631v3](http://arxiv.org/pdf/2503.07631v3)**

> **作者:** Thuy Nguyen; Dang Nguyen; Hoang Nguyen; Thuan Luong; Long Hoang Dang; Viet Dac Lai
>
> **备注:** 8 pages + appendix
>
> **摘要:** We present a challenging benchmark for the Open WorLd VISual question answering (OWLViz) task. OWLViz presents concise, unambiguous queries that require integrating multiple capabilities, including visual understanding, web exploration, and specialized tool usage. While humans achieve 69.2% accuracy on these intuitive tasks, even state-of-the-art VLMs struggle, with the best model, Gemini 2.0, achieving only 26.6% accuracy. Current agentic VLMs, which rely on limited vision and vision-language models as tools, perform even worse. This performance gap reveals significant limitations in multimodal systems' ability to select appropriate tools and execute complex reasoning sequences, establishing new directions for advancing practical AI research.
>
---
#### [replaced 043] The Importance of Facial Features in Vision-based Sign Language Recognition: Eyes, Mouth or Full Face?
- **分类: cs.CV; cs.CL; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.20884v2](http://arxiv.org/pdf/2507.20884v2)**

> **作者:** Dinh Nam Pham; Eleftherios Avramidis
>
> **备注:** Accepted at 9th International Workshop on Sign Language Translation and Avatar Technologies @ ACM IVA'25
>
> **摘要:** Non-manual facial features play a crucial role in sign language communication, yet their importance in automatic sign language recognition (ASLR) remains underexplored. While prior studies have shown that incorporating facial features can improve recognition, related work often relies on hand-crafted feature extraction and fails to go beyond the comparison of manual features versus the combination of manual and facial features. In this work, we systematically investigate the contribution of distinct facial regionseyes, mouth, and full faceusing two different deep learning models (a CNN-based model and a transformer-based model) trained on an SLR dataset of isolated signs with randomly selected classes. Through quantitative performance and qualitative saliency map evaluation, we reveal that the mouth is the most important non-manual facial feature, significantly improving accuracy. Our findings highlight the necessity of incorporating facial features in ASLR.
>
---
#### [replaced 044] DeepSieve: Information Sieving via LLM-as-a-Knowledge-Router
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.22050v2](http://arxiv.org/pdf/2507.22050v2)**

> **作者:** Minghao Guo; Qingcheng Zeng; Xujiang Zhao; Yanchi Liu; Wenchao Yu; Mengnan Du; Haifeng Chen; Wei Cheng
>
> **备注:** 22 pages, work in progress
>
> **摘要:** Large Language Models (LLMs) excel at many reasoning tasks but struggle with knowledge-intensive queries due to their inability to dynamically access up-to-date or domain-specific information. Retrieval-Augmented Generation (RAG) has emerged as a promising solution, enabling LLMs to ground their responses in external sources. However, existing RAG methods lack fine-grained control over both the query and source sides, often resulting in noisy retrieval and shallow reasoning. In this work, we introduce DeepSieve, an agentic RAG framework that incorporates information sieving via LLM-as-a-knowledge-router. DeepSieve decomposes complex queries into structured sub-questions and recursively routes each to the most suitable knowledge source, filtering irrelevant information through a multi-stage distillation process. Our design emphasizes modularity, transparency, and adaptability, leveraging recent advances in agentic system design. Experiments on multi-hop QA tasks across heterogeneous sources demonstrate improved reasoning depth, retrieval precision, and interpretability over conventional RAG approaches. Our codes are available at https://github.com/MinghoKwok/DeepSieve.
>
---
#### [replaced 045] BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level Activation Sparsity
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2507.08771v2](http://arxiv.org/pdf/2507.08771v2)**

> **作者:** Chenyang Song; Weilin Zhao; Xu Han; Chaojun Xiao; Yingfa Chen; Yuxuan Li; Zhiyuan Liu; Maosong Sun
>
> **备注:** 21 pages, 7 figures, 15 tables
>
> **摘要:** To alleviate the computational burden of large language models (LLMs), architectures with activation sparsity, represented by mixture-of-experts (MoE), have attracted increasing attention. However, the non-differentiable and inflexible routing of vanilla MoE hurts model performance. Moreover, while each token activates only a few parameters, these sparsely-activated architectures exhibit low chunk-level sparsity, indicating that the union of multiple consecutive tokens activates a large ratio of parameters. Such a sparsity pattern is unfriendly for acceleration under low-resource conditions (e.g., end-side devices) and incompatible with mainstream acceleration techniques (e.g., speculative decoding). To address these challenges, we introduce a novel MoE architecture, BlockFFN, as well as its efficient training and deployment techniques. Specifically, we use a router integrating ReLU activation and RMSNorm for differentiable and flexible routing. Next, to promote both token-level sparsity (TLS) and chunk-level sparsity (CLS), CLS-aware training objectives are designed, making BlockFFN more acceleration-friendly. Finally, we implement efficient acceleration kernels, combining activation sparsity and speculative decoding for the first time. The experimental results demonstrate the superior performance of BlockFFN over other MoE baselines, achieving over 80% TLS and 70% 8-token CLS. Our kernels achieve up to 3.67$\times$ speedup on real end-side devices than dense models. All codes and checkpoints are available publicly (https://github.com/thunlp/BlockFFN).
>
---
#### [replaced 046] LLM-as-a-qualitative-judge: automating error analysis in natural language generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.09147v2](http://arxiv.org/pdf/2506.09147v2)**

> **作者:** Nadezhda Chirkova; Tunde Oluwaseyi Ajayi; Seth Aycock; Zain Muhammad Mujahid; Vladana Perlić; Ekaterina Borisova; Markarit Vartampetian
>
> **摘要:** Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that LLM-as-a-qualitative-judge correctly recognizes instance-specific issues in 2/3 cases and is capable of producing error type reports resembling the reports composed by human annotators. Our code and data are publicly available at https://github.com/tunde-ajayi/llm-as-a-qualitative-judge.
>
---
#### [replaced 047] Masked Language Models are Good Heterogeneous Graph Generalizers
- **分类: cs.SI; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.06157v2](http://arxiv.org/pdf/2506.06157v2)**

> **作者:** Jinyu Yang; Cheng Yang; Shanyuan Cui; Zeyuan Guo; Liangwei Yang; Muhan Zhang; Zhiqiang Zhang; Chuan Shi
>
> **摘要:** Heterogeneous graph neural networks (HGNNs) excel at capturing structural and semantic information in heterogeneous graphs (HGs), while struggling to generalize across domains and tasks. With the rapid advancement of large language models (LLMs), a recent study explored the integration of HGNNs with LLMs for generalizable heterogeneous graph learning. However, this approach typically encodes structural information as HG tokens using HGNNs, and disparities in embedding spaces between HGNNs and LLMs have been shown to bias the LLM's comprehension of HGs. Moreover, since these HG tokens are often derived from node-level tasks, the model's ability to generalize across tasks remains limited. To this end, we propose a simple yet effective Masked Language Modeling-based method, called MLM4HG. MLM4HG introduces metapath-based textual sequences instead of HG tokens to extract structural and semantic information inherent in HGs, and designs customized textual templates to unify different graph tasks into a coherent cloze-style 'mask' token prediction paradigm. Specifically,MLM4HG first converts HGs from various domains to texts based on metapaths, and subsequently combines them with the unified task texts to form a HG-based corpus. Moreover, the corpus is fed into a pretrained LM for fine-tuning with a constrained target vocabulary, enabling the fine-tuned LM to generalize to unseen target HGs. Extensive cross-domain and multi-task experiments on four real-world datasets demonstrate the superior generalization performance of MLM4HG over state-of-the-art methods in both few-shot and zero-shot scenarios. Our code is available at https://github.com/BUPT-GAMMA/MLM4HG.
>
---
#### [replaced 048] Positive-Augmented Contrastive Learning for Vision-and-Language Evaluation and Training
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2410.07336v2](http://arxiv.org/pdf/2410.07336v2)**

> **作者:** Sara Sarto; Nicholas Moratelli; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara
>
> **备注:** International Journal of Computer Vision (2025)
>
> **摘要:** Despite significant advancements in caption generation, existing evaluation metrics often fail to capture the full quality or fine-grained details of captions. This is mainly due to their reliance on non-specific human-written references or noisy pre-training data. Still, finding an effective metric is crucial not only for captions evaluation but also for the generation phase. Metrics can indeed play a key role in the fine-tuning stage of captioning models, ultimately enhancing the quality of the generated captions. In this paper, we propose PAC-S++, a learnable metric that leverages the CLIP model, pre-trained on both web-collected and cleaned data and regularized through additional pairs of generated visual and textual positive samples. Exploiting this stronger and curated pre-training, we also apply PAC-S++ as a reward in the Self-Critical Sequence Training (SCST) stage typically employed to fine-tune captioning models. Extensive experiments on different image and video datasets highlight the effectiveness of PAC-S++ compared to popular metrics for the task, including its sensitivity to object hallucinations. Furthermore, we show that integrating PAC-S++ into the fine-tuning stage of a captioning model results in semantically richer captions with fewer repetitions and grammatical errors. Evaluations on out-of-domain benchmarks further demonstrate the efficacy of our fine-tuning approach in enhancing model capabilities. Source code and trained models are publicly available at: https://github.com/aimagelab/pacscore.
>
---
#### [replaced 049] Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.15677v3](http://arxiv.org/pdf/2506.15677v3)**

> **作者:** Yining Hong; Rui Sun; Bingxuan Li; Xingcheng Yao; Maxine Wu; Alexander Chien; Da Yin; Ying Nian Wu; Zhecan James Wang; Kai-Wei Chang
>
> **摘要:** AI agents today are mostly siloed - they either retrieve and reason over vast amount of digital information and knowledge obtained online; or interact with the physical world through embodied perception, planning and action - but rarely both. This separation limits their ability to solve tasks that require integrated physical and digital intelligence, such as cooking from online recipes, navigating with dynamic map data, or interpreting real-world landmarks using web knowledge. We introduce Embodied Web Agents, a novel paradigm for AI agents that fluidly bridge embodiment and web-scale reasoning. To operationalize this concept, we first develop the Embodied Web Agents task environments, a unified simulation platform that tightly integrates realistic 3D indoor and outdoor environments with functional web interfaces. Building upon this platform, we construct and release the Embodied Web Agents Benchmark, which encompasses a diverse suite of tasks including cooking, navigation, shopping, tourism, and geolocation - all requiring coordinated reasoning across physical and digital realms for systematic assessment of cross-domain intelligence. Experimental results reveal significant performance gaps between state-of-the-art AI systems and human capabilities, establishing both challenges and opportunities at the intersection of embodied cognition and web-scale knowledge access. All datasets, codes and websites are publicly available at our project page https://embodied-web-agent.github.io/.
>
---
#### [replaced 050] Probing Information Distribution in Transformer Architectures through Entropy Analysis
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.15347v2](http://arxiv.org/pdf/2507.15347v2)**

> **作者:** Amedeo Buonanno; Alessandro Rivetti; Francesco A. N. Palmieri; Giovanni Di Gennaro; Gianmarco Romano
>
> **备注:** Presented to the Italian Workshop on Neural Networks (WIRN2025) and it will appear in a Springer Chapter
>
> **摘要:** This work explores entropy analysis as a tool for probing information distribution within Transformer-based architectures. By quantifying token-level uncertainty and examining entropy patterns across different stages of processing, we aim to investigate how information is managed and transformed within these models. As a case study, we apply the methodology to a GPT-based large language model, illustrating its potential to reveal insights into model behavior and internal representations. This approach may offer insights into model behavior and contribute to the development of interpretability and evaluation frameworks for transformer-based models
>
---
#### [replaced 051] What Are They Talking About? A Benchmark of Knowledge-Grounded Discussion Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12474v2](http://arxiv.org/pdf/2505.12474v2)**

> **作者:** Weixiao Zhou; Junnan Zhu; Gengyao Li; Xianfu Cheng; Xinnian Liang; Feifei Zhai; Zhoujun Li
>
> **备注:** 20 pages, 17 figures and 8 tables
>
> **摘要:** Traditional dialogue summarization primarily focuses on dialogue content, assuming it comprises adequate information for a clear summary. However, this assumption often fails for discussions grounded in shared background, where participants frequently omit context and use implicit references. This results in summaries that are confusing to readers unfamiliar with the background. To address this, we introduce Knowledge-Grounded Discussion Summarization (KGDS), a novel task that produces a supplementary background summary for context and a clear opinion summary with clarified references. To facilitate research, we construct the first KGDS benchmark, featuring news-discussion pairs and expert-created multi-granularity gold annotations for evaluating sub-summaries. We also propose a novel hierarchical evaluation framework with fine-grained and interpretable metrics. Our extensive evaluation of 12 advanced large language models (LLMs) reveals that KGDS remains a significant challenge. The models frequently miss key facts and retain irrelevant ones in background summarization, and often fail to resolve implicit references in opinion summary integration.
>
---
#### [replaced 052] Déjà Vu: Multilingual LLM Evaluation through the Lens of Machine Translation Evaluation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.11829v3](http://arxiv.org/pdf/2504.11829v3)**

> **作者:** Julia Kreutzer; Eleftheria Briakou; Sweta Agrawal; Marzieh Fadaee; Kocmi Tom
>
> **摘要:** Generation capabilities and language coverage of multilingual large language models (mLLMs) are advancing rapidly. However, evaluation practices for generative abilities of mLLMs are still lacking comprehensiveness, scientific rigor, and consistent adoption across research labs, which undermines their potential to meaningfully guide mLLM development. We draw parallels with machine translation (MT) evaluation, a field that faced similar challenges and has, over decades, developed transparent reporting standards and reliable evaluations for multilingual generative models. Through targeted experiments across key stages of the generative evaluation pipeline, we demonstrate how best practices from MT evaluation can deepen the understanding of quality differences between models. Additionally, we identify essential components for robust meta-evaluation of mLLMs, ensuring the evaluation methods themselves are rigorously assessed. We distill these insights into a checklist of actionable recommendations for mLLM research and development.
>
---
