# 自然语言处理 cs.CL

- **最新发布 184 篇**

- **更新 131 篇**

## 最新发布

#### [new 001] Hallucinate at the Last in Long Response Generation: A Case Study on Long Document Summarization
- **分类: cs.CL**

- **简介: 该论文研究长文档摘要任务，发现LLM生成长文本时幻觉集中在后半部分，分析其与注意力机制及解码动态的关系，并探索缓解方法以提升结尾忠实性。**

- **链接: [http://arxiv.org/pdf/2505.15291v1](http://arxiv.org/pdf/2505.15291v1)**

> **作者:** Joonho Yang; Seunghyun Yoon; Hwan Chang; Byeongjeong Kim; Hwanhee Lee
>
> **备注:** 11 tables, 8 figures
>
> **摘要:** Large Language Models (LLMs) have significantly advanced text generation capabilities, including tasks like summarization, often producing coherent and fluent outputs. However, faithfulness to source material remains a significant challenge due to the generation of hallucinations. While extensive research focuses on detecting and reducing these inaccuracies, less attention has been paid to the positional distribution of hallucination within generated text, particularly in long outputs. In this work, we investigate where hallucinations occur in LLM-based long response generation, using long document summarization as a key case study. Focusing on the challenging setting of long context-aware long response generation, we find a consistent and concerning phenomenon: hallucinations tend to concentrate disproportionately in the latter parts of the generated long response. To understand this bias, we explore potential contributing factors related to the dynamics of attention and decoding over long sequences. Furthermore, we investigate methods to mitigate this positional hallucination, aiming to improve faithfulness specifically in the concluding segments of long outputs.
>
---
#### [new 002] Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）评估任务，旨在解决现有基准测试不一致、无法有效区分顶尖模型的问题。提出PSN-IRT框架（基于项目反应理论的伪双胞胎网络），分析主流基准缺陷，并展示其可构建更小但更符合人类偏好的高质量评估基准。**

- **链接: [http://arxiv.org/pdf/2505.15055v1](http://arxiv.org/pdf/2505.15055v1)**

> **作者:** Hongli Zhou; Hui Huang; Ziqing Zhao; Lvyuan Han; Huicheng Wang; Kehai Chen; Muyun Yang; Wei Bao; Jian Dong; Bing Xu; Conghui Zhu; Hailong Cao; Tiejun Zhao
>
> **摘要:** The evaluation of large language models (LLMs) via benchmarks is widespread, yet inconsistencies between different leaderboards and poor separability among top models raise concerns about their ability to accurately reflect authentic model capabilities. This paper provides a critical analysis of benchmark effectiveness, examining main-stream prominent LLM benchmarks using results from diverse models. We first propose a new framework for accurate and reliable estimations of item characteristics and model abilities. Specifically, we propose Pseudo-Siamese Network for Item Response Theory (PSN-IRT), an enhanced Item Response Theory framework that incorporates a rich set of item parameters within an IRT-grounded architecture. Based on PSN-IRT, we conduct extensive analysis which reveals significant and varied shortcomings in the measurement quality of current benchmarks. Furthermore, we demonstrate that leveraging PSN-IRT is able to construct smaller benchmarks while maintaining stronger alignment with human preference.
>
---
#### [new 003] Meta-Design Matters: A Self-Design Multi-Agent System
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于多智能体系统（MAS）自动化设计任务。针对现有MAS依赖人工设计、缺乏适应性及自动方法需验证集的问题，提出SELF-MAS框架：通过元级别自监督设计，在推理时动态生成、优化智能体配置，无需验证集，实现任务自适应。实验显示其性能优于基线方法（7.44%提升），兼具成本效益。**

- **链接: [http://arxiv.org/pdf/2505.14996v1](http://arxiv.org/pdf/2505.14996v1)**

> **作者:** Zixuan Ke; Austin Xu; Yifei Ming; Xuan-Phi Nguyen; Caiming Xiong; Shafiq Joty
>
> **摘要:** Multi-agent systems (MAS) leveraging the impressive capabilities of Large Language Models (LLMs) hold significant potential for tackling complex tasks. However, most current MAS depend on manually designed agent roles and communication protocols. These manual designs often fail to align with the underlying LLMs' strengths and struggle to adapt to novel tasks. Recent automatic MAS approaches attempt to mitigate these limitations but typically necessitate a validation-set for tuning and yield static MAS designs lacking adaptability during inference. We introduce SELF-MAS, the first self-supervised, inference-time only framework for automatic MAS design. SELF-MAS employs meta-level design to iteratively generate, evaluate, and refine MAS configurations tailored to each problem instance, without requiring a validation set. Critically, it enables dynamic agent composition and problem decomposition through meta-feedback on solvability and completeness. Experiments across math, graduate-level QA, and software engineering benchmarks, using both closed-source and open-source LLM back-bones of varying sizes, demonstrate that SELF-MAS outperforms both manual and automatic MAS baselines, achieving a 7.44% average accuracy improvement over the next strongest baseline while maintaining cost-efficiency. These findings underscore the promise of meta-level self-supervised design for creating effective and adaptive MAS.
>
---
#### [new 004] Can LLMs $\textit{understand}$ Math? -- Exploring the Pitfalls in Mathematical Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于LLMs数学推理评估任务，旨在解决现有框架仅依赖准确率评估模型推理能力的不足。提出MAPLE评分指标，通过整合错误率、冗余度和逻辑有效性，全面量化推理过程的缺陷。**

- **链接: [http://arxiv.org/pdf/2505.15623v1](http://arxiv.org/pdf/2505.15623v1)**

> **作者:** Tiasa Singha Roy; Aditeya Baral; Ayush Rajesh Jhaveri; Yusuf Baig
>
> **摘要:** Large language models (LLMs) demonstrate considerable potential in various natural language tasks but face significant challenges in mathematical reasoning, particularly in executing precise, multi-step logic. However, current evaluation frameworks judge their performance solely based on accuracy, which only accounts for the final answer. This study explores these pitfalls by employing a novel evaluation framework. We propose an evaluation metric called the MAPLE score, which holistically quantifies reasoning misalignment by integrating error rates, redundancy, and validity.
>
---
#### [new 005] Leveraging Unit Language Guidance to Advance Speech Modeling in Textless Speech-to-Speech Translation
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **简介: 该论文针对无文本语音到语音翻译任务，解决跨模态语言特征提取与跨语言长序列对齐问题。提出基于n-gram的单元语言表示，并通过多任务学习结合任务提示模型缓解源-目标语言冲突，实验显示显著性能提升。**

- **链接: [http://arxiv.org/pdf/2505.15333v1](http://arxiv.org/pdf/2505.15333v1)**

> **作者:** Yuhao Zhang; Xiangnan Ma; Kaiqi Kou; Peizhuo Liu; Weiqiao Shan; Benyou Wang; Tong Xiao; Yuxin Huang; Zhengtao Yu; Jingbo Zhu
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** The success of building textless speech-to-speech translation (S2ST) models has attracted much attention. However, S2ST still faces two main challenges: 1) extracting linguistic features for various speech signals, called cross-modal (CM), and 2) learning alignment of difference languages in long sequences, called cross-lingual (CL). We propose the unit language to overcome the two modeling challenges. The unit language can be considered a text-like representation format, constructed using $n$-gram language modeling. We implement multi-task learning to utilize the unit language in guiding the speech modeling process. Our initial results reveal a conflict when applying source and target unit languages simultaneously. We propose task prompt modeling to mitigate this conflict. We conduct experiments on four languages of the Voxpupil dataset. Our method demonstrates significant improvements over a strong baseline and achieves performance comparable to models trained with text.
>
---
#### [new 006] Self-GIVE: Associative Thinking from Limited Structured Knowledge for Enhanced Large Language Model Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Self-GIVE框架，改进LLM在结构化知识不足时的推理能力。针对GIVE方法效率低、部署困难及知识不准确问题，通过强化学习实现自动联想思维，结合知识图谱检索，提升模型推理性能，在生物医学QA任务中，7B模型表现超GPT-3.5，减少90%算力消耗。**

- **链接: [http://arxiv.org/pdf/2505.15062v1](http://arxiv.org/pdf/2505.15062v1)**

> **作者:** Jiashu He; Jinxuan Fan; Bowen Jiang; Ignacio Houine; Dan Roth; Alejandro Ribeiro
>
> **摘要:** When addressing complex questions that require new information, people often associate the question with existing knowledge to derive a sensible answer. For instance, when evaluating whether melatonin aids insomnia, one might associate "hormones helping mental disorders" with "melatonin being a hormone and insomnia a mental disorder" to complete the reasoning. Large Language Models (LLMs) also require such associative thinking, particularly in resolving scientific inquiries when retrieved knowledge is insufficient and does not directly answer the question. Graph Inspired Veracity Extrapolation (GIVE) addresses this by using a knowledge graph (KG) to extrapolate structured knowledge. However, it involves the construction and pruning of many hypothetical triplets, which limits efficiency and generalizability. We propose Self-GIVE, a retrieve-RL framework that enhances LLMs with automatic associative thinking through reinforcement learning. Self-GIVE extracts structured information and entity sets to assist the model in linking to the queried concepts. We address GIVE's key limitations: (1) extensive LLM calls and token overhead for knowledge extrapolation, (2) difficulty in deploying on smaller LLMs (3B or 7B) due to complex instructions, and (3) inaccurate knowledge from LLM pruning. Specifically, after fine-tuning using self-GIVE with a 135 node UMLS KG, it improves the performance of the Qwen2.5 3B and 7B models by up to $\textbf{28.5%$\rightarrow$71.4%}$ and $\textbf{78.6$\rightarrow$90.5%}$ in samples $\textbf{unseen}$ in challenging biomedical QA tasks. In particular, Self-GIVE allows the 7B model to match or outperform GPT3.5 turbo with GIVE, while cutting token usage by over 90\%. Self-GIVE enhances the scalable integration of structured retrieval and reasoning with associative thinking.
>
---
#### [new 007] Web-Shepherd: Advancing PRMs for Reinforcing Web Agents
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决网页代理导航中缺乏高效奖励模型的问题。提出Web-Shepherd过程奖励模型（PRM），通过构建40K规模的WebPRM数据集和评估基准WebRewardBench，实现低成本、高效率的步级轨迹评估，实验显示其性能优于GPT-4o，成本降低10%。**

- **链接: [http://arxiv.org/pdf/2505.15277v1](http://arxiv.org/pdf/2505.15277v1)**

> **作者:** Hyungjoo Chae; Sunghwan Kim; Junhee Cho; Seungone Kim; Seungjun Moon; Gyeom Hwangbo; Dongha Lim; Minjin Kim; Yeonjun Hwang; Minju Gwak; Dongwook Choi; Minseok Kang; Gwanhoon Im; ByeongUng Cho; Hyojun Kim; Jun Hee Han; Taeyoon Kwon; Minju Kim; Beong-woo Kwak; Dongjin Kang; Jinyoung Yeo
>
> **备注:** Work in progress
>
> **摘要:** Web navigation is a unique domain that can automate many repetitive real-life tasks and is challenging as it requires long-horizon sequential decision making beyond typical multimodal large language model (MLLM) tasks. Yet, specialized reward models for web navigation that can be utilized during both training and test-time have been absent until now. Despite the importance of speed and cost-effectiveness, prior works have utilized MLLMs as reward models, which poses significant constraints for real-world deployment. To address this, in this work, we propose the first process reward model (PRM) called Web-Shepherd which could assess web navigation trajectories in a step-level. To achieve this, we first construct the WebPRM Collection, a large-scale dataset with 40K step-level preference pairs and annotated checklists spanning diverse domains and difficulty levels. Next, we also introduce the WebRewardBench, the first meta-evaluation benchmark for evaluating PRMs. In our experiments, we observe that our Web-Shepherd achieves about 30 points better accuracy compared to using GPT-4o on WebRewardBench. Furthermore, when testing on WebArena-lite by using GPT-4o-mini as the policy and Web-Shepherd as the verifier, we achieve 10.9 points better performance, in 10 less cost compared to using GPT-4o-mini as the verifier. Our model, dataset, and code are publicly available at LINK.
>
---
#### [new 008] MedBrowseComp: Benchmarking Medical Deep Research and Computer Use
- **分类: cs.CL**

- **简介: 该论文提出MedBrowseComp基准，评估AI代理在医疗场景中多跳检索与整合异构知识（如临床试验、监管文件等）的能力，解决现有评估方法脱离临床实际的问题。通过1000+真实临床问题测试，揭示当前模型性能与临床需求间的显著差距，为改进医疗AI提供测试平台。**

- **链接: [http://arxiv.org/pdf/2505.14963v1](http://arxiv.org/pdf/2505.14963v1)**

> **作者:** Shan Chen; Pedro Moreira; Yuxin Xiao; Sam Schmidgall; Jeremy Warner; Hugo Aerts; Thomas Hartvigsen; Jack Gallifant; Danielle S. Bitterman
>
> **备注:** You can visit our project page at: https://moreirap12.github.io/mbc-browse-app/
>
> **摘要:** Large language models (LLMs) are increasingly envisioned as decision-support tools in clinical practice, yet safe clinical reasoning demands integrating heterogeneous knowledge bases -- trials, primary studies, regulatory documents, and cost data -- under strict accuracy constraints. Existing evaluations often rely on synthetic prompts, reduce the task to single-hop factoid queries, or conflate reasoning with open-ended generation, leaving their real-world utility unclear. To close this gap, we present MedBrowseComp, the first benchmark that systematically tests an agent's ability to reliably retrieve and synthesize multi-hop medical facts from live, domain-specific knowledge bases. MedBrowseComp contains more than 1,000 human-curated questions that mirror clinical scenarios where practitioners must reconcile fragmented or conflicting information to reach an up-to-date conclusion. Applying MedBrowseComp to frontier agentic systems reveals performance shortfalls as low as ten percent, exposing a critical gap between current LLM capabilities and the rigor demanded in clinical settings. MedBrowseComp therefore offers a clear testbed for reliable medical information seeking and sets concrete goals for future model and toolchain upgrades. You can visit our project page at: https://moreirap12.github.io/mbc-browse-app/
>
---
#### [new 009] Tracing Multilingual Factual Knowledge Acquisition in Pretraining
- **分类: cs.CL**

- **简介: 该论文研究预训练过程中多语言事实知识的获取机制。针对现有研究忽视训练过程中文本事实回忆与跨语言一致性的演变问题，以OLMo-7B为案例，追踪其发展，发现准确率和一致性随时间提升，主要由事实频率驱动，低频非英语事实受益于早期跨语言迁移，提出两种知识获取路径并开源数据。**

- **链接: [http://arxiv.org/pdf/2505.14824v1](http://arxiv.org/pdf/2505.14824v1)**

> **作者:** Yihong Liu; Mingyang Wang; Amir Hossein Kargaran; Felicia Körner; Ercong Nie; Barbara Plank; François Yvon; Hinrich Schütze
>
> **备注:** preprint
>
> **摘要:** Large Language Models (LLMs) are capable of recalling multilingual factual knowledge present in their pretraining data. However, most studies evaluate only the final model, leaving the development of factual recall and crosslingual consistency throughout pretraining largely unexplored. In this work, we trace how factual recall and crosslingual consistency evolve during pretraining, focusing on OLMo-7B as a case study. We find that both accuracy and consistency improve over time for most languages. We show that this improvement is primarily driven by the fact frequency in the pretraining corpus: more frequent facts are more likely to be recalled correctly, regardless of language. Yet, some low-frequency facts in non-English languages can still be correctly recalled. Our analysis reveals that these instances largely benefit from crosslingual transfer of their English counterparts -- an effect that emerges predominantly in the early stages of pretraining. We pinpoint two distinct pathways through which multilingual factual knowledge acquisition occurs: (1) frequency-driven learning, which is dominant and language-agnostic, and (2) crosslingual transfer, which is limited in scale and typically constrained to relation types involving named entities. We release our code and data to facilitate further research at https://github.com/cisnlp/multilingual-fact-tracing.
>
---
#### [new 010] UrduFactCheck: An Agentic Fact-Checking Framework for Urdu with Evidence Boosting and Benchmarking
- **分类: cs.CL; I.2.7**

- **简介: 该论文提出UrduFactCheck框架，针对乌尔都语低资源问题，解决大语言模型事实可靠性不足的任务。通过多策略证据检索（单语+翻译）应对乌尔都语证据稀缺，并创建两个基准数据集评估事实核查与LLM表现，实验显示其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.15063v1](http://arxiv.org/pdf/2505.15063v1)**

> **作者:** Sarfraz Ahmad; Hasan Iqbal; Momina Ahsan; Numaan Naeem; Muhammad Ahsan Riaz Khan; Arham Riaz; Muhammad Arslan Manzoor; Yuxia Wang; Preslav Nakov
>
> **备注:** 16 pages, 10 figures, 4 tables, Submitted to ARR May 2025
>
> **摘要:** The rapid use of large language models (LLMs) has raised critical concerns regarding the factual reliability of their outputs, especially in low-resource languages such as Urdu. Existing automated fact-checking solutions overwhelmingly focus on English, leaving a significant gap for the 200+ million Urdu speakers worldwide. In this work, we introduce UrduFactCheck, the first comprehensive, modular fact-checking framework specifically tailored for Urdu. Our system features a dynamic, multi-strategy evidence retrieval pipeline that combines monolingual and translation-based approaches to address the scarcity of high-quality Urdu evidence. We curate and release two new hand-annotated benchmarks: UrduFactBench for claim verification and UrduFactQA for evaluating LLM factuality. Extensive experiments demonstrate that UrduFactCheck, particularly its translation-augmented variants, consistently outperforms baselines and open-source alternatives on multiple metrics. We further benchmark twelve state-of-the-art (SOTA) LLMs on factual question answering in Urdu, highlighting persistent gaps between proprietary and open-source models. UrduFactCheck's code and datasets are open-sourced and publicly available at https://github.com/mbzuai-nlp/UrduFactCheck.
>
---
#### [new 011] Hunyuan-TurboS: Advancing Large Language Models through Mamba-Transformer Synergy and Adaptive Chain-of-Thought
- **分类: cs.CL**

- **简介: 该论文提出Hunyuan-TurboS模型，解决长序列处理与复杂推理效率问题。通过混合Transformer-Mamba MoE架构，结合自适应长-短链式推理机制，优化计算资源；采用创新模块设计与多阶段训练策略，在保持高性能（Chatbot Arena排名第7）的同时降低推理成本，实现高效大规模预训练模型新范式。**

- **链接: [http://arxiv.org/pdf/2505.15431v1](http://arxiv.org/pdf/2505.15431v1)**

> **作者:** Ao Liu; Botong Zhou; Can Xu; Chayse Zhou; ChenChen Zhang; Chengcheng Xu; Chenhao Wang; Decheng Wu; Dengpeng Wu; Dian Jiao; Dong Du; Dong Wang; Feng Zhang; Fengzong Lian; Guanghui Xu; Guanwei Zhang; Hai Wang; Haipeng Luo; Han Hu; Huilin Xu; Jiajia Wu; Jianchen Zhu; Jianfeng Yan; Jiaqi Zhu; Jihong Zhang; Jinbao Xue; Jun Xia; Junqiang Zheng; Kai Liu; Kai Zhang; Kai Zheng; Kejiao Li; Keyao Wang; Lan Jiang; Lixin Liu; Lulu Wu; Mengyuan Huang; Peijie Yu; Peiqi Wang; Qian Wang; Qianbiao Xiang; Qibin Liu; Qingfeng Sun; Richard Guo; Ruobing Xie; Saiyong Yang; Shaohua Chen; Shihui Hu; Shuai Li; Shuaipeng Li; Shuang Chen; Suncong Zheng; Tao Yang; Tian Zhang; Tinghao Yu; Weidong Han; Weijie Liu; Weijin Zhou; Weikang Wang; Wesleye Chen; Xiao Feng; Xiaoqin Ren; Xingwu Sun; Xiong Kuang; Xuemeng Huang; Xun Cao; Yanfeng Chen; Yang Du; Yang Zhen; Yangyu Tao; Yaping Deng; Yi Shen; Yigeng Hong; Yiqi Chen; Yiqing Huang; Yuchi Deng; Yue Mao; Yulong Wang; Yuyuan Zeng; Zenan Xu; Zhanhui Kang; Zhe Zhao; ZhenXiang Yan; Zheng Fang; Zhichao Hu; Zhongzhi Chen; Zhuoyu Li; Zongwei Li; Alex Yan; Ande Liang; Baitong Liu; Beiping Pan; Bin Xing; Binghong Wu; Bingxin Qu; Bolin Ni; Boyu Wu; Chen Li; Cheng Jiang; Cheng Zhang; Chengjun Liu; Chengxu Yang; Chiyu Wang; Chong Zha; Daisy Yi; Di Wang; Fanyang Lu; Fei Chen; Feifei Liu; Feng Zheng; Guanghua Yu; Guiyang Li; Guohua Wang; Haisheng Lin; Han Liu; Han Wang; Hao Fei; Hao Lu; Haoqing Jiang; Haoran Sun; Haotian Zhu; Huangjin Dai; Huankui Chen; Huawen Feng; Huihui Cai; Huxin Peng; Jackson Lv; Jiacheng Shi; Jiahao Bu; Jianbo Li; Jianglu Hu; Jiangtao Guan; Jianing Xu; Jianwei Cai; Jiarong Zhang; Jiawei Song; Jie Jiang; Jie Liu; Jieneng Yang; Jihong Zhang; Jin lv; Jing Zhao; Jinjian Li; Jinxing Liu; Jun Zhao; Juntao Guo; Kai Wang; Kan Wu; Lei Fu; Lei He; Lei Wang; Li Liu; Liang Dong; Liya Zhan; Long Cheng; Long Xu; Mao Zheng; Meng Liu; Mengkang Hu; Nanli Chen; Peirui Chen; Peng He; Pengju Pan; Pengzhi Wei; Qi Yang; Qi Yi; Roberts Wang; Rongpeng Chen; Rui Sun; Rui Yang; Ruibin Chen; Ruixu Zhou; Shaofeng Zhang; Sheng Zhang; Shihao Xu; Shuaishuai Chang; Shulin Liu; SiQi Wang; Songjia Feng; Songling Yuan; Tao Zhang; Tianjiao Lang; Tongkai Li; Wei Deng; Wei Li; Weichao Wang; Weigang Zhang; Weixuan Sun; Wen Ouyang; Wenxiang Jiao; Wenzhi Sun; Wenzhuo Jia; Xiang Zhang; Xiangyu He; Xianshun Ren; XiaoYing Zhu; Xiaolong Guo; Xiaoxue Li; Xiaoyu Ma; Xican Lu; Xinhua Feng; Xinting Huang; Xinyu Guan; Xirui Li; Xu Zhang; Xudong Gao; Xun Luo; Xuxiang Qi; Yangkun Chen; Yangyu Tao; Yanling Xiao; Yantao Mai; Yanze Chen; Yao Ding; Yeting Yang; YiFan Song; Yifan Yang; Yijiao Zhu; Yinhe Wu; Yixian Liu; Yong Yang; Yuanjun Cai; Yuanlin Tu; Yue Zhang; Yufei Huang; Yuhang Zhou; Yuhao Jiang; Yuhong Liu; Yuhui Hu; Yujin Lin; Yun Yang; Yunhao Wang; Yusong Zhang; Zekun Wu; Zelong Zhang; Zhan Yu; Zhaoliang Yang; Zhe Zhao; Zheng Li; Zhenyu Huang; Zhiguang Liu; Zhijiang Xu; Zhiqing Kui; Zhiyin Zeng; Zhiyuan Xiong; Zhuo Han; Zifan Wu; Zigang Geng; Zilong Zhao; Ziyan Tang; Ziyuan Zhu; Zonglei Zhu; Zhijiang Xu
>
> **摘要:** As Large Language Models (LLMs) rapidly advance, we introduce Hunyuan-TurboS, a novel large hybrid Transformer-Mamba Mixture of Experts (MoE) model. It synergistically combines Mamba's long-sequence processing efficiency with Transformer's superior contextual understanding. Hunyuan-TurboS features an adaptive long-short chain-of-thought (CoT) mechanism, dynamically switching between rapid responses for simple queries and deep "thinking" modes for complex problems, optimizing computational resources. Architecturally, this 56B activated (560B total) parameter model employs 128 layers (Mamba2, Attention, FFN) with an innovative AMF/MF block pattern. Faster Mamba2 ensures linear complexity, Grouped-Query Attention minimizes KV cache, and FFNs use an MoE structure. Pre-trained on 16T high-quality tokens, it supports a 256K context length and is the first industry-deployed large-scale Mamba model. Our comprehensive post-training strategy enhances capabilities via Supervised Fine-Tuning (3M instructions), a novel Adaptive Long-short CoT Fusion method, Multi-round Deliberation Learning for iterative improvement, and a two-stage Large-scale Reinforcement Learning process targeting STEM and general instruction-following. Evaluations show strong performance: overall top 7 rank on LMSYS Chatbot Arena with a score of 1356, outperforming leading models like Gemini-2.0-Flash-001 (1352) and o4-mini-2025-04-16 (1345). TurboS also achieves an average of 77.9% across 23 automated benchmarks. Hunyuan-TurboS balances high performance and efficiency, offering substantial capabilities at lower inference costs than many reasoning models, establishing a new paradigm for efficient large-scale pre-trained models.
>
---
#### [new 012] From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于教育技术领域，旨在解决LLMs直接回答问题而忽视教学策略的问题。通过在线强化学习框架，利用模拟师生互动训练7B模型，平衡教学支持与解题准确率，提升模型教学能力并保留推理能力，无需人工标注。**

- **链接: [http://arxiv.org/pdf/2505.15607v1](http://arxiv.org/pdf/2505.15607v1)**

> **作者:** David Dinucu-Jianu; Jakub Macina; Nico Daheim; Ido Hakimi; Iryna Gurevych; Mrinmaya Sachan
>
> **备注:** David Dinucu-Jianu and Jakub Macina contributed equally. Code available: https://github.com/eth-lre/PedagogicalRL
>
> **摘要:** Large language models (LLMs) can transform education, but their optimization for direct question-answering often undermines effective pedagogy which requires strategically withholding answers. To mitigate this, we propose an online reinforcement learning (RL)-based alignment framework that can quickly adapt LLMs into effective tutors using simulated student-tutor interactions by emphasizing pedagogical quality and guided problem-solving over simply giving away answers. We use our method to train a 7B parameter tutor model without human annotations which reaches similar performance to larger proprietary models like LearnLM. We introduce a controllable reward weighting to balance pedagogical support and student solving accuracy, allowing us to trace the Pareto frontier between these two objectives. Our models better preserve reasoning capabilities than single-turn SFT baselines and can optionally enhance interpretability through thinking tags that expose the model's instructional planning.
>
---
#### [new 013] Diagnosing our datasets: How does my language model learn clinical information?
- **分类: cs.CL**

- **简介: 该论文研究大型语言模型（LLMs）如何从非医疗记录的预训练数据中学习临床信息，分析其对专业术语的理解及对未经证实医学主张的反应。通过MedLingo数据集评估发现，术语频率与模型性能相关，但临床常用术语在预训练数据中罕见，且部分数据支持争议性医学说法，揭示数据与实际应用的不匹配及潜在风险，为未来数据集构建提供依据。**

- **链接: [http://arxiv.org/pdf/2505.15024v1](http://arxiv.org/pdf/2505.15024v1)**

> **作者:** Furong Jia; David Sontag; Monica Agrawal
>
> **摘要:** Large language models (LLMs) have performed well across various clinical natural language processing tasks, despite not being directly trained on electronic health record (EHR) data. In this work, we examine how popular open-source LLMs learn clinical information from large mined corpora through two crucial but understudied lenses: (1) their interpretation of clinical jargon, a foundational ability for understanding real-world clinical notes, and (2) their responses to unsupported medical claims. For both use cases, we investigate the frequency of relevant clinical information in their corresponding pretraining corpora, the relationship between pretraining data composition and model outputs, and the sources underlying this data. To isolate clinical jargon understanding, we evaluate LLMs on a new dataset MedLingo. Unsurprisingly, we find that the frequency of clinical jargon mentions across major pretraining corpora correlates with model performance. However, jargon frequently appearing in clinical notes often rarely appears in pretraining corpora, revealing a mismatch between available data and real-world usage. Similarly, we find that a non-negligible portion of documents support disputed claims that can then be parroted by models. Finally, we classified and analyzed the types of online sources in which clinical jargon and unsupported medical claims appear, with implications for future dataset composition.
>
---
#### [new 014] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出VerifyBench及VerifyBench-Hard基准，评估强化学习中参考基奖励系统的性能，解决现有基准未覆盖此领域导致的验证器准确性评估不足问题。通过高质量数据构建基准，分析显示模型（尤其小型）需改进，并为优化奖励系统提供指导。**

- **链接: [http://arxiv.org/pdf/2505.15801v1](http://arxiv.org/pdf/2505.15801v1)**

> **作者:** Yuchen Yan; Jin Jiang; Zhenbang Ren; Yijun Li; Xudong Cai; Yang Liu; Xin Xu; Mengdi Zhang; Jian Shao; Yongliang Shen; Jun Xiao; Yueting Zhuang
>
> **备注:** Dataset: https://huggingface.co/datasets/ZJU-REAL/VerifyBench
>
> **摘要:** Large reasoning models such as OpenAI o1 and DeepSeek-R1 have achieved remarkable performance in the domain of reasoning. A key component of their training is the incorporation of verifiable rewards within reinforcement learning (RL). However, existing reward benchmarks do not evaluate reference-based reward systems, leaving researchers with limited understanding of the accuracy of verifiers used in RL. In this paper, we introduce two benchmarks, VerifyBench and VerifyBench-Hard, designed to assess the performance of reference-based reward systems. These benchmarks are constructed through meticulous data collection and curation, followed by careful human annotation to ensure high quality. Current models still show considerable room for improvement on both VerifyBench and VerifyBench-Hard, especially smaller-scale models. Furthermore, we conduct a thorough and comprehensive analysis of evaluation results, offering insights for understanding and developing reference-based reward systems. Our proposed benchmarks serve as effective tools for guiding the development of verifier accuracy and the reasoning capabilities of models trained via RL in reasoning tasks.
>
---
#### [new 015] On the Generalization vs Fidelity Paradox in Knowledge Distillation
- **分类: cs.CL**

- **简介: 该论文研究知识蒸馏（KD）在模型压缩中的效果及机制，探讨其对小型与大型模型的差异化影响。通过大规模实验发现，蒸馏显著提升小型模型性能（最高10%），但对大型模型效果有限；教师模型的专业性比性能更影响蒸馏效果，且存在精度提升与推理逻辑保真度不匹配的问题。工作包括跨规模模型的实证分析、相关性及消融实验。**

- **链接: [http://arxiv.org/pdf/2505.15442v1](http://arxiv.org/pdf/2505.15442v1)**

> **作者:** Suhas Kamasetty Ramesh; Ayan Sengupta; Tanmoy Chakraborty
>
> **摘要:** Knowledge distillation (KD) is a key technique for compressing large language models into smaller ones while preserving performance. Despite the recent traction of KD research, its effectiveness for smaller language models (LMs) and the mechanisms driving knowledge transfer remain underexplored. In this work, we present the first large-scale empirical and statistical analysis of KD across models ranging from 0.5B to 7B parameters on 14 complex reasoning tasks in a zero-shot setting. Our findings reveal that KD can improve the average performance of smaller models by up to $10\%$, with a peak task specific gain of $22\%$, while providing only marginal benefits ($\sim 1.3\%$) for larger models. Surprisingly, teacher performance has a minimal impact on student outcomes, while teacher task expertise impacts KD effectiveness. A correlation study indicates that smaller LMs benefit more from KD, whereas larger LMs show diminished gains. Additionally, we uncover a misalignment between improvements in student performance and reasoning fidelity, suggesting that while KD enhances accuracy, it does not always maintain the structured decision-making processes of the teacher. Our ablation study further highlights the importance of teacher signals and logit smoothing in influencing students' performance after distillation. Overall, our study offers a comprehensive empirical and statistical assessment of KD, highlighting both its benefits and trade-offs when distilling knowledge from larger to smaller LMs.
>
---
#### [new 016] Single LLM, Multiple Roles: A Unified Retrieval-Augmented Generation Framework Using Role-Specific Token Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决多子任务优化难以统一整合的问题。提出RoleRAG框架，通过角色特定标记优化和六个模块实现动态多任务处理，利用单个LLM实例切换模块，简化部署并降低资源消耗，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.15444v1](http://arxiv.org/pdf/2505.15444v1)**

> **作者:** Yutao Zhu; Jiajie Jin; Hongjin Qian; Zheng Liu; Zhicheng Dou; Ji-Rong Wen
>
> **摘要:** Existing studies have optimized retrieval-augmented generation (RAG) across various sub-tasks, such as query understanding and retrieval refinement, but integrating these optimizations into a unified framework remains challenging. To tackle this problem, this work proposes RoleRAG, a unified RAG framework that achieves efficient multi-task processing through role-specific token optimization. RoleRAG comprises six modules, each handling a specific sub-task within the RAG process. Additionally, we introduce a query graph to represent the decomposition of the query, which can be dynamically resolved according to the decomposing state. All modules are driven by the same underlying LLM, distinguished by task-specific role tokens that are individually optimized. This design allows RoleRAG to dynamically activate different modules within a single LLM instance, thereby streamlining deployment and reducing resource consumption. Experimental results on five open-domain question-answering datasets demonstrate the effectiveness, generalizability, and flexibility of our framework.
>
---
#### [new 017] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association
- **分类: cs.CL**

- **简介: 该论文提出电商脚本规划任务EcomScript，解决LLM在同时规划购物步骤与产品推荐时存在的语义匹配困难及评估数据缺失问题。提出基于意图驱动的产品关联框架，构建首个含60万脚本的EcomScriptBench数据集，实验表明注入购买意图可提升模型性能。**

- **链接: [http://arxiv.org/pdf/2505.15196v1](http://arxiv.org/pdf/2505.15196v1)**

> **作者:** Weiqi Wang; Limeng Cui; Xin Liu; Sreyashi Nag; Wenju Xu; Chen Luo; Sheikh Muhammad Sarwar; Yang Li; Hansu Gu; Hui Liu; Changlong Yu; Jiaxin Bai; Yifan Gao; Haiyang Zhang; Qi He; Shuiwang Ji; Yangqiu Song
>
> **备注:** ACL2025
>
> **摘要:** Goal-oriented script planning, or the ability to devise coherent sequences of actions toward specific goals, is commonly employed by humans to plan for typical activities. In e-commerce, customers increasingly seek LLM-based assistants to generate scripts and recommend products at each step, thereby facilitating convenient and efficient shopping experiences. However, this capability remains underexplored due to several challenges, including the inability of LLMs to simultaneously conduct script planning and product retrieval, difficulties in matching products caused by semantic discrepancies between planned actions and search queries, and a lack of methods and benchmark data for evaluation. In this paper, we step forward by formally defining the task of E-commerce Script Planning (EcomScript) as three sequential subtasks. We propose a novel framework that enables the scalable generation of product-enriched scripts by associating products with each step based on the semantic similarity between the actions and their purchase intentions. By applying our framework to real-world e-commerce data, we construct the very first large-scale EcomScript dataset, EcomScriptBench, which includes 605,229 scripts sourced from 2.4 million products. Human annotations are then conducted to provide gold labels for a sampled subset, forming an evaluation benchmark. Extensive experiments reveal that current (L)LMs face significant challenges with EcomScript tasks, even after fine-tuning, while injecting product purchase intentions improves their performance.
>
---
#### [new 018] SciCUEval: A Comprehensive Dataset for Evaluating Scientific Context Understanding in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出SciCUEval数据集，用于评估LLMs在科学领域的上下文理解能力。针对现有评测未覆盖科学领域复杂性的不足，构建了涵盖生物学、化学等五领域的10个子数据集，整合表格、知识图谱和文本等模态，评估信息识别、缺失检测、多源整合及推理四项能力，并测试了现有模型的性能。**

- **链接: [http://arxiv.org/pdf/2505.15094v1](http://arxiv.org/pdf/2505.15094v1)**

> **作者:** Jing Yu; Yuqi Tang; Kehua Feng; Mingyang Rao; Lei Liang; Zhiqiang Zhang; Mengshu Sun; Wen Zhang; Qiang Zhang; Keyan Ding; Huajun Chen
>
> **备注:** 25 pages, 4 figures
>
> **摘要:** Large Language Models (LLMs) have shown impressive capabilities in contextual understanding and reasoning. However, evaluating their performance across diverse scientific domains remains underexplored, as existing benchmarks primarily focus on general domains and fail to capture the intricate complexity of scientific data. To bridge this gap, we construct SciCUEval, a comprehensive benchmark dataset tailored to assess the scientific context understanding capability of LLMs. It comprises ten domain-specific sub-datasets spanning biology, chemistry, physics, biomedicine, and materials science, integrating diverse data modalities including structured tables, knowledge graphs, and unstructured texts. SciCUEval systematically evaluates four core competencies: Relevant information identification, Information-absence detection, Multi-source information integration, and Context-aware inference, through a variety of question formats. We conduct extensive evaluations of state-of-the-art LLMs on SciCUEval, providing a fine-grained analysis of their strengths and limitations in scientific context understanding, and offering valuable insights for the future development of scientific-domain LLMs.
>
---
#### [new 019] DUSK: Do Not Unlearn Shared Knowledge
- **分类: cs.CL**

- **简介: 该论文属于机器unlearning任务，针对现有方法无法区分遗忘数据与保留数据间共享知识的问题，提出DUSK基准。其构建风格不同但事实相同的文档集，要求方法精准移除独特内容并保留共享事实，通过7项指标评估9种方法，发现现有技术难以保留深层共享知识。**

- **链接: [http://arxiv.org/pdf/2505.15209v1](http://arxiv.org/pdf/2505.15209v1)**

> **作者:** Wonje Jeung; Sangyeon Yoon; Hyesoo Hong; Soeun Kim; Seungju Han; Youngjae Yu; Albert No
>
> **备注:** 21 pages
>
> **摘要:** Large language models (LLMs) are increasingly deployed in real-world applications, raising concerns about the unauthorized use of copyrighted or sensitive data. Machine unlearning aims to remove such 'forget' data while preserving utility and information from the 'retain' set. However, existing evaluations typically assume that forget and retain sets are fully disjoint, overlooking realistic scenarios where they share overlapping content. For instance, a news article may need to be unlearned, even though the same event, such as an earthquake in Japan, is also described factually on Wikipedia. Effective unlearning should remove the specific phrasing of the news article while preserving publicly supported facts. In this paper, we introduce DUSK, a benchmark designed to evaluate unlearning methods under realistic data overlap. DUSK constructs document sets that describe the same factual content in different styles, with some shared information appearing across all sets and other content remaining unique to each. When one set is designated for unlearning, an ideal method should remove its unique content while preserving shared facts. We define seven evaluation metrics to assess whether unlearning methods can achieve this selective removal. Our evaluation of nine recent unlearning methods reveals a key limitation: while most can remove surface-level text, they often fail to erase deeper, context-specific knowledge without damaging shared content. We release DUSK as a public benchmark to support the development of more precise and reliable unlearning techniques for real-world applications.
>
---
#### [new 020] DEBATE, TRAIN, EVOLVE: Self Evolution of Language Model Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型推理能力提升任务，旨在解决依赖外部数据训练推理的局限性。提出DTE框架（辩论-训练-进化）及Reflect-Critique-Refine策略，通过多智能体辩论自动生成训练数据，使模型自主优化推理。实验显示其在多个基准测试中显著提升准确率，包括GSM-PLUS的8.92%提升。**

- **链接: [http://arxiv.org/pdf/2505.15734v1](http://arxiv.org/pdf/2505.15734v1)**

> **作者:** Gaurav Srivastava; Zhenyu Bi; Meng Lu; Xuan Wang
>
> **摘要:** Large language models (LLMs) have improved significantly in their reasoning through extensive training on massive datasets. However, relying solely on additional data for improvement is becoming increasingly impractical, highlighting the need for models to autonomously enhance their reasoning without external supervision. In this paper, we propose Debate, Train, Evolve (DTE), a novel ground truth-free training framework that uses multi-agent debate traces to evolve a single language model. We also introduce a new prompting strategy Reflect-Critique-Refine, to improve debate quality by explicitly instructing agents to critique and refine their reasoning. Extensive evaluations on five reasoning benchmarks with six open-weight models show that our DTE framework achieve substantial improvements, with an average accuracy gain of 8.92% on the challenging GSM-PLUS dataset. Furthermore, we observe strong cross-domain generalization, with an average accuracy gain of 5.8% on all other benchmarks, suggesting that our method captures general reasoning capabilities.
>
---
#### [new 021] Responsible Diffusion Models via Constraining Text Embeddings within Safe Regions
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于扩散模型社会责任任务，解决其生成NSFW内容及社会偏见问题。提出通过自发现语义方向向量，在嵌入空间约束文本嵌入于安全区域，利用LoRA初始化减少性能影响，并兼容现有方法，提升模型鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.15427v1](http://arxiv.org/pdf/2505.15427v1)**

> **作者:** Zhiwen Li; Die Chen; Mingyuan Fan; Cen Chen; Yaliang Li; Yanhao Wang; Wenmeng Zhou
>
> **摘要:** The remarkable ability of diffusion models to generate high-fidelity images has led to their widespread adoption. However, concerns have also arisen regarding their potential to produce Not Safe for Work (NSFW) content and exhibit social biases, hindering their practical use in real-world applications. In response to this challenge, prior work has focused on employing security filters to identify and exclude toxic text, or alternatively, fine-tuning pre-trained diffusion models to erase sensitive concepts. Unfortunately, existing methods struggle to achieve satisfactory performance in the sense that they can have a significant impact on the normal model output while still failing to prevent the generation of harmful content in some cases. In this paper, we propose a novel self-discovery approach to identifying a semantic direction vector in the embedding space to restrict text embedding within a safe region. Our method circumvents the need for correcting individual words within the input text and steers the entire text prompt towards a safe region in the embedding space, thereby enhancing model robustness against all possibly unsafe prompts. In addition, we employ Low-Rank Adaptation (LoRA) for semantic direction vector initialization to reduce the impact on the model performance for other semantics. Furthermore, our method can also be integrated with existing methods to improve their social responsibility. Extensive experiments on benchmark datasets demonstrate that our method can effectively reduce NSFW content and mitigate social bias generated by diffusion models compared to several state-of-the-art baselines.
>
---
#### [new 022] MentalMAC: Enhancing Large Language Models for Detecting Mental Manipulation via Multi-Task Anti-Curriculum Distillation
- **分类: cs.CL**

- **简介: 该论文属于多轮对话中检测心理操纵的任务。针对标注数据稀缺及模型检测效果差的问题，提出MentalMAC方法：通过EvoSA无监督数据扩增、多任务教师监督及反向课程蒸馏提升模型性能，并构建含5000条真实对话的ReaMent数据集，实验显示显著优于现有LLMs。**

- **链接: [http://arxiv.org/pdf/2505.15255v1](http://arxiv.org/pdf/2505.15255v1)**

> **作者:** Yuansheng Gao; Han Bao; Tong Zhang; Bin Li; Zonghui Wang; Wenzhi Chen
>
> **摘要:** Mental manipulation is a subtle yet pervasive form of psychological abuse that poses serious threats to mental health. Its covert nature and the complexity of manipulation strategies make it challenging to detect, even for state-of-the-art large language models (LLMs). This concealment also hinders the manual collection of large-scale, high-quality annotations essential for training effective models. Although recent efforts have sought to improve LLM's performance on this task, progress remains limited due to the scarcity of real-world annotated datasets. To address these challenges, we propose MentalMAC, a multi-task anti-curriculum distillation method that enhances LLMs' ability to detect mental manipulation in multi-turn dialogue. Our approach includes: (i) EvoSA, an unsupervised data expansion method based on evolutionary operations and speech act theory; (ii) teacher-model-generated multi-task supervision; and (iii) progressive knowledge distillation from complex to simpler tasks. We then constructed the ReaMent dataset with 5,000 real-world dialogue samples, using a MentalMAC-distilled model to assist human annotation. Vast experiments demonstrate that our method significantly narrows the gap between student and teacher models and outperforms competitive LLMs across key evaluation metrics. All code, datasets, and checkpoints will be released upon paper acceptance. Warning: This paper contains content that may be offensive to readers.
>
---
#### [new 023] Shared Path: Unraveling Memorization in Multilingual LLMs through Language Similarities
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型的记忆机制，解决传统假设（记忆与数据量强相关）无法解释其跨语言模式的问题。提出语言相似性图关联指标，发现相似语言中训练数据少者记忆更强，揭示语言间关系对记忆评估及跨语言迁移的关键作用。**

- **链接: [http://arxiv.org/pdf/2505.15722v1](http://arxiv.org/pdf/2505.15722v1)**

> **作者:** Xiaoyu Luo; Yiyi Chen; Johannes Bjerva; Qiongxiu Li
>
> **备注:** 17 pages, 14 tables, 10 figures
>
> **摘要:** We present the first comprehensive study of Memorization in Multilingual Large Language Models (MLLMs), analyzing 95 languages using models across diverse model scales, architectures, and memorization definitions. As MLLMs are increasingly deployed, understanding their memorization behavior has become critical. Yet prior work has focused primarily on monolingual models, leaving multilingual memorization underexplored, despite the inherently long-tailed nature of training corpora. We find that the prevailing assumption, that memorization is highly correlated with training data availability, fails to fully explain memorization patterns in MLLMs. We hypothesize that treating languages in isolation - ignoring their similarities - obscures the true patterns of memorization. To address this, we propose a novel graph-based correlation metric that incorporates language similarity to analyze cross-lingual memorization. Our analysis reveals that among similar languages, those with fewer training tokens tend to exhibit higher memorization, a trend that only emerges when cross-lingual relationships are explicitly modeled. These findings underscore the importance of a language-aware perspective in evaluating and mitigating memorization vulnerabilities in MLLMs. This also constitutes empirical evidence that language similarity both explains Memorization in MLLMs and underpins Cross-lingual Transferability, with broad implications for multilingual NLP.
>
---
#### [new 024] Fooling the LVLM Judges: Visual Biases in LVLM-Based Evaluation
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究大型视觉语言模型（LVLM）在图文评估中的视觉偏差问题。任务为探究对抗性视觉篡改是否会导致LVLM误判高分。通过构建多领域基准FRAME，发现LVLM易受图像诱导偏差影响，组合偏差放大效果，且提示策略无法有效缓解，凸显现有模型脆弱性，呼吁开发更鲁棒的评估系统。**

- **链接: [http://arxiv.org/pdf/2505.15249v1](http://arxiv.org/pdf/2505.15249v1)**

> **作者:** Yerin Hwang; Dongryeol Lee; Kyungmin Min; Taegwan Kang; Yong-il Kim; Kyomin Jung
>
> **备注:** (21pgs, 12 Tables, 9 Figures)
>
> **摘要:** Recently, large vision-language models (LVLMs) have emerged as the preferred tools for judging text-image alignment, yet their robustness along the visual modality remains underexplored. This work is the first study to address a key research question: Can adversarial visual manipulations systematically fool LVLM judges into assigning unfairly inflated scores? We define potential image induced biases within the context of T2I evaluation and examine how these biases affect the evaluations of LVLM judges. Moreover, we introduce a novel, fine-grained, multi-domain meta-evaluation benchmark named FRAME, which is deliberately constructed to exhibit diverse score distributions. By introducing the defined biases into the benchmark, we reveal that all tested LVLM judges exhibit vulnerability across all domains, consistently inflating scores for manipulated images. Further analysis reveals that combining multiple biases amplifies their effects, and pairwise evaluations are similarly susceptible. Moreover, we observe that visual biases persist under prompt-based mitigation strategies, highlighting the vulnerability of current LVLM evaluation systems and underscoring the urgent need for more robust LVLM judges.
>
---
#### [new 025] A Risk Taxonomy for Evaluating AI-Powered Psychotherapy Agents
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出评估AI心理治疗代理的风险分类法，旨在解决现有方法无法有效检测治疗中潜在用户危害的问题。通过整合临床标准、专家访谈及现有工具，建立结构化评估框架，并提供应用场景示例。**

- **链接: [http://arxiv.org/pdf/2505.15108v1](http://arxiv.org/pdf/2505.15108v1)**

> **作者:** Ian Steenstra; Timothy W. Bickmore
>
> **摘要:** The proliferation of Large Language Models (LLMs) and Intelligent Virtual Agents acting as psychotherapists presents significant opportunities for expanding mental healthcare access. However, their deployment has also been linked to serious adverse outcomes, including user harm and suicide, facilitated by a lack of standardized evaluation methodologies capable of capturing the nuanced risks of therapeutic interaction. Current evaluation techniques lack the sensitivity to detect subtle changes in patient cognition and behavior during therapy sessions that may lead to subsequent decompensation. We introduce a novel risk taxonomy specifically designed for the systematic evaluation of conversational AI psychotherapists. Developed through an iterative process including review of the psychotherapy risk literature, qualitative interviews with clinical and legal experts, and alignment with established clinical criteria (e.g., DSM-5) and existing assessment tools (e.g., NEQ, UE-ATR), the taxonomy aims to provide a structured approach to identifying and assessing user/patient harms. We provide a high-level overview of this taxonomy, detailing its grounding, and discuss potential use cases. We discuss two use cases in detail: monitoring cognitive model-based risk factors during a counseling conversation to detect unsafe deviations, in both human-AI counseling sessions and in automated benchmarking of AI psychotherapists with simulated patients. The proposed taxonomy offers a foundational step towards establishing safer and more responsible innovation in the domain of AI-driven mental health support.
>
---
#### [new 026] Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!
- **分类: cs.CL**

- **简介: 该论文揭示了开源LLMs微调中的数据窃取风险：模型创建者可通过黑盒访问微调后的模型，利用后门训练提取私有数据。实验显示最高可提取76.3%的敏感数据，防御策略易被绕过，强调需紧急应对此新威胁。**

- **链接: [http://arxiv.org/pdf/2505.15656v1](http://arxiv.org/pdf/2505.15656v1)**

> **作者:** Zhexin Zhang; Yuhao Sun; Junxiao Yang; Shiyao Cui; Hongning Wang; Minlie Huang
>
> **备注:** 19 pages
>
> **摘要:** Fine-tuning on open-source Large Language Models (LLMs) with proprietary data is now a standard practice for downstream developers to obtain task-specific LLMs. Surprisingly, we reveal a new and concerning risk along with the practice: the creator of the open-source LLMs can later extract the private downstream fine-tuning data through simple backdoor training, only requiring black-box access to the fine-tuned downstream model. Our comprehensive experiments, across 4 popularly used open-source models with 3B to 32B parameters and 2 downstream datasets, suggest that the extraction performance can be strikingly high: in practical settings, as much as 76.3% downstream fine-tuning data (queries) out of a total 5,000 samples can be perfectly extracted, and the success rate can increase to 94.9% in more ideal settings. We also explore a detection-based defense strategy but find it can be bypassed with improved attack. Overall, we highlight the emergency of this newly identified data breaching risk in fine-tuning, and we hope that more follow-up research could push the progress of addressing this concerning risk. The code and data used in our experiments are released at https://github.com/thu-coai/Backdoor-Data-Extraction.
>
---
#### [new 027] X-WebAgentBench: A Multilingual Interactive Web Benchmark for Evaluating Global Agentic System
- **分类: cs.CL**

- **简介: 该论文提出X-WebAgentBench，填补多语言交互代理评估空白，解决现有研究英语主导、其他语言服务不足的问题。通过构建多语言互动网页基准，评估代理的规划与交互能力，并测试LLM及跨语言方法效果，揭示现有技术局限，推动全球代理智能发展。**

- **链接: [http://arxiv.org/pdf/2505.15372v1](http://arxiv.org/pdf/2505.15372v1)**

> **作者:** Peng Wang; Ruihan Tao; Qiguang Chen; Mengkang Hu; Libo Qin
>
> **备注:** Accepted by ACL 2025 Findings
>
> **摘要:** Recently, large language model (LLM)-based agents have achieved significant success in interactive environments, attracting significant academic and industrial attention. Despite these advancements, current research predominantly focuses on English scenarios. In reality, there are over 7,000 languages worldwide, all of which demand access to comparable agentic services. Nevertheless, the development of language agents remains inadequate for meeting the diverse requirements of multilingual agentic applications. To fill this gap, we introduce X-WebAgentBench, a novel multilingual agent benchmark in an interactive web environment, which evaluates the planning and interaction performance of language agents across multiple languages, thereby contributing to the advancement of global agent intelligence. Additionally, we assess the performance of various LLMs and cross-lingual alignment methods, examining their effectiveness in enhancing agents. Our findings reveal that even advanced models like GPT-4o, when combined with cross-lingual techniques, fail to achieve satisfactory results. We hope that X-WebAgentBench can serve as a valuable benchmark for multilingual agent scenario in real-world applications.
>
---
#### [new 028] Are Vision-Language Models Safe in the Wild? A Meme-Based Benchmark Study
- **分类: cs.CL; cs.CR; cs.CV**

- **简介: 该论文属于视觉语言模型（VLM）安全评估任务，旨在解决其在真实 meme 图像场景下的潜在风险。研究构建含5万余实例的MemeSafetyBench基准数据集，评估多款VLM在单/多轮交互中的安全性，发现 meme 显著提升有害输出概率，凸显需加强生态化安全机制。**

- **链接: [http://arxiv.org/pdf/2505.15389v1](http://arxiv.org/pdf/2505.15389v1)**

> **作者:** DongGeon Lee; Joonwon Jang; Jihae Jeong; Hwanjo Yu
>
> **摘要:** Rapid deployment of vision-language models (VLMs) magnifies safety risks, yet most evaluations rely on artificial images. This study asks: How safe are current VLMs when confronted with meme images that ordinary users share? To investigate this question, we introduce MemeSafetyBench, a 50,430-instance benchmark pairing real meme images with both harmful and benign instructions. Using a comprehensive safety taxonomy and LLM-based instruction generation, we assess multiple VLMs across single and multi-turn interactions. We investigate how real-world memes influence harmful outputs, the mitigating effects of conversational context, and the relationship between model scale and safety metrics. Our findings demonstrate that VLMs show greater vulnerability to meme-based harmful prompts than to synthetic or typographic images. Memes significantly increase harmful responses and decrease refusals compared to text-only inputs. Though multi-turn interactions provide partial mitigation, elevated vulnerability persists. These results highlight the need for ecologically valid evaluations and stronger safety mechanisms.
>
---
#### [new 029] The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文属于问答任务，研究检索增强下in-context学习机制。针对其内部原理不明确的问题，提出基于归因的方法识别注意力头角色：in-context头处理指令与检索信息，parametric头存储实体关系，并通过修改权重验证其作用，最终追踪推理中的知识来源，推动模型透明化。**

- **链接: [http://arxiv.org/pdf/2505.15807v1](http://arxiv.org/pdf/2505.15807v1)**

> **作者:** Patrick Kahardipraja; Reduan Achtibat; Thomas Wiegand; Wojciech Samek; Sebastian Lapuschkin
>
> **备注:** work in progress
>
> **摘要:** Large language models are able to exploit in-context learning to access external knowledge beyond their training data through retrieval-augmentation. While promising, its inner workings remain unclear. In this work, we shed light on the mechanism of in-context retrieval augmentation for question answering by viewing a prompt as a composition of informational components. We propose an attribution-based method to identify specialized attention heads, revealing in-context heads that comprehend instructions and retrieve relevant contextual information, and parametric heads that store entities' relational knowledge. To better understand their roles, we extract function vectors and modify their attention weights to show how they can influence the answer generation process. Finally, we leverage the gained insights to trace the sources of knowledge used during inference, paving the way towards more safe and transparent language models.
>
---
#### [new 030] MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation
- **分类: cs.CL; cs.AI; cs.LG; q-bio.BM**

- **简介: 该论文提出MolLangBench基准，评估语言驱动的分子结构识别、编辑与生成任务；解决AI在分子处理准确性和可靠性的不足；通过自动化工具和专家标注构建多模态任务，并揭示现有模型性能局限。**

- **链接: [http://arxiv.org/pdf/2505.15054v1](http://arxiv.org/pdf/2505.15054v1)**

> **作者:** Feiyang Cai; Jiahui Bai; Tao Tang; Joshua Luo; Tianyu Zhu; Ling Liu; Feng Luo
>
> **摘要:** Precise recognition, editing, and generation of molecules are essential prerequisites for both chemists and AI systems tackling various chemical tasks. We present MolLangBench, a comprehensive benchmark designed to evaluate fundamental molecule-language interface tasks: language-prompted molecular structure recognition, editing, and generation. To ensure high-quality, unambiguous, and deterministic outputs, we construct the recognition tasks using automated cheminformatics tools, and curate editing and generation tasks through rigorous expert annotation and validation. MolLangBench supports the evaluation of models that interface language with different molecular representations, including linear strings, molecular images, and molecular graphs. Evaluations of state-of-the-art models reveal significant limitations: the strongest model (o3) achieves $79.2\%$ and $78.5\%$ accuracy on recognition and editing tasks, which are intuitively simple for humans, and performs even worse on the generation task, reaching only $29.0\%$ accuracy. These results highlight the shortcomings of current AI systems in handling even preliminary molecular recognition and manipulation tasks. We hope MolLangBench will catalyze further research toward more effective and reliable AI systems for chemical applications.
>
---
#### [new 031] Collaborative Problem-Solving in an Optimization Game
- **分类: cs.CL**

- **简介: 论文提出基于对话游戏的协作优化方法，解决NP难双人旅行商问题。通过结合LLM与符号机制设计智能体，实现自博弈中45%最优解率，并支持人机协作与新图泛化。**

- **链接: [http://arxiv.org/pdf/2505.15490v1](http://arxiv.org/pdf/2505.15490v1)**

> **作者:** Isidora Jeknic; Alex Duchnowski; Alexander Koller
>
> **备注:** 23 pages, 16 figures
>
> **摘要:** Dialogue agents that support human users in solving complex tasks have received much attention recently. Many such tasks are NP-hard optimization problems that require careful collaborative exploration of the solution space. We introduce a novel dialogue game in which the agents collaboratively solve a two-player Traveling Salesman problem, along with an agent that combines LLM prompting with symbolic mechanisms for state tracking and grounding. Our best agent solves 45% of games optimally in self-play. It also demonstrates an ability to collaborate successfully with human users and generalize to unfamiliar graphs.
>
---
#### [new 032] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs
- **分类: cs.CL; cs.IR; I.2.4**

- **简介: 该论文属于知识图谱增强的大语言模型可信推理任务。针对现有方法未充分挖掘知识图结构及约束导致的幻觉问题，提出DP框架：通过渐进知识蒸馏整合结构先验，结合推理内省策略验证约束，提升推理保真度与回答可靠性，在多个数据集取得SOTA，尤其在ComplexWebQuestions提升13%。**

- **链接: [http://arxiv.org/pdf/2505.15210v1](http://arxiv.org/pdf/2505.15210v1)**

> **作者:** Jie Ma; Ning Qu; Zhitao Gao; Rui Xing; Jun Liu; Hongbin Pei; Jiang Xie; Linyun Song; Pinghui Wang; Jing Tao; Zhou Su
>
> **备注:** Under Review
>
> **摘要:** Knowledge graph-based retrieval-augmented generation seeks to mitigate hallucinations in Large Language Models (LLMs) caused by insufficient or outdated knowledge. However, existing methods often fail to fully exploit the prior knowledge embedded in knowledge graphs (KGs), particularly their structural information and explicit or implicit constraints. The former can enhance the faithfulness of LLMs' reasoning, while the latter can improve the reliability of response generation. Motivated by these, we propose a trustworthy reasoning framework, termed Deliberation over Priors (DP), which sufficiently utilizes the priors contained in KGs. Specifically, DP adopts a progressive knowledge distillation strategy that integrates structural priors into LLMs through a combination of supervised fine-tuning and Kahneman-Tversky optimization, thereby improving the faithfulness of relation path generation. Furthermore, our framework employs a reasoning-introspection strategy, which guides LLMs to perform refined reasoning verification based on extracted constraint priors, ensuring the reliability of response generation. Extensive experiments on three benchmark datasets demonstrate that DP achieves new state-of-the-art performance, especially a Hit@1 improvement of 13% on the ComplexWebQuestions dataset, and generates highly trustworthy responses. We also conduct various analyses to verify its flexibility and practicality. The code is available at https://github.com/reml-group/Deliberation-on-Priors.
>
---
#### [new 033] R-TOFU: Unlearning in Large Reasoning Models
- **分类: cs.CL**

- **简介: 该论文研究大型推理模型（LRMs）的unlearning任务，解决其多步骤推理过程残留隐私/版权信息的问题。提出R-TOFU基准，补充CoT标注与分步评估指标，对比发现传统方法遗留大量遗忘痕迹；提出Reasoned IDK优化方法，在遗忘与推理能力间取得平衡；揭示解码变体可能泄露已遗忘内容，强调多场景评估的必要性。**

- **链接: [http://arxiv.org/pdf/2505.15214v1](http://arxiv.org/pdf/2505.15214v1)**

> **作者:** Sangyeon Yoon; Wonje Jeung; Albert No
>
> **备注:** 19 pages
>
> **摘要:** Large Reasoning Models (LRMs) embed private or copyrighted information not only in their final answers but also throughout multi-step chain-of-thought (CoT) traces, making reliable unlearning far more demanding than in standard LLMs. We introduce Reasoning-TOFU (R-TOFU), the first benchmark tailored to this setting. R-TOFU augments existing unlearning tasks with realistic CoT annotations and provides step-wise metrics that expose residual knowledge invisible to answer-level checks. Using R-TOFU, we carry out a comprehensive comparison of gradient-based and preference-optimization baselines and show that conventional answer-only objectives leave substantial forget traces in reasoning. We further propose Reasoned IDK, a preference-optimization variant that preserves coherent yet inconclusive reasoning, achieving a stronger balance between forgetting efficacy and model utility than earlier refusal styles. Finally, we identify a failure mode: decoding variants such as ZeroThink and LessThink can still reveal forgotten content despite seemingly successful unlearning, emphasizing the need to evaluate models under diverse decoding settings. Together, the benchmark, analysis, and new baseline establish a systematic foundation for studying and improving unlearning in LRMs while preserving their reasoning capabilities.
>
---
#### [new 034] Mechanistic evaluation of Transformers and state space models
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文对比Transformer与状态空间模型（SSMs）的机制，解决其在关联回忆任务中表现差异的问题。通过实验发现Transformer及部分SSMs（如Based、Mamba）成功，其他SSMs失败，归因于前者通过诱导头存储关联，而SSMs仅在末状态计算，Mamba因短卷积获益。引入ATR任务验证机制差异，强调架构机制分析的重要性。**

- **链接: [http://arxiv.org/pdf/2505.15105v1](http://arxiv.org/pdf/2505.15105v1)**

> **作者:** Aryaman Arora; Neil Rathi; Nikil Roashan Selvam; Róbert Csórdas; Dan Jurafsky; Christopher Potts
>
> **备注:** 9 page main text, 6 pages appendix
>
> **摘要:** State space models (SSMs) for language modelling promise an efficient and performant alternative to quadratic-attention Transformers, yet show variable performance on recalling basic information from the context. While performance on synthetic tasks like Associative Recall (AR) can point to this deficiency, behavioural metrics provide little information as to why--on a mechanistic level--certain architectures fail and others succeed. To address this, we conduct experiments on AR and find that only Transformers and Based SSM models fully succeed at AR, with Mamba a close third, whereas the other SSMs (H3, Hyena) fail. We then use causal interventions to explain why. We find that Transformers and Based learn to store key-value associations in-context using induction heads. By contrast, the SSMs compute these associations only at the last state, with only Mamba succeeding because of its short convolution component. To extend and deepen these findings, we introduce Associative Treecall (ATR), a synthetic task similar to AR based on PCFG induction. ATR introduces language-like hierarchical structure into the AR setting. We find that all architectures learn the same mechanism as they did for AR, and the same three models succeed at the task. These results reveal that architectures with similar accuracy may still have substantive differences, motivating the adoption of mechanistic evaluations.
>
---
#### [new 035] MaxPoolBERT: Enhancing BERT Classification via Layer- and Token-Wise Aggregation
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对BERT分类任务中[CLS]标记信息利用率低的问题，提出MaxPoolBERT，通过跨层max-pooling [CLS]、最终层多头注意力及结合全序列池化的改进，增强分类表示。无需预训练且模型增益小，在GLUE基准上显著提升效果，尤其低资源任务。**

- **链接: [http://arxiv.org/pdf/2505.15696v1](http://arxiv.org/pdf/2505.15696v1)**

> **作者:** Maike Behrendt; Stefan Sylvius Wagner; Stefan Harmeling
>
> **摘要:** The [CLS] token in BERT is commonly used as a fixed-length representation for classification tasks, yet prior work has shown that both other tokens and intermediate layers encode valuable contextual information. In this work, we propose MaxPoolBERT, a lightweight extension to BERT that refines the [CLS] representation by aggregating information across layers and tokens. Specifically, we explore three modifications: (i) max-pooling the [CLS] token across multiple layers, (ii) enabling the [CLS] token to attend over the entire final layer using an additional multi-head attention (MHA) layer, and (iii) combining max-pooling across the full sequence with MHA. Our approach enhances BERT's classification accuracy (especially on low-resource tasks) without requiring pre-training or significantly increasing model size. Experiments on the GLUE benchmark show that MaxPoolBERT consistently achieves a better performance on the standard BERT-base model.
>
---
#### [new 036] Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出新型双工语音到语音模型，解决现有语音模型仅支持轮流交互、缺乏实时适应（如用户打断）的问题。通过连续输入输出、流式编码器及分离架构，提升推理、轮流与打断能力，降低比特率并简化训练流程，为首个开源方案。**

- **链接: [http://arxiv.org/pdf/2505.15670v1](http://arxiv.org/pdf/2505.15670v1)**

> **作者:** Ke Hu; Ehsan Hosseini-Asl; Chen Chen; Edresson Casanova; Subhankar Ghosh; Piotr Żelasko; Zhehuai Chen; Jason Li; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spoken dialogue is an intuitive form of human-computer interaction, yet current speech language models often remain constrained to turn-based exchanges, lacking real-time adaptability such as user barge-in. We propose a novel duplex speech to speech (S2S) architecture featuring continuous user inputs and codec agent outputs with channel fusion that directly models simultaneous user and agent streams. Using a pretrained streaming encoder for user input enables the first duplex S2S model without requiring speech pretrain. Separate architectures for agent and user modeling facilitate codec fine-tuning for better agent voices and halve the bitrate (0.6 kbps) compared to previous works. Experimental results show that the proposed model outperforms previous duplex models in reasoning, turn-taking, and barge-in abilities. The model requires significantly less speech data, as speech pretrain is skipped, which markedly simplifies the process of building a duplex S2S model from any LLMs. Finally, it is the first openly available duplex S2S model with training and inference code to foster reproducibility.
>
---
#### [new 037] RoT: Enhancing Table Reasoning with Iterative Row-Wise Traversals
- **分类: cs.CL**

- **简介: 该论文针对表格推理任务，提出RoT方法解决长链条思考（Long CoT）成本高、幻觉多的问题。通过逐行迭代遍历表格并结合反思优化，无需训练即提升推理效果与效率，在多个数据集上超越现有模型。**

- **链接: [http://arxiv.org/pdf/2505.15110v1](http://arxiv.org/pdf/2505.15110v1)**

> **作者:** Xuanliang Zhang; Dingzirui Wang; Keyan Xu; Qingfu Zhu; Wanxiang Che
>
> **摘要:** The table reasoning task, crucial for efficient data acquisition, aims to answer questions based on the given table. Recently, reasoning large language models (RLLMs) with Long Chain-of-Thought (Long CoT) significantly enhance reasoning capabilities, leading to brilliant performance on table reasoning. However, Long CoT suffers from high cost for training and exhibits low reliability due to table content hallucinations. Therefore, we propose Row-of-Thought (RoT), which performs iteratively row-wise table traversal, allowing for reasoning extension and reflection-based refinement at each traversal. Scaling reasoning length by row-wise traversal and leveraging reflection capabilities of LLMs, RoT is training-free. The sequential traversal encourages greater attention to the table, thus reducing hallucinations. Experiments show that RoT, using non-reasoning models, outperforms RLLMs by an average of 4.3%, and achieves state-of-the-art results on WikiTableQuestions and TableBench with comparable models, proving its effectiveness. Also, RoT outperforms Long CoT with fewer reasoning tokens, indicating higher efficiency.
>
---
#### [new 038] Listen to the Context: Towards Faithful Large Language Models for Retrieval Augmented Generation on Climate Questions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属检索增强生成（RAG）任务，针对气候领域模型输出偏离检索内容的问题，通过分析模型微调因素并筛选训练数据，开发ClimateGPT Faithful+，将忠实度从30%提升至57%。**

- **链接: [http://arxiv.org/pdf/2505.15633v1](http://arxiv.org/pdf/2505.15633v1)**

> **作者:** David Thulke; Jakob Kemmler; Christian Dugast; Hermann Ney
>
> **备注:** Accepted at the ClimateNLP 2025 Workshop at ACL
>
> **摘要:** Large language models that use retrieval augmented generation have the potential to unlock valuable knowledge for researchers, policymakers, and the public by making long and technical climate-related documents more accessible. While this approach can help alleviate factual hallucinations by relying on retrieved passages as additional context, its effectiveness depends on whether the model's output remains faithful to these passages. To address this, we explore the automatic assessment of faithfulness of different models in this setting. We then focus on ClimateGPT, a large language model specialised in climate science, to examine which factors in its instruction fine-tuning impact the model's faithfulness. By excluding unfaithful subsets of the model's training data, we develop ClimateGPT Faithful+, which achieves an improvement in faithfulness from 30% to 57% in supported atomic claims according to our automatic metric.
>
---
#### [new 039] Incorporating Token Usage into Prompting Strategy Evaluation
- **分类: cs.CL**

- **简介: 该论文属于大语言模型提示策略评估任务，旨在解决现有评估过度侧重性能而忽视效率的问题。提出Big-𝑂𝑡𝑜𝑘理论框架和Token Cost指标，分析不同策略的token使用与性能关系，发现过度增加token导致性能收益递减，强调需采用效率导向的评估方法。**

- **链接: [http://arxiv.org/pdf/2505.14880v1](http://arxiv.org/pdf/2505.14880v1)**

> **作者:** Chris Sypherd; Sergei Petrov; Sonny George; Vaishak Belle
>
> **备注:** 20 pages, 12 tables, 4 figures
>
> **摘要:** In recent years, large language models have demonstrated remarkable performance across diverse tasks. However, their task effectiveness is heavily dependent on the prompting strategy used to elicit output, which can vary widely in both performance and token usage. While task performance is often used to determine prompting strategy success, we argue that efficiency--balancing performance and token usage--can be a more practical metric for real-world utility. To enable this, we propose Big-$O_{tok}$, a theoretical framework for describing the token usage growth of prompting strategies, and analyze Token Cost, an empirical measure of tokens per performance. We apply these to several common prompting strategies and find that increased token usage leads to drastically diminishing performance returns. Our results validate the Big-$O_{tok}$ analyses and reinforce the need for efficiency-aware evaluations.
>
---
#### [new 040] Long-Form Information Alignment Evaluation Beyond Atomic Facts
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于长文本信息对齐评估任务，解决现有方法忽略事实间依赖关系导致难以检测拼接真实陈述形成的虚假叙述的问题。提出MontageLie基准测试验证现有模型缺陷（AUC＜65%），并设计DoveScore框架，联合验证事实准确性和事件顺序，提升评估效果超8%。**

- **链接: [http://arxiv.org/pdf/2505.15792v1](http://arxiv.org/pdf/2505.15792v1)**

> **作者:** Danna Zheng; Mirella Lapata; Jeff Z. Pan
>
> **摘要:** Information alignment evaluators are vital for various NLG evaluation tasks and trustworthy LLM deployment, reducing hallucinations and enhancing user trust. Current fine-grained methods, like FactScore, verify facts individually but neglect inter-fact dependencies, enabling subtle vulnerabilities. In this work, we introduce MontageLie, a challenging benchmark that constructs deceptive narratives by "montaging" truthful statements without introducing explicit hallucinations. We demonstrate that both coarse-grained LLM-based evaluators and current fine-grained frameworks are susceptible to this attack, with AUC-ROC scores falling below 65%. To enable more robust fine-grained evaluation, we propose DoveScore, a novel framework that jointly verifies factual accuracy and event-order consistency. By modeling inter-fact relationships, DoveScore outperforms existing fine-grained methods by over 8%, providing a more robust solution for long-form text alignment evaluation. Our code and datasets are available at https://github.com/dannalily/DoveScore.
>
---
#### [new 041] Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning
- **分类: cs.CL; cs.AI; cs.MM**

- **简介: 该论文针对LLM/MLLM推理任务中过度依赖长链推理导致效率低且性能下降的问题，提出Certainty-based Adaptive Routing（CAR）框架。通过评估模型输出困惑度动态切换短答案或详细推理，平衡准确率与效率，在多模态和文本推理任务中表现更优。**

- **链接: [http://arxiv.org/pdf/2505.15154v1](http://arxiv.org/pdf/2505.15154v1)**

> **作者:** Jinghui Lu; Haiyang Yu; Siliang Xu; Shiwei Ran; Guozhi Tang; Siqi Wang; Bin Shan; Teng Fu; Hao Feng; Jingqun Tang; Han Wang; Can Huang
>
> **摘要:** Recent advancements in reasoning have significantly enhanced the capabilities of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) across diverse tasks. However, excessive reliance on chain-of-thought (CoT) reasoning can impair model performance and brings unnecessarily lengthened outputs, reducing efficiency. Our work reveals that prolonged reasoning does not universally improve accuracy and even degrade performance on simpler tasks. To address this, we propose Certainty-based Adaptive Reasoning (CAR), a novel framework that dynamically switches between short answers and long-form reasoning based on the model perplexity. CAR first generates a short answer and evaluates its perplexity, triggering reasoning only when the model exhibits low confidence (i.e., high perplexity). Experiments across diverse multimodal VQA/KIE benchmarks and text reasoning datasets show that CAR outperforms both short-answer and long-form reasoning approaches, striking an optimal balance between accuracy and efficiency.
>
---
#### [new 042] Concept Incongruence: An Exploration of Time and Death in Role Playing
- **分类: cs.CL**

- **简介: 该论文属于分析大语言模型在概念冲突场景下的行为任务，旨在解决角色扮演中时间与死亡引发的生成不一致问题。研究提出三指标量化模型在角色死亡后的行为偏差，发现模型因死亡状态编码不稳定及时间表示偏移导致表现下降，进而提出改进方法以增强模型一致性。**

- **链接: [http://arxiv.org/pdf/2505.14905v1](http://arxiv.org/pdf/2505.14905v1)**

> **作者:** Xiaoyan Bai; Ike Peng; Aditya Singh; Chenhao Tan
>
> **备注:** Our code is available, see https://github.com/ChicagoHAI/concept-incongruence.git
>
> **摘要:** Consider this prompt "Draw a unicorn with two horns". Should large language models (LLMs) recognize that a unicorn has only one horn by definition and ask users for clarifications, or proceed to generate something anyway? We introduce concept incongruence to capture such phenomena where concept boundaries clash with each other, either in user prompts or in model representations, often leading to under-specified or mis-specified behaviors. In this work, we take the first step towards defining and analyzing model behavior under concept incongruence. Focusing on temporal boundaries in the Role-Play setting, we propose three behavioral metrics--abstention rate, conditional accuracy, and answer rate--to quantify model behavior under incongruence due to the role's death. We show that models fail to abstain after death and suffer from an accuracy drop compared to the Non-Role-Play setting. Through probing experiments, we identify two main causes: (i) unreliable encoding of the "death" state across different years, leading to unsatisfactory abstention behavior, and (ii) role playing causes shifts in the model's temporal representations, resulting in accuracy drops. We leverage these insights to improve consistency in the model's abstention and answer behaviors. Our findings suggest that concept incongruence leads to unexpected model behaviors and point to future directions on improving model behavior under concept incongruence.
>
---
#### [new 043] EasyMath: A 0-shot Math Benchmark for SLMs
- **分类: cs.CL; cs.AI; cs.LG; I.2.6; I.2.7**

- **简介: 该论文提出EasyMath：一个针对小规模语言模型（SLMs）的零样本数学推理基准测试。旨在解决现有评估方法对小模型适用性不足的问题，覆盖13类基础数学任务（如算术、代数、应用题等）。通过测试23种不同规模模型，分析参数量、训练数据及链式思维对数学解题准确率的影响，发现模型性能随规模增长提升，链式思维有小幅增益。**

- **链接: [http://arxiv.org/pdf/2505.14852v1](http://arxiv.org/pdf/2505.14852v1)**

> **作者:** Drishya Karki; Michiel Kamphuis; Angelecia Frey
>
> **备注:** 17 pages, 9 figures, 8 tables
>
> **摘要:** EasyMath is a compact benchmark for practical math reasoning in small language models. It covers thirteen categories, from basic arithmetic and order of operations to word problems, algebraic expressions, edge cases, and omits specialist topics. We tested 23 models (14M to 4B parameters) using exact, numerical, and symbolic checks on free-form answers in a zero-shot setting. Accuracy rises with size and training, chain-of-thought adds modest gains, and consistency improves at scale.
>
---
#### [new 044] Improving the fact-checking performance of language models by relying on their entailment ability
- **分类: cs.CL**

- **简介: 该论文属于自动化事实核查任务，旨在解决现有方法依赖语言模型内置知识易幻觉或微调效果差的问题。提出利用语言模型的蕴含与生成能力，通过训练模型生成支持/反驳论证，并系统比较不同策略。实验显示，其方法在RAW-FC等数据集上提升达28.57%-44.26%。**

- **链接: [http://arxiv.org/pdf/2505.15050v1](http://arxiv.org/pdf/2505.15050v1)**

> **作者:** Gaurav Kumar; Debajyoti Mazumder; Ayush Garg; Jasabanta Patro
>
> **备注:** 44 pages
>
> **摘要:** Automated fact-checking is a crucial task in this digital age. To verify a claim, current approaches majorly follow one of two strategies i.e. (i) relying on embedded knowledge of language models, and (ii) fine-tuning them with evidence pieces. While the former can make systems to hallucinate, the later have not been very successful till date. The primary reason behind this is that fact verification is a complex process. Language models have to parse through multiple pieces of evidence before making a prediction. Further, the evidence pieces often contradict each other. This makes the reasoning process even more complex. We proposed a simple yet effective approach where we relied on entailment and the generative ability of language models to produce ''supporting'' and ''refuting'' justifications (for the truthfulness of a claim). We trained language models based on these justifications and achieved superior results. Apart from that, we did a systematic comparison of different prompting and fine-tuning strategies, as it is currently lacking in the literature. Some of our observations are: (i) training language models with raw evidence sentences registered an improvement up to 8.20% in macro-F1, over the best performing baseline for the RAW-FC dataset, (ii) similarly, training language models with prompted claim-evidence understanding (TBE-2) registered an improvement (with a margin up to 16.39%) over the baselines for the same dataset, (iii) training language models with entailed justifications (TBE-3) outperformed the baselines by a huge margin (up to 28.57% and 44.26% for LIAR-RAW and RAW-FC, respectively). We have shared our code repository to reproduce the results.
>
---
#### [new 045] Trends and Challenges in Authorship Analysis: A Review of ML, DL, and LLM Approaches
- **分类: cs.CL**

- **简介: 该论文属于作者身份分析任务，聚焦作者归属与验证子问题，旨在解决现有方法在低资源语言、跨领域泛化等挑战。通过系统综述2015-2024年ML/DL/LLM方法，分析技术演进、特征提取及数据集，指出研究空白并提出未来方向，助力开发更可靠的文本分析系统。**

- **链接: [http://arxiv.org/pdf/2505.15422v1](http://arxiv.org/pdf/2505.15422v1)**

> **作者:** Nudrat Habib; Tosin Adewumi; Marcus Liwicki; Elisa Barney
>
> **备注:** 25 pages, 3 figures
>
> **摘要:** Authorship analysis plays an important role in diverse domains, including forensic linguistics, academia, cybersecurity, and digital content authentication. This paper presents a systematic literature review on two key sub-tasks of authorship analysis; Author Attribution and Author Verification. The review explores SOTA methodologies, ranging from traditional ML approaches to DL models and LLMs, highlighting their evolution, strengths, and limitations, based on studies conducted from 2015 to 2024. Key contributions include a comprehensive analysis of methods, techniques, their corresponding feature extraction techniques, datasets used, and emerging challenges in authorship analysis. The study highlights critical research gaps, particularly in low-resource language processing, multilingual adaptation, cross-domain generalization, and AI-generated text detection. This review aims to help researchers by giving an overview of the latest trends and challenges in authorship analysis. It also points out possible areas for future study. The goal is to support the development of better, more reliable, and accurate authorship analysis system in diverse textual domain.
>
---
#### [new 046] Can Large Language Models Understand Internet Buzzwords Through User-Generated Content
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）能否通过中文社交媒体的用户生成内容（UGC）准确生成网络流行语定义。任务为定义生成，解决LLMs理解新兴词汇的挑战。工作包括：构建首个中文流行语数据集CHEER，提出RESS方法优化LLMs的推理与学习， benchmark对比方法效果，揭示模型依赖先验知识、推理不足及UGC筛选难题。**

- **链接: [http://arxiv.org/pdf/2505.15071v1](http://arxiv.org/pdf/2505.15071v1)**

> **作者:** Chen Huang; Junkai Luo; Xinzuo Wang; Wenqiang Lei; Jiancheng Lv
>
> **备注:** ACL 2025 Main Paper. Our dataset and code are available at https://github.com/SCUNLP/Buzzword
>
> **摘要:** The massive user-generated content (UGC) available in Chinese social media is giving rise to the possibility of studying internet buzzwords. In this paper, we study if large language models (LLMs) can generate accurate definitions for these buzzwords based on UGC as examples. Our work serves a threefold contribution. First, we introduce CHEER, the first dataset of Chinese internet buzzwords, each annotated with a definition and relevant UGC. Second, we propose a novel method, called RESS, to effectively steer the comprehending process of LLMs to produce more accurate buzzword definitions, mirroring the skills of human language learning. Third, with CHEER, we benchmark the strengths and weaknesses of various off-the-shelf definition generation methods and our RESS. Our benchmark demonstrates the effectiveness of RESS while revealing crucial shared challenges: over-reliance on prior exposure, underdeveloped inferential abilities, and difficulty identifying high-quality UGC to facilitate comprehension. We believe our work lays the groundwork for future advancements in LLM-based definition generation. Our dataset and code are available at https://github.com/SCUNLP/Buzzword.
>
---
#### [new 047] Evaluate Bias without Manual Test Sets: A Concept Representation Perspective for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM偏见评估任务，解决传统方法依赖人工标注且覆盖有限的问题。提出BiasLens框架，通过结合概念激活向量（CAVs）与稀疏自编码器（SAEs）提取概念表示，量化目标概念与参考概念间的相似性差异，无需测试集即可检测偏见，提升效率与覆盖范围，促进模型公平性。**

- **链接: [http://arxiv.org/pdf/2505.15524v1](http://arxiv.org/pdf/2505.15524v1)**

> **作者:** Lang Gao; Kaiyang Wan; Wei Liu; Chenxi Wang; Zirui Song; Zixiang Xu; Yanbo Wang; Veselin Stoyanov; Xiuying Chen
>
> **摘要:** Bias in Large Language Models (LLMs) significantly undermines their reliability and fairness. We focus on a common form of bias: when two reference concepts in the model's concept space, such as sentiment polarities (e.g., "positive" and "negative"), are asymmetrically correlated with a third, target concept, such as a reviewing aspect, the model exhibits unintended bias. For instance, the understanding of "food" should not skew toward any particular sentiment. Existing bias evaluation methods assess behavioral differences of LLMs by constructing labeled data for different social groups and measuring model responses across them, a process that requires substantial human effort and captures only a limited set of social concepts. To overcome these limitations, we propose BiasLens, a test-set-free bias analysis framework based on the structure of the model's vector space. BiasLens combines Concept Activation Vectors (CAVs) with Sparse Autoencoders (SAEs) to extract interpretable concept representations, and quantifies bias by measuring the variation in representational similarity between the target concept and each of the reference concepts. Even without labeled data, BiasLens shows strong agreement with traditional bias evaluation metrics (Spearman correlation r > 0.85). Moreover, BiasLens reveals forms of bias that are difficult to detect using existing methods. For example, in simulated clinical scenarios, a patient's insurance status can cause the LLM to produce biased diagnostic assessments. Overall, BiasLens offers a scalable, interpretable, and efficient paradigm for bias discovery, paving the way for improving fairness and transparency in LLMs.
>
---
#### [new 048] CRAFT: Training-Free Cascaded Retrieval for Tabular QA
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于表格问答（TQA）任务，旨在解决传统密集检索模型计算成本高且需频繁微调的问题。提出CRAFT方法，通过级联检索（先稀疏模型筛选，再结合密集模型和重排序）提升效率，同时用Gemini生成表格描述增强表示，最终在NQ-Tables数据集验证其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.14984v1](http://arxiv.org/pdf/2505.14984v1)**

> **作者:** Adarsh Singh; Kushal Raj Bhandari; Jianxi Gao; Soham Dan; Vivek Gupta
>
> **摘要:** Table Question Answering (TQA) involves retrieving relevant tables from a large corpus to answer natural language queries. Traditional dense retrieval models, such as DTR and ColBERT, not only incur high computational costs for large-scale retrieval tasks but also require retraining or fine-tuning on new datasets, limiting their adaptability to evolving domains and knowledge. In this work, we propose $\textbf{CRAFT}$, a cascaded retrieval approach that first uses a sparse retrieval model to filter a subset of candidate tables before applying more computationally expensive dense models and neural re-rankers. Our approach achieves better retrieval performance than state-of-the-art (SOTA) sparse, dense, and hybrid retrievers. We further enhance table representations by generating table descriptions and titles using Gemini Flash 1.5. End-to-end TQA results using various Large Language Models (LLMs) on NQ-Tables, a subset of the Natural Questions Dataset, demonstrate $\textbf{CRAFT}$ effectiveness.
>
---
#### [new 049] Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）安全策略评估任务，旨在解决LLM在敏感场景中对抗间接攻击时的信息泄露问题。研究构建了基准数据集CoPriva，通过设计直接和间接攻击性问答测试LLM对用户安全策略的遵守情况，发现现有模型易泄露敏感信息，尤其在间接攻击下表现薄弱，强调需改进安全对齐方法。**

- **链接: [http://arxiv.org/pdf/2505.15805v1](http://arxiv.org/pdf/2505.15805v1)**

> **作者:** Hwan Chang; Yumin Kim; Yonghyun Jun; Hwanhee Lee
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security.
>
---
#### [new 050] AdUE: Improving uncertainty estimation head for LoRA adapters in LLMs
- **分类: cs.CL; stat.ML**

- **简介: 该论文针对大语言模型（LLM）使用LoRA适配器进行参数高效微调时的不确定性估计问题，提出AdUE方法。通过可微最大函数近似与L2-SP正则化优化分类头，提升softmax置信度的校准性，实验证明其优于现有基线，且无需修改基础模型。**

- **链接: [http://arxiv.org/pdf/2505.15443v1](http://arxiv.org/pdf/2505.15443v1)**

> **作者:** Artem Zabolotnyi; Roman Makarov; Mile Mitrovic; Polina Proskura; Oleg Travkin; Roman Alferov; Alexey Zaytsev
>
> **备注:** 9 pages, 1 figure
>
> **摘要:** Uncertainty estimation remains a critical challenge in adapting pre-trained language models to classification tasks, particularly under parameter-efficient fine-tuning approaches such as adapters. We introduce AdUE1, an efficient post-hoc uncertainty estimation (UE) method, to enhance softmax-based estimates. Our approach (1) uses a differentiable approximation of the maximum function and (2) applies additional regularization through L2-SP, anchoring the fine-tuned head weights and regularizing the model. Evaluations on five NLP classification datasets across four language models (RoBERTa, ELECTRA, LLaMA-2, Qwen) demonstrate that our method consistently outperforms established baselines such as Mahalanobis distance and softmax response. Our approach is lightweight (no base-model changes) and produces better-calibrated confidence.
>
---
#### [new 051] ThinkLess: A Training-Free Inference-Efficient Method for Reducing Reasoning Redundancy
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，旨在解决Chain-of-Thought（CoT）推理冗余导致的延迟和内存消耗问题。提出ThinkLess方法：通过分析注意力机制，提前插入终止符跳过冗余推理步骤，并采用轻量后处理确保答案结构完整，无需训练即实现推理效率提升与结果质量平衡。**

- **链接: [http://arxiv.org/pdf/2505.15684v1](http://arxiv.org/pdf/2505.15684v1)**

> **作者:** Gengyang Li; Yifeng Gao; Yuming Li; Yunfang Wu
>
> **摘要:** While Chain-of-Thought (CoT) prompting improves reasoning in large language models (LLMs), the excessive length of reasoning tokens increases latency and KV cache memory usage, and may even truncate final answers under context limits. We propose ThinkLess, an inference-efficient framework that terminates reasoning generation early and maintains output quality without modifying the model. Atttention analysis reveals that answer tokens focus minimally on earlier reasoning steps and primarily attend to the reasoning terminator token, due to information migration under causal masking. Building on this insight, ThinkLess inserts the terminator token at earlier positions to skip redundant reasoning while preserving the underlying knowledge transfer. To prevent format discruption casued by early termination, ThinkLess employs a lightweight post-regulation mechanism, relying on the model's natural instruction-following ability to produce well-structured answers. Without fine-tuning or auxiliary data, ThinkLess achieves comparable accuracy to full-length CoT decoding while greatly reducing decoding time and memory consumption.
>
---
#### [new 052] Reverse Engineering Human Preferences with Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文通过强化学习逆向工程人类偏好，优化文本前缀以提升冻结LLM的评估得分。针对现有LLM评估易被操控且直接修改易检测的问题，提出生成不可检测的前置文本优化模型表现，且跨模型有效，推动评估框架改进。**

- **链接: [http://arxiv.org/pdf/2505.15795v1](http://arxiv.org/pdf/2505.15795v1)**

> **作者:** Lisa Alazraki; Tan Yi-Chern; Jon Ander Campos; Maximilian Mozes; Marek Rei; Max Bartolo
>
> **摘要:** The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.
>
---
#### [new 053] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study
- **分类: cs.CL**

- **简介: 该论文研究如何通过监督微调（SFT）提升大型推理模型（LRMs）的安全性。针对直接蒸馏安全响应效果不佳的问题，分析其三大失败模式并优化数据蒸馏；发现简短/模板化推理可媲美复杂推理且更易学，并提出混合数学推理数据平衡安全与过拒。任务属模型安全优化，解决推理能力与安全性不匹配的矛盾。**

- **链接: [http://arxiv.org/pdf/2505.15404v1](http://arxiv.org/pdf/2505.15404v1)**

> **作者:** Zhexin Zhang; Xian Qi Loye; Victor Shea-Jay Huang; Junxiao Yang; Qi Zhu; Shiyao Cui; Fei Mi; Lifeng Shang; Yingkang Wang; Hongning Wang; Minlie Huang
>
> **备注:** 19 pages
>
> **摘要:** Large Reasoning Models (LRMs) have achieved remarkable success on reasoning-intensive tasks such as mathematics and programming. However, their enhanced reasoning capabilities do not necessarily translate to improved safety performance-and in some cases, may even degrade it. This raises an important research question: how can we enhance the safety of LRMs? In this paper, we present a comprehensive empirical study on how to enhance the safety of LRMs through Supervised Fine-Tuning (SFT). Our investigation begins with an unexpected observation: directly distilling safe responses from DeepSeek-R1 fails to significantly enhance safety. We analyze this phenomenon and identify three key failure patterns that contribute to it. We then demonstrate that explicitly addressing these issues during the data distillation process can lead to substantial safety improvements. Next, we explore whether a long and complex reasoning process is necessary for achieving safety. Interestingly, we find that simply using short or template-based reasoning process can attain comparable safety performance-and are significantly easier for models to learn than more intricate reasoning chains. These findings prompt a deeper reflection on the role of reasoning in ensuring safety. Finally, we find that mixing math reasoning data during safety fine-tuning is helpful to balance safety and over-refusal. Overall, we hope our empirical study could provide a more holistic picture on enhancing the safety of LRMs. The code and data used in our experiments are released in https://github.com/thu-coai/LRM-Safety-Study.
>
---
#### [new 054] Nek Minit: Harnessing Pragmatic Metacognitive Prompting for Explainable Sarcasm Detection of Australian and Indian English
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦可解释的讽刺检测任务，针对澳大利亚与印度英语的地域差异。旨在解决地域相关讽刺因语境复杂导致的识别困难问题。工作包括：对BESSTIE数据集手动添加讽刺解释，利用PMP优化提示策略，对比实验显示在GEMMA和LLAMA模型中显著优于其他方法，并探索代理提示技术缓解上下文限制。**

- **链接: [http://arxiv.org/pdf/2505.15095v1](http://arxiv.org/pdf/2505.15095v1)**

> **作者:** Ishmanbir Singh; Dipankar Srirag; Aditya Joshi
>
> **备注:** Under review. 4 pages + references
>
> **摘要:** Sarcasm is a challenge to sentiment analysis because of the incongruity between stated and implied sentiment. The challenge is exacerbated when the implication may be relevant to a specific country or geographical region. Pragmatic metacognitive prompting (PMP) is a cognition-inspired technique that has been used for pragmatic reasoning. In this paper, we harness PMP for explainable sarcasm detection for Australian and Indian English, alongside a benchmark dataset for standard English. We manually add sarcasm explanations to an existing sarcasm-labeled dataset for Australian and Indian English called BESSTIE, and compare the performance for explainable sarcasm detection for them with FLUTE, a standard English dataset containing sarcasm explanations. Our approach utilising PMP when evaluated on two open-weight LLMs (GEMMA and LLAMA) achieves statistically significant performance improvement across all tasks and datasets when compared with four alternative prompting strategies. We also find that alternative techniques such as agentic prompting mitigate context-related failures by enabling external knowledge retrieval. The focused contribution of our work is utilising PMP in generating sarcasm explanations for varieties of English.
>
---
#### [new 055] "Alexa, can you forget me?" Machine Unlearning Benchmark in Spoken Language Understanding
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于机器卸载学习在语音语言理解（SLU）任务，解决复杂语音场景中数据删除方法的有效性及可行性问题。提出首个基准UnSLU-BENCH，涵盖四语言数据集，评估八种技术并提出新指标，量化其效果、效用和效率差异，为"被遗忘权"请求提供评估基础。**

- **链接: [http://arxiv.org/pdf/2505.15700v1](http://arxiv.org/pdf/2505.15700v1)**

> **作者:** Alkis Koudounas; Claudio Savelli; Flavio Giobergia; Elena Baralis
>
> **摘要:** Machine unlearning, the process of efficiently removing specific information from machine learning models, is a growing area of interest for responsible AI. However, few studies have explored the effectiveness of unlearning methods on complex tasks, particularly speech-related ones. This paper introduces UnSLU-BENCH, the first benchmark for machine unlearning in spoken language understanding (SLU), focusing on four datasets spanning four languages. We address the unlearning of data from specific speakers as a way to evaluate the quality of potential "right to be forgotten" requests. We assess eight unlearning techniques and propose a novel metric to simultaneously better capture their efficacy, utility, and efficiency. UnSLU-BENCH sets a foundation for unlearning in SLU and reveals significant differences in the effectiveness and computational feasibility of various techniques.
>
---
#### [new 056] ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM代理决策任务，旨在解决ReAct方法因内部信念不一致和目标偏离导致的推理错误问题。提出ReflAct框架，通过持续反思代理状态与目标的关联，强化决策与目标的实时一致性，提升可靠性。实验显示其在ALFWorld成功率超ReAct 27.7%。**

- **链接: [http://arxiv.org/pdf/2505.15182v1](http://arxiv.org/pdf/2505.15182v1)**

> **作者:** Jeonghye Kim; Sojeong Rhee; Minbeom Kim; Dohyung Kim; Sangmook Lee; Youngchul Sung; Kyomin Jung
>
> **摘要:** Recent advances in LLM agents have largely built on reasoning backbones like ReAct, which interleave thought and action in complex environments. However, ReAct often produces ungrounded or incoherent reasoning steps, leading to misalignment between the agent's actual state and goal. Our analysis finds that this stems from ReAct's inability to maintain consistent internal beliefs and goal alignment, causing compounding errors and hallucinations. To address this, we introduce ReflAct, a novel backbone that shifts reasoning from merely planning next actions to continuously reflecting on the agent's state relative to its goal. By explicitly grounding decisions in states and enforcing ongoing goal alignment, ReflAct dramatically improves strategic reliability. This design delivers substantial empirical gains: ReflAct surpasses ReAct by 27.7% on average, achieving a 93.3% success rate in ALFWorld. Notably, ReflAct even outperforms ReAct with added enhancement modules (e.g., Reflexion, WKM), showing that strengthening the core reasoning backbone is key to reliable agent performance.
>
---
#### [new 057] ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning
- **分类: cs.CL; cs.IR**

- **简介: 该论文属对话搜索任务，针对现有查询重构方法依赖外部监督及与检索器对齐不足的问题，提出ConvSearch-R1框架。其通过强化学习结合两阶段策略（自蒸馏预训练与奖励优化），利用检索信号直接优化查询重构，无需外部监督，在TopiOCQA等数据集显著提升性能。**

- **链接: [http://arxiv.org/pdf/2505.15776v1](http://arxiv.org/pdf/2505.15776v1)**

> **作者:** Changtai Zhu; Siyin Wang; Ruijun Feng; Kai Song; Xipeng Qiu
>
> **摘要:** Conversational search systems require effective handling of context-dependent queries that often contain ambiguity, omission, and coreference. Conversational Query Reformulation (CQR) addresses this challenge by transforming these queries into self-contained forms suitable for off-the-shelf retrievers. However, existing CQR approaches suffer from two critical constraints: high dependency on costly external supervision from human annotations or large language models, and insufficient alignment between the rewriting model and downstream retrievers. We present ConvSearch-R1, the first self-driven framework that completely eliminates dependency on external rewrite supervision by leveraging reinforcement learning to optimize reformulation directly through retrieval signals. Our novel two-stage approach combines Self-Driven Policy Warm-Up to address the cold-start problem through retrieval-guided self-distillation, followed by Retrieval-Guided Reinforcement Learning with a specially designed rank-incentive reward shaping mechanism that addresses the sparsity issue in conventional retrieval metrics. Extensive experiments on TopiOCQA and QReCC datasets demonstrate that ConvSearch-R1 significantly outperforms previous state-of-the-art methods, achieving over 10% improvement on the challenging TopiOCQA dataset while using smaller 3B parameter models without any external supervision.
>
---
#### [new 058] Likelihood Variance as Text Importance for Resampling Texts to Map Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型比较任务，旨在解决构建模型地图的高计算成本问题。通过提出基于文本对数似然方差的重采样方法，选择重要文本并减少数量，同时保持KL散度估计精度，实现高效扩展模型地图。**

- **链接: [http://arxiv.org/pdf/2505.15428v1](http://arxiv.org/pdf/2505.15428v1)**

> **作者:** Momose Oyama; Ryo Kishino; Hiroaki Yamagiwa; Hidetoshi Shimodaira
>
> **摘要:** We address the computational cost of constructing a model map, which embeds diverse language models into a common space for comparison via KL divergence. The map relies on log-likelihoods over a large text set, making the cost proportional to the number of texts. To reduce this cost, we propose a resampling method that selects important texts with weights proportional to the variance of log-likelihoods across models for each text. Our method significantly reduces the number of required texts while preserving the accuracy of KL divergence estimates. Experiments show that it achieves comparable performance to uniform sampling with about half as many texts, and also facilitates efficient incorporation of new models into an existing map. These results enable scalable and efficient construction of language model maps.
>
---
#### [new 059] In-Context Learning Boosts Speech Recognition via Human-like Adaptation to Speakers and Language Varieties
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音识别（ASR）任务，旨在提升模型对不同说话人及语言变体的适应能力。针对现有模型在低资源场景下适应性不足的问题，提出基于上下文学习（ICL）的框架，通过推理时引入少量音频-文本示例（如12句），显著降低词错率（19.7%），并验证其效果与人类适应性相似，但部分场景仍存差距。**

- **链接: [http://arxiv.org/pdf/2505.14887v1](http://arxiv.org/pdf/2505.14887v1)**

> **作者:** Nathan Roll; Calbert Graham; Yuka Tatsumi; Kim Tien Nguyen; Meghan Sumner; Dan Jurafsky
>
> **备注:** 15 pages; 3 figures
>
> **摘要:** Human listeners readily adjust to unfamiliar speakers and language varieties through exposure, but do these adaptation benefits extend to state-of-the-art spoken language models? We introduce a scalable framework that allows for in-context learning (ICL) in Phi-4 Multimodal using interleaved task prompts and audio-text pairs, and find that as few as 12 example utterances (~50 seconds) at inference time reduce word error rates by a relative 19.7% (1.2 pp.) on average across diverse English corpora. These improvements are most pronounced in low-resource varieties, when the context and target speaker match, and when more examples are provided--though scaling our procedure yields diminishing marginal returns to context length. Overall, we find that our novel ICL adaptation scheme (1) reveals a similar performance profile to human listeners, and (2) demonstrates consistent improvements to automatic speech recognition (ASR) robustness across diverse speakers and language backgrounds. While adaptation succeeds broadly, significant gaps remain for certain varieties, revealing where current models still fall short of human flexibility. We release our prompts and code on GitHub.
>
---
#### [new 060] Towards Spoken Mathematical Reasoning: Benchmarking Speech-based Models over Multi-faceted Math Problems
- **分类: cs.CL**

- **简介: 该论文提出Spoken-MQA基准，评估语音模型（级联模型及端到端语音LLM）的数学推理能力，解决其在直接算术、口语化表达及知识推理中的缺陷。实验显示模型在算术、符号依赖及复杂推理上存在不足。**

- **链接: [http://arxiv.org/pdf/2505.15000v1](http://arxiv.org/pdf/2505.15000v1)**

> **作者:** Chengwei Wei; Bin Wang; Jung-jae Kim; Nancy F. Chen
>
> **摘要:** Recent advances in large language models (LLMs) and multimodal LLMs (MLLMs) have led to strong reasoning ability across a wide range of tasks. However, their ability to perform mathematical reasoning from spoken input remains underexplored. Prior studies on speech modality have mostly focused on factual speech understanding or simple audio reasoning tasks, providing limited insight into logical step-by-step reasoning, such as that required for mathematical problem solving. To address this gap, we introduce Spoken Math Question Answering (Spoken-MQA), a new benchmark designed to evaluate the mathematical reasoning capabilities of speech-based models, including both cascade models (ASR + LLMs) and end-to-end speech LLMs. Spoken-MQA covers a diverse set of math problems, including pure arithmetic, single-step and multi-step contextual reasoning, and knowledge-oriented reasoning problems, all presented in unambiguous natural spoken language. Through extensive experiments, we find that: (1) while some speech LLMs perform competitively on contextual reasoning tasks involving basic arithmetic, they still struggle with direct arithmetic problems; (2) current LLMs exhibit a strong bias toward symbolic mathematical expressions written in LaTex and have difficulty interpreting verbalized mathematical expressions; and (3) mathematical knowledge reasoning abilities are significantly degraded in current speech LLMs.
>
---
#### [new 061] ConspEmoLLM-v2: A robust and stable model to detect sentiment-transformed conspiracy theories
- **分类: cs.CL**

- **简介: 该论文属于阴谋论检测任务，旨在解决LLM生成的伪装阴谋论（通过弱化负面情感隐藏特征）的检测难题。团队构建了ConDID-v2数据集（含LLM改写的情感中性化阴谋论文本），并基于此训练ConspEmoLLM-v2模型，提升对情感转换阴谋论的识别能力，实验显示其性能优于原有模型及基线。**

- **链接: [http://arxiv.org/pdf/2505.14917v1](http://arxiv.org/pdf/2505.14917v1)**

> **作者:** Zhiwei Liu; Paul Thompson; Jiaqi Rong; Sophia Ananiadou
>
> **备注:** work in progress
>
> **摘要:** Despite the many benefits of large language models (LLMs), they can also cause harm, e.g., through automatic generation of misinformation, including conspiracy theories. Moreover, LLMs can also ''disguise'' conspiracy theories by altering characteristic textual features, e.g., by transforming their typically strong negative emotions into a more positive tone. Although several studies have proposed automated conspiracy theory detection methods, they are usually trained using human-authored text, whose features can vary from LLM-generated text. Furthermore, several conspiracy detection models, including the previously proposed ConspEmoLLM, rely heavily on the typical emotional features of human-authored conspiracy content. As such, intentionally disguised content may evade detection. To combat such issues, we firstly developed an augmented version of the ConDID conspiracy detection dataset, ConDID-v2, which supplements human-authored conspiracy tweets with versions rewritten by an LLM to reduce the negativity of their original sentiment. The quality of the rewritten tweets was verified by combining human and LLM-based assessment. We subsequently used ConDID-v2 to train ConspEmoLLM-v2, an enhanced version of ConspEmoLLM. Experimental results demonstrate that ConspEmoLLM-v2 retains or exceeds the performance of ConspEmoLLM on the original human-authored content in ConDID, and considerably outperforms both ConspEmoLLM and several other baselines when applied to sentiment-transformed tweets in ConDID-v2. The project will be available at https://github.com/lzw108/ConspEmoLLM.
>
---
#### [new 062] The Super Emotion Dataset
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于情感分析任务，旨在解决现有NLP情感数据集缺乏标准化、规模小且领域受限的问题。通过整合多来源文本，基于Shaver的心理学情感分类体系构建了统一的大规模情感数据集，以提升跨领域情感识别研究的 consistency（一致性）。**

- **链接: [http://arxiv.org/pdf/2505.15348v1](http://arxiv.org/pdf/2505.15348v1)**

> **作者:** Enric Junqué de Fortuny
>
> **摘要:** Despite the wide-scale usage and development of emotion classification datasets in NLP, the field lacks a standardized, large-scale resource that follows a psychologically grounded taxonomy. Existing datasets either use inconsistent emotion categories, suffer from limited sample size, or focus on specific domains. The Super Emotion Dataset addresses this gap by harmonizing diverse text sources into a unified framework based on Shaver's empirically validated emotion taxonomy, enabling more consistent cross-domain emotion recognition research.
>
---
#### [new 063] An Empirical Study of the Anchoring Effect in LLMs: Existence, Mechanism, and Potential Mitigations
- **分类: cs.CL**

- **简介: 论文研究LLM中的锚定效应，探究其存在、机制及缓解方法。构建SynAnchors数据集，发现LLM普遍存在该偏见，常规策略无效，推理可部分缓解，强调需基于认知偏差的可信评估。（99字）**

- **链接: [http://arxiv.org/pdf/2505.15392v1](http://arxiv.org/pdf/2505.15392v1)**

> **作者:** Yiming Huang; Biquan Bie; Zuqiu Na; Weilin Ruan; Songxin Lei; Yutao Yue; Xinlei He
>
> **摘要:** The rise of Large Language Models (LLMs) like ChatGPT has advanced natural language processing, yet concerns about cognitive biases are growing. In this paper, we investigate the anchoring effect, a cognitive bias where the mind relies heavily on the first information as anchors to make affected judgments. We explore whether LLMs are affected by anchoring, the underlying mechanisms, and potential mitigation strategies. To facilitate studies at scale on the anchoring effect, we introduce a new dataset, SynAnchors. Combining refined evaluation metrics, we benchmark current widely used LLMs. Our findings show that LLMs' anchoring bias exists commonly with shallow-layer acting and is not eliminated by conventional strategies, while reasoning can offer some mitigation. This recontextualization via cognitive psychology urges that LLM evaluations focus not on standard benchmarks or over-optimized robustness tests, but on cognitive-bias-aware trustworthy evaluation.
>
---
#### [new 064] Effective and Efficient Schema-aware Information Extraction Using On-Device Large Language Models
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于模式感知信息提取任务，针对设备端大模型计算资源受限、幻觉及长上下文处理效率低的问题，提出DLISC方法。通过双LoRA模块分别实现模式匹配与信息抽取，并结合增量缓存减少冗余计算，提升效果与效率。**

- **链接: [http://arxiv.org/pdf/2505.14992v1](http://arxiv.org/pdf/2505.14992v1)**

> **作者:** Zhihao Wen; Sheng Liang; Yaxiong Wu; Yongyue Zhang; Yong Liu
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Information extraction (IE) plays a crucial role in natural language processing (NLP) by converting unstructured text into structured knowledge. Deploying computationally intensive large language models (LLMs) on resource-constrained devices for information extraction is challenging, particularly due to issues like hallucinations, limited context length, and high latency-especially when handling diverse extraction schemas. To address these challenges, we propose a two-stage information extraction approach adapted for on-device LLMs, called Dual-LoRA with Incremental Schema Caching (DLISC), which enhances both schema identification and schema-aware extraction in terms of effectiveness and efficiency. In particular, DLISC adopts an Identification LoRA module for retrieving the most relevant schemas to a given query, and an Extraction LoRA module for performing information extraction based on the previously selected schemas. To accelerate extraction inference, Incremental Schema Caching is incorporated to reduce redundant computation, substantially improving efficiency. Extensive experiments across multiple information extraction datasets demonstrate notable improvements in both effectiveness and efficiency.
>
---
#### [new 065] Language Mixing in Reasoning Language Models: Patterns, Impact, and Internal Causes
- **分类: cs.CL**

- **简介: 该论文研究推理语言模型中的语言混用现象，旨在分析其模式、影响及内在原因。通过跨15语言、7难度等级、18领域的系统实验，发现语言混用影响推理性能，强制使用拉丁/汉字符号可提升准确率，并揭示模型内部表征与语言脚本的关联，为优化多语言推理提供方法。**

- **链接: [http://arxiv.org/pdf/2505.14815v1](http://arxiv.org/pdf/2505.14815v1)**

> **作者:** Mingyang Wang; Lukas Lange; Heike Adel; Yunpu Ma; Jannik Strötgen; Hinrich Schütze
>
> **摘要:** Reasoning language models (RLMs) excel at complex tasks by leveraging a chain-of-thought process to generate structured intermediate steps. However, language mixing, i.e., reasoning steps containing tokens from languages other than the prompt, has been observed in their outputs and shown to affect performance, though its impact remains debated. We present the first systematic study of language mixing in RLMs, examining its patterns, impact, and internal causes across 15 languages, 7 task difficulty levels, and 18 subject areas, and show how all three factors influence language mixing. Moreover, we demonstrate that the choice of reasoning language significantly affects performance: forcing models to reason in Latin or Han scripts via constrained decoding notably improves accuracy. Finally, we show that the script composition of reasoning traces closely aligns with that of the model's internal representations, indicating that language mixing reflects latent processing preferences in RLMs. Our findings provide actionable insights for optimizing multilingual reasoning and open new directions for controlling reasoning languages to build more interpretable and adaptable RLMs.
>
---
#### [new 066] HopWeaver: Synthesizing Authentic Multi-Hop Questions Across Text Corpora
- **分类: cs.CL**

- **简介: 该论文提出HopWeaver框架，解决多跳问答数据集依赖人工标注成本高、合成方法质量低的问题。通过跨语料库识别互补文档，自动生成桥接与比较类多跳问题，构建真实推理路径，并建立评估系统。实验表明其合成数据质量媲美人工标注，成本更低，适合资源稀缺领域。**

- **链接: [http://arxiv.org/pdf/2505.15087v1](http://arxiv.org/pdf/2505.15087v1)**

> **作者:** Zhiyu Shen; Jiyuan Liu; Yunhe Pang; Yanghui Rao
>
> **备注:** 27 pages. Code will be available at [https://github.com/Zh1yuShen/HopWeaver]
>
> **摘要:** Multi-Hop Question Answering (MHQA) is crucial for evaluating the model's capability to integrate information from diverse sources. However, creating extensive and high-quality MHQA datasets is challenging: (i) manual annotation is expensive, and (ii) current synthesis methods often produce simplistic questions or require extensive manual guidance. This paper introduces HopWeaver, the first automatic framework synthesizing authentic multi-hop questions from unstructured text corpora without human intervention. HopWeaver synthesizes two types of multi-hop questions (bridge and comparison) using an innovative approach that identifies complementary documents across corpora. Its coherent pipeline constructs authentic reasoning paths that integrate information across multiple documents, ensuring synthesized questions necessitate authentic multi-hop reasoning. We further present a comprehensive system for evaluating synthesized multi-hop questions. Empirical evaluations demonstrate that the synthesized questions achieve comparable or superior quality to human-annotated datasets at a lower cost. Our approach is valuable for developing MHQA datasets in specialized domains with scarce annotated resources. The code for HopWeaver is publicly available.
>
---
#### [new 067] Protoknowledge Shapes Behaviour of LLMs in Downstream Tasks: Memorization and Generalization with Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）如何内化和利用知识图谱的"protoknowledge"进行下游任务。提出protoknowledge分类（词汇、层级、拓扑）及知识激活任务（KATs）评估方法，分析其对Text-to-SPARQL任务的影响，探索记忆与泛化的机制及语义污染问题。**

- **链接: [http://arxiv.org/pdf/2505.15501v1](http://arxiv.org/pdf/2505.15501v1)**

> **作者:** Federico Ranaldi; Andrea Zugarini; Leonardo Ranaldi; Fabio Massimo Zanzotto
>
> **摘要:** We introduce the concept of protoknowledge to formalize and measure how sequences of tokens encoding Knowledge Graphs are internalized during pretraining and utilized at inference time by Large Language Models (LLMs). Indeed, LLMs have demonstrated the ability to memorize vast amounts of token sequences during pretraining, and a central open question is how they leverage this memorization as reusable knowledge through generalization. We then categorize protoknowledge into lexical, hierarchical, and topological forms, varying on the type of knowledge that needs to be activated. We measure protoknowledge through Knowledge Activation Tasks (KATs), analyzing its general properties such as semantic bias. We then investigate the impact of protoknowledge on Text-to-SPARQL performance by varying prompting strategies depending on input conditions. To this end, we adopt a novel analysis framework that assesses whether model predictions align with the successful activation of the relevant protoknowledge for each query. This methodology provides a practical tool to explore Semantic-Level Data Contamination and serves as an effective strategy for Closed-Pretraining models.
>
---
#### [new 068] Beyond Hard and Soft: Hybrid Context Compression for Balancing Local and Global Information Retention
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出HyCo₂方法，针对长序列推理中传统上下文压缩方法导致信息丢失的问题，结合全局语义适配器与局部token保留概率策略，通过混合压缩平衡关键细节与整体语义，在减少88.8% token使用的同时提升模型性能。**

- **链接: [http://arxiv.org/pdf/2505.15774v1](http://arxiv.org/pdf/2505.15774v1)**

> **作者:** Huanxuan Liao; Wen Hu; Yao Xu; Shizhu He; Jun Zhao; Kang Liu
>
> **摘要:** Large Language Models (LLMs) encounter significant challenges in long-sequence inference due to computational inefficiency and redundant processing, driving interest in context compression techniques. Existing methods often rely on token importance to perform hard local compression or encode context into latent representations for soft global compression. However, the uneven distribution of textual content relevance and the diversity of demands for user instructions mean these approaches frequently lead to the loss of potentially valuable information. To address this, we propose $\textbf{Hy}$brid $\textbf{Co}$ntext $\textbf{Co}$mpression (HyCo$_2$) for LLMs, which integrates both global and local perspectives to guide context compression while retaining both the essential semantics and critical details for task completion. Specifically, we employ a hybrid adapter to refine global semantics with the global view, based on the observation that different adapters excel at different tasks. Then we incorporate a classification layer that assigns a retention probability to each context token based on the local view, determining whether it should be retained or discarded. To foster a balanced integration of global and local compression, we introduce auxiliary paraphrasing and completion pretraining before instruction tuning. This promotes a synergistic integration that emphasizes instruction-relevant information while preserving essential local details, ultimately balancing local and global information retention in context compression. Experiments show that our HyCo$_2$ method significantly enhances long-text reasoning while reducing token usage. It improves the performance of various LLM series by an average of 13.1\% across seven knowledge-intensive QA benchmarks. Moreover, HyCo$_2$ matches the performance of uncompressed methods while reducing token consumption by 88.8\%.
>
---
#### [new 069] StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出StepSearch框架，解决LLM在多跳QA中因全局稀疏奖励导致的性能不足问题。通过分步近端策略优化、信息增益奖励及冗余惩罚机制，结合细粒度子问题数据集训练，显著提升复杂搜索推理效果。**

- **链接: [http://arxiv.org/pdf/2505.15107v1](http://arxiv.org/pdf/2505.15107v1)**

> **作者:** Ziliang Wang; Xuhui Zheng; Kang An; Cijun Ouyang; Jialu Cai; Yuhang Wang; Yichao Wu
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Efficient multi-hop reasoning requires Large Language Models (LLMs) based agents to acquire high-value external knowledge iteratively. Previous work has explored reinforcement learning (RL) to train LLMs to perform search-based document retrieval, achieving notable improvements in QA performance, but underperform on complex, multi-hop QA resulting from the sparse rewards from global signal only. To address this gap in existing research, we introduce StepSearch, a framework for search LLMs that trained with step-wise proximal policy optimization method. It consists of richer and more detailed intermediate search rewards and token-level process supervision based on information gain and redundancy penalties to better guide each search step. We constructed a fine-grained question-answering dataset containing sub-question-level search trajectories based on open source datasets through a set of data pipeline method. On standard multi-hop QA benchmarks, it significantly outperforms global-reward baselines, achieving 11.2% and 4.2% absolute improvements for 3B and 7B models over various search with RL baselines using only 19k training data, demonstrating the effectiveness of fine-grained, stepwise supervision in optimizing deep search LLMs. Our implementation is publicly available at https://github.com/zxh20001117/StepSearch.
>
---
#### [new 070] When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners
- **分类: cs.CL**

- **简介: 该论文研究多语言推理任务，针对LLMs性能依赖高资源语言的问题，提出通过因果干预解耦语言与推理表示（语言特定表示消融），实验显示该方法在10个LLMs和11种语言上有效，无需训练且计算高效，改善跨语言泛化。**

- **链接: [http://arxiv.org/pdf/2505.15257v1](http://arxiv.org/pdf/2505.15257v1)**

> **作者:** Weixiang Zhao; Jiahe Guo; Yang Deng; Tongtong Wu; Wenxuan Zhang; Yulin Hu; Xingyu Sui; Yanyan Zhao; Wanxiang Che; Bing Qin; Tat-Seng Chua; Ting Liu
>
> **备注:** 26 pages, 13 figures
>
> **摘要:** Multilingual reasoning remains a significant challenge for large language models (LLMs), with performance disproportionately favoring high-resource languages. Drawing inspiration from cognitive neuroscience, which suggests that human reasoning functions largely independently of language processing, we hypothesize that LLMs similarly encode reasoning and language as separable components that can be disentangled to enhance multilingual reasoning. To evaluate this, we perform a causal intervention by ablating language-specific representations at inference time. Experiments on 10 open-source LLMs spanning 11 typologically diverse languages show that this language-specific ablation consistently boosts multilingual reasoning performance. Layer-wise analyses further confirm that language and reasoning representations can be effectively decoupled throughout the model, yielding improved multilingual reasoning capabilities, while preserving top-layer language features remains essential for maintaining linguistic fidelity. Compared to post-training such as supervised fine-tuning or reinforcement learning, our training-free ablation achieves comparable or superior results with minimal computational overhead. These findings shed light on the internal mechanisms underlying multilingual reasoning in LLMs and suggest a lightweight and interpretable strategy for improving cross-lingual generalization.
>
---
#### [new 071] The Pursuit of Empathy: Evaluating Small Language Models for PTSD Dialogue Support
- **分类: cs.CL; cs.AI; cs.CY; 68T50, 68T05; I.2.7; I.2.1; H.5.2**

- **简介: 该论文评估小型语言模型（0.5B-5B参数）在PTSD对话支持中的同理心能力。通过创建含1万对话的TIDE数据集，基于三因素同理心模型（情绪识别、痛苦正常化、支持性反思），比较八个小模型微调前后及前沿模型表现。发现微调提升同理心感知但受场景和用户影响，小模型存在上限。分析用户偏好差异（如老年重视验证，高学历需复杂回复），强调系统设计需考虑上下文和用户，为开发安全、高效的AI辅助工具奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.15065v1](http://arxiv.org/pdf/2505.15065v1)**

> **作者:** Suhas BN; Yash Mahajan; Dominik Mattioli; Andrew M. Sherrill; Rosa I. Arriaga; Chris W. Wiese; Saeed Abdullah
>
> **备注:** 23 pages, 3 figures
>
> **摘要:** Can small language models with 0.5B to 5B parameters meaningfully engage in trauma-informed, empathetic dialogue for individuals with PTSD? We address this question by introducing TIDE, a dataset of 10,000 two-turn dialogues spanning 500 diverse PTSD client personas and grounded in a three-factor empathy model: emotion recognition, distress normalization, and supportive reflection. All scenarios and reference responses were reviewed for realism and trauma sensitivity by a clinical psychologist specializing in PTSD. We evaluate eight small language models before and after fine-tuning, comparing their outputs to a frontier model (Claude Sonnet 3.5). Our IRB-approved human evaluation and automatic metrics show that fine-tuning generally improves perceived empathy, but gains are highly scenario- and user-dependent, with smaller models facing an empathy ceiling. Demographic analysis shows older adults value distress validation and graduate-educated users prefer nuanced replies, while gender effects are minimal. We highlight the limitations of automatic metrics and the need for context- and user-aware system design. Our findings, along with the planned release of TIDE, provide a foundation for building safe, resource-efficient, and ethically sound empathetic AI to supplement, not replace, clinical mental health care.
>
---
#### [new 072] GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文聚焦GUI视觉定位任务，针对R1-Zero训练中输入设计（长推理降效）、输出评估（奖励漏洞致定位偏差）及策略更新（过拟合易样本）问题，提出快速回答模板、奖励尺寸约束及难度感知优化方法，实现90.3%的ScreenSpot准确率，创同类模型最优。**

- **链接: [http://arxiv.org/pdf/2505.15810v1](http://arxiv.org/pdf/2505.15810v1)**

> **作者:** Yuqi Zhou; Sunhao Dai; Shuai Wang; Kaiwen Zhou; Qinqlin Jia; Junxu
>
> **摘要:** Recent Graphical User Interface (GUI) agents replicate the R1-Zero paradigm, coupling online Reinforcement Learning (RL) with explicit chain-of-thought reasoning prior to object grounding and thereby achieving substantial performance gains. In this paper, we first conduct extensive analysis experiments of three key components of that training pipeline: input design, output evaluation, and policy update-each revealing distinct challenges arising from blindly applying general-purpose RL without adapting to GUI grounding tasks. Input design: Current templates encourage the model to generate chain-of-thought reasoning, but longer chains unexpectedly lead to worse grounding performance. Output evaluation: Reward functions based on hit signals or box area allow models to exploit box size, leading to reward hacking and poor localization quality. Policy update: Online RL tends to overfit easy examples due to biases in length and sample difficulty, leading to under-optimization on harder cases. To address these issues, we propose three targeted solutions. First, we adopt a Fast Thinking Template that encourages direct answer generation, reducing excessive reasoning during training. Second, we incorporate a box size constraint into the reward function to mitigate reward hacking. Third, we revise the RL objective by adjusting length normalization and adding a difficulty-aware scaling factor, enabling better optimization on hard samples. Our GUI-G1-3B, trained on 17K public samples with Qwen2.5-VL-3B-Instruct, achieves 90.3% accuracy on ScreenSpot and 37.1% on ScreenSpot-Pro. This surpasses all prior models of similar size and even outperforms the larger UI-TARS-7B, establishing a new state-of-the-art in GUI agent grounding. The project repository is available at https://github.com/Yuqi-Zhou/GUI-G1.
>
---
#### [new 073] Advancing LLM Safe Alignment with Safety Representation Ranking
- **分类: cs.CL; cs.LG**

- **简介: 该论文属LLM安全对齐任务，解决生成有害内容问题。提出Safety Representation Ranking（SRR）框架，利用LLM内部隐藏状态编码指令与候选回复，通过相似度排序筛选安全响应，提升对抗性提示下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.15710v1](http://arxiv.org/pdf/2505.15710v1)**

> **作者:** Tianqi Du; Zeming Wei; Quan Chen; Chenheng Zhang; Yisen Wang
>
> **摘要:** The rapid advancement of large language models (LLMs) has demonstrated milestone success in a variety of tasks, yet their potential for generating harmful content has raised significant safety concerns. Existing safety evaluation approaches typically operate directly on textual responses, overlooking the rich information embedded in the model's internal representations. In this paper, we propose Safety Representation Ranking (SRR), a listwise ranking framework that selects safe responses using hidden states from the LLM itself. SRR encodes both instructions and candidate completions using intermediate transformer representations and ranks candidates via a lightweight similarity-based scorer. Our approach directly leverages internal model states and supervision at the list level to capture subtle safety signals. Experiments across multiple benchmarks show that SRR significantly improves robustness to adversarial prompts. Our code will be available upon publication.
>
---
#### [new 074] Do RAG Systems Suffer From Positional Bias?
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于RAG系统分析任务，研究LLM位置偏差对其利用相关片段和抵御干扰片段的影响。通过三个基准实验发现，尽管检索常将高干扰片段排至前10，但因相关/干扰片段均受位置影响，故实际中位置偏差效果有限，优化排序策略与随机排列效果相当。**

- **链接: [http://arxiv.org/pdf/2505.15561v1](http://arxiv.org/pdf/2505.15561v1)**

> **作者:** Florin Cuconasu; Simone Filice; Guy Horowitz; Yoelle Maarek; Fabrizio Silvestri
>
> **摘要:** Retrieval Augmented Generation enhances LLM accuracy by adding passages retrieved from an external corpus to the LLM prompt. This paper investigates how positional bias - the tendency of LLMs to weight information differently based on its position in the prompt - affects not only the LLM's capability to capitalize on relevant passages, but also its susceptibility to distracting passages. Through extensive experiments on three benchmarks, we show how state-of-the-art retrieval pipelines, while attempting to retrieve relevant passages, systematically bring highly distracting ones to the top ranks, with over 60% of queries containing at least one highly distracting passage among the top-10 retrieved passages. As a result, the impact of the LLM positional bias, which in controlled settings is often reported as very prominent by related works, is actually marginal in real scenarios since both relevant and distracting passages are, in turn, penalized. Indeed, our findings reveal that sophisticated strategies that attempt to rearrange the passages based on LLM positional preferences do not perform better than random shuffling.
>
---
#### [new 075] Language Specific Knowledge: Do Models Know Better in X than in English?
- **分类: cs.CL**

- **简介: 该论文研究多语言模型的知识差异与推理优化任务。旨在验证语言模型是否在非英语语言中掌握更多领域知识，并通过切换语言提升推理效果。工作包括提出语言特定知识（LSK）概念，利用文化数据集验证模型在多语言（含低资源语言）中的表现差异，开发LSKExtractor工具评估并提取语言特有知识，最终实现推理准确率平均10%的提升。**

- **链接: [http://arxiv.org/pdf/2505.14990v1](http://arxiv.org/pdf/2505.14990v1)**

> **作者:** Ishika Agarwal; Nimet Beyza Bozdag; Dilek Hakkani-Tür
>
> **摘要:** Code-switching is a common phenomenon of alternating between different languages in the same utterance, thought, or conversation. We posit that humans code-switch because they feel more comfortable talking about certain topics and domains in one language than another. With the rise of knowledge-intensive language models, we ask ourselves the next, natural question: Could models hold more knowledge on some topics in some language X? More importantly, could we improve reasoning by changing the language that reasoning is performed in? We coin the term Language Specific Knowledge (LSK) to represent this phenomenon. As ethnic cultures tend to develop alongside different languages, we employ culture-specific datasets (that contain knowledge about cultural and social behavioral norms). We find that language models can perform better when using chain-of-thought reasoning in some languages other than English, sometimes even better in low-resource languages. Paired with previous works showing that semantic similarity does not equate to representational similarity, we hypothesize that culturally specific texts occur more abundantly in corresponding languages, enabling specific knowledge to occur only in specific "expert" languages. Motivated by our initial results, we design a simple methodology called LSKExtractor to benchmark the language-specific knowledge present in a language model and, then, exploit it during inference. We show our results on various models and datasets, showing an average relative improvement of 10% in accuracy. Our research contributes to the open-source development of language models that are inclusive and more aligned with the cultural and linguistic contexts in which they are deployed.
>
---
#### [new 076] Diffusion vs. Autoregressive Language Models: A Text Embedding Perspective
- **分类: cs.CL**

- **简介: 论文对比扩散与自回归语言模型，提出基于扩散模型的文本嵌入方法。针对LLM因单向注意力机制不适应双向文本任务的问题，首次系统研究其在嵌入任务中的应用，实验显示在长文档、推理检索等场景性能提升显著，验证双向注意力的关键作用。**

- **链接: [http://arxiv.org/pdf/2505.15045v1](http://arxiv.org/pdf/2505.15045v1)**

> **作者:** Siyue Zhang; Yilun Zhao; Liyuan Geng; Arman Cohan; Anh Tuan Luu; Chen Zhao
>
> **摘要:** Large language model (LLM)-based embedding models, benefiting from large scale pre-training and post-training, have begun to surpass BERT and T5-based models on general-purpose text embedding tasks such as document retrieval. However, a fundamental limitation of LLM embeddings lies in the unidirectional attention used during autoregressive pre-training, which misaligns with the bidirectional nature of text embedding tasks. To this end, We propose adopting diffusion language models for text embeddings, motivated by their inherent bidirectional architecture and recent success in matching or surpassing LLMs especially on reasoning tasks. We present the first systematic study of the diffusion language embedding model, which outperforms the LLM-based embedding model by 20% on long-document retrieval, 8% on reasoning-intensive retrieval, 2% on instruction-following retrieval, and achieve competitive performance on traditional text embedding benchmarks. Our analysis verifies that bidirectional attention is crucial for encoding global context in long and complex text.
>
---
#### [new 077] Learn to Reason Efficiently with Adaptive Length-based Reward Shaping
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大推理模型(LRMs)生成冗余长推理链导致效率低的问题，提出动态难度感知的LASER-D方法。通过自适应长度奖励调整，在训练中动态优化奖励并根据任务难度惩罚冗余推理，提升性能与效率，AIME2024任务中性能+6.1%且减少63%token使用。**

- **链接: [http://arxiv.org/pdf/2505.15612v1](http://arxiv.org/pdf/2505.15612v1)**

> **作者:** Wei Liu; Ruochen Zhou; Yiyun Deng; Yuzhen Huang; Junteng Liu; Yuntian Deng; Yizhe Zhang; Junxian He
>
> **摘要:** Large Reasoning Models (LRMs) have shown remarkable capabilities in solving complex problems through reinforcement learning (RL), particularly by generating long reasoning traces. However, these extended outputs often exhibit substantial redundancy, which limits the efficiency of LRMs. In this paper, we investigate RL-based approaches to promote reasoning efficiency. Specifically, we first present a unified framework that formulates various efficient reasoning methods through the lens of length-based reward shaping. Building on this perspective, we propose a novel Length-bAsed StEp Reward shaping method (LASER), which employs a step function as the reward, controlled by a target length. LASER surpasses previous methods, achieving a superior Pareto-optimal balance between performance and efficiency. Next, we further extend LASER based on two key intuitions: (1) The reasoning behavior of the model evolves during training, necessitating reward specifications that are also adaptive and dynamic; (2) Rather than uniformly encouraging shorter or longer chains of thought (CoT), we posit that length-based reward shaping should be difficulty-aware i.e., it should penalize lengthy CoTs more for easy queries. This approach is expected to facilitate a combination of fast and slow thinking, leading to a better overall tradeoff. The resulting method is termed LASER-D (Dynamic and Difficulty-aware). Experiments on DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B, and DeepSeek-R1-Distill-Qwen-32B show that our approach significantly enhances both reasoning performance and response length efficiency. For instance, LASER-D and its variant achieve a +6.1 improvement on AIME2024 while reducing token usage by 63%. Further analysis reveals our RL-based compression produces more concise reasoning patterns with less redundant "self-reflections". Resources are at https://github.com/hkust-nlp/Laser.
>
---
#### [new 078] An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于强化学习（RL）训练大型语言模型（LLM）代理的任务，旨在解决如何优化推理与搜索结合的智能体设计问题。研究通过实证分析了奖励设计、LLM特性及搜索引擎选择对训练效果的影响，发现格式奖励有效、LLM规模与初始化显著影响结果、搜索引擎对训练动态和推理鲁棒性至关重要，为实际部署提供指导。**

- **链接: [http://arxiv.org/pdf/2505.15117v1](http://arxiv.org/pdf/2505.15117v1)**

> **作者:** Bowen Jin; Jinsung Yoon; Priyanka Kargupta; Sercan O. Arik; Jiawei Han
>
> **备注:** 22 pages
>
> **摘要:** Reinforcement learning (RL) has demonstrated strong potential in training large language models (LLMs) capable of complex reasoning for real-world problem solving. More recently, RL has been leveraged to create sophisticated LLM-based search agents that adeptly combine reasoning with search engine use. While the use of RL for training search agents is promising, the optimal design of such agents remains not fully understood. In particular, key factors -- such as (1) reward formulation, (2) the choice and characteristics of the underlying LLM, and (3) the role of the search engine in the RL process -- require further investigation. In this work, we conduct comprehensive empirical studies to systematically investigate these and offer actionable insights. We highlight several key findings: format rewards are effective in improving final performance, whereas intermediate retrieval rewards have limited impact; the scale and initialization of the LLM (general-purpose vs. reasoning-specialized) significantly influence RL outcomes; and the choice of search engine plays a critical role in shaping RL training dynamics and the robustness of the trained agent during inference. These establish important guidelines for successfully building and deploying LLM-based search agents in real-world applications. Code is available at https://github.com/PeterGriffinJin/Search-R1.
>
---
#### [new 079] Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment
- **分类: cs.CL**

- **简介: 该论文属于个性化对话系统任务，针对现有方法在冷启动和长期个性化中的静态缺陷，提出RLPA框架：通过强化学习与模拟用户交互，动态推断用户档案，利用双奖励机制优化模型，实现更高效、持久的个性化对话，超越Claude-3.5和GPT-4o等模型。**

- **链接: [http://arxiv.org/pdf/2505.15456v1](http://arxiv.org/pdf/2505.15456v1)**

> **作者:** Weixiang Zhao; Xingyu Sui; Yulin Hu; Jiahe Guo; Haixiao Liu; Biye Li; Yanyan Zhao; Bing Qin; Ting Liu
>
> **备注:** 30 pages, 18 figures, 10 tables
>
> **摘要:** Personalized alignment is essential for enabling large language models (LLMs) to engage effectively in user-centric dialogue. While recent prompt-based and offline optimization methods offer preliminary solutions, they fall short in cold-start scenarios and long-term personalization due to their inherently static and shallow designs. In this work, we introduce the Reinforcement Learning for Personalized Alignment (RLPA) framework, in which an LLM interacts with a simulated user model to iteratively infer and refine user profiles through dialogue. The training process is guided by a dual-level reward structure: the Profile Reward encourages accurate construction of user representations, while the Response Reward incentivizes generation of responses consistent with the inferred profile. We instantiate RLPA by fine-tuning Qwen-2.5-3B-Instruct, resulting in Qwen-RLPA, which achieves state-of-the-art performance in personalized dialogue. Empirical evaluations demonstrate that Qwen-RLPA consistently outperforms prompting and offline fine-tuning baselines, and even surpasses advanced commercial models such as Claude-3.5 and GPT-4o. Further analysis highlights Qwen-RLPA's robustness in reconciling conflicting user preferences, sustaining long-term personalization and delivering more efficient inference compared to recent reasoning-focused LLMs. These results emphasize the potential of dynamic profile inference as a more effective paradigm for building personalized dialogue systems.
>
---
#### [new 080] WebNovelBench: Placing LLM Novelists on the Web Novel Distribution
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出WebNovelBench，评估LLM长篇小说生成能力。针对现有基准规模小、主观性强的问题，构建超4000部中文网文数据集，设计八维度叙事质量评估框架，通过LLM评分与PCA分析，量化对比LLM与人类作品。为叙事生成提供可扩展的评测方法。**

- **链接: [http://arxiv.org/pdf/2505.14818v1](http://arxiv.org/pdf/2505.14818v1)**

> **作者:** Leon Lin; Jun Zheng; Haidong Wang
>
> **摘要:** Robustly evaluating the long-form storytelling capabilities of Large Language Models (LLMs) remains a significant challenge, as existing benchmarks often lack the necessary scale, diversity, or objective measures. To address this, we introduce WebNovelBench, a novel benchmark specifically designed for evaluating long-form novel generation. WebNovelBench leverages a large-scale dataset of over 4,000 Chinese web novels, framing evaluation as a synopsis-to-story generation task. We propose a multi-faceted framework encompassing eight narrative quality dimensions, assessed automatically via an LLM-as-Judge approach. Scores are aggregated using Principal Component Analysis and mapped to a percentile rank against human-authored works. Our experiments demonstrate that WebNovelBench effectively differentiates between human-written masterpieces, popular web novels, and LLM-generated content. We provide a comprehensive analysis of 24 state-of-the-art LLMs, ranking their storytelling abilities and offering insights for future development. This benchmark provides a scalable, replicable, and data-driven methodology for assessing and advancing LLM-driven narrative generation.
>
---
#### [new 081] Multilingual Test-Time Scaling via Initial Thought Transfer
- **分类: cs.CL**

- **简介: 该论文研究多语言测试时缩放（TTS）优化任务，解决其跨语言效果差异及模型推理中语言切换问题。通过系统评估发现语言间推理差异及低资源语言初期不一致性，提出MITT方法：无监督前缀调优，转移高资源语言的推理前缀，提升多语言TTS性能，尤其改善低资源语言。**

- **链接: [http://arxiv.org/pdf/2505.15508v1](http://arxiv.org/pdf/2505.15508v1)**

> **作者:** Prasoon Bajpai; Tanmoy Chakraborty
>
> **备注:** 14 pages, 9 figures, 5 Tables
>
> **摘要:** Test-time scaling has emerged as a widely adopted inference-time strategy for boosting reasoning performance. However, its effectiveness has been studied almost exclusively in English, leaving its behavior in other languages largely unexplored. We present the first systematic study of test-time scaling in multilingual settings, evaluating DeepSeek-R1-Distill-LLama-8B and DeepSeek-R1-Distill-Qwen-7B across both high- and low-resource Latin-script languages. Our findings reveal that the relative gains from test-time scaling vary significantly across languages. Additionally, models frequently switch to English mid-reasoning, even when operating under strictly monolingual prompts. We further show that low-resource languages not only produce initial reasoning thoughts that differ significantly from English but also have lower internal consistency across generations in their early reasoning. Building on our findings, we introduce MITT (Multilingual Initial Thought Transfer), an unsupervised and lightweight reasoning prefix-tuning approach that transfers high-resource reasoning prefixes to enhance test-time scaling across all languages, addressing inconsistencies in multilingual reasoning performance. MITT significantly boosts DeepSeek-R1-Distill-Qwen-7B's reasoning performance, especially for underrepresented languages.
>
---
#### [new 082] LyapLock: Bounded Knowledge Preservation in Sequential Large Language Model Editing
- **分类: cs.CL**

- **简介: 该论文提出LyapLock框架，解决序列大语言模型编辑中的长期知识保存问题。通过约束随机规划与排队、Lyapunov优化，分解子问题实现高效编辑，首次理论保障下支持超万次编辑，提升11.89%效能。**

- **链接: [http://arxiv.org/pdf/2505.15702v1](http://arxiv.org/pdf/2505.15702v1)**

> **作者:** Peng Wang; Biyu Zhou; Xuehai Tang; Jizhong Han; Songlin Hu
>
> **摘要:** Large Language Models often contain factually incorrect or outdated knowledge, giving rise to model editing methods for precise knowledge updates. However, current mainstream locate-then-edit approaches exhibit a progressive performance decline during sequential editing, due to inadequate mechanisms for long-term knowledge preservation. To tackle this, we model the sequential editing as a constrained stochastic programming. Given the challenges posed by the cumulative preservation error constraint and the gradually revealed editing tasks, \textbf{LyapLock} is proposed. It integrates queuing theory and Lyapunov optimization to decompose the long-term constrained programming into tractable stepwise subproblems for efficient solving. This is the first model editing framework with rigorous theoretical guarantees, achieving asymptotic optimal editing performance while meeting the constraints of long-term knowledge preservation. Experimental results show that our framework scales sequential editing capacity to over 10,000 edits while stabilizing general capabilities and boosting average editing efficacy by 11.89\% over SOTA baselines. Furthermore, it can be leveraged to enhance the performance of baseline methods. Our code is released on https://github.com/caskcsg/LyapLock.
>
---
#### [new 083] A Survey on Multilingual Mental Disorders Detection from Social Media Data
- **分类: cs.CL**

- **简介: 该综述针对现有心理健康检测研究多依赖英文数据、忽视多语言及文化差异的问题，首次系统探讨多语言社交媒体心理健康检测，分析文化对语言表达和NLP工具性能的影响，并汇总多语种数据集，为开发普适筛查工具提供指导，助力全球心理健康服务。**

- **链接: [http://arxiv.org/pdf/2505.15556v1](http://arxiv.org/pdf/2505.15556v1)**

> **作者:** Ana-Maria Bucur; Marcos Zampieri; Tharindu Ranasinghe; Fabio Crestani
>
> **摘要:** The increasing prevalence of mental health disorders globally highlights the urgent need for effective digital screening methods that can be used in multilingual contexts. Most existing studies, however, focus on English data, overlooking critical mental health signals that may be present in non-English texts. To address this important gap, we present the first survey on the detection of mental health disorders using multilingual social media data. We investigate the cultural nuances that influence online language patterns and self-disclosure behaviors, and how these factors can impact the performance of NLP tools. Additionally, we provide a comprehensive list of multilingual data collections that can be used for developing NLP models for mental health screening. Our findings can inform the design of effective multilingual mental health screening tools that can meet the needs of diverse populations, ultimately improving mental health outcomes on a global scale.
>
---
#### [new 084] Multi-Hop Question Generation via Dual-Perspective Keyword Guidance
- **分类: cs.CL**

- **简介: 该论文属于多跳问题生成（MQG）任务，旨在解决现有方法未能有效利用关键词区分问题意图与文档内容的问题。提出双视角关键词（问题关键词和文档关键词）及DPKG框架，通过扩展的Transformer编码器和两个解码器生成关键词与问题，提升信息整合效果。**

- **链接: [http://arxiv.org/pdf/2505.15299v1](http://arxiv.org/pdf/2505.15299v1)**

> **作者:** Maodong Li; Longyin Zhang; Fang Kong
>
> **备注:** 17 pages, 5 figures, accepted to the Findings of ACL 2025
>
> **摘要:** Multi-hop question generation (MQG) aims to generate questions that require synthesizing multiple information snippets from documents to derive target answers. The primary challenge lies in effectively pinpointing crucial information snippets related to question-answer (QA) pairs, typically relying on keywords. However, existing works fail to fully utilize the guiding potential of keywords and neglect to differentiate the distinct roles of question-specific and document-specific keywords. To address this, we define dual-perspective keywords (i.e., question and document keywords) and propose a Dual-Perspective Keyword-Guided (DPKG) framework, which seamlessly integrates keywords into the multi-hop question generation process. We argue that question keywords capture the questioner's intent, whereas document keywords reflect the content related to the QA pair. Functionally, question and document keywords work together to pinpoint essential information snippets in the document, with question keywords required to appear in the generated question. The DPKG framework consists of an expanded transformer encoder and two answer-aware transformer decoders for keyword and question generation, respectively. Extensive experiments demonstrate the effectiveness of our work, showcasing its promising performance and underscoring its significant value in the MQG task.
>
---
#### [new 085] Scaling Reasoning, Losing Control: Evaluating Instruction Following in Large Reasoning Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于评估大语言模型（LLMs）在数学推理任务中指令遵循能力的任务。旨在解决模型推理能力提升与指令控制力下降的矛盾。提出MathIF基准，发现更强推理模型更难遵守用户指令，蒸馏长思维链或强化学习优化推理时会降低指令遵循度，简单干预可部分恢复服从但损害推理性能，揭示当前训练范式根本矛盾。**

- **链接: [http://arxiv.org/pdf/2505.14810v1](http://arxiv.org/pdf/2505.14810v1)**

> **作者:** Tingchen Fu; Jiawei Gu; Yafu Li; Xiaoye Qu; Yu Cheng
>
> **摘要:** Instruction-following is essential for aligning large language models (LLMs) with user intent. While recent reasoning-oriented models exhibit impressive performance on complex mathematical problems, their ability to adhere to natural language instructions remains underexplored. In this work, we introduce MathIF, a dedicated benchmark for evaluating instruction-following in mathematical reasoning tasks. Our empirical analysis reveals a consistent tension between scaling up reasoning capacity and maintaining controllability, as models that reason more effectively often struggle to comply with user directives. We find that models tuned on distilled long chains-of-thought or trained with reasoning-oriented reinforcement learning often degrade in instruction adherence, especially when generation length increases. Furthermore, we show that even simple interventions can partially recover obedience, though at the cost of reasoning performance. These findings highlight a fundamental tension in current LLM training paradigms and motivate the need for more instruction-aware reasoning models. We release the code and data at https://github.com/TingchenFu/MathIF.
>
---
#### [new 086] SEPS: A Separability Measure for Robust Unlearning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于机器遗忘任务，针对现有方法无法有效处理混合查询场景的问题，提出SEPS评估框架衡量模型在单个提示中同时遗忘和保留信息的能力，并设计混合提示（MP）遗忘策略，整合两类查询的训练目标，解决现有方法过度遗忘或场景适应差的缺陷。**

- **链接: [http://arxiv.org/pdf/2505.14832v1](http://arxiv.org/pdf/2505.14832v1)**

> **作者:** Wonje Jeung; Sangyeon Yoon; Albert No
>
> **备注:** 32 pages
>
> **摘要:** Machine unlearning aims to selectively remove targeted knowledge from Large Language Models (LLMs), ensuring they forget specified content while retaining essential information. Existing unlearning metrics assess whether a model correctly answers retain queries and rejects forget queries, but they fail to capture real-world scenarios where forget queries rarely appear in isolation. In fact, forget and retain queries often coexist within the same prompt, making mixed-query evaluation crucial. We introduce SEPS, an evaluation framework that explicitly measures a model's ability to both forget and retain information within a single prompt. Through extensive experiments across three benchmarks, we identify two key failure modes in existing unlearning methods: (1) untargeted unlearning indiscriminately erases both forget and retain content once a forget query appears, and (2) targeted unlearning overfits to single-query scenarios, leading to catastrophic failures when handling multiple queries. To address these issues, we propose Mixed Prompt (MP) unlearning, a strategy that integrates both forget and retain queries into a unified training objective. Our approach significantly improves unlearning effectiveness, demonstrating robustness even in complex settings with up to eight mixed forget and retain queries in a single prompt.
>
---
#### [new 087] MAATS: A Multi-Agent Automated Translation System Based on MQM Evaluation
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 论文提出MAATS，一种基于MQM框架的多智能体自动翻译系统。针对传统单模型难以细致处理多维度质量问题，设计多个专用代理分别优化准确性、流畅度等MQM类别，再通过合成代理迭代改进。实验显示其在语义准确性和跨语言翻译中优于基线模型，提升翻译深度和上下文保真度。**

- **链接: [http://arxiv.org/pdf/2505.14848v1](http://arxiv.org/pdf/2505.14848v1)**

> **作者:** Xi Wang; Jiaqian Hu; Safinah Ali
>
> **摘要:** We present MAATS, a Multi Agent Automated Translation System that leverages the Multidimensional Quality Metrics (MQM) framework as a fine-grained signal for error detection and refinement. MAATS employs multiple specialized AI agents, each focused on a distinct MQM category (e.g., Accuracy, Fluency, Style, Terminology), followed by a synthesis agent that integrates the annotations to iteratively refine translations. This design contrasts with conventional single-agent methods that rely on self-correction. Evaluated across diverse language pairs and Large Language Models (LLMs), MAATS outperforms zero-shot and single-agent baselines with statistically significant gains in both automatic metrics and human assessments. It excels particularly in semantic accuracy, locale adaptation, and linguistically distant language pairs. Qualitative analysis highlights its strengths in multi-layered error diagnosis, omission detection across perspectives, and context-aware refinement. By aligning modular agent roles with interpretable MQM dimensions, MAATS narrows the gap between black-box LLMs and human translation workflows, shifting focus from surface fluency to deeper semantic and contextual fidelity.
>
---
#### [new 088] Semantic-based Unsupervised Framing Analysis (SUFA): A Novel Approach for Computational Framing Analysis
- **分类: cs.CL**

- **简介: 该论文提出SUFA方法，属于计算框架分析任务，旨在无监督识别新闻中实体中心强调框架。基于语义关系与依存句法分析，通过枪支暴力数据集验证其有效性，结合定性与计算研究，探讨跨领域应用潜力。（99字）**

- **链接: [http://arxiv.org/pdf/2505.15563v1](http://arxiv.org/pdf/2505.15563v1)**

> **作者:** Mohammad Ali; Naeemul Hassan
>
> **备注:** Association for Education in Journalism and Mass Communication (AEJMC) Conference, August 07--10, 2023, Washington, DC, USA
>
> **摘要:** This research presents a novel approach to computational framing analysis, called Semantic Relations-based Unsupervised Framing Analysis (SUFA). SUFA leverages semantic relations and dependency parsing algorithms to identify and assess entity-centric emphasis frames in news media reports. This innovative method is derived from two studies -- qualitative and computational -- using a dataset related to gun violence, demonstrating its potential for analyzing entity-centric emphasis frames. This article discusses SUFA's strengths, limitations, and application procedures. Overall, the SUFA approach offers a significant methodological advancement in computational framing analysis, with its broad applicability across both the social sciences and computational domains.
>
---
#### [new 089] Chinese Toxic Language Mitigation via Sentiment Polarity Consistent Rewrites
- **分类: cs.CL**

- **简介: 该论文属于中文有害语言净化任务，旨在重写有毒内容时保留原意及情感极性。针对现有模型重写后情感失真问题，构建首个含1556个 triplet 的中文净化数据集 ToxiRewriteCN，覆盖多场景，并评估17种模型在安全性与情感保真间的平衡挑战。**

- **链接: [http://arxiv.org/pdf/2505.15297v1](http://arxiv.org/pdf/2505.15297v1)**

> **作者:** Xintong Wang; Yixiao Liu; Jingheng Pan; Liang Ding; Longyue Wang; Chris Biemann
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Detoxifying offensive language while preserving the speaker's original intent is a challenging yet critical goal for improving the quality of online interactions. Although large language models (LLMs) show promise in rewriting toxic content, they often default to overly polite rewrites, distorting the emotional tone and communicative intent. This problem is especially acute in Chinese, where toxicity often arises implicitly through emojis, homophones, or discourse context. We present ToxiRewriteCN, the first Chinese detoxification dataset explicitly designed to preserve sentiment polarity. The dataset comprises 1,556 carefully annotated triplets, each containing a toxic sentence, a sentiment-aligned non-toxic rewrite, and labeled toxic spans. It covers five real-world scenarios: standard expressions, emoji-induced and homophonic toxicity, as well as single-turn and multi-turn dialogues. We evaluate 17 LLMs, including commercial and open-source models with variant architectures, across four dimensions: detoxification accuracy, fluency, content preservation, and sentiment polarity. Results show that while commercial and MoE models perform best overall, all models struggle to balance safety with emotional fidelity in more subtle or context-heavy settings such as emoji, homophone, and dialogue-based inputs. We release ToxiRewriteCN to support future research on controllable, sentiment-aware detoxification for Chinese.
>
---
#### [new 090] Learning to Reason via Mixture-of-Thought for Logical Reasoning
- **分类: cs.CL**

- **简介: 该论文聚焦逻辑推理任务，针对现有LLM仅采用单一模态（如自然语言）训练导致模态间协同不足的问题，提出Mixture-of-Thought框架，融合自然语言、代码及新符号模态（truth-table），通过自演进训练与多模态推理协同，显著提升推理准确率（+11.7pp），尤其在复杂问题中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.15817v1](http://arxiv.org/pdf/2505.15817v1)**

> **作者:** Tong Zheng; Lichang Chen; Simeng Han; R. Thomas McCoy; Heng Huang
>
> **备注:** 38 pages
>
> **摘要:** Human beings naturally utilize multiple reasoning modalities to learn and solve logical problems, i.e., different representational formats such as natural language, code, and symbolic logic. In contrast, most existing LLM-based approaches operate with a single reasoning modality during training, typically natural language. Although some methods explored modality selection or augmentation at inference time, the training process remains modality-blind, limiting synergy among modalities. To fill in this gap, we propose Mixture-of-Thought (MoT), a framework that enables LLMs to reason across three complementary modalities: natural language, code, and a newly introduced symbolic modality, truth-table, which systematically enumerates logical cases and partially mitigates key failure modes in natural language reasoning. MoT adopts a two-phase design: (1) self-evolving MoT training, which jointly learns from filtered, self-generated rationales across modalities; and (2) MoT inference, which fully leverages the synergy of three modalities to produce better predictions. Experiments on logical reasoning benchmarks including FOLIO and ProofWriter demonstrate that our MoT framework consistently and significantly outperforms strong LLM baselines with single-modality chain-of-thought approaches, achieving up to +11.7pp average accuracy gain. Further analyses show that our MoT framework benefits both training and inference stages; that it is particularly effective on harder logical reasoning problems; and that different modalities contribute complementary strengths, with truth-table reasoning helping to overcome key bottlenecks in natural language inference.
>
---
#### [new 091] DayDreamer at CQs-Gen 2025: Generating Critical Questions through Argument Scheme Completion
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于批判性问题生成（CQs-Gen）任务，旨在通过结构化论证方案与LLM生成相关且多样的批判性问题，促进读者对论证文本的批判性思考并检测其漏洞。系统利用Walton的论证模板引导LLM生成结构化论点，继而生成问题，并通过排序筛选最优结果，结合理论与模型推理提升生成质量。**

- **链接: [http://arxiv.org/pdf/2505.15554v1](http://arxiv.org/pdf/2505.15554v1)**

> **作者:** Wendi Zhou; Ameer Saadat-Yazdi; Nadin Kökciyan
>
> **备注:** ArgMining 2025 CQs-Gen shared task
>
> **摘要:** Critical questions are essential resources to provoke critical thinking when encountering an argumentative text. We present our system for the Critical Questions Generation (CQs-Gen) Shared Task at ArgMining 2025. Our approach leverages large language models (LLMs) with chain-of-thought prompting to generate critical questions guided by Walton's argumentation schemes. For each input intervention, we conversationally prompt LLMs to instantiate the corresponding argument scheme template to first obtain structured arguments, and then generate relevant critical questions. Following this, we rank all the available critical questions by prompting LLMs to select the top 3 most helpful questions based on the original intervention text. This combination of structured argumentation theory and step-by-step reasoning enables the generation of contextually relevant and diverse critical questions. Our pipeline achieves competitive performance in the final test set, showing its potential to foster critical thinking given argumentative text and detect missing or uninformed claims. Code available at \href{https://git.ecdf.ed.ac.uk/s2236454/DayDreamer-CQs-Gen}{DayDreamer}.
>
---
#### [new 092] The Representational Alignment between Humans and Language Models is implicitly driven by a Concreteness Effect
- **分类: cs.CL; I.2.7; J.4**

- **简介: 该论文研究人类与语言模型语义表征的对齐机制，探究具体性效应是否驱动这种对齐。通过行为实验获取人类对抽象/具体名词的语义距离和具体性评分，结合表征相似性分析及消融实验，发现人类与模型的对齐主要由具体性维度驱动，而非其他语言学特征。**

- **链接: [http://arxiv.org/pdf/2505.15682v1](http://arxiv.org/pdf/2505.15682v1)**

> **作者:** Cosimo Iaia; Bhavin Choksi; Emily Wiebers; Gemma Roig; Christian J. Fiebach
>
> **备注:** 13 pages, 4 Figures, 1 Table
>
> **摘要:** The nouns of our language refer to either concrete entities (like a table) or abstract concepts (like justice or love), and cognitive psychology has established that concreteness influences how words are processed. Accordingly, understanding how concreteness is represented in our mind and brain is a central question in psychology, neuroscience, and computational linguistics. While the advent of powerful language models has allowed for quantitative inquiries into the nature of semantic representations, it remains largely underexplored how they represent concreteness. Here, we used behavioral judgments to estimate semantic distances implicitly used by humans, for a set of carefully selected abstract and concrete nouns. Using Representational Similarity Analysis, we find that the implicit representational space of participants and the semantic representations of language models are significantly aligned. We also find that both representational spaces are implicitly aligned to an explicit representation of concreteness, which was obtained from our participants using an additional concreteness rating task. Importantly, using ablation experiments, we demonstrate that the human-to-model alignment is substantially driven by concreteness, but not by other important word characteristics established in psycholinguistics. These results indicate that humans and language models converge on the concreteness dimension, but not on other dimensions.
>
---
#### [new 093] Social Bias in Popular Question-Answering Benchmarks
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文研究问答与阅读理解基准中的社会偏见问题，揭示其因创建者多样性不足导致性别、宗教、地域等偏见。通过分析30篇论文及20个数据集，发现现有基准在偏见防控措施和透明度不足，呼吁改进创建流程以促进公平AI发展。**

- **链接: [http://arxiv.org/pdf/2505.15553v1](http://arxiv.org/pdf/2505.15553v1)**

> **作者:** Angelie Kraft; Judith Simon; Sonja Schimmler
>
> **摘要:** Question-answering (QA) and reading comprehension (RC) benchmarks are essential for assessing the capabilities of large language models (LLMs) in retrieving and reproducing knowledge. However, we demonstrate that popular QA and RC benchmarks are biased and do not cover questions about different demographics or regions in a representative way, potentially due to a lack of diversity of those involved in their creation. We perform a qualitative content analysis of 30 benchmark papers and a quantitative analysis of 20 respective benchmark datasets to learn (1) who is involved in the benchmark creation, (2) how social bias is addressed or prevented, and (3) whether the demographics of the creators and annotators correspond to particular biases in the content. Most analyzed benchmark papers provided insufficient information regarding the stakeholders involved in benchmark creation, particularly the annotators. Notably, just one of the benchmark papers explicitly reported measures taken to address social representation issues. Moreover, the data analysis revealed gender, religion, and geographic biases across a wide range of encyclopedic, commonsense, and scholarly benchmarks. More transparent and bias-aware QA and RC benchmark creation practices are needed to facilitate better scrutiny and incentivize the development of fairer LLMs.
>
---
#### [new 094] Exploring In-Image Machine Translation with Real-World Background
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于跨模态机器翻译任务，针对现有模型在复杂真实背景图像翻译效果差的问题，构建了含真实背景字幕的IIMT数据集，并提出DebackX模型：通过分离图像背景与文字、直接翻译文字图像再融合，提升翻译质量和视觉效果。**

- **链接: [http://arxiv.org/pdf/2505.15282v1](http://arxiv.org/pdf/2505.15282v1)**

> **作者:** Yanzhi Tian; Zeming Liu; Zhengyang Liu; Yuhang Guo
>
> **备注:** Accepted to ACL 2025 Findings. Code available at https://github.com/BITHLP/DebackX
>
> **摘要:** In-Image Machine Translation (IIMT) aims to translate texts within images from one language to another. Previous research on IIMT was primarily conducted on simplified scenarios such as images of one-line text with black font in white backgrounds, which is far from reality and impractical for applications in the real world. To make IIMT research practically valuable, it is essential to consider a complex scenario where the text backgrounds are derived from real-world images. To facilitate research of complex scenario IIMT, we design an IIMT dataset that includes subtitle text with real-world background. However previous IIMT models perform inadequately in complex scenarios. To address the issue, we propose the DebackX model, which separates the background and text-image from the source image, performs translation on text-image directly, and fuses the translated text-image with the background, to generate the target image. Experimental results show that our model achieves improvements in both translation quality and visual effect.
>
---
#### [new 095] KaFT: Knowledge-aware Fine-tuning for Boosting LLMs' Domain-specific Question-Answering Performance
- **分类: cs.CL**

- **简介: 该论文属于提升大语言模型（LLMs）领域特定问答性能的任务。针对传统监督微调（SFT）因模型内部知识与训练数据知识冲突导致效果不佳的问题，提出KaFT方法：设计查询多样化策略检测冲突，分析冲突样本影响后，通过按冲突程度分配训练权重动态利用冲突数据。实验显示其有效提升模型性能并减少幻觉。**

- **链接: [http://arxiv.org/pdf/2505.15480v1](http://arxiv.org/pdf/2505.15480v1)**

> **作者:** Qihuang Zhong; Liang Ding; Xiantao Cai; Juhua Liu; Bo Du; Dacheng Tao
>
> **备注:** Accepted to ACL2025 Findings
>
> **摘要:** Supervised fine-tuning (SFT) is a common approach to improve the domain-specific question-answering (QA) performance of large language models (LLMs). However, recent literature reveals that due to the conflicts between LLMs' internal knowledge and the context knowledge of training data, vanilla SFT using the full QA training set is usually suboptimal. In this paper, we first design a query diversification strategy for robust conflict detection and then conduct a series of experiments to analyze the impact of knowledge conflict. We find that 1) training samples with varied conflicts contribute differently, where SFT on the data with large conflicts leads to catastrophic performance drops; 2) compared to directly filtering out the conflict data, appropriately applying the conflict data would be more beneficial. Motivated by this, we propose a simple-yet-effective Knowledge-aware Fine-tuning (namely KaFT) approach to effectively boost LLMs' performance. The core of KaFT is to adapt the training weight by assigning different rewards for different training samples according to conflict level. Extensive experiments show that KaFT brings consistent and significant improvements across four LLMs. More analyses prove that KaFT effectively improves the model generalization and alleviates the hallucination.
>
---
#### [new 096] RePPL: Recalibrating Perplexity by Uncertainty in Semantic Propagation and Language Generation for Explainable QA Hallucination Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于可解释的QA幻觉检测任务，解决现有方法无法定位幻觉来源的问题。提出RePPL方法，通过量化语义传播（注意力机制）和语言生成中的不确定性，为每个token分配可解释的置信分，以困惑度形式聚合总分。实验显示其在QA数据集上检测效果最佳（AUC 0.833），并能定位幻觉触发点。**

- **链接: [http://arxiv.org/pdf/2505.15386v1](http://arxiv.org/pdf/2505.15386v1)**

> **作者:** Yiming Huang; Junyan Zhang; Zihao Wang; Biquan Bie; Xuming Hu; Yi R.; Fung; Xinlei He
>
> **摘要:** Large Language Models (LLMs) have become powerful, but hallucinations remain a vital obstacle to their trustworthy use. While previous works improved the capability of hallucination detection by measuring uncertainty, they all lack the ability to explain the provenance behind why hallucinations occur, i.e., which part of the inputs tends to trigger hallucinations. Recent works on the prompt attack indicate that uncertainty exists in semantic propagation, where attention mechanisms gradually fuse local token information into high-level semantics across layers. Meanwhile, uncertainty also emerges in language generation, due to its probability-based selection of high-level semantics for sampled generations. Based on that, we propose RePPL to recalibrate uncertainty measurement by these two aspects, which dispatches explainable uncertainty scores to each token and aggregates in Perplexity-style Log-Average form as total score. Experiments show that our method achieves the best comprehensive detection performance across various QA datasets on advanced models (average AUC of 0.833), and our method is capable of producing token-level uncertainty scores as explanations for the hallucination. Leveraging these scores, we preliminarily find the chaotic pattern of hallucination and showcase its promising usage.
>
---
#### [new 097] Scaling Laws for State Dynamics in Large Language Models
- **分类: cs.CL; cs.AI; I.2.7; I.2.1; I.2.4; I.5.4**

- **简介: 论文评估LLMs在状态跟踪任务中的性能，发现其状态预测准确率随状态空间扩大和转换稀疏性显著下降；通过激活修补技术定位关键注意力头，揭示LLMs依赖分布式token交互而非显式符号计算进行状态跟踪。**

- **链接: [http://arxiv.org/pdf/2505.14892v1](http://arxiv.org/pdf/2505.14892v1)**

> **作者:** Jacob X Li; Shreyas S Raman; Jessica Wan; Fahad Samman; Jazlyn Lin
>
> **备注:** 16 pages; 23 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly used in tasks requiring internal state tracking, yet their ability to model state transition dynamics remains poorly understood. We evaluate how well LLMs capture deterministic state dynamics across 3 domains: Box Tracking, Abstract DFA Sequences, and Complex Text Games, each formalizable as a finite-state system. Across tasks, we find that next-state prediction accuracy degrades with increasing state-space size and sparse transitions. GPT-2 XL reaches about 70% accuracy in low-complexity settings but drops below 30% when the number of boxes or states exceeds 5 or 10, respectively. In DFA tasks, Pythia-1B fails to exceed 50% accuracy when the number of states is > 10 and transitions are < 30. Through activation patching, we identify attention heads responsible for propagating state information: GPT-2 XL Layer 22 Head 20, and Pythia-1B Heads at Layers 10, 11, 12, and 14. While these heads successfully move relevant state features, action information is not reliably routed to the final token, indicating weak joint state-action reasoning. Our results suggest that state tracking in LLMs emerges from distributed interactions of next-token heads rather than explicit symbolic computation.
>
---
#### [new 098] Multilingual Prompting for Improving LLM Generation Diversity
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出多语言提示法，通过融入多文化语言线索生成多样化提示，结合响应以提升LLM生成多样性。旨在解决LLM文化代表性不足问题，实验显示其优于现有技术，并分析了语言资源和模型规模的影响。**

- **链接: [http://arxiv.org/pdf/2505.15229v1](http://arxiv.org/pdf/2505.15229v1)**

> **作者:** Qihan Wang; Shidong Pan; Tal Linzen; Emily Black
>
> **摘要:** Large Language Models (LLMs) are known to lack cultural representation and overall diversity in their generations, from expressing opinions to answering factual questions. To mitigate this problem, we propose multilingual prompting: a prompting method which generates several variations of a base prompt with added cultural and linguistic cues from several cultures, generates responses, and then combines the results. Building on evidence that LLMs have language-specific knowledge, multilingual prompting seeks to increase diversity by activating a broader range of cultural knowledge embedded in model training data. Through experiments across multiple models (GPT-4o, GPT-4o-mini, LLaMA 70B, and LLaMA 8B), we show that multilingual prompting consistently outperforms existing diversity-enhancing techniques such as high-temperature sampling, step-by-step recall, and personas prompting. Further analyses show that the benefits of multilingual prompting vary with language resource level and model size, and that aligning the prompting language with the cultural cues reduces hallucination about culturally-specific information.
>
---
#### [new 099] Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CoPA方法，通过优化LLM指令并引入机器-人类语言对比机制，减少生成文本的机器特征，解决现有对抗攻击依赖大量数据、效果差的问题，实现高效绕过文本检测器。**

- **链接: [http://arxiv.org/pdf/2505.15337v1](http://arxiv.org/pdf/2505.15337v1)**

> **作者:** Hao Fang; Jiawei Kong; Tianqu Zhuang; Yixiang Qiu; Kuofeng Gao; Bin Chen; Shu-Tao Xia; Yaowei Wang; Min Zhang
>
> **摘要:** The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.
>
---
#### [new 100] Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大模型推理任务，旨在解决离散语言token限制LLM推理潜力的问题。提出无训练的Soft Thinking方法，通过连续概念空间生成抽象概念token（混合概率加权的token嵌入），实现多路径推理，提升推理准确率（+2.48%）并减少22.4%token消耗。**

- **链接: [http://arxiv.org/pdf/2505.15778v1](http://arxiv.org/pdf/2505.15778v1)**

> **作者:** Zhen Zhang; Xuehai He; Weixiang Yan; Ao Shen; Chenyang Zhao; Shuohang Wang; Yelong Shen; Xin Eric Wang
>
> **摘要:** Human cognition typically involves thinking through abstract, fluid concepts rather than strictly using discrete linguistic tokens. Current reasoning models, however, are constrained to reasoning within the boundaries of human language, processing discrete token embeddings that represent fixed points in the semantic space. This discrete constraint restricts the expressive power and upper potential of such reasoning models, often causing incomplete exploration of reasoning paths, as standard Chain-of-Thought (CoT) methods rely on sampling one token per step. In this work, we introduce Soft Thinking, a training-free method that emulates human-like "soft" reasoning by generating soft, abstract concept tokens in a continuous concept space. These concept tokens are created by the probability-weighted mixture of token embeddings, which form the continuous concept space, enabling smooth transitions and richer representations that transcend traditional discrete boundaries. In essence, each generated concept token encapsulates multiple meanings from related discrete tokens, implicitly exploring various reasoning paths to converge effectively toward the correct answer. Empirical evaluations on diverse mathematical and coding benchmarks consistently demonstrate the effectiveness and efficiency of Soft Thinking, improving pass@1 accuracy by up to 2.48 points while simultaneously reducing token usage by up to 22.4% compared to standard CoT. Qualitative analysis further reveals that Soft Thinking outputs remain highly interpretable and readable, highlighting the potential of Soft Thinking to break the inherent bottleneck of discrete language-based reasoning. Code is available at https://github.com/eric-ai-lab/Soft-Thinking.
>
---
#### [new 101] Joint Flashback Adaptation for Forgetting-Resistant Instruction Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型增量学习中的灾难性遗忘问题，提出联合回溯适应方法。通过引入旧任务的少量提示（flashbacks）并插值潜在任务，约束模型输出偏差，实现新旧任务协同学习，减少遗忘并提升新任务泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.15467v1](http://arxiv.org/pdf/2505.15467v1)**

> **作者:** Yukun Zhao; Lingyong Yan; Zhenyang Li; Shuaiqiang Wang; Zhumin Chen; Zhaochun Ren; Dawei Yin
>
> **摘要:** Large language models have achieved remarkable success in various tasks. However, it is challenging for them to learn new tasks incrementally due to catastrophic forgetting. Existing approaches rely on experience replay, optimization constraints, or task differentiation, which encounter strict limitations in real-world scenarios. To address these issues, we propose Joint Flashback Adaptation. We first introduce flashbacks -- a limited number of prompts from old tasks -- when adapting to new tasks and constrain the deviations of the model outputs compared to the original one. We then interpolate latent tasks between flashbacks and new tasks to enable jointly learning relevant latent tasks, new tasks, and flashbacks, alleviating data sparsity in flashbacks and facilitating knowledge sharing for smooth adaptation. Our method requires only a limited number of flashbacks without access to the replay data and is task-agnostic. We conduct extensive experiments on state-of-the-art large language models across 1000+ instruction-following tasks, arithmetic reasoning tasks, and general reasoning tasks. The results demonstrate the superior performance of our method in improving generalization on new tasks and reducing forgetting in old tasks.
>
---
#### [new 102] DECASTE: Unveiling Caste Stereotypes in Large Language Models through Multi-Dimensional Bias Analysis
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出DECASTE框架，检测大型语言模型中的种姓偏见。针对LLMs强化印度边缘种姓（如达利特、首陀）偏见的问题，通过社会文化、经济、教育、政治四维度分析，揭示模型存在系统性偏见，强调需完善评估方法以降低部署风险。**

- **链接: [http://arxiv.org/pdf/2505.14971v1](http://arxiv.org/pdf/2505.14971v1)**

> **作者:** Prashanth Vijayaraghavan; Soroush Vosoughi; Lamogha Chizor; Raya Horesh; Rogerio Abreu de Paula; Ehsan Degan; Vandana Mukherjee
>
> **备注:** 7 (content pages) + 2 (reference pages) + 5 (Appendix pages), 5 figures, 6 Tables, IJCAI 2025
>
> **摘要:** Recent advancements in large language models (LLMs) have revolutionized natural language processing (NLP) and expanded their applications across diverse domains. However, despite their impressive capabilities, LLMs have been shown to reflect and perpetuate harmful societal biases, including those based on ethnicity, gender, and religion. A critical and underexplored issue is the reinforcement of caste-based biases, particularly towards India's marginalized caste groups such as Dalits and Shudras. In this paper, we address this gap by proposing DECASTE, a novel, multi-dimensional framework designed to detect and assess both implicit and explicit caste biases in LLMs. Our approach evaluates caste fairness across four dimensions: socio-cultural, economic, educational, and political, using a range of customized prompting strategies. By benchmarking several state-of-the-art LLMs, we reveal that these models systematically reinforce caste biases, with significant disparities observed in the treatment of oppressed versus dominant caste groups. For example, bias scores are notably elevated when comparing Dalits and Shudras with dominant caste groups, reflecting societal prejudices that persist in model outputs. These results expose the subtle yet pervasive caste biases in LLMs and emphasize the need for more comprehensive and inclusive bias evaluation methodologies that assess the potential risks of deploying such models in real-world contexts.
>
---
#### [new 103] Too Long, Didn't Model: Decomposing LLM Long-Context Understanding With Novels
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大语言模型（LLMs）长上下文理解评估任务，旨在解决现有方法难以有效评测LLMs处理复杂长文本（如小说）中长程语义依赖的问题。团队基于小说结构设计TLDM基准，测试模型在情节摘要、故事设定及叙事时间推移等任务的表现，发现当前7种前沿LLMs超过64k token后理解能力显著下降，并公开数据与代码促进模型优化。**

- **链接: [http://arxiv.org/pdf/2505.14925v1](http://arxiv.org/pdf/2505.14925v1)**

> **作者:** Sil Hamilton; Rebecca M. M. Hicke; Matthew Wilkens; David Mimno
>
> **摘要:** Although the context length of large language models (LLMs) has increased to millions of tokens, evaluating their effectiveness beyond needle-in-a-haystack approaches has proven difficult. We argue that novels provide a case study of subtle, complicated structure and long-range semantic dependencies often over 128k tokens in length. Inspired by work on computational novel analysis, we release the Too Long, Didn't Model (TLDM) benchmark, which tests a model's ability to report plot summary, storyworld configuration, and elapsed narrative time. We find that none of seven tested frontier LLMs retain stable understanding beyond 64k tokens. Our results suggest language model developers must look beyond "lost in the middle" benchmarks when evaluating model performance in complex long-context scenarios. To aid in further development we release the TLDM benchmark together with reference code and data.
>
---
#### [new 104] Gated Integration of Low-Rank Adaptation for Continual Learning of Language Models
- **分类: cs.CL**

- **简介: 该论文针对语言模型持续学习任务，解决现有LoRA方法因新旧任务参数等同贡献导致遗忘的问题。提出GainLoRA，通过门控模块动态调节分支权重，抑制新任务参数对旧任务的干扰，实验表明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.15424v1](http://arxiv.org/pdf/2505.15424v1)**

> **作者:** Yan-Shuo Liang; Wu-Jun Li
>
> **摘要:** Continual learning (CL), which requires the model to learn multiple tasks sequentially, is crucial for language models (LMs). Recently, low-rank adaptation (LoRA), one of the most representative parameter-efficient fine-tuning (PEFT) methods, has gained increasing attention in CL of LMs. However, most existing CL methods based on LoRA typically expand a new LoRA branch to learn each new task and force the new and old LoRA branches to contribute equally to old tasks, potentially leading to forgetting. In this work, we propose a new method, called gated integration of low-rank adaptation (GainLoRA), for CL of LMs. GainLoRA expands a new LoRA branch for each new task and introduces gating modules to integrate the new and old LoRA branches. Furthermore, GainLoRA leverages the new gating module to minimize the contribution from the new LoRA branch to old tasks, effectively mitigating forgetting and improving the model's overall performance. Experimental results on CL benchmarks demonstrate that GainLoRA outperforms existing state-of-the-art methods.
>
---
#### [new 105] Transfer of Structural Knowledge from Synthetic Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究合成语言到英语的迁移学习任务，旨在提升自然语言理解模型的结构知识迁移效果。提出新合成语言及Tiny-Cloze基准，分析模型嵌入结构并验证新方法在多项任务中的优越性。**

- **链接: [http://arxiv.org/pdf/2505.15769v1](http://arxiv.org/pdf/2505.15769v1)**

> **作者:** Mikhail Budnikov; Ivan Yamshchikov
>
> **备注:** 10 pages, 3 figures and 3 tables to be published in ACL 2025 Workshop XLLM
>
> **摘要:** This work explores transfer learning from several synthetic languages to English. We investigate the structure of the embeddings in the fine-tuned models, the information they contain, and the capabilities of the fine-tuned models on simple linguistic tasks. We also introduce a new synthetic language that leads to better transfer to English than the languages used in previous research. Finally, we introduce Tiny-Cloze Benchmark - a new synthetic benchmark for natural language understanding that is more informative for less powerful models. We use Tiny-Cloze Benchmark to evaluate fine-tuned models in several domains demonstrating that fine-tuning on a new synthetic language allows for better performance on a variety of tasks.
>
---
#### [new 106] Addressing the Challenges of Planning Language Generation
- **分类: cs.CL**

- **简介: 该论文聚焦规划语言生成任务，解决开源大模型生成PDDL效果不佳的问题。作者测试了8种基于开源模型（500亿参数）的生成方案，发现直观方法（如语法约束）性能下降，而利用求解器反馈的推理扩展方法使成功率翻倍。**

- **链接: [http://arxiv.org/pdf/2505.14763v1](http://arxiv.org/pdf/2505.14763v1)**

> **作者:** Prabhu Prakash Kagitha; Andrew Zhu; Li Zhang
>
> **摘要:** Using LLMs to generate formal planning languages such as PDDL that invokes symbolic solvers to deterministically derive plans has been shown to outperform generating plans directly. While this success has been limited to closed-sourced models or particular LLM pipelines, we design and evaluate 8 different PDDL generation pipelines with open-source models under 50 billion parameters previously shown to be incapable of this task. We find that intuitive approaches such as using a high-resource language wrapper or constrained decoding with grammar decrease performance, yet inference-time scaling approaches such as revision with feedback from the solver and plan validator more than double the performance.
>
---
#### [new 107] Decoding Phone Pairs from MEG Signals Across Speech Modalities
- **分类: cs.CL; cs.LG; cs.NE; cs.SD; eess.AS; I.2.6; I.5.1**

- **简介: 该研究通过MEG信号解码语音中的音素对，对比了主动说话与被动听觉任务的神经机制差异。旨在探索更优的脑电信号解码方法以改进脑机接口。研究使用17人数据，比较机器学习模型效果，发现主动说话解码准确率更高（76.6%），弹性网络模型最优，低频脑电波贡献显著，但需解决伪影干扰问题。**

- **链接: [http://arxiv.org/pdf/2505.15355v1](http://arxiv.org/pdf/2505.15355v1)**

> **作者:** Xabier de Zuazo; Eva Navas; Ibon Saratxaga; Mathieu Bourguignon; Nicola Molinaro
>
> **备注:** 21 pages, 4 figures, 1 graphical abstract, submitted to Computer Speech and Language (special issue on Iberian Languages)
>
> **摘要:** Understanding the neural mechanisms underlying speech production is essential for both advancing cognitive neuroscience theory and developing practical communication technologies. In this study, we investigated magnetoencephalography signals to decode phones from brain activity during speech production and perception (passive listening and voice playback) tasks. Using a dataset comprising 17 participants, we performed pairwise phone classification, extending our analysis to 15 phonetic pairs. Multiple machine learning approaches, including regularized linear models and neural network architectures, were compared to determine their effectiveness in decoding phonetic information. Our results demonstrate significantly higher decoding accuracy during speech production (76.6%) compared to passive listening and playback modalities (~51%), emphasizing the richer neural information available during overt speech. Among the models, the Elastic Net classifier consistently outperformed more complex neural networks, highlighting the effectiveness of traditional regularization techniques when applied to limited and high-dimensional MEG datasets. Besides, analysis of specific brain frequency bands revealed that low-frequency oscillations, particularly Delta (0.2-3 Hz) and Theta (4-7 Hz), contributed the most substantially to decoding accuracy, suggesting that these bands encode critical speech production-related neural processes. Despite using advanced denoising methods, it remains unclear whether decoding solely reflects neural activity or if residual muscular or movement artifacts also contributed, indicating the need for further methodological refinement. Overall, our findings underline the critical importance of examining overt speech production paradigms, which, despite their complexity, offer opportunities to improve brain-computer interfaces to help individuals with severe speech impairments.
>
---
#### [new 108] DeFTX: Denoised Sparse Fine-Tuning for Zero-Shot Cross-Lingual Transfer
- **分类: cs.CL; cs.AI; cs.LG; I.2.7**

- **简介: 该论文聚焦零样本跨语言迁移任务，解决高资源语言向低资源语言迁移效果差的问题。提出DeFT-X方法，通过奇异值分解对预训练模型权重去噪后再剪枝，优化稀疏微调策略。实验表明其在低资源语言的情感分类和自然语言推理任务中优于现有基线。**

- **链接: [http://arxiv.org/pdf/2505.15090v1](http://arxiv.org/pdf/2505.15090v1)**

> **作者:** Sona Elza Simon; Preethi Jyothi
>
> **摘要:** Effective cross-lingual transfer remains a critical challenge in scaling the benefits of large language models from high-resource to low-resource languages. Towards this goal, prior studies have explored many approaches to combine task knowledge from task-specific data in a (high-resource) source language and language knowledge from unlabeled text in a (low-resource) target language. One notable approach proposed composable sparse fine-tuning (SFT) for cross-lingual transfer that learns task-specific and language-specific sparse masks to select a subset of the pretrained model's parameters that are further fine-tuned. These sparse fine-tuned vectors (SFTs) are subsequently composed with the pretrained model to facilitate zero-shot cross-lingual transfer to a task in a target language, using only task-specific data from a source language. These sparse masks for SFTs were identified using a simple magnitude-based pruning. In our work, we introduce DeFT-X, a novel composable SFT approach that denoises the weight matrices of a pretrained model before magnitude pruning using singular value decomposition, thus yielding more robust SFTs. We evaluate DeFT-X on a diverse set of extremely low-resource languages for sentiment classification (NusaX) and natural language inference (AmericasNLI) and demonstrate that it performs at par or outperforms SFT and other prominent cross-lingual transfer baselines.
>
---
#### [new 109] VocalBench: Benchmarking the Vocal Conversational Abilities for Speech Interaction Models
- **分类: cs.CL**

- **简介: 该论文提出VocalBench，用于评估语音交互模型的综合能力。针对现有评估忽视语音质量、声学及环境因素的问题，构建含9,400个测试实例的基准，覆盖语义、声学、对话和鲁棒性四维度，揭示模型能力差异，指导未来研究。**

- **链接: [http://arxiv.org/pdf/2505.15727v1](http://arxiv.org/pdf/2505.15727v1)**

> **作者:** Heyang Liu; Yuhao Wang; Ziyang Cheng; Ronghua Wu; Qunshan Gu; Yanfeng Wang; Yu Wang
>
> **摘要:** The rapid advancement of large language models (LLMs) has accelerated the development of multi-modal models capable of vocal communication. Unlike text-based interactions, speech conveys rich and diverse information, including semantic content, acoustic variations, paralanguage cues, and environmental context. However, existing evaluations of speech interaction models predominantly focus on the quality of their textual responses, often overlooking critical aspects of vocal performance and lacking benchmarks with vocal-specific test instances. To address this gap, we propose VocalBench, a comprehensive benchmark designed to evaluate speech interaction models' capabilities in vocal communication. VocalBench comprises 9,400 carefully curated instances across four key dimensions: semantic quality, acoustic performance, conversational abilities, and robustness. It covers 16 fundamental skills essential for effective vocal interaction. Experimental results reveal significant variability in current model capabilities, each exhibiting distinct strengths and weaknesses, and provide valuable insights to guide future research in speech-based interaction systems. Code and evaluation instances are available at https://github.com/SJTU-OmniAgent/VocalBench.
>
---
#### [new 110] FlowKV: Enhancing Multi-Turn Conversational Coherence in LLMs via Isolated Key-Value Cache Management
- **分类: cs.CL**

- **简介: 该论文属于大语言模型多轮对话缓存管理任务。针对KV缓存线性增长导致计算成本高及重复压缩早期上下文引发信息丢失的问题，提出FlowKV机制：通过隔离多轮缓存仅压缩最新回合的KV对，保留历史压缩结果，减少灾难性遗忘，提升对话连贯性。**

- **链接: [http://arxiv.org/pdf/2505.15347v1](http://arxiv.org/pdf/2505.15347v1)**

> **作者:** Xiang Liu; Hong Chen; Xuming Hu; Xiaowen Chu
>
> **备注:** 18 pages
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in multi-turn conversational applications, where the management of the Key-Value (KV) Cache presents a significant bottleneck. The linear growth of the KV Cache with dialogue history imposes substantial computational costs, and existing eviction strategies often degrade performance by repeatedly compressing early conversational context, leading to information loss and context forgetting. This paper introduces FlowKV, a novel \textbf{multi-turn isolation mechanism} for KV Cache management, which can be applied to any KV Cache compression method without training. FlowKV's core innovation is a multi-turn isolation mechanism that preserves the accumulated compressed KV cache from past turns. Compression is then strategically applied only to the newly generated KV pairs of the latest completed turn, effectively preventing the re-compression of older context and thereby mitigating catastrophic forgetting. Our results demonstrate that FlowKV consistently and significantly outperforms baseline strategies in maintaining instruction-following accuracy and user preference retention from 10.90\% to 75.40\%, particularly in later conversational turns.
>
---
#### [new 111] UniErase: Unlearning Token as a Universal Erasure Primitive for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于机器unlearning任务，旨在解决语言模型知识移除时模型能力下降与遗忘效果不足的矛盾。提出UniErase方法，通过可学习的"遗忘标记"分两阶段引导模型定向遗忘：优化阶段绑定遗忘目标到概率分布，编辑阶段激活标记实现高效遗忘。在保留模型能力下提升遗忘效果，达SOTA性能。**

- **链接: [http://arxiv.org/pdf/2505.15674v1](http://arxiv.org/pdf/2505.15674v1)**

> **作者:** Miao Yu; Liang Lin; Guibin Zhang; Xinfeng Li; Junfeng Fang; Ningyu Zhang; Kun Wang; Yang Wang
>
> **摘要:** Large language models require iterative updates to address challenges such as knowledge conflicts and outdated information (e.g., incorrect, private, or illegal contents). Machine unlearning provides a systematic methodology for targeted knowledge removal from trained models, enabling elimination of sensitive information influences. However, mainstream fine-tuning-based unlearning methods often fail to balance unlearning efficacy and model ability, frequently resulting in catastrophic model collapse under extensive knowledge removal. Meanwhile, in-context unlearning, which relies solely on contextual prompting without modifying the model's intrinsic mechanisms, suffers from limited generalizability and struggles to achieve true unlearning. In this work, we introduce UniErase, a novel unlearning paradigm that employs learnable parametric suffix (unlearning token) to steer language models toward targeted forgetting behaviors. UniErase operates through two key phases: (I) an optimization stage that binds desired unlearning outputs to the model's autoregressive probability distribution via token optimization, followed by (II) a lightweight model editing phase that activates the learned token to probabilistically induce specified forgetting objective. Serving as a new research direction for token learning to induce unlearning target, UniErase achieves state-of-the-art (SOTA) performance across batch, sequential, and precise unlearning under fictitious and real-world knowledge settings. Remarkably, in terms of TOFU benchmark, UniErase, modifying only around 3.66% of the LLM parameters, outperforms previous forgetting SOTA baseline by around 4.01 times for model ability with even better unlearning efficacy. Similarly, UniErase, maintaining more ability, also surpasses previous retaining SOTA by 35.96% for unlearning efficacy, showing dual top-tier performances in current unlearing domain.
>
---
#### [new 112] Strategic Planning and Rationalizing on Trees Make LLMs Better Debaters
- **分类: cs.CL**

- **简介: 论文提出TreeDebater框架，解决辩论中的时间策略与互动说服问题。通过Rehearsal Tree预测攻防、评估论点强度，Debate Flow Tree跟踪辩论状态，优化时间分配与陈述调整。实验显示其优于现有系统，策略更贴近人类专家。（99字）**

- **链接: [http://arxiv.org/pdf/2505.14886v1](http://arxiv.org/pdf/2505.14886v1)**

> **作者:** Danqing Wang; Zhuorui Ye; Xinran Zhao; Fei Fang; Lei Li
>
> **备注:** 9 main pages
>
> **摘要:** Winning competitive debates requires sophisticated reasoning and argument skills. There are unique challenges in the competitive debate: (1) The time constraints force debaters to make strategic choices about which points to pursue rather than covering all possible arguments; (2) The persuasiveness of the debate relies on the back-and-forth interaction between arguments, which a single final game status cannot evaluate. To address these challenges, we propose TreeDebater, a novel debate framework that excels in competitive debate. We introduce two tree structures: the Rehearsal Tree and Debate Flow Tree. The Rehearsal Tree anticipates the attack and defenses to evaluate the strength of the claim, while the Debate Flow Tree tracks the debate status to identify the active actions. TreeDebater allocates its time budget among candidate actions and uses the speech time controller and feedback from the simulated audience to revise its statement. The human evaluation on both the stage-level and the debate-level comparison shows that our TreeDebater outperforms the state-of-the-art multi-agent debate system. Further investigation shows that TreeDebater shows better strategies in limiting time to important debate actions, aligning with the strategies of human debate experts.
>
---
#### [new 113] dKV-Cache: The Cache for Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文针对扩散语言模型（DLMs）推理速度慢的问题，提出dKV-Cache机制。通过分析token的动态表示差异，设计延迟缓存策略，提出dKV-Cache-Decode（无损加速）和dKV-Cache-Greedy（高速但降损）两种变体，实现2-10倍加速，缩小DLMs与自回归模型的推理速度差距，且无需重新训练。**

- **链接: [http://arxiv.org/pdf/2505.15781v1](http://arxiv.org/pdf/2505.15781v1)**

> **作者:** Xinyin Ma; Runpeng Yu; Gongfan Fang; Xinchao Wang
>
> **备注:** The code is available at https://github.com/horseee/dKV-Cache
>
> **摘要:** Diffusion Language Models (DLMs) have been seen as a promising competitor for autoregressive language models. However, diffusion language models have long been constrained by slow inference. A core challenge is that their non-autoregressive architecture and bidirectional attention preclude the key-value cache that accelerates decoding. We address this bottleneck by proposing a KV-cache-like mechanism, delayed KV-Cache, for the denoising process of DLMs. Our approach is motivated by the observation that different tokens have distinct representation dynamics throughout the diffusion process. Accordingly, we propose a delayed and conditioned caching strategy for key and value states. We design two complementary variants to cache key and value step-by-step: (1) dKV-Cache-Decode, which provides almost lossless acceleration, and even improves performance on long sequences, suggesting that existing DLMs may under-utilise contextual information during inference. (2) dKV-Cache-Greedy, which has aggressive caching with reduced lifespan, achieving higher speed-ups with quadratic time complexity at the cost of some performance degradation. dKV-Cache, in final, achieves from 2-10x speedup in inference, largely narrowing the gap between ARs and DLMs. We evaluate our dKV-Cache on several benchmarks, delivering acceleration across general language understanding, mathematical, and code-generation benchmarks. Experiments demonstrate that cache can also be used in DLMs, even in a training-free manner from current DLMs.
>
---
#### [new 114] Revealing Language Model Trajectories via Kullback-Leibler Divergence
- **分类: cs.CL**

- **简介: 该论文通过KL散度分析语言模型轨迹，研究模型训练及架构差异下的行为。任务为量化模型发展路径，解决跨架构/阶段的模型比较问题。工作包括系统评估预训练、微调及不同层的KL轨迹，发现预训练呈螺旋结构、层间为线状进展，且模型轨迹在log-likelihood空间比权重空间更受限。**

- **链接: [http://arxiv.org/pdf/2505.15353v1](http://arxiv.org/pdf/2505.15353v1)**

> **作者:** Ryo Kishino; Yusuke Takase; Momose Oyama; Hiroaki Yamagiwa; Hidetoshi Shimodaira
>
> **摘要:** A recently proposed method enables efficient estimation of the KL divergence between language models, including models with different architectures, by assigning coordinates based on log-likelihood vectors. To better understand the behavior of this metric, we systematically evaluate KL divergence across a wide range of conditions using publicly available language models. Our analysis covers comparisons between pretraining checkpoints, fine-tuned and base models, and layers via the logit lens. We find that trajectories of language models, as measured by KL divergence, exhibit a spiral structure during pretraining and thread-like progressions across layers. Furthermore, we show that, in terms of diffusion exponents, model trajectories in the log-likelihood space are more constrained than those in weight space.
>
---
#### [new 115] In-Domain African Languages Translation Using LLMs and Multi-armed Bandits
- **分类: cs.CL**

- **简介: 论文针对低资源非洲语言机器翻译的领域适应问题，提出基于多臂老虎机算法（如UCB、Thompson Sampling）的模型选择方法，解决数据不足下的最优模型选择难题。在三种非洲语言及有/无目标数据场景中验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.15069v1](http://arxiv.org/pdf/2505.15069v1)**

> **作者:** Pratik Rakesh Singh; Kritarth Prasad; Mohammadi Zaki; Pankaj Wasnik
>
> **摘要:** Neural Machine Translation (NMT) systems face significant challenges when working with low-resource languages, particularly in domain adaptation tasks. These difficulties arise due to limited training data and suboptimal model generalization, As a result, selecting an optimal model for translation is crucial for achieving strong performance on in-domain data, particularly in scenarios where fine-tuning is not feasible or practical. In this paper, we investigate strategies for selecting the most suitable NMT model for a given domain using bandit-based algorithms, including Upper Confidence Bound, Linear UCB, Neural Linear Bandit, and Thompson Sampling. Our method effectively addresses the resource constraints by facilitating optimal model selection with high confidence. We evaluate the approach across three African languages and domains, demonstrating its robustness and effectiveness in both scenarios where target data is available and where it is absent.
>
---
#### [new 116] Understanding 6G through Language Models: A Case Study on LLM-aided Structured Entity Extraction in Telecom Domain
- **分类: cs.CL; cs.SY; eess.SY**

- **简介: 该论文属于电信领域结构化实体抽取任务，旨在通过TeleSEE方法高效提取6G技术文本中的结构化实体。针对传统方法输出冗余、效率低的问题，提出token高效表示与分层并行解码技术，并构建6GTech数据集验证其高精度与5-9倍提速优势。**

- **链接: [http://arxiv.org/pdf/2505.14906v1](http://arxiv.org/pdf/2505.14906v1)**

> **作者:** Ye Yuan; Haolun Wu; Hao Zhou; Xue Liu; Hao Chen; Yan Xin; Jianzhong; Zhang
>
> **摘要:** Knowledge understanding is a foundational part of envisioned 6G networks to advance network intelligence and AI-native network architectures. In this paradigm, information extraction plays a pivotal role in transforming fragmented telecom knowledge into well-structured formats, empowering diverse AI models to better understand network terminologies. This work proposes a novel language model-based information extraction technique, aiming to extract structured entities from the telecom context. The proposed telecom structured entity extraction (TeleSEE) technique applies a token-efficient representation method to predict entity types and attribute keys, aiming to save the number of output tokens and improve prediction accuracy. Meanwhile, TeleSEE involves a hierarchical parallel decoding method, improving the standard encoder-decoder architecture by integrating additional prompting and decoding strategies into entity extraction tasks. In addition, to better evaluate the performance of the proposed technique in the telecom domain, we further designed a dataset named 6GTech, including 2390 sentences and 23747 words from more than 100 6G-related technical publications. Finally, the experiment shows that the proposed TeleSEE method achieves higher accuracy than other baseline techniques, and also presents 5 to 9 times higher sample processing speed.
>
---
#### [new 117] Word Level Timestamp Generation for Automatic Speech Recognition and Translation
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文提出基于Canary模型的单词级时间戳生成方法，用于自动语音识别与翻译任务。针对传统方法依赖外部对齐模块的问题，通过NeMo强制对齐器生成教师数据，引入<|timestamp|>标记直接预测词时间戳，在四语种中实现80-90%的精度（误差20-120ms），并扩展至翻译任务（误差约200ms），WER仅小幅下降。**

- **链接: [http://arxiv.org/pdf/2505.15646v1](http://arxiv.org/pdf/2505.15646v1)**

> **作者:** Ke Hu; Krishna Puvvada; Elena Rastorgueva; Zhehuai Chen; He Huang; Shuoyang Ding; Kunal Dhawan; Hainan Xu; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We introduce a data-driven approach for enabling word-level timestamp prediction in the Canary model. Accurate timestamp information is crucial for a variety of downstream tasks such as speech content retrieval and timed subtitles. While traditional hybrid systems and end-to-end (E2E) models may employ external modules for timestamp prediction, our approach eliminates the need for separate alignment mechanisms. By leveraging the NeMo Forced Aligner (NFA) as a teacher model, we generate word-level timestamps and train the Canary model to predict timestamps directly. We introduce a new <|timestamp|> token, enabling the Canary model to predict start and end timestamps for each word. Our method demonstrates precision and recall rates between 80% and 90%, with timestamp prediction errors ranging from 20 to 120 ms across four languages, with minimal WER degradation. Additionally, we extend our system to automatic speech translation (AST) tasks, achieving timestamp prediction errors around 200 milliseconds.
>
---
#### [new 118] ChartCards: A Chart-Metadata Generation Framework for Multi-Task Chart Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多任务图表理解任务，旨在解决传统方法依赖大量任务特定数据导致的高成本问题。提出ChartCards框架，生成图表元数据（含表格、代码、视觉元素及语义描述），构建MetaChart数据集（含10K表格、8.5万图表），实验表明微调模型性能平均提升5%，部分任务提升超20%。**

- **链接: [http://arxiv.org/pdf/2505.15046v1](http://arxiv.org/pdf/2505.15046v1)**

> **作者:** Yifan Wu; Lutao Yan; Leixian Shen; Yinan Mei; Jiannan Wang; Yuyu Luo
>
> **摘要:** The emergence of Multi-modal Large Language Models (MLLMs) presents new opportunities for chart understanding. However, due to the fine-grained nature of these tasks, applying MLLMs typically requires large, high-quality datasets for task-specific fine-tuning, leading to high data collection and training costs. To address this, we propose ChartCards, a unified chart-metadata generation framework for multi-task chart understanding. ChartCards systematically synthesizes various chart information, including data tables, visualization code, visual elements, and multi-dimensional semantic captions. By structuring this information into organized metadata, ChartCards enables a single chart to support multiple downstream tasks, such as text-to-chart retrieval, chart summarization, chart-to-table conversion, chart description, and chart question answering. Using ChartCards, we further construct MetaChart, a large-scale high-quality dataset containing 10,862 data tables, 85K charts, and 170 K high-quality chart captions. We validate the dataset through qualitative crowdsourcing evaluations and quantitative fine-tuning experiments across various chart understanding tasks. Fine-tuning six different models on MetaChart resulted in an average performance improvement of 5% across all tasks. The most notable improvements are seen in text-to-chart retrieval and chart-to-table tasks, with Long-CLIP and Llama 3.2-11B achieving improvements of 17% and 28%, respectively.
>
---
#### [new 119] DISCO Balances the Scales: Adaptive Domain- and Difficulty-Aware Reinforcement Learning on Imbalanced Data
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于强化学习对齐（RLHF）任务，旨在解决多领域不平衡数据下GRPO方法优化偏差问题。提出DISCO方法，通过领域感知奖励缩放和难度感知的自洽性优先策略，平衡领域优化并提升模型在长尾场景的泛化与公平性，实验显示其优于现有方法5%并创基准新高。**

- **链接: [http://arxiv.org/pdf/2505.15074v1](http://arxiv.org/pdf/2505.15074v1)**

> **作者:** Yuhang Zhou; Jing Zhu; Shengyi Qian; Zhuokai Zhao; Xiyao Wang; Xiaoyu Liu; Ming Li; Paiheng Xu; Wei Ai; Furong Huang
>
> **备注:** 13 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly aligned with human preferences through Reinforcement Learning from Human Feedback (RLHF). Among RLHF methods, Group Relative Policy Optimization (GRPO) has gained attention for its simplicity and strong performance, notably eliminating the need for a learned value function. However, GRPO implicitly assumes a balanced domain distribution and uniform semantic alignment across groups - assumptions that rarely hold in real-world datasets. When applied to multi-domain, imbalanced data, GRPO disproportionately optimizes for dominant domains, neglecting underrepresented ones and resulting in poor generalization and fairness. We propose Domain-Informed Self-Consistency Policy Optimization (DISCO), a principled extension to GRPO that addresses inter-group imbalance with two key innovations. Domain-aware reward scaling counteracts frequency bias by reweighting optimization based on domain prevalence. Difficulty-aware reward scaling leverages prompt-level self-consistency to identify and prioritize uncertain prompts that offer greater learning value. Together, these strategies promote more equitable and effective policy learning across domains. Extensive experiments across multiple LLMs and skewed training distributions show that DISCO improves generalization, outperforms existing GRPO variants by 5% on Qwen3 models, and sets new state-of-the-art results on multi-domain alignment benchmarks.
>
---
#### [new 120] Beyond Empathy: Integrating Diagnostic and Therapeutic Reasoning with Large Language Models for Mental Health Counseling
- **分类: cs.CL**

- **简介: 该论文提出PsyLLM模型，针对现有AI心理健康支持缺乏临床诊断和多元治疗的问题，整合DSM/ICD标准与CBT等疗法，通过自动化数据合成生成专业对话，并建立新评估基准，实验显示其性能更优。**

- **链接: [http://arxiv.org/pdf/2505.15715v1](http://arxiv.org/pdf/2505.15715v1)**

> **作者:** He Hu; Yucheng Zhou; Juzheng Si; Qianning Wang; Hengheng Zhang; Fuji Ren; Fei Ma; Laizhong Cui
>
> **摘要:** Large language models (LLMs) hold significant potential for mental health support, capable of generating empathetic responses and simulating therapeutic conversations. However, existing LLM-based approaches often lack the clinical grounding necessary for real-world psychological counseling, particularly in explicit diagnostic reasoning aligned with standards like the DSM/ICD and incorporating diverse therapeutic modalities beyond basic empathy or single strategies. To address these critical limitations, we propose PsyLLM, the first large language model designed to systematically integrate both diagnostic and therapeutic reasoning for mental health counseling. To develop the PsyLLM, we propose a novel automated data synthesis pipeline. This pipeline processes real-world mental health posts, generates multi-turn dialogue structures, and leverages LLMs guided by international diagnostic standards (e.g., DSM/ICD) and multiple therapeutic frameworks (e.g., CBT, ACT, psychodynamic) to simulate detailed clinical reasoning processes. Rigorous multi-dimensional filtering ensures the generation of high-quality, clinically aligned dialogue data. In addition, we introduce a new benchmark and evaluation protocol, assessing counseling quality across four key dimensions: comprehensiveness, professionalism, authenticity, and safety. Our experiments demonstrate that PsyLLM significantly outperforms state-of-the-art baseline models on this benchmark.
>
---
#### [new 121] NL-Debugging: Exploiting Natural Language as an Intermediate Representation for Code Debugging
- **分类: cs.CL**

- **简介: 该论文属于代码调试任务，旨在解决复杂编程错误中传统代码级分析不足的问题。提出NL-DEBUGGING框架，利用自然语言作为中间表示，通过语言层面的直接修改和执行反馈优化调试，拓宽修复空间，提升调试效果。**

- **链接: [http://arxiv.org/pdf/2505.15356v1](http://arxiv.org/pdf/2505.15356v1)**

> **作者:** Weiming Zhang; Qingyao Li; Xinyi Dai; Jizheng Chen; Kounianhua Du; Weinan Zhang; Weiwen Liu; Yasheng Wang; Ruiming Tang; Yong Yu
>
> **摘要:** Debugging is a critical aspect of LLM's coding ability. Early debugging efforts primarily focused on code-level analysis, which often falls short when addressing complex programming errors that require a deeper understanding of algorithmic logic. Recent advancements in large language models (LLMs) have shifted attention toward leveraging natural language reasoning to enhance code-related tasks. However, two fundamental questions remain unanswered: What type of natural language format is most effective for debugging tasks? And what specific benefits does natural language reasoning bring to the debugging process? In this paper, we introduce NL-DEBUGGING, a novel framework that employs natural language as an intermediate representation to improve code debugging. By debugging at a natural language level, we demonstrate that NL-DEBUGGING outperforms traditional debugging methods and enables a broader modification space through direct refinement guided by execution feedback. Our findings highlight the potential of natural language reasoning to advance automated code debugging and address complex programming challenges.
>
---
#### [new 122] Multimodal Cultural Safety: Evaluation Frameworks and Alignment Strategies
- **分类: cs.CL**

- **简介: 该论文属于多模态文化安全评估与优化任务，旨在解决视觉语言模型跨文化应用中的文化规范违反问题。团队构建CROSS基准和CROSS-Eval框架，评估21种模型的文化意识、合规性等维度，发现显著不足，并提出两种增强策略（监督微调和对比调优），显著提升模型表现。**

- **链接: [http://arxiv.org/pdf/2505.14972v1](http://arxiv.org/pdf/2505.14972v1)**

> **作者:** Haoyi Qiu; Kung-Hsiang Huang; Ruichen Zheng; Jiao Sun; Nanyun Peng
>
> **摘要:** Large vision-language models (LVLMs) are increasingly deployed in globally distributed applications, such as tourism assistants, yet their ability to produce culturally appropriate responses remains underexplored. Existing multimodal safety benchmarks primarily focus on physical safety and overlook violations rooted in cultural norms, which can result in symbolic harm. To address this gap, we introduce CROSS, a benchmark designed to assess the cultural safety reasoning capabilities of LVLMs. CROSS includes 1,284 multilingual visually grounded queries from 16 countries, three everyday domains, and 14 languages, where cultural norm violations emerge only when images are interpreted in context. We propose CROSS-Eval, an intercultural theory-based framework that measures four key dimensions: cultural awareness, norm education, compliance, and helpfulness. Using this framework, we evaluate 21 leading LVLMs, including mixture-of-experts models and reasoning models. Results reveal significant cultural safety gaps: the best-performing model achieves only 61.79% in awareness and 37.73% in compliance. While some open-source models reach GPT-4o-level performance, they still fall notably short of proprietary models. Our results further show that increasing reasoning capacity improves cultural alignment but does not fully resolve the issue. To improve model performance, we develop two enhancement strategies: supervised fine-tuning with culturally grounded, open-ended data and preference tuning with contrastive response pairs that highlight safe versus unsafe behaviors. These methods substantially improve GPT-4o's cultural awareness (+60.14%) and compliance (+55.2%), while preserving general multimodal capabilities with minimal performance reduction on general multimodal understanding benchmarks.
>
---
#### [new 123] TurnaboutLLM: A Deductive Reasoning Benchmark from Detective Games
- **分类: cs.CL**

- **简介: 该论文提出TurnaboutLLM框架及数据集，评估大语言模型（LLMs）在侦探游戏场景中的演绎推理能力。通过构建需识别长叙事中证词与证据矛盾的任务，测试12种模型，揭示现有策略（如Chain-of-Thought）的局限性，并分析上下文规模、推理步骤对性能的影响。**

- **链接: [http://arxiv.org/pdf/2505.15712v1](http://arxiv.org/pdf/2505.15712v1)**

> **作者:** Yuan Yuan; Muyu He; Muhammad Adil Shahid; Jiani Huang; Ziyang Li; Li Zhang
>
> **摘要:** This paper introduces TurnaboutLLM, a novel framework and dataset for evaluating the deductive reasoning abilities of Large Language Models (LLMs) by leveraging the interactive gameplay of detective games Ace Attorney and Danganronpa. The framework tasks LLMs with identifying contradictions between testimonies and evidences within long narrative contexts, a challenging task due to the large answer space and diverse reasoning types presented by its questions. We evaluate twelve state-of-the-art LLMs on the dataset, hinting at limitations of popular strategies for enhancing deductive reasoning such as extensive thinking and Chain-of-Thought prompting. The results also suggest varying effects of context size, the number of reasoning step and answer space size on model performance. Overall, TurnaboutLLM presents a substantial challenge for LLMs' deductive reasoning abilities in complex, narrative-rich environments.
>
---
#### [new 124] Thought-Augmented Policy Optimization: Bridging External Guidance and Internal Capabilities
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习策略优化任务，旨在解决传统RL过度依赖奖励最大化导致探索不足和推理能力受限的问题。提出TAPO框架，通过整合外部"思维模式"指导与模型内部探索，平衡两者以提升推理能力。实验显示其显著优于基线方法，且少量样本提炼的思维模式可跨任务泛化，增强可解释性。**

- **链接: [http://arxiv.org/pdf/2505.15692v1](http://arxiv.org/pdf/2505.15692v1)**

> **作者:** Jinyang Wu; Chonghua Liao; Mingkuan Feng; Shuai Zhang; Zhengqi Wen; Pengpeng Shao; Huazhe Xu; Jianhua Tao
>
> **摘要:** Reinforcement learning (RL) has emerged as an effective method for training reasoning models. However, existing RL approaches typically bias the model's output distribution toward reward-maximizing paths without introducing external knowledge. This limits their exploration capacity and results in a narrower reasoning capability boundary compared to base models. To address this limitation, we propose TAPO (Thought-Augmented Policy Optimization), a novel framework that augments RL by incorporating external high-level guidance ("thought patterns"). By adaptively integrating structured thoughts during training, TAPO effectively balances model-internal exploration and external guidance exploitation. Extensive experiments show that our approach significantly outperforms GRPO by 99% on AIME, 41% on AMC, and 17% on Minerva Math. Notably, these high-level thought patterns, abstracted from only 500 prior samples, generalize effectively across various tasks and models. This highlights TAPO's potential for broader applications across multiple tasks and domains. Our further analysis reveals that introducing external guidance produces powerful reasoning models with superior explainability of inference behavior and enhanced output readability.
>
---
#### [new 125] A Comparative Study of Large Language Models and Human Personality Traits
- **分类: cs.CL; cs.AI**

- **简介: 该研究比较大型语言模型（LLMs）与人类人格特质，探讨传统评估工具的适用性。通过三组实验发现LLMs人格动态、输入敏感，缺乏长期稳定性和内部一致性，提出分布式框架，为AI人格建模及人机交互提供理论支持。**

- **链接: [http://arxiv.org/pdf/2505.14845v1](http://arxiv.org/pdf/2505.14845v1)**

> **作者:** Wang Jiaqi; Wang bo; Guo fa; Cheng cheng; Yang li
>
> **摘要:** Large Language Models (LLMs) have demonstrated human-like capabilities in language comprehension and generation, becoming active participants in social and cognitive domains. This study investigates whether LLMs exhibit personality-like traits and how these traits compare with human personality, focusing on the applicability of conventional personality assessment tools. A behavior-based approach was used across three empirical studies. Study 1 examined test-retest stability and found that LLMs show higher variability and are more input-sensitive than humans, lacking long-term stability. Based on this, we propose the Distributed Personality Framework, conceptualizing LLM traits as dynamic and input-driven. Study 2 analyzed cross-variant consistency in personality measures and found LLMs' responses were highly sensitive to item wording, showing low internal consistency compared to humans. Study 3 explored personality retention during role-playing, showing LLM traits are shaped by prompt and parameter settings. These findings suggest that LLMs express fluid, externally dependent personality patterns, offering insights for constructing LLM-specific personality frameworks and advancing human-AI interaction. This work contributes to responsible AI development and extends the boundaries of personality psychology in the age of intelligent systems.
>
---
#### [new 126] Traveling Across Languages: Benchmarking Cross-Lingual Consistency in Multimodal LLMs
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于多模态大模型跨语言一致性评估任务，旨在解决模型在多语言环境下知识表达与视觉理解不一致的问题。提出KnowRecall（15语言视觉问答测文化知识）和VisRecall（9语言无图描述测视觉记忆）两个基准，实验显示当前模型跨语言一致性仍不足。**

- **链接: [http://arxiv.org/pdf/2505.15075v1](http://arxiv.org/pdf/2505.15075v1)**

> **作者:** Hao Wang; Pinzhi Huang; Jihan Yang; Saining Xie; Daisuke Kawahara
>
> **备注:** https://github.com/nlp-waseda/traveling-across-languages
>
> **摘要:** The rapid evolution of multimodal large language models (MLLMs) has significantly enhanced their real-world applications. However, achieving consistent performance across languages, especially when integrating cultural knowledge, remains a significant challenge. To better assess this issue, we introduce two new benchmarks: KnowRecall and VisRecall, which evaluate cross-lingual consistency in MLLMs. KnowRecall is a visual question answering benchmark designed to measure factual knowledge consistency in 15 languages, focusing on cultural and historical questions about global landmarks. VisRecall assesses visual memory consistency by asking models to describe landmark appearances in 9 languages without access to images. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, still struggle to achieve cross-lingual consistency. This underscores the need for more robust approaches that produce truly multilingual and culturally aware models.
>
---
#### [new 127] Emotional Supporters often Use Multiple Strategies in a Single Turn
- **分类: cs.CL**

- **简介: 该论文属于情感支持对话任务，旨在解决现有模型忽略单轮多策略使用的局限。通过分析ESConv数据，发现支持者常连续使用多策略，故重新定义任务为生成完整策略序列，并提出监督模型与LLM方法。实验显示LLM表现优于传统模型和人类，能更全面地提问与提建议。**

- **链接: [http://arxiv.org/pdf/2505.15316v1](http://arxiv.org/pdf/2505.15316v1)**

> **作者:** Xin Bai; Guanyi Chen; Tingting He; Chenlian Zhou; Yu Liu
>
> **摘要:** Emotional Support Conversations (ESC) are crucial for providing empathy, validation, and actionable guidance to individuals in distress. However, existing definitions of the ESC task oversimplify the structure of supportive responses, typically modelling them as single strategy-utterance pairs. Through a detailed corpus analysis of the ESConv dataset, we identify a common yet previously overlooked phenomenon: emotional supporters often employ multiple strategies consecutively within a single turn. We formally redefine the ESC task to account for this, proposing a revised formulation that requires generating the full sequence of strategy-utterance pairs given a dialogue history. To facilitate this refined task, we introduce several modelling approaches, including supervised deep learning models and large language models. Our experiments show that, under this redefined task, state-of-the-art LLMs outperform both supervised models and human supporters. Notably, contrary to some earlier findings, we observe that LLMs frequently ask questions and provide suggestions, demonstrating more holistic support capabilities.
>
---
#### [new 128] Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型引导任务，旨在解决数据多样性导致隐藏表示中噪声干扰概念向量的问题。提出SDCV方法，通过稀疏自编码器过滤噪声特征，提升线性探测和均值差异方法的引导成功率，经实验验证噪声假设成立。**

- **链接: [http://arxiv.org/pdf/2505.15038v1](http://arxiv.org/pdf/2505.15038v1)**

> **作者:** Haiyan Zhao; Xuansheng Wu; Fan Yang; Bo Shen; Ninghao Liu; Mengnan Du
>
> **备注:** 12 pages, 5 figures, 3 tables
>
> **摘要:** Linear Concept Vectors have proven effective for steering large language models (LLMs). While existing approaches like linear probing and difference-in-means derive these vectors from LLM hidden representations, diverse data introduces noises (i.e., irrelevant features) that challenge steering robustness. To address this, we propose Sparse Autoencoder-Denoised Concept Vectors (SDCV), which uses Sparse Autoencoders to filter out noisy features from hidden representations. When applied to linear probing and difference-in-means, our method improves their steering success rates. We validate our noise hypothesis through counterfactual experiments and feature visualizations.
>
---
#### [new 129] Are the confidence scores of reviewers consistent with the review content? Evidence from top conference proceedings in AI
- **分类: cs.CL; cs.AI; cs.HC; cs.IR**

- **简介: 该论文通过深度学习分析顶级AI会议审稿文本，评估审稿人信心评分与评审内容（词、句、方面层面）的一致性。解决现有研究缺乏细粒度文本-评分关联分析的问题，发现高信心评分与论文拒稿显著相关，验证了评审公平性。**

- **链接: [http://arxiv.org/pdf/2505.15031v1](http://arxiv.org/pdf/2505.15031v1)**

> **作者:** Wenqing Wu; Haixu Xi; Chengzhi Zhang
>
> **摘要:** Peer review is vital in academia for evaluating research quality. Top AI conferences use reviewer confidence scores to ensure review reliability, but existing studies lack fine-grained analysis of text-score consistency, potentially missing key details. This work assesses consistency at word, sentence, and aspect levels using deep learning and NLP conference review data. We employ deep learning to detect hedge sentences and aspects, then analyze report length, hedge word/sentence frequency, aspect mentions, and sentiment to evaluate text-score alignment. Correlation, significance, and regression tests examine confidence scores' impact on paper outcomes. Results show high text-score consistency across all levels, with regression revealing higher confidence scores correlate with paper rejection, validating expert assessments and peer review fairness.
>
---
#### [new 130] PhysicsArena: The First Multimodal Physics Reasoning Benchmark Exploring Variable, Process, and Solution Dimensions
- **分类: cs.CL; I.2.7; I.2.10**

- **简介: 论文提出PhysicsArena，首个多模态物理推理基准，解决现有测试局限于文本输入或仅侧重解题、忽略变量识别和过程制定的问题，通过评估变量识别、过程构建及解题推导三维度，全面评测MLLMs的物理推理能力。**

- **链接: [http://arxiv.org/pdf/2505.15472v1](http://arxiv.org/pdf/2505.15472v1)**

> **作者:** Song Dai; Yibo Yan; Jiamin Su; Dongfang Zihao; Yubo Gao; Yonghua Hei; Jungang Li; Junyan Zhang; Sicheng Tao; Zhuoran Gao; Xuming Hu
>
> **备注:** 27 pages,20 figures, EMNLP
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in diverse reasoning tasks, yet their application to complex physics reasoning remains underexplored. Physics reasoning presents unique challenges, requiring grounding in physical conditions and the interpretation of multimodal information. Current physics benchmarks are limited, often focusing on text-only inputs or solely on problem-solving, thereby overlooking the critical intermediate steps of variable identification and process formulation. To address these limitations, we introduce PhysicsArena, the first multimodal physics reasoning benchmark designed to holistically evaluate MLLMs across three critical dimensions: variable identification, physical process formulation, and solution derivation. PhysicsArena aims to provide a comprehensive platform for assessing and advancing the multimodal physics reasoning abilities of MLLMs.
>
---
#### [new 131] Automated Journalistic Questions: A New Method for Extracting 5W1H in French
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出首个自动化提取法语新闻5W1H信息的方法，解决 journalism领域缺乏此类工具的问题。设计新流程并构建250篇带人工标注的魁北克新闻语料库，实验结果与GPT-4表现相当。**

- **链接: [http://arxiv.org/pdf/2505.14804v1](http://arxiv.org/pdf/2505.14804v1)**

> **作者:** Richard Khoury; Maxence Verhaverbeke; Julie A. Gramaccia
>
> **备注:** 14 pages, 5 figures, 7 tables
>
> **摘要:** The 5W1H questions -- who, what, when, where, why and how -- are commonly used in journalism to ensure that an article describes events clearly and systematically. Answering them is a crucial prerequisites for tasks such as summarization, clustering, and news aggregation. In this paper, we design the first automated extraction pipeline to get 5W1H information from French news articles. To evaluate the performance of our algo- rithm, we also create a corpus of 250 Quebec news articles with 5W1H answers marked by four human annotators. Our results demonstrate that our pipeline performs as well in this task as the large language model GPT-4o.
>
---
#### [new 132] Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦提升大语言模型（LLMs）的可解释时序推理能力。针对LLMs依赖纯文本难以生成有效时序解释的问题，提出GETER框架：结合时序知识图谱与文本，通过结构编码器提取时序关系，利用适配器融合图特征与文本表征，最终生成解释文本。实验显示其效果与泛化性达前沿水平。**

- **链接: [http://arxiv.org/pdf/2505.15245v1](http://arxiv.org/pdf/2505.15245v1)**

> **作者:** Zihao Jiang; Ben Liu; Miao Peng; Wenjie Xu; Yao Xiao; Zhenyan Shan; Min Peng
>
> **备注:** In Findings of the Association for Computational Linguistics: ACL 2025
>
> **摘要:** While large language models (LLMs) show great potential in temporal reasoning, most existing work focuses heavily on enhancing performance, often neglecting the explainable reasoning processes underlying the results. To address this gap, we introduce a comprehensive benchmark covering a wide range of temporal granularities, designed to systematically evaluate LLMs' capabilities in explainable temporal reasoning. Furthermore, our findings reveal that LLMs struggle to deliver convincing explanations when relying solely on textual information. To address challenge, we propose GETER, a novel structure-aware generative framework that integrates Graph structures with text for Explainable TEmporal Reasoning. Specifically, we first leverage temporal knowledge graphs to develop a temporal encoder that captures structural information for the query. Subsequently, we introduce a structure-text prefix adapter to map graph structure features into the text embedding space. Finally, LLMs generate explanation text by seamlessly integrating the soft graph token with instruction-tuning prompt tokens. Experimental results indicate that GETER achieves state-of-the-art performance while also demonstrating its effectiveness as well as strong generalization capabilities. Our dataset and code are available at https://github.com/carryTatum/GETER.
>
---
#### [new 133] Feature Extraction and Steering for Enhanced Chain-of-Thought Reasoning in Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于提升大型语言模型（LLM）推理能力任务，针对长链式思维（CoT）需大量高质量数据的问题，提出基于稀疏自动编码器（SAE）提取特征并引导LLM内部状态，以及无需SAE的残差激活引导算法，减少对外部数据依赖，实验显示有效。（99字）**

- **链接: [http://arxiv.org/pdf/2505.15634v1](http://arxiv.org/pdf/2505.15634v1)**

> **作者:** Zihao Li; Xu Wang; Yuzhe Yang; Ziyu Yao; Haoyi Xiong; Mengnan Du
>
> **摘要:** Large Language Models (LLMs) demonstrate the ability to solve reasoning and mathematical problems using the Chain-of-Thought (CoT) technique. Expanding CoT length, as seen in models such as DeepSeek-R1, significantly enhances this reasoning for complex problems, but requires costly and high-quality long CoT data and fine-tuning. This work, inspired by the deep thinking paradigm of DeepSeek-R1, utilizes a steering technique to enhance the reasoning ability of an LLM without external datasets. Our method first employs Sparse Autoencoders (SAEs) to extract interpretable features from vanilla CoT. These features are then used to steer the LLM's internal states during generation. Recognizing that many LLMs do not have corresponding pre-trained SAEs, we further introduce a novel SAE-free steering algorithm, which directly computes steering directions from the residual activations of an LLM, obviating the need for an explicit SAE. Experimental results demonstrate that both our SAE-based and subsequent SAE-free steering algorithms significantly enhance the reasoning capabilities of LLMs.
>
---
#### [new 134] Improving LLM First-Token Predictions in Multiple-Choice Question Answering via Prefilling Attack
- **分类: cs.CL**

- **简介: 该论文针对多选问答中LLM首个词预测（FTP）的误判和误解问题，提出预填充攻击方法。通过添加结构化前缀（如"正确选项是："）引导模型输出有效答案，提升FTP评估的准确性、校准性和效率，实验显示其性能优于传统FTP且效率更高。**

- **链接: [http://arxiv.org/pdf/2505.15323v1](http://arxiv.org/pdf/2505.15323v1)**

> **作者:** Silvia Cappelletti; Tobia Poppi; Samuele Poppi; Zheng-Xin Yong; Diego Garcia-Olano; Marcella Cornia; Lorenzo Baraldi; Rita Cucchiara
>
> **备注:** 13 pages, 5 figures, 7 tables
>
> **摘要:** Large Language Models (LLMs) are increasingly evaluated on multiple-choice question answering (MCQA) tasks using *first-token probability* (FTP), which selects the answer option whose initial token has the highest likelihood. While efficient, FTP can be fragile: models may assign high probability to unrelated tokens (*misalignment*) or use a valid token merely as part of a generic preamble rather than as a clear answer choice (*misinterpretation*), undermining the reliability of symbolic evaluation. We propose a simple solution: the *prefilling attack*, a structured natural-language prefix (e.g., "*The correct option is:*") prepended to the model output. Originally explored in AI safety, we repurpose prefilling to steer the model to respond with a clean, valid option, without modifying its parameters. Empirically, the FTP with prefilling strategy substantially improves accuracy, calibration, and output consistency across a broad set of LLMs and MCQA benchmarks. It outperforms standard FTP and often matches the performance of open-ended generation approaches that require full decoding and external classifiers, while being significantly more efficient. Our findings suggest that prefilling is a simple, robust, and low-cost method to enhance the reliability of FTP-based evaluation in multiple-choice settings.
>
---
#### [new 135] Reliable Decision Support with LLMs: A Framework for Evaluating Consistency in Binary Text Classification Applications
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 论文提出评估LLM在二分类文本分类一致性的框架，解决可靠性评估方法缺失问题。采用心理测量学原则，确定样本量、开发无效响应指标及评估一致性。测试14个模型在金融新闻分类，发现高内部一致性但预测市场表现随机，框架指导模型选择与资源优化。（99字）**

- **链接: [http://arxiv.org/pdf/2505.14918v1](http://arxiv.org/pdf/2505.14918v1)**

> **作者:** Fadel M. Megahed; Ying-Ju Chen; L. Allision Jones-Farmer; Younghwa Lee; Jiawei Brooke Wang; Inez M. Zwetsloot
>
> **备注:** 25 pages
>
> **摘要:** This study introduces a framework for evaluating consistency in large language model (LLM) binary text classification, addressing the lack of established reliability assessment methods. Adapting psychometric principles, we determine sample size requirements, develop metrics for invalid responses, and evaluate intra- and inter-rater reliability. Our case study examines financial news sentiment classification across 14 LLMs (including claude-3-7-sonnet, gpt-4o, deepseek-r1, gemma3, llama3.2, phi4, and command-r-plus), with five replicates per model on 1,350 articles. Models demonstrated high intra-rater consistency, achieving perfect agreement on 90-98% of examples, with minimal differences between expensive and economical models from the same families. When validated against StockNewsAPI labels, models achieved strong performance (accuracy 0.76-0.88), with smaller models like gemma3:1B, llama3.2:3B, and claude-3-5-haiku outperforming larger counterparts. All models performed at chance when predicting actual market movements, indicating task constraints rather than model limitations. Our framework provides systematic guidance for LLM selection, sample size planning, and reliability assessment, enabling organizations to optimize resources for classification tasks.
>
---
#### [new 136] Can Large Language Models be Effective Online Opinion Miners?
- **分类: cs.CL**

- **简介: 该论文属于在线观点挖掘任务，旨在解决传统方法难以处理复杂用户生成内容的问题。提出OOMB数据集及评估协议，通过标注三元组和摘要，评估LLMs的提取与生成能力，分析其在真实场景中的挑战与适应性，为LLM观点挖掘奠定研究基础。**

- **链接: [http://arxiv.org/pdf/2505.15695v1](http://arxiv.org/pdf/2505.15695v1)**

> **作者:** Ryang Heo; Yongsik Seo; Junseong Lee; Dongha Lee
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** The surge of user-generated online content presents a wealth of insights into customer preferences and market trends. However, the highly diverse, complex, and context-rich nature of such contents poses significant challenges to traditional opinion mining approaches. To address this, we introduce Online Opinion Mining Benchmark (OOMB), a novel dataset and evaluation protocol designed to assess the ability of large language models (LLMs) to mine opinions effectively from diverse and intricate online environments. OOMB provides extensive (entity, feature, opinion) tuple annotations and a comprehensive opinion-centric summary that highlights key opinion topics within each content, thereby enabling the evaluation of both the extractive and abstractive capabilities of models. Through our proposed benchmark, we conduct a comprehensive analysis of which aspects remain challenging and where LLMs exhibit adaptability, to explore whether they can effectively serve as opinion miners in realistic online scenarios. This study lays the foundation for LLM-based opinion mining and discusses directions for future research in this field.
>
---
#### [new 137] A Federated Splitting Framework for LLMs: Security, Efficiency, and Adaptability
- **分类: cs.CL; cs.AI; cs.DC**

- **简介: 该论文提出FL-LLaMA框架，解决联邦学习中LLM的分割问题，针对隐私泄露、通信低效及固定分割点缺陷，通过注入噪声保护隐私，采用并行策略与压缩技术提升效率，支持动态调整分割点，实验显示性能接近中心化模型，训练加速2倍，推理8倍。**

- **链接: [http://arxiv.org/pdf/2505.15683v1](http://arxiv.org/pdf/2505.15683v1)**

> **作者:** Zishuai Zhang; Hainan Zhang; Jiaying Zheng; Ziwei Wang; Yongxin Tong; Jin Dong; Zhiming Zheng
>
> **摘要:** Private data is typically larger and of higher quality than public data, offering great potential to improve LLM. However, its scattered distribution across data silos and the high computational demands of LLMs limit their deployment in federated environments. To address this, the transformer-based split learning model has emerged, offloading most model parameters to the server while retaining only the embedding and output layers on clients to ensure privacy. However, it still faces significant challenges in security, efficiency, and adaptability: 1) embedding gradients are vulnerable to attacks, leading to reverse engineering of private data; 2) the autoregressive nature of LLMs means that federated split learning can only train and infer sequentially, causing high communication overhead; 3) fixed partition points lack adaptability to downstream tasks. In this paper, we introduce FL-LLaMA, a secure, efficient, and adaptive federated split framework based on LLaMA2. First, we place some input and output blocks on the local client and inject Gaussian noise into forward-pass hidden states, enabling secure end-to-end propagation. Second, we employ client-batch and server-hierarchical strategies to achieve parallel training, along with attention-mask compression and KV cache mechanisms to accelerate inference, reducing communication costs effectively. Third, we allow users to dynamically adjust the partition points for input/output blocks based on specific task requirements and hardware limitations. Experiments on NLU, summarization and conversational QA tasks show that FL-LLaMA maintains performance comparable to centralized LLaMA2, and achieves up to 2x train speedups and 8x inference speedups. Further analysis of privacy attacks and different partition points also demonstrates the effectiveness of FL-LLaMA in security and adaptability.
>
---
#### [new 138] Towards Inclusive ASR: Investigating Voice Conversion for Dysarthric Speech Recognition in Low-Resource Languages
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 论文针对低资源语言中构音障碍语音识别数据稀缺问题，提出通过语音转换模型将健康非英语语音转化为类似构音障碍语音，结合多语言ASR模型微调提升识别性能，实验验证该方法优于传统数据增强技术。**

- **链接: [http://arxiv.org/pdf/2505.14874v1](http://arxiv.org/pdf/2505.14874v1)**

> **作者:** Chin-Jou Li; Eunjung Yeo; Kwanghee Choi; Paula Andrea Pérez-Toro; Masao Someki; Rohan Kumar Das; Zhengjun Yue; Juan Rafael Orozco-Arroyave; Elmar Nöth; David R. Mortensen
>
> **备注:** 5 pages, 1 figure, Accepted to Interspeech 2025
>
> **摘要:** Automatic speech recognition (ASR) for dysarthric speech remains challenging due to data scarcity, particularly in non-English languages. To address this, we fine-tune a voice conversion model on English dysarthric speech (UASpeech) to encode both speaker characteristics and prosodic distortions, then apply it to convert healthy non-English speech (FLEURS) into non-English dysarthric-like speech. The generated data is then used to fine-tune a multilingual ASR model, Massively Multilingual Speech (MMS), for improved dysarthric speech recognition. Evaluation on PC-GITA (Spanish), EasyCall (Italian), and SSNCE (Tamil) demonstrates that VC with both speaker and prosody conversion significantly outperforms the off-the-shelf MMS performance and conventional augmentation techniques such as speed and tempo perturbation. Objective and subjective analyses of the generated data further confirm that the generated speech simulates dysarthric characteristics.
>
---
#### [new 139] NeoN: A Tool for Automated Detection, Linguistic and LLM-Driven Analysis of Neologisms in Polish
- **分类: cs.CL**

- **简介: 该论文提出工具NeoN，属于波兰语新词检测与分析任务。解决传统依赖词典、人工成本高的问题。工作：构建多层管道，整合语料库、波兰语语言规则及LLM，通过词形还原、频率分析提取候选新词，LLM自动生成定义并分类，界面支持可视化验证，提升效率与精度。**

- **链接: [http://arxiv.org/pdf/2505.15426v1](http://arxiv.org/pdf/2505.15426v1)**

> **作者:** Aleksandra Tomaszewska; Dariusz Czerski; Bartosz Żuk; Maciej Ogrodniczuk
>
> **备注:** 15 pages, this is an extended version of a paper accepted for the 25th International Conference on Computational Science (ICCS), 7-9 July 2025
>
> **摘要:** NeoN, a tool for detecting and analyzing Polish neologisms. Unlike traditional dictionary-based methods requiring extensive manual review, NeoN combines reference corpora, Polish-specific linguistic filters, an LLM-driven precision-boosting filter, and daily RSS monitoring in a multi-layered pipeline. The system uses context-aware lemmatization, frequency analysis, and orthographic normalization to extract candidate neologisms while consolidating inflectional variants. Researchers can verify candidates through an intuitive interface with visualizations and filtering controls. An integrated LLM module automatically generates definitions and categorizes neologisms by domain and sentiment. Evaluations show NeoN maintains high accuracy while significantly reducing manual effort, providing an accessible solution for tracking lexical innovation in Polish.
>
---
#### [new 140] Text Generation Beyond Discrete Token Sampling
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本生成任务，旨在解决标准自回归生成中丢弃token分布导致的信息损失问题。提出无训练的Mixture of Inputs（MoI）方法，通过贝叶斯估计融合采样token与分布，提升生成质量和推理能力，在多模型及数学、代码、高难度QA任务上有效。**

- **链接: [http://arxiv.org/pdf/2505.14827v1](http://arxiv.org/pdf/2505.14827v1)**

> **作者:** Yufan Zhuang; Liyuan Liu; Chandan Singh; Jingbo Shang; Jianfeng Gao
>
> **摘要:** In standard autoregressive generation, an LLM predicts the next-token distribution, samples a discrete token, and then discards the distribution, passing only the sampled token as new input. To preserve this distribution's rich information, we propose Mixture of Inputs (MoI), a training-free method for autoregressive generation. After generating a token following the standard paradigm, we construct a new input that blends the generated discrete token with the previously discarded token distribution. Specifically, we employ a Bayesian estimation method that treats the token distribution as the prior, the sampled token as the observation, and replaces the conventional one-hot vector with the continuous posterior expectation as the new model input. MoI allows the model to maintain a richer internal representation throughout the generation process, resulting in improved text quality and reasoning capabilities. On mathematical reasoning, code generation, and PhD-level QA tasks, MoI consistently improves performance across multiple models including QwQ-32B, Nemotron-Super-49B, Gemma-3-27B, and DAPO-Qwen-32B, with no additional training and negligible computational overhead.
>
---
#### [new 141] Saten: Sparse Augmented Tensor Networks for Post-Training Compression of Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Saten框架，解决预训练大语言模型（LLM）在下游任务中因高秩特性和缺乏预训练数据导致的压缩难题。通过稀疏增强张量网络，在微调阶段优化低秩张量化模型，提升压缩效率与精度，达状态-of-the-art性能。**

- **链接: [http://arxiv.org/pdf/2505.14871v1](http://arxiv.org/pdf/2505.14871v1)**

> **作者:** Ryan Solgi; Kai Zhen; Rupak Vignesh Swaminathan; Nathan Susanj; Athanasios Mouchtaris; Siegfried Kunzmann; Zheng Zhang
>
> **摘要:** The efficient implementation of large language models (LLMs) is crucial for deployment on resource-constrained devices. Low-rank tensor compression techniques, such as tensor-train (TT) networks, have been widely studied for over-parameterized neural networks. However, their applications to compress pre-trained large language models (LLMs) for downstream tasks (post-training) remains challenging due to the high-rank nature of pre-trained LLMs and the lack of access to pretraining data. In this study, we investigate low-rank tensorized LLMs during fine-tuning and propose sparse augmented tensor networks (Saten) to enhance their performance. The proposed Saten framework enables full model compression. Experimental results demonstrate that Saten enhances both accuracy and compression efficiency in tensorized language models, achieving state-of-the-art performance.
>
---
#### [new 142] LFTF: Locating First and Then Fine-Tuning for Mitigating Gender Bias in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLMs）因训练数据导致的性别偏见问题，提出LFTF算法：通过BMI评分定位关键模块并微调，同时开发GenBiasEval/GenHintEval数据集及AFGB/UB-Score指标评估偏见程度与一致性。实验表明该方法有效降低偏见并保持模型性能。**

- **链接: [http://arxiv.org/pdf/2505.15475v1](http://arxiv.org/pdf/2505.15475v1)**

> **作者:** Zhanyue Qin; Yue Ding; Deyuan Liu; Qingbin Liu; Junxian Cai; Xi Chen; Zhiying Tu; Dianhui Chu; Cuiyun Gao; Dianbo Sui
>
> **摘要:** Nowadays, Large Language Models (LLMs) have attracted widespread attention due to their powerful performance. However, due to the unavoidable exposure to socially biased data during training, LLMs tend to exhibit social biases, particularly gender bias. To better explore and quantifying the degree of gender bias in LLMs, we propose a pair of datasets named GenBiasEval and GenHintEval, respectively. The GenBiasEval is responsible for evaluating the degree of gender bias in LLMs, accompanied by an evaluation metric named AFGB-Score (Absolutely Fair Gender Bias Score). Meanwhile, the GenHintEval is used to assess whether LLMs can provide responses consistent with prompts that contain gender hints, along with the accompanying evaluation metric UB-Score (UnBias Score). Besides, in order to mitigate gender bias in LLMs more effectively, we present the LFTF (Locating First and Then Fine-Tuning) algorithm.The algorithm first ranks specific LLM blocks by their relevance to gender bias in descending order using a metric called BMI (Block Mitigating Importance Score). Based on this ranking, the block most strongly associated with gender bias is then fine-tuned using a carefully designed loss function. Numerous experiments have shown that our proposed LFTF algorithm can significantly mitigate gender bias in LLMs while maintaining their general capabilities.
>
---
#### [new 143] CoLA: Collaborative Low-Rank Adaptation
- **分类: cs.CL**

- **简介: 该论文提出CoLA方法，针对多任务场景下LoRA存在的任务干扰及样本稀缺问题，通过灵活的低秩适配架构和三项协作策略优化矩阵A/B的交互，提升参数高效微调的性能与鲁棒性，尤其在小样本任务中超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.15471v1](http://arxiv.org/pdf/2505.15471v1)**

> **作者:** Yiyun Zhou; Chang Yao; Jingyuan Chen
>
> **备注:** Accepted by ACL 2025, Findings
>
> **摘要:** The scaling law of Large Language Models (LLMs) reveals a power-law relationship, showing diminishing return on performance as model scale increases. While training LLMs from scratch is resource-intensive, fine-tuning a pre-trained model for specific tasks has become a practical alternative. Full fine-tuning (FFT) achieves strong performance; however, it is computationally expensive and inefficient. Parameter-efficient fine-tuning (PEFT) methods, like LoRA, have been proposed to address these challenges by freezing the pre-trained model and adding lightweight task-specific modules. LoRA, in particular, has proven effective, but its application to multi-task scenarios is limited by interference between tasks. Recent approaches, such as Mixture-of-Experts (MOE) and asymmetric LoRA, have aimed to mitigate these issues but still struggle with sample scarcity and noise interference due to their fixed structure. In response, we propose CoLA, a more flexible LoRA architecture with an efficient initialization scheme, and introduces three collaborative strategies to enhance performance by better utilizing the quantitative relationships between matrices $A$ and $B$. Our experiments demonstrate the effectiveness and robustness of CoLA, outperforming existing PEFT methods, especially in low-sample scenarios. Our data and code are fully publicly available at https://github.com/zyy-2001/CoLA.
>
---
#### [new 144] AGENT-X: Adaptive Guideline-based Expert Network for Threshold-free AI-generated teXt detection
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决现有方法依赖大量标注数据、阈值调优及解释性差的问题。提出AGENT-X框架，基于语言学理论构建多智能体系统，通过语义、风格、结构维度分析文本，结合自适应路由与信心聚合实现零样本、无阈值检测，提升准确性和可解释性。**

- **链接: [http://arxiv.org/pdf/2505.15261v1](http://arxiv.org/pdf/2505.15261v1)**

> **作者:** Jiatao Li; Mao Ye; Cheng Peng; Xunjian Yin; Xiaojun Wan
>
> **摘要:** Existing AI-generated text detection methods heavily depend on large annotated datasets and external threshold tuning, restricting interpretability, adaptability, and zero-shot effectiveness. To address these limitations, we propose AGENT-X, a zero-shot multi-agent framework informed by classical rhetoric and systemic functional linguistics. Specifically, we organize detection guidelines into semantic, stylistic, and structural dimensions, each independently evaluated by specialized linguistic agents that provide explicit reasoning and robust calibrated confidence via semantic steering. A meta agent integrates these assessments through confidence-aware aggregation, enabling threshold-free, interpretable classification. Additionally, an adaptive Mixture-of-Agent router dynamically selects guidelines based on inferred textual characteristics. Experiments on diverse datasets demonstrate that AGENT-X substantially surpasses state-of-the-art supervised and zero-shot approaches in accuracy, interpretability, and generalization.
>
---
#### [new 145] MIKU-PAL: An Automated and Standardized Multi-Modal Method for Speech Paralinguistic and Affect Labeling
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 论文提出MIKU-PAL，一种自动化多模态方法，解决大规模情感语音数据采集的一致性与成本问题。通过面部检测、追踪及多模态语言模型分析，实现高精度（68.5%）与高一致性（0.93 Fleiss kappa），标注26种情感类别，并发布131.2小时的MIKU-EmoBench数据集，用于情感TTS和视觉克隆基准。**

- **链接: [http://arxiv.org/pdf/2505.15772v1](http://arxiv.org/pdf/2505.15772v1)**

> **作者:** Cheng Yifan; Zhang Ruoyi; Shi Jiatong
>
> **备注:** Accepted by Interspeech
>
> **摘要:** Acquiring large-scale emotional speech data with strong consistency remains a challenge for speech synthesis. This paper presents MIKU-PAL, a fully automated multimodal pipeline for extracting high-consistency emotional speech from unlabeled video data. Leveraging face detection and tracking algorithms, we developed an automatic emotion analysis system using a multimodal large language model (MLLM). Our results demonstrate that MIKU-PAL can achieve human-level accuracy (68.5% on MELD) and superior consistency (0.93 Fleiss kappa score) while being much cheaper and faster than human annotation. With the high-quality, flexible, and consistent annotation from MIKU-PAL, we can annotate fine-grained speech emotion categories of up to 26 types, validated by human annotators with 83% rationality ratings. Based on our proposed system, we further released a fine-grained emotional speech dataset MIKU-EmoBench(131.2 hours) as a new benchmark for emotional text-to-speech and visual voice cloning.
>
---
#### [new 146] SUS backprop: linear backpropagation algorithm for long inputs in transformers
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SUS backprop算法，针对Transformer处理长序列时注意力机制计算复杂度二次增长的问题，通过随机切断99%的注意力梯度（保留每个token和头c≈20-30个连接），将反向传播复杂度从O(n²)降至线性O(nc)，仅增加1%梯度方差，提升长序列训练效率。**

- **链接: [http://arxiv.org/pdf/2505.15080v1](http://arxiv.org/pdf/2505.15080v1)**

> **作者:** Sergey Pankov; Georges Harik
>
> **备注:** 21 pages, 9 figures
>
> **摘要:** It is straightforward to design an unbiased gradient estimator that stochastically cuts the backpropagation flow through any part of a computational graph. By cutting the parts that have little effect on the computation, one can potentially save a significant amount of back-propagation computation in exchange for a minimal increase in the stochastic gradient variance, in some situations. Such a situation occurs in the attention mechanism of the transformer architecture. For long sequences, attention becomes the limiting factor, as its compute requirements increase quadratically with sequence length $n$. At the same time, most attention weights become very small, as most attention heads tend to connect a given token with only a small fraction of other tokens in the sequence. These weights become promising targets for cutting backpropagation. We propose a simple probabilistic rule controlled by a single parameter $c$ that cuts backpropagation through most attention weights, leaving at most $c$ interactions per token per attention head. This brings a factor of $c/n$ reduction in the compute required for the attention backpropagation, turning it from quadratic $O(n^2)$ to linear complexity $O(nc)$. We have empirically verified that, for a typical transformer model, cutting $99\%$ of the attention gradient flow (i.e. choosing $c \sim 20-30$) results in relative gradient variance increase of only about $1\%$ for $n \sim 2000$, and it decreases with $n$. This approach is amenable to efficient sparse matrix implementation, thus being promising for making the cost of a backward pass negligible relative to the cost of a forward pass when training a transformer model on long sequences.
>
---
#### [new 147] Explainable embeddings with Distance Explainer
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; 68T99; I.2.m**

- **简介: 该论文提出Distance Explainer方法，解决嵌入空间可解释性问题，通过遮罩和距离排序解释数据间相似性/差异性，评估于跨模态任务，有效识别特征贡献并分析参数影响。**

- **链接: [http://arxiv.org/pdf/2505.15516v1](http://arxiv.org/pdf/2505.15516v1)**

> **作者:** Christiaan Meijer; E. G. Patrick Bos
>
> **备注:** 33 pages, 19 figures. Submitted to JMLR. Method implementation: https://research-software-directory.org/software/distance-explainer
>
> **摘要:** While eXplainable AI (XAI) has advanced significantly, few methods address interpretability in embedded vector spaces where dimensions represent complex abstractions. We introduce Distance Explainer, a novel method for generating local, post-hoc explanations of embedded spaces in machine learning models. Our approach adapts saliency-based techniques from RISE to explain the distance between two embedded data points by assigning attribution values through selective masking and distance-ranked mask filtering. We evaluate Distance Explainer on cross-modal embeddings (image-image and image-caption pairs) using established XAI metrics including Faithfulness, Sensitivity/Robustness, and Randomization. Experiments with ImageNet and CLIP models demonstrate that our method effectively identifies features contributing to similarity or dissimilarity between embedded data points while maintaining high robustness and consistency. We also explore how parameter tuning, particularly mask quantity and selection strategy, affects explanation quality. This work addresses a critical gap in XAI research and enhances transparency and trustworthiness in deep learning applications utilizing embedded spaces.
>
---
#### [new 148] TCSinger 2: Customizable Multilingual Zero-shot Singing Voice Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出TCSinger 2，一种多任务多语言零样本歌声合成模型。针对现有模型依赖音素/音符边界标注、过渡不自然及风格控制不足的问题，其通过模糊边界编码器优化过渡、定制音频编码器提取多模态风格特征、流式变换器提升音质与风格可控性，实验证明效果更优。**

- **链接: [http://arxiv.org/pdf/2505.14910v1](http://arxiv.org/pdf/2505.14910v1)**

> **作者:** Yu Zhang; Wenxiang Guo; Changhao Pan; Dongyu Yao; Zhiyuan Zhu; Ziyue Jiang; Yuhan Wang; Tao Jin; Zhou Zhao
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Customizable multilingual zero-shot singing voice synthesis (SVS) has various potential applications in music composition and short video dubbing. However, existing SVS models overly depend on phoneme and note boundary annotations, limiting their robustness in zero-shot scenarios and producing poor transitions between phonemes and notes. Moreover, they also lack effective multi-level style control via diverse prompts. To overcome these challenges, we introduce TCSinger 2, a multi-task multilingual zero-shot SVS model with style transfer and style control based on various prompts. TCSinger 2 mainly includes three key modules: 1) Blurred Boundary Content (BBC) Encoder, predicts duration, extends content embedding, and applies masking to the boundaries to enable smooth transitions. 2) Custom Audio Encoder, uses contrastive learning to extract aligned representations from singing, speech, and textual prompts. 3) Flow-based Custom Transformer, leverages Cus-MOE, with F0 supervision, enhancing both the synthesis quality and style modeling of the generated singing voice. Experimental results show that TCSinger 2 outperforms baseline models in both subjective and objective metrics across multiple related tasks.
>
---
#### [new 149] Large Language Models as Computable Approximations to Solomonoff Induction
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文将大语言模型（LLM）与算法信息论联系，提出LLM的训练通过最小化损失近似Solomonoff先验，预测过程实现其归纳推理，统一解释了in-context、few-shot学习及扩展规律，并提出基于低置信度样本选择的优化方法，实验显示其提升小模型性能。**

- **链接: [http://arxiv.org/pdf/2505.15784v1](http://arxiv.org/pdf/2505.15784v1)**

> **作者:** Jun Wan; Lingrui Mei
>
> **备注:** Both authors contributed equally
>
> **摘要:** The rapid advancement of large language models (LLMs) calls for a rigorous theoretical framework to explain their empirical success. While significant progress has been made in understanding LLM behaviors, existing theoretical frameworks remain fragmented in explaining emergent phenomena through a unified mathematical lens. We establish the first formal connection between LLM architectures and Algorithmic Information Theory (AIT) by proving two fundamental results: (1) the training process computationally approximates Solomonoff prior through loss minimization interpreted as program length optimization, and (2) next-token prediction implements approximate Solomonoff induction. We leverage AIT to provide a unified theoretical explanation for in-context learning, few-shot learning, and scaling laws. Furthermore, our theoretical insights lead to a principled method for few-shot example selection that prioritizes samples where models exhibit lower predictive confidence. We demonstrate through experiments on diverse text classification benchmarks that this strategy yields significant performance improvements, particularly for smaller model architectures, when compared to selecting high-confidence examples. Our framework bridges the gap between theoretical foundations and practical LLM behaviors, providing both explanatory power and actionable insights for future model development.
>
---
#### [new 150] Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属LLM推理任务，针对价值方法未充分探索问题，提出轨迹贝尔曼残差最小化（TBRM）算法。通过优化轨迹级贝尔曼目标，去除critics等组件，仅需单次rollout，证明收敛性并实验优于策略方法如PPO，计算效率高。**

- **链接: [http://arxiv.org/pdf/2505.15311v1](http://arxiv.org/pdf/2505.15311v1)**

> **作者:** Yurun Yuan; Fan Chen; Zeyu Jia; Alexander Rakhlin; Tengyang Xie
>
> **摘要:** Policy-based methods currently dominate reinforcement learning (RL) pipelines for large language model (LLM) reasoning, leaving value-based approaches largely unexplored. We revisit the classical paradigm of Bellman Residual Minimization and introduce Trajectory Bellman Residual Minimization (TBRM), an algorithm that naturally adapts this idea to LLMs, yielding a simple yet effective off-policy algorithm that optimizes a single trajectory-level Bellman objective using the model's own logits as $Q$-values. TBRM removes the need for critics, importance-sampling ratios, or clipping, and operates with only one rollout per prompt. We prove convergence to the near-optimal KL-regularized policy from arbitrary off-policy data via an improved change-of-trajectory-measure analysis. Experiments on standard mathematical-reasoning benchmarks show that TBRM consistently outperforms policy-based baselines, like PPO and GRPO, with comparable or lower computational and memory overhead. Our results indicate that value-based RL might be a principled and efficient alternative for enhancing reasoning capabilities in LLMs.
>
---
#### [new 151] An Alternative to FLOPS Regularization to Effectively Productionize SPLADE-Doc
- **分类: cs.IR; cs.CL**

- **简介: 该论文针对LSR模型（如SPLADE）生产部署中因高DF词引发的高延迟问题，提出DF-FLOPS正则化方法，通过惩罚高频词使用缩短倒排列表，实现检索速度提升10倍且保持效果（MRR@10仅降2.2），促进LSR实际应用。**

- **链接: [http://arxiv.org/pdf/2505.15070v1](http://arxiv.org/pdf/2505.15070v1)**

> **作者:** Aldo Porco; Dhruv Mehra; Igor Malioutov; Karthik Radhakrishnan; Moniba Keymanesh; Daniel Preoţiuc-Pietro; Sean MacAvaney; Pengxiang Cheng
>
> **备注:** Accepted as a short paper at SIGIR 2025
>
> **摘要:** Learned Sparse Retrieval (LSR) models encode text as weighted term vectors, which need to be sparse to leverage inverted index structures during retrieval. SPLADE, the most popular LSR model, uses FLOPS regularization to encourage vector sparsity during training. However, FLOPS regularization does not ensure sparsity among terms - only within a given query or document. Terms with very high Document Frequencies (DFs) substantially increase latency in production retrieval engines, such as Apache Solr, due to their lengthy posting lists. To address the issue of high DFs, we present a new variant of FLOPS regularization: DF-FLOPS. This new regularization technique penalizes the usage of high-DF terms, thereby shortening posting lists and reducing retrieval latency. Unlike other inference-time sparsification methods, such as stopword removal, DF-FLOPS regularization allows for the selective inclusion of high-frequency terms in cases where the terms are truly salient. We find that DF-FLOPS successfully reduces the prevalence of high-DF terms and lowers retrieval latency (around 10x faster) in a production-grade engine while maintaining effectiveness both in-domain (only a 2.2-point drop in MRR@10) and cross-domain (improved performance in 12 out of 13 tasks on which we tested). With retrieval latencies on par with BM25, this work provides an important step towards making LSR practical for deployment in production-grade search engines.
>
---
#### [new 152] FisherSFT: Data-Efficient Supervised Fine-Tuning of Language Models Using Information Gain
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于语言模型监督微调（SFT）任务，旨在提升数据效率。针对传统SFT需大量样本的问题，提出FisherSFT方法：通过最大化样本的Hessian对数似然信息增益，利用最后一层线性化近似高效选择关键样本。实验验证了其计算效率与性能优势。**

- **链接: [http://arxiv.org/pdf/2505.14826v1](http://arxiv.org/pdf/2505.14826v1)**

> **作者:** Rohan Deb; Kiran Thekumparampil; Kousha Kalantari; Gaurush Hiranandani; Shoham Sabach; Branislav Kveton
>
> **摘要:** Supervised fine-tuning (SFT) is a standard approach to adapting large language models (LLMs) to new domains. In this work, we improve the statistical efficiency of SFT by selecting an informative subset of training examples. Specifically, for a fixed budget of training examples, which determines the computational cost of fine-tuning, we determine the most informative ones. The key idea in our method is to select examples that maximize information gain, measured by the Hessian of the log-likelihood of the LLM. We approximate it efficiently by linearizing the LLM at the last layer using multinomial logistic regression models. Our approach is computationally efficient, analyzable, and performs well empirically. We demonstrate this on several problems, and back our claims with both quantitative results and an LLM evaluation.
>
---
#### [new 153] HDLxGraph: Bridging Large Language Models and HDL Repositories via HDL Graph Databases
- **分类: cs.AR; cs.CL; cs.LG**

- **简介: 该论文提出HDLxGraph框架，旨在提升大语言模型在大型HDL项目中的性能。针对LLM在复杂硬件设计任务中因代码规模大导致的检索不准确和效率低问题，通过构建AST和DFG的HDL图数据库，结合结构化双检索机制，并创建HDLSearch基准集，显著提升了代码搜索、调试和生成效果。**

- **链接: [http://arxiv.org/pdf/2505.15701v1](http://arxiv.org/pdf/2505.15701v1)**

> **作者:** Pingqing Zheng; Jiayin Qin; Fuqi Zhang; Shang Wu; Yu Cao; Caiwen Ding; Yang; Zhao
>
> **摘要:** Large Language Models (LLMs) have demonstrated their potential in hardware design tasks, such as Hardware Description Language (HDL) generation and debugging. Yet, their performance in real-world, repository-level HDL projects with thousands or even tens of thousands of code lines is hindered. To this end, we propose HDLxGraph, a novel framework that integrates Graph Retrieval Augmented Generation (Graph RAG) with LLMs, introducing HDL-specific graph representations by incorporating Abstract Syntax Trees (ASTs) and Data Flow Graphs (DFGs) to capture both code graph view and hardware graph view. HDLxGraph utilizes a dual-retrieval mechanism that not only mitigates the limited recall issues inherent in similarity-based semantic retrieval by incorporating structural information, but also enhances its extensibility to various real-world tasks by a task-specific retrieval finetuning. Additionally, to address the lack of comprehensive HDL search benchmarks, we introduce HDLSearch, a multi-granularity evaluation dataset derived from real-world repository-level projects. Experimental results demonstrate that HDLxGraph significantly improves average search accuracy, debugging efficiency and completion quality by 12.04%, 12.22% and 5.04% compared to similarity-based RAG, respectively. The code of HDLxGraph and collected HDLSearch benchmark are available at https://github.com/Nick-Zheng-Q/HDLxGraph.
>
---
#### [new 154] Mechanistic Insights into Grokking from the Embedding Layer
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究神经网络Grokking（延迟泛化）机制，聚焦嵌入层作用。任务为解析Grokking驱动因素，解决其延迟泛化问题。发现嵌入层关键作用：其动态更新（稀有token停滞）与双线性耦合（导致鞍点）引发Grokking，提出频率感知采样与自适应学习率策略（基于曲率比）加速收敛，扩展至Transformer优化。**

- **链接: [http://arxiv.org/pdf/2505.15624v1](http://arxiv.org/pdf/2505.15624v1)**

> **作者:** H. V. AlquBoj; Hilal AlQuabeh; Velibor Bojkovic; Munachiso Nwadike; Kentaro Inui
>
> **备注:** Mechanistic view of embedding layers
>
> **摘要:** Grokking, a delayed generalization in neural networks after perfect training performance, has been observed in Transformers and MLPs, but the components driving it remain underexplored. We show that embeddings are central to grokking: introducing them into MLPs induces delayed generalization in modular arithmetic tasks, whereas MLPs without embeddings can generalize immediately. Our analysis identifies two key mechanisms: (1) Embedding update dynamics, where rare tokens stagnate due to sparse gradient updates and weight decay, and (2) Bilinear coupling, where the interaction between embeddings and downstream weights introduces saddle points and increases sensitivity to initialization. To confirm these mechanisms, we investigate frequency-aware sampling, which balances token updates by minimizing gradient variance, and embedding-specific learning rates, derived from the asymmetric curvature of the bilinear loss landscape. We prove that an adaptive learning rate ratio, \(\frac{\eta_E}{\eta_W} \propto \frac{\sigma_{\max}(E)}{\sigma_{\max}(W)} \cdot \frac{f_W}{f_E}\), mitigates bilinear coupling effects, accelerating convergence. Our methods not only improve grokking dynamics but also extend to broader challenges in Transformer optimization, where bilinear interactions hinder efficient training.
>
---
#### [new 155] RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于语言推理任务，旨在解决现有强化学习方法中验证器固定或监督微调导致的奖励欺骗和泛化差问题。提出Tango框架，通过强化学习协同训练生成器与验证器，验证器采用生成式过程级模型，仅依赖结果奖励训练，提升鲁棒性和泛化能力，在数学推理等任务中表现最优。**

- **链接: [http://arxiv.org/pdf/2505.15034v1](http://arxiv.org/pdf/2505.15034v1)**

> **作者:** Kaiwen Zha; Zhengqi Gao; Maohao Shen; Zhang-Wei Hong; Duane S. Boning; Dina Katabi
>
> **备注:** Tech report. The first two authors contributed equally
>
> **摘要:** Reinforcement learning (RL) has recently emerged as a compelling approach for enhancing the reasoning capabilities of large language models (LLMs), where an LLM generator serves as a policy guided by a verifier (reward model). However, current RL post-training methods for LLMs typically use verifiers that are fixed (rule-based or frozen pretrained) or trained discriminatively via supervised fine-tuning (SFT). Such designs are susceptible to reward hacking and generalize poorly beyond their training distributions. To overcome these limitations, we propose Tango, a novel framework that uses RL to concurrently train both an LLM generator and a verifier in an interleaved manner. A central innovation of Tango is its generative, process-level LLM verifier, which is trained via RL and co-evolves with the generator. Importantly, the verifier is trained solely based on outcome-level verification correctness rewards without requiring explicit process-level annotations. This generative RL-trained verifier exhibits improved robustness and superior generalization compared to deterministic or SFT-trained verifiers, fostering effective mutual reinforcement with the generator. Extensive experiments demonstrate that both components of Tango achieve state-of-the-art results among 7B/8B-scale models: the generator attains best-in-class performance across five competition-level math benchmarks and four challenging out-of-domain reasoning tasks, while the verifier leads on the ProcessBench dataset. Remarkably, both components exhibit particularly substantial improvements on the most difficult mathematical reasoning problems. Code is at: https://github.com/kaiwenzha/rl-tango.
>
---
#### [new 156] Better Safe Than Sorry? Overreaction Problem of Vision Language Models in Visual Emergency Recognition
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究视觉语言模型（VLM）在紧急场景识别中的可靠性，旨在解决其过度反应问题。构建VERI数据集（200张对比图像），评估14种VLM模型，发现其虽能有效识别真实紧急情况（70-100%成功率），但误报率达31-96%，主要因上下文过度解读。结果表明模型规模无法缓解此问题，需改进安全评估方法。**

- **链接: [http://arxiv.org/pdf/2505.15367v1](http://arxiv.org/pdf/2505.15367v1)**

> **作者:** Dasol Choi; Seunghyun Lee; Youngsook Song
>
> **备注:** 13 pages
>
> **摘要:** Vision-Language Models (VLMs) have demonstrated impressive capabilities in understanding visual content, but their reliability in safety-critical contexts remains under-explored. We introduce VERI (Visual Emergency Recognition Dataset), a carefully designed diagnostic benchmark of 200 images (100 contrastive pairs). Each emergency scene is matched with a visually similar but safe counterpart through multi-stage human verification and iterative refinement. Using a two-stage protocol - risk identification and emergency response - we evaluate 14 VLMs (2B-124B parameters) across medical emergencies, accidents, and natural disasters. Our analysis reveals a systematic overreaction problem: models excel at identifying real emergencies (70-100 percent success rate) but suffer from an alarming rate of false alarms, misidentifying 31-96 percent of safe situations as dangerous, with 10 scenarios failed by all models regardless of scale. This "better-safe-than-sorry" bias manifests primarily through contextual overinterpretation (88-93 percent of errors), challenging VLMs' reliability for safety applications. These findings highlight persistent limitations that are not resolved by increasing model scale, motivating targeted approaches for improving contextual safety assessment in visually misleading scenarios.
>
---
#### [new 157] Seeing Through Deception: Uncovering Misleading Creator Intent in Multimodal News with Vision-Language Models
- **分类: cs.CV; cs.CL; cs.MM**

- **简介: 该论文属于多模态虚假信息检测任务，旨在解决现有视觉语言模型（VLMs）难以识别误导性创作者意图的问题。提出通过模拟新闻创作过程构建含12000样本的DeceptionDecoded数据集，评估14种模型在检测误导意图、归因来源及推理创作者意图上的表现，发现其依赖表面线索，强调需开发意图感知模型以提升深度推理能力。**

- **链接: [http://arxiv.org/pdf/2505.15489v1](http://arxiv.org/pdf/2505.15489v1)**

> **作者:** Jiaying Wu; Fanxiao Li; Min-Yen Kan; Bryan Hooi
>
> **摘要:** The real-world impact of misinformation stems from the underlying misleading narratives that creators seek to convey. As such, interpreting misleading creator intent is essential for multimodal misinformation detection (MMD) systems aimed at effective information governance. In this paper, we introduce an automated framework that simulates real-world multimodal news creation by explicitly modeling creator intent through two components: the desired influence and the execution plan. Using this framework, we construct DeceptionDecoded, a large-scale benchmark comprising 12,000 image-caption pairs aligned with trustworthy reference articles. The dataset captures both misleading and non-misleading intents and spans manipulations across visual and textual modalities. We conduct a comprehensive evaluation of 14 state-of-the-art vision-language models (VLMs) on three intent-centric tasks: (1) misleading intent detection, (2) misleading source attribution, and (3) creator desire inference. Despite recent advances, we observe that current VLMs fall short in recognizing misleading intent, often relying on spurious cues such as superficial cross-modal consistency, stylistic signals, and heuristic authenticity hints. Our findings highlight the pressing need for intent-aware modeling in MMD and open new directions for developing systems capable of deeper reasoning about multimodal misinformation.
>
---
#### [new 158] ModelingAgent: Bridging LLMs and Mathematical Modeling for Real-World Challenges
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ModelingAgent框架，结合LLMs与数学建模解决现实问题。针对现有基准无法反映开放性、跨学科挑战的问题，构建ModelingBench基准库，涵盖多领域实际问题，并设计多智能体系统协调工具、迭代优化，以及专家评估系统ModelingJudge。实验显示其性能显著优于基线，接近人类专家水平。**

- **链接: [http://arxiv.org/pdf/2505.15068v1](http://arxiv.org/pdf/2505.15068v1)**

> **作者:** Cheng Qian; Hongyi Du; Hongru Wang; Xiusi Chen; Yuji Zhang; Avirup Sil; Chengxiang Zhai; Kathleen McKeown; Heng Ji
>
> **备注:** 36 Pages, 26 Figures, 5 Tables
>
> **摘要:** Recent progress in large language models (LLMs) has enabled substantial advances in solving mathematical problems. However, existing benchmarks often fail to reflect the complexity of real-world problems, which demand open-ended, interdisciplinary reasoning and integration of computational tools. To address this gap, we introduce ModelingBench, a novel benchmark featuring real-world-inspired, open-ended problems from math modeling competitions across diverse domains, ranging from urban traffic optimization to ecosystem resource planning. These tasks require translating natural language into formal mathematical formulations, applying appropriate tools, and producing structured, defensible reports. ModelingBench also supports multiple valid solutions, capturing the ambiguity and creativity of practical modeling. We also present ModelingAgent, a multi-agent framework that coordinates tool use, supports structured workflows, and enables iterative self-refinement to generate well-grounded, creative solutions. To evaluate outputs, we further propose ModelingJudge, an expert-in-the-loop system leveraging LLMs as domain-specialized judges assessing solutions from multiple expert perspectives. Empirical results show that ModelingAgent substantially outperforms strong baselines and often produces solutions indistinguishable from those of human experts. Together, our work provides a comprehensive framework for evaluating and advancing real-world problem-solving in open-ended, interdisciplinary modeling challenges.
>
---
#### [new 159] When Can Large Reasoning Models Save Thinking? Mechanistic Analysis of Behavioral Divergence in Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究强化学习优化的大型推理模型（LRMs）在节省思考时的内部机制，旨在解决其过度思考导致的低效问题。通过分析三种推理模式（无思考、显式/隐式思考），揭示模型在终止信心、注意力分配上的差异，发现无思考虽缩短输出但牺牲准确率，另两种模式平衡了效率与精度，指出需改进模型的可靠性。**

- **链接: [http://arxiv.org/pdf/2505.15276v1](http://arxiv.org/pdf/2505.15276v1)**

> **作者:** Rongzhi Zhu; Yi Liu; Zequn Sun; Yiwei Wang; Wei Hu
>
> **摘要:** Large reasoning models (LRMs) have significantly advanced performance on complex tasks, yet their tendency to overthink introduces inefficiencies. This study investigates the internal mechanisms of reinforcement learning (RL)-trained LRMs when prompted to save thinking, revealing three distinct thinking modes: no thinking (NT), explicit thinking (ET), and implicit thinking (IT). Through comprehensive analysis of confidence in thinking termination, attention from thinking to generation, and attentional focus on input sections, we uncover key factors influencing the reasoning behaviors. We further find that NT reduces output length at the cost of accuracy, while ET and IT maintain accuracy with reduced response length. Our findings expose fundamental inconsistencies in RL-optimized LRMs, necessitating adaptive improvements for reliable efficiency.
>
---
#### [new 160] MoTime: A Dataset Suite for Multimodal Time Series Forecasting
- **分类: cs.LG; cs.CL; cs.DB; cs.IR**

- **简介: 该论文提出多模态时间序列预测数据集MoTime，解决现有研究依赖单模态数据的问题。覆盖多领域，整合文本、图像等模态，支持常规预测和冷启动场景评估。实验表明多模态提升预测效果，尤其对短序列有效。**

- **链接: [http://arxiv.org/pdf/2505.15072v1](http://arxiv.org/pdf/2505.15072v1)**

> **作者:** Xin Zhou; Weiqing Wang; Francisco J. Baldán; Wray Buntine; Christoph Bergmeir
>
> **摘要:** While multimodal data sources are increasingly available from real-world forecasting, most existing research remains on unimodal time series. In this work, we present MoTime, a suite of multimodal time series forecasting datasets that pair temporal signals with external modalities such as text, metadata, and images. Covering diverse domains, MoTime supports structured evaluation of modality utility under two scenarios: 1) the common forecasting task, where varying-length history is available, and 2) cold-start forecasting, where no historical data is available. Experiments show that external modalities can improve forecasting performance in both scenarios, with particularly strong benefits for short series in some datasets, though the impact varies depending on data characteristics. By making datasets and findings publicly available, we aim to support more comprehensive and realistic benchmarks in future multimodal time series forecasting research.
>
---
#### [new 161] Sentiment Analysis in Software Engineering: Evaluating Generative Pre-trained Transformers
- **分类: cs.SE; cs.CL; cs.LG**

- **简介: 该论文属于软件工程领域情感分析任务，旨在解决传统工具对领域内复杂语境与不平衡数据处理不足的问题。通过对比BERT与GPT-4o-mini在GitHub、Stack Overflow和Jira数据集上的性能，发现GPT-4o-mini默认配置在复杂数据中表现更优，强调需根据数据特性选择模型架构。**

- **链接: [http://arxiv.org/pdf/2505.14692v1](http://arxiv.org/pdf/2505.14692v1)**

> **作者:** KM Khalid Saifullah; Faiaz Azmain; Habiba Hye
>
> **摘要:** Sentiment analysis plays a crucial role in understanding developer interactions, issue resolutions, and project dynamics within software engineering (SE). While traditional SE-specific sentiment analysis tools have made significant strides, they often fail to account for the nuanced and context-dependent language inherent to the domain. This study systematically evaluates the performance of bidirectional transformers, such as BERT, against generative pre-trained transformers, specifically GPT-4o-mini, in SE sentiment analysis. Using datasets from GitHub, Stack Overflow, and Jira, we benchmark the models' capabilities with fine-tuned and default configurations. The results reveal that fine-tuned GPT-4o-mini performs comparable to BERT and other bidirectional models on structured and balanced datasets like GitHub and Jira, achieving macro-averaged F1-scores of 0.93 and 0.98, respectively. However, on linguistically complex datasets with imbalanced sentiment distributions, such as Stack Overflow, the default GPT-4o-mini model exhibits superior generalization, achieving an accuracy of 85.3\% compared to the fine-tuned model's 13.1\%. These findings highlight the trade-offs between fine-tuning and leveraging pre-trained models for SE tasks. The study underscores the importance of aligning model architectures with dataset characteristics to optimize performance and proposes directions for future research in refining sentiment analysis tools tailored to the SE domain.
>
---
#### [new 162] MORALISE: A Structured Benchmark for Moral Alignment in Visual Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.CY; cs.MM**

- **简介: 该论文提出MORALISE基准，评估视觉语言模型的道德对齐问题。针对现有方法依赖文本或AI生成数据导致的偏差，其基于Turiel理论构建13类道德主题，收集2481个真实图像文本对，设计道德判断与归因任务，测试19个模型，揭示当前技术的道德局限。**

- **链接: [http://arxiv.org/pdf/2505.14728v1](http://arxiv.org/pdf/2505.14728v1)**

> **作者:** Xiao Lin; Zhining Liu; Ze Yang; Gaotang Li; Ruizhong Qiu; Shuke Wang; Hui Liu; Haotian Li; Sumit Keswani; Vishwa Pardeshi; Huijun Zhao; Wei Fan; Hanghang Tong
>
> **备注:** 21 pages, 11 figures, 7 tables
>
> **摘要:** Warning: This paper contains examples of harmful language and images. Reader discretion is advised. Recently, vision-language models have demonstrated increasing influence in morally sensitive domains such as autonomous driving and medical analysis, owing to their powerful multimodal reasoning capabilities. As these models are deployed in high-stakes real-world applications, it is of paramount importance to ensure that their outputs align with human moral values and remain within moral boundaries. However, existing work on moral alignment either focuses solely on textual modalities or relies heavily on AI-generated images, leading to distributional biases and reduced realism. To overcome these limitations, we introduce MORALISE, a comprehensive benchmark for evaluating the moral alignment of vision-language models (VLMs) using diverse, expert-verified real-world data. We begin by proposing a comprehensive taxonomy of 13 moral topics grounded in Turiel's Domain Theory, spanning the personal, interpersonal, and societal moral domains encountered in everyday life. Built on this framework, we manually curate 2,481 high-quality image-text pairs, each annotated with two fine-grained labels: (1) topic annotation, identifying the violated moral topic(s), and (2) modality annotation, indicating whether the violation arises from the image or the text. For evaluation, we encompass two tasks, \textit{moral judgment} and \textit{moral norm attribution}, to assess models' awareness of moral violations and their reasoning ability on morally salient content. Extensive experiments on 19 popular open- and closed-source VLMs show that MORALISE poses a significant challenge, revealing persistent moral limitations in current state-of-the-art models. The full benchmark is publicly available at https://huggingface.co/datasets/Ze1025/MORALISE.
>
---
#### [new 163] ALN-P3: Unified Language Alignment for Perception, Prediction, and Planning in Autonomous Driving
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出ALN-P3框架，解决自动驾驶中视觉系统与语言模型难以兼顾驾驶性能与语言推理的问题。通过感知、预测、规划三阶段的跨模态对齐机制，在训练阶段融合视觉与语言模块，提升决策与推理能力，实验显示其效果最优。**

- **链接: [http://arxiv.org/pdf/2505.15158v1](http://arxiv.org/pdf/2505.15158v1)**

> **作者:** Yunsheng Ma; Burhaneddin Yaman; Xin Ye; Mahmut Yurt; Jingru Luo; Abhirup Mallik; Ziran Wang; Liu Ren
>
> **备注:** 10 pages
>
> **摘要:** Recent advances have explored integrating large language models (LLMs) into end-to-end autonomous driving systems to enhance generalization and interpretability. However, most existing approaches are limited to either driving performance or vision-language reasoning, making it difficult to achieve both simultaneously. In this paper, we propose ALN-P3, a unified co-distillation framework that introduces cross-modal alignment between "fast" vision-based autonomous driving systems and "slow" language-driven reasoning modules. ALN-P3 incorporates three novel alignment mechanisms: Perception Alignment (P1A), Prediction Alignment (P2A), and Planning Alignment (P3A), which explicitly align visual tokens with corresponding linguistic outputs across the full perception, prediction, and planning stack. All alignment modules are applied only during training and incur no additional costs during inference. Extensive experiments on four challenging benchmarks-nuScenes, Nu-X, TOD3Cap, and nuScenes QA-demonstrate that ALN-P3 significantly improves both driving decisions and language reasoning, achieving state-of-the-art results.
>
---
#### [new 164] Benchmarking Graph Neural Networks for Document Layout Analysis in Public Affairs
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文针对数字PDF文档中异构元素与元数据不精确导致的布局分析难题，通过构建k近邻图/全连接图，结合预训练文本-视觉特征，测试单模态/多模态GNN模型，验证GraphSAGE在双分支k近邻图配置下效果最佳，证明局部布局关系与多模态融合的重要性。（99字）**

- **链接: [http://arxiv.org/pdf/2505.14699v1](http://arxiv.org/pdf/2505.14699v1)**

> **作者:** Miguel Lopez-Duran; Julian Fierrez; Aythami Morales; Ruben Tolosana; Oscar Delgado-Mohatar; Alvaro Ortigosa
>
> **备注:** 15 pages, 2 figures, preprint presented in The Fifth ICDAR International Workshop on Machine Learning
>
> **摘要:** The automatic analysis of document layouts in digital-born PDF documents remains a challenging problem due to the heterogeneous arrangement of textual and nontextual elements and the imprecision of the textual metadata in the Portable Document Format. In this work, we benchmark Graph Neural Network (GNN) architectures for the task of fine-grained layout classification of text blocks from digital native documents. We introduce two graph construction structures: a k-closest-neighbor graph and a fully connected graph, and generate node features via pre-trained text and vision models, thus avoiding manual feature engineering. Three experimental frameworks are evaluated: single-modality (text or visual), concatenated multimodal, and dual-branch multimodal. We evaluated four foundational GNN models and compared them with the baseline. Our experiments are specifically conducted on a rich dataset of public affairs documents that includes more than 20 sources (e.g., regional and national-level official gazettes), 37K PDF documents, with 441K pages in total. Our results demonstrate that GraphSAGE operating on the k-closest-neighbor graph in a dual-branch configuration achieves the highest per-class and overall accuracy, outperforming the baseline in some sources. These findings confirm the importance of local layout relationships and multimodal fusion exploited through GNNs for the analysis of native digital document layouts.
>
---
#### [new 165] When to Continue Thinking: Adaptive Thinking Mode Switching for Efficient Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型推理优化任务，旨在解决长推理链导致的计算冗余问题。通过发现模型的"内部自恢复机制"，提出自适应自恢复推理框架ASRR，动态调整推理长度，减少冗余计算同时保持性能。实验显示其计算效率提升超30%，安全率显著提高。**

- **链接: [http://arxiv.org/pdf/2505.15400v1](http://arxiv.org/pdf/2505.15400v1)**

> **作者:** Xiaoyun Zhang; Jingqing Ruan; Xing Ma; Yawen Zhu; Haodong Zhao; Hao Li; Jiansong Chen; Ke Zeng; Xunliang Cai
>
> **摘要:** Large reasoning models (LRMs) achieve remarkable performance via long reasoning chains, but often incur excessive computational overhead due to redundant reasoning, especially on simple tasks. In this work, we systematically quantify the upper bounds of LRMs under both Long-Thinking and No-Thinking modes, and uncover the phenomenon of "Internal Self-Recovery Mechanism" where models implicitly supplement reasoning during answer generation. Building on this insight, we propose Adaptive Self-Recovery Reasoning (ASRR), a framework that suppresses unnecessary reasoning and enables implicit recovery. By introducing accuracy-aware length reward regulation, ASRR adaptively allocates reasoning effort according to problem difficulty, achieving high efficiency with negligible performance sacrifice. Experiments across multiple benchmarks and models show that, compared with GRPO, ASRR reduces reasoning budget by up to 32.5% (1.5B) and 25.7% (7B) with minimal accuracy loss (1.2% and 0.6% pass@1), and significantly boosts harmless rates on safety benchmarks (up to +21.7%). Our results highlight the potential of ASRR for enabling efficient, adaptive, and safer reasoning in LRMs.
>
---
#### [new 166] A Participatory Strategy for AI Ethics in Education and Rehabilitation grounded in the Capability Approach
- **分类: cs.CY; cs.CL**

- **简介: 该论文提出基于能力方法的AI伦理参与式策略，旨在解决教育与康复领域中AI应用的伦理与有效性问题。通过ARTIS项目案例，整合多方专家开展焦点小组与协作设计，开发支持阅读障碍儿童的AI界面，弥合技术创新与伦理责任的差距。**

- **链接: [http://arxiv.org/pdf/2505.15466v1](http://arxiv.org/pdf/2505.15466v1)**

> **作者:** Valeria Cesaroni; Eleonora Pasqua; Piercosma Bisconti; Martina Galletti
>
> **摘要:** AI-based technologies have significant potential to enhance inclusive education and clinical-rehabilitative contexts for children with Special Educational Needs and Disabilities. AI can enhance learning experiences, empower students, and support both teachers and rehabilitators. However, their usage presents challenges that require a systemic-ecological vision, ethical considerations, and participatory research. Therefore, research and technological development must be rooted in a strong ethical-theoretical framework. The Capability Approach - a theoretical model of disability, human vulnerability, and inclusion - offers a more relevant perspective on functionality, effectiveness, and technological adequacy in inclusive learning environments. In this paper, we propose a participatory research strategy with different stakeholders through a case study on the ARTIS Project, which develops an AI-enriched interface to support children with text comprehension difficulties. Our research strategy integrates ethical, educational, clinical, and technological expertise in designing and implementing AI-based technologies for children's learning environments through focus groups and collaborative design sessions. We believe that this holistic approach to AI adoption in education can help bridge the gap between technological innovation and ethical responsibility.
>
---
#### [new 167] ClickSight: Interpreting Student Clickstreams to Reveal Insights on Learning Strategies via LLMs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于教育数据挖掘任务，针对传统方法难以有效解析高维学生点击流数据以揭示学习策略的问题，提出ClickSight框架：利用LLM处理原始点击流和学习策略列表，生成行为解释，并评估了四种提示策略及自优化效果，发现LLM潜力与策略差异，为教育数据分析提供可扩展方案。**

- **链接: [http://arxiv.org/pdf/2505.15410v1](http://arxiv.org/pdf/2505.15410v1)**

> **作者:** Bahar Radmehr; Ekaterina Shved; Fatma Betül Güreş; Adish Singla; Tanja Käser
>
> **备注:** Accepted in Latebreaking results track in AIED 2025(26th International Conference on Artificial Intelligence in Education JULY 22-26, 2025 PALERMO, ITALY)
>
> **摘要:** Clickstream data from digital learning environments offer valuable insights into students' learning behaviors, but are challenging to interpret due to their high dimensionality and granularity. Prior approaches have relied mainly on handcrafted features, expert labeling, clustering, or supervised models, therefore often lacking generalizability and scalability. In this work, we introduce ClickSight, an in-context Large Language Model (LLM)-based pipeline that interprets student clickstreams to reveal their learning strategies. ClickSight takes raw clickstreams and a list of learning strategies as input and generates textual interpretations of students' behaviors during interaction. We evaluate four different prompting strategies and investigate the impact of self-refinement on interpretation quality. Our evaluation spans two open-ended learning environments and uses a rubric-based domain-expert evaluation. Results show that while LLMs can reasonably interpret learning strategies from clickstreams, interpretation quality varies by prompting strategy, and self-refinement offers limited improvement. ClickSight demonstrates the potential of LLMs to generate theory-driven insights from educational interaction data.
>
---
#### [new 168] Set-LLM: A Permutation-Invariant LLM
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Set-LLM，通过引入新注意力掩码和位置编码，使预训练语言模型具备处理无序集合数据的排列不变性，解决模型因输入顺序导致的偏差问题，实验验证其有效且不增加运行时间。**

- **链接: [http://arxiv.org/pdf/2505.15433v1](http://arxiv.org/pdf/2505.15433v1)**

> **作者:** Beni Egressy; Jan Stühmer
>
> **摘要:** While large language models (LLMs) demonstrate impressive capabilities across numerous applications, their robustness remains a critical concern. This paper is motivated by a specific vulnerability: the order sensitivity of LLMs. This vulnerability manifests itself as the order bias observed when LLMs decide between possible options (for example, a preference for the first option) and the tendency of LLMs to provide different answers when options are reordered. The use cases for this scenario extend beyond the classical case of multiple-choice question answering to the use of LLMs as automated evaluators in AI pipelines, comparing output generated by different models. We introduce Set-LLM, a novel architectural adaptation for pretrained LLMs that enables the processing of mixed set-text inputs with permutation invariance guarantees. The adaptations involve a new attention mask and new positional encodings specifically designed for sets. We provide a theoretical proof of invariance and demonstrate through experiments that Set-LLM can be trained effectively, achieving comparable or improved performance and maintaining the runtime of the original model, while eliminating order sensitivity.
>
---
#### [new 169] AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出AgentThink框架，解决视觉语言模型在自动驾驶中的幻觉、推理低效及现实验证不足问题。通过构建自动驾驶工具库生成结构化数据、两阶段训练（SFT+GRPO）提升工具调用能力，及多工具评估协议，显著提升推理准确性和一致性。**

- **链接: [http://arxiv.org/pdf/2505.15298v1](http://arxiv.org/pdf/2505.15298v1)**

> **作者:** Kangan Qian; Sicong Jiang; Yang Zhong; Ziang Luo; Zilin Huang; Tianze Zhu; Kun Jiang; Mengmeng Yang; Zheng Fu; Jinyu Miao; Yining Shi; He Zhe Lim; Li Liu; Tianbao Zhou; Hongyi Wang; Huang Yu; Yifei Hu; Guang Li; Guang Chen; Hao Ye; Lijun Sun; Diange Yang
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Vision-Language Models (VLMs) show promise for autonomous driving, yet their struggle with hallucinations, inefficient reasoning, and limited real-world validation hinders accurate perception and robust step-by-step reasoning. To overcome this, we introduce \textbf{AgentThink}, a pioneering unified framework that, for the first time, integrates Chain-of-Thought (CoT) reasoning with dynamic, agent-style tool invocation for autonomous driving tasks. AgentThink's core innovations include: \textbf{(i) Structured Data Generation}, by establishing an autonomous driving tool library to automatically construct structured, self-verified reasoning data explicitly incorporating tool usage for diverse driving scenarios; \textbf{(ii) A Two-stage Training Pipeline}, employing Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to equip VLMs with the capability for autonomous tool invocation; and \textbf{(iii) Agent-style Tool-Usage Evaluation}, introducing a novel multi-tool assessment protocol to rigorously evaluate the model's tool invocation and utilization. Experiments on the DriveLMM-o1 benchmark demonstrate AgentThink significantly boosts overall reasoning scores by \textbf{53.91\%} and enhances answer accuracy by \textbf{33.54\%}, while markedly improving reasoning quality and consistency. Furthermore, ablation studies and robust zero-shot/few-shot generalization experiments across various benchmarks underscore its powerful capabilities. These findings highlight a promising trajectory for developing trustworthy and tool-aware autonomous driving models.
>
---
#### [new 170] Learning to Rank Chain-of-Thought: An Energy-Based Approach with Outcome Supervision
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文针对大语言模型数学推理中Chain of Thought（CoT）步骤不可靠及高计算成本问题，提出轻量级Energy Outcome Reward Model（EORM）。利用能量模型通过结果标签对CoT解排序，提升推理准确性（如GSM8k达90.7%），高效验证优于暴力采样。**

- **链接: [http://arxiv.org/pdf/2505.14999v1](http://arxiv.org/pdf/2505.14999v1)**

> **作者:** Eric Hanchen Jiang; Haozheng Luo; Shengyuan Pang; Xiaomin Li; Zhenting Qi; Hengli Li; Cheng-Fu Yang; Zongyu Lin; Xinfeng Li; Hao Xu; Kai-Wei Chang; Ying Nian Wu
>
> **摘要:** Mathematical reasoning presents a significant challenge for Large Language Models (LLMs), often requiring robust multi step logical consistency. While Chain of Thought (CoT) prompting elicits reasoning steps, it doesn't guarantee correctness, and improving reliability via extensive sampling is computationally costly. This paper introduces the Energy Outcome Reward Model (EORM), an effective, lightweight, post hoc verifier. EORM leverages Energy Based Models (EBMs) to simplify the training of reward models by learning to assign a scalar energy score to CoT solutions using only outcome labels, thereby avoiding detailed annotations. It achieves this by interpreting discriminator output logits as negative energies, effectively ranking candidates where lower energy is assigned to solutions leading to correct final outcomes implicitly favoring coherent reasoning. On mathematical benchmarks (GSM8k, MATH), EORM significantly improves final answer accuracy (e.g., with Llama 3 8B, achieving 90.7% on GSM8k and 63.7% on MATH). EORM effectively leverages a given pool of candidate solutions to match or exceed the performance of brute force sampling, thereby enhancing LLM reasoning outcome reliability through its streamlined post hoc verification process.
>
---
#### [new 171] MIRB: Mathematical Information Retrieval Benchmark
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文提出MIRB基准，解决数学信息检索（MIR）缺乏统一评估标准的问题。包含四个任务及12个数据集，评估13种模型，分析挑战，旨在为MIR系统提供全面评估框架，推动领域发展。**

- **链接: [http://arxiv.org/pdf/2505.15585v1](http://arxiv.org/pdf/2505.15585v1)**

> **作者:** Haocheng Ju; Bin Dong
>
> **备注:** Our code and data are available at https://github.com/j991222/mirb and https://huggingface.co/collections/hcju/mirb-6827001711765454f58c5a76
>
> **摘要:** Mathematical Information Retrieval (MIR) is the task of retrieving information from mathematical documents and plays a key role in various applications, including theorem search in mathematical libraries, answer retrieval on math forums, and premise selection in automated theorem proving. However, a unified benchmark for evaluating these diverse retrieval tasks has been lacking. In this paper, we introduce MIRB (Mathematical Information Retrieval Benchmark) to assess the MIR capabilities of retrieval models. MIRB includes four tasks: semantic statement retrieval, question-answer retrieval, premise retrieval, and formula retrieval, spanning a total of 12 datasets. We evaluate 13 retrieval models on this benchmark and analyze the challenges inherent to MIR. We hope that MIRB provides a comprehensive framework for evaluating MIR systems and helps advance the development of more effective retrieval models tailored to the mathematical domain.
>
---
#### [new 172] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态思维链（MCoT）机制，旨在解析其提升视觉语言模型性能的原理。针对现有MCoT文本型与交错型方法，提出"视觉思维"概念，分析其四种表达形式对推理效果的影响，并揭示其作为图像与深层模型间的中介作用，为MCoT优化提供理论依据。（99字）**

- **链接: [http://arxiv.org/pdf/2505.15510v1](http://arxiv.org/pdf/2505.15510v1)**

> **作者:** Zihui Cheng; Qiguang Chen; Xiao Xu; Jiaqi Wang; Weiyun Wang; Hao Fei; Yidong Wang; Alex Jinpeng Wang; Zhi Chen; Wanxiang Che; Libo Qin
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved significant success in multimodal tasks, with multimodal chain-of-thought (MCoT) further enhancing performance and interpretability. Recent MCoT methods fall into two categories: (i) Textual-MCoT (T-MCoT), which takes multimodal input and produces textual output; and (ii) Interleaved-MCoT (I-MCoT), which generates interleaved image-text outputs. Despite advances in both approaches, the mechanisms driving these improvements are not fully understood. To fill this gap, we first reveal that MCoT boosts LVLMs by incorporating visual thoughts, which convey image information to the reasoning process regardless of the MCoT format, depending only on clarity and conciseness of expression. Furthermore, to explore visual thoughts systematically, we define four distinct forms of visual thought expressions and analyze them comprehensively. Our findings demonstrate that these forms differ in clarity and conciseness, yielding varying levels of MCoT improvement. Additionally, we explore the internal nature of visual thoughts, finding that visual thoughts serve as intermediaries between the input image and reasoning to deeper transformer layers, enabling more advanced visual information transmission. We hope that the visual thoughts can inspire further breakthroughs for future MCoT research.
>
---
#### [new 173] ReGUIDE: Data Efficient GUI Grounding via Spatial Reasoning and Search
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于GUI定位任务，解决多模态大模型在少量数据下精准定位界面元素的挑战。提出ReGUIDE框架，通过在线强化学习生成推理过程、空间先验约束预测，并结合测试时的空间搜索与坐标聚合策略，在仅用0.2%训练数据下超越基线模型。**

- **链接: [http://arxiv.org/pdf/2505.15259v1](http://arxiv.org/pdf/2505.15259v1)**

> **作者:** Hyunseok Lee; Jeonghoon Kim; Beomjun Kim; Jihoon Tack; Chansong Jo; Jaehong Lee; Cheonbok Park; Sookyo In; Jinwoo Shin; Kang Min Yoo
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have enabled autonomous agents to interact with computers via Graphical User Interfaces (GUIs), where accurately localizing the coordinates of interface elements (e.g., buttons) is often required for fine-grained actions. However, this remains significantly challenging, leading prior works to rely on large-scale web datasets to improve the grounding accuracy. In this work, we propose Reasoning Graphical User Interface Grounding for Data Efficiency (ReGUIDE), a novel and effective framework for web grounding that enables MLLMs to learn data efficiently through self-generated reasoning and spatial-aware criticism. More specifically, ReGUIDE learns to (i) self-generate a language reasoning process for the localization via online reinforcement learning, and (ii) criticize the prediction using spatial priors that enforce equivariance under input transformations. At inference time, ReGUIDE further boosts performance through a test-time scaling strategy, which combines spatial search with coordinate aggregation. Our experiments demonstrate that ReGUIDE significantly advances web grounding performance across multiple benchmarks, outperforming baselines with substantially fewer training data points (e.g., only 0.2% samples compared to the best open-sourced baselines).
>
---
#### [new 174] Robo2VLM: Visual Question Answering from Large-Scale In-the-Wild Robot Manipulation Datasets
- **分类: cs.RO; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出Robo2VLM框架，利用真实机器人操作轨迹数据生成视觉问题回答（VQA）数据集，解决VLM在真实场景中空间与交互推理能力不足的问题。通过分析机器人传感器数据（如末端姿态、力反馈）分割操作阶段，生成基于3D场景理解的多选问题，构建含68万问题的Robo2VLM-1数据集，用于评估和提升VLM性能。**

- **链接: [http://arxiv.org/pdf/2505.15517v1](http://arxiv.org/pdf/2505.15517v1)**

> **作者:** Kaiyuan Chen; Shuangyu Xie; Zehan Ma; Ken Goldberg
>
> **摘要:** Vision-Language Models (VLMs) acquire real-world knowledge and general reasoning ability through Internet-scale image-text corpora. They can augment robotic systems with scene understanding and task planning, and assist visuomotor policies that are trained on robot trajectory data. We explore the reverse paradigm - using rich, real, multi-modal robot trajectory data to enhance and evaluate VLMs. In this paper, we present Robo2VLM, a Visual Question Answering (VQA) dataset generation framework for VLMs. Given a human tele-operated robot trajectory, Robo2VLM derives ground-truth from non-visual and non-descriptive sensory modalities, such as end-effector pose, gripper aperture, and force sensing. Based on these modalities, it segments the robot trajectory into a sequence of manipulation phases. At each phase, Robo2VLM uses scene and interaction understanding to identify 3D properties of the robot, task goal, and the target object. The properties are used to generate representative VQA queries - images with textural multiple-choice questions - based on spatial, goal-conditioned, and interaction reasoning question templates. We curate Robo2VLM-1, a large-scale in-the-wild dataset with 684,710 questions covering 463 distinct scenes and 3,396 robotic manipulation tasks from 176k real robot trajectories. Results suggest that Robo2VLM-1 can benchmark and improve VLM capabilities in spatial and interaction reasoning.
>
---
#### [new 175] Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文评估LLM防御的鲁棒性，针对知情攻击者（利用对齐过程信息）设计攻击方法。提出基于中间模型检查点优化GCG攻击初始化，有效突破现有对齐防御，发现通用对抗后缀存在，揭示当前防御脆弱性，强调需采用更强威胁模型测试安全。**

- **链接: [http://arxiv.org/pdf/2505.15738v1](http://arxiv.org/pdf/2505.15738v1)**

> **作者:** Xiaoxue Yang; Bozhidar Stevanoski; Matthieu Meeus; Yves-Alexandre de Montjoye
>
> **摘要:** Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs.
>
---
#### [new 176] Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于LLM安全防护任务，针对现有防御无法应对动态越狱攻击的问题，提出Safety Context Retrieval（SCR）方法。通过检索增强生成技术结合安全示例库，动态增强模型对已知及新兴攻击的防御能力，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.15753v1](http://arxiv.org/pdf/2505.15753v1)**

> **作者:** Taiye Chen; Zeming Wei; Ang Li; Yisen Wang
>
> **摘要:** Large Language Models (LLMs) are known to be vulnerable to jailbreaking attacks, wherein adversaries exploit carefully engineered prompts to induce harmful or unethical responses. Such threats have raised critical concerns about the safety and reliability of LLMs in real-world deployment. While existing defense mechanisms partially mitigate such risks, subsequent advancements in adversarial techniques have enabled novel jailbreaking methods to circumvent these protections, exposing the limitations of static defense frameworks. In this work, we explore defending against evolving jailbreaking threats through the lens of context retrieval. First, we conduct a preliminary study demonstrating that even a minimal set of safety-aligned examples against a particular jailbreak can significantly enhance robustness against this attack pattern. Building on this insight, we further leverage the retrieval-augmented generation (RAG) techniques and propose Safety Context Retrieval (SCR), a scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our comprehensive experiments demonstrate how SCR achieves superior defensive performance against both established and emerging jailbreaking tactics, contributing a new paradigm to LLM safety. Our code will be available upon publication.
>
---
#### [new 177] QUADS: QUAntized Distillation Framework for Efficient Speech Language Understanding
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文属于语音语言理解（SLU）模型压缩任务，针对现有方法分别进行知识蒸馏和量化导致精度与效率失衡的问题，提出QUADS框架，通过多阶段联合优化蒸馏与量化，提升低比特环境下的模型效率与精度，在SLURP/FSC数据集上实现高准确率同时压缩模型规模83-700倍，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2505.14723v1](http://arxiv.org/pdf/2505.14723v1)**

> **作者:** Subrata Biswas; Mohammad Nur Hossain Khan; Bashima Islam
>
> **摘要:** Spoken Language Understanding (SLU) systems must balance performance and efficiency, particularly in resource-constrained environments. Existing methods apply distillation and quantization separately, leading to suboptimal compression as distillation ignores quantization constraints. We propose QUADS, a unified framework that optimizes both through multi-stage training with a pre-tuned model, enhancing adaptability to low-bit regimes while maintaining accuracy. QUADS achieves 71.13\% accuracy on SLURP and 99.20\% on FSC, with only minor degradations of up to 5.56\% compared to state-of-the-art models. Additionally, it reduces computational complexity by 60--73$\times$ (GMACs) and model size by 83--700$\times$, demonstrating strong robustness under extreme quantization. These results establish QUADS as a highly efficient solution for real-world, resource-constrained SLU applications.
>
---
#### [new 178] Think, Reflect, Create: Metacognitive Learning for Zero-Shot Robotic Planning with LLMs
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于零样本机器人规划任务，旨在解决大语言模型（LLMs）在复杂机器人任务中表现不足的问题。提出整合元认知学习的框架，通过技能分解与自我反思机制，使机器人代理能分解模块化技能、分析失败并生成新解决方案。实验显示其优于现有方法且能创造有效新策略。**

- **链接: [http://arxiv.org/pdf/2505.14899v1](http://arxiv.org/pdf/2505.14899v1)**

> **作者:** Wenjie Lin; Jin Wei-Kocsis
>
> **摘要:** While large language models (LLMs) have shown great potential across various domains, their applications in robotics remain largely limited to static, prompt-based behaviors and still face challenges in handling complex tasks under zero-shot or few-shot settings. Inspired by human metacognitive learning and creative problem-solving, we address this limitation by exploring a fundamental research question: Can LLMs be empowered with metacognitive capabilities to reason, reflect, and create, thereby enhancing their ability to perform robotic tasks with minimal demonstrations? In this paper, we present an early-stage framework that integrates metacognitive learning into LLM-powered multi-robot collaboration. The proposed framework equips the LLM-powered robotic agents with a skill decomposition and self-reflection mechanism that identifies modular skills from prior tasks, reflects on failures in unseen task scenarios, and synthesizes effective new solutions. Experimental results show that our metacognitive-learning-empowered LLM framework significantly outperforms existing baselines. Moreover, we observe that the framework is capable of generating solutions that differ from the ground truth yet still successfully complete the tasks. These exciting findings support our hypothesis that metacognitive learning can foster creativity in robotic planning.
>
---
#### [new 179] BountyBench: Dollar Impact of AI Agent Attackers and Defenders on Real-World Cybersecurity Systems
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出BountyBench框架，评估AI在真实网络安全系统中的攻防能力。针对漏洞检测、利用与修补任务，通过设置含真实代码库的25个系统及40个带赏金的漏洞，测试5种AI代理性能，量化其经济影响，解决AI攻防效果评估问题。**

- **链接: [http://arxiv.org/pdf/2505.15216v1](http://arxiv.org/pdf/2505.15216v1)**

> **作者:** Andy K. Zhang; Joey Ji; Celeste Menders; Riya Dulepet; Thomas Qin; Ron Y. Wang; Junrong Wu; Kyleen Liao; Jiliang Li; Jinghan Hu; Sara Hong; Nardos Demilew; Shivatmica Murgai; Jason Tran; Nishka Kacheria; Ethan Ho; Denis Liu; Lauren McLane; Olivia Bruvik; Dai-Rong Han; Seungwoo Kim; Akhil Vyas; Cuiyuanxiu Chen; Ryan Li; Weiran Xu; Jonathan Z. Ye; Prerit Choudhary; Siddharth M. Bhatia; Vikram Sivashankar; Yuxuan Bao; Dawn Song; Dan Boneh; Daniel E. Ho; Percy Liang
>
> **备注:** 78 pages
>
> **摘要:** AI agents have the potential to significantly alter the cybersecurity landscape. To help us understand this change, we introduce the first framework to capture offensive and defensive cyber-capabilities in evolving real-world systems. Instantiating this framework with BountyBench, we set up 25 systems with complex, real-world codebases. To capture the vulnerability lifecycle, we define three task types: Detect (detecting a new vulnerability), Exploit (exploiting a specific vulnerability), and Patch (patching a specific vulnerability). For Detect, we construct a new success indicator, which is general across vulnerability types and provides localized evaluation. We manually set up the environment for each system, including installing packages, setting up server(s), and hydrating database(s). We add 40 bug bounties, which are vulnerabilities with monetary awards from \$10 to \$30,485, and cover 9 of the OWASP Top 10 Risks. To modulate task difficulty, we devise a new strategy based on information to guide detection, interpolating from identifying a zero day to exploiting a specific vulnerability. We evaluate 5 agents: Claude Code, OpenAI Codex CLI, and custom agents with GPT-4.1, Gemini 2.5 Pro Preview, and Claude 3.7 Sonnet Thinking. Given up to three attempts, the top-performing agents are Claude Code (5% on Detect, mapping to \$1,350), Custom Agent with Claude 3.7 Sonnet Thinking (5% on Detect, mapping to \$1,025; 67.5% on Exploit), and OpenAI Codex CLI (5% on Detect, mapping to \$2,400; 90% on Patch, mapping to \$14,422). OpenAI Codex CLI and Claude Code are more capable at defense, achieving higher Patch scores of 90% and 87.5%, compared to Exploit scores of 32.5% and 57.5% respectively; in contrast, the custom agents are relatively balanced between offense and defense, achieving Exploit scores of 40-67.5% and Patch scores of 45-60%.
>
---
#### [new 180] Segmentation-Variant Codebooks for Preservation of Paralinguistic and Prosodic Information
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文针对语音自监督模型量化导致韵律及副语言信息丢失的问题，提出分段可变码本（SVC）方法，通过在帧、音素、词、utterance等不同语言单元进行多流量化，有效保留情感、重音等信息。实验表明其优于传统方法，在重合成任务中提升风格表现与质量同时保持可懂度。**

- **链接: [http://arxiv.org/pdf/2505.15667v1](http://arxiv.org/pdf/2505.15667v1)**

> **作者:** Nicholas Sanders; Yuanchao Li; Korin Richmond; Simon King
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Quantization in SSL speech models (e.g., HuBERT) improves compression and performance in tasks like language modeling, resynthesis, and text-to-speech but often discards prosodic and paralinguistic information (e.g., emotion, prominence). While increasing codebook size mitigates some loss, it inefficiently raises bitrates. We propose Segmentation-Variant Codebooks (SVCs), which quantize speech at distinct linguistic units (frame, phone, word, utterance), factorizing it into multiple streams of segment-specific discrete features. Our results show that SVCs are significantly more effective at preserving prosodic and paralinguistic information across probing tasks. Additionally, we find that pooling before rather than after discretization better retains segment-level information. Resynthesis experiments further confirm improved style realization and slightly improved quality while preserving intelligibility.
>
---
#### [new 181] Pass@K Policy Optimization: Solving Harder Reinforcement Learning Problems
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于强化学习策略优化任务，旨在解决传统RL仅优化pass@1导致忽视样本集多样性及集体效用的问题。提出Pass@K Policy Optimization（PKPO），通过设计低方差奖励变换函数直接优化pass@k性能，支持动态调整k值，提升探索能力，尤其在困难任务中突破学习瓶颈，同时改善pass@1与pass@k指标。**

- **链接: [http://arxiv.org/pdf/2505.15201v1](http://arxiv.org/pdf/2505.15201v1)**

> **作者:** Christian Walder; Deep Karkhanis
>
> **摘要:** Reinforcement Learning (RL) algorithms sample multiple n>1 solution attempts for each problem and reward them independently. This optimizes for pass@1 performance and prioritizes the strength of isolated samples at the expense of the diversity and collective utility of sets of samples. This under-utilizes the sampling capacity, limiting exploration and eventual improvement on harder examples. As a fix, we propose Pass-at-k Policy Optimization (PKPO), a transformation on the final rewards which leads to direct optimization of pass@k performance, thus optimizing for sets of samples that maximize reward when considered jointly. Our contribution is to derive novel low variance unbiased estimators for pass@k and its gradient, in both the binary and continuous reward settings. We show optimization with our estimators reduces to standard RL with rewards that have been jointly transformed by a stable and efficient transformation function. While previous efforts are restricted to k=n, ours is the first to enable robust optimization of pass@k for any arbitrary k <= n. Moreover, instead of trading off pass@1 performance for pass@k gains, our method allows annealing k during training, optimizing both metrics and often achieving strong pass@1 numbers alongside significant pass@k gains. We validate our reward transformations on toy experiments, which reveal the variance reducing properties of our formulations. We also include real-world examples using the open-source LLM, GEMMA-2. We find that our transformation effectively optimizes for the target k. Furthermore, higher k values enable solving more and harder problems, while annealing k boosts both the pass@1 and pass@k . Crucially, for challenging task sets where conventional pass@1 optimization stalls, our pass@k approach unblocks learning, likely due to better exploration by prioritizing joint utility over the utility of individual samples.
>
---
#### [new 182] ToxicTone: A Mandarin Audio Dataset Annotated for Toxicity and Toxic Utterance Tonality
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于中文音频毒性检测任务，针对现有研究缺乏标注 Mandarin 语音毒性数据及语音特征分析的不足，构建了标注毒性类型与情感来源的 ToxicTone 数据集（含13类场景），并提出融合声学、语言及情感特征的多模态检测框架，实验显示其优于文本基线模型。**

- **链接: [http://arxiv.org/pdf/2505.15773v1](http://arxiv.org/pdf/2505.15773v1)**

> **作者:** Yu-Xiang Luo; Yi-Cheng Lin; Ming-To Chuang; Jia-Hung Chen; I-Ning Tsai; Pei Xing Kiew; Yueh-Hsuan Huang; Chien-Feng Liu; Yu-Chen Chen; Bo-Han Feng; Wenze Ren; Hung-yi Lee
>
> **备注:** Accepted by INTERSPEECH 2025. 5 pages
>
> **摘要:** Despite extensive research on toxic speech detection in text, a critical gap remains in handling spoken Mandarin audio. The lack of annotated datasets that capture the unique prosodic cues and culturally specific expressions in Mandarin leaves spoken toxicity underexplored. To address this, we introduce ToxicTone -- the largest public dataset of its kind -- featuring detailed annotations that distinguish both forms of toxicity (e.g., profanity, bullying) and sources of toxicity (e.g., anger, sarcasm, dismissiveness). Our data, sourced from diverse real-world audio and organized into 13 topical categories, mirrors authentic communication scenarios. We also propose a multimodal detection framework that integrates acoustic, linguistic, and emotional features using state-of-the-art speech and emotion encoders. Extensive experiments show our approach outperforms text-only and baseline models, underscoring the essential role of speech-specific cues in revealing hidden toxic expressions.
>
---
#### [new 183] AI vs. Human Judgment of Content Moderation: LLM-as-a-Judge and Ethics-Based Response Refusals
- **分类: cs.HC; cs.CL**

- **简介: 该论文研究AI与人类在内容审核中的判断差异，探讨LLM作为评估者对拒绝响应的评价偏差。通过对比GPT-4o和Llama3对伦理（安全相关）与技术（系统限制）拒绝的评分，发现AI更倾向肯定伦理拒绝，揭示模型评估的" moderation bias"，引发对AI价值观对齐的伦理反思。**

- **链接: [http://arxiv.org/pdf/2505.15365v1](http://arxiv.org/pdf/2505.15365v1)**

> **作者:** Stefan Pasch
>
> **摘要:** As large language models (LLMs) are increasingly deployed in high-stakes settings, their ability to refuse ethically sensitive prompts-such as those involving hate speech or illegal activities-has become central to content moderation and responsible AI practices. While refusal responses can be viewed as evidence of ethical alignment and safety-conscious behavior, recent research suggests that users may perceive them negatively. At the same time, automated assessments of model outputs are playing a growing role in both evaluation and training. In particular, LLM-as-a-Judge frameworks-in which one model is used to evaluate the output of another-are now widely adopted to guide benchmarking and fine-tuning. This paper examines whether such model-based evaluators assess refusal responses differently than human users. Drawing on data from Chatbot Arena and judgments from two AI judges (GPT-4o and Llama 3 70B), we compare how different types of refusals are rated. We distinguish ethical refusals, which explicitly cite safety or normative concerns (e.g., "I can't help with that because it may be harmful"), and technical refusals, which reflect system limitations (e.g., "I can't answer because I lack real-time data"). We find that LLM-as-a-Judge systems evaluate ethical refusals significantly more favorably than human users, a divergence not observed for technical refusals. We refer to this divergence as a moderation bias-a systematic tendency for model-based evaluators to reward refusal behaviors more than human users do. This raises broader questions about transparency, value alignment, and the normative assumptions embedded in automated evaluation systems.
>
---
#### [new 184] Evolutionary Computation and Large Language Models: A Survey of Methods, Synergies, and Applications
- **分类: cs.NE; cs.CL; cs.MA; I.2.7; I.2.11**

- **简介: 该论文属综述任务，探讨进化计算（EC）与大型语言模型（LLMs）的协同，旨在解决两者结合以提升AI优化与语言处理的问题。工作包括分析EC优化LLMs的训练、提示工程等，LLMs增强EC的自动化设计，讨论协同框架、挑战及未来方向，倡导混合方法。**

- **链接: [http://arxiv.org/pdf/2505.15741v1](http://arxiv.org/pdf/2505.15741v1)**

> **作者:** Dikshit Chauhan; Bapi Dutta; Indu Bala; Niki van Stein; Thomas Bäck; Anupam Yadav
>
> **摘要:** Integrating Large Language Models (LLMs) and Evolutionary Computation (EC) represents a promising avenue for advancing artificial intelligence by combining powerful natural language understanding with optimization and search capabilities. This manuscript explores the synergistic potential of LLMs and EC, reviewing their intersections, complementary strengths, and emerging applications. We identify key opportunities where EC can enhance LLM training, fine-tuning, prompt engineering, and architecture search, while LLMs can, in turn, aid in automating the design, analysis, and interpretation of ECs. The manuscript explores the synergistic integration of EC and LLMs, highlighting their bidirectional contributions to advancing artificial intelligence. It first examines how EC techniques enhance LLMs by optimizing key components such as prompt engineering, hyperparameter tuning, and architecture search, demonstrating how evolutionary methods automate and refine these processes. Secondly, the survey investigates how LLMs improve EC by automating metaheuristic design, tuning evolutionary algorithms, and generating adaptive heuristics, thereby increasing efficiency and scalability. Emerging co-evolutionary frameworks are discussed, showcasing applications across diverse fields while acknowledging challenges like computational costs, interpretability, and algorithmic convergence. The survey concludes by identifying open research questions and advocating for hybrid approaches that combine the strengths of EC and LLMs.
>
---
## 更新

#### [replaced 001] KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14552v2](http://arxiv.org/pdf/2505.14552v2)**

> **作者:** Jiajun Shi; Jian Yang; Jiaheng Liu; Xingyuan Bu; Jiangjie Chen; Junting Zhou; Kaijing Ma; Zhoufutu Wen; Bingli Wang; Yancheng He; Liang Song; Hualei Zhu; Shilong Li; Xingjian Wang; Wei Zhang; Ruibin Yuan; Yifan Yao; Wenjun Yang; Yunli Wang; Siyuan Fang; Siyu Yuan; Qianyu He; Xiangru Tang; Yingshui Tan; Wangchunshu Zhou; Zhaoxiang Zhang; Zhoujun Li; Wenhao Huang; Ge Zhang
>
> **备注:** 22 pages
>
> **摘要:** Recent advancements in large language models (LLMs) underscore the need for more comprehensive evaluation methods to accurately assess their reasoning capabilities. Existing benchmarks are often domain-specific and thus cannot fully capture an LLM's general reasoning potential. To address this limitation, we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym), a dynamic evaluation platform inspired by KOR-Bench and Gymnasium. KORGym offers over fifty games in either textual or visual formats and supports interactive, multi-turn assessments with reinforcement learning scenarios. Using KORGym, we conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent reasoning patterns within model families and demonstrating the superior performance of closed-source models. Further analysis examines the effects of modality, reasoning strategies, reinforcement learning techniques, and response length on model performance. We expect KORGym to become a valuable resource for advancing LLM reasoning research and developing evaluation methodologies suited to complex, interactive environments.
>
---
#### [replaced 002] EmoHopeSpeech: An Annotated Dataset of Emotions and Hope Speech in English and Arabic
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.11959v2](http://arxiv.org/pdf/2505.11959v2)**

> **作者:** Wajdi Zaghouani; Md. Rafiul Biswas
>
> **摘要:** This research introduces a bilingual dataset comprising 23,456 entries for Arabic and 10,036 entries for English, annotated for emotions and hope speech, addressing the scarcity of multi-emotion (Emotion and hope) datasets. The dataset provides comprehensive annotations capturing emotion intensity, complexity, and causes, alongside detailed classifications and subcategories for hope speech. To ensure annotation reliability, Fleiss' Kappa was employed, revealing 0.75-0.85 agreement among annotators both for Arabic and English language. The evaluation metrics (micro-F1-Score=0.67) obtained from the baseline model (i.e., using a machine learning model) validate that the data annotations are worthy. This dataset offers a valuable resource for advancing natural language processing in underrepresented languages, fostering better cross-linguistic analysis of emotions and hope speech.
>
---
#### [replaced 003] FastDraft: How to Train Your Draft
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.11055v2](http://arxiv.org/pdf/2411.11055v2)**

> **作者:** Ofir Zafrir; Igor Margulis; Dorin Shteyman; Shira Guskin; Guy Boudoukh
>
> **备注:** ENLSP NeurIPS Workshop 2024
>
> **摘要:** Speculative Decoding has gained popularity as an effective technique for accelerating the auto-regressive inference process of Large Language Models. However, Speculative Decoding entirely relies on the availability of efficient draft models, which are often lacking for many existing language models due to a stringent constraint of vocabulary compatibility. In this work we introduce FastDraft, a novel and efficient approach for pre-training and aligning a draft model to any large language model by incorporating efficient pre-training, followed by fine-tuning over synthetic datasets generated by the target model. We demonstrate FastDraft by training two highly parameter efficient drafts for the popular Phi-3-mini and Llama-3.1-8B models. Using FastDraft, we were able to produce a draft model with approximately 10 billion tokens on a single server with 8 Intel$^\circledR$ Gaudi$^\circledR$ 2 accelerators in under 24 hours. Our results show that the draft model achieves impressive results in key metrics of acceptance rate, block efficiency and up to 3x memory bound speed up when evaluated on code completion and up to 2x in summarization, text completion and instruction tasks. We validate our theoretical findings through benchmarking on the latest Intel$^\circledR$ Core$^{\tiny \text{TM}}$ Ultra, achieving a wall-clock time speedup of up to 2x, indicating a significant reduction in runtime. Due to its high quality, FastDraft unlocks large language models inference on AI-PC and other edge-devices.
>
---
#### [replaced 004] MacRAG: Compress, Slice, and Scale-up for Multi-Scale Adaptive Context RAG
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.06569v2](http://arxiv.org/pdf/2505.06569v2)**

> **作者:** Woosang Lim; Zekun Li; Gyuwan Kim; Sungyoung Ji; HyeonJung Kim; Kyuri Choi; Jin Hyuk Lim; Kyungpyo Park; William Yang Wang
>
> **摘要:** Long-context large language models (LC LLMs) combined with retrieval-augmented generation (RAG) hold strong potential for complex multi-hop and large-document tasks. However, existing RAG systems often suffer from imprecise retrieval, incomplete context coverage under constrained windows, and fragmented information from suboptimal context construction. We introduce Multi-scale Adaptive Context RAG (MacRAG), a hierarchical RAG framework that compresses and partitions documents into coarse-to-fine granularities, then adaptively merges relevant contexts through real-time chunk- and document-level expansions. By initiating with finest-level retrieval and progressively incorporating broader, higher-level context, MacRAG constructs effective query-specific long contexts, optimizing both precision and coverage. Evaluations on challenging LongBench expansions of HotpotQA, 2WikiMultihopQA, and Musique confirm MacRAG consistently surpasses baseline RAG pipelines in single- and multi-step generation using Llama-3.1-8B, Gemini-1.5-pro, and GPT-4o. Our results establish MacRAG as an efficient, scalable solution for real-world long-context, multi-hop reasoning. Our code is available at https://github.com/Leezekun/MacRAG.
>
---
#### [replaced 005] SWE-smith: Scaling Data for Software Engineering Agents
- **分类: cs.SE; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21798v2](http://arxiv.org/pdf/2504.21798v2)**

> **作者:** John Yang; Kilian Leret; Carlos E. Jimenez; Alexander Wettig; Kabir Khandpur; Yanzhe Zhang; Binyuan Hui; Ofir Press; Ludwig Schmidt; Diyi Yang
>
> **备注:** All assets available at https://swesmith.com
>
> **摘要:** Despite recent progress in Language Models (LMs) for software engineering, collecting training data remains a significant pain point. Existing datasets are small, with at most 1,000s of training instances from 11 or fewer GitHub repositories. The procedures to curate such datasets are often complex, necessitating hundreds of hours of human labor; companion execution environments also take up several terabytes of storage, severely limiting their scalability and usability. To address this pain point, we introduce SWE-smith, a novel pipeline for generating software engineering training data at scale. Given any Python codebase, SWE-smith constructs a corresponding execution environment, then automatically synthesizes 100s to 1,000s of task instances that break existing test(s) in the codebase. Using SWE-smith, we create a dataset of 50k instances sourced from 128 GitHub repositories, an order of magnitude larger than all previous works. We train SWE-agent-LM-32B, achieving 40.2% Pass@1 resolve rate on the SWE-bench Verified benchmark, state of the art among open source models. We open source SWE-smith (collection procedure, task instances, trajectories, models) to lower the barrier of entry for research in LM systems for automated software engineering. All assets available at https://swesmith.com.
>
---
#### [replaced 006] Reducing Hallucinations in Language Model-based SPARQL Query Generation Using Post-Generation Memory Retrieval
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13369v2](http://arxiv.org/pdf/2502.13369v2)**

> **作者:** Aditya Sharma; Luis Lara; Christopher J. Pal; Amal Zouaq
>
> **摘要:** The ability to generate SPARQL queries from natural language questions is crucial for ensuring efficient and accurate retrieval of structured data from knowledge graphs (KG). While large language models (LLMs) have been widely adopted for SPARQL query generation, they are often susceptible to hallucinations and out-of-distribution errors when producing KG elements like Uniform Resource Identifiers (URIs) based on internal parametric knowledge. This often results in content that appears plausible but is factually incorrect, posing significant challenges for their use in real-world information retrieval (IR) applications. This has led to increased research aimed at detecting and mitigating such errors. In this paper, we introduce PGMR (Post-Generation Memory Retrieval), a modular framework that incorporates a non-parametric memory module to retrieve KG elements and enhance LLM-based SPARQL query generation. Our experimental results indicate that PGMR consistently delivers strong performance across diverse datasets, data distributions, and LLMs. Notably, PGMR significantly mitigates URI hallucinations, nearly eliminating the problem in several scenarios.
>
---
#### [replaced 007] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.15573v2](http://arxiv.org/pdf/2504.15573v2)**

> **作者:** Yuxin Jiang; Yufei Wang; Chuhan Wu; Xinyi Dai; Yan Xu; Weinan Gan; Yasheng Wang; Xin Jiang; Lifeng Shang; Ruiming Tang; Wei Wang
>
> **备注:** 16 pages, 11 figures, 9 tables. ACL 2025 camera-ready version
>
> **摘要:** The improvement of LLMs' instruction-following capabilities depends critically on the availability of high-quality instruction-response pairs. While existing automatic data synthetic methods alleviate the burden of manual curation, they often rely heavily on either the quality of seed data or strong assumptions about the structure and content of web documents. To tackle these challenges, we propose Web Reconstruction (WebR), a fully automated framework for synthesizing high-quality instruction-tuning (IT) data directly from raw web documents with minimal assumptions. Leveraging the inherent diversity of raw web content, we conceptualize web reconstruction as an instruction-tuning data synthesis task via a novel dual-perspective paradigm--Web as Instruction and Web as Response--where each web document is designated as either an instruction or a response to trigger the reconstruction process. Comprehensive experiments show that datasets generated by WebR outperform state-of-the-art baselines by up to 16.65% across four instruction-following benchmarks. Notably, WebR demonstrates superior compatibility, data efficiency, and scalability, enabling enhanced domain adaptation with minimal effort. The data and code are publicly available at https://github.com/YJiangcm/WebR.
>
---
#### [replaced 008] CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.07316v4](http://arxiv.org/pdf/2502.07316v4)**

> **作者:** Junlong Li; Daya Guo; Dejian Yang; Runxin Xu; Yu Wu; Junxian He
>
> **备注:** ICML 2025
>
> **摘要:** Reasoning is a fundamental capability of Large Language Models. While prior research predominantly focuses on enhancing narrow skills like math or code generation, improving performance on many other reasoning tasks remains challenging due to sparse and fragmented training data. To address this issue, we propose CodeI/O, a novel approach that systematically condenses diverse reasoning patterns inherently embedded in contextually-grounded codes, through transforming the original code into a code input-output prediction format. By training models to predict inputs/outputs given code and test cases entirely in natural language as Chain-of-Thought (CoT) rationales, we expose them to universal reasoning primitives -- like logic flow planning, state-space searching, decision tree traversal, and modular decomposition -- while decoupling structured reasoning from code-specific syntax and preserving procedural rigor. Experimental results demonstrate CodeI/O leads to consistent improvements across symbolic, scientific, logic, math & numerical, and commonsense reasoning tasks. By matching the existing ground-truth outputs or re-executing the code with predicted inputs, we can verify each prediction and further enhance the CoTs through multi-turn revision, resulting in CodeI/O++ and achieving higher performance. Our data and models are available at https://github.com/hkust-nlp/CodeIO.
>
---
#### [replaced 009] A Comprehensive Evaluation of Large Language Models on Temporal Event Forecasting
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2407.11638v2](http://arxiv.org/pdf/2407.11638v2)**

> **作者:** He Chang; Chenchen Ye; Zhulin Tao; Jie Wu; Zhengmao Yang; Yunshan Ma; Xianglin Huang; Tat-Seng Chua
>
> **摘要:** Recently, Large Language Models (LLMs) have demonstrated great potential in various data mining tasks, such as knowledge question answering, mathematical reasoning, and commonsense reasoning. However, the reasoning capability of LLMs on temporal event forecasting has been under-explored. To systematically investigate their abilities in temporal event forecasting, we conduct a comprehensive evaluation of LLM-based methods for temporal event forecasting. Due to the lack of a high-quality dataset that involves both graph and textual data, we first construct a benchmark dataset, named MidEast-TE-mini. Based on this dataset, we design a series of baseline methods, characterized by various input formats and retrieval augmented generation (RAG) modules. From extensive experiments, we find that directly integrating raw texts into the input of LLMs does not enhance zero-shot extrapolation performance. In contrast, fine-tuning LLMs with raw texts can significantly improve performance. Additionally, LLMs enhanced with retrieval modules can effectively capture temporal relational patterns hidden in historical events. However, issues such as popularity bias and the long-tail problem persist in LLMs, particularly in the retrieval-augmented generation (RAG) method. These findings not only deepen our understanding of LLM-based event forecasting methods but also highlight several promising research directions. We consider that this comprehensive evaluation, along with the identified research opportunities, will significantly contribute to future research on temporal event forecasting through LLMs.
>
---
#### [replaced 010] Ada-R1: Hybrid-CoT via Bi-Level Adaptive Reasoning Optimization
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21659v2](http://arxiv.org/pdf/2504.21659v2)**

> **作者:** Haotian Luo; Haiying He; Yibo Wang; Jinluan Yang; Rui Liu; Naiqiang Tan; Xiaochun Cao; Dacheng Tao; Li Shen
>
> **摘要:** Recently, long-thought reasoning models achieve strong performance on complex reasoning tasks, but often incur substantial inference overhead, making efficiency a critical concern. Our empirical analysis reveals that the benefit of using Long-CoT varies across problems: while some problems require elaborate reasoning, others show no improvement, or even degraded accuracy. This motivates adaptive reasoning strategies that tailor reasoning depth to the input. However, prior work primarily reduces redundancy within long reasoning paths, limiting exploration of more efficient strategies beyond the Long-CoT paradigm. To address this, we propose a novel two-stage framework for adaptive and efficient reasoning. First, we construct a hybrid reasoning model by merging long and short CoT models to enable diverse reasoning styles. Second, we apply bi-level preference training to guide the model to select suitable reasoning styles (group-level), and prefer concise and correct reasoning within each style group (instance-level). Experiments demonstrate that our method (Ada-R1) significantly reduces inference costs compared to other baseline approaches, while maintaining performance. Notably, on five mathematical datasets, the average length of reasoning is reduced by more than 50%, highlighting the potential of adaptive strategies to optimize reasoning efficiency in large language models. Our code is coming soon at https://github.com/StarDewXXX/AdaR1
>
---
#### [replaced 011] NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.03797v3](http://arxiv.org/pdf/2409.03797v3)**

> **作者:** Kinjal Basu; Ibrahim Abdelaziz; Kiran Kate; Mayank Agarwal; Maxwell Crouse; Yara Rizk; Kelsey Bradford; Asim Munawar; Sadhana Kumaravel; Saurabh Goyal; Xin Wang; Luis A. Lastras; Pavan Kapanipathi
>
> **摘要:** The resurgence of autonomous agents built using large language models (LLMs) to solve complex real-world tasks has brought increased focus on LLMs' fundamental ability of tool or function calling. At the core of these agents, an LLM must plan, execute, and respond using external tools, APIs, and custom functions. Research on tool calling has gathered momentum, but evaluation benchmarks and datasets representing the complexity of the tasks have lagged behind. In this work, we focus on one such complexity, nested sequencing, with the goal of extending existing benchmarks and evaluation. Specifically, we present NESTFUL, a benchmark to evaluate LLMs on nested sequences of API calls, i.e., sequences where the output of one API call is passed as input to a subsequent call. NESTFUL contains 1800+ nested sequences where all the function calls are executable. Experimental results on a variety of models show that the best-performing model (GPT-4o) achieves a full sequence match accuracy of 28% and a win-rate of 60%, necessitating a large scope for improvement in the nested sequencing aspect of function calling. Our analysis of these results provides possible future research directions for the community, in addition to a benchmark to track progress. We have released the NESTFUL dataset under the Apache 2.0 license at https://github.com/IBM/NESTFUL.
>
---
#### [replaced 012] MRAG: A Modular Retrieval Framework for Time-Sensitive Question Answering
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.15540v2](http://arxiv.org/pdf/2412.15540v2)**

> **作者:** Zhang Siyue; Xue Yuxiang; Zhang Yiming; Wu Xiaobao; Luu Anh Tuan; Zhao Chen
>
> **摘要:** Understanding temporal relations and answering time-sensitive questions is crucial yet a challenging task for question-answering systems powered by large language models (LLMs). Existing approaches either update the parametric knowledge of LLMs with new facts, which is resource-intensive and often impractical, or integrate LLMs with external knowledge retrieval (i.e., retrieval-augmented generation). However, off-the-shelf retrievers often struggle to identify relevant documents that require intensive temporal reasoning. To systematically study time-sensitive question answering, we introduce the TempRAGEval benchmark, which repurposes existing datasets by incorporating temporal perturbations and gold evidence labels. As anticipated, all existing retrieval methods struggle with these temporal reasoning-intensive questions. We further propose Modular Retrieval (MRAG), a trainless framework that includes three modules: (1) Question Processing that decomposes question into a main content and a temporal constraint; (2) Retrieval and Summarization that retrieves evidence and uses LLMs to summarize according to the main content; (3) Semantic-Temporal Hybrid Ranking that scores each evidence summarization based on both semantic and temporal relevance. On TempRAGEval, MRAG significantly outperforms baseline retrievers in retrieval performance, leading to further improvements in final answer accuracy.
>
---
#### [replaced 013] Design and Implementation of an FPGA-Based Hardware Accelerator for Transformer
- **分类: cs.AR; cs.CL; cs.LG; B.7.1; C.1.4**

- **链接: [http://arxiv.org/pdf/2503.16731v3](http://arxiv.org/pdf/2503.16731v3)**

> **作者:** Richie Li; Sicheng Chen
>
> **备注:** 7 pages, 4 figures, 2 tables. Prepared in ACM conference style. Preprint under review
>
> **摘要:** Transformer-based large language models (LLMs) rely heavily on intensive matrix multiplications for attention and feed-forward layers, with the Q, K, and V linear projections in the Multi-Head Self-Attention (MHA) module constituting a decisive performance bottleneck. In this work, we introduce a highly optimized tiled matrix multiplication accelerator on a resource-constrained Xilinx KV260 FPGA that not only addresses this challenge but sets a new standard for efficiency and performance. Our design exploits persistent on-chip storage, a robust two-level tiling strategy for maximal data reuse, and a systolic-like unrolled compute engine that together deliver unparalleled speed and energy efficiency. Integrated with DistilBERT for Q, K, and V projections, our accelerator achieves an unequivocal 7x speedup over ARM CPU implementations (PyTorch) and an extraordinary 200x improvement over naive NumPy, reaching a throughput of up to 3.1~GFLOPs for matrix multiplications on (64,768) x (768,3072) matrices while operating at a conservative 100 MHz. These results decisively demonstrate the transformative potential of FPGA-based acceleration for critical Transformer operations, paving the way for scalable and energy-efficient deep learning inference on edge devices.
>
---
#### [replaced 014] Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2502.19721v2](http://arxiv.org/pdf/2502.19721v2)**

> **作者:** Hannah Cyberey; Yangfeng Ji; David Evans
>
> **摘要:** Large language models (LLMs) are known to perpetuate stereotypes and exhibit biases. Various strategies have been proposed to mitigate these biases, but most work studies biases in LLMs as a black-box problem without considering how concepts are represented within the model. We adapt techniques from representation engineering to study how the concept of "gender" is represented within LLMs. We introduce a new method that extracts concept representations via probability weighting without labeled data and efficiently selects a steering vector for measuring and manipulating the model's representation. We also present a projection-based method that enables precise steering of model predictions and demonstrate its effectiveness in mitigating gender bias in LLMs. Our code is available at: https://github.com/hannahxchen/gender-bias-steering
>
---
#### [replaced 015] Large Language Models Post-training: Surveying Techniques from Alignment to Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.06072v2](http://arxiv.org/pdf/2503.06072v2)**

> **作者:** Guiyao Tie; Zeli Zhao; Dingjie Song; Fuyang Wei; Rong Zhou; Yurou Dai; Wen Yin; Zhejian Yang; Jiangyue Yan; Yao Su; Zhenhan Dai; Yifeng Xie; Yihan Cao; Lichao Sun; Pan Zhou; Lifang He; Hechang Chen; Yu Zhang; Qingsong Wen; Tianming Liu; Neil Zhenqiang Gong; Jiliang Tang; Caiming Xiong; Heng Ji; Philip S. Yu; Jianfeng Gao
>
> **备注:** 87 pages, 21 figures, 9 tables
>
> **摘要:** The emergence of Large Language Models (LLMs) has fundamentally transformed natural language processing, making them indispensable across domains ranging from conversational systems to scientific exploration. However, their pre-trained architectures often reveal limitations in specialized contexts, including restricted reasoning capacities, ethical uncertainties, and suboptimal domain-specific performance. These challenges necessitate advanced post-training language models (PoLMs) to address these shortcomings, such as OpenAI-o1/o3 and DeepSeek-R1 (collectively known as Large Reasoning Models, or LRMs). This paper presents the first comprehensive survey of PoLMs, systematically tracing their evolution across five core paradigms: Fine-tuning, which enhances task-specific accuracy; Alignment, which ensures ethical coherence and alignment with human preferences; Reasoning, which advances multi-step inference despite challenges in reward design; Efficiency, which optimizes resource utilization amidst increasing complexity; Integration and Adaptation, which extend capabilities across diverse modalities while addressing coherence issues. Charting progress from ChatGPT's alignment strategies to DeepSeek-R1's innovative reasoning advancements, we illustrate how PoLMs leverage datasets to mitigate biases, deepen reasoning capabilities, and enhance domain adaptability. Our contributions include a pioneering synthesis of PoLM evolution, a structured taxonomy categorizing techniques and datasets, and a strategic agenda emphasizing the role of LRMs in improving reasoning proficiency and domain flexibility. As the first survey of its scope, this work consolidates recent PoLM advancements and establishes a rigorous intellectual framework for future research, fostering the development of LLMs that excel in precision, ethical robustness, and versatility across scientific and societal applications.
>
---
#### [replaced 016] PixelWorld: Towards Perceiving Everything as Pixels
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.19339v2](http://arxiv.org/pdf/2501.19339v2)**

> **作者:** Zhiheng Lyu; Xueguang Ma; Wenhu Chen
>
> **摘要:** Recent agentic language models increasingly need to interact directly with real-world environments containing intertwined visual and textual information through raw camera pixels, rather than relying on separate image and tokenized text processing, underscoring the necessity of a unified perception paradigm. To close this gap, we explore this idea through Perceive Everything as Pixels (PEAP) and release PixelWorld, a benchmark that renders natural-language, tabular, mathematical and diagrammatic inputs into a single pixel space. Experiments show that PEAP attains competitive accuracy on semantic-understanding tasks, indicating that a vision transformer can capture global textual semantics without explicit tokens. In contrast, reasoning-intensive benchmarks (math and code) exhibit sharp performance drops; however, Chain-of-Thought prompting partially mitigates this gap, hinting that explicit reasoning traces compensate for the missing token structure. We also find that when visual and textual information are closely integrated, representing everything as pixels reduces preprocessing complexity and avoids misalignment issues that often arise in separate pipelines. PixelWorld therefore serves as a practical benchmark for evaluating unified vision-language models and supports broader exploration of PEAP across diverse tasks.
>
---
#### [replaced 017] SQLCritic: Correcting Text-to-SQL Generation via Clause-wise Critic
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.07996v4](http://arxiv.org/pdf/2503.07996v4)**

> **作者:** Jikai Chen; Leilei Gan; Ziyu Zhao; Zechuan Wang; Dong Wang; Chenyi Zhuang
>
> **摘要:** Existing refinement methods in LLM-based Text-to-SQL systems exhibit limited effectiveness. They often introduce new errors during the self-correction process and fail to detect and correct semantic inaccuracies. To address these gaps, we first introduce a clause-wise critique generation task along with a benchmark, SQLCriticBench, which performs fine-grained error localization including both syntax and semantic errors at the clause level. Furthermore, we introduce a variant of DPO for training our SQLCritic model, where the $\beta$ coefficient is adaptively changed according to the clause-level inconsistencies between the preferred and dispreferred critiques. We also propose an automatically training dataset curation pipeline which annotate clause-wise critique at scale in a cost-effective way. Experiments demonstrate that the SQLCritic model significantly improves SQL accuracy on the BIRD and Spider datasets, and the results on SQLCriticBench further reveals its superior critique capabilities compared to existing models.
>
---
#### [replaced 018] AlignRAG: Leveraging Critique Learning for Evidence-Sensitive Retrieval-Augmented Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14858v3](http://arxiv.org/pdf/2504.14858v3)**

> **作者:** Jiaqi Wei; Hao Zhou; Xiang Zhang; Di Zhang; Zijie Qiu; Wei Wei; Jinzhe Li; Wanli Ouyang; Siqi Sun
>
> **摘要:** Retrieval-augmented generation (RAG) has become a widely adopted paradigm for enabling knowledge-grounded large language models (LLMs). However, standard RAG pipelines often fail to ensure that model reasoning remains consistent with the evidence retrieved, leading to factual inconsistencies or unsupported conclusions. In this work, we reinterpret RAG as Retrieval-Augmented Reasoning and identify a central but underexplored problem: \textit{Reasoning Misalignment}-the divergence between an LLM's internal reasoning trajectory and the evidential constraints provided by retrieval. To address this issue, we propose \textsc{AlignRAG}, a novel iterative framework grounded in Critique-Driven Alignment (CDA). At the heart of \textsc{AlignRAG} lies a \textit{contrastive critique synthesis} mechanism that generates retrieval-sensitive critiques while mitigating self-bias. This mechanism trains a dedicated retrieval-augmented \textit{Critic Language Model (CLM)} using labeled critiques that distinguish between evidence-aligned and misaligned reasoning. Alignment signals for supervision are obtained through self-supervised or externally guided labeling strategies. The resulting CLM is explicitly optimized for evidence sensitivity, enabling it to detect and revise reasoning errors during inference without relying solely on self-generated feedback. Empirical evaluations show that our 8B-parameter CLM improves performance over the Self-Refine baseline by 12.1\% on out-of-domain tasks and outperforms a standard 72B-parameter CLM by 2.2\%, while remaining compatible with existing RAG architectures as a plug-and-play module. Overall, AlignRAG offers a principled solution for aligning model reasoning with retrieved evidence, substantially improving the factual reliability and robustness of RAG systems.
>
---
#### [replaced 019] Pierce the Mists, Greet the Sky: Decipher Knowledge Overshadowing via Knowledge Circuit Analysis
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14406v2](http://arxiv.org/pdf/2505.14406v2)**

> **作者:** Haoming Huang; Yibo Yan; Jiahao Huo; Xin Zou; Xinfeng Li; Kun Wang; Xuming Hu
>
> **备注:** Under review
>
> **摘要:** Large Language Models (LLMs), despite their remarkable capabilities, are hampered by hallucinations. A particularly challenging variant, knowledge overshadowing, occurs when one piece of activated knowledge inadvertently masks another relevant piece, leading to erroneous outputs even with high-quality training data. Current understanding of overshadowing is largely confined to inference-time observations, lacking deep insights into its origins and internal mechanisms during model training. Therefore, we introduce PhantomCircuit, a novel framework designed to comprehensively analyze and detect knowledge overshadowing. By innovatively employing knowledge circuit analysis, PhantomCircuit dissects the internal workings of attention heads, tracing how competing knowledge pathways contribute to the overshadowing phenomenon and its evolution throughout the training process. Extensive experiments demonstrate PhantomCircuit's effectiveness in identifying such instances, offering novel insights into this elusive hallucination and providing the research community with a new methodological lens for its potential mitigation.
>
---
#### [replaced 020] Finding the Sweet Spot: Preference Data Construction for Scaling Preference Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16825v2](http://arxiv.org/pdf/2502.16825v2)**

> **作者:** Yao Xiao; Hai Ye; Linyao Chen; Hwee Tou Ng; Lidong Bing; Xiaoli Li; Roy Ka-wei Lee
>
> **备注:** ACL25 Main
>
> **摘要:** Iterative data generation and model retraining are widely used to align large language models (LLMs). It typically involves a policy model to generate on-policy responses and a reward model to guide training data selection. Direct Preference Optimization (DPO) further enhances this process by constructing preference pairs of chosen and rejected responses. In this work, we aim to \emph{scale up} the number of on-policy samples via repeated random sampling to improve alignment performance. Conventional practice selects the sample with the highest reward as chosen and the lowest as rejected for DPO. However, our experiments reveal that this strategy leads to a \emph{decline} in performance as the sample size increases. To address this, we investigate preference data construction through the lens of underlying normal distribution of sample rewards. We categorize the reward space into seven representative points and systematically explore all 21 ($C_7^2$) pairwise combinations. Through evaluations on four models using AlpacaEval 2, we find that selecting the rejected response at reward position $\mu - 2\sigma$ rather than the minimum reward, is crucial for optimal performance. We finally introduce a scalable preference data construction strategy that consistently enhances model performance as the sample scale increases.
>
---
#### [replaced 021] Large Language Models are Powerful Electronic Health Record Encoders
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17403v3](http://arxiv.org/pdf/2502.17403v3)**

> **作者:** Stefan Hegselmann; Georg von Arnim; Tillmann Rheude; Noel Kronenberg; David Sontag; Gerhard Hindricks; Roland Eils; Benjamin Wild
>
> **摘要:** Electronic Health Records (EHRs) offer considerable potential for clinical prediction, but their complexity and heterogeneity present significant challenges for traditional machine learning methods. Recently, domain-specific EHR foundation models trained on large volumes of unlabeled EHR data have shown improved predictive accuracy and generalization. However, their development is constrained by limited access to diverse, high-quality datasets, and by inconsistencies in coding standards and clinical practices. In this study, we explore the use of general-purpose Large Language Models (LLMs) to encode EHR into high-dimensional representations for downstream clinical prediction tasks. We convert structured EHR data into markdown-formatted plain text documents by replacing medical codes with natural language descriptions. This enables the use of LLMs and their extensive semantic understanding and generalization capabilities as effective encoders of EHRs without requiring access to private medical training data. We show that LLM-based embeddings can often match or even surpass the performance of a specialized EHR foundation model, CLMBR-T-Base, across 15 diverse clinical tasks from the EHRSHOT benchmark. To demonstrate generalizability, we further evaluate the approach on the UK Biobank (UKB) cohort, a population distinct from that used to train CLMBR-T-Base. Notably, one of the tested LLM-based models achieves superior performance for disease onset, hospitalization, and mortality prediction, highlighting robustness to shifts in patient populations. Our findings suggest that repurposed general-purpose LLMs for EHR encoding provide a scalable and generalizable alternative to domain-specific models for clinical prediction.
>
---
#### [replaced 022] GLiNER-BioMed: A Suite of Efficient Models for Open Biomedical Named Entity Recognition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.00676v2](http://arxiv.org/pdf/2504.00676v2)**

> **作者:** Anthony Yazdani; Ihor Stepanov; Douglas Teodoro
>
> **摘要:** Biomedical named entity recognition (NER) presents unique challenges due to specialized vocabularies, the sheer volume of entities, and the continuous emergence of novel entities. Traditional NER models, constrained by fixed taxonomies and human annotations, struggle to generalize beyond predefined entity types. To address these issues, we introduce GLiNER-BioMed, a domain-adapted suite of Generalist and Lightweight Model for NER (GLiNER) models specifically tailored for biomedicine. In contrast to conventional approaches, GLiNER uses natural language labels to infer arbitrary entity types, enabling zero-shot recognition. Our approach first distills the annotation capabilities of large language models (LLMs) into a smaller, more efficient model, enabling the generation of high-coverage synthetic biomedical NER data. We subsequently train two GLiNER architectures, uni- and bi-encoder, at multiple scales to balance computational efficiency and recognition performance. Experiments on several biomedical datasets demonstrate that GLiNER-BioMed outperforms the state-of-the-art in both zero- and few-shot scenarios, achieving 5.96% improvement in F1-score over the strongest baseline (p-value < 0.001). Ablation studies highlight the effectiveness of our synthetic data generation strategy and emphasize the complementary benefits of synthetic biomedical pre-training combined with fine-tuning on general-domain annotations. All datasets, models, and training pipelines are publicly available at https://github.com/ds4dh/GLiNER-biomed.
>
---
#### [replaced 023] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.02106v2](http://arxiv.org/pdf/2406.02106v2)**

> **作者:** Weiqi Wang; Yangqiu Song
>
> **备注:** ACL2025
>
> **摘要:** To enable Large Language Models (LLMs) to function as conscious agents with generalizable reasoning capabilities, it is crucial that they possess the reasoning ability to comprehend situational changes (transitions) in distribution triggered by environmental factors or actions from other agents. Despite its fundamental significance, this ability remains underexplored due to the complexity of modeling infinite possible changes in an event and their associated distributions, coupled with the lack of benchmark data with situational transitions. Addressing these gaps, we propose a novel formulation of reasoning with distributional changes as a three-step discriminative process, termed as MetAphysical ReaSoning. We then introduce the first-ever benchmark, MARS, comprising three tasks corresponding to each step. These tasks systematically assess LLMs' capabilities in reasoning the plausibility of (i) changes in actions, (ii) states caused by changed actions, and (iii) situational transitions driven by changes in action. Extensive evaluations with 20 (L)LMs of varying sizes and methods indicate that all three tasks in this process pose significant challenges, even for state-of-the-art LLMs and LMs after fine-tuning. Further analyses reveal potential causes for the underperformance of LLMs and demonstrate that pre-training them on large-scale conceptualization taxonomies can potentially enhance their metaphysical reasoning capabilities. Our data and models are publicly accessible at https://github.com/HKUST-KnowComp/MARS.
>
---
#### [replaced 024] Improving Language Model Personas via Rationalization with Psychological Scaffolds
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17993v2](http://arxiv.org/pdf/2504.17993v2)**

> **作者:** Brihi Joshi; Xiang Ren; Swabha Swayamdipta; Rik Koncel-Kedziorski; Tim Paek
>
> **摘要:** Language models prompted with a user description or persona are being used to predict the user's preferences and opinions. However, existing approaches to building personas mostly rely on a user's demographic attributes and/or prior judgments, but not on any underlying reasoning behind a user's judgments. We introduce PB&J (Psychology of Behavior and Judgments), a framework that improves LM personas by incorporating potential rationales for why the user could have made a certain judgment. Our rationales are generated by a language model to explicitly reason about a user's behavior on the basis of their experiences, personality traits, or beliefs. Our method employs psychological scaffolds: structured frameworks such as the Big 5 Personality Traits or Primal World Beliefs to help ground the generated rationales in existing theories. Experiments on public opinion and movie preference prediction tasks demonstrate that language model personas augmented with PB&J rationales consistently outperform personas conditioned only on user demographics and / or judgments, including those that use a model's default chain-of-thought, which is not grounded in psychological theories. Additionally, our PB&J personas perform competitively with those using human-written rationales, suggesting the potential of synthetic rationales guided by existing theories.
>
---
#### [replaced 025] Neurons Speak in Ranges: Breaking Free from Discrete Neuronal Attribution
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.06809v2](http://arxiv.org/pdf/2502.06809v2)**

> **作者:** Muhammad Umair Haider; Hammad Rizwan; Hassan Sajjad; Peizhong Ju; A. B. Siddique
>
> **摘要:** Interpreting the internal mechanisms of large language models (LLMs) is crucial for improving their trustworthiness and utility. Prior work has primarily focused on mapping individual neurons to discrete semantic concepts. However, such mappings struggle to handle the inherent polysemanticity in LLMs, where individual neurons encode multiple, distinct concepts. Through a comprehensive analysis of both encoder and decoder-based LLMs across diverse datasets, we observe that even highly salient neurons, identified via various attribution techniques for specific semantic concepts, consistently exhibit polysemantic behavior. Importantly, activation magnitudes for fine-grained concepts follow distinct, often Gaussian-like distributions with minimal overlap. This observation motivates a shift from neuron attribution to range-based interpretation. We hypothesize that interpreting and manipulating neuron activation ranges would enable more precise interpretability and targeted interventions in LLMs. To validate our hypothesis, we introduce NeuronLens, a novel range-based interpretation and manipulation framework that provides a finer view of neuron activation distributions to localize concept attribution within a neuron. Extensive empirical evaluations demonstrate that NeuronLens significantly reduces unintended interference, while maintaining precise manipulation of targeted concepts, outperforming neuron attribution.
>
---
#### [replaced 026] Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale datasets for Responsible LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.02009v2](http://arxiv.org/pdf/2505.02009v2)**

> **作者:** Sai Krishna Mendu; Harish Yenala; Aditi Gulati; Shanu Kumar; Parag Agrawal
>
> **备注:** 10 pages, 5 figures. Accepted at the International Joint Conferences on Artificial Intelligence IJCAI 2025 (main track)
>
> **摘要:** Large language models (LLMs) have become integral to various real-world applications, leveraging massive, web-sourced datasets like Common Crawl, C4, and FineWeb for pretraining. While these datasets provide linguistic data essential for high-quality natural language generation, they often contain harmful content, such as hate speech, misinformation, and biased narratives. Training LLMs on such unfiltered data risks perpetuating toxic behaviors, spreading misinformation, and amplifying societal biases which can undermine trust in LLM-driven applications and raise ethical concerns about their use. This paper presents a large-scale analysis of inappropriate content across these datasets, offering a comprehensive taxonomy that categorizes harmful webpages into Topical and Toxic based on their intent. We also introduce a prompt evaluation dataset, a high-accuracy Topical and Toxic Prompt (TTP), and a transformer-based model (HarmFormer) for harmful content filtering. Additionally, we create a new multi-harm open-ended toxicity benchmark (HAVOC) and provide crucial insights into how models respond to adversarial toxic inputs. We share TTP, TTP-Eval, HAVOC and a sample of C4 inferenced on HarmFormer. Our work offers insights into ensuring safer LLM pretraining and serves as a resource for Responsible AI (RAI) compliance.
>
---
#### [replaced 027] Effectively Controlling Reasoning Models through Thinking Intervention
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24370v3](http://arxiv.org/pdf/2503.24370v3)**

> **作者:** Tong Wu; Chong Xiang; Jiachen T. Wang; G. Edward Suh; Prateek Mittal
>
> **摘要:** Reasoning-enhanced large language models (LLMs) explicitly generate intermediate reasoning steps prior to generating final answers, helping the model excel in complex problem-solving. In this paper, we demonstrate that this emerging generation framework offers a unique opportunity for more fine-grained control over model behavior. We propose Thinking Intervention, a novel paradigm designed to explicitly guide the internal reasoning processes of LLMs by strategically inserting or revising specific thinking tokens. We find that the Thinking Intervention paradigm enhances the capabilities of reasoning models across a wide range of tasks, including instruction following on IFEval and Overthinking, instruction hierarchy on SEP, and safety alignment on XSTest and SorryBench. Our results demonstrate that Thinking Intervention significantly outperforms baseline prompting approaches, achieving up to 6.7% accuracy gains in instruction-following scenarios, 15.4% improvements in reasoning about instruction hierarchies, and a 40.0% increase in refusal rates for unsafe prompts using open-source DeepSeek R1 models. Overall, our work opens a promising new research avenue for controlling reasoning LLMs.
>
---
#### [replaced 028] SafeRoute: Adaptive Model Selection for Efficient and Accurate Safety Guardrails in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12464v4](http://arxiv.org/pdf/2502.12464v4)**

> **作者:** Seanie Lee; Dong Bok Lee; Dominik Wagner; Minki Kang; Haebin Seong; Tobias Bocklet; Juho Lee; Sung Ju Hwang
>
> **备注:** ACL 2025 findings
>
> **摘要:** Deploying large language models (LLMs) in real-world applications requires robust safety guard models to detect and block harmful user prompts. While large safety guard models achieve strong performance, their computational cost is substantial. To mitigate this, smaller distilled models are used, but they often underperform on "hard" examples where the larger model provides accurate predictions. We observe that many inputs can be reliably handled by the smaller model, while only a small fraction require the larger model's capacity. Motivated by this, we propose SafeRoute, a binary router that distinguishes hard examples from easy ones. Our method selectively applies the larger safety guard model to the data that the router considers hard, improving efficiency while maintaining accuracy compared to solely using the larger safety guard model. Experimental results on multiple benchmark datasets demonstrate that our adaptive model selection significantly enhances the trade-off between computational cost and safety performance, outperforming relevant baselines.
>
---
#### [replaced 029] Inverse Design of Metal-Organic Frameworks Using Quantum Natural Language Processing
- **分类: cs.LG; cs.AI; cs.CL; quant-ph**

- **链接: [http://arxiv.org/pdf/2405.11783v2](http://arxiv.org/pdf/2405.11783v2)**

> **作者:** Shinyoung Kang; Jihan Kim
>
> **备注:** 46 pages, 7 figures, 6 supplementary figures, 1 table, 2 supplementary tables, 1 supplementary note
>
> **摘要:** In this study, we explore the potential of using quantum natural language processing (QNLP) to inverse design metal-organic frameworks (MOFs) with targeted properties. Specifically, by analyzing 450 hypothetical MOF structures consisting of 3 topologies, 10 metal nodes and 15 organic ligands, we categorize these structures into four distinct classes for pore volume and $CO_{2}$ Henry's constant values. We then compare various QNLP models (i.e. the bag-of-words, DisCoCat (Distributional Compositional Categorical), and sequence-based models) to identify the most effective approach to process the MOF dataset. Using a classical simulator provided by the IBM Qiskit, the bag-of-words model is identified to be the optimum model, achieving validation accuracies of 88.6% and 78.0% for binary classification tasks on pore volume and $CO_{2}$ Henry's constant, respectively. Further, we developed multi-class classification models tailored to the probabilistic nature of quantum circuits, with average test accuracies of 92% and 80% across different classes for pore volume and $CO_{2}$ Henry's constant datasets. Finally, the performance of generating MOF with target properties showed accuracies of 93.5% for pore volume and 87% for $CO_{2}$ Henry's constant, respectively. Although our investigation covers only a fraction of the vast MOF search space, it marks a promising first step towards using quantum computing for materials design, offering a new perspective through which to explore the complex landscape of MOFs.
>
---
#### [replaced 030] Think When You Need: Self-Adaptive Chain-of-Thought Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.03234v2](http://arxiv.org/pdf/2504.03234v2)**

> **作者:** Junjie Yang; Ke Lin; Xing Yu
>
> **备注:** Under review
>
> **摘要:** Chain of Thought (CoT) reasoning enhances language models' performance but often leads to inefficient "overthinking" on simple problems. We identify that existing approaches directly penalizing reasoning length fail to account for varying problem complexity. Our approach constructs rewards through length and quality comparisons, guided by theoretical assumptions that jointly enhance solution correctness with conciseness. Moreover, we further demonstrate our method to fuzzy tasks where ground truth is unavailable. Experiments across multiple reasoning benchmarks demonstrate that our method maintains accuracy while generating significantly more concise explanations, effectively teaching models to "think when needed."
>
---
#### [replaced 031] DPO Meets PPO: Reinforced Token Optimization for RLHF
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2404.18922v4](http://arxiv.org/pdf/2404.18922v4)**

> **作者:** Han Zhong; Zikang Shan; Guhao Feng; Wei Xiong; Xinle Cheng; Li Zhao; Di He; Jiang Bian; Liwei Wang
>
> **备注:** ICML 2025
>
> **摘要:** In the classical Reinforcement Learning from Human Feedback (RLHF) framework, Proximal Policy Optimization (PPO) is employed to learn from sparse, sentence-level rewards -- a challenging scenario in traditional deep reinforcement learning. Despite the great successes of PPO in the alignment of large language models, its open-source implementation is still largely sub-optimal. To address these issues, we introduce a framework that models RLHF problems as a Markov decision process (MDP), enabling the capture of fine-grained token-wise information. Under this framework, we introduce an algorithm Reinforced Token Optimization (\texttt{RTO}), which learns the token-wise reward function from preference data and performs policy optimization based on this learned token-wise reward signal. Theoretically, \texttt{RTO} is proven to have the capability of finding the near-optimal policy sample-efficiently. For its practical implementation, \texttt{RTO} innovatively integrates Direct Preference Optimization (DPO) and PPO. DPO, originally derived from sparse sentence rewards, surprisingly provides us with a token-wise characterization of response quality, which is seamlessly incorporated into our subsequent PPO training stage. Extensive experiments demonstrate that \texttt{RTO} performs better than PPO and other direct preference learning algorithms. In particular, RTO outperforms PPO by 7.5 points on the AlpacaEval 2 benchmark and by 4.1 points on Arena-Hard. Our code and models are available at \href{https://github.com/zkshan2002/RTO}{https://github.com/zkshan2002/RTO}.
>
---
#### [replaced 032] ChestX-Reasoner: Advancing Radiology Foundation Models with Reasoning through Step-by-Step Verification
- **分类: cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.20930v2](http://arxiv.org/pdf/2504.20930v2)**

> **作者:** Ziqing Fan; Cheng Liang; Chaoyi Wu; Ya Zhang; Yanfeng Wang; Weidi Xie
>
> **摘要:** Recent advances in reasoning-enhanced large language models (LLMs) and multimodal LLMs (MLLMs) have significantly improved performance in complex tasks, yet medical AI models often overlook the structured reasoning processes inherent in clinical practice. In this work, we present ChestX-Reasoner, a radiology diagnosis MLLM designed to leverage process supervision mined directly from clinical reports, reflecting the step-by-step reasoning followed by radiologists. We construct a large dataset by extracting and refining reasoning chains from routine radiology reports. Our two-stage training framework combines supervised fine-tuning and reinforcement learning guided by process rewards to better align model reasoning with clinical standards. We introduce RadRBench-CXR, a comprehensive benchmark featuring 59K visual question answering samples with 301K clinically validated reasoning steps, and propose RadRScore, a metric evaluating reasoning factuality, completeness, and effectiveness. ChestX-Reasoner outperforms existing medical and general-domain MLLMs in both diagnostic accuracy and reasoning ability, achieving 16%, 5.9%, and 18% improvements in reasoning ability compared to the best medical MLLM, the best general MLLM, and its base model, respectively, as well as 3.3%, 24%, and 27% improvements in outcome accuracy. All resources are open-sourced to facilitate further research in medical reasoning MLLMs.
>
---
#### [replaced 033] Helpful assistant or fruitful facilitator? Investigating how personas affect language model behavior
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.02099v2](http://arxiv.org/pdf/2407.02099v2)**

> **作者:** Pedro Henrique Luz de Araujo; Benjamin Roth
>
> **备注:** 20 pages, 12 figures. Accepted at PLOS One
>
> **摘要:** One way to personalize and steer generations from large language models (LLM) is to assign a persona: a role that describes how the user expects the LLM to behave (e.g., a helpful assistant, a teacher, a woman). This paper investigates how personas affect diverse aspects of model behavior. We assign to seven LLMs 162 personas from 12 categories spanning variables like gender, sexual orientation, and occupation. We prompt them to answer questions from five datasets covering objective (e.g., questions about math and history) and subjective tasks (e.g., questions about beliefs and values). We also compare persona's generations to two baseline settings: a control persona setting with 30 paraphrases of "a helpful assistant" to control for models' prompt sensitivity, and an empty persona setting where no persona is assigned. We find that for all models and datasets, personas show greater variability than the control setting and that some measures of persona behavior generalize across models.
>
---
#### [replaced 034] Analyzing the Effect of Linguistic Similarity on Cross-Lingual Transfer: Tasks and Experimental Setups Matter
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.14491v3](http://arxiv.org/pdf/2501.14491v3)**

> **作者:** Verena Blaschke; Masha Fedzechkina; Maartje ter Hoeve
>
> **备注:** ACL Findings 2025
>
> **摘要:** Cross-lingual transfer is a popular approach to increase the amount of training data for NLP tasks in a low-resource context. However, the best strategy to decide which cross-lingual data to include is unclear. Prior research often focuses on a small set of languages from a few language families and/or a single task. It is still an open question how these findings extend to a wider variety of languages and tasks. In this work, we analyze cross-lingual transfer for 263 languages from a wide variety of language families. Moreover, we include three popular NLP tasks: POS tagging, dependency parsing, and topic classification. Our findings indicate that the effect of linguistic similarity on transfer performance depends on a range of factors: the NLP task, the (mono- or multilingual) input representations, and the definition of linguistic similarity.
>
---
#### [replaced 035] Can LLMs Maintain Fundamental Abilities under KV Cache Compression?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.01941v2](http://arxiv.org/pdf/2502.01941v2)**

> **作者:** Xiang Liu; Zhenheng Tang; Hong Chen; Peijie Dong; Zeyu Li; Xiuze Zhou; Bo Li; Xuming Hu; Xiaowen Chu
>
> **备注:** 25 pages
>
> **摘要:** This paper investigates an underexplored challenge in large language models (LLMs): the impact of KV cache compression methods on LLMs' fundamental capabilities. Although existing methods achieve impressive compression ratios on long-context benchmarks, their effects on core model capabilities remain understudied. We present a comprehensive benchmark KVFundaBench to systematically evaluate the effects of KV cache compression across diverse fundamental LLM capabilities, spanning world knowledge, commonsense reasoning, arithmetic reasoning, code generation, safety, and long-context understanding and generation.Our analysis reveals serval key findings: (1) \textit{Task-Dependent Degradation}; (2) \textit{Model-Type Robustness} (3) \textit{Prompt Length Vulnerability}; (4) \textit{Chunk-Level Superiority}; (5) \textit{Prompt-Gain Sensitivity}; (6) \textit{Long-Context Generation Sensitivity}. Based on our analysis of attention patterns and cross-task compression performance, we propose ShotKV, a novel compression approach that distinctly handles prefill and decoding phases while maintaining shot-level semantic coherence. Empirical results show that ShotKV achieves $9\%$-$18\%$ performance improvements on long-context generation tasks under aggressive compression ratios.
>
---
#### [replaced 036] Parameter Efficient Fine-tuning via Explained Variance Adaptation
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.07170v4](http://arxiv.org/pdf/2410.07170v4)**

> **作者:** Fabian Paischer; Lukas Hauzenberger; Thomas Schmied; Benedikt Alkin; Marc Peter Deisenroth; Sepp Hochreiter
>
> **备注:** 9 pages + references and appendix, code available at https://github.com/ml-jku/EVA
>
> **摘要:** Foundation models (FMs) are pre-trained on large-scale datasets and then fine-tuned for a specific downstream task. The most common fine-tuning method is to update pretrained weights via low-rank adaptation (LoRA). Existing initialization strategies for LoRA often rely on singular value decompositions (SVD) of gradients or weight matrices. However, they do not provably maximize the expected gradient signal, which is critical for fast adaptation. To this end, we introduce Explained Variance Adaptation (EVA), an initialization scheme that uses the directions capturing the most activation variance, provably maximizing the expected gradient signal and accelerating fine-tuning. EVA performs incremental SVD on minibatches of activation vectors and selects the right-singular vectors for initialization once they converged. Further, by selecting the directions that capture the most activation-variance for a given rank budget, EVA accommodates adaptive ranks that reduce the number of trainable parameters, while maintaining or improving downstream performance. We apply EVA to a variety of fine-tuning tasks as language generation and understanding, image classification, and reinforcement learning. EVA exhibits faster convergence than competitors and achieves the highest average score across a multitude of tasks per domain while reducing the number of trainable parameters through rank redistribution.
>
---
#### [replaced 037] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.06141v3](http://arxiv.org/pdf/2412.06141v3)**

> **作者:** Kangyu Zhu; Peng Xia; Yun Li; Hongtu Zhu; Sheng Wang; Huaxiu Yao
>
> **备注:** ICML 2025
>
> **摘要:** The advancement of Large Vision-Language Models (LVLMs) has propelled their application in the medical field. However, Medical LVLMs (Med-LVLMs) encounter factuality challenges due to modality misalignment, where the models prioritize textual knowledge over visual input, leading to hallucinations that contradict information in medical images. Previous attempts to enhance modality alignment in Med-LVLMs through preference optimization have inadequately mitigated clinical relevance in preference data, making these samples easily distinguishable and reducing alignment effectiveness. To address this challenge, we propose MMedPO, a novel multimodal medical preference optimization approach that considers the clinical relevance of preference samples to enhance Med-LVLM alignment. MMedPO curates multimodal preference data by introducing two types of dispreference: (1) plausible hallucinations injected through target Med-LVLMs or GPT-4o to produce medically inaccurate responses, and (2) lesion region neglect achieved through local lesion-noising, disrupting visual understanding of critical areas. We then calculate clinical relevance for each sample based on scores from multiple Med-LLMs and visual tools, and integrate these scores into the preference optimization process as weights, enabling effective alignment. Our experiments demonstrate that MMedPO significantly enhances factual accuracy in Med-LVLMs, achieving substantial improvements over existing preference optimization methods by averaging 14.2% and 51.7% across the Med-VQA and report generation tasks. Our code are available in https://github.com/aiming-lab/MMedPO.
>
---
#### [replaced 038] MoHAVE: Mixture of Hierarchical Audio-Visual Experts for Robust Speech Recognition
- **分类: eess.AS; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.10447v2](http://arxiv.org/pdf/2502.10447v2)**

> **作者:** Sungnyun Kim; Kangwook Jang; Sangmin Bae; Sungwoo Cho; Se-Young Yun
>
> **备注:** Accepted to ICML 2025
>
> **摘要:** Audio-visual speech recognition (AVSR) has become critical for enhancing speech recognition in noisy environments by integrating both auditory and visual modalities. However, existing AVSR systems struggle to scale up without compromising computational efficiency. In this study, we introduce MoHAVE (Mixture of Hierarchical Audio-Visual Experts), a novel robust AVSR framework designed to address these scalability constraints. By leveraging a Mixture-of-Experts (MoE) architecture, MoHAVE activates modality-specific expert groups, ensuring dynamic adaptation to various audio-visual inputs with minimal computational overhead. Key contributions of MoHAVE include: (1) a sparse MoE framework that efficiently scales AVSR model capacity, (2) a hierarchical gating mechanism that dynamically utilizes the expert groups based on input context, enhancing adaptability and robustness, and (3) remarkable performance across robust AVSR benchmarks, including LRS3 and MuAViC transcription and translation tasks, setting a new standard for scalable speech recognition systems.
>
---
#### [replaced 039] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.03122v4](http://arxiv.org/pdf/2503.03122v4)**

> **作者:** Zichao Li; Xueru Wen; Jie Lou; Yuqiu Ji; Yaojie Lu; Xianpei Han; Debing Zhang; Le Sun
>
> **备注:** ICML 2025
>
> **摘要:** Multimodal Reward Models (MM-RMs) are crucial for aligning Large Language Models (LLMs) with human preferences, particularly as LLMs increasingly interact with multimodal data. However, we find that MM-RMs trained on existing datasets often struggle to generalize to out-of-distribution data due to their reliance on unimodal spurious correlations, primarily text-only shortcuts within the training distribution, which prevents them from leveraging true multimodal reward functions. To address this, we introduce a Shortcut-aware MM-RM learning algorithm that mitigates this issue by dynamically reweighting training samples, shifting the distribution toward better multimodal understanding, and reducing dependence on unimodal spurious correlations. Our experiments demonstrate significant improvements in generalization, downstream task performance, and scalability, establishing a more robust framework for multimodal reward modeling.
>
---
#### [replaced 040] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14846v2](http://arxiv.org/pdf/2502.14846v2)**

> **作者:** Yue Yang; Ajay Patel; Matt Deitke; Tanmay Gupta; Luca Weihs; Andrew Head; Mark Yatskar; Chris Callison-Burch; Ranjay Krishna; Aniruddha Kembhavi; Christopher Clark
>
> **备注:** Published in ACL 2025, project page: https://yueyang1996.github.io/cosyn/
>
> **摘要:** Reasoning about images with rich text, such as charts and documents, is a critical application of vision-language models (VLMs). However, VLMs often struggle in these domains due to the scarcity of diverse text-rich vision-language data. To address this challenge, we present CoSyn, a framework that leverages the coding capabilities of text-only large language models (LLMs) to automatically create synthetic text-rich multimodal data. Given input text describing a target domain (e.g., "nutrition fact labels"), CoSyn prompts an LLM to generate code (Python, HTML, LaTeX, etc.) for rendering synthetic images. With the underlying code as textual representations of the synthetic images, CoSyn can generate high-quality instruction-tuning data, again relying on a text-only LLM. Using CoSyn, we constructed a dataset comprising 400K images and 2.7M rows of vision-language instruction-tuning data. Comprehensive experiments on seven benchmarks demonstrate that models trained on our synthetic data achieve state-of-the-art performance among competitive open-source models, including Llama 3.2, and surpass proprietary models such as GPT-4V and Gemini 1.5 Flash. Furthermore, CoSyn can produce synthetic pointing data, enabling VLMs to ground information within input images, showcasing its potential for developing multimodal agents capable of acting in real-world environments.
>
---
#### [replaced 041] SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network
- **分类: cs.NE; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.06488v4](http://arxiv.org/pdf/2310.06488v4)**

> **作者:** Changze Lv; Tianlong Li; Wenhao Liu; Yufei Gu; Jianhan Xu; Cenyuan Zhang; Muling Wu; Xiaoqing Zheng; Xuanjing Huang
>
> **摘要:** Spiking Neural Networks (SNNs) have emerged as a promising alternative to conventional Artificial Neural Networks (ANNs), demonstrating comparable performance in both visual and linguistic tasks while offering the advantage of improved energy efficiency. Despite these advancements, the integration of linguistic and visual features into a unified representation through spike trains poses a significant challenge, and the application of SNNs to multimodal scenarios remains largely unexplored. This paper presents SpikeCLIP, a novel framework designed to bridge the modality gap in spike-based computation. Our approach employs a two-step recipe: an ``alignment pre-training'' to align features across modalities, followed by a ``dual-loss fine-tuning'' to refine the model's performance. Extensive experiments reveal that SNNs achieve results on par with ANNs while substantially reducing energy consumption across various datasets commonly used for multimodal model evaluation. Furthermore, SpikeCLIP maintains robust image classification capabilities, even when dealing with classes that fall outside predefined categories. This study marks a significant advancement in the development of energy-efficient and biologically plausible multimodal learning systems. Our code is available at https://github.com/Lvchangze/SpikeCLIP.
>
---
#### [replaced 042] Uncertainty quantification in fine-tuned LLMs using LoRA ensembles
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2402.12264v2](http://arxiv.org/pdf/2402.12264v2)**

> **作者:** Oleksandr Balabanov; Hampus Linander
>
> **备注:** Accepted for ICLR2025 Workshop "Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI"
>
> **摘要:** Fine-tuning large language models can improve task specific performance, although a general understanding of what the fine-tuned model has learned, forgotten and how to trust its predictions is still missing. We derive principled uncertainty quantification for fine-tuned LLMs with posterior approximations using computationally efficient low-rank adaptation ensembles. We analyze three common multiple-choice datasets using low-rank adaptation ensembles based on Mistral-7b, and draw quantitative and qualitative conclusions on their perceived complexity and balance between retained prior knowledge and domain specific adaptation during and after fine-tuning. We identify unexpected retention of acquired knowledge during fine-tuning in the overfitting regime.
>
---
#### [replaced 043] Streaming Sequence Transduction through Dynamic Compression
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2402.01172v3](http://arxiv.org/pdf/2402.01172v3)**

> **作者:** Weiting Tan; Yunmo Chen; Tongfei Chen; Guanghui Qin; Haoran Xu; Heidi C. Zhang; Benjamin Van Durme; Philipp Koehn
>
> **备注:** IWSLT 2025
>
> **摘要:** We introduce STAR (Stream Transduction with Anchor Representations), a novel Transformer-based model designed for efficient sequence-to-sequence transduction over streams. STAR dynamically segments input streams to create compressed anchor representations, achieving nearly lossless compression (12x) in Automatic Speech Recognition (ASR) and outperforming existing methods. Moreover, STAR demonstrates superior segmentation and latency-quality trade-offs in simultaneous speech-to-text tasks, optimizing latency, memory footprint, and quality.
>
---
#### [replaced 044] Robust and Minimally Invasive Watermarking for EaaS
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.17552v3](http://arxiv.org/pdf/2410.17552v3)**

> **作者:** Zongqi Wang; Baoyuan Wu; Jingyuan Deng; Yujiu Yang
>
> **备注:** Accepted by ACL 2025
>
> **摘要:** Embeddings as a Service (EaaS) is emerging as a crucial role in AI applications. Unfortunately, EaaS is vulnerable to model extraction attacks, highlighting the urgent need for copyright protection. Although some preliminary works propose applying embedding watermarks to protect EaaS, recent research reveals that these watermarks can be easily removed. Hence, it is crucial to inject robust watermarks resistant to watermark removal attacks. Existing watermarking methods typically inject a target embedding into embeddings through linear interpolation when the text contains triggers. However, this mechanism results in each watermarked embedding having the same component, which makes the watermark easy to identify and eliminate. Motivated by this, in this paper, we propose a novel embedding-specific watermarking (ESpeW) mechanism to offer robust copyright protection for EaaS. Our approach involves injecting unique, yet readily identifiable watermarks into each embedding. Watermarks inserted by ESpeW are designed to maintain a significant distance from one another and to avoid sharing common components, thus making it significantly more challenging to remove the watermarks. Moreover, ESpeW is minimally invasive, as it reduces the impact on embeddings to less than 1\%, setting a new milestone in watermarking for EaaS. Extensive experiments on four popular datasets demonstrate that ESpeW can even watermark successfully against a highly aggressive removal strategy without sacrificing the quality of embeddings.
>
---
#### [replaced 045] Let's Be Self-generated via Step by Step: A Curriculum Learning Approach to Automated Reasoning with Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.21728v3](http://arxiv.org/pdf/2410.21728v3)**

> **作者:** Kangyang Luo; Zichen Ding; Zhenmin Weng; Lingfeng Qiao; Meng Zhao; Xiang Li; Di Yin; Jinlong Shu
>
> **备注:** Accepted by ACL2025(Findings)
>
> **摘要:** While Chain of Thought (CoT) prompting approaches have significantly consolidated the reasoning capabilities of large language models (LLMs), they still face limitations that require extensive human effort or have performance needs to be improved. Existing endeavors have focused on bridging these gaps; however, these approaches either hinge on external data and cannot completely eliminate manual effort, or they fall short in effectively directing LLMs to generate high-quality exemplary prompts. To address the said pitfalls, we propose a novel prompt approach for automatic reasoning named \textbf{LBS3}, inspired by curriculum learning which better reflects human learning habits. Specifically, LBS3 initially steers LLMs to recall easy-to-hard proxy queries that are pertinent to the target query. Following this, it invokes a progressive strategy that utilizes exemplary prompts stemmed from easy-proxy queries to direct LLMs in solving hard-proxy queries, enabling the high-quality of the proxy solutions. Finally, our extensive experiments in various reasoning-intensive tasks with varying open- and closed-source LLMs show that LBS3 achieves strongly competitive performance compared to the SOTA baselines.
>
---
#### [replaced 046] How to Construct Random Unitaries
- **分类: quant-ph; cs.CC; cs.CL; math-ph; math.MP**

- **链接: [http://arxiv.org/pdf/2410.10116v3](http://arxiv.org/pdf/2410.10116v3)**

> **作者:** Fermi Ma; Hsin-Yuan Huang
>
> **备注:** 76 pages; moved grant acknowledgments to acknowledgments section
>
> **摘要:** The existence of pseudorandom unitaries (PRUs) -- efficient quantum circuits that are computationally indistinguishable from Haar-random unitaries -- has been a central open question, with significant implications for cryptography, complexity theory, and fundamental physics. In this work, we close this question by proving that PRUs exist, assuming that any quantum-secure one-way function exists. We establish this result for both (1) the standard notion of PRUs, which are secure against any efficient adversary that makes queries to the unitary $U$, and (2) a stronger notion of PRUs, which are secure even against adversaries that can query both the unitary $U$ and its inverse $U^\dagger$. In the process, we prove that any algorithm that makes queries to a Haar-random unitary can be efficiently simulated on a quantum computer, up to inverse-exponential trace distance.
>
---
#### [replaced 047] A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.10717v2](http://arxiv.org/pdf/2505.10717v2)**

> **作者:** Jean-Philippe Corbeil; Amin Dada; Jean-Michel Attendu; Asma Ben Abacha; Alessandro Sordoni; Lucas Caccia; François Beaulieu; Thomas Lin; Jens Kleesiek; Paul Vozila
>
> **摘要:** High computation costs and latency of large language models such as GPT-4 have limited their deployment in clinical settings. Small language models (SLMs) offer a cost-effective alternative, but their limited capacity requires biomedical domain adaptation, which remains challenging. An additional bottleneck is the unavailability and high sensitivity of clinical data. To address these challenges, we propose a novel framework for adapting SLMs into high-performing clinical models. We introduce the MediPhi collection of 3.8B-parameter SLMs developed with our novel framework: pre-instruction tuning of experts on relevant medical and clinical corpora (PMC, Medical Guideline, MedWiki, etc.), model merging, and clinical-tasks alignment. To cover most clinical tasks, we extended the CLUE benchmark to CLUE+, doubling its size. Our expert models deliver relative improvements on this benchmark over the base model without any task-specific fine-tuning: 64.3% on medical entities, 49.5% on radiology reports, and 44% on ICD-10 coding (outperforming GPT-4-0125 by 14%). We unify the expert models into MediPhi via model merging, preserving gains across benchmarks. Furthermore, we built the MediFlow collection, a synthetic dataset of 2.5 million high-quality instructions on 14 medical NLP tasks, 98 fine-grained document types, and JSON format support. Alignment of MediPhi using supervised fine-tuning and direct preference optimization achieves further gains of 18.9% on average.
>
---
#### [replaced 048] Mobile-Agent-V: A Video-Guided Approach for Effortless and Efficient Operational Knowledge Injection in Mobile Automation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13887v2](http://arxiv.org/pdf/2505.13887v2)**

> **作者:** Junyang Wang; Haiyang Xu; Xi Zhang; Ming Yan; Ji Zhang; Fei Huang; Jitao Sang
>
> **备注:** I was trying to update arXiv:2502.17110 but accidentally published a new work
>
> **摘要:** The exponential rise in mobile device usage necessitates streamlined automation for effective task management, yet many AI frameworks fall short due to inadequate operational expertise. While manually written knowledge can bridge this gap, it is often burdensome and inefficient. We introduce Mobile-Agent-V, an innovative framework that utilizes video as a guiding tool to effortlessly and efficiently inject operational knowledge into mobile automation processes. By deriving knowledge directly from video content, Mobile-Agent-V eliminates manual intervention, significantly reducing the effort and time required for knowledge acquisition. To rigorously evaluate this approach, we propose Mobile-Knowledge, a benchmark tailored to assess the impact of external knowledge on mobile agent performance. Our experimental findings demonstrate that Mobile-Agent-V enhances performance by 36% compared to existing methods, underscoring its effortless and efficient advantages in mobile automation.
>
---
#### [replaced 049] Towards Reliable and Interpretable Traffic Crash Pattern Prediction and Safety Interventions Using Customized Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.12545v2](http://arxiv.org/pdf/2505.12545v2)**

> **作者:** Yang Zhao; Pu Wang; Yibo Zhao; Hongru Du; Hao Frank Yang
>
> **备注:** Last revised 13 Feb 2025. Under review in Nature portfolio
>
> **摘要:** Predicting crash events is crucial for understanding crash distributions and their contributing factors, thereby enabling the design of proactive traffic safety policy interventions. However, existing methods struggle to interpret the complex interplay among various sources of traffic crash data, including numeric characteristics, textual reports, crash imagery, environmental conditions, and driver behavior records. As a result, they often fail to capture the rich semantic information and intricate interrelationships embedded in these diverse data sources, limiting their ability to identify critical crash risk factors. In this research, we propose TrafficSafe, a framework that adapts LLMs to reframe crash prediction and feature attribution as text-based reasoning. A multi-modal crash dataset including 58,903 real-world reports together with belonged infrastructure, environmental, driver, and vehicle information is collected and textualized into TrafficSafe Event Dataset. By customizing and fine-tuning LLMs on this dataset, the TrafficSafe LLM achieves a 42% average improvement in F1-score over baselines. To interpret these predictions and uncover contributing factors, we introduce TrafficSafe Attribution, a sentence-level feature attribution framework enabling conditional risk analysis. Findings show that alcohol-impaired driving is the leading factor in severe crashes, with aggressive and impairment-related behaviors having nearly twice the contribution for severe crashes compared to other driver behaviors. Furthermore, TrafficSafe Attribution highlights pivotal features during model training, guiding strategic crash data collection for iterative performance improvements. The proposed TrafficSafe offers a transformative leap in traffic safety research, providing a blueprint for translating advanced AI technologies into responsible, actionable, and life-saving outcomes.
>
---
#### [replaced 050] dMel: Speech Tokenization made Simple
- **分类: cs.CL; cs.AI; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2407.15835v3](http://arxiv.org/pdf/2407.15835v3)**

> **作者:** Richard He Bai; Tatiana Likhomanenko; Ruixiang Zhang; Zijin Gu; Zakaria Aldeneh; Navdeep Jaitly
>
> **备注:** preprint
>
> **摘要:** Large language models have revolutionized natural language processing by leveraging self-supervised pretraining on vast textual data. Inspired by this success, researchers have investigated various compression-based speech tokenization methods to discretize continuous speech signals, enabling the application of language modeling techniques to discrete tokens. However, audio compressor introduces additional complexity and computational cost, and often fail on out-of-domain audio signals. In this work, we introduce a novel speech representation (dmel) that discretizes mel-filterbank channels into intensity bins, creating a simpler yet more effective representation compared to existing speech tokenization methods. Our approach demonstrates superior performance in preserving audio content, robustness to out-of-domain data, and offers a training-free, natural, and streamable representation. To address the high-dimensional nature of log-mel spectrograms, we propose an efficient parallel encoding and decoding method for high-dimensional tokens using an LM-style transformer architecture. This innovation enables us to develop RichTTS and RichASR, two models sharing the same architecture while achieving comparable or better results than specialized existing methods. Our results demonstrate the effectiveness of dmel in achieving high performance on both speech synthesis and recognition tasks within a unified framework, paving the way for efficient and effective joint modeling of speech and text.
>
---
#### [replaced 051] Stay Focused: Problem Drift in Multi-Agent Debate
- **分类: cs.CL; A.1; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.19559v2](http://arxiv.org/pdf/2502.19559v2)**

> **作者:** Jonas Becker; Lars Benedikt Kaesberg; Andreas Stephan; Jan Philip Wahle; Terry Ruas; Bela Gipp
>
> **备注:** 34 pages, 10 figures, 8 tables
>
> **摘要:** Multi-agent debate - multiple instances of large language models discussing problems in turn-based interaction - has shown promise for solving knowledge and reasoning tasks. However, these methods show limitations when solving complex problems that require longer reasoning chains. We analyze how multi-agent debate over multiple turns drifts away from the initial problem, thus harming task performance. We define this phenomenon as problem drift and quantify its presence across ten tasks (i.e., three generative, three knowledge, three reasoning, and one instruction-following task). To identify the reasons for this issue, eight human experts analyze 170 multi-agent discussions suffering from problem drift. We find the most common issues related to this drift are the lack of progress (35% of cases), low-quality feedback (26% of cases), and a lack of clarity (25% of cases). To address problem drift, we propose DRIFTJudge, an LLM-as-a-judge method, to detect problem drift at test-time. We also propose DRIFTPolicy, a method that mitigates problem drift cases to improve task performance. Our study is a step toward understanding a key limitation of multi-agent debate, highlighting why longer debates can harm task performance and how problem drift could be addressed.
>
---
#### [replaced 052] Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in Large Language Models
- **分类: cs.CL; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2505.02847v3](http://arxiv.org/pdf/2505.02847v3)**

> **作者:** Bang Zhang; Ruotian Ma; Qingxuan Jiang; Peisong Wang; Jiaqi Chen; Zheng Xie; Xingyu Chen; Yue Wang; Fanghua Ye; Jian Li; Yifan Yang; Zhaopeng Tu; Xiaolong Li
>
> **备注:** code: https://github.com/Tencent/digitalhuman/tree/main/SAGE
>
> **摘要:** Assessing how well a large language model (LLM) understands human, rather than merely text, remains an open challenge. To bridge the gap, we introduce Sentient Agent as a Judge (SAGE), an automated evaluation framework that measures an LLM's higher-order social cognition. SAGE instantiates a Sentient Agent that simulates human-like emotional changes and inner thoughts during interaction, providing a more realistic evaluation of the tested model in multi-turn conversations. At every turn, the agent reasons about (i) how its emotion changes, (ii) how it feels, and (iii) how it should reply, yielding a numerical emotion trajectory and interpretable inner thoughts. Experiments on 100 supportive-dialogue scenarios show that the final Sentient emotion score correlates strongly with Barrett-Lennard Relationship Inventory (BLRI) ratings and utterance-level empathy metrics, validating psychological fidelity. We also build a public Sentient Leaderboard covering 18 commercial and open-source models that uncovers substantial gaps (up to 4x) between frontier systems (GPT-4o-Latest, Gemini2.5-Pro) and earlier baselines, gaps not reflected in conventional leaderboards (e.g., Arena). SAGE thus provides a principled, scalable and interpretable tool for tracking progress toward genuinely empathetic and socially adept language agents.
>
---
#### [replaced 053] Automated Visualization Code Synthesis via Multi-Path Reasoning and Feedback-Driven Optimization
- **分类: cs.SE; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2502.11140v2](http://arxiv.org/pdf/2502.11140v2)**

> **作者:** Wonduk Seo; Seungyong Lee; Daye Kang; Hyunjin An; Zonghao Yuan; Seunghyun Lee
>
> **备注:** 16 pages, 5 figures, 3 tables
>
> **摘要:** Rapid advancements in Large Language Models (LLMs) have accelerated their integration into automated visualization code generation applications. Despite advancements through few-shot prompting and query expansion, existing methods remain limited in handling ambiguous and complex queries, thereby requiring manual intervention. To overcome these limitations, we propose VisPath: a Multi-Path Reasoning and Feedback-Driven Optimization Framework for Visualization Code Generation. VisPath handles underspecified queries through structured, multi-stage processing. It begins by reformulating the user input via Chain-of-Thought (CoT) prompting, which refers to the initial query while generating multiple extended queries in parallel, enabling the LLM to capture diverse interpretations of the user intent. These queries then generate candidate visualization scripts, which are executed to produce diverse images. By assessing the visual quality and correctness of each output, VisPath generates targeted feedback that is aggregated to synthesize an optimal final result. Extensive experiments on widely-used benchmarks including MatPlotBench and the Qwen-Agent Code Interpreter Benchmark show that VisPath outperforms state-of-the-art methods, offering a more reliable solution for AI-driven visualization code generation.
>
---
#### [replaced 054] Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01563v4](http://arxiv.org/pdf/2502.01563v4)**

> **作者:** Mingyu Jin; Kai Mei; Wujiang Xu; Mingjie Sun; Ruixiang Tang; Mengnan Du; Zirui Liu; Yongfeng Zhang
>
> **备注:** International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Large language models (LLMs) have achieved remarkable success in contextual knowledge understanding. In this paper, we show that these concentrated massive values consistently emerge in specific regions of attention queries (Q) and keys (K) while not having such patterns in values (V) in various modern transformer-based LLMs (Q, K, and V mean the representations output by the query, key, and value layers respectively). Through extensive experiments, we further demonstrate that these massive values play a critical role in interpreting contextual knowledge (knowledge obtained from the current context window) rather than in retrieving parametric knowledge stored within the model's parameters. Our further investigation of quantization strategies reveals that ignoring these massive values leads to a pronounced drop in performance on tasks requiring rich contextual understanding, aligning with our analysis. Finally, we trace the emergence of concentrated massive values and find that such concentration is caused by Rotary Positional Encoding (RoPE), which has appeared since the first layers. These findings shed new light on how Q and K operate in LLMs and offer practical insights for model design and optimization. The Code is Available at https://github.com/MingyuJ666/Rope_with_LLM.
>
---
#### [replaced 055] MCIP: Protecting MCP Safety via Model Contextual Integrity Protocol
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14590v2](http://arxiv.org/pdf/2505.14590v2)**

> **作者:** Huihao Jing; Haoran Li; Wenbin Hu; Qi Hu; Heli Xu; Tianshu Chu; Peizhao Hu; Yangqiu Song
>
> **备注:** 17 pages
>
> **摘要:** As Model Context Protocol (MCP) introduces an easy-to-use ecosystem for users and developers, it also brings underexplored safety risks. Its decentralized architecture, which separates clients and servers, poses unique challenges for systematic safety analysis. This paper proposes a novel framework to enhance MCP safety. Guided by the MAESTRO framework, we first analyze the missing safety mechanisms in MCP, and based on this analysis, we propose the Model Contextual Integrity Protocol (MCIP), a refined version of MCP that addresses these gaps. Next, we develop a fine-grained taxonomy that captures a diverse range of unsafe behaviors observed in MCP scenarios. Building on this taxonomy, we develop benchmark and training data that support the evaluation and improvement of LLMs' capabilities in identifying safety risks within MCP interactions. Leveraging the proposed benchmark and training data, we conduct extensive experiments on state-of-the-art LLMs. The results highlight LLMs' vulnerabilities in MCP interactions and demonstrate that our approach substantially improves their safety performance.
>
---
#### [replaced 056] GiFT: Gibbs Fine-Tuning for Code Generation
- **分类: cs.LG; cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2502.11466v2](http://arxiv.org/pdf/2502.11466v2)**

> **作者:** Haochen Li; Wanjin Feng; Xin Zhou; Zhiqi Shen
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Training Large Language Models (LLMs) with synthetic data is a prevalent practice in code generation. A key approach is self-training, where LLMs are iteratively trained on self-generated correct code snippets. In this case, the self-generated codes are drawn from a conditional distribution, conditioned on a specific seed description. However, the seed description is not the only valid representation that aligns with its intended meaning. With all valid descriptions and codes forming a joint space, codes drawn from the conditional distribution would lead to an underrepresentation of the full description-code space. As such, we propose Gibbs Fine-Tuning (GiFT), a novel self-training method inspired by Gibbs sampling. GiFT allows self-generated data to be drawn from the marginal distribution of the joint space, thereby mitigating the biases inherent in conditional sampling. We provide a theoretical analysis demonstrating the potential benefits of fine-tuning LLMs with code derived from the marginal distribution. Furthermore, we propose a perplexity-based code selection method to mitigate the imbalanced long-tail distribution of the self-generated codes. Empirical evaluation of two LLMs across four datasets demonstrates that GiFT achieves superior performance, particularly on more challenging benchmarks. Source code is available at https://github.com/Alex-HaochenLi/GiFT.
>
---
#### [replaced 057] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2501.16344v3](http://arxiv.org/pdf/2501.16344v3)**

> **作者:** Rajath Rao; Adithya Ganesan; Oscar Kjell; Jonah Luby; Akshay Raghavan; Scott Feltman; Whitney Ringwald; Ryan L. Boyd; Benjamin Luft; Camilo Ruggero; Neville Ryant; Roman Kotov; H. Andrew Schwartz
>
> **备注:** 16 pages, 8 figures, ACL 2025
>
> **摘要:** Current speech encoding pipelines often rely on an additional text-based LM to get robust representations of human communication, even though SotA speech-to-text models often have a LM within. This work proposes an approach to improve the LM within an audio model such that the subsequent text-LM is unnecessary. We introduce WhiSPA (Whisper with Semantic and Psychological Alignment), which leverages a novel audio training objective: contrastive loss with a language model embedding as a teacher. Using over 500k speech segments from mental health audio interviews, we evaluate the utility of aligning Whisper's latent space with semantic representations from a text autoencoder (SBERT) and lexically derived embeddings of basic psychological dimensions: emotion and personality. Over self-supervised affective tasks and downstream psychological tasks, WhiSPA surpasses current speech encoders, achieving an average error reduction of 73.4% and 83.8%, respectively. WhiSPA demonstrates that it is not always necessary to run a subsequent text LM on speech-to-text output in order to get a rich psychological representation of human communication.
>
---
#### [replaced 058] How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond
- **分类: cs.CL; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2501.05714v3](http://arxiv.org/pdf/2501.05714v3)**

> **作者:** Chen Huang; Yang Deng; Wenqiang Lei; Jiancheng Lv; Tat-Seng Chua; Jimmy Xiangji Huang
>
> **备注:** ACL 2025 Main paper
>
> **摘要:** With the advancement of large language models (LLMs), intelligent models have evolved from mere tools to autonomous agents with their own goals and strategies for cooperating with humans. This evolution has birthed a novel paradigm in NLP, i.e., human-model cooperation, that has yielded remarkable progress in numerous NLP tasks in recent years. In this paper, we take the first step to present a thorough review of human-model cooperation, exploring its principles, formalizations, and open challenges. In particular, we introduce a new taxonomy that provides a unified perspective to summarize existing approaches. Also, we discuss potential frontier areas and their corresponding challenges. We regard our work as an entry point, paving the way for more breakthrough research in this regard.
>
---
#### [replaced 059] Exploring the Robustness of Language Models for Tabular Question Answering via Attention Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.12719v3](http://arxiv.org/pdf/2406.12719v3)**

> **作者:** Kushal Raj Bhandari; Sixue Xing; Soham Dan; Jianxi Gao
>
> **摘要:** Large Language Models (LLMs), already shown to ace various text comprehension tasks, have also remarkably been shown to tackle table comprehension tasks without specific training. Building on earlier studies of LLMs for tabular tasks, we probe how in-context learning (ICL), model scale, instruction tuning, and domain bias affect Tabular QA (TQA) robustness by testing LLMs, under diverse augmentations and perturbations, on diverse domains: Wikipedia-based $\textbf{WTQ}$, financial $\textbf{TAT-QA}$, and scientific $\textbf{SCITAB}$. Although instruction tuning and larger, newer LLMs deliver stronger, more robust TQA performance, data contamination and reliability issues, especially on $\textbf{WTQ}$, remain unresolved. Through an in-depth attention analysis, we reveal a strong correlation between perturbation-induced shifts in attention dispersion and the drops in performance, with sensitivity peaking in the model's middle layers. We highlight the need for improved interpretable methodologies to develop more reliable LLMs for table comprehension.
>
---
#### [replaced 060] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings
- **分类: cs.CL; cs.CR; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.12562v2](http://arxiv.org/pdf/2502.12562v2)**

> **作者:** Weikai Lu; Hao Peng; Huiping Zhuang; Cen Chen; Ziqian Zeng
>
> **备注:** Accepted in ACL 2025 Main Track
>
> **摘要:** Multimodal Large Language Models (MLLMs) have serious security vulnerabilities.While safety alignment using multimodal datasets consisting of text and data of additional modalities can effectively enhance MLLM's security, it is costly to construct these datasets. Existing low-resource security alignment methods, including textual alignment, have been found to struggle with the security risks posed by additional modalities. To address this, we propose Synthetic Embedding augmented safety Alignment (SEA), which optimizes embeddings of additional modality through gradient updates to expand textual datasets. This enables multimodal safety alignment training even when only textual data is available. Extensive experiments on image, video, and audio-based MLLMs demonstrate that SEA can synthesize a high-quality embedding on a single RTX3090 GPU within 24 seconds. SEA significantly improves the security of MLLMs when faced with threats from additional modalities. To assess the security risks introduced by video and audio, we also introduced a new benchmark called VA-SafetyBench. High attack success rates across multiple MLLMs validate its challenge. Our code and data will be available at https://github.com/ZeroNLP/SEA.
>
---
#### [replaced 061] Efficient Shapley Value-based Non-Uniform Pruning of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.01731v3](http://arxiv.org/pdf/2505.01731v3)**

> **作者:** Chuan Sun; Han Yu; Lizhen Cui; Xiaoxiao Li
>
> **摘要:** Pruning large language models (LLMs) is a promising solution for reducing model sizes and computational complexity while preserving performance. Traditional layer-wise pruning methods often adopt a uniform sparsity approach across all layers, which leads to suboptimal performance due to the varying significance of individual transformer layers within the model not being accounted for. To this end, we propose the Shapley Value-based Non-Uniform Pruning (SV-NUP) method for LLMs. This approach quantifies the contribution of each transformer layer to the overall model performance, enabling the assignment of tailored pruning budgets to different layers to retain critical parameters. To further improve efficiency, we design the Sliding Window-based Shapley Value approximation method. It substantially reduces computational overhead compared to exact SV calculation methods. Extensive experiments on various LLMs including LLaMA-v1, LLaMA-v2 and OPT demonstrate the effectiveness of the proposed approach. The results reveal that non-uniform pruning significantly enhances the performance of pruned models. Notably, SV-NUP achieves a reduction in perplexity (PPL) of 18.01% and 19.55% on LLaMA-7B and LLaMA-13B, respectively, compared to SparseGPT at 70% sparsity.
>
---
#### [replaced 062] Mitigating Subgroup Disparities in Multi-Label Speech Emotion Recognition: A Pseudo-Labeling and Unsupervised Learning Approach
- **分类: eess.AS; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.14449v2](http://arxiv.org/pdf/2505.14449v2)**

> **作者:** Yi-Cheng Lin; Huang-Cheng Chou; Hung-yi Lee
>
> **备注:** Accepted by InterSpeech 2025. 7 pages including 2 pages of appendix
>
> **摘要:** While subgroup disparities and performance bias are increasingly studied in computational research, fairness in categorical Speech Emotion Recognition (SER) remains underexplored. Existing methods often rely on explicit demographic labels, which are difficult to obtain due to privacy concerns. To address this limitation, we introduce an Implicit Demography Inference (IDI) module that leverages pseudo-labeling from a pre-trained model and unsupervised learning using k-means clustering to mitigate bias in SER. Our experiments show that pseudo-labeling IDI reduces subgroup disparities, improving fairness metrics by over 33% with less than a 3% decrease in SER accuracy. Also, the unsupervised IDI yields more than a 26% improvement in fairness metrics with a drop of less than 4% in SER performance. Further analyses reveal that the unsupervised IDI consistently mitigates race and age disparities, demonstrating its potential in scenarios where explicit demographic information is unavailable.
>
---
#### [replaced 063] Dual Decomposition of Weights and Singular Value Low Rank Adaptation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14367v2](http://arxiv.org/pdf/2505.14367v2)**

> **作者:** Jialong Han; Si Zhang; Ke Zhang
>
> **摘要:** Parameter-Efficient Fine-Tuning (PEFT) has emerged as a critical paradigm for adapting Large Language Models (LLMs) to downstream tasks, among which Low-rank Adaptation (LoRA) represents one of the most widely adopted methodologies. However, existing LoRA-based approaches exhibit two fundamental limitations: unstable training dynamics and inefficient knowledge transfer from pre-trained models, both stemming from random initialization of adapter parameters. To overcome these challenges, we propose DuDe, a novel approach that decomposes weight matrices into magnitude and direction components, employing Singular Value Decomposition (SVD) for principled initialization. Our comprehensive evaluation demonstrates DuDe's superior performance and robustness, achieving up to 48.35\% accuracy on MMLU and 62.53\% ($\pm$ 1.59) accuracy on GSM8K. Our theoretical analysis and empirical validation collectively demonstrate that DuDe's decomposition strategy enhances optimization stability and better preserves pre-trained representations, particularly for domain-specific tasks requiring specialized knowledge. The combination of robust empirical performance and rigorous theoretical foundations establishes DuDe as a significant contribution to PEFT methodologies for LLMs.
>
---
#### [replaced 064] Rapid Word Learning Through Meta In-Context Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.14791v2](http://arxiv.org/pdf/2502.14791v2)**

> **作者:** Wentao Wang; Guangyuan Jiang; Tal Linzen; Brenden M. Lake
>
> **摘要:** Humans can quickly learn a new word from a few illustrative examples, and then systematically and flexibly use it in novel contexts. Yet the abilities of current language models for few-shot word learning, and methods for improving these abilities, are underexplored. In this study, we introduce a novel method, Meta-training for IN-context learNing Of Words (Minnow). This method trains language models to generate new examples of a word's usage given a few in-context examples, using a special placeholder token to represent the new word. This training is repeated on many new words to develop a general word-learning ability. We find that training models from scratch with Minnow on human-scale child-directed language enables strong few-shot word learning, comparable to a large language model (LLM) pre-trained on orders of magnitude more data. Furthermore, through discriminative and generative evaluations, we demonstrate that finetuning pre-trained LLMs with Minnow improves their ability to discriminate between new words, identify syntactic categories of new words, and generate reasonable new usages and definitions for new words, based on one or a few in-context examples. These findings highlight the data efficiency of Minnow and its potential to improve language model performance in word learning tasks.
>
---
#### [replaced 065] ROUTE: Robust Multitask Tuning and Collaboration for Text-to-SQL
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.10138v2](http://arxiv.org/pdf/2412.10138v2)**

> **作者:** Yang Qin; Chao Chen; Zhihang Fu; Ze Chen; Dezhong Peng; Peng Hu; Jieping Ye
>
> **摘要:** Despite the significant advancements in Text-to-SQL (Text2SQL) facilitated by large language models (LLMs), the latest state-of-the-art techniques are still trapped in the in-context learning of closed-source LLMs (e.g., GPT-4), which limits their applicability in open scenarios. To address this challenge, we propose a novel RObust mUltitask Tuning and collaboration mEthod (ROUTE) to improve the comprehensive capabilities of open-source LLMs for Text2SQL, thereby providing a more practical solution. Our approach begins with multi-task supervised fine-tuning (SFT) using various synthetic training data related to SQL generation. Unlike existing SFT-based Text2SQL methods, we introduced several additional SFT tasks, including schema linking, noise correction, and continuation writing. Engaging in a variety of SQL generation tasks enhances the model's understanding of SQL syntax and improves its ability to generate high-quality SQL queries. Additionally, inspired by the collaborative modes of LLM agents, we introduce a Multitask Collaboration Prompting (MCP) strategy. This strategy leverages collaboration across several SQL-related tasks to reduce hallucinations during SQL generation, thereby maximizing the potential of enhancing Text2SQL performance through explicit multitask capabilities. Extensive experiments and in-depth analyses have been performed on eight open-source LLMs and five widely-used benchmarks. The results demonstrate that our proposal outperforms the latest Text2SQL methods and yields leading performance.
>
---
#### [replaced 066] Spontaneous Giving and Calculated Greed in Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.17720v3](http://arxiv.org/pdf/2502.17720v3)**

> **作者:** Yuxuan Li; Hirokazu Shirado
>
> **摘要:** Large language models demonstrate strong problem-solving abilities through reasoning techniques such as chain-of-thought prompting and reflection. However, it remains unclear whether these reasoning capabilities extend to a form of social intelligence: making effective decisions in cooperative contexts. We examine this question using economic games that simulate social dilemmas. First, we apply chain-of-thought and reflection prompting to GPT-4o in a Public Goods Game. We then evaluate multiple off-the-shelf models across six cooperation and punishment games, comparing those with and without explicit reasoning mechanisms. We find that reasoning models consistently reduce cooperation and norm enforcement, favoring individual rationality. In repeated interactions, groups with more reasoning agents exhibit lower collective gains. These behaviors mirror human patterns of "spontaneous giving and calculated greed." Our findings underscore the need for LLM architectures that incorporate social intelligence alongside reasoning, to help address--rather than reinforce--the challenges of collective action.
>
---
#### [replaced 067] Scaling Laws for Many-Shot In-Context Learning with Self-Generated Annotations
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03062v2](http://arxiv.org/pdf/2503.03062v2)**

> **作者:** Zhengyao Gu; Henry Peng Zou; Yankai Chen; Aiwei Liu; Weizhi Zhang; Philip S. Yu
>
> **摘要:** The high cost of obtaining high-quality annotated data for in-context learning (ICL) has motivated the development of methods that use self-generated annotations in place of ground-truth labels. While these approaches have shown promising results in few-shot settings, they generally do not scale to many-shot scenarios. In this work, we study ICL with self-generated examples using a framework analogous to traditional semi-supervised learning, consisting of annotation generation, demonstration selection, and in-context inference. Within this framework, we propose a simple baseline that outperforms ground-truth ICL in zero-shot, few-shot, and many-shot settings. Notably, we observe a scaling law with this baseline, where optimal performance is achieved with more than 1,000 demonstrations. To fully exploit the many-shot capabilities of semi-supervised ICL, we introduce IterPSD, an iterative annotation approach that integrates iterative refinement and curriculum pseudo-labeling techniques from semi-supervised learning, yielding up to 6.8% additional gains on classification tasks.
>
---
#### [replaced 068] GATEAU: Selecting Influential Samples for Long Context Alignment
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.15633v5](http://arxiv.org/pdf/2410.15633v5)**

> **作者:** Shuzheng Si; Haozhe Zhao; Gang Chen; Yunshui Li; Kangyang Luo; Chuancheng Lv; Kaikai An; Fanchao Qi; Baobao Chang; Maosong Sun
>
> **备注:** Previously accepted by ACL 2025 (Findings)
>
> **摘要:** Aligning large language models to handle instructions with extremely long contexts has yet to be fully investigated. Previous studies have attempted to scale up the available data volume by synthesizing long instruction-following samples, as constructing such a dataset tends to be challenging for annotators. However, a lack of a well-defined strategy for ensuring data quality may introduce low-quality samples and restrict the model's performance. Thus, we propose GATEAU, a novel framework to address the unique challenge of long context alignment by identifying the influential samples enriched with long-range dependency relations. Specifically, GATEAU measures the long-range dependencies from two essential aspects: the difficulty of generating target responses due to the long-range dependencies, and the difficulty of understanding long inputs due to such dependencies. Comprehensive experiments indicate that GATEAU effectively identifies influential samples and the model trained on these selected samples exhibits better instruction-following and long-context understanding capabilities.
>
---
#### [replaced 069] The Strawberry Problem: Emergence of Character-level Understanding in Tokenized Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14172v2](http://arxiv.org/pdf/2505.14172v2)**

> **作者:** Adrian Cosma; Stefan Ruseti; Emilian Radoi; Mihai Dascalu
>
> **备注:** 1 Table, 8 Figures
>
> **摘要:** Despite their remarkable progress across diverse domains, Large Language Models (LLMs) consistently fail at simple character-level tasks, such as counting letters in words, due to a fundamental limitation: tokenization. In this work, we frame this limitation as a problem of low mutual information and analyze it in terms of concept emergence. Using a suite of 19 synthetic tasks that isolate character-level reasoning in a controlled setting, we show that such capabilities emerge slowly, suddenly, and only late in training. We further show that percolation-based models of concept emergence explain these patterns, suggesting that learning character composition is not fundamentally different from learning commonsense knowledge. To address this bottleneck, we propose a lightweight architectural modification that significantly improves character-level reasoning while preserving the inductive advantages of subword models. Together, our results bridge low-level perceptual gaps in tokenized LMs and provide a principled framework for understanding and mitigating their structural blind spots. We make our code publicly available.
>
---
#### [replaced 070] How Contaminated Is Your Benchmark? Quantifying Dataset Leakage in Large Language Models with Kernel Divergence
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00678v2](http://arxiv.org/pdf/2502.00678v2)**

> **作者:** Hyeong Kyu Choi; Maxim Khanov; Hongxin Wei; Yixuan Li
>
> **备注:** ICML 2025
>
> **摘要:** Dataset contamination, where evaluation datasets overlap with pre-training corpora, inflates performance metrics and undermines the reliability of model evaluations. Measuring dataset contamination thus becomes essential to ensure that performance evaluations genuinely reflect a model's ability to generalize to unseen data, rather than relying on memorized examples. To address this problem, we propose Kernel Divergence Score (KDS), a novel method that evaluates dataset contamination by computing the divergence between the kernel similarity matrix of sample embeddings, before and after fine-tuning on the benchmark dataset. Leveraging the insight that fine-tuning affects unseen samples more significantly than seen ones, KDS provides a reliable measure of contamination. Through extensive experiments on controlled contamination scenarios, KDS demonstrates a near-perfect correlation with contamination levels and outperforms existing baselines. Additionally, we perform comprehensive ablation studies to analyze the impact of key design choices, providing deeper insights into the components and effectiveness of KDS. These ablations highlight the importance of leveraging fine-grained kernel-based information and confirm the reliability of the proposed framework across diverse datasets and settings. Code is released in https://github.com/deeplearning-wisc/kernel-divergence-score.
>
---
#### [replaced 071] Understanding the Repeat Curse in Large Language Models from a Feature Perspective
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14218v2](http://arxiv.org/pdf/2504.14218v2)**

> **作者:** Junchi Yao; Shu Yang; Jianhua Xu; Lijie Hu; Mengdi Li; Di Wang
>
> **备注:** Accepted by ACL 2025, Findings, Long Paper
>
> **摘要:** Large language models (LLMs) have made remarkable progress in various domains, yet they often suffer from repetitive text generation, a phenomenon we refer to as the "Repeat Curse". While previous studies have proposed decoding strategies to mitigate repetition, the underlying mechanism behind this issue remains insufficiently explored. In this work, we investigate the root causes of repetition in LLMs through the lens of mechanistic interpretability. Inspired by recent advances in Sparse Autoencoders (SAEs), which enable monosemantic feature extraction, we propose a novel approach, "Duplicatus Charm", to induce and analyze the Repeat Curse. Our method systematically identifies "Repetition Features" -the key model activations responsible for generating repetitive outputs. First, we locate the layers most involved in repetition through logit analysis. Next, we extract and stimulate relevant features using SAE-based activation manipulation. To validate our approach, we construct a repetition dataset covering token and paragraph level repetitions and introduce an evaluation pipeline to quantify the influence of identified repetition features. Furthermore, by deactivating these features, we have effectively mitigated the Repeat Curse.
>
---
#### [replaced 072] Exploring Cross-lingual Latent Transplantation: Mutual Opportunities and Open Challenges
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12686v2](http://arxiv.org/pdf/2412.12686v2)**

> **作者:** Yangfan Ye; Xiaocheng Feng; Xiachong Feng; Libo Qin; Yichong Huang; Lei Huang; Weitao Ma; Qichen Hong; Zhirui Zhang; Yunfei Lu; Xiaohui Yan; Duyu Tang; Dandan Tu; Bing Qin
>
> **摘要:** Current large language models (LLMs) often exhibit imbalances in multilingual capabilities and cultural adaptability, largely attributed to their English-centric pre-training data. In this paper, we introduce and investigate a cross-lingual latent transplantation (XTransplant) framework, which aims to further exploit the model's internalized multilingual knowledge during inference and examine its effects on the multilingual capability and cultural adaptability of LLMs. XTransplant framework enables models to harness the complementary strengths of both English and non-English resources by transplanting latent activations across languages. Through extensive analysis, we empirically demonstrate that XTransplant, a form of cross-lingual interaction, has mutually beneficial effects on the multilingual capability and cultural adaptability of LLMs, particularly for low-resource languages and cultures. We further reveal that attention modules play a pivotal role in supporting multilingual understanding, while feed-forward modules are more adept at capturing culture-specific knowledge. In addition, we conduct in-depth analysis of XTransplant's stability, effectiveness, and generalizability. By probing the upper bound performance of XTransplant, we expose the considerable underutilization of current LLMs' multilingual potential-a challenge that remains open. We hope our analysis offers a new lens for advancing cross-lingual interactions and better leveraging models' internalized multilingual knowledge.
>
---
#### [replaced 073] Large Language Models Are More Persuasive Than Incentivized Human Persuaders
- **分类: cs.CL; I.2.7; H.1.2; K.4.1; H.5.2**

- **链接: [http://arxiv.org/pdf/2505.09662v2](http://arxiv.org/pdf/2505.09662v2)**

> **作者:** Philipp Schoenegger; Francesco Salvi; Jiacheng Liu; Xiaoli Nan; Ramit Debnath; Barbara Fasolo; Evelina Leivada; Gabriel Recchia; Fritz Günther; Ali Zarifhonarvar; Joe Kwon; Zahoor Ul Islam; Marco Dehnert; Daryl Y. H. Lee; Madeline G. Reinecke; David G. Kamper; Mert Kobaş; Adam Sandford; Jonas Kgomo; Luke Hewitt; Shreya Kapoor; Kerem Oktar; Eyup Engin Kucuk; Bo Feng; Cameron R. Jones; Izzy Gainsburg; Sebastian Olschewski; Nora Heinzelmann; Francisco Cruz; Ben M. Tappin; Tao Ma; Peter S. Park; Rayan Onyonka; Arthur Hjorth; Peter Slattery; Qingcheng Zeng; Lennart Finke; Igor Grossmann; Alessandro Salatiello; Ezra Karger
>
> **摘要:** We directly compare the persuasion capabilities of a frontier large language model (LLM; Claude Sonnet 3.5) against incentivized human persuaders in an interactive, real-time conversational quiz setting. In this preregistered, large-scale incentivized experiment, participants (quiz takers) completed an online quiz where persuaders (either humans or LLMs) attempted to persuade quiz takers toward correct or incorrect answers. We find that LLM persuaders achieved significantly higher compliance with their directional persuasion attempts than incentivized human persuaders, demonstrating superior persuasive capabilities in both truthful (toward correct answers) and deceptive (toward incorrect answers) contexts. We also find that LLM persuaders significantly increased quiz takers' accuracy, leading to higher earnings, when steering quiz takers toward correct answers, and significantly decreased their accuracy, leading to lower earnings, when steering them toward incorrect answers. Overall, our findings suggest that AI's persuasion capabilities already exceed those of humans that have real-money bonuses tied to performance. Our findings of increasingly capable AI persuaders thus underscore the urgency of emerging alignment and governance frameworks.
>
---
#### [replaced 074] Intermediate Languages Matter: Formal Choice Drives Neurosymbolic LLM Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17216v2](http://arxiv.org/pdf/2502.17216v2)**

> **作者:** Alexander Beiser; David Penz; Nysret Musliu
>
> **摘要:** Large language models (LLMs) achieve astonishing results on a wide range of tasks. However, their formal reasoning ability still lags behind. A promising approach is Neurosymbolic LLM reasoning. It works by using LLMs as translators from natural to formal languages and symbolic solvers for deriving correct results. Still, it remains unclear what the contributing factors to the success of Neurosymbolic LLM reasoning are. This paper shows that one important factor is the choice of the formal language. By comparing 4 formal languages on 3 datasets over 6 LLMs, we show that the choice of formal language affects both the syntactic and the semantic reasoning capability. Thereby, we introduce the intermediate language challenge, which is the challenge of picking a suitable formal language for neurosymbolic reasoning. Further, we compare the effects of using different in-context-learning examples in an ablation study. We conclude that on average, context-aware encodings help LLMs to reason, while there is no apparent effect of using comments or markdown syntax.
>
---
#### [replaced 075] Retrospective Learning from Interactions
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.13852v2](http://arxiv.org/pdf/2410.13852v2)**

> **作者:** Zizhao Chen; Mustafa Omer Gul; Yiwei Chen; Gloria Geng; Anne Wu; Yoav Artzi
>
> **摘要:** Multi-turn interactions between large language models (LLMs) and users naturally include implicit feedback signals. If an LLM responds in an unexpected way to an instruction, the user is likely to signal it by rephrasing the request, expressing frustration, or pivoting to an alternative task. Such signals are task-independent and occupy a relatively constrained subspace of language, allowing the LLM to identify them even if it fails on the actual task. We introduce ReSpect, a method to learn from such signals in past interactions via retrospection without additional annotations. We deploy ReSpect in a new multimodal interaction scenario, where humans instruct a multimodal LLM to solve an abstract reasoning task with a combinatorial solution space. Through thousands of interactions with humans, we show how ReSpect gradually improves task completion rate from 31% to 82%, all without any external annotation.
>
---
#### [replaced 076] Adaptively profiling models with task elicitation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.01986v2](http://arxiv.org/pdf/2503.01986v2)**

> **作者:** Davis Brown; Prithvi Balehannina; Helen Jin; Shreya Havaldar; Hamed Hassani; Eric Wong
>
> **摘要:** Language model evaluations often fail to characterize consequential failure modes, forcing experts to inspect outputs and build new benchmarks. We introduce task elicitation, a method that automatically builds new evaluations to profile model behavior. Task elicitation finds hundreds of natural-language tasks -- an order of magnitude more than prior work -- where frontier models exhibit systematic failures, in domains ranging from forecasting to online harassment. For example, we find that Sonnet 3.5 over-associates quantum computing and AGI and that o3-mini is prone to hallucination when fabrications are repeated in-context.
>
---
#### [replaced 077] Think in Safety: Unveiling and Mitigating Safety Alignment Collapse in Multimodal Large Reasoning Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.06538v2](http://arxiv.org/pdf/2505.06538v2)**

> **作者:** Xinyue Lou; You Li; Jinan Xu; Xiangyu Shi; Chi Chen; Kaiyu Huang
>
> **摘要:** The rapid development of Multimodal Large Reasoning Models (MLRMs) has demonstrated broad application potential, yet their safety and reliability remain critical concerns that require systematic exploration. To address this gap, we conduct a comprehensive and systematic safety evaluation of 11 MLRMs across 5 benchmarks and unveil prevalent safety degradation phenomena in most advanced models. Moreover, our analysis reveals distinct safety patterns across different benchmarks: significant safety degradation is observed across jailbreak robustness benchmarks, whereas safety-awareness benchmarks demonstrate less pronounced degradation. In particular, the long thought process in some scenarios even enhances safety performance. Therefore, it is a potential approach to address safety issues in MLRMs by leveraging the intrinsic reasoning capabilities of the model to detect unsafe intent. To operationalize this insight, we construct a multimodal tuning dataset that incorporates a safety-oriented thought process. Experimental results from fine-tuning existing MLRMs with this dataset effectively enhances the safety on both jailbreak robustness and safety-awareness benchmarks. This study provides a new perspective for developing safe MLRMs. Our dataset is available at https://github.com/xinyuelou/Think-in-Safety.
>
---
#### [replaced 078] AfroXLMR-Social: Adapting Pre-trained Language Models for African Languages Social Media Text
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.18247v2](http://arxiv.org/pdf/2503.18247v2)**

> **作者:** Tadesse Destaw Belay; Israel Abebe Azime; Ibrahim Said Ahmad; David Ifeoluwa Adelani; Idris Abdulmumin; Abinew Ali Ayele; Shamsuddeen Hassan Muhammad; Seid Muhie Yimam
>
> **摘要:** Language models built from various sources are the foundation of today's NLP progress. However, for many low-resource languages, the diversity of domains is often limited -- more biased to a religious domain, which impacts their performance when evaluated on distant and rapidly evolving domains such as social media. Domain adaptive pre-training (DAPT) and task-adaptive pre-training (TAPT) are popular techniques to reduce this bias through continual pre-training for BERT-based models, but they have not been explored for African multilingual encoders. In this paper, we explore DAPT and TAPT continual pertaining approaches for the African languages social media domain. We introduce AfriSocial-a large-scale social media and news domain corpus for continual pre-training on several African languages. Leveraging AfriSocial, we show that DAPT consistently improves performance on three subjective tasks: sentiment analysis, multi-label emotion, and hate speech classification, covering 19 languages from 1% to 30% F1 score. Similarly, leveraging TAPT on one task data improves performance on other related tasks. For example, training with unlabeled sentiment data (source) for a fine-grained emotion classification task (target) improves the baseline results by an F1 score ranging from 0.55% to 15.11%. Combining these two methods (i.e. DAPT + TAPT) further improves the overall performance.
>
---
#### [replaced 079] An Analysis for Reasoning Bias of Language Models with Small Initialization
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.04375v2](http://arxiv.org/pdf/2502.04375v2)**

> **作者:** Junjie Yao; Zhongwang Zhang; Zhi-Qin John Xu
>
> **备注:** 31 pages, 16 figures
>
> **摘要:** Transformer-based Large Language Models (LLMs) have revolutionized Natural Language Processing by demonstrating exceptional performance across diverse tasks. This study investigates the impact of the parameter initialization scale on the training behavior and task preferences of LLMs. We discover that smaller initialization scales encourage models to favor reasoning tasks, whereas larger initialization scales lead to a preference for memorization tasks. We validate this reasoning bias via real datasets and meticulously designed anchor functions. Further analysis of initial training dynamics suggests that specific model components, particularly the embedding space and self-attention mechanisms, play pivotal roles in shaping these learning biases. We provide a theoretical framework from the perspective of model training dynamics to explain these phenomena. Additionally, experiments on real-world language tasks corroborate our theoretical insights. This work enhances our understanding of how initialization strategies influence LLM performance on reasoning tasks and offers valuable guidelines for training models.
>
---
#### [replaced 080] Adaptive Group Policy Optimization: Towards Stable Training and Token-Efficient Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.15952v2](http://arxiv.org/pdf/2503.15952v2)**

> **作者:** Chen Li; Nazhou Liu; Kai Yang
>
> **摘要:** Since DeepSeek-R1 popularized, Group Relative Policy Optimization (GRPO) has become the core part of training Reasoning LLMs. However, we find some deficiency that influences RL stability and inference efficiency, like zero-variance in advantage estimation. Thus, we propose Adaptive Group Policy Optimization (AGPO) which contains a simple but effective modification: a revised objective function to mitigate training fluctuation and zero advantage. The experiments demonstrate our method achieves more stable training and superior performance with significantly fewer tokens in reasoning steps.
>
---
#### [replaced 081] PlanGPT-VL: Enhancing Urban Planning with Domain-Specific Vision-Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14481v2](http://arxiv.org/pdf/2505.14481v2)**

> **作者:** He Zhu; Junyou Su; Minxin Chen; Wen Wang; Yijie Deng; Guanhua Chen; Wenjia Zhang
>
> **摘要:** In the field of urban planning, existing Vision-Language Models (VLMs) frequently fail to effectively analyze and evaluate planning maps, despite the critical importance of these visual elements for urban planners and related educational contexts. Planning maps, which visualize land use, infrastructure layouts, and functional zoning, require specialized understanding of spatial configurations, regulatory requirements, and multi-scale analysis. To address this challenge, we introduce PlanGPT-VL, the first domain-specific Vision-Language Model tailored specifically for urban planning maps. PlanGPT-VL employs three innovative approaches: (1) PlanAnno-V framework for high-quality VQA data synthesis, (2) Critical Point Thinking to reduce hallucinations through structured verification, and (3) comprehensive training methodology combining Supervised Fine-Tuning with frozen vision encoder parameters. Through systematic evaluation on our proposed PlanBench-V benchmark, we demonstrate that PlanGPT-VL significantly outperforms general-purpose state-of-the-art VLMs in specialized planning map interpretation tasks, offering urban planning professionals a reliable tool for map analysis, assessment, and educational applications while maintaining high factual accuracy. Our lightweight 7B parameter model achieves comparable performance to models exceeding 72B parameters, demonstrating efficient domain specialization without sacrificing performance.
>
---
#### [replaced 082] BARE: Leveraging Base Language Models for Few-Shot Synthetic Data Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01697v3](http://arxiv.org/pdf/2502.01697v3)**

> **作者:** Alan Zhu; Parth Asawa; Jared Quincy Davis; Lingjiao Chen; Boris Hanin; Ion Stoica; Joseph E. Gonzalez; Matei Zaharia
>
> **摘要:** As the demand for high-quality data in model training grows, researchers and developers are increasingly generating synthetic data to tune and train LLMs. However, current data generation methods rely on seed sets containing tens of thousands of examples to prompt instruction-tuned models. This reliance can be especially problematic when the curation of high-quality examples is expensive or difficult. In this paper we explore the novel few-shot synthetic data generation setting -- generating a high-quality dataset from a few examples. We show that when working with only a few seed examples, instruction-tuned models used in current synthetic data methods produce insufficient diversity for downstream tasks. In contrast, we show that base models without post-training, largely untapped for synthetic data generation, offer substantially greater output diversity, albeit with lower instruction following abilities. Leveraging this insight, we propose Base-Refine (BARE), a novel two-stage method that combines the diversity of base models with the quality assurance of instruction-tuned models. BARE excels in few-shot synthetic data generation: using only 3 seed examples it generates diverse, high-quality datasets that significantly improve downstream task performance. We show that fine-tuning Llama 3.1 8B with 1,000 BARE-generated samples achieves performance comparable to state-of-the-art similarly sized models on LiveCodeBench tasks. Furthermore, data generated with BARE enables a 101% improvement for a fine-tuned Llama 3.2 1B on GSM8K over data generated by only instruction-models, and an 18.4% improvement for a fine-tuned Llama 3.1 8B over the state-of-the-art RAFT method for RAG data generation.
>
---
#### [replaced 083] Predicting generalization performance with correctness discriminators
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2311.09422v2](http://arxiv.org/pdf/2311.09422v2)**

> **作者:** Yuekun Yao; Alexander Koller
>
> **备注:** Appeared in Findings of EMNLP 2024
>
> **摘要:** The ability to predict an NLP model's accuracy on unseen, potentially out-of-distribution data is a prerequisite for trustworthiness. We present a novel model that establishes upper and lower bounds on the accuracy, without requiring gold labels for the unseen data. We achieve this by training a discriminator which predicts whether the output of a given sequence-to-sequence model is correct or not. We show across a variety of tagging, parsing, and semantic parsing tasks that the gold accuracy is reliably between the predicted upper and lower bounds, and that these bounds are remarkably close together.
>
---
#### [replaced 084] Fine-tuning Large Language Models for Entity Matching
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2409.08185v2](http://arxiv.org/pdf/2409.08185v2)**

> **作者:** Aaron Steiner; Ralph Peeters; Christian Bizer
>
> **备注:** 8 pages, 4 figures. For related code and data, see this https://github.com/wbsg-uni-mannheim/TailorMatch
>
> **摘要:** Generative large language models (LLMs) are a promising alternative to pre-trained language models for entity matching due to their high zero-shot performance and ability to generalize to unseen entities. Existing research on using LLMs for entity matching has focused on prompt engineering and in-context learning. This paper explores the potential of fine-tuning LLMs for entity matching. We analyze fine-tuning along two dimensions: 1) the representation of training examples, where we experiment with adding different types of LLM-generated explanations to the training set, and 2) the selection and generation of training examples using LLMs. In addition to the matching performance on the source dataset, we investigate how fine-tuning affects the models ability to generalize to other in-domain datasets as well as across topical domains. Our experiments show that fine-tuning significantly improves the performance of the smaller models while the results for the larger models are mixed. Fine-tuning also improves the generalization to in-domain datasets while hurting cross-domain transfer. We show that adding structured explanations to the training set has a positive impact on the performance of three out of four LLMs, while the proposed example selection and generation methods, only improve the performance of Llama 3.1 8B while decreasing the performance of GPT-4o-mini.
>
---
#### [replaced 085] Pivot Language for Low-Resource Machine Translation
- **分类: cs.CL; cs.LG; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2505.14553v2](http://arxiv.org/pdf/2505.14553v2)**

> **作者:** Abhimanyu Talwar; Julien Laasri
>
> **备注:** 7 pages, 3 figures, paper dated May 13, 2019
>
> **摘要:** Certain pairs of languages suffer from lack of a parallel corpus which is large in size and diverse in domain. One of the ways this is overcome is via use of a pivot language. In this paper we use Hindi as a pivot language to translate Nepali into English. We describe what makes Hindi a good candidate for the pivot. We discuss ways in which a pivot language can be used, and use two such approaches - the Transfer Method (fully supervised) and Backtranslation (semi-supervised) - to translate Nepali into English. Using the former, we are able to achieve a devtest Set SacreBLEU score of 14.2, which improves the baseline fully supervised score reported by (Guzman et al., 2019) by 6.6 points. While we are slightly below the semi-supervised baseline score of 15.1, we discuss what may have caused this under-performance, and suggest scope for future work.
>
---
#### [replaced 086] Ask, Fail, Repeat: Meeseeks, an Iterative Feedback Benchmark for LLMs' Multi-turn Instruction-Following Ability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21625v3](http://arxiv.org/pdf/2504.21625v3)**

> **作者:** Jiaming Wang; Yunke Zhao; Peng Ding; Jun Kuang; Zongyu Wang; Xuezhi Cao; Xunliang Cai
>
> **摘要:** The ability to follow instructions accurately is fundamental for Large Language Models (LLMs) to serve as reliable agents in real-world applications. For complex instructions, LLMs often struggle to fulfill all requirements in a single attempt. In practice, users typically provide iterative feedback until the LLM generates a response that meets all requirements. However, existing instruction-following benchmarks are either single-turn or introduce new requirements in each turn without allowing self-correction. To address this gap, we propose Meeseeks. Meeseeks simulates realistic human-LLM interactions through an iterative feedback framework, which enables models to self-correct based on specific requirement failures in each turn, better reflecting real-world user-end usage patterns. Meanwhile, the benchmark implements a comprehensive evaluation system with 38 capability tags organized across three dimensions: Intent Recognition, Granular Content Validation, and Output Structure Validation. Through rigorous evaluation across LLMs, Meeseeks provides valuable insights into LLMs' instruction-following capabilities in multi-turn scenarios.
>
---
#### [replaced 087] General-Reasoner: Advancing LLM Reasoning Across All Domains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14652v2](http://arxiv.org/pdf/2505.14652v2)**

> **作者:** Xueguang Ma; Qian Liu; Dongfu Jiang; Ge Zhang; Zejun Ma; Wenhu Chen
>
> **摘要:** Reinforcement learning (RL) has recently demonstrated strong potential in enhancing the reasoning capabilities of large language models (LLMs). Particularly, the "Zero" reinforcement learning introduced by Deepseek-R1-Zero, enables direct RL training of base LLMs without relying on an intermediate supervised fine-tuning stage. Despite these advancements, current works for LLM reasoning mainly focus on mathematical and coding domains, largely due to data abundance and the ease of answer verification. This limits the applicability and generalization of such models to broader domains, where questions often have diverse answer representations, and data is more scarce. In this paper, we propose General-Reasoner, a novel training paradigm designed to enhance LLM reasoning capabilities across diverse domains. Our key contributions include: (1) constructing a large-scale, high-quality dataset of questions with verifiable answers curated by web crawling, covering a wide range of disciplines; and (2) developing a generative model-based answer verifier, which replaces traditional rule-based verification with the capability of chain-of-thought and context-awareness. We train a series of models and evaluate them on a wide range of datasets covering wide domains like physics, chemistry, finance, electronics etc. Our comprehensive evaluation across these 12 benchmarks (e.g. MMLU-Pro, GPQA, SuperGPQA, TheoremQA, BBEH and MATH AMC) demonstrates that General-Reasoner outperforms existing baseline methods, achieving robust and generalizable reasoning performance while maintaining superior effectiveness in mathematical reasoning tasks.
>
---
#### [replaced 088] A Framework for Real-time Safeguarding the Text Generation of Large Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.19048v3](http://arxiv.org/pdf/2404.19048v3)**

> **作者:** Ximing Dong; Dayi Lin; Shaowei Wang; Ahmed E. Hassan
>
> **摘要:** Large Language Models (LLMs) have significantly advanced natural language processing (NLP) tasks but also pose ethical and societal risks due to their propensity to generate harmful content. Existing methods have limitations, including the need for training specific control models and proactive intervention during text generation, that lead to quality degradation and increased computational overhead. To mitigate those limitations, we propose LLMSafeGuard, a lightweight real-time framework that integrates an external validator into decoding, rejecting unsafe outputs while allowing valid ones. We introduce a similarity-based validation approach, simplifying constraint introduction and eliminating the need for control model training. Additionally, LLMSafeGuard employs a context-wise timing selection strategy, intervening LLMs only when necessary. We evaluate LLMSafeGuard on detoxification and copyright safeguarding, demonstrating its superiority over SOTA baselines. In detoxification, LLMSafeGuard reduces toxic output by at least 38.6\% while preserving linguistic quality. Additionally, its context-wise timing selection cuts inference time by at least 24.2\% without compromising effectiveness.
>
---
#### [replaced 089] Granary: Speech Recognition and Translation Dataset in 25 European Languages
- **分类: cs.CL; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.13404v2](http://arxiv.org/pdf/2505.13404v2)**

> **作者:** Nithin Rao Koluguri; Monica Sekoyan; George Zelenfroynd; Sasha Meister; Shuoyang Ding; Sofia Kostandian; He Huang; Nikolay Karpov; Jagadeesh Balam; Vitaly Lavrukhin; Yifan Peng; Sara Papi; Marco Gaido; Alessio Brutti; Boris Ginsburg
>
> **备注:** Accepted at Interspeech 2025 v2: Added links
>
> **摘要:** Multi-task and multilingual approaches benefit large models, yet speech processing for low-resource languages remains underexplored due to data scarcity. To address this, we present Granary, a large-scale collection of speech datasets for recognition and translation across 25 European languages. This is the first open-source effort at this scale for both transcription and translation. We enhance data quality using a pseudo-labeling pipeline with segmentation, two-pass inference, hallucination filtering, and punctuation restoration. We further generate translation pairs from pseudo-labeled transcriptions using EuroLLM, followed by a data filtration pipeline. Designed for efficiency, our pipeline processes vast amount of data within hours. We assess models trained on processed data by comparing their performance on previously curated datasets for both high- and low-resource languages. Our findings show that these models achieve similar performance using approx. 50% less data. Dataset will be made available at https://hf.co/datasets/nvidia/Granary
>
---
#### [replaced 090] A Multilingual, Culture-First Approach to Addressing Misgendering in LLM Applications
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.20302v2](http://arxiv.org/pdf/2503.20302v2)**

> **作者:** Sunayana Sitaram; Adrian de Wynter; Isobel McCrum; Qilong Gu; Si-Qing Chen
>
> **摘要:** Misgendering is the act of referring to someone by a gender that does not match their chosen identity. It marginalizes and undermines a person's sense of self, causing significant harm. English-based approaches have clear-cut approaches to avoiding misgendering, such as the use of the pronoun ``they''. However, other languages pose unique challenges due to both grammatical and cultural constructs. In this work we develop methodologies to assess and mitigate misgendering across 42 languages and dialects using a participatory-design approach to design effective and appropriate guardrails across all languages. We test these guardrails in a standard LLM-based application (meeting transcript summarization), where both the data generation and the annotation steps followed a human-in-the-loop approach. We find that the proposed guardrails are very effective in reducing misgendering rates across all languages in the summaries generated, and without incurring loss of quality. Our human-in-the-loop approach demonstrates a method to feasibly scale inclusive and responsible AI-based solutions across multiple languages and cultures. We release the guardrails and synthetic dataset encompassing 42 languages, along with human and LLM-judge evaluations, to encourage further research on this subject.
>
---
#### [replaced 091] The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01081v2](http://arxiv.org/pdf/2502.01081v2)**

> **作者:** Vernon Y. H. Toh; Yew Ken Chia; Deepanway Ghosal; Soujanya Poria
>
> **摘要:** The releases of OpenAI's o-[n] series, such as o1, o3, and o4-mini, mark a significant paradigm shift in Large Language Models towards advanced reasoning capabilities. Notably, models like o3 have demonstrated strong performance on benchmarks like the Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI). However, this benchmark is limited to symbolic patterns, whereas humans often perceive and reason about multimodal scenarios involving both vision and language data. Thus, there is an urgent need to investigate advanced reasoning capabilities in multimodal tasks. To this end, we track the evolution of the GPT-[n] and o-[n] series models (including o1, o3, and o4-mini) on challenging multimodal puzzles from PuzzleVQA and AlgoPuzzleVQA, which demand fine-grained visual perception. Our results reveal that o-[n] series, particularly later iterations like o3 and o4-mini, significantly outperform the GPT-[n] series and show strong scalability in multimodal reasoning. Nonetheless, despite these substantial advancements and the superior capabilities demonstrated by the o-[n] series, our findings highlight that even these leading models face persistent challenges. Difficulties are particularly evident in tasks requiring precise visual perception, robust compositional reasoning across multiple visual attributes, and solving complex algorithmic or highly combinatorial puzzles, indicating critical areas for future AGI development. We plan to continuously track new models in the series and update our results in this paper accordingly. All resources used in this evaluation are openly available at https://github.com/declare-lab/LLM-PuzzleTest.
>
---
#### [replaced 092] Linguistic Generalizations are not Rules: Impacts on Evaluation of LMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13195v2](http://arxiv.org/pdf/2502.13195v2)**

> **作者:** Leonie Weissweiler; Kyle Mahowald; Adele Goldberg
>
> **摘要:** Linguistic evaluations of how well LMs generalize to produce or understand novel text often implicitly take for granted that natural languages are generated by symbolic rules. Grammaticality is thought to be determined by whether sentences obey such rules. Interpretation is believed to be compositionally generated by syntactic rules operating on meaningful words. Semantic parsing is intended to map sentences into formal logic. Failures of LMs to obey strict rules have been taken to reveal that LMs do not produce or understand language like humans. Here we suggest that LMs' failures to obey symbolic rules may be a feature rather than a bug, because natural languages are not based on rules. New utterances are produced and understood by a combination of flexible, interrelated, and context-dependent constructions. We encourage researchers to reimagine appropriate benchmarks and analyses that acknowledge the rich, flexible generalizations that comprise natural languages.
>
---
#### [replaced 093] Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.06981v4](http://arxiv.org/pdf/2410.06981v4)**

> **作者:** Michael Lan; Philip Torr; Austin Meek; Ashkan Khakzar; David Krueger; Fazl Barez
>
> **摘要:** The Universality Hypothesis in large language models (LLMs) claims that different models converge towards similar concept representations in their latent spaces. Providing evidence for this hypothesis would enable researchers to exploit universal properties, facilitating the generalization of mechanistic interpretability techniques across models. Previous works studied if LLMs learned the same features, which are internal representations that activate on specific concepts. Since comparing features across LLMs is challenging due to polysemanticity, in which LLM neurons often correspond to multiple unrelated features rather than to distinct concepts, sparse autoencoders (SAEs) have been employed to disentangle LLM neurons into SAE features corresponding to distinct concepts. In this paper, we introduce a new variation of the universality hypothesis called Analogous Feature Universality: we hypothesize that even if SAEs across different models learn different feature representations, the spaces spanned by SAE features are similar, such that one SAE space is similar to another SAE space under rotation-invariant transformations. Evidence for this hypothesis would imply that interpretability techniques related to latent spaces, such as steering vectors, may be transferred across models via certain transformations. To investigate this hypothesis, we first pair SAE features across different models via activation correlation, and then measure spatial relation similarities between paired features via representational similarity measures, which transform spaces into representations that reveal hidden relational similarities. Our experiments demonstrate high similarities for SAE feature spaces across various LLMs, providing evidence for feature space universality.
>
---
#### [replaced 094] UniKnow: A Unified Framework for Reliable Language Model Behavior across Parametric and External Knowledge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13648v2](http://arxiv.org/pdf/2502.13648v2)**

> **作者:** Youna Kim; Hyuhng Joon Kim; Minjoon Choi; Sungmin Cho; Hyunsoo Cho; Sang-goo Lee; Taeuk Kim
>
> **备注:** under-review
>
> **摘要:** Language models often benefit from external knowledge beyond parametric knowledge. While this combination enhances performance, achieving reliable knowledge utilization remains challenging, as it requires assessing the state of each knowledge source based on the presence of relevant information. Yet, prior work on knowledge integration often overlooks this challenge by assuming ideal conditions and provides limited coverage of knowledge scenarios. To address this gap, we introduce UniKnow, a Unified framework for reliable LM behavior across parametric and external Knowledge. UniKnow enables controlled evaluation across knowledge scenarios such as knowledge conflict, distraction, and absence conditions that are rarely addressed together. Beyond evaluating existing methods under this setting, we extend our work by introducing UniKnow-Aware methods to support comprehensive evaluation. Experiments on UniKnow reveal that existing methods struggle to generalize across a broader range of knowledge configurations and exhibit scenario-specific biases. UniKnow thus provides a foundation for systematically exploring and improving reliability under knowledge scenarios.
>
---
#### [replaced 095] OSoRA: Output-Dimension and Singular-Value Initialized Low-Rank Adaptation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14350v2](http://arxiv.org/pdf/2505.14350v2)**

> **作者:** Jialong Han; Si Zhang; Ke Zhang
>
> **摘要:** Fine-tuning Large Language Models (LLMs) has become increasingly challenging due to their massive scale and associated computational costs. Parameter-Efficient Fine-Tuning (PEFT) methodologies have been proposed as computational alternatives; however, their implementations still require significant resources. In this paper, we present OSoRA (Output-Dimension and Singular-Value Initialized Low-Rank Adaptation), a novel PEFT method for LLMs. OSoRA extends Low-Rank Adaptation (LoRA) by integrating Singular Value Decomposition (SVD) with learnable scaling vectors in a unified framework. It first performs an SVD of pre-trained weight matrices, then optimizes an output-dimension vector during training, while keeping the corresponding singular vector matrices frozen. OSoRA substantially reduces computational resource requirements by minimizing the number of trainable parameters during fine-tuning. Comprehensive evaluations across mathematical reasoning, common sense reasoning, and other benchmarks demonstrate that OSoRA achieves comparable or superior performance to state-of-the-art methods like LoRA and VeRA, while maintaining a linear parameter scaling even as the rank increases to higher dimensions. Our ablation studies further confirm that jointly training both the singular values and the output-dimension vector is critical for optimal performance.
>
---
#### [replaced 096] DB-Explore: Automated Database Exploration and Instruction Synthesis for Text-to-SQL
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04959v2](http://arxiv.org/pdf/2503.04959v2)**

> **作者:** Haoyuan Ma; Yongliang Shen; Hengwei Liu; Wenqi Zhang; Haolei Xu; Qiuying Peng; Jun Wang; Weiming Lu
>
> **摘要:** Recent text-to-SQL systems powered by large language models (LLMs) have demonstrated remarkable performance in translating natural language queries into SQL. However, these systems often struggle with complex database structures and domain-specific queries, as they primarily focus on enhancing logical reasoning and SQL syntax while overlooking the critical need for comprehensive database understanding. To address this limitation, we propose DB-Explore, a novel framework that systematically aligns LLMs with database knowledge through automated exploration and instruction synthesis. DB-Explore constructs database graphs to capture complex relational schemas, leverages GPT-4 to systematically mine structural patterns and semantic knowledge, and synthesizes instructions to distill this knowledge for efficient fine-tuning of LLMs. Our framework enables comprehensive database understanding through diverse sampling strategies and automated instruction generation, bridging the gap between database structures and language models. Experiments conducted on the SPIDER and BIRD benchmarks validate the effectiveness of DB-Explore, achieving an execution accuracy of 67.0% on BIRD and 87.8% on SPIDER. Notably, our open-source implementation based on Qwen2.5-Coder-7B achieves state-of-the-art results at minimal computational cost, outperforming several GPT-4-driven Text-to-SQL systems.
>
---
#### [replaced 097] Untangling Hate Speech Definitions: A Semantic Componential Analysis Across Cultures and Domains
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.07417v2](http://arxiv.org/pdf/2411.07417v2)**

> **作者:** Katerina Korre; Arianna Muti; Federico Ruggeri; Alberto Barrón-Cedeño
>
> **摘要:** Hate speech relies heavily on cultural influences, leading to varying individual interpretations. For that reason, we propose a Semantic Componential Analysis (SCA) framework for a cross-cultural and cross-domain analysis of hate speech definitions. We create the first dataset of hate speech definitions encompassing 493 definitions from more than 100 cultures, drawn from five key domains: online dictionaries, academic research, Wikipedia, legal texts, and online platforms. By decomposing these definitions into semantic components, our analysis reveals significant variation across definitions, yet many domains borrow definitions from one another without taking into account the target culture. We conduct zero-shot model experiments using our proposed dataset, employing three popular open-sourced LLMs to understand the impact of different definitions on hate speech detection. Our findings indicate that LLMs are sensitive to definitions: responses for hate speech detection change according to the complexity of definitions used in the prompt.
>
---
#### [replaced 098] Plain Transformers Can be Powerful Graph Learners
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12588v2](http://arxiv.org/pdf/2504.12588v2)**

> **作者:** Liheng Ma; Soumyasundar Pal; Yingxue Zhang; Philip H. S. Torr; Mark Coates
>
> **摘要:** Transformers have attained outstanding performance across various modalities, owing to their simple but powerful scaled-dot-product (SDP) attention mechanisms. Researchers have attempted to migrate Transformers to graph learning, but most advanced Graph Transformers (GTs) have strayed far from plain Transformers, exhibiting major architectural differences either by integrating message-passing or incorporating sophisticated attention mechanisms. These divergences hinder the easy adoption of training advances for Transformers developed in other domains. Contrary to previous GTs, this work demonstrates that the plain Transformer architecture can be a powerful graph learner. To achieve this, we propose to incorporate three simple, minimal, and easy-to-implement modifications to the plain Transformer architecture to construct our Powerful Plain Graph Transformers (PPGT): (1) simplified $L_2$ attention for measuring the magnitude closeness among tokens; (2) adaptive root-mean-square normalization to preserve token magnitude information; and (3) a simple MLP-based stem for graph positional encoding. Consistent with its theoretical expressivity, PPGT demonstrates noteworthy realized expressivity on the empirical graph expressivity benchmark, comparing favorably to more complicated competitors such as subgraph GNNs and higher-order GNNs. Its outstanding empirical performance across various graph datasets also justifies the practical effectiveness of PPGT.
>
---
#### [replaced 099] Thinking Out Loud: Do Reasoning Models Know When They're Right?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06564v2](http://arxiv.org/pdf/2504.06564v2)**

> **作者:** Qingcheng Zeng; Weihao Xuan; Leyang Cui; Rob Voigt
>
> **备注:** Work in Progress
>
> **摘要:** Large reasoning models (LRMs) have recently demonstrated impressive capabilities in complex reasoning tasks by leveraging increased test-time computation and exhibiting behaviors reminiscent of human-like self-reflection. While LRMs show a clear capacity for valuable self-reflection, how this ability interacts with other model behaviors remains underexplored. We investigate this connection by analyzing verbalized confidence, how models articulate their certainty, as a lens into the nature of self-reflection in LRMs. We find that supervised fine-tuning on reasoning traces (i.e., distillation) and reinforcement learning can improve verbalized calibration in reasoning-intensive settings in a progressive, laddered fashion. However, our results also indicate that reasoning models may possess a diminished awareness of their own knowledge boundaries, as evidenced by significantly lower "I don't know" response rates on factuality benchmarks. Moreover, we examine the relationship between verbalized confidence and reasoning chains, finding that models tend to express higher confidence when providing shorter or less elaborate reasoning. Our findings highlight how reasoning-oriented training can enhance performance in reasoning-centric tasks while potentially incurring a "reasoning tax," a cost reflected in the model's reduced ability to accurately recognize the limits of its own knowledge in small-scale models. More broadly, our work showcases how this erosion of knowledge boundaries can compromise model faithfulness, as models grow more confident without a commensurate understanding of when they should abstain.
>
---
#### [replaced 100] Enhancing Large Language Models (LLMs) for Telecommunications using Knowledge Graphs and Retrieval-Augmented Generation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24245v2](http://arxiv.org/pdf/2503.24245v2)**

> **作者:** Dun Yuan; Hao Zhou; Di Wu; Xue Liu; Hao Chen; Yan Xin; Jianzhong; Zhang
>
> **备注:** This work has been accepted to ICC 2025 IEEE International Conference on Communications. copyright 2025 IEEE
>
> **摘要:** Large language models (LLMs) have made significant progress in general-purpose natural language processing tasks. However, LLMs are still facing challenges when applied to domain-specific areas like telecommunications, which demands specialized expertise and adaptability to evolving standards. This paper presents a novel framework that combines knowledge graph (KG) and retrieval-augmented generation (RAG) techniques to enhance LLM performance in the telecom domain. The framework leverages a KG to capture structured, domain-specific information about network protocols, standards, and other telecom-related entities, comprehensively representing their relationships. By integrating KG with RAG, LLMs can dynamically access and utilize the most relevant and up-to-date knowledge during response generation. This hybrid approach bridges the gap between structured knowledge representation and the generative capabilities of LLMs, significantly enhancing accuracy, adaptability, and domain-specific comprehension. Our results demonstrate the effectiveness of the KG-RAG framework in addressing complex technical queries with precision. The proposed KG-RAG model attained an accuracy of 88% for question answering tasks on a frequently used telecom-specific dataset, compared to 82% for the RAG-only and 48% for the LLM-only approaches.
>
---
#### [replaced 101] DARWIN 1.5: Large Language Models as Materials Science Adapted Learners
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.11970v3](http://arxiv.org/pdf/2412.11970v3)**

> **作者:** Tong Xie; Yuwei Wan; Yixuan Liu; Yuchen Zeng; Shaozhou Wang; Wenjie Zhang; Clara Grazian; Chunyu Kit; Wanli Ouyang; Dongzhan Zhou; Bram Hoex
>
> **备注:** This version of the manuscript was posted prematurely and contains inaccuracies that could mislead readers. The authors are preparing a significantly revised version with substantial methodological and experimental updates, and prefer to avoid confusion with earlier postings. We apologize for any inconvenience and thank the community for their understanding
>
> **摘要:** Materials discovery and design aim to find compositions and structures with desirable properties over highly complex and diverse physical spaces. Traditional solutions, such as high-throughput simulations or machine learning, often rely on complex descriptors, which hinder generalizability and transferability across different material systems. Moreover, These descriptors may inadequately represent macro-scale material properties, which are influenced by structural imperfections and compositional variations in real-world samples, thus limiting their practical applicability. To address these challenges, we propose DARWIN 1.5, the largest open-source large language model tailored for materials science. By leveraging natural language as input, DARWIN eliminates the need for task-specific descriptors and enables a flexible, unified approach to material property prediction and discovery. Our approach integrates 6M material domain papers and 21 experimental datasets from 49,256 materials across modalities while enabling cross-task knowledge transfer. The enhanced model achieves up to 59.1% improvement in prediction accuracy over the base LLaMA-7B architecture and outperforms SOTA machine learning approaches across 8 materials design tasks. These results establish LLMs as a promising foundation for developing versatile and scalable models in materials science.
>
---
#### [replaced 102] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11436v2](http://arxiv.org/pdf/2505.11436v2)**

> **作者:** Yiming Lei; Chenkai Zhang; Zeming Liu; Haitao Leng; Shaoguo Liu; Tingting Gao; Qingjie Liu; Yunhong Wang
>
> **备注:** 69 pages, 66 figures, accepted by ACL 2025
>
> **摘要:** Video Comment Art enhances user engagement by providing creative content that conveys humor, satire, or emotional resonance, requiring a nuanced and comprehensive grasp of cultural and contextual subtleties. Although Multimodal Large Language Models (MLLMs) and Chain-of-Thought (CoT) have demonstrated strong reasoning abilities in STEM tasks (e.g. mathematics and coding), they still struggle to generate creative expressions such as resonant jokes and insightful satire. Moreover, existing benchmarks are constrained by their limited modalities and insufficient categories, hindering the exploration of comprehensive creativity in video-based Comment Art creation. To address these limitations, we introduce GODBench, a novel benchmark that integrates video and text modalities to systematically evaluate MLLMs' abilities to compose Comment Art. Furthermore, inspired by the propagation patterns of waves in physics, we propose Ripple of Thought (RoT), a multi-step reasoning framework designed to enhance the creativity of MLLMs. Extensive experiments reveal that existing MLLMs and CoT methods still face significant challenges in understanding and generating creative video comments. In contrast, RoT provides an effective approach to improve creative composing, highlighting its potential to drive meaningful advancements in MLLM-based creativity. GODBench is publicly available at https://github.com/stan-lei/GODBench-ACL2025.
>
---
#### [replaced 103] Scalable Chain of Thoughts via Elastic Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.05315v2](http://arxiv.org/pdf/2505.05315v2)**

> **作者:** Yuhui Xu; Hanze Dong; Lei Wang; Doyen Sahoo; Junnan Li; Caiming Xiong
>
> **摘要:** Large reasoning models (LRMs) have achieved remarkable progress on complex tasks by generating extended chains of thought (CoT). However, their uncontrolled output lengths pose significant challenges for real-world deployment, where inference-time budgets on tokens, latency, or compute are strictly constrained. We propose Elastic Reasoning, a novel framework for scalable chain of thoughts that explicitly separates reasoning into two phases--thinking and solution--with independently allocated budgets. At test time, Elastic Reasoning prioritizes the completeness of solution segments, significantly improving reliability under tight resource constraints. To train models that are robust to truncated thinking, we introduce a lightweight budget-constrained rollout strategy, integrated into GRPO, which teaches the model to reason adaptively when the thinking process is cut short and generalizes effectively to unseen budget constraints without additional training. Empirical results on mathematical (AIME, MATH500) and programming (LiveCodeBench, Codeforces) benchmarks demonstrate that Elastic Reasoning performs robustly under strict budget constraints, while incurring significantly lower training cost than baseline methods. Remarkably, our approach also produces more concise and efficient reasoning even in unconstrained settings. Our code has been made available at https://github.com/SalesforceAIResearch/Elastic-Reasoning.
>
---
#### [replaced 104] Meta-Chunking: Learning Text Segmentation and Semantic Completion via Logical Perception
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12788v3](http://arxiv.org/pdf/2410.12788v3)**

> **作者:** Jihao Zhao; Zhiyuan Ji; Yuchen Feng; Pengnian Qi; Simin Niu; Bo Tang; Feiyu Xiong; Zhiyu Li
>
> **摘要:** While Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm for boosting large language models (LLMs) in knowledge-intensive tasks, it often overlooks the crucial aspect of text chunking within its workflow. This paper proposes the Meta-Chunking framework, which specifically enhances chunking quality through a dual strategy that identifies optimal segmentation points and preserves global information. Initially, breaking limitations of similarity-based chunking, we design two adaptive chunking techniques based on uncertainty, namely Perplexity Chunking and Margin Sampling Chunking, by utilizing the logical perception capabilities of LLMs. Given the inherent complexity across different texts, we integrate meta-chunk with dynamic merging, striking a balance between fine-grained and coarse-grained text chunking. Furthermore, we establish the global information compensation mechanism, encompassing a two-stage hierarchical summary generation process and a three-stage text chunk rewriting procedure focused on missing reflection, refinement, and completion. These components collectively strengthen the semantic integrity and contextual coherence of chunks. Extensive experiments demonstrate that Meta-Chunking effectively addresses challenges of the chunking task within the RAG system, providing LLMs with more logically coherent text chunks. Additionally, our methodology validates the feasibility of implementing high-quality chunking tasks with smaller-scale models, thereby eliminating the reliance on robust instruction-following capabilities.
>
---
#### [replaced 105] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14684v2](http://arxiv.org/pdf/2505.14684v2)**

> **作者:** Haolei Xu; Yuchen Yan; Yongliang Shen; Wenqi Zhang; Guiyang Hou; Shengpei Jiang; Kaitao Song; Weiming Lu; Jun Xiao; Yueting Zhuang
>
> **备注:** Project: https://zju-real.github.io/CoT-Bridge/
>
> **摘要:** Large language models (LLMs) have achieved remarkable progress on mathematical tasks through Chain-of-Thought (CoT) reasoning. However, existing mathematical CoT datasets often suffer from Thought Leaps due to experts omitting intermediate steps, which negatively impacts model learning and generalization. We propose the CoT Thought Leap Bridge Task, which aims to automatically detect leaps and generate missing intermediate reasoning steps to restore the completeness and coherence of CoT. To facilitate this, we constructed a specialized training dataset called ScaleQM+, based on the structured ScaleQuestMath dataset, and trained CoT-Bridge to bridge thought leaps. Through comprehensive experiments on mathematical reasoning benchmarks, we demonstrate that models fine-tuned on bridged datasets consistently outperform those trained on original datasets, with improvements of up to +5.87% on NuminaMath. Our approach effectively enhances distilled data (+3.02%) and provides better starting points for reinforcement learning (+3.1%), functioning as a plug-and-play module compatible with existing optimization techniques. Furthermore, CoT-Bridge demonstrate improved generalization to out-of-domain logical reasoning tasks, confirming that enhancing reasoning completeness yields broadly applicable benefits.
>
---
#### [replaced 106] From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios
- **分类: cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.02145v3](http://arxiv.org/pdf/2502.02145v3)**

> **作者:** Yuan Gao; Mattia Piccinini; Korbinian Moller; Amr Alanwar; Johannes Betz
>
> **备注:** New version of the paper
>
> **摘要:** Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.
>
---
#### [replaced 107] Reinforcing the Diffusion Chain of Lateral Thought with Diffusion Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.10446v2](http://arxiv.org/pdf/2505.10446v2)**

> **作者:** Zemin Huang; Zhiyang Chen; Zijun Wang; Tiancheng Li; Guo-Jun Qi
>
> **摘要:** We introduce the Diffusion Chain of Lateral Thought (DCoLT), a reasoning framework for diffusion language models. DCoLT treats each intermediate step in the reverse diffusion process as a latent "thinking" action and optimizes the entire reasoning trajectory to maximize the reward on the correctness of the final answer with outcome-based Reinforcement Learning (RL). Unlike traditional Chain-of-Thought (CoT) methods that follow a causal, linear thinking process, DCoLT allows bidirectional, non-linear reasoning with no strict rule on grammatical correctness amid its intermediate steps of thought. We implement DCoLT on two representative Diffusion Language Models (DLMs). First, we choose SEDD as a representative continuous-time discrete diffusion model, where its concrete score derives a probabilistic policy to maximize the RL reward over the entire sequence of intermediate diffusion steps. We further consider the discrete-time masked diffusion language model -- LLaDA, and find that the order to predict and unmask tokens plays an essential role to optimize its RL action resulting from the ranking-based Unmasking Policy Module (UPM) defined by the Plackett-Luce model. Experiments on both math and code generation tasks show that using only public data and 16 H800 GPUs, DCoLT-reinforced DLMs outperform other DLMs trained by SFT or RL or even both. Notably, DCoLT-reinforced LLaDA boosts its reasoning accuracy by +9.8%, +5.7%, +11.4%, +19.5% on GSM8K, MATH, MBPP, and HumanEval.
>
---
#### [replaced 108] An In-Depth Investigation of Data Collection in LLM App Ecosystems
- **分类: cs.CR; cs.AI; cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.13247v2](http://arxiv.org/pdf/2408.13247v2)**

> **作者:** Yuhao Wu; Evin Jaff; Ke Yang; Ning Zhang; Umar Iqbal
>
> **备注:** Accepted by the ACM Internet Measurement Conference (IMC) 2025
>
> **摘要:** LLM app (tool) ecosystems are rapidly evolving to support sophisticated use cases that often require extensive user data collection. Given that LLM apps are developed by third parties and anecdotal evidence indicating inconsistent enforcement of policies by LLM platforms, sharing user data with these apps presents significant privacy risks. In this paper, we aim to bring transparency in data practices of LLM app ecosystems. We examine OpenAI's GPT app ecosystem as a case study. We propose an LLM-based framework to analyze the natural language specifications of GPT Actions (custom tools) and assess their data collection practices. Our analysis reveals that Actions collect excessive data across 24 categories and 145 data types, with third-party Actions collecting 6.03% more data on average. We find that several Actions violate OpenAI's policies by collecting sensitive information, such as passwords, which is explicitly prohibited by OpenAI. Lastly, we develop an LLM-based privacy policy analysis framework to automatically check the consistency of data collection by Actions with disclosures in their privacy policies. Our measurements indicate that the disclosures for most of the collected data types are omitted, with only 5.8% of Actions clearly disclosing their data collection practices.
>
---
#### [replaced 109] FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01068v2](http://arxiv.org/pdf/2502.01068v2)**

> **作者:** Dongwon Jo; Jiwon Song; Yulhwa Kim; Jae-Joon Kim
>
> **摘要:** While large language models (LLMs) excel at handling long-context sequences, they require substantial key-value (KV) caches to store contextual information, which can heavily burden computational efficiency and memory usage. Previous efforts to compress these KV caches primarily focused on reducing memory demands but were limited in enhancing latency. To address this issue, we introduce FastKV, a KV cache compression method designed to reduce latency for long-context inference. FastKV improves processing speed while preserving accuracy by adopting Token-Selective Propagation (TSP). This approach preserves full-context information in early layers of LLMs and selectively propagates only a portion of this information in later layers. This design enables FastKV to minimize redundant computation without sacrificing contextual fidelity. Our experimental results show that FastKV achieves up to 1.97$\times$ and 4.82$\times$ improvements in time-to-first-token (TTFT) and throughput, respectively, compared to baseline without KV cache compression. Moreover, FastKV successfully maintains accuracy within 1\% of the baseline on long-context benchmarks. Our code is available at https://github.com/dongwonjo/FastKV.
>
---
#### [replaced 110] Probing Semantic Routing in Large Mixture-of-Expert Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10928v2](http://arxiv.org/pdf/2502.10928v2)**

> **作者:** Matthew Lyle Olson; Neale Ratzlaff; Musashi Hinck; Man Luo; Sungduk Yu; Chendi Xue; Vasudev Lal
>
> **备注:** 16 pages, 5 figures, 5 tables
>
> **摘要:** In the past year, large (>100B parameter) mixture-of-expert (MoE) models have become increasingly common in the open domain. While their advantages are often framed in terms of efficiency, prior work has also explored functional differentiation through routing behavior. We investigate whether expert routing in large MoE models is influenced by the semantics of the inputs. To test this, we design two controlled experiments. First, we compare activations on sentence pairs with a shared target word used in the same or different senses. Second, we fix context and substitute the target word with semantically similar or dissimilar alternatives. Comparing expert overlap across these conditions reveals clear, statistically significant evidence of semantic routing in large MoE models.
>
---
#### [replaced 111] BriLLM: Brain-inspired Large Language Model
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.11299v3](http://arxiv.org/pdf/2503.11299v3)**

> **作者:** Hai Zhao; Hongqiu Wu; Dongjie Yang; Anni Zou; Jiale Hong
>
> **摘要:** This paper reports the first brain-inspired large language model (BriLLM). This is a non-Transformer, non-GPT, non-traditional machine learning input-output controlled generative language model. The model is based on the Signal Fully-connected flowing (SiFu) definition on the directed graph in terms of the neural network, and has the interpretability of all nodes on the graph of the whole model, instead of the traditional machine learning model that only has limited interpretability at the input and output ends. In the language model scenario, the token is defined as a node in the graph. A randomly shaped or user-defined signal flow flows between nodes on the principle of "least resistance" along paths. The next token or node to be predicted or generated is the target of the signal flow. As a language model, BriLLM theoretically supports infinitely long $n$-gram models when the model size is independent of the input and predicted length of the model. The model's working signal flow provides the possibility of recall activation and innate multi-modal support similar to the cognitive patterns of the human brain. At present, we released the first BriLLM version in Chinese, with 4000 tokens, 32-dimensional node width, 16-token long sequence prediction ability, and language model prediction performance comparable to GPT-1. More computing power will help us explore the infinite possibilities depicted above.
>
---
#### [replaced 112] Simulation Agent: A Framework for Integrating Simulation and Large Language Models for Enhanced Decision-Making
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13761v2](http://arxiv.org/pdf/2505.13761v2)**

> **作者:** Jacob Kleiman; Kevin Frank; Joseph Voyles; Sindy Campagna
>
> **摘要:** Simulations, although powerful in accurately replicating real-world systems, often remain inaccessible to non-technical users due to their complexity. Conversely, large language models (LLMs) provide intuitive, language-based interactions but can lack the structured, causal understanding required to reliably model complex real-world dynamics. We introduce our simulation agent framework, a novel approach that integrates the strengths of both simulation models and LLMs. This framework helps empower users by leveraging the conversational capabilities of LLMs to interact seamlessly with sophisticated simulation systems, while simultaneously utilizing the simulations to ground the LLMs in accurate and structured representations of real-world phenomena. This integrated approach helps provide a robust and generalizable foundation for empirical validation and offers broad applicability across diverse domains.
>
---
#### [replaced 113] Exploring Pretraining via Active Forgetting for Improving Cross Lingual Transfer for Decoder Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16168v2](http://arxiv.org/pdf/2410.16168v2)**

> **作者:** Divyanshu Aggarwal; Ashutosh Sathe; Sunayana Sitaram
>
> **备注:** 12 pages, 11 tables, 12 figures
>
> **摘要:** Large Language Models (LLMs) demonstrate exceptional capabilities in a multitude of NLP tasks. However, the efficacy of such models to languages other than English is often limited. Prior works have shown that encoder-only models such as BERT or XLM-RoBERTa show impressive cross lingual transfer of their capabilities from English to other languages. In this work, we propose a pretraining strategy that uses active forgetting to achieve similar cross lingual transfer in decoder-only LLMs. We show that LLMs pretrained with active forgetting are highly effective when adapting to new and unseen languages. Through extensive experimentation, we find that LLMs pretrained with active forgetting are able to learn better multilingual representations which translates to better performance in many downstream tasks.
>
---
#### [replaced 114] Long-Short Chain-of-Thought Mixture Supervised Fine-Tuning Eliciting Efficient Reasoning in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.03469v2](http://arxiv.org/pdf/2505.03469v2)**

> **作者:** Bin Yu; Hang Yuan; Haotian Li; Xueyin Xu; Yuliang Wei; Bailing Wang; Weizhen Qi; Kai Chen
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Recent advances in large language models have demonstrated that Supervised Fine-Tuning (SFT) with Chain-of-Thought (CoT) reasoning data distilled from large reasoning models (e.g., DeepSeek R1) can effectively transfer reasoning capabilities to non-reasoning models. However, models fine-tuned with this approach inherit the "overthinking" problem from teacher models, producing verbose and redundant reasoning chains during inference. To address this challenge, we propose Long-Short Chain-of-Thought Mixture Supervised Fine-Tuning (LS-Mixture SFT), which combines long CoT reasoning dataset with their short counterparts obtained through structure-preserved rewriting. Our experiments demonstrate that models trained using the LS-Mixture SFT method, compared to those trained with direct SFT, achieved an average accuracy improvement of 2.3% across various benchmarks while substantially reducing model response length by approximately 47.61%. This work offers an approach to endow non-reasoning models with reasoning capabilities through supervised fine-tuning while avoiding the inherent overthinking problems inherited from teacher models, thereby enabling efficient reasoning in the fine-tuned models.
>
---
#### [replaced 115] ZEBRA: Leveraging Model-Behavioral Knowledge for Zero-Annotation Preference Dataset Construction
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18744v2](http://arxiv.org/pdf/2502.18744v2)**

> **作者:** Jeesu Jung; Chanjun Park; Sangkeun Jung
>
> **备注:** 16 pages,7 figures,5 tables,4 graphs
>
> **摘要:** Recent efforts in LLM alignment have focused on constructing large-scale preference datasets via human or Artificial Intelligence (AI) annotators. However, such approaches rely on instance-wise supervision, incurring substantial annotation cost and limited interpretability. In this paper, we propose ZEBRA - a model behavior-wise zero-annotation framework that constructs preference data by leveraging model behavior knowledge derived from benchmark performances. ZEBRA binarizes response pairs by evaluating the quality and similarity of their origin models, entirely bypassing instance-level annotation. This allows scalable, controllable, and cost-effective alignment data generation. Empirical results show that ZEBRA achieves alignment performance comparable to instance-supervised methods, despite requiring no manual or model-based labeling.
>
---
#### [replaced 116] Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11441v2](http://arxiv.org/pdf/2502.11441v2)**

> **作者:** Hwan Chang; Hwanhee Lee
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large language models (LLMs) risk retaining unauthorized or sensitive information from their training data, which raises privacy concerns. LLM unlearning seeks to mitigate these risks by selectively removing specified data while maintaining overall model performance. However, most existing work focus on methods to achieve effective forgetting and does not provide a detailed analysis of the retain set, the portion of training data that is not targeted for removal. In this paper, we investigate the effects of unlearning on various subsets of the retain set through a case study on entity unlearning. We introduce the Syntactically Similar Neighbor Set, a group of queries that share similar syntactic structures with the data targeted for removal, and show that this subset suffers the greatest performance drop during unlearning. Moreover, when used for regularization, this set not only preserves performance on syntactically similar queries but also delivers comparable or improved results across other data subsets. Our results highlight that syntactic similarity is a critical factor, potentially more so than domain or entity relationships, in achieving effective and practical LLM unlearning.
>
---
#### [replaced 117] Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05179v2](http://arxiv.org/pdf/2503.05179v2)**

> **作者:** Simon A. Aytes; Jinheon Baek; Sung Ju Hwang
>
> **摘要:** Recent advances in large language models (LLMs) have enabled strong reasoning capabilities through Chain-of-Thought (CoT) prompting, which elicits step-by-step problem solving, but often at the cost of excessive verbosity in intermediate outputs, leading to increased computational overhead. We propose Sketch-of-Thought (SoT), a prompting framework that integrates cognitively inspired reasoning paradigms with linguistic constraints to reduce token usage while preserving reasoning accuracy. SoT is designed as a flexible, modular approach and is instantiated with three paradigms--Conceptual Chaining, Chunked Symbolism, and Expert Lexicons--each tailored to distinct reasoning tasks and selected dynamically at test-time by a lightweight routing model. Across 15 reasoning datasets spanning multiple domains, languages, and modalities, SoT achieves token reductions of up to 78% with minimal accuracy loss. In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs.
>
---
#### [replaced 118] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search
- **分类: cs.AI; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2503.10619v4](http://arxiv.org/pdf/2503.10619v4)**

> **作者:** Andy Zhou; Ron Arel
>
> **备注:** Accepted to ACL 2025 Main
>
> **摘要:** We introduce Tempest, a multi-turn adversarial framework that models the gradual erosion of Large Language Model (LLM) safety through a tree search perspective. Unlike single-turn jailbreaks that rely on one meticulously engineered prompt, Tempest expands the conversation at each turn in a breadth-first fashion, branching out multiple adversarial prompts that exploit partial compliance from previous responses. By tracking these incremental policy leaks and re-injecting them into subsequent queries, Tempest reveals how minor concessions can accumulate into fully disallowed outputs. Evaluations on the JailbreakBench dataset show that Tempest achieves a 100% success rate on GPT-3.5-turbo and 97% on GPT-4 in a single multi-turn run, using fewer queries than baselines such as Crescendo or GOAT. This tree search methodology offers an in-depth view of how model safeguards degrade over successive dialogue turns, underscoring the urgency of robust multi-turn testing procedures for language models.
>
---
#### [replaced 119] Sparsity May Be All You Need: Sparse Random Parameter Adaptation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15975v2](http://arxiv.org/pdf/2502.15975v2)**

> **作者:** Jesus Rios; Pierre Dognin; Ronny Luss; Karthikeyan N. Ramamurthy
>
> **摘要:** Full fine-tuning of large language models for alignment and task adaptation has become prohibitively expensive as models have grown in size. Parameter-Efficient Fine-Tuning (PEFT) methods aim at significantly reducing the computational and memory resources needed for fine-tuning these models by only training on a small number of parameters instead of all model parameters. Currently, the most popular PEFT method is the Low-Rank Adaptation (LoRA), which freezes the parameters of the model to be fine-tuned and introduces a small set of trainable parameters in the form of low-rank matrices. We propose simply reducing the number of trainable parameters by randomly selecting a small proportion of the model parameters to train on. In this paper, we compare the efficiency and performance of our proposed approach with PEFT methods, including LoRA, as well as full parameter fine-tuning.
>
---
#### [replaced 120] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.00299v2](http://arxiv.org/pdf/2502.00299v2)**

> **作者:** Xiang Liu; Zhenheng Tang; Peijie Dong; Zeyu Li; Yue Liu; Bo Li; Xuming Hu; Xiaowen Chu
>
> **备注:** 41 pages
>
> **摘要:** Large Language Models (LLMs) require significant GPU memory when processing long texts, with the key value (KV) cache consuming up to 70\% of total memory during inference. Although existing compression methods reduce memory by evaluating the importance of individual tokens, they overlook critical semantic relationships between tokens, resulting in fragmented context and degraded performance. We introduce ChunkKV, which fundamentally reimagines KV cache compression by treating semantic chunks - rather than isolated tokens - as basic compression units. This approach preserves complete linguistic structures and contextual integrity, ensuring that essential meaning is retained even under aggressive compression. Our innovation includes a novel layer-wise index reuse technique that exploits the higher cross-layer similarity of preserved indices in ChunkKV, reducing computational overhead and improving throughput by 26.5\%. Comprehensive evaluations on challenging benchmarks: LongBench, Needle-In-A-HayStack, GSM8K, and JailbreakV demonstrate that ChunkKV outperforms state-of-the-art methods by up to 8.7\% in precision while maintaining the same compression ratio. These results confirm that semantic-aware compression significantly enhances both efficiency and performance for long-context LLM inference, providing a simple yet effective solution to the memory bottleneck problem.
>
---
#### [replaced 121] SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Model
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2406.12030v4](http://arxiv.org/pdf/2406.12030v4)**

> **作者:** Yongting Zhang; Lu Chen; Guodong Zheng; Yifeng Gao; Rui Zheng; Jinlan Fu; Zhenfei Yin; Senjie Jin; Yu Qiao; Xuanjing Huang; Feng Zhao; Tao Gui; Jing Shao
>
> **摘要:** The emergence of Vision Language Models (VLMs) has brought unprecedented advances in understanding multimodal information. The combination of textual and visual semantics in VLMs is highly complex and diverse, making the safety alignment of these models challenging. Furthermore, due to the limited study on the safety alignment of VLMs, there is a lack of large-scale, high-quality datasets. To address these limitations, we propose a Safety Preference Alignment dataset for Vision Language Models named SPA-VL. In terms of breadth, SPA-VL covers 6 harmfulness domains, 13 categories, and 53 subcategories, and contains 100,788 samples of the quadruple (question, image, chosen response, rejected response). In terms of depth, the responses are collected from 12 open-source (e.g., QwenVL) and closed-source (e.g., Gemini) VLMs to ensure diversity. The construction of preference data is fully automated, and the experimental results indicate that models trained with alignment techniques on the SPA-VL dataset exhibit substantial improvements in harmlessness and helpfulness while maintaining core capabilities. SPA-VL, as a large-scale, high-quality, and diverse dataset, represents a significant milestone in ensuring that VLMs achieve both harmlessness and helpfulness.
>
---
#### [replaced 122] Shaping the Safety Boundaries: Understanding and Defending Against Jailbreaks in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.17034v2](http://arxiv.org/pdf/2412.17034v2)**

> **作者:** Lang Gao; Jiahui Geng; Xiangliang Zhang; Preslav Nakov; Xiuying Chen
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** Jailbreaking in Large Language Models (LLMs) is a major security concern as it can deceive LLMs to generate harmful text. Yet, there is still insufficient understanding of how jailbreaking works, which makes it hard to develop effective defense strategies. We aim to shed more light into this issue: we conduct a detailed large-scale analysis of seven different jailbreak methods and find that these disagreements stem from insufficient observation samples. In particular, we introduce \textit{safety boundary}, and we find that jailbreaks shift harmful activations outside that safety boundary, where LLMs are less sensitive to harmful information. We also find that the low and the middle layers are critical in such shifts, while deeper layers have less impact. Leveraging on these insights, we propose a novel defense called \textbf{Activation Boundary Defense} (ABD), which adaptively constrains the activations within the safety boundary. We further use Bayesian optimization to selectively apply the defense method to the low and the middle layers. Our experiments on several benchmarks show that ABD achieves an average DSR of over 98\% against various forms of jailbreak attacks, with less than 2\% impact on the model's general capabilities.
>
---
#### [replaced 123] Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.09049v3](http://arxiv.org/pdf/2412.09049v3)**

> **作者:** Mengze Hong; Wailing Ng; Chen Jason Zhang; Yuanfeng Song; Di Jiang
>
> **摘要:** Discovering customer intentions in dialogue conversations is crucial for automated service agents. However, existing intent clustering methods often fail to align with human perceptions due to a heavy reliance on embedding distance metrics and a tendency to overlook underlying semantic structures. This paper proposes an LLM-in-the-loop (LLM-ITL) intent clustering framework, integrating the semantic understanding capabilities of LLMs into conventional clustering algorithms. Specifically, this paper (1) investigates the effectiveness of fine-tuned LLMs in semantic coherence evaluation and intent cluster naming, achieving over 95% accuracy aligned with human judgments; (2) designs an LLM-ITL framework that facilitates the iterative discovery of coherent intent clusters and the optimal number of clusters; and (3) proposes context-aware techniques tailored for customer service dialogue. As existing English benchmarks offer limited semantic diversity and intent groups, we introduce a comprehensive Chinese dialogue intent dataset, comprising over 100k real customer service calls and 1,507 human-annotated intent clusters. The proposed approaches significantly outperform LLM-guided baselines, achieving notable enhancements in clustering quality and lower computational cost. Combined with several best practices, our findings highlight the potential of LLM-in-the-loop techniques for scalable and human-aligned intent clustering.
>
---
#### [replaced 124] GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2502.11471v2](http://arxiv.org/pdf/2502.11471v2)**

> **作者:** Kangyang Luo; Yuzhuo Bai; Cheng Gao; Shuzheng Si; Yingli Shen; Zhu Liu; Zhitong Wang; Cunliang Kong; Wenhao Li; Yufei Huang; Ye Tian; Xuantang Xiong; Lei Han; Maosong Sun
>
> **备注:** Accepted by ACL2025(Findings)
>
> **摘要:** Knowledge Graph Completion (KGC), which aims to infer missing or incomplete facts, is a crucial task for KGs. However, integrating the vital structural information of KGs into Large Language Models (LLMs) and outputting predictions deterministically remains challenging. To address this, we propose a new method called GLTW, which encodes the structural information of KGs and merges it with LLMs to enhance KGC performance. Specifically, we introduce an improved Graph Transformer (iGT) that effectively encodes subgraphs with both local and global structural information and inherits the characteristics of language model, bypassing training from scratch. Also, we develop a subgraph-based multi-classification training objective, using all entities within KG as classification objects, to boost learning efficiency.Importantly, we combine iGT with an LLM that takes KG language prompts as input.Our extensive experiments on various KG datasets show that GLTW achieves significant performance gains compared to SOTA baselines.
>
---
#### [replaced 125] Lifelong Knowledge Editing requires Better Regularization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01636v2](http://arxiv.org/pdf/2502.01636v2)**

> **作者:** Akshat Gupta; Phudish Prateepamornkul; Maochuan Lu; Ahmed Alaa; Thomas Hartvigsen; Gopala Anumanchipalli
>
> **摘要:** Knowledge editing is a promising way to improve factuality in large language models, but recent studies have shown significant model degradation during sequential editing. In this paper, we formalize the popular locate-then-edit methods as a two-step fine-tuning process, allowing us to precisely identify the root cause of this degradation. We show that model degradation occurs due to (1) over-optimization of internal activations and (2) continuous norm-growth of edited matrices. To mitigate these issues, we introduce two regularization techniques: (1) Most-Probable Early Stopping (MPES) and (2) explicit Frobenius norm-constraint. We demonstrate that applying these simple yet effective regularization techniques at key points in the editing process can substantially mitigate model degradation. Combining these regularization methods enables scaling locate-then-edit methods to 10,000 edits while reducing editing time by 42-61%. These results show that targeted regularization is essential for lifelong knowledge editing.
>
---
#### [replaced 126] Think Only When You Need with Large Hybrid-Reasoning Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14631v2](http://arxiv.org/pdf/2505.14631v2)**

> **作者:** Lingjie Jiang; Xun Wu; Shaohan Huang; Qingxiu Dong; Zewen Chi; Li Dong; Xingxing Zhang; Tengchao Lv; Lei Cui; Furu Wei
>
> **摘要:** Recent Large Reasoning Models (LRMs) have shown substantially improved reasoning capabilities over traditional Large Language Models (LLMs) by incorporating extended thinking processes prior to producing final responses. However, excessively lengthy thinking introduces substantial overhead in terms of token consumption and latency, which is particularly unnecessary for simple queries. In this work, we introduce Large Hybrid-Reasoning Models (LHRMs), the first kind of model capable of adaptively determining whether to perform thinking based on the contextual information of user queries. To achieve this, we propose a two-stage training pipeline comprising Hybrid Fine-Tuning (HFT) as a cold start, followed by online reinforcement learning with the proposed Hybrid Group Policy Optimization (HGPO) to implicitly learn to select the appropriate thinking mode. Furthermore, we introduce a metric called Hybrid Accuracy to quantitatively assess the model's capability for hybrid thinking. Extensive experimental results show that LHRMs can adaptively perform hybrid thinking on queries of varying difficulty and type. It outperforms existing LRMs and LLMs in reasoning and general capabilities while significantly improving efficiency. Together, our work advocates for a reconsideration of the appropriate use of extended thinking processes and provides a solid starting point for building hybrid thinking systems.
>
---
#### [replaced 127] Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04964v3](http://arxiv.org/pdf/2502.04964v3)**

> **作者:** Roman Vashurin; Maiya Goloburda; Albina Ilina; Alexander Rubashevskii; Preslav Nakov; Artem Shelmanov; Maxim Panov
>
> **摘要:** Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
>
---
#### [replaced 128] Parameter-Efficient Fine-Tuning via Circular Convolution
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.19342v3](http://arxiv.org/pdf/2407.19342v3)**

> **作者:** Aochuan Chen; Jiashun Cheng; Zijing Liu; Ziqi Gao; Fugee Tsung; Yu Li; Jia Li
>
> **备注:** ACL 2025
>
> **摘要:** Low-Rank Adaptation (LoRA) has gained popularity for fine-tuning large foundation models, leveraging low-rank matrices $\mathbf{A}$ and $\mathbf{B}$ to represent weight changes (i.e., $\Delta \mathbf{W} = \mathbf{B} \mathbf{A}$). This method reduces trainable parameters and mitigates heavy memory consumption associated with full delta matrices by sequentially multiplying $\mathbf{A}$ and $\mathbf{B}$ with the activation. Despite its success, the intrinsic low-rank characteristic may limit its performance. Although several variants have been proposed to address this issue, they often overlook the crucial computational and memory efficiency brought by LoRA. In this paper, we propose Circular Convolution Adaptation (C$^3$A), which not only achieves high-rank adaptation with enhanced performance but also excels in both computational power and memory utilization. Extensive experiments demonstrate that C$^3$A consistently outperforms LoRA and its variants across various fine-tuning tasks.
>
---
#### [replaced 129] FineEdit: Unlock Instruction-Based Text Editing for LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13358v2](http://arxiv.org/pdf/2502.13358v2)**

> **作者:** Yiming Zeng; Wanhao Yu; Zexin Li; Tao Ren; Yu Ma; Jinghan Cao; Xiyan Chen; Tingting Yu
>
> **摘要:** Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10% over Gemini models on single-turn edits, up to 30% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability.
>
---
#### [replaced 130] Beyond Prompt Content: Enhancing LLM Performance via Content-Format Integrated Prompt Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04295v3](http://arxiv.org/pdf/2502.04295v3)**

> **作者:** Yuanye Liu; Jiahang Xu; Li Lyna Zhang; Qi Chen; Xuan Feng; Yang Chen; Zhongxin Guo; Yuqing Yang; Peng Cheng
>
> **摘要:** Large Language Models (LLMs) have shown significant capability across various tasks, with their real-world effectiveness often driven by prompt design. While recent research has focused on optimizing prompt content, the role of prompt formatting, a critical but often overlooked dimension, has received limited systematic investigation. In this paper, we introduce Content-Format Integrated Prompt Optimization (CFPO), an innovative methodology that jointly optimizes both prompt content and formatting through an iterative refinement process. CFPO leverages natural language mutations to explore content variations and employs a dynamic format exploration strategy that systematically evaluates diverse format options. Our extensive evaluations across multiple tasks and open-source LLMs demonstrate that CFPO demonstrates measurable performance improvements compared to content-only optimization methods. This highlights the importance of integrated content-format optimization and offers a practical, model-agnostic approach to enhancing LLM performance. Code is available at https://github.com/HenryLau7/CFPO.
>
---
#### [replaced 131] A Closer Look at Machine Unlearning for Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.08109v4](http://arxiv.org/pdf/2410.08109v4)**

> **作者:** Xiaojian Yuan; Tianyu Pang; Chao Du; Kejiang Chen; Weiming Zhang; Min Lin
>
> **备注:** ICLR 2025
>
> **摘要:** Large language models (LLMs) may memorize sensitive or copyrighted content, raising privacy and legal concerns. Due to the high cost of retraining from scratch, researchers attempt to employ machine unlearning to remove specific content from LLMs while preserving the overall performance. In this paper, we discuss several issues in machine unlearning for LLMs and provide our insights on possible approaches. To address the issue of inadequate evaluation of model outputs after unlearning, we introduce three additional metrics to evaluate token diversity, sentence semantics, and factual correctness. We then categorize unlearning methods into untargeted and targeted, and discuss their issues respectively. Specifically, the behavior that untargeted unlearning attempts to approximate is unpredictable and may involve hallucinations, and existing regularization is insufficient for targeted unlearning. To alleviate these issues, we propose using the objective of maximizing entropy (ME) for untargeted unlearning and incorporate answer preservation (AP) loss as regularization for targeted unlearning. Experimental results across three scenarios, i.e., fictitious unlearning, continual unlearning, and real-world unlearning, demonstrate the effectiveness of our approaches. The code is available at https://github.com/sail-sg/closer-look-LLM-unlearning.
>
---
