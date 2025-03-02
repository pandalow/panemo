# 嵌入融合方法研究报告

将多个预训练语言模型（PLM）的嵌入进行融合可以充分利用不同模型的优势，从而提升下游任务性能。以下将围绕常见融合策略、相关研究进展、训练技巧和实现方案展开讨论。

## 常见的嵌入融合方法

- **加权平均（Weighted Average）**：对不同模型的嵌入向量按照一定权重求平均。一种简单做法是人为设定或训练可学习的权重，将各模型的Embedding线性组合。这种方法实现简单，计算开销低。在模型嵌入空间相似或经过对齐的情况下，加权平均能起到融合信息的作用。但需要注意，不同模型的嵌入往往位于各自的向量空间，直接平均可能导致信息损失或相互干扰，需确保嵌入在融合前经过一定对齐或归一化处理 ([Embeddings from multiple providers? - API - OpenAI Developer Community](https://community.openai.com/t/embeddings-from-multiple-providers/662994#:~:text=You%20can%20use%20the%20other,the%20resulting%20correlations%20are%20garbage)) ([Embeddings from multiple providers? - API - OpenAI Developer Community](https://community.openai.com/t/embeddings-from-multiple-providers/662994#:~:text=As%20previously%20mentioned%20by%20%40curt,an%20open%20source%20model%20running))。

- **拼接（Concatenation）**：将多个模型的嵌入向量直接拼接在一起构成一个更高维的向量。拼接能够保留各模型Embedding的全部信息，常用于下游分类器或其他模型直接读取融合后的特征。优点是信息容量大，不会彼此稀释；但缺点是维度随模型数量线性增长，可能带来参数和计算的增加。拼接融合在文本分类等任务中常作为**late fusion**基线，即先分别从每个模型获取表示，再合并输入分类器。有研究表明，拼接多源嵌入可提升模型判别力，前提是后续网络能有效利用高维特征。

- **注意力机制融合**：利用注意力（Attention）来学习如何选择和强调各模型嵌入中的关键信息，包括自注意力（Self-Attention）和交叉注意力（Cross-Attention）两种典型方式。**自注意力融合**是将不同模型的嵌入视作一组“序列”，通过自注意力层让它们彼此交互，生成融合后的表示；**交叉注意力融合**则是用一个模型的嵌入作为Query，去检索另一个模型嵌入（作为Key/Value）中的相关信息。例如在机器翻译中，有研究将预训练的BERT表示作为辅助，通过在NMT编码器和解码器各层加入BERT的交叉注意力来融合信息 ([](https://openreview.net/pdf?id=Hyl7ygStwB#:~:text=how%20to%20better%20leverage%20BERT,Our%20code))。这种方法相当于让下游模型“关注”预训练模型提供的语义特征，实现深度融合。注意力融合的优势在于具有动态性——模型可以根据具体输入学习何时依赖哪个模型的表示，更加灵活 ([](https://openreview.net/pdf?id=Hyl7ygStwB#:~:text=how%20to%20better%20leverage%20BERT,Our%20code))。在需要综合多源信息的任务（如阅读理解、多文档QA等）中，注意力机制能有效筛选关联信息。

- **高级融合策略**：包括**专家混合（Mixture-of-Experts, MoE）**和**门控机制（Gating Mechanism）**等更复杂的融合架构。MoE将每个预训练模型视为一个“专家”，引入一个额外的门控网络根据输入动态选择或加权专家输出 ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20Mixture%20of%20Experts%20,between%20computational%20efficiency%20and%20performance))。例如，有工作将GPT-2、BERT和RoBERTa各自产生的嵌入作为专家输出，利用稀疏门控网络为每个输入分配不同专家权重，再将加权结果用于下游分类 ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20quest%20for%20models%20that,bit%20quantization%20for%20efficient%20inference))。这种方法允许模型针对不同类型的输入自动选择最适合的知识来源，在保证性能的同时控制计算成本 ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20Mixture%20of%20Experts%20,between%20computational%20efficiency%20and%20performance))。门控机制本质上也是一种注意力，只是通常以一个小型前馈网络或软max层输出各模型的权重。除此之外，还有研究探索**门控单元融合**（如在Transformer层引入门控单元融合多种特征）以及**专家路由**等技术。总体而言，MoE和门控策略在模型具有互补专长（如不同语言、不同行业文本）或需要弹性伸缩时表现优势。

## 相关研究论文与开源实现

近年来，针对PLM嵌入融合的研究逐渐增多。一些代表性工作和资源包括：

- **BERT融合NMT模型**：Zhu等人在ICLR 2020提出了“BERT-fused”神经机器翻译模型 ([](https://openreview.net/pdf?id=Hyl7ygStwB#:~:text=how%20to%20better%20leverage%20BERT,Our%20code))。他们将预训练的BERT作为外部信息源：首先用BERT对输入句子编码抽取表示，然后在NMT编码器和解码器的每一层通过注意力机制融合BERT表示 ([](https://openreview.net/pdf?id=Hyl7ygStwB#:~:text=how%20to%20better%20leverage%20BERT,Our%20code))。这种深度融合方法显著提升了机器翻译性能，在多项翻译基准上取得SOTA结果。该研究表明，将一个PLM的嵌入逐层注入下游模型是可行且有效的。这一思路也被后续很多多模态或文档级翻译模型借鉴，用于融合额外上下文。

- **Mixture-of-Experts实践**：2025年有开发者构建了一个融合GPT-2、BERT和RoBERTa的MoE模型用于文本分类 ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20quest%20for%20models%20that,bit%20quantization%20for%20efficient%20inference))。其实现以三个预训练Transformer作为并行分支，设计了一个稀疏门控网络根据输入内容选择性激活不同分支，再将它们的输出融合后接入分类头 ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20Mixture%20of%20Experts%20,between%20computational%20efficiency%20and%20performance))。通过这种**专家路由**，模型在保证推理效率的同时集成了多模型的能力。相关代码开源在博客 ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20quest%20for%20models%20that,bit%20quantization%20for%20efficient%20inference))中，并应用了8-bit量化等技巧来降低部署开销。这一案例证明了利用门控机制直接融合多PLM是可行的工程方案。

- **Embedding融合用于知识强化**：有研究尝试将外部知识库或其它来源的向量与PLM嵌入融合，例如**KEPLER**模型将知识图谱嵌入和语言模型结合；又如**K-BERT**引入知识三元组节点的嵌入与BERT表示融合来强化特定领域理解。这类方法通常通过在Transformer自注意力中添加额外token或通过门控单元将知识向量融合进模型隐层，从而提升模型对领域知识的编码能力。

- **最新学术探索**：2024年出现了一些新颖的融合思路，例如*“分层随机嵌入融合”* ([Latent Structure Modulation in Large Language Models Through Stochastic Concept Embedding Transitions](https://arxiv.org/html/2502.05553v1#:~:text=,Webb%2C%20%E2%80%9CDynamic))。Raines等人提出在LLM中引入多级嵌入的随机融合机制，使模型的表示空间更具多样性和稳健性 ([Latent Structure Modulation in Large Language Models Through Stochastic Concept Embedding Transitions](https://arxiv.org/html/2502.05553v1#:~:text=,Webb%2C%20%E2%80%9CDynamic))。虽然该方法主要针对单一LLM的多层表示融合，但从侧面反映出**融合不同表示来增强模型**已成为研究热点。此外，在开放域问答(ODQA)中也有“**量子嵌入融合**”的尝试 ([Quantum-Inspired Fusion for Open-Domain Question Answering](https://www.mdpi.com/2079-9292/13/20/4135#:~:text=In%20quantum%20embedding%20fusion%20layer%2C,corresponding%20passage%20is%20a%20rationale))——通过类比量子态叠加的方法融合多个段落的表征，提升模型判断多文档相关性的能力。这些前沿研究丰富了嵌入融合的手段，为多PLM融合提供了新的灵感。

- **开源工具/框架**：目前主流深度学习框架并没有“开箱即用”的多PLM嵌入融合模块，但一些库提供了相关支持。例如HuggingFace Transformers可以方便地获取不同模型的隐藏状态，用于后续自定义融合；还有项目探索**AdapterFusion**（在不同任务微调的BERT适配器间进行融合）等思想。值得一提的是，PyTorch社区的[Torchtune](https://pytorch.org/torchtune)提供了`FusionEmbedding`和`DeepFusionModel`等模块，方便组合预训练编码器与解码器 ([DeepFusionModel — torchtune 0.3 documentation](https://pytorch.org/torchtune/0.3/generated/torchtune.modules.model_fusion.DeepFusionModel.html#:~:text=DeepFusion%20is%20a%20type%20of,Evolution%20of%20Multimodal%20Model%20Architectures))。虽然这些主要用于多模态（如将图像编码器融合进文本解码器），但技术上与将两种文本PLM组合类似：都需要处理不兼容的词表和维度，以及设计融合层来衔接。

## 融合训练策略与任务适用性

**训练策略**：在融合多模型嵌入时，有效的训练策略至关重要。首先是**参数训练与冻结**的权衡——由于直接微调所有PLM可能导致过大的内存和算力开销，常用做法是**冻结预训练模型的参数，只训练融合层和少量额外参数**。例如上文提到的MoE分类模型，就冻结了GPT-2/BERT/RoBERTa的主体，只训练门控网络和最终分类器 ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20quest%20for%20models%20that,bit%20quantization%20for%20efficient%20inference))。这样既保留了预训练模型的原有知识，又减少了训练难度，避免“灾难性遗忘” ([Integrating Pre-trained Language Model into Neural Machine Translation](https://arxiv.org/html/2310.19680v4#:~:text=However%2C%20incorporating%20PLM%20into%20NMT,restoring%20masked%20monolingual%20language%20data)) ([Integrating Pre-trained Language Model into Neural Machine Translation](https://arxiv.org/html/2310.19680v4#:~:text=effectively%20transforms%20the%20deep%20and,code%20implementation%20is%20publicly%20accessible))。另一策略是**分阶段训练**：先单独或逐步训练融合层的权重，让模型学会协同，再解冻部分PLM细调。此外，可针对不同模型采用**不同学习率**（如预训练部分用极低学习率，融合层用较高学习率）以稳定训练 ([Integrating Pre-trained Language Model into Neural Machine Translation](https://arxiv.org/html/2310.19680v4#:~:text=PLM%20into%20information%20suitable%20for,1%7D1%20Available%20at))。

**信息完整性与性能优化**：融合应尽量在保留各模型信息的同时避免冗余。为此，一些技巧包括：引入**对齐损失**（如余弦相似度损失）促使不同模型的嵌入在共享空间对齐 ([Integrating Pre-trained Language Model into Neural Machine Translation](https://arxiv.org/html/2310.19680v4#:~:text=effectively%20transforms%20the%20deep%20and,code%20implementation%20is%20publicly%20accessible))；使用**残差连接**保留各模型原始特征 ([](https://openreview.net/pdf?id=Hyl7ygStwB#:~:text=how%20to%20better%20leverage%20BERT,Our%20code))；或通过**门控稀疏化**只选择最相关的部分信息，减少噪音干扰 ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20Mixture%20of%20Experts%20,between%20computational%20efficiency%20and%20performance))。优化性能方面，融合模型训练完毕后，可以考虑**蒸馏（Knowledge Distillation）**手段，将多模型融合的效果压缩到一个模型中，从而在推理时无需同时运行多个PLM。这种知识蒸馏在BERT集成、榜单提交流程中较常见，用单模型近似多个模型的输出分布 ([[PDF] Ensemble Distillation for BERT-Based Ranking Models](https://hongleizhuang.github.io/files/ICTIR21.pdf#:~:text=Models%20hongleizhuang,gating%20BERT))。总体而言，高效融合要求在保持信息增益的同时控制模型复杂度和推理成本，必要时通过剪枝、量化等手段来加速部署。

**不同任务上的效果差异**：融合策略的收益取决于任务类型和模型互补性。

- 在**文本分类**等判别式任务中，拼接或简单加权常作为有效手段：如果不同模型擅长捕获不同层次的特征（如一个擅长语义，另一个擅长句法），融合后往往提高分类准确率。例如有工作将BERT与基于知识的嵌入拼接，用于谣言检测取得比单一模型更高的F1值。同样地，若融合语言模型与传统特征（TF-IDF等），也常能微升分类性能。

- 在**信息检索/语义检索**任务上，直接融合嵌入用于向量搜索并不简单，因为不同模型的嵌入空间不直接可比 ([Embeddings from multiple providers? - API - OpenAI Developer Community](https://community.openai.com/t/embeddings-from-multiple-providers/662994#:~:text=You%20can%20use%20the%20other,the%20resulting%20correlations%20are%20garbage)) ([Embeddings from multiple providers? - API - OpenAI Developer Community](https://community.openai.com/t/embeddings-from-multiple-providers/662994#:~:text=As%20previously%20mentioned%20by%20%40curt,an%20open%20source%20model%20running))。一项稳妥做法是**分别检索再融合结果**：例如用模型A的嵌入召回候选集，再用模型B重新排序。但如果一定要融合嵌入，可考虑训练一个小型投影网络，将两模型嵌入映射到同一空间再平均或拼接。不过需要充足训练数据让投影网络学到对齐关系，否则融合向量距离的物理意义会失真 ([Embeddings from multiple providers? - API - OpenAI Developer Community](https://community.openai.com/t/embeddings-from-multiple-providers/662994#:~:text=You%20can%20use%20the%20other,the%20resulting%20correlations%20are%20garbage))。因此，在检索任务中，融合更多体现在决策层而非直接embedding层。

- 在**文本生成**任务（如对话、摘要）中，融合策略更为复杂。通常生成任务以单个模型为核心（如GPT类解码器），融合另一模型的嵌入需要设计特殊结构。例如前述BERT-fused NMT属于生成场景的融合应用，通过交叉注意力让解码器在生成时参考BERT编码 ([](https://openreview.net/pdf?id=Hyl7ygStwB#:~:text=how%20to%20better%20leverage%20BERT,Our%20code))。类似地，在对话系统中也可让一个大型语言模型在解码时“窥视”另一模型对对话背景的编码。生成任务强调输出的连贯和多样，融合时要谨慎避免引入不一致的信息源。因此常用**浅融合**（只在高层有限交互）或**译后重排名**（不同模型分别生成候选后再融合选择）。总的来说，如果目标是提升生成质量且多模型优势互补（如一个善于事实回忆，一个善于语言流畅），那么融合其嵌入信息是有潜力的，但需要精心调校融合方式以确保稳定性。

## PyTorch/TensorFlow 下的实现方案

在深度学习框架下实现多PLM嵌入融合需要解决**模型加载、嵌入提取和融合操作**三个环节。下面以PyTorch为例描述方案，TensorFlow实现思路类似：

1. **加载预训练模型**：利用Transformers等工具加载所需的PLM，例如T5、LLaMA、Velvet等。通常使用对应的`Tokenizer`和`Model`类。例如：
   ```python
   from transformers import T5EncoderModel, LlamaModel, AutoModel
   t5 = T5EncoderModel.from_pretrained('t5-base')
   llama = LlamaModel.from_pretrained('decapoda-research/llama-7b-hf')
   velvet = AutoModel.from_pretrained('velvet-14b-model-path')  # 假设Velvet有HF权重
   ```
   加载后将模型设为评估模式（不启用dropout）。

2. **获取嵌入表示**：将输入文本通过各自的tokenizer编码张量，然后喂入模型获取其隐藏表示。根据任务需要，可提取**序列级别embedding**（如取`[CLS]` token的向量或平均池化所有token）或者**逐token的embedding序列**。例如对句子获取句向量：
   ```python
   inputs = tokenizer(text, return_tensors='pt')
   with torch.no_grad():
       emb_t5 = t5(**inputs).last_hidden_state[:,0]      # T5取encoder第一个位置
       emb_llama = llama(**inputs).last_hidden_state[:,0]# LLaMA取最后隐层第一个token
       emb_velvet = velvet(**inputs).last_hidden_state[:,0]# Velvet类似处理
   ```
   这里简化处理直接取每个模型输出的第一个token隐藏向量作为句子表示（需注意不同模型结构，如T5有编码器-解码器，需使用编码器部分）。

3. **融合操作**：根据选定策略对得到的多个嵌入向量进行融合：
   - *加权平均*: 定义可学习标量参数`w1,w2,w3`，通过`out = w1*emb_t5 + w2*emb_llama + w3*emb_velvet`得到融合向量；也可以对每个向量先做归一化再平均以防范尺度不一致。
   - *拼接*: 使用`torch.cat([emb_t5, emb_llama, emb_velvet], dim=-1)`拼接成一个大向量，然后送入后续层（例如一个全连接层降维）。
   - *注意力*: 若使用自注意力融合，可将三个嵌入堆叠成形如`(batch, seq_len=3, hidden_dim)`的张量，接入一个`nn.MultiheadAttention`模块，使其输出融合后的`(batch, hidden_dim)`向量。对于交叉注意力，可实现一个简单的**融合Transformer层**：例如令T5的表示作为“上下文”，对LLaMA的表示做一次交叉注意力查询，然后再对Velvet做一次，或者并行交叉注意力后再平均。这部分PyTorch需自定义模块，实现Query/Key/Value的线性变换和Attention计算。
   - *门控融合*: 实现一个小型前馈网络接受三个嵌入拼接`[emb_t5||emb_llama||emb_velvet]`作为输入，输出三个非负权重（通过softmax归一化）。例如：
     ```python
     gate = nn.Linear(hidden_dim*3, 3)
     weights = torch.softmax(gate(torch.cat(...)), dim=-1)  # 输出shape (batch,3)
     out = weights[:,0:1]*emb_t5 + weights[:,1:2]*emb_llama + weights[:,2:3]*emb_velvet
     ```
     这样网络会自动学习在不同输入下如何分配权重给各模型的表示。

4. **训练下游任务**：将融合后的表示作为特征，连接任务相关的预测层。例如文本分类则接`nn.Linear`映射到类别数；检索任务可计算融合向量与查询向量的相似度；生成任务可能需要将融合表示喂入解码器的初始隐状态或作为附加的记忆。训练时，通常**冻结各PLM**参数，只训练融合层和后续任务层。从工程实现看，可将各PLM的`requires_grad`设为False，仅融合部分的参数为True。利用标准的反向传播即可学习融合参数。

5. **性能与调试**：训练过程中需要监控各模型贡献。如果采用门控机制，可以观察学习到的权重分布，判断某模型是否被忽略或占据主导，必要时调整学习率或者增加正则约束让各通路均有贡献。对于注意力融合，可可视化注意力权重以理解模型如何在不同输入下选择信息来源。一旦在验证集上取得满意效果，即可保存融合模型参数用于推理部署。

**TensorFlow实现**与上述类似，使用TensorFlow Hub或Keras加载模型，然后通过Keras Functional API将各模型输出张量拼接或加权，并构建可训练模型。需要注意词表的处理：例如输入需要分别经过对应模型的tokenizer；另外输出维度不一致时，可在融合前添加全连接层将它们映射到同一维度再融合。

综上，融合多个PLM嵌入是一项具有潜力的技术，可通过加权、拼接、注意力、门控等多种策略实现。现有研究和实践表明，合理的融合能在文本分类、机器翻译、问答等任务上取得比单一模型更优的表现。然而，成功的融合依赖于谨慎的架构设计和训练调优——需要平衡模型复杂度与信息增益，并针对任务特点选择最合适的融合方式。随着更多开源工具和经验的积累，融合多PLM嵌入将成为提升下游任务效果的有力手段。

**参考文献：**

1. Zhu et al. *Incorporating BERT into Neural Machine Translation*. ICLR 2020. （提出BERT-fused模型，将BERT表示通过注意力融合进NMT各层） ([](https://openreview.net/pdf?id=Hyl7ygStwB#:~:text=how%20to%20better%20leverage%20BERT,Our%20code))

2. Robert McMenemy. *Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization*. GoPenAI Blog, 2025. （介绍如何用门控MoE融合多个Transformer模型） ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20quest%20for%20models%20that,bit%20quantization%20for%20efficient%20inference)) ([Building a Mixture of Experts Model with GPT-2, BERT, RoBERTa, and 8-Bit Quantization | by Robert McMenemy | Jan, 2025 | GoPenAI](https://blog.gopenai.com/building-a-mixture-of-experts-model-with-gpt-2-bert-roberta-and-8-bit-quantization-85fa5d9692ca#:~:text=The%20Mixture%20of%20Experts%20,between%20computational%20efficiency%20and%20performance))

3. OpenAI Developer Forum. *Embeddings from multiple providers?* (Discussion on combining embeddings from different models) ([Embeddings from multiple providers? - API - OpenAI Developer Community](https://community.openai.com/t/embeddings-from-multiple-providers/662994#:~:text=You%20can%20use%20the%20other,the%20resulting%20correlations%20are%20garbage)) ([Embeddings from multiple providers? - API - OpenAI Developer Community](https://community.openai.com/t/embeddings-from-multiple-providers/662994#:~:text=As%20previously%20mentioned%20by%20%40curt,an%20open%20source%20model%20running))

4. Duan et al. *Quantum-Inspired Fusion for Open-Domain QA*. Electronics, 2024. （提出量子态的嵌入融合方法） ([Quantum-Inspired Fusion for Open-Domain Question Answering](https://www.mdpi.com/2079-9292/13/20/4135#:~:text=In%20quantum%20embedding%20fusion%20layer%2C,corresponding%20passage%20is%20a%20rationale))

5. Raines et al. *Enhancing LLMs with Stochastic Multi-Level Embedding Fusion*. Preprint, 2024. （探索分层随机嵌入融合的前沿方法） ([Latent Structure Modulation in Large Language Models Through Stochastic Concept Embedding Transitions](https://arxiv.org/html/2502.05553v1#:~:text=,Webb%2C%20%E2%80%9CDynamic))

