## Bert系列模型

token: 词条, 一个词条是字符的任意组合.

* GPT
* Bert: 输入(英文: token(BERT的分词用的是sub words的形式，会把词拆分成一些字词来减少OOV。), 中文: 字) 如何做各种任务: https://www.jianshu.com/p/f4ed3a7bec7c 特殊的分离符: CLS/SEP 输出: 最后输出的向量维度是768
  * 自编码语言模型（Autoencoder LM）
  * 两个任务预训练模型：
* GPT 2.0: https://yuanxiaosc.github.io/2019/08/27/text-generation/ https://zhuanlan.zhihu.com/p/81013931
* MT-DNN: 多任务版本的Bert
* XLM: 跨语言版的Bert
* BERT强大能力的解释：注意力（解析）和组合。https://zhuanlan.zhihu.com/p/58430637
* ERNIE: 知识图谱结合BERT，清华
* ERBIE: 百度

* RoBERTa https://blog.csdn.net/weixin_37947156/article/details/99235621
* BERT-wwm-ext
* SpanBert https://zhuanlan.zhihu.com/p/75893972
  * Span Mask
  * Span Boundary Objective (SBO) 训练目标
  * 发现不加入 Next Sentence Prediction (NSP) 任务，直接用连续一长句训练效果更好

* XLNET: 基于BERT的哪些不足进行的改善?
* ALBERT
* ZEN: 基于BERT的哪些不足进行的改善? 

## Transformer-XL

* Segment-Leevel Recurrence：**上一个segment的所有隐向量序列只参与前向计算，不再进行反向传播。**
* Relative Position Encodings：相对位置编码用于Attention阶段的计算。两个Wk向量，**用于生成基于内容和位置的key向量**。

## BERT

BERT-Base:

12-layer, 768-hidden, 12-heads, 768-word_dim, 110M parameters

learning_rate: 2e-5/5e-5(bert系列学习率小点，其他系列一般是1e-3), weight_decay: 0.01

batch_size: 16/32/64 for bert

epoch: 10 for bert

BERT-Large:

24-layer, 1024-hidden, 16-heads, 1024-word_dim, 340M parameters

## GPT

BPE(Byte Pair Encoding)

## RoBERTa

* 更大bacth size(从256到8K)、更多的训练数据160G(最初BERT 16G)
* 去掉下一句预测(NSP)任务：Bert 原版的 NSP 目标过于简单了，它把”topic prediction”和“coherence prediction”融合了起来。而RoBERTa去除了NSP，而是每次输入连续的多个句子，直到最大长度512（可以跨文章）。这种训练方式叫做（FULL - SENTENCES），而原来的Bert每次只输入两个句子。实验表明在MNLI这种推断句子关系的任务上RoBERTa也能有更好性能。
* **动态掩码**：每次向模型输入一个序列时都会生成新的掩码模式。这样，在大量数据不断输入的过程中，模型会逐渐适应不同的掩码策略，学习不同的语言表征。
* 文本编码：Byte-Pair Encoding（BPE）是字符级和词级别表征的混合，支持处理自然语言语料库中的众多常见词汇。原版的 BERT 实现使用字符级别的 BPE 词汇，大小为 30K，是在利用启发式分词规则对输入进行预处理之后学得的。Facebook 研究者没有采用这种方式，而是考虑用更大的 byte 级别 BPE 词汇表来训练 BERT，这一词汇表包含 50K 的 subword 单元，且没有对输入作任何额外的预处理或分词。

## ALBERT

* 分解式嵌入参数化，将隐藏层的大小与词嵌入的大小分开。这种分隔使得在不显著增加词汇表嵌入参数大小的情况下能够更容易增加隐藏层的大小。
* 跨层参数共享。这种技术可以防止参数随着网络深度的增加而增大。参数共享有三种方式：只共享feed-forward network的参数、只共享attention的参数、共享全部参数。ALBERT默认是共享全部参数的。
* 对BERT的预训练任务Next-sentence prediction (NSP)进行了改进。提出了Sentence-order prediction (SOP)来取代NSP。具体来说，其正例与NSP相同，但负例是通过选择一篇文档中的两个连续的句子并将它们的顺序交换构造的。这样两个句子就会有相同的话题，模型学习到的就更多是句子间的连贯性。

## XLNet

自回归/自编码语言模型

提出了PLM（permutation language model）：按照从左到右的顺序不断对下一个字进行预测（不是按照句子的自然顺序，而是按照句子自然顺序的一个重排列，这个重排列可以称为**预测顺序**；但句子的自然顺序或者位置信息需要被编码到模型中，**需要指明预测的位置**）。

* Two-Stream Self-Attention：Content流编码到当前时刻的所有内容，而Query流只能参考之前的历史以及当前要预测的位置。
* 部分预测：一般只预测句子后面的词（且只预测后面1/K的词），因为上下文比较多。**这样前面词不用算query流。**
* 融入Transformer-XL
* 建模多个segment：对于下游任务包含多个输入序列，如何训练两个segment；比如BERT有NSP任务。两个segment输入格式 \[A, SEP, B, SEP, CLS\]
* **相对segment编码**：避免换句子顺序导致segment编码出现了变化

## 说明

和本项目配套的文章《从Word Embedding到Bert模型-自然语言处理中的预训练技术发展史》。

pytorch版本的bert，写的结构很不错，很适合理解。项目地址：https://github.com/codertimo/BERT-pytorch。

tesorflow版本的bert，没仔细分析。项目地址：https://github.com/YC-wind/embedding_study/tree/master/bert。

## 参考

[Bert源码阅读](https://blog.csdn.net/yujianmin1990/article/details/85175905)

