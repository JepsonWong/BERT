## Bert系列模型

token: 词条, 一个词条是字符的任意组合.

* GPT
* Bert: 输入(英文: token(BERT的分词用的是sub words的形式，会把词拆分成一些字词来减少OOV。), 中文: 字) 如何做各种任务: https://www.jianshu.com/p/f4ed3a7bec7c 特殊的分离符: CLS/SEP 输出: 最后输出的向量维度是768
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

## 说明

和本项目配套的文章《从Word Embedding到Bert模型-自然语言处理中的预训练技术发展史》。

pytorch版本的bert，写的结构很不错，很适合理解。项目地址：https://github.com/codertimo/BERT-pytorch。

tesorflow版本的bert，没仔细分析。项目地址：https://github.com/YC-wind/embedding_study/tree/master/bert。

## 参考

[Bert源码阅读](https://blog.csdn.net/yujianmin1990/article/details/85175905)

