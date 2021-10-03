# Bert for seq2seq

## 1. 原理
```
直接用单个Bert模型就可以做Seq2Seq任务，而不用区分encoder和decoder。假如输入是“你想吃啥”，目标句子是“白切鸡”，将这两个句子拼成一个：[CLS] 你 想 吃 啥 [SEP] 白 切 鸡 [SEP]。
经过这样转化之后，最简单的方案就是训练一个语言模型，然后输入“[CLS] 你 想 吃 啥 [SEP]”来逐字预测“白 切 鸡”，直到出现“[SEP]”为止。
为此，我们需要构造一个特别的Mask即可，使得输入部分的Attention是双向的，输出部分的Attention是单向的以防止其看到“未来信息”。mask的构建参考 mask_demo.py。
```

## 2. 参考
- [从语言模型到Seq2Seq：Transformer如戏，全靠Mask](https://kexue.fm/archives/6933)