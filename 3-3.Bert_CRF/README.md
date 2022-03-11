# BILSTM_CRF_NER

### 参考

1. [最通俗易懂的BiLSTM-CRF模型中的CRF层介绍](https://zhuanlan.zhihu.com/p/44042528) 

### 模型预测效果

run demo_eval.py

```
Sent: [CLS]走出博物馆，漫步在毕加索走过的小巷里，回首望去，那用水泥和石块砌就、不加任何修饰的旧居，在夕阳下显得古朴而凝重。[SEP]
NER: [['毕加索', 'PER']]
Predicted NER: [['毕加索', 'PER']]
---------------

Sent: [CLS]冬游北海道[SEP]
NER: [['北海道', 'LOC']]
Predicted NER: [['北海道', 'LOC']]
---------------

Sent: [CLS]李彦春[SEP]
NER: [['李彦春', 'PER']]
Predicted NER: [['李彦春', 'PER']]
---------------

Sent: [CLS]冬游北海道，别有一番滋味。[SEP]
NER: [['北海道', 'LOC']]
Predicted NER: [['北海道', 'LOC']]
---------------

Sent: [CLS]出函馆机场，白雪耀目，凉气袭人，但不刺骨。[SEP]
NER: [['函馆机场', 'LOC']]
Predicted NER: [['函馆机场', 'LOC']]
---------------

.
.
.

Sent: [CLS]希腊人将瓦西里斯与奥纳西斯比较时总不忘补充一句：他和奥纳西斯不同，他没有改组家庭。[SEP]
NER: [['希腊', 'LOC'], ['瓦西里斯', 'PER'], ['奥纳西斯', 'PER'], ['奥纳西斯', 'PER']]
Predicted NER: [['希腊', 'LOC'], ['瓦西里斯', 'PER'], ['奥纳西斯', 'PER'], ['奥纳西斯', 'PER']]
---------------

Sent: [CLS]重视传统家庭观念的希腊人，对瓦西里斯幸福的家庭充满赞誉。[SEP]
NER: [['希腊', 'LOC'], ['瓦西里斯', 'PER']]
Predicted NER: [['希腊', 'LOC'], ['瓦西里斯', 'PER']]
---------------

gold_num = 6181
predict_num = 6106
correct_num = 5423
precision = 0.8881428103504749
recall = 0.8773661219867336
f1-score = 0.8827215756490601
```
