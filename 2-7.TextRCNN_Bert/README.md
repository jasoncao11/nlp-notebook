# Pytorch implementation of TextRCNN

![RCNN](../images/RCNN.jpeg)

```
[I 2021-04-13 17:32:30,931] A new study created in memory with name: no-name-94ccb397-3ed1-4b43-995a-77ebeddb5bb6
[I 2021-04-13 17:34:25,076] Trial 0 finished with value: 0.8268571428571428 and parameters: {'n_embedding': 300, 'hidden_size': 112, 'optimizer': 'Adam', 'lr': 0.047586622031950694}. Best is trial 0 with value: 0.8268571428571428.
[I 2021-04-13 17:36:01,800] Trial 1 finished with value: 0.9542857142857143 and parameters: {'n_embedding': 200, 'hidden_size': 100, 'optimizer': 'Adam', 'lr': 0.00034340377369448017}. Best is trial 1 with value: 0.9542857142857143.
[I 2021-04-13 17:37:37,659] Trial 2 finished with value: 0.9498571428571428 and parameters: {'n_embedding': 300, 'hidden_size': 84, 'optimizer': 'RMSprop', 'lr': 0.0001603223255650622}. Best is trial 1 with value: 0.9542857142857143.
[I 2021-04-13 17:38:53,826] Trial 3 finished with value: 0.9407142857142857 and parameters: {'n_embedding': 200, 'hidden_size': 70, 'optimizer': 'RMSprop', 'lr': 0.040442355704366446}. Best is trial 1 with value: 0.9542857142857143.
[I 2021-04-13 17:40:17,066] Trial 4 finished with value: 0.7987142857142857 and parameters: {'n_embedding': 200, 'hidden_size': 86, 'optimizer': 'SGD', 'lr': 0.0018078495296908097}. Best is trial 1 with value: 0.9542857142857143.
[I 2021-04-13 17:40:27,771] Trial 5 pruned. 
[I 2021-04-13 17:42:18,497] Trial 6 finished with value: 0.9321428571428572 and parameters: {'n_embedding': 250, 'hidden_size': 124, 'optimizer': 'Adam', 'lr': 0.021200314994974392}. Best is trial 1 with value: 0.9542857142857143.
[I 2021-04-13 17:42:29,568] Trial 7 pruned. 
Study statistics: 
  Number of finished trials:  8
  Number of pruned trials:  2
  Number of complete trials:  6
Best trial:
  Value:  0.9542857142857143
  Params: 
    n_embedding: 200
    hidden_size: 100
    optimizer: Adam
    lr: 0.00034340377369448017
```
