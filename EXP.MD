2021.5.20: 
eval-try-20210514-112546  pdarts中的架构训练一遍，得到97.46 的精度， pdarts 声明的精度为small 97.5% ， 基本合理
search-try-20210514-131318   搜索实验 ： 57633s   得到架构  
eval-try-20210517-111637    基于上面搜索实验得到的架构（number of skip-connect为1）进行训练 600 个epoch 所有的超参不变，用了51 个
     小时 186098s 得到， 97.45/97.5 的精度 。
search-try-20210519-181951  固定supernet,纯测试 reinforce 有效
search-try-20210520-163122  固定supernet， 纯测试 reinforce 有效
search-try-20210524-231522  采用随机选择策略，decay 的方式是hansong的方式，出现了 REINFORCE的精度很低（子网），始终跟不上超网的情况；（ 同时出现了valid很差， 也是一直跟不上TRAIN 精度 ---  已确认原因 ）
search-try-20210525-194830  这个是search-try-20210522-144938 的复现实验， 是argmax 的采样方式，但是decay 不是hansong的方式。目前问题是总是出现POOL    --- pooling 的出现是因为“**马太效应**”
search-try-20210526-110429  基于上一实验， 把 decay 恢复成hansong的方式 , 仍然还是总采集出pooling 操作
eval-try-20210526-111004  把search-try-20210525-194830搜出的全是pooling 的网络 从头训练600 epoch   达到了95.6, 奇怪的是valid 大于 train 很多是否确实是该架构对valid 数据友好，对train不友好？ 不是，因为用的是test数据集 



search-try-20210527-220733（**pdarts_modify**）  rl_interval_steps=2， lr = 1e-3， epoch = 40 ， 搜出全是conv 3X3 -- REINFORCE 的效果不好 (尤其是第一个stage ，只能达到 0.3 左右， 是强化学习收敛得慢？ )， 更不上超网络。 

search-try-20210528-104454(**pdarts**)   interval 1 lr=1e-2   epoch=40 搜出特别多的CONV 3x3 应该出现了**过拟合 train 97/ valid 84** , REINFORCE 效果还不好 

search-try-20210530-192043 (**pdarts**)    基于以上， 把epoch 改为了25 ， 还是出现过拟合（train 95/ valid 85） ， 大部分 CONV 3x3 ， 多了conv 1x1_3x3 . 

search-try-20210602-133418 （**pdarts**）  把reward 改为**random** ， 确认 **REINFROCE** 是否有效 ？？？   **--- 确认是有效的， 是否还要再做实验  ？？？？**

eval-try-20210601-165727  （**pdarts**）把搜出来的几乎全是 **CONV 3x3** 的模型， 做full training --- valid 精度能到**97.5%** ， 看起来还不错。 

20210606 （pdarts_modify ） 打算对比  1e-2 和  1e-3 ， **最终程序运行失败**，可能爆内存了。 

search-try-20210602-130324 (pdarts_modify)  lr = **1e-3**  epoch=25   -- 待其跑完，看效果，1e-2 是否前期收敛的太快，导致，局部最优了。 毕竟浅网络的好连接，不一定是深网络的好连接。   -- 1e-3 的收敛效果不好， 可以**调整到5e-3尝试一下。**  

search-try-20210602-170606    LR 调整到**5e-3 ，** 同时 打印 各个 连接权重（softmax 后的）变化过程的， 进行分析。    -----  效果不好。 

search-try-20210603-160037(pdarts_modify)    LR 调整到1e-2， 同时 打印 各个 连接权重（softmax 后的）变化过程的， 进行分析。   ---   经过分析，发现参数始终趋向于一个简单的操作  ，     并且每一行都一样  --一家独大，遥遥领先。 

search-sep-20210604-133214（pdarts_modify）     基于上一个问题， 为了解决， 采用sep 的方式， RL 的时候，直接初始化， 不做架构训练， 试试RL 会是什么效果，理论上，不会再出现一家独大。   ---- **还是存在“一步领先，步步领先的情况” ， 但是没有出现始终选择 CONV3x3的情况。** 

search-sep-20210604-165743 (pdarts_modify)    为解决“上一个实验”问题， **修改的随机采样策略**， 算是一种off-policy 策略 ， 或者是一种  探索和利用 的尝试，（甚至可以加上交替）， **--  看上去比较均衡** 

search-try-20210604-192647     pdarts 上再跑一次随机采样   ，lr =1e-2 , 注意打印softmax 权重， 进行一次完整的搜索。       ------   **被错误停掉了 ，** REINFORCE 精度没有的到持续强化， 但应该是合理的，因为策略是random sample。

增加tensorboard scalar   监控每一个操作中，最优index 的变化情况， 划分stage。    **---- 调试OK** 

增加tensorboard 监控  10次 *  20 （间隔） =  **200** 采样中，最大的reward 对应的index  ， 用来看下，operation 是否收敛到了最好的采样， 确认强化的效果（因为是随机采样策略， sample avg reward 已经看不出强化效果了。）  **-- 调试OK** 



**search-try-20210605-153913**  (pdart) ， 预训练 5 轮， 总共25 轮  ， 采样策略为multinominal   ----  conv 3X3 严重， 根据TB ， 很早期即收敛 

```
2021-06-06 21:47:51,529 Genotype(normal=[('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_5x5', 1)], reduce_concat=range(2, 6))
2021-06-06 21:47:51,551 Total searching time: 108517s
```

![image-20210608103036083](EXP/image-20210608103036083.png)

![image-20210608103059085](EXP/image-20210608103059085.png)

**分析**： “一家独大” 、“马太效应” 、  急剧收敛，空间急剧收缩，导致 best arch 都非常局限，严重的局部最优。



**search-try-20210605-181137**  （**pdarts_modify**）  , 预训练 5 轮， 总共 25 轮， 采样策略为 random    ----  有一定的效果， tb 分析在后期扔有变化，但是在sort 出前序节点以后，dil / sep 被干掉了， 还是只剩下了CONV3X3 . 

```
2021-06-06 20:32:28,902 Genotype(normal=[('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_5x5', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 1), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))
2021-06-06 20:32:28,924 Total searching time: 94851s
```

![image-20210608102141554](EXP/image-20210608102141554.png)

![image-20210608103139672](EXP/image-20210608103139672.png)

**分析： “一家独大” 且 “马太效应”   ， 但是收敛的稍微慢一些 。**   

search-try-20210607-145425( pdarts_modify ) 修改代码， 增加 argmax（index） 后的验证精度。 相当于对policy 的验证， 确认强化学习的效果。 做到off-policy 。   同时，将预训练增加到  15 ， 试图验证 ”一家独大“ 问题 。  以决定是否把工作重心切换到”sep “上面。 

```
2021-06-08 08:05:49,110 Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('conv 3x3', 0), ('conv_3x1_1x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv_3x1_1x3', 1), ('conv 3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_7x7', 0), ('max_pool_3x3', 1), ('max_pool_7x7', 0), ('avg_pool_3x3', 2), ('max_pool_7x7', 1), ('max_pool_5x5', 3), ('max_pool_7x7', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
2021-06-08 08:05:49,146 Total searching time: 61883s    --  17hours
```

![image-20210608094659785](EXP/image-20210608094659785.png)

**分析**： **没有再出现“一家独大”， 应该和前期训练的次数有关系（从5 提升到了15 ， 所以应该做sep 是有用的）。** 但是仍然具有“**马太效应**”，即收敛的太快 。 

架构大小是9.81M   ，  还是属于比较大的。     --- **超网精度80/82/85左右。** 



eval-try-20210607-150352 (**pdarts**)   随便改一个架构，和搜出来的架构进行对比 ， 确认是不是任意一个架构都行 ？？？ =----  **6.03 M      正在训练   353/95.6**     

针对cifar-这样的数据集，并不需要太深的模型 ？？？  ---- 看下pdarts 原文， 其他的文章和综述，是否有针对这个的抨击，讨论.    缩短 step ， layers  等 架构深度超参

search-try-20210607-223726 （**pdarts**）把 conv 3x3   conv 1x1_3x3 去掉  ， 采用随机搜索的模式， 预训练15 轮再强化（加了一些当前）。   ---  正在搜索     当前超网络精度   56/70 



pdarts_modify                    始终感觉 训练和强化，不能交替 ， 必须分离， 否则太容易“马太效应了”， 导致搜索空间急剧缩小。 

sep 和  普通的方式，本质差别是什么？？？   普通的让马太效应持续累加， sep 能够稀释 ？？？ 



监控pdarts/darts 的原始效果， 准备分析和评论。 

把这个架构 search-try-20210607-145425  重头训练一遍    。   

SHEBO：  随机采集出来的  最好架构 中的10个  ， 可以取平均， 然后和当前收敛的架构进行比较。 收敛的不能和这个平均值持平  ？？？？？、  --- 需要实验体现出来。 