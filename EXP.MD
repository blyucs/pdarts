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



1. search-try-20210527-220733（**pdarts_modify**）  rl_interval_steps=2， lr = 1e-3， epoch = 40 ， 搜出全是conv 3X3 -- REINFORCE 的效果不好 (尤其是第一个stage ，只能达到 0.3 左右， 是强化学习收敛得慢？ )， 更不上超网络。 

2. search-try-20210528-104454(**pdarts**)   interval 1 lr=1e-2   epoch=40 搜出特别多的CONV 3x3 应该出现了**过拟合 train 97/ valid 84** , REINFORCE 效果还不好 

3. search-try-20210530-192043 (**pdarts**)    基于以上， 把epoch 改为了25 ， 还是出现过拟合（train 95/ valid 85） ， 大部分 CONV 3x3 ， 多了conv 1x1_3x3 . 

4. search-try-20210602-133418 （**pdarts**）  把reward 改为**random** ， 确认 **REINFROCE** 是否有效 ？？？   **--- 确认是有效的， 是否还要再做实验  ？？？？**

5. eval-try-20210601-165727  （**pdarts**）把搜出来的几乎全是 **CONV 3x3** 的模型， 做full training --- valid 精度能到**97.5%** ， 看起来还不错。 

6. 20210606 （pdarts_modify ） 打算对比  1e-2 和  1e-3 ， **最终程序运行失败**，可能爆内存了。 

7. search-try-20210602-130324 (pdarts_modify)  lr = **1e-3**  epoch=25   -- 待其跑完，看效果，1e-2 是否前期收敛的太快，导致，局部最优了。 毕竟浅网络的好连接，不一定是深网络的好连接。   -- 1e-3 的收敛效果不好， 可以**调整到5e-3尝试一下。**  

8. search-try-20210602-170606    LR 调整到**5e-3 ，** 同时 打印 各个 连接权重（softmax 后的）变化过程的， 进行分析。    -----  效果不好。 

9. search-try-20210603-160037(pdarts_modify)    LR 调整到1e-2， 同时 打印 各个 连接权重（softmax 后的）变化过程的， 进行分析。   ---   经过分析，发现参数始终趋向于一个简单的操作  ，     并且每一行都一样  --一家独大，遥遥领先。 

10. search-sep-20210604-133214（pdarts_modify）     基于上一个问题， 为了解决， 采用sep 的方式， RL 的时候，直接初始化， 不做架构训练， 试试RL 会是什么效果，理论上，不会再出现一家独大。   ---- **还是存在“一步领先，步步领先的情况” ， 但是没有出现始终选择 CONV3x3的情况。** 

11. search-sep-20210604-165743 (pdarts_modify)    为解决“上一个实验”问题， **修改的随机采样策略**， 算是一种off-policy 策略 ， 或者是一种  探索和利用 的尝试，（甚至可以加上交替）， **--  看上去比较均衡** 

12. search-try-20210604-192647     pdarts 上再跑一次随机采样   ，lr =1e-2 , 注意打印softmax 权重， 进行一次完整的搜索。       ------   **被错误停掉了 ，** REINFORCE 精度没有的到持续强化， 但应该是合理的，因为策略是random sample。

13. 增加tensorboard scalar   监控每一个操作中，最优index 的变化情况， 划分stage。    **---- 调试OK** 

14. 增加tensorboard 监控  10次 *  20 （间隔） =  **200** 采样中，最大的reward 对应的index  ， 用来看下，operation 是否收敛到了最好的采样， 确认强化的效果（因为是随机采样策略， sample avg reward 已经看不出强化效果了。）  **-- 调试OK** 



15. **search-try-20210605-153913**  (pdart) ， 预训练 5 轮， 总共25 轮  ， 采样策略为**multinominal**   ----  conv 3X3 严重， 根据TB ， 很早期即收敛 

```
2021-06-06 21:47:51,529 Genotype(normal=[('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('avg_pool_3x3', 3), ('max_pool_3x3', 0), ('max_pool_5x5', 1)], reduce_concat=range(2, 6))
2021-06-06 21:47:51,551 Total searching time: 108517s
```

![image-20210608103036083](EXP/image-20210608103036083.png)

![image-20210608103059085](EXP/image-20210608103059085.png)

**分析**： “一家独大” 、“马太效应” 、  急剧收敛，空间急剧收缩，导致 best arch 都非常局限，严重的局部最优。



16. **search-try-20210605-181137**  （**pdarts_modify**）  , 预训练 5 轮， 总共 25 轮， 采样策略为 random    ----  有一定的效果， tb 分析在后期扔有变化，但是在sort 出前序节点以后，dil / sep 被干掉了， 还是只剩下了CONV3X3 . 

```
2021-06-06 20:32:28,902 Genotype(normal=[('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv 3x3', 0), ('conv 3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_5x5', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 1), ('avg_pool_3x3', 3)], reduce_concat=range(2, 6))
2021-06-06 20:32:28,924 Total searching time: 94851s
```

![image-20210608102141554](EXP/image-20210608102141554.png)

![image-20210608103139672](EXP/image-20210608103139672.png)

**分析： “一家独大” 且 “马太效应”   ， 但是收敛的稍微慢一些 。**   



17. search-try-20210607-145425( pdarts_modify )  采样方式来random sample  。 同时**，将预训练增加到  15** ， 试图验证 ”一家独大“ 问题 。 修改代码， 增加 argmax（index） 后的验证精度。 相当于对policy 的验证， 确认强化学习的效果。 做到off-policy 。    以决定是否把工作重心切换到”sep “上面。 

```
2021-06-08 08:05:49,110 Genotype(normal=[('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('conv 3x3', 0), ('conv_3x1_1x3', 1), ('conv 3x3', 0), ('conv 3x3', 1), ('conv_3x1_1x3', 1), ('conv 3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_7x7', 0), ('max_pool_3x3', 1), ('max_pool_7x7', 0), ('avg_pool_3x3', 2), ('max_pool_7x7', 1), ('max_pool_5x5', 3), ('max_pool_7x7', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
2021-06-08 08:05:49,146 Total searching time: 61883s    --  17hours
```

![image-20210608094659785](EXP/image-20210608094659785.png)

**分析**： **没有再出现“一家独大”， 应该和前期训练的次数有关系（从5 提升到了15 ， 所以应该做sep 是有用的）。** 但是仍然具有“**马太效应**”，即收敛的太快 。 

架构大小是9.81M   ，  还是属于比较大的。     --- **超网精度80/82/85左右。** 



18. eval-try-20210607-150352 (**pdarts**)   随便改一个架构，和搜出来的架构进行对比 ， 确认是不是任意一个架构都行 ？？？ =----  **6.03 M      正在训练   353/95.6**     

```
2021-06-09 01:16:09,992 Valid_acc: 97.320000   2021-06-09 01:16:10,081 Eval time: 123138s.
```

**分析**： 在CIFAR10 数据集上，基于darts 搜索空间， 只要模型参数量够大， 随便改一个，训出来精度应该都还可以。  



19. search-try-20210607-223726 （**pdarts**）把 conv 3x3   conv 1x1_3x3 去掉  （参数量太大？）， 采用随机搜索的模式， 预训练15 轮再强化（加了一些当前）。   

![image-20210609164516593](EXP/image-20210609164516593.png)

**分析：** 可以看出有一定的强化效果，但是整体都不太高（valid 导致 ？ ） 。 而且reward 增加是否是来自于 权重参数的训练 ？？？ 

![image-20210609164737375](EXP/image-20210609164737375.png)

**分析：  过程看起来可以，**  

```
2021-06-08 17:28:42,902 Genotype(normal=[('skip_connect', 0), ('conv 1x1', 1), ('conv 1x1', 0), ('dil_conv_3x3', 2), ('conv 1x1', 0), ('conv 1x1', 1), ('conv 1x1', 0), ('conv 1x1', 1)], normal_concat=range(2, 6), reduce=[('max_pool_5x5', 0), ('max_pool_5x5', 1), ('max_pool_7x7', 0), ('max_pool_3x3', 1), ('max_pool_5x5', 0), ('max_pool_3x3', 2), ('max_pool_5x5', 0), ('skip_connect', 1)], reduce_concat=range(2, 6))2021-06-08 17:28:42,902 Restricting skipconnect...2021-06-08 17:28:42,924 Total searching time: 67876s
```

**分析：  剩余架构结果看起来合理。但是 parsing 以后， 剩下太多conv 1X1 ** 





20. search-sep-20210608-170448   （ pdarts_modify ）  始终感觉 训练和强化，不能交替 ， 必须分离， 否则太容易“马太效应了”， 导致搜索空间急剧缩小。 

sep 和  普通的方式，本质差别是什么？？？   普通的让马太效应持续累加， sep 能够稀释 ？？？ 

共  3* 40 epoch  =   8 （5+3） *  5

分析： 采样效率不够。采样架构始终reward 很低， 

![image-20210609161605794](EXP/image-20210609161605794.png)

分析： 随机采样的best_reward 架构， 并没有显著上升， 因为整个过程还包含了 间隔的训练过程（5个epoch），说明是整个空间太大了，采样没有方向性； 或是网络被有指向性的训练了， 所以随机采出的整体效果都差 。



![image-20210609161654172](EXP/image-20210609161654172.png)



分析： valid 中的 eval 导致 了一些BN 的问题 ？？？ 

![image-20210609162918941](EXP/image-20210609162918941.png)

分析： “马太效应”  不明显， “一家独大”  明显

```
2021-06-09 13:21:45,591 Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('conv 1x1', 0), ('skip_connect', 1), ('conv 1x1', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_7x7', 0), ('max_pool_5x5', 1), ('max_pool_7x7', 0), ('max_pool_3x3', 2), ('max_pool_5x5', 1), ('max_pool_3x3', 2), ('avg_pool_3x3', 1), ('max_pool_5x5', 2)], reduce_concat=range(2, 6))2021-06-09 13:21:45,591 Restricting skipconnect...2021-06-09 13:21:45,620 Total searching time: 73016s
```

分析： 一家独大， normal 里面 全是**skip-connection** 





21. search-sep-20210609-171251 （**pdarts_modify**）再次确认随机采样的强化效果，打死卷积核参数

    随机：

    ![image-20210609200945879](EXP/image-20210609200945879.png)

    

    分析：  没效果， 可能时间不够， 直接对比以下两个实验： 

 **argmax** （pdarts_modify）：search-sep-20210609-195419

![image-20210610161311119](EXP/image-20210610161311119.png)

分析：  左边是 multinominal 出来的最优架构，   右边是当前的argmax 最优架构。 

策略：random ， multinominal，  argmax 

涉及的环节：体现出off-policy    

1. 采样  take action  and  get validation  reward :    multinomial sample  by  probability
2. 淘汰架构时（阶段性淘汰）：  argmax N / argmin N      或者   一次性选择架构时（似乎更加不合理）： argmax   ----- 根据对上图的分析， 在一定的搜索方向上， multinominal 采集出来的架构（左边）比 argmax （右边） 的要更好， 那么是否可以改变最后策略的应用方式，就是淘汰或者选择架构的时候--- **加一定的随机因素， 或者投票机制。** 

 

分析一下， best_reward 在不同环节（stage） 上， 同 max_arch_reward 的关系 。理论上在最后一个stage， 两者的match 程度（波动 和  精度效果） 会更高 。  



**random**  （**pdarts**）： search-sep-20210609-215504  （训练精度）

![image-20210610100248454](EXP/image-20210610100248454.png)



22. search-sep-20210610-125518 (**pdarts**)： 重复上面实验，预训练次数多一些， 50 次    ----  停止  没有显著效果

    ![image-20210611103319282](EXP/image-20210611103319282.png)



23：(pdarts_modify )   search-try-20210610-164322  搜索实验      无epsilon 的multinomial 

![image-20210611204210465](EXP/image-20210611204210465.png)

![image-20210611204326582](EXP/image-20210611204326582.png)


```
2021-06-11 19:33:36,415 Genotype(normal=[('sep_conv_3x3', 0), ('conv 1x1', 1), ('sep_conv_3x3', 0), ('conv 1x1', 1), ('conv 1x1', 0), ('conv 1x1', 1), ('conv 1x1', 0), ('conv 1x1', 1)], normal_concat=range(2, 6), reduce=[('max_pool_5x5', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_5x5', 1), ('avg_pool_3x3', 3), ('max_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))2021-06-11 19:33:36,415 Restricting skipconnect...2021-06-11 19:33:36,442 Total searching time: 96613s
```

早期就收敛



23. ​    search-try-20210611-002256   （pdarts）启动一个epsilon greedy 的 搜索。  XXXXX 代码同步错误了 。重启

    search-try-20210611-135812  (pdarts  DF)   epsilon = 0.3 

    ![image-20210611162545505](EXP/image-20210611162545505.png)

分析：max 很早就收敛  



  search-try-20210611-185359（pdarts）  增加 argmin 的监控， 同时， **epsilon 修改为 0.6**   **,LR 改为1e-3**    **bs=256** epoch=25   -- 待check 监控

 ![image-20210612095021649](EXP/image-20210612095021649.png)

![image-20210612095436205](EXP/image-20210612095436205.png)

**分析: 参数量没有极致增大 .** 

multinomial 必须配合学习率， 否则失效 , 有一个点持续增大. 



24,    search-try-20210611-210200   (pdarts_modify)     **epsilon 修改为 0  ,**  **修改LR为 1e-3**  **bs=192** epoch=20

​        ![image-20210612094940729](EXP/image-20210612094940729.png)

![image-20210612095509853](EXP/image-20210612095509853.png)

​     分析: 参数量没有极致增大, 说明和LR 有关系.             epsilon 存在一定的关系. 

​                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

23. 确定 策略后的搜索   ， 比较 带epsilon 和不带epsilon 

search-try-20210612-095828 (pdarts_modify )    epsilon=0.0



search-try-20210612-095736 (pdarts)  epsilon=0.2 



24. ​    search-try-20210613-204816(pdarts_modify)  恢复搜索空间到 pdarts 和PCDARTs   的设置 。   同时增加了 grad_clip_norm 

    分析： 全是pooling ， 

25. search-try-20210613-224014 (pdarts_original)  监控pdarts 原始的index 分布 。 

    ![image-20210615105853461](EXP/image-20210615105853461.png)

  分析： 发现 weights 变化很小     --- **架构可微的方式， 有效的过滤掉了 pooling 操作。** 

26. search-try-20210614-201028  pdarts_modify      修改  LR and weight decay    ---- **效果仍然是MAX_POOLING多， 和24 效果差不多**    
27.  search-try-20210614-231311（ pdarts ）   分离搜索空间， normal/reduce  ， 搜索空间为 8/6 配置 。 



28. 分离搜索空间， normal/reduce  ， 搜索空间为10/6 配置

29. ​     加入参数量的  惩罚    的搜索实验    。 



26. 恢复ADAM超参为pdarts的超参  





1. 针对cifar-这样的数据集，并不需要太深的模型 ？？？  ---- 看下pdarts 原文， 其他的文章和综述，是否有针对这个的抨击，讨论.    缩短 step ， layers  等 架构深度超参



23. 探索和利用结合 。 交替 采样。   XXXXXX  用epsilon greedy 策略。 

监控pdarts/darts 的原始效果， 准备分析和评论。 

把这个架构 search-try-20210607-145425  重头训练一遍    。   









































SHEBO：  随机采集出来的  最好架构 中的10个  ， 可以取平均， 然后和当前收敛的架构进行比较。 收敛的不能和这个平均值持平  ？？？？？、  --- 需要实验体现出来。 



正式流程（tain/train_arch 交替的方式）的fomulate ：  强化学习过程。 策略： 架构权重      a:   选择一堆子架构      r:  得到精度值，  更新策略： 根据精度值和    e：依该架构权重，更新得到一个新的权重的超网络，。  这个过程本来就是不断缩减搜索空间， 让当前优的架构得到更优的训练， 从而强化的过程。 

测试流程（一个supernet 训练好了， 一直不变，然后在里面强化搜索）： **策略**： 架构权重    全部一样， 只是environment  不做更新。 

**如果采用测试流程（训练到一段时间， 比如超网络精度到达 95 %）就启动搜索，这样相当于失去了探索的过程（不是子网，超网络一同进化） ， 最后可能快速收敛到一个最优的子网精度就是最高 50 %，  从50% 到 90% 的进化的过程就丧失了。**      所以 ， 不能采用sep 的方式来。    





# Off-policy policy gradient reinforcement learning algorithms

On-policy algorithms are using target policy to sample the actions, and the same policy is used to optimise for. REINFORCE, and vanilla actor-critic algorithms are an example of on-policy methods. In the off-policy algorithm, actions are sample using behaviour policy and separate target policy is used to optimise for. Q-learning is an example of the off-policy algorithm where ϵ-greedy is the behaviour policy and target policy or update policy is the absolute greedy regardless of the behaviour policy.

Off-policy algorithms have mainly two advantages over on-policy methods[1]:

1. Sample efficiency: It does not require full trajectories as it uses temporal difference learning approach and can reuse the past episodes as it used experience replay buffer.
2. Better exploration: sample collected using behaviour policy which id different from the target policy

Now let’s see how the off-policy gradient works.

The objective function of the REINFORCE (on-policy algorithm) is defined as below:

重要性采样 。 。。





Torwards  the local-optimal  and  un  differentiable   target    NAS      ----  致力于解决  NAS 的问题 。 

 因为有采样  ， 可以执行策略，  不容易局部最优   



and escaping  the local-optimal area  .



1.  回答为什么选择渐进式的基线 。   逐步淘汰， 从宽而浅的网络， 到浅而深， 有效避免局部最优。 