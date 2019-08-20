# Tencent2019_Finals_Rank1st
# 2019腾讯广告算法大赛完整代码（冠军）

## 1.环境配置和所需依赖库
- scikit-learn
- tqdm
- lightgbm
- pandas
- numpy
- scipy
- tensorFlow=1.12.0 (其他版本≥1.4且不等于1.5或1.6)
- Linux Ubuntu 16.04, 128G内存，一张显卡

## 2.复现结果
原始数据统一保存在data文件夹,包括复赛AB榜数据(不要有子目录)。
``` shell
bash run.sh
```
最后输出结果为./submission.csv

## 3.步骤说明
（1）原始数据统一保存在data文件夹,包括复赛AB榜数据(不要有子目录)。
（2）run.sh会依次运行文件夹A、gdy、wh和lyy中的run.sh文件。
（3）A、gdy和wh会分别从data中读取原始数据，提取特征，然后生成结果。
（4）lyy文件夹用来对gdy和wh产出的结果进行融合，然后得到最终的提交结果。


## 4.特征说明
（1）特征维度：主要从历史角度和全局角度去构建特征，具体维度有前一天、最近一天、历史所有、前n天和五折交叉统计全局特征。
（2）基础特征：广告在当天的竞争胜率，广告在当天竞争次数，广告在当天竞争胜利次数，广告在当天竞争失败次数。然后可以扩展为商品id和账户id等。然后将基础特征按特征维度进行构造。对于新广告，可以将商品id和账户id与基础特征进行组合。


## 5.模型说明
### （1）目录A
#### 模型: lightgbm
```python
    参数：  lgb_model = lgb.LGBMRegressor( num_leaves=256, reg_alpha=0., reg_lambda=0.01, objective='mae', metric=False,max_depth=-1, learning_rate=0.03,min_child_samples=25,  n_estimators=1000, subsample=0.7, colsample_bytree=0.45)
```
### （2）目录gdy 
#### 模型: Xdeepfm https://arxiv.org/pdf/1803.05170.pdf
```python
    参数：  hparam=tf.contrib.training.HParams(
            model='xdeepfm',
            norm=True,
            batch_norm_decay=0.9,
            hidden_size=[1024,512],
            dense_hidden_size=[300],
            cross_layer_sizes=[128,128],
            k=8,
            single_k=8,
            num_units=64,
            num_layer=1,
            encoder_type='uni',
            max_length=100,
            cross_hash_num=int(5e6),
            single_hash_num=int(5e6),
            multi_hash_num=int(1e6),
            sequence_hash_num=int(1e4),
            batch_size=128,
            infer_batch_size=2**10,
            optimizer="adam",
            dropout=0,
            kv_batch_num=20,
            learning_rate=0.0002,
            num_display_steps=1000,
            num_eval_steps=1000,
            epoch=1, #don't modify
            metric='score',
            activation=['relu','relu','relu'],
            init_method='tnormal',
            cross_activation='relu',
            init_value=0.001,
            single_features=single_features,
            cross_features=cross_features,
            multi_features=multi_features,
            sequence_features=sequence_features,
            dense_features=dense_features,
            kv_features=kv_features,
            label='imp',
            model_name="xdeepfm",
            bid='bid_feature',
            use_bid=False,
            bias=1)
```
#### 模型: lightgbm
```python
    参数：  lgb_model = lgb.LGBMRegressor(num_leaves=256, reg_alpha=0., reg_lambda=0.01,objective='mae',metric=False,max_depth=-1, learning_rate=0.03,min_child_samples=25, n_estimators=1200, subsample=0.7, colsample_bytree=0.45)
```
### （3）目录wh
#### 模型: lightgbm
```python
    参数：  lgb_params = {'num_leaves': 2**7-1,
              'min_data_in_leaf': 25, 
              'objective':'regression_l2',
              'max_depth': -1,
              'learning_rate': 0.1,
              'min_child_samples': 20,
              'boosting': 'gbdt',
              'feature_fraction': 0.6,
              'bagging_fraction': 0.9,
              'bagging_seed': 11,
              'metric': 'mae',
              'seed':1024,
              'lambda_l1': 0.2}
```

### 竞赛社区（知识星球）
```
就在前不久我和Datawhale的晶晶，还有杰少一起计划推出有关数据竞赛的高质量社区，并邀请了圈内大咖，有Kaggle上的Grand Master，也有天池的数据科学家，还有顶会科研大佬。筹备社区前，我们也一直考虑如何提供更好的体验和学习服务，为此做出大量的筹划，力求为学习者提供数据竞赛的一站式服务。

【Kaggle数据竞赛】
```
![avatar](picture/Mind_Mapping.jpg)
```
【适合群体】
1. 对数据竞赛特别感兴趣的你
2. 想在数据科学领域一展拳脚的你
3. 学习了一堆理论但缺乏实践的你
4. 想找工作但缺乏相关项目的你
5. 想结识更多优秀同伴和竞赛大佬

【星球嘉宾】
```
![avatar](picture/guest.jpg)
```
【加入方式】
知识星球「Kaggle数据竞赛」
1. 扫描下方二维码申请加入
2.「特价」限额折扣，前1k人享受5折特惠，99元/年，之后将恢复至199/年
3. 支付完成后，加微信号datawhale2020，邀您进会员群
```
![avatar](picture/Kaggle_Datawhale.jpg)
