# wahaha
[Runner-up Award](https://2024.sigmod.org/sigmod_awards.shtml) code for [2024 SIGMOD Programming Contest](https://dbgroup.cs.tsinghua.edu.cn/sigmod2024/leaders.shtml)

比赛要求在20分钟内完成编译、建图、搜索、保存搜索结果等任务，因此不是一个纯搜索比赛。在纯ANNS任务上，以SIFT1M距离，8线程情况下为其构建索引通常需要30秒以上，而搜索recall在0.9时，单线程每秒搜索个数就可达到10k以上。那么假如有10k的查询，单线程就可在1秒内完成查询。因此其关键点在于构建索引的时间，赢索引者赢比赛。索引构建的时间上，NSG > HNSW = 2X kgraph = 4X RNN-Descent，在搜索效率差不多的情况下，以上数据基于我的经验，当然与建图参数有很大关系。可以看到，在比赛中我使用的建图方法是Relative NN-Descent（RNN-Descent）。

除去构建索引的时间，这里搜索是混合向量搜索。具体的，有4种子任务：
```txt
There are four types of queries, i.e., the query_type takes values from 0, 1, 2 and 3. The 4 types of queries correspond to:

If query_type=0: Vector-only query, i.e., the conventional approximate nearest neighbor (ANN) search query.
If query_type=1: Vector query with categorical attribute constraint, i.e., ANN search for data points satisfying C=v.
If query_type=2: Vector query with timestamp attribute constraint, i.e., ANN search for data points satisfying l≤T≤r.
If query_type=3: Vector query with both categorical and timestamp attribute constraints, i.e. ANN search for data points satisfying C=v and l≤T≤r.
```
先不说查询向量，就数据样本点而言，在向量本身前头，还有2个维度的数据，分别是`类别`和`时间戳`。那么搜索时候就可以：
1. 不管这两个维度，纯搜索最近邻向量；
2. 指定某个类别，搜索这个类别内的最近邻向量；
3. 指定某个时间间隔，搜索这个间隔内的最近邻向量；
4. 指定类别和时间间隔，搜索符合两个要求下的最近邻向量。



update soon.
