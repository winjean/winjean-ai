* Faiss：
Facebook开源的库，专为密集向量设计，支持多种索引类型，如Flat、IVF、PQ等。
特点是速度快、内存占用低，适合大规模向量数据的近似最近邻搜索。
* Milvus：
由Zilliz开发，支持多种向量索引类型，包括HNSW、IVF等。
具有良好的可扩展性，支持分布式部署，适用于大规模数据集。
param 参数详解
param 参数是一个字典，通常包含以下键值对：
  metric_type:
    指定使用的距离度量类型。
    常见的距离度量类型包括：
      L2: 欧几里得距离。
      IP (Inner Product): 内积。
      COSINE (Cosine Similarity): 余弦相似度。
      JACCARD: Jaccard 相似度。
      HAMMING: 汉明距离。
      TANIMOTO: Tanimoto 相似度。
      SUBSTRUCTURE: 子结构相似度。
      SUPERSTRUCTURE: 超结构相似度。

  params:
    指定索引参数，通常用于优化搜索性能。
    不同的索引类型有不同的参数，常见的参数包括：
      nprobe: 对于 IVF 索引类型，表示在搜索过程中考虑的候选簇的数量。
      search_list: 对于 HNSW 索引类型，表示搜索时的候选列表大小。
      efConstruction: 对于 HNSW 索引类型，表示构建索引时的候选列表大小。
      efSearch: 对于 HNSW 索引类型，表示搜索时的候选列表大小。
expr:
  过滤表达式，用于进一步过滤搜索结果。例如，expr="id > 100" 可以过滤出 id 大于 100 的结果
* Chroma：
一个开源的向量数据库，易于使用，支持多种索引类型。
特别适合快速原型设计和小型项目。