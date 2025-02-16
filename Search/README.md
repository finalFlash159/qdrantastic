# Query API

## Nearest Neighbors Search	(Vector Similarity Search, also known as k-NN)


```python
client.query_points(
    collection_name="{collection_name}",
    query=[0.2, 0.1, 0.9, 0.7], # <--- Dense vector
	query="43cf51e2-8777-4f52-bc74-c2cbde0c8b04", # <--- point id
)
```

Example:
```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.query_points(
    collection_name="{collection_name}",
    query=[0.2, 0.1, 0.9, 0.7],
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="city",
                match=models.MatchValue(
                    value="London",
                ),
            )
        ]
    ),
    search_params=models.SearchParams(hnsw_ef=128, exact=False),
    limit=3,
)
```

In this example, we are looking for vectors similar to vector [0.2, 0.1, 0.9, 0.7]. Parameter limit (or its alias - top) specifies the amount of most similar results we would like to retrieve.

Values under the key `params` specify custom parameters for the search. Currently, it could be:

`hnsw_ef`  - value that specifies `ef` parameter of the HNSW algorithm.
`exact` - option to not use the approximate search (ANN). If set to true, the search may run for a long as it performs a full scan to retrieve exact results.
`indexed_only` - With this option you can disable the search in those segments where vector index is not built yet. This may be useful if you want to minimize the impact to the search performance whilst the collection is also being updated. Using this option may lead to a partial result if the collection is not fully indexed yet, consider using it only if eventual consistency is acceptable for your use case.


### Filtering results by score

In addition to payload filtering, it might be useful to filter out results with a low similarity score. For example, if you know the minimal acceptance score for your model and do not want any results which are less similar than the threshold. In this case, you can use `score_threshold` parameter of the search query. It will exclude all results with a score worse than the given.

>This parameter may exclude lower or higher scores depending on the used metric. For example, higher scores of Euclidean metric are considered more distant and, therefore, will be excluded.

### Batch search API

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

filter_ = models.Filter(
    must=[
        models.FieldCondition(
            key="city",
            match=models.MatchValue(
                value="London",
            ),
        )
    ]
)

search_queries = [
    models.QueryRequest(query=[0.2, 0.1, 0.9, 0.7], filter=filter_, limit=3),
    models.QueryRequest(query=[0.5, 0.3, 0.2, 0.3], filter=filter_, limit=3),
]

client.query_batch_points(collection_name="{collection_name}", requests=search_queries)
```

---
## Recommendation API

In addition to the regular search, Qdrant also allows you to search based on multiple positive and negative examples

```python
client.query_points(
    collection_name="{collection_name}",
    query=models.RecommendQuery(
        recommend=models.RecommendInput(
            positive=[100, 231],
            negative=[718, [0.2, 0.3, 0.4, 0.5]],
            strategy=models.RecommendStrategy.AVERAGE_VECTOR,
        )
    ),
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="city",
                match=models.MatchValue(
                    value="London",
                ),
            )
        ]
    ),
    limit=3,
)
```
**Strategy:**

`average_vector`
```python
avg_positive + avg_positive - avg_negative
```
\
`best_score`
```python
let score = if best_positive_score > best_negative_score {
    best_positive_score
} else {
    -(best_negative_score * best_negative_score)
};
```


**Multi-vector**
If the collection was created with multiple vectors, the name of the vector should be specified in the recommendation request:

```python
client.query_points(
    collection_name="{collection_name}",
    query=models.RecommendQuery(
        recommend=models.RecommendInput(
            positive=[100, 231],
            negative=[718],
        )
    ),
    using="image",
    limit=10,
)
```

---

## Discovery API

### Mục tiêu
- Cho phép tìm kiếm các điểm (points) trong không gian vector dựa trên:
  - **Target vector** và **Các cặp positive-negative context.**
- **Discovery search:** Tìm các điểm gần target nhất, nhưng bị ràng buộc bởi context.
- **Context search:** Tìm các điểm nằm trong vùng “tốt” nhất, chỉ dựa trên context mà không cần target cụ thể.

![discovery](https://qdrant.tech/docs/discovery-search.png)

### Cách thức hoạt động
- **Bước 1:** Bạn cung cấp một target vector (ví dụ: `[0.2, 0.1, 0.9, 0.7]`).
- **Bước 2:** Đồng thời, bạn cung cấp một danh sách các cặp context, mỗi cặp bao gồm:
  - **positive:** 
  - **negative:** 
- **Bước 3:** Điểm số của mỗi điểm được tính dựa trên:
  - Hàm tương đồng giữa target và các ví dụ dương,
  $$\text{rank}(v^+, v^-) = \begin{cases} 1, &\quad s(v^+) \geq s(v^-) \\-1, &\quad s(v^+) < s(v^- )\end{cases}$$
$$v^+\text{: positive example}$$
$$v^-\text{: negative example}$$
  - Trừ đi sự phạt nếu điểm đó gần với ví dụ âm.
  - Hàm sigmoid được dùng để chuẩn hóa điểm số về khoảng [0, 1].
  - Cuối cùng, tổng các thứ hạng (ranks) được dùng để xác định mức độ thuộc về vùng dương.
  $$\text{discovery score} = \text{sigmoid}(s(v_t))+ \sum \text{rank}(v_i^+, v_i^-),$$

```python
discover_queries = [
    models.QueryRequest(
        query=models.DiscoverQuery(
            discover=models.DiscoverInput(
                target=[0.2, 0.1, 0.9, 0.7],
                context=[
                    models.ContextPair(
                        positive=100,
                        negative=718,
                    ),
                    models.ContextPair(
                        positive=200,
                        negative=300,
                    ),
                ],
            )
        ),
        limit=10,
    ),
]

client.query_batch_points(
    collection_name="{collection_name}", requests=discover_queries
)
```
**Lưu ý về discovery search:**

- Khi cung cấp các id làm ví dụ, chúng sẽ bị loại khỏi kết quả.
- Điểm số luôn được sắp xếp giảm dần (giá trị lớn hơn có nghĩa là tốt hơn), bất kể metric nào được sử dụng.
- Vì không gian bị ràng buộc nghiêm ngặt bởi context, độ chính xác có thể giảm khi dùng cài đặt mặc định. Để khắc phục, tăng tham số `ef `thành số lơn hơn 64 (ví dụ: ``"params": { "ef": 128 })`` sẽ giúp cải thiện kết quả so với giá trị mặc định (16).

---
## Context Search

### Mục tiêu
- Tìm kiếm các điểm chỉ dựa trên các cặp context mà không cần target vector cụ thể.
![context](https://qdrant.tech/docs/context-search.png)
### Cách thức hoạt động
- Sử dụng một hàm số được điều chỉnh từ khái niệm **triplet-loss** nhằm “hướng” tìm kiếm tới các vùng có ít ví dụ negative (âm) hơn.
- **Điểm số cao (tốt nhất là 0.0)** biểu thị rằng điểm đó hoàn toàn nằm trong vùng dương; càng gần ví dụ âm thì loss càng cao.

$$\text{context score} = \sum \min(s(v^+_i) - s(v^-_i), 0.0)$$

---
## Distance Matrix API

### Mục tiêu
- Tính toán khoảng cách giữa các cặp vector được lấy mẫu từ collection.
- Trả về kết quả dưới dạng **ma trận rời rạc (sparse matrix).**

### Ứng dụng
- Khai thác dữ liệu, ví dụ như:
  - Phân cụm (clustering)
  - Trực quan hóa (visualization)
  - Giảm chiều (dimension reduction)

### Quy trình
- **Bước 1:** Lấy mẫu một số lượng vector từ collection (ví dụ: 100 điểm).
- **Bước 2:** Với mỗi điểm, tính toán 10 điểm gần nhất trong số các mẫu.
- **Kết quả:** Tổng số 1000 điểm số được biểu diễn dưới dạng ma trận rời rạc.

### Định dạng đầu ra
- **Pairwise format:**  
  - Trả về danh sách các cặp (id, id, score).
- **Offset format:**  
  - Trả về 4 mảng:
    - `offsets_row` và `offsets_col`: Đại diện cho vị trí của các giá trị khác không trong ma trận.
    - `scores`: Chứa các giá trị khoảng cách.
    - `ids`: Chứa các id điểm tương ứng với các giá trị khoảng cách.

