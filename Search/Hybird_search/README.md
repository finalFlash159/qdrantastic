## Hybrid and Multi-Stage Queries


Thành phần chính để kết hợp các truy vấn kết hợp (combinations) trong `Query API` là tham số `prefetch`, cho phép thực hiện các sub-request (yêu cầu phụ).

Cụ thể, mỗi khi một truy vấn có ít nhất một prefetch, Qdrant sẽ:

- Thực hiện truy vấn prefetch (hoặc nhiều truy vấn prefetch),
- Sau đó áp dụng truy vấn chính lên kết quả của các prefetch đó.

Hơn nữa, các **prefetch** cũng có thể có **prefetch** riêng của chúng, cho phép lồng các truy vấn phụ **(nested prefetches)**.


## Hybrid Search

![search](https://qdrant.tech/docs/fusion-idea.png)


### Fusing results from multiple queries

Trong tìm kiếm văn bản, thông thường ta cần kết hợp các vector dense và sparse để vừa tận dụng được ngữ nghĩa tổng quát, vừa đảm bảo khớp các từ cụ thể.

Qdrant có hai cách để kết hợp kết quả từ các truy vấn khác nhau:

- `rrf` - Reciprocal Rank Fusion
Phương pháp này xem xét vị trí (rank) của kết quả trong từng truy vấn, và tăng trọng số cho những kết quả xuất hiện gần đầu trong nhiều truy vấn.
	```python
	client.query_points(
    collection_name="{collection_name}",
    prefetch=[
        models.Prefetch(
            query=models.SparseVector(indices=[1, 42], values=[0.22, 0.8]),
            using="sparse",
            limit=20,
        ),
        models.Prefetch(
            query=[0.01, 0.45, 0.67],  # <-- dense vector
            using="dense",
            limit=20,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
	)
	```

- `dbsf` - Distribution-Based Score Fusion (áp dụng từ v1.11.0)
Phương pháp này chuẩn hóa điểm số của các điểm trong mỗi truy vấn (sử dụng trung bình ± độ lệch chuẩn thứ 3 làm giới hạn), sau đó cộng điểm số của cùng một điểm qua các truy vấn khác nhau.

>Phương pháp này không lưu trạng thái (stateless) và tính toán giới hạn chuẩn hóa chỉ dựa trên kết quả của mỗi truy vấn, không phụ thuộc vào tất cả các điểm số đã được thấy.

## Multi-stage Queries

Trong nhiều trường hợp, sử dụng biểu diễn vector lớn hơn cho kết quả tìm kiếm mang lại độ chính xác cao hơn, nhưng lại tốn kém hơn về tính toán.

**Kỹ thuật multi-stage:**

- Giai đoạn 1: Sử dụng biểu diễn vector nhỏ hơn, rẻ tiền để lấy danh sách ứng viên lớn.
- Giai đoạn 2: Sau đó, tái tính điểm (re-score) các ứng viên này bằng biểu diễn vector lớn hơn và chính xác hơn.

**Có vài cách xây dựng kiến trúc tìm kiếm theo ý tưởng này:**

- Dùng vector lượng tử **(quantized vectors)** làm giai đoạn đầu và vector đầy đủ **(full-precision vectors)** làm giai đoạn thứ hai.
- Sử dụng **Matryoshka Representation Learning** (MRL): Tạo vector ứng viên với vector ngắn hơn, sau đó tinh chỉnh bằng vector dài hơn.
- Sử dụng vector dense thông thường để tiền truy vấn (pre-fetch) ứng viên, rồi re-score chúng với một mô hình multi-vector như ColBERT.

## Re-scoring examples
Fetch 1000 results using a shorter MRL byte vector, then re-score them using the full vector and get the top 10.

```python
client.query_points(
    collection_name="{collection_name}",
    prefetch=models.Prefetch(
        query=[1, 23, 45, 67],  # <------------- small byte vector
        using="mrl_byte",
        limit=1000,
    ),
    query=[0.01, 0.299, 0.45, 0.67],  # <-- full vector
    using="full",
    limit=10,
)
```

Fetch 100 results using the default vector, then re-score them using a multi-vector to get the top 10.

```python
client.query_points(
    collection_name="{collection_name}",
    prefetch=models.Prefetch(
        query=[0.01, 0.45, 0.67, 0.53],  # <-- dense vector
        limit=100,
    ),
    query=[
        [0.1, 0.2, 0.32],  # <─┐
        [0.2, 0.1, 0.52],  # < ├─ multi-vector
        [0.8, 0.9, 0.93],  # < ┘
    ],
    using="colbert",
    limit=10,
)
```

It is possible to combine all the above techniques in a single query:

```python
client.query_points(
    collection_name="{collection_name}",
    prefetch=models.Prefetch(
        prefetch=models.Prefetch(
            query=[1, 23, 45, 67],  # <------ small byte vector
            using="mrl_byte",
            limit=1000,
        ),
        query=[0.01, 0.45, 0.67],  # <-- full dense vector
        using="full",
        limit=100,
    ),
    query=[
        [0.17, 0.23, 0.52],  # <─┐
        [0.22, 0.11, 0.63],  # < ├─ multi-vector
        [0.86, 0.93, 0.12],  # < ┘
    ],
    using="colbert",
    limit=10,
)
```