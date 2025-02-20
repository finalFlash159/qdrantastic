## Optimization is a balancing Act

![](https://qdrant.tech/articles_data/vector-search-resource-optimization/optimization.png)

| **Intended Result** | **Optimization Strategy** |
| --- | --- |
| High Search Precision + Low Memory Expenditure | On-Disk Indexing |
| Low Memory Expenditure + Fast Search Speed | Quantization |
| High Search Precision + Fast Search Speed | RAM Storage + Quantization |
| Balance Latency vs Throughput | Segment Configuration |

## Configure Indexing for Faster Searches

### HNSW (Hierarchical Navigable Small World Graph) algorithm a

HNSW xây dựng một đồ thị phân cấp cho phép tìm kiếm gần đúng nhanh chóng và hiệu quả trong không gian vector lớn. Bằng cách sử dụng nhiều tầng, mỗi tầng có mật độ khác nhau, thuật toán tìm kiếm từ mức tổng quát đến chi tiết, giúp tìm được các vector gần nhất một cách nhanh chóng mà không phải quét toàn bộ dữ liệu. Các tham số như `m`, `ef_construct`, và `ef` có thể điều chỉnh để cân bằng giữa tốc độ, độ chính xác và việc sử dụng bộ nhớ.

- `m` quy định số lượng cạnh (edges) tối đa mà mỗi nút trong đồ thị HNSW có thể có.  
    (giá trị `m` lớn thì tăng độ chính xác nhưng đánh đổi bộ nhớ và thời gian xây dựng)
- `ef_construct` chỉ định số lượng hàng xóm (neighbors) được xem xét trong quá trình xây dựng đồ thị HNSW.
    
    ```python
    client.update_collection(
        collection_name="{collection_name}",
        vectors_config={
            "my_vector": models.VectorParamsDiff(
                hnsw_config=models.HnswConfigDiff(
                    m=32,
                    ef_construct=123,
                ),
            ),
        }
    )
    ```
    
- `ef` quyết định số lượng hàng xóm được đánh giá trong quá trình thực hiện truy vấn.
    
    ```python
    client.query_points(
       collection_name="{collection_name}",
       query=[...]
       search_params=models.SearchParams(hnsw_ef=128, exact=False),
    )
    ```
    

![](https://qdrant.tech/articles_data/vector-search-resource-optimization/hnsw.png)  
![](https://qdrant.tech/articles_data/vector-search-resource-optimization/hnsw-parameters.png)  
**Quá trình tìm kiếm**  
**Mỗi tầng:**  
Ở mỗi tầng, thuật toán bắt đầu từ một node và duyệt qua các node lân cận để tìm node nào gần query hơn. Nếu không có node nào cải thiện được khoảng cách, nó dừng lại.

**Hạ tầng:**  
Sau khi tìm kiếm ở một tầng, thuật toán sử dụng node tốt nhất làm điểm khởi đầu cho tầng dưới. Quá trình này tiếp tục cho đến khi đạt tầng 0.

**Trường hợp không có node cải thiện:**  
Nếu tại một tầng nào đó không có node nào gần query hơn, thuật toán giữ nguyên điểm hiện tại và chuyển xuống tầng dưới, sử dụng điểm đó làm điểm khởi đầu.

## Data Compression Techniques

Nén dữ liệu hiệu quả là yếu tố then chốt giúp giảm dung lượng bộ nhớ và tăng tốc độ truy vấn mà vẫn đảm bảo độ chính xác của các vector. Qdrant – một hệ quản trị cơ sở dữ liệu vector – cung cấp nhiều kỹ thuật nén và tối ưu hóa như ***lượng tử hóa (quantization)***, ***đa người dùng (multitenancy)***, ***phân mảnh (sharding)*** và các phương pháp tối ưu truy vấn.

### Lượng Tử Hóa (Quantization)

**1\. Scalar Quantization**  
Phương pháp này giảm số bit cần thiết cho từng thành phần của vector. Ví dụ, Qdrant chuyển đổi giá trị 32-bit float (float32) thành 8-bit unsigned integer (uint8) giúp giảm dung lượng bộ nhớ xuống đến 1/4 (giảm 75% kích thước ban đầu).

![](https://qdrant.tech/articles_data/vector-search-resource-optimization/scalar-quantization.png)

**Benefits of Scalar Quantization:**

| Benefit | Description |
| --- | --- |
| Memory usage will drop | Compression cuts memory usage by a factor of 4. Qdrant compresses 32-bit floating-point values (float32) into 8-bit unsigned integers (uint8). |
| Accuracy loss is minimal | Converting from float32 to uint8 introduces a small loss in precision. Typical error rates remain below 1%, making this method highly efficient. |
| Best for specific use cases | To be used with high-dimensional vectors where minor accuracy losses are acceptable. |

**Setup when create collection**

```python
client.create_collection(
   collection_name="{collection_name}",
   vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
   quantization_config=models.ScalarQuantization(
       scalar=models.ScalarQuantizationConfig(
           type=models.ScalarType.INT8,
           quantile=0.99,
           always_ram=True,
       ),
   ),
)
```

- `type`: Specifies the quantized vector type (currently supports only int8).
- `quantile`: Sets bounds for quantization, excluding outliers. For example, 0.99 excludes the top 1% of extreme values to maintain better accuracy. (Xác định ngưỡng để loại trừ các giá trị ngoại lai)
- `always_ram`: Keeps quantized vectors in RAM to speed up searches.

### 2. Binary Quantization
Phương pháp này chuyển đổi mỗi thành phần của vector thành chỉ 1 bit, giúp tối đa hóa hiệu quả nén. Nhờ đó, dung lượng bộ nhớ giảm tới 1/32 và tốc độ truy vấn có thể nhanh hơn lên đến 40 lần.
![](https://qdrant.tech/articles_data/vector-search-resource-optimization/binary-quantization.png)

- Tính toán độ tương đồng nhanh: Sử dụng các phép so sánh dựa trên khoảng cách `Hamming`.
- Phù hợp với dữ liệu lớn: Đặc biệt hiệu quả với các mô hình nhúng (ví dụ: `OpenAI’s text-embedding-ada-002`).
- Cần điều chỉnh sự chính xác: Có thể cần áp dụng thêm kỹ thuật như `rescore` hoặc `oversampling` để bù đắp sự mất mát về độ chính xác.

```python
client.create_collection(
   collection_name="{collection_name}",
   vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
   quantization_config=models.BinaryQuantization(
       binary=models.BinaryQuantizationConfig(
           always_ram=True,
       ),
   ),
)
```

>By default, quantized vectors load like original vectors unless you set always_ram to True for instant access and faster queries.

**Note**: Binary Quantization requires a centered distribution of vector components. It is recommended to use with tested models only

## Scaling the Database (Mở rộng cơ sở dữ liệu)
### Multitenancy (Đa người dùng)
Kiến trúc cho phép nhiều người dùng (hoặc tenant) chia sẻ cùng một collection nhưng được phân chia logic riêng biệt.
- Cách ly dữ liệu theo tenant: Đảm bảo dữ liệu của từng người dùng không bị trộn lẫn.
- Giảm tải tài nguyên: So với việc tạo một collection riêng cho mỗi người dùng, việc sử dụng một collection duy nhất với phân vùng logic tiết kiệm tài nguyên hơn.

![](https://qdrant.tech/articles_data/vector-search-resource-optimization/multitenancy.png)

```python
client.create_payload_index(
    collection_name="{collection_name}",
    field_name="group_id",
    field_schema=models.KeywordIndexParams(
        type="keyword",
        is_tenant=True,
    ),
)
```

Tạo chỉ mục payload dựa trên `group_id` với tham số `is_tenant=True` sẽ giúp Qdrant tổ chức dữ liệu sao cho các vector của cùng một tenant được lưu trữ gần nhau.
Upload:
```python
client.upsert(
   collection_name="{collection_name}",
   points=[
       models.PointStruct(
           id=1,
           payload={"group_id": "user_1"},
           vector=[0.9, 0.1, 0.1],
       ),

       models.PointStruct(
           id=2,
           payload={"group_id": "user_2"},
           vector=[0.5, 0.9, 0.4],
       )
   ]
)
```

### Sharding
Sharding is a critical strategy in Qdrant for splitting collections into smaller units, called shards, to efficiently distribute data across multiple nodes. It’s a powerful tool for improving scalability and maintaining performance in large-scale systems.
### User-Defined Sharding:
User-Defined Sharding allows you to take control of data placement by specifying a shard key. This feature is particularly useful in multi-tenant setups, as it enables the isolation of each tenant’s data within separate shards, ensuring better organization and enhanced data security.

![](https://qdrant.tech/articles_data/vector-search-resource-optimization/user-defined-sharding.png)
```python
client.create_collection(
    collection_name="my_custom_sharded_collection",
    shard_number=1,
    sharding_method=models.ShardingMethod.CUSTOM
)
client.create_shard_key("my_custom_sharded_collection", "tenant_id")
```

```python
client.upsert(
    collection_name="my_custom_sharded_collection", 
    points=[
        models.PointStruct(
            id=1111, 
            vector=[0.1, 0.2, 0.3]
        )
    ], 
    shard_key_selector="tenant_1"
)
```

## Storage: Disk vs RAM

### Lưu Trữ Trong RAM
- Lưu trữ dữ liệu (nhất là vector đã được lập chỉ mục) trong bộ nhớ RAM cho phép truy cập cực nhanh, rất quan trọng đối với các ứng dụng cần hiệu suất cao.
Hạn chế:
- Dung lượng RAM có giới hạn; nếu dataset quá lớn, việc lưu trữ toàn bộ trong RAM có thể không khả thi.
### Lưu Trữ Trên Disk
- Dữ liệu ít cần truy xuất (như payload hay thông tin phụ) có thể được lưu trữ trên ổ cứng.
- Local SSD được khuyến nghị vì tốc độ truy xuất cao, so với các giải pháp lưu trữ qua mạng (NAS) có thể gây độ trễ.

### Quản lý bộ nhớ:
- In-Memory Storage: Tất cả dữ liệu được tải vào RAM – tối ưu về tốc độ nhưng không phù hợp với dữ liệu lớn.
- Memmap Storage: Dữ liệu được ánh xạ trực tiếp từ file trên ổ đĩa vào không gian ảo; phương pháp này cho phép làm việc với tập dữ liệu lớn hơn so với RAM, với hiệu suất gần như lưu trữ trong bộ nhớ.

**Cấu hình mẫu cho Memmap:**
```python
client.create_collection(
   collection_name="{collection_name}",
   vectors_config=models.VectorParams(
      …
      on_disk=True
   )
)
```
Cho payload:
```python
client.create_collection(
    collection_name="{collection_name}",
    on_disk_payload=True
)

```


## Monitoring the Database (Giám sát cơ sở dữ liệu)

Continuous monitoring is essential for maintaining system health and identifying potential issues before they escalate. Tools like Prometheus and Grafana are widely used to achieve this.

- **Prometheus:** An open-source monitoring and alerting toolkit, Prometheus collects and stores metrics in a time-series database. It scrapes metrics from predefined endpoints and supports powerful querying and visualization capabilities.
- **Grafana:** Often paired with Prometheus, Grafana provides an intuitive interface for visualizing metrics and creating interactive dashboards.

Qdrant exposes metrics in the **Prometheus/OpenMetrics** format through the */metrics* endpoint. Prometheus can scrape this endpoint to monitor various aspects of the Qdrant system.

For a local Qdrant instance, the metrics endpoint is typically available at:
```python
http://localhost:6333/metrics
```

Một số metrics quan trọng để theo dõi database

|Metric Name	|	Meaning|
|---------------|--------|
`collections_total`	|	Tổng số collections
`collections_vector_total`		|Tổng số vector trên toàn bộ collections
`rest_responses_avg_duration_seconds`		| Thời gian phản hồi trung bình của REST API
`grpc_responses_avg_duration_seconds`	|	A Thời gian phản hồi trung bình của gRPC API
`rest_responses_fail_total`		| Số lượng phản hồi REST thất bại