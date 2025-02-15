# Python Commands

## Client Libraries

**Python + (Client Docs):**

```bash
pip install qdrant-client[fastembed]
```


## 1. Collections

**1.1 Get collection details**

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.get_collection("{collection_name}")
```

**1.2 Create a collection (with given parameter)**

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="{collection_name}",
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
)
```

**1.3 Delete a collection**

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.delete_collection(collection_name="{collection_name}")
```

**1.4 Update collection parameters**

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.update_collection(
    collection_name="{collection_name}",
    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10000),
)
```

**1.5 List all collections**

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.get_collections()
```

**1.6 Check collection existence**

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.collection_exists(collection_name="{collection_name}")
```

---
## Collection with multiple vector

```python
client.create_collection(
    collection_name="{collection_name}",
    vectors_config={
        "image": models.VectorParams(size=4, distance=models.Distance.DOT),
        "text": models.VectorParams(size=8, distance=models.Distance.COSINE),
    },
)
```

---
### Vector datatypes

Vectors with uint8 datatype are stored in a more compact format, which can save memory and improve search speed at the cost of some precision. If you choose to use the uint8 datatype, elements of the vector will be stored as unsigned 8-bit integers, which can take values from 0 to 255.

```python
client.create_collection(
    collection_name="{collection_name}",
    vectors_config=models.VectorParams(
        size=1024,
        distance=models.Distance.COSINE,
        datatype=models.Datatype.UINT8,
    ),
)
```

---
### Collection with sparse vectors

Collections can contain sparse vectors as additional <span style="color:red">named vectors</span> along side regular dense vectors in a single point.

Unlike dense vectors, sparse vectors must be named. And additionally, sparse vectors and dense vectors must have different names within a collection.

```python
client.create_collection(
    collection_name="{collection_name}",
    vectors_config={},
    sparse_vectors_config={
        "text": models.SparseVectorParams(),
    },
)
```

The distance function for sparse vectors is always `Dot` and does not need to be specified.