# Python Commands

## Client Libraries

**Python + (Client Docs):**

```bash
pip install qdrant-client[fastembed]
```


## Collectións

**Get collection details**

```bash
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.get_collection("{collection_name}")
```

**Create a collection (with given parameter)**

```bash
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="{collection_name}",
    vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
)
```

**Delete a collection**

```bash
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.delete_collection(collection_name="{collection_name}")
```

**Update collection parameters**

```bash
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.update_collection(
    collection_name="{collection_name}",
    optimizer_config=models.OptimizersConfigDiff(indexing_threshold=10000),
)
```

**List all collections**

```bash
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.get_collections()
```

**Check collection existence**

```bash
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

client.collection_exists(collection_name="{collection_name}")
```