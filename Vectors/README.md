## 🧙🏼‍♂️ Parse Vectors 

Sparse vectors are a special type of vectors. Mathematically, they are the same as dense vectors, but they contain many zeros so they are stored in a special format.

Sparse vectors in Qdrant don’t have a fixed length, as it is dynamically allocated during vector insertion. The amount of non-zero values in sparse vectors is currently limited to u32 datatype range (4294967295).

**In order to define a sparse vector, you need to provide a list of non-zero elements and their indexes.**

```python
// A sparse vector with 4 non-zero elements
{
    "indexes": [1, 3, 5, 7],
    "values": [0.1, 0.2, 0.3, 0.4]
}
```

Sparse vectors in Qdrant are kept in special storage and indexed in a separate index, so their configuration is different from dense vectors.

**To create a collection with sparse vectors:**
```python
client.create_collection(
    collection_name="{collection_name}",
    vectors_config={},
    sparse_vectors_config={
        "text": models.SparseVectorParams(),
    },
)
```

**Insert a point with a sparse vector into the created collection:**

```python
client.upsert(
    collection_name="{collection_name}",
    points=[
        models.PointStruct(
            id=1,
            payload={},  # Add any additional payload if necessary
            vector={
                "text": models.SparseVector(
                    indices=[1, 3, 5, 7],
                    values=[0.1, 0.2, 0.3, 0.4]
                )
            },
        )
    ],
)
```

**Search with sparse vector**
```python
result = client.query_points(
    collection_name="{collection_name}",
    query=models.SparseVector(indices=[1, 3, 5, 7], values=[0.1, 0.2, 0.3, 0.4]),
    using="text",
).points
```

## 🧝🏽‍♂️ Mulivectors
Qdrant supports the storing of a variable amount of same-shaped dense vectors in a single point. This means that instead of a single dense vector, you can upload a matrix of dense vectors.

<span style="color:orange">The length of the matrix is fixed, but the number of vectors in the matrix can be different for each point.</span>

**Multivectors look like this:**

```python
// A multivector of size 4
"vector": [
    [-0.013,  0.020, -0.007, -0.111],
    [-0.030, -0.055,  0.001,  0.072],
    [-0.041,  0.014, -0.032, -0.062],
    ....
]
```

In order to use multivectors, we need to specify a function that will be used to compare between matrices of vectors

Currently, Qdrant supports `max_sim` function, which is defined as a sum of maximum similarities between each pair of vectors in the matrices.

$$score = \sum_{i=1}^{N} \max_{j=1}^{M} \text{Sim}(\text{vectorA}_i, \text{vectorB}_j)$$

Where $N$ is the number of vectors in the first matrix,  $M$ is the number of vectors in the second matrix, and $Sim$ is a similarity function, for example, cosine similarity.

**To use multivectors, create a collection with the following configuration:**
```python
client.create_collection(
    collection_name="{collection_name}",
    vectors_config=models.VectorParams(
        size=128,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    ),
)
```

**To insert a point with multivector:**
```python
client.upsert(
    collection_name="{collection_name}",
    points=[
        models.PointStruct(
            id=1,
            vector=[
                [-0.013,  0.020, -0.007, -0.111],
                [-0.030, -0.055,  0.001,  0.072],
                [-0.041,  0.014, -0.032, -0.062]
            ],
        )
    ],
)
```

**To search with multivector (available in `query` API):**
```python
client.query_points(
    collection_name="{collection_name}",
    query=[
        [-0.013,  0.020, -0.007, -0.111],
        [-0.030, -0.055,  0.001,  0.072],
        [-0.041,  0.014, -0.032, -0.062]
    ],
)
```

## 🧛🏽‍♂️ Named Vecotors

In Qdrant, you can store multiple vectors of <span style="color:orange">different sizes and types</span> in the <span style="color:orange">same data point</span>. This is useful when you need to define your data with multiple embeddings to represent different features or modalities (e.g., image, text or video).

To store different vectors for each point, you need to create separate named vector spaces in the collection. You can define these vector spaces during collection creation and manage them independently.

>Each vector should have a unique name. Vectors can represent different modalities and you can use different embedding models to generate them.

```python
client.create_collection(
    collection_name="{collection_name}",
    vectors_config={
        "image": models.VectorParams(size=4, distance=models.Distance.DOT),
        "text": models.VectorParams(size=5, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={"text-sparse": models.SparseVectorParams()},
)
```
**To insert a point with named vectors:**
```pythonclient.upsert(
    collection_name="{collection_name}",
    points=[
        models.PointStruct(
            id=1,
            vector={
                "image": [0.9, 0.1, 0.1, 0.2],
                "text": [0.4, 0.7, 0.1, 0.8, 0.1],
                "text-sparse": {
                    "indices": [1, 3, 5, 7],
                    "values": [0.1, 0.2, 0.3, 0.4],
                },
            },
        ),
    ],
)
```

**You can also upload To search with named vectors (available in query API):**
```python
client.query_points(
    collection_name="{collection_name}",
    query=[0.2, 0.1, 0.9, 0.7],
    using="image",
    limit=3,
)
```