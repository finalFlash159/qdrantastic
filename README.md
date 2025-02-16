# 🛩️ Vector Database

## 1. Sparse vectors 🗿

A sparse vector is a special representation of high-dimensional vectors where most elements are zero, and only a few dimensions have non-zero values. 
(Ví dụ, nếu từ điển có 50,000 từ, một văn bản chỉ chứa 100 từ sẽ có vector với 49,900 giá trị 0 và 100 giá trị khác 0.)
  * Chính xác trong khớp từ: Giúp nắm bắt chính xác các từ khóa xuất hiện trong văn bản, hữu ích cho các tác vụ truy xuất dựa trên từ khóa &#9989;
  * Thiếu khả năng khái quát hoá: Không nắm bắt được ý nghĩa ngữ nghĩa tổng quát của văn bản mà chỉ dựa vào sự xuất hiện của từ khóa. &#10060;

## 1.1 Sparse Embedding ☕️

***
### **📘 BM25 (Best Matching 25)**

BM25 là một phương pháp xếp hạng tài liệu dựa trên nguyên tắc của TF-IDF nhưng được cải tiến bằng cách chuẩn hóa độ dài của tài liệu. Điều này giúp giảm thiểu sự thiên lệch đối với các tài liệu có độ dài khác nhau.

#### Công thức 

$$
\text{score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{TF(q, D) \cdot (k_1 + 1)}{TF(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
$$

#### Thành phần

- **TF (Term Frequency):**  
  Tần suất xuất hiện của từ $q$ trong tài liệu $D$. Tần suất cao thường cho thấy từ đó có ý nghĩa quan trọng trong nội dung tài liệu.

- **IDF (Inverse Document Frequency):**  
  Độ hiếm của từ $q$ trong toàn bộ kho tài liệu, giúp giảm trọng số cho các từ quá phổ biến. Giúp giảm trọng số của các từ xuất hiện thường xuyên trên toàn bộ tập tài liệu, nhấn mạnh các từ có ý nghĩa phân biệt cao.

- **$k_1$ và $b$:**  
  Các tham số điều chỉnh (thường mặc định $k_1 \approx 1.2$ và $b \approx 0.75$):
  - $k_1$ điều chỉnh ảnh hưởng của tần suất từ.
  - $b$ điều chỉnh ảnh hưởng của độ dài tài liệu.

- **$|D|$ và $\text{avgdl}$:**  
  - $|D|$: Độ dài của tài liệu $D$ (số từ trong tài liệu).
  - $\text{avgdl}$: Độ dài trung bình của tất cả các tài liệu.

***
### **📗 SPLADE (Sparse Lexical and Expansion Model)**

#### Cách hoạt động của SPLADE

SPLADE (Sparse Lexical and Expansion model) là một mô hình kết hợp ưu điểm của các biểu diễn sparse (rời rạc) và khả năng mở rộng từ vựng thông qua học sâu. Mục tiêu chính của SPLADE là giải quyết vấn đề vocabulary mismatch trong truy vấn thông tin bằng cách không chỉ dựa vào cá

##### a. Khởi đầu với Transformer và BERT
- **BERT với Masked Language Modeling (MLM):**  
  SPLADE sử dụng mô hình BERT đã được tiền huấn luyện với nhiệm vụ MLM.
- **Tokenization & Embedding:**  
  Văn bản được chia thành các token (hoặc sub-word tokens) theo kiểu BERT, mỗi token được ánh xạ thành vector thông qua embedding matrix.
  ![tokenizer](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fd773f2c0a10dc37381b4688626e4fdb9da5fc5a4-2310x1457.png&w=3840&q=75)
- **Encoder Blocks:**  
  Các vector token được xử lý qua nhiều lớp encoder với cơ chế attention, thu được các vector "information-rich" chứa đầy đủ ngữ cảnh.
  ![encoder-block](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fe8fe02e5887ff8dda56dff29c18940b0125ebc6b-2318x1466.png&w=3840&q=75)

  ![output](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2F00a694f2f4e9f7ad6006f538df551c5ec3c23347-2458x1363.png&w=3840&q=75)

##### b. Vai trò của MLM Head
- **MLM Head:**  
  Một số token trong văn bản được thay bằng `[MASK]` và mô hình dự đoán lại token gốc dựa trên ngữ cảnh.
- **Output Logits:**  
  Kết quả đầu ra là tập hợp logits cho mỗi token, với số chiều bằng kích thước từ vựng (ví dụ: 30522), biểu diễn xác suất của các từ có thể xuất hiện tại vị trí đó.
  ![mask](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fd64d431fb1b50ae9aa94b5cd85e1cdffe5eb7ca1-2318x1516.png&w=3840&q=75)

##### c. Tạo Sparse Embeddings
- **Chuyển đổi logits thành sparse vector:**  
  SPLADE tính toán trọng số cho mỗi từ \(j\) trong từ vựng theo công thức:
  
  $$w_j = \sum_{i \in t} \log\Big(1 + \text{ReLU}(w_{ij})\Big)$$
  
  Trong đó:
  - $t$: Tập các token của văn bản.
  - $w_{ij}$: Logit được dự đoán cho token $i$ đối với từ $j$.
  
- **Mở rộng từ khóa (Term Expansion):**  
  SPLADE không chỉ biểu diễn các từ có trong văn bản mà còn gán trọng số cho các từ liên quan (ví dụ: từ "rainforest" có thể mở rộng thành "jungle", "land", "forest"), giúp tăng khả năng khớp giữa truy vấn và tài liệu.

***
## 2. Multi-vector  🗿

Matrices of numbers with fixed length but variable height. Usually obtained from late interaction models like ColBERT.

### 🧝🏼‍♀️ Colbert (Contextualized Late Interaction over BERT) 

Không giống như BERT, vốn gộp các vector của từng token thành một biểu diễn duy nhất, ColBERT giữ nguyên các biểu diễn theo từng token, giúp tính toán mức độ tương đồng một cách chi tiết hơn. Điểm nổi bật của ColBERT chính là cơ chế tương tác muộn (late interaction). Cơ chế này cho phép xếp hạng và truy xuất thông tin một cách hiệu quả và chính xác bằng cách xử lý truy vấn và tài liệu một cách độc lập cho đến những giai đoạn cuối của quá trình truy xuất.

**ColBERT Architecture** 🏛️
![ colbert-architecture](https://assets.zilliz.com/The_general_architecture_of_Col_BERT_30db3739a3.png)

## 2.1 Cách hoạt động của ColBERT

###  Query Encoder

- **Tokenization**: Truy vấn $Q$được chia thành các token $q_1, q_2, \dots, q_l$.
- **Đánh dấu**: Thêm token đặc biệt `[Q]` ngay sau `[CLS]` để đánh dấu đây là truy vấn.
- **Padding/Truncation**: Nếu số token ít hơn $N_q$, đệm thêm token `[MASK]`; nếu nhiều hơn, cắt ngắn về $N_q$ token đầu.
- **Xử lý**: Chuỗi token được đưa qua BERT và sau đó qua một CNN để tạo ra tập hợp vector nhúng, rồi được chuẩn hóa:
  
  $$E_q := \text{Normalize}\Big(\text{CNN}\big(\text{BERT}("[Q], q_0, q_1, \dots, q_l, [MASK], \dots, [MASK]")\big)\Big)$$

---

### Document Encoder

- **Tokenization**: Tài liệu $D$ được chia thành các token \( d_0, d_1, \dots, d_n \).
- **Đánh dấu**: Thêm token đặc biệt `[D]` ngay sau `[CLS]` để đánh dấu đầu tài liệu.
- **Xử lý**: Chuỗi token được đưa qua BERT, qua CNN, sau đó chuẩn hóa và lọc bỏ các vector liên quan đến dấu câu:
  
$$E_d := \text{Filter}\Big(\text{Normalize}\big(\text{CNN}\big(\text{BERT}("[D], d_0, d_1, \dots, d_n")\big)\big)\Big)$$

---

### Cơ chế tương tác trễ (Late Interaction Mechanism)
Ttương tác trễ trong ColBERT nghĩa là truy vấn và tài liệu được mã hóa riêng biệt, chỉ so sánh với nhau ở bước cuối cùng. Điều này giúp:

- Tăng tốc độ truy xuất tài liệu: Có thể tính trước vector tài liệu và lưu trữ trước khi có truy vấn.
- Cải thiện độ chính xác: So sánh chi tiết từng token thay vì một vector duy nhất.

- **MaxSim**:  
  Với mỗi token $t_q$ trong tập $E_q$:
  1. Tìm token $t_d$ trong $E_d$ có độ tương đồng cao nhất (sử dụng cosine similarity hoặc squared L2 distance).
  2. Tổng hợp các điểm số tương đồng để tính điểm liên quan tổng thể $S_{q,d}$.

## 2.2 ColBERTv2 

- **Vấn đề của ColBERT**: Lưu trữ vector cho từng token tiêu tốn nhiều bộ nhớ.
- **Giải pháp trong ColBERTv2**:
  - **Product Quantization (PQ)**: Nén vector token mà không mất nhiều thông tin.
  - **Centroid-Based Encoding**: Nhóm các vector theo trọng tâm để giảm kích thước lưu trữ.
- **Quy trình truy vấn**:
  - So sánh chỉ với các vector thuộc các cụm (centroid) gần nhất (sử dụng số lượng cụm định trước, hay nprobe).
  - Sau đó, tải lại toàn bộ vector của tài liệu phù hợp để tính toán chi tiết hơn.

> **Phân tích**:  
> - ColBERTv2 giảm đáng kể dung lượng lưu trữ trong khi vẫn giữ được độ chính xác cao.
> - So khớp với các vector thuộc cụm gần nhất giúp tăng tốc độ truy xuất.

---