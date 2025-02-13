# 🛩️ Vector Database

## 1. Sparse vectors 🗿

A sparse vector is a special representation of high-dimensional vectors where most elements are zero, and only a few dimensions have non-zero values. 
(Ví dụ, nếu từ điển có 50,000 từ, một văn bản chỉ chứa 100 từ sẽ có vector với 49,900 giá trị 0 và 100 giá trị khác 0.)
  * Chính xác trong khớp từ: Giúp nắm bắt chính xác các từ khóa xuất hiện trong văn bản, hữu ích cho các tác vụ truy xuất dựa trên từ khóa &#9989;
  * Thiếu khả năng khái quát hoá: Không nắm bắt được ý nghĩa ngữ nghĩa tổng quát của văn bản mà chỉ dựa vào sự xuất hiện của từ khóa. &#10060;

## 1.1 Sparse Embedding ☕️

**📘 BM25 (Best Matching 25)**

BM25 là một phương pháp xếp hạng tài liệu dựa trên nguyên tắc của TF-IDF nhưng được cải tiến bằng cách chuẩn hóa độ dài của tài liệu. Điều này giúp giảm thiểu sự thiên lệch đối với các tài liệu có độ dài khác nhau.

### Công thức

$$
\text{score}(D, Q) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{TF(q, D) \cdot (k_1 + 1)}{TF(q, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
$$

### Thành phần

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




**📗 SPLADE (Sparse Lexical and Expansion Model)**