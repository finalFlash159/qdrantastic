# Vector Database

## 1. Sparse vectors

A sparse vector is a special representation of high-dimensional vectors where most elements are zero, and only a few dimensions have non-zero values (Ví dụ, nếu từ điển có 50,000 từ, một văn bản chỉ chứa 100 từ sẽ có vector với 49,900 giá trị 0 và 100 giá trị khác 0.)
  * Chính xác trong khớp từ: Giúp nắm bắt chính xác các từ khóa xuất hiện trong văn bản, hữu ích cho các tác vụ truy xuất dựa trên từ khóa &#9989;
  * Thiếu khả năng khái quát hoá: Không nắm bắt được ý nghĩa ngữ nghĩa tổng quát của văn bản mà chỉ dựa vào sự xuất hiện của từ khóa. &#10060;

### Sparse Embedding

**📘 BM25 (Best Matching 25)**


**📗 SPLADE (Sparse Lexical and Expansion Model)**