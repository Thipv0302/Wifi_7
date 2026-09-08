# Phương pháp 1 — GNN surrogate + gieo quần thể bằng policy

Bài báo: `main.tex` · Hình: `Image/` · Dữ liệu: `data/`

Bài này trả lời một câu hỏi: **chi phí thật của việc giải P1 nằm ở đâu?**
Câu trả lời hoá ra không phải "giá một lần đánh giá hàm thích nghi".

---

## 1. Lập luận của bài, theo đúng thứ tự

| Bước | Nội dung | Bằng chứng |
|---|---|---|
| Vấn đề | Một lần đánh giá tốn 16 ms; một lần chạy GA tốn 153 s; Fig. 6 của \[Yi2025\] cần 10 lần chạy | đo trực tiếp |
| Giải pháp hiển nhiên | Làm cho hàm đánh giá rẻ đi → GNN surrogate | ρ = 0.9996, giảm 19.8× số lần gọi mô hình giải tích |
| **Kết quả phủ định** | Chất lượng nghiệm **không đổi**: 8/70 so với 4/70 | 70 lần chạy mỗi phương pháp |
| Chẩn đoán | Vùng khả thi mỏng tới mức 0/600 mẫu rơi vào trong | Table II của bài |
| Giải pháp thật | Gieo quần thể bằng policy đã học | **69/70**, và N_pop = 10 là đủ |

Kết quả phủ định ở dòng 3 **được giữ lại** chứ không bị bỏ đi. Đó là điều duy
nhất tách được hai đóng góp: nếu chỉ báo cáo cấu hình cuối (surrogate + gieo)
thì không ai biết phần nào đem lại kết quả.

---

## 2. Vì sao là GNN, không phải MLP

Không phải phép loại suy — là trùng khớp cấu trúc. Lấy log phần tích trong
Eq. (4) của \[Yi2025\]:

```
log q_j = log Π_{k∈Z_j} r_k^{n_k} = Σ_{k∈Z_j} n_k · log r_k
```

Vế phải là **sum-aggregation bất biến hoán vị** trên các node láng giềng, với
`n_k` là trọng số cạnh và `log r_k` là message. Lớp ngoài — trung bình có trọng
số `π_j` theo vùng AIFS — là phép aggregation thứ hai. Vòng lặp điểm cố định
Eq.(4)–(5) **chính là** T bước message passing với trọng số chia sẻ.

Ba hệ quả, mỗi hệ quả chốt một lựa chọn thiết kế:

| | MLP | GNN |
|---|---|---|
| Bất biến hoán vị các AC | phải học từ dữ liệu | có sẵn trong kiến trúc |
| Số AC thay đổi (3…6) | không dùng lại được | dùng chung một mạng |
| Phép gán link MLO | chỉ là một giá trị đầu vào | **là ma trận kề** |

Dòng cuối là lý do mạnh nhất: biến `link` quyết định AC nào tranh chấp với AC
nào, tức nó thay đổi *cấu trúc* bài toán chứ không phải giá trị đặc trưng.

---

## 3. Biểu diễn đồ thị

| Thành phần | Nội dung |
|---|---|
| Node | mỗi AC là một node (không có node link riêng) |
| Cạnh | AC_i — AC_j nếu **cùng link**; ma trận kề chính là phép gán MLO |
| Đặc trưng node (17) | kịch bản `n_i, L_i, D_max,i, log₁₀ε_i` · biến quyết định `CW_min, log₂(CW_max/CW_min), AIFSN, TXOP, R` · các đại lượng closed-form Eq.(7)–(9) · tải trên link |
| Đặc trưng cạnh (5) | `n_j` (đúng là trọng số trong Eq. 4) · độ lệch AIFS `h_j − h_i` · hai chỉ báo trùng vùng |

Mọi đại lượng đã có công thức đóng và rẻ đều được **cho vào làm đặc trưng** thay
vì bắt mạng học lại — chính xác tuyệt đối và miễn phí.

Đồ thị chỉ 3–8 node nên dùng **ma trận kề dày** trong torch thuần: ở quy mô này
dense einsum nhanh hơn scatter thưa, và tránh hẳn việc cài `torch_geometric`
trên Windows.

---

## 4. Sinh dữ liệu — vấn đề vùng khả thi mỏng như dao cạo

Nhãn do chính mô hình giải tích sinh ra nên dữ liệu miễn phí. Nhưng lấy mẫu
ngẫu nhiên thất bại hoàn toàn:

| Cách lấy mẫu | Tỉ lệ khả thi |
|---|---|
| Ngẫu nhiên đều trên miền Sec. IV.B | **0 / 300** |
| "Có cấu trúc" (CW lớn, span hẹp, R = 7) | **0 / 300** |
| Nhiễu quanh *chính nghiệm tối ưu* (σ = 0.15) | **1.5 %** |

Riêng từng AC trong vùng có cấu trúc: AC1 4 %, AC2 5 %, AC3 25 %, AC4 90 %,
AC5 98 %. **Bài toán nằm ở phép hội**, không ở ràng buộc nào riêng lẻ.

Cách xử lý: **ghi lại quỹ đạo GA**. Mọi cấu hình GA đánh giá đều đã chạy qua
mô hình giải tích rồi — ghi lại kết quả sẵn có nên **không tốn thêm lần đánh
giá nào**, mà phân bố của nó khớp đúng phân bố surrogate sẽ gặp khi làm việc.
Tập dữ liệu trộn 45 % quỹ đạo GA + 55 % lấy mẫu rộng.

Kết quả trên 200 000 mẫu: **8.19 % khả thi (16 373), 19.10 % sát biên ràng buộc
(38 201)**. Chi phí: **159 s** trên 26 nhân.

---

## 5. Kiến trúc và độ chính xác

```
h_i⁰ = Enc(x_i)
lặp T = 6 lần (CHIA SẺ TRỌNG SỐ):
    m_ij = Msg([h_i, h_j, e_ij])
    a_i  = Σ_j A_ij · m_ij   ‖  max_j A_ij · m_ij
    h_i  = h_i + Upd([h_i, a_i])          ← residual: một bước lặp điểm cố định
(log₁₀ c_i, θ_i) = Out(h_i)
```

156 386 tham số (hidden 128, 6 lớp, edge_hidden 32). 60 epoch, batch 512,
AdamW + OneCycle lr 2e-3, Huber chuẩn hoá theo σ từng nhãn, bf16: **387 s** trên
RTX 4060.

| Lựa chọn | Lý do |
|---|---|
| **Sum**-aggregation | Eq.(4) chứa `Σ_k n_k log r_k` — một TỔNG. Mean sẽ chuẩn hoá mất thông tin "có bao nhiêu trạm đang tranh chấp" |
| **Chia sẻ trọng số** qua T lớp | `solve_link` lặp cùng một toán tử mỗi vòng. T lớp mô phỏng T *vòng lặp*, không phải T hop (đồ thị đầy đủ trong link, 1 hop là hết) |
| Học `log₁₀ c`, không học `P_loss` | Eq.(18) tuyến tính theo `log₁₀ c`. `P_loss = c^R` tính CHÍNH XÁC sau đó — mạng không bao giờ học phép luỹ thừa ^R vốn khuếch đại sai số |
| Học `θ`, không học `Pr(D ≥ D_max)` | Ràng buộc nằm ở 1e-7…1e-4; hồi quy xác suất thô ở thang đó là vô vọng |

Độ chính xác trên 20 000 mẫu kiểm tra tách riêng:

| Chỉ số | Toàn tập | Riêng tập **sát biên** |
|---|---|---|
| RMSE `log₁₀ c` | 0.0108 | — |
| RMSE `θ` | 0.3929 | **0.4519** |
| Spearman ρ của fitness | 0.9996 | **0.9824** |
| Độ chính xác quyết định khả thi | 99.19 % | **95.74 %** |

**Cột phải mới là cột đáng đọc.** GA luôn đẩy nghiệm tới sát ràng buộc, nên sai
số ở vùng sát biên mới quyết định, không phải RMSE tổng.

RMSE `θ` = 0.393 ⇔ sai khoảng 2.47× về xác suất — **đúng bằng mức sai lệch của
chính mô hình giải tích so với mô phỏng mức slot** (1.2–4×). Surrogate đã chạm
sàn tự nhiên: nó chính xác ngang cái mà nó xấp xỉ.

---

## 6. Kết quả — 70 lần chạy mỗi phương pháp

Lưới `N_pop ∈ {6,10,16,25,50,100,200}` × 10 seed. Cùng `N_gen = 300`,
`N_stag = 50`, cùng tập seed. *Thành công* = fitness ≥ 49.207 (trong 1 % của
giá trị tốt nhất quan sát được trên toàn bộ phép đo, ngưỡng dùng chung).

| Phương pháp | Thành công | Trung vị fitness | Thời gian |
|---|---|---|---|
| GA gốc | **4 / 70** | 23.7 … 38.2 | 4.6 – 152.6 s |
| GA + GNN surrogate | **8 / 70** | 23.6 … 41.7 | 14.4 – 35.2 s |
| **GA + GNN + gieo bằng policy** | **69 / 70** | **49.70 ở mọi ngân sách** | 4.7 – 13.6 s |

Ở `N_pop = 200`: 152.6 s / 29 309 lần gọi giải tích → 13.6 s / 608 lần.

### Con số đáng chú ý nhất không phải tốc độ

`N_pop = 10`: gieo bằng policy đạt **10/10** với quần thể 10 cá thể trong 4.7 s.
GA gốc với quần thể gấp **20 lần** (`N_pop = 200`, 152.6 s) đạt **1/10**.
Policy không chỉ làm GA nhanh hơn — nó **xoá bỏ nhu cầu về quần thể lớn**.

Lý do nằm ở quần thể ban đầu: seed chiếm **1/3** quần thể (`n_seed = N_pop//3`),
và 32.1 % toàn bộ quần thể ban đầu là khả thi — tức gần như mọi cá thể được gieo
đều khả thi. Hai cách gieo còn lại cho **0 %** ở mọi `N_pop`.

---

## 7. Hai điều phải nói thật trong bài

**Surrogate một mình gần như không đem lại gì về chất lượng.** 8/70 so với 4/70
nằm trong nhiễu. Con số "tăng tốc 10.03×" trong ghi chép cũ là ảo giác một-seed;
đo trên 10 seed cho **4.7×** ở `N_pop = 200`, và **chậm hơn** GA gốc ở mọi
`N_pop ≤ 25` (chậm 3.1× ở `N_pop = 6`).

Nguyên nhân chậm: `ga_surrogate.polish_screened` chấm **toàn bộ mức** của mỗi
gen qua surrogate, mà `datagen.graph.build_graph` dựng từng đồ thị một trong
Python thuần. Chi phí đó gần như **không phụ thuộc `N_pop`** nên khi quần thể
nhỏ nó nuốt hết phần tiết kiệm. Đây là hạn chế cài đặt, không phải hạn chế của ý
tưởng — `training/batch_graph.py` đã có bản vector hoá nhưng `polish_screened`
chưa dùng. Bài báo nói đúng như vậy.

**Toàn bộ 70 lần chạy dùng một kịch bản duy nhất**, mà kịch bản đó chiếm 50 %
kho huấn luyện của policy. Ghi rõ trong mục Limitations.

---

## 8. Vẽ lại hình

```bash
py -3 make_figures.py
```

Đọc `data/benchmark.json`, ghi `Image/fig03…fig05` ở cả `.pdf` (LaTeX dùng) và
`.png` 600 dpi (xem nhanh). Script in ra font matplotlib thực sự phân giải được
— phải là `times.ttf`, nếu ra thứ khác thì hình lại lệch font so với bài.
