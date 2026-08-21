# Tái hiện paper: Intelligent Multi-link EDCA Optimization for Delay-Bounded QoS in Wi-Fi 7

Cài đặt lại toàn bộ mô hình giải tích và thuật toán tối ưu của Yi, Cheng, Wang, Pan,
Ouyang, Zhang — *"Intelligent Multi-link EDCA Optimization for Delay-Bounded QoS in
Wi-Fi 7"* (arXiv:2509.25855v1), kèm tái hiện các hình kết quả (Fig. 2–6).

---

## 1. Cấu trúc module

```
wifi7_edca/
├── config.py                  # Toàn bộ tham số (TABLE I, kịch bản Sec. IV.A và Sec. V)
│
├── common/                    # ===== HÀM TÍNH TOÁN / UTILITY DÙNG CHUNG =====
│   ├── zones.py               #   Mô hình vùng AIFS: Eq.(1)(2) · π_j Eq.(3)
│   ├── collision.py           #   c_k Eq.(4) · p_k Eq.(5) · P_loss Eq.(6) · điểm cố định
│   ├── timing.py              #   T_DATA Eq.(7) · N_k Eq.(8) · T^C, T^S Eq.(9)
│   ├── delay.py               #   Hàm sinh Eq.(10)-(15) · nghịch đảo Fourier Eq.(16) · θ Eq.(17)
│   └── utility.py             #   Đánh giá QoS một cấu hình · mục tiêu & ràng buộc P1 Eq.(18)
│
├── datagen/                   # ===== SINH DỮ LIỆU =====
│   ├── scenario.py            #   Tập AC (n_i, L_i, D_max,i, ε_i) cho từng thí nghiệm
│   └── genome.py              #   Không gian tham số EDCA + mã hoá/giải mã nhiễm sắc thể
│
├── training/                  # ===== TỐI ƯU / "HUẤN LUYỆN" =====
│   ├── ga.py                  #   Algorithm 1 — thuật toán di truyền
│   ├── baselines.py           #   EDCA mặc định 802.11e · tìm kiếm ngẫu nhiên
│   ├── simulate.py            #   Mô phỏng EDCA mức slot — KIỂM CHỨNG mô hình giải tích
│   └── experiments.py         #   exp_fig2 … exp_fig6 → dict dữ liệu thô (results/*.json)
│
├── plotting/                  # ===== VẼ KẾT QUẢ =====
│   ├── style.py               #   Bảng màu đã kiểm chứng CVD, gán màu cố định theo thực thể
│   └── figures.py             #   plot_fig2 … plot_fig6 → figures/*.png
│
└── main.py                    # pipeline: gen data → utility → training → vẽ
```

Luồng dữ liệu một chiều: `datagen → common (utility) → training → plotting`.
Module `plotting` **không tính lại** bất kỳ đại lượng nào; module `training` **không vẽ**.

## 2. Cách chạy

```bash
python main.py               # chạy tất cả (Fig. 2-6)
python main.py 3             # chỉ chạy một hình
python main.py --quick       # GA nhỏ, dùng để kiểm tra pipeline nhanh

python -m common.zones       # in cấu trúc vùng AIFS, đối chiếu Fig. 2
python -m common.collision   # giải điểm cố định EDCA cho cấu hình mặc định
python -m common.timing      # in bảng thời lượng T_DATA, Δ_k, N_k, T^C, T^S
python -m common.delay       # kiểm tra D(1) = 1 và tính Pr(D ≥ D_max)
python -m common.utility     # đánh giá QoS đầy đủ, so sánh 1 link và 2 link
python -m training.simulate  # KIỂM CHỨNG mô hình bằng mô phỏng mức slot
```

Thời gian chạy tham khảo (2 nhân CPU): Fig. 2 và 3 khoảng 10 giây; Fig. 4 và 5 khoảng
16 phút (2 lần chạy GA, ~8 600 lần đánh giá mỗi lần); Fig. 6 khoảng 15 phút (10 lần
chạy GA có warm-start).

Yêu cầu: `numpy`, `scipy`, `matplotlib`.

## 3. Các phương trình đã cài đặt

| Paper | Nội dung | Vị trí trong code |
|---|---|---|
| Eq. (1) | φ_j — tập AC đủ điều kiện tranh chấp ở slot j | `common/zones.active_mask_at_slot` |
| Eq. (2) | h_k = (AIFS_k − AIFS_1)/σ | `common/zones.build_zone_model` |
| Eq. (3) | π_k — xác suất dừng của vùng k | `common/zones.zone_stationary` |
| Eq. (4) | c_k — xác suất va chạm | `common/collision.collision_prob` |
| Eq. (5) | p_k — xác suất phát (xấp xỉ giá trị trung bình) | `common/collision.transmit_prob` |
| Eq. (6) | P_loss,k = c_k^{R_k} | `common/collision.solve_link` |
| Eq. (7) | T_DATA,k | `common/timing.ac_timing` |
| Eq. (8) | N_k = max{⌊TXOP_k/Δ_k⌋, 1} | `common/timing.ac_timing` |
| Eq. (9) | T_k^C, T_k^S | `common/timing.ac_timing` |
| Eq. (10) | D(z) — hàm sinh tổng trễ | `common/delay.eval_dhat` |
| Eq. (11)(12) | ê(z), ξ_ℓ, ρ_k(ℓ) | `common/delay.build_delay_model`, `eval_e` |
| Eq. (13) | Â(z) — backoff + va chạm | `common/delay.eval_a` |
| Eq. (14) | Ŷ(z), γ_{k,ℓ}, ν_k | `common/delay.eval_y`, `build_delay_model` |
| Eq. (15) | T̂(z), Ĥ(z) = Ĉ(z), Ĝ_ℓ(z) | `common/delay.eval_dhat` |
| Eq. (16) | Pr(D_k ≥ x) — nghịch đảo chuỗi Fourier | `common/delay.delay_ccdf` |
| Eq. (17) | θ_k = −log Pr(D_k ≥ D_max,k) | `common/delay.reliability_index` |
| Eq. (18) | P1 — mục tiêu + ràng buộc | `common/utility.QoSResult.objective`, `fitness` |
| Algorithm 1 | GA tối ưu tham số EDCA | `training/ga.run_ga` |

**Kiểm chứng nội tại:** `python -m common.delay` in ra `D(1⁻) = 1.00000` cho mọi AC —
hàm sinh Eq.(10)–(15) là một hàm sinh xác suất hợp lệ (tổng khối lượng bằng 1).

**Kiểm chứng cấu trúc vùng:** `python -m common.zones` với AIFS = {30, 50, 50, 90} µs
cho `Z_j = [1, 3, 4]` — đúng bằng hàng `Z_j` trong Fig. 2 của paper.

## 4. Những chỗ paper không nói rõ và cách xử lý

| Vấn đề | Xử lý trong code |
|---|---|
| **Cơ số logarit trong Eq. (17), (18)** | Paper chỉ viết "log". Thang giá trị của Fig. 4 (fitness đạt ~35 với 5 AC) và các mức P_loss trong Fig. 5b (1e-10 … 1e-1) chỉ khớp khi dùng **log₁₀**. Toàn bộ code dùng log₁₀ nhất quán cho cả θ và hàm mục tiêu. |
| **Eq. (15) tính AIFS hai lần** | Paper viết `Ĉ(z) = ê(z)·z^{T_C/δ}` trong khi `T_k^C = T_RTS + AIFS_k` (Eq. 9) đã chứa AIFS, mà ê(z) cũng là hàm sinh của khoảng chờ AIFS. Code dùng cách đọc nhất quán *(airtime thuần) × (hàm sinh chờ AIFS)*: `Ĉ(z) = ê(z)·z^{T_RTS/δ}`, `Ĝ_ℓ(z) = ê(z)·z^{(T_RTS+T_CTS+N_ℓΔ_ℓ)/δ}`. Chỉ với cách đọc này mới có `Ŷ(1) = 1` và `D(1) = 1`. |
| **Tham số nghịch đảo Eq. (16)** | Paper nói `l = 1, r = 10^{-8/N}`. Thực nghiệm cho thấy điều kiện bắt buộc là **2N > x**, nếu không các hệ số `a_{x−2N}` bị khuếch đại bởi `r^{−2N}` và kết quả sai hoàn toàn. Code dùng `l = 0.6, γ = 1.5`; đã đối chiếu với cấu hình chính xác cao (`l = 2, γ = 4`): lệch < 1e-3 tương đối, kể cả ở vùng xác suất 1e-8. |
| **Bước lượng tử hoá δ** | Số mũ của hàm sinh phải là số nguyên (nếu không z^a đa trị trên mặt phẳng phức). Mọi thời lượng được làm tròn về bội của δ = 10 µs; riêng AC có D_max = 300 ms dùng δ = 20 µs để giảm một nửa số điểm DFT (`common/utility.choose_delta_us`). |
| **EDCA "mặc định"** | Paper so sánh với "default single-link EDCA settings" nhưng không liệt kê. Code dùng tham số chuẩn IEEE 802.11e: AC1, AC2 → AC_VO; AC3 → AC_VI; AC4 → AC_BE; AC5 → AC_BK. |
| **Cách xử lý ràng buộc trong GA** | Paper chỉ nói "Define constraints: Pr(D ≥ D_max) − ε ≤ 0". Code dùng **ưu tiên khả thi**: mọi cá thể khả thi đều tốt hơn mọi cá thể vi phạm, trong nhóm vi phạm thì xếp hạng theo tổng số bậc độ lớn vượt ngưỡng. Đây là cách diễn đạt sát nghĩa "s.t." nhất và giúp GA vào vùng khả thi nhanh hơn nhiều so với phạt tuyến tính. |
| **Kích thước GA** | TABLE I cho N_pop = 2000, N_gen = 500 ≈ 10⁶ lần đánh giá; mỗi lần phải nghịch đảo hàm sinh cho từng AC (~80 ms) nên tương đương ~22 giờ CPU. Code mặc định dùng N_pop = 80, N_gen = 120 (giữ nguyên tỉ lệ tinh hoa 10 % và p_cross = 0,8). Gọi `GAParams.paper()` nếu muốn đúng cấu hình của paper. |
| **CW_min = 1..3** | Paper cho miền [1, 1023], nhưng CW ≤ 3 khiến mọi trạm phát tức thì (p → 1, c → 1) — điểm suy biến vô nghĩa về vận hành. Lưới tìm kiếm bắt đầu từ CW = 4. |

## 5. Kiểm chứng mô hình bằng mô phỏng

`training/simulate.py` mô phỏng EDCA ở mức slot với đúng các quy ước của mô hình giải
tích (bão hoà, N_k gói mỗi TXOP, đếm lùi chỉ sau AIFSN_k slot rảnh, nhân đôi cửa sổ khi
va chạm, bỏ gói sau R_k lần).

Kịch bản kiểm chứng: 2 AC × 3 trạm, L = 500 B, CW = 16…256, R = 5,
AIFSN = 2 và 5, D_max = 20 ms, 3 triệu slot.

| Đại lượng | Mô phỏng | Mô hình giải tích |
|---|---|---|
| Pr(D₁ ≥ 20 ms) | 3.28e-2 | 3.50e-2 (lệch 6 %) |
| Pr(D₂ ≥ 20 ms) | 4.03e-1 | 3.21e-1 (lệch 25 %) |
| P_loss,1 | 1.40e-4 | 6.25e-4 |
| P_loss,2 | 2.12e-3 | 5.79e-3 |

**Nhận xét.** Phần trễ — thứ mà paper thực sự dùng để đặt ràng buộc QoS — khớp tốt
(6–25 %). Riêng P_loss bị mô hình ước lượng cao hơn khoảng 3–4 lần. Truy nguyên được
rõ: `P_loss = c^R` nên sai số ~25 % trên c bị khuếch đại luỹ thừa R = 5. Sai số trên c
đến từ hai nguồn, cả hai đều nằm trong chính công thức của paper:

1. Eq. (5) dùng `(f_{k,j} − 1)`; với backoff đều trên [0, f−1] thì giá trị đúng theo
   Bianchi là `(f_{k,j} + 1)`. Riêng điểm này làm p_k cao hơn ~13 %.
2. Xấp xỉ giá trị trung bình của cặp Eq. (4)–(5) vốn không bảo toàn chính xác xác suất
   va chạm khi các AC có AIFS khác nhau nhiều.

Code giữ **nguyên văn công thức của paper**; muốn thử biến thể Bianchi thì thay
`common.collision.transmit_prob` (đã kiểm tra: rút ngắn khoảng cách nhưng không xoá hết).

## 6. Kết quả tái hiện so với paper

| Chỉ số | Paper | Tái hiện |
|---|---|---|
| Cấu trúc vùng AIFS (Fig. 2) | Z_j = [1, 3, 4] | **trùng khít** |
| Fig. 3b — P_loss theo AIFSN₂ | AC₁ giảm, AC₂ tăng, cắt nhau tại AIFSN₂ = AIFSN₁ | **trùng khít**, cắt đúng tại AIFSN₂ = 8 |
| Fig. 3a — θ theo AIFSN₂, TXOP₂ | hai mặt cắt nhau; TXOP lớn có hại khi AIFSN nhỏ, có lợi khi AIFSN lớn | **cùng dạng** |
| Fig. 4 — fitness tốt nhất (đơn link) | ≈ 5 | **4.28** |
| Fig. 4 — fitness tốt nhất (MLO) | ≈ 32 | 9.48 |
| Fig. 4 — dạng đường hội tụ | bậc thang, ổn định sau ~150–250 thế hệ | bậc thang, ổn định sau ~80 thế hệ |
| Fig. 5 — EDCA mặc định | vi phạm ngưỡng ε ở nhiều AC | vi phạm ở **cả 5** AC |
| Fig. 5 — hai cấu hình tối ưu | đều đưa Pr(D ≥ D_max) xuống dưới ε | **cả hai đều thoả mãn toàn bộ ràng buộc** |
| Fig. 5b — MLO so với đơn link | P_loss thấp hơn rõ rệt | thấp hơn **1.5–2 bậc độ lớn** ở AC1, AC4, AC5 |
| Fig. 6a — fitness theo ε₁ | MLO luôn cao hơn đơn link | **cùng kết luận** (MLO 5.95→14.78 so với đơn link 2.54) |
| Fig. 6b — đánh đổi fitness ↔ tin cậy | ε₁ chặt hơn ⇒ Σθ cao hơn nhưng fitness thấp hơn | **cùng kết luận** (MLO: Σθ 21.7 → 34.0 khi ε₁ đi từ 1e-4 xuống 1e-8, fitness 14.78 → 5.95) |

### Cấu hình tối ưu tìm được (kịch bản Sec. V)

| AC | link | CW_min | CW_max | AIFSN | TXOP (µs) | R | Pr(D ≥ D_max) | ε |
|---|---|---|---|---|---|---|---|---|
| AC1 | 1 | 8 | 32 | 3 | 256 | 4 | 2.9e-8 | 1e-7 |
| AC2 | 0 | 6 | 36 | 2 | 128 | 4 | 3.0e-7 | 1e-6 |
| AC3 | 0 | 4 | 512 | 3 | 64 | 4 | 1.4e-6 | 1e-4 |
| AC4 | 1 | 45 | 270 | 3 | 672 | 7 | 4.8e-3 | 1e-2 |
| AC5 | 1 | 256 | 1023 | 15 | 4640 | 7 | 5.0e-1 | 5e-1 |

Quy luật mà GA tự tìm ra trùng với phân tích ở Sec. IV.A của paper: AC có ràng buộc
trễ chặt cần **cửa sổ backoff nhỏ, TXOP ngắn và ít lần phát lại** (chặn trên của trễ
hữu hạn và nằm dưới D_max), còn AC nền được đẩy sang AIFSN lớn và TXOP dài để nhường
kênh. Thêm liên kết thứ hai cho phép tách hai nhóm này ra, nên hàm mục tiêu tăng hơn
gấp đôi (4.28 → 9.48).

### Điểm khác biệt đáng chú ý

**Giá trị fitness của MLO thấp hơn paper (9.5 so với ~32).** Hàm mục tiêu Σ −log₁₀(P_loss,i)
tăng khi P_loss giảm, mà P_loss = c^R lại giảm rất nhanh khi số AC trên mỗi liên kết ít
đi. Paper không nói rõ số liên kết M dùng trong Sec. V (Fig. 1 minh hoạ M = 2); nếu dùng
nhiều liên kết hơn thì mỗi liên kết chỉ còn 1–2 AC và giá trị mục tiêu sẽ tăng mạnh.
Với M = 2 như trong code, mỗi liên kết vẫn phải gánh 2–3 AC nên c không thể xuống rất
thấp. Chạy `exp_fig4_fig5(n_links=3)` cho giá trị cao hơn đáng kể — đây là biến số
nhạy nhất khi đối chiếu con số tuyệt đối.

**Đường EDCA đơn link trong Fig. 6 nằm ngang.** Hàm mục tiêu Eq.(18) không phụ thuộc ε;
ε chỉ quyết định tính khả thi. Nghiệm đơn link tốt nhất tìm được đã thoả mãn cả ngưỡng
chặt nhất (ε₁ = 1e-8) nên nó là nghiệm tối ưu cho mọi ε₁ trong dải quét — đường cong
phẳng là kết quả đúng, không phải lỗi. Với MLO, không gian tìm kiếm rộng hơn nên nghiệm
tốt nhất ở ε₁ lỏng không còn khả thi khi ε₁ siết lại, tạo ra bậc thang thấy trong hình.

**Ngân sách tìm kiếm nhỏ hơn paper 100 lần** (8 600 lần đánh giá so với 10⁶). Đường hội
tụ đã phẳng ở ~80 thế hệ nên nghiệm khó cải thiện nhiều, nhưng không loại trừ khả năng
GA đầy đủ tìm được cấu hình tốt hơn.

## 7. Các hình xuất ra

| File | Nội dung |
|---|---|
| `figures/fig02_aifs_zone_model.png` | Cấu trúc vùng AIFS (φ_j, Z_j, slot, vùng) sinh từ chính mô hình |
| `figures/fig03_parameter_sensitivity.png` | (a) mặt θ theo AIFSN₂ và TXOP₂ · (a') bản đồ chênh lệch θ₂−θ₁ · (b) P_loss theo AIFSN₂ |
| `figures/fig04_ga_convergence.png` | Hội tụ của GA cho MLO EDCA và EDCA đơn link |
| `figures/fig05_qos_comparison.png` | So sánh QoS ba cấu hình: (a) xác suất vi phạm trễ · (b) xác suất mất gói |
| `figures/fig06_epsilon_impact.png` | Ảnh hưởng của ε₁: (a) giá trị thích nghi · (b) tổng chỉ số tin cậy |

Dữ liệu thô của mỗi hình lưu song song trong `results/figN.json`, vẽ lại được mà không
cần chạy lại thí nghiệm:

```python
from training.experiments import load
from plotting.figures import plot_fig5
plot_fig5(load("fig45"))
```

## 8. Ghi chú về trình bày hình

- Bảng màu phân loại đã chạy qua bộ kiểm chứng: nằm trong dải độ sáng L 0.43–0.77,
  chroma ≥ 0.1, khoảng cách CVD nhỏ nhất ΔE = 9.1 (protan).
- Màu gán **cố định theo thực thể** (AC / cấu hình), không xoay vòng theo thứ tự vẽ.
- Không dùng hai trục y trên cùng một ô vẽ; đại lượng khác thang đo được tách thành
  các ô riêng dùng chung trục x.
- Fig. 3 giữ mặt 3D như paper, đồng thời bổ sung bản đồ nhiệt θ₂−θ₁ vì mặt 3D che khuất
  nhau ở vùng giao.
