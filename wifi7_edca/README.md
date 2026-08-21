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
│   ├── genome.py              #   Không gian tham số EDCA + mã hoá/giải mã nhiễm sắc thể
│   ├── graph.py               #   [mục 9] Cấu hình EDCA → đồ thị cho GNN
│   └── dataset.py             #   [mục 9] Sinh tập huấn luyện (song song đa nhân)
│
├── training/                  # ===== TỐI ƯU / "HUẤN LUYỆN" =====
│   ├── ga.py                  #   Algorithm 1 — thuật toán di truyền
│   ├── baselines.py           #   EDCA mặc định 802.11e · tìm kiếm ngẫu nhiên
│   ├── simulate.py            #   Mô phỏng EDCA mức slot — KIỂM CHỨNG mô hình giải tích
│   ├── experiments.py         #   exp_fig2 … exp_fig6 → dict dữ liệu thô (results/*.json)
│   ├── gnn.py                 #   [mục 9] GNN surrogate + lớp `Surrogate`
│   ├── train_gnn.py           #   [mục 9] Huấn luyện surrogate
│   └── ga_surrogate.py        #   [mục 9] Algorithm 1 có GNN sàng lọc
│
├── plotting/                  # ===== VẼ KẾT QUẢ =====
│   ├── style.py               #   Bảng màu đã kiểm chứng CVD, gán màu cố định theo thực thể
│   └── figures.py             #   plot_fig2 … plot_fig6 → figures/*.png
│
└── main.py                    # pipeline: gen data → utility → training → vẽ
```

Luồng dữ liệu một chiều: `datagen → common (utility) → training → plotting`.
Module `plotting` **không tính lại** bất kỳ đại lượng nào; module `training` **không vẽ**.

Năm file đánh dấu `[mục 9]` là **phần mở rộng ngoài paper** (GNN surrogate). Bỏ hẳn
chúng đi thì toàn bộ phần tái hiện Fig. 2–6 vẫn chạy nguyên vẹn; chúng không nằm
trên đường dữ liệu của `main.py`.

## 2. Cách chạy

```bash
python main.py               # chạy tất cả (Fig. 2-6)
python main.py 3             # chỉ chạy một hình
python main.py --quick       # GA nhỏ, kiểm tra pipeline nhanh (ghi ra *_quick.json/png)

python -m common.zones       # in cấu trúc vùng AIFS, đối chiếu Fig. 2
python -m common.collision   # giải điểm cố định EDCA cho cấu hình mặc định
python -m common.timing      # in bảng thời lượng T_DATA, Δ_k, N_k, T^C, T^S
python -m common.delay       # kiểm tra D(1) = 1 và tính Pr(D ≥ D_max)
python -m common.utility     # đánh giá QoS đầy đủ, so sánh 1 link và 2 link
python -m training.simulate  # KIỂM CHỨNG mô hình bằng mô phỏng mức slot
```

Thời gian chạy tham khảo (Python 3.10, 1 nhân): một lần đánh giá cấu hình 5 AC mất
~16 ms, nên Fig. 2 và 3 xong trong vài giây; Fig. 4 và 5 khoảng 30–35 phút (2 lần chạy
GA, N_pop = 200, N_gen ≤ 300); Fig. 6 khoảng 25–30 phút (10 lần chạy GA độc lập).

Yêu cầu: `numpy`, `scipy`, `matplotlib`.

Phần mở rộng GNN (mục 9) — cần thêm `torch`, **không** cần `torch_geometric`:

```bash
python -m datagen.dataset --n 200000 --ga-frac 0.45 --out results/gnn_data.npz
python -m training.train_gnn --data results/gnn_data.npz --epochs 60 --out results/gnn_model.pt
python -m training.ga_surrogate --model results/gnn_model.pt --links 2
```

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

**Kiểm chứng nội tại:** `python -m common.delay` in ra `D(1⁻) = 1.00000` — hàm sinh
Eq.(10)–(15) là một hàm sinh xác suất hợp lệ (tổng khối lượng bằng 1). Ngoại lệ duy
nhất là AC5 của cấu hình EDCA mặc định (`D(1⁻) = 0.58`): AIFSN = 7 trong khi c = 0.974
nên xác suất kênh rảnh liên tục 5 slot chỉ cỡ 1e-8, kỳ vọng thời gian chờ AIFS lớn tới
mức phép đánh giá tại `z = 1 − 1e-12` chưa hội tụ về 1. Đây không phải lỗi mà chính là
hiện tượng AC nền bị bỏ đói — đúng bằng cột `Pr(D > D_max) = 1.0` của AC5 trong Fig. 5a.

**Kiểm chứng cấu trúc vùng:** `python -m common.zones` với AIFS = {30, 50, 50, 90} µs
cho `Z_j = [1, 3, 4]` — đúng bằng hàng `Z_j` trong Fig. 2 của paper.

## 4. Những chỗ paper không nói rõ và cách xử lý

| Vấn đề | Xử lý trong code |
|---|---|
| **Cơ số logarit trong Eq. (17), (18)** | Paper chỉ viết "log". Thang giá trị của Fig. 4 (fitness đạt ~35 với 5 AC) và các mức P_loss trong Fig. 5b (1e-10 … 1e-1) chỉ khớp khi dùng **log₁₀**. Toàn bộ code dùng log₁₀ nhất quán cho cả θ và hàm mục tiêu. |
| **Eq. (15) tính AIFS hai lần** | Paper viết `Ĉ(z) = ê(z)·z^{T_C/δ}` trong khi `T_k^C = T_RTS + AIFS_k` (Eq. 9) đã chứa AIFS, mà ê(z) cũng là hàm sinh của khoảng chờ AIFS. Code dùng cách đọc nhất quán *(airtime thuần) × (hàm sinh chờ AIFS)*: `Ĉ(z) = ê(z)·z^{T_RTS/δ}`, `Ĝ_ℓ(z) = ê(z)·z^{(T_RTS+T_CTS+N_ℓΔ_ℓ)/δ}`. Chỉ với cách đọc này mới có `Ŷ(1) = 1` và `D(1) = 1`. |
| **Tham số nghịch đảo Eq. (16)** | Paper nói `l = 1, r = 10^{-8/N}`. Thực nghiệm cho thấy điều kiện bắt buộc là **2N > x**, nếu không các hệ số `a_{x−2N}` bị khuếch đại bởi `r^{−2N}` và kết quả sai hoàn toàn. Code dùng `l = 0.6, γ = 1.5`; đã đối chiếu với cấu hình chính xác cao (`l = 2, γ = 4`): lệch < 1e-3 tương đối, kể cả ở vùng xác suất 1e-8. |
| **Bước lượng tử hoá δ** | Số mũ của hàm sinh phải là số nguyên (nếu không z^a đa trị trên mặt phẳng phức). Mọi thời lượng được làm tròn về bội của δ = 10 µs; riêng AC có D_max = 300 ms dùng δ = 20 µs để giảm một nửa số điểm DFT (`common/utility.choose_delta_us`). |
| **EDCA "mặc định"** | Paper so sánh với "default single-link EDCA settings" nhưng không liệt kê bảng. Code dùng bảng mặc định của **PHY OFDM** (aCWmin = 15, IEEE 802.11-2020 Table 9-155): AC_VO (3, 7, 2, 1504 µs), AC_VI (7, 15, 2, 3008 µs), AC_BE (15, 1023, 3, 0), AC_BK (15, 1023, 7, 0); ánh xạ AC1, AC2 → AC_VO; AC3 → AC_VI; AC4 → AC_BE; AC5 → AC_BK. Đây là bảng **duy nhất** tái hiện đúng dạng Fig. 5: cửa sổ tranh chấp rất nhỏ ⇒ P_loss của cả 5 AC nằm ở mức 0.5–0.9 (Fig. 5b), đồng thời trễ của AC1–AC3 bị chặn chặt nên Pr(D ≥ D_max) của chúng nằm sâu dưới ε, còn AC4/AC5 vi phạm nặng (Fig. 5a). Bảng aCWmin = 31 (PHY DSSS) cho P_loss ≈ 0.1 và làm AC1–AC3 vi phạm trễ — không khớp Fig. 5. |
| **Cách xử lý ràng buộc trong GA** | Paper chỉ nói "Define constraints: Pr(D ≥ D_max) − ε ≤ 0". Vì Eq.(18) luôn ≥ 0 (do P_loss ≤ 1), code cho cá thể vi phạm giá trị `−λ·Σ excess` (λ = 0.5 điểm cho mỗi bậc độ lớn vượt ngưỡng) — luôn ≤ 0 nên **mọi nghiệm khả thi tự động xếp trên mọi nghiệm vi phạm**, đúng nghĩa "s.t." mà không cần hằng số phạt khổng lồ. Cách này cũng tái hiện đúng dạng Fig. 4 của paper: đường *Best* xuất phát từ ~0, đường *Mean* quanh −4, cả hai nhảy bậc thang khi GA tìm được vùng khả thi. |
| **Kích thước GA** | TABLE I cho N_pop = 2000, N_gen = 500 ≈ 10⁶ lần đánh giá; mỗi lần phải nghịch đảo hàm sinh cho từng AC nên tương đương nhiều giờ CPU. Code mặc định dùng N_pop = 200 nhưng **giữ nguyên N_gen = 300 và N_stag = 50** để trục hoành của Fig. 4 trùng thang với paper; phần quần thể bị cắt bớt được bù bằng một bước tinh chỉnh theo toạ độ (`training/ga.coordinate_polish`). Gọi `GAParams.paper()` nếu muốn đúng cấu hình của paper. |
| **CW_min = 1..3** | Paper cho miền [1, 1023], nhưng CW ≤ 3 khiến mọi trạm phát tức thì (p → 1, c → 1) — điểm suy biến vô nghĩa về vận hành. Lưới tìm kiếm bắt đầu từ CW = 4. |
| **Tỉ số CW_max / CW_min** | Miền [1, 1023] cho phép **CW_max = CW_min** (m_k = 0, không nhân đôi cửa sổ). Điều này thiết yếu: chỉ khi cửa sổ không nhân đôi thì đuôi trễ mới đủ ngắn để đồng thời đạt P_loss ~ 1e-10 *và* Pr(D ≥ D_max) < ε. Lưới gen mã hoá CW_max = CW_min · 2^m với m = 0…10 (`datagen/genome.CW_SPAN_LEVELS`). Nếu ép m ≥ 2 thì fitness của MLO đứng ở ~10 thay vì ~35. |
| **f_{k,j} phải là số nguyên** | Paper viết `f_{k,j} = 2^{min(j, m_k)} CW_min,k` với `m_k = log2(CW_max/CW_min)`. Khi tỉ số không phải luỹ thừa của 2 thì m_k lẻ và `Y(z)^{f}` rơi vào nhánh cắt của hàm mũ phức. Code dùng dạng tương đương **`f_{k,j} = min(2^j·CW_min, CW_max)`** — trùng công thức của paper khi tỉ số là luỹ thừa của 2, và luôn cho số nguyên. |

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

### 5.1. Kiểm chứng đuôi trễ ngay tại nghiệm tối ưu

Kịch bản trên chỉ chạm tới xác suất ~1e-2. Vì ràng buộc của bài toán nằm ở vùng
1e-7…1e-4, cần kiểm tra riêng **đúng cấu hình mà GA chọn**. Mô phỏng 6 triệu slot
trên link 0 và link 1 của nghiệm *Opt. MLO EDCA*:

| AC | x | Pr(D ≥ x) mô phỏng | Pr(D ≥ x) mô hình |
|---|---|---|---|
| AC1 | 10 ms | 2.12e-1 | 1.78e-1 |
| AC1 | 20 ms | 2.72e-3 | 4.06e-3 |
| AC1 | 35 ms | 8.61e-6 | 1.83e-5 |
| AC2 | 24 ms | 4.84e-3 | 7.33e-3 |
| AC2 | 42 ms | 1.25e-5 | 5.07e-5 |
| AC3 | 70 ms | 2.64e-3 | 2.38e-3 |
| AC3 | 100 ms (= D_max) | 1.12e-4 | 6.92e-5 |
| AC4 | 120 ms | 6.60e-3 | 5.17e-3 |
| AC5 | 210 ms | 3.81e-3 | 1.29e-3 |

Mô hình bám đuôi trễ qua **5–6 bậc độ lớn**, sai lệch trong khoảng 1.2–4 lần và phần
lớn là **ước lượng thừa** (bảo thủ) ở AC1/AC2 — tức các ràng buộc chặt nhất không
được thoả mãn nhờ sai số số học. Ngoại suy đuôi thực nghiệm của AC1 (2.7e-3 tại 20 ms
→ 8.6e-6 tại 35 ms) tới D_max = 50 ms cho ~3e-8, cùng bậc với 7.7e-8 mà mô hình đưa ra.

**Cảnh báo cần ghi nhận:** AC3 nằm sát biên — mô hình cho 6.9e-5 (đạt ε₃ = 1e-4) trong
khi mô phỏng cho 1.1e-4 (vượt nhẹ). Nghiệm tối ưu luôn bị đẩy tới sát ràng buộc, nên ở
đó sai số ±2 lần của mô hình là quyết định. Đây là hạn chế của **chính mô hình giải
tích trong paper**, không phải của bản tái hiện.

## 6. Kết quả tái hiện so với paper

Tất cả số liệu dưới đây lấy từ `results/*.json` của lần chạy `python main.py`
(N_pop = 200, N_gen = 300, N_stag = 50, p_cross = 0.8, p_mutate = 0.05, seed = 2025).

### 6.1. Đối chiếu từng hình

| Chỉ số | Paper | Tái hiện |
|---|---|---|
| Cấu trúc vùng AIFS (Fig. 2) | Z_j = [1, 3, 4] | **[1, 3, 4] — trùng khít** |
| Fig. 3b — P_loss theo AIFSN₂ | AC₁ 1.8e-4 → 3.3e-6, AC₂ ngược lại, cắt nhau tại AIFSN₂ = 8 ở ~8e-5 | **1.6e-4 → 3.0e-6, cắt tại AIFSN₂ = 8 ở 7.6e-5 — gần như trùng khít** |
| Fig. 3a — θ theo AIFSN₂, TXOP₂ | hai mặt cắt nhau, θ ∈ [0.2, 3.5] | **cùng dạng**, θ ∈ [0.35, 2.7] |
| Fig. 4 — dạng đường hội tụ | bậc thang, Best và Mean cùng dâng, dừng sớm ~195/278 thế hệ | **cùng dạng**, dừng sớm ở 177/175 thế hệ |
| Fig. 4 — Best (đơn link / MLO) | 4.7 / 34.8 | **12.9 / 49.7** (xem 6.3) |
| Fig. 4 — Mean cuối (đơn link / MLO) | ≈ 1 / ≈ 28 | 5.3 / **24.5** |
| Fig. 5a — Default EDCA | AC1–AC3 nằm sâu dưới ε, AC4 ≈ 0.35 và AC5 ≈ 1.0 vi phạm | **cùng kết luận**: AC1–AC3 ≤ 2.3e-9, AC4 = 0.82, AC5 = 1.0 |
| Fig. 5a — hai cấu hình tối ưu | mọi AC đều dưới ε | **cả hai đều thoả mãn toàn bộ ràng buộc** |
| Fig. 5b — Default EDCA | P_loss ≈ 0.5–0.9 cho cả 5 AC | **0.73–0.83** |
| Fig. 5b — Opt. EDCA đơn link | vài AC bị bỏ đói (P_loss ≈ 1), AC1/AC4 ≈ 1e-3…1e-4 | **cùng dạng**: AC2 = 0.88, AC3 = 0.96; AC1/AC4/AC5 ≈ 2e-5…9e-5 |
| Fig. 5b — Opt. MLO EDCA | AC3–AC5 chạm sàn ~1e-10, AC1/AC2 ≈ 1e-4 | **cả 5 AC đều chạm sàn 1e-10** |
| Fig. 6a — fitness theo ε₁ | không đơn điệu, thấp nhất tại ε₁ = 1e-8; MLO luôn cao hơn đơn link | **cùng dạng**, EDCA [3.6, 6.8, 4.5, 4.8, 4.7] so với paper [1.6, 4.5, 5.6, 6.2, 5.2] |
| Fig. 6b — Σθ của EDCA đơn link | 25.0 → 17.9 khi ε₁ đi từ 1e-8 tới 1e-4 | **25.2 → 17.6 — gần như trùng khít** |
| Fig. 6b — Σθ của MLO EDCA | 31.4 … 22.9, luôn nằm trên đường EDCA | **28.5 … 31.7**, luôn nằm trên đường EDCA |
| Fig. 6b — đường Target ε | Σ −log₁₀ ε_i = 20.3 → 16.3 | **20.3 → 16.3 — trùng khít** |

### 6.2. Cấu hình tối ưu tìm được (kịch bản Sec. V)

**Opt. MLO EDCA** — mục tiêu Eq.(18) = 49.70, thoả mãn toàn bộ ràng buộc:

| AC | link | CW_min | CW_max | AIFSN | TXOP (µs) | R | Pr(D ≥ D_max) | ε | P_loss |
|---|---|---|---|---|---|---|---|---|---|
| AC1 | 0 | 181 | 181 | 2 | 0 | 7 | 7.7e-8 | 1e-7 | 1.4e-11 |
| AC2 | 1 | 256 | 256 | 3 | 32 | 7 | 3.2e-7 | 1e-6 | 1.4e-11 |
| AC3 | 0 | 512 | 512 | 5 | 0 | 7 | 6.9e-5 | 1e-4 | 9.9e-11 |
| AC4 | 1 | 1023 | 1023 | 8 | 608 | 7 | 1.1e-7 | 1e-2 | 6.2e-11 |
| AC5 | 0 | 1023 | 1023 | 15 | 3616 | 7 | 3.1e-5 | 5e-1 | 2.0e-10 |

**Opt. EDCA đơn link** — mục tiêu = 12.91: AC2 và AC3 bị bỏ đói (P_loss 0.88 và 0.96)
để AC1, AC4, AC5 giữ được P_loss ≈ 1e-5…1e-4. Đúng cơ chế mà Fig. 5b của paper cho
thấy: khi chỉ có một liên kết, GA buộc phải hy sinh hẳn vài AC.

Quy luật GA tự tìm ra trùng với phân tích Sec. IV.A của paper: **CW_max = CW_min**
(cửa sổ không nhân đôi) cho chặn trên của trễ chặt nhất, CW_min lớn cho c nhỏ, AC nền
được đẩy sang AIFSN lớn và TXOP dài để nhường kênh. Liên kết thứ hai cho phép tách
AC1/AC2 (ràng buộc trễ chặt) khỏi nhau, nên mục tiêu tăng gần **4 lần** (12.9 → 49.7).

### 6.3. Vì sao giá trị fitness CAO HƠN paper

Đây là khác biệt lớn nhất và cần nói rõ: bản tái hiện cho **49.7 / 12.9** so với
**34.8 / 4.7** của paper — tức bộ tối ưu ở đây tìm được nghiệm *tốt hơn*, không phải
kém hơn. Ba việc đã làm để loại trừ khả năng đây là lỗi:

1. **Không phải nhiễu số học của phép nghịch đảo.** Tính lại Pr(D ≥ D_max) của nghiệm
   tối ưu với cấu hình nghịch đảo chặt gấp 3 (`l = 2, γ = 4` thay cho `l = 0.6,
   γ = 1.5`) cho kết quả trùng tới **4 chữ số có nghĩa** ở cả 5 AC.
2. **Không phải sai của mô hình trễ.** Mô phỏng mức slot ngay tại nghiệm tối ưu
   (mục 5.1) cho thấy mô hình bám đuôi trễ qua 5–6 bậc độ lớn, và ở hai AC có ràng
   buộc chặt nhất (AC1, AC2) mô hình còn **ước lượng thừa**.
3. **Khác biệt nằm ở cấu hình được chọn.** Trong Fig. 5b của paper, AC1 và AC2 của
   nghiệm MLO dừng ở P_loss ≈ 1e-4 — nghĩa là nghiệm của họ phải dùng cửa sổ tranh
   chấp NHỎ cho hai AC này để giữ chặn trễ. Nghiệm ở đây dùng CW = 181 và 256 mà vẫn
   đạt Pr(D ≥ D_max) < ε, nên P_loss = c^7 rơi xuống tận 1e-11.

Nguyên nhân trực tiếp là lưới tìm kiếm: nếu ép `CW_max ≥ 4·CW_min` (như bản cài đặt
trước) thì đuôi trễ dài ra và fitness MLO đứng ở ~10. Chỉ khi cho phép **m_k = 0** —
điều mà miền [1, 1023] của Sec. IV.B hoàn toàn cho phép — mới xuất hiện lớp nghiệm
này. Không loại trừ khả năng cài đặt của paper vô tình loại bỏ lớp nghiệm đó.

**Hệ quả cần lưu ý:** với sàn P_loss = 1e-10, trần lý thuyết của hàm mục tiêu là
5 × 10 = 50, và nghiệm MLO đạt 49.7 — tức **chạm trần**. Nói cách khác, dưới mô hình
này chỉ còn ràng buộc trễ là ràng buộc thực sự; xác suất mất gói đã bị đẩy xuống dưới
độ phân giải của chính mô hình. Con số 49.7 vì vậy nên đọc là "hết trần báo cáo",
không phải một giá trị tối ưu có ý nghĩa tuyệt đối.

### 6.4. Các khác biệt còn lại

**Nghiệm nằm sát biên ràng buộc.** AC3 của nghiệm MLO có Pr(D ≥ 100 ms) = 6.9e-5 theo
mô hình (đạt ε₃ = 1e-4) nhưng 1.1e-4 theo mô phỏng (vượt nhẹ). GA luôn đẩy nghiệm tới
sát ràng buộc, nên ở đó sai số ±2 lần của mô hình giải tích là quyết định. Đây là hạn
chế của mô hình trong paper chứ không riêng bản tái hiện.

**Ngân sách tìm kiếm nhỏ hơn paper.** ~31 000 lần đánh giá cho mỗi lần chạy so với
10⁶ của TABLE I. Đường hội tụ đã phẳng và dừng sớm (N_stag = 50) trước khi hết
N_gen = 300, nên nghiệm khó cải thiện thêm đáng kể.

**Fig. 6 dùng GA độc lập cho từng ε₁.** Đúng như cách paper sinh Fig. 6 — và vì thế
đường cong không đơn điệu, đúng như paper nhận xét. Một bản cài đặt trước gộp chung kho
nghiệm giữa các ε₁ để khử nhiễu; cách đó cho đường phẳng, tối ưu hơn nhưng không còn
giống hình của paper, nên đã bỏ.

## 7. Các hình xuất ra

| File | Nội dung |
|---|---|
| `figures/fig02_aifs_zone_model.png` | Cấu trúc vùng AIFS (φ_j, Z_j, slot, vùng) sinh từ chính mô hình |
| `figures/fig03_parameter_sensitivity.png` | (a) hai mặt θ theo AIFSN₂ và TXOP₂ · (b) P_loss theo AIFSN₂ — cùng bố cục với Fig. 3 của paper |
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

Fig. 3–6 được vẽ theo **đúng kiểu trình bày của paper** (`plotting/style.paper_style`)
để đặt cạnh bản PDF là đối chiếu được từng đường / từng cột:

- nền trắng, khung hộp đầy đủ, tick hướng vào trong, chữ serif — như hình MATLAB
  trong paper;
- nhãn trục và legend giữ **nguyên văn tiếng Anh của paper** (`AC₂ AIFSN`,
  `Packet Loss Rate P_loss`, `Fitness Value`, `Generation`, `Pr(D > D_max)`,
  `Target ε`, …);
- mã màu lấy trực tiếp từ các hình trong bản PDF: Fig. 3 — AC₁ cam / AC₂ xanh;
  Fig. 4 và 6 — EDCA đơn link xanh lam nét liền, MLO EDCA đỏ nét đứt;
  Fig. 5 — Opt. EDCA xanh, Opt. MLO EDCA hồng, Default EDCA cam, Target ε tím;
- thang trục giữ nguyên như paper: Fig. 4 `x ∈ [0, 300]`, `y ∈ [−10, 40]`;
  Fig. 5a `y ∈ [1e-8, 1]`, Fig. 5b `y ∈ [1e-10, 1]`; Fig. 6a `y ∈ [0, 40]`.
  Riêng trục tung của Fig. 4 và Fig. 6a được **nới ra** khi dữ liệu vượt 40 (bản tái
  hiện đạt tới ~50) — không bao giờ cắt mất đường cong để giữ đúng khung hình;
- cột nào rơi dưới cận dưới của trục trong Fig. 5 được vẽ thành một đoạn ngắn **gạch
  chéo** sát đáy trục thay vì biến mất; giá trị thật nằm trong `results/fig45.json`.

Riêng Fig. 2 là sơ đồ minh hoạ mô hình vùng AIFS (paper vẽ tay), nên giữ kiểu trình
bày riêng của repo: bảng màu phân loại đã kiểm chứng CVD (dải độ sáng L 0.43–0.77,
chroma ≥ 0.1, ΔE nhỏ nhất 9.1 với protan), màu gán cố định theo thực thể.

## 9. Mở rộng ngoài paper: GNN surrogate cho hàm thích nghi

Toàn bộ mục này **không thuộc paper**. Nó trả lời một câu hỏi mà paper để ngỏ: mỗi
lần đổi kịch bản phải chạy lại GA từ đầu, mà một lần chạy đúng cấu hình TABLE I
(N_pop = 2000, N_gen = 500 ≈ 10⁶ lần đánh giá) tốn hàng giờ CPU. Fig. 6 của paper
cần **10 lần chạy GA độc lập** chỉ để quét một tham số.

Điểm cần nói rõ ngay: **GNN không thay thế mô hình giải tích, nó chỉ sàng lọc.**
Nghiệm cuối cùng luôn được `common.utility.evaluate_config` kiểm chứng, nên mọi con
số báo cáo vẫn chính xác tuyệt đối — chỉ quá trình *tìm kiếm* được tăng tốc.

### 9.1. Vì sao là GNN chứ không phải MLP

Đây không phải phép loại suy mà là trùng khớp cấu trúc. Eq. (4) trong
`common/collision.py`:

```
c_k = Σ_j (π_j / Σπ) · (1 − (Π_l r_l^{n_l}) / r_k)
```

Lấy log phần tích: `log Π_l r_l^{n_l} = Σ_l n_l · log r_l`.

Vế phải là một phép **sum-aggregation bất biến hoán vị** trên các node láng giềng,
với `n_l` là trọng số cạnh và `log r_l` là message. Lớp ngoài — trung bình có trọng
số `π_j` theo vùng AIFS — là phép aggregation thứ hai, đúng dạng attention theo vùng.
Nghĩa là **vòng lặp điểm cố định Eq.(4)–(5) chính là T bước message passing trên đồ
thị đầy đủ**. Ba hệ quả:

| | MLP | GNN |
|---|---|---|
| Bất biến hoán vị các AC | phải học từ dữ liệu | có sẵn trong kiến trúc |
| Số AC thay đổi (3…6) | không dùng lại được | dùng chung một mạng |
| Phép gán link MLO | chỉ là một giá trị đầu vào | **là topology của đồ thị** |

Dòng cuối là lý do mạnh nhất: biến `link` trong `datagen/genome.py` quyết định AC nào
tranh chấp với AC nào, tức nó thay đổi *cấu trúc* bài toán chứ không phải giá trị đặc
trưng. MLO vốn dĩ là bài toán đồ thị.

### 9.2. Biểu diễn đồ thị (`datagen/graph.py`)

| Thành phần | Nội dung |
|---|---|
| Node | Mỗi AC là một node (không có node link riêng) |
| Cạnh | AC_i — AC_j nếu **cùng link**; ma trận kề chính là phép gán MLO |
| Đặc trưng node (17) | Kịch bản `n_i, L_i, D_max,i, log₁₀ε_i` · biến quyết định `CW_min, log₂(CW_max/CW_min), AIFSN, TXOP, R` · **các đại lượng closed-form Eq.(7)–(9)** từ `common/timing.py` · tải trên link |
| Đặc trưng cạnh (5) | `n_j` (đúng là trọng số trong Eq. 4) · độ lệch AIFS `h_j − h_i` (Eq. 2) · j có tranh chấp từ vùng của i không · cùng vùng AIFS không |

Mọi đại lượng đã có công thức đóng và rẻ (Eq. 2, Eq. 7–9) đều được **cho vào làm đặc
trưng** thay vì bắt mạng học lại — chính xác tuyệt đối và miễn phí.

Đồ thị chỉ 3–8 node nên dùng **ma trận kề dày** trong torch thuần, không cần
`torch_geometric`: ở quy mô này dense einsum nhanh hơn scatter thưa, và tránh hẳn
việc cài PyG trên Windows.

### 9.3. Sinh dữ liệu — và vấn đề vùng khả thi mỏng như dao cạo

Nhãn do **chính mô hình giải tích** sinh ra, nên dữ liệu là miễn phí: không cần ns-3,
không cần đo đạc. Nhưng lấy mẫu ngẫu nhiên thất bại hoàn toàn. Đo thực tế trên kịch
bản 5 AC của Sec. V:

| Cách lấy mẫu | Tỉ lệ khả thi |
|---|---|
| Ngẫu nhiên đều trên miền Sec. IV.B | **0 / 300** |
| "Có cấu trúc" (CW lớn, m_k nhỏ, R = 7) | **0 / 300** |
| Nhiễu quanh *chính nghiệm tối ưu* (σ = 0.15) | **1.5 %** |

Riêng từng AC trong vùng có cấu trúc: AC1 4 %, AC2 5 %, AC3 25 %, AC4 90 %, AC5 98 %.
AC1 và AC2 (ε = 1e-7, 1e-6) là ràng buộc siết, và khả thi đòi hỏi **cả 5 AC đồng
thời**. Surrogate huấn luyện bằng mẫu ngẫu nhiên sẽ không bao giờ nhìn thấy vùng mà
GA thực sự ra quyết định.

Cách xử lý: **ghi lại quỹ đạo GA**. Mọi cấu hình mà GA đánh giá đều cần
`evaluate_config` để chạy — ta chỉ ghi lại kết quả sẵn có, nên nguồn dữ liệu này
**không tốn thêm một lần đánh giá nào**, mà phân bố của nó khớp đúng phân bố mà
surrogate sẽ gặp khi làm việc. Tập dữ liệu trộn 45 % quỹ đạo GA + 55 % lấy mẫu rộng
(để giữ khả năng tổng quát hoá sang kịch bản 3–6 AC).

Kết quả trên 200 000 mẫu: **8.19 % khả thi (16 373 mẫu), 19.10 % sát biên ràng buộc
(38 201 mẫu)** — so với 0 % của lấy mẫu thuần ngẫu nhiên. Chi phí: **159 giây** trên
26 nhân (1 257 mẫu/s). Bước này thuần CPU, GPU không tham gia.

### 9.4. Kiến trúc và huấn luyện

```
h_i⁰ = Enc(x_i)
lặp T lần (CHIA SẺ TRỌNG SỐ):
    m_ij = Msg([h_i, h_j, e_ij])
    a_i  = Σ_j adj_ij · m_ij   ‖  max_j adj_ij · m_ij
    h_i  = h_i + Upd([h_i, a_i])          ← residual: một bước lặp điểm cố định
(log₁₀ c_i, θ_i) = Out(h_i)
```

| Lựa chọn | Lý do |
|---|---|
| **Sum**-aggregation (không phải mean) | Eq.(4) chứa `Σ_l n_l log r_l` — một TỔNG. Mean sẽ chuẩn hoá mất thông tin "có bao nhiêu trạm đang tranh chấp", đúng là đại lượng quyết định |
| **Chia sẻ trọng số** qua T = 6 lớp | `solve_link` lặp cùng một toán tử mỗi vòng. T lớp mô phỏng T *vòng lặp*, không phải T bước lan truyền khoảng cách (đồ thị đầy đủ, 1 hop là hết) |
| Học `log₁₀ c` chứ không học `P_loss` | Eq.(18) = `Σ_i −R_i·log₁₀ c_i` — **tuyến tính** theo log₁₀c. `P_loss = c^R` được tính CHÍNH XÁC sau đó, mạng không bao giờ phải học phép luỹ thừa ^R vốn khuếch đại sai số |
| Học `θ` chứ không học `Pr(D ≥ D_max)` | Ràng buộc nằm ở vùng 1e-7…1e-4; hồi quy xác suất thô ở thang đó là vô vọng |

156 386 tham số (hidden 128, 6 lớp, edge_hidden 32). Huấn luyện 60 epoch, batch 512,
AdamW + OneCycle lr 2e-3, Huber loss chuẩn hoá theo độ lệch chuẩn từng nhãn, bf16
autocast: **387 giây trên RTX 4060**.

### 9.5. Độ chính xác của surrogate

Trên 20 000 mẫu kiểm tra tách riêng (1 640 mẫu khả thi, 3 778 mẫu sát biên):

| Chỉ số | Toàn tập | Riêng tập **sát biên ràng buộc** |
|---|---|---|
| RMSE `log₁₀ c` | **0.0108** | — |
| RMSE `θ` | **0.3929** (≈ 2.47× về xác suất) | **0.4519** |
| MAE `θ` | 0.1528 | — |
| RMSE `excess` | 0.0586 | — |
| Spearman ρ của fitness | **0.9996** | **0.9824** |
| Độ chính xác quyết định khả thi | **99.19 %** | **95.74 %** |

Cột bên phải mới là cột đáng đọc: GA luôn đẩy nghiệm tới sát ràng buộc (mục 6.4), nên
sai số ở vùng sát biên mới là thứ quyết định, không phải RMSE tổng.

RMSE `log₁₀ c` = 0.0108 tương đương **0.076 điểm fitness mỗi AC** khi R = 7, tức cỡ
0.2–0.4 điểm trên tổng ~50. Nhỏ hơn hẳn khoảng cách 34.8 → 49.7 giữa paper và bản tái
hiện (mục 6.3), nên surrogate không thể là nguồn gây nhầm lẫn khi đối chiếu.

RMSE `θ` = 0.393 nghĩa là sai khoảng 2.47× về xác suất — **đúng bằng mức sai số của
chính mô hình giải tích so với mô phỏng mức slot** (1.2–4×, mục 5.1). Surrogate đã
chạm sàn tự nhiên: nó chính xác ngang cái mà nó xấp xỉ, không thể hơn.

### 9.6. Kết quả: GA được surrogate hỗ trợ (`training/ga_surrogate.py`)

Mỗi thế hệ: GNN chấm điểm **cả quần thể** trong một forward pass, chỉ `verify_top = 8`
cá thể hứa hẹn nhất được mô hình giải tích kiểm chứng. Cá thể tốt nhất **luôn** được
kiểm chứng. Bước tinh chỉnh toạ độ cũng được GNN sàng lọc: mạng chấm toàn bộ các mức
của một gen trong một lần gọi, chỉ 3 mức tốt nhất mới đánh giá chính xác — vừa rẻ hơn
`coordinate_polish` vừa tốt hơn vì duyệt hết mọi mức thay vì chỉ lân cận ±2.

Kịch bản Sec. V, 2 link, N_pop = 200, N_gen = 300, N_stag = 50, seed = 2025:

| | GA gốc | GA + GNN |
|---|---|---|
| Thời gian | 157.0 s | **15.7 s** |
| Số lần đánh giá giải tích | 31 244 | **1 667** |
| Số lần đánh giá bằng GNN | 0 | 36 588 |
| Fitness (**đã kiểm chứng giải tích**) | 49.704 | 49.370 |
| Khả thi | True | True |
| Thế hệ dừng | 177 | 191 |

**Tăng tốc 10.03× · giảm 18.74× số lần gọi mô hình giải tích.**

Đáng chú ý: GA + GNN *tìm kiếm nhiều hơn* mà vẫn rẻ hơn 10 lần — 36 588 lần đánh giá
so với 31 244, chạy 191 thế hệ so với 177. Không có đánh đổi "nhanh hơn nhờ tìm ít
hơn".

Nghiệm tìm được (mọi ràng buộc đạt, mục tiêu Eq.(18) = 49.370):

| AC | link | CW_min | CW_max | AIFSN | TXOP (µs) | R | Pr(D ≥ D_max) | ε | P_loss |
|---|---|---|---|---|---|---|---|---|---|
| AC1 | 0 | 181 | 181 | 2 | 32 | 7 | 5.20e-8 | 1e-7 | 1.9e-11 |
| AC2 | 1 | 256 | 256 | 2 | 32 | 7 | 6.67e-7 | 1e-6 | 1.2e-11 |
| AC3 | 0 | 362 | 362 | 14 | 544 | 7 | 9.55e-5 | 1e-4 | 3.1e-10 |
| AC4 | 0 | 1023 | 1023 | 3 | 64 | 7 | 6.21e-8 | 1e-2 | 1.4e-10 |
| AC5 | 1 | 1023 | 1023 | 11 | 352 | 7 | 1.04e-6 | 5e-1 | 6.2e-11 |

**Kiểm chứng chéo mạnh nhất:** AC1 và AC2 — hai AC mang ràng buộc siết — cho ra
`CW = 181` và `CW = 256`, **trùng khít** với nghiệm ở mục 6.2 (181 và 256) dù đi bằng
đường tìm kiếm hoàn toàn khác. Trong khi đó AC3–AC5 lại khác hẳn về AIFSN, TXOP và cả
phép gán link, mà cả hai nghiệm đều cho P_loss chạm sàn 1e-10 ở toàn bộ 5 AC. Điều này
xác nhận nhận định ở mục 6.3: một khi P_loss chạm sàn, hàm mục tiêu **phẳng** và chỉ
còn ràng buộc trễ là thực sự ràng buộc — nên có cả một lớp nghiệm tương đương, và
chênh lệch 49.370 so với 49.704 nằm dưới độ phân giải có ý nghĩa của mô hình.

### 9.7. Hạn chế cần ghi nhận

**Surrogate không thể chính xác hơn cái nó học.** Nhãn do mô hình giải tích sinh ra,
nên GNN **kế thừa nguyên vẹn** sai lệch của mô hình đó — trong đó có việc P_loss bị
ước lượng cao hơn mô phỏng 3–4 lần (mục 5). GNN tăng tốc, không sửa sai số mô hình.
Muốn thực sự chính xác hơn thì phải **tiền huấn luyện trên mô hình giải tích (hàng
trăm nghìn mẫu, rẻ) rồi tinh chỉnh trên `training/simulate.py` (vài nghìn mẫu, đắt)** —
việc này chưa làm.

**Con số 10× phụ thuộc ngân sách tìm kiếm.** Với ngân sách nhỏ, bước tinh chỉnh toạ độ
chiếm tỉ trọng lớn và tỉ lệ tăng tốc thấp hơn; với cấu hình TABLE I (10⁶ lần đánh giá)
tỉ lệ sẽ cao hơn. Ngoại suy: ~84 phút mỗi lần chạy GA gốc → khoảng 8 phút.

**Chưa kiểm tra ngoài miền huấn luyện.** Tập dữ liệu phủ 3–6 AC và 1–2 link. Kịch bản
7+ AC hay 3+ link về nguyên tắc chạy được (kiến trúc không cố định số node,
`MAX_AC = 8`) nhưng chưa đo.

**Toàn bộ tập dữ liệu được nạp thẳng lên VRAM** (~450 MB với 200 000 mẫu). Trên 8 GB
thì tới ~600 000 mẫu vẫn ổn; vượt nữa cần `--device cpu` hoặc nạp theo batch.
