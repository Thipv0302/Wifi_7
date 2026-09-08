Du lieu do duoc, xuat tu results/*.json bang `python -m export_data`.
Moi gia tri fitness deu do MO HINH GIAI TICH Eqs.(1)-(18) tinh,
khong bao gio do surrogate tinh.

PP1 -- GA / GA + GNN surrogate / GA + GNN + policy seeding
  pp1_runs.csv               210 lan chay: 3 pp x 7 N_pop x 10 seed
  pp1_summary.csv            gop theo (pp, N_pop)
  pp1_anytime.csv            duong fitness theo thoi gian / so lan danh gia
  pp1_headline_npop200.csv   bang chinh o N_pop = 200

Quet nguong / anytime / QoS (ban thao Paper 1)
  pp1_eps_sweep.csv          eps_1 x ngan sach x phuong phap
  pp1_anytime_budget.csv     muc tieu theo ngan sach chung, 2 truc
  pp1_anytime_runs.csv       tung lan chay cua thi nghiem anytime
  pp1_qos_per_ac.csv         QoS tung AC, bon cau hinh
  gnn_eval_training.csv      hoi tu mang danh gia tren thang muc tieu

PP2 -- GNN policy
  pp2_best_of_k.csv          chat luong theo so mau K (policy vs ngau nhien)
  pp2_raw_samples.csv        6000 mau tho da danh gia chinh xac
  pp2_constraint_margin.csv  bien rang buoc tung AC
  pp2_training_curve.csv     16 moc trong 3000 buoc huan luyen

Nguong 'success' = 1% cua gia tri tot nhat quan sat duoc tren toan bo
phep do: fitness >= 49.207331 (gia tri tot nhat = 49.704374).
