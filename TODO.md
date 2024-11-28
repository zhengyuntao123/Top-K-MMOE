baseline：MMOE w/wo dropout

experiment：Top-K MMOE，Top-K MMOE+Router-Z-Loss，Top-K MMOE+Router-Z-Loss+OLMO's load balancing loss

待做的实验：

1、Top-K MMOE：调节gate_dropout，通过tensorboard观察dropout对负载均衡的影响

| dropout | AUC  | variance |
| ------- | ---- | -------- |
| 0       |      |          |
| 0.2     |      |          |
| 0.4     |      |          |

2、Top-K MMOE：调节sparse_load_balancing_loss_coef为0和1e-2，研究sparsely-gated中的load_balancing_loss对Top-K MMOE负载均衡的影响

| sparse_load_balancing_loss_coef | AUC  | variance |
| ------------------------------- | ---- | -------- |
| 0                               |      |          |
| 0.01                            |      |          |
| 0.05                            |      |          |

3、Top-K MMOE+OLMO's load balancing loss+Router-Z-loss：调节olmo_load_balancing_loss_coef，研究OLMO的load balancing loss对负载均衡的影响

| olmo_load_balancing_loss_coef | AUC  | variance |
| ----------------------------- | ---- | -------- |
|                               |      |          |
|                               |      |          |
|                               |      |          |

上表最好的用来固定

| router_z_loss_coef | AUC  | variance |
| ------------------ | ---- | -------- |
|                    |      |          |
|                    |      |          |
|                    |      |          |

注：以上实验都要考虑exp1和exp2，合计8个实验





