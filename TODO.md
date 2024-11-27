baseline：MMOE w/wo dropout

experiment：Top-K MMOE，Top-K MMOE+Router-Z-Loss，Top-K MMOE+Router-Z-Loss+OLMO's load balancing loss

待做的实验：

1、MMOE：调节gate_dropout，通过tensorboard观察dropout对负载均衡的影响

2、Top-K MMOE：调节load_balancing_loss_coef为0和1e-2，研究sparsely-gated中的load_balancing_loss对Top-K MMOE负载均衡的影响

3、Top-K MMOE+Router-Z-Loss：调节router_z_loss_coef，研究router_z_loss对训练稳定性的影响

4、Top-K MMOE+Router-Z-Loss+OLMO's load balancing loss：调节load_balancing_loss_coef，研究OLMO的load balancing loss对负载均衡的影响

5、当前的模型表达能力太强了，使得Top-K MMOE和MMOE在性能(AUC)上几乎没有差别，调节Expert中的hidden_layers，减弱模型的表达能力，观察Top-K MMOE是否比MMOE更优秀

注：以上实验都要考虑exp1和exp2，合计10个实验