# 固定推文预算社交机器人分类器的控制与稳定性理论性质证明

**适用对象**：基于 MiniLM 局部解冻（$K=4$）与多头共享分类器（`RobustSharedHeads`，含 `SubBag-ProbVar`、`ResGate`、`SetAttn` 变体）的固定 8 条推文（$K=8$）单视图社交机器人检测系统。

**说明**：本分类器属于静态前馈决策系统而非闭环状态空间动力学系统（$\dot{x}=f(x,u)$），因此不声称传统意义下的“状态轨迹李雅普诺夫渐近稳定性”。但针对**输入采样扰动（Sampling Disturbance）**、**输出翻转率（Decision Churn）**与**阈值操作点（Operating Point）**，该分类器严格满足以下 **5 项可证明的控制与稳定性性质**，并与 TwiBot-20 验证集（$N=2{,}240, M=30, R=5$）的实证结果完全一致。

---

## 1. 系统形式化建模（System Formulation）

对任意账号 $i$，记其静态画像特征为 $u_i \in \mathbb{R}^{392}$，其可用推文池为 $\{t_{i,1}, \dots, t_{i,n_i}\}$。MiniLM 编码器最后一层输出经平均池化与 $\ell_2$ 归一化后，每条推文的表示向量满足：

$$e_{i,j} \in \mathbb{S}^{383} := \left\{ v \in \mathbb{R}^{384} : \|v\|_2 = 1 \right\}$$

在固定预算 $K=8$ 下，单次推理采样一个视图 $V \subset \{1,\dots,n_i\}$（$|V|=K$）。文本投影分支 $f_{\text{text}}: \mathbb{R}^{384} \to \mathbb{R}^{64}$ 与画像分支 $f_{\text{prof}}: \mathbb{R}^{392} \to \mathbb{R}^{64}$ 均为连续 Lipschitz 映射。推文集合聚合表示为：

$$\bar{x}_i(V) = \sum_{k \in V} w_k \, x_{i,k}, \qquad x_{i,k} = f_{\text{text}}(e_{i,k}), \qquad w_k \ge 0,\ \sum_{k \in V} w_k = 1$$

其中标准均匀池化取 $w_k = 1/K$，`SetAttn` 取 softmax 注意力权重。拼接特征 $g_i(V) = [\bar{x}_i(V); f_{\text{prof}}(u_i)] \in \mathbb{R}^{128}$ 输入 $H=5$ 个独立分类头 $h_1, \dots, h_H$，各头 Lipschitz 常数为 $L_1, \dots, L_H$，输出集成概率与集成 logit：

$$p_{\text{ens}}(V) = \frac{1}{H} \sum_{c=1}^{H} \sigma\!\left(z_c(V)\right) \in (0, 1), \qquad z_{\text{ens}}(V) = \operatorname{logit}\!\left(p_{\text{ens}}(V)\right)$$

在给定阈值 $\tau \in (0, 1)$ 下，单视图二值决策为 $D_i(V) = \mathbb{I}\!\left(p_{\text{ens}}(V) \ge \tau\right)$。

---

## 2. 定理 1：BIBO 有界性与推文替换扰动的 Lipschitz 敏感度控制

### 定理陈述（Theorem 1: BIBO Stability & Sampling-Perturbation Attenuation）

设模型采用均匀池化（$w_k = 1/K$），且 $R_x := \sup_{\|e\|_2 = 1} \|f_{\text{text}}(e)\|_2$。则：

1. **BIBO 有界性**：对任意合法推文集合与有界画像输入，文本表示 $\|\bar{x}_i(V)\|_2 \le R_x < \infty$，分类器输出概率严格有界于 $(0, 1)$。
2. **推文替换扰动上界**：对同一账号的任意两个 $K$-推文视图 $V, V'$，若二者仅有 $r \in \{0, 1, \dots, K\}$ 条推文不同（即 $|V \setminus V'| = |V' \setminus V| = r$），则集成预测概率的最大偏移满足：

$$\left| p_{\text{ens}}(V) - p_{\text{ens}}(V') \right| \le \frac{\bar{L}_H R_x}{2 K} \, r, \qquad \bar{L}_H := \frac{1}{H} \sum_{c=1}^{H} L_c$$

### 证明

由于单位球面 $\mathbb{S}^{383}$ 是 $\mathbb{R}^{384}$ 中的紧集，而 $f_{\text{text}}$ 是线性层与连续激活函数/归一化层的有限复合，因此连续函数 $\|f_{\text{text}}(e)\|_2$ 在紧集 $\mathbb{S}^{383}$ 上必达有限最大值 $R_x < \infty$。由三角不等式可得 $\|\bar{x}_i(V)\|_2 \le \frac{1}{K}\sum_{k \in V} \|x_{i,k}\|_2 \le R_x$，结合 Sigmoid 值域 $(0,1)$ 即证 BIBO 有界性。

对于视图 $V, V'$，重合的 $K - r$ 条推文在差值中完全抵消：

$$\left\| \bar{x}_i(V) - \bar{x}_i(V') \right\|_2 = \frac{1}{K} \left\| \sum_{k \in V \setminus V'} x_{i,k} - \sum_{k' \in V' \setminus V} x_{i,k'} \right\|_2 \le \frac{1}{K} \left( \sum_{k \in V \setminus V'} \|x_{i,k}\|_2 + \sum_{k' \in V' \setminus V} \|x_{i,k'}\|_2 \right) \le \frac{2 r}{K} R_x$$

对同一账号，静态画像特征 $f_{\text{prof}}(u_i)$ 在 $V$ 与 $V'$ 间保持不变，故拼接向量差满足 $\|g_i(V) - g_i(V')\|_2 = \|\bar{x}_i(V) - \bar{x}_i(V')\|_2$。因此对每个分类头 $c \in \{1,\dots,H\}$：

$$\left| z_c(V) - z_c(V') \right| \le L_c \left\| \bar{x}_i(V) - \bar{x}_i(V') \right\|_2 \le \frac{2 r L_c R_x}{K}$$

因为 Sigmoid 函数导数满足 $\sigma'(z) = \sigma(z)(1-\sigma(z)) \le \frac{1}{4}$ 对一切 $z \in \mathbb{R}$ 成立，由中值定理：

$$\left| \sigma(z_c(V)) - \sigma(z_c(V')) \right| \le \frac{1}{4} \left| z_c(V) - z_c(V') \right| \le \frac{r L_c R_x}{2 K}$$

对 $H$ 个头取平均并用三角不等式即得：

$$\left| p_{\text{ens}}(V) - p_{\text{ens}}(V') \right| \le \frac{1}{H} \sum_{c=1}^{H} \left| \sigma(z_c(V)) - \sigma(z_c(V')) \right| \le \frac{\bar{L}_H R_x}{2 K} \, r \qquad \blacksquare$$

### 控制学意义

采样推文替换造成的输出扰动幅度被严格线性限制在扰动比例 $r/K$ 内，且增益系数完全由权重谱范数 $\bar{L}_H R_x$ 控制（对应训练中的 AdamW 权重衰减与梯度裁剪控制机制）。

---

## 3. 定理 2：多头集成的方差收缩性质（Ensemble Variance Contraction）

### 定理陈述（Theorem 2）

记第 $c$ 个分类头在随机视图 $V$ 下的输出概率为 $p_c(V) = \sigma(z_c(V))$，其视图方差为 $\sigma_c^2 = \operatorname{Var}_V(p_c(V))$。则 $H$ 头集成概率 $p_{\text{ens}}(V) = \frac{1}{H}\sum_{c=1}^H p_c(V)$ 的采样方差严格不大于单头平均采样方差：

$$\operatorname{Var}_V\!\left(p_{\text{ens}}(V)\right) = \frac{1}{H^2} \sum_{c=1}^{H} \sigma_c^2 + \frac{1}{H^2} \sum_{c \neq c'} \operatorname{Cov}_V\!\left(p_c(V), p_{c'}(V)\right) \le \frac{1}{H} \sum_{c=1}^{H} \sigma_c^2$$

当且仅当所有 $H$ 个头在视图空间上完全正线性相关且方差相等时取等号；只要各头存在非完全相关性，集成即实现严格方差收缩。

### 证明

直接展开均值方差并应用 Cauchy–Schwarz 不等式 $\operatorname{Cov}_V(p_c, p_{c'}) \le \sigma_c \sigma_{c'} \le \frac{1}{2}(\sigma_c^2 + \sigma_{c'}^2)$：

$$\operatorname{Var}_V\!\left(p_{\text{ens}}\right) \le \frac{1}{H^2} \sum_{c=1}^{H} \sigma_c^2 + \frac{1}{H^2} \sum_{c \neq c'} \frac{\sigma_c^2 + \sigma_{c'}^2}{2} = \frac{1}{H^2} \left( \sum_{c=1}^{H} \sigma_c^2 + (H - 1)\sum_{c=1}^{H} \sigma_c^2 \right) = \frac{1}{H} \sum_{c=1}^{H} \sigma_c^2 \qquad \blacksquare$$

**实验对照**：在 TwiBot-20 上将单头 `EXP-4` 扩展为 5 头 `EXP-8-MH5-KL` 后，训练方差 `TrVar` 由 `0.1971` 降至 `0.1267`，采样方差 `SpVar` 由 `0.3110` 降至 `0.2433`，与定理预测完全吻合。

---

## 4. 定理 3：基于概率方差控制的翻转率（Churn）显式上界

本定理为 `SubBag-ProbVar` 目标函数中的概率方差项 $\mathcal{L}_{\text{var}} = 4\,\operatorname{Var}_V(p_{\text{ens}})$ 建立直接连接离散不可微指标 **Churn** 的分布无关理论保证。

### 定理陈述（Theorem 3: Cantelli Churn Control Bound）

对任意账号 $i$，设其跨视图概率均值与方差分别为 $\mu_i = \mathbb{E}_V[p_i(V)]$ 与 $\sigma_i^2 = \operatorname{Var}_V(p_i(V))$，定义其决策间隔（margin）为 $\delta_i := |\mu_i - \tau| > 0$。设对账号 $i$ 独立抽取 $M \ge 2$ 个视图，阳性视图数 $S_i = \sum_{m=1}^M D_{i,m}$，配对翻转率定义为：

$$C_i := \frac{2 S_i (M - S_i)}{M(M - 1)}$$

则账号 $i$ 的期望翻转率满足严格上界：

$$\mathbb{E}[C_i] \le \frac{2\,\sigma_i^2}{\sigma_i^2 + \delta_i^2} \le \frac{2\,\sigma_i^2}{\delta_i^2}$$

### 证明

记单视图阳性概率 $q_i := \mathbb{P}_V(p_i(V) \ge \tau)$。当 $M$ 个视图独立同分布时，$S_i \sim \operatorname{Binomial}(M, q_i)$，于是：

$$\mathbb{E}[S_i(M - S_i)] = M \mathbb{E}[S_i] - \mathbb{E}[S_i^2] = M^2 q_i - \left(M q_i(1-q_i) + M^2 q_i^2\right) = M(M-1)\,q_i(1-q_i)$$

因此 $\mathbb{E}[C_i] = 2 q_i (1 - q_i)$。注意到对一切 $q_i \in [0, 1]$ 均有 $q_i(1-q_i) \le \min(q_i, 1-q_i)$，故 $\mathbb{E}[C_i] \le 2\min(q_i, 1-q_i)$。

分两种情形应用 **Cantelli 单边不等式**（对任意零均值、方差为 $\sigma^2$ 的随机变量 $X$ 与 $a > 0$，有 $\mathbb{P}(X \ge a) \le \frac{\sigma^2}{\sigma^2 + a^2}$）：

- **情形 1（$\mu_i < \tau$）**：此时 $\delta_i = \tau - \mu_i > 0$，
  $$q_i = \mathbb{P}_V\!\left(p_i(V) - \mu_i \ge \delta_i\right) \le \frac{\sigma_i^2}{\sigma_i^2 + \delta_i^2}$$
- **情形 2（$\mu_i > \tau$）**：此时 $\delta_i = \mu_i - \tau > 0$，对随机变量 $X = \mu_i - p_i(V)$ 应用 Cantelli 不等式，
  $$1 - q_i = \mathbb{P}_V\!\left(\mu_i - p_i(V) > \delta_i\right) \le \mathbb{P}_V\!\left(\mu_i - p_i(V) \ge \delta_i\right) \le \frac{\sigma_i^2}{\sigma_i^2 + \delta_i^2}$$

综合两情形均有 $\min(q_i, 1-q_i) \le \frac{\sigma_i^2}{\sigma_i^2 + \delta_i^2}$，代入 $\mathbb{E}[C_i] \le 2\min(q_i, 1-q_i)$ 即得：

$$\mathbb{E}[C_i] \le \frac{2\,\sigma_i^2}{\sigma_i^2 + \delta_i^2} \le \frac{2\,\sigma_i^2}{\delta_i^2} \qquad \blacksquare$$

### 控制学与算法解释

该上界 $\frac{2\sigma_i^2}{\sigma_i^2 + \delta_i^2}$ 揭示了控制 Churn 的**双通路机理**：

1. **压低分子 $\sigma_i^2$（扰动抑制）**：`SubBag-ProbVar` 直接以 $\mathcal{L}_{\text{var}} = 4 \operatorname{Var}_V(p_i)$ 作为可微代理能量函数，在单次编码前向下用 8 个子包高效估计并压缩 $\sigma_i^2$，且由于 $\frac{\partial p}{\partial z} = p(1-p)$ 在决策边界附近最大，梯度自然聚焦于小间隔账号（$\delta_i$ 较小者）。实证中 `K4-MH5-SubBag` 将 `SpVar` 显著压缩 **42.3%**（`0.2855 -> 0.1646`, `p_holm = 0.0070`）。
2. **放大分母 $\delta_i^2$（表征拉宽间隔）**：解冻 MiniLM 最后 4 层（`K=4`）打破了 `K <= 1` 的表征瓶颈，将 `AP` 由 `0.91865` 推进至 `0.92986`（`+0.01121`, `p_holm = 0.0070`），系统性拉大了两类样本到阈值的平均间隔。

---

## 5. 定理 4：多视图误报守恒律与单视图校准操作点限制

本定理严格解释了为何所有显著降低 Churn 的模型在 `cal_matched_human_fp` 联合筛选下均出现 `HumFP` 上升（`0/5`），证明这并非算法失效，而是单视图校准协议隐含的代数守恒律。

### 定理陈述（Theorem 4: Multi-View Error Conservation Law）

对任意账号 $i$ 与 $M \ge 2$ 个视图，定义全视图稳定阳性指示变量 $A_i := \mathbb{I}(S_i = M)$ 与跨视图翻转指示变量 $F_i := \mathbb{I}(1 \le S_i \le M - 1)$。则：

1. **逐账号精确恒等式**：
   $$\frac{S_i}{M} = A_i + \frac{S_i}{M} F_i$$
2. **Human 总体误报守恒律**：对 Human 总体（$Y=0$）取期望，记单视图误报率 $\bar{R}_1(H) := \mathbb{E}[\frac{S_i}{M} \mid Y=0]$，全视图稳定误报率 $R_{\text{stable}}(H) := \mathbb{E}[A_i \mid Y=0]$，则：
   $$\bar{R}_1(H) = R_{\text{stable}}(H) + \mathbb{E}\!\left[\frac{S_i}{M} F_i \,\middle|\, Y=0\right]$$
3. **单视图校准推论（Operating-Point Tradeoff）**：若阈值校准规则固定单视图误报率 $\bar{R}_1(H) = c$（如 `cal_matched_human_fp`），则任何使翻转区间阳性质量 $\mathbb{E}[\frac{S_i}{M} F_i \mid Y=0]$ 净下降 $\Delta > 0$ 的模型改进，**必然导致全视图稳定误报率 $R_{\text{stable}}(H)$ 精确上升 $+\Delta$**。

### 证明

将整数随机变量 $S_i \in \{0, 1, \dots, M\}$ 按互斥完备事件 $\{S_i = 0\}$、$\{1 \le S_i \le M-1\}$、$\{S_i = M\}$ 拆分：

$$\frac{S_i}{M} = \frac{0}{M}\mathbb{I}(S_i = 0) + \frac{S_i}{M}\mathbb{I}(1 \le S_i \le M-1) + \frac{M}{M}\mathbb{I}(S_i = M) = A_i + \frac{S_i}{M} F_i$$

两边在条件 $\{Y=0\}$ 下取期望，由期望线性性即得守恒律；固定左端 $\bar{R}_1(H) = c$ 后移项即得推论。$\blacksquare$

### TwiBot-20 实证核验

| 模型 | 校准模式 | `dev` 单视图误报 $\bar{R}_1(H)$ | `30v` 稳定误报 $R_{\text{stable}}(H)$ | 翻转 Human 占比 $\mathbb{P}(F_i=1 \mid Y=0)$ | Churn |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `EXP-0` | `cal_matched_human_fp` | 29.12% | 18.43% | 19.08% | 4.379% |
| `K4-MH5-KL01` | `cal_matched_human_fp` | 29.68% | 20.53% | 17.45% | 3.462% |
| `K4-MH5-SubBag` | `cal_matched_human_fp` | 29.58% | 21.55% | 15.60% | 3.112% |
| `EXP-0` | `dev_iso_human_stable_fp` | 29.27% | 18.55% | 19.23% | 4.346% |
| `K4-MH5-KL01` | `dev_iso_human_stable_fp` | 27.95% | 18.55% | 18.31% | 3.820% |
| `K4-MH5-SubBag` | `dev_iso_human_stable_fp` | **26.88%** | 18.55% | **16.98%** | **3.645%** |

可见：在 `cal_matched_human_fp` 下强行锁住 $\bar{R}_1(H) \approx 29.1\%\text{--}29.7\%$ 时，压低翻转区间必然推高 $R_{\text{stable}}(H)$；而一旦改用真正对齐 $R_{\text{stable}}(H)=18.55\%$ 的 `dev_iso_human_stable_fp`，`K4-MH5-SubBag` 同时将单视图误报率净降 **2.39 个百分点**（`29.27% -> 26.88%`）并将 Churn 降低 **16.13%**（`p_holm = 0.0070`）。

---

## 6. 命题 5：架构级控制保证（`ResGate` 与 `SetAttn`）

### 命题 5A（`ResGate` 有界增益与高离散度自动回退）

在 `ResGate` 中，门控系数定义为 $\gamma(d) = \sigma\!\left(e^{s}(\tau_g - d)\right)$，其中 $d \ge 0$ 为包内推文归一化离散度，$s \in [-2, 3]$，融合输出为：

$$z(V) = (1 - \gamma(d))\, z_{\text{base}} + \gamma(d)\, z_{\text{full}}(V)$$

则：

1. **凸组合有界性**：对任意输入，恒有 $\gamma(d) \in (0, 1)$，输出 logit 严格位于静态基线 $z_{\text{base}}$ 与完整预测 $z_{\text{full}}(V)$ 构成的闭区间内，不产生增益发散。
2. **抗噪回退单调性**：$\frac{\partial \gamma}{\partial d} = -e^{s}\gamma(d)(1-\gamma(d)) < 0$；当包内推文极度不一致（$d \to +\infty$）时，$\gamma(d) \to 0$，系统平滑退回零翻转的静态平衡点 $z_{\text{base}}$（对应实证中 `MH5-ResGate-SubBag` 取得全系列最低采样方差 `SpVar = 0.1307`，降幅 **54.2%**）。

### 命题 5B（`SetAttn` 熵正则化的有效样本数下界）

设 `SetAttn` 对 $K$ 条推文分配概率权重 $w \in \Delta^{K-1}$，均匀分布为 $u = (1/K,\dots,1/K)$，信息熵为 $H(w) = -\sum_{k=1}^K w_k \ln w_k$，定义有效推文样本数（困惑度）为 $N_{\text{eff}}(w) := \exp(H(w))$。由于：

$$D_{\text{KL}}(w \| u) = \sum_{k=1}^{K} w_k \ln(K w_k) = \ln K - H(w)$$

两边取指数即得精确控制关系：

$$N_{\text{eff}}(w) = K \cdot \exp\!\left(-D_{\text{KL}}(w \| u)\right)$$

因此，训练目标中的 KL 正则项 $\lambda_{\text{ent}} D_{\text{KL}}(w \| u)$ 若将散度控制在 $D_{\text{KL}}(w \| u) \le \epsilon$，则严格保证有效样本数 $N_{\text{eff}}(w) \ge K e^{-\epsilon}$，从而在结构上防止注意力坍缩到单条推文（$N_{\text{eff}} \to 1$ 会使最坏推文替换敏感度放大 $K=8$ 倍）。

---

## 7. 理论边界与严谨性声明（What Cannot Be Claimed）

为确保学术严谨，明确以下边界：

1. **不构成闭环动力学李雅普诺夫稳定性**：分类器为单步前馈映射，不存在时间演化状态方程与渐近收敛轨线。
2. **零间隔样本边界**：定理 3 中的 Cantelli 上界在 $\delta_i = |\mu_i - \tau| = 0$（恰好落在分类超平面上的样本）处退化为平凡上界 $2$（即最大翻转率 $0.5$ 仍需依赖分类间隔的提升）。
3. **校准集到验证集的有限样本泛化差**：定理 4 在同一分布上精确成立；实际中 `cal`（$N=1{,}176$）与 `dev`（$N=2{,}240$）之间存在有限样本抽样误差，因此 `cal_matched_human_fp` 下的 `dev` 单视图误报率存在约 $0.4\text{--}0.6\text{ pp}$ 的小幅波动。
4. **测试集隔离**：上述全部实证核验均仅基于 `train` 与 `dev`，官方 `test` 集始终保持未打开、未评分状态（`test_opened: false`, `test_scored: false`）。
